#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from multiprocessing import Process, Queue
from typing import Any

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_dynamic_eplb_group
from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory


class EplbWorker:
    def __init__(
        self,
        shared_dict,
        policy_type,
        enable_d2d: bool = True,
        tp_size: int | None = None,
    ):
        self.policy_type = policy_type
        self.default_policy_type = policy_type
        self.policy = PolicyFactory.generate_policy(policy_type)
        self.shared_dict = shared_dict
        self.old_expert_maps = None
        self.enable_d2d = enable_d2d
        self.tp_size = tp_size
        dynamic_eplb_group = get_dynamic_eplb_group()
        self.rank_id = dynamic_eplb_group.rank_in_group
        self.rank_id_to_initial_global = list(range(dynamic_eplb_group.world_size))
        n_ranks_per_node = torch.npu.device_count() if hasattr(torch, "npu") else 8
        self.rank_id_to_node_id = [rank_id // n_ranks_per_node for rank_id in range(dynamic_eplb_group.world_size)]
        self.multi_stage = policy_type == 3
        self.enable_dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        self.old_load_info = None

    def do_update(self):
        # put data in to queue
        # in process self.policy.generate_policy()
        # get epxert table && tensor

        # async stream
        # D2D
        # H2D
        # Get initial expert_map
        torch.set_num_threads(1)
        if self.shared_dict.get("reset_old_expert_maps", False):
            self.old_expert_maps = self.get_init_expert_maps()
            if self.old_expert_maps is not None:
                self.num_local_experts = self.old_expert_maps.max() + 1
            else:
                raise ValueError("Failed to reset expert_maps from shared_dict.")
            world_size = self.shared_dict.get("new_ep_size")
            if world_size is None:
                world_size = self.old_expert_maps.shape[1]
            self.rank_id_to_initial_global = list(range(world_size))
            n_ranks_per_node = torch.npu.device_count() if hasattr(torch, "npu") else 8
            self.rank_id_to_node_id = [rank_id // n_ranks_per_node for rank_id in range(world_size)]
            self.shared_dict["reset_old_expert_maps"] = False
            logger.info(
                "[EPLB] reset cached old expert maps: rank=%s, world_size=%s, shape=%s",
                self.rank_id,
                world_size,
                tuple(self.old_expert_maps.shape),
            )
        if self.old_expert_maps is None:
            self.old_expert_maps = self.get_init_expert_maps()
            if self.old_expert_maps is not None:
                self.num_local_experts = self.old_expert_maps.max() + 1
            else:
                raise ValueError("Failed to get expert_maps from shared_dict.")
        if self.shared_dict["scale_down"] and self.enable_dynamic_eplb:
            self.old_expert_maps = self.get_init_expert_maps()

        # Get MOE load information
        load_info = self.fetch_and_sum_load_info()
        scale_down = self.shared_dict.get("scale_down", False)
        if load_info is None and not scale_down:
            logger.debug("[eplb/worker] No moe_load data available yet, skipping this cycle")
            return

        scale = self.shared_dict.get("scale", False)
        if scale:
            old_ep_size = self.shared_dict["old_ep_size"]
            new_ep_size = self.shared_dict["new_ep_size"]
            assert old_ep_size != new_ep_size
            self.policy.set_new_ep_size(new_ep_size)
            if new_ep_size > old_ep_size:
                self._restore_rank_id_for_scale_up(old_ep_size, new_ep_size)
            if load_info is not None and load_info.shape[1] > old_ep_size:
                load_info = load_info[:, :old_ep_size]
            if self.old_expert_maps.shape[1] > old_ep_size:
                self.old_expert_maps = self.old_expert_maps[:, :old_ep_size]

        # Get the updated expert table based on the workload information
        old_placement = self.global2local(self.old_expert_maps, self.num_local_experts)
        num_add_experts_per_rank = 0
        if scale_down:
            exclude_dp_ranks = self.shared_dict["excluded_dp_ranks"]
            enable_d2d_after_failure = self.shared_dict["enable_d2d_after_failure"]
            update_layer_id = self.shared_dict.get("update_layer_id", -1)
            self.update_rank_id(exclude_dp_ranks)
            effective_load_info = load_info if load_info is not None else self.old_load_info
            new_placement, old_deployment, need_load_h2d, num_add_experts_per_rank = self.trigger_fault_redeployment(
                effective_load_info,
                old_placement,
                exclude_dp_ranks,
                enable_d2d_after_failure,
                update_layer_id,
            )
            if not torch.is_tensor(old_deployment):
                old_placement = torch.tensor(old_deployment)
            self.old_expert_maps = self.local2global(old_placement)
            self.shared_dict["need_load_h2d"] = need_load_h2d
            self.shared_dict["num_add_experts_per_rank"] = num_add_experts_per_rank
        else:
            _, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)
            if load_info is not None:
                self.old_load_info = load_info

        if self.rank_id == 0 and load_info is not None and not scale_down:
            if self.multi_stage:
                hotness = self._calculate_hotness(old_placement, load_info.sum(0))
            else:
                hotness = self._calculate_hotness(old_placement, load_info)
            # ms-service-metric begin: expose EPLB hotness details for metrics collection.
            current_mean, current_max, current_imbalance_list = self._compute_imbalance(
                old_placement, hotness, return_list=True
            )
            update_mean, update_max, update_imbalance_list = self._compute_imbalance(
                new_placement, hotness, return_list=True
            )
            self.latest_expert_hotness = {
                "current_mean": current_mean,
                "current_max": current_max,
                "update_mean": update_mean,
                "update_max": update_max,
                "current_imbalance_list": current_imbalance_list,
                "update_imbalance_list": update_imbalance_list,
            }
            # ms-service-metric end.
            logger.info(
                "[Expert Hotness] Current: mean=%.3f, max=%.3f, Updated: mean=%.3f, max=%.3f",
                current_mean,
                current_max,
                update_mean,
                update_max,
            )

        if not torch.is_tensor(new_placement):
            new_placement = torch.tensor(new_placement)
        if not scale and not scale_down:
            self.check_expert_placement(old_placement, new_placement)
        new_expert_maps = self.local2global(new_placement)
        new_expert_maps_clone = new_expert_maps.clone()

        if scale_down and self.shared_dict["enable_d2d_after_failure"]:
            self.update_expert_map(self.old_expert_maps.clone())
        else:
            self.update_expert_map(new_expert_maps)

        if scale:
            shape = list(new_expert_maps.shape)
            shape[1] = abs(old_ep_size - new_ep_size)
            if old_ep_size > new_ep_size:
                shutdown_rank_expert_maps = torch.full(shape, -1, dtype=new_expert_maps.dtype)
                new_expert_maps = torch.cat([new_expert_maps, shutdown_rank_expert_maps], dim=1)
            else:
                new_rank_expert_maps = torch.full(shape, -1, dtype=new_expert_maps.dtype)
                self.old_expert_maps = torch.cat([self.old_expert_maps, new_rank_expert_maps], dim=1)

        update_info = self.compose_expert_update_info_greedy(new_expert_maps, self.old_expert_maps)
        self.old_expert_maps = new_expert_maps_clone
        logger.debug("[eplb/worker] EPLB Process compute complete")

        if scale_down and not self.shared_dict["enable_d2d_after_failure"]:
            packed_update_info = []
        else:
            packed_update_info = self.pack_update_info(update_info)

        if num_add_experts_per_rank > 0:
            self.rank_id_to_initial_global = list(range(len(self.rank_id_to_initial_global)))
        if scale:
            self.shared_dict["scale"] = False
            self.shared_dict["old_ep_size"] = None
            self.shared_dict["new_ep_size"] = None
        self.shared_dict["scale_down"] = False
        if scale_down and self.policy_type != self.default_policy_type:
            self.policy_type = self.default_policy_type
            self.policy = PolicyFactory.generate_policy(self.default_policy_type)

        return packed_update_info

    def check_expert_placement(self, old_placement, new_placement):
        num_layers = old_placement.shape[0]
        num_ranks = old_placement.shape[1]

        for layer_id in range(num_layers):
            # check if any logical expert is not placed on any rank
            if torch.unique(new_placement[layer_id]).numel() < torch.unique(old_placement[layer_id]).numel():
                logger.error("[eplb/worker] There exists expert not placed on any rank in layer %s", layer_id)
                new_placement[layer_id] = old_placement[layer_id]
                continue

            for rank_id in range(num_ranks):
                new_placement_check = new_placement[layer_id][rank_id]
                old_placement_check = old_placement[layer_id][rank_id]

                # check if same logical experts are placed on the same NPU
                if new_placement_check.numel() != torch.unique(new_placement_check).numel():
                    logger.error(
                        "[eplb/worker] Replicated experts are placed on the same NPU; "
                        "expert placement on layer %s, rank %s is invalid",
                        layer_id,
                        rank_id,
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

                # check if there is any experts movement inside one NPU
                expert_not_move = torch.isin(new_placement_check, old_placement_check)
                if not torch.equal(new_placement_check[expert_not_move], old_placement_check[expert_not_move]):
                    logger.error(
                        "[eplb/worker] Expert movement inside NPU detected; "
                        "expert placement on layer %s, rank %s is invalid",
                        layer_id,
                        rank_id,
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

    # TODO: Here only expert weight exchange is considered, need to be extended to cover other weight update cases
    def compose_expert_update_info_greedy(self, updated_expert_maps, current_expert_maps):
        num_layers = current_expert_maps.shape[0]
        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]

            expert_send_info_this_layer: dict[Any, Any] = {}
            expert_recv_info_this_layer: dict[Any, Any] = {}

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if torch.equal(updated_expert_maps_this_layer, current_expert_maps_this_layer):
                yield (
                    expert_send_info_this_layer,
                    expert_recv_info_this_layer,
                    updated_expert_maps_this_layer,
                    layer_id,
                )
                continue

            # Parse expert_ids each rank needs to receive from other ranks
            dst_rank_indices, experts_to_recv = torch.where(
                (current_expert_maps_this_layer == -1) & (updated_expert_maps_this_layer != -1)
            )

            # Parse expert_ids each rank needs to send to other ranks
            src_rank_indices, experts_to_send = torch.where(
                (current_expert_maps_this_layer != -1) & (updated_expert_maps_this_layer == -1)
            )

            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_recv[idx].item()
                if dst_rank_id not in expert_recv_info_this_layer:
                    expert_recv_info_this_layer[dst_rank_id] = []

                if not torch.isin(torch.tensor(expert_id), experts_to_send).any():
                    # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
                    candidate_src_rank_indices = torch.where(current_expert_maps_this_layer[:, expert_id] != -1)[0]
                else:
                    candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]

                # TODO: improve selection criterion of NPU sending expert_id,
                # considering intra-node or inter-node...
                src_rank_id = candidate_src_rank_indices[0].item()
                if src_rank_id not in expert_send_info_this_layer:
                    expert_send_info_this_layer[src_rank_id] = []

                dst_global_rank_id = self.rank_id_to_initial_global[dst_rank_id]
                src_global_rank_id = self.rank_id_to_initial_global[src_rank_id]
                expert_send_info_this_layer[src_rank_id].append((dst_global_rank_id, expert_id))
                expert_recv_info_this_layer[dst_rank_id].append((src_global_rank_id, expert_id))

            yield (
                expert_send_info_this_layer,
                expert_recv_info_this_layer,
                updated_expert_maps_this_layer,
                layer_id,
            )

    def calculate_rebalance_experts(self, load_info, old_placement):
        """
        Compute `new_map` by calling the `rebalance_experts` method of the policy instance.
        """
        if self.old_expert_maps is None:
            return False, None, None

        changed, priority, new_map = self.policy.rebalance_experts(old_placement, load_info)
        return changed, priority, new_map

    def trigger_fault_redeployment(
        self,
        load_info,
        old_placement,
        exclude_dp_ranks,
        enable_d2d_after_failure,
        update_layer_id=-1,
    ):
        if self.policy_type != 4:
            self.policy = PolicyFactory.generate_policy(4)
            self.policy_type = 4

        self.policy.failed_cards = exclude_dp_ranks
        self.policy.rank_id_to_node_id = self.rank_id_to_node_id
        self.policy.enable_d2d_after_failure = enable_d2d_after_failure
        self.policy.update_layer_id = update_layer_id
        new_deployment, old_deployment, need_load_h2d, num_add_experts_per_rank = self.policy.rebalance_experts(
            old_placement, load_info
        )
        return new_deployment, old_deployment, need_load_h2d, num_add_experts_per_rank

    def update_rank_id(self, exclude_dp_ranks):
        unique_fault_ids = sorted(set(exclude_dp_ranks))
        fault_count = 0
        for failed_rank in unique_fault_ids:
            if failed_rank <= self.rank_id:
                fault_count += 1
            else:
                break
        self.rank_id -= fault_count
        for failed_rank in reversed(unique_fault_ids):
            self.rank_id_to_initial_global.pop(failed_rank)
            self.rank_id_to_node_id.pop(failed_rank)

    def _restore_rank_id_for_scale_up(self, old_ep_size: int, new_ep_size: int):
        if len(self.rank_id_to_initial_global) >= new_ep_size:
            return

        n_ranks_per_node = torch.npu.device_count() if hasattr(torch, "npu") else 8
        for rank_id in range(old_ep_size, new_ep_size):
            if rank_id in self.rank_id_to_initial_global:
                continue
            self.rank_id_to_initial_global.append(rank_id)
            self.rank_id_to_node_id.append(rank_id // n_ranks_per_node)

    def get_init_expert_maps(self):
        """
        Read the initial expert_map from shared_dict.
        """
        return self.shared_dict.get("expert_maps", None)

    def fetch_and_sum_load_info(self):
        """
        Each time the subprocess is awakened, read the latest moe_load
        (shape: [num_moe_layers, num_experts_per_layer]) from shared_dict.
        """
        return self.shared_dict.get("moe_load", None)

    def update_expert_map(self, expert_maps):
        self.shared_dict["expert_maps"] = expert_maps

    def global2local(self, placement: torch.Tensor, E_local: int) -> tuple[torch.Tensor, torch.Tensor]:
        L, G, _ = placement.shape
        device = placement.device

        pt_local = torch.full((L, G, E_local), fill_value=-1, dtype=torch.long, device=device)

        valid = placement >= 0
        l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

        slot_idx = placement[l_idx, g_idx, k_idx]

        pt_local[l_idx, g_idx, slot_idx] = k_idx

        return pt_local

    def local2global(self, placement_local: torch.Tensor) -> torch.Tensor:
        L, G, E_local = placement_local.shape
        device = placement_local.device

        max_id = torch.max(placement_local)
        E_global = (max_id + 1).item() if max_id >= 0 else 0

        if E_global == 0:
            return torch.empty((L, G, 0), dtype=torch.long, device=device)

        placement_global = torch.full((L, G, E_global), fill_value=-1, dtype=torch.long, device=device)

        valid = placement_local >= 0
        l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
        gid_idx = placement_local[l_idx, g_idx, slot_idx]

        placement_global[l_idx, g_idx, gid_idx] = slot_idx

        return placement_global

    def pack_update_info(self, update_info_generator):
        """
        Pack a list of update info tuples for efficient IPC.
        """
        send_all = []
        recv_all = []
        maps = []
        log2phy_all = []
        layer_ids = []

        for send_info, recv_info, new_expert_map, layer_id in update_info_generator:
            send_info_this_rank = send_info.get(self.rank_id, [])
            recv_info_this_rank = recv_info.get(self.rank_id, [])
            send_all.append(send_info_this_rank)
            recv_all.append(recv_info_this_rank)

            maps.append(new_expert_map.numpy().tolist())

            log2phy_map = generate_log2phy_map(
                new_expert_map,
                self.rank_id,
                tp_size=self.tp_size,
            )
            log2phy_all.append(log2phy_map.numpy().tolist())

            layer_ids.append(layer_id)

        return list(zip(send_all, recv_all, maps, log2phy_all, layer_ids))

    def get_original_workload(self, load_info) -> np.ndarray:
        n_layer, n_rank, n_experts_per_card = load_info.shape
        workload_new = np.zeros((n_layer, self.num_local_experts))

        for layer_idx in range(n_layer):
            for card_idx in range(n_rank):
                for index in range(n_experts_per_card):
                    cur_expert = self.old_expert_maps[layer_idx][card_idx][index]
                    cur_load = load_info[layer_idx][card_idx][index]
                    workload_new[layer_idx][cur_expert] += cur_load

        return workload_new

    def warm_up_shared_dict(self):
        old_expert_maps = self.get_init_expert_maps()
        if old_expert_maps is not None:
            _ = old_expert_maps.max()

    @staticmethod
    def _compute_imbalance(deployment_all_layer, hotness_all_layer: np.ndarray, return_list: bool = False):
        imbalance_list = []
        deployment_all_layer = np.array(deployment_all_layer)
        for deployment, hotness in zip(deployment_all_layer, hotness_all_layer):
            counts = np.bincount(deployment.reshape(-1), minlength=hotness.shape[0])

            unit_hotness = np.divide(hotness, counts, out=np.zeros_like(hotness, dtype=float), where=counts != 0)

            stage_load = unit_hotness[deployment].sum(-1)
            stage_mean = stage_load.mean()
            stage_par = stage_load.max() / stage_mean if stage_mean != 0 else 1.0
            imbalance_list.append(stage_par)

        max_val = max(imbalance_list)
        mean_val = sum(imbalance_list) / len(imbalance_list)
        # ms-service-metric begin: optionally expose per-layer imbalance without recomputing it.
        if return_list:
            return mean_val, max_val, imbalance_list
        # ms-service-metric end.
        return mean_val, max_val

    @staticmethod
    def _calculate_hotness(deployment_all_layer, moe_load_all_layer):
        hotnesses = []
        num_of_expert = deployment_all_layer.shape[1] * deployment_all_layer.shape[2]
        for deployment, rank_load in zip(deployment_all_layer, moe_load_all_layer.numpy()):
            hotness = np.zeros(num_of_expert, dtype=rank_load.dtype)
            deployment_flat = deployment.ravel()
            rank_load_flat = rank_load.ravel()
            np.add.at(hotness, deployment_flat, rank_load_flat)
            hotnesses.append(hotness)

        return np.array(hotnesses)


class EplbProcess:
    def __init__(
        self,
        shared_dict,
        policy_type: int = 0,
        enable_d2d: bool = True,
        tp_size: int | None = None,
    ):
        """
        Args:
            shared_dict: Cross-process shared dict returned by Manager().dict()
            policy_type: Integer passed to PolicyFactory.generate_policy
            enable_d2d: Whether to enable D2D loading
        """
        self.shared_dict = shared_dict
        self.policy_type = policy_type
        self.enable_d2d = enable_d2d
        self.planner_q: Queue[Any] = Queue()
        self.block_update_q: Queue[Any] = Queue(maxsize=1)

        # Create EplbWorker instance
        self.worker = EplbWorker(
            self.shared_dict,
            self.policy_type,
            self.enable_d2d,
            tp_size=tp_size,
        )

    def worker_process(self, planner_q, block_update_q):
        """
        Subprocess entry: bind to specified NPU, loop waiting for planner_q to wake up,
        call do_update, then notify main process update is complete.
        """
        try:
            from ms_service_metric.adapters.vllm.adapter import get_vllm_adapter, initialize_vllm_metric  # type: ignore

            initialize_vllm_metric()
            adapter = get_vllm_adapter()
            logger.info("[EPLB metrics] The adapter initialized: %s", adapter.is_initialized())
        except Exception as e:
            logger.warning("[EPLB metrics] Failed to initialize metrics: %s", e)

        if self.policy_type == 3:
            from vllm_ascend.eplb.core.policy.policy_flashlb import warm_up

            warm_up()
        self.worker.warm_up_shared_dict()
        while True:
            try:
                planner_q.get()

                packed_update_info = self.worker.do_update()

                while True:
                    if not block_update_q.empty():
                        continue
                    if self.shared_dict["scale_down"]:
                        break
                    block_update_q.put(packed_update_info)
                    break

            except Exception as e:
                logger.warning(
                    "[eplb/worker] Subprocess crashed, EPLB optimization will stop. error=%s",
                    e,
                    exc_info=True,
                )
                break

    def clear_block_update_q(self):
        while not self.block_update_q.empty():
            try:
                self.block_update_q.get_nowait()
            except Exception as e:
                logger.error("[EPLB subprocess exiting due to error: %s]", e)
                break

    def _launch_process(self):
        """
        Use spawn method to launch subprocess and return (planner_q, block_update_q, proc).
        """
        proc = Process(target=self.worker_process, args=(self.planner_q, self.block_update_q), daemon=True)

        proc.start()
        return proc
