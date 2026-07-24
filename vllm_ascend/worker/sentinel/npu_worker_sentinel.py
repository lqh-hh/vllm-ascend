# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
from torch.distributed.distributed_c10d import _set_pg_timeout
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.distributed.utils import create_tcp_store
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.serial_utils import run_method

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_elastic_info
from vllm_ascend.worker.sentinel.scale_down import ScaleDownHelper

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class WorkerSentinel:
    """Holds FT state for a single worker (mask tensors, DP config).

    Methods are called via collective_rpc from EngineCoreSentinel.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        self.worker = worker
        self.device = device
        self.dp_rank = worker.parallel_config.data_parallel_rank
        self.dp_size = worker.parallel_config.data_parallel_size
        self.data_parallel_master_ip = worker.parallel_config.data_parallel_master_ip
        self._fault_coord_store = None
        self.set_dp_gloo_timeout()

    def set_dp_gloo_timeout(self) -> None:
        timeout = timedelta(seconds=self.worker.vllm_config.parallel_config.cpu_distributed_timeout_seconds)
        dp_cpu_group = get_dp_group()
        _set_pg_timeout(timeout=timeout, group=dp_cpu_group.cpu_group)

    def handle_command(self, ft_request: FaultToleranceRequest):
        """Dispatch an FT command by instruction name."""
        with set_current_vllm_config(self.worker.vllm_config):
            return run_method(self, ft_request.instruction, (ft_request,), {})

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """In vllm_ascend, `query_mask` has no practical significance"""
        return {"mask": get_ep_group().world_size * [0]}

    def retry(self, ft_request: FaultToleranceRequest):
        torch.accelerator.synchronize()
        params = ft_request.params
        self._clean_worker_state()
        if self.dp_size > 1:
            old_cpu_group = get_dp_group().cpu_group
            stateless_destroy_torch_distributed_process_group(old_cpu_group)
            port = params["new_stateless_dp_group_port"]
            get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
                self.data_parallel_master_ip,
                port,
                self.dp_rank,
                self.dp_size,
                backend="gloo",
            )

    def scale_down(self, ft_request: FaultToleranceRequest):
        self._clean_worker_state()
        self.scale_down_worker(ft_request)
        self.worker.execute_dummy_batch()
        torch.npu.synchronize()

    def scale_down_worker(self, ft_request: FaultToleranceRequest):
        """
        Reconfigure data-parallel (DP) layout and MoE expert placement after
        excluding one or more DP ranks (e.g., due to failure).

        Args:
            excluded_ep_ranks: EP ranks to exclude from service.
            ft_request: FaultToleranceRequest.
        """
        assert self.worker.vllm_config.parallel_config.enable_fault_tolerance is True, "enable_fault_tolerance is False"
        if not self.worker.model_loaded:
            raise RuntimeError("model has not been loaded yet")

        new_dp_rank = ft_request.params["new_dp_rank"]
        removed_dp_ranks = ft_request.params["removed_dp_ranks"]
        new_dp_size = ft_request.params["new_dp_size"]
        new_stateless_dp_group_port = ft_request.params["new_stateless_dp_group_port"]
        new_stateless_coord_store_port = ft_request.params[
            "new_stateless_coord_store_port"
        ]
        new_master_ip = ft_request.params.get(
            "dp_master_ip", self.data_parallel_master_ip
        )

        enable_d2d_rebalance = get_ascend_config().enable_d2d_rebalance

        if self.worker.model_runner.shared_dict["moe_load"] is None or torch.all(
            self.worker.model_runner.shared_dict["moe_load"][0] == 0
        ):
            enable_d2d_rebalance = False

        scale_down_helper = ScaleDownHelper(self.worker.vllm_config, self.worker.model_runner, self.worker.quant)
        # Currently,only TP=1 is supported.Therefore excluded_dp_ranks = excluded_ep_ranks
        # TODO: In scenarios TP>1,the logic for converting from
        #  excluded_ep_ranks to excluded_dp_ranks needs to be added

        # Phase 1: Expert distribution recalculation
        experts_to_load = scale_down_helper.get_expert_distribution_after_scale_down(
            removed_dp_ranks, enable_d2d_rebalance, new_dp_rank
        )
        num_add_experts_per_rank = self.worker.model_runner.shared_dict["num_add_experts_per_rank"]

        if num_add_experts_per_rank > 0:
            # use_mask_mc2 is False
            raise RuntimeError("only support mask mc2")

        # Phase 2: Expert weight reloading
        saved_weights = scale_down_helper.load_expert_weights_to_cpu(experts_to_load, self.worker.weight_name_to_tensor)
        scale_down_helper.reload_expert_weights(experts_to_load, saved_weights)

        # Phase 3：EPLB adaptor update
        if get_ascend_config().eplb_config.dynamic_eplb:
            scale_down_helper.update_eplb_info(num_add_experts_per_rank, new_dp_rank)

        # Phase 4: Log2phy map generation
        if enable_d2d_rebalance:
            all_layer_log2phy = scale_down_helper.d2d_transmission_for_scaling_down()
        else:
            all_layer_log2phy = scale_down_helper.gen_all_layer_log2phy(new_dp_rank)

        elastic_ep_executor = self.worker.elastic_ep_executor
        if elastic_ep_executor is None:
            raise RuntimeError(
                "Fault scale-down requires the Elastic EP scaling executor"
            )
        if new_dp_rank == 0:
            self._fault_coord_store = create_tcp_store(
                new_master_ip,
                new_stateless_coord_store_port,
                world_size=-1,
                is_master=True,
                wait_for_workers=False,
            )
        elastic_ep_executor.create_fault_scale_down_metadata_groups(
            new_dp_size,
            new_dp_rank,
            new_master_ip,
            new_stateless_coord_store_port,
        )

        self.worker.global_experts_distribution = self.worker.model_runner.eplb_process.worker.local2global(
            self.worker.model_runner.shared_dict["expert_maps"]
        )

        # Phase 5: Configuration and state update
        old_ep_size = len(self.worker.ep2dp_map)
        scale_down_helper.update_parallel_config(new_dp_size, new_dp_rank, new_stateless_dp_group_port)
        self.worker.vllm_config.parallel_config.data_parallel_master_ip = (
            new_master_ip
        )
        self.data_parallel_master_ip = new_master_ip
        self.dp_size = self.worker.vllm_config.parallel_config.data_parallel_size
        self.dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank
        self.worker.model_runner.dp_size = self.dp_size
        self.worker.model_runner.dp_rank = self.dp_rank

        from vllm_ascend.worker.sentinel.scale_down import get_mapping

        rank_mapping = get_mapping(removed_dp_ranks, new_dp_size)
        self.worker.ep2dp_map = scale_down_helper.update_ep2dp_map(
            self.worker.ep2dp_map, removed_dp_ranks, rank_mapping
        )
        elastic_info = get_elastic_info()
        num_new_phy_experts = (self.worker.model_runner.shared_dict["expert_maps"][0] != -1).sum().item()
        scale_down_helper.update_elastic_info(elastic_info, num_new_phy_experts, old_ep_size, self.worker.ep2dp_map)

        # Phase 6: Communication group reinitialization
        elastic_ep_executor.commit_fault_scale_down_metadata_groups()
        active_dp_group = get_dp_group()
        active_dp_size = active_dp_group.world_size
        active_dp_rank = active_dp_group.rank_in_group
        if active_dp_size != new_dp_size or active_dp_rank != new_dp_rank:
            raise RuntimeError(
                "Fault scale-down left inconsistent DP metadata: "
                f"group_rank={active_dp_rank}/{active_dp_size}, "
                f"config_rank={new_dp_rank}/{new_dp_size}"
            )

        # Phase 7: update routing without changing the captured MoE/MC2
        # physical dimensions. This matches active scale-down and keeps the
        # V3 restore scale-up path compatible with the existing ACL graphs.
        elastic_ep_executor.apply_fault_scale_down_log2phy(all_layer_log2phy)
        logger.info(
            "[FT] Worker scale-down metadata committed: dp_rank=%s/%s, "
            "dp_group_rank=%s/%s",
            self.dp_rank,
            self.dp_size,
            active_dp_rank,
            active_dp_size,
        )

    def _clean_worker_state(self):
        import torch_npu

        from vllm_ascend.platform import NPUPlatform

        NPUPlatform.set_device(self.device)
        torch_npu.npu.stop_device(self.device.index)
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()
        self.worker.model_runner.execute_model_state = None
        self.worker.model_runner.kv_connector_output = None
        input_batch = self.worker.model_runner.input_batch
        cached_req_ids = input_batch.req_id_to_index.keys()
        for req_id in list(cached_req_ids):
            input_batch.remove_request(req_id)
        input_batch.condense()
        input_batch.refresh_metadata()
        input_batch.req_prompt_embeds.clear()
        self.worker.model_runner.async_output_copy_stream = torch.cuda.Stream()
        self.worker.model_runner.prepare_inputs_event = torch.Event()
