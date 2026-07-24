# NOTE:
# This file is adapted from vLLM's elastic_execute.py
#
# Key differences:
# 1. Device-specific adaptations: Replaces CUDA-specific operations with NPU (Ascend) equivalents
#    - Uses `torch_npu` instead of CUDA APIs
#    - Replaces `torch.accelerator.synchronize()` with `torch.npu.synchronize()`
#    - Replaces `torch.accelerator.empty_cache()` with `torch.npu.empty_cache()`
#    - Uses `ACLGraphWrapper` instead of `CUDAGraphWrapper` for graph management
#
# 2. Custom weight transfer implementation: Implements `ascend_batch_transfer_weights()`
#    - Adds support for quantized weight names (aclnn_input_scale, aclnn_input_scale_reciprocal, aclnn_input_offset)
#    - Uses threading lock (`_PATCH_LOCK`) for thread-safe weight transfer patching
#
# 3. Enhanced broadcast_expert_mapping: Simplified signature and implementation
#    - Removed `physical_to_logical`, `num_local_physical_experts`, `num_logical_experts` parameters
#    - Uses `expert_maps` tensor directly for broadcasting
#
# 4. Extended AscendElasticEPScalingExecutor class:
#    - Adds `_use_ascend_transfer_impl()` context manager for patching weight transfer
#    - Implements `_release_acl_graphs()` to clear ACL graphs instead of CUDA graphs
#    - Adds `_replace_ascend_active_groups()` calls for Ascend-specific group management
#    - Integrates with `create_ascend_standby_groups()` and `pop_ascend_standby_groups()`
#    - Adds support for Ascend-specific MoE modules (AscendFusedMoE, AscendSharedFusedMoE)
#    - Handles Ascend-specific quantization method (AscendW8A8DynamicFusedMoEMethod)
#    - Integrates with `get_mc2_group()` and `get_dynamic_eplb_group()` for Ascend communication
#    - Adds `setup_moe_comm_method()` calls for MoE communication setup
#
# 5. EPLB (Expert Parallel Load Balancing) adaptations:
#    - Uses `eplb_loader`, `eplb_adaptor`, `eplb_updator` from model_runner
#    - Implements `_perform_eplb_reshuffle()` with expert resharding logic
#    - Handles dynamic EPLB configuration via `get_ascend_config().eplb_config`
#
# ============================================================

import copy
import gc
import os
import threading
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn as nn
import torch_npu
import vllm.distributed.parallel_state as vllm_parallel_state
from torch.distributed import P2POp
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import (
    CompilationMode,
    set_current_vllm_config,
)
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tp_group,
)
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
from vllm.distributed.elastic_ep.standby_state import (
    create_standby_groups,
    get_standby_dp_group,
    get_standby_ep_group,
    pop_standby_groups,
)
from vllm.distributed.parallel_state import _replace_active_groups
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEParallelConfig
from vllm.v1.attention.backend import AttentionImplBase
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    reset_graph_params,
    set_draft_graph_params,
    set_graph_params,
)
from vllm_ascend.distributed.elastic_ep.standby_state import (
    create_ascend_standby_groups,
    get_standby_mc2_group,
    pop_ascend_standby_groups,
)
from vllm_ascend.distributed.elastic_ep.v3_capture_dp_sync import (
    clear_old_active_dp_sync_group,
    configure_force_v3_during_scale_up,
    configure_new_rank_capture_dp_sync,
    configure_old_active_dp_sync_group,
    finish_new_rank_capture_dp_sync,
    is_capture_dp_sync_companion_done,
    is_old_active_dp_sync_enabled,
    raise_capture_dp_sync_companion_error_if_any,
    start_capture_dp_sync_companion_background,
)
from vllm_ascend.distributed.parallel_state import (
    _replace_ascend_active_groups,
    _replace_dynamic_eplb_group,
    get_dynamic_eplb_group,
    get_elastic_info,
    get_mc2_group,
    set_elastic_info,
)
from vllm_ascend.distributed.utils import (
    stateless_destroy_pg_with_world_cleanup,
    use_stateless_pg_with_world_registration,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    get_moe_comm_method,
    setup_moe_comm_method,
    setup_moe_mc2_comm_method,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import update_moe_distribute_v3_contexts
from vllm_ascend.quantization.methods import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.worker.sentinel.scale_down import (
    ScaleDownHelper,
    init_ep2dp_map,
    update_ep2dp_map,
)

from .eplb_manager import ElasticEplbManager, generate_expert_maps_file

_PATCH_LOCK = threading.Lock()


def ascend_batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
) -> None:
    device_comm = dp_group.device_communicator
    tcp_store_group = dp_group.tcp_store_group
    if device_comm is None:
        raise ValueError("No device communicator found")

    expert_weights_set = set()
    for weight_group in expert_weights:
        for weight in weight_group:
            expert_weights_set.add(weight.data_ptr())

    state_dict = model.state_dict()
    all_params = []
    all_params_ptrs = set()
    all_params_name = []

    for name, param in state_dict.items():
        if name.endswith("expert_map"):
            continue
        ptr = param.data_ptr()
        if ptr not in expert_weights_set and ptr not in all_params_ptrs:
            if param.device.type == "npu":
                all_params.append(param.data)
                all_params_ptrs.add(ptr)
                all_params_name.append(name)

    def handle_sub_module(submodule, submodule_name):
        for attr_name, attr_value in submodule.__dict__.items():
            if attr_name.endswith("expert_map"):
                continue
            if isinstance(attr_value, torch.Tensor):
                data_ptr = attr_value.data_ptr()
                if data_ptr not in expert_weights_set and data_ptr not in all_params_ptrs:
                    if attr_value.device.type == "npu":
                        all_params.append(attr_value)
                        all_params_ptrs.add(data_ptr)
                        all_params_name.append(submodule_name + "." + attr_name)
            if isinstance(attr_value, AttentionImplBase):
                handle_sub_module(attr_value, submodule_name + "." + attr_name)

    for module_name, module in model.named_modules():
        handle_sub_module(module, module_name)

    expert_map_params = []
    if not get_ascend_config().eplb_config.dynamic_eplb:
        for module_name, module in model.named_modules():
            if (expert_map := getattr(module, "expert_map", None)) is not None:
                if isinstance(expert_map, torch.Tensor):
                    expert_map_params.append((expert_map.npu(), expert_map))
                    all_params_name.append(module_name + "." + "expert_map")
        all_params.extend([npu_tensor for npu_tensor, _ in expert_map_params])

    if is_sender:
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
    else:
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)

    if len(all_params_name) != len(peer_rank_all_params_name):
        common = list(set(all_params_name) & set(peer_rank_all_params_name))
        ids = [all_params_name.index(name) for name in common]
        all_params = [param for idx, param in enumerate(all_params) if idx in ids]

    assert len(all_params) > 0
    p2p_ops = []
    for param in all_params:
        op = object.__new__(P2POp)
        if is_sender:
            op.op = torch.distributed.isend
            op.tensor = param
        else:
            op.op = torch.distributed.irecv
            op.tensor = param
        op.group_peer = peer_rank
        p2p_ops.append(op)

    device_comm.batch_isend_irecv(p2p_ops)

    if expert_map_params:
        for npu_tensor, cpu_tensor in expert_map_params:
            cpu_tensor.copy_(npu_tensor)


def broadcast_expert_mapping(
    expert_maps: torch.Tensor | None,
    group: StatelessGroupCoordinator,
    src_rank: int = 0,
    old_active_ep_size: int | None = None,
    return_old_active_ep_size: bool = False,
):
    if group.rank_in_group == src_rank:
        assert expert_maps is not None
        shape_tensor = torch.tensor(list(expert_maps.shape), dtype=torch.int64, device="cpu")
        old_active_ep_size_tensor = torch.tensor(
            [old_active_ep_size if old_active_ep_size is not None else expert_maps.shape[1]],
            dtype=torch.int64,
            device="cpu",
        )
    else:
        shape_tensor = torch.empty(3, dtype=torch.int64, device="cpu")
        old_active_ep_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")

    shape_tensor = group.tcp_store_group.broadcast(shape_tensor, src_rank)
    old_active_ep_size_tensor = group.tcp_store_group.broadcast(old_active_ep_size_tensor, src_rank)

    if group.rank_in_group != src_rank:
        expert_maps = torch.empty(
            tuple(shape_tensor.tolist()),
            dtype=torch.int64,
            device="cpu",
        )

    assert expert_maps is not None
    expert_maps = group.tcp_store_group.broadcast(expert_maps, src_rank)

    if return_old_active_ep_size:
        return expert_maps, int(old_active_ep_size_tensor.item())
    return expert_maps


def setup_moe_comm_and_quant_method(module: nn.Module) -> None:
    if isinstance(
        quant_method := getattr(module.quant_method, "quant_method", None),
        (AscendW8A8DynamicFusedMoEMethod, AscendW4A8DynamicFusedMoEMethod),
    ):
        quant_method.ep_group = get_ep_group()
        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = get_mc2_group().rank_in_group
            backend = device_group._get_backend(torch.device("npu"))
            quant_method.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            quant_method.moe_all_to_all_group_name = ""
    setup_moe_comm_method(module.moe_config)


def setup_moe_mc2_comm_and_quant_method(module: nn.Module, setup_comm_method: bool = True) -> None:
    if isinstance(
        quant_method := getattr(module.quant_method, "quant_method", None),
        (AscendW8A8DynamicFusedMoEMethod, AscendW4A8DynamicFusedMoEMethod),
    ):
        quant_method.ep_group = get_ep_group()
        try:
            device_group = get_mc2_group().device_group
            local_rank = get_mc2_group().rank_in_group
            backend = device_group._get_backend(torch.device("npu"))
            quant_method.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            quant_method.moe_all_to_all_group_name = ""
    if setup_comm_method:
        setup_moe_mc2_comm_method(module.moe_config)


class AscendElasticEPScalingExecutor(ElasticEPScalingExecutor):
    def __init__(self, worker):
        super().__init__(worker)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        if not self.dynamic_eplb and os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0") != "1":
            get_ascend_config().eplb_config.expert_map_path = generate_expert_maps_file()
        self.eplb_manager = None
        self.old_ep_size = None
        self._skip_next_eplb_workspace_rewarm = False
        self._old_active_dp_cpu_group = None

    @contextmanager
    def _timed_switch_stage(self, action: str):
        dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank
        full_action = f"worker_switch_and_prepare:{action}"
        start = time.perf_counter()
        print(
            "[EEP_PAUSE_TIMING] event=BEGIN "
            f"action={full_action} worker_type=worker dp_rank={dp_rank} "
            "state=SWITCH_AND_PREPARE "
            f"wall_time={time.time():.6f}",
            flush=True,
        )
        result = "ok"
        try:
            yield
        except BaseException:
            result = "error"
            raise
        finally:
            print(
                "[EEP_PAUSE_TIMING] event=END "
                f"action={full_action} worker_type=worker dp_rank={dp_rank} "
                "state=SWITCH_AND_PREPARE "
                f"result={result} "
                f"elapsed_ms={(time.perf_counter() - start) * 1000:.3f} "
                f"wall_time={time.time():.6f}",
                flush=True,
            )

    def init_eplb_manager(self):
        self.eplb_manager = ElasticEplbManager(self.worker)

    @contextmanager
    def _use_ascend_transfer_impl(self):
        with patch(
            "vllm.distributed.elastic_ep.elastic_execute.batch_transfer_weights", new=ascend_batch_transfer_weights
        ):
            yield

    def _prepare_new_rank_capture_elastic_info(
        self,
        old_active_ep_size: int,
        ep_size: int,
        num_local_experts: int,
    ) -> None:
        if not envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3:
            return
        if old_active_ep_size >= ep_size:
            return

        ep_rank = get_ep_group().rank_in_group
        if ep_rank < old_active_ep_size:
            return

        self._v3_new_rank_capture_elastic_info = True
        valid_ep_ranks = list(range(old_active_ep_size, ep_size))
        active_phy_experts = len(valid_ep_ranks) * num_local_experts
        table1 = torch.full((ep_size,), -1, dtype=torch.int32, device="npu")
        table2 = torch.full((ep_size,), -1, dtype=torch.int32, device="npu")
        for local_ep_rank, valid_ep_rank in enumerate(valid_ep_ranks):
            table1[valid_ep_rank] = local_ep_rank
            table2[local_ep_rank] = valid_ep_rank

        base_config = torch.tensor(
            [1, len(valid_ep_ranks), 0, active_phy_experts],
            dtype=torch.int32,
            device="npu",
        )
        elastic_info = torch.cat([base_config, table1, table2], dim=0).contiguous()
        elastic_info.requires_grad_(False)
        set_elastic_info(elastic_info)
        logger.info(
            "[Elastic EP scale-up] prepared new-rank V3 capture elastic_info: "
            "ep_rank=%s, old_active_ep_size=%s, ep_size=%s, elastic_info=%s",
            ep_rank,
            old_active_ep_size,
            ep_size,
            elastic_info.detach().cpu().tolist(),
        )

    def load_model(self) -> None:
        (
            expert_maps,
            num_local_experts,
            num_logical_experts,
            old_active_ep_size,
        ) = self.receive_expert_mapping()
        dp_size = self.worker.parallel_config.data_parallel_size
        tp_size = self.worker.parallel_config.tensor_parallel_size
        pcp_size = self.worker.parallel_config.prefill_context_parallel_size
        ep_size = dp_size * tp_size * pcp_size
        get_ascend_config().eplb_config.num_redundant_experts = ep_size * num_local_experts - num_logical_experts
        if self.dynamic_eplb:
            self.worker.model_runner.shared_dict["expert_maps"] = expert_maps
            self.worker.model_runner.shared_dict["old_ep_size"] = expert_maps.shape[1]
            self._mark_dynamic_eplb_scale_up(ep_size)
        else:
            with set_current_vllm_config(self.worker.vllm_config):
                get_ascend_config().eplb_config.expert_map_path = generate_expert_maps_file()
        self.old_ep_size = old_active_ep_size
        self._prepare_new_rank_capture_elastic_info(old_active_ep_size, ep_size, num_local_experts)
        self.worker.load_model(load_dummy_weights=True)
        self.eplb_manager.expert_maps = expert_maps

    def create_standby_groups(self, reconfig_request: ReconfigureDistributedRequest) -> None:
        self.reconfig_request = reconfig_request
        self._skip_next_eplb_workspace_rewarm = False
        new_dp_size = reconfig_request.new_data_parallel_size
        scale_down = new_dp_size < self.worker.vllm_config.parallel_config.data_parallel_size
        configure_force_v3_during_scale_up(envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3 and not scale_down)
        world_size = self.worker.vllm_config.parallel_config.world_size
        new_world_size_across_dp = world_size * new_dp_size
        updated_config = copy.copy(self.worker.vllm_config)
        updated_config.parallel_config = copy.deepcopy(self.worker.vllm_config.parallel_config)
        updated_config.parallel_config.data_parallel_size = new_dp_size
        with set_current_vllm_config(updated_config), use_stateless_pg_with_world_registration():
            create_standby_groups(
                new_dp_size=new_dp_size,
                new_world_size_across_dp=new_world_size_across_dp,
                master_ip=reconfig_request.new_data_parallel_master_ip,
                coord_store_port=reconfig_request.coord_store_port,
                enable_eplb=updated_config.parallel_config.enable_eplb,
            )
            if not scale_down:
                create_ascend_standby_groups(
                    new_dp_size=new_dp_size,
                    new_world_size_across_dp=new_world_size_across_dp,
                    master_ip=reconfig_request.new_data_parallel_master_ip,
                    coord_store_port=reconfig_request.coord_store_port,
                )

    def create_fault_scale_down_metadata_groups(
        self,
        new_dp_size: int,
        new_dp_rank: int,
        master_ip: str,
        coord_store_port: int,
    ) -> None:
        """Create the metadata groups used after Sentinel scale-down.

        Fault recovery cannot run the active scale-down state machine because
        the failed rank is gone. Build the same surviving-rank standby groups
        here so the commit phase still produces the canonical active-scale
        metadata state.
        """
        parallel_config = self.worker.vllm_config.parallel_config
        if parallel_config.tensor_parallel_size != 1:
            raise RuntimeError(
                "Fault scale-down metadata replacement currently supports TP=1"
            )
        world_size = parallel_config.world_size
        updated_config = copy.copy(self.worker.vllm_config)
        updated_config.parallel_config = copy.deepcopy(parallel_config)
        updated_config.parallel_config.data_parallel_size = new_dp_size
        updated_config.parallel_config.data_parallel_rank = new_dp_rank

        # _init_stateless_group derives the process global rank from the
        # current world coordinator. Fault scale-down densifies surviving DP
        # ranks, so expose the new rank while constructing the standby groups.
        old_world_group = vllm_parallel_state.get_world_group()
        old_world_rank = old_world_group.rank
        try:
            old_world_group.rank = new_dp_rank
            with set_current_vllm_config(
                updated_config
            ), use_stateless_pg_with_world_registration():
                create_standby_groups(
                    new_dp_size=new_dp_size,
                    new_world_size_across_dp=world_size * new_dp_size,
                    master_ip=master_ip,
                    coord_store_port=coord_store_port,
                    enable_eplb=updated_config.parallel_config.enable_eplb,
                )
        finally:
            old_world_group.rank = old_world_rank

    def commit_fault_scale_down_metadata_groups(self) -> None:
        """Install Sentinel scale-down groups using the active-shrink rules."""
        with use_stateless_pg_with_world_registration():
            self._replace_scale_down_metadata_groups()

        dynamic_eplb_group = get_dynamic_eplb_group()
        model_runner = self.worker.model_runner
        model_runner.eplb_updator.comm_group = dynamic_eplb_group
        model_runner.eplb_updator.world_size = dynamic_eplb_group.world_size
        model_runner.eplb_updator.cur_iterations = 0
        model_runner.eplb_loader.comm_group = dynamic_eplb_group

    def transfer_weights(self, old_dp_size: int, new_dp_size: int) -> None:
        model = self.worker.model_runner.get_model()
        if self.dynamic_eplb:
            model.expert_weights = [item[1] for item in self.worker.model_runner.eplb_adaptor.param_dict.items()]
        else:
            model.expert_weights = []
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().transfer_weights(old_dp_size=old_dp_size, new_dp_size=new_dp_size)

    def broadcast_expert_mapping(self):
        standby_dp_group = get_standby_dp_group()
        assert standby_dp_group is not None
        expert_maps = (
            self.worker.model_runner.shared_dict["expert_maps"]
            if self.dynamic_eplb
            else self.eplb_manager.get_expert_maps()
        )
        parallel_config = self.worker.vllm_config.parallel_config
        old_active_ep_size = (
            parallel_config.data_parallel_size
            * parallel_config.tensor_parallel_size
            * parallel_config.prefill_context_parallel_size
        )
        broadcast_expert_mapping(
            expert_maps=expert_maps,
            group=standby_dp_group,
            src_rank=0,
            old_active_ep_size=old_active_ep_size,
        )

    def materialize_new_communication_groups(self) -> None:
        standby_mc2_group = get_standby_mc2_group()
        mc2_group = standby_mc2_group or get_mc2_group()
        torch.distributed.barrier(mc2_group.cpu_group)
        device_group = mc2_group.device_group
        backend = device_group._get_backend(torch.device("npu"))
        comm_name = backend.get_hccl_comm_name(mc2_group.rank_in_group)
        torch.npu.synchronize()
        torch.distributed.barrier(mc2_group.cpu_group)
        logger.info(
            "[Elastic EP scale-up] materialized MC2 HCCL communicator: rank=%s/%s, device_group=%s, comm_name=%s",
            mc2_group.rank_in_group,
            mc2_group.world_size,
            device_group,
            comm_name,
        )

    def start_new_rank_capture_dp_companion(self) -> None:
        start_capture_dp_sync_companion_background()

    def is_new_rank_capture_dp_companion_done(self) -> bool:
        raise_capture_dp_sync_companion_error_if_any()
        return is_capture_dp_sync_companion_done()

    def _release_acl_graphs(self) -> None:
        if isinstance(self.worker.model_runner.model, UBatchWrapper):
            raise RuntimeError("DBO is not yet supported in elastic EP")

        ACLGraphWrapper.clear_all_graphs()

        torch.compiler.reset()
        with set_current_vllm_config(self.worker.vllm_config):
            reset_compile_wrapper(self.worker.model_runner.get_model())

        reset_graph_params()

        capture_descs = self.worker.model_runner.cudagraph_dispatcher.get_capture_descs()
        capture_sizes = sorted({desc.num_tokens for _, descs in capture_descs for desc in descs})
        if self.worker.model_runner.use_aclgraph:
            set_graph_params(capture_sizes)
            if self.worker.model_runner.speculative_config:
                set_draft_graph_params(capture_sizes)

        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def _release_cuda_graphs(self) -> None:
        self._release_acl_graphs()

    def rewarm_workspace(self) -> None:
        # A V3 restore scale-up keeps the captured EP size and reuses the
        # existing MoeDistributeBuffer. Existing ranks only update its HCCL
        # context, while new ranks allocate the buffer before their own graph
        # capture. In that case the post-EPLB rewarm would only repeat graph
        # capture without changing a captured workspace address.
        #
        # Make the decision collectively because the max-token dummy run in
        # the fallback path contains DP collectives. A single-rank mismatch
        # would otherwise deadlock the whole DP group.
        local_skip = self._skip_next_eplb_workspace_rewarm
        self._skip_next_eplb_workspace_rewarm = False
        skip_rewarm = torch.tensor(
            [1 if local_skip else 0],
            dtype=torch.int32,
            device="cpu",
        )
        dp_group = get_dp_group()
        torch.distributed.all_reduce(
            skip_rewarm,
            op=torch.distributed.ReduceOp.MIN,
            group=dp_group.cpu_group,
        )
        if bool(skip_rewarm.item()):
            if dp_group.rank_in_group == 0:
                logger.info(
                    "[Elastic EP scale-up] skipped post-EPLB workspace rewarm "
                    "for V3 restore fast path"
                )
            return

        super().rewarm_workspace()

    def switch_and_remove(self) -> None:
        self._release_cuda_graphs()
        with use_stateless_pg_with_world_registration():
            _replace_active_groups(world=None, dp=None, ep=None, eplb=None, node_count=None)
            _replace_ascend_active_groups(mc2=None, dynamic_eplb=None, fc3_quant_x=None)

    def _replace_scale_down_metadata_groups(self) -> None:
        groups = pop_standby_groups()
        standby_ep = groups.pop("ep", None)
        if standby_ep is not None:
            standby_ep.destroy()
        old_ep = get_ep_group()
        old_mc2 = get_mc2_group()
        old_dynamic_eplb = get_dynamic_eplb_group()
        old_dynamic_eplb_device_group = old_dynamic_eplb.device_group
        vllm_parallel_state._WORLD = groups["world"]
        vllm_parallel_state._DP = groups["dp"]
        vllm_parallel_state._EP = old_ep
        vllm_parallel_state._NODE_COUNT = groups["node_count"]
        new_eplb = groups["eplb"]
        if new_eplb is not None:
            old_dynamic_eplb.ranks = new_eplb.ranks
            old_dynamic_eplb.world_size = new_eplb.world_size
            old_dynamic_eplb.rank_in_group = new_eplb.rank_in_group
            old_dynamic_eplb.cpu_group = new_eplb.cpu_group
            if hasattr(new_eplb, "tcp_store_group"):
                old_dynamic_eplb.tcp_store_group = new_eplb.tcp_store_group
            _replace_dynamic_eplb_group(old_dynamic_eplb)
            vllm_parallel_state._EPLB = old_dynamic_eplb
        else:
            vllm_parallel_state._EPLB = new_eplb
        # Keep the old MC2 and dynamic EPLB device groups for captured MoE
        # graphs and EPLB D2D transfer. Only CPU-side metadata groups shrink.
        assert get_mc2_group() is old_mc2
        assert get_dynamic_eplb_group().device_group is old_dynamic_eplb_device_group

    def _switch_and_prepare_scale_down(self, reconfig_request: ReconfigureDistributedRequest) -> None:
        parallel_config = self.worker.vllm_config.parallel_config
        new_dp_size = reconfig_request.new_data_parallel_size
        old_dp_size = parallel_config.data_parallel_size
        if parallel_config.data_parallel_rank >= new_dp_size:
            return

        with use_stateless_pg_with_world_registration():
            self._replace_scale_down_metadata_groups()

        parallel_config.data_parallel_size = new_dp_size
        if reconfig_request.new_data_parallel_rank != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if reconfig_request.new_data_parallel_rank_local != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank_local = reconfig_request.new_data_parallel_rank_local
        parallel_config.data_parallel_master_ip = reconfig_request.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = reconfig_request.new_data_parallel_master_port

        self.worker.model_runner.dp_size = new_dp_size
        self.worker.model_runner.dp_rank = parallel_config.data_parallel_rank
        self.worker.model_runner.eplb_updator.comm_group = get_dynamic_eplb_group()
        self.worker.model_runner.eplb_updator.world_size = get_dynamic_eplb_group().world_size
        self.worker.model_runner.eplb_updator.cur_iterations = 0
        self.worker.model_runner.eplb_loader.comm_group = get_dynamic_eplb_group()
        reconfigure_args = getattr(self, "_scale_down_moe_reconfigure", None)
        if reconfigure_args is not None:
            _, _, all_layer_log2phy = reconfigure_args
            self._apply_scale_down_log2phy(all_layer_log2phy)
            self._scale_down_moe_reconfigure = None
        logger.info(
            "[Elastic EP scale-down] keep old EP/MC2 groups and old captured graphs: old_dp_size=%s, new_dp_size=%s",
            old_dp_size,
            new_dp_size,
        )

    def _apply_scale_down_log2phy(
        self, all_layer_log2phy: list[torch.Tensor]
    ) -> None:
        """Update routing while preserving captured MoE buffer dimensions."""
        moe_modules = [
            module
            for module in self.worker.model_runner.get_model().modules()
            if isinstance(module, FusedMoE)
        ]
        draft_model = getattr(
            getattr(self.worker.model_runner, "drafter", None), "model", None
        )
        if draft_model is not None:
            moe_modules.extend(
                module
                for module in draft_model.modules()
                if isinstance(module, FusedMoE)
            )
        if len(all_layer_log2phy) < len(moe_modules):
            raise RuntimeError(
                "Insufficient log2phy maps for scale-down: "
                f"maps={len(all_layer_log2phy)}, modules={len(moe_modules)}"
            )
        for layer_id, module in enumerate(moe_modules):
            module.log2phy.copy_(
                all_layer_log2phy[layer_id].npu(), non_blocking=True
            )

    def apply_fault_scale_down_log2phy(
        self, all_layer_log2phy: list[torch.Tensor]
    ) -> None:
        """Apply the graph-preserving MoE update used by active scale-down."""
        self._apply_scale_down_log2phy(all_layer_log2phy)

    def _local_v3_restore_scale_up_fast_path_candidate(
        self,
        old_ep_size: int,
        new_ep_size: int,
        new_dp_size: int,
    ) -> bool:
        parallel_config = self.worker.vllm_config.parallel_config
        elastic_info = get_elastic_info()
        if not envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3:
            return False
        if elastic_info is None:
            return False
        if new_dp_size <= parallel_config.data_parallel_size:
            return False
        if old_ep_size != new_ep_size:
            return False
        if elastic_info.numel() != 4 + 2 * old_ep_size:
            return False
        try:
            return bool(elastic_info[0].item()) and int(elastic_info[1].item()) < old_ep_size
        except RuntimeError:
            elastic_info_cpu = elastic_info.detach().cpu()
            return bool(elastic_info_cpu[0].item()) and int(elastic_info_cpu[1].item()) < old_ep_size

    def _use_v3_restore_scale_up_fast_path(
        self,
        old_ep_size: int,
        new_ep_size: int,
        new_dp_size: int,
    ) -> bool:
        standby_dp_group = get_standby_dp_group()
        if standby_dp_group is None:
            return False
        local_candidate = self._local_v3_restore_scale_up_fast_path_candidate(
            old_ep_size,
            new_ep_size,
            new_dp_size,
        )
        decision = torch.tensor(
            [1 if local_candidate else 0],
            dtype=torch.int64,
            device="cpu",
        )
        decision = standby_dp_group.tcp_store_group.broadcast(decision, 0)
        return bool(decision.item())

    @staticmethod
    def _get_v3_num_local_experts(moe_modules: list[FusedMoE]) -> int:
        num_local_experts = int(moe_modules[0].local_num_experts)
        if num_local_experts <= 0:
            raise RuntimeError(
                "V3 elastic restore requires a positive local expert count, "
                f"got {num_local_experts}."
            )
        if any(
            int(module.local_num_experts) != num_local_experts
            for module in moe_modules[1:]
        ):
            raise RuntimeError(
                "V3 elastic restore requires all MoE modules to have the same "
                "number of local physical expert slots."
            )
        return num_local_experts

    def _set_identity_elastic_info_for_current_mc2(self) -> None:
        elastic_info = get_elastic_info()
        device = elastic_info.device if elastic_info is not None else torch.device("npu")
        ep_size = get_mc2_group().world_size
        moe_modules = [
            module for module in self.worker.model_runner.get_model().modules() if isinstance(module, FusedMoE)
        ]
        if not moe_modules:
            return
        num_local_experts = self._get_v3_num_local_experts(moe_modules)
        moe_expert_num = ep_size * num_local_experts
        base_config = torch.tensor(
            [0, ep_size, 0, moe_expert_num],
            dtype=torch.int32,
            device=device,
        )
        table = torch.arange(ep_size, dtype=torch.int32, device=device)
        identity_elastic_info = torch.cat([base_config, table, table], dim=0).contiguous()
        identity_elastic_info.requires_grad_(False)
        set_elastic_info(identity_elastic_info)

    def _set_old_active_elastic_info_for_current_mc2(
        self,
        old_active_ep_size: int,
    ) -> None:
        elastic_info = get_elastic_info()
        device = elastic_info.device if elastic_info is not None else torch.device("npu")
        ep_size = get_mc2_group().world_size
        moe_modules = [
            module for module in self.worker.model_runner.get_model().modules() if isinstance(module, FusedMoE)
        ]
        if not moe_modules:
            return
        if not 0 < old_active_ep_size <= ep_size:
            raise RuntimeError(
                "Invalid old active EP size for V3 elastic restore: "
                f"old_active_ep_size={old_active_ep_size}, ep_size={ep_size}."
            )
        num_local_experts = self._get_v3_num_local_experts(moe_modules)
        active_phy_experts = old_active_ep_size * num_local_experts
        table1 = torch.full((ep_size,), -1, dtype=torch.int32, device=device)
        table2 = torch.full((ep_size,), -1, dtype=torch.int32, device=device)
        for ep_rank in range(old_active_ep_size):
            table1[ep_rank] = ep_rank
            table2[ep_rank] = ep_rank
        base_config = torch.tensor(
            [1, old_active_ep_size, 0, active_phy_experts],
            dtype=torch.int32,
            device=device,
        )
        old_active_elastic_info = torch.cat([base_config, table1, table2], dim=0).contiguous()
        old_active_elastic_info.requires_grad_(False)
        set_elastic_info(old_active_elastic_info)

    def _mark_dynamic_eplb_scale_up(self, new_ep_size: int) -> None:
        if not get_ascend_config().eplb_config.dynamic_eplb:
            return
        shared_dict = self.worker.model_runner.shared_dict
        expert_maps = shared_dict.get("expert_maps")
        if expert_maps is None:
            return
        old_active_ep_size = expert_maps.shape[1]
        if old_active_ep_size == new_ep_size:
            return
        shared_dict["scale"] = True
        shared_dict["old_ep_size"] = old_active_ep_size
        shared_dict["new_ep_size"] = new_ep_size
        shared_dict["moe_load"] = None
        logger.info(
            "[Elastic EP scale-up] marked dynamic EPLB scale state: old_active_ep_size=%s, new_ep_size=%s",
            old_active_ep_size,
            new_ep_size,
        )

    def _is_v3_elastic_restore_active(self) -> bool:
        if not envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3:
            return False
        elastic_info = get_elastic_info()
        if elastic_info is None:
            return False
        try:
            return bool(elastic_info[0].item())
        except RuntimeError:
            return bool(elastic_info.detach().cpu()[0].item())

    def _switch_and_prepare_v3_restore_scale_up(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        parallel_config = self.worker.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        new_dp_size = reconfig_request.new_data_parallel_size
        old_dp_group = get_dp_group()
        if not isinstance(old_dp_group, StatelessGroupCoordinator):
            raise RuntimeError(
                "V3 restore scale-up requires a stateless old DP group"
            )
        old_active_dp_cpu_group = old_dp_group.cpu_group
        old_elastic_info = get_elastic_info()
        old_active_ep_size = old_dp_size
        with self._timed_switch_stage("read_old_elastic_info"):
            if old_elastic_info is not None:
                old_elastic_info_cpu = old_elastic_info.detach().cpu()
                if bool(old_elastic_info_cpu[0].item()):
                    old_active_ep_size = int(old_elastic_info_cpu[1].item())

        parallel_config.data_parallel_size = new_dp_size
        # Capture-time inference on existing ranks still synchronizes metadata
        # through the old DP CPU group. Detach it before replacing active
        # coordinators so _replace_active_groups() does not destroy it.
        old_dp_group.cpu_group = None
        try:
            with use_stateless_pg_with_world_registration():
                with self._timed_switch_stage("replace_base_groups"):
                    _replace_active_groups(**pop_standby_groups())
                with self._timed_switch_stage("replace_ascend_groups"):
                    _replace_ascend_active_groups(**pop_ascend_standby_groups())
        except BaseException:
            old_dp_group.cpu_group = old_active_dp_cpu_group
            raise
        self._old_active_dp_cpu_group = old_active_dp_cpu_group

        with self._timed_switch_stage("update_parallel_config"):
            if (
                reconfig_request.new_data_parallel_rank
                != ReconfigureRankType.KEEP_CURRENT_RANK
            ):
                parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
            if (
                reconfig_request.new_data_parallel_rank_local
                != ReconfigureRankType.KEEP_CURRENT_RANK
            ):
                parallel_config.data_parallel_rank_local = (
                    reconfig_request.new_data_parallel_rank_local
                )
            parallel_config.data_parallel_master_ip = (
                reconfig_request.new_data_parallel_master_ip
            )
            parallel_config.data_parallel_master_port = (
                reconfig_request.new_data_parallel_master_port
            )
            self.worker.model_runner.dp_size = new_dp_size
            self.worker.model_runner.dp_rank = parallel_config.data_parallel_rank

        with self._timed_switch_stage("update_eplb_refs"):
            dynamic_eplb_group = get_dynamic_eplb_group()
            self.worker.model_runner.eplb_updator.comm_group = dynamic_eplb_group
            self.worker.model_runner.eplb_updator.world_size = (
                dynamic_eplb_group.world_size
            )
            self.worker.model_runner.eplb_updator.cur_iterations = 0
            self.worker.model_runner.eplb_loader.comm_group = dynamic_eplb_group
            self._mark_dynamic_eplb_scale_up(dynamic_eplb_group.world_size)

        with self._timed_switch_stage("update_moe_module_refs"):
            moe_modules = [
                module
                for module in self.worker.model_runner.get_model().modules()
                if isinstance(module, FusedMoE)
            ]
            draft_model = getattr(
                getattr(self.worker.model_runner, "drafter", None), "model", None
            )
            if draft_model is not None:
                moe_modules.extend(
                    module
                    for module in draft_model.modules()
                    if isinstance(module, FusedMoE)
                )
            for module in moe_modules:
                module.moe_config.ep_group = get_ep_group()
                module.moe_config.mc2_group = get_mc2_group()

        with self._timed_switch_stage("configure_old_dp_sync"):
            configure_old_active_dp_sync_group(
                old_active_dp_cpu_group,
                old_dp_group.rank_in_group,
                old_dp_group.world_size,
            )
        with self._timed_switch_stage("set_elastic_info"):
            self._set_old_active_elastic_info_for_current_mc2(old_active_ep_size)
        with self._timed_switch_stage("update_v3_ctx"):
            updated_ctx_count = update_moe_distribute_v3_contexts()
        with self._timed_switch_stage("npu_synchronize"):
            torch.npu.synchronize()
        with self._timed_switch_stage("log_v3_restore_result"):
            logger.info(
                "[Elastic EP scale-up] restored V3 MoE contexts without graph recapture: "
                "old_dp_size=%s, new_dp_size=%s, ep_size=%s, "
                "updated_v3_buffers=%s, elastic_info=%s",
                old_dp_size,
                new_dp_size,
                get_mc2_group().world_size,
                updated_ctx_count,
                get_elastic_info().detach().cpu().tolist()
                if get_elastic_info() is not None
                else None,
            )

    def activate_v3_scale_up_after_capture(self) -> None:
        if not envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3:
            configure_force_v3_during_scale_up(False)
            return
        if not is_old_active_dp_sync_enabled():
            configure_force_v3_during_scale_up(False)
            return
        self._set_identity_elastic_info_for_current_mc2()
        updated_ctx_count = update_moe_distribute_v3_contexts()
        clear_old_active_dp_sync_group()
        configure_force_v3_during_scale_up(False)
        torch.npu.synchronize()
        old_active_dp_cpu_group = self._old_active_dp_cpu_group
        if old_active_dp_cpu_group is not None:
            stateless_destroy_pg_with_world_cleanup(old_active_dp_cpu_group)
            self._old_active_dp_cpu_group = None
        self._skip_next_eplb_workspace_rewarm = updated_ctx_count > 0
        logger.info(
            "[Elastic EP scale-up] activated new rank after V3 capture: "
            "ep_size=%s, updated_v3_buffers=%s, elastic_info=%s",
            get_mc2_group().world_size,
            updated_ctx_count,
            get_elastic_info().detach().cpu().tolist() if get_elastic_info() is not None else None,
        )

    def switch_and_prepare(self) -> None:
        old_ep_size = get_ep_group().world_size
        self.old_ep_size = old_ep_size
        self.worker.model_runner.shared_dict["old_ep_size"] = old_ep_size

        parallel_config = self.worker.vllm_config.parallel_config
        reconfig_request = self.reconfig_request
        assert reconfig_request is not None
        new_dp_size = reconfig_request.new_data_parallel_size
        if new_dp_size < parallel_config.data_parallel_size:
            configure_force_v3_during_scale_up(False)
            self._switch_and_prepare_scale_down(reconfig_request)
            return

        new_ep_size = get_standby_ep_group().world_size
        self.new_ep_size = new_ep_size
        if self._use_v3_restore_scale_up_fast_path(old_ep_size, new_ep_size, new_dp_size):
            self._switch_and_prepare_v3_restore_scale_up(reconfig_request)
            return

        configure_force_v3_during_scale_up(False)
        self._release_acl_graphs()

        parallel_config.data_parallel_size = new_dp_size

        with use_stateless_pg_with_world_registration():
            _replace_active_groups(**pop_standby_groups())
            _replace_ascend_active_groups(**pop_ascend_standby_groups())

        if reconfig_request.new_data_parallel_rank != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if reconfig_request.new_data_parallel_rank_local != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank_local = reconfig_request.new_data_parallel_rank_local
        parallel_config.data_parallel_master_ip = reconfig_request.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = reconfig_request.new_data_parallel_master_port
        self.worker.model_runner.dp_size = new_dp_size
        self.eplb_manager.reset_eplb_updator()
        self.eplb_manager.set_new_comm_group()

        # Reconfigure MoE modules with new EP size
        moe_modules = [module for module in self.worker.model_runner.model.modules() if isinstance(module, FusedMoE)]
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        assert all(module.moe_config.num_local_experts == num_local_experts for module in moe_modules), (
            "All MoE modules must have the same number of experts"
        )
        for module in moe_modules:
            num_logical_experts = self.eplb_manager.get_expert_maps().shape[-1]
            module.global_redundant_expert_num = module.local_num_experts * new_ep_size - num_logical_experts
            module.moe_config.num_experts = num_local_experts * new_ep_size
            module.global_num_experts = module.moe_config.num_experts
            tp_size = get_tp_group().world_size
            is_sequence_parallel = parallel_config.use_sequence_parallel_moe
            sp_size = tp_size if is_sequence_parallel else 1
            module.moe_parallel_config = FusedMoEParallelConfig.make(
                tp_size_=tp_size,
                pcp_size_=get_pcp_group().world_size,
                dp_size_=get_dp_group().world_size,
                sp_size_=sp_size,
                vllm_parallel_config=parallel_config,
            )
            module.moe_config.moe_parallel_config = module.moe_parallel_config

            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()

            with set_current_vllm_config(self.worker.vllm_config):
                setup_moe_comm_and_quant_method(module)

        if self.worker.vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
            # NOTE(yongji): when using stock torch.compile,
            # torch.compile is triggered during GPUModelRunner's load_model()
            # TODO(yongji):check do we need to re-trigger torch.compile here?
            # any changes to the tensor shapes in execution should already
            # be handled internally by torch.compile.
            backend = self.worker.vllm_config.compilation_config.init_backend(self.worker.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.worker.model_runner.model.compile(fullgraph=True, backend=backend)

        multi_block_table = self.worker.model_runner.input_batch.block_table
        saved_block_tables: list[tuple[torch.Tensor, torch.Tensor]] = []
        for bt in multi_block_table.block_tables:
            saved_block_tables.append((bt.block_table.gpu.clone(), bt.block_table.cpu.clone()))
        multi_block_table.clear()

        unlock_workspace()
        try:
            self.worker.compile_or_warm_up_model()
        finally:
            lock_workspace()
            finish_new_rank_capture_dp_sync()

        for bt, (saved_gpu, saved_cpu) in zip(multi_block_table.block_tables, saved_block_tables):
            bt.block_table.gpu.copy_(saved_gpu)
            bt.block_table.cpu.copy_(saved_cpu)

    def _sync_scale_up_eplb_baseline(self) -> None:
        if not get_ascend_config().eplb_config.dynamic_eplb:
            return
        shared_dict = self.worker.model_runner.shared_dict
        if not shared_dict.get("scale", False):
            return

        group = get_dynamic_eplb_group()
        expert_maps = shared_dict.get("expert_maps")
        if group.rank_in_group == 0 and expert_maps is None:
            expert_maps = self.worker.model_runner.eplb_adaptor.get_global_expert_map()

        old_ep_size = shared_dict.get("old_ep_size")
        if group.rank_in_group == 0 and old_ep_size is None and expert_maps is not None:
            old_ep_size = expert_maps.shape[1]

        expert_maps, old_ep_size = broadcast_expert_mapping(
            expert_maps=expert_maps if group.rank_in_group == 0 else None,
            group=group,
            src_rank=0,
            old_active_ep_size=old_ep_size,
            return_old_active_ep_size=True,
        )
        shared_dict["expert_maps"] = expert_maps
        shared_dict["old_ep_size"] = old_ep_size
        shared_dict["new_ep_size"] = group.world_size
        shared_dict["reset_old_expert_maps"] = True
        shared_dict["moe_load"] = None
        logger.info(
            "[Elastic EP scale-up] synchronized EPLB baseline: "
            "rank=%s/%s, old_ep_size=%s, new_ep_size=%s, expert_maps_shape=%s",
            group.rank_in_group,
            group.world_size,
            old_ep_size,
            group.world_size,
            tuple(expert_maps.shape),
        )

    def perform_eplb_reshuffle(self) -> None:
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding...")
        new_ep_size = get_ep_group().world_size
        self._sync_scale_up_eplb_baseline()
        old_ep_size = self.old_ep_size
        shared_dict = None
        if self.dynamic_eplb:
            shared_dict = self.worker.model_runner.shared_dict
            old_ep_size = shared_dict.get("old_ep_size")
        if old_ep_size is None:
            raise RuntimeError("Missing old expert parallel size for EPLB reshuffle.")

        self.old_ep_size = old_ep_size
        self.eplb_manager.eplb(old_ep_size, new_ep_size)

        if shared_dict is not None:
            shared_dict["scale"] = False
            shared_dict["old_ep_size"] = None
            shared_dict["new_ep_size"] = None

    def perform_scale_down_eplb_reshuffle(self, new_dp_size: int) -> None:
        old_ep_size = get_ep_group().world_size
        parallel_config = self.worker.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        old_dp_rank = parallel_config.data_parallel_rank
        if old_dp_rank >= new_dp_size:
            logger.info(
                "[Elastic EP scale-down] skip EPLB shrink reshuffle on removed rank: rank=%s, new_dp_size=%s",
                old_dp_rank,
                new_dp_size,
            )
            return

        tp_size = parallel_config.tensor_parallel_size
        if tp_size != 1:
            raise RuntimeError("External active scale-down with old EP/MC2 groups currently only supports TP=1.")
        excluded_dp_ranks = list(range(new_dp_size, old_dp_size))
        rank_mapping = {rank: rank for rank in range(new_dp_size)}
        model_runner = self.worker.model_runner
        if model_runner.shared_dict["expert_maps"] is None:
            model_runner.shared_dict["expert_maps"] = model_runner.eplb_adaptor.get_global_expert_map()

        quant = self.worker.model_config.quantization is not None
        scale_down_helper = ScaleDownHelper(self.worker.vllm_config, model_runner, quant)
        experts_to_load = scale_down_helper.get_expert_distribution_after_scale_down(
            excluded_dp_ranks,
            False,
            old_dp_rank,
        )
        num_add_experts_per_rank = model_runner.shared_dict["num_add_experts_per_rank"]
        if num_add_experts_per_rank > 0:
            raise RuntimeError("External active scale-down only supports existing expert slots for now.")

        saved_weights = scale_down_helper.load_expert_weights_to_cpu(
            experts_to_load,
            self.worker.weight_name_to_tensor,
        )
        scale_down_helper.reload_expert_weights(experts_to_load, saved_weights)

        if get_ascend_config().eplb_config.dynamic_eplb:
            scale_down_helper.update_eplb_info(num_add_experts_per_rank, old_dp_rank)
        all_layer_log2phy = scale_down_helper.gen_all_layer_log2phy(old_dp_rank)

        ep2dp_map = getattr(self.worker, "ep2dp_map", None)
        if ep2dp_map is None:
            ep2dp_map = init_ep2dp_map(old_dp_size, tp_size)
        self.worker.ep2dp_map = update_ep2dp_map(ep2dp_map, excluded_dp_ranks, rank_mapping)

        num_logical_expert = model_runner.shared_dict["expert_maps"].shape[-1]
        num_new_phy_experts = (model_runner.shared_dict["expert_maps"][0] != -1).sum().item()
        scale_down_helper.update_elastic_info(
            get_elastic_info(),
            num_new_phy_experts,
            old_ep_size,
            self.worker.ep2dp_map,
        )
        self._scale_down_moe_reconfigure = (
            num_logical_expert,
            num_new_phy_experts,
            all_layer_log2phy,
        )

        model_runner.shared_dict["old_ep_size"] = old_ep_size
        model_runner.shared_dict["new_ep_size"] = new_dp_size * tp_size
        model_runner.eplb_adaptor.clear_all_moe_loads()
        model_runner.shared_dict["moe_load"] = None
        model_runner.eplb_updator.cur_iterations = 0
        torch_npu.npu.synchronize()
        logger.info(
            "[Elastic EP scale-down] EPLB shrink reshuffle finished: excluded_dp_ranks=%s, elastic_info=%s",
            excluded_dp_ranks,
            get_elastic_info().detach().cpu().tolist() if get_elastic_info() is not None else None,
        )

    def receive_weights(self) -> None:
        model = self.worker.model_runner.get_model()
        if self.dynamic_eplb:
            model.expert_weights = [item[1] for item in self.worker.model_runner.eplb_adaptor.param_dict.items()]
        else:
            model.expert_weights = []
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().receive_weights()

    def receive_expert_mapping(self) -> tuple[torch.Tensor, int, int, int]:
        dp_group = get_dp_group()
        assert isinstance(dp_group, StatelessGroupCoordinator)
        expert_maps, old_active_ep_size = broadcast_expert_mapping(
            expert_maps=None,
            group=dp_group,
            src_rank=0,
            return_old_active_ep_size=True,
        )
        num_local_experts = (expert_maps[0, 0] != -1).sum().item()
        num_logical_experts = expert_maps.shape[-1]

        return expert_maps, num_local_experts, num_logical_experts, old_active_ep_size

    def _prepare_new_rank_v3_buffer(self, module: FusedMoE) -> bool:
        if not getattr(self, "_v3_new_rank_capture_elastic_info", False):
            return False

        if not self._is_v3_elastic_restore_active():
            return False

        # prepare_new_worker runs outside model forward/capture context, so
        # _EXTRA_CTX is unavailable here. The V3 restore setup above creates
        # the MC2/FUSED_MC2 methods explicitly; use them directly.
        moe_comm_method = get_moe_comm_method(MoECommType.MC2)
        if moe_comm_method is None:
            moe_comm_method = get_moe_comm_method(MoECommType.FUSED_MC2)
        if moe_comm_method is None:
            return False

        token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
        prepare_v3_buffer = getattr(token_dispatcher, "prepare_v3_buffer", None)
        if prepare_v3_buffer is None:
            return False

        dtype = getattr(module.moe_config, "in_dtype", self.worker.vllm_config.model_config.dtype)
        return bool(
            prepare_v3_buffer(
                hidden_size=module.hidden_size,
                moe_expert_num=module.moe_config.num_experts,
                topk=module.top_k,
                dtype=dtype,
                device="npu",
            )
        )

    def prepare_new_worker(self) -> None:
        moe_modules = [module for module in self.worker.model_runner.model.modules() if isinstance(module, FusedMoE)]
        prepared_v3_buffers = 0
        use_v3_restore_setup = getattr(self, "_v3_new_rank_capture_elastic_info", False)
        configure_new_rank_capture_dp_sync(use_v3_restore_setup)
        for module_index, module in enumerate(moe_modules):
            with set_current_vllm_config(self.worker.vllm_config):
                if use_v3_restore_setup:
                    setup_moe_mc2_comm_and_quant_method(module, setup_comm_method=(module_index == 0))
                else:
                    setup_moe_comm_and_quant_method(module)
        if moe_modules and self._prepare_new_rank_v3_buffer(moe_modules[0]):
            prepared_v3_buffers = 1
        self._skip_next_eplb_workspace_rewarm = prepared_v3_buffers > 0
        if prepared_v3_buffers:
            torch.npu.synchronize()
            logger.info(
                "[Elastic EP scale-up] prepared new-rank V3 MoE buffers: count=%s, elastic_info=%s",
                prepared_v3_buffers,
                get_elastic_info().detach().cpu().tolist() if get_elastic_info() is not None else None,
            )
