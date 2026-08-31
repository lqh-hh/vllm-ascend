# Adapted from vLLM's elastic_execute.py with Ascend-specific changes:
# NPU/ACL graphs, quantized weight transfer, MC2 comm groups, PyHccl EPLB.

import gc
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn as nn
import vllm.distributed.elastic_ep.elastic_execute as upstream_elastic_execute
from torch.distributed import P2POp
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import set_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
from vllm.distributed.elastic_ep.standby_state import pop_standby_groups
from vllm.distributed.parallel_state import _replace_active_groups
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import is_moe_layer
from vllm.v1.attention.backend import AttentionImplBase
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    reset_graph_params,
    set_draft_graph_params,
    set_graph_params,
)
from vllm_ascend.distributed.elastic_ep.standby_state import (
    create_ascend_standby_groups,
    get_standby_v3_capture_dp_group,
    pop_ascend_standby_groups,
    pop_standby_v3_capture_dp_group,
)
from vllm_ascend.distributed.elastic_ep.v3_capture import (
    V3CaptureDPSyncSession,
)
from vllm_ascend.distributed.parallel_state import (
    _detach_ascend_active_groups,
    _replace_ascend_active_groups,
    get_mc2_group,
    get_v3_elastic_info,
    set_v3_elastic_info,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    get_moe_comm_method,
    setup_moe_comm_method,
)

_PATCH_LOCK = threading.Lock()
logger = init_logger(__name__)


def _match_peer_parameters(
    parameter_names: list[str],
    parameters: list[torch.Tensor],
    peer_parameter_names: list[str],
) -> list[torch.Tensor]:
    if parameter_names == peer_parameter_names:
        return parameters
    parameters_by_name = dict(zip(parameter_names, parameters, strict=True))
    common_names = sorted(parameters_by_name.keys() & set(peer_parameter_names))
    return [parameters_by_name[name] for name in common_names]


@torch.inference_mode()
def ascend_batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
    stream=None,
) -> None:
    # Ascend HCCL P2P weight transfer. Replaces upstream batch_transfer_weights via
    # monkey-patch. Differs from upstream: collects params from __dict__/AttentionImplBase,
    # negotiates parameter names via the TCP store.
    device_comm = dp_group.device_communicator
    tcp_store_group = dp_group.tcp_store_group
    if device_comm is None:
        raise ValueError("No device communicator found")

    expert_weights_set = set()
    for weight_group in expert_weights:
        for weight in weight_group:
            if isinstance(weight, torch.Tensor):
                expert_weights_set.add(weight.data_ptr())
            else:
                expert_weights_set.update(w.data_ptr() for w in weight)

    state_dict = model.state_dict()
    all_params = []
    all_params_ptrs = set()
    all_params_name = []

    for name, param in state_dict.items():
        if name.endswith("expert_map"):
            continue
        ptr = param.data_ptr()
        if ptr not in all_params_ptrs and ptr not in expert_weights_set:
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
                if data_ptr not in all_params_ptrs and data_ptr not in expert_weights_set:
                    if attr_value.device.type == "npu":
                        all_params.append(attr_value)
                        all_params_ptrs.add(data_ptr)
                        all_params_name.append(submodule_name + "." + attr_name)
            if isinstance(attr_value, AttentionImplBase):
                handle_sub_module(attr_value, submodule_name + "." + attr_name)

    for module_name, module in model.named_modules():
        handle_sub_module(module, module_name)

    if is_sender:
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
    else:
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)

    all_params = _match_peer_parameters(
        all_params_name,
        all_params,
        peer_rank_all_params_name,
    )

    assert len(all_params) > 0
    transfer_stream = stream or torch.npu.current_stream()
    with torch.npu.stream(transfer_stream):
        p2p_ops = []
        for param in all_params:
            # HCCL P2P transfers flat memory and does not honor tensor strides.
            transfer_param = param.contiguous()
            op = object.__new__(P2POp)
            op.op = torch.distributed.isend if is_sender else torch.distributed.irecv
            op.tensor = transfer_param
            op.group_peer = peer_rank
            p2p_ops.append(op)
            if transfer_param is not param:
                device_comm.batch_isend_irecv(
                    p2p_ops,
                    stream=transfer_stream,
                )
                p2p_ops.clear()
                if not is_sender:
                    param.copy_(transfer_param)

        if p2p_ops:
            device_comm.batch_isend_irecv(
                p2p_ops,
                stream=transfer_stream,
            )


def setup_moe_comm_and_quant_method(module: nn.Module) -> None:
    quant_method = getattr(module.routed_experts.quant_method, "quant_method", None)
    if hasattr(quant_method, "moe_all_to_all_group_name"):
        try:
            device_group = get_mc2_group().device_group
            local_rank = get_mc2_group().rank_in_group
            backend = device_group._get_backend(torch.device("npu"))
            quant_method.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            quant_method.moe_all_to_all_group_name = ""
    setup_moe_comm_method(module.moe_config)


class AscendElasticEPScalingExecutor(ElasticEPScalingExecutor):
    @staticmethod
    def _v3_capture_key(operation_id: str, name: str) -> str:
        return f"v3_capture/{operation_id}/bootstrap/{name}"

    @staticmethod
    def _v3_capture_store(reconfig_request: ReconfigureDistributedRequest):
        return get_cached_tcp_store_client(
            reconfig_request.new_data_parallel_master_ip,
            reconfig_request.coord_store_port,
        )

    def _publish_v3_capture_decision(
        self,
        reconfig_request: ReconfigureDistributedRequest,
        old_dp_size: int,
    ) -> bool:
        parallel_config = self.worker.vllm_config.parallel_config
        target_ep_size = (
            reconfig_request.new_data_parallel_size
            * parallel_config.tensor_parallel_size
            * parallel_config.prefill_context_parallel_size
        )
        elastic_info = get_v3_elastic_info()
        has_inactive_ranks = False
        if elastic_info is not None:
            elastic_info_cpu = elastic_info.detach().cpu()
            has_inactive_ranks = bool(elastic_info_cpu[0].item()) and (
                int(elastic_info_cpu[1].item()) < target_ep_size
            )
        enabled = bool(
            envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3
            and reconfig_request.operation_id
            and reconfig_request.new_data_parallel_size > old_dp_size
            and parallel_config.pipeline_parallel_size == 1
            and self.worker.vllm_config.lora_config is None
            and get_mc2_group().world_size == target_ep_size
            and has_inactive_ranks
        )
        store = self._v3_capture_store(reconfig_request)
        store.set(
            self._v3_capture_key(reconfig_request.operation_id, "enabled"),
            b"1" if enabled else b"0",
        )
        store.set(
            self._v3_capture_key(reconfig_request.operation_id, "old_dp_size"),
            str(old_dp_size).encode(),
        )
        self._v3_precommit_capture = enabled
        self._v3_old_dp_size = old_dp_size
        return enabled

    def _read_v3_capture_decision(
        self,
        reconfig_request: ReconfigureDistributedRequest,
    ) -> bool:
        if not reconfig_request.operation_id:
            self._v3_precommit_capture = False
            return False
        store = self._v3_capture_store(reconfig_request)
        enabled = store.get(
            self._v3_capture_key(reconfig_request.operation_id, "enabled")
        ) == b"1"
        self._v3_precommit_capture = enabled
        self._v3_old_dp_size = int(
            store.get(
                self._v3_capture_key(
                    reconfig_request.operation_id,
                    "old_dp_size",
                )
            ).decode()
        )
        return enabled

    @contextmanager
    def _use_ascend_transfer_impl(self):
        original_transfer = upstream_elastic_execute.batch_transfer_weights
        if getattr(self, "_weight_transfer_stream", None) is not None:
            raise RuntimeError("An Ascend weight transfer is already active")
        self._weight_transfer_stream = torch.npu.Stream()
        upstream_elastic_execute.batch_transfer_weights = partial(
            ascend_batch_transfer_weights,
            stream=self._weight_transfer_stream,
        )
        try:
            yield
        finally:
            # Also drain the stream on exceptional exits before temporary
            # contiguous transfer tensors leave scope.
            try:
                self._weight_transfer_stream.synchronize()
            finally:
                upstream_elastic_execute.batch_transfer_weights = original_transfer
                self._weight_transfer_stream = None

    def _synchronize_weight_transfer(self) -> None:
        stream = getattr(self, "_weight_transfer_stream", None)
        if stream is None:
            torch.npu.synchronize()
            return
        stream.synchronize()

    def supports_precommit_graph_capture(self, operation_id: str) -> bool:
        request = getattr(self, "reconfig_request", None)
        return bool(
            getattr(self, "_v3_precommit_capture", False)
            and request is not None
            and request.operation_id == operation_id
        )

    def _make_v3_capture_session(
        self,
        operation_id: str,
    ) -> V3CaptureDPSyncSession:
        group = get_standby_v3_capture_dp_group()
        if group is None:
            raise RuntimeError("V3 capture DP group is not initialized")
        return V3CaptureDPSyncSession(
            group=group,
            operation_id=operation_id,
            old_dp_size=self._v3_old_dp_size,
        )

    def run_new_rank_capture_companion(self, operation_id: str) -> None:
        session = self._make_v3_capture_session(operation_id)
        steps = session.run_existing_rank_companion()
        self._v3_capture_companion_done = True
        if session.group.rank_in_group == 0:
            print(
                "[Elastic EP] V3 capture companion completed: "
                f"operation={operation_id}, steps={steps}",
                flush=True,
            )

    def capture_new_rank_graphs(self, operation_id: str) -> None:
        session = self._make_v3_capture_session(operation_id)
        try:
            self.warmup_local_kernels()
            with session.activate_for_capture():
                self.warm_and_capture()
        except BaseException as error:
            session.mark_capture_failed(error)
            raise
        session.mark_capture_done()
        self._v3_precommit_capture_done = True

    def _set_v3_identity_elastic_info(self) -> None:
        current = get_v3_elastic_info()
        if current is None:
            raise RuntimeError("V3 elastic_info is not initialized")
        world_size = get_mc2_group().world_size
        if current.numel() != 4 + 2 * world_size:
            raise RuntimeError(
                "V3 elastic_info shape does not match the committed MC2 group: "
                f"numel={current.numel()}, mc2_size={world_size}"
            )
        rank_table = torch.arange(
            world_size,
            dtype=torch.int32,
            device=current.device,
        )
        moe_module = next(
            (
                module
                for module in self.worker.get_model().modules()
                if is_moe_layer(module)
            ),
            None,
        )
        if moe_module is None:
            raise RuntimeError("V3 Elastic EP requires at least one MoE layer")
        moe_expert_num = moe_module.moe_config.num_experts
        set_v3_elastic_info(
            torch.cat(
                (
                    torch.tensor(
                        [0, world_size, 0, moe_expert_num],
                        dtype=torch.int32,
                        device=current.device,
                    ),
                    rank_table,
                    rank_table,
                )
            ).contiguous()
        )

    @staticmethod
    def _current_v3_dispatchers() -> list:
        dispatchers = []
        for comm_type in (MoECommType.MC2, MoECommType.FUSED_MC2):
            comm_method = get_moe_comm_method(comm_type)
            if comm_method is None:
                continue
            dispatcher = comm_method.token_dispatcher
            if getattr(dispatcher, "v3_adapter", None) is not None:
                dispatchers.append(dispatcher)
        return dispatchers

    def _cleanup_v3_capture_group(self) -> None:
        group = pop_standby_v3_capture_dp_group()
        if group is not None:
            group.destroy()

    def prepare_reconfiguration(self, reconfig_request: ReconfigureDistributedRequest, use_all2all: bool) -> None:
        # Let upstream prepare world / DP / EP / EPLB and weight transfer, then
        # add the Ascend-specific MC2 standby group used at commit time.
        old_dp_size = get_dp_group().world_size
        use_v3_capture = self._publish_v3_capture_decision(
            reconfig_request,
            old_dp_size,
        )
        super().prepare_reconfiguration(reconfig_request, use_all2all)
        create_ascend_standby_groups(
            new_dp_size=reconfig_request.new_data_parallel_size,
            new_world_size_across_dp=(
                self.worker.vllm_config.parallel_config.world_size * reconfig_request.new_data_parallel_size
            ),
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
            create_v3_capture_dp=use_v3_capture,
        )
        if use_v3_capture:
            # Pair with receive_expert_mapping on newly launched ranks. Doing
            # this in prepare keeps mapping setup out of the commit pause.
            self.broadcast_expert_mapping()

    def transfer_weights(self, old_dp_size: int, new_dp_size: int) -> None:
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().transfer_weights(old_dp_size=old_dp_size, new_dp_size=new_dp_size)

    def _release_cuda_graphs(self) -> None:
        if getattr(self, "_preserve_v3_graphs_during_switch", False):
            return
        if isinstance(self.worker.model_runner.model, UBatchWrapper):
            raise RuntimeError("DBO is not yet supported in elastic EP")

        ACLGraphWrapper.clear_all_graphs()

        torch.compiler.reset()
        with set_current_vllm_config(self.worker.vllm_config):
            reset_compile_wrapper(self.worker.model_runner.get_model())

        reset_graph_params()

        mgr = self.worker.model_runner.cudagraph_manager
        if mgr is not None:
            mgr.graphs.clear()
            mgr._graphs_captured = False
            # NPU graph pools cache old allocations; a fresh pool is
            # required before re-capture (NPUCachingAllocator.cpp:2106).
            mgr.pool = current_platform.graph_pool_handle()
            if hasattr(mgr, "capture_sizes"):
                capture_sizes = mgr.capture_sizes
                if self.worker.model_runner.use_aclgraph:
                    set_graph_params(capture_sizes)
                    if self.worker.model_runner.speculative_config:
                        set_draft_graph_params(capture_sizes)

        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def switch_and_remove(self) -> None:
        super().switch_and_remove()
        _replace_ascend_active_groups(mc2=None)

    def _activate_ascend_standby_groups(self) -> None:
        _replace_ascend_active_groups(**pop_ascend_standby_groups())
        parallel_config = self.worker.vllm_config.parallel_config
        self.worker.model_runner.dp_size = parallel_config.data_parallel_size
        self.worker.model_runner.dp_rank = parallel_config.data_parallel_rank
        moe_modules = [module for module in self.worker.model_runner.model.modules() if is_moe_layer(module)]
        for module in moe_modules:
            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()

    def switch_and_prepare(self):
        retired_groups = super().switch_and_prepare()
        self._activate_ascend_standby_groups()
        self._setup_moe_comm_and_quant_method()
        return retired_groups

    def commit_scale_up(self, is_existing_worker: bool) -> None:
        if getattr(self, "_v3_precommit_capture", False):
            if is_existing_worker:
                if not getattr(self, "_v3_capture_companion_done", False):
                    raise RuntimeError("V3 capture companion has not completed")
                captured_dispatchers = self._current_v3_dispatchers()
                self._preserve_v3_graphs_during_switch = True
                try:
                    retired_groups = ElasticEPScalingExecutor.switch_and_prepare(
                        self
                    )
                finally:
                    self._preserve_v3_graphs_during_switch = False
                self._activate_ascend_standby_groups()
                for dispatcher in captured_dispatchers:
                    dispatcher.refresh_hccl_group()
            else:
                if not getattr(self, "_v3_precommit_capture_done", False):
                    raise RuntimeError("New-rank V3 graph capture has not completed")
                retired_groups = None

            self._set_v3_identity_elastic_info()
            self._perform_eplb_reshuffle(async_op=True)
            if retired_groups is not None:
                self._start_group_cleanup(retired_groups)
            self._cleanup_v3_capture_group()
            return

        if not is_existing_worker:
            # New workers already use the new upstream DP/EP groups and do not
            # run switch_and_prepare. Install the MC2 group they created in
            # prepare_new_worker before expert mapping initializes MoE comms.
            self._activate_ascend_standby_groups()
        super().commit_scale_up(is_existing_worker)

    def _can_preserve_v3_scale_down(self, new_dp_size: int) -> bool:
        parallel_config = self.worker.vllm_config.parallel_config
        eplb_state = self.worker.model_runner.eplb_state
        old_dp_size = parallel_config.data_parallel_size
        physical_ep_size = get_mc2_group().world_size
        expected_physical_ep_size = (
            old_dp_size
            * parallel_config.tensor_parallel_size
            * parallel_config.prefill_context_parallel_size
        )
        elastic_info = get_v3_elastic_info()
        return bool(
            envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3
            and 0 < new_dp_size < old_dp_size
            and parallel_config.tensor_parallel_size == 1
            and parallel_config.pipeline_parallel_size == 1
            and parallel_config.prefill_context_parallel_size == 1
            and self.worker.vllm_config.lora_config is None
            and physical_ep_size == expected_physical_ep_size
            and elastic_info is not None
            and elastic_info.numel() == 4 + 2 * physical_ep_size
            # Async EPLB performs collectives through the active EP group.
            # The fast path intentionally retains the larger physical group,
            # so it is safe only with synchronous EPLB until scale-up restores
            # all physical ranks.
            and eplb_state is not None
            and not eplb_state.is_async
            and self._current_v3_dispatchers()
        )

    def _set_v3_active_elastic_info(self, new_dp_size: int) -> None:
        current = get_v3_elastic_info()
        if current is None:
            raise RuntimeError("V3 elastic_info is not initialized")
        physical_ep_size = get_mc2_group().world_size
        active_ep_size = new_dp_size
        if not 0 < active_ep_size < physical_ep_size:
            raise RuntimeError(
                "Invalid V3 scale-down size: "
                f"active_ep_size={active_ep_size}, "
                f"physical_ep_size={physical_ep_size}"
            )
        moe_module = next(
            (
                module
                for module in self.worker.model_runner.get_model().modules()
                if is_moe_layer(module)
            ),
            None,
        )
        if moe_module is None:
            raise RuntimeError("V3 Elastic EP requires at least one MoE layer")
        num_local_experts = moe_module.moe_config.num_local_experts
        orig_to_dense = torch.full(
            (physical_ep_size,),
            -1,
            dtype=torch.int32,
            device=current.device,
        )
        dense_to_orig = torch.full_like(orig_to_dense, -1)
        active_ranks = torch.arange(
            active_ep_size,
            dtype=torch.int32,
            device=current.device,
        )
        orig_to_dense[:active_ep_size] = active_ranks
        dense_to_orig[:active_ep_size] = active_ranks
        set_v3_elastic_info(
            torch.cat(
                (
                    torch.tensor(
                        [
                            1,
                            active_ep_size,
                            0,
                            active_ep_size * num_local_experts,
                        ],
                        dtype=torch.int32,
                        device=current.device,
                    ),
                    orig_to_dense,
                    dense_to_orig,
                )
            ).contiguous()
        )

    def _update_parallel_config_after_scale_down(self) -> None:
        request = self.reconfig_request
        if request is None:
            raise RuntimeError("Missing Elastic EP reconfiguration request")
        parallel_config = self.worker.vllm_config.parallel_config
        parallel_config.data_parallel_size = request.new_data_parallel_size
        if request.new_data_parallel_rank != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank = request.new_data_parallel_rank
        if (
            request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            request.new_data_parallel_master_port
        )
        parallel_config._data_parallel_master_port_list = (
            request.new_data_parallel_master_port_list
        )
        parallel_config._coord_store_port = request.coord_store_port
        self.worker.model_runner.dp_size = request.new_data_parallel_size
        self.worker.model_runner.dp_rank = parallel_config.data_parallel_rank

    def _commit_v3_scale_down_model_state(self, new_dp_size: int) -> None:
        model_runner = self.worker.model_runner
        eplb_state = model_runner.eplb_state
        if eplb_state is None:
            raise RuntimeError("V3 Elastic EP scale-down requires EPLB state")
        model = model_runner.get_model()
        model_config = model_runner.model_config
        model_state = eplb_state.model_states[model_config.compute_hash()]
        moe_module = next(
            (module for module in model.modules() if is_moe_layer(module)),
            None,
        )
        if moe_module is None:
            raise RuntimeError("V3 Elastic EP requires at least one MoE layer")
        num_local_experts = moe_module.moe_config.num_local_experts
        num_physical_experts = new_dp_size * num_local_experts
        num_logical_experts = model_state.logical_replica_count.shape[1]
        if num_physical_experts < num_logical_experts:
            raise RuntimeError(
                "Scale-down leaves fewer physical than logical experts: "
                f"physical={num_physical_experts}, logical={num_logical_experts}"
            )

        model_state.expert_load_pass = model_state.expert_load_pass[
            :, :num_physical_experts
        ]
        model_state.expert_load_window = model_state.expert_load_window[
            :, :, :num_physical_experts
        ]
        parallel_config = self.worker.vllm_config.parallel_config
        parallel_config.eplb_config.num_redundant_experts = (
            num_physical_experts - num_logical_experts
        )
        model.expert_weights = []
        with set_current_vllm_config(self.worker.vllm_config):
            model.set_eplb_state(
                model_state.expert_load_pass,
                model_state.logical_to_physical_map,
                model_state.logical_replica_count,
            )
            eplb_state._propagate_shared_tensors(
                model,
                model_state.num_unpadded_tokens_tensors,
            )
            model.update_physical_experts_metadata(
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_local_experts,
            )

        if self._prepared_eplb_communicator is None:
            raise RuntimeError("Standby EPLB communicator was not prepared")
        eplb_state.update_communicator(
            model_config,
            self._prepared_eplb_communicator,
        )
        self._prepared_eplb_communicator = None
        self._staged_moe_quant_methods.clear()

        for module in model.modules():
            if not is_moe_layer(module):
                continue
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()

    def _switch_v3_scale_down_survivor(self, new_dp_size: int) -> None:
        standby_groups = pop_standby_groups()
        unused_standby_ep = standby_groups["ep"]
        standby_groups["ep"] = get_ep_group()
        retired_groups = list(_replace_active_groups(**standby_groups))
        # Keep the original physical EP group. The newly-created, smaller EP
        # group is unused by the captured V3 graph and can be retired instead.
        retired_groups[1] = unused_standby_ep

        standby_mc2 = pop_ascend_standby_groups()["mc2"]
        self._update_parallel_config_after_scale_down()
        self._commit_v3_scale_down_model_state(new_dp_size)
        self._set_v3_active_elastic_info(new_dp_size)
        for dispatcher in self._current_v3_dispatchers():
            dispatcher.refresh_hccl_group()
        self._start_group_cleanup(tuple(retired_groups) + (standby_mc2,))
        logger.info(
            "[Elastic EP] V3 scale-down preserved physical EP/MC2 and NPU "
            "graphs: active_dp=%d, physical_ep=%d",
            new_dp_size,
            get_mc2_group().world_size,
        )

    def _switch_v3_scale_down_removed_rank(self) -> None:
        retired_groups = _replace_active_groups(
            world=None,
            dp=None,
            ep=None,
            eplb=None,
            node_count=None,
        )
        _detach_ascend_active_groups()
        # Match survivor cleanup order for the old groups, but intentionally
        # keep the physical EP/MC2 communicators alive on survivor ranks.
        self._destroy_retired_groups(
            (
                retired_groups[0],
                None,
                retired_groups[2],
                retired_groups[3],
            )
        )

    def commit_scale_down(self, new_dp_size: int, removing: bool) -> None:
        if not self._can_preserve_v3_scale_down(new_dp_size):
            super().commit_scale_down(new_dp_size, removing)
            return

        self.perform_scale_down_eplb_reshuffle(new_dp_size)
        if removing:
            self._switch_v3_scale_down_removed_rank()
        else:
            self._switch_v3_scale_down_survivor(new_dp_size)

    def receive_expert_mapping(self) -> torch.Tensor:
        mapping = super().receive_expert_mapping()
        self._setup_moe_comm_and_quant_method()
        return mapping

    def prepare_new_worker(
        self,
        reconfig_request: ReconfigureDistributedRequest | None = None,
    ) -> str | None:
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().prepare_new_worker(reconfig_request)
        parallel_config = self.worker.vllm_config.parallel_config
        if reconfig_request is None:
            coord_store = get_cached_tcp_store_client(
                parallel_config.data_parallel_master_ip,
                parallel_config._coord_store_port,
            )
            current_epoch_key = "elastic_ep/external/current_epoch"
            if coord_store.check([current_epoch_key]):
                operation_id = coord_store.get(current_epoch_key).decode()
                reconfig_request = ReconfigureDistributedRequest(
                    new_data_parallel_size=parallel_config.data_parallel_size,
                    new_data_parallel_rank=parallel_config.data_parallel_rank,
                    new_data_parallel_rank_local=(
                        parallel_config.data_parallel_rank_local
                    ),
                    new_data_parallel_master_ip=(
                        parallel_config.data_parallel_master_ip
                    ),
                    new_data_parallel_master_port=(
                        parallel_config.data_parallel_master_port
                    ),
                    new_data_parallel_master_port_list=(
                        parallel_config._data_parallel_master_port_list
                    ),
                    coord_store_port=parallel_config._coord_store_port,
                    operation_id=operation_id,
                )
                self.reconfig_request = reconfig_request
        use_v3_capture = bool(
            reconfig_request is not None
            and self._read_v3_capture_decision(reconfig_request)
        )
        # Existing workers create this group after their upstream preparation.
        # The new worker follows prepare_new_worker instead, so it must join the
        # same stateless MC2 creation here to avoid leaving old ranks waiting.
        create_ascend_standby_groups(
            new_dp_size=parallel_config.data_parallel_size,
            new_world_size_across_dp=parallel_config.world_size * parallel_config.data_parallel_size,
            master_ip=parallel_config.data_parallel_master_ip,
            coord_store_port=parallel_config._coord_store_port,
            create_v3_capture_dp=use_v3_capture,
        )
        if not use_v3_capture:
            return (
                reconfig_request.operation_id
                if reconfig_request is not None
                else None
            )

        # New ranks have no active MC2 group yet. Install the final-size
        # standby group early so graph capture can use it before commit.
        self._activate_ascend_standby_groups()

        mapping = self.receive_expert_mapping()
        self.worker.model_runner.setup_eplb_from_mapping(mapping)

        moe_modules = [
            module
            for module in self.worker.get_model().modules()
            if is_moe_layer(module)
        ]
        if not moe_modules:
            raise RuntimeError("V3 pre-commit capture requires at least one MoE layer")

        new_ep_size = get_mc2_group().world_size
        new_dp_size = parallel_config.data_parallel_size
        if new_ep_size % new_dp_size != 0:
            raise RuntimeError(
                "MC2 size must divide evenly by DP size for V3 capture: "
                f"mc2={new_ep_size}, dp={new_dp_size}"
            )
        old_ep_size = self._v3_old_dp_size * (new_ep_size // new_dp_size)
        active_ranks = list(range(old_ep_size, new_ep_size))
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        table_orig_to_dense = torch.full(
            (new_ep_size,),
            -1,
            dtype=torch.int32,
            device=self.worker.device,
        )
        table_dense_to_orig = torch.full_like(table_orig_to_dense, -1)
        table_orig_to_dense[active_ranks] = torch.arange(
            len(active_ranks),
            dtype=torch.int32,
            device=self.worker.device,
        )
        table_dense_to_orig[: len(active_ranks)] = torch.tensor(
            active_ranks,
            dtype=torch.int32,
            device=self.worker.device,
        )
        set_v3_elastic_info(
            torch.cat(
                (
                    torch.tensor(
                        [
                            1,
                            len(active_ranks),
                            0,
                            len(active_ranks) * num_local_experts,
                        ],
                        dtype=torch.int32,
                        device=self.worker.device,
                    ),
                    table_orig_to_dense,
                    table_dense_to_orig,
                )
            ).contiguous(),
            allow_shape_change=True,
        )
        self._setup_moe_comm_and_quant_method()

        moe_comm_method = get_moe_comm_method(MoECommType.MC2)
        if moe_comm_method is None:
            raise RuntimeError("MC2 communication method is not initialized")
        dispatcher = moe_comm_method.token_dispatcher
        first_moe_config = moe_modules[0].moe_config
        prepared = dispatcher.prepare_v3_buffer(
            hidden_size=first_moe_config.hidden_dim,
            moe_expert_num=first_moe_config.num_experts,
            topk=first_moe_config.experts_per_token,
            dtype=self.worker.vllm_config.model_config.dtype,
            device=self.worker.device,
        )
        if not prepared:
            raise RuntimeError("Failed to prepare MoeDistribute V3 buffer")
        return reconfig_request.operation_id

    def warmup_local_kernels(self) -> None:
        pass

    def warm_and_capture(self) -> None:
        # No need to save/clear/restore the KV-cache block tables like the
        # upstream warm_and_capture: the V2 runner's dummy attention uses
        # all-zero block tables (reserved null block) and PAD_SLOT_ID slot
        # mappings, so real KV-cache blocks are never written during the
        # dummy run.
        runner = self.worker.model_runner
        self._release_cuda_graphs()
        unlock_workspace()
        try:
            runner._dummy_run(
                runner.max_num_tokens,
                is_profile=True,
                skip_eplb=True,
            )
            self.worker.compile_or_warm_up_model()
        finally:
            lock_workspace()

    def _setup_moe_comm_and_quant_method(self) -> None:
        moe_modules = [module for module in self.worker.get_model().modules() if is_moe_layer(module)]
        for module in moe_modules:
            with set_current_vllm_config(self.worker.vllm_config):
                setup_moe_comm_and_quant_method(module)
