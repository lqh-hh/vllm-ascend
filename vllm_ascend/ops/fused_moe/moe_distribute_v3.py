# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""MoeDistribute V3 adapter for the Ascend MC2 token dispatcher.

The optional ``npu_ops_transformer`` dependency is imported lazily so the
normal V2 path keeps the same import and startup behavior.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _world
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from vllm_ascend.distributed.parallel_state import (
    get_mc2_group,
    get_v3_elastic_info,
)
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoEMC2CombineMetadata,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

EXPERT_TOKEN_NUMS_TYPE_COUNT = 1


class MoeDistributeV3Adapter:
    """Wrap ``MoeDistributeBuffer`` behind the current MC2 interface.

    A buffer is keyed by its process group and static tensor dimensions. A
    graph-captured buffer can later switch to a same-sized MC2 group through
    ``update_ctx_to_mc2_group`` without rebuilding the graph.
    """

    _COMM_ALG = ""

    def __init__(self, max_tokens_per_rank: int) -> None:
        self.max_tokens_per_rank = max_tokens_per_rank
        self._buffer = None
        self._buffer_key: tuple | None = None
        self._ccl_buffer_size = None
        self._elastic_info_signature: tuple[int, ...] | None = None

    @staticmethod
    def _current_elastic_info_signature() -> tuple[int, ...] | None:
        elastic_info = get_v3_elastic_info()
        if elastic_info is None:
            return None
        return tuple(int(value) for value in elastic_info.detach().cpu().tolist())

    @staticmethod
    @contextmanager
    def _register_stateless_group_rank(
        device_group,
        rank: int,
        world_size: int,
    ):
        """Temporarily expose a stateless PG to PyTorch rank lookup.

        ``MoeDistributeBuffer`` calls ``dist.get_rank(group)`` internally,
        while vLLM's stateless process groups deliberately are not registered
        in the default c10d world. Keep this compatibility shim tightly scoped
        to buffer construction/update until the operator accepts an explicit
        rank.
        """
        try:
            default_rank = dist.get_rank()
        except Exception:
            yield
            return

        had_mapping = device_group in _world.pg_group_ranks
        previous_mapping = _world.pg_group_ranks.get(device_group)
        try:
            group_ranks = {group_rank: group_rank for group_rank in range(world_size)}
            group_ranks[default_rank] = rank
            _world.pg_group_ranks[device_group] = group_ranks
            yield
        finally:
            if had_mapping:
                _world.pg_group_ranks[device_group] = previous_mapping
            else:
                _world.pg_group_ranks.pop(device_group, None)

    @staticmethod
    def _load_buffer_cls():
        try:
            from npu_ops_transformer.ops import MoeDistributeBuffer
        except ImportError as exc:
            raise RuntimeError(
                "MoeDistribute V3 requires npu_ops_transformer. Install a "
                "version that provides npu_ops_transformer.ops."
            ) from exc
        return MoeDistributeBuffer

    def _ensure_buffer(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        moe_expert_num: int,
    ):
        mc2_group = get_mc2_group()
        device_group = mc2_group.device_group
        hidden_size = hidden_states.shape[-1]
        topk = topk_ids.shape[-1]
        buffer_key = (
            id(device_group),
            mc2_group.world_size,
            self.max_tokens_per_rank,
            hidden_size,
            moe_expert_num,
            topk,
            self._COMM_ALG,
        )
        if self._buffer is not None and self._buffer_key == buffer_key:
            return self._buffer

        buffer_cls = self._load_buffer_cls()
        ccl_buffer_size = buffer_cls.get_low_latency_ccl_buffer_size(
            mc2_group.world_size,
            self.max_tokens_per_rank,
            hidden_size,
            moe_expert_num,
            topk,
            comm_alg=self._COMM_ALG,
        )
        with self._register_stateless_group_rank(
            device_group,
            mc2_group.rank_in_group,
            mc2_group.world_size,
        ):
            self._buffer = buffer_cls(
                device_group,
                ccl_buffer_size=ccl_buffer_size,
                comm_alg=0,
            )
        self._buffer_key = buffer_key
        self._ccl_buffer_size = ccl_buffer_size
        self._elastic_info_signature = self._current_elastic_info_signature()
        logger.info(
            "Initialized MoeDistribute V3 buffer: ep_rank=%s, ep_size=%s, "
            "hidden_size=%s, experts=%s, topk=%s",
            mc2_group.rank_in_group,
            mc2_group.world_size,
            hidden_size,
            moe_expert_num,
            topk,
        )
        return self._buffer

    def prepare_for_shape(
        self,
        *,
        hidden_size: int,
        topk: int,
        moe_expert_num: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> bool:
        hidden_states = torch.empty((1, hidden_size), dtype=dtype, device=device)
        topk_ids = torch.empty((1, topk), dtype=torch.int32, device=device)
        self._ensure_buffer(hidden_states, topk_ids, moe_expert_num)
        return True

    def update_ctx_to_mc2_group(
        self,
        mc2_group: GroupCoordinator | None = None,
    ) -> bool:
        """Switch an existing buffer to a same-sized MC2 process group."""
        if self._buffer is None or self._buffer_key is None:
            return False

        mc2_group = mc2_group or get_mc2_group()
        if self._buffer_key[1] != mc2_group.world_size:
            raise RuntimeError(
                "MoeDistribute V3 update_ctx requires the captured EP size "
                f"to remain unchanged, but got old={self._buffer_key[1]}, "
                f"new={mc2_group.world_size}."
            )
        elastic_info_signature = self._current_elastic_info_signature()
        if (
            self._buffer_key[0] == id(mc2_group.device_group)
            and self._elastic_info_signature == elastic_info_signature
        ):
            return False

        if self._ccl_buffer_size is not None:
            self._buffer.ccl_buffer_size.value = self._ccl_buffer_size
        with (
            self._register_stateless_group_rank(
                mc2_group.device_group,
                mc2_group.rank_in_group,
                mc2_group.world_size,
            ),
            torch.inference_mode(),
        ):
            self._buffer.update_ctx(mc2_group.device_group)
        self._buffer_key = (id(mc2_group.device_group), *self._buffer_key[1:])
        self._elastic_info_signature = elastic_info_signature
        return True

    @staticmethod
    def _check_quant_mode(quant_mode: int) -> None:
        if quant_mode not in (0, 2):
            raise RuntimeError(
                "MoeDistribute V3 supports communication quant_mode 0 or 2, "
                f"but got {quant_mode}."
            )

    @staticmethod
    def _normalize_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
        if topk_ids.dtype == torch.int32:
            return topk_ids
        return topk_ids.to(torch.int32)

    @staticmethod
    def _remap_topk_ids_for_capture(
        topk_ids: torch.Tensor,
        elastic_info: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm_ascend.distributed.elastic_ep.v3_capture import (
            get_v3_capture_session,
        )

        if elastic_info is None or get_v3_capture_session() is None:
            return topk_ids
        # Capture uses only the newly added ranks. Route deterministic dummy
        # tokens to their dense expert range and keep experts unique per row.
        active_experts = torch.clamp(
            elastic_info[3].to(topk_ids.device),
            min=1,
        )
        row = torch.arange(
            topk_ids.shape[0],
            dtype=torch.int32,
            device=topk_ids.device,
        ).unsqueeze(1)
        column = torch.arange(
            topk_ids.shape[1],
            dtype=torch.int32,
            device=topk_ids.device,
        ).unsqueeze(0)
        return torch.remainder(row + column, active_experts)

    @staticmethod
    def _is_graph_mode() -> bool:
        try:
            forward_context = get_forward_context()
        except Exception:
            return False
        additional_kwargs = getattr(forward_context, "additional_kwargs", None)
        if additional_kwargs is not None:
            mode = additional_kwargs.get("cudagraph_runtime_mode")
        else:
            mode = getattr(forward_context, "cudagraph_runtime_mode", None)
        if mode is None:
            return False
        if isinstance(mode, CUDAGraphMode):
            return mode != CUDAGraphMode.NONE
        try:
            return CUDAGraphMode(mode) != CUDAGraphMode.NONE
        except (TypeError, ValueError):
            return bool(mode)

    @classmethod
    def _active_mask_for_graph(
        cls,
        active_mask: torch.Tensor | None,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            active_mask is None
            or not cls._is_graph_mode()
            or active_mask.shape[0] != topk_ids.shape[0]
        ):
            return None
        return active_mask

    def dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
        moe_expert_num: int,
        quant_mode: int,
    ) -> MoETokenDispatchOutput[MoEMC2CombineMetadata]:
        self._check_quant_mode(quant_mode)
        hidden_states = token_dispatch_input.hidden_states
        topk_ids = self._normalize_topk_ids(token_dispatch_input.topk_ids)
        elastic_info = get_v3_elastic_info()
        topk_ids = self._remap_topk_ids_for_capture(topk_ids, elastic_info)
        buffer = self._ensure_buffer(hidden_states, topk_ids, moe_expert_num)

        kwargs = {
            "x": hidden_states,
            "topk_idx": topk_ids,
            "num_experts": moe_expert_num,
            "quant_mode": quant_mode,
            "comm_alg": self._COMM_ALG,
            "num_max_dispatch_tokens_per_rank": topk_ids.shape[0],
        }
        active_mask = self._active_mask_for_graph(
            token_dispatch_input.routing.mc2_mask,
            topk_ids,
        )
        if active_mask is not None:
            kwargs["x_active_mask"] = active_mask
        if elastic_info is not None:
            kwargs["elastic_info"] = elastic_info

        (
            expand_x,
            dynamic_scale,
            assist_info_for_combine,
            expert_token_nums,
            ep_recv_counts,
            expand_scales,
        ) = buffer.npu_low_latency_dispatch(**kwargs)
        if not token_dispatch_input.quant.dispatch_with_quant:
            dynamic_scale = None

        return MoETokenDispatchOutput(
            hidden_states=expand_x,
            dynamic_scale=dynamic_scale,
            group_list=expert_token_nums,
            group_list_type=EXPERT_TOKEN_NUMS_TYPE_COUNT,
            combine_metadata=MoEMC2CombineMetadata(
                topk_ids=topk_ids,
                topk_weights=token_dispatch_input.topk_weights,
                expert_map=token_dispatch_input.routing.expert_map,
                ep_recv_counts=ep_recv_counts,
                tp_recv_counts=torch.empty(
                    0,
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                assist_info_for_combine=assist_info_for_combine,
                expand_scales=expand_scales,
                quant=token_dispatch_input.quant,
                mc2_mask=token_dispatch_input.routing.mc2_mask,
                ori_x=hidden_states,
            ),
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: MoEMC2CombineMetadata,
        moe_expert_num: int,
    ) -> torch.Tensor:
        if combine_metadata.ori_x is None:
            raise RuntimeError("MoeDistribute V3 combine requires the dispatch input ori_x.")
        topk_ids = self._normalize_topk_ids(combine_metadata.topk_ids)
        buffer = self._ensure_buffer(hidden_states, topk_ids, moe_expert_num)
        kwargs = {
            "x": hidden_states,
            "topk_idx": topk_ids,
            "topk_weights": combine_metadata.topk_weights.to(torch.float32),
            "assist_info_for_combine": combine_metadata.assist_info_for_combine,
            "ep_send_counts": combine_metadata.ep_recv_counts,
            "num_experts": moe_expert_num,
            "comm_alg": self._COMM_ALG,
            "num_max_dispatch_tokens_per_rank": topk_ids.shape[0],
            "ori_x": combine_metadata.ori_x,
            "expand_scales": combine_metadata.expand_scales,
        }
        active_mask = self._active_mask_for_graph(
            combine_metadata.mc2_mask,
            topk_ids,
        )
        if active_mask is not None:
            kwargs["x_active_mask"] = active_mask
        if (elastic_info := get_v3_elastic_info()) is not None:
            kwargs["elastic_info"] = elastic_info
        return buffer.npu_low_latency_combine(**kwargs)
