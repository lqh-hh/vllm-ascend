# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generic

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.distributed_c10d import _world
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import get_forward_context

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.elastic_ep.v3_capture_dp_sync import (
    capture_dp_sync_group,
)
from vllm_ascend.distributed.parallel_state import (
    get_elastic_info,
    get_mc2_group,
    set_elastic_info,
)
from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all, gather_from_sequence_parallel_region
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoEAllToAllCombineMetadata,
    MoEMC2CombineMetadata,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
    TMoECombineMetadata,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
    is_hierarchical_communication_enabled,
    should_skip_allreduce_across_dp_group,
)

EXPERT_TOKEN_NUMS_TYPE_CUMSUM = 0
EXPERT_TOKEN_NUMS_TYPE_COUNT = 1


def _get_expert_token_nums_type(token_dispatch_input: MoETokenDispatchInput) -> int:
    # grouped_matmul_swiglu_quant_v2 consumes per-expert counts; existing
    # MC2 grouped-matmul paths consume prefix sums.
    if token_dispatch_input.quant.use_w4a8_per_channel_gmm_swiglu:
        return EXPERT_TOKEN_NUMS_TYPE_COUNT
    return EXPERT_TOKEN_NUMS_TYPE_CUMSUM


class MoETokenDispatcher(ABC, Generic[TMoECombineMetadata]):
    def __init__(self, **kwargs) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.top_k = kwargs.get("top_k", 0)
        self.num_experts = kwargs.get("num_experts", 0)

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_ep_group().device_group

    @property
    def ep_rank(self):
        return get_ep_group().rank_in_group

    @property
    def ep_size(self):
        return get_ep_group().world_size

    @abstractmethod
    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput[TMoECombineMetadata]:
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: TMoECombineMetadata,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Combine function not implemented.")


class _MoeDistributeV3Adapter:
    """Adapter for MoeDistribute V3 low latency dispatch/combine."""

    _COMM_ALG = ""
    _INSTANCES: "weakref.WeakSet[_MoeDistributeV3Adapter]" = weakref.WeakSet()

    def __init__(self, max_tokens_per_rank: int) -> None:
        self.max_tokens_per_rank = max_tokens_per_rank
        self._buffer = None
        self._buffer_key = None
        self._ccl_buffer_size = None
        self._elastic_info_signature = None
        self._INSTANCES.add(self)

    @staticmethod
    def _current_elastic_info_signature():
        elastic_info = get_elastic_info()
        if elastic_info is None:
            return None
        try:
            return tuple(int(x) for x in elastic_info.detach().cpu().tolist())
        except RuntimeError:
            return "<unavailable>"

    @staticmethod
    @contextmanager
    def _use_stateless_group_rank(device_group, rank: int, world_size: int):
        try:
            default_rank = dist.get_rank()
        except Exception:
            yield
            return

        had_old_group_ranks = device_group in _world.pg_group_ranks
        old_group_ranks = _world.pg_group_ranks.get(device_group)
        try:
            group_ranks = {i: i for i in range(world_size)}
            group_ranks[default_rank] = int(rank)
            _world.pg_group_ranks[device_group] = group_ranks
            yield
        finally:
            if had_old_group_ranks:
                _world.pg_group_ranks[device_group] = old_group_ranks
            else:
                _world.pg_group_ranks.pop(device_group, None)

    def _ensure_buffer(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        moe_expert_num: int,
    ):
        from npu_ops_transformer.ops import MoeDistributeBuffer

        mc2_group = get_mc2_group()
        device_group = mc2_group.device_group
        ep_world_size = mc2_group.world_size
        hidden_size = hidden_states.shape[-1]
        topk = topk_ids.shape[-1]
        buffer_key = (
            id(device_group),
            ep_world_size,
            self.max_tokens_per_rank,
            hidden_size,
            moe_expert_num,
            topk,
            self._COMM_ALG,
        )
        if self._buffer is not None and self._buffer_key == buffer_key:
            return self._buffer

        ccl_buffer_size = MoeDistributeBuffer.get_low_latency_ccl_buffer_size(
            ep_world_size,
            self.max_tokens_per_rank,
            hidden_size,
            moe_expert_num,
            topk,
            comm_alg=self._COMM_ALG,
        )
        with self._use_stateless_group_rank(device_group, mc2_group.rank_in_group, ep_world_size):
            self._buffer = MoeDistributeBuffer(
                device_group,
                ccl_buffer_size=ccl_buffer_size,
                comm_alg=0,
            )
        self._buffer_key = buffer_key
        self._ccl_buffer_size = ccl_buffer_size
        self._elastic_info_signature = self._current_elastic_info_signature()
        return self._buffer

    def ensure_buffer_for_shape(
        self,
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

    def update_ctx_to_mc2_group(self, mc2_group=None) -> bool:
        if self._buffer is None or self._buffer_key is None:
            return False

        mc2_group = mc2_group or get_mc2_group()
        device_group = mc2_group.device_group
        ep_world_size = mc2_group.world_size
        old_key = self._buffer_key
        if old_key[1] != ep_world_size:
            raise RuntimeError(
                "MoeDistribute V3 update_ctx only supports switching to a group "
                f"with the captured EP size, but got old_ep_size={old_key[1]}, "
                f"new_ep_size={ep_world_size}."
            )
        elastic_info_signature = self._current_elastic_info_signature()
        if old_key[0] == id(device_group) and self._elastic_info_signature == elastic_info_signature:
            return False

        if self._ccl_buffer_size is not None:
            self._buffer.ccl_buffer_size.value = self._ccl_buffer_size
        with (
            self._use_stateless_group_rank(device_group, mc2_group.rank_in_group, ep_world_size),
            torch.inference_mode(),
        ):
            self._buffer.update_ctx(device_group)
        self._buffer_key = (id(device_group), *old_key[1:])
        self._elastic_info_signature = elastic_info_signature
        return True

    @staticmethod
    def _check_quant_mode(quant_mode: int) -> None:
        if quant_mode not in (0, 2):
            raise RuntimeError(
                f"MoeDistribute V3 low latency dispatch only supports quant_mode 0 or 2, but got {quant_mode}."
            )

    @staticmethod
    def _normalize_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
        if topk_ids.dtype == torch.int32:
            return topk_ids
        return topk_ids.to(torch.int32)

    @staticmethod
    def _remap_topk_ids_for_elastic_capture(
        topk_ids: torch.Tensor,
        elastic_info: torch.Tensor | None,
    ) -> torch.Tensor:
        if elastic_info is None or capture_dp_sync_group() is None:
            return topk_ids

        # When a new rank captures its graph alone, elastic_info may expose
        # only the new rank's local experts. Build deterministic dummy routing
        # inside that active expert range while keeping each token's top-k
        # experts unique. A plain modulo remap can create duplicate experts in
        # one token row, which CombineV3 does not handle reliably.
        active_expert_num = torch.clamp(elastic_info[3].to(topk_ids.device), min=1)
        row_offsets = torch.arange(
            topk_ids.shape[0],
            dtype=torch.int32,
            device=topk_ids.device,
        ).unsqueeze(1)
        col_offsets = torch.arange(
            topk_ids.shape[1],
            dtype=torch.int32,
            device=topk_ids.device,
        ).unsqueeze(0)
        remapped_topk_ids = torch.remainder(row_offsets + col_offsets, active_expert_num)
        return torch.where(elastic_info[0].to(topk_ids.device) != 0, remapped_topk_ids, topk_ids)

    @staticmethod
    def _num_dispatch_tokens_for_op(topk_ids: torch.Tensor) -> int:
        # In graph mode the MoE prepare path pads hidden_states/topk_ids to the
        # graph bucket first, so this shape is already the stable bucket size.
        return int(topk_ids.shape[0])

    @staticmethod
    def _get_forward_context_attr(name: str, default=None):
        try:
            forward_context = get_forward_context()
        except Exception:
            return default

        additional_kwargs = getattr(forward_context, "additional_kwargs", None)
        if additional_kwargs is not None and name in additional_kwargs:
            return additional_kwargs[name]
        return getattr(forward_context, name, default)

    @classmethod
    def _is_graph_mode(cls) -> bool:
        mode = cls._get_forward_context_attr("cudagraph_runtime_mode", CUDAGraphMode.NONE)
        if mode is None:
            return False
        if isinstance(mode, CUDAGraphMode):
            return mode != CUDAGraphMode.NONE
        try:
            return CUDAGraphMode(mode) != CUDAGraphMode.NONE
        except Exception:
            return bool(mode)

    @classmethod
    def _should_pass_active_mask(cls, active_mask: torch.Tensor | None, topk_ids: torch.Tensor) -> bool:
        if active_mask is None:
            return False
        if not cls._is_graph_mode():
            return False
        return active_mask.shape[0] == topk_ids.shape[0]

    def dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
        moe_expert_num: int,
        quant_mode: int,
    ) -> MoETokenDispatchOutput[MoEMC2CombineMetadata]:
        self._check_quant_mode(quant_mode)

        hidden_states = token_dispatch_input.hidden_states
        topk_ids = self._normalize_topk_ids(token_dispatch_input.topk_ids)
        elastic_info = get_elastic_info()
        topk_ids = self._remap_topk_ids_for_elastic_capture(topk_ids, elastic_info)
        buffer = self._ensure_buffer(hidden_states, topk_ids, moe_expert_num)

        kwargs = {
            "x": hidden_states,
            "topk_idx": topk_ids,
            "num_experts": moe_expert_num,
            "quant_mode": quant_mode,
            "comm_alg": self._COMM_ALG,
            "num_max_dispatch_tokens_per_rank": self._num_dispatch_tokens_for_op(topk_ids),
        }
        if self._should_pass_active_mask(token_dispatch_input.routing.mc2_mask, topk_ids):
            kwargs["x_active_mask"] = token_dispatch_input.routing.mc2_mask
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

        tp_recv_counts = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
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
                tp_recv_counts=tp_recv_counts,
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
            "num_max_dispatch_tokens_per_rank": self._num_dispatch_tokens_for_op(topk_ids),
            "ori_x": combine_metadata.ori_x,
            "expand_scales": combine_metadata.expand_scales,
        }
        if self._should_pass_active_mask(combine_metadata.mc2_mask, topk_ids):
            kwargs["x_active_mask"] = combine_metadata.mc2_mask
        elastic_info = get_elastic_info()
        if elastic_info is not None:
            kwargs["elastic_info"] = elastic_info

        output = buffer.npu_low_latency_combine(**kwargs)
        return output


def update_moe_distribute_v3_contexts(mc2_group=None) -> int:
    updated = 0
    adapters = list(_MoeDistributeV3Adapter._INSTANCES)
    skipped_no_buffer = 0
    skipped_same_ctx = 0
    for adapter in adapters:
        had_buffer = adapter._buffer is not None and adapter._buffer_key is not None
        if adapter.update_ctx_to_mc2_group(mc2_group):
            updated += 1
        elif not had_buffer:
            skipped_no_buffer += 1
        else:
            skipped_same_ctx += 1
    elastic_info = get_elastic_info()
    print(
        "[MoE V3 ctx] update_moe_distribute_v3_contexts: "
        f"adapters={len(adapters)}, updated={updated}, "
        f"skipped_no_buffer={skipped_no_buffer}, "
        f"skipped_same_ctx={skipped_same_ctx}, "
        f"elastic_info={elastic_info.detach().cpu().tolist() if elastic_info is not None else None}",
        flush=True,
    )
    return updated


class TokenDispatcherWithMC2(MoETokenDispatcher[MoEMC2CombineMetadata]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device_group = get_mc2_group().device_group
        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = get_mc2_group().rank_in_group
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        self.ep_rank_id = get_mc2_group().rank_in_group
        self.ep_world_size = get_mc2_group().world_size
        self.enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
        self.need_extra_args = get_ascend_device_type() in [AscendDeviceType.A3, AscendDeviceType.A5]
        self.a5_need_extra_args = get_ascend_device_type() == AscendDeviceType.A5
        # NOTE: When in A2, setting the environment variables HCCL_INTRA_PCIE_ENABLE=1 and
        # HCCL_INTRA_ROCE_ENABLE=0 can reduce cross-machine communication traffic and significantly
        # improve communication performance.
        # When enable hierarchical communication, param `expert_scales` need to be passed in.
        self.need_expert_scale = is_hierarchical_communication_enabled()

        # Here we need to calculate the global_bs = max_bs_per_rank * ep_world_size to execute
        # dispatch & combine operators with different input num_tokens per rank.
        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        compilation_config = vllm_config.compilation_config
        speculative_config = vllm_config.speculative_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        uniform_decode_query_len = 1 if not speculative_config else 1 + speculative_config.num_speculative_tokens
        decode_max_num_seqs = getattr(scheduler_config, "decode_max_num_seqs", 0)
        max_num_reqs = max(scheduler_config.max_num_seqs, decode_max_num_seqs)
        if compilation_config.cudagraph_capture_sizes:
            max_num_tokens = compilation_config.max_cudagraph_capture_size
        else:
            max_num_tokens = min(max_num_reqs * uniform_decode_query_len, 512)
        num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
        self.max_tokens_per_rank = num_tokens_per_tp_rank
        _max_global_bs = num_tokens_per_tp_rank * self.ep_world_size

        # When allreduce across DP is not skipped, tokens are uniform across ranks:
        # use global_bs=0 (uniform mode) and pass mc2_mask.
        # When allreduce is skipped, tokens may differ per rank:
        # use the real global_bs and do NOT pass mc2_mask.
        self.global_bs = _max_global_bs if should_skip_allreduce_across_dp_group(vllm_config) else 0

        # NOTE: When enable_mc2_hierarchy_comm is true, we need pass in `comm_alg` to mc2 op.
        self.need_comm_alg = get_ascend_config().enable_mc2_hierarchy_comm

        if not self.enable_dispatch_v2 and self.need_comm_alg:
            raise RuntimeError(
                "PTA and CANN version is too old to support mc2 hierarchy comm, please upgrade your version."
            )
        self.v3_adapter = (
            _MoeDistributeV3Adapter(self.max_tokens_per_rank)
            if envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3
            else None
        )
        self.elastic_info = None
        self._initial_moe_expert_num = None

    def refresh_hccl_group(self) -> None:
        """Refresh MC2 communicator metadata after HCCL groups are recreated."""
        device_group = get_mc2_group().device_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def prepare_v3_buffer(
        self,
        hidden_size: int,
        moe_expert_num: int,
        topk: int,
        dtype: torch.dtype,
        device: torch.device | str = "npu",
    ) -> bool:
        if self.v3_adapter is None:
            return False
        if hidden_size <= 0 or moe_expert_num <= 0 or topk <= 0:
            return False

        self._ensure_identity_elastic_info(moe_expert_num, torch.device(device))
        self.elastic_info = get_elastic_info()
        if self._initial_moe_expert_num is None:
            self._initial_moe_expert_num = moe_expert_num
        self.moe_expert_num = self._initial_moe_expert_num if self.elastic_info is not None else moe_expert_num
        return self.v3_adapter.ensure_buffer_for_shape(
            hidden_size=hidden_size,
            topk=topk,
            moe_expert_num=self.moe_expert_num,
            dtype=dtype,
            device=device,
        )

    def _ensure_identity_elastic_info(self, moe_expert_num: int, device: torch.device) -> None:
        if get_elastic_info() is not None:
            return
        base_config = torch.tensor(
            [0, self.ep_world_size, 0, moe_expert_num],
            dtype=torch.int32,
            device=device,
        )
        table = torch.arange(self.ep_world_size, dtype=torch.int32, device=device)
        elastic_info = torch.cat([base_config, table, table], dim=0).contiguous()
        elastic_info.requires_grad_(False)
        set_elastic_info(elastic_info)

    def _get_moe_expert_num(self, token_dispatch_input: MoETokenDispatchInput) -> int:
        expert_map = token_dispatch_input.routing.expert_map
        assert expert_map is not None, "expert_map is required for MC2 token dispatch."
        return len(expert_map) + token_dispatch_input.routing.global_redundant_expert_num

    def _get_dispatch_quant_mode(self, token_dispatch_input: MoETokenDispatchInput) -> int:
        comm_quant_mode = token_dispatch_input.quant.comm_quant_mode
        if comm_quant_mode is not None:
            return comm_quant_mode
        if token_dispatch_input.quant.dispatch_with_quant:
            return 4 if self.a5_need_extra_args and token_dispatch_input.quant.is_mxfp else 2
        return 0

    def get_dispatch_mc2_kwargs(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num

        assert expert_map is not None, "expert_map is required for MC2 token dispatch."
        # NOTE: quant_mode differs by quant feature:
        # - Legacy int communication quantization uses quant_mode=2.
        # - A5 MXFP communication uses quant_mode=4.
        quant_mode = self._get_dispatch_quant_mode(token_dispatch_input)
        current_moe_expert_num = len(expert_map) + global_redundant_expert_num
        if self._initial_moe_expert_num is None:
            self._initial_moe_expert_num = current_moe_expert_num
        self.moe_expert_num = self._initial_moe_expert_num if self.elastic_info is not None else current_moe_expert_num
        expert_token_nums_type = _get_expert_token_nums_type(token_dispatch_input)
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
            "expert_token_nums_type": expert_token_nums_type,
            "elastic_info": self.elastic_info,
        }
        if self.elastic_info is not None:
            kwargs_mc2["elastic_info"] = self.elastic_info
        if self.global_bs == 0:
            kwargs_mc2["x_active_mask"] = token_dispatch_input.routing.mc2_mask

        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        if self.need_extra_args:
            stage1_kwargs.update(
                {
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
        # Only dispatch-enabled MXFP paths pass y_dtype through MC2.
        if (
            self.a5_need_extra_args
            and (token_dispatch_input.quant.is_mxfp or token_dispatch_input.quant.is_fp8)
            and token_dispatch_input.quant.dispatch_with_quant
        ):
            y_dtype = torch.float8_e4m3fn
            if (
                token_dispatch_input.quant.mxfp is not None
                and token_dispatch_input.quant.mxfp.act_quant_type is not None
            ):
                y_dtype = token_dispatch_input.quant.mxfp.act_quant_type
            stage1_kwargs.update({"tp_world_size": 1, "tp_rank_id": 0, "y_dtype": y_dtype})
        if self.need_expert_scale or self.a5_need_extra_args:
            stage1_kwargs.update(
                {
                    "expert_scales": topk_weights.to(torch.float32),
                }
            )
        if self.need_comm_alg:
            stage1_kwargs.update({"comm_alg": "hierarchy"})

        kwargs_mc2.update(stage1_kwargs)
        return kwargs_mc2

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        current_moe_expert_num = self._get_moe_expert_num(token_dispatch_input)
        if self.v3_adapter is not None:
            self._ensure_identity_elastic_info(current_moe_expert_num, token_dispatch_input.hidden_states.device)
            self.elastic_info = get_elastic_info()
            quant_mode = self._get_dispatch_quant_mode(token_dispatch_input)
            if self._initial_moe_expert_num is None:
                self._initial_moe_expert_num = current_moe_expert_num
            self.moe_expert_num = (
                self._initial_moe_expert_num if self.elastic_info is not None else current_moe_expert_num
            )
            return self.v3_adapter.dispatch(token_dispatch_input, self.moe_expert_num, quant_mode)

        self.elastic_info = get_elastic_info()
        kwargs_mc2 = self.get_dispatch_mc2_kwargs(token_dispatch_input)
        output = (
            torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)
            if self.enable_dispatch_v2
            else torch_npu.npu_moe_distribute_dispatch(**kwargs_mc2)
        )
        # comm_stream.wait_stream(torch.npu.current_stream())
        (
            expand_x,
            dynamic_scale,
            assist_info_for_combine,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = output[0:7]
        group_list_type = kwargs_mc2["expert_token_nums_type"]
        return MoETokenDispatchOutput(
            hidden_states=expand_x,
            dynamic_scale=dynamic_scale,
            group_list=expert_token_nums,
            group_list_type=group_list_type,
            combine_metadata=MoEMC2CombineMetadata(
                topk_ids=token_dispatch_input.topk_ids,
                topk_weights=token_dispatch_input.topk_weights,
                expert_map=token_dispatch_input.routing.expert_map,
                ep_recv_counts=ep_recv_counts,
                tp_recv_counts=tp_recv_counts,
                assist_info_for_combine=assist_info_for_combine,
                expand_scales=expand_scales,
                quant=token_dispatch_input.quant,
                mc2_mask=token_dispatch_input.routing.mc2_mask if self.global_bs == 0 else None,
                ori_x=token_dispatch_input.hidden_states,
            ),
        )

    def get_combine_mc_kwargs(self, hidden_states: torch.Tensor, combine_metadata: MoEMC2CombineMetadata):
        expert_map = combine_metadata.expert_map
        topk_ids = combine_metadata.topk_ids
        topk_weights = combine_metadata.topk_weights
        ep_recv_counts = combine_metadata.ep_recv_counts
        tp_recv_counts = combine_metadata.tp_recv_counts
        assist_info_for_combine = combine_metadata.assist_info_for_combine
        expand_scales = combine_metadata.expand_scales
        quant_type = combine_metadata.quant.quant_type
        comm_quant_mode = combine_metadata.quant.comm_quant_mode

        assert expert_map is not None
        # NOTE: quant_mode differs by quant features:
        # - A5 MXFP communication uses quant_mode=4 only for MXFP8 currently.
        if comm_quant_mode is not None:
            quant_mode = comm_quant_mode
        elif quant_type == QuantType.MXFP8:
            quant_mode = 4
        else:
            quant_mode = 0
        kwargs_mc2 = {
            "expand_x": hidden_states,
            "expert_ids": topk_ids,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
            "elastic_info": self.elastic_info,
        }
        if self.elastic_info is not None:
            kwargs_mc2["elastic_info"] = self.elastic_info
        if self.global_bs == 0:
            kwargs_mc2["x_active_mask"] = combine_metadata.mc2_mask

        if combine_metadata.quant.dispatch_with_quant:
            tp_recv_counts = torch.empty(1, dtype=torch.int32, device=hidden_states.device)

        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
            "expand_scales": expand_scales,
            "comm_quant_mode": quant_mode,
        }

        if self.enable_dispatch_v2:
            stage3_kwargs["assist_info_for_combine"] = assist_info_for_combine
        else:
            stage3_kwargs["expand_idx"] = assist_info_for_combine

        if self.need_extra_args:
            stage3_kwargs.update(
                {
                    "tp_send_counts": tp_recv_counts,
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
        if self.need_comm_alg:
            stage3_kwargs.update({"comm_alg": "hierarchy"})

        kwargs_mc2.update(stage3_kwargs)
        return kwargs_mc2

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        if self.v3_adapter is not None:
            return self.v3_adapter.combine(hidden_states, combine_metadata, self.moe_expert_num)

        kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states, combine_metadata)
        combined_output = (
            torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2)
            if self.enable_dispatch_v2
            else torch_npu.npu_moe_distribute_combine(**kwargs_mc2)
        )

        return combined_output


class TokenDispatcherWithAllGather(MoETokenDispatcher[MoEAllGatherCombineMetadata]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_num_tokens = kwargs.get("max_num_tokens")
        num_experts_local = kwargs.get("num_local_experts", 0)
        self.num_experts_local = (
            num_experts_local.item() if torch.is_tensor(num_experts_local) else int(num_experts_local)
        )

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        # TODO: After AllGather MXFP4 communication quantization thorough verification, remove this judgment.
        #  MXFP4 keeps dispatch unquantized in AllGather path, and quantizes again inside the MLP path.
        with_quant = (
            token_dispatch_input.quant.dispatch_with_quant
            and token_dispatch_input.quant.quant_type != QuantType.MXFP4
            and token_dispatch_input.quant.quant_type != QuantType.W8A8FP8
        )
        is_mxfp = token_dispatch_input.quant.is_mxfp
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        dynamic_scale = token_dispatch_input.routing.pertoken_scale
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
        restore_shape = hidden_states.shape
        # Fuse the first dynamic quant of moe_mlp into initrouting when
        # dispatch_with_quant is on but got a None dynamic_scale.
        if with_quant and dynamic_scale is None:
            quant_mode = 3 if is_mxfp else 1
        else:
            quant_mode = -1

        num_tokens = hidden_states.shape[:-1].numel()
        apply_router_weight_on_input = token_dispatch_input.routing.apply_router_weight_on_input
        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            mask = expert_map[topk_ids] != -1
            topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local
        sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,
            scale=dynamic_scale,
            active_num=num_tokens * self.top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=quant_mode,
        )
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode

        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            dynamic_scale=dynamic_scale if with_quant else None,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=topk_weights,
                expanded_row_idx=expanded_row_idx,
                restore_shape=restore_shape,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        final_hidden_states = DeviceOperator.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=combine_metadata.expanded_row_idx,
            probs=combine_metadata.topk_weights,
        )
        if len(combine_metadata.restore_shape) == 3:
            final_hidden_states = final_hidden_states.view(combine_metadata.restore_shape)

        # these values are no longer used, so they need to be set to None for memory release.
        return final_hidden_states


class TokenDispatcherWithAll2AllV(MoETokenDispatcher[MoEAllToAllCombineMetadata]):
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_local_experts = kwargs.get("num_local_experts", 0)

        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
            )

        local_expert_indices_offset = self.ep_rank * self.num_local_experts

        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        assert len(self.local_expert_indices) == self.num_local_experts, "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1, (
                "local_expert_indices must be continuous"
            )

        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = get_ep_group().rank_in_group
        backend = self.ep_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        with_quant = token_dispatch_input.quant.is_int_quant or token_dispatch_input.quant.is_fp8
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids

        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        ) = self._dispatch_preprocess(hidden_states, topk_ids)

        dynamic_scale_after_all2all = None
        if with_quant:
            dst_type = torch.float8_e4m3fn if token_dispatch_input.quant.is_fp8 else torch.int8
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens, dst_type=dst_type
            )
            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group
            )
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits, self.ep_group
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # Postprocess
        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = (
            self._dispatch_postprocess(
                global_input_tokens,
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
                with_quant,
            )
        )

        return MoETokenDispatchOutput(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            combine_metadata=MoEAllToAllCombineMetadata(
                input_splits=input_splits,
                output_splits=output_splits,
                topk_weights=topk_weights,
                reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
                reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
                hidden_shape=hidden_shape,
                hidden_shape_before_permute=hidden_shape_before_permute,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        # 1. Preprocess using metadata
        hidden_states = self._combine_preprocess(hidden_states, combine_metadata)

        # 2. AllToAll
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            combine_metadata.input_splits,
            combine_metadata.output_splits,
            self.ep_group,
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        # 3. Postprocess using metadata
        output = self._combine_postprocess(permutated_local_input_tokens, combine_metadata)

        return output

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        (
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        ) = self._preprocess(topk_ids)
        hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=num_out_tokens,
        )

        return (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        )

    def _preprocess(self, topk_ids: torch.Tensor):
        num_local_tokens_per_expert = torch.histc(topk_ids, bins=self.num_experts, min=0, max=self.num_experts)

        ep_size = self.ep_size
        num_out_tokens = topk_ids.numel()

        input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )

        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, group=self.ep_group
        ).reshape(ep_size, self.num_experts)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]
        if num_global_tokens_per_local_expert is None:
            raise ValueError("num_global_tokens_per_local_expert must be set before sum.")

        output_splits = (
            num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
        )
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)

        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            if num_global_tokens_per_local_expert is None:
                raise ValueError("num_global_tokens_per_local_expert must be set before operations.")
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
            )
        else:
            torch.npu.synchronize()

        return (
            num_tokens_per_local_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        )

    def _dispatch_postprocess(
        self, global_input_tokens, dynamic_scale_after_all2all, global_input_tokens_local_experts_indices, with_quant
    ):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, dynamic_scale_after_all2all, None

        # Handle quantized case
        if with_quant:
            assert global_input_tokens_local_experts_indices is not None, (
                "global_input_tokens_local_experts_indices must be provided"
            )
            dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
                dynamic_scale_after_all2all.unsqueeze(-1), global_input_tokens_local_experts_indices
            )
            dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(-1)

        # Non-quantized case
        global_input_tokens, reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens, global_input_tokens_local_experts_indices
        )
        return global_input_tokens, dynamic_scale_after_all2all, reversed_global_input_permutation_mapping

    def _combine_preprocess(
        self, hidden_states: torch.Tensor, combine_metadata: MoEAllToAllCombineMetadata
    ) -> torch.Tensor:
        # Unpermutation 2: expert output to AlltoAll input
        rev_global = combine_metadata.reversed_global_input_permutation_mapping
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1 and rev_global is not None:
            hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, rev_global)
        return hidden_states

    def _combine_postprocess(
        self,
        permutated_local_input_tokens: torch.Tensor,
        combine_metadata: MoEAllToAllCombineMetadata,
    ) -> torch.Tensor:
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=combine_metadata.reversed_local_input_permutation_mapping.to(torch.int32),
            probs=combine_metadata.topk_weights,
            restore_shape=combine_metadata.hidden_shape_before_permute,
        )
        output = output.view(combine_metadata.hidden_shape)
        return output
