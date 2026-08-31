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

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
from vllm.distributed.utils import StatelessProcessGroup

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


class _NpuAll2AllManager:
    """All2All-manager adapter for MC2 fault tolerance.

    Owns the dead-rank mask together with the `elastic_info` tensor the mask
    is encoded into: the MC2 dispatch/combine operators take `elastic_info`
    fresh on every call, so this tensor doubles as the mask (there is no
    kernel-side mask buffer like DeepEP/nixl-ep). The public interface mirrors
    the upstream All2AllManagerBase mask API so the upstream FT sentinel
    drives it unchanged; future Ascend operators with FT support are expected
    to follow the same shape.
    """

    # Unlike DeepEP/nixl-ep, the MC2 kernels neither detect faults nor set
    # the mask themselves on timeout — a dead peer surfaces as an aborted op
    # raising out of the forward, and the mask is only ever written host-side
    # by FT recovery (scale_down, plus the retry mask replay). A per-step
    # query_fault() could therefore never observe anything; reporting False
    # keeps the upstream runners from paying for that query every step.
    support_fault_tolerance = False

    def __init__(self, ep_world_size: int, device: torch.device | None = None) -> None:
        self._ep_world_size = ep_world_size
        if device is None:
            device = torch.device("npu", torch.npu.current_device())
        elif not isinstance(device, torch.device):
            device = torch.device("npu", device)
        self._device = device
        self._dead: set[int] = set()
        self._num_physical_experts: int = 0

        # elastic_info layout: [is_scaling_down, dense ep world size,
        #  shared_expert_rank_num, num_physical_experts] + table1(orig->dense)
        #  + table2(dense->orig).
        size = 4 + 2 * ep_world_size
        self._elastic_info_host = torch.zeros(size, dtype=torch.int32)
        # Allocate lazily on the first MC2 call. NPUCommunicator instances are
        # also created for non-EP groups, which never consume elastic_info.
        self._elastic_info: torch.Tensor | None = None

    def update_mask(self, rank: int, masked: bool = True) -> None:
        """Mark an EP rank dead/alive and rebuild elastic_info in place."""
        if masked:
            self._dead.add(rank)
        else:
            self._dead.discard(rank)
        self._rebuild_elastic_info()

    def query_active_mask(self) -> torch.Tensor:
        """Per-EP-rank mask (1=dead, 0=live) as a CPU tensor, matching the
        upstream mask-buffer convention.

        Built on CPU on purpose: this is called while a fault is being
        probed, when the NPU may be hung — any device op would fail.
        """
        mask = torch.zeros(self._ep_world_size, dtype=torch.int32)
        for rank in self._dead:
            mask[rank] = 1
        return mask

    def query_fault(self) -> torch.Tensor:
        # NPU counterpart of the upstream per-step fault check. Unlike
        # DeepEP/nixl-ep there is no in-kernel timeout that flips the mask,
        # so a fault can never be observed this way — always report no fault.
        # (Faults surface as aborted HCCL ops raising out of execute_model;
        # see support_fault_tolerance.)
        return torch.tensor(False)

    def clean_buffers(self) -> None:
        """No-op, kept for the upstream retry flow which calls it
        unconditionally.

        Unlike DeepEP/nixl-ep there is no kernel-side mask buffer or RDMA
        state to clean: the elastic_info tensor is passed fresh to every MC2
        call, so the mask itself is intentionally left untouched (it survives
        across recovery rounds via the replayed cumulative dead set).
        """

    def get_elastic_info(self) -> torch.Tensor:
        """The device elastic_info tensor for the next MC2 dispatch/combine."""
        if self._elastic_info is None:
            self._elastic_info = self._elastic_info_host.to(self._device)
        return self._elastic_info

    def set_num_physical_experts(self, num_physical_experts: int) -> None:
        """Shrink the expert-space width after scale-down and rebuild."""
        self._num_physical_experts = num_physical_experts
        self._rebuild_elastic_info()

    def _rebuild_elastic_info(self) -> None:
        """Rebuild elastic_info from the dead set into the existing device
        tensor (never reallocates, so captured graphs stay valid)."""

        world_size = self._ep_world_size
        alive = sorted(set(range(world_size)) - self._dead)
        table1 = torch.full((world_size,), -1, dtype=torch.int32)
        table1[alive] = torch.arange(len(alive), dtype=torch.int32)
        table2 = torch.full((world_size,), -1, dtype=torch.int32)
        table2[: len(alive)] = torch.tensor(alive, dtype=torch.int32)
        self._elastic_info_host.copy_(
            torch.cat(
                [torch.tensor([1, len(alive), 0, self._num_physical_experts], dtype=torch.int32), table1, table2]
            )
        )
        if self._elastic_info is not None:
            self._elastic_info.copy_(self._elastic_info_host, non_blocking=True)


class NPUCommunicator(DeviceCommunicatorBase):
    # main2main compat: `use_all2all` was added to upstream
    # DeviceCommunicatorBase.__init__() in vllm main after 0.26.0.
    # NPU does not support all2all (uses mc2 / all_gather for MoE),
    # so the parameter is only accepted for interface alignment.
    # Remove the version gate once 0.26.0 support is dropped.

    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            use_all2all,
        )
        self.device = torch.npu.current_device()

        self.pyhccl_comm: PyHcclCommunicator | None = None
        if self.world_size > 1 and tcp_store_group is not None:
            self.pyhccl_comm = PyHcclCommunicator(group=tcp_store_group, device=self.device, warmup=False)

        self.ca_comm = None
        self.all2all_manager = _NpuAll2AllManager(self.world_size, self.device)

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]
            output_list = []
            tensor_shape_base = input_list[self.rank].size()
            for i in range(self.world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))

        else:
            input_list = [t.contiguous() for t in torch.tensor_split(input_, self.world_size, scatter_dim)]
            output_list = [torch.empty_like(input_list[i]) for i in range(self.world_size)]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        output_tensor = torch.cat(output_list, dim=gather_dim).contiguous()
        return output_tensor

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.pyhccl_comm is not None:
            if dim < 0:
                # Convert negative dim to positive.
                dim += input_.dim()
            input_size = input_.size()
            # Use concat-style all-gather: stack-style has torch.compile
            # compatibility issues (pytorch/pytorch#138795).
            output_size = (input_size[0] * self.world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
            # All-gather.
            output_tensor = self.pyhccl_comm.all_gather(input_, output_tensor)
            # Reshape
            output_tensor = output_tensor.reshape((self.world_size,) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1 :]
            )
            return output_tensor
        else:
            return super().all_gather(input_, dim)

    def destroy(self):
        if self.pyhccl_comm is not None:
            self.pyhccl_comm.destroy()
            self.pyhccl_comm = None

    def batch_isend_irecv(self, p2p_ops: list):
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is not None and not pyhccl_comm.disabled:
            pyhccl_comm.batch_isend_irecv(p2p_ops)
        else:
            raise ValueError("No PyHccl communicator found")
