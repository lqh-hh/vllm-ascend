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
    """No-op all2all_manager for NPU. Used by vLLM main's fault-tolerance
    check (data_parallel_size > 1 and is_moe); NPU does not register a real
    one because it uses mc2 / all_gather for MoE communication.
    """

    @property
    def support_fault_tolerance(self) -> bool:
        return False

    def query_fault(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    def query_active_mask(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")


class NPUCommunicator(DeviceCommunicatorBase):
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
            use_all2all=use_all2all,
        )
        if tcp_store_group is not None:
            # StatelessGroupCoordinator passes its logical device index here.
            # Under Ray, the worker may already be bound to a different NPU,
            # especially when Ray does not rewrite ASCEND_RT_VISIBLE_DEVICES.
            # HCCL requires every communicator rank to use the device selected
            # by its worker, so prefer the active NPU for stateless groups.
            self.device = torch.device(f"npu:{torch.npu.current_device()}")
        else:
            self.device = device or torch.device(f"npu:{torch.npu.current_device()}")

        self.pyhccl_comm: PyHcclCommunicator | None = None
        if self.world_size > 1 and tcp_store_group is not None:
            self.pyhccl_comm = PyHcclCommunicator(
                group=tcp_store_group,
                device=self.device,
                warmup=False,
            )

        self.ca_comm = None
        self.all2all_manager = _NpuAll2AllManager()

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            return super().all_gather(input_, dim)

        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size,
            dtype=input_.dtype,
            device=input_.device,
        )
        pyhccl_comm.all_gather(input_.contiguous(), output_tensor)
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1 :])

    def destroy(self) -> None:
        if self.pyhccl_comm is not None:
            self.pyhccl_comm.destroy()
            self.pyhccl_comm = None

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            return super().broadcast(tensor, src)
        pyhccl_comm.broadcast(tensor, src)
        return tensor

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            return super().send(tensor, dst)
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        pyhccl_comm.send(tensor, dst)

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        src: int | None = None,
    ) -> torch.Tensor:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            return super().recv(size, dtype, src)
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pyhccl_comm.recv(tensor, src)
        return tensor

    def batch_isend_irecv(self, p2p_ops: list) -> None:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            raise ValueError("No PyHccl communicator found")
        pyhccl_comm.batch_isend_irecv(p2p_ops)
