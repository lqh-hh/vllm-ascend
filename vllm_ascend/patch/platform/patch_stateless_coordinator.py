#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
# Patch vllm.distributed.stateless_coordinator: use NPUCommunicator and
# register HCCL stateless groups into torch's global ``_world``.

import torch
import vllm.distributed.stateless_coordinator as _stateless_coordinator
from torch.distributed import ProcessGroup, Store
from torch.distributed.distributed_c10d import BackendConfig, _world

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator

# Keep references to original functions for use inside the wrappers.
_orig_stateless_init = _stateless_coordinator.stateless_init_torch_distributed_process_group
_orig_stateless_destroy = _stateless_coordinator.stateless_destroy_torch_distributed_process_group
_orig_stateless_coordinator_init = _stateless_coordinator.StatelessGroupCoordinator.__init__
_orig_stateless_broadcast = _stateless_coordinator.StatelessGroupCoordinator.broadcast


# Register HCCL stateless groups into torch's ``_world`` on init so the
# standard torch.distributed APIs can find them.
def _ascend_stateless_init_pg(**kwargs) -> ProcessGroup | tuple[ProcessGroup, Store]:
    # Call the original helper first to create the stateless group.
    if kwargs.get("return_store", False):
        pg, store = _orig_stateless_init(**kwargs)
    else:
        pg = _orig_stateless_init(**kwargs)

    # HCCL groups are not registered by the upstream helper; register them
    # into torch's global ``_world`` state so torch.distributed APIs work.
    if kwargs["backend"] == "hccl":
        backend = "hccl"
        prefix_store = pg.get_group_store()
        group_name = pg.group_name
        backend_config = BackendConfig(backend)

        # Each rank of this stateless group maps 1:1 to itself (rank i in
        # the group is global rank i).
        _world.pg_group_ranks[pg] = {i: i for i in range(pg.size())}
        _world.pg_map[pg] = (backend, prefix_store)
        _world.pg_names[pg] = group_name
        _world.pg_backend_config[pg] = str(backend_config)

    if kwargs.get("return_store", False):
        return pg, store
    else:
        return pg


# Mirror the registration above: drop the group from ``_world`` on destroy.
def _ascend_stateless_destroy_pg(pg: ProcessGroup) -> None:
    _orig_stateless_destroy(pg)

    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)


def _ascend_stateless_coordinator_init(self, *args, **kwargs) -> None:
    """Bind stateless groups to the NPU selected for this worker.

    With Ray DP, every nested executor can have ``local_rank == 0`` while its
    worker is assigned a different physical NPU. Upstream copies the WORLD
    group's logical device index into each stateless group, so Elastic EP's
    warmup tensor would otherwise be created on physical NPU 0 for every rank.
    The Elastic EP async thread sets the worker device before group creation;
    use that active device for both the coordinator and its warmup collectives.
    """
    _orig_stateless_coordinator_init(self, *args, **kwargs)
    current_device = torch.npu.current_device()
    self.device_index = current_device
    self.device = torch.device(f"npu:{current_device}")


def _ascend_stateless_broadcast(self, input_, src: int = 0):
    """Use the NPU device communicator for NPU tensors.

    Upstream checks ``Tensor.is_cuda`` because Elastic EP is CUDA-only there.
    NPU tensors need the equivalent device path instead of TCP-store pickling.
    """
    if input_.device.type == "npu" and self.device_communicator is not None:
        return self.device_communicator.broadcast(input_, src)
    return _orig_stateless_broadcast(self, input_, src)


_stateless_coordinator.stateless_init_torch_distributed_process_group = _ascend_stateless_init_pg
_stateless_coordinator.stateless_destroy_torch_distributed_process_group = _ascend_stateless_destroy_pg
_stateless_coordinator.StatelessGroupCoordinator.__init__ = _ascend_stateless_coordinator_init
_stateless_coordinator.StatelessGroupCoordinator.broadcast = _ascend_stateless_broadcast
_stateless_coordinator.CudaCommunicator = NPUCommunicator
