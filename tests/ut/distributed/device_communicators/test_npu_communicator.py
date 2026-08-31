from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.distributed.device_communicators.npu_communicator import (
    NPUCommunicator,
    _NpuAll2AllManager,
)


class TestNPUCommunicator(TestBase):
    @staticmethod
    def _stateless_cpu_group(rank: int = 0, world_size: int = 2):
        cpu_group = MagicMock()
        cpu_group.rank.return_value = rank
        cpu_group.size.return_value = world_size
        return cpu_group

    @patch("torch.npu.current_device", return_value=0)
    @patch("vllm_ascend.distributed.device_communicators.npu_communicator.PyHcclCommunicator")
    def test_stateless_group_initializes_pyhccl(self, pyhccl_cls, _):
        cpu_group = self._stateless_cpu_group()
        tcp_store_group = MagicMock()

        communicator = NPUCommunicator(
            cpu_group=cpu_group,
            global_ranks=[2, 3],
            global_world_size=4,
            tcp_store_group=tcp_store_group,
        )

        pyhccl_cls.assert_called_once_with(
            group=tcp_store_group,
            device=torch.device("npu:0"),
            warmup=False,
        )
        self.assertIs(communicator.pyhccl_comm, pyhccl_cls.return_value)

    @patch("torch.npu.current_device", return_value=3)
    @patch("vllm_ascend.distributed.device_communicators.npu_communicator.PyHcclCommunicator")
    def test_stateless_group_uses_current_worker_device(self, pyhccl_cls, _):
        tcp_store_group = MagicMock()

        communicator = NPUCommunicator(
            cpu_group=self._stateless_cpu_group(),
            device=torch.device("npu:0"),
            global_ranks=[0, 1],
            global_world_size=2,
            tcp_store_group=tcp_store_group,
        )

        pyhccl_cls.assert_called_once_with(
            group=tcp_store_group,
            device=torch.device("npu:3"),
            warmup=False,
        )
        self.assertEqual(communicator.device, torch.device("npu:3"))

    @patch("torch.npu.current_device", return_value=0)
    @patch("vllm_ascend.distributed.device_communicators.npu_communicator.PyHcclCommunicator")
    def test_destroy_releases_pyhccl(self, pyhccl_cls, _):
        communicator = NPUCommunicator(
            cpu_group=self._stateless_cpu_group(),
            global_ranks=[0, 1],
            global_world_size=2,
            tcp_store_group=MagicMock(),
        )

        communicator.destroy()

        pyhccl_cls.return_value.destroy.assert_called_once_with()
        self.assertIsNone(communicator.pyhccl_comm)

    def test_all_gather_uses_pyhccl_for_stateless_group(self):
        communicator = object.__new__(NPUCommunicator)
        communicator.world_size = 2
        communicator.pyhccl_comm = MagicMock(disabled=False)
        input_tensor = torch.tensor([[1, 2], [3, 4]])

        def all_gather(input_, output):
            output.copy_(torch.cat((input_, input_ + 10), dim=0))
            return output

        communicator.pyhccl_comm.all_gather.side_effect = all_gather

        result = communicator.all_gather(input_tensor, dim=-1)

        torch.testing.assert_close(
            result,
            torch.tensor([[1, 2, 11, 12], [3, 4, 13, 14]]),
        )

    def test_batch_isend_irecv_uses_pyhccl(self):
        communicator = object.__new__(NPUCommunicator)
        communicator.pyhccl_comm = MagicMock(disabled=False)
        p2p_ops = [MagicMock()]

        communicator.batch_isend_irecv(p2p_ops)

        communicator.pyhccl_comm.batch_isend_irecv.assert_called_once_with(p2p_ops)

    def test_batch_isend_irecv_forwards_explicit_stream(self):
        communicator = object.__new__(NPUCommunicator)
        communicator.pyhccl_comm = MagicMock(disabled=False)
        p2p_ops = [MagicMock()]
        stream = MagicMock()

        communicator.batch_isend_irecv(p2p_ops, stream=stream)

        communicator.pyhccl_comm.batch_isend_irecv.assert_called_once_with(
            p2p_ops,
            stream=stream,
        )

    def test_batch_isend_irecv_requires_pyhccl(self):
        communicator = object.__new__(NPUCommunicator)
        communicator.pyhccl_comm = None

        with self.assertRaisesRegex(ValueError, "No PyHccl communicator"):
            communicator.batch_isend_irecv([])

    def test_broadcast_uses_pyhccl_local_rank(self):
        communicator = object.__new__(NPUCommunicator)
        communicator.pyhccl_comm = MagicMock(disabled=False)
        tensor = MagicMock()

        result = communicator.broadcast(tensor, src=1)

        communicator.pyhccl_comm.broadcast.assert_called_once_with(tensor, 1)
        self.assertIs(result, tensor)

    def test_all2all_manager_builds_dense_rank_tables_in_place(self):
        manager = _NpuAll2AllManager(4, torch.device("cpu"))
        manager.set_num_physical_experts(12)
        manager.update_mask(1)

        self.assertEqual(manager.query_active_mask().tolist(), [0, 1, 0, 0])
        elastic_info = manager.get_elastic_info()
        self.assertEqual(
            elastic_info.tolist(),
            [1, 3, 0, 12, 0, -1, 1, 2, 0, 2, 3, -1],
        )

        data_ptr = elastic_info.data_ptr()
        manager.update_mask(2)
        self.assertEqual(manager.get_elastic_info().data_ptr(), data_ptr)
        self.assertEqual(
            manager.get_elastic_info().tolist(),
            [1, 2, 0, 12, 0, -1, -1, 1, 0, 3, -1, -1],
        )
