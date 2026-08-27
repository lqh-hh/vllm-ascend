import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.utils import StatelessProcessGroup

from tests.ut.base import TestBase
from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


class MockHcclLib:
    pass


class MockUniqueId:
    pass


class TestPyHcclCommunicator(TestBase):
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
    def test_world_size_1_return_early(self):
        comm = PyHcclCommunicator(
            group=StatelessProcessGroup(0, 1, None, None),
            device="npu:0",
        )
        self.assertTrue(comm.disabled)
        self.assertFalse(comm.available)

    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"})
    def test_load_hccl_fail(self):
        comm = PyHcclCommunicator(
            group=StatelessProcessGroup(0, 2, None, None), device="npu:0", library_path="/not/exist/path/libhccl.so"
        )
        self.assertTrue(comm.disabled)

    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.HCCLLibrary", MockHcclLib)
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.hcclUniqueId", MockUniqueId)
    @patch("torch.npu.device")
    @patch("vllm_ascend.utils.current_stream", return_value=MagicMock(npu_stream=5678))
    def test_stateless_group(self, *_):
        group = StatelessProcessGroup(rank=3, world_size=4, store=None)

        comm = PyHcclCommunicator(group=group, device=3)

        self.assertEqual(comm.rank, 3)
        self.assertEqual(comm.world_size, 4)

    @patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"})
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.HCCLLibrary", MockHcclLib)
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.hcclUniqueId", MockUniqueId)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="nccl")
    @patch("torch.distributed.Backend.HCCL", "hccl", create=True)
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.distributed.broadcast")
    @patch("torch.npu.device")
    @patch("vllm_ascend.utils.current_stream", return_value=MagicMock(npu_stream=1234))
    def test_multi_gpu_pg_torch(
        self,
        *_,
    ):
        fake_pg = MagicMock()
        comm = PyHcclCommunicator(group=fake_pg, device="npu:1")

        self.assertEqual(comm.rank, 1)
        self.assertEqual(comm.world_size, 2)
        self.assertFalse(comm.available)
        self.assertTrue(comm.disabled)

    @patch("vllm_ascend.distributed.device_communicators.pyhccl.current_stream")
    def test_all_gather_uses_hccl(self, current_stream):
        current_stream.return_value = MagicMock(npu_stream=1234)
        comm = object.__new__(PyHcclCommunicator)
        comm.available = True
        comm.disabled = False
        comm.device = torch.device("cpu")
        comm.comm = MagicMock()
        comm.hccl = MagicMock()
        input_tensor = torch.tensor([1, 2])
        output_tensor = torch.empty(4, dtype=input_tensor.dtype)

        result = comm.all_gather(input_tensor, output_tensor)

        self.assertIs(result, output_tensor)
        comm.hccl.hcclAllGather.assert_called_once()

    def test_batch_isend_irecv_dispatches_in_tag_order(self):
        comm = object.__new__(PyHcclCommunicator)
        comm.disabled = False
        comm.send = MagicMock()
        comm.recv = MagicMock()
        stream = MagicMock()
        comm._get_stream = MagicMock(return_value=stream)
        send_tensor = MagicMock()
        recv_tensor = MagicMock()
        ops = [
            SimpleNamespace(
                op=torch.distributed.isend,
                tensor=send_tensor,
                group_peer=1,
            ),
            SimpleNamespace(
                op=torch.distributed.irecv,
                tensor=recv_tensor,
                group_peer=0,
            ),
        ]

        comm.batch_isend_irecv(ops)

        comm.send.assert_called_once_with(send_tensor, 1, stream)
        comm.recv.assert_called_once_with(recv_tensor, 0, stream)

    @patch("torch.accelerator.device_index")
    def test_destroy_releases_hccl_handle(self, device_index):
        device_index.return_value.__enter__.return_value = None
        comm = object.__new__(PyHcclCommunicator)
        comm.available = True
        comm.disabled = False
        comm.device = torch.device("npu:0")
        comm.comm = MagicMock()
        comm.hccl = MagicMock()

        comm.destroy()

        comm.hccl.hcclCommDestroy.assert_called_once_with(comm.comm)
        self.assertFalse(comm.available)
        self.assertTrue(comm.disabled)
