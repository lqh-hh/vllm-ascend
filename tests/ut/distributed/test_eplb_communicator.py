# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.eplb import eplb_communicator as upstream_communicator
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
    TorchDistNcclEplbCommunicator,
)

from vllm_ascend.distributed.eplb_communicator import (
    HcclEplbCommunicator,
    PyHcclEplbCommunicator,
)


@pytest.fixture
def communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return HcclEplbCommunicator(MagicMock())


def test_communicator_reuses_upstream_torch_distributed_transport(communicator):
    assert isinstance(communicator, TorchDistNcclEplbCommunicator)
    assert communicator.needs_profile_buffer_reservation is False


def test_send_and_recv_use_persistent_expert_tensors_directly(communicator, monkeypatch):
    monkeypatch.setattr(
        upstream_communicator,
        "P2POp",
        lambda op, tensor, rank, group: (op, tensor, rank, group),
    )
    send_tensor = torch.arange(2)
    recv_tensor = torch.zeros(2)

    communicator.add_send([send_tensor], dst_rank=1, expert_id=3)
    communicator.add_recv([recv_tensor], src_rank=1, expert_id=3)

    assert communicator._p2p_ops[0][1] is send_tensor
    assert communicator._p2p_ops[1][1] is recv_tensor
    assert all(op[1].storage_offset() == 0 for op in communicator._p2p_ops)


def test_set_stream_is_ready_for_async_transfer(communicator):
    stream = object()

    communicator.set_stream(stream)

    assert communicator._cuda_stream is stream


def test_execute_clears_queue_after_failure(communicator, monkeypatch):
    communicator._p2p_ops.append(object())
    monkeypatch.setattr(
        upstream_communicator.torch.cuda,
        "stream",
        lambda stream: nullcontext(),
    )
    monkeypatch.setattr(
        upstream_communicator,
        "batch_isend_irecv",
        MagicMock(side_effect=RuntimeError("transfer failed")),
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        communicator.execute()

    assert communicator._p2p_ops == []


@pytest.fixture
def pyhccl_communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return PyHcclEplbCommunicator(MagicMock(), stream=MagicMock())


def test_pyhccl_execute_orders_ops_by_expert_id(pyhccl_communicator):
    first_tensor = torch.ones(1)
    second_tensor = torch.ones(1)
    captured_ops = []
    pyhccl_communicator._pyhccl_comm.batch_isend_irecv.side_effect = lambda ops, stream: captured_ops.extend(ops)
    pyhccl_communicator.add_send([first_tensor], dst_rank=1, expert_id=2)
    pyhccl_communicator.add_recv([second_tensor], src_rank=1, expert_id=1)

    pyhccl_communicator.execute()

    assert [op.tag for op in captured_ops] == [1, 2]
    assert pyhccl_communicator._p2p_ops == []


def test_pyhccl_set_stream_updates_transfer_stream(pyhccl_communicator):
    stream = MagicMock()

    pyhccl_communicator.set_stream(stream)
    pyhccl_communicator.add_send([torch.ones(1)], dst_rank=1, expert_id=1)
    pyhccl_communicator.execute()

    pyhccl_communicator._pyhccl_comm.batch_isend_irecv.assert_called_once()
    assert pyhccl_communicator._pyhccl_comm.batch_isend_irecv.call_args.args[1] is stream


def test_pyhccl_execute_clears_queue_after_failure(pyhccl_communicator):
    pyhccl_communicator.add_send([torch.ones(1)], dst_rank=1, expert_id=1)
    pyhccl_communicator._pyhccl_comm.batch_isend_irecv.side_effect = RuntimeError("transfer failed")

    with pytest.raises(RuntimeError, match="transfer failed"):
        pyhccl_communicator.execute()

    assert pyhccl_communicator._p2p_ops == []
