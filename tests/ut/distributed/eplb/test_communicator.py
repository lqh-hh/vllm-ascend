# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.eplb import eplb_communicator as upstream_communicator
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
    TorchDistGlooStagedEplbCommunicator,
    TorchDistNcclEplbCommunicator,
)

from vllm_ascend.distributed.eplb.communicator import (
    AscendGlooEplbCommunicator,
    HcclEplbCommunicator,
    PyHcclEplbCommunicator,
)


@pytest.fixture
def communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return AscendGlooEplbCommunicator(cpu_group=MagicMock())


def test_communicator_reuses_upstream_gloo_staging(communicator):
    assert isinstance(communicator, TorchDistGlooStagedEplbCommunicator)
    assert communicator.needs_profile_buffer_reservation is False


@pytest.fixture
def hccl_communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return HcclEplbCommunicator(MagicMock())


def test_hccl_communicator_reuses_upstream_transport(hccl_communicator):
    assert isinstance(hccl_communicator, TorchDistNcclEplbCommunicator)
    assert hccl_communicator.needs_profile_buffer_reservation is False


def test_hccl_send_and_recv_use_persistent_tensors(hccl_communicator, monkeypatch):
    monkeypatch.setattr(
        upstream_communicator,
        "P2POp",
        lambda op, tensor, rank, group: (op, tensor, rank, group),
    )
    send_tensor = torch.arange(2)
    recv_tensor = torch.zeros(2)

    hccl_communicator.add_send([send_tensor], dst_rank=1, expert_id=3)
    hccl_communicator.add_recv([recv_tensor], src_rank=1, expert_id=3)

    assert hccl_communicator._p2p_ops[0][1] is send_tensor
    assert hccl_communicator._p2p_ops[1][1] is recv_tensor


def test_hccl_execute_clears_queue_after_failure(hccl_communicator, monkeypatch):
    hccl_communicator._p2p_ops.append(object())
    monkeypatch.setattr(upstream_communicator.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        upstream_communicator,
        "batch_isend_irecv",
        MagicMock(side_effect=RuntimeError("transfer failed")),
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        hccl_communicator.execute()

    assert hccl_communicator._p2p_ops == []


@pytest.fixture
def pyhccl_communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return PyHcclEplbCommunicator(MagicMock(), stream=MagicMock())


def test_pyhccl_execute_orders_ops_by_expert_id(pyhccl_communicator):
    captured_ops = []
    pyhccl_communicator._pyhccl_comm.batch_isend_irecv.side_effect = (
        lambda ops, stream: captured_ops.extend(ops)
    )
    pyhccl_communicator.add_send([torch.ones(1)], dst_rank=1, expert_id=2)
    pyhccl_communicator.add_recv([torch.ones(1)], src_rank=1, expert_id=1)

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
