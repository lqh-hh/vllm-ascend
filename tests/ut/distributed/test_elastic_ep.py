from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

from vllm_ascend.distributed.elastic_ep.elastic_execute import (
    AscendElasticEPScalingExecutor,
    _match_peer_parameters,
    setup_moe_comm_and_quant_method,
)
from vllm_ascend.distributed.elastic_ep.standby_state import _mc2_group_ranks


def _executor_and_worker():
    executor = object.__new__(AscendElasticEPScalingExecutor)
    parallel_config = SimpleNamespace(
        world_size=2,
        data_parallel_size=2,
        data_parallel_rank=0,
        data_parallel_master_ip="127.0.0.1",
        _coord_store_port=1234,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
    )
    executor.worker_ref = lambda: worker
    return executor, worker


def test_prepare_reconfiguration_adds_ascend_standby_group():
    executor, _ = _executor_and_worker()
    request = SimpleNamespace(
        new_data_parallel_size=3,
        new_data_parallel_master_ip="127.0.0.1",
        coord_store_port=1234,
    )

    with (
        patch.object(
            ElasticEPScalingExecutor,
            "prepare_reconfiguration",
        ) as upstream_prepare,
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.create_ascend_standby_groups"
        ) as create_ascend_groups,
    ):
        executor.prepare_reconfiguration(request, use_all2all=True)

    upstream_prepare.assert_called_once_with(request, True)
    create_ascend_groups.assert_called_once_with(
        new_dp_size=3,
        new_world_size_across_dp=6,
        master_ip="127.0.0.1",
        coord_store_port=1234,
    )


def test_receive_expert_mapping_preserves_new_upstream_return_type():
    executor, _ = _executor_and_worker()
    mapping = torch.tensor([[0, 1]])
    executor._setup_moe_comm_and_quant_method = MagicMock()

    with patch.object(
        ElasticEPScalingExecutor,
        "receive_expert_mapping",
        return_value=mapping,
    ):
        result = executor.receive_expert_mapping()

    assert result is mapping
    executor._setup_moe_comm_and_quant_method.assert_called_once_with()


def test_prepare_new_worker_uses_ascend_weight_transfer():
    executor, _ = _executor_and_worker()
    events = []

    with (
        patch.object(
            ElasticEPScalingExecutor,
            "prepare_new_worker",
            side_effect=lambda: events.append("upstream"),
        ) as upstream_prepare,
        patch.object(
            executor,
            "_use_ascend_transfer_impl",
            return_value=nullcontext(),
        ) as use_ascend_transfer,
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.create_ascend_standby_groups",
            side_effect=lambda **_: events.append("ascend_mc2"),
        ) as create_ascend_groups,
    ):
        executor.prepare_new_worker()

    use_ascend_transfer.assert_called_once_with()
    upstream_prepare.assert_called_once_with()
    create_ascend_groups.assert_called_once_with(
        new_dp_size=2,
        new_world_size_across_dp=4,
        master_ip="127.0.0.1",
        coord_store_port=1234,
    )
    assert events == ["upstream", "ascend_mc2"]


def test_new_worker_activates_ascend_mc2_before_commit():
    executor, _ = _executor_and_worker()
    events = []

    with (
        patch.object(
            executor,
            "_activate_ascend_standby_groups",
            side_effect=lambda: events.append("activate_mc2"),
        ) as activate_groups,
        patch.object(
            ElasticEPScalingExecutor,
            "commit_scale_up",
            side_effect=lambda _: events.append("upstream_commit"),
        ) as upstream_commit,
    ):
        executor.commit_scale_up(is_existing_worker=False)

    activate_groups.assert_called_once_with()
    upstream_commit.assert_called_once_with(False)
    assert events == ["activate_mc2", "upstream_commit"]


def test_switch_and_prepare_preserves_retired_groups():
    executor, worker = _executor_and_worker()
    worker.model_runner = MagicMock()
    worker.model_runner.model.modules.return_value = []
    retired_groups = (MagicMock(),)
    standby_mc2 = MagicMock()
    executor._setup_moe_comm_and_quant_method = MagicMock()

    with (
        patch.object(
            ElasticEPScalingExecutor,
            "switch_and_prepare",
            return_value=retired_groups,
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.pop_ascend_standby_groups",
            return_value={"mc2": standby_mc2},
        ),
        patch("vllm_ascend.distributed.elastic_ep.elastic_execute._replace_ascend_active_groups") as replace_groups,
    ):
        result = executor.switch_and_prepare()

    assert result is retired_groups
    replace_groups.assert_called_once_with(mc2=standby_mc2)


def test_setup_moe_comm_refreshes_deferred_quant_group_name():
    backend = MagicMock()
    backend.get_hccl_comm_name.return_value = "new-mc2-group"
    device_group = MagicMock()
    device_group._get_backend.return_value = backend
    mc2_group = SimpleNamespace(device_group=device_group, rank_in_group=3)
    quant_method = SimpleNamespace(moe_all_to_all_group_name="")
    module = SimpleNamespace(
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(quant_method=quant_method),
        ),
        moe_config=MagicMock(),
    )

    with (
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.get_mc2_group",
            return_value=mc2_group,
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.setup_moe_comm_method"
        ) as setup_moe_comm_method,
    ):
        setup_moe_comm_and_quant_method(module)

    assert quant_method.moe_all_to_all_group_name == "new-mc2-group"
    backend.get_hccl_comm_name.assert_called_once_with(3)
    setup_moe_comm_method.assert_called_once_with(module.moe_config)


def test_match_peer_parameters_is_deterministic_on_mismatch():
    parameters = [object(), object(), object()]

    sender = _match_peer_parameters(
        ["z", "a", "sender_only"],
        parameters,
        ["receiver_only", "a", "z"],
    )
    receiver_parameters = [object(), object(), object()]
    receiver = _match_peer_parameters(
        ["receiver_only", "a", "z"],
        receiver_parameters,
        ["z", "a", "sender_only"],
    )

    assert sender == [parameters[1], parameters[0]]
    assert receiver == [receiver_parameters[1], receiver_parameters[2]]


def test_mc2_standby_ranks_match_dp_ep_layout():
    assert _mc2_group_ranks(
        world_size=4,
        dp_size=2,
        pp_size=2,
        pcp_size=1,
        tp_size=1,
    ) == [[0, 2], [1, 3]]
