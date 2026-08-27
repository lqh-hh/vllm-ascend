from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.patch.platform import patch_stateless_coordinator


def _world_state():
    return SimpleNamespace(
        pg_group_ranks={},
        pg_map={},
        pg_names={},
        pg_backend_config={},
        default_pg=None,
    )


def test_stateless_hccl_group_is_registered():
    process_group = MagicMock()
    process_group.size.return_value = 2
    process_group.group_name = "dp:0_device"
    process_group.get_group_store.return_value = MagicMock()
    world = _world_state()

    with (
        patch.object(
            patch_stateless_coordinator,
            "_orig_stateless_init",
            return_value=process_group,
        ),
        patch.object(patch_stateless_coordinator, "_world", world),
        patch.object(patch_stateless_coordinator, "BackendConfig", return_value="hccl-config"),
    ):
        result = patch_stateless_coordinator._ascend_stateless_init_pg(
            backend="hccl",
            return_store=False,
        )

    assert result is process_group
    assert world.pg_group_ranks[process_group] == {0: 0, 1: 1}
    assert world.pg_map[process_group] == ("hccl", process_group.get_group_store.return_value)
    assert world.pg_names[process_group] == "dp:0_device"
    assert world.pg_backend_config[process_group] == "hccl-config"
    assert world.default_pg is None


def test_stateless_non_hccl_group_is_not_registered():
    process_group = MagicMock()
    world = _world_state()

    with (
        patch.object(
            patch_stateless_coordinator,
            "_orig_stateless_init",
            return_value=process_group,
        ),
        patch.object(patch_stateless_coordinator, "_world", world),
    ):
        result = patch_stateless_coordinator._ascend_stateless_init_pg(
            backend="gloo",
            return_store=False,
        )

    assert result is process_group
    assert world.pg_map == {}


def test_destroy_removes_stateless_group_registration():
    process_group = MagicMock()
    world = _world_state()
    world.pg_group_ranks[process_group] = {0: 0}
    world.pg_map[process_group] = ("hccl", MagicMock())
    world.pg_names[process_group] = "dp:0_device"
    world.pg_backend_config[process_group] = "hccl-config"

    with (
        patch.object(patch_stateless_coordinator, "_orig_stateless_destroy") as original_destroy,
        patch.object(patch_stateless_coordinator, "_world", world),
    ):
        patch_stateless_coordinator._ascend_stateless_destroy_pg(process_group)

    original_destroy.assert_called_once_with(process_group)
    assert world.pg_group_ranks == {}
    assert world.pg_map == {}
    assert world.pg_names == {}
    assert world.pg_backend_config == {}
    assert world.default_pg is None


def test_stateless_coordinator_uses_current_npu():
    coordinator = SimpleNamespace()
    original_init = MagicMock()
    torch_mock = MagicMock()
    torch_mock.npu.current_device.return_value = 3
    current_device = MagicMock()
    torch_mock.device.return_value = current_device

    with (
        patch.object(
            patch_stateless_coordinator,
            "_orig_stateless_coordinator_init",
            original_init,
        ),
        patch.object(patch_stateless_coordinator, "torch", torch_mock),
    ):
        patch_stateless_coordinator._ascend_stateless_coordinator_init(
            coordinator,
            "group-ranks",
            local_rank=0,
        )

    original_init.assert_called_once_with(coordinator, "group-ranks", local_rank=0)
    torch_mock.npu.current_device.assert_called_once_with()
    torch_mock.device.assert_called_once_with("npu:3")
    assert coordinator.device_index == 3
    assert coordinator.device is current_device


def test_npu_tensor_broadcast_uses_device_communicator():
    coordinator = SimpleNamespace(device_communicator=MagicMock())
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
    coordinator.device_communicator.broadcast.return_value = tensor

    result = patch_stateless_coordinator._ascend_stateless_broadcast(
        coordinator,
        tensor,
        src=1,
    )

    coordinator.device_communicator.broadcast.assert_called_once_with(tensor, 1)
    assert result is tensor
