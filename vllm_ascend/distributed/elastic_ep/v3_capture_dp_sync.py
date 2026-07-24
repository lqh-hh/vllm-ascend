import threading
import time

import torch
import torch.distributed as dist

from vllm_ascend.distributed.elastic_ep.standby_state import get_standby_v3_capture_dp_group

_ACTIVE = False
_EXPECTED = False
_SEQ = 0
_PREFIX = "v3_capture_dp_sync"
_COMPANION_THREAD: threading.Thread | None = None
_COMPANION_ERROR: BaseException | None = None
_OLD_ACTIVE_DP_SYNC_GROUP = None
_OLD_ACTIVE_DP_SYNC_RANK = 0
_OLD_ACTIVE_DP_SYNC_WORLD_SIZE = 0
_FORCE_V3_DURING_SCALE_UP = False
_SESSION = ""
# Shared CPU tensor layout: [entered_epoch, completed_epoch].
_DP_COLLECTIVE_STATE: torch.Tensor | None = None


def configure_dp_collective_state(state: torch.Tensor) -> None:
    global _DP_COLLECTIVE_STATE
    if (
        state.numel() != 2
        or state.dtype != torch.int64
        or state.device.type != "cpu"
    ):
        raise ValueError(
            "DP collective state must be a two-element CPU int64 tensor"
        )
    state.zero_()
    _DP_COLLECTIVE_STATE = state


def enter_dp_metadata_collective() -> int | None:
    state = _DP_COLLECTIVE_STATE
    if state is None:
        return None
    epoch = int(state[0].item()) + 1
    state[0] = epoch
    return epoch


def complete_dp_metadata_collective(
    epoch: int | None, synced_epoch: int | None = None
) -> None:
    if epoch is None:
        return
    state = _DP_COLLECTIVE_STATE
    if state is None:
        return
    completed_epoch = epoch if synced_epoch is None else synced_epoch
    state[0] = completed_epoch
    state[1] = completed_epoch


def _group():
    return get_standby_v3_capture_dp_group()


def _root_key(name: str) -> str:
    group = _group()
    world_size = getattr(group, "world_size", "unknown")
    return f"{_PREFIX}/world_size={world_size}/{name}"


def _key(name: str) -> str:
    session = _SESSION or "unknown"
    return f"{_root_key('session')}/{session}/{name}"


def configure_new_rank_capture_dp_sync(active: bool) -> None:
    global _ACTIVE, _EXPECTED, _SEQ, _SESSION
    _EXPECTED = active
    _SEQ = 0
    group = _group()
    if group is None:
        _ACTIVE = False
        _SESSION = ""
        if active:
            print("[Elastic EP scale-up] V3 new-rank capture DP sync requested but v3_capture_dp group is missing")
        return
    if active:
        _SESSION = str(time.time_ns())
        group.tcp_store_group.store.set(_root_key("active_session"), _SESSION.encode())
    elif _SESSION:
        group.tcp_store_group.store.set(_root_key("active_session"), b"")
    _ACTIVE = active
    group.tcp_store_group.store.set(_key("active"), b"1" if active else b"0")
    if not active:
        _SESSION = ""
    print(
        "[Elastic EP scale-up] V3 new-rank capture DP sync "
        f"{'enabled' if active else 'disabled'}: "
        f"rank={group.rank_in_group}/{group.world_size}"
    )


def is_capture_dp_sync_expected() -> bool:
    return _EXPECTED


def configure_force_v3_during_scale_up(active: bool) -> None:
    global _FORCE_V3_DURING_SCALE_UP
    if active == _FORCE_V3_DURING_SCALE_UP:
        return
    _FORCE_V3_DURING_SCALE_UP = active
    print(
        f"[Elastic EP scale-up] force V3 MoE routing {'enabled' if active else 'disabled'}",
        flush=True,
    )


def is_force_v3_during_scale_up_enabled() -> bool:
    return _FORCE_V3_DURING_SCALE_UP


def describe_capture_dp_sync_state() -> str:
    group = _group()
    if group is None:
        group_desc = "None"
    else:
        group_desc = (
            f"name={getattr(group, 'unique_name', 'unknown')}, "
            f"rank={group.rank_in_group}/{group.world_size}, "
            f"cpu_group={group.cpu_group}"
        )
    return f"expected={_EXPECTED}, active={_ACTIVE}, seq={_SEQ}, session={_SESSION or 'unknown'}, group={group_desc}"


def capture_dp_sync_group():
    if not _ACTIVE:
        return None
    group = _group()
    return None if group is None else group.cpu_group


def configure_old_active_dp_sync_group(
    cpu_group,
    rank: int,
    world_size: int,
) -> None:
    global _OLD_ACTIVE_DP_SYNC_GROUP
    global _OLD_ACTIVE_DP_SYNC_RANK
    global _OLD_ACTIVE_DP_SYNC_WORLD_SIZE
    _OLD_ACTIVE_DP_SYNC_GROUP = cpu_group
    _OLD_ACTIVE_DP_SYNC_RANK = rank
    _OLD_ACTIVE_DP_SYNC_WORLD_SIZE = world_size
    print(
        f"[Elastic EP scale-up] old-active DP metadata sync enabled: rank={rank}/{world_size}, group={cpu_group}",
        flush=True,
    )


def clear_old_active_dp_sync_group() -> None:
    global _OLD_ACTIVE_DP_SYNC_GROUP
    global _OLD_ACTIVE_DP_SYNC_RANK
    global _OLD_ACTIVE_DP_SYNC_WORLD_SIZE
    if _OLD_ACTIVE_DP_SYNC_GROUP is not None:
        print(
            "[Elastic EP scale-up] old-active DP metadata sync disabled",
            flush=True,
        )
    _OLD_ACTIVE_DP_SYNC_GROUP = None
    _OLD_ACTIVE_DP_SYNC_RANK = 0
    _OLD_ACTIVE_DP_SYNC_WORLD_SIZE = 0


def old_active_dp_sync_group():
    if _OLD_ACTIVE_DP_SYNC_GROUP is None:
        return None
    return (
        _OLD_ACTIVE_DP_SYNC_GROUP,
        _OLD_ACTIVE_DP_SYNC_RANK,
        _OLD_ACTIVE_DP_SYNC_WORLD_SIZE,
    )


def is_old_active_dp_sync_enabled() -> bool:
    return _OLD_ACTIVE_DP_SYNC_GROUP is not None


def notify_capture_dp_allreduce() -> int | None:
    global _SEQ
    if not _ACTIVE:
        return None
    group = _group()
    if group is None:
        return None
    seq = _SEQ
    group.tcp_store_group.store.set(_key(f"step/{seq}"), b"go")
    _SEQ += 1
    return seq


def finish_new_rank_capture_dp_sync() -> None:
    global _ACTIVE, _EXPECTED, _SESSION
    if not _ACTIVE:
        _EXPECTED = False
        _SESSION = ""
        return
    group = _group()
    if group is None:
        _ACTIVE = False
        _EXPECTED = False
        _SESSION = ""
        return
    group.tcp_store_group.store.set(_key("active"), b"0")
    group.tcp_store_group.store.set(_key(f"step/{_SEQ}"), b"done")
    group.tcp_store_group.store.set(_root_key("active_session"), b"")
    print(
        "[Elastic EP scale-up] V3 new-rank capture DP sync finished: "
        f"rank={group.rank_in_group}/{group.world_size}, steps={_SEQ}",
        flush=True,
    )
    _ACTIVE = False
    _EXPECTED = False
    _SESSION = ""


def run_capture_dp_sync_companion() -> bool:
    global _SESSION
    group = _group()
    if group is None:
        print("[Elastic EP scale-up] V3 capture DP companion cannot start: v3_capture_dp group is missing")
        return False

    print(
        "[Elastic EP scale-up] V3 capture DP companion waiting for active "
        f"marker: rank={group.rank_in_group}/{group.world_size}, "
        f"group={getattr(group, 'unique_name', 'unknown')}",
        flush=True,
    )
    while True:
        session = group.tcp_store_group.store.get(_root_key("active_session"))
        if session:
            _SESSION = session.decode()
            break
        time.sleep(0.05)
    active = group.tcp_store_group.store.get(_key("active"))
    if active != b"1":
        print(
            f"[Elastic EP scale-up] V3 capture DP companion skipped: rank={group.rank_in_group}/{group.world_size}",
            flush=True,
        )
        return False

    print(
        f"[Elastic EP scale-up] V3 capture DP companion active: rank={group.rank_in_group}/{group.world_size}",
        flush=True,
    )
    seq = 0
    tensor = torch.zeros(2, group.world_size, dtype=torch.int32, device="cpu")
    while True:
        step = group.tcp_store_group.store.get(_key(f"step/{seq}"))
        if step == b"done":
            break
        if step != b"go":
            raise RuntimeError(f"Unexpected V3 capture DP sync step marker: {step!r}")
        dist.all_reduce(tensor, group=group.cpu_group)
        tensor.zero_()
        seq += 1

    print(
        "[Elastic EP scale-up] V3 capture DP companion finished: "
        f"rank={group.rank_in_group}/{group.world_size}, steps={seq}",
        flush=True,
    )
    return True


def start_capture_dp_sync_companion_background() -> bool:
    global _COMPANION_ERROR, _COMPANION_THREAD

    if _COMPANION_THREAD is not None and _COMPANION_THREAD.is_alive():
        print(
            "[Elastic EP scale-up] V3 capture DP companion background already running",
            flush=True,
        )
        return True

    _COMPANION_ERROR = None

    def _run() -> None:
        global _COMPANION_ERROR
        try:
            run_capture_dp_sync_companion()
        except BaseException as exc:
            _COMPANION_ERROR = exc
            print(
                f"[Elastic EP scale-up] V3 capture DP companion background failed: {exc!r}",
                flush=True,
            )

    _COMPANION_THREAD = threading.Thread(
        target=_run,
        name="v3-capture-dp-sync-companion",
        daemon=True,
    )
    _COMPANION_THREAD.start()
    print(
        "[Elastic EP scale-up] V3 capture DP companion background started",
        flush=True,
    )
    return True


def is_capture_dp_sync_companion_done() -> bool:
    thread = _COMPANION_THREAD
    return thread is None or not thread.is_alive()


def raise_capture_dp_sync_companion_error_if_any() -> None:
    if _COMPANION_ERROR is not None:
        raise RuntimeError("V3 capture DP companion background failed") from (_COMPANION_ERROR)
