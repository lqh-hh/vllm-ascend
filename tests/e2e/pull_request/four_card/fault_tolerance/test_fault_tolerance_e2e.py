# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the fault-tolerance framework on Ascend NPU.

Requires 4 NPUs (DP=4); gated behind ``has_npu_ft_capability()``.
"""

import contextlib
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psutil
import pytest
import requests
import torch

from tests.e2e.conftest import RemoteOpenAIServer

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
DP_SIZE = 4

# Fault-detection timeout budget:
# - CPU: Gloo DP allreduce timeout (30s) detects the dead peer.
# - NPU: HCSP operator timeout detects the dead peer.
# - Deadline (45s): slowest fallback (30s) + margin.
CPU_DISTRIBUTED_TIMEOUT_S = 30
FT_COMMUNICATION_OPS_ABORT_TIMEOUT_MS = 15_000
FAULT_DETECTION_DEADLINE_S = 45
SCALE_DOWN_DEADLINE_S = 180
NUM_REDUNDANT_EXPERTS = 128


# ---------------------------------------------------------------------------
# Fault-injection via sitecustomize.py
# ---------------------------------------------------------------------------
# Patches ``NPUModelRunner._sync_metadata_across_dp`` to raise on ``rank`` at
# a chosen step. Gated on VLLM_FT_TEST_INJECT_FAULT.
#
# The import hook waits for ``vllm_ascend.worker.model_runner_v1`` to land
# in sys.modules and for ``NPUModelRunner._sync_metadata_across_dp`` to be
# defined, then wraps the method with a step-counting wrapper.
_FAULT_INJECT_SITECUSTOMIZE = """\
import builtins
import os
import sys

_SPEC = os.environ.get("VLLM_FT_TEST_INJECT_FAULT")
_MODULE = "vllm_ascend.worker.model_runner_v1"
_CLASS = "NPUModelRunner"
_METHOD = "_sync_metadata_across_dp"

if _SPEC:
    _f = dict(kv.split("=", 1) for kv in _SPEC.split(","))
    _RANK, _STEP = int(_f["rank"]), int(_f["step"])
    _steps = [0]

    def _patch(m):
        cls = getattr(m, _CLASS)
        _orig = getattr(cls, _METHOD)

        def _wrapped(self, *args, **kwargs):
            result = _orig(self, *args, **kwargs)
            if self.dp_rank == _RANK:
                _steps[0] += 1
                if _steps[0] == _STEP:
                    raise RuntimeError(
                        "FT test fault injection (rank=%d step=%d)"
                        % (_RANK, _STEP)
                    )
            return result

        setattr(cls, _METHOD, _wrapped)

    _real_import = builtins.__import__

    def _hook(name, *a, **k):
        module = _real_import(name, *a, **k)
        m = sys.modules.get(_MODULE)
        if (
            m is not None
            and hasattr(m, _CLASS)
            and not getattr(m, "_ft_patched", False)
        ):
            cls = getattr(m, _CLASS)
            if hasattr(cls, _METHOD):
                m._ft_patched = True
                _patch(m)
        return module

    builtins.__import__ = _hook
"""


def _install_fault_injection(monkeypatch, tmp_path, rank: int, step: int) -> None:
    """Arrange for the DP-sync method to raise on ``rank`` at serving ``step``.

    Writes a ``sitecustomize.py`` and prepends its dir to PYTHONPATH so every
    vLLM subprocess picks it up; the fault spec is read from the environment.
    """
    site_dir = tmp_path / "ft_inject"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(_FAULT_INJECT_SITECUSTOMIZE)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(site_dir) + (os.pathsep + existing if existing else ""),
    )
    monkeypatch.setenv("VLLM_FT_TEST_INJECT_FAULT", f"rank={rank},step={step}")


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def _ft_server_args() -> list[str]:
    return [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "16",
        "--gpu-memory-utilization",
        "0.7",
        "--enable-expert-parallel",
        "--enable-eplb",
        "--eplb-config.num-redundant-experts",
        str(NUM_REDUNDANT_EXPERTS),
        "--enable-fault-tolerance",
        "--cpu-distributed-timeout-seconds",
        str(CPU_DISTRIBUTED_TIMEOUT_S),
        "--fault-tolerance-config",
        '{"engine_recovery_timeout_sec": 120}',
        "--additional-config",
        f'{{"ft_communication_ops_abort_timeout_ms": {FT_COMMUNICATION_OPS_ABORT_TIMEOUT_MS}}}',
    ]


class FTServerManager:
    """Manages DP=4 vLLM server instances for fault-tolerance testing.

    Starts one process per DP rank with fixed ports (8000 + rank).
    """

    def __init__(
        self,
        model_name: str,
        dp_size: int,
        base_server_args: list[str],
        tp_size: int = 1,
    ):
        self.model_name = model_name
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.base_server_args = base_server_args
        self.servers: list[tuple[RemoteOpenAIServer, list[str]]] = []
        self.server_threads: list[threading.Thread] = []

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        for rank in range(self.dp_size):
            server_args = self.base_server_args.copy()
            server_args.extend(
                [
                    "--data-parallel-size",
                    str(self.dp_size),
                    "--data-parallel-rank",
                    str(rank),
                    "--data-parallel-size-local",
                    "1",
                    "--tensor-parallel-size",
                    str(self.tp_size),
                    "--port",
                    str(8000 + rank),
                    "--api-server-count",
                    "1",
                ]
            )

            def start_server(r: int, sargs: list[str]) -> None:
                try:
                    server = RemoteOpenAIServer(
                        self.model_name,
                        sargs,
                        server_host="localhost",
                        server_port=8000 + r,
                        auto_port=False,
                        env_dict={"ASCEND_RT_VISIBLE_DEVICES": str(r)},
                    )
                    self.servers.append((server, sargs))
                except Exception:
                    print(f"Failed to start server rank {r}")
                    raise

            thread = threading.Thread(target=start_server, args=(rank, server_args))
            thread.start()
            self.server_threads.append(thread)

        for thread in self.server_threads:
            thread.join()

        if len(self.servers) != self.dp_size:
            raise RuntimeError(f"Only {len(self.servers)}/{self.dp_size} servers started")

        return self.servers

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for server, _ in reversed(self.servers):
            with contextlib.suppress(Exception):
                server.__exit__(None, None, None)
        self.servers.clear()


def _ft_manager() -> FTServerManager:
    return FTServerManager(
        MODEL_NAME,
        DP_SIZE,
        base_server_args=_ft_server_args(),
        tp_size=1,
    )


def _server_for_rank(servers: list[tuple[RemoteOpenAIServer, list[str]]], rank: int) -> RemoteOpenAIServer:
    """Locate the server for a DP rank."""
    for server, sargs in servers:
        if "--data-parallel-rank" in sargs:
            idx = sargs.index("--data-parallel-rank")
            if int(sargs[idx + 1]) == rank:
                return server
    raise AssertionError(f"no server found for DP rank {rank}")


# ---------------------------------------------------------------------------
# Test primitives
# ---------------------------------------------------------------------------


def _complete(client) -> Any:
    """Issue one completion request; used to drive the serving loop."""
    return client.completions.create(
        model=MODEL_NAME,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=0.0,
        timeout=10.0,
    )


def _in_parallel(fn, servers) -> list[Any]:
    """Run ``fn(server)`` for all servers concurrently; return in order."""
    with ThreadPoolExecutor(max_workers=len(servers)) as ex:
        return list(ex.map(fn, servers))


def _get_ft_status(server: RemoteOpenAIServer) -> dict:
    resp = requests.get(server.url_for("fault_tolerance/status"), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _apply_ft(
    server: RemoteOpenAIServer,
    instruction: str,
    params: dict | None = None,
    request_id: str = "",
) -> dict:
    """POST an FT instruction; assert it is accepted (202) and return body."""
    resp = requests.post(
        server.url_for("fault_tolerance/apply"),
        json={
            "instruction": instruction,
            "params": params or {},
            "request_id": request_id,
        },
        timeout=10,
    )
    assert resp.status_code == 202, resp.text
    return resp.json()


def _assert_serving_and_healthy(
    servers: tuple[RemoteOpenAIServer, ...],
    deadline_s: int = FAULT_DETECTION_DEADLINE_S,
) -> None:
    """Wait until every engine is healthy, then serve one request per server."""
    healthy = _wait_for_engines(
        list(servers),
        match_key="status",
        match_values={"healthy"},
        deadline_s=deadline_s,
    )
    assert all(healthy), healthy
    _in_parallel(lambda s: _complete(s.get_client()), servers)


def _kill_worker_process(server: RemoteOpenAIServer) -> None:
    """SIGKILL only the worker proc, leaving EngineCore and API server alive."""
    workers = [
        process
        for process in psutil.Process(server.proc.pid).children(recursive=True)
        if "Worker" in " ".join(process.cmdline())
    ]
    assert len(workers) == 1, f"expected 1 worker proc, found: {workers}"
    workers[0].kill()


def _wait_for_engines(
    servers: list[RemoteOpenAIServer],
    match_key: str,
    match_values: set[str],
    deadline_s: int = FAULT_DETECTION_DEADLINE_S,
) -> list[dict[str, Any] | None]:
    """Poll ``/fault_tolerance/status`` until each server's engine status matches.

    A server matches when its engine-status dict has ``match_key`` equal to
    one of ``match_values``. Returns one engine-status dict per server.
    Servers still unmatched after ``deadline_s`` get None.
    """
    results: dict[int, dict[str, Any]] = {}
    pending = dict(enumerate(servers))
    start = time.time()
    while pending and time.time() - start < deadline_s:
        for i, server in list(pending.items()):
            with contextlib.suppress(Exception):
                for engine_status in _get_ft_status(server)["engines"]:
                    if engine_status.get(match_key) in match_values:
                        results[i] = engine_status
                        del pending[i]
                        break
        if pending:
            time.sleep(1.0)
    return [results.get(i) for i in range(len(servers))]


@contextlib.contextmanager
def _driving(*servers: RemoteOpenAIServer):
    """Pump completions at each server in the background for the block's duration.

    Keeps every engine stepping into its failed component so a fault surfaces.
    Errors are expected once faulted and are ignored.
    """
    stop = threading.Event()

    def _drive(server):
        client = server.get_client()
        while not stop.is_set():
            with contextlib.suppress(Exception):
                _complete(client)
            time.sleep(0.2)

    threads = [threading.Thread(target=_drive, args=(s,), daemon=True) for s in servers]
    for t in threads:
        t.start()
    try:
        yield
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)


def _wait_for_ft_apply_outcome(server: RemoteOpenAIServer, request_id: str, deadline_s: int) -> str | None:
    """Wait until ``/fault_tolerance/status`` records the FT apply outcome."""
    engine_status = _wait_for_engines(
        [server],
        match_key="last_ft_request_id",
        match_values={request_id},
        deadline_s=deadline_s,
    )[0]
    return engine_status.get("ft_error") if engine_status else None


# ---------------------------------------------------------------------------
# Feature guard
# ---------------------------------------------------------------------------


def has_npu_ft_capability() -> bool:
    """Require at least 4 visible NPUs for DP=4 fault-tolerance tests."""
    if not torch.npu.is_available():
        return False
    try:
        return torch.npu.device_count() >= DP_SIZE
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not has_npu_ft_capability(),
    reason="Requires at least 4 NPUs for DP=4 fault-tolerance testing",
)
def test_injected_fault_retry_recovers_all_ranks(monkeypatch, tmp_path):
    """An exception injected into _sync_metadata_across_dp drives full
    retry recovery on all 4 DP ranks.

    Inject a fault at a chosen step on rank 3:

    - Rank 3 raises after allreduce and goes UNHEALTHY.
    - Ranks 0, 1, 2 detect the now-absent peer via the Gloo DP allreduce
      timeout and also go UNHEALTHY.

    All 4 being UNHEALTHY is the precondition for ``retry``.  The fault
    is patched via a generated ``sitecustomize.py``.
    """
    fault_step = int(os.getenv("FT_FAULT_STEP", "50"))
    _install_fault_injection(monkeypatch, tmp_path, rank=3, step=fault_step)

    with _ft_manager() as servers:
        assert len(servers) == DP_SIZE
        all_ranks = tuple(_server_for_rank(servers, r) for r in range(DP_SIZE))

        # 1. All engines healthy and serving.
        _assert_serving_and_healthy(all_ranks)

        # 2. Drive all ranks so rank 3 accumulates steps and trips the
        #    injected fault; ranks 0,1,2 then time out on DP allreduce.
        with _driving(*all_ranks):
            faulted = _wait_for_engines(
                list(all_ranks),
                match_key="status",
                match_values={"unhealthy"},
            )

        for rank, engine_status in enumerate(faulted):
            assert engine_status is not None, (
                f"rank {rank} did not report UNHEALTHY within {FAULT_DETECTION_DEADLINE_S}s -- it likely hung"
            )

        # The rank that raised carries the fault info from its own exception.
        assert faulted[3] is not None
        assert faulted[3].get("fault_info"), faulted[3]

        # 3. retry all engines.
        retry_request_id = str(uuid.uuid4())
        for server in all_ranks:
            _apply_ft(server, "retry", request_id=retry_request_id)

        # 4. Recovery completes: all engines return to healthy and serve again.
        _assert_serving_and_healthy(all_ranks)


@pytest.mark.skipif(
    not has_npu_ft_capability(),
    reason="Requires at least 4 NPUs for DP=4 fault-tolerance testing",
)
def test_worker_kill_scale_down_recovers_survivors():
    """SIGKILL a middle Worker and keep serving after MC2 scale-down.

    Killing only rank 1's worker leaves all EngineCores alive, so the same
    fault is seen two ways:

    - Survivors (ranks 0, 2, 3): detect the dead peer via Gloo DP allreduce
      / HCSP timeout. Their own executor is fine, so ``on_fault`` marks them
      UNHEALTHY with a ``fault_info``.
    - Victim (rank 1): detects its own executor failure and marks itself DEAD.

    The DEAD engine rejects retry. The three survivors then receive one shared
    scale-down request, redistribute the victim's experts, densify MC2 routing,
    and must complete inference afterwards. Removing a middle rank exercises
    the non-identity orig-to-dense rank mapping.
    """
    with _ft_manager() as servers:
        assert len(servers) == DP_SIZE
        servers_by_rank = {rank: _server_for_rank(servers, rank) for rank in range(DP_SIZE)}
        victim_rank = 1
        victim = servers_by_rank[victim_rank]
        survivor_ranks = [rank for rank in range(DP_SIZE) if rank != victim_rank]
        survivors = tuple(servers_by_rank[rank] for rank in survivor_ranks)
        all_ranks = tuple(servers_by_rank[rank] for rank in range(DP_SIZE))

        # 1. Confirm all engines are healthy and serving.
        _assert_serving_and_healthy(all_ranks)

        # 2. Kill only the victim's worker; all EngineCores stay alive.
        _kill_worker_process(victim)

        # 3. Drive all engines so each keeps stepping into the failed component.
        with _driving(*all_ranks):
            faulted_results = _wait_for_engines(
                list(all_ranks),
                match_key="status",
                match_values={"dead", "unhealthy"},
            )

        # Survivors must report the peer fault as UNHEALTHY.
        for rank in survivor_ranks:
            result = faulted_results[rank]
            assert result is not None, (
                f"rank {rank} did not report the peer fault within "
                f"{FAULT_DETECTION_DEADLINE_S}s -- it likely hung"
            )
            assert result["status"] == "unhealthy", result
            assert result.get("fault_info"), result

        # Victim must report DEAD (its own worker is gone).
        victim_faulted = faulted_results[victim_rank]
        assert victim_faulted is not None, (
            f"victim (rank {victim_rank}) did not report its worker's death "
            f"within {FAULT_DETECTION_DEADLINE_S}s"
        )
        assert victim_faulted["status"] == "dead", victim_faulted

        # 4. retry is accepted at the HTTP layer (202 = background dispatch)...
        rejected_request_id = str(uuid.uuid4())
        request_id = _apply_ft(
            victim,
            "retry",
            request_id=rejected_request_id,
        )["request_id"]

        # 5. ...but the DEAD engine must reject it: recovery requires UNHEALTHY.
        ft_error = _wait_for_ft_apply_outcome(victim, request_id, FAULT_DETECTION_DEADLINE_S)
        assert ft_error is not None, "rejection was never recorded in /fault_tolerance/status"
        assert "status is DEAD" in ft_error, ft_error

        # 6. All survivors use the same request id so their rebuilt Gloo groups
        #    coordinate on the same store keys.
        scale_down_request_id = str(uuid.uuid4())
        for server in survivors:
            _apply_ft(
                server,
                "scale_down",
                {"removed_dp_ranks": [victim_rank]},
                request_id=scale_down_request_id,
            )

        # 7. Expert reload, MC2 densification and a real post-recovery request
        #    must all succeed on every survivor.
        _assert_serving_and_healthy(
            survivors,
            deadline_s=SCALE_DOWN_DEADLINE_S,
        )
