# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""
End-to-end Elastic EP scaling tests for vllm-ascend.

Launches a vLLM serve instance with Elastic EP enabled and performs a
scale-down (dp=4 -> dp=3) followed by a scale-up (dp=3 -> dp=4), validating
that each reconfiguration completes successfully. Accuracy is intentionally
not evaluated during scaling.
"""

import os
import subprocess
import time
from dataclasses import dataclass, field

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

# ---------------------------------------------------------------------------
# Server / model constants
# ---------------------------------------------------------------------------

QWEN3_30B_A3B_MODEL = "Qwen/Qwen3-30B-A3B"
"""HuggingFace / ModelScope identifier of the MoE model used for testing."""

MAX_MODEL_LEN = 16384
MAX_NUM_SEQS = 16

# How long (seconds) to wait after a scale request before continuing,
# giving the Elastic EP state machine time to finish reconfiguration.
_SCALE_DELAY_SECONDS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_ray_between_tests():
    """Force-stop any lingering Ray processes between tests."""
    subprocess.run(["ray", "stop", "--force"], timeout=30, capture_output=True)
    time.sleep(5)

    env_dict = _make_env_dict()
    for key, value in env_dict.items():
        os.environ[key] = value

    subprocess.run(["ray", "start", "--head"], timeout=30, capture_output=True)
    time.sleep(5)
    yield


def _send_scale_command(server, new_dp_size: int) -> bool:
    """POST a scale request to the server's Elastic EP endpoint.

    Returns ``True`` on HTTP 200, ``False`` otherwise.
    """
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=1000)
        code = response.status_code
        if code != 200:
            print(f"[scale] HTTP {code}: {response.text}")
        return code == 200
    except requests.exceptions.RequestException as exc:
        print(f"[scale] Request failed: {exc}")
        return False


def _make_env_dict() -> dict[str, str]:
    """Return the common environment-variable overrides for the server."""
    env = {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "2048",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }
    if os.environ.get("VLLM_USE_MODELSCOPE", "").lower() not in ("", "0", "false"):
        env["VLLM_USE_MODELSCOPE"] = "true"
    return env


# ---------------------------------------------------------------------------
# Configuration dataclasses and test runner
# ---------------------------------------------------------------------------


@dataclass
class ScaleSequence:
    """Defines a sequence of scaling operations."""

    name: str
    steps: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class ElasticEPTestConfig:
    """Configuration for an Elastic EP test."""

    name: str
    data_parallel_size: int
    data_parallel_size_local: int
    tensor_parallel_size: int
    gpu_memory_utilization: float = 0.9
    num_redundant_experts: int = 0
    scale_sequence: ScaleSequence = field(
        default_factory=lambda: ScaleSequence(
            name="4_to_3_to_4",
            steps=[
                (3, "Scale down (dp=4 -> dp=3)"),
                (4, "Scale up (dp=3 -> dp=4)"),
            ],
        )
    )


CONFIG_QWEN3_30B_DEFAULT = ElasticEPTestConfig(
    name="Qwen3-30B-A3B, Default Graph",
    data_parallel_size=4,
    data_parallel_size_local=4,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    num_redundant_experts=128,
)


def _build_vllm_args(config: ElasticEPTestConfig) -> list[str]:
    """Build vLLM server arguments from configuration."""
    args = [
        "--host",
        "0.0.0.0",
        "--port",
        str(get_open_port()),
        "--trust-remote-code",
        "--data-parallel-size",
        str(config.data_parallel_size),
        "--data-parallel-size-local",
        str(config.data_parallel_size_local),
        "--data-parallel-backend",
        "ray",
        "--enable-eplb",
        "--eplb-config.log_balancedness",
        "true",
        "--eplb-config.log_balancedness_interval",
        "10",
        "--eplb-config.use_async",
        "false",
        "--enable-elastic-ep",
        "--enable-expert-parallel",
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
    ]

    args.extend(["--eplb-config.num_redundant_experts", str(config.num_redundant_experts)])

    return args


def _run_elastic_ep_test(config: ElasticEPTestConfig, model_name: str) -> None:
    """Run the scaling sequence without accuracy evaluation.

    Args:
        config: The Elastic EP test configuration to use.
        model_name: Identifier of the model to serve.
    """
    vllm_serve_args = _build_vllm_args(config)
    env_dict = _make_env_dict()

    # Extract port from args (last port specification)
    port_index = vllm_serve_args.index("--port") + 1
    server_port = int(vllm_serve_args[port_index])

    with RemoteOpenAIServer(
        model_name,
        vllm_serve_args,
        server_host="127.0.0.1",
        server_port=server_port,
        env_dict=env_dict,
        auto_port=False,
        max_wait_seconds=1800,
    ) as server:
        print(f"Server started on port {server.port}")

        # Run the scaling steps without accuracy evaluation
        for new_dp_size, stage_description in config.scale_sequence.steps:
            assert _send_scale_command(server, new_dp_size), f"{stage_description} failed"
            time.sleep(_SCALE_DELAY_SECONDS)
            print(f"  {stage_description} completed")


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_elastic_ep_scaling_qwen3_30b() -> None:
    """Scale dp 4 -> 3 -> 4 (tp=1) with Default Graph, no accuracy check."""
    _run_elastic_ep_test(CONFIG_QWEN3_30B_DEFAULT, QWEN3_30B_A3B_MODEL)
