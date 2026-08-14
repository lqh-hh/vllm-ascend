import vllm.envs as envs
from vllm.config.vllm import VllmConfig

from vllm_ascend.utils import is_310p

_original_validate_v2_model_runner = VllmConfig._validate_v2_model_runner


def _patched_use_v2_model_runner(self) -> bool:
    """Return VLLM_USE_V2_MODEL_RUNNER env directly.

    The upstream use_v2_model_runner gate-keeps the v2 runner with
    per-model architecture whitelists, Triton availability checks, and
    feature-support inspections. On Ascend the v2 runner is controlled
    purely by the VLLM_USE_V2_MODEL_RUNNER environment variable;
    model-compatibility decisions are deferred to the NPU runner itself.
    """
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2
    return False


_original_get_v2_model_runner_unsupported_features = (
    VllmConfig._get_v2_model_runner_unsupported_features
)

# Upstream EEP (elastic expert parallelism) does not support the V2 model
# runner, so it is listed as unsupported; Ascend's V2 runner supports
# elastic EP, so drop that entry here.
def _patched_get_v2_model_runner_unsupported_features(self) -> list[str]:
    unsupported = _original_get_v2_model_runner_unsupported_features(self)
    if "elastic expert parallelism" in unsupported:
        unsupported.remove("elastic expert parallelism")
    return unsupported


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)


def _patched_validate_v2_model_runner(self) -> None:
    if is_310p():
        return
    _original_validate_v2_model_runner(self)


VllmConfig._validate_v2_model_runner = _patched_validate_v2_model_runner
