from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.patch.platform import patch_use_v2_model_runner


@patch.object(patch_use_v2_model_runner, "is_310p", return_value=False)
def test_validate_v2_temporarily_hides_elastic_ep(_):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_elastic_ep=True),
    )

    def validate(vllm_config):
        assert vllm_config.parallel_config.enable_elastic_ep is False

    with patch.object(
        patch_use_v2_model_runner,
        "_original_validate_v2_model_runner",
        side_effect=validate,
    ) as original_validate:
        patch_use_v2_model_runner._patched_validate_v2_model_runner(config)

    original_validate.assert_called_once_with(config)
    assert config.parallel_config.enable_elastic_ep is True


@patch.object(patch_use_v2_model_runner, "is_310p", return_value=False)
def test_validate_v2_restores_elastic_ep_after_failure(_):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_elastic_ep=True),
    )

    with (
        patch.object(
            patch_use_v2_model_runner,
            "_original_validate_v2_model_runner",
            side_effect=RuntimeError("validation failed"),
        ),
        pytest.raises(RuntimeError, match="validation failed"),
    ):
        patch_use_v2_model_runner._patched_validate_v2_model_runner(config)

    assert config.parallel_config.enable_elastic_ep is True


@patch.object(patch_use_v2_model_runner, "is_310p", return_value=False)
def test_validate_v2_preserves_other_upstream_validation(_):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_elastic_ep=False),
    )
    original_validate = MagicMock()

    with patch.object(
        patch_use_v2_model_runner,
        "_original_validate_v2_model_runner",
        original_validate,
    ):
        patch_use_v2_model_runner._patched_validate_v2_model_runner(config)

    original_validate.assert_called_once_with(config)
