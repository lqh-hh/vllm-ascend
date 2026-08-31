from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.quantization.methods.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.worker.sentinel.eplb_redistribute import (
    build_orig_to_dense_rank_table,
    densify_routing_table_physical_ids,
    validate_expert_reload_support,
)


def test_densify_routing_table_with_middle_rank_removed():
    orig_to_dense = build_orig_to_dense_rank_table(4, {1})
    routing_table = torch.tensor([0, 3, 8, 12, 15], dtype=torch.int32)

    densify_routing_table_physical_ids(
        routing_table,
        orig_to_dense,
        num_local_experts=4,
    )

    torch.testing.assert_close(
        routing_table,
        torch.tensor([0, 3, 4, 8, 11], dtype=torch.int32),
    )


def test_w4a8_scale_down_is_rejected_before_topology_changes():
    w4a8_method = object.__new__(AscendW4A8DynamicFusedMoEMethod)
    routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(quant_method=w4a8_method)
    )
    model = SimpleNamespace(
        moe_layers=[SimpleNamespace(routed_experts=routed_experts)]
    )

    with pytest.raises(NotImplementedError, match="W4A8 must be disabled"):
        validate_expert_reload_support(model)
