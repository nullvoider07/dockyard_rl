"""Tests for MoE router-replay capture alignment (pure, CPU).

Validates the next-token alignment of vLLM's per-token routed-expert array onto
the trainer's right-padded sequence layout: prompt/response concatenation, the
arange default route for the final token + padding, sentinel-marking of missing
interior routes, the route-accounting stats, dtype coercion, and the vLLM-object
adapter. All leaf logic, no GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from dockyard_rl.models.generation.vllm.router_capture import (
    MISSING_ROUTE_SENTINEL,
    align_routed_expert_indices,
    routed_experts_empty,
    routed_experts_from_vllm_output,
)


def _routes(num_tokens: int, num_layers: int, top_k: int, start: int = 0) -> torch.Tensor:
    """Distinct per-position routes so copy placement is verifiable."""
    base = torch.arange(num_tokens, dtype=torch.int32).view(num_tokens, 1, 1)
    grid = torch.arange(num_layers * top_k, dtype=torch.int32).view(1, num_layers, top_k)
    return base * 100 + grid + start


def test_empty_builder_is_arange_default_route():
    full = routed_experts_empty(5, 3, 2)
    assert full.shape == (5, 3, 2)
    assert full.dtype == torch.int32
    expected_row = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int32)
    assert torch.equal(full[0], expected_row)
    assert torch.equal(full[4], expected_row)


def test_basic_alignment_next_token_positions():
    # valid_length=4 -> 3 next-token routes; padded to 6.
    routed = _routes(3, 2, 2)
    full = align_routed_expert_indices(
        routed, valid_length=4, padded_length=6
    )
    assert full is not None
    assert full.shape == (6, 2, 2)
    assert full.dtype == torch.int32
    # First 3 positions carry the recorded routes.
    assert torch.equal(full[:3], routed)
    # Final token (position 3) + padding (4,5) take the arange default route.
    default_row = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    for p in (3, 4, 5):
        assert torch.equal(full[p], default_row)


def test_prompt_and_response_concatenated():
    prompt = _routes(2, 2, 2, start=0)
    resp = _routes(3, 2, 2, start=1000)
    # valid_length = 2 prompt + 3 resp = 5 -> 4 next-token routes.
    full = align_routed_expert_indices(
        resp, prompt, valid_length=5, padded_length=7
    )
    assert full is not None
    cat = torch.cat((prompt, resp), dim=0)
    # 4 routes copied from the concatenated [prompt; resp].
    assert torch.equal(full[:4], cat[:4])
    # Position 4 (5th token, the final one) is default.
    assert torch.equal(full[4], torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))


def test_prompt_only():
    prompt = _routes(4, 1, 3)
    full = align_routed_expert_indices(
        None, prompt, valid_length=4, padded_length=4
    )
    assert full is not None
    assert torch.equal(full[:3], prompt[:3])


def test_none_routing_returns_none():
    assert (
        align_routed_expert_indices(None, None, valid_length=4, padded_length=6)
        is None
    )


def test_missing_interior_routes_filled_with_sentinel():
    # expected 4 routes (valid_length=5) but backend returned only 2.
    routed = _routes(2, 2, 2)
    full, stats = align_routed_expert_indices(
        routed, valid_length=5, padded_length=6, return_stats=True
    )
    assert full is not None
    assert stats["expected_routes"] == 4
    assert stats["actual_routes"] == 2
    assert stats["missing_routes"] == 2
    # Positions 0,1 copied; 2,3 are the missing interior -> sentinel.
    assert torch.equal(full[:2], routed)
    assert (full[2] == MISSING_ROUTE_SENTINEL).all()
    assert (full[3] == MISSING_ROUTE_SENTINEL).all()
    # Position 4 (final) + 5 (pad) stay default arange (not sentinel).
    assert torch.equal(full[4], torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))


def test_require_complete_raises_on_missing_without_fallback():
    routed = _routes(2, 2, 2)
    try:
        align_routed_expert_indices(
            routed,
            valid_length=5,
            padded_length=6,
            require_complete=True,
            allow_missing_fallback=False,
        )
    except ValueError as e:
        assert "incomplete routed_experts" in str(e)
    else:
        raise AssertionError("expected ValueError on incomplete routes")


def test_require_complete_tolerates_missing_with_fallback():
    routed = _routes(2, 2, 2)
    full = align_routed_expert_indices(
        routed,
        valid_length=5,
        padded_length=6,
        require_complete=True,
        allow_missing_fallback=True,
    )
    assert full is not None
    assert (full[2] == MISSING_ROUTE_SENTINEL).all()


def test_surplus_route_tolerated_when_not_require_complete():
    # expected 3 routes (valid_length=4) but backend returned 4 (one surplus
    # final-token route). Only the first 3 are copied; the surplus is dropped.
    routed = _routes(4, 2, 2)
    full, stats = align_routed_expert_indices(
        routed, valid_length=4, padded_length=6, return_stats=True
    )
    assert full is not None
    assert stats["surplus_routes"] == 0  # 4 == expected(3)+1, within allowance
    assert torch.equal(full[:3], routed[:3])


def test_require_complete_raises_on_excess_surplus():
    # 5 routes vs expected 3 -> more than one surplus -> raise.
    routed = _routes(5, 2, 2)
    try:
        align_routed_expert_indices(
            routed,
            valid_length=4,
            padded_length=8,
            require_complete=True,
        )
    except ValueError as e:
        assert "too many routed_experts" in str(e)
    else:
        raise AssertionError("expected ValueError on excess surplus routes")


def test_wrong_dim_raises():
    bad = torch.zeros(3, 2, dtype=torch.int32)  # 2D, not [tokens, L, topk]
    try:
        align_routed_expert_indices(bad, valid_length=3, padded_length=4)
    except ValueError as e:
        assert "num_moe_layers" in str(e)
    else:
        raise AssertionError("expected ValueError on 2D routing")


def test_int16_numpy_input_coerced_to_int32():
    routed = np.zeros((3, 2, 2), dtype=np.int16)
    routed[1, 0, 0] = 7
    full = align_routed_expert_indices(routed, valid_length=4, padded_length=5)
    assert full is not None
    assert full.dtype == torch.int32
    assert int(full[1, 0, 0]) == 7


def test_valid_length_one_yields_all_default():
    # valid_length=1 -> 0 next-token routes; everything default.
    routed = _routes(2, 2, 2)
    full = align_routed_expert_indices(
        routed, valid_length=1, padded_length=4
    )
    assert full is not None
    default_row = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    for p in range(4):
        assert torch.equal(full[p], default_row)


def test_expected_routes_capped_by_padded_length():
    # valid_length implies more routes than padded_length allows.
    routed = _routes(10, 1, 2)
    full, stats = align_routed_expert_indices(
        routed, valid_length=10, padded_length=4, return_stats=True
    )
    assert full is not None
    assert full.shape == (4, 1, 2)
    assert stats["expected_routes"] == 4  # min(9, 4)


def test_vllm_output_adapter():
    routed = _routes(3, 2, 2)
    prompt = _routes(2, 2, 2, start=500)
    completion = SimpleNamespace(routed_experts=routed.numpy().astype(np.int16))
    request = SimpleNamespace(prompt_routed_experts=prompt.numpy().astype(np.int16))
    full = routed_experts_from_vllm_output(
        request, completion, valid_length=5, padded_length=7
    )
    assert full is not None
    assert full.shape == (7, 2, 2)
    cat = torch.cat((prompt, routed), dim=0)
    assert torch.equal(full[:4], cat[:4])


def test_vllm_output_adapter_missing_fields_returns_none():
    completion = SimpleNamespace()  # no routed_experts attr
    request = SimpleNamespace()
    full = routed_experts_from_vllm_output(
        request, completion, valid_length=4, padded_length=6
    )
    assert full is None
