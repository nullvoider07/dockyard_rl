"""Tests for MoE router-replay capture alignment (pure, CPU).

Validates the next-token alignment of vLLM's per-token routed-expert array onto
the trainer's right-padded sequence layout: prompt/response concatenation, the
arange default route for the final token + padding, sentinel-marking of missing
interior routes, the route-accounting stats, the dtype the column carries, and
the vLLM-object adapter. All leaf logic, no GPU.
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
    stack_routed_experts,
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


def _aligned(*args, **kwargs) -> torch.Tensor:
    """`align_routed_expert_indices` narrowed to its tensor return.

    The public signature is a union over `return_stats` / the no-routing case;
    these dtype tests always take the plain-tensor branch.
    """
    out = align_routed_expert_indices(*args, **kwargs)
    assert isinstance(out, torch.Tensor)
    return out


def test_int16_numpy_input_keeps_its_width():
    """vLLM returns int16; widening it doubled the column for no consumer.

    ``MoERouter._replayed_topk_ids`` casts to ``torch.long`` before indexing, so
    nothing downstream needs 32 bits.
    """
    routed = np.zeros((3, 2, 2), dtype=np.int16)
    routed[1, 0, 0] = 7
    full = _aligned(routed, valid_length=4, padded_length=5)
    assert full.dtype == torch.int16
    assert int(full[1, 0, 0]) == 7


def test_the_default_route_matches_the_captured_width():
    """The arange tail and the copied head are one tensor; they cannot differ."""
    routed = np.zeros((3, 2, 2), dtype=np.int16)
    full = _aligned(routed, valid_length=4, padded_length=6)
    assert full.dtype == torch.int16
    # Final token + padding take the identity route, in the same dtype.
    assert torch.equal(
        full[4], torch.tensor([[0, 1], [0, 1]], dtype=torch.int16)
    )


def test_the_missing_route_sentinel_survives_the_narrow_width():
    """int16 is signed, so -1 round-trips; that is why unsigned is refused."""
    routed = np.zeros((1, 2, 2), dtype=np.int16)
    full = _aligned(routed, valid_length=5, padded_length=6)
    assert full.dtype == torch.int16
    assert int(full[2, 0, 0]) == MISSING_ROUTE_SENTINEL


def test_unsigned_input_is_widened_not_carried():
    """uint8 cannot hold -1, so carrying it would corrupt the sentinel."""
    routed = np.zeros((1, 1, 2), dtype=np.uint8)
    full = _aligned(routed, valid_length=4, padded_length=4)
    assert full.dtype == torch.int32
    assert int(full[1, 0, 0]) == MISSING_ROUTE_SENTINEL


def test_list_input_falls_back_to_int32():
    routed = [[[0, 1]], [[2, 3]]]
    full = _aligned(routed, valid_length=3, padded_length=4)
    assert full.dtype == torch.int32


def test_int64_input_is_carried_unchanged():
    routed = torch.zeros((2, 1, 2), dtype=torch.int64)
    full = _aligned(routed, valid_length=3, padded_length=4)
    assert full.dtype == torch.int64


def test_stack_synthesizes_empties_at_the_batch_width():
    narrow = _aligned(
        np.zeros((2, 1, 2), dtype=np.int16), valid_length=3, padded_length=4
    )
    stacked = stack_routed_experts([narrow, None], padded_length=4)
    assert stacked is not None
    assert stacked.dtype == torch.int16
    assert tuple(stacked.shape) == (2, 4, 1, 2)


def test_stack_refuses_mixed_widths():
    """torch.stack type-promotes rather than refusing, hiding the mismatch."""
    narrow = _aligned(
        np.zeros((2, 1, 2), dtype=np.int16), valid_length=3, padded_length=4
    )
    wide = _aligned(
        np.zeros((2, 1, 2), dtype=np.int32), valid_length=3, padded_length=4
    )
    try:
        stack_routed_experts([narrow, wide], padded_length=4)
    except ValueError as e:
        assert "dtypes" in str(e)
    else:
        raise AssertionError("expected ValueError on mixed routed_experts dtypes")


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


def test_stack_all_none_returns_none():
    assert stack_routed_experts([None, None], padded_length=5) is None


def test_stack_builds_batch_with_arange_fill_for_none_samples():
    # Sample 0 has routing; sample 1 is None -> filled with the default route.
    sample0 = align_routed_expert_indices(
        _routes(3, 2, 2), valid_length=4, padded_length=6
    )
    batch = stack_routed_experts([sample0, None], padded_length=6)
    assert batch is not None
    assert batch.shape == (2, 6, 2, 2)
    assert batch.dtype == torch.int32
    # Row 0 is the aligned sample; row 1 is the identity arange default route.
    assert torch.equal(batch[0], sample0)
    assert torch.equal(batch[1], routed_experts_empty(6, 2, 2))


def test_stack_preserves_each_sample():
    s0 = align_routed_expert_indices(_routes(2, 1, 2), valid_length=3, padded_length=4)
    s1 = align_routed_expert_indices(
        _routes(3, 1, 2, start=900), valid_length=4, padded_length=4
    )
    batch = stack_routed_experts([s0, s1], padded_length=4)
    assert batch is not None
    assert batch.shape == (2, 4, 1, 2)
    assert torch.equal(batch[0], s0)
    assert torch.equal(batch[1], s1)


def test_stack_inconsistent_shape_raises():
    s0 = routed_experts_empty(4, 2, 2)
    s1 = routed_experts_empty(4, 3, 2)  # different L
    try:
        stack_routed_experts([s0, s1], padded_length=4)
    except ValueError as e:
        assert "inconsistent aligned routed_experts shapes" in str(e)
    else:
        raise AssertionError("expected ValueError on inconsistent shapes")
