"""Tests for the token-choice top-K router (pure, no GPU)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from dockyard_rl.models.dtensor.moe.router import TokenChoiceTopKRouter


def _router(**kw) -> TokenChoiceTopKRouter:
    torch.manual_seed(0)
    defaults: dict[str, Any] = dict(dim=8, num_experts=4, top_k=2)
    defaults.update(kw)
    return TokenChoiceTopKRouter(**defaults)


def test_output_shapes_and_dtypes():
    r = _router(top_k=2, num_experts=4)
    x = torch.randn(2, 3, 8)
    scores, ids, full = r(x)
    assert scores.shape == (2, 3, 2)
    assert ids.shape == (2, 3, 2)
    assert full.shape == (2, 3, 4)
    assert ids.dtype == torch.int64


def test_selected_ids_in_range_and_distinct():
    r = _router(top_k=3, num_experts=5)
    x = torch.randn(4, 6, 8)
    _, ids, _ = r(x)
    assert int(ids.min()) >= 0 and int(ids.max()) < 5
    # top-k indices are distinct per token (no expert chosen twice)
    for b in range(4):
        for l in range(6):
            sel = ids[b, l].tolist()
            assert len(set(sel)) == len(sel)


def test_sigmoid_scores_in_unit_interval():
    r = _router(score_func="sigmoid", num_experts=4, top_k=2)
    x = torch.randn(2, 2, 8)
    scores, _, full = r(x)
    assert torch.all((full >= 0) & (full <= 1))
    assert torch.all((scores >= 0) & (scores <= 1))


def test_softmax_full_scores_sum_to_one():
    r = _router(score_func="softmax", num_experts=4, top_k=2)
    x = torch.randn(2, 5, 8)
    _, _, full = r(x)
    assert torch.allclose(full.sum(dim=-1), torch.ones(2, 5), atol=1e-5)


def test_route_norm_normalizes_topk():
    r = _router(score_func="softmax", route_norm=True, num_experts=4, top_k=2)
    x = torch.randn(3, 3, 8)
    scores, _, _ = r(x)
    assert torch.allclose(scores.sum(dim=-1), torch.ones(3, 3), atol=1e-5)


def test_route_scale_applied():
    x = torch.randn(2, 2, 8)
    r1 = _router(score_func="softmax", route_norm=True, route_scale=1.0)
    s1, _, _ = r1(x)
    r2 = _router(score_func="softmax", route_norm=True, route_scale=2.5)
    s2, _, _ = r2(x)
    assert torch.allclose(s2, s1 * 2.5, atol=1e-5)


def test_gating_scores_match_gathered_full_scores():
    # Without norm/scale, topk_scores must equal full scores gathered at ids.
    r = _router(score_func="sigmoid", num_experts=6, top_k=3)
    x = torch.randn(2, 4, 8)
    scores, ids, full = r(x)
    assert torch.allclose(scores, full.gather(-1, ids))


def test_expert_bias_changes_choice_not_gating():
    # A huge bias on one expert forces it into every token's selection, but the
    # returned gating weight for it is still the unbiased score.
    r = _router(score_func="sigmoid", num_experts=4, top_k=1)
    x = torch.randn(3, 3, 8)
    bias = torch.tensor([0.0, 0.0, 1e9, 0.0])
    scores, ids, full = r(x, expert_bias_E=bias)
    assert torch.all(ids.squeeze(-1) == 2)
    # gating value equals unbiased score of expert 2, not a biased/sigmoid(1e9)
    assert torch.allclose(scores.squeeze(-1), full[..., 2])


def test_node_limited_routing_masks_other_groups():
    # 2 groups of 3 experts; keep 1 group. Selected experts must all lie in the
    # single kept group for each token.
    r = _router(
        score_func="sigmoid",
        num_experts=6,
        top_k=2,
        num_expert_groups=2,
        num_limited_groups=1,
    )
    x = torch.randn(4, 4, 8)
    _, ids, _ = r(x)
    for b in range(4):
        for l in range(4):
            groups = {int(e) // 3 for e in ids[b, l]}
            assert len(groups) == 1


def test_invalid_top_k_rejected():
    with pytest.raises(ValueError):
        TokenChoiceTopKRouter(dim=8, num_experts=4, top_k=5)


def test_node_limited_requires_limit():
    with pytest.raises(ValueError):
        TokenChoiceTopKRouter(dim=8, num_experts=6, num_expert_groups=2)
