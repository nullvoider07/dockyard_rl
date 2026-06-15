"""Tests for the MoE block routing/wiring (pure, no GPU).

The routing-map + token-count core is tested directly. The block forward is
exercised with a stub experts module so it never touches the CUDA-only
grouped-GEMM (HV-8) — this validates the router->counts->experts->shared wiring,
the usage counter, and the load-balance buffer, all on CPU.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from dockyard_rl.models.dtensor.moe.block import (
    MoEBlock,
    routing_map_and_counts,
)
from dockyard_rl.models.dtensor.moe.router import TokenChoiceTopKRouter


def test_routing_map_and_counts_matches_manual():
    # 1 token routed to experts {0, 2}; 1 token routed to {2, 3}. E=4.
    scores_BLE = torch.rand(1, 2, 4)
    ids = torch.tensor([[[0, 2], [2, 3]]])  # (B=1, L=2, K=2)
    routing_map, counts = routing_map_and_counts(scores_BLE, ids)
    assert routing_map.dtype == torch.bool
    assert routing_map[0, 0].tolist() == [True, False, True, False]
    assert routing_map[0, 1].tolist() == [False, False, True, True]
    # expert 2 chosen by both tokens; 0 and 3 once; 1 never.
    assert counts.tolist() == [1, 0, 2, 1]


def test_counts_sum_equals_tokens_times_topk():
    r = TokenChoiceTopKRouter(dim=8, num_experts=5, top_k=2)
    x = torch.randn(2, 3, 8)
    _, ids, scores = r(x)
    _, counts = routing_map_and_counts(scores, ids)
    # NOTE: top-k ids are distinct per token, so total assignments = T*K.
    assert int(counts.sum()) == 2 * 3 * 2


class _StubExperts(nn.Module):
    """Records routing inputs; returns a deterministic per-token output without
    invoking grouped-GEMM. Shape-faithful to GroupedExperts.forward."""

    def __init__(self) -> None:
        super().__init__()
        self.last_counts = None

    def forward(self, x_BLD, topk_scores_BLK, topk_expert_ids_BLK, counts_E):
        self.last_counts = counts_E
        # Identity-ish: scale by the summed gating weight so shared-expert
        # addition is observable.
        gate = topk_scores_BLK.sum(dim=-1, keepdim=True)  # (B, L, 1)
        return x_BLD * gate


def _block(num_experts=4, top_k=2, shared=None, lb=None) -> MoEBlock:
    torch.manual_seed(0)
    router = TokenChoiceTopKRouter(dim=8, num_experts=num_experts, top_k=top_k)
    return MoEBlock(
        router,
        _StubExperts(),  # type: ignore[arg-type]
        num_experts=num_experts,
        shared_experts=shared,
        load_balance_coeff=lb,
    )


def test_forward_shape_and_experts_receive_counts():
    blk = _block()
    x = torch.randn(2, 3, 8)
    out = blk(x)
    assert out.shape == (2, 3, 8)
    last_counts = cast(torch.Tensor, blk.experts.last_counts)
    per_expert = cast(torch.Tensor, blk.tokens_per_expert_E)
    assert last_counts.tolist() == [int(c) for c in per_expert.tolist()]


def test_tokens_per_expert_accumulates_across_calls():
    blk = _block()
    x = torch.randn(2, 3, 8)
    blk(x)
    after_one = cast(torch.Tensor, blk.tokens_per_expert_E).clone()
    blk(x)
    assert torch.equal(cast(torch.Tensor, blk.tokens_per_expert_E), after_one * 2)
    assert int(after_one.sum()) == 2 * 3 * 2


def test_shared_expert_is_added():
    shared = nn.Linear(8, 8, bias=False)
    blk = _block(shared=shared)
    x = torch.randn(1, 2, 8)
    # Reconstruct expected: routed (stub) + shared(x)
    topk_scores, _, _ = blk.router(x, blk.expert_bias_E)
    routed = x * topk_scores.sum(dim=-1, keepdim=True)
    expected = routed + shared(x)
    assert torch.allclose(blk(x), expected, atol=1e-5)


def test_load_balance_buffer_present_when_enabled():
    blk = _block(lb=1e-3)
    assert blk.expert_bias_E is not None
    assert blk.expert_bias_E.shape == (4,)
    # default off
    assert _block().expert_bias_E is None
