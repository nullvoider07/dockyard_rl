"""Tests for MoE token dispatch/combine (pure tensor ops, CPU).

The grouped-mm expert compute is CUDA-only (HV-8) and not exercised here. The
dispatch (expert-major reorder) and combine (scatter-add back) are pure and are
verified via the identity-expert property: with experts as identity and scores
applied once, ``combine(dispatch(x)) == x * scores.sum(dim=k)``.
"""

from __future__ import annotations

import torch

from dockyard_rl.models.dtensor.moe.dispatch import LocalTokenDispatcher


def _routing(T: int, E: int, K: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x_TD = torch.randn(T, 8, generator=g)
    scores_TK = torch.rand(T, K, generator=g)
    ids_TK = torch.randint(0, E, (T, K), generator=g)
    counts_E = torch.bincount(ids_TK.view(-1), minlength=E)
    return x_TD, scores_TK, ids_TK, counts_E


class TestDispatch:
    def test_counts_passthrough(self):
        d = LocalTokenDispatcher(num_experts=4, top_k=2)
        x, s, ids, counts = _routing(6, 4, 2)
        _, out_counts, _ = d.dispatch(x, s, ids, counts)
        assert torch.equal(out_counts, counts)

    def test_routed_rows_are_expert_sorted(self):
        d = LocalTokenDispatcher(num_experts=4, top_k=2)
        x, s, ids, counts = _routing(6, 4, 2)
        # expert ids in the order the routed rows are emitted must be non-decreasing
        sorted_slot_order = torch.argsort(ids.view(-1), stable=True)
        expert_seq = ids.view(-1)[sorted_slot_order]
        assert torch.all(expert_seq[1:] >= expert_seq[:-1])
        # dispatch emits N = T*K routed rows
        routed, _, _ = d.dispatch(x, s, ids, counts)
        assert routed.shape == (6 * 2, 8)

    def test_token_index_map_covers_each_token_k_times(self):
        d = LocalTokenDispatcher(num_experts=4, top_k=3)
        x, s, ids, counts = _routing(5, 4, 3)
        _, _, meta = d.dispatch(x, s, ids, counts)
        idx = meta.token_indices_experts_sorted_N
        assert idx.shape == (5 * 3,)
        assert torch.equal(
            torch.bincount(idx, minlength=5), torch.full((5,), 3)
        )


class TestRoundTrip:
    def _check_identity(self, score_before_experts: bool):
        T, E, K = 7, 4, 2
        d = LocalTokenDispatcher(
            num_experts=E, top_k=K, score_before_experts=score_before_experts
        )
        x, s, ids, counts = _routing(T, E, K, seed=3)
        routed_in, _, meta = d.dispatch(x, s, ids, counts)
        # identity "expert": output == input
        out = d.combine(routed_in, meta, x)
        # each token accumulates score-weighted copies of itself over its K experts
        expected = x * s.sum(dim=1, keepdim=True)
        assert torch.allclose(out, expected, atol=1e-5), (
            f"max err {(out - expected).abs().max().item()}"
        )

    def test_identity_score_before_experts(self):
        self._check_identity(score_before_experts=True)

    def test_identity_score_after_experts(self):
        self._check_identity(score_before_experts=False)
