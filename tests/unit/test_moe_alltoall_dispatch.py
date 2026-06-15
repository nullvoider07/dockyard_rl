"""AllToAllTokenDispatcher (EP>1) — CPU coverage of the pure parts (Phase 4).

The cross-rank ``all_to_all_single`` collectives are device-bound (NCCL, HV-10)
and not exercised here. What IS pure and tested:
  - ``_permute``: rank-major -> expert-major reorder index math + the collapsed
    per-local-expert token counts;
  - ``_unpermute``: exact inverse of ``_permute`` (round-trip identity);
  - the ``ep_mesh is None`` fallback: dispatch/combine match LocalTokenDispatcher
    byte-for-byte (the pre-wired / single-rank path).
"""

from __future__ import annotations

import torch

from dockyard_rl.models.dtensor.moe.dispatch import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
    LocalTokenDispatcher,
)


class _StubMesh:
    """Minimal stand-in for the EP submesh — _permute only calls .size()."""

    def __init__(self, n: int) -> None:
        self._n = n

    def size(self) -> int:
        return self._n


class TestPermute:
    def test_permute_rank_major_to_expert_major(self):
        # ep_size=2, num_local=2. Rank-major counts (e,r) view(ep,e):
        #   rank0: e0=2, e1=1 ; rank1: e0=3, e1=2  -> [2,1,3,2]
        disp = AllToAllTokenDispatcher(num_experts=4, top_k=2)
        disp.ep_mesh = _StubMesh(2)  # type: ignore[assignment]
        counts = torch.tensor([2, 1, 3, 2])
        # tag each of the 8 rank-major rows with its index for an exact check.
        routed = torch.arange(8, dtype=torch.float32).reshape(8, 1)

        input_shape, permuted, permuted_indices, counts_e = disp._permute(
            routed, counts
        )

        # Expert-major: e0 = rank0's e0 rows (0,1) + rank1's e0 rows (3,4,5);
        #               e1 = rank0's e1 row (2)   + rank1's e1 rows (6,7).
        expected_order = torch.tensor([0, 1, 3, 4, 5, 2, 6, 7])
        assert torch.equal(permuted_indices, expected_order)
        assert torch.equal(permuted[:, 0], expected_order.to(torch.float32))
        # per-local-expert totals summed over ranks: e0=2+3=5, e1=1+2=3
        assert torch.equal(counts_e, torch.tensor([5, 3]))
        assert input_shape == routed.shape

    def test_unpermute_inverts_permute(self):
        disp = AllToAllTokenDispatcher(num_experts=4, top_k=2)
        disp.ep_mesh = _StubMesh(2)  # type: ignore[assignment]
        counts = torch.tensor([2, 1, 3, 2])
        routed = torch.randn(8, 5)

        input_shape, permuted, permuted_indices, _ = disp._permute(routed, counts)
        # Treat the expert outputs as the permuted inputs (identity expert):
        # unpermute must restore the original rank-major ordering exactly.
        restored = disp._unpermute(permuted, input_shape, permuted_indices)
        assert torch.equal(restored, routed)


class TestEpDisabledFallback:
    """ep_mesh is None => behave exactly like LocalTokenDispatcher."""

    def _inputs(self):
        torch.manual_seed(0)
        T, D, K, E = 4, 3, 2, 4
        x_TD = torch.randn(T, D)
        topk_scores_TK = torch.rand(T, K)
        topk_expert_ids_TK = torch.randint(0, E, (T, K))
        counts_E = torch.zeros(E, dtype=torch.long)
        for e in topk_expert_ids_TK.view(-1):
            counts_E[e] += 1
        return x_TD, topk_scores_TK, topk_expert_ids_TK, counts_E

    def test_dispatch_matches_local(self):
        x_TD, s_TK, ids_TK, counts_E = self._inputs()
        a2a = AllToAllTokenDispatcher(num_experts=4, top_k=2)  # ep_mesh None
        local = LocalTokenDispatcher(num_experts=4, top_k=2)

        ri_a, counts_a, meta_a = a2a.dispatch(x_TD, s_TK, ids_TK, counts_E)
        ri_l, counts_l, meta_l = local.dispatch(x_TD, s_TK, ids_TK, counts_E)

        assert torch.equal(ri_a, ri_l)
        assert torch.equal(counts_a, counts_l)
        # Fallback returns the LOCAL metadata type, not the AllToAll one.
        assert isinstance(meta_a, LocalDispatchMetadata)
        assert not isinstance(meta_a, AllToAllDispatchMetadata)
        assert torch.equal(
            meta_a.token_indices_experts_sorted_N,
            meta_l.token_indices_experts_sorted_N,
        )

    def test_combine_matches_local(self):
        x_TD, s_TK, ids_TK, counts_E = self._inputs()
        a2a = AllToAllTokenDispatcher(num_experts=4, top_k=2)
        local = LocalTokenDispatcher(num_experts=4, top_k=2)

        ri_a, _, meta_a = a2a.dispatch(x_TD, s_TK, ids_TK, counts_E)
        ri_l, _, meta_l = local.dispatch(x_TD, s_TK, ids_TK, counts_E)
        # Identity "expert": feed the routed inputs straight back to combine.
        # Fallback path returns LocalDispatchMetadata; combine() accepts it.
        out_a = a2a.combine(ri_a, meta_a, x_TD)  # type: ignore[arg-type]
        out_l = local.combine(ri_l, meta_l, x_TD)
        assert torch.equal(out_a, out_l)

    def test_wire_ep_mesh_setter(self):
        disp = AllToAllTokenDispatcher(num_experts=8, top_k=2)
        assert disp.ep_mesh is None
        disp.wire_ep_mesh(_StubMesh(2))  # type: ignore[arg-type]
        assert disp.ep_mesh is not None
        disp.wire_ep_mesh(None)
        assert disp.ep_mesh is None
