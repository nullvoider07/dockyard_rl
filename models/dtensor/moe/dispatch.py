"""Token dispatch / combine for MoE expert routing.

Reorders tokens by expert assignment so each expert's tokens are contiguous for
the grouped matmul, then scatter-adds the expert outputs back to token
positions. Distilled from torchtitan's LocalTokenDispatcher (pytorch/torchtitan,
BSD-3-Clause; Copyright (c) Meta Platforms, Inc. and affiliates) into a plain
dockyard module — no torchtitan ``Configurable`` / ``ops`` framework.

``LocalTokenDispatcher`` is the EP=1 (single expert-parallel rank) path: pure
local reordering, no all-to-all. ``AllToAllTokenDispatcher`` (EP>1) adds the
cross-rank token shuffle (the DeepEP-equivalent ``all_to_all_single``): the
all-to-all collectives are device-bound (NCCL, HV-10), but the reorder /
permute index math and the EP-disabled fallback are pure tensor ops and are
unit-tested on CPU.

Shape suffixes: T = num tokens (B*L), D = model dim, K = top-k, E = num
(local) experts, N = T*K routed slots, R = routed tokens (= N for EP=1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import DeviceMesh


@dataclass
class LocalDispatchMetadata:
    """State carried from dispatch() to combine()."""

    token_indices_experts_sorted_N: torch.Tensor
    topk_scores_experts_sorted_N: torch.Tensor


class LocalTokenDispatcher:
    """Local token reordering for expert compute (EP=1; base for EP variants).

    Not an nn.Module — holds no learnable state.

    Args:
        num_experts: Number of (local) experts.
        top_k: Experts selected per token.
        score_before_experts: If True, routing scores are applied to inputs
            before the expert FFN; otherwise to the outputs in combine().
    """

    def __init__(
        self, num_experts: int, top_k: int, score_before_experts: bool = True
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_before_experts = score_before_experts

    def _local_reorder(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Group the T*K routed slots by expert id (stable argsort).

        Returns the expert-sorted routed inputs (score-weighted when
        ``score_before_experts``), the slot->original-token map, and the
        expert-sorted scores.
        """
        token_indices_experts_sorted_N = torch.argsort(
            topk_expert_ids_TK.view(-1), stable=True
        )
        topk_scores_experts_sorted_N = topk_scores_TK.view(-1)[
            token_indices_experts_sorted_N
        ]
        # N routed slots map back to T tokens (K slots per token).
        token_indices_experts_sorted_N = token_indices_experts_sorted_N // self.top_k
        routed_input_ND = x_TD[token_indices_experts_sorted_N]

        if self.score_before_experts:
            routed_input_ND = (
                routed_input_ND.to(torch.float32)
                * topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(x_TD.dtype)

        return (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert; returns (routed_input_RD, counts_E, metadata)."""
        (
            routed_input_RD,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        )
        return routed_input_RD, num_local_tokens_per_expert_E, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter-add expert outputs back to their original token rows.

        Each token's top-k expert outputs accumulate into one row, weighted by
        the routing score (applied here when ``score_before_experts`` is False).
        """
        if not self.score_before_experts:
            routed_output_RD = (
                routed_output_RD.to(torch.float32)
                * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(routed_output_RD.dtype)

        out_TD = torch.zeros_like(x_TD)
        out_TD.index_add_(
            0, metadata.token_indices_experts_sorted_N, routed_output_RD.to(out_TD.dtype)
        )
        return out_TD


@dataclass
class AllToAllDispatchMetadata(LocalDispatchMetadata):
    """State carried from AllToAllTokenDispatcher.dispatch() to combine().

    Extends the local metadata with the EP all-to-all bookkeeping needed to
    reverse the shuffle: the pre-permute shape and the rank-major->expert-major
    permutation, plus the per-rank token splits (reused, swapped, by the reverse
    all-to-all in combine()).
    """

    input_shape: torch.Size
    permuted_indices: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]


class AllToAllTokenDispatcher(LocalTokenDispatcher):
    """EP>1 token dispatcher: local reorder + cross-rank all-to-all (HV-10).

    Distributes routed tokens across the expert-parallel ranks so each rank's
    grouped-GEMM sees only the tokens destined for its LOCAL experts, then
    reverses the shuffle in ``combine``. Distilled from torchtitan's
    ``AllToAllTokenDispatcher`` (BSD-3-Clause).

    The ``ep_mesh`` is wired AFTER construction (``wire_ep_mesh``) because the
    device mesh does not exist when the surgery/config path builds the
    dispatcher. Until wired (``ep_mesh is None``) ``dispatch``/``combine`` fall
    back to the local EP=1 path, so a misconfigured or pre-parallelize call is
    correct (local) rather than a crash.

    Sequence parallelism is NOT supported on the MoE path (see
    ``parallelize.py::_parallelize_qwen3_moe``), so there is no SP shard
    offset — ``combine`` scatters into a full ``(T, D)`` buffer directly.

    The ``all_to_all_single`` collectives are device-bound (NCCL); the
    ``_permute``/``_unpermute`` index math and the EP-disabled fallback are pure
    and CPU-tested.
    """

    def __init__(
        self, num_experts: int, top_k: int, score_before_experts: bool = True
    ) -> None:
        super().__init__(num_experts, top_k, score_before_experts)
        self.ep_mesh: Optional[DeviceMesh] = None

    def wire_ep_mesh(self, ep_mesh: Optional[DeviceMesh]) -> None:
        """Install the 1-D EP submesh used by dispatch/combine (None disables EP)."""
        self.ep_mesh = ep_mesh

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Union[LocalDispatchMetadata, AllToAllDispatchMetadata],
    ]:
        """Local reorder, then all-to-all tokens to their experts' EP ranks.

        ``num_local_tokens_per_expert_E`` is this rank's token count over the
        GLOBAL experts ``(E,)``; experts are assigned contiguously per EP rank
        (see ``mesh.experts_for_rank``), so ``view(ep_size, -1)`` rows are the
        per-destination-rank groups. Returns the expert-major routed inputs for
        this rank's LOCAL experts and their global token counts.
        """
        if self.ep_mesh is None:
            return super().dispatch(
                x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
            )

        ep_size = self.ep_mesh.size()
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        with torch.no_grad():
            # Exchange per-expert counts so every rank learns how many tokens it
            # will receive for its local experts (rank-major layout).
            num_global_tokens_per_local_expert_E = all_to_all_single(
                num_local_tokens_per_expert_E, None, None, group=self.ep_mesh
            )
            num_global_tokens_per_local_expert_E = (
                torch.ops._c10d_functional.wait_tensor(
                    num_global_tokens_per_local_expert_E
                )
            )
            input_splits = (
                num_local_tokens_per_expert_E.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"))
            )
            output_splits = (
                num_global_tokens_per_local_expert_E.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"))
            )
            input_splits_list = input_splits.tolist()
            output_splits_list = output_splits.tolist()

        routed_input_RD = all_to_all_single_autograd(
            routed_input_ND,
            output_splits_list,
            input_splits_list,
            self.ep_mesh,
        )

        # Received tokens are rank-major ((e0,r0),(e1,r0),...,(e0,r1),...);
        # reorder to expert-major so each local expert's tokens are contiguous.
        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        ) = self._permute(routed_input_RD, num_global_tokens_per_local_expert_E)

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def _permute(
        self,
        routed_input_RD: torch.Tensor,
        num_global_tokens_per_local_expert_E: torch.Tensor,
    ) -> tuple[torch.Size, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder received tokens from rank-major to expert-major layout.

        Input layout:  (e0,r0),(e1,r0),...,(e0,r1),(e1,r1),...  (rank-major)
        Output layout: (e0,r0),(e0,r1),...,(e1,r0),(e1,r1),...  (expert-major)

        Pure index arithmetic (no comms). Also collapses the ``(EP, e)`` count
        matrix to ``num_global_tokens_per_local_expert_e`` ``(e,)`` by summing
        over ranks — the grouped-GEMM's per-local-expert token counts.
        """
        assert self.ep_mesh is not None
        ep_size = self.ep_mesh.size()
        device = num_global_tokens_per_local_expert_E.device

        t_mat = num_global_tokens_per_local_expert_E.view(ep_size, -1)
        input_starts = (
            num_global_tokens_per_local_expert_E.cumsum(0)
            - num_global_tokens_per_local_expert_E
        ).view(ep_size, -1)

        # Transpose (EP, e) -> (e, EP) and flatten for expert-major segments.
        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        seg_ids = torch.arange(
            segment_lens.shape[0], device=device
        ).repeat_interleave(segment_lens)
        output_starts = segment_lens.cumsum(0) - segment_lens
        permuted_indices = (
            input_starts[seg_ids]
            + torch.arange(seg_ids.shape[0], device=device)
            - output_starts[seg_ids]
        )

        num_global_tokens_per_local_expert_e = t_mat.sum(0)
        return (
            routed_input_RD.shape,
            routed_input_RD[permuted_indices, :],
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        )

    def _unpermute(
        self,
        routed_output_RD: torch.Tensor,
        input_shape: torch.Size,
        permuted_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Invert ``_permute`` (expert-major -> rank-major)."""
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD

    def combine(  # type: ignore[override]
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute, all-to-all back, score, scatter-add."""
        if self.ep_mesh is None:
            return super().combine(routed_output_RD, metadata, x_TD)

        routed_output_RD = self._unpermute(
            routed_output_RD, metadata.input_shape, metadata.permuted_indices
        )
        # Reverse all-to-all: the dispatch splits swap roles (send back what we
        # received, receive what we sent).
        routed_output_RD = all_to_all_single_autograd(
            routed_output_RD,
            metadata.input_splits,
            metadata.output_splits,
            self.ep_mesh,
        )

        if not self.score_before_experts:
            routed_output_RD = (
                routed_output_RD.to(torch.float32)
                * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(routed_output_RD.dtype)

        out_TD = torch.zeros_like(x_TD)
        out_TD.index_add_(
            0,
            metadata.token_indices_experts_sorted_N,
            routed_output_RD.to(out_TD.dtype),
        )
        return out_TD
