"""Token dispatch / combine for MoE expert routing, JAX.

JAX mirror of ``models/dtensor/moe/dispatch.py`` (the EP=1 ``LocalTokenDispatcher``):
reorder the ``T*K`` routed slots so each expert's tokens are contiguous (a stable
argsort on the expert ids — the layout ``jax.lax.ragged_dot`` needs), then
scatter-add the expert outputs back to their original token rows. Pure, static-
shape (``N = T*K``), jit-friendly, and CPU-validatable.

The EP>1 all-to-all dispatcher (cross-rank token shuffle via ``jax.lax.all_to_all``
under ``shard_map``) is a later sub-phase; its live numerics are hardware-deferred.

Shape suffixes: T = tokens (B*L), D = model dim, K = top-k, E = experts,
N = T*K routed slots (= R routed tokens for EP=1).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class LocalDispatchMetadata(NamedTuple):
    """State carried from ``dispatch`` to ``combine`` (a pytree)."""

    token_indices_experts_sorted_N: Array  # routed slot -> original token row
    topk_scores_experts_sorted_N: Array    # expert-sorted gating weights


class LocalTokenDispatcher:
    """Local token reordering for expert compute (EP=1). Holds no learnable state.

    ``score_before_experts``: apply routing weights to the inputs before the FFN
    (matching torchtitan's default) vs to the outputs in ``combine``.
    """

    def __init__(self, num_experts: int, top_k: int, score_before_experts: bool = True) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_before_experts = score_before_experts

    def _counts(self, topk_expert_ids_TK: Array) -> Array:
        """Per-expert token counts ``(E,)`` int32 — the ragged_dot group sizes."""
        oh = jax.nn.one_hot(
            topk_expert_ids_TK.reshape(-1), self.num_experts, dtype=jnp.int32
        )
        return jnp.sum(oh, axis=0)

    def _local_reorder(
        self, x_TD: Array, topk_scores_TK: Array, topk_expert_ids_TK: Array
    ) -> tuple[Array, Array, Array]:
        """Group the ``T*K`` routed slots by expert id (stable argsort)."""
        ids_N = topk_expert_ids_TK.reshape(-1)
        order_N = jnp.argsort(ids_N, stable=True)
        topk_scores_experts_sorted_N = topk_scores_TK.reshape(-1)[order_N]
        # N routed slots map back to T tokens (K slots per token).
        token_indices_experts_sorted_N = order_N // self.top_k
        routed_input_ND = x_TD[token_indices_experts_sorted_N]
        if self.score_before_experts:
            routed_input_ND = (
                routed_input_ND.astype(jnp.float32)
                * topk_scores_experts_sorted_N.reshape(-1, 1)
            ).astype(x_TD.dtype)
        return routed_input_ND, token_indices_experts_sorted_N, topk_scores_experts_sorted_N

    def dispatch(
        self, x_TD: Array, topk_scores_TK: Array, topk_expert_ids_TK: Array
    ) -> tuple[Array, Array, LocalDispatchMetadata]:
        """Reorder tokens by expert; returns ``(routed_input_RD, counts_E, metadata)``."""
        routed_input_RD, tok_idx_N, scores_N = self._local_reorder(
            x_TD, topk_scores_TK, topk_expert_ids_TK
        )
        counts_E = self._counts(topk_expert_ids_TK)
        return routed_input_RD, counts_E, LocalDispatchMetadata(tok_idx_N, scores_N)

    def combine(
        self, routed_output_RD: Array, metadata: LocalDispatchMetadata, x_TD: Array
    ) -> Array:
        """Scatter-add expert outputs back to their original token rows.

        Each token's top-K expert outputs accumulate into one row, weighted by
        the routing score (applied here when ``score_before_experts`` is False).
        """
        if not self.score_before_experts:
            routed_output_RD = (
                routed_output_RD.astype(jnp.float32)
                * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).astype(routed_output_RD.dtype)
        out_TD = jnp.zeros_like(x_TD)
        out_TD = out_TD.at[metadata.token_indices_experts_sorted_N].add(
            routed_output_RD.astype(out_TD.dtype)
        )
        return out_TD
