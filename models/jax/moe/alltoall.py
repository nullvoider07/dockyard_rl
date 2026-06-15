"""EP all-to-all token dispatch for MoE (expert parallelism), JAX (J8c).

JAX mirror of ``models/dtensor/moe/dispatch.py::AllToAllTokenDispatcher``. Under
expert parallelism each device owns ``num_experts / ep`` experts, so a token
routed to a non-local expert must be shuffled to the device that owns it
(``jax.lax.ragged_all_to_all`` over the ``ep`` mesh axis, run inside a
``shard_map``), expert-grouped, computed, then shuffled back in ``combine``.

Two pieces, split by validatability:
  - PURE index math (CPU-tested): the local reorder (inherited), the send/recv
    split + offset computation, and ``permute_rank_to_expert`` /
    ``unpermute_expert_to_rank`` (the rank-major <-> expert-major reordering of
    received tokens). These mirror torch's ``_permute`` / ``_unpermute`` exactly.
  - The ``ragged_all_to_all`` collective itself is device-bound (needs a real
    multi-device ``ep`` axis) — its live numerics are hardware-deferred. With no
    EP wired (``ep_axis is None``), dispatch/combine fall back to the local
    (EP=1) path, so a pre-parallelize or single-device call is correct.

Shape suffixes: T tokens, D dim, K top-k, E experts, N=T*K routed slots,
R routed tokens on this rank, ep = expert-parallel degree, le = E/ep local experts.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from dockyard_rl.models.jax.moe.dispatch import LocalDispatchMetadata, LocalTokenDispatcher

Array = jax.Array


def make_token_dispatcher(
    kind: str, num_experts: int, top_k: int, score_before_experts: bool = True
) -> LocalTokenDispatcher:
    """Construct the routed-token dispatcher by name (mirrors torch ``build_token_dispatcher``).

    ``local`` -> ``LocalTokenDispatcher`` (EP=1). ``alltoall`` ->
    ``AllToAllTokenDispatcher`` whose ``ep`` axis is wired later (``wire_ep``,
    after the mesh exists); until wired it falls back to local routing.
    """
    if kind == "local":
        return LocalTokenDispatcher(num_experts, top_k, score_before_experts)
    if kind == "alltoall":
        return AllToAllTokenDispatcher(num_experts, top_k, score_before_experts)
    raise ValueError(f"unknown token_dispatcher {kind!r}; expected 'local' or 'alltoall'")


def expert_major_permutation(counts_rank_major_le: Array, ep_size: int) -> Array:
    """Indices reordering received tokens from rank-major to expert-major layout.

    Received layout after the all-to-all is rank-major:
    ``(e0,r0),(e1,r0),...,(e0,r1),(e1,r1),...`` — ``counts_rank_major_le`` is the
    flattened ``(ep, le)`` matrix of how many tokens this rank got for each of its
    local experts from each source rank. The returned ``permuted_indices`` gather
    those into expert-major order ``(e0,r0),(e0,r1),...,(e1,r0),...`` so each local
    expert's tokens are contiguous for the grouped GEMM. Pure index arithmetic,
    mirroring torch ``AllToAllTokenDispatcher._permute``.
    """
    total = int(counts_rank_major_le.sum())
    t_mat = counts_rank_major_le.reshape(ep_size, -1)  # (ep, le)
    input_starts = (jnp.cumsum(counts_rank_major_le) - counts_rank_major_le).reshape(ep_size, -1)
    # transpose (ep, le) -> (le, ep) and flatten => expert-major segments
    segment_lens = t_mat.T.reshape(-1)
    seg_input_starts = input_starts.T.reshape(-1)
    num_segments = segment_lens.shape[0]
    seg_ids = jnp.repeat(jnp.arange(num_segments), segment_lens, total_repeat_length=total)
    output_starts = jnp.cumsum(segment_lens) - segment_lens
    permuted_indices = seg_input_starts[seg_ids] + jnp.arange(total) - output_starts[seg_ids]
    return permuted_indices


def local_expert_counts(counts_rank_major_le: Array, ep_size: int) -> Array:
    """Per-local-expert token totals ``(le,)`` (sum the rank-major counts over ranks)."""
    return jnp.sum(counts_rank_major_le.reshape(ep_size, -1), axis=0)


class AllToAllDispatchMetadata(NamedTuple):
    """State carried from ``dispatch`` to ``combine`` (a pytree)."""

    local: LocalDispatchMetadata
    permuted_indices: Optional[Array]      # expert-major permutation of received tokens
    recv_shape: tuple[int, ...]            # pre-permute received shape (for unpermute)
    send_sizes: Optional[Array]            # per-ep-rank token counts sent (reused, swapped)
    recv_sizes: Optional[Array]            # per-ep-rank token counts received


class AllToAllTokenDispatcher(LocalTokenDispatcher):
    """EP>1 token dispatcher: local reorder + ragged all-to-all over the ``ep`` axis.

    ``wire_ep`` installs the mesh axis name + ep degree after construction (the
    mesh does not exist when the model is built). Until wired (``ep_axis is
    None``) ``dispatch``/``combine`` use the inherited local (EP=1) path, so the
    result is correct (local) rather than a crash. The ``ragged_all_to_all``
    collective is device-bound (HV); ``dispatch``/``combine`` must run inside a
    ``shard_map`` over ``ep_axis`` on a real multi-device mesh.
    """

    def __init__(self, num_experts: int, top_k: int, score_before_experts: bool = True) -> None:
        super().__init__(num_experts, top_k, score_before_experts)
        self.ep_axis: Optional[str] = None
        self.ep_size: int = 1

    def wire_ep(self, ep_axis: Optional[str], ep_size: int) -> None:
        """Install the EP mesh axis name + degree (``ep_axis=None`` disables EP)."""
        if ep_axis is not None and ep_size > 1 and self.num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by ep_size({ep_size})"
            )
        self.ep_axis = ep_axis
        self.ep_size = ep_size

    def _ep_enabled(self) -> bool:
        return self.ep_axis is not None and self.ep_size > 1

    def dispatch(  # type: ignore[override]
        self, x_TD: Array, topk_scores_TK: Array, topk_expert_ids_TK: Array
    ) -> tuple[Array, Array, AllToAllDispatchMetadata]:
        """Local reorder, then all-to-all tokens to their experts' EP ranks.

        Returns the expert-major routed inputs for THIS rank's local experts and
        their per-local-expert counts (the grouped-GEMM group sizes). The
        ``ragged_all_to_all`` is device-bound (HV); the local fallback (no EP) is
        the inherited EP=1 path.
        """
        routed_input_ND, tok_idx_N, scores_N = self._local_reorder(
            x_TD, topk_scores_TK, topk_expert_ids_TK
        )
        local_meta = LocalDispatchMetadata(tok_idx_N, scores_N)
        counts_E = self._counts(topk_expert_ids_TK)

        if not self._ep_enabled():
            meta = AllToAllDispatchMetadata(local_meta, None, routed_input_ND.shape, None, None)
            return routed_input_ND, counts_E, meta

        ep = self.ep_size
        # Tokens destined for each ep rank = sum over that rank's experts.
        send_sizes = jnp.sum(counts_E.reshape(ep, -1), axis=1)  # (ep,)
        # Exchange counts so each rank learns how many it will receive (HV collective).
        recv_counts_rank_major = jax.lax.all_to_all(
            counts_E.reshape(ep, -1), self.ep_axis, 0, 0, tiled=True
        )  # (ep, le): [source_rank, local_expert]
        recv_sizes = jnp.sum(recv_counts_rank_major, axis=1)  # (ep,)

        send_offsets = jnp.cumsum(send_sizes) - send_sizes
        recv_offsets = jnp.cumsum(recv_sizes) - recv_sizes
        total_recv = int(recv_sizes.sum())
        output = jnp.zeros((total_recv, x_TD.shape[-1]), routed_input_ND.dtype)
        recv_RD = jax.lax.ragged_all_to_all(
            routed_input_ND, output, send_offsets, send_sizes, recv_offsets, recv_sizes,
            axis_name=self.ep_axis,
        )

        # Received tokens are rank-major; reorder to expert-major for the grouped GEMM.
        counts_flat = recv_counts_rank_major.reshape(-1)
        perm = expert_major_permutation(counts_flat, ep)
        routed_RD = recv_RD[perm]
        counts_le = local_expert_counts(counts_flat, ep)
        meta = AllToAllDispatchMetadata(
            local_meta, perm, recv_RD.shape, send_sizes, recv_sizes
        )
        return routed_RD, counts_le, meta

    def combine(  # type: ignore[override]
        self, routed_output_RD: Array, metadata: AllToAllDispatchMetadata, x_TD: Array
    ) -> Array:
        """Reverse the dispatch: unpermute, all-to-all back, score, scatter-add."""
        if not self._ep_enabled() or metadata.permuted_indices is None:
            return super().combine(routed_output_RD, metadata.local, x_TD)

        # expert-major -> rank-major (invert the dispatch permutation)
        unperm = jnp.zeros(metadata.recv_shape, routed_output_RD.dtype)
        unperm = unperm.at[metadata.permuted_indices].set(routed_output_RD)

        # reverse all-to-all: send/recv splits swap roles
        assert metadata.send_sizes is not None and metadata.recv_sizes is not None
        send_sizes, recv_sizes = metadata.recv_sizes, metadata.send_sizes
        send_offsets = jnp.cumsum(send_sizes) - send_sizes
        recv_offsets = jnp.cumsum(recv_sizes) - recv_sizes
        total = int(recv_sizes.sum())
        output = jnp.zeros((total, x_TD.shape[-1]), unperm.dtype)
        back_RD = jax.lax.ragged_all_to_all(
            unperm, output, send_offsets, send_sizes, recv_offsets, recv_sizes,
            axis_name=self.ep_axis,
        )

        if not self.score_before_experts:
            back_RD = (
                back_RD.astype(jnp.float32)
                * metadata.local.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).astype(back_RD.dtype)
        out_TD = jnp.zeros_like(x_TD)
        out_TD = out_TD.at[metadata.local.token_indices_experts_sorted_N].add(
            back_RD.astype(out_TD.dtype)
        )
        return out_TD
