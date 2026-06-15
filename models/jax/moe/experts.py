"""Grouped-GEMM routed-expert compute for MoE (SwiGLU), Flax NNX.

JAX mirror of ``models/dtensor/moe/experts.py``. The routed-expert FFN runs as a
single grouped matmul over the local experts via ``jax.lax.ragged_dot`` (the JAX
equivalent of torch ``_grouped_mm``) instead of a per-expert loop. Unlike the
torch kernel (CUDA-only, HV-8), ``ragged_dot`` lowers on CPU, so this is
CPU-parity-validatable against a per-expert reference.

Expert weights are 3D ``(num_experts, *, *)`` and shard ``Shard(0)`` over the EP
mesh (J8b); under EP, ``num_experts`` here is the LOCAL expert count. ``__call__``
orchestrates dispatch -> grouped compute -> combine via an injected dispatcher.

Shape suffixes: D=model dim, F=hidden dim, E=experts, R=routed tokens.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.moe.dispatch import LocalTokenDispatcher
from dockyard_rl.models.jax.sharding import EXPERT_DOWN_AXES, EXPERT_GATE_UP_AXES

Array = jax.Array

# ragged_dot rhs is (groups, k, n); torch stores experts as (E, out, in) and
# transposes to (E, in, out) for the matmul. The JAX params mirror the torch
# names/shapes (so the HF/refit name-map is shared); swap the last two axes at
# call time to feed ragged_dot.


class GroupedExperts(nnx.Module):
    """SwiGLU routed experts computed as a grouped matmul.

    Args:
        dim: Model dimension (D).
        hidden_dim: FFN intermediate dimension (F).
        num_experts: Local expert count (E / ep under expert parallelism).
        dispatcher: Token dispatcher providing ``dispatch``/``combine``.
        compute_dtype: Optional matmul dtype (the GPU path uses bfloat16 for the
            grouped GEMM); ``None`` keeps the input dtype (fp32 CPU parity).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        dispatcher: LocalTokenDispatcher,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: Optional[jnp.dtype] = None,
    ) -> None:
        self.num_experts = num_experts
        key = rngs.params()
        k1, k2, k3 = jax.random.split(key, 3)
        scale = 1.0 / jnp.sqrt(jnp.asarray(dim, jnp.float32))
        dscale = 1.0 / jnp.sqrt(jnp.asarray(hidden_dim, jnp.float32))
        # Expert dim shards over ``ep``, hidden dim over ``tp`` (J8b); inert
        # metadata until shard_model places it on a mesh (CPU parity unaffected).
        self.w1_EFD = nnx.Param(
            jax.random.normal(k1, (num_experts, hidden_dim, dim), param_dtype) * scale,
            logical_axes=EXPERT_GATE_UP_AXES,
        )
        self.w2_EDF = nnx.Param(
            jax.random.normal(k2, (num_experts, dim, hidden_dim), param_dtype) * dscale,
            logical_axes=EXPERT_DOWN_AXES,
        )
        self.w3_EFD = nnx.Param(
            jax.random.normal(k3, (num_experts, hidden_dim, dim), param_dtype) * scale,
            logical_axes=EXPERT_GATE_UP_AXES,
        )
        self.dispatcher = dispatcher
        self.compute_dtype = compute_dtype

    def _experts_forward(self, x_RD: Array, counts_E: Array) -> Array:
        """Grouped SwiGLU over expert-contiguous tokens via ``ragged_dot``."""
        cdt = self.compute_dtype
        x = x_RD if cdt is None else x_RD.astype(cdt)
        w1 = self.w1_EFD[...]
        w2 = self.w2_EDF[...]
        w3 = self.w3_EFD[...]
        if cdt is not None:
            w1, w2, w3 = w1.astype(cdt), w2.astype(cdt), w3.astype(cdt)
        # (R,D) @ (E,D,F) -> (R,F); ragged over expert groups.
        h_RF = jax.nn.silu(jax.lax.ragged_dot(x, jnp.swapaxes(w1, -1, -2), counts_E))
        h_RF = h_RF * jax.lax.ragged_dot(x, jnp.swapaxes(w3, -1, -2), counts_E)
        out_RD = jax.lax.ragged_dot(h_RF, jnp.swapaxes(w2, -1, -2), counts_E)
        return out_RD.astype(x_RD.dtype)

    def __call__(
        self,
        x_BLD: Array,
        topk_scores_BLK: Array,
        topk_expert_ids_BLK: Array,
    ) -> Array:
        """Dispatch tokens to experts, grouped-compute, and combine back."""
        B, L, D = x_BLD.shape
        K = topk_scores_BLK.shape[-1]
        x_TD = x_BLD.reshape(B * L, D)
        topk_scores_TK = topk_scores_BLK.reshape(B * L, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.reshape(B * L, K)

        routed_input_RD, counts_E, metadata = self.dispatcher.dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK
        )
        routed_output_RD = self._experts_forward(routed_input_RD, counts_E)
        out_TD = self.dispatcher.combine(routed_output_RD, metadata, x_TD)
        return out_TD.reshape(B, L, D)
