"""Token-choice top-K router for MoE (the gating network), Flax NNX.

JAX mirror of ``models/dtensor/moe/router.py``: each token is scored against all
experts by a small bias-free gate, then routed to its top-K experts, with
optional node-limited (group-limited) routing for large-expert-count models. The
gate runs in float32 for load-balance stability (matching the torch CUDA
autocast override). Pure tensor logic — CPU-parity-validatable.

Shape suffixes: B=batch, L=seq, D=model dim, E=num experts, K=top-k.
"""

from __future__ import annotations

from typing import Literal, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.layers import Linear

Array = jax.Array
ScoreFunc = Literal["softmax", "sigmoid"]


class TokenChoiceTopKRouter(nnx.Module):
    """Top-K token-choice gating.

    Args mirror the torch ``TokenChoiceTopKRouter``: ``score_func`` selects the
    per-expert (``sigmoid``) vs normalized (``softmax``) gate activation;
    ``route_norm`` renormalizes the selected top-K weights to sum to 1;
    ``route_scale`` multiplies the final weights; ``num_expert_groups`` /
    ``num_limited_groups`` enable node-limited routing (group score = sum of the
    group's top-2 expert scores).
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        *,
        top_k: int = 1,
        score_func: ScoreFunc = "sigmoid",
        route_norm: bool = False,
        route_scale: float = 1.0,
        num_expert_groups: Optional[int] = None,
        num_limited_groups: Optional[int] = None,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k({top_k}) must be in [1, num_experts({num_experts})]")
        if num_expert_groups is not None and num_limited_groups is None:
            raise ValueError("num_limited_groups must be set when num_expert_groups is set")
        # Router gate is replicated (tiny; every token needs full routing logits).
        self.gate = Linear(dim, num_experts, rngs=rngs, param_dtype=param_dtype)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups

    def _node_limited_scores(self, scores_BLE: Array) -> Array:
        """Mask experts outside the top ``num_limited_groups`` groups to -inf."""
        assert self.num_expert_groups is not None and self.num_limited_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by "
                f"num_expert_groups({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group({experts_per_group}) must be >= 2")
        grouped = scores_BLE.reshape(
            *scores_BLE.shape[:-1], self.num_expert_groups, experts_per_group
        )
        top2, _ = jax.lax.top_k(grouped, 2)
        group_scores = jnp.sum(top2, axis=-1)  # (..., num_expert_groups)
        # Keep the top num_limited_groups groups; mask the rest to -inf. one_hot
        # over the kept indices then OR across the K kept slots -> per-group mask.
        _, keep_idx = jax.lax.top_k(group_scores, self.num_limited_groups)
        keep = jax.nn.one_hot(keep_idx, self.num_expert_groups, dtype=bool).any(axis=-2)
        neg_inf = jnp.asarray(jnp.finfo(jnp.float32).min, dtype=grouped.dtype)
        masked = jnp.where(keep[..., :, None], grouped, neg_inf)
        return masked.reshape(scores_BLE.shape)

    def __call__(
        self, x_BLD: Array, expert_bias_E: Optional[Array] = None
    ) -> tuple[Array, Array, Array]:
        """Route tokens to top-K experts.

        Returns ``(topk_scores_BLK, topk_expert_ids_BLK, scores_BLE)``. The
        ``expert_bias_E`` (aux-loss-free load balance) shifts the top-K CHOICE
        only; the returned gating weights come from the unbiased scores.
        """
        # Gate in float32 for load-balance stability (the torch path's CUDA
        # autocast override); a no-op when params are already float32.
        scores_BLE = self.gate(x_BLD.astype(jnp.float32))

        if self.score_func == "sigmoid":
            scores_BLE = jax.nn.sigmoid(scores_BLE)
        elif self.score_func == "softmax":
            scores_BLE = jax.nn.softmax(scores_BLE, axis=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice_BLE = (
            scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
        )
        if self.num_expert_groups is not None:
            scores_for_choice_BLE = self._node_limited_scores(scores_for_choice_BLE)

        _, topk_expert_ids_BLK = jax.lax.top_k(scores_for_choice_BLE, self.top_k)
        # Gating weights derive from the UNBIASED scores (bias is routing-only).
        topk_scores_BLK = jnp.take_along_axis(scores_BLE, topk_expert_ids_BLK, axis=-1)

        if self.route_norm:
            denom = jnp.sum(topk_scores_BLK, axis=-1, keepdims=True) + 1e-20
            topk_scores_BLK = topk_scores_BLK / denom
        topk_scores_BLK = topk_scores_BLK * self.route_scale

        return topk_scores_BLK, topk_expert_ids_BLK.astype(jnp.int32), scores_BLE
