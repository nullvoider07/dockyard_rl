"""MoE block: router + grouped routed experts (+ optional shared experts), NNX.

JAX mirror of ``models/dtensor/moe/block.py``. Forward:
  1. router -> per-token top-K expert ids + gating weights + full scores;
  2. per-expert token counts (drives the load-balance counter);
  3. routed experts (dispatch -> grouped compute -> combine);
  4. optional shared (always-on) expert, summed with the routed output.

Aux-loss-free load balancing (https://arxiv.org/abs/2408.15664): when
``load_balance_coeff`` is set, an ``expert_bias_E`` buffer biases the routing
CHOICE (not the gating weights); its update is the external step in
``load_balance.py``. ``tokens_per_expert_E`` accumulates per-expert load each
forward — the signal that updater reads. Both are ``Buffer`` (non-trained) state.

Shape suffixes: B=batch, L=seq, D=model dim, E=experts, K=top-k.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.moe import Buffer
from dockyard_rl.models.jax.moe.experts import GroupedExperts
from dockyard_rl.models.jax.moe.router import TokenChoiceTopKRouter

Array = jax.Array


def per_expert_counts(topk_expert_ids_BLK: Array, num_experts: int) -> Array:
    """Per-expert token counts ``(E,)`` (float32) summed over batch+sequence."""
    oh = jax.nn.one_hot(topk_expert_ids_BLK.reshape(-1), num_experts, dtype=jnp.float32)
    return jnp.sum(oh, axis=0)


class MoEBlock(nnx.Module):
    """Router + grouped routed experts (+ optional shared experts).

    Args:
        router: The gating network.
        experts: Grouped routed experts.
        num_experts: Total expert count (E) — for the load-balance buffers.
        shared_experts: Optional always-on expert (dense FFN) summed with the
            routed output.
        load_balance_coeff: If set (> 0), allocates the aux-loss-free
            ``expert_bias_E`` routing-bias buffer (updater is external).
    """

    def __init__(
        self,
        router: TokenChoiceTopKRouter,
        experts: GroupedExperts,
        *,
        num_experts: int,
        shared_experts: Optional[nnx.Module] = None,
        load_balance_coeff: Optional[float] = None,
    ) -> None:
        self.router = router
        self.experts = experts
        self.shared_experts = shared_experts
        self.num_experts = num_experts

        self.load_balance_coeff = load_balance_coeff
        if load_balance_coeff is not None:
            if load_balance_coeff <= 0.0:
                raise ValueError(f"load_balance_coeff must be > 0, got {load_balance_coeff}")
            self.expert_bias_E = Buffer(jnp.zeros((num_experts,), dtype=jnp.float32))
        else:
            self.expert_bias_E = None
        # Usage counter; accumulated each forward, read by the external updater.
        self.tokens_per_expert_E = Buffer(jnp.zeros((num_experts,), dtype=jnp.float32))

    def __call__(self, x_BLD: Array) -> Array:
        bias = None if self.expert_bias_E is None else self.expert_bias_E[...]
        topk_scores_BLK, topk_expert_ids_BLK, _scores_BLE = self.router(x_BLD, bias)

        counts_E = per_expert_counts(topk_expert_ids_BLK, self.num_experts)
        self.tokens_per_expert_E[...] = self.tokens_per_expert_E[...] + counts_E

        out_BLD = self.experts(x_BLD, topk_scores_BLK, topk_expert_ids_BLK)
        if self.shared_experts is not None:
            out_BLD = out_BLD + self.shared_experts(x_BLD)
        return out_BLD
