"""Flax NNX model definitions for the JAX trainer backend.

Dense architectures first (Qwen3 dense in J1); MoE in J8; linear/hybrid
attention in J11. See ``handoff/jax-trainer-replatform-plan.md``.
"""

from dockyard_rl.models.jax.models.qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
)

__all__ = ["Qwen3Config", "Qwen3ForCausalLM", "Qwen3Model"]
