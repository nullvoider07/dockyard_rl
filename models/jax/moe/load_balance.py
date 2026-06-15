"""Aux-loss-free load-balance bias updater for MoE routing, JAX.

JAX mirror of ``models/dtensor/moe/load_balance.py``. DeepSeek aux-loss-free load
balancing (https://arxiv.org/abs/2408.15664): a per-expert bias ``expert_bias_E``
nudges the routing CHOICE (added to the router scores before top-k, not to the
gating weights), stepped each optimizer update from the observed per-expert load —
under-loaded experts get a positive bump, over-loaded a negative one, by a fixed
magnitude ``load_balance_coeff``.

``compute_expert_bias_delta`` (the math) is pure and CPU-validatable. The
cross-rank reduction that makes ``tokens_per_expert_E`` a GLOBAL load before the
step is EP/data-parallel-aware (an ``all_reduce`` over the dp/cp axes) and
hardware-deferred — injected as ``reduce_tokens_fn``. With no reduction (single
device / CPU) the step is exact for that device.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.moe.block import MoEBlock

Array = jax.Array
ReduceTokensFn = Callable[[Array], Array]


def compute_expert_bias_delta(tokens_per_expert_E: Array, load_balance_coeff: float) -> Array:
    """Sign-based, zero-centered expert-bias step.

    ``delta = coeff * sign(mean_load - load)`` moves under-loaded experts up and
    over-loaded down by ``coeff``; subtracting the mean conserves the total bias
    (it only redistributes routing pressure).
    """
    tokens = tokens_per_expert_E.astype(jnp.float32)
    delta = load_balance_coeff * jnp.sign(jnp.mean(tokens) - tokens)
    return delta - jnp.mean(delta)


def iter_lb_moe_blocks(model: nnx.Module) -> Iterator[MoEBlock]:
    """Yield every ``MoEBlock`` with aux-loss-free load balancing enabled.

    ``nnx.iter_modules`` recurses the whole graph, so nested blocks are found.
    """
    for _path, module in nnx.iter_modules(model):
        if isinstance(module, MoEBlock) and module.expert_bias_E is not None:
            yield module


def update_expert_biases(
    blocks: Sequence[MoEBlock],
    reduce_tokens_fn: Optional[ReduceTokensFn] = None,
) -> None:
    """Step each block's ``expert_bias_E`` from its accumulated token load.

    For every block: optionally reduce the local ``tokens_per_expert_E`` to a
    global load (``reduce_tokens_fn``, device-bound), apply the sign step, and
    zero the counter for the next window. When the reduction yields identical
    global counts on every device, all devices step ``expert_bias_E`` identically
    and stay in sync. The buffers are non-trained ``Buffer`` state (no grad).
    """
    for block in blocks:
        assert block.load_balance_coeff is not None and block.expert_bias_E is not None
        tokens = block.tokens_per_expert_E[...]
        if reduce_tokens_fn is not None:
            tokens = reduce_tokens_fn(tokens)
        delta = compute_expert_bias_delta(tokens, block.load_balance_coeff)
        block.expert_bias_E[...] = block.expert_bias_E[...] + delta.astype(
            block.expert_bias_E[...].dtype
        )
        block.tokens_per_expert_E[...] = jnp.zeros_like(block.tokens_per_expert_E[...])
