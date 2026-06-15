"""Microbatched GRPO train step: value_and_grad + grad accumulation + optax.

Recompute-logprobs → core GRPO loss (J3) → grads over the NNX params → optax
update. Each microbatch loss is normalized by the *global* valid-seq/token
counts, so summing microbatch grads equals the global-batch gradient and a
single optax update is applied to the accumulated grad.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from dockyard_rl.models.jax.loss import clipped_pg_loss, logprobs_from_logits

Array = jax.Array


def training_loss_fn(
    model: nnx.Module,
    mb: Mapping[str, Array],
    global_valid_seqs: Array,
    global_valid_toks: Array,
    loss_cfg: Any,
    temperature: float = 1.0,
) -> tuple[Array, dict[str, Array]]:
    """Forward → recompute next-token logprobs → core GRPO loss.

    ``temperature`` divides the logits before the log-softmax to match the
    generation temperature, mirroring the torch worker's
    ``_apply_temperature_scaling`` (dtensor_policy_worker.py:608); 1.0 is a no-op.
    """
    logits = model(mb["input_ids"])
    if temperature != 1.0:
        logits = logits / temperature
    curr_logprobs = logprobs_from_logits(logits, mb["input_ids"])
    return clipped_pg_loss(curr_logprobs, mb, global_valid_seqs, global_valid_toks, loss_cfg)


def split_microbatches(data: Mapping[str, Array], mbs: int) -> list[dict[str, Array]]:
    """Slice every column into microbatches of ``mbs`` rows along the batch axis."""
    batch = next(iter(data.values())).shape[0]
    return [{k: v[i : i + mbs] for k, v in data.items()} for i in range(0, batch, mbs)]


def global_valid_counts(data: Mapping[str, Array]) -> tuple[Array, Array]:
    """Global valid-seq and valid-token counts over the whole batch.

    ``gvs`` = sum(sample_mask); ``gvt`` = sum(token_mask[:,1:] * sample_mask),
    matching the slicing used inside the loss.
    """
    sample_mask = data["sample_mask"]
    token_mask = data["token_mask"][:, 1:]
    mask = token_mask * sample_mask[:, None]
    return jnp.sum(sample_mask), jnp.sum(mask)


def accumulate_grads(
    model: nnx.Module,
    data: Mapping[str, Array],
    global_valid_seqs: Array,
    global_valid_toks: Array,
    loss_cfg: Any,
    mbs: int,
    temperature: float = 1.0,
) -> tuple[Any, dict[str, Array]]:
    """Sum grads (and metrics) over microbatches; one update applied by caller."""
    grads_accum: Any = None
    metric_accum: dict[str, Array] = {}
    for mb in split_microbatches(data, mbs):
        (_loss, metrics), grads = nnx.value_and_grad(training_loss_fn, has_aux=True)(
            model, mb, global_valid_seqs, global_valid_toks, loss_cfg, temperature
        )
        grads_accum = grads if grads_accum is None else jax.tree.map(jnp.add, grads_accum, grads)
        if not metric_accum:
            metric_accum = dict(metrics)
        else:
            metric_accum = {k: metric_accum[k] + metrics[k] for k in metrics}
    return grads_accum, metric_accum


def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    data: Mapping[str, Array],
    loss_cfg: Any,
    mbs: int,
    temperature: float = 1.0,
    pre_optim_step_fn: Optional[Callable[[], None]] = None,
) -> dict[str, Array]:
    """One optimizer step over a global batch; returns metrics + grad_norm.

    ``pre_optim_step_fn`` runs once after grad accumulation and before the optax
    parameter update — the JAX analogue of torch's ``register_step_pre_hook``
    (fires once per optimizer update, composing with grad accumulation). The MoE
    aux-loss-free expert-bias updater is wired here (reads the per-expert load
    accumulated over the microbatch forwards, steps ``expert_bias_E``, zeroes the
    counter); ``None`` for the dense path.
    """
    gvs, gvt = global_valid_counts(data)
    grads, metrics = accumulate_grads(model, data, gvs, gvt, loss_cfg, mbs, temperature)
    grad_norm = optax.tree.norm(grads)
    if pre_optim_step_fn is not None:
        pre_optim_step_fn()
    optimizer.update(model, grads)
    metrics = dict(metrics)
    metrics["grad_norm"] = grad_norm
    return metrics
