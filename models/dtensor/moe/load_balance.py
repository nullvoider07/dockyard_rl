"""Aux-loss-free load-balance bias updater for MoE routing.

DeepSeek aux-loss-free load balancing (https://arxiv.org/abs/2408.15664):
instead of an auxiliary load-balance loss, a per-expert bias ``expert_bias_E``
nudges the routing CHOICE (added to the router logits before top-k, not to the
gating weights). The bias is stepped each optimizer step from the observed
per-expert token load: under-loaded experts get a positive bump, over-loaded a
negative one, by a fixed magnitude ``load_balance_coeff``.

The update is a sign step, registered as an optimizer ``step`` pre-hook so it
fires once per optimizer update (after gradient accumulation), matching
torchtitan's flow (pytorch/torchtitan, BSD-3-Clause; Copyright (c) Meta
Platforms, Inc. and affiliates).

``compute_expert_bias_delta`` (the math) is pure and CPU-tested. The cross-rank
reduction that makes ``tokens_per_expert_E`` a GLOBAL load before the step is
EP/data-parallel-aware and device-bound — injected as ``reduce_tokens_fn`` and
exercised at bring-up (hardware-deferred-validation.md, HV-14). With no
reduction (single rank / CPU) the step is exact for that rank.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional, Sequence

import torch

from dockyard_rl.models.dtensor.moe.block import MoEBlock

ReduceTokensFn = Callable[[torch.Tensor], torch.Tensor]


def compute_expert_bias_delta(
    tokens_per_expert_E: torch.Tensor, load_balance_coeff: float
) -> torch.Tensor:
    """Sign-based, zero-centered expert-bias step.

    ``delta = coeff * sign(mean_load - load)`` moves under-loaded experts up and
    over-loaded experts down by ``coeff``; subtracting the mean keeps the total
    bias conserved (it only redistributes routing pressure). Returns a float
    tensor shaped like ``tokens_per_expert_E``.
    """
    tokens = tokens_per_expert_E.float()
    delta = load_balance_coeff * torch.sign(tokens.mean() - tokens)
    return delta - delta.mean()


def iter_lb_moe_blocks(model: torch.nn.Module) -> Iterator[MoEBlock]:
    """Yield every ``MoEBlock`` with aux-loss-free load balancing enabled."""
    for module in model.modules():
        if (
            isinstance(module, MoEBlock)
            and module.load_balance_coeff is not None
            and module.expert_bias_E is not None
        ):
            yield module


def update_expert_biases(
    blocks: Sequence[MoEBlock],
    reduce_tokens_fn: Optional[ReduceTokensFn] = None,
) -> None:
    """Step each block's ``expert_bias_E`` from its accumulated token load.

    For every block: optionally reduce the local ``tokens_per_expert_E`` to a
    global load (``reduce_tokens_fn``, device-bound), apply the sign step, and
    zero the counter for the next window. Runs under ``no_grad``; the bias is a
    buffer, not a trained parameter. When the reduction yields identical global
    counts on every rank, all ranks step ``expert_bias_E`` identically and stay
    in sync.
    """
    for block in blocks:
        assert block.load_balance_coeff is not None
        assert isinstance(block.tokens_per_expert_E, torch.Tensor)
        assert isinstance(block.expert_bias_E, torch.Tensor)
        tokens: torch.Tensor = block.tokens_per_expert_E
        if reduce_tokens_fn is not None:
            tokens = reduce_tokens_fn(tokens)
        delta = compute_expert_bias_delta(tokens, block.load_balance_coeff)
        with torch.no_grad():  # type: ignore[attr-defined]
            block.expert_bias_E.add_(delta.to(block.expert_bias_E.dtype))  # type: ignore[attr-defined]
            block.tokens_per_expert_E.zero_()  # type: ignore[attr-defined]


def register_expert_bias_update_hook(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    reduce_tokens_fn: Optional[ReduceTokensFn] = None,
) -> bool:
    """Register the expert-bias updater as an optimizer ``step`` pre-hook.

    No-op (returns ``False``) when the model has no load-balanced ``MoEBlock``.
    The pre-hook fires before each ``optimizer.step()`` — once per optimizer
    update, so it composes with gradient accumulation. ``reduce_tokens_fn`` is
    the cross-rank token-load reduction (device-bound; HV-14).
    """
    blocks = list(iter_lb_moe_blocks(model))
    if not blocks:
        return False
    optimizer.register_step_pre_hook(
        lambda *args, **kwargs: update_expert_biases(blocks, reduce_tokens_fn)
    )
    return True
