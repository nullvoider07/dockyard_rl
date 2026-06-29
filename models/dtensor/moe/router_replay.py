"""Trainer-side MoE router-replay binding (#2908).

During generation an MoE router selects top-K experts per token. Re-running the
router at train/logprob time can pick different experts (batch composition,
padding, kernel/dtype nondeterminism), so the same token gets a different
logprob than it was generated with — biasing the importance-sampling correction
and inflating the train/generation logprob error that dockyard uses to filter
rollouts. Router-replay forces the trainer's MoE forward to reuse generation's
recorded expert selection.

This module is the binding layer: ``bind_router_replay`` walks the model's
``MoEBlock`` modules in module order, binds each block's per-layer route slice
(set-and-consume), and clears them on exit. The router consumes the bound slice
(``router.py``), forcing ``topk_expert_ids`` to the recorded ids while the
gating weights still come from the train model's own gate (gradient intact).

The recorded routing arrives as ``[B, T, L_moe, K]`` — per token, per MoE layer,
the recorded top-K expert ids (see ``router_capture.align_routed_expert_indices``
for the capture/alignment). ``L_moe`` MUST equal the number of ``MoEBlock``
modules; mismatch is a hard error (a silent mis-bind would replay the wrong
layer's routing). The binding is pure module-attribute mutation — CPU-testable.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import torch
from torch import nn

from dockyard_rl.models.dtensor.moe.block import MoEBlock


def iter_moe_blocks(model: nn.Module) -> list[MoEBlock]:
    """The model's ``MoEBlock`` modules, in ``named_modules`` (layer) order."""
    return [m for _, m in model.named_modules() if isinstance(m, MoEBlock)]


def count_moe_blocks(model: nn.Module) -> int:
    """Number of ``MoEBlock`` modules in the model (0 for a dense model)."""
    return len(iter_moe_blocks(model))


@contextmanager
def bind_router_replay(
    model: nn.Module, routed_experts_BTLK: torch.Tensor
) -> Generator[None, None, None]:
    """Bind recorded routing onto the model's MoE blocks for one forward.

    Args:
        model: The policy model containing ``MoEBlock`` modules.
        routed_experts_BTLK: Recorded routing ``[B, T, L_moe, K]`` (int) aligned
            to the forward's token layout. ``L_moe`` must equal the MoEBlock
            count; ``T`` and ``B`` must match the forward's sequence/batch.

    Raises:
        ValueError: if the model has no MoE blocks, the tensor is not 4D, or the
            recorded MoE-layer count does not match the model's MoEBlock count.
    """
    blocks = iter_moe_blocks(model)
    if not blocks:
        raise ValueError(
            "router replay requested but the model has no MoEBlock modules"
        )
    if routed_experts_BTLK.dim() != 4:
        raise ValueError(
            "routed_experts must be [B, T, L_moe, K], got "
            f"{tuple(routed_experts_BTLK.shape)}"
        )
    n_layers = int(routed_experts_BTLK.shape[2])
    if n_layers != len(blocks):
        raise ValueError(
            f"recorded MoE-layer count ({n_layers}) does not match the model's "
            f"MoEBlock count ({len(blocks)}); router replay cannot bind layers"
        )
    try:
        for layer_idx, block in enumerate(blocks):
            block._replay_route_BLK = routed_experts_BTLK[:, :, layer_idx, :]
        yield
    finally:
        for block in blocks:
            block._replay_route_BLK = None


def resolve_router_replay_enabled(policy_cfg: Any) -> bool:
    """Read ``policy.router_replay.enabled`` from a dict or attr-style config."""
    if isinstance(policy_cfg, dict):
        rr = policy_cfg.get("router_replay")
    else:
        rr = getattr(policy_cfg, "router_replay", None)
    if not rr:
        return False
    if isinstance(rr, dict):
        return bool(rr.get("enabled", False))
    return bool(getattr(rr, "enabled", False))


def validate_router_replay_config(enabled: bool, model: nn.Module) -> None:
    """Raise if router replay is enabled on a model without MoE blocks."""
    if enabled and count_moe_blocks(model) == 0:
        raise ValueError(
            "policy.router_replay.enabled=true requires an MoE model, but the "
            "policy model has no MoEBlock modules"
        )
