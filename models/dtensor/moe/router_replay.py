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


@contextmanager
def router_replay_context(
    model: nn.Module,
    routed_experts_BTLK: Any,
    *,
    enabled: bool,
    seq_packing: bool = False,
    context_parallel: bool = False,
) -> Generator[None, None, None]:
    """Bind router replay for one forward, or a null context when inapplicable.

    The single hook the policy worker wraps around its MoE forward so the same
    binding covers both the prev-logprob recompute (``get_logprobs``) and the
    train forward. Yields without binding when replay is disabled or the batch
    carries no ``routed_experts`` column (the default training path is therefore
    byte-unchanged).

    Sequence packing and context parallelism re-lay-out the token/sequence axis
    (packed rows, seq-sharded buffers), so the recorded routing would have to be
    transformed identically to stay aligned — that layout work is HV-47. Until
    then this refuses those combinations loudly rather than silently replaying a
    misaligned route.
    """
    if not enabled or routed_experts_BTLK is None:
        yield
        return
    if seq_packing or context_parallel:
        raise NotImplementedError(
            "router replay does not yet support sequence packing or context "
            "parallelism (the recorded routing must be re-laid-out to match the "
            "packed/seq-sharded tokens; HV-47). Disable policy.sequence_packing "
            "/ context parallel, or policy.router_replay."
        )
    with bind_router_replay(model, routed_experts_BTLK):
        yield


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


def validate_router_replay_generation_compat(
    *,
    enabled: bool,
    backend: str,
    data_plane_enabled: bool,
    vllm_cfg: Any,
    vllm_kwargs: Any,
) -> None:
    """Fail fast on generation/orchestration configs incompatible with capture.

    Pure config-logic (no model / cluster objects) so the orchestrator can
    pre-validate at setup and surface a dockyard-native message instead of a
    deep failure later. Inert when ``enabled`` is false.

    - Non-vLLM backend: routed-expert capture reads vLLM's
      ``CompletionOutput.routed_experts``; other backends do not surface it.
    - data_plane async: the 4D ``[B, T, L, K]`` route column does not ride the
      data-plane bulk-field set (HV-47).
    - vLLM engine config: ``enable_return_routed_experts`` is rejected by vLLM
      under ``async_scheduling`` / pipeline parallelism / prefill- or
      decode-context parallelism (validated scope is TP/EP/DP + prefix caching).
    """
    if not enabled:
        return

    if backend != "vllm":
        raise ValueError(
            "policy.router_replay.enabled=true requires the vLLM generation "
            "backend: routed-expert capture reads vLLM CompletionOutput."
            f"routed_experts, which the {backend!r} backend does not surface. "
            "Use backend=vllm or disable router replay."
        )

    if data_plane_enabled:
        raise ValueError(
            "policy.router_replay.enabled=true is not supported on the "
            "data_plane async path: the per-token routed_experts column does "
            "not ride the data-plane bulk-field set (HV-47). Disable data_plane "
            "or router replay."
        )

    cfg = vllm_cfg or {}
    kwargs = vllm_kwargs or {}

    def _val(name: str, default: int) -> int:
        return kwargs.get(name, cfg.get(name, default))

    rejected: list[str] = []
    if cfg.get("pipeline_parallel_size", 1) > 1 or _val("pipeline_parallel_size", 1) > 1:
        rejected.append("pipeline_parallel_size>1")
    if kwargs.get("async_scheduling", False):
        rejected.append("async_scheduling")
    if _val("prefill_context_parallel_size", 1) > 1:
        rejected.append("prefill_context_parallel_size>1")
    if _val("decode_context_parallel_size", 1) > 1:
        rejected.append("decode_context_parallel_size>1")
    if rejected:
        raise ValueError(
            "policy.router_replay.enabled=true is rejected by vLLM under "
            f"{', '.join(rejected)} (enable_return_routed_experts is only "
            "validated for tensor/expert/data parallelism + prefix caching). "
            "Disable the listed vLLM feature(s) or router replay."
        )
