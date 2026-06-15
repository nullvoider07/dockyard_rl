"""HF per-expert -> native fused-expert weight conversion (the load side).

The exact inverse of ``refit.py``: where refit UNBINDS a fused EP-sharded stack
into per-expert HF tensors for the inference backend, this STACKS an HF
checkpoint's per-expert Linears (``...mlp.experts.{i}.{gate,up,down}_proj.weight``)
into the native fused params (``...mlp.experts.w1_EFD`` etc.) that
``GroupedExperts`` consumes. Used when surgically replacing an HF MoE model's
per-expert MLP with the native grouped block.

Pure tensor restructuring (stack + rename), so it round-trips against refit and
is CPU-tested. Applying it to a live HF module (and the grouped-GEMM forward) is
device-bound and arch-specific — that surgery is separate.

The proj<->fused name convention mirrors ``refit.py`` (Mixtral / Qwen3-MoE):
gate_proj->w1_EFD, up_proj->w3_EFD, down_proj->w2_EDF.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import torch

# HF per-expert projection -> native fused param name.
_PROJ_TO_FUSED: dict[str, str] = {
    "gate_proj": "w1_EFD",
    "up_proj": "w3_EFD",
    "down_proj": "w2_EDF",
}

# Captures (experts-prefix, expert-index, proj-name) from an HF per-expert key,
# e.g. "model.layers.0.mlp.experts.3.gate_proj.weight".
_PER_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def is_hf_per_expert_key(name: str) -> bool:
    """Whether ``name`` is an HF per-expert projection weight."""
    return _PER_EXPERT_RE.match(name) is not None


def fuse_hf_expert_state_dict(
    items: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack HF per-expert weights into native fused params.

    Per-expert keys for one ``...experts`` prefix are collected and stacked along
    a new leading expert dim (in expert-index order) into the fused param. All
    other keys pass through unchanged. The number of experts is inferred per
    prefix from the indices present; a gap (missing expert index) raises.

    Returns a new state dict (not in-place).
    """
    # prefix -> proj -> {expert_idx: tensor}
    collected: dict[str, dict[str, dict[int, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    passthrough: dict[str, torch.Tensor] = {}

    for name, tensor in items:
        m = _PER_EXPERT_RE.match(name)
        if m is None:
            passthrough[name] = tensor
            continue
        collected[m["prefix"]][m["proj"]][int(m["idx"])] = tensor

    out: dict[str, torch.Tensor] = dict(passthrough)
    for prefix, by_proj in collected.items():
        for proj, by_idx in by_proj.items():
            num_experts = len(by_idx)
            indices = sorted(by_idx)
            if indices != list(range(num_experts)):
                raise ValueError(
                    f"non-contiguous expert indices for {prefix}.{proj}: "
                    f"{indices} (expected 0..{num_experts - 1})"
                )
            stacked = torch.stack([by_idx[i] for i in indices], dim=0)
            out[f"{prefix}.{_PROJ_TO_FUSED[proj]}"] = stacked

    return out
