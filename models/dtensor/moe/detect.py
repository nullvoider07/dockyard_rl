"""Model-agnostic MoE detection.

The v2 worker selects the expert-parallel path when the model is MoE. Following
upstream's heuristic, a model is treated as MoE when any of its state-dict keys
names an expert parameter — no per-architecture registry needed.
"""

from __future__ import annotations

from typing import Iterable

# Matches routed-expert params across HF / native MoE checkpoints, e.g.
# Qwen3-MoE / Mixtral ``...mlp.experts.*`` and the native GroupedExperts
# (``experts.w1_EFD`` ...). Uses the permissive upstream heuristic ("expert" in
# key) deliberately: a false negative (MoE model wrongly taking the dense path)
# would mis-shard silently, whereas a false positive is caught early (no expert
# params to shard). Case-insensitive for robustness across naming.
_EXPERT_MARKER = "expert"


def is_moe_state_dict(state_dict_keys: Iterable[str]) -> bool:
    """Whether a state dict belongs to a MoE model (has routed-expert params).

    Args:
        state_dict_keys: The model's parameter/buffer key names.

    Returns:
        True if any key names a routed-expert parameter.
    """
    return any(_EXPERT_MARKER in key.lower() for key in state_dict_keys)
