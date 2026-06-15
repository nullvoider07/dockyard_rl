"""EP-aware refit: native GroupedExperts -> per-expert HF weight expansion.

The native ``GroupedExperts`` (B.4) fuses an MoE layer's experts into stacked 3D
params (``w1_EFD`` ``[E,F,D]`` gate, ``w3_EFD`` ``[E,F,D]`` up, ``w2_EDF``
``[E,D,F]`` down), sharded ``Shard(0)`` over the EP axis. The inference backend
(vLLM / SGLang) loads the standard HF checkpoint layout — one tensor per expert,
named ``...mlp.experts.{i}.{gate,up,down}_proj.weight`` — and fuses internally.

Refit must therefore RESHARD across the trainer->inference boundary: gather the
EP-sharded stack to a full tensor, then unbind dim 0 into per-expert HF-named
tensors. This module is the pure name/shape/unbind logic; the EP gather itself
(``DTensor.full_tensor()``) is device-bound and lives in the v2 worker.

Expert ordering: ``full_tensor()`` over ``Shard(0)``-on-ep reconstructs experts
in global index order (the EP partition is contiguous — see ``mesh.py``
``experts_for_rank``), so ``stack[i]`` is global expert ``i``. This depends on
the EP-axis ordering tracked as HV-1.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, TypeVar

# Native fused-expert param suffix -> HF per-expert projection name. The SwiGLU
# convention matches Mixtral / Qwen3-MoE: w1=gate (silu), w3=up (multiply),
# w2=down (output).
_GROUPED_EXPERT_SUFFIXES: dict[str, str] = {
    "w1_EFD": "gate_proj",
    "w3_EFD": "up_proj",
    "w2_EDF": "down_proj",
}


def is_grouped_expert_key(name: str) -> bool:
    """Whether ``name`` is a native fused GroupedExperts param (w1/w2/w3_E**)."""
    return name.rsplit(".", 1)[-1] in _GROUPED_EXPERT_SUFFIXES


def expand_grouped_expert_key(name: str, expert_index: int) -> str:
    """Map a fused-expert key + expert index to its per-expert HF name.

    ``...mlp.experts.w1_EFD`` + ``i`` -> ``...mlp.experts.{i}.gate_proj.weight``.
    """
    prefix, suffix = name.rsplit(".", 1)
    proj = _GROUPED_EXPERT_SUFFIXES[suffix]
    return f"{prefix}.{expert_index}.{proj}.weight"


def expand_grouped_expert_info(
    name: str, shape: tuple[int, ...]
) -> list[tuple[str, tuple[int, ...]]]:
    """Expand fused-expert metadata into per-expert ``(name, shape)`` entries.

    ``shape`` is the global fused shape ``(E, *per_expert)``; the result has one
    entry per expert with the leading expert dim dropped.
    """
    num_experts = shape[0]
    per_expert_shape = tuple(shape[1:])
    return [
        (expand_grouped_expert_key(name, i), per_expert_shape)
        for i in range(num_experts)
    ]


T = TypeVar("T")


def iter_expanded_refit_tensors(
    items: Iterable[tuple[str, T]],
    materialize: Callable[[T], "object"],
) -> Iterator[tuple[str, object]]:
    """Flat-map a refit state-dict stream, expanding fused-expert params.

    For each ``(name, tensor)``:
      - fused-expert param -> ``materialize`` it (EP gather to a full tensor),
        then yield one ``(per_expert_name, stack[i])`` per expert;
      - any other param -> passed through unchanged (NOT materialized, so the
        caller's existing per-item materialization still applies).

    ``materialize`` performs the device-bound EP gather (``full_tensor()``); it
    is injected so this function stays pure/testable. The unbound per-expert
    tensors are views into the gathered stack.
    """
    for name, tensor in items:
        if is_grouped_expert_key(name):
            full = materialize(tensor)
            for i in range(full.shape[0]):  # type: ignore[attr-defined]
                yield expand_grouped_expert_key(name, i), full[i]  # type: ignore[index]
        else:
            yield name, tensor
