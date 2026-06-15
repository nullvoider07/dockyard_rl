"""EP-aware MoE refit: native GroupedExperts -> per-expert HF tensors (J8d).

JAX mirror of ``models/dtensor/moe/refit.py``. The native fused experts
(``w1_EFD`` gate, ``w3_EFD`` up, ``w2_EDF`` down) must reach the inference
backend (vLLM/SGLang) as the standard per-expert checkpoint layout
``...mlp.experts.{i}.{gate,up,down}_proj.weight``, which the ``FusedMoE`` loader
re-fuses. Refit therefore unbinds the fused stack on dim 0 (the EP gather, when
ep>1, is the injected ``materialize`` — device-bound HV; identity on a single
device). No transpose: the grouped layout ``(E, out, in)`` unbinds to the HF
``nn.Linear.weight`` ``(out, in)`` directly.

The router gate ``...mlp.router.gate.kernel`` ``(D,E)`` maps back to HF
``...mlp.gate.weight`` ``(E,D)`` (transposed); all other params follow the dense
J5 refit (kernel transpose / scale passthrough). The pure name/shape/unbind logic
is CPU-validatable; the ep>1 gather + live vLLM load is hardware-deferred.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

from flax import nnx

from dockyard_rl.models.jax.refit import _TORCH_DTYPE_NAME, _hf_oriented, _jax_to_torch
from dockyard_rl.models.jax.weights import nnx_path_to_hf_name

# Native fused-expert leaf -> HF per-expert projection (SwiGLU: w1=gate, w3=up, w2=down).
_GROUPED_EXPERT_SUFFIXES: dict[str, str] = {
    "w1_EFD": "gate_proj",
    "w3_EFD": "up_proj",
    "w2_EDF": "down_proj",
}
_EXPERT_LEAVES = tuple(_GROUPED_EXPERT_SUFFIXES)


def is_grouped_expert_key(name: str) -> bool:
    """Whether ``name`` is a native fused GroupedExperts param (``...experts.wX_E**``)."""
    return name.rsplit(".", 1)[-1] in _GROUPED_EXPERT_SUFFIXES


def expand_grouped_expert_key(name: str, expert_index: int) -> str:
    """``...mlp.experts.w1_EFD`` + ``i`` -> ``...mlp.experts.{i}.gate_proj.weight``."""
    prefix, suffix = name.rsplit(".", 1)
    return f"{prefix}.{expert_index}.{_GROUPED_EXPERT_SUFFIXES[suffix]}.weight"


def expand_grouped_expert_info(
    name: str, shape: tuple[int, ...]
) -> list[tuple[str, tuple[int, ...]]]:
    """Fused ``(E, *per_expert)`` metadata -> one ``(per_expert_name, shape)`` per expert."""
    return [(expand_grouped_expert_key(name, i), tuple(shape[1:])) for i in range(shape[0])]


T = TypeVar("T")


def iter_expanded_refit_tensors(
    items: Iterable[tuple[str, T]],
    materialize: Callable[[T], Any],
) -> Iterator[tuple[str, Any]]:
    """Flat-map a refit stream, expanding fused-expert params per-expert.

    Fused-expert entries are ``materialize``d (the EP ``full_tensor`` gather when
    ep>1; identity on a single device) then unbound on dim 0; all other entries
    pass through untouched.
    """
    for name, tensor in items:
        if is_grouped_expert_key(name):
            full = materialize(tensor)
            for i in range(full.shape[0]):
                yield expand_grouped_expert_key(name, i), full[i]
        else:
            yield name, tensor


def _base_refit_stream(model: nnx.Module) -> Iterator[tuple[str, Any]]:
    """HF-named JAX-array stream, experts kept FUSED (``...experts.w1_EFD``)."""
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    for path, var in flat:
        p = tuple(path)
        leaf = p[-1]
        if leaf in _EXPERT_LEAVES:
            # fused name carrying the leaf, so iter_expanded_refit_tensors expands it
            yield ".".join(str(c) for c in p), var[...]
        elif leaf == "kernel" and p[-3:-1] == ("router", "gate"):
            name = ".".join(str(c) for c in p[:-3]) + ".gate.weight"
            yield name, var[...].T
        else:
            yield nnx_path_to_hf_name(p), _hf_oriented(p, var[...])


def iter_moe_refit_state_dict(
    model: nnx.Module,
    *,
    materialize: Optional[Callable[[Any], Any]] = None,
    to_torch: bool = True,
) -> Iterator[tuple[str, Any]]:
    """Yield ``(hf_name, tensor)`` for a Qwen3-MoE model, experts expanded per-expert.

    ``materialize`` performs the ep>1 gather on fused-expert arrays (device-bound;
    HV); defaults to identity (single device). ``to_torch`` converts the final
    arrays to torch CPU tensors for the unchanged packed-broadcast producer.
    """
    mat = materialize if materialize is not None else (lambda t: t)
    for name, arr in iter_expanded_refit_tensors(_base_refit_stream(model), mat):
        yield name, (_jax_to_torch(arr) if to_torch else arr)


def prepare_moe_refit_info(
    model: nnx.Module,
    *,
    materialize: Optional[Callable[[Any], Any]] = None,
    dtype: Optional[str] = None,
) -> dict[str, Any]:
    """``{hf_name: (torch.Size, dtype_name)}`` matching the expanded MoE refit stream.

    Same per-expert expansion + transposes as ``iter_moe_refit_state_dict`` (so the
    declared map matches the streamed tensors), reading shapes off the JAX arrays
    without converting to torch tensors.
    """
    import torch

    out: dict[str, Any] = {}
    mat = materialize if materialize is not None else (lambda t: t)
    for name, arr in iter_expanded_refit_tensors(_base_refit_stream(model), mat):
        dt = dtype or _TORCH_DTYPE_NAME.get(str(arr.dtype), str(arr.dtype))
        out[name] = (torch.Size(tuple(arr.shape)), dt)
    return out
