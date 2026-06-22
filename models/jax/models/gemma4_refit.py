"""Gemma 4 refit seam: NNX params -> HF-named torch tensors (C7).

The inverse of ``load_hf_gemma4_state_dict``. Gemma 4's parameter tree has
multi-param modules and an extra nesting level that the generic dense name-map
(``nnx_path_to_hf_name``: one ``.weight`` leaf per module) cannot express, so it
would collapse distinct params onto a single name and mis-nest the PLE table:

  - the MoE ``Gemma4Router`` holds three params (``proj`` kernel + ``scale`` +
    ``per_expert_scale``) and ``Gemma4Experts`` two fused stacks (``gate_up_proj``,
    ``down_proj``); the generic map would emit ``router.weight`` / ``experts.weight``
    for each, colliding two params onto one name;
  - per-layer embeddings live under ``model.per_layer.*`` in NNX but flat at
    ``model.*`` in HF.

This stream maps those explicitly and defers everything else (attention/MLP
kernels, sandwich norms, embedding, lm_head) to the dense J5 map. Experts stay
**fused**: HF's ``gemma4`` checkpoint layout is the fused ``(E, ...)`` stack, so
there is no per-expert expansion (unlike ``qwen3_moe``). Because ``gemma4`` MoE
with ``expert_parallel_size > 1`` is gated off (policy_worker; HV-29/30), no EP
``materialize`` gather is needed — the fused params are whole on a single device.

The pure name/shape/value logic is CPU-validatable: a ``load -> refit`` round-trip
reproduces the HF state dict. The live vLLM broadcast is hardware-deferred (HV-27).
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from flax import nnx

from dockyard_rl.models.jax.refit import _TORCH_DTYPE_NAME, _hf_oriented, _jax_to_torch
from dockyard_rl.models.jax.weights import nnx_path_to_hf_name

# Router/expert Param leaves that pass through under their own name: no ".weight"
# rename and no transpose. The fused expert stacks (gate_up_proj (E,2F,D),
# down_proj (E,D,F)) and the scalar router params (scale (D,), per_expert_scale
# (E,)) keep the HF checkpoint layout exactly.
_PASSTHROUGH_LEAVES = ("gate_up_proj", "down_proj", "scale", "per_expert_scale")


def _gemma4_base_refit_stream(model: nnx.Module) -> Iterator[tuple[str, Any]]:
    """HF-named JAX-array stream for a Gemma 4 model (experts kept fused).

    Yields ``(hf_name, array)`` in HF layout (kernels transposed (in,out)->(out,in)).
    """
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    for path, var in flat:
        p = tuple(path)
        leaf = p[-1]
        arr = var[...]

        # MoE router scalars + fused expert stacks: own name, no transpose. The
        # expert leaves are the Param attribute names themselves (the MLP's
        # down_proj is a Linear whose leaf is "kernel", so it does not match here).
        if leaf in _PASSTHROUGH_LEAVES:
            yield ".".join(str(c) for c in p), arr
            continue

        # PLE: NNX nests the table under model.per_layer.*; HF keeps it flat at
        # model.* . Drop the "per_layer" segment, then apply the dense map.
        if len(p) >= 2 and p[1] == "per_layer":
            flat_path = (p[0],) + p[2:]
            yield nnx_path_to_hf_name(flat_path), _hf_oriented(flat_path, arr)
            continue

        # everything else: attention/MLP kernels, sandwich norms (incl. per-layer
        # decoder norms), embedding, lm_head -> dense kernel-transpose / passthrough.
        yield nnx_path_to_hf_name(p), _hf_oriented(p, arr)


def iter_gemma4_refit_state_dict(
    model: nnx.Module, *, to_torch: bool = True
) -> Iterator[tuple[str, Any]]:
    """Yield ``(hf_name, tensor)`` for a Gemma 4 model (experts fused).

    ``to_torch`` converts the final arrays to torch CPU tensors for the unchanged
    packed-broadcast producer; pass ``False`` to keep JAX arrays (tests).
    """
    for name, arr in _gemma4_base_refit_stream(model):
        yield name, (_jax_to_torch(arr) if to_torch else arr)


def prepare_gemma4_refit_info(
    model: nnx.Module, *, dtype: Optional[str] = None
) -> dict[str, Any]:
    """``{hf_name: (torch.Size, dtype_name)}`` matching the Gemma 4 refit stream."""
    import torch

    out: dict[str, Any] = {}
    for name, arr in _gemma4_base_refit_stream(model):
        dt = dtype or _TORCH_DTYPE_NAME.get(str(arr.dtype), str(arr.dtype))
        out[name] = (torch.Size(tuple(arr.shape)), dt)
    return out
