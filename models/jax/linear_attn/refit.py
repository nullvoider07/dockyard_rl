"""Qwen3-Next refit seam: NNX params -> HF-named torch tensors (J11 follow-on).

The inverse of ``load_hf_qwen3_next_state_dict``. Reuses the J8 per-expert
expansion (fused GroupedExperts -> ``...experts.{i}.{gate,up,down}_proj.weight``)
and the dense name-map for standard params, and adds the Qwen3-Next specials:

  - linear attention: ``conv_weight`` (C,K) -> HF ``linear_attn.conv1d.weight``
    (C,1,K) (unsqueeze dim 1); ``dt_bias`` / ``A_log`` -> passthrough (no
    ``.weight`` suffix, no transpose); ``in_proj_*`` / ``out_proj`` kernels
    transposed; gated ``norm.weight`` passthrough;
  - shared expert: ``mlp.shared_experts.mlp.{proj}.kernel`` -> HF
    ``mlp.shared_expert.{proj}.weight``; ``mlp.shared_experts.shared_expert_gate``
    -> HF ``mlp.shared_expert_gate.weight``;
  - router gate transposed; ``(1+weight)`` norms / embedding via the dense map.

The pure name/shape/value logic is CPU-validatable (a load->refit round-trip
reproduces the HF state dict); the ep>1 gather (``materialize``) and the live
vLLM broadcast are hardware-deferred (HV-27/30).
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.moe.refit import _EXPERT_LEAVES, iter_expanded_refit_tensors
from dockyard_rl.models.jax.refit import _TORCH_DTYPE_NAME, _hf_oriented, _jax_to_torch
from dockyard_rl.models.jax.weights import nnx_path_to_hf_name

_LINEAR_SCALARS = ("dt_bias", "A_log")


def _qwen3_next_base_refit_stream(model: nnx.Module) -> Iterator[tuple[str, Any]]:
    """HF-named JAX-array stream (experts kept FUSED for later per-expert expansion)."""
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    for path, var in flat:
        p = tuple(path)
        leaf = p[-1]
        arr = var[...]

        # fused experts -> keep the fused name so iter_expanded_refit_tensors unbinds it
        if leaf in _EXPERT_LEAVES:
            yield ".".join(str(c) for c in p), arr
            continue

        # router gate: ...mlp.router.gate.kernel (D,E) -> ...mlp.gate.weight (E,D)
        if leaf == "kernel" and p[-3:-1] == ("router", "gate"):
            name = ".".join(str(c) for c in p[:-3]) + ".gate.weight"
            yield name, arr.T
            continue

        # shared-expert gate: ...shared_experts.shared_expert_gate.kernel
        #   -> ...mlp.shared_expert_gate.weight
        if leaf == "kernel" and len(p) >= 2 and p[-2] == "shared_expert_gate":
            j = p.index("shared_experts")
            name = ".".join(str(c) for c in p[:j]) + ".shared_expert_gate.weight"
            yield name, arr.T
            continue

        # shared-expert MLP: ...shared_experts.mlp.{proj}.kernel -> ...mlp.shared_expert.{proj}.weight
        if leaf == "kernel" and "shared_experts" in p:
            j = p.index("shared_experts")
            hf = list(p[:j]) + ["shared_expert"] + list(p[j + 2 : -1])
            yield ".".join(str(c) for c in hf) + ".weight", arr.T
            continue

        # linear-attn depthwise conv: conv_weight (C,K) -> conv1d.weight (C,1,K)
        if leaf == "conv_weight":
            name = ".".join(str(c) for c in p[:-1]) + ".conv1d.weight"
            yield name, arr[:, None, :]
            continue

        # linear-attn scalars: dt_bias / A_log (passthrough, no .weight, no transpose)
        if leaf in _LINEAR_SCALARS:
            yield ".".join(str(c) for c in p), arr
            continue

        # everything else: attention / in-out projs / norms (scale|weight) / embedding / lm_head
        yield nnx_path_to_hf_name(p), _hf_oriented(p, arr)


def iter_qwen3_next_refit_state_dict(
    model: nnx.Module, *, materialize: Optional[Any] = None, to_torch: bool = True
) -> Iterator[tuple[str, Any]]:
    """Yield ``(hf_name, tensor)`` for a Qwen3-Next model, experts expanded per-expert.

    ``materialize`` performs the ep>1 gather on fused-expert arrays (device-bound;
    HV); defaults to identity. ``to_torch`` converts to torch CPU tensors for the
    unchanged packed-broadcast producer.
    """
    mat = materialize if materialize is not None else (lambda t: t)
    for name, arr in iter_expanded_refit_tensors(_qwen3_next_base_refit_stream(model), mat):
        yield name, (_jax_to_torch(arr) if to_torch else arr)


def prepare_qwen3_next_refit_info(
    model: nnx.Module, *, materialize: Optional[Any] = None, dtype: Optional[str] = None
) -> dict[str, Any]:
    """``{hf_name: (torch.Size, dtype_name)}`` matching the expanded refit stream."""
    import torch

    out: dict[str, Any] = {}
    mat = materialize if materialize is not None else (lambda t: t)
    for name, arr in iter_expanded_refit_tensors(_qwen3_next_base_refit_stream(model), mat):
        dt = dtype or _TORCH_DTYPE_NAME.get(str(jnp.asarray(arr).dtype), str(jnp.asarray(arr).dtype))
        out[name] = (torch.Size(tuple(arr.shape)), dt)
    return out
