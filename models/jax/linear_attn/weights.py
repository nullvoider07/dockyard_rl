"""HF Qwen3-Next state-dict -> NNX loader (hybrid linear/full attention + MoE).

Extends the J8 fused-MoE mapping with the Qwen3-Next-specific params:
  - linear-attention: `linear_attn.{in_proj_qkvz,in_proj_ba,out_proj}.kernel`
    <- HF `.weight` (transposed); `linear_attn.conv_weight` <- HF
    `linear_attn.conv1d.weight` (E,1,K) squeezed to (C,K); `dt_bias`/`A_log`
    passthrough; `linear_attn.norm.weight` (gated RMSNorm) passthrough;
  - shared expert: JAX `mlp.shared_experts.mlp.*` <- HF `mlp.shared_expert.*`
    (the dockyard wrapper nests an inner `mlp`); JAX
    `mlp.shared_experts.shared_expert_gate.kernel` <- HF `mlp.shared_expert_gate.weight`;
  - `(1+weight)` RMSNorm leaves are NNX `scale` <- HF `.weight` (no transpose),
    distinct from the gated-norm `weight` leaf.

Fused experts + router gate reuse the J8 rule. Handles tied embeddings.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.models.qwen3_next import Qwen3NextForCausalLM
from dockyard_rl.models.jax.weights import _to_numpy

_EXPERT_LEAVES = ("w1_EFD", "w3_EFD", "w2_EDF")


def load_hf_qwen3_next_state_dict(
    model: Qwen3NextForCausalLM,
    hf_state: Mapping[str, Any],
    *,
    param_dtype: jnp.dtype = jnp.float32,
    strict: bool = True,
) -> Qwen3NextForCausalLM:
    """Load an HF Qwen3-Next state dict into the NNX model in place."""
    cfg = model.cfg
    f = cfg.moe_intermediate_size
    tie = cfg.tie_word_embeddings
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    consumed: set[str] = set()

    def assign(var: Any, arr: Any, source: str) -> None:
        target_shape = tuple(var[...].shape)
        if tuple(arr.shape) != target_shape:
            raise ValueError(
                f"Shape mismatch for {source!r}: loaded {tuple(arr.shape)} vs param {target_shape}"
            )
        var[...] = jnp.asarray(arr, dtype=param_dtype)
        consumed.add(source)

    for path, var in flat:
        leaf = path[-1]
        prefix = path[:-1]
        joined = ".".join(str(p) for p in prefix)

        # routed experts: fused gate_up_proj / down_proj (no transpose)
        if leaf in _EXPERT_LEAVES:
            if leaf == "w2_EDF":
                src = f"{joined}.down_proj"
                assign(var, _to_numpy(hf_state[src]), src)
            else:
                src = f"{joined}.gate_up_proj"
                gu = _to_numpy(hf_state[src])  # (E, 2F, D)
                assign(var, gu[:, :f, :] if leaf == "w1_EFD" else gu[:, f:, :], src)
            continue

        # router gate: ...mlp.router.gate.kernel <- ...mlp.gate.weight (transpose)
        if leaf == "kernel" and len(prefix) >= 2 and prefix[-2] == "router" and prefix[-1] == "gate":
            base = ".".join(str(p) for p in prefix[:-2])
            src = f"{base}.gate.weight"
            assign(var, _to_numpy(hf_state[src]).T, src)
            continue

        # shared-expert gate: ...mlp.shared_experts.shared_expert_gate.kernel
        #   <- ...mlp.shared_expert_gate.weight (transpose)
        if leaf == "kernel" and len(prefix) >= 2 and prefix[-1] == "shared_expert_gate":
            base = ".".join(str(p) for p in prefix[:-2])  # strip shared_experts.shared_expert_gate
            src = f"{base}.shared_expert_gate.weight"
            assign(var, _to_numpy(hf_state[src]).T, src)
            continue

        # shared-expert MLP: ...mlp.shared_experts.mlp.{proj}.kernel
        #   <- ...mlp.shared_expert.{proj}.weight (transpose)
        if leaf == "kernel" and "shared_experts" in prefix:
            j = prefix.index("shared_experts")
            hf_prefix = list(prefix[:j]) + ["shared_expert"] + list(prefix[j + 2 :])
            src = ".".join(str(p) for p in hf_prefix) + ".weight"
            assign(var, _to_numpy(hf_state[src]).T, src)
            continue

        # linear-attention depthwise conv: conv_weight (C,K) <- conv1d.weight (C,1,K)
        if leaf == "conv_weight":
            src = f"{joined}.conv1d.weight"
            assign(var, _to_numpy(hf_state[src]).squeeze(1), src)
            continue

        # linear-attention scalars: dt_bias / A_log (passthrough)
        if leaf in ("dt_bias", "A_log"):
            src = f"{joined}.{leaf}"
            assign(var, _to_numpy(hf_state[src]), src)
            continue

        # norms: RMSNorm1p `scale` and gated-norm `weight` both <- HF `.weight` (no transpose)
        if leaf in ("scale", "weight"):
            src = f"{joined}.weight"
            assign(var, _to_numpy(hf_state[src]), src)
            continue

        # embedding (no transpose)
        if leaf == "embedding":
            src = f"{joined}.weight"
            assign(var, _to_numpy(hf_state[src]), src)
            continue

        # generic kernel: attention projections, in/out projs, dense MLP, lm_head (transpose)
        hf_name = f"{joined}.weight"
        source = hf_name
        if hf_name == "lm_head.weight" and tie and hf_name not in hf_state:
            source = "model.embed_tokens.weight"
        if source not in hf_state:
            if strict:
                raise KeyError(f"Missing HF weight for NNX param {tuple(path)}: expected {source!r}")
            continue
        arr = _to_numpy(hf_state[source])
        if arr.ndim != 2:
            raise ValueError(f"Kernel param {tuple(path)} expected 2D HF weight, got {arr.shape}")
        assign(var, arr.T, source)

    if strict:
        unexpected = set(hf_state.keys()) - consumed
        if tie:
            unexpected.discard("lm_head.weight")
        if unexpected:
            raise KeyError(f"Unconsumed HF weights (name-map gap): {sorted(unexpected)[:8]}")

    nnx.update(model, nnx.from_flat_state(flat))
    return model
