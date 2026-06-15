"""HF Qwen3-MoE state-dict -> NNX loader (fused-expert aware).

Mirrors the dense ``models/jax/weights.py`` name-map, with three MoE-specific
cases for the routed layers (transformers 5.x fused layout):

  - router gate ``...mlp.router.gate.kernel`` <- HF ``...mlp.gate.weight`` (E,D),
    transposed to the NNX ``(D,E)`` kernel (the extra ``router`` nesting is
    dockyard's; HF nests the gate directly under ``mlp``);
  - ``...mlp.experts.w1_EFD`` / ``w3_EFD`` <- HF fused ``...mlp.experts.gate_up_proj``
    ``(E,2F,D)`` sliced ``[:, :F, :]`` (gate) / ``[:, F:, :]`` (up) — NO transpose;
  - ``...mlp.experts.w2_EDF`` <- HF ``...mlp.experts.down_proj`` ``(E,D,F)`` — NO transpose.

All other params (attention, norms, embedding, dense-MLP layers, lm_head) follow
the dense kernel-transpose / scale-passthrough rule. Handles tied embeddings.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.moe.qwen3_moe import Qwen3MoeForCausalLM
from dockyard_rl.models.jax.weights import _to_numpy

_EXPERT_LEAVES = ("w1_EFD", "w3_EFD", "w2_EDF")


def load_hf_qwen3_moe_state_dict(
    model: Qwen3MoeForCausalLM,
    hf_state: Mapping[str, Any],
    *,
    param_dtype: jnp.dtype = jnp.float32,
    strict: bool = True,
) -> Qwen3MoeForCausalLM:
    """Load an HF Qwen3-MoE state dict into the NNX model in place."""
    cfg = model.cfg
    f = cfg.moe_intermediate_size
    tie = cfg.tie_word_embeddings
    # Only trainable params have HF sources; the MoE Buffer state (expert_bias_E,
    # tokens_per_expert_E) is left at its zero init.
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

        # routed experts: fused gate_up_proj / down_proj -> w1/w3/w2_EFD (no transpose)
        if leaf in _EXPERT_LEAVES:
            base = ".".join(str(p) for p in prefix)  # ...mlp.experts
            if leaf == "w2_EDF":
                src = f"{base}.down_proj"
                assign(var, _to_numpy(hf_state[src]), src)
            else:
                src = f"{base}.gate_up_proj"
                gu = _to_numpy(hf_state[src])  # (E, 2F, D)
                arr = gu[:, :f, :] if leaf == "w1_EFD" else gu[:, f:, :]
                assign(var, arr, src)
            continue

        # router gate: ...mlp.router.gate.kernel <- ...mlp.gate.weight (transpose)
        if leaf == "kernel" and len(prefix) >= 2 and prefix[-2] == "router" and prefix[-1] == "gate":
            base = ".".join(str(p) for p in prefix[:-2])  # ...mlp
            src = f"{base}.gate.weight"
            assign(var, _to_numpy(hf_state[src]).T, src)
            continue

        # dense path (attention / norms / embedding / dense-MLP / lm_head)
        hf_name = ".".join(str(p) for p in prefix) + ".weight"
        source = hf_name
        if leaf == "kernel" and hf_name == "lm_head.weight" and tie and hf_name not in hf_state:
            source = "model.embed_tokens.weight"
        if source not in hf_state:
            if strict:
                raise KeyError(f"Missing HF weight for NNX param {tuple(path)}: expected {source!r}")
            continue
        arr = _to_numpy(hf_state[source])
        if leaf == "kernel":
            if arr.ndim != 2:
                raise ValueError(f"Kernel param {tuple(path)} expected 2D HF weight, got {arr.shape}")
            arr = arr.T
        assign(var, arr, source)

    if strict:
        unexpected = set(hf_state.keys()) - consumed
        if tie:
            unexpected.discard("lm_head.weight")
        if unexpected:
            raise KeyError(f"Unconsumed HF weights (name-map gap): {sorted(unexpected)[:8]}")

    nnx.update(model, nnx.from_flat_state(flat))
    return model
