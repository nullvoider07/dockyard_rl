"""HF <-> NNX parameter name mapping and weight loading.

The NNX parameter tree (``models/jax/models/qwen3.py``) is named to mirror the
HF module hierarchy, so the map is a mechanical path transform rather than a
hand-maintained table:

    nnx flat path                                   HF state-dict key
    ('model','embed_tokens','embedding')            model.embed_tokens.weight
    ('model','layers',i,'self_attn','q_proj','kernel')  model.layers.i.self_attn.q_proj.weight
    ('model','layers',i,'self_attn','q_norm','scale')   model.layers.i.self_attn.q_norm.weight
    ('model','norm','scale')                        model.norm.weight
    ('lm_head','kernel')                            lm_head.weight

The only value transform is the linear-kernel transpose: HF ``nn.Linear.weight``
is ``(out, in)``; the NNX ``Linear`` kernel is ``(in, out)``. Embedding and
RMSNorm weights are layout-identical. The transpose set (leaf == ``"kernel"``)
is exactly the J5 refit inverse.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
from flax import nnx

from dockyard_rl.models.jax.models.qwen3 import Qwen3ForCausalLM

# HF param keys whose tensors must be transposed when loading into NNX (and
# transposed back on refit). Determined structurally by the NNX leaf name.
_KERNEL_LEAF = "kernel"


def nnx_path_to_hf_name(path: tuple[Any, ...]) -> str:
    """Map an NNX flat-state path tuple to the HF state-dict key.

    The leaf (``kernel`` / ``scale`` / ``embedding``) always corresponds to HF
    ``weight``; intermediate components (including integer layer indices) map
    one-to-one.
    """
    *prefix, _leaf = path
    parts = [str(p) for p in prefix]
    parts.append("weight")
    return ".".join(parts)


def path_needs_transpose(path: tuple[Any, ...]) -> bool:
    """True iff this param is a linear kernel (HF ``(out, in)`` -> NNX ``(in, out)``)."""
    return path[-1] == _KERNEL_LEAF


def hf_name_to_nnx_path(name: str, num_layers: int) -> tuple[Any, ...]:
    """Inverse map: HF state-dict key -> NNX flat-state path tuple.

    Used by the J5 refit seam to walk params in HF order. ``num_layers`` is
    accepted for symmetry/validation; the transform itself is positional.
    """
    comps = name.split(".")
    if comps[-1] != "weight":
        raise KeyError(f"Unexpected HF param key (no .weight leaf): {name!r}")
    body = comps[:-1]
    # Choose the NNX leaf name from the module type implied by the body.
    last_module = body[-1]
    if last_module == "embed_tokens":
        leaf = "embedding"
    elif last_module in ("input_layernorm", "post_attention_layernorm", "norm", "q_norm", "k_norm"):
        leaf = "scale"
    else:
        leaf = _KERNEL_LEAF
    path: list[Any] = []
    for c in body:
        if c.isdigit():
            idx = int(c)
            if idx >= num_layers:
                raise IndexError(f"Layer index {idx} in {name!r} exceeds num_layers={num_layers}")
            path.append(idx)
        else:
            path.append(c)
    path.append(leaf)
    return tuple(path)


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert a source weight (torch tensor or array-like) to a numpy array.

    Duck-typed so ``weights.py`` does not import torch at module load; on CPU
    the test path passes torch tensors, which expose ``.detach().cpu().numpy()``.
    """
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        return np.asarray(arr.numpy())
    return np.asarray(arr)


def load_hf_state_dict(
    model: Qwen3ForCausalLM,
    hf_state: Mapping[str, Any],
    *,
    param_dtype: jnp.dtype = jnp.float32,
    strict: bool = True,
) -> Qwen3ForCausalLM:
    """Load an HF state dict into an NNX ``Qwen3ForCausalLM`` in place.

    Handles tied embeddings: if ``lm_head.weight`` is absent and the config ties
    word embeddings, the lm_head kernel is filled from
    ``model.embed_tokens.weight`` (still transposed, since lm_head is a kernel).

    Returns the same ``model`` (updated) for convenience.
    """
    state = nnx.state(model)
    flat = nnx.to_flat_state(state)
    tie = model.cfg.tie_word_embeddings

    consumed: set[str] = set()
    for path, var in flat:
        hf_name = nnx_path_to_hf_name(path)
        source_name = hf_name
        if hf_name not in hf_state and hf_name == "lm_head.weight" and tie:
            source_name = "model.embed_tokens.weight"
        if source_name not in hf_state:
            if strict:
                raise KeyError(f"Missing HF weight for NNX param {path}: expected key {source_name!r}")
            continue
        np_arr = _to_numpy(hf_state[source_name])
        if path_needs_transpose(path):
            if np_arr.ndim != 2:
                raise ValueError(f"Kernel param {path} expected 2D HF weight, got shape {np_arr.shape}")
            np_arr = np_arr.T
        target_shape = tuple(var[...].shape)
        if tuple(np_arr.shape) != target_shape:
            raise ValueError(
                f"Shape mismatch for {path} (HF {source_name!r}): "
                f"loaded {tuple(np_arr.shape)} vs param {target_shape}"
            )
        var[...] = jnp.asarray(np_arr, dtype=param_dtype)
        consumed.add(source_name)

    if strict:
        unexpected = set(hf_state.keys()) - consumed
        # For tied embeddings the NNX model has no separate lm_head param (logits
        # reuse the embedding), so an HF lm_head.weight is redundant — not a gap.
        if tie:
            unexpected.discard("lm_head.weight")
        if unexpected:
            raise KeyError(f"Unconsumed HF weights (name-map gap): {sorted(unexpected)[:8]}")

    nnx.update(model, nnx.from_flat_state(flat))
    return model
