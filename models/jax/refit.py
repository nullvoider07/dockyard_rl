"""Refit seam (Seam 1): JAX params -> HF-named torch tensors for weight sync.

Mirrors the torch worker's ``_iter_refit_state_dict`` /``prepare_refit_info``
contract (`models/policy/workers/dtensor_policy_worker_v2.py`,
`base_policy_worker.py`): yield ``(hf_name, torch.Tensor)`` pairs that the
unchanged ``utils/packed_tensor.py::packed_broadcast_producer`` packs and
broadcasts over the torch NCCL ``model_update_group`` into vLLM. JAX never owns
the NCCL group — it only produces tensors.

The HF name + transpose set is exactly the inverse of the J1 loader
(`models/jax/weights.py`): NNX ``kernel`` (in, out) -> HF ``weight`` (out, in)
transposed; ``scale``/``embedding`` pass through. dense-only at J5; MoE
re-expansion of fused experts is J8.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import jax.numpy as jnp
import numpy as np
from flax import nnx

from dockyard_rl.models.jax.weights import nnx_path_to_hf_name, path_needs_transpose

# JAX dtype -> torch dtype, for prepare_refit_info's (shape, dtype) map.
_TORCH_DTYPE_NAME = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float64": "float64",
}


def _jax_to_torch(arr: Any) -> Any:
    """Convert a (already HF-oriented) JAX array to a torch CPU tensor.

    Tries dlpack (zero-copy; the intended CUDA path), falling back to numpy on
    CPU / non-contiguous arrays. GPU zero-copy is validated at J10.
    """
    import torch

    try:
        return torch.from_dlpack(arr)
    except Exception:
        # np.array (copy) ensures a writable, C-contiguous buffer for torch.
        return torch.from_numpy(np.array(arr))


def _hf_oriented(path: tuple[Any, ...], arr: jnp.ndarray) -> jnp.ndarray:
    """Return the array in HF layout: transpose kernels (in,out)->(out,in)."""
    if path_needs_transpose(path):
        if arr.ndim != 2:
            raise ValueError(f"Kernel param {path} expected 2D, got shape {arr.shape}")
        return arr.T
    return arr


def iter_refit_state_dict(model: nnx.Module) -> Iterable[tuple[str, Any]]:
    """Yield ``(hf_name, torch.Tensor)`` for every parameter, in HF layout."""
    flat = nnx.to_flat_state(nnx.state(model))
    for path, var in flat:
        hf_name = nnx_path_to_hf_name(tuple(path))
        arr = _hf_oriented(tuple(path), var[...])
        yield hf_name, _jax_to_torch(arr)


def prepare_refit_info(
    model: nnx.Module, dtype: Optional[str] = None
) -> dict[str, tuple[Any, str]]:
    """Return ``{hf_name: (torch.Size, dtype_name)}`` mirroring the streamed tensors.

    Shapes are reported in HF layout (kernels transposed). ``dtype`` overrides
    the per-param dtype when the refit casts (e.g. bf16); otherwise the param's
    own dtype name is used.
    """
    import torch

    out: dict[str, tuple[Any, str]] = {}
    flat = nnx.to_flat_state(nnx.state(model))
    for path, var in flat:
        hf_name = nnx_path_to_hf_name(tuple(path))
        arr = var[...]
        shape = tuple(arr.shape[::-1]) if path_needs_transpose(tuple(path)) else tuple(arr.shape)
        dtype_name = dtype or _TORCH_DTYPE_NAME.get(str(arr.dtype), str(arr.dtype))
        out[hf_name] = (torch.Size(shape), dtype_name)
    return out
