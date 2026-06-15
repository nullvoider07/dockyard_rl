"""torch <-> jnp boundary conversion for the JAX worker.

The trainer fleet's data plane carries torch tensors (`BatchedDataDict`); the
JAX worker converts only the columns it needs at its own boundary. On CPU
(tests) this goes through numpy; on CUDA the GPU path uses dlpack (zero-copy) —
added when the GPU bring-up lands (J10). J6 builds the column-wise
`BatchedDataDict` conversion on top of these primitives.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def torch_to_jnp(t: Any) -> jnp.ndarray:
    """Convert a torch tensor (or array-like) to a jnp array via numpy (CPU path)."""
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "numpy"):
        return jnp.asarray(t.numpy())
    return jnp.asarray(np.asarray(t))


def jnp_to_torch(a: Any) -> Any:
    """Convert a jnp array to a torch CPU tensor via numpy.

    torch is imported lazily so this module stays import-light for pure-JAX
    code paths that never touch the torch boundary.
    """
    import torch

    # np.array (copy) yields a writable buffer; np.asarray of a jax array can be
    # read-only, which torch.from_numpy warns about.
    return torch.from_numpy(np.array(a))
