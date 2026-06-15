"""Reusable Flax NNX primitives built on explicit ``nnx.Param`` arrays.

Each primitive stores its weight as a raw ``nnx.Param`` in JAX-native layout
(linear kernels as ``(in, out)``), so the HF-weights loader name-map, the J2
sharding ``PartitionSpec`` attachment, and the J5 refit inverse are all explicit
and live in one place rather than threaded through framework layer internals.
The per-projection boilerplate is written once here and composed everywhere.

A ``sharding`` metadata tuple may be attached to each ``nnx.Param`` (logical
mesh-axis names per array dim, or ``None`` for replicated). It is unused at J1
(single-device parity) and consumed by J2.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

Array = jax.Array
Sharding = Optional[Sequence[Optional[str]]]

# Logical mesh-axis names are stored under this non-reserved Variable metadata
# key. nnx treats the reserved ``sharding`` key as an *eager* annotation (FLIP
# 4844) that shards the array at construction and requires a live mesh, which is
# unavailable at J1 (single-device). J2 reads ``param.logical_axes`` and
# translates it to a ``PartitionSpec`` under the trainer mesh.
_AXES_META_KEY = "logical_axes"


class Linear(nnx.Module):
    """``y = x @ kernel`` with ``kernel`` stored ``(in_features, out_features)``.

    Bias-free: every Qwen3 dense projection (q/k/v/o, gate/up/down, lm_head) has
    ``bias=False``. HF ``nn.Linear`` stores ``weight`` as ``(out, in)``; the
    loader transposes into this ``(in, out)`` convention.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        kernel_init: Initializer = nnx.initializers.lecun_normal(),
        sharding: Sharding = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        kernel = kernel_init(rngs.params(), (in_features, out_features), param_dtype)
        self.kernel = nnx.Param(kernel, logical_axes=sharding)

    def __call__(self, x: Array) -> Array:
        return jnp.einsum("...i,io->...o", x, self.kernel[...])


class Embedding(nnx.Module):
    """Token embedding table ``(num_embeddings, features)``; gather by id.

    Matches HF ``nn.Embedding.weight`` layout exactly (no transpose on load).
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        embedding_init: Initializer = nnx.initializers.normal(stddev=1.0),
        sharding: Sharding = None,
    ) -> None:
        self.num_embeddings = num_embeddings
        self.features = features
        table = embedding_init(rngs.params(), (num_embeddings, features), param_dtype)
        self.embedding = nnx.Param(table, logical_axes=sharding)

    def __call__(self, ids: Array) -> Array:
        return jnp.take(self.embedding[...], ids, axis=0)


class RMSNorm(nnx.Module):
    """RMSNorm matching ``Qwen3RMSNorm`` numerics exactly.

    Variance is computed in float32, the normalized activation is cast back to
    the input dtype, then multiplied by the (param-dtype) scale — identical to
    HF's ``self.weight * hidden_states.to(input_dtype)`` ordering, which matters
    for bf16 parity. The scale initializes to ones.
    """

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        param_dtype: jnp.dtype = jnp.float32,
        sharding: Sharding = None,
    ) -> None:
        self.dim = dim
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((dim,), dtype=param_dtype), logical_axes=sharding)

    def __call__(self, x: Array) -> Array:
        in_dtype = x.dtype
        xf = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
        xf = xf * jax.lax.rsqrt(variance + self.eps)
        return self.scale[...] * xf.astype(in_dtype)
