"""MoE compute core in JAX (Flax NNX) — the J8 mirror of ``models/dtensor/moe``.

The pure compute core (router, grouped experts, local token dispatch, block,
aux-loss-free load balance) is CPU-parity-validatable: ``jax.lax.ragged_dot``
lowers on CPU (unlike the torch side's CUDA-only ``torch._grouped_mm``), so the
expert grouped-GEMM runs and is numerically checked without a GPU. Expert/router
sharding over an ``ep`` mesh axis and the EP all-to-all dispatch are later
sub-phases (the live multi-device numerics are hardware-deferred).
"""

from __future__ import annotations

from flax import nnx


class Buffer(nnx.Variable):
    """Non-trained MoE state (load-balance bias, per-expert token counter).

    A distinct ``nnx.Variable`` subclass so ``nnx.state(model, nnx.Param)`` (and
    the optimizer's ``wrt=nnx.Param``) excludes it from gradients, while Orbax
    still captures it via the full ``nnx.state``.
    """
