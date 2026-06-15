"""Mesh construction and parameter sharding for the JAX trainer.

Each ``nnx.Param`` carries a ``logical_axes`` metadata tuple (set by the model,
e.g. ``models/jax/models/qwen3.py``) naming the mesh axis along each array dim,
or ``None`` for a replicated dim. This module builds a ``jax.sharding.Mesh``
over the device grid and maps that metadata to concrete ``NamedSharding`` /
``PartitionSpec``. Dense tensor-parallelism is then handled by GSPMD: with the
params and inputs annotated, ``jax.jit``/``nnx.jit`` inserts the all-reduce /
all-gather collectives automatically — no explicit ``shard_map`` for dense
matmuls (that is reserved for J8 MoE expert-parallel all-to-all and J11
sequence-parallel attention).

Logical axis names mirror the torch mesh: ``dp`` (data), ``cp`` (context),
``tp`` (tensor). DP shards the activation batch dim; TP shards model weights
(column/row parallel). CP (sequence-sharded attention) needs ring-attention and
is wired into the mesh but its activation sharding is deferred (see J11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import jax
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

AXIS_DP = "dp"
AXIS_CP = "cp"
AXIS_TP = "tp"
AXIS_EP = "ep"
MESH_AXES = (AXIS_DP, AXIS_CP, AXIS_TP)
MESH_AXES_MOE = (AXIS_DP, AXIS_CP, AXIS_TP, AXIS_EP)

# Expert-weight logical axes for GroupedExperts (J8b). Expert dim 0 shards over
# ``ep``; the hidden dim shards over ``tp`` (gate/up colwise on F, down rowwise
# on F) so EP and dense-TP compose. GSPMD treats size-1 axes as replicated, so
# the same static annotation serves ep=1 (TP-only) and tp=1 (EP-only):
#   w1_EFD / w3_EFD (E, F, D): (ep, tp, None)
#   w2_EDF          (E, D, F): (ep, None, tp)
EXPERT_GATE_UP_AXES = (AXIS_EP, AXIS_TP, None)
EXPERT_DOWN_AXES = (AXIS_EP, None, AXIS_TP)


def build_mesh(
    dp: int,
    tp: int,
    cp: int = 1,
    devices: Optional[Sequence[Any]] = None,
) -> Mesh:
    """Build a ``(dp, cp, tp)`` device mesh.

    The product ``dp * cp * tp`` must not exceed the available device count.
    Device order is row-major over ``(dp, cp, tp)``, matching the torch
    ``NamedSharding`` layout used by the policy wrapper.
    """
    devs = list(devices) if devices is not None else jax.devices()
    need = dp * cp * tp
    if need > len(devs):
        raise ValueError(f"mesh dp*cp*tp={need} exceeds available devices={len(devs)}")
    grid = np.array(devs[:need], dtype=object).reshape(dp, cp, tp)
    return Mesh(grid, axis_names=MESH_AXES)


def build_moe_mesh(
    dp: int,
    tp: int,
    cp: int = 1,
    ep: int = 1,
    devices: Optional[Sequence[Any]] = None,
) -> Mesh:
    """Build a ``(dp, cp, tp, ep)`` device mesh for MoE expert parallelism.

    Unlike the torch build (which carves ``ep`` from ``dp_shard*cp*tp`` as a
    second unflatten view of one world mesh), JAX/GSPMD treats ``ep`` as a
    first-class orthogonal axis: ``dp * cp * tp * ep`` devices, expert params
    sharded over ``ep`` and the dispatch all-to-all run over the ``ep`` axis
    (J8c). Device order is row-major over ``(dp, cp, tp, ep)``.
    """
    devs = list(devices) if devices is not None else jax.devices()
    need = dp * cp * tp * ep
    if need > len(devs):
        raise ValueError(f"moe mesh dp*cp*tp*ep={need} exceeds available devices={len(devs)}")
    grid = np.array(devs[:need], dtype=object).reshape(dp, cp, tp, ep)
    return Mesh(grid, axis_names=MESH_AXES_MOE)


@dataclass(frozen=True)
class MoEMeshDims:
    """Pure MoE parallelism factoring for JAX (no device dependency).

    JAX-idiomatic: ``ep`` is an orthogonal mesh axis, so the device-count
    constraint is the product ``dp * cp * tp * ep`` (contrast the torch
    ``MoEParallelDims``, where ep is carved from ``dp_shard*cp*tp``). Validates
    the factoring + the ``num_experts % ep`` divisibility used to size the local
    expert shard.
    """

    dp: int = 1
    cp: int = 1
    tp: int = 1
    ep: int = 1

    def __post_init__(self) -> None:
        for name, d in (("dp", self.dp), ("cp", self.cp), ("tp", self.tp), ("ep", self.ep)):
            if d < 1:
                raise ValueError(f"{name} must be >= 1, got {d}")

    @property
    def world_size(self) -> int:
        return self.dp * self.cp * self.tp * self.ep

    @property
    def enable_ep(self) -> bool:
        return self.ep > 1

    def validate_devices(self, num_devices: int) -> None:
        if self.world_size > num_devices:
            raise ValueError(
                f"moe mesh world_size={self.world_size} (dp*cp*tp*ep) exceeds "
                f"available devices={num_devices}"
            )

    def num_local_experts(self, num_experts: int) -> int:
        """Experts owned per ``ep`` shard (``num_experts`` must divide by ep)."""
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if num_experts % self.ep != 0:
            raise ValueError(f"num_experts({num_experts}) must be divisible by ep({self.ep})")
        return num_experts // self.ep


def _spec_from_axes(logical_axes: Optional[Sequence[Optional[str]]]) -> PartitionSpec:
    """Map a param's ``logical_axes`` metadata to a ``PartitionSpec``.

    ``None`` (no metadata) or all-``None`` axes means fully replicated.
    """
    if not logical_axes:
        return PartitionSpec()
    return PartitionSpec(*logical_axes)


def shard_model(model: nnx.Module, mesh: Mesh) -> nnx.Module:
    """Place each parameter of ``model`` onto ``mesh`` per its ``logical_axes``.

    Mutates the model's params in place (via ``device_put``) and returns it.
    Params without ``logical_axes`` metadata are replicated across the mesh.
    """
    state = nnx.state(model)
    flat = nnx.to_flat_state(state)
    updated = []
    for path, var in flat:
        axes = getattr(var, "logical_axes", None)
        sharding = NamedSharding(mesh, _spec_from_axes(axes))
        var[...] = jax.device_put(var[...], sharding)
        updated.append((path, var))
    nnx.update(model, nnx.from_flat_state(updated))
    return model


def data_sharding(mesh: Mesh) -> NamedSharding:
    """NamedSharding for a token-id batch: shard the batch dim over ``dp``."""
    return NamedSharding(mesh, PartitionSpec(AXIS_DP, None))


def param_named_shardings(model: nnx.Module, mesh: Mesh) -> dict[tuple[Any, ...], NamedSharding]:
    """Return the ``NamedSharding`` per parameter path (for inspection/tests)."""
    state = nnx.state(model)
    flat = nnx.to_flat_state(state)
    out: dict[tuple[Any, ...], NamedSharding] = {}
    for path, var in flat:
        axes = getattr(var, "logical_axes", None)
        out[tuple(path)] = NamedSharding(mesh, _spec_from_axes(axes))
    return out
