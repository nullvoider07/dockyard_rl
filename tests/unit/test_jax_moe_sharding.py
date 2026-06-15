"""J8b: MoE EP mesh axis + expert/router sharding (pure factoring + specs).

The pure factoring (``MoEMeshDims``) and the expert ``PartitionSpec`` mapping are
CPU-validatable. A live multi-device EP mesh (and the all-to-all numerics, J8c)
is hardware-deferred; here a 4-device host-platform mesh checks that expert
params actually place with the expert dim sharded over ``ep``.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")

from jax.sharding import PartitionSpec

from dockyard_rl.models.jax.moe.dispatch import LocalTokenDispatcher
from dockyard_rl.models.jax.moe.experts import GroupedExperts
from dockyard_rl.models.jax.sharding import (
    AXIS_EP,
    AXIS_TP,
    MoEMeshDims,
    build_moe_mesh,
    param_named_shardings,
    shard_model,
)


# --- pure factoring ---

def test_moe_mesh_dims_factoring():
    dims = MoEMeshDims(dp=2, cp=1, tp=2, ep=2)
    assert dims.world_size == 8
    assert dims.enable_ep is True
    assert dims.num_local_experts(8) == 4
    dims.validate_devices(8)
    with pytest.raises(ValueError):
        dims.validate_devices(4)  # world 8 > 4 devices


def test_moe_mesh_dims_validation():
    with pytest.raises(ValueError):
        MoEMeshDims(dp=0)
    with pytest.raises(ValueError):
        MoEMeshDims(ep=3).num_local_experts(8)  # 8 % 3 != 0
    assert MoEMeshDims(ep=1).enable_ep is False


def test_expert_logical_axes_present():
    # The GroupedExperts params carry the ep/tp logical axes (inert without a mesh).
    ge = GroupedExperts(8, 16, 4, LocalTokenDispatcher(4, 1), rngs=nnx.Rngs(params=0))
    assert ge.w1_EFD.logical_axes == (AXIS_EP, AXIS_TP, None)
    assert ge.w2_EDF.logical_axes == (AXIS_EP, None, AXIS_TP)
    assert ge.w3_EFD.logical_axes == (AXIS_EP, AXIS_TP, None)


# --- spec mapping + live placement on a 4-device host mesh ---

@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 host devices")
def test_expert_partition_specs_and_placement():
    # ep=2, tp=2, dp=1: expert dim over ep, hidden over tp.
    mesh = build_moe_mesh(dp=1, tp=2, cp=1, ep=2)
    ge = GroupedExperts(8, 16, 4, LocalTokenDispatcher(4, 1), rngs=nnx.Rngs(params=0))

    specs = param_named_shardings(ge, mesh)
    w1 = specs[("w1_EFD",)].spec
    w2 = specs[("w2_EDF",)].spec
    assert w1 == PartitionSpec(AXIS_EP, AXIS_TP, None)
    assert w2 == PartitionSpec(AXIS_EP, None, AXIS_TP)

    shard_model(ge, mesh)
    # expert dim (4) shards over ep=2 -> 2 experts per shard; hidden (16) over tp=2.
    s1 = ge.w1_EFD[...].sharding
    assert s1.shard_shape((4, 16, 8)) == (2, 8, 8)
