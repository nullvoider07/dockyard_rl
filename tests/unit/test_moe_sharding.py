"""Tests for the MoE expert sharding placement policy (pure, no GPU).

Asserts the ordered DTensor placements each routed-expert weight / router gate
takes for both the EP-on (sparse mesh) and EP-off (dense TP) cases. The
device-bound application is exercised at bring-up, not here.
"""

from __future__ import annotations

import pytest
from torch.distributed.tensor.placement_types import Replicate, Shard

from dockyard_rl.models.dtensor.moe.sharding import (
    num_local_experts,
    routed_expert_placements,
    router_gate_placements,
)

SPARSE = ("dp_replicate", "efsdp", "ep")
DENSE = ("dp_replicate", "dp_shard", "cp", "tp")


class TestRoutedExpertEP:
    def test_ep_on_shards0_on_ep_only(self):
        for role in ("gate_up", "down"):
            pl = routed_expert_placements(SPARSE, enable_ep=True, role=role)
            assert pl == [Replicate(), Replicate(), Shard(0)]

    def test_ep_on_role_irrelevant(self):
        a = routed_expert_placements(SPARSE, enable_ep=True, role="gate_up")
        b = routed_expert_placements(SPARSE, enable_ep=True, role="down")
        assert a == b


class TestRoutedExpertDenseTP:
    def test_ep_off_gate_up_colwise_shard1_on_tp(self):
        pl = routed_expert_placements(DENSE, enable_ep=False, role="gate_up")
        assert pl == [Replicate(), Replicate(), Replicate(), Shard(1)]

    def test_ep_off_down_rowwise_shard2_on_tp(self):
        pl = routed_expert_placements(DENSE, enable_ep=False, role="down")
        assert pl == [Replicate(), Replicate(), Replicate(), Shard(2)]

    def test_ep_off_no_tp_axis_all_replicate(self):
        # If the mesh has no "tp" dim, nothing is sharded.
        pl = routed_expert_placements(("dp_replicate", "dp_shard"), enable_ep=False, role="gate_up")
        assert pl == [Replicate(), Replicate()]


class TestRouterGate:
    def test_router_replicated_everywhere(self):
        assert router_gate_placements(SPARSE) == [Replicate(), Replicate(), Replicate()]
        assert router_gate_placements(DENSE) == [Replicate()] * 4


class TestNumLocalExperts:
    def test_divides(self):
        assert num_local_experts(16, 4) == 4
        assert num_local_experts(8, 1) == 8

    @pytest.mark.parametrize("E,ep", [(10, 4), (0, 2), (8, 0)])
    def test_invalid_raises(self, E, ep):
        with pytest.raises(ValueError):
            num_local_experts(E, ep)
