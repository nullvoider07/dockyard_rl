"""Tests for MoE parallel-dims factoring and rank->expert assignment.

Exercises the pure ``MoEParallelDims`` topology math (no torch.distributed, no
GPU): the world-size constraint, the derived degrees (dp/fsdp/efsdp), and the
rank->expert partition that the sparse mesh implies. ``build_moe_meshes`` is
device-bound and not exercised here.
"""

from __future__ import annotations

import pytest

from dockyard_rl.models.dtensor.moe.mesh import MoEParallelDims


class TestFactoringConstraints:
    def test_valid_dense_only(self):
        d = MoEParallelDims(world_size=8, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1)
        assert d.dp == 8 and d.fsdp == 8 and d.efsdp == 8 and not d.enable_ep

    def test_valid_with_ep_and_derived_degrees(self):
        # world = dr*ds*cp*tp = 1*8*1*1 = 8; ep=2 carved from ds*cp*tp=8 -> efsdp=4
        d = MoEParallelDims(world_size=8, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=2)
        assert d.dp == 8
        assert d.fsdp == 8
        assert d.efsdp == 4
        assert d.enable_ep
        # sparse mesh (dp_replicate, efsdp, ep) must re-cover all world ranks
        assert d.dp_replicate * d.efsdp * d.ep == d.world_size

    def test_hsdp_and_tp_and_cp(self):
        # world = dr2*ds4*cp2*tp2 = 32; ep=4 carved from ds*cp*tp=16 -> efsdp=4
        d = MoEParallelDims(world_size=32, dp_replicate=2, dp_shard=4, cp=2, tp=2, ep=4)
        assert d.dp == 8 and d.fsdp == 8 and d.efsdp == 4
        assert d.dp_replicate * d.efsdp * d.ep == d.world_size

    def test_world_mismatch_raises(self):
        with pytest.raises(ValueError, match="world_size"):
            MoEParallelDims(world_size=8, dp_replicate=1, dp_shard=4, cp=1, tp=1, ep=1)

    def test_ep_not_dividing_budget_raises(self):
        # ds*cp*tp = 4, ep=3 does not divide 4
        with pytest.raises(ValueError, match="must divide"):
            MoEParallelDims(world_size=4, dp_replicate=1, dp_shard=4, cp=1, tp=1, ep=3)

    def test_degree_below_one_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            MoEParallelDims(world_size=4, dp_replicate=1, dp_shard=4, cp=1, tp=1, ep=0)


class TestRankToExpert:
    def test_ep_coord_is_rank_mod_ep(self):
        d = MoEParallelDims(world_size=8, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=2)
        assert [d.ep_coord(r) for r in range(8)] == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_ep_coord_out_of_range_raises(self):
        d = MoEParallelDims(world_size=4, dp_replicate=1, dp_shard=4, cp=1, tp=1, ep=2)
        with pytest.raises(ValueError, match="out of range"):
            d.ep_coord(4)

    def test_num_local_experts(self):
        d = MoEParallelDims(world_size=8, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=4)
        assert d.num_local_experts(16) == 4

    def test_num_experts_indivisible_raises(self):
        d = MoEParallelDims(world_size=8, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=4)
        with pytest.raises(ValueError, match="divisible"):
            d.experts_for_rank(0, num_experts=10)

    def test_concrete_partition(self):
        # world=8, ep=2, 8 experts -> 4/rank; even ranks hold 0-3, odd hold 4-7
        d = MoEParallelDims(world_size=8, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=2)
        for r in (0, 2, 4, 6):
            assert d.experts_for_rank(r, 8) == [0, 1, 2, 3]
        for r in (1, 3, 5, 7):
            assert d.experts_for_rank(r, 8) == [4, 5, 6, 7]

    def test_partition_covers_all_experts_without_overlap(self):
        d = MoEParallelDims(world_size=16, dp_replicate=2, dp_shard=8, cp=1, tp=1, ep=4)
        num_experts = 32
        # one full EP group = the first `ep` consecutive ranks
        union: set[int] = set()
        for ep_rank in range(d.ep):
            experts = d.experts_for_rank(ep_rank, num_experts)
            assert len(experts) == num_experts // d.ep
            assert union.isdisjoint(experts), "EP partitions must not overlap"
            union.update(experts)
        assert union == set(range(num_experts)), "EP partitions must cover all experts"

    def test_same_partition_across_efsdp_and_replicate(self):
        # ranks sharing an ep coordinate (different efsdp/dp_replicate) replicate
        d = MoEParallelDims(world_size=16, dp_replicate=2, dp_shard=8, cp=1, tp=1, ep=4)
        num_experts = 32
        by_coord: dict[int, list[int]] = {}
        for r in range(d.world_size):
            c = d.ep_coord(r)
            experts = d.experts_for_rank(r, num_experts)
            if c in by_coord:
                assert by_coord[c] == experts, "same ep coord must hold same experts"
            else:
                by_coord[c] = experts
        assert len(by_coord) == d.ep


class TestFromConfig:
    def test_derives_dp_shard(self):
        cfg = {"tensor_parallel_size": 2, "context_parallel_size": 1,
               "expert_parallel_size": 2, "dp_replicate_size": 1}
        d = MoEParallelDims.from_dtensor_cfg(cfg, world_size=8)
        # dp_shard = 8 // (dr1*cp1*tp2) = 4
        assert d.dp_shard == 4 and d.tp == 2 and d.ep == 2
        assert d.dp_replicate * d.dp_shard * d.cp * d.tp == 8

    def test_defaults_to_dense(self):
        d = MoEParallelDims.from_dtensor_cfg({}, world_size=4)
        assert d.dp_shard == 4 and d.ep == 1 and not d.enable_ep

    def test_bad_world_raises(self):
        cfg = {"tensor_parallel_size": 3}
        with pytest.raises(ValueError, match="not divisible"):
            MoEParallelDims.from_dtensor_cfg(cfg, world_size=8)
