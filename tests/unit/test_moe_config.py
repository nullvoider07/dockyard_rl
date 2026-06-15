"""Tests for the MoE/EP config surface (B.7).

Pure CPU coverage of ``moe_parallelizer`` resolution, the early
divisibility/dispatcher-consistency validation, the dispatcher factory (incl.
the honest ``alltoall`` HV-10 NotImplementedError), and the routed-expert-count
extraction. No torch.distributed / GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dockyard_rl.models.dtensor.moe.config import (
    _extract_num_routed_experts,
    build_token_dispatcher,
    resolve_moe_parallelizer,
    validate_moe_parallel_config,
)
from dockyard_rl.models.dtensor.moe.dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)


class TestResolveMoeParallelizer:
    def test_defaults_when_absent(self):
        assert resolve_moe_parallelizer({}) == {
            "token_dispatcher": "local",
            "grouped_gemm": True,
            "load_balance_coeff": None,
        }

    def test_explicit_alltoall(self):
        cfg = {"moe_parallelizer": {"token_dispatcher": "alltoall", "grouped_gemm": True}}
        assert resolve_moe_parallelizer(cfg)["token_dispatcher"] == "alltoall"

    def test_unknown_dispatcher_rejected(self):
        cfg = {"moe_parallelizer": {"token_dispatcher": "deepep"}}
        with pytest.raises(ValueError, match="token_dispatcher"):
            resolve_moe_parallelizer(cfg)

    def test_grouped_gemm_false_rejected(self):
        cfg = {"moe_parallelizer": {"grouped_gemm": False}}
        with pytest.raises(ValueError, match="grouped_gemm"):
            resolve_moe_parallelizer(cfg)

    def test_load_balance_coeff_passthrough(self):
        cfg = {"moe_parallelizer": {"load_balance_coeff": 0.001}}
        assert resolve_moe_parallelizer(cfg)["load_balance_coeff"] == 0.001

    def test_load_balance_coeff_non_positive_rejected(self):
        cfg = {"moe_parallelizer": {"load_balance_coeff": 0.0}}
        with pytest.raises(ValueError, match="load_balance_coeff"):
            resolve_moe_parallelizer(cfg)

    def test_load_balance_coeff_bool_rejected(self):
        # bool is an int subclass; a stray `true` must not be read as a coeff.
        cfg = {"moe_parallelizer": {"load_balance_coeff": True}}
        with pytest.raises(ValueError, match="load_balance_coeff"):
            resolve_moe_parallelizer(cfg)


class TestValidateMoeParallelConfig:
    def test_dense_ep1_ok(self):
        dims = validate_moe_parallel_config({}, world_size=8)
        assert dims.ep == 1 and not dims.enable_ep

    def test_local_dispatcher_with_ep_gt_1_rejected(self):
        cfg = {
            "expert_parallel_size": 2,
            "moe_parallelizer": {"token_dispatcher": "local"},
        }
        with pytest.raises(ValueError, match="token_dispatcher='local'"):
            validate_moe_parallel_config(cfg, world_size=8)

    def test_alltoall_ep_gt_1_ok_with_divisible_experts(self):
        cfg = {
            "expert_parallel_size": 2,
            "moe_parallelizer": {"token_dispatcher": "alltoall"},
        }
        dims = validate_moe_parallel_config(cfg, world_size=8, num_experts=128)
        assert dims.ep == 2 and dims.num_local_experts(128) == 64

    def test_bad_world_divisibility_rejected(self):
        cfg = {"tensor_parallel_size": 3}
        with pytest.raises(ValueError, match="world_size"):
            validate_moe_parallel_config(cfg, world_size=8)

    def test_num_experts_not_divisible_by_ep_rejected(self):
        cfg = {
            "expert_parallel_size": 2,
            "moe_parallelizer": {"token_dispatcher": "alltoall"},
        }
        with pytest.raises(ValueError, match="divisible by ep"):
            validate_moe_parallel_config(cfg, world_size=8, num_experts=127)


class TestBuildTokenDispatcher:
    def test_local_builds_local_dispatcher(self):
        disp = build_token_dispatcher({}, num_experts=8, top_k=2)
        assert isinstance(disp, LocalTokenDispatcher)
        assert disp.num_experts == 8 and disp.top_k == 2 and disp.score_before_experts

    def test_local_respects_score_before_experts(self):
        disp = build_token_dispatcher(
            {}, num_experts=4, top_k=1, score_before_experts=False
        )
        assert disp.score_before_experts is False

    def test_alltoall_builds_alltoall_dispatcher(self):
        cfg = {"moe_parallelizer": {"token_dispatcher": "alltoall"}}
        disp = build_token_dispatcher(cfg, num_experts=8, top_k=2)
        assert isinstance(disp, AllToAllTokenDispatcher)
        assert disp.num_experts == 8 and disp.top_k == 2
        # EP mesh is wired later (during _parallelize); unwired => local fallback.
        assert disp.ep_mesh is None


class TestExtractNumRoutedExperts:
    def test_qwen3_moe_num_experts(self):
        cfg = SimpleNamespace(num_experts=128)
        assert _extract_num_routed_experts(cfg) == 128

    def test_deepseek_n_routed_experts(self):
        cfg = SimpleNamespace(n_routed_experts=256)
        assert _extract_num_routed_experts(cfg) == 256

    def test_nested_text_config(self):
        cfg = SimpleNamespace(text_config=SimpleNamespace(num_experts=64))
        assert _extract_num_routed_experts(cfg) == 64

    def test_dense_model_returns_none(self):
        cfg = SimpleNamespace(hidden_size=4096)
        assert _extract_num_routed_experts(cfg) is None

    def test_zero_or_invalid_ignored(self):
        cfg = SimpleNamespace(num_experts=0, n_routed_experts=8)
        assert _extract_num_routed_experts(cfg) == 8
