"""Tests for HF MoE model surgery (Phase 3, B.5a) on a tiny Qwen3-MoE config.

CPU-validates the surgery PLAN against the real transformers Qwen3-MoE module
(transformers 5.x fused-expert layout): the module tree after surgery, the fused
native param names/shapes, the gate/up split correctness, and router-config
propagation. The live grouped-GEMM forward, FSDP wrap, and 30B weight load are
device-bound (HV-15) and not exercised here.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

pytest.importorskip("transformers")

from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from dockyard_rl.models.dtensor.moe.block import MoEBlock
from dockyard_rl.models.dtensor.moe.detect import is_moe_state_dict
from dockyard_rl.models.dtensor.moe.dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)
from dockyard_rl.models.dtensor.moe.experts import GroupedExperts
from dockyard_rl.models.dtensor.moe.router import TokenChoiceTopKRouter
from dockyard_rl.models.dtensor.moe.surgery import (
    apply_moe_surgery,
    is_moe_surgery_supported,
)

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 32
MOE_INTERMEDIATE = 16  # F


def _tiny_qwen3_moe() -> Qwen3MoeForCausalLM:
    cfg = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=MOE_INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        decoder_sparse_step=1,   # every layer is a routed-MoE layer
        mlp_only_layers=[],
        norm_topk_prob=True,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg)


def _capture_hf_experts(model):
    """Snapshot the HF fused expert + gate tensors before surgery replaces them."""
    snap = []
    for layer in model.model.layers:
        snap.append(
            (
                layer.mlp.experts.gate_up_proj.detach().clone(),
                layer.mlp.experts.down_proj.detach().clone(),
                layer.mlp.gate.weight.detach().clone(),
            )
        )
    return snap


class TestQwen3MoeSurgery:
    def test_module_tree_and_count(self):
        model = _tiny_qwen3_moe()
        n = apply_moe_surgery(model, dtensor_cfg={})
        assert n == 2
        for layer in model.model.layers:
            assert isinstance(layer.mlp, MoEBlock)
            assert isinstance(layer.mlp.experts, GroupedExperts)
            assert isinstance(layer.mlp.router, TokenChoiceTopKRouter)
            assert isinstance(layer.mlp.experts.dispatcher, LocalTokenDispatcher)

    def test_fused_param_shapes(self):
        model = _tiny_qwen3_moe()
        apply_moe_surgery(model, dtensor_cfg={})
        for layer in model.model.layers:
            e = layer.mlp.experts
            assert e.w1_EFD.shape == (NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN)
            assert e.w3_EFD.shape == (NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN)
            assert e.w2_EDF.shape == (NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE)

    def test_gate_up_split_and_down_values(self):
        model = _tiny_qwen3_moe()
        snap = _capture_hf_experts(model)
        apply_moe_surgery(model, dtensor_cfg={})
        for (gate_up, down, gate_w), layer in zip(snap, model.model.layers):
            e = layer.mlp.experts
            assert torch.equal(cast(torch.Tensor, e.w1_EFD), gate_up[:, :MOE_INTERMEDIATE, :])  # gate
            assert torch.equal(cast(torch.Tensor, e.w3_EFD), gate_up[:, MOE_INTERMEDIATE:, :])  # up
            assert torch.equal(cast(torch.Tensor, e.w2_EDF), down)
            assert torch.equal(cast(torch.Tensor, layer.mlp.router.gate.weight), gate_w)

    def test_router_config_propagation(self):
        model = _tiny_qwen3_moe()
        apply_moe_surgery(model, dtensor_cfg={})
        router = model.model.layers[0].mlp.router
        assert router.top_k == TOP_K
        assert router.num_experts == NUM_EXPERTS
        assert router.score_func == "softmax"
        assert router.route_norm is True  # norm_topk_prob

    def test_fused_state_dict_names(self):
        model = _tiny_qwen3_moe()
        apply_moe_surgery(model, dtensor_cfg={})
        keys = list(model.state_dict().keys())
        assert is_moe_state_dict(keys)
        assert any(k.endswith("mlp.experts.w1_EFD") for k in keys)
        assert any(k.endswith("mlp.experts.w2_EDF") for k in keys)
        assert any(k.endswith("mlp.experts.w3_EFD") for k in keys)
        # The fused HF param names are gone (surgery replaced the module).
        assert not any("gate_up_proj" in k for k in keys)

    def test_load_balance_coeff_threaded(self):
        model = _tiny_qwen3_moe()
        apply_moe_surgery(model, dtensor_cfg={}, load_balance_coeff=0.001)
        block = model.model.layers[0].mlp
        assert block.load_balance_coeff == 0.001
        assert block.expert_bias_E is not None
        assert block.expert_bias_E.shape == (NUM_EXPERTS,)


class TestSurgeryGuards:
    def test_alltoall_builds_alltoall_dispatcher(self):
        # ep>1 surgery now constructs AllToAll dispatchers (ep_mesh wired later
        # during _parallelize); it no longer raises HV-10.
        model = _tiny_qwen3_moe()
        cfg = {"moe_parallelizer": {"token_dispatcher": "alltoall"}}
        apply_moe_surgery(model, dtensor_cfg=cfg)
        for layer in model.model.layers:
            disp = layer.mlp.experts.dispatcher
            assert isinstance(disp, AllToAllTokenDispatcher)
            assert disp.ep_mesh is None

    def test_unsupported_arch_raises(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            apply_moe_surgery(torch.nn.Linear(4, 4), dtensor_cfg={})

    def test_is_moe_surgery_supported(self):
        assert is_moe_surgery_supported(_tiny_qwen3_moe()) is True
        assert is_moe_surgery_supported(torch.nn.Linear(4, 4)) is False


class TestQwen3MoeParallelPlan:
    """The dense-side TP plan registered for the surgically-converted arch."""

    def test_registered_and_attention_only(self):
        from dockyard_rl.models.dtensor.parallelize import (
            PARALLIZE_FUNCTIONS,
            _parallelize_qwen3_moe,
        )
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeForCausalLM as _Cls,
        )

        assert PARALLIZE_FUNCTIONS[_Cls] is _parallelize_qwen3_moe
        plan = _parallelize_qwen3_moe(_tiny_qwen3_moe(), sequence_parallel=False)
        # Attention + embeddings + head only; no MoE/expert/router/mlp keys (the
        # EP path shards experts, the router gate is TP-replicated).
        assert any("self_attn.q_proj" in k for k in plan)
        assert "model.embed_tokens" in plan
        assert "lm_head" in plan
        assert not any("mlp" in k or "expert" in k or "router" in k for k in plan)

    def test_sequence_parallel_rejected(self):
        from dockyard_rl.models.dtensor.parallelize import _parallelize_qwen3_moe

        with pytest.raises(AssertionError, match="sequence_parallel"):
            _parallelize_qwen3_moe(_tiny_qwen3_moe(), sequence_parallel=True)
