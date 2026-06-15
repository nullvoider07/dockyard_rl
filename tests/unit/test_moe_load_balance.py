"""Tests for the aux-loss-free MoE load-balance bias updater (B.5b).

Pure CPU coverage of the sign-based bias math, the in-place block update
(counter reset, cross-rank reduce injection), and the optimizer-pre-hook
registration. No GPU / no grouped-GEMM forward (device-bound, HV-8/HV-14).
"""

from __future__ import annotations

from typing import Optional, cast

import torch

from dockyard_rl.models.dtensor.moe.block import MoEBlock
from dockyard_rl.models.dtensor.moe.dispatch import LocalTokenDispatcher
from dockyard_rl.models.dtensor.moe.experts import GroupedExperts
from dockyard_rl.models.dtensor.moe.load_balance import (
    compute_expert_bias_delta,
    iter_lb_moe_blocks,
    register_expert_bias_update_hook,
    update_expert_biases,
)
from dockyard_rl.models.dtensor.moe.router import TokenChoiceTopKRouter


def _make_moe_block(
    num_experts=4, dim=8, hidden=16, top_k=2, coeff: Optional[float] = 0.01
):
    router = TokenChoiceTopKRouter(dim=dim, num_experts=num_experts, top_k=top_k)
    dispatcher = LocalTokenDispatcher(num_experts=num_experts, top_k=top_k)
    experts = GroupedExperts(
        dim=dim, hidden_dim=hidden, num_experts=num_experts, dispatcher=dispatcher
    )
    return MoEBlock(
        router=router,
        experts=experts,
        num_experts=num_experts,
        load_balance_coeff=coeff,
    )


class TestComputeExpertBiasDelta:
    def test_underloaded_up_overloaded_down(self):
        tokens = torch.tensor([0.0, 0.0, 4.0, 4.0])  # mean 2
        delta = compute_expert_bias_delta(tokens, 0.01)
        assert torch.allclose(delta, torch.tensor([0.01, 0.01, -0.01, -0.01]))

    def test_zero_centered(self):
        tokens = torch.tensor([0.0, 1.0, 1.0, 1.0])  # asymmetric
        delta = compute_expert_bias_delta(tokens, 0.05)
        assert torch.allclose(delta.sum(), torch.tensor(0.0), atol=1e-6)

    def test_coeff_scales_linearly(self):
        tokens = torch.tensor([0.0, 0.0, 4.0, 4.0])
        d1 = compute_expert_bias_delta(tokens, 0.01)
        d2 = compute_expert_bias_delta(tokens, 0.02)
        assert torch.allclose(d2, 2.0 * d1)

    def test_uniform_load_no_change(self):
        tokens = torch.tensor([3.0, 3.0, 3.0, 3.0])
        delta = compute_expert_bias_delta(tokens, 0.01)
        assert torch.allclose(delta, torch.zeros(4))


class TestUpdateExpertBiases:
    def test_steps_bias_and_resets_counter(self):
        block = _make_moe_block(num_experts=4, coeff=0.01)
        counts = cast(torch.Tensor, block.tokens_per_expert_E)
        counts.copy_(torch.tensor([0.0, 0.0, 4.0, 4.0]))

        update_expert_biases([block])

        assert torch.allclose(
            cast(torch.Tensor, block.expert_bias_E),
            torch.tensor([0.01, 0.01, -0.01, -0.01]),
        )
        assert torch.allclose(counts, torch.zeros(4))

    def test_accumulates_across_steps(self):
        block = _make_moe_block(num_experts=4, coeff=0.01)
        counts = cast(torch.Tensor, block.tokens_per_expert_E)
        for _ in range(3):
            counts.copy_(torch.tensor([0.0, 0.0, 4.0, 4.0]))
            update_expert_biases([block])
        assert torch.allclose(
            cast(torch.Tensor, block.expert_bias_E),
            torch.tensor([0.03, 0.03, -0.03, -0.03]),
        )

    def test_reduce_fn_changes_outcome(self):
        # Local load is imbalanced; the cross-rank reduce sums in a complementary
        # shard, making the global load uniform -> zero delta.
        block = _make_moe_block(num_experts=4, coeff=0.01)
        cast(torch.Tensor, block.tokens_per_expert_E).copy_(
            torch.tensor([0.0, 0.0, 4.0, 4.0])
        )

        def reduce_fn(t):
            return t + torch.tensor([4.0, 4.0, 0.0, 0.0])

        update_expert_biases([block], reduce_tokens_fn=reduce_fn)
        assert torch.allclose(cast(torch.Tensor, block.expert_bias_E), torch.zeros(4))


class TestIterAndRegister:
    def test_iter_skips_disabled_blocks(self):
        enabled = _make_moe_block(coeff=0.01)
        disabled = _make_moe_block(coeff=None)
        model = torch.nn.ModuleList([enabled, disabled])
        found = list(iter_lb_moe_blocks(model))
        assert found == [enabled]

    def test_register_returns_false_without_lb_blocks(self):
        block = _make_moe_block(coeff=None)
        opt = torch.optim.SGD(block.parameters(), lr=0.0)
        assert register_expert_bias_update_hook(opt, block) is False

    def test_hook_fires_on_optimizer_step(self):
        block = _make_moe_block(num_experts=4, coeff=0.01)
        opt = torch.optim.SGD(block.parameters(), lr=0.0)
        assert register_expert_bias_update_hook(opt, block) is True

        counts = cast(torch.Tensor, block.tokens_per_expert_E)
        counts.copy_(torch.tensor([0.0, 0.0, 4.0, 4.0]))
        opt.step()

        assert torch.allclose(
            cast(torch.Tensor, block.expert_bias_E),
            torch.tensor([0.01, 0.01, -0.01, -0.01]),
        )
        assert torch.allclose(counts, torch.zeros(4))
