"""FP8 MoE refit composition (Phase 3b, slice 3b.1) — CPU coverage.

Locks the chain that serves the policy in FP8 while training stays bf16:
the trainer streams per-expert bf16 tensors (slices 3.1-3.3, refit.py expands
native fused GroupedExperts), the inference side
(``generation/vllm/quantization/fp8.py::load_weights``) block-casts each to FP8 +
appends ``_scale_inv``, then vLLM's Qwen3-MoE FusedMoE loader fuses them.

Two CPU-checkable invariants:
  1. the block-FP8 cast yields the per-expert FP8 weight + a block-scale whose
     shape matches what vLLM's per-expert ``w13/w2_weight_scale_inv`` consumes;
  2. the per-expert names AND their ``_scale_inv`` companions route through
     vLLM's expert-params mapping to the FusedMoE params
     (``experts.w13_weight``/``experts.w13_weight_scale_inv`` etc.).

The live FP8 grouped-GEMM + numerical parity are device-bound (HV-16). Source
verified: vllm/model_executor/layers/quantization/fp8.py::Fp8MoEMethod
(``weight_scale_name='weight_scale_inv'`` for block-quant) +
fused_moe/layer.py::make_expert_params_mapping.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest
import torch

_FP8_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "models",
    "generation",
    "vllm",
    "quantization",
    "fp8.py",
)


def _load_fp8_module() -> Any:
    """Load fp8.py directly (its top-level imports are vllm/torch only, no
    dockyard package), skipping if vLLM/triton are unavailable in the env."""
    pytest.importorskip("vllm")
    spec = importlib.util.spec_from_file_location("dockyard_fp8_direct", _FP8_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - env-dependent heavy imports
        pytest.skip(f"fp8.py not importable in this environment: {exc}")
    return module


class TestFp8BlockCast:
    """The block-wise FP8 cast on per-expert (2D) weights."""

    def test_gate_proj_shape_and_scale(self):
        fp8 = _load_fp8_module()
        fp8.global_fp8_config = fp8.FP8Config(model_parallel_size=1)
        # per-expert gate_proj weight is (F, D)
        F, D = 256, 128
        w = torch.randn(F, D, dtype=torch.float32)
        lp, scale = fp8.cast_tensor_to_fp8_blockwise(w, weight_block_size=[128, 128])
        scale = torch.squeeze(scale, dim=-1)
        assert lp.dtype == torch.float8_e4m3fn
        assert tuple(lp.shape) == (F, D)
        # one scale per 128x128 block: (F/128, D/128)
        assert tuple(scale.shape) == (F // 128, D // 128)

    def test_down_proj_shape_and_scale(self):
        fp8 = _load_fp8_module()
        fp8.global_fp8_config = fp8.FP8Config(model_parallel_size=1)
        # per-expert down_proj weight is (D, F)
        D, F = 128, 256
        w = torch.randn(D, F, dtype=torch.float32)
        lp, scale = fp8.cast_tensor_to_fp8_blockwise(w, weight_block_size=[128, 128])
        scale = torch.squeeze(scale, dim=-1)
        assert lp.dtype == torch.float8_e4m3fn
        assert tuple(lp.shape) == (D, F)
        assert tuple(scale.shape) == (D // 128, F // 128)


class TestFp8MoeNameRouting:
    """Per-expert weight + ``_scale_inv`` names route to the FusedMoE FP8 params.

    Replays vLLM's loader without importing it: for block-quant,
    ``weight_scale_name='weight_scale_inv'`` so the FusedMoE registers
    ``w13_weight_scale_inv``/``w2_weight_scale_inv``; the expert mapping matches
    ``experts.{i}.{proj}.`` as a substring of BOTH the weight and its
    ``_scale_inv`` companion, replacing with ``experts.w13_``/``experts.w2_``.
    """

    BASE = "model.layers.0.mlp.experts"

    def _vllm_expert_mapping(self, i: int) -> list[tuple[str, str]]:
        # (param_name, weight_name) for the three shards of expert i.
        return [
            ("experts.w13_", f"experts.{i}.gate_proj."),
            ("experts.w2_", f"experts.{i}.down_proj."),
            ("experts.w13_", f"experts.{i}.up_proj."),
        ]

    def _route(self, name: str, i: int) -> str:
        matches = [
            (param_name, weight_name)
            for (param_name, weight_name) in self._vllm_expert_mapping(i)
            if weight_name in name
        ]
        assert len(matches) == 1, (name, matches)
        param_name, weight_name = matches[0]
        return name.replace(weight_name, param_name)

    def test_weight_names_route_to_w13_w2(self):
        i = 0
        assert (
            self._route(f"{self.BASE}.{i}.gate_proj.weight", i)
            == f"{self.BASE}.w13_weight"
        )
        assert (
            self._route(f"{self.BASE}.{i}.up_proj.weight", i)
            == f"{self.BASE}.w13_weight"
        )
        assert (
            self._route(f"{self.BASE}.{i}.down_proj.weight", i)
            == f"{self.BASE}.w2_weight"
        )

    def test_scale_inv_names_route_to_w13_w2_scale_inv(self):
        # fp8.load_weights emits `k + "_scale_inv"` alongside the quantized weight.
        i = 3
        for proj, expected in (
            ("gate_proj", "w13_weight_scale_inv"),
            ("up_proj", "w13_weight_scale_inv"),
            ("down_proj", "w2_weight_scale_inv"),
        ):
            scale_name = f"{self.BASE}.{i}.{proj}.weight_scale_inv"
            assert self._route(scale_name, i) == f"{self.BASE}.{expected}"

    def test_scale_inv_not_caught_by_ignore_suffixes(self):
        # vLLM's loader skips names ending in the literal `_weight_scale` /
        # `.weight_scale`; the block-quant `_weight_scale_inv` must NOT match.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )
        routed = f"{self.BASE}.w13_weight_scale_inv"
        assert not routed.endswith(ignore_suffixes)
