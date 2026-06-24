"""Dummy-weight NaN-amax calibration guard (Wave 2) — CPU coverage.

`vllm_quant_patch.py` imports modelopt + vllm (GPU image only). To exercise the
real `_tolerate_dummy_weight_nan_amax` contextmanager on a plain host, the heavy
import chain is stubbed in `sys.modules` and the real file is exec-loaded; the
stub `MaxCalibrator` records the tensor its `collect` receives so the guard's
sanitize-then-delegate behavior and its restore-on-exit can be asserted. `torch`
is real.

The live calibration (real `mtq.quantize` over dummy vLLM weights producing the
actual non-finite cascade, then refit loading real amax over the -1.0 sentinel)
is GPU-only — tracked as HV-34.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Any

import torch


class _StubMaxCalibrator:
    """Stand-in for modelopt's MaxCalibrator; records the input it reduces."""

    last_input: Any = None

    def collect(self, x: torch.Tensor) -> torch.Tensor:
        type(self).last_input = x
        return x


def _register(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    if "." not in name or name in {
        "modelopt",
        "modelopt.torch",
        "modelopt.torch.quantization",
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
    }:
        mod.__path__ = []  # type: ignore[attr-defined]
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load_patch_module() -> Any:
    """Stub the modelopt/vllm import chain, then exec-load the real patch file."""

    from contextlib import contextmanager

    @contextmanager
    def _disable_compilation(_model: Any):
        yield

    _register("modelopt")
    _register("modelopt.torch")
    _register("modelopt.torch.quantization", quantize=lambda *a, **k: None,
              print_quant_summary=lambda *a, **k: None)
    _register("modelopt.torch.quantization.calib")
    _register("modelopt.torch.quantization.calib.max", MaxCalibrator=_StubMaxCalibrator)
    _register("modelopt.torch.quantization.nn")
    _register("modelopt.torch.quantization.nn.modules")
    _register("modelopt.torch.quantization.nn.modules.tensor_quantizer",
              TensorQuantizer=type("TensorQuantizer", (), {}))
    _register("modelopt.torch.quantization.plugins")
    _register("modelopt.torch.quantization.plugins.vllm",
              disable_compilation=_disable_compilation)
    _register("vllm")
    _register("vllm.v1")
    _register("vllm.v1.worker")
    _register("vllm.v1.worker.gpu_worker", Worker=type("Worker", (), {}))
    _register("dockyard_rl")
    _register("dockyard_rl.modelopt")
    _register("dockyard_rl.modelopt.utils", resolve_quant_cfg=lambda s: {})

    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "modelopt", "models",
        "generation", "vllm_quant_patch.py",
    )
    spec = importlib.util.spec_from_file_location("dockyard_quant_patch_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNanAmaxGuard:
    def test_all_nan_input_reduced_to_finite_zero(self):
        mod = _load_patch_module()
        inst = mod.MaxCalibrator()
        nan_x = torch.full((4,), float("nan"))
        with mod._tolerate_dummy_weight_nan_amax():
            mod.MaxCalibrator.collect(inst, nan_x)
        recorded = _StubMaxCalibrator.last_input
        assert torch.isfinite(recorded).all()
        assert float(recorded.abs().max()) == 0.0

    def test_mixed_nonfinite_sanitized(self):
        mod = _load_patch_module()
        inst = mod.MaxCalibrator()
        x = torch.tensor([1.0, float("inf"), float("-inf"), float("nan"), 2.0])
        with mod._tolerate_dummy_weight_nan_amax():
            mod.MaxCalibrator.collect(inst, x)
        recorded = _StubMaxCalibrator.last_input
        assert torch.isfinite(recorded).all()
        # finite magnitudes preserved, non-finite zeroed
        assert float(recorded.abs().max()) == 2.0

    def test_finite_input_untouched(self):
        mod = _load_patch_module()
        inst = mod.MaxCalibrator()
        finite = torch.tensor([1.0, -2.0, 3.0])
        with mod._tolerate_dummy_weight_nan_amax():
            mod.MaxCalibrator.collect(inst, finite)
        # same object passed straight through (no nan_to_num allocation)
        assert _StubMaxCalibrator.last_input is finite

    def test_collect_restored_on_normal_exit(self):
        mod = _load_patch_module()
        original = mod.MaxCalibrator.collect
        with mod._tolerate_dummy_weight_nan_amax():
            assert mod.MaxCalibrator.collect is not original
        assert mod.MaxCalibrator.collect is original

    def test_collect_restored_on_exception(self):
        mod = _load_patch_module()
        original = mod.MaxCalibrator.collect
        try:
            with mod._tolerate_dummy_weight_nan_amax():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert mod.MaxCalibrator.collect is original
