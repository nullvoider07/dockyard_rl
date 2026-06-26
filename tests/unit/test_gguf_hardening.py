"""GGUF dequant dimension guard (models/generation/vllm/quantization/gguf_hardening.py).

The guard rejects out-of-range dequant dimensions before vLLM's kernel runs, preventing
the int-truncation / uninitialized-GPU-memory leak (GHSA-5jv2-g5wq-cmr4). vLLM is
imported only to locate the op symbol; no GPU/kernel is invoked — the underlying op is
replaced with a recorder, so the test verifies the boundary check in isolation.
"""

from __future__ import annotations

import pytest


def test_guard_rejects_oversized_and_passes_valid(monkeypatch):
    ops = pytest.importorskip("vllm._custom_ops")
    from dockyard_rl.models.generation.vllm.quantization import gguf_hardening as gh

    calls = []
    monkeypatch.setattr(ops, "ggml_dequantize", lambda *a, **k: calls.append(a) or "ok")
    monkeypatch.setattr(gh, "_guard_installed", False)
    gh.install_gguf_dequant_guard()

    # Valid dims pass through to the (recorded) original op.
    assert ops.ggml_dequantize(object(), 0, 16, 32, None) == "ok"
    assert len(calls) == 1

    # Oversized n is rejected before the kernel.
    with pytest.raises(ValueError, match="GHSA-5jv2"):
        ops.ggml_dequantize(object(), 0, 16, 2**31, None)
    # Non-positive m is rejected too.
    with pytest.raises(ValueError, match="GHSA-5jv2"):
        ops.ggml_dequantize(object(), 0, 0, 32, None)
    assert len(calls) == 1  # the original op was not called for the rejected dims

    # Reset the module flag so a real engine init can install the guard.
    monkeypatch.setattr(gh, "_guard_installed", False)
