"""#3430a: the multimodal encoder cache was never invalidated across a refit.

vLLM caches multimodal encoder outputs -- vision embeddings produced by the
vision tower. Those are weight-dependent, and dockyard refits every step, so
the entries go stale immediately. vLLM clears this cache inside its own
reload_weights; dockyard's refit path (collective / IPC) does not go through
that, so nothing ever told the dependent cache its input had changed.

Consequence: a VLM run serves vision embeddings computed from the *initial*
weights for the entire run while the language model keeps advancing. Nothing
errors, and the only symptom is that vision grounding silently fails to improve.

Before this, dockyard had no reference to an encoder cache anywhere in its vLLM
layer. reset_encoder_cache was verified present in the pinned vLLM 0.27.1 --
AsyncLLM.reset_encoder_cache (async_llm.py:948, a coroutine) and the sync twin
LLMEngine.reset_encoder_cache (llm_engine.py:353), both reaching
scheduler.encoder_cache_manager.reset(). The manager is constructed
unconditionally, so the call is a no-op on text-only models rather than an
error.

Scope note: the plan expected three refit hook points, mirroring upstream.
dockyard has one -- the post-refit invalidation shared by the two vLLM
transports -- so that is where it lands.
"""

from __future__ import annotations

import logging

import pytest

from dockyard_rl.models.generation.constants import SGLANG_BACKEND, VLLM_BACKEND
from dockyard_rl.weight_sync.factory import create_weight_synchronizer


class _FakeGeneration:
    def __init__(self, encoder_result: bool = True) -> None:
        self.kv_calls = 0
        self.encoder_calls = 0
        self.order: list[str] = []
        self._encoder_result = encoder_result

    def invalidate_kv_cache(self) -> bool:
        self.kv_calls += 1
        self.order.append("kv")
        return True

    def invalidate_encoder_cache(self) -> bool:
        self.encoder_calls += 1
        self.order.append("encoder")
        return self._encoder_result

    def prepare_for_generation(self, *a, **kw) -> bool:
        return True

    def update_weights_from_collective(self):
        return ["f"]

    def update_weights_via_ipc_zmq(self):
        return ["f"]

    def get_sglang_url_to_gpu_uuids(self):
        return {}


class _SglangGeneration(_FakeGeneration):
    """SGLang exposes no encoder-cache reset; its absence is not a gap."""

    invalidate_encoder_cache = None  # type: ignore[assignment]


class _FakePolicy:
    def offload_before_refit(self): ...
    def offload_after_refit(self): ...
    def broadcast_weights_for_collective(self, **kw):
        return ["f"]
    def stream_weights_via_ipc_zmq(self, **kw):
        return ["f"]
    def stream_weights_via_http(self, **kw):
        return ["f"]


@pytest.fixture
def no_ray(monkeypatch):
    import dockyard_rl.weight_sync.collective_weight_synchronizer as coll
    import dockyard_rl.weight_sync.http_weight_synchronizer as http
    import dockyard_rl.weight_sync.ipc_weight_synchronizer as ipc

    for mod in (coll, ipc, http):
        monkeypatch.setattr(mod.ray, "get", lambda futures, **kw: [True])


def _sync(generation, backend=VLLM_BACKEND, colocated=False, **kw):
    extra = {} if colocated else {"train_cluster": object(), "inference_world_size": 8}
    if colocated and backend == VLLM_BACKEND:
        extra["refit_buffer_size_gb"] = 1
    return create_weight_synchronizer(
        policy=_FakePolicy(), generation=generation,
        generation_backend=backend, colocated=colocated, **extra, **kw,
    )


class TestEncoderCacheIsInvalidated:
    def test_collective_refit_resets_the_encoder_cache(self, no_ray):
        gen = _FakeGeneration()
        _sync(gen).sync_weights()
        assert gen.encoder_calls == 1

    def test_ipc_refit_resets_the_encoder_cache(self, no_ray):
        gen = _FakeGeneration()
        _sync(gen, colocated=True).sync_weights()
        assert gen.encoder_calls == 1

    def test_it_happens_after_the_weight_transfer(self, no_ray):
        gen = _FakeGeneration()
        _sync(gen).sync_weights()
        assert gen.order == ["kv", "encoder"]


class TestItIsNotGatedOnTheKvFlag:
    """Serving embeddings from superseded weights is never wanted."""

    def test_encoder_cache_resets_even_when_kv_recompute_is_disabled(self, no_ray):
        gen = _FakeGeneration()
        _sync(gen, recompute_kv_cache=False).sync_weights()
        assert gen.kv_calls == 0
        assert gen.encoder_calls == 1


class TestBackendsWithoutTheMethod:
    def test_sglang_is_skipped_without_a_warning(self, no_ray, caplog):
        gen = _SglangGeneration()
        with caplog.at_level(logging.WARNING):
            _sync(gen, backend=SGLANG_BACKEND, colocated=True).sync_weights()
        assert "Encoder cache" not in caplog.text


class TestFailureDoesNotAbortTheStep:
    def test_a_failed_reset_is_logged_not_raised(self, no_ray, caplog):
        gen = _FakeGeneration(encoder_result=False)
        with caplog.at_level(logging.WARNING):
            _sync(gen).sync_weights()
        assert "Encoder cache invalidation reported failure" in caplog.text


class TestUpstreamApiStillExists:
    """Pins the vLLM contract this fix depends on; a bump that drops it fails here."""

    def test_vllm_exposes_reset_encoder_cache_on_both_engines(self):
        import inspect

        pytest.importorskip("vllm")
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.engine.llm_engine import LLMEngine

        assert hasattr(AsyncLLM, "reset_encoder_cache")
        assert inspect.iscoroutinefunction(AsyncLLM.reset_encoder_cache)
        assert hasattr(LLMEngine, "reset_encoder_cache")
        assert not inspect.iscoroutinefunction(LLMEngine.reset_encoder_cache)

    def test_the_worker_methods_match_those_shapes(self):
        pytest.importorskip("vllm")
        from dockyard_rl.models.generation.vllm.vllm_worker import (
            VllmGenerationWorkerImpl,
        )

        assert hasattr(VllmGenerationWorkerImpl, "reset_encoder_cache")
