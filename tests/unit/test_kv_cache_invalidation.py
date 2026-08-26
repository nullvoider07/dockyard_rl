"""#3152: recompute_kv_cache_after_weight_updates was declared and read by nothing.

Declared at grpo.py:107 in AsyncGRPOConfig, set true in ten shipped configs,
and consumed nowhere in the tree. Behind it sat a real gap, narrower and
differently located than "the collective transport never invalidates":

  - Synchronous loop, vLLM: covered already. finish_generation calls
    invalidate_kv_cache after every rollout batch (vllm_generation.py:393), so
    the next sync_weights runs on an empty cache.
  - Colocated SGLang: covered already. HTTPWeightSynchronizer invalidates
    before its transfer, and that is SGLang's *only* flush -- its
    finish_generation deliberately preserves the prefix cache to keep
    RadixAttention warm (sglang_generation.py:439).
  - Asynchronous loop, vLLM: NOT covered. The cycle is prepare_for_refit ->
    sync_weights -> resume_after_refit (grpo.py:2885-2896) with no
    finish_generation anywhere, so generation resumed on a cache built from the
    previous weights. finish_generation appears only in the validation branch,
    which runs every val_period steps.

That is why the flag lives in AsyncGRPOConfig, and it is what these tests pin.

The plan's prescribed fix -- make every transport conditional, including HTTP --
is deliberately not implemented: it would have made SGLang's only flush
optional, turning a dead flag into a correctness regression.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dockyard_rl.models.generation.constants import SGLANG_BACKEND, VLLM_BACKEND
from dockyard_rl.weight_sync.factory import create_weight_synchronizer


class _FakeGeneration:
    def __init__(self, invalidate_result: bool = True) -> None:
        self.invalidate_calls = 0
        self._result = invalidate_result
        self.phases: list[Any] = []

    def invalidate_kv_cache(self) -> bool:
        self.invalidate_calls += 1
        return self._result

    def prepare_for_generation(self, *a, **kw) -> bool:
        self.phases.append(("prepare", kw.get("tags")))
        return True

    def update_weights_from_collective(self):
        return ["f"]

    def update_weights_via_ipc_zmq(self):
        return ["f"]

    def get_sglang_url_to_gpu_uuids(self):
        return {}


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
    """ray.get is the only Ray surface sync_weights touches."""
    import dockyard_rl.weight_sync.collective_weight_synchronizer as coll
    import dockyard_rl.weight_sync.http_weight_synchronizer as http
    import dockyard_rl.weight_sync.ipc_weight_synchronizer as ipc

    for mod in (coll, ipc, http):
        monkeypatch.setattr(mod.ray, "get", lambda futures, **kw: [True])


def _collective(generation, **kw):
    return create_weight_synchronizer(
        policy=_FakePolicy(), generation=generation,
        generation_backend=VLLM_BACKEND, colocated=False,
        train_cluster=object(), inference_world_size=8, **kw,
    )


def _ipc(generation, **kw):
    return create_weight_synchronizer(
        policy=_FakePolicy(), generation=generation,
        generation_backend=VLLM_BACKEND, colocated=True,
        refit_buffer_size_gb=1, **kw,
    )


class TestVllmTransportsInvalidateAfterRefit:
    """The async gap: without this, generation resumes on pre-refit KV blocks."""

    def test_collective_invalidates(self, no_ray):
        gen = _FakeGeneration()
        _collective(gen).sync_weights()
        assert gen.invalidate_calls == 1

    def test_ipc_invalidates(self, no_ray):
        gen = _FakeGeneration()
        _ipc(gen).sync_weights()
        assert gen.invalidate_calls == 1

    def test_collective_invalidates_after_the_transfer_not_before(self, no_ray):
        """Invalidating first would leave the window open for the new weights."""
        gen = _FakeGeneration()
        order: list[str] = []
        policy = _FakePolicy()
        policy.broadcast_weights_for_collective = (  # type: ignore[method-assign]
            lambda **kw: order.append("transfer") or ["f"]
        )
        gen.invalidate_kv_cache = (  # type: ignore[method-assign]
            lambda: order.append("invalidate") or True
        )
        create_weight_synchronizer(
            policy=policy, generation=gen, generation_backend=VLLM_BACKEND,
            colocated=False, train_cluster=object(), inference_world_size=8,
        ).sync_weights()
        assert order == ["transfer", "invalidate"]


class TestTheFlagIsActuallyRead:
    """It was declared in AsyncGRPOConfig and consumed nowhere."""

    def test_collective_honours_false(self, no_ray, caplog):
        gen = _FakeGeneration()
        with caplog.at_level(logging.WARNING):
            _collective(gen, recompute_kv_cache=False).sync_weights()
        assert gen.invalidate_calls == 0
        assert "recompute_kv_cache_after_weight_updates" in caplog.text

    def test_ipc_honours_false(self, no_ray, caplog):
        gen = _FakeGeneration()
        with caplog.at_level(logging.WARNING):
            _ipc(gen, recompute_kv_cache=False).sync_weights()
        assert gen.invalidate_calls == 0

    def test_default_is_enabled(self, no_ray):
        """A run that never sets the key must still be correct."""
        gen = _FakeGeneration()
        _collective(gen).sync_weights()
        assert gen.invalidate_calls == 1


class TestSglangFlushStaysUnconditional:
    """HTTP's call is SGLang's only flush; the flag must not reach it."""

    def test_http_invalidates_even_when_the_flag_is_false(self, no_ray):
        gen = _FakeGeneration()
        create_weight_synchronizer(
            policy=_FakePolicy(), generation=gen,
            generation_backend=SGLANG_BACKEND, colocated=True,
            recompute_kv_cache=False,
        ).sync_weights()
        assert gen.invalidate_calls == 1, (
            "making SGLang's flush conditional would turn a dead config flag "
            "into a stale-KV correctness bug"
        )


class TestFailuresDoNotAbortTheStep:
    """Weights are already in place; aborting would leave the engine mid-refit."""

    def test_a_failed_invalidation_is_logged_not_raised(self, no_ray, caplog):
        gen = _FakeGeneration(invalidate_result=False)
        with caplog.at_level(logging.WARNING):
            _collective(gen).sync_weights()
        assert "invalidation reported failure" in caplog.text

    def test_a_backend_without_the_method_is_tolerated(self, no_ray, caplog):
        class _NoInvalidate(_FakeGeneration):
            invalidate_kv_cache = None  # type: ignore[assignment]

        gen = _NoInvalidate()
        with caplog.at_level(logging.WARNING):
            _collective(gen).sync_weights()
        assert "no invalidate_kv_cache" in caplog.text
