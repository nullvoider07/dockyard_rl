"""Tests for the per-sample streaming generate_async path (vLLM + SGLang).

Exercises the driver-side async generator: it routes a single sample to a
round-robin DP-leader worker, awaits the streamed ObjectRef, tags the result
with the leader index, and yields (original_idx, batch). The worker proxy and
its result ref are faked so no Ray/GPU is needed. Also checks the async_engine
gate raises when disabled.
"""

from __future__ import annotations

import asyncio

import pytest

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.models.generation.sglang import sglang_generation as sglang_mod
from dockyard_rl.models.generation.vllm import vllm_generation as vllm_mod

VllmGeneration = vllm_mod.VllmGeneration
SGLangGeneration = sglang_mod.SGLangGeneration


class _FakeRef:
    """Awaitable standing in for a Ray ObjectRef resolving to the sample tuple."""

    def __init__(self, value):
        self._value = value

    def __await__(self):
        async def _resolve():
            return self._value

        return _resolve().__await__()


class _FakeProxy:
    """Async iterator standing in for an ObjectRefGenerator (one chunk)."""

    def __init__(self, value):
        self._value = value
        self._done = False

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return _FakeRef(self._value)


class _FakeWorkerGroup:
    def __init__(self, result_value):
        self.dp_size = 2
        self.result_value = result_value
        self.calls: list[tuple[str, int]] = []

    def get_dp_leader_worker_idx(self, shard_idx: int) -> int:
        return shard_idx  # identity for the test

    def run_single_worker_single_data(self, method_name, worker_idx, data, greedy):
        self.calls.append((method_name, worker_idx))
        return _FakeProxy(self.result_value)

    def shutdown(self, *args, **kwargs):  # quiet SGLangGeneration.__del__ at teardown
        return True


def _single_sample() -> BatchedDataDict:
    return BatchedDataDict(
        {"input_ids": [[1, 2, 3]], "input_lengths": [3]}
    )


def _result_batch() -> BatchedDataDict:
    return BatchedDataDict({"output_ids": [[1, 2, 3, 4]]})


def _drain(gen):
    async def _run():
        return [item async for item in gen]

    return asyncio.run(_run())


class TestVllmGenerateAsync:
    def _gen(self):
        g = object.__new__(VllmGeneration)
        g.cfg = {"vllm_cfg": {"async_engine": True}}
        g.current_generate_dp_shard_idx = 0
        g.worker_group = _FakeWorkerGroup((7, _result_batch()))
        return g

    def test_yields_indexed_result_and_tags_leader(self):
        g = self._gen()
        out = _drain(g.generate_async(_single_sample(), greedy=False))
        assert len(out) == 1
        idx, batch = out[0]
        assert idx == 7
        assert batch["gen_leader_worker_idx"] == [0]
        assert g.worker_group.calls == [("generate_async", 0)]

    def test_round_robin_advances(self):
        g = self._gen()
        _drain(g.generate_async(_single_sample()))
        assert g.current_generate_dp_shard_idx == 1
        _drain(g.generate_async(_single_sample()))
        assert g.current_generate_dp_shard_idx == 0  # wraps at dp_size=2

    def test_disabled_async_engine_raises(self):
        g = self._gen()
        g.cfg = {"vllm_cfg": {"async_engine": False}}
        with pytest.raises(RuntimeError):
            _drain(g.generate_async(_single_sample()))

    def test_empty_input_yields_nothing(self):
        g = self._gen()
        empty = BatchedDataDict({"input_ids": [], "input_lengths": []})
        assert _drain(g.generate_async(empty)) == []


class TestSGLangGenerateAsync:
    def _gen(self):
        g = object.__new__(SGLangGeneration)
        g.cfg = {"sglang_cfg": {"async_engine": True}}
        g.current_generate_dp_shard_idx = 0
        g.worker_group = _FakeWorkerGroup((0, _result_batch()))
        return g

    def test_yields_indexed_result_and_tags_leader(self):
        g = self._gen()
        out = _drain(g.generate_async(_single_sample(), greedy=False))
        assert len(out) == 1
        idx, batch = out[0]
        assert idx == 0
        assert batch["gen_leader_worker_idx"] == [0]
        assert g.worker_group.calls == [("generate_async", 0)]

    def test_round_robin_advances(self):
        g = self._gen()
        _drain(g.generate_async(_single_sample()))
        assert g.current_generate_dp_shard_idx == 1

    def test_disabled_async_engine_raises(self):
        g = self._gen()
        g.cfg = {"sglang_cfg": {"async_engine": False}}
        with pytest.raises(RuntimeError):
            _drain(g.generate_async(_single_sample()))

    def test_empty_input_yields_nothing(self):
        g = self._gen()
        empty = BatchedDataDict({"input_ids": [], "input_lengths": []})
        assert _drain(g.generate_async(empty)) == []
