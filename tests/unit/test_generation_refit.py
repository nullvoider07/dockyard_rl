"""Tests for the WeightSynchronizer transports.

Validates that each transport drives the right backend path:

- Collective (non-colocated): NCCL broadcast + update_from_collective, no
  phase transitions, raises on a failed update;
- IPC (colocated vLLM): offload -> CUDA-IPC ZMQ -> restore, with the staging
  buffer size derived from free memory; raises on a failed update;
- HTTP (colocated SGLang): offload -> url discovery -> KV invalidation ->
  trainer HTTP stream -> restore.

The policy/generation stand-ins record the methods each transport invokes;
``ray.get`` is patched to an identity collect so plain lists flow through the
success check. The recording policy also implements the offload hooks the
colocated transports drive.
"""

from __future__ import annotations

import pytest
import torch

from dockyard_rl.models.dtensor.moe.refit import iter_expanded_refit_tensors
from dockyard_rl.weight_sync import collective_weight_synchronizer as coll_mod
from dockyard_rl.weight_sync import http_weight_synchronizer as http_mod
from dockyard_rl.weight_sync import ipc_weight_synchronizer as ipc_mod
from dockyard_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from dockyard_rl.weight_sync.http_weight_synchronizer import HTTPWeightSynchronizer
from dockyard_rl.weight_sync.ipc_weight_synchronizer import IPCWeightSynchronizer


class _RecordingPolicy:
    """Records which trainer-side weight-export / offload method ran."""

    def __init__(self, free_bytes: int = 100 * 1024**3) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._free_bytes = free_bytes

    def get_free_memory_bytes(self) -> int:
        return self._free_bytes

    def offload_before_refit(self) -> None:
        self.calls.append(("offload_before", {}))

    def offload_after_refit(self) -> None:
        self.calls.append(("offload_after", {}))

    def stream_weights_via_ipc_zmq(self, **kw):
        self.calls.append(("ipc_zmq", kw))
        return ["train"]

    def broadcast_weights_for_collective(self, **kw):
        self.calls.append(("collective", kw))
        return ["train"]

    def stream_weights_via_http(self, **kw):
        self.calls.append(("http", kw))
        return ["train"]


@pytest.fixture(autouse=True)
def _identity_ray_get(monkeypatch):
    """ray.get(x) -> x so list stand-ins flow through the success check."""
    for mod in (coll_mod, ipc_mod, http_mod):
        monkeypatch.setattr(mod.ray, "get", lambda x, **kw: x)


class TestCollective:
    """vLLM non-colocated path."""

    def _gen(self, inference_result):
        class _Gen:
            def update_weights_from_collective(self):
                return inference_result

        return _Gen()

    def test_uses_collective_no_phase_transitions(self):
        policy = _RecordingPolicy()
        ws = CollectiveWeightSynchronizer(
            policy, self._gen([True, True]),
            train_cluster=None, inference_world_size=8,
        )
        ws.sync_weights(kv_scales=None)
        assert [c[0] for c in policy.calls] == ["collective"]
        assert ws.is_stale is False

    def test_failure_when_worker_reports_false(self):
        policy = _RecordingPolicy()
        ws = CollectiveWeightSynchronizer(
            policy, self._gen([True, False]),
            train_cluster=None, inference_world_size=8,
        )
        with pytest.raises(RuntimeError):
            ws.sync_weights()
        assert ws.is_stale is True


class TestIPC:
    """vLLM colocated path."""

    def _gen(self, inference_result):
        class _Gen:
            def __init__(self):
                self.phases: list = []

            def update_weights_via_ipc_zmq(self):
                return inference_result

            def prepare_for_generation(self, tags=None):
                self.phases.append(tags)

        return _Gen()

    def test_offload_ipc_restore_with_computed_buffer(self, monkeypatch):
        monkeypatch.setenv("DOCKYARD_REFIT_BUFFER_MEMORY_RATIO", "0.5")
        policy = _RecordingPolicy(free_bytes=2 * 1024**3)
        gen = self._gen([True])
        ws = IPCWeightSynchronizer(policy, gen)
        ws.sync_weights()
        assert [c[0] for c in policy.calls] == [
            "offload_before", "ipc_zmq", "offload_after",
        ]
        ipc_call = next(c for c in policy.calls if c[0] == "ipc_zmq")
        # 0.5 * 2GiB = 1GiB
        assert ipc_call[1]["buffer_size_bytes"] == 1024**3
        assert gen.phases == [["weights"], ["kv_cache"]]
        assert ws.is_stale is False

    def test_explicit_buffer_size_overrides_ratio(self):
        policy = _RecordingPolicy()
        ws = IPCWeightSynchronizer(policy, self._gen([True]), refit_buffer_size_gb=4)
        ws.sync_weights()
        ipc_call = next(c for c in policy.calls if c[0] == "ipc_zmq")
        assert ipc_call[1]["buffer_size_bytes"] == 4 * 1024**3

    def test_failure_raises_and_restores_phase(self):
        policy = _RecordingPolicy()
        gen = self._gen([True, False])
        ws = IPCWeightSynchronizer(policy, gen)
        with pytest.raises(RuntimeError):
            ws.sync_weights()
        # finally-block restores the kv_cache phase even on failure.
        assert gen.phases == [["weights"], ["kv_cache"]]
        assert ws.is_stale is True


class TestHTTP:
    """SGLang colocated path."""

    def _gen(self, *, kv_ok=True, url_map=None):
        class _Gen:
            def __init__(self):
                self.order: list[str] = []
                self.phases: list = []

            def get_sglang_url_to_gpu_uuids(self):
                self.order.append("get_urls")
                return url_map if url_map is not None else {"http://s0": ["GPU-0"]}

            def invalidate_kv_cache(self):
                self.order.append("invalidate")
                return kv_ok

            def prepare_for_generation(self, tags=None):
                self.phases.append(tags)

        return _Gen()

    def test_http_push_order_and_success(self):
        policy = _RecordingPolicy()
        gen = self._gen()
        ws = HTTPWeightSynchronizer(policy, gen)
        ws.sync_weights()
        # URL discovery and KV invalidation precede the trainer HTTP stream.
        assert gen.order == ["get_urls", "invalidate"]
        http_call = next(c for c in policy.calls if c[0] == "http")
        assert http_call[1]["sglang_url_to_gpu_uuids"] == {"http://s0": ["GPU-0"]}
        assert gen.phases == [["weights"], ["kv_cache"]]
        assert ws.is_stale is False

    def test_succeeds_even_if_kv_invalidation_fails(self):
        policy = _RecordingPolicy()
        gen = self._gen(kv_ok=False)
        ws = HTTPWeightSynchronizer(policy, gen)
        ws.sync_weights()
        assert [c[0] for c in policy.calls if c[0] == "http"] == ["http"]
        assert ws.is_stale is False


class _MoERefitPolicy:
    """Policy stand-in whose collective broadcast streams an EP-resharded MoE
    state dict through the real ``iter_expanded_refit_tensors`` reshard, the same
    path the v2 worker's ``_iter_refit_state_dict`` drives. Records the names
    that reach the wire."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._state_dict = state_dict
        self.streamed_names: list[str] = []

    def broadcast_weights_for_collective(self, **kw):
        for name, _tensor in iter_expanded_refit_tensors(
            self._state_dict.items(), materialize=lambda t: t
        ):
            self.streamed_names.append(name)
        return ["train"]


def test_moe_reshard_streams_per_expert_names_through_collective():
    # One MoE layer (E=2) + an attention param; experts must reach the wire as
    # per-expert HF names, the dense param untouched.
    state_dict = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        "model.layers.0.mlp.experts.w1_EFD": torch.zeros(2, 4, 3),
        "model.layers.0.mlp.experts.w3_EFD": torch.zeros(2, 4, 3),
        "model.layers.0.mlp.experts.w2_EDF": torch.zeros(2, 3, 4),
    }
    policy = _MoERefitPolicy(state_dict)

    class _Gen:
        def update_weights_from_collective(self):
            return [True, True]

    ws = CollectiveWeightSynchronizer(
        policy, _Gen(), train_cluster=None, inference_world_size=8
    )
    ws.sync_weights()

    assert ws.is_stale is False
    assert set(policy.streamed_names) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    }
