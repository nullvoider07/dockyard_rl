"""CPU tests for colocated-SGLang memory lifecycle and GPU-fraction reservation.

SGLang is colocated-only in dockyard (it shares GPUs with the trainer). With
``sglang_cfg.enable_memory_saver`` set, the server time-shares the GPU: it
releases its memory occupation during the training phase and resumes it before
generation. These tests cover the dispatch logic (prepare -> wake_up,
finish -> sleep, gated on the flag) and the colocation GPU reservation cap,
both of which are pure/CPU-checkable. Live memory reclaim is GPU-deferred.
"""

import pytest

from dockyard_rl.distributed.worker_groups import colocation_reserved_num_gpus


# -- GPU-fraction reservation -------------------------------------------------

def test_reservation_cpu_cluster_is_zero():
    assert colocation_reserved_num_gpus(use_gpus=False, max_colocated_worker_groups=1) == 0.0
    assert colocation_reserved_num_gpus(use_gpus=False, max_colocated_worker_groups=4) == 0.0


def test_reservation_dedicated_fleet_is_full_gpu():
    # max_colocated_worker_groups == 1 must reserve the full GPU; the 0.2 cap
    # must NOT apply here (that would under-reserve a dedicated worker).
    assert colocation_reserved_num_gpus(use_gpus=True, max_colocated_worker_groups=1) == 1.0


def test_reservation_colocated_is_capped_at_0_2():
    # 1/2 = 0.5 -> capped to 0.2; 1/4 = 0.25 -> capped to 0.2; 1/5 = 0.2 -> 0.2.
    assert colocation_reserved_num_gpus(use_gpus=True, max_colocated_worker_groups=2) == 0.2
    assert colocation_reserved_num_gpus(use_gpus=True, max_colocated_worker_groups=4) == 0.2
    assert colocation_reserved_num_gpus(use_gpus=True, max_colocated_worker_groups=5) == 0.2


def test_reservation_many_groups_below_cap_uses_share():
    # When the even share is already below the cap, use the share (1/10 = 0.1).
    assert colocation_reserved_num_gpus(use_gpus=True, max_colocated_worker_groups=10) == pytest.approx(0.1)


# -- Memory lifecycle dispatch ------------------------------------------------

def _make_generation(monkeypatch, *, memory_saver: bool, worker_result):
    """Bypass-construct an SGLangGeneration with a mock worker group.

    Patches the module's ``ray.get`` to an identity collect so a plain list
    flows through the ``all(r is not False ...)`` success check.
    """
    from unittest.mock import MagicMock

    from dockyard_rl.models.generation.sglang import sglang_generation as mod

    monkeypatch.setattr(mod.ray, "get", lambda x, **kw: x)

    gen = mod.SGLangGeneration.__new__(mod.SGLangGeneration)
    gen.sglang_cfg = {"enable_memory_saver": memory_saver}
    gen.worker_group = MagicMock()
    gen.worker_group.run_all_workers_single_data.return_value = worker_result
    return gen, mod


def test_prepare_resumes_with_tags_when_memory_saver_on(monkeypatch):
    gen, _ = _make_generation(monkeypatch, memory_saver=True, worker_result=[True])
    ok = gen.prepare_for_generation(tags=["weights"])
    assert ok is True
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "wake_up",
        run_rank_0_only_axes=["tensor_parallel"],
        tags=["weights"],
    )


def test_prepare_without_tags_passes_none(monkeypatch):
    gen, _ = _make_generation(monkeypatch, memory_saver=True, worker_result=[True])
    gen.prepare_for_generation()
    _, kwargs = gen.worker_group.run_all_workers_single_data.call_args
    assert kwargs["tags"] is None


def test_finish_releases_all_when_memory_saver_on(monkeypatch):
    gen, _ = _make_generation(monkeypatch, memory_saver=True, worker_result=[True])
    ok = gen.finish_generation()
    assert ok is True
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "sleep",
        run_rank_0_only_axes=["tensor_parallel"],
        tags=None,
    )


def test_lifecycle_is_noop_when_memory_saver_off(monkeypatch):
    gen, _ = _make_generation(monkeypatch, memory_saver=False, worker_result=[True])
    assert gen.prepare_for_generation(tags=["weights"]) is True
    assert gen.finish_generation() is True
    gen.worker_group.run_all_workers_single_data.assert_not_called()


def test_prepare_reports_failure_when_worker_returns_false(monkeypatch):
    gen, _ = _make_generation(monkeypatch, memory_saver=True, worker_result=[True, False])
    assert gen.prepare_for_generation() is False


def test_finish_reports_failure_on_exception(monkeypatch):
    from unittest.mock import MagicMock

    from dockyard_rl.models.generation.sglang import sglang_generation as mod

    def _raise(*_a, **_k):
        raise RuntimeError("release failed")

    monkeypatch.setattr(mod.ray, "get", _raise)
    gen = mod.SGLangGeneration.__new__(mod.SGLangGeneration)
    gen.sglang_cfg = {"enable_memory_saver": True}
    gen.worker_group = MagicMock()
    gen.worker_group.run_all_workers_single_data.return_value = [object()]
    assert gen.finish_generation() is False
