"""CPU tests for colocated-distillation memory accounting and schedule selection.

These cover the safety property that makes colocated distillation memory-safe
without a GPU: the per-phase peak is computed from model shapes, the fastest
schedule that fits is selected, and an oversized configuration is refused at
startup (INFEASIBLE) rather than OOMing mid-training.
"""

import pytest

from dockyard_rl.algorithms.distillation_memory import (
    DEFAULT_OPTIMIZER_BYTES_PER_PARAM,
    ColocationSchedule,
    MemoryReport,
    build_memory_report,
    dtype_bytes,
    estimate_activation_bytes,
    select_colocation_schedule,
)

GIB = 1024**3


def test_dtype_bytes_known_and_default():
    assert dtype_bytes("bfloat16") == 2
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("float32") == 4
    assert dtype_bytes("fp8") == 1
    assert dtype_bytes("torch.bfloat16") == 2
    # Unknown dtype falls back to bf16 (2).
    assert dtype_bytes("nonsense") == 2


def test_static_terms_are_exact_and_sharded():
    # 1B params, bf16 weights, fp32 grads, 4-way shard.
    report = build_memory_report(
        student_num_params=1_000_000_000,
        teacher_num_params=1_000_000_000,
        num_shards=4,
        free_bytes=80 * GIB,
    )
    assert report.student_weight_bytes == 1_000_000_000 * 2 // 4
    assert report.student_optimizer_bytes == (
        1_000_000_000 * DEFAULT_OPTIMIZER_BYTES_PER_PARAM // 4
    )
    assert report.student_grad_bytes == 1_000_000_000 * 4 // 4
    assert report.teacher_weight_bytes == 1_000_000_000 * 2 // 4


def test_phase_peaks_anti_phase_exclude_each_other():
    report = build_memory_report(
        student_num_params=1_000_000_000,
        teacher_num_params=1_000_000_000,
        num_shards=1,
        free_bytes=200 * GIB,
        student_activation_bytes=GIB,
        teacher_activation_bytes=GIB,
        fixed_overhead_bytes=GIB,
    )
    # Scoring phase excludes the student optimizer and grads.
    assert report.scoring_phase_bytes == (
        report.student_weight_bytes
        + report.teacher_weight_bytes
        + report.teacher_activation_bytes
        + report.fixed_overhead_bytes
    )
    assert report.student_optimizer_bytes not in (0,)  # sanity: optimizer is heavy
    # Training phase excludes the teacher entirely.
    assert report.training_phase_bytes == (
        report.student_weight_bytes
        + report.student_optimizer_bytes
        + report.student_grad_bytes
        + report.student_activation_bytes
        + report.fixed_overhead_bytes
    )
    # The anti-phase peak is strictly below the naive sum of both footprints.
    naive_sum = report.training_phase_bytes + report.teacher_weight_bytes
    assert max(report.scoring_phase_bytes, report.training_phase_bytes) < naive_sum


def test_auto_picks_fast_when_everything_fits():
    report = build_memory_report(
        student_num_params=1_000_000_000,
        teacher_num_params=1_000_000_000,
        num_shards=8,
        free_bytes=80 * GIB,
    )
    schedule, msg = select_colocation_schedule(report, requested="auto")
    assert schedule is ColocationSchedule.FAST
    assert "fast" in msg


def test_auto_falls_back_to_tight_when_fast_does_not_fit():
    # Size so that both-resident exceeds budget but each phase fits alone.
    report = build_memory_report(
        student_num_params=10_000_000_000,
        teacher_num_params=10_000_000_000,
        num_shards=8,
        free_bytes=24 * GIB,
        fixed_overhead_bytes=GIB,
    )
    assert report.both_resident_bytes > 24 * GIB * 0.95
    assert report.scoring_phase_bytes <= 24 * GIB * 0.95
    assert report.training_phase_bytes <= 24 * GIB * 0.95
    schedule, msg = select_colocation_schedule(report, requested="auto")
    assert schedule is ColocationSchedule.TIGHT
    assert "tight" in msg


def test_auto_refuses_when_a_phase_cannot_fit():
    report = build_memory_report(
        student_num_params=70_000_000_000,
        teacher_num_params=70_000_000_000,
        num_shards=2,
        free_bytes=24 * GIB,
    )
    schedule, msg = select_colocation_schedule(report, requested="auto")
    assert schedule is ColocationSchedule.INFEASIBLE
    assert "infeasible" in msg.lower()
    # The message names the limiting phase and a remedy.
    assert "training" in msg or "scoring" in msg
    assert "separate-cluster" in msg


def test_forced_fast_errors_when_it_does_not_fit():
    report = build_memory_report(
        student_num_params=10_000_000_000,
        teacher_num_params=10_000_000_000,
        num_shards=8,
        free_bytes=24 * GIB,
        fixed_overhead_bytes=GIB,
    )
    schedule, msg = select_colocation_schedule(report, requested="fast")
    assert schedule is ColocationSchedule.INFEASIBLE
    assert "fast" in msg.lower()


def test_forced_tight_succeeds_when_phases_fit():
    report = build_memory_report(
        student_num_params=10_000_000_000,
        teacher_num_params=10_000_000_000,
        num_shards=8,
        free_bytes=40 * GIB,
        fixed_overhead_bytes=GIB,
    )
    schedule, _ = select_colocation_schedule(report, requested="tight")
    assert schedule is ColocationSchedule.TIGHT


def test_safety_margin_shrinks_budget():
    # Construct a report that fits at 0% margin but not at a large margin.
    report = build_memory_report(
        student_num_params=1_000_000_000,
        teacher_num_params=1_000_000_000,
        num_shards=1,
        free_bytes=int(15.5 * GIB),
        fixed_overhead_bytes=GIB,
    )
    fast_bytes = report.both_resident_bytes
    # Pick free just above fast_bytes so a 5% margin pushes it under.
    tuned = MemoryReport(
        student_weight_bytes=report.student_weight_bytes,
        student_optimizer_bytes=report.student_optimizer_bytes,
        student_grad_bytes=report.student_grad_bytes,
        student_activation_bytes=report.student_activation_bytes,
        teacher_weight_bytes=report.teacher_weight_bytes,
        teacher_activation_bytes=report.teacher_activation_bytes,
        fixed_overhead_bytes=report.fixed_overhead_bytes,
        free_bytes=int(fast_bytes * 1.02),
    )
    schedule_zero, _ = select_colocation_schedule(tuned, requested="fast", safety_margin=0.0)
    schedule_marg, _ = select_colocation_schedule(tuned, requested="fast", safety_margin=0.10)
    assert schedule_zero is ColocationSchedule.FAST
    assert schedule_marg is ColocationSchedule.INFEASIBLE


def _make_master_config(distillation: dict):
    from types import SimpleNamespace

    return SimpleNamespace(distillation=distillation)


def _import_orchestration():
    """Import the residency-orchestration helpers.

    ``utils.logger`` pulls ``torch.utils.tensorboard`` (and thus ``tensorboard``),
    which is not part of the CPU test image. Stub it so the pure orchestration
    logic can be exercised here; the real image has tensorboard.
    """
    import sys
    import types

    if "torch.utils.tensorboard" not in sys.modules:
        stub = types.ModuleType("torch.utils.tensorboard")

        class _SummaryWriter:  # minimal stand-in
            def __init__(self, *_args, **_kwargs):
                pass

        stub.SummaryWriter = _SummaryWriter  # type: ignore[attr-defined]
        sys.modules["torch.utils.tensorboard"] = stub
    return pytest.importorskip("dockyard_rl.algorithms.distillation")


def test_get_schedule_none_when_not_colocated():
    d = _import_orchestration()
    cfg = _make_master_config({"colocated": False})
    assert d._get_colocation_schedule(cfg) is None


def test_get_schedule_resolved_roundtrip():
    d = _import_orchestration()
    cfg = _make_master_config(
        {"colocated": True, "_resolved_colocation_schedule": "tight"}
    )
    assert d._get_colocation_schedule(cfg) is ColocationSchedule.TIGHT


def test_get_schedule_none_when_unresolved():
    d = _import_orchestration()
    cfg = _make_master_config({"colocated": True})
    assert d._get_colocation_schedule(cfg) is None


def test_tight_scoring_phase_offloads_student_loads_teacher():
    d = _import_orchestration()
    from unittest.mock import MagicMock

    student, teacher = MagicMock(), MagicMock()
    d._enter_scoring_phase(student, teacher, ColocationSchedule.TIGHT)
    student.offload_before_refit.assert_called_once()
    teacher.prepare_for_lp_inference.assert_called_once()
    # The teacher is not evicted during scoring.
    teacher.offload_after_refit.assert_not_called()


def test_tight_training_phase_evicts_teacher_restores_student():
    d = _import_orchestration()
    from unittest.mock import MagicMock

    student, teacher = MagicMock(), MagicMock()
    d._enter_training_phase(student, teacher, ColocationSchedule.TIGHT)
    teacher.offload_after_refit.assert_called_once()
    student.prepare_for_training.assert_called_once()


def test_fast_phases_are_per_step_noops():
    d = _import_orchestration()
    from unittest.mock import MagicMock

    student, teacher = MagicMock(), MagicMock()
    d._enter_scoring_phase(student, teacher, ColocationSchedule.FAST)
    d._enter_training_phase(student, teacher, ColocationSchedule.FAST)
    # FAST keeps both resident and the student train-ready from the loop's
    # initial prepare; no per-step transfers or prepares.
    student.offload_before_refit.assert_not_called()
    teacher.prepare_for_lp_inference.assert_not_called()
    teacher.offload_after_refit.assert_not_called()
    student.prepare_for_training.assert_not_called()


def test_none_schedule_separate_cluster_is_fully_noop():
    d = _import_orchestration()
    from unittest.mock import MagicMock

    student, teacher = MagicMock(), MagicMock()
    d._enter_scoring_phase(student, teacher, None)
    d._enter_training_phase(student, teacher, None)
    # Separate-cluster default: the per-step behavior is byte-unchanged, so the
    # helpers must touch nothing (the loop's own prepare_for_training stands).
    student.offload_before_refit.assert_not_called()
    teacher.prepare_for_lp_inference.assert_not_called()
    teacher.offload_after_refit.assert_not_called()
    student.prepare_for_training.assert_not_called()


def test_resting_state_tight_offloads_teacher_fast_loads_it():
    d = _import_orchestration()
    from unittest.mock import MagicMock

    s_tight, t_tight = MagicMock(), MagicMock()
    d._set_colocation_resting_state(s_tight, t_tight, ColocationSchedule.TIGHT)
    t_tight.offload_after_refit.assert_called_once()
    t_tight.prepare_for_lp_inference.assert_not_called()

    s_fast, t_fast = MagicMock(), MagicMock()
    d._set_colocation_resting_state(s_fast, t_fast, ColocationSchedule.FAST)
    t_fast.prepare_for_lp_inference.assert_called_once()
    t_fast.offload_after_refit.assert_not_called()


def test_estimate_activation_bytes_scales_and_shards():
    base = estimate_activation_bytes(
        micro_batch_size=2,
        seq_len=4096,
        hidden_size=4096,
        num_layers=32,
        elem_bytes=2,
        num_shards=1,
    )
    doubled_mbs = estimate_activation_bytes(
        micro_batch_size=4,
        seq_len=4096,
        hidden_size=4096,
        num_layers=32,
        elem_bytes=2,
        num_shards=1,
    )
    sharded = estimate_activation_bytes(
        micro_batch_size=2,
        seq_len=4096,
        hidden_size=4096,
        num_layers=32,
        elem_bytes=2,
        num_shards=4,
    )
    assert doubled_mbs == pytest.approx(2 * base, rel=1e-6)
    assert sharded == pytest.approx(base / 4, rel=1e-6)
