"""Memory accounting and schedule selection for colocated distillation.

Colocated distillation places the student and the frozen teacher on the same
GPU mesh. The student optimizer (the heavyweight, ~12 bytes/param) and the
teacher weights are required in anti-phase windows: the teacher is resident only
while it produces top-k logits; the student optimizer is resident only while the
student trains. This module computes the per-phase per-GPU peak from the model
shapes and selects the fastest schedule that fits, or refuses at startup with an
actionable shortfall rather than OOMing mid-training.

All arithmetic here is pure (no CUDA), so the safety property is unit-testable on
CPU. The static terms (weights, optimizer state, gradients) are exact given the
parameter count, dtype, and shard count; activations are an estimated reserve and
are deliberately separated so callers can supply a measured figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Bytes per parameter for common dtypes.
DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,
    "fp8": 1,
}

# AdamW master copy (fp32) + exp_avg (fp32) + exp_avg_sq (fp32). Conservative
# default; bf16-state optimizers without a master copy use less.
DEFAULT_OPTIMIZER_BYTES_PER_PARAM = 12

# Fixed per-GPU reserve for the CUDA context, allocator fragmentation, and
# transient buffers (NCCL, top-k scratch). Bytes.
DEFAULT_FIXED_OVERHEAD_BYTES = 2 * 1024**3


def dtype_bytes(dtype: str) -> int:
    """Bytes per element for a dtype name. Defaults to bf16 (2) if unknown."""
    return DTYPE_BYTES.get(str(dtype).lower().replace("torch.", ""), 2)


class ColocationSchedule(str, Enum):
    """Selected residency schedule for a colocated distillation step.

    FAST     teacher and student optimizer both stay resident; zero per-step
             host<->device transfer.
    TIGHT    anti-phase eviction: student optimizer offloaded while the teacher
             scores, teacher offloaded while the student trains.
    INFEASIBLE
             one phase exceeds free memory even fully time-shared; the run is
             refused at startup.
    """

    FAST = "fast"
    TIGHT = "tight"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class MemoryReport:
    """Per-GPU memory accounting (bytes) for a colocated distillation step."""

    student_weight_bytes: int
    student_optimizer_bytes: int
    student_grad_bytes: int
    student_activation_bytes: int
    teacher_weight_bytes: int
    teacher_activation_bytes: int
    fixed_overhead_bytes: int
    free_bytes: int

    @property
    def scoring_phase_bytes(self) -> int:
        """Phase 2: teacher resident + student weights; optimizer/grads offloaded."""
        return (
            self.student_weight_bytes
            + self.teacher_weight_bytes
            + self.teacher_activation_bytes
            + self.fixed_overhead_bytes
        )

    @property
    def training_phase_bytes(self) -> int:
        """Phase 3: full student training footprint; teacher offloaded."""
        return (
            self.student_weight_bytes
            + self.student_optimizer_bytes
            + self.student_grad_bytes
            + self.student_activation_bytes
            + self.fixed_overhead_bytes
        )

    @property
    def both_resident_bytes(self) -> int:
        """Fast path: full student footprint + resident teacher, no eviction."""
        return (
            self.student_weight_bytes
            + self.student_optimizer_bytes
            + self.student_grad_bytes
            + max(self.student_activation_bytes, self.teacher_activation_bytes)
            + self.teacher_weight_bytes
            + self.fixed_overhead_bytes
        )


def estimate_activation_bytes(
    micro_batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    elem_bytes: int,
    num_shards: int,
    activation_factor: float = 16.0,
) -> int:
    """Rough per-GPU activation estimate.

    Transformer activation memory scales with mbs * seq * hidden * layers. The
    ``activation_factor`` rolls up the per-layer activation multiplier (attention
    + MLP intermediates) and is deliberately generous so the estimate is an upper
    bound rather than an underestimate that would let a run start and then OOM.
    Sharded by ``num_shards`` (tensor/sequence parallel reduce the per-GPU slice).
    """
    if num_shards <= 0:
        num_shards = 1
    per_gpu = (
        activation_factor
        * micro_batch_size
        * seq_len
        * hidden_size
        * num_layers
        * elem_bytes
    ) / num_shards
    return int(per_gpu)


def build_memory_report(
    *,
    student_num_params: int,
    teacher_num_params: int,
    num_shards: int,
    free_bytes: int,
    student_weight_dtype: str = "bfloat16",
    teacher_weight_dtype: str = "bfloat16",
    optimizer_bytes_per_param: int = DEFAULT_OPTIMIZER_BYTES_PER_PARAM,
    grad_dtype: str = "float32",
    student_activation_bytes: Optional[int] = None,
    teacher_activation_bytes: Optional[int] = None,
    fixed_overhead_bytes: int = DEFAULT_FIXED_OVERHEAD_BYTES,
) -> MemoryReport:
    """Assemble a per-GPU :class:`MemoryReport` from model shapes.

    Static terms (weights, optimizer, grads) are exact given param count, dtype,
    and shard count. Activations default to 0 if not supplied — callers that have
    a measured or estimated activation figure should pass it so the fast/tight
    decision accounts for it.
    """
    if num_shards <= 0:
        num_shards = 1

    s_w_bytes = student_num_params * dtype_bytes(student_weight_dtype) // num_shards
    s_opt_bytes = student_num_params * optimizer_bytes_per_param // num_shards
    s_grad_bytes = student_num_params * dtype_bytes(grad_dtype) // num_shards
    t_w_bytes = teacher_num_params * dtype_bytes(teacher_weight_dtype) // num_shards

    return MemoryReport(
        student_weight_bytes=s_w_bytes,
        student_optimizer_bytes=s_opt_bytes,
        student_grad_bytes=s_grad_bytes,
        student_activation_bytes=student_activation_bytes or 0,
        teacher_weight_bytes=t_w_bytes,
        teacher_activation_bytes=teacher_activation_bytes or 0,
        fixed_overhead_bytes=fixed_overhead_bytes,
        free_bytes=free_bytes,
    )


def select_colocation_schedule(
    report: MemoryReport,
    requested: str = "auto",
    safety_margin: float = 0.05,
) -> tuple[ColocationSchedule, str]:
    """Choose the fastest residency schedule that fits within free memory.

    ``requested`` is one of ``"auto"`` (pick the fastest that fits), ``"fast"``
    (force the no-transfer path; error if it does not fit), or ``"tight"`` (force
    anti-phase eviction; error if a single phase still does not fit).

    ``safety_margin`` shrinks the usable budget (default 5%) to absorb estimate
    error and allocator fragmentation.

    Returns the selected schedule and a human-readable report line. An infeasible
    selection returns ``ColocationSchedule.INFEASIBLE`` with the limiting phase and
    shortfall; the caller is expected to raise.
    """
    budget = int(report.free_bytes * (1.0 - safety_margin))
    gib = 1024**3

    fast_fits = report.both_resident_bytes <= budget
    scoring_fits = report.scoring_phase_bytes <= budget
    training_fits = report.training_phase_bytes <= budget
    tight_fits = scoring_fits and training_fits

    def line(schedule: ColocationSchedule) -> str:
        return (
            f"colocation schedule={schedule.value} "
            f"free={report.free_bytes / gib:.1f}GiB budget={budget / gib:.1f}GiB "
            f"fast={report.both_resident_bytes / gib:.1f}GiB "
            f"scoring={report.scoring_phase_bytes / gib:.1f}GiB "
            f"training={report.training_phase_bytes / gib:.1f}GiB"
        )

    req = requested.lower()
    if req == "fast":
        if fast_fits:
            return ColocationSchedule.FAST, line(ColocationSchedule.FAST)
        shortfall = (report.both_resident_bytes - budget) / gib
        return ColocationSchedule.INFEASIBLE, (
            f"colocation schedule='fast' requested but the fast (no-transfer) path "
            f"needs {report.both_resident_bytes / gib:.1f}GiB, exceeding the "
            f"{budget / gib:.1f}GiB budget by {shortfall:.1f}GiB. Use schedule='auto' "
            f"or 'tight', add GPUs, or use separate-cluster distillation."
        )

    if req == "tight":
        if tight_fits:
            return ColocationSchedule.TIGHT, line(ColocationSchedule.TIGHT)
        return ColocationSchedule.INFEASIBLE, _infeasible_message(report, budget)

    # auto: fastest that fits.
    if fast_fits:
        return ColocationSchedule.FAST, line(ColocationSchedule.FAST)
    if tight_fits:
        return ColocationSchedule.TIGHT, line(ColocationSchedule.TIGHT)
    return ColocationSchedule.INFEASIBLE, _infeasible_message(report, budget)


def _infeasible_message(report: MemoryReport, budget: int) -> str:
    gib = 1024**3
    scoring_over = (report.scoring_phase_bytes - budget) / gib
    training_over = (report.training_phase_bytes - budget) / gib
    if report.training_phase_bytes >= report.scoring_phase_bytes:
        phase, over, need = "training", training_over, report.training_phase_bytes
    else:
        phase, over, need = "scoring", scoring_over, report.scoring_phase_bytes
    return (
        f"colocated distillation is infeasible on this mesh: the {phase} phase needs "
        f"{need / gib:.1f}GiB per GPU, exceeding the {budget / gib:.1f}GiB budget by "
        f"{over:.1f}GiB even with anti-phase eviction. Add GPUs (more sharding), "
        f"shrink the teacher (e.g. bf16/fp8 teacher dtype), reduce the micro-batch, "
        f"or use separate-cluster distillation (distillation.colocated=false)."
    )
