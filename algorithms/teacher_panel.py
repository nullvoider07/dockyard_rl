"""Teacher-panel bursting for on-policy distillation (Track F3c).

A V4-style consolidation scores every student rollout against ten or more
specialist teachers. dockyard routes statically instead: ``opd.py`` maps each
agent name to exactly one alias, so every row is scored once, by one teacher.
Static routing is cheap and it is also blind -- the routed teacher may be the
wrong specialist for a particular trajectory, and nothing notices.

The burst form keeps the cheap default and pays for the panel only where the
default looks out of its depth: score everything with the routed teacher, then
re-score a bounded fraction of rows with the panel and combine. A full panel over
a batch costs ``B x (1 + P)`` teacher-rows; this costs ``B x (1 + f * P)``, and
``f`` is the config.

**The escalation signal is the base teacher's own surprise.** Mean per-token
logprob over a row's valid tokens, lowest first: a trajectory the routed teacher
finds unlikely is one its competence does not cover, which is exactly the case
where a second opinion is worth buying. Nothing else is needed -- no student
logprobs, no reward, no extra forward -- because the base pass has already
produced it.

**Why the default combine is a mixture and not an average of logprobs.** Teacher
``t`` assigns probability ``p_t`` to the student's token. A uniform mixture over
the panel assigns ``(1/n) * sum(p_t)``, whose logprob is
``logsumexp(lp) - log(n)``. Averaging the *logprobs* instead is a geometric mean
-- a product of experts -- and it lets any single teacher veto a token by
assigning it low probability. That is precisely wrong for a panel of
specialists, where most members are irrelevant to a given trajectory and their
disagreement is expected rather than informative. ``combine: "geometric"`` is
offered for the case where the panel really is an ensemble of interchangeable
generalists, and it is not the default.

**Scope, stated because it is not the scope of the rest of Track F.** The teacher
path lives entirely in the async trajectory collector (``async_grpo_train``
builds the worker groups; nothing on the sync path scores a teacher), so this is
async-only where F2, F3a and F3b are sync-only. It composes with neither the
graded allocator nor the routing-diversity column -- those are sync-path
machinery -- and an enabled panel on a sync run is refused at the launcher rather
than left to read as live.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch

# How panel logprobs are folded together with the base teacher's.
COMBINE_MODES = frozenset({"mixture", "geometric"})

# Trainer entry points that score teachers at all.
SUPPORTED_TRAINERS = frozenset({"async"})


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    """Read one key from a dict-style OR attribute-style config.

    Both shapes reach this module and they are not interchangeable: the
    collector holds ``on_policy_distillation`` as a plain dict (``_opd_cfg``
    model_dumps it), while the launcher validates the live
    ``OnPolicyDistillationConfig`` pydantic model, which has no ``.get``. A
    dict-only reader returns the default for every key on the pydantic side --
    so the launcher refusal silently never fires, which is the failure this
    module exists to prevent.
    """
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def teacher_panel_cfg(opd_cfg: Any) -> dict[str, Any]:
    """The ``teacher_panel`` sub-config as a plain dict, or empty."""
    if not opd_cfg:
        return {}
    panel = _cfg(opd_cfg, "teacher_panel", None)
    if panel is None:
        return {}
    if hasattr(panel, "model_dump"):
        return panel.model_dump(exclude_none=True)
    return dict(panel) if hasattr(panel, "keys") else {}


def teacher_panel_enabled(opd_cfg: Any) -> bool:
    """True when a config asks for panel bursting."""
    panel = teacher_panel_cfg(opd_cfg)
    return bool(panel.get("enabled", False))


def validate_teacher_panel_config(
    opd_cfg: Any,
    *,
    available_aliases: Sequence[str],
    trainer: str,
    context: str = "on_policy_distillation.teacher_panel",
) -> None:
    """Refuse a panel that cannot burst before the run starts.

    Inert when disabled, so the default path is unchanged.

    Raises:
        ValueError: If the escalation fraction or combine mode is invalid, the
            panel names no teacher or names one that was never built, or the
            trainer scores no teachers at all.
    """
    if not teacher_panel_enabled(opd_cfg):
        return

    panel = teacher_panel_cfg(opd_cfg)

    if trainer not in SUPPORTED_TRAINERS:
        raise ValueError(
            f"{context}.enabled=true is not supported on the {trainer!r} "
            "trainer. On-policy distillation scores teachers only from the "
            "async trajectory collector; on any other path no teacher is "
            "called, so the panel would read as configured and never burst. "
            f"Supported: {sorted(SUPPORTED_TRAINERS)}."
        )

    fraction = _cfg(panel, "escalation_fraction", 0.0)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"{context}.escalation_fraction must be in [0, 1], got {fraction}. "
            "It is the share of rows re-scored by the panel, and the whole "
            "point of bursting is that it is bounded."
        )

    combine = _cfg(panel, "combine", "mixture")
    if combine not in COMBINE_MODES:
        raise ValueError(
            f"{context}.combine must be one of {sorted(COMBINE_MODES)}, got "
            f"{combine!r}."
        )

    aliases = list(_cfg(panel, "panel_aliases", []) or [])
    if not aliases:
        raise ValueError(
            f"{context}.enabled=true with no panel_aliases. There is nothing to "
            "escalate to, so every row would be scored exactly as it is today "
            "while the config reads as a panel."
        )

    unknown = [a for a in aliases if a not in set(available_aliases)]
    if unknown:
        raise ValueError(
            f"{context}.panel_aliases names {unknown}, which are not teachers "
            "this run built. Available: "
            f"{sorted(available_aliases)}. A panel alias with no worker group "
            "is a burst that silently never happens."
        )


def mean_valid_logprob(
    logprobs: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-row mean logprob over valid (unpadded) tokens.

    The escalation signal. Right-padding carries whatever the teacher returned
    for positions that do not exist, so a plain row mean would rank rows by how
    much padding they have -- which is length, not surprise.
    """
    if logprobs.dim() != 2:
        raise ValueError(
            f"expected [B, S] teacher logprobs, got {tuple(logprobs.shape)}"
        )
    rows, seq = logprobs.shape
    if input_lengths is None:
        return logprobs.mean(dim=1) if seq else torch.zeros(rows)

    lengths = input_lengths.reshape(-1).to(torch.long).clamp(min=0, max=seq)
    positions = torch.arange(seq).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    counts = mask.sum(dim=1).clamp(min=1)
    return (logprobs * mask).sum(dim=1) / counts


def select_escalation_rows(
    surprise_scores: torch.Tensor,
    escalation_fraction: float,
) -> list[int]:
    """Rows to re-score: the lowest-scoring `fraction` of them, ascending.

    Lowest mean logprob first -- the trajectories the routed teacher found least
    likely. Ties break on row index so the selection is deterministic, and the
    count is rounded down so the configured fraction is an upper bound on what
    the burst costs rather than a target it may exceed.
    """
    total = int(surprise_scores.reshape(-1).numel())
    if total == 0 or escalation_fraction <= 0.0:
        return []
    count = min(total, int(math.floor(total * escalation_fraction)))
    if count <= 0:
        return []
    scores = surprise_scores.reshape(-1).to(torch.float64).tolist()
    order = sorted(range(total), key=lambda i: (scores[i], i))
    return sorted(order[:count])


def combine_panel_logprobs(
    rows: Sequence[torch.Tensor],
    combine: str = "mixture",
) -> torch.Tensor:
    """Fold several teachers' per-token logprobs for the same rows into one.

    ``mixture`` is the logprob of the uniform mixture of the teachers'
    distributions, ``logsumexp(lp) - log(n)``. ``geometric`` is the plain mean of
    logprobs, a product of experts, in which one teacher assigning low
    probability drags the result down on its own.
    """
    if not rows:
        raise ValueError("combine_panel_logprobs needs at least one logprob tensor")
    if combine not in COMBINE_MODES:
        raise ValueError(
            f"combine must be one of {sorted(COMBINE_MODES)}, got {combine!r}"
        )
    shapes = {tuple(r.shape) for r in rows}
    if len(shapes) > 1:
        raise ValueError(
            f"panel logprobs must all share a shape, got {sorted(shapes)}"
        )
    if len(rows) == 1:
        return rows[0]

    stacked = torch.stack(list(rows), dim=0)
    if combine == "geometric":
        return stacked.mean(dim=0)
    return torch.logsumexp(stacked, dim=0) - math.log(stacked.shape[0])


def teacher_panel_metrics(
    escalated: list[int],
    batch_size: int,
    panel_size: int,
    surprise_scores: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Report what the burst cost and what it was spent on.

    ``teacher_rows_saved_ratio`` is against a full panel over every row, which is
    the thing bursting exists to avoid -- not against the single-teacher baseline,
    which bursting is strictly more expensive than.
    """
    escalated_rows = len(escalated)
    full_panel_rows = batch_size * (1 + panel_size)
    burst_rows = batch_size + escalated_rows * panel_size
    metrics = {
        "teacher_panel_escalated_rows": float(escalated_rows),
        "teacher_panel_escalated_ratio": float(
            escalated_rows / batch_size if batch_size else 0.0
        ),
        "teacher_panel_rows_scored": float(burst_rows),
        "teacher_panel_rows_saved_ratio": float(
            1.0 - burst_rows / full_panel_rows if full_panel_rows else 0.0
        ),
    }
    if surprise_scores is not None and surprise_scores.numel() and escalated_rows:
        flat = surprise_scores.reshape(-1)
        metrics["teacher_panel_escalated_surprise_mean"] = float(
            flat[escalated].mean().item()
        )
        metrics["teacher_panel_batch_surprise_mean"] = float(flat.mean().item())
    return metrics
