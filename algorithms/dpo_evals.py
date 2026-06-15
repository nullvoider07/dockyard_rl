"""DPO evaluation gates (D7).

Metrics for the three DPO objectives, computed on fixtures (no cluster):

  - Calibration: Expected Calibration Error (ECE), Brier score, accuracy — for
    truth-seeking / "knows what it doesn't know".
  - Safety / truthfulness pass-rate: run a judge (the D5 ``PreferenceJudge`` or any
    object with a ``.score`` method) over an eval set and report the fraction at or
    above a pass threshold.

These are pure functions intended to be emitted alongside the existing DPO
validation metrics (loss / accuracy / reward means). Wiring them into
``dpo.validate`` as a gate is an additive integration; the computation here is the
validated unit.
"""

from __future__ import annotations

from typing import Any, Sequence


def _as_correct(correctness: Sequence[Any]) -> list[float]:
    return [1.0 if bool(c) else 0.0 for c in correctness]


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[Any],
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """Expected Calibration Error with equal-width confidence bins.

    ECE = Σ_b (|b| / N) · |acc(b) − conf(b)|, over bins of predicted confidence.
    Confidences are clamped to [0, 1]. Returns ``(ece, per_bin_details)``.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    confs = [min(1.0, max(0.0, float(c))) for c in confidences]
    corr = _as_correct(correctness)
    if len(confs) != len(corr):
        raise ValueError("confidences and correctness must have equal length")
    n = len(confs)
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for c, y in zip(confs, corr):
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx].append((c, y))

    ece = 0.0
    details: list[dict[str, float]] = []
    for b, items in enumerate(bins):
        if not items:
            details.append(
                {"bin": float(b), "count": 0.0, "confidence": 0.0, "accuracy": 0.0, "gap": 0.0}
            )
            continue
        avg_conf = sum(c for c, _ in items) / len(items)
        avg_acc = sum(y for _, y in items) / len(items)
        gap = abs(avg_acc - avg_conf)
        ece += (len(items) / n) * gap
        details.append(
            {
                "bin": float(b),
                "count": float(len(items)),
                "confidence": avg_conf,
                "accuracy": avg_acc,
                "gap": gap,
            }
        )
    return ece, details


def brier_score(confidences: Sequence[float], correctness: Sequence[Any]) -> float:
    """Mean squared error between confidence and outcome (lower is better)."""
    confs = [min(1.0, max(0.0, float(c))) for c in confidences]
    corr = _as_correct(correctness)
    if len(confs) != len(corr):
        raise ValueError("confidences and correctness must have equal length")
    if not confs:
        return 0.0
    return sum((c - y) ** 2 for c, y in zip(confs, corr)) / len(confs)


def accuracy(correctness: Sequence[Any]) -> float:
    corr = _as_correct(correctness)
    return sum(corr) / len(corr) if corr else 0.0


def compute_calibration_metrics(
    confidences: Sequence[float],
    correctness: Sequence[Any],
    n_bins: int = 10,
) -> dict[str, float]:
    """ECE + Brier + accuracy in one dict, for DPO validation emission."""
    ece, _ = expected_calibration_error(confidences, correctness, n_bins)
    return {
        "ece": ece,
        "brier": brier_score(confidences, correctness),
        "accuracy": accuracy(correctness),
        "num_samples": float(len(correctness)),
    }


def judge_pass_rate(
    judge: Any,
    items: Sequence[dict],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Run a ``.score(prompt, response, trajectory=...)`` judge over an eval set.

    Each item carries ``prompt`` / ``response`` and optional ``trajectory``. Returns
    pass-rate (fraction with score ≥ threshold), mean score, and scored/unscored
    counts. A safety or truthfulness gate is this harness with the corresponding
    judge template.
    """
    scores: list[float] = []
    n_unscored = 0
    for it in items:
        s = judge.score(
            it.get("prompt", ""), it.get("response", ""), trajectory=it.get("trajectory")
        )
        if s is None:
            n_unscored += 1
            continue
        scores.append(s)
    if not scores:
        return {"pass_rate": 0.0, "mean_score": 0.0, "num_scored": 0.0, "num_unscored": float(n_unscored)}
    n_pass = sum(1 for s in scores if s >= threshold)
    return {
        "pass_rate": n_pass / len(scores),
        "mean_score": sum(scores) / len(scores),
        "num_scored": float(len(scores)),
        "num_unscored": float(n_unscored),
    }
