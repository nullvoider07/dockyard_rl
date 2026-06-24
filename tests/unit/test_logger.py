"""flatten_dict expand_lists + MLflow scalar-only summarization (Wave 2).

`utils/logger.py` imports `torch.utils.tensorboard.SummaryWriter`, whose package
init eagerly imports the optional `tensorboard` dependency (absent on a plain
host, present in the ubuntu-swe image). These tests cover the pure metric
helpers and the MLflow path, none of which touch TensorBoard, so a minimal
`torch.utils.tensorboard` stub is injected before import.

Covers the bug where per-worker generation-timeline metrics
(`{metric: {dp_idx: [values]}}`) and any list-valued metric were index-expanded
by `flatten_dict` into one MLflow key per element. The fix retains lists
(`expand_lists=False`) and reduces each to a bounded `name/{mean,p50,p90,max}`.
"""

from __future__ import annotations

import sys
import types

if "torch.utils.tensorboard" not in sys.modules:
    _tb = types.ModuleType("torch.utils.tensorboard")
    _tb.SummaryWriter = type("SummaryWriter", (), {})  # type: ignore[attr-defined]
    sys.modules["torch.utils.tensorboard"] = _tb

import dockyard_rl.utils.logger as logmod
from dockyard_rl.utils.logger import (
    MLflowLogger,
    _merge_generation_logger_workers,
    _summarize_list,
    flatten_dict,
)


class TestFlattenDict:
    def test_expand_lists_default_index_expands(self):
        assert flatten_dict({"a": [1, 2], "b": {"c": [3, 4]}}) == {
            "a.0": 1,
            "a.1": 2,
            "b.c.0": 3,
            "b.c.1": 4,
        }

    def test_no_expand_retains_lists(self):
        assert flatten_dict(
            {"a": [1, 2], "b": {"c": 3}}, expand_lists=False
        ) == {"a": [1, 2], "b.c": 3}

    def test_no_expand_still_recurses_dicts(self):
        out = flatten_dict(
            {"x": {"y": {"z": 1}}, "lst": [{"k": 2}]}, sep="/", expand_lists=False
        )
        assert out == {"x/y/z": 1, "lst": [{"k": 2}]}


class TestSummarizeList:
    def test_basic_stats(self):
        out = _summarize_list([1.0, 2.0, 3.0, 4.0])
        assert out["mean"] == 2.5
        assert out["p50"] == 2.5
        assert out["max"] == 4.0
        assert abs(out["p90"] - 3.7) < 1e-9

    def test_drops_nonfinite_and_nonnumeric(self):
        out = _summarize_list([1.0, float("inf"), float("nan"), "x", True, 3.0])
        # only 1.0 and 3.0 survive (inf/nan dropped, str/bool dropped)
        assert out["mean"] == 2.0
        assert out["max"] == 3.0

    def test_empty_when_nothing_finite(self):
        assert _summarize_list([]) == {}
        assert _summarize_list([float("nan"), "a", None]) == {}


class TestMergeGenerationLoggerWorkers:
    def test_concatenates_in_dp_order(self):
        merged = _merge_generation_logger_workers(
            {"latency": {1: [3.0, 4.0], 0: [1.0, 2.0]}}
        )
        assert merged == {"latency": [1.0, 2.0, 3.0, 4.0]}

    def test_scalar_per_worker_values(self):
        merged = _merge_generation_logger_workers({"m": {0: 1.0, 1: 2.0}})
        assert merged == {"m": [1.0, 2.0]}

    def test_non_dict_inner_does_not_crash(self):
        # Defensive: a bare list or scalar inner value is tolerated, not indexed.
        assert _merge_generation_logger_workers({"a": [1.0, 2.0], "b": 3.0}) == {
            "a": [1.0, 2.0],
            "b": [3.0],
        }


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, metrics, step, run_id):  # mirrors mlflow.log_metrics kwargs
        self.calls.append({"metrics": metrics, "step": step, "run_id": run_id})


def _make_logger():
    logger = MLflowLogger.__new__(MLflowLogger)
    logger.run_id = "test-run"
    return logger


def _patch_mlflow(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(
        logmod.mlflow,
        "log_metrics",
        lambda metrics, step, run_id: rec(metrics, step, run_id),
    )
    return rec


class TestMLflowLogMetrics:
    def test_generation_metrics_collapse_to_bounded_stats(self, monkeypatch):
        rec = _patch_mlflow(monkeypatch)
        logger = _make_logger()
        logger.log_metrics(
            {
                "loss": 0.5,
                "generation_logger_metrics": {
                    "latency": {0: [1.0, 2.0], 1: [3.0, 4.0]}
                },
            },
            step=7,
            prefix="train",
        )
        assert len(rec.calls) == 1
        logged = rec.calls[0]["metrics"]
        assert rec.calls[0]["step"] == 7
        assert logged["train/loss"] == 0.5
        for stat in ("mean", "p50", "p90", "max"):
            assert f"train/generation_logger_metrics/latency/{stat}" in logged
        # no per-element keys leaked
        assert not any(k.endswith(".0") or k.endswith("/0") for k in logged)
        # bounded: 1 scalar + 4 stats
        assert len(logged) == 5

    def test_plain_list_metric_is_summarized(self, monkeypatch):
        rec = _patch_mlflow(monkeypatch)
        logger = _make_logger()
        logger.log_metrics({"rewards": [1.0, 2.0, 3.0]}, step=0)
        logged = rec.calls[0]["metrics"]
        assert set(logged) == {
            "rewards/mean",
            "rewards/p50",
            "rewards/p90",
            "rewards/max",
        }

    def test_non_numeric_scalar_skipped(self, monkeypatch):
        rec = _patch_mlflow(monkeypatch)
        logger = _make_logger()
        logger.log_metrics({"name": "abc", "x": 1.0}, step=0)
        logged = rec.calls[0]["metrics"]
        assert logged == {"x": 1.0}

    def test_no_call_when_nothing_loggable(self, monkeypatch):
        rec = _patch_mlflow(monkeypatch)
        logger = _make_logger()
        logger.log_metrics({"name": "abc", "empty": []}, step=0)
        assert rec.calls == []
