"""#3748: histograms were produced on every step and reached no backend.

Two independent wiring failures, both of which present as a clean run with no
histograms rather than as an error:

1. Three of the four logger backends implemented log_histogram as a bare
   `return` -- TensorBoard, swanlab and MLflow. Only WandB logged anything.
2. Routing in the GRPO loop was gated on vllm_cfg.async_engine, while the
   producers (experience/rollouts.py:1431, experience/cua/rollout.py:440) are
   unconditional. A sync-engine run therefore emitted every histogram/* key and
   routed none; they fell through to log_metrics, where _coerce_to_scalar
   rejects a list and the key is skipped with a warning.

The backends are exercised against fakes rather than live writers: the claim
under test is that the value reaches the backend call, which is exactly the
wire that was cut.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dockyard_rl.algorithms.grpo import _route_histogram_metrics
from dockyard_rl.utils.logger import _histogram_summary


class _RecordingRun:
    """Stands in for a swanlab / wandb run object."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, Any]] = []

    def log(self, payload: dict, step: int | None = None) -> None:
        self.calls.append((payload, step))


class _RecordingLogger:
    def __init__(self) -> None:
        self.histograms: list[tuple[Any, int, str]] = []
        self.metrics: list[tuple[dict, int]] = []

    def log_histogram(self, histogram, step: int, name: str) -> None:
        self.histograms.append((histogram, step, name))

    def log_metrics(self, metrics: dict, step: int, prefix: str = "") -> None:
        self.metrics.append((metrics, step))


class TestRoutingIsNotGatedOnTheEngine:
    """The gate was the defect; these fail if it comes back."""

    def test_histograms_route_with_no_engine_config_at_all(self):
        logger = _RecordingLogger()
        metrics = {
            "loss": 1.0,
            "histogram/gen_tokens_length": [3, 4, 5],
            "histogram/input_tokens_length": [7, 8],
        }
        _route_histogram_metrics(logger, metrics, 11)

        assert len(logger.histograms) == 2
        by_name = {name: values for values, _, name in logger.histograms}
        assert by_name["generation_metrics/histogram/gen_tokens_length"] == [3, 4, 5]
        assert by_name["generation_metrics/histogram/input_tokens_length"] == [7, 8]
        assert all(step == 11 for _, step, _ in logger.histograms)

    def test_routed_keys_are_removed_from_the_scalar_dict(self):
        """They are lists; leaving them in is what produced the skip warning."""
        logger = _RecordingLogger()
        metrics = {"loss": 1.0, "histogram/gen_tokens_length": [3, 4, 5]}
        _route_histogram_metrics(logger, metrics, 1)

        assert "histogram/gen_tokens_length" not in metrics
        assert metrics == {"loss": 1.0}

    def test_non_histogram_metrics_are_untouched(self):
        logger = _RecordingLogger()
        metrics = {"loss": 1.0, "kl_penalty": 0.5, "not_a_histogram/x": [1, 2]}
        _route_histogram_metrics(logger, metrics, 1)

        assert logger.histograms == []
        assert metrics == {"loss": 1.0, "kl_penalty": 0.5, "not_a_histogram/x": [1, 2]}

    def test_empty_metrics_dict_is_a_no_op(self):
        logger = _RecordingLogger()
        metrics: dict = {}
        _route_histogram_metrics(logger, metrics, 1)
        assert logger.histograms == []


class TestTensorboardBackend:
    """TensorboardLogger.log_histogram was `return`; SummaryWriter has add_histogram."""

    def _logger(self, monkeypatch):
        from dockyard_rl.utils import logger as logger_mod

        recorded: dict[str, list] = {"hist": [], "scalar": []}

        class _FakeWriter:
            def add_histogram(self, name, values, step):
                recorded["hist"].append((name, np.asarray(values), step))

            def add_scalar(self, name, value, step):
                recorded["scalar"].append((name, value, step))

        monkeypatch.setattr(logger_mod, "SummaryWriter", lambda log_dir=None: _FakeWriter())
        # Typed Any: these fakes stand in for SummaryWriter / run objects the
        # real classes build in __init__, which needs a live backend.
        tb: Any = logger_mod.TensorboardLogger.__new__(logger_mod.TensorboardLogger)
        tb.writer = _FakeWriter()
        return tb, recorded

    def test_values_reach_add_histogram(self, monkeypatch):
        tb, recorded = self._logger(monkeypatch)
        tb.log_histogram([1, 2, 3, 4], 7, "train/lengths")

        assert len(recorded["hist"]) == 1
        name, values, step = recorded["hist"][0]
        assert name == "train/lengths"
        assert step == 7
        np.testing.assert_array_equal(values, np.asarray([1, 2, 3, 4]))

    def test_numpy_input_is_accepted(self, monkeypatch):
        """grpo.py:1097 passes tensor.numpy(), not a list."""
        tb, recorded = self._logger(monkeypatch)
        tb.log_histogram(np.asarray([0.5, 1.5]), 2, "train/advantages/histogram")
        assert len(recorded["hist"]) == 1

    def test_empty_input_logs_nothing_and_does_not_raise(self, monkeypatch):
        tb, recorded = self._logger(monkeypatch)
        tb.log_histogram([], 1, "train/empty")
        assert recorded["hist"] == []
        assert recorded["scalar"] == []

    def test_a_writer_failure_does_not_propagate(self, monkeypatch):
        """A logging failure must not abort a training step."""
        from dockyard_rl.utils import logger as logger_mod

        class _ExplodingWriter:
            def add_histogram(self, name, values, step):
                raise ValueError("cannot bin")

            def add_scalar(self, name, value, step):
                self.scalar = (name, value, step)

        tb: Any = logger_mod.TensorboardLogger.__new__(logger_mod.TensorboardLogger)
        tb.writer = _ExplodingWriter()
        tb.log_histogram([2.0, 2.0, 2.0], 3, "train/degenerate")
        assert tb.writer.scalar == ("train/degenerate", 2.0, 3)


class TestSummaryFallbackBackends:
    """swanlab and MLflow have no histogram type; they log summary statistics."""

    def test_swanlab_logs_summary_statistics(self, monkeypatch):
        from dockyard_rl.utils import logger as logger_mod

        sw: Any = logger_mod.SwanlabLogger.__new__(logger_mod.SwanlabLogger)
        sw.run = _RecordingRun()
        sw.log_histogram([1.0, 2.0, 3.0, 4.0], 5, "train/lengths")

        assert len(sw.run.calls) == 1
        payload, step = sw.run.calls[0]
        assert step == 5
        assert payload["train/lengths/mean"] == pytest.approx(2.5)
        assert payload["train/lengths/min"] == pytest.approx(1.0)
        assert payload["train/lengths/max"] == pytest.approx(4.0)
        assert payload["train/lengths/count"] == pytest.approx(4.0)

    def test_swanlab_empty_input_logs_nothing(self):
        from dockyard_rl.utils import logger as logger_mod

        sw: Any = logger_mod.SwanlabLogger.__new__(logger_mod.SwanlabLogger)
        sw.run = _RecordingRun()
        sw.log_histogram([], 1, "train/empty")
        assert sw.run.calls == []

    def test_mlflow_logs_one_scalar_per_statistic(self, monkeypatch):
        from dockyard_rl.utils import logger as logger_mod

        recorded: list[tuple[str, float, int | None]] = []
        monkeypatch.setattr(
            logger_mod.mlflow,
            "log_metric",
            lambda key, value, step=None: recorded.append((key, value, step)),
        )
        mf: Any = logger_mod.MLflowLogger.__new__(logger_mod.MLflowLogger)
        mf.log_histogram([1.0, 3.0], 9, "train/lengths")

        keys = {k for k, _, _ in recorded}
        assert "train/lengths/mean" in keys
        assert "train/lengths/count" in keys
        assert all(step == 9 for _, _, step in recorded)


class TestHistogramSummary:
    def test_statistics_are_correct(self):
        stats = _histogram_summary([1.0, 2.0, 3.0, 4.0])
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(4.0)
        assert stats["p50"] == pytest.approx(2.5)
        assert stats["count"] == pytest.approx(4.0)

    def test_non_finite_values_are_dropped(self):
        stats = _histogram_summary([1.0, float("inf"), float("nan"), 3.0])
        assert stats["count"] == pytest.approx(2.0)
        assert stats["mean"] == pytest.approx(2.0)

    @pytest.mark.parametrize("bad", [[], [None, None], ["a", "b"]])
    def test_unusable_input_returns_empty_rather_than_raising(self, bad):
        assert _histogram_summary(bad) == {}
