"""Unit tests for DPO evaluation gates (D7): calibration + judge pass-rate."""

import pytest

from dockyard_rl.algorithms.dpo_evals import (
    accuracy,
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
    judge_pass_rate,
)


class TestECE:
    def test_perfect_calibration_zero_ece(self):
        # Confidence 1.0 + always correct, and 0.0 + always wrong → ECE 0.
        confs = [1.0, 1.0, 0.0, 0.0]
        corr = [1, 1, 0, 0]
        ece, _ = expected_calibration_error(confs, corr, n_bins=10)
        assert ece == pytest.approx(0.0)

    def test_full_miscalibration(self):
        # Confidence 1.0 but always wrong → gap 1.0 in the top bin → ECE 1.0.
        ece, _ = expected_calibration_error([1.0, 1.0], [0, 0], n_bins=10)
        assert ece == pytest.approx(1.0)

    def test_hand_computed(self):
        # Two samples at conf 0.8: one correct, one wrong → acc 0.5, conf 0.8,
        # gap 0.3, single populated bin weight 1.0 → ECE 0.3.
        ece, details = expected_calibration_error([0.8, 0.8], [1, 0], n_bins=10)
        assert ece == pytest.approx(0.3)
        populated = [d for d in details if d["count"] > 0]
        assert len(populated) == 1
        assert populated[0]["accuracy"] == pytest.approx(0.5)
        assert populated[0]["confidence"] == pytest.approx(0.8)

    def test_clamps_out_of_range(self):
        ece, _ = expected_calibration_error([1.5, -0.5], [1, 0], n_bins=10)
        assert ece == pytest.approx(0.0)  # clamped to 1.0/0.0, both correct-aligned

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [1, 0])

    def test_empty(self):
        ece, details = expected_calibration_error([], [], n_bins=5)
        assert ece == 0.0 and len(details) == 5

    def test_bad_nbins(self):
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [1], n_bins=0)


class TestBrierAccuracy:
    def test_brier_perfect(self):
        assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)

    def test_brier_worst(self):
        assert brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)

    def test_accuracy(self):
        assert accuracy([1, 0, 1, 1]) == pytest.approx(0.75)

    def test_accuracy_empty(self):
        assert accuracy([]) == 0.0

    def test_compute_calibration_metrics_bundle(self):
        m = compute_calibration_metrics([0.9, 0.1], [1, 0], n_bins=10)
        assert set(m) == {"ece", "brier", "accuracy", "num_samples"}
        assert m["accuracy"] == pytest.approx(0.5)
        assert m["num_samples"] == 2.0


class _StubJudge:
    """Scores items by a preset map keyed on response text."""

    def __init__(self, scores):
        self._scores = scores

    def score(self, prompt, response, *, trajectory=None):
        return self._scores.get(response)


class TestJudgePassRate:
    def test_pass_rate_and_mean(self):
        judge = _StubJudge({"good": 0.9, "ok": 0.6, "bad": 0.2})
        items = [
            {"prompt": "p", "response": "good"},
            {"prompt": "p", "response": "ok"},
            {"prompt": "p", "response": "bad"},
        ]
        out = judge_pass_rate(judge, items, threshold=0.5)
        assert out["pass_rate"] == pytest.approx(2 / 3)
        assert out["mean_score"] == pytest.approx((0.9 + 0.6 + 0.2) / 3)
        assert out["num_scored"] == 3.0

    def test_unscored_counted_not_in_rate(self):
        judge = _StubJudge({"a": 0.8})  # "b" -> None
        items = [{"prompt": "p", "response": "a"}, {"prompt": "p", "response": "b"}]
        out = judge_pass_rate(judge, items, threshold=0.5)
        assert out["num_unscored"] == 1.0
        assert out["num_scored"] == 1.0
        assert out["pass_rate"] == 1.0  # only the scored "a" counts

    def test_all_unscored(self):
        judge = _StubJudge({})
        out = judge_pass_rate(judge, [{"prompt": "p", "response": "x"}])
        assert out["pass_rate"] == 0.0 and out["num_scored"] == 0.0
