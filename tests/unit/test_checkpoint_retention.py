"""#3071: checkpoint selection and retention ranked checkpoints differently.

remove_old_checkpoints sorted on the tuple (metric, step) -- the recency
tiebreak its docstring promises -- while get_best_checkpoint_path sorted on the
metric alone. Python's sort is stable, so ties in the selection path resolved
by the order _load_checkpoint_history happened to return, and that order is
glob.glob order: arbitrary, filesystem-dependent, and unrelated to step.

Consequence: when two checkpoints tie on the metric, retention deterministically
keeps the higher step while selection may return either. get_best_checkpoint_path
could therefore name a directory retention had already deleted, and could answer
differently on a different filesystem.

Both now go through _sort_checkpoints_best_first, so the two cannot drift apart
again. The agreement tests below are the ones that matter: they compare what
selection returns against what retention actually left on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dockyard_rl.utils.checkpoint import CheckpointManager


def _make_checkpoints(root: Path, steps_and_metrics: dict[int, float], metric: str) -> None:
    """Write step_N/training_info.json for each entry."""
    for step, value in steps_and_metrics.items():
        d = root / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "training_info.json").write_text(json.dumps({metric: value, "step": step}))


def _manager(root: Path, metric: str | None, higher_is_better: bool, keep_top_k: int | None):
    cfg: dict = {
        "checkpoint_dir": str(root),
        "metric_name": metric,
        "higher_is_better": higher_is_better,
        "keep_top_k": keep_top_k,
    }
    return CheckpointManager(cfg)  # pyright: ignore[reportArgumentType]


def _surviving_steps(root: Path) -> set[int]:
    return {int(p.name.split("_")[1]) for p in root.glob("step_*")}


@pytest.fixture(params=["ascending", "descending"])
def glob_order(request, monkeypatch):
    """Force _load_checkpoint_history's directory order to a known permutation.

    Its order comes from glob.glob -- arbitrary, and on this filesystem it
    happened to favour the right answer, so tie tests written against the real
    order passed even with the defect present. Pinning both permutations makes
    them actual detectors: a stable single-key sort returns the wrong
    checkpoint under exactly one of the two.
    """
    from dockyard_rl.utils import checkpoint as checkpoint_mod

    real_glob = checkpoint_mod.glob.glob
    reverse = request.param == "descending"

    def ordered_glob(pattern, *args, **kwargs):
        return sorted(real_glob(pattern, *args, **kwargs), reverse=reverse)

    monkeypatch.setattr(checkpoint_mod.glob, "glob", ordered_glob)
    return request.param


class TestTiesBreakTowardRecency:
    """The tiebreak retention documents must also govern selection."""

    def test_higher_is_better_prefers_the_later_step_on_a_tie(self, tmp_path, glob_order):
        _make_checkpoints(tmp_path, {10: 0.9, 20: 0.9, 30: 0.5}, "val_acc")
        mgr = _manager(tmp_path, "val_acc", higher_is_better=True, keep_top_k=1)
        assert mgr.get_best_checkpoint_path() == str(tmp_path / "step_20")

    def test_lower_is_better_prefers_the_later_step_on_a_tie(self, tmp_path, glob_order):
        _make_checkpoints(tmp_path, {10: 0.1, 20: 0.1, 30: 0.8}, "val_loss")
        mgr = _manager(tmp_path, "val_loss", higher_is_better=False, keep_top_k=1)
        assert mgr.get_best_checkpoint_path() == str(tmp_path / "step_20")

    def test_a_clear_winner_is_unaffected_by_the_tiebreak(self, tmp_path, glob_order):
        """Discrimination: the fix must not change non-tied selection."""
        _make_checkpoints(tmp_path, {10: 0.5, 20: 0.9, 30: 0.7}, "val_acc")
        mgr = _manager(tmp_path, "val_acc", higher_is_better=True, keep_top_k=1)
        assert mgr.get_best_checkpoint_path() == str(tmp_path / "step_20")

    def test_lower_is_better_clear_winner(self, tmp_path, glob_order):
        _make_checkpoints(tmp_path, {10: 0.5, 20: 0.1, 30: 0.7}, "val_loss")
        mgr = _manager(tmp_path, "val_loss", higher_is_better=False, keep_top_k=1)
        assert mgr.get_best_checkpoint_path() == str(tmp_path / "step_20")


class TestSelectionAgreesWithRetention:
    """The load-bearing property: the best checkpoint must still exist."""

    @pytest.mark.parametrize(
        "metrics,higher,keep",
        [
            ({10: 0.9, 20: 0.9, 30: 0.5}, True, 1),
            ({10: 0.9, 20: 0.9, 30: 0.5}, True, 2),
            ({10: 0.1, 20: 0.1, 30: 0.8}, False, 1),
            ({5: 0.4, 6: 0.4, 7: 0.4, 8: 0.4}, True, 2),
            ({5: 0.4, 6: 0.4, 7: 0.4, 8: 0.4}, False, 2),
            ({1: 0.3, 2: 0.7, 3: 0.7, 4: 0.2}, True, 2),
        ],
    )
    def test_best_checkpoint_survives_retention(
        self, tmp_path, metrics, higher, keep, glob_order
    ):
        _make_checkpoints(tmp_path, metrics, "m")
        mgr = _manager(tmp_path, "m", higher_is_better=higher, keep_top_k=keep)

        best = mgr.get_best_checkpoint_path()
        mgr.remove_old_checkpoints(exclude_latest=False)

        assert best is not None
        assert Path(best).exists(), (
            f"get_best_checkpoint_path returned {best}, which remove_old_checkpoints "
            f"deleted; survivors are {sorted(_surviving_steps(tmp_path))}"
        )

    def test_best_is_still_best_after_retention_runs(self, tmp_path, glob_order):
        """Selection must be stable across a retention pass, not merely survive it."""
        _make_checkpoints(tmp_path, {10: 0.9, 20: 0.9, 30: 0.5}, "m")
        mgr = _manager(tmp_path, "m", higher_is_better=True, keep_top_k=2)

        before = mgr.get_best_checkpoint_path()
        mgr.remove_old_checkpoints(exclude_latest=False)
        after = mgr.get_best_checkpoint_path()
        assert before == after


class TestOrderingIsIndependentOfDirectoryOrder:
    """_load_checkpoint_history returns glob order; ranking must not inherit it."""

    def test_creation_order_does_not_change_the_answer(self, tmp_path):
        forward = tmp_path / "forward"
        reverse = tmp_path / "reverse"
        pairs = {10: 0.9, 20: 0.9, 30: 0.9}

        _make_checkpoints(forward, dict(sorted(pairs.items())), "m")
        _make_checkpoints(reverse, dict(sorted(pairs.items(), reverse=True)), "m")

        best_f = _manager(forward, "m", True, 1).get_best_checkpoint_path()
        best_r = _manager(reverse, "m", True, 1).get_best_checkpoint_path()

        assert best_f is not None and best_r is not None
        assert Path(best_f).name == Path(best_r).name == "step_30"


class TestUnchangedBehaviour:
    """Paths the fix must leave alone."""

    def test_no_metric_falls_back_to_latest(self, tmp_path):
        _make_checkpoints(tmp_path, {10: 0.9, 20: 0.1}, "m")
        mgr = _manager(tmp_path, None, higher_is_better=True, keep_top_k=1)
        assert mgr.get_best_checkpoint_path() == str(tmp_path / "step_20")

    def test_no_checkpoints_returns_none(self, tmp_path):
        mgr = _manager(tmp_path, "m", higher_is_better=True, keep_top_k=1)
        assert mgr.get_best_checkpoint_path() is None

    def test_checkpoints_missing_the_metric_are_filtered_with_a_warning(self, tmp_path):
        _make_checkpoints(tmp_path, {10: 0.9}, "m")
        d = tmp_path / "step_20"
        d.mkdir()
        (d / "training_info.json").write_text(json.dumps({"other": 1.0}))

        mgr = _manager(tmp_path, "m", higher_is_better=True, keep_top_k=1)
        with pytest.warns(UserWarning, match="do not have metric"):
            assert mgr.get_best_checkpoint_path() == str(tmp_path / "step_10")

    def test_keep_top_k_none_deletes_nothing(self, tmp_path):
        _make_checkpoints(tmp_path, {10: 0.9, 20: 0.5, 30: 0.1}, "m")
        mgr = _manager(tmp_path, "m", higher_is_better=True, keep_top_k=None)
        mgr.remove_old_checkpoints(exclude_latest=False)
        assert _surviving_steps(tmp_path) == {10, 20, 30}

    def test_retention_keeps_exactly_k(self, tmp_path):
        _make_checkpoints(tmp_path, {10: 0.9, 20: 0.5, 30: 0.1}, "m")
        mgr = _manager(tmp_path, "m", higher_is_better=True, keep_top_k=2)
        mgr.remove_old_checkpoints(exclude_latest=False)
        assert _surviving_steps(tmp_path) == {10, 20}
