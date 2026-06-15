"""Unit tests for DTensor pretrained-checkpoint path resolution (D8 wiring).

CheckpointManager.get_pretrained_paths / get_resume_paths are pure path logic
(no GPU): they locate policy/weights[/optimizer] under a checkpoint dir. The
worker's actual tensor load is the same loader resume uses and is GPU-deferred.
"""

import pytest

from dockyard_rl.utils.checkpoint import CheckpointManager


def _make_ckpt(root, with_optimizer=True):
    (root / "policy" / "weights").mkdir(parents=True)
    if with_optimizer:
        (root / "policy" / "optimizer").mkdir(parents=True)
    return root


class TestPretrainedPaths:
    def test_weights_and_optimizer(self, tmp_path):
        ckpt = _make_ckpt(tmp_path / "step_100")
        w, o = CheckpointManager.get_pretrained_paths({"path": str(ckpt)})
        assert w == ckpt / "policy" / "weights"
        assert o == ckpt / "policy" / "optimizer"

    def test_restore_optimizer_false(self, tmp_path):
        ckpt = _make_ckpt(tmp_path / "step_100")
        w, o = CheckpointManager.get_pretrained_paths(
            {"path": str(ckpt), "restore_optimizer": False}
        )
        assert w == ckpt / "policy" / "weights"
        assert o is None  # optimizer present but restore disabled

    def test_missing_optimizer_warns_and_nulls(self, tmp_path):
        ckpt = _make_ckpt(tmp_path / "step_100", with_optimizer=False)
        with pytest.warns(UserWarning):
            w, o = CheckpointManager.get_pretrained_paths({"path": str(ckpt)})
        assert w == ckpt / "policy" / "weights"
        assert o is None


class TestResumePaths:
    def test_none_returns_none_none(self):
        assert CheckpointManager.get_resume_paths(None) == (None, None)

    def test_with_optimizer(self, tmp_path):
        ckpt = _make_ckpt(tmp_path / "step_5")
        w, o = CheckpointManager.get_resume_paths(ckpt)
        assert w == ckpt / "policy" / "weights"
        assert o == ckpt / "policy" / "optimizer"

    def test_no_optimizer_warns_and_nulls(self, tmp_path):
        ckpt = _make_ckpt(tmp_path / "step_5", with_optimizer=False)
        with pytest.warns(UserWarning):
            w, o = CheckpointManager.get_resume_paths(ckpt)
        assert w == ckpt / "policy" / "weights"
        assert o is None
