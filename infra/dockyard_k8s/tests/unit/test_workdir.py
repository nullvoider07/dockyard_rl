"""Tests for dockyard_k8s.workdir — working_dir staging for Ray uploads."""

from __future__ import annotations

from pathlib import Path

from dockyard_k8s.workdir import DEFAULT_RAY_UPLOAD_PATHS, stage_workdir


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "algorithms").mkdir(parents=True)
    (root / "algorithms" / "grpo.py").write_text("x = 1\n")
    (root / "algorithms" / "__pycache__").mkdir()
    (root / "algorithms" / "__pycache__" / "grpo.cpython-314.pyc").write_text("junk")
    (root / "examples").mkdir()
    (root / "examples" / "run_grpo_swe.py").write_text("print('hi')\n")
    (root / "pyproject.toml").write_text("[project]\nname='dockyard_rl'\n")
    (root / ".gitignore").write_text("*.log\n")
    return root


class TestStageWorkdir:
    def test_copies_requested_subtrees(self, tmp_path) -> None:
        root = _make_repo(tmp_path)
        dest = stage_workdir(root, include_paths=["algorithms", "pyproject.toml"])
        assert (dest / "algorithms" / "grpo.py").read_text() == "x = 1\n"
        assert (dest / "pyproject.toml").exists()
        assert not (dest / "examples").exists()  # not requested

    def test_strips_pycache_and_gitignore(self, tmp_path) -> None:
        root = _make_repo(tmp_path)
        dest = stage_workdir(root, include_paths=["algorithms"])
        assert not (dest / "algorithms" / "__pycache__").exists()

    def test_missing_paths_skipped(self, tmp_path) -> None:
        root = _make_repo(tmp_path)
        dest = stage_workdir(root, include_paths=["algorithms", "does-not-exist"])
        assert (dest / "algorithms").exists()
        assert not (dest / "does-not-exist").exists()

    def test_extra_files_written(self, tmp_path) -> None:
        root = _make_repo(tmp_path)
        dest = stage_workdir(
            root, include_paths=["pyproject.toml"], extra_files={"dockyard_run.yaml": "a: 1\n"}
        )
        assert (dest / "dockyard_run.yaml").read_text() == "a: 1\n"

    def test_default_paths_are_repo_relative(self) -> None:
        assert "algorithms" in DEFAULT_RAY_UPLOAD_PATHS
        assert "examples" in DEFAULT_RAY_UPLOAD_PATHS
        assert all(not p.startswith("/") for p in DEFAULT_RAY_UPLOAD_PATHS)
