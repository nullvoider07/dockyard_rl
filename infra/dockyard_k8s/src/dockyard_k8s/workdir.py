"""Stage a working directory for Ray Job ``runtime_env.working_dir`` upload.

Ray's client-side packager honours ``.gitignore``, which can silently drop
files needed at runtime — the staged copy strips the ``.gitignore`` so Ray
uploads everything requested. Caches/venvs/git are skipped to keep the zip
under Ray's 100 MiB dashboard cap.

Each call stages into a fresh tmpdir; the caller owns cleanup (Ray uploads
to GCS before the SDK call returns, so deletion is safe afterwards).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

_IGNORE_PATTERNS = shutil.ignore_patterns(
    ".gitignore",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".venv",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
)


# Repo-relative subtrees uploaded by default for the dockyard_rl GRPO stack.
# The training entrypoint (examples/run_grpo_swe.py) imports the package
# modules at the repo root plus the examples/ configs and launcher.
DEFAULT_RAY_UPLOAD_PATHS = [
    "algorithms",
    "cluster",
    "data",
    "distributed",
    "environments",
    "experience",
    "models",
    "rewards",
    "sandbox",
    "utils",
    "examples",
    "pyproject.toml",
]


def stage_workdir(
    repo_root: Path,
    *,
    include_paths: list[str] | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Copy the requested subtrees of ``repo_root`` into a fresh tmpdir.

    Args:
        repo_root: absolute path to the dockyard_rl repo root.
        include_paths: subset of paths (relative to ``repo_root``) to stage.
            Defaults to :data:`DEFAULT_RAY_UPLOAD_PATHS`.
        extra_files: ``{relative_path: content}`` — additional files to
            create in the staged tree (e.g. the merged recipe YAML).

    Returns:
        Absolute path to the staged working_dir.
    """
    if include_paths is None:
        include_paths = DEFAULT_RAY_UPLOAD_PATHS

    dest = Path(tempfile.mkdtemp(prefix="dockyard-k8s-workdir-"))

    for rel in include_paths:
        src = (repo_root / rel).resolve()
        if not src.exists():
            # Missing optional paths are OK (e.g. pyproject.toml may move).
            continue
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, tgt, ignore=_IGNORE_PATTERNS)
        else:
            shutil.copy2(src, tgt)

    for rel, content in (extra_files or {}).items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    return dest


__all__ = ["DEFAULT_RAY_UPLOAD_PATHS", "stage_workdir"]
