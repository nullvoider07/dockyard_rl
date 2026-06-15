#!/bin/bash
# SWE-Bench episode environment setup — runs inside ubuntu-swe containers.
#
# Called by the task executor (port 9090) at the start of each episode
# BEFORE the agent takes any actions.  Sets up the target repository at
# the correct commit so the agent finds a reproducible starting state.
#
# Arguments (all required):
#   $1  INSTANCE_ID   — SWE-bench instance identifier (e.g. "django__django-11099")
#   $2  REPO          — GitHub repo slug (e.g. "django/django")
#   $3  BASE_COMMIT   — Git commit SHA the agent starts from
#   $4  ENV_COMMIT    — Git commit for environment/dependency setup
#                       (may differ from BASE_COMMIT for some instances)
#   $5  WORKSPACE_DIR — Absolute path where the repo will be placed
#                       (default: /workspace)
#
# On success: exits 0 and prints "SETUP_COMPLETE:<instance_id>" to stdout.
# On failure: exits non-zero with error details on stderr.
#
# Idempotent: if the workspace already contains the correct commit,
# the script skips the clone/checkout steps.

set -euo pipefail

# Arguments and paths
INSTANCE_ID="${1:?Usage: $0 INSTANCE_ID REPO BASE_COMMIT ENV_COMMIT [WORKSPACE_DIR]}"
REPO="${2:?REPO is required}"
BASE_COMMIT="${3:?BASE_COMMIT is required}"
ENV_COMMIT="${4:?ENV_COMMIT is required}"
WORKSPACE_DIR="${5:-/workspace}"

REPO_DIR="$WORKSPACE_DIR/repo"
CACHE_DIR="/tmp/dockyard/swe_bench_cache"
SETUP_LOG="$WORKSPACE_DIR/setup.log"

mkdir -p "$WORKSPACE_DIR" "$CACHE_DIR"

echo "[setup] Instance: $INSTANCE_ID" | tee -a "$SETUP_LOG"
echo "[setup] Repo: $REPO @ $BASE_COMMIT" | tee -a "$SETUP_LOG"
echo "[setup] Env commit: $ENV_COMMIT" | tee -a "$SETUP_LOG"

# Idempotency check
# If the workspace already has the right commit, skip clone/checkout.
if [[ -d "$REPO_DIR/.git" ]]; then
    CURRENT_COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "")
    if [[ "$CURRENT_COMMIT" == "$BASE_COMMIT"* ]] || \
       [[ "$BASE_COMMIT" == "$CURRENT_COMMIT"* ]]; then
        echo "[setup] Workspace already at correct commit $CURRENT_COMMIT, skipping clone." \
            | tee -a "$SETUP_LOG"
        echo "SETUP_COMPLETE:$INSTANCE_ID"
        exit 0
    fi
    echo "[setup] Workspace at wrong commit ($CURRENT_COMMIT vs $BASE_COMMIT), resetting." \
        | tee -a "$SETUP_LOG"
    rm -rf "$REPO_DIR"
fi

# Clone
# Use a bare cache clone to avoid redundant full downloads across episodes
# targeting the same repo.
CACHE_REPO_DIR="$CACHE_DIR/$(echo "$REPO" | tr '/' '__').git"

if [[ ! -d "$CACHE_REPO_DIR" ]]; then
    echo "[setup] Cloning $REPO into cache..." | tee -a "$SETUP_LOG"
    git clone --bare --filter=blob:none \
        "https://github.com/$REPO.git" \
        "$CACHE_REPO_DIR" \
        2>> "$SETUP_LOG"
else
    echo "[setup] Cache hit for $REPO, fetching latest..." | tee -a "$SETUP_LOG"
    git -C "$CACHE_REPO_DIR" fetch --filter=blob:none origin 2>> "$SETUP_LOG" || true
fi

# Clone from the local cache into the workspace.
echo "[setup] Checking out $BASE_COMMIT..." | tee -a "$SETUP_LOG"
git clone --local "$CACHE_REPO_DIR" "$REPO_DIR" 2>> "$SETUP_LOG"
git -C "$REPO_DIR" fetch origin "$BASE_COMMIT" 2>> "$SETUP_LOG" || true
git -C "$REPO_DIR" checkout "$BASE_COMMIT" 2>> "$SETUP_LOG"

# Python environment setup
# Detect the Python version required by this repo at ENV_COMMIT and install
# dependencies using the appropriate tool (pip, poetry, tox, etc.).
#
# ubuntu-swe has Python 3.14 as the system Python.  Many SWE-bench repos
# target Python 3.8–3.11.  We use uv (already installed in ubuntu-swe at
# /usr/local/bin/uvx and /usr/local/bin/uv) to create isolated venvs with
# the correct interpreter version when needed.

VENV_DIR="$WORKSPACE_DIR/.venv"
PYTHON_BIN="/usr/local/bin/python3"

detect_python_version() {
    # Try to read the required Python version from common config files.
    local pyver=""

    # .python-version (pyenv convention)
    if [[ -f "$REPO_DIR/.python-version" ]]; then
        pyver=$(cat "$REPO_DIR/.python-version" | tr -d '[:space:]')
    fi

    # setup.cfg python_requires
    if [[ -z "$pyver" ]] && [[ -f "$REPO_DIR/setup.cfg" ]]; then
        pyver=$(grep -oP '(?<=python_requires\s=\s)[\d.]+' \
            "$REPO_DIR/setup.cfg" 2>/dev/null || true)
    fi

    # pyproject.toml requires-python
    if [[ -z "$pyver" ]] && [[ -f "$REPO_DIR/pyproject.toml" ]]; then
        pyver=$(grep -oP '(?<=requires-python\s=\s")[^"]+' \
            "$REPO_DIR/pyproject.toml" 2>/dev/null \
            | grep -oP '[\d]+\.[\d]+' | head -1 || true)
    fi

    echo "${pyver:-3.11}"
}

REQUIRED_PYTHON=$(detect_python_version)
echo "[setup] Required Python: $REQUIRED_PYTHON" | tee -a "$SETUP_LOG"

# Create a venv with the required Python version using uv.
if [[ -x "/usr/local/bin/uv" ]]; then
    /usr/local/bin/uv venv \
        --python "$REQUIRED_PYTHON" \
        "$VENV_DIR" \
        2>> "$SETUP_LOG" || {
        echo "[setup] uv venv creation failed, falling back to system Python" \
            | tee -a "$SETUP_LOG"
        python3 -m venv "$VENV_DIR" 2>> "$SETUP_LOG"
    }
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    python3 -m venv "$VENV_DIR" 2>> "$SETUP_LOG"
    PYTHON_BIN="$VENV_DIR/bin/python"
fi

PIP_BIN="$VENV_DIR/bin/pip"

# Install dependencies based on what the repo provides.
install_dependencies() {
    local failed=0

    # Always upgrade pip first.
    "$PIP_BIN" install --quiet --upgrade pip 2>> "$SETUP_LOG" || true

    if [[ -f "$REPO_DIR/requirements.txt" ]]; then
        echo "[setup] Installing requirements.txt..." | tee -a "$SETUP_LOG"
        "$PIP_BIN" install --quiet -r "$REPO_DIR/requirements.txt" \
            2>> "$SETUP_LOG" || failed=1
    fi

    if [[ -f "$REPO_DIR/requirements-dev.txt" ]]; then
        "$PIP_BIN" install --quiet -r "$REPO_DIR/requirements-dev.txt" \
            2>> "$SETUP_LOG" || true
    fi

    if [[ -f "$REPO_DIR/setup.py" ]] || [[ -f "$REPO_DIR/setup.cfg" ]]; then
        echo "[setup] Installing package in editable mode (setup.py)..." \
            | tee -a "$SETUP_LOG"
        "$PIP_BIN" install --quiet -e "$REPO_DIR" \
            2>> "$SETUP_LOG" || failed=1
    elif [[ -f "$REPO_DIR/pyproject.toml" ]]; then
        echo "[setup] Installing package in editable mode (pyproject.toml)..." \
            | tee -a "$SETUP_LOG"
        "$PIP_BIN" install --quiet -e "$REPO_DIR" \
            2>> "$SETUP_LOG" || failed=1
    fi

    if [[ $failed -ne 0 ]]; then
        echo "[setup] WARNING: Some dependency installations failed. \
The agent may still be able to work." | tee -a "$SETUP_LOG"
    fi
}

install_dependencies

# Test runner detection
# Write a small config file the task executor reads to know which test runner
# and discovery path to use when evaluating the agent's solution.
TEST_RUNNER="pytest"
TEST_PATHS="."

if [[ -f "$REPO_DIR/tox.ini" ]]; then
    TEST_RUNNER="tox"
elif [[ -f "$REPO_DIR/Makefile" ]] && grep -q "^test:" "$REPO_DIR/Makefile" 2>/dev/null; then
    TEST_RUNNER="make test"
fi

# Write test runner config for the task executor.
cat > "$WORKSPACE_DIR/test_runner.json" << TRCFG
{
    "runner":       "$TEST_RUNNER",
    "test_paths":   "$TEST_PATHS",
    "python_bin":   "$PYTHON_BIN",
    "repo_dir":     "$REPO_DIR",
    "instance_id":  "$INSTANCE_ID"
}
TRCFG

# Integrity snapshot
# Capture a hash of the test tree at episode start.  The reward function
# (rewards/integrity.py) re-hashes after the agent acts and compares.
# Any change to test files zeroes the reward regardless of test outcomes.
#
# Snapshot stored at $WORKSPACE_DIR/test_tree_snapshot.json — the task
# executor owns this file; the agent cannot reach it.
python3 - << 'SNAPSHOT'
import hashlib, json, os, sys

repo_dir   = os.environ.get("REPO_DIR", sys.argv[1] if len(sys.argv) > 1 else ".")
out_path   = os.path.join(os.environ.get("WORKSPACE_DIR", "/workspace"),
                          "test_tree_snapshot.json")

snapshot = {}
test_patterns = ("test_*.py", "*_test.py", "tests.py")

for root, dirs, files in os.walk(repo_dir):
    # Prune non-test directories for speed.
    dirs[:] = [
        d for d in dirs
        if d not in (".git", "__pycache__", ".tox", ".venv", "build", "dist")
    ]
    for fname in files:
        is_test = any(
            fname.startswith("test_") or fname.endswith("_test.py")
            or fname == "tests.py"
            for _ in [0]
        )
        if not is_test:
            continue
        fpath = os.path.join(root, fname)
        rel   = os.path.relpath(fpath, repo_dir)
        try:
            with open(fpath, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            snapshot[rel] = digest
        except OSError:
            pass

with open(out_path, "w") as f:
    json.dump(snapshot, f, indent=2, sort_keys=True)

print(f"[setup] Test tree snapshot: {len(snapshot)} files hashed → {out_path}")
SNAPSHOT

export REPO_DIR="$REPO_DIR"
export WORKSPACE_DIR="$WORKSPACE_DIR"

# Expose venv to agent shell
# Write a shell profile that activates the venv so any bash command the
# agent runs picks up the correct Python and installed packages.
cat > "$WORKSPACE_DIR/agent_profile.sh" << PROFILE
# Sourced by the task executor before running agent shell commands.
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:\$PATH"
export REPO_DIR="$REPO_DIR"
export WORKSPACE_DIR="$WORKSPACE_DIR"
unset PYTHONHOME
cd "$REPO_DIR"
PROFILE

echo "[setup] Environment ready." | tee -a "$SETUP_LOG"
echo "SETUP_COMPLETE:$INSTANCE_ID"