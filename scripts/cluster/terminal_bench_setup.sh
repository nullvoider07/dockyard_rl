#!/bin/bash
# Terminal-Bench 2.1 task provisioning — one-time, fleet/data-side (not per-episode).
#
# Terminal-Bench 2.1 ships each task as a directory under tasks/ in the
# harbor-framework/terminal-bench-2-1 repo: task.toml (image ref, timeouts,
# resource/network policy), instruction.md (the agent prompt), environment/
# (a Dockerfile + build context — the build fallback when task.toml has no
# prebuilt docker_image), and tests/ (test.sh + test_outputs.py — the held-out
# verifier). This script fetches tasks/ at a pinned ref into a directory the
# dataset loader (data/datasets/terminal_bench.py) reads to embed each task's
# image ref / Dockerfile / verifier into its row metadata.
#
# The agent never sees tests/: the environment late-injects it via the executor
# session /finish step (docker cp after the agent phase), so it cannot be
# tampered with during the rollout.
#
# Environment (all optional):
#   DOCKYARD_TERMINAL_BENCH_TASKS_DIR  target dir (default: ./data/terminal_bench/tasks)
#   DOCKYARD_TERMINAL_BENCH_REPO       source repo (default: harbor-framework/terminal-bench-2-1)
#   DOCKYARD_TERMINAL_BENCH_REF        pinned commit SHA or branch (default: main)
#
# On success: prints "TERMINAL_BENCH_SETUP_COMPLETE:<resolved_sha>:<tasks>".

set -euo pipefail

TASKS_DIR="${DOCKYARD_TERMINAL_BENCH_TASKS_DIR:-./data/terminal_bench/tasks}"
REPO_URL="${DOCKYARD_TERMINAL_BENCH_REPO:-https://github.com/harbor-framework/terminal-bench-2-1.git}"
REF="${DOCKYARD_TERMINAL_BENCH_REF:-main}"

TASKS_DIR="$(mkdir -p "$TASKS_DIR" && cd "$TASKS_DIR" && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "[tb-setup] Source: $REPO_URL @ $REF"
echo "[tb-setup] Target: $TASKS_DIR"

# Blobless + sparse clone so only tasks/ is materialised.
git clone --filter=blob:none --no-checkout --depth 1 --branch "$REF" \
    "$REPO_URL" "$TMP_DIR/repo" 2>/dev/null \
  || git clone --filter=blob:none --no-checkout "$REPO_URL" "$TMP_DIR/repo"

git -C "$TMP_DIR/repo" sparse-checkout init --cone
git -C "$TMP_DIR/repo" sparse-checkout set tasks
# --depth/--branch only resolves named refs; a raw SHA needs an explicit fetch.
if ! git -C "$TMP_DIR/repo" checkout "$REF" 2>/dev/null; then
    git -C "$TMP_DIR/repo" fetch --filter=blob:none origin "$REF"
    git -C "$TMP_DIR/repo" checkout "$REF"
fi

SRC="$TMP_DIR/repo/tasks"
if [[ ! -d "$SRC" ]]; then
    echo "[tb-setup] ERROR: tasks/ not found at $REF" >&2
    exit 1
fi

RESOLVED_SHA="$(git -C "$TMP_DIR/repo" rev-parse HEAD)"

# Sync into the target (rsync if present, else cp); --delete keeps it canonical.
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$SRC/" "$TASKS_DIR/"
else
    rm -rf "${TASKS_DIR:?}/"*
    cp -a "$SRC/." "$TASKS_DIR/"
fi

# Record provenance for reproducibility.
cat > "$TASKS_DIR/../PROVENANCE.json" <<EOF
{
  "repo": "$REPO_URL",
  "ref": "$REF",
  "resolved_sha": "$RESOLVED_SHA",
  "fetched_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# A task is any subdir containing a task.toml.
TASKS="$(find "$TASKS_DIR" -mindepth 2 -maxdepth 2 -name task.toml | wc -l | tr -d ' ')"
echo "[tb-setup] Provisioned $TASKS task dirs at $RESOLVED_SHA"

echo "TERMINAL_BENCH_SETUP_COMPLETE:$RESOLVED_SHA:$TASKS"
