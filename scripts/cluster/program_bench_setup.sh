#!/bin/bash
# ProgramBench provisioning — one-time, fleet/data-side (not per-episode).
#
# ProgramBench grades a cleanroom reconstruction: the agent rebuilds a program
# (in its fixed source language) from an execute-only binary + docs that ship
# inside the per-instance image programbench/<id>:task_cleanroom, then the
# submission is rebuilt in a clean image and scored by per-"branch" pytest
# suites. Two sources must be merged per instance:
#   - task.yaml + tests.json  → facebookresearch/ProgramBench repo
#     (src/programbench/data/tasks/<id>/): language, repo/commit, eval_clean_hashes,
#     and the per-branch list of expected pytest node IDs.
#   - tests/<branch>.tar.gz   → HF dataset programbench/ProgramBench-Tests
#     (<id>/tests/): the generated behavioural test suites (eval/run.sh + eval/tests/).
#
# This script provisions a merged tree the dataset loader
# (data/datasets/program_bench.py) reads:
#   <dir>/<id>/{task.yaml, tests.json, tests/<branch>.tar.gz}
# The dataset embeds the selected branches' tars (+ expected node IDs) into each
# row; the reward stages them into a clean-image grading task. The agent never
# sees the tests.
#
# Environment (all optional):
#   DOCKYARD_PROGRAM_BENCH_DIR        target dir (default: ./data/program_bench/tasks)
#   DOCKYARD_PROGRAM_BENCH_REPO       repo (default: facebookresearch/ProgramBench)
#   DOCKYARD_PROGRAM_BENCH_REF        repo ref (default: main)
#   DOCKYARD_PROGRAM_BENCH_HF         HF dataset (default: programbench/ProgramBench-Tests)
#   DOCKYARD_PROGRAM_BENCH_HF_REV     HF revision (default: main)
#   DOCKYARD_PROGRAM_BENCH_INSTANCES  space/comma-separated subset of instance ids
#                                     (default: all 200 — the full corpus is ~8.2 GB)
#
# On success: prints "PROGRAM_BENCH_SETUP_COMPLETE:<repo_sha>:<instances>".

set -euo pipefail

DIR="${DOCKYARD_PROGRAM_BENCH_DIR:-./data/program_bench/tasks}"
REPO_URL="${DOCKYARD_PROGRAM_BENCH_REPO:-https://github.com/facebookresearch/ProgramBench.git}"
REF="${DOCKYARD_PROGRAM_BENCH_REF:-main}"
HF_NAME="${DOCKYARD_PROGRAM_BENCH_HF:-programbench/ProgramBench-Tests}"
HF_REV="${DOCKYARD_PROGRAM_BENCH_HF_REV:-main}"
INSTANCES="${DOCKYARD_PROGRAM_BENCH_INSTANCES:-}"

DIR="$(mkdir -p "$DIR" && cd "$DIR" && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "[pb-setup] Repo:    $REPO_URL @ $REF"
echo "[pb-setup] HF:      $HF_NAME @ $HF_REV"
echo "[pb-setup] Target:  $DIR"

# 1. Sparse-fetch the harness task manifests (task.yaml + tests.json; small).
git clone --filter=blob:none --no-checkout --depth 1 --branch "$REF" \
    "$REPO_URL" "$TMP_DIR/repo" 2>/dev/null \
  || git clone --filter=blob:none --no-checkout "$REPO_URL" "$TMP_DIR/repo"
git -C "$TMP_DIR/repo" sparse-checkout init --cone
git -C "$TMP_DIR/repo" sparse-checkout set src/programbench/data/tasks
if ! git -C "$TMP_DIR/repo" checkout "$REF" 2>/dev/null; then
    git -C "$TMP_DIR/repo" fetch --filter=blob:none origin "$REF"
    git -C "$TMP_DIR/repo" checkout "$REF"
fi
REPO_SHA="$(git -C "$TMP_DIR/repo" rev-parse HEAD)"
TASKS_SRC="$TMP_DIR/repo/src/programbench/data/tasks"
if [[ ! -d "$TASKS_SRC" ]]; then
    echo "[pb-setup] ERROR: data/tasks not found at $REF" >&2
    exit 1
fi

# Resolve the instance set (subset or all).
if [[ -n "$INSTANCES" ]]; then
    IFS=', ' read -r -a INSTANCE_LIST <<< "$INSTANCES"
else
    INSTANCE_LIST=()
    while IFS= read -r d; do INSTANCE_LIST+=("$(basename "$d")"); done \
        < <(find "$TASKS_SRC" -mindepth 1 -maxdepth 1 -type d | sort)
fi
echo "[pb-setup] Provisioning ${#INSTANCE_LIST[@]} instance(s)"

# 2. Download the per-instance test tars from HF (only the selected instances).
ALLOW=()
for inst in "${INSTANCE_LIST[@]}"; do ALLOW+=("$inst/tests/*"); done
python3 - "$HF_NAME" "$HF_REV" "$TMP_DIR/hf" "${ALLOW[@]}" <<'PY'
import sys
from huggingface_hub import snapshot_download
name, rev, dest, *allow = sys.argv[1:]
snapshot_download(repo_id=name, repo_type="dataset", revision=rev,
                  local_dir=dest, allow_patterns=allow or None)
print(f"[pb-setup] HF tests downloaded to {dest}")
PY

# 3. Merge task.yaml + tests.json + tests/ into <dir>/<id>/.
provisioned=0
for inst in "${INSTANCE_LIST[@]}"; do
    src_task="$TASKS_SRC/$inst"
    src_tests="$TMP_DIR/hf/$inst/tests"
    if [[ ! -f "$src_task/task.yaml" || ! -f "$src_task/tests.json" ]]; then
        echo "[pb-setup]   ⚠ skip $inst (no task.yaml/tests.json)"; continue
    fi
    if [[ ! -d "$src_tests" ]]; then
        echo "[pb-setup]   ⚠ skip $inst (no test tars on HF)"; continue
    fi
    dst="$DIR/$inst"
    mkdir -p "$dst/tests"
    cp -f "$src_task/task.yaml" "$src_task/tests.json" "$dst/"
    cp -f "$src_tests"/*.tar.gz "$dst/tests/" 2>/dev/null || true
    provisioned=$((provisioned + 1))
done

cat > "$DIR/../PROVENANCE.json" <<EOF
{
  "repo": "$REPO_URL",
  "ref": "$REF",
  "repo_sha": "$REPO_SHA",
  "hf_dataset": "$HF_NAME",
  "hf_revision": "$HF_REV",
  "instances": $provisioned,
  "fetched_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "[pb-setup] Provisioned $provisioned instance dir(s) at $REPO_SHA"
echo "PROGRAM_BENCH_SETUP_COMPLETE:$REPO_SHA:$provisioned"
