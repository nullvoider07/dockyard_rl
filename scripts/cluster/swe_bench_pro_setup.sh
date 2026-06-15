#!/bin/bash
# SWE-bench Pro harness provisioning — one-time, fleet/data-side (not per-episode).
#
# SWE-bench Pro scores each instance inside its prebuilt per-instance Docker
# image by running that instance's own run_script.sh and parser.py. Those
# scripts live in the scaleapi/SWE-bench_Pro-os repo under run_scripts/, NOT in
# the HuggingFace dataset. This script sparse-fetches run_scripts/ at a pinned
# ref into a directory the dataset loader (data/datasets/swe_bench_pro.py) reads
# to embed each instance's harness into its row metadata.
#
# The executor never sees these scripts directly: the reward stages them as
# image-task workspace files at scoring time.
#
# Environment (all optional):
#   DOCKYARD_SWE_BENCH_PRO_SCRIPTS_DIR  target dir (default: ./data/swe_bench_pro/run_scripts)
#   DOCKYARD_SWE_BENCH_PRO_REPO         source repo (default: scaleapi/SWE-bench_Pro-os)
#   DOCKYARD_SWE_BENCH_PRO_REF          pinned commit SHA or branch (default: main)
#   DOCKYARD_SWE_BENCH_PRO_PREFETCH     "1" to also pre-cache the HF dataset
#   DOCKYARD_SWE_BENCH_PRO_HF_NAME      HF dataset name (default: ScaleAI/SWE-bench_Pro)
#
# On success: prints "SWE_BENCH_PRO_SETUP_COMPLETE:<resolved_sha>:<instances>".

set -euo pipefail

SCRIPTS_DIR="${DOCKYARD_SWE_BENCH_PRO_SCRIPTS_DIR:-./data/swe_bench_pro/run_scripts}"
REPO_URL="${DOCKYARD_SWE_BENCH_PRO_REPO:-https://github.com/scaleapi/SWE-bench_Pro-os.git}"
REF="${DOCKYARD_SWE_BENCH_PRO_REF:-main}"
PREFETCH="${DOCKYARD_SWE_BENCH_PRO_PREFETCH:-0}"
HF_NAME="${DOCKYARD_SWE_BENCH_PRO_HF_NAME:-ScaleAI/SWE-bench_Pro}"

SCRIPTS_DIR="$(mkdir -p "$SCRIPTS_DIR" && cd "$SCRIPTS_DIR" && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "[pro-setup] Source: $REPO_URL @ $REF"
echo "[pro-setup] Target: $SCRIPTS_DIR"

# Blobless + sparse clone so only run_scripts/ is materialised (the repo also
# carries SWE-agent/mini-swe-agent submodules and trajectories we do not need).
git clone --filter=blob:none --no-checkout --depth 1 --branch "$REF" \
    "$REPO_URL" "$TMP_DIR/repo" 2>/dev/null \
  || git clone --filter=blob:none --no-checkout "$REPO_URL" "$TMP_DIR/repo"

git -C "$TMP_DIR/repo" sparse-checkout init --cone
git -C "$TMP_DIR/repo" sparse-checkout set run_scripts
# --depth/--branch only resolves named refs; a raw SHA needs an explicit fetch.
if ! git -C "$TMP_DIR/repo" checkout "$REF" 2>/dev/null; then
    git -C "$TMP_DIR/repo" fetch --filter=blob:none origin "$REF"
    git -C "$TMP_DIR/repo" checkout "$REF"
fi

SRC="$TMP_DIR/repo/run_scripts"
if [[ ! -d "$SRC" ]]; then
    echo "[pro-setup] ERROR: run_scripts/ not found at $REF" >&2
    exit 1
fi

RESOLVED_SHA="$(git -C "$TMP_DIR/repo" rev-parse HEAD)"

# Sync into the target (rsync if present, else cp); --delete keeps it canonical.
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$SRC/" "$SCRIPTS_DIR/"
else
    rm -rf "${SCRIPTS_DIR:?}/"*
    cp -a "$SRC/." "$SCRIPTS_DIR/"
fi

# Record provenance for reproducibility.
cat > "$SCRIPTS_DIR/../PROVENANCE.json" <<EOF
{
  "repo": "$REPO_URL",
  "ref": "$REF",
  "resolved_sha": "$RESOLVED_SHA",
  "fetched_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

INSTANCES="$(find "$SCRIPTS_DIR" -maxdepth 1 -type d -name 'instance_*' | wc -l | tr -d ' ')"
echo "[pro-setup] Provisioned $INSTANCES instance harness dirs at $RESOLVED_SHA"

if [[ "$PREFETCH" == "1" ]]; then
    echo "[pro-setup] Pre-caching HF dataset $HF_NAME ..."
    python3 - "$HF_NAME" <<'PY'
import sys
from datasets import load_dataset
name = sys.argv[1]
ds = load_dataset(name, split="test")
print(f"[pro-setup] Cached {len(ds)} rows of {name}")
PY
fi

echo "SWE_BENCH_PRO_SETUP_COMPLETE:$RESOLVED_SHA:$INSTANCES"
