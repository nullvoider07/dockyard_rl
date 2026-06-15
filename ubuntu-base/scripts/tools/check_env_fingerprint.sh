#!/bin/bash
# Validate that the running container's pip environment matches the fingerprint
# baked at image build time. Exits nonzero with a clear error on mismatch.
# Set SKIP_FINGERPRINT_CHECK=1 to bypass (development only).

set -euo pipefail

FINGERPRINT_FILE="/etc/rl-env-fingerprint"
GENERATOR="/usr/local/lib/generate-env-fingerprint.py"

if [ ! -f "$FINGERPRINT_FILE" ]; then
    echo "[fingerprint] WARNING: $FINGERPRINT_FILE not found — skipping check." >&2
    exit 0
fi

BAKED_HASH=$(python3 -c "
import json, sys
data = json.load(open('$FINGERPRINT_FILE'))
print(data.get('composite_hash', ''))
")

if [ -z "$BAKED_HASH" ]; then
    echo "[fingerprint] WARNING: composite_hash missing from fingerprint file — skipping check." >&2
    exit 0
fi

LIVE_HASH=$(python3 "$GENERATOR" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('composite_hash', ''))
")

if [ "$BAKED_HASH" != "$LIVE_HASH" ]; then
    echo "[fingerprint] ERROR: Environment mismatch detected." >&2
    echo "[fingerprint]   Baked hash : $BAKED_HASH" >&2
    echo "[fingerprint]   Live hash  : $LIVE_HASH" >&2
    echo "[fingerprint] The mounted workspace or a volume has altered the pip environment." >&2
    echo "[fingerprint] Set SKIP_FINGERPRINT_CHECK=1 to bypass (development only)." >&2
    exit 1
fi

echo "[fingerprint] OK — environment matches baked fingerprint."