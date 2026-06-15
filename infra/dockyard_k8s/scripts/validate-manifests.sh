#!/usr/bin/env bash
# Offline manifest validation with kubeconform — no cluster required.
#
# Validates the hand-written example manifests and the generator's rendered
# output against the upstream Kubernetes JSON schemas. CRDs (RayJob/RayCluster/
# ComputeDomain) have no bundled schema, so `-ignore-missing-schemas` skips
# them while every built-in object (Deployment, Service, NetworkPolicy,
# PodDisruptionBudget, ResourceQuota, LimitRange) is checked with `-strict`.
#
# Usage:  bash infra/dockyard_k8s/scripts/validate-manifests.sh
# Install kubeconform: https://github.com/yannh/kubeconform#installation
set -euo pipefail

if ! command -v kubeconform >/dev/null 2>&1; then
  echo "kubeconform not found on PATH — install it to run manifest validation." >&2
  echo "  https://github.com/yannh/kubeconform#installation" >&2
  exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pkg_root="$(dirname "$here")"          # infra/dockyard_k8s
infra_root="$(dirname "$pkg_root")"    # infra
examples_dir="$infra_root/examples"

KC=(kubeconform -strict -ignore-missing-schemas -summary)

echo "== hand-written manifests ($examples_dir) =="
for f in "$examples_dir"/*.yaml; do
  echo "-- $f"
  "${KC[@]}" "$f"
done

# Render the generator example and validate it too, when the CLI + a recipe are
# available. Skipped (not failed) if the recipe path is absent.
recipe="$infra_root/../examples/configs/grpo_swe.yaml"
gen_infra="$pkg_root/examples/grpo_swe.infra.yaml"
if command -v dockyard-k8s >/dev/null 2>&1 && [[ -f "$recipe" ]]; then
  echo "== generator render (grpo_swe) =="
  dockyard-k8s render "$recipe" --infra "$gen_infra" | "${KC[@]}" -
else
  echo "== generator render: skipped (dockyard-k8s CLI or recipe not available) =="
fi

echo "All manifests valid."
