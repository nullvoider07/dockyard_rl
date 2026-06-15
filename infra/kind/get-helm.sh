#!/bin/bash
set -eou pipefail
HELM_VERSION=${HELM_VERSION:-v3.17.3}

HELM_BIN="$HOME/bin/helm"
mkdir -p "$HOME/bin"

ARCH=$(uname -m)
case $ARCH in
  x86_64)  ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
tmp_helm_dir=$(mktemp -d)
curl -sSL "https://get.helm.sh/helm-${HELM_VERSION}-${OS}-${ARCH}.tar.gz" | tar -xz -C "$tmp_helm_dir" --strip-components=1
cp "$tmp_helm_dir/helm" "$HELM_BIN"
rm -rf "$tmp_helm_dir"

echo "Installed helm $HELM_VERSION at $HELM_BIN"
if command -v helm &>/dev/null; then
  echo "helm is already on your PATH: $(command -v helm)"
else
  echo "helm is not on your PATH. To add it, run:"
  echo "  export PATH=\"\$HOME/bin:\$PATH\""
fi
