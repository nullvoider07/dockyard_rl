"""Dev pod manifest builder for ``dockyard-k8s dev``."""

from __future__ import annotations

from typing import Any

# Dev pod default: the vLLM ubuntu-swe build. The dev shell runs no inference
# engine, so this is backend-agnostic; pass --image nullvoider/ubuntu-swe-sglang
# to dev connect if you want to poke at the SGLang image specifically.
DEFAULT_IMAGE = "nullvoider/ubuntu-swe:latest"
DEFAULT_IMAGE_PULL_SECRET = "regcred"
_PVC_NAME = "dockyard-workspace"
_MOUNT_PATH = "/mnt/dockyard"


def build_dev_pod_manifest(
    username: str,
    namespace: str,
    image: str = DEFAULT_IMAGE,
) -> dict[str, Any]:
    user_dir = f"{_MOUNT_PATH}/{username}"
    secret_name = f"{username}-secrets"
    pod_name = f"{username}-dev-pod"

    command = (
        f"mkdir -p {user_dir} /root/.ssh && "
        'if [ -n "$SSH_KEY_CONTENT" ]; then '
        'printf "%s\\n" "$SSH_KEY_CONTENT" > /root/.ssh/$SSH_KEY_NAME && '
        "chmod 600 /root/.ssh/$SSH_KEY_NAME; "
        "fi && "
        'if [ -n "$RCLONE_CONF" ]; then '
        "mkdir -p /root/.config/rclone && "
        'printf "%s\\n" "$RCLONE_CONF" > /root/.config/rclone/rclone.conf && '
        "if ! command -v rclone >/dev/null 2>&1; then "
        "curl -sSf https://rclone.org/install.sh | bash; "
        "fi; "
        "fi && "
        "if ! command -v kubectl >/dev/null 2>&1; then "
        'ARCH=$(uname -m | sed "s/x86_64/amd64/;s/aarch64/arm64/") && '
        'curl -sLo /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/${ARCH}/kubectl" && '
        "chmod +x /usr/local/bin/kubectl; "
        "fi && "
        "sleep infinity"
    )

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "dockyard-k8s",
                "dockyard-k8s/owner": username,
                "dockyard-k8s/component": "dev-pod",
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "imagePullSecrets": [{"name": DEFAULT_IMAGE_PULL_SECRET}],
            # Land on a CPU node — the dev pod requests no GPU.
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {
                                        "key": "nvidia.com/gpu.product",
                                        "operator": "DoesNotExist",
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            "containers": [
                {
                    "name": "dev",
                    "image": image,
                    "command": ["sh", "-c", command],
                    "workingDir": user_dir,
                    # Set USER so getpass.getuser() / $USER returns the real
                    # owner, not "root". uid=0 stays so users can apt-install.
                    "env": [{"name": "USER", "value": username}],
                    "envFrom": [{"secretRef": {"name": secret_name, "optional": True}}],
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "5", "memory": "10Gi"},
                    },
                    "volumeMounts": [
                        {"name": "dockyard-workspace", "mountPath": _MOUNT_PATH},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "dockyard-workspace",
                    "persistentVolumeClaim": {"claimName": _PVC_NAME},
                },
            ],
        },
    }


__all__ = ["DEFAULT_IMAGE", "DEFAULT_IMAGE_PULL_SECRET", "build_dev_pod_manifest"]
