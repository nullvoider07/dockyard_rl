"""Offline manifest validation with kubeconform.

Renders the hardened manifests (the H1/H2 additions — securityContext,
NetworkPolicy, PodDisruptionBudget, ResourceQuota, LimitRange — plus the
sandbox Deployment + Service) and validates them against the upstream
Kubernetes JSON schemas with ``kubeconform``. All of these are built-in
resources, so ``-strict`` catches unknown / misspelled fields.

Skipped when ``kubeconform`` is not on PATH (local dev); runs in CI where the
binary is installed. CRDs (RayJob/RayCluster/ComputeDomain) are intentionally
out of scope here — they need their own schema sources and are covered by the
cluster-side ``kubectl --dry-run`` path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from dockyard_k8s.config import LoadedConfig
from dockyard_k8s.render import render_manifests
from dockyard_k8s.schema import InfraConfig

_KUBECONFORM = shutil.which("kubeconform")
_REQUIRES_KUBECONFORM = pytest.mark.skipif(
    _KUBECONFORM is None, reason="kubeconform not installed"
)

# Repo paths (this file: infra/dockyard_k8s/tests/unit/test_kubeconform.py).
_INFRA_ROOT = Path(__file__).resolve().parents[3]
_SANDBOX_POOL_YAML = _INFRA_ROOT / "examples" / "sandbox-pool.yaml"


def _kubeconform(path: Path) -> subprocess.CompletedProcess:
    assert _KUBECONFORM is not None
    return subprocess.run(
        [
            _KUBECONFORM,
            "-strict",
            "-ignore-missing-schemas",
            "-summary",
            str(path),
        ],
        capture_output=True,
        text=True,
    )


def _hardened_infra() -> InfraConfig:
    """attach-mode infra whose render is all built-in resources (no CRDs)."""
    return InfraConfig.model_validate(
        {
            "namespace": "dockyard",
            "image": "nullvoider/ubuntu-swe:latest",
            "priorityClassName": "dockyard-high",
            "launch": {"mode": "attach", "attach": "rc-gpu"},
            "sandbox": {
                "name": "dockyard-sandbox",
                "replicas": 4,
                "resources": {
                    "cpu": "1",
                    "memory": "2Gi",
                    "ephemeralStorage": "10Gi",
                    "cpuLimit": "4",
                    "memoryLimit": "8Gi",
                },
                "topologySpreadConstraints": [
                    {
                        "maxSkew": 1,
                        "topologyKey": "kubernetes.io/hostname",
                        "whenUnsatisfiable": "ScheduleAnyway",
                        "labelSelector": {
                            "matchLabels": {"app.kubernetes.io/name": "dockyard-sandbox"}
                        },
                    }
                ],
            },
            "resourceQuota": {"enabled": True, "hard": {"requests.cpu": "200"}},
            "limitRange": {
                "enabled": True,
                "default": {"cpu": "2", "memory": "4Gi"},
                "max": {"cpu": "16", "memory": "64Gi"},
            },
        }
    )


@_REQUIRES_KUBECONFORM
def test_rendered_hardening_manifests_validate(tmp_path) -> None:
    infra = _hardened_infra()
    loaded = LoadedConfig(
        recipe=OmegaConf.create({}), infra=infra, source_path=Path("r.yaml")
    )
    manifests = render_manifests(loaded)
    # Sanity: the built-in hardening objects are present (no CRDs in attach mode).
    kinds = {m["kind"] for m in manifests}
    assert {
        "ResourceQuota",
        "LimitRange",
        "Deployment",
        "Service",
        "NetworkPolicy",
        "PodDisruptionBudget",
    } <= kinds

    out = tmp_path / "rendered.yaml"
    out.write_text(yaml.safe_dump_all(manifests, sort_keys=False))
    result = _kubeconform(out)
    assert result.returncode == 0, f"kubeconform failed:\n{result.stdout}\n{result.stderr}"


@_REQUIRES_KUBECONFORM
def test_handwritten_sandbox_pool_validates() -> None:
    result = _kubeconform(_SANDBOX_POOL_YAML)
    assert result.returncode == 0, f"kubeconform failed:\n{result.stdout}\n{result.stderr}"
