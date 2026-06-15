"""Shared fixtures for the dockyard-k8s unit tests.

The fixtures isolate the loader from the host environment (no stray
``~/.config/dockyard-k8s/defaults.yaml`` and a deterministic ``${user:}``)
and provide small factories for the two free-form objects the builders take:
an :class:`InfraConfig` and a GPU :class:`ClusterSpec`.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from dockyard_k8s import config as config_mod
from src.dockyard_k8s.schema import ClusterSpec, InfraConfig, SandboxSpec


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch) -> None:
    """Pin the username and point user-defaults at a path that does not exist."""
    monkeypatch.setenv("DOCKYARD_K8S_USER", "tester")
    monkeypatch.setattr(config_mod, "_USER_DEFAULTS", tmp_path / "no-such-defaults.yaml")


def _minimal_cluster_spec() -> dict[str, Any]:
    """A two-worker-group RayCluster body (head + trainer + inference)."""
    def _worker(group: str, role: str) -> dict[str, Any]:
        return {
            "groupName": group,
            "replicas": 1,
            "minReplicas": 1,
            "maxReplicas": 1,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "ray-worker",
                            "env": [{"name": "DOCKYARD_FLEET_ROLE", "value": role}],
                        }
                    ]
                }
            },
        }

    return {
        "rayVersion": "2.52.0",
        "headGroupSpec": {
            "template": {"spec": {"containers": [{"name": "ray-head"}]}}
        },
        "workerGroupSpecs": [
            _worker("trainer", "trainer"),
            _worker("inference", "inference"),
        ],
    }


@pytest.fixture
def make_cluster() -> Callable[..., ClusterSpec]:
    def _make(name: str = "dockyard-grpo", **kwargs: Any) -> ClusterSpec:
        spec = kwargs.pop("spec", None) or _minimal_cluster_spec()
        return ClusterSpec(name=name, spec=spec, **kwargs)

    return _make


@pytest.fixture
def make_infra(make_cluster) -> Callable[..., InfraConfig]:
    def _make(**kwargs: Any) -> InfraConfig:
        payload: dict[str, Any] = {
            "namespace": "dockyard",
            "image": "nullvoider/ubuntu-swe:latest",
        }
        payload.update(kwargs)
        return InfraConfig.model_validate(payload)

    return _make


@pytest.fixture
def sandbox() -> SandboxSpec:
    return SandboxSpec(name="dockyard-sandbox", replicas=3, port=9090)
