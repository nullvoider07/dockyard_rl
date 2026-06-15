"""Tests for dockyard_k8s.render — the ordered object list per deployment.

Pins manifest ordering (sandbox first), the launch-mode dispatch (rayjob /
bringup / attach), and the cross-cutting env injection into every GPU pod
(RAY_ENABLE_UV_RUN_RUNTIME_ENV + DOCKYARD_SANDBOX_URLS), including the
add-if-absent semantics that leave operator-set values untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dockyard_k8s.config import LoadedConfig
from dockyard_k8s.render import render_manifests
from dockyard_k8s.schema import SandboxSpec


def _loaded(infra) -> LoadedConfig:
    return LoadedConfig(recipe=OmegaConf.create({}), infra=infra, source_path=Path("r.yaml"))


def _kinds(manifests: list[dict]) -> list[str]:
    return [m["kind"] for m in manifests]


def _pod_specs_of_rayjob(rj: dict) -> list[dict]:
    rcs = rj["spec"]["rayClusterSpec"]
    specs = [rcs["headGroupSpec"]["template"]["spec"]]
    specs += [wg["template"]["spec"] for wg in rcs["workerGroupSpecs"]]
    return specs


def _env_of(pod_spec: dict, container_idx: int = 0) -> dict[str, str]:
    return {e["name"]: e["value"] for e in pod_spec["containers"][container_idx].get("env", [])}


class TestRenderOrderingAndMode:
    def test_rayjob_with_sandbox(self, make_cluster, make_infra, sandbox) -> None:
        infra = make_infra(
            kuberay=make_cluster().model_dump(),
            sandbox=sandbox.model_dump(),
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
        )
        manifests = render_manifests(_loaded(infra))
        # sandbox objects (Deployment + Service + default-on NetworkPolicy + PDB)
        # emitted before the GPU object.
        assert _kinds(manifests) == [
            "Deployment",
            "Service",
            "NetworkPolicy",
            "PodDisruptionBudget",
            "RayJob",
        ]

    def test_bringup_emits_raycluster(self, make_cluster, make_infra) -> None:
        infra = make_infra(kuberay=make_cluster().model_dump(), launch={"mode": "bringup"})
        manifests = render_manifests(_loaded(infra))
        assert _kinds(manifests) == ["RayCluster"]

    def test_attach_emits_no_gpu_object(self, make_infra, sandbox) -> None:
        infra = make_infra(
            sandbox=sandbox.model_dump(), launch={"mode": "attach", "attach": "rc-gpu"}
        )
        manifests = render_manifests(_loaded(infra))
        assert _kinds(manifests) == [
            "Deployment",
            "Service",
            "NetworkPolicy",
            "PodDisruptionBudget",
        ]

    def test_no_sandbox_renders_only_gpu(self, make_cluster, make_infra) -> None:
        infra = make_infra(
            kuberay=make_cluster().model_dump(),
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
        )
        manifests = render_manifests(_loaded(infra))
        assert _kinds(manifests) == ["RayJob"]

    def test_rayjob_without_entrypoint_raises(self, make_cluster, make_infra) -> None:
        infra = make_infra(kuberay=make_cluster().model_dump(), launch={"mode": "rayjob"})
        with pytest.raises(ValueError, match="entrypoint is required"):
            render_manifests(_loaded(infra))

    def test_rayjob_without_kuberay_raises(self, make_infra) -> None:
        infra = make_infra(launch={"mode": "rayjob", "entrypoint": "run.sh"})
        with pytest.raises(ValueError, match="kuberay is not defined"):
            render_manifests(_loaded(infra))


class TestSandboxHardening:
    def test_netpol_and_pdb_can_be_disabled(self, make_infra) -> None:
        sb = SandboxSpec.model_validate(
            {"networkPolicy": {"enabled": False}, "pdb": {"enabled": False}}
        )
        infra = make_infra(
            sandbox=sb.model_dump(), launch={"mode": "attach", "attach": "rc"}
        )
        assert _kinds(render_manifests(_loaded(infra))) == ["Deployment", "Service"]

    def test_bringup_gpu_pdb_emitted_when_enabled(self, make_cluster, make_infra) -> None:
        cluster = make_cluster()
        cluster_payload = cluster.model_dump()
        cluster_payload["pdb"] = {"enabled": True, "minAvailable": 3}
        infra = make_infra(kuberay=cluster_payload, launch={"mode": "bringup"})
        manifests = render_manifests(_loaded(infra))
        assert _kinds(manifests) == ["RayCluster", "PodDisruptionBudget"]
        pdb = manifests[-1]
        assert pdb["spec"]["selector"]["matchLabels"] == {
            "ray.io/cluster": "dockyard-grpo"
        }
        assert pdb["spec"]["minAvailable"] == 3

    def test_bringup_no_gpu_pdb_by_default(self, make_cluster, make_infra) -> None:
        infra = make_infra(kuberay=make_cluster().model_dump(), launch={"mode": "bringup"})
        assert _kinds(render_manifests(_loaded(infra))) == ["RayCluster"]


class TestNamespaceGovernance:
    def test_quota_and_limitrange_emitted_first(self, make_cluster, make_infra) -> None:
        infra = make_infra(
            kuberay=make_cluster().model_dump(),
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
            resourceQuota={"enabled": True, "hard": {"requests.cpu": "200"}},
            limitRange={"enabled": True, "default": {"cpu": "2", "memory": "4Gi"}},
        )
        kinds = _kinds(render_manifests(_loaded(infra)))
        assert kinds[:2] == ["ResourceQuota", "LimitRange"]
        assert kinds[-1] == "RayJob"

    def test_quota_spec_hard(self, make_infra) -> None:
        infra = make_infra(
            launch={"mode": "attach", "attach": "rc"},
            resourceQuota={"enabled": True, "hard": {"requests.nvidia.com/gpu": "32"}},
        )
        rq = [m for m in render_manifests(_loaded(infra)) if m["kind"] == "ResourceQuota"][0]
        assert rq["spec"]["hard"] == {"requests.nvidia.com/gpu": "32"}

    def test_limitrange_container_item(self, make_infra) -> None:
        infra = make_infra(
            launch={"mode": "attach", "attach": "rc"},
            limitRange={
                "enabled": True,
                "default": {"cpu": "2"},
                "max": {"cpu": "16", "memory": "64Gi"},
            },
        )
        lr = [m for m in render_manifests(_loaded(infra)) if m["kind"] == "LimitRange"][0]
        item = lr["spec"]["limits"][0]
        assert item["type"] == "Container"
        assert item["default"] == {"cpu": "2"}
        assert item["max"] == {"cpu": "16", "memory": "64Gi"}
        assert "min" not in item

    def test_disabled_by_default(self, make_infra) -> None:
        infra = make_infra(launch={"mode": "attach", "attach": "rc"})
        kinds = _kinds(render_manifests(_loaded(infra)))
        assert "ResourceQuota" not in kinds and "LimitRange" not in kinds


class TestEnvInjection:
    def test_ray_uv_flag_in_every_pod(self, make_cluster, make_infra) -> None:
        infra = make_infra(
            kuberay=make_cluster().model_dump(),
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
        )
        rj = render_manifests(_loaded(infra))[0]
        for pod in _pod_specs_of_rayjob(rj):
            assert _env_of(pod)["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"

    def test_sandbox_url_injected_when_pool_present(self, make_cluster, make_infra, sandbox) -> None:
        infra = make_infra(
            kuberay=make_cluster().model_dump(),
            sandbox=sandbox.model_dump(),
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
        )
        rj = [m for m in render_manifests(_loaded(infra)) if m["kind"] == "RayJob"][0]
        expected = "http://dockyard-sandbox.dockyard.svc.cluster.local:9090"
        for pod in _pod_specs_of_rayjob(rj):
            assert _env_of(pod)["DOCKYARD_SANDBOX_URLS"] == expected

    def test_sandbox_url_absent_without_pool(self, make_cluster, make_infra) -> None:
        infra = make_infra(
            kuberay=make_cluster().model_dump(),
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
        )
        rj = render_manifests(_loaded(infra))[0]
        for pod in _pod_specs_of_rayjob(rj):
            assert "DOCKYARD_SANDBOX_URLS" not in _env_of(pod)

    def test_inject_urls_false_skips_url(self, make_cluster, make_infra) -> None:
        sb = SandboxSpec(injectUrls=False)
        infra = make_infra(
            kuberay=make_cluster().model_dump(),
            sandbox=sb.model_dump(),
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
        )
        rj = [m for m in render_manifests(_loaded(infra)) if m["kind"] == "RayJob"][0]
        for pod in _pod_specs_of_rayjob(rj):
            assert "DOCKYARD_SANDBOX_URLS" not in _env_of(pod)

    def test_existing_env_value_not_overwritten(self, make_cluster, make_infra) -> None:
        # A worker that already pins RAY_ENABLE_UV_RUN_RUNTIME_ENV keeps its value.
        spec = make_cluster().spec
        spec["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]["env"].append(
            {"name": "RAY_ENABLE_UV_RUN_RUNTIME_ENV", "value": "1"}
        )
        infra = make_infra(
            kuberay={"name": "c", "spec": spec},
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
        )
        rj = render_manifests(_loaded(infra))[0]
        trainer = rj["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"]["spec"]
        assert _env_of(trainer)["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "1"
