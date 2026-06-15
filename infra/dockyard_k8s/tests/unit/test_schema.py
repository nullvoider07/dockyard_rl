"""Tests for dockyard_k8s.schema.

The schema is the contract between the infra YAML and every builder, so these
pin the validation rules: required top-level fields, strict rejection of
typos, the ``kind`` sentinels that need a companion field, the launch-mode
validators, and the sandbox pool's GPU-free invariant.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dockyard_k8s.schema import (
    CheckpointsSpec,
    ClusterSpec,
    CodeSource,
    HFCacheSpec,
    InfraConfig,
    LaunchMode,
    LaunchSpec,
    PodResources,
    RunMode,
    SandboxSpec,
    SchedulerKind,
    SchedulerSpec,
    WorkspaceKind,
    WorkspaceSpec,
)


def _min_infra() -> dict:
    return {"namespace": "dockyard", "image": "nullvoider/ubuntu-swe:latest"}


class TestInfraConfigRequiredFields:
    def test_minimal_config_validates(self) -> None:
        cfg = InfraConfig.model_validate(_min_infra())
        assert cfg.namespace == "dockyard"
        assert cfg.image == "nullvoider/ubuntu-swe:latest"
        assert cfg.scheduler.kind is SchedulerKind.DEFAULT
        assert cfg.launch.mode is LaunchMode.RAYJOB
        assert cfg.kuberay is None
        assert cfg.sandbox is None

    def test_missing_namespace_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate({"image": "foo:bar"})

    def test_missing_image_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate({"namespace": "ns"})

    def test_blank_namespace_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate({"namespace": "   ", "image": "foo:bar"})

    def test_blank_image_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate({"namespace": "ns", "image": "  "})


class TestStrictness:
    def test_unknown_top_level_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate(_min_infra() | {"totally_fake": True})

    def test_unknown_nested_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate(
                _min_infra() | {"scheduler": {"kind": "default", "typo": "x"}}
            )


class TestSchedulerSpec:
    def test_default_needs_no_queue(self) -> None:
        spec = SchedulerSpec()
        assert spec.kind is SchedulerKind.DEFAULT
        assert spec.queue is None

    @pytest.mark.parametrize("kind", ["kai", "kueue"])
    def test_queue_required(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            SchedulerSpec.model_validate({"kind": kind})

    def test_kai_with_queue_accepted(self) -> None:
        spec = SchedulerSpec.model_validate({"kind": "kai", "queue": "rl"})
        assert spec.queue == "rl"

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SchedulerSpec.model_validate({"kind": "slurm"})


class TestStorageKinds:
    def test_workspace_default_is_ray_upload(self) -> None:
        assert WorkspaceSpec().kind is WorkspaceKind.RAY_UPLOAD
        assert WorkspaceSpec().mountPath == "/mnt/dockyard"

    @pytest.mark.parametrize("kind", ["lustre", "pvc"])
    def test_workspace_pvc_kinds_need_name(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            WorkspaceSpec.model_validate({"kind": kind})

    def test_workspace_host_path_needs_path(self) -> None:
        with pytest.raises(ValidationError):
            WorkspaceSpec.model_validate({"kind": "hostPath"})

    def test_workspace_host_path_ok(self) -> None:
        spec = WorkspaceSpec.model_validate({"kind": "hostPath", "hostPath": "/data"})
        assert spec.hostPath == "/data"

    def test_hf_cache_pvc_needs_name(self) -> None:
        with pytest.raises(ValidationError):
            HFCacheSpec.model_validate({"kind": "pvc"})

    def test_checkpoints_default_mountpath(self) -> None:
        assert CheckpointsSpec().mountPath == "/mnt/dockyard/checkpoints"

    def test_checkpoints_lustre_needs_pvc(self) -> None:
        with pytest.raises(ValidationError):
            CheckpointsSpec.model_validate({"kind": "lustre"})


class TestLaunchSpec:
    def test_defaults(self) -> None:
        spec = LaunchSpec()
        assert spec.mode is LaunchMode.RAYJOB
        assert spec.runMode is RunMode.INTERACTIVE
        assert spec.codeSource is CodeSource.UPLOAD
        assert spec.codePath is None

    def test_attach_requires_target(self) -> None:
        with pytest.raises(ValidationError, match="attach is required"):
            LaunchSpec.model_validate({"mode": "attach"})

    def test_attach_with_name_ok(self) -> None:
        spec = LaunchSpec.model_validate({"mode": "attach", "attach": "rc-gpu"})
        assert spec.attach == "rc-gpu"

    @pytest.mark.parametrize("src", ["image", "lustre"])
    def test_code_path_required_for_non_upload(self, src: str) -> None:
        with pytest.raises(ValidationError, match="codePath is required"):
            LaunchSpec.model_validate({"codeSource": src})

    def test_code_path_ok_with_image(self) -> None:
        spec = LaunchSpec.model_validate(
            {"codeSource": "image", "codePath": "/opt/dockyard_rl"}
        )
        assert spec.codePath == "/opt/dockyard_rl"

    def test_code_path_not_required_for_upload(self) -> None:
        assert LaunchSpec.model_validate({"codeSource": "upload"}).codePath is None


class TestClusterSpecSegmentSize:
    def test_defaults_to_none(self) -> None:
        assert ClusterSpec.model_validate({"name": "x", "spec": {}}).segmentSize is None

    def test_positive_accepted(self) -> None:
        cs = ClusterSpec.model_validate({"name": "x", "spec": {}, "segmentSize": 8})
        assert cs.segmentSize == 8

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_rejected(self, bad: int) -> None:
        with pytest.raises(ValidationError):
            ClusterSpec.model_validate({"name": "x", "spec": {}, "segmentSize": bad})


class TestSandboxSpec:
    def test_defaults(self) -> None:
        sb = SandboxSpec()
        assert sb.name == "dockyard-sandbox"
        assert sb.replicas == 2
        assert sb.port == 9090
        assert sb.injectUrls is True
        assert sb.spec is None

    def test_replicas_floor(self) -> None:
        with pytest.raises(ValidationError):
            SandboxSpec.model_validate({"replicas": 0})

    def test_rejects_gpu_request(self) -> None:
        with pytest.raises(ValidationError, match="GPU-free"):
            SandboxSpec.model_validate({"resources": {"cpu": "4", "gpu": 1}})

    def test_cpu_memory_ok(self) -> None:
        sb = SandboxSpec.model_validate({"resources": {"cpu": "8", "memory": "32Gi"}})
        assert sb.resources == PodResources(cpu="8", memory="32Gi")
