"""Tests for dockyard_k8s.dev — dev pod manifest builder."""

from __future__ import annotations

from src.dockyard_k8s.dev import DEFAULT_IMAGE, build_dev_pod_manifest


class TestDevPodManifest:
    def test_names_and_labels(self) -> None:
        m = build_dev_pod_manifest("tester", "dockyard")
        assert m["kind"] == "Pod"
        assert m["metadata"]["name"] == "tester-dev-pod"
        assert m["metadata"]["namespace"] == "dockyard"
        labels = m["metadata"]["labels"]
        assert labels["app.kubernetes.io/managed-by"] == "dockyard-k8s"
        assert labels["dockyard-k8s/owner"] == "tester"

    def test_workspace_mount_and_image(self) -> None:
        m = build_dev_pod_manifest("tester", "dockyard", image="custom:tag")
        spec = m["spec"]
        container = spec["containers"][0]
        assert container["image"] == "custom:tag"
        assert container["workingDir"] == "/mnt/dockyard/tester"
        mount = container["volumeMounts"][0]
        assert mount["mountPath"] == "/mnt/dockyard"
        vol = spec["volumes"][0]
        assert vol["persistentVolumeClaim"]["claimName"] == "dockyard-workspace"

    def test_default_image(self) -> None:
        m = build_dev_pod_manifest("tester", "dockyard")
        assert m["spec"]["containers"][0]["image"] == DEFAULT_IMAGE

    def test_secret_envfrom_optional(self) -> None:
        m = build_dev_pod_manifest("tester", "dockyard")
        envfrom = m["spec"]["containers"][0]["envFrom"][0]
        assert envfrom["secretRef"]["name"] == "tester-secrets"
        assert envfrom["secretRef"]["optional"] is True
