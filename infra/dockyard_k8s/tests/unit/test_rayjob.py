"""Tests for dockyard_k8s.rayjob — the RayJob envelope around a RayCluster."""

from __future__ import annotations

from src.dockyard_k8s.rayjob import (
    DEFAULT_SUBMISSION_MODE,
    DEFAULT_TTL_SECONDS,
    build_rayjob_manifest,
)


class TestRayJobManifest:
    def test_envelope(self, make_cluster, make_infra) -> None:
        rj = build_rayjob_manifest(make_cluster(), make_infra(), entrypoint="run.sh")
        assert rj["apiVersion"] == "ray.io/v1"
        assert rj["kind"] == "RayJob"
        assert rj["metadata"]["namespace"] == "dockyard"

    def test_defaults(self, make_cluster, make_infra) -> None:
        rj = build_rayjob_manifest(make_cluster(), make_infra(), entrypoint="run.sh")
        spec = rj["spec"]
        assert spec["entrypoint"] == "run.sh"
        assert spec["submissionMode"] == DEFAULT_SUBMISSION_MODE == "HTTPMode"
        assert spec["shutdownAfterJobFinishes"] is True
        assert spec["ttlSecondsAfterFinished"] == DEFAULT_TTL_SECONDS == 3600

    def test_name_defaults_to_cluster_name(self, make_cluster, make_infra) -> None:
        rj = build_rayjob_manifest(
            make_cluster(name="grpo-swe"), make_infra(), entrypoint="x"
        )
        assert rj["metadata"]["name"] == "grpo-swe"

    def test_name_override(self, make_cluster, make_infra) -> None:
        rj = build_rayjob_manifest(
            make_cluster(name="grpo-swe"), make_infra(), entrypoint="x", name="job-1"
        )
        assert rj["metadata"]["name"] == "job-1"

    def test_raycluster_spec_is_patched_body(self, make_cluster, make_infra) -> None:
        rj = build_rayjob_manifest(
            make_cluster(), make_infra(), entrypoint="x"
        )
        rcs = rj["spec"]["rayClusterSpec"]
        # image patched into the embedded cluster body
        assert (
            rcs["headGroupSpec"]["template"]["spec"]["containers"][0]["image"]
            == "nullvoider/ubuntu-swe:latest"
        )
        assert {g["groupName"] for g in rcs["workerGroupSpecs"]} == {"trainer", "inference"}

    def test_managed_by_and_extra_labels(self, make_cluster, make_infra) -> None:
        rj = build_rayjob_manifest(
            make_cluster(),
            make_infra(),
            entrypoint="x",
            extra_labels={"kai.scheduler/queue": "rl"},
        )
        labels = rj["metadata"]["labels"]
        assert labels["app.kubernetes.io/managed-by"] == "dockyard-k8s"
        assert labels["kai.scheduler/queue"] == "rl"
