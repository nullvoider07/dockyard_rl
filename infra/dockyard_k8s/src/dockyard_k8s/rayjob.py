"""Build KubeRay ``RayJob`` objects.

A RayJob is a RayCluster + Ray Job in one resource: KubeRay creates the
cluster, waits for it to be ready, submits ``spec.entrypoint`` over the
dashboard HTTP API, polls until terminal, then (optionally) tears the cluster
down. The convenient shape for one-shot runs that should not leave a
long-lived RayCluster behind.

The inline RayCluster body comes from the infra config — this module reuses
:func:`dockyard_k8s.manifest.build_raycluster_manifest`, then lifts its
``.spec`` under ``rayClusterSpec`` and wraps a RayJob envelope around it.
"""

from __future__ import annotations

from typing import Any

from .manifest import _MANAGED_BY_LABEL, build_raycluster_manifest
from .schema import ClusterSpec, InfraConfig

DEFAULT_SUBMISSION_MODE = "HTTPMode"
DEFAULT_TTL_SECONDS = 3600


def build_rayjob_manifest(
    cluster: ClusterSpec,
    infra: InfraConfig,
    *,
    entrypoint: str,
    name: str | None = None,
    shutdown_after_finishes: bool = True,
    ttl_seconds_after_finished: int = DEFAULT_TTL_SECONDS,
    submission_mode: str = DEFAULT_SUBMISSION_MODE,
    extra_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Wrap a RayCluster inline body in a RayJob envelope.

    ``entrypoint`` is the shell command KubeRay submits via HTTP to the
    dashboard (typically ``infra.launch.entrypoint``). ``submissionMode``
    stays ``HTTPMode``: ``K8sJobMode`` creates a separate K8s Job which breaks
    gang scheduling under KAI.
    """
    cluster_manifest = build_raycluster_manifest(cluster, infra)
    ray_cluster_spec = cluster_manifest["spec"]

    job_name = name or cluster.name
    merged_labels = {
        **_MANAGED_BY_LABEL,
        **infra.labels,
        **cluster.labels,
        **(extra_labels or {}),
    }
    metadata: dict[str, Any] = {"name": job_name, "namespace": infra.namespace}
    if merged_labels:
        metadata["labels"] = merged_labels
    merged_annotations = {**infra.annotations, **cluster.annotations}
    if merged_annotations:
        metadata["annotations"] = merged_annotations

    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": metadata,
        "spec": {
            "entrypoint": entrypoint,
            "submissionMode": submission_mode,
            "shutdownAfterJobFinishes": shutdown_after_finishes,
            "ttlSecondsAfterFinished": ttl_seconds_after_finished,
            "rayClusterSpec": ray_cluster_spec,
        },
    }


__all__ = [
    "DEFAULT_SUBMISSION_MODE",
    "DEFAULT_TTL_SECONDS",
    "build_rayjob_manifest",
]
