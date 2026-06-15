"""Read-only introspection of the resources a recipe owns.

Used by ``dockyard-k8s status`` and ``dockyard-k8s logs`` to summarise what's
running and stream logs from it. Everything here is idempotent — no apply, no
delete, no job submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from kubernetes import client  # type: ignore[import-not-found]

from . import k8s
from .config import LoadedConfig
from .schema import InfraConfig


@dataclass
class RayClusterPods:
    head_name: str | None = None
    head_phase: str | None = None
    worker_names: list[str] = field(default_factory=list)
    worker_phases: list[str | None] = field(default_factory=list)


@dataclass
class ClusterStatus:
    name: str
    state: str  # "ready" | "—" | "(not found)" | other KubeRay states
    head_pod: str | None
    head_phase: str | None
    worker_phases: list[str | None]


@dataclass
class SandboxStatus:
    name: str
    replicas: int
    ready: int
    available: bool


def collect_cluster_status(loaded: LoadedConfig) -> ClusterStatus | None:
    """Build a :class:`ClusterStatus` for the GPU RayCluster, if declared."""
    infra = loaded.infra
    if infra.kuberay is None:
        return None
    return _status_for(infra.kuberay.name, infra)


def collect_sandbox_status(loaded: LoadedConfig) -> SandboxStatus | None:
    """Build a :class:`SandboxStatus` for the sandbox pool, if declared."""
    infra = loaded.infra
    if infra.sandbox is None:
        return None
    name = infra.sandbox.name
    dep = k8s.get_deployment(name, infra.namespace)
    if dep is None:
        return SandboxStatus(name=name, replicas=infra.sandbox.replicas, ready=0, available=False)
    spec_replicas = (dep.spec.replicas if dep.spec else None) or infra.sandbox.replicas
    ready = (dep.status.ready_replicas if dep.status else 0) or 0
    return SandboxStatus(
        name=name, replicas=spec_replicas, ready=ready, available=ready >= spec_replicas
    )


def _status_for(name: str, infra: InfraConfig) -> ClusterStatus:
    obj = k8s.get_raycluster(name, infra.namespace)
    state = (obj or {}).get("status", {}).get("state", "—") if obj else "(not found)"
    pods = list_cluster_pods(name, infra.namespace)
    return ClusterStatus(
        name=name,
        state=state,
        head_pod=pods.head_name,
        head_phase=pods.head_phase,
        worker_phases=pods.worker_phases,
    )


def list_cluster_pods(cluster_name: str, namespace: str) -> RayClusterPods:
    """Return the head and worker pods for a RayCluster."""
    k8s.load_kubeconfig()
    core: Any = client.CoreV1Api()
    out = RayClusterPods()
    for p in core.list_namespaced_pod(
        namespace=namespace, label_selector=f"ray.io/cluster={cluster_name}"
    ).items:
        kind = (p.metadata.labels or {}).get("ray.io/node-type", "")
        if kind == "head":
            out.head_name = p.metadata.name
            out.head_phase = p.status.phase if p.status else None
        else:
            out.worker_names.append(p.metadata.name)
            out.worker_phases.append(p.status.phase if p.status else None)
    return out


def stream_pod_logs(
    pod_name: str,
    namespace: str,
    *,
    container: str | None = None,
    follow: bool = False,
    tail_lines: int | None = 200,
) -> Iterator[str]:
    """Stream stdout/stderr from a specific pod."""
    k8s.load_kubeconfig()
    core: Any = client.CoreV1Api()
    stream = core.read_namespaced_pod_log(
        name=pod_name,
        namespace=namespace,
        container=container,
        follow=follow,
        tail_lines=tail_lines,
        _preload_content=False,
    )
    try:
        for raw in stream.stream():
            yield (
                raw.decode("utf-8", errors="replace")
                if isinstance(raw, (bytes, bytearray))
                else raw
            )
    finally:
        stream.release_conn()


def head_pod_name(cluster_name: str, namespace: str) -> str:
    """Return the head pod's name for a RayCluster, raising if missing."""
    pods = list_cluster_pods(cluster_name, namespace)
    if pods.head_name is None:
        raise RuntimeError(
            f"no head pod found for RayCluster {cluster_name} in {namespace}"
        )
    return pods.head_name


__all__ = [
    "ClusterStatus",
    "RayClusterPods",
    "SandboxStatus",
    "collect_cluster_status",
    "collect_sandbox_status",
    "head_pod_name",
    "list_cluster_pods",
    "stream_pod_logs",
]
