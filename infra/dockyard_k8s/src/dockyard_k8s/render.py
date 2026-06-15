"""Render the K8s manifests for a loaded recipe + infra config.

Produces the ordered list of objects to apply for one deployment:

  * the sandbox executor Deployment + ClusterIP Service (when ``infra.sandbox``
    is set), emitted first so the pool is reachable before training starts;
  * the GPU compute object, depending on ``infra.launch.mode`` — a RayJob
    (``rayjob``), a long-lived RayCluster (``bringup``), or nothing
    (``attach``, which submits onto an existing cluster — a submit-path
    concern, not a render-path one).

Cross-cutting env is injected into every GPU pod template (head + workers):
``RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`` (defence-in-depth against a runtime venv,
matching ``ray.sub``) and, when ``infra.sandbox.injectUrls`` is set,
``DOCKYARD_SANDBOX_URLS`` pointing at the sandbox Service DNS.
``init_ray()`` forwards the head's environment to every Ray actor, so the
``CodeEnvironment`` scoring workers see the URL — provided the recipe does not
hardcode ``env.code.sandbox_urls`` (pass ``env.code.sandbox_urls=null`` in the
launch entrypoint to let the env var win).
"""

from __future__ import annotations

from typing import Any

from .config import LoadedConfig
from .manifest import (
    build_compute_domain_manifest,
    build_limit_range,
    build_pdb,
    build_raycluster_manifest,
    build_resource_quota,
    build_roce_template_manifest,
    build_sandbox_deployment,
    build_sandbox_network_policy,
    build_sandbox_service,
    dra_resources_for_cluster,
    sandbox_service_dns,
)
from .rayjob import build_rayjob_manifest
from .schema import ClusterSpec, InfraConfig, LaunchMode


def render_manifests(loaded: LoadedConfig) -> list[dict[str, Any]]:
    """Return the ordered list of K8s objects for this deployment."""
    infra = loaded.infra
    manifests: list[dict[str, Any]] = []

    # Namespace governance first, so the quota/limits exist before any workload
    # pods are admitted.
    if infra.resourceQuota.enabled:
        manifests.append(build_resource_quota(infra))
    if infra.limitRange.enabled:
        manifests.append(build_limit_range(infra))

    cross_env: dict[str, str] = {"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"}
    if infra.sandbox is not None:
        sandbox = infra.sandbox
        manifests.append(build_sandbox_deployment(sandbox, infra))
        manifests.append(build_sandbox_service(sandbox, infra))
        if sandbox.networkPolicy.enabled:
            manifests.append(build_sandbox_network_policy(sandbox, infra))
        if sandbox.pdb.enabled:
            manifests.append(
                build_pdb(
                    f"{sandbox.name}-pdb",
                    infra.namespace,
                    {"app.kubernetes.io/name": sandbox.name},
                    sandbox.pdb,
                    infra,
                )
            )
        if sandbox.injectUrls:
            cross_env["DOCKYARD_SANDBOX_URLS"] = sandbox_service_dns(sandbox, infra)

    mode = infra.launch.mode
    if mode is LaunchMode.RAYJOB:
        cluster = _require_cluster(infra)
        if not infra.launch.entrypoint:
            raise ValueError("infra.launch.entrypoint is required when launch.mode=rayjob")
        # DRA CRDs must exist before the workers schedule — emit them first.
        manifests.extend(_dra_manifests(infra, cluster))
        rayjob = build_rayjob_manifest(
            cluster, infra, entrypoint=infra.launch.entrypoint
        )
        _inject_env_into_raycluster(rayjob["spec"]["rayClusterSpec"], cross_env)
        manifests.append(rayjob)
    elif mode is LaunchMode.BRINGUP:
        cluster = _require_cluster(infra)
        manifests.extend(_dra_manifests(infra, cluster))
        raycluster = build_raycluster_manifest(cluster, infra)
        _inject_env_into_raycluster(raycluster["spec"], cross_env)
        manifests.append(raycluster)
        # KubeRay labels every cluster pod with ray.io/cluster=<name>; in bringup
        # the name is known (= cluster.name), so a PDB can select the gang. (Not
        # emitted for rayjob mode, where KubeRay generates the cluster name.)
        if cluster.pdb.enabled:
            manifests.append(
                build_pdb(
                    f"{cluster.name}-pdb",
                    infra.namespace,
                    {"ray.io/cluster": cluster.name},
                    cluster.pdb,
                    infra,
                )
            )
    elif mode is LaunchMode.ATTACH:
        # No GPU object is rendered — training attaches to an existing cluster.
        pass

    return manifests


def _dra_manifests(infra: InfraConfig, cluster: ClusterSpec) -> list[dict[str, Any]]:
    """Build the ComputeDomain / RoCE CRDs the cluster's worker claims need."""
    if not infra.dra.autoCreate:
        return []
    out: list[dict[str, Any]] = []
    for kind, name in dra_resources_for_cluster(cluster.name, cluster.spec):
        if kind == "compute-domain":
            out.append(
                build_compute_domain_manifest(
                    name, infra.namespace, num_nodes=infra.dra.computeDomainNumNodes
                )
            )
        elif kind == "roce":
            out.append(
                build_roce_template_manifest(
                    name, infra.namespace, count=infra.dra.roceCount
                )
            )
    return out


def _require_cluster(infra) -> Any:
    if infra.kuberay is None:
        raise ValueError(
            "infra.kuberay is not defined — a GPU RayCluster spec is required "
            f"for launch.mode={infra.launch.mode.value}"
        )
    return infra.kuberay


def _inject_env_into_raycluster(raycluster_spec: dict, env: dict[str, str]) -> None:
    """Upsert env vars into every container of the head + worker pod templates."""
    head = raycluster_spec.get("headGroupSpec") or {}
    pod_specs = []
    head_spec = head.get("template", {}).get("spec")
    if isinstance(head_spec, dict):
        pod_specs.append(head_spec)
    for wg in raycluster_spec.get("workerGroupSpecs") or []:
        wg_spec = wg.get("template", {}).get("spec")
        if isinstance(wg_spec, dict):
            pod_specs.append(wg_spec)
    for pod_spec in pod_specs:
        for container in pod_spec.get("containers", []):
            _env_upsert(container, env)


def _env_upsert(container: dict, env: dict[str, str]) -> None:
    """Add env vars to a container, leaving any already-present names untouched."""
    existing = container.setdefault("env", [])
    present = {e.get("name") for e in existing if isinstance(e, dict)}
    for name, value in env.items():
        if name not in present:
            existing.append({"name": name, "value": value})


__all__ = ["render_manifests"]
