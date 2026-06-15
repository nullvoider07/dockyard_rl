"""Build K8s manifests from the infra config.

The GPU RayCluster spec lives under ``infra.kuberay.spec``; the sandbox
executor pool is described by ``infra.sandbox``. This module wraps each in
the standard ``apiVersion/kind/metadata`` envelope and patches cross-cutting
fields (image, imagePullSecrets, optional serviceAccountName, labels) from the
top-level ``infra`` block so they are not repeated per pod template.

The sandbox builders either patch an inline ``infra.sandbox.spec`` or, when
none is given, synthesize a complete Deployment for the ubuntu-swe task
executor (GPU-free, ``DOCKYARD_FLEET_ROLE=sandbox``, TCP readiness probe on
the executor port) plus a matching ClusterIP Service for stable DNS.
"""

from __future__ import annotations

import copy
from typing import Any

from .schema import ClusterSpec, InfraConfig, PDBSpec, SandboxSpec, SecuritySpec

# Every resource the CLI creates carries this label so admins can find
# orphans:  kubectl get rayclusters -l app.kubernetes.io/managed-by=dockyard-k8s
_MANAGED_BY_LABEL = {"app.kubernetes.io/managed-by": "dockyard-k8s"}


def build_raycluster_manifest(cluster: ClusterSpec, infra: InfraConfig) -> dict[str, Any]:
    """Build the full RayCluster dict for apply.

    ``infra`` supplies namespace, image, pull secrets and optional
    serviceAccount, which are patched into every container / pod template in
    the inline ``cluster.spec``.
    """
    spec = copy.deepcopy(cluster.spec)

    if cluster.segmentSize is not None:
        _expand_worker_segments(spec, cluster.segmentSize)

    _patch_images(spec, infra.image)
    _patch_image_pull_secrets(spec, list(infra.imagePullSecrets))
    if infra.serviceAccount is not None:
        _patch_service_account(spec, infra.serviceAccount)
    _patch_pod_labels(spec, {**_MANAGED_BY_LABEL, **infra.labels, **cluster.labels})
    if infra.priorityClassName is not None:
        _patch_priority_class(spec, infra.priorityClassName)
    # securityContext for the GPU fleets — opt-in (infra.security.enabled).
    pod_sc, container_sc = build_security_contexts(infra.security)
    _apply_security(spec, pod_sc, container_sc)
    # Rewrite DRA resourceClaimTemplateName references to cluster-unique names
    # the CLI also auto-creates. Skipped when autoCreate is off (pass-through).
    if infra.dra.autoCreate:
        _rewrite_dra_template_names(spec, cluster.name)

    metadata = _metadata(
        cluster.name,
        infra.namespace,
        labels={**_MANAGED_BY_LABEL, **infra.labels, **cluster.labels},
        annotations={**infra.annotations, **cluster.annotations},
    )
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayCluster",
        "metadata": metadata,
        "spec": spec,
    }


def _metadata(
    name: str,
    namespace: str,
    *,
    labels: dict[str, str],
    annotations: dict[str, str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": name, "namespace": namespace}
    if labels:
        metadata["labels"] = labels
    if annotations:
        metadata["annotations"] = annotations
    return metadata


def _expand_worker_segments(spec: dict, segment_size: int) -> None:
    """Split worker groups whose replicas exceed *segment_size*.

    Each qualifying group is replaced by ``replicas // segment_size`` identical
    copies named ``{groupName}-segment-{i}`` with replica counts set to
    *segment_size*. Mutates *spec* in place.
    """
    original_groups = spec.get("workerGroupSpecs") or []
    if not original_groups:
        return

    expanded: list[dict] = []
    for wg in original_groups:
        replicas = int(wg.get("replicas", 0))
        if replicas <= segment_size:
            expanded.append(wg)
            continue
        if replicas % segment_size != 0:
            group_name = wg.get("groupName", "<unnamed>")
            raise ValueError(
                f"workerGroup '{group_name}' has replicas={replicas} which is "
                f"not evenly divisible by segmentSize={segment_size}"
            )
        num_segments = replicas // segment_size
        base_name = wg.get("groupName", "workers")
        for i in range(num_segments):
            segment = copy.deepcopy(wg)
            segment["groupName"] = f"{base_name}-segment-{i}"
            segment["replicas"] = segment_size
            segment["minReplicas"] = segment_size
            segment["maxReplicas"] = segment_size
            expanded.append(segment)

    spec["workerGroupSpecs"] = expanded


def _walk_pod_templates(raycluster_spec: dict) -> list[dict]:
    """Return every PodSpec inside a RayCluster (head + all worker groups)."""
    specs: list[dict] = []
    head = raycluster_spec.get("headGroupSpec") or {}
    head_spec = head.get("template", {}).get("spec")
    if isinstance(head_spec, dict):
        specs.append(head_spec)
    for wg in raycluster_spec.get("workerGroupSpecs") or []:
        wg_spec = wg.get("template", {}).get("spec")
        if isinstance(wg_spec, dict):
            specs.append(wg_spec)
    return specs


def _patch_pod_labels(raycluster_spec: dict, labels: dict[str, str]) -> None:
    head = raycluster_spec.get("headGroupSpec") or {}
    templates = [head.get("template")]
    for wg in raycluster_spec.get("workerGroupSpecs") or []:
        templates.append(wg.get("template"))
    for tpl in templates:
        if not isinstance(tpl, dict):
            continue
        meta = tpl.setdefault("metadata", {})
        existing = meta.get("labels") or {}
        meta["labels"] = {**labels, **existing}


def _patch_images(raycluster_spec: dict, image: str) -> None:
    for pod_spec in _walk_pod_templates(raycluster_spec):
        for container in pod_spec.get("containers", []):
            container["image"] = image


def _patch_image_pull_secrets(raycluster_spec: dict, secrets: list[str]) -> None:
    if not secrets:
        return
    for pod_spec in _walk_pod_templates(raycluster_spec):
        # Fresh list per pod — sharing one object would alias across templates
        # (and emit YAML anchors a later mutation could corrupt).
        pod_spec["imagePullSecrets"] = [{"name": s} for s in secrets]


def _patch_service_account(raycluster_spec: dict, service_account: str) -> None:
    for pod_spec in _walk_pod_templates(raycluster_spec):
        pod_spec["serviceAccountName"] = service_account


def _patch_priority_class(raycluster_spec: dict, priority_class: str) -> None:
    """Set priorityClassName on every pod template, leaving an explicitly
    authored per-pod value untouched (a fleet can pin its own priority)."""
    for pod_spec in _walk_pod_templates(raycluster_spec):
        pod_spec.setdefault("priorityClassName", priority_class)


# Security context

def build_security_contexts(sec: SecuritySpec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(pod_securityContext, container_securityContext)`` for *sec*.

    Both are empty when ``sec.enabled`` is false. Pod-level carries identity
    (``runAs*`` / ``fsGroup``) and the seccomp profile; container-level carries
    the privilege / capability / root-filesystem hardening. ``runAsNonRoot`` is
    set on both so the kubelet enforces it per container.
    """
    pod_sc: dict[str, Any] = {}
    container_sc: dict[str, Any] = {}
    if not sec.enabled:
        return pod_sc, container_sc

    if sec.runAsNonRoot is not None:
        pod_sc["runAsNonRoot"] = sec.runAsNonRoot
        container_sc["runAsNonRoot"] = sec.runAsNonRoot
    if sec.runAsUser is not None:
        pod_sc["runAsUser"] = sec.runAsUser
    if sec.runAsGroup is not None:
        pod_sc["runAsGroup"] = sec.runAsGroup
    if sec.fsGroup is not None:
        pod_sc["fsGroup"] = sec.fsGroup
    if sec.seccompProfile is not None:
        pod_sc["seccompProfile"] = {"type": sec.seccompProfile}
        container_sc["seccompProfile"] = {"type": sec.seccompProfile}

    container_sc["allowPrivilegeEscalation"] = sec.allowPrivilegeEscalation
    caps: dict[str, list[str]] = {}
    if sec.dropCapabilities:
        caps["drop"] = list(sec.dropCapabilities)
    if sec.addCapabilities:
        caps["add"] = list(sec.addCapabilities)
    if caps:
        container_sc["capabilities"] = caps
    if sec.readOnlyRootFilesystem is not None:
        container_sc["readOnlyRootFilesystem"] = sec.readOnlyRootFilesystem

    return pod_sc, container_sc


def _apply_security(
    raycluster_spec: dict, pod_sc: dict[str, Any], container_sc: dict[str, Any]
) -> None:
    """Set pod + container securityContext on every template, never clobbering
    an existing one (an inline spec's own securityContext wins)."""
    if not pod_sc and not container_sc:
        return
    for pod_spec in _walk_pod_templates(raycluster_spec):
        if pod_sc:
            pod_spec.setdefault("securityContext", copy.deepcopy(pod_sc))
        if container_sc:
            for container in pod_spec.get("containers", []):
                container.setdefault("securityContext", copy.deepcopy(container_sc))


# Sandbox executor pool

def sandbox_service_dns(sandbox: SandboxSpec, infra: InfraConfig) -> str:
    """Return the cluster-internal URL the trainer uses to reach the pool."""
    return (
        f"http://{sandbox.name}.{infra.namespace}.svc.cluster.local:{sandbox.port}"
    )


def _sandbox_selector(sandbox: SandboxSpec) -> dict[str, str]:
    return {"app.kubernetes.io/name": sandbox.name}


def _pod_resources(res: Any) -> dict[str, Any]:
    """Translate PodResources into a K8s resources block.

    ``cpu`` / ``memory`` / ``ephemeralStorage`` are the requests; the optional
    ``*Limit`` fields raise the limit above the request for burst headroom
    (e.g. a heavy test run) without inflating the scheduler reservation. A limit
    defaults to its request when unset, so a config that sets only requests gets
    ``requests == limits`` as before.
    """
    requests: dict[str, str] = {}
    if res.cpu is not None:
        requests["cpu"] = str(res.cpu)
    if res.memory is not None:
        requests["memory"] = str(res.memory)
    if res.ephemeralStorage is not None:
        requests["ephemeral-storage"] = str(res.ephemeralStorage)
    if not requests:
        return {}
    limits = dict(requests)
    if res.cpuLimit is not None:
        limits["cpu"] = str(res.cpuLimit)
    if res.memoryLimit is not None:
        limits["memory"] = str(res.memoryLimit)
    if res.ephemeralStorageLimit is not None:
        limits["ephemeral-storage"] = str(res.ephemeralStorageLimit)
    return {"requests": requests, "limits": limits}


def build_sandbox_deployment(sandbox: SandboxSpec, infra: InfraConfig) -> dict[str, Any]:
    """Build the sandbox executor Deployment.

    When ``sandbox.spec`` is given it is patched (image / pullSecrets / SA /
    labels) like the RayCluster path. Otherwise a complete default Deployment
    for the ubuntu-swe task executor is synthesized.
    """
    selector = _sandbox_selector(sandbox)
    pod_labels = {**_MANAGED_BY_LABEL, **infra.labels, **sandbox.labels, **selector}

    if sandbox.spec is not None:
        spec = copy.deepcopy(sandbox.spec)
        template = spec.setdefault("template", {})
        pod_spec = template.setdefault("spec", {})
        for container in pod_spec.get("containers", []):
            container.setdefault("image", infra.image)
        if infra.imagePullSecrets:
            pod_spec["imagePullSecrets"] = [{"name": s} for s in infra.imagePullSecrets]
        if infra.serviceAccount is not None:
            pod_spec["serviceAccountName"] = infra.serviceAccount
        if infra.priorityClassName is not None:
            pod_spec.setdefault("priorityClassName", infra.priorityClassName)
        tmeta = template.setdefault("metadata", {})
        tmeta["labels"] = {**pod_labels, **(tmeta.get("labels") or {})}
        spec.setdefault("selector", {"matchLabels": selector})
        spec.setdefault("replicas", sandbox.replicas)
    else:
        env = [{"name": "DOCKYARD_FLEET_ROLE", "value": "sandbox"},
               {"name": "API_PORT", "value": str(sandbox.port)}]
        env += [{"name": k, "value": v} for k, v in sandbox.env.items()]
        container: dict[str, Any] = {
            "name": "task-executor",
            "image": infra.image,
            "ports": [{"containerPort": sandbox.port, "name": "http"}],
            "env": env,
            "readinessProbe": {
                "tcpSocket": {"port": sandbox.port},
                "initialDelaySeconds": 10,
                "periodSeconds": 10,
                "timeoutSeconds": 5,
                "failureThreshold": 6,
            },
            "livenessProbe": {
                "tcpSocket": {"port": sandbox.port},
                "initialDelaySeconds": 30,
                "periodSeconds": 20,
                "timeoutSeconds": 5,
                "failureThreshold": 3,
            },
        }
        resources = _pod_resources(sandbox.resources)
        if resources:
            container["resources"] = resources
        pod_sc, container_sc = build_security_contexts(sandbox.security)
        if container_sc:
            container["securityContext"] = container_sc
        pod_spec: dict[str, Any] = {"containers": [container]}
        if pod_sc:
            pod_spec["securityContext"] = pod_sc
        if infra.imagePullSecrets:
            pod_spec["imagePullSecrets"] = [{"name": s} for s in infra.imagePullSecrets]
        if infra.serviceAccount is not None:
            pod_spec["serviceAccountName"] = infra.serviceAccount
        if infra.priorityClassName is not None:
            pod_spec["priorityClassName"] = infra.priorityClassName
        if sandbox.topologySpreadConstraints:
            pod_spec["topologySpreadConstraints"] = copy.deepcopy(
                sandbox.topologySpreadConstraints
            )
        spec = {
            "replicas": sandbox.replicas,
            "selector": {"matchLabels": selector},
            "template": {"metadata": {"labels": pod_labels}, "spec": pod_spec},
        }

    metadata = _metadata(
        sandbox.name,
        infra.namespace,
        labels={**_MANAGED_BY_LABEL, **infra.labels, **sandbox.labels},
        annotations={**infra.annotations, **sandbox.annotations},
    )
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": metadata,
        "spec": spec,
    }


def build_sandbox_service(sandbox: SandboxSpec, infra: InfraConfig) -> dict[str, Any]:
    """Build the ClusterIP Service fronting the sandbox pool for stable DNS."""
    selector = _sandbox_selector(sandbox)
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": _metadata(
            sandbox.name,
            infra.namespace,
            labels={**_MANAGED_BY_LABEL, **infra.labels, **sandbox.labels},
            annotations={},
        ),
        "spec": {
            "type": "ClusterIP",
            "selector": selector,
            "ports": [
                {
                    "name": "http",
                    "port": sandbox.port,
                    "targetPort": sandbox.port,
                    "protocol": "TCP",
                }
            ],
        },
    }


# Network isolation + disruption budgets

def build_sandbox_network_policy(
    sandbox: SandboxSpec, infra: InfraConfig
) -> dict[str, Any]:
    """Build the sandbox NetworkPolicy.

    Ingress is restricted to the executor ``port``: from pods carrying
    ``networkPolicy.fromLabels`` if set, otherwise any pod in the namespace
    (so ``attach``-mode clusters without the managed-by label still reach it),
    denying cross-namespace and external ingress. Egress allows DNS always,
    then either all traffic (``allowEgress``) or only ``allowedEgressCidrs``.
    """
    np = sandbox.networkPolicy
    selector = _sandbox_selector(sandbox)

    if np.fromLabels:
        from_peers: list[dict[str, Any]] = [
            {"podSelector": {"matchLabels": dict(np.fromLabels)}}
        ]
    else:
        from_peers = [{"podSelector": {}}]
    ingress = [{"from": from_peers, "ports": [{"protocol": "TCP", "port": sandbox.port}]}]

    if np.allowEgress:
        egress: list[dict[str, Any]] = [{}]  # unrestricted egress
    else:
        # DNS to kube-dns is always required for repo hostname resolution.
        egress = [{"ports": [{"protocol": "UDP", "port": 53},
                             {"protocol": "TCP", "port": 53}]}]
        if np.allowedEgressCidrs:
            egress.append(
                {"to": [{"ipBlock": {"cidr": c}} for c in np.allowedEgressCidrs]}
            )

    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": _metadata(
            f"{sandbox.name}-netpol",
            infra.namespace,
            labels={**_MANAGED_BY_LABEL, **infra.labels, **sandbox.labels},
            annotations={},
        ),
        "spec": {
            "podSelector": {"matchLabels": selector},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": ingress,
            "egress": egress,
        },
    }


def build_pdb(
    name: str,
    namespace: str,
    selector_labels: dict[str, str],
    pdb: PDBSpec,
    infra: InfraConfig,
) -> dict[str, Any]:
    """Build a PodDisruptionBudget over pods matching *selector_labels*.

    Exactly one of ``minAvailable`` / ``maxUnavailable`` is set on the spec
    (enforced by :class:`PDBSpec`).
    """
    spec: dict[str, Any] = {"selector": {"matchLabels": dict(selector_labels)}}
    if pdb.minAvailable is not None:
        spec["minAvailable"] = pdb.minAvailable
    if pdb.maxUnavailable is not None:
        spec["maxUnavailable"] = pdb.maxUnavailable
    return {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": _metadata(
            name,
            namespace,
            labels={**_MANAGED_BY_LABEL, **infra.labels},
            annotations={},
        ),
        "spec": spec,
    }


# Namespace governance: ResourceQuota + LimitRange

def build_resource_quota(infra: InfraConfig) -> dict[str, Any]:
    """Build the namespace ResourceQuota from ``infra.resourceQuota.hard``."""
    rq = infra.resourceQuota
    return {
        "apiVersion": "v1",
        "kind": "ResourceQuota",
        "metadata": _metadata(
            rq.name,
            infra.namespace,
            labels={**_MANAGED_BY_LABEL, **infra.labels},
            annotations={},
        ),
        "spec": {"hard": dict(rq.hard)},
    }


def build_limit_range(infra: InfraConfig) -> dict[str, Any]:
    """Build a namespace LimitRange with a single Container limit item."""
    lr = infra.limitRange
    item: dict[str, Any] = {"type": "Container"}
    if lr.default:
        item["default"] = dict(lr.default)
    if lr.defaultRequest:
        item["defaultRequest"] = dict(lr.defaultRequest)
    if lr.max:
        item["max"] = dict(lr.max)
    if lr.min:
        item["min"] = dict(lr.min)
    return {
        "apiVersion": "v1",
        "kind": "LimitRange",
        "metadata": _metadata(
            lr.name,
            infra.namespace,
            labels={**_MANAGED_BY_LABEL, **infra.labels},
            annotations={},
        ),
        "spec": {"limits": [item]},
    }


# Dynamic Resource Allocation (DRA): ComputeDomain + RoCE ResourceClaimTemplate
#
# A worker pod opts in by declaring `resourceClaims` whose `name` is one of the
# well-known keys below. The CLI rewrites each claim's `resourceClaimTemplateName`
# to a cluster-unique value and auto-creates the matching CRD. Names are
# `{prefix}{cluster_name}` (single GPU cluster, so no role suffix).

_DRA_CLAIM_PREFIX: dict[str, str] = {
    "compute-domain-channel": "compute-domain-",
    "roce-channel": "roce-",
}


def _rewrite_dra_template_names(spec: dict, cluster_name: str) -> None:
    """Rewrite ``resourceClaimTemplateName`` in worker pods to cluster-unique names."""
    for wg in spec.get("workerGroupSpecs") or []:
        pod_spec = wg.get("template", {}).get("spec")
        if not isinstance(pod_spec, dict):
            continue
        for claim in pod_spec.get("resourceClaims") or []:
            prefix = _DRA_CLAIM_PREFIX.get(claim.get("name", ""))
            if prefix:
                claim["resourceClaimTemplateName"] = f"{prefix}{cluster_name}"


def dra_resources_for_cluster(cluster_name: str, spec: dict) -> list[tuple[str, str]]:
    """Return ``[(kind, name), ...]`` for the DRA resources a cluster spec needs.

    ``kind`` is ``"compute-domain"`` or ``"roce"``. Scans every worker pod
    template for the well-known claim names; each distinct claim yields one
    resource. Empty when no worker references DRA.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for wg in spec.get("workerGroupSpecs") or []:
        pod_spec = wg.get("template", {}).get("spec")
        if not isinstance(pod_spec, dict):
            continue
        for claim in pod_spec.get("resourceClaims") or []:
            claim_name = claim.get("name", "")
            prefix = _DRA_CLAIM_PREFIX.get(claim_name)
            if prefix and claim_name not in seen:
                seen.add(claim_name)
                kind = "compute-domain" if "compute-domain" in prefix else "roce"
                found.append((kind, f"{prefix}{cluster_name}"))
    return found


def build_compute_domain_manifest(
    name: str, namespace: str, *, num_nodes: int = 0
) -> dict[str, Any]:
    """ComputeDomain CR (the controller auto-creates its ResourceClaimTemplate)."""
    return {
        "apiVersion": "resource.nvidia.com/v1beta1",
        "kind": "ComputeDomain",
        "metadata": {"name": name, "namespace": namespace, "labels": dict(_MANAGED_BY_LABEL)},
        "spec": {
            "channel": {"resourceClaimTemplate": {"name": name}},
            "numNodes": num_nodes,
        },
    }


def build_roce_template_manifest(
    name: str, namespace: str, *, count: int = 8
) -> dict[str, Any]:
    """RoCE (RDMA) NIC ResourceClaimTemplate — `count` NICs per worker pod."""
    return {
        "apiVersion": "resource.k8s.io/v1",
        "kind": "ResourceClaimTemplate",
        "metadata": {"name": name, "namespace": namespace, "labels": dict(_MANAGED_BY_LABEL)},
        "spec": {
            "spec": {
                "devices": {
                    "requests": [
                        {
                            "exactly": {
                                "count": count,
                                "deviceClassName": "roce.networking.k8s.aws",
                            },
                            "name": "roce",
                        }
                    ],
                },
            },
        },
    }


__all__ = [
    "build_compute_domain_manifest",
    "build_limit_range",
    "build_pdb",
    "build_raycluster_manifest",
    "build_resource_quota",
    "build_roce_template_manifest",
    "build_sandbox_deployment",
    "build_sandbox_network_policy",
    "build_sandbox_service",
    "build_security_contexts",
    "dra_resources_for_cluster",
    "sandbox_service_dns",
]
