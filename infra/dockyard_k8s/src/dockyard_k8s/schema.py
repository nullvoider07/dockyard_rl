"""Pydantic schema for a dockyard_rl ``.infra.yaml``.

A run is described by two files: a recipe (the dockyard_rl GRPO config —
``policy`` / ``grpo`` / ``data`` / ``env`` / ...) and an infra config (this
schema — namespace, image, the GPU RayCluster shape, the sandbox executor
pool). The CLI merges shipped defaults, an optional user defaults file, the
infra file (or a recipe-level ``infra:`` block), and CLI overrides before
validating the result through :class:`InfraConfig`.

Strict validation (``extra='forbid'``) surfaces typos early. RayCluster
topology is intentionally not modelled: ``kuberay.spec`` is a free-form
RayCluster ``.spec`` dict so every upstream KubeRay field works without a
schema change. The CLI only patches cross-cutting fields (image,
imagePullSecrets, serviceAccount, labels) into it.

Topology note: trainer and inference are two worker groups inside a single
GPU Ray cluster (NCCL weight-sync requires them co-resident; the runtime
``dockyard_rl.cluster`` code carves them into per-fleet placement groups).
The sandbox is a separate, GPU-free, horizontally-scalable executor pool
(``POST /task/submit`` on port 9090) addressed over HTTP via
``DOCKYARD_SANDBOX_URLS`` — off the GPU compute path.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# Scheduling

class SchedulerKind(str, Enum):
    KAI = "kai"
    KUEUE = "kueue"
    DEFAULT = "default"


class SchedulerSpec(_StrictModel):
    kind: SchedulerKind = SchedulerKind.DEFAULT
    queue: str | None = None

    @model_validator(mode="after")
    def _queue_required_for_kai_kueue(self) -> "SchedulerSpec":
        if self.kind in (SchedulerKind.KAI, SchedulerKind.KUEUE) and not self.queue:
            raise ValueError(
                f"infra.scheduler.queue is required when scheduler.kind={self.kind.value}"
            )
        return self


# Placement

class Toleration(_StrictModel):
    key: str
    operator: Literal["Equal", "Exists"] = "Equal"
    value: str | None = None
    effect: Literal["NoSchedule", "PreferNoSchedule", "NoExecute"] = "NoSchedule"
    tolerationSeconds: int | None = None


class PlacementSpec(_StrictModel):
    nodeSelector: dict[str, str] = Field(default_factory=dict)
    tolerations: list[Toleration] = Field(default_factory=list)
    # Raw affinity passthrough; rarely needed (falls back to nodeSelector).
    affinity: dict[str, Any] | None = None


# Networking

class NetworkingSpec(_StrictModel):
    hostNetwork: bool = False
    gloo_socket_ifname: str | None = None
    nccl_socket_ifname: str | None = None
    nccl_ib_disable: bool = False
    nccl_net: Literal["Socket", "IB", "OFI"] | None = None
    # Additional NCCL / runtime env vars the cluster needs; user-managed.
    extra_env: dict[str, str] = Field(default_factory=dict)


# Storage (workspace, HF cache, checkpoints)

class WorkspaceKind(str, Enum):
    LUSTRE = "lustre"  # managed parallel-FS PVC
    PVC = "pvc"  # any RWX PVC
    HOST_PATH = "hostPath"  # dev / kind only
    RAY_UPLOAD = "rayUpload"  # Ray Job SDK working_dir upload (100 MiB cap)
    AUTO = "auto"  # prefer the PVC if it exists, else rayUpload


class WorkspaceSpec(_StrictModel):
    kind: WorkspaceKind = WorkspaceKind.RAY_UPLOAD
    pvcName: str | None = None
    mountPath: str = "/mnt/dockyard"
    repoSubdir: str = "workdirs"  # ${mountPath}/${repoSubdir}/<hash>/ holds the repo
    size: str | None = None  # consulted only if the PVC must be created (lustre)
    hostPath: str | None = None

    @model_validator(mode="after")
    def _required_fields_by_kind(self) -> "WorkspaceSpec":
        if self.kind in (WorkspaceKind.LUSTRE, WorkspaceKind.PVC) and not self.pvcName:
            raise ValueError(
                f"infra.workspace.pvcName is required when kind={self.kind.value}"
            )
        if self.kind is WorkspaceKind.HOST_PATH and not self.hostPath:
            raise ValueError("infra.workspace.hostPath is required when kind=hostPath")
        return self


class HFCacheKind(str, Enum):
    LUSTRE = "lustre"
    PVC = "pvc"
    EMPTY_DIR = "emptyDir"
    NONE = "none"


class HFCacheSpec(_StrictModel):
    kind: HFCacheKind = HFCacheKind.NONE
    pvcName: str | None = None
    mountPath: str = "/root/.cache/huggingface"

    @model_validator(mode="after")
    def _pvc_required(self) -> "HFCacheSpec":
        if self.kind in (HFCacheKind.LUSTRE, HFCacheKind.PVC) and not self.pvcName:
            raise ValueError(
                f"infra.hf_cache.pvcName is required when kind={self.kind.value}"
            )
        return self


class CheckpointsKind(str, Enum):
    LUSTRE = "lustre"
    PVC = "pvc"
    NONE = "none"  # checkpoints land on pod-local storage (smoke tests only)


class CheckpointsSpec(_StrictModel):
    kind: CheckpointsKind = CheckpointsKind.NONE
    pvcName: str | None = None
    mountPath: str = "/mnt/dockyard/checkpoints"

    @model_validator(mode="after")
    def _pvc_required(self) -> "CheckpointsSpec":
        if self.kind in (CheckpointsKind.LUSTRE, CheckpointsKind.PVC) and not self.pvcName:
            raise ValueError(
                f"infra.checkpoints.pvcName is required when kind={self.kind.value}"
            )
        return self


# Submission (how the CLI delivers a job onto the cluster)
#
# Carried in the schema for forward-compatibility with the submit/orchestrate
# CLI. The first-pass `check` / `render` commands do not consult these.

class SubmitKind(str, Enum):
    SDK = "sdk"  # Ray Job SDK (default)
    RAYJOB = "rayjob"  # RayJob CRD


class PortForwardMode(str, Enum):
    KUBECTL_RAY_PLUGIN = "kubectl-ray-plugin"
    KUBECTL_PORT_FORWARD = "kubectl-port-forward"
    AUTO = "auto"


class SubmitterMode(str, Enum):
    """How the CLI ships the training entrypoint onto the cluster.

    ``portForward`` goes through Ray's Job SDK via a brief
    ``kubectl port-forward`` to the head dashboard. ``exec`` shells into the
    head pod with ``kubectl exec`` and launches the entrypoint under
    ``nohup`` — the same shape as a Slurm driver on a login node.
    """

    PORT_FORWARD = "portForward"
    EXEC = "exec"


class SubmitSpec(_StrictModel):
    kind: SubmitKind = SubmitKind.SDK
    portForward: PortForwardMode = PortForwardMode.AUTO
    localDashboardPort: int = 18265
    submitter: SubmitterMode = SubmitterMode.PORT_FORWARD
    # Pod directory the exec submitter uses for launcher script + stdout + pidfile.
    execTmpDir: str = "/tmp"


# Launch

class LaunchMode(str, Enum):
    RAYJOB = "rayjob"  # ephemeral cluster per run, auto teardown (default)
    BRINGUP = "bringup"  # create a long-lived RayCluster, no job
    ATTACH = "attach"  # submit onto an existing RayCluster by name


class RunMode(str, Enum):
    """Interactive vs batch defaults for ``run`` / ``launch``.

    ``interactive`` preserves dev-iteration UX (port-forward submitter,
    ``codeSource=upload``, foreground log tailing). ``batch`` flips
    production defaults (exec submitter, ``codeSource=image``, no wait).
    Each dimension stays independently overridable.
    """

    INTERACTIVE = "interactive"
    BATCH = "batch"


class CodeSource(str, Enum):
    """Where the training code lives at submission time.

    ``upload`` stages a working_dir from the laptop and uploads it via Ray's
    Job SDK (100 MiB cap). ``image`` assumes the code is baked into the
    container at ``codePath``. ``lustre`` is structurally identical to
    ``image`` but implies ``codePath`` points at a shared-FS mount populated
    out of band.
    """

    UPLOAD = "upload"
    IMAGE = "image"
    LUSTRE = "lustre"


class LaunchSpec(_StrictModel):
    mode: LaunchMode = LaunchMode.RAYJOB
    runMode: RunMode = RunMode.INTERACTIVE
    # Name of the RayCluster to submit onto when mode=attach.
    attach: str | None = None
    # Shell command the training job runs inside the Ray cluster. Required for
    # `run` / RayJob rendering. The CLI stages the resolved recipe as
    # `dockyard_run.yaml` at the working_dir root (codeSource=upload only);
    # in image/lustre mode the entrypoint must reference the in-container path.
    entrypoint: str | None = None
    # Env vars injected into the training job's runtime_env.
    env: dict[str, str] = Field(default_factory=dict)
    # Repo-relative paths to stage into the Ray Job working_dir (upload mode).
    # None means "use the built-in default set" (see dockyard_k8s.workdir).
    rayUploadPaths: list[str] | None = None
    codeSource: CodeSource = CodeSource.UPLOAD
    codePath: str | None = None

    @model_validator(mode="after")
    def _attach_name_required(self) -> "LaunchSpec":
        if self.mode is LaunchMode.ATTACH and not self.attach:
            raise ValueError("infra.launch.attach is required when launch.mode=attach")
        return self

    @model_validator(mode="after")
    def _code_path_required_for_non_upload(self) -> "LaunchSpec":
        if self.codeSource in (CodeSource.IMAGE, CodeSource.LUSTRE) and not self.codePath:
            raise ValueError(
                f"infra.launch.codePath is required when codeSource={self.codeSource.value}"
            )
        return self


# Security context hardening
#
# The baseline (drop ALL Linux capabilities, allowPrivilegeEscalation=false,
# seccompProfile=RuntimeDefault) is safe for compute workloads and applied when
# `enabled`. `runAsNonRoot` / `readOnlyRootFilesystem` carry real breakage risk
# on the CUDA/NCCL image and the repo-cloning sandbox, so they are opt-in
# (None = leave unset) rather than defaulted on. The sandbox pool turns the
# baseline on by default (untrusted-code boundary); the GPU fleets get the same
# knob but default it off, since dropped capabilities on the GPU stack are
# unverified without hardware.

class SecuritySpec(_StrictModel):
    enabled: bool = True
    runAsNonRoot: bool | None = None
    runAsUser: int | None = None
    runAsGroup: int | None = None
    fsGroup: int | None = None
    readOnlyRootFilesystem: bool | None = None
    allowPrivilegeEscalation: bool = False
    dropCapabilities: list[str] = Field(default_factory=lambda: ["ALL"])
    addCapabilities: list[str] = Field(default_factory=list)
    seccompProfile: Literal["RuntimeDefault", "Unconfined"] | None = "RuntimeDefault"


# Network isolation for the sandbox pool
#
# Ingress is restricted to the executor port from pods in the same namespace
# (the GPU fleets reach it over the ClusterIP Service); cross-namespace and
# external ingress are denied. Egress stays open by default because the executor
# clones repositories over HTTPS — set `allowEgress=false` to restrict egress to
# `allowedEgressCidrs` (DNS to kube-dns is always permitted). Requires a CNI that
# enforces NetworkPolicy; on a non-enforcing CNI the object is an inert no-op.

class NetworkPolicySpec(_StrictModel):
    enabled: bool = True
    allowEgress: bool = True
    allowedEgressCidrs: list[str] = Field(default_factory=list)
    # Restrict ingress to pods carrying these labels; empty = any pod in the
    # namespace (keeps `attach`-mode external clusters working).
    fromLabels: dict[str, str] = Field(default_factory=dict)


# PodDisruptionBudget
#
# Protects a pool from voluntary disruption (node drain / upgrade). Exactly one
# of `minAvailable` / `maxUnavailable` must be set when enabled. The sandbox
# defaults to `maxUnavailable=1`; the GPU gang is opt-in (a too-strict PDB can
# wedge node maintenance).

class PDBSpec(_StrictModel):
    enabled: bool = False
    minAvailable: int | str | None = None
    maxUnavailable: int | str | None = None

    @model_validator(mode="after")
    def _exactly_one_when_enabled(self) -> "PDBSpec":
        if not self.enabled:
            return self
        set_count = (self.minAvailable is not None) + (self.maxUnavailable is not None)
        if set_count != 1:
            raise ValueError(
                "exactly one of pdb.minAvailable / pdb.maxUnavailable must be set "
                "when pdb.enabled"
            )
        return self


# Pod resources

class PodResources(_StrictModel):
    # cpu / memory / ephemeralStorage set the K8s *requests* (what the scheduler
    # reserves). Keep them modest for the sandbox pool: a clone-and-pytest
    # executor needs little at steady state, and large requests × replicas leave
    # pods Pending. The *Limit fields set the burst ceiling; each defaults to its
    # request when unset (requests == limits, Guaranteed QoS).
    cpu: str | None = None  # request, e.g. "1" or "500m"
    memory: str | None = None  # request, e.g. "2Gi"
    gpu: int | None = None  # nvidia.com/gpu; omitted for the sandbox pool
    # ephemeral-storage request: the sandbox clones repos to pod-local disk; a
    # request guards against node disk-pressure eviction.
    ephemeralStorage: str | None = None
    cpuLimit: str | None = None
    memoryLimit: str | None = None
    ephemeralStorageLimit: str | None = None


# GPU Ray cluster
#
# Topology is not modelled: `spec` is the inline RayCluster `.spec` body. The
# CLI wraps it in apiVersion/kind/metadata and patches image, imagePullSecrets,
# serviceAccount and labels from the top-level infra block. Trainer and
# inference live here as separate workerGroupSpecs, each setting its own
# DOCKYARD_FLEET_ROLE and `dockyard_fleet_<role>` Ray resource tag.

class ClusterSpec(_StrictModel):
    name: str
    spec: dict[str, Any]
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    # When set, each worker group with replicas > segmentSize is split into
    # replicas/segmentSize identical groups so a topology-aware scheduler
    # (e.g. KAI) can place each segment on one rack. Smaller values schedule
    # more easily. replicas must be evenly divisible by segmentSize.
    segmentSize: int | None = Field(default=None, gt=0)
    # PodDisruptionBudget over the cluster's pods (selected by ray.io/cluster).
    # Opt-in; honoured for bringup mode where the RayCluster name is known.
    pdb: PDBSpec = Field(default_factory=PDBSpec)


# Sandbox executor pool
#
# A GPU-free Deployment of ubuntu-swe task executors behind a ClusterIP
# Service. Reached over HTTP at `http://<name>.<namespace>.svc:<port>` and
# exposed to the trainer via DOCKYARD_SANDBOX_URLS. Scaled independently of
# the GPU cluster; never co-scheduled into the GPU compute path.

class SandboxSpec(_StrictModel):
    name: str = "dockyard-sandbox"
    replicas: int = Field(default=2, ge=1)
    port: int = 9090
    # Optional inline Deployment `.spec` override. When None the CLI builds a
    # default Deployment from `replicas`, `port`, `resources`, `env` and the
    # top-level image (DOCKYARD_FLEET_ROLE=sandbox, TCP readiness probe on
    # `port`). Provide this only for cluster-specific security contexts or
    # volume mounts.
    spec: dict[str, Any] | None = None
    resources: PodResources = Field(default_factory=PodResources)
    env: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    # Security/isolation defaults are on: the sandbox runs untrusted agent code.
    # They apply only to the synthesized default Deployment; an inline `spec` is
    # treated as a full passthrough (author its own securityContext there).
    security: SecuritySpec = Field(default_factory=SecuritySpec)
    networkPolicy: NetworkPolicySpec = Field(default_factory=NetworkPolicySpec)
    pdb: PDBSpec = Field(default_factory=lambda: PDBSpec(enabled=True, maxUnavailable=1))
    # Raw topologySpreadConstraints applied to the synthesized Deployment's pods
    # (e.g. spread replicas across zones/nodes). Free-form passthrough.
    topologySpreadConstraints: list[dict[str, Any]] = Field(default_factory=list)
    # Seconds to wait for the Deployment to be Ready before submitting training.
    healthCheckTimeoutS: int = 300
    # When true, the trainer cluster's DOCKYARD_SANDBOX_URLS is auto-populated
    # from this pool's Service DNS at render time.
    injectUrls: bool = True

    @model_validator(mode="after")
    def _reject_gpu(self) -> "SandboxSpec":
        if self.resources.gpu:
            raise ValueError(
                "infra.sandbox.resources.gpu must be unset — the sandbox pool "
                "is GPU-free and runs off the GPU compute path"
            )
        return self


# Namespace governance (opt-in templates)
#
# Rendered into the manifest list so `render | kubectl apply` installs them
# alongside the workload. Not auto-applied by the live `run` / `cluster` path —
# namespace-wide quota is cluster-admin territory, and a too-tight quota applied
# mid-run would block the very pods being deployed.

class ResourceQuotaSpec(_StrictModel):
    enabled: bool = False
    name: str = "dockyard-quota"
    # ResourceQuota `.spec.hard`, e.g. {"requests.cpu": "200",
    # "requests.memory": "1Ti", "requests.nvidia.com/gpu": "32",
    # "count/pods": "200"}.
    hard: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _hard_required_when_enabled(self) -> "ResourceQuotaSpec":
        if self.enabled and not self.hard:
            raise ValueError("resourceQuota.hard must be set when resourceQuota.enabled")
        return self


class LimitRangeSpec(_StrictModel):
    enabled: bool = False
    name: str = "dockyard-limits"
    # Per-Container default / defaultRequest / max / min maps, e.g.
    # default={"cpu": "2", "memory": "4Gi"}, max={"cpu": "16", "memory": "64Gi"}.
    default: dict[str, str] = Field(default_factory=dict)
    defaultRequest: dict[str, str] = Field(default_factory=dict)
    max: dict[str, str] = Field(default_factory=dict)
    min: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _some_limit_when_enabled(self) -> "LimitRangeSpec":
        if self.enabled and not (self.default or self.defaultRequest or self.max or self.min):
            raise ValueError(
                "limitRange needs at least one of default/defaultRequest/max/min "
                "when enabled"
            )
        return self


# Dynamic Resource Allocation (DRA) — optional GB300/RoCE auto-provisioning
#
# When a worker pod's `resourceClaims` reference the well-known claim names
# (`compute-domain-channel`, `roce-channel`), the CLI rewrites each
# `resourceClaimTemplateName` to a cluster-unique value and auto-creates the
# matching ComputeDomain (NVLink/MNNVL channel) + RoCE ResourceClaimTemplate,
# deleting them on teardown. The claims themselves are authored in the
# free-form RayCluster `spec`; this block only tunes the auto-created CRDs.
# Set `autoCreate: false` to leave `resourceClaims` untouched and manage the
# ComputeDomain / ResourceClaimTemplate yourself.

class DRASpec(_StrictModel):
    autoCreate: bool = True
    # RoCE (RDMA) NICs requested per worker pod (e.g. 8 on a GB300/NVL72 node).
    roceCount: int = Field(default=8, ge=1)
    # ComputeDomain node count; 0 = elastic mode (recommended).
    computeDomainNumNodes: int = Field(default=0, ge=0)


# Top-level InfraConfig

class InfraConfig(_StrictModel):
    namespace: str
    image: str
    imagePullSecrets: list[str] = Field(default_factory=list)
    rayVersion: str | None = None
    serviceAccount: str | None = None
    # Cross-cutting PriorityClass applied to every managed pod (GPU + sandbox).
    # Set per-pod in `kuberay.spec` to override a fleet (e.g. trainer high,
    # inference lower). Used by a preempting scheduler to protect the gang.
    priorityClassName: str | None = None

    scheduler: SchedulerSpec = Field(default_factory=SchedulerSpec)
    placement: PlacementSpec = Field(default_factory=PlacementSpec)
    networking: NetworkingSpec = Field(default_factory=NetworkingSpec)
    # securityContext applied to GPU head + worker pods. Defaults off: dropped
    # capabilities / non-root on the CUDA/NCCL stack are unverified without a
    # GPU, so it is an explicit opt-in (`infra.security.enabled=true`).
    security: SecuritySpec = Field(default_factory=lambda: SecuritySpec(enabled=False))
    workspace: WorkspaceSpec = Field(default_factory=WorkspaceSpec)
    hf_cache: HFCacheSpec = Field(default_factory=HFCacheSpec)
    checkpoints: CheckpointsSpec = Field(default_factory=CheckpointsSpec)
    submit: SubmitSpec = Field(default_factory=SubmitSpec)
    launch: LaunchSpec = Field(default_factory=LaunchSpec)
    # The single GPU Ray cluster (trainer + inference worker groups).
    kuberay: ClusterSpec | None = None
    # The sandbox executor pool. None means an external sandbox: provide
    # DOCKYARD_SANDBOX_URLS via launch.env (or env.code.sandbox_urls in the recipe).
    sandbox: SandboxSpec | None = None
    # DRA auto-provisioning for ComputeDomain / RoCE worker resourceClaims.
    dra: DRASpec = Field(default_factory=DRASpec)
    # Opt-in namespace governance templates (render-only).
    resourceQuota: ResourceQuotaSpec = Field(default_factory=ResourceQuotaSpec)
    limitRange: LimitRangeSpec = Field(default_factory=LimitRangeSpec)

    # Opaque extra labels / annotations the platform may require.
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)

    @field_validator("namespace", "image")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


__all__ = [
    "CheckpointsKind",
    "CheckpointsSpec",
    "ClusterSpec",
    "CodeSource",
    "DRASpec",
    "HFCacheKind",
    "HFCacheSpec",
    "InfraConfig",
    "LaunchMode",
    "LaunchSpec",
    "LimitRangeSpec",
    "NetworkPolicySpec",
    "NetworkingSpec",
    "PDBSpec",
    "ResourceQuotaSpec",
    "PlacementSpec",
    "PodResources",
    "PortForwardMode",
    "RunMode",
    "SandboxSpec",
    "SecuritySpec",
    "SchedulerKind",
    "SchedulerSpec",
    "SubmitKind",
    "SubmitSpec",
    "SubmitterMode",
    "Toleration",
    "WorkspaceKind",
    "WorkspaceSpec",
]
