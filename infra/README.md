# Kubernetes infrastructure for dockyard_rl

> [!WARNING]
> These instructions are in active development and should not be relied on as
> stable yet. Manifests, tooling, and APIs may change without notice.

Deploys the Project Dockyard async-GRPO stack on Kubernetes: one GPU RayCluster
(trainer + inference worker groups) and a GPU-free sandbox executor pool. The
`kind` + `helm` setup here gives you a local GPU playground; the same manifests
carry to a production cluster with adaptation (work with your cluster operator).

The helmfile is for convenience. A production system should manage these
components with Terraform or another infrastructure-as-code tool.

## Topology

Three fleets, each declaring `DOCKYARD_FLEET_ROLE`:

- **trainer** (GPU) — DTensor policy workers; GRPO optimizer.
- **inference** (GPU) — vLLM async engine; weights synced from trainer via NCCL.
- **sandbox** (CPU-only) — `ubuntu-swe` task executors that clone a repo, apply
  the agent patch + gold tests, run them, and return a verdict.

Trainer and inference are **two worker groups inside one GPU Ray cluster** — the
trainer→inference NCCL weight sync needs them co-resident. The `dockyard_rl`
runtime carves them into per-fleet placement groups by the
`dockyard_fleet_<role>` Ray resource tag (each worker group sets the tag in its
`ray start --resources` and exports `DOCKYARD_FLEET_ROLE`). There is no
trainer-vs-inference cluster split.

The **only** disaggregation is GPU Ray cluster vs sandbox. The sandbox is a
separate Deployment + Service of executors reached over HTTP via
`DOCKYARD_SANDBOX_URLS`, off the GPU compute path. Within a deployment, one
`ubuntu-swe` image is used by every pod — "CPU-only" describes only the
sandbox's resource request.

### Two inference images (vLLM / SGLang)

There are **two `ubuntu-swe` image builds**, one per inference backend — they
pin conflicting `torch`/`flashinfer` stacks so they cannot coexist in one
image:

| Backend | Image | Dockerfile |
|---------|-------|------------|
| vLLM (default) | `nullvoider/ubuntu-swe:latest` | `ubuntu-base/ubuntu-swe-v2.dockerfile` |
| SGLang | `nullvoider/ubuntu-swe-sglang:latest` | `ubuntu-base/ubuntu-swe-sglang.dockerfile` |

The GPU cluster image **must match** the recipe's
`policy.generation.backend`. Pick the matching pair:

- vLLM → `examples/configs/grpo_swe.yaml` + `grpo_swe.infra.yaml`
- SGLang → `examples/configs/grpo_swe_sglang.yaml` + `grpo_swe_sglang.infra.yaml`

For the raw `examples/*.yaml` manifests, set every `image:` to the build that
matches your backend. The sandbox runs no inference engine, so either build
works there; matching the GPU fleets keeps one image per deployment.

```
┌──────────────────────────────────────────────┐      ┌─────────────────────┐
│            GPU Ray cluster                     │      │   Sandbox pool      │
│                                                │      │  (Deployment+Svc)   │
│  ┌────────────────┐   ┌────────────────────┐  │ HTTP │  ┌───────────────┐  │
│  │ trainer group  │   │ inference group    │  │─────▶│  │ task-executor │  │
│  │ (DTensor, GPU) │◀─▶│ (vLLM, GPU)        │  │ 9090 │  │ × replicas    │  │
│  └────────────────┘NCCL└───────────────────┘  │      │  └───────────────┘  │
└──────────────────────────────────────────────┘      └─────────────────────┘
   DOCKYARD_SANDBOX_URLS=http://dockyard-sandbox:9090
```

## Prerequisites

- Docker with the systemd cgroup driver and **cgroup v2**
- NVIDIA driver on the host (`nvidia-smi` works)
- `nvidia-container-toolkit` on the host
- `go` (for the nvkind install)
- `helmfile` (`curl -sSL https://github.com/helmfile/helmfile/releases/latest/download/helmfile_$(uname -s | tr '[:upper:]' '[:lower:]')_amd64.tar.gz | tar xz -C ~/bin helmfile`)

### One-time host setup (requires sudo)

```sh
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
sudo systemctl restart docker

docker info | grep "Default Runtime"   # should show "nvidia"
stat -fc %T /sys/fs/cgroup/            # should show "cgroup2fs"
```

## Quick start (local kind cluster)

```sh
# 1. Install tools
cd kind
bash install-nvkind.sh
bash get-kubectl.sh
bash get-helm.sh

# 2. Create the cluster (all host GPUs exposed to a worker node)
bash create-cluster.sh

# 3. Deploy infrastructure (KAI scheduler, KubeRay operator, JobSet controller)
cd ../helm
helmfile -e kind sync

# 4. Create the KAI scheduler queues
kubectl apply -f ../examples/kai-queue.yaml

# 5. Deploy the sandbox pool, then a GPU workload (pick one)
kubectl apply -f ../examples/sandbox-pool.yaml
kubectl apply -f ../examples/rayjob-monolithic.yaml   # RayJob via KubeRay
# or
kubectl apply -f ../examples/jobset-monolithic.yaml   # same topology via JobSet
```

Watch progress:

```sh
kubectl get rayjobs -w
kubectl get jobsets.jobset.x-k8s.io
kubectl get pods -o wide
kubectl logs <pod-name>
```

## Generated manifests

For non-trivial topologies, generate the manifests from a recipe + infra pair
with the `dockyard-k8s` CLI instead of hand-editing YAML:

```sh
pip install ./dockyard_k8s
dockyard-k8s render ../examples/configs/grpo_swe.yaml \
  --infra dockyard_k8s/examples/grpo_swe.infra.yaml | kubectl apply -f -
```

See `dockyard_k8s/README.md`. The raw `examples/` manifests are the
hand-written equivalents for direct `kubectl apply`.

## Deploy on a real cluster

```sh
cd helm
helmfile -e prod sync
kubectl apply -f ../examples/kai-queue.yaml   # adapt queues to your cluster
```

This installs KAI scheduler, KubeRay, and JobSet. The cluster is expected to
already provide GPUs (GPU Operator or equivalent). Edit `kai-queue.yaml` to
match your GPU count and queue hierarchy first.

## RayJob vs JobSet

Both example workloads express the same one-cluster topology; they differ in
the controller that manages it.

| | RayJob (`rayjob-monolithic.yaml`) | JobSet (`jobset-monolithic.yaml`) |
|---|---|---|
| Controller | KubeRay operator | JobSet controller |
| Lifecycle | KubeRay creates cluster, submits, tears down | raw Jobs; driver job gates success |
| Failure cascading | KubeRay built-in | native `failurePolicy: FailJobSet` |
| Gang scheduling | single PodGroup | single PodGroup |
| Head discovery | KubeRay Service | predictable JobSet DNS |
| Startup ordering | KubeRay built-in | init containers (`ray health-check` poll) |
| KubeRay required | yes | no |

> [!NOTE]
> **Why init containers, not `dependsOn`?** KAI gang-schedules every pod in a
> JobSet as one PodGroup. `dependsOn` would block worker creation until the
> head is Ready, which deadlocks against gang scheduling (KAI waits for all
> pods to exist; JobSet waits for the head). Workers instead poll the head with
> `ray health-check` from an init container.

## File layout

```
infra/
├── dockyard_k8s/                     # Manifest generator CLI (pip install)
│   ├── src/dockyard_k8s/             # schema, config loader, manifest/render, CLI
│   ├── examples/grpo_swe.infra.yaml         # Infra spec (vLLM image)
│   ├── examples/grpo_swe_sglang.infra.yaml  # Infra spec (SGLang image)
│   └── README.md
├── kind/                             # Local-dev cluster setup
│   ├── create-cluster.sh             # Creates an nvkind cluster (name: dockyard)
│   ├── install-nvkind.sh             # Installs kind + nvkind
│   ├── get-kubectl.sh / get-helm.sh  # Tool installers
│   ├── nvkind-config-values.yaml     # Default: workers with all GPUs
│   ├── nvkind-config-values-dev.yaml # Dev: + local code mount
│   └── nvkind-config-template.yaml   # Custom Go template with extraMounts
├── helm/                             # Infrastructure (helmfile)
│   ├── helmfile.yaml                 # environments: kind, prod
│   └── values/
│       ├── nvidia-device-plugin.yaml # kind only
│       ├── kai-scheduler.yaml
│       └── kuberay-operator.yaml
└── examples/                         # Hand-written workload manifests
    ├── sandbox-pool.yaml             # Sandbox Deployment + Service
    ├── rayjob-monolithic.yaml        # GPU cluster via KubeRay RayJob
    ├── jobset-monolithic.yaml        # Same topology via JobSet (no KubeRay)
    └── kai-queue.yaml                # Example KAI queues (adapt to your cluster)
```

## Helmfile environments

| Environment | GPU component | Use case |
|-------------|---------------|----------|
| `kind` | nvidia-device-plugin | Local dev — nvkind handles toolkit/runtime |
| `prod` | (none — cluster provides GPU Operator) | Real clusters |

Both include KAI scheduler, the KubeRay operator, and the JobSet controller.

## Tear down (kind only)

```sh
kind delete cluster --name dockyard
```

## Notes

- **nvkind vs vanilla kind**: nvkind automates GPU device injection,
  nvidia-container-toolkit install inside nodes, containerd runtime config, and
  RuntimeClass registration.
- **nvidia-device-plugin** (kind only): the GPU Operator's driver validation
  fails inside kind nodes; the lightweight device plugin with CDI discovery is
  sufficient since nvkind sets up the runtime.
- **KAI scheduler** creates PodGroups automatically for recognized workload
  types (RayCluster, RayJob, Job, JobSet). For bare pods, create a PodGroup and
  annotate with `pod-group-name`.
- **DRA (GB300/RoCE topology placement)** is supported as an opt-in extension.
  Declare `resourceClaims` named `compute-domain-channel` / `roce-channel` on a
  GPU worker pod and the `dockyard-k8s` CLI auto-creates/deletes the matching
  `ComputeDomain` (NVLink) + RoCE `ResourceClaimTemplate`, rewriting the
  template names to be cluster-unique. Tune via `infra.dra` (`roceCount`,
  `computeDomainNumNodes`); set `infra.dra.autoCreate: false` to pass the
  claims through untouched and manage the CRDs yourself.

## Production hardening

> [!NOTE]
> Kubernetes production hardening — like the launcher itself — is **in active
> development**. The objects below are generated and statically validated
> (`kubeconform`, unit tests), but whether each one actually *enforces* on a live
> cluster is unverified without GPU/cluster resources (tracked as `HV-34`…`HV-36`
> in `hardware-deferred-validation.md`).

The generator and the hand-written `examples/sandbox-pool.yaml` apply a
hardening baseline. **On by default for the sandbox pool** (it runs untrusted
agent-generated code — the highest-value boundary):

- **securityContext** — `allowPrivilegeEscalation: false`, drop `ALL`
  capabilities, `seccompProfile: RuntimeDefault`.
- **NetworkPolicy** — ingress restricted to the executor port from in-namespace
  pods (cross-namespace + external ingress denied); egress left open so the
  executor can clone repos (lock it to DNS + CIDRs via
  `sandbox.networkPolicy.{allowEgress,allowedEgressCidrs}`).
- **PodDisruptionBudget** — `maxUnavailable: 1`.
- **ephemeral-storage** request — guards against disk-pressure eviction during
  large checkouts.

Sandbox **resource requests are deliberately small** (`cpu: 1`, `memory: 2Gi`):
requests are what the scheduler reserves per replica, so `replicas × request`
must fit on the nodes — large requests leave the pool `Pending`. The `cpuLimit`
/ `memoryLimit` / `ephemeralStorageLimit` fields raise the burst ceiling without
inflating that reservation (each limit defaults to its request when unset).

Tunable in the `.infra.yaml` under `sandbox.{security,networkPolicy,pdb}` and
`sandbox.resources.{cpu,memory,ephemeralStorage,*Limit}`; set any
`enabled: false` to drop a piece.

**Scheduling / placement.** `infra.priorityClassName` is applied to every
managed pod (a fleet can override its own in `kuberay.spec`) so a preempting
scheduler protects the gang; `sandbox.topologySpreadConstraints` spreads the
executor replicas (GPU-fleet topology is authored in the free-form `kuberay.spec`).

**Namespace governance (opt-in, render-only).** `infra.resourceQuota`
(`.hard`) and `infra.limitRange` (`.default` / `.defaultRequest` / `.max` /
`.min`) render a `ResourceQuota` + `LimitRange` ahead of the workload. They are
emitted by `render` (so `render | kubectl apply` installs them) but **not**
auto-applied by the live `run` / `cluster` path — namespace-wide quota is
cluster-admin territory and a too-tight quota applied mid-run would block the
pods being deployed.

**Live vs render.** The sandbox NetworkPolicy + PDB and the GPU `kuberay.pdb`
are applied and torn down by the live path too — `dockyard-k8s run`,
`cluster up`, and `cluster down` create/delete them alongside the Deployment /
Service / RayCluster, not only via `render | kubectl apply`.

**Opt-in for the GPU fleets.** `infra.security` (head + workers) and
`kuberay.pdb` (bringup mode) default **off**: dropped capabilities / non-root /
a disruption budget on the CUDA + NCCL stack are unverified without a GPU, so
enabling them blind risks breaking training. Turn them on (`infra.security.enabled:
true`, `kuberay.pdb.enabled: true`) once validated on hardware.

**Offline validation.** `bash dockyard_k8s/scripts/validate-manifests.sh` runs
`kubeconform` over the hand-written examples and the generator's rendered output
(built-in resources checked with `-strict`; CRDs skipped). The same check runs
as a unit test (`tests/unit/test_kubeconform.py`), skipped when the binary is
absent.

**Deliberately not built — `ServiceMonitor`.** A Prometheus `ServiceMonitor`
was scoped but dropped: the sandbox task executor exposes no `/metrics`
endpoint, and Ray's metrics Service is created/owned by KubeRay, not this tool —
so a generated ServiceMonitor would scrape nothing. Wire one to KubeRay's
metrics Service directly if you run Prometheus Operator.

Whether each hardening object actually *enforces* on a real cluster
(NetworkPolicy under a CNI, the PDB during a drain, the securityContext baseline
not breaking the workloads) is unverified without hardware — see
`hardware-deferred-validation.md` (`HV-34`…`HV-36`).

## Fairshare scheduling

KAI distributes GPUs with hierarchical fair-share in two phases:

1. **Guaranteed quota** — each queue gets its `quota` unconditionally.
2. **Over-quota surplus** — the rest is distributed by `priority` (higher
   first), then `overQuotaWeight` within the same priority.

| Field | Meaning |
|-------|---------|
| `quota` | Guaranteed GPUs. `-1` = unlimited, `0` = no guarantee |
| `limit` | Hard cap on total GPUs. `-1` = no limit |
| `overQuotaWeight` | Weight for surplus distribution (higher = bigger share) |
| `priority` | Over-quota order (higher = served first, reclaimed last) |
| `preemptMinRuntime` | Min runtime before a higher-priority queue can preempt |
| `reclaimMinRuntime` | Min runtime before over-quota resources can be reclaimed |

**Preempt** — a higher-priority queue takes from a lower-priority one.
**Reclaim** — a queue takes back what it is entitled to from an over-allocated
queue. `reclaimMinRuntime` is shorter than `preemptMinRuntime` because reclaim
is about fairness, while preempt protects long-running jobs from interruption.

`kai-queue.yaml` ships an example `root` / `rl` / `backfill` hierarchy — adapt
the quotas and names to your cluster.
