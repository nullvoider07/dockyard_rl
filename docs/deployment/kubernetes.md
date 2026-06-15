# Kubernetes

:::{warning}
The Kubernetes path is under active development — manifests, schema, and the CLI
surface may change. Use it for a local GPU playground and as the basis for a
production deployment built with your cluster operator, not as a frozen API.
:::

The Kubernetes deployment lives under `infra/`. `dockyard_k8s`
(`infra/dockyard_k8s/`) is a config-driven launcher that turns a `dockyard_rl`
recipe plus an infra spec into a running job: one GPU RayCluster (trainer +
inference worker groups) and a GPU-free sandbox executor pool.

## Topology on Kubernetes

The three fleets map onto Kubernetes as:

- **Trainer + inference** — two worker groups **inside one GPU RayCluster**. They
  must be co-resident because the trainer→inference NCCL weight sync needs them in
  the same cluster. The runtime carves them into per-fleet placement groups via a
  `dockyard_fleet_<role>` Ray resource tag, which each worker group sets in its
  `ray start --resources` alongside exporting `DOCKYARD_FLEET_ROLE`. There is no
  trainer-vs-inference cluster split.
- **Sandbox** — a separate `Deployment` + `Service` of `ubuntu-swe` task
  executors (`POST /task/submit`, port 9090), GPU-free, fronted by a ClusterIP for
  stable DNS, reached over HTTP via `DOCKYARD_SANDBOX_URLS`. This is the *only*
  disaggregation: GPU Ray cluster vs. sandbox pool.

## The launcher

`dockyard_k8s` takes one YAML pair — the **recipe** (what to train) and the
**infra** spec (where to run it) — and has two halves:

- **Generate / inspect** — `check` and `render` produce manifests with no cluster
  contact.
- **Live orchestration** — `run`, `cluster`, `status`, `logs`, `job`, `dev`,
  `doctor` apply the manifests, submit training, and observe it via the
  `kubernetes` client and `kubectl`.

It renders, in apply order: the sandbox `Deployment` + `Service`, then a `RayJob`
(`launch.mode: rayjob`) — or a bare `RayCluster` (`bringup`), or nothing
(`attach`, for an externally-managed cluster).

Into every GPU pod the CLI injects the runtime invariants: `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`
(no runtime venvs), the image and pull secrets, and — when `sandbox.injectUrls`
is set — `DOCKYARD_SANDBOX_URLS` pointing at the sandbox Service DNS (launch the
recipe with `env.code.sandbox_urls=null` so the injected value wins over the
recipe's localhost default).

## Backend images

There are two `ubuntu-swe` image builds, one per inference backend (vLLM and
SGLang): they pin conflicting `torch`/`flashinfer` stacks and cannot coexist in
one image, so the backend is chosen at the image level. Example infra specs are
in `infra/dockyard_k8s/examples/` (`grpo_swe.infra.yaml`,
`grpo_swe_sglang.infra.yaml`); the `kind` + `helm` assets under `infra/` give a
local GPU playground.

## Production hardening

:::{note}
Kubernetes production hardening — like the launcher itself — is in **active
development**. The objects below are generated and statically validated
(`kubeconform`, unit tests), but whether each one actually *enforces* on a live
cluster is unverified without GPU/cluster resources; those checks are tracked as
`HV-34`…`HV-36` in the `hardware-deferred-validation.md` ledger at the repo root.
:::

The generator and the hand-written `infra/examples/sandbox-pool.yaml` apply a
hardening baseline. It is **on by default for the sandbox pool** — the pool runs
untrusted agent-generated code, so it is the highest-value boundary:

- **securityContext** — `allowPrivilegeEscalation: false`, drop `ALL`
  capabilities, `seccompProfile: RuntimeDefault`. (`runAsNonRoot` /
  `readOnlyRootFilesystem` are opt-in: the executor writes checkouts to disk and
  the image's default user is root.)
- **NetworkPolicy** — ingress restricted to the executor port from in-namespace
  pods (cross-namespace and external ingress denied); egress left open so the
  executor can clone repos. Lock it down with
  `sandbox.networkPolicy.{allowEgress, allowedEgressCidrs}` (DNS is always
  permitted). Requires a CNI that enforces NetworkPolicy; on a non-enforcing CNI
  the object is an inert no-op.
- **PodDisruptionBudget** — `maxUnavailable: 1`.
- **ephemeral-storage** request — guards against disk-pressure eviction during
  large checkouts.

### Right-sizing the sandbox pool

Sandbox **resource requests are deliberately small** (`cpu: 1`, `memory: 2Gi`):
requests are what the scheduler reserves per replica, so `replicas × request`
must fit on the nodes — large requests leave the pool `Pending`. The `cpuLimit`
/ `memoryLimit` / `ephemeralStorageLimit` fields raise the burst ceiling (for a
heavy test run) without inflating that reservation; each limit defaults to its
request when unset.

Tune everything under `sandbox.{security, networkPolicy, pdb}` and
`sandbox.resources.{cpu, memory, ephemeralStorage, *Limit}`; set any
`enabled: false` to drop a piece.

### Scheduling, placement, governance

- **`infra.priorityClassName`** — applied to every managed pod (GPU gang +
  sandbox) so a preempting scheduler protects the gang; a fleet can override its
  own in `kuberay.spec`. The named PriorityClass must already exist in the cluster.
- **`sandbox.topologySpreadConstraints`** — spreads the executor replicas
  (GPU-fleet topology is authored in the free-form `kuberay.spec`).
- **`infra.resourceQuota` / `infra.limitRange`** — opt-in namespace governance.
  Rendered ahead of the workload so `render | kubectl apply` installs them, but
  **not** auto-applied by the live `run` / `cluster` path: namespace-wide quota is
  cluster-admin territory, and a too-tight quota applied mid-run would block the
  pods being deployed.

### Live vs. render, and GPU opt-in

The sandbox NetworkPolicy + PDB and the GPU `kuberay.pdb` are applied and torn
down by the live path too — `dockyard-k8s run`, `cluster up`, and `cluster down`
create/delete them alongside the Deployment / Service / RayCluster, not only via
`render | kubectl apply`.

Hardening for the **GPU fleets** is opt-in. `infra.security` (head + workers) and
`kuberay.pdb` (bringup mode) default **off**: dropped capabilities / non-root / a
disruption budget on the CUDA + NCCL stack are unverified without a GPU, so
enabling them blind risks breaking training. Turn them on
(`infra.security.enabled: true`, `kuberay.pdb.enabled: true`) once validated on
hardware.

### Offline validation

```sh
bash infra/dockyard_k8s/scripts/validate-manifests.sh
```

runs `kubeconform` over the hand-written examples and the generator's rendered
output — built-in resources checked with `-strict`, CRDs skipped. The same check
runs as a unit test (`infra/dockyard_k8s/tests/unit/test_kubeconform.py`),
skipped when the binary is absent.

:::{note}
A Prometheus `ServiceMonitor` was scoped but **deliberately not built**: the
sandbox task executor exposes no `/metrics` endpoint, and Ray's metrics Service
is created/owned by KubeRay, not this launcher — a generated ServiceMonitor would
scrape nothing. Wire one to KubeRay's metrics Service directly if you run
Prometheus Operator.
:::
