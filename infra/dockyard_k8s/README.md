# dockyard-k8s

> [!WARNING]
> Active development. Manifests, schema, and CLI surface may change.

A config-driven launcher that turns a `dockyard_rl` recipe plus an infra spec
into a running job on Kubernetes: one GPU RayCluster (trainer + inference
worker groups) and a GPU-free sandbox executor pool (Deployment + Service).
One command brings the stack up, submits the training entrypoint, and streams
its logs.

Two halves:

* **Generate / inspect** — `check` and `render` turn the recipe + infra into
  manifests with no cluster contact.
* **Live orchestration** — `run`, `cluster`, `status`, `logs`, `job`, `dev`,
  and `doctor` apply those manifests, submit training, and observe it. These
  talk to the cluster via the `kubernetes` client + `kubectl`.

## What it generates

One YAML pair captures *what to train* (the recipe) and *where to run it*
(the infra). From that pair the CLI renders, in apply order:

1. **Sandbox `Deployment` + `Service`** — a pool of `ubuntu-swe` task
   executors (`POST /task/submit`, port 9090), GPU-free, fronted by a
   ClusterIP for stable DNS. Off the GPU compute path.
2. **`RayJob`** (`launch.mode: rayjob`) — a single GPU RayCluster with two
   worker groups, `trainer` and `inference`, sharing one cluster so the
   trainer→inference NCCL weight sync is co-resident. The runtime carves them
   into per-fleet placement groups via the `dockyard_fleet_<role>` Ray
   resource tag. Alternatively a bare **`RayCluster`** (`launch.mode: bringup`)
   or nothing (`launch.mode: attach`, for an externally-managed cluster).

Cross-cutting fields the CLI injects into every GPU pod:

- `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` — no runtime venvs (deps are baked into
  the image).
- `DOCKYARD_SANDBOX_URLS` — the sandbox Service DNS, when `sandbox.injectUrls`
  is set. Launch the recipe with `env.code.sandbox_urls=null` so this env var
  wins over the recipe's localhost default.
- `image`, `imagePullSecrets`, optional `serviceAccountName`, and the
  `app.kubernetes.io/managed-by: dockyard-k8s` label.

## Install

This is a laptop / ops-host tool — it runs on an operator's machine (or an
in-cluster ops pod), never inside a Ray training actor, so installing it with
pip carries none of the runtime-venv concerns that apply to the cluster image.

```bash
pip install ./infra/dockyard_k8s
dockyard-k8s --version
```

Editable for development:

```bash
pip install -e "./infra/dockyard_k8s[test]"
pytest infra/dockyard_k8s
```

It depends on `click`, `omegaconf`, `pydantic`, `kubernetes`, `ray[default]`,
`tenacity`, and `pyyaml`. It does **not** require `dockyard_rl` itself to be
importable — but if it is on the path, its recipe loader
(`dockyard_rl.utils.config`) is used for `defaults:` inheritance and Hydra
overrides; otherwise a built-in OmegaConf fallback handles both.

## Config layout

Each run is two files: a recipe and an infra.

- **`<recipe>.yaml`** — a pure `dockyard_rl` config (`policy`, `grpo`, `data`,
  `env`, `logger`, …). Inherits via a standard `defaults:` field. Portable;
  no environmental assumptions.
- **`<recipe>.infra.yaml`** — K8s-only. Namespace, image, the inline
  RayCluster spec, the sandbox pool, launch entrypoint, and code source.
  Validated against `dockyard_k8s.schema.InfraConfig`
  (`src/dockyard_k8s/schema.py`).

You can also bundle the two — put an `infra:` top-level key on the recipe and
omit `--infra`. The split is preferred for anything you plan to share.

### Override priority

Four layers stack low-to-high (last wins):

1. Shipped defaults: `src/dockyard_k8s/defaults/defaults.example.yaml`
2. User defaults: `~/.config/dockyard-k8s/defaults.yaml` (optional; repoint
   with `DOCKYARD_K8S_DEFAULTS=/path/to/file.yaml`)
3. The infra file (via `--infra`) *or* the recipe's `infra:` block. Not both.
4. Hydra-style CLI overrides: `infra.scheduler.queue=team-a`,
   `grpo.max_num_steps=10`. Overrides prefixed `infra.*` target the infra
   layer; everything else targets the recipe.

`namespace`, when unset, is inferred from the active kube context (falling
back to the in-pod service-account namespace, then `default`).

## Command reference

Both commands take the recipe path first, then positional Hydra overrides,
then flags. Pass `--infra <path>` when recipe and infra are split.

### `dockyard-k8s check`

Load and validate a recipe/infra pair. Default mode prints a one-page summary
(namespace, image, scheduler/queue, worker groups, sandbox pool + resolved
`DOCKYARD_SANDBOX_URLS`, the full entrypoint, and the manifest list). Pass
`-o <file>` to write the fully-resolved bundle (`InfraConfig` + recipe +
rendered manifests) instead; format follows the extension (`.yaml` / `.json`).

```bash
# summary
dockyard-k8s check examples/configs/grpo_swe.yaml \
  --infra infra/dockyard_k8s/examples/grpo_swe.infra.yaml

# full bundle for diffs
dockyard-k8s check examples/configs/grpo_swe.yaml \
  --infra infra/dockyard_k8s/examples/grpo_swe.infra.yaml -o /tmp/bundle.yaml
```

### `dockyard-k8s render`

Emit the rendered manifests as multi-document YAML on stdout.

```bash
dockyard-k8s render examples/configs/grpo_swe.yaml \
  --infra infra/dockyard_k8s/examples/grpo_swe.infra.yaml | kubectl apply -f -
```

`--dry-run` pipes the output through `kubectl apply --dry-run=client -f -` for
structural validation (needs a reachable API server for the OpenAPI schema;
not usable fully offline).

### `dockyard-k8s run`

Submit a recipe. Dispatch is on `infra.launch.mode`:

| mode | behaviour |
|---|---|
| `rayjob` | Apply a KubeRay RayJob; KubeRay creates the cluster, submits the entrypoint, polls to terminal, tears down. `--dry-run` renders without applying; `--no-wait` returns after apply. |
| `bringup` | Ensure the sandbox pool + GPU RayCluster, then submit training via the transport. Cluster stays up. `--replace` stops running jobs first; `--recreate` re-applies a drifted cluster. |
| `attach` | Submit onto the existing RayCluster named `launch.attach`. |

`--mode interactive` (default) = port-forward + working_dir upload + tail logs;
`--mode batch` = `kubectl exec` + in-image code + return after launch. Each axis
is independently overridable (`--submitter`, `--code-source`, `--code-path`,
`--run-id`, `--wait/--no-wait`). `--repo-root` points at the dockyard_rl
checkout to stage for upload mode.

```bash
dockyard-k8s run examples/configs/grpo_swe.yaml \
  --infra infra/dockyard_k8s/examples/grpo_swe.infra.yaml
```

### Cluster, job, status, logs, dev, doctor

- `status` — RayCluster state + head/worker pod phases + sandbox readiness.
- `logs --source {head,worker,sandbox} [-f]` — stream a pod's logs.
- `cluster up|down [--recreate] [--dry-run]` — bring the sandbox + RayCluster up or tear them down.
- `cluster list [-n ns]` / `cluster dashboard <name> [--port]` — list clusters / port-forward the Ray dashboard.
- `job list|logs <id>|stop <id> [--force]` — inspect and control Ray jobs (transport-aware via the cached run handle in `~/.cache/dockyard-k8s/runs/`).
- `dev connect|stop|setup-secrets` — a CPU dev pod on the `dockyard-workspace` PVC at `/mnt/dockyard`.
- `doctor` — preflight: kubectl reachable + authorized, KubeRay CRDs installed.

## Example

`examples/grpo_swe.infra.yaml` is the infra spec for the real SWE-bench GRPO
recipe (`examples/configs/grpo_swe.yaml`): a 4-node × 8-GPU RayCluster split
into a 3-node `trainer` group and a 1-node `inference` group, plus an 8-replica
sandbox pool. See the raw hand-written equivalents (RayJob, JobSet, sandbox
pool, KAI queue) under `infra/examples/`.
