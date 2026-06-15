# Installation

## The runtime model

dockyard_rl does **not** use `uv` or virtualenvs at runtime. Every runtime
dependency — PyTorch, vLLM, JAX, transformers, the task executor's toolchain —
is baked into the `ubuntu-swe` container image's **system Python**. Ray actors
and `runtime_env` use `sys.executable` directly, and `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`
disables Ray's implicit `uv` integration.

This is a deliberate constraint: a Ray `runtime_env={"pip": ...}` would
re-create a per-actor virtualenv at runtime, which is exactly what the image is
meant to eliminate. Light, fast-changing dependencies belong in a thin image
layer, not in a runtime pip install.

`uv` **is** used at build time (`uv.lock` + `uv pip install --system`) to
resolve and install into the image's system Python. Build-time `uv` is
encouraged; runtime `uv` is forbidden.

## Building the `ubuntu-swe` image

The image is the unit of deployment for all three fleets. Its sources live in
`ubuntu-base/`:

| Dockerfile | Purpose |
| --- | --- |
| `ubuntu-swe-v2.dockerfile` | Default image — CUDA base, PyTorch, vLLM, JAX, and the task executor. |
| `ubuntu-swe-sglang.dockerfile` | SGLang generation backend variant. |
| `ubuntu-swe-gdpval.dockerfile` | GDPval file-producing environment variant. |

Build the default image:

```bash
cd ubuntu-base
docker build -f ubuntu-swe-v2.dockerfile -t ubuntu-swe:v2 .
```

The Dockerfile contains in-build `import jax` / `import vllm` smoke checks. A
build that fails there has a broken dependency graph — that gate is the primary
way to catch dependency breakage without a cluster.

## Local development

For editing, type-checking, and the CPU parity tests you only need the package
itself plus a CPU stack; the heavy GPU dependencies are not required to run the
unit suite or the type checker.

```bash
# From the repository root.
pip install -e .
```

`pyproject.toml` pins only the lightweight local-dev/CI dependencies
(`ray[default]`, `numpy`, `tqdm`). The rest come from the image.

### Type checking

The canonical type-check runs `pyright` from the **parent** of the package
directory so imports resolve as `dockyard_rl.*`:

```bash
cd ..            # parent of the dockyard_rl/ package directory
pyright dockyard_rl/...
```

### Running the unit tests

The unit suite is CPU-only and skips anything that requires a GPU or a live
cluster:

```bash
pytest tests/unit -q
```

GPU- and cluster-only behaviour is not silently skipped — it is tracked in
`handoff/hardware-deferred-validation.md` with the exact bring-up check for each
item, so validating on hardware later is "run the harness," not "re-derive what
to test."

## Environment variables

All dockyard_rl environment variables use the `DOCKYARD_` prefix.

| Variable | Meaning |
| --- | --- |
| `DOCKYARD_FLEET_ROLE` | `trainer`, `inference`, or `sandbox`. Every container must set this before `init_ray()`; it selects the placement strategy, resource spec, and NCCL (NVIDIA Collective Communications Library) topology. |
| `DOCKYARD_SANDBOX_URLS` | Comma-separated task-executor endpoints (alternative to setting `env.code.sandbox_urls` in the config). |
| `RAY_ENABLE_UV_RUN_RUNTIME_ENV` | Set to `0` — disables Ray's `uv` runtime integration (see the runtime model above). |

The full process environment is forwarded to every Ray worker, so image-level
settings (`DOCKYARD_FLEET_ROLE`, `NCCL_SOCKET_IFNAME`, `CUDA_DEVICE_ORDER`, …)
are visible inside remote actors without extra wiring.
