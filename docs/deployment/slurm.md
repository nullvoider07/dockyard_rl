# Slurm

`ray.sub` (repo root) is the Slurm launcher: it brings up a Ray cluster across
the allocated nodes inside the `ubuntu-swe` container, exports each node's
`DOCKYARD_FLEET_ROLE` before `ray start` so `cluster/bootstrap.py` reads it
correctly, and then runs the training command.

It is written for the runtime model this project mandates: it uses the
`ubuntu-swe` image's **system Python** (`/usr/local/bin/python3`) rather than any
virtualenv, carries no `uv` venv machinery (deps are baked into the image at
build time), and keeps `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` set for defence in
depth.

## Usage

```bash
CONTAINER=<your-registry>/ubuntu-swe:latest \
MOUNTS="$PWD:$PWD" \
COMMAND="python3 examples/run_grpo_swe.py --config examples/configs/grpo_swe.yaml" \
sbatch \
  --nodes=4 \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_PARTITION \
  --gres=gpu:8 \
  ray.sub
```

Key environment variables (all optional except `CONTAINER`):

| Variable | Meaning |
| --- | --- |
| `CONTAINER` | The image to run (required). |
| `MOUNTS` | Comma-separated `host:container` mount pairs. |
| `COMMAND` | The command to run once the cluster is ready. If empty, the cluster idles and an attach script is generated. |
| `GPUS_PER_NODE` | GPUs per node (default 8). |

## Non-Slurm bare metal

For bare-metal, Docker, or direct-SSH clusters without Slurm, use the
`scripts/cluster/` scripts instead:

- `start_head.sh` — starts the Ray head on the current machine and prints the
  `RAY_ADDRESS` that workers join.
- `start_worker.sh` — attaches a worker to an existing head and declares the
  node's fleet role:

  ```bash
  DOCKYARD_FLEET_ROLE=<trainer|inference|sandbox> \
  RAY_ADDRESS=<head-ip>:<port> \
  bash scripts/cluster/start_worker.sh
  ```

The role is read at runtime by `get_fleet_role()`, which selects the placement
strategy and resource spec for that node (see
[cluster and fleets](../architecture/cluster-and-fleets.md)).

The per-benchmark setup scripts in `scripts/cluster/` (`swe_bench_setup.sh`,
`program_bench_setup.sh`, `terminal_bench_setup.sh`, …) prepare the dataset- and
environment-specific assets a given config expects.
