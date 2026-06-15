# The sandbox task executor

The sandbox fleet's job is to turn a patch into a verdict, safely and
repeatably. Each `ubuntu-swe` container runs a REST **task executor**
(`ubuntu-base/scripts/tools/task_executor.py`); the trainer side talks to it
through the `sandbox/` client. The agent never reaches the executor directly —
only the environment and reward layers do.

## The executor

The executor is a Flask app served by waitress on port `9090` (configurable via
`API_PORT`), with an optional `API_TOKEN`. It is **stateless per task**: each
submission gets a fresh working tree under `TASK_BASE_DIR`, so concurrent tasks
never interfere and no per-episode provisioning is required.

It exposes two usage patterns:

- **Single-shot batch** (`POST /task/submit`) — the SWE path. The request
  carries the repo, base commit, the agent patch, the gold `test_patch`, and the
  test node IDs. The executor clones the repo, applies the agent patch, **force-
  applies the gold `test_patch`** (so the held-out tests are canonical no matter
  what the agent patch did — see [rewards and integrity](rewards-and-integrity.md)),
  runs the tests under a wall-clock timeout, and returns the verdict.
- **Multi-turn session** — for agentic environments that need an interactive
  shell: a session is started, the agent issues `exec` commands turn by turn, and
  a `finish` step scores the result. Each in-flight session holds a worker thread
  for the duration of an `exec`/`finish`, so the thread pool
  (`DOCKYARD_TASK_EXECUTOR_THREADS`, default 32) is sized larger than the
  single-shot default.

Per-task wall-clock timeouts are enforced executor-side (`env.code.task_timeout`),
and all activity is logged to `TASK_BASE_DIR/task_executor.log`.

## The client

`sandbox/` is the trainer-side client. `run_task(TaskSpec)` submits a single-shot
task and returns a `TaskResult`; `SessionStartSpec`, `ExecResult`, and
`FinishResult` cover the multi-turn session API. `TaskExecutorError` wraps
transport and protocol failures so reward functions can map them to an
`execution_error` status rather than crashing the loop.

Executor endpoints are configured by `env.code.sandbox_urls` (or the
`DOCKYARD_SANDBOX_URLS` environment variable) and are round-robined across the
batch, so scoring throughput scales with the number of sandbox containers — the
reason the sandbox fleet is CPU-only and `SPREAD`: it never competes with the
inference fleet for GPUs, and episode slots distribute across CPU capacity.

## Why a separate fleet

Scoring is CPU-bound, bursty, and untrusted code execution. Isolating it in its
own fleet means: the trainer and inference GPUs are never blocked on a slow test
suite; a hostile or runaway patch is contained in a throwaway working tree on a
CPU node; and scoring capacity scales independently of training and generation.
