# dockyard_rl

Distributed RL post-training for **Project Dockyard** — async GRPO on Ray + PyTorch (DTensor/FSDP2), with vLLM/SGLang generation and an `ubuntu-swe` sandbox for code-execution rewards. Derived in lineage from NeMo-RL but **re-implemented as its own system** — treat it as standalone.

## Hard conventions (non-negotiable)

1. **This is NOT NeMo.** Never reference NeMo / NVIDIA / nemo_rl in code, comments, config, names, or conversation. All Python imports use `dockyard_rl.*`. **Attribution exception (intentional, preserve it):** the README **Acknowledgements** section and the root **`NOTICE`** file credit NeMo-RL as the development reference dockyard_rl was built against. This is deliberate, owner-approved, license-compliant attribution — do not strip it under this rule. The rule still bars NeMo/NVIDIA/nemo_rl names everywhere else (code, comments, config, identifiers, imports).
2. **Env-var prefix is `DOCKYARD_`** — never `NRL_` / `NEMO_`.
3. **Runtime: no `uv`, no virtualenvs.** Use `sys.executable` everywhere (Ray actors, `runtime_env`); `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`. Deps are baked into the `ubuntu-swe` image's system Python. **Build-time uv is fine/encouraged** (`uv.lock` + `uv pip install --system`); for light, fast-changing deps use a thin image layer, not Ray `runtime_env={"pip": ...}` (that reintroduces a runtime venv).
4. **Placement:** `STRICT_PACK` for per-node placement groups.
5. **Comments are technical and concise** — no colloquial/anthropomorphic phrasing, no decorative `===` banners.
6. **No placeholder/TODO code** — production-grade output only. Deferred imports for modules not yet written are intentional (wrap in `try/except ImportError` or guard under `TYPE_CHECKING`).
7. **Read the actual source before writing** — never reconstruct an API from memory. Ask before assuming a prior change was applied.
8. **Edit format:** before/after diffs for small-to-medium changes; full scripts for large/new files.
9. **Single author; voice.** dockyard_rl is developed by one person — **Kartik A** (alias **NullVoider**). Never use "we"/"our"/"us" in docs, comments, READMEs, or commit/PR text — write in the first-person singular ("I") or impersonally ("the project", passive). Author/copyright/citation metadata uses **Kartik A (NullVoider)** (e.g. the README citation's `author` field and the `NOTICE` copyright line).

## Architecture

Three async fleets, each declaring `DOCKYARD_FLEET_ROLE`:
- **trainer** (GPU, SPREAD) — DTensor policy workers; GRPO optimizer.
- **inference** (GPU, PACK) — vLLM async engine (SGLang backend also available); weights synced from trainer via NCCL collective.
- **sandbox** (CPU-only, SPREAD) — `ubuntu-swe` containers running a batch task executor (`POST /task/submit`, port 9090) that clones a repo, applies the agent patch + gold `test_patch`, runs tests, and returns a verdict.

Key directories: `algorithms/` (GRPO, loss), `models/` (generation: vllm/sglang; policy/dtensor), `distributed/` (virtual cluster, worker groups, batched data dict), `cluster/` (bootstrap, fleet, nccl_sync, placement), `data/` (datasets, processors, registries), `environments/` (`CodeEnvironment` — single-shot SWE patch scoring), `rewards/` (`IntegrityReward(TestRunnerReward)` — the anti-reward-hacking path), `sandbox/` (task-executor client), `ubuntu-base/` (`ubuntu-swe` image + `task_executor.py`), `examples/` (`run_grpo_swe.py` + `configs/`), `scripts/cluster/`, `ray.sub` (Slurm launcher).

Entry point: `python3 examples/run_grpo_swe.py --config examples/configs/grpo_swe.yaml <hydra overrides>`.

## Working in this repo

- **Validate without a cluster:** there are no live GPU/cluster resources for test runs. Validate code with `py_compile` and a clean Pylance/pyright pass (use the IDE diagnostics when available); validate the `ubuntu-swe` image with a `docker build` (the in-Dockerfile `import jax` / `import vllm` checks gate dependency breakage).
- **Reward/sandbox path:** the agent emits a unified diff; `CodeEnvironment` submits it to the task executor; `IntegrityReward` zeroes the reward if the patch edits a held-out test file, and the executor force-restores the gold tests so tampering is structurally inert.

## Code review

- Be concise and actionable: bugs, logic errors, missing tests, outdated docs, convention violations.
- Do NOT flag: style/formatting, minor naming, architectural opinions, or performance without a measurable issue.
- **Calibrate, don't suppress.** Skip only low-signal noise (style, speculative perf, subjective taste). Anything that could be a real correctness/safety bug, surface even when uncertain — state the confidence explicitly ("not certain, but X may Y — verify") rather than dropping it silently. When unsure about correctness, resolve it by reading the actual API/source before concluding; don't guess and don't stay quiet. "LGTM" is valid only when you genuinely found nothing, not as a way to avoid a hard call.
- **Verify upstream API usage.** When code calls vLLM, SGLang, Ray, torch/DTensor, or transformers, look up the real signature/return type/semantics — don't assume.

## Deployment

Slurm is wired (`ray.sub` + `scripts/cluster/`). Kubernetes orchestration is **built but in active development** — `infra/dockyard_k8s/` is a config-driven launcher (`check`/`render`/`run`/`cluster`/`status`/`logs`/`job`/`dev`/`doctor`) that renders the sandbox `Deployment`+`Service` and a `RayJob`/`RayCluster` from a recipe + infra spec; manifests/schema/CLI may still change (see `infra/README.md`). Original adaptation plan: `infra-adaptation-handoff.md`.
