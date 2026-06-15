# Quickstart — GRPO on SWE-bench

This walks through a single-shot SWE-bench (Software Engineering benchmark)
coding-agent run: the agent produces one patch per task, the sandbox scores it,
and GRPO (Group Relative Policy Optimization) optimizes against the verdict.

## Prerequisites

- The `ubuntu-swe` image built and deployable to your cluster
  (see [installation](installation.md)).
- A Ray cluster with the three fleets, or a single-node local-dev cluster.
- At least one **sandbox** task executor reachable at the URL(s) in
  `env.code.sandbox_urls` (default `http://localhost:9090`).

## Launch

```bash
python3 examples/run_grpo_swe.py \
    --config examples/configs/grpo_swe.yaml \
    cluster.gpus_per_node=8 \
    cluster.num_nodes=4 \
    policy.model_name=Qwen/Qwen2.5-7B-Instruct
```

`--config` selects the YAML; every argument after it is a Hydra-style
dot-notation override (`key=value`, `key.nested=value`) applied on top. The
[configuration guide](configuration.md) covers the override system.

The default `grpo_swe.yaml` topology is **non-colocated, async GRPO** on a vLLM
async engine, sized for 4 nodes × 8 GPUs (3 trainer + 1 inference).

## What happens at launch

`run_grpo_swe.py` is thin; the work is in `algorithms/grpo.py`. In order:

1. **Config load.** The YAML is parsed by OmegaConf, CLI overrides are applied,
   and the result is validated into a Pydantic `MasterConfig`. Async-GRPO
   constraints are checked *before* any model loads, so misconfigurations fail
   fast.
2. **Ray init.** `init_ray()` attaches to the cluster (or starts a local-dev
   one) and forwards the process environment to all workers.
3. **Tokenizer / processor.** Loaded from `policy.tokenizer`; the
   computer-use path loads a multimodal `AutoProcessor` instead.
4. **Generation config.** Pad-token IDs and related fields are finalized for the
   generation backend.
5. **Data and environments.** `setup_response_data()` loads the datasets through
   the registry (binding each dataset's processor) and constructs the
   environments, returning the `task_name → environment` maps GRPO consumes.
6. **`setup()`.** Builds and returns the full component set: the policy, the
   generation engine, the virtual cluster, train/val dataloaders, the loss
   function, the logger, the checkpoint manager, the GRPO save-state, the
   resolved master config, and the weight synchronizer.
7. **Train.** If `grpo.async_grpo.enabled` is true, `async_grpo_train()` runs
   (replay buffer, in-flight weight updates); otherwise `grpo_train()` runs the
   synchronous loop.

## Synchronous vs. asynchronous

The two loops share the same math but differ in how generation and training
overlap:

- **Synchronous** (`grpo_train`): generate the full batch, score, then train —
  one phase at a time.
- **Asynchronous** (`async_grpo_train`): a replay buffer decouples generation
  from optimization. Trajectories remain valid for
  `async_grpo.max_trajectory_age_steps` optimizer steps, and weights can be
  pushed to the inference engine *while it is generating*
  (`in_flight_weight_updates`). Async GRPO **requires**
  `loss_fn.use_importance_sampling_correction=true` to correct for the resulting
  off-policy staleness.

See the [architecture overview](../architecture/overview.md) for the full loop,
including the weight-staleness handshake between the trainer and inference
fleets.
