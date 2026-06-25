# Configuration

Configuration is YAML, resolved by [OmegaConf](https://omegaconf.readthedocs.io/)
with Hydra-style overrides, then validated into a Pydantic `MasterConfig`
(`algorithms/grpo.py`). A run is fully described by one config file plus any CLI
overrides.

## The override flow

```text
YAML file ──▶ OmegaConf ──▶ apply CLI overrides ──▶ resolve ${...} ──▶ MasterConfig(**dict)
```

`MasterConfig` is declared `extra="allow"`, so configs can carry keys beyond the
documented schema (subsystems read their own sub-trees), but the listed
top-level sections are required.

### CLI overrides

Everything after `--config` is a Hydra dot-notation override:

```bash
python3 examples/run_grpo_swe.py --config examples/configs/grpo_swe.yaml \
    policy.model_name=Qwen/Qwen2.5-14B-Instruct \
    grpo.num_prompts_per_step=32 \
    loss_fn.reference_policy_kl_penalty=0.02
```

### Interpolation and resolvers

Configs use OmegaConf interpolation (`${policy.max_total_sequence_length}`) to
keep derived values in sync, plus custom resolvers registered by
`register_omegaconf_resolvers()`. For example `mul:` multiplies, used to derive
micro-batch token budgets:

```yaml
train_mb_tokens: ${mul:${policy.max_total_sequence_length}, ${policy.train_micro_batch_size}}
```

## Top-level sections

`MasterConfig` has these sections:

| Section | Type | Controls |
| --- | --- | --- |
| `policy` | dict | The trained model: name, precision, batch sizes, sequence length, the parallelism backend (`dtensor_cfg` / `megatron_cfg` / JAX), optimizer, scheduler, generation engine, LoRA (Low-Rank Adaptation), sequence packing. |
| `loss_fn` | `ClippedPGLossConfig` | The clipped policy-gradient loss: clip ranges, reference-KL (Kullback–Leibler) penalty and type, importance-sampling correction, token- vs. sequence-level reduction. |
| `grpo` | `GRPOConfig` | The RL loop: prompts/generations per step, rollout turns, advantage estimator, reward shaping/scaling, invalid-action penalty, structured tool use, async-GRPO settings, validation cadence. |
| `env` | dict | Per-environment settings (for SWE: scoring workers, reward mode, integrity check, sandbox URLs, task timeout). |
| `data` | dict | Datasets and dataloaders: train/validation dataset names, splits, prompt-length caps, the default processor and environment. |
| `cluster` | `ClusterConfig` | Cluster shape: `gpus_per_node`, `num_nodes`. |
| `logger` | dict | Logging backends (W&B, TensorBoard, MLflow, SwanLab), GPU monitoring, sample printing. |
| `checkpointing` | dict | Save cadence, top-k retention by metric, save format (`safetensors`), optimizer-state saving. |
| `data_plane` | `DataPlaneConfig` or `None` | Optional external experience-transfer plane for non-colocated fleets. `None` uses the in-memory path. |

## Key knobs

A few settings disproportionately shape a run:

- **RL group size** — `grpo.num_prompts_per_step` ×
  `grpo.num_generations_per_prompt`. This is the number of rollouts per step and
  should match `policy.train_global_batch_size`.
- **Sequence budget** — `policy.max_total_sequence_length` bounds prompt +
  response and feeds the packing/dynamic-batching token budgets.
- **Async GRPO** — `grpo.async_grpo.enabled` switches loops. When on, several
  features are unsupported (dynamic sampling, reward scaling/shaping, multiple
  dataloaders) and IS correction is mandatory; these are validated at launch.
- **Parallelism** — `policy.dtensor_cfg` selects tensor/context/expert parallel
  degrees for the trainer; `policy.generation.vllm_cfg` selects the independent
  inference-side parallelism.
- **Precision and FP8** — `policy.precision` is the training/master precision;
  `policy.generation.vllm_cfg.precision=fp8` serves the inference fleet in FP8
  while training stays in bf16 (weights are quantized on each refit).
- **Generation engine** — `policy.generation` configures sampling and the
  inference backend (vLLM or SGLang). Two knobs beyond the usual sampling fields:
  `ignore_eos` lets a rollout keep generating past the end-of-sequence token (off
  by default; useful when a reward needs a fixed-length completion), and
  `policy.generation.vllm_cfg.env_vars` passes a per-recipe map of environment
  variables through to the vLLM workers — for example to select a fused-MoE
  backend for a particular model without baking it into the image. `temperature`
  and `top_p` are validated for finiteness at worker startup — a NaN/Inf value is
  rejected rather than forwarded to the engine. Per-sample image inputs are gated
  by `allow_multimodal_inputs` (default `false`): a VLM/CUA recipe must set it to
  `true` to forward images to vLLM, and when enabled images are EXIF- and
  transparency-normalized before the engine sees them. The text generation path is
  unaffected by the flag.
- **Logging backends** — `logger` enables any of W&B, TensorBoard, MLflow, and
  SwanLab together. W&B receives full per-step series (including the per-worker
  generation timeline) as-is. Scalar-only backends (MLflow) cannot hold a list
  metric, so a list-valued metric is summarized to `<name>/{mean,p50,p90,max}`
  and the per-worker generation timeline is merged across workers before that
  reduction — one bounded set of scalar keys per metric instead of one key per
  element.

## The config family

`examples/configs/` holds one config per target environment; they share the
GRPO core and differ in dataset, environment, and rollout shape:

| Config | Target |
| --- | --- |
| `grpo_swe.yaml` | SWE-bench coding agent (single-shot patch). |
| `grpo_swe_pro.yaml` | SWE-bench Pro variant. |
| `grpo_swe_sglang.yaml` | SWE-bench on the SGLang generation backend. |
| `grpo_swe_jax.yaml` | SWE-bench with the JAX (Flax NNX) trainer backend. |
| `grpo_program_bench.yaml` | Program-synthesis benchmark. |
| `grpo_terminal_bench.yaml` | Terminal-bench agentic tasks. |
| `grpo_osworld.yaml` | OSWorld computer-use agent. |
| `grpo_gdpval.yaml` / `grpo_gdpval_agentic.yaml` | GDPval file-producing tasks. |
| `grpo_hle.yaml` | Humanity's Last Exam. |

Each config's header comment documents the topology and target hardware it was
written for.
