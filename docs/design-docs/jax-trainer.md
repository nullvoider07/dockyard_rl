# The JAX trainer backend

The trainer fleet has two interchangeable backends. The default is PyTorch
DTensor/FSDP2; the alternative is a **JAX** backend built on Flax NNX,
`jax.sharding`, and optax. Both sit below the same `models/policy/interfaces.py`
ABCs, so the GRPO loop, the rewards, the sandbox, and the inference fleet are
unchanged regardless of which trainer runs. Only the trainer fleet differs; the
inference fleet stays on vLLM and the sandbox on the executor.

The backend is selected per config (`policy.jax_cfg.enabled`); see
`examples/configs/grpo_swe_jax.yaml`. The implementation lives in `models/jax/`.

## Why a second backend

JAX/XLA brings GSPMD sharding (mesh + `PartitionSpec`) and a different
compilation and numerics story to the same training objective. Keeping it below
the policy interface means it is a drop-in: the refit seam emits the same HF
weight names the inference engine expects, so a JAX-trained policy serves through
the identical vLLM path.

## What it implements

The backend reproduces the trainer's responsibilities in JAX:

- **Models** — Flax NNX modules with HF-weight loaders and an exact name map:
  Qwen3 dense, the MoE path (mirroring `models/dtensor/moe/`), and the hybrid
  linear-attention path (Gated-DeltaNet) for the Qwen3-Next lineage.
- **Loss and log-probs** — a pure-JAX mirror of the clipped-PG loss and the
  log-prob gather, validated for value- and grad-parity against the torch loss.
- **Train step** — `nnx.split` + `jax.value_and_grad`, microbatch grad
  accumulation, an optax optimizer chain (AdamW + scheduler + global-norm clip),
  and the aux-loss-free load-balance bias stepped via a pre-optimizer hook.
- **Refit / checkpoint** — JAX→torch weight conversion (dlpack on GPU, numpy on
  CPU) through the shared name map, including the MoE per-expert re-expansion;
  Orbax checkpoints plus an HF-safetensors export so a JAX run's weights reload
  through the torch loader.

## Build phases

The re-platform was built and validated in phases (J0–J11), recorded in
`handoff/jax-trainer-replatform-plan.md` and `handoff/jax-build-progress.md`:
scaffolding and the config flag (J0); dense Qwen3 with HF-logit parity (J1);
mesh/`PartitionSpec` sharding (J2); loss, train step, refit, log-prob and
checkpoint seams (J3–J7); the Ray/driver glue (J9); MoE (J8); and hybrid
linear/hybrid attention for Qwen3-Next (J11).

## Validation posture

The entire CPU-validatable surface — dense → MoE → hybrid, plus loss/grad,
train-step, refit, checkpoint, and load-balance — is proven on CPU against HF
reference logits, with a clean type-check. Anything that genuinely needs a GPU or
multiple devices (the sharded multi-GPU train step, the live NCCL refit into
vLLM, the expert-parallel `ragged_all_to_all` collective, bf16 numerics) is
GPU-gated and tracked, with its exact bring-up check, in
`handoff/hardware-deferred-validation.md`. GPU validation is currently parked by
project decision; the ledger means resuming is "run the harness," not re-derive.
