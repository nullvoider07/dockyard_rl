# dockyard_rl

**dockyard_rl** is distributed reinforcement-learning post-training
infrastructure for Project Dockyard. It runs **asynchronous GRPO** (Group
Relative Policy Optimization) on [Ray](https://docs.ray.io/) with PyTorch
(DTensor/FSDP2 — Distributed Tensors + Fully Sharded Data Parallel v2) policy
workers and an optional JAX (Flax NNX) trainer backend,
[vLLM](https://docs.vllm.ai/) (or SGLang) generation, and an `ubuntu-swe`
sandbox that turns code execution into a reward signal. Acronyms used throughout
these docs are expanded in the [glossary](glossary.md).

The training signal is grounded in real execution: the agent emits a unified
diff, a sandbox applies it on top of the repository's held-out gold tests, runs
those tests, and returns a verdict. That verdict — not a learned reward model —
is the reward.

## The three-fleet model

The system is split into three asynchronous fleets, each declaring its role
through the `DOCKYARD_FLEET_ROLE` environment variable. They run concurrently
and exchange weights and experience over dedicated transports rather than
sharing a process.

| Fleet | Hardware | Placement | Responsibility |
| --- | --- | --- | --- |
| **trainer** | GPU | `SPREAD` | DTensor/FSDP2 policy workers and the GRPO optimizer. Optional Flax NNX (JAX) backend below the same policy interface. |
| **inference** | GPU | `PACK` | vLLM async engine (SGLang also available); serves rollouts. Receives updated weights from the trainer over a NCCL (NVIDIA Collective Communications Library) collective. |
| **sandbox** | CPU only | `SPREAD` | `ubuntu-swe` containers running a batch task executor (`POST /task/submit`) that clones a repo, applies the agent patch and gold `test_patch`, runs the tests, and returns a verdict. |

Because the fleets are not colocated, the trainer never blocks on generation and
generation never blocks on scoring: the async GRPO loop overlaps rollout,
reward, and optimization, refreshing the inference fleet's weights in flight.

## How the system trains

1. The trainer pulls a batch of prompts and the inference fleet generates
   `num_generations_per_prompt` rollouts each (the RL group).
2. Each rollout's patch is scored by the sandbox; the verdict becomes the
   reward, optionally shaped and penalized.
3. GRPO computes group-relative advantages (leave-one-out baseline), the policy
   recomputes log-probs, and the clipped policy-gradient loss with a reference
   KL (Kullback–Leibler) penalty drives an optimizer step.
4. Updated weights are synchronized to the inference fleet; the loop continues.

The [architecture overview](architecture/overview.md) walks through this loop in
detail, including the async replay buffer and weight-staleness handshake.

## Where to start

- **New here?** Read the [architecture overview](architecture/overview.md) for
  the mental model, then run the [quickstart](getting-started/quickstart.md).
- **Setting up an environment?** See
  [installation](getting-started/installation.md).
- **Tuning a run?** The [configuration guide](getting-started/configuration.md)
  maps every top-level config section to what it controls.

## Acknowledgements

dockyard_rl was developed with reference to **NVIDIA's NeMo-RL** post-training
library, used as a design and implementation reference throughout the project's
development and evolution. dockyard_rl is an independent, standalone
re-implementation rather than a fork, but its lineage traces to NeMo-RL. See the
`NOTICE` file at the repository root for attribution details.

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
architecture/index
design-docs/index
environments/index
deployment/index
glossary
apidocs/index
```
