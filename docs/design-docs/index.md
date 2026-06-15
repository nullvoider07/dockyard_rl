# Design docs

The "why" behind each subsystem — the reasoning, invariants, and trade-offs that
the code alone doesn't make obvious. Where the [architecture](../architecture/index.md)
docs describe how the system fits together at runtime, these describe why each
piece is built the way it is.

- [GRPO and the clipped-PG loss](grpo-and-loss.md) — advantage estimation and
  the single configurable loss behind PPO / GRPO / RLOO / DAPO / GSPO.
- [Rewards and integrity](rewards-and-integrity.md) — execution-grounded
  scoring and the anti-reward-hacking path.
- [The sandbox task executor](sandbox-executor.md) — the `ubuntu-swe` REST
  executor and its client.
- [Structured tool-use protocol](tool-use-protocol.md) — native Hermes
  tool-calls, validation, and RL-safe constraining.
- [Mixture-of-Experts](moe.md) — expert parallelism, grouped GEMM, and
  aux-loss-free load balancing.
- [The JAX trainer backend](jax-trainer.md) — the Flax NNX re-platform below the
  policy interface.
- [Preference optimization (DPO)](dpo.md) — DPO and online DPO.

```{toctree}
:hidden:
:maxdepth: 1

grpo-and-loss
rewards-and-integrity
sandbox-executor
tool-use-protocol
moe
jax-trainer
dpo
```
