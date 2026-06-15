# Mixture-of-Experts

MoE support lets dockyard_rl train sparse models where each token is routed to a
few of many experts. The default configuration reproduces the dense path exactly
(`expert_parallel_size=1`, local dispatch), so enabling MoE support changes
nothing for dense models — the sparse machinery only engages for MoE
architectures.

The torch implementation lives in `models/dtensor/moe/`; the JAX backend mirrors
it under `models/jax/moe/` (see [the JAX trainer](jax-trainer.md)).

## The pipeline

A MoE layer (`block.py`) runs: route → dispatch → grouped expert GEMM → combine.

1. **Router** (`router.py`) — a top-k gate scores each token over the experts and
   selects `num_experts_per_tok`. The routing weights and the per-expert
   assignment drive the dispatch.
2. **Dispatch** (`dispatch.py`) — gathers each expert's assigned tokens. Two
   token dispatchers exist:
   - `local` — single-device / no expert parallelism (`ep_size=1`).
   - `alltoall` — expert-parallel; an all-to-all collective exchanges tokens so
     each rank processes only its local experts.
3. **Grouped GEMM** (`experts.py`) — the routed experts are evaluated as one
   grouped matmul (`torch._grouped_mm`) over the ragged per-expert token groups,
   rather than a Python loop over experts.
4. **Combine** — expert outputs are scattered back and weighted by the routing
   weights.

## Expert parallelism

The expert-parallel mesh axis (`mesh.py`, `sharding.py`) shards the routed
experts across ranks. `parallelize_moe.py` applies the sharding plan; `detect.py`
and `surgery.py` identify MoE structure in an arbitrary HF model and splice in
the parallel-aware blocks. The trainer-side `expert_parallel_size` is carved from
the `dp_shard × cp × tp` world and is distinct from the inference-side
`generation.vllm_cfg.expert_parallel_size`.

## Aux-loss-free load balancing

Naive routing collapses — a few experts attract most tokens. Rather than add an
auxiliary load-balance loss (which fights the policy-gradient objective), the
load balancer (`load_balance.py`) uses the **aux-loss-free** bias scheme
(DeepSeek 2408.15664): a per-expert routing bias is nudged each step toward
balanced utilization — up for under-used experts, down for over-used ones — by a
step size `load_balance_coeff`. The bias steers routing without contributing a
gradient term, so it never competes with the RL objective. With
`load_balance_coeff=null` (the default) it is off.

## EP-aware refit

The inference engine's fused-MoE kernel expects per-expert weight tensors
(`experts.{i}.{gate,up,down}_proj`), but the trainer holds them fused and
expert-parallel-sharded. The refit path (`refit.py`) re-expands the fused,
sharded representation back into the per-expert layout during weight sync, so the
inference fleet receives weights in the shape its kernel consumes. This is the
MoE-specific part of the [refit seam](../architecture/weight-sync.md).

## Validation

The pure parts — router/dispatch identity, the grouped expert math, the
load-balance bias update, the refit re-expansion round-trip — are CPU-tested. The
live expert-parallel all-to-all collective is GPU-only and tracked in the
hardware-deferred-validation ledger.
