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

## Router replay (R3)

An MoE router selects top-K experts per token. Re-running that router at
train/log-prob time can pick *different* experts than generation did — batch
composition, padding, and kernel/dtype nondeterminism all perturb the gate — so
the same token gets a different log-prob than it was generated with. That biases
the off-policy importance-sampling correction and inflates the
train/generation log-prob error dockyard uses to filter stale rollouts. Router
replay removes this nondeterminism by forcing the trainer's MoE forward to reuse
generation's recorded expert selection. It is gated by
`policy.router_replay.enabled` (default `false`) and is a no-op on dense models.

- **Capture** (`models/generation/vllm/router_capture.py`) — vLLM exposes the
  per-token selection on `CompletionOutput.routed_experts`
  (`[tokens, num_moe_layers, top_k]`) when the engine is built with
  `enable_return_routed_experts=True`. Capture aligns it onto the trainer's
  right-padded `[padded_length, L, top_k]` layout. Routing is a **next-token**
  quantity — the route at position `i` drove the prediction of token `i+1` — so
  only the first `valid_length − 1` positions carry a real route; the final token
  and padding take the identity route `arange(top_k)`, and genuinely-missing
  interior routes (rare; prefix-caching + chunked prefill) are flagged with a
  sentinel so the consumer rejects rather than silently mis-replays them.
- **Replay** (`models/dtensor/moe/router_replay.py`) — the recorded routing
  `[B, T, L_moe, K]` rides the rollout data dict into the trainer.
  `bind_router_replay` walks the model's `MoEBlock` modules in module order and
  binds each block's per-layer route slice (set-and-consume); the router
  (`router.py`) consumes the bound slice, forcing `topk_expert_ids` to the
  recorded ids while the **gating weights still come from the train model's own
  gate, so the gradient is intact**. `L_moe` must equal the number of `MoEBlock`
  modules — a mismatch is a hard error rather than a silent mis-bind.

A compatibility guard (`validate_router_replay_generation_compat`) fails fast on
generation/orchestration configs that cannot surface aligned routing (e.g.
pipeline parallelism or async scheduling) instead of silently degrading.

## Validation

The pure parts — router/dispatch identity, the grouped expert math, the
load-balance bias update, the refit re-expansion round-trip, and the
router-replay capture-alignment + set-and-consume binding — are CPU-tested. The
live expert-parallel all-to-all collective and live capture + multi-rank replay
(success metric: the log-prob error drops for an MoE policy) are GPU-only and
tracked in the hardware-deferred-validation ledger.
