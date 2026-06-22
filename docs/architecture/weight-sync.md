# Weight synchronization

The trainer's weights advance every optimizer step; the inference fleet only
sees new weights on a **refit**. The `WeightSynchronizer` abstraction
(`weight_sync/`) owns that transfer and hides the transport and topology from the
GRPO loop, which never branches on backend type or colocation.

## The interface

`WeightSynchronizer` (`weight_sync/interfaces.py`) is a small ABC the loop drives
explicitly:

| Method | Role |
| --- | --- |
| `init_communicator()` | Called once at setup, after workers exist. Colocated transports prepare refit metadata; the collective transport also initializes its process group. |
| `mark_stale()` | Called after every training step. Flags that the inference weights are behind. |
| `is_stale` | Property — `True` between `mark_stale()` and the next successful `sync_weights()`. |
| `sync_weights()` | Transfers the latest policy weights into the generation backend. |
| `shutdown()` | Releases communication resources. |

The loop's usage is the **staleness handshake**: after an optimizer step it
calls `mark_stale()`; before generating, if `is_stale`, it calls
`sync_weights()` and then `prepare_for_generation()` on the engine.

The interface assumes **global** weight updates — all generation workers update
atomically and share one weight version. In async GRPO, heterogeneous weight
ages are tracked per-sample in the replay buffer (`target_weight_versions`), not
here.

## Transports

`create_weight_synchronizer()` (`weight_sync/factory.py`) selects the transport
from the deployment topology and the generation backend:

| Topology | Backend | Transport | Class |
| --- | --- | --- | --- |
| Non-colocated | vLLM | NCCL collective | `CollectiveWeightSynchronizer` |
| Colocated | vLLM | IPC / ZMQ | `IPCWeightSynchronizer` |
| Colocated | SGLang | HTTP | `HTTPWeightSynchronizer` |

Non-colocated SGLang is not supported (the factory raises `NotImplementedError`).
The non-colocated path additionally requires `train_cluster` and
`inference_world_size`.

### Colocated vs. non-colocated lifecycle

This distinction shapes what `sync_weights()` does:

- **Colocated transports (IPC, HTTP)** own GPU phase transitions internally —
  offload optimizer state before the refit, `prepare_for_generation`, offload
  again after — because the policy and the engine share GPUs and must hand the
  device back and forth.
- **The NCCL collective transport is a pure data mover.** Policy and generation
  run on separate GPU clusters, so there are no phase transitions; it only
  forwards weights (and optional FP8 KV-cache scales) over the collective.

`sync_weights()` accepts an optional `kv_scales` dict; only the collective
transport honors it, forwarding to `policy.broadcast_weights_for_collective()`.
IPC and HTTP ignore it.

## The refit seam

Whatever the transport, the trainer must expose its weights in the layout the
inference engine expects — HF parameter names and shapes. The policy worker
implements a **refit seam** that maps its internal representation to that layout:

- The **DTensor** backend gathers sharded `DTensor` parameters and emits the HF
  state dict.
- The **JAX** backend converts NNX arrays (via dlpack on GPU, numpy on CPU) and
  applies a **model-specific** inverse of that model's HF loader. The dense path
  transposes linear kernels back to HF layout; Qwen3-MoE additionally re-expands
  fused expert tensors into the per-expert `experts.{i}.{gate,up,down}_proj`
  layout the inference engine's fused-MoE kernel consumes; Qwen3-Next adds its
  linear-attention parameters; and Gemma 4 maps its multi-parameter router and
  per-layer embedding table, keeping the experts **fused** because that is the
  layout its checkpoint and engine use. Each map is the exact inverse of the
  model's loader, so a load→refit round-trip reproduces the HF state dict.

`prepare_refit_info()` reports the `{hf_name: (shape, dtype)}` map the collective
producer uses to pack the broadcast; the consumer side unpacks it into the
engine's weights. The NCCL broadcast into a live engine is exercised on hardware
(see `handoff/hardware-deferred-validation.md`); the name-map inverse and the
JAX→torch value round-trip are covered by CPU tests.
