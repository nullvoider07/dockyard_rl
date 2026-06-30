# Distillation

Distillation trains a student to match a frozen teacher. dockyard supports three
modes that differ in *what* the teacher scores and *whose* tokenizer is in play:
off-policy (teacher scores a fixed dataset), on-policy (teacher scores the
student's own rollouts), and cross-tokenizer (teacher and student tokenize
differently). All three reduce to a KL between teacher and student distributions;
the machinery differs in how the teacher signal is produced and transported.

## Off-policy distillation

The baseline path (`algorithms/distillation.py`). A frozen teacher scores a fixed
prompt/response dataset and emits, per position, its **top-k** log-probabilities
and indices; the student matches them with a top-k KL (`DistillationLossFn`,
`LossInputType.DISTILLATION`). Shipping only the top-k — not the full vocab —
keeps the teacher signal small enough to precompute or stream.

### Colocated mode

By default the student and teacher run on separate clusters. Colocated mode
places both on **one GPU mesh** to roughly halve the GPU count. The student
optimizer (~12 bytes/param) and the teacher weights are needed in **anti-phase**
windows — the teacher is resident only while it produces top-k logits, the
student optimizer only while the student trains — so they need not co-reside at
peak. Because a colocated mesh has no headroom to spare, a preflight memory
estimator (`algorithms/distillation_memory.py`) computes the per-phase per-GPU
peak from the model shapes and **refuses at startup** with an actionable
shortfall rather than OOMing mid-run. The arithmetic is pure (no CUDA), so the
safety property is unit-tested on CPU; the separate-cluster path is unchanged.

## On-policy distillation (OPD)

On-policy distillation (`algorithms/opd.py`) moves the teacher onto the
student's **own** rollouts inside async GRPO. The student generates on-policy;
one or more frozen, non-colocated teacher groups score the generated tokens via
log-probs; `OPDAdvantageEstimator` forms the token-level distillation advantage

```text
Â = stop_grad[ log π_teacher − log π_student ]
```

which flows into the same `ClippedPGLossFn` as every other estimator (selected by
`grpo.adv_estimator.name='opd'`; the heavy-tailed ratio is truncated by the
loss's ICE-POP mode). Multiple teachers are routed per sample by agent name, so a
mixed batch can be scored by different teachers (MOPD). Teachers run as DTensor
`Policy` worker groups on **dedicated fleets** (`STRICT_PACK`); when the cluster
exposes NVLink topology they are placed on
[NVLink segments](../architecture/cluster-and-fleets.md), otherwise on plain
dedicated clusters.

## Cross-tokenizer distillation (xtoken)

When the teacher's tokenizer differs from the student's, position-aligned KL is
meaningless — the two models segment the same text into different tokens. The
`algorithms/x_token/` subsystem bridges this:

- **Alignment** (`token_aligner.py`) — maps teacher token spans to student token
  spans, producing the chunk structure over which the KL is averaged.
- **Projection** (`loss_utils.py`) — the teacher's full-vocab logits are
  projected onto the student vocabulary through a sparse projection matrix, then
  reduced with a **chunk-averaged** cross-tokenizer KL so each aligned chunk
  contributes once regardless of how many tokens it spans (`LossInputType.
  DISTILLATION_CROSS_TOKENIZER`).
- **Transport** — rebuilding the teacher's full-vocab logits is bandwidth-heavy.
  Two transports are supported: node-local CUDA IPC (fast when teacher and
  student share a node) and a cross-cluster transport (so cross-tokenizer
  distillation works on a real multi-node cluster, not only single-node).

## Validation

The CPU-testable cores — the top-k KL math, the colocation memory accounting and
schedule selection, the `OPDAdvantageEstimator` advantage, the token alignment
and chunk-averaged projection math — are unit-tested. Live teacher scoring,
multi-GPU IPC transport, and the dedicated-fleet teacher placement are GPU /
multi-node and tracked in the hardware-deferred-validation ledger.
