# Distillation

Distillation trains a student to match a frozen teacher. In dockyard the student
always generates **on-policy** — every mode reuses the GRPO rollout +
environment infrastructure, so the teacher scores the student's *own* rollouts.
The modes differ in the **teacher signal** and how it enters the loss: a direct
KL on the teacher's top-k logits, a teacher-log-prob advantage routed through the
policy-gradient loss, or a cross-tokenizer projection when the two tokenizers
disagree.

## Logit distillation

The distillation loop (`algorithms/distillation.py`). The student generates
rollouts against an environment; the frozen teacher scores those tokens and
supplies its **top-k** log-probabilities and indices (`num_topk_logits`); the
student is trained to match the teacher's distribution with a top-k KL
(`DistillationLossFn`, `LossInputType.DISTILLATION`). Shipping only the top-k —
not the full vocab — keeps the teacher signal small enough to transport each
step.

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

## Advantage distillation (OPD)

On-policy distillation (`algorithms/opd.py`) keeps the same on-policy rollouts but
changes the teacher signal: instead of a direct KL on logits, the teacher scores
the student's tokens with per-token **log-probs**, and `OPDAdvantageEstimator`
turns the gap into a token-level advantage

```text
Â = stop_grad[ log π_teacher − log π_student ]
```

which flows into the same `ClippedPGLossFn` as every other estimator (selected by
`grpo.adv_estimator.name='opd'`; the heavy-tailed ratio is truncated by the
loss's ICE-POP mode). Casting distillation as an *advantage* rather than a
separate loss lets it compose with the rest of the GRPO machinery. Multiple
teachers are routed per sample by agent name (MOPD), so a mixed batch can be
scored by different teachers. Teachers run as DTensor `Policy` worker groups on
**dedicated fleets** (`STRICT_PACK`); when the cluster exposes NVLink topology
they are placed on [NVLink segments](../architecture/cluster-and-fleets.md),
otherwise on plain dedicated clusters.

## Cross-tokenizer distillation (xtoken)

When the teacher's tokenizer differs from the student's, position-aligned KL is
meaningless — the two models segment the same text into different tokens. The
`algorithms/x_token/` subsystem bridges this, and generalizes to **several
teachers** distilled into one student at once (the single-teacher case is just a
one-entry teacher list):

- **Alignment** (`token_aligner.py`) — maps teacher token spans to student token
  spans, producing the chunk structure over which the KL is averaged. Each
  cross-tokenizer teacher is aligned independently.
- **Projection** (`loss_utils.py`) — a teacher's full-vocab logits are projected
  onto the student vocabulary through *its own* sparse projection matrix, then
  reduced with a **chunk-averaged** cross-tokenizer KL so each aligned chunk
  contributes once regardless of how many tokens it spans
  (`LossInputType.DISTILLATION_CROSS_TOKENIZER`). A teacher that *shares* the
  student's tokenizer sets no projection matrix (`projection_matrix_path: null`)
  and takes a direct top-k KL with no alignment.
- **Multi-teacher aggregation** — each teacher contributes a KD term; the terms
  combine per `kd_loss_mode`: a **weighted sum** (static per-teacher `weight`, or
  dynamic weights from a `sum_weights_metric` of teacher CE / entropy / max-prob
  softmaxed at temperature `alpha`), an **averaged-logits** convex combination of
  same-tokenizer teachers followed by one KL, or **select-teacher** (only the
  lowest-CE teacher). The aggregate KD is combined with a single student
  cross-entropy; per-teacher metrics are logged with a `_t{i}` suffix.
- **Transport** — rebuilding each teacher's full-vocab logits is bandwidth-heavy.
  Two transports are supported per teacher: node-local CUDA IPC (fast when
  teacher and student share a node) and a cross-cluster transport (so
  cross-tokenizer distillation works on a real multi-node cluster, not only
  single-node).

## Validation

The CPU-testable cores — the top-k KL math, the colocation memory accounting and
schedule selection, the `OPDAdvantageEstimator` advantage, the token alignment
and chunk-averaged projection math, and the multi-teacher aggregation modes and
dynamic weighting — are unit-tested. Live teacher scoring, multi-GPU IPC
transport, and the dedicated-fleet teacher placement are GPU / multi-node and
tracked in the hardware-deferred-validation ledger.
