# Preference optimization (DPO)

Alongside GRPO, dockyard_rl supports preference-based post-training: Direct
Preference Optimization and its online variant. DPO trains directly on
`(chosen, rejected)` pairs without an explicit reward model or an RL loop, by
making the policy assign higher likelihood to the chosen response than a frozen
reference does, and lower to the rejected one.

The loop is in `algorithms/dpo.py`; the loss is `DPOLossFn`
(`algorithms/loss/loss_functions.py`).

## The preference loss

`PreferenceLossFn` is the base class; chosen and rejected examples are
interleaved in the batch (even/odd rows) and split inside the loss. The core
operates on the **reward delta** — the difference between the chosen and rejected
implicit rewards (each a β-scaled log-ratio of policy to reference) — passed
through a log-sigmoid:

```text
L = − log σ( β · (r_chosen − r_rejected) )
```

The implementation is parameterized so a single core expresses several published
variants, each defaulting to the identity so the base DPO path is byte-identical
when unused:

| Knob | Variant |
| --- | --- |
| `extra_margin` (added before β) | DPOP's positive-penalty term. |
| `pre_sigmoid_offset` (added after β) | R-DPO's length penalty. |
| `label_smoothing` ε | cDPO (mixes the flipped target). |
| `reference_free` | Drops the reference term (SimPO/ORPO-style). |

`DPOLossConfig` also carries a `preference_loss_weight` and an
`sft_loss_weight`: a positive SFT weight adds a supervised NLL term on the chosen
response, the standard DPO+SFT blend that keeps the policy from drifting off the
data distribution while it learns the preference.

`build_preference_loss` selects the concrete loss from the registry, and
`is_reference_free` lets the loop skip computing reference log-probs entirely when
the chosen loss doesn't need them.

## Reference-free vs. reference-anchored

When the loss is reference-anchored, the loop computes the frozen reference's
log-probs for both responses (once, since the reference doesn't change).
Reference-free losses skip that pass — cheaper, at the cost of the explicit
KL anchor that a reference provides.

## Online DPO

`algorithms/online_dpo.py` closes the loop: instead of a fixed offline preference
dataset, it **generates** candidate responses on-policy and forms preference
pairs from them (scored by a judge or reward signal), then applies the same DPO
loss. This reuses the generation and reward infrastructure the GRPO path already
provides, putting online DPO between offline DPO and full GRPO on the
on-policy/off-policy spectrum.

Data comes through the preference dataset loaders (`data/datasets/preference_datasets/`)
and `preference_collate_fn`. Evaluation helpers live in `algorithms/dpo_evals.py`.
