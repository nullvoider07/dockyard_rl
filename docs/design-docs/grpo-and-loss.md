# GRPO and the clipped policy-gradient loss

GRPO (Group Relative Policy Optimization) replaces PPO's value network with a
**group-relative baseline**: for each prompt, several rollouts are sampled, and
each rollout's advantage is its reward minus the group's mean. There is no critic
to train, which removes a whole model and its failure modes from the loop.

## Advantage estimation

`algorithms/advantage_estimator.py` provides three estimators (`GRPOAdvantageEstimator`,
`GDPOAdvantageEstimator`, `ReinforcePlusPlusAdvantageEstimator`). The default
GRPO path:

- Groups the `num_generations_per_prompt` rollouts of each prompt.
- Computes a per-prompt baseline. With `use_leave_one_out_baseline`, each
  sample's baseline excludes itself (RLOO), which removes the bias a
  self-inclusive mean introduces in small groups.
- Optionally normalizes rewards (`normalize_rewards`) by the per-prompt standard
  deviation, and can clamp the result to `[advantage_clip_low, advantage_clip_high]`.

A degenerate group — every rollout earning the same reward — has zero advantage
and contributes no gradient; dynamic sampling (sync GRPO only) can resample to
avoid wasting the step on such groups.

### GDPO — multi-reward advantage

`GDPOAdvantageEstimator` (selected by `grpo.adv_estimator.name='gdpo'`)
generalizes GRPO to **multiple reward components**. Some environments score a
rollout on more than one axis — e.g. tests passing *and* the patch staying
lint-clean *and* a runtime budget, or several independent graders — exposed in
the batch as `reward1`, `reward2`, … Rather than collapse them to a single
scalar up front, GDPO:

- computes a GRPO-style per-prompt leave-one-out baseline for **each** component
  independently (optionally per-component std-normalized),
- **sums** the per-component advantages, then
- renormalizes the total to zero mean / unit standard deviation.

Keeping each objective on its own baseline before combining them stops a
component with a different scale or hit-rate from dominating the gradient. It
requires at least two reward components (otherwise it raises, pointing you back
to `grpo`). The name is not an expanded acronym in this codebase — read it as
"multi-reward GRPO". The combined advantage then flows into the same
`ClippedPGLossFn` as every other estimator.

## One loss, many algorithms

`ClippedPGLossFn` (`algorithms/loss/loss_functions.py`) is a single configurable
loss that expresses PPO, GRPO, RLOO, DAPO, GSPO, CISPO, and dual-clipping. The
core is the clipped surrogate:

```text
L(θ) = E_t[ min(r_t · A_t, clip(r_t, 1−ε_lo, 1+ε_hi) · A_t) ] − β · KL(π_θ ‖ π_ref)
```

with `r_t = π_θ(a_t) / π_θ_old(a_t)`. The configuration knobs (`loss_fn` in the
config) select the variant:

| Knob | Effect |
| --- | --- |
| `ratio_clip_min` / `ratio_clip_max` | The clip range ε; asymmetric values give the DAPO clip-higher behaviour. |
| `ratio_clip_c` | Dual-clip lower bound on negative-advantage terms; `null` disables it. |
| `token_level_loss` | Token-level vs. sequence-level reduction. |
| `sequence_level_importance_ratios` | GSPO-style sequence-level ratios (mutually exclusive with token-level loss). |
| `disable_ppo_ratio` | Drop the ratio entirely (REINFORCE). |
| `force_on_policy_ratio` | Pin the ratio to 1 (on-policy). |
| `use_cispo` | CISPO surrogate: keep every token in the gradient with a stop-gradient clipped weight instead of the pessimistic `min`/`max` (token-level only). |

### CISPO — keeping clipped tokens in the gradient

`use_cispo` selects the CISPO surrogate (MiniMax-M1) in place of the pessimistic
`min`/`max` clip. When the clip branch of `min(r_t · A_t, clip(r_t) · A_t)` wins,
that token's ratio is detached and it stops contributing gradient — so the very
tokens whose policy moved the most are silenced. CISPO instead keeps every token
but freezes its importance weight at the clipped value behind a stop-gradient:

```text
L_CISPO(θ) = − E_t[ A_t · stop_grad(clip(r_t, 1−ε_lo, 1+ε_hi)) · log π_θ(a_t) ]
```

The clipped ratio still scales each token's contribution, but because it enters
only through the stop-gradient, gradient always flows through the log-prob. This
preserves the low-probability, high-information tokens (often the decisive
reasoning steps) that the standard clip would zero out — the property CISPO was
introduced for. It is token-level only and mutually exclusive with the dual clip
(`ratio_clip_c`), `disable_ppo_ratio`, `force_on_policy_ratio`, and sequence-level
ratios; the loss asserts these at construction. Both trainer backends implement
it identically, value- and gradient-parity tested with CISPO on.

## Reference-KL penalty

A penalty toward the reference policy controls drift. `reference_policy_kl_type`
selects the estimator — `k1` (log-ratio), `k2`, or the low-variance unbiased
`k3` — and `reference_policy_kl_penalty` is its coefficient β.
`kl_input_clamp_value` / `kl_output_clamp_value` bound the estimator's inputs and
output so a single pathological token can't blow up the penalty. The KL can also
be folded into the reward (`use_kl_in_reward`) instead of the loss.

Because the penalty is an expectation *under the current policy*, its gradient
carries the score-function term, not only `∇KL` — and on the async path the
on-policy importance weight multiplies into it (clamp-saturated tokens have that
weight detached). Both trainer backends keep this term and `nan_to_num` the
result, so the penalty actually steers the policy rather than only shrinking the
divergence *estimate*; the torch/JAX KL gradients are parity-tested at
`reference_policy_kl_penalty>0`.

## Off-policy correction (async GRPO)

Async GRPO consumes rollouts generated by a slightly older policy, so the loss
must reweight by the importance ratio of the current to the generating policy
(`use_importance_sampling_correction`, mandatory when async is on). Because raw
IS ratios are heavy-tailed, **truncated importance sampling** caps them:

| `truncated_importance_sampling_type` | Behaviour |
| --- | --- |
| `tis` | Truncate the ratio at `truncated_importance_sampling_ratio`. |
| `icepop` | Two-sided clamp using the additional `..._ratio_min` floor. |
| `seq-mask-tis` | Sequence-masking variant; incompatible with sequence-level IS ratios. |

Truncated IS requires `use_importance_sampling_correction=True`; the loss
function asserts the valid combinations at construction so a misconfiguration
fails immediately.

## Diagnostics

The loss returns a metrics dict alongside the scalar — clip fractions, the KL
estimate, the importance-ratio statistics, and a staleness/sequence-logprob
error signal (`grpo.seq_logprob_error_threshold`) that flags when the trainer's
recomputed log-probs diverge too far from the generating engine's, the canonical
symptom of a broken refit or a tokenizer mismatch.
