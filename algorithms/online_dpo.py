"""Online / iterative DPO (Tier 2).

Replaces the static preference dataset with an on-policy loop: generate K
candidates per prompt from the current policy, score them with a judge, build
chosen/rejected pairs (best-vs-worst with a score-margin gate), and run a DPO
step on the existing preference loss — then repeat. Targets security/honesty by
reusing the judge infrastructure as the preference labeler (decision #4), with a
verifiable RewardFunction preferred where one exists (anti-reward-hacking,
mirroring IntegrityReward). The frozen-reference KL anchor stays active via the
selected DPO-family loss, so capability is protected.

Split of responsibilities (no-cluster validation posture):
  - CPU-testable core (this module): the preference judge + verifiable fallback,
    best-vs-worst pair construction with the margin gate, pair→preference-datum
    conversion, the over-optimization monitor, and the per-iteration driver. All
    unit-tested.
  - GPU-deferred wiring (HV): the loop consumes ``generate_fn`` (policy generation
    of K candidates + decode) and ``train_fn`` (add_ref_logprobs + policy.train +
    weight sync) as injected callables. Binding them to the real inference/trainer
    fleets and measuring live throughput is the bring-up step.

The DPO step reuses the D0–D4 loss factory unchanged; ``online_dpo`` only
produces the preference batch and controls the loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, NotRequired, Optional, TypedDict

import torch

from dockyard_rl.data.interfaces import PreferenceDatumSpec

# Candidate generation returns groups of K candidates, one group per prompt.
CandidateGroups = list[list["Candidate"]]
GenerateFn = Callable[[list[Any]], CandidateGroups]
BuildBatchFn = Callable[[list[PreferenceDatumSpec]], Any]
TrainFn = Callable[[Any], dict[str, Any]]


class OnlineDPOConfig(TypedDict):
    """Configuration for the online/iterative DPO loop."""

    candidates_per_prompt: int       # K samples generated per prompt
    score_margin: float              # drop a pair if |score_chosen − score_rejected| < this
    prefer_verifiable: bool          # use a verifiable RewardFunction when applicable
    regeneration_period: int         # generate fresh candidates every N steps (1 = fully on-policy)
    max_num_steps: int

    kl_target: NotRequired[float]    # over-optimization monitor threshold (policy↔ref KL)
    # Judge backend (reuses the shared OpenAI-compatible HLEJudgeClient transport).
    judge_base_url: NotRequired[str]
    judge_api_key: NotRequired[str]
    judge_model: NotRequired[str]
    judge_timeout: NotRequired[float]
    judge_temperature: NotRequired[float]


@dataclass
class Candidate:
    """One generated candidate for a prompt.

    ``prompt_token_ids`` / ``response_token_ids`` build the preference message log;
    the text fields feed the judge; ``trajectory`` (optional) lets a verifiable
    RewardFunction score the candidate instead of the LLM judge.
    """

    prompt_token_ids:   torch.Tensor
    response_token_ids: torch.Tensor
    prompt_text:        str = ""
    response_text:      str = ""
    score:              Optional[float] = None
    trajectory:         Optional[dict] = field(default=None)


_DEFAULT_SCORE_TEMPLATE = (
    "You are evaluating an AI assistant's response for honesty, harmlessness, and "
    "helpfulness.\n\n"
    "Prompt:\n{prompt}\n\nResponse:\n{response}\n\n"
    "Rate the response from 0 to 100, where higher means more honest, safe, and "
    "helpful. Reply with exactly one line:\nSCORE: <number>"
)
_SCORE_RE = re.compile(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_judge_score(content: Optional[str]) -> Optional[float]:
    """Parse ``SCORE: <0-100>`` from a judge reply → a float in [0, 1], or None."""
    if not content:
        return None
    m = _SCORE_RE.search(content)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, value / 100.0))


class PreferenceJudge:
    """Scores a (prompt, response) in [0, 1] for online preference construction.

    Reuses the shared LLM-judge transport (``judge.chat``); where a verifiable
    RewardFunction is supplied and applicable to the candidate's trajectory, that
    ground-truth signal is preferred (``prefer_verifiable``) — the anti-hacking
    posture from decision #4. Returns None when neither source can score.
    """

    def __init__(
        self,
        judge: Optional[Any] = None,
        *,
        score_template: str = _DEFAULT_SCORE_TEMPLATE,
        verifiable_reward: Optional[Callable[[dict], Any]] = None,
        prefer_verifiable: bool = True,
        parse_fn: Callable[[Optional[str]], Optional[float]] = parse_judge_score,
        temperature: float = 0.0,
    ) -> None:
        self._judge = judge
        self._template = score_template
        self._verifiable = verifiable_reward
        self._prefer_verifiable = prefer_verifiable
        self._parse = parse_fn
        self._temperature = temperature

    @property
    def enabled(self) -> bool:
        judge_on = self._judge is not None and getattr(self._judge, "enabled", False)
        return bool(self._verifiable is not None or judge_on)

    def _verifiable_score(self, trajectory: Optional[dict]) -> Optional[float]:
        if self._verifiable is None or trajectory is None:
            return None
        result = self._verifiable(trajectory)
        if getattr(result, "status", None) == "ok":
            return float(result.reward)
        return None

    def _judge_score(self, prompt: str, response: str) -> Optional[float]:
        if self._judge is None or not getattr(self._judge, "enabled", False):
            return None
        content = self._judge.chat(
            [{"role": "user", "content": self._template.format(prompt=prompt, response=response)}],
            temperature=self._temperature,
        )
        return self._parse(content)

    def score(
        self,
        prompt: str,
        response: str,
        *,
        trajectory: Optional[dict] = None,
    ) -> Optional[float]:
        if self._prefer_verifiable:
            v = self._verifiable_score(trajectory)
            if v is not None:
                return v
            return self._judge_score(prompt, response)
        j = self._judge_score(prompt, response)
        if j is not None:
            return j
        return self._verifiable_score(trajectory)


def build_preference_pairs(
    candidate_groups: CandidateGroups,
    score_margin: float,
) -> tuple[list[tuple[Candidate, Candidate]], dict[str, int]]:
    """Best-vs-worst pair construction with a score-margin gate.

    For each prompt's candidate group, pair the highest- and lowest-scored
    candidate, dropping the group when fewer than two candidates are scored or
    when the score gap is below ``score_margin`` (a judge-noise guard). Returns
    the ``(chosen, rejected)`` pairs and counts.
    """
    pairs: list[tuple[Candidate, Candidate]] = []
    n_dropped_unscored = 0
    n_dropped_margin = 0
    for group in candidate_groups:
        scored = [c for c in group if c.score is not None]
        if len(scored) < 2:
            n_dropped_unscored += 1
            continue
        best = max(scored, key=lambda c: c.score)   # type: ignore[arg-type,return-value]
        worst = min(scored, key=lambda c: c.score)  # type: ignore[arg-type,return-value]
        assert best.score is not None and worst.score is not None
        if best.score - worst.score < score_margin:
            n_dropped_margin += 1
            continue
        pairs.append((best, worst))
    metrics = {
        "num_groups": len(candidate_groups),
        "num_pairs": len(pairs),
        "num_dropped_unscored": n_dropped_unscored,
        "num_dropped_margin": n_dropped_margin,
    }
    return pairs, metrics


def pairs_to_preference_data(
    pairs: list[tuple[Candidate, Candidate]],
    *,
    start_idx: int = 0,
) -> list[PreferenceDatumSpec]:
    """Convert ``(chosen, rejected)`` candidate pairs into PreferenceDatumSpecs.

    The output feeds ``preference_collate_fn`` unchanged, so the online path
    produces exactly the paired batch the offline DPO step already consumes.
    """
    data: list[PreferenceDatumSpec] = []
    for i, (chosen, rejected) in enumerate(pairs):
        chosen_log = [
            {"role": "user", "token_ids": chosen.prompt_token_ids},
            {"role": "assistant", "token_ids": chosen.response_token_ids},
        ]
        rejected_log = [
            {"role": "user", "token_ids": rejected.prompt_token_ids},
            {"role": "assistant", "token_ids": rejected.response_token_ids},
        ]
        data.append(
            PreferenceDatumSpec(
                message_log_chosen=chosen_log,
                message_log_rejected=rejected_log,
                length_chosen=int(chosen.prompt_token_ids.shape[0] + chosen.response_token_ids.shape[0]),
                length_rejected=int(rejected.prompt_token_ids.shape[0] + rejected.response_token_ids.shape[0]),
                loss_multiplier=1.0,
                idx=start_idx + i,
            )
        )
    return data


class JudgeOveroptMonitor:
    """Tracks judge-score drift and policy↔reference KL as over-optimization signals.

    Online preference training can hack the judge: judge scores climb while the KL
    to the reference grows. This records the running judge-score mean and flags
    when the measured KL exceeds ``kl_target`` (the loss's β still penalizes KL;
    this is the monitoring guard on top).
    """

    def __init__(self, kl_target: Optional[float] = None) -> None:
        self.kl_target = kl_target
        self.num_updates = 0
        self.running_mean_score = 0.0

    def update(
        self,
        mean_judge_score: Optional[float],
        kl: Optional[float] = None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if mean_judge_score is not None:
            self.num_updates += 1
            self.running_mean_score += (
                mean_judge_score - self.running_mean_score
            ) / self.num_updates
            metrics["judge_score_mean"] = mean_judge_score
            metrics["judge_score_running_mean"] = self.running_mean_score
        if kl is not None:
            metrics["kl"] = kl
            # Only report the guard when a target is actually configured.
            if self.kl_target is not None:
                metrics["kl_over_target"] = bool(kl > self.kl_target)
        return metrics


def run_online_dpo_iteration(
    prompts: list[Any],
    *,
    generate_fn: GenerateFn,
    judge: PreferenceJudge,
    score_margin: float,
    build_batch_fn: BuildBatchFn,
    train_fn: TrainFn,
    monitor: Optional[JudgeOveroptMonitor] = None,
    start_idx: int = 0,
) -> dict[str, Any]:
    """One online DPO iteration: generate → score → pair → DPO step.

    ``generate_fn`` returns K candidates per prompt; ``build_batch_fn`` is the
    preference collate (partial); ``train_fn`` runs the reference-logprob +
    policy.train + weight-sync step and returns its metrics. Skips training (no
    step) when the margin gate leaves no usable pairs.
    """
    candidate_groups = generate_fn(prompts)

    scores: list[float] = []
    for group in candidate_groups:
        for c in group:
            if c.score is None:
                c.score = judge.score(c.prompt_text, c.response_text, trajectory=c.trajectory)
            if c.score is not None:
                scores.append(c.score)

    pairs, metrics_int = build_preference_pairs(candidate_groups, score_margin)
    metrics: dict[str, Any] = dict(metrics_int)
    mean_score = sum(scores) / len(scores) if scores else None
    if mean_score is not None:
        metrics["judge_score_mean"] = mean_score

    if not pairs:
        metrics["trained"] = False
        return metrics

    data = pairs_to_preference_data(pairs, start_idx=start_idx)
    batch = build_batch_fn(data)
    train_metrics = train_fn(batch) or {}
    metrics["trained"] = True
    metrics.update({f"train/{k}": v for k, v in train_metrics.items()})

    if monitor is not None:
        mon = monitor.update(mean_score, kl=train_metrics.get("kl"))
        metrics.update({f"monitor/{k}": v for k, v in mon.items()})
    return metrics


def online_dpo_train(
    prompt_batches: list[list[Any]],
    *,
    generate_fn: GenerateFn,
    judge: PreferenceJudge,
    build_batch_fn: BuildBatchFn,
    train_fn: TrainFn,
    config: OnlineDPOConfig,
    monitor: Optional[JudgeOveroptMonitor] = None,
    logger: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Drive the online DPO loop over a sequence of prompt batches.

    Composes the injected generation/judge/train callables; the callables bind to
    the real inference/trainer fleets at bring-up (HV). Returns per-step metrics.
    Honors ``max_num_steps``; ``regeneration_period`` is reserved for buffered
    (slightly off-policy) reuse and currently regenerates every step.
    """
    if monitor is None and config.get("kl_target") is not None:
        monitor = JudgeOveroptMonitor(kl_target=config.get("kl_target"))

    score_margin = config["score_margin"]
    max_steps = config["max_num_steps"]
    all_metrics: list[dict[str, Any]] = []
    consumed_pairs = 0
    for step, prompts in enumerate(prompt_batches):
        if step >= max_steps:
            break
        metrics = run_online_dpo_iteration(
            prompts,
            generate_fn=generate_fn,
            judge=judge,
            score_margin=score_margin,
            build_batch_fn=build_batch_fn,
            train_fn=train_fn,
            monitor=monitor,
            start_idx=consumed_pairs,
        )
        metrics["step"] = step
        consumed_pairs += int(metrics.get("num_pairs", 0))
        if logger is not None:
            logger.log_metrics(metrics, step, prefix="online_dpo")
        all_metrics.append(metrics)
    return all_metrics
