# GDPval verifier environment (text-deliverable path) — single-turn, no sandbox.
#
# The model produces a textual deliverable for a GDPval task; this environment
# scores it against the task's rubric_json with an LLM judge (GDPvalRubricReward).
# The episode ends after this one scoring step. Rubric grading is judgmental and
# has NO deterministic fallback, so a judge endpoint must be configured (else
# every sample is an execution_error). Intended primarily as an eval/validation
# signal — GDPval is 220 long-horizon, file-oriented tasks; this is the reduced
# text path (the faithful agentic file-producing env is deferred).

from concurrent.futures import ThreadPoolExecutor
from typing import Any, NotRequired, Optional, TypedDict, cast

import ray
import torch

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from dockyard_rl.rewards.gdpval_rubric import GDPvalRubricReward

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dockyard_rl.data.interfaces import LLMMessageLogType
else:
    try:
        from dockyard_rl.data.interfaces import LLMMessageLogType
    except ImportError:
        LLMMessageLogType = list  # type: ignore


class GDPvalEnvConfig(TypedDict):
    # "weighted" → weighted fraction of rubric criteria met; "binary" → all-met.
    reward_mode: NotRequired[str]
    # Actor concurrency for servicing parallel step() calls (async rollout path).
    max_concurrency: NotRequired[int]
    # Grading threads per step() (judge calls are I/O-bound).
    num_threads: NotRequired[int]
    # Judge config; each falls back to DOCKYARD_GDPVAL_JUDGE_* then the shared HLE
    # judge env vars. A judge is REQUIRED (no deterministic rubric fallback).
    judge_base_url: NotRequired[Optional[str]]
    judge_api_key: NotRequired[Optional[str]]
    judge_model: NotRequired[Optional[str]]
    judge_timeout: NotRequired[Optional[float]]


@ray.remote(  # type: ignore[call-overload]
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class GDPvalEnvironment(EnvironmentInterface):
    """Single-turn rubric-judge verifier environment for GDPval (text path)."""

    def __init__(self, cfg: GDPvalEnvConfig):
        self.cfg = cfg
        self.num_threads = int(cfg.get("num_threads", 8))
        self._reward = GDPvalRubricReward(
            reward_mode=cfg.get("reward_mode", "weighted"),
            base_url=cfg.get("judge_base_url"),
            api_key=cfg.get("judge_api_key"),
            model=cfg.get("judge_model"),
            timeout=cfg.get("judge_timeout"),
        )

    def _grade_one(self, response: str, meta: dict) -> tuple[float, str, Optional[str]]:
        outcome = self._reward({
            "deliverable": response,
            "prompt":      meta.get("prompt", ""),
            "rubric_json": meta.get("rubric_json", ""),
        })
        return float(outcome.reward), outcome.status, outcome.failure_reason

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[dict],
    ) -> EnvironmentReturn:
        responses = [
            "".join(
                str(m["content"]) for m in conv if m.get("role") == "assistant"
            )
            for conv in message_log_batch
        ]
        pairs = list(zip(responses, metadata))

        if len(pairs) <= 1:
            graded = [self._grade_one(r, m) for r, m in pairs]
        else:
            with ThreadPoolExecutor(max_workers=min(len(pairs), self.num_threads)) as pool:
                graded = list(pool.map(lambda rm: self._grade_one(rm[0], rm[1]), pairs))

        rewards = torch.tensor([g[0] for g in graded], dtype=torch.float32)
        terminateds = torch.ones(len(graded), dtype=torch.bool)
        observations = [
            {"role": "environment", "content": f"Environment: {g[2] or 'graded'}"}
            for g in graded
        ]
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=cast(Any, next_stop_strings),
            rewards=rewards,
            terminateds=terminateds,
            answers=None,
        )

    def shutdown(self) -> None:
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        rewards = (
            batch["rewards"] if batch["rewards"].ndim == 1 else batch["rewards"][:, 0]
        )
        if "is_end" in batch:
            rewards = rewards * batch["is_end"]
        metrics = {
            "accuracy": rewards.mean().item(),
            "num_problems_in_batch": int(rewards.shape[0]),
        }
        return batch, metrics
