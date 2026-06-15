# HLE (Humanity's Last Exam) verifier environment — single-turn, no sandbox.
#
# The agent answers a question in the fixed HLE format (Explanation / Answer /
# Confidence); this environment grades the extracted answer against the gold
# answer with an LLM judge when one is configured, falling back to a normalized
# exact-match otherwise (rewards/hle_grader.py). The episode ends after this one
# scoring step. Intended as a VALIDATION benchmark (judge cost is bounded to
# validation passes); training on the 2,500-question exam is not recommended.

from concurrent.futures import ThreadPoolExecutor
from typing import Any, NotRequired, Optional, TypedDict, cast

import ray
import torch

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from dockyard_rl.rewards.hle_grader import HLEJudgeClient, grade

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dockyard_rl.data.interfaces import LLMMessageLogType
else:
    try:
        from dockyard_rl.data.interfaces import LLMMessageLogType
    except ImportError:
        LLMMessageLogType = list  # type: ignore


class HLEEnvConfig(TypedDict):
    # Actor concurrency for servicing parallel step() calls (async rollout path).
    max_concurrency: NotRequired[int]
    # Grading threads per step() (judge calls are I/O-bound).
    num_threads: NotRequired[int]
    # Judge config; each falls back to its DOCKYARD_HLE_JUDGE_* env var, and the
    # judge is disabled (→ exact-match) unless both base_url and model resolve.
    judge_base_url: NotRequired[Optional[str]]
    judge_api_key: NotRequired[Optional[str]]
    judge_model: NotRequired[Optional[str]]
    judge_timeout: NotRequired[Optional[float]]


@ray.remote(  # type: ignore[call-overload]
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class HLEEnvironment(EnvironmentInterface):
    """Single-turn exact-match / LLM-judge verifier environment for HLE."""

    def __init__(self, cfg: HLEEnvConfig):
        self.cfg = cfg
        self.num_threads = int(cfg.get("num_threads", 16))
        self._judge = HLEJudgeClient(
            base_url=cfg.get("judge_base_url"),
            api_key=cfg.get("judge_api_key"),
            model=cfg.get("judge_model"),
            timeout=cfg.get("judge_timeout"),
        )

    def _grade_one(self, response: str, meta: dict) -> tuple[float, Optional[str]]:
        return grade(
            response,
            str(meta.get("ground_truth", "")),
            str(meta.get("question", "")),
            judge=self._judge,
        )

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[dict],
    ) -> EnvironmentReturn:
        # Concatenate each sample's assistant turns (single-turn, but be robust).
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

        scores = [g[0] for g in graded]
        answers = [g[1] for g in graded]
        rewards = torch.tensor(scores, dtype=torch.float32)
        terminateds = torch.ones(len(scores), dtype=torch.bool)
        observations = [
            {"role": "environment",
             "content": "Environment: correct" if s >= 1.0 else "Environment: incorrect"}
            for s in scores
        ]
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=cast(Any, next_stop_strings),
            rewards=rewards,
            terminateds=terminateds,
            answers=answers,
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
