import itertools
import sys
from typing import Any, NotRequired, TypedDict
import ray
import torch
from dockyard_rl.data.interfaces import LLMMessageLogType
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from dockyard_rl.environments.metrics import calculate_pass_rate_per_prompt
from dockyard_rl.environments.utils import chunk_list_to_workers

class CodeJaccardEnvConfig(TypedDict):
    num_workers: int
    stop_strings: NotRequired[list[str] | None]

class CodeJaccardEnvironmentMetadata(TypedDict):
    ground_truth: str

@ray.remote  # pragma: no cover
class CodeJaccardVerifyWorker:
    """Worker that computes Jaccard similarity between predicted code and ground truth."""

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
    ) -> list[float]:
        """Compute Jaccard similarity reward for each (prediction, ground_truth) pair.

        Jaccard similarity is computed over the whitespace-tokenized word sets of the
        two strings. This gives a soft, differentiable-friendly reward signal that
        avoids the binary pass/fail cliff of test-execution-based rewards, and is
        useful during early training when the model's code quality is too low to pass
        any tests.

        Args:
            pred_responses: Predicted response strings from the LLM.
            ground_truths:  Ground truth strings to compare against.

        Returns:
            List of Jaccard similarity scores in [0, 1].
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                pred_words = set(response.split())
                gt_words = set(ground_truth.split())
                intersection = pred_words & gt_words
                union = pred_words | gt_words
                score = len(intersection) / len(union) if union else 0.0
                results.append(float(score))
            except Exception:
                results.append(0.0)
        return results

@ray.remote(  # type: ignore[call-overload]
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class CodeJaccardEnvironment(EnvironmentInterface[CodeJaccardEnvironmentMetadata]):
    """Environment that scores code generation via Jaccard similarity.

    This is a soft-reward alternative to binary test execution. Useful when:
    - Test runners are not yet set up for a new task domain.
    - Early-stage training where the model cannot yet pass any tests.
    - Tasks where partial credit (overlapping tokens) is a meaningful signal.

    Rewards are continuous in [0, 1]. Episodes always terminate after one step.
    """

    def __init__(self, cfg: CodeJaccardEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self._worker_counter = itertools.count()
        self.workers = [
            CodeJaccardVerifyWorker.options(  # type: ignore[attr-defined]
                runtime_env={"py_executable": sys.executable}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)  # type: ignore[arg-type]

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CodeJaccardEnvironmentMetadata],
    ) -> EnvironmentReturn[CodeJaccardEnvironmentMetadata]:
        """Score a batch of model completions via Jaccard similarity.

        Args:
            message_log_batch: Batch of conversation message logs. The last assistant
                message in each log is taken as the prediction.
            metadata: Per-sample metadata dicts; must contain a ``"ground_truth"`` key.

        Returns:
            EnvironmentReturn with scalar rewards in [0, 1] and all episodes terminated.
        """
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_responses = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        worker_index = next(self._worker_counter) % self.num_workers
        futures = [
            self.workers[(worker_index + i) % self.num_workers].verify.remote(  # type: ignore[union-attr]
                chunk, gt_chunk
            )
            for i, (chunk, gt_chunk) in enumerate(
                zip(chunked_responses, chunked_ground_truths)
            )
        ]

        results = [
            item for sublist in ray.get(futures) for item in sublist
        ]

        observations = [
            {
                "role": "environment",
                "content": f"Environment: jaccard_score={result:.4f}",
            }
            for result in results
        ]

        rewards = torch.tensor(results, dtype=torch.float32).cpu()
        done = torch.ones_like(rewards).cpu()
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Compute aggregate metrics over the completed rollout batch.

        Args:
            batch: Batched rollout data. Expected keys: ``"rewards"``,
                ``"is_end"``, ``"generation_lengths"``, ``"prompt_lengths"``,
                ``"text"``.

        Returns:
            Tuple of (unmodified batch, metrics dict).
        """
        rewards = batch["rewards"] * batch["is_end"]

        metrics: dict[str, float | int] = {
            "jaccard/mean_reward": rewards.mean().item(),
            "jaccard/max_reward": rewards.max().item(),
            "jaccard/fraction_properly_ended": batch["is_end"].float().mean().item(),
            "jaccard/num_problems_in_batch": int(batch["is_end"].shape[0]),
            "jaccard/generation_lengths": batch["generation_lengths"]
            .float()
            .mean()
            .item(),
            "jaccard/prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "jaccard/pass_at_samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], rewards
            ),
        }

        return batch, metrics