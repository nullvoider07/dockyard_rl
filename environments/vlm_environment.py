import contextlib
import io
import logging
import sys
from functools import partial
from typing import Any, Callable, NotRequired, Optional, TypedDict, cast
import ray
import torch
from dockyard_rl.data.interfaces import LLMMessageLogType
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from dockyard_rl.environments.metrics import calculate_pass_rate_per_prompt
from dockyard_rl.environments.rewards import (
    bbox_giou_reward,
    combine_reward_functions,
    exact_answer_alphanumeric_reward,
    format_reward,
    math_expression_reward,
)
from dockyard_rl.environments.utils import chunk_list_to_workers

class VLMEnvConfig(TypedDict):
    num_workers: int
    stop_strings: NotRequired[Optional[list[str]]]  # Default stop strings for this env
    reward_functions: list[dict[str, Any]]  # List of reward functions and weights

@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield

@ray.remote  # pragma: no cover
class VLMVerifyWorker:
    def __init__(self, cfg: VLMEnvConfig) -> None:
        logging.getLogger("vlm_worker").setLevel(logging.CRITICAL)

        from typing import cast as _cast
        reward_functions: list[tuple[Callable[[str, str], tuple[float, bool]], float]] = []
        for reward_func_cfg in cfg["reward_functions"]:
            reward_func_name: str = reward_func_cfg["name"]
            reward_func_weight: float = reward_func_cfg["weight"]
            reward_func_kwargs: Optional[dict] = reward_func_cfg.get("kwargs", None)

            if reward_func_name == "format":
                reward_func: Callable = format_reward
            elif reward_func_name == "exact_alnum":
                reward_func = exact_answer_alphanumeric_reward
            elif reward_func_name == "math_expr":
                reward_func = math_expression_reward
            elif reward_func_name == "bbox_giou":
                reward_func = bbox_giou_reward
            else:
                # Try resolving via Hydra for custom reward functions registered
                # via the dockyard_rl config. This allows extending the reward
                # function registry without modifying this file.
                try:
                    from hydra.utils import get_method  # type: ignore[import]
                    reward_func = get_method(reward_func_name)
                except Exception:
                    raise ValueError(
                        f"Invalid reward function: {reward_func_name}. "
                        "Built-in options: 'format', 'exact_alnum', 'math_expr', 'bbox_giou'. "
                        "For custom functions, pass the fully-qualified dotted path and ensure "
                        "it is resolvable by hydra.utils.get_method."
                    )

            if reward_func_kwargs is not None:
                reward_func = partial(reward_func, **reward_func_kwargs)

            reward_functions.append((reward_func, reward_func_weight))

        if len(reward_functions) == 0:
            raise ValueError("No reward functions provided")

        self.verify_func = combine_reward_functions(reward_functions)  # type: ignore[arg-type]

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
    ) -> list[float]:
        """Score a batch of predictions against their ground truths.

        Args:
            pred_responses: Predicted response strings from the LLM.
            ground_truths:  Ground truth strings.

        Returns:
            List of combined reward scores.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func(ground_truth, response)
                    except Exception as e:
                        ret_score = 0.0
                        print(f"Error in verify_func: {e}")
                results.append(float(ret_score))
            except Exception as e:
                print(f"Error in verify: {e}")
                results.append(0.0)
        return results

class VLMEnvironmentMetadata(TypedDict):
    ground_truth: str

@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class VLMEnvironment(EnvironmentInterface[VLMEnvironmentMetadata]):
    """Environment that scores VLM (vision-language model) task responses.

    A pool of ``VLMVerifyWorker`` Ray actors applies a composition of reward
    functions (format, exact-alnum, math-expression, bbox-GIoU, or any custom
    Hydra-resolvable function) to the assistant's responses.

    All episodes terminate after one step. Rewards are continuous floats produced
    by ``combine_reward_functions``.
    """

    def __init__(self, cfg: VLMEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            VLMVerifyWorker.options(  # type: ignore[attr-defined]
                runtime_env={"py_executable": sys.executable}
            ).remote(cfg)
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)  # type: ignore[arg-type]

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[VLMEnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[VLMEnvironmentMetadata]:
        """Run a scoring step in the VLM environment.

        Args:
            message_log_batch: Batch of OpenAI-API-like message logs. All
                assistant turns are concatenated and scored as a single response.
            metadata: Per-sample metadata dicts; must contain a ``"ground_truth"``
                key used by the reward functions.
            return_extracted_answer: Unused — VLM rewards are scalar, not symbolic.

        Returns:
            EnvironmentReturn with rewards and all episodes terminated.
        """
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                cast(str, interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_responses = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(
            ground_truths, self.num_workers
        )

        futures = [
            self.workers[i].verify.remote(chunk, gt_chunk)  # type: ignore[union-attr]
            for i, (chunk, gt_chunk) in enumerate(
                zip(chunked_responses, chunked_ground_truths)
            )
        ]

        results = [item for sublist in ray.get(futures) for item in sublist]

        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
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
            batch: Rollout batch. Expected keys: ``"rewards"``, ``"is_end"``,
                ``"generation_lengths"``, ``"prompt_lengths"``, ``"text"``.

        Returns:
            Tuple of (batch with rewards zeroed for improperly-ended sequences,
            metrics dict).
        """
        # Zero reward for any sequence that did not end properly.
        batch["rewards"] = batch["rewards"] * batch["is_end"]

        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics: dict[str, float | int] = {
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"]
            .float()
            .mean()
            .item(),
            "num_problems_in_batch": int(batch["is_end"].shape[0]),
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics