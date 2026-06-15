"""CUA multimodal multi-turn rollout for OSWorld.

A self-contained variant of ``experience/rollouts.py``'s async multi-turn loop
that threads per-turn SCREENSHOTS back into the message log as image content —
the piece the core text rollout lacks (its env-observation path is text-only).
The core rollout is left byte-untouched; this loop is used only for the OSWorld
(CUA) task.

Per sample:
  1. ``env.begin_episode(extra_env_info)`` provisions the desktop and returns the
     turn-0 screenshot (the desktop is not booted at data-load time, so the
     prompt is text-only and the first frame is seeded here).
  2. Each turn: generate an action (the vLLM request carries the cumulative
     images via ``vllm_images`` -> ``multi_modal_data``), step the environment,
     then append the returned screenshot as a new multimodal user message
     (token_ids with image placeholders + packed pixel tensors), produced the
     same way as the data processor / ``get_formatted_message_log``.

The environment returns the screenshot bytes in the per-turn metadata
(``_screenshot``); this loop decodes them to PIL, threads them, and strips the
bytes before carrying metadata forward. ``tokenizer`` is the VLM processor
(AutoProcessor) — the same object the data path uses for Qwen-VL.

Output contract matches ``run_async_multi_turn_rollout``: ``(final_batch,
rollout_metrics)``, so GRPO consumes it unchanged.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Optional, cast

import ray
import torch

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import EnvironmentInterface
from dockyard_rl.experience.rollouts import (
    batched_message_log_to_flat_message,
    calculate_rewards,
    generate_responses_async,
)
from dockyard_rl.experience.cua.vision import screenshot_to_image
from dockyard_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
)

try:
    from dockyard_rl.data.multimodal_utils import (  # type: ignore[import]
        PackedTensor,
        get_dim_to_pack_along,
        get_multimodal_keys_from_processor,
    )
except ImportError:  # pragma: no cover - image deps absent on some hosts
    PackedTensor = cast(Any, None)

    def get_multimodal_keys_from_processor(processor: Any) -> list[str]:  # type: ignore[misc]
        return []

    def get_dim_to_pack_along(processor: Any, key: str) -> int:  # type: ignore[misc]
        return 0


def _pad_token_id(processor: Any) -> int:
    pad = getattr(processor, "pad_token_id", None)
    if pad is None:
        inner = getattr(processor, "tokenizer", None)
        pad = getattr(inner, "pad_token_id", None)
    return int(pad) if pad is not None else 0


def _encode_observation_message(
    processor: Any, text: str, image: Optional[Any]
) -> dict[str, Any]:
    """Build a user-turn message for an environment observation.

    With an image, the content is tokenized via the processor so the token_ids
    carry the model's image-placeholder tokens and the packed pixel tensors are
    attached (mirroring get_formatted_message_log's multimodal branch). Without
    an image, a plain text user turn is produced. Either way the chat template's
    generation prompt is appended so the sequence is ready for the next turn.
    """
    if image is None:
        formatted: str = processor.apply_chat_template(
            [{"role": "user", "content": text or ""}],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        ids = processor(text=[formatted], return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]
        return {"role": "user", "content": formatted, "token_ids": ids.to(torch.int64)}

    content_list: list[dict[str, Any]] = [{"type": "image", "image": image}]
    if text:
        content_list.append({"type": "text", "text": text})
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": content_list}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    processed = processor(
        text=[formatted], images=[image], return_tensors="pt", add_special_tokens=False
    )
    message: dict[str, Any] = {
        "role": "user",
        "content": formatted,
        "token_ids": processed["input_ids"][0].to(torch.int64),
    }
    for key in get_multimodal_keys_from_processor(processor):
        if key not in processed:
            continue
        if key in ("token_type_ids", "mm_token_type_ids"):
            message[key] = processed[key][0]
        elif PackedTensor is not None:
            message[key] = PackedTensor(
                processed[key], dim_to_pack=get_dim_to_pack_along(processor, key)
            )
    return message


async def _cua_generate_for_turn(
    policy_generation: GenerationInterface,
    message_log: list[dict],
    stop_strings: list[str] | None,
    processor: Any,
    vllm_images: list[Any],
    greedy: bool,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate one action turn, routing the cumulative images to vLLM.

    The flat token sequence already contains one image-placeholder run per
    screenshot turn; ``vllm_images`` (the matching PIL list) is attached so
    format_prompt_for_vllm_generation supplies ``multi_modal_data``. The packed
    pixel tensors live in the message log for the later training step, not here.
    """
    flat_messages, input_lengths = batched_message_log_to_flat_message(
        [message_log],
        pad_value_dict=cast("dict[str, int]", {"token_ids": _pad_token_id(processor)}),
    )

    generation_input_data = BatchedDataDict[GenerationDatumSpec](
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "stop_strings": [stop_strings],
        }
    )
    if vllm_images:
        generation_input_data["vllm_images"] = [list(vllm_images)]

    dummy_batch = BatchedDataDict(
        {"message_log": [message_log], "stop_strings": [stop_strings]}
    )

    updated_batch, generated_ids, gen_metrics = await generate_responses_async(
        policy_generation,
        generation_input_data,
        dummy_batch,
        processor,
        input_lengths=input_lengths,
        include_logprobs=True,
        greedy=greedy,
    )
    updated_message_log = updated_batch["message_log"][0]
    generated_tokens = generated_ids[0] if generated_ids else torch.empty(0)
    return updated_message_log, generated_tokens, input_lengths, gen_metrics


async def _run_cua_sample(
    sample_idx: int,
    initial_sample_state: dict,
    policy_generation: GenerationInterface,
    processor: Any,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int,
    greedy: bool,
) -> tuple[dict, dict[str, Any]]:
    message_log = copy.deepcopy(initial_sample_state["message_log"])
    extra_env_info = copy.deepcopy(initial_sample_state["extra_env_info"])
    stop_strings = initial_sample_state.get("stop_strings", None)
    task_name = initial_sample_state["task_name"]
    env = task_to_env[task_name]

    total_reward = 0.0
    turn_count = 0
    token_count = 0
    assistant_token_count = 0
    env_token_count = 0
    terminated = False
    truncated = False
    max_turns_reached = False
    turn_gen_tokens: list[int] = []
    turn_input_tokens: list[int] = []
    turn_total_tokens: list[int] = []
    per_worker_token_counts: dict[int, int] = {}
    vllm_images: list[Any] = []

    def _empty_metrics() -> dict[str, Any]:
        return {
            "turn_count": turn_count,
            "total_tokens": token_count,
            "assistant_tokens": assistant_token_count,
            "env_tokens": env_token_count,
            "terminated": terminated,
            "truncated": truncated,
            "max_turns_reached": max_turns_reached,
            "total_reward": total_reward,
            "turn_gen_tokens": turn_gen_tokens,
            "turn_input_tokens": turn_input_tokens,
            "turn_total_tokens": turn_total_tokens,
            "per_worker_token_counts": per_worker_token_counts,
        }

    def _final_state() -> dict[str, Any]:
        return {
            "message_log": message_log,
            "extra_env_info": extra_env_info,
            "task_name": task_name,
            "total_reward": torch.tensor(total_reward),
            "stop_strings": stop_strings,
            "idx": sample_idx,
        }

    # Seed the turn-0 observation (provisions the desktop).
    seed = cast(
        dict,
        await asyncio.to_thread(
            lambda: ray.get(cast(Any, env).begin_episode.remote(extra_env_info))
        ),
    )
    extra_env_info = seed["metadata"]
    if seed.get("error"):
        terminated = True
        return _final_state(), _empty_metrics()

    seed_shot = seed.get("screenshot")
    seed_image = screenshot_to_image(seed_shot) if seed_shot else None
    if seed_image is not None:
        vllm_images.append(seed_image)
    seed_msg = _encode_observation_message(
        processor, seed.get("text") or "Here is the current screen.", seed_image
    )
    message_log.append(seed_msg)
    env_token_count += len(seed_msg["token_ids"])
    token_count += len(seed_msg["token_ids"])

    for _turn in range(max_rollout_turns):
        if terminated or truncated:
            break
        turn_count += 1

        try:
            (
                message_log,
                generated_tokens,
                input_lengths,
                gen_metrics,
            ) = await _cua_generate_for_turn(
                policy_generation,
                message_log,
                stop_strings,
                processor,
                vllm_images,
                greedy,
            )
        except Exception as e:  # noqa: BLE001 - end this sample cleanly
            print(f"Error generating response for CUA sample {sample_idx}: {e}")
            break

        response_truncated = gen_metrics.pop("_response_truncated", None)
        if response_truncated is not None and response_truncated[0]:
            truncated = True

        gen_token_count = len(generated_tokens)
        assistant_token_count += gen_token_count
        token_count += gen_token_count
        turn_gen_tokens.append(gen_token_count)
        turn_input_tokens.append(int(input_lengths.item()))
        turn_total_tokens.append(int(input_lengths.item()) + gen_token_count)
        if "gen_leader_worker_idx" in gen_metrics:
            widx = int(gen_metrics["gen_leader_worker_idx"])
            per_worker_token_counts[widx] = (
                per_worker_token_counts.get(widx, 0) + gen_token_count
            )

        sample_batch = BatchedDataDict(
            {
                "message_log": [message_log],
                "extra_env_info": [extra_env_info],
                "task_name": [task_name],
            }
        )
        env_output = await asyncio.to_thread(
            calculate_rewards, sample_batch, task_to_env
        )
        total_reward += float(env_output.rewards[0].item())
        terminated = bool(env_output.terminateds[0].item())

        obs = env_output.observations[0]
        obs_text = cast(str, obs.get("content", "") if isinstance(obs, dict) else "")
        meta = env_output.metadata[0]
        shot = meta.get("_screenshot") if isinstance(meta, dict) else None
        obs_image = screenshot_to_image(shot) if shot else None

        env_message = _encode_observation_message(processor, obs_text, obs_image)
        obs_tokens = len(env_message["token_ids"])

        # A multimodal message cannot be token-truncated without desyncing the
        # image placeholders from the pixel tensors, so on overflow stop rather
        # than slice. The terminal (grading) turn is always recorded.
        if (
            not terminated
            and int(input_lengths.item()) + gen_token_count + obs_tokens >= max_seq_len
        ):
            truncated = True
            break

        if obs_image is not None:
            vllm_images.append(obs_image)
        message_log.append(env_message)
        env_token_count += obs_tokens
        token_count += obs_tokens

        # Carry state forward; drop the transient screenshot bytes from metadata.
        if isinstance(meta, dict):
            meta = {k: v for k, v in meta.items() if k != "_screenshot"}
            extra_env_info = meta
        if not terminated and not truncated:
            if env_output.next_stop_strings[0] is not None:
                stop_strings = env_output.next_stop_strings[0]

    if turn_count >= max_rollout_turns:
        max_turns_reached = True

    return _final_state(), _empty_metrics()


def run_async_cua_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[Any],
    tokenizer: Any,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> tuple[BatchedDataDict[Any], dict[str, Any]]:
    """CUA multimodal multi-turn rollout (drop-in for run_async_multi_turn_rollout).

    ``tokenizer`` is the VLM processor (AutoProcessor) used for Qwen-VL.
    """

    async def _impl():
        batch_size = len(input_batch["message_log"])
        initial_states = [
            {
                "message_log": input_batch["message_log"][i],
                "extra_env_info": input_batch["extra_env_info"][i],
                "task_name": input_batch["task_name"][i],
                "stop_strings": input_batch.get("stop_strings", [None] * batch_size)[i],
                "idx": input_batch.get("idx", list(range(batch_size)))[i],
            }
            for i in range(batch_size)
        ]

        async def _run(i: int, state: dict):
            try:
                return await _run_cua_sample(
                    sample_idx=i,
                    initial_sample_state=state,
                    policy_generation=policy_generation,
                    processor=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=max_seq_len,
                    max_rollout_turns=max_rollout_turns,
                    greedy=greedy,
                )
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Error in CUA sample {i} rollout: {e}") from e

        results = await asyncio.gather(
            *[_run(i, s) for i, s in enumerate(initial_states)]
        )
        final_states = [r[0] for r in results]
        all_metrics = [r[1] for r in results]

        final_batch = BatchedDataDict(
            {
                "message_log": [s["message_log"] for s in final_states],
                "extra_env_info": [s["extra_env_info"] for s in final_states],
                "task_name": [s["task_name"] for s in final_states],
                "total_reward": torch.stack([s["total_reward"] for s in final_states]),
                "idx": [s.get("idx", i) for i, s in enumerate(final_states)],
                "truncated": torch.tensor(
                    [m["truncated"] for m in all_metrics], dtype=torch.bool
                ),
            }
        )
        for key in input_batch.keys():
            if key not in final_batch:
                final_batch[key] = input_batch[key]

        rollout_metrics: dict[str, Any] = {
            "total_turns": sum(m["turn_count"] for m in all_metrics),
            "avg_turns_per_sample": sum(m["turn_count"] for m in all_metrics)
            / batch_size,
            "max_turns_per_sample": max(m["turn_count"] for m in all_metrics),
            "natural_termination_rate": sum(m["terminated"] for m in all_metrics)
            / batch_size,
            "truncation_rate": sum(m["truncated"] for m in all_metrics) / batch_size,
            "max_turns_reached_rate": sum(m["max_turns_reached"] for m in all_metrics)
            / batch_size,
            "mean_total_tokens_per_sample": sum(m["total_tokens"] for m in all_metrics)
            / batch_size,
            "mean_gen_tokens_per_sample": sum(
                m["assistant_tokens"] for m in all_metrics
            )
            / batch_size,
            "max_gen_tokens_per_sample": max(
                m["assistant_tokens"] for m in all_metrics
            ),
            "mean_env_tokens_per_sample": sum(m["env_tokens"] for m in all_metrics)
            / batch_size,
            "mean_total_reward": sum(m["total_reward"] for m in all_metrics)
            / batch_size,
            "max_total_reward": max(m["total_reward"] for m in all_metrics),
            "min_total_reward": min(m["total_reward"] for m in all_metrics),
        }
        if "per_worker_token_counts" in all_metrics[0]:
            merged: dict[int, int] = {}
            for m in all_metrics:
                for k, v in cast(dict, m["per_worker_token_counts"]).items():
                    merged[k] = merged.get(k, 0) + v
            rollout_metrics["per_worker_token_counts"] = merged
        rollout_metrics["histogram/gen_tokens_length"] = [
            t for m in all_metrics for t in m["turn_gen_tokens"]
        ]
        rollout_metrics["histogram/input_tokens_length"] = [
            t for m in all_metrics for t in m["turn_input_tokens"]
        ]
        rollout_metrics["histogram/total_tokens_length"] = [
            t for m in all_metrics for t in m["turn_total_tokens"]
        ]
        return final_batch, rollout_metrics

    return asyncio.run(_impl())
