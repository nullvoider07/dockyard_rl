# Generate rollouts for arbitrary environments.
# Supports multi-turn rollouts and many simultaneous environments
# (e.g. you can train on coding tasks, math, multi-turn games, and more at once).

import asyncio
import copy
import logging
from typing import TYPE_CHECKING, Any, Optional, cast
import ray
import torch
from transformers import PreTrainedTokenizerBase
from dockyard_rl.algorithms.utils import get_gdpo_reward_component_keys
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from dockyard_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from dockyard_rl.rewards.invalid_action import (
    InvalidActionPenaltyConfig,
    assess_assistant_turn,
    pop_env_flags,
)
from dockyard_rl.tool_protocol.protocol import (
    THINKING_PAIRED,
    StructuredToolUseConfig,
)
from dockyard_rl.tool_protocol.hermes import (
    HERMES_TOOL_CALL_END,
    HERMES_TOOL_CALL_START,
)

logger = logging.getLogger(__name__)

# Qwen/Hermes turn-envelope markers (structured tool-use protocol, design fork 3).
# On the structured path a tool-result turn is wrapped into the model's native
# multi-turn envelope BEFORE tokenizing, instead of being tokenized as the env's
# raw plain-text content:
#
#   <|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n
#
# The whole envelope is appended as one non-assistant (role-masked) message, so the
# trailing generation prompt is masked out and the next assistant turn's generated
# token_ids begin immediately after it. The byte sequence (role markers,
# <tool_response> newlines, generation-prompt suffix) is the documented Qwen3 chat
# template default; it is PINNED/VERIFIED at the build-time gate by loading the real
# Qwen3.x tokenizer (design "Build-time verification gate", item 1), not from memory.
# These constants must match what that gate confirms before the structured path is
# relied on in a live run.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_TOOL_RESPONSE_OPEN = "<tool_response>"
_TOOL_RESPONSE_CLOSE = "</tool_response>"
_TOOL_RESPONSE_ENVELOPE_PREFIX = f"{_IM_START}user\n{_TOOL_RESPONSE_OPEN}\n"
_TOOL_RESPONSE_ENVELOPE_SUFFIX = (
    f"\n{_TOOL_RESPONSE_CLOSE}{_IM_END}\n{_IM_START}assistant\n"
)


def _wrap_tool_response_envelope(content: str) -> str:
    """Wrap an env observation's plain content in the Qwen/Hermes turn envelope.

    Used only on the structured tool-use path (design fork 3). The returned
    string is tokenized with add_special_tokens=False and appended as a single
    role-masked (non-assistant) message; the envelope's trailing generation
    prompt is therefore not trained.
    """
    return f"{_TOOL_RESPONSE_ENVELOPE_PREFIX}{content}{_TOOL_RESPONSE_ENVELOPE_SUFFIX}"


def _structured_thinking_style(
    structured_cfg: Optional[StructuredToolUseConfig],
) -> str:
    """Reasoning-tag style for assess_assistant_turn (default when cfg absent)."""
    if structured_cfg is None:
        return THINKING_PAIRED
    return structured_cfg.get("thinking_style", THINKING_PAIRED)


def _tool_tags_survive_skip_special(tokenizer: "TokenizerType") -> bool:
    """Whether the Hermes tool-call tags survive ``skip_special_tokens=True``.

    Encodes ``<tool_call>`` / ``</tool_call>`` and decodes them back with
    ``skip_special_tokens=True``; returns False if either tag is dropped, i.e.
    the tokenizer marks it special. Memoized on the tokenizer (the result is a
    fixed property of the tokenizer's vocabulary). On any tokenizer error,
    conservatively returns False so the caller keeps special tokens rather than
    risk silently stripping the tags.
    """
    cached = getattr(tokenizer, "_dockyard_tool_tags_survive_skip", None)
    if cached is not None:
        return bool(cached)
    survives = True
    try:
        for tag in (HERMES_TOOL_CALL_START, HERMES_TOOL_CALL_END):
            ids = tokenizer(tag, add_special_tokens=False).input_ids
            if tag not in tokenizer.decode(ids, skip_special_tokens=True):
                survives = False
                break
    except Exception:  # noqa: BLE001 — be conservative on any tokenizer quirk
        survives = False
    try:
        tokenizer._dockyard_tool_tags_survive_skip = survives
    except Exception:  # noqa: BLE001 — tokenizer may forbid attribute set
        pass
    return survives


def _structured_decode_skip_special(
    structured_cfg: Optional[StructuredToolUseConfig],
    tokenizer: "TokenizerType",
) -> bool:
    """Resolve ``skip_special_tokens`` for decoding the assistant turn.

    Legacy path (no structured cfg) → True, unchanged. Structured path:
    ``decode_skip_special_tokens`` in the config wins when set explicitly;
    otherwise (None/"auto") it is derived from the tokenizer — skip special
    tokens only when the tool-call tags are NOT marked special (so skipping
    cannot strip them). This removes the silent-failure mode where an enabled
    protocol on a model that specials the tags would otherwise drop them from
    the decoded content the env parses.
    """
    if structured_cfg is None:
        return True
    explicit = structured_cfg.get("decode_skip_special_tokens")
    if explicit is not None:
        return bool(explicit)
    survives = _tool_tags_survive_skip_special(tokenizer)
    if not survives:
        logger.info(
            "structured_tool_use: tokenizer marks the tool-call tags special; "
            "decoding assistant turns with skip_special_tokens=False so "
            "<tool_call> survives for the env parser."
        )
    return survives

# Deferred: data/interfaces.py (#5 in backlog)
if TYPE_CHECKING:
    from dockyard_rl.data.interfaces import (
        DatumSpec,
        FlatMessagesType,
        LLMMessageLogType,
    )
else:
    try:
        from dockyard_rl.data.interfaces import (
            DatumSpec,
            FlatMessagesType,
            LLMMessageLogType,
        )
    except ImportError:
        # Pylance sees 'Any' as a valid type, not a runtime variable
        DatumSpec = Any  # type: ignore[assignment,misc]
        FlatMessagesType = Any  # type: ignore[assignment,misc]
        LLMMessageLogType = Any  # type: ignore[assignment,misc]

# Deferred: data/llm_message_utils.py (#5 in backlog)
try:
    from dockyard_rl.data.llm_message_utils import (
        batched_message_log_to_flat_message,
        get_keys_from_message_log,
    )
except ImportError:
    batched_message_log_to_flat_message = cast(Any, None)
    get_keys_from_message_log = cast(Any, None)

# Deferred: utils/timer.py (#18 in backlog)
try:
    from dockyard_rl.utils.timer import Timer
except ImportError:
    Timer = cast(Any, None)

# Deferred: models/generation/interfaces.py (GenerationConfig may not yet include all fields)
try:
    from dockyard_rl.models.generation.interfaces import GenerationConfig
except ImportError:
    GenerationConfig = dict  # type: ignore

TokenizerType = PreTrainedTokenizerBase

def generate_responses(
    policy_generation: GenerationInterface,
    generation_input_data: BatchedDataDict[GenerationDatumSpec],
    batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    input_lengths: torch.Tensor,
    include_logprobs: bool = True,
    greedy: bool = False,
    structured_cfg: Optional[StructuredToolUseConfig] = None,
) -> tuple[BatchedDataDict[DatumSpec], list[torch.Tensor], dict[str, Any]]:
    """Generate responses from policy using synchronous generation.

    structured_cfg: When set with decode_skip_special_tokens=False, the
        assistant message ``content`` is decoded with skip_special_tokens=False
        so structured tool-call tags (<tool_call>/</tool_call>) survive for the
        env parser. The generated token_ids are returned verbatim regardless;
        only the decoded text differs. Default/None leaves the decode as-is.
    """
    # Add stop_strings to generation_input_data if present in the batch
    if "stop_strings" in batch:
        generation_input_data["stop_strings"] = batch["stop_strings"]
    else:
        # Ensure the key exists even if it's None, matching GenerationDatumSpec
        generation_input_data["stop_strings"] = [None] * len(input_lengths)

    # Always use synchronous generation
    generation_outputs = policy_generation.generate(
        generation_input_data, greedy=greedy
    )

    # Extract everything we need from the generation outputs
    output_ids = generation_outputs["output_ids"]
    generation_lengths = generation_outputs["generation_lengths"]
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]

    # Extract truncated info if available (response hit max_tokens without stop token)
    response_truncated = generation_outputs.get("truncated")

    # Extract generated parts
    generated_ids = []
    for i in range(len(input_lengths)):
        input_len = input_lengths[i].item()
        total_length = unpadded_sequence_lengths[i].item()
        full_output = output_ids[i]
        generated_part = full_output[input_len:total_length]
        generated_ids.append(generated_part)

    # Structured path: the tool-call tags must survive the decode so the env
    # parser sees them. _structured_decode_skip_special auto-detects from the
    # tokenizer (or honors an explicit config override); legacy path → True.
    # token_ids are unaffected regardless; only the decoded content differs.
    decode_skip_special = _structured_decode_skip_special(structured_cfg, tokenizer)
    generated_texts = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=decode_skip_special
    )

    # Append to message log
    for i, (text, input_length, total_length) in enumerate(
        zip(generated_texts, input_lengths, unpadded_sequence_lengths)
    ):
        assistant_message = {
            "role": "assistant",
            "content": text,
            "token_ids": output_ids[i, input_length:total_length],
        }

        if include_logprobs and "logprobs" in generation_outputs:
            assistant_message["generation_logprobs"] = generation_outputs["logprobs"][
                i, input_length:total_length
            ]

        batch["message_log"][i].append(assistant_message)

    # Generation metrics
    gen_metrics = {
        "mean_generation_length": generation_lengths.float().mean().item(),
        "total_generated_tokens": generation_lengths.sum().item(),
    }

    # Add response_truncated to gen_metrics for use by caller
    if response_truncated is not None:
        gen_metrics["_response_truncated"] = response_truncated

    return batch, generated_ids, gen_metrics

async def generate_responses_async(
    policy_generation: GenerationInterface,
    generation_input_data: BatchedDataDict[GenerationDatumSpec],
    batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    input_lengths: torch.Tensor,
    include_logprobs: bool = True,
    greedy: bool = False,
    structured_cfg: Optional[StructuredToolUseConfig] = None,
) -> tuple[BatchedDataDict[DatumSpec], list[torch.Tensor], dict[str, Any]]:
    """Async version of generate_responses that properly calls generate_async.

    structured_cfg: see ``generate_responses``; controls only the decode of the
        assistant ``content`` (token_ids returned verbatim).
    """
    # Add stop_strings to generation_input_data if present in the batch
    if "stop_strings" in batch:
        generation_input_data["stop_strings"] = batch["stop_strings"]
    else:
        generation_input_data["stop_strings"] = [None] * len(input_lengths)

    # Check if the configured backend has its async engine enabled. Both vLLM
    # (vllm_cfg) and SGLang (sglang_cfg) expose generate_async + an async_engine
    # flag; accept whichever backend's config block is present.
    cfg = cast(Any, policy_generation).cfg if hasattr(policy_generation, "cfg") else {}
    backend_async_enabled = (
        cfg.get("vllm_cfg", {}).get("async_engine", False)
        or cfg.get("sglang_cfg", {}).get("async_engine", False)
    )
    use_async_generation = backend_async_enabled and hasattr(
        policy_generation, "generate_async"
    )

    assert use_async_generation, (
        "Async generation is not enabled. Set async_engine=True in the "
        "vllm_cfg (vLLM) or sglang_cfg (SGLang) section of the policy config."
    )

    # Use async generation with per-sample streaming
    collected_indexed_outputs: list[
        tuple[int, BatchedDataDict[GenerationOutputSpec]]
    ] = []
    async for original_idx, single_item_output in cast(Any, policy_generation).generate_async(
        generation_input_data, greedy=greedy
    ):
        collected_indexed_outputs.append((original_idx, single_item_output))

    # Sort by original_idx to ensure order matches generation_input_data
    collected_indexed_outputs.sort(key=lambda x: x[0])

    # Extract in correct order
    ordered_batched_data_dicts = [item for _, item in collected_indexed_outputs]

    assert ordered_batched_data_dicts, (
        "Generation returned no outputs for a non-empty batch."
    )

    generation_outputs = BatchedDataDict.from_batches(
        ordered_batched_data_dicts,
        pad_value_dict={"output_ids": tokenizer.pad_token_id, "logprobs": 0.0},
    )

    # Extract everything we need from the generation outputs
    output_ids = generation_outputs["output_ids"]
    generation_lengths = generation_outputs["generation_lengths"]
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]

    # Extract truncated info if available
    response_truncated = generation_outputs.get("truncated")

    # Extract generated parts
    generated_ids = []
    for i in range(len(input_lengths)):
        input_len = input_lengths[i].item()
        total_length = unpadded_sequence_lengths[i].item()
        full_output = output_ids[i]
        generated_part = full_output[input_len:total_length]
        generated_ids.append(generated_part)

    # See generate_responses: auto-detect (or honor explicit override) whether to
    # keep special tokens so the tool-call tags survive in the decoded content;
    # token_ids are untouched.
    decode_skip_special = _structured_decode_skip_special(structured_cfg, tokenizer)
    generated_texts = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=decode_skip_special
    )

    # Append to message log
    for i, (text, input_length, total_length) in enumerate(
        zip(generated_texts, input_lengths, unpadded_sequence_lengths)
    ):
        assistant_message = {
            "role": "assistant",
            "content": text,
            "token_ids": output_ids[i, input_length:total_length],
        }

        if include_logprobs and "logprobs" in generation_outputs:
            assistant_message["generation_logprobs"] = generation_outputs["logprobs"][
                i, input_length:total_length
            ]

        batch["message_log"][i].append(assistant_message)

    # Generation metrics
    gen_metrics = {
        "mean_generation_length": generation_lengths.float().mean().item(),
        "total_generated_tokens": generation_lengths.sum().item(),
    }
    # Attach worker metadata if present (async vLLM path)
    if "gen_leader_worker_idx" in generation_outputs:
        v = generation_outputs["gen_leader_worker_idx"][0]
        try:
            if isinstance(v, list):
                # Explicitly cast to list to satisfy the static subscript checker
                v_list = cast(list[Any], v)
                gen_metrics["gen_leader_worker_idx"] = int(v_list[0])
            else:
                # Safe fallback for standard ints/floats
                gen_metrics["gen_leader_worker_idx"] = int(cast(Any, v))
        except Exception as e:
            print(f"Error occurred while extracting gen_leader_worker_idx: {e}")

    # Add response_truncated to gen_metrics for use by caller
    if response_truncated is not None:
        gen_metrics["_response_truncated"] = response_truncated

    return batch, generated_ids, gen_metrics

def calculate_rewards(
    batch: BatchedDataDict[DatumSpec],
    task_to_env: dict[str, EnvironmentInterface],
) -> EnvironmentReturn:
    """Calculate rewards for generated responses and get environment feedback.

    Args:
        batch: Batch containing message_log (LLMMessageLogType) with generated responses
        task_to_env: Dictionary mapping task names to their corresponding environments

    Returns:
        EnvironmentReturn namedtuple containing:
            - observations: List of observations from the environment for the next turn.
            - metadata: List of extracted metadata from the environment.
            - next_stop_strings: List of stop strings for the next generation step.
            - rewards: Tensor of rewards for the last turn.
            - terminateds: Tensor of booleans indicating if an episode ended naturally.
    """
    # Extract message logs for environment (most recent interaction)
    to_env = [
        get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
        for i in range(len(batch["message_log"]))
    ]
    task_names = batch["task_name"]

    # Group messages by task type
    task_groups: dict[str, list[tuple[int, LLMMessageLogType]]] = {}
    for i, task_name in enumerate(task_names):
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append((i, to_env[i]))

    # Calculate rewards for each task group concurrently
    futures = []
    future_to_indices = {}  # Map future to its corresponding indices
    for task_name, group in task_groups.items():
        if task_name not in task_to_env:
            raise ValueError(f"No environment found for task type: {task_name}")

        # Extract indices and messages for this group
        indices = [idx for idx, _ in group]
        messages = [msg for _, msg in group]

        # Get corresponding environment info
        env_info = [batch["extra_env_info"][i] for i in indices]

        # Submit task to environment and store future
        future = task_to_env[task_name].step.remote(messages, env_info)  # type: ignore  # ray actor call
        futures.append(future)
        future_to_indices[future] = indices

    results = ray.get(futures)
    all_rewards: list = []  # per-sample scalars/tensors (single-reward envs)
    all_dict_rewards: dict[str, list] | None = None  # named components (multi-reward envs)
    is_dict_rewards = False
    all_env_observations = []
    all_terminateds = []
    all_next_stop_strings = []
    all_metadata = []
    all_indices_order = []
    all_answers = []

    for future, result in zip(futures, results):
        indices = future_to_indices[future]
        # Environment step returns: EnvironmentReturn
        (
            env_observations,
            metadata,
            next_stop_strings,
            task_rewards,
            terminateds,
            answers,
        ) = result

        is_dict_rewards = isinstance(task_rewards, dict)

        if next_stop_strings is None:
            next_stop_strings = [None] * len(terminateds)
        if answers is None:
            answers = [None] * len(terminateds)

        # Initialize the dict-reward accumulator on first encounter (outside inner loop).
        if is_dict_rewards and all_dict_rewards is None:
            all_dict_rewards = {name: [] for name in task_rewards}

        # Store results with their original indices
        for i, idx in enumerate(indices):
            all_indices_order.append(idx)
            if is_dict_rewards:
                for name in task_rewards:
                    all_dict_rewards[name].append(task_rewards[name][i])  # type: ignore[index]
            else:
                all_rewards.append(task_rewards[i])
            all_env_observations.append(env_observations[i])
            all_terminateds.append(terminateds[i])
            all_next_stop_strings.append(next_stop_strings[i])
            all_metadata.append(metadata[i])
            all_answers.append(answers[i])

    # Sort results by original index to maintain order
    sorted_indices = sorted(
        range(len(all_indices_order)), key=lambda k: all_indices_order[k]
    )

    # Build rewards: dict-of-tensors for multi-reward envs, single tensor otherwise.
    if all_dict_rewards is not None:
        assert len(all_rewards) == 0, (
            "Mixing dict-based and scalar rewards across environments is not supported. "
            "All environments must return the same reward format (all dict or all scalar)."
        )
        rewards: torch.Tensor | dict[str, torch.Tensor] = {
            name: torch.stack([vals[i] for i in sorted_indices])
            for name, vals in all_dict_rewards.items()
        }
    elif len(all_rewards) > 0 and isinstance(all_rewards[0], torch.Tensor):
        rewards = torch.stack([all_rewards[i] for i in sorted_indices])
    else:
        rewards = torch.tensor([all_rewards[i] for i in sorted_indices])

    env_observations = [all_env_observations[i] for i in sorted_indices]
    terminateds = torch.tensor([all_terminateds[i] for i in sorted_indices])
    next_stop_strings = [all_next_stop_strings[i] for i in sorted_indices]
    metadata = [all_metadata[i] for i in sorted_indices]
    answers = [all_answers[i] for i in sorted_indices]

    return EnvironmentReturn(
        observations=env_observations,
        metadata=metadata,
        next_stop_strings=next_stop_strings,
        rewards=rewards,
        terminateds=terminateds,
        answers=answers,
    )

def run_multi_turn_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
    invalid_action_cfg: Optional[InvalidActionPenaltyConfig] = None,
    structured_cfg: Optional[StructuredToolUseConfig] = None,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, Any]]:
    """Runs a multi-turn rollout loop, interacting with the environment.

    Args:
        policy_generation: The generation interface (policy).
        input_batch: The starting batch containing initial message logs.
        tokenizer: The tokenizer.
        task_to_env: Dictionary mapping task names to environment instances.
        max_rollout_turns: Maximum number of agent-environment interaction turns.
        max_seq_len: Maximum sequence length allowed.
        greedy: Whether to use greedy decoding.
        invalid_action_cfg: When enabled, per-turn invalid-action /
            malformed-thinking verdicts are collected and emitted as
            per-sample count fields (invalid_action_count,
            malformed_thinking_count) for the reward penalty.
        structured_cfg: When set, the structured (Hermes) tool-use protocol is
            active: env observations are wrapped in the Qwen turn envelope
            before tokenizing (fork 3), the assistant decode may keep special
            tokens (fork "special-token decode risk"), and the thinking style is
            threaded into the malformed-thinking detector. None = fenced-text
            path, byte-identical to before.

    Returns:
        Tuple containing:
            - BatchedDataDict with the full interaction history and accumulated rewards
            - Dictionary of rollout metrics
    """
    current_batch = input_batch.copy()  # Work on a copy
    batch_size = len(current_batch["message_log"])
    active_indices = torch.arange(batch_size)
    total_rewards = torch.zeros(batch_size, dtype=torch.float32)

    # Multi-reward accumulator: dict of {name: Tensor[B]} for multi-reward envs
    # (e.g. GDPO), None for single-reward envs.
    multi_rewards: dict[str, torch.Tensor] | None = None

    # Initialize stop_strings from the initial batch if present
    current_stop_strings = current_batch.get("stop_strings", [None] * batch_size)

    # Tracking metrics for each sample
    sample_turn_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_token_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_assistant_token_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_env_token_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_terminated = torch.zeros(batch_size, dtype=torch.bool)
    sample_truncated = torch.zeros(batch_size, dtype=torch.bool)
    sample_max_turns_reached = torch.zeros(batch_size, dtype=torch.bool)

    # Per-sample invalid-action / malformed-thinking verdict counts (#2656)
    detect_invalid = invalid_action_cfg is not None and invalid_action_cfg.get(
        "enabled", False
    )
    sample_invalid_action_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_malformed_thinking_counts = torch.zeros(batch_size, dtype=torch.int32)
    # Structured tool-use protocol (None = fenced-text path, unchanged).
    structured_enabled = structured_cfg is not None
    thinking_style = _structured_thinking_style(structured_cfg)

    # Tracking per-turn metrics
    total_gen_tokens_per_turn = []
    active_samples_per_turn = []

    for turn in range(max_rollout_turns):
        if len(active_indices) == 0:
            break

        active_samples_per_turn.append(len(active_indices))

        # Convert LLMMessageLogType to FlatMessagesType for generation
        active_batch = current_batch.select_indices(active_indices)
        active_stop_strings = [current_stop_strings[i] for i in active_indices.tolist()]

        active_flat_messages: BatchedDataDict[FlatMessagesType]
        active_flat_messages, active_input_lengths = (
            batched_message_log_to_flat_message(
                active_batch["message_log"],
                pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
            )
        )

        # Extract input_ids and lengths from the flat messages
        active_input_ids = active_flat_messages["token_ids"]

        # Prepare generation input data
        generation_input_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": active_input_ids,
                "input_lengths": active_input_lengths,
                "stop_strings": active_stop_strings,
            }
        )
        # Add the multimodal data to the generation input data
        multimodal_data = active_flat_messages.get_multimodal_dict(as_tensors=False)
        generation_input_data.update(multimodal_data)

        # Keep message log for generation
        if "vllm_content" in active_batch:
            generation_input_data["vllm_content"] = active_batch["vllm_content"]
        if "vllm_images" in active_batch:
            generation_input_data["vllm_images"] = active_batch["vllm_images"]
        if "vllm_videos" in active_batch:
            generation_input_data["vllm_videos"] = active_batch["vllm_videos"]
        if "vllm_audios" in active_batch:
            generation_input_data["vllm_audios"] = active_batch["vllm_audios"]

        # generate_responses updates active_batch["message_log"] in-place
        active_batch, generated_ids, gen_metrics = generate_responses(
            policy_generation,
            generation_input_data,
            active_batch,
            tokenizer,
            input_lengths=active_input_lengths,
            greedy=greedy,
            structured_cfg=structured_cfg,
        )

        # Record response truncation (response hit max_tokens without stop token)
        response_truncated = gen_metrics.pop("_response_truncated", None)
        if response_truncated is not None:
            for i, global_idx in enumerate(active_indices.tolist()):
                if response_truncated[i]:
                    sample_truncated[global_idx] = True

        # Record token usage - assistant
        for i, global_idx in enumerate(active_indices.tolist()):
            sample_assistant_token_counts[global_idx] += len(generated_ids[i])
            sample_token_counts[global_idx] += len(generated_ids[i])

        # Track total generated tokens this turn
        total_gen_tokens_per_turn.append(sum(len(ids) for ids in generated_ids))

        # Calculate rewards and get environment feedback
        env_output: EnvironmentReturn = calculate_rewards(active_batch, task_to_env)

        # Accumulate rewards: env returns dict[str, Tensor] for multi-reward, Tensor for single-reward.
        if isinstance(env_output.rewards, dict):
            # Initialize accumulators on first encounter.
            if multi_rewards is None:
                multi_rewards = {
                    name: torch.zeros(batch_size, dtype=torch.float32)
                    for name in env_output.rewards
                }
            reward_dict: dict[str, torch.Tensor] = multi_rewards
            for name, r in env_output.rewards.items():
                reward_dict[name][active_indices] += r
            total_rewards[active_indices] += sum(env_output.rewards.values())
        else:
            total_rewards[active_indices] += env_output.rewards

        # Update message log for ALL active samples with env observation.
        # This must happen BEFORE filtering based on done flags.
        truncation_mask = torch.zeros_like(env_output.terminateds, dtype=torch.bool)
        for i, global_idx in enumerate(active_indices.tolist()):
            # Per-turn invalid-action verdict: env-stamped flags + generic text
            # detectors over the just-generated assistant message (the last
            # entry in the shared message log; env obs is appended below).
            if detect_invalid:
                assert invalid_action_cfg is not None
                assistant_text = str(
                    current_batch["message_log"][global_idx][-1].get("content", "")
                )
                verdict = assess_assistant_turn(
                    assistant_text,
                    invalid_action_cfg,
                    env_output.metadata[i],
                    thinking_style=thinking_style,
                )
                sample_invalid_action_counts[global_idx] += int(verdict.invalid_action)
                sample_malformed_thinking_counts[global_idx] += int(
                    verdict.malformed_thinking
                )
            # Verdict keys are per-turn; never carry into next-turn extra_env_info.
            pop_env_flags(env_output.metadata[i])

            env_obs_content = env_output.observations[i]["content"]
            # Structured path: wrap the env's plain content in the Qwen turn
            # envelope before tokenizing so turns 2+ are faithful to the model's
            # native multi-turn tool format (fork 3). The envelope is still a
            # single role-masked (non-assistant) message, so loss masking is
            # unchanged. Fenced-text path tokenizes the raw content as before.
            obs_text_to_tokenize = (
                _wrap_tool_response_envelope(env_obs_content)
                if structured_enabled
                else env_obs_content
            )
            # Tokenize the (possibly wrapped) content from the environment
            tokenized_obs = tokenizer(
                obs_text_to_tokenize, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            # tokenizer returns torch.float32 when env_obs_content is empty
            tokenized_obs = tokenized_obs.to(dtype=torch.int64)

            # check if new message overflows max_seq_len
            if (
                len(tokenized_obs) + len(generated_ids[i]) + int(active_input_lengths[i].item())
                >= max_seq_len
            ):
                tokens_left_for_obs = max_seq_len - (
                    len(generated_ids[i]) + int(active_input_lengths[i].item())
                )
                assert tokens_left_for_obs >= 0, (
                    f"tokens_left_for_obs={tokens_left_for_obs} should not be negative. "
                    "This should not happen if the inference engine respects the max sequence length."
                )
                # truncate
                tokenized_obs = tokenized_obs[:tokens_left_for_obs]
                truncation_mask[i] = True
                # Record truncation
                sample_truncated[active_indices[i]] = True

            tokenized_env_obs_message = {
                "role": env_output.observations[i]["role"],
                "content": env_obs_content,
                "token_ids": tokenized_obs,
            }
            current_batch["message_log"][global_idx].append(tokenized_env_obs_message)

            # Record token usage - environment
            sample_env_token_counts[global_idx] += len(tokenized_obs)
            sample_token_counts[global_idx] += len(tokenized_obs)

            # Increment turn count
            sample_turn_counts[global_idx] += 1

        # Determine done samples and update active set
        terminateds = env_output.terminateds.bool()
        done = truncation_mask | terminateds
        sample_terminated[active_indices] |= done

        # Update active indices for the next iteration
        active_indices_local_next = torch.where(~done)[0]
        active_indices = active_indices[active_indices_local_next]
        continuing_indices_global = active_indices  # Indices relative to original batch
        # Get next stop strings and infos corresponding to the indices that are *continuing*
        continuing_next_stops = [
            env_output.next_stop_strings[i] for i in active_indices_local_next.tolist()
        ]
        # Get metadata corresponding to continuing indices
        continuing_metadata = [
            env_output.metadata[i] for i in active_indices_local_next.tolist()
        ]

        for i, global_idx in enumerate(continuing_indices_global.tolist()):
            # Update stop strings for the next turn
            current_stop_strings[global_idx] = continuing_next_stops[i]
            # Update metadata (extra_env_info) using info from environment
            if continuing_metadata[i] is not None:
                current_batch["extra_env_info"][global_idx] = continuing_metadata[i]

    # Record samples that reached max turns
    sample_max_turns_reached[active_indices] = True

    # Add total rewards to the final batch
    current_batch["total_reward"] = total_rewards
    current_batch["truncated"] = sample_truncated
    if detect_invalid:
        current_batch["invalid_action_count"] = sample_invalid_action_counts
        current_batch["malformed_thinking_count"] = sample_malformed_thinking_counts
    # Expose per-component rewards for multi-reward envs (e.g. GDPO advantage calculation);
    # GRPO uses total_reward only.
    if multi_rewards is not None:
        for name, reward_tensor in multi_rewards.items():
            current_batch[name] = reward_tensor

    # Calculate aggregate metrics
    rollout_metrics = {
        # Overall metrics
        "total_turns": int(sample_turn_counts.sum().item()),
        "avg_turns_per_sample": float(sample_turn_counts.float().mean().item()),
        "max_turns_per_sample": int(sample_turn_counts.max().item()),
        "natural_termination_rate": float(sample_terminated.float().mean().item()),
        "truncation_rate": float(sample_truncated.float().mean().item()),
        "max_turns_reached_rate": float(sample_max_turns_reached.float().mean().item()),
        **(
            {
                "invalid_action_rate": float(
                    (sample_invalid_action_counts > 0).float().mean().item()
                ),
                "malformed_thinking_rate": float(
                    (sample_malformed_thinking_counts > 0).float().mean().item()
                ),
            }
            if detect_invalid
            else {}
        ),
        # Token usage metrics
        "mean_total_tokens_per_sample": float(
            sample_token_counts.float().mean().item()
        ),
        "mean_gen_tokens_per_sample": float(
            sample_assistant_token_counts.float().mean().item()
        ),
        "max_gen_tokens_per_sample": float(
            sample_assistant_token_counts.float().max().item()
        ),
        "mean_env_tokens_per_sample": float(
            sample_env_token_counts.float().mean().item()
        ),
        # Reward metrics
        "mean_total_reward": float(total_rewards.mean().item()),
        "max_total_reward": float(total_rewards.max().item()),
        "min_total_reward": float(total_rewards.min().item()),
    }
    return current_batch, rollout_metrics

async def async_generate_response_for_sample_turn(
    policy_generation: GenerationInterface,
    sample_message_log: list[dict],
    sample_stop_strings: list[str] | None,
    tokenizer: TokenizerType,
    max_seq_len: int,
    greedy: bool = False,
    structured_cfg: Optional[StructuredToolUseConfig] = None,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate a response for a single sample's turn using async generation.

    Args:
        policy_generation: The generation interface to use
        sample_message_log: Message log for a single sample
        sample_stop_strings: Stop strings for this sample
        tokenizer: Tokenizer to use
        max_seq_len: Maximum sequence length
        greedy: Whether to use greedy decoding

    Returns:
        Tuple of (updated_message_log, generated_tokens, input_lengths, generation_metrics)
    """
    # Convert single sample to batch format
    batch_message_logs = [sample_message_log]

    # Convert to flat format for generation
    flat_messages, input_lengths = batched_message_log_to_flat_message(
        batch_message_logs,
        pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
    )

    # Create generation input
    generation_input_data = BatchedDataDict[GenerationDatumSpec](
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "stop_strings": [sample_stop_strings],
        }
    )

    # Create a dummy batch for generate_responses_async
    dummy_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": batch_message_logs,
            "stop_strings": [sample_stop_strings],
        }
    )

    # Generate response using the async version
    updated_batch, generated_ids, gen_metrics = await generate_responses_async(
        policy_generation,
        generation_input_data,
        dummy_batch,
        tokenizer,
        input_lengths=input_lengths,
        include_logprobs=True,
        greedy=greedy,
        structured_cfg=structured_cfg,
    )

    # Extract results for the single sample
    updated_message_log = updated_batch["message_log"][0]
    generated_tokens = generated_ids[0] if generated_ids else torch.empty(0)

    return updated_message_log, generated_tokens, input_lengths, gen_metrics

async def run_sample_multi_turn_rollout(
    sample_idx: int,
    initial_sample_state: dict,
    policy_generation: GenerationInterface,
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
    invalid_action_cfg: Optional[InvalidActionPenaltyConfig] = None,
    structured_cfg: Optional[StructuredToolUseConfig] = None,
) -> tuple[dict, dict[str, Any]]:
    """Run a multi-turn rollout for a single sample.

    This function manages the complete lifecycle of one sample's interaction.
    Async generation is used internally when available.

    Args:
        sample_idx: Index of this sample in the original batch
        initial_sample_state: Initial state containing message_log, extra_env_info, etc.
        policy_generation: The generation interface
        tokenizer: Tokenizer to use
        task_to_env: Environment mapping
        max_seq_len: Maximum sequence length
        max_rollout_turns: Maximum number of turns
        greedy: Whether to use greedy decoding

    Returns:
        Tuple of (final_sample_state, sample_metrics)
    """
    # Initialize sample state
    current_message_log = copy.deepcopy(initial_sample_state["message_log"])
    current_extra_env_info = copy.deepcopy(initial_sample_state["extra_env_info"])
    current_stop_strings = initial_sample_state.get("stop_strings", None)
    task_name = initial_sample_state["task_name"]

    # Sample-level metrics
    total_reward = 0.0
    reward_acc_dict: dict[str, float] = {}  # per-component reward accumulators (named)
    multi_reward_seen = False
    turn_count = 0
    token_count = 0
    assistant_token_count = 0
    env_token_count = 0
    terminated = False
    truncated = False
    max_turns_reached = False

    # Per-sample invalid-action / malformed-thinking verdict counts (#2656)
    detect_invalid = invalid_action_cfg is not None and invalid_action_cfg.get(
        "enabled", False
    )
    invalid_action_count = 0
    malformed_thinking_count = 0
    # Structured tool-use protocol (None = fenced-text path, unchanged).
    structured_enabled = structured_cfg is not None
    thinking_style = _structured_thinking_style(structured_cfg)

    # Track per-turn metrics
    turn_gen_tokens = []
    turn_input_tokens = []
    turn_total_tokens = []
    # Track per-turn per-worker token accounting if available
    per_worker_token_counts = {}  # worker_idx -> token_count

    for turn in range(max_rollout_turns):
        if terminated or truncated:
            break

        turn_count += 1

        # Generate response for this sample using async generation
        try:
            (
                updated_message_log,
                generated_tokens,
                input_lengths,
                gen_metrics,
            ) = await async_generate_response_for_sample_turn(
                policy_generation,
                current_message_log,
                current_stop_strings,
                tokenizer,
                max_seq_len,
                greedy=greedy,
                structured_cfg=structured_cfg,
            )
            current_message_log = updated_message_log

            # Check if response was truncated (hit max_tokens without stop token)
            response_truncated = gen_metrics.pop("_response_truncated", None)
            if response_truncated is not None and response_truncated[0]:
                truncated = True

            # Update token counts
            gen_token_count = len(generated_tokens)
            assistant_token_count += gen_token_count
            token_count += gen_token_count
            turn_gen_tokens.append(gen_token_count)
            turn_input_tokens.append(int(input_lengths.item()))
            turn_total_tokens.append(int(input_lengths.item()) + gen_token_count)
            # Per-worker load accounting
            if "gen_leader_worker_idx" in gen_metrics:
                worker_idx = int(gen_metrics["gen_leader_worker_idx"])
                per_worker_token_counts[worker_idx] = (
                    per_worker_token_counts.get(worker_idx, 0) + gen_token_count
                )

        except Exception as e:
            print(f"Error generating response for sample {sample_idx}: {e}")
            break

        # Create single-sample batch for environment interaction
        sample_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [current_message_log],
                "extra_env_info": [current_extra_env_info],
                "task_name": [task_name],
            }
        )

        # Get environment feedback.
        # calculate_rewards uses blocking ray.get internally. Running it
        # directly on the asyncio event loop (which this coroutine runs on)
        # blocks every other in-flight rollout coroutine for the entire env
        # step. Wrap with asyncio.to_thread to make this function yieldable.
        env_output = await asyncio.to_thread(
            calculate_rewards, sample_batch, task_to_env
        )
        # Per-turn invalid-action verdict: env-stamped flags + generic text
        # detectors over this turn's assistant message (last log entry; the
        # env observation is appended below).
        if detect_invalid:
            assert invalid_action_cfg is not None
            verdict = assess_assistant_turn(
                str(current_message_log[-1].get("content", "")),
                invalid_action_cfg,
                env_output.metadata[0],
                thinking_style=thinking_style,
            )
            invalid_action_count += int(verdict.invalid_action)
            malformed_thinking_count += int(verdict.malformed_thinking)
        # Verdict keys are per-turn; never carry into next-turn extra_env_info.
        pop_env_flags(env_output.metadata[0])

        # Update total reward and optional per-component reward signals (named).
        if isinstance(env_output.rewards, dict):
            multi_reward_seen = True
            for name, r in env_output.rewards.items():
                reward_acc_dict[name] = reward_acc_dict.get(name, 0.0) + float(
                    r[0].item()
                )
            total_reward += sum(
                float(r[0].item()) for r in env_output.rewards.values()
            )
        else:
            total_reward += float(env_output.rewards[0].item())
        # Check termination
        terminated = env_output.terminateds[0].item()
        env_obs_content = env_output.observations[0]["content"]
        # Structured path: wrap in the Qwen turn envelope before tokenizing
        # (fork 3); fenced-text path tokenizes the raw content unchanged.
        obs_text_to_tokenize = (
            _wrap_tool_response_envelope(env_obs_content)
            if structured_enabled
            else env_obs_content
        )
        # Tokenize environment response
        tokenized_obs = tokenizer(
            obs_text_to_tokenize, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        # Check for sequence length overflow
        if int(input_lengths.item()) + gen_token_count + len(tokenized_obs) >= max_seq_len:
            # Truncate environment observation
            max_env_tokens = max_seq_len - int(input_lengths.item()) - gen_token_count
            if max_env_tokens > 0:
                tokenized_obs = tokenized_obs[:max_env_tokens]
            else:
                tokenized_obs = torch.empty(0, dtype=tokenized_obs.dtype)
            truncated = True

        env_message = {
            "role": env_output.observations[0]["role"],
            "content": env_obs_content,
            "token_ids": tokenized_obs,
        }
        current_message_log.append(env_message)

        # Update token counts
        env_token_count += len(tokenized_obs)
        token_count += len(tokenized_obs)

        # Update sample state for next turn
        if not terminated and not truncated:
            if env_output.next_stop_strings[0] is not None:
                current_stop_strings = env_output.next_stop_strings[0]
            if env_output.metadata[0] is not None:
                current_extra_env_info = env_output.metadata[0]

    # Check if max turns reached
    if turn_count >= max_rollout_turns:
        max_turns_reached = True

    # Prepare final sample state
    final_sample_state = {
        "message_log": current_message_log,
        "extra_env_info": current_extra_env_info,
        "task_name": task_name,
        "total_reward": torch.tensor(total_reward),
        "stop_strings": current_stop_strings,
        "idx": sample_idx,
    }
    if detect_invalid:
        final_sample_state["invalid_action_count"] = invalid_action_count
        final_sample_state["malformed_thinking_count"] = malformed_thinking_count
    if multi_reward_seen:
        for name, acc in reward_acc_dict.items():
            final_sample_state[name] = torch.tensor(acc)

    # Sample metrics
    sample_metrics = {
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
        # Pass-through per-worker per-turn accounting for aggregation at batch level
        "per_worker_token_counts": per_worker_token_counts,
    }

    return final_sample_state, sample_metrics

def run_async_multi_turn_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
    invalid_action_cfg: Optional[InvalidActionPenaltyConfig] = None,
    structured_cfg: Optional[StructuredToolUseConfig] = None,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, Any]]:
    """Run multi-turn rollouts with sample-level processing.

    Each sample in the batch proceeds through its interaction independently.
    Async generation is used internally when available but the function is synchronous.

    Args:
        policy_generation: The generation interface (policy)
        input_batch: The starting batch containing initial message logs
        tokenizer: The tokenizer
        task_to_env: Dictionary mapping task names to environment instances
        max_seq_len: Maximum sequence length allowed
        max_rollout_turns: Maximum number of agent-environment interaction turns
        greedy: Whether to use greedy decoding

    Returns:
        Tuple containing:
            - BatchedDataDict with the full interaction history and accumulated rewards
            - Dictionary of rollout metrics
    """

    async def _async_rollout_implementation():
        """Internal async implementation."""
        batch_size = len(input_batch["message_log"])

        # Prepare initial states for each sample
        sample_initial_states = []
        for i in range(batch_size):
            sample_state = {
                "message_log": input_batch["message_log"][i],
                "extra_env_info": input_batch["extra_env_info"][i],
                "task_name": input_batch["task_name"][i],
                "stop_strings": input_batch.get("stop_strings", [None] * batch_size)[i],
                "idx": input_batch.get("idx", list(range(batch_size)))[i],
            }
            sample_initial_states.append(sample_state)

        # Run all samples concurrently
        async def run_single_sample_with_error_handling(i, sample_state):
            """Wrapper to handle errors for individual sample rollouts."""
            try:
                result = await run_sample_multi_turn_rollout(
                    sample_idx=i,
                    initial_sample_state=sample_state,
                    policy_generation=policy_generation,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=max_seq_len,
                    max_rollout_turns=max_rollout_turns,
                    greedy=greedy,
                    invalid_action_cfg=invalid_action_cfg,
                    structured_cfg=structured_cfg,
                )
                return result
            except Exception as e:
                raise RuntimeError(f"Error in sample {i} rollout: {e}") from e

        # Create tasks for all samples and run them concurrently
        sample_tasks = [
            run_single_sample_with_error_handling(i, sample_state)
            for i, sample_state in enumerate(sample_initial_states)
        ]

        # Execute all sample rollouts concurrently
        sample_results = await asyncio.gather(*sample_tasks, return_exceptions=False)

        # Process results
        final_sample_states = []
        all_sample_metrics = []

        for final_state, sample_metrics in sample_results:
            final_sample_states.append(final_state)
            all_sample_metrics.append(sample_metrics)

        # Reconstruct batch from sample results
        final_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [state["message_log"] for state in final_sample_states],
                "extra_env_info": [
                    state["extra_env_info"] for state in final_sample_states
                ],
                "task_name": [state["task_name"] for state in final_sample_states],
                "total_reward": torch.stack(
                    [state["total_reward"] for state in final_sample_states]
                ),
                "idx": [
                    state.get("idx", i)
                    for i, state in enumerate(final_sample_states)
                ],
                "truncated": torch.tensor(
                    [metrics["truncated"] for metrics in all_sample_metrics],
                    dtype=torch.bool,
                ),
            }
        )

        if invalid_action_cfg is not None and invalid_action_cfg.get("enabled", False):
            final_batch["invalid_action_count"] = torch.tensor(
                [state["invalid_action_count"] for state in final_sample_states],
                dtype=torch.int32,
            )
            final_batch["malformed_thinking_count"] = torch.tensor(
                [state["malformed_thinking_count"] for state in final_sample_states],
                dtype=torch.int32,
            )

        # Expose per-component rewards for multi-reward envs (GDPO advantage calculation).
        # Collect named reward keys (e.g. "reward/correctness") from sample states.
        reward_component_keys = sorted(
            set(
                k
                for state in final_sample_states
                for k in get_gdpo_reward_component_keys(state)
            )
        )
        for key in reward_component_keys:
            # Stack per-sample values; use 0.0 for samples that did not have this component
            final_batch[key] = torch.stack(
                [
                    state[key]
                    if key in state
                    else torch.tensor(0.0, dtype=torch.float32)
                    for state in final_sample_states
                ]
            )

        # Preserve additional fields from the original input_batch
        for key in input_batch.keys():
            if key not in final_batch:
                final_batch[key] = input_batch[key]

        # Aggregate metrics across all samples
        rollout_metrics = {
            # Overall metrics
            "total_turns": sum(m["turn_count"] for m in all_sample_metrics),
            "avg_turns_per_sample": sum(m["turn_count"] for m in all_sample_metrics)
            / batch_size,
            "max_turns_per_sample": max(m["turn_count"] for m in all_sample_metrics),
            "natural_termination_rate": sum(
                m["terminated"] for m in all_sample_metrics
            )
            / batch_size,
            "truncation_rate": sum(m["truncated"] for m in all_sample_metrics)
            / batch_size,
            "max_turns_reached_rate": sum(
                m["max_turns_reached"] for m in all_sample_metrics
            )
            / batch_size,
            # Token usage metrics
            "mean_total_tokens_per_sample": sum(
                m["total_tokens"] for m in all_sample_metrics
            )
            / batch_size,
            "mean_gen_tokens_per_sample": sum(
                m["assistant_tokens"] for m in all_sample_metrics
            )
            / batch_size,
            "max_gen_tokens_per_sample": max(
                m["assistant_tokens"] for m in all_sample_metrics
            ),
            "mean_env_tokens_per_sample": sum(
                m["env_tokens"] for m in all_sample_metrics
            )
            / batch_size,
            # Reward metrics
            "mean_total_reward": sum(
                m["total_reward"] for m in all_sample_metrics
            )
            / batch_size,
            "max_total_reward": max(m["total_reward"] for m in all_sample_metrics),
            "min_total_reward": min(m["total_reward"] for m in all_sample_metrics),
        }

        if "invalid_action_count" in final_batch:
            rollout_metrics["invalid_action_rate"] = float(
                (final_batch["invalid_action_count"] > 0).float().mean().item()
            )
            rollout_metrics["malformed_thinking_rate"] = float(
                (final_batch["malformed_thinking_count"] > 0).float().mean().item()
            )

        # Calculate per-worker token counts
        if "per_worker_token_counts" in all_sample_metrics[0]:
            per_worker_token_counts = {}
            for m in all_sample_metrics:
                worker_counts = cast(dict[str, int], m["per_worker_token_counts"])
                for k, v in worker_counts.items():
                    per_worker_token_counts[k] = (
                        per_worker_token_counts.get(k, 0) + v
                    )
            rollout_metrics["per_worker_token_counts"] = per_worker_token_counts

        # Collect ISL, OSL, and ISL+OSL metrics for all samples
        rollout_metrics["histogram/gen_tokens_length"] = [
            t for m in all_sample_metrics for t in m["turn_gen_tokens"]
        ]
        rollout_metrics["histogram/input_tokens_length"] = [
            t for m in all_sample_metrics for t in m["turn_input_tokens"]
        ]
        rollout_metrics["histogram/total_tokens_length"] = [
            t for m in all_sample_metrics for t in m["turn_total_tokens"]
        ]

        return final_batch, rollout_metrics

    timer = Timer() if Timer is not None else None
    if timer is not None:
        timer.start("timing/rollout/total")

    final_batch, rollout_metrics = asyncio.run(_async_rollout_implementation())

    if timer is not None:
        timer.stop("timing/rollout/total")
        rollout_metrics.update(timer.get_timing_metrics("sum"))

    return final_batch, rollout_metrics