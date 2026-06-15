"""Contains data processors for training and evaluation."""

import json
import logging
from typing import Any, Dict, Optional, cast
import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase
from dockyard_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    PreferenceDatumSpec,
    TaskDataProcessFnCallable,
    TaskDataSpec,
    VLMMessageLogType,
)
from dockyard_rl.data.llm_message_utils import get_formatted_message_log
from dockyard_rl.tool_protocol.protocol import (
    STRUCTURED_TOOL_USE_KEY,
    resolve_structured_tool_use,
)
from dockyard_rl.tool_protocol.registry import CODE_TOOLS, SESSION_TOOLS, ToolRegistry

TokenizerType = PreTrainedTokenizerBase


def _resolve_structured_tools(
    task_data_spec: TaskDataSpec,
    registry: ToolRegistry,
) -> Optional[list[dict[str, Any]]]:
    """Return the prompt-advertised tool specs when the protocol is enabled.

    The structured tool-use protocol is gated by the ``grpo.structured_tool_use``
    config block (``tool_protocol.protocol``). That block is threaded onto the
    per-task :class:`TaskDataSpec` by the dataset-loading layer under the
    :data:`STRUCTURED_TOOL_USE_KEY` attribute; reading it here with ``getattr``
    keeps the protocol off (and the prompt byte-identical to the fenced-text
    path) whenever the attribute is absent or the block is disabled.

    When enabled, returns ``registry.chat_template_tools()`` — the OpenAI-style
    function specs ``apply_chat_template(..., tools=…)`` renders into the system
    turn (design fork 1/4). When disabled, returns ``None`` so the downstream
    ``tools`` argument is omitted entirely, leaving the chat-template call exactly
    as it was before this slice.

    ``registry`` is the environment's frozen tool set: ``CODE_TOOLS`` for the
    single-shot SWE coding env, ``SESSION_TOOLS`` for the multi-turn session envs.
    """
    config_block = getattr(task_data_spec, STRUCTURED_TOOL_USE_KEY, None)
    if resolve_structured_tool_use(config_block) is None:
        return None
    return registry.chat_template_tools()

def helpsteer3_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a HelpSteer3 preference datum into a DatumSpec for GRPO training.

    This function converts HelpSteer3 preference data to work with GRPO by:
    1. Using the context as the prompt
    2. Using the preferred completion as the target response
    3. Creating a reward signal based on preference scores
    """
    # Extract context and completions from HelpSteer3 format
    context = datum_dict["context"]
    preferred_completion = datum_dict["response"]

    # Build the conversation from context
    message_log: LLMMessageLogType = []

    # Add context messages
    if isinstance(context, list):
        for msg in context:
            message_log.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )
    else:
        # If context is a string, treat it as a user message
        message_log.append(
            {
                "role": "user",
                "content": context,
            }
        )

    # Add the preferred completion as the target
    for completion_msg in preferred_completion:
        message_log.append(
            {
                "role": completion_msg["role"],
                "content": completion_msg["content"],
            }
        )

    # Apply chat template and tokenize
    formatted_conversation = cast(str, tokenizer.apply_chat_template(
        cast(list[dict[str, str]], message_log),
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=True,
    ))

    # Tokenize the entire conversation
    full_tokens = tokenizer(
        formatted_conversation,
        return_tensors="pt",
        add_special_tokens=False,  # Already added by chat template
    )["input_ids"][0]

    # For simplicity, assign all tokens to the first message.
    # In a more sophisticated implementation, you might want to split tokens properly.
    message_log[0]["token_ids"] = full_tokens
    message_log[0]["content"] = formatted_conversation

    # Clear token_ids for other messages to avoid double counting
    for i in range(1, len(message_log)):
        message_log[i]["token_ids"] = tokenizer("", return_tensors="pt")["input_ids"][
            0
        ]  # Empty tensor

    length = sum(len(m["token_ids"]) for m in message_log)

    # Create ground truth from the preferred completion for environment evaluation
    ground_truth = " ".join([msg["content"] for msg in preferred_completion])
    extra_env_info = {"ground_truth": ground_truth}

    loss_multiplier = 1.0
    if length >= max_seq_length:
        # Truncate if too long
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(
                    max_seq_length // len(message_log), len(chat_message["token_ids"])
                )
            ]
        loss_multiplier = 0.0  # Reduce loss for truncated sequences

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def sft_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
    add_bos: bool = True,
    add_eos: bool = True,
    add_generation_prompt: bool = False,
) -> DatumSpec:
    """Process a datum dictionary for SFT training."""
    # optional preprocessor
    if datum_dict["task_name"] == "clevr-cogent":
        from dockyard_rl.data.datasets.response_datasets.clevr import (  # type: ignore[import]
            format_clevr_cogent_dataset,
        )

        datum_dict = format_clevr_cogent_dataset(datum_dict)

    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
        tools=datum_dict.get("tools", None),  # Pass tools from data if present
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length >= max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output

def preference_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> PreferenceDatumSpec:
    """Process a datum dictionary for RM/DPO training."""
    assert len(datum_dict["completions"]) == 2, (
        "RM/DPO training supports only two completions"
    )
    # Lower rank is preferred
    if datum_dict["completions"][0]["rank"] < datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][0]
        rejected_completion = datum_dict["completions"][1]
    elif datum_dict["completions"][0]["rank"] > datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][1]
        rejected_completion = datum_dict["completions"][0]
    else:
        raise NotImplementedError(
            "Ties are not supported yet. You can use the following command to filter out ties: "
            "`cat <PathToPreferenceDataset> | jq 'select(.completions[0].rank != .completions[1].rank)'`."
        )

    messages_chosen = datum_dict["context"] + chosen_completion["completion"]
    messages_rejected = datum_dict["context"] + rejected_completion["completion"]

    message_log_chosen = get_formatted_message_log(
        messages_chosen, tokenizer, task_data_spec
    )
    message_log_rejected = get_formatted_message_log(
        messages_rejected, tokenizer, task_data_spec
    )

    length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
    length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

    loss_multiplier = 1.0
    if max(length_chosen, length_rejected) > max_seq_length:
        logging.warning(
            f"Sequence length {max(length_chosen, length_rejected)} exceeds max_seq_length {max_seq_length}. Ignoring example."
        )

        # make smaller and mask out
        for message in message_log_chosen:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_chosen))
            ]
        for message in message_log_rejected:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_rejected))
            ]
        loss_multiplier = 0.0

        length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
        length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

        # safeguard against edge case where there are too many turns to fit within the max length
        assert max(length_chosen, length_rejected) <= max_seq_length

    output: PreferenceDatumSpec = {
        "message_log_chosen": message_log_chosen,
        "message_log_rejected": message_log_rejected,
        "length_chosen": length_chosen,
        "length_rejected": length_rejected,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output

# Generic text→prompt preprocessor used by the evaluation datasets
# (data/datasets/eval_datasets/*). Builds a (system +) user prompt from
# datum_dict["input"] and carries any ground-truth answer
# (datum_dict["ground_truth"]) in extra_env_info for the verifier environment.
def text_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_input_seq_length: int,
    idx: int = 0,
) -> DatumSpec:
    """Tokenize a single text prompt into a DatumSpec for evaluation rollouts.

    The prompt text is ``datum_dict["input"]``; the optional
    ``datum_dict["ground_truth"]`` is threaded into ``extra_env_info`` so the
    scoring environment can verify the generated answer. Mirrors the prompt
    construction of ``math_data_processor`` so eval samples collate and roll out
    identically to training samples.
    """
    problem = datum_dict["input"]
    message_log: LLMMessageLogType = []

    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = cast(str, tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        ))
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    if task_data_spec.prompt:
        problem = cast(str, task_data_spec.prompt).format(problem)
    user_message = {"role": "user", "content": problem}
    message = cast(str, tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ))
    user_message["token_ids"] = tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length >= max_input_seq_length:
        for indiv_message in message_log:
            indiv_message["token_ids"] = indiv_message["token_ids"][
                : min(4, max_input_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": {"ground_truth": str(datum_dict.get("ground_truth", ""))},
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

# Example of a generic math data processor
def math_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for the Math Environment."""
    problem = datum_dict["problem"]
    solution = str(datum_dict["expected_answer"])
    extra_env_info = {"ground_truth": solution}

    message_log: LLMMessageLogType = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = cast(str, tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        ))
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        problem = cast(str, task_data_spec.prompt).format(problem)
    user_message = {"role": "user", "content": problem}
    message = cast(str, tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ))
    user_message["token_ids"] = tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length >= max_seq_length:
        # make smaller and mask out
        for indiv_message in message_log:
            indiv_message["token_ids"] = indiv_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def math_hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from data/hf_datasets/openmathinstruct2.py)
    into a DatumSpec for the Reward Model Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    # merge system prompt and user prompt
    message_list = []
    if task_data_spec.system_prompt:
        message_list.append(
            {
                "role": "system",
                "content": task_data_spec.system_prompt,
            }
        )
    formatted_content = (
        cast(str, task_data_spec.prompt).format(problem) if task_data_spec.prompt else problem
    )
    message_list.append({"role": "user", "content": formatted_content})

    message: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
        message_list,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    token_ids = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    message_log: LLMMessageLogType = [
        {"role": "user", "content": message, "token_ids": token_ids}
    ]

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length >= max_seq_length:
        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output

def vlm_hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from response_datasets/<dataset_name>.py)
    into a DatumSpec for the VLM Environment."""
    from dockyard_rl.data.datasets.response_datasets.clevr import (  # type: ignore[import]
        format_clevr_cogent_dataset,
    )
    from dockyard_rl.data.datasets.response_datasets.geometry3k import (  # type: ignore[import]
        format_geometry3k_dataset,
    )
    from dockyard_rl.data.datasets.response_datasets.refcoco import format_refcoco_dataset  # type: ignore[import]
    from dockyard_rl.data.multimodal_utils import (  # type: ignore[import]
        PackedTensor,
        get_dim_to_pack_along,
        get_multimodal_keys_from_processor,
        resolve_to_image,
    )

    # depending on the task, format the data differently
    if datum_dict["task_name"] == "clevr-cogent":
        datum_dict = format_clevr_cogent_dataset(datum_dict)
    elif datum_dict["task_name"] == "refcoco":
        datum_dict = format_refcoco_dataset(datum_dict)
    elif datum_dict["task_name"] == "geometry3k":
        datum_dict = format_geometry3k_dataset(datum_dict)
    elif datum_dict["task_name"] == "avqa":
        pass  # AVQA data is already formatted by AVQADataset.format_data
    elif datum_dict["task_name"] == "mmau":
        pass  # MMAU data is already formatted by MMAUDataset.format_data
    else:
        raise ValueError(f"No data processor for task {datum_dict['task_name']}")

    user_message_raw = datum_dict["messages"]
    problem = user_message_raw[0]["content"]
    extra_env_info = {"ground_truth": user_message_raw[1]["content"]}
    if "choices" in datum_dict:
        extra_env_info["choices"] = datum_dict["choices"]

    message_log: VLMMessageLogType = []
    user_message: dict[str, Any] = {"role": "user", "content": []}

    images = []
    audios = []
    if isinstance(problem, list):
        for content in problem:
            if content["type"] == "text":
                user_message["content"].append(
                    {
                        "type": "text",
                        "text": cast(str, task_data_spec.prompt).format(content["text"])
                        if task_data_spec.prompt
                        else content["text"],
                    }
                )
            elif content["type"] == "image":
                user_message["content"].append(content)
                images.append(content["image"])
            elif content["type"] == "audio":
                user_message["content"].append(content)
                audios.append(
                    (content["audio"], processor.feature_extractor.sampling_rate)
                )
            else:
                raise ValueError(f"Unsupported content type: {content['type']}")
    else:
        user_message["content"] = (
            cast(str, task_data_spec.prompt).format(problem)
            if task_data_spec.prompt
            else problem
        )

    images = [resolve_to_image(image) for image in images]

    # get formatted user message
    if hasattr(processor, "conversation_preprocessor"):
        user_message_for_chat_template = processor.conversation_preprocessor(
            user_message
        )
    else:
        user_message_for_chat_template = user_message

    # the string-tokenized conversation template for the generation policy (for vllm)
    string_formatted_dialog = processor.apply_chat_template(
        [user_message_for_chat_template],
        tokenize=False,
        add_generation_prompt=True,
    )

    # the id-tokenized and image processed conversation template for the policy
    message: dict = processor.apply_chat_template(
        [user_message],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # add this for backward compatibility
    user_message["token_ids"] = message["input_ids"][0]
    # add all keys and values to the user message
    multimodal_keys = get_multimodal_keys_from_processor(processor)
    for key in multimodal_keys:
        if key in message:
            user_message[key] = PackedTensor(
                message[key], dim_to_pack=get_dim_to_pack_along(processor, key)
            )

    # specifically for gemma, we need to add token_type_ids to the user message as a sequence-type value
    if "token_type_ids" in message:
        user_message["token_type_ids"] = message["token_type_ids"][0]

    # for qwen2.5-vl (transformers>=5.3), mm_token_type_ids tells the model which tokens are text/image/video for 3D RoPE
    if "mm_token_type_ids" in message:
        user_message["mm_token_type_ids"] = message["mm_token_type_ids"][0]

    message_log.append(user_message)

    length = sum(len(cast(torch.Tensor, m["token_ids"])) for m in message_log)
    loss_multiplier = 1.0
    if length >= max_seq_length:
        # Treat truncated messages as text only
        vllm_kwargs = {
            "vllm_content": None,
            "vllm_images": [],
            "vllm_audios": [],
        }

        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = cast(torch.Tensor, chat_message["token_ids"])[
                : min(4, max_seq_length // len(message_log))
            ]
            for key, value in chat_message.items():
                if isinstance(value, PackedTensor):
                    chat_message[key] = PackedTensor.empty_like(value)
        loss_multiplier = 0.0
    else:
        # get the prompt content for vllm-backend that needs formatted dialog and list of images/audios
        vllm_kwargs = {
            "vllm_content": string_formatted_dialog,
            "vllm_images": images,
            "vllm_audios": audios,
        }

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
        **vllm_kwargs,  # pyrefly: ignore[bad-unpacking]
    }
    return output

def _construct_multichoice_prompt(
    prompt: str, question: str, options: dict[str, str]
) -> str:
    """Construct prompt from question and options."""
    output = prompt
    output += f"\n\nQuestion: {question}\nOptions:\n"
    output += "\n".join(
        [
            f"{letter}) {option}"
            for letter, option in options.items()
            if option is not None
        ]
    )
    return output

def multichoice_qa_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for multiple-choice problems."""
    question = datum_dict["question"]
    answer = str(datum_dict["answer"])
    options = datum_dict["options"]
    extra_env_info = {"ground_truth": answer}
    if "subject" in datum_dict:
        extra_env_info.update({"subject": datum_dict["subject"]})

    message_log: LLMMessageLogType = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = cast(str, tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        ))
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        question = _construct_multichoice_prompt(
            task_data_spec.prompt, question, options
        )
    user_message = {"role": "user", "content": question}
    message = cast(str, tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ))
    user_message["token_ids"] = tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": 1.0,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def swe_bench_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process a SWE-bench row (from ``SWEBenchDataset.format_data``) into a DatumSpec.

    The prompt is the system + user message log with no answer turn (the agent
    generates the fix at rollout time). The held-out test metadata is routed
    into ``extra_env_info`` so the reward function can verify the solution.
    """
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_eos_token=False,  # prompt only — the agent's turn is generated
        add_generation_prompt=True,
        # CodeEnvironment (env_name "code") → submit_patch tool registry.
        tools=_resolve_structured_tools(task_data_spec, CODE_TOOLS),
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    extra_env_info = {
        "instance_id":              datum_dict["instance_id"],
        "repo":                     datum_dict["repo"],
        "base_commit":              datum_dict["base_commit"],
        "environment_setup_commit": datum_dict["environment_setup_commit"],
        "fail_to_pass":             datum_dict["fail_to_pass"],
        "pass_to_pass":             datum_dict["pass_to_pass"],
        # Held-out scoring inputs (never in the prompt): the gold test diff and
        # the reference fix diff, consumed only by the terminal-turn reward.
        "test_patch":               datum_dict.get("test_patch", ""),
        "gold_patch":               datum_dict.get("gold_patch", ""),
    }

    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def swe_bench_pro_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process a SWE-bench Pro row (from ``SWEBenchProDataset.format_data``).

    The prompt is the system + user message log with no answer turn. The
    held-out scoring inputs — the per-instance image, harness scripts, the gold
    test checkout command, the selected test files, the named fail/pass tests,
    and the gold test/fix diffs — are routed into ``extra_env_info`` so the
    image-mode reward can score the solution without exposing them to the agent.
    """
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_eos_token=False,
        add_generation_prompt=True,
        # CodeEnvironment (env_name "code") → submit_patch tool registry.
        tools=_resolve_structured_tools(task_data_spec, CODE_TOOLS),
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    extra_env_info = {
        "instance_id":         datum_dict["instance_id"],
        "repo":                datum_dict["repo"],
        "base_commit":         datum_dict["base_commit"],
        "repo_language":       datum_dict["repo_language"],
        "image":               datum_dict["image"],
        "before_repo_set_cmd": datum_dict["before_repo_set_cmd"],
        "selected_test_files": datum_dict["selected_test_files"],
        "fail_to_pass":        datum_dict["fail_to_pass"],
        "pass_to_pass":        datum_dict["pass_to_pass"],
        "test_patch":          datum_dict.get("test_patch", ""),
        "gold_patch":          datum_dict.get("gold_patch", ""),
        "run_script":          datum_dict["run_script"],
        "parser_script":       datum_dict["parser_script"],
    }

    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def terminal_bench_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process a Terminal-Bench row (from ``TerminalBenchDataset._build_row``).

    The prompt is the system + user message log with no answer turn. Everything
    the multi-turn environment needs — the image/Dockerfile + build context, the
    held-out ``tests/`` (injected only at finish), the verifier command/mount/
    result paths, timeouts, container env, network policy, and turn budget — is
    routed into ``extra_env_info``. The dict-valued fields are carried as JSON
    strings in the dataset row (Arrow cannot hold variable-key structs) and
    decoded back here.
    """
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_eos_token=False,
        add_generation_prompt=True,
        # MultiTurnSessionEnvironment → run_shell/read_file/write_file/task_complete.
        tools=_resolve_structured_tools(task_data_spec, SESSION_TOOLS),
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    extra_env_info = {
        "task_id":             datum_dict["task_id"],
        "mode":                datum_dict["mode"],
        "image":               datum_dict["image"],
        "dockerfile":          datum_dict["dockerfile"],
        "build_context":       json.loads(datum_dict["build_context_json"]),
        "tests":               json.loads(datum_dict["tests_json"]),
        "container_env":       json.loads(datum_dict["container_env_json"]),
        "allow_internet":      datum_dict["allow_internet"],
        "harness_mount":       datum_dict["harness_mount"],
        "test_command":        datum_dict["test_command"],
        "result_files":        datum_dict["result_files"],
        "agent_timeout_sec":   datum_dict["agent_timeout_sec"],
        "verifier_timeout_sec": datum_dict["verifier_timeout_sec"],
        "build_timeout_sec":   datum_dict["build_timeout_sec"],
        "exec_timeout_sec":    datum_dict["exec_timeout_sec"],
        "max_turns":           datum_dict["max_turns"],
    }

    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def program_bench_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process a ProgramBench row (from ``ProgramBenchDataset._build_row``).

    The prompt is the system + user message log with no answer turn. The image
    ref, the held-out test branches (base64 tars + expected node IDs, carried as
    a JSON string since Arrow cannot hold variable-key structs), language, and
    the per-episode budgets are routed into ``extra_env_info`` for the multi-turn
    environment and the clean-rebuild grading reward.
    """
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_eos_token=False,
        add_generation_prompt=True,
        # MultiTurnSessionEnvironment → run_shell/read_file/write_file/task_complete.
        tools=_resolve_structured_tools(task_data_spec, SESSION_TOOLS),
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    extra_env_info = {
        "task_id":             datum_dict["task_id"],
        "language":            datum_dict["language"],
        "image":               datum_dict["image"],
        "branches":            json.loads(datum_dict["branches_json"]),
        "max_turns":           datum_dict["max_turns"],
        "exec_timeout_sec":    datum_dict["exec_timeout_sec"],
        "grading_timeout_sec": datum_dict["grading_timeout_sec"],
    }

    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def hle_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process an HLE row (from ``HLEDataset.format_data``) into a DatumSpec.

    The prompt is the system (HLE response format) + user (question) message log
    with no answer turn. The gold answer, the question (needed by the LLM judge),
    and the answer type/category are routed into ``extra_env_info`` for the HLE
    verifier environment.
    """
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_eos_token=False,
        add_generation_prompt=True,
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    extra_env_info = {
        "ground_truth": datum_dict["ground_truth"],
        "question":     datum_dict["question"],
        "answer_type":  datum_dict.get("answer_type", ""),
        "category":     datum_dict.get("category", ""),
        "id":           datum_dict.get("id", ""),
    }

    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def gdpval_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process a GDPval row (from ``GDPvalDataset.format_data``) into a DatumSpec.

    The prompt is the system (text-deliverable instructions) + user (task prompt)
    message log with no answer turn. The task prompt and ``rubric_json`` (needed
    by the rubric judge) plus occupation/sector are routed into ``extra_env_info``.
    """
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_eos_token=False,
        add_generation_prompt=True,
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    extra_env_info = {
        "prompt":      datum_dict["prompt"],
        "rubric_json": datum_dict["rubric_json"],
        "occupation":  datum_dict.get("occupation", ""),
        "sector":      datum_dict.get("sector", ""),
        "task_id":     datum_dict.get("task_id", ""),
    }

    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def gdpval_agentic_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process a GDPval agentic row (from ``GDPvalAgenticDataset.format_data``).

    The prompt is the system (tool-use instructions) + user (task prompt) message
    log with no answer turn. Everything the multi-turn file-producing environment
    needs — the container image, the deliverable/reference dirs, the turn/exec
    budgets, the task prompt and ``rubric_json`` for the judge, and any base64
    reference files (carried as a JSON string since Arrow cannot hold variable-key
    structs) — is routed into ``extra_env_info``.
    """
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_eos_token=False,
        add_generation_prompt=True,
        # GDPvalAgenticEnvironment subclasses MultiTurnSessionEnvironment →
        # run_shell/read_file/write_file/task_complete.
        tools=_resolve_structured_tools(task_data_spec, SESSION_TOOLS),
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    extra_env_info = {
        "prompt":           datum_dict["prompt"],
        "rubric_json":      datum_dict["rubric_json"],
        "occupation":       datum_dict.get("occupation", ""),
        "sector":           datum_dict.get("sector", ""),
        "task_id":          datum_dict.get("task_id", ""),
        "image":            datum_dict["image"],
        "deliverable_dir":  datum_dict["deliverable_dir"],
        "reference_dir":    datum_dict["reference_dir"],
        "max_turns":        datum_dict["max_turns"],
        "exec_timeout_sec": datum_dict["exec_timeout_sec"],
        "reference_files":  json.loads(datum_dict.get("reference_files_json", "{}")),
    }

    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output

def gym_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for Gym.

    Note: this processor is kept for registry compatibility but Gym integration
    is not a Dockyard target. The processor is deferred-importable without side effects.
    """
    output: DatumSpec = {
        # load to dict format here since `Dataset` cannot handle nested structure well in `GymDataset`
        "extra_env_info": json.loads(datum_dict["extra_env_info"]),
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": datum_dict["task_name"],
        # fake keys for compatibility with the current GRPO implementation
        "message_log": [{"role": "user", "content": "", "token_ids": torch.tensor([])}],
        "length": 0,
    }
    return output

# Processor registry. Key is the processor name, value is the processor function.
# Note: We cast the literal dict to Dict[str, TaskDataProcessFnCallable] because
# type checkers see each concrete function's signature as a distinct callable type.
# Without the cast, the registry's inferred type becomes a union of those specific
# callables, which is not assignable to the uniform TaskDataProcessFnCallable.
# The cast asserts our intent that all entries conform to the common callable protocol.
PROCESSOR_REGISTRY: Dict[str, TaskDataProcessFnCallable] = cast(
    Dict[str, TaskDataProcessFnCallable],
    {
        "default": math_hf_data_processor,
        "helpsteer3_data_processor": helpsteer3_data_processor,
        "math_data_processor": math_data_processor,
        "math_hf_data_processor": math_hf_data_processor,
        "multichoice_qa_processor": multichoice_qa_processor,
        "sft_processor": sft_processor,
        "swe_bench_data_processor": swe_bench_data_processor,
        "swe_bench_pro_data_processor": swe_bench_pro_data_processor,
        "terminal_bench_data_processor": terminal_bench_data_processor,
        "program_bench_data_processor": program_bench_data_processor,
        "hle_data_processor": hle_data_processor,
        "gdpval_data_processor": gdpval_data_processor,
        "gdpval_agentic_data_processor": gdpval_agentic_data_processor,
        "vlm_hf_data_processor": vlm_hf_data_processor,
        "gym_data_processor": gym_data_processor,
    },
)

def register_processor(
    processor_name: str, processor_function: TaskDataProcessFnCallable
) -> None:
    if processor_name in PROCESSOR_REGISTRY:
        raise ValueError(f"Processor name {processor_name} already registered")
    PROCESSOR_REGISTRY[processor_name] = processor_function

    print(f"[INFO] Dataset processor {processor_name} registered")