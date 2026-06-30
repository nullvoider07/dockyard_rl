import warnings
from typing import Any, Optional, Union, cast
import numpy as np
import torch
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dockyard_rl.data.interfaces import (
    FlatMessagesType,
    LLMMessageLogType,
    TaskDataSpec,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

Tensor = torch.Tensor
TokenizerType = PreTrainedTokenizerBase

def message_log_to_flat_messages(
    message_log: LLMMessageLogType,
) -> FlatMessagesType:
    """Convert a message log (sequence of message turns) into a flattened representation.

    This function takes a message log (list of dict messages with 'role', 'content', 'token_ids', etc.)
    and converts it to a flat dictionary where all tensors of the same key are concatenated and
    all strings of the same key are put into lists.

    Args:
        message_log: List of message dictionaries with 'role', 'content', and potentially 'token_ids'

    Returns:
        FlatMessagesType: Dictionary mapping keys to concatenated tensors and string lists
    """
    result: dict[str, list[Any]] = {}

    if len(message_log) == 0:
        return cast(FlatMessagesType, result)

    # Get all unique keys across all messages
    all_keys: set[str] = set()
    for msg in message_log:
        all_keys.update(msg.keys())

    # Initialize result with empty lists for each key
    for key in all_keys:
        result[key] = []

    # Collect values for each key
    for msg in message_log:
        for key in all_keys:
            if key in msg:
                result[key].append(msg[key])

    # Concatenate tensors for each key
    concat: FlatMessagesType = {}
    for key in result:
        if result[key] and isinstance(result[key][0], Tensor):
            try:
                concat[key] = torch.cat(result[key])
            except RuntimeError as e:
                if "same number of dimensions" in str(e):
                    raise RuntimeError(
                        f"tensors for {key=} must have same number of dimensions: {[t.shape for t in result[key]]}"
                    ) from e
                raise
        else:
            # Check for PackedTensor (multimodal) — deferred to avoid hard dependency
            try:
                from dockyard_rl.data.multimodal_utils import PackedTensor  # type: ignore[import]

                if result[key] and isinstance(result[key][0], PackedTensor):
                    try:
                        # PackedTensor is an optional, deferred type not expressible in
                        # FlatMessagesType's value union; cast to satisfy the checker.
                        concat[key] = cast(Any, PackedTensor.concat(result[key]))
                    except Exception as e:
                        raise RuntimeError(
                            f"Error concatenating packed multimodal data for {key=}"
                        ) from e
            except ImportError:
                pass  # multimodal_utils not yet ported; skip PackedTensor handling

    output: FlatMessagesType = {**result, **concat}
    return output

def get_keys_from_message_log(
    message_log: LLMMessageLogType, keys: list[str]
) -> LLMMessageLogType:
    """Return a new LLMMessageLogType containing only the specified keys from each message.

    Args:
        message_log: Original message log to extract keys from
        keys: List of keys to keep in each message

    Returns:
        LLMMessageLogType: New list with only specified keys
    """
    return [{k: msg[k] for k in keys if k in msg} for msg in message_log]

def add_loss_mask_to_message_log(
    batch_message_log: list[LLMMessageLogType],
    roles_to_train_on: Optional[list[str]] = None,
    only_unmask_final: bool = False,
) -> None:
    """Add token-level loss masks to each message in a message log.

    Args:
        batch_message_log (list[LLMMessageLogType]): Batched list of message logs
        roles_to_train_on (list[str]): List of strings indicating which speakers to unmask. Default: ["assistant"]
        only_unmask_final (bool): If True, only unmask the final message in the log. Default: False
    """
    # Use a fresh list rather than a mutable default, and normalise into a new
    # list so we never mutate the caller's argument in place.
    if roles_to_train_on is None:
        roles_to_train_on = ["assistant"]
    roles_to_train_on = [role.lower() for role in roles_to_train_on]

    for message_log in batch_message_log:
        for i, message in enumerate(message_log):
            if only_unmask_final:
                if i == len(message_log) - 1:
                    message["token_loss_mask"] = torch.ones_like(
                        cast(Tensor, message["token_ids"])
                    )
                else:
                    message["token_loss_mask"] = torch.zeros_like(
                        cast(Tensor, message["token_ids"])
                    )
            else:
                if message["role"] in roles_to_train_on:
                    message["token_loss_mask"] = torch.ones_like(
                        cast(Tensor, message["token_ids"])
                    )
                else:
                    message["token_loss_mask"] = torch.zeros_like(
                        cast(Tensor, message["token_ids"])
                    )

def backfill_routed_experts(message_logs: list, *, sentinel: int) -> None:
    """Backfill MoE router-replay routing placeholders (#2908) in place.

    The captured ``routed_experts`` column rides only on rollout-generated
    assistant messages. Without a placeholder on the other messages (prompt /
    tool / non-generated turns), ``message_log_to_flat_messages`` concatenates a
    short tensor that then right-pads to the front — silently placing the
    assistant routing at the wrong (prompt) token offsets. For every message log
    that carries routing on at least one turn, this fills each message lacking it
    with a ``[token_len, L, K]`` ``sentinel`` tensor, so the flat column
    interleaves at the same offsets as ``token_ids`` (mirrors the
    ``generation_logprobs`` zero-backfill). Logs with no routing at all are left
    untouched.
    """
    for message_log in message_logs:
        reference = None
        for message in message_log:
            recorded = message.get("routed_experts")
            if recorded is not None:
                reference = cast(Tensor, recorded)
                break
        if reference is None:
            continue
        route_shape = tuple(reference.shape[1:])
        for message in message_log:
            if "routed_experts" not in message:
                token_ids = cast(Tensor, message["token_ids"])
                # Match the captured routing's dtype/device so the per-key
                # consistency check in the flatten does not trip.
                message["routed_experts"] = torch.full(
                    (token_ids.shape[0], *route_shape),
                    sentinel,
                    dtype=reference.dtype,
                    device=reference.device,
                )


def _pad_tensor(
    tensor: Tensor,
    max_len: int,
    pad_side: str,
    pad_value: int = 0,
) -> Tensor:
    """Pad a tensor to the specified length.

    Args:
        tensor: Tensor to pad
        max_len: Length to pad to
        pad_side: Whether to pad on the 'left' or 'right'
        pad_value: Value to use for padding

    Returns:
        torch.Tensor: Padded tensor
    """
    pad_len = max_len - tensor.size(0)
    if pad_len <= 0:
        return tensor

    padding = torch.full(
        (pad_len, *tensor.shape[1:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat(
        [padding, tensor] if pad_side == "left" else [tensor, padding], dim=0
    )

def _validate_tensor_consistency(tensors: list[Tensor]) -> None:
    """Validate that all tensors have consistent dtypes and devices.

    Args:
        tensors: List of tensors to validate

    Raises:
        RuntimeError: If tensors have different dtypes or devices
    """
    if not tensors:
        return

    first = tensors[0]
    if not all(t is None or t.dtype == first.dtype for t in tensors):
        raise RuntimeError(
            f"expected consistent types but got: {[t.dtype for t in tensors]}"
        )
    if not all(t is None or t.device == first.device for t in tensors):
        raise RuntimeError(
            f"expected tensors on the same device but got: {[t.device for t in tensors]}"
        )

def batched_message_log_to_flat_message(
    message_log_batch: list[LLMMessageLogType],
    pad_value_dict: Optional[dict[str, int]] = None,
    make_sequence_length_divisible_by: int = 1,
) -> tuple[BatchedDataDict[FlatMessagesType], Tensor]:
    """Process and pad a batch of message logs for model input.

    For each message log in the batch:
    1. Converts it to a flat representation using message_log_to_flat_messages
    2. Pads all resulting tensors to the same length for batching
    3. Returns a BatchedDataDict and sequence lengths tensor

    Padding is always applied to the right side of sequences.

    Args:
        message_log_batch: List of LLMMessageLogType (each a conversation with multiple turns)
        pad_value_dict: Dictionary mapping keys to padding values (default is 0)
        make_sequence_length_divisible_by: forces the data to be divisible by this value

    Returns:
        BatchedDataDict[FlatMessagesType]: Dictionary containing padded stacked tensors
        torch.Tensor: Input lengths tensor with shape [batch_size] (pre-padding lengths)

    Raises:
        RuntimeError: If tensors have different dtypes or devices
    """
    if not message_log_batch:
        return BatchedDataDict(), torch.empty(0)

    # Process each message log into a flat representation
    sequenced_lists = [message_log_to_flat_messages(ml) for ml in message_log_batch]
    all_keys = {k for seq in sequenced_lists for k in seq}

    # Find max length and identify tensor keys
    max_len = 0
    tensor_keys = []
    for seq in sequenced_lists:
        for key, value in seq.items():
            if isinstance(value, Tensor):
                tensor_keys.append(key)
                max_len = max(max_len, value.size(0))

    if max_len % make_sequence_length_divisible_by != 0:
        max_len = (
            (max_len // make_sequence_length_divisible_by) + 1
        ) * make_sequence_length_divisible_by

    # Handle non-tensor case
    if not tensor_keys:
        result: BatchedDataDict[FlatMessagesType] = BatchedDataDict(
            {
                k: [seq[k][0] if k in seq else None for seq in sequenced_lists]
                for k in all_keys
            }
        )
        return result, torch.empty(0)

    # Create input_lengths tensor
    input_lengths = []
    for seq in sequenced_lists:
        # Find the maximum length among all tensors in the dictionary, default to 0 if none exist.
        # Use maximum here since there may be keys that aren't populated for all messages yet.
        # For example, logprobs don't get populated for non-generated tokens until post-processing.
        seq_len = max(
            (v.size(0) for v in seq.values() if isinstance(v, Tensor)), default=0
        )
        input_lengths.append(seq_len)
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.int32)

    # Process each key
    result = BatchedDataDict()
    for key in all_keys:
        values = [seq.get(key) for seq in sequenced_lists]

        # A key may be sparsely populated across the batch (e.g. logprobs are
        # absent until post-processing), so inspect the first *non-None* value
        # to decide the key's type rather than assuming values[0] is present.
        first_value = next((v for v in values if v is not None), None)

        # Check for PackedTensor (multimodal) — deferred to avoid hard dependency
        try:
            from dockyard_rl.data.multimodal_utils import PackedTensor  # type: ignore[import]

            if isinstance(first_value, PackedTensor):
                # Guarded by the isinstance check above: these values are
                # PackedTensors at runtime, so cast away the broader static type.
                result[key] = PackedTensor.flattened_concat(
                    cast("list[PackedTensor]", values)
                )
                continue
        except ImportError:
            pass  # multimodal_utils not yet ported; skip PackedTensor handling

        if not isinstance(first_value, Tensor):
            result[key] = values
            continue

        # Filter out None values and validate consistency
        values_typed: list[Tensor | None] = cast(list[Tensor | None], values)
        tensors = cast(list[Tensor], [t for t in values_typed if t is not None])
        _validate_tensor_consistency(tensors)

        # Create zero tensors for None values
        filled_values: list[Tensor] = [
            (
                torch.zeros(0, dtype=tensors[0].dtype, device=tensors[0].device)  # type: ignore[arg-type]
                if v is None
                else v
            )
            for v in values_typed
        ]

        # Pad and stack tensors (always right padding)
        pad_value = pad_value_dict.get(key, 0) if pad_value_dict else 0
        padded = [_pad_tensor(t, max_len, "right", pad_value) for t in filled_values]
        result[key] = torch.stack(padded)

    return result, input_lengths_tensor

def message_log_shape(message_log: LLMMessageLogType) -> list[dict[str, torch.Size]]:
    """Get the shape of the tensors in the message log.

    This utility function examines each message in the message log and reports
    the shape of tensor values or recursively processes list values.

    Args:
        message_log: The message log to analyze

    Returns:
        List of dictionaries containing tensor shapes for each key in messages
    """
    shapes = []
    for message in message_log:
        shape = {}
        for k in message.keys():
            if isinstance(message[k], Tensor):
                shape[k] = message[k].shape  # type: ignore[assignment]
            elif isinstance(message[k], list):
                shape[k] = [message_log_shape(v) for v in message[k]]  # type: ignore[assignment]
        shapes.append(shape)
    return shapes

def get_first_index_that_differs(str1: str, str2: str) -> int:
    """Get the first index that differs between two strings."""
    for i, (c1, c2) in enumerate(zip(str1, str2)):
        if c1 != c2:
            return i
    return min(len(str1), len(str2))

def get_formatted_message_log(
    message_log: LLMMessageLogType,
    tokenizer: TokenizerType,
    task_data_spec: TaskDataSpec,
    add_bos_token: bool = True,
    add_eos_token: bool = True,
    add_generation_prompt: bool = False,
    tools: Optional[list[dict[str, Any]]] = None,
    debug: bool = False,
) -> LLMMessageLogType:
    """Format and tokenize chat messages using the specified template.

    Args:
        message_log: List of message dicts with 'role' and 'content' keys
        tokenizer: Tokenizer for converting text to token IDs
        task_data_spec: Task spec for this dataset.
        add_bos_token: Whether to add bos token to first message if it is not already present. Default: True
        add_eos_token: Whether to add eos token to last message if it is not already present. Default: True
        add_generation_prompt: Whether to include assistant's generation prompt in user messages. Default: False
        tools: Optional list of tool/function definitions to pass to the chat template. Default: None
        debug: Whether to print debug information showing each message turn. Default: False
    Returns:
        The message log with updated 'token_ids' and 'content' fields.
    """
    # Deferred multimodal imports — only needed when tokenizer is a VLM processor
    try:
        from dockyard_rl.data.multimodal_utils import (  # type: ignore[import]
            get_multimodal_default_settings_from_processor,
            get_multimodal_keys_from_processor,
            load_media_from_message,
            get_dim_to_pack_along,
            PackedTensor,
        )
        multimodal_keys = get_multimodal_keys_from_processor(tokenizer)
        multimodal_load_kwargs = get_multimodal_default_settings_from_processor(tokenizer)
    except ImportError:
        multimodal_keys = []
        multimodal_load_kwargs = {}

        def load_media_from_message(message, multimodal_load_kwargs=None):  # type: ignore[misc]
            return {}

    new_message_log: LLMMessageLogType = []
    prev_formatted_message = ""
    message_log_strs: list[dict[str, Any]] = cast(
        list[dict[str, Any]], message_log
    )

    def _format_content_helper(
        content: Union[str, list[dict[str, Any]]],
    ) -> Union[str, list[dict[str, Any]]]:
        """Format the text portion of the first user message with the task prompt.

        The `content` argument could either be a string (user text prompt) or a dict
        (user text prompt + multimodal data).
        """
        prompt_str = cast(str, task_data_spec.prompt)
        if isinstance(content, str):
            return prompt_str.format(content)
        # this is a list of dicts, format only the text ones
        for item in content:
            if item["type"] == "text":
                item["text"] = prompt_str.format(item["text"])
        return content

    # ignore any system prompts
    first_user_msg_id = 0
    for i, msg in enumerate(message_log_strs):
        if msg["role"] == "user":
            first_user_msg_id = i
            break

    if task_data_spec.prompt:
        message_log_strs = (
            message_log_strs[:first_user_msg_id]
            + [
                {
                    "role": "user",
                    "content": _format_content_helper(
                        message_log_strs[first_user_msg_id]["content"]
                    ),
                }
            ]
            + message_log_strs[first_user_msg_id + 1 :]
        )

    for i, message in enumerate(message_log_strs):
        # If enabled, add_generation_prompt is only used on user messages to include
        # the assistant's generation prompt as part of the user message.

        # Only pass tools parameter if tools exist
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt
            and message["role"] in ["user", "tool"],
            "tokenize": False,
            "add_special_tokens": False,
        }
        if tools is not None:
            template_kwargs["tools"] = tools

        formatted_message: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            message_log_strs[: i + 1], **template_kwargs
        )

        ## get the length of the previous message, excluding the eos token (if present)
        prev_message_len_no_eos: int = get_first_index_that_differs(
            prev_formatted_message,
            formatted_message,
        )

        ## pull out the chunk corresponding to the current message
        message_chunk = formatted_message[prev_message_len_no_eos:]

        # Debug: Print each message turn separately
        if debug:
            if i == 0:
                print("\n" + "=" * 80)
                print("DEBUG: Individual message turns from apply_chat_template")
                print("=" * 80)

            print(f"\n[Turn {i + 1}/{len(message_log_strs)}] Role: {message['role']}")
            print("-" * 40)
            print("Extracted message chunk:")
            print(repr(message_chunk))
            print(f"Raw text (len={len(message_chunk)}):")
            print(message_chunk)
            print("-" * 40)

            if i == len(message_log_strs) - 1:
                print("\n" + "=" * 80)
                print("DEBUG: Complete formatted conversation:")
                print("-" * 80)
                print(formatted_message)
                print("=" * 80 + "\n")

        if i == 0:
            if add_bos_token:
                if tokenizer.bos_token is None:
                    warnings.warn(
                        "add_bos_token is True but the tokenizer does not have a BOS token. Skipping BOS token addition."
                    )
                else:
                    bos_token = cast(str, tokenizer.bos_token)
                    if not message_chunk.startswith(bos_token):
                        message_chunk = bos_token + message_chunk

        if i == len(message_log_strs) - 1:
            r"""
            This is an attempt to robustly append the eos token. The origin is Qwen
            chat templates always append <eos>\n and some models like gemma do not
            use the <eos> at all in the chat template. Adding a <eos> if the <eos> is
            already at the end, is likely a user error, and since we know Qwen likes to
            have <eos>\n we'll check for that case.

            This makes the logic slightly more robust to the model family's chat template
            so users don't need to know whether they need to add add_eos or not.
            """
            stripped_message_chunk = message_chunk.rstrip("\n")
            if add_eos_token:
                if tokenizer.eos_token is None:
                    warnings.warn(
                        "add_eos_token is True but the tokenizer does not have an EOS token. Skipping EOS token addition."
                    )
                else:
                    eos_token = cast(str, tokenizer.eos_token)
                    if not stripped_message_chunk.endswith(eos_token):
                        message_chunk += eos_token

        # get images too (extend this for other modalities)
        media_cur_message = load_media_from_message(
            message, multimodal_load_kwargs=multimodal_load_kwargs
        )

        new_message = message.copy()
        # extend this if statement to check for all(len(modality)) == 0 when adding other modalities
        if len(media_cur_message) == 0:
            new_message["token_ids"] = tokenizer(
                text=message_chunk, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
        else:
            # extend the else statement to add other modalities
            # (in this case, tokenizer will be a processor)
            media_kwargs = {}
            if "image" in media_cur_message:
                media_kwargs["images"] = media_cur_message["image"]
            if "audio" in media_cur_message:
                media_kwargs["audio"] = media_cur_message["audio"]
            if "video" in media_cur_message:
                media_kwargs["videos"] = media_cur_message["video"]

            processed_chunk = tokenizer(
                text=[message_chunk],
                return_tensors="pt",
                add_special_tokens=False,
                **media_kwargs,
            )
            new_message["token_ids"] = processed_chunk["input_ids"][0]

            # add all vlm keys to the message
            for key in multimodal_keys:
                if key in processed_chunk:
                    # token_type_ids and mm_token_type_ids are sequence-length tensors
                    # (one label per token), not visual patch tensors. They must be
                    # stored as plain tensors and padded like input_ids rather than
                    # packed as multimodal data.
                    if key in ("token_type_ids", "mm_token_type_ids"):
                        new_message[key] = processed_chunk[key][0]
                    else:
                        try:
                            from dockyard_rl.data.multimodal_utils import PackedTensor, get_dim_to_pack_along  # type: ignore[import]

                            new_message[key] = PackedTensor(
                                processed_chunk[key],
                                dim_to_pack=get_dim_to_pack_along(tokenizer, key),
                            )
                        except ImportError:
                            new_message[key] = processed_chunk[key][0]

        if len(new_message["token_ids"]) == 0:
            # if there is an empty message, the empty `token_ids` tensor ends up being in fp32,
            # which causes `_validate_tensor_consistency` to fail. To fix this, we convert the
            # empty tensor to int64.
            new_message["token_ids"] = new_message["token_ids"].to(torch.int64)  # type: ignore[assignment]

        # format content correctly
        content = message.get("content")
        if content is None or not content:
            # Handle None or missing content (e.g., assistant messages with only tool_calls)
            new_message["content"] = message_chunk
        elif isinstance(content, str):
            new_message["content"] = message_chunk
        else:
            # format the content list of new message the same way as the original message but replace the text with the new message chunk
            new_message["content"] = []
            for item in content:
                if item["type"] == "text":
                    new_message["content"].append(
                        {"type": "text", "text": message_chunk}
                    )
                else:
                    new_message["content"].append(item)

        new_message_log.append(new_message)
        prev_formatted_message = formatted_message

    return new_message_log

def remap_dataset_keys(
    dataset: Dataset,
    mapping_dict: dict[str, str],
) -> Dataset:
    """Remap dataset keys as per mapping.

    Args:
        dataset: The input dataset to remap keys in
        mapping_dict: A dictionary mapping input keys to output keys

    Returns:
        Dataset: A new dataset with remapped keys
    """
    # no need to remap if the keys are already correct
    if all(k == v for k, v in mapping_dict.items()):
        return dataset

    # return the remapped dataset
    return dataset.map(
        lambda x: {v: x[k] for k, v in mapping_dict.items()},
        remove_columns=list(mapping_dict.keys()),
    )


# ── message-log wire decomposition (data-plane companion) ─────────────
# The data plane ships message_log as decomposed per-field arrays (tensor
# views, never per-row pickle) and reconstructs the list-of-dicts view on the
# consumer. Read side (reconstruct / attach) is used by data_plane.column_io;
# write side (decompose) by the rollout producer.

# Fields ridden in bulk and consumed by ``reconstruct_message_log`` to rebuild
# the list-of-dicts view.
MESSAGE_LOG_BULK_FIELDS = ("turn_lengths", "turn_roles", "turn_contents")


def decompose_message_log(
    message_log_batch: list[LLMMessageLogType],
) -> dict[str, Any]:
    """Split a list-of-lists-of-dicts ``message_log`` into per-field arrays.

    Returns a dict with:

    - ``turn_lengths`` — ``torch.LongTensor(B, max_turns)``, zero in unused slots.
    - ``turn_roles`` — ``np.ndarray(object, (B,))`` of ``list[str]``.
    - ``turn_contents`` — ``np.ndarray(object, (B,))`` of ``list[str]``.
    - ``response_token_lengths`` — ``torch.LongTensor(B,)``, assistant-turn
      length per sample (0 if no assistant turn).
    """
    batch_size = len(message_log_batch)
    max_turns = max((len(ml) for ml in message_log_batch), default=0)

    turn_roles = np.empty(batch_size, dtype=object)
    turn_contents = np.empty(batch_size, dtype=object)
    # Build Python lists in the hot loop; one tensor allocation at the end
    # avoids per-turn 0-d tensor writes inside the loop.
    turn_lengths_lol: list[list[int]] = [[0] * max_turns for _ in range(batch_size)]
    response_lengths: list[int] = [0] * batch_size

    for i, ml in enumerate(message_log_batch):
        roles: list[str] = []
        contents: list[str] = []
        lengths_i = turn_lengths_lol[i]
        for t, m in enumerate(ml):
            role = cast(str, m["role"])  # required; surface bad data loudly here
            roles.append(role)
            contents.append(cast(str, m.get("content", "")))
            tok = m.get("token_ids")
            if tok is None:
                continue
            length = int(tok.shape[0]) if isinstance(tok, torch.Tensor) else len(tok)
            lengths_i[t] = length
            if role == "assistant" and response_lengths[i] == 0:
                response_lengths[i] = length
        turn_roles[i] = roles
        turn_contents[i] = contents

    return {
        "turn_lengths": torch.tensor(turn_lengths_lol, dtype=torch.long),
        "turn_roles": turn_roles,
        "turn_contents": turn_contents,
        "response_token_lengths": torch.tensor(response_lengths, dtype=torch.long),
    }


def attach_message_log_view(batch: BatchedDataDict[Any]) -> None:
    """Attach ``batch['message_log']`` in place if decomposed fields are present.

    Rebuilds ``message_log`` as views into the consumer-local ``input_ids``
    / ``generation_logprobs``. Aliasing is harmless because the local
    tensors own their storage and consumers do not re-pickle ``message_log``.
    No-op when the decomposed fields are absent (legacy pickle-shipped path).
    """
    if "input_ids" not in batch or any(k not in batch for k in MESSAGE_LOG_BULK_FIELDS):
        return
    batch["message_log"] = reconstruct_message_log(
        input_ids=batch["input_ids"],
        turn_lengths=batch["turn_lengths"],
        turn_roles=batch["turn_roles"],
        turn_contents=batch["turn_contents"],
        generation_logprobs=batch.get("generation_logprobs"),
    )


def reconstruct_message_log(
    input_ids: Tensor,
    turn_lengths: Tensor,
    turn_roles: "np.ndarray",
    turn_contents: "np.ndarray",
    generation_logprobs: Optional[Tensor] = None,
) -> list[LLMMessageLogType]:
    """Inverse of :func:`decompose_message_log`.

    Per-turn ``token_ids`` and ``generation_logprobs`` are **views** into
    the consumer-local ``input_ids`` / ``generation_logprobs`` tensors.
    The aliasing is harmless because the local tensors own their storage
    (decoded from the wire) and consumers do not re-pickle ``message_log``.
    """
    batch_size = int(input_ids.shape[0])
    # Single host-side materialization — avoids a per-turn .item() sync.
    turn_lengths_list = turn_lengths.tolist()
    out: list[LLMMessageLogType] = []
    for i in range(batch_size):
        roles_i = turn_roles[i]
        contents_i = turn_contents[i]
        lengths_i = turn_lengths_list[i]
        turns: LLMMessageLogType = []
        offset = 0
        for t, role in enumerate(roles_i):
            length = lengths_i[t]
            if length == 0:
                turns.append({"role": role, "content": contents_i[t]})
                continue
            turn: dict[str, Any] = {
                "role": role,
                "content": contents_i[t],
                "token_ids": input_ids[i, offset : offset + length],
            }
            if generation_logprobs is not None and role == "assistant":
                turn["generation_logprobs"] = generation_logprobs[
                    i, offset : offset + length
                ]
            offset += length
            turns.append(turn)
        out.append(turns)
    return out