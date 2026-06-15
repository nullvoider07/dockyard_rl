from typing import Any, Union, cast
import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase
from dockyard_rl.data.interfaces import DatumSpec, KTODatumSpec, PreferenceDatumSpec
from dockyard_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

TokenizerType = Union[PreTrainedTokenizerBase, AutoProcessor]

def rl_collate_fn(data_batch: list[DatumSpec]) -> BatchedDataDict[Any]:
    """Collate function for RL training."""
    message_log = [datum_spec["message_log"] for datum_spec in data_batch]
    length = torch.tensor([datum_spec["length"] for datum_spec in data_batch])
    loss_multiplier = torch.tensor(
        [datum_spec["loss_multiplier"] for datum_spec in data_batch]
    )
    extra_env_info = [datum_spec["extra_env_info"] for datum_spec in data_batch]

    task_names = []
    for datum_spec in data_batch:
        task_names.append(datum_spec.get("task_name", None))

    idx = [datum_spec["idx"] for datum_spec in data_batch]
    batch_max_length = torch.ones_like(length) * length.max()

    # Extract stop_strings if present
    stop_strings = [datum.get("stop_strings", None) for datum in data_batch]

    # check if any of the data batch has vllm content and images
    extra_args = {}
    if any(
        [datum_spec.get("vllm_content", None) is not None for datum_spec in data_batch]
    ):
        vllm_content = [
            datum_spec.get("vllm_content", None) for datum_spec in data_batch
        ]
        vllm_images = [datum_spec.get("vllm_images", []) for datum_spec in data_batch]
        vllm_videos = [datum_spec.get("vllm_videos", []) for datum_spec in data_batch]
        vllm_audios = [datum_spec.get("vllm_audios", []) for datum_spec in data_batch]
        extra_args["vllm_content"] = vllm_content
        extra_args["vllm_images"] = vllm_images
        extra_args["vllm_videos"] = vllm_videos
        extra_args["vllm_audios"] = vllm_audios

    output: BatchedDataDict[Any] = BatchedDataDict(
        {
            "message_log": message_log,
            "length": length,
            "loss_multiplier": loss_multiplier,
            "extra_env_info": extra_env_info,
            "task_name": task_names,
            "idx": idx,
            "batch_max_length": batch_max_length,
            "stop_strings": stop_strings,
            **extra_args,
        }
    )
    return output

def eval_collate_fn(data_batch: list[DatumSpec]) -> BatchedDataDict[Any]:
    """Collate function for evaluation.

    Takes a list of data samples and combines them into a single batched dictionary
    for model evaluation.

    Args:
        data_batch: List of data samples with message_log, extra_env_info, and idx fields.

    Returns:
        BatchedDataDict with message_log, extra_env_info, and idx fields.
    """
    message_log = [datum_spec["message_log"] for datum_spec in data_batch]
    extra_env_info = [datum_spec["extra_env_info"] for datum_spec in data_batch]
    idx = [datum_spec["idx"] for datum_spec in data_batch]

    # Check if any of the data batch has vllm content (multimodal data)
    extra_args = {}
    if any(
        datum_spec.get("vllm_content", None) is not None for datum_spec in data_batch
    ):
        extra_args["vllm_content"] = [
            datum_spec.get("vllm_content", None) for datum_spec in data_batch
        ]
        extra_args["vllm_images"] = [
            datum_spec.get("vllm_images", []) for datum_spec in data_batch
        ]
        extra_args["vllm_audios"] = [
            datum_spec.get("vllm_audios", []) for datum_spec in data_batch
        ]

    output: BatchedDataDict[Any] = BatchedDataDict(
        {
            "message_log": message_log,
            "extra_env_info": extra_env_info,
            "idx": idx,
            **extra_args,
        }
    )
    return output

def preference_collate_fn(
    data_batch: list[PreferenceDatumSpec],
    tokenizer: TokenizerType,
    make_sequence_length_divisible_by: int,
    add_loss_mask: bool,
) -> BatchedDataDict[Any]:
    """Collate function for preference data training.

    This function separates the chosen and rejected responses to create
    two examples per prompt. The chosen and rejected examples are interleaved
    along the batch dimension, resulting in a batch size of 2 * len(data_batch).

    Args:
        data_batch: List of data samples with message_log_chosen, message_log_rejected,
            length_chosen, length_rejected, loss_multiplier, idx, and task_name fields.
        tokenizer: Tokenizer for text processing
        make_sequence_length_divisible_by: Make the sequence length divisible by this value
        add_loss_mask: Whether to add a token_mask to the returned data
    Returns:
        BatchedDataDict with input_ids, input_lengths, token_mask (optional), and
        sample_mask fields.
    """
    message_log = []
    length = []
    loss_multiplier = []
    idx = []
    task_names = []
    for datum_spec in data_batch:
        ## interleave chosen and rejected examples
        message_log.append(datum_spec["message_log_chosen"])
        message_log.append(datum_spec["message_log_rejected"])
        length.append(datum_spec["length_chosen"])
        length.append(datum_spec["length_rejected"])
        loss_multiplier.extend([datum_spec["loss_multiplier"]] * 2)
        idx.extend([datum_spec["idx"]] * 2)
        task_names.extend([datum_spec.get("task_name", None)] * 2)
    length_batch: torch.Tensor = torch.tensor(length)
    loss_multiplier_batch: torch.Tensor = torch.tensor(loss_multiplier)

    batch_max_length = torch.ones_like(length_batch) * length_batch.max()

    batch: BatchedDataDict[Any] = BatchedDataDict(
        {
            "message_log": message_log,
            "length": length_batch,
            "loss_multiplier": loss_multiplier_batch,
            "task_name": task_names,
            "idx": idx,
            "batch_max_length": batch_max_length,
        }
    )

    if add_loss_mask:
        add_loss_mask_to_message_log(
            batch["message_log"],
            only_unmask_final=True,
        )

    cat_and_padded, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )

    data: BatchedDataDict[Any] = BatchedDataDict(
        {
            "input_ids": cat_and_padded["token_ids"],
            "input_lengths": input_lengths,
            "sample_mask": batch["loss_multiplier"],
        }
    )
    if add_loss_mask:
        data["token_mask"] = cat_and_padded["token_loss_mask"]

    return data

def kto_collate_fn(
    data_batch: list[KTODatumSpec],
    tokenizer: TokenizerType,
    make_sequence_length_divisible_by: int,
    add_loss_mask: bool,
) -> BatchedDataDict[Any]:
    """Collate function for KTO (unpaired) preference data.

    Unlike ``preference_collate_fn``, this does not interleave chosen/rejected:
    each datum is a single prompt+completion with a binary desirable/undesirable
    label, so the batch size equals ``len(data_batch)``. The per-example label is
    emitted as a float ``preference_label`` (1.0 desirable, 0.0 undesirable) that
    ``KTOLossFn`` uses to select the desirable/undesirable branch.

    Args:
        data_batch: List of KTODatumSpec with message_log, length, preference_label,
            loss_multiplier, idx, and task_name fields.
        tokenizer: Tokenizer for text processing.
        make_sequence_length_divisible_by: Make the sequence length divisible by this value.
        add_loss_mask: Whether to add a token_mask to the returned data.

    Returns:
        BatchedDataDict with input_ids, input_lengths, sample_mask, preference_label,
        and (optionally) token_mask fields.
    """
    message_log = []
    length = []
    loss_multiplier = []
    idx = []
    task_names = []
    preference_label = []
    for datum_spec in data_batch:
        message_log.append(datum_spec["message_log"])
        length.append(datum_spec["length"])
        loss_multiplier.append(datum_spec["loss_multiplier"])
        idx.append(datum_spec["idx"])
        task_names.append(datum_spec.get("task_name", None))
        preference_label.append(1.0 if datum_spec["preference_label"] else 0.0)
    length_batch: torch.Tensor = torch.tensor(length)
    loss_multiplier_batch: torch.Tensor = torch.tensor(loss_multiplier)
    preference_label_batch: torch.Tensor = torch.tensor(preference_label)

    batch_max_length = torch.ones_like(length_batch) * length_batch.max()

    batch: BatchedDataDict[Any] = BatchedDataDict(
        {
            "message_log": message_log,
            "length": length_batch,
            "loss_multiplier": loss_multiplier_batch,
            "task_name": task_names,
            "idx": idx,
            "batch_max_length": batch_max_length,
        }
    )

    if add_loss_mask:
        add_loss_mask_to_message_log(
            batch["message_log"],
            only_unmask_final=True,
        )

    cat_and_padded, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )

    data: BatchedDataDict[Any] = BatchedDataDict(
        {
            "input_ids": cat_and_padded["token_ids"],
            "input_lengths": input_lengths,
            "sample_mask": batch["loss_multiplier"],
            "preference_label": preference_label_batch,
        }
    )
    if add_loss_mask:
        data["token_mask"] = cat_and_padded["token_loss_mask"]

    return data