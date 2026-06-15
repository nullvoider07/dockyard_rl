from typing import Any, Callable, Optional, Union, cast

import torch
from datasets import Dataset
from transformers import AutoProcessor, PreTrainedTokenizerBase
from dockyard_rl.data.datasets.utils import assert_no_double_bos
from dockyard_rl.data.interfaces import (
    DatumSpec,
    TaskDataPreProcessFnCallable,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)

TokenizerType = Union[PreTrainedTokenizerBase, AutoProcessor]


class AllTaskProcessedDataset:
    """Dataset for processing single or multi-task data with task-specific tokenization and processing.

    Args:
        dataset: Input dataset containing raw data
        tokenizer: Tokenizer or processor for text processing
        default_task_data_spec: Default task processing specifications.
            In the case of single-task, this is the spec used for processing all entries.
            In the case of multi-task, any values not specified in the task-specific specs
            will be taken from the default spec.
        task_data_processors: Either a single TaskDataProcessFnCallable for single-task,
            or a dict mapping task names to (TaskDataSpec, TaskDataProcessFnCallable) for multi-task
        max_seq_length: Maximum sequence length for tokenized outputs
    """

    def __init__(
        self,
        dataset: Dataset | Any,
        tokenizer: TokenizerType,
        default_task_data_spec: TaskDataSpec,
        task_data_processors: (
            dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]]
            | TaskDataProcessFnCallable
        ),
        task_data_preprocessors: Optional[
            Union[dict[str, TaskDataPreProcessFnCallable], TaskDataPreProcessFnCallable]
        ] = None,
        max_seq_length: Optional[int] = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.default_task_data_spec = default_task_data_spec
        self.task_data_processors = task_data_processors
        self.task_data_preprocessors = task_data_preprocessors
        self.max_seq_length = max_seq_length
        self._bos_checked = False

        if (
            isinstance(task_data_processors, dict)
            and default_task_data_spec is not None
        ):
            # apply defaults to all task data specs
            for _, (task_data_spec, _) in task_data_processors.items():
                task_data_spec.copy_defaults(self.default_task_data_spec)

    def __len__(self) -> int:
        return len(self.dataset)

    def encode_single(
        self, text: Union[str, list[str]]
    ) -> tuple[list[int] | torch.Tensor, int]:
        """Takes either a single string or a list of strings that represent multiple turns
        for the same conversation.

        Returns a single (concatenated) list of tokenized ids and the length of the tokenized ids.
        """
        _tokenizer = cast(Any, self.tokenizer)
        if isinstance(text, str):
            text_ids = _tokenizer.text_to_ids(text)
            return text_ids, len(text_ids)
        elif isinstance(text, list):
            text_ids = [_tokenizer.text_to_ids(t) for t in text]
            return torch.cat(text_ids), sum(len(t) for t in text_ids)
        else:
            raise ValueError(
                f"text must be a string or a list of strings, got {type(text)}"
            )

    def __getitem__(self, idx: int) -> DatumSpec:
        """Return a single prompt."""
        entry = self.dataset[idx]

        # preprocessing
        task_data_preprocessor = None
        if self.task_data_preprocessors:
            if isinstance(self.task_data_preprocessors, dict):
                task_name = entry["task_name"]
                if task_name in self.task_data_preprocessors:
                    task_data_preprocessor = self.task_data_preprocessors[task_name]
            else:
                task_data_preprocessor = self.task_data_preprocessors

        if task_data_preprocessor is not None:
            entry = cast(Callable, task_data_preprocessor)(entry)

        # processing
        if isinstance(self.task_data_processors, dict):
            task_name = entry["task_name"]

            assert task_name in self.task_data_processors, (
                f"task processor not provided for {task_name}. "
                f"Provided processors: {self.task_data_processors.keys()}"
            )
            task_data_spec, task_data_processor = self.task_data_processors[task_name]
        else:
            task_data_spec = self.default_task_data_spec
            task_data_processor = self.task_data_processors

        datum_spec = cast(Callable, task_data_processor)(
            entry, task_data_spec, cast(Any, self.tokenizer), self.max_seq_length, idx
        )

        # Check the first processed item for BOS token assertion
        if (
            not self._bos_checked
            and "message_log" in datum_spec
            and datum_spec["message_log"]
        ):
            first_message = datum_spec["message_log"][0]
            if "token_ids" in first_message:
                token_ids = first_message["token_ids"]
                assert isinstance(token_ids, torch.Tensor), (
                    f"token_ids must be a torch.Tensor, got {type(token_ids)}"
                )
                assert_no_double_bos(token_ids, self.tokenizer)
            self._bos_checked = True

        return datum_spec