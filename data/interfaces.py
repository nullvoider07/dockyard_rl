import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired, Optional, Protocol, TypedDict, Union
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# OpenAI-API-like message log, but every message may contain associated tensors
# (i.e. tokenized strings and logprobs) in addition to the original "content" string.
LLMMessageLogType = list[dict[str, Union[str, torch.Tensor]]]

if TYPE_CHECKING:
    try:
        from dockyard_rl.data.multimodal_utils import PackedTensor as _PackedTensor
    except ImportError:
        _PackedTensor = Any  # type: ignore[assignment,misc]
    
    # Static type definition for Pylance
    VLMMessageLogType = list[dict[str, Union[str, torch.Tensor, _PackedTensor]]]
else:
    # Dynamic definition for runtime execution
    try:
        from dockyard_rl.data.multimodal_utils import PackedTensor as _PackedTensorRuntime

        VLMMessageLogType = list[dict[str, Union[str, torch.Tensor, _PackedTensorRuntime]]]
    except ImportError:
        # multimodal_utils not yet ported; VLM paths are text-only for Dockyard
        VLMMessageLogType = list[dict[str, Any]]  # type: ignore[assignment]

# Flattened message log where all tensors and data are concatenated together for a conversation.
# Converts a conversation from list-of-turns format to key-value format with concatenated tensors.
FlatMessagesType = dict[str, Union[list[str], torch.Tensor]]

PathLike = Union[str, "os.PathLike[Any]"]
TokenizerType = PreTrainedTokenizerBase

class DatumSpec(TypedDict):
    message_log: LLMMessageLogType | VLMMessageLogType
    length: int  # total (concatenated) length of the message tensors
    extra_env_info: Optional[dict[str, Any]]
    loss_multiplier: float  # multiplier for the loss for this datum. 0 to mask out (say the sample is invalid)
    idx: int
    task_name: NotRequired[str]
    stop_strings: NotRequired[list[str]]  # Optional stop strings for generation
    __extra__: NotRequired[Any]  # This allows additional fields of any type

class PreferenceDatumSpec(TypedDict):
    message_log_chosen: LLMMessageLogType
    message_log_rejected: LLMMessageLogType
    length_chosen: int
    length_rejected: int
    loss_multiplier: float
    idx: int

class KTODatumSpec(TypedDict):
    """Unpaired preference datum for KTO.

    Each example is a single prompt+completion carrying a binary
    desirable/undesirable label, rather than a chosen/rejected pair.
    """
    message_log: LLMMessageLogType
    length: int
    preference_label: bool  # True = desirable, False = undesirable
    loss_multiplier: float
    idx: int
    task_name: NotRequired[str]

@dataclass
class TaskDataSpec:
    task_name: Optional[str] = None
    # prompt
    prompt_file: Optional[PathLike] = None

    system_prompt_file: Optional[PathLike] = None

    # Structured tool-use protocol block (grpo.structured_tool_use), populated by
    # the data-setup layer from the GRPO config so tool-use processors can
    # advertise the env's tool registry in the prompt. None → fenced-text path.
    structured_tool_use: Optional[Any] = None

    def __post_init__(self) -> None:
        def load_prompt_file(
            prompt_file: Optional[PathLike],
        ) -> Optional[str]:
            """Load prompt from file if it exists, otherwise return as is."""
            if prompt_file is None:
                return None
            if os.path.exists(prompt_file):
                with open(prompt_file, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                raise FileNotFoundError(f"Prompt file {prompt_file} not found")

        # Load prompts from files if they exist
        self.system_prompt = load_prompt_file(self.system_prompt_file)
        self.prompt = load_prompt_file(self.prompt_file)

    def copy_defaults(self, from_spec: "TaskDataSpec") -> None:
        """Apply default values from another Task instance for any None attributes."""
        default_attrs = {
            "system_prompt": from_spec.system_prompt,
            "prompt": from_spec.prompt,
        }

        for attr_name, default_value in default_attrs.items():
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, default_value)

class TaskDataProcessFnCallable(Protocol):
    """A callable that processes a loaded datum dictionary into a DatumSpec."""

    def __call__(
        self,
        datum_dict: dict[str, Any],
        task_data_spec: TaskDataSpec,
        tokenizer: TokenizerType,
        max_seq_length: int | None,
        idx: int,
    ) -> DatumSpec:
        raise NotImplementedError("Task data process not implemented")

class TaskDataPreProcessFnCallable(Protocol):
    """A callable that processes a loaded raw datum dictionary into a dictionary with required format for further processing."""

    def __call__(self, datum_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Task data preprocess not implemented")