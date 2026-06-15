import abc
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar
from torch import Tensor
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

# Separate static type definition from runtime execution path
if TYPE_CHECKING:
    # A clean import statement without try/except ensures Pylance treats it as a strict type symbol
    from dockyard_rl.data.interfaces import LLMMessageLogType
else:
    try:
        from dockyard_rl.data.interfaces import LLMMessageLogType
    except ImportError:
        LLMMessageLogType = list  # type: ignore

# Type variable for environment-specific metadata
MetadataT = TypeVar("MetadataT")

class EnvironmentReturn(NamedTuple, Generic[MetadataT]):
    """Standard batched return type for environment step methods.

    **All elements are batched.**
    observations: New observation from the environment.
                  It's a (batched) 'message' type, which is a dict
                  with keys 'role' and 'content'.
    metadata: Updated metadata from the environment.
    next_stop_strings: The stop strings for the next turn.
                       If your environment is a game or similar,
                       you may want to return a list of stop strings
                       that are valid actions for the next turn or
                       similar. This field lets you control this per turn.
    rewards: the rewards for this turn.
             Shape [B] for single-reward, [B, num_reward_components] for multi-reward (e.g. GDPO).
    terminateds: whether the episode ended this turn.
    answers: the answers for this turn.
    """

    observations: list[dict[str, str]]
    metadata: list[MetadataT]
    next_stop_strings: list[list[str] | None] | list[None]
    rewards: Tensor
    terminateds: Tensor
    answers: list[str | None] | None

class EnvironmentInterface(abc.ABC, Generic[MetadataT]):
    @abc.abstractmethod
    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MetadataT],
    ) -> EnvironmentReturn[MetadataT]:
        """Runs a step in the environment. Allows for asynchrony with remote servers, but it's not required (this function is a ray remote).

        message_log_batch: batch of OpenAI-API-like message logs that represent interactions with the LLM.
                  Each element is a list[dict[str, Union[str, torch.Tensor]]].
                  For example, if this were a Coding Environment, the message log would be:
                  [
                    {"role": "user", "content": "implement this function"},
                    {"role": "assistant", "content": "```python\ndef foo(): ...```"},
                  ]
                  and after a tool turn:
                  [
                    {"role": "user", "content": "implement this function"},
                    {"role": "assistant", "content": "```python\ndef foo(): ...```"},
                    {"role": "user", "content": "Tests passed: 3/5"},
                    {"role": "assistant", "content": "let me fix that..."},
                  ]
        metadata:     batch of whatever the environment needs to keep track of. I.e.
                      sandbox URLs, task specs, test suites, or agent states. Can be None if episode terminated.

        Returns:
        - EnvironmentReturn NamedTuple containing observations, metadata, next_stop_strings, rewards, and terminateds flags.
        """

    @abc.abstractmethod
    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Post processing function after all rollouts are done for the batch and returns metrics."""