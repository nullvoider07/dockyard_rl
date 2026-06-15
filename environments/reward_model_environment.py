import os
from typing import Any, Dict, List, NotRequired, Optional, Tuple, TypedDict, cast
import ray
import torch
from dockyard_rl.algorithms.utils import get_tokenizer
from dockyard_rl.data.interfaces import LLMMessageLogType, TaskDataSpec
from dockyard_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_formatted_message_log,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.distributed.virtual_cluster import RayVirtualCluster
from dockyard_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from dockyard_rl.models.generation.interfaces import GenerationDatumSpec
from dockyard_rl.models.generation.vllm.config import VllmConfig

class RewardModelEnvironmentConfig(TypedDict):
    """Configuration for RewardModelEnvironment.

    Attributes:
        enabled:              Whether the reward model environment is enabled.
        model_name:           Name/path of the reward model (e.g., "Skywork/Skywork-Reward-V2-Qwen3-0.6B").
        tokenizer:            Tokenizer config dict (passed to get_tokenizer).
        precision:            Model precision ("bfloat16", "float16", "float32").
        batch_size:           Batch size for processing conversations.
        checkpoint_path:      Optional path to a fine-tuned checkpoint to load on top of model_name.
        logprob_batch_size:   Batch size for log-probability computation.
        resources:            Dict with ``"gpus_per_node"`` and ``"num_nodes"`` keys.
        reward_model_cfg:     Reward model specific config (must include ``"enabled"`` and ``"reward_model_type"``).
        dtensor_cfg:          DTensor config (must have ``"enabled": True``).
        dynamic_batching:     Dynamic batching config (must be ``{"enabled": False}``).
        sequence_packing:     Sequence packing config (must be ``{"enabled": False}``).
        max_grad_norm:        Must be ``None`` — RM env does not train.
        generation:           Optional VllmConfig for generation.
    """

    enabled: bool
    model_name: str
    precision: str
    batch_size: int
    checkpoint_path: str
    logprob_batch_size: int
    resources: Dict[str, Any]
    dtensor_cfg: Optional[Dict[str, Any]]
    dynamic_batching: NotRequired[Dict[str, Any]]
    sequence_packing: NotRequired[Dict[str, Any]]
    max_grad_norm: NotRequired[Optional[float]]
    generation: NotRequired[Optional[VllmConfig]]

@ray.remote  # pragma: no cover
class RewardModelEnvironment(EnvironmentInterface):
    """Environment that uses a reward model to score completed conversations.

    The reward model is loaded once as a Dockyard ``Policy`` (inference-only,
    no optimiser, no reference model) inside a ``RayVirtualCluster``. Each call
    to ``step()`` tokenises the full conversation, passes it through the Bradley-Terry
    classification head, and returns the raw logit score as the reward.

    Constraints enforced at init time:
    - ``reward_model_cfg.reward_model_type`` must be ``"bradley_terry"``.
    - ``dynamic_batching.enabled`` must be ``False``.
    - ``sequence_packing.enabled`` must be ``False``.
    - ``dtensor_cfg.enabled`` must be ``True``.
    - ``dtensor_cfg.cpu_offload`` must be ``False``.
    - ``dtensor_cfg.activation_checkpointing`` must be ``False``.
    - ``max_grad_norm`` must be ``None``.
    """

    def __init__(self, config: Dict[str, Any]):
        print("🚀 REWARD MODEL ENVIRONMENT INITIALIZATION STARTED")
        print("=" * 60)

        self.config = config

        assert self.config["reward_model_cfg"]["enabled"], (
            "Please set reward_model_cfg.enabled = True in the reward model "
            "environment config to enable reward model."
        )
        assert (
            self.config["reward_model_cfg"]["reward_model_type"] == "bradley_terry"
        ), (
            "Reward model environment currently only supports the Bradley-Terry "
            "reward model."
        )
        assert not self.config["dynamic_batching"]["enabled"], (
            "Dynamic batching is currently not supported with reward model environment."
        )
        assert not self.config["sequence_packing"]["enabled"], (
            "Sequence packing is currently not supported with reward model environment."
        )
        assert self.config["dtensor_cfg"]["enabled"], (
            "Reward model environment currently only supports DTensor."
        )
        assert self.config["max_grad_norm"] is None, (
            "max_grad_norm must be None in reward model environment."
        )
        assert not self.config["dtensor_cfg"]["cpu_offload"], (
            "CPU offload is currently not supported with reward model environment."
        )
        assert not self.config["dtensor_cfg"]["activation_checkpointing"], (
            "Activation checkpointing is currently not supported with reward model "
            "environment."
        )

        # Enforce constraints in-place so downstream code is consistent.
        self.config.setdefault("reward_model_cfg", {})
        self.config["reward_model_cfg"]["enabled"] = True
        self.config["reward_model_cfg"]["reward_model_type"] = "bradley_terry"
        self.config.setdefault("dynamic_batching", {})
        self.config.setdefault("sequence_packing", {})
        self.config["dynamic_batching"]["enabled"] = False
        self.config["sequence_packing"]["enabled"] = False
        self.config["max_grad_norm"] = None
        self.config["dtensor_cfg"]["enabled"] = True
        self.config["dtensor_cfg"]["cpu_offload"] = False
        self.config["dtensor_cfg"]["activation_checkpointing"] = False

        self.task_data_spec = TaskDataSpec(task_name="reward_model_env")

        # Remove CUDA_VISIBLE_DEVICES so Ray can control GPU allocation fully.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self.virtual_cluster = RayVirtualCluster(
            name="grpo_reward_model_cluster",
            bundle_ct_per_node_list=[self.config["resources"]["gpus_per_node"]]
            * self.config["resources"]["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=self.config["resources"]["gpus_per_node"],
            max_colocated_worker_groups=1,
        )
        print(
            f"🔧 Virtual cluster created with {self.virtual_cluster.get_placement_groups()}"
        )

        # Deferred import — lm_policy.py must be written before this env is used.
        from dockyard_rl.models.policy.lm_policy import Policy  # type: ignore[import]

        print("🔧 Setting up reward model worker...")
        weights_path = self.config.get("checkpoint_path", None)
        self.tokenizer = get_tokenizer(self.config["tokenizer"])
        print(
            f"✅ Tokenizer initialized with pad_token_id: {self.tokenizer.pad_token_id}"
        )

        self.reward_model_policy: Optional[Any] = None
        self.reward_model_policy = Policy(
            cluster=self.virtual_cluster,
            config=self.config,
            tokenizer=self.tokenizer,
            name_prefix="reward_model_policy",
            init_optimizer=False,
            init_reference_model=False,
            weights_path=weights_path,
        )

        print("✅ REWARD MODEL ENVIRONMENT INITIALIZATION COMPLETE")

    def preprocess_data(
        self, message_logs: List[LLMMessageLogType]
    ) -> BatchedDataDict[GenerationDatumSpec]:
        """Tokenise and pad a list of conversation logs for reward model inference.

        Args:
            message_logs: List of conversation message logs, each a list of
                role/content dicts.

        Returns:
            BatchedDataDict with ``"input_ids"`` and ``"input_lengths"`` tensors.
        """
        tokenized_message_logs = []
        for message_log in message_logs:
            tokenized_log = get_formatted_message_log(
                message_log,
                tokenizer=self.tokenizer,
                task_data_spec=self.task_data_spec,
                add_bos_token=True,
                add_eos_token=True,
                add_generation_prompt=False,
            )
            tokenized_message_logs.append(tokenized_log)

        cat_and_padded, input_lengths = batched_message_log_to_flat_message(
            tokenized_message_logs,
            pad_value_dict=cast(dict[str, int], {"token_ids": self.tokenizer.pad_token_id}),
        )

        reward_data: BatchedDataDict[GenerationDatumSpec] = BatchedDataDict(
            {
                "input_ids": cat_and_padded["token_ids"],
                "input_lengths": input_lengths,
            }
        )
        return reward_data

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata: List[Dict[str, Any]],
    ) -> EnvironmentReturn:
        """Score completed conversations with the reward model.

        Args:
            message_log_batch: List of conversation message logs to score. Each log
                should contain alternating user and assistant messages.
            metadata: Per-sample environment info dicts (currently unused but
                required by the interface).

        Returns:
            EnvironmentReturn with:
                - ``observations``: List of dicts with the scalar reward as content.
                - ``metadata``: List of ``None`` (no step-level metadata).
                - ``next_stop_strings``: List of ``None`` (single-step env).
                - ``rewards``: CPU tensor of reward model scores, shape ``(B,)``.
                - ``terminateds``: CPU bool tensor of all ``True``, shape ``(B,)``.
                - ``answers``: Last assistant message from each conversation.
        """
        reward_data = self.preprocess_data(message_log_batch)
        assert self.reward_model_policy is not None, "reward_model_policy is not initialized"
        rewards: torch.Tensor = self.reward_model_policy.score(reward_data)["scores"]

        observations = [
            {"role": "environment", "content": f"Environment: {float(r):.6f}"}
            for r in rewards
        ]
        terminateds = [True] * len(message_log_batch)
        step_metadata = [None] * len(message_log_batch)
        next_stop_strings = [None] * len(message_log_batch)
        answers: list[str | None] = [str(ml[-1]["content"]) for ml in message_log_batch]

        return EnvironmentReturn(
            observations=observations,
            metadata=step_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards.cpu(),
            terminateds=torch.tensor(terminateds, dtype=torch.bool).cpu(),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Return aggregate reward statistics for the completed batch.

        Args:
            batch: Rollout batch. Expected optional key: ``"rewards"``.

        Returns:
            Tuple of (unmodified batch, metrics dict).
        """
        metrics: dict[str, Any] = {
            "reward_model_env/num_samples": len(batch.get("message_log", [])),
        }

        if "rewards" in batch:
            r: torch.Tensor = batch["rewards"]
            if isinstance(r, torch.Tensor):
                metrics.update(
                    {
                        "reward_model_env/mean_reward": float(r.mean()),
                        "reward_model_env/std_reward": float(r.std()),
                        "reward_model_env/min_reward": float(r.min()),
                        "reward_model_env/max_reward": float(r.max()),
                    }
                )

        return batch, metrics

    def shutdown(self) -> None:
        """Tear down the reward model policy and virtual cluster cleanly."""
        if (
            hasattr(self, "reward_model_policy")
            and self.reward_model_policy is not None
        ):
            try:
                self.reward_model_policy.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down reward model policy: {e}")
            self.reward_model_policy = None
            if self.virtual_cluster is not None:
                try:
                    self.virtual_cluster.shutdown()
                except Exception as e:
                    print(f"Warning: Error shutting down virtual cluster: {e}")
            self.virtual_cluster = None

    def __del__(self):
        self.shutdown()