"""Non-colocated teacher worker group for MOPD on-policy distillation.

Each TeacherWorkerGroup wraps a frozen, inference-only DTensor Policy for a
single teacher checkpoint: no optimizer, no reference model, weights loaded once
at startup, exposing only get_logprobs(). This is the DTensor analogue of the
training policy used for teacher scoring in async on-policy distillation.
"""

from __future__ import annotations

import threading
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizerBase

from dockyard_rl.algorithms.opd import TeacherResourceConfig
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.distributed.virtual_cluster import RayVirtualCluster
from dockyard_rl.models.policy.teacher_scorer import select_rows


@dataclass
class TeacherConfig:
    """Resolved config for a single non-colocated DTensor teacher."""

    alias: str
    model_name: str  # checkpoint path
    tensor_parallel_size: int
    context_parallel_size: int
    num_nodes: int
    gpus_per_node: int
    precision: str
    micro_batch_size: int
    dtensor_cfg_overrides: dict[str, Any]


def create_teacher_configs_from_opd_config(
    opd_cfg: dict[str, Any],
) -> list[TeacherConfig]:
    """Build per-teacher configs from on_policy_distillation config.

    Handles deduplication (multiple aliases sharing one checkpoint produce one
    TeacherConfig) and per-teacher overrides on top of defaults.
    """
    teacher_model_by_agent_name: dict[str, str] = dict(
        opd_cfg.get("teacher_model_by_agent_name", {})
    )
    non_coloc_cfg = dict(opd_cfg.get("non_colocated_teachers", {}))
    default_cfg = dict(non_coloc_cfg.get("default_teacher_cfg", {}))
    overrides = dict(non_coloc_cfg.get("teacher_overrides", {}))
    deduplicate = bool(opd_cfg.get("deduplicate_shared_teacher_checkpoints", True))

    configs: list[TeacherConfig] = []
    seen_models: set[str] = set()

    for alias, model_name in teacher_model_by_agent_name.items():
        if deduplicate and model_name in seen_models:
            continue
        seen_models.add(model_name)

        # defaults <- per-alias override, then validated/typed by the schema.
        merged = {**default_cfg, **dict(overrides.get(alias, {}))}
        res = TeacherResourceConfig(**merged)

        # Unknown top-level keys (extra="allow") fold into dtensor_cfg_overrides;
        # explicit dtensor_cfg_overrides take precedence.
        all_overrides = {**(res.model_extra or {}), **res.dtensor_cfg_overrides}

        configs.append(
            TeacherConfig(
                alias=alias,
                model_name=model_name,
                tensor_parallel_size=res.tensor_parallel_size,
                context_parallel_size=res.context_parallel_size,
                num_nodes=res.num_nodes,
                gpus_per_node=res.gpus_per_node,
                precision=res.precision,
                micro_batch_size=res.micro_batch_size,
                dtensor_cfg_overrides=all_overrides,
            )
        )

    return configs


def disable_student_only_features(cfg: dict[str, Any], *, alias: str) -> None:
    """Clear student-only features from a teacher's copy of the policy config.

    A teacher config is a deep copy of the student's, so every student-side
    feature arrives switched on. A frozen teacher trains nothing, refits
    nothing and adds no parameters, so each of these is at best inert on it --
    and ``router_replay`` is not inert: the teacher's worker still runs
    ``validate_router_replay_config``, which refuses a model with no MoEBlock,
    so a dense teacher under an MoE student fails at startup naming a key the
    operator set for the student. Replay has nothing to bind there in any case:
    it re-applies a rollout's recorded routing to a second forward over the
    same tokens, and the teacher scores the student's trajectory from a batch
    that carries no ``routed_experts`` column.

    Mutates ``cfg`` in place. Keys absent from the config are left absent
    rather than introduced.
    """
    if "peft" in cfg:
        cfg["peft"]["enabled"] = False
    if "draft" in cfg:
        cfg["draft"]["enabled"] = False
    if isinstance(cfg.get("router_replay"), dict):
        cfg["router_replay"]["enabled"] = False
    if cfg.get("quant_cfg") is not None:
        warnings.warn(
            f"Teacher '{alias}': quantization is not supported for "
            "teachers; running the teacher unquantized (ignoring quant_cfg)."
        )
        cfg["quant_cfg"] = None


class TeacherWorkerGroup:
    """Inference-only DTensor Policy for a single teacher model.

    Unlike the training policy, this group never initializes an optimizer or a
    reference model, loads the checkpoint once at startup, and only exposes
    get_logprobs(). Built on dockyard's Policy (DTensor backend), so it reuses
    the policy worker-group, sharding, and logprob path directly.
    """

    def __init__(
        self,
        teacher_cfg: TeacherConfig,
        cluster: RayVirtualCluster,
        policy_config: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.alias = teacher_cfg.alias
        self.model_name = teacher_cfg.model_name
        self.teacher_cfg = teacher_cfg

        # Build an inference-only DTensor policy config for this teacher.
        cfg = deepcopy(policy_config)
        cfg["model_name"] = self.model_name

        # Teachers run the DTensor backend; force it on and clear training-only
        # / parameter-adding features so the student's config can't leak onto the
        # frozen teacher.
        dtensor_cfg = cfg.setdefault("dtensor_cfg", {})
        dtensor_cfg["enabled"] = True
        dtensor_cfg["tensor_parallel_size"] = teacher_cfg.tensor_parallel_size
        dtensor_cfg["context_parallel_size"] = teacher_cfg.context_parallel_size
        for key, value in teacher_cfg.dtensor_cfg_overrides.items():
            dtensor_cfg[key] = value

        if "precision" in cfg:
            cfg["precision"] = teacher_cfg.precision
        disable_student_only_features(cfg, alias=self.alias)

        tp = teacher_cfg.tensor_parallel_size
        cp = teacher_cfg.context_parallel_size

        # Validate parallelism fits the cluster (mirrors lm_policy.py).
        world_size = cluster.world_size()
        model_parallel_size = tp * cp
        if world_size < model_parallel_size:
            raise ValueError(
                f"Teacher '{self.alias}': world_size ({world_size}) < "
                f"TP({tp}) * CP({cp}) = {model_parallel_size}"
            )
        if world_size % model_parallel_size != 0:
            raise ValueError(
                f"Teacher '{self.alias}': world_size ({world_size}) not divisible "
                f"by TP({tp}) * CP({cp}) = {model_parallel_size}"
            )

        # Deferred import: Policy pulls heavy training deps not needed for config.
        from dockyard_rl.models.policy.lm_policy import Policy

        self.policy = Policy(
            cluster=cluster,
            config=cfg,
            tokenizer=tokenizer,
            weights_path=cfg.get("weights_path", None),
            optimizer_path=None,
            init_optimizer=False,
            init_reference_model=False,
        )
        self.cfg = cfg
        self._micro_batch_size = teacher_cfg.micro_batch_size

        # The worker group stays exposed for the readiness check and teardown
        # in create_teacher_worker_groups, which builds local groups and so
        # already knows they are local. Sharding does not: the batch-axis
        # padding it exists to drive now happens inside score().
        self.worker_group = self.policy.worker_group
        self._dp_size = self.policy.sharding_annotations.get_axis_size(
            "data_parallel"
        )
        self._packs_internally = self.policy.use_sequence_packing
        # SP-forward divisor. Reported through sequence_pad_multiple; the caller
        # pre-pads the shared flattened batch with it, because that axis is
        # shared by every teacher scoring the same batch.
        self._sequence_pad_multiple = cp * 2 * tp if cp > 1 else tp
        # Forwards on one worker group are serialized here rather than by the
        # caller: the group's workers run NCCL collectives per forward, so two
        # overlapping calls would interleave them. It is a property of this
        # implementation, not of scoring, so it belongs on this side of the
        # TeacherScorer boundary.
        self._score_lock = threading.Lock()

    @property
    def requires_prepadded_input(self) -> bool:
        """Packed teachers re-pad their own forward; unpacked ones need the pad."""
        return not self._packs_internally

    @property
    def sequence_pad_multiple(self) -> int:
        return self._sequence_pad_multiple

    def score(self, data: BatchedDataDict[Any]) -> torch.Tensor:
        """Per-token logprobs for ``data``'s rows, one row out per row in.

        The batch axis is padded up to a multiple of this group's data-parallel
        size and trimmed back afterwards. Every column is padded by repeating
        the same final row -- padding the token ids alone, or padding a
        multimodal column with anything else, puts that column out of step with
        ``input_ids`` and the teacher scores an input the student never saw.
        """
        input_ids = data["input_ids"]
        rows = int(input_ids.shape[0])
        remainder = rows % self._dp_size
        padded = data
        if remainder != 0:
            pad_count = self._dp_size - remainder
            row_indices = list(range(rows)) + [rows - 1] * pad_count
            padded = BatchedDataDict(
                {
                    key: select_rows(key, value, row_indices)
                    for key, value in data.items()
                }
            )

        with self._score_lock:
            result = self.get_logprobs(padded)
        return result["reference_logprobs"][:rows]

    def get_logprobs(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[Any]:
        """Run a teacher forward pass and return per-token logprobs.

        The returned BatchedDataDict carries a ``reference_logprobs`` key of shape
        ``[B, S]`` (the teacher is a frozen reference; micro-batching is driven by
        the teacher's config). ``micro_batch_size`` is accepted for interface
        parity with the training policy.
        """
        result = self.policy.get_logprobs(data)
        return BatchedDataDict({"reference_logprobs": result["logprobs"]})

    def shutdown(self) -> bool:
        """Shut down the teacher policy workers."""
        try:
            return self.policy.shutdown()
        except Exception as e:
            print(f"Error during teacher worker group shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Safety net for cleanup."""
        if hasattr(self, "policy"):
            self.shutdown()
