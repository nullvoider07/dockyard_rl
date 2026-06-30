"""Async trajectory collector for off-policy GRPO.

A Ray actor that runs continuous background rollouts on the inference fleet and
feeds completed per-prompt trajectory groups into the shared ReplayBuffer, tagged
with the generation weight_version and the target_weight_version (the training
step the group is intended to be consumed at).

Concurrency model
-----------------
The collector is an *async* Ray actor. `start_collection` is a long-running
coroutine; the blocking rollout primitive (`run_async_multi_turn_rollout`, which
calls `asyncio.run` internally and therefore cannot run on an already-running
event loop) is offloaded to a worker thread via `run_in_executor`. This yields
the actor's event loop between generations so the control methods
(`set_weight_version`, `pause`/`resume`, `prepare_for_refit`/`resume_after_refit`,
`get_dataloader_state`) interleave with collection rather than queueing behind it.

Refit coordination
-------------------
`prepare_for_refit` flips a flag so no new generation starts, then waits on the
generation lock until any in-flight rollout finishes, then returns — guaranteeing
the trainer can safely refit weights without a concurrent `generate` on the shared
generation engine. `resume_after_refit` clears the flag; the trainer calls
`set_weight_version` in between so subsequent trajectories carry the new version.

Versioning
----------
Each generated group is tagged target_weight_version = the training step it should
train on; `num_prompts_per_step` groups are produced per target before advancing to
the next. Generation never runs more than `max_trajectory_age_steps` ahead of the
current weight version (a group generated at version v consumed at step T has age
T - v, which the buffer requires to be <= max_age), and backs off when the buffer
reports "full".
"""

import asyncio
import uuid
from typing import Any, Optional

import ray


def _flatten_group_for_dataplane(
    group_batch: Any,
    tokenizer: Any,
    make_sequence_length_divisible_by: int,
) -> tuple[Any, dict[str, Any]]:
    """Flatten one per-prompt rollout group into (bulk_batch, driver_carry).

    Per-group counterpart of the sync rollout actor's flatten step. ``bulk_batch``
    holds the tensor columns written to the data plane (DP_TRAIN_FIELDS +
    multimodal + decomposed message_log); ``driver_carry`` holds only the
    lightweight per-row values the trainer needs for reward/advantage compute
    without a plane fetch. Deferred imports avoid a module-load cycle with
    ``algorithms.grpo`` / ``experience.rollouts``.
    """
    import numpy as np
    import torch

    from dockyard_rl.algorithms.grpo import (
        add_grpo_token_loss_masks_and_generation_logprobs,
        extract_initial_prompt_messages,
    )
    from dockyard_rl.algorithms.utils import get_gdpo_reward_component_keys
    from dockyard_rl.data.llm_message_utils import (
        MESSAGE_LOG_BULK_FIELDS,
        batched_message_log_to_flat_message,
        decompose_message_log,
    )
    from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

    fb = group_batch
    pad: dict[str, Any] = {"pad_value_dict": {"token_ids": tokenizer.pad_token_id}}

    # Original prompt ids (advantage estimator input) — GRPO masks only
    # generated assistant turns, even when the dataset prompt carries
    # assistant history.
    prompt_message_logs = extract_initial_prompt_messages(fb["message_log"], fb["length"])
    prompt_flat, _ = batched_message_log_to_flat_message(prompt_message_logs, **pad)

    add_grpo_token_loss_masks_and_generation_logprobs(fb["message_log"])
    flat, input_lengths = batched_message_log_to_flat_message(
        fb["message_log"],
        **pad,
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )

    bulk = BatchedDataDict[Any](
        {
            "input_ids": flat["token_ids"],
            "input_lengths": input_lengths,
            "generation_logprobs": flat["generation_logprobs"],
            "token_mask": flat["token_loss_mask"],
            "sample_mask": fb["loss_multiplier"],
        }
    )
    for k, v in flat.get_multimodal_dict(as_tensors=False).items():
        if isinstance(v, torch.Tensor):
            bulk[k] = v
    if "content" in flat:
        bulk["content"] = np.asarray(flat["content"], dtype=object)

    # Decompose message_log into per-field arrays instead of pickling the
    # list-of-dicts-with-tensors per row; the consumer rebuilds it on read.
    decomposed = decompose_message_log(fb["message_log"])
    for k in MESSAGE_LOG_BULK_FIELDS:
        bulk[k] = decomposed[k]

    truncated = fb["truncated"]
    if not isinstance(truncated, torch.Tensor):
        truncated = torch.tensor(truncated, dtype=torch.bool)
    length = fb.get("length", input_lengths)
    if not isinstance(length, torch.Tensor):
        length = torch.tensor(length)

    driver_carry: dict[str, Any] = {
        "total_reward": fb["total_reward"],
        "loss_multiplier": fb["loss_multiplier"],
        "truncated": truncated,
        "length": length,
        "input_lengths": input_lengths,
        "prompt_ids_for_adv": prompt_flat["token_ids"],
        "response_token_lengths": decomposed["response_token_lengths"],
    }
    # GDPO multi-reward components: the advantage estimator reads these from
    # the carry-derived adv_inputs, mirroring the sync path.
    for k in get_gdpo_reward_component_keys(fb):
        driver_carry[k] = fb[k]
    # Invalid-action verdict counts (#2656): per-sample scalars consumed by
    # apply_invalid_action_penalty on the driver. Present only when the
    # penalty is enabled (the rollout collects them).
    for k in ("invalid_action_count", "malformed_thinking_count"):
        if k in fb:
            driver_carry[k] = fb[k]
    return bulk, driver_carry


class _StopSentinel:
    """Marker returned by the threaded dataloader step when an epoch ends."""


_EPOCH_END = _StopSentinel()


class AsyncTrajectoryCollectorImpl:
    """Continuous background rollout producer for async GRPO."""

    def __init__(
        self,
        policy_generation: Any,
        tokenizer: Any,
        task_to_env: dict[str, Any],
        master_config: Any,
        replay_buffer: Any,
        start_step: int,
        teacher_worker_groups: Optional[dict[str, Any]] = None,
        alias_to_group_alias: Optional[dict[str, str]] = None,
        teacher_seq_pad_multiple: int = 1,
    ) -> None:
        self._policy_generation = policy_generation
        self._tokenizer = tokenizer
        self._task_to_env = task_to_env
        self._master_config = master_config
        self._replay_buffer = replay_buffer

        # On-policy distillation teachers (OPD). When teachers are present the
        # collector scores each generated trajectory through the routed teacher
        # and stamps `teacher_reference_logprobs` onto the batch so the trainer's
        # OPDAdvantageEstimator can form `Â = sg[teacher - student]`.
        import threading
        from collections import defaultdict

        from dockyard_rl.algorithms import opd as _opd_module

        self._teacher_worker_groups: dict[str, Any] = teacher_worker_groups or {}
        self._alias_to_group_alias: dict[str, str] = alias_to_group_alias or {}
        self._teacher_seq_pad_multiple = int(teacher_seq_pad_multiple)
        self._has_distillation_teachers = bool(self._teacher_worker_groups)
        self._on_policy_distillation_cfg = _opd_module._opd_cfg(master_config)
        # One lock per teacher group: serialize forwards on a given teacher to
        # avoid NCCL collective desync when groups are fanned out concurrently.
        self._teacher_locks: dict[str, Any] = defaultdict(threading.Lock)

        grpo_cfg = master_config.grpo
        self._num_generations_per_prompt = int(grpo_cfg["num_generations_per_prompt"])
        self._num_prompts_per_step = int(grpo_cfg["num_prompts_per_step"])
        self._max_rollout_turns = int(grpo_cfg.get("max_rollout_turns", 999999))
        self._max_trajectory_age_steps = int(
            grpo_cfg.get("max_trajectory_age_steps", 1)
        )
        self._max_seq_len = int(master_config.policy["max_total_sequence_length"])

        # Versioning state.
        self._weight_version: int = start_step
        self._next_target: int = start_step
        self._groups_for_target: int = 0

        # Control state (manipulated only on the actor event loop).
        self._paused: bool = False
        self._refit_requested: bool = False
        self._stopped: bool = False
        self._gen_lock = asyncio.Lock()

        self._dataloader: Any = None

        # Data-plane write path (opt-in). When master_config.data_plane is
        # enabled the collector writes each group's bulk tensors to the plane
        # and enqueues only a KVBatchMeta + lightweight driver_carry, so
        # neither the buffer nor the trainer holds bulk bytes. Disabled (None)
        # keeps the legacy in-buffer batch path byte-identical.
        dp_cfg = getattr(master_config, "data_plane", None)
        self._dp_enabled: bool = bool(dp_cfg) and bool(dp_cfg.get("enabled", False))
        self._dp_client: Any = None
        # Must match the partition id TQPolicy registers (default "train").
        self._dp_partition_id: str = "train"
        self._dp_pad_multiple: int = int(
            master_config.policy.get("make_sequence_length_divisible_by") or 1
        )
        if self._dp_enabled:
            from dockyard_rl.data_plane.factory import build_data_plane_client

            # bootstrap=False — the trainer's TQPolicy created the controller.
            self._dp_client = build_data_plane_client(dp_cfg, bootstrap=False)

    # Control surface
    def set_weight_version(self, weight_version: int) -> None:
        """Record the weight version that subsequent generations were produced with."""
        self._weight_version = int(weight_version)

    def pause(self) -> None:
        """Pause collection (e.g. during validation)."""
        self._paused = True

    def resume(self) -> None:
        """Resume collection after a pause."""
        self._paused = False

    async def prepare_for_refit(self) -> bool:
        """Quiesce generation so the trainer can safely refit weights.

        Returns once no rollout is in flight; the loop will not start a new one
        until `resume_after_refit` is called.
        """
        self._refit_requested = True
        # Block until any in-flight generation completes.
        async with self._gen_lock:
            pass
        return True

    def resume_after_refit(self) -> None:
        """Allow generation to continue after a weight refit."""
        self._refit_requested = False

    def get_dataloader_state(self) -> Optional[dict[str, Any]]:
        """Return the underlying dataloader's state dict for checkpointing."""
        if self._dataloader is None:
            return None
        return self._dataloader.state_dict()

    def stop(self) -> None:
        """Stop the collection loop."""
        self._stopped = True

    # Collection loop
    async def start_collection(self, dataloader: Any) -> None:
        """Continuously generate trajectory groups and add them to the buffer."""
        self._dataloader = dataloader
        loop = asyncio.get_event_loop()
        data_iter = iter(dataloader)

        while not self._stopped:
            # Hold off while paused, during a refit, or when running too far ahead
            # of the current weight version.
            if (
                self._paused
                or self._refit_requested
                or self._next_target
                > self._weight_version + self._max_trajectory_age_steps
            ):
                await asyncio.sleep(0.1)
                continue

            # Pull the next prompt batch (re-iterating the dataloader each epoch).
            batch = await loop.run_in_executor(None, self._next_batch, data_iter)
            if batch is _EPOCH_END:
                data_iter = iter(dataloader)
                continue

            # Generate rollouts for the whole prompt batch in a worker thread so
            # the event loop stays responsive to control calls.
            async with self._gen_lock:
                if self._stopped or self._paused or self._refit_requested:
                    continue
                gen_version = self._weight_version
                try:
                    repeated_batch, rollout_metrics = await loop.run_in_executor(
                        None, self._generate_batch, batch
                    )
                except Exception as exc:  # noqa: BLE001 - surface and continue
                    print(f"⚠️ AsyncTrajectoryCollector rollout failed: {exc}")
                    await asyncio.sleep(0.5)
                    continue

            # OPD: score the generated trajectories through the routed teacher(s)
            # and stamp teacher_reference_logprobs onto the batch (off the event
            # loop, like generation). No-op when no teachers are configured.
            if self._has_distillation_teachers:
                try:
                    await loop.run_in_executor(
                        None, self._score_teacher_logprobs, repeated_batch
                    )
                except Exception as exc:  # noqa: BLE001 - surface and continue
                    print(f"⚠️ OPD teacher scoring failed: {exc}")
                    await asyncio.sleep(0.5)
                    continue

            # Split the rollout batch back into per-prompt groups and enqueue each
            # with its target step.
            await self._enqueue_groups(repeated_batch, rollout_metrics, gen_version)

    # Internals
    @staticmethod
    def _next_batch(data_iter: Any) -> Any:
        """Advance the dataloader iterator; returns _EPOCH_END at epoch boundary."""
        try:
            return next(data_iter)
        except StopIteration:
            return _EPOCH_END

    def _generate_batch(self, batch: Any) -> tuple[Any, dict[str, Any]]:
        """Run a multi-turn rollout for one prompt batch (blocking; thread-only)."""
        # Deferred import: avoids a module-load cycle with experience/rollouts.
        # Select the multimodal computer-use rollout when grpo.cua_rollout is set
        # (per-turn screenshots threaded into the message log), else the text
        # multi-turn rollout. Mirrors grpo._select_async_rollout_fn so the async
        # collector and the sync path drive the same rollout for a given config.
        if self._master_config.grpo.get("cua_rollout", False):
            from dockyard_rl.experience.cua.rollout import (
                run_async_cua_rollout as rollout_fn,
            )
        else:
            from dockyard_rl.experience.rollouts import (
                run_async_multi_turn_rollout as rollout_fn,
            )

        # Non-None only for the text rollout (the resolvers reject the CUA
        # combination), so the kwargs are always accepted by rollout_fn.
        from dockyard_rl.algorithms.grpo import (
            resolve_invalid_action_cfg,
            resolve_structured_tool_use_cfg,
        )

        invalid_action_cfg = resolve_invalid_action_cfg(self._master_config)
        structured_cfg = resolve_structured_tool_use_cfg(self._master_config)
        extra_kwargs: dict[str, Any] = {}
        if invalid_action_cfg is not None:
            extra_kwargs["invalid_action_cfg"] = invalid_action_cfg
        if structured_cfg is not None:
            extra_kwargs["structured_cfg"] = structured_cfg

        repeated_batch = batch.repeat_interleave(self._num_generations_per_prompt)
        return rollout_fn(
            policy_generation=self._policy_generation,
            input_batch=repeated_batch,
            tokenizer=self._tokenizer,
            task_to_env=self._task_to_env,
            max_seq_len=self._max_seq_len,
            max_rollout_turns=self._max_rollout_turns,
            greedy=False,
            **extra_kwargs,
        )

    async def _enqueue_groups(
        self,
        repeated_batch: Any,
        rollout_metrics: dict[str, Any],
        gen_version: int,
    ) -> None:
        """Slice a rollout batch into per-prompt groups and add them to the buffer.

        Rows are grouped by prompt (`repeat_interleave` emits each prompt's
        `num_generations_per_prompt` samples contiguously). Each group is one
        ReplayBuffer entry tagged with its target training step.
        """
        n = self._num_generations_per_prompt
        total = repeated_batch.size
        num_groups = total // n

        for g in range(num_groups):
            group_batch = repeated_batch.select(list(range(g * n, (g + 1) * n)))
            if self._dp_enabled:
                trajectory = self._group_to_dataplane_trajectory(
                    group_batch, rollout_metrics
                )
            else:
                trajectory = {"batch": group_batch, "rollout_metrics": rollout_metrics}
            target = self._next_target

            status = await self._buffer_add(trajectory, gen_version, target)
            while status == "full" and not self._stopped:
                # Buffer at capacity: back off and retry so no group is dropped.
                await asyncio.sleep(0.2)
                status = await self._buffer_add(trajectory, gen_version, target)

            self._groups_for_target += 1
            if self._groups_for_target >= self._num_prompts_per_step:
                self._next_target += 1
                self._groups_for_target = 0

            if (
                self._refit_requested
                or self._next_target
                > self._weight_version + self._max_trajectory_age_steps
            ):
                # Stop emitting further groups from this batch until the trainer
                # advances the weight version.
                break

    def _group_to_dataplane_trajectory(
        self, group_batch: Any, rollout_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Write one group's bulk to the data plane; return a meta+carry entry.

        Mints one uid per prompt group (``{uid}_g{i}`` per generation), flattens
        the group, and ``kv_first_write``s the bulk columns. The buffer entry
        carries only the ``KVBatchMeta`` + ``driver_carry`` — no bulk tensors.
        """
        from dockyard_rl.data_plane.column_io import kv_first_write
        from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

        fb = group_batch.to("cpu")
        bulk, driver_carry = _flatten_group_for_dataplane(
            fb, self._tokenizer, self._dp_pad_multiple
        )
        n = int(bulk["sample_mask"].shape[0])
        uid = str(uuid.uuid4())
        sample_ids = [f"{uid}_g{i}" for i in range(n)]
        meta = kv_first_write(
            bulk,
            sample_ids=sample_ids,
            dp_client=self._dp_client,
            partition_id=self._dp_partition_id,
            extra_info={"rollout_metrics": rollout_metrics},
            task_name=self._dp_partition_id,
            pad_to_multiple=self._dp_pad_multiple,
        )
        return {
            "meta": meta,
            "driver_carry": BatchedDataDict(driver_carry),
            "rollout_metrics": rollout_metrics,
        }

    async def _buffer_add(
        self, trajectory: dict[str, Any], weight_version: int, target: int
    ) -> str:
        return await self._replay_buffer.add.remote(
            trajectory=trajectory,
            weight_version=weight_version,
            target_weight_version=target,
        )

    # On-policy distillation: teacher scoring
    def _score_teacher_logprobs(self, repeated_batch: Any) -> None:
        """Score the rollout batch through the routed teacher(s) in place.

        Flattens the trajectory message logs to the teacher's pad grid, computes
        per-token teacher logprobs, and stamps ``teacher_reference_logprobs`` onto
        ``repeated_batch`` (``from_batches`` then carries it through the per-group
        slicing into the trainer's advantage computation). No-op when there are no
        teachers. Runs off the event loop (called via run_in_executor).
        """
        from dockyard_rl.data.llm_message_utils import (
            batched_message_log_to_flat_message,
        )

        if "message_log" not in repeated_batch:
            return

        flat_for_teacher, teacher_input_lengths = batched_message_log_to_flat_message(
            repeated_batch["message_log"],
            pad_value_dict={"token_ids": self._tokenizer.pad_token_id},
            make_sequence_length_divisible_by=self._teacher_seq_pad_multiple,
        )
        token_ids = flat_for_teacher["token_ids"]
        batch_size = int(token_ids.shape[0])

        # Per-sample agent routing. Without per-sample agent refs (single-teacher
        # recipes), route every sample to the default teacher alias.
        agent_refs = repeated_batch.get("agent_ref")
        if not isinstance(agent_refs, list) or len(agent_refs) != batch_size:
            agent_refs = [{"name": "__opd_default__"}] * batch_size

        teacher_logprobs, _ = self._compute_teacher_logprobs(
            token_ids, agent_refs, input_lengths=teacher_input_lengths
        )
        repeated_batch["teacher_reference_logprobs"] = teacher_logprobs

    def _compute_teacher_logprobs(
        self,
        input_ids: Any,
        agent_refs: list[dict[str, Any]],
        input_lengths: Optional[Any] = None,
    ) -> tuple[Any, float]:
        """Compute teacher logprobs for non-colocated teachers.

        Groups samples by routed teacher, fans out in parallel (one thread per
        teacher, serialized per teacher by a lock), and stitches the results back
        into a ``[B, S]`` tensor index-aligned with ``input_ids``.
        """
        import concurrent.futures
        import time
        from collections import defaultdict

        import torch

        from dockyard_rl.algorithms.opd import resolve_reference_aliases
        from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

        opd_cfg = self._on_policy_distillation_cfg
        teacher_model_by_agent_name = opd_cfg.get("teacher_model_by_agent_name", {})
        default_teacher_alias = opd_cfg.get("default_teacher_alias")
        strict = opd_cfg.get("strict_agent_name_match", False)

        reference_aliases = resolve_reference_aliases(
            agent_refs,
            teacher_model_by_agent_name,
            default_teacher_alias=default_teacher_alias,
            strict_agent_name_match=strict,
        )
        group_keys = [self._alias_to_group_alias.get(a, a) for a in reference_aliases]

        group_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, gk in enumerate(group_keys):
            group_to_indices[gk].append(i)

        B, S = input_ids.shape
        result = torch.zeros(B, S, dtype=torch.float32)
        if not group_to_indices:  # empty batch — nothing to route
            return result, 0.0

        def _get_logprobs_for_group(group_key, indices):
            twg = self._teacher_worker_groups[group_key]
            sub_input_ids = input_ids[indices]
            sub_lengths = input_lengths[indices] if input_lengths is not None else None

            # Pad the sub-batch to a multiple of the teacher's DP size.
            dp_size = twg.sharding_annotations.get_axis_size("data_parallel")
            actual_batch_size = sub_input_ids.shape[0]
            remainder = actual_batch_size % dp_size
            if remainder != 0:
                pad_count = dp_size - remainder
                pad_rows = sub_input_ids[-1:].expand(pad_count, -1)
                sub_input_ids = torch.cat([sub_input_ids, pad_rows], dim=0)
                if sub_lengths is not None:
                    sub_lengths = torch.cat(
                        [sub_lengths, sub_lengths[-1:].expand(pad_count)], dim=0
                    )

            sub_data = BatchedDataDict({"input_ids": sub_input_ids})
            if sub_lengths is not None:
                sub_data["input_lengths"] = sub_lengths

            # Serialize forwards on a given teacher (NCCL collective safety).
            with self._teacher_locks[group_key]:
                logprobs_result = twg.get_logprobs(sub_data)
            logprobs = logprobs_result["reference_logprobs"][:actual_batch_size]
            return indices, logprobs

        t_total_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(group_to_indices)
        ) as executor:
            futures = {
                executor.submit(_get_logprobs_for_group, gk, idxs): gk
                for gk, idxs in group_to_indices.items()
            }
            for future in concurrent.futures.as_completed(futures):
                indices, logprobs = future.result()
                result[indices] = logprobs
        return result, time.time() - t_total_start


@ray.remote  # pragma: no cover
class AsyncTrajectoryCollector(AsyncTrajectoryCollectorImpl):
    pass
