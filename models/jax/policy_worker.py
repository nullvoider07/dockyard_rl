"""JaxPolicyWorker — the trainer-side worker, built incrementally across phases.

J4 adds the training path (`train`/`prepare_for_training`/`finish_training`).
J6 adds the logprob methods (and PolicyInterface inheritance); J5 the refit
seam; J7 checkpointing; J9 the Ray-actor wrapping + full `MasterConfig`
construction and the `lm_policy` selection. For now it is a plain class
constructed directly in tests; the trainer fleet's lifecycle/offload methods
are no-ops (the JAX trainer is a separate fleet from vLLM; XLA grows-on-demand
via XLA_PYTHON_CLIENT_PREALLOCATE=false).
"""

from __future__ import annotations

import sys
from typing import Any, Mapping, Optional, cast

import jax.numpy as jnp
from jax import lax
from flax import nnx
import ray

from dockyard_rl.models.jax.convert import jnp_to_torch, torch_to_jnp
from dockyard_rl.models.jax.loss import logprobs_from_logits
from dockyard_rl.models.jax.optimizer import build_optimizer
from dockyard_rl.models.jax.train_step import accumulate_grads, global_valid_counts, train_step

# Columns the GRPO loss reads; converted torch->jnp at the worker boundary.
_LOSS_COLUMNS = (
    "input_ids",
    "advantages",
    "token_mask",
    "sample_mask",
    "prev_logprobs",
    "generation_logprobs",
    "reference_policy_logprobs",
    "curr_logprobs_unfiltered",
)


def _to_jnp_column(value: Any) -> jnp.ndarray:
    """Convert a torch tensor to jnp; pass through arrays already on the JAX side."""
    if hasattr(value, "detach") or hasattr(value, "numpy"):
        return torch_to_jnp(value)
    return jnp.asarray(value)


def _to_float(scalar: Any) -> float:
    """Python float from a 0-d jax/numpy scalar.

    ``.item()`` already returns a python number at runtime; ``cast`` relabels it
    (jax's stub types ``.item()`` as ArrayLike, which ``float()`` rejects).
    """
    return cast(float, jnp.asarray(scalar).item())


# Fields clipped_pg_loss reads off the loss config; a ClippedPGLossFn instance
# exposes all of them as attributes except token_level_loss (it stores loss_type).
_LOSS_CFG_FIELDS = (
    "ratio_clip_min", "ratio_clip_max", "ratio_clip_c", "reference_policy_kl_penalty",
    "reference_policy_kl_type", "kl_input_clamp_value", "kl_output_clamp_value",
    "use_importance_sampling_correction", "truncated_importance_sampling_type",
    "truncated_importance_sampling_ratio", "truncated_importance_sampling_ratio_min",
    "use_on_policy_kl_approximation", "sequence_level_importance_ratios",
    "force_on_policy_ratio", "disable_ppo_ratio",
)


def _temperature_from_generation(generation_cfg: Optional[Mapping[str, Any]]) -> float:
    """Logprob temperature from the policy ``generation`` block.

    Mirrors the torch worker (dtensor_policy_worker.py:279): training-time logits
    are divided by the generation temperature before the log-softmax. top_k/top_p
    filtering of training logits is rejected (the JAX worker has no filtered-logprob
    path; it interacts with the -inf masking convention) rather than silently ignored.
    """
    if not generation_cfg:
        return 1.0
    top_k = generation_cfg.get("top_k")
    top_p = generation_cfg.get("top_p")
    if (top_k is not None and top_k > 0) or (top_p is not None and top_p != 1.0):
        raise NotImplementedError(
            "JAX trainer logprobs do not implement top-k/top-p filtering of training "
            f"logits (got top_k={top_k!r}, top_p={top_p!r}); only temperature scaling "
            "is supported. Set generation.top_k=null and generation.top_p=1.0."
        )
    temperature = generation_cfg.get("temperature", 1.0)
    return float(temperature) if temperature is not None else 1.0


class _DuckLossCfg:
    """Plain attribute bag duck-typed as a ClippedPGLossConfig for clipped_pg_loss."""

    token_level_loss: bool = True


def _cfg_from_loss_fn(loss_fn: Any) -> Any:
    """Derive a clipped_pg_loss config from the torch ClippedPGLossFn passed to train.

    Mirrors the torch contract (the loss config travels with the loss_fn), so the
    JAX worker uses exactly the run's configured loss without storing it itself.
    """
    cfg = _DuckLossCfg()
    for field in _LOSS_CFG_FIELDS:
        setattr(cfg, field, getattr(loss_fn, field, None))
    loss_type = getattr(loss_fn, "loss_type", None)
    cfg.token_level_loss = getattr(loss_type, "name", "TOKEN_LEVEL") == "TOKEN_LEVEL"
    return cfg


def _model_is_moe(model: nnx.Module) -> bool:
    """Whether ``model`` contains native GroupedExperts (routes refit through the
    EP-aware MoE expansion). Structural, so it also covers test-injected models."""
    from dockyard_rl.models.jax.moe.experts import GroupedExperts

    return any(isinstance(m, GroupedExperts) for _p, m in nnx.iter_modules(model))


def _model_is_qwen3_next(model: nnx.Module) -> bool:
    """Whether ``model`` is the Qwen3-Next hybrid (has a Gated-DeltaNet linear-attn
    layer). Structural, so it also covers test-injected models."""
    from dockyard_rl.models.jax.linear_attn.gated_deltanet import Qwen3NextGatedDeltaNet

    return any(isinstance(m, Qwen3NextGatedDeltaNet) for _p, m in nnx.iter_modules(model))


def _model_is_gemma4(model: nnx.Module) -> bool:
    """Whether ``model`` is a Gemma 4 model (its base ``Gemma4Model`` is present).

    Structural, so it also covers test-injected models. Routes refit through the
    Gemma 4 name-map: ``Gemma4Experts`` is not a ``GroupedExperts`` (so ``_model_is_moe``
    is False for it) and the PLE table / router scalars need explicit mapping the
    generic dense map cannot express."""
    from dockyard_rl.models.jax.models.gemma4 import Gemma4Model

    return any(isinstance(m, Gemma4Model) for _p, m in nnx.iter_modules(model))


def _wire_ep_dispatchers(model: nnx.Module, ep_axis: str, ep_size: int) -> int:
    """Install the EP mesh axis on every all-to-all token dispatcher in ``model``.

    Pure attribute wiring (mirrors the torch worker's ``dispatcher.wire_ep_mesh``
    at B.5b); the dispatcher's ``ragged_all_to_all`` only runs under a ``shard_map``
    on a live multi-device ``ep`` mesh (HV-29). Called after the mesh is built.
    Returns the number of dispatchers wired.
    """
    from dockyard_rl.models.jax.moe.alltoall import AllToAllTokenDispatcher
    from dockyard_rl.models.jax.moe.experts import GroupedExperts

    wired = 0
    for _p, module in nnx.iter_modules(model):
        if isinstance(module, GroupedExperts) and isinstance(
            module.dispatcher, AllToAllTokenDispatcher
        ):
            module.dispatcher.wire_ep(ep_axis, ep_size)
            wired += 1
    return wired


def _make_lb_update_fn(model: nnx.Module) -> Optional[Any]:
    """Build the per-optimizer-step expert-bias updater, or None if no LB blocks.

    Mirrors the torch ``register_expert_bias_update_hook``: discovers the
    load-balanced ``MoEBlock`` modules once, and returns a zero-arg callable that
    steps every block's ``expert_bias_E`` from its accumulated ``tokens_per_expert_E``
    and zeroes the counter. ``reduce_tokens_fn`` (the cross-rank dp/cp load
    all-reduce) stays ``None`` on a single process; the global reduction is
    GPU/cluster-bound (HV-29). Returns ``None`` for the dense path.
    """
    from dockyard_rl.models.jax.moe.load_balance import iter_lb_moe_blocks, update_expert_biases

    blocks = list(iter_lb_moe_blocks(model))
    if not blocks:
        return None
    return lambda: update_expert_biases(blocks, reduce_tokens_fn=None)


def _build_model_and_reference(
    config: Mapping[str, Any], weights_path: Optional[str], init_reference_model: bool
) -> tuple[nnx.Module, Optional[nnx.Module]]:
    """Build the NNX model (+ frozen reference) from a MasterConfig policy dict.

    Dispatches on the HF ``model_type``: ``qwen3`` -> dense ``Qwen3ForCausalLM``;
    ``qwen3_moe`` -> ``Qwen3MoeForCausalLM`` with the fused-expert loader. Real
    model download + load is GPU/cluster-validated (HV); the forward + loaders are
    CPU-parity-validated.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    model_name = config["model_name"]
    overrides = config.get("hf_config_overrides", {}) or {}
    hf_cfg = AutoConfig.from_pretrained(model_name, **overrides)
    model_type = getattr(hf_cfg, "model_type", "")
    precision = str(config.get("precision", "float32"))
    param_dtype = {"bfloat16": jnp.bfloat16, "float16": jnp.float16}.get(precision, jnp.float32)
    jax_cfg = config.get("jax_cfg") or {}
    # MoE knobs live in jax_cfg (alongside expert_parallel_size); a top-level
    # moe_load_balance_coeff stays accepted as a fallback for direct-dict tests.
    lb_coeff = config.get("moe_load_balance_coeff", jax_cfg.get("moe_load_balance_coeff"))
    dispatcher_kind = jax_cfg.get("moe_token_dispatcher", "local")
    ep_size = int(jax_cfg.get("expert_parallel_size", 1) or 1)

    # model_type "qwen3_moe" contains "qwen3" — check MoE first to avoid building
    # the dense model for an MoE checkpoint.
    if model_type == "qwen3_moe":
        from dockyard_rl.models.jax.moe.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
        from dockyard_rl.models.jax.moe.weights import load_hf_qwen3_moe_state_dict

        jcfg = Qwen3MoeConfig.from_hf_config(hf_cfg)
        if ep_size > 1:
            # EP routing needs a live multi-device ``ep`` mesh + ragged_all_to_all
            # (run under shard_map). _init_jax_distributed is a single-process no-op
            # here, so wire_ep + the collective cannot run; gate until GPU bring-up.
            raise NotImplementedError(
                f"expert_parallel_size={ep_size} requires a live multi-device JAX mesh and "
                "the ragged_all_to_all EP collective; single-process EP is unsupported "
                "(see HV-29/HV-30). Set jax_cfg.expert_parallel_size=1 for the CPU path."
            )

        def _build(seed: int) -> nnx.Module:
            m = Qwen3MoeForCausalLM(
                jcfg, rngs=nnx.Rngs(params=seed), param_dtype=param_dtype,
                load_balance_coeff=lb_coeff, token_dispatcher=dispatcher_kind,
            )
            load_hf_qwen3_moe_state_dict(m, state_dict, param_dtype=param_dtype)
            return m
    elif model_type == "qwen3_next":
        from dockyard_rl.models.jax.linear_attn.weights import load_hf_qwen3_next_state_dict
        from dockyard_rl.models.jax.models.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM

        jcfg_next = Qwen3NextConfig.from_hf_config(hf_cfg)
        if ep_size > 1:
            raise NotImplementedError(
                f"expert_parallel_size={ep_size} for qwen3_next requires a live EP mesh + "
                "alltoall dispatch (HV-29/30); set jax_cfg.expert_parallel_size=1."
            )

        def _build(seed: int) -> nnx.Module:
            m = Qwen3NextForCausalLM(jcfg_next, rngs=nnx.Rngs(params=seed), param_dtype=param_dtype)
            load_hf_qwen3_next_state_dict(m, state_dict, param_dtype=param_dtype)
            return m
    elif "qwen3" in model_type:
        from dockyard_rl.models.jax.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
        from dockyard_rl.models.jax.weights import load_hf_state_dict

        jcfg = Qwen3Config.from_hf_config(hf_cfg)

        def _build(seed: int) -> nnx.Module:
            m = Qwen3ForCausalLM(jcfg, rngs=nnx.Rngs(params=seed), param_dtype=param_dtype)
            load_hf_state_dict(m, state_dict, param_dtype=param_dtype)
            return m
    elif "gemma4" in model_type:
        from dockyard_rl.models.jax.models.gemma4 import (
            Gemma4ForCausalLM, Gemma4TextConfig, load_hf_gemma4_state_dict,
        )

        # Multimodal "gemma4" nests the text hyperparameters under text_config;
        # "gemma4_text" exposes them directly. Only the text path is wired.
        text_cfg = getattr(hf_cfg, "text_config", None) or hf_cfg
        jcfg_g4 = Gemma4TextConfig.from_hf_config(text_cfg)
        if jcfg_g4.enable_moe_block and ep_size > 1:
            raise NotImplementedError(
                f"expert_parallel_size={ep_size} for gemma4 MoE requires a live EP mesh + "
                "alltoall dispatch (HV-29/30); set jax_cfg.expert_parallel_size=1."
            )

        def _build(seed: int) -> nnx.Module:
            m = Gemma4ForCausalLM(jcfg_g4, rngs=nnx.Rngs(params=seed), param_dtype=param_dtype)
            load_hf_gemma4_state_dict(m, state_dict, param_dtype=param_dtype)
            return m
    else:
        raise NotImplementedError(
            "JAX trainer backend supports Qwen3 dense + Qwen3-MoE + Qwen3-Next + Gemma4; "
            f"got model_type={model_type!r}. Gemma3/Llama dense land in later phases."
        )

    src = weights_path or model_name
    state_dict = AutoModelForCausalLM.from_pretrained(src).state_dict()
    model = _build(0)
    reference = _build(1) if init_reference_model else None
    return model, reference


class JaxPolicyWorkerImpl:
    """Trainer-side JAX policy worker implementing the PolicyInterface /
    ColocatablePolicyInterface contract by duck typing (J9).

    Plain (inheritable) implementation; the Ray actor is the ``@ray.remote``
    ``JaxPolicyWorker`` subclass at module end. Construct this class directly in
    tests; the driver/RayWorkerBuilder uses the actor wrapper.

    Two construction modes:
      * production / Ray: ``JaxPolicyWorker(config, **worker_kwargs)`` builds the
        model + optimizer + frozen reference from the MasterConfig policy dict;
      * tests: ``JaxPolicyWorker(model, optimizer_cfg=..., loss_cfg=..., ...)``.

    Offload / generation-handoff methods are no-ops: the JAX trainer is a
    separate fleet from vLLM, and XLA grows on demand (PREALLOCATE=false).
    """

    # Attached by the Policy wrapper after worker construction.
    worker_group = None

    def __init__(
        self,
        model_or_config: Any,
        optimizer_cfg: Optional[Mapping[str, Any]] = None,
        loss_cfg: Any = None,
        *,
        init_optimizer: bool = True,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_reference_model: bool = True,
        worker_sharding_annotations: Any = None,
        pre_init_communication_queue: Any = None,
        reference_model: Optional[nnx.Module] = None,
        scheduler_cfg: Optional[Any] = None,
        max_grad_norm: Optional[float] = None,
        train_micro_batch_size: int = 1,
        logprob_batch_size: int = 1,
        generation_temperature: float = 1.0,
        **_ignored: Any,
    ) -> None:
        if isinstance(model_or_config, nnx.Module):
            self._setup(
                model_or_config, optimizer_cfg, loss_cfg, reference_model, scheduler_cfg,
                max_grad_norm, train_micro_batch_size, logprob_batch_size, init_optimizer,
                generation_temperature,
            )
        else:
            config = model_or_config
            self._init_jax_distributed(config, worker_sharding_annotations)
            model, reference = _build_model_and_reference(config, weights_path, init_reference_model)
            self._setup(
                model, config.get("optimizer", {}), None, reference, config.get("scheduler"),
                config.get("max_grad_norm"), int(config.get("train_micro_batch_size", 1)),
                int(config.get("logprob_batch_size", 1)), init_optimizer,
                _temperature_from_generation(config.get("generation")),
            )
        self.model_update_group = None  # torch NCCL refit group (init_collective; HV)

    def _setup(
        self, model: nnx.Module, optimizer_cfg: Optional[Mapping[str, Any]], loss_cfg: Any,
        reference_model: Optional[nnx.Module], scheduler_cfg: Optional[Any],
        max_grad_norm: Optional[float], tmbs: int, lpbs: int, init_optimizer: bool,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.reference_model = reference_model
        self.loss_cfg = loss_cfg
        self.temperature = temperature
        self._is_moe = _model_is_moe(model)
        self._is_qwen3_next = _model_is_qwen3_next(model)
        self._is_gemma4 = _model_is_gemma4(model)
        # Aux-loss-free expert-bias updater (None unless the model has load-balanced
        # MoEBlocks); runs once per optimizer step, mirroring the torch step pre-hook.
        self._lb_update_fn = _make_lb_update_fn(model) if self._is_moe else None
        self.train_micro_batch_size = tmbs
        self.logprob_batch_size = lpbs
        self.schedule: Any = None
        self.optimizer: Any = None
        if init_optimizer and optimizer_cfg:
            tx, schedule = build_optimizer(optimizer_cfg, scheduler_cfg, max_grad_norm)
            self.schedule = schedule
            self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        self._step = 0

    def _init_jax_distributed(self, config: Mapping[str, Any], sharding_annotations: Any) -> None:
        """Join the global JAX mesh (one process per device).

        Single-process / CPU is a no-op. Multi-host coordinator wiring (address,
        process_id, num_processes from the Ray placement) is GPU/cluster-only (HV).
        """
        return None

    # --- lifecycle (trainer-fleet subset; offload is a no-op) ---
    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        self.model.train()

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        pass

    # --- train ---
    def _convert_batch(self, data: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
        return {k: _to_jnp_column(data[k]) for k in _LOSS_COLUMNS if k in data}

    def train(
        self,
        data: Mapping[str, Any],
        loss_fn: Any = None,
        eval_mode: bool = False,
        *,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        timer: Optional[Any] = None,
    ) -> dict[str, Any]:
        """One GRPO training step. The loss config travels with ``loss_fn`` (the
        torch ClippedPGLossFn), matching the torch contract; ``self.loss_cfg`` is
        the test fallback when no ``loss_fn`` is passed."""
        loss_cfg = _cfg_from_loss_fn(loss_fn) if loss_fn is not None else self.loss_cfg
        batch = self._convert_batch(data)
        micro = mbs or self.train_micro_batch_size

        if eval_mode:
            gvs, gvt = global_valid_counts(batch)
            _grads, metrics = accumulate_grads(
                self.model, batch, gvs, gvt, loss_cfg, micro, self.temperature
            )
            applied_step = self._step
        else:
            metrics = train_step(
                self.model, self.optimizer, batch, loss_cfg, micro, self.temperature,
                pre_optim_step_fn=self._lb_update_fn,
            )
            # optax applies schedule(count) with count starting at 0; log that same
            # step, then advance (lr logged matches the lr actually applied).
            applied_step = self._step
            self._step += 1

        lr = _to_float(self.schedule(applied_step))
        out: dict[str, Any] = {k: _to_float(v) for k, v in metrics.items()}
        out["lr"] = lr
        return out

    # --- checkpoint ---
    def save_checkpoint(self, path: str, *args: Any, **kwargs: Any) -> None:
        """Persist model params + optimizer state + step via Orbax."""
        from dockyard_rl.models.jax.checkpoint import save_checkpoint

        save_checkpoint(path, self.model, optimizer=self.optimizer, step=self._step)

    # --- logprob / ingestion seam (Seam 2) ---
    def _logprobs_full(self, model: nnx.Module, input_ids: jnp.ndarray, mbs: int) -> jnp.ndarray:
        """Per-token logprobs [B, S] with the position-0 = 0.0 convention.

        Forward in microbatches (no grad), gather next-token logprobs ([B, S-1]),
        then prepend a zero column so position i holds the logprob of token i
        (matching the torch worker, dtensor_policy_worker.py:1257).
        """
        batch = input_ids.shape[0]
        chunks: list[jnp.ndarray] = []
        for i in range(0, batch, mbs):
            ids = input_ids[i : i + mbs]
            logits = model(ids)
            if self.temperature != 1.0:
                logits = logits / self.temperature
            lp = logprobs_from_logits(logits, ids)  # [b, S-1]
            lp = jnp.concatenate([jnp.zeros_like(lp[:, :1]), lp], axis=1)  # [b, S]
            chunks.append(lp)
        return jnp.concatenate(chunks, axis=0)

    def _apply_length_mask(self, logprobs: jnp.ndarray, data: Mapping[str, Any]) -> jnp.ndarray:
        """Zero logprobs at/after each sequence's length (padding positions)."""
        if "input_lengths" not in data:
            return logprobs
        lengths = _to_jnp_column(data["input_lengths"]).reshape(-1)
        positions = jnp.arange(logprobs.shape[1])[None, :]
        keep = (positions < lengths[:, None]).astype(logprobs.dtype)
        return logprobs * keep

    def get_logprobs(
        self, data: Mapping[str, Any], micro_batch_size: Optional[int] = None,
        timer: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Per-token logprobs of the active policy as ``{"logprobs": [B, S]}``."""
        mbs = micro_batch_size or self.logprob_batch_size
        input_ids = _to_jnp_column(data["input_ids"])
        lp = self._apply_length_mask(self._logprobs_full(self.model, input_ids, mbs), data)
        return {"logprobs": jnp_to_torch(lp)}

    def get_reference_policy_logprobs(
        self, data: Mapping[str, Any], micro_batch_size: Optional[int] = None,
        timer: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Per-token logprobs of the frozen reference policy (moved to CPU)."""
        if self.reference_model is None:
            raise RuntimeError("get_reference_policy_logprobs called but no reference_model is set.")
        mbs = micro_batch_size or self.logprob_batch_size
        input_ids = _to_jnp_column(data["input_ids"])
        lp = self._apply_length_mask(self._logprobs_full(self.reference_model, input_ids, mbs), data)
        return {"reference_logprobs": jnp_to_torch(lp).cpu()}

    def get_topk_logits(
        self, data: Mapping[str, Any], k: int, micro_batch_size: Optional[int] = None,
        timer: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Top-k logits + global token indices, shape [B, S-1, k] (next-token aligned).

        Correct on a single device / replicated vocab. Under a tensor-parallel
        mesh the lm_head output is vocab-sharded, so a globally-correct top-k
        needs an all-gather/reduction over the tp axis (cf. the torch path's
        `distributed_vocab_topk`); that sharded reduction is GPU-deferred (HV).
        get_topk_logits is used for PRM/distillation, not the main GRPO loop.
        """
        mbs = micro_batch_size or self.logprob_batch_size
        input_ids = _to_jnp_column(data["input_ids"])
        vals_chunks: list[jnp.ndarray] = []
        idx_chunks: list[jnp.ndarray] = []
        for i in range(0, input_ids.shape[0], mbs):
            logits = self.model(input_ids[i : i + mbs]).astype(jnp.float32)[:, :-1]
            if self.temperature != 1.0:
                logits = logits / self.temperature
            vals, idx = lax.top_k(logits, k)
            vals_chunks.append(vals)
            idx_chunks.append(idx)
        return {
            "topk_logits": jnp_to_torch(jnp.concatenate(vals_chunks, axis=0)),
            "topk_indices": jnp_to_torch(jnp.concatenate(idx_chunks, axis=0)),
        }

    def calibrate_qkv_fp8_scales(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "FP8 KV-cache calibration is an inference-side concern; the JAX trainer is bf16."
        )

    # --- refit seam (Seam 1) wiring ---
    def prepare_refit_info(self, state_dict_info: Optional[Any] = None) -> Optional[dict[str, Any]]:
        """HF-name -> (shape, dtype) map for the refit transport (CPU-wired).

        MoE models expand fused experts into per-expert HF names so the declared
        map matches the streamed tensors (same iteration in both)."""
        if self._is_qwen3_next:
            from dockyard_rl.models.jax.linear_attn.refit import prepare_qwen3_next_refit_info

            return prepare_qwen3_next_refit_info(self.model)
        if self._is_gemma4:
            from dockyard_rl.models.jax.models.gemma4_refit import prepare_gemma4_refit_info

            return prepare_gemma4_refit_info(self.model)
        if self._is_moe:
            from dockyard_rl.models.jax.moe.refit import prepare_moe_refit_info

            return prepare_moe_refit_info(self.model)
        from dockyard_rl.models.jax.refit import prepare_refit_info

        return prepare_refit_info(self.model)

    def broadcast_weights_for_collective(self, kv_scales: Optional[dict[str, float]] = None) -> list:
        """Stream JAX params (HF-named, dlpack->torch) into the NCCL refit producer.

        The tensor source is JAX; the NCCL ``model_update_group`` and
        ``packed_broadcast_producer`` stay torch (unchanged). The live broadcast
        is GPU/cluster-only (HV-27); this assembles the producer call.
        """
        if self.model_update_group is None:
            raise RuntimeError("broadcast_weights_for_collective requires init_collective() first.")
        from dockyard_rl.utils.packed_tensor import packed_broadcast_producer

        if self._is_qwen3_next:
            from dockyard_rl.models.jax.linear_attn.refit import iter_qwen3_next_refit_state_dict

            stream = iter_qwen3_next_refit_state_dict(self.model)
        elif self._is_gemma4:
            from dockyard_rl.models.jax.models.gemma4_refit import iter_gemma4_refit_state_dict

            stream = iter_gemma4_refit_state_dict(self.model)
        elif self._is_moe:
            from dockyard_rl.models.jax.moe.refit import iter_moe_refit_state_dict

            stream = iter_moe_refit_state_dict(self.model)
        else:
            from dockyard_rl.models.jax.refit import iter_refit_state_dict

            stream = iter_refit_state_dict(self.model)
        packed_broadcast_producer(
            iter(stream),
            self.model_update_group,
            src=0,
            post_iter_func=lambda nt: nt[1],
        )
        return []

    def init_collective(self, ip: str, port: int, world_size: int, *, train_world_size: int) -> list:
        """Set up the torch NCCL refit group (GPU/cluster-only; HV-27)."""
        raise NotImplementedError(
            "init_collective (torch NCCL model_update_group) is GPU/cluster-only; see HV-27."
        )

    # --- data-plane (torch transport) variants — integration deferred ---
    def train_from_meta(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "Data-plane (*_from_meta) routing for the JAX worker is deferred (torch transport); "
            "use the colocated path (train/get_logprobs)."
        )

    def get_logprobs_from_meta(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Data-plane get_logprobs_from_meta is deferred for the JAX worker.")

    def get_reference_policy_logprobs_from_meta(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "Data-plane get_reference_policy_logprobs_from_meta is deferred for the JAX worker."
        )

    # --- colocation lifecycle (trainer-fleet subset: no-ops) ---
    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def prepare_for_lp_inference(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def invalidate_kv_cache(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def offload_before_refit(self) -> None:
        return None

    def offload_after_refit(self) -> None:
        return None

    def shutdown(self) -> bool:
        """Release worker resources (XLA manages device memory)."""
        return True


# Ray actor wrapper. @ray.remote classes can't be inherited from, so the logic
# lives in JaxPolicyWorkerImpl and this thin subclass is the actor the
# RayWorkerBuilder constructs (FQN: dockyard_rl.models.jax.policy_worker.
# JaxPolicyWorker). runtime_env pins the worker to sys.executable (convention #3:
# no uv/venv at runtime).
@ray.remote(runtime_env={"py_executable": sys.executable})  # pragma: no cover
class JaxPolicyWorker(JaxPolicyWorkerImpl):
    pass
