import contextlib
import gc
import itertools
import os
import sys
import warnings
from typing import Iterator
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Generator, Iterable, Optional, Set, Union, cast
import ray
import torch
from accelerate import init_empty_weights
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import set_rotate_method
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)

from dockyard_rl.data_plane.worker_mixin import TQWorkerMixin
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.models.dtensor.moe.router_replay import (
    resolve_router_replay_enabled,
    router_replay_context,
    validate_router_replay_config,
)
from dockyard_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
    ScoreOutputSpec,
)
from dockyard_rl.utils.packed_tensor import packed_broadcast_producer

# Deferred: algorithms/loss submodules
try:
    from dockyard_rl.algorithms.logits_sampling_utils import (
        TrainingSamplingParams,
        apply_top_k_top_p,
        need_top_k_or_top_p_filtering,
    )
except ImportError:
    TrainingSamplingParams = cast(Any, None)
    apply_top_k_top_p = cast(Any, None)
    need_top_k_or_top_p_filtering = cast(Any, None)

try:
    from dockyard_rl.algorithms.loss import SequencePackingLossWrapper, prepare_loss_input
except ImportError:
    SequencePackingLossWrapper = cast(Any, None)
    prepare_loss_input = cast(Any, None)

try:
    from dockyard_rl.algorithms.loss.interfaces import LossFunction, LossType
except ImportError:
    LossFunction = Any  # type: ignore
    LossType = cast(Any, None)

try:
    from dockyard_rl.algorithms.utils import mask_out_neg_inf_logprobs
except ImportError:
    mask_out_neg_inf_logprobs = cast(Any, None)

# Deferred: distributed/model_utils.py (not yet written)
try:
    from dockyard_rl.distributed.model_utils import (
        allgather_cp_sharded_tensor,
        distributed_vocab_topk,
        get_logprobs_from_vocab_parallel_logits,
    )
except ImportError:
    allgather_cp_sharded_tensor = cast(Any, None)
    distributed_vocab_topk = cast(Any, None)
    get_logprobs_from_vocab_parallel_logits = cast(Any, None)

# Deferred: models/dtensor/parallelize.py (not yet written)
try:
    from dockyard_rl.models.dtensor.parallelize import (
        _parallelize_model,
        clip_grad_by_total_norm_,
        get_grad_norm,
        to_local_if_dtensor,
    )
except ImportError:
    _parallelize_model = cast(Any, None)
    clip_grad_by_total_norm_ = cast(Any, None)
    get_grad_norm = cast(Any, None)
    to_local_if_dtensor = cast(Any, None)

# Deferred: models/huggingface/common.py (not yet written)
try:
    from dockyard_rl.models.huggingface.common import (
        get_flash_attention_kwargs,
        pack_sequences,
    )
except ImportError:
    get_flash_attention_kwargs = cast(Any, None)
    pack_sequences = cast(Any, None)

# Deferred: models/policy/__init__.py with PolicyConfig
try:
    from dockyard_rl.models.policy import PolicyConfig
except ImportError:
    PolicyConfig = dict  # type: ignore

# Deferred: models/policy/utils.py (not yet written)
try:
    from dockyard_rl.models.policy.utils import (
        configure_dynamo_cache,
        get_runtime_env_for_policy_worker,
        resolve_model_class,
    )
except ImportError:
    def configure_dynamo_cache() -> None:  # type: ignore
        pass

    def get_runtime_env_for_policy_worker(name: str) -> dict:  # type: ignore
        return {"py_executable": sys.executable}

    def resolve_model_class(model_type: str) -> Any:  # type: ignore
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM

# Deferred: models/policy/workers/base_policy_worker.py (not yet written)
try:
    from dockyard_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
except ImportError:
    class AbstractPolicyWorker:  # type: ignore
        pass

# Deferred: utils/native_checkpoint.py (not yet written)
try:
    from dockyard_rl.utils.native_checkpoint import (
        load_checkpoint,
        save_checkpoint,
    )
except ImportError:
    load_checkpoint = None  # type: ignore
    save_checkpoint = None  # type: ignore

# Deferred: utils/nsys.py (#17 in backlog)
try:
    from dockyard_rl.utils.nsys import wrap_with_nvtx_name
except ImportError:
    def wrap_with_nvtx_name(name: str):  # type: ignore
        def decorator(fn):
            return fn
        return decorator

# Deferred: hydra dependency for optimizer class resolution
try:
    from hydra.utils import get_class
except ImportError:
    def get_class(path: str) -> Any:  # type: ignore
        module_path, cls_name = path.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)

def _attach_context_parallel_hooks(model: nn.Module) -> None:
    """Attach forward pre-hooks to self_attn modules for context parallelism.

    CP shards Q/K/V on the sequence dimension as DTensors, so explicit 4D
    attention masks cause shape mismatches. This registers a hook on every
    ``self_attn`` sub-module that strips ``attention_mask`` and sets
    ``is_causal=True``, letting SDPA handle causal masking internally.
    """

    def _hook(_module, module_args, module_kwargs):
        if "attention_mask" in module_kwargs:
            module_kwargs["attention_mask"] = None
            module_kwargs["is_causal"] = True
        return module_args, module_kwargs

    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            module.register_forward_pre_hook(_hook, with_kwargs=True, prepend=True)

@contextmanager
def unshard_fsdp2_model(model: nn.Module) -> Iterator[None]:
    """Explicitly unshard and then reshard the FSDP2 modules. Useful for logprob inference."""
    try:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        yield
    finally:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

@torch.no_grad()
def get_cpu_state_dict(
    state_generator: Iterable[tuple[str, Union[torch.Tensor, DTensor]]],
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    """Copy the state dict generator to CPU memory.

    Args:
        state_generator: An iterable that yields (key, tensor) pairs from a model state.
        pin_memory: Whether to allocate the CPU tensors in pinned memory for faster GPU transfer.

    Returns:
        dict[str, torch.Tensor]: A dictionary mapping parameter names to CPU tensors.
    """
    new_state_dict = {}
    for k, v in state_generator:
        val = to_local_if_dtensor(v)

        if len(val.shape) == 0:
            new_state_dict[k] = val.cpu()
        else:
            cpu_tensor = torch.empty(
                *val.shape, device="cpu", pin_memory=pin_memory, dtype=val.dtype
            )
            cpu_tensor.copy_(val, non_blocking=True)
            new_state_dict[k] = cpu_tensor

    torch.cuda.synchronize()
    return new_state_dict

# Classes with @ray.remote can't be inherited from, so we split the implementation out.
# This is useful when using worker extension classes.
class DTensorPolicyWorkerImpl(
    TQWorkerMixin, AbstractPolicyWorker, ColocatablePolicyInterface
):  # type: ignore[misc]
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs."""
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def _get_replica_group(self) -> Optional[Any]:
        """Replica group = flattened (cp, tp) sub-mesh, for NCCL broadcast in ``_fetch``."""
        return self.device_mesh[("cp", "tp")]._flatten().get_group()

    def _local_coords(self) -> dict[str, int]:
        # No PP axis on this worker's mesh; pipeline_parallel is absent and
        # NamedSharding.is_axis_zero treats a missing axis as rank 0.
        return {
            "tensor_parallel": self.device_mesh["tp"].get_local_rank(),
            "context_parallel": self.device_mesh["cp"].get_local_rank(),
        }

    def __init__(
        self,
        config: PolicyConfig,  # type: ignore[valid-type]
        tokenizer: PreTrainedTokenizerBase,
        processor: Optional[AutoProcessor] = None,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        **kwargs: Any,
    ):
        """Initialize the DTensorPolicyWorker."""
        self.tokenizer = tokenizer
        self.processor = processor
        self.is_vlm = processor is not None

        print(f"Initializing DTensorPolicyWorker with is_vlm={self.is_vlm}")

        self.is_generation_colocated = None
        self.sampling_params = None
        if "generation" in config and config["generation"] is not None:
            generation_cfg = config["generation"]
            self.is_generation_colocated = generation_cfg["colocated"]["enabled"]
            self.sampling_params = TrainingSamplingParams(
                top_k=generation_cfg["top_k"],
                top_p=generation_cfg["top_p"],
                temperature=generation_cfg["temperature"],
            )

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        self.cfg = config
        torch.distributed.init_process_group(backend="nccl")
        self.rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_name = self.cfg["model_name"]

        self.cpu_offload = self.cfg["dtensor_cfg"]["cpu_offload"]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]
        self.max_grad_norm = self.cfg["max_grad_norm"]

        if self.cfg["precision"] == "float32":
            self.dtype = torch.float32
        elif self.cfg["precision"] == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.cfg["precision"] == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unknown precision: {self.cfg['precision']}")

        print(f"[Rank {self.rank}] Loading model {model_name} on CPU...")
        self.enable_seq_packing = self.cfg["sequence_packing"]["enabled"]
        if self.enable_seq_packing:
            assert not self.is_vlm, (
                "Sequence packing is not supported for VLM models. Please set policy.sequence_packing.enabled = False to train VLM models."
            )
            print(
                f"[Rank {self.rank}] Sequence packing is enabled for model {model_name}"
            )
            print(f"[Rank {self.rank}] Using FlashAttention2 for sequence packing")

        hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}

        model_config = AutoConfig.from_pretrained(
            model_name,
            # Always load the model in float32 to keep master weights in float32.
            # Keeping the master weights in lower precision has shown to cause issues with convergence.
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
            if self.enable_seq_packing
            else None,
            **hf_config_overrides,
        )

        # reward model
        self._is_reward_model = (
            "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]
        )
        if self._is_reward_model:
            if self.enable_seq_packing:
                raise NotImplementedError(
                    "Sequence packing is not supported for reward models"
                )
            rm_type = self.cfg["reward_model_cfg"]["reward_model_type"]
            if rm_type == "bradley_terry":
                model_class = AutoModelForSequenceClassification
                if model_config.num_labels != 1:
                    print(
                        "model_config.num_labels is not 1. Setting it to 1 since this value is used as the out_features "
                        "for the linear head of Bradley-Terry reward models."
                    )
                    model_config.num_labels = 1
            else:
                raise ValueError(f"Unknown reward model type: {rm_type}")
        else:
            model_class = resolve_model_class(model_config.model_type)

        full_state_dict = None
        if self.rank == 0:
            print(f"[Rank {self.rank}] Loading model {model_name} on CPU...")
            model = model_class.from_pretrained(
                model_name,
                device_map="cpu",
                trust_remote_code=True,
                config=model_config,
            )
            self._apply_moe_surgery(model)
            full_state_dict = model.state_dict()
            del model

        print(f"[Rank {self.rank}] Initializing empty model for FSDP...")
        with init_empty_weights():
            self.model = model_class.from_config(
                model_config,
                trust_remote_code=True,
            )
            # Surgery inside init_empty_weights keeps the native expert params on
            # meta (no transient real allocation); the broadcast state dict from
            # rank 0 (also surgically converted) supplies the actual weights.
            self._apply_moe_surgery(self.model)

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = tokenizer.pad_token_id

        tp_size = self.cfg["dtensor_cfg"]["tensor_parallel_size"]
        cp_size = self.cfg["dtensor_cfg"]["context_parallel_size"]
        if cp_size > 1 and self.enable_seq_packing:
            raise ValueError(
                "Context parallel is not supported for sequence packing. "
                "Refer to model-quirks documentation for more details."
            )
        dp_size = world_size // tp_size // cp_size
        sequence_parallel_enabled = self.cfg["dtensor_cfg"]["sequence_parallel"]
        assert world_size == dp_size * tp_size * cp_size, (
            f"World size({world_size}) must equal to dp_size({dp_size}) * tp_size({tp_size}) * cp_size({cp_size}) to use DTensor"
        )

        if sequence_parallel_enabled and tp_size == 1:
            print(
                "[WARNING]: sequence_parallel=True, but tp_size=1 which has no effect. Enable tp_size > 1 to use sequence parallelism."
            )

        self.is_gemma3 = isinstance(self.model, Gemma3ForCausalLM) or isinstance(
            self.model, Gemma3ForConditionalGeneration
        )
        if cp_size > 1:
            assert not self.is_gemma3, (
                "Context parallel is not supported for Gemma3 models. "
                "Please refer to model-quirks documentation for more details."
            )
            assert not (tp_size > 1 and sequence_parallel_enabled), (
                "It's a known issue that context parallel can't be used together with sequence parallel in DTensor worker. "
                "Please either set cp_size = 1 or disable sequence parallel."
            )
            assert not self.is_vlm, (
                "Context parallel is yet not supported for VLM models. Please set cp_size = 1 to train VLM models."
            )

        self._build_device_mesh(dp_size, cp_size, tp_size)

        self._parallelize(sequence_parallel_enabled)

        print(f"[Rank {self.rank}] Loading state dict from rank 0...")
        set_model_state_dict(
            self.model,
            model_state_dict=cast(dict[str, Any], full_state_dict),
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

        # Handle tied word embeddings after loading the state dict
        is_tied_lm_head = hasattr(self.model, "lm_head") and getattr(
            getattr(self.model, "config", {}), "tie_word_embeddings", False
        )
        if is_tied_lm_head:
            # tie_weights() is a HF PreTrainedModel method, not on nn.Module;
            # the hasattr guard above makes this a deliberate dynamic call.
            cast(Any, self.model).tie_weights()

        # Manually broadcast buffers in a deterministic order
        buf_dict = dict(self.model.named_buffers())
        ordered_names = sorted(buf_dict.keys())
        for name in ordered_names:
            buf = buf_dict[name]
            torch.distributed.broadcast(to_local_if_dtensor(buf), src=0)

        if self.cpu_offload:
            self.model = self.move_to_device(self.model, "cpu")

        if init_reference_model:
            self.reference_model_state_dict = get_cpu_state_dict(
                self.model.state_dict().items(), pin_memory=True
            )

        if init_optimizer:
            optimizer_cls = get_class(self.cfg["optimizer"]["name"])
            self.optimizer = optimizer_cls(
                self.model.parameters(), **self.cfg["optimizer"]["kwargs"]
            )
        else:
            self.optimizer = None

        if "scheduler" in self.cfg and self.optimizer is not None:
            if isinstance(self.cfg["scheduler"], dict):
                scheduler_cls = get_class(cast(str, self.cfg["scheduler"]["name"]))
                self.scheduler = scheduler_cls(
                    self.optimizer, **self.cfg["scheduler"]["kwargs"]
                )
            else:
                schedulers = []
                milestones: list[int] = []
                for scheduler_cfg in self.cfg["scheduler"]:
                    if "name" in scheduler_cfg:
                        schedulers.append(
                            get_class(scheduler_cfg["name"])(
                                self.optimizer, **scheduler_cfg["kwargs"]
                            )
                        )
                    else:
                        assert "milestones" in scheduler_cfg, (
                            "unknown scheduler config: ",
                            scheduler_cfg,
                        )
                        milestones: list[int] = scheduler_cfg["milestones"]

                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers, milestones
                )

        elif self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )

        self._setup_moe_load_balancing()
        self._setup_router_replay()

        # Restore from checkpoint
        if weights_path:
            self.load_checkpoint(weights_path, optimizer_path)
        else:
            print(
                "No weights path provided. Starting from scratch (default policy init)"
            )

    def _setup_moe_load_balancing(self) -> None:
        """Seam: register the aux-loss-free expert-bias updater (MoE only).

        No-op for the dense v1 worker; the v2 MoE worker overrides this to
        register the optimizer ``step`` pre-hook when the model has
        load-balanced ``MoEBlock`` modules.
        """
        return

    def _setup_router_replay(self) -> None:
        """Resolve and validate MoE router-replay (#2908).

        Reads ``policy.router_replay.enabled`` and fails fast if it is set on a
        model without ``MoEBlock`` modules. The forward sites consume the flag
        via ``router_replay_context``; the default (disabled) path is inert.

        Replay binds the recorded routing to the right-padded ``[B, T, L, K]``
        token layout, so it is incompatible with the layouts that re-arrange the
        token/sequence axis: sequence packing (packed rows), dynamic batching
        (variable shapes), and context parallelism (seq-sharded buffers). The
        recorded routing would have to be transformed identically to stay
        aligned (HV-47). Fail fast at setup with a clear message rather than
        crash deep in the pack/shard machinery (the forward sites also refuse
        these combinations as defense in depth).
        """
        self._router_replay_enabled = resolve_router_replay_enabled(self.cfg)
        validate_router_replay_config(self._router_replay_enabled, self.model)
        if self._router_replay_enabled:
            cp_size = self.cfg["dtensor_cfg"]["context_parallel_size"]
            incompatible = []
            if self.cfg["sequence_packing"]["enabled"]:
                incompatible.append("policy.sequence_packing.enabled")
            if self.cfg["dynamic_batching"]["enabled"]:
                incompatible.append("policy.dynamic_batching.enabled")
            if cp_size > 1:
                incompatible.append("policy.dtensor_cfg.context_parallel_size>1")
            if incompatible:
                raise ValueError(
                    "policy.router_replay.enabled=true is not compatible with "
                    f"{', '.join(incompatible)}: router replay binds the recorded "
                    "routing to the unpacked right-padded token layout, which "
                    "those features re-arrange (HV-47). Disable router replay or "
                    "the listed feature(s)."
                )

    def _apply_moe_surgery(self, model: nn.Module) -> None:
        """Seam: swap HF routed-expert MLPs for native MoEBlocks (MoE only).

        No-op for the dense v1 worker — leaves ``model`` byte-identical. The v2
        worker overrides this to run ``apply_moe_surgery`` so the model becomes
        native-MoE before the device mesh / parallel plan are built. Called on
        both the rank-0 source model (so its broadcast state dict carries native
        param names) and the empty per-rank model (so the keys match).
        """
        return

    def _build_device_mesh(
        self, dp_size: int, cp_size: int, tp_size: int
    ) -> None:
        """Build the ``(dp, cp, tp)`` device mesh and store the mesh handles.

        Sets ``self.device_mesh``/``dp_mesh``/``tp_mesh``/``cp_mesh``/``dp_cp_mesh``
        and ``self.dp_size``/``tp_size``/``cp_size``. The v2 worker overrides this
        to additionally build the MoE sparse (expert-parallel) mesh.
        """
        # torch==2.8 uses LOCAL_RANK to set the device (see PyTorch source).
        # CUDA_VISIBLE_DEVICES is set to only 1 GPU, so temporarily set LOCAL_RANK to 0.
        prev_local_rank = os.environ["LOCAL_RANK"]
        os.environ["LOCAL_RANK"] = "0"

        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", (dp_size, cp_size, tp_size), mesh_dim_names=("dp", "cp", "tp")
        )
        os.environ["LOCAL_RANK"] = prev_local_rank

        self.dp_cp_mesh = device_mesh[("dp", "cp")]._flatten(mesh_dim_name="dp_cp")

        self.dp_mesh, self.tp_mesh, self.cp_mesh = (
            device_mesh["dp"],
            device_mesh["tp"],
            device_mesh["cp"],
        )
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.device_mesh = device_mesh

    def _parallelize(self, sequence_parallel_enabled: bool) -> None:
        """Apply tensor/data parallelism (FSDP2 + TP) to ``self.model``.

        Wraps ``self.model`` in place. The v2 worker overrides this to apply
        MoE expert-parallel sharding to the routed experts before the dense
        FSDP/TP plan covers the remaining (attention / shared) parameters.
        """
        # fully_shard (inside _parallelize_model) is typed to return FSDPModule,
        # which the stubs don't model as an nn.Module subclass even though the
        # returned object is the same module at runtime. Cast so self.model stays
        # nn.Module — otherwise FSDPModule poisons every downstream model use.
        self.model = cast(nn.Module, _parallelize_model(
            self.model,
            self.dp_cp_mesh,
            self.tp_mesh,
            param_dtype=self.dtype,
            sequence_parallel=sequence_parallel_enabled,
            cpu_offload=self.cpu_offload,
            activation_checkpointing=self.cfg["dtensor_cfg"][
                "activation_checkpointing"
            ],
            custom_parallel_plan=self.cfg["dtensor_cfg"]["custom_parallel_plan"],
        ))

        # Attach CP attention-mask hooks (strip mask, set is_causal=True).
        if self.cp_size > 1:
            _attach_context_parallel_hooks(self.model)

    # based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
    @staticmethod
    def create_context_parallel_ctx(
        cp_mesh: torch.distributed.device_mesh.DeviceMesh,
        cp_buffers: list[torch.Tensor],
        cp_seq_dims: list[int],
        cp_no_restore_buffers: Set[torch.Tensor],
        cp_rotate_method: Optional[str] = None,
    ):
        """Create a context parallel context."""
        if cp_rotate_method is not None:
            set_rotate_method(cp_rotate_method)

        return context_parallel(
            cp_mesh,
            buffers=cp_buffers,
            buffer_seq_dims=cp_seq_dims,
            no_restore_buffers=cp_no_restore_buffers,
        )

    def _apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sampling_params is not None and self.sampling_params.temperature != 1.0:
            logits.div_(self.sampling_params.temperature)
        return logits

    def _apply_top_k_top_p_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k and top-p filtering to the logits locally when TP is disabled."""
        if self.sampling_params is not None and need_top_k_or_top_p_filtering(self.sampling_params):
            logits, _ = apply_top_k_top_p(
                logits,
                top_k=self.sampling_params.top_k,
                top_p=self.sampling_params.top_p,
            )
        return logits

    @staticmethod
    @contextlib.contextmanager
    def train_context(cp_context: Optional[AbstractContextManager[Any]] = None):
        with contextlib.ExitStack() as stack:
            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                        ]
                    )
                )
                stack.enter_context(cp_context)
            yield

    @wrap_with_nvtx_name("dtensor_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,  # type: ignore[valid-type]
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        if gbs is None:
            gbs = cast(int, self.cfg["train_global_batch_size"])
        if mbs is None:
            mbs = cast(int, self.cfg["train_micro_batch_size"])
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=self.dp_mesh.get_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                )

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            self.model.train()

        with ctx:
            data.to("cuda")

            losses = []
            all_mb_metrics = []
            grad_norm: Optional[float | torch.Tensor] = None
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if "token_mask" not in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(to_reduce, group=self.dp_mesh.get_group())
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and getattr(loss_fn, "loss_type") == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                mb_losses = []
                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                dummy_iterator = iter([])
                if self.cfg["dynamic_batching"]["enabled"]:
                    mb_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    iterator_len = batch.get_microbatch_iterator_dynamic_shapes_len()
                elif self.enable_seq_packing:
                    mb_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    iterator_len, max_seqlen = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    max_batch_ct = torch.tensor([iterator_len], device="cuda")
                    torch.distributed.all_reduce(
                        max_batch_ct, op=torch.distributed.ReduceOp.MAX
                    )

                    # Sequence packing can end up with unevenly distributed batch counts across DP ranks.
                    # We add dummy batches to the end of the iterator to make the batch counts equal.
                    dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                    dummy_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    dummy_iterator = itertools.islice(
                        itertools.cycle(dummy_iterator), dummy_batch_ct
                    )
                else:
                    mb_iterator = batch.make_microbatch_iterator(mbs)
                    iterator_len = batch.size // mbs

                empty_cache_steps = self.cfg.get("dtensor_cfg", {}).get(
                    "clear_cache_every_n_steps"
                )
                if empty_cache_steps:
                    warnings.warn(
                        f"Emptying cache every {empty_cache_steps} microbatches, doing so unnecessarily would incur a large performance overhead."
                    )

                for mb_idx, mb in enumerate(
                    itertools.chain(mb_iterator, dummy_iterator)
                ):
                    if empty_cache_steps and mb_idx % empty_cache_steps == 0:
                        torch.cuda.empty_cache()

                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        if self.enable_seq_packing:
                            input_ids = mb.get("input_ids").cuda()
                            input_ids, position_ids, _ = pack_sequences(
                                input_ids=input_ids,
                                input_lengths=mb["input_lengths"],
                                packed_sequence_size=[len(mb["input_lengths"])],
                                padding_value=cast(int, self.tokenizer.eos_token_id),
                                return_attention_mask=False,
                                min_seq_len=self.cfg["sequence_packing"][
                                    "train_mb_tokens"
                                ],
                            )
                            seq_len = input_ids.shape[1]
                            attention_mask = None
                            flash_attn_kwargs = get_flash_attention_kwargs(
                                input_lengths=mb["input_lengths"],
                            )

                        else:
                            input_ids = mb.get("input_ids").cuda()
                            batch_size, seq_len = input_ids.shape

                            # When sequence parallelism is enabled, pass None instead of a
                            # full-size attention mask to avoid shape mismatches in SDPA/eager.
                            if (
                                self.cfg["dtensor_cfg"]["sequence_parallel"]
                                or self.cp_size > 1
                            ):
                                attention_mask = None
                            else:
                                attention_mask = torch.ones(
                                    (batch_size, seq_len),
                                    dtype=torch.bool,
                                    device=input_ids.device,
                                )
                            position_ids = torch.arange(
                                seq_len, device=input_ids.device
                            ).repeat(batch_size, 1)
                            flash_attn_kwargs = {}

                        vlm_kwargs = mb.get_multimodal_dict(
                            as_tensors=True, device=input_ids.device
                        )
                        if len(vlm_kwargs) > 0:
                            position_ids = None
                            assert not self.cfg["dtensor_cfg"]["sequence_parallel"], (
                                "Sequence parallel is not supported with multimodal since there's an issue when you do not pass position_ids."
                            )

                    context_parallel_ctx = None
                    seq_index = None
                    cp_buffers = []
                    if self.cp_size > 1:
                        assert len(vlm_kwargs) == 0, (
                            f"multimodal kwargs={vlm_kwargs} are not supported for context parallel"
                        )
                        seq_index = torch.arange(
                            seq_len, device=input_ids.device
                        ).repeat(1, 1)
                        cp_buffers = (
                            [input_ids, position_ids, seq_index]
                            if self.cp_size > 1
                            else []
                        )

                        context_parallel_ctx = self.create_context_parallel_ctx(
                            cp_mesh=self.cp_mesh,
                            cp_buffers=cp_buffers,
                            cp_seq_dims=[sequence_dim] * len(cp_buffers),
                            cp_no_restore_buffers=set(cp_buffers),
                        )

                    with DTensorPolicyWorkerImpl.train_context(context_parallel_ctx):
                        with torch.autocast(device_type="cuda", dtype=self.dtype):
                            model_args = dict(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                use_cache=False,
                                flash_attn_kwargs=flash_attn_kwargs,
                                **vlm_kwargs,
                            )

                            # Gemma3 requires token_type_ids when model.training=True.
                            if self.is_gemma3 and "token_type_ids" not in model_args:
                                model_args["token_type_ids"] = torch.zeros_like(
                                    input_ids
                                )

                            if self._is_reward_model:
                                assert not flash_attn_kwargs
                                del model_args["flash_attn_kwargs"]
                            if len(vlm_kwargs) > 0:
                                del model_args["flash_attn_kwargs"]

                            # MoE router-replay: force the recorded generation
                            # routing for this microbatch (inert when disabled or
                            # no routed_experts column).
                            with router_replay_context(
                                self.model,
                                mb.get("routed_experts", None),
                                enabled=self._router_replay_enabled,
                                seq_packing=self.enable_seq_packing,
                                context_parallel=self.cp_size > 1,
                            ):
                                outputs = self.model(**model_args)

                        if not hasattr(outputs, "logits"):
                            logits = cast(Any, self.model).lm_head(outputs.last_hidden_state)
                        else:
                            logits = outputs.logits
                        del outputs

                        logits = self._apply_temperature_scaling(logits)

                        if self.cp_size > 1:
                            seq_index_dtensor = (
                                DTensor.from_local(
                                    cast(torch.Tensor, seq_index),
                                    device_mesh=self.cp_mesh,
                                    placements=[Shard(1)],
                                )
                                .full_tensor()
                                .squeeze(0)
                            )

                            mb["seq_index"] = seq_index_dtensor

                            for tensor_name in mb:
                                current_tensor = mb[tensor_name]
                                for buffer in cp_buffers:
                                    if current_tensor is buffer:
                                        assert type(current_tensor) == torch.Tensor, (
                                            f"tensor {tensor_name} is not a tensor"
                                        )
                                        mb[tensor_name] = DTensor.from_local(
                                            cast(torch.Tensor, current_tensor),
                                            device_mesh=self.cp_mesh,
                                            placements=[Shard(sequence_dim)],
                                        )
                                        break

                            if isinstance(logits, DTensor):
                                assert (
                                    logits.device_mesh.ndim == 1
                                    and logits.device_mesh.mesh_dim_names is not None
                                    and logits.device_mesh.mesh_dim_names[0] == "tp"
                                ), "logits must be tp sharded"
                                logits = DTensor.from_local(
                                    logits.to_local(),
                                    device_mesh=getattr(self, "device_mesh")[("cp", "tp")],
                                    placements=[Shard(sequence_dim), Shard(-1)],
                                )
                            else:
                                logits = DTensor.from_local(
                                    logits,
                                    device_mesh=getattr(self, "device_mesh")[("cp", "tp")],
                                    placements=[Shard(sequence_dim), Shard(-1)],
                                )

                        prepare_loss_input_wrapped = partial(
                            prepare_loss_input, sampling_params=self.sampling_params
                        )
                        if self.enable_seq_packing:
                            # In this branch flash_attn_kwargs is a FlashAttentionKwargs
                            # (set in the packing path above), never the {} dict default.
                            assert not isinstance(flash_attn_kwargs, dict)
                            loss_fn_ = SequencePackingLossWrapper(
                                loss_fn=loss_fn,
                                prepare_fn=prepare_loss_input_wrapped,
                                cu_seqlens_q=flash_attn_kwargs.cu_seqlens_q,
                                cu_seqlens_q_padded=flash_attn_kwargs.cu_seqlens_q,
                            )
                            loss, loss_metrics = loss_fn_(
                                logits,
                                mb,
                                global_valid_seqs,
                                global_valid_toks,
                            )
                        else:
                            loss_input, mb = prepare_loss_input_wrapped(
                                logits, mb, loss_fn
                            )
                            loss, loss_metrics = loss_fn(
                                data=mb,
                                global_valid_seqs=global_valid_seqs,
                                global_valid_toks=global_valid_toks,
                                **loss_input,
                            )
                        del logits

                        # skip the update for dummy batches
                        if mb_idx < iterator_len:
                            for k in loss_metrics.keys():
                                if "_min" in k or "_max" in k:
                                    continue
                                loss_metrics[k] /= num_global_batches
                            num_valid_samples = loss_metrics["num_valid_samples"]
                            if self.optimizer is not None:
                                loss_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                            loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                            loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        else:
                            loss *= 0
                            num_valid_samples = 0

                        if not eval_mode:
                            # when FSDP reduces the gradients over the DP dim, they're automatically averaged
                            # but we want to sum them so we cancel out the average here
                            loss *= self.dp_size * self.cp_size
                            loss.backward()

                    if num_valid_samples > 0:
                        mb_losses.append(loss.item())
                        all_mb_metrics.append(loss_metrics)

                if not eval_mode and self.optimizer is not None:
                    with torch.no_grad():
                        grad_norm = get_grad_norm(
                            cast("list[torch.Tensor]", self.model.parameters()),
                            dp_cp_group=self.dp_cp_mesh.get_group(),
                            tp_group=self.tp_mesh.get_group(),
                            dtype=torch.float32,
                        )
                        if self.max_grad_norm is not None:
                            clip_grad_by_total_norm_(
                                cast("list[torch.Tensor]", self.model.parameters()),
                                max_grad_norm=self.max_grad_norm,
                                total_norm=cast(float, grad_norm),
                            )
                        grad_norm = torch.tensor([grad_norm])

                    self.optimizer.step()

                losses.append(torch.tensor(mb_losses).sum().item())

            # release gradient memory before rollouts
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            if not eval_mode and getattr(self, "scheduler", None) is not None:
                self.scheduler.step()
            torch.cuda.empty_cache()

            with torch.no_grad():
                global_loss = torch.tensor(losses, device="cuda")
                torch.distributed.all_reduce(
                    global_loss, group=self.dp_mesh.get_group()
                )
            mb_metrics = defaultdict(list)
            for m in all_mb_metrics:
                for k, v in m.items():
                    mb_metrics[k].append(v)

            metrics = {
                "global_loss": global_loss.cpu(),
                "grad_norm": grad_norm,
                "rank": torch.distributed.get_rank(),
                "gpu_name": torch.cuda.get_device_name(),
                "model_dtype": self.dtype,
                "all_mb_metrics": dict(mb_metrics),
            }

            return metrics

    @wrap_with_nvtx_name("dtensor_policy_worker/get_logprobs")
    def get_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.

        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )
        logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)

        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                )

        all_log_probs = []
        self.model.eval()

        with unshard_fsdp2_model(self.model), torch.no_grad():
            data.to("cuda")
            dummy_iterator = iter([])
            if self.cfg["dynamic_batching"]["enabled"]:
                mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
                iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            elif self.enable_seq_packing:
                mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
                iterator_len, max_seqlen = (
                    data.get_microbatch_iterator_for_packable_sequences_len()
                )
                max_batch_ct = torch.tensor([iterator_len], device="cuda")
                torch.distributed.all_reduce(
                    max_batch_ct, op=torch.distributed.ReduceOp.MAX
                )
                dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                dummy_iterator = data.make_microbatch_iterator_for_packable_sequences()
                dummy_iterator = itertools.islice(
                    itertools.cycle(dummy_iterator), dummy_batch_ct
                )
            else:
                mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
                iterator_len = data.size // logprob_batch_size

            step = 0
            for batch_idx, lp_batch in enumerate(
                itertools.chain(mb_iterator, dummy_iterator)
            ):
                step += 1
                input_ids = lp_batch.get("input_ids").cuda()
                input_lengths = lp_batch.get("input_lengths")
                vlm_kwargs = lp_batch.get_multimodal_dict(
                    as_tensors=True, device=input_ids.device
                )

                batch_size, seq_len = input_ids.shape
                seq_index = None
                post_attention_mask = None
                if self.enable_seq_packing:
                    assert len(vlm_kwargs) == 0, (
                        "multimodal kwargs are not supported for sequence packing"
                    )
                    input_ids, position_ids, _ = pack_sequences(
                        input_ids=input_ids,
                        input_lengths=input_lengths,
                        packed_sequence_size=[batch_size],
                        padding_value=cast(int, self.tokenizer.eos_token_id),
                        return_attention_mask=False,
                    )
                    seq_len = input_ids.shape[1]
                    attention_mask = None
                    flash_attn_kwargs = get_flash_attention_kwargs(
                        input_lengths=input_lengths,
                    )
                else:
                    post_attention_mask = torch.zeros(
                        (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
                    )
                    for i, length in enumerate(input_lengths):
                        post_attention_mask[i, :length] = 1

                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).repeat(batch_size, 1)
                    flash_attn_kwargs = {}

                    if self.cp_size > 1:
                        attention_mask = None
                    else:
                        attention_mask = torch.ones(
                            (batch_size, seq_len),
                            dtype=torch.bool,
                            device=input_ids.device,
                        )

                if len(vlm_kwargs) > 0:
                    position_ids = None

                context_parallel_ctx = None
                if self.cp_size > 1:
                    assert len(vlm_kwargs) == 0, (
                        "multimodal kwargs are not supported for context parallel"
                    )
                    seq_index = torch.arange(seq_len, device=input_ids.device).repeat(
                        1, 1
                    )
                    cp_buffers = [input_ids, position_ids, seq_index]

                    context_parallel_ctx = self.create_context_parallel_ctx(
                        cp_mesh=self.cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=[sequence_dim] * len(cp_buffers),
                        cp_no_restore_buffers=set(cp_buffers),
                    )

                with DTensorPolicyWorkerImpl.train_context(context_parallel_ctx):
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        model_args = dict(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                            flash_attn_kwargs=flash_attn_kwargs,
                            **vlm_kwargs,
                        )
                        if len(vlm_kwargs) > 0:
                            del model_args["flash_attn_kwargs"]

                        # MoE router-replay: the prev-logprob recompute is the
                        # primary consumer — re-routing tokens differently here
                        # than at generation is what inflates the train/gen
                        # logprob error. Inert when disabled / no routing column.
                        with router_replay_context(
                            self.model,
                            lp_batch.get("routed_experts", None),
                            enabled=self._router_replay_enabled,
                            seq_packing=self.enable_seq_packing,
                            context_parallel=self.cp_size > 1,
                        ):
                            outputs = self.model(**model_args)

                    logits = outputs.logits
                    logits = self._apply_temperature_scaling(logits)

                    if self.cp_size > 1:
                        seq_index_tensor = (
                            DTensor.from_local(
                                cast(torch.Tensor, seq_index),
                                device_mesh=self.cp_mesh,
                                placements=[Shard(1)],
                            )
                            .full_tensor()
                            .squeeze(0)
                        )

                        input_ids_dtensor = DTensor.from_local(
                            input_ids,
                            device_mesh=self.cp_mesh,
                            placements=[Shard(sequence_dim)],
                        )

                        if isinstance(logits, DTensor):
                            assert (
                                logits.device_mesh.ndim == 1
                                and logits.device_mesh.mesh_dim_names is not None
                                and logits.device_mesh.mesh_dim_names[0] == "tp"
                            ), "logits must be tp sharded"
                            logits = DTensor.from_local(
                                logits.to_local(),
                                device_mesh=cast(Any, self.device_mesh)[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )
                        else:
                            logits = DTensor.from_local(
                                logits,
                                device_mesh=cast(Any, self.device_mesh)[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )

                        token_logprobs = get_logprobs_from_vocab_parallel_logits(
                            logits,
                            input_ids_dtensor,
                            seq_index_tensor,
                            chunk_size=logprob_chunk_size,
                            sampling_params=self.sampling_params,
                        )

                        assert token_logprobs.shape[1] == seq_len - 1
                    else:
                        if isinstance(logits, DTensor):
                            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                                logits,
                                input_ids,
                                chunk_size=logprob_chunk_size,
                                sampling_params=self.sampling_params,
                            )
                        else:
                            if logprob_chunk_size is not None:
                                logits_seq_len = int(logits.shape[1])
                                num_chunks = (
                                    logits_seq_len + logprob_chunk_size - 1
                                ) // logprob_chunk_size
                                chunked_log_probs = []
                                for chunk_idx in range(num_chunks):
                                    chunk_start = chunk_idx * logprob_chunk_size
                                    chunk_end = min(
                                        logits_seq_len,
                                        (chunk_idx + 1) * logprob_chunk_size,
                                    )
                                    chunk_logits = logits[
                                        :, chunk_start:chunk_end, :
                                    ].to(torch.float32)
                                    chunk_logits = self._apply_top_k_top_p_filtering(
                                        chunk_logits
                                    )
                                    log_probs = torch.nn.functional.log_softmax(
                                        chunk_logits, dim=-1
                                    )
                                    chunked_log_probs.append(log_probs)
                                log_probs = torch.cat(chunked_log_probs, dim=1)
                                del chunked_log_probs
                            else:
                                logits = logits.to(torch.float32)
                                logits = self._apply_top_k_top_p_filtering(logits)
                                log_probs = torch.nn.functional.log_softmax(
                                    logits, dim=-1
                                )
                            next_tokens = input_ids[:, 1:]
                            log_probs = log_probs[:, :-1]
                            token_logprobs = log_probs.gather(
                                dim=-1, index=next_tokens.unsqueeze(-1)
                            ).squeeze(-1)
                            del log_probs

                del outputs, logits

                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )

                if batch_idx >= iterator_len:
                    continue

                if not self.enable_seq_packing:
                    token_logprobs = token_logprobs * cast(torch.Tensor, post_attention_mask)
                else:
                    unpacked_logprobs = torch.zeros(
                        (batch_size, seq_dim_size),
                        dtype=token_logprobs.dtype,
                        device=token_logprobs.device,
                    )
                    assert not isinstance(flash_attn_kwargs, dict)
                    cu_seqlens = flash_attn_kwargs.cu_seqlens_q
                    for i in range(batch_size):
                        start = cu_seqlens[i].item() + 1
                        end = cu_seqlens[i + 1].item()
                        seq_len_actual = input_lengths[i].item()
                        unpacked_logprobs[i, 1:seq_len_actual] = token_logprobs[
                            0, start:end
                        ]
                    token_logprobs = unpacked_logprobs

                all_log_probs.append(token_logprobs)

        return_data = BatchedDataDict[LogprobOutputSpec]()

        all_log_probs_padded = []
        for lp in all_log_probs:
            padding_needed = seq_dim_size - lp.shape[1]
            if padding_needed > 0:
                lp = torch.nn.functional.pad(
                    lp, (0, padding_needed), mode="constant", value=0.0
                )
            all_log_probs_padded.append(lp)
        token_logprobs = torch.cat(all_log_probs_padded, dim=0)

        if need_top_k_or_top_p_filtering(self.sampling_params):
            mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
            token_logprobs = mask_out_neg_inf_logprobs(
                token_logprobs, mask, "prev_logprobs"
            )

        return_data["logprobs"] = token_logprobs.cpu()
        return return_data

    @wrap_with_nvtx_name("dtensor_policy_worker/score")
    def score(self, data: BatchedDataDict) -> BatchedDataDict[ScoreOutputSpec]:
        global_batch_size = min(self.cfg["batch_size"], data.size)

        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                )
        self.model.eval()

        with unshard_fsdp2_model(self.model), torch.no_grad():
            data.to("cuda")
            dummy_iterator = iter([])
            if self.cfg["dynamic_batching"]["enabled"]:
                mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
                iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            elif self.enable_seq_packing:
                mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
                iterator_len, max_seqlen = (
                    data.get_microbatch_iterator_for_packable_sequences_len()
                )
                max_batch_ct = torch.tensor([iterator_len], device="cuda")
                torch.distributed.all_reduce(
                    max_batch_ct, op=torch.distributed.ReduceOp.MAX
                )
                dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                dummy_iterator = data.make_microbatch_iterator_for_packable_sequences()
                dummy_iterator = itertools.islice(
                    itertools.cycle(dummy_iterator), dummy_batch_ct
                )
            else:
                mb_iterator = data.make_microbatch_iterator(global_batch_size)
                iterator_len = data.size // global_batch_size

            step = 0
            all_rm_scores = []
            for batch_idx, generate_batch in enumerate(
                itertools.chain(mb_iterator, dummy_iterator)
            ):
                step += 1
                input_ids = generate_batch.get("input_ids").cuda()
                input_lengths = generate_batch.get("input_lengths")
                batch_size, seq_len = input_ids.shape
                if self.enable_seq_packing:
                    input_ids, position_ids, _ = pack_sequences(
                        input_ids=input_ids,
                        input_lengths=input_lengths,
                        packed_sequence_size=[batch_size],
                        padding_value=cast(int, self.tokenizer.eos_token_id),
                        return_attention_mask=False,
                    )
                    seq_len = input_ids.shape[1]
                    attention_mask = None
                    flash_attn_kwargs = get_flash_attention_kwargs(
                        input_lengths=input_lengths,
                    )
                else:
                    post_attention_mask = torch.zeros(
                        (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
                    )
                    for i, length in enumerate(input_lengths):
                        post_attention_mask[i, :length] = 1
                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).repeat(batch_size, 1)
                    attention_mask = torch.ones(
                        (batch_size, seq_len),
                        dtype=torch.bool,
                        device=input_ids.device,
                    )

                context_parallel_ctx = None
                if self.cp_size > 1:
                    seq_index = torch.arange(seq_len, device=input_ids.device).repeat(
                        1, 1
                    )
                    cp_buffers = [input_ids, position_ids, seq_index]
                    context_parallel_ctx = self.create_context_parallel_ctx(
                        cp_mesh=self.cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=[sequence_dim] * len(cp_buffers),
                        cp_no_restore_buffers=set(cp_buffers),
                    )
                with DTensorPolicyWorkerImpl.train_context(context_parallel_ctx):
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        model_args = dict(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                        )
                        outputs = self.model(**model_args)

                    if not hasattr(outputs, "logits"):
                        logits = cast(Any, self.model).lm_head(outputs.last_hidden_state)
                    else:
                        logits = outputs.logits
                    logits = self._apply_temperature_scaling(logits)
                if isinstance(logits, DTensor):
                    logits = logits.to(torch.float32)
                else:
                    logits = outputs.logits.to(torch.float32)

                rm_scores = to_local_if_dtensor(logits)
                rm_scores = rm_scores.squeeze(-1)
                all_rm_scores.append(rm_scores)

        all_rm_scores = torch.cat(all_rm_scores, dim=0)
        all_rm_scores = all_rm_scores.squeeze(-1).cpu()
        return_data = BatchedDataDict[ScoreOutputSpec](
            {
                "scores": all_rm_scores,
            }
        )
        return return_data

    @wrap_with_nvtx_name("dtensor_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        data: BatchedDataDict[Any],
        k: int,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[Any]:
        """Return per-position top-k logits and corresponding global indices.

        Notes:
        - Return shapes are [B, S, k].
        - Computes top-k over the full sequence (no trimming of the last position).
        - If alignment with next-token targets is required, the caller should handle it.
        - If logits are TP-sharded DTensor, performs distributed global top-k across TP.
        - Supports context parallelism with proper CP gather.
        - Otherwise, computes local top-k on full-vocab tensor.
        """
        topk_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]

        out_topk_vals = []
        out_topk_idx = []
        self.model.eval()

        with torch.no_grad():
            data.to("cuda")
            dummy_iterator = iter([])
            if self.cfg["dynamic_batching"]["enabled"]:
                mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
                iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            elif self.enable_seq_packing:
                mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
                iterator_len, max_seqlen = (
                    data.get_microbatch_iterator_for_packable_sequences_len()
                )
                max_batch_ct = torch.tensor([iterator_len], device="cuda")
                torch.distributed.all_reduce(
                    max_batch_ct, op=torch.distributed.ReduceOp.MAX
                )
                dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                dummy_iterator = data.make_microbatch_iterator_for_packable_sequences()
                dummy_iterator = itertools.islice(
                    itertools.cycle(dummy_iterator), dummy_batch_ct
                )
            else:
                mb_iterator = data.make_microbatch_iterator(topk_batch_size)
                iterator_len = data.size // topk_batch_size

            for batch_idx, lp_batch in enumerate(
                itertools.chain(mb_iterator, dummy_iterator)
            ):
                input_ids = lp_batch.get("input_ids").cuda()
                input_lengths = lp_batch.get("input_lengths")
                vlm_kwargs = lp_batch.get_multimodal_dict(
                    as_tensors=True, device=input_ids.device
                )
                batch_size, seq_len = input_ids.shape
                seq_index = None

                original_batch_size = batch_size
                original_seq_len = seq_len

                if self.enable_seq_packing:
                    assert len(vlm_kwargs) == 0, (
                        "multimodal kwargs are not supported for sequence packing"
                    )
                    input_ids, position_ids, _ = pack_sequences(
                        input_ids=input_ids,
                        input_lengths=input_lengths,
                        packed_sequence_size=[batch_size],
                        padding_value=cast(int, self.tokenizer.eos_token_id),
                        return_attention_mask=False,
                    )
                    seq_len = input_ids.shape[1]
                    attention_mask = None
                    flash_attn_kwargs = get_flash_attention_kwargs(
                        input_lengths=input_lengths,
                    )
                else:
                    attention_mask = torch.zeros(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )
                    for i, length in enumerate(input_lengths):
                        attention_mask[i, :length] = 1

                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).repeat(batch_size, 1)

                    flash_attn_kwargs = {}

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    attention_mask_input_all_ones = torch.ones(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )

                if len(vlm_kwargs) > 0:
                    position_ids = None

                context_parallel_ctx = None
                if self.cp_size > 1:
                    assert len(vlm_kwargs) == 0, (
                        "multimodal kwargs are not supported for context parallel"
                    )
                    seq_index = torch.arange(seq_len, device=input_ids.device).repeat(
                        1, 1
                    )
                    cp_buffers = [input_ids, position_ids, seq_index]
                    context_parallel_ctx = self.create_context_parallel_ctx(
                        cp_mesh=self.cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=[sequence_dim] * len(cp_buffers),
                        cp_no_restore_buffers=set(cp_buffers),
                    )

                with DTensorPolicyWorkerImpl.train_context(context_parallel_ctx):
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        model_args = dict(
                            input_ids=input_ids,
                            attention_mask=attention_mask_input_all_ones,
                            position_ids=position_ids,
                            use_cache=False,
                            flash_attn_kwargs=flash_attn_kwargs,
                            **vlm_kwargs,
                        )
                        if len(vlm_kwargs) > 0:
                            del model_args["flash_attn_kwargs"]

                        outputs = self.model(**model_args)

                    if not hasattr(outputs, "logits"):
                        logits = cast(Any, self.model).lm_head(outputs.last_hidden_state)
                    else:
                        logits = outputs.logits
                    del outputs

                    logits = self._apply_temperature_scaling(logits)

                    if self.cp_size > 1:
                        if isinstance(logits, DTensor):
                            assert (
                                logits.device_mesh.ndim == 1
                                and logits.device_mesh.mesh_dim_names is not None
                                and logits.device_mesh.mesh_dim_names[0] == "tp"
                            ), "logits must be tp sharded"
                            logits = DTensor.from_local(
                                logits.to_local(),
                                device_mesh=cast(Any, self.device_mesh)[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )
                        else:
                            logits = DTensor.from_local(
                                logits,
                                device_mesh=cast(Any, self.device_mesh)[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )
                        logits = allgather_cp_sharded_tensor(
                            logits, self.cp_mesh, sequence_dim
                        )

                    if isinstance(logits, DTensor):
                        tp_rank = self.tp_mesh.get_local_rank()
                        vocab_size = logits.shape[-1]
                        vocab_start_index = tp_rank * vocab_size
                        vocab_end_index = vocab_start_index + vocab_size
                        vals, idx = distributed_vocab_topk(
                            vocab_parallel_logits=logits.to_local().to(torch.float32),
                            k=k,
                            tp_group=self.tp_mesh.get_group(),
                            vocab_start_index=vocab_start_index,
                            vocab_end_index=vocab_end_index,
                        )
                    else:
                        full_logits = logits.to(torch.float32)
                        vals, idx = torch.topk(full_logits, k=k, dim=-1)

                if self.enable_seq_packing:
                    unpacked_vals = torch.zeros(
                        (original_batch_size, original_seq_len, k),
                        dtype=vals.dtype,
                        device=vals.device,
                    )
                    unpacked_idx = torch.zeros(
                        (original_batch_size, original_seq_len, k),
                        dtype=idx.dtype,
                        device=idx.device,
                    )
                    assert not isinstance(flash_attn_kwargs, dict)
                    cu_seqlens = flash_attn_kwargs.cu_seqlens_q
                    for i in range(original_batch_size):
                        start = cu_seqlens[i].item()
                        end = cu_seqlens[i + 1].item()
                        seq_len_actual = input_lengths[i].item()
                        unpacked_vals[i, :seq_len_actual, :] = vals[0, start:end, :]
                        unpacked_idx[i, :seq_len_actual, :] = idx[0, start:end, :]
                    vals = unpacked_vals
                    idx = unpacked_idx
                    batch_size = original_batch_size
                    seq_len = original_seq_len

                out_topk_vals.append(vals.cpu())
                out_topk_idx.append(idx.cpu())

        ret = BatchedDataDict[Any]()
        all_topk_vals_padded = []
        all_topk_idx_padded = []
        target_seq_len = seq_dim_size
        for vals, idx in zip(out_topk_vals, out_topk_idx):
            pad_needed = target_seq_len - vals.shape[1]
            if pad_needed > 0:
                vals = torch.nn.functional.pad(
                    vals, (0, 0, 0, pad_needed, 0, 0), mode="constant", value=0.0
                )
                idx = torch.nn.functional.pad(
                    idx, (0, 0, 0, pad_needed, 0, 0), mode="constant", value=0
                )
            all_topk_vals_padded.append(vals)
            all_topk_idx_padded.append(idx)

        ret["topk_logits"] = (
            torch.cat(all_topk_vals_padded, dim=0)
            if len(all_topk_vals_padded) > 1
            else all_topk_vals_padded[0]
        ).cpu()
        ret["topk_indices"] = (
            torch.cat(all_topk_idx_padded, dim=0)
            if len(all_topk_idx_padded) > 1
            else all_topk_idx_padded[0]
        ).cpu()
        return ret

    @contextmanager
    def use_reference_model(self) -> Generator[None, None, None]:
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references.
                  Also disables top-k/top-p filtering since the reference policy's distribution
                  is different from the current policy, making filtered logprobs incompatible.
        On exit: Restores original references and re-flips cuda/cpu, restores sampling_params.
        """
        with torch.no_grad():
            curr_state_dict = get_cpu_state_dict(
                self.model.state_dict().items(), pin_memory=True
            )

            for k, v in self.model.state_dict().items():
                val = to_local_if_dtensor(v)
                val.copy_(self.reference_model_state_dict[k])

            # Temporarily disable top-k/top-p filtering for reference policy logprobs.
            # The reference policy has different weights, so its top-k/top-p set is
            # inherently different from the current policy. Using filtered logprobs
            # would cause -inf mismatches that cannot be resolved by masking.
            # Note: We keep temperature scaling since it was applied to prev_logprobs.
            saved_sampling_params = self.sampling_params
            if saved_sampling_params is not None:
                self.sampling_params = TrainingSamplingParams(
                    top_k=None,
                    top_p=1.0,
                    temperature=saved_sampling_params.temperature,
                )
            else:
                self.sampling_params = None

            yield

            self.sampling_params = saved_sampling_params

            for k, v in self.model.state_dict().items():
                val = to_local_if_dtensor(v)
                val.copy_(curr_state_dict[k])

    def _add_noise_to_weights(self) -> None:
        """Add small Gaussian noise to the weights of the model. Used for testing purposes only."""
        noise_std = 0.01
        for p in self.model.parameters():
            if p.requires_grad:
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)
        torch.cuda.synchronize()

    def return_state_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.model.state_dict())

    def return_model_config(self) -> dict[str, Any]:
        """Return the model configuration as a dictionary."""
        return cast(dict[str, Any], self.model.config)

    @torch.no_grad()
    def _iter_refit_state_dict(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Named tensors streamed during refit.

        The single source the tensor-streaming transports
        (``broadcast_weights_for_collective`` / ``stream_weights_via_http``)
        iterate. The v2 worker overrides this to expand native GroupedExperts
        params into per-expert HF-named tensors (EP-layout reshard).
        """
        return self.model.state_dict().items()

    def prepare_refit_info(self, state_dict_info: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        """Prepare state dict metadata for weight refitting."""
        result: dict[str, Any] = {}
        for name, tensor in self.model.state_dict().items():
            result[name] = (tensor.shape, self.dtype)
        return result

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """Placeholder for FP8 Q/K/V scale calibration, not implemented for DTensorPolicyWorker."""
        raise NotImplementedError(
            "calibrate_qkv_fp8_scales is not implemented for DTensorPolicyWorker"
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self,
        buffer_size_bytes: int = 0,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        raise NotImplementedError(
            "stream_weights_via_ipc_zmq requires ipc_utils which is not yet implemented. "
            "Use broadcast_weights_for_collective for NCCL-based weight sync instead."
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(  # type: ignore[override]
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        if kv_scales is not None:
            raise NotImplementedError(
                "FP8 kvcache is not currently supported for DTensor path."
            )

        if self.cpu_offload:
            print(
                "[WARNING]: Unless you are lacking of memory, it is not recommended to enable cpu_offload when "
                "using non-colocated generation since it will have an extra onload and offload at refit stage."
            )
            self.model = self.move_to_cuda(self.model)

        def _dtensor_post_iter_func(tensor, dtype):
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            tensor = tensor.to(dtype, non_blocking=True)
            return tensor

        dtensor_post_iter_func = lambda x: _dtensor_post_iter_func(x[1], self.dtype)

        packed_broadcast_producer(
            iterator=iter(self._iter_refit_state_dict()),
            group=self.model_update_group,
            src=0,
            post_iter_func=dtensor_post_iter_func,
        )

        if self.cpu_offload:
            self.model = self.move_to_cpu(self.model)

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/stream_weights_via_http")
    def stream_weights_via_http(
        self, sglang_url_to_gpu_uuids: dict[str, list[str]]
    ) -> None:
        """Stream model weights to the SGLang server owning this worker's GPU.

        Gathers full (un-sharded) parameters in the policy dtype and hands them
        to stream_weights_via_http_impl, which serializes CUDA-IPC handles and
        POSTs them to the matching SGLang ``/update_weights_from_tensor``
        endpoint. The trainer-side counterpart to
        broadcast_weights_for_collective for the SGLang backend.
        """
        from dockyard_rl.models.generation.sglang.sglang_copied_utils import (
            monkey_patch_torch_reductions,
        )
        from dockyard_rl.models.policy.utils import stream_weights_via_http_impl

        # Rewrite CUDA-tensor IPC handles to carry the GPU UUID instead of the
        # local device index, matching what the SGLang server expects across a
        # differing CUDA_VISIBLE_DEVICES mapping. Idempotent (self-guarded), so
        # repeated refits are safe.
        monkey_patch_torch_reductions()

        if self.cpu_offload:
            self.model = self.move_to_cuda(self.model)

        def _params() -> Generator[tuple[str, torch.Tensor], None, None]:
            for name, tensor in self._iter_refit_state_dict():
                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()
                yield name, tensor.to(self.dtype, non_blocking=True)

        try:
            stream_weights_via_http_impl(
                params_generator=_params(),
                sglang_url_to_gpu_uuids=sglang_url_to_gpu_uuids,
                rank=self.rank,
                worker_name=type(self).__name__,
                current_device_uuid=self.report_device_id(),
            )
        finally:
            if self.cpu_offload:
                self.model = self.move_to_cpu(self.model)

    @wrap_with_nvtx_name("dtensor_policy_worker/prepare_for_lp_inference")
    def prepare_for_lp_inference(self) -> None:
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.eval()

        torch.randn(1).cuda()  # wake up torch allocator
        if self.optimizer is not None and self.offload_optimizer_for_logprob:
            self.move_optimizer_to_device("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @wrap_with_nvtx_name("dtensor_policy_worker/prepare_for_training")
    def prepare_for_training(self, *args, **kwargs) -> None:
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.train()
        if (
            self.optimizer is not None
            and not self.cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer_to_device("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/offload_before_refit")
    def offload_before_refit(self) -> None:
        """Offload the optimizer to the CPU.

        Uses the non-blocking / pinned-buffer transfer: plain optimizer states
        stage through reusable pinned host memory for full-bandwidth copies,
        DTensor states keep their sharding via .to(), and a synchronize before
        the freed GPU memory is reused keeps it correct. This is the anti-phase
        release valve in colocated distillation (the teacher's resident weights
        reuse the window the optimizer just vacated) and the refit offload in GRPO.
        """
        torch.randn(1).cuda()  # wake up torch allocator
        if self.optimizer is not None:
            self.move_optimizer_to_device("cpu", non_blocking=True)

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/offload_after_refit")
    def offload_after_refit(self) -> None:
        """Offload as much as possible on the CPU."""
        self.model = self.move_to_cpu(self.model)
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def move_optimizer_to_device(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> None:
        """Move optimizer state to ``device``.

        When ``non_blocking`` is set and the target is CPU, plain (non-DTensor)
        optimizer states are staged through reusable pinned host buffers for
        full-bandwidth, asynchronous transfer; DTensor states keep their
        sharding via ``.to()`` and use the standard path. A synchronize at the
        end guarantees the copies complete before the freed GPU memory is reused.
        """
        if self.optimizer is None:
            return
        target = torch.device(device) if isinstance(device, str) else device
        stage_pinned = non_blocking and target.type == "cpu"
        cache: Optional[dict[tuple[int, str], torch.Tensor]] = getattr(
            self, "_opt_pinned_cache", None
        )
        if stage_pinned and cache is None:
            cache = {}
            self._opt_pinned_cache = cache
        for param, state in self.optimizer.state.items():
            for k, v in state.items():
                if not isinstance(v, (DTensor, torch.Tensor)):
                    continue
                if stage_pinned and not isinstance(v, DTensor) and cache is not None:
                    key = (id(param), k)
                    buf = cache.get(key)
                    if buf is None or buf.shape != v.shape or buf.dtype != v.dtype:
                        buf = torch.empty(
                            *v.shape, device="cpu", pin_memory=True, dtype=v.dtype
                        )
                        cache[key] = buf
                    buf.copy_(v, non_blocking=True)
                    state[k] = buf
                else:
                    state[k] = v.to(target, non_blocking=non_blocking)
        if non_blocking:
            torch.cuda.synchronize()

    def get_model_parameter_count(self) -> int:
        """Return the global (unsharded) parameter count of the model.

        ``numel()`` on a DTensor reflects the logical global shape, so this is
        the full model parameter count regardless of FSDP/TP sharding — the
        figure the colocated-distillation memory estimator divides by the shard
        count to get a per-GPU footprint.
        """
        return int(sum(p.numel() for p in self.model.parameters()))

    def move_to_device(self, model: nn.Module, device: str | torch.device) -> nn.Module:
        model = self.move_buffer_to_device(model, device)
        return model.to(device)

    def move_buffer_to_device(
        self, model: nn.Module, device: str | torch.device
    ) -> nn.Module:
        # FSDP modules do not move buffers to the device automatically
        for v in model.buffers():
            torch.utils.swap_tensors(v, v.to(device))
        return model

    def move_to_cuda(self, model: torch.nn.Module) -> torch.nn.Module:
        model = self.move_to_device(model, "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def move_to_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        model = self.move_to_device(model, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        """Save a checkpoint of the model.

        Optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        """
        cast(Any, save_checkpoint)(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
            tokenizer=self.tokenizer if tokenizer_path else None,
            tokenizer_path=tokenizer_path,
        )

    def load_checkpoint(
        self, weights_path: str, optimizer_path: Optional[str] = None
    ) -> None:
        """Load a checkpoint into the model."""
        cast(Any, load_checkpoint)(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
        )

@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker")
)  # pragma: no cover
class DTensorPolicyWorker(DTensorPolicyWorkerImpl):
    pass