"""vLLM synchronous generation worker"""

import copy
import functools
import gc
import logging
import os
import sys
from importlib.util import find_spec
from typing import Any, Optional, cast
import ray
import torch
from transformers import AutoConfig

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.distributed.worker_groups_utils import (
    get_nsight_config_if_pattern_matches,
)
from dockyard_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    assert_finite_sampling_params,
    verify_right_padding,
)
from dockyard_rl.models.generation.vllm.config import VllmConfig
from dockyard_rl.models.generation.vllm.router_capture import (
    routed_experts_from_vllm_output,
    stack_routed_experts,
)
from .utils import (
    format_prompt_for_vllm_generation,
)

logger = logging.getLogger(__name__)

# ── NVTX helper
def _nvtx_available() -> bool:
    try:
        import torch.cuda.nvtx  # noqa: F401
        return True
    except Exception:
        return False

def wrap_with_nvtx_name(name: str):
    """Decorator that wraps a method with an NVTX range for profiling.

    Falls back to a no-op when CUDA NVTX is not available (CPU-only
    environments, containers without the CUDA toolkit in PATH).
    """
    def decorator(fn):
        if not _nvtx_available():
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        return wrapper
    return decorator

# vLLM v1 engine detection
def _is_vllm_v1_engine_enabled() -> bool:
    """Return True when the vLLM v1 engine path is active.

    Defaults to enabled (VLLM_USE_V1=1) which matches vLLM ≥ 0.6.x
    default behaviour.
    """
    return os.environ.get("VLLM_USE_V1", "1") != "0"

# Base worker
class BaseVllmGenerationWorker:
    """Shared base for synchronous and asynchronous vLLM generation workers.

    Handles:
    - Worker role detection (model owner vs. secondary TP/PP member).
    - vLLM source-file monkey-patching for Dockyard compatibility.
    - Engine construction argument assembly.
    - Sampling parameter building.
    - Spec-decode metric collection.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    # Static configuration hook for Ray worker creation (overridden by async subclass)
    @staticmethod
    def configure_worker(
        num_gpus: int | float,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """Provide complete worker configuration for TP/PP parallelism.

        Returns:
            (resources, env_vars, init_kwargs) triple consumed by
            RayWorkerBuilder.create_worker().
        """
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        local_bundle_indices = None
        if bundle_indices is not None:
            node_idx, local_bundle_indices = bundle_indices
            init_kwargs["bundle_indices"] = local_bundle_indices

            # Unique per-DP-group seed derived from node + bundle position.
            if len(local_bundle_indices) == 1:
                seed = node_idx * 1024 + local_bundle_indices[0]
            else:
                bundle_id = local_bundle_indices[0] // len(local_bundle_indices)
                seed = node_idx * 1024 + bundle_id

            init_kwargs["seed"] = seed
            # Per-DP vLLM cache root avoids cache-sharing race conditions.
            # See https://github.com/vllm-project/vllm/issues/18851
            env_vars["VLLM_CACHE_ROOT"] = os.path.expanduser(
                f"~/.cache/vllm_{seed}"
            )

        is_parallel = (
            local_bundle_indices is not None
            and len(local_bundle_indices) > 1
        ) or local_bundle_indices is None

        if is_parallel:
            # vLLM manages CUDA device assignment internally for TP/PP groups.
            resources["num_gpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            init_kwargs["fraction_of_gpus"] = num_gpus

        env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Skip vLLM's P2P check; rely on the CUDA driver instead.
        env_vars["VLLM_SKIP_P2P_CHECK"] = "1"

        return resources, env_vars, init_kwargs

    # Constructor
    def __init__(
        self,
        config: VllmConfig,
        bundle_indices: Optional[list[int]] = None,
        fraction_of_gpus: float = 1.0,
        seed: Optional[int] = None,
        extra_env_vars: Optional[list[str]] = None,
    ) -> None:
        """Initialise a vLLM generation worker.

        Args:
            config:           VllmConfig for this worker.
            bundle_indices:   Local bundle indices within the placement group
                              for the first worker in a tied TP/PP group.
                              None for secondary (non-model-owner) workers.
            fraction_of_gpus: Fraction of a GPU claimed by this worker.
            seed:             Per-DP-group random seed.
            extra_env_vars:   Additional env var names forwarded into the
                              vLLM executor subprocess.
        """
        self.cfg              = config
        assert_finite_sampling_params(self.cfg["temperature"], self.cfg["top_p"])
        self.model_name       = self.cfg.get("model_name", "")
        self.tensor_parallel_size   = self.cfg["vllm_cfg"]["tensor_parallel_size"]
        self.pipeline_parallel_size = self.cfg["vllm_cfg"]["pipeline_parallel_size"]
        self.expert_parallel_size   = self.cfg["vllm_cfg"]["expert_parallel_size"]
        self.enable_expert_parallel = self.expert_parallel_size > 1
        self.gpu_memory_utilization = self.cfg["vllm_cfg"]["gpu_memory_utilization"]
        self.precision              = self.cfg["vllm_cfg"].get("precision", "bfloat16")
        self.fraction_of_gpus       = fraction_of_gpus
        self.is_model_owner         = bundle_indices is not None
        self.py_executable          = sys.executable
        # MoE router-replay capture (#2908): when enabled the engine is built
        # with enable_return_routed_experts and generate() aligns each sample's
        # per-token routing into a routed_experts column. Plain {"enabled": bool}
        # dict propagated by the GRPO setup; absent → disabled, path unchanged.
        _router_replay = self.cfg.get("router_replay") or {}
        self.router_replay_capture  = bool(
            _router_replay.get("enabled", False)
            if isinstance(_router_replay, dict)
            else getattr(_router_replay, "enabled", False)
        )

        # Non-model-owner workers don't load the model.
        if not self.is_model_owner:
            self.llm       = None
            self.tokenizer = None
            self.rank      = 0
            self.world_size = 1
            return

        self.rank       = 0
        self.world_size = 1

        # vLLM source-file patches for compatibility
        from vllm.logger import init_logger as _vllm_logger

        logger = _vllm_logger("dockyard_vllm_patch")

        def _get_vllm_file(rel_path: str) -> str:
            spec = find_spec("vllm")
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError(
                    f"vLLM package not found while patching '{rel_path}'."
                )
            base = next(iter(spec.submodule_search_locations))
            path = os.path.join(base, *rel_path.split("/"))
            if not os.path.exists(path):
                raise RuntimeError(
                    f"vLLM file to patch not found: '{path}'. "
                    "vLLM version mismatch?"
                )
            return path

        def _patch_init_workers_ray() -> None:
            """Forward py_executable and extra NCCL vars into vLLM's Ray executor."""
            try:
                f = _get_vllm_file("v1/executor/ray_executor.py")
            except RuntimeError:
                return
            with open(f) as fh:
                content = fh.read()

            extra = ", ".join(f'"{v}"' for v in (extra_env_vars or []))
            old_lines = [
                "self._init_workers_ray(placement_group)",
                'ADDITIONAL_ENV_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}',
            ]
            new_lines = [
                f'self._init_workers_ray(placement_group, runtime_env={{"py_executable": "{self.py_executable}"}})',
                f'ADDITIONAL_ENV_VARS = {{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE", "RAY_ENABLE_UV_RUN_RUNTIME_ENV", {extra}}}',
            ]
            changed = False
            for old, new in zip(old_lines, new_lines):
                if new not in content and old in content:
                    content = content.replace(old, new)
                    changed = True
            if changed:
                with open(f, "w") as fh:
                    fh.write(content)

        def _patch_hermes_tool_parser_thread_safety() -> None:
            """Cache tokenizer calls in Hermes2ProToolParser.__init__."""
            try:
                f = _get_vllm_file("tool_parsers/hermes_tool_parser.py")
            except RuntimeError:
                return
            with open(f) as fh:
                content = fh.read()
            if "_tokenizer_cache" in content:
                return
            old_import = "import json\nfrom collections.abc import Sequence"
            new_import = "import json\nimport threading\nfrom collections.abc import Sequence"
            old_class = "class Hermes2ProToolParser(ToolParser):"
            new_class = (
                "class Hermes2ProToolParser(ToolParser):\n"
                "    _tokenizer_lock = threading.Lock()\n"
                "    _tokenizer_cache = {}"
            )
            if old_import in content:
                content = content.replace(old_import, new_import, 1)
            if old_class in content:
                content = content.replace(old_class, new_class, 1)
            with open(f, "w") as fh:
                fh.write(content)
            logger.info("Patched hermes_tool_parser for thread-safety.")

        _patch_init_workers_ray()
        _patch_hermes_tool_parser_thread_safety()

        # Import vLLM now that patches are applied
        try:
            import vllm
            self.SamplingParams = vllm.SamplingParams
            # Structured tool-use (fork 2): only referenced when a per-sample
            # structured_spec is supplied; importing here keeps it bound once and
            # avoids a per-call import on the constrained path.
            from vllm.sampling_params import StructuredOutputsParams
            self.StructuredOutputsParams = StructuredOutputsParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Ensure it was built during the "
                "ubuntu-swe image build (§5c)."
            ) from exc

        vllm_kwargs: dict[str, Any] = copy.deepcopy(
            self.cfg.get("vllm_kwargs", {})
        )
        model_parallel_size = self.tensor_parallel_size * self.pipeline_parallel_size

        if model_parallel_size > 1:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(
                self.fraction_of_gpus / model_parallel_size
            )
            assert bundle_indices is not None
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
                map(str, bundle_indices)
            )
            vllm_kwargs["distributed_executor_backend"] = "ray"
        else:
            vllm_kwargs["distributed_executor_backend"] = None

        os.environ["VLLM_USE_V1"] = (
            "1" if _is_vllm_v1_engine_enabled() else "0"
        )
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # Expert-parallel (MoE) DP address wiring.
        if self.expert_parallel_size > self.tensor_parallel_size:
            world_size = (
                int(os.environ["VLLM_DP_SIZE"]) * model_parallel_size
            )
            rank = int(os.environ["RANK"]) % world_size
            os.environ["VLLM_DP_RANK"] = str(rank // model_parallel_size)
            os.environ["VLLM_DP_RANK_LOCAL"] = str(
                (rank % 8) // model_parallel_size
            )
            leader_rank = (
                int(os.environ["RANK"]) // world_size * world_size
            )
            addr_list = eval(os.environ["AVAILABLE_ADDR_LIST"])
            port_list = eval(os.environ["AVAILABLE_PORT_LIST"])
            os.environ["VLLM_DP_MASTER_IP"]   = addr_list[leader_rank]
            os.environ["VLLM_DP_MASTER_PORT"] = str(port_list[leader_rank])

        load_format = self.cfg["vllm_cfg"].get("load_format")

        # Nsight profiling warning for TP/PP Ray executor.
        if (
            len(get_nsight_config_if_pattern_matches("vllm_generation_worker")) > 0
            and vllm_kwargs.get("distributed_executor_backend") == "ray"
        ):
            logger.warning(
                "Nsight profiling with vLLM Ray executor: "
                "output files are named by the executor automatically."
            )
            vllm_kwargs["ray_workers_use_nsight"] = True

        # fp8 quantization (optional ModelOpt integration).
        if self.cfg["vllm_cfg"].get("precision") == "fp8":
            try:
                from dockyard_rl.models.generation.vllm.quantization.fp8 import (  # type: ignore[import]
                    init_fp8,
                )
                fp8_kwargs = init_fp8(
                    self.cfg["vllm_cfg"], self.model_name, model_parallel_size
                )
                vllm_kwargs.update(fp8_kwargs)
                self.precision = "bfloat16"
            except ImportError:
                logger.warning(
                    "fp8 quantization requested but "
                    "dockyard_rl.models.generation.vllm.quantization.fp8 "
                    "not found. Using precision as-is."
                )

        if not isinstance(vllm_kwargs.get("hf_overrides"), dict):
            vllm_kwargs["hf_overrides"] = {}

        hf_config = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        architectures = getattr(hf_config, "architectures", [])

        # Gemma3 / Qwen3.5 MoE: disable skip_tokenizer_init.
        _needs_tokenizer_init = (
            "Gemma3ForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        )
        if any(a in architectures for a in _needs_tokenizer_init):
            if self.cfg["vllm_cfg"].get("skip_tokenizer_init"):
                logger.info(
                    "Forcing skip_tokenizer_init=False for %s architecture.",
                    [a for a in architectures if a in _needs_tokenizer_init],
                )
            self.cfg["vllm_cfg"]["skip_tokenizer_init"] = False

        llm_kwargs = dict(
            model                    = self.model_name,
            served_model_name        = self.model_name,
            load_format              = load_format,
            skip_tokenizer_init      = self.cfg["vllm_cfg"]["skip_tokenizer_init"],
            tensor_parallel_size     = self.tensor_parallel_size,
            pipeline_parallel_size   = self.pipeline_parallel_size,
            enable_expert_parallel   = self.enable_expert_parallel,
            gpu_memory_utilization   = self.gpu_memory_utilization,
            enable_prefix_caching    = self.cfg["vllm_cfg"].get(
                "enable_prefix_caching",
                torch.cuda.get_device_capability()[0] >= 8,
            ),
            dtype                    = self.precision,
            seed                     = seed,
            enforce_eager            = self.cfg["vllm_cfg"].get("enforce_eager", False),
            max_model_len            = self.cfg["vllm_cfg"]["max_model_len"],
            trust_remote_code        = True,
            worker_extension_cls     = (
                "dockyard_rl.models.generation.vllm.vllm_backend"
                ".VllmInternalWorkerExtension"
            ),
            enable_sleep_mode        = True,
            disable_log_stats        = False,
            logprobs_mode            = "processed_logprobs",
            # KV-cache dtype — covers both plain fp8 and all TurboQuant presets.
            # "auto" is vLLM's default (standard BF16/FP16 KV cache).
            kv_cache_dtype           = self.cfg["vllm_cfg"].get("kv_cache_dtype", "auto"),
            **vllm_kwargs,
        )

        # MoE router-replay (#2908): enable the vLLM routed-expert capture
        # subsystem so CompletionOutput.routed_experts / prompt_routed_experts are
        # populated. vLLM rejects this under async_scheduling / PP>1 / PCP>1 /
        # DCP>1 (I4 pre-validates and fails fast with a dockyard-native message
        # before engine construction); only set it when replay is on.
        if self.router_replay_capture:
            llm_kwargs["enable_return_routed_experts"] = True

        # MLA KV-cache compression.
        #
        # The turboquant-plus-vllm third-party plugin is no longer used.  TurboQuant
        # is integrated directly into upstream vLLM and is activated for all
        # architectures — including MLA models — via the kv_cache_dtype engine
        # argument already forwarded above (e.g. "turboquant_k8v4").  No pre-engine
        # monkey-patching is required.
        #
        # If a legacy config still carries turboquant_mla=True, emit a clear warning
        # so the operator knows to migrate to kv_cache_dtype instead.
        if self.cfg.get("turboquant_mla", False):
            logger.warning(
                "[dockyard_rl] 'turboquant_mla' is no longer supported. "
                "TurboQuant KV-cache compression is now handled entirely by vLLM's "
                "native integration via the 'kv_cache_dtype' engine argument "
                "(e.g. kv_cache_dtype='turboquant_k8v4').  "
                "Remove 'turboquant_mla' from your VllmConfig and set 'kv_cache_dtype' "
                "to the desired preset.  The flag is currently ignored."
            )

        self._create_engine(llm_kwargs)
        # Populated by post_init() — used in weight update flows.
        self.vllm_device_ids: Optional[list[str]] = None

    # Engine creation (overridden by async subclass)
    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        import vllm
        self.llm = vllm.LLM(**llm_kwargs)

    # Liveness check (overridden by async subclass)
    def is_alive(self) -> bool:
        return True

    # Stop-string merging
    def _merge_stop_strings(
        self,
        batch_stop_strings,
    ) -> Optional[list[str]]:
        stop_set: set[str] = set()
        if stop_strings_val := self.cfg.get("stop_strings"):
            stop_set.update(stop_strings_val)
        if batch_stop_strings is not None:
            for sample_ss in batch_stop_strings:
                if sample_ss:
                    stop_set.update(sample_ss)
        return list(stop_set) if stop_set else None

    # Sampling params
    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        stop_strings,
        max_new_tokens: Optional[int] = None,
        structured_spec: Optional[dict] = None,
    ):
        top_k_cfg = self.cfg["top_k"]
        top_k_val = 1 if greedy else (
            top_k_cfg if top_k_cfg is not None else -1
        )
        temperature  = 0.0 if greedy else self.cfg["temperature"]
        max_tokens   = (
            max_new_tokens
            if max_new_tokens is not None
            else self.cfg["max_new_tokens"]
        )
        params = self.SamplingParams(
            temperature    = temperature,
            top_p          = self.cfg["top_p"],
            top_k          = top_k_val,
            max_tokens     = max_tokens,
            logprobs       = 0,
            stop_token_ids = self.cfg["stop_token_ids"],
            stop           = stop_strings,
            include_stop_str_in_output = True,
            ignore_eos     = self.cfg.get("ignore_eos", False),
        )
        # Structured tool-use (fork 2): constrain only when a spec is present, so
        # the unconstrained path produces byte-identical SamplingParams. The spec
        # carries the pre-built structural_tag JSON string (utils.build_structural_tag).
        if structured_spec is not None:
            structural_tag = structured_spec.get("structural_tag")
            if structural_tag:
                params.structured_outputs = self.StructuredOutputsParams(
                    structural_tag=structural_tag
                )
        return params

    # GPU profiling
    def start_gpu_profiling(self) -> None:
        torch.cuda.profiler.start()
        if self.llm is not None:
            self.llm.collective_rpc("start_gpu_profiling", args=tuple())

    def stop_gpu_profiling(self) -> None:
        torch.cuda.profiler.stop()
        if self.llm is not None:
            self.llm.collective_rpc("stop_gpu_profiling", args=tuple())

    # Spec-decode metrics
    def _get_raw_spec_counters(
        self,
    ) -> dict[str, float | list[float]]:
        metrics: dict[str, float | list[float]] = {}
        if self.llm is None:
            return metrics
        if hasattr(self.llm, "get_metrics"):
            vllm_metrics = self.llm.get_metrics()
        else:
            from vllm.v1.metrics.reader import get_metrics_snapshot
            vllm_metrics = get_metrics_snapshot()
        for m in vllm_metrics:
            if hasattr(m, "values"):
                metrics[m.name] = m.values
            elif hasattr(m, "value"):
                metrics[m.name] = m.value
        return metrics

# Synchronous worker implementation
class VllmGenerationWorkerImpl(BaseVllmGenerationWorker):
    """Synchronous (non-async-engine) vLLM generation worker."""

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        import vllm
        self.llm = vllm.LLM(**llm_kwargs)

    def post_init(self) -> None:
        self.vllm_device_ids = self.report_device_id()

    def init_collective(
        self,
        rank_prefix:       int,
        ip:                str,
        port:              int,
        world_size:        int,
        train_world_size:  int,
    ) -> None:
        assert self.llm is not None
        self.llm.collective_rpc(
            "init_collective",
            args=(rank_prefix, ip, port, world_size, train_world_size),
        )

    @wrap_with_nvtx_name("vllm_generation_worker/generate")
    def generate(
        self,
        data:   BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Synchronous batch generation."""
        if len(data["input_ids"]) == 0:
            return BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids":                 torch.zeros((0, 0), dtype=torch.long),
                    "logprobs":                   torch.zeros((0, 0), dtype=torch.float),
                    "generation_lengths":         torch.zeros(0, dtype=torch.long),
                    "unpadded_sequence_lengths":  torch.zeros(0, dtype=torch.long),
                    "truncated":                  torch.zeros(0, dtype=torch.bool),
                }
            )

        input_ids     = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_stop    = data.get("stop_strings", [])
        stop_strings  = self._merge_stop_strings(batch_stop)
        sampling_params = self._build_sampling_params(
            greedy=greedy, stop_strings=stop_strings
        )

        verify_right_padding(data, pad_value=cast(int, self.cfg.get("_pad_token_id")))
        padded_input_length = input_ids.size(1)
        prompts = format_prompt_for_vllm_generation(
            data,
            allow_multimodal_inputs=self.cfg.get("allow_multimodal_inputs", False),
            max_image_pixels=self.cfg.get("max_image_pixels"),
            max_images_per_sample=self.cfg.get("max_images_per_sample"),
        )

        assert self.llm is not None
        outputs = self.llm.generate(prompts, sampling_params) # type: ignore[arg-type]

        output_ids_list:              list[torch.Tensor] = []
        logprobs_list:                list[torch.Tensor] = []
        generation_lengths:           list[int]          = []
        unpadded_sequence_lengths:    list[int]          = []
        truncated_list:               list[bool]         = []
        routed_experts_list:          list[Optional[torch.Tensor]] = []

        max_gen = max(len(o.outputs[0].token_ids) for o in outputs)
        total_len = padded_input_length + max_gen

        for i, output in enumerate(outputs):
            seq_len = int(input_lengths[i])
            gen     = output.outputs[0]
            gen_ids = list(gen.token_ids)
            n_gen   = len(gen_ids)

            full_output = torch.full(
                (total_len,), cast(int, self.cfg.get("_pad_token_id")),
                dtype=input_ids.dtype,
            )
            full_output[:seq_len] = input_ids[i, :seq_len]
            full_output[seq_len: seq_len + n_gen] = torch.tensor(gen_ids)
            output_ids_list.append(full_output)

            full_logprobs = torch.zeros(total_len, dtype=torch.float32)
            if hasattr(gen, "logprobs") and gen.logprobs:
                try:
                    for idx, lp_dict in enumerate(gen.logprobs):
                        if lp_dict:
                            position = seq_len + idx
                            full_logprobs[position] = next(
                                iter(lp_dict.items())
                            )[1].logprob
                except Exception:
                    import traceback
                    traceback.print_exc()
            logprobs_list.append(full_logprobs)

            generation_lengths.append(n_gen)
            unpadded_sequence_lengths.append(seq_len + n_gen)
            truncated_list.append(gen.finish_reason == "length")

            if self.router_replay_capture:
                routed_experts_list.append(
                    routed_experts_from_vllm_output(
                        output,
                        gen,
                        valid_length=seq_len + n_gen,
                        padded_length=total_len,
                    )
                )

        result = BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids":  torch.stack(output_ids_list),
                "logprobs":    torch.stack(logprobs_list),
                "generation_lengths": torch.tensor(
                    generation_lengths, dtype=torch.long
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    unpadded_sequence_lengths, dtype=torch.long
                ),
                "truncated": torch.tensor(truncated_list, dtype=torch.bool),
            }
        )
        if self.router_replay_capture:
            routed_experts = stack_routed_experts(
                routed_experts_list, padded_length=total_len
            )
            if routed_experts is not None:
                result["routed_experts"] = routed_experts
        return result

    @wrap_with_nvtx_name("vllm_generation_worker/generate_text")
    def generate_text(
        self,
        data:   BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate text responses (string output instead of token ids)."""
        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_text cannot be used with async_engine=True."
            )
        batch_stop = data.get(
            "stop_strings",
            [self.cfg.get("stop_strings")] * len(data["prompts"]),
        )
        stop_set: set[str] = set()
        for ss in batch_stop:
            if ss:
                stop_set.update(ss)
        if stop_strings_val := self.cfg.get("stop_strings"):
            stop_set.update(stop_strings_val)
        stop_strings = list(stop_set) if stop_set else None

        top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
        sampling_params = self.SamplingParams(
            temperature    = self.cfg["temperature"] if not greedy else 0,
            top_p          = self.cfg["top_p"],
            top_k          = top_k if not greedy else 1,
            max_tokens     = self.cfg["max_new_tokens"],
            stop_token_ids = self.cfg["stop_token_ids"],
            stop           = stop_strings,
            include_stop_str_in_output = True,
            ignore_eos     = self.cfg.get("ignore_eos", False),
        )
        assert self.llm is not None
        outputs = self.llm.generate(data["prompts"], sampling_params)
        texts = [o.outputs[0].text for o in outputs]
        return BatchedDataDict[GenerationOutputSpec]({"texts": texts})

    def report_device_id(self) -> list[str]:
        assert self.llm is not None
        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "Use report_device_id_async for async engine."
            )
        return cast(
            list[str],
            self.llm.collective_rpc("report_device_id", args=tuple()),
        )

    def prepare_refit_info(
        self,
        state_dict_info: dict[str, Any],
    ) -> None:
        assert self.llm is not None
        self.llm.collective_rpc("prepare_refit_info", args=(state_dict_info,))

    @wrap_with_nvtx_name("vllm_generation_worker/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        try:
            assert self.llm is not None
            if self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "Use update_weights_via_ipc_zmq_async for async engine."
                )
            result = self.llm.collective_rpc(
                "update_weights_via_ipc_zmq", args=tuple()
            )
            worker_result = result[0]
            if not worker_result:
                print(f"Error: worker failed to update weights. Result: {worker_result}")
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback; traceback.print_exc()
            return False

    @wrap_with_nvtx_name("vllm_generation_worker/update_weights_from_collective")
    def update_weights_from_collective(self) -> bool:
        try:
            assert self.llm is not None
            if self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "Use update_weights_from_collective_async for async engine."
                )
            result = self.llm.collective_rpc(
                "update_weights_from_collective", args=tuple()
            )
            worker_result = result[0]
            if not worker_result:
                print(f"Error: worker failed to update weights. Result: {worker_result}")
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback; traceback.print_exc()
            return False

    def reset_prefix_cache(self) -> None:
        assert self.llm is not None
        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError("Use reset_prefix_cache_async.")
        self.llm.llm_engine.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def sleep(self) -> None:
        assert self.llm is not None
        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError("Use sleep_async.")
        self.llm.llm_engine.reset_prefix_cache()
        if hasattr(self.llm, "renderer") and hasattr(
            self.llm.renderer, "clear_mm_cache"
        ):
            self.llm.renderer.clear_mm_cache()
        self.llm.sleep(level=1)
        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, **kwargs) -> None:
        assert self.llm is not None
        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError("Use wake_up_async.")
        tags = kwargs.get("tags")
        self.llm.wake_up(**({"tags": tags} if tags is not None else {}))

    def shutdown(self) -> bool:
        try:
            if self.llm is not None:
                self.llm.collective_rpc("cleanup", args=tuple())
                del self.llm
            self.llm       = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False

@ray.remote(
    runtime_env={
        **get_nsight_config_if_pattern_matches("vllm_generation_worker")
    }
)  # pragma: no cover
class VllmGenerationWorker(VllmGenerationWorkerImpl):
    pass