"""VllmConfig TypedDict for Project Dockyard.

Collocated mode has been removed.  VllmGeneration is always non-collocated
(dedicated inference fleet, NCCL weight sync from trainer).
"""

from typing import Any, Literal, NotRequired, TypedDict
from dockyard_rl.models.generation.interfaces import GenerationConfig

class VllmSpecificArgs(TypedDict):
    """vLLM engine arguments managed by VllmGeneration."""

    tensor_parallel_size:   int
    pipeline_parallel_size: int
    expert_parallel_size:   int
    gpu_memory_utilization: float
    max_model_len:          int
    skip_tokenizer_init:    bool
    async_engine:           bool
    load_format:            NotRequired[str]
    precision:              NotRequired[str]
    kv_cache_dtype:         Literal["auto", "fp8", "fp8_e4m3"]
    enforce_eager:          NotRequired[bool]

    # FP8 inference serving (precision="fp8"; training stays bf16). Block-wise
    # (128x128) FP8 weights + dynamic activation scales; see
    # quantization/fp8.py::init_fp8. All optional with safe defaults — when
    # precision != "fp8" none of these are read (bf16 serving byte-unchanged).
    pow2_weight_scaling_factors:   NotRequired[bool]   # E8M0 pow2 weight scales (research)
    pow2_activation_scaling_factors: NotRequired[bool] # E8M0 pow2 activation scales (research)
    num_first_layers_in_bf16:      NotRequired[int]    # keep first N layers in bf16
    num_last_layers_in_bf16:       NotRequired[int]    # keep last N layers in bf16
    quantization_ignored_layer_kws: NotRequired[list[str]]  # substrings of layers to skip
    use_deep_gemm:                 NotRequired[bool]   # route block-FP8 GEMMs through DeepGEMM

    # OpenAI-compatible HTTP server (optional — used when coding agents
    # need to call the inference fleet via the chat completions endpoint).
    expose_http_server:               NotRequired[bool]
    http_server_serving_chat_kwargs:  NotRequired[dict[str, Any]]
    tool_parser_plugin:               NotRequired[str]
    reasoning_parser_plugin:          NotRequired[str]

    # Internal performance knobs.
    enable_prefix_caching:            NotRequired[bool]
    vllm_metrics_logger_interval:     NotRequired[float]
    enable_vllm_metrics_logger:       NotRequired[bool]


class VllmConfig(GenerationConfig):
    """Complete configuration for a VllmGeneration instance."""

    vllm_cfg:    VllmSpecificArgs
    vllm_kwargs: NotRequired[dict[str, Any]]

    # Optional post-training quantization config (ModelOpt integration).
    quant_cfg:   NotRequired[str | None]