from typing import Any, Literal, NotRequired, TypedDict
from dockyard_rl.models.generation.interfaces import GenerationConfig

class VllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Additional arguments for vLLM inserted by dockyard_rl based on the context of when vllm is used
    skip_tokenizer_init: bool
    async_engine: bool
    load_format: NotRequired[str]
    precision: NotRequired[str]
    # KV-cache storage dtype.
    # "auto"               — standard BF16/FP16 KV cache (default)
    # "fp8" / "fp8_e4m3"  — vLLM native FP8 KV cache
    # TurboQuant presets (upstream vLLM, GQA/MHA models only):
    # "turboquant_k8v4"    — FP8 keys + 4-bit values  (~2.6x compression, +1.2% PPL)
    # "turboquant_4bit_nc" — 4-bit keys + values       (~3.8x compression, +2.7% PPL)
    # "turboquant_k3v4_nc" — 3-bit keys + 4-bit values (~3.5x compression, +10.6% PPL)
    kv_cache_dtype: NotRequired[
        Literal[
            "auto",
            "fp8",
            "fp8_e4m3",
            "turboquant_k8v4",
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
        ]
    ]
    enforce_eager: NotRequired[bool]
    # By default, dockyard_rl only has a Python handle to the vllm.LLM generation engine.
    # The expose_http_server flag will expose that generation engine as an HTTP server.
    # Exposing vLLM as a server is useful in instances where the multi-turn rollout is
    # performed with utilities outside of dockyard_rl, but the user still wants to take
    # advantage of the refit logic that keeps the policy and generation up to date.
    # Currently it will expose the /tokenize and /v1/chat/completions endpoints.
    expose_http_server: NotRequired[bool]
    # These kwargs are passed to the vllm.LLM HTTP server Chat Completions endpoint config.
    # Typically this will include things like tool parser, chat template, etc.
    http_server_serving_chat_kwargs: NotRequired[dict[str, Any]]
    # A filepath that can be imported to register a vLLM tool parser.
    tool_parser_plugin: NotRequired[str]


class VllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs
    vllm_kwargs: NotRequired[dict[str, Any]]

    # ModelOpt weight quantization config string.
    # Examples: "FP8_DEFAULT_CFG", "NVFP4_DEFAULT_CFG",
    #           "general/ptq/nvfp4_default-fp8_kv"
    # Set to None (default) to skip weight quantization entirely.
    # When set, VllmGeneration will dispatch to VllmQuantGenerationWorker
    # instead of the standard VllmGenerationWorker.
    quant_cfg: NotRequired[str | None]

    # Enable turboquant-plus-vllm MLA KV compression for DeepSeek-V3 / GLM models.
    # Requires the `turboquant-plus-vllm` package on the inference fleet.
    # Has no effect on GQA/MHA models — use vllm_cfg.kv_cache_dtype for those instead.
    turboquant_mla: NotRequired[bool]