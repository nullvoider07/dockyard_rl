"""Utility functions for the vLLM generation layer."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional, Sequence

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.models.generation.vllm.config import VllmConfig
from dockyard_rl.models.generation.interfaces import GenerationDatumSpec

if TYPE_CHECKING:
    from dockyard_rl.tool_protocol.registry import ToolRegistry

# Prompt formatting for vLLM
def format_prompt_for_vllm_generation(
    data: BatchedDataDict[GenerationDatumSpec],
    sample_idx: Optional[int] = None,
    allow_multimodal_inputs: bool = False,
    max_image_pixels: Optional[int] = None,
    max_images_per_sample: Optional[int] = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Convert a BatchedDataDict to vLLM's generate() prompt format.

    vLLM's generate() accepts a list of ``{"prompt_token_ids": [...]}``
    dicts.  This function strips padding using input_lengths so that vLLM
    never receives pad tokens as part of the actual prompt.

    Args:
        data:       BatchedDataDict containing input_ids and input_lengths.
        sample_idx: If provided, return the single prompt for that index
                    (as a plain dict, not wrapped in a list).  If None,
                    return prompts for the entire batch (list of dicts).
        allow_multimodal_inputs: When False (default), per-sample images are
                    rejected rather than forwarded to the engine — the vLLM
                    image-decode path through 0.23.0 carries unpatched
                    advisories (GHSA-8jr5-v98p-w75m). A VLM/CUA config must opt
                    in explicitly. The text path never sets images, so it is
                    unaffected regardless of this flag.
        max_image_pixels: Per-image decoded-pixel cap (decompression-bomb guard);
                    None uses multimodal_utils.DEFAULT_MAX_IMAGE_PIXELS.
        max_images_per_sample: Reject a sample carrying more than this many images;
                    None disables the count cap.

    Returns:
        List of prompt dicts when sample_idx is None; a single prompt dict
        when sample_idx is specified.
    """
    input_ids     = data["input_ids"]
    batch_size    = input_ids.shape[0]
    input_lengths = data["input_lengths"]
    # Optional per-sample multimodal images (list aligned to the batch; each entry
    # a list of PIL images). Present only on the VLM/CUA path — the prompt
    # token_ids already carry the image placeholder tokens, and vLLM expands them
    # against multi_modal_data. Absent on the text path, so those prompts stay
    # byte-identical.
    vllm_images = data.get("vllm_images")

    return_all = sample_idx is None
    start_idx  = 0          if sample_idx is None else sample_idx
    end_idx    = batch_size if sample_idx is None else sample_idx + 1

    prompts: list[dict[str, Any]] = []
    for i in range(start_idx, end_idx):
        valid_length = input_lengths[i].item()
        valid_ids = (
            input_ids[i, :valid_length]
            if valid_length > 0
            else input_ids[i, :0]
        )
        prompt: dict[str, Any] = {"prompt_token_ids": valid_ids.tolist()}
        if vllm_images is not None and i < len(vllm_images) and vllm_images[i]:
            if not allow_multimodal_inputs:
                raise ValueError(
                    "Multimodal image inputs are present but disabled. The vLLM "
                    "image-decode path (<= 0.23.0) has unpatched advisories "
                    "(GHSA-8jr5-v98p-w75m); set generation config "
                    "'allow_multimodal_inputs: true' to opt in once you accept "
                    "that exposure."
                )
            from dockyard_rl.data.multimodal_utils import (
                DEFAULT_MAX_IMAGE_PIXELS,
                normalize_and_validate_image,
            )

            if (
                max_images_per_sample is not None
                and len(vllm_images[i]) > max_images_per_sample
            ):
                raise ValueError(
                    f"Sample {i} carries {len(vllm_images[i])} images, exceeding "
                    f"max_images_per_sample={max_images_per_sample}."
                )
            cap = (
                max_image_pixels
                if max_image_pixels is not None
                else DEFAULT_MAX_IMAGE_PIXELS
            )
            prompt["multi_modal_data"] = {
                "image": [
                    normalize_and_validate_image(img, max_pixels=cap)
                    for img in vllm_images[i]
                ]
            }
        prompts.append(prompt)

    return prompts if return_all else prompts[0]

# Structured tool-use: constrained-decoding payload + submask derivation
#
# The Hermes tool-call wrapper (the native Qwen format, matching
# tool_protocol.hermes) is:
#     <tool_call>\n{"name": "<tool>", "arguments": { ... }}\n</tool_call>
# The structural tag triggers on "<tool_call>" and constrains the enclosed JSON
# object to the union of the constrained tools' call schemas. Everything outside
# a triggered region — leading prose, the <think>…</think> block, and (within the
# JSON) the string *values* the schema leaves as free `type: "string"` — stays
# unconstrained, which is what fork 2 requires for GRPO fidelity and to keep the
# #2656 penalty signal alive.

# Delimiters of the constrained region. Match the Qwen/Hermes chat-template
# rendering (`<tool_call>\n{json}\n</tool_call>`) and tool_protocol.hermes.
_TOOL_CALL_BEGIN = "<tool_call>\n"
_TOOL_CALL_END = "\n</tool_call>"
_TOOL_CALL_TRIGGER = "<tool_call>"


def _call_object_schema(tool_spec: dict) -> dict:
    """JSON schema for the full Hermes call object of one tool.

    Constrains the wrapper object ``{"name": <const>, "arguments": <params>}``:
    the ``name`` is pinned to this tool, and ``arguments`` to the tool's own
    parameter schema. The argument string *values* remain free wherever the
    parameter schema declares ``type: "string"`` without an enum/pattern.
    """
    function = tool_spec["function"]
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "const": function["name"]},
            "arguments": parameters,
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def build_structural_tag(
    registry: "ToolRegistry",
    constrained_tools: Sequence[str] | None = None,
) -> str:
    """Build the vLLM 0.21 ``structural_tag`` payload for the constrained tools.

    Produces the JSON-serialized legacy structural-tag shape that vLLM's xgrammar
    backend accepts (``structures``/``triggers``; validated via
    ``xgr.StructuralTagItem`` + ``xgr.Grammar.from_structural_tag(tags, triggers)``
    — see vLLM ``v1/structured_output/backend_xgrammar.py``). Each structure pins
    one tool's call object; the shared trigger ``"<tool_call>"`` activates the
    constraint only when the model opens a tool call, leaving thinking/prose free.

    Args:
        registry: The environment's tool registry (source of the schemas).
        constrained_tools: Tool names to constrain. None ⇒ constrain every tool in
            the registry. Each name must be present in the registry.

    Returns:
        A JSON string suitable for ``StructuredOutputsParams(structural_tag=...)``.

    Raises:
        ValueError: if ``constrained_tools`` is empty, or names a tool absent from
            the registry.
    """
    names = list(registry.names()) if constrained_tools is None else list(constrained_tools)
    if not names:
        raise ValueError("build_structural_tag requires at least one constrained tool")

    structures = []
    for name in names:
        spec = registry.get(name)
        if spec is None:
            raise ValueError(f"tool {name!r} is not in the registry {registry.names()}")
        structures.append(
            {
                "begin": _TOOL_CALL_BEGIN,
                "schema": _call_object_schema(spec),
                "end": _TOOL_CALL_END,
            }
        )

    payload = {"structures": structures, "triggers": [_TOOL_CALL_TRIGGER]}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int], start: int) -> int:
    """Return the first index >= ``start`` where ``needle`` occurs, else -1."""
    n = len(needle)
    if n == 0:
        return -1
    limit = len(haystack) - n
    i = start
    while i <= limit:
        if list(haystack[i : i + n]) == list(needle):
            return i
        i += 1
    return -1


def derive_structural_submask(
    generated_token_ids: Sequence[int],
    begin_token_ids: Sequence[int],
    end_token_ids: Sequence[int],
) -> list[int]:
    """Derive the CPU-determinable structural-scaffolding submask for a turn.

    Marks (with 1) the ``<tool_call>``-open and ``</tool_call>``-close delimiter
    token spans within ``generated_token_ids``; all other positions are 0. The
    delimiter tokens are unambiguously grammar-forced under the structural tag,
    so they are the conservative, exactly-determinable part of the fork-2
    loss-mask. The mask is aligned to ``generated_token_ids`` (response-only); the
    caller offsets it into the full prompt+response ``output_ids``.

    What this does NOT capture (HV-gated): the JSON-skeleton tokens *inside* the
    delimiters (the ``{"name": ...,"arguments":{`` braces/keys) that the grammar
    also forces. Separating those forced skeleton tokens from the free argument
    *values* requires the per-step matcher bitmask cardinality, which vLLM 0.21
    does not surface on ``CompletionOutput`` (no field exists). Surfacing it needs
    an engine-side patch and a live grammar backend → hardware-deferred. Until
    then the trainer loss-masks at least the delimiter spans (correct subset, no
    false positives), which removes the dominant forced-token bias.

    Args:
        generated_token_ids: Response-only generated token ids for one sample.
        begin_token_ids: Tokenized ``<tool_call>`` open delimiter.
        end_token_ids: Tokenized ``</tool_call>`` close delimiter.

    Returns:
        A list of 0/1 ints the same length as ``generated_token_ids``.
    """
    mask = [0] * len(generated_token_ids)
    for delim in (begin_token_ids, end_token_ids):
        if not delim:
            continue
        cursor = 0
        while True:
            hit = _find_subsequence(generated_token_ids, delim, cursor)
            if hit < 0:
                break
            for j in range(hit, hit + len(delim)):
                mask[j] = 1
            cursor = hit + len(delim)
    return mask

# Speculative-decode counter helpers
def aggregate_spec_decode_counters(
    worker_metrics: list[dict[str, float | list[float]]],
) -> dict[str | tuple[str, int], float]:
    """Aggregate speculative-decoding counters from multiple DP leaders.

    Args:
        worker_metrics: List of metric dicts from each DP leader worker.
                        Values may be scalars or per-position lists.

    Returns:
        Combined counter dict keyed by metric name (scalars) or
        (metric_name, position) tuple (per-position metrics).
    """
    counters: dict[str | tuple[str, int], float] = defaultdict(float)

    for report in worker_metrics:
        for metric_name, value in report.items():
            if "spec_decode" not in metric_name:
                continue
            if isinstance(value, list):
                for position, pos_value in enumerate(value, 1):
                    counters[metric_name, position] += pos_value
            else:
                counters[metric_name] += value

    return dict(counters)

def compute_spec_decode_metrics(
    start_counters: dict[str | tuple[str, int], float],
    end_counters:   dict[str | tuple[str, int], float],
) -> dict[str, float]:
    """Compute delta and derived metrics for speculative decoding.

    Args:
        start_counters: Snapshot taken before generation.
        end_counters:   Snapshot taken after generation.

    Returns:
        Dict of ``vllm/<metric_name>`` entries suitable for W&B / TensorBoard.
    """
    keys  = set(start_counters) | set(end_counters)
    delta = {
        k: end_counters.get(k, 0.0) - start_counters.get(k, 0.0)
        for k in keys
    }

    num_drafts          = delta.get("vllm:spec_decode_num_drafts", 0.0)
    num_draft_tokens    = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    num_accepted_tokens = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)

    acceptance_length = (
        1.0 + (num_accepted_tokens / num_drafts)
        if num_drafts > 0
        else 1.0
    )
    acceptance_rate = (
        num_accepted_tokens / num_draft_tokens
        if num_draft_tokens > 0
        else 0.0
    )

    spec_metrics: dict[str, float] = {
        "vllm/spec_num_drafts":           num_drafts,
        "vllm/spec_num_draft_tokens":     num_draft_tokens,
        "vllm/spec_num_accepted_tokens":  num_accepted_tokens,
        "vllm/spec_acceptance_length":    acceptance_length,
        "vllm/spec_acceptance_rate":      acceptance_rate,
    }

    for key, value in delta.items():
        if isinstance(key, tuple):
            metric_name, position = key
            spec_metrics[f"vllm/{metric_name}-{position}"] = value
            if num_drafts > 0:
                spec_metrics[f"vllm/spec_acceptance_rate-pos-{position}"] = (
                    value / num_drafts
                )

    return spec_metrics

# Worker class resolution for optional quantization support
# Maps the default (non-quantized) worker FQN to its quantized counterpart
# when quant_cfg is set in VllmConfig.  The core generation path has no
# direct import of ModelOpt; this registry keeps the dependency optional.
GENERATION_WORKER_OVERRIDES = {
    "dockyard_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": (
        "dockyard_rl.models.generation.vllm.vllm_quant_worker"
        ".VllmQuantGenerationWorker"
    ),
    "dockyard_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": (
        "dockyard_rl.models.generation.vllm.vllm_quant_worker"
        ".VllmQuantAsyncGenerationWorker"
    ),
}

def resolve_generation_worker_cls(default_cls: str, config: VllmConfig) -> str:
    """Return the quantized worker FQN when quant_cfg is set, else default_cls.

    Safe to call even when ModelOpt is not installed — returns default_cls
    unchanged whenever quant_cfg is None.

    Args:
        default_cls: FQN of the standard worker class.
        config:      VllmConfig dict (or any dict with optional quant_cfg key).

    Returns:
        Resolved FQN string.
    """
    if config.get("quant_cfg") is None:
        return default_cls
    return GENERATION_WORKER_OVERRIDES.get(default_cls, default_cls)