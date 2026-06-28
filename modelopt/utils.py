"""Lightweight quantization config resolver usable by both vLLM workers."""

from fnmatch import fnmatchcase
from typing import Any, Iterator

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_config

_QUANT_IGNORE_NAME_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_scale_2",
)

# Layers kept in native dtype by the real-quant NVFP4 vLLM rollout. Shared
# between the vLLM deployment quantization_config (ignore list) and the
# export-side ignore patterns. Attention and router/gate layers stay unquantized
# for accuracy; lm_head/output stay native because the sampler reads them.
DEFAULT_NVFP4_IGNORE = [
    "lm_head",
    "*output_layer*",
    "*mlp.gate",
    "*router*",
    "*block_sparse_moe.gate*",
    "*self_attention*",
    "*self_attn*",
]


def _iter_quant_ignore_suffix_variants(name: str) -> Iterator[str]:
    """Yield ``name`` and, if it ends in a known quant suffix, the stripped form."""
    yield name
    for suffix in _QUANT_IGNORE_NAME_SUFFIXES:
        if name.endswith(suffix):
            yield name[: -len(suffix)]
            break


def iter_quant_ignore_name_candidates(name: str) -> Iterator[str]:
    """Yield name variants that ModelOpt real-quant ignore patterns may match.

    Covers both the bare param suffix forms and the ``model.`` prefix variants,
    since refit-streamed names and vLLM's registered names differ on that prefix.
    """
    yield from _iter_quant_ignore_suffix_variants(name)

    alternate = (
        name.removeprefix("model.") if name.startswith("model.") else f"model.{name}"
    )
    if alternate == name:
        return

    yield from _iter_quant_ignore_suffix_variants(alternate)


def matches_quant_ignore_pattern(name: str, patterns: list[str]) -> bool:
    """Return whether ``name`` matches any ModelOpt real-quant ignore pattern."""
    return any(
        fnmatchcase(candidate, pattern)
        for candidate in iter_quant_ignore_name_candidates(name)
        for pattern in patterns
    )


def build_vllm_modelopt_nvfp4_config(
    *,
    ignore: list[str] | None = None,
) -> dict[str, Any]:
    """Build the HuggingFace ``quantization_config`` consumed by vLLM ModelOpt NVFP4.

    ``quant_cfg`` recipes are ModelOpt PTQ/QAT configs consumed by
    ``mtq.quantize``; vLLM instead expects the deployment-side
    ``quantization_config`` shape. ``quant_algo: NVFP4`` selects the dense linear
    method; the ``weight_only`` / ``quant_mode`` markers route it to the W4A16
    Marlin path inside the patched method (see ``vllm_modelopt_patch``).
    """
    return {
        "quant_method": "modelopt",
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": ignore if ignore is not None else list(DEFAULT_NVFP4_IGNORE),
        "quant_algo": "NVFP4",
        "quant_mode": "w4a16_nvfp4",
        "weight_only": True,
        "group_size": 16,
        "producer": {"name": "modelopt"},
    }


def resolve_quant_cfg(quant_cfg: str) -> dict[str, Any]:
    """Resolve a quantization config string into a dict consumable by ``mtq.quantize``.

    Resolution order:

    1. Built-in ModelOpt config constant exposed on ``modelopt.torch.quantization``
       (e.g. ``"NVFP4_DEFAULT_CFG"``, ``"FP8_DEFAULT_CFG"``).
    2. A ModelOpt PTQ recipe — either the name of a built-in recipe shipped under
       ``modelopt_recipes/`` (e.g. ``"general/ptq/nvfp4_default-fp8_kv"``; the
       ``.yml`` / ``.yaml`` suffix is optional) or the path to a user-authored
       YAML recipe. Resolution is performed by ``modelopt.recipe.load_config``,
       which searches the filesystem first and then the built-in recipe library.

    YAML recipes are expected to follow the standard ModelOpt PTQ recipe layout
    with a top-level ``quantize:`` section in the
    ``{"quant_cfg": [...], "algorithm": ...}`` shape that ``mtq.quantize``
    expects. A bare ``{"quant_cfg": [...], "algorithm": ...}`` document (without
    a wrapping ``quantize:`` key) is also accepted for convenience.
    The extracted dict — not the full recipe — is returned.
    """
    builtin = getattr(mtq, quant_cfg, None)
    if builtin is not None:
        return builtin

    try:
        loaded = load_config(quant_cfg)
    except (ValueError, FileNotFoundError) as e:
        raise ValueError(
            f"Unknown quant_cfg '{quant_cfg}'. Must be either a built-in "
            f"ModelOpt config name (e.g. 'NVFP4_DEFAULT_CFG'), a built-in "
            f"ModelOpt PTQ recipe name (e.g. 'general/ptq/nvfp4_default-fp8_kv'), "
            f"or a path to a YAML quantization recipe."
        ) from e

    quantize = loaded.get("quantize", loaded)
    if not isinstance(quantize, dict) or "quant_cfg" not in quantize:
        raise ValueError(
            f"Quantization recipe '{quant_cfg}' must contain a 'quant_cfg' "
            f"entry (optionally nested under a top-level 'quantize:' section)."
        )
    return quantize