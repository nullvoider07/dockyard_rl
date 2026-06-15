"""Lightweight quantization config resolver usable by both vLLM workers."""

from typing import Any

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_config


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