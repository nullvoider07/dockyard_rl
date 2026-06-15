"""dockyard_rl.models.generation — generation backends and interfaces."""

from typing import Any, cast

from transformers import PreTrainedTokenizerBase

from dockyard_rl.models.generation.interfaces import (
    GenerationConfig, GenerationDatumSpec, GenerationInterface,
    GenerationOutputSpec, verify_right_padding,
)

__all__ = [
    "GenerationConfig", "GenerationDatumSpec", "GenerationInterface",
    "GenerationOutputSpec", "verify_right_padding",
    "configure_generation_config",
]


def configure_generation_config(
    generation_config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    has_refit_draft_weights: bool = False,
) -> dict[str, Any]:
    """Finalise a ``policy.generation`` config before it reaches a backend.

    Populates the private fields the generation workers expect at runtime.
    These are ``NotRequired`` on :class:`GenerationConfig` and are this
    function's responsibility, not the caller's:

    * ``_pad_token_id`` — the tokenizer's pad token id (falling back to its
      eos token id). Read by every backend's padding logic — ``vllm_worker``,
      ``vllm_worker_async``, ``vllm_generation``, ``sglang_worker``,
      ``sglang_generation`` and ``lm_policy`` (some index it directly, so it
      must always be present).
    * ``_refit_draft_weights`` — whether the policy refits draft
      (speculative-decoding) weights. Recorded here as the single source of
      truth for the weight-sync path instead of re-deriving it from
      ``policy.draft.enabled`` at each call site.

    The ``stop_token_ids`` / ``stop_strings`` keys are also ensured to exist,
    since the vLLM worker indexes ``cfg["stop_token_ids"]`` directly.

    Args:
        generation_config:       The ``policy.generation`` config dict.
        tokenizer:               The tokenizer the policy was built with.
        has_refit_draft_weights: True when ``policy.draft.enabled`` is set.

    Returns:
        ``generation_config``, mutated in place and returned for convenience.

    Raises:
        ValueError: If the tokenizer exposes neither a pad nor an eos token id.
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError(
            "Tokenizer has neither pad_token_id nor eos_token_id; cannot "
            "populate generation_config['_pad_token_id']."
        )

    generation_config["_pad_token_id"] = int(cast(int, pad_token_id))
    generation_config["_refit_draft_weights"] = bool(has_refit_draft_weights)

    generation_config.setdefault("stop_token_ids", None)
    generation_config.setdefault("stop_strings", None)

    return generation_config