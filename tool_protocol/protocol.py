"""Structured tool-use protocol configuration contract.

Defines the config block that gates the structured protocol and the helpers
consumers (rollout loop, environments, processors, generation) use to read it.
Disabled is a strict no-op: when ``enabled`` is false the fenced-text path runs
unchanged and byte-identically.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

# Config block name under ``grpo`` (parallel to ``invalid_action_penalty``).
STRUCTURED_TOOL_USE_KEY = "structured_tool_use"

# Thinking-tag styles, matching the two Qwen3 template behaviours (see the
# qwen3_reasoning_parser and the design's fork 5):
#   - "paired":        the model generates both <think> and </think> (Qwen2.5
#                      has no thinking; older Qwen3 generates the pair).
#   - "closing_only":  the template injects <think> into the prompt, so only
#                      </think> appears in the output (Qwen3.5); a lone </think>
#                      is well-formed, and a <tool_call> before </think> is an
#                      implicit close, not a malformation.
THINKING_PAIRED = "paired"
THINKING_CLOSING_ONLY = "closing_only"
_THINKING_STYLES = (THINKING_PAIRED, THINKING_CLOSING_ONLY)


class StructuredToolUseConfig(TypedDict):
    """Configuration for the structured (Hermes) tool-use protocol.

    Disabled (the default) is a strict no-op.
    """

    enabled: bool

    # Generation backend the protocol targets. v1 scopes to vLLM.
    backend: NotRequired[str]

    # Thinking-tag style for the malformed-thinking detector (see constants).
    thinking_style: NotRequired[str]

    # Whether the rollout decodes assistant turns with skip_special_tokens
    # (the tool-call tags must survive so the env parser sees them). None = auto:
    # the rollout derives it from the tokenizer (keeps special tokens only when
    # the tokenizer marks <tool_call> special). Set True/False to override.
    decode_skip_special_tokens: NotRequired[bool | None]

    # Tool names for which constrained decoding (structural_tag) is applied.
    # Empty (default) = pure free-gen + post-hoc validation everywhere.
    constrained_tools: NotRequired[list[str]]


def resolve_structured_tool_use(
    config_block: Any,
) -> StructuredToolUseConfig | None:
    """Return the structured-tool-use config when enabled, else None.

    Args:
        config_block: The ``grpo.structured_tool_use`` mapping (or None).
    """
    if not config_block:
        return None
    if not bool(config_block.get("enabled", False)):
        return None
    style = config_block.get("thinking_style", THINKING_PAIRED)
    if style not in _THINKING_STYLES:
        raise ValueError(
            f"structured_tool_use.thinking_style must be one of {_THINKING_STYLES}, "
            f"got {style!r}"
        )
    backend = config_block.get("backend", "vllm")
    if backend != "vllm":
        raise ValueError(
            f"structured_tool_use v1 supports only the 'vllm' backend, got {backend!r}"
        )
    return config_block  # type: ignore[return-value]
