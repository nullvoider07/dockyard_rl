"""Native invalid-action / malformed-thinking detection (#2656).

Pure text-level detection over assistant-message content, composed with
environment-authoritative per-turn flags. Environments that parse actions
natively (fenced shell blocks, unified diffs, structured tool calls) stamp
``invalid_action`` / ``malformed_thinking`` booleans into the metadata they
return from ``step``; the rollout loop reads those flags per turn and falls
back to the generic detectors here for environments that do not opt in.

The verdict feeds a per-sample reward penalty (see
``algorithms.reward_functions.apply_invalid_action_penalty``); detection and
penalty arithmetic are both pure and CPU-testable.

The detection contract is protocol-agnostic: a future structured tool-call
protocol (typed tool_use blocks validated against per-tool JSON schemas)
reports through the same ``invalid_action`` / ``malformed_thinking`` channel
with no change to the penalty machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, Sequence, TypedDict

# Metadata keys environments use to report per-turn verdicts. The rollout loop
# pops these after reading so a stale flag never carries into the next turn's
# extra_env_info.
ENV_INVALID_ACTION_KEY = "invalid_action"
ENV_MALFORMED_THINKING_KEY = "malformed_thinking"

DEFAULT_THINKING_TAGS: tuple[str, str] = ("<think>", "</think>")


class InvalidActionPenaltyConfig(TypedDict):
    """Configuration for the native invalid-action reward penalty.

    Disabled (the default) is a strict no-op: no detection runs during
    rollouts and rewards are untouched.
    """

    enabled: bool

    # Reward subtracted per assistant turn flagged as an invalid action.
    invalid_action_penalty: NotRequired[float]

    # Reward subtracted per assistant turn flagged as malformed thinking.
    malformed_thinking_penalty: NotRequired[float]

    # Honor environment-stamped per-turn verdicts (tier 1). When False, only
    # the generic text detectors below run.
    use_environment_flags: NotRequired[bool]

    # Generic fallback (tier 2): substrings in assistant text that indicate an
    # unexecuted tool-call intent. Empty (the default) disables generic
    # invalid-action detection so enabling the penalty without configuring
    # patterns penalizes nothing by surprise.
    invalid_tool_call_patterns: NotRequired[list[str]]

    # Thinking tag pair checked for malformed usage (unbalanced or repeated).
    thinking_tags: NotRequired[list[str]]

    # Optional clamp on the post-penalty reward.
    reward_floor: NotRequired[float | None]


@dataclass(frozen=True)
class InvalidActionVerdict:
    """Per-assistant-turn verdict combining environment and generic tiers."""

    invalid_action: bool
    malformed_thinking: bool


def detect_invalid_tool_call(text: str, patterns: Sequence[str]) -> bool:
    """Flag assistant text containing an unexecuted tool-call pattern.

    Args:
        text: Assistant message content.
        patterns: Substrings indicating a tool-call intent the harness did not
            parse/execute. Empty sequence never flags.
    """
    return any(p in text for p in patterns if p)


# Thinking-tag styles (mirror tool_protocol.protocol; kept as literals here to
# keep this module importable without the tool_protocol package).
THINKING_STYLE_PAIRED = "paired"
THINKING_STYLE_CLOSING_ONLY = "closing_only"


def detect_malformed_thinking(
    text: str,
    thinking_tags: Sequence[str] = DEFAULT_THINKING_TAGS,
    style: str = THINKING_STYLE_PAIRED,
) -> bool:
    """Flag malformed reasoning-tag usage in assistant text.

    Two template styles:
      - ``"paired"`` (default): the model generates both tags. Malformed when the
        open/close counts are unbalanced or more than one pair appears; zero tags
        or exactly one balanced pair are well-formed.
      - ``"closing_only"``: the chat template injects the opening tag into the
        prompt, so only the closing tag appears in the output (Qwen3.5). A lone
        closing tag (or none — an implicit close before a tool call) is
        well-formed; malformed when the opening tag is regenerated in the output
        or more than one closing tag appears.

    Args:
        text: Assistant message content.
        thinking_tags: (open_tag, close_tag) pair.
        style: ``"paired"`` or ``"closing_only"``.
    """
    if len(thinking_tags) != 2:
        raise ValueError(
            f"thinking_tags must be an (open, close) pair, got {list(thinking_tags)}"
        )
    open_tag, close_tag = thinking_tags
    # Count closing tags first: with nested-substring tags like
    # "<think>"/"</think>", a naive open-tag count would also match inside the
    # closing tag.
    n_close = text.count(close_tag)
    n_open = text.replace(close_tag, "").count(open_tag)

    if style == THINKING_STYLE_CLOSING_ONLY:
        # Opening tag is prompt-supplied; the model must not regenerate it, and
        # at most one closing tag is expected. Zero closes = implicit close.
        return n_open > 0 or n_close > 1
    if style != THINKING_STYLE_PAIRED:
        raise ValueError(
            f"style must be {THINKING_STYLE_PAIRED!r} or "
            f"{THINKING_STYLE_CLOSING_ONLY!r}, got {style!r}"
        )

    if n_open == 0 and n_close == 0:
        return False
    return n_open != n_close or n_open > 1


def assess_assistant_turn(
    assistant_text: str,
    cfg: InvalidActionPenaltyConfig,
    env_metadata: dict | None,
    thinking_style: str = THINKING_STYLE_PAIRED,
) -> InvalidActionVerdict:
    """Combine environment flags (tier 1) and generic detectors (tier 2).

    Args:
        assistant_text: The assistant message content for this turn.
        cfg: Penalty config (must be enabled; callers gate on ``enabled``).
        env_metadata: The environment's returned metadata dict for this sample
            this turn, or None. Callers are responsible for popping the flag
            keys afterwards (see ``pop_env_flags``).
        thinking_style: Reasoning-tag style for the generic malformed-thinking
            detector (``"paired"`` or ``"closing_only"``); the rollout passes the
            structured-protocol style when active, else the default.
    """
    invalid = False
    malformed = False

    if cfg.get("use_environment_flags", True) and env_metadata is not None:
        invalid = bool(env_metadata.get(ENV_INVALID_ACTION_KEY, False))
        malformed = bool(env_metadata.get(ENV_MALFORMED_THINKING_KEY, False))

    patterns = cfg.get("invalid_tool_call_patterns") or []
    if not invalid and patterns:
        invalid = detect_invalid_tool_call(assistant_text, patterns)

    tags = cfg.get("thinking_tags") or list(DEFAULT_THINKING_TAGS)
    if not malformed:
        malformed = detect_malformed_thinking(assistant_text, tags, style=thinking_style)

    return InvalidActionVerdict(invalid_action=invalid, malformed_thinking=malformed)


def pop_env_flags(env_metadata: dict | None) -> None:
    """Remove per-turn verdict keys so they never persist across turns."""
    if env_metadata is None:
        return
    env_metadata.pop(ENV_INVALID_ACTION_KEY, None)
    env_metadata.pop(ENV_MALFORMED_THINKING_KEY, None)
