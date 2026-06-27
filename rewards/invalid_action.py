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
from typing import Any, NotRequired, Optional, Sequence, TypedDict

# Metadata keys environments use to report per-turn verdicts. The rollout loop
# pops these after reading so a stale flag never carries into the next turn's
# extra_env_info.
ENV_INVALID_ACTION_KEY = "invalid_action"
ENV_MALFORMED_THINKING_KEY = "malformed_thinking"

DEFAULT_THINKING_TAGS: tuple[str, str] = ("<think>", "</think>")

# --- Typed / graded violations (N4) ---------------------------------------
# Application locus for a violation's penalty. "advantage" overwrites the
# advantage of the offending assistant-message token span (surgical, token-local
# credit assignment); "reward" subtracts from the per-sample reward (a
# whole-trajectory, task-semantic signal). penalty_mode="auto" applies each
# violation at its own default locus exactly once (no double-count); "reward" /
# "advantage" force all to one locus; "both" stacks (documented, not default).
LOCUS_REWARD = "reward"
LOCUS_ADVANTAGE = "advantage"

# Violation type identifiers.
VIOLATION_UNEXECUTED_PATTERN = "unexecuted_pattern"   # tier-2 pattern match
VIOLATION_MALFORMED_TOOL_CALL = "malformed_tool_call"  # schema: parse failure
VIOLATION_SCHEMA_UNKNOWN_TOOL = "schema_unknown_tool"  # schema: tool not in registry
VIOLATION_SCHEMA_ARGS = "schema_args"                  # schema: argument validation failed
VIOLATION_MALFORMED_THINKING = "malformed_thinking"    # reasoning-tag misuse
VIOLATION_ENV_INVALID_ACTION = "env_invalid_action"    # env-authoritative task-semantic
VIOLATION_ENV_MALFORMED_THINKING = "env_malformed_thinking"

# Default per-type locus: structural/format violations are token-local (advantage);
# the env's task-semantic invalid-action verdict is whole-trajectory (reward).
_DEFAULT_LOCUS: dict[str, str] = {
    VIOLATION_UNEXECUTED_PATTERN: LOCUS_ADVANTAGE,
    VIOLATION_MALFORMED_TOOL_CALL: LOCUS_ADVANTAGE,
    VIOLATION_SCHEMA_UNKNOWN_TOOL: LOCUS_ADVANTAGE,
    VIOLATION_SCHEMA_ARGS: LOCUS_ADVANTAGE,
    VIOLATION_MALFORMED_THINKING: LOCUS_ADVANTAGE,
    VIOLATION_ENV_MALFORMED_THINKING: LOCUS_ADVANTAGE,
    VIOLATION_ENV_INVALID_ACTION: LOCUS_REWARD,
}

# Default per-type severity (penalty weight multiplier). Graded so a soft
# argument-schema slip weighs less than an unparseable or unknown-tool call.
_DEFAULT_SEVERITY: dict[str, float] = {
    VIOLATION_UNEXECUTED_PATTERN: 1.0,
    VIOLATION_MALFORMED_TOOL_CALL: 1.0,
    VIOLATION_SCHEMA_UNKNOWN_TOOL: 1.0,
    VIOLATION_SCHEMA_ARGS: 0.5,
    VIOLATION_MALFORMED_THINKING: 0.5,
    VIOLATION_ENV_MALFORMED_THINKING: 0.5,
    VIOLATION_ENV_INVALID_ACTION: 1.0,
}

# Violation types that count toward the back-compat invalid_action / malformed_thinking
# bools (and the existing per-sample count fields).
_INVALID_ACTION_TYPES = frozenset({
    VIOLATION_UNEXECUTED_PATTERN,
    VIOLATION_MALFORMED_TOOL_CALL,
    VIOLATION_SCHEMA_UNKNOWN_TOOL,
    VIOLATION_SCHEMA_ARGS,
    VIOLATION_ENV_INVALID_ACTION,
})
_MALFORMED_THINKING_TYPES = frozenset({
    VIOLATION_MALFORMED_THINKING,
    VIOLATION_ENV_MALFORMED_THINKING,
})


@dataclass(frozen=True)
class Violation:
    """One typed, graded violation flagged on an assistant turn.

    ``severity`` is the penalty weight multiplier; ``locus`` is where its penalty
    is applied under ``penalty_mode="auto"`` (``LOCUS_REWARD`` / ``LOCUS_ADVANTAGE``).
    ``detail`` is an optional human-readable reason (e.g. the schema error string).
    """

    type: str
    severity: float
    locus: str
    detail: Optional[str] = None


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

    # --- Typed / graded penalties + application locus (N) ---
    # How penalties are applied:
    #   "auto" (default) — each violation at its own default locus, once (no double-count)
    #   "reward"        — force every violation to the per-sample reward penalty (legacy)
    #   "advantage"     — force every violation to the advantage-span penalty
    #   "both"          — apply at both loci (stacks; not recommended)
    penalty_mode: NotRequired[str]

    # Per-violation-type severity multipliers, merged over _DEFAULT_SEVERITY.
    severity_weights: NotRequired[dict[str, float]]

    # Per-violation-type locus overrides ("reward"|"advantage"), merged over
    # _DEFAULT_LOCUS (only consulted under penalty_mode="auto").
    locus_overrides: NotRequired[dict[str, str]]

    # Optional structured tool-call schema validation tier (N3). When the rollout
    # supplies a tool registry, a parsed call that names an unknown tool or fails
    # its parameter schema becomes a schema_* violation. Off when no registry.
    enable_schema_validation: NotRequired[bool]

    # Optional step-level penalty weight schedule (N): scales every violation's
    # severity by a factor that may vary with the training step. Absent → 1.0.
    #   {"schedule": "constant"|"linear", "start": float, "end": float, "steps": int}
    penalty_step_scale: NotRequired[dict[str, Any]]


@dataclass(frozen=True)
class InvalidActionVerdict:
    """Per-assistant-turn verdict combining environment and generic tiers.

    ``invalid_action`` / ``malformed_thinking`` are retained for back-compat with
    the per-sample count path; ``violations`` carries the typed, graded, locus-tagged
    detail used by the graded reward and advantage-span penalties.
    """

    invalid_action: bool
    malformed_thinking: bool
    violations: tuple[Violation, ...] = ()

    @property
    def has_violation(self) -> bool:
        return bool(self.violations)


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


def _resolve_severity(cfg: InvalidActionPenaltyConfig, vtype: str) -> float:
    weights = cfg.get("severity_weights") or {}
    return float(weights.get(vtype, _DEFAULT_SEVERITY.get(vtype, 1.0)))


def _resolve_locus(cfg: InvalidActionPenaltyConfig, vtype: str) -> str:
    overrides = cfg.get("locus_overrides") or {}
    return overrides.get(vtype, _DEFAULT_LOCUS.get(vtype, LOCUS_REWARD))


def _make_violation(
    cfg: InvalidActionPenaltyConfig, vtype: str, detail: Optional[str] = None
) -> Violation:
    return Violation(
        type=vtype,
        severity=_resolve_severity(cfg, vtype),
        locus=_resolve_locus(cfg, vtype),
        detail=detail,
    )


def penalty_scale_at_step(
    cfg: InvalidActionPenaltyConfig, step: Optional[int]
) -> float:
    """Step-level penalty weight scale. Identity (1.0) unless a schedule is set.

    ``penalty_step_scale`` schedule dict:
    ``{"schedule": "constant"|"linear", "start": s, "end": e, "steps": n}``.
    ``"constant"`` returns ``start``; ``"linear"`` interpolates ``start``→``end``
    over ``[0, steps]`` and holds at ``end`` thereafter. ``step is None`` → ``start``.
    """
    sched = cfg.get("penalty_step_scale")
    if not sched:
        return 1.0
    start = float(sched.get("start", 1.0))
    if step is None:
        return start
    kind = sched.get("schedule", "constant")
    if kind == "constant":
        return start
    if kind == "linear":
        end = float(sched.get("end", start))
        n = int(sched.get("steps", 0))
        if n <= 0:
            return end
        frac = min(max(step / n, 0.0), 1.0)
        return start + (end - start) * frac
    raise ValueError(f"Unknown penalty_step_scale schedule {kind!r}")


def _classify_schema_failure(reason: str) -> str:
    """Map a tool_protocol assessment reason to a graded schema violation type."""
    if reason.startswith("unknown tool") or "unknown tool" in reason:
        return VIOLATION_SCHEMA_UNKNOWN_TOOL
    # assess_tool_calls reports per-tool argument errors as "<tool>: <errors>".
    head = reason.split(";", 1)[0]
    if ": " in head and not head.lower().startswith(("invalid", "could not", "no ")):
        return VIOLATION_SCHEMA_ARGS
    return VIOLATION_MALFORMED_TOOL_CALL


def assess_assistant_turn(
    assistant_text: str,
    cfg: InvalidActionPenaltyConfig,
    env_metadata: dict | None,
    thinking_style: str = THINKING_STYLE_PAIRED,
    registry: Any = None,
) -> InvalidActionVerdict:
    """Combine environment flags (tier 1), schema validation (tier 2, optional),
    and generic text detectors (tier 3) into a typed, graded verdict.

    Args:
        assistant_text: The assistant message content for this turn.
        cfg: Penalty config (must be enabled; callers gate on ``enabled``).
        env_metadata: The environment's returned metadata dict for this sample
            this turn, or None. Callers are responsible for popping the flag
            keys afterwards (see ``pop_env_flags``).
        thinking_style: Reasoning-tag style for the generic malformed-thinking
            detector (``"paired"`` or ``"closing_only"``); the rollout passes the
            structured-protocol style when active, else the default.
        registry: Optional ``tool_protocol.ToolRegistry``. When supplied and
            ``enable_schema_validation`` is set, a parsed tool call that names an
            unknown tool or fails its parameter schema becomes a graded
            ``schema_*`` / ``malformed_tool_call`` violation.

    The back-compat ``invalid_action`` / ``malformed_thinking`` bools are derived
    from the typed ``violations`` so the existing per-sample count path is
    unchanged.
    """
    violations: list[Violation] = []

    env_invalid = False
    env_malformed = False
    if cfg.get("use_environment_flags", True) and env_metadata is not None:
        env_invalid = bool(env_metadata.get(ENV_INVALID_ACTION_KEY, False))
        env_malformed = bool(env_metadata.get(ENV_MALFORMED_THINKING_KEY, False))
    if env_invalid:
        violations.append(_make_violation(cfg, VIOLATION_ENV_INVALID_ACTION))
    if env_malformed:
        violations.append(_make_violation(cfg, VIOLATION_ENV_MALFORMED_THINKING))

    # Invalid-action detection: env flag takes priority; otherwise the structured
    # schema tier (when a registry is supplied), then the generic pattern fallback.
    invalid_detected = env_invalid
    if (
        not invalid_detected
        and registry is not None
        and cfg.get("enable_schema_validation", True)
    ):
        from dockyard_rl.tool_protocol.registry import assess_tool_calls

        assessment = assess_tool_calls(assistant_text, registry)
        if assessment.attempted and assessment.malformed:
            reason = assessment.reason or ""
            violations.append(
                _make_violation(
                    cfg, _classify_schema_failure(reason), detail=reason or None
                )
            )
            invalid_detected = True

    patterns = cfg.get("invalid_tool_call_patterns") or []
    if (
        not invalid_detected
        and patterns
        and detect_invalid_tool_call(assistant_text, patterns)
    ):
        violations.append(_make_violation(cfg, VIOLATION_UNEXECUTED_PATTERN))

    # Malformed thinking: env flag priority, else the generic tag detector.
    if not env_malformed:
        tags = cfg.get("thinking_tags") or list(DEFAULT_THINKING_TAGS)
        if detect_malformed_thinking(assistant_text, tags, style=thinking_style):
            violations.append(_make_violation(cfg, VIOLATION_MALFORMED_THINKING))

    invalid = any(v.type in _INVALID_ACTION_TYPES for v in violations)
    malformed = any(v.type in _MALFORMED_THINKING_TYPES for v in violations)
    return InvalidActionVerdict(
        invalid_action=invalid,
        malformed_thinking=malformed,
        violations=tuple(violations),
    )


def pop_env_flags(env_metadata: dict | None) -> None:
    """Remove per-turn verdict keys so they never persist across turns."""
    if env_metadata is None:
        return
    env_metadata.pop(ENV_INVALID_ACTION_KEY, None)
    env_metadata.pop(ENV_MALFORMED_THINKING_KEY, None)
