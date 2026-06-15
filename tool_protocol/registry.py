"""Per-environment tool registry and the extraction+validation bridge.

A :class:`ToolRegistry` holds OpenAI-style function specs (the shape Qwen's chat
template consumes via its ``tools=`` argument) and is the single source of truth
for (a) what to advertise in the prompt and (b) what to validate parsed calls
against. :func:`assess_tool_calls` is the bridge environments call: it extracts
Hermes tool calls (``tool_protocol.hermes``), validates each against its tool's
schema (``tool_protocol.schema``), and returns a verdict the env uses both to
dispatch the action and to stamp the #2656 ``invalid_action`` flag.

The base tool sets here are frozen so the environment-migration and prompt
slices build against fixed schemas rather than inventing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dockyard_rl.tool_protocol.hermes import ParsedToolCall, parse_hermes_tool_calls
from dockyard_rl.tool_protocol.schema import validate_arguments


def make_tool(name: str, description: str, parameters: dict) -> dict:
    """Build an OpenAI-style function tool spec.

    ``parameters`` is a JSON-schema object (``{"type": "object", ...}``). The
    returned dict is exactly what ``tokenizer.apply_chat_template(..., tools=[…])``
    expects for a single tool.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


@dataclass(frozen=True)
class ToolRegistry:
    """An ordered, name-keyed collection of tool specs for one environment."""

    tools: tuple[dict, ...]

    def __post_init__(self) -> None:
        names = [t["function"]["name"] for t in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tool names in registry: {names}")

    def names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools]

    def __contains__(self, name: object) -> bool:
        return any(t["function"]["name"] == name for t in self.tools)

    def get(self, name: str) -> dict | None:
        for t in self.tools:
            if t["function"]["name"] == name:
                return t
        return None

    def parameters(self, name: str) -> dict | None:
        spec = self.get(name)
        return spec["function"]["parameters"] if spec else None

    def chat_template_tools(self) -> list[dict]:
        """Return the spec list for ``apply_chat_template(tools=…)``."""
        return [dict(t) for t in self.tools]


@dataclass(frozen=True)
class ToolUseAssessment:
    """Verdict for one assistant turn under the structured protocol.

    ``valid_calls`` are calls that parsed and passed schema + registry checks;
    ``malformed`` is True when a call was *attempted* but could not be turned
    into a valid, known, schema-conforming call. Whether the *absence* of a call
    is itself invalid is the environment's policy decision, not this function's.
    """

    attempted: bool
    valid_calls: list[ParsedToolCall] = field(default_factory=list)
    malformed: bool = False
    reason: str | None = None

    @property
    def has_valid_call(self) -> bool:
        return bool(self.valid_calls)


def assess_tool_calls(text: str, registry: ToolRegistry) -> ToolUseAssessment:
    """Extract and validate tool calls in an assistant turn against a registry.

    Returns a :class:`ToolUseAssessment`. A turn is ``malformed`` when it
    attempts a call (a ``<tool_call>`` tag is present) but the call cannot be
    parsed, names a tool absent from ``registry``, or has arguments that fail the
    tool's parameter schema. A turn with no ``<tool_call>`` is not malformed
    (``attempted=False``); the env decides whether that counts as invalid.

    Args:
        text: Decoded assistant message content (tool-call tags must survive
            decoding — see the design's special-token note).
        registry: The environment's tool registry.
    """
    ext = parse_hermes_tool_calls(text)

    if not ext.attempted:
        return ToolUseAssessment(attempted=False)

    if ext.error is not None:
        return ToolUseAssessment(attempted=True, malformed=True, reason=ext.error)

    reasons: list[str] = []
    valid: list[ParsedToolCall] = []
    for call in ext.tool_calls:
        if call.name not in registry:
            reasons.append(f"unknown tool {call.name!r}")
            continue
        schema = registry.parameters(call.name) or {}
        errs = validate_arguments(call.arguments, schema)
        if errs:
            reasons.append(f"{call.name}: " + "; ".join(errs))
            continue
        valid.append(call)

    malformed = bool(reasons)
    return ToolUseAssessment(
        attempted=True,
        valid_calls=valid,
        malformed=malformed,
        reason="; ".join(reasons) if reasons else None,
    )


# ── frozen base tool sets ───────────────────────────────────────────

_STR = {"type": "string"}

# Multi-turn terminal-session environments (terminal_bench / program_bench /
# gdpval_agentic). One action per turn; task_complete signals episode end.
SESSION_TOOLS = ToolRegistry(
    tools=(
        make_tool(
            "run_shell",
            "Run a single shell command in the session container and return its output.",
            {
                "type": "object",
                "properties": {
                    "command": {**_STR, "description": "The shell command to execute."}
                },
                "required": ["command"],
            },
        ),
        make_tool(
            "read_file",
            "Read a file from the session container and return its contents.",
            {
                "type": "object",
                "properties": {
                    "path": {**_STR, "description": "Absolute or workspace-relative path."}
                },
                "required": ["path"],
            },
        ),
        make_tool(
            "write_file",
            "Write content to a file in the session container, creating parent dirs.",
            {
                "type": "object",
                "properties": {
                    "path": {**_STR, "description": "Destination path."},
                    "content": {**_STR, "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
        ),
        make_tool(
            "task_complete",
            "Signal that the task is finished; the episode is then graded.",
            {"type": "object", "properties": {}},
        ),
    )
)

# Single-shot SWE coding environment: emit the unified-diff solution.
CODE_TOOLS = ToolRegistry(
    tools=(
        make_tool(
            "submit_patch",
            "Submit the unified-diff patch that solves the task.",
            {
                "type": "object",
                "properties": {
                    "patch": {**_STR, "description": "A unified diff (git format)."}
                },
                "required": ["patch"],
            },
        ),
    )
)
