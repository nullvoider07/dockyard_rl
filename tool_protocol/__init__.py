"""Structured tool-use protocol primitives.

Native Hermes tool-call extraction (the format Qwen emits) and a dependency-free
JSON-schema validator for tool-call arguments. These are pure, CPU-only building
blocks; the per-environment tool registry and the rollout/penalty wiring layer on
top of them (see ``tool_protocol.registry`` and ``rewards.invalid_action``).

The extractor mirrors vLLM's ``Hermes2ProToolParser.extract_tool_calls``
(``<tool_call>{json}</tool_call>``) so a model trained against vLLM serving and a
model parsed here see identical structure, but it carries none of vLLM's
OpenAI-protocol dependencies and reports per-call parse detail the penalty path
needs.
"""

from dockyard_rl.tool_protocol.hermes import (
    HERMES_TOOL_CALL_END,
    HERMES_TOOL_CALL_START,
    HermesExtraction,
    ParsedToolCall,
    parse_hermes_tool_calls,
)
from dockyard_rl.tool_protocol.protocol import (
    STRUCTURED_TOOL_USE_KEY,
    THINKING_CLOSING_ONLY,
    THINKING_PAIRED,
    StructuredToolUseConfig,
    resolve_structured_tool_use,
)
from dockyard_rl.tool_protocol.registry import (
    CODE_TOOLS,
    SESSION_TOOLS,
    ToolRegistry,
    ToolUseAssessment,
    assess_tool_calls,
    make_tool,
)
from dockyard_rl.tool_protocol.schema import (
    SchemaError,
    validate_arguments,
    validate_against_schema,
)

__all__ = [
    "HERMES_TOOL_CALL_START",
    "HERMES_TOOL_CALL_END",
    "HermesExtraction",
    "ParsedToolCall",
    "parse_hermes_tool_calls",
    "SchemaError",
    "validate_arguments",
    "validate_against_schema",
    "make_tool",
    "ToolRegistry",
    "ToolUseAssessment",
    "assess_tool_calls",
    "SESSION_TOOLS",
    "CODE_TOOLS",
    "STRUCTURED_TOOL_USE_KEY",
    "THINKING_PAIRED",
    "THINKING_CLOSING_ONLY",
    "StructuredToolUseConfig",
    "resolve_structured_tool_use",
]
