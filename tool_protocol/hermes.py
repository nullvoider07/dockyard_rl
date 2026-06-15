"""Hermes tool-call extraction (the native Qwen tool format).

Mirrors the semantics of vLLM's ``Hermes2ProToolParser.extract_tool_calls`` so a
policy trained against vLLM serving and one parsed here observe identical
structure, without importing vLLM or its OpenAI-protocol chain.

vLLM's non-streaming parser is all-or-nothing: it regex-finds every
``<tool_call>…</tool_call>`` region, ``json.loads`` each, and reads the ``name``
and ``arguments`` keys; **any** failure (bad JSON, a missing key) makes it report
``tools_called=False`` with the whole output as content. We reproduce that exactly
(so byte-for-byte parity holds) but additionally surface *why* it failed in
``HermesExtraction.error`` — the structured-tool-use penalty needs that detail to
distinguish "no tool call attempted" from "malformed tool call" at the tier-1
signal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

HERMES_TOOL_CALL_START = "<tool_call>"
HERMES_TOOL_CALL_END = "</tool_call>"

# Identical to vLLM's Hermes2ProToolParser.tool_call_regex: a complete region, or
# an unterminated trailing region (truncated generation). DOTALL so arguments may
# span newlines.
_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
)


@dataclass(frozen=True)
class ParsedToolCall:
    """A single successfully parsed Hermes tool call.

    ``arguments`` is the decoded JSON object; ``arguments_json`` is its canonical
    re-serialization (``ensure_ascii=False``), matching how vLLM populates
    ``FunctionCall.arguments`` as a JSON string.
    """

    name: str
    arguments: dict
    arguments_json: str


@dataclass(frozen=True)
class HermesExtraction:
    """Result of parsing an assistant turn for Hermes tool calls.

    ``tools_called`` and ``content`` mirror vLLM's
    ``ExtractedToolCallInformation``: on any parse failure ``tools_called`` is
    False, ``tool_calls`` is empty, and ``content`` is the full original text.
    ``error`` is None on success and on the no-tool-call case; it is a short
    reason string only when a tool call was attempted (a ``<tool_call>`` tag is
    present) but could not be parsed.
    """

    tools_called: bool
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    content: str | None = None
    error: str | None = None

    @property
    def attempted(self) -> bool:
        """True when the turn attempted a tool call (parsed or malformed).

        Distinguishes "no tool call" (content-only turn, both fields falsey)
        from a successful call (``tools_called``) or a malformed one (``error``).
        """
        return self.tools_called or self.error is not None


def parse_hermes_tool_calls(text: str) -> HermesExtraction:
    """Parse an assistant message for ``<tool_call>…</tool_call>`` blocks.

    Returns a :class:`HermesExtraction`. The ``tools_called``/``content`` fields
    are byte-for-byte equivalent to vLLM's ``Hermes2ProToolParser`` for the same
    input; ``error`` adds the failure reason for the penalty path.

    Args:
        text: The decoded assistant message content. The caller is responsible
            for ensuring the tool-call tags survive decoding (some tokenizers
            mark them special — see the design's special-token decode note).
    """
    if HERMES_TOOL_CALL_START not in text:
        return HermesExtraction(tools_called=False, tool_calls=[], content=text)

    try:
        # vLLM: findall yields (closed_region, trailing_region) tuples; exactly
        # one element of each tuple is populated.
        matches = _TOOL_CALL_REGEX.findall(text)
        raw_calls = [
            json.loads(closed if closed else trailing)
            for closed, trailing in matches
        ]
        tool_calls = [
            ParsedToolCall(
                name=call["name"],
                arguments=call["arguments"],
                arguments_json=json.dumps(call["arguments"], ensure_ascii=False),
            )
            for call in raw_calls
        ]
    except json.JSONDecodeError as exc:
        return HermesExtraction(
            tools_called=False,
            tool_calls=[],
            content=text,
            error=f"tool-call JSON decode error: {exc}",
        )
    except (KeyError, TypeError) as exc:
        # Missing "name"/"arguments", or arguments not a mapping. vLLM also fails
        # the whole extraction here (it indexes both keys unconditionally).
        return HermesExtraction(
            tools_called=False,
            tool_calls=[],
            content=text,
            error=f"tool-call missing/invalid name or arguments: {exc}",
        )

    leading = text[: text.find(HERMES_TOOL_CALL_START)]
    return HermesExtraction(
        tools_called=True,
        tool_calls=tool_calls,
        content=leading if leading else None,
        error=None,
    )
