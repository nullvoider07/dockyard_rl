# Parse a CUA assistant turn into a normalized OSWorld action.
#
# OSWorld's pyautogui action space is a Python code string executed in the guest,
# plus the bare control tokens WAIT / FAIL / DONE. The policy emits a fenced
# ```python block of pyautogui calls and/or one of the control tokens; this
# module normalizes both into a ParsedAction the backends execute uniformly. A
# turn may carry both a code block and a terminal token (run the code, then
# grade), mirroring the (command, done) split in multi_turn_session_env.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

Control = Literal["wait", "fail", "done"]

# A fenced python/pyautogui block; the language tag is optional.
_CODE_FENCE_RE = re.compile(r"```(?:python|py|pyautogui)?\s*\n(.*?)```", re.DOTALL)
# Bare control tokens, each on its own line (OSWorld is case-sensitive here).
_TOKEN_RE = re.compile(r"(?m)^\s*(WAIT|FAIL|DONE)\s*$")


@dataclass
class ParsedAction:
    """A normalized single-turn desktop action.

    code:    pyautogui code to execute, or "" when the turn carried no fenced
             block.
    control: a terminal/pause control token (wait/fail/done) if present, else
             None. fail/done end the episode; wait is a no-op pause.
    raw:     the original assistant text (kept for logging/debugging).
    """

    code: str
    control: Optional[Control]
    raw: str

    @property
    def is_terminal(self) -> bool:
        return self.control in ("fail", "done")

    @property
    def is_noop(self) -> bool:
        """True when the turn produced nothing actionable (no code, no token)."""
        return not self.code and self.control is None


def parse_action(text: str) -> ParsedAction:
    """Normalize an assistant message into a ParsedAction.

    Precedence: the first fenced code block is the executable body; the first
    control token (WAIT/FAIL/DONE) sets ``control``. Both may be present.
    """
    m = _CODE_FENCE_RE.search(text)
    code = m.group(1).strip() if m else ""

    control: Optional[Control] = None
    tok = _TOKEN_RE.search(text)
    if tok:
        control = tok.group(1).lower()  # type: ignore[assignment]

    return ParsedAction(code=code, control=control, raw=text)
