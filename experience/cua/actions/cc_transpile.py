# pyautogui -> control-center actuation transpiler.
#
# OSWorld agents emit pyautogui code (ParsedAction.code). The control-center
# backend actuates through control-center, whose Linux agent executes raw
# xdotool shell strings (crates/agent execute_linux runs the command via sh -c;
# the Python controller LinuxActuation translates the documented command grammar
# into xdotool first). This module reproduces that path in two stages:
#
#   pyautogui_to_grammar(code)  -> [control-center grammar commands]
#   grammar_to_xdotool(grammar) -> one xdotool shell string
#
# The grammar is the portable intermediate (Phase 3 Windows swaps only the
# grammar->OS stage); grammar_to_xdotool is a faithful port of the pinned
# LinuxActuation (_build_mouse_command/_build_keyboard_command/
# _translate_modifier_keys + SPECIAL_KEYS_MAP). pyautogui_to_xdotool combines
# both. Parsing uses ast (robust to spacing/quoting); unsupported pyautogui calls
# are skipped with a warning so a single odd action never aborts a rollout.

from __future__ import annotations

import ast
import logging
from typing import Any, Optional

logger = logging.getLogger("dockyard_rl.cua.actions.cc_transpile")

# pyautogui key name -> control-center grammar token (used inside `press`).
_KEY_TO_GRAMMAR = {
    "enter": "{Enter}", "return": "{Enter}", "\n": "{Enter}",
    "tab": "{Tab}",
    "esc": "{Esc}", "escape": "{Esc}",
    "backspace": "{Backspace}",
    "delete": "{Delete}", "del": "{Delete}",
    "space": "{Space}", " ": "{Space}",
    "up": "{Up}", "down": "{Down}", "left": "{Left}", "right": "{Right}",
    "home": "{Home}", "end": "{End}",
    "pageup": "{PgUp}", "pgup": "{PgUp}", "pagedown": "{PgDn}", "pgdn": "{PgDn}",
    "insert": "{Insert}",
    "capslock": "{CapsLock}",
    "printscreen": "{PrintScreen}", "prtsc": "{PrintScreen}",
    "pause": "{Pause}",
    **{f"f{i}": f"{{F{i}}}" for i in range(1, 13)},
    # Standalone modifier keys (pressed alone, not as a hotkey prefix).
    "ctrl": "{LCtrl}", "control": "{LCtrl}", "ctrlleft": "{LCtrl}", "ctrlright": "{RCtrl}",
    "alt": "{LAlt}", "altleft": "{LAlt}", "altright": "{RAlt}", "option": "{LAlt}",
    "shift": "{LShift}", "shiftleft": "{LShift}", "shiftright": "{RShift}",
    "win": "{LWin}", "winleft": "{LWin}", "winright": "{RWin}",
    "super": "{LWin}", "command": "{LWin}", "cmd": "{LWin}", "meta": "{LWin}",
}

# pyautogui modifier key name -> grammar modifier symbol (hotkey prefix).
_MODIFIER_TO_SYMBOL = {
    "ctrl": "^", "control": "^", "ctrlleft": "^", "ctrlright": "^",
    "shift": "+", "shiftleft": "+", "shiftright": "+",
    "alt": "!", "altleft": "!", "altright": "!", "option": "!",
    "win": "#", "winleft": "#", "winright": "#", "super": "#",
    "command": "#", "cmd": "#", "meta": "#",
}

# control-center grammar special-key braces -> xdotool keysyms (from the pinned
# LinuxActuation.SPECIAL_KEYS_MAP).
_SPECIAL_KEYS_MAP = {
    "{Enter}": "Return", "{Esc}": "Escape", "{Tab}": "Tab",
    "{Backspace}": "BackSpace", "{BS}": "BackSpace",
    "{Delete}": "Delete", "{Del}": "Delete", "{Space}": "space",
    "{Up}": "Up", "{Down}": "Down", "{Left}": "Left", "{Right}": "Right",
    "{Home}": "Home", "{End}": "End", "{PgUp}": "Prior", "{PgDn}": "Next",
    **{f"{{F{i}}}": f"F{i}" for i in range(1, 13)},
    "{LWin}": "Super_L", "{RWin}": "Super_R",
    "{LCtrl}": "Control_L", "{RCtrl}": "Control_R",
    "{LAlt}": "Alt_L", "{RAlt}": "Alt_R",
    "{LShift}": "Shift_L", "{RShift}": "Shift_R",
    "{Insert}": "Insert", "{CapsLock}": "Caps_Lock", "{NumLock}": "Num_Lock",
    "{ScrollLock}": "Scroll_Lock", "{PrintScreen}": "Print", "{Pause}": "Pause",
}

_UNRESOLVED = object()


def _arg(args: list[Any], kwargs: dict[str, Any], index: int, name: str, default: Any = None) -> Any:
    """Fetch a pyautogui call argument by positional index or keyword name."""
    if index < len(args) and args[index] is not _UNRESOLVED:
        return args[index]
    return kwargs.get(name, default)


# ── pyautogui -> grammar ────────────────────────────────────────────────────

class _GrammarBuilder:
    """Translate an ordered pyautogui call stream into grammar commands.

    Tracks the last known cursor position so relative/drag actions (which
    pyautogui anchors to the current position) can be expressed in the grammar's
    absolute ``<x> <y> drag <x2> <y2>`` form.
    """

    def __init__(self, screen_size: Optional[tuple[int, int]], target_size: Optional[tuple[int, int]]):
        self._screen = screen_size
        self._target = target_size
        self._cur: Optional[tuple[int, int]] = None
        self.commands: list[str] = []

    def _scale(self, x: Any, y: Any) -> tuple[int, int]:
        sx, sy = float(x), float(y)
        if self._screen and self._target and self._screen[0] and self._screen[1]:
            sx = sx * self._target[0] / self._screen[0]
            sy = sy * self._target[1] / self._screen[1]
        xi, yi = int(round(sx)), int(round(sy))
        self._cur = (xi, yi)
        return xi, yi

    def _click(self, x: Any, y: Any, button: str, clicks: int) -> None:
        action = {"left": "left", "right": "right", "middle": "middle"}.get(button, "left")
        if x is None or y is None:
            # Click at the current position (`here`).
            base = f"here {action}" if clicks == 1 else "here double" if clicks >= 2 else f"here {action}"
            self.commands.append(base)
            for _ in range(max(0, clicks - 2) if clicks >= 2 else 0):
                self.commands.append("here left")
            return
        xi, yi = self._scale(x, y)
        if clicks <= 1:
            self.commands.append(f"{xi} {yi} {action}")
        else:
            self.commands.append(f"{xi} {yi} double")
            for _ in range(clicks - 2):
                self.commands.append("here left")

    def _move(self, x: Any, y: Any) -> None:
        xi, yi = self._scale(x, y)
        self.commands.append(f"{xi} {yi} move")

    def _drag_to(self, x: Any, y: Any) -> None:
        start = self._cur
        xi, yi = self._scale(x, y)
        if start is not None:
            self.commands.append(f"{start[0]} {start[1]} drag {xi} {yi}")
        else:
            # No known start: press-hold at current, move, release.
            self.commands += ["here hold", f"{xi} {yi} move", "here release"]

    def _drag_rel(self, dx: Any, dy: Any) -> None:
        if self._cur is None:
            logger.warning("pyautogui.drag with unknown cursor position; skipped")
            return
        self._drag_to(self._cur[0] + float(dx), self._cur[1] + float(dy))

    def _scroll(self, clicks: Any, x: Any, y: Any) -> None:
        try:
            n = int(clicks)
        except (TypeError, ValueError):
            logger.warning("pyautogui.scroll with non-numeric amount; skipped")
            return
        direction = "scroll_up" if n >= 0 else "scroll_down"
        count = abs(n) or 1
        if x is not None and y is not None:
            xi, yi = self._scale(x, y)
            self.commands.append(f"{xi} {yi} {direction} {count}")
        else:
            self.commands.append(f"here {direction} {count}")

    def _type(self, text: Any) -> None:
        if not isinstance(text, str):
            logger.warning("pyautogui write/typewrite with non-literal text; skipped")
            return
        self.commands.append(f"type {text}")

    def _press(self, keys: Any) -> None:
        seq = keys if isinstance(keys, (list, tuple)) else [keys]
        for k in seq:
            if not isinstance(k, str):
                logger.warning("pyautogui.press with non-literal key; skipped")
                continue
            token = _KEY_TO_GRAMMAR.get(k.lower(), k)
            self.commands.append(f"press {token}")

    def _hotkey(self, keys: list[Any]) -> None:
        literal = [k for k in keys if isinstance(k, str)]
        if not literal:
            logger.warning("pyautogui.hotkey with non-literal keys; skipped")
            return
        *mods, last = literal
        prefix = "".join(_MODIFIER_TO_SYMBOL.get(m.lower(), "") for m in mods)
        token = _KEY_TO_GRAMMAR.get(last.lower(), last)
        self.commands.append(f"press {prefix}{token}")

    def dispatch(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> None:
        if name in ("click", "leftClick"):
            button = _arg(args, kwargs, 4, "button", "left")
            clicks = _arg(args, kwargs, 2, "clicks", 1) or 1
            self._click(_arg(args, kwargs, 0, "x"), _arg(args, kwargs, 1, "y"), str(button), int(clicks))
        elif name == "rightClick":
            self._click(_arg(args, kwargs, 0, "x"), _arg(args, kwargs, 1, "y"), "right", 1)
        elif name == "middleClick":
            self._click(_arg(args, kwargs, 0, "x"), _arg(args, kwargs, 1, "y"), "middle", 1)
        elif name == "doubleClick":
            self._click(_arg(args, kwargs, 0, "x"), _arg(args, kwargs, 1, "y"), "left", 2)
        elif name == "tripleClick":
            self._click(_arg(args, kwargs, 0, "x"), _arg(args, kwargs, 1, "y"), "left", 3)
        elif name in ("moveTo", "move"):
            self._move(_arg(args, kwargs, 0, "x"), _arg(args, kwargs, 1, "y"))
        elif name == "dragTo":
            self._drag_to(_arg(args, kwargs, 0, "x"), _arg(args, kwargs, 1, "y"))
        elif name == "drag":
            self._drag_rel(_arg(args, kwargs, 0, "xOffset"), _arg(args, kwargs, 1, "yOffset"))
        elif name in ("scroll", "vscroll"):
            self._scroll(_arg(args, kwargs, 0, "clicks", 0), _arg(args, kwargs, 1, "x"), _arg(args, kwargs, 2, "y"))
        elif name == "mouseDown":
            x, y = _arg(args, kwargs, 0, "x"), _arg(args, kwargs, 1, "y")
            if x is not None and y is not None:
                self._move(x, y)
            self.commands.append("here hold")
        elif name == "mouseUp":
            x, y = _arg(args, kwargs, 0, "x"), _arg(args, kwargs, 1, "y")
            if x is not None and y is not None:
                self._move(x, y)
            self.commands.append("here release")
        elif name in ("write", "typewrite"):
            self._type(_arg(args, kwargs, 0, "message"))
        elif name == "press":
            self._press(_arg(args, kwargs, 0, "keys"))
        elif name == "hotkey":
            self._hotkey([a for a in args if a is not _UNRESOLVED])
        elif name in ("sleep", "PAUSE"):
            pass  # pacing is owned by the environment, not actuation
        else:
            logger.warning("unsupported pyautogui call %r; skipped", name)


def _call_info(node: ast.Call) -> Optional[tuple[str, list[Any], dict[str, Any]]]:
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return None
    args: list[Any] = []
    for a in node.args:
        if isinstance(a, ast.Starred):
            return name, [_UNRESOLVED], {}  # *args: give up on positional resolution
        try:
            args.append(ast.literal_eval(a))
        except (ValueError, SyntaxError):
            args.append(_UNRESOLVED)
    kwargs: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            kwargs[kw.arg] = _UNRESOLVED
    return name, args, kwargs


def pyautogui_to_grammar(
    code: str,
    *,
    screen_size: Optional[tuple[int, int]] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> list[str]:
    """Translate pyautogui code into control-center grammar commands.

    Args:
        code: raw pyautogui source (one or more statements).
        screen_size: resolution the model's coordinates are in (the prompt's
            advertised screen size). Required, with target_size, to scale.
        target_size: physical desktop resolution control-center actuates on.
            When either size is None, coordinates pass through unscaled.

    Returns:
        Ordered grammar command strings (possibly empty if nothing actionable).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.warning("could not parse pyautogui code; no actuation emitted")
        return []
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    calls.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    builder = _GrammarBuilder(screen_size, target_size)
    for node in calls:
        info = _call_info(node)
        if info is not None:
            builder.dispatch(*info)
    return builder.commands


# ── grammar -> xdotool (faithful port of LinuxActuation) ────────────────────

def _translate_modifier_keys(text: str) -> str:
    modifiers: list[str] = []
    i = 0
    while i < len(text) and text[i] in "^+!#":
        modifiers.append({"^": "ctrl", "+": "shift", "!": "alt", "#": "super"}[text[i]])
        i += 1
    key_part = text[i:]
    for ahk, xdo in _SPECIAL_KEYS_MAP.items():
        key_part = key_part.replace(ahk, xdo)
    if modifiers and key_part:
        return "+".join(modifiers + [key_part])
    if key_part:
        return key_part
    if modifiers:
        return "+".join(modifiers)
    return text


def _mouse_to_xdotool(parts: list[str], display: str) -> Optional[str]:
    prefix = f"DISPLAY={display} xdotool"
    if parts[0] == "position":
        return f"{prefix} getmouselocation --shell"
    if parts[0] == "here":
        if len(parts) < 2:
            return None
        action = parts[1]
        if action in ("scroll_up", "scroll_down"):
            if len(parts) < 3:
                return None
            button = "4" if action == "scroll_up" else "5"
            return f"{prefix} click --repeat {parts[2]} {button}"
        return {
            "left": f"{prefix} click 1",
            "right": f"{prefix} click 3",
            "middle": f"{prefix} click 2",
            "double": f"{prefix} click --repeat 2 1",
            "hold": f"{prefix} mousedown 1",
            "release": f"{prefix} mouseup 1",
        }.get(action)
    try:
        x, y = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        if parts[0] in ("scroll_up", "scroll_down") and len(parts) >= 2:
            button = "4" if parts[0] == "scroll_up" else "5"
            return f"{prefix} click --repeat {parts[1]} {button}"
        return None
    if len(parts) == 2:
        return f"{prefix} mousemove {x} {y}"
    action = parts[2]
    if action == "move":
        return f"{prefix} mousemove {x} {y}"
    if action == "left":
        return f"{prefix} mousemove {x} {y} click 1"
    if action == "right":
        return f"{prefix} mousemove {x} {y} click 3"
    if action == "middle":
        return f"{prefix} mousemove {x} {y} click 2"
    if action == "double":
        return f"{prefix} mousemove {x} {y} click --repeat 2 1"
    if action == "drag" and len(parts) >= 5:
        x2, y2 = int(parts[3]), int(parts[4])
        return f"{prefix} mousemove {x} {y} mousedown 1 mousemove {x2} {y2} mouseup 1"
    if action in ("scroll_up", "scroll_down"):
        count = parts[3] if len(parts) > 3 else "5"
        button = "4" if action == "scroll_up" else "5"
        return f"{prefix} mousemove {x} {y} click --repeat {count} {button}"
    return None


def _keyboard_to_xdotool(command: str, display: str) -> Optional[str]:
    prefix = f"DISPLAY={display} xdotool"
    parts = command.strip().split(maxsplit=1)
    action = parts[0]
    if action == "type":
        if len(parts) < 2:
            return None
        escaped = parts[1].replace("\\", "\\\\").replace('"', '\\"')
        return f'{prefix} type "{escaped}"'
    if action == "press":
        if len(parts) < 2:
            return None
        return f"{prefix} key {_translate_modifier_keys(parts[1])}"
    return None


_MOUSE_FIRST_TOKENS = {"here", "position", "scroll_up", "scroll_down"}


def grammar_to_xdotool(grammar: str, display: str = ":0") -> Optional[str]:
    """Translate one control-center grammar command into an xdotool shell string.

    Faithful to the pinned LinuxActuation builders. Returns None for an
    unrecognised command.
    """
    command = grammar.strip()
    if not command:
        return None
    first = command.split(maxsplit=1)[0]
    if first in ("type", "press"):
        return _keyboard_to_xdotool(command, display)
    parts = command.split()
    if first in _MOUSE_FIRST_TOKENS or first.lstrip("-").isdigit():
        return _mouse_to_xdotool(parts, display)
    return None


def pyautogui_to_xdotool(
    code: str,
    *,
    screen_size: Optional[tuple[int, int]] = None,
    target_size: Optional[tuple[int, int]] = None,
    display: str = ":0",
) -> list[str]:
    """pyautogui code -> ordered xdotool shell strings (grammar in between)."""
    out: list[str] = []
    for cmd in pyautogui_to_grammar(code, screen_size=screen_size, target_size=target_size):
        xdo = grammar_to_xdotool(cmd, display=display)
        if xdo is not None:
            out.append(xdo)
        else:
            logger.warning("grammar command %r produced no xdotool", cmd)
    return out


# ── grammar -> Windows (control-center AHK cmd-file actuation) ───────────────
#
# On Windows the control-center agent writes the command to C:\mouse_cmd.txt /
# C:\keyboard_cmd.txt; a persistent AHK v2 watcher in the guest polls those files
# and actuates. Mouse grammar is echoed as-is (the watcher parses coords/actions);
# keyboard grammar is rewritten to AHK down/up syntax because the agent passes the
# command pre-tokenized so cmd.exe never runs its '^' escape pass — modifier '^'
# must be removed entirely (faithful to the pinned LinuxActuation's Windows twin
# windows_actuation.py: _process_keyboard_command + _convert_modifiers_to_explicit).

_WIN_STANDALONE_MODIFIER = {"#": "{LWin}", "^": "{LCtrl}", "+": "{LShift}", "!": "{LAlt}"}
_WIN_MODIFIER_DOWNUP = {
    "^": ("{Ctrl down}", "{Ctrl up}"),
    "+": ("{Shift down}", "{Shift up}"),
    "!": ("{Alt down}", "{Alt up}"),
    "#": ("{LWin down}", "{LWin up}"),
}


def _win_convert_modifiers_to_explicit(keys: str) -> str:
    """``^+{Esc}`` -> ``{Ctrl down}{Shift down}{Esc}{Shift up}{Ctrl up}``; a key
    with no modifier prefix (``{F5}``, ``{LCtrl}``) is unchanged."""
    down: list[str] = []
    up: list[str] = []
    i = 0
    while i < len(keys) and keys[i] in _WIN_MODIFIER_DOWNUP:
        d, u = _WIN_MODIFIER_DOWNUP[keys[i]]
        down.append(d)
        up.insert(0, u)  # reverse order: last pressed, first released
        i += 1
    if not down:
        return keys
    return "".join(down) + keys[i:] + "".join(up)


def _win_keyboard_payload(command: str) -> Optional[str]:
    parts = command.strip().split(maxsplit=1)
    action = parts[0]
    if action == "type":
        if len(parts) < 2:
            return None
        text = parts[1]
        # A literal caret cannot survive the echo, so type it via its AHK code.
        if "^" in text:
            return f"press {text.replace('^', '{U+005E}')}"
        return f"type {text}"
    if action == "press":
        if len(parts) < 2:
            return None
        keys = _WIN_STANDALONE_MODIFIER.get(parts[1], parts[1])
        return f"press {_win_convert_modifiers_to_explicit(keys)}"
    return None


def grammar_to_windows(grammar: str) -> Optional[str]:
    """Translate one control-center grammar command into the Windows cmd that
    writes it to the AHK watcher's command file. None for an unrecognised one."""
    command = grammar.strip()
    if not command:
        return None
    first = command.split(maxsplit=1)[0]
    if first in ("type", "press"):
        payload = _win_keyboard_payload(command)
        return f"cmd /c echo {payload} > C:\\keyboard_cmd.txt" if payload else None
    if first in _MOUSE_FIRST_TOKENS or first.lstrip("-").isdigit():
        return f"cmd /c echo {command} > C:\\mouse_cmd.txt"
    return None


def pyautogui_to_windows(
    code: str,
    *,
    screen_size: Optional[tuple[int, int]] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> list[str]:
    """pyautogui code -> ordered Windows control-center cmd strings."""
    out: list[str] = []
    for cmd in pyautogui_to_grammar(code, screen_size=screen_size, target_size=target_size):
        win = grammar_to_windows(cmd)
        if win is not None:
            out.append(win)
        else:
            logger.warning("grammar command %r produced no windows cmd", cmd)
    return out


# ── grammar -> macOS (control-center cliclick/osascript actuation) ───────────
#
# On macOS the control-center controller (MacOSActuation) translates the grammar
# into a cliclick / osascript shell string and sends THAT over gRPC; the agent's
# execute_macos runs a pure cliclick command directly and anything containing
# `&&` / starting with `osascript` / a quoted `t:"` via `sh -c`. Unlike Windows
# there is no cmd-file/AHK watcher — this mirrors the Linux path (grammar -> shell
# string). Faithful port of the pinned control-center
# crates/controller/os_specific/macos_actuation.py
# (build_mouse_command / build_scroll_command / parse_keyboard_command); the
# grammar tokens this transpiler emits are well-formed, so the controller's
# detect_command_type heuristics are bypassed by dispatching on the first token.

_MAC_CLICLICK = "cliclick"

# grammar special-key brace -> cliclick key name (cliclick `kp:` set).
_MAC_SPECIAL_KEYS = {
    **{f"{{F{i}}}": f"f{i}" for i in range(1, 17)},
    "{Enter}": "return", "{Return}": "return",
    "{Tab}": "tab", "{Esc}": "esc", "{Escape}": "esc", "{Space}": "space",
    "{Backspace}": "delete", "{BS}": "delete",
    "{Delete}": "fwd-delete", "{Del}": "fwd-delete",
    "{Up}": "arrow-up", "{Down}": "arrow-down",
    "{Left}": "arrow-left", "{Right}": "arrow-right",
    "{Home}": "home", "{End}": "end",
    "{PgUp}": "page-up", "{PgDn}": "page-down",
    "{VolumeUp}": "volume-up", "{VolumeDown}": "volume-down", "{Mute}": "mute",
    "{BrightnessUp}": "brightness-up", "{BrightnessDown}": "brightness-down",
    "{PlayPause}": "play-pause",
}

# grammar modifier symbol (and Unicode/brace aliases) -> cliclick modifier name.
_MAC_MODIFIER_MAP = {
    "^": "ctrl", "⌃": "ctrl", "+": "shift", "⇧": "shift",
    "!": "alt", "⌥": "alt", "#": "cmd", "⌘": "cmd",
    "{Cmd}": "cmd", "{Command}": "cmd", "{Option}": "alt", "{Alt}": "alt",
    "{Control}": "ctrl", "{Ctrl}": "ctrl", "{Shift}": "shift", "{Fn}": "fn",
}

# cliclick key name -> AppleScript key code (keys cliclick `kp:` cannot reach).
_MAC_OSASCRIPT_KEYCODES = {
    "return": 36, "tab": 48, "delete": 51, "fwd-delete": 117,
    "page-up": 116, "page-down": 121, "home": 115, "end": 119,
    "arrow-left": 123, "arrow-right": 124, "arrow-down": 125, "arrow-up": 126,
}
_MAC_OSASCRIPT_MODS = {
    "cmd": "command down", "alt": "option down",
    "ctrl": "control down", "shift": "shift down",
}
# grammar scroll direction -> AppleScript arrow key code (scroll via key repeat).
_MAC_SCROLL_KEYCODE = {
    "scroll_up": 126, "scroll-up": 126, "scrollup": 126,
    "scroll_down": 125, "scroll-down": 125, "scrolldown": 125,
    "scroll_left": 123, "scroll-left": 123, "scrollleft": 123,
    "scroll_right": 124, "scroll-right": 124, "scrollright": 124,
}


def _mac_build_scroll(
    x: Optional[int], y: Optional[int], direction: str, amount: int
) -> Optional[str]:
    key_code = _MAC_SCROLL_KEYCODE.get(direction)
    if key_code is None:
        return None
    focus = (
        f"{_MAC_CLICLICK} c:{x},{y} w:50"
        if x is not None and y is not None
        else f"{_MAC_CLICLICK} c:. w:50"
    )
    scroll = (
        "osascript -e 'tell application \"System Events\" to "
        f"repeat {amount} times' -e 'key code {key_code}' "
        "-e 'delay 0.02' -e 'end repeat'"
    )
    return f"{focus} && {scroll}"


def _mac_build_mouse(parts: list[str]) -> Optional[str]:
    if not parts:
        return None
    cli = _MAC_CLICLICK

    if parts[0] == "position":
        return f"{cli} p:."

    # "here <action> [amount]"
    if parts[0] == "here":
        action = parts[1] if len(parts) > 1 else "left"
        simple = {
            "left": "c:.", "click": "c:.", "right": "rc:.", "double": "dc:.",
            "triple": "tc:.", "middle": "mc:.", "hold": "dd:.", "release": "du:.",
        }.get(action)
        if simple is not None:
            return f"{cli} {simple}"
        if "scroll" in action:
            amount = int(parts[2]) if len(parts) > 2 else 5
            return _mac_build_scroll(None, None, action, amount)
        return None

    # "<x> <y> <action> [amount]"
    try:
        x, y = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        x = y = None
    if x is not None and y is not None:
        if len(parts) == 2:
            return f"{cli} m:{x},{y}"
        action = parts[2]
        coord = {
            "move": f"m:{x},{y}", "left": f"c:{x},{y}", "click": f"c:{x},{y}",
            "right": f"rc:{x},{y}", "double": f"dc:{x},{y}", "triple": f"tc:{x},{y}",
            "middle": f"mc:{x},{y}", "hold": f"dd:{x},{y}", "release": f"du:{x},{y}",
        }.get(action)
        if coord is not None:
            return f"{cli} {coord}"
        if action == "drag" and len(parts) >= 5:
            x2, y2 = int(parts[3]), int(parts[4])
            return f"{cli} dd:{x},{y} w:50 m:{x2},{y2} w:50 du:{x2},{y2}"
        if "scroll" in action:
            amount = int(parts[3]) if len(parts) > 3 else 5
            return _mac_build_scroll(x, y, action, amount)
    return None


def _mac_parse_keyboard(command: str) -> Optional[str]:
    parts = command.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None
    action, text = parts[0], parts[1]
    cli = _MAC_CLICLICK

    if action == "type":
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        escaped_sh = escaped.replace("'", "'\"'\"'")
        return (
            "osascript -e 'tell application \"System Events\" to keystroke \""
            + escaped_sh
            + "\"'"
        )

    if action not in ("press", "key"):
        return None

    modifiers: list[str] = []
    main_key = text
    i = 0
    while i < len(text):
        if text[i : i + 1] in _MAC_MODIFIER_MAP:
            mod = _MAC_MODIFIER_MAP[text[i]]
            if mod not in modifiers:
                modifiers.append(mod)
            i += 1
        elif text[i : i + 2] in ("⌘", "⌥", "⌃", "⇧"):
            mod = _MAC_MODIFIER_MAP[text[i : i + 2]]
            if mod not in modifiers:
                modifiers.append(mod)
            i += 2
        else:
            main_key = text[i:]
            break

    normalized_key = _MAC_SPECIAL_KEYS.get(main_key, main_key)
    mod_str = ",".join(modifiers)

    if normalized_key in _MAC_OSASCRIPT_KEYCODES:
        code = _MAC_OSASCRIPT_KEYCODES[normalized_key]
        cmd = f"osascript -e 'tell application \"System Events\" to key code {code}"
        mod_list = [_MAC_OSASCRIPT_MODS[m] for m in modifiers if m in _MAC_OSASCRIPT_MODS]
        if mod_list:
            cmd += " using {" + ", ".join(mod_list) + "}"
        return cmd + "'"

    if main_key in _MAC_SPECIAL_KEYS:
        key_code = _MAC_SPECIAL_KEYS[main_key]
        if modifiers:
            return f"{cli} kd:{mod_str} kp:{key_code} ku:{mod_str}"
        return f"{cli} kp:{key_code}"

    if len(main_key) == 1:
        if modifiers:
            return f"{cli} kd:{mod_str} t:{main_key} ku:{mod_str}"
        return f"{cli} t:{main_key}"

    if main_key.lower() == "space":
        if modifiers:
            return f"{cli} kd:{mod_str} kp:space ku:{mod_str}"
        return f"{cli} kp:space"

    if not main_key and modifiers:
        return f"{cli} kd:{mod_str} w:50 ku:{mod_str}"

    if main_key:
        if modifiers:
            return f"{cli} kd:{mod_str} t:{main_key} ku:{mod_str}"
        return f"{cli} t:{main_key}"
    return None


def grammar_to_macos(grammar: str) -> Optional[str]:
    """Translate one control-center grammar command into a macOS cliclick /
    osascript shell string. None for an unrecognised command."""
    command = grammar.strip()
    if not command:
        return None
    first = command.split(maxsplit=1)[0]
    if first in ("type", "press", "key"):
        return _mac_parse_keyboard(command)
    parts = command.split()
    if first in _MOUSE_FIRST_TOKENS or first.lstrip("-").isdigit():
        return _mac_build_mouse(parts)
    return None


def pyautogui_to_macos(
    code: str,
    *,
    screen_size: Optional[tuple[int, int]] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> list[str]:
    """pyautogui code -> ordered macOS control-center cliclick/osascript strings."""
    out: list[str] = []
    for cmd in pyautogui_to_grammar(code, screen_size=screen_size, target_size=target_size):
        mac = grammar_to_macos(cmd)
        if mac is not None:
            out.append(mac)
        else:
            logger.warning("grammar command %r produced no macos cmd", cmd)
    return out
