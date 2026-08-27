"""Shape-tolerant reads for config sections.

Two config shapes reach the algorithm modules and neither is going away:
``GRPOConfig`` and its 80-odd siblings are ``TypedDict``s, which are plain
dicts at runtime, while ``opd.py`` declares its blocks as pydantic
``BaseModel``s. A reader written for one shape does not fail on the other --
``dict.get`` is simply absent on a model, so a dict-only reader returns the
caller's default for every key and the feature reads as configured while doing
nothing.

That is not hypothetical here. ``teacher_panel`` shipped with a dict-only
reader against a config the launcher hands over as a pydantic model, which
silently disabled the refusal whose only job was catching a panel on a sync
run. This module exists so the question is answered once.
"""

from __future__ import annotations

from typing import Any

__all__ = ["read_config_value"]


def read_config_value(section: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a config section of any shape.

    Args:
        section: A mapping, a pydantic ``BaseModel``, any object carrying the
            key as an attribute, or ``None``.
        key: The config key to read.
        default: Returned when ``section`` is ``None`` or does not carry ``key``.

    Returns:
        The configured value, or ``default``. ``None`` is returned as-is; a
        caller that treats an explicit null as "unset" coalesces it itself.
    """
    if section is None:
        return default
    getter = getattr(section, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(section, key, default)
