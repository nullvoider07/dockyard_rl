"""Defense-in-depth for vLLM's Anthropic-router PIL repr leak (GHSA-hgg8-fqqc-vfmw).

vLLM's Anthropic-compatible HTTP router (`/v1/messages`, `/v1/messages/count_tokens`)
returns `str(e)` in its JSON error body. When an image request fails, `str(e)` can embed
a PIL object's `repr()` carrying a raw heap address
(`<PIL.Image.Image ... at 0x7f3c1a2b4d90>`), disclosing memory layout to the client (an
ASLR-bypass aid). No vLLM release patches this (vuln through 0.23.0); the upstream fix
(`sanitize_message`) does not exist in 0.21.

This reimplements that sanitization for the routes dockyard actually exposes. It is only
relevant when the optional HTTP server is mounted (`expose_http_server: true`, default
off); the RL/training path uses the in-process engine and never builds this router.

The hook is FastAPI's per-route `dependant.call` (the invocation target in
`fastapi/routing.py`). Wrapping it on dockyard's own app instance — not vLLM's module
state — is deterministic and signature-agnostic: the dependant is already solved, so the
wrapper just forwards `*args, **kwargs` and post-processes the returned response. Remove
and re-pin when vLLM ships a complete fix in a release.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Heap addresses embedded in object reprs, e.g. " at 0x7f3c1a2b4d90". Stripping this turns
# "<PIL.Image.Image image mode=RGB size=8x8 at 0x7f..>" into "<PIL.Image.Image ... size=8x8>".
_ADDR_RE = re.compile(rb" at 0x[0-9A-Fa-f]+")

# Routes whose error bodies can carry a leaked repr (the Anthropic messages router).
_TARGET_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens"})

_SENTINEL = "_dockyard_repr_sanitized"


def _sanitize_response(resp: Any) -> Any:
    """Strip heap addresses from a client-bound error response body.

    Only rewrites a rendered JSON response with a >= 400 status; success and streaming
    responses (which carry no error repr and may not have a materialized body) pass
    through untouched.
    """
    if getattr(resp, "status_code", 0) < 400:
        return resp
    body = getattr(resp, "body", None)
    if not isinstance(body, (bytes, bytearray)):
        return resp
    sanitized = _ADDR_RE.sub(b"", bytes(body))
    if sanitized != body:
        resp.body = sanitized
        # Keep Content-Length consistent with the shortened body.
        try:
            resp.headers["content-length"] = str(len(sanitized))
        except Exception:
            pass
    return resp


def install_anthropic_repr_sanitizer(app: Any) -> None:
    """Wrap the Anthropic messages routes on `app` to sanitize leaked reprs.

    Idempotent and self-contained: a no-op if the target routes are absent (e.g. a build
    that does not mount the Anthropic router). Call once, after `build_app`, before the
    server starts serving.
    """
    wrapped = 0
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) not in _TARGET_PATHS:
            continue
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        original = getattr(dependant, "call", None)
        if original is None or getattr(original, _SENTINEL, False):
            continue

        @functools.wraps(original)
        async def _guarded(*args: Any, __original=original, **kwargs: Any) -> Any:
            return _sanitize_response(await __original(*args, **kwargs))

        setattr(_guarded, _SENTINEL, True)
        dependant.call = _guarded
        wrapped += 1

    if wrapped:
        logger.info(
            "Installed Anthropic-router repr sanitizer on %d route(s) "
            "(GHSA-hgg8-fqqc-vfmw).",
            wrapped,
        )
