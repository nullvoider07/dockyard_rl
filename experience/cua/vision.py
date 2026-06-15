# The-Eye vision client + screenshot encoding for the CUA rollout.
#
# Observation pull (control-center backend): GET {base}/snapshot.png returns the
# latest captured frame as raw image bytes, with optional Bearer auth. For the
# official backend the screenshot bytes come straight from DesktopEnv. Both paths
# funnel through screenshot_to_image / screenshot_to_data_url so the policy's vLLM
# image input is produced identically regardless of source.

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Optional

import requests

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

_SNAPSHOT_PATH = "/snapshot.png"
_HEALTH_PATH = "/health"


class TheEyeClient:
    """Minimal client for a The-Eye frame server (Rust/axum, default :8080).

    The server exposes ``GET /snapshot.png`` (latest frame as raw image bytes)
    and ``GET /health``. Auth is an optional ``Authorization: Bearer <token>``
    header (``/health`` is always exempt server-side).
    """

    def __init__(
        self,
        base_url: str,
        api_token: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if self.api_token:
            return {"Authorization": f"Bearer {self.api_token}"}
        return {}

    def snapshot(self) -> bytes:
        """Return the latest captured frame as raw image bytes."""
        resp = requests.get(
            self.base_url + _SNAPSHOT_PATH,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.content

    def health(self) -> bool:
        try:
            resp = requests.get(self.base_url + _HEALTH_PATH, timeout=self.timeout)
            return resp.ok
        except requests.RequestException:
            return False


def screenshot_to_image(png_bytes: bytes) -> "PILImage":
    """Decode screenshot bytes into an RGB PIL image for the vLLM image path.

    Pillow is imported lazily so this module imports without it on hosts that do
    not run the multimodal path.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on image deps
        raise RuntimeError(
            "Pillow is required to encode CUA screenshots; install it in the "
            "image running the OSWorld environment/rollout."
        ) from exc
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def screenshot_to_data_url(png_bytes: bytes, mime: str = "image/png") -> str:
    """Encode screenshot bytes as a data: URL (``resolve_to_image``-compatible)."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"
