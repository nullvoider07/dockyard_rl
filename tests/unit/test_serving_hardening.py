"""Anthropic-router PIL repr sanitizer (models/generation/vllm/serving_hardening.py).

Strips leaked heap addresses ("<PIL... at 0x7f..>") from the Anthropic messages router's
JSON error bodies before they reach the client (GHSA-hgg8-fqqc-vfmw). No vLLM engine is
involved: the sanitizer operates on FastAPI's per-route dependant.call, so a stub app with
the same route paths exercises the exact wrapping mechanism.

Imports resolve via tests/conftest.py (repo parent on sys.path).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from dockyard_rl.models.generation.vllm.serving_hardening import (
    _sanitize_response,
    install_anthropic_repr_sanitizer,
)

# A representative PIL repr as it would appear inside str(e) -> the error message.
_LEAK = "<PIL.PngImagePlugin.PngImageFile image mode=RGB size=8x8 at 0x7f3c1a2b4d90>"


class TestSanitizeResponse:
    def test_strips_address_from_error_body(self):
        resp = JSONResponse(status_code=500, content={"error": {"message": _LEAK}})
        out = _sanitize_response(resp)
        body = out.body.decode()
        assert "0x7f3c1a2b4d90" not in body
        assert " at 0x" not in body
        assert "size=8x8" in body  # only the address is removed, repr is otherwise intact

    def test_content_length_kept_consistent(self):
        resp = JSONResponse(status_code=500, content={"error": {"message": _LEAK}})
        out = _sanitize_response(resp)
        assert out.headers["content-length"] == str(len(out.body))

    def test_clean_error_unchanged(self):
        resp = JSONResponse(status_code=400, content={"error": {"message": "bad request"}})
        before = resp.body
        out = _sanitize_response(resp)
        assert out.body == before

    def test_success_response_not_sanitized(self):
        # A 2xx response is passed through untouched even if it contains the pattern.
        resp = JSONResponse(status_code=200, content={"note": _LEAK})
        out = _sanitize_response(resp)
        assert b"0x7f3c1a2b4d90" in out.body


def _app_with_leaky_routes() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages():
        return JSONResponse(
            status_code=500, content={"error": {"type": "internal_error", "message": _LEAK}}
        )

    @app.post("/v1/messages/count_tokens")
    async def count_tokens():
        return JSONResponse(status_code=500, content={"error": {"message": _LEAK}})

    @app.post("/v1/chat/completions")  # not a target path -> must stay untouched
    async def chat():
        return JSONResponse(status_code=500, content={"error": {"message": _LEAK}})

    return app


class TestInstallSanitizer:
    def test_messages_route_sanitized(self):
        app = _app_with_leaky_routes()
        install_anthropic_repr_sanitizer(app)
        client = TestClient(app)
        text = client.post("/v1/messages").text
        assert "0x7f3c1a2b4d90" not in text and " at 0x" not in text
        assert "size=8x8" in text

    def test_count_tokens_route_sanitized(self):
        app = _app_with_leaky_routes()
        install_anthropic_repr_sanitizer(app)
        text = TestClient(app).post("/v1/messages/count_tokens").text
        assert " at 0x" not in text

    def test_non_target_route_untouched(self):
        app = _app_with_leaky_routes()
        install_anthropic_repr_sanitizer(app)
        text = TestClient(app).post("/v1/chat/completions").text
        assert "0x7f3c1a2b4d90" in text  # sanitizer scoped to the Anthropic router only

    def test_idempotent(self):
        app = _app_with_leaky_routes()
        install_anthropic_repr_sanitizer(app)
        install_anthropic_repr_sanitizer(app)  # second call is a no-op
        text = TestClient(app).post("/v1/messages").text
        assert " at 0x" not in text

    def test_no_target_routes_is_noop(self):
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"ok": True}

        install_anthropic_repr_sanitizer(app)  # must not raise
        assert TestClient(app).get("/health").json() == {"ok": True}
