# gRPC client for the control-center ControlService (:50051) — the actuation
# transport for the in-house CUA backend (Phase 2). Sends a single control-center
# command string per call via ExecuteCommand (execute scope), authenticated with
# a JWT bearer token.
#
# The generated stubs (clients/proto/control_center_pb2*.py) are produced at
# image-build time by clients/proto/generate.sh; the import is guarded so this
# module loads offline (matching how backends/official.py guards desktop_env).
#
# Env vars read here are control-center's OWN names (so they match the user's
# deployment), intentionally not DOCKYARD_-prefixed:
#   CONTROL_CENTER_TOKEN  pre-minted bearer token (canonical path)
#   CC_JWT_SECRET         HS256 secret for the mint fallback
#   JWT_AUDIENCE/JWT_ISSUER  override the server's default aud/iss for minting

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Optional

import grpc

try:
    from dockyard_rl.experience.cua.clients.proto import (  # type: ignore[import]
        control_center_pb2 as pb2,
        control_center_pb2_grpc as pb2_grpc,
    )
except ImportError:  # pragma: no cover - stubs are generated at image build
    pb2 = None  # type: ignore[assignment]
    pb2_grpc = None  # type: ignore[assignment]

logger = logging.getLogger("dockyard_rl.cua.clients.control_center")

# Control-center server defaults (crates/auth: audience/issuer; HS256).
DEFAULT_PORT = 50051
DEFAULT_JWT_AUDIENCE = "control-center-api"
DEFAULT_JWT_ISSUER = "control-center-server"

_CHANNEL_OPTIONS = [
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.keepalive_permit_without_calls", 1),
]


def _mint_token(
    *,
    secret: str,
    user: str,
    scopes: tuple[str, ...],
    audience: str,
    issuer: str,
    ttl_seconds: int,
) -> str:
    """Mint an HS256 JWT matching the control-center server's claim schema.

    Mirrors crates/auth create_jwt / crates/tools generate_token: sub/exp/iat/
    scopes/aud/iss, HS256. Requires the secret + aud + iss to match the server.
    """
    import jwt  # lazy: only needed for the mint fallback

    now = int(time.time())
    claims = {
        "sub": user,
        "iat": now,
        "exp": now + ttl_seconds,
        "scopes": list(scopes),
        "aud": audience,
        "iss": issuer,
    }
    return jwt.encode(claims, secret, algorithm="HS256")


class ControlCenterClient:
    """Thin ExecuteCommand client for control-center actuation.

    One instance drives one agent connection. ``connect`` opens the channel and
    resolves the bearer token; ``execute`` sends one command string; ``close``
    tears the channel down. Safe to use as a context manager.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        *,
        timeout: float = 30.0,
        use_ssl: bool = False,
        token: Optional[str] = None,
        jwt_secret: Optional[str] = None,
        jwt_user: str = "dockyard-cua",
        jwt_scopes: tuple[str, ...] = ("execute",),
        jwt_audience: Optional[str] = None,
        jwt_issuer: Optional[str] = None,
        jwt_ttl_seconds: int = 24 * 3600,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.use_ssl = use_ssl
        self._token_override = token
        self._jwt_secret = jwt_secret
        self._jwt_user = jwt_user
        self._jwt_scopes = jwt_scopes
        self._jwt_audience = jwt_audience
        self._jwt_issuer = jwt_issuer
        self._jwt_ttl_seconds = jwt_ttl_seconds

        self._channel: Optional[grpc.Channel] = None
        self._stub: Any = None
        self._token: Optional[str] = None

    def _resolve_token(self) -> str:
        """Token precedence: explicit override > CONTROL_CENTER_TOKEN > mint."""
        token = self._token_override or os.environ.get("CONTROL_CENTER_TOKEN")
        if token:
            return token
        secret = self._jwt_secret or os.environ.get("CC_JWT_SECRET")
        if not secret:
            raise RuntimeError(
                "No control-center token. Set CONTROL_CENTER_TOKEN (a token with "
                "'execute' scope), or pass/set CC_JWT_SECRET to mint one."
            )
        audience = (
            self._jwt_audience
            or os.environ.get("JWT_AUDIENCE")
            or DEFAULT_JWT_AUDIENCE
        )
        issuer = (
            self._jwt_issuer
            or os.environ.get("JWT_ISSUER")
            or DEFAULT_JWT_ISSUER
        )
        return _mint_token(
            secret=secret,
            user=self._jwt_user,
            scopes=self._jwt_scopes,
            audience=audience,
            issuer=issuer,
            ttl_seconds=self._jwt_ttl_seconds,
        )

    def connect(self) -> None:
        if pb2_grpc is None:
            raise RuntimeError(
                "control-center gRPC stubs are not generated. Run "
                "experience/cua/clients/proto/generate.sh (build-time "
                "grpcio-tools) — it is wired into the image build."
            )
        target = f"{self.host}:{self.port}"
        if self.use_ssl:
            self._channel = grpc.secure_channel(
                target, grpc.ssl_channel_credentials(), options=_CHANNEL_OPTIONS
            )
        else:
            self._channel = grpc.insecure_channel(target, options=_CHANNEL_OPTIONS)
        grpc.channel_ready_future(self._channel).result(timeout=self.timeout)
        self._stub = pb2_grpc.ControlServiceStub(self._channel)
        self._token = self._resolve_token()
        logger.info("control-center connected: %s", target)

    def _metadata(self) -> list[tuple[str, str]]:
        if not self._token:
            raise RuntimeError("Not connected; call connect() first.")
        return [("authorization", f"Bearer {self._token}")]

    def execute(self, command: str) -> dict[str, Any]:
        """Send one control-center command string; return the response fields."""
        if self._stub is None or pb2 is None:
            raise RuntimeError("Not connected; call connect() first.")
        request = pb2.CommandRequest(id=str(uuid.uuid4()), command=command)
        response = self._stub.ExecuteCommand(
            request, metadata=self._metadata(), timeout=self.timeout
        )
        return {
            "success": response.success,
            "message": response.message,
            "execution_time_ms": response.execution_time_ms,
            "mouse_x": response.mouse_x if response.HasField("mouse_x") else None,
            "mouse_y": response.mouse_y if response.HasField("mouse_y") else None,
            "position_captured": (
                response.position_captured
                if response.HasField("position_captured")
                else False
            ),
        }

    def close(self) -> None:
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
        self._channel = None
        self._stub = None

    def __enter__(self) -> "ControlCenterClient":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
