"""Tests for the async worker's engine-kwarg filtering and serve-arg construction.

Both surfaces are version-coupled to vLLM and had drifted undetected: the engine
filter used a hardcoded exclusion list that silently dropped settings the sync
path applies, and the HTTP bootstrap passed a dict where vLLM expects an
argparse.Namespace. Neither failure was reachable by the existing tests, so both
are pinned here against the installed vLLM rather than against a recorded shape.

No Ray, GPU or engine process is required: AsyncLLMEngine.from_engine_args is
faked, and the serve-arg builder is pure.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import cast

import pytest

vllm = pytest.importorskip("vllm")

from dockyard_rl.models.generation.vllm.config import VllmConfig  # noqa: E402
from dockyard_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)

MODEL = "Qwen/Qwen3-0.6B"


def _worker(**cfg_overrides) -> VllmAsyncGenerationWorkerImpl:
    """A worker instance with only the attributes these methods read.

    __init__ builds an engine; bypass it so the units under test stay pure.
    """
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.model_name = MODEL
    # A partial config: these units read only cfg["vllm_cfg"], so building a
    # full VllmConfig would add required keys none of them touch. cast keeps
    # that deliberate narrowing visible instead of suppressing it.
    worker.cfg = cast(VllmConfig, {"vllm_cfg": dict(cfg_overrides)})
    return worker


class TestBuildServingArgs:
    def test_returns_namespace_with_full_cli_surface(self):
        args = _worker()._build_serving_args({}, port=9001)

        assert isinstance(args, argparse.Namespace)
        # build_app/init_app_state read far beyond the keys set here; the point
        # of deriving from vLLM's parser is that the whole surface is present.
        for attr in (
            "tool_call_parser",
            "enable_auto_tool_choice",
            "chat_template",
            "response_role",
        ):
            assert hasattr(args, attr), f"{attr} missing from serve args"

    def test_pins_model_identity_and_port(self):
        args = _worker()._build_serving_args({}, port=9001)

        assert args.model == MODEL
        assert args.served_model_name == [MODEL]
        assert args.port == 9001
        assert args.host == "0.0.0.0"

    def test_serving_kwargs_override_defaults(self):
        args = _worker()._build_serving_args(
            {"response_role": "bot", "enable_auto_tool_choice": True},
            port=9001,
        )

        assert args.response_role == "bot"
        assert args.enable_auto_tool_choice is True

    def test_unknown_serving_kwarg_raises(self):
        # A typo'd serving option must not be silently attached to the
        # Namespace and then ignored by vLLM.
        with pytest.raises(ValueError, match="not accepted by the vLLM"):
            _worker()._build_serving_args({"respnse_role": "bot"}, port=9001)

    def test_accepted_by_live_build_app_signature(self):
        """The Namespace satisfies what build_app/init_app_state destructure.

        Guards the F3/F4 class of drift: a dict or ad-hoc object passes every
        assertion above and still fails at the call site.
        """
        import inspect

        from vllm.entrypoints.openai.api_server import build_app, init_app_state

        args = _worker()._build_serving_args({}, port=9001)

        build_params = inspect.signature(build_app).parameters
        assert list(build_params)[0] == "args"
        assert isinstance(args, build_params["args"].annotation)

        init_params = inspect.signature(init_app_state).parameters
        assert list(init_params)[:3] == ["engine_client", "state", "args"]
        assert isinstance(args, init_params["args"].annotation)
        # The pre-fix call site used these names; both are gone upstream.
        assert "engine" not in init_params
        assert "engine_config" not in init_params


class TestEngineKwargFiltering:
    @pytest.fixture
    def captured(self, monkeypatch):
        """Capture the AsyncEngineArgs the worker constructs."""
        box = {}

        def _fake_from_engine_args(engine_args, *a, **k):
            box["engine_args"] = engine_args
            return object()

        monkeypatch.setattr(
            vllm.AsyncLLMEngine, "from_engine_args", _fake_from_engine_args
        )
        return box

    def _run(self, captured, llm_kwargs):
        worker = _worker()
        worker._create_engine(dict(llm_kwargs))
        worker._loop.call_soon_threadsafe(worker._loop.stop)
        worker._loop_thread.join(timeout=5)
        return captured["engine_args"]

    def test_logprobs_mode_reaches_the_async_engine(self, captured):
        """Regression: this was stripped, silently desyncing sync and async runs."""
        engine_args = self._run(
            captured,
            {"model": MODEL, "logprobs_mode": "processed_logprobs"},
        )

        assert engine_args.logprobs_mode == "processed_logprobs"

    @pytest.mark.parametrize(
        "key", ["disable_log_stats", "logprobs_mode", "served_model_name"]
    )
    def test_previously_stripped_keys_are_real_fields(self, key):
        """The old exclusion list named these as unsupported; all three are fields."""
        fields = {f.name for f in dataclasses.fields(vllm.AsyncEngineArgs)}
        assert key in fields

    def test_unsupported_key_is_dropped_and_warned(self, captured, caplog):
        with caplog.at_level(
            logging.WARNING,
            logger="dockyard_rl.models.generation.vllm.vllm_worker_async",
        ):
            engine_args = self._run(
                captured,
                {"model": MODEL, "not_an_engine_arg": 1},
            )

        assert not hasattr(engine_args, "not_an_engine_arg")
        # Dropping silently is the failure this path already had once.
        assert "not_an_engine_arg" in caplog.text
