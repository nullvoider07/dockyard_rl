# GDPval agentic file-producing environment — multi-turn, sandbox-backed.
#
# The faithful path of GDPval (the reduced text path is gdpval_environment.py):
# the agent drives a long-lived container, using shell/python tools to PRODUCE
# real deliverable files (xlsx/docx/pptx/pdf/...) under /workspace/deliverable.
# On completion or budget exhaustion the produced files are extracted to text
# in-container (manifest + per-file content) and graded with the same judgmental
# GDPvalRubricReward used by the text path — so file/format rubric criteria the
# text path cannot satisfy become demonstrable to the judge.
#
# Built on MultiTurnSessionEnvironment: the agent emits one ```bash block per turn
# (or TASK_COMPLETE to finish), driven by the framework's generic rollout loop.

import base64
import os
import posixpath
from typing import NotRequired, Optional, TypedDict, cast

import ray

from dockyard_rl.environments.multi_turn_session_env import (
    MultiTurnSessionEnvironment,
)
from dockyard_rl.rewards.gdpval_rubric import GDPvalRubricReward
from dockyard_rl.sandbox import SessionStartSpec, session_exec

_EXTRACTOR_PATH = os.path.join(os.path.dirname(__file__), "_gdpval_extract.py")

_DELIVERABLE_HEADER = (
    "The following is an automated extraction of the files the worker produced for "
    "this task. It begins with a MANIFEST of every produced file (name, size, type), "
    "followed by the extracted text content of each. Treat the manifest as authoritative "
    "evidence of which files/formats exist when grading file/format criteria.\n\n"
)


class GDPvalAgenticEnvConfig(TypedDict):
    # "weighted" → weighted fraction of rubric criteria met; "binary" → all-met.
    reward_mode: NotRequired[str]
    # Default per-episode turn budget when a task row omits max_turns.
    max_turns: NotRequired[int]
    # Where the agent is told to write its deliverable; extracted from here.
    deliverable_dir: NotRequired[str]
    # Where reference files (if provisioned) are written.
    reference_dir: NotRequired[str]
    # Default container image (a task row may override via metadata "image").
    image: NotRequired[str]
    # Block container network access.
    block_network: NotRequired[bool]
    # Max bytes of stdout/stderr returned per agent command observation.
    output_limit: NotRequired[int]
    # Char cap on the extracted deliverable bundle handed to the judge.
    extraction_char_limit: NotRequired[int]
    # Timeout (s) for the in-container extraction step.
    extraction_timeout: NotRequired[int]
    # Actor concurrency for servicing parallel step() calls (async rollout path).
    max_concurrency: NotRequired[int]
    # Task-executor endpoints; falls back to DOCKYARD_SANDBOX_URLS.
    sandbox_urls: NotRequired[list[str]]
    # Bearer token; falls back to DOCKYARD_SANDBOX_API_TOKEN.
    api_token: NotRequired[Optional[str]]
    # Judge config (REQUIRED — rubric grading has no deterministic fallback).
    judge_base_url: NotRequired[Optional[str]]
    judge_api_key: NotRequired[Optional[str]]
    judge_model: NotRequired[Optional[str]]
    judge_timeout: NotRequired[Optional[float]]


def _read_extractor_source() -> str:
    with open(_EXTRACTOR_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def build_extraction_command(
    deliverable_dir: str, char_limit: int, *, extractor_source: Optional[str] = None
) -> str:
    """Build a shell command that ships the extractor into the container and runs it.

    The extractor source is base64-encoded (avoids all quoting/heredoc issues),
    decoded to a temp path, and run against the deliverable directory; its stdout
    is the deliverable text bundle.
    """
    src = extractor_source if extractor_source is not None else _read_extractor_source()
    b64 = base64.b64encode(src.encode("utf-8")).decode("ascii")
    script_path = "/tmp/_gdpval_extract.py"
    return (
        f"printf %s {b64} | base64 -d > {script_path} && "
        f"python3 {script_path} {_sh_quote(deliverable_dir)} {int(char_limit)}"
    )


def _sh_quote(value: str) -> str:
    """Single-quote a value for safe shell interpolation."""
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _safe_member_name(name: str) -> str:
    """Reduce a reference-file name to a path-traversal-safe basename."""
    base = posixpath.basename(str(name).replace("\\", "/")).strip()
    base = base.lstrip(".") or "ref"
    return base.replace("'", "")


def build_write_file_command(target_dir: str, name: str, b64_content: str) -> str:
    """Build a shell command that writes a base64 file into the container dir."""
    safe = _safe_member_name(name)
    dest = posixpath.join(target_dir, safe)
    return (
        f"mkdir -p {_sh_quote(target_dir)} && "
        f"printf %s {_sh_quote(b64_content)} | base64 -d > {_sh_quote(dest)}"
    )


def compose_deliverable(extraction_stdout: str) -> str:
    """Wrap the extractor stdout as the deliverable text handed to the judge."""
    body = (extraction_stdout or "").strip()
    if not body:
        return ""
    return _DELIVERABLE_HEADER + body


class _GDPvalAgenticEnvironment(MultiTurnSessionEnvironment):
    """Multi-turn GDPval environment that grades produced files via the rubric judge.

    Implementation class (no Ray decorator) so the grading hooks are unit-testable
    on CPU. The registered actor is the thin ``GDPvalAgenticEnvironment`` subclass.
    """

    def _setup(self, cfg: dict) -> None:
        self.deliverable_dir = str(cfg.get("deliverable_dir", "/workspace/deliverable"))
        self.reference_dir = str(cfg.get("reference_dir", "/workspace/reference"))
        self.default_image = str(cfg.get("image", ""))
        self.block_network = bool(cfg.get("block_network", False))
        self.extraction_char_limit = int(cfg.get("extraction_char_limit", 200_000))
        self.extraction_timeout = int(cfg.get("extraction_timeout", 300))
        self._extractor_source = _read_extractor_source()
        self._reward = GDPvalRubricReward(
            reward_mode=self.reward_mode,
            base_url=cfg.get("judge_base_url"),
            api_key=cfg.get("judge_api_key"),
            model=cfg.get("judge_model"),
            timeout=cfg.get("judge_timeout"),
        )

    def _start_spec(self, meta: dict) -> SessionStartSpec:
        image = str(meta.get("image") or self.default_image)
        spec: SessionStartSpec = {
            "mode": "image",
            "image": image,
            "env": cast(dict, meta.get("container_env") or {}),
            "block_network": self.block_network,
            "pull_timeout": int(meta.get("pull_timeout_sec", 1800)),
        }
        return spec

    def _after_start(self, url: str, session_id: str, meta: dict) -> None:
        """Provision reference files (if any) into the container reference dir."""
        refs = meta.get("reference_files") or {}
        if not refs:
            return
        ref_dir = str(meta.get("reference_dir") or self.reference_dir)
        for name, b64_content in refs.items():
            if not b64_content:
                continue
            session_exec(
                url,
                session_id,
                build_write_file_command(ref_dir, name, str(b64_content)),
                timeout=int(meta.get("exec_timeout_sec", 120)),
                output_limit=self.output_limit,
                api_token=self.api_token,
            )

    def _finish_and_score(self, url: str, session_id: str, meta: dict) -> tuple[float, str]:
        deliverable_dir = str(meta.get("deliverable_dir") or self.deliverable_dir)
        try:
            ex = cast(dict, session_exec(
                url,
                session_id,
                build_extraction_command(
                    deliverable_dir,
                    self.extraction_char_limit,
                    extractor_source=self._extractor_source,
                ),
                timeout=self.extraction_timeout,
                output_limit=self.extraction_char_limit + 4096,
                api_token=self.api_token,
            ))
        except Exception as exc:  # noqa: BLE001
            return 0.0, f"[deliverable extraction error: {exc}]"

        deliverable = compose_deliverable(ex.get("stdout") or "")
        if not deliverable:
            return 0.0, "[no deliverable files produced]"

        outcome = self._reward({
            "deliverable": deliverable,
            "prompt": meta.get("prompt", ""),
            "rubric_json": meta.get("rubric_json", ""),
        })
        verdict = (
            "resolved"
            if outcome.reward >= 1.0 and outcome.status == "ok"
            else (outcome.failure_reason or outcome.status)
        )
        return float(outcome.reward), f"[rubric] {verdict}"


@ray.remote  # pragma: no cover
class GDPvalAgenticEnvironment(_GDPvalAgenticEnvironment):
    """Ray-actor wrapper over :class:`_GDPvalAgenticEnvironment` (registered env)."""
