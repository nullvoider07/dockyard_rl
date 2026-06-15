# Shared base for multi-turn terminal-session environments scored via the
# ubuntu-swe task-executor session API.
#
# The agent drives a long-lived per-episode container across turns: each turn it
# issues one shell command, which this environment execs (executor /session/exec)
# and returns as the next observation. On the agent's completion signal
# (TASK_COMPLETE) or the last allowed turn, the episode is graded and terminated.
#
# The framework's generic rollout loop (experience/rollouts.py) drives this: it
# calls step() each turn, tokenizes the returned observation as the next user
# message, and threads the returned metadata forward as extra_env_info — the
# channel that carries per-episode session state (session id, executor URL, turn
# counter) across turns. Reward accrues only through step(), so a concrete
# environment self-terminates with its grading score on the last allowed turn;
# keep grpo.max_rollout_turns above the per-task max_turns.
#
# Subclasses implement two hooks: _start_spec(meta) (how to start the session)
# and _finish_and_score(url, sid, meta) -> (reward, verdict). _setup(cfg) lets a
# subclass read its own config and build its reward.

import base64
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import torch

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from dockyard_rl.sandbox import (
    SessionStartSpec,
    delete_session,
    session_exec,
    start_session,
)
from dockyard_rl.tool_protocol import SESSION_TOOLS, assess_tool_calls

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dockyard_rl.data.interfaces import LLMMessageLogType
else:
    try:
        from dockyard_rl.data.interfaces import LLMMessageLogType
    except ImportError:
        LLMMessageLogType = list  # type: ignore


# One shell command per turn, inside a fenced block; completion sentinel.
_CMD_FENCE_RE = re.compile(r"```(?:bash|sh|shell|console)?\s*\n(.*?)```", re.DOTALL)
_DONE_RE = re.compile(r"(?m)^\s*TASK_COMPLETE\s*$")


def _parse_action(text: str) -> tuple[str, bool]:
    """Return (command, done) parsed from an assistant message.

    ``command`` is the first fenced shell block (empty if none). ``done`` is True
    when a standalone TASK_COMPLETE line is present. A turn may carry both — the
    command is run, then grading.
    """
    done = bool(_DONE_RE.search(text))
    m = _CMD_FENCE_RE.search(text)
    command = m.group(1).strip() if m else ""
    return command, done


def _sh_quote(value: str) -> str:
    """Single-quote a value for safe shell interpolation."""
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def build_read_file_command(path: str) -> str:
    """Build a shell command that reads a file by exact path.

    ``--`` terminates option parsing so paths beginning with ``-`` are read as
    files, and the path is single-quoted so spaces/metacharacters are literal.
    """
    return f"cat -- {_sh_quote(path)}"


def build_session_write_command(path: str, content: str) -> str:
    """Build a shell command that writes ``content`` to ``path`` exactly.

    Unlike the gdpval reference-file writer (which reduces names to a safe
    basename), this honours the model-supplied path verbatim: parent dirs are
    created, and the content is base64-encoded to avoid all quoting/heredoc
    issues with arbitrary file bytes.
    """
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    quoted = _sh_quote(path)
    parent = f"$(dirname -- {quoted})"
    return (
        f"mkdir -p -- \"{parent}\" && "
        f"printf %s {_sh_quote(b64)} | base64 -d > {quoted} && "
        f"printf 'wrote %s bytes to %s\\n' {len(content.encode('utf-8'))} {quoted}"
    )


def _format_exec(exec_result: dict, exec_output_limit: int) -> str:
    """Render an exec result as the terminal observation shown to the agent."""
    stdout = exec_result.get("stdout") or ""
    stderr = exec_result.get("stderr") or ""
    head = f"[exit={exec_result.get('exit_code')}"
    if exec_result.get("timed_out"):
        head += " timed-out"
    head += "]"
    body = stdout
    if stderr:
        body = f"{body}\n[stderr]\n{stderr}" if body else f"[stderr]\n{stderr}"
    out = f"{head}\n{body}".rstrip()
    if len(out) > exec_output_limit:
        out = out[:exec_output_limit] + "\n... [truncated]"
    return out


def resolve_sandbox_urls(cfg: dict) -> list[str]:
    urls = cfg.get("sandbox_urls")
    if not urls:
        raw = os.environ.get("DOCKYARD_SANDBOX_URLS", "")
        urls = [u.strip() for u in raw.split(",") if u.strip()]
    if not urls:
        raise ValueError(
            "No task-executor endpoints configured. Set the environment's "
            "sandbox_urls or the DOCKYARD_SANDBOX_URLS environment variable."
        )
    return list(urls)


class MultiTurnSessionEnvironment(EnvironmentInterface):
    """Base multi-turn terminal-session environment over the executor session API.

    Per-episode state (session id, executor URL, turn counter) is carried in the
    metadata dict returned each turn. The session is created lazily on the first
    step (the agent's first command has already been generated), exec'd per turn,
    and graded on completion or budget exhaustion via subclass hooks.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reward_mode = cfg.get("reward_mode", "binary")
        self.default_max_turns = int(cfg.get("max_turns", 30))
        self.output_limit = int(cfg.get("output_limit", 8192))
        # Structured tool-use protocol (Hermes <tool_call>) vs. legacy fenced
        # ```bash + TASK_COMPLETE parsing. Off → behaviour byte-identical.
        self.structured_tools = bool(cfg.get("structured_tools", False))
        self.sandbox_urls = resolve_sandbox_urls(cast(dict, cfg))
        self.api_token = cfg.get("api_token")
        self._rr_lock = threading.Lock()
        self._rr = 0
        self._setup(cfg)

    # ---- subclass hooks ---------------------------------------------------
    def _setup(self, cfg: dict) -> None:
        """Read subclass-specific config and build the reward. Override."""

    def _start_spec(self, meta: dict) -> SessionStartSpec:
        """Build the /session/start spec for a task. Override."""
        raise NotImplementedError

    def _after_start(self, url: str, session_id: str, meta: dict) -> None:
        """Hook called once, right after the session container starts and before
        the first agent command is exec'd. Default no-op; override to provision
        task inputs into the container (e.g. reference files)."""

    def _finish_and_score(self, url: str, session_id: str, meta: dict) -> tuple[float, str]:
        """Grade a finished episode; return (reward, verdict text). Override."""
        raise NotImplementedError

    # ---- action parsing ---------------------------------------------------
    def _parse_structured_action(self, text: str) -> tuple[str, bool, bool]:
        """Parse one assistant turn under the structured tool-use protocol.

        Returns ``(command, done, invalid)`` mirroring the legacy parser's
        ``(command, done)`` contract so the exec/grade flow is unchanged:

        - ``run_shell``  → ``command`` is the shell command.
        - ``read_file``  → ``command`` is a ``cat`` of the requested path.
        - ``write_file`` → ``command`` writes the content to the path.
        - ``task_complete`` → ``done=True`` (graded, like TASK_COMPLETE).

        ``invalid`` is the #2656 verdict: True when the turn is malformed, names
        an unknown tool, has schema-invalid arguments, emits no tool call, or
        emits more than one call (fork 7: single action per turn).
        """
        assessment = assess_tool_calls(text, SESSION_TOOLS)
        if assessment.malformed:
            return "", False, True
        # Fork 7: a single action per turn is expected; >1 valid call is invalid.
        if len(assessment.valid_calls) != 1:
            return "", False, True

        call = assessment.valid_calls[0]
        if call.name == "task_complete":
            return "", True, False
        if call.name == "run_shell":
            return str(call.arguments["command"]), False, False
        if call.name == "read_file":
            return build_read_file_command(str(call.arguments["path"])), False, False
        if call.name == "write_file":
            cmd = build_session_write_command(
                str(call.arguments["path"]), str(call.arguments["content"])
            )
            return cmd, False, False
        # Schema-valid call to a registered tool with no dispatch branch: a tool
        # in the registry the env does not execute. Treat as invalid.
        return "", False, True

    # ---- shared machinery -------------------------------------------------
    def _next_url(self) -> str:
        with self._rr_lock:
            url = self.sandbox_urls[self._rr % len(self.sandbox_urls)]
            self._rr += 1
        return url

    def _process_one(self, response_text: str, meta: dict) -> tuple[float, dict, dict, bool]:
        """Drive one sample one turn. Returns (reward, observation, meta, done)."""
        meta = dict(meta)
        session_id = meta.get("_session_id")
        url = meta.get("_sandbox_url")
        turn = int(meta.get("_turn", 0))
        max_turns = int(meta.get("max_turns", self.default_max_turns))

        try:
            if not session_id:
                url = self._next_url()
                session_id = start_session(
                    url, self._start_spec(meta), api_token=self.api_token
                )
                meta["_sandbox_url"] = url
                meta["_session_id"] = session_id
                self._after_start(url, session_id, meta)

            assert url is not None and session_id is not None
            if self.structured_tools:
                command, done, invalid = self._parse_structured_action(response_text)
                # #2656: malformed call, unknown/non-dispatchable tool,
                # schema-invalid args, no call, or >1 call (fork 7).
                meta["invalid_action"] = invalid
            else:
                command, done = _parse_action(response_text)
                # Environment-authoritative invalid-action verdict (#2656): the
                # turn produced neither a runnable command nor a completion
                # signal. The rollout loop reads and pops this key per turn.
                meta["invalid_action"] = not command and not done

            exec_obs = ""
            if command:
                ex = cast(dict, session_exec(
                    url, session_id, command,
                    timeout=int(meta.get("exec_timeout_sec", 120)),
                    output_limit=self.output_limit,
                    api_token=self.api_token,
                ))
                exec_obs = _format_exec(ex, self.output_limit)

            last_turn = (turn + 1) >= max_turns
            if done or last_turn:
                reward, verdict = self._finish_and_score(url, session_id, meta)
                obs = f"{exec_obs}\n{verdict}".strip() if exec_obs else verdict
                return reward, {"role": "user", "content": obs}, meta, True

            meta["_turn"] = turn + 1
            if not command:
                if self.structured_tools:
                    exec_obs = (
                        "No runnable tool call found. Reply with exactly one "
                        "tool call (run_shell, read_file, write_file, or "
                        "task_complete to finish)."
                    )
                else:
                    exec_obs = (
                        "No shell command found. Reply with exactly one ```bash "
                        "code block, or TASK_COMPLETE to finish."
                    )
            return 0.0, {"role": "user", "content": exec_obs}, meta, False

        except Exception as exc:  # noqa: BLE001 — never crash the batch (incl. TaskExecutorError)
            if session_id and url:
                delete_session(url, session_id, api_token=self.api_token)
            return (
                0.0,
                {"role": "user", "content": f"[environment error: {exc}]"},
                meta,
                True,
            )

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[dict],
    ) -> EnvironmentReturn:
        responses = [
            str(ml[-1]["content"]) if ml else "" for ml in message_log_batch
        ]
        pairs = list(zip(responses, metadata))

        # Session I/O is HTTP-bound; run the batch concurrently. Actor-level
        # max_concurrency covers the async (per-sample) rollout path.
        if len(pairs) <= 1:
            outcomes = [self._process_one(r, m) for r, m in pairs]
        else:
            with ThreadPoolExecutor(max_workers=min(len(pairs), 32)) as pool:
                outcomes = list(
                    pool.map(lambda rm: self._process_one(rm[0], rm[1]), pairs)
                )

        rewards = torch.tensor([o[0] for o in outcomes], dtype=torch.float32)
        observations = [o[1] for o in outcomes]
        new_metadata = [o[2] for o in outcomes]
        terminateds = torch.tensor([o[3] for o in outcomes], dtype=torch.bool)
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata,
            next_stop_strings=cast(Any, next_stop_strings),
            rewards=rewards,
            terminateds=terminateds,
            answers=None,
        )

    def shutdown(self) -> None:
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Compute batch-level metrics after all rollouts complete."""
        rewards = (
            batch["rewards"]
            if batch["rewards"].ndim == 1
            else batch["rewards"][:, 0]
        )
        if "is_end" in batch:
            rewards = rewards * batch["is_end"]
        metrics = {
            "accuracy": rewards.mean().item(),
            "resolved_rate": (rewards >= 1.0).float().mean().item(),
            "num_problems_in_batch": int(rewards.shape[0]),
        }
        return batch, metrics
