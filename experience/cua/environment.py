# OSWorld CUA environment — multimodal, multi-turn desktop episodes graded by the
# backend's host-side evaluate().
#
# Mirrors environments/multi_turn_session_env.py: the agent drives one desktop per
# episode across turns; on a terminal control token (DONE/FAIL) or the last
# allowed turn the episode is graded and terminated. The live backend handle (a
# DesktopEnv, not serializable) is held in the actor keyed by an episode id; only
# the id travels in metadata across turns. Screenshots cannot ride in the
# text-only EnvironmentReturn.observations, so each turn's PNG bytes are returned
# in metadata (_screenshot, overwritten each turn); experience/cua/rollout.py
# injects them into the message log as image content. begin_episode() seeds the
# turn-0 screenshot before the first generation (the desktop is not booted at
# data-load time).

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import ray
import torch

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from dockyard_rl.experience.cua.actions.pyautogui_parse import parse_action
from dockyard_rl.experience.cua.backends.interface import Observation, OSWorldBackend

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dockyard_rl.data.interfaces import LLMMessageLogType
else:
    try:
        from dockyard_rl.data.interfaces import LLMMessageLogType
    except ImportError:
        LLMMessageLogType = list  # type: ignore

_NOOP_REPROMPT = (
    "No action found. Reply with exactly one ```python pyautogui code block, or a "
    "single bare WAIT / DONE / FAIL line."
)


def _build_backend(cfg: dict) -> OSWorldBackend:
    """Construct the configured actuation/observation backend.

    Selectable via env.osworld.backend: "official" (DesktopEnv) or
    "control_center" (the in-house The-Eye / control-center / guest-server stack).
    """
    name = cfg.get("backend", "official")
    if name == "official":
        from dockyard_rl.experience.cua.backends.official import (
            OfficialDesktopEnvBackend,
        )

        return OfficialDesktopEnvBackend(cfg)
    if name == "control_center":
        from dockyard_rl.experience.cua.backends.control_center import (
            ControlCenterBackend,
        )

        return ControlCenterBackend(cfg)
    if name == "windows":
        from dockyard_rl.experience.cua.backends.windows import WindowsBackend

        return WindowsBackend(cfg)
    if name == "macos":
        from dockyard_rl.experience.cua.backends.macos import MacBackend

        return MacBackend(cfg)
    raise ValueError(
        f"Unknown OSWorld backend {name!r}. Supported: 'official', "
        "'control_center', 'windows', 'macos'."
    )


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class OSWorldEnvironment(EnvironmentInterface):
    """Multimodal multi-turn OSWorld environment, backend-agnostic.

    Config keys: ``backend`` ("official"); ``max_turns`` default budget;
    ``max_concurrency`` (actor-level, set by create_env); plus backend-specific
    keys forwarded to the backend (provider_name, screen_width/height, headless,
    require_a11y_tree, require_terminal, client_password, cache_dir, ...).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.default_max_turns = int(cfg.get("max_turns", 15))
        self.include_a11y_text = bool(cfg.get("include_a11y_text", False))
        self.a11y_text_limit = int(cfg.get("a11y_text_limit", 4096))
        self.backend = _build_backend(cfg)
        self._episodes: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ---- episode lifecycle ------------------------------------------------
    def _store(self, handle: Any) -> str:
        episode_id = uuid.uuid4().hex
        with self._lock:
            self._episodes[episode_id] = handle
        return episode_id

    def _get(self, episode_id: Optional[str]) -> Any:
        if episode_id is None:
            return None
        with self._lock:
            return self._episodes.get(episode_id)

    def _drop(self, episode_id: Optional[str]) -> Any:
        if episode_id is None:
            return None
        with self._lock:
            return self._episodes.pop(episode_id, None)

    def _a11y_text(self, obs: Observation) -> str:
        if not self.include_a11y_text or not obs.accessibility_tree:
            return ""
        tree = obs.accessibility_tree
        if len(tree) > self.a11y_text_limit:
            tree = tree[: self.a11y_text_limit] + "\n... [a11y truncated]"
        return f"\n[accessibility tree]\n{tree}"

    def begin_episode(self, metadata: dict) -> dict:
        """Provision a desktop for the task and return the turn-0 observation.

        Called once per sample by the CUA rollout before the first generation.
        Returns a serializable dict: ``screenshot`` (PNG bytes or None), ``text``
        (any textual observation to accompany it), and the updated ``metadata``
        (carrying the episode id and turn counter).
        """
        meta = dict(metadata)
        try:
            task_config = meta.get("task_config") or {}
            handle = self.backend.start(task_config)
            episode_id = self._store(handle)
            meta["_episode_id"] = episode_id
            meta["_turn"] = 0
            obs = self.backend.observe(handle)
            return {
                "screenshot": obs.screenshot,
                "text": self._a11y_text(obs),
                "metadata": meta,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - never crash the batch
            return {
                "screenshot": None,
                "text": f"[environment error during start: {exc}]",
                "metadata": meta,
                "error": str(exc),
            }

    def _process_one(
        self, response_text: str, meta: dict
    ) -> tuple[float, dict, dict, bool]:
        """Drive one sample one turn. Returns (reward, observation, meta, done).

        ``observation`` is text-only (``{role, content}``); the turn's screenshot
        PNG bytes ride in ``meta['_screenshot']`` (the CUA rollout reads them from
        the metadata and threads them in as image content). Keeping bytes out of
        ``observations`` honours the text-only EnvironmentReturn.observations type.
        """
        meta = dict(meta)
        episode_id = meta.get("_episode_id")
        turn = int(meta.get("_turn", 0))
        max_turns = int(meta.get("max_turns", self.default_max_turns))

        handle = self._get(episode_id) if episode_id else None
        if handle is None:
            meta["_screenshot"] = None
            return (
                0.0,
                {"role": "user", "content": "[environment error: no live episode]"},
                meta,
                True,
            )

        try:
            action = parse_action(response_text)
            self.backend.act(handle, action)

            last_turn = (turn + 1) >= max_turns
            done = action.is_terminal or last_turn

            if done:
                # Capture the final screen before grading/teardown.
                final_obs = self.backend.observe(handle)
                reward = float(self.backend.evaluate(handle))
                self.backend.stop(handle)
                self._drop(episode_id)
                verdict = "solved" if reward >= 1.0 else "graded"
                meta["_screenshot"] = final_obs.screenshot
                return (
                    reward,
                    {"role": "user", "content": f"[task {verdict}: reward={reward:.3f}]"},
                    meta,
                    True,
                )

            meta["_turn"] = turn + 1
            obs = self.backend.observe(handle)
            content = _NOOP_REPROMPT if action.is_noop else self._a11y_text(obs)
            meta["_screenshot"] = obs.screenshot
            return (
                0.0,
                {"role": "user", "content": content},
                meta,
                False,
            )

        except Exception as exc:  # noqa: BLE001 - never crash the batch
            self.backend.stop(handle)
            self._drop(episode_id)
            meta["_screenshot"] = None
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
            observations=cast(Any, observations),
            metadata=new_metadata,
            next_stop_strings=cast(Any, next_stop_strings),
            rewards=rewards,
            terminateds=terminateds,
            answers=None,
        )

    def shutdown(self) -> None:
        with self._lock:
            handles = list(self._episodes.values())
            self._episodes.clear()
        for handle in handles:
            self.backend.stop(handle)

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        rewards = (
            batch["rewards"]
            if batch["rewards"].ndim == 1
            else batch["rewards"][:, 0]
        )
        if "is_end" in batch:
            rewards = rewards * batch["is_end"]
        metrics = {
            "accuracy": rewards.mean().item(),
            "success_rate": (rewards >= 1.0).float().mean().item(),
            "num_problems_in_batch": int(rewards.shape[0]),
        }
        return batch, metrics
