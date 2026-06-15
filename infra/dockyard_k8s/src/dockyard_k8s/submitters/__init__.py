"""Training-job submitters.

Two transports, same shape. The abstraction keeps the orchestrator and CLI
agnostic about whether a run lives under Ray's Job framework or as a raw
process on the head pod.

* :class:`PortForwardSubmitter` opens a brief ``kubectl port-forward`` to the
  head dashboard and uses Ray's ``JobSubmissionClient``. Dev-iteration default.

* :class:`ExecSubmitter` shells into the head pod with ``kubectl exec`` and
  launches the entrypoint as a ``nohup`` process — the same shape as a Slurm
  driver on a login node. Production / batch default. No Ray Job abstraction;
  the training code calls ``ray.init(address="auto")`` to attach.

Handles are persisted to ``~/.cache/dockyard-k8s/runs/<run-id>.json`` so that
``dockyard-k8s job logs`` / ``job stop`` on a later invocation knows which
transport to talk to.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol

from ..schema import InfraConfig, SubmitterMode

HandleKind = Literal["ray", "exec"]
JobStatusStr = Literal["running", "succeeded", "failed", "stopped", "unknown"]


@dataclass
class SubmissionHandle:
    """Opaque reference to a submitted training run.

    Carries just enough to find the run later: which transport submitted it,
    what cluster it's on, and transport-specific identifiers (Ray submission
    id, or a head-pod / pidfile pair for exec).
    """

    kind: HandleKind
    run_id: str
    cluster_name: str
    namespace: str
    # exec-mode only
    pod: str | None = None
    tmp_dir: str | None = None
    # ray-mode only
    dashboard_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubmissionHandle":
        return cls(**d)


class JobSubmitter(Protocol):
    """Transport abstraction for training-job submission + observability."""

    def submit(
        self,
        cluster_name: str,
        namespace: str,
        *,
        entrypoint: str,
        run_id: str,
        env_vars: dict[str, str] | None = None,
        working_dir: Path | None = None,
    ) -> SubmissionHandle: ...

    def follow(self, handle: SubmissionHandle) -> Iterator[str]: ...

    def status(self, handle: SubmissionHandle) -> JobStatusStr: ...

    def stop(self, handle: SubmissionHandle, *, force: bool = False) -> None: ...


def build_submitter(infra: InfraConfig) -> JobSubmitter:
    """Pick a submitter based on the resolved infra config."""
    if infra.submit.submitter is SubmitterMode.EXEC:
        from .exec_ import ExecSubmitter

        return ExecSubmitter(exec_tmp_dir=infra.submit.execTmpDir)
    from .portforward import PortForwardSubmitter

    return PortForwardSubmitter()


def _cache_root() -> Path:
    """Resolve the handle cache dir at call time.

    Reading the env var on every call (instead of once at import) keeps tests
    simple — a fixture can ``monkeypatch.setenv`` and immediately see the new
    location without a package reload.
    """
    base = os.environ.get("DOCKYARD_K8S_CACHE_DIR")
    if base:
        return Path(base) / "runs"
    return Path.home() / ".cache" / "dockyard-k8s" / "runs"


def handle_path(run_id: str) -> Path:
    return _cache_root() / f"{run_id}.json"


def save_handle(handle: SubmissionHandle) -> Path:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    p = handle_path(handle.run_id)
    p.write_text(json.dumps(handle.to_dict(), indent=2))
    return p


def load_handle(run_id: str) -> SubmissionHandle | None:
    p = handle_path(run_id)
    if not p.exists():
        return None
    try:
        return SubmissionHandle.from_dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, TypeError):
        return None


__all__ = [
    "HandleKind",
    "JobStatusStr",
    "JobSubmitter",
    "SubmissionHandle",
    "build_submitter",
    "handle_path",
    "load_handle",
    "save_handle",
]
