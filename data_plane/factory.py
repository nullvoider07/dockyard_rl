"""Single entrypoint that maps a :class:`DataPlaneConfig` to a client."""

from __future__ import annotations

from dockyard_rl.data_plane.interfaces import DataPlaneClient, DataPlaneConfig


def build_data_plane_client(
    cfg: DataPlaneConfig | None, *, bootstrap: bool = True
) -> DataPlaneClient:
    """Construct the configured data-plane client.

    Dispatches on ``cfg["impl"]``:
      * ``"transfer_queue"`` — the production TransferQueue-backed adapter.
      * ``"noop"`` — the in-memory reference adapter (single-process / CI /
        local dry-runs); a dockyard addition over the upstream, which kept
        NoOp test-only. Selecting it is explicit (``impl="noop"``), never a
        silent fallback.

    Raises if data_plane is disabled — the colocated trainer
    (``dockyard_rl.algorithms.grpo.grpo_train``) should be used in that
    case rather than a NoOp fallback here.

    Args:
        cfg: Data-plane config; must have ``enabled=True``.
        bootstrap: ``True`` on the driver — bootstraps the TQ
            controller. ``False`` on worker processes — connects to the
            existing controller (avoids creating a second named actor).
            Ignored by the noop adapter (single-process).

    Returns:
        A configured ``DataPlaneClient``; wrapped in
        :class:`MetricsDataPlaneClient` when observability is enabled.
    """
    if cfg is None or not cfg["enabled"]:
        raise ValueError(
            "build_data_plane_client called with data_plane disabled. "
            "Use the colocated dockyard_rl.algorithms.grpo.grpo_train trainer "
            "(which never engages the data plane) for that case."
        )

    impl = cfg["impl"]
    if impl == "transfer_queue":
        try:
            from dockyard_rl.data_plane.adapters.transfer_queue import (
                TQDataPlaneClient,
            )
        except ImportError as exc:
            # The adapter imports transfer_queue and tensordict at module
            # level. Neither, nor mooncake, is in ubuntu-base/pyproject.toml
            # or uv.lock, so in the shipped image this surfaces as a bare
            # ModuleNotFoundError from inside an adapter rather than as a
            # statement about the deployment.
            raise RuntimeError(
                "data_plane.impl='transfer_queue' needs the transfer_queue "
                "and tensordict packages (and mooncake for the mooncake_cpu "
                "backend) in the worker's system Python. The ubuntu-swe image "
                "does not carry them today -- see ubuntu-base/pyproject.toml. "
                "Use data_plane.impl='noop' for a single-process run, or add "
                "the packages to the image. "
                f"Underlying import error: {exc}"
            ) from exc

        client: DataPlaneClient = TQDataPlaneClient(cfg, bootstrap=bootstrap)
    elif impl == "noop":
        from dockyard_rl.data_plane.adapters.noop import NoOpDataPlaneClient

        client = NoOpDataPlaneClient()
    else:
        raise ValueError(f"unknown data_plane impl: {impl!r}")

    obs = cfg.get("observability") or {}
    if obs.get("enabled", False):
        from dockyard_rl.data_plane.observability import (
            MetricsDataPlaneClient,
            log_event,
        )

        on_event = obs.get("callback") or log_event
        client = MetricsDataPlaneClient(client, on_event=on_event)  # type: ignore[arg-type]
    return client
