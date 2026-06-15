# Host provisioning for the control-center backend (Phase 3 scale).
#
# The control-center backend attaches to a running ubuntu-desktop-code container;
# at scale one OSWorldEnvironment actor with max_concurrency=N drives N episodes
# at once, each needing its own container. A Provisioner owns a warm pool of
# container hosts and leases one per episode (acquire on backend.start, release on
# backend.stop). The pool is bounded, so acquire blocks when every host is busy —
# natural backpressure that caps live containers at the pool size regardless of
# how hard the async collector pushes.
#
# Implementations:
#   StaticPoolProvisioner  — a fixed set of externally-provisioned hosts (the
#                            warm pool is launched by the orchestration/fleet
#                            layer; this just leases from it). "attach" is the
#                            degenerate single-host case.
#   DockerPoolProvisioner  — pre-boots N containers from an image via docker-py
#                            (a deferred dep, guarded) and leases their IPs.

from __future__ import annotations

import abc
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("dockyard_rl.cua.provisioner")


@dataclass
class HostLease:
    """A leased container host. ``host`` is reachable for the service ports the
    backend connects to (The-Eye/control-center/guest/executor). ``container_id``
    and ``_handle`` are set only by the docker provisioner."""

    host: str
    container_id: Optional[str] = None
    _handle: Any = field(default=None, repr=False)


class Provisioner(abc.ABC):
    """Leases container hosts to the backend, one per concurrent episode."""

    @abc.abstractmethod
    def acquire(self, task: dict[str, Any]) -> HostLease:
        """Lease a host, blocking until one is free (bounded pool = backpressure)."""

    @abc.abstractmethod
    def release(self, lease: HostLease) -> None:
        """Return a leased host to the pool for reuse."""

    def shutdown(self) -> None:
        """Tear down any owned resources (no-op for externally-managed pools)."""


class StaticPoolProvisioner(Provisioner):
    """Lease from a fixed pool of already-running hosts.

    The hosts are provisioned outside this process (a single attached container,
    or a warm pool stood up by the fleet/orchestration layer). Reset between
    episodes is the backend's task-setup re-run; this layer only arbitrates
    exclusive access so two episodes never drive the same container at once.
    """

    def __init__(self, hosts: list[str], acquire_timeout: Optional[float] = None) -> None:
        if not hosts:
            raise ValueError("StaticPoolProvisioner requires at least one host.")
        self._acquire_timeout = acquire_timeout
        self._pool: "queue.Queue[str]" = queue.Queue()
        for h in hosts:
            self._pool.put(h)

    def acquire(self, task: dict[str, Any]) -> HostLease:
        try:
            host = self._pool.get(
                block=True,
                timeout=self._acquire_timeout,
            )
        except queue.Empty as exc:
            raise TimeoutError(
                "No free CUA host within acquire_timeout; the pool is smaller "
                "than the env actor's max_concurrency, or hosts are not being "
                "released."
            ) from exc
        return HostLease(host=host)

    def release(self, lease: HostLease) -> None:
        self._pool.put(lease.host)


class DockerPoolProvisioner(Provisioner):
    """Warm pool of N ubuntu-desktop-code containers launched via docker-py.

    Containers are started headless (software-rendered) on a shared network so
    each gets its own IP and serves the fixed service ports; the backend connects
    per-lease IP. ``docker_run_kwargs`` is passed through to ``containers.run`` so
    the operator supplies host-specific bits (devices, gpus, shm_size, …) — those
    are environment-specific and cannot be validated offline. docker is a deferred
    dependency: the import is guarded so this module loads without it.
    """

    def __init__(
        self,
        image: str,
        pool_size: int,
        docker_run_kwargs: Optional[dict[str, Any]] = None,
        network: Optional[str] = None,
        startup_timeout: float = 300.0,
        name_prefix: str = "dockyard-cua",
    ) -> None:
        try:
            import docker  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - docker is an optional dep
            raise RuntimeError(
                "DockerPoolProvisioner needs the docker SDK (pip install docker) "
                "on the host running the OSWorld environment."
            ) from exc
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self._docker = docker.from_env()
        self._image = image
        self._pool_size = pool_size
        self._run_kwargs = dict(docker_run_kwargs or {})
        self._network = network
        self._startup_timeout = startup_timeout
        self._name_prefix = name_prefix
        self._lock = threading.Lock()
        self._containers: dict[str, Any] = {}
        self._pool: "queue.Queue[HostLease]" = queue.Queue()
        self._started = False

    def _container_ip(self, container: Any) -> str:
        container.reload()
        nets = container.attrs["NetworkSettings"]
        if self._network and self._network in nets.get("Networks", {}):
            ip = nets["Networks"][self._network].get("IPAddress")
        else:
            ip = nets.get("IPAddress") or next(
                (n.get("IPAddress") for n in nets.get("Networks", {}).values() if n.get("IPAddress")),
                None,
            )
        if not ip:
            raise RuntimeError(f"container {container.short_id} has no IP yet")
        return ip

    def start(self) -> None:
        """Boot the pool. Idempotent; called lazily on first acquire."""
        with self._lock:
            if self._started:
                return
            for i in range(self._pool_size):
                kwargs: dict[str, Any] = {
                    "image": self._image,
                    "detach": True,
                    "name": f"{self._name_prefix}-{i}",
                    **self._run_kwargs,
                }
                if self._network:
                    kwargs["network"] = self._network
                container = self._docker.containers.run(**kwargs)
                ip = self._container_ip(container)
                self._containers[container.id] = container
                self._pool.put(HostLease(host=ip, container_id=container.id, _handle=container))
                logger.info("CUA pool container %s up at %s", container.short_id, ip)
            self._started = True

    def acquire(self, task: dict[str, Any]) -> HostLease:
        if not self._started:
            self.start()
        return self._pool.get(block=True, timeout=self._startup_timeout)

    def release(self, lease: HostLease) -> None:
        self._pool.put(lease)

    def shutdown(self) -> None:
        with self._lock:
            for container in self._containers.values():
                try:
                    container.remove(force=True)
                except Exception as e:  # noqa: BLE001 - best-effort teardown
                    logger.warning("failed to remove container: %s", e)
            self._containers.clear()
            self._started = False


def build_provisioner(cfg: dict[str, Any]) -> Provisioner:
    """Construct the provisioner selected by ``cfg['provision']``.

    - "attach"      : single host from cfg['host'] (pool of one).
    - "static_pool" : cfg['hosts'] (list) — an externally-managed warm pool.
    - "docker_pool" : pre-boot cfg['pool_size'] containers from cfg['docker_image'].
    """
    mode = cfg.get("provision", "attach")
    if mode == "attach":
        host = cfg.get("host")
        if not host:
            raise RuntimeError(
                "provision='attach' requires env.osworld.host (the provisioned "
                "ubuntu-desktop-code container address)."
            )
        return StaticPoolProvisioner([host], acquire_timeout=cfg.get("acquire_timeout"))
    if mode == "static_pool":
        hosts = cfg.get("hosts") or ([cfg["host"]] if cfg.get("host") else [])
        if not hosts:
            raise RuntimeError("provision='static_pool' requires env.osworld.hosts (a list).")
        return StaticPoolProvisioner(list(hosts), acquire_timeout=cfg.get("acquire_timeout"))
    if mode == "docker_pool":
        return DockerPoolProvisioner(
            image=cfg["docker_image"],
            pool_size=int(cfg.get("pool_size", 1)),
            docker_run_kwargs=cfg.get("docker_run_kwargs"),
            network=cfg.get("docker_network"),
            startup_timeout=float(cfg.get("acquire_timeout", 300.0)),
        )
    raise ValueError(
        f"Unknown provision mode {mode!r}. Supported: attach, static_pool, docker_pool."
    )
