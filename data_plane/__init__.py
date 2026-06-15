"""dockyard_rl data-plane package.

Experience/rollout-data transfer plane (not weights) for non-colocated RL:
the trainer and generation fleets exchange per-sample tensor columns through
an external store instead of Ray plasma, so a (B, S) logprob tensor never
makes a driver roundtrip.

The boundary (``interfaces``) imports without ``tensordict`` — only the
concrete adapters that move tensor data require it. To keep importing this
package dependency-light, the heavier symbols (factory, codec, adapters,
observability) are re-exported lazily via ``__getattr__``: they resolve on
first attribute access, not at package import.
"""

from typing import TYPE_CHECKING, Any

from dockyard_rl.data_plane.interfaces import (
    DataPlaneClient,
    DataPlaneConfig,
    KVBatchMeta,
)

if TYPE_CHECKING:
    from dockyard_rl.data_plane.codec import materialize
    from dockyard_rl.data_plane.factory import build_data_plane_client
    from dockyard_rl.data_plane.observability import (
        MetricsDataPlaneClient,
        log_event,
    )

# Lazily-resolved public symbols: name -> (submodule, attribute).
_LAZY: dict[str, tuple[str, str]] = {
    "build_data_plane_client": ("factory", "build_data_plane_client"),
    "materialize": ("codec", "materialize"),
    "MetricsDataPlaneClient": ("observability", "MetricsDataPlaneClient"),
    "log_event": ("observability", "log_event"),
}

__all__ = [
    "DataPlaneClient",
    "DataPlaneConfig",
    "KVBatchMeta",
    "build_data_plane_client",
    "materialize",
    "MetricsDataPlaneClient",
    "log_event",
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{target[0]}")
    return getattr(module, target[1])
