# Architecture

How dockyard_rl fits together at runtime.

- [Overview](overview.md) — the three fleets, the async GRPO loop, and the
  weight-staleness handshake that lets training and generation overlap.
- [Cluster, fleets, and placement](cluster-and-fleets.md) — bootstrap, fleet
  specs, placement strategies, `STRICT_PACK`, and pre-flight validation.
- [Weight synchronization](weight-sync.md) — the staleness handshake, the
  NCCL / IPC / HTTP transports, and the refit seam.
- [Experience data plane](data-plane.md) — moving rollout data between
  non-colocated fleets without a driver round-trip.

```{toctree}
:hidden:
:maxdepth: 1

overview
cluster-and-fleets
weight-sync
data-plane
```
