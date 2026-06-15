# The experience data plane

Weight sync moves weights from the trainer to the inference fleet. The **data
plane** moves the other direction's bulk data — rollout experience and the
per-sample tensor columns (log-probs, token IDs, advantages) — between
non-colocated fleets, without routing large tensors through Ray's object store
and the driver.

It is **optional**. With `data_plane=None` (the default) the in-memory colocated
path is used and no external store is involved. When configured, it lets a large
`(B, S)` tensor pass producer→consumer without a driver round-trip.

## The boundary

Every call site in `algorithms/`, `experience/`, and `models/` goes through the
`DataPlaneClient` ABC (`data_plane/interfaces.py`) — never an adapter directly.
That indirection is what makes the implementation swappable. The boundary module
imports without `tensordict` (every `TensorDict` reference is a deferred
annotation), so importing the package stays dependency-light; only the concrete
adapters that touch tensor data require it.

Two adapters ship:

| `impl` | Adapter | Use |
| --- | --- | --- |
| `noop` | `data_plane/adapters/noop.py` | Disabled / passthrough. |
| `transfer_queue` | `data_plane/adapters/transfer_queue.py` | External queue/store with `simple` or `mooncake_cpu` backends. |

Optional observability middleware (`ObservabilityConfig`) wraps the chosen
adapter to record per-operation metrics.

## Control plane vs. data plane

The design separates **metadata** (cheap, routable) from **tensor data**
(expensive). `KVBatchMeta` carries per-sample IDs, field names, sequence lengths,
and per-sample primitive `tags` — enough to filter, shard, and balance a batch
**without fetching any tensor data**. For example, the driver can compute a
balanced per-DP-rank shard from `sequence_lengths` alone, and dynamic sampling
can drop zero-std rows by inspecting `tags`, before ever materializing tensors.

`KVBatchMeta` is a pure value type with `subset` / `slice` / `concat` transforms
(each returns a fresh meta), and `stamp_tags` to mirror scalar columns
(`std`, `total_reward`, `weight_version`, …) onto the per-sample sidecar at
production time.

## The client API

`DataPlaneClient` methods fall into three intents:

- **Task-mediated** — for stages that wait on upstream production:
  `register_partition` (declare the partition schema and consumer tasks),
  `claim_meta` (discover and atomically claim ready samples, advancing a
  per-task cursor), `get_data` (resolve a claimed meta to a `TensorDict`),
  `check_consumption_status`.
- **Direct-by-key** — for stages that already know the uids:
  `put_samples` (the producer entrypoint), `get_samples` (fetch a known slice
  without advancing any cursor), `clear_samples`.
- **Lifecycle** — `close` (idempotent).

The completion signal is **field production**, not an explicit "done": when a
stage calls `put_samples` for a field, the controller flips the
`production_status` bit for that `(sample, field)` pair, and downstream consumers
waiting on that field become unblocked. There is intentionally no
`mark_consumed`.

`get_samples` and `get_data` both **require** an explicit field set — there is no
implicit "fetch every field" fallback, because bulk schemas are wide and a silent
over-fetch is the most expensive shape the wire can take. Callers must name what
they read.
