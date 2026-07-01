# Cluster bootstrap, fleets, and placement

This document covers how a container joins the cluster, how its fleet role
determines its placement, and how parallelism requests are validated against the
available topology before any GPU is claimed.

## Bootstrap

Every container declares its role through `DOCKYARD_FLEET_ROLE` and calls
`init_ray()` (`cluster/bootstrap.py`). There are two operating modes:

- **Data-centre** — `RAY_ADDRESS` is set; the process attaches to the existing
  cluster unconditionally.
- **Local-dev** — `RAY_ADDRESS` is absent; a local cluster is reused or started,
  keyed by `CUDA_VISIBLE_DEVICES` so a second process with the same device set
  reuses the cluster without a cold start.

`init_ray()` forwards the full process environment to every Ray worker, so
image-level settings (`DOCKYARD_FLEET_ROLE`, `NCCL_SOCKET_IFNAME`,
`CUDA_DEVICE_ORDER`, `TOKENIZERS_PARALLELISM`, …) are visible inside remote
actors without extra configuration. `get_fleet_role()` validates the role
against the three known values and raises if it is missing or unrecognized.

## Fleet specifications

A fleet's topology is captured by an immutable `FleetSpec` (`cluster/fleet.py`),
built from environment variables via `FleetSpec.from_env()`:

| Variable | Meaning |
| --- | --- |
| `DOCKYARD_FLEET_ROLE` | `trainer` / `inference` / `sandbox` (required). |
| `DOCKYARD_NUM_NODES` | Node count, or concurrent episode slots for sandbox (required). |
| `DOCKYARD_GPUS_PER_NODE` | GPUs per node; must be `0` for sandbox, `>0` otherwise. |
| `DOCKYARD_PG_STRATEGY` | Placement override (`SPREAD` / `PACK` / `STRICT_PACK`). |
| `DOCKYARD_MAX_COLOCATED_WORKER_GROUPS` | CPUs allocated per bundle. |

`FleetSpec.__post_init__` enforces the invariants — notably that the sandbox
fleet is CPU-only (`gpus_per_node == 0`) and the GPU fleets are not.

`build_cluster(spec)` dispatches to a per-role builder that returns a
`RayVirtualCluster`. Call `cluster._init_placement_groups()` before constructing
any workers.

## Placement strategy per fleet

Each fleet has a default placement strategy chosen for its collective pattern:

| Fleet | Strategy | Rationale |
| --- | --- | --- |
| **trainer** | `SPREAD` | Each node gets its own placement group, so intra-node NVLink is fully available for tensor-parallel traffic while inter-node RDMA carries pipeline/data-parallel collectives. |
| **inference** | `PACK` | Keeps an engine's tensor-parallel ranks together on one node. When TP exceeds `gpus_per_node`, the generation engine promotes to a unified cross-node placement group automatically. |
| **sandbox** | `SPREAD` | Episode slots distribute evenly across CPU nodes; claiming no GPU resources, sandbox actors never compete with the inference fleet for devices. |

Independent of the fleet-level strategy, the per-node sub-groups inside a
`RayVirtualCluster` always use **`STRICT_PACK`** (`distributed/virtual_cluster.py`):
one strict-pack group per logical node guarantees every bundle of that group
lands on the same physical node, preserving NVLink locality.

## NVLink-segment placement

`STRICT_PACK` keeps a bundle on one node, but a multi-node collective group (a
tensor-parallel replica spanning nodes, or a dedicated teacher fleet) also wants
its nodes to share one NVLink switch fabric rather than scatter across the
cluster. A `FleetSpec` can request this with `segment_size` (default `None` —
topology-agnostic placement, unchanged behaviour).

`ray.sub` registers two custom Ray resources per node: `nvlink_domain_<UUID>`
(the NVLink fabric, parsed from `nvidia-smi`) and `topo_rank` (the SLURM
topological rank). `get_ray_cluster_topology()` reads them into
`node_id → (nvlink_domain, topo_rank)`, and `select_segment_nodes(topology,
segment_size, num_nodes)` greedily takes complete `segment_size`-node segments
from each NVLink domain, in topological order, until `num_nodes` are chosen —
returning the selected nodes plus the remainder. `segment_size` must divide
`num_nodes`; if not enough complete segments can be formed it raises rather than
spill a segment across fabrics.

Nodes with no topology resources collapse into a single `unknown` pseudo-domain
and are **excluded** from segment selection (they fall through to the remainder).
Selecting them would emit a placement constraint naming a Ray resource `ray.sub`
never registered, so the bundle could never schedule — excluding them pins
segments only to real, registered NVLink domains. This underpins topology-aware
teacher placement for [on-policy distillation](../design-docs/distillation.md)
and any TP replica that must span nodes.

The selection/sort logic is pure (CPU-tested); the `ray.nodes()` read, the
domain-pinned placement, and the `ray.sub` topology probe are cluster-bound and
hardware-deferred.

## Pre-flight validation

`cluster/placement.py` is pure computation — it creates no Ray objects, so it
can run before `init_ray()` for pre-flight checks. A `ParallelismSpec`
(tensor / pipeline / data / expert parallel sizes) is validated against the
fleet topology *before* placement groups exist, catching impossible
configurations (e.g. an expert-parallel size that is not a multiple of the
tensor-parallel size, or a model-parallel size that does not fit the node) at
launch rather than mid-run.

A model replica consumes `tensor_parallel_size × pipeline_parallel_size` GPUs;
when that exceeds `gpus_per_node`, the placement helpers signal that a unified
cross-node placement group is required rather than per-node groups.

## The virtual cluster

`RayVirtualCluster` (`distributed/virtual_cluster.py`) is the object the
worker-group builders consume. It owns the placement groups and exposes them via
`get_placement_groups()`. Its constructor takes a `bundle_ct_per_node_list` (the
per-node bundle counts), `use_gpus`, `num_gpus_per_node`, and the fleet's
`placement_group_strategy`; the per-node sub-groups it creates are always
`STRICT_PACK` regardless of that top-level strategy.
