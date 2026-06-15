# Deployment

dockyard_rl runs anywhere Ray and the `ubuntu-swe` image run. Two deployment
paths are wired:

- [Slurm](slurm.md) — `ray.sub` plus the `scripts/cluster/` bootstrap scripts,
  for HPC and bare-metal clusters.
- [Kubernetes](kubernetes.md) — the `dockyard_k8s` config-driven launcher under
  `infra/`, for cluster-orchestrated deployments.

Both deploy the same three-fleet topology and the same image; they differ only in
how nodes are provisioned and how each node learns its `DOCKYARD_FLEET_ROLE`.

```{toctree}
:hidden:
:maxdepth: 1

slurm
kubernetes
```
