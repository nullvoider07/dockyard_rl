"""dockyard-k8s: Kubernetes deployment for the dockyard_rl GRPO stack.

A config-driven launcher that renders and applies the three-fleet
(trainer / inference / sandbox) async-GRPO topology onto a Kubernetes
cluster running the KubeRay operator.

One ``.infra.yaml`` captures where to run (namespace, image, per-role
RayCluster shape, the sandbox executor pool); the recipe YAML captures
what to train. The trainer and inference fleets run as GPU Ray pods; the
sandbox runs as a separate, horizontally-scalable CPU executor pool
(``POST /task/submit`` on port 9090) addressed over HTTP via
``DOCKYARD_SANDBOX_URLS`` — off the GPU compute path.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
