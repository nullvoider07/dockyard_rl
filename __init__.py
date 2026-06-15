"""dockyard_rl — Distributed RL infrastructure for Project Dockyard.

Package layout
--------------
  cluster/      Ray cluster bootstrap, fleet management, NCCL weight sync.
  distributed/  Virtual cluster, worker groups, sharding, data containers.
  models/       Generation (vLLM) and policy interfaces.
  algorithms/   GRPO and loss functions.
  environments/ Coding task sandbox integration.
  rewards/      Deterministic reward computation and integrity verification.
  data/         Task datasets and dataloaders.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dockyard-rl")
except PackageNotFoundError:
    # Package is not installed (e.g. running directly from the source tree).
    __version__ = "0.0.0.dev0"