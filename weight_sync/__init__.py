"""Weight synchronization between the trainer and inference fleets.

Exposes the WeightSynchronizer abstraction and a factory that selects the
transport (IPC/ZMQ, HTTP, NCCL collective) from deployment topology.
"""

from dockyard_rl.weight_sync.factory import create_weight_synchronizer
from dockyard_rl.weight_sync.interfaces import WeightSynchronizer

__all__ = [
    "WeightSynchronizer",
    "create_weight_synchronizer",
]
