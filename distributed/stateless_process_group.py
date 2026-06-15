"""Stateless NCCL process group for cross-fleet weight synchronisation.

Used by both trainer policy workers (to broadcast) and vLLM inference
workers (to receive) in the NCCL weight-sync communicator.

The group is "stateless" in the sense that it carries no torch.distributed
global state — it uses its own TCPStore for rendezvous and its own NCCL
communicator object, completely isolated from any intra-fleet process group
already initialised by the trainer or by vLLM's internal TP communicator.

This isolation is the critical property: cross-contamination of NCCL
communicators across fleets causes deadlocks that are very difficult to
diagnose at data-centre scale.
"""

from typing import Optional

import torch
from nccl.core.communicator import Communicator  # type: ignore[import]
from nccl.core.utils import UniqueId, get_unique_id  # type: ignore[import]

class StatelessProcessGroup:
    """Minimal NCCL communicator wrapper for the trainer↔inference sync group.

    The TCPStore rendezvous is performed synchronously in __init__.
    All ranks must construct this object simultaneously (same host/port/
    world_size) for the store initialisation to succeed — the trainer/inference
    init_collective handshake (driven from the weight_sync WeightSynchronizer's
    init_communicator) guarantees this ordering.

    Args:
        master_address: IP of the rank-0 process (trainer DP leader 0).
        port:           Free port on master_address — obtained via
                        RayVirtualCluster.get_master_address_and_port().
        rank:           This process's rank within the sync communicator.
        world_size:     Total ranks = trainer_dp_size + inference_dp_size.
    """

    def __init__(
        self,
        master_address: str,
        port: int,
        rank: int,
        world_size: int,
    ) -> None:
        self.master_address = master_address
        self.port           = port
        self.rank           = rank
        self.world_size     = world_size

        # TCPStore: rank 0 acts as the store server; others connect as clients.
        self.tcp_store = torch.distributed.TCPStore(  # type: ignore[attr-defined]
            host_name  = self.master_address,
            port       = self.port,
            world_size = self.world_size,
            is_master  = (self.rank == 0),
        )

        # nccl_communicator is set by init_nccl_communicator().
        self.nccl_communicator: Optional[Communicator] = None

    def init_nccl_communicator(self, device: int) -> None:
        """Initialise the NCCL communicator and verify with a warmup broadcast.

        Rank 0 generates a UniqueId and distributes it to all other ranks via
        the TCPStore.  All ranks then construct the Communicator simultaneously.
        A warmup broadcast of a single float tensor verifies that the
        communicator is functional before any weight tensor is sent.

        Args:
            device: CUDA device index for this rank.
        """
        UNIQUE_ID_KEY = "dockyard_nccl_unique_id"

        if self.rank == 0:
            unique_id       = get_unique_id()
            unique_id_bytes = unique_id.as_bytes
            self.tcp_store.set(UNIQUE_ID_KEY, unique_id_bytes)
        else:
            self.tcp_store.wait([UNIQUE_ID_KEY])
            unique_id_bytes = self.tcp_store.get(UNIQUE_ID_KEY)
            unique_id       = UniqueId.from_bytes(unique_id_bytes)

        with torch.cuda.device(device):
            self.nccl_communicator = Communicator.init(
                nranks    = self.world_size,
                rank      = self.rank,
                unique_id = unique_id,
            )

            # Warmup: rank 0 sends ones; all others expect ones.
            stream = torch.cuda.current_stream()
            data   = (
                torch.ones(1, device=device)
                if self.rank == 0
                else torch.zeros(1, device=device)
            )
            self.broadcast(data, src=0, stream=stream)
            torch.cuda.current_stream().synchronize()

            if not torch.allclose(data, torch.ones(1, device=device)):
                raise RuntimeError(
                    f"NCCL warmup broadcast failed on rank {self.rank}. "
                    "The sync communicator is not functional."
                )

    def broadcast(
        self,
        tensor: torch.Tensor,
        src:    int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Broadcast tensor from src rank to all ranks in the sync group.

        Args:
            tensor: Tensor to broadcast (both send buffer on src and receive
                    buffer on all other ranks — broadcast is in-place).
            src:    Source rank index within this communicator.
            stream: CUDA stream.  Defaults to the current stream.
        """
        if self.nccl_communicator is None:
            raise RuntimeError(
                "init_nccl_communicator() must be called before broadcast()."
            )
        if stream is None:
            stream = torch.cuda.current_stream()

        self.nccl_communicator.broadcast(
            sendbuf = tensor,
            recvbuf = tensor,
            root    = src,
            stream  = int(stream.cuda_stream),
        )

    def destroy(self) -> None:
        """Release the NCCL communicator and TCPStore resources.

        Idempotent; safe to call multiple times.
        """
        if self.nccl_communicator is not None:
            try:
                del self.nccl_communicator
            except Exception:
                pass
            self.nccl_communicator = None