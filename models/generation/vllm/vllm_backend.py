"""vLLM internal worker extension for Project Dockyard.

VllmInternalWorkerExtension is registered as worker_extension_cls in vLLM's
LLM constructor.  vLLM mixes it into every internal Worker instance on every
TP/PP rank so that collective_rpc() calls reach all ranks simultaneously.

Responsibilities
----------------
- init_collective:                Join the NCCL weight-sync communicator.
- update_weights_from_collective: Receive packed parameter tensors and refit
                                  the vLLM model via load_weights().
- update_weights_via_ipc_zmq:     Receive weights via CUDA IPC + ZMQ
                                  (kept for completeness; not used in the
                                  non-collocated Dockyard path).
- prepare_refit_info:             Store param metadata for the above.
- report_device_id:               Return CUDA device UUID for topology checks.
- start/stop_gpu_profiling:       NVTX/torch.cuda.profiler integration.
- cleanup:                        Release ZMQ resources on shutdown.
"""

import gc
import traceback
from typing import Any

import torch
import zmq

from dockyard_rl.distributed.stateless_process_group import StatelessProcessGroup
from dockyard_rl.utils.nvml import get_device_uuid
from dockyard_rl.utils.packed_tensor import packed_broadcast_consumer

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Ensure it was built during the ubuntu-swe "
        "image build (§5c) and that sys.executable in the Ray runtime_env "
        "of VllmGenerationWorker points to the correct Python."
    )

# Architecture-specific weight fixup for GptOss down_proj (Megatron export vs vLLM layout)
def fix_gpt_oss_export_transpose(key: str, weight: torch.Tensor) -> torch.Tensor:
    """Apply GptOss down_proj transpose fix.

    HuggingFace and Megatron store MoE down_proj as [in, out]; vLLM expects
    [out, in].  This fixup is applied during weight loading for GptOss models.

    See https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3271.
    """
    if key.endswith("mlp.experts.down_proj"):
        weight = weight.transpose(-2, -1).contiguous()
    return weight

# vLLM internal worker extension
class VllmInternalWorkerExtension:
    """Mixed into vLLM's internal Worker on all TP/PP ranks via collective_rpc.

    This class has no __init__ — all attribute assignments use implicit
    definition (the Worker base class owns __init__).  Self here is the
    combined Worker + Extension instance.
    """

    # Collective initialisation and device reporting
    def init_collective(
        self,
        rank_prefix:       int,
        ip:                str,
        port:              int,
        world_size:        int,
        train_world_size:  int,
    ) -> None:
        """Join the NCCL weight-sync communicator.

        Each vLLM TP rank gets a unique rank within the sync communicator:
          rank = train_world_size + rank_prefix + local_tp_rank

        This ensures all TP ranks on this inference DP group participate,
        which is necessary because update_weights_from_collective() must be
        called on all TP ranks simultaneously (vLLM collective_rpc semantics).

        Args:
            rank_prefix:      DP-shard offset assigned by VllmGeneration (= I
                              inference shard index in train_world_size + shard).
            ip:               Master address for TCPStore rendezvous.
            port:             Free port on ip.
            world_size:       Total ranks in sync communicator.
            train_world_size: Number of trainer DP leaders in the communicator.
        """
        local_rank = torch.distributed.get_rank()
        rank       = train_world_size + rank_prefix + local_rank

        self.model_update_group = StatelessProcessGroup(
            master_address = ip,
            port           = port,
            rank           = rank,
            world_size     = world_size,
        )
        self.model_update_group.init_nccl_communicator(device=self.device)

    # Device identity and topology checks
    def report_device_id(self) -> str:
        """Return the UUID of the current CUDA device."""
        return get_device_uuid(self.device.index)

    # ZMQ socket (IPC weight streaming path)
    def get_zmq_address(self) -> str:
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self) -> None:
        """Lazily initialise a ZMQ REP socket for IPC weight streaming."""
        if hasattr(self, "zmq_socket"):
            return
        self.zmq_context = zmq.Context()
        self.zmq_socket  = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.setsockopt(zmq.SNDTIMEO, 120_000)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 120_000)
        self.zmq_socket.setsockopt(zmq.LINGER,   0)
        self.zmq_socket.connect(self.get_zmq_address())

    # Weight metadata preparation
    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Store param metadata forwarded from the trainer-side refit path.

        Args:
            state_dict_info: Dict mapping param_name → (shape, dtype).
                             Built by the trainer policy's prepare_refit_info().
        """
        self.state_dict_info = state_dict_info

    # FP8 KV cache post-processing
    def _maybe_process_fp8_kv_cache(self) -> None:
        """Run process_weights_after_loading() when FP8 KV cache is active."""
        use_fp8_kv = False
        if hasattr(self.model_runner, "vllm_config") and hasattr(
            self.model_runner.vllm_config, "cache_config"
        ):
            kv_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv = kv_dtype is not None and "fp8" in str(kv_dtype).lower()

        if not use_fp8_kv:
            return

        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )
        target_device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(
            self.model_runner.model,
            self.model_runner.model_config,
            target_device,
        )

    # Draft model weight splitting and loading
    @staticmethod
    def _split_policy_and_draft_weights(
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Separate draft model weights from policy weights.

        Trainer exports Eagle3 / speculative-decoding draft parameters under
        a 'draft.' prefix.  This splits them so each model receives its own
        weights via the appropriate load_weights() call.
        """
        policy_weights: list[tuple[str, torch.Tensor]] = []
        draft_weights:  list[tuple[str, torch.Tensor]] = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
            else:
                policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    def _load_draft_weights(
        self,
        draft_weights: list[tuple[str, torch.Tensor]],
    ) -> None:
        if not draft_weights:
            return
        drafter     = getattr(self.model_runner, "drafter",     None)
        draft_model = getattr(drafter,           "model",       None) if drafter else None
        if draft_model is None:
            print(
                "[vllm_backend] Received draft weights but vLLM drafter is "
                "unavailable; skipping draft update."
            )
            return
        draft_model.load_weights(weights=draft_weights)

    # Core weight loading
    def _load_weights(
        self,
        weights: list[tuple[str, torch.Tensor]],
    ) -> None:
        """Load weights into the vLLM model, handling GptOss, FP8, and draft.

        This is the single entry point for all weight loading paths
        (NCCL collective and IPC ZMQ).  It:
        1. Applies GptOss down_proj transpose fix if needed.
        2. Splits policy and draft weights.
        3. Routes to FP8 or standard load_weights().
        4. Loads draft weights into the drafter model.
        """
        architectures = (
            self.model_runner.vllm_config.model_config.architectures
            if hasattr(self.model_runner, "vllm_config")
            else []
        )

        if "GptOssForCausalLM" in architectures:
            weights = [
                (key, fix_gpt_oss_export_transpose(key, tensor))
                for key, tensor in weights
            ]

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)

        # FP8 quantized weight loading (optional ModelOpt integration).
        try:
            from dockyard_rl.models.generation.vllm.quantization import fp8  # type: ignore[import]
            if fp8.is_fp8_model(self.model_runner.vllm_config):
                fp8.load_weights(policy_weights, self.model_runner)
                self._load_draft_weights(draft_weights)
                return
        except ImportError:
            pass

        self.model_runner.model.load_weights(weights=policy_weights)
        self._load_draft_weights(draft_weights)

    # NCCL collective weight update
    def update_weights_from_collective(self) -> bool:
        """Receive updated policy weights via NCCL broadcast and refit model.

        Called by VllmGenerationWorker.update_weights_from_collective() via
        collective_rpc, which invokes this on every TP rank simultaneously.

        The packed_broadcast_consumer() drives the receive loop — it must
        mirror the packed_broadcast_producer() loop on the trainer side
        exactly (same iteration order, same buffer sizes).

        Returns:
            True on success, False on any exception.
        """
        assert hasattr(self, "state_dict_info") and self.state_dict_info is not None, (
            "state_dict_info is not set. "
            "VllmGeneration.prepare_refit_info() must be called before "
            "update_weights_from_collective()."
        )

        try:
            packed_broadcast_consumer(
                iterator         = iter(self.state_dict_info.items()),
                group            = self.model_update_group,
                src              = 0,
                post_unpack_func = self._load_weights,
            )
            self._maybe_process_fp8_kv_cache()
            return True

        except Exception as exc:
            print(
                f"Error in VllmInternalWorkerExtension"
                f".update_weights_from_collective: {exc}\n"
                f"{traceback.format_exc()}"
            )
            return False

    # IPC ZMQ weight update (non-collocated kept for completeness)
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive weights via CUDA IPC handles over a ZMQ REP socket.

        This path is preserved for completeness but is NOT used in the
        Dockyard non-collocated inference architecture.  The NCCL collective
        path (update_weights_from_collective) is always used.

        Returns:
            True on success, False on any exception.
        """
        from dockyard_rl.models.policy.ipc_utils import (  # type: ignore[import]
            IPCProtocol,
            calculate_aligned_size,
            rebuild_cuda_tensor_from_ipc,
        )

        buffer  = None
        weights = None

        try:
            self.maybe_init_zmq()

            while True:
                payload = self.zmq_socket.recv_pyobj()

                if payload == IPCProtocol.COMPLETE:
                    from vllm.model_executor.model_loader.utils import (
                        process_weights_after_loading,
                    )
                    process_weights_after_loading(
                        self.model_runner.model,
                        self.model_config,
                        self.device,
                    )
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer  = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)
                weights = []
                offset  = 0

                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]
                    if isinstance(shape, list):
                        shape = torch.Size(shape)
                    size_bytes    = dtype.itemsize * shape.numel()
                    weight        = (
                        buffer[offset: offset + size_bytes]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    if "GptOssForCausalLM" in (
                        self.model_runner.vllm_config.model_config.architectures
                        if hasattr(self.model_runner, "vllm_config") else []
                    ):
                        weight = fix_gpt_oss_export_transpose(key, weight)
                    weights.append((key, weight))
                    offset += calculate_aligned_size(size_bytes)

                assert offset == used_bytes, (
                    f"Offset ({offset}) != used_bytes ({used_bytes}). "
                    "Likely a state_dict_info shape/dtype mismatch."
                )

                self._load_weights(weights)
                torch.cuda.current_stream().synchronize()

                # CRITICAL: delete views before ACK to prevent use-after-free.
                del weights, buffer
                weights = buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            self._maybe_process_fp8_kv_cache()
            gc.collect()
            torch.cuda.empty_cache()
            return True

        except Exception as exc:
            print(
                f"Error in VllmInternalWorkerExtension"
                f".update_weights_via_ipc_zmq: {exc}\n"
                f"{traceback.format_exc()}"
            )
            return False

    # Lifecycle
    def cleanup(self) -> None:
        """Release ZMQ socket resources on worker shutdown."""
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()
        if hasattr(self, "model_update_group"):
            self.model_update_group.destroy()

    # Profiling
    def start_gpu_profiling(self) -> None:
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        torch.cuda.profiler.stop()