import os
from typing import Any
import ray
from dockyard_rl.distributed.worker_groups_utils import get_nsight_config_if_pattern_matches
from dockyard_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from dockyard_rl.models.generation.vllm.vllm_worker_async import VllmAsyncGenerationWorkerImpl

@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)  # pragma: no cover
class VllmQuantGenerationWorker(VllmGenerationWorkerImpl):
    def __init__(self, *args, **kwargs):
        kwargs["extra_env_vars"] = ["VLLM_QUANT_CFG"]
        super().__init__(*args, **kwargs)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        llm_kwargs["worker_cls"] = (
            "dockyard_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
        )
        llm_kwargs["worker_extension_cls"] = (
            "dockyard_rl.modelopt.models.generation.vllm_quant_backend.VllmQuantInternalWorkerExtension"
        )
        # Expert fakequant needs a fused-MoE backend that calibrates per-expert;
        # default to triton when the recipe does not pin one (explicit config wins).
        llm_kwargs.setdefault("moe_backend", "triton")
        quant_cfg = self.cfg.get("quant_cfg")
        if quant_cfg:
            print("setting VLLM_QUANT_CFG to:", quant_cfg)
            os.environ["VLLM_QUANT_CFG"] = quant_cfg

        super()._create_engine(llm_kwargs)

    def _collective_rpc_or_empty(self, method: str) -> dict[str, Any]:
        """Best-effort RPC call; returns {} on any failure.

        collective_rpc can propagate arbitrary exceptions from the internal
        worker, so broad except is intentional here.
        """
        if not hasattr(self, "llm"):
            return {}
        assert self.llm is not None
        try:
            results = self.llm.collective_rpc(method, args=tuple())
            return results[0] if results else {}
        except Exception:
            return {}

    def export_amax(self) -> dict[str, Any]:
        """Export amax buffers for testing/debugging."""
        return self._collective_rpc_or_empty("export_amax")

    def get_quantizer_stats(self) -> dict[str, Any]:
        """Return quantizer statistics."""
        return self._collective_rpc_or_empty("get_quantizer_stats")

    def get_weight_snapshot(self, name: str) -> Any:
        """Return a CPU copy of a named parameter for before/after comparison."""
        if not hasattr(self, "llm"):
            return None
        assert self.llm is not None
        results = self.llm.collective_rpc("get_weight_snapshot", args=(name,))
        return results[0] if results else None

@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmQuantAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    def __init__(self, *args, **kwargs):
        kwargs["extra_env_vars"] = ["VLLM_QUANT_CFG"]
        super().__init__(*args, **kwargs)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        llm_kwargs["worker_cls"] = (
            "dockyard_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
        )
        llm_kwargs["worker_extension_cls"] = (
            "dockyard_rl.modelopt.models.generation.vllm_quant_backend.VllmQuantInternalWorkerExtension"
        )
        # Expert fakequant needs a fused-MoE backend that calibrates per-expert;
        # default to triton when the recipe does not pin one (explicit config wins).
        llm_kwargs.setdefault("moe_backend", "triton")
        quant_cfg = self.cfg.get("quant_cfg")
        if quant_cfg:
            print("setting VLLM_QUANT_CFG to:", quant_cfg)
            os.environ["VLLM_QUANT_CFG"] = quant_cfg

        super()._create_engine(llm_kwargs)

    async def _collective_rpc_or_empty(self, method: str) -> dict[str, Any]:
        """Best-effort async RPC call; returns {} on any failure."""
        if not hasattr(self, "llm"):
            return {}
        try:
            results = await self.llm.collective_rpc(method, args=tuple())
            return results[0] if results else {}
        except Exception:
            return {}

    async def export_amax(self) -> dict[str, Any]:
        return await self._collective_rpc_or_empty("export_amax")

    async def get_quantizer_stats(self) -> dict[str, Any]:
        return await self._collective_rpc_or_empty("get_quantizer_stats")

    async def get_weight_snapshot(self, name: str) -> Any:
        if not hasattr(self, "llm"):
            return None
        results = await self.llm.collective_rpc("get_weight_snapshot", args=(name,))
        return results[0] if results else None