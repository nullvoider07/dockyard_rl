import os
from typing import Any, cast
import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.plugins.vllm import disable_compilation
from vllm.v1.worker.gpu_worker import Worker as BaseWorker
from dockyard_rl.modelopt.utils import resolve_quant_cfg


def _fakequant_run_prolog_worker(self) -> None:
    def calibrate_loop(model: Any = None) -> None:
        self.model_runner._dummy_run(1, skip_eplb=True, remove_lora=False)

    quant_cfg = resolve_quant_cfg(os.environ["VLLM_QUANT_CFG"])
    print(f"quant_cfg: {quant_cfg}")

    model: Any = self.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    with disable_compilation(model):
        print("quantizing model...")
        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        mtq.print_quant_summary(model)

    # Using dummy data for calibration — amax values will be loaded from the
    # policy actor during weight sync.
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            setattr(module, "_is_active", True)
            if hasattr(module, "amax") and module.amax is not None:
                cast(Any, module.amax).fill_(-1.0)
            # Disable weight quantizers for CUDA graph capture.
            if name.endswith("weight_quantizer"):
                module.disable()

class FakeQuantWorker(BaseWorker):
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        from typing import cast, Any as _Any
        model: _Any = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        with disable_compilation(model):
            return super().determine_available_memory()

    def compile_or_warm_up_model(self) -> Any:  # type: ignore[override]
        print(
            "os.environ.get('VLLM_QUANT_CFG'): ", os.environ.get("VLLM_QUANT_CFG", None)
        )
        if os.environ.get("VLLM_QUANT_CFG", None) is not None:
            _fakequant_run_prolog_worker(self)
        return super().compile_or_warm_up_model()