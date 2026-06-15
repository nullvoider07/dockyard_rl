import types
from contextlib import ExitStack, contextmanager
import torch
import vllm  # noqa: F401
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from dockyard_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension

class VllmQuantInternalWorkerExtension(VllmInternalWorkerExtension):
    @contextmanager
    def _patch_named_parameters_to_include_buffers(self, model):
        """Temporarily patches model.named_parameters() to also yield input_quantizer buffers.

        Weights arrive pre-folded from the policy side, so only input_quantizer
        amax buffers need to be loaded. Weight quantizer buffers are skipped.
        """
        original_named_parameters = model.named_parameters
        patched_quantizer_buffers = []

        def input_amax_loader(param, loaded_weight, *args, **kwargs):
            param.copy_(torch.max(param, loaded_weight))

        def new_named_parameters(self, *args, **kwargs):
            yield from original_named_parameters(*args, **kwargs)
            for name, buf in self.named_buffers(*args, **kwargs):
                if "input_quantizer" not in name:
                    continue
                if not hasattr(buf, "weight_loader"):
                    buf.weight_loader = input_amax_loader
                    patched_quantizer_buffers.append(buf)
                yield name, buf

        model.named_parameters = types.MethodType(new_named_parameters, model)
        try:
            yield
        finally:
            model.named_parameters = original_named_parameters
            for buf in patched_quantizer_buffers:
                if hasattr(buf, "weight_loader"):
                    del buf.weight_loader

    def _load_weights(self, weights):
        """Load pre-folded weights and input_quantizer amax buffers.

        Weights arrive already folded (weight_quantizer applied during export),
        so no fold_weight step is needed here.
        """
        with ExitStack() as contexts:
            for _, child in self.model_runner.model.named_children():
                contexts.enter_context(
                    self._patch_named_parameters_to_include_buffers(child)
                )
            return super()._load_weights(weights)

    def get_weight_snapshot(self, name: str) -> torch.Tensor:
        """Return a CPU copy of a named parameter for before/after comparison."""
        model = self.model_runner.model
        for n, p in model.named_parameters():
            if n == name:
                return p.detach().cpu().clone()
        raise KeyError(f"Parameter '{name}' not found in model")

    def export_amax(self) -> dict[str, torch.Tensor]:
        """Export amax buffers from the model for testing/debugging."""
        try:
            model = self.model_runner.model
            return {
                n: b.detach().cpu()
                for n, b in model.named_buffers()
                if n.endswith("amax")
            }
        except AttributeError:
            return {}

    def get_quantizer_stats(self) -> dict:
        """Return summary statistics for all TensorQuantizer modules."""
        total = 0
        enabled = 0
        with_amax = 0
        positive_amax = 0
        try:
            model = self.model_runner.model
        except AttributeError:
            return {"total": 0, "enabled": 0, "with_amax": 0, "positive_amax": 0}
        for _, module in model.named_modules():
            if isinstance(module, TensorQuantizer):
                total += 1
                if module.is_enabled:
                    enabled += 1
                    if hasattr(module, "amax") and module.amax is not None:
                        with_amax += 1
                        if isinstance(module.amax, torch.Tensor) and (module.amax > 0).all():
                            positive_amax += 1
        return {
            "total": total,
            "enabled": enabled,
            "with_amax": with_amax,
            "positive_amax": positive_amax,
        }