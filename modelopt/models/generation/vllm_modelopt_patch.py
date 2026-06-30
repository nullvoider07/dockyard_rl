"""vLLM ModelOpt NVFP4 patches for dense rollout weight reloads.

The real-quant rollout serves a ModelOpt NVFP4 model through vLLM's native FP4
kernel. vLLM's ``process_weights_after_loading`` converts the loaded HF-named
params (``weight`` / ``weight_scale`` / ``weight_scale_2``) into the kernel
layout once, deleting/renaming params and dropping ``weight_loader`` references
in the process. That is fine for a static deployment but breaks RL refit, where
fresh NVFP4 weights stream in every step and must be re-loadable.

These patches make the conversion repeatable:

- ``apply_modelopt_nvfp4_patches`` replaces the dense NVFP4 linear method's
  ``process_weights_after_loading`` / ``apply`` and wraps the config's
  ``_from_config`` so a weight-only (W4A16) checkpoint is tagged and routed to
  the Marlin FP4 GEMM path.
- Before converting, the loaded param shape/dtype/loader are captured
  (``_capture_modelopt_dense_param_reload_meta``) so a subsequent refit can
  restore loadable parameters via ``prepare_modelopt_for_weight_reload``.
- After a refit streams new weights, ``modelopt_process_weights_after_loading``
  re-runs the conversion.

W4A16 detection: the deployment config (built by
``modelopt.utils.build_vllm_modelopt_nvfp4_config``) carries ``quant_algo:
NVFP4`` so vLLM instantiates the dense ``ModelOptNvFp4LinearMethod``; the
``weight_only`` / ``quant_mode`` markers on the original config then select the
W4A16 Marlin path inside the patched method.
"""

import torch
from torch.nn import Parameter

_DENSE_HF_PARAMS = ("weight", "weight_scale", "weight_scale_2")
_W4A16_QUANT_MODES = frozenset({"w4a16_nvfp4", "nvfp4_w4a16"})
_W4A16_ATTR = "_dky_weight_only_w4a16"
_ORIGINAL_NVFP4_CONFIG_FROM_CONFIG_ATTR = "_dky_original_from_config"
_ORIGINAL_LINEAR_APPLY_ATTR = "_dky_original_apply"
_PARAM_META_ATTR = "_dky_modelopt_param_meta"
_WEIGHT_LOADERS_ATTR = "_dky_modelopt_weight_loaders"


def _unwrap_vllm_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def _canonicalize_nvfp4_weight_scale(layer: torch.nn.Module) -> None:
    weight_scale = layer.weight_scale
    scale = weight_scale.data.to(torch.float32).abs().to(weight_scale.dtype)
    weight_scale.data.copy_(scale)


def _requests_w4a16_modelopt_config(config: dict) -> bool:
    quant_mode = config.get("quant_mode")
    if isinstance(quant_mode, str) and quant_mode.lower() in _W4A16_QUANT_MODES:
        return True
    if config.get("weight_only") is True:
        return True

    nested = config.get("quantization")
    return isinstance(nested, dict) and _requests_w4a16_modelopt_config(nested)


def _is_w4a16_modelopt_quant_config(quant_config) -> bool:
    return bool(getattr(quant_config, _W4A16_ATTR, False))


def _modelopt_nvfp4_config_from_config(cls, *args, **kwargs):
    original_from_config = getattr(cls, _ORIGINAL_NVFP4_CONFIG_FROM_CONFIG_ATTR)
    quant_config = original_from_config(*args, **kwargs)

    original_config = kwargs.get("original_config")
    if isinstance(original_config, dict) and _requests_w4a16_modelopt_config(
        original_config
    ):
        setattr(quant_config, _W4A16_ATTR, True)

    return quant_config


def _convert_nvfp4_linear_kernel_format(quant_method, layer: torch.nn.Module) -> None:
    kernel = getattr(quant_method, "kernel", None)
    if kernel is not None:
        # vLLM's dense NVFP4 linear method carries a kernel adapter that performs
        # the post-load conversion (the path taken on current vLLM).
        kernel.process_weights_after_loading(layer)
        return

    # Fallback for a kernel-less method object (older/other vLLM builds): convert
    # via the standalone helper if this build exposes it.
    try:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (  # type: ignore[attr-defined]
            convert_to_nvfp4_linear_kernel_format,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ModelOpt NVFP4 linear method exposes no 'kernel' and this vLLM build "
            "has no convert_to_nvfp4_linear_kernel_format fallback; cannot convert "
            "weights to the NVFP4 kernel format."
        ) from exc

    convert_to_nvfp4_linear_kernel_format(quant_method.backend, layer)


def _convert_w4a16_linear_kernel_format(layer: torch.nn.Module) -> None:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_fp4_layer_for_marlin,
    )

    prepare_fp4_layer_for_marlin(layer)


def _capture_modelopt_dense_param_reload_meta(layer: torch.nn.Module) -> None:
    if not hasattr(layer, _PARAM_META_ATTR):
        setattr(layer, _PARAM_META_ATTR, {})
        setattr(layer, _WEIGHT_LOADERS_ATTR, {})
    elif not hasattr(layer, _WEIGHT_LOADERS_ATTR):
        setattr(layer, _WEIGHT_LOADERS_ATTR, {})

    param_meta = getattr(layer, _PARAM_META_ATTR)
    weight_loaders = getattr(layer, _WEIGHT_LOADERS_ATTR)

    for param_name in _DENSE_HF_PARAMS:
        if param_name in param_meta:
            continue
        param = getattr(layer, param_name)
        meta = {
            "shape": tuple(param.shape),
            "dtype": param.dtype,
            "device": str(param.device),
            "param_class": type(param),
        }
        if hasattr(param, "_input_dim"):
            meta["input_dim"] = param._input_dim
        if hasattr(param, "_output_dim"):
            meta["output_dim"] = param._output_dim
        param_meta[param_name] = meta
        if hasattr(param, "weight_loader"):
            weight_loaders[param_name] = param.weight_loader


def _modelopt_dense_process_w4a16_weights(self, layer: torch.nn.Module) -> None:
    """Convert dense ModelOpt NVFP4 W4A16 weights for the Marlin weight-only GEMM."""
    _capture_modelopt_dense_param_reload_meta(layer)

    weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    delattr(layer, "weight_scale_2")

    for attr in (
        "input_scale",
        "input_global_scale",
        "alpha",
        "input_global_scale_inv",
    ):
        if hasattr(layer, attr):
            delattr(layer, attr)

    _canonicalize_nvfp4_weight_scale(layer)
    _convert_w4a16_linear_kernel_format(layer)


def _modelopt_dense_process_weights(self, layer: torch.nn.Module) -> None:
    """Convert dense ModelOpt NVFP4 weights after initial load or refit."""
    if _is_w4a16_modelopt_quant_config(getattr(self, "quant_config", None)):
        _modelopt_dense_process_w4a16_weights(self, layer)
        return

    _capture_modelopt_dense_param_reload_meta(layer)

    input_global_scale = torch.ones(
        (),
        dtype=torch.float32,
        device=layer.weight.device,
    )
    layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)

    weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    delattr(layer, "weight_scale_2")

    layer.alpha = Parameter(
        layer.input_global_scale * layer.weight_global_scale,
        requires_grad=False,
    )
    layer.input_global_scale_inv = Parameter(
        (1.0 / layer.input_global_scale).to(torch.float32),
        requires_grad=False,
    )

    _canonicalize_nvfp4_weight_scale(layer)
    _convert_nvfp4_linear_kernel_format(self, layer)


def _modelopt_dense_apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if _is_w4a16_modelopt_quant_config(getattr(self, "quant_config", None)):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
        )

        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    original_apply = getattr(type(self), _ORIGINAL_LINEAR_APPLY_ATTR, None)
    if original_apply is not None:
        return original_apply(self, layer, x, bias)

    return self.kernel.apply_weights(layer=layer, x=x, bias=bias)


def prepare_modelopt_for_weight_reload(model, device=None) -> None:
    """Restore loadable HF-named params before a refit weight reload cycle.

    The kernel conversion replaces ``weight`` / ``weight_scale`` /
    ``weight_scale_2`` with packed buffers that have no ``weight_loader``. This
    rebuilds each captured param to its loaded shape/dtype/class with the
    original loader attached, so vLLM's weight loader can stream fresh NVFP4
    weights into them.
    """
    inner_model = _unwrap_vllm_model(model)
    for module in inner_model.modules():
        layer_meta = getattr(module, _PARAM_META_ATTR, None)
        if layer_meta is None:
            continue
        weight_loaders = getattr(module, _WEIGHT_LOADERS_ATTR, {})
        for param_name, meta in layer_meta.items():
            param = getattr(module, param_name, None)
            weight_loader = weight_loaders.get(param_name)
            param_class = meta["param_class"]
            if (
                param is None
                or tuple(param.shape) != tuple(meta["shape"])
                or param.dtype != meta["dtype"]
                or (
                    weight_loader is not None
                    and (
                        not isinstance(param, param_class)
                        or not hasattr(param, "weight_loader")
                    )
                )
            ):
                data = torch.empty(
                    meta["shape"],
                    dtype=meta["dtype"],
                    device=device or meta["device"],
                )
                if param_class is not Parameter and weight_loader is not None:
                    kwargs = {"data": data, "weight_loader": weight_loader}
                    if "input_dim" in meta:
                        kwargs["input_dim"] = meta["input_dim"]
                    if "output_dim" in meta:
                        kwargs["output_dim"] = meta["output_dim"]
                    replacement = param_class(**kwargs)
                else:
                    replacement = Parameter(data, requires_grad=False)
                    if weight_loader is not None:
                        replacement.weight_loader = weight_loader
                setattr(module, param_name, replacement)


def modelopt_process_weights_after_loading(model) -> None:
    """Re-run vLLM ModelOpt post-load conversion for dense quantized layers."""
    actual_model = _unwrap_vllm_model(model)

    for module in actual_model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method.__class__.__name__ == "ModelOptNvFp4LinearMethod":
            quant_method.process_weights_after_loading(module)


_patched = False


def apply_modelopt_nvfp4_patches() -> None:
    """Patch vLLM's dense ModelOpt NVFP4 method for rollout refits (idempotent)."""
    global _patched

    if _patched:
        return

    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4LinearMethod,
    )

    if not hasattr(ModelOptNvFp4Config, _ORIGINAL_NVFP4_CONFIG_FROM_CONFIG_ATTR):
        setattr(
            ModelOptNvFp4Config,
            _ORIGINAL_NVFP4_CONFIG_FROM_CONFIG_ATTR,
            ModelOptNvFp4Config._from_config,
        )
    ModelOptNvFp4Config._from_config = classmethod(_modelopt_nvfp4_config_from_config)

    if not hasattr(ModelOptNvFp4LinearMethod, _ORIGINAL_LINEAR_APPLY_ATTR):
        setattr(
            ModelOptNvFp4LinearMethod,
            _ORIGINAL_LINEAR_APPLY_ATTR,
            ModelOptNvFp4LinearMethod.apply,
        )
    ModelOptNvFp4LinearMethod.process_weights_after_loading = (
        _modelopt_dense_process_weights
    )
    ModelOptNvFp4LinearMethod.apply = _modelopt_dense_apply

    _patched = True
