"""CPU tests for the ModelOpt NVFP4 real-quant rollout wiring.

The live FP4 kernel and the end-to-end refit re-pack require Blackwell/H100-class
FP4 hardware and are GPU-deferred. These tests cover the CPU-checkable surface:
the deployment quantization_config builder, the quant-ignore pattern matching,
the W4A16 detection, and the refit param capture/reload roundtrip (pure torch).
"""

import torch
import pytest
from torch.nn import Module, Parameter

from dockyard_rl.modelopt import utils as U
from dockyard_rl.modelopt.models.generation import vllm_modelopt_patch as P


# -- deployment quantization_config -------------------------------------------

def test_build_nvfp4_config_shape_and_markers():
    cfg = U.build_vllm_modelopt_nvfp4_config()
    assert cfg["quant_method"] == "modelopt"
    assert cfg["quant_algo"] == "NVFP4"
    assert cfg["quant_mode"] == "w4a16_nvfp4"
    assert cfg["weight_only"] is True
    assert cfg["group_size"] == 16
    weights = cfg["config_groups"]["group_0"]["weights"]
    assert weights["num_bits"] == 4 and weights["type"] == "float"
    assert cfg["config_groups"]["group_0"]["input_activations"] is None


def test_build_nvfp4_config_default_vs_custom_ignore():
    default = U.build_vllm_modelopt_nvfp4_config()
    assert default["ignore"] == U.DEFAULT_NVFP4_IGNORE
    assert default["ignore"] is not U.DEFAULT_NVFP4_IGNORE  # copied, not aliased
    custom = U.build_vllm_modelopt_nvfp4_config(ignore=["only_this"])
    assert custom["ignore"] == ["only_this"]


# -- quant-ignore pattern matching --------------------------------------------

def test_ignore_matches_lm_head_and_attention():
    pats = U.DEFAULT_NVFP4_IGNORE
    assert U.matches_quant_ignore_pattern("lm_head.weight", pats)
    assert U.matches_quant_ignore_pattern("model.layers.3.self_attn.q_proj.weight", pats)
    assert U.matches_quant_ignore_pattern("model.layers.3.self_attention.qkv.weight", pats)


def test_ignore_does_not_match_mlp_projection():
    pats = U.DEFAULT_NVFP4_IGNORE
    assert not U.matches_quant_ignore_pattern("model.layers.0.mlp.up_proj.weight", pats)
    assert not U.matches_quant_ignore_pattern("model.layers.0.mlp.down_proj.weight_scale", pats)


def test_ignore_candidate_variants_cover_prefix_and_suffix():
    cands = set(U.iter_quant_ignore_name_candidates("model.lm_head.weight"))
    # bare, suffix-stripped, and model.-prefix-toggled forms all present.
    assert "model.lm_head.weight" in cands
    assert "model.lm_head" in cands
    assert "lm_head.weight" in cands
    assert "lm_head" in cands


# -- W4A16 detection ----------------------------------------------------------

@pytest.mark.parametrize(
    "config,expected",
    [
        ({"weight_only": True}, True),
        ({"quant_mode": "w4a16_nvfp4"}, True),
        ({"quant_mode": "NVFP4_W4A16"}, True),
        ({"quantization": {"weight_only": True}}, True),
        ({"quant_algo": "NVFP4"}, False),
        ({"quant_mode": "nvfp4"}, False),
    ],
)
def test_requests_w4a16_detection(config, expected):
    assert P._requests_w4a16_modelopt_config(config) is expected


def test_is_w4a16_quant_config_reads_attr():
    class _C:
        pass

    c = _C()
    assert P._is_w4a16_modelopt_quant_config(c) is False
    setattr(c, P._W4A16_ATTR, True)
    assert P._is_w4a16_modelopt_quant_config(c) is True
    assert P._is_w4a16_modelopt_quant_config(None) is False


# -- refit param capture / reload roundtrip -----------------------------------

def _mk_param(shape, dtype):
    p = Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
    p.weight_loader = lambda *a, **k: None  # marker loader
    return p


def _make_loaded_layer():
    layer = Module()
    layer.weight = _mk_param((4, 2), torch.uint8)           # packed NVFP4
    layer.weight_scale = _mk_param((4, 1), torch.float8_e4m3fn)
    layer.weight_scale_2 = _mk_param((1,), torch.float32)   # global scale
    return layer


def test_capture_records_meta_and_loaders():
    layer = _make_loaded_layer()
    P._capture_modelopt_dense_param_reload_meta(layer)
    meta = getattr(layer, P._PARAM_META_ATTR)
    loaders = getattr(layer, P._WEIGHT_LOADERS_ATTR)
    assert set(meta) == {"weight", "weight_scale", "weight_scale_2"}
    assert meta["weight"]["shape"] == (4, 2)
    assert meta["weight"]["dtype"] == torch.uint8
    assert set(loaders) == {"weight", "weight_scale", "weight_scale_2"}


def test_prepare_reload_restores_deleted_and_mangled_params():
    layer = _make_loaded_layer()
    P._capture_modelopt_dense_param_reload_meta(layer)

    # Simulate the kernel conversion mangling the layer: drop the global scale
    # and replace the weight with a Marlin-packed buffer of a different shape
    # that has no weight_loader.
    delattr(layer, "weight_scale_2")
    layer.weight = Parameter(torch.zeros((4, 8), dtype=torch.uint8), requires_grad=False)

    P.prepare_modelopt_for_weight_reload(layer)

    # weight restored to loaded shape with a loader reattached.
    assert tuple(layer.weight.shape) == (4, 2)
    assert layer.weight.dtype == torch.uint8
    assert hasattr(layer.weight, "weight_loader")
    # weight_scale_2 recreated with a loader.
    assert hasattr(layer, "weight_scale_2")
    assert tuple(layer.weight_scale_2.shape) == (1,)
    assert hasattr(layer.weight_scale_2, "weight_loader")


def test_prepare_reload_leaves_intact_params_untouched():
    layer = _make_loaded_layer()
    P._capture_modelopt_dense_param_reload_meta(layer)
    untouched = layer.weight_scale  # correct shape/dtype/loader, not mangled
    P.prepare_modelopt_for_weight_reload(layer)
    assert layer.weight_scale is untouched


def test_capture_is_idempotent():
    layer = _make_loaded_layer()
    P._capture_modelopt_dense_param_reload_meta(layer)
    first = dict(getattr(layer, P._PARAM_META_ATTR))
    # Re-capture after a shape change must NOT overwrite the original meta.
    layer.weight = Parameter(torch.zeros((4, 99), dtype=torch.uint8), requires_grad=False)
    P._capture_modelopt_dense_param_reload_meta(layer)
    assert getattr(layer, P._PARAM_META_ATTR)["weight"]["shape"] == first["weight"]["shape"]
