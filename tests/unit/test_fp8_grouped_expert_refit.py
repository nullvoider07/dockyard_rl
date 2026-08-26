"""Grouped-expert and wrapper-model refit through the FP8 quantizer (#2744).

``quantization/fp8.py::_is_fp8_weight`` decided what to quantize from a
``.weight`` suffix. Two refit names that target FP8 parameters do not have one,
and both were silently forwarded at high precision:

  * a grouped expert slab -- ``...mlp.experts.gate_up_proj`` / ``.down_proj`` --
    which ``DTensorPolicyWorker`` streams verbatim for an HF model that keeps
    its experts fused (``vllm_backend.fix_gpt_oss_export_transpose`` keys on
    exactly that name), and which vLLM maps straight onto ``w13_weight`` /
    ``w2_weight``;
  * every weight of a multimodal wrapper, whose vLLM module tree is addressed
    through ``hf_to_vllm_mapper`` (Qwen2.5-VL rewrites the prefix ``model.`` to
    ``language_model.model.``) -- the raw name failed on its first component,
    the traversal returned a partially walked module, and the FP8 dtype check
    answered "not FP8".

Nothing errors in either case: vLLM's weight loader casts the high-precision
tensor element-wise into the FP8 parameter while the block scales keep their
previous values.

The scale companion is why the grouped slab is refused rather than quantized:
``load_weights`` emits ``<name>_scale_inv``, which only reaches the parameter
because the per-linear and per-expert mappings key on ``.weight``. Verified
against vLLM 0.27.1 -- ``gpt_oss.py``'s ``WeightsMapper`` maps ``.gate_up_proj``
to ``.w13_weight`` and carries no entry for ``.gate_up_proj_scale_inv``.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest
import torch

_FP8_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "models",
    "generation",
    "vllm",
    "quantization",
    "fp8.py",
)


def _load_fp8_module() -> Any:
    pytest.importorskip("vllm")
    spec = importlib.util.spec_from_file_location("dockyard_fp8_grouped", _FP8_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - env-dependent heavy imports
        pytest.skip(f"fp8.py not importable in this environment: {exc}")
    setattr(module, "global_fp8_config", module.FP8Config(model_parallel_size=1))
    module.fp8_state.seen_params.clear()
    module.fp8_state.fp8_param_names.clear()
    return module


def _weights_mapper():
    utils = pytest.importorskip("vllm.model_executor.models.utils")
    return utils.WeightsMapper


class _FakeFusedMoE(torch.nn.Module):
    """Holds the two parameters ``is_fused_moe_module`` keys on, in FP8."""

    def __init__(self, fp8: bool = True) -> None:
        super().__init__()
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        # Plain attributes, not Parameters: the discriminator uses hasattr and
        # the caller reads only `.dtype`.
        self.w13_weight = torch.zeros(2, 256, 128, dtype=dtype)
        self.w2_weight = torch.zeros(2, 128, 256, dtype=dtype)


def _build_model(fp8: bool = True) -> Any:
    """`model.layers.0.mlp.experts` -> a fused-MoE module, no name mapping."""
    experts = _FakeFusedMoE(fp8=fp8)
    mlp = torch.nn.Module()
    mlp.experts = experts
    layer = torch.nn.Module()
    layer.mlp = mlp
    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([layer])
    root = torch.nn.Module()
    root.model = inner
    return root


def _build_vl_model(mapper_cls) -> Any:
    """A wrapper whose experts live under `language_model.model.layers.*`.

    Mirrors Qwen2.5-VL: the wrapper has no `model` attribute at all, and its
    `hf_to_vllm_mapper` is what turns the checkpoint prefix into the real path.
    """
    root = _build_model()
    wrapper = torch.nn.Module()
    wrapper.language_model = root
    wrapper.hf_to_vllm_mapper = mapper_cls(
        orig_to_new_prefix={"model.": "language_model.model."}
    )
    return wrapper


class _FakeModelRunner:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.loaded: list[Any] = []
        model.load_weights = self.loaded.extend  # type: ignore[assignment]


class TestGroupedExpertClassification:
    """The prefilter change: a grouped slab is now seen as an FP8 target."""

    NAME = "model.layers.0.mlp.experts.gate_up_proj"

    def test_grouped_slab_resolves_to_the_fused_moe_module(self):
        fp8 = _load_fp8_module()
        model = _build_model()

        resolved = fp8._get_module_from_param_name(model, self.NAME)

        assert resolved is model.model.layers[0].mlp.experts

    def test_grouped_slab_is_classified_fp8(self):
        # Fails if the `name.endswith("weight")` prefilter is restored: the slab
        # would never be inspected and would be forwarded unquantized.
        fp8 = _load_fp8_module()

        assert fp8._is_fp8_weight(self.NAME, _build_model())

    def test_grouped_slab_on_a_bf16_moe_is_not_fp8(self):
        fp8 = _load_fp8_module()

        assert not fp8._is_fp8_weight(self.NAME, _build_model(fp8=False))

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.w13_input_scale",
            "model.layers.0.mlp.experts.w13_weight_scale_inv",
        ],
    )
    def test_biases_and_scales_are_never_quantized(self, name):
        fp8 = _load_fp8_module()

        assert not fp8._is_fp8_weight(name, _build_model())

    def test_per_expert_weight_still_classified_fp8(self):
        # The path that already worked, pinned so the prefilter change does not
        # narrow it.
        fp8 = _load_fp8_module()
        name = "model.layers.0.mlp.experts.3.gate_proj.weight"

        assert fp8._is_fp8_weight(name, _build_model())


class TestScaleNameRouting:
    """`<name>_scale_inv` must land on the block scale of `<name>`'s param."""

    def test_plain_weight_routes_without_a_mapper(self):
        fp8 = _load_fp8_module()

        assert fp8._fp8_scale_name_routes(
            _build_model(), "model.layers.0.mlp.experts.3.gate_proj.weight"
        )

    def test_prefix_mapper_rewrites_weight_and_scale_alike(self):
        fp8 = _load_fp8_module()
        model = _build_vl_model(_weights_mapper())

        assert fp8._fp8_scale_name_routes(
            model, "model.layers.0.mlp.experts.3.gate_proj.weight"
        )

    def test_grouped_slab_has_no_scale_route_under_the_gpt_oss_mapping(self):
        # The real GptOss suffix mapping: `.gate_up_proj` -> `.w13_weight`, with
        # no entry for `.gate_up_proj_scale_inv`.
        fp8 = _load_fp8_module()
        model = _build_model()
        model.hf_to_vllm_mapper = _weights_mapper()(
            orig_to_new_suffix={
                ".gate_up_proj": ".w13_weight",
                ".down_proj": ".w2_weight",
            }
        )

        assert not fp8._fp8_scale_name_routes(
            model, "model.layers.0.mlp.experts.gate_up_proj"
        )

    def test_gpt_oss_carries_no_grouped_scale_entry_upstream(self):
        """Pins the premise against the installed vLLM, not a recollection."""
        gpt_oss = pytest.importorskip("vllm.model_executor.models.gpt_oss")
        suffixes = gpt_oss.GptOssForCausalLM.hf_to_vllm_mapper.orig_to_new_suffix

        assert suffixes[".gate_up_proj"] == ".w13_weight"
        assert suffixes[".down_proj"] == ".w2_weight"
        assert not any(key.endswith("_scale_inv") for key in suffixes)


class TestWrapperTraversal:
    """Multimodal wrappers resolve through `hf_to_vllm_mapper`."""

    NAME = "model.layers.0.mlp.experts.gate_up_proj"

    def test_wrapper_resolves_through_the_mapper(self):
        fp8 = _load_fp8_module()
        model = _build_vl_model(_weights_mapper())

        resolved = fp8._get_module_from_param_name(model, self.NAME)

        assert resolved is model.language_model.model.layers[0].mlp.experts

    def test_wrapper_without_the_mapping_resolves_to_nothing(self):
        # Cutting the wire: the same tree with no mapper is exactly the state
        # this fix repairs, and it must report failure rather than hand back a
        # partially walked module that answers "not FP8".
        fp8 = _load_fp8_module()
        model = _build_vl_model(_weights_mapper())
        del model.hf_to_vllm_mapper

        assert fp8._get_module_from_param_name(model, self.NAME) is None
        assert not fp8._is_fp8_weight(self.NAME, model)

    def test_dropped_name_resolves_to_nothing(self):
        fp8 = _load_fp8_module()
        model = _build_model()
        model.hf_to_vllm_mapper = _weights_mapper()(orig_to_new_prefix={"model.": None})

        assert fp8._get_module_from_param_name(model, self.NAME) is None


class TestLoadWeightsRefusal:
    """`load_weights` refuses what it cannot scale instead of forwarding it."""

    def test_grouped_slab_is_refused(self):
        fp8 = _load_fp8_module()
        model = _build_model()
        model.hf_to_vllm_mapper = _weights_mapper()(
            orig_to_new_suffix={".gate_up_proj": ".w13_weight"}
        )
        runner = _FakeModelRunner(model)
        name = "model.layers.0.mlp.experts.gate_up_proj"

        # The scale-route check runs first, so a grouped slab is reported for
        # the reason that actually blocks it rather than for its rank.
        with pytest.raises(ValueError, match=r"drop the scale"):
            fp8.load_weights([(name, torch.randn(2, 256, 128))], runner)

        assert runner.loaded == []

    def test_unroutable_scale_is_refused_at_any_rank(self):
        # Isolates the scale-route rule from the 2D cast limit: a rank-2 tensor
        # passes the cast but its scale still has nowhere to go.
        fp8 = _load_fp8_module()
        model = _build_model()
        model.hf_to_vllm_mapper = _weights_mapper()(
            orig_to_new_suffix={".gate_up_proj": ".w13_weight"}
        )
        runner = _FakeModelRunner(model)
        name = "model.layers.0.mlp.experts.gate_up_proj"

        with pytest.raises(ValueError, match=r"drop the scale"):
            fp8.load_weights([(name, torch.randn(256, 128))], runner)

        assert runner.loaded == []

    def test_three_dimensional_fp8_target_is_refused(self):
        fp8 = _load_fp8_module()
        runner = _FakeModelRunner(_build_model())
        name = "model.layers.0.mlp.experts.gate_up_proj"

        with pytest.raises(ValueError, match=r"2D-only"):
            fp8.load_weights([(name, torch.randn(2, 256, 128))], runner)

    def test_per_expert_weight_is_quantized_with_its_scale(self):
        fp8 = _load_fp8_module()
        runner = _FakeModelRunner(_build_model())
        name = "model.layers.0.mlp.experts.3.gate_proj.weight"

        fp8.load_weights([(name, torch.randn(256, 128))], runner)

        loaded = dict((k, v) for k, v in runner.loaded)
        assert loaded[name].dtype == torch.float8_e4m3fn
        assert tuple(loaded[f"{name}_scale_inv"].shape) == (2, 1)

    def test_non_fp8_weight_passes_through_untouched(self):
        fp8 = _load_fp8_module()
        runner = _FakeModelRunner(_build_model(fp8=False))
        name = "model.layers.0.mlp.experts.3.gate_proj.weight"
        tensor = torch.randn(256, 128)

        fp8.load_weights([(name, tensor)], runner)

        assert runner.loaded == [(name, tensor)]
