"""C1: device capability resolution and precision validation.

Before C1 the entire tree held exactly one arch check --
`torch.cuda.get_device_capability()[0] >= 8`, inline in vllm_worker.py as a
default for enable_prefix_caching. Nothing gated a requested precision on what
the hardware could run, so `precision: fp8` on an A100 failed somewhere inside
vLLM instead of at startup.

The thresholds asserted here were measured against the shipped vLLM 0.27.1
binary, whose `cutlass_scaled_mm_supports_fp8` / `..._block_fp8` predicates take
a capability integer and so are checkable with no GPU. Two of them contradict
what documentation alone would suggest, and both have their own test below:
dockyard's FP8 needs sm90 (not sm89) because it only emits block-scaled FP8,
and native bf16 needs sm80 even though torch.cuda.is_bf16_supported() reports
True below that by counting emulation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dockyard_rl.utils.device_capability import (
    SM_BLOCK_FP8,
    SM_FP8,
    SM_NATIVE_BF16,
    DeviceCapability,
    resolve_device_capability,
    validate_precision,
)


def _cap(major: int, minor: int, name: str = "test device") -> DeviceCapability:
    return DeviceCapability(major=major, minor=minor, name=name)


# The measured table. Keeping it as data makes the source of truth explicit and
# a threshold change fail loudly rather than drift.
ARCH_TABLE = [
    # (major, minor, name,              bf16,  fp8,   block_fp8, mxfp8)
    (7, 0, "V100 Volta",                False, False, False,     False),
    (7, 5, "T4 Turing",                 False, False, False,     False),
    (8, 0, "A100 Ampere",               True,  False, False,     False),
    (8, 6, "A10 / RTX 3090",            True,  False, False,     False),
    (8, 9, "L40S / L4 Ada",             True,  True,  False,     False),
    (9, 0, "H100 / H200 Hopper",        True,  True,  True,      False),
    (10, 0, "B200 Blackwell",           True,  True,  True,      True),
    (10, 3, "GB300",                    True,  True,  True,      True),
    (12, 0, "RTX PRO 6000 Blackwell",   True,  True,  True,      True),
]


class TestCapabilityTable:
    @pytest.mark.parametrize(
        "major,minor,name,bf16,fp8,block_fp8,mxfp8", ARCH_TABLE
    )
    def test_predicates_match_the_measured_table(
        self, major, minor, name, bf16, fp8, block_fp8, mxfp8
    ):
        cap = _cap(major, minor, name)
        assert cap.supports_native_bf16 is bf16
        assert cap.supports_fp8 is fp8
        assert cap.supports_block_fp8 is block_fp8
        assert cap.supports_mxfp8_nvfp4 is mxfp8

    def test_sm_is_the_integer_vllm_predicates_take(self):
        assert _cap(9, 0).sm == 90
        assert _cap(10, 3).sm == 103
        assert _cap(7, 5).sm == 75

    def test_describe_names_the_device_and_capability(self):
        """The refusal message is the whole point; it has to identify the card."""
        text = _cap(8, 0, "NVIDIA A100-SXM4-80GB").describe()
        assert "A100" in text and "8.0" in text and "sm80" in text


class TestTheTwoTrapsC1Exists_For:
    def test_fp8_and_block_fp8_diverge_at_ada(self):
        """sm89 passes a naive fp8 check and still cannot run dockyard's fp8.

        dockyard only ever emits block-scaled FP8: FP8_BLOCK_QUANT_KWARGS pins
        weight_block_size=[128, 128] with no config override, and is_fp8_model
        asserts block scaling is present. Block FP8 starts at sm90.
        """
        ada = _cap(8, 9, "L40S")
        assert ada.supports_fp8 is True
        assert ada.supports_block_fp8 is False
        assert SM_FP8 == 89 and SM_BLOCK_FP8 == 90

    def test_native_bf16_starts_at_ampere_not_wherever_torch_says(self):
        """torch.cuda.is_bf16_supported() counts emulation and lies on a T4.

        It defaults to including_emulation=True: below sm80 it falls through to
        _check_bf16_tensor_supported, which only asks whether a bf16 tensor can
        be created. This module must not inherit that answer.
        """
        assert _cap(7, 5, "Tesla T4").supports_native_bf16 is False
        assert _cap(8, 0, "A100").supports_native_bf16 is True
        assert SM_NATIVE_BF16 == 80


class TestValidatePrecisionRefuses:
    def test_bfloat16_on_a_t4_is_refused(self):
        with pytest.raises(ValueError, match="requires Ampere"):
            validate_precision("bfloat16", _cap(7, 5, "Tesla T4"), context="policy.precision")

    def test_the_bf16_refusal_names_the_emulation_trap(self):
        """An operator who checks is_bf16_supported() themselves needs this."""
        with pytest.raises(ValueError, match="is_bf16_supported"):
            validate_precision("bfloat16", _cap(7, 5, "Tesla T4"), context="policy.precision")

    def test_fp8_on_an_a100_is_refused(self):
        with pytest.raises(ValueError, match="Hopper"):
            validate_precision("fp8", _cap(8, 0, "A100"), context="vllm_cfg.precision")

    def test_fp8_on_ada_is_refused_and_explains_why(self):
        """The confusing case: the card does support FP8, just not this FP8."""
        with pytest.raises(ValueError, match="block-scaled FP8"):
            validate_precision("fp8", _cap(8, 9, "L40S"), context="vllm_cfg.precision")

    def test_the_message_carries_the_device_and_the_config_key(self):
        with pytest.raises(ValueError) as exc:
            validate_precision("fp8", _cap(8, 0, "NVIDIA A100"), context="policy.generation.vllm_cfg.precision")
        message = str(exc.value)
        assert "NVIDIA A100" in message
        assert "policy.generation.vllm_cfg.precision" in message

    def test_it_refuses_rather_than_downgrading(self):
        """The documented policy: never substitute a dtype silently."""
        with pytest.raises(ValueError, match="refuses rather than substituting"):
            validate_precision("fp8", _cap(8, 0), context="policy.precision")


class TestValidatePrecisionAllows:
    @pytest.mark.parametrize("major,minor", [(9, 0), (10, 0), (10, 3), (12, 0)])
    def test_fp8_on_hopper_and_newer(self, major, minor):
        validate_precision("fp8", _cap(major, minor), context="c")

    @pytest.mark.parametrize("major,minor", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0)])
    def test_bfloat16_on_ampere_and_newer(self, major, minor):
        validate_precision("bfloat16", _cap(major, minor), context="c")

    @pytest.mark.parametrize("precision", ["float32", "float16"])
    @pytest.mark.parametrize("major,minor", [(7, 0), (7, 5), (8, 0), (9, 0)])
    def test_universal_precisions_run_anywhere(self, precision, major, minor):
        validate_precision(precision, _cap(major, minor), context="c")

    def test_an_unknown_precision_is_not_this_module_s_problem(self):
        """Unknown strings are the caller's validation; do not invent a rule."""
        validate_precision("float8_e5m2", _cap(7, 0), context="c")


class TestNoCudaIsNotAnError:
    def test_none_capability_skips_validation(self):
        """The sandbox fleet is CPU-only and the unit suite has no GPU."""
        validate_precision("fp8", None, context="policy.precision")
        validate_precision("bfloat16", None, context="policy.precision")

    def test_resolve_returns_none_without_cuda(self, monkeypatch):
        import dockyard_rl.utils.device_capability as mod

        monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
        assert resolve_device_capability() is None

    def test_resolve_reads_the_device_when_cuda_is_present(self, monkeypatch):
        import dockyard_rl.utils.device_capability as mod

        monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(mod.torch.cuda, "get_device_capability", lambda d=None: (9, 0))
        monkeypatch.setattr(mod.torch.cuda, "get_device_name", lambda d=None: "H100")

        cap = resolve_device_capability()
        assert cap is not None
        assert (cap.major, cap.minor, cap.name) == (9, 0, "H100")
        assert cap.supports_block_fp8 is True


def _validate_precision_contexts(module_path: str) -> list[str]:
    """Every `context=` string passed to validate_precision in a module.

    AST rather than a text grep so the test survives reformatting and line
    moves, and still fails if the call is deleted.
    """
    tree = ast.parse(Path(module_path).read_text())
    contexts: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "validate_precision":
            continue
        for kw in node.keywords:
            if kw.arg == "context" and isinstance(kw.value, ast.Constant):
                contexts.append(str(kw.value.value))
    return contexts


class TestConsumersAreWired:
    """A capability module nothing calls is the exact defect class Track B was.

    These fail if either call site is removed.
    """

    def test_the_trainer_validates_its_precision(self):
        contexts = _validate_precision_contexts(
            "models/policy/workers/dtensor_policy_worker.py"
        )
        assert "policy.precision" in contexts, (
            "the DTensor worker no longer validates policy.precision against the "
            "device; bfloat16 on a pre-Ampere card would run emulated"
        )

    def test_the_generation_worker_validates_its_precision(self):
        contexts = _validate_precision_contexts(
            "models/generation/vllm/vllm_worker.py"
        )
        assert "policy.generation.vllm_cfg.precision" in contexts, (
            "the vLLM worker no longer validates its precision; an fp8 config "
            "below Hopper would fail inside vLLM instead of at startup"
        )

    def test_the_sglang_worker_validates_its_dtype(self):
        """Every transport, not the convenient one.

        sglang_cfg.dtype defaults to ${policy.precision} and is forwarded to the
        SGLang server, so the pre-Ampere bf16 trap is identical on this path.
        Applying a capability gate to vLLM only is the shape of #3152.
        """
        contexts = _validate_precision_contexts(
            "models/generation/sglang/sglang_worker.py"
        )
        assert "policy.generation.sglang_cfg.dtype" in contexts, (
            "the SGLang worker no longer validates its dtype; the capability "
            "gate would cover vLLM only"
        )

    def test_every_generation_transport_is_covered(self):
        """A new backend must not silently skip the gate."""
        covered = {
            "models/generation/vllm/vllm_worker.py",
            "models/generation/sglang/sglang_worker.py",
        }
        for path in covered:
            assert _validate_precision_contexts(path), f"{path} validates nothing"

    def test_the_inline_arch_check_is_gone(self):
        """The one pre-C1 check must route through the capability layer now."""
        src = Path("models/generation/vllm/vllm_worker.py").read_text()
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        assert "get_device_capability()[0]" not in code, (
            "an inline arch check reappeared; capability belongs in one module"
        )
        assert "supports_prefix_caching" in code
