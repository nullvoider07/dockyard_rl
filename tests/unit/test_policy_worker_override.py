"""`resolve_policy_worker_cls` refuses an override that was never written.

`POLICY_WORKER_OVERRIDES` redirects the DTensor policy workers to their
ModelOpt-quantized counterparts whenever `policy.quant_cfg` is set. Both
targets live under `dockyard_rl.modelopt.models.policy.workers`, and that
subpackage does not exist: `dockyard_rl/modelopt/` carries only the inference
half (`models/generation/`). Returning the FQN anyway pushed the failure into
Ray actor construction, where a `ModuleNotFoundError` names a module path and
not the config key that selected it.

The precise shape is the one the plan's wiring table calls "a module is
imported that was never written" (#3226). It differs from #3226 in severity --
this one fails loudly rather than silently -- so the fix is to fail *earlier*
and name the cause, not to build the missing worker.
"""

from __future__ import annotations

import pytest

from dockyard_rl.models.policy.utils import (
    POLICY_WORKER_OVERRIDES,
    resolve_policy_worker_cls,
)

_V1 = "dockyard_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"
_V2 = "dockyard_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"


class TestUnquantizedPathIsUntouched:
    """quant_cfg absent or null must leave the core path byte-identical."""

    @pytest.mark.parametrize("config", [{}, {"quant_cfg": None}])
    @pytest.mark.parametrize("default_cls", [_V1, _V2])
    def test_default_is_returned(self, config, default_cls):
        assert resolve_policy_worker_cls(default_cls, config) == default_cls

    def test_a_worker_with_no_override_entry_passes_through(self):
        cls = "dockyard_rl.models.jax.policy_worker.JaxPolicyWorker"
        assert cls not in POLICY_WORKER_OVERRIDES

        assert resolve_policy_worker_cls(cls, {"quant_cfg": "NVFP4_DEFAULT_CFG"}) == cls


class TestMissingOverrideIsRefused:
    @pytest.mark.parametrize("default_cls", [_V1, _V2])
    def test_quant_cfg_refuses_with_the_reason(self, default_cls):
        with pytest.raises(ValueError) as excinfo:
            resolve_policy_worker_cls(default_cls, {"quant_cfg": "NVFP4_DEFAULT_CFG"})

        message = str(excinfo.value)
        # The refusal has to name the config key, the missing target, and the
        # way out -- a bare "not found" is what this replaces.
        assert "policy.quant_cfg" in message
        assert POLICY_WORKER_OVERRIDES[default_cls] in message
        assert "policy.generation.quant_cfg" in message

    def test_both_override_targets_are_absent_today(self):
        """Pins the premise. If either lands, this fails and the refusal goes."""
        import importlib.util

        for target in POLICY_WORKER_OVERRIDES.values():
            module_path = target.rsplit(".", 1)[0]
            with pytest.raises(ModuleNotFoundError):
                importlib.util.find_spec(module_path)


class TestImportabilityProbe:
    def test_a_present_module_resolves(self):
        from dockyard_rl.models.policy.utils import _worker_cls_is_importable

        assert _worker_cls_is_importable(_V1)

    def test_a_missing_module_does_not(self):
        from dockyard_rl.models.policy.utils import _worker_cls_is_importable

        assert not _worker_cls_is_importable(
            "dockyard_rl.models.policy.workers.does_not_exist.Worker"
        )
