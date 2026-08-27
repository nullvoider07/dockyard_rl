"""Refit dtype: what survives the cast to the policy dtype, and what is refused.

Every refit tensor was cast to `self.dtype` on its way out -- the fp32 MoE
router correction bias included. That bias is added to the router scores before
top-k and steps by the load-balance coefficient (order 1e-3) per update, so a
bf16 cast quantizes it at roughly one update step and the rollout engine can
rank experts differently from the trainer that produced it. vLLM keeps it in
fp32 on every noaux_tc model (`deepseek_v2.py:317` creates it as
`torch.empty(..., dtype=torch.float32)`; only the ROCm AITER path narrows it).

The decision is made in one function because it is read in two different
kinds of place. The transports stream with it, and `prepare_refit_info` declares
with it -- and the IPC receiver slices the packed buffer using that declaration,
so a name sent at one width and declared at another is read at the wrong offset,
not merely at the wrong precision. Hence a test per transport that fails when
its call is cut.

The adjacent question, answered here rather than assumed: dockyard's own
`MoEBlock.expert_bias_E` is the same kind of fp32 router state, but it is
dockyard-native and has no counterpart to refit into -- vLLM's Qwen3-MoE router,
the family `apply_moe_surgery` supports, carries no correction bias, and
`AutoWeightsLoader` raises on a name it cannot place. Refusing at the refit
seam beats aborting inside the engine's loader, and beats dropping the key,
which would leave the rollout engine routing without the bias the trainer
routes with.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest
import torch
from torch import nn

from dockyard_rl.models.dtensor.moe.refit import (
    check_refit_names,
    refit_tensor_dtype,
)

BF16 = torch.bfloat16
FP32 = torch.float32

BIAS = "model.layers.0.mlp.gate.e_score_correction_bias"


class TestTheDtypeDecision:
    def test_the_router_correction_bias_keeps_fp32(self):
        assert refit_tensor_dtype(BIAS, torch.zeros(8, dtype=FP32), BF16) is FP32

    def test_an_ordinary_fp32_parameter_still_casts(self):
        """The exception is one name, not "fp32 is preserved"."""
        weight = torch.zeros(4, 4, dtype=FP32)
        assert refit_tensor_dtype("model.norm.weight", weight, BF16) is BF16

    def test_a_bias_that_is_not_fp32_is_not_promoted_to_fp32(self):
        """Preserve a source dtype; never invent one the trainer did not hold."""
        assert refit_tensor_dtype(BIAS, torch.zeros(8, dtype=BF16), BF16) is BF16

    def test_the_match_is_on_the_final_name_component(self):
        """A substring match would catch unrelated params on a renamed model."""
        near_miss = "model.layers.0.gate.e_score_correction_bias_scale"
        assert refit_tensor_dtype(near_miss, torch.zeros(8, dtype=FP32), BF16) is BF16

    def test_an_undotted_name_is_handled(self):
        assert refit_tensor_dtype(
            "e_score_correction_bias", torch.zeros(8, dtype=FP32), BF16
        ) is FP32

    @pytest.mark.parametrize("default", [torch.float16, torch.float32, BF16])
    def test_the_default_is_the_callers_to_choose(self, default):
        weight = torch.zeros(2, 2, dtype=FP32)
        assert refit_tensor_dtype("model.norm.weight", weight, default) is default


class TestUntransferableStateIsRefused:
    def test_the_load_balance_bias_is_refused(self):
        with pytest.raises(ValueError, match="expert_bias_E"):
            check_refit_names(["model.layers.0.mlp.expert_bias_E"])

    def test_the_refusal_names_the_config_key_that_produced_it(self):
        """An operator has to know which knob to turn, not just which tensor."""
        with pytest.raises(ValueError) as exc:
            check_refit_names(["model.layers.0.mlp.expert_bias_E"])
        assert "load_balance_coeff" in str(exc.value)

    def test_ordinary_names_pass(self):
        check_refit_names([BIAS, "model.norm.weight", "lm_head.weight"])

    def test_an_empty_stream_passes(self):
        check_refit_names([])


class _FakeModel(nn.Module):
    """A state dict with one of each: fp32 router bias, bf16 weight."""

    def __init__(self, *, load_balanced: bool = False) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(4, 4, dtype=BF16))
        self.register_buffer("e_score_correction_bias", torch.zeros(8, dtype=FP32))
        if load_balanced:
            self.register_buffer("expert_bias_E", torch.zeros(8, dtype=FP32))


def _worker(model: nn.Module):
    from dockyard_rl.models.policy.workers.dtensor_policy_worker import (
        DTensorPolicyWorkerImpl,
    )

    worker = DTensorPolicyWorkerImpl.__new__(DTensorPolicyWorkerImpl)
    worker.model = model  # type: ignore[assignment]
    worker.dtype = BF16  # type: ignore[assignment]
    return worker


class TestTheManifestAgreesWithTheStream:
    """`prepare_refit_info` is what the IPC receiver slices its buffer with."""

    def test_the_bias_is_declared_fp32_and_everything_else_bf16(self):
        info = _worker(_FakeModel()).prepare_refit_info()

        assert info is not None
        assert info["e_score_correction_bias"][1] is FP32
        assert info["weight"][1] is BF16

    def test_shapes_are_unchanged_by_the_dtype_work(self):
        info = _worker(_FakeModel()).prepare_refit_info()

        assert info is not None
        assert tuple(info["weight"][0]) == (4, 4)
        assert tuple(info["e_score_correction_bias"][0]) == (8,)

    def test_both_refit_seams_read_the_same_gated_state_dict(self):
        """Streamed names and declared names come from one source."""
        worker = _worker(_FakeModel())
        info = worker.prepare_refit_info()

        assert info is not None
        assert set(dict(worker._iter_refit_state_dict())) == set(info)

    def test_the_streaming_seam_refuses_untransferable_state_too(self):
        """Not only the metadata seam -- the collective transport never calls it."""
        worker = _worker(_FakeModel(load_balanced=True))
        with pytest.raises(ValueError, match="expert_bias_E"):
            dict(worker._iter_refit_state_dict())

    def test_the_metadata_seam_refuses_it(self):
        worker = _worker(_FakeModel(load_balanced=True))
        with pytest.raises(ValueError, match="expert_bias_E"):
            worker.prepare_refit_info()


def _method_source(module, name: str, next_name: str) -> str:
    """Source of one method, read from the file.

    Read from disk rather than via `inspect.getsource`: the nvtx decorator is
    not functools.wraps'd, so getsource returns the wrapper.
    """
    text = pathlib.Path(module.__file__).read_text()
    start = text.index(f"    def {name}(")
    end = text.index(f"    def {next_name}(", start)
    return text[start:end]


class TestEveryTransportReadsTheDecision:
    """Three transports and two metadata maps; a fix on one is not a fix.

    `#3152` is what happens when a refit behaviour lands on the transport that
    was easiest to reach. These fail if any single call is dropped.
    """

    @pytest.mark.parametrize(
        "method,next_method",
        [
            ("stream_weights_via_ipc_zmq", "get_ipc_zmq_address"),
            ("broadcast_weights_for_collective", "stream_weights_via_http"),
            ("stream_weights_via_http", "prepare_for_lp_inference"),
        ],
    )
    def test_each_transport_casts_through_the_helper(self, method, next_method):
        from dockyard_rl.models.policy.workers import dtensor_policy_worker

        source = _method_source(dtensor_policy_worker, method, next_method)

        assert "refit_tensor_dtype(" in source, (
            f"{method} no longer routes its cast through refit_tensor_dtype; "
            "the fp32 router bias would be streamed at the policy dtype while "
            "prepare_refit_info still declares it fp32"
        )

    def test_the_collective_transport_is_given_the_name_not_only_the_tensor(self):
        """It passed `x[1]` alone; the decision needs the name."""
        from dockyard_rl.models.policy.workers import dtensor_policy_worker

        source = _method_source(
            dtensor_policy_worker,
            "broadcast_weights_for_collective",
            "stream_weights_via_http",
        )
        assert "x[0], x[1]" in source

    def test_the_v2_metadata_map_reads_it_too(self):
        """v2 overrides prepare_refit_info outright, so it can drift alone."""
        from dockyard_rl.models.policy.workers import dtensor_policy_worker_v2

        source = _method_source(
            dtensor_policy_worker_v2, "prepare_refit_info", "_setup_moe_load_balancing"
        )
        assert "refit_tensor_dtype(" in source


class TestTeacherConfigsDropStudentOnlyFeatures:
    def _student_cfg(self) -> dict:
        return {
            "peft": {"enabled": True},
            "draft": {"enabled": True},
            "router_replay": {"enabled": True},
            "quant_cfg": {"algorithm": "fp8"},
        }

    def test_router_replay_is_disabled(self):
        from dockyard_rl.models.policy.teacher_worker_group import (
            disable_student_only_features,
        )

        cfg = self._student_cfg()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            disable_student_only_features(cfg, alias="code")

        assert cfg["router_replay"]["enabled"] is False

    def test_the_other_three_stay_disabled(self):
        """Regression guard: adding router_replay must not drop the rest."""
        from dockyard_rl.models.policy.teacher_worker_group import (
            disable_student_only_features,
        )

        cfg = self._student_cfg()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            disable_student_only_features(cfg, alias="code")

        assert cfg["peft"]["enabled"] is False
        assert cfg["draft"]["enabled"] is False
        assert cfg["quant_cfg"] is None

    def test_absent_keys_are_not_introduced(self):
        from dockyard_rl.models.policy.teacher_worker_group import (
            disable_student_only_features,
        )

        cfg: dict = {}
        disable_student_only_features(cfg, alias="code")

        assert cfg == {}

    def test_a_non_dict_router_replay_is_left_alone(self):
        """The config is schema-less; a bool here must not raise."""
        from dockyard_rl.models.policy.teacher_worker_group import (
            disable_student_only_features,
        )

        cfg: dict = {"router_replay": True}
        disable_student_only_features(cfg, alias="code")

        assert cfg["router_replay"] is True

    def test_the_teacher_group_still_calls_it(self):
        """Cut the call and every student feature leaks onto the teacher."""
        import inspect

        from dockyard_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

        source = inspect.getsource(TeacherWorkerGroup.__init__)
        assert "disable_student_only_features(cfg" in source

    def test_a_dense_teacher_under_a_replaying_student_would_have_refused(self):
        """Pins why this matters: the refusal the teacher used to hit."""
        from dockyard_rl.models.dtensor.moe.router_replay import (
            resolve_router_replay_enabled,
            validate_router_replay_config,
        )
        from dockyard_rl.models.policy.teacher_worker_group import (
            disable_student_only_features,
        )

        dense = nn.Linear(4, 4)
        student_cfg = {"router_replay": {"enabled": True}}
        with pytest.raises(ValueError, match="MoEBlock"):
            validate_router_replay_config(
                resolve_router_replay_enabled(student_cfg), dense
            )

        teacher_cfg = {"router_replay": {"enabled": True}}
        disable_student_only_features(teacher_cfg, alias="code")
        validate_router_replay_config(
            resolve_router_replay_enabled(teacher_cfg), dense
        )
