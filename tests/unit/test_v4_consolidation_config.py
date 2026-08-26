"""The V4-style consolidation recipe (Track D2) stays valid against the code.

dockyard has every stage of DeepSeek-V4's two-stage post-training -- per-domain
SFT, per-domain GRPO, multi-teacher on-policy distillation with per-context
routing -- and shipped no config composing them. `on_policy_distillation` had a
full pydantic surface (`algorithms/opd.py`) that **no YAML in the tree set**, so
the whole block was reachable only by hand-writing one.

`examples/configs/v4_consolidation.yaml` is stage 2. These tests assert it
against the code that reads it rather than eyeballing it, because a worked
example that silently stops parsing is worse than none -- someone follows it
and the run fails several minutes in.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

yaml = pytest.importorskip("yaml")

from dockyard_rl.algorithms.opd import (  # noqa: E402
    OnPolicyDistillationConfig,
    assert_prev_logprobs_available,
    is_non_colocated_teachers_enabled,
    is_opd_enabled,
    resolve_reference_aliases,
)

_CONFIG = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples"
    / "configs"
    / "v4_consolidation.yaml"
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(_CONFIG.read_text())


@pytest.fixture(scope="module")
def opd(cfg) -> OnPolicyDistillationConfig:
    return OnPolicyDistillationConfig(**cfg["on_policy_distillation"])


class TestItParsesAgainstTheRealSchema:
    def test_the_opd_block_validates(self, opd):
        assert opd.enabled is True
        assert opd.teacher_model_by_agent_name
        assert opd.non_colocated_teachers is not None
        assert opd.non_colocated_teachers.enabled is True

    def test_opd_is_detected_as_enabled(self, cfg):
        assert is_opd_enabled(cfg)
        assert is_non_colocated_teachers_enabled(cfg)

    def test_the_advantage_estimator_is_opd(self, cfg):
        assert cfg["grpo"]["adv_estimator"]["name"] == "opd"

    def test_the_estimator_name_is_one_the_factory_accepts(self, cfg):
        """Guards against the name drifting out of grpo.py's dispatch."""
        source = (
            pathlib.Path(__file__).resolve().parents[2] / "algorithms" / "grpo.py"
        ).read_text()
        name = cfg["grpo"]["adv_estimator"]["name"]

        assert f'elif name == "{name}"' in source or f'name == "{name}"' in source


class TestTheConstraintsTheRecipeDependsOn:
    """Each is enforced somewhere in code; the config has to satisfy all three."""

    def test_the_async_path_is_on(self, cfg):
        """The OPD collector lives in async_grpo_train; the sync loop has none."""
        assert cfg["grpo"]["async_grpo"]["enabled"] is True

    def test_the_data_plane_is_not_declared(self, cfg):
        """grpo.py raises NotImplementedError for OPD on the data-plane path."""
        assert "data_plane" not in cfg

    def test_prev_logprobs_are_not_zeroed(self, cfg, opd):
        """Â subtracts them, so the guard refuses a config that zeros them."""
        master = SimpleNamespace(
            on_policy_distillation=opd,
            loss_fn=SimpleNamespace(**cfg["loss_fn"]),
            grpo=cfg["grpo"],
        )

        assert_prev_logprobs_available(master)  # raises if the config is wrong


class TestPerContextRouting:
    def test_each_agent_name_routes_to_its_own_teacher(self, opd):
        routed = resolve_reference_aliases(
            [{"name": name} for name in opd.teacher_model_by_agent_name],
            opd.teacher_model_by_agent_name,
            default_teacher_alias=opd.default_teacher_alias,
            strict_agent_name_match=opd.strict_agent_name_match,
        )

        assert routed == list(opd.teacher_model_by_agent_name)

    def test_an_unlisted_agent_falls_back_to_the_default(self, opd):
        routed = resolve_reference_aliases(
            [{"name": "not-a-configured-domain"}],
            opd.teacher_model_by_agent_name,
            default_teacher_alias=opd.default_teacher_alias,
            strict_agent_name_match=opd.strict_agent_name_match,
        )

        assert routed == [opd.default_teacher_alias]

    def test_the_default_alias_is_one_of_the_configured_teachers(self, opd):
        """Otherwise the fallback routes to a teacher that is never built."""
        assert opd.default_teacher_alias in opd.teacher_model_by_agent_name


class TestItStaysHonestAboutWhatItIsNot:
    def test_it_records_that_full_vocab_kl_is_not_this_path(self):
        """V4 takes KL over the whole vocabulary; HV-49 is why this cannot yet."""
        text = _CONFIG.read_text()

        assert "HV-49" in text
        assert "full-vocab" in text.lower()

    def test_it_points_at_the_stage_one_configs_that_exist(self):
        text = _CONFIG.read_text()
        configs = _CONFIG.parent

        for referenced in ("sft.yaml", "grpo_swe.yaml"):
            assert referenced in text
            assert (configs / referenced).exists()
