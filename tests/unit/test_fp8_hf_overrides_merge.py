"""#2904: fp8 initialization discarded the user's hf_config_overrides.

init_fp8 returns its own hf_overrides carrying quantization_config, and the
worker folded it in with dict.update -- replacing whatever the user configured
through policy.hf_config_overrides (plumbed at grpo.py:641) with that one key.

Silent by construction: the normalization two lines below only substitutes {}
for a non-dict value, so it never noticed a dict had been replaced by another
dict. A run under precision: fp8 simply behaved as though hf_config_overrides
had not been set.

Upstream fixed this once and reintroduced it (#1413 -> #2904), so the merge
semantics are pinned here rather than left to the call site's shape.

merge_fp8_kwargs is exercised directly. The surrounding init path needs a live
vLLM engine, while the claim under test is purely how two dicts combine.
"""

from __future__ import annotations

import logging

import pytest

from dockyard_rl.models.generation.vllm.vllm_worker import merge_fp8_kwargs

# What init_fp8 actually returns (quantization/fp8.py:242-247).
FP8_KWARGS = {
    "quantization": "fp8",
    "kv_cache_dtype": "fp8",
    "hf_overrides": {
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }
    },
}


class TestUserOverridesSurvive:
    def test_user_keys_are_preserved(self):
        vllm_kwargs = {"hf_overrides": {"rope_scaling": {"factor": 4.0}}}
        merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)

        assert vllm_kwargs["hf_overrides"]["rope_scaling"] == {"factor": 4.0}
        assert "quantization_config" in vllm_kwargs["hf_overrides"]

    def test_multiple_user_keys_are_preserved(self):
        vllm_kwargs = {
            "hf_overrides": {
                "rope_scaling": {"factor": 4.0},
                "max_position_embeddings": 32768,
                "num_hidden_layers": 4,
            }
        }
        merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)

        overrides = vllm_kwargs["hf_overrides"]
        assert overrides["rope_scaling"] == {"factor": 4.0}
        assert overrides["max_position_embeddings"] == 32768
        assert overrides["num_hidden_layers"] == 4
        assert overrides["quantization_config"]["quant_method"] == "fp8"

    def test_fp8_quantization_config_is_applied(self):
        vllm_kwargs: dict = {}
        merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)
        assert vllm_kwargs["hf_overrides"]["quantization_config"]["fmt"] == "e4m3"


class TestFp8WinsWhereItMust:
    def test_fp8_overrides_a_conflicting_quantization_config(self):
        """fp8 cannot load without its own block-quant config."""
        vllm_kwargs = {"hf_overrides": {"quantization_config": {"quant_method": "awq"}}}
        merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)
        assert (
            vllm_kwargs["hf_overrides"]["quantization_config"]["quant_method"] == "fp8"
        )

    def test_the_collision_is_logged(self, caplog):
        """Losing a user setting must be visible, even when it is correct to."""
        vllm_kwargs = {"hf_overrides": {"quantization_config": {"quant_method": "awq"}}}
        with caplog.at_level(logging.WARNING):
            merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)
        assert "quantization_config" in caplog.text

    def test_no_warning_without_a_collision(self, caplog):
        vllm_kwargs = {"hf_overrides": {"rope_scaling": {"factor": 4.0}}}
        with caplog.at_level(logging.WARNING):
            merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)
        assert "hf_config_overrides" not in caplog.text

    def test_non_override_fp8_kwargs_still_overwrite(self):
        """quantization and kv_cache_dtype are fp8's to decide, not the user's."""
        vllm_kwargs = {"quantization": None, "kv_cache_dtype": "auto"}
        merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)
        assert vllm_kwargs["quantization"] == "fp8"
        assert vllm_kwargs["kv_cache_dtype"] == "fp8"


class TestDegenerateInputs:
    @pytest.mark.parametrize("bad", [None, "not-a-dict", 42, []])
    def test_a_non_dict_hf_overrides_is_replaced_not_merged(self, bad):
        """Matches the normalization the worker already applied downstream."""
        vllm_kwargs = {"hf_overrides": bad}
        merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)
        assert isinstance(vllm_kwargs["hf_overrides"], dict)
        assert "quantization_config" in vllm_kwargs["hf_overrides"]

    def test_absent_hf_overrides_is_created(self):
        vllm_kwargs: dict = {"max_model_len": 4096}
        merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)
        assert vllm_kwargs["max_model_len"] == 4096
        assert "quantization_config" in vllm_kwargs["hf_overrides"]

    def test_fp8_kwargs_without_overrides_leaves_user_dict_intact(self):
        vllm_kwargs = {"hf_overrides": {"rope_scaling": {"factor": 4.0}}}
        merge_fp8_kwargs(vllm_kwargs, {"quantization": "fp8"})
        assert vllm_kwargs["hf_overrides"] == {"rope_scaling": {"factor": 4.0}}

    def test_the_caller_s_fp8_kwargs_are_not_mutated(self):
        """init_fp8's return value is popped from; it must be copied first."""
        fp8_kwargs = {
            "quantization": "fp8",
            "hf_overrides": {"quantization_config": {"quant_method": "fp8"}},
        }
        merge_fp8_kwargs({}, fp8_kwargs)
        assert "hf_overrides" in fp8_kwargs

    def test_the_user_dict_is_not_mutated_in_place(self):
        """The config object is shared; merging must not write back into it."""
        user_overrides = {"rope_scaling": {"factor": 4.0}}
        vllm_kwargs = {"hf_overrides": user_overrides}
        merge_fp8_kwargs(vllm_kwargs, FP8_KWARGS)
        assert user_overrides == {"rope_scaling": {"factor": 4.0}}
