"""CPU validation of the xtoken_distillation example config (M4.b).

Confirms the example YAML is coherent: the loss_fn section constructs
CrossTokenizerDistillationLossFn (with runtime-injected vocab sizes), the
transport switch is one of the supported values, and the distillation section
carries the expected driver + collate + transport keys.
"""

from pathlib import Path

import yaml

from dockyard_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn

_CONFIG = (
    Path(__file__).parents[2] / "examples" / "configs" / "xtoken_distillation.yaml"
)


def _load():
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


def test_config_file_exists_and_loads():
    cfg = _load()
    assert {"distillation", "loss_fn", "policy", "teacher_policy"} <= set(cfg)


def test_loss_fn_section_constructs_the_loss():
    cfg = _load()
    loss_cfg = dict(cfg["loss_fn"])
    # student/teacher_vocab_size are runtime-injected by the driver from the
    # tokenizers (not user YAML knobs).
    loss_cfg["student_vocab_size"] = 151936
    loss_cfg["teacher_vocab_size"] = 128256
    fn = CrossTokenizerDistillationLossFn(loss_cfg)
    assert fn.gold_loss is False and fn.xtoken_loss is False
    assert fn.vocab_topk == cfg["loss_fn"]["vocab_topk"]


def test_loss_fn_has_all_config_keys():
    loss_cfg = _load()["loss_fn"]
    expected = {
        "projection_matrix_path", "gold_loss", "xtoken_loss", "temperature",
        "vocab_topk", "uncommon_topk", "reverse_kl", "exact_token_match_only",
        "kl_loss_weight", "ce_loss_scale", "dynamic_loss_scaling",
    }
    assert expected == set(loss_cfg)


def test_transport_switch_is_supported_value():
    transport = _load()["distillation"]["transport"]
    assert transport in {"ipc", "cross_cluster"}


def test_distillation_section_has_driver_and_collate_keys():
    dist = _load()["distillation"]
    expected = {
        # driver
        "num_prompts_per_step", "max_num_steps", "max_num_epochs", "seed",
        "val_period", "val_at_start", "val_at_end", "val_teacher_micro_batch_size",
        # transport
        "transport", "num_seq_chunks",
        # collate
        "ctx_length_student", "ctx_length_teacher",
        "make_seq_div_by_student", "make_seq_div_by_teacher",
    }
    assert expected <= set(dist)


def test_xtoken_loss_requires_gold_loss_rejected_at_construction():
    # The (gold=False, xtoken=True) combination is rejected — guards a config typo.
    cfg = _load()
    loss_cfg = dict(cfg["loss_fn"])
    loss_cfg.update(gold_loss=False, xtoken_loss=True,
                    student_vocab_size=100, teacher_vocab_size=100)
    try:
        CrossTokenizerDistillationLossFn(loss_cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
