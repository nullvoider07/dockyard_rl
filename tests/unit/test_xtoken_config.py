"""CPU validation of the xtoken_distillation example config.

Confirms the example YAML is coherent: the loss_fn section constructs
CrossTokenizerDistillationLossFn (with the runtime-injected student vocab size +
per-teacher lists), the transport switch is one of the supported values, and the
distillation section carries the expected driver + collate + transport keys.
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


def _inject_driver_fields(loss_cfg: dict, teachers: list) -> dict:
    """Mirror the driver: inject the student vocab size + parallel per-teacher lists."""
    loss_cfg = dict(loss_cfg)
    loss_cfg["student_vocab_size"] = 151936
    loss_cfg["projection_matrix_paths"] = [t["projection_matrix_path"] for t in teachers]
    loss_cfg["teacher_vocab_sizes"] = [128256 for _ in teachers]
    loss_cfg["teacher_weights"] = [t["weight"] for t in teachers]
    loss_cfg["teacher_gold_loss"] = [t.get("gold_loss") for t in teachers]
    loss_cfg["teacher_xtoken_loss"] = [t.get("xtoken_loss") for t in teachers]
    return loss_cfg


def test_config_file_exists_and_loads():
    cfg = _load()
    assert {"distillation", "loss_fn", "policy", "teachers"} <= set(cfg)
    assert isinstance(cfg["teachers"], list) and len(cfg["teachers"]) >= 1


def test_loss_fn_section_constructs_the_loss():
    cfg = _load()
    loss_cfg = _inject_driver_fields(cfg["loss_fn"], cfg["teachers"])
    fn = CrossTokenizerDistillationLossFn(loss_cfg)
    assert fn.gold_loss is False and fn.xtoken_loss is False
    assert fn.vocab_topk == cfg["loss_fn"]["vocab_topk"]
    assert fn.num_teachers == len(cfg["teachers"])
    assert fn.kd_loss_mode == "sum"


def test_loss_fn_has_all_config_keys():
    loss_cfg = _load()["loss_fn"]
    expected = {
        "gold_loss", "xtoken_loss", "temperature",
        "vocab_topk", "uncommon_topk", "reverse_kl", "exact_token_match_only",
        "kl_loss_weight", "ce_loss_scale", "dynamic_loss_scaling",
        "kd_loss_mode", "normalize_teacher_by_vocab", "alpha", "sum_weights_metric",
    }
    assert expected == set(loss_cfg)


def test_each_teacher_has_projection_weight_and_overrides():
    for t in _load()["teachers"]:
        assert {"projection_matrix_path", "weight", "gold_loss", "xtoken_loss"} <= set(t)


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
        # collate (teacher caps/divisors are per-teacher lists)
        "ctx_length_student", "ctx_length_teachers",
        "make_seq_div_by_student", "make_seq_div_by_teachers",
    }
    assert expected <= set(dist)


def test_xtoken_loss_requires_gold_loss_rejected_at_construction():
    # The (gold=False, xtoken=True) combination is rejected — guards a config typo.
    cfg = _load()
    loss_cfg = _inject_driver_fields(cfg["loss_fn"], cfg["teachers"])
    loss_cfg.update(gold_loss=False, xtoken_loss=True)
    try:
        CrossTokenizerDistillationLossFn(loss_cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
