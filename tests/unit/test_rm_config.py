"""CPU tests for the RM config/state type migration (TypedDict -> BaseModel/dataclass).

`RMConfig` is a pydantic `BaseModel(extra="allow")` with defaults; `RMSaveState` and
`RMValMetrics` are dataclasses. These cover the default-fill + extra-key tolerance of
the config, the dataclass round-trip, and the checkpoint (de)serialization glue the
migrated `setup()`/`rm_train()` rely on: filter-to-known-fields on load (+ backcompat
`total_valid_tokens`), `vars()` serialization, and the dynamic val-metric attrs that
ride on the save-state instance via `setattr`/`delattr`.
"""

from dataclasses import asdict, fields, is_dataclass

from dockyard_rl.algorithms.rm import (
    RMConfig,
    RMSaveState,
    RMValMetrics,
    _initial_rm_save_state,
)


# -- RMConfig (BaseModel, extra="allow") --------------------------------------

def test_rmconfig_fills_defaults_when_empty():
    cfg = RMConfig()
    assert cfg.max_num_steps == -1
    assert cfg.max_num_epochs == 1
    assert cfg.val_period == 16
    assert cfg.val_batches == -1
    assert cfg.val_global_batch_size == 32
    assert cfg.val_micro_batch_size == 1
    assert cfg.val_at_start is False
    assert cfg.val_at_end is False
    assert cfg.seed == 42


def test_rmconfig_partial_override_keeps_other_defaults():
    cfg = RMConfig(max_num_steps=5, seed=7)
    assert cfg.max_num_steps == 5
    assert cfg.seed == 7
    assert cfg.val_period == 16  # untouched default


def test_rmconfig_allows_extra_keys():
    cfg = RMConfig.model_validate({"some_future_knob": 123})  # extra="allow"
    assert getattr(cfg, "some_future_knob") == 123
    assert cfg.seed == 42


# -- RMSaveState (dataclass) --------------------------------------------------

def test_rmsavestate_is_dataclass_with_expected_fields():
    assert is_dataclass(RMSaveState)
    names = {f.name for f in fields(RMSaveState)}
    assert names == {
        "epoch", "step", "total_steps", "consumed_samples", "total_valid_tokens",
    }


def test_initial_rm_save_state_is_zeroed():
    st = _initial_rm_save_state()
    assert vars(st) == {
        "epoch": 0, "step": 0, "total_steps": 0,
        "consumed_samples": 0, "total_valid_tokens": 0,
    }


def test_checkpoint_load_filters_unknown_keys_and_backfills():
    # Mirrors setup(): a checkpoint dict may carry extra (val-metric) keys and may
    # predate total_valid_tokens. Filter to known fields + setdefault the new one.
    loaded_state = {
        "epoch": 2, "step": 3, "total_steps": 40, "consumed_samples": 80,
        "validation-default_loss": 0.5,  # extra key from a previous run
        # total_valid_tokens intentionally absent (old checkpoint)
    }
    loaded_state.setdefault("total_valid_tokens", 0)
    known = {f.name for f in fields(RMSaveState)}
    st = RMSaveState(**{k: v for k, v in loaded_state.items() if k in known})
    assert st.epoch == 2 and st.total_steps == 40
    assert st.total_valid_tokens == 0  # backfilled
    assert not hasattr(st, "validation-default_loss")  # extra key dropped


def test_dynamic_val_metric_attrs_roundtrip_through_vars():
    # rm_train attaches prefixed val-metric keys to the save-state via setattr and
    # serializes with vars(); outdated ones are removed with delattr.
    st = _initial_rm_save_state()
    setattr(st, "validation-default_loss", 0.25)
    setattr(st, "val:validation-default_loss", 0.25)
    serialized = vars(st)
    assert serialized["validation-default_loss"] == 0.25
    assert serialized["epoch"] == 0  # fields still present
    delattr(st, "validation-default_loss")
    assert "validation-default_loss" not in vars(st)


# -- RMValMetrics (dataclass) -------------------------------------------------

def test_rmvalmetrics_dataclass_asdict_and_getattr():
    m = RMValMetrics(
        loss=1.0, accuracy=0.5, rewards_chosen_mean=2.0,
        rewards_rejected_mean=1.0, num_valid_samples=8.0,
    )
    assert is_dataclass(RMValMetrics)
    d = asdict(m)
    assert d == {
        "loss": 1.0, "accuracy": 0.5, "rewards_chosen_mean": 2.0,
        "rewards_rejected_mean": 1.0, "num_valid_samples": 8.0,
    }
    # attribute access (replaces the old TypedDict subscripting in validate()).
    for name in [f.name for f in fields(RMValMetrics)]:
        assert getattr(m, name) == d[name]
