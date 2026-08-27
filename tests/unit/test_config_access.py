"""The config-shape reader, and that every Track F module routes through it.

Two config shapes reach these modules: TypedDict blocks (plain dicts at
runtime) and the pydantic BaseModels declared in opd.py. A dict-only reader
does not fail on a pydantic config -- it returns the caller's default for
every key, so the feature reads as configured and does nothing. teacher_panel
shipped exactly that defect, which is why the reader is shared now.
"""

import pytest
import torch
from pydantic import BaseModel

from dockyard_rl.algorithms import effort_reward, rollout_allocation, routing_diversity
from dockyard_rl.algorithms import teacher_panel
from dockyard_rl.algorithms.config_access import read_config_value


class _Model(BaseModel, extra="allow"):
    enabled: bool = True
    coefficient: float = 0.25


class _Plain:
    enabled = True
    coefficient = 0.25


# -- the reader ---------------------------------------------------------------

@pytest.mark.parametrize(
    "section",
    [{"enabled": True, "coefficient": 0.25}, _Model(), _Plain()],
    ids=["dict", "pydantic", "object"],
)
def test_reads_every_shape(section):
    assert read_config_value(section, "enabled", False) is True
    assert read_config_value(section, "coefficient", 0.0) == 0.25


def test_missing_key_and_none_section_take_the_default():
    assert read_config_value({"a": 1}, "b", "fallback") == "fallback"
    assert read_config_value(_Model(), "nope", "fallback") == "fallback"
    assert read_config_value(None, "anything", "fallback") == "fallback"


def test_explicit_none_is_returned_not_coalesced():
    """Coalescing is the caller's policy; opd.py relies on the raw value."""
    assert read_config_value({"k": None}, "k", "default") is None


# -- every module's private reader must handle both shapes --------------------

_READERS = [
    (effort_reward, "effort_reward"),
    (rollout_allocation, "rollout_allocation"),
    (routing_diversity, "routing_diversity"),
    (teacher_panel, "teacher_panel"),
]


@pytest.mark.parametrize("module,name", _READERS, ids=[n for _, n in _READERS])
def test_module_cfg_reader_handles_a_pydantic_config(module, name):
    """Wire cut: revert any _cfg to a dict-only read and this goes red.

    A dict-only reader returns the default here instead of the configured
    value -- silently, which is how the feature disables itself while the
    config still reads as live.
    """
    assert module._cfg(_Model(), "coefficient", 0.0) == 0.25, (
        f"{name}._cfg dropped a value off a pydantic config; it would return "
        "the default for every key and silently disable the feature"
    )
    assert module._cfg({"coefficient": 0.25}, "coefficient", 0.0) == 0.25
    assert module._cfg(_Model(), "absent", "fallback") == "fallback"
    assert module._cfg({"coefficient": None}, "coefficient", 0.5) == 0.5


@pytest.mark.parametrize("module,name", _READERS, ids=[n for _, n in _READERS])
def test_module_does_not_carry_its_own_dict_only_read(module, name):
    """No module may reintroduce a `hasattr(cfg, "get")` fork of the reader."""
    import inspect

    src = inspect.getsource(module)
    assert 'hasattr(cfg, "get")' not in src, (
        f"{name} reintroduced a local dict-only config read; use "
        "config_access.read_config_value so all four stay in step"
    )


# -- the gates themselves, which are the reads that matter most ---------------

def test_enabled_gates_read_a_pydantic_config():
    """An enable flag misread as False disables the feature silently."""
    assert rollout_allocation.graded_allocation_enabled(_Model()) is True
    assert routing_diversity.routing_diversity_enabled(_Model()) is True
    assert rollout_allocation.graded_allocation_enabled(
        _Model(enabled=False)
    ) is False
