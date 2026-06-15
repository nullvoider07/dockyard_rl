"""Guards the Megatron-lineage excision (D8, convention #1).

Megatron is not a dockyard_rl dependency. These tests assert the modules that
used to hard-import ``megatron.core`` now import without it, and that no
``megatron`` references remain in the touched source.
"""

import importlib
from pathlib import Path

import pytest

# Repo root (the package dir is itself named dockyard_rl; modules live at its top level).
_REPO = Path(__file__).resolve().parents[2]


def test_model_utils_imports_without_megatron():
    # megatron is not installed; the import must still succeed.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("megatron.core")
    mu = importlib.import_module("dockyard_rl.distributed.model_utils")
    # DTensor-needed symbols survive the excision.
    assert hasattr(mu, "DistributedCrossEntropy")
    assert hasattr(mu, "allgather_cp_sharded_tensor")


@pytest.mark.parametrize(
    "relpath",
    [
        "distributed/model_utils.py",
        "algorithms/sft.py",
        "algorithms/dpo.py",
        "algorithms/rm.py",
        "algorithms/grpo.py",
        "algorithms/loss/wrapper.py",
        "models/policy/lm_policy.py",
        "models/policy/utils.py",
        "utils/checkpoint.py",
    ],
)
def test_no_megatron_references_in_source(relpath):
    text = (_REPO / relpath).read_text()
    lowered = text.lower()
    assert "megatron" not in lowered, f"{relpath} still references megatron"
