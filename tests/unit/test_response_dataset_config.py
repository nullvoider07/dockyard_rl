"""Config keys a response dataset accepts must reach the loader.

GSM8K ships two HuggingFace configs, "main" and "socratic". The subset was
hardcoded to "main", so "socratic" was unreachable no matter what a config
asked for -- the "declared and read by nothing" shape, on the data side.
"""

import sys
import types

import pytest


@pytest.fixture
def captured_load_dataset(monkeypatch):
    """Capture the args GSM8KDataset hands to datasets.load_dataset."""
    calls: list[tuple[tuple, dict]] = []

    class _FakeSplit:
        column_names = ["question", "answer"]

        def map(self, *args, **kwargs):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return {"train": _FakeSplit(), "test": _FakeSplit()}

    import dockyard_rl.data.datasets.response_datasets.gsm8k as gsm8k

    monkeypatch.setattr(gsm8k, "load_dataset", fake_load_dataset)
    return gsm8k, calls


def test_subset_defaults_to_main(captured_load_dataset):
    gsm8k, calls = captured_load_dataset
    gsm8k.GSM8KDataset()
    assert calls[0][0] == ("openai/gsm8k", "main")


def test_socratic_subset_is_reachable(captured_load_dataset):
    """Wire cut: rehardcode the subset and this goes red."""
    gsm8k, calls = captured_load_dataset
    gsm8k.GSM8KDataset(subset="socratic")
    assert calls[0][0] == ("openai/gsm8k", "socratic"), (
        "the subset argument did not reach load_dataset; the second GSM8K "
        "config is unreachable"
    )


def test_split_still_selects_the_split(captured_load_dataset):
    gsm8k, _calls = captured_load_dataset
    ds = gsm8k.GSM8KDataset(split="test")
    assert ds.dataset is not None
