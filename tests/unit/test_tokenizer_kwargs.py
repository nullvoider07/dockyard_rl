"""tokenizer_kwargs forwarding in algorithms/utils.py::get_tokenizer (#3798).

These are wiring tests, not behaviour tests: each asserts the configured kwarg
*arrives at* ``from_pretrained``. A test that only checked ``get_tokenizer``
returns a tokenizer would still pass with the ``**tokenizer_kwargs`` splat
deleted, which is the regression this file exists to catch.

The two branches are covered separately because they treat ``use_fast``
differently — the processor branch pins it and must pop a caller-supplied
value, the tokenizer branch lets it through.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dockyard_rl.algorithms.utils import get_tokenizer


def _fake_tokenizer():
    """Stand-in carrying every attribute get_tokenizer reads after loading.

    The processor branch forwards the full special-token set onto the processor
    (algorithms/utils.py:325-334), so a thinner fake fails there for reasons
    unrelated to kwarg forwarding.
    """
    return SimpleNamespace(
        pad_token="<pad>",
        eos_token="</s>",
        bos_token="<s>",
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        name_or_path="dummy/model",
        chat_template=None,
    )


@pytest.fixture
def captured():
    """Patch both loaders and record the kwargs each was called with."""
    calls: dict = {}

    def _tok(name, **kwargs):
        calls["tokenizer"] = (name, kwargs)
        return _fake_tokenizer()

    def _proc(name, **kwargs):
        calls["processor"] = (name, kwargs)
        return SimpleNamespace(tokenizer=_fake_tokenizer())

    with (
        patch("dockyard_rl.algorithms.utils.AutoTokenizer.from_pretrained", _tok),
        patch("dockyard_rl.algorithms.utils.AutoProcessor.from_pretrained", _proc),
    ):
        yield calls


def test_tokenizer_branch_forwards_kwargs(captured):
    get_tokenizer(
        {
            "name": "dummy/model",
            "tokenizer_kwargs": {"fix_mistral_regex": False, "use_fast": False},
        }
    )
    _, kwargs = captured["tokenizer"]
    # Both reach from_pretrained: this branch pins no use_fast of its own.
    assert kwargs["fix_mistral_regex"] is False
    assert kwargs["use_fast"] is False
    assert kwargs["trust_remote_code"] is True


def test_processor_branch_forwards_kwargs_and_pops_use_fast(captured):
    get_tokenizer(
        {
            "name": "dummy/model",
            "tokenizer_kwargs": {"fix_mistral_regex": False, "use_fast": False},
        },
        get_processor=True,
    )
    _, kwargs = captured["processor"]
    assert kwargs["fix_mistral_regex"] is False
    # Popped, not splatted — passing it twice would raise TypeError.
    assert kwargs["use_fast"] is False


def test_use_fast_defaults_true_on_processor_branch(captured):
    get_tokenizer({"name": "dummy/model"}, get_processor=True)
    _, kwargs = captured["processor"]
    assert kwargs["use_fast"] is True


def test_absent_and_none_tokenizer_kwargs_are_both_inert(captured):
    """Neither an omitted key nor an explicit null adds anything to the call."""
    get_tokenizer({"name": "dummy/model"})
    _, absent = captured["tokenizer"]

    get_tokenizer({"name": "dummy/model", "tokenizer_kwargs": None})
    _, explicit_none = captured["tokenizer"]

    assert absent == explicit_none == {"trust_remote_code": True}


@pytest.mark.parametrize("get_processor", [False, True])
def test_trust_remote_code_is_overridable_not_a_collision(captured, get_processor):
    """Passing it must not raise "got multiple values for keyword argument".

    Both loaders are called with trust_remote_code pinned, so an unpopped
    caller value would collide. Upstream #3798 leaves this unhandled.
    """
    get_tokenizer(
        {"name": "dummy/model", "tokenizer_kwargs": {"trust_remote_code": False}},
        get_processor=get_processor,
    )
    _, kwargs = captured["processor" if get_processor else "tokenizer"]
    assert kwargs["trust_remote_code"] is False


def test_trust_remote_code_defaults_true(captured):
    get_tokenizer({"name": "dummy/model"})
    _, kwargs = captured["tokenizer"]
    assert kwargs["trust_remote_code"] is True


def test_caller_config_is_not_mutated(captured):
    """get_tokenizer pops use_fast off a copy, never the caller's dict."""
    cfg = {"name": "dummy/model", "tokenizer_kwargs": {"use_fast": False}}
    get_tokenizer(cfg, get_processor=True)
    assert cfg["tokenizer_kwargs"] == {"use_fast": False}
    # And the popped value still reached the loader, so the copy is a copy and
    # not a silent drop.
    _, kwargs = captured["processor"]
    assert kwargs["use_fast"] is False
