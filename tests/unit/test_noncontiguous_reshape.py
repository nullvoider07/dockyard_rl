"""`.view()` on a tensor the function did not create.

`.view` requires a compatible stride layout and raises when it does not get
one; `.reshape` falls back to a copy. The difference only matters for a tensor
whose layout the calling function does not own -- a parameter, or an autograd
`grad_output` -- and it is invisible until something upstream hands over a
slice.

Found via #3709, which changed one such `.view` upstream. Sweeping this tree
for the same shape turned up eight more across four files, so it is a class
rather than an instance. The shared rule: a function that *creates* a tensor
may `view` it, and a function that *receives* one may not.

Two properties make it easy to ship: a sequence-dim slice is contiguous when
the batch size is 1, so a single-sample smoke test passes and the failure
appears at real batch size; and the error is a stride complaint from torch
rather than anything naming the config that reached it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from dockyard_rl.algorithms.logits_sampling_utils import apply_top_k_top_p

ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestTheHazardIsReal:
    def test_a_sequence_dim_slice_is_not_contiguous(self):
        logits = torch.randn(3, 5, 7)
        assert not logits[:, :-1].is_contiguous()

    def test_but_it_is_contiguous_at_batch_size_one(self):
        """Why a smoke test would not have caught this."""
        assert torch.randn(1, 5, 7)[:, :-1].is_contiguous()

    def test_view_raises_on_it_and_reshape_does_not(self):
        sliced = torch.randn(3, 5, 7)[:, :-1]
        with pytest.raises(RuntimeError, match="not compatible"):
            sliced.view(-1, 7)
        assert sliced.reshape(-1, 7).shape == (12, 7)

    def test_a_same_dtype_to_is_a_passthrough_not_a_copy(self):
        """The second feeder: `.to(float32)` on fp32 logits keeps the layout."""
        sliced = torch.randn(3, 5, 7)[:, 1:4]
        assert sliced.to(torch.float32) is sliced
        assert not sliced.to(torch.float32).is_contiguous()
        # A dtype change does copy, which is why a bf16 policy hid this.
        assert torch.randn(3, 5, 7, dtype=torch.bfloat16)[:, 1:4].to(
            torch.float32
        ).is_contiguous()


class TestTopKTopPAcceptsNonContiguousLogits:
    """The proven site: `apply_top_k_top_p` flattens whatever it is handed."""

    @pytest.mark.parametrize("batch", [1, 2, 5])
    def test_a_trailing_slice_is_filtered(self, batch):
        """`next_token_logits[:, :-1]`, as model_utils hands it over."""
        logits = torch.randn(batch, 6, 11)[:, :-1]

        filtered, _ = apply_top_k_top_p(logits, top_k=None, top_p=0.9)

        assert filtered.shape == logits.shape

    def test_a_chunked_sequence_slice_is_filtered(self):
        """`logits[:, start:end, :]`, as the worker's chunked logprob loop does."""
        logits = torch.randn(4, 8, 11)[:, 2:5, :]

        filtered, _ = apply_top_k_top_p(logits, top_k=None, top_p=0.9)

        assert filtered.shape == (4, 3, 11)

    def test_the_values_match_the_contiguous_computation(self):
        """reshape may copy; the result must not depend on that."""
        base = torch.randn(3, 6, 11)
        sliced = base[:, :-1]

        got, _ = apply_top_k_top_p(sliced, top_k=None, top_p=0.9)
        want, _ = apply_top_k_top_p(sliced.contiguous(), top_k=None, top_p=0.9)

        torch.testing.assert_close(got, want)

    def test_top_k_only_still_takes_the_early_return(self):
        """The path that never reached the flatten stays unchanged."""
        logits = torch.randn(3, 6, 11)[:, :-1]

        filtered, mask = apply_top_k_top_p(logits, top_k=3, top_p=1.0)

        assert filtered.shape == logits.shape
        assert mask is not None

    def test_no_filtering_returns_the_input_untouched(self):
        logits = torch.randn(3, 6, 11)[:, :-1]

        filtered, mask = apply_top_k_top_p(logits, top_k=None, top_p=1.0)

        assert filtered is logits and mask is None


def _parameter_view_sites(path: pathlib.Path) -> list[str]:
    """`.view()` calls whose receiver is a parameter of the enclosing function.

    This is the class, stated as a rule rather than a list of line numbers, so
    it keeps holding as the files move. Receivers built inside the function --
    a fresh `torch.arange`, anything after an explicit `.contiguous()` -- are
    the function's own to `view`, and are not flagged.
    """
    # Calls that produce a known-contiguous tensor: once one of these appears
    # between the parameter and the .view, the layout is the function's own.
    NORMALIZERS = {"contiguous", "reshape", "flatten", "clone", "detach_copy"}

    def root(expr):
        """(root Name id, normalizers seen on the way) for an attribute chain."""
        seen: set[str] = set()
        while True:
            if isinstance(expr, ast.Call):
                expr = expr.func
            elif isinstance(expr, ast.Attribute):
                if expr.attr in NORMALIZERS:
                    seen.add(expr.attr)
                expr = expr.value
            elif isinstance(expr, ast.Name):
                return expr.id, seen
            else:
                return None, seen

    tree = ast.parse(path.read_text())
    found: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        names = {
            a.arg
            for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
            if a.arg not in ("self", "cls")
        }
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Attribute) and f.attr == "view"):
                continue
            name, normalized = root(f.value)
            if name in names and not normalized:
                found.append(f"{path.name}:{node.lineno} .view() rooted at {name!r}")
    return found


class TestTheClassStaysFixed:
    """A new `.view()` on a parameter is the same defect, wherever it lands."""

    SWEPT = [
        "algorithms/logits_sampling_utils.py",
        "algorithms/grpo.py",
        "distributed/model_utils.py",
        "models/dtensor/moe/experts.py",
        "models/dtensor/moe/dispatch.py",
        "models/policy/utils.py",
    ]

    @pytest.mark.parametrize("relpath", SWEPT)
    def test_no_view_on_a_received_tensor(self, relpath):
        sites = _parameter_view_sites(ROOT / relpath)
        assert not sites, (
            "`.view()` on a parameter: the caller owns that tensor's layout, so "
            "a slice reaches it and raises a stride error. Use `.reshape()`.\n  "
            + "\n  ".join(sites)
        )

    def test_the_detector_finds_planted_ones(self):
        """Otherwise an empty result would prove nothing.

        Both shapes: a bare parameter, and a parameter reached through an
        attribute chain (`tensor.data.view(...)`, which the refit pack path
        used and a Name-only detector walked straight past).
        """
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(
                "def f(x, y):\n"
                "    a = x.view(-1)\n"          # bare parameter
                "    b = y.data.view(-1)\n"     # through an attribute chain
                "    return a, b\n"
            )
            planted = pathlib.Path(fh.name)
        try:
            assert len(_parameter_view_sites(planted)) == 2
        finally:
            planted.unlink()

    def test_it_does_not_flag_a_normalized_or_locally_built_tensor(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(
                "import torch\n"
                "def f(n, t):\n"
                "    a = torch.arange(n).view(-1, 1)\n"        # built here
                "    b = t.contiguous().view(-1)\n"            # normalized first
                "    c = t.data.reshape(-1).view(dtype=torch.uint8)\n"
                "    return a, b, c\n"
            )
            benign = pathlib.Path(fh.name)
        try:
            assert not _parameter_view_sites(benign)
        finally:
            benign.unlink()


class TestKeepTopKAcceptsNull:
    """The other half of #3709: the type said int, everything else said optional."""

    def test_the_config_type_admits_none(self):
        import typing

        from dockyard_rl.utils.checkpoint import CheckpointingConfig

        hints = typing.get_type_hints(CheckpointingConfig, include_extras=True)

        def flatten(annotation):
            args = typing.get_args(annotation)
            if not args:
                return {annotation}
            out: set = set()
            for a in args:
                out |= flatten(a)
            return out

        assert type(None) in flatten(hints["keep_top_k"]), (
            "keep_top_k is annotated int-only while CheckpointManager defaults it "
            "to None and branches on None to mean 'keep everything'"
        )

    def test_none_means_keep_everything(self, tmp_path):
        from dockyard_rl.utils.checkpoint import CheckpointManager

        manager = CheckpointManager(
            {
                "enabled": True,
                "checkpoint_dir": str(tmp_path),
                "metric_name": None,
                "higher_is_better": True,
                "save_period": 1,
                "keep_top_k": None,
            }
        )
        assert manager.keep_top_k is None
