"""Transport selection in weight_sync/factory.py.

The factory had no tests. The case that motivated these: colocated + vLLM falls
through to the IPC/ZMQ transport, which is not wired end to end -- Policy has no
stream_weights_via_ipc_zmq, the DTensor worker's raises NotImplementedError, and
models/policy/ipc_utils.py does not exist. The factory used to hand back an
IPCWeightSynchronizer anyway, so the failure surfaced at the first
sync_weights() call (grpo.py:1376) as an AttributeError naming none of the
causes -- after cluster bring-up, model load, engine init and a full rollout.

Three shipped configs select that combination: distillation.yaml, eval.yaml and
gpqa.yaml. The rest are colocated: false (collective) or SGLang (HTTP).

No Ray, GPU or engine is needed: the factory only inspects its arguments before
choosing, and the objects are opaque to it.
"""

from typing import Any

import pytest

from dockyard_rl.models.generation.constants import SGLANG_BACKEND, VLLM_BACKEND
from dockyard_rl.weight_sync.factory import create_weight_synchronizer

_SENTINEL = object()  # the factory never introspects policy/generation


def _make(**kw):
    # Annotated so the ** splat keeps parameter types; a bare dict(...) infers
    # dict[str, object] and every argument then reads as object.
    args: dict[str, Any] = dict(
        policy=_SENTINEL,
        generation=_SENTINEL,
        generation_backend=VLLM_BACKEND,
        colocated=True,
    )
    args.update(kw)
    return create_weight_synchronizer(**args)


class TestColocatedVllmIsRefused:
    def test_raises_at_construction_not_at_first_sync(self):
        with pytest.raises(NotImplementedError) as exc:
            _make(generation_backend=VLLM_BACKEND, colocated=True)
        assert "not implemented" in str(exc.value).lower()

    def test_message_names_all_three_missing_pieces(self):
        """A caller must be able to act on this without reading the source."""
        with pytest.raises(NotImplementedError) as exc:
            _make()
        msg = str(exc.value)
        for piece in (
            "ipc_utils.py",
            "stream_weights_via_ipc_zmq",
            "Policy",
        ):
            assert piece in msg, f"message does not name {piece!r}"

    def test_message_names_the_working_alternative(self):
        with pytest.raises(NotImplementedError) as exc:
            _make()
        assert "colocated.enabled=false" in str(exc.value)

    def test_refit_buffer_validation_still_runs_first(self):
        """A bad buffer size is a config error and keeps its own message."""
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            _make(refit_buffer_size_gb=0)


class TestOtherTransportsUnaffected:
    """The refusal must be narrow -- only colocated + vLLM."""

    def test_colocated_sglang_selects_http(self):
        sync = _make(generation_backend=SGLANG_BACKEND, colocated=True)
        assert type(sync).__name__ == "HTTPWeightSynchronizer"

    def test_non_colocated_vllm_selects_collective(self):
        sync = _make(
            colocated=False, train_cluster=_SENTINEL, inference_world_size=8
        )
        assert type(sync).__name__ == "CollectiveWeightSynchronizer"

    def test_non_colocated_sglang_still_rejected_for_its_own_reason(self):
        with pytest.raises(NotImplementedError, match="non-colocated"):
            _make(generation_backend=SGLANG_BACKEND, colocated=False)

    def test_unknown_backend_still_rejected(self):
        with pytest.raises(ValueError, match="Unknown generation backend"):
            _make(generation_backend="not-a-backend")

    def test_non_colocated_requires_cluster_args(self):
        with pytest.raises(ValueError, match="train_cluster and inference_world_size"):
            _make(colocated=False)
