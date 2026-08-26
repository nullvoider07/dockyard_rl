"""Transport selection in weight_sync/factory.py.

The factory had no tests. The case that motivated these: colocated + vLLM falls
through to the IPC/ZMQ transport, which is not wired end to end -- Policy has no
stream_weights_via_ipc_zmq, the DTensor worker's raises NotImplementedError, and
the policy worker has no ZMQ plumbing. The factory used to hand back an
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


class TestColocatedVllmSelectsIPC:
    """The guard that used to live here is gone: the transport is now wired.

    It is NOT hardware-validated -- CUDA-IPC handle exchange and the REQ/REP
    handshake are unreachable from a CPU suite -- so these assert selection and
    the sender/receiver contract, not that a refit succeeds.
    """

    def test_selects_the_ipc_synchronizer(self):
        sync = _make(generation_backend=VLLM_BACKEND, colocated=True)
        assert type(sync).__name__ == "IPCWeightSynchronizer"

    def test_refit_buffer_validation_still_runs_first(self):
        """A bad buffer size is a config error and keeps its own message."""
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            _make(refit_buffer_size_gb=0)


class TestIPCSenderContract:
    """Both ends must derive the same socket path and reject the same inputs."""

    def test_sender_and_receiver_addresses_are_derived_identically(self):
        """Colocated means one GPU, so each side computes it from its own UUID.

        Pinned as a string comparison because a divergence here would surface
        only as a refit that hangs until the 120s ZMQ timeout.
        """
        from dockyard_rl.models.generation.vllm.vllm_backend import (
            VllmInternalWorkerExtension,
        )
        from dockyard_rl.models.policy.utils import refit_socket_dir
        from dockyard_rl.models.policy.workers.dtensor_policy_worker import (
            DTensorPolicyWorkerImpl,
        )

        uuid = "GPU-deadbeef-0000-0000-0000-000000000000"

        sender = object.__new__(DTensorPolicyWorkerImpl)
        sender.report_device_id = lambda: uuid  # type: ignore[method-assign]
        receiver = object.__new__(VllmInternalWorkerExtension)
        receiver.report_device_id = lambda: uuid  # type: ignore[method-assign]

        assert sender.get_ipc_zmq_address() == receiver.get_zmq_address()
        # The endpoint moved out of shared /tmp into a per-uid 0700 directory
        # (2026-08-26 audit follow-up); what this test pins is that both ends
        # still derive the identical path, since a divergence surfaces only as
        # a refit hanging to the 120s ZMQ timeout.
        assert sender.get_ipc_zmq_address().endswith(f"/{uuid}.sock")
        assert sender.get_ipc_zmq_address().startswith(
            f"ipc://{refit_socket_dir()}/"
        )

    def test_kv_scales_is_rejected_not_silently_dropped(self):
        """#3226 as originally reported: accepted and never referenced.

        The DTensor path does not support FP8 KV cache at all, which
        broadcast_weights_for_collective already says out loud. The IPC sender
        now says the same rather than accepting the argument and ignoring it.
        """
        from dockyard_rl.models.policy.workers.dtensor_policy_worker import (
            DTensorPolicyWorkerImpl,
        )

        worker = object.__new__(DTensorPolicyWorkerImpl)
        with pytest.raises(NotImplementedError, match="FP8 kvcache"):
            DTensorPolicyWorkerImpl.stream_weights_via_ipc_zmq(
                worker, buffer_size_bytes=0, kv_scales={"k": 1.0}
            )

    def test_policy_exposes_the_dispatch_the_synchronizer_calls(self):
        """IPCWeightSynchronizer:69 calls this; its absence was the real defect."""
        from dockyard_rl.models.policy.lm_policy import Policy

        assert hasattr(Policy, "stream_weights_via_ipc_zmq")


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


class TestReceiverImportsResolve:
    """The IPC receiver's deferred import must name a module that exists.

    It read from dockyard_rl.models.policy.ipc_utils, which has never existed,
    so the receiver would have raised ImportError at refit time even once the
    sender was built. Deferred imports are the convention for not-yet-written
    modules (CLAUDE.md), which is precisely why a permanently-wrong path went
    unnoticed: nothing fails until the call.
    """

    def test_the_three_symbols_resolve_from_policy_utils(self):
        from dockyard_rl.models.policy.utils import (
            IPCProtocol,
            calculate_aligned_size,
            rebuild_cuda_tensor_from_ipc,
        )

        assert IPCProtocol.COMPLETE.value == "complete"
        assert IPCProtocol.ACK.value == "ack"
        assert callable(calculate_aligned_size)
        assert callable(rebuild_cuda_tensor_from_ipc)

    def test_sender_and_receiver_agree_on_alignment(self):
        """Both sides advance the buffer offset with the same helper.

        The receiver asserts offset == used_bytes after unpacking; a divergent
        alignment rule would surface there as a confusing shape error.
        """
        from dockyard_rl.models.policy.utils import calculate_aligned_size

        assert calculate_aligned_size(1) == 512
        assert calculate_aligned_size(512) == 512
        assert calculate_aligned_size(513) == 1024

    def test_phantom_module_is_gone_from_the_import_graph(self):
        import importlib.util

        assert (
            importlib.util.find_spec("dockyard_rl.models.policy.ipc_utils") is None
        ), "ipc_utils now exists; the receiver import should point at it instead"
