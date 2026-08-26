"""The colocated-refit IPC socket: private directory, and a narrowed unpickler.

Raised by the 2026-08-26 branch security audit. The endpoint was
`ipc:///tmp/{device_uuid}.sock` -- a world-writable directory, under a name any
local process can read out of `nvidia-smi -L` -- and the receiver *connects*
while the trainer *binds*, so whoever holds the bind drives the receiver's
`recv_pyobj()`, which is `pickle.loads`.

The audit's verdict was that this is not exploitable in the shipped
deployment: the socket lives in the mount namespace the trainer and vLLM
actors share, the six configs that select this transport run no
model-generated code, and every config that does run it selects NCCL or
SGLang/HTTP instead. So it was an absent hardening measure, not a finding --
and this closes it before anything untrusted is ever colocated.

Two independent changes, either of which alone would leave a gap:
  * the directory is 0700 and ownership-checked, so a *different* account can no
    longer create the socket. A same-uid process still can, but a same-uid
    process can already attach to the trainer, so no boundary is crossed there;
  * the payload is unpickled under an allowlist, so a `__reduce__` gadget
    cannot resolve regardless of who reaches the socket.
"""

from __future__ import annotations

import io
import os
import pickle
import stat

import pytest
import torch

from dockyard_rl.models.policy.utils import (
    REFIT_SOCKET_DIR_ENV,
    IPCProtocol,
    loads_refit_payload,
    refit_socket_address,
    refit_socket_dir,
)

_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


@pytest.fixture
def socket_dir(tmp_path, monkeypatch):
    path = tmp_path / "refit"
    monkeypatch.setenv(REFIT_SOCKET_DIR_ENV, str(path))
    return path


class TestThePrivateDirectory:
    def test_it_is_created_owner_only(self, socket_dir):
        resolved = refit_socket_dir()

        assert resolved == str(socket_dir)
        mode = stat.S_IMODE(os.stat(resolved).st_mode)
        assert mode & 0o077 == 0, oct(mode)

    def test_creating_it_twice_is_fine(self, socket_dir):
        assert refit_socket_dir() == refit_socket_dir()

    def test_the_address_lives_inside_it(self, socket_dir):
        address = refit_socket_address(_UUID)

        assert address == f"ipc://{socket_dir}/{_UUID}.sock"
        # The point of the change: no longer directly in a shared directory.
        assert not address.startswith("ipc:///tmp/GPU-")

    def test_both_ends_derive_the_same_address(self, socket_dir):
        assert refit_socket_address(_UUID) == refit_socket_address(_UUID)

    def test_a_group_or_world_accessible_directory_is_refused(self, socket_dir):
        socket_dir.mkdir(parents=True)
        os.chmod(socket_dir, 0o755)

        with pytest.raises(RuntimeError, match="group- or"):
            refit_socket_dir()

    def test_a_symlink_is_refused(self, tmp_path, monkeypatch):
        target = tmp_path / "elsewhere"
        target.mkdir(mode=0o700)
        link = tmp_path / "refit"
        link.symlink_to(target)
        monkeypatch.setenv(REFIT_SOCKET_DIR_ENV, str(link))

        with pytest.raises(RuntimeError, match="symlink"):
            refit_socket_dir()

    def test_a_non_directory_is_refused(self, tmp_path, monkeypatch):
        path = tmp_path / "refit"
        path.write_text("")
        monkeypatch.setenv(REFIT_SOCKET_DIR_ENV, str(path))

        with pytest.raises(RuntimeError, match="not a directory"):
            refit_socket_dir()

    def test_a_directory_owned_by_someone_else_is_refused(
        self, socket_dir, monkeypatch
    ):
        socket_dir.mkdir(parents=True, mode=0o700)
        real_lstat = os.lstat

        def foreign(path, *args, **kwargs):
            info = real_lstat(path, *args, **kwargs)
            if str(path) == str(socket_dir):
                return os.stat_result(
                    (info.st_mode, info.st_ino, info.st_dev, info.st_nlink,
                     info.st_uid + 1, *tuple(info)[5:10])
                )
            return info

        monkeypatch.setattr(os, "lstat", foreign)

        with pytest.raises(RuntimeError, match="owned by uid"):
            refit_socket_dir()

    def test_the_default_location_is_uid_scoped(self, monkeypatch):
        """Without the override, the name still cannot collide across accounts."""
        monkeypatch.delenv(REFIT_SOCKET_DIR_ENV, raising=False)
        monkeypatch.setattr(
            "dockyard_rl.models.policy.utils._ensure_private_dir", lambda path: None
        )

        assert refit_socket_dir() == f"/tmp/dockyard-refit-{os.getuid()}"


def _handle_payload():
    """The shape reduce_tensor's CUDA branch produces, built on CPU."""
    args = (
        torch.Tensor,
        torch.Size([2, 3]),
        (3, 1),
        0,
        torch.storage.TypedStorage,
        torch.bfloat16,
        0,
        b"\x01" * 64,
        24,
        0,
        False,
        b"\x02" * 64,
        0,
        b"\x03" * 64,
        True,
    )
    return ((args,), ["model.layers.0.mlp.gate_proj.weight"], 512)


class _Gadget:
    def __reduce__(self):
        return (os.system, ("touch /tmp/dockyard-refit-pwned",))


class _EvalGadget:
    def __reduce__(self):
        return (eval, ("1+1",))


class TestTheRestrictedUnpickler:
    def test_a_real_handle_payload_round_trips(self):
        payload = _handle_payload()

        restored = loads_refit_payload(pickle.dumps(payload))

        assert restored == payload

    def test_the_protocol_sentinel_round_trips(self):
        assert loads_refit_payload(pickle.dumps(IPCProtocol.COMPLETE)) is (
            IPCProtocol.COMPLETE
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.uint8])
    def test_every_dtype_the_refit_streams_resolves(self, dtype):
        """Dtypes pickle as module-level torch globals, so they need the escape."""
        assert loads_refit_payload(pickle.dumps(dtype)) is dtype

    @pytest.mark.parametrize("gadget", [_Gadget, _EvalGadget])
    def test_a_reduce_gadget_is_refused(self, gadget):
        blob = pickle.dumps(gadget())

        with pytest.raises(pickle.UnpicklingError, match="refusing to resolve"):
            loads_refit_payload(blob)

    def test_plain_pickle_would_resolve_the_gadget_this_refuses(self):
        """Pins that the allowlist is load-bearing, not decorative.

        Resolving the callable is the step that matters -- `pickle.loads` hands
        back `os.system` itself, which is exactly what a `__reduce__` payload
        needs. The restricted loader never gets that far.
        """
        blob = pickle.dumps(os.system)

        assert pickle.loads(blob) is os.system

        with pytest.raises(pickle.UnpicklingError, match="refusing to resolve"):
            loads_refit_payload(blob)

    def test_a_non_dtype_torch_global_is_still_refused(self):
        """`module == "torch"` is not a blanket allow; only dtypes get through.

        Hand-assembled because nothing reachable pickles as a bare `torch.X`
        global by default -- `pickle.dumps(torch.load)` names
        `torch.serialization.load` -- but a crafted payload can emit the GLOBAL
        opcode directly, which is the case the narrow rule exists for.
        """
        crafted = b"\x80\x04ctorch\ncompile\n."

        assert pickle.loads(crafted) is torch.compile  # plain pickle resolves it

        with pytest.raises(pickle.UnpicklingError, match="refusing to resolve"):
            loads_refit_payload(crafted)


class TestTheReceiversUseIt:
    @pytest.mark.parametrize(
        "module_path,method,next_method",
        [
            (
                "models/generation/vllm/vllm_backend.py",
                "update_weights_via_ipc_zmq",
                "def cleanup",
            ),
            (
                "modelopt/models/generation/vllm_quant_backend.py",
                "update_weights_via_ipc_zmq",
                "def update_weights_from_collective",
            ),
        ],
    )
    def test_no_receiver_calls_recv_pyobj(self, module_path, method, next_method):
        """The wire: both receive loops must go through the allowlist."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        text = (root / module_path).read_text()
        start = text.index(f"def {method}(")
        end = text.index(next_method, start)
        body = text[start:end]

        assert "recv_refit_payload(" in body, module_path
        assert "recv_pyobj(" not in body, module_path

    def test_no_address_is_still_built_by_hand(self):
        """Every end must derive the endpoint from the one helper."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        for module_path in (
            "models/policy/workers/base_policy_worker.py",
            "models/policy/workers/dtensor_policy_worker.py",
            "models/generation/vllm/vllm_backend.py",
        ):
            text = (root / module_path).read_text()
            assert 'f"ipc:///tmp/{' not in text, module_path
            assert "refit_socket_address(" in text, module_path
