"""Shared pytest fixtures and import shims for the dockyard_rl core suite.

The `dockyard_rl.distributed` package eagerly imports a compiled `nccl`
extension that is only present on the GPU image. Tests here exercise pure
Python control flow (no NCCL/GPU), so we register a minimal `nccl` stub before
any `dockyard_rl.*` import lets the package load on a plain host. The stub only
answers the three names `stateless_process_group` imports; anything else raises
AttributeError so it never masks a real attribute.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Make `import dockyard_rl` resolve (the package dir is the repo root's name).
_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

_NCCL_PACKAGES = {"nccl", "nccl.core"}
_NCCL_ATTRS = {"Communicator", "UniqueId", "get_unique_id"}


class _NcclStub(types.ModuleType):
    def __getattr__(self, name: str):
        if name in _NCCL_ATTRS:
            return type(name, (), {})
        raise AttributeError(name)


for _name in ["nccl", "nccl.core", "nccl.core.communicator", "nccl.core.utils"]:
    if _name not in sys.modules:
        _mod = _NcclStub(_name)
        if _name in _NCCL_PACKAGES:
            _mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[_name] = _mod

# `dockyard_rl.utils.logger` imports `torch.utils.tensorboard.SummaryWriter` at
# module top. tensorboard is a logging-only dep present on the training image but
# absent on a plain host, and SummaryWriter is never instantiated in unit tests.
# Register a minimal stub only when the real module is unavailable so logger-
# importing modules (e.g. algorithms.utils) load on a plain host; the image is unaffected.
import importlib.util as _importlib_util

if _importlib_util.find_spec("tensorboard") is None:
    _tb_stub = types.ModuleType("torch.utils.tensorboard")
    _tb_stub.SummaryWriter = type("SummaryWriter", (), {})  # type: ignore[attr-defined]
    sys.modules["torch.utils.tensorboard"] = _tb_stub
