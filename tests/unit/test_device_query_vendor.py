"""E1: device queries and visible-device resolution without a vendor in the name.

Three couplings measured against the tree, all real and all smaller or larger
than Track E claimed:

  * `utils/nvml.py` bound the whole package to `pynvml` with a module-level
    import. A host without it could not import `dockyard_rl.utils` at all --
    the failure landed before any code could dispatch on what hardware it had.
    Three queries, twelve call sites; now behind a backend selected once.
  * The visible-device surface is ten non-test files, not the two lines the plan
    named. Ray sets `HIP_VISIBLE_DEVICES` on an AMD node, so every site reading
    `CUDA_VISIBLE_DEVICES` alone reads nothing there.
  * `quantization/fp8.py` hardcoded `torch.float8_e4m3fn` at five sites. CDNA3
    uses `float8_e4m3fnuz`, which encodes zero's sign and NaN differently.

The third is the dangerous one and the first is the loudest, but the second is
the subtle one: the UUID these queries return is what the trainer and the
colocated inference worker each use to derive the refit IPC socket path. A wrong
device answer does not raise -- the two sides compute different paths and the
refit hangs.

The precedence encoded here is torch's, read from `torch.cuda._parse_visible_devices`
rather than recalled, and corroborated by Ray's own AMD accelerator
(`_private/accelerators/amd_gpu.py`), which prefers HIP over CUDA, warns when
ROCR is set without HIP, and refuses inconsistent pairs.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from dockyard_rl.utils import device_query as dq


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in dq.VISIBLE_DEVICE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    dq.set_device_query_backend(None)
    yield
    dq.set_device_query_backend(None)


def _hip(monkeypatch, on: bool = True):
    monkeypatch.setattr(torch.version, "hip", "6.4.0" if on else None, raising=False)


class _FakeBackend:
    name = "fake"

    def __init__(self):
        self.seen: list[int] = []

    def device_uuid(self, physical_index: int) -> str:
        self.seen.append(physical_index)
        return f"GPU-{physical_index:08d}"

    def memory_info(self, physical_index: int) -> tuple[int, int]:
        self.seen.append(physical_index)
        return 1000 + physical_index, 8000 + physical_index


class TestTheQueriesRunWithoutAVendorLibrary:
    def test_a_fake_backend_answers_every_query(self):
        """The point of the seam: no pynvml, no amdsmi, no GPU."""
        dq.set_device_query_backend(_FakeBackend())

        assert dq.get_device_uuid(0) == "GPU-00000000"
        assert dq.get_free_memory_bytes(0) == 1000
        assert dq.get_total_memory_bytes(0) == 8000

    def test_queries_translate_the_logical_index_before_asking(self, monkeypatch):
        backend = _FakeBackend()
        dq.set_device_query_backend(backend)
        monkeypatch.setenv(dq.CUDA_VISIBLE_DEVICES, "4,5,6")

        assert dq.get_device_uuid(1) == "GPU-00000005"
        assert backend.seen == [5]

    def test_the_backend_is_a_protocol_the_fake_satisfies(self):
        assert isinstance(_FakeBackend(), dq.DeviceQueryBackend)

    def test_the_module_does_not_import_a_vendor_library_at_import_time(self):
        """`import dockyard_rl.utils` must not require pynvml to be installed."""
        source = pathlib.Path(dq.__file__).read_text()
        header = source.split("class NvmlBackend")[0]
        assert "import pynvml" not in header


class TestBackendSelection:
    def test_a_rocm_build_refuses_rather_than_selecting_nvml(self, monkeypatch):
        """pynvml is pure Python and installs anywhere; importability is not proof.

        Selecting NVML on a ROCm host would answer every query about hardware
        that is not there.
        """
        _hip(monkeypatch)
        with pytest.raises(RuntimeError, match="amdsmi"):
            dq.resolve_device_query_backend()

    def test_the_refusal_says_what_to_implement(self, monkeypatch):
        _hip(monkeypatch)
        with pytest.raises(RuntimeError, match="set_device_query_backend"):
            dq.resolve_device_query_backend()

    def test_an_installed_backend_wins_over_selection(self, monkeypatch):
        _hip(monkeypatch)
        dq.set_device_query_backend(_FakeBackend())

        assert dq.resolve_device_query_backend().name == "fake"

    def test_selection_is_cached(self, monkeypatch):
        _hip(monkeypatch, on=False)
        first = dq.resolve_device_query_backend()
        assert dq.resolve_device_query_backend() is first


class TestVisibleDeviceResolutionOnCuda:
    def test_nothing_set_is_unrestricted(self):
        assert dq.visible_devices_entries() is None
        assert dq.device_id_to_physical_device_id(3) == 3

    def test_the_cuda_variable_is_read(self, monkeypatch):
        monkeypatch.setenv(dq.CUDA_VISIBLE_DEVICES, "2,3")
        assert dq.visible_devices_entries() == ["2", "3"]
        assert dq.device_id_to_physical_device_id(1) == 3

    def test_hip_variables_are_ignored_on_a_cuda_build(self, monkeypatch):
        """Not a rename: on a CUDA build only the CUDA variable means anything."""
        _hip(monkeypatch, on=False)
        monkeypatch.setenv(dq.HIP_VISIBLE_DEVICES, "6,7")
        assert dq.visible_devices_entries() is None

    def test_an_out_of_range_index_is_refused(self, monkeypatch):
        monkeypatch.setenv(dq.CUDA_VISIBLE_DEVICES, "2")
        with pytest.raises(RuntimeError, match="logical device 3"):
            dq.device_id_to_physical_device_id(3)

    def test_a_uuid_entry_is_refused_rather_than_guessed(self, monkeypatch):
        monkeypatch.setenv(dq.CUDA_VISIBLE_DEVICES, "GPU-deadbeef")
        with pytest.raises(RuntimeError):
            dq.device_id_to_physical_device_id(0)


class TestVisibleDeviceResolutionOnRocm:
    def test_hip_overrides_cuda(self, monkeypatch):
        _hip(monkeypatch)
        monkeypatch.setenv(dq.CUDA_VISIBLE_DEVICES, "0,1")
        monkeypatch.setenv(dq.HIP_VISIBLE_DEVICES, "6,7")

        assert dq.visible_devices_entries() == ["6", "7"]
        assert dq.device_id_to_physical_device_id(0) == 6

    def test_hip_alone_is_read(self, monkeypatch):
        """The case that made every site reading only CUDA read nothing."""
        _hip(monkeypatch)
        monkeypatch.setenv(dq.HIP_VISIBLE_DEVICES, "3")

        assert dq.visible_devices_entries() == ["3"]

    def test_rocr_without_hip_is_refused(self, monkeypatch):
        """torch renumbers the reduced pool from zero; no physical index survives."""
        _hip(monkeypatch)
        monkeypatch.setenv(dq.ROCR_VISIBLE_DEVICES, "2,3")

        with pytest.raises(RuntimeError, match="renumbers"):
            dq.visible_devices_entries()

    def test_hip_naming_more_devices_than_rocr_left_is_refused(self, monkeypatch):
        _hip(monkeypatch)
        monkeypatch.setenv(dq.ROCR_VISIBLE_DEVICES, "2")
        monkeypatch.setenv(dq.HIP_VISIBLE_DEVICES, "0,1")

        with pytest.raises(RuntimeError, match="more devices"):
            dq.visible_devices_entries()

    def test_hip_within_the_rocr_pool_is_accepted(self, monkeypatch):
        _hip(monkeypatch)
        monkeypatch.setenv(dq.ROCR_VISIBLE_DEVICES, "2,3,4")
        monkeypatch.setenv(dq.HIP_VISIBLE_DEVICES, "0,1")

        assert dq.visible_devices_entries() == ["0", "1"]


class TestPropagationCarriesTheVariableName:
    def test_the_cuda_name_travels_on_a_cuda_build(self, monkeypatch):
        monkeypatch.setenv(dq.CUDA_VISIBLE_DEVICES, "1,2")
        assert dq.visible_devices_env() == (dq.CUDA_VISIBLE_DEVICES, "1,2")

    def test_the_hip_name_travels_when_hip_is_what_is_set(self, monkeypatch):
        """Writing CUDA_VISIBLE_DEVICES here would leave the child disagreeing."""
        _hip(monkeypatch)
        monkeypatch.setenv(dq.HIP_VISIBLE_DEVICES, "6,7")

        assert dq.visible_devices_env() == (dq.HIP_VISIBLE_DEVICES, "6,7")

    def test_unrestricted_propagates_nothing(self):
        assert dq.visible_devices_env() is None
        assert dq.visible_devices_value() is None


class TestClearingAndSuppressing:
    def test_clearing_drops_every_vendor_variable(self, monkeypatch):
        """Leaving one set silently remaps whatever the delegated allocator picks."""
        env = {
            dq.CUDA_VISIBLE_DEVICES: "0",
            dq.HIP_VISIBLE_DEVICES: "1",
            dq.ROCR_VISIBLE_DEVICES: "2",
            "KEEP": "yes",
        }
        dq.clear_visible_devices(env)

        assert env == {"KEEP": "yes"}

    def test_clearing_defaults_to_the_process_environment(self, monkeypatch):
        monkeypatch.setenv(dq.HIP_VISIBLE_DEVICES, "1")
        dq.clear_visible_devices()

        assert dq.HIP_VISIBLE_DEVICES not in __import__("os").environ

    def test_suppressing_covers_both_vendors(self):
        env: dict[str, str] = {}
        dq.suppress_ray_visible_devices(env)

        assert set(env) == set(dq.NOSET_VISIBLE_DEVICE_ENV_VARS)
        assert all(v == "1" for v in env.values())

    def test_dropping_opt_outs_covers_both_vendors(self):
        env = {name: "1" for name in dq.NOSET_VISIBLE_DEVICE_ENV_VARS}
        env["KEEP"] = "yes"
        dq.drop_ray_visible_device_opt_outs(env)

        assert env == {"KEEP": "yes"}

    def test_the_ray_variable_names_match_the_installed_ray(self):
        """Guessed env-var names are how this kind of port fails silently."""
        ray_amd = pathlib.Path(
            __import__("ray").__file__
        ).parent / "_private" / "accelerators" / "amd_gpu.py"
        ray_nvidia = ray_amd.parent / "nvidia_gpu.py"
        if not ray_amd.exists() or not ray_nvidia.exists():
            pytest.skip("installed Ray does not expose the accelerator modules")

        text = ray_amd.read_text() + ray_nvidia.read_text()
        for name in dq.NOSET_VISIBLE_DEVICE_ENV_VARS:
            assert name in text, f"{name!r} is not a variable this Ray reads"


class TestEveryConsumerWentThroughTheSweep:
    """Ten non-test files touched the variable; a missed one reads nothing on AMD."""

    ROOT = pathlib.Path(__file__).resolve().parents[2]

    @pytest.mark.parametrize(
        "path",
        [
            "cluster/bootstrap.py",
            "environments/reward_model_environment.py",
            "models/policy/utils.py",
            "models/generation/sglang/sglang_generation.py",
            "models/generation/sglang/sglang_worker.py",
            "models/generation/vllm/vllm_worker.py",
            "distributed/worker_groups.py",
        ],
    )
    def test_no_module_reads_the_raw_variable(self, path):
        source = (self.ROOT / path).read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        # The variable name itself, not just a read of it: writing
        # env_vars["CUDA_VISIBLE_DEVICES"] on a host pinned by HIP is the same
        # bug pointed the other way, and it propagates to every child.
        for pattern in (
            '"CUDA_VISIBLE_DEVICES"',
            '"HIP_VISIBLE_DEVICES"',
            '"ROCR_VISIBLE_DEVICES"',
            '"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"',
            '"RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"',
        ):
            assert pattern not in code, (
                f"{path} names {pattern} directly instead of going through "
                "device_query; on an AMD node Ray sets HIP_VISIBLE_DEVICES, so "
                "reading the CUDA name sees nothing and writing it is ignored"
            )


class TestTheFp8EncodingIsResolved:
    def test_a_cuda_build_gets_ocp_e4m3fn(self):
        from dockyard_rl.utils.device_capability import fp8_e4m3_dtype

        assert fp8_e4m3_dtype() is torch.float8_e4m3fn

    def test_both_encodings_exist_and_differ(self):
        """If they were the same dtype the whole item would be moot."""
        assert torch.float8_e4m3fn is not torch.float8_e4m3fnuz
        assert (
            torch.finfo(torch.float8_e4m3fn).max
            != torch.finfo(torch.float8_e4m3fnuz).max
        )

    def test_it_defers_to_the_pinned_vllm_platform(self):
        """One source of the rule, so it cannot drift from the kernels."""
        from vllm.platforms import current_platform

        from dockyard_rl.utils.device_capability import fp8_e4m3_dtype

        assert fp8_e4m3_dtype() is current_platform.fp8_dtype()

    def test_the_rocm_rule_is_the_one_vllm_encodes(self):
        """Pins the fact the resolver relies on: gfx94 -> fnuz, gfx950 -> fn."""
        import inspect

        from vllm.platforms.rocm import RocmPlatform

        assert "gfx94" in inspect.getsource(RocmPlatform.is_fp8_fnuz)

    def test_no_fp8_site_hardcodes_the_encoding(self):
        fp8 = (
            pathlib.Path(__file__).resolve().parents[2]
            / "models/generation/vllm/quantization/fp8.py"
        )
        source = fp8.read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        assert "torch.float8_e4m3fn" not in code, (
            "a hardcoded e4m3 encoding is back in fp8.py; on CDNA3 that is "
            "silent numerical corruption, not an error"
        )
