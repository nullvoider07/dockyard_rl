"""RoCE rail discovery and the missing-dependency message (C4 / #2935 groundwork).

`data_plane/rdma.py::_mooncake_transport_config` picked its RDMA rail with a shell pipeline
(`for d in /sys/class/infiniband/mlx5_*/ports/1/link_layer; ... | head -1`),
which could only be exercised on a node that had the hardware, read port 1
alone, and -- when it found nothing -- returned `device_name: ""` with
`protocol: "rdma"`, which reads as a configured RDMA transport and is not one.

Offering *every* rail, which is upstream #2935's actual content, is not
implemented: it needs to know whether mooncake's `device_name` accepts a list,
and neither mooncake nor transfer_queue is installed here or present in
`ubuntu-base/uv.lock`. The single-rail choice is reported rather than silently
made, and the discovery is now a pure function over a sysfs root.
"""

from __future__ import annotations

import pytest

from dockyard_rl.data_plane import rdma as tq_adapter


def _sysfs(tmp_path, layout: dict[str, dict[str, str]]):
    """Build a fake /sys/class/infiniband: {device: {port: link_layer}}."""
    root = tmp_path / "infiniband"
    for device, ports in layout.items():
        for port, link_layer in ports.items():
            port_dir = root / device / "ports" / port
            port_dir.mkdir(parents=True)
            (port_dir / "link_layer").write_text(link_layer + "\n")
    return str(root)


class TestDiscovery:
    def test_a_roce_rail_is_found(self, tmp_path):
        root = _sysfs(tmp_path, {"mlx5_0": {"1": "Ethernet"}})

        assert tq_adapter.discover_roce_devices(root) == ["mlx5_0"]

    def test_an_infiniband_rail_is_excluded(self, tmp_path):
        """Deliberate: the caller pins MC_GID_INDEX=3, the RoCEv2 GID index."""
        root = _sysfs(tmp_path, {"mlx5_0": {"1": "InfiniBand"}})

        assert tq_adapter.discover_roce_devices(root) == []

    def test_a_second_port_is_no_longer_invisible(self, tmp_path):
        """The shell pipeline read ports/1 only."""
        root = _sysfs(tmp_path, {"mlx5_0": {"1": "InfiniBand", "2": "Ethernet"}})

        assert tq_adapter.discover_roce_devices(root) == ["mlx5_0"]

    def test_every_rail_is_reported_sorted_and_once(self, tmp_path):
        root = _sysfs(
            tmp_path,
            {
                "mlx5_2": {"1": "Ethernet"},
                "mlx5_0": {"1": "Ethernet", "2": "Ethernet"},
                "mlx5_1": {"1": "InfiniBand"},
            },
        )

        assert tq_adapter.discover_roce_devices(root) == ["mlx5_0", "mlx5_2"]

    def test_non_mlx5_devices_are_ignored(self, tmp_path):
        root = _sysfs(tmp_path, {"irdma0": {"1": "Ethernet"}})

        assert tq_adapter.discover_roce_devices(root) == []

    def test_a_missing_sysfs_root_is_not_an_error(self, tmp_path):
        assert tq_adapter.discover_roce_devices(str(tmp_path / "absent")) == []

    def test_a_device_without_a_ports_directory_is_skipped(self, tmp_path):
        root = tmp_path / "infiniband"
        (root / "mlx5_0").mkdir(parents=True)

        assert tq_adapter.discover_roce_devices(str(root)) == []


class TestTransportConfig:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for key in ("MC_MOONCAKE_PROTOCOL", "MC_MOONCAKE_DEVICE", "MC_GID_INDEX"):
            monkeypatch.delenv(key, raising=False)

    def test_tcp_is_the_default_and_reads_no_sysfs(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            tq_adapter, "discover_roce_devices", lambda *a, **k: called.append(1) or []
        )

        assert tq_adapter._mooncake_transport_config() == {"protocol": "tcp"}
        assert called == []

    def test_an_explicit_device_short_circuits_discovery(self, monkeypatch):
        monkeypatch.setenv("MC_MOONCAKE_PROTOCOL", "rdma")
        monkeypatch.setenv("MC_MOONCAKE_DEVICE", "mlx5_7")
        monkeypatch.setattr(
            tq_adapter,
            "discover_roce_devices",
            lambda *a, **k: pytest.fail("discovery ran despite MC_MOONCAKE_DEVICE"),
        )

        config = tq_adapter._mooncake_transport_config()

        assert config == {"protocol": "rdma", "device_name": "mlx5_7"}

    def test_a_discovered_device_is_used_and_pins_the_gid_index(self, monkeypatch):
        monkeypatch.setenv("MC_MOONCAKE_PROTOCOL", "rdma")
        monkeypatch.setattr(
            tq_adapter, "discover_roce_devices", lambda *a, **k: ["mlx5_0", "mlx5_1"]
        )

        config = tq_adapter._mooncake_transport_config()

        assert config == {"protocol": "rdma", "device_name": "mlx5_0"}
        import os

        assert os.environ["MC_GID_INDEX"] == "3"

    def test_multiple_rails_are_reported_not_silently_dropped(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("MC_MOONCAKE_PROTOCOL", "rdma")
        monkeypatch.setattr(
            tq_adapter, "discover_roce_devices", lambda *a, **k: ["mlx5_0", "mlx5_1"]
        )

        with caplog.at_level("INFO", logger=tq_adapter.__name__):
            tq_adapter._mooncake_transport_config()

        assert "mlx5_1" in caplog.text

    def test_finding_nothing_warns_instead_of_passing_an_empty_device(
        self, monkeypatch, caplog
    ):
        """The silent case: protocol=rdma with device_name='' reads as configured."""
        monkeypatch.setenv("MC_MOONCAKE_PROTOCOL", "rdma")
        monkeypatch.setattr(tq_adapter, "discover_roce_devices", lambda *a, **k: [])

        with caplog.at_level("WARNING", logger=tq_adapter.__name__):
            config = tq_adapter._mooncake_transport_config()

        assert config == {"protocol": "rdma", "device_name": ""}
        assert "no RoCE device" in caplog.text


class TestTheMissingDependencyIsNamed:
    def test_the_image_lock_does_not_carry_the_data_plane_packages(self):
        """Pins the premise behind the factory's message. Goes red when added."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        lock = (root / "ubuntu-base" / "uv.lock").read_text()
        pyproject = (root / "ubuntu-base" / "pyproject.toml").read_text()

        for package in ("transfer_queue", "transfer-queue", "mooncake", "tensordict"):
            assert package not in lock, f"{package} is now in the image lock"
            assert package not in pyproject, f"{package} is now in the image deps"

    def test_the_factory_explains_the_gap_rather_than_raising_import_error(
        self, monkeypatch
    ):
        import builtins

        from dockyard_rl.data_plane import factory

        real_import = builtins.__import__

        def fail_on_adapter(name, *args, **kwargs):
            if name.endswith("adapters.transfer_queue"):
                raise ImportError("No module named 'transfer_queue'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_on_adapter)

        with pytest.raises(RuntimeError) as excinfo:
            factory.build_data_plane_client(
                {"enabled": True, "impl": "transfer_queue"}  # type: ignore[arg-type]
            )

        message = str(excinfo.value)
        assert "ubuntu-base/pyproject.toml" in message
        assert "impl='noop'" in message
