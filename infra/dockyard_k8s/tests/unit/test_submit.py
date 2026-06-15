"""Tests for dockyard_k8s.submit — dashboard access + kubectl preflight."""

from __future__ import annotations

import pytest

from dockyard_k8s import submit


class TestInCluster:
    def test_true_when_service_host_set(self, monkeypatch) -> None:
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        assert submit.is_in_cluster() is True

    def test_false_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        assert submit.is_in_cluster() is False


class TestDashboardUrl:
    def test_in_cluster_uses_service_dns(self, monkeypatch) -> None:
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        with submit.dashboard_url("dockyard-grpo", "ns") as url:
            assert url == "http://dockyard-grpo-head-svc.ns.svc.cluster.local:8265"


class TestPreflight:
    def test_missing_kubectl_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(submit, "_KUBECTL_OK", None)
        monkeypatch.setattr(submit.shutil, "which", lambda _: None)
        with pytest.raises(RuntimeError, match="kubectl not found"):
            submit.kubectl_preflight("ns")


def test_free_port_is_int() -> None:
    port = submit._free_port()
    assert isinstance(port, int) and port > 0
