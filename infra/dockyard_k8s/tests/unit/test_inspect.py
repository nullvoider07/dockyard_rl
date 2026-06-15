"""Tests for dockyard_k8s.inspect — cluster + sandbox status (boundary stubbed)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from dockyard_k8s import inspect as ins
from src.dockyard_k8s.config import LoadedConfig


def _loaded(infra) -> LoadedConfig:
    return LoadedConfig(recipe=OmegaConf.create({}), infra=infra, source_path=Path("r.yaml"))


class TestClusterStatus:
    def test_none_when_no_kuberay(self, make_infra) -> None:
        assert ins.collect_cluster_status(_loaded(make_infra())) is None

    def test_reports_state_and_pods(self, make_infra, make_cluster, monkeypatch) -> None:
        monkeypatch.setattr(
            ins.k8s, "get_raycluster", lambda n, ns: {"status": {"state": "ready"}}
        )
        monkeypatch.setattr(
            ins, "list_cluster_pods",
            lambda n, ns: ins.RayClusterPods(
                head_name="c-head", head_phase="Running",
                worker_names=["c-w0"], worker_phases=["Running"],
            ),
        )
        infra = make_infra(kuberay=make_cluster().model_dump())
        st = ins.collect_cluster_status(_loaded(infra))
        assert st is not None
        assert st.state == "ready"
        assert st.head_phase == "Running"
        assert st.worker_phases == ["Running"]


class TestSandboxStatus:
    def test_none_when_no_sandbox(self, make_infra) -> None:
        assert ins.collect_sandbox_status(_loaded(make_infra())) is None

    def test_not_found_reports_unavailable(self, make_infra, sandbox, monkeypatch) -> None:
        monkeypatch.setattr(ins.k8s, "get_deployment", lambda n, ns: None)
        infra = make_infra(sandbox=sandbox.model_dump())
        st = ins.collect_sandbox_status(_loaded(infra))
        assert st is not None
        assert st.ready == 0 and st.available is False

    def test_ready_when_replicas_match(self, make_infra, sandbox, monkeypatch) -> None:
        dep = SimpleNamespace(
            spec=SimpleNamespace(replicas=3),
            status=SimpleNamespace(ready_replicas=3),
        )
        monkeypatch.setattr(ins.k8s, "get_deployment", lambda n, ns: dep)
        infra = make_infra(sandbox=sandbox.model_dump())
        st = ins.collect_sandbox_status(_loaded(infra))
        assert st is not None
        assert st.ready == 3 and st.available is True
