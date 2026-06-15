"""Tests for dockyard_k8s.cli — the check / render commands.

The loader is stubbed (it has its own tests); these exercise the CLI surface:
summary formatting, bundle writing, multi-doc YAML rendering, and that
``--infra`` is threaded through.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from omegaconf import OmegaConf

from dockyard_k8s import cli as cli_mod
from src.dockyard_k8s.config import LoadedConfig
from src.dockyard_k8s.schema import InfraConfig


@pytest.fixture
def files(tmp_path) -> tuple[Path, Path]:
    recipe = tmp_path / "grpo_swe.yaml"
    recipe.write_text("grpo: {}\n")
    infra = tmp_path / "grpo_swe.infra.yaml"
    infra.write_text("namespace: dockyard\nimage: img\n")
    return recipe, infra


@pytest.fixture
def stub_loaded(monkeypatch):
    """Make cli.load_recipe_with_infra return a fixed LoadedConfig."""
    def _install(infra: InfraConfig) -> LoadedConfig:
        loaded = LoadedConfig(
            recipe=OmegaConf.create({"grpo": {}}),
            infra=infra,
            source_path=Path("grpo_swe.yaml"),
        )
        monkeypatch.setattr(
            cli_mod, "load_recipe_with_infra", lambda *a, **k: loaded
        )
        return loaded

    return _install


def _infra(make_cluster, **kw) -> InfraConfig:
    payload = {
        "namespace": "dockyard",
        "image": "nullvoider/ubuntu-swe:latest",
        "scheduler": {"kind": "kai", "queue": "rl"},
        "kuberay": make_cluster().model_dump(),
        "sandbox": {"name": "dockyard-sandbox", "replicas": 4, "port": 9090},
        "launch": {"mode": "rayjob", "entrypoint": "run.sh"},
    }
    payload.update(kw)
    return InfraConfig.model_validate(payload)


class TestCheck:
    def test_summary(self, files, stub_loaded, make_cluster) -> None:
        stub_loaded(_infra(make_cluster))
        recipe, infra = files
        res = CliRunner().invoke(cli_mod.main, ["check", str(recipe), "--infra", str(infra)])
        assert res.exit_code == 0, res.output
        assert "namespace:  dockyard" in res.output
        assert "nullvoider/ubuntu-swe:latest" in res.output
        assert "kai (queue=rl)" in res.output
        assert "workerGroups=[trainer, inference]" in res.output
        # sandbox URL line present
        assert "dockyard-sandbox.dockyard.svc.cluster.local:9090" in res.output
        assert "RayJob/dockyard-grpo" in res.output

    def test_bundle_written(self, files, stub_loaded, make_cluster, tmp_path) -> None:
        stub_loaded(_infra(make_cluster))
        recipe, infra = files
        out = tmp_path / "bundle.yaml"
        res = CliRunner().invoke(
            cli_mod.main,
            ["check", str(recipe), "--infra", str(infra), "-o", str(out)],
        )
        assert res.exit_code == 0, res.output
        bundle = yaml.safe_load(out.read_text())
        assert set(bundle) == {"infra", "recipe", "manifests"}
        assert bundle["infra"]["namespace"] == "dockyard"
        kinds = [m["kind"] for m in bundle["manifests"]]
        assert kinds == ["Deployment", "Service", "RayJob"]


class TestRender:
    def test_multidoc_yaml(self, files, stub_loaded, make_cluster) -> None:
        stub_loaded(_infra(make_cluster))
        recipe, infra = files
        res = CliRunner().invoke(cli_mod.main, ["render", str(recipe), "--infra", str(infra)])
        assert res.exit_code == 0, res.output
        docs = [d for d in yaml.safe_load_all(res.output) if d]
        assert [d["kind"] for d in docs] == ["Deployment", "Service", "RayJob"]

    def test_render_no_sandbox(self, files, stub_loaded, make_cluster) -> None:
        stub_loaded(_infra(make_cluster, sandbox=None))
        recipe, infra = files
        res = CliRunner().invoke(cli_mod.main, ["render", str(recipe), "--infra", str(infra)])
        assert res.exit_code == 0, res.output
        docs = [d for d in yaml.safe_load_all(res.output) if d]
        assert [d["kind"] for d in docs] == ["RayJob"]


class TestRun:
    def test_rayjob_dry_run_emits_manifests(self, files, stub_loaded, make_cluster) -> None:
        # launch.mode=rayjob + --dry-run renders without touching a cluster.
        stub_loaded(_infra(make_cluster))
        recipe, infra = files
        res = CliRunner().invoke(
            cli_mod.main, ["run", str(recipe), "--infra", str(infra), "--dry-run"]
        )
        assert res.exit_code == 0, res.output
        docs = [d for d in yaml.safe_load_all(res.output) if d]
        kinds = [d["kind"] for d in docs]
        assert "RayJob" in kinds and "Deployment" in kinds

    def test_missing_entrypoint_errors(self, files, stub_loaded, make_cluster) -> None:
        stub_loaded(_infra(make_cluster, launch={"mode": "rayjob"}))  # no entrypoint
        recipe, infra = files
        res = CliRunner().invoke(
            cli_mod.main, ["run", str(recipe), "--infra", str(infra), "--dry-run"]
        )
        assert res.exit_code != 0
        assert "entrypoint" in res.output


class TestStatus:
    def test_reports_cluster_and_sandbox(self, files, stub_loaded, make_cluster, monkeypatch) -> None:
        loaded = stub_loaded(_infra(make_cluster))
        from dockyard_k8s import inspect as ins

        monkeypatch.setattr(
            ins, "collect_cluster_status",
            lambda lo: ins.ClusterStatus(
                name="dockyard-grpo", state="ready", head_pod="h", head_phase="Running",
                worker_phases=["Running", "Running"],
            ),
        )
        monkeypatch.setattr(
            ins, "collect_sandbox_status",
            lambda lo: ins.SandboxStatus(name="dockyard-sandbox", replicas=4, ready=4, available=True),
        )
        recipe, infra = files
        res = CliRunner().invoke(cli_mod.main, ["status", str(recipe), "--infra", str(infra)])
        assert res.exit_code == 0, res.output
        assert "state=ready" in res.output
        assert "4/4 ready" in res.output
