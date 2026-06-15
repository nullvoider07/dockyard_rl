"""Tests for dockyard_k8s.orchestrate — pure helpers + submit/run dispatch.

The k8s / Ray boundaries are stubbed; these pin the path-rewrite, drift, and
dispatch logic that decides what gets submitted where.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dockyard_k8s import orchestrate
from dockyard_k8s.config import LoadedConfig
from dockyard_k8s.submitters import SubmissionHandle


def _loaded(infra, recipe=None, source_path="examples/configs/grpo_swe.yaml") -> LoadedConfig:
    return LoadedConfig(
        recipe=OmegaConf.create(recipe or {"grpo": {}}),
        infra=infra,
        source_path=Path(source_path),
    )


class _FakeSubmitter:
    def __init__(self):
        self.calls = []

    def submit(self, cluster_name, namespace, *, entrypoint, run_id, env_vars=None, working_dir=None):
        self.calls.append(
            dict(cluster_name=cluster_name, namespace=namespace, entrypoint=entrypoint,
                 run_id=run_id, env_vars=env_vars, working_dir=working_dir)
        )
        return SubmissionHandle(kind="ray", run_id=run_id or "auto", cluster_name=cluster_name, namespace=namespace)


class TestRecipePathInPod:
    def test_upload_uses_staged_name(self, make_infra) -> None:
        loaded = _loaded(make_infra())
        assert orchestrate._recipe_path_in_pod(loaded, Path("/repo"), upload=True) == orchestrate.STAGED_RECIPE_NAME

    def test_image_uses_repo_relative(self, make_infra, tmp_path) -> None:
        (tmp_path / "examples" / "configs").mkdir(parents=True)
        recipe = tmp_path / "examples" / "configs" / "grpo_swe.yaml"
        recipe.write_text("{}\n")
        loaded = _loaded(make_infra(), source_path=str(recipe))
        out = orchestrate._recipe_path_in_pod(loaded, tmp_path, upload=False)
        assert out == "examples/configs/grpo_swe.yaml"

    def test_image_outside_repo_returns_none(self, make_infra, tmp_path) -> None:
        recipe = tmp_path / "elsewhere.yaml"
        recipe.write_text("{}\n")
        loaded = _loaded(make_infra(), source_path=str(recipe))
        assert orchestrate._recipe_path_in_pod(loaded, tmp_path / "repo", upload=False) is None


class TestRewriteEntrypoint:
    def test_rewrites_config_flag_upload(self, make_infra) -> None:
        loaded = _loaded(make_infra())
        ep = "python3 examples/run_grpo_swe.py --config examples/configs/grpo_swe.yaml x=1"
        out = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, Path("/repo"), upload=True, log=lambda _: None
        )
        assert "--config dockyard_run.yaml" in out
        assert "x=1" in out

    def test_noop_without_config_flag(self, make_infra) -> None:
        loaded = _loaded(make_infra())
        ep = "python3 examples/run_grpo_swe.py"
        out = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, Path("/repo"), upload=True, log=lambda _: None
        )
        assert out == ep

    def test_does_not_match_config_name(self, make_infra) -> None:
        # Hydra's --config-name must not be rewritten.
        loaded = _loaded(make_infra())
        ep = "python3 run.py --config-name foo"
        out = orchestrate._rewrite_entrypoint_recipe(
            ep, loaded, Path("/repo"), upload=True, log=lambda _: None
        )
        assert out == ep


class TestDrift:
    def test_strip_server_fields(self) -> None:
        live = {"status": {"state": "ready"}, "metadata": {"uid": "x", "name": "c"}, "a": 1}
        rendered = {"metadata": {"name": "c"}, "a": 1}
        assert not orchestrate._spec_drifted(live, rendered)

    def test_detects_real_drift(self) -> None:
        assert orchestrate._spec_drifted({"a": 1}, {"a": 2})


class TestMisc:
    def test_default_run_id(self) -> None:
        assert orchestrate.default_run_id("training").startswith("training-")

    def test_upload_paths_default(self, make_infra) -> None:
        paths = orchestrate._upload_paths(make_infra())
        assert "algorithms" in paths

    def test_upload_paths_override(self, make_infra) -> None:
        infra = make_infra(launch={"rayUploadPaths": ["only_this"], "entrypoint": "x"})
        assert orchestrate._upload_paths(infra) == ["only_this"]


class _FakeK8s:
    """Records apply/delete calls so the lifecycle wiring can be asserted."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __getattr__(self, name):
        def _record(*args, **_kwargs):
            # First positional is a manifest dict (apply) or a name str (delete).
            ident = ""
            if args:
                a = args[0]
                ident = a["metadata"]["name"] if isinstance(a, dict) else str(a)
            self.calls.append((name, ident))
            return None
        return _record


class TestSandboxLifecycle:
    def _infra(self, make_infra, **sandbox_over):
        from dockyard_k8s.schema import SandboxSpec

        sb = SandboxSpec.model_validate({"name": "dockyard-sandbox", **sandbox_over})
        return make_infra(sandbox=sb.model_dump())

    def test_ensure_applies_netpol_and_pdb(self, make_infra, monkeypatch) -> None:
        fake = _FakeK8s()
        monkeypatch.setattr(orchestrate, "k8s", fake)
        monkeypatch.setattr(orchestrate, "get_username", lambda: "tester")
        orchestrate.ensure_sandbox(self._loaded(make_infra), log=lambda *_: None)
        applied = [c for c in fake.calls if c[0].startswith("apply")]
        assert ("apply_network_policy", "dockyard-sandbox-netpol") in applied
        assert ("apply_pod_disruption_budget", "dockyard-sandbox-pdb") in applied

    def test_delete_removes_netpol_and_pdb(self, make_infra, monkeypatch) -> None:
        fake = _FakeK8s()
        monkeypatch.setattr(orchestrate, "k8s", fake)
        orchestrate.delete_sandbox(self._loaded(make_infra), log=lambda *_: None)
        deleted = [c for c in fake.calls if c[0].startswith("delete")]
        assert ("delete_pod_disruption_budget", "dockyard-sandbox-pdb") in deleted
        assert ("delete_network_policy", "dockyard-sandbox-netpol") in deleted

    def test_hardening_disabled_skips_objects(self, make_infra, monkeypatch) -> None:
        fake = _FakeK8s()
        monkeypatch.setattr(orchestrate, "k8s", fake)
        monkeypatch.setattr(orchestrate, "get_username", lambda: "tester")
        infra = self._infra(
            make_infra, networkPolicy={"enabled": False}, pdb={"enabled": False}
        )
        orchestrate.ensure_sandbox(self._loaded_infra(infra), log=lambda *_: None)
        names = [c[0] for c in fake.calls]
        assert "apply_network_policy" not in names
        assert "apply_pod_disruption_budget" not in names

    def test_cluster_pdb_applied_when_enabled(self, make_cluster, make_infra, monkeypatch) -> None:
        fake = _FakeK8s()
        monkeypatch.setattr(orchestrate, "k8s", fake)
        cluster_payload = make_cluster().model_dump()
        cluster_payload["pdb"] = {"enabled": True, "minAvailable": 2}
        infra = make_infra(kuberay=cluster_payload)
        orchestrate.ensure_cluster_pdb(self._loaded_infra(infra), log=lambda *_: None)
        assert ("apply_pod_disruption_budget", "dockyard-grpo-pdb") in fake.calls

    def test_cluster_pdb_skipped_by_default(self, make_cluster, make_infra, monkeypatch) -> None:
        fake = _FakeK8s()
        monkeypatch.setattr(orchestrate, "k8s", fake)
        infra = make_infra(kuberay=make_cluster().model_dump())
        orchestrate.ensure_cluster_pdb(self._loaded_infra(infra), log=lambda *_: None)
        assert fake.calls == []

    # helpers
    def _loaded(self, make_infra):
        from dockyard_k8s.schema import SandboxSpec

        infra = make_infra(sandbox=SandboxSpec(name="dockyard-sandbox").model_dump())
        return _loaded(infra)

    def _loaded_infra(self, infra):
        return _loaded(infra)


class TestSubmitTraining:
    def test_exec_with_upload_rejected(self, make_infra) -> None:
        infra = make_infra(
            submit={"submitter": "exec", "execTmpDir": "/tmp"},
            launch={"codeSource": "upload", "entrypoint": "python run.py"},
        )
        with pytest.raises(ValueError, match="incompatible"):
            orchestrate.submit_training(
                _loaded(infra), "c", log=lambda _: None, repo_root=Path("/repo")
            )

    def test_happy_path_image_mode(self, make_infra, monkeypatch) -> None:
        fake = _FakeSubmitter()
        monkeypatch.setattr(orchestrate, "build_submitter", lambda infra: fake)
        monkeypatch.setattr(orchestrate, "save_handle", lambda h: None)
        infra = make_infra(
            launch={"codeSource": "image", "codePath": "/opt/dockyard_rl",
                    "entrypoint": "python3 run.py --config a.yaml", "env": {"FOO": "bar"}},
        )
        result = orchestrate.submit_training(
            _loaded(infra), "dockyard-grpo", log=lambda _: None, repo_root=Path("/repo"),
            run_id="run-xyz",
        )
        assert result.training_job_id == "run-xyz"
        call = fake.calls[0]
        assert call["cluster_name"] == "dockyard-grpo"
        assert call["working_dir"] is None  # image mode: no staging
        assert call["env_vars"]["DOCKYARD_K8S_RUN_ID"] == "run-xyz"
        assert call["env_vars"]["FOO"] == "bar"


class TestRunDispatch:
    def test_rayjob_mode_rejected(self, make_infra) -> None:
        infra = make_infra(launch={"mode": "rayjob", "entrypoint": "x"})
        with pytest.raises(ValueError, match="rayjob mode"):
            orchestrate.run(_loaded(infra), log=lambda _: None, repo_root=Path("/repo"))

    def test_attach_resolves_cluster_name(self, make_infra, monkeypatch) -> None:
        seen = {}
        monkeypatch.setattr(orchestrate, "ensure_sandbox", lambda loaded, *, log: None)

        def _fake_submit(loaded, cluster_name, *, log, repo_root, replace=False, run_id=None):
            seen["cluster_name"] = cluster_name
            return orchestrate.RunResult(
                handle=SubmissionHandle(kind="ray", run_id="r", cluster_name=cluster_name, namespace="ns")
            )

        monkeypatch.setattr(orchestrate, "submit_training", _fake_submit)
        infra = make_infra(launch={"mode": "attach", "attach": "live-cluster", "entrypoint": "x"})
        orchestrate.run(_loaded(infra), log=lambda _: None, repo_root=Path("/repo"))
        assert seen["cluster_name"] == "live-cluster"
