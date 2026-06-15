"""Tests for dockyard_k8s.config — the four-layer loader and its helpers.

The recipe half of the loader defers to dockyard_rl when importable; these
tests isolate the infra half (merge order, override partition, resolvers,
defaults inheritance, namespace inference) by stubbing the recipe loader so
they pass regardless of whether dockyard_rl is on the path.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dockyard_k8s import config as config_mod
from dockyard_k8s.config import (
    _load_with_inheritance,
    _merge_infra,
    _partition_overrides,
    _register_resolvers,
    get_username,
    load_recipe_with_infra,
)


class TestGetUsername:
    def test_env_override_sanitised(self, monkeypatch) -> None:
        monkeypatch.setenv("DOCKYARD_K8S_USER", "Kartik_S.W")
        assert get_username() == "kartik-s-w"


class TestPartitionOverrides:
    def test_infra_prefix_routed_and_stripped(self) -> None:
        recipe, infra = _partition_overrides(
            ["infra.scheduler.queue=rl", "grpo.max_num_steps=10"]
        )
        assert recipe == ["grpo.max_num_steps=10"]
        assert infra == ["scheduler.queue=rl"]

    def test_append_marker_preserved(self) -> None:
        _recipe, infra = _partition_overrides(["+infra.labels.team=rl"])
        assert infra == ["+labels.team=rl"]

    def test_lookalike_recipe_key_not_misrouted(self) -> None:
        # Only the `infra.` prefix (or exact `infra`) routes to the infra layer;
        # a recipe key that merely starts with "infra" stays on the recipe side.
        recipe, infra = _partition_overrides(["infrastructure.size=1"])
        assert recipe == ["infrastructure.size=1"]
        assert infra == []


class TestResolvers:
    def test_user_mul_div_max(self, monkeypatch) -> None:
        monkeypatch.setenv("DOCKYARD_K8S_USER", "tester")
        _register_resolvers()
        cfg = OmegaConf.create(
            {"u": "${user:}", "m": "${mul:3,4}", "d": "${div:10,2}", "x": "${max:2,9}"}
        )
        assert cfg.u == "tester"
        assert cfg.m == 12
        assert cfg.d == 5
        assert cfg.x == 9


class TestInheritance:
    def test_defaults_chain_merges_parent_then_child(self, tmp_path) -> None:
        parent = tmp_path / "base.yaml"
        parent.write_text("a: 1\nb: 1\n")
        child = tmp_path / "child.yaml"
        child.write_text("defaults: base.yaml\nb: 2\nc: 3\n")
        cfg = _load_with_inheritance(child)
        assert cfg.a == 1  # inherited
        assert cfg.b == 2  # child overrides parent
        assert cfg.c == 3
        assert "defaults" not in cfg


class TestMergeInfra:
    def test_file_over_shipped_and_cli_over_file(self, tmp_path) -> None:
        infra_file = tmp_path / "infra.yaml"
        infra_file.write_text(
            "namespace: dockyard\n"
            "image: nullvoider/ubuntu-swe:latest\n"
            "scheduler: {kind: kai, queue: rl}\n"
        )
        recipe = OmegaConf.create({})  # split mode: no recipe-level infra
        merged = _merge_infra(
            recipe, infra_path=infra_file, overrides=["scheduler.queue=team-b"]
        )
        # shipped default placement survives, file sets namespace/image,
        # CLI override wins on queue.
        assert merged.namespace == "dockyard"
        assert merged.scheduler.kind == "kai"
        assert merged.scheduler.queue == "team-b"
        assert merged.workspace.mountPath == "/mnt/dockyard"  # from shipped defaults

    def test_strips_anchor_scratch_sections(self, tmp_path) -> None:
        infra_file = tmp_path / "infra.yaml"
        infra_file.write_text(
            "_anchors: {x: &a 1}\nnamespace: dockyard\nimage: img\n"
        )
        merged = _merge_infra(OmegaConf.create({}), infra_path=infra_file)
        assert "_anchors" not in merged


class TestLoadRecipeWithInfra:
    def _stub_recipe(self, monkeypatch, recipe: dict) -> None:
        monkeypatch.setattr(
            config_mod,
            "_load_recipe",
            lambda recipe_path, overrides: OmegaConf.create(recipe),
        )

    def test_split_mode(self, tmp_path, monkeypatch) -> None:
        self._stub_recipe(monkeypatch, {"grpo": {"max_num_steps": 5}})
        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text("grpo: {max_num_steps: 5}\n")
        infra_file = tmp_path / "i.yaml"
        infra_file.write_text("namespace: dockyard\nimage: img\n")

        loaded = load_recipe_with_infra(recipe_path, infra_path=infra_file)
        assert loaded.infra.namespace == "dockyard"
        assert loaded.infra.image == "img"
        assert "infra" not in loaded.recipe

    def test_bundled_mode_peels_infra(self, tmp_path, monkeypatch) -> None:
        self._stub_recipe(
            monkeypatch,
            {"grpo": {"x": 1}, "infra": {"namespace": "bundled-ns", "image": "img"}},
        )
        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text("x: 1\n")
        loaded = load_recipe_with_infra(recipe_path)
        assert loaded.infra.namespace == "bundled-ns"
        assert "infra" not in loaded.recipe  # peeled off the recipe

    def test_namespace_inferred_when_absent(self, tmp_path, monkeypatch) -> None:
        self._stub_recipe(monkeypatch, {})
        monkeypatch.setattr(config_mod, "_infer_kube_namespace", lambda: "inferred-ns")
        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text("{}\n")
        infra_file = tmp_path / "i.yaml"
        infra_file.write_text("image: img\n")  # namespace omitted
        loaded = load_recipe_with_infra(recipe_path, infra_path=infra_file)
        assert loaded.infra.namespace == "inferred-ns"

    def test_both_infra_sources_rejected(self, tmp_path, monkeypatch) -> None:
        self._stub_recipe(monkeypatch, {"infra": {"namespace": "ns", "image": "img"}})
        recipe_path = tmp_path / "r.yaml"
        recipe_path.write_text("{}\n")
        infra_file = tmp_path / "i.yaml"
        infra_file.write_text("namespace: dockyard\nimage: img\n")
        with pytest.raises(ValueError, match="choose one or the other"):
            load_recipe_with_infra(recipe_path, infra_path=infra_file)
