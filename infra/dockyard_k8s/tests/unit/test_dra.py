"""Tests for the optional DRA (ComputeDomain + RoCE) auto-provisioning path.

Covers the schema knobs, the manifest builders / claim detection / name
rewrite, the render integration (and the autoCreate=false pass-through), the
k8s CRD lifecycle (mocked), and the orchestrate ensure/delete dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kubernetes.client.exceptions import ApiException
from omegaconf import OmegaConf
from pydantic import ValidationError

from dockyard_k8s import k8s, manifest, orchestrate
from dockyard_k8s.config import LoadedConfig
from dockyard_k8s.render import render_manifests
from dockyard_k8s.schema import DRASpec, InfraConfig


def _worker_with_claims() -> dict:
    return {
        "groupName": "trainer",
        "replicas": 1,
        "minReplicas": 1,
        "maxReplicas": 1,
        "template": {
            "spec": {
                "containers": [{"name": "ray-worker"}],
                "resourceClaims": [
                    {"name": "compute-domain-channel", "resourceClaimTemplateName": "X"},
                    {"name": "roce-channel", "resourceClaimTemplateName": "X"},
                ],
            }
        },
    }


def _spec_with_claims() -> dict:
    return {
        "headGroupSpec": {"template": {"spec": {"containers": [{"name": "ray-head"}]}}},
        "workerGroupSpecs": [_worker_with_claims()],
    }


def _loaded(infra) -> LoadedConfig:
    return LoadedConfig(recipe=OmegaConf.create({}), infra=infra, source_path=Path("r.yaml"))


class TestDRASpec:
    def test_defaults(self) -> None:
        d = DRASpec()
        assert d.autoCreate is True
        assert d.roceCount == 8
        assert d.computeDomainNumNodes == 0

    def test_default_on_infra(self) -> None:
        infra = InfraConfig.model_validate({"namespace": "ns", "image": "img"})
        assert infra.dra.autoCreate is True

    @pytest.mark.parametrize("field,bad", [("roceCount", 0), ("computeDomainNumNodes", -1)])
    def test_bounds(self, field: str, bad: int) -> None:
        with pytest.raises(ValidationError):
            DRASpec.model_validate({field: bad})

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            DRASpec.model_validate({"typo": 1})


class TestManifestBuilders:
    def test_detection_and_dedup(self) -> None:
        found = manifest.dra_resources_for_cluster("c", _spec_with_claims())
        assert found == [("compute-domain", "compute-domain-c"), ("roce", "roce-c")]

    def test_detection_none_when_no_claims(self) -> None:
        spec = {"workerGroupSpecs": [{"template": {"spec": {"containers": []}}}]}
        assert manifest.dra_resources_for_cluster("c", spec) == []

    def test_rewrite_template_names(self) -> None:
        spec = _spec_with_claims()
        manifest._rewrite_dra_template_names(spec, "dockyard-grpo")
        claims = spec["workerGroupSpecs"][0]["template"]["spec"]["resourceClaims"]
        names = {c["name"]: c["resourceClaimTemplateName"] for c in claims}
        assert names["compute-domain-channel"] == "compute-domain-dockyard-grpo"
        assert names["roce-channel"] == "roce-dockyard-grpo"

    def test_compute_domain_manifest(self) -> None:
        m = manifest.build_compute_domain_manifest("cd-c", "ns", num_nodes=0)
        assert m["apiVersion"] == "resource.nvidia.com/v1beta1"
        assert m["kind"] == "ComputeDomain"
        assert m["spec"]["numNodes"] == 0
        assert m["spec"]["channel"]["resourceClaimTemplate"]["name"] == "cd-c"

    def test_roce_manifest_count(self) -> None:
        m = manifest.build_roce_template_manifest("roce-c", "ns", count=8)
        assert m["kind"] == "ResourceClaimTemplate"
        req = m["spec"]["spec"]["devices"]["requests"][0]
        assert req["exactly"]["count"] == 8
        assert req["exactly"]["deviceClassName"] == "roce.networking.k8s.aws"


class TestRayClusterRewriteGate:
    def test_autocreate_rewrites(self, make_infra) -> None:
        from dockyard_k8s.schema import ClusterSpec

        infra = make_infra()  # dra.autoCreate defaults True
        rc = manifest.build_raycluster_manifest(
            ClusterSpec(name="dockyard-grpo", spec=_spec_with_claims()), infra
        )
        claims = rc["spec"]["workerGroupSpecs"][0]["template"]["spec"]["resourceClaims"]
        assert claims[0]["resourceClaimTemplateName"] == "compute-domain-dockyard-grpo"

    def test_autocreate_off_leaves_untouched(self, make_infra) -> None:
        from dockyard_k8s.schema import ClusterSpec

        infra = make_infra(dra={"autoCreate": False})
        rc = manifest.build_raycluster_manifest(
            ClusterSpec(name="dockyard-grpo", spec=_spec_with_claims()), infra
        )
        claims = rc["spec"]["workerGroupSpecs"][0]["template"]["spec"]["resourceClaims"]
        assert claims[0]["resourceClaimTemplateName"] == "X"  # original


class TestRenderIntegration:
    def test_dra_emitted_before_gpu_object(self, make_infra) -> None:
        infra = make_infra(
            kuberay={"name": "c", "spec": _spec_with_claims()},
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
            dra={"roceCount": 4, "computeDomainNumNodes": 2},
        )
        kinds = [m["kind"] for m in render_manifests(_loaded(infra))]
        assert kinds == ["ComputeDomain", "ResourceClaimTemplate", "RayJob"]

    def test_dra_honours_config(self, make_infra) -> None:
        infra = make_infra(
            kuberay={"name": "c", "spec": _spec_with_claims()},
            launch={"mode": "bringup"},
            dra={"roceCount": 4, "computeDomainNumNodes": 2},
        )
        docs = render_manifests(_loaded(infra))
        cd = [d for d in docs if d["kind"] == "ComputeDomain"][0]
        rct = [d for d in docs if d["kind"] == "ResourceClaimTemplate"][0]
        assert cd["spec"]["numNodes"] == 2
        assert rct["spec"]["spec"]["devices"]["requests"][0]["exactly"]["count"] == 4

    def test_autocreate_off_suppresses(self, make_infra) -> None:
        infra = make_infra(
            kuberay={"name": "c", "spec": _spec_with_claims()},
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
            dra={"autoCreate": False},
        )
        kinds = [m["kind"] for m in render_manifests(_loaded(infra))]
        assert kinds == ["RayJob"]

    def test_no_claims_no_dra(self, make_infra, make_cluster) -> None:
        infra = make_infra(
            kuberay=make_cluster().model_dump(),  # no resourceClaims
            launch={"mode": "rayjob", "entrypoint": "run.sh"},
        )
        kinds = [m["kind"] for m in render_manifests(_loaded(infra))]
        assert "ComputeDomain" not in kinds and "ResourceClaimTemplate" not in kinds


class _FakeCO:
    def __init__(self, *, create_raises=None):
        self.create_raises = create_raises
        self.created = self.deleted = False

    def create_namespaced_custom_object(self, **kw):
        if self.create_raises is not None:
            raise self.create_raises
        self.created = True
        return {"ok": True}

    def delete_namespaced_custom_object(self, **kw):
        self.deleted = True


class TestK8sLifecycle:
    def test_apply_compute_domain_409_noop(self, monkeypatch) -> None:
        fake = _FakeCO(create_raises=ApiException(status=409))
        monkeypatch.setattr(k8s, "custom_objects_api", lambda: fake)
        assert k8s.apply_compute_domain({"metadata": {"name": "cd"}}, "ns") == {}

    def test_apply_rct_creates(self, monkeypatch) -> None:
        fake = _FakeCO()
        monkeypatch.setattr(k8s, "custom_objects_api", lambda: fake)
        k8s.apply_resource_claim_template({"metadata": {"name": "r"}}, "ns")
        assert fake.created

    def test_delete_404_ignored(self, monkeypatch) -> None:
        class _CO404:
            def delete_namespaced_custom_object(self, **kw):
                raise ApiException(status=404)

        monkeypatch.setattr(k8s, "custom_objects_api", lambda: _CO404())
        k8s.delete_compute_domain("cd", "ns")  # no raise
        k8s.delete_resource_claim_template("r", "ns")  # no raise


class TestOrchestrateDispatch:
    def test_ensure_applies_both(self, make_infra, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(k8s, "apply_compute_domain", lambda m, ns: calls.append(("cd", m["metadata"]["name"])))
        monkeypatch.setattr(k8s, "apply_resource_claim_template", lambda m, ns: calls.append(("rct", m["metadata"]["name"])))
        infra = make_infra(kuberay={"name": "c", "spec": _spec_with_claims()})
        orchestrate.ensure_dra_resources(_loaded(infra), log=lambda _: None)
        assert calls == [("cd", "compute-domain-c"), ("rct", "roce-c")]

    def test_ensure_skips_when_autocreate_off(self, make_infra, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(k8s, "apply_compute_domain", lambda m, ns: calls.append(1))
        monkeypatch.setattr(k8s, "apply_resource_claim_template", lambda m, ns: calls.append(1))
        infra = make_infra(kuberay={"name": "c", "spec": _spec_with_claims()}, dra={"autoCreate": False})
        orchestrate.ensure_dra_resources(_loaded(infra), log=lambda _: None)
        assert calls == []

    def test_delete_reverses(self, make_infra, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(k8s, "delete_compute_domain", lambda n, ns: calls.append(("cd", n)))
        monkeypatch.setattr(k8s, "delete_resource_claim_template", lambda n, ns: calls.append(("rct", n)))
        infra = make_infra(kuberay={"name": "c", "spec": _spec_with_claims()})
        orchestrate.delete_dra_resources(_loaded(infra), log=lambda _: None)
        # reverse of [compute-domain, roce] → roce deleted first
        assert calls == [("rct", "roce-c"), ("cd", "compute-domain-c")]

    def test_ensure_skips_without_kuberay(self, make_infra, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(k8s, "apply_compute_domain", lambda m, ns: calls.append(1))
        orchestrate.ensure_dra_resources(_loaded(make_infra()), log=lambda _: None)
        assert calls == []
