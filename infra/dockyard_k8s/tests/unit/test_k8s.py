"""Tests for dockyard_k8s.k8s — create-or-replace, 404 handling, wait loops.

The CustomObjectsApi / typed-API boundary is faked so these run with no
cluster. They pin the conflict-becomes-patch behaviour, 404-tolerance, and
the readiness/terminal poll loops.
"""

from __future__ import annotations

import pytest
from kubernetes.client.exceptions import ApiException

from dockyard_k8s import k8s


class _FakeCO:
    """Fake CustomObjectsApi recording calls and raising scripted errors."""

    def __init__(self, *, create_raises=None, get_obj=None):
        self.create_raises = create_raises
        self.get_obj = get_obj
        self.created = self.patched = self.deleted = False

    def create_namespaced_custom_object(self, **kw):
        if self.create_raises is not None:
            raise self.create_raises
        self.created = True
        return {"metadata": {"name": kw["body"]["metadata"]["name"]}}

    def patch_namespaced_custom_object(self, **kw):
        self.patched = True
        return {"patched": True}

    def delete_namespaced_custom_object(self, **kw):
        self.deleted = True

    def get_namespaced_custom_object(self, **kw):
        if isinstance(self.get_obj, Exception):
            raise self.get_obj
        return self.get_obj


def _api409() -> ApiException:
    return ApiException(status=409, reason="conflict")


def _api404() -> ApiException:
    return ApiException(status=404, reason="not found")


class TestApplyRayCluster:
    def test_create_when_absent(self, monkeypatch) -> None:
        fake = _FakeCO()
        monkeypatch.setattr(k8s, "custom_objects_api", lambda: fake)
        out = k8s.apply_raycluster({"metadata": {"name": "c"}}, "ns")
        assert fake.created and not fake.patched
        assert out["metadata"]["name"] == "c"

    def test_conflict_patches(self, monkeypatch) -> None:
        fake = _FakeCO(create_raises=_api409())
        monkeypatch.setattr(k8s, "custom_objects_api", lambda: fake)
        out = k8s.apply_raycluster({"metadata": {"name": "c"}}, "ns")
        assert fake.patched
        assert out == {"patched": True}

    def test_other_error_attaches_redacted_manifest(self, monkeypatch) -> None:
        fake = _FakeCO(create_raises=ApiException(status=422, reason="bad"))
        monkeypatch.setattr(k8s, "custom_objects_api", lambda: fake)
        manifest = {"metadata": {"name": "c"}, "spec": {"x": 1}}
        with pytest.raises(ApiException) as ei:
            k8s.apply_raycluster(manifest, "ns")
        assert hasattr(ei.value, "dockyard_k8s_manifest")


class TestGetRayCluster:
    def test_404_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(k8s, "custom_objects_api", lambda: _FakeCO(get_obj=_api404()))
        assert k8s.get_raycluster("c", "ns") is None

    def test_returns_object(self, monkeypatch) -> None:
        obj = {"metadata": {"name": "c"}, "status": {"state": "ready"}}
        monkeypatch.setattr(k8s, "custom_objects_api", lambda: _FakeCO(get_obj=obj))
        assert k8s.get_raycluster("c", "ns") == obj


class TestWaitLoops:
    def test_ready_returns_immediately(self, monkeypatch) -> None:
        monkeypatch.setattr(
            k8s, "get_raycluster", lambda n, ns: {"status": {"state": "ready"}}
        )
        k8s.wait_for_raycluster_ready("c", "ns", timeout_s=5, poll_s=1)  # no raise

    def test_ready_times_out(self, monkeypatch) -> None:
        monkeypatch.setattr(
            k8s, "get_raycluster", lambda n, ns: {"status": {"state": "provisioning"}}
        )
        with pytest.raises(TimeoutError, match="never reached state=ready"):
            k8s.wait_for_raycluster_ready("c", "ns", timeout_s=0, poll_s=0)

    def test_gone_returns_when_absent(self, monkeypatch) -> None:
        monkeypatch.setattr(k8s, "get_raycluster", lambda n, ns: None)
        k8s.wait_for_raycluster_gone("c", "ns", timeout_s=5, poll_s=1)

    def test_rayjob_terminal_complete(self, monkeypatch) -> None:
        obj = {"status": {"jobDeploymentStatus": "Complete", "jobStatus": "SUCCEEDED"}}
        monkeypatch.setattr(k8s, "get_rayjob", lambda n, ns: obj)
        seen = []
        final = k8s.wait_for_rayjob_terminal(
            "j", "ns", timeout_s=5, poll_s=0, on_update=lambda d, j: seen.append((d, j))
        )
        assert final is obj
        assert seen == [("Complete", "SUCCEEDED")]

    def test_rayjob_disappeared_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(k8s, "get_rayjob", lambda n, ns: None)
        with pytest.raises(RuntimeError, match="disappeared"):
            k8s.wait_for_rayjob_terminal("j", "ns", timeout_s=5, poll_s=0)
