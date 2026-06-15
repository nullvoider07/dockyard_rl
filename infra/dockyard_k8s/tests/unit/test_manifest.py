"""Tests for dockyard_k8s.manifest.

Covers cross-cutting patching of the RayCluster body (image, imagePullSecrets,
serviceAccount, managed-by label), the segmentSize worker-group split, and the
sandbox pool builders (synthesized default Deployment, inline override,
Service, DNS).
"""

from __future__ import annotations

import pytest

from src.dockyard_k8s.manifest import (
    _MANAGED_BY_LABEL,
    build_pdb,
    build_raycluster_manifest,
    build_sandbox_deployment,
    build_sandbox_network_policy,
    build_sandbox_service,
    build_security_contexts,
    sandbox_service_dns,
)
from src.dockyard_k8s.schema import (
    ClusterSpec,
    PDBSpec,
    PodResources,
    SandboxSpec,
    SecuritySpec,
)


def _all_pod_specs(rc: dict) -> list[dict]:
    specs = [rc["spec"]["headGroupSpec"]["template"]["spec"]]
    specs += [wg["template"]["spec"] for wg in rc["spec"]["workerGroupSpecs"]]
    return specs


class TestRayClusterPatching:
    def test_envelope(self, make_cluster, make_infra) -> None:
        rc = build_raycluster_manifest(make_cluster(), make_infra())
        assert rc["apiVersion"] == "ray.io/v1"
        assert rc["kind"] == "RayCluster"
        assert rc["metadata"]["name"] == "dockyard-grpo"
        assert rc["metadata"]["namespace"] == "dockyard"

    def test_image_patched_into_every_container(self, make_cluster, make_infra) -> None:
        rc = build_raycluster_manifest(make_cluster(), make_infra())
        for pod in _all_pod_specs(rc):
            for c in pod["containers"]:
                assert c["image"] == "nullvoider/ubuntu-swe:latest"

    def test_managed_by_label_present(self, make_cluster, make_infra) -> None:
        rc = build_raycluster_manifest(make_cluster(), make_infra())
        key, val = next(iter(_MANAGED_BY_LABEL.items()))
        assert rc["metadata"]["labels"][key] == val

    def test_service_account_patched(self, make_cluster, make_infra) -> None:
        rc = build_raycluster_manifest(
            make_cluster(), make_infra(serviceAccount="dockyard-rl")
        )
        for pod in _all_pod_specs(rc):
            assert pod["serviceAccountName"] == "dockyard-rl"

    def test_no_service_account_when_unset(self, make_cluster, make_infra) -> None:
        rc = build_raycluster_manifest(make_cluster(), make_infra())
        for pod in _all_pod_specs(rc):
            assert "serviceAccountName" not in pod

    def test_image_pull_secrets_fresh_list_per_pod(self, make_cluster, make_infra) -> None:
        # Regression: a shared list object aliases across pods and emits YAML
        # anchors that a later mutation could corrupt.
        rc = build_raycluster_manifest(
            make_cluster(), make_infra(imagePullSecrets=["regcred"])
        )
        pods = _all_pod_specs(rc)
        for pod in pods:
            assert pod["imagePullSecrets"] == [{"name": "regcred"}]
        ids = {id(pod["imagePullSecrets"]) for pod in pods}
        assert len(ids) == len(pods)  # no two pods share the same list object

    def test_input_spec_not_mutated(self, make_cluster, make_infra) -> None:
        cluster = make_cluster()
        before = cluster.spec["headGroupSpec"]["template"]["spec"]["containers"][0]
        assert "image" not in before
        build_raycluster_manifest(cluster, make_infra())
        # build works on a deepcopy; the caller's ClusterSpec is untouched.
        assert "image" not in before


class TestSegmentSize:
    def _one_group(self, replicas: int) -> dict:
        return {
            "workerGroupSpecs": [
                {
                    "groupName": "trainer",
                    "replicas": replicas,
                    "minReplicas": replicas,
                    "maxReplicas": replicas,
                    "template": {"spec": {"containers": [{"name": "ray-worker"}]}},
                }
            ]
        }

    def test_split_divisible(self, make_infra) -> None:
        cluster = ClusterSpec(name="c", spec=self._one_group(8), segmentSize=2)
        rc = build_raycluster_manifest(cluster, make_infra())
        groups = rc["spec"]["workerGroupSpecs"]
        assert [g["groupName"] for g in groups] == [
            "trainer-segment-0",
            "trainer-segment-1",
            "trainer-segment-2",
            "trainer-segment-3",
        ]
        assert all(g["replicas"] == 2 for g in groups)
        assert all(g["minReplicas"] == 2 and g["maxReplicas"] == 2 for g in groups)

    def test_group_at_or_below_segment_untouched(self, make_infra) -> None:
        cluster = ClusterSpec(name="c", spec=self._one_group(2), segmentSize=2)
        rc = build_raycluster_manifest(cluster, make_infra())
        assert [g["groupName"] for g in rc["spec"]["workerGroupSpecs"]] == ["trainer"]

    def test_non_divisible_raises(self, make_infra) -> None:
        cluster = ClusterSpec(name="c", spec=self._one_group(7), segmentSize=2)
        with pytest.raises(ValueError, match="not evenly divisible"):
            build_raycluster_manifest(cluster, make_infra())


class TestSandboxDeploymentDefault:
    def test_envelope_and_replicas(self, make_infra, sandbox) -> None:
        dep = build_sandbox_deployment(sandbox, make_infra())
        assert dep["apiVersion"] == "apps/v1"
        assert dep["kind"] == "Deployment"
        assert dep["metadata"]["name"] == "dockyard-sandbox"
        assert dep["spec"]["replicas"] == 3
        assert dep["spec"]["selector"]["matchLabels"] == {
            "app.kubernetes.io/name": "dockyard-sandbox"
        }

    def test_container_env_and_probes(self, make_infra, sandbox) -> None:
        dep = build_sandbox_deployment(sandbox, make_infra())
        container = dep["spec"]["template"]["spec"]["containers"][0]
        assert container["name"] == "task-executor"
        assert container["image"] == "nullvoider/ubuntu-swe:latest"
        env = {e["name"]: e["value"] for e in container["env"]}
        assert env["DOCKYARD_FLEET_ROLE"] == "sandbox"
        assert env["API_PORT"] == "9090"
        assert container["readinessProbe"]["tcpSocket"]["port"] == 9090
        assert container["livenessProbe"]["tcpSocket"]["port"] == 9090
        assert container["ports"][0]["containerPort"] == 9090

    def test_extra_env_merged(self, make_infra) -> None:
        sb = SandboxSpec(env={"TASK_MAX_AGE": "7200"})
        dep = build_sandbox_deployment(sb, make_infra())
        env = {e["name"]: e["value"] for e in dep["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert env["TASK_MAX_AGE"] == "7200"
        assert env["DOCKYARD_FLEET_ROLE"] == "sandbox"

    def test_resources_requests_equal_limits(self, make_infra) -> None:
        # No *Limit fields → limits default to requests (Guaranteed QoS).
        sb = SandboxSpec(resources=PodResources(cpu="8", memory="32Gi"))
        dep = build_sandbox_deployment(sb, make_infra())
        res = dep["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert res["requests"] == {"cpu": "8", "memory": "32Gi"}
        assert res["limits"] == {"cpu": "8", "memory": "32Gi"}

    def test_resources_limits_decoupled_from_requests(self, make_infra) -> None:
        # Modest requests (small scheduler reservation) + higher burst limits.
        sb = SandboxSpec(
            resources=PodResources(
                cpu="1",
                memory="2Gi",
                ephemeralStorage="10Gi",
                cpuLimit="4",
                memoryLimit="8Gi",
                ephemeralStorageLimit="20Gi",
            )
        )
        dep = build_sandbox_deployment(sb, make_infra())
        res = dep["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert res["requests"] == {
            "cpu": "1",
            "memory": "2Gi",
            "ephemeral-storage": "10Gi",
        }
        assert res["limits"] == {
            "cpu": "4",
            "memory": "8Gi",
            "ephemeral-storage": "20Gi",
        }

    def test_partial_limit_falls_back_to_request(self, make_infra) -> None:
        # Only cpuLimit raised; memory limit stays equal to its request.
        sb = SandboxSpec(resources=PodResources(cpu="1", memory="2Gi", cpuLimit="4"))
        dep = build_sandbox_deployment(sb, make_infra())
        res = dep["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert res["requests"] == {"cpu": "1", "memory": "2Gi"}
        assert res["limits"] == {"cpu": "4", "memory": "2Gi"}

    def test_pull_secrets_and_sa(self, make_infra, sandbox) -> None:
        dep = build_sandbox_deployment(
            sandbox, make_infra(imagePullSecrets=["regcred"], serviceAccount="dockyard-rl")
        )
        pod = dep["spec"]["template"]["spec"]
        assert pod["imagePullSecrets"] == [{"name": "regcred"}]
        assert pod["serviceAccountName"] == "dockyard-rl"


class TestSandboxDeploymentInline:
    def test_inline_spec_image_defaulted_and_labels_added(self, make_infra) -> None:
        sb = SandboxSpec(
            name="custom-sb",
            spec={
                "template": {
                    "spec": {
                        "containers": [{"name": "task-executor"}],
                        "securityContext": {"runAsUser": 1000},
                    }
                }
            },
        )
        dep = build_sandbox_deployment(sb, make_infra(imagePullSecrets=["regcred"]))
        pod = dep["spec"]["template"]["spec"]
        assert pod["containers"][0]["image"] == "nullvoider/ubuntu-swe:latest"
        assert pod["securityContext"] == {"runAsUser": 1000}  # preserved
        assert pod["imagePullSecrets"] == [{"name": "regcred"}]
        # selector + replicas synthesized when the inline spec omits them.
        assert dep["spec"]["selector"]["matchLabels"] == {
            "app.kubernetes.io/name": "custom-sb"
        }
        assert dep["spec"]["replicas"] == sb.replicas


class TestSandboxService:
    def test_clusterip_and_ports(self, make_infra, sandbox) -> None:
        svc = build_sandbox_service(sandbox, make_infra())
        assert svc["kind"] == "Service"
        assert svc["spec"]["type"] == "ClusterIP"
        assert svc["spec"]["selector"] == {"app.kubernetes.io/name": "dockyard-sandbox"}
        port = svc["spec"]["ports"][0]
        assert port["port"] == 9090 and port["targetPort"] == 9090

    def test_dns(self, make_infra, sandbox) -> None:
        url = sandbox_service_dns(sandbox, make_infra())
        assert url == "http://dockyard-sandbox.dockyard.svc.cluster.local:9090"


class TestSecurityContexts:
    def test_disabled_yields_empty(self) -> None:
        pod_sc, container_sc = build_security_contexts(SecuritySpec(enabled=False))
        assert pod_sc == {} and container_sc == {}

    def test_baseline_defaults(self) -> None:
        pod_sc, container_sc = build_security_contexts(SecuritySpec())
        # runAs*/readOnlyRootFilesystem are opt-in → absent by default.
        assert pod_sc == {"seccompProfile": {"type": "RuntimeDefault"}}
        assert container_sc == {
            "seccompProfile": {"type": "RuntimeDefault"},
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        }

    def test_opt_in_fields(self) -> None:
        pod_sc, container_sc = build_security_contexts(
            SecuritySpec(
                runAsNonRoot=True,
                runAsUser=1000,
                fsGroup=1000,
                readOnlyRootFilesystem=True,
                addCapabilities=["NET_BIND_SERVICE"],
            )
        )
        assert pod_sc["runAsNonRoot"] is True
        assert pod_sc["runAsUser"] == 1000 and pod_sc["fsGroup"] == 1000
        assert container_sc["runAsNonRoot"] is True
        assert container_sc["readOnlyRootFilesystem"] is True
        assert container_sc["capabilities"] == {
            "drop": ["ALL"],
            "add": ["NET_BIND_SERVICE"],
        }

    def test_sandbox_default_deployment_hardened(self, make_infra, sandbox) -> None:
        dep = build_sandbox_deployment(sandbox, make_infra())
        pod = dep["spec"]["template"]["spec"]
        container = pod["containers"][0]
        assert pod["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        assert container["securityContext"]["capabilities"] == {"drop": ["ALL"]}

    def test_sandbox_security_off_omits_context(self, make_infra) -> None:
        sb = SandboxSpec.model_validate({"security": {"enabled": False}})
        dep = build_sandbox_deployment(sb, make_infra())
        pod = dep["spec"]["template"]["spec"]
        assert "securityContext" not in pod
        assert "securityContext" not in pod["containers"][0]

    def test_gpu_security_opt_in(self, make_cluster, make_infra) -> None:
        # Default (off) leaves GPU pods untouched.
        rc = build_raycluster_manifest(make_cluster(), make_infra())
        for pod in _all_pod_specs(rc):
            assert "securityContext" not in pod
        # Enabled applies the baseline to head + worker pods/containers.
        infra = make_infra(security={"enabled": True})
        rc = build_raycluster_manifest(make_cluster(), infra)
        for pod in _all_pod_specs(rc):
            assert pod["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
            for c in pod["containers"]:
                assert c["securityContext"]["capabilities"] == {"drop": ["ALL"]}


class TestNetworkPolicy:
    def test_default_ingress_any_pod_in_namespace(self, make_infra, sandbox) -> None:
        np = build_sandbox_network_policy(sandbox, make_infra())
        assert np["apiVersion"] == "networking.k8s.io/v1"
        assert np["kind"] == "NetworkPolicy"
        assert np["metadata"]["name"] == "dockyard-sandbox-netpol"
        assert np["spec"]["podSelector"]["matchLabels"] == {
            "app.kubernetes.io/name": "dockyard-sandbox"
        }
        ingress = np["spec"]["ingress"][0]
        assert ingress["from"] == [{"podSelector": {}}]
        assert ingress["ports"] == [{"protocol": "TCP", "port": 9090}]
        # allowEgress default → unrestricted egress.
        assert np["spec"]["egress"] == [{}]

    def test_from_labels_restricts_ingress(self, make_infra) -> None:
        sb = SandboxSpec.model_validate(
            {"networkPolicy": {"fromLabels": {"app.kubernetes.io/managed-by": "dockyard-k8s"}}}
        )
        np = build_sandbox_network_policy(sb, make_infra())
        assert np["spec"]["ingress"][0]["from"] == [
            {"podSelector": {"matchLabels": {"app.kubernetes.io/managed-by": "dockyard-k8s"}}}
        ]

    def test_restricted_egress_keeps_dns_and_cidrs(self, make_infra) -> None:
        sb = SandboxSpec.model_validate(
            {"networkPolicy": {"allowEgress": False, "allowedEgressCidrs": ["10.0.0.0/8"]}}
        )
        np = build_sandbox_network_policy(sb, make_infra())
        egress = np["spec"]["egress"]
        dns_ports = egress[0]["ports"]
        assert {"protocol": "UDP", "port": 53} in dns_ports
        assert {"protocol": "TCP", "port": 53} in dns_ports
        assert egress[1]["to"] == [{"ipBlock": {"cidr": "10.0.0.0/8"}}]


class TestPodDisruptionBudget:
    def test_max_unavailable(self, make_infra) -> None:
        pdb = build_pdb(
            "dockyard-sandbox-pdb",
            "dockyard",
            {"app.kubernetes.io/name": "dockyard-sandbox"},
            PDBSpec(enabled=True, maxUnavailable=1),
            make_infra(),
        )
        assert pdb["apiVersion"] == "policy/v1"
        assert pdb["kind"] == "PodDisruptionBudget"
        assert pdb["spec"]["maxUnavailable"] == 1
        assert "minAvailable" not in pdb["spec"]
        assert pdb["spec"]["selector"]["matchLabels"] == {
            "app.kubernetes.io/name": "dockyard-sandbox"
        }

    def test_min_available(self, make_infra) -> None:
        pdb = build_pdb(
            "c-pdb", "dockyard", {"ray.io/cluster": "c"},
            PDBSpec(enabled=True, minAvailable=3), make_infra(),
        )
        assert pdb["spec"]["minAvailable"] == 3
        assert "maxUnavailable" not in pdb["spec"]

    def test_pdb_rejects_both_or_neither(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            PDBSpec(enabled=True)
        with pytest.raises(ValueError, match="exactly one of"):
            PDBSpec(enabled=True, minAvailable=1, maxUnavailable=1)


class TestEphemeralStorage:
    def test_ephemeral_storage_in_resources(self, make_infra) -> None:
        sb = SandboxSpec(resources=PodResources(cpu="8", ephemeralStorage="20Gi"))
        dep = build_sandbox_deployment(sb, make_infra())
        res = dep["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert res["requests"]["ephemeral-storage"] == "20Gi"
        assert res["limits"]["ephemeral-storage"] == "20Gi"


class TestPriorityClassAndSpread:
    def test_priority_class_on_gpu_pods(self, make_cluster, make_infra) -> None:
        rc = build_raycluster_manifest(
            make_cluster(), make_infra(priorityClassName="dockyard-high")
        )
        for pod in _all_pod_specs(rc):
            assert pod["priorityClassName"] == "dockyard-high"

    def test_priority_class_not_clobbering_inline(self, make_cluster, make_infra) -> None:
        cluster = make_cluster()
        # A worker group that pins its own priority keeps it.
        cluster.spec["workerGroupSpecs"][0]["template"]["spec"]["priorityClassName"] = "fleet-low"
        rc = build_raycluster_manifest(cluster, make_infra(priorityClassName="dockyard-high"))
        trainer = rc["spec"]["workerGroupSpecs"][0]["template"]["spec"]
        head = rc["spec"]["headGroupSpec"]["template"]["spec"]
        assert trainer["priorityClassName"] == "fleet-low"   # explicit wins
        assert head["priorityClassName"] == "dockyard-high"  # default fills in

    def test_priority_class_on_sandbox(self, make_infra, sandbox) -> None:
        dep = build_sandbox_deployment(sandbox, make_infra(priorityClassName="dockyard-high"))
        assert dep["spec"]["template"]["spec"]["priorityClassName"] == "dockyard-high"

    def test_topology_spread_on_sandbox(self, make_infra) -> None:
        tsc = [
            {
                "maxSkew": 1,
                "topologyKey": "topology.kubernetes.io/zone",
                "whenUnsatisfiable": "ScheduleAnyway",
                "labelSelector": {"matchLabels": {"app.kubernetes.io/name": "dockyard-sandbox"}},
            }
        ]
        sb = SandboxSpec.model_validate({"topologySpreadConstraints": tsc})
        dep = build_sandbox_deployment(sb, make_infra())
        assert dep["spec"]["template"]["spec"]["topologySpreadConstraints"] == tsc
