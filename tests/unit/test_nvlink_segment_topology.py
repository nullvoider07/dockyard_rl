"""CPU tests for the NVLink-domain segment-topology selection/sort logic.

The selection/sort functions are pure (operate on a topology dict / bundle list),
so they're fully unit-testable; the ray.nodes() discovery and the domain-pinned
placement-group scheduling are cluster-bound (HV-deferred).
"""

import pytest

from dockyard_rl.distributed.virtual_cluster import (
    NVLINK_DOMAIN_UNKNOWN,
    RayVirtualCluster,
    ResourceInsufficiencyError,
    TOPO_RANK_UNKNOWN,
    _parse_topology_from_resources,
    _sort_bundle_indices_by_topology,
    prepare_segment_topology,
    select_segment_nodes,
)


# -- resource parse -----------------------------------------------------------

def test_parse_reads_domain_and_topo_rank():
    res = {"nvlink_domain_ABC": 1.0, "topo_rank": 5.0, "GPU": 8.0, "CPU": 192.0}
    assert _parse_topology_from_resources(res) == ("nvlink_domain_ABC", 5)


def test_parse_returns_sentinels_when_absent():
    assert _parse_topology_from_resources({"GPU": 8.0}) == (
        NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN
    )


# -- select_segment_nodes -----------------------------------------------------

def test_select_takes_lowest_topo_domain_first():
    topo = {
        "n0": ("nvlink_domain_A", 0), "n1": ("nvlink_domain_A", 1),
        "n2": ("nvlink_domain_B", 2), "n3": ("nvlink_domain_B", 3),
    }
    selected, remaining = select_segment_nodes(topo, segment_size=2, num_nodes=2)
    assert selected == ["n0", "n1"]  # domain A has the lower min topo_rank
    assert set(remaining) == {"n2", "n3"}


def test_select_spans_domains_to_meet_num_nodes():
    topo = {
        "n0": ("nvlink_domain_A", 0), "n1": ("nvlink_domain_A", 1),
        "n2": ("nvlink_domain_B", 2), "n3": ("nvlink_domain_B", 3),
    }
    selected, remaining = select_segment_nodes(topo, segment_size=2, num_nodes=4)
    assert selected == ["n0", "n1", "n2", "n3"]
    assert remaining == []


def test_select_excludes_unknown_domain_nodes_2924():
    # The #2924 fix: unknown-domain nodes must not be selected (their pseudo
    # resource constraint is never registered, so the bundle can't schedule).
    topo = {
        "u": (NVLINK_DOMAIN_UNKNOWN, -1),
        "n0": ("nvlink_domain_A", 0), "n1": ("nvlink_domain_A", 1),
    }
    selected, remaining = select_segment_nodes(topo, segment_size=2, num_nodes=2)
    assert selected == ["n0", "n1"]
    assert remaining == ["u"]


def test_select_raises_on_indivisible_num_nodes():
    with pytest.raises(ValueError, match="divisible"):
        select_segment_nodes({"n0": ("nvlink_domain_A", 0)}, segment_size=2, num_nodes=3)


def test_select_raises_when_not_enough_complete_segments():
    # Only one domain with 2 nodes => one segment; can't form two.
    topo = {"n0": ("nvlink_domain_A", 0), "n1": ("nvlink_domain_A", 1)}
    with pytest.raises(ResourceInsufficiencyError, match="complete segments"):
        select_segment_nodes(topo, segment_size=2, num_nodes=4)


def test_select_incomplete_segment_in_a_domain_is_unusable():
    # Domain A has 3 nodes but segment_size=2 -> only 1 usable segment (2 nodes).
    topo = {
        "n0": ("nvlink_domain_A", 0), "n1": ("nvlink_domain_A", 1),
        "n2": ("nvlink_domain_A", 2),
    }
    selected, _ = select_segment_nodes(topo, segment_size=2, num_nodes=2)
    assert selected == ["n0", "n1"]


# -- prepare_segment_topology -------------------------------------------------

def test_prepare_none_segment_size_is_passthrough():
    assert prepare_segment_topology(None, 4) == (None, [], {})


def test_prepare_no_topology_info_falls_back():
    topo = {"n0": (NVLINK_DOMAIN_UNKNOWN, -1), "n1": (NVLINK_DOMAIN_UNKNOWN, -1)}
    constraints, remaining, returned = prepare_segment_topology(2, 2, topology=topo)
    assert constraints is None
    assert set(remaining) == {"n0", "n1"}
    assert returned == topo


def test_prepare_builds_per_node_domain_constraints():
    topo = {
        "n0": ("nvlink_domain_A", 0), "n1": ("nvlink_domain_A", 1),
        "n2": ("nvlink_domain_B", 2), "n3": ("nvlink_domain_B", 3),
    }
    constraints, remaining, _ = prepare_segment_topology(2, 2, topology=topo)
    assert constraints == [{"nvlink_domain_A": 0.001}, {"nvlink_domain_A": 0.001}]
    assert set(remaining) == {"n2", "n3"}


# -- _sort_bundle_indices_by_topology -----------------------------------------

def test_sort_falls_back_to_node_gpu_without_topology():
    bundle_data = [
        (1, NVLINK_DOMAIN_UNKNOWN, -1, "nodeB"),
        (0, NVLINK_DOMAIN_UNKNOWN, -1, "nodeA"),
        (0, NVLINK_DOMAIN_UNKNOWN, -1, "nodeB"),
    ]
    # sort by (node_id, gpu_id): nodeA/0 (idx1), nodeB/0 (idx2), nodeB/1 (idx0)
    assert _sort_bundle_indices_by_topology(bundle_data) == [1, 2, 0]


def test_sort_orders_by_domain_min_then_topo_then_gpu():
    bundle_data = [
        (0, "nvlink_domain_B", 2, "n2"),
        (0, "nvlink_domain_A", 0, "n0"),
        (0, "nvlink_domain_A", 1, "n1"),
        (0, "nvlink_domain_B", 3, "n3"),
    ]
    # domain A (min topo 0) first: idx1(topo0), idx2(topo1); then B: idx0, idx3
    assert _sort_bundle_indices_by_topology(bundle_data) == [1, 2, 0, 3]


def test_sort_segment_size_requires_gpus_per_node():
    with pytest.raises(ValueError, match="gpus_per_node"):
        _sort_bundle_indices_by_topology(
            [(0, "nvlink_domain_A", 0, "n0")], segment_size=2
        )


def test_sort_segment_size_drops_incomplete_domain_bundles():
    # Domain A: 2 nodes (1 gpu each) -> complete segment kept. Domain B: 1 node
    # -> can't form a segment_size=2 segment -> dropped.
    bundle_data = [
        (0, "nvlink_domain_A", 0, "n0"),
        (0, "nvlink_domain_A", 1, "n1"),
        (0, "nvlink_domain_B", 2, "n2"),
    ]
    order = _sort_bundle_indices_by_topology(
        bundle_data, segment_size=2, gpus_per_node=1
    )
    assert order == [0, 1]  # B's lone node discarded


# -- RayVirtualCluster wiring (constructor params + bundle-spec pinning) -------
# Constructing the cluster only stores params (no Ray); the placement-group
# scheduling + the topology actor reads are cluster-bound (HV-deferred).

def test_cluster_stores_segment_size_and_constraints():
    vc = RayVirtualCluster(
        [2, 2], segment_size=1,
        node_resource_constraints=[{"nvlink_domain_A": 0.001}, {"nvlink_domain_B": 0.001}],
    )
    assert vc._segment_size == 1
    assert len(vc._node_resource_constraints) == 2


def test_cluster_defaults_have_no_topology():
    vc = RayVirtualCluster([2])
    assert vc._segment_size is None
    assert vc._node_resource_constraints is None


def test_node_bundle_specs_merge_domain_pin():
    vc = RayVirtualCluster(
        [2, 2],
        node_resource_constraints=[{"nvlink_domain_A": 0.001}, {"nvlink_domain_B": 0.001}],
    )
    b0 = vc._node_bundle_specs(0, 2, 1.0, 1.0)
    assert b0 == [{"CPU": 1.0, "GPU": 1.0, "nvlink_domain_A": 0.001}] * 2
    b1 = vc._node_bundle_specs(1, 2, 1.0, 1.0)
    assert b1[0]["nvlink_domain_B"] == 0.001
    # Each bundle is an independent dict (no shared mutable state).
    b0[0]["CPU"] = 99.0
    assert b0[1]["CPU"] == 1.0


def test_node_bundle_specs_plain_without_constraints():
    vc = RayVirtualCluster([2])
    assert vc._node_bundle_specs(0, 2, 1.0, 1.0) == [{"CPU": 1.0, "GPU": 1.0}] * 2


def test_cluster_rejects_mismatched_constraint_length():
    with pytest.raises(AssertionError, match="one entry per logical node"):
        RayVirtualCluster([2, 2], node_resource_constraints=[{"nvlink_domain_A": 0.001}])
