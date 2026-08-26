"""#3178: ray.sub could register topo_rank 0, which Ray silently drops.

Two defects in the same three lines, in both the head and worker fragments.

Value-0 drop: Ray does not register a custom resource whose value is 0. The
reader (virtual_cluster._parse_topology_from_resources) then returns
TOPO_RANK_UNKNOWN (-1), which sorts *first* -- and nvlink_domain, being 1, still
registers, so select_segment_nodes does not skip the node. A rank-0 node is
therefore promoted to lowest rank rather than merely losing its order, and it
drags its whole domain to the front of the domain sort.

Collision: the fallback chain read SLURM_PROCID, but ray.sub launches the head
and each worker as separate `--nodes=1 --ntasks=1` srun steps, so PROCID is 0 in
every one of them. Without a topology address, every node in the job resolved to
the same rank -- and, being 0, to no resource at all.

These tests execute the fragments as bash exactly as ray.sub emits them, rather
than asserting on the text, so they fail on any expression that can yield 0 or
collide -- not only on the two that did.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

RAY_SUB = Path(__file__).resolve().parents[2] / "ray.sub"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


def _fragments() -> list[str]:
    """The two TOPO_RANK assignment blocks, un-escaped as the heredoc emits them.

    Sliced by block boundary -- from the first TOPO_RANK assignment to the
    RAY_TOPOLOGY_RESOURCES line that consumes it -- rather than matched against
    a specific expression, so these tests keep exercising the block through any
    rewrite of it instead of erroring out when the syntax changes.

    ray.sub writes these inside an unquoted heredoc, so `\\$` becomes `$` in the
    generated script while a bare `$i` is expanded by the launcher first.
    """
    text = RAY_SUB.read_text()
    blocks = re.findall(
        r"\n( *TOPO_RANK=.*?)\n *RAY_TOPOLOGY_RESOURCES=\", ", text, re.S
    )
    assert len(blocks) == 2, f"expected head + worker fragments, found {len(blocks)}"
    return [b.replace("\\$", "$") for b in blocks]


def _run(fragment: str, env: dict[str, str], node_index: int | None = None) -> int:
    """Evaluate one fragment and return the topo_rank it would register."""
    if node_index is not None:
        # The launcher expands $i before writing the worker script.
        fragment = fragment.replace("${TOPO_RANK:-$i}", f"${{TOPO_RANK:-{node_index}}}")
    script = "set -u\n" + fragment + '\necho "$TOPO_RANK"\n'
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, check=True
    )
    return int(out.stdout.strip())


@pytest.fixture
def head_fragment() -> str:
    return _fragments()[0]


@pytest.fixture
def worker_fragment() -> str:
    return _fragments()[1]


class TestNeverEmitsZero:
    """A zero is not a low rank; it is no resource at all."""

    def test_head_without_a_topology_address(self, head_fragment):
        assert _run(head_fragment, {"PATH": "/usr/bin:/bin"}) > 0

    @pytest.mark.parametrize("i", [1, 2, 5])
    def test_worker_without_a_topology_address(self, worker_fragment, i):
        assert _run(worker_fragment, {"PATH": "/usr/bin:/bin"}, node_index=i) > 0

    def test_head_with_a_zero_topology_address(self, head_fragment):
        """SLURM_TOPOLOGY_ADDR ending in 0 was the other route to a dropped rank."""
        env = {"PATH": "/usr/bin:/bin", "SLURM_TOPOLOGY_ADDR": "rack1.node0"}
        assert _run(head_fragment, env) > 0

    def test_worker_with_a_zero_topology_address(self, worker_fragment):
        env = {"PATH": "/usr/bin:/bin", "SLURM_TOPOLOGY_ADDR": "rack1.node0"}
        assert _run(worker_fragment, env, node_index=3) > 0

    def test_procid_zero_does_not_produce_rank_zero(self, worker_fragment):
        """PROCID is 0 in every srun step here, so it must not reach the rank."""
        env = {"PATH": "/usr/bin:/bin", "SLURM_PROCID": "0"}
        assert _run(worker_fragment, env, node_index=2) > 0


class TestRanksAreUniqueAcrossTheJob:
    def test_head_and_workers_do_not_collide_without_a_topology_address(
        self, head_fragment, worker_fragment
    ):
        env = {"PATH": "/usr/bin:/bin", "SLURM_PROCID": "0"}
        head = _run(head_fragment, env)
        workers = [_run(worker_fragment, env, node_index=i) for i in range(1, 6)]
        ranks = [head, *workers]
        assert len(set(ranks)) == len(ranks), f"colliding topo_ranks: {ranks}"

    def test_head_sorts_before_every_worker(self, head_fragment, worker_fragment):
        env = {"PATH": "/usr/bin:/bin"}
        head = _run(head_fragment, env)
        assert all(
            head < _run(worker_fragment, env, node_index=i) for i in range(1, 6)
        )


class TestTopologyOrderingIsPreserved:
    """The shift must be monotonic; a fix that reorders nodes is not a fix."""

    def test_address_order_is_preserved(self, head_fragment):
        addrs = ["r1.node0", "r1.node1", "r1.node4", "r2.node11"]
        ranks = [
            _run(head_fragment, {"PATH": "/usr/bin:/bin", "SLURM_TOPOLOGY_ADDR": a})
            for a in addrs
        ]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)


class TestConsumerDamageFromADroppedRank:
    """Why zero matters: the reader turns a missing rank into a winning one."""

    def test_missing_rank_reads_back_as_minus_one(self):
        from dockyard_rl.distributed.virtual_cluster import (
            TOPO_RANK_UNKNOWN,
            _parse_topology_from_resources,
        )

        # What Ray stores when topo_rank was 0: the key is simply absent, while
        # nvlink_domain (value 1) survives, so the node is not filtered out.
        domain, rank = _parse_topology_from_resources({"nvlink_domain_ABC": 1.0})
        assert domain == "nvlink_domain_ABC"
        assert rank == TOPO_RANK_UNKNOWN == -1

    def test_a_dropped_rank_would_win_the_domain_sort(self):
        """-1 beats every real rank, so the affected domain is selected first."""
        from dockyard_rl.distributed.virtual_cluster import select_segment_nodes

        topo = {
            "good0": ("nvlink_domain_A", 1), "good1": ("nvlink_domain_A", 2),
            "drop0": ("nvlink_domain_B", -1), "drop1": ("nvlink_domain_B", 3),
        }
        selected, _ = select_segment_nodes(topo, segment_size=2, num_nodes=2)
        assert selected == ["drop0", "drop1"], (
            "a domain containing a dropped rank outranks a correctly ranked one; "
            "this is the corruption ray.sub must not be able to cause"
        )
