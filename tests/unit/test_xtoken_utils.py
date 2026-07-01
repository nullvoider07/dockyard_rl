"""CPU tests for cross-tokenizer distillation batch-grid + transport guards."""

import pytest
import torch

from dockyard_rl.algorithms.x_token.utils import (
    assert_teacher_student_batch_grid,
    assert_xtoken_ipc_node_local,
    pad_distillation_val_batch,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict


# -- batch-grid guard ---------------------------------------------------------

def test_batch_grid_accepts_tileable():
    assert_teacher_student_batch_grid(
        global_batch_size=32,
        student_gbs=32,
        teacher_gbs=32,
        student_dp=4,
        teacher_dp=2,
        student_mbs=2,
        teacher_mbs=4,
    )


def test_batch_grid_rejects_mismatched_gbs():
    with pytest.raises(AssertionError, match="must all match"):
        assert_teacher_student_batch_grid(
            global_batch_size=32, student_gbs=16, teacher_gbs=32,
            student_dp=4, teacher_dp=2, student_mbs=2, teacher_mbs=4,
        )


def test_batch_grid_rejects_non_divisible_dp():
    with pytest.raises(AssertionError, match="divisible by student"):
        assert_teacher_student_batch_grid(
            global_batch_size=30, student_gbs=30, teacher_gbs=30,
            student_dp=4, teacher_dp=2, student_mbs=1, teacher_mbs=1,
        )


def test_batch_grid_rejects_non_divisible_mbs():
    with pytest.raises(AssertionError, match="micro batch size"):
        assert_teacher_student_batch_grid(
            global_batch_size=32, student_gbs=32, teacher_gbs=32,
            student_dp=4, teacher_dp=2, student_mbs=3, teacher_mbs=4,
        )


# -- node-local IPC guard -----------------------------------------------------

def test_ipc_node_local_single_node_always_ok():
    # Single node: any config is safe (no cross-node IPC possible).
    assert_xtoken_ipc_node_local(
        num_nodes=1, gpus_per_node=8,
        student_tp=4, student_cp=1, teacher_tp=2, teacher_cp=1,
        student_dp=2, teacher_dp=4,
    )


def test_ipc_node_local_multinode_requires_matching_dp():
    with pytest.raises(AssertionError, match="share a data_parallel degree"):
        assert_xtoken_ipc_node_local(
            num_nodes=2, gpus_per_node=8,
            student_tp=2, student_cp=1, teacher_tp=2, teacher_cp=1,
            student_dp=4, teacher_dp=2,
        )


def test_ipc_node_local_multinode_requires_matching_group():
    with pytest.raises(AssertionError, match="same model-parallel group"):
        assert_xtoken_ipc_node_local(
            num_nodes=2, gpus_per_node=8,
            student_tp=2, student_cp=1, teacher_tp=4, teacher_cp=1,
            student_dp=4, teacher_dp=4,
        )


def test_ipc_node_local_multinode_group_must_fit_node():
    with pytest.raises(AssertionError, match="fit within one node"):
        assert_xtoken_ipc_node_local(
            num_nodes=2, gpus_per_node=4,
            student_tp=8, student_cp=1, teacher_tp=8, teacher_cp=1,
            student_dp=2, teacher_dp=2,
        )


def test_ipc_node_local_multinode_valid():
    # 2 nodes, group tp*cp=4 fits in 8-GPU node and divides it evenly.
    assert_xtoken_ipc_node_local(
        num_nodes=2, gpus_per_node=8,
        student_tp=4, student_cp=1, teacher_tp=4, teacher_cp=1,
        student_dp=4, teacher_dp=4,
    )


# -- validation-batch padding -------------------------------------------------

def test_pad_val_batch_pads_tensors_and_lists_and_zeros_mask():
    batch = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0]),
            "agent_ref": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        }
    )
    padded = pad_distillation_val_batch(batch, target_size=5)
    assert padded["input_ids"].shape == (5, 2)
    assert len(padded["agent_ref"]) == 5
    # Original rows keep mask 1; padded rows are zeroed.
    assert torch.equal(padded["sample_mask"], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
    # Padding repeats the last row.
    assert torch.equal(padded["input_ids"][3], torch.tensor([5, 6]))


def test_pad_val_batch_noop_when_equal():
    batch = BatchedDataDict(
        {"input_ids": torch.zeros(4, 2), "sample_mask": torch.ones(4)}
    )
    out = pad_distillation_val_batch(batch, target_size=4)
    assert out is batch


def test_pad_val_batch_rejects_shrink():
    batch = BatchedDataDict(
        {"input_ids": torch.zeros(4, 2), "sample_mask": torch.ones(4)}
    )
    with pytest.raises(AssertionError, match="must be >="):
        pad_distillation_val_batch(batch, target_size=2)
