"""Shared CPU-math utilities for cross-tokenizer distillation.

This module holds the single-rank, framework-pure pieces of the
cross-tokenizer distillation loss — the parts that carry no CUDA-IPC
transport and no live process group, so they validate on CPU:

- :class:`Fp32SparseMM` — FP32 sparse-dense matmul that ignores BF16
  autocast (no BF16 sparse-mm kernel exists).
- Chunk aggregation: :func:`chunk_log_prob_sums` / :func:`chunk_average_finalize`
  / :func:`chunk_average_log_probs` / :func:`valid_chunk_mask`. The
  partial/finalize split lets a caller insert a CP all-reduce between the local
  bucket-sum and the divide (mean is non-linear, so the reduce must precede it).
- Projection: :func:`parse_projection_file`, the
  :func:`get_sparse_projection_matrix` / :func:`get_topk_projection`
  process-local caches, :func:`slice_sparse_projection_rows`,
  :func:`project_student_to_teacher_vocab`, and :func:`build_exact_token_map`
  (cached common/uncommon partition).
- Top-k teacher-vocab selection: :func:`select_teacher_topk_indices`.
- :func:`alignment_from_flat_batch` rehydrates the flat ``alignment_*``
  data-dict keys into an :class:`AlignmentBatch`.

The TP/CP-collective wrappers (``group_all_reduce_sum_with_grad`` and the
CP-localization / IPC teacher-logit rebuild) are deferred to the M3 transport
layer; the two functions here that reduce across a TP/CP group import that
helper lazily inside the ``world > 1`` branch, so the single-rank CPU path
stays import-clean before those primitives land.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch.distributed.tensor import DTensor

from dockyard_rl.algorithms.x_token.token_aligner import AlignmentBatch


def alignment_from_flat_batch(data: Mapping[str, Any]) -> AlignmentBatch:
    """Rebuild :class:`AlignmentBatch` from the flat ``alignment_*`` keys.

    The field set is driven off :class:`AlignmentBatch` so the helper
    can't drift from the schema.
    """
    return AlignmentBatch(
        **{f.name: data[f"alignment_{f.name}"] for f in fields(AlignmentBatch)}
    )


class Fp32SparseMM(torch.autograd.Function):
    """FP32 ``M.t() @ dense`` (sparse-dense matmul) ignoring surrounding autocast.

    ``addmm_sparse_cuda`` has no BF16 kernel on either forward or backward.
    The worker wraps forward + loss + backward in ``autocast(BF16)``, so a
    plain ``with autocast(enabled=False):`` around the forward call is not
    enough — ``loss.backward()`` runs inside the outer autocast and the
    sparse-mm backward kernel is still dispatched as BF16. The
    ``custom_fwd(cast_inputs=torch.float32)`` / ``custom_bwd`` decorators
    are PyTorch's official escape: they force FP32 inputs on forward and
    run the backward as if autocast were disabled.

    autograd's builtin sparse-mm backward computes ``M @ grad_out``. The
    gradient w.r.t. the sparse argument isn't needed (the projection matrix
    is frozen), so it's returned as ``None``.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx: Any, sparse_M: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        ctx.sparse_M = sparse_M
        return torch.sparse.mm(sparse_M.t(), dense)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[None, torch.Tensor]:  # type: ignore[override]
        sparse_M = ctx.sparse_M
        # out = sparse_M.t() @ dense, so d/d_dense = sparse_M @ grad_out.
        grad_dense = torch.sparse.mm(sparse_M, grad_out)
        return None, grad_dense


def chunk_log_prob_sums(
    log_probs: torch.Tensor,
    chunk_id: torch.Tensor,
    max_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local bmm + bucket count, no division.

    Output is summable across CP; callers that need cross-rank chunks to
    aggregate correctly should ``group_all_reduce_sum_with_grad`` both tensors
    before :func:`chunk_average_finalize`. ``chunk_id == -1`` contributes to no
    bucket.
    """
    device = log_probs.device
    chunk_arange = torch.arange(max_chunks, device=device).view(1, 1, -1)
    chunk_mask = chunk_id.unsqueeze(-1) == chunk_arange
    chunk_mask_f = chunk_mask.transpose(1, 2).to(log_probs.dtype)
    chunk_sums = torch.bmm(chunk_mask_f, log_probs)  # [B, C, V]
    chunk_sizes = chunk_mask.sum(dim=1).float()  # [B, C]
    return chunk_sums, chunk_sizes


def chunk_average_finalize(
    chunk_sums: torch.Tensor,
    chunk_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide sums by sizes; ``eps`` guards empty buckets."""
    eps = 1e-10
    chunk_log_probs = chunk_sums / (chunk_sizes.unsqueeze(-1) + eps)
    return chunk_log_probs, chunk_sizes


def chunk_average_log_probs(
    log_probs: torch.Tensor,
    chunk_id: torch.Tensor,
    max_chunks: int,
    *,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ``log_probs`` over chunks defined by ``chunk_id``.

    Builds a one-hot chunk mask from ``chunk_id`` (``-1`` = no chunk), then
    ``bmm``-aggregates and divides by chunk sizes. When ``cp_group`` has world
    > 1, the per-chunk sums are ``group_all_reduce_sum_with_grad``'d across CP
    ranks before the divide (mean is non-linear, so the reduce must precede it).

    Args:
        log_probs: ``[B, T, V]`` log-probabilities.
        chunk_id: ``[B, T]`` long tensor, values in ``[-1, max_chunks)``.
        max_chunks: number of chunk buckets.
        cp_group: context-parallel group for cross-rank chunk aggregation.

    Returns:
        chunk_log_probs: ``[B, max_chunks, V]`` averaged log-probs.
        chunk_sizes: ``[B, max_chunks]`` float tensor of bucket sizes.
    """
    chunk_sums, chunk_sizes = chunk_log_prob_sums(log_probs, chunk_id, max_chunks)
    if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
        # CP/TP collective lands with the M3 transport layer; imported lazily
        # so the single-rank CPU path here stays import-clean until then.
        from dockyard_rl.distributed.model_utils import group_all_reduce_sum_with_grad

        chunk_sums = group_all_reduce_sum_with_grad(chunk_sums, cp_group)
        chunk_sizes = group_all_reduce_sum_with_grad(chunk_sizes, cp_group)
    return chunk_average_finalize(chunk_sums, chunk_sizes)


def slice_sparse_projection_rows(
    sparse_matrix: torch.Tensor,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    """Row-slice a sparse-COO projection ``[V_s, V_t]`` to ``[row_end-row_start, V_t]``.

    Filters COO indices: keeps entries with row in ``[row_start, row_end)``
    and shifts the row index by ``-row_start``. Used by the TP-aware P-KL path
    where each rank owns a contiguous slab of the student vocab axis.
    """
    indices = sparse_matrix.indices()
    values = sparse_matrix.values()
    mask = (indices[0] >= row_start) & (indices[0] < row_end)
    local_indices = indices[:, mask].clone()
    local_indices[0] -= row_start
    local_values = values[mask]
    return torch.sparse_coo_tensor(
        local_indices,
        local_values,
        (row_end - row_start, sparse_matrix.size(1)),
        device=sparse_matrix.device,
        dtype=sparse_matrix.dtype,
    ).coalesce()


# ---------------------------------------------------------------------------
# TP/CP-aware loss primitives
#
# Each collapses to the plain single-rank torch op when the relevant process
# group has world size 1, so the cross-tokenizer loss body stays free of any
# ``tp_world > 1`` / rank / offset branching.
# ---------------------------------------------------------------------------
def project_student_to_teacher_vocab(
    student_probs: torch.Tensor,
    sparse_projection: torch.Tensor,
    *,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Project student vocab probs ``[B, T, V_s(/TP)]`` to teacher vocab ``[B, T, V_t]``.

    ``sparse_projection`` is the full ``[V_s, V_t]`` sparse-COO matrix. With
    ``tp_group`` world > 1 the student probs cover only this rank's ``V_s/TP``
    rows, so the matrix is row-sliced to that range, the sparse matmul produces a
    partial teacher-vocab sum, and a ``group_all_reduce_sum_with_grad`` over the
    TP group combines the partials into the full ``V_s`` contraction. Otherwise a
    single sparse matmul over the full matrix is used.
    """
    batch_size, seq_len, local_vocab_size = student_probs.shape
    flat = student_probs.reshape(batch_size * seq_len, local_vocab_size)
    tp_world = torch.distributed.get_world_size(tp_group) if tp_group is not None else 1
    if tp_world > 1:
        # CP/TP collective lands with the M3 transport layer; imported lazily
        # so the single-rank CPU path here stays import-clean until then.
        from dockyard_rl.distributed.model_utils import group_all_reduce_sum_with_grad

        tp_rank = torch.distributed.get_rank(tp_group)
        full_student_vocab_size = sparse_projection.size(0)
        rows_per_rank = full_student_vocab_size // tp_world
        local_projection = slice_sparse_projection_rows(
            sparse_projection,
            row_start=tp_rank * rows_per_rank,
            row_end=(tp_rank + 1) * rows_per_rank,
        )
        projected_partial = Fp32SparseMM.apply(local_projection, flat.t()).t()
        projected = group_all_reduce_sum_with_grad(
            projected_partial.contiguous(), tp_group
        )
    else:
        # Fp32SparseMM internally computes M.t() @ dense; passing M (not M.t())
        # avoids a sparse ``.t()`` on a saved tensor in backward.
        projected = Fp32SparseMM.apply(sparse_projection, flat.t()).t()
    teacher_vocab_size = projected.shape[-1]
    return projected.reshape(batch_size, seq_len, teacher_vocab_size)


def select_teacher_topk_indices(
    teacher_logits: torch.Tensor,
    k: int,
    *,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Sorted global top-``k`` teacher-vocab ids by max importance over the microbatch.

    Importance is the per-vocab max over flattened ``(B*T)`` teacher logits. With
    ``cp_group`` world > 1 the sequence is CP-sharded, so the local max only sees
    this rank's slice; an ``all_reduce(MAX)`` makes every rank pick the same
    subset. No gradient.
    """
    vocab_size = teacher_logits.shape[-1]
    with torch.no_grad():
        # reshape (not view): a preceding next-token shift can leave the teacher
        # logits non-contiguous.
        teacher_flat = teacher_logits.reshape(-1, vocab_size)
        importance = teacher_flat.max(dim=0).values
        if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
            torch.distributed.all_reduce(
                importance, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )
        top_indices = torch.topk(importance, k=k, dim=-1).indices
        return top_indices.sort().values


def valid_chunk_mask(
    s_sizes: torch.Tensor,
    t_sizes: torch.Tensor,
    pair_valid: torch.Tensor,
) -> torch.Tensor:
    """Per-chunk validity gate: both sides non-empty and pair is valid."""
    return (s_sizes > 0) & (t_sizes > 0) & pair_valid


def parse_projection_file(
    path: Union[str, os.PathLike],
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Parse a projection-matrix file into COO components.

    Detects either the dense top-k format (``dict["indices"]`` /
    ``dict["likelihoods"]``) or the sparse multi-token format
    (``dict[(student_id, teacher_id)] -> count``) and converts both to
    a uniform COO representation.

    The function does **not** apply any sizing or validity policy: the
    ``-1`` sentinel used by ``_exact_map_remapped`` projection files is
    preserved in the returned ``indices``, and the inferred vocab sizes
    are derived from the file alone (caller may override them upward
    against tokenizer / config knowledge). This keeps a single parser
    while letting :mod:`token_aligner` and the loss fn keep their own
    clipping rules.

    Args:
        path: Path to a ``torch.save``d projection-matrix file.

    Returns:
        indices: ``LongTensor[2, nnz]`` — ``(student_idx, teacher_idx)``.
        values:  ``FloatTensor[nnz]``.
        v_student_inferred: ``int`` — dense format: row count; sparse
            format: ``max(student_idx) + 1``.
        v_teacher_inferred: ``int`` — ``max(positive teacher_idx) + 1``
            (``0`` if no positive entries exist).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: the file is not in a recognized format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Projection matrix file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict) and "indices" in data and "likelihoods" in data:
        # Dense top-k format: indices [V_s, top_k] holds teacher token ids;
        # likelihoods [V_s, top_k] holds the projection weights. Unfold to
        # COO so downstream code uses a uniform sparse-matmul path.
        top_indices: torch.Tensor = data["indices"].long()
        top_likelihoods: torch.Tensor = data["likelihoods"].float()
        if top_indices.shape != top_likelihoods.shape:
            raise ValueError(
                f"indices/likelihoods shape mismatch in {path}: "
                f"{top_indices.shape} vs {top_likelihoods.shape}"
            )
        v_student, top_k = top_indices.shape
        student_idx = torch.arange(v_student).unsqueeze(1).expand(-1, top_k).reshape(-1)
        teacher_idx = top_indices.reshape(-1)
        values = top_likelihoods.reshape(-1)
        indices = torch.stack([student_idx, teacher_idx], dim=0)
        positive = teacher_idx[teacher_idx >= 0]
        v_teacher = int(positive.max().item()) + 1 if positive.numel() > 0 else 0
        return indices, values, int(v_student), v_teacher

    if isinstance(data, dict) and all(
        isinstance(k, tuple) and len(k) == 2 for k in data.keys()
    ):
        # Sparse multi-token format: dict[(student_id, teacher_id)] -> count.
        keys = list(data.keys())
        values_list = list(data.values())
        student_idx = torch.tensor([k[0] for k in keys], dtype=torch.long)
        teacher_idx = torch.tensor([k[1] for k in keys], dtype=torch.long)
        indices = torch.stack([student_idx, teacher_idx], dim=0)
        values = torch.tensor(values_list, dtype=torch.float32)
        v_student = int(student_idx.max().item()) + 1 if student_idx.numel() > 0 else 0
        v_teacher = int(teacher_idx.max().item()) + 1 if teacher_idx.numel() > 0 else 0
        return indices, values, v_student, v_teacher

    raise ValueError(
        f"Unrecognized projection matrix format at {path}; expected dict "
        f"with 'indices'/'likelihoods' tensors or "
        f"dict[(student_id, teacher_id)] -> count."
    )


# Process-local projection-matrix caches. Each Ray worker / dataloader
# process has its own Python interpreter, so these dicts are effectively
# worker-local: a cache miss on one worker doesn't fill caches on other
# workers, and the driver process — which never enters a forward / loss
# path — never populates them.
#
# Keyed by ``(path, device, student_vocab_size, teacher_vocab_size)`` for
# the sparse cache because the sparse-COO shape's ``V_s`` and ``V_t`` are
# both sized from the configured vocab sizes; same path with a different
# size would build a different tensor. The top-k cache key is
# ``(path, device)`` — the raw top-k arrays don't depend on a vocab-size
# knob.
_SPARSE_PROJECTION_CACHE: dict[Tuple[str, torch.device, int, int], torch.Tensor] = {}
_TOPK_PROJECTION_CACHE: dict[
    Tuple[str, torch.device], Tuple[torch.Tensor, torch.Tensor]
] = {}


def get_sparse_projection_matrix(
    path: Union[str, os.PathLike],
    device: torch.device,
    *,
    student_vocab_size: int,
    teacher_vocab_size: int,
) -> torch.Tensor:
    """Return the sparse-COO projection matrix on ``device`` (cached).

    On a cache miss, parses the file via :func:`parse_projection_file`,
    drops ``-1`` teacher sentinels (illegal in sparse-COO), sizes
    ``V_s = max(student_vocab_size, max_observed_student_idx + 1)`` and
    ``V_t = max(teacher_vocab_size, max_observed_teacher_idx + 1)``, and
    builds a coalesced ``torch.sparse_coo_tensor`` on ``device``.
    Subsequent calls with the same
    ``(path, device, student_vocab_size, teacher_vocab_size)`` return the
    cached tensor — no disk I/O, no re-materialization.

    Both vocab sizes are keyword-only to prevent a positional swap (two
    same-magnitude ints, no error if confused).

    Args:
        path: Path to a ``torch.save``d projection-matrix file.
        device: Device the sparse tensor must live on.
        student_vocab_size: Minimum width of the student-side axis.
        teacher_vocab_size: Minimum width of the teacher-side axis.

    Returns:
        ``torch.sparse_coo_tensor`` of shape ``(V_s, V_t)``, coalesced,
        ``dtype=float32``.
    """
    key = (
        str(path),
        device,
        int(student_vocab_size),
        int(teacher_vocab_size),
    )
    cached = _SPARSE_PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    indices, values, _v_student, _ = parse_projection_file(path)
    # `_exact_map_remapped` projection files use -1 as a padding sentinel for
    # student rows with fewer than top_k teacher mappings. A negative column
    # index is illegal in a sparse tensor and causes a CUDA illegal-memory-access
    # in sparse.mm (forward and backward). Drop those entries entirely.
    keep = indices[1] >= 0
    indices = indices[:, keep]
    values = values[keep]
    # Size both axes from the configured tokenizer vocabs, not from the highest
    # ids observed in the projection file. The sparse format only stores entries
    # for (student_id, teacher_id) pairs that appeared during projection prep, so
    # the highest valid vocab ids may be absent. Sizing V_s from
    # `max(observed student_id)+1` would then make V_s < logits.shape[-1] and
    # silently break the sparse matmul; the symmetric concern on V_t lets the
    # P-KL global top-k gather go out of bounds. Clamp up against the projection's
    # observed max as a defensive fallback in case the file happens to cover ids
    # beyond the configured size.
    projection_max_student = (
        int(indices[0].max().item()) + 1 if indices.numel() > 0 else 0
    )
    projection_max_teacher = (
        int(indices[1].max().item()) + 1 if indices.numel() > 0 else 0
    )
    v_student = max(int(student_vocab_size), projection_max_student)
    v_teacher = max(int(teacher_vocab_size), projection_max_teacher)

    sparse = torch.sparse_coo_tensor(
        indices,
        values,
        (v_student, v_teacher),
        device=device,
        dtype=torch.float32,
    ).coalesce()
    _SPARSE_PROJECTION_CACHE[key] = sparse
    return sparse


def get_topk_projection(
    path: Union[str, os.PathLike],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the dense top-k ``(indices, likelihoods)`` projection on ``device`` (cached).

    Used by the gold-loss exact-map builder, which needs the per-row top-k
    weights — the sparse ``dict[(s, t)] -> count`` projection format doesn't
    carry those, so this loader rejects it.

    Args:
        path: Path to a ``torch.save``d projection-matrix file.
        device: Device the returned tensors must live on.

    Returns:
        ``(indices, likelihoods)`` — ``LongTensor[V_s, top_k]`` and
        ``FloatTensor[V_s, top_k]`` on ``device``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: the file is not in the dense top-k format.
    """
    key = (str(path), device)
    cached = _TOPK_PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    if not os.path.exists(path):
        raise FileNotFoundError(f"Projection matrix file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(data, dict) and "indices" in data and "likelihoods" in data):
        raise ValueError(
            f"gold_loss requires the dense projection-matrix format "
            f"(dict with 'indices' and 'likelihoods' tensors). File "
            f"{path} uses an unsupported format."
        )
    indices = data["indices"].long().to(device)
    likelihoods = data["likelihoods"].float().to(device)
    result = (indices, likelihoods)
    _TOPK_PROJECTION_CACHE[key] = result
    return result


# Process-local cache. Keyed by every input that affects the partition: the
# same file with a different ``xtoken_loss`` or ``teacher_vocab_size`` would
# yield a different partition. Lives alongside ``_TOPK_PROJECTION_CACHE`` so the
# gold-loss build is amortized to one pass per (path, device, knob) on each
# worker.
_EXACT_TOKEN_MAP_CACHE: dict[
    Tuple[str, torch.device, bool, int], Dict[str, torch.Tensor]
] = {}


def build_exact_token_map(
    path: Union[str, os.PathLike],
    device: torch.device,
    *,
    xtoken_loss: bool,
    teacher_vocab_size: int,
) -> Dict[str, torch.Tensor]:
    """Build the common/uncommon vocab partition for the gold path (cached).

    Reads the dense projection arrays via :func:`get_topk_projection`, sorts each
    student row's projection weights descending, then picks an exact-token
    map per the ``xtoken_loss`` flag:

    - ``xtoken_loss=False`` (strict): ``has_exact_map = (sorted_values[:, 0] == 1.0) & (projection_indices[:, 1] == -1)``.
      On collision (multiple students mapping to the same teacher id),
      the earliest (lowest) student index wins.
    - ``xtoken_loss=True`` (relaxed): ``has_exact_map = sorted_values[:, 0] >= 0.6``.
      On collision, the student with the highest first-projection
      weight wins; ties are broken by lowest student index.

    Both branches are vectorized via ``scatter_reduce`` so the build is
    O(V_s) and happens once per ``(path, device, xtoken_loss,
    teacher_vocab_size)`` for the run.

    Args:
        path: Path to a ``torch.save``d projection-matrix file (dense
            top-k format).
        device: Device the returned tensors must live on.
        xtoken_loss: Selects strict vs relaxed exact-map rule (see above).
        teacher_vocab_size: Width of the teacher-side vocab axis. The
            partition is bounded by this — teacher ids outside the range
            are dropped.

    Returns:
        Dict with keys ``common_student``, ``common_teacher`` (paired),
        ``uncommon_student``, ``uncommon_teacher`` (each independently
        sorted). All ``[long]`` tensors on ``device``.
    """
    key = (str(path), device, bool(xtoken_loss), int(teacher_vocab_size))
    cached = _EXACT_TOKEN_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    indices, likelihoods = get_topk_projection(path, device)
    v_student = indices.shape[0]
    v_teacher = int(teacher_vocab_size)

    sorted_values, sorted_in_topk = torch.sort(likelihoods, dim=-1, descending=True)
    if xtoken_loss:
        has_exact_map = sorted_values[:, 0] >= 0.6
    else:
        # Strict: exactly one top-k entry with weight 1.0, no second mapping.
        # `indices[:, 1] == -1` is the sentinel used by the
        # `_exact_map_remapped` projection files for "no second mapping".
        has_exact_map = (sorted_values[:, 0] == 1.0) & (indices[:, 1] == -1)

    # Gather (s_idx, t_idx, prob) for each exact-map candidate.
    s_candidates = torch.where(has_exact_map)[0]
    if s_candidates.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        result = {
            "common_student": empty,
            "common_teacher": empty,
            "uncommon_student": torch.arange(v_student, device=device),
            "uncommon_teacher": torch.arange(v_teacher, device=device),
        }
        _EXACT_TOKEN_MAP_CACHE[key] = result
        return result

    t_candidates = indices[s_candidates, sorted_in_topk[s_candidates, 0]]
    prob_candidates = sorted_values[s_candidates, 0]

    in_bounds = (t_candidates >= 0) & (t_candidates < v_teacher)
    s_vec = s_candidates[in_bounds]
    t_vec = t_candidates[in_bounds]
    prob_vec = prob_candidates[in_bounds]

    # Strict mode: any candidate is eligible (first one wins). Relaxed mode: only
    # candidates whose prob ties the per-teacher max.
    if xtoken_loss:
        max_prob_per_t = torch.full(
            (v_teacher,),
            float("-inf"),
            device=device,
            dtype=prob_vec.dtype,
        )
        max_prob_per_t.scatter_reduce_(
            0, t_vec, prob_vec, reduce="amax", include_self=True
        )
        eligible = prob_vec >= max_prob_per_t[t_vec]
    else:
        eligible = torch.ones_like(t_vec, dtype=torch.bool)

    # For each teacher id, pick the smallest student index among the eligible
    # candidates. Sentinel = v_student so non-eligible rows lose the amin reduce.
    sentinel = torch.tensor(v_student, dtype=s_vec.dtype, device=device)
    eligible_s = torch.where(eligible, s_vec, sentinel.expand_as(s_vec))
    min_s_per_t = torch.full((v_teacher,), v_student, device=device, dtype=s_vec.dtype)
    min_s_per_t.scatter_reduce_(0, t_vec, eligible_s, reduce="amin", include_self=True)
    winner_mask = eligible & (s_vec == min_s_per_t[t_vec])

    common_student = s_vec[winner_mask]
    common_teacher = t_vec[winner_mask]
    # Sort by student index so the paired arrays match.
    sort_perm = torch.argsort(common_student)
    common_student = common_student[sort_perm]
    common_teacher = common_teacher[sort_perm]

    common_s_mask = torch.zeros(v_student, dtype=torch.bool, device=device)
    common_s_mask[common_student] = True
    common_t_mask = torch.zeros(v_teacher, dtype=torch.bool, device=device)
    common_t_mask[common_teacher] = True
    uncommon_student = (~common_s_mask).nonzero(as_tuple=True)[0]
    uncommon_teacher = (~common_t_mask).nonzero(as_tuple=True)[0]

    result = {
        "common_student": common_student,
        "common_teacher": common_teacher,
        "uncommon_student": uncommon_student,
        "uncommon_teacher": uncommon_teacher,
    }
    _EXACT_TOKEN_MAP_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# CP-localized alignment + student-side CE / accuracy.
#
# The CP-collective helpers (vocab_parallel_argmax / cp_shift_next) and the
# DTensor unwrap (to_local_if_dtensor) are imported inside the functions: the
# single-rank / non-DTensor CPU path here stays import-clean, and the multi-rank
# behavior is GPU-deferred.
# ---------------------------------------------------------------------------
@dataclass
class LocalizedAlignment:
    """CP-localized chunk-alignment tensors consumed by the loss reductions."""

    student_chunk_id: torch.Tensor
    teacher_chunk_id: torch.Tensor
    pair_valid: torch.Tensor
    pair_is_correct: torch.Tensor
    sample_mask: torch.Tensor
    # Filled post-construction by the loss-input keystone (None when built via
    # localize_alignment); read only by the next-token-accuracy metric -- the
    # gold path leaves them unset.
    student_input_ids: Optional[torch.Tensor] = None
    student_token_mask: Optional[torch.Tensor] = None


def localize_alignment(
    data: Mapping[str, Any],
    *,
    teacher_seq_len: int,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> LocalizedAlignment:
    """Localize the chunk-alignment data-dict fields for the local CP shard.

    Unwraps the ``alignment_*`` / ``sample_mask`` entries from DTensor to their
    local tensors. The teacher-seq ``teacher_chunk_id`` is full, so it is sliced
    contiguously to this CP rank's ``teacher_seq_len`` window to match the
    transport consumer's contiguous teacher-logit slice. Student-seq fields stay
    as handed in; the caller (the loss-input keystone) relayouts them from the CP
    load-balanced layout to this rank's contiguous window before use.
    """
    from dockyard_rl.models.dtensor.parallelize import to_local_if_dtensor

    teacher_chunk_id_full = to_local_if_dtensor(data["alignment_teacher_chunk_id"])
    cp_rank = (
        torch.distributed.get_rank(cp_group)
        if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1
        else 0
    )
    teacher_seq_start = cp_rank * teacher_seq_len
    teacher_chunk_id = teacher_chunk_id_full[
        :, teacher_seq_start : teacher_seq_start + teacher_seq_len
    ]
    return LocalizedAlignment(
        student_chunk_id=to_local_if_dtensor(data["alignment_student_chunk_id"]),
        teacher_chunk_id=teacher_chunk_id,
        pair_valid=to_local_if_dtensor(data["alignment_pair_valid"]),
        pair_is_correct=to_local_if_dtensor(data["alignment_pair_is_correct"]),
        sample_mask=to_local_if_dtensor(data["sample_mask"]),
    )


def student_next_token_ce(
    logits: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    seq_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token next-token cross-entropy ``[B, T-1]`` on the student.

    DTensor (TP/CP) logits route through the vocab-parallel log-prob helper
    (which also handles the CP roll); plain logits use a local shifted
    ``cross_entropy``. The next-token shift (drop the last predictor) matches the
    convention the KL terms use.
    """
    if isinstance(logits, DTensor):
        from dockyard_rl.distributed.model_utils import (
            get_logprobs_from_vocab_parallel_logits,
        )

        next_token_logprobs = get_logprobs_from_vocab_parallel_logits(
            logits, input_ids, seq_index=seq_index
        )
        return -next_token_logprobs
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(shift_labels.shape)


def ce_label_mask(
    *,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    ce_seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Next-token label mask ``[B, ce_seq_len]`` = shifted token_mask * sample_mask.

    ``token_mask`` is gathered to the full sequence (CP) before the shift; both
    inputs are DTensor-unwrapped.
    """
    from dockyard_rl.models.dtensor.parallelize import to_local_if_dtensor

    token_mask = (
        token_mask.full_tensor() if isinstance(token_mask, DTensor) else token_mask
    )
    sample_mask = to_local_if_dtensor(sample_mask)
    return (token_mask[:, 1 : ce_seq_len + 1] * sample_mask.unsqueeze(-1)).to(dtype)


def next_token_accuracy(
    logits: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Masked next-token top-1 accuracy of the student (scalar, no gradient).

    Uses :func:`vocab_parallel_argmax` for the (possibly TP-sharded) argmax. The
    next-token shift on labels/mask is CP-aware (:func:`cp_shift_next`) so the
    boundary token crosses CP ranks, and the correct/total counts are CP-reduced
    so every rank reports the same global accuracy.
    """
    from dockyard_rl.distributed.model_utils import cp_shift_next, vocab_parallel_argmax
    from dockyard_rl.models.dtensor.parallelize import to_local_if_dtensor

    with torch.no_grad():
        argmax = vocab_parallel_argmax(logits, tp_group=tp_group)
        next_labels = cp_shift_next(to_local_if_dtensor(input_ids), cp_group, fill=0)
        next_mask = cp_shift_next(to_local_if_dtensor(token_mask), cp_group, fill=0)
        acc_mask = (
            next_mask.float() * to_local_if_dtensor(sample_mask).unsqueeze(-1).float()
        )
        correct = ((argmax == next_labels).float() * acc_mask).sum()
        denom = acc_mask.sum()
        if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
            stats = torch.stack([correct, denom])
            torch.distributed.all_reduce(stats, group=cp_group)
            correct, denom = stats[0], stats[1]
        return correct / denom.clamp(min=1.0)
