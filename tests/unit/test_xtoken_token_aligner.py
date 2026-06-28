"""CPU tests for the xtoken canonicalization layer (M1.a).

The DP alignment core (M1.b) depends on these pure string transforms, so they
are built and tested first: token canonicalization, multi-token mojibake merges,
byte-value lookup, and byte-fallback re-merging into Unicode characters.
"""

import torch

from dockyard_rl.algorithms.x_token import token_aligner as ta


# -- canonical_token ----------------------------------------------------------

def test_canonical_token_space_prefixes_unify_to_g():
    assert ta.canonical_token(" hello") == "Ġhello"
    assert ta.canonical_token("_hello") == "Ġhello"
    assert ta.canonical_token("▁hello") == "Ġhello"  # ▁


def test_canonical_token_newline_normalization():
    assert ta.canonical_token("Ċ") == "\n"
    assert ta.canonical_token("ĉ") == "\n"
    assert ta.canonical_token("Ġ\n") == "\n"
    assert ta.canonical_token("a\\nb") == "a\nb"  # embedded literal backslash-n


def test_canonical_token_space_punctuation():
    assert ta.canonical_token("Ġ,") == ","
    assert ta.canonical_token("Ġ.") == "."
    assert ta.canonical_token("Ġ;") == ";"
    assert ta.canonical_token("Ġ:") == ":"


def test_canonical_token_sentencepiece_byte_fallback():
    assert ta.canonical_token("<0x41>") == "A"  # 0x41 = 65 = 'A'
    assert ta.canonical_token("<0x20>") == " "  # space
    # Malformed hex falls through unchanged.
    assert ta.canonical_token("<0xZZ>") == "<0xZZ>"


def test_canonical_token_special_token_map():
    assert ta.canonical_token("<|begin_of_text|>") == "<bos>"
    assert ta.canonical_token("<bos>") == "<bos>"
    assert ta.canonical_token("<pad>") == ""


def test_canonical_token_disabled_and_empty():
    assert ta.canonical_token("anything", enabled=False) == "anything"
    assert ta.canonical_token("") == ""


# -- _merge_encoding_artifacts ------------------------------------------------

def test_merge_encoding_artifacts_collapses_known_pattern():
    merged, ranges = ta._merge_encoding_artifacts(["âĪ", "ĳ"])
    assert merged == ["∑"]
    assert ranges == [(0, 2)]


def test_merge_encoding_artifacts_passthrough_and_ranges():
    merged, ranges = ta._merge_encoding_artifacts(["a", "b", "c"])
    assert merged == ["a", "b", "c"]
    assert ranges == [(0, 1), (1, 2), (2, 3)]


def test_merge_encoding_artifacts_empty():
    assert ta._merge_encoding_artifacts([]) == ([], [])


# -- _get_byte_value ----------------------------------------------------------

def test_get_byte_value_ascii_visual_and_invalid():
    assert ta._get_byte_value("A") == 65
    assert ta._get_byte_value(chr(0xF0)) == 240  # latin-1 char < 256
    assert ta._get_byte_value("ð") == 240  # VISUAL_BYTE_MAP entry
    assert ta._get_byte_value("ab") is None  # multi-char
    assert ta._get_byte_value("中") is None  # ord > 256, not in map


# -- byte buffer re-merge -----------------------------------------------------

def test_try_merge_byte_buffer_decodes_utf8_char():
    # 0xC3 0xA9 -> 'é'. chr(195)='Ã', chr(169)='©'.
    merged, ranges = ta._try_merge_byte_buffer(
        [chr(195), chr(169)], [(0, 1), (1, 2)]
    )
    assert merged == ["é"]
    assert ranges == [(0, 2)]


def test_try_merge_byte_buffer_preserves_space_prefix():
    merged, _r = ta._try_merge_byte_buffer(
        ["Ġ" + chr(195), chr(169)], [(0, 1), (1, 2)]
    )
    assert merged == ["Ġé"]


def test_try_merge_byte_buffer_single_short_token_passthrough():
    merged, ranges = ta._try_merge_byte_buffer(["A"], [(0, 1)])
    assert merged == ["A"]
    assert ranges == [(0, 1)]


def test_try_merge_byte_buffer_invalid_utf8_passthrough():
    # 0xC3 alone is not a complete UTF-8 sequence -> unchanged.
    toks = [chr(195), chr(195)]
    merged, _r = ta._try_merge_byte_buffer(toks, [(0, 1), (1, 2)])
    assert merged == toks


def test_merge_consecutive_bytes_merges_run_and_propagates_ranges():
    merged, ranges = ta._merge_consecutive_bytes(
        [chr(195), chr(169)], [(0, 1), (1, 2)]
    )
    assert merged == ["é"]
    assert ranges == [(0, 2)]


def test_merge_consecutive_bytes_flushes_before_non_byte():
    # '中' is a non-byte token; the byte run after it merges independently.
    merged, ranges = ta._merge_consecutive_bytes(
        ["中", chr(195), chr(169)], [(0, 1), (1, 2), (2, 3)]
    )
    assert merged == ["中", "é"]
    assert ranges == [(0, 1), (1, 3)]


# -- _canonicalize_sequence (end-to-end) --------------------------------------

def test_canonicalize_sequence_covers_input_and_normalizes():
    seq = [" hello", "Ċ", "world"]
    canon, canon_to_orig = ta._canonicalize_sequence(seq)
    assert canon == ["Ġhello", "\n", "world"]
    # Ranges are non-overlapping, increasing, and jointly cover the input.
    flat = []
    prev_end = 0
    for s, e in canon_to_orig:
        assert s == prev_end
        flat.extend(range(s, e))
        prev_end = e
    assert flat == list(range(len(seq)))


def test_canonicalize_sequence_merges_artifact_then_byte():
    # Multi-token artifact collapses first; ranges still cover the 2 inputs.
    canon, canon_to_orig = ta._canonicalize_sequence(["âĪ", "ĳ"])
    assert canon == ["∑"]
    assert canon_to_orig == [(0, 2)]


# -- _strings_equal_flexible --------------------------------------------------

def test_strings_equal_flexible_exact_vs_canonical():
    assert ta._strings_equal_flexible("Ġ.", "Ġ.", ignore_leading_char_diff=False)
    assert not ta._strings_equal_flexible("Ġ.", ".", ignore_leading_char_diff=False)
    # Canonicalized: "Ġ." -> "." so they match under flexible compare.
    assert ta._strings_equal_flexible("Ġ.", ".", ignore_leading_char_diff=True)


# -- DP core (M1.b): AlignmentPair + _align_single ----------------------------

def _aligner():
    # Tokenizers are unused by _align_single / _align_dp (they take token lists).
    return ta.TokenAligner(None, None, projection_matrix_path="", max_comb_len=4)


def _as_tuples(pairs):
    return [
        (p.s_tokens, p.t_tokens, p.s_start, p.s_end, p.t_start, p.t_end, p.is_correct)
        for p in pairs
    ]


def test_align_single_identical_is_one_to_one_all_correct():
    al = _aligner()
    pairs = al._align_single(["a", "b", "c"], ["a", "b", "c"])
    assert _as_tuples(pairs) == [
        (["a"], ["a"], 0, 1, 0, 1, True),
        (["b"], ["b"], 1, 2, 1, 2, True),
        (["c"], ["c"], 2, 3, 2, 3, True),
    ]


def test_align_single_one_to_many_combination():
    al = _aligner()
    # Student 'abc' matches the teacher span 'ab'+'c'.
    pairs = al._align_single(["abc"], ["ab", "c"])
    assert len(pairs) == 1
    p = pairs[0]
    assert p.s_tokens == ["abc"] and p.t_tokens == ["ab", "c"]
    assert (p.s_start, p.s_end) == (0, 1)
    assert (p.t_start, p.t_end) == (0, 2)  # teacher span covers both tokens
    assert p.is_correct is True


def test_align_single_many_to_one_combination():
    al = _aligner()
    # Teacher 'abc' matches the student span 'ab'+'c'.
    pairs = al._align_single(["ab", "c"], ["abc"])
    assert len(pairs) == 1
    p = pairs[0]
    assert p.s_tokens == ["ab", "c"] and p.t_tokens == ["abc"]
    assert (p.s_start, p.s_end) == (0, 2)
    assert (p.t_start, p.t_end) == (0, 1)
    assert p.is_correct is True


def test_align_single_teacher_deletion_uses_sentinel():
    al = _aligner()
    # Teacher is missing 'b' -> a student-only pair with t_start/t_end == -1.
    pairs = al._align_single(["a", "b", "c"], ["a", "c"])
    by = {tuple(p.s_tokens): p for p in pairs}
    assert by[("a",)].is_correct and by[("c",)].is_correct
    delp = by[("b",)]
    assert delp.t_tokens == [] and delp.t_start == -1 and delp.t_end == -1
    assert delp.is_correct is False  # 'b' vs '' is not a match


def test_align_single_mismatch_marks_incorrect():
    al = _aligner()
    pairs = al._align_single(["a"], ["x"])
    assert len(pairs) == 1
    assert pairs[0].is_correct is False


def test_align_dp_score_and_traceback_identical():
    # Direct kernel check: identical sequences -> all-diag, score = N * match.
    pairs, score = ta.TokenAligner._align_dp(
        ["a", "b"],
        ["a", "b"],
        exact_match_score=3.0,
        combination_score_multiplier=1.5,
        gap_penalty=-1.5,
        max_combination_len=4,
        ignore_leading_char_diff=False,
    )
    assert score == 6.0  # 2 exact matches * 3.0
    assert all(len(p.s_tokens) == 1 and len(p.t_tokens) == 1 for p in pairs)
    assert [p.s_start for p in pairs] == [0, 1]


# -- M1.c: batch assembly + align() entry -------------------------------------

def test_pairs_to_batch_dense_tensors():
    pairs = [
        ta.AlignmentPair(["a"], ["a"], 0, 1, 0, 1, is_correct=True),
        ta.AlignmentPair(["b", "c"], ["bc"], 1, 3, 1, 2, is_correct=False),
    ]
    batch = ta.TokenAligner._pairs_to_batch([pairs], b=1, t_s=3, t_t=2)
    assert batch.student_chunk_id.tolist() == [[0, 1, 1]]
    assert batch.teacher_chunk_id.tolist() == [[0, 1]]
    assert batch.pair_valid.tolist() == [[True, True]]
    assert batch.pair_is_correct.tolist() == [[True, False]]
    # Only the 1-1 exact pair contributes to the gold partition.
    assert batch.student_exact_partition_mask.tolist() == [[True, False, False]]
    assert batch.teacher_exact_partition_mask.tolist() == [[True, False]]
    assert batch.num_chunks.tolist() == [2]


def test_pairs_to_batch_insertion_leaves_chunk_id_unset():
    # A teacher-only insertion (s_start == -1) writes no student chunk id.
    pairs = [ta.AlignmentPair([], ["x"], -1, -1, 0, 1, is_correct=False)]
    batch = ta.TokenAligner._pairs_to_batch([pairs], b=1, t_s=2, t_t=1)
    assert batch.student_chunk_id.tolist() == [[-1, -1]]
    assert batch.teacher_chunk_id.tolist() == [[0]]


def test_drop_padding_gates_chunk_ids_and_partition():
    pairs = [
        ta.AlignmentPair(["a"], ["a"], 0, 1, 0, 1, is_correct=True),
        ta.AlignmentPair(["b"], ["b"], 1, 2, 1, 2, is_correct=True),
        ta.AlignmentPair(["c"], ["c"], 2, 3, 2, 3, is_correct=True),
    ]
    batch = ta.TokenAligner._pairs_to_batch([pairs], b=1, t_s=3, t_t=3)
    # Mark the last student/teacher position as padding.
    s_mask = torch.tensor([[1, 1, 0]])
    t_mask = torch.tensor([[1, 1, 0]])
    ta.TokenAligner._drop_padding(
        batch, student_attention_mask=s_mask, teacher_attention_mask=t_mask
    )
    assert batch.student_chunk_id.tolist() == [[0, 1, -1]]
    assert batch.teacher_chunk_id.tolist() == [[0, 1, -1]]
    assert batch.student_exact_partition_mask.tolist() == [[True, True, False]]


class _FakeTok:
    """Minimal tokenizer: maps token ids to token strings for align()."""

    def __init__(self, id_to_token: dict[int, str]):
        self._m = id_to_token

    def convert_ids_to_tokens(self, ids):
        return [self._m[i] for i in ids]


def test_align_end_to_end_identical_tokenizers():
    vocab = {0: "a", 1: "b", 2: "c"}
    al = ta.TokenAligner(_FakeTok(vocab), _FakeTok(vocab), projection_matrix_path="")
    student_ids = torch.tensor([[0, 1, 2]])
    teacher_ids = torch.tensor([[0, 1, 2]])
    batch = al.align(student_ids, teacher_ids)
    assert batch.student_chunk_id.tolist() == [[0, 1, 2]]
    assert batch.teacher_chunk_id.tolist() == [[0, 1, 2]]
    assert batch.pair_is_correct.tolist() == [[True, True, True]]
    assert batch.student_exact_partition_mask.tolist() == [[True, True, True]]
    assert batch.num_chunks.tolist() == [3]


def test_align_end_to_end_respects_attention_mask():
    vocab = {0: "a", 1: "b", 2: "c"}
    al = ta.TokenAligner(_FakeTok(vocab), _FakeTok(vocab), projection_matrix_path="")
    student_ids = torch.tensor([[0, 1, 2]])
    teacher_ids = torch.tensor([[0, 1, 2]])
    batch = al.align(
        student_ids,
        teacher_ids,
        student_attention_mask=torch.tensor([[1, 1, 0]]),
        teacher_attention_mask=torch.tensor([[1, 1, 0]]),
    )
    # Padded last position is gated out of chunks + partition.
    assert batch.student_chunk_id.tolist() == [[0, 1, -1]]
    assert batch.student_exact_partition_mask.tolist() == [[True, True, False]]
