"""CPU tests for the CrossTokenizerCollator (M4.a).

Exercised with a char-level fake tokenizer (the collator + TokenAligner only use
a small tokenizer surface: __call__ -> input_ids/attention_mask,
convert_ids_to_tokens, padding_side, pad_token_id). Validates the key mapping,
the seq-divisibility padding, the sample_mask from loss_multiplier, and that the
flat alignment_* payload is carried through.
"""

import torch

from dockyard_rl.algorithms.x_token.token_aligner import TokenAligner
from dockyard_rl.data.cross_tokenizer_collate import CrossTokenizerCollator


class _FakeTokenizer:
    """Deterministic char-level tokenizer (id 0 = pad)."""

    def __init__(self, vocab_chars):
        self.id_to_char = {0: "<pad>"}
        self.char_to_id = {}
        for i, c in enumerate(vocab_chars, start=1):
            self.id_to_char[i] = c
            self.char_to_id[c] = i
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<pad>"
        self.padding_side = "right"

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        ids, masks = [], []
        for t in texts:
            row = [self.char_to_id.get(c, 0) for c in t][:max_length]
            mask = [1] * len(row)
            pad = max_length - len(row)
            ids.append(row + [self.pad_token_id] * pad)
            masks.append(mask + [0] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_char[int(i)] for i in ids]


def _datum(text, loss_multiplier, idx):
    return {
        "message_log": [{"role": "assistant", "content": text}],
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }


def _collator(ctx_s=8, ctx_t=8, div_s=1, div_t=1):
    vocab = list("abcde")
    tok = _FakeTokenizer(vocab)
    tok2 = _FakeTokenizer(vocab)
    aligner = TokenAligner(tok, tok2, projection_matrix_path="")
    return CrossTokenizerCollator(
        student_tokenizer=tok, teacher_tokenizer=tok2, aligner=aligner,
        ctx_length_student=ctx_s, ctx_length_teacher=ctx_t,
        make_seq_div_by_student=div_s, make_seq_div_by_teacher=div_t,
    )


def test_collate_produces_expected_keys_and_shapes():
    col = _collator()
    batch = [_datum("abc", 1.0, 0), _datum("ab", 1.0, 1)]
    out = col(batch)
    expected = {
        "input_ids", "input_lengths", "token_mask", "sample_mask",
        "teacher_input_ids", "teacher_input_lengths", "teacher_token_mask",
        "alignment_pair_valid", "alignment_pair_is_correct",
        "alignment_student_exact_partition_mask",
        "alignment_teacher_exact_partition_mask",
        "alignment_student_chunk_id", "alignment_teacher_chunk_id",
        "alignment_num_chunks", "idx",
    }
    assert expected <= set(out.keys())
    assert out["input_ids"].shape == (2, 8)
    assert out["token_mask"].shape == (2, 8)
    # input_lengths = real-token counts (3 for "abc", 2 for "ab").
    assert out["input_lengths"].tolist() == [3, 2]
    assert out["idx"] == [0, 1]


def test_collate_sample_mask_from_loss_multiplier():
    col = _collator()
    out = col([_datum("abc", 1.0, 0), _datum("ab", 0.0, 1)])
    assert out["sample_mask"].dtype == torch.float32
    assert out["sample_mask"].tolist() == [1.0, 0.0]


def test_collate_pads_student_seq_to_div_multiple():
    # ctx 6, divisor 8 -> student seq rounded up to 8; teacher divisor 4 -> 8.
    col = _collator(ctx_s=6, ctx_t=6, div_s=8, div_t=4)
    out = col([_datum("abc", 1.0, 0)])
    assert out["input_ids"].shape[1] == 8  # 6 rounded up to mult of 8
    assert out["teacher_input_ids"].shape[1] == 8  # 6 rounded up to mult of 4


def test_collate_alignment_matches_direct_align():
    col = _collator()
    out = col([_datum("abc", 1.0, 0)])
    # identical tokenizers + text => 1:1 chunks over the 3 real tokens, pad -1.
    assert out["alignment_student_chunk_id"][0, :3].tolist() == [0, 1, 2]
    assert (out["alignment_student_chunk_id"][0, 3:] == -1).all()
    # num_chunks counts ALL aligned pairs (3 real + 5 pad pairs); the pad chunks
    # are neutralized via chunk_id=-1 (valid_chunk_mask drops zero-size chunks).
    assert out["alignment_num_chunks"].tolist() == [8]


def test_collate_padding_positions_masked_to_no_chunk():
    col = _collator()
    out = col([_datum("ab", 1.0, 0)])  # 2 real tokens, 6 pad
    # pad positions (attention 0) forced to chunk_id -1 by align's _drop_padding.
    tok_mask = out["token_mask"][0]
    chunk = out["alignment_student_chunk_id"][0]
    assert (chunk[tok_mask == 0] == -1).all()
