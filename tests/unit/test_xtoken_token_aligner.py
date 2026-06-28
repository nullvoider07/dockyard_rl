"""CPU tests for the xtoken canonicalization layer (M1.a).

The DP alignment core (M1.b) depends on these pure string transforms, so they
are built and tested first: token canonicalization, multi-token mojibake merges,
byte-value lookup, and byte-fallback re-merging into Unicode characters.
"""

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
