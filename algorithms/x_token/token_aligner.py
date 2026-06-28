"""Cross-tokenizer token alignment.

Aligns the student and teacher tokenizations of the same source text so the
teacher's per-token distribution can be projected onto the student's tokens. The
pipeline is: canonicalize each tokenizer's tokens to a common surface form
(this module's lower layer), run a Needleman-Wunsch DP over the canonicalized
strings to pair student spans with teacher spans (added in a later layer), then
dense-pad the per-sample pairs into an ``AlignmentBatch`` the loss consumes.

This file is built leaf-up. The current layer is canonicalization: the pure
string normalization the DP depends on (space/newline normalization, byte
fallbacks, mojibake/unicode repair, and byte-token re-merging). Constants here
are content-coupled to the BPE/SentencePiece tokenizers being aligned across.
"""

from __future__ import annotations

from typing import List, Tuple

# Visual byte representations used by some BPE tokenizers (especially for
# emojis / non-ASCII bytes), mapping the visual character back to its byte value.
VISUAL_BYTE_MAP = {
    "ð": 240,
    "Ɩ": 241,
    "Ɨ": 242,
    "Ƙ": 243,
    "ƙ": 244,
    "ƚ": 245,
    "ƛ": 246,
    "Ɯ": 247,
    "Ɲ": 248,
    "ƞ": 249,
    "Ɵ": 250,
    "Ơ": 251,
    "ơ": 252,
    "Ƣ": 253,
    "ƣ": 254,
    "Ƥ": 255,
    "Ł": 156,
    "ł": 157,
    "Ń": 158,
    "ń": 159,
    "ĺ": 149,
    "Ļ": 150,
    "ļ": 151,
    "Ľ": 152,
    "ľ": 153,
    "Ŀ": 154,
    "ŀ": 155,
    "Ĭ": 135,
    "ĭ": 136,
    "Į": 137,
    "į": 138,
    "İ": 139,
    "ı": 140,
    "Ĳ": 141,
    "ĳ": 142,
    "Ĵ": 143,
    "ĵ": 144,
    "Ķ": 145,
    "ķ": 146,
    "ĸ": 147,
    "Ĺ": 148,
    "ĥ": 128,
    "Ħ": 129,
    "ħ": 130,
    "Ĩ": 131,
    "ĩ": 132,
    "Ī": 133,
    "ī": 134,
    "Ģ": 162,
    "ģ": 163,
    "Ĝ": 28,
    "ĝ": 29,
    "Ğ": 30,
    "ğ": 31,
}

# Multi-token encoding artifacts (mojibake) where the broken byte sequence spans
# tokens. Each pattern is N input tokens -> exactly one replacement token;
# checked left-to-right, first match wins.
_MULTI_TOKEN_ARTIFACT_FIXES = [
    (["ĠâĪ", "ĳ"], ["Ġ∑"]),
    (["âĪ", "ĳ"], ["∑"]),
    (["ĠâĪ", "ı"], ["Ġ∏"]),
    (["âĪ", "ı"], ["∏"]),
    (["ĠâĪ", "Ĥ"], ["Ġ∂"]),
    (["âĪ", "Ĥ"], ["∂"]),
    (["ĠâĪ", "ĩ"], ["Ġ∇"]),
    (["âĪ", "ĩ"], ["∇"]),
    (["ĠâĪ", "ŀ"], ["Ġ∞"]),
    (["âĪ", "ŀ"], ["∞"]),
    (["ĠâĪ", "ļ"], ["Ġ√"]),
    (["âĪ", "ļ"], ["√"]),
    (["ĠâĪ", "«"], ["Ġ∫"]),
    (["âĪ", "«"], ["∫"]),
    (["Ġâī", "ł"], ["Ġ≠"]),
    (["âī", "ł"], ["≠"]),
    (["Ġä¸", "Ń"], ["Ġ中"]),
    (["ä¸", "Ń"], ["中"]),
    (["æĸ", "ĩ"], ["文"]),
    (["Ġæĸ", "ĩ"], ["Ġ文"]),
]

# Per-token canonicalizations applied after multi-token artifact fixes.
_UNICODE_FIXES = {
    "Ã±": "ñ",
    "Ã¡": "á",
    "Ã©": "é",
    "Ã­": "í",
    "Ã³": "ó",
    "Ãº": "ú",
    "Ã": "À",
    "Ã¢": "â",
    "Ã§": "ç",
    "Ã¨": "è",
    "Ã«": "ë",
    "Ã®": "î",
    "Ã´": "ô",
    "Ã¹": "ù",
    "Ã»": "û",
    "Ã¿": "ÿ",
    "ä¸Ń": "中",
    "æĸĩ": "文",
    "æĹ¥æľ¬": "日本",
    "èªŀ": "語",
    "ÐłÑĥÑģ": "Рус",
    "ÑģÐºÐ¸Ð¹": "ский",
    "Ø§ÙĦØ¹Ø±Ø¨ÙĬØ©": "العربية",
    "à¤¹": "ह",
    "à¤¿à¤Ĥ": "हिं",
    "à¤¦à¥Ģ": "दी",
    "âĪĳ": "∑",
    "âĪı": "∏",
    "âĪĤ": "∂",
    "âĪĩ": "∇",
    "âĪŀ": "∞",
    "âĪļ": "√",
    "âĪ«": "∫",
    "âīĪ": "≈",
    "âīł": "≠",
    "âī¤": "≤",
    "âī¥": "≥",
}

_SPECIAL_TOKEN_MAP = {
    "<|begin_of_text|>": "<bos>",
    "<bos>": "<bos>",
    "<pad>": "",
}


def canonical_token(token: str, *, enabled: bool = True) -> str:
    """Return a canonical surface form for a tokenizer token.

    Normalizes space prefixes (``Ġ``/``_``/``▁`` → ``Ġ``), newlines/whitespace,
    a few space-prefixed punctuation forms, SentencePiece byte fallbacks
    (``<0x20>`` → the byte char), mojibake (``_UNICODE_FIXES``), and special
    tokens (``_SPECIAL_TOKEN_MAP``). ``enabled=False`` returns the input
    unchanged so callers can gate canonicalization with a single flag.
    """
    if not enabled:
        return token
    if not token:
        return token

    # Normalize space prefixes.
    if token.startswith(" "):
        token = "Ġ" + token[1:]
    elif token.startswith("_"):
        token = "Ġ" + token[1:]
    elif token.startswith("▁"):
        token = "Ġ" + token[1:]

    # Newline and whitespace normalization.
    if token == "Ċ":
        token = "\n"
    elif token == "\\n":
        token = "\n"
    elif token == "ĉ":
        token = "\n"
    elif token == "Ġ\n":
        token = "\n"
    elif "Ċ" in token:
        token = token.replace("Ċ", "\n")
    elif "\\n" in token:
        token = token.replace("\\n", "\n")

    if token == "Ġ,":
        token = ","
    elif token == "Ġ.":
        token = "."
    elif token == "Ġ;":
        token = ";"
    elif token == "Ġ:":
        token = ":"

    # SentencePiece byte fallback like <0x20>.
    if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
        try:
            byte_val = int(token[3:5], 16)
            if 0 <= byte_val <= 255:
                return chr(byte_val)
        except ValueError:
            pass

    for broken, fixed in _UNICODE_FIXES.items():
        if broken in token:
            token = token.replace(broken, fixed)

    if token in _SPECIAL_TOKEN_MAP:
        return _SPECIAL_TOKEN_MAP[token]

    return token


def _canonicalize_sequence(
    seq: List[str],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Canonicalize every token in a sequence, including byte-merging.

    Returns ``(canon, canon_to_orig)``. ``canon_to_orig[k]`` is a half-open
    ``[orig_start, orig_end)`` range giving the original-token positions that
    canonical token ``k`` was built from. Ranges are non-overlapping, strictly
    increasing, and jointly cover ``range(len(seq))`` — so DP-output indices over
    ``canon`` can be remapped back to positions on the original input-id axis.
    """
    merged, ranges = _merge_encoding_artifacts(seq)
    canon = [canonical_token(t) for t in merged]
    return _merge_consecutive_bytes(canon, ranges)


def _merge_encoding_artifacts(
    tokens: List[str],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Merge known multi-token mojibake patterns into single tokens.

    Returns ``(merged, ranges)`` with one ``(orig_start, orig_end)`` entry per
    output token. Every ``_MULTI_TOKEN_ARTIFACT_FIXES`` entry rewrites to a
    single replacement token, so each merge contributes exactly one range
    covering the matched pattern.
    """
    if not tokens:
        return [], []
    result: List[str] = []
    ranges: List[Tuple[int, int]] = []
    i = 0
    while i < len(tokens):
        matched = False
        for pattern, replacement in _MULTI_TOKEN_ARTIFACT_FIXES:
            pl = len(pattern)
            if i + pl <= len(tokens) and tokens[i : i + pl] == pattern:
                # Every fix is N->1; the remap relies on that to attach the
                # matched original range to a single output token.
                assert len(replacement) == 1, (
                    "Multi-token artifact fix replacement must be a single "
                    f"token; got {replacement!r}"
                )
                result.extend(replacement)
                ranges.append((i, i + pl))
                i += pl
                matched = True
                break
        if not matched:
            result.append(tokens[i])
            ranges.append((i, i + 1))
            i += 1
    return result, ranges


def _get_byte_value(token_char: str) -> int | None:
    """Return the byte value (0..255) for a single character, or None."""
    if len(token_char) != 1:
        return None
    char_ord = ord(token_char)
    if char_ord < 256:
        return char_ord
    return VISUAL_BYTE_MAP.get(token_char)


def _merge_consecutive_bytes(
    tokens: List[str],
    in_ranges: List[Tuple[int, int]],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Merge consecutive byte-fallback tokens back into Unicode characters.

    Propagates ``in_ranges`` parallel to ``tokens``: when a byte buffer collapses
    to one character, its parallel range slice collapses to a single
    ``(start, end)``; otherwise ranges pass through unchanged.
    """
    if not tokens:
        return [], []
    assert len(tokens) == len(in_ranges), (
        f"tokens/ranges length mismatch: {len(tokens)} vs {len(in_ranges)}"
    )
    result: List[str] = []
    result_ranges: List[Tuple[int, int]] = []
    byte_buffer: List[str] = []
    byte_buffer_ranges: List[Tuple[int, int]] = []
    for token, rng in zip(tokens, in_ranges):
        clean = token.lstrip("Ġ")
        if not clean:
            all_bytes = False
        else:
            all_bytes = all(_get_byte_value(c) is not None for c in clean)
        if all_bytes:
            byte_buffer.append(token)
            byte_buffer_ranges.append(rng)
        else:
            if byte_buffer:
                merged, merged_ranges = _try_merge_byte_buffer(
                    byte_buffer, byte_buffer_ranges
                )
                result.extend(merged)
                result_ranges.extend(merged_ranges)
                byte_buffer = []
                byte_buffer_ranges = []
            result.append(token)
            result_ranges.append(rng)
    if byte_buffer:
        merged, merged_ranges = _try_merge_byte_buffer(byte_buffer, byte_buffer_ranges)
        result.extend(merged)
        result_ranges.extend(merged_ranges)
    return result, result_ranges


def _try_merge_byte_buffer(
    byte_tokens: List[str],
    byte_ranges: List[Tuple[int, int]],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Decode 2-4 buffered byte tokens as a single UTF-8 character.

    Returns the merged single-character token plus a single collapsed range
    covering the whole buffer, or the unchanged buffer + ranges when no merge is
    possible.
    """
    if not byte_tokens:
        return [], []
    if len(byte_tokens) == 1:
        token = byte_tokens[0]
        clean = token.lstrip("Ġ")
        if len(clean) <= 1:
            return byte_tokens, byte_ranges

    space_prefix = "Ġ" if byte_tokens[0].startswith("Ġ") else ""
    raw_bytes: List[int] = []
    for token in byte_tokens:
        clean = token.lstrip("Ġ")
        for c in clean:
            v = _get_byte_value(c)
            if v is None:
                return byte_tokens, byte_ranges
            raw_bytes.append(v)

    if len(raw_bytes) < 2 or len(raw_bytes) > 4:
        return byte_tokens, byte_ranges
    try:
        decoded = bytes(raw_bytes).decode("utf-8")
        if len(decoded) == 1 and ord(decoded) > 127:
            return (
                [space_prefix + decoded],
                [(byte_ranges[0][0], byte_ranges[-1][1])],
            )
        return byte_tokens, byte_ranges
    except UnicodeDecodeError:
        return byte_tokens, byte_ranges


def _strings_equal_flexible(s1: str, s2: str, ignore_leading_char_diff: bool) -> bool:
    """Compare two strings, optionally after canonicalization."""
    if not ignore_leading_char_diff:
        return s1 == s2
    return canonical_token(s1) == canonical_token(s2)
