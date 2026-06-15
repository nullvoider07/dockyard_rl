from __future__ import annotations

import re

# Match the file paths a unified (git) diff touches. `diff --git a/X b/Y`
# carries both pre- and post-image paths; the +++/--- headers are the
# fallback for diffs without the `diff --git` line.
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)
_PLUS_RE     = re.compile(r"^\+\+\+ (?:b/)?(.+?)\s*$", re.MULTILINE)
_MINUS_RE    = re.compile(r"^--- (?:a/)?(.+?)\s*$", re.MULTILINE)


def parse_diff_paths(diff: str) -> set[str]:
    """Return the repo-relative paths a unified diff touches.

    Collects both pre- and post-image paths so renames, additions, and
    deletions are all captured. ``/dev/null`` (new/deleted file sentinel) is
    excluded.
    """
    if not diff:
        return set()
    paths: set[str] = set()
    for pre, post in _DIFF_GIT_RE.findall(diff):
        paths.add(pre)
        paths.add(post)
    for path in _PLUS_RE.findall(diff):
        if path != "/dev/null":
            paths.add(path)
    for path in _MINUS_RE.findall(diff):
        if path != "/dev/null":
            paths.add(path)
    return paths
