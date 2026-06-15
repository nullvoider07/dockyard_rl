"""Worker group utility functions.

Provides:
  recursive_merge_options      — Deep-merge Ray actor option dicts with
                                  correct precedence semantics.
  get_nsight_config_if_pattern_matches
                               — Optional Nsight Systems profiling config,
                                  activated by DOCKYARD_NSYS_* env vars.
"""

import fnmatch
import logging
import os
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)

# Nsight Systems profiling knobs
# Set both variables to enable Nsight profiling on matching workers.
#
#   DOCKYARD_NSYS_WORKER_PATTERNS  — Comma-separated fnmatch patterns matched
#                                    against worker names.  Glob wildcards ok.
#                                    Example: "dockyard-inference-*,dockyard-trainer-*"
#   DOCKYARD_NSYS_PROFILE_STEP_RANGE — Freeform step-range label embedded in
#                                    the output filename.
#                                    Example: "steps10-20"
#
# If only one is set, a startup assertion will fire to prevent silent
# misconfiguration.
DOCKYARD_NSYS_WORKER_PATTERNS: str = os.environ.get(
    "DOCKYARD_NSYS_WORKER_PATTERNS", ""
)
DOCKYARD_NSYS_PROFILE_STEP_RANGE: str = os.environ.get(
    "DOCKYARD_NSYS_PROFILE_STEP_RANGE", ""
)

def get_nsight_config_if_pattern_matches(
    worker_name: str,
) -> dict[str, Any]:
    """Return an Nsight Systems runtime_env config for matching workers.

    If DOCKYARD_NSYS_WORKER_PATTERNS is set, checks whether worker_name
    matches any comma-separated fnmatch pattern.  On a match, returns a
    dict ``{"nsight": {...}}`` ready to merge into a Ray runtime_env.

    Both DOCKYARD_NSYS_WORKER_PATTERNS and DOCKYARD_NSYS_PROFILE_STEP_RANGE
    must be set together or neither.

    Args:
        worker_name: The Ray actor name to match against patterns.

    Returns:
        ``{"nsight": <config>}`` on a match, otherwise ``{}``.
    """
    both_set = bool(DOCKYARD_NSYS_WORKER_PATTERNS) and bool(
        DOCKYARD_NSYS_PROFILE_STEP_RANGE
    )
    neither_set = not DOCKYARD_NSYS_WORKER_PATTERNS and not DOCKYARD_NSYS_PROFILE_STEP_RANGE
    assert both_set or neither_set, (
        "Either both DOCKYARD_NSYS_WORKER_PATTERNS and "
        "DOCKYARD_NSYS_PROFILE_STEP_RANGE must be set, or neither. "
        "Partial configuration will produce silent profiling failures."
    )

    if not DOCKYARD_NSYS_WORKER_PATTERNS:
        return {}

    patterns = [
        p.strip()
        for p in DOCKYARD_NSYS_WORKER_PATTERNS.split(",")
        if p.strip()
    ]

    for pattern in patterns:
        if fnmatch.fnmatch(worker_name, pattern):
            logger.info(
                "Nsight profiling enabled for worker %r (matched %r).",
                worker_name,
                pattern,
            )
            return {
                "nsight": {
                    "t": "cuda,cudnn,cublas,nvtx",
                    "o": f"'{worker_name}_{DOCKYARD_NSYS_PROFILE_STEP_RANGE}_%p'",
                    "stop-on-exit": "true",
                    # Profile is gated by torch.cuda.profiler.start()/stop()
                    # so we don't capture the entire process lifetime.
                    "capture-range": "cudaProfilerApi",
                    "capture-range-end": "stop",
                    "cuda-graph-trace": "node",
                }
            }

    return {}

def recursive_merge_options(
    default_options: dict[str, Any],
    extra_options: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge extra_options into default_options.

    ``extra_options`` takes precedence at every level.  Dict values are
    merged recursively; all other types are overwritten.

    Special case: if ``default_options["runtime_env"]`` contains an
    ``_nsight`` key (Ray's internal pending-transform key) but no ``nsight``
    key, the ``_nsight`` entry is promoted to ``nsight`` before merging so
    that ``extra_options`` can properly override it.

    Args:
        default_options: Lower-precedence option dict.
        extra_options:   Higher-precedence option dict.

    Returns:
        New merged dict (deep copies of both inputs; originals untouched).
    """
    base    = deepcopy(default_options)
    incoming = deepcopy(extra_options)

    # Promote pending _nsight transform in defaults before merge.
    if "runtime_env" in base and isinstance(base["runtime_env"], dict):
        re = base["runtime_env"]
        if "_nsight" in re and "nsight" not in re:
            re["nsight"] = re.pop("_nsight")

    def _merge(base_dict: dict, inc_dict: dict) -> None:
        for k, v in inc_dict.items():
            if (
                k in base_dict
                and isinstance(base_dict[k], dict)
                and isinstance(v, dict)
            ):
                _merge(base_dict[k], v)
            else:
                # All non-dict cases: scalar→dict, dict→scalar, scalar→scalar
                base_dict[k] = deepcopy(v)

    _merge(base, incoming)
    return base