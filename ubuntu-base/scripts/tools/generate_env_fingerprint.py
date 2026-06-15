#!/usr/bin/env python3
"""Generate an environment fingerprint for the SWE RL container.

Hashes the installed Python package set (pip freeze) and key binary
versions (Rust, Go, Node, Java) baked into the image. Written to
/etc/rl-env-fingerprint at build time. Checked at container start
to catch workspace/image version drift early.

Uses only Python stdlib — no external packages.

Usage (Dockerfile):
    RUN python3.14 /usr/local/lib/generate-env-fingerprint.py > /etc/rl-env-fingerprint
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone


def run(cmd: list[str]) -> str:
    """Run a command and return stripped stdout. Returns 'unavailable' on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"


def hash_text(text: str) -> str:
    """Return SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pip_freeze_hash() -> str:
    """Hash the full pip freeze output — canonical record of all installed packages."""
    freeze_output = run([sys.executable, "-m", "pip", "freeze", "--all"])
    if freeze_output == "unavailable":
        return "unavailable"
    # Sort for stability (pip freeze is deterministic but be explicit)
    lines = sorted(freeze_output.splitlines())
    return hash_text("\n".join(lines))


def binary_versions() -> dict[str, str]:
    """Capture version strings for key non-Python binaries baked into the image."""
    return {
        "python":     run([sys.executable, "--version"]),
        "uv":         run(["uv", "--version"]),
        "rust":       run(["rustc", "--version"]),
        "cargo":      run(["cargo", "--version"]),
        "go":         run(["go", "version"]),
        "node":       run(["node", "--version"]),
        "java":       run(["java", "--version"]).splitlines()[0] if run(["java", "--version"]) != "unavailable" else "unavailable",
        "kotlin":     run(["kotlinc", "-version"]),
        "scala":      run(["scala", "-version"]),
        "dotnet":     run(["dotnet", "--version"]),
        "cmake":      run(["cmake", "--version"]).splitlines()[0],
        "ray":        run([sys.executable, "-c", "import ray; print(ray.__version__)"]),
    }


def generate_fingerprint() -> dict:
    """Assemble the full fingerprint payload."""
    versions = binary_versions()
    freeze_hash = pip_freeze_hash()

    return {
        "built_at":        datetime.now(timezone.utc).isoformat(),
        "pip_freeze_hash": freeze_hash,
        "binaries":        versions,
        # Composite hash across pip state + all binary version strings — this is
        # the single value the checker compares at runtime.
        "composite_hash":  hash_text(
            freeze_hash + json.dumps(versions, sort_keys=True)
        ),
    }


def main():
    fingerprint = generate_fingerprint()
    print(json.dumps(fingerprint, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()