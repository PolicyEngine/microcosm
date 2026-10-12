"""Runtime identity attached to the first telemetry event."""

from __future__ import annotations

import os
import subprocess
from importlib import metadata
from platform import platform
from typing import Any, Final

_IDENTITY_DISTRIBUTIONS: Final = (
    "microcosm-build",
    "microcosm-graph",
    "policyengine-us",
    "policyengine-uk",
)
_GIT_IDENTITY_TIMEOUT_SECONDS: Final = 1.0


def runtime_identity() -> dict[str, Any]:
    """Describe source, host capacity, and installed runtime versions."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_IDENTITY_TIMEOUT_SECONDS,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    try:
        memory_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
    except (AttributeError, OSError, ValueError):
        memory_bytes = None
    versions = {}
    for distribution in _IDENTITY_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return {
        "git_commit": commit,
        "host": {
            "platform": platform(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": memory_bytes,
        },
        "runtime": versions,
    }
