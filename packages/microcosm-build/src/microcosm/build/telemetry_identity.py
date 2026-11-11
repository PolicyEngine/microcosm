"""Runtime identity attached to the first telemetry event."""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Mapping
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
#: The Modal stage runner (``tools/modal_us_stage_plan.py``) sets this in a
#: stage tool's environment to the resource class the stage's container
#: requested, as JSON: ``class``, ``cpu`` and ``cpu_limit`` in cores (the
#: limit null when the class sets none) and ``memory_mib``. The runner imports
#: nothing from the workspace, so it writes the name itself.
MODAL_RESOURCES_ENV: Final = "MICROCOSM_MODAL_RESOURCES"
_MODAL_RESOURCE_KEYS: Final = frozenset({"class", "cpu", "cpu_limit", "memory_mib"})
_MODAL_RESOURCE_CLASS_MAX_CHARS: Final = 64


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def modal_resources(environ: Mapping[str, str] = os.environ) -> dict[str, Any] | None:
    """The resource request ``MODAL_RESOURCES_ENV`` names, or None.

    None outside a Modal stage, where the variable is unset, and for a value
    that is not that JSON object exactly, which is left out rather than sent.
    """

    text = environ.get(MODAL_RESOURCES_ENV)
    if text is None:
        return None
    try:
        request = json.loads(text)
    except ValueError:
        return None
    if not isinstance(request, dict) or set(request) != _MODAL_RESOURCE_KEYS:
        return None
    name = request["class"]
    memory_mib = request["memory_mib"]
    if (
        not isinstance(name, str)
        or not 0 < len(name) <= _MODAL_RESOURCE_CLASS_MAX_CHARS
        or not _positive_number(request["cpu"])
        or not (request["cpu_limit"] is None or _positive_number(request["cpu_limit"]))
        or not isinstance(memory_mib, int)
        or isinstance(memory_mib, bool)
        or memory_mib <= 0
    ):
        return None
    return request


def runtime_identity() -> dict[str, Any]:
    """Describe source, host capacity, and installed runtime versions.

    ``modal_resources`` is the resource request of the Modal container a stage
    runs in (see ``modal_resources``), and None for a build anywhere else.
    """

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
        "modal_resources": modal_resources(),
    }
