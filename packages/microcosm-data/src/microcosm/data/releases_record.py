"""Reading ``releases.json``: which build of each release is current.

A dataset repository whose releases publish each build into its own folder,
``releases/<release>/<build>/``, has one ``releases.json`` at its root. The
emitter's release writer is its only writer. It records each release's role
(``national`` or ``local_area``), its current build or ``null``, and every
published build with its ``published_at`` and whether it is withdrawn::

    {
      "schema_version": 1,
      "default": "microcosm-uk-2024-25-national",
      "releases": {
        "microcosm-uk-2024-25-national": {
          "role": "national",
          "current": "uk-frs-calibration-attempt-20261019T083323Z-6cba5d71",
          "builds": {
            "uk-frs-calibration-attempt-20261019T083323Z-6cba5d71": {
              "published_at": "2026-10-19T09:00:00+00:00",
              "withdrawn": false
            }
          }
        }
      }
    }

The writer's rules live in ``microcosm_emitter_interface.releases``. This
package also supports Python versions that the interface does not, so it
repeats the rules a reader needs instead of importing them; a build-shard
test runs both on the same records. A reader accepts exactly the records the
writer accepts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

__all__ = [
    "MAX_RECORD_BYTES",
    "RELEASES_RECORD_PATH",
    "BuildEntry",
    "ReleaseEntry",
    "ReleasesRecord",
    "build_folder",
    "parse_releases_record",
]

RELEASES_RECORD_PATH: Final = "releases.json"
_SCHEMA_VERSION: Final = 1
#: The largest record the writer writes, and so the largest a reader reads.
MAX_RECORD_BYTES: Final = 4 * 1024 * 1024
_SAFE_IDENTIFIER: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROLE_IDENTIFIER: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RECORD_KEYS: Final = frozenset({"schema_version", "default", "releases"})
_RELEASE_KEYS: Final = frozenset({"role", "current", "builds"})
_BUILD_KEYS: Final = frozenset({"published_at", "withdrawn"})


@dataclass(frozen=True)
class BuildEntry:
    """One published build: when it was published and whether it is withdrawn."""

    published_at: str
    withdrawn: bool


@dataclass(frozen=True)
class ReleaseEntry:
    """One release: its role, its current build or ``None``, and its builds."""

    role: str
    current: str | None
    builds: Mapping[str, BuildEntry]


@dataclass(frozen=True)
class ReleasesRecord:
    """A ``releases.json`` the release writer would accept."""

    default: str | None
    releases: Mapping[str, ReleaseEntry]


def build_folder(release: str, build: str) -> str:
    """A build's folder in the repository, ``releases/<release>/<build>``."""
    return f"releases/{release}/{build}"


def _safe(value: object) -> bool:
    return isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) is not None


def parse_releases_record(content: bytes) -> ReleasesRecord:
    """Parse ``releases.json``; raise ``ValueError`` naming the first broken rule.

    The rules are the writer's: at most ``MAX_RECORD_BYTES``; exactly the
    documented keys at every level; the integer ``schema_version`` 1; release
    and build names the emitter accepts for a folder; role names; a non-empty
    ``published_at`` and a boolean ``withdrawn`` on every build; a ``default``
    that is ``null`` or a listed release; and a ``current`` that is ``null``
    or a listed build that is not withdrawn.
    """
    if len(content) > MAX_RECORD_BYTES:
        raise ValueError(f"it is larger than {MAX_RECORD_BYTES} bytes")
    try:
        document = json.loads(content)
    except (ValueError, RecursionError):
        raise ValueError("it is not a readable JSON document") from None
    if not isinstance(document, dict) or set(document) != _RECORD_KEYS:
        raise ValueError(f"it must have exactly the keys {sorted(_RECORD_KEYS)}")
    version = document["schema_version"]
    # True and 1.0 equal 1 in Python; only the integer is the version.
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError(f"its schema_version must be {_SCHEMA_VERSION}")
    releases_document = document["releases"]
    if not isinstance(releases_document, dict):
        raise ValueError("its releases must be an object")
    releases: dict[str, ReleaseEntry] = {}
    for release, entry in releases_document.items():
        if not _safe(release):
            raise ValueError("a release name is not a safe identifier")
        if not isinstance(entry, dict) or set(entry) != _RELEASE_KEYS:
            raise ValueError(f"release {release} must have {sorted(_RELEASE_KEYS)}")
        role = entry["role"]
        if not isinstance(role, str) or _ROLE_IDENTIFIER.fullmatch(role) is None:
            raise ValueError(f"release {release} has an invalid role")
        current = entry["current"]
        if current is not None and not isinstance(current, str):
            raise ValueError(f"release {release} current must be a build or null")
        if not isinstance(entry["builds"], dict):
            raise ValueError(f"release {release} builds must be an object")
        builds: dict[str, BuildEntry] = {}
        for build, build_entry in entry["builds"].items():
            if not _safe(build):
                raise ValueError("a build name is not a safe identifier")
            if (
                not isinstance(build_entry, dict)
                or set(build_entry) != _BUILD_KEYS
                or not isinstance(build_entry["published_at"], str)
                or not build_entry["published_at"]
                or not isinstance(build_entry["withdrawn"], bool)
            ):
                raise ValueError(f"build {build} of {release} is malformed")
            builds[build] = BuildEntry(
                build_entry["published_at"], build_entry["withdrawn"]
            )
        if current is not None and current not in builds:
            raise ValueError(f"current build {current} of {release} is not listed")
        if current is not None and builds[current].withdrawn:
            raise ValueError(f"current build {current} of {release} is withdrawn")
        releases[release] = ReleaseEntry(role, current, builds)
    default = document["default"]
    if default is not None and not isinstance(default, str):
        raise ValueError("its default must be a release name or null")
    if default is not None and default not in releases:
        raise ValueError(f"its default {default} is not listed")
    return ReleasesRecord(default=default, releases=releases)
