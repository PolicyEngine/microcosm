"""Release fixtures in the build-folder layout (hand-over step H2c).

A release that publishes each build into its own folder,
``releases/<release>/<build>/``, names the release and the build as two
strings in both manifests. Its release manifest records ``built_at`` and the
built-with packages at the top level, and each artifact's ``role`` without a
``revision``. These helpers rewrite a fixture written in the older tagged
shape, so each test keeps its own fixture builder.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: A calibration attempt ID, the build every UK fixture names by default.
UK_NATIONAL_BUILD = "uk-frs-calibration-attempt-20260920T120000Z-deadbeef"
UK_DENSE_BUILD = "uk-local-candidate-f100-s42-20260920T120000Z-deadbeef"
UK_LOCAL_BUILD = "uk-local-candidate-k55000-20260920T120000Z-deadbeef"

#: The fields that named the build before the build-folder layout.
_RETIRED_BUILD_FIELDS = ("build_id", "attempt_id", "cut_tag")


def artifact_role(key: str, kind: object) -> str:
    """The file role a build-folder manifest records for one artifact."""
    if kind == "microdata":
        return "dataset"
    if kind == "calibration":
        return "calibration_weights"
    return key


def build_folder_build_manifest(
    manifest: Mapping[str, Any], *, release: str, build: str
) -> dict[str, Any]:
    """The build manifest with ``release`` and ``build`` as its identity."""
    rewritten = {
        key: value
        for key, value in manifest.items()
        if key not in _RETIRED_BUILD_FIELDS
    }
    return {"release": release, "build": build, **rewritten}


def build_folder_release_manifest(
    manifest: Mapping[str, Any], *, release: str, build: str
) -> dict[str, Any]:
    """The release manifest in the build-folder layout.

    The old ``build`` object's ``built_at`` and built-with packages move to
    the top level; ``built_at`` gets a fixed time when the fixture has none.
    """
    rewritten = dict(manifest)
    old_build = rewritten.pop("build", None)
    details = dict(old_build) if isinstance(old_build, Mapping) else {}
    for field in _RETIRED_BUILD_FIELDS:
        details.pop(field, None)
    details.setdefault("built_at", "2026-09-20T12:00:00+00:00")
    artifacts = rewritten.get("artifacts")
    if isinstance(artifacts, Mapping):
        rewritten["artifacts"] = {
            key: {
                **{name: value for name, value in entry.items() if name != "revision"},
                "role": artifact_role(key, entry.get("kind")),
            }
            if isinstance(entry, Mapping)
            else entry
            for key, entry in artifacts.items()
        }
    return {"release": release, "build": build, **details, **rewritten}


def to_build_folder_layout(release_dir: Path, *, build: str) -> Path:
    """Rewrite a release directory's two manifests in the build-folder layout.

    The directory's name is the release. A checksum ledger is not refreshed.
    """
    release = release_dir.name
    for name, rewrite in (
        ("build_manifest.json", build_folder_build_manifest),
        ("release_manifest.json", build_folder_release_manifest),
    ):
        path = release_dir / name
        if path.is_file():
            manifest = json.loads(path.read_text())
            path.write_text(json.dumps(rewrite(manifest, release=release, build=build)))
    return release_dir
