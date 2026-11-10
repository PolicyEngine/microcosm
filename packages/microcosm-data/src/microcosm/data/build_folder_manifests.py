"""Rewrite a release directory into the build-folder layout (microcosm#1570).

A release that publishes each build into its own folder,
``releases/<release>/<build>/``, names the release and the build as two
strings in both manifests, records the build's details (``built_at``, the
built-with packages and the rest of the tagged layout's ``build`` object) at
the release manifest's top level, and gives each artifact a ``role`` and no
``revision``, since a build folder is never retagged.

Each US release writer builds its manifests in the tagged shape it has always
written, in a directory named after its release, and then calls
:func:`rewrite_release_dir`. :func:`publication_commands` gives the
``microcosm-publish-release-build`` commands the writer prints: one that
publishes the build, and one that makes it its release's current build.

A build that names another build (an SPM enrichment or a recertification its
parent, an ACS build its donor) names a build in a folder by ``release`` and
``build``, and a build that keeps only its tag by its ID and ``revision``.
:func:`referenced_build` says where either one's files are read.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microcosm.data.contract import (
    CHECKSUMS_FILE,
    RETIRED_BUILD_FIELDS,
    ReleaseContractError,
    publishes_to_build_folder,
    release_artifact_role,
)

__all__ = [
    "ReferencedBuild",
    "build_folder_build_manifest",
    "build_folder_release_manifest",
    "publication_commands",
    "referenced_build",
    "rewrite_release_dir",
]

_RELEASE_MANIFEST = "release_manifest.json"
_BUILD_MANIFEST = "build_manifest.json"
#: One line of a checksum ledger, as ``sha256sum`` writes it.
_LEDGER_LINE = re.compile(r"(?P<digest>[0-9a-f]{64}) [ *](?P<name>.+)")


def _refuse(release: str, message: str) -> ReleaseContractError:
    return ReleaseContractError(Path(release), [message])


def build_folder_release_manifest(
    manifest: Mapping[str, Any], *, release: str, build: str
) -> dict[str, Any]:
    """A release manifest written in the tagged shape, rewritten for ``release``.

    ``release`` and ``build`` become strings; the ``build`` object's other
    fields move to the top level, where a field already there must have the
    same value; and each artifact loses its ``revision`` and gains its
    ``role``.
    """
    old_build = manifest.get("build")
    if not isinstance(old_build, Mapping):
        raise _refuse(
            release,
            f"{_RELEASE_MANIFEST} has no build object: it is already in the "
            "build-folder layout, or not a release manifest",
        )
    if "release" in manifest:
        raise _refuse(release, f"{_RELEASE_MANIFEST} already names a release")
    rest = {key: value for key, value in manifest.items() if key != "build"}
    details = {
        key: value
        for key, value in old_build.items()
        if key not in RETIRED_BUILD_FIELDS
    }
    clashes = sorted(
        key for key in set(details) & set(rest) if details[key] != rest[key]
    )
    if clashes:
        raise _refuse(
            release,
            f"{_RELEASE_MANIFEST}'s build object and top level differ on {clashes}",
        )
    artifacts = rest.pop("artifacts", None)
    rewritten_artifacts = (
        {
            key: {
                **{name: value for name, value in entry.items() if name != "revision"},
                "role": release_artifact_role(key, entry.get("kind")),
            }
            if isinstance(entry, Mapping)
            else entry
            for key, entry in artifacts.items()
        }
        if isinstance(artifacts, Mapping)
        else artifacts
    )
    schema = (
        {"schema_version": rest.pop("schema_version")}
        if "schema_version" in rest
        else {}
    )
    return {
        **schema,
        "release": release,
        "build": build,
        **{key: value for key, value in details.items() if key not in rest},
        **rest,
        "artifacts": rewritten_artifacts,
    }


def build_folder_build_manifest(
    manifest: Mapping[str, Any], *, release: str, build: str
) -> dict[str, Any]:
    """A build manifest written in the tagged shape, rewritten for ``release``."""
    if "release" in manifest:
        raise _refuse(release, f"{_BUILD_MANIFEST} already names a release")
    return {
        "release": release,
        "build": build,
        **{
            key: value
            for key, value in manifest.items()
            if key not in RETIRED_BUILD_FIELDS and key != "build"
        },
    }


def _serialized(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=False) + "\n"


def _refreshed_ledger(release_dir: Path, ledger: str, rewritten: set[str]) -> str:
    """The ledger with the rewritten files' lines given their new digests."""
    lines = []
    for line in ledger.splitlines():
        match = _LEDGER_LINE.fullmatch(line)
        if match is None or match["name"] not in rewritten:
            lines.append(line)
            continue
        digest = hashlib.sha256((release_dir / match["name"]).read_bytes()).hexdigest()
        lines.append(f"{digest}  {match['name']}")
    return "\n".join(lines) + "\n"


def rewrite_release_dir(release_dir: Path | str, *, build: str) -> Path:
    """Rewrite a writer's release directory into the build-folder layout.

    The directory's name is the release, which must publish into build
    folders. Both manifests are rewritten in place, and a checksum ledger's
    lines for them get their new digests.
    """
    release_dir = Path(release_dir)
    release = release_dir.name
    if not publishes_to_build_folder(release):
        raise _refuse(
            release,
            f"{release_dir} is named {release!r}, a release that does not publish "
            "into build folders; name the directory after its release",
        )
    rewritten: set[str] = set()
    for name, rewrite in (
        (_RELEASE_MANIFEST, build_folder_release_manifest),
        (_BUILD_MANIFEST, build_folder_build_manifest),
    ):
        path = release_dir / name
        if path.is_file():
            manifest = json.loads(path.read_text())
            path.write_text(
                _serialized(rewrite(manifest, release=release, build=build))
            )
            rewritten.add(name)
    ledger_path = release_dir / CHECKSUMS_FILE
    if ledger_path.is_file():
        ledger_path.write_text(
            _refreshed_ledger(release_dir, ledger_path.read_text(), rewritten)
        )
    return release_dir


def publication_commands(
    release_dir: Path | str, *, repo_id: str, artifact_root: Path | str | None
) -> dict[str, str]:
    """The emitter commands a writer prints for its release directory.

    ``publish_command`` publishes the build without changing its release's
    current build; ``promote_command`` then makes it current.
    """
    command = ["uv", "run", "microcosm-publish-release-build", str(release_dir)]
    publish = [*command, "--repo-id", repo_id]
    if artifact_root is not None:
        publish += ["--artifact-root", str(artifact_root)]
    return {
        "publish_command": shlex.join(publish),
        "promote_command": shlex.join([*command, "--repo-id", repo_id, "--promote"]),
    }


@dataclass(frozen=True)
class ReferencedBuild:
    """Where a referenced build's files are: a repository, the revision to
    read them at, and the folder that holds its manifests."""

    repo_id: str
    revision: str
    folder: str


def referenced_build(reference: Mapping[str, Any]) -> ReferencedBuild:
    """Where the build a reference names is read.

    A build-folder reference has ``release`` and ``build`` and is read at
    ``main`` from ``releases/<release>/<build>``, which never changes. A tag
    reference has a ``revision`` and the build's ID as ``build_id`` or
    ``release_id``, and is read at that tag from ``releases/<id>``. Both have
    ``repo_id``; a reference with parts of both forms is refused.
    """
    repo_id = reference.get("repo_id")
    release, build = reference.get("release"), reference.get("build")
    revision = reference.get("revision")
    build_id = reference.get("build_id") or reference.get("release_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"a build reference names no repo_id: {dict(reference)}")
    folder_form = release is not None or build is not None
    if folder_form:
        if revision is not None or not (
            isinstance(release, str) and release and isinstance(build, str) and build
        ):
            raise ValueError(
                "a build-folder reference names a release and a build, and no "
                f"revision: {dict(reference)}"
            )
        return ReferencedBuild(
            repo_id=repo_id, revision="main", folder=f"releases/{release}/{build}"
        )
    if not (isinstance(revision, str) and revision and isinstance(build_id, str)):
        raise ValueError(
            "a tag reference names the build's ID (build_id or release_id) and "
            f"its revision: {dict(reference)}"
        )
    return ReferencedBuild(
        repo_id=repo_id, revision=revision, folder=f"releases/{build_id}"
    )
