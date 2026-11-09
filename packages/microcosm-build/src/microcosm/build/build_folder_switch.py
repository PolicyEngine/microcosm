"""Switch a dataset repository from tags and pointer files to build folders.

A one-time tool, run once for each dataset repository (tracker 2c, plan item
P3). Before the switch, a repository's builds are tagged commits: a build's
files are in its tagged commit's ``releases/<release>/`` folder and at the
repository's root, and ``latest.json`` or a ``latest-<line>.json`` pointer
file names the current build. After it, each build is in its own folder,
``releases/<release>/<build>/``, and the repository's ``releases.json`` is the
only record of each release's builds and which one is current. From then on
the emitter's release writer is the only thing that changes it.

The switch is two steps, each a separate run of this tool:

``copy``
    One commit for each build that moves adds its folder:

    - its release and build manifests, rewritten in the build-folder layout:
      ``release`` and ``build`` as strings, the build details (``built_at``
      and the built-with packages) at the top level, a ``role`` on each
      artifact and no ``revision``, plus ``migrated_from``, which names the
      tag, its commit and the SHA-256 of the manifest it was rewritten from.
      A release manifest that declares its own ``build_manifest.json``, as
      the US SPM builds' do, declares the rewritten one by its new SHA-256;
    - its checksum ledger, when the build has one, with the two manifests'
      lines changed to the rewritten manifests' digests;
    - every other file the build's release manifest declares, copied from the
      tagged commit with ``CommitOperationCopy``: the Hub copies an LFS file,
      which every large file is, on its own servers, and ``huggingface_hub``
      downloads a small regular file and sends it in the commit, so nothing
      large is downloaded. Each must match the SHA-256 the manifest records;
    - every file in the build's tagged folder that its checksum ledger
      records but its release manifest does not declare, as each US ACS
      local-area build has, copied the same way. Each must match the SHA-256
      the ledger records, and stays undeclared.

    A folder that already holds a copy of the same tagged build is skipped,
    so the step can run again for builds published since. It writes no
    ``releases.json``, so the repository keeps publishing and promoting
    through its pointer files.

``switch``
    One commit writes ``releases.json`` and deletes every ``latest*.json``
    pointer file. It refuses unless each moving build's folder holds a copy
    of it: exactly its files, a release manifest naming the release, the
    build and the tagged manifest it was rewritten from, and every declared
    file with the SHA-256 it had at the tag. It also refuses a build folder
    on ``main`` that the record would not list, a change to the
    repository's tags between its first read and its commit (a tagged
    publication that does not commit to ``main`` would otherwise be lost),
    and, at its first read or just before its commit, any
    ``release-staging/`` branch, which the tagged publisher keeps from
    before it uploads a build until after it tags it. ``--finished-branch
    BRANCH`` accepts one that a finished publication left behind, while its
    build is tagged and the commit it names is on ``main`` (a publication
    still running has either no tag yet or a branch on the tagged commit,
    which is not on ``main`` until the branch is deleted); the branch stays,
    and must not move before the commit. A retry of the same build's
    publication that has just made its branch from ``main``, and not yet
    committed to it, meets both conditions too, so name a branch only while
    nobody publishes.
    The record lists every moved build, with the date of its tagged commit
    as ``published_at``. The build
    each pointer file names becomes its release's current build, and
    ``default`` is the release of ``latest-national.json``'s build, or else
    of ``latest.json``'s. The
    emitter's release writer would refuse this commit, because ``current``
    may not change while a pointer file exists, so this tool makes it itself,
    after running the interface's checks on the record, on the repository's
    build folders and on the commit, with the pointer files counted as
    deleted.

**Which builds move.**

Each tag is resolved to the commit it names: the Hub lists an annotated tag
with the ID of its tag object instead.

- Every tag ``<release>-<YYYYMMDDTHHMMSSZ>-<8 hex digits>`` of a release that
  publishes into build folders
  (:func:`microcosm.data.contract.publishes_to_build_folder`) whose release
  manifest's ``build.cut_tag`` is the tag. Its build is the manifest's
  ``build.attempt_id``. ``--exclude TAG`` leaves one out.
- Each tag that ``--include TAG=RELEASE`` names: a release whose every build
  had its own release name and folder ``releases/<TAG>/`` (the June 2023 UK
  release, and every US release). Its build is ``TAG``, in ``RELEASE``.

``--current RELEASE=BUILD`` makes a moved build current where no pointer file
names one, and ``--default RELEASE`` chooses the default. The switch refuses
a pointer file whose build does not move, unless ``--retire-pointer PATH``
says that the switch should delete it with nothing in its place.

Run the ``switch`` step only while nobody publishes to the repository. The
tagged publisher checks for ``releases.json`` only when it starts, so a
publication that starts just before the switch commit, and makes its staging
branch after the switch's last read, would still be in neither the record nor
a folder. A build tagged while ``copy`` runs is safe: the switch refuses until
``copy`` runs again.

Both steps read the Hub and print, as JSON, what they would do. They change
nothing without ``--write``. They read and write with ``HF_TOKEN``, then
``HUGGINGFACE_TOKEN``, then the login ``hf auth login`` saved, and refuse to
run with ``HF_HUB_OFFLINE`` or ``TRANSFORMERS_OFFLINE`` set. Each commit names
the head it was checked against as its ``parent_commit``, so if another
commit lands first, the Hub refuses this one; run the step again.

Run it as ``uv run python -m microcosm.build.build_folder_switch copy
<repository> ...``, then the same with ``switch``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC
from pathlib import Path
from typing import Any

from microcosm_emitter_interface.releases import (
    RELEASE_MANIFEST_NAME,
    RELEASES_RECORD_PATH,
    ReleaseRecordError,
    ReleasesRecord,
    add_build,
    build_folder,
    check_change,
    check_commit,
    check_repository,
    is_pointer_file,
    make_current,
)

from microcosm.build.staging_dataset import hub_offline
from microcosm.data.contract import (
    CHECKSUMS_FILE,
    NATIONAL_DEFAULT_DATASET_ROLE,
    RELEASE_ROLES,
    RETIRED_BUILD_FIELDS,
    publishes_to_build_folder,
    release_artifact_role,
)

__all__ = [
    "BuildMove",
    "Selection",
    "SwitchRefusedError",
    "build_folder_build_manifest",
    "build_folder_release_manifest",
    "copy_builds",
    "main",
    "switch_repository",
]

_DATASET = "dataset"
#: The branch the tagged publisher (``microcosm.data.release``) makes for each
#: publication: from before it uploads the build until after it tags it.
_STAGING_BRANCH_PREFIX = "release-staging/"
_BUILD_MANIFEST_NAME = "build_manifest.json"
#: A tag of a release that publishes into build folders: the release, then
#: the calibration attempt's time and digest.
_DATED_TAG = re.compile(r"(?P<release>.+)-(?P<suffix>[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8})")
#: The pointer files whose build's release becomes ``default``, in order.
_DEFAULT_POINTER_FILES = ("latest-national.json", "latest.json")
#: One line of a checksum ledger, as ``sha256sum`` writes it.
_LEDGER_LINE = re.compile(r"(?P<digest>[0-9a-f]{64}) [ *](?P<name>[^/]+)")
#: The field a rewritten release manifest records its origin in.
MIGRATED_FROM_FIELD = "migrated_from"
#: How the copy step reports each build, by the state of its folder.
_COPY_STATUS = {
    "missing": "to_copy",
    "copied": "already_copied",
    "occupied": "occupied",
}


class SwitchRefusedError(ValueError):
    """A step cannot go ahead; nothing was written."""


@dataclass(frozen=True)
class Selection:
    """Which builds move, and what the switch makes current and default.

    ``include`` maps a release-per-build tag to the release it moves into;
    ``exclude`` leaves automatically selected tags out; ``current`` maps a
    release to a moved build to make current where no pointer file names
    one; ``retired_pointers`` are pointer files the switch deletes with
    nothing in their place; ``finished_branches`` are staging branches that
    finished publications left behind, which the switch accepts while each
    one's build is tagged and its head is on ``main``.
    """

    include: Mapping[str, str]
    exclude: frozenset[str]
    current: Mapping[str, str]
    default: str | None
    retired_pointers: frozenset[str]
    finished_branches: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CopiedFile:
    """A file copied into a build folder from the build's tagged commit.

    ``declared`` is false for a file that the build's checksum ledger records
    but its release manifest does not declare, as each US ACS local-area
    build has; it is copied, and stays undeclared.
    """

    name: str
    source: str
    sha256: str
    declared: bool = True


@dataclass(frozen=True)
class BuildMove:
    """One tagged build, and what its folder holds once it is copied.

    ``written`` maps each file written with new bytes (the two manifests, and
    the checksum ledger) to those bytes; ``copied`` lists every other file;
    ``declared`` maps each file the rewritten release manifest declares to
    its SHA-256 there; ``left_behind`` names files in the tagged commit's
    release folder that neither the build's manifests nor its checksum
    ledger name (files an earlier build left there).
    ``tagged_release`` is the release's name at the tag: ``release`` for a
    dated tag, and the tag itself for a release-per-build tag.
    """

    release: str
    build: str
    release_role: str
    tagged_release: str
    tag: str
    commit: str
    published_at: str
    written: Mapping[str, bytes]
    copied: tuple[CopiedFile, ...]
    declared: Mapping[str, str]
    left_behind: tuple[str, ...]

    @property
    def folder(self) -> str:
        """The build's folder, ``releases/<release>/<build>``."""
        return build_folder(self.release, self.build)

    @property
    def migrated_from(self) -> dict[str, str]:
        """What the rewritten release manifest records as its origin."""
        manifest = json.loads(self.written[RELEASE_MANIFEST_NAME])
        return dict(manifest[MIGRATED_FROM_FIELD])

    def paths(self) -> list[str]:
        """Every path the build's folder holds once it is copied."""
        names = [*self.written, *(item.name for item in self.copied)]
        return [f"{self.folder}/{name}" for name in sorted(names)]

    def summary(self) -> dict[str, Any]:
        """The build as the tool prints it."""
        return {
            "release": self.release,
            "build": self.build,
            "release_role": self.release_role,
            "tag": self.tag,
            "commit": self.commit,
            "published_at": self.published_at,
            "folder": self.folder,
            "written": sorted(self.written),
            "copied": [
                {"name": item.name, "from": item.source, "sha256": item.sha256}
                for item in self.copied
            ],
            "undeclared": [item.name for item in self.copied if not item.declared],
            "left_behind": list(self.left_behind),
        }


@dataclass(frozen=True)
class _Candidate:
    """A tag to move: where its build is, and how its build is named."""

    tag: str
    commit: str
    release: str
    source_folder: str
    #: ``None`` for a dated tag, whose build is its manifest's attempt ID.
    build: str | None


class _Hub:
    """The ``HfApi`` calls the tool makes, on one dataset repository."""

    def __init__(self, api: Any, repository: str) -> None:
        self.api = api
        self.repository = repository

    def head(self) -> str:
        return self.resolve("main")

    def resolve(self, revision: str) -> str:
        """The commit a revision (a branch, a tag or a commit) names."""
        info = self.api.repo_info(
            self.repository, repo_type=_DATASET, revision=revision
        )
        return str(info.sha)

    def files(self, revision: str) -> list[str]:
        return list(
            self.api.list_repo_files(
                self.repository, repo_type=_DATASET, revision=revision
            )
        )

    def read(self, path: str, revision: str) -> bytes:
        local = self.api.hf_hub_download(
            self.repository, path, repo_type=_DATASET, revision=revision
        )
        return Path(local).read_bytes()

    def refs(self) -> tuple[dict[str, str], dict[str, str]]:
        """Each tag with the ID the Hub lists for it, and each of the tagged
        publisher's staging branches with the commit it names, from one
        request.

        The Hub lists an annotated tag with the ID of its tag object, not of
        its commit (:meth:`resolve` gives the commit); a tag that moves gets
        another ID either way. A staging branch means a tagged publication
        has started and not yet tagged its build, or failed before deleting
        the branch.
        """
        refs = self.api.list_repo_refs(self.repository, repo_type=_DATASET)
        listed = {tag.name: str(tag.target_commit) for tag in refs.tags}
        staging = {
            branch.name: str(branch.target_commit)
            for branch in refs.branches
            if branch.name.startswith(_STAGING_BRANCH_PREFIX)
        }
        return listed, staging

    def on_main(self, commit: str) -> bool:
        """Whether ``commit`` is in ``main``'s history."""
        history = self.api.list_repo_commits(
            self.repository, repo_type=_DATASET, revision="main"
        )
        return any(item.commit_id == commit for item in history)

    def commit_date(self, commit: str) -> str:
        """The commit's date, in UTC to the second, as ``published_at``."""
        history = self.api.list_repo_commits(
            self.repository, repo_type=_DATASET, revision=commit
        )
        entry = next(
            (item for item in history if item.commit_id == commit),
            None,
        )
        if entry is None:
            raise SwitchRefusedError(
                f"the Hub lists no commit {commit} in {self.repository}"
            )
        return entry.created_at.astimezone(UTC).replace(microsecond=0).isoformat()

    def sha256s(self, paths: Sequence[str], revision: str) -> dict[str, str]:
        """Each file's SHA-256 at ``revision``: an LFS file's from the Hub's
        file information, any other file's by reading it."""
        infos = self.api.get_paths_info(
            self.repository, list(paths), repo_type=_DATASET, revision=revision
        )
        by_path = {info.path: info for info in infos}
        digests: dict[str, str] = {}
        for path in paths:
            info = by_path.get(path)
            if info is None:
                raise SwitchRefusedError(f"{path} is not in commit {revision}")
            lfs = getattr(info, "lfs", None)
            digests[path] = (
                str(lfs.sha256)
                if lfs is not None
                else hashlib.sha256(self.read(path, revision)).hexdigest()
            )
        return digests

    def commit(self, operations: list[Any], *, message: str, parent: str) -> str:
        info = self.api.create_commit(
            self.repository,
            operations,
            commit_message=message,
            repo_type=_DATASET,
            parent_commit=parent,
        )
        return str(info.oid)


def _serialized(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=1) + "\n").encode()


def _merged_field(
    name: str, build_value: Any, release_value: Any, clashes: list[str]
) -> Any:
    """The value of a field both the build object and the top level have.

    Equal values stay; two objects whose shared fields are equal become one
    object with the fields of both. Each field on which they differ is
    added to ``clashes``, by its dotted name.
    """
    if build_value == release_value:
        return release_value
    if isinstance(build_value, Mapping) and isinstance(release_value, Mapping):
        differing = sorted(
            f"{name}.{key}"
            for key in set(build_value) & set(release_value)
            if build_value[key] != release_value[key]
        )
        clashes.extend(differing)
        return {**build_value, **release_value}
    clashes.append(name)
    return release_value


def build_folder_release_manifest(
    manifest: Mapping[str, Any],
    *,
    release: str,
    build: str,
    artifact_names: Mapping[str, str],
    migrated_from: Mapping[str, str],
    rewritten_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """A tagged release manifest rewritten in the build-folder layout.

    ``release`` and ``build`` become strings; the old ``build`` object's
    other fields, ``built_at`` and the built-with packages among them, move
    to the top level; each artifact loses its ``revision``, gains its
    ``role`` and records ``artifact_names``'s bare file name as its
    ``path``; and ``migrated_from`` records the origin. An artifact in
    ``rewritten_sha256`` (a build manifest the release manifest declares,
    which the switch rewrites too) records its new SHA-256.

    A field both the ``build`` object and the top level have must be equal,
    or be two objects that do not disagree on any field: the June 2023
    release has a ``metadata`` object in both, describing its one build,
    and they become one object.
    """
    old_build = manifest.get("build")
    if not isinstance(old_build, Mapping):
        raise SwitchRefusedError("its release manifest has no build object")
    rest = {key: value for key, value in manifest.items() if key != "build"}
    taken = sorted({"release", MIGRATED_FROM_FIELD} & set(rest))
    if taken:
        raise SwitchRefusedError(f"its release manifest already has {taken}")
    details = {
        key: value
        for key, value in old_build.items()
        if key not in RETIRED_BUILD_FIELDS
    }
    clashes: list[str] = []
    for key in sorted(set(details) & set(rest)):
        rest[key] = _merged_field(key, details[key], rest[key], clashes)
    if clashes:
        raise SwitchRefusedError(
            f"its build object and its top level differ on {clashes}"
        )
    details = {key: value for key, value in details.items() if key not in rest}
    rewritten = rewritten_sha256 or {}
    artifacts = {
        key: {
            **{name: value for name, value in entry.items() if name != "revision"},
            "path": artifact_names[key],
            "role": release_artifact_role(key, entry.get("kind")),
            **({"sha256": rewritten[key]} if key in rewritten else {}),
        }
        for key, entry in manifest["artifacts"].items()
    }
    schema = (
        {"schema_version": rest.pop("schema_version")}
        if "schema_version" in rest
        else {}
    )
    return {
        **schema,
        "release": release,
        "build": build,
        **details,
        **rest,
        "artifacts": artifacts,
        MIGRATED_FROM_FIELD: dict(migrated_from),
    }


def build_folder_build_manifest(
    manifest: Mapping[str, Any], *, release: str, build: str
) -> dict[str, Any]:
    """A tagged build manifest with ``release`` and ``build`` as its identity."""
    taken = sorted({"release", "build"} & set(manifest))
    if taken:
        raise SwitchRefusedError(f"its build manifest already has {taken}")
    return {
        "release": release,
        "build": build,
        **{
            key: value
            for key, value in manifest.items()
            if key not in RETIRED_BUILD_FIELDS
        },
    }


def _ledger_entries(ledger: bytes) -> dict[str, str]:
    """Each file a checksum ledger records, with its digest, in its order."""
    entries: dict[str, str] = {}
    for line in ledger.decode().splitlines():
        if not line.strip():
            continue
        match = _LEDGER_LINE.fullmatch(line)
        if match is None:
            raise SwitchRefusedError(f"its {CHECKSUMS_FILE} has a line it cannot read")
        if entries.setdefault(match["name"], match["digest"]) != match["digest"]:
            raise SwitchRefusedError(
                f"its {CHECKSUMS_FILE} records {match['name']} twice"
            )
    return entries


def _rewritten_ledger(
    ledger: bytes, *, expected: Mapping[str, str], replaced: Mapping[str, str]
) -> bytes:
    """The checksum ledger with the rewritten manifests' digests.

    Every line must name a file of the build with the digest that file had
    at the tagged commit (``expected``); the lines of the files in
    ``replaced`` get their new digests.
    """
    lines = []
    for name, digest in _ledger_entries(ledger).items():
        if expected.get(name) != digest:
            raise SwitchRefusedError(
                f"its {CHECKSUMS_FILE} records {name} with a digest that is not "
                "the tagged commit's"
            )
        lines.append(f"{replaced.get(name, digest)}  {name}")
    return ("\n".join(lines) + "\n").encode()


def _artifact_names(
    artifacts: Mapping[str, Any], *, folder: str, repository: str, tag: str
) -> dict[str, str]:
    """Each artifact's bare file name; refuse one this tool cannot move."""
    names: dict[str, str] = {}
    for key, entry in artifacts.items():
        if not isinstance(entry, Mapping):
            raise SwitchRefusedError(f"its artifact {key} is not an object")
        path, digest = entry.get("path"), entry.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise SwitchRefusedError(f"its artifact {key} has no path or sha256")
        if entry.get("repo_id") != repository:
            raise SwitchRefusedError(f"its artifact {key} is in another repository")
        if entry.get("revision") != tag:
            raise SwitchRefusedError(
                f"its artifact {key} is pinned to {entry.get('revision')!r}, not "
                f"the tag {tag}"
            )
        name = path.removeprefix(f"{folder}/")
        if not name or "/" in name or name in {".", ".."}:
            raise SwitchRefusedError(f"its artifact {key} has the path {path!r}")
        names[str(key)] = name
    if len(set(names.values())) != len(names):
        raise SwitchRefusedError("two of its artifacts have the same file name")
    # A release manifest may declare its own build manifest, as the US SPM
    # builds' do: the switch rewrites that file and declares the new bytes.
    taken = {RELEASE_MANIFEST_NAME, CHECKSUMS_FILE}
    if taken & set(names.values()):
        raise SwitchRefusedError(f"an artifact is named like one of {sorted(taken)}")
    return names


def _describe(hub: _Hub, candidate: _Candidate) -> BuildMove:
    """Read one tagged build and work out its folder; refuse if it cannot move."""
    tag, commit, folder = candidate.tag, candidate.commit, candidate.source_folder
    files_at_tag = set(hub.files(commit))
    manifest_path = f"{folder}/{RELEASE_MANIFEST_NAME}"
    if manifest_path not in files_at_tag:
        raise SwitchRefusedError(f"its tagged commit has no {manifest_path}")
    manifest_bytes = hub.read(manifest_path, commit)
    manifest = json.loads(manifest_bytes)
    old_build = manifest.get("build") if isinstance(manifest, Mapping) else None
    if not isinstance(old_build, Mapping):
        raise SwitchRefusedError("its release manifest has no build object")
    if candidate.build is None:
        if old_build.get("build_id") != candidate.release:
            raise SwitchRefusedError(
                f"its manifest's build_id is not {candidate.release}"
            )
        if old_build.get("cut_tag") != tag:
            raise SwitchRefusedError("its manifest's cut_tag is not the tag")
        build = old_build.get("attempt_id")
        if not isinstance(build, str) or not build:
            raise SwitchRefusedError("its manifest names no attempt_id")
    else:
        if old_build.get("build_id") != tag:
            raise SwitchRefusedError("its manifest's build_id is not the tag")
        build = candidate.build
    dataset_role = manifest.get("dataset_role", NATIONAL_DEFAULT_DATASET_ROLE)
    release_role = RELEASE_ROLES.get(dataset_role)
    if release_role is None:
        raise SwitchRefusedError(f"its dataset_role {dataset_role!r} has no role")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise SwitchRefusedError("its release manifest declares no artifacts")
    names = _artifact_names(
        artifacts, folder=folder, repository=hub.repository, tag=tag
    )
    sources = {}
    for key, name in names.items():
        in_folder = f"{folder}/{name}"
        if in_folder in files_at_tag:
            sources[key] = in_folder
        # A declared build manifest is this build's own, in its folder.
        elif name in files_at_tag and name != _BUILD_MANIFEST_NAME:
            sources[key] = name
        else:
            raise SwitchRefusedError(f"its tagged commit has no {name}")
    digests = hub.sha256s(sorted(set(sources.values())), commit)
    for key, source in sources.items():
        if digests[source] != artifacts[key]["sha256"]:
            raise SwitchRefusedError(
                f"{source} at its tagged commit does not match the SHA-256 its "
                "manifest records"
            )
    original_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    written: dict[str, bytes] = {}
    expected = {
        RELEASE_MANIFEST_NAME: original_sha256,
        **{names[key]: artifacts[key]["sha256"] for key in names},
    }
    replaced: dict[str, str] = {}
    build_manifest_path = f"{folder}/{_BUILD_MANIFEST_NAME}"
    if build_manifest_path in files_at_tag:
        original = hub.read(build_manifest_path, commit)
        rewritten = _serialized(
            build_folder_build_manifest(
                json.loads(original), release=candidate.release, build=build
            )
        )
        written[_BUILD_MANIFEST_NAME] = rewritten
        expected[_BUILD_MANIFEST_NAME] = hashlib.sha256(original).hexdigest()
        replaced[_BUILD_MANIFEST_NAME] = hashlib.sha256(rewritten).hexdigest()
    # The keys of a declared build manifest; the rewritten release manifest
    # declares the rewritten bytes.
    rewritten_keys = [
        key for key, name in names.items() if name == _BUILD_MANIFEST_NAME
    ]
    release_manifest = _serialized(
        build_folder_release_manifest(
            manifest,
            release=candidate.release,
            build=build,
            artifact_names=names,
            migrated_from={
                "tag": tag,
                "commit": commit,
                "release_manifest_sha256": original_sha256,
            },
            rewritten_sha256={
                key: replaced[_BUILD_MANIFEST_NAME] for key in rewritten_keys
            },
        )
    )
    written[RELEASE_MANIFEST_NAME] = release_manifest
    replaced[RELEASE_MANIFEST_NAME] = hashlib.sha256(release_manifest).hexdigest()
    undeclared: list[CopiedFile] = []
    ledger_path = f"{folder}/{CHECKSUMS_FILE}"
    if ledger_path in files_at_tag:
        ledger = hub.read(ledger_path, commit)
        recorded = _ledger_entries(ledger)
        # Files the ledger records that the release manifest doesn't declare
        # are the build's too, as each US ACS local-area build has them: they
        # are copied, checked against the ledger, and stay undeclared.
        own = {RELEASE_MANIFEST_NAME, _BUILD_MANIFEST_NAME, CHECKSUMS_FILE}
        extra = sorted(set(recorded) - own - set(names.values()))
        absent = [name for name in extra if f"{folder}/{name}" not in files_at_tag]
        if absent:
            raise SwitchRefusedError(
                f"its {CHECKSUMS_FILE} records {absent}, which its tagged commit's "
                f"{folder}/ does not hold"
            )
        extra_digests = (
            hub.sha256s([f"{folder}/{name}" for name in extra], commit) if extra else {}
        )
        for name in extra:
            if extra_digests[f"{folder}/{name}"] != recorded[name]:
                raise SwitchRefusedError(
                    f"its {CHECKSUMS_FILE} records {name} with a digest that is not "
                    "the tagged commit's"
                )
            expected[name] = recorded[name]
            undeclared.append(
                CopiedFile(
                    name=name,
                    source=f"{folder}/{name}",
                    sha256=recorded[name],
                    declared=False,
                )
            )
        written[CHECKSUMS_FILE] = _rewritten_ledger(
            ledger, expected=expected, replaced=replaced
        )
    copied = tuple(
        sorted(
            [
                *(
                    CopiedFile(
                        name=names[key],
                        source=sources[key],
                        sha256=artifacts[key]["sha256"],
                    )
                    for key in names
                    if key not in rewritten_keys
                ),
                *undeclared,
            ],
            key=lambda item: item.name,
        )
    )
    declared = {
        names[key]: (
            replaced[_BUILD_MANIFEST_NAME]
            if key in rewritten_keys
            else artifacts[key]["sha256"]
        )
        for key in names
    }
    kept = {*written, *(item.name for item in copied)}
    left_behind = tuple(
        sorted(
            path.removeprefix(f"{folder}/")
            for path in files_at_tag
            if path.rsplit("/", 1)[0] == folder
            and path.removeprefix(f"{folder}/") not in kept
        )
    )
    return BuildMove(
        release=candidate.release,
        build=build,
        release_role=release_role,
        tagged_release=folder.removeprefix("releases/"),
        tag=tag,
        commit=commit,
        published_at=hub.commit_date(commit),
        written=written,
        copied=copied,
        declared=declared,
        left_behind=left_behind,
    )


@dataclass
class _Survey:
    """The builds that move, and every tag that does not, with the reason."""

    #: Every tag the survey read, and the commit it names.
    tags: dict[str, str]
    #: Every tag the survey read, with the ID the Hub listed for it.
    listed_tags: dict[str, str]
    #: The tagged publisher's staging branches the survey's listing showed,
    #: each with the commit it names.
    staging_branches: dict[str, str]
    moves: list[BuildMove]
    skipped: list[dict[str, str]]
    not_moved: list[str]
    problems: list[str]


def _survey(hub: _Hub, selection: Selection) -> _Survey:
    """Find the builds that move and read each one."""
    listed_tags, staging_branches = hub.refs()
    tags = {tag: hub.resolve(tag) for tag in listed_tags}
    survey = _Survey(
        tags=tags,
        listed_tags=listed_tags,
        staging_branches=staging_branches,
        moves=[],
        skipped=[],
        not_moved=[],
        problems=[],
    )
    unknown = sorted({*selection.include, *selection.exclude} - set(tags))
    survey.problems.extend(f"{tag} is not a tag of {hub.repository}" for tag in unknown)
    for tag, commit in sorted(tags.items()):
        if tag in selection.exclude:
            survey.skipped.append({"tag": tag, "reason": "--exclude"})
            continue
        if tag in selection.include:
            candidate = _Candidate(
                tag=tag,
                commit=commit,
                release=selection.include[tag],
                source_folder=f"releases/{tag}",
                build=tag,
            )
        else:
            match = _DATED_TAG.fullmatch(tag)
            if match is None or not publishes_to_build_folder(match["release"]):
                survey.not_moved.append(tag)
                continue
            candidate = _Candidate(
                tag=tag,
                commit=commit,
                release=match["release"],
                source_folder=f"releases/{match['release']}",
                build=None,
            )
        try:
            survey.moves.append(_describe(hub, candidate))
        except (ValueError, KeyError) as error:
            survey.problems.append(f"{tag} cannot move: {error}")
    seen: dict[tuple[str, str], str] = {}
    for move in survey.moves:
        earlier = seen.setdefault((move.release, move.build), move.tag)
        if earlier != move.tag:
            survey.problems.append(
                f"{earlier} and {move.tag} are both build {move.build} of "
                f"{move.release}"
            )
    survey.moves.sort(key=lambda move: (move.release, move.published_at, move.build))
    return survey


def _folder_state(
    hub: _Hub, move: BuildMove, *, head: str, head_files: Sequence[str]
) -> tuple[str, str | None]:
    """The build's folder at ``head``, and why it is not a copy of the build.

    ``missing`` when the folder holds nothing; ``copied`` when it holds
    exactly the build's files, its release manifest names the release, the
    build and the tagged manifest it was rewritten from, every file the
    manifest declares has the SHA-256 the manifest records, and every
    undeclared file the checksum ledger records has the SHA-256 it had at
    the tag; ``occupied`` otherwise, with the reason. The rewritten
    manifests' bytes are not compared, and the checksum ledger at ``head`` is
    not read, nor the build manifest unless the release manifest declares
    it, so a folder copied by an earlier version of this tool still counts
    as copied.
    """
    folder = move.folder
    present = {path for path in head_files if path.startswith(folder + "/")}
    if not present:
        return "missing", None
    expected = set(move.paths())
    if present - expected:
        extra = sorted(path.removeprefix(f"{folder}/") for path in present - expected)
        return "occupied", f"{folder}/ also holds {extra}"
    if expected - present:
        absent = sorted(path.removeprefix(f"{folder}/") for path in expected - present)
        return "occupied", f"{folder}/ lacks {absent}"
    manifest_path = f"{folder}/{RELEASE_MANIFEST_NAME}"
    try:
        manifest = json.loads(hub.read(manifest_path, head))
    except ValueError:
        manifest = None
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("release") != move.release
        or manifest.get("build") != move.build
        or manifest.get(MIGRATED_FROM_FIELD) != move.migrated_from
    ):
        return "occupied", (
            f"{manifest_path} does not name release {move.release}, build "
            f"{move.build} and {MIGRATED_FROM_FIELD} {move.migrated_from}"
        )
    artifacts = manifest.get("artifacts")
    declared = (
        {
            entry.get("path"): entry.get("sha256")
            for entry in artifacts.values()
            if isinstance(entry, Mapping)
        }
        if isinstance(artifacts, Mapping)
        else {}
    )
    if declared != dict(move.declared):
        return "occupied", (
            f"{manifest_path} does not declare the files and SHA-256 values of "
            f"the manifest at {move.tag}"
        )
    recorded = {
        **move.declared,
        **{item.name: item.sha256 for item in move.copied if not item.declared},
    }
    digests = hub.sha256s([f"{folder}/{name}" for name in sorted(recorded)], head)
    for name, digest in sorted(recorded.items()):
        if digests[f"{folder}/{name}"] != digest:
            source = (
                "its manifest" if name in move.declared else f"its {CHECKSUMS_FILE}"
            )
            return "occupied", (
                f"{folder}/{name} does not match the SHA-256 {source} records"
            )
    return "copied", None


def _build_folder_of(path: str) -> str | None:
    """The build folder ``releases/<release>/<build>`` a path is inside, if
    any, as the emitter interface's ``check_commit`` finds it."""
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "releases":
        return "/".join(parts[:3])
    return None


def _refuse_switched(head_files: Sequence[str], repository: str) -> None:
    if RELEASES_RECORD_PATH in head_files:
        raise SwitchRefusedError(
            f"{repository} already has {RELEASES_RECORD_PATH}: it has switched, "
            "and the emitter's release writer publishes its builds"
        )


def _copy_operations(move: BuildMove) -> list[Any]:
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy

    return [
        *(
            CommitOperationAdd(
                path_in_repo=f"{move.folder}/{name}", path_or_fileobj=content
            )
            for name, content in sorted(move.written.items())
        ),
        *(
            CommitOperationCopy(
                src_path_in_repo=item.source,
                path_in_repo=f"{move.folder}/{item.name}",
                src_revision=move.commit,
            )
            for item in move.copied
        ),
    ]


def _copy_violations(move: BuildMove, head_files: Sequence[str]) -> list[str]:
    """The interface's rules for one copy commit and the folder it adds."""
    record = add_build(
        ReleasesRecord.empty(),
        release=move.release,
        build=move.build,
        release_role=move.release_role,
        published_at=move.published_at,
        make_current=False,
    )
    by_path = {
        f"{move.folder}/{name}": content for name, content in move.written.items()
    }
    violations = [
        *check_commit(
            before_files=head_files,
            added_paths=move.paths(),
            deleted_paths=[],
            before=None,
            after=ReleasesRecord.empty(),
        ),
        *check_repository(record, move.paths(), by_path.__getitem__),
    ]
    return [f"{violation.code}: {violation.message}" for violation in violations]


def copy_builds(
    api: Any, repository: str, selection: Selection, *, write: bool
) -> dict[str, Any]:
    """The ``copy`` step: add each moving build's folder, one commit per build.

    Without ``write`` it only reads, and reports every problem, the emitter
    interface's checks on the record the moving builds make and on each copy
    commit included. With it, it refuses before any commit when the report
    has a problem.
    """
    hub = _Hub(api, repository)
    head = hub.head()
    head_files = hub.files(head)
    _refuse_switched(head_files, repository)
    survey = _survey(hub, selection)
    builds = []
    to_copy = []
    for move in survey.moves:
        state, reason = _folder_state(hub, move, head=head, head_files=head_files)
        if state == "occupied":
            survey.problems.append(
                f"{move.folder}/ already holds files that are not a copy of "
                f"{move.tag}: {reason}"
            )
        elif state == "missing":
            to_copy.append(move)
        builds.append({**move.summary(), "status": _COPY_STATUS[state]})
    # Every moving build, copied or not, goes into one record at the switch,
    # so a name or role the interface refuses is a problem now, before any
    # folder is written.
    record_problems: list[str] = []
    _switch_record(survey.moves, {}, None, record_problems)
    survey.problems.extend(record_problems)
    if not record_problems:
        planned_files = list(head_files)
        for move in to_copy:
            survey.problems.extend(
                f"{move.tag} cannot be copied: {violation}"
                for violation in _copy_violations(move, planned_files)
            )
            planned_files.extend(move.paths())
    report: dict[str, Any] = {
        "step": "copy",
        "repository": repository,
        "head": head,
        "builds": builds,
        "skipped": survey.skipped,
        "not_moved": survey.not_moved,
        "problems": survey.problems,
        "written": False,
        "commits": [],
    }
    if not write:
        return report
    if survey.problems:
        raise SwitchRefusedError("nothing was copied: " + "; ".join(survey.problems))
    for move in to_copy:
        try:
            head = hub.commit(
                _copy_operations(move),
                message=(
                    f"Copy build {move.build} of {move.release} into its folder "
                    f"from tag {move.tag} ({move.commit})"
                ),
                parent=head,
            )
        # Whatever the Hub raised, the builds before this one are copied and
        # this one is not; running the step again copies the rest.
        except Exception as error:
            copied = [entry["tag"] for entry in report["commits"]]
            raise SwitchRefusedError(
                f"copying {move.tag} failed ({type(error).__name__}: {error}); "
                f"copied before it in this run: {copied or 'none'}; run the step "
                "again to copy the rest"
            ) from error
        report["commits"].append(
            {"tag": move.tag, "folder": move.folder, "commit": head}
        )
    report["written"] = True
    return report


def _pointer_tag(content: bytes) -> tuple[str | None, str | None]:
    """The tag a pointer file names, and the release it says that is."""
    try:
        payload = json.loads(content)
    except ValueError:
        return None, None
    if not isinstance(payload, Mapping):
        return None, None
    release_id, revision = payload.get("release_id"), payload.get("revision")
    release_id = release_id if isinstance(release_id, str) else None
    if isinstance(revision, str) and revision:
        return revision, release_id
    # A pointer with no revision names a release-per-build release, whose
    # tag is its release name.
    return release_id, None


def _pointer_currents(
    hub: _Hub,
    head: str,
    pointer_files: Sequence[str],
    moves: Sequence[BuildMove],
    selection: Selection,
    problems: list[str],
) -> tuple[dict[str, str], dict[str, Any], str | None]:
    """Each release's current build from the pointer files, and the default."""
    by_tag = {move.tag: move for move in moves}
    currents: dict[str, str] = {}
    report: dict[str, Any] = {}
    default = None
    for path in pointer_files:
        if path in selection.retired_pointers:
            report[path] = {"retired": True}
            continue
        tag, named_release = _pointer_tag(hub.read(path, head))
        move = by_tag.get(tag) if tag is not None else None
        if move is None:
            problems.append(
                f"{path} names {tag!r}, which this switch does not move; move it "
                f"with --include {tag}=<release>, or pass --retire-pointer {path} "
                "to delete the pointer file with nothing in its place"
            )
            continue
        if named_release is not None and named_release != move.tagged_release:
            problems.append(
                f"{path} names release {named_release}, not {move.tagged_release}"
            )
            continue
        earlier = currents.setdefault(move.release, move.build)
        if earlier != move.build:
            problems.append(
                f"the pointer files name two current builds of {move.release}: "
                f"{earlier} and {move.build}"
            )
        report[path] = {"tag": tag, "release": move.release, "current": move.build}
    for path in _DEFAULT_POINTER_FILES:
        if isinstance(report.get(path), Mapping) and "release" in report[path]:
            default = report[path]["release"]
            break
    for release, build in selection.current.items():
        earlier = currents.setdefault(release, build)
        if earlier != build:
            problems.append(
                f"--current {release}={build} conflicts with the pointer files' "
                f"current build {earlier}"
            )
    return currents, report, default


def _switch_record(
    moves: Sequence[BuildMove],
    currents: Mapping[str, str],
    default: str | None,
    problems: list[str],
) -> ReleasesRecord:
    """The ``releases.json`` the switch writes, through the interface's changes."""
    record = ReleasesRecord.empty()
    for move in moves:
        try:
            record = add_build(
                record,
                release=move.release,
                build=move.build,
                release_role=move.release_role,
                published_at=move.published_at,
                make_current=False,
            )
        except (ReleaseRecordError, ValueError) as error:
            problems.append(f"{move.tag}: {error}")
    for release, build in sorted(currents.items()):
        try:
            record = make_current(record, release=release, build=build)
        except ReleaseRecordError as error:
            problems.append(f"current build {build} of {release}: {error}")
    if default is not None and default not in record.releases:
        problems.append(f"the default {default} is not a release that moves")
        return record
    return replace(record, default=default)


def _publication_in_progress(branch: str) -> str:
    return (
        f"branch {branch} exists, so a tagged publication is in progress: its "
        "build would be in neither releases.json nor a build folder; run the "
        "step again once it has finished, or, if that publication failed, "
        "delete the branch first; if it finished and only its branch was left, "
        f"name the branch with --finished-branch {branch}"
    )


def _staging_branches(
    hub: _Hub, survey: _Survey, selection: Selection, problems: list[str]
) -> dict[str, str]:
    """The staging branches the switch accepts, each with the commit it names.

    A branch is accepted only when ``--finished-branch`` names it, a tag of
    its build exists, and the commit it names is on ``main``: the tagged
    publisher makes the branch before it uploads, tags the build, and only
    then commits to ``main`` and deletes the branch, so a publication still
    running never meets both. Every other staging branch is a problem.
    """
    accepted: dict[str, str] = {}
    problems.extend(
        f"--finished-branch {branch} is not a {_STAGING_BRANCH_PREFIX} branch of "
        f"{hub.repository}"
        for branch in sorted(selection.finished_branches - set(survey.staging_branches))
    )
    for branch, commit in sorted(survey.staging_branches.items()):
        if branch not in selection.finished_branches:
            problems.append(_publication_in_progress(branch))
            continue
        build = branch.removeprefix(_STAGING_BRANCH_PREFIX)
        if build not in survey.tags:
            problems.append(
                f"--finished-branch {branch}: {build} has no tag, so its "
                "publication did not finish"
            )
        elif not hub.on_main(commit):
            problems.append(
                f"--finished-branch {branch}: the commit it names, {commit}, is not "
                "on main, so its publication may still be running"
            )
        else:
            accepted[branch] = commit
    return accepted


def switch_repository(
    api: Any, repository: str, selection: Selection, *, write: bool
) -> dict[str, Any]:
    """The ``switch`` step: write ``releases.json`` and delete the pointer files.

    Without ``write`` it only reads. With it, it refuses before the commit
    unless every moving build has been copied, every pointer file is
    accounted for and the interface's checks pass.
    """
    hub = _Hub(api, repository)
    head = hub.head()
    head_files = hub.files(head)
    _refuse_switched(head_files, repository)
    survey = _survey(hub, selection)
    problems = survey.problems
    finished_branches = _staging_branches(hub, survey, selection, problems)
    for move in survey.moves:
        state, reason = _folder_state(hub, move, head=head, head_files=head_files)
        if state == "missing":
            problems.append(
                f"{move.folder}/ is not a copy of {move.tag}; run the copy step first"
            )
        elif state == "occupied":
            problems.append(f"{move.folder}/ is not a copy of {move.tag}: {reason}")
    pointer_files = sorted(path for path in head_files if is_pointer_file(path))
    unknown_retired = sorted(selection.retired_pointers - set(pointer_files))
    problems.extend(
        f"{path} is not a pointer file of {repository}" for path in unknown_retired
    )
    currents, pointers, pointer_default = _pointer_currents(
        hub, head, pointer_files, survey.moves, selection, problems
    )
    default = selection.default if selection.default is not None else pointer_default
    record = _switch_record(survey.moves, currents, default, problems)
    listed = {
        build_folder(release, build)
        for release, entry in record.releases.items()
        for build in entry.builds
    }
    unlisted = sorted({_build_folder_of(path) for path in head_files} - listed - {None})
    problems.extend(
        f"{folder}/ is a build folder that this switch would not list: move its "
        "tag with this switch, as the copy step did, or delete the folder first"
        for folder in unlisted
    )
    if not problems:
        remaining = [path for path in head_files if path not in pointer_files]
        violations = [
            *check_change(None, record),
            *check_repository(record, head_files, lambda path: hub.read(path, head)),
            # The commit deletes every pointer file, so the rule that current
            # may not change while one exists is checked on what remains.
            *check_commit(
                before_files=remaining,
                added_paths=[RELEASES_RECORD_PATH],
                deleted_paths=pointer_files,
                before=None,
                after=record,
            ),
        ]
        problems.extend(f"{item.code}: {item.message}" for item in violations)
    message = (
        f"Switch to {RELEASES_RECORD_PATH}: list "
        f"{sum(len(entry.builds) for entry in record.releases.values())} builds of "
        f"{len(record.releases)} releases and delete {', '.join(pointer_files) or 'no pointer files'}"
    )
    report: dict[str, Any] = {
        "step": "switch",
        "repository": repository,
        "head": head,
        "builds": [move.summary() for move in survey.moves],
        "skipped": survey.skipped,
        "not_moved": survey.not_moved,
        "pointer_files": pointers,
        "finished_branches": sorted(finished_branches),
        "delete": pointer_files,
        "releases_json": record.to_document(),
        "commit_message": message,
        "problems": problems,
        "written": False,
        "commit": None,
    }
    if not write:
        return report
    if problems:
        raise SwitchRefusedError("nothing was written: " + "; ".join(problems))
    # A tagged publication creates its tag before any commit to main, and a
    # tag-only one never commits to main, so parent_commit would not notice a
    # build tagged after the survey; list the tags again just before the
    # commit, in one request, and compare the IDs the Hub lists. A
    # publication that has checked for releases.json but not yet tagged its
    # build has a staging branch, so refuse that too, unless it is a finished
    # publication's branch that the survey accepted and that has not moved.
    listed_tags, staging_branches = hub.refs()
    new_branches = sorted(
        branch
        for branch, commit in staging_branches.items()
        if finished_branches.get(branch) != commit
    )
    if new_branches:
        raise SwitchRefusedError(
            "nothing was written: "
            + "; ".join(_publication_in_progress(branch) for branch in new_branches)
        )
    if listed_tags != survey.listed_tags:
        raise SwitchRefusedError(
            "nothing was written: the repository's tags changed while the switch "
            "read it, so a build may be missing from the record; run the step "
            "again once nobody is publishing"
        )
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    content = record.serialize()
    ReleasesRecord.parse(content)
    try:
        report["commit"] = hub.commit(
            [
                CommitOperationAdd(
                    path_in_repo=RELEASES_RECORD_PATH, path_or_fileobj=content
                ),
                *(CommitOperationDelete(path_in_repo=path) for path in pointer_files),
            ],
            message=message,
            parent=head,
        )
    # One commit writes the record and deletes the pointer files, so it
    # either landed whole or not at all; a second run refuses if it landed.
    except Exception as error:
        raise SwitchRefusedError(
            f"the switch commit failed ({type(error).__name__}: {error}); run "
            "the step again, which refuses if the commit did land"
        ) from error
    report["written"] = True
    return report


def _pairs(values: Sequence[str], option: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values:
        key, separator, other = value.partition("=")
        if not separator or not key or not other:
            raise SwitchRefusedError(f"{option} takes NAME=VALUE, not {value!r}")
        if pairs.setdefault(key, other) != other:
            raise SwitchRefusedError(f"{option} names {key} twice")
    return pairs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m microcosm.build.build_folder_switch",
        description=(
            "Switch a dataset repository from tags and latest*.json pointer files "
            "to build folders and releases.json: first copy every build into its "
            "folder, then switch. Each step prints what it would do and changes "
            "nothing without --write."
        ),
    )
    parser.add_argument("step", choices=("copy", "switch"))
    parser.add_argument("repository", help="The Hugging Face dataset repository.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="TAG=RELEASE",
        help="Move the release-per-build tag TAG as build TAG of RELEASE.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="TAG",
        help="Leave out a tag that would move automatically.",
    )
    parser.add_argument(
        "--current",
        action="append",
        default=[],
        metavar="RELEASE=BUILD",
        help="Make BUILD the current build of RELEASE, which no pointer file names.",
    )
    parser.add_argument("--default", help="The release releases.json loads by default.")
    parser.add_argument(
        "--retire-pointer",
        action="append",
        default=[],
        metavar="PATH",
        help="Delete this pointer file with nothing in its place.",
    )
    parser.add_argument(
        "--finished-branch",
        action="append",
        default=[],
        metavar="BRANCH",
        help=(
            "Accept this release-staging/ branch, which a finished publication "
            "left behind: the switch checks that its build is tagged and that "
            "the commit it names is on main, and leaves it in place."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Make the commits; without it the step only reads and reports.",
    )
    return parser


def _default_api() -> Any:
    import huggingface_hub

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
    return huggingface_hub.HfApi(token=token)


def main(argv: Sequence[str] | None = None, *, api: Any = None) -> int:
    """Run one step; 0 when it found nothing wrong (and, with ``--write``, wrote)."""
    args = _parser().parse_args(argv)
    if api is None and hub_offline():
        print(
            "refusing to switch: HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set, and "
            "this tool reads the Hub",
            file=sys.stderr,
        )
        return 1
    try:
        selection = Selection(
            include=_pairs(args.include, "--include"),
            exclude=frozenset(args.exclude),
            current=_pairs(args.current, "--current"),
            default=args.default,
            retired_pointers=frozenset(args.retire_pointer),
            finished_branches=frozenset(args.finished_branch),
        )
        step: Callable[..., dict[str, Any]] = (
            copy_builds if args.step == "copy" else switch_repository
        )
        report = step(
            api if api is not None else _default_api(),
            args.repository,
            selection,
            write=args.write,
        )
    except SwitchRefusedError as error:
        print(f"refusing to {args.step}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    if report["problems"]:
        print(
            f"{len(report['problems'])} problem(s); --write would refuse.",
            file=sys.stderr,
        )
        return 1
    if not args.write:
        print("Nothing was written; run again with --write.", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
