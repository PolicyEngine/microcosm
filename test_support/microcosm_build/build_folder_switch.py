"""An in-memory dataset repository for the build-folder switch tool's tests.

``FakeDatasetRepository`` answers the ``HfApi`` calls
``microcosm.build.build_folder_switch`` makes, so its tests never reach the
Hub. Every commit keeps the exact bytes of every file; ``*.h5`` and ``*.npz``
files count as LFS files, whose SHA-256 the file information carries, as the
Hub's does. ``create_commit`` applies the real ``huggingface_hub`` operations
(adding bytes, copying a file from another commit, deleting a file) and
refuses a ``parent_commit`` that is no longer ``main``'s head. An annotated
tag is listed, as the Hub lists one, with the ID of its tag object rather
than its commit; a revision names either.

``uk_repository`` builds a small repository in the tagged layout the UK
private repository has before its switch: a June 2023 release-per-build
release that ``latest.json`` names, whose release manifest has a
``metadata`` object both at its top level and in its ``build`` object, two
tagged national builds of which ``latest-national.json`` names the older,
and a tag-only dense build with a checksum ledger, whose tag is annotated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)

UK_REPOSITORY = "policyengine/populace-uk-private"
NATIONAL = "microcosm-uk-2024-25-national"
DENSE = "microcosm-uk-2024-25-dense"
JUNE_2023_TAG = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
OLDER_NATIONAL_BUILD = "uk-frs-calibration-attempt-20261019T083323Z-6cba5d71"
NEWER_NATIONAL_BUILD = "uk-frs-calibration-attempt-20261105T133052Z-51f60255"
DENSE_BUILD = "uk-local-candidate-f100-s42-20261104T093024Z-83802786"
OLDER_NATIONAL_TAG = f"{NATIONAL}-20261019T083323Z-6cba5d71"
NEWER_NATIONAL_TAG = f"{NATIONAL}-20261105T133052Z-51f60255"
DENSE_TAG = f"{DENSE}-20261104T093024Z-83802786"
#: When each tagged commit was made: its build's ``published_at``.
PUBLISHED_AT = {
    JUNE_2023_TAG: datetime(2026, 6, 19, 2, 38, 38, tzinfo=UTC),
    OLDER_NATIONAL_TAG: datetime(2026, 10, 19, 10, 21, 4, 512000, tzinfo=UTC),
    DENSE_TAG: datetime(2026, 11, 4, 13, 6, 25, tzinfo=UTC),
    NEWER_NATIONAL_TAG: datetime(2026, 11, 5, 23, 30, 42, tzinfo=UTC),
}
#: The June 2023 release manifest's two ``metadata`` objects, which differ:
#: the release's at its top level, and the build's in its ``build`` object.
JUNE_2023_RELEASE_METADATA: dict[str, Any] = {
    "region_datasets": {"populace_uk_2023": "national"}
}
JUNE_2023_BUILD_METADATA: dict[str, Any] = {
    "builder": "populace",
    "populace_commit": "4aa4b14cdd8bb693df594d822256a843dd8a8645",
    "private": "UK Data Service licence: private dataset repo.",
}
_LFS_SUFFIXES = (".h5", ".npz")


def sha256(content: bytes) -> str:
    """The SHA-256 of ``content``, as manifests record it."""
    return hashlib.sha256(content).hexdigest()


def dumped(document: Any) -> bytes:
    """A JSON document's bytes as the tagged publisher wrote them."""
    return json.dumps(document, indent=1).encode()


@dataclass
class FakeDatasetRepository:
    """One dataset repository's commits, tags and ``main`` branch, in memory."""

    repository: str
    download_dir: Path
    commits: dict[str, dict[str, bytes]] = field(default_factory=dict)
    dates: dict[str, datetime] = field(default_factory=dict)
    parents: dict[str, str | None] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    #: The tag object of each annotated tag, by tag name.
    tag_objects: dict[str, str] = field(default_factory=dict)
    #: Every branch other than ``main``, such as the tagged publisher's
    #: ``release-staging/<release>``, and the commit it names.
    branches: dict[str, str] = field(default_factory=dict)
    main: str = ""
    #: Every commit the tool made, with its operations, in order.
    made: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.main = self._new_commit(
            {}, parent=None, date=datetime(2026, 6, 1, tzinfo=UTC)
        )

    def _new_commit(
        self, files: dict[str, bytes], *, parent: str | None, date: datetime
    ) -> str:
        commit = f"{len(self.commits):040x}"
        self.commits[commit] = files
        self.parents[commit] = parent
        self.dates[commit] = date
        return commit

    def seed(
        self,
        files: dict[str, bytes],
        *,
        date: datetime,
        tag: str | None = None,
        on_main: bool = True,
    ) -> str:
        """Add a commit on top of ``main``: on ``main``, or only under a tag."""
        commit = self._new_commit(
            {**self.commits[self.main], **files}, parent=self.main, date=date
        )
        if on_main:
            self.main = commit
        if tag is not None:
            self.tags[tag] = commit
        return commit

    def annotate(self, tag: str) -> str:
        """Make ``tag`` an annotated tag: the Hub lists its tag object's ID,
        which, like git's, depends on the commit it names."""
        tag_object = sha256(f"tag object {tag} {self.tags[tag]}".encode())[:40]
        self.tag_objects[tag] = tag_object
        return tag_object

    def move_tag(self, tag: str, commit: str) -> None:
        """Point ``tag`` at another commit, as deleting and creating it again
        does; an annotated tag gets a new tag object."""
        self.tags[tag] = commit
        if tag in self.tag_objects:
            self.annotate(tag)

    def files_at(self, revision: str | None = None) -> dict[str, bytes]:
        """Every file at a revision (``main`` by default), by path."""
        return self.commits[self._commit(revision)]

    def _commit(self, revision: str | None) -> str:
        if revision in (None, "main"):
            return self.main
        if revision in self.tags:
            return self.tags[revision]
        if revision in self.commits:
            return revision
        for tag, tag_object in self.tag_objects.items():
            if revision == tag_object:
                return self.tags[tag]
        raise FileNotFoundError(f"no revision {revision}")

    def _check(self, repo_id: str, repo_type: str | None) -> None:
        assert repo_id == self.repository
        assert repo_type == "dataset"

    # The HfApi calls the switch tool makes ---------------------------------

    def repo_info(
        self, repo_id: str, *, repo_type: str | None = None, revision: str | None = None
    ) -> Any:
        self._check(repo_id, repo_type)
        return SimpleNamespace(sha=self._commit(revision))

    def list_repo_refs(self, repo_id: str, *, repo_type: str | None = None) -> Any:
        self._check(repo_id, repo_type)
        return SimpleNamespace(
            branches=[
                SimpleNamespace(name=name, target_commit=commit)
                for name, commit in {"main": self.main, **self.branches}.items()
            ],
            tags=[
                SimpleNamespace(
                    name=name, target_commit=self.tag_objects.get(name, commit)
                )
                for name, commit in self.tags.items()
            ],
        )

    def list_repo_files(
        self, repo_id: str, *, repo_type: str | None = None, revision: str | None = None
    ) -> list[str]:
        self._check(repo_id, repo_type)
        return sorted(self.files_at(revision))

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str | None = None,
        revision: str | None = None,
    ) -> str:
        self._check(repo_id, repo_type)
        commit = self._commit(revision)
        content = self.commits[commit][filename]
        local = self.download_dir / commit / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        return str(local)

    def get_paths_info(
        self,
        repo_id: str,
        paths: list[str],
        *,
        repo_type: str | None = None,
        revision: str | None = None,
    ) -> list[Any]:
        self._check(repo_id, repo_type)
        files = self.files_at(revision)
        return [
            SimpleNamespace(
                path=path,
                lfs=SimpleNamespace(sha256=sha256(files[path]))
                if path.endswith(_LFS_SUFFIXES)
                else None,
            )
            for path in paths
            if path in files
        ]

    def list_repo_commits(
        self, repo_id: str, *, repo_type: str | None = None, revision: str | None = None
    ) -> list[Any]:
        self._check(repo_id, repo_type)
        history = []
        commit: str | None = self._commit(revision)
        while commit is not None:
            history.append(
                SimpleNamespace(commit_id=commit, created_at=self.dates[commit])
            )
            commit = self.parents[commit]
        return history

    def create_commit(
        self,
        repo_id: str,
        operations: list[Any],
        *,
        commit_message: str,
        repo_type: str | None = None,
        parent_commit: str | None = None,
    ) -> Any:
        self._check(repo_id, repo_type)
        if parent_commit != self.main:
            raise RuntimeError(
                f"412: parent commit {parent_commit} is not main's head {self.main}"
            )
        files = dict(self.commits[self.main])
        recorded = []
        for operation in operations:
            if isinstance(operation, CommitOperationAdd):
                content = operation.path_or_fileobj
                assert isinstance(content, bytes)
                files[operation.path_in_repo] = content
                recorded.append(("add", operation.path_in_repo, None, None))
            elif isinstance(operation, CommitOperationCopy):
                source = self.commits[self._commit(operation.src_revision)]
                files[operation.path_in_repo] = source[operation.src_path_in_repo]
                recorded.append(
                    (
                        "copy",
                        operation.path_in_repo,
                        operation.src_path_in_repo,
                        operation.src_revision,
                    )
                )
            elif isinstance(operation, CommitOperationDelete):
                del files[operation.path_in_repo]
                recorded.append(("delete", operation.path_in_repo, None, None))
            else:  # pragma: no cover - the tool makes no other operation
                raise TypeError(type(operation).__name__)
        commit = self._new_commit(
            files, parent=self.main, date=datetime(2026, 11, 7, tzinfo=UTC)
        )
        self.main = commit
        self.made.append(
            {
                "commit": commit,
                "message": commit_message,
                "parent": parent_commit,
                "operations": recorded,
            }
        )
        return SimpleNamespace(oid=commit)


def _package(name: str, version: str) -> dict[str, str]:
    return {"name": name, "version": version}


def _tagged_manifests(
    *,
    release: str,
    tag: str,
    attempt_id: str | None,
    artifacts: dict[str, tuple[str, str, bytes]],
    dataset_role: str | None = None,
    release_metadata: dict[str, Any] | None = None,
    build_metadata: dict[str, Any] | None = None,
) -> tuple[bytes, bytes]:
    """A tagged build's release and build manifests.

    ``artifacts`` maps each key to its kind, file name and bytes. A dated
    tag's ``build`` names the release as ``build_id`` and carries the
    attempt and the tag; a release-per-build tag's names itself. The two
    ``metadata`` objects go at the top level and in ``build``, as the June
    2023 release has them.
    """
    build: dict[str, Any] = {
        "build_id": release if attempt_id is not None else tag,
        "built_at": "2026-10-19T08:41:02.123456+00:00",
        "built_with_core_package": _package("policyengine-core", "3.33.3"),
        "built_with_model_package": _package("policyengine-uk", "2.174.1"),
    }
    if attempt_id is not None:
        build.update(attempt_id=attempt_id, cut_tag=tag)
    if build_metadata is not None:
        build["metadata"] = build_metadata
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "data_package": _package("microcosm-data", "0.1.0"),
        "build": build,
        "compatible_core_packages": [
            {"name": "policyengine-core", "specifier": "==3.33.3"}
        ],
        "compatible_model_packages": [
            {"name": "policyengine-uk", "specifier": "==2.174.1"}
        ],
        "artifacts": {
            key: {
                "kind": kind,
                "path": name,
                "repo_id": UK_REPOSITORY,
                "revision": tag,
                "sha256": sha256(content),
            }
            for key, (kind, name, content) in artifacts.items()
        },
    }
    if release_metadata is not None:
        manifest["metadata"] = release_metadata
    if dataset_role is None:
        dataset_key = next(
            key for key, (kind, _, _) in artifacts.items() if kind == "microdata"
        )
        manifest["default_datasets"] = {"national": dataset_key}
    else:
        manifest["dataset_role"] = dataset_role
        manifest["default_datasets"] = {}
    build_manifest: dict[str, Any] = {
        "build_id": build["build_id"],
        "code": {"repository": "PolicyEngine/microcosm"},
    }
    if attempt_id is not None:
        build_manifest.update(attempt_id=attempt_id, cut_tag=tag)
    return dumped(manifest), dumped(build_manifest)


def _national_files(tag: str, attempt_id: str, *, dataset: bytes) -> dict[str, bytes]:
    """A tagged national build's files: its release folder's and the root's."""
    folder = f"releases/{NATIONAL}"
    diagnostics = dumped({"schema_version": 9, "build": attempt_id})
    certification = dumped({"kind": "uk_release_certification", "attempt": attempt_id})
    weights = b"weights for " + attempt_id.encode()
    release_manifest, build_manifest = _tagged_manifests(
        release=NATIONAL,
        tag=tag,
        attempt_id=attempt_id,
        artifacts={
            "microcosm_uk_2024_25": ("microdata", "microcosm_uk_2024_25.h5", dataset),
            "microcosm_uk_2024_25_calibration": (
                "calibration",
                "microcosm_uk_2024_25_calibration.npz",
                weights,
            ),
            "calibration_diagnostics": (
                "diagnostics",
                "calibration_diagnostics.json",
                diagnostics,
            ),
            "release_certification": (
                "diagnostics",
                "release_certification.json",
                certification,
            ),
        },
    )
    return {
        f"{folder}/release_manifest.json": release_manifest,
        f"{folder}/build_manifest.json": build_manifest,
        f"{folder}/calibration_diagnostics.json": diagnostics,
        f"{folder}/release_certification.json": certification,
        "microcosm_uk_2024_25.h5": dataset,
        "microcosm_uk_2024_25_calibration.npz": weights,
    }


def _dense_files() -> dict[str, bytes]:
    """The tag-only dense build's files, with its checksum ledger."""
    folder = f"releases/{DENSE}"
    dataset = b"dense dataset"
    diagnostics = dumped({"schema_version": 9, "build": DENSE_BUILD})
    summary = dumped({"gates": "passed"})
    release_manifest, build_manifest = _tagged_manifests(
        release=DENSE,
        tag=DENSE_TAG,
        attempt_id=DENSE_BUILD,
        dataset_role="non_default_local_area",
        artifacts={
            "microcosm_uk_2024_25_dense": (
                "microdata",
                "microcosm_uk_2024_25_dense.h5",
                dataset,
            ),
            "calibration_diagnostics": (
                "diagnostics",
                "calibration_diagnostics.json",
                diagnostics,
            ),
            "gate_summary": ("diagnostics", "gate_summary.json", summary),
        },
    )
    ledger = "".join(
        f"{sha256(content)}  {name}\n"
        for name, content in sorted(
            {
                "build_manifest.json": build_manifest,
                "calibration_diagnostics.json": diagnostics,
                "gate_summary.json": summary,
                "microcosm_uk_2024_25_dense.h5": dataset,
                "release_manifest.json": release_manifest,
            }.items()
        )
    ).encode()
    return {
        f"{folder}/release_manifest.json": release_manifest,
        f"{folder}/build_manifest.json": build_manifest,
        f"{folder}/calibration_diagnostics.json": diagnostics,
        f"{folder}/gate_summary.json": summary,
        f"{folder}/sha256sums.txt": ledger,
        "microcosm_uk_2024_25_dense.h5": dataset,
    }


def _june_2023_files() -> dict[str, bytes]:
    """The June 2023 release-per-build release's files."""
    folder = f"releases/{JUNE_2023_TAG}"
    dataset = b"2023 dataset"
    diagnostics = dumped({"schema_version": 6})
    release_manifest, build_manifest = _tagged_manifests(
        release=JUNE_2023_TAG,
        tag=JUNE_2023_TAG,
        attempt_id=None,
        artifacts={
            "populace_uk_2023": ("microdata", "populace_uk_2023.h5", dataset),
            "calibration_diagnostics": (
                "diagnostics",
                "calibration_diagnostics.json",
                diagnostics,
            ),
        },
        release_metadata=JUNE_2023_RELEASE_METADATA,
        build_metadata=JUNE_2023_BUILD_METADATA,
    )
    return {
        f"{folder}/release_manifest.json": release_manifest,
        f"{folder}/build_manifest.json": build_manifest,
        f"{folder}/calibration_diagnostics.json": diagnostics,
        "populace_uk_2023.h5": dataset,
    }


def _pointer(release_id: str, *, revision: str | None = None) -> bytes:
    payload: dict[str, Any] = {"schema_version": 1, "release_id": release_id}
    if revision is not None:
        payload.update(line="national", revision=revision)
    return dumped(payload)


def uk_repository(download_dir: Path) -> FakeDatasetRepository:
    """The UK private repository's tagged layout, before its switch.

    ``main`` holds the newer national build's files and its release folder,
    which still has the older build's ``score_vs_enhanced_frs.json``;
    ``latest-national.json`` names the older national build, ``latest.json``
    the June 2023 release, and the dense build is only under its tag.
    """
    hub = FakeDatasetRepository(UK_REPOSITORY, download_dir)
    hub.seed(
        {".gitattributes": b"*.h5 filter=lfs\n"}, date=datetime(2026, 6, 1, tzinfo=UTC)
    )
    hub.seed(
        {**_june_2023_files(), "latest.json": _pointer(JUNE_2023_TAG)},
        date=PUBLISHED_AT[JUNE_2023_TAG],
        tag=JUNE_2023_TAG,
    )
    older = _national_files(
        OLDER_NATIONAL_TAG, OLDER_NATIONAL_BUILD, dataset=b"older national dataset"
    )
    older[f"releases/{NATIONAL}/score_vs_enhanced_frs.json"] = b"{}"
    hub.seed(older, date=PUBLISHED_AT[OLDER_NATIONAL_TAG], tag=OLDER_NATIONAL_TAG)
    hub.seed(
        {
            "latest-national.json": _pointer(NATIONAL, revision=OLDER_NATIONAL_TAG),
        },
        date=datetime(2026, 10, 26, 14, 5, 59, tzinfo=UTC),
    )
    hub.seed(_dense_files(), date=PUBLISHED_AT[DENSE_TAG], tag=DENSE_TAG, on_main=False)
    # As the Hub lists microcosm-uk-2024-25-dense-20261026T105309Z-c9df4b86.
    hub.annotate(DENSE_TAG)
    hub.seed(
        _national_files(
            NEWER_NATIONAL_TAG, NEWER_NATIONAL_BUILD, dataset=b"newer national dataset"
        ),
        date=PUBLISHED_AT[NEWER_NATIONAL_TAG],
        tag=NEWER_NATIONAL_TAG,
    )
    return hub
