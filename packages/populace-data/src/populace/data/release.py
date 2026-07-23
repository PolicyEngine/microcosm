"""Publish a release and point ``latest.json`` at it.

The Hub repo publishes builds under ``releases/<build_id>/``, but nothing
identified which release is *current*: a consumer had to list the tree and
guess, and because build ids end in a date (``populace-us-2024-9f1260b-
20260611``), two builds published the same day have no defined ordering.
``latest.json`` at the repo root is that missing pointer — a tiny manifest
naming the current release and the path of each of its contract files.

Two sides of the pointer live here:

- :func:`publish_release` is the producer: it validates the local release
  directory against the :mod:`release contract <populace.data.contract>`
  (a release that fails the contract refuses to publish), uploads its
  files, and uploads ``latest.json`` **last** — so a reader never sees the
  pointer before the release it points at.
- :func:`latest_release` is the consumer: it downloads ``latest.json`` and
  returns the typed pointer, the one-call answer to "which release is
  current?" for dashboards and scorers.

The Hub client is injected (``api=``) everywhere it is used, so the suite
exercises the real branch, commit, tag, and pointer ordering against a fake —
no network and no mocking of our own internals.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from populace.data.contract import (
    NATIONAL_DEFAULT_DATASET_ROLE,
    release_dataset_role,
    required_release_files,
    validate_release_dir,
)
from populace.data.slack import notify_release

__all__ = [
    "LATEST_POINTER_PATH",
    "LATEST_POINTER_SCHEMA_VERSION",
    "LatestPointer",
    "latest_pointer_payload",
    "publish_release",
    "latest_release",
]

#: Where the pointer lives in the dataset repo. The root, not a release
#: directory: the pointer is repo state, not release state.
LATEST_POINTER_PATH = "latest.json"

#: Version of the pointer payload itself, so the pointer can evolve without
#: consumers guessing (the same discipline the release manifest learned).
LATEST_POINTER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LatestPointer:
    """The parsed ``latest.json``: which release is current, and where.

    Attributes:
        release_id: The current build id (the ``releases/`` directory name).
        updated_at: ISO-8601 UTC timestamp of when the pointer was written.
        paths: Repo-relative path of each contract file, keyed by its stem
            (``"build_manifest"``, ``"release_manifest"``,
            ``"calibration_diagnostics"``, plus country-specific contract
            files such as US source coverage).
    """

    release_id: str
    updated_at: str
    paths: dict[str, str]


def latest_pointer_payload(release_id: str, *, updated_at: str | None = None) -> dict:
    """The ``latest.json`` payload for ``release_id``.

    Paths are derived from the release contract — the pointer names exactly
    the files :func:`~populace.data.contract.required_release_files`
    guarantees exist for this release id.

    Args:
        release_id: The build id being made current.
        updated_at: ISO-8601 UTC timestamp; defaults to now.
    """
    if updated_at is None:
        updated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    return {
        "schema_version": LATEST_POINTER_SCHEMA_VERSION,
        "release_id": release_id,
        "updated_at": updated_at,
        "paths": {
            filename.removesuffix(".json"): f"releases/{release_id}/{filename}"
            for filename in required_release_files(release_id)
        },
    }


def _hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "populace-data needs huggingface_hub to publish releases; "
            "reinstall populace-data with its dependencies."
        ) from exc
    return HfApi()


def publish_release(
    release_dir: Path | str,
    repo_id: str,
    *,
    api=None,
    artifact_root: Path | str | None = None,
    create_tag: bool = True,
    tag_name: str | None = None,
    extra_files: tuple[str, ...] = (),
    updated_at: str | None = None,
    update_latest: bool = True,
    notify: bool = True,
) -> dict:
    """Upload a release directory and point ``latest.json`` at it.

    The order is the guarantee: the release contract is validated first (an
    invalid release never reaches the Hub), then an immutable branch commit is
    created with every release file and artifact and tagged. Only after that
    certificate exists does one atomic main-branch commit update the release
    copies, mutable root conveniences, and ``latest.json`` (the final
    operation). Backends without the branch and atomic-commit surface are
    refused before any remote mutation.

    Args:
        release_dir: Local ``releases/<build_id>`` directory.
        repo_id: Hub dataset repo, e.g. ``"policyengine/populace-us"``.
        api: A ``huggingface_hub.HfApi``-shaped object with branch, atomic
            commit, tag, and branch-deletion methods; constructed lazily when
            omitted. Non-atomic upload-only backends are refused.
        artifact_root: Directory holding root dataset artifacts declared in
            ``release_manifest.json`` (for example ``populace_us_2024.h5``).
            Contract files are always read from ``release_dir`` and uploaded
            under ``releases/<build_id>/``; artifact paths are uploaded to their
            manifest-declared repo paths.
        create_tag: Create an immutable Hub tag for the release snapshot before
            updating main. The tag defaults to the release id. This is required
            when artifact revisions in ``release_manifest.json`` are pinned to
            the release id.
        tag_name: Optional tag name override when ``create_tag=True``.
        extra_files: Additional filenames in ``release_dir`` to upload
            beyond the contract files (e.g. a diagnostics artifact).
        updated_at: Pointer timestamp; defaults to now (UTC).
        notify: Post a best-effort Slack release alert once ``latest.json`` is
            live (no-op unless the country ``SLACK_WEBHOOK_POPULACE_*`` env var
            is set; never fatal). Coupling the alert to the promotion here means
            every publish path announces the release, not just the CLI. Only
            fires when ``update_latest`` is set — a non-default publish moves no
            pointer, so there is no "new latest release" to announce. Set
            ``False`` to suppress it (tests, dry-runs, re-publishes).

    Returns:
        The ``latest.json`` payload that was published.

    Raises:
        ReleaseContractError: If the release directory violates the
            contract. Nothing is uploaded in that case.
        FileNotFoundError: If an ``extra_files`` entry does not exist.
    """
    release_dir = Path(release_dir)
    validate_release_dir(release_dir)
    release_id = release_dir.name
    role = release_dataset_role(release_dir)
    if role != NATIONAL_DEFAULT_DATASET_ROLE and update_latest:
        # populace#398 defense in depth beyond --no-latest: a non-default
        # release can never move the global default pointer, even if a
        # caller asks.
        raise ValueError(
            f"release {release_id!r} declares dataset_role {role!r}; "
            "non-default releases publish immutably (tag only) and can "
            "never update latest.json. Pass update_latest=False "
            "(publish CLI: --no-latest)."
        )
    artifact_root = Path(artifact_root) if artifact_root is not None else None

    contract_files = required_release_files(release_id)
    release_artifacts = _release_manifest_release_artifacts(release_dir)
    filenames = _ordered_unique((*contract_files, *release_artifacts, *extra_files))
    for filename in filenames:
        local = release_dir / filename
        if not local.is_file():
            raise FileNotFoundError(
                f"extra release file {filename!r} not found in {release_dir}."
            )
    root_artifacts = _release_manifest_root_artifacts(release_dir)
    artifact_revisions = _release_manifest_artifact_revisions(release_dir)
    tag = tag_name or release_id
    if release_id in artifact_revisions and not create_tag:
        raise ValueError(
            "release_manifest.json pins artifacts to the release id; "
            "publish_release must create the matching Hugging Face tag before "
            "updating latest.json."
        )
    if tag != release_id and release_id in artifact_revisions:
        raise ValueError(
            "release_manifest.json pins artifacts to the release id; tag_name "
            "must match the release id."
        )
    if root_artifacts and artifact_root is None:
        raise ValueError(
            "release_manifest.json declares root artifacts; pass artifact_root "
            "so publish_release can upload and verify them before latest.json."
        )
    for path_in_repo, expected_sha in root_artifacts.items():
        if artifact_root is None:  # pragma: no cover - guarded above
            continue
        local = artifact_root / path_in_repo
        if not local.is_file():
            raise FileNotFoundError(
                f"release artifact {path_in_repo!r} not found under {artifact_root}."
            )
        observed_sha = _sha256(local)
        if observed_sha != expected_sha:
            raise ValueError(
                f"release artifact {path_in_repo!r} has sha256 {observed_sha} "
                f"but release_manifest.json declares {expected_sha}."
            )

    if api is None:
        api = _hf_api()
    payload = latest_pointer_payload(release_id, updated_at=updated_at)
    if create_tag and not callable(getattr(api, "create_tag", None)):
        raise TypeError(
            "publish_release requires a Hub backend with create_tag support; "
            "release manifests pin artifacts to immutable release tags."
        )
    if not _supports_atomic_publication(api):
        raise TypeError(
            "publish_release requires a Hub backend with create_branch, "
            "create_commit, create_tag, and delete_branch support so release "
            "publication is immutable-first and atomic."
        )
    _publish_atomic(
        api,
        release_dir=release_dir,
        artifact_root=artifact_root,
        repo_id=repo_id,
        release_id=release_id,
        tag=tag,
        filenames=filenames,
        root_artifacts=root_artifacts,
        payload=payload,
        create_tag=create_tag,
        update_latest=update_latest,
    )
    # The pointer is live: announce it. Best-effort and coupled to the promotion
    # so every publish path alerts; warn (don't fail) if the webhook is unset.
    # Skip when no pointer moved — a non-default publish is not a new release.
    if notify and update_latest:
        notify_release(
            repo_id, release_id, payload.get("updated_at"), warn_if_unset=True
        )
    return payload


def _supports_atomic_publication(api: object) -> bool:
    return all(
        callable(getattr(api, method, None))
        for method in ("create_branch", "create_commit", "create_tag", "delete_branch")
    )


def _commit_operations(
    *,
    release_dir: Path,
    artifact_root: Path | None,
    release_id: str,
    filenames: list[str],
    root_artifacts: Mapping[str, str],
    pointer: bytes | None = None,
) -> list:
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "populace-data needs huggingface_hub to publish releases; "
            "reinstall populace-data with its dependencies."
        ) from exc

    operations = [
        CommitOperationAdd(
            path_in_repo=f"releases/{release_id}/{filename}",
            path_or_fileobj=str(release_dir / filename),
        )
        for filename in filenames
    ]
    if artifact_root is not None:
        operations.extend(
            CommitOperationAdd(
                path_in_repo=path_in_repo,
                path_or_fileobj=str(artifact_root / path_in_repo),
            )
            for path_in_repo in root_artifacts
        )
    if pointer is not None:
        operations.append(
            CommitOperationAdd(
                path_in_repo=LATEST_POINTER_PATH,
                path_or_fileobj=pointer,
            )
        )
    return operations


def _repo_revision(api: object, *, repo_id: str) -> str | None:
    repo_info = getattr(api, "repo_info", None)
    if not callable(repo_info):
        return None
    info = repo_info(repo_id=repo_id, repo_type="dataset", revision="main")
    if isinstance(info, Mapping):
        value = info.get("sha")
    else:
        value = getattr(info, "sha", None)
    return str(value) if value else None


def _publish_atomic(
    api: object,
    *,
    release_dir: Path,
    artifact_root: Path | None,
    repo_id: str,
    release_id: str,
    tag: str,
    filenames: list[str],
    root_artifacts: Mapping[str, str],
    payload: dict,
    create_tag: bool,
    update_latest: bool = True,
) -> None:
    staging_branch = f"release-staging/{release_id}"
    main_revision = _repo_revision(api, repo_id=repo_id)
    api.create_branch(
        repo_id=repo_id,
        branch=staging_branch,
        repo_type="dataset",
        revision=main_revision,
    )
    immutable_commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision=staging_branch,
        commit_message=f"Publish immutable release {release_id}",
        operations=_commit_operations(
            release_dir=release_dir,
            artifact_root=artifact_root,
            release_id=release_id,
            filenames=filenames,
            root_artifacts=root_artifacts,
        ),
    )
    immutable_revision = _commit_revision(immutable_commit)
    if immutable_revision is None:
        raise RuntimeError(
            "Hub create_commit returned no revision for immutable release "
            f"{release_id!r}; refusing to update main or latest.json."
        )
    if create_tag:
        _create_release_tag(
            api,
            repo_id=repo_id,
            tag=tag,
            revision=immutable_revision,
        )
    api.delete_branch(
        repo_id=repo_id,
        branch=staging_branch,
        repo_type="dataset",
    )
    if update_latest:
        message = f"Update latest release to {release_id}"
        pointer = json.dumps(payload, indent=1).encode()
    else:
        message = f"Publish non-default release {release_id}"
        pointer = None
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
        parent_commit=main_revision,
        operations=_commit_operations(
            release_dir=release_dir,
            artifact_root=artifact_root,
            release_id=release_id,
            filenames=filenames,
            root_artifacts=root_artifacts,
            pointer=pointer,
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_manifest_root_artifacts(release_dir: Path) -> dict[str, str]:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):  # pragma: no cover - validated already
        return {}
    contract_files = set(required_release_files(release_dir.name))
    root_artifacts: dict[str, str] = {}
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):  # pragma: no cover - validated already
            continue
        path = artifact.get("path")
        sha = artifact.get("sha256")
        if not isinstance(path, str) or not path:
            continue
        if path in contract_files:
            continue
        if path.startswith("releases/"):
            continue
        if (release_dir / path).is_file():
            continue
        if isinstance(sha, str):
            root_artifacts[path] = sha
    return root_artifacts


def _release_manifest_release_artifacts(release_dir: Path) -> tuple[str, ...]:
    """Non-contract release-dir artifacts declared by the release manifest."""
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):  # pragma: no cover - validated already
        return ()
    contract_files = set(required_release_files(release_dir.name))
    filenames: list[str] = []
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):  # pragma: no cover - validated already
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            continue
        filename = path
        release_prefix = f"releases/{release_dir.name}/"
        if filename.startswith(release_prefix):
            filename = filename.removeprefix(release_prefix)
        if filename in contract_files:
            continue
        if "/" in filename:
            continue
        if (release_dir / filename).is_file():
            filenames.append(filename)
    return tuple(filenames)


def _release_manifest_artifact_paths(release_dir: Path) -> dict[str, str]:
    """Root artifact paths from the validated release manifest.

    Contract files have their public home under ``releases/<release_id>/`` and
    are uploaded by filename elsewhere in :func:`publish_release`; this helper
    returns only root dataset artifacts such as H5/NPZ payloads.
    """
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):  # pragma: no cover - validated already
        return {}
    contract_files = set(required_release_files(release_dir.name))
    paths: dict[str, str] = {}
    for key, artifact in artifacts.items():
        if not isinstance(artifact, dict):  # pragma: no cover - validated already
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            continue
        if path in contract_files:
            continue
        if path.startswith("releases/"):
            continue
        paths[str(key)] = path
    return paths


def _ordered_unique(names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _release_manifest_artifact_revisions(release_dir: Path) -> set[str]:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):  # pragma: no cover - validated already
        return set()
    revisions: set[str] = set()
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):  # pragma: no cover - validated already
            continue
        revision = artifact.get("revision")
        if isinstance(revision, str) and revision:
            revisions.add(revision)
    return revisions


def _commit_revision(commit_info: Any) -> str | None:
    """Best-effort revision from ``huggingface_hub`` upload return objects."""
    if commit_info is None:
        return None
    for attr in ("oid", "commit_hash", "commit_id"):
        value = getattr(commit_info, attr, None)
        if value:
            return str(value)
    if isinstance(commit_info, dict):
        for key in ("oid", "commit_hash", "commit_id"):
            value = commit_info.get(key)
            if value:
                return str(value)
    return None


def _create_release_tag(api: object, *, repo_id: str, tag: str, revision: str | None):
    kwargs = {
        "repo_id": repo_id,
        "tag": tag,
        "repo_type": "dataset",
    }
    if revision is not None:
        kwargs["revision"] = revision
    create_tag = getattr(api, "create_tag", None)
    if create_tag is None:
        raise TypeError(
            "publish_release requires a Hub backend with create_tag support."
        )
    return create_tag(**kwargs)


def latest_release(repo_id: str, *, api=None) -> LatestPointer:
    """Read ``latest.json`` from a dataset repo: which release is current.

    Args:
        repo_id: Hub dataset repo, e.g. ``"policyengine/populace-us"``.
        api: A ``huggingface_hub.HfApi``-shaped object (anything with
            ``hf_hub_download(repo_id=, filename=, repo_type=)``);
            constructed lazily when omitted.

    Raises:
        ValueError: If the pointer is malformed or its schema version is
            newer than this library understands.
    """
    if api is None:
        api = _hf_api()
    local = api.hf_hub_download(
        repo_id=repo_id, filename=LATEST_POINTER_PATH, repo_type="dataset"
    )
    payload = json.loads(Path(local).read_text())
    schema_version = payload.get("schema_version")
    if schema_version != LATEST_POINTER_SCHEMA_VERSION:
        raise ValueError(
            f"{LATEST_POINTER_PATH} in {repo_id} has schema_version "
            f"{schema_version!r}; this populace-data reads version "
            f"{LATEST_POINTER_SCHEMA_VERSION}. Upgrade populace-data."
        )
    release_id = payload.get("release_id")
    if not release_id:
        raise ValueError(f"{LATEST_POINTER_PATH} in {repo_id} has no 'release_id'.")
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"{LATEST_POINTER_PATH} in {repo_id} has no 'paths' object.")
    expected_paths = latest_pointer_payload(str(release_id), updated_at="")["paths"]
    observed_paths = {str(key): value for key, value in paths.items()}
    missing_paths = sorted(set(expected_paths) - set(observed_paths))
    unexpected_paths = sorted(set(observed_paths) - set(expected_paths))
    malformed_paths = sorted(
        key
        for key in set(expected_paths) & set(observed_paths)
        if observed_paths[key] != expected_paths[key]
    )
    if missing_paths or unexpected_paths or malformed_paths:
        raise ValueError(
            f"{LATEST_POINTER_PATH} in {repo_id} has incomplete paths: "
            f"missing={missing_paths}, unexpected={unexpected_paths}, "
            f"malformed={malformed_paths}."
        )
    return LatestPointer(
        release_id=str(release_id),
        updated_at=str(payload.get("updated_at", "")),
        paths={str(k): str(v) for k, v in paths.items()},
    )
