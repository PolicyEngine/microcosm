"""Publish a release and point ``latest.json`` at it.

The Hub repo publishes builds under ``releases/<build_id>/``, but nothing
identified which release is *current*: a consumer had to list the tree and
guess, and because build ids end in a date (``populace-us-2024-9f1260b-
20260611``), two builds published the same day have no defined ordering.
``latest.json`` at the repo root is that missing pointer — a tiny manifest
naming the current release and the path of each of its contract files.

Two sides of the pointer live here:

- :func:`publish_release` is the producer: it validates the local release
  directory against the :mod:`release contract <microcosm.data.contract>`
  (a release that fails the contract refuses to publish), uploads its
  files, and either stops at an immutable tag (the exact-k candidate lane) or
  uploads ``latest.json`` **last** — so a reader never sees the pointer before
  the release it points at.
- :func:`latest_release` is the consumer: it downloads ``latest.json`` and
  returns the typed pointer, the one-call answer to "which release is
  current?" for dashboards and scorers.

The EVIDENCE tier (microcosm#506) publishes through the same producer with
``evidence=True``: identical immutable-tag mechanics, but the pointer that
moves is ``latest-evidence.json`` — structurally never ``latest.json`` —
and :func:`latest_evidence_release` is its consumer.

The Hub client is injected (``api=``) everywhere it is used, so the suite
exercises the real branch, commit, tag, and pointer ordering against a fake —
no network and no mocking of our own internals.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microcosm.data.contract import (
    EVIDENCE_RELEASE_ID_SEGMENT,
    NATIONAL_DEFAULT_DATASET_ROLE,
    release_dataset_role,
    required_release_files,
    validate_evidence_release_dir,
    validate_release_dir,
)
from microcosm.data.slack import notify_release

__all__ = [
    "LATEST_EVIDENCE_POINTER_PATH",
    "LATEST_POINTER_PATH",
    "LATEST_POINTER_SCHEMA_VERSION",
    "RELEASE_TIER_CERTIFIED",
    "RELEASE_TIER_EVIDENCE",
    "LatestPointer",
    "latest_evidence_pointer_payload",
    "latest_pointer_payload",
    "publish_release",
    "latest_release",
    "latest_evidence_release",
]

#: Where the pointer lives in the dataset repo. The root, not a release
#: directory: the pointer is repo state, not release state.
LATEST_POINTER_PATH = "latest.json"

#: The evidence-tier pointer (microcosm#506): which evidence release is the
#: best *current* one. A separate file at the repo root, so certified
#: consumers reading ``latest.json`` (and the pe.py certification path built
#: on it) can never pick up an evidence artifact by accident.
LATEST_EVIDENCE_POINTER_PATH = "latest-evidence.json"

#: Version of the pointer payload itself, so the pointer can evolve without
#: consumers guessing (the same discipline the release manifest learned).
LATEST_POINTER_SCHEMA_VERSION = 1

#: Publication tiers (microcosm#506). Certified is the default everywhere;
#: the evidence tier is opted into explicitly and carries its tier in the
#: pointer payload, the release id, and the release manifest.
RELEASE_TIER_CERTIFIED = "certified"
RELEASE_TIER_EVIDENCE = "evidence"


@dataclass(frozen=True)
class LatestPointer:
    """A parsed release pointer: which release is current, and where.

    Attributes:
        release_id: The current build id (the ``releases/`` directory name).
        updated_at: ISO-8601 UTC timestamp of when the pointer was written.
        paths: Repo-relative path of each contract file, keyed by its stem
            (``"build_manifest"``, ``"release_manifest"``,
            ``"calibration_diagnostics"``, plus country-specific contract
            files such as US source coverage).
        tier: The publication tier the pointer names —
            :data:`RELEASE_TIER_CERTIFIED` for ``latest.json`` (whose payload
            predates tiers and carries no field), or
            :data:`RELEASE_TIER_EVIDENCE` for ``latest-evidence.json``.
    """

    release_id: str
    updated_at: str
    paths: dict[str, str]
    tier: str = RELEASE_TIER_CERTIFIED


def latest_pointer_payload(release_id: str, *, updated_at: str | None = None) -> dict:
    """The ``latest.json`` payload for ``release_id``.

    Paths are derived from the release contract — the pointer names exactly
    the files :func:`~microcosm.data.contract.required_release_files`
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


def latest_evidence_pointer_payload(
    release_id: str, *, updated_at: str | None = None
) -> dict:
    """The ``latest-evidence.json`` payload for ``release_id``.

    Mirrors :func:`latest_pointer_payload` exactly, plus a ``tier`` field —
    so evidence consumers reuse certified pointer tooling, while a reader
    that lands on the wrong file sees the tier immediately.
    """
    return {
        **latest_pointer_payload(release_id, updated_at=updated_at),
        "tier": RELEASE_TIER_EVIDENCE,
    }


def _hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "microcosm-data needs huggingface_hub to publish releases; "
            "reinstall microcosm-data with its dependencies."
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
    tag_only: bool = False,
    notify: bool = True,
    evidence: bool = False,
) -> dict:
    """Publish a release directory and optionally point ``latest.json`` at it.

    The order is the guarantee: the release contract is validated first (an
    invalid release never reaches the Hub), then an immutable branch commit is
    created with every release file and artifact and tagged. A tag-only publish
    stops there. Otherwise, only after that certificate exists does one atomic
    main-branch commit update the release copies, mutable root conveniences,
    and, for standard publication, ``latest.json`` (the final operation).
    Backends without the branch and atomic-commit surface are refused before
    any remote mutation.

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
        update_latest: Update the production ``latest.json`` pointer after the
            immutable release tag is created. ``False`` preserves the legacy
            non-default publication behavior: release copies and root artifacts
            are still committed to main, but the pointer is not.
        tag_only: Publish only the immutable tagged revision, with no main-branch
            commit. Requires ``update_latest=False`` and ``create_tag=True``.
            This is the exact-k candidate lane; it is separate from legacy
            ``update_latest=False`` publication so existing non-default releases
            retain their main-branch copies.
        notify: Post a best-effort Slack release alert once ``latest.json`` is
            live (no-op unless the country ``SLACK_WEBHOOK_MICROCOSM_*`` env var
            is set; never fatal). Coupling the alert to the promotion here means
            every publish path announces the release, not just the CLI. Only
            fires when ``update_latest`` is set — a non-default publish moves no
            pointer, so there is no "new latest release" to announce. Set
            ``False`` to suppress it (tests, dry-runs, re-publishes).
        evidence: Publish at the EVIDENCE tier (microcosm#506). The release
            is validated against
            :func:`~microcosm.data.contract.validate_evidence_release_dir`
            instead of the certified contract, and the pointer that moves is
            ``latest-evidence.json`` — this path is structurally incapable of
            writing ``latest.json``, so an evidence artifact can never become
            the certified default or feed pe.py certification. Tag and upload
            mechanics are otherwise identical. ``update_latest`` then governs
            the evidence pointer.

    Returns:
        The release's pointer payload (``latest.json`` shape, plus a ``tier``
        field at the evidence tier). It is uploaded only when
        ``update_latest=True``.

    Raises:
        ReleaseContractError: If the release directory violates its tier's
            contract. Nothing is uploaded in that case.
        FileNotFoundError: If an ``extra_files`` entry does not exist.
    """
    release_dir = Path(release_dir)
    if evidence:
        validate_evidence_release_dir(release_dir)
    else:
        validate_release_dir(release_dir)
    release_id = release_dir.name
    role = release_dataset_role(release_dir)
    if role != NATIONAL_DEFAULT_DATASET_ROLE and update_latest:
        # microcosm#398 defense in depth beyond --no-latest: a non-default
        # release can never move the global default pointer, even if a
        # caller asks.
        raise ValueError(
            f"release {release_id!r} declares dataset_role {role!r}; "
            "non-default releases publish immutably (tag only) and can "
            "never update latest.json. Pass update_latest=False "
            "(publish CLI: --no-latest)."
        )
    if tag_only and update_latest:
        raise ValueError(
            "tag_only=True requires update_latest=False; a tag-only publication "
            "cannot update latest.json."
        )
    if tag_only and not create_tag:
        raise ValueError(
            "tag_only=True requires create_tag=True; deleting the staging branch "
            "without a tag would leave no published revision."
        )
    artifact_root = Path(artifact_root) if artifact_root is not None else None

    if role == NATIONAL_DEFAULT_DATASET_ROLE:
        contract_files = required_release_files(release_id)
    else:
        # A non-default release's directory IS its bundle: publishing a
        # subset would leave a remote release that fails its own role
        # contract (the checksum ledger and sidecars are required files).
        contract_files = tuple(
            sorted(path.name for path in release_dir.iterdir() if path.is_file())
        )
    release_artifacts = _release_manifest_release_artifacts(release_dir)
    filenames = _ordered_unique((*contract_files, *release_artifacts, *extra_files))
    for filename in filenames:
        # Every release-dir upload lands at releases/<id>/<filename> — a name
        # carrying path components ('../../latest.json') could escape that
        # prefix once the Hub canonicalizes the path. Bare file names only.
        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise ValueError(
                f"release file name {filename!r} must be a bare file name; "
                "path components cannot ride into the release upload."
            )
        local = release_dir / filename
        if not local.is_file():
            raise FileNotFoundError(
                f"extra release file {filename!r} not found in {release_dir}."
            )
    root_artifacts = _release_manifest_root_artifacts(release_dir)
    # Root artifacts upload at their manifest-declared repo paths — the one
    # surface where a manifest author could smuggle a pointer write past the
    # tier's pointer selection. Two layers, both on BOTH tiers: the declared
    # path must already be in canonical clean relative form (so './latest.json'
    # or 'x/../latest.json' cannot dodge a raw-string comparison and be
    # canonicalized by the Hub afterwards), and the canonical pointer paths
    # are reserved outright — pointers move only via the publisher's own
    # pointer operation.
    for path_in_repo in root_artifacts:
        if (
            "\\" in path_in_repo
            or path_in_repo.startswith("/")
            or posixpath.normpath(path_in_repo) != path_in_repo
        ):
            raise ValueError(
                f"release_manifest.json root artifact path {path_in_repo!r} "
                "is not a clean relative POSIX path; refusing to upload it."
            )
    pointer_collisions = sorted(
        {LATEST_POINTER_PATH, LATEST_EVIDENCE_POINTER_PATH} & set(root_artifacts)
    )
    if pointer_collisions:
        raise ValueError(
            "release_manifest.json declares root artifact(s) at reserved "
            f"pointer path(s) {pointer_collisions}; latest.json and "
            "latest-evidence.json are written only by the publisher itself, "
            "never as release artifacts."
        )
    artifact_revisions = _release_manifest_artifact_revisions(release_dir)
    tag = tag_name or release_id
    if artifact_revisions and not create_tag:
        raise ValueError(
            "release_manifest.json pins artifacts to revisions; "
            "publish_release must create the matching Hugging Face tag before "
            "updating latest.json."
        )
    if artifact_revisions and artifact_revisions != {tag}:
        raise ValueError(
            "release_manifest.json pins artifacts to revisions; tag_name must "
            "match the release id or uniform per-cut artifact revision."
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
    if evidence:
        payload = latest_evidence_pointer_payload(release_id, updated_at=updated_at)
    else:
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
        tag_only=tag_only,
        evidence=evidence,
    )
    # The pointer is live: announce it. Best-effort and coupled to the promotion
    # so every publish path alerts; warn (don't fail) if the webhook is unset.
    # Skip when no pointer moved — a non-default publish is not a new release.
    if notify and update_latest:
        notify_kwargs: dict = {"warn_if_unset": True}
        if evidence:
            # The alert must never read as a certified release announcement.
            notify_kwargs["tier"] = RELEASE_TIER_EVIDENCE
        notify_release(repo_id, release_id, payload.get("updated_at"), **notify_kwargs)
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
    pointer_path: str = LATEST_POINTER_PATH,
) -> list:
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "microcosm-data needs huggingface_hub to publish releases; "
            "reinstall microcosm-data with its dependencies."
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
                path_in_repo=pointer_path,
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
    tag_only: bool = False,
    evidence: bool = False,
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
    if tag_only:
        # The release-id tag points directly at immutable_revision, so deleting
        # the temporary branch does not make the candidate unreachable. Exact-k
        # candidates deliberately stop here: neither canonical root artifacts
        # nor release-directory copies are written to main.
        return
    # The evidence tier writes ONLY its own pointer file: the certified
    # ``latest.json`` path never appears in an evidence commit, so no bug in
    # flag-plumbing can promote an evidence artifact to certified default.
    tier_label = "evidence release" if evidence else "release"
    pointer_path = LATEST_EVIDENCE_POINTER_PATH if evidence else LATEST_POINTER_PATH
    if update_latest:
        message = f"Update latest {tier_label} to {release_id}"
        pointer = json.dumps(payload, indent=1).encode()
    else:
        message = f"Publish non-default {tier_label} {release_id}"
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
            pointer_path=pointer_path,
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


def _read_pointer(repo_id: str, api, *, pointer_path: str) -> dict:
    """Download and structurally validate a release pointer file."""
    if api is None:
        api = _hf_api()
    local = api.hf_hub_download(
        repo_id=repo_id, filename=pointer_path, repo_type="dataset"
    )
    payload = json.loads(Path(local).read_text())
    schema_version = payload.get("schema_version")
    if schema_version != LATEST_POINTER_SCHEMA_VERSION:
        raise ValueError(
            f"{pointer_path} in {repo_id} has schema_version "
            f"{schema_version!r}; this microcosm-data reads version "
            f"{LATEST_POINTER_SCHEMA_VERSION}. Upgrade microcosm-data."
        )
    release_id = payload.get("release_id")
    if not release_id:
        raise ValueError(f"{pointer_path} in {repo_id} has no 'release_id'.")
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"{pointer_path} in {repo_id} has no 'paths' object.")
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
            f"{pointer_path} in {repo_id} has incomplete paths: "
            f"missing={missing_paths}, unexpected={unexpected_paths}, "
            f"malformed={malformed_paths}."
        )
    return payload


def latest_release(repo_id: str, *, api=None) -> LatestPointer:
    """Read ``latest.json`` from a dataset repo: which release is current.

    Args:
        repo_id: Hub dataset repo, e.g. ``"policyengine/populace-us"``.
        api: A ``huggingface_hub.HfApi``-shaped object (anything with
            ``hf_hub_download(repo_id=, filename=, repo_type=)``);
            constructed lazily when omitted.

    Raises:
        ValueError: If the pointer is malformed, its schema version is newer
            than this library understands, or it carries a ``tier`` field at
            all — the certified payload predates tiers and no certified
            producer writes one, so any tier field (even ``"certified"``) is
            foreign and must never be consumed as the certified default.
    """
    payload = _read_pointer(repo_id, api, pointer_path=LATEST_POINTER_PATH)
    if "tier" in payload:
        raise ValueError(
            f"{LATEST_POINTER_PATH} in {repo_id} carries a 'tier' field "
            f"({payload.get('tier')!r}); the certified pointer never does — "
            f"evidence releases live at {LATEST_EVIDENCE_POINTER_PATH}."
        )
    return LatestPointer(
        release_id=str(payload["release_id"]),
        updated_at=str(payload.get("updated_at", "")),
        paths={str(k): str(v) for k, v in payload["paths"].items()},
        tier=RELEASE_TIER_CERTIFIED,
    )


def latest_evidence_release(repo_id: str, *, api=None) -> LatestPointer:
    """Read ``latest-evidence.json``: the best *current* evidence release.

    The evidence-tier sibling of :func:`latest_release` (microcosm#506) — how
    consumers discover the best available artifact when no certified release
    carries it yet. Each evidence publish supersedes the last, so this
    pointer always names the current one.

    Args:
        repo_id: Hub dataset repo, e.g. ``"policyengine/populace-us"``.
        api: A ``huggingface_hub.HfApi``-shaped object (anything with
            ``hf_hub_download(repo_id=, filename=, repo_type=)``);
            constructed lazily when omitted.

    Raises:
        ValueError: If the pointer is malformed, does not declare the
            evidence tier, or names a release id without the
            ``-evidence-`` segment.
    """
    payload = _read_pointer(repo_id, api, pointer_path=LATEST_EVIDENCE_POINTER_PATH)
    tier = payload.get("tier")
    if tier != RELEASE_TIER_EVIDENCE:
        raise ValueError(
            f"{LATEST_EVIDENCE_POINTER_PATH} in {repo_id} declares tier "
            f"{tier!r}; the evidence pointer must declare "
            f"{RELEASE_TIER_EVIDENCE!r}."
        )
    release_id = str(payload["release_id"])
    if EVIDENCE_RELEASE_ID_SEGMENT not in release_id:
        raise ValueError(
            f"{LATEST_EVIDENCE_POINTER_PATH} in {repo_id} names release "
            f"{release_id!r}, which does not carry the "
            f"{EVIDENCE_RELEASE_ID_SEGMENT!r} segment."
        )
    return LatestPointer(
        release_id=release_id,
        updated_at=str(payload.get("updated_at", "")),
        paths={str(k): str(v) for k, v in payload["paths"].items()},
        tier=RELEASE_TIER_EVIDENCE,
    )
