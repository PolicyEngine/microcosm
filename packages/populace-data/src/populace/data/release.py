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
exercises the real ordering and payloads against a fake — no network, no
mocking of our own internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from populace.data.contract import (
    REQUIRED_RELEASE_FILES,
    validate_release_dir,
)

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
            ``"sound_ecps_replacement_comparison"``).
    """

    release_id: str
    updated_at: str
    paths: dict[str, str]


def latest_pointer_payload(
    release_id: str, *, updated_at: str | None = None
) -> dict:
    """The ``latest.json`` payload for ``release_id``.

    Paths are derived from the release contract — the pointer names exactly
    the files :data:`~populace.data.contract.REQUIRED_RELEASE_FILES`
    guarantees exist.

    Args:
        release_id: The build id being made current.
        updated_at: ISO-8601 UTC timestamp; defaults to now.
    """
    if updated_at is None:
        updated_at = (
            datetime.now(UTC).replace(microsecond=0).isoformat()
        )
    return {
        "schema_version": LATEST_POINTER_SCHEMA_VERSION,
        "release_id": release_id,
        "updated_at": updated_at,
        "paths": {
            filename.removesuffix(".json"): f"releases/{release_id}/{filename}"
            for filename in REQUIRED_RELEASE_FILES
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
    extra_files: tuple[str, ...] = (),
    updated_at: str | None = None,
) -> dict:
    """Upload a release directory and point ``latest.json`` at it.

    The order is the guarantee: the release contract is validated first (an
    invalid release never reaches the Hub), every release file is uploaded
    next, and the pointer goes up **last**, so a consumer that reads
    ``latest.json`` always finds the files it names.

    Args:
        release_dir: Local ``releases/<build_id>`` directory.
        repo_id: Hub dataset repo, e.g. ``"policyengine/populace-us"``.
        api: A ``huggingface_hub.HfApi``-shaped object (anything with
            ``upload_file(path_or_fileobj=, path_in_repo=, repo_id=,
            repo_type=)``); constructed lazily when omitted.
        extra_files: Additional filenames in ``release_dir`` to upload
            beyond the contract files (e.g. a diagnostics artifact).
        updated_at: Pointer timestamp; defaults to now (UTC).

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
    if api is None:
        api = _hf_api()

    filenames = list(REQUIRED_RELEASE_FILES) + [
        name for name in extra_files if name not in REQUIRED_RELEASE_FILES
    ]
    for filename in filenames:
        local = release_dir / filename
        if not local.is_file():
            raise FileNotFoundError(
                f"extra release file {filename!r} not found in {release_dir}."
            )
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"releases/{release_id}/{filename}",
            repo_id=repo_id,
            repo_type="dataset",
        )

    payload = latest_pointer_payload(release_id, updated_at=updated_at)
    api.upload_file(
        path_or_fileobj=json.dumps(payload, indent=1).encode(),
        path_in_repo=LATEST_POINTER_PATH,
        repo_id=repo_id,
        repo_type="dataset",
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
        raise ValueError(
            f"{LATEST_POINTER_PATH} in {repo_id} has no 'release_id'."
        )
    return LatestPointer(
        release_id=str(release_id),
        updated_at=str(payload.get("updated_at", "")),
        paths={str(k): str(v) for k, v in (payload.get("paths") or {}).items()},
    )
