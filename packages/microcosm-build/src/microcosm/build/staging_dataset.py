"""Country-neutral staging of a finished build's dataset bundle.

Staging telemetry (:mod:`microcosm.build.staging_v2`) carries reviewed
aggregate JSON only; population files never pass its content policy. The
dataset a build produced is staged separately: every file the build's manifest
registers as an output, plus the manifest and two sidecars, goes into a private
dataset repository under ``<prefix>/<run_id>/`` in one commit, keyed by the same
run id as the telemetry. A staged bundle is inspectable, never published:
nothing here reads or writes a release contract, ``releases/`` or a latest
pointer.

The module knows nothing about a country's manifest beyond the ``outputs``
mapping shape ``{name: {"path", "sha256", "bytes"}}``; callers pass the manifest
filename and any summary fields. Transport failures are recorded, never
raised, and no exception text enters the evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from microcosm.build.staging_storage import HuggingFaceDatasetStorage

__all__ = [
    "DEFAULT_STAGED_DATASET_PREFIX",
    "SHA256SUMS_FILENAME",
    "STAGED_DATASET_CONTRACT_VERSION",
    "STAGED_MANIFEST_FILENAME",
    "STAGED_MANIFEST_SCHEMA",
    "StagedDatasetBundle",
    "StagedDatasetError",
    "StagedFile",
    "delivery_covers_bundle",
    "disabled_staged_dataset",
    "fetch_bundle",
    "local_only_staged_dataset",
    "parse_sha256sums",
    "refresh_sha256sums_entry",
    "sha256_file",
    "stage_bundle",
    "validate_staged_dataset_delivery",
    "write_sidecars",
]

STAGED_DATASET_CONTRACT_VERSION = 1
STAGED_MANIFEST_SCHEMA = "microcosm.staged-dataset.manifest"
STAGED_MANIFEST_FILENAME = "staged_manifest.json"
SHA256SUMS_FILENAME = "sha256sums.txt"
DEFAULT_STAGED_DATASET_PREFIX = "staged"

_MODES = ("local_and_remote", "local_only", "disabled")
_STATUSES = ("uploaded", "already_staged", "failed", "skipped")
_ERROR_CODES = ("UPLOAD_FAILED", "REMOTE_DIFFERS", "REPOSITORY_UNAVAILABLE")
_DELIVERY_KEYS = frozenset(
    {
        "contract_version",
        "mode",
        "repository",
        "prefix",
        "run_id",
        "revision",
        "status",
        "error_code",
        "opt_out_reason",
        "files",
    }
)
_DEFAULT_SUMMARY_KEYS = (
    "build_kind",
    "created_at",
    "git_commit",
    "git_dirty",
    "releasable",
    "failing_gate_ids",
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUMS_LINE = re.compile(r"^(?P<sha>[0-9a-f]{64})  (?P<name>[^/\\\s][^/\\]*)$")


class StagedDatasetError(ValueError):
    """A staged-dataset bundle or delivery record is malformed."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise StagedDatasetError(
            f"{label} must match {_SAFE_ID.pattern}; got {value!r}."
        )
    return value


def _safe_prefix(prefix: object) -> str:
    if not isinstance(prefix, str) or not prefix.strip():
        raise StagedDatasetError("prefix must be a non-empty relative path.")
    parts = prefix.strip("/").split("/")
    for part in parts:
        _safe_identifier(part, "prefix segment")
    return "/".join(parts)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


@dataclass(frozen=True)
class StagedFile:
    """One manifest-registered output, verified against its recorded digest."""

    name: str
    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True)
class StagedDatasetBundle:
    """The files a build's manifest vouches for, ready to stage."""

    run_dir: Path
    run_id: str
    manifest_name: str
    manifest_sha256: str
    manifest_bytes: int
    files: tuple[StagedFile, ...]
    summary: dict[str, Any]

    @classmethod
    def from_manifest(
        cls,
        run_dir: Path | str,
        *,
        run_id: str,
        manifest_name: str,
        outputs_key: str = "outputs",
        summary_keys: Sequence[str] = _DEFAULT_SUMMARY_KEYS,
        extra_summary: Mapping[str, Any] | None = None,
    ) -> StagedDatasetBundle:
        """Read the manifest and verify every registered output on disk.

        Only files the manifest registers are eligible; each must sit in
        ``run_dir`` and match its recorded SHA-256 and byte count. Anything
        else in the directory (logs, spools, checkpoints) is never staged.
        """

        directory = Path(run_dir).expanduser().resolve()
        run_id = _safe_identifier(run_id, "run_id")
        manifest_path = directory / manifest_name
        if not manifest_path.is_file():
            raise StagedDatasetError(f"manifest missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise StagedDatasetError(
                f"manifest is not valid JSON: {manifest_path}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise StagedDatasetError("manifest must be a JSON object.")
        outputs = manifest.get(outputs_key)
        if not isinstance(outputs, Mapping) or not outputs:
            raise StagedDatasetError(
                f"manifest registers no {outputs_key!r}; nothing to stage."
            )
        reserved = {manifest_name, STAGED_MANIFEST_FILENAME, SHA256SUMS_FILENAME}
        files: list[StagedFile] = []
        seen: set[str] = set()
        for key in sorted(outputs):
            entry = outputs[key]
            if not isinstance(entry, Mapping):
                raise StagedDatasetError(f"{outputs_key}.{key} must be an object.")
            name = PurePosixPath(str(entry.get("path") or "")).name
            if not name or name in reserved:
                raise StagedDatasetError(
                    f"{outputs_key}.{key} names an ineligible file: {name!r}."
                )
            if name in seen:
                raise StagedDatasetError(f"{outputs_key} registers {name!r} twice.")
            seen.add(name)
            path = directory / name
            if not path.is_file():
                raise StagedDatasetError(
                    f"{outputs_key}.{key} is missing from {directory}: {name}"
                )
            recorded = entry.get("sha256")
            if not isinstance(recorded, str) or not _SHA256.fullmatch(recorded):
                raise StagedDatasetError(
                    f"{outputs_key}.{key} carries no valid sha256 digest."
                )
            measured = sha256_file(path)
            if measured != recorded:
                raise StagedDatasetError(
                    f"{outputs_key}.{key} ({name}) does not match its manifest "
                    "digest; refusing to stage mixed bytes."
                )
            size = int(path.stat().st_size)
            recorded_bytes = entry.get("bytes")
            if (
                isinstance(recorded_bytes, int)
                and not isinstance(recorded_bytes, bool)
                and recorded_bytes != size
            ):
                raise StagedDatasetError(
                    f"{outputs_key}.{key} ({name}) byte count differs from the "
                    "manifest."
                )
            files.append(StagedFile(name=name, path=path, sha256=measured, bytes=size))
        summary = {key: manifest[key] for key in summary_keys if key in manifest}
        if extra_summary:
            summary.update(dict(extra_summary))
        return cls(
            run_dir=directory,
            run_id=run_id,
            manifest_name=manifest_name,
            manifest_sha256=sha256_file(manifest_path),
            manifest_bytes=int(manifest_path.stat().st_size),
            files=tuple(files),
            summary=_jsonable(summary),
        )

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / self.manifest_name

    def digests(self) -> dict[str, dict[str, Any]]:
        return {
            item.name: {"sha256": item.sha256, "bytes": item.bytes}
            for item in self.files
        }

    def remote_prefix(self, prefix: str = DEFAULT_STAGED_DATASET_PREFIX) -> str:
        return f"{_safe_prefix(prefix)}/{self.run_id}"

    def staged_manifest(
        self,
        *,
        repository: str | None,
        prefix: str = DEFAULT_STAGED_DATASET_PREFIX,
        telemetry: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The self-describing inventory written beside the bundle."""

        return {
            "schema_name": STAGED_MANIFEST_SCHEMA,
            "schema_version": STAGED_DATASET_CONTRACT_VERSION,
            "run_id": self.run_id,
            "repository": repository,
            "prefix": self.remote_prefix(prefix),
            "manifest": {
                "name": self.manifest_name,
                "sha256": self.manifest_sha256,
                "bytes": self.manifest_bytes,
            },
            "files": self.digests(),
            "summary": dict(self.summary),
            "telemetry": None if telemetry is None else _jsonable(dict(telemetry)),
        }


def write_sidecars(
    bundle: StagedDatasetBundle,
    *,
    repository: str | None,
    prefix: str = DEFAULT_STAGED_DATASET_PREFIX,
    telemetry: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``staged_manifest.json`` then ``sha256sums.txt`` into the run dir.

    The sums file lists every output, the manifest as it will be uploaded, and
    the staged manifest, so a fetched bundle verifies itself.
    """

    staged_manifest_path = bundle.run_dir / STAGED_MANIFEST_FILENAME
    staged_manifest_path.write_bytes(
        _json_bytes(
            bundle.staged_manifest(
                repository=repository, prefix=prefix, telemetry=telemetry
            )
        )
    )
    lines = [(item.sha256, item.name) for item in bundle.files]
    lines.append((bundle.manifest_sha256, bundle.manifest_name))
    lines.append((sha256_file(staged_manifest_path), STAGED_MANIFEST_FILENAME))
    text = "".join(
        f"{sha}  {name}\n" for sha, name in sorted(lines, key=lambda item: item[1])
    )
    sums_path = bundle.run_dir / SHA256SUMS_FILENAME
    sums_path.write_text(text, encoding="utf-8")
    return sums_path, staged_manifest_path


def refresh_sha256sums_entry(run_dir: Path | str, name: str) -> str:
    """Re-digest one listed file so the local sums stay self-verifying.

    The build appends its evidence blocks to the manifest after the sidecars
    were written and uploaded; the remote copy lists the manifest as
    uploaded, the local copy must list the manifest as it now is.
    """

    directory = Path(run_dir)
    sums_path = directory / SHA256SUMS_FILENAME
    entries = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    if name not in {listed for _, listed in entries}:
        raise StagedDatasetError(f"{SHA256SUMS_FILENAME} does not list {name!r}.")
    digest = sha256_file(directory / name)
    text = "".join(
        f"{digest if listed == name else sha}  {listed}\n" for sha, listed in entries
    )
    sums_path.write_text(text, encoding="utf-8")
    return digest


def parse_sha256sums(text: str) -> list[tuple[str, str]]:
    """Parse ``sha256sum`` lines into ``(digest, name)`` pairs."""

    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        match = _SUMS_LINE.fullmatch(line)
        if match is None:
            raise StagedDatasetError(f"malformed sha256sums line: {line!r}")
        name = match.group("name")
        if name in seen:
            raise StagedDatasetError(f"sha256sums lists {name!r} twice.")
        seen.add(name)
        entries.append((match.group("sha"), name))
    if not entries:
        raise StagedDatasetError("sha256sums lists no files.")
    return entries


def validate_staged_dataset_delivery(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a staged-dataset delivery record."""

    if not isinstance(payload, Mapping):
        raise StagedDatasetError("staged_dataset delivery must be an object.")
    record = _jsonable(dict(payload))
    keys = set(record)
    if keys != _DELIVERY_KEYS:
        missing = sorted(_DELIVERY_KEYS - keys)
        extra = sorted(keys - _DELIVERY_KEYS)
        raise StagedDatasetError(
            f"staged_dataset delivery keys differ: missing {missing}, extra {extra}."
        )
    if record["contract_version"] != STAGED_DATASET_CONTRACT_VERSION:
        raise StagedDatasetError("staged_dataset contract_version must be 1.")
    mode = record["mode"]
    status = record["status"]
    if mode not in _MODES:
        raise StagedDatasetError(f"staged_dataset mode {mode!r} is unknown.")
    if status not in _STATUSES:
        raise StagedDatasetError(f"staged_dataset status {status!r} is unknown.")
    repository = record["repository"]
    if repository is not None and (
        not isinstance(repository, str) or not repository.strip()
    ):
        raise StagedDatasetError("staged_dataset repository must be null or non-empty.")
    prefix = record["prefix"]
    if prefix is not None:
        _safe_prefix(prefix)
    run_id = record["run_id"]
    if run_id is not None:
        _safe_identifier(run_id, "staged_dataset run_id")
    revision = record["revision"]
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise StagedDatasetError("staged_dataset revision must be null or non-empty.")
    error_code = record["error_code"]
    if error_code is not None and error_code not in _ERROR_CODES:
        raise StagedDatasetError(f"staged_dataset error_code {error_code!r} unknown.")
    reason = record["opt_out_reason"]
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise StagedDatasetError("staged_dataset opt_out_reason must be non-empty.")
    files = record["files"]
    if not isinstance(files, Mapping):
        raise StagedDatasetError("staged_dataset files must be an object.")
    for name, entry in files.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise StagedDatasetError(f"staged_dataset files name {name!r} invalid.")
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"sha256", "bytes"}
            or not isinstance(entry["sha256"], str)
            or not _SHA256.fullmatch(entry["sha256"])
            or isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] < 0
        ):
            raise StagedDatasetError(f"staged_dataset files entry {name!r} invalid.")
    if mode == "disabled":
        if status != "skipped" or reason is None:
            raise StagedDatasetError(
                "a disabled staged dataset is skipped with an opt-out reason."
            )
        if (
            any(
                record[key] is not None
                for key in ("repository", "prefix", "run_id", "revision", "error_code")
            )
            or files
        ):
            raise StagedDatasetError("a disabled staged dataset records nothing else.")
        return record
    if reason is not None:
        raise StagedDatasetError(
            "only a disabled staged dataset carries an opt-out reason."
        )
    if run_id is None or prefix is None or not files:
        raise StagedDatasetError(
            "an enabled staged dataset names its run_id, prefix and files."
        )
    if mode == "local_only":
        if status != "skipped" or repository is not None or revision is not None:
            raise StagedDatasetError(
                "a local-only staged dataset is skipped without a repository or revision."
            )
        if error_code is not None:
            raise StagedDatasetError("a local-only staged dataset has no error code.")
        return record
    if repository is None:
        raise StagedDatasetError("a remote staged dataset names its repository.")
    if status == "skipped":
        raise StagedDatasetError(
            "a remote staged dataset is uploaded, already staged or failed."
        )
    if status == "failed":
        if error_code is None or revision is not None:
            raise StagedDatasetError(
                "a failed staged dataset carries an error code and no revision."
            )
        return record
    if error_code is not None:
        raise StagedDatasetError("a delivered staged dataset carries no error code.")
    if status == "uploaded" and revision is None:
        raise StagedDatasetError("an uploaded staged dataset records its revision.")
    return record


def delivery_covers_bundle(
    delivery: Any, bundle: StagedDatasetBundle, *, repository: str | None
) -> bool:
    """Whether an existing delivery record already accounts for this bundle.

    True when the record says the same outputs reached ``repository`` (status
    ``uploaded`` or ``already_staged``); a re-stage is then a no-op that must
    keep the record's revision rather than overwrite it.
    """

    if not isinstance(delivery, Mapping) or repository is None:
        return False
    try:
        record = validate_staged_dataset_delivery(delivery)
    except StagedDatasetError:
        return False
    return (
        record["status"] in ("uploaded", "already_staged")
        and record["repository"] == repository
        and record["run_id"] == bundle.run_id
        and _same_digests(record["files"], bundle.digests())
    )


def disabled_staged_dataset(reason: str) -> dict[str, Any]:
    """Explicit evidence for a deliberate staged-dataset opt-out."""

    if not isinstance(reason, str) or not reason.strip():
        raise StagedDatasetError("A staged-dataset opt-out requires a reason.")
    return validate_staged_dataset_delivery(
        {
            "contract_version": STAGED_DATASET_CONTRACT_VERSION,
            "mode": "disabled",
            "repository": None,
            "prefix": None,
            "run_id": None,
            "revision": None,
            "status": "skipped",
            "error_code": None,
            "opt_out_reason": reason.strip(),
            "files": {},
        }
    )


def local_only_staged_dataset(
    bundle: StagedDatasetBundle, *, prefix: str = DEFAULT_STAGED_DATASET_PREFIX
) -> dict[str, Any]:
    """Evidence for a bundle verified and inventoried on disk but not uploaded."""

    return validate_staged_dataset_delivery(
        {
            "contract_version": STAGED_DATASET_CONTRACT_VERSION,
            "mode": "local_only",
            "repository": None,
            "prefix": bundle.remote_prefix(prefix),
            "run_id": bundle.run_id,
            "revision": None,
            "status": "skipped",
            "error_code": None,
            "opt_out_reason": None,
            "files": bundle.digests(),
        }
    )


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr, flush=True)


def stage_bundle(
    bundle: StagedDatasetBundle,
    *,
    storage: HuggingFaceDatasetStorage,
    prefix: str = DEFAULT_STAGED_DATASET_PREFIX,
    message: str | None = None,
) -> dict[str, Any]:
    """Upload the bundle and its sidecars in one commit; record, never raise.

    Idempotent on the outputs' digests: a remote ``staged_manifest.json`` whose
    ``files`` equal the local digests is ``already_staged``; a differing one
    is refused as ``REMOTE_DIFFERS`` so a run id never silently changes
    meaning. Transport failures return ``status: failed`` with a reviewed
    error code; the exception text goes to stderr only.
    """

    remote_prefix = bundle.remote_prefix(prefix)
    sidecars = (
        bundle.run_dir / STAGED_MANIFEST_FILENAME,
        bundle.run_dir / SHA256SUMS_FILENAME,
    )
    for sidecar in sidecars:
        if not sidecar.is_file():
            raise StagedDatasetError(
                f"{sidecar.name} missing; call write_sidecars before stage_bundle."
            )
    record: dict[str, Any] = {
        "contract_version": STAGED_DATASET_CONTRACT_VERSION,
        "mode": "local_and_remote",
        "repository": storage.repo_id,
        "prefix": remote_prefix,
        "run_id": bundle.run_id,
        "revision": None,
        "status": "failed",
        "error_code": None,
        "opt_out_reason": None,
        "files": bundle.digests(),
    }
    remote_manifest = f"{remote_prefix}/{STAGED_MANIFEST_FILENAME}"
    try:
        exists = storage.file_exists(remote_manifest)
    except Exception as error:
        _warn(
            f"staged dataset repository {storage.repo_id} is unavailable "
            f"({type(error).__name__}); the bundle stays local."
        )
        return validate_staged_dataset_delivery(
            {**record, "error_code": "REPOSITORY_UNAVAILABLE"}
        )
    if exists:
        try:
            remote = json.loads(storage.download(remote_manifest))
            head = storage.head_revision()
        except Exception as error:
            _warn(
                f"staged dataset at {remote_prefix} could not be read back "
                f"({type(error).__name__}); the bundle stays local."
            )
            return validate_staged_dataset_delivery(
                {**record, "error_code": "REPOSITORY_UNAVAILABLE"}
            )
        remote_files = remote.get("files") if isinstance(remote, Mapping) else None
        if _same_digests(remote_files, bundle.digests()):
            # The bundle's own commit, not the repository head at re-run time.
            try:
                revision = storage.last_commit(remote_manifest) or head
            except Exception:
                revision = head
            return validate_staged_dataset_delivery(
                {**record, "status": "already_staged", "revision": revision}
            )
        _warn(
            f"{remote_prefix} already holds a different bundle; refusing to "
            "overwrite it. Re-stage under another run id if that is intended."
        )
        return validate_staged_dataset_delivery(
            {**record, "error_code": "REMOTE_DIFFERS"}
        )
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "microcosm-build needs huggingface_hub to stage datasets; "
            "reinstall microcosm-build with its dependencies."
        ) from exc
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{remote_prefix}/{item.name}",
            path_or_fileobj=str(item.path),
        )
        for item in bundle.files
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=f"{remote_prefix}/{bundle.manifest_name}",
            path_or_fileobj=str(bundle.manifest_path),
        )
    )
    operations.extend(
        CommitOperationAdd(
            path_in_repo=f"{remote_prefix}/{sidecar.name}",
            path_or_fileobj=str(sidecar),
        )
        for sidecar in sidecars
    )
    try:
        parent = storage.head_revision()
        revision = storage.commit(
            operations,
            message=message or f"Stage {bundle.run_id} under {remote_prefix}",
            parent_commit=parent,
        )
    except Exception as error:
        _warn(
            f"staged dataset upload failed for {remote_prefix} "
            f"({type(error).__name__}); the bundle stays local and can be "
            "re-staged."
        )
        return validate_staged_dataset_delivery(
            {**record, "error_code": "UPLOAD_FAILED"}
        )
    return validate_staged_dataset_delivery(
        {**record, "status": "uploaded", "revision": revision}
    )


def _same_digests(remote: Any, local: Mapping[str, Mapping[str, Any]]) -> bool:
    """Compare the outputs' digests only.

    The manifest is deliberately excluded: the local copy gains its evidence
    blocks after the upload, so its digest differs from the uploaded copy by
    construction while the dataset it describes is unchanged.
    """

    if not isinstance(remote, Mapping) or set(remote) != set(local):
        return False
    for name, entry in local.items():
        other = remote.get(name)
        if not isinstance(other, Mapping) or other.get("sha256") != entry["sha256"]:
            return False
    return True


def fetch_bundle(
    *,
    storage: HuggingFaceDatasetStorage,
    run_id: str,
    dest: Path | str,
    prefix: str = DEFAULT_STAGED_DATASET_PREFIX,
    revision: str | None = None,
    h5_only: bool = False,
) -> list[Path]:
    """Download a staged bundle into ``dest`` and verify every digest.

    Files are copied out of the Hub cache so each lands under its own name
    (a ``.h5`` path is what the simulation loaders require). A digest
    mismatch removes the file and raises.
    """

    remote_prefix = f"{_safe_prefix(prefix)}/{_safe_identifier(run_id, 'run_id')}"
    destination = Path(dest).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sums_local = storage.download_file(
        f"{remote_prefix}/{SHA256SUMS_FILENAME}", revision=revision
    )
    entries = parse_sha256sums(Path(sums_local).read_text(encoding="utf-8"))
    fetched: list[Path] = []
    for sha, name in entries:
        if h5_only and not name.lower().endswith((".h5", ".hdf5")):
            continue
        local = storage.download_file(f"{remote_prefix}/{name}", revision=revision)
        target = destination / name
        shutil.copyfile(local, target)
        measured = sha256_file(target)
        if measured != sha:
            target.unlink(missing_ok=True)
            raise StagedDatasetError(
                f"{name} does not match the staged digest after download."
            )
        fetched.append(target)
    shutil.copyfile(sums_local, destination / SHA256SUMS_FILENAME)
    return fetched
