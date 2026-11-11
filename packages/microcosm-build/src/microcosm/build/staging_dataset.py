"""Country-neutral staging of a finished build's dataset bundle.

Staging telemetry (:mod:`microcosm.build.staging_v2`) carries reviewed
aggregate JSON only; population files never pass its content policy. The
dataset a build produced is staged separately: every file the build's manifest
registers as an output, plus the manifest and two sidecars, goes into a private
dataset repository under ``<prefix>/<run_id>/`` in one commit, keyed by the same
run id as the telemetry. A staged bundle is inspectable, never published:
nothing here reads or writes a release contract, ``releases/`` or a latest
pointer.

Microcosm verifies the bundle and writes the sidecars; the build's emitter
uploads it. :func:`stage_bundle` hands every file over by path with its role,
SHA-256 and size (``stage_dataset_bundle``), waits for the emitter's first
upload attempt, and turns the emitter's receipt into the build's
``staged_dataset`` delivery record. The emitter copies the bundle's small
files before it answers; it uploads every other file from disk, so the build
rewrites a bundle file afterwards only when the receipt lists it in
``copied_files`` (:meth:`StagedBundleHandOver.may_rewrite`).

The module knows nothing about a country's manifest beyond the ``outputs``
mapping shape ``{name: {"path", "sha256", "bytes"}}``; callers pass the manifest
filename and any summary fields. Hand-over failures are recorded, never
raised, and no exception text enters the evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, TypedDict, cast, get_args

from microcosm_emitter_interface import BundleFile, StageDatasetBundle
from microcosm_emitter_interface.bundles import (
    BUILD_MANIFEST_ROLE,
    BUNDLE_INVENTORY_ROLE,
    CHECKSUMS_ROLE,
    SIDECAR_ROLES,
    DatasetBundleReceipt,
    unavailable_bundle_receipt,
)
from microcosm_emitter_interface.identifiers import repository_identifier

from microcosm.build.staging_storage import HuggingFaceDatasetStorage
from microcosm.build.telemetry_emitter import BundleEmitter, emitter_problem

__all__ = [
    "DEFAULT_STAGED_DATASET_PREFIX",
    "AT_BUILD_RECORD_KEY",
    "HANDOVER_RECORD_KEY",
    "RECEIPT_POLL_SECONDS",
    "RESTAGE_WAIT_SECONDS",
    "RETRYING_ERROR_CODES",
    "SHA256SUMS_FILENAME",
    "STAGED_DATASET_CONTRACT_VERSION",
    "STAGED_DATASET_WAIT_SECONDS",
    "STAGED_MANIFEST_FILENAME",
    "STAGED_MANIFEST_SCHEMA",
    "BundleEmitter",
    "StagedBundleHandOver",
    "StagedDatasetBundle",
    "StagedDatasetDelivery",
    "StagedDatasetError",
    "StagedDatasetErrorCode",
    "StagedFile",
    "at_build_record",
    "bundle_request",
    "delivery_covers_bundle",
    "disabled_staged_dataset",
    "fetch_bundle",
    "handover_record",
    "hub_offline",
    "local_only_staged_dataset",
    "output_role",
    "parse_sha256sums",
    "recorded_handover",
    "refresh_sha256sums_entry",
    "resume_hand_over",
    "sha256_file",
    "stage_bundle",
    "validate_staged_dataset_delivery",
    "write_access_problem",
    "write_sidecars",
]

STAGED_DATASET_CONTRACT_VERSION = 1
STAGED_MANIFEST_SCHEMA = "microcosm.staged-dataset.manifest"
STAGED_MANIFEST_FILENAME = "staged_manifest.json"
SHA256SUMS_FILENAME = "sha256sums.txt"
DEFAULT_STAGED_DATASET_PREFIX = "staged"
#: How long a build waits, after handing its bundle to the emitter, for the
#: emitter's first upload attempt to finish. It bounds only a hand-over that
#: never settles: an attempt that hangs, or never starts because another
#: emitter on the machine holds the repository. A healthy upload of the
#: largest UK bundle recorded (about 2.4 GB) takes about 16 minutes at
#: 20 Mbit/s. After it the build records the bundle as failed with
#: ``UPLOAD_PENDING``; the emitter keeps it queued.
STAGED_DATASET_WAIT_SECONDS = 1800.0
#: The re-stage command's default wait, whose only job is to finish the
#: upload: three hours, in which a 2.4 GB bundle uploads at 2 Mbit/s (about
#: 2 hours 40 minutes). When the command stops waiting it closes its
#: emitter, which abandons an upload still running; the next emitter started
#: on the machine retries it.
RESTAGE_WAIT_SECONDS = 10800.0
#: How often the build asks the emitter for the bundle's receipt meanwhile.
RECEIPT_POLL_SECONDS = 1.0

_MODES = ("local_and_remote", "local_only", "disabled")
_STATUSES = ("uploaded", "already_staged", "failed", "skipped")
#: The ``error_code`` of a ``failed`` record. ``REMOTE_DIFFERS`` and
#: ``PRESERVED_BYTES_CHANGED`` are the emitter's final failures;
#: ``REPOSITORY_UNAVAILABLE``, ``UPLOAD_FAILED``, ``CREDENTIAL_MISSING`` and
#: ``PRESERVED_FILES_MISSING`` its last failed attempt on a hand-over it keeps
#: retrying. Microcosm adds four: ``UPLOAD_PENDING``, a hand-over whose first
#: attempt had not finished when the wait ended; ``HANDOVER_REFUSED``, one the
#: emitter refused to queue; ``EMITTER_UNAVAILABLE``, one no emitter took; and
#: ``RECEIPT_UNRECOGNIZED``, a settled answer it does not recognize (a
#: ``failed`` receipt with an unknown code, or ``uploaded`` with no revision).
#: With a code in ``RETRYING_ERROR_CODES`` the bundle may still arrive: the
#: emitter keeps it queued, and a re-stage records the outcome.
type StagedDatasetErrorCode = Literal[
    "UPLOAD_FAILED",
    "REMOTE_DIFFERS",
    "REPOSITORY_UNAVAILABLE",
    "CREDENTIAL_MISSING",
    "PRESERVED_FILES_MISSING",
    "PRESERVED_BYTES_CHANGED",
    "UPLOAD_PENDING",
    "HANDOVER_REFUSED",
    "EMITTER_UNAVAILABLE",
    "RECEIPT_UNRECOGNIZED",
]
_ERROR_CODES: Final[tuple[StagedDatasetErrorCode, ...]] = get_args(
    StagedDatasetErrorCode.__value__
)
#: The codes of a bundle the emitter still holds and may yet deliver.
RETRYING_ERROR_CODES: Final[frozenset[StagedDatasetErrorCode]] = frozenset(
    {
        "UPLOAD_PENDING",
        "REPOSITORY_UNAVAILABLE",
        "UPLOAD_FAILED",
        "CREDENTIAL_MISSING",
        "PRESERVED_FILES_MISSING",
    }
)
#: The only codes the emitter settles a hand-over with as ``failed``.
_EMITTER_FINAL_CODES: Final[frozenset[StagedDatasetErrorCode]] = frozenset(
    {"REMOTE_DIFFERS", "PRESERVED_BYTES_CHANGED"}
)
#: The manifest block that keeps what the build recorded in its staging run
#: and Logbook row (``mode``, ``status``, ``error_code``), which re-stages
#: never change.
AT_BUILD_RECORD_KEY: Final = "staged_dataset_at_build"
#: The manifest block that names the build's hand-over, so that a later
#: re-stage asks the emitter for its outcome instead of handing over again.
HANDOVER_RECORD_KEY: Final = "staged_dataset_handover"
_HANDOVER_ID = re.compile(r"^[0-9a-f]{32}$")
#: Receipt statuses of a hand-over the emitter queued; ``refused`` and
#: ``unavailable`` mean no emitter holds it.
_QUEUED_STATUSES = frozenset({"pending", "uploaded", "already_staged", "failed"})
#: The values ``huggingface_hub`` reads as true in ``HF_HUB_OFFLINE``.
_TRUE_VALUES = frozenset({"1", "ON", "YES", "TRUE"})
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
_ROLE_SEPARATORS = re.compile(r"[^a-z0-9]+")
#: A digest or identifier inside a key: eight or more hexadecimal characters
#: with at least one digit. It says which file, not what kind of file.
_HEXADECIMAL_TOKEN = re.compile(r"^(?=[0-9a-f]*[0-9])[0-9a-f]{8,}$")
#: An emitter's own error code, printed only when it has this shape.
_EMITTER_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class StagedDatasetError(ValueError):
    """A staged-dataset bundle or delivery record is malformed."""


class StagedFileDigest(TypedDict):
    """The exact bytes registered for one staged file."""

    sha256: str
    bytes: int


class StagedDatasetDelivery(TypedDict):
    """Validated result of staging a dataset or graph-evidence bundle.

    ``failed`` with an ``error_code`` in ``RETRYING_ERROR_CODES``
    (``UPLOAD_PENDING`` and the emitter's retried failures) means the bundle
    may still arrive: the emitter keeps it queued, and re-staging the run
    directory records the outcome. Every other ``failed`` code is final.
    """

    contract_version: Literal[1]
    mode: Literal["local_and_remote", "local_only", "disabled"]
    repository: str | None
    prefix: str | None
    run_id: str | None
    revision: str | None
    status: Literal["uploaded", "already_staged", "failed", "skipped"]
    error_code: StagedDatasetErrorCode | None
    opt_out_reason: str | None
    files: dict[str, StagedFileDigest]


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


def output_role(key: str) -> str:
    """The role a manifest output is handed to the emitter with: its key.

    Lowercased, with each run of other characters as one ``_``, so that
    ``dataset`` and ``calibration_diagnostics`` keep their names and
    ``orrery/graph.schema.json`` becomes ``orrery_graph_schema_json``.
    Digests and identifiers in the key are left out
    (``orrery/evidence-<id>-<digest>-binding.json`` becomes
    ``orrery_evidence_binding_json``). A key that would not start with a
    letter, or would name a sidecar, gains an ``output_`` prefix.
    """

    words = [
        word
        for word in _ROLE_SEPARATORS.split(str(key).lower())
        if word and not _HEXADECIMAL_TOKEN.fullmatch(word)
    ]
    role = "_".join(words)
    if not role[:1].isalpha() or role in SIDECAR_ROLES:
        role = "_".join(["output", *words])
    return role[:64].rstrip("_")


@dataclass(frozen=True)
class StagedFile:
    """One manifest-registered output, verified against its recorded digest.

    ``role`` is its ``outputs`` key as a role name (:func:`output_role`).
    """

    name: str
    path: Path
    sha256: str
    bytes: int
    role: str


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
            files.append(
                StagedFile(
                    name=name,
                    path=path,
                    sha256=measured,
                    bytes=size,
                    role=output_role(key),
                )
            )
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


def validate_staged_dataset_delivery(
    payload: Mapping[str, Any],
) -> StagedDatasetDelivery:
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
        return cast(StagedDatasetDelivery, record)
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
        return cast(StagedDatasetDelivery, record)
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
        return cast(StagedDatasetDelivery, record)
    if error_code is not None:
        raise StagedDatasetError("a delivered staged dataset carries no error code.")
    if status == "uploaded" and revision is None:
        raise StagedDatasetError("an uploaded staged dataset records its revision.")
    return cast(StagedDatasetDelivery, record)


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


def disabled_staged_dataset(reason: str) -> StagedDatasetDelivery:
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
) -> StagedDatasetDelivery:
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


def hub_offline(environment: Mapping[str, str] | None = None) -> bool:
    """Whether ``huggingface_hub`` refuses every request in this environment.

    It reads ``HF_HUB_OFFLINE``, then ``TRANSFORMERS_OFFLINE``, as
    ``huggingface_hub`` does. The Modal runner sets ``HF_HUB_OFFLINE=1`` for
    every build it runs, and the build's emitter inherits it, so a bundle
    handed over there would never upload.
    """

    values = os.environ if environment is None else environment
    value = values.get("HF_HUB_OFFLINE") or values.get("TRANSFORMERS_OFFLINE")
    return str(value or "").upper() in _TRUE_VALUES


def write_access_problem(
    emitter: BundleEmitter | None,
    repository: str,
    *,
    action: str = "remote dataset staging",
    emitter_task: str = "the build's emitter, which uploads the bundle",
    proof: str = "the upload at the end of the run",
) -> str | None:
    """Why a remote stage to ``repository`` could not complete, or None.

    ``emitter`` is the build's emitter client, running or not. The refusal
    names the cause and its remedy: a repository that is not ``owner/name``,
    no emitter command, an emitter older than the minimum (with the upgrade
    command), an emitter that did not start, one that did not answer the
    check, or the emitter's own check of the credential, the repository and
    the credential's write scope (``check_write_access``). A scope the Hub
    does not describe is a warning: only the upload itself proves it.

    Another hand-over to the Hub, such as a release change, names itself in
    the messages: ``action`` is what needs the access, ``emitter_task`` the
    emitter and what it does for it, and ``proof`` what proves an unread
    write scope.
    """

    try:
        repository_identifier(repository)
    except ValueError:
        return (
            f"{action} needs a Hugging Face repository identifier of "
            f"the form owner/name; got {repository!r}"
        )
    needs = f"{action} to {repository} needs {emitter_task}, but"
    problem = emitter_problem(emitter)
    if problem is not None:
        return f"{needs} {problem}"
    assert emitter is not None, "emitter_problem refuses a missing emitter."
    access: Mapping[str, Any] = emitter.check_write_access(repository)
    if access.get("status") != "checked":
        # The client's latest failure is shared by every call, so it is only
        # a hint about this one.
        failure = emitter.last_failure
        hint = ""
        if failure is not None:
            details = ", ".join(
                str(value) for value in (failure.reason, failure.error_type) if value
            )
            hint = f" (its latest failed call: {details})"
        return (
            f"{needs} it did not answer the write-access check in time{hint}; try again"
        )
    if access.get("credential") != "present":
        return (
            f"{action} to {repository} needs a Hugging Face write "
            "credential (HF_TOKEN or `hf auth login`)"
        )
    if access.get("reachable") is False:
        error_type = access.get("error_type")
        reason = error_type if isinstance(error_type, str) else "no answer"
        return f"{action} cannot reach {repository} ({reason})"
    can_write = access.get("can_write")
    if can_write is False:
        return (
            f"{action} to {repository} needs a write credential: "
            "the ambient Hugging Face token is read-only or is not scoped to this "
            "repository or its owner (a fine-grained token needs repo.write on "
            f"{repository} or on {repository.split('/', 1)[0]})"
        )
    if can_write is None:
        _warn(
            "the Hugging Face credential's write scope could not be read; "
            f"{proof} will prove it."
        )
    return None


@dataclass(frozen=True)
class StagedBundleHandOver:
    """A bundle's delivery record, and which of its files the build may rewrite.

    ``handed_over`` names every file of a hand-over the emitter queued, and
    ``copied_files`` the ones it copied before answering. The emitter reads
    the others from disk when it uploads, now or in a later attempt, so they
    must keep their bytes. Nothing was queued in local-only and disabled
    modes, or when no emitter took the bundle. ``handover_id`` is the
    emitter's ID of a queued hand-over (:func:`handover_record`).
    """

    delivery: StagedDatasetDelivery
    handed_over: frozenset[str] = frozenset()
    copied_files: frozenset[str] = frozenset()
    handover_id: str | None = None
    #: The run directory the hand-over names its files in.
    run_dir: str | None = None

    def may_rewrite(self, name: str) -> bool:
        """Whether the build may change ``name`` in the run directory now."""

        return name not in self.handed_over or name in self.copied_files


def handover_record(staged: StagedBundleHandOver) -> dict[str, Any] | None:
    """The manifest's ``staged_dataset_handover`` block, or None if none queued.

    It names the emitter's hand-over, the run directory its job names the
    files in, and the outputs' digests as handed over, so a later re-stage
    asks for that hand-over's outcome (:func:`resume_hand_over`) instead of
    handing over a second bundle, whatever the run directory's sidecars say
    by then.
    """

    delivery = staged.delivery
    if (
        staged.handover_id is None
        or staged.run_dir is None
        or delivery["repository"] is None
    ):
        return None
    return {
        "handover_id": staged.handover_id,
        "repository": delivery["repository"],
        "run_id": str(delivery["run_id"]),
        "run_dir": staged.run_dir,
        "files": {name: dict(entry) for name, entry in delivery["files"].items()},
    }


def at_build_record(delivery: Mapping[str, Any]) -> dict[str, Any]:
    """What a build's staging run and Logbook row recorded of its bundle.

    The manifest keeps it as ``staged_dataset_at_build``, so a re-stage that
    replaces ``staged_dataset`` can still say what those records hold.
    """

    return {
        "mode": delivery["mode"],
        "status": delivery["status"],
        "error_code": delivery["error_code"],
    }


def recorded_handover(
    manifest: Mapping[str, Any], *, repository: str, run_id: str
) -> dict[str, Any] | None:
    """The hand-over a manifest records for ``repository`` and ``run_id``."""

    record = manifest.get(HANDOVER_RECORD_KEY)
    if not isinstance(record, Mapping):
        return None
    handover_id = record.get("handover_id")
    files = record.get("files")
    if (
        not isinstance(handover_id, str)
        or not _HANDOVER_ID.fullmatch(handover_id)
        or record.get("repository") != repository
        or record.get("run_id") != run_id
        or not isinstance(record.get("run_dir"), str)
        or not isinstance(files, Mapping)
        or not files
    ):
        return None
    return dict(record)


def _sidecar_paths(bundle: StagedDatasetBundle) -> tuple[Path, Path]:
    return (
        bundle.run_dir / STAGED_MANIFEST_FILENAME,
        bundle.run_dir / SHA256SUMS_FILENAME,
    )


def bundle_request(
    bundle: StagedDatasetBundle,
    *,
    repository: str,
    prefix: str = DEFAULT_STAGED_DATASET_PREFIX,
    message: str | None = None,
) -> StageDatasetBundle:
    """The hand-over of every bundle file, by path, with its role and digest.

    The outputs keep their digests from the manifest check; the manifest and
    the two sidecars are measured now, as they are handed over. The interface
    raises ``ValueError`` for a request that breaks its rules.
    """

    files = [
        BundleFile(
            name=item.name,
            role=item.role,
            local_path=item.path,
            sha256=item.sha256,
            bytes=item.bytes,
        )
        for item in bundle.files
    ]
    staged_manifest, sums = _sidecar_paths(bundle)
    for path, role in (
        (bundle.manifest_path, BUILD_MANIFEST_ROLE),
        (staged_manifest, BUNDLE_INVENTORY_ROLE),
        (sums, CHECKSUMS_ROLE),
    ):
        files.append(
            BundleFile(
                name=path.name,
                role=role,
                local_path=path,
                sha256=sha256_file(path),
                bytes=int(path.stat().st_size),
            )
        )
    return StageDatasetBundle(
        repository=repository,
        run_id=bundle.run_id,
        files=files,
        prefix=_safe_prefix(prefix),
        commit_message=message,
    )


def _next_attempt(
    emitter: BundleEmitter,
    receipt: DatasetBundleReceipt,
    *,
    wait_seconds: float,
    poll_seconds: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> DatasetBundleReceipt:
    """A hand-over's receipt once its next attempt has finished, or at the limit.

    The wait ends when the receipt settles or its ``attempts`` grows, when
    the emitter does not answer, or after ``wait_seconds``. A hand-over the
    emitter already holds may count earlier attempts; the wait is then for
    the next one.
    """

    handover_id = receipt.get("handover_id")
    if receipt.get("status") != "pending" or not handover_id:
        return receipt
    attempts = receipt.get("attempts", 0)
    deadline = clock() + max(0.0, float(wait_seconds))
    while clock() < deadline:
        latest = emitter.dataset_bundle_receipt(str(handover_id))
        if latest is None:
            break
        receipt = latest
        if receipt.get("status") != "pending" or receipt.get("attempts", 0) != attempts:
            break
        sleep(poll_seconds)
    return receipt


def _emitter_code(receipt: Mapping[str, Any]) -> str | None:
    code = receipt.get("error_code")
    return code if isinstance(code, str) and _EMITTER_CODE.fullmatch(code) else None


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:g} minutes"


def _delivery_from_receipt(
    record: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    wait_seconds: float,
    earlier: bool = False,
    may_hold_earlier_bundle: bool = False,
) -> StagedDatasetDelivery:
    """The build's delivery record of the emitter's receipt; warns on failure.

    ``earlier`` is a receipt of a hand-over made before this staging: its
    upload means the bundle was already staged. ``may_hold_earlier_bundle``
    says that this directory handed a bundle over before and its outputs
    have changed since, so a folder that holds a different bundle may hold
    that one.
    """

    prefix = record["prefix"]
    status = receipt.get("status")
    code = _emitter_code(receipt)
    revision = receipt.get("revision")
    revision = revision if isinstance(revision, str) and revision.strip() else None
    if status == "uploaded" and revision is not None:
        landed = "already_staged" if earlier else "uploaded"
        return validate_staged_dataset_delivery(
            {**record, "status": landed, "revision": revision}
        )
    if status == "already_staged":
        return validate_staged_dataset_delivery(
            {**record, "status": "already_staged", "revision": revision}
        )
    if status == "pending":
        if receipt.get("error_code") is None:
            error_code = "UPLOAD_PENDING"
        else:
            error_code = code if code in _ERROR_CODES else "UPLOAD_FAILED"
        if error_code == "UPLOAD_PENDING":
            _warn(
                f"the emitter had not finished uploading the staged dataset for "
                f"{prefix} when the wait ended, which it does after at most "
                f"{_minutes(wait_seconds)} (UPLOAD_PENDING); it keeps the bundle "
                "queued, and it or the next emitter started on this machine "
                "uploads it. Leave the run directory unchanged until then, and "
                "re-stage it to record the outcome."
            )
        else:
            _warn(
                f"staged dataset upload failed for {prefix} ({error_code}); the "
                "emitter keeps the bundle queued and retries it, as does the next "
                "emitter started on this machine. Leave the run directory unchanged "
                "until then, and re-stage it to record the outcome."
            )
    elif status == "failed" and code == "REMOTE_DIFFERS":
        error_code = "REMOTE_DIFFERS"
        if earlier:
            _warn(
                f"the emitter refused the build's own hand-over of {prefix} "
                "(REMOTE_DIFFERS): the folder already holds another bundle for this "
                "run, for example one a re-stage of a copy of this directory "
                "uploaded first. Compare the folder's staged_manifest.json with the "
                "manifest's staged_dataset_handover files before staging anything "
                "else: if it holds this build's outputs, a new run id would upload "
                "a second copy."
            )
        elif may_hold_earlier_bundle:
            _warn(
                f"{prefix} already holds a bundle whose outputs differ from this "
                "directory's, and the emitter refuses to overwrite it "
                "(REMOTE_DIFFERS). This directory's outputs changed after its "
                "earlier hand-over, so the folder may hold the bundle handed over "
                "earlier: re-stage where the emitter keeps that hand-over to record "
                "it. Do not stage under a new run id, which would upload a second "
                "copy."
            )
        else:
            _warn(
                f"{prefix} already holds a different bundle, and the emitter "
                "refuses to overwrite it (REMOTE_DIFFERS). Re-stage under another "
                "run id if that is intended."
            )
    elif status == "failed" and code in _EMITTER_FINAL_CODES:
        error_code = code
        _warn(
            f"staged dataset upload failed for {prefix} ({error_code}); the bundle "
            "stays local and can be re-staged."
        )
    elif status in ("failed", "uploaded"):
        error_code = "RECEIPT_UNRECOGNIZED"
        answer = status if code is None else f"{status}, {code}"
        _warn(
            f"the emitter settled the staged dataset hand-over for {prefix} with an "
            f"answer this build does not recognize ({answer}; RECEIPT_UNRECOGNIZED); "
            "the bundle stays local and can be re-staged."
        )
    elif status == "refused":
        error_code = "HANDOVER_REFUSED"
        _warn(
            f"the emitter refused the staged dataset hand-over for {prefix} "
            f"(HANDOVER_REFUSED: {code or 'no code'}); the bundle stays local and "
            "can be re-staged."
        )
    else:
        error_code = "EMITTER_UNAVAILABLE"
        _warn(
            f"no emitter took the staged dataset hand-over for {prefix} "
            "(EMITTER_UNAVAILABLE); the bundle stays local and can be re-staged."
        )
    return validate_staged_dataset_delivery({**record, "error_code": error_code})


def _remote_record(
    bundle: StagedDatasetBundle,
    *,
    repository: str,
    prefix: str,
    files: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(repository, str) or not repository.strip():
        raise StagedDatasetError("Remote staging requires a repository identifier.")
    return {
        "contract_version": STAGED_DATASET_CONTRACT_VERSION,
        "mode": "local_and_remote",
        "repository": repository.strip(),
        "prefix": bundle.remote_prefix(prefix),
        "run_id": bundle.run_id,
        "revision": None,
        "status": "failed",
        "error_code": None,
        "opt_out_reason": None,
        "files": dict(files),
    }


def stage_bundle(
    bundle: StagedDatasetBundle,
    *,
    emitter: BundleEmitter | None,
    repository: str,
    prefix: str = DEFAULT_STAGED_DATASET_PREFIX,
    message: str | None = None,
    wait_seconds: float = STAGED_DATASET_WAIT_SECONDS,
    poll_seconds: float = RECEIPT_POLL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    earlier_files: Mapping[str, Any] | None = None,
) -> StagedBundleHandOver:
    """Hand the bundle and its sidecars to the emitter; record, never raise.

    ``earlier_files`` are the outputs' digests of an earlier hand-over of
    this run, if any; a ``REMOTE_DIFFERS`` warning then says when the folder
    may hold that bundle.

    The emitter writes every file under ``<prefix>/<run_id>/`` in one commit,
    idempotent on the outputs' digests (``already_staged``, or
    ``REMOTE_DIFFERS`` for a different bundle under the run id), and retries
    failed attempts from its durable queue. The build waits for the first
    attempt after the hand-over, at most ``wait_seconds``; a hand-over still
    queued then is recorded as ``failed`` with the emitter's last error code,
    or ``UPLOAD_PENDING``, and stays queued in the emitter
    (``RETRYING_ERROR_CODES``).
    """

    remote_prefix = bundle.remote_prefix(prefix)
    for sidecar in _sidecar_paths(bundle):
        if not sidecar.is_file():
            raise StagedDatasetError(
                f"{sidecar.name} missing; call write_sidecars before stage_bundle."
            )
    record = _remote_record(
        bundle, repository=repository, prefix=prefix, files=bundle.digests()
    )
    try:
        request = bundle_request(
            bundle, repository=record["repository"], prefix=prefix, message=message
        )
    except (OSError, ValueError) as error:
        _warn(
            f"the staged dataset hand-over for {remote_prefix} could not be built "
            f"({type(error).__name__}); the bundle stays local and can be re-staged."
        )
        return StagedBundleHandOver(
            validate_staged_dataset_delivery(
                {**record, "error_code": "HANDOVER_REFUSED"}
            )
        )
    if emitter is None or not emitter.available:
        answer: DatasetBundleReceipt = unavailable_bundle_receipt(request)
        receipt = answer
    else:
        print(
            f"handing the staged dataset bundle to the emitter; waiting up to "
            f"{_minutes(wait_seconds)} for its upload...",
            file=sys.stderr,
            flush=True,
        )
        answer = emitter.stage_dataset_bundle(request, wait_seconds=0)
        receipt = _next_attempt(
            emitter,
            answer,
            wait_seconds=wait_seconds,
            poll_seconds=poll_seconds,
            clock=clock,
            sleep=sleep,
        )
    delivery = _delivery_from_receipt(
        record,
        receipt,
        wait_seconds=wait_seconds,
        may_hold_earlier_bundle=earlier_files is not None
        and dict(earlier_files) != record["files"],
    )
    if answer.get("status") not in _QUEUED_STATUSES:
        return StagedBundleHandOver(delivery)
    return StagedBundleHandOver(
        delivery,
        handed_over=frozenset(item.name for item in request.files),
        copied_files=_copied_files(answer, receipt),
        handover_id=_handover_id(answer),
        run_dir=str(bundle.run_dir),
    )


def _copied_files(
    answer: Mapping[str, Any], receipt: Mapping[str, Any]
) -> frozenset[str]:
    copied = answer.get("copied_files")
    if not isinstance(copied, list):
        copied = receipt.get("copied_files")
    return frozenset(name for name in copied or () if isinstance(name, str))


def _handover_id(receipt: Mapping[str, Any]) -> str | None:
    handover_id = receipt.get("handover_id")
    return (
        handover_id
        if isinstance(handover_id, str) and _HANDOVER_ID.fullmatch(handover_id)
        else None
    )


def _landed_receipt(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("status") in ("uploaded", "already_staged")


def _stuck(receipt: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    """Whether a hand-over that has not landed can never finish.

    The emitter settled it because its files changed
    (``PRESERVED_BYTES_CHANGED``), or a file it reads from disk is gone from
    the path the job names, the recorded run directory, as after the
    directory moved. Another settled failure is final and is recorded.
    """

    if receipt.get("error_code") == "PRESERVED_BYTES_CHANGED":
        return True
    if receipt.get("status") != "pending":
        return False
    run_dir = Path(str(record["run_dir"]))
    return any(not (run_dir / str(name)).is_file() for name in record["files"])


def resume_hand_over(
    bundle: StagedDatasetBundle,
    *,
    emitter: BundleEmitter | None,
    record: Mapping[str, Any],
    repository: str,
    prefix: str = DEFAULT_STAGED_DATASET_PREFIX,
    wait_seconds: float = STAGED_DATASET_WAIT_SECONDS,
    poll_seconds: float = RECEIPT_POLL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> StagedBundleHandOver | None:
    """Record the outcome of an earlier hand-over the emitter still holds.

    ``record`` is the manifest's ``staged_dataset_handover``. The bundle stays
    the one handed over then, with the outputs' digests the record lists:
    nothing is handed over again, so a build that rewrote an output
    afterwards (the national build record) is not refused as
    ``REMOTE_DIFFERS``. The receipt is read first: a hand-over that landed is
    recorded as ``already_staged``, with its commit, wherever the run
    directory is now. One still queued is waited for as :func:`stage_bundle`
    waits, from the paths its job names; one that settled as a failure is
    recorded with its final code.

    None, so that the caller hands the directory over as a new job, when the
    emitter does not know the hand-over (another machine, another user or
    cache directory), or when a hand-over that has not landed can never
    finish: its files changed, or a file it reads is gone from the recorded
    run directory (the directory moved).
    """

    handover_id = str(record.get("handover_id"))
    if (
        emitter is None
        or not emitter.available
        or not _HANDOVER_ID.fullmatch(handover_id)
    ):
        return None
    receipt = emitter.dataset_bundle_receipt(handover_id)
    if receipt is None:
        return None
    if not _landed_receipt(receipt):
        if _stuck(receipt, record):
            return None
        if receipt.get("status") == "pending":
            print(
                f"the emitter still holds the earlier hand-over of "
                f"{bundle.remote_prefix(prefix)}; waiting up to "
                f"{_minutes(wait_seconds)} for its upload...",
                file=sys.stderr,
                flush=True,
            )
            receipt = _next_attempt(
                emitter,
                receipt,
                wait_seconds=wait_seconds,
                poll_seconds=poll_seconds,
                clock=clock,
                sleep=sleep,
            )
            if not _landed_receipt(receipt) and _stuck(receipt, record):
                return None
    remote = _remote_record(
        bundle, repository=repository, prefix=prefix, files=record["files"]
    )
    delivery = _delivery_from_receipt(
        remote, receipt, wait_seconds=wait_seconds, earlier=True
    )
    run_dir = str(record["run_dir"])
    # The job reads only the recorded run directory; files elsewhere are free.
    handed_over = (
        frozenset(record["files"])
        | {bundle.manifest_name, STAGED_MANIFEST_FILENAME, SHA256SUMS_FILENAME}
        if run_dir == str(bundle.run_dir)
        else frozenset()
    )
    return StagedBundleHandOver(
        delivery,
        handed_over=handed_over,
        copied_files=_copied_files(receipt, receipt),
        handover_id=handover_id,
        run_dir=run_dir,
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
