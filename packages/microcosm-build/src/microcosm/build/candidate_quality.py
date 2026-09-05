"""Local, content-addressed aggregate evidence; never a release publisher.

An attachment's identity is a producer attestation, not scientific certification.
No population microdata are read. Existing aggregate schemas are copied unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "microcosm.candidate_quality.v1"
INDEX_SCHEMA = "microcosm.candidate_quality_index.v1"
INDEX_PATH = "index.json"
MAX_ATTACHMENT_BYTES = 64 * 1024 * 1024
EXPECTED_ATTACHMENTS = (
    "calibration_diagnostics",
    "build_manifest",
    "release_manifest",
    "demographics",
    "cross_dataset",
    "validation_reach",
    "support",
    "distributions",
    "reform_validation",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXECUTION = {"pending", "running", "completed", "failed", "cancelled"}
_ELIGIBILITY = {"not_evaluated", "eligible", "ineligible", "excluded"}


class CandidateQualityError(ValueError):
    """An evidence reference or identity cannot be authenticated."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateQualityError(f"{label} must be an object.")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    """A nested receipt field that is absent or not an object reads as empty."""
    return value if isinstance(value, Mapping) else {}


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateQualityError(f"{label} must be a nonempty string.")
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CandidateQualityError("Invalid SHA-256 digest.")
    return value


def _bytes(payload: object) -> bytes:
    try:
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise CandidateQualityError(f"Evidence must be finite JSON: {exc}") from exc


def _json(data: bytes, label: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise CandidateQualityError(f"Non-finite JSON value {value} in {label}.")

    try:
        text = data.decode("utf-8")
        if text.startswith("\ufeff"):
            raise CandidateQualityError("JSON must not begin with a byte order mark.")
        result = json.loads(text, parse_constant=reject)

        def finite(value: object) -> None:
            if isinstance(value, (float, int)):
                try:
                    valid_number = math.isfinite(value)
                except OverflowError:
                    valid_number = False
                if not valid_number:
                    raise CandidateQualityError("JSON numbers must be finite.")
            if isinstance(value, dict):
                for child in value.values():
                    finite(child)
            elif isinstance(value, list):
                for child in value:
                    finite(child)

        finite(result)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CandidateQualityError(f"Invalid JSON in {label}: {exc}") from exc
    return _object(result, label)


def _read_bytes(path: Path, label: str, *, limit: int | None = None) -> bytes:
    """Read a bounded regular file through one descriptor, including on a FIFO path."""
    maximum = MAX_ATTACHMENT_BYTES if limit is None else limit
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise CandidateQualityError(f"{label} must be a bounded regular file.")
            data = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
            if len(data) > maximum:
                raise CandidateQualityError(f"{label} exceeds the size bound.")
            fields = ("st_size", "st_mtime_ns", "st_ctime_ns")
            if len(data) != before.st_size or any(
                getattr(before, key) != getattr(after, key) for key in fields
            ):
                raise CandidateQualityError(f"{label} changed while being read.")
            return data
    except OSError as exc:
        raise CandidateQualityError(f"Cannot read {label}: {exc.strerror}.") from exc


def read_candidate_quality_json(path: Path) -> dict[str, Any]:
    """Read a bounded local JSON control document without changing its values."""
    return _json(
        _read_bytes(Path(path), "candidate JSON input"), "candidate JSON input"
    )


def _timestamp(value: object, label: str) -> str:
    value = _string(value, label)
    parts = re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
        r"(?:\.[0-9]+)?(?:Z|[+-]([0-9]{2}):([0-9]{2}))",
        value,
    )
    if parts is None or (
        parts[1] is not None and (int(parts[1]) > 23 or int(parts[2]) > 59)
    ):
        raise CandidateQualityError(f"{label} must be an ISO timestamp with timezone.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidateQualityError(f"Invalid {label}.") from exc
    return value


def _relative(root: Path, value: object) -> Path:
    path = PurePosixPath(_string(value, "relative path"))
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value)
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in str(value)
        or ":" in str(value)
        or str(path) != value
    ):
        raise CandidateQualityError(
            "Evidence path must be safe and relative to its root."
        )
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise CandidateQualityError("Evidence path escapes its root.")
    return resolved


def _metadata(value: Mapping[str, object]) -> dict[str, Any]:
    meta = _json(_bytes(dict(value)), "candidate metadata")
    if "attachments" in meta or "schema_version" in meta:
        raise CandidateQualityError(
            "Metadata cannot override attachments or schema_version."
        )
    for key in ("country", "candidate_release_id", "run_id", "created_at"):
        _string(meta.get(key), key)
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,299}", meta["candidate_release_id"]
    ):
        raise CandidateQualityError("Invalid candidate_release_id.")
    if not re.fullmatch(r"[a-z]{2}", meta["country"]):
        raise CandidateQualityError("country must be a two-letter lowercase code.")
    _timestamp(meta["created_at"], "created_at")
    if (
        not isinstance(meta.get("execution_status"), str)
        or meta["execution_status"] not in _EXECUTION
    ):
        raise CandidateQualityError("Unknown execution_status.")
    eligibility = _object(meta.get("release_eligibility"), "release_eligibility")
    if (
        not isinstance(eligibility.get("status"), str)
        or eligibility["status"] not in _ELIGIBILITY
    ):
        raise CandidateQualityError("Unknown release eligibility status.")
    _string(eligibility.get("reason"), "release eligibility reason")
    if eligibility["status"] == "eligible" and meta["execution_status"] != "completed":
        raise CandidateQualityError(
            "Only a completed run can carry an eligible attestation."
        )
    population = meta.get("population")
    if population is None:
        if meta["execution_status"] not in {"pending", "failed", "cancelled"}:
            raise CandidateQualityError(
                "This execution status requires a population identity."
            )
        if eligibility["status"] == "eligible":
            raise CandidateQualityError(
                "An eligible attestation requires a population."
            )
        meta["population"] = None
    else:
        population = _object(population, "population")
        _sha(population.get("sha256"))
        _string(population.get("period"), "population period")
        _string(population.get("entity"), "population entity")
    provenance = _object(meta.get("provenance", {}), "provenance")
    for key, item in provenance.items():
        _string(item, f"provenance {key}")
    if meta.get("geography") is not None:
        geography = _object(meta["geography"], "geography")
        if not isinstance(geography.get("levels"), list):
            raise CandidateQualityError("geography levels must be a list of names.")
        for item in geography["levels"]:
            _string(item, "geography level")
        if "vintage" in geography:
            _string(geography["vintage"], "geography vintage")
    return meta


def _bindings(
    meta: Mapping[str, Any], item: Mapping[str, Any], name: str
) -> dict[str, Any]:
    """Authoritative bindings for present fields, never facts inferred from payloads."""
    provenance = meta.get("provenance", {})
    if item["scope"] == "model":
        # A model inspection may have its own run/snapshot, independent of a build.
        expected = {
            key: value
            for key, value in provenance.items()
            if key in {"model_package", "model_version", "model_commit"}
        }
    else:
        expected = {
            **provenance,
            "candidate_release_id": meta["candidate_release_id"],
            "release_id": meta["candidate_release_id"],
            "run_id": meta["run_id"],
        }
        if "chronicle_snapshot_id" in provenance:
            expected["snapshot_id"] = provenance["chronicle_snapshot_id"]
        if item["scope"] == "population":
            expected["population_sha256"] = meta["population"]["sha256"]
    expected["country"] = meta["country"]
    if name == "cross_dataset" and item["scope"] == "population":
        # Evaluation identity belongs to the bundle, not the calibration build.
        for key in ("run_id", "snapshot_id", "chronicle_snapshot_id"):
            expected.pop(key, None)
        if "snapshot_id" in item["identity"]:
            expected["chronicle_snapshot_id"] = item["identity"]["snapshot_id"]
    return expected


def _identity(meta: Mapping[str, Any], item: Mapping[str, Any], name: str) -> None:
    identity = _object(item.get("identity"), "attachment identity")
    scope = item.get("scope")
    population = meta.get("population")
    if scope == "population":
        if population is None:
            raise CandidateQualityError(
                "Population-scoped attachment requires a population."
            )
        required = {
            "candidate_release_id": meta["candidate_release_id"],
            "population_sha256": population["sha256"],
        }
    elif scope == "run":
        required = {
            "candidate_release_id": meta["candidate_release_id"],
            "run_id": meta["run_id"],
        }
        if "population_sha256" in identity:
            raise CandidateQualityError(
                "Run-scoped identity cannot claim a population."
            )
    elif scope == "model":
        required = {
            key: _string(identity.get(key), f"identity {key}")
            for key in ("model_package", "model_version")
        }
        if {"population_sha256", "candidate_release_id", "release_id"}.intersection(
            identity
        ):
            raise CandidateQualityError(
                "Model-scoped identity cannot claim a population or candidate."
            )
    else:
        raise CandidateQualityError(
            "Attachment scope must be population, model or run."
        )
    expected = _bindings(meta, item, name)
    for key, value in required.items():
        if identity.get(key) != value:
            raise CandidateQualityError(f"Attachment identity mismatch: {key}.")
    for key, value in identity.items():
        _string(value, f"identity {key}")
        if key in expected and expected[key] != value:
            raise CandidateQualityError(f"Attachment identity mismatch: {key}.")


def _embedded_identity(
    meta: Mapping[str, Any],
    item: Mapping[str, Any],
    name: str,
    payload: Mapping[str, Any],
) -> None:
    forbidden = set()
    if item["scope"] == "run":
        forbidden = {"population_sha256"}
    elif item["scope"] == "model":
        forbidden = {"population_sha256", "candidate_release_id", "release_id"}
    if forbidden.intersection(payload):
        raise CandidateQualityError("Embedded attachment claims an incompatible scope.")
    expected = {**_bindings(meta, item, name), **item["identity"]}
    if name == "cross_dataset" and item["scope"] == "population":
        # The optional alias describes the same evaluation snapshot when present.
        if "snapshot_id" in item["identity"]:
            expected["chronicle_snapshot_id"] = item["identity"]["snapshot_id"]
    for key, value in expected.items():
        if key not in payload:
            continue
        actual = payload[key]
        if key == "country" and isinstance(actual, dict):
            # Existing country presentation blocks use `code`; labels are not IDs.
            if "code" not in actual:
                continue
            actual = actual["code"]
        if actual != value:
            raise CandidateQualityError(
                f"Embedded attachment identity mismatch: {key}."
            )


def _attachment(
    meta: Mapping[str, Any], item: Mapping[str, Any], name: str, *, source: bool
) -> dict[str, Any]:
    item = _json(_bytes(dict(item)), "attachment")
    status = item.get("status")
    if isinstance(status, str) and status in {"pending", "missing", "failed"}:
        if set(item) != {"status", "reason"}:
            raise CandidateQualityError(
                "Unavailable attachment requires only status and reason."
            )
        _string(item.get("reason"), "attachment reason")
        return item

    if status != "available":
        raise CandidateQualityError("Unknown attachment status.")
    _identity(meta, item, name)
    if source:
        _string(item.get("source"), "attachment source")
        if set(item) != {"status", "source", "scope", "identity"}:
            raise CandidateQualityError(
                "Available source attachment has unexpected fields."
            )
    else:
        _sha(item.get("sha256"))
        _string(item.get("path"), "attachment path")
        if set(item) != {"status", "path", "sha256", "scope", "identity"}:
            raise CandidateQualityError("Available attachment has unexpected fields.")
    return item


def _bundle_blobs(
    source_root: Path,
    payload: Mapping[str, Any],
    item: Mapping[str, Any],
    country: str,
    manifest_size: int,
) -> dict[str, bytes]:
    """Validate and preserve the existing bundle's relative partition layout."""
    if (
        payload.get("schema_version") != "cross_dataset.frontend_bundle.v1"
        or item["scope"] != "population"
    ):
        raise CandidateQualityError(
            "cross_dataset requires a population-scoped frontend_bundle.v1 manifest."
        )
    identity = item["identity"]
    for key in ("run_id", "snapshot_id", "source_id"):
        _string(identity.get(key), f"cross-dataset identity {key}")
    if any(payload.get(key) != identity[key] for key in ("run_id", "snapshot_id")):
        raise CandidateQualityError("Cross-dataset run/snapshot identity mismatch.")
    if identity["source_id"] not in payload.get("source_ids", []):
        raise CandidateQualityError(
            "Cross-dataset source identity is not in the bundle."
        )
    if country.upper() not in payload.get("jurisdictions", []):
        raise CandidateQualityError(
            "Cross-dataset jurisdiction is not the candidate country."
        )
    partitions = _object(payload.get("partitions"), "cross-dataset partitions")
    facts = partitions.get("facts")
    if not isinstance(facts, list):
        raise CandidateQualityError("Cross-dataset fact partitions missing.")
    if not all(isinstance(part, dict) for part in facts):
        raise CandidateQualityError("Cross-dataset fact partition must be an object.")
    if len(facts) != payload.get("page_count") or any(
        part.get("page") != index + 1 for index, part in enumerate(facts)
    ):
        raise CandidateQualityError("Cross-dataset fact page sequence incomplete.")
    if any(
        type(part.get("count")) is not int or part["count"] < 0 for part in facts
    ) or sum(part["count"] for part in facts) != payload.get("fact_count"):
        raise CandidateQualityError("Cross-dataset fact counts do not reconcile.")
    descriptors = [
        partitions.get(key) for key in ("summary", "groups", "fact_index")
    ] + facts
    blobs = {}
    total = manifest_size
    for raw in descriptors:
        descriptor = _object(raw, "cross-dataset partition")
        relative = _string(descriptor.get("path"), "partition path")
        path = _relative(source_root, relative)
        if relative == "manifest.json" or relative in blobs:
            raise CandidateQualityError("Cross-dataset partition paths collide.")
        expected = _sha(descriptor.get("sha256"))
        data = _read_bytes(path, "cross-dataset partition")
        total += len(data)
        if total > 4 * MAX_ATTACHMENT_BYTES:
            raise CandidateQualityError("Cross-dataset aggregate size bound exceeded.")
        if hashlib.sha256(data).hexdigest() != expected:
            raise CandidateQualityError("Cross-dataset partition digest mismatch.")
        partition = _json(data, relative)
        if partition.get("schema_version") != "cross_dataset.frontend_bundle.v1":
            raise CandidateQualityError("Cross-dataset partition schema mismatch.")
        if any(
            partition.get(key) != identity[key] for key in ("run_id", "snapshot_id")
        ):
            raise CandidateQualityError("Cross-dataset partition identity mismatch.")
        blobs[relative] = data
    return blobs


def _immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_bytes(path, "existing immutable artifact", limit=len(data)) != data:
            raise CandidateQualityError(
                f"Existing immutable artifact differs: {path.name}."
            )
        return
    fd, temp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if (
                _read_bytes(path, "existing immutable artifact", limit=len(data))
                != data
            ):
                raise CandidateQualityError(
                    f"Existing immutable artifact differs: {path.name}."
                ) from None
    finally:
        os.unlink(temp)


def _prepare_index(
    root: Path, meta: Mapping[str, Any], reference: Mapping[str, str]
) -> bytes:
    path = root / INDEX_PATH
    if path.is_symlink():
        raise CandidateQualityError("Candidate index must not be a symlink.")
    if path.exists():
        index = _json(_read_bytes(path, "candidate index"), INDEX_PATH)
        if (
            index.get("schema_version") != INDEX_SCHEMA
            or index.get("country") != meta["country"]
            or not isinstance(index.get("candidates"), list)
        ):
            raise CandidateQualityError("Candidate index schema/country mismatch.")
        _timestamp(index.get("updated_at"), "index updated_at")
        seen = set()
        for raw in index["candidates"]:
            entry = _object(raw, "candidate index row")
            identifier = _string(entry.get("candidate_release_id"), "candidate ID")
            if (
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,299}", identifier)
                or identifier in seen
            ):
                raise CandidateQualityError("Invalid or duplicate candidate index ID.")
            seen.add(identifier)
            _string(entry.get("run_id"), "index run_id")
            status = entry.get("execution_status")
            if not isinstance(status, str) or status not in _EXECUTION:
                raise CandidateQualityError("Unknown index execution_status.")
            eligibility = _object(entry.get("release_eligibility"), "index eligibility")
            eligible_status = eligibility.get("status")
            if (
                not isinstance(eligible_status, str)
                or eligible_status not in _ELIGIBILITY
            ):
                raise CandidateQualityError("Unknown index eligibility status.")
            _string(eligibility.get("reason"), "index eligibility reason")
            if eligible_status == "eligible" and status != "completed":
                raise CandidateQualityError("Eligible index entry requires completion.")
            _timestamp(entry.get("updated_at"), "index row updated_at")
            manifest = _object(entry.get("manifest"), "index manifest")
            digest = _sha(manifest.get("sha256"))
            if manifest.get("path") != f"manifests/{digest}.json":
                raise CandidateQualityError(
                    "Index manifest path must be content-addressed."
                )
    else:
        index = {
            "schema_version": INDEX_SCHEMA,
            "country": meta["country"],
            "candidates": [],
        }
    row = {
        key: meta[key]
        for key in (
            "candidate_release_id",
            "run_id",
            "execution_status",
            "release_eligibility",
        )
    }
    row.update(updated_at=meta["created_at"], manifest=dict(reference))
    # One discovery row per candidate; older direct manifests remain usable.
    rows = [
        entry
        for entry in index["candidates"]
        if entry["candidate_release_id"] != meta["candidate_release_id"]
    ]
    index.update(updated_at=meta["created_at"], candidates=[*rows, row])
    data = _bytes(index)
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise CandidateQualityError("Candidate index exceeds the size bound.")
    return data


def _write_index(root: Path, data: bytes) -> None:
    # The caller preflights the whole index before immutable writes. Single writer.
    fd, temp = tempfile.mkstemp(dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, root / INDEX_PATH)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_candidate_quality(
    metadata: Mapping[str, object],
    attachments: Mapping[str, Mapping[str, object]],
    output_root: Path,
    *,
    update_index: bool = True,
) -> dict[str, str]:
    """Copy explicit aggregate JSON sources, then seal a candidate-only manifest.

    Eligibility is an explicit producer attestation, never inferred from run
    completion or aggregate metrics. Callers own the scientific gate decision.
    Index updates require a single writer; immutable direct references also work
    with ``update_index=False`` and do not depend on discovery freshness.
    """
    meta = _metadata(metadata)
    items: dict[str, Any] = {}
    blobs: dict[str, bytes] = {}
    for name, supplied in attachments.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise CandidateQualityError("Invalid attachment name.")
        item = _attachment(meta, supplied, name, source=True)
        if item["status"] == "available":
            source_path = Path(item.pop("source"))
            if source_path.suffix.lower() != ".json":
                raise CandidateQualityError(
                    "Only bounded aggregate JSON attachments are supported."
                )
            raw = _read_bytes(source_path, "aggregate JSON attachment")
            payload = _json(raw, name)
            _embedded_identity(meta, item, name, payload)
            digest = hashlib.sha256(raw).hexdigest()
            if name == "cross_dataset":
                prefix = f"artifacts/{digest}"
                for relative, data in _bundle_blobs(
                    source_path.parent, payload, item, meta["country"], len(raw)
                ).items():
                    blobs[f"{prefix}/{relative}"] = data
                item.update(path=f"{prefix}/manifest.json", sha256=digest)
            else:
                item.update(path=f"artifacts/{digest}.json", sha256=digest)
            blobs[item["path"]] = raw
        items[name] = item
    for name in EXPECTED_ATTACHMENTS:
        items.setdefault(
            name,
            {
                "status": "missing",
                "reason": "No aggregate artifact supplied for this candidate.",
            },
        )
    payload = {"schema_version": SCHEMA, **meta, "attachments": items}
    raw = _bytes(payload)
    if len(raw) > MAX_ATTACHMENT_BYTES:
        raise CandidateQualityError("Candidate manifest exceeds the size bound.")
    digest = hashlib.sha256(raw).hexdigest()
    reference = {"path": f"manifests/{digest}.json", "sha256": digest}
    root = Path(output_root).resolve()
    index = _prepare_index(root, meta, reference) if update_index else None
    for relative, data in blobs.items():
        _immutable(_relative(root, relative), data)
    _immutable(_relative(root, reference["path"]), raw)
    if index is not None:
        _write_index(root, index)
    return reference


def read_candidate_quality(root: Path, manifest_sha256: str) -> dict[str, Any]:
    """Authenticate a direct manifest and all available local aggregate bytes."""
    root = Path(root).resolve()
    digest = _sha(manifest_sha256)
    raw = _read_bytes(_relative(root, f"manifests/{digest}.json"), "candidate manifest")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise CandidateQualityError("Candidate manifest digest mismatch.")
    payload = _json(raw, "candidate manifest")
    if payload.get("schema_version") != SCHEMA:
        raise CandidateQualityError("Unknown candidate manifest schema.")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"attachments", "schema_version"}
    }
    meta = _metadata(metadata)
    for name, value in _object(payload.get("attachments"), "attachments").items():
        item = _attachment(meta, value, name, source=False)
        if item["status"] == "available":
            path = _relative(root, item["path"])
            data = _read_bytes(path, "candidate attachment")
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise CandidateQualityError(f"Attachment digest mismatch: {name}.")
            parsed = _json(data, name)
            _embedded_identity(meta, item, name, parsed)
            if name == "cross_dataset":
                _bundle_blobs(path.parent, parsed, item, meta["country"], len(data))
    return payload


def summarize_pool_diagnostics(
    manifest: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, Any]:
    """Project existing pool receipts without reinterpreting failure units.

    This is a compact diagnostic attachment, never a certification certificate.
    Omitted or unexpected fields remain null; no H5 is loaded or re-scored.
    """
    failures = gates.get("failures")
    if not isinstance(failures, list) or not all(
        isinstance(item, str) for item in failures
    ):
        failures = None
    population = _mapping(manifest.get("pool_h5"))
    counts = _mapping(manifest.get("provenance_counts"))
    result = {
        "schema_version": "microcosm.pool_quality_summary.v1",
        "candidate_release_id": manifest.get("release_id"),
        "population_sha256": population.get("sha256"),
        "status": manifest.get("status"),
        "simulation_ready": manifest.get("simulation_ready"),
        "passed": gates.get("passed"),
        "household_record_count": _mapping(counts.get("household")).get("rows"),
        "failure_string_count": len(failures) if failures is not None else None,
        "failures": failures,
        "interpretation": "Failure strings are not counts of failed comparison IDs. Diagnostic evidence only.",
    }
    return _json(_bytes(result), "pool summary")
