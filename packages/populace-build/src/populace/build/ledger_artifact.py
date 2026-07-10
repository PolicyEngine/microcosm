"""Pinned loading for PolicyEngine Ledger consumer artifacts.

Ledger publishes consumer artifacts: a directory with ``manifest.json``
(schema version, content hashes, embedded profile hashes),
``consumer_facts.jsonl``, per-profile JSON, and coverage diagnostics. A
Populace build should consume facts through this loader so the release
manifest can record exactly which Ledger data resolved its target values
(PolicyEngine/populace#160, #271) and so tampered or mismatched feeds fail
before they calibrate anything.

Strict loading (findings #7, #8): ``validate_rows`` (default on) checks every
row against the vendored ``ledger.consumer_fact.v1`` schema and enforces unique
``aggregate_fact_key``; ``require_pins`` fails closed for release builds ---
bare feeds are rejected, a manifest is required, and both content-hash pins
must be supplied and match. Verified feeds can be vendored byte-for-byte into a
release with :func:`vendor_ledger_consumer_artifact` so the exact facts that
resolved a release travel with it.

Like the rest of Populace's Ledger consumption, this module is duck-typed
against the published artifact contract (stdlib only); it does not import the
Ledger implementation package. The row-schema pin lives in the sibling
``populace.build.ledger_schema`` module (also stdlib-only).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from populace.build.ledger_schema import (
    CONSUMER_FACT_SCHEMA_SHA256,
    validate_consumer_fact_row,
)

__all__ = [
    "ALLOWED_LEDGER_ASSERTIONS",
    "CONSUMER_ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_LEDGER_ASSERTION",
    "VENDORED_ARTIFACT_SCHEMA_VERSION",
    "LedgerConsumerArtifact",
    "load_ledger_consumer_artifact",
    "vendor_ledger_consumer_artifact",
    "verify_vendored_ledger_artifact",
]

CONSUMER_ARTIFACT_SCHEMA_VERSION = "policyengine_ledger.consumer_artifact.v1"
VENDORED_ARTIFACT_SCHEMA_VERSION = "populace.vendored_ledger_artifact.v1"
ALLOWED_LEDGER_ASSERTIONS = frozenset(("observation", "source_projection"))
DEFAULT_LEDGER_ASSERTION = "observation"


@dataclass(frozen=True)
class LedgerConsumerArtifact:
    """A loaded, hash-verified Ledger consumer fact feed.

    ``manifest`` and ``manifest_sha256`` are ``None`` when the feed was a
    bare ``consumer_facts.jsonl`` file rather than an artifact directory;
    bare feeds are still content-addressed by ``facts_sha256`` so builds can
    pin them, but they carry no Ledger-side provenance and are rejected under
    ``require_pins``. When ``rows_validated`` is set every row passed the
    vendored ``ledger.consumer_fact.v1`` schema; otherwise rows are exactly as
    published: an ``assertion`` field is validated when present and never
    fabricated when absent (missing means observation-by-default to readers).
    """

    path: Path
    facts: tuple[dict[str, Any], ...]
    facts_sha256: str
    manifest: dict[str, Any] | None = None
    manifest_sha256: str | None = None
    require_pins: bool = False
    rows_validated: bool = False
    schema_sha256: str | None = None
    profile_hash_semantics: Mapping[str, str] = field(default_factory=dict)

    @property
    def fact_row_count(self) -> int:
        """Number of consumer fact rows in the feed."""
        return len(self.facts)

    def provenance(self) -> dict[str, Any]:
        """Ledger-artifact identity block for build and release manifests."""
        payload: dict[str, Any] = {
            "path_name": self.path.name,
            "path": str(self.path),
            "fact_row_count": self.fact_row_count,
            "facts_sha256": self.facts_sha256,
            "require_pins": self.require_pins,
            "rows_validated": self.rows_validated,
            "consumer_fact_schema_sha256": self.schema_sha256,
        }
        if self.manifest is not None:
            payload["schema_version"] = self.manifest.get("schema_version")
            payload["manifest_sha256"] = self.manifest_sha256
            profiles = self.manifest.get("profiles")
            if isinstance(profiles, dict):
                payload["profiles"] = {
                    str(profile_id): {
                        "sha256": (profile_meta or {}).get("sha256"),
                        "target_count": (profile_meta or {}).get("target_count"),
                        "hash_semantics": self.profile_hash_semantics.get(
                            str(profile_id)
                        ),
                    }
                    for profile_id, profile_meta in sorted(profiles.items())
                }
        else:
            payload["schema_version"] = None
            payload["manifest_sha256"] = None
        return payload


def load_ledger_consumer_artifact(
    path: str | Path,
    *,
    expected_facts_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    require_pins: bool = False,
    validate_rows: bool = True,
) -> LedgerConsumerArtifact:
    """Load a Ledger consumer artifact directory or bare consumer-facts file.

    Artifact directories are verified against their own manifest: the fact
    file's SHA-256 must match ``manifest.facts_sha256`` and every profile file
    the manifest lists is re-hashed from bytes. The optional ``expected_*``
    pins let a build config assert the exact artifact it was reviewed against;
    any mismatch raises before facts are used.

    ``validate_rows`` (default) validates every row against the vendored
    ``ledger.consumer_fact.v1`` schema and enforces unique ``aggregate_fact_key``.
    ``require_pins`` fails closed for release builds: a bare feed is rejected, a
    manifest is required, and both ``expected_facts_sha256`` and
    ``expected_manifest_sha256`` must be provided and match.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Ledger consumer artifact not found: {artifact_path}")
    artifact_path = artifact_path.resolve()

    if require_pins:
        if not artifact_path.is_dir():
            raise ValueError(
                "Ledger consumer artifact must be a hash-pinned artifact "
                "directory under require_pins; bare consumer-facts files carry "
                f"no manifest and are rejected: {artifact_path}."
            )
        if expected_facts_sha256 is None or expected_manifest_sha256 is None:
            raise ValueError(
                "Ledger consumer artifact require_pins demands both "
                "expected_facts_sha256 and expected_manifest_sha256; got "
                f"facts={expected_facts_sha256!r}, manifest={expected_manifest_sha256!r}."
            )

    manifest: dict[str, Any] | None = None
    manifest_sha256: str | None = None
    if artifact_path.is_dir():
        manifest_path = artifact_path / "manifest.json"
        facts_path = artifact_path / "consumer_facts.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Ledger consumer artifact has no manifest.json: {artifact_path}"
            )
        if not facts_path.exists():
            raise FileNotFoundError(
                "Ledger consumer artifact has no consumer_facts.jsonl: "
                f"{artifact_path}"
            )
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError(
                f"Ledger consumer artifact manifest must be an object: "
                f"{manifest_path}"
            )
        schema_version = manifest.get("schema_version")
        if schema_version != CONSUMER_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Ledger consumer artifact schema_version "
                f"{schema_version!r}; expected "
                f"{CONSUMER_ARTIFACT_SCHEMA_VERSION!r}."
            )
    else:
        facts_path = artifact_path

    facts_sha256 = _sha256_file(facts_path)
    if manifest is not None:
        declared = manifest.get("facts_sha256")
        if declared != facts_sha256:
            raise ValueError(
                "Ledger consumer artifact fact rows do not match the "
                f"manifest hash: {facts_sha256} != {declared}."
            )
    if (
        expected_facts_sha256 is not None
        and expected_facts_sha256 != facts_sha256
    ):
        raise ValueError(
            "Ledger consumer facts do not match the pinned hash: "
            f"{facts_sha256} != {expected_facts_sha256}."
        )
    if expected_manifest_sha256 is not None:
        if manifest_sha256 is None:
            raise ValueError(
                "A manifest hash pin was provided but the Ledger feed is a "
                f"bare consumer-facts file with no manifest: {artifact_path}."
            )
        if expected_manifest_sha256 != manifest_sha256:
            raise ValueError(
                "Ledger consumer artifact manifest does not match the pinned "
                f"hash: {manifest_sha256} != {expected_manifest_sha256}."
            )

    profile_hash_semantics: dict[str, str] = {}
    if manifest is not None:
        profile_hash_semantics = _verify_manifest_profiles(
            artifact_path, manifest, require_pins=require_pins
        )

    return LedgerConsumerArtifact(
        path=artifact_path,
        facts=_load_fact_rows(facts_path, validate_rows=validate_rows),
        facts_sha256=facts_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        require_pins=require_pins,
        rows_validated=validate_rows,
        schema_sha256=CONSUMER_FACT_SCHEMA_SHA256 if validate_rows else None,
        profile_hash_semantics=profile_hash_semantics,
    )


def _load_fact_rows(
    path: Path, *, validate_rows: bool
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen_aggregate_keys: set[str] = set()
    source = str(path)
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid Ledger facts JSONL row {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Invalid Ledger facts JSONL row {line_number}: expected "
                    f"object, got {type(row).__name__}."
                )
            if validate_rows:
                validate_consumer_fact_row(
                    row, line_number=line_number, source=source
                )
                aggregate_key = row["aggregate_fact_key"]
                if aggregate_key in seen_aggregate_keys:
                    raise ValueError(
                        f"Ledger facts JSONL row {line_number} repeats "
                        f"aggregate_fact_key {aggregate_key!r}; consumer facts "
                        "must be unique."
                    )
                seen_aggregate_keys.add(aggregate_key)
            else:
                # Legacy passthrough: the pinned schema is not enforced, so
                # only the assertion enum is checked and a missing assertion is
                # left untouched (readers treat absence as observation-by-
                # default; fabricating it would mistype publisher projections).
                assertion = row.get("assertion", DEFAULT_LEDGER_ASSERTION)
                if assertion not in ALLOWED_LEDGER_ASSERTIONS:
                    raise ValueError(
                        f"Ledger facts JSONL row {line_number} has unsupported "
                        f"assertion {assertion!r}; expected one of "
                        f"{sorted(ALLOWED_LEDGER_ASSERTIONS)}."
                    )
            rows.append(row)
    if not rows:
        raise ValueError(f"Ledger facts feed is empty: {path}")
    return tuple(rows)


def _verify_manifest_profiles(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    *,
    require_pins: bool,
) -> dict[str, str]:
    """Re-hash each manifest-listed profile file from bytes.

    Accepts an exact byte match, or a match against the bytes minus one
    trailing newline (upstream Ledger's legacy pre-newline hash semantics),
    recording ``legacy_pre_newline`` in provenance rather than accepting it
    silently. Under ``require_pins`` a listed profile whose file is absent is a
    hard error.
    """
    profiles = manifest.get("profiles")
    semantics: dict[str, str] = {}
    if not isinstance(profiles, dict):
        return semantics
    for profile_id, profile_meta in sorted(profiles.items()):
        meta = profile_meta or {}
        profile_file = _find_profile_file(artifact_dir, str(profile_id), meta)
        if profile_file is None:
            if require_pins:
                raise ValueError(
                    "Ledger consumer artifact manifest lists profile "
                    f"{profile_id!r} but no profile file was found in "
                    f"{artifact_dir}."
                )
            continue
        declared = meta.get("sha256")
        if not declared:
            raise ValueError(
                f"Ledger consumer artifact profile {profile_id!r} file "
                f"{profile_file.name} has no manifest sha256 to verify against."
            )
        actual, semantic = _profile_hash_match(profile_file, str(declared))
        if semantic is None:
            raise ValueError(
                f"Ledger consumer artifact profile {profile_id!r} file "
                f"{profile_file.name} does not match its manifest hash: "
                f"{actual} != {declared}."
            )
        semantics[str(profile_id)] = semantic
    return semantics


def _find_profile_file(
    artifact_dir: Path, profile_id: str, meta: Mapping[str, Any]
) -> Path | None:
    for key in ("path", "file", "filename"):
        declared_name = meta.get(key)
        if declared_name:
            candidate = artifact_dir / str(declared_name)
            return candidate if candidate.is_file() else None
    for candidate in (
        artifact_dir / "profiles" / f"{profile_id}.json",
        artifact_dir / f"{profile_id}.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _profile_hash_match(path: Path, declared: str) -> tuple[str, str | None]:
    raw = path.read_bytes()
    exact = hashlib.sha256(raw).hexdigest()
    if exact == declared:
        return exact, "exact"
    if raw.endswith(b"\n"):
        pre_newline = hashlib.sha256(raw[:-1]).hexdigest()
        if pre_newline == declared:
            return pre_newline, "legacy_pre_newline"
    return exact, None


def vendor_ledger_consumer_artifact(
    artifact: LedgerConsumerArtifact,
    dest_dir: str | Path,
    *,
    verified_facts_sha256: str | None = None,
    verified_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Copy a verified artifact's exact bytes into a release directory.

    Writes ``manifest.json``, ``consumer_facts.jsonl``, any ``profiles/*.json``
    and ``coverage.json`` verbatim, plus a ``vendored_manifest.json`` recording
    each file's sha256, byte count, and the verified pins. Each copy's hash is
    recomputed from the written bytes and must match the source, so a copy that
    diverges fails the release. Returns the vendored block for the release
    manifest.
    """
    source_dir = artifact.path
    if not source_dir.is_dir():
        raise ValueError(
            "Only directory artifacts can be vendored into a release; got a "
            f"bare consumer-facts feed: {source_dir}."
        )
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)

    files_meta: list[dict[str, Any]] = []
    for relative in _artifact_files_to_vendor(source_dir):
        source_file = source_dir / relative
        source_bytes = source_file.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        copy_path = destination / relative
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_bytes(source_bytes)
        copy_bytes = copy_path.read_bytes()
        copy_sha = hashlib.sha256(copy_bytes).hexdigest()
        if copy_sha != source_sha:
            raise ValueError(
                f"Vendored Ledger artifact copy {relative!r} diverged from its "
                f"source: {copy_sha} != {source_sha}."
            )
        files_meta.append(
            {
                "path": relative,
                "sha256": copy_sha,
                "size_bytes": len(copy_bytes),
            }
        )

    vendored: dict[str, Any] = {
        "schema_version": VENDORED_ARTIFACT_SCHEMA_VERSION,
        "source_path": str(source_dir),
        "facts_sha256": artifact.facts_sha256,
        "manifest_sha256": artifact.manifest_sha256,
        "verified_facts_sha256": verified_facts_sha256,
        "verified_manifest_sha256": verified_manifest_sha256,
        "consumer_fact_schema_sha256": artifact.schema_sha256,
        "files": files_meta,
    }
    (destination / "vendored_manifest.json").write_text(
        json.dumps(vendored, indent=2, sort_keys=True) + "\n"
    )
    return vendored


def verify_vendored_ledger_artifact(dest_dir: str | Path) -> dict[str, Any]:
    """Re-hash a vendored artifact against its ``vendored_manifest.json``.

    Raises ``ValueError`` if any vendored file is missing or its bytes no
    longer match the recorded sha256 --- the tamper check for a release's
    embedded facts.
    """
    destination = Path(dest_dir)
    manifest_path = destination / "vendored_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"Vendored Ledger artifact has no vendored_manifest.json: {destination}."
        )
    vendored = json.loads(manifest_path.read_bytes())
    for entry in vendored.get("files", ()):
        relative = entry["path"]
        copy_path = destination / relative
        if not copy_path.is_file():
            raise ValueError(
                f"Vendored Ledger artifact file {relative!r} is missing from "
                f"{destination}."
            )
        raw = copy_path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(
                f"Vendored Ledger artifact file {relative!r} was tampered: "
                f"{actual} != {entry['sha256']}."
            )
        if len(raw) != entry["size_bytes"]:
            raise ValueError(
                f"Vendored Ledger artifact file {relative!r} size changed: "
                f"{len(raw)} != {entry['size_bytes']}."
            )
    return vendored


def _artifact_files_to_vendor(source_dir: Path) -> list[str]:
    relatives = ["manifest.json", "consumer_facts.jsonl"]
    profiles_dir = source_dir / "profiles"
    if profiles_dir.is_dir():
        relatives.extend(
            f"profiles/{profile.name}"
            for profile in sorted(profiles_dir.glob("*.json"))
        )
    if (source_dir / "coverage.json").is_file():
        relatives.append("coverage.json")
    return relatives


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
