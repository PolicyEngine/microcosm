"""Pinned loading for PolicyEngine Ledger consumer artifacts.

Ledger publishes consumer artifacts: a directory with ``manifest.json``
(schema version, content hashes, embedded profile hashes),
``consumer_facts.jsonl``, per-profile JSON, and coverage diagnostics. A
Populace build should consume facts through this loader so the release
manifest can record exactly which Ledger data resolved its target values
(PolicyEngine/populace#160, #271) and so tampered or mismatched feeds fail
before they calibrate anything.

Like the rest of Populace's Ledger consumption, this module is duck-typed
against the published artifact contract (stdlib only); it does not import
the Ledger implementation package.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_LEDGER_ASSERTIONS",
    "CONSUMER_ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_LEDGER_ASSERTION",
    "LedgerConsumerArtifact",
    "load_ledger_consumer_artifact",
]

CONSUMER_ARTIFACT_SCHEMA_VERSION = "policyengine_ledger.consumer_artifact.v1"
ALLOWED_LEDGER_ASSERTIONS = frozenset(("observation", "source_projection"))
DEFAULT_LEDGER_ASSERTION = "observation"


@dataclass(frozen=True)
class LedgerConsumerArtifact:
    """A loaded, hash-verified Ledger consumer fact feed.

    ``manifest`` and ``manifest_sha256`` are ``None`` when the feed was a
    bare ``consumer_facts.jsonl`` file rather than an artifact directory;
    bare feeds are still content-addressed by ``facts_sha256`` so builds can
    pin them, but they carry no Ledger-side provenance.
    """

    path: Path
    facts: tuple[dict[str, Any], ...]
    facts_sha256: str
    manifest: dict[str, Any] | None = None
    manifest_sha256: str | None = None

    @property
    def fact_row_count(self) -> int:
        """Number of consumer fact rows in the feed."""
        return len(self.facts)

    def provenance(self) -> dict[str, Any]:
        """Ledger-artifact identity block for build and release manifests."""
        payload: dict[str, Any] = {
            "path_name": self.path.name,
            "fact_row_count": self.fact_row_count,
            "facts_sha256": self.facts_sha256,
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
) -> LedgerConsumerArtifact:
    """Load a Ledger consumer artifact directory or bare consumer-facts file.

    Artifact directories are verified against their own manifest: the fact
    file's SHA-256 must match ``manifest.facts_sha256``. The optional
    ``expected_*`` pins let a build config assert the exact artifact it was
    reviewed against; any mismatch raises before facts are used.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Ledger consumer artifact not found: {artifact_path}")

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

    return LedgerConsumerArtifact(
        path=artifact_path,
        facts=_load_fact_rows(facts_path),
        facts_sha256=facts_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )


def _load_fact_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
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
            assertion = row.get("assertion", DEFAULT_LEDGER_ASSERTION)
            if assertion not in ALLOWED_LEDGER_ASSERTIONS:
                raise ValueError(
                    f"Ledger facts JSONL row {line_number} has unsupported "
                    f"assertion {assertion!r}; expected one of "
                    f"{sorted(ALLOWED_LEDGER_ASSERTIONS)}."
                )
            rows.append({**row, "assertion": assertion})
    if not rows:
        raise ValueError(f"Ledger facts feed is empty: {path}")
    return tuple(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
