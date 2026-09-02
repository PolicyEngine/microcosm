"""Pinned loading for PolicyEngine Chronicle consumer artifacts.

Chronicle (formerly Ledger) publishes consumer artifacts: a directory with
``manifest.json`` (schema version, content hashes, embedded profile hashes),
``consumer_facts.jsonl``, per-profile JSON, and coverage diagnostics. A
Microcosm build should consume facts through this loader so the release
manifest can record exactly which Chronicle data resolved its target values
(PolicyEngine/microcosm#160, #271) and so tampered or mismatched feeds fail
before they calibrate anything.

Like the rest of Microcosm's Chronicle consumption, this module is duck-typed
against the published artifact contract (stdlib only); it does not import the
Chronicle implementation package.

**Both eras load.** The manifest's ``schema_version`` is checked for
membership in :data:`ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS`, never for
equality with one era, so an artifact published after Chronicle's rename
cutover loads here without a code change — see
:mod:`microcosm.build.chronicle_epoch` and PolicyEngine/chronicle#143. The
observed id, its epoch, the fact-key epochs present in the feed, and the
per-row schema ids the rows themselves declare are all recorded in
:meth:`LedgerConsumerArtifact.provenance`, so a release manifest witnesses
which era it actually consumed rather than which era it assumed.

Acceptance widened here; nothing narrowed. Only the *manifest* schema id is
gated, and only against the two ids chronicle#143 declares. Row-level ids and
fact keys are carried as published — real feeds mint rows in namespaces that
belong to neither era, and a consumer that rejected them would fail closed on
data that has always loaded.
"""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microcosm.build.chronicle_epoch import (
    ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS,
    LEDGER_CONSUMER_ARTIFACT_SCHEMA_VERSION,
    consumer_artifact_schema_epoch,
    describe_accepted_consumer_artifact_schema_versions,
    feed_fact_key_epochs,
    is_accepted_consumer_artifact_schema_version,
)

__all__ = [
    "ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS",
    "ALLOWED_LEDGER_ASSERTIONS",
    "CONSUMER_ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_LEDGER_ASSERTION",
    "LedgerConsumerArtifact",
    "add_ledger_artifact_args",
    "load_ledger_consumer_artifact",
    "resolve_ledger_artifact",
]

#: The era Microcosm's own minted artifacts still declare. Frozen at v1 by
#: microcosm#639; loading is governed by the accepted *set*, not by this.
CONSUMER_ARTIFACT_SCHEMA_VERSION = LEDGER_CONSUMER_ARTIFACT_SCHEMA_VERSION
ALLOWED_LEDGER_ASSERTIONS = frozenset(("observation", "source_projection"))
DEFAULT_LEDGER_ASSERTION = "observation"


@dataclass(frozen=True)
class LedgerConsumerArtifact:
    """A loaded, hash-verified Chronicle consumer fact feed.

    ``manifest`` and ``manifest_sha256`` are ``None`` when the feed was a
    bare ``consumer_facts.jsonl`` file rather than an artifact directory;
    bare feeds are still content-addressed by ``facts_sha256`` so builds can
    pin them, but they carry no Chronicle-side provenance. Rows are exactly as
    published: an ``assertion`` field is validated when present and never
    fabricated when absent (missing means observation-by-default to readers).

    The class name is ledger-era and stays: renaming an exported symbol
    Microcosm's tools and experiments import buys nothing the alias in
    :mod:`microcosm.build` does not, and chronicle#143 migrates identities,
    not vocabulary.
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

    @property
    def schema_version(self) -> str | None:
        """The schema id this feed actually declared, verbatim."""
        if self.manifest is None:
            return None
        schema_version = self.manifest.get("schema_version")
        return None if schema_version is None else str(schema_version)

    @property
    def schema_epoch(self) -> str | None:
        """``"ledger"`` or ``"chronicle"`` for the declared schema id."""
        return consumer_artifact_schema_epoch(self.schema_version)

    @property
    def fact_key_epochs(self) -> tuple[str, ...]:
        """Chronicle epochs observed across the feed's fact keys.

        Empty when the feed carries only Microcosm-minted keys, and both
        epochs when a cutover-window feed mixes ledger-era history with
        chronicle-era rows.
        """
        return feed_fact_key_epochs(self.facts)

    @property
    def fact_schema_versions(self) -> tuple[str, ...]:
        """Distinct per-row ``schema_version`` values in the feed, sorted.

        Reported verbatim and never gated on — see :func:`_load_fact_rows`.
        Empty when no row declares one; more than one entry when a feed mixes
        producers, which the pinned US fiscal-refresh feed already does.
        """
        observed = {
            str(row["schema_version"])
            for row in self.facts
            if isinstance(row, dict) and row.get("schema_version") is not None
        }
        return tuple(sorted(observed))

    def provenance(self) -> dict[str, Any]:
        """Chronicle-artifact identity block for build and release manifests.

        Records the schema id as *observed*, plus the epoch it belongs to and
        the epochs of the fact keys in the feed, so a manifest witnesses which
        era of Chronicle actually resolved its targets.
        """
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
        payload["schema_epoch"] = self.schema_epoch
        payload["fact_key_epochs"] = list(self.fact_key_epochs)
        payload["fact_schema_versions"] = list(self.fact_schema_versions)
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
                f"Ledger consumer artifact has no consumer_facts.jsonl: {artifact_path}"
            )
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError(
                f"Ledger consumer artifact manifest must be an object: {manifest_path}"
            )
        schema_version = manifest.get("schema_version")
        # Membership, not equality: ledger-era and chronicle-era artifacts are
        # the same contract under two names, and both must load through the
        # rename cutover (chronicle#143).
        if not is_accepted_consumer_artifact_schema_version(schema_version):
            raise ValueError(
                "Unsupported Chronicle consumer artifact schema_version "
                f"{schema_version!r}; expected one of "
                f"{describe_accepted_consumer_artifact_schema_versions()}."
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
    if expected_facts_sha256 is not None and expected_facts_sha256 != facts_sha256:
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


def add_ledger_artifact_args(parser: ArgumentParser) -> None:
    """Add shared Ledger consumer-artifact arguments to a CLI parser."""

    parser.add_argument(
        "--ledger-facts",
        type=Path,
        help=(
            "PolicyEngine Ledger consumer artifact directory (manifest.json "
            "+ consumer_facts.jsonl) or a bare consumer_facts.jsonl file."
        ),
    )
    parser.add_argument(
        "--ledger-facts-sha256",
        help="Pin: expected SHA-256 of consumer_facts.jsonl.",
    )
    parser.add_argument(
        "--ledger-manifest-sha256",
        help=(
            "Pin: expected SHA-256 of the Ledger consumer artifact manifest.json; "
            "requires an artifact directory feed."
        ),
    )


def resolve_ledger_artifact(args: Namespace) -> LedgerConsumerArtifact | None:
    """Load the CLI-selected Ledger artifact, returning ``None`` when absent."""

    ledger_facts = getattr(args, "ledger_facts", None)
    if ledger_facts is None:
        if getattr(args, "ledger_facts_sha256", None) is not None:
            raise ValueError("--ledger-facts-sha256 requires --ledger-facts.")
        if getattr(args, "ledger_manifest_sha256", None) is not None:
            raise ValueError("--ledger-manifest-sha256 requires --ledger-facts.")
        return None
    return load_ledger_consumer_artifact(
        ledger_facts,
        expected_facts_sha256=getattr(args, "ledger_facts_sha256", None),
        expected_manifest_sha256=getattr(args, "ledger_manifest_sha256", None),
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
                    f"Invalid Chronicle facts JSONL row {line_number}: expected "
                    f"object, got {type(row).__name__}."
                )
            # A per-row ``schema_version`` is carried, never gated on. Real
            # feeds stamp ids from namespaces that are neither era: the pinned
            # US fiscal-refresh feed (consumer_facts_buildn_v9_4.jsonl) mixes
            # 'arch.consumer_fact.v1' with 'ledger.consumer_fact.v1'. Rejecting
            # an unrecognized id would fail the build closed on its own pinned
            # input, and rejecting is not what dual acceptance asks for. The
            # observed ids are reported through
            # :attr:`LedgerConsumerArtifact.fact_schema_versions` instead.
            assertion = row.get("assertion", DEFAULT_LEDGER_ASSERTION)
            if assertion not in ALLOWED_LEDGER_ASSERTIONS:
                raise ValueError(
                    f"Ledger facts JSONL row {line_number} has unsupported "
                    f"assertion {assertion!r}; expected one of "
                    f"{sorted(ALLOWED_LEDGER_ASSERTIONS)}."
                )
            # Do not stamp the default onto rows that omit the field: legacy
            # feeds predate the assertion schema, and fabricating the key
            # would make downstream assertion checks treat unlabeled
            # publisher projections as mistyped observations.
            rows.append(row)
    if not rows:
        raise ValueError(f"Ledger facts feed is empty: {path}")
    return tuple(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
