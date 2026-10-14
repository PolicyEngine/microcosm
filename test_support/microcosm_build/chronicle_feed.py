"""Small self-consistent Chronicle artifact fixtures for engine-free tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from microcosm.build.ledger_artifact import CONSUMER_ARTIFACT_SCHEMA_VERSION


def write_pinned_artifact(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    """Write a small artifact and return its complete six-field feed pin."""
    path.mkdir(parents=True, exist_ok=True)
    facts = [{"schema_version": "chronicle.consumer_fact.v4", **row} for row in rows]
    facts_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in facts
    ).encode()
    (path / "consumer_facts.jsonl").write_bytes(facts_bytes)
    versions = sorted({str(row["schema_version"]) for row in facts})
    manifest = {
        "schema_version": CONSUMER_ARTIFACT_SCHEMA_VERSION,
        "facts_sha256": hashlib.sha256(facts_bytes).hexdigest(),
        "fact_row_count": len(facts),
        "consumer_fact_schema_versions": versions,
        "consumer_fact_schema_sha256": "ab" * 32,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (path / "manifest.json").write_bytes(manifest_bytes)
    pin = {
        "facts_sha256": manifest["facts_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "artifact_schema_version": manifest["schema_version"],
        "consumer_fact_schema_versions": versions,
        "consumer_fact_schema_sha256": manifest["consumer_fact_schema_sha256"],
        "fact_row_count": manifest["fact_row_count"],
    }
    return path, pin
