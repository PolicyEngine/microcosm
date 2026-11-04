"""Small self-consistent Chronicle artifact fixtures for engine-free tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from microcosm.build.ledger_artifact import CONSUMER_ARTIFACT_SCHEMA_VERSION
from test_support.paths import paths_for

#: Chronicle's v4 and v5 consumer-fact schemas and sample rows: v5 from
#: PolicyEngine/chronicle at 44eebe037 (the merge of chronicle#377), v4 from
#: 1a3f2c4 (Chronicle main just before it). The directory's SOURCE.md records
#: the paths and digests.
CHRONICLE_FIXTURES = (
    paths_for("microcosm-build").tests / "fixtures/chronicle_consumer_fact"
)
CHRONICLE_V4_SCHEMA_PATH = CHRONICLE_FIXTURES / "consumer_fact.v4.schema.json"
CHRONICLE_V4_FACTS_PATH = CHRONICLE_FIXTURES / "consumer_facts.v4.jsonl"
CHRONICLE_V5_SCHEMA_PATH = CHRONICLE_FIXTURES / "consumer_fact.v5.schema.json"
CHRONICLE_V5_FACTS_PATH = CHRONICLE_FIXTURES / "consumer_facts.v5.jsonl"

#: The field chronicle#377 added to the v5 row: a package's own wording of a
#: dimension value where it differs from the label the row exports.
PACKAGE_DIMENSION_VALUE_LABELS = "package_dimension_value_labels"

_AGI_DIMENSION = "us:statutes/26/62#adjusted_gross_income"

#: Values of ``package_dimension_value_labels`` that the v5 schema refuses: the
#: field must map each dimension id to an object of value id -> non-blank text.
MALFORMED_PACKAGE_DIMENSION_VALUE_LABELS: tuple[tuple[str, object], ...] = (
    ("not_an_object", ["All returns"]),
    ("null", None),
    ("dimension_not_an_object", {_AGI_DIMENSION: "All returns"}),
    ("dimension_null", {_AGI_DIMENSION: None}),
    ("label_empty", {_AGI_DIMENSION: {"all": ""}}),
    ("label_blank", {_AGI_DIMENSION: {"all": "   "}}),
    ("label_not_text", {_AGI_DIMENSION: {"all": 1}}),
    ("label_null", {_AGI_DIMENSION: {"all": None}}),
)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def chronicle_v5_schema() -> dict[str, Any]:
    """Chronicle's v5 consumer-fact JSON Schema, parsed."""
    return json.loads(CHRONICLE_V5_SCHEMA_PATH.read_bytes())


def chronicle_v5_schema_sha256() -> str:
    """The digest a v5 artifact manifest declares as its fact schema digest."""
    return hashlib.sha256(CHRONICLE_V5_SCHEMA_PATH.read_bytes()).hexdigest()


def chronicle_v5_schema_version() -> str:
    """The per-row ``schema_version`` the v5 schema requires, read from it."""
    return chronicle_v5_schema()["properties"]["schema_version"]["const"]


def chronicle_v5_fact_rows() -> list[dict[str, Any]]:
    """Fresh copies of Chronicle's sample v5 rows."""
    return _jsonl_rows(CHRONICLE_V5_FACTS_PATH)


def chronicle_v4_schema() -> dict[str, Any]:
    """Chronicle's v4 consumer-fact JSON Schema, parsed."""
    return json.loads(CHRONICLE_V4_SCHEMA_PATH.read_bytes())


def chronicle_v4_schema_sha256() -> str:
    """The digest the UK, NZ and Belgian feed pins carry for v4."""
    return hashlib.sha256(CHRONICLE_V4_SCHEMA_PATH.read_bytes()).hexdigest()


def chronicle_v4_fact_rows() -> list[dict[str, Any]]:
    """Fresh copies of Chronicle's sample v4 rows, from just before chronicle#377."""
    return _jsonl_rows(CHRONICLE_V4_FACTS_PATH)


def write_pinned_artifact(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    consumer_fact_schema_sha256: str = "ab" * 32,
) -> tuple[Path, dict[str, Any]]:
    """Write a small artifact and return its complete six-field feed pin.

    Rows without a ``schema_version`` are stamped v4; a row's own stamp is
    kept, so v5 rows write a v5 artifact.
    """
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
        "consumer_fact_schema_sha256": consumer_fact_schema_sha256,
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
