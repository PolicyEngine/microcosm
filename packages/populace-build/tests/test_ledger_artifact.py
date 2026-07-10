"""Tests for pinned Ledger consumer-artifact loading."""

from __future__ import annotations

import hashlib
import json

import pytest

from populace.build.ledger_artifact import (
    CONSUMER_ARTIFACT_SCHEMA_VERSION,
    LedgerConsumerArtifact,
    load_ledger_consumer_artifact,
    vendor_ledger_consumer_artifact,
    verify_vendored_ledger_artifact,
)
from populace.build.ledger_schema import CONSUMER_FACT_SCHEMA_SHA256

# 24 lowercase-hex characters: the identity-key suffix the schema pins.
_HEX = "0123456789abcdef01234567"


def _key(prefix: str, tag: str) -> str:
    suffix = (hashlib.sha256(tag.encode()).hexdigest())[:24]
    return f"ledger.{prefix}.v2:{suffix}"


def _schema_complete_fact_row(tag: str = "agi", **overrides):
    """A row satisfying every required field of consumer_fact.v1."""
    row = {
        "schema_version": "ledger.consumer_fact.v1",
        "aggregate_fact_key": _key("aggregate_fact", tag),
        "semantic_fact_key": _key("semantic_fact", tag),
        "legacy_fact_key": f"ledger.fact.v1:{_HEX}",
        "source_release_key": f"ledger.source_release.v2:{_HEX}",
        "source_series_key": f"ledger.source_series.v2:{_HEX}",
        "observed_measure_key": f"ledger.observed_measure.v2:{_HEX}",
        "dimension_set_key": f"ledger.dimension_set.v2:{_HEX}",
        "universe_constraint_set_key": f"ledger.universe_constraint_set.v2:{_HEX}",
        "value": 100,
        "value_type": "integer",
        "assertion": "observation",
        "period": {"type": "tax_year", "value": 2023},
        "geography": {"level": "country", "id": "0100000US"},
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "observed_measure": {
            "source_name": "irs_soi",
            "source_table": "t",
            "source_measure_id": "agi",
            "source_concept": "irs_soi.agi",
            "unit": "usd",
        },
        "dimensions": {"income_range": "all"},
        "universe_constraints": {"domain": "all_individual_income_tax_returns"},
        "source": {
            "source_name": "irs_soi",
            "source_table": "t",
            "source_file": "f.xls",
            "vintage": "2023",
            "extracted_at": "2024-01-01T00:00:00Z",
            "extraction_method": "manual",
            "source_sha256": "ab" * 32,
            "source_size_bytes": 1024,
            "raw_r2_uri": "r2://ledger/f.xls",
        },
        "lineage": {
            "source_record_id": "irs_soi.ty2023.t.all.agi",
            "source_cell_keys": ["ledger.source_cell.v1:cell"],
        },
    }
    row.update(overrides)
    return row


def _legacy_fact_row(**overrides):
    """A deliberately minimal, pre-schema row (no assertion key)."""
    row = {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:abc123",
        "value": 100,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {"level": "country", "id": "0100000US"},
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "observed_measure": {"source_name": "irs_soi", "unit": "usd"},
        "source": {"source_name": "irs_soi"},
        "lineage": {"source_record_id": "irs_soi.ty2023.t.all.agi"},
    }
    row.update(overrides)
    return row


def _write_facts(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact_dir(
    tmp_path,
    rows,
    *,
    manifest_overrides=None,
    profile_body=b'{"targets": []}',
    profile_trailing_newline=True,
    declared_profile_sha=None,
):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    facts_sha = _write_facts(artifact_dir / "consumer_facts.jsonl", rows)
    profiles_dir = artifact_dir / "profiles"
    profiles_dir.mkdir()
    profile_bytes = profile_body + (b"\n" if profile_trailing_newline else b"")
    (profiles_dir / "us_fiscal.json").write_bytes(profile_bytes)
    profile_sha = declared_profile_sha or hashlib.sha256(profile_bytes).hexdigest()
    manifest = {
        "schema_version": CONSUMER_ARTIFACT_SCHEMA_VERSION,
        "fact_row_count": len(rows),
        "facts_sha256": facts_sha,
        "profiles": {
            "us_fiscal": {"sha256": profile_sha, "target_count": 3},
        },
    }
    manifest.update(manifest_overrides or {})
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return artifact_dir


def test_loads_bare_consumer_facts_file(tmp_path):
    facts_path = tmp_path / "consumer_facts.jsonl"
    facts_sha = _write_facts(facts_path, [_legacy_fact_row()])

    # Bare, pre-schema rows are only accepted with validation off: they predate
    # the pinned contract and deliberately omit the assertion key.
    artifact = load_ledger_consumer_artifact(facts_path, validate_rows=False)

    assert isinstance(artifact, LedgerConsumerArtifact)
    assert artifact.fact_row_count == 1
    assert artifact.facts_sha256 == facts_sha
    assert artifact.manifest is None
    assert artifact.rows_validated is False
    # Legacy rows that omit the assertion field pass through untouched: readers
    # treat a missing assertion as observation-by-default, but the loader must
    # not fabricate the key.
    assert "assertion" not in artifact.facts[0]
    provenance = artifact.provenance()
    assert provenance["facts_sha256"] == facts_sha
    assert provenance["schema_version"] is None
    assert provenance["consumer_fact_schema_sha256"] is None


def test_loads_artifact_directory_and_records_provenance(tmp_path):
    rows = [
        _schema_complete_fact_row(tag="agi"),
        _schema_complete_fact_row(tag="tax", assertion="source_projection"),
    ]
    artifact_dir = _write_artifact_dir(tmp_path, rows)

    artifact = load_ledger_consumer_artifact(artifact_dir)

    assert artifact.fact_row_count == 2
    assert artifact.manifest["schema_version"] == CONSUMER_ARTIFACT_SCHEMA_VERSION
    assert artifact.facts[1]["assertion"] == "source_projection"
    assert artifact.rows_validated is True
    provenance = artifact.provenance()
    assert provenance["schema_version"] == CONSUMER_ARTIFACT_SCHEMA_VERSION
    assert provenance["manifest_sha256"] == artifact.manifest_sha256
    assert provenance["consumer_fact_schema_sha256"] == CONSUMER_FACT_SCHEMA_SHA256
    assert provenance["profiles"]["us_fiscal"]["target_count"] == 3
    assert provenance["profiles"]["us_fiscal"]["hash_semantics"] == "exact"
    assert provenance["path"].endswith("artifact")


def test_validate_rows_rejects_structurally_incomplete_row(tmp_path):
    incomplete = _schema_complete_fact_row()
    del incomplete["entity"]
    artifact_dir = _write_artifact_dir(tmp_path, [incomplete])

    with pytest.raises(ValueError, match="'entity' is required but missing"):
        load_ledger_consumer_artifact(artifact_dir)


def test_rejects_duplicate_aggregate_fact_key(tmp_path):
    rows = [
        _schema_complete_fact_row(tag="dup"),
        _schema_complete_fact_row(tag="dup", value=200),
    ]
    artifact_dir = _write_artifact_dir(tmp_path, rows)

    with pytest.raises(ValueError, match="repeats aggregate_fact_key"):
        load_ledger_consumer_artifact(artifact_dir)


def test_rejects_tampered_fact_rows(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])
    facts_path = artifact_dir / "consumer_facts.jsonl"
    tampered = json.loads(facts_path.read_text())
    tampered["value"] = 999
    facts_path.write_text(json.dumps(tampered, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="manifest hash"):
        load_ledger_consumer_artifact(artifact_dir)


def test_rejects_facts_pin_mismatch(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])

    with pytest.raises(ValueError, match="pinned hash"):
        load_ledger_consumer_artifact(
            artifact_dir,
            expected_facts_sha256="0" * 64,
        )


def test_rejects_manifest_pin_mismatch_and_bare_manifest_pin(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])
    with pytest.raises(ValueError, match="manifest does not match"):
        load_ledger_consumer_artifact(
            artifact_dir,
            expected_manifest_sha256="0" * 64,
        )

    facts_path = tmp_path / "bare.jsonl"
    _write_facts(facts_path, [_schema_complete_fact_row()])
    with pytest.raises(ValueError, match="no manifest"):
        load_ledger_consumer_artifact(
            facts_path,
            expected_manifest_sha256="0" * 64,
        )


def test_matching_pins_pass(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])
    unpinned = load_ledger_consumer_artifact(artifact_dir)

    pinned = load_ledger_consumer_artifact(
        artifact_dir,
        expected_facts_sha256=unpinned.facts_sha256,
        expected_manifest_sha256=unpinned.manifest_sha256,
    )
    assert pinned.facts == unpinned.facts


def test_require_pins_rejects_bare_feed(tmp_path):
    facts_path = tmp_path / "bare.jsonl"
    _write_facts(facts_path, [_schema_complete_fact_row()])

    with pytest.raises(ValueError, match="must be a hash-pinned artifact directory"):
        load_ledger_consumer_artifact(
            facts_path,
            expected_facts_sha256="0" * 64,
            expected_manifest_sha256="0" * 64,
            require_pins=True,
        )


def test_require_pins_requires_both_pins(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])
    loaded = load_ledger_consumer_artifact(artifact_dir)

    with pytest.raises(ValueError, match="demands both"):
        load_ledger_consumer_artifact(
            artifact_dir,
            expected_facts_sha256=loaded.facts_sha256,
            require_pins=True,
        )
    with pytest.raises(ValueError, match="demands both"):
        load_ledger_consumer_artifact(
            artifact_dir,
            expected_manifest_sha256=loaded.manifest_sha256,
            require_pins=True,
        )


def test_require_pins_rejects_wrong_pin(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])
    loaded = load_ledger_consumer_artifact(artifact_dir)

    with pytest.raises(ValueError, match="pinned hash"):
        load_ledger_consumer_artifact(
            artifact_dir,
            expected_facts_sha256="0" * 64,
            expected_manifest_sha256=loaded.manifest_sha256,
            require_pins=True,
        )


def test_require_pins_passes_with_matching_pins(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])
    loaded = load_ledger_consumer_artifact(artifact_dir)

    pinned = load_ledger_consumer_artifact(
        artifact_dir,
        expected_facts_sha256=loaded.facts_sha256,
        expected_manifest_sha256=loaded.manifest_sha256,
        require_pins=True,
    )
    assert pinned.require_pins is True
    assert pinned.provenance()["require_pins"] is True
    assert pinned.provenance()["profiles"]["us_fiscal"]["hash_semantics"] == "exact"


def test_require_pins_rejects_missing_profile_file(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])
    loaded = load_ledger_consumer_artifact(artifact_dir)
    (artifact_dir / "profiles" / "us_fiscal.json").unlink()

    with pytest.raises(ValueError, match="no profile file was found"):
        load_ledger_consumer_artifact(
            artifact_dir,
            expected_facts_sha256=loaded.facts_sha256,
            expected_manifest_sha256=loaded.manifest_sha256,
            require_pins=True,
        )


def test_profile_hash_legacy_pre_newline_semantics(tmp_path):
    body = b'{"targets": [1, 2, 3]}'
    pre_newline_sha = hashlib.sha256(body).hexdigest()
    artifact_dir = _write_artifact_dir(
        tmp_path,
        [_schema_complete_fact_row()],
        profile_body=body,
        profile_trailing_newline=True,
        declared_profile_sha=pre_newline_sha,
    )

    artifact = load_ledger_consumer_artifact(artifact_dir)
    assert (
        artifact.provenance()["profiles"]["us_fiscal"]["hash_semantics"]
        == "legacy_pre_newline"
    )


def test_profile_hash_mismatch_rejected(tmp_path):
    artifact_dir = _write_artifact_dir(
        tmp_path,
        [_schema_complete_fact_row()],
        declared_profile_sha="0" * 64,
    )

    with pytest.raises(ValueError, match="does not match its manifest hash"):
        load_ledger_consumer_artifact(artifact_dir)


def test_vendor_and_verify_roundtrip(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])
    loaded = load_ledger_consumer_artifact(
        artifact_dir,
        expected_facts_sha256=None,
    )

    dest = tmp_path / "release" / "ledger_artifact"
    vendored = vendor_ledger_consumer_artifact(
        loaded,
        dest,
        verified_facts_sha256=loaded.facts_sha256,
        verified_manifest_sha256=loaded.manifest_sha256,
    )

    paths = {entry["path"] for entry in vendored["files"]}
    assert "manifest.json" in paths
    assert "consumer_facts.jsonl" in paths
    assert "profiles/us_fiscal.json" in paths
    assert vendored["verified_facts_sha256"] == loaded.facts_sha256
    assert vendored["consumer_fact_schema_sha256"] == CONSUMER_FACT_SCHEMA_SHA256
    # The vendored consumer facts reload byte-for-byte to the same content hash.
    assert (
        hashlib.sha256((dest / "consumer_facts.jsonl").read_bytes()).hexdigest()
        == loaded.facts_sha256
    )
    # A clean vendored copy verifies.
    verify_vendored_ledger_artifact(dest)


def test_vendor_detects_tampered_copy(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_schema_complete_fact_row()])
    loaded = load_ledger_consumer_artifact(artifact_dir)
    dest = tmp_path / "release" / "ledger_artifact"
    vendor_ledger_consumer_artifact(loaded, dest)

    tampered = json.loads((dest / "consumer_facts.jsonl").read_text())
    tampered["value"] = 424242
    (dest / "consumer_facts.jsonl").write_text(json.dumps(tampered) + "\n")

    with pytest.raises(ValueError, match="was tampered"):
        verify_vendored_ledger_artifact(dest)


def test_rejects_unknown_assertion_and_schema_version(tmp_path):
    facts_path = tmp_path / "bad_assertion.jsonl"
    _write_facts(facts_path, [_schema_complete_fact_row(assertion="policyengine_aged")])
    # Validation on (default): the pinned schema's assertion enum rejects it.
    with pytest.raises(ValueError, match="is not one of"):
        load_ledger_consumer_artifact(facts_path)
    # Legacy passthrough still guards the assertion enum without the schema.
    with pytest.raises(ValueError, match="unsupported assertion"):
        load_ledger_consumer_artifact(facts_path, validate_rows=False)

    artifact_dir = _write_artifact_dir(
        tmp_path,
        [_schema_complete_fact_row()],
        manifest_overrides={"schema_version": "policyengine_ledger.other.v9"},
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_ledger_consumer_artifact(artifact_dir)


def test_sol_counterexample_non_finite_fact_number_is_rejected(tmp_path):
    # json.dumps writes the bare tokens NaN / Infinity / -Infinity, which
    # json.loads accepts by default. The feed loader must reject them at parse
    # time rather than compile a non-finite value into a target (finding #7).
    for bad in (float("nan"), float("inf"), float("-inf")):
        facts_path = tmp_path / "nonfinite.jsonl"
        _write_facts(facts_path, [_schema_complete_fact_row(value=bad)])
        raw = facts_path.read_text()
        assert ("NaN" in raw) or ("Infinity" in raw)
        with pytest.raises(ValueError, match="non-finite JSON constant|not finite"):
            load_ledger_consumer_artifact(facts_path)
        # Legacy passthrough (validation off) must also refuse the constant.
        with pytest.raises(ValueError, match="non-finite JSON constant|not finite"):
            load_ledger_consumer_artifact(facts_path, validate_rows=False)


def test_rejects_missing_and_empty_feeds(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ledger_consumer_artifact(tmp_path / "missing.jsonl")

    empty_dir = tmp_path / "empty_artifact"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_ledger_consumer_artifact(empty_dir)

    empty_facts = tmp_path / "empty.jsonl"
    empty_facts.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load_ledger_consumer_artifact(empty_facts)
