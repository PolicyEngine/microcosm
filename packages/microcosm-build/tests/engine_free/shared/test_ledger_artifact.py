"""Tests for pinned Ledger consumer-artifact loading."""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser, Namespace

import pytest

from microcosm.build.ledger_artifact import (
    CONSUMER_ARTIFACT_SCHEMA_VERSION,
    LedgerConsumerArtifact,
    add_ledger_artifact_args,
    load_ledger_consumer_artifact,
    resolve_ledger_artifact,
)


def _fact_row(**overrides):
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
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact_dir(tmp_path, rows, *, manifest_overrides=None):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    facts_sha = _write_facts(artifact_dir / "consumer_facts.jsonl", rows)
    manifest = {
        "schema_version": CONSUMER_ARTIFACT_SCHEMA_VERSION,
        "fact_row_count": len(rows),
        "facts_sha256": facts_sha,
        "profiles": {
            "us_fiscal": {"sha256": "ab" * 32, "target_count": 3},
        },
    }
    manifest.update(manifest_overrides or {})
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return artifact_dir


def test_loads_bare_consumer_facts_file(tmp_path):
    facts_path = tmp_path / "consumer_facts.jsonl"
    facts_sha = _write_facts(facts_path, [_fact_row()])

    artifact = load_ledger_consumer_artifact(facts_path)

    assert isinstance(artifact, LedgerConsumerArtifact)
    assert artifact.fact_row_count == 1
    assert artifact.facts_sha256 == facts_sha
    assert artifact.manifest is None
    # Legacy rows that omit the assertion field pass through untouched:
    # readers treat a missing assertion as observation-by-default, but the
    # loader must not fabricate the key (structural projection checks would
    # then see unlabeled projections as mistyped observations).
    assert "assertion" not in artifact.facts[0]
    provenance = artifact.provenance()
    assert provenance["facts_sha256"] == facts_sha
    assert provenance["schema_version"] is None


def test_loads_artifact_directory_and_records_provenance(tmp_path):
    rows = [_fact_row(), _fact_row(assertion="source_projection")]
    artifact_dir = _write_artifact_dir(tmp_path, rows)

    artifact = load_ledger_consumer_artifact(artifact_dir)

    assert artifact.fact_row_count == 2
    assert artifact.manifest["schema_version"] == CONSUMER_ARTIFACT_SCHEMA_VERSION
    assert artifact.facts[1]["assertion"] == "source_projection"
    provenance = artifact.provenance()
    assert provenance["schema_version"] == CONSUMER_ARTIFACT_SCHEMA_VERSION
    assert provenance["manifest_sha256"] == artifact.manifest_sha256
    assert provenance["profiles"]["us_fiscal"]["target_count"] == 3


def test_rejects_tampered_fact_rows(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_fact_row()])
    facts_path = artifact_dir / "consumer_facts.jsonl"
    tampered = json.loads(facts_path.read_text())
    tampered["value"] = 999
    facts_path.write_text(json.dumps(tampered, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="manifest hash"):
        load_ledger_consumer_artifact(artifact_dir)


def test_rejects_facts_pin_mismatch(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_fact_row()])

    with pytest.raises(ValueError, match="pinned hash"):
        load_ledger_consumer_artifact(
            artifact_dir,
            expected_facts_sha256="0" * 64,
        )


def test_rejects_manifest_pin_mismatch_and_bare_manifest_pin(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_fact_row()])
    with pytest.raises(ValueError, match="manifest does not match"):
        load_ledger_consumer_artifact(
            artifact_dir,
            expected_manifest_sha256="0" * 64,
        )

    facts_path = tmp_path / "bare.jsonl"
    _write_facts(facts_path, [_fact_row()])
    with pytest.raises(ValueError, match="no manifest"):
        load_ledger_consumer_artifact(
            facts_path,
            expected_manifest_sha256="0" * 64,
        )


def test_matching_pins_pass(tmp_path):
    artifact_dir = _write_artifact_dir(tmp_path, [_fact_row()])
    unpinned = load_ledger_consumer_artifact(artifact_dir)

    pinned = load_ledger_consumer_artifact(
        artifact_dir,
        expected_facts_sha256=unpinned.facts_sha256,
        expected_manifest_sha256=unpinned.manifest_sha256,
    )
    assert pinned.facts == unpinned.facts


def test_rejects_unknown_assertion_and_schema_version(tmp_path):
    facts_path = tmp_path / "bad_assertion.jsonl"
    _write_facts(facts_path, [_fact_row(assertion="policyengine_aged")])
    with pytest.raises(ValueError, match="unsupported assertion"):
        load_ledger_consumer_artifact(facts_path)

    artifact_dir = _write_artifact_dir(
        tmp_path,
        [_fact_row()],
        manifest_overrides={"schema_version": "policyengine_ledger.other.v9"},
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_ledger_consumer_artifact(artifact_dir)


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


def test_shared_cli_args_load_optional_ledger_artifact(tmp_path):
    facts_path = tmp_path / "consumer_facts.jsonl"
    facts_sha = _write_facts(facts_path, [_fact_row()])
    parser = ArgumentParser()
    add_ledger_artifact_args(parser)
    args = parser.parse_args(
        ["--ledger-facts", str(facts_path), "--ledger-facts-sha256", facts_sha]
    )

    artifact = resolve_ledger_artifact(args)

    assert artifact is not None
    assert artifact.facts_sha256 == facts_sha


def test_shared_cli_resolver_allows_absent_artifact_but_not_orphan_pins():
    assert resolve_ledger_artifact(Namespace(ledger_facts=None)) is None

    with pytest.raises(ValueError, match="requires --ledger-facts"):
        resolve_ledger_artifact(
            Namespace(
                ledger_facts=None,
                ledger_facts_sha256="0" * 64,
                ledger_manifest_sha256=None,
            )
        )
