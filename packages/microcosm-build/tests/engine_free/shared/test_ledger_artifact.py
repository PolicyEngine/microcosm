"""Tests for pinned Ledger consumer-artifact loading."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.ledger_artifact import (
    CONSUMER_ARTIFACT_SCHEMA_VERSION,
    ChronicleFeedPinMismatchError,
    LedgerConsumerArtifact,
    add_ledger_artifact_args,
    load_ledger_consumer_artifact,
    resolve_ledger_artifact,
    validate_chronicle_feed_pin,
)
from test_support.microcosm_build.chronicle_feed import write_pinned_artifact
from test_support.paths import paths_for


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


def _rewrite_manifest(artifact_dir, **overrides):
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(overrides)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _assert_pin_mismatch(error, field, expected, actual):
    assert isinstance(error, ChronicleFeedPinMismatchError)
    assert error.field == field
    assert error.expected == expected
    assert error.actual == actual
    message = str(error)
    assert field in message
    assert str(expected) in message
    assert str(actual) in message


def test_complete_feed_pin_accepts_exact_match(tmp_path):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])

    artifact = load_ledger_consumer_artifact(artifact_dir, feed_pin=pin)

    validate_chronicle_feed_pin(artifact, pin)
    assert artifact.facts_sha256 == pin["facts_sha256"]
    assert artifact.manifest_sha256 == pin["manifest_sha256"]
    assert artifact.fact_row_count == pin["fact_row_count"]


def test_feed_pin_rejects_changed_facts_with_self_consistent_manifest(tmp_path):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])
    facts_path = artifact_dir / "consumer_facts.jsonl"
    rows = [json.loads(line) for line in facts_path.read_text().splitlines()]
    rows[0]["value"] = 999
    changed_facts_sha256 = _write_facts(facts_path, rows)
    _rewrite_manifest(artifact_dir, facts_sha256=changed_facts_sha256)
    # The artifact's own checksum is valid; the reviewed feed pin must still
    # reject it before any consumer can use the substituted facts.
    assert load_ledger_consumer_artifact(artifact_dir).facts[0]["value"] == 999

    with pytest.raises(ChronicleFeedPinMismatchError) as exc:
        load_ledger_consumer_artifact(artifact_dir, feed_pin=pin)

    _assert_pin_mismatch(
        exc.value, "facts_sha256", pin["facts_sha256"], changed_facts_sha256
    )


def test_feed_pin_rejects_manifest_only_change(tmp_path):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])
    changed_manifest_sha256 = _rewrite_manifest(artifact_dir, producer_note="changed")
    assert (
        load_ledger_consumer_artifact(artifact_dir).facts_sha256 == pin["facts_sha256"]
    )

    with pytest.raises(ChronicleFeedPinMismatchError) as exc:
        load_ledger_consumer_artifact(artifact_dir, feed_pin=pin)

    _assert_pin_mismatch(
        exc.value, "manifest_sha256", pin["manifest_sha256"], changed_manifest_sha256
    )


@pytest.mark.parametrize(
    ("field", "altered"),
    [
        ("facts_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("artifact_schema_version", "policyengine_ledger.consumer_artifact.v2"),
        ("consumer_fact_schema_versions", ["chronicle.consumer_fact.v5"]),
        ("consumer_fact_schema_sha256", "0" * 64),
        ("fact_row_count", 2),
    ],
    ids=[
        "facts_sha256",
        "manifest_sha256",
        "artifact_schema_version",
        "consumer_fact_schema_versions",
        "consumer_fact_schema_sha256",
        "fact_row_count",
    ],
)
def test_feed_pin_rejects_each_identity_field(tmp_path, field, altered):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])
    changed_pin = {**pin, field: altered}

    with pytest.raises(ChronicleFeedPinMismatchError) as exc:
        load_ledger_consumer_artifact(artifact_dir, feed_pin=changed_pin)

    _assert_pin_mismatch(exc.value, field, altered, pin[field])


@pytest.mark.parametrize(
    "field",
    [
        "facts_sha256",
        "manifest_sha256",
        "artifact_schema_version",
        "consumer_fact_schema_versions",
        "consumer_fact_schema_sha256",
        "fact_row_count",
    ],
)
def test_feed_pin_requires_every_identity_field(tmp_path, field):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])
    del pin[field]

    with pytest.raises(ValueError, match=field):
        load_ledger_consumer_artifact(artifact_dir, feed_pin=pin)


@pytest.mark.parametrize(
    ("field", "manifest_value"),
    [
        ("consumer_fact_schema_versions", ["chronicle.consumer_fact.v5"]),
        ("fact_row_count", 2),
    ],
)
def test_feed_pin_checks_manifest_declarations_against_observed_rows(
    tmp_path, field, manifest_value
):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])
    # Re-pin the changed manifest bytes to isolate its false declaration from
    # the separate byte-identity check. Rows still declare v4 and number one.
    pin["manifest_sha256"] = _rewrite_manifest(artifact_dir, **{field: manifest_value})

    with pytest.raises(ChronicleFeedPinMismatchError) as exc:
        load_ledger_consumer_artifact(artifact_dir, feed_pin=pin)

    _assert_pin_mismatch(exc.value, field, pin[field], manifest_value)


def test_feed_pin_checks_observed_rows_when_manifest_matches_pin(tmp_path):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])
    artifact = load_ledger_consumer_artifact(artifact_dir)
    changed_row = {**artifact.facts[0], "schema_version": "chronicle.consumer_fact.v5"}
    changed_artifact = replace(artifact, facts=(changed_row,))

    with pytest.raises(ChronicleFeedPinMismatchError) as exc:
        validate_chronicle_feed_pin(changed_artifact, pin)

    _assert_pin_mismatch(
        exc.value,
        "consumer_fact_schema_versions",
        pin["consumer_fact_schema_versions"],
        ["chronicle.consumer_fact.v5"],
    )


def test_feed_pin_requires_artifact_manifest(tmp_path):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])

    with pytest.raises(ChronicleFeedPinMismatchError) as exc:
        load_ledger_consumer_artifact(
            artifact_dir / "consumer_facts.jsonl", feed_pin=pin
        )

    _assert_pin_mismatch(exc.value, "manifest_sha256", pin["manifest_sha256"], None)


def test_feed_pin_cannot_omit_schema_digest_even_when_manifest_omits_it(tmp_path):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])
    pin["manifest_sha256"] = _rewrite_manifest(
        artifact_dir, consumer_fact_schema_sha256=None
    )
    pin["consumer_fact_schema_sha256"] = None
    with pytest.raises(ChronicleFeedPinMismatchError) as exc:
        load_ledger_consumer_artifact(artifact_dir, feed_pin=pin)
    _assert_pin_mismatch(exc.value, "consumer_fact_schema_sha256", None, None)


def test_full_feed_pin_error_precedes_legacy_hash_error(tmp_path):
    artifact_dir, pin = write_pinned_artifact(tmp_path / "artifact", [_fact_row()])
    facts_path = artifact_dir / "consumer_facts.jsonl"
    rows = [json.loads(line) for line in facts_path.read_text().splitlines()]
    rows[0]["value"] = 999
    changed_hash = _write_facts(facts_path, rows)
    _rewrite_manifest(artifact_dir, facts_sha256=changed_hash)
    with pytest.raises(ChronicleFeedPinMismatchError) as exc:
        load_ledger_consumer_artifact(
            artifact_dir, feed_pin=pin, expected_facts_sha256=pin["facts_sha256"]
        )
    _assert_pin_mismatch(exc.value, "facts_sha256", pin["facts_sha256"], changed_hash)


@pytest.fixture(scope="module")
def valid_feed_artifact(tmp_path_factory):
    artifact_dir, pin = write_pinned_artifact(
        tmp_path_factory.mktemp("feed-pin-property"), [_fact_row()]
    )
    return load_ledger_consumer_artifact(artifact_dir), pin


@settings(max_examples=60, deadline=None, database=None)
@given(
    field=st.sampled_from(
        [
            "facts_sha256",
            "manifest_sha256",
            "artifact_schema_version",
            "consumer_fact_schema_versions",
            "consumer_fact_schema_sha256",
            "fact_row_count",
        ]
    ),
    delta=st.integers(min_value=1, max_value=65535),
)
def test_single_field_feed_pin_perturbation_is_always_rejected(
    valid_feed_artifact, field, delta
):
    artifact, pin = valid_feed_artifact
    validate_chronicle_feed_pin(artifact, pin)
    previous = pin[field]
    if field.endswith("sha256"):
        index = delta % len(previous)
        replacement = format((int(previous[index], 16) + 1 + delta % 15) % 16, "x")
        altered = previous[:index] + replacement + previous[index + 1 :]
    elif field == "fact_row_count":
        altered = previous + delta
    elif field == "consumer_fact_schema_versions":
        altered = [f"chronicle.consumer_fact.v{4 + delta}"]
    else:
        altered = f"policyengine_ledger.consumer_artifact.v{1 + delta}"

    with pytest.raises(ChronicleFeedPinMismatchError) as exc:
        validate_chronicle_feed_pin(artifact, {**pin, field: altered})

    _assert_pin_mismatch(exc.value, field, altered, previous)


def test_committed_feed_pin_preserves_uk_runtime_expectations():
    build_root = paths_for("microcosm-build").package / "src/microcosm/build"
    # Run the unchanged reader without importing the entire country runtime.
    # Its resource lookup and parsing remain exactly the shipped implementation.
    reader = runpy.run_path(str(build_root / "uk_runtime/chronicle_feed.py"))
    resource = build_root / "uk/chronicle_feed.json"
    raw = json.loads(resource.read_text())
    runtime_pin = reader["load_uk_chronicle_feed"]().to_dict()

    assert {key: runtime_pin[key] for key in raw} == raw
    assert (
        runtime_pin["resource_sha256"]
        == hashlib.sha256(resource.read_bytes()).hexdigest()
    )
    assert runtime_pin["resource_size_bytes"] == resource.stat().st_size


def test_real_nz_artifact_matches_committed_feed_pin():
    artifact_path = os.environ.get("NZ_CHRONICLE_ARTIFACT")
    if not artifact_path:
        pytest.skip(
            "NZ_CHRONICLE_ARTIFACT is unset; real NZ Chronicle artifact unavailable"
        )

    from microcosm.build import load_country_spec

    spec = load_country_spec("nz")
    assert "chronicle_feed.json" in spec.resources
    resource = (
        paths_for("microcosm-build").package
        / "src/microcosm/build/nz/chronicle_feed.json"
    )
    pin = json.loads(resource.read_text())

    artifact = load_ledger_consumer_artifact(Path(artifact_path), feed_pin=pin)

    validate_chronicle_feed_pin(artifact, pin)
    assert artifact.fact_row_count == pin["fact_row_count"]
    assert artifact.schema_version == pin["artifact_schema_version"]
    assert list(artifact.fact_schema_versions) == pin["consumer_fact_schema_versions"]
