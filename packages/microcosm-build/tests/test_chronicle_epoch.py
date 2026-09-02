"""Dual-era acceptance of Chronicle (formerly Ledger) identities.

Chronicle introduces chronicle-era hash domains and chronicle-named schema
ids for newly emitted rows at a declared cutover; ledger-era ids stay valid
history forever (PolicyEngine/chronicle#143). Microcosm consumes both, and
these tests are the contract that says so: no validator may reject a row for
being on the other side of the cutover, and no minted Microcosm identity may
move because a source row crossed it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from microcosm.build.chronicle_epoch import (
    ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS,
    ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS,
    CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION,
    CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
    CHRONICLE_EPOCH,
    EPOCHS,
    FACT_KEY_FIELDS,
    LEDGER_CONSUMER_ARTIFACT_SCHEMA_VERSION,
    LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
    LEDGER_EPOCH,
    LEDGER_FACT_KEY_DOMAINS,
    consumer_artifact_schema_epoch,
    fact_key_epoch,
    feed_fact_key_epochs,
    is_accepted_consumer_artifact_schema_version,
    is_accepted_consumer_fact_schema_version,
    is_chronicle_fact_key,
    parse_fact_key,
    row_fact_key_epochs,
)
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact

#: A chronicle-era row as the migration spec describes it: identical canonical
#: payload, ``chronicle.*.v3`` domain in place of ``ledger.*.v2``.
_CHRONICLE_AGGREGATE_KEY = "chronicle.aggregate_fact.v3:abc123"
_LEDGER_AGGREGATE_KEY = "ledger.aggregate_fact.v2:abc123"


def _fact_row(**overrides):
    row = {
        "aggregate_fact_key": _LEDGER_AGGREGATE_KEY,
        "semantic_fact_key": "ledger.semantic_fact.v2:abc123",
        "value": 100,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {"level": "country", "id": "0100000US"},
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "observed_measure": {"source_name": "irs_soi", "unit": "usd"},
        "source": {"source_name": "irs_soi"},
        "lineage": {
            "source_record_id": "irs_soi.ty2023.t.all.agi",
            "source_cell_keys": ["ledger.source_cell.v1:cell"],
        },
    }
    row.update(overrides)
    return row


def _chronicle_fact_row(**overrides):
    """The same row after Chronicle's cutover: every key domain re-epoched."""
    row = _fact_row(
        aggregate_fact_key=_CHRONICLE_AGGREGATE_KEY,
        semantic_fact_key="chronicle.semantic_fact.v3:abc123",
    )
    row["lineage"] = dict(row["lineage"])
    row["lineage"]["source_cell_keys"] = ["chronicle.source_cell.v3:cell"]
    row.update(overrides)
    return row


def _write_artifact_dir(tmp_path, rows, *, schema_version, name="artifact"):
    artifact_dir = tmp_path / name
    artifact_dir.mkdir()
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    (artifact_dir / "consumer_facts.jsonl").write_text(payload)
    manifest = {
        "schema_version": schema_version,
        "fact_row_count": len(rows),
        "facts_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "profiles": {"us_fiscal": {"sha256": "ab" * 32, "target_count": 3}},
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return artifact_dir


def test_parse_fact_key_splits_domain_namespace_family_and_version() -> None:
    identity = parse_fact_key(_LEDGER_AGGREGATE_KEY)

    assert identity is not None
    assert identity.domain == "ledger.aggregate_fact.v2"
    assert identity.namespace == "ledger"
    assert identity.family == "aggregate_fact"
    assert identity.version == "v2"
    assert identity.digest == "abc123"
    assert identity.epoch == LEDGER_EPOCH


@pytest.mark.parametrize("domain", sorted(LEDGER_FACT_KEY_DOMAINS.values()))
def test_every_observed_ledger_era_domain_resolves_to_the_ledger_epoch(
    domain: str,
) -> None:
    assert fact_key_epoch(f"{domain}:digest") == LEDGER_EPOCH


def test_chronicle_era_keys_resolve_without_a_declared_version_number() -> None:
    """Epoch detection is structural, so undeclared families still resolve.

    chronicle#143 names the ``v3`` spelling for the aggregate and semantic
    fact families; it does not say which version number the remaining
    families take. A validator that answered "unknown" for those would
    fail closed on the cutover, so detection reads the namespace segment
    rather than matching a frozen list of domain strings.
    """
    for key in (
        _CHRONICLE_AGGREGATE_KEY,
        "chronicle.semantic_fact.v3:abc123",
        "chronicle.source_cell.v7:cell",
        "chronicle.some_family_nobody_has_declared_yet.v11:digest",
    ):
        assert fact_key_epoch(key) == CHRONICLE_EPOCH
        assert is_chronicle_fact_key(key)


def test_microcosm_minted_namespaces_are_outside_both_epochs() -> None:
    """Microcosm's own derived keys must not be mistaken for Chronicle's.

    They are frozen at v1 by microcosm#639 and carry no Chronicle epoch, so
    the cutover cannot re-identify them.
    """
    for key in (
        "microcosm.derived_fact.congressional_district_vintage.v1:deadbeef",
        "microcosm.semantic_fact.congressional_district_state_total_proxy.v1:dead",
        "populace_us_trade.aggregate_fact.v1:digest",
        "populace_us_trade.semantic_fact.v1:digest",
    ):
        assert fact_key_epoch(key) is None
        assert not is_chronicle_fact_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "",
        "ledger.aggregate_fact.v2",  # no digest
        "ledger.aggregate_fact:digest",  # no version
        "ledger:digest",  # no family
        ":digest",
        "not a key at all",
        None,
        42,
    ],
)
def test_non_key_shapes_parse_to_none_rather_than_raising(key) -> None:
    assert parse_fact_key(key) is None
    assert fact_key_epoch(key) is None


def test_row_epochs_cover_every_key_field_including_nested_lists() -> None:
    ledger_row = _fact_row()
    chronicle_row = _chronicle_fact_row()

    assert row_fact_key_epochs(ledger_row) == frozenset({LEDGER_EPOCH})
    assert row_fact_key_epochs(chronicle_row) == frozenset({CHRONICLE_EPOCH})

    # A row whose only chronicle-era key is a nested source-cell key still
    # reports both eras: lineage keys are Chronicle identities too.
    straddling = _fact_row()
    straddling["lineage"] = dict(straddling["lineage"])
    straddling["lineage"]["source_cell_keys"] = ["chronicle.source_cell.v3:cell"]
    assert row_fact_key_epochs(straddling) == frozenset({LEDGER_EPOCH, CHRONICLE_EPOCH})


def test_row_epochs_read_the_concept_alignment_key() -> None:
    row = _fact_row(
        concept_alignment={
            "concept_alignment_key": "chronicle.concept_alignment.v3:aligned",
            "relation": "source_label",
        }
    )

    assert row_fact_key_epochs(row) == frozenset({LEDGER_EPOCH, CHRONICLE_EPOCH})


def test_feed_epochs_report_a_mixed_feed_in_epoch_order() -> None:
    assert feed_fact_key_epochs([_fact_row()]) == (LEDGER_EPOCH,)
    assert feed_fact_key_epochs([_chronicle_fact_row()]) == (CHRONICLE_EPOCH,)
    assert feed_fact_key_epochs([_chronicle_fact_row(), _fact_row()]) == EPOCHS
    assert (
        feed_fact_key_epochs([{"aggregate_fact_key": "populace_us_trade.a.v1:x"}]) == ()
    )
    assert feed_fact_key_epochs([]) == ()


def test_schema_id_membership_covers_both_eras_and_nothing_else() -> None:
    assert ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS == {
        LEDGER_CONSUMER_ARTIFACT_SCHEMA_VERSION,
        CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION,
    }
    assert ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS == {
        LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
        CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
    }
    assert is_accepted_consumer_artifact_schema_version(
        CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION
    )
    assert is_accepted_consumer_fact_schema_version(
        CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION
    )
    assert not is_accepted_consumer_artifact_schema_version(
        "policyengine_chronicle.other.v9"
    )
    assert not is_accepted_consumer_fact_schema_version("chronicle.consumer_fact.v1")


@pytest.mark.parametrize(
    ("schema_version", "expected_epoch"),
    [
        (LEDGER_CONSUMER_ARTIFACT_SCHEMA_VERSION, LEDGER_EPOCH),
        (CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION, CHRONICLE_EPOCH),
    ],
)
def test_artifact_loads_under_each_schema_id_and_records_the_observed_one(
    tmp_path, schema_version: str, expected_epoch: str
) -> None:
    rows = [
        _fact_row() if expected_epoch == LEDGER_EPOCH else _chronicle_fact_row(),
    ]
    artifact_dir = _write_artifact_dir(tmp_path, rows, schema_version=schema_version)

    artifact = load_ledger_consumer_artifact(artifact_dir)

    assert artifact.fact_row_count == 1
    assert artifact.schema_version == schema_version
    assert artifact.schema_epoch == expected_epoch
    provenance = artifact.provenance()
    # The id is recorded as observed, not as assumed.
    assert provenance["schema_version"] == schema_version
    assert provenance["schema_epoch"] == expected_epoch
    assert provenance["fact_key_epochs"] == [expected_epoch]
    assert (
        consumer_artifact_schema_epoch(provenance["schema_version"]) == expected_epoch
    )


def test_mixed_epoch_feed_loads_and_witnesses_both_eras(tmp_path) -> None:
    """The cutover window: ledger-era history beside chronicle-era rows.

    Both must calibrate, and the manifest must say the feed straddled the
    cutover rather than silently reporting one era.
    """
    rows = [
        _fact_row(),
        _chronicle_fact_row(),
        # Microcosm-minted rows carry neither epoch and must not perturb it.
        _fact_row(
            aggregate_fact_key="populace_us_trade.aggregate_fact.v1:digest",
            semantic_fact_key="populace_us_trade.semantic_fact.v1:digest",
            lineage={"source_record_id": "us_trade.month_2025_01.national"},
        ),
    ]
    artifact_dir = _write_artifact_dir(
        tmp_path, rows, schema_version=CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION
    )

    artifact = load_ledger_consumer_artifact(artifact_dir)

    assert artifact.fact_row_count == 3
    assert artifact.fact_key_epochs == EPOCHS
    assert artifact.provenance()["fact_key_epochs"] == list(EPOCHS)


def test_bare_feed_has_no_schema_epoch_but_still_reports_fact_key_epochs(
    tmp_path,
) -> None:
    facts_path = tmp_path / "consumer_facts.jsonl"
    facts_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (_fact_row(), _chronicle_fact_row())
        )
    )

    artifact = load_ledger_consumer_artifact(facts_path)

    provenance = artifact.provenance()
    assert provenance["schema_version"] is None
    assert provenance["schema_epoch"] is None
    assert provenance["fact_key_epochs"] == list(EPOCHS)


def test_unknown_schema_id_is_rejected_naming_both_accepted_eras(tmp_path) -> None:
    artifact_dir = _write_artifact_dir(
        tmp_path, [_fact_row()], schema_version="policyengine_chronicle.other.v9"
    )

    with pytest.raises(ValueError) as excinfo:
        load_ledger_consumer_artifact(artifact_dir)

    message = str(excinfo.value)
    assert "schema_version" in message
    assert LEDGER_CONSUMER_ARTIFACT_SCHEMA_VERSION in message
    assert CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION in message


@pytest.mark.parametrize(
    "row_schema_version",
    [LEDGER_CONSUMER_FACT_SCHEMA_VERSION, CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION],
)
def test_per_row_schema_id_is_accepted_in_either_era(
    tmp_path, row_schema_version: str
) -> None:
    facts_path = tmp_path / "consumer_facts.jsonl"
    facts_path.write_text(
        json.dumps(_fact_row(schema_version=row_schema_version), sort_keys=True) + "\n"
    )

    artifact = load_ledger_consumer_artifact(facts_path)

    assert artifact.facts[0]["schema_version"] == row_schema_version


def test_per_row_schema_id_is_optional(tmp_path) -> None:
    # A row that declares no schema id must not acquire one: the loader
    # reports what was published, and never fabricates the field.
    bare = tmp_path / "bare.jsonl"
    bare.write_text(json.dumps(_fact_row(), sort_keys=True) + "\n")
    assert "schema_version" not in load_ledger_consumer_artifact(bare).facts[0]


def test_per_row_schema_id_outside_both_eras_still_loads(tmp_path) -> None:
    """Dual acceptance widens what loads; it must not narrow it.

    Real feeds stamp rows from namespaces that are neither era. The pinned US
    fiscal-refresh feed ``consumer_facts_buildn_v9_4.jsonl`` declares
    ``arch.consumer_fact.v1`` on the overwhelming majority of its rows and
    ``ledger.consumer_fact.v1`` on the rest, and it loads through this loader
    on the release path. A consumer that gated the per-row id against the two
    ids chronicle#143 names would fail the build closed on its own pinned
    input — so the id is carried and reported, never gated.
    """
    facts_path = tmp_path / "consumer_facts.jsonl"
    rows = [
        _fact_row(schema_version="arch.consumer_fact.v1"),
        _fact_row(schema_version=LEDGER_CONSUMER_FACT_SCHEMA_VERSION),
        _chronicle_fact_row(schema_version=CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION),
        _fact_row(schema_version="ledger.consumer_fact.v99"),
    ]
    facts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )

    artifact = load_ledger_consumer_artifact(facts_path)

    assert artifact.fact_row_count == len(rows)
    assert artifact.fact_schema_versions == (
        "arch.consumer_fact.v1",
        CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
        LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
        "ledger.consumer_fact.v99",
    )
    # The unrecognized ids are reported verbatim in provenance, so a release
    # manifest still witnesses exactly what it consumed.
    assert artifact.provenance()["fact_schema_versions"] == [
        "arch.consumer_fact.v1",
        CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
        LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
        "ledger.consumer_fact.v99",
    ]


def test_every_published_key_field_is_witnessed_for_its_epoch() -> None:
    """Epoch witnessing covers every Chronicle key a published row carries.

    The captured feed fixture carries nine single-key fields and two key
    lists, not just the four identifiers targets resolve by. Chronicle's
    cutover moves families independently — the spec declares ``v3`` only for
    the aggregate and semantic families — so a row can straddle it: ledger-era
    aggregate key, chronicle-era source-release key. Reading only the
    resolution set would report that row as pure ledger-era.
    """
    for field in (
        "observed_measure_key",
        "source_release_key",
        "source_series_key",
        "universe_constraint_set_key",
    ):
        straddling = _fact_row(**{field: f"chronicle.{field[:-4]}.v3:straddle"})

        assert row_fact_key_epochs(straddling) == frozenset(
            {LEDGER_EPOCH, CHRONICLE_EPOCH}
        ), field

    row_keys = _fact_row()
    row_keys["lineage"] = dict(row_keys["lineage"])
    row_keys["lineage"]["source_row_keys"] = ["chronicle.source_row.v3:row"]

    assert row_fact_key_epochs(row_keys) == frozenset({LEDGER_EPOCH, CHRONICLE_EPOCH})


def test_witnessed_key_fields_match_the_captured_feed() -> None:
    """The inventory is grounded in a real feed, not in a guess.

    Every ``*_key`` field the captured UK feed rows carry is either witnessed
    for its epoch or is a plain source identifier rather than a Chronicle key.
    """
    fixture = Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
    rows = [json.loads(line) for line in fixture.read_text().splitlines() if line]
    assert rows

    for row in rows:
        for field, value in row.items():
            if not field.endswith("_key") or not isinstance(value, str):
                continue
            assert field in FACT_KEY_FIELDS, field
            assert is_chronicle_fact_key(value), (field, value)

    assert feed_fact_key_epochs(rows) == (LEDGER_EPOCH,)
