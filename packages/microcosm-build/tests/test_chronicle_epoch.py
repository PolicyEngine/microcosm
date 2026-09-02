"""Dual-era acceptance of Chronicle (formerly Ledger) identities.

Chronicle introduces chronicle-era hash domains and chronicle-named schema
ids for newly emitted rows at a declared cutover; ledger-era ids stay valid
history forever (PolicyEngine/chronicle#143). Microcosm consumes both, and
these tests are the contract that says so: no validator may reject a row for
being on the other side of the cutover, and no minted Microcosm identity may
move because a source row crossed it.

The other half of the contract is that acceptance follows a *declaration*.
An identity Microcosm has not been told about is reported as undeclared, not
waved through as Chronicle's — a consumer that inferred issued identity from
the shape of a string would accept anything that happened to be spelled like
a Chronicle key.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from microcosm.build.chronicle_epoch import (
    ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS,
    ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS,
    CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION,
    CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
    CHRONICLE_EPOCH,
    CHRONICLE_FACT_KEY_DOMAINS,
    CHRONICLE_NAMESPACES,
    DECLARED_IDENTITIES,
    DECLARED_IDENTITY_EPOCHS,
    EPOCHS,
    FACT_KEY_FIELDS,
    FACT_KEY_IDENTITY_KIND,
    LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
    LEDGER_EPOCH,
    LEDGER_FACT_KEY_DOMAINS,
    MICROCOSM_CONSUMER_ARTIFACT_SCHEMA_VERSION,
    PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION,
    UNDECLARED,
    consumer_artifact_schema_epoch,
    consumer_fact_schema_epoch,
    declared_identity,
    fact_key_epoch,
    fact_key_epoch_label,
    feed_fact_key_epochs,
    feed_undeclared_fact_key_domains,
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

#: Every consumer-artifact manifest id a real producer stamps. Microcosm's
#: own minted v1 and Chronicle's published v2 are both ledger-era; v3 is the
#: chronicle-era successor.
_ARTIFACT_SCHEMA_IDS_BY_EPOCH = (
    (MICROCOSM_CONSUMER_ARTIFACT_SCHEMA_VERSION, LEDGER_EPOCH),
    (PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION, LEDGER_EPOCH),
    (CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION, CHRONICLE_EPOCH),
)


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
    """The same row after Chronicle's cutover: every key domain re-epoched.

    Each successor spelling comes from the declaration table rather than
    being written out here, so a row this fixture builds is one the loader
    recognises as chronicle-era rather than one it reports as undeclared.
    """
    row = _fact_row(
        aggregate_fact_key=_CHRONICLE_AGGREGATE_KEY,
        semantic_fact_key="chronicle.semantic_fact.v3:abc123",
    )
    row["lineage"] = dict(row["lineage"])
    row["lineage"]["source_cell_keys"] = [
        CHRONICLE_FACT_KEY_DOMAINS["source_cell"] + ":cell"
    ]
    row.update(overrides)
    return row


#: The captured Chronicle feed this repo carries. Its rows are ledger-era and
#: frozen: they are the only published Chronicle rows available in-tree.
_CAPTURED_FEED = (
    Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
)


def _captured_feed_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in _CAPTURED_FEED.read_text().splitlines()
        if line.strip()
    ]


def _keys_in(value, _pattern=re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\.v\d+:")):
    """Every Chronicle-shaped key anywhere in a row, at any depth."""
    if isinstance(value, str):
        return [value] if _pattern.match(value) else []
    if isinstance(value, dict):
        return [key for item in value.values() for key in _keys_in(item)]
    if isinstance(value, list):
        return [key for item in value for key in _keys_in(item)]
    return []


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


def test_observed_domain_map_covers_the_captured_feed() -> None:
    """Every domain a real feed carries is one the registry declares.

    ``LEDGER_FACT_KEY_DOMAINS`` is the ledger-era half of the declaration
    table, transcribed from Chronicle's own hash-domain constants. This pins
    it against the captured feed: a domain the repo has actually seen and the
    table has not would make that feed load with an ``undeclared`` witness,
    which is a review event, not a silent pass. The version numbers differ by
    family (``fact`` and ``source_cell`` are v1 where the rest are v2), which
    is the concrete reason a version can never be inferred from a family.
    """
    observed = {
        key.split(":", 1)[0] for row in _captured_feed_rows() for key in _keys_in(row)
    }

    assert observed
    assert observed <= set(LEDGER_FACT_KEY_DOMAINS.values()), sorted(
        observed - set(LEDGER_FACT_KEY_DOMAINS.values())
    )


@pytest.mark.parametrize("domain", sorted(CHRONICLE_FACT_KEY_DOMAINS.values()))
def test_every_declared_chronicle_era_domain_resolves_to_the_chronicle_epoch(
    domain: str,
) -> None:
    assert fact_key_epoch(f"{domain}:digest") == CHRONICLE_EPOCH
    assert fact_key_epoch_label(f"{domain}:digest") == CHRONICLE_EPOCH
    assert is_chronicle_fact_key(f"{domain}:digest")


def test_each_declared_family_pairs_one_ledger_id_with_one_chronicle_id() -> None:
    """The two eras name the same families, one version apart.

    Successor spellings are declared, so this pins the shape of the
    declaration rather than deriving it at call time: the source-side
    families sit at v1/v2 where the derived ones sit at v2/v3, which is
    precisely why a version can never be assumed from a family name.
    """
    assert set(LEDGER_FACT_KEY_DOMAINS) == set(CHRONICLE_FACT_KEY_DOMAINS)
    for family, ledger_domain in LEDGER_FACT_KEY_DOMAINS.items():
        chronicle_domain = CHRONICLE_FACT_KEY_DOMAINS[family]
        ledger_version = int(ledger_domain.rsplit(".v", 1)[1])
        chronicle_version = int(chronicle_domain.rsplit(".v", 1)[1])
        assert chronicle_domain == f"chronicle.{family}.v{chronicle_version}"
        assert chronicle_version == ledger_version + 1, family
    assert LEDGER_FACT_KEY_DOMAINS["source_cell"] == "ledger.source_cell.v1"
    assert CHRONICLE_FACT_KEY_DOMAINS["source_cell"] == "chronicle.source_cell.v2"
    assert LEDGER_FACT_KEY_DOMAINS["aggregate_fact"] == "ledger.aggregate_fact.v2"
    assert CHRONICLE_FACT_KEY_DOMAINS["aggregate_fact"] == (
        "chronicle.aggregate_fact.v3"
    )


@pytest.mark.parametrize(
    "key",
    [
        # source_cell's declared successor is v2; v3 is nobody's spelling.
        "chronicle.source_cell.v3:cell",
        "chronicle.source_cell.v7:cell",
        "chronicle.some_family_nobody_has_declared_yet.v11:digest",
        "chronicle.aggregate_fact.v4:digest",
        # A ledger-era namespace can go undeclared the same way.
        "ledger.aggregate_fact.v9:digest",
        "ledger.invented_family.v1:digest",
        # Declared, but as a *schema* id — never as a fact-key domain.
        "ledger.consumer_fact.v1:digest",
    ],
)
def test_undeclared_chronicle_namespace_keys_are_reported_not_issued(key: str) -> None:
    """An undeclared spelling is undeclared, not Chronicle-issued identity.

    Reading the namespace segment tells you where a key *claims* to come
    from, not that Chronicle issued it. Treating the claim as identity would
    let ``chronicle.anything.vN`` pass as a witnessed Chronicle key, which is
    the opposite of what pinning a fact key is for. The claim is still worth
    surfacing, so it is labelled ``undeclared`` and reported in provenance.
    """
    assert fact_key_epoch(key) is None
    assert not is_chronicle_fact_key(key)
    assert fact_key_epoch_label(key) == UNDECLARED

    identity = parse_fact_key(key)
    assert identity is not None
    assert not identity.declared
    assert identity.namespace_epoch == CHRONICLE_NAMESPACES[identity.namespace]


def test_undeclared_domains_are_named_in_the_feed_report() -> None:
    row = _fact_row(semantic_fact_key="chronicle.semantic_fact.v9:abc123")

    assert row_fact_key_epochs(row) == frozenset({LEDGER_EPOCH, UNDECLARED})
    assert feed_fact_key_epochs([row]) == (LEDGER_EPOCH, UNDECLARED)
    assert feed_undeclared_fact_key_domains([row]) == ("chronicle.semantic_fact.v9",)
    assert feed_undeclared_fact_key_domains([_fact_row()]) == ()


def test_undeclared_domains_reach_the_artifact_provenance(tmp_path) -> None:
    """A release manifest has to name what the build did not recognise."""
    rows = [_fact_row(), _fact_row(source_series_key="chronicle.source_series.v9:s")]
    artifact_dir = _write_artifact_dir(
        tmp_path, rows, schema_version=PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION
    )

    provenance = load_ledger_consumer_artifact(artifact_dir).provenance()

    assert provenance["fact_key_epochs"] == [LEDGER_EPOCH, UNDECLARED]
    assert provenance["undeclared_fact_key_domains"] == ["chronicle.source_series.v9"]


def test_the_registry_is_keyed_by_namespace_family_and_version() -> None:
    """Every accepted identity comes from one reviewable table.

    The registry, not a parser, is what makes an identity Chronicle's. This
    walks it end to end: each declaration round-trips through its
    ``(namespace, family, version)`` key, and the epoch a namespace belongs
    to always agrees with the epoch the declaration claims.
    """
    assert DECLARED_IDENTITIES
    for declaration in DECLARED_IDENTITIES:
        key = (declaration.namespace, declaration.family, declaration.version)
        assert DECLARED_IDENTITY_EPOCHS[key] == declaration.epoch
        assert declared_identity(declaration.identity) is declaration
        assert CHRONICLE_NAMESPACES[declaration.namespace] == declaration.epoch
        assert declaration.epoch in EPOCHS
        if declaration.kind == FACT_KEY_IDENTITY_KIND:
            assert fact_key_epoch(f"{declaration.identity}:digest") == (
                declaration.epoch
            )

    assert declared_identity("arch.consumer_fact.v1") is None
    assert declared_identity("not-an-identity") is None
    assert declared_identity(None) is None


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
    straddling["lineage"]["source_cell_keys"] = [
        CHRONICLE_FACT_KEY_DOMAINS["source_cell"] + ":cell"
    ]
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


def test_schema_id_membership_covers_every_real_producer_id() -> None:
    """Acceptance is keyed to what producers actually stamp, not to a pair.

    Three artifact ids exist, not two. Microcosm's own minted artifacts
    declare ``policyengine_ledger.consumer_artifact.v1``; Chronicle's ``main``
    stamps ``policyengine_ledger.consumer_artifact.v2`` today
    (``policyengine_chronicle/consumer.py``); the chronicle-era successor is
    ``policyengine_chronicle.consumer_artifact.v3``. A loader that accepted
    only the first and the last would reject every artifact Chronicle
    publishes right now.
    """
    assert ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS == {
        MICROCOSM_CONSUMER_ARTIFACT_SCHEMA_VERSION,
        PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION,
        CHRONICLE_CONSUMER_ARTIFACT_SCHEMA_VERSION,
    }
    assert ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS == {
        LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
        CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
    }
    for schema_id, expected_epoch in _ARTIFACT_SCHEMA_IDS_BY_EPOCH:
        assert is_accepted_consumer_artifact_schema_version(schema_id), schema_id
        assert consumer_artifact_schema_epoch(schema_id) == expected_epoch, schema_id
    assert consumer_fact_schema_epoch(LEDGER_CONSUMER_FACT_SCHEMA_VERSION) == (
        LEDGER_EPOCH
    )
    assert consumer_fact_schema_epoch(CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION) == (
        CHRONICLE_EPOCH
    )
    assert not is_accepted_consumer_artifact_schema_version(
        "policyengine_chronicle.other.v9"
    )
    # Version numbers are declared, not inferred: the chronicle-era per-row id
    # is v2, so v1 under the chronicle namespace is nobody's id.
    assert not is_accepted_consumer_fact_schema_version("chronicle.consumer_fact.v1")
    assert consumer_fact_schema_epoch("chronicle.consumer_fact.v1") is None
    # A fact-key domain is a declared identity but not a schema id, and the
    # kinds do not leak into one another.
    assert not is_accepted_consumer_fact_schema_version("ledger.aggregate_fact.v2")
    assert consumer_artifact_schema_epoch(LEDGER_CONSUMER_FACT_SCHEMA_VERSION) is None


def test_chronicle_main_manifest_loads_exactly_as_published(tmp_path) -> None:
    """The manifest chronicle ``main`` writes today, field for field.

    Mirrors ``build_consumer_artifact`` in ``policyengine_chronicle/consumer.py``:
    ``policyengine_ledger.consumer_artifact.v2`` over rows stamped
    ``ledger.consumer_fact.v1``, with ``consumer_fact_schema_versions`` and
    ``consumer_fact_schema_sha256`` beside the hashes. This is the artifact a
    Microcosm build is handed right now, so it is the one that has to load.
    """
    rows = [_fact_row(schema_version=LEDGER_CONSUMER_FACT_SCHEMA_VERSION)]
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    artifact_dir = tmp_path / "chronicle-main"
    artifact_dir.mkdir()
    (artifact_dir / "consumer_facts.jsonl").write_text(payload)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION,
                "consumer_fact_schema_versions": [LEDGER_CONSUMER_FACT_SCHEMA_VERSION],
                "consumer_fact_schema_sha256": "cd" * 32,
                "fact_row_count": len(rows),
                "facts_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    artifact = load_ledger_consumer_artifact(artifact_dir)

    assert artifact.schema_version == PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION
    assert artifact.schema_epoch == LEDGER_EPOCH
    assert artifact.fact_schema_versions == (LEDGER_CONSUMER_FACT_SCHEMA_VERSION,)
    assert artifact.fact_key_epochs == (LEDGER_EPOCH,)
    assert artifact.undeclared_fact_key_domains == ()


@pytest.mark.parametrize(
    ("schema_version", "expected_epoch"), _ARTIFACT_SCHEMA_IDS_BY_EPOCH
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


def test_unknown_schema_id_is_rejected_naming_every_accepted_id(tmp_path) -> None:
    artifact_dir = _write_artifact_dir(
        tmp_path, [_fact_row()], schema_version="policyengine_chronicle.other.v9"
    )

    with pytest.raises(ValueError) as excinfo:
        load_ledger_consumer_artifact(artifact_dir)

    message = str(excinfo.value)
    assert "schema_version" in message
    for schema_id, _epoch in _ARTIFACT_SCHEMA_IDS_BY_EPOCH:
        assert schema_id in message, schema_id


@pytest.mark.parametrize(
    "schema_version",
    [
        ["policyengine_ledger.consumer_artifact.v1"],
        {"id": "policyengine_ledger.consumer_artifact.v1"},
        {},
        [],
        1,
        True,
        None,
    ],
)
def test_malformed_schema_version_types_raise_the_documented_error(
    tmp_path, schema_version
) -> None:
    """A JSON list or object where a schema id belongs is *unsupported*.

    ``schema_version`` arrives from an untrusted manifest, so its type is not
    guaranteed. A membership test against a frozenset raises ``TypeError`` on
    an unhashable value, which would escape the loader as a bare
    ``unhashable type: 'list'`` instead of the message that names what the
    loader accepts. The predicate is total over JSON so the loader's own
    error is the one an operator sees.
    """
    artifact_dir = _write_artifact_dir(
        tmp_path, [_fact_row()], schema_version=schema_version
    )

    assert not is_accepted_consumer_artifact_schema_version(schema_version)
    assert not is_accepted_consumer_fact_schema_version(schema_version)
    assert consumer_artifact_schema_epoch(schema_version) is None
    assert consumer_fact_schema_epoch(schema_version) is None

    with pytest.raises(ValueError) as excinfo:
        load_ledger_consumer_artifact(artifact_dir)

    assert "Unsupported Chronicle consumer artifact schema_version" in str(
        excinfo.value
    )


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
    row_keys["lineage"]["source_row_keys"] = [
        CHRONICLE_FACT_KEY_DOMAINS["source_row"] + ":row"
    ]

    assert row_fact_key_epochs(row_keys) == frozenset({LEDGER_EPOCH, CHRONICLE_EPOCH})


def test_witnessed_key_fields_match_the_captured_feed() -> None:
    """The inventory is grounded in a real feed, not in a guess.

    Every ``*_key`` field the captured UK feed rows carry is either witnessed
    for its epoch or is a plain source identifier rather than a Chronicle key.
    """
    rows = _captured_feed_rows()
    assert rows

    for row in rows:
        for field, value in row.items():
            if not field.endswith("_key") or not isinstance(value, str):
                continue
            assert field in FACT_KEY_FIELDS, field
            assert is_chronicle_fact_key(value), (field, value)

    assert feed_fact_key_epochs(rows) == (LEDGER_EPOCH,)
