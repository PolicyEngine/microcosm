from __future__ import annotations

import json
from importlib import resources as importlib_resources
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "uk_bus05i_fy2025.json"
BUS_PREFIX = "dft.bus_"


def _resource(name: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk").joinpath(name).read_text()
    )


def test_bus05i_fy2025_subareas_reconcile_to_england() -> None:
    fixture = json.loads(FIXTURE.read_text())
    rows = fixture["rows"]

    assert len(rows) == 6
    assert {row["period"]["type"] for row in rows} == {"fiscal_year"}
    assert {row["period"]["value"] for row in rows} == {2025}
    assert {row["unit"] for row in rows} == {"gbp"}

    for concept in {
        "dft.local_bus_passenger_fare_receipts",
        "dft.local_bus_total_estimated_net_support",
    }:
        values = {
            row["groupby_value_id"]: row["value"]
            for row in rows
            if row["source_concept"] == concept
        }
        assert values["london"] + values["england_outside_london"] == pytest.approx(
            values["england"], abs=1_000
        )


def test_bus_targets_are_active_or_signed_excluded_as_declared() -> None:
    contract = _resource("uk_population_targets.json")
    references = {
        row["name"]: row
        for row in _resource("target_references.json")["target_references"]
    }
    membership = _resource("target_reference_membership.json")
    exclusions = {
        row["target_id"]: row
        for row in _resource("target_reference_signed_exclusions.json")["exclusions"]
    }
    bus = {
        row["target_id"]: row
        for row in contract["targets"]
        if row["target_id"].startswith(BUS_PREFIX)
    }

    active = {
        "dft.bus_fare_receipts.england",
        "dft.bus_net_support.england",
        "dft.bus_fare_receipts.london",
        "dft.bus_net_support.london",
    }
    excluded = set(bus) - active
    assert len(bus) == 8
    assert active <= set(references)
    assert excluded.isdisjoint(references)
    assert set(exclusions) >= excluded
    assert {exclusions[target_id]["reason_id"] for target_id in excluded} == {
        "derived_partition_member",
        "no_publisher_uk_total",
        "replaced_by_heterogeneous_country_legs",
    }
    assert all(
        membership["targets"][target_id]["status"] == "signed_excluded"
        for target_id in excluded
    )
    # The BUS0415 alignment is a declaration on the fare-receipt references
    # (contract uprating_index), applied by the generator so the committed
    # membership records the aligned value with its factor; net support
    # declares no index and holds unchanged.
    holds = {row["name"]: row for row in membership["uprating_holds"]}
    for target_id in ("dft.bus_fare_receipts.england", "dft.bus_fare_receipts.london"):
        assert references[target_id]["uprating_index"] == "dft.local_bus_fares_index"
        assert holds[target_id]["index"] == "dft.local_bus_fares_index"
        (candidate,) = membership["targets"][target_id]["candidates"]
        uprating = candidate["uprating"]
        assert uprating["index"] == "dft.local_bus_fares_index"
        assert candidate["resolved_value"] == pytest.approx(
            float(uprating["value_before_uprating"]) * float(uprating["factor"])
        )
    for target_id in ("dft.bus_net_support.england", "dft.bus_net_support.london"):
        assert "uprating_index" not in references[target_id]
        assert set(holds[target_id]) == {"name", "from", "to"}
        (candidate,) = membership["targets"][target_id]["candidates"]
        assert "uprating" not in candidate
    assert float(holds["dft.bus_fare_receipts.london"]["factor"]) == pytest.approx(1.0)
    assert float(holds["dft.bus_fare_receipts.england"]["factor"]) == pytest.approx(
        204.125 / 193.125
    )
    # The London rows pin the publisher's region-stamped fact, not a country
    # row, and the England rows keep the country pin.
    for target_id in ("dft.bus_fare_receipts.london", "dft.bus_net_support.london"):
        selector = references[target_id]["ledger_selector"]
        assert selector["geography_level"] == "region"
        assert selector["geography_id"] == "E12000007"
        assert membership["targets"][target_id]["status"] == "active"
    for target_id in ("dft.bus_fare_receipts.england", "dft.bus_net_support.england"):
        selector = references[target_id]["ledger_selector"]
        assert selector["geography_level"] == "country"
        assert selector["geography_id"] == "E92000001"


def test_london_bus_rows_resolve_the_publisher_partition() -> None:
    fixture = json.loads(FIXTURE.read_text())
    published = {
        (row["source_concept"], row["groupby_value_id"]): row["value"]
        for row in fixture["rows"]
    }
    membership = _resource("target_reference_membership.json")["targets"]
    for target_id, concept in (
        ("dft.bus_fare_receipts.london", "dft.local_bus_passenger_fare_receipts"),
        ("dft.bus_net_support.london", "dft.local_bus_total_estimated_net_support"),
    ):
        resolved = membership[target_id]["candidates"][0]["resolved_value"]
        assert resolved == pytest.approx(published[(concept, "london")], abs=1.0)


def test_bus_selectors_do_not_consume_nts_or_bus0415() -> None:
    targets = [
        row
        for row in _resource("uk_population_targets.json")["targets"]
        if row["target_id"].startswith(BUS_PREFIX)
    ]
    selected_concepts = {
        row["ledger_selector"].get("source_concept", "") for row in targets
    }

    assert selected_concepts == {
        "dft.local_bus_passenger_fare_receipts",
        "dft.local_bus_total_estimated_net_support",
    }
    england_notes = " ".join(
        row["bindings"]["policyengine"]["notes"]
        for row in targets
        if row["target_id"].endswith("england")
    )
    assert "NTS0705a" in england_notes
    assert "BUS0415" in england_notes
    assert "#890" in england_notes
    assert "#790" not in england_notes


def test_generator_pins_region_only_targets_at_region_level() -> None:
    from tools.generate_uk_target_references import _geography_pins

    pins = _geography_pins(_resource("uk_population_targets.json"))
    assert pins["dft.bus_fare_receipts.london"] == {
        "geography_level": "region",
        "geography_id": "E12000007",
    }
    assert pins["dft.bus_fare_receipts.england"] == {
        "geography_level": "country",
        "geography_id": "E92000001",
    }
    assert pins["dft.bus_fare_receipts.england_outside_london"]["geography_level"] == (
        "country"
    )


def test_english_region_roster_has_catalog_labels() -> None:
    """Every region a national reference may pin resolves a hierarchy label."""

    from microcosm.build.uk_runtime.ledger_targets import UK_NATIONAL_REGION_ROSTER
    from microcosm.calibrate.geography_constants import UK_GEOGRAPHY_ID_TO_LABEL

    assert UK_NATIONAL_REGION_ROSTER <= set(UK_GEOGRAPHY_ID_TO_LABEL)
    assert UK_GEOGRAPHY_ID_TO_LABEL["E12000007"] == "London"


def test_compiler_refuses_region_pins_outside_the_english_region_roster() -> None:
    from microcosm.build.ledger_targets import LedgerTargetReference
    from microcosm.build.uk_runtime.ledger_targets import (
        UK_NATIONAL_REGION_ROSTER,
        _assert_national_region_pin,
    )

    assert UK_NATIONAL_REGION_ROSTER == frozenset(
        f"E1200000{index}" for index in range(1, 10)
    )
    good = LedgerTargetReference(
        name="probe",
        ledger_selector={"geography_level": "region", "geography_id": "E12000007"},
        entity="household",
        measure="probe",
        family="dft_local_bus",
        period=2025,
    )
    _assert_national_region_pin(good)
    bad = LedgerTargetReference(
        name="probe",
        ledger_selector={"geography_level": "region", "geography_id": "W99999999"},
        entity="household",
        measure="probe",
        family="dft_local_bus",
        period=2025,
    )
    with pytest.raises(ValueError, match="not in the English region roster"):
        _assert_national_region_pin(bad)


def _fare_fact(
    concept: str, geography: dict, period: dict, value: float, **extra
) -> dict:
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{concept}:{geography['id']}:{period['value']}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{concept}:{geography['id']}:{period['value']}",
        "assertion": "observation",
        "aggregation": {"method": "sum"},
        "dimensions": {},
        "entity": {
            "name": "institutional_sector",
            "role": "local_bus_service_operators",
        },
        # Consumer facts carry a display label; the schema-8 hierarchy
        # completion refuses a fact without one.
        "label": f"{geography['id']} {period['value']} {concept}",
        "geography": geography,
        "layout": {"groupby_value_id": extra.get("groupby", "england")},
        "lineage": {
            "source_record_id": f"probe.{concept}.{geography['id']}.{period['value']}"
        },
        "observed_measure": {
            "source_concept": concept,
            "source_name": "dft",
            "source_measure_id": concept.rsplit(".", 1)[-1],
            "unit": extra.get("unit", "gbp"),
        },
        "period": period,
        "period_coverage": extra.get("coverage", {}),
        "source": {"source_name": "dft", "source_sha256": "0" * 64},
        "value": value,
    }


def _index_facts(geography: dict, values: dict[str, float]) -> list[dict]:
    return [
        _fare_fact(
            "dft.local_bus_fares_index",
            geography,
            {"type": "month", "value": month},
            value,
            unit="index_2005_100",
        )
        for month, value in values.items()
    ]


def test_quarter_end_months_follow_the_fiscal_and_calendar_windows() -> None:
    from microcosm.build.uk_runtime.ledger_targets import _quarter_end_months

    assert _quarter_end_months(2024, 4) == ("2024-06", "2024-09", "2024-12", "2025-03")
    assert _quarter_end_months(2025, 1) == ("2025-03", "2025-06", "2025-09", "2025-12")


def _vendored_index_rows(geography: dict, values: dict[str, float]) -> list[dict]:
    """Rows in the vendored-resource shape ``tools/vendor_uk_ledger_facts.py`` writes."""

    return [
        {
            "concept": "dft.local_bus_fares_index",
            "geography": geography,
            "period": {"type": "month", "value": month},
            "value": value,
            "source_record_id": (
                f"probe.dft.local_bus_fares_index.{geography['id']}.{month}"
            ),
        }
        for month, value in values.items()
    ]


def test_fare_receipts_align_with_the_bus0415_series_and_support_does_not() -> None:
    from microcosm.build.ledger_targets import (
        LedgerTargetReference,
        compile_ledger_target_references,
    )
    from microcosm.build.uk_runtime.ledger_targets import (
        align_dft_bus_fare_receipts_to_period,
        apply_declared_uk_uprating,
    )

    england = {"id": "E92000001", "level": "country", "vintage": "current"}
    coverage = {"start_date": "2024-04-01", "end_date": "2025-03-31", "basis": "fiscal"}
    fare = _fare_fact(
        "dft.local_bus_passenger_fare_receipts",
        england,
        {"type": "fiscal_year", "value": 2025},
        1_000.0,
        coverage=coverage,
    )
    support = _fare_fact(
        "dft.local_bus_total_estimated_net_support",
        england,
        {"type": "fiscal_year", "value": 2025},
        500.0,
        coverage=coverage,
    )
    index = _vendored_index_rows(
        england,
        {
            "2024-03": 100.0,
            "2024-06": 100.0,
            "2024-09": 100.0,
            "2024-12": 100.0,
            "2025-03": 120.0,
            "2025-06": 120.0,
            "2025-09": 120.0,
            "2025-12": 120.0,
        },
    )

    def reference(name: str, concept: str, **extra: object) -> LedgerTargetReference:
        return LedgerTargetReference(
            name=name,
            ledger_selector={
                "source_name": "dft",
                "source_concept": concept,
                "period_type": "fiscal_year",
                "geography_level": "country",
                "geography_id": "E92000001",
            },
            entity="household",
            measure=name,
            family="dft_local_bus",
            period=2025,
            uprating_from_period="2024",
            uprating_to_period=2025,
            **extra,
        )

    fare_reference = reference(
        "fares",
        "dft.local_bus_passenger_fare_receipts",
        uprating_index="dft.local_bus_fares_index",
    )
    support_reference = reference(
        "support", "dft.local_bus_total_estimated_net_support"
    )

    fares = compile_ledger_target_references([fare], [fare_reference], country="uk")
    # The generic compiler records the declaration and leaves the value alone;
    # the UK applier transports it and declares the factor.
    assert fares.specs[0].metadata["uprating_index"] == "dft.local_bus_fares_index"
    assert fares.specs[0].value == 1_000.0
    aligned = align_dft_bus_fare_receipts_to_period(
        fare_reference, fares, index_rows=index
    )
    (spec,) = aligned.specs
    # from-window mean = (100+100+100+120)/4 = 105; to-window mean = 120
    assert spec.value == pytest.approx(1_000.0 * 120.0 / 105.0)
    assert spec.metadata["uprating_index"] == "dft.local_bus_fares_index"
    assert spec.metadata["uprating_index_resource"] == "dft_bus_value_anchors.json"
    assert (
        spec.metadata["uprating_index_from_months"] == "2024-06,2024-09,2024-12,2025-03"
    )
    assert (
        spec.metadata["uprating_index_to_months"] == "2025-03,2025-06,2025-09,2025-12"
    )
    assert spec.metadata["ledger_value_before_alignment"] == "1000"
    assert spec.metadata["uprating_factor"] == f"{120.0 / 105.0:.15g}"
    assert (
        "probe.dft.local_bus_fares_index.E92000001.2025-12"
        in (spec.metadata["uprating_index_source_record_ids"])
    )

    supports = compile_ledger_target_references(
        [support], [support_reference], country="uk"
    )
    assert "uprating_index" not in supports.specs[0].metadata
    untouched = align_dft_bus_fare_receipts_to_period(
        support_reference, supports, index_rows=index
    )
    assert untouched.specs[0].value == 500.0
    assert "uprating_factor" not in untouched.specs[0].metadata
    assert apply_declared_uk_uprating(support_reference, supports) is supports

    with pytest.raises(ValueError, match="lacks quarter-end month"):
        align_dft_bus_fare_receipts_to_period(
            fare_reference, fares, index_rows=index[:-1]
        )
    # The index transports fare receipts only; declaring it on another
    # concept is a contract error, as is an index no UK applier implements.
    misdeclared = reference(
        "support",
        "dft.local_bus_total_estimated_net_support",
        uprating_index="dft.local_bus_fares_index",
    )
    with pytest.raises(ValueError, match="transports"):
        align_dft_bus_fare_receipts_to_period(misdeclared, supports, index_rows=index)
    unknown = reference(
        "fares", "dft.local_bus_passenger_fare_receipts", uprating_index="cpi"
    )
    with pytest.raises(ValueError, match="no UK applier"):
        apply_declared_uk_uprating(unknown, fares)


def test_bus0415_alignment_from_the_vendored_series() -> None:
    """The production factors reproduce without the licensed feed.

    Both sides come from the vendored resource: the BUS05ai receipts rows and
    the BUS0415 index series, hash-pinned to the same Chronicle feed as the
    references, so this check runs in CI.
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.ledger_targets import compile_ledger_target_references
    from microcosm.build.uk_runtime.ledger_fact_vendoring import (
        load_vendored_resource,
        rows_matching,
    )
    from microcosm.build.uk_runtime.ledger_targets import apply_declared_uk_uprating

    payload = load_vendored_resource("dft_bus_value_anchors.json")
    references = {
        reference.name: reference
        for reference in load_country_spec("uk").target_references
        if reference.family == "dft_local_bus"
    }
    membership = _resource("target_reference_membership.json")
    expected_factor = {"E92000001": 204.125 / 193.125, "E12000007": 1.0}
    for name, geography_id, groupby in (
        ("dft.bus_fare_receipts.england", "E92000001", "england"),
        ("dft.bus_fare_receipts.london", "E12000007", "london"),
    ):
        (row,) = rows_matching(
            payload,
            concept="dft.local_bus_passenger_fare_receipts",
            geography_id=geography_id,
            period_value=2025,
        )
        # The consumer fact carries no geography name for a region, so the
        # hierarchy completion labels it from Microcosm's geography catalog.
        fact = _fare_fact(
            "dft.local_bus_passenger_fare_receipts",
            dict(row["geography"]),
            dict(row["period"]),
            row["value"],
            coverage=dict(row["period_coverage"]),
            groupby=groupby,
        )
        reference = references[name]
        registry = compile_ledger_target_references([fact], [reference], country="uk")
        aligned = apply_declared_uk_uprating(reference, registry)
        (spec,) = aligned.specs
        assert float(spec.metadata["uprating_factor"]) == pytest.approx(
            expected_factor[geography_id]
        )
        assert spec.value == pytest.approx(row["value"] * expected_factor[geography_id])
        # One value on the surface: the committed membership records the same.
        (candidate,) = membership["targets"][name]["candidates"]
        assert candidate["resolved_value"] == pytest.approx(spec.value)
    for name in ("dft.bus_net_support.england", "dft.bus_net_support.london"):
        assert references[name].uprating_index is None


def test_bus0415_alignment_on_the_pinned_feed() -> None:
    root = Path(__file__).resolve().parents[3]
    feed = root / ".codex-work/uk-artifact"
    if not feed.is_dir():
        pytest.skip("pinned UK Chronicle national artifact directory is not present")
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.uk_runtime.chronicle_feed import (
        load_uk_chronicle_feed,
    )
    from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry

    pin = load_uk_chronicle_feed()
    artifact = load_ledger_consumer_artifact(
        feed,
        expected_facts_sha256=pin.facts_sha256,
        expected_manifest_sha256=pin.manifest_sha256,
    )
    compiled = compile_uk_target_registry(artifact.facts, target_period=2025)
    specs = {spec.name: spec for spec in compiled.registry.specs}
    # Until chronicle#261 ships dimension labels on the consumer artifact,
    # the schema-8 hierarchy completion refuses every UK reference against
    # the pinned feed (0 compiled / 424 unsupported, on main as here).
    assert "dft.bus_fare_receipts.england" in specs, (
        "runtime compile produced no fare rows: "
        + "; ".join(row["reason"] for row in compiled.unsupported[:1])
    )
    england = specs["dft.bus_fare_receipts.england"]
    london = specs["dft.bus_fare_receipts.london"]
    assert float(england.metadata["uprating_factor"]) == pytest.approx(
        204.125 / 193.125
    )
    assert england.value == pytest.approx(3_417_388_656.43538 * 204.125 / 193.125)
    assert float(london.metadata["uprating_factor"]) == pytest.approx(1.0)
    assert london.value == pytest.approx(1_347_434_943.01459)
    assert "uprating_factor" not in specs["dft.bus_net_support.england"].metadata
    assert "uprating_factor" not in specs["dft.bus_net_support.london"].metadata
