"""Guarantees for the UK active-subset Ledger target references.

``uk/target_references.json`` is the typed activation surface derived from the
value-free UK national contract. Observed values remain in Ledger facts; this
resource only declares which 2023 country-level facts Microcosm activates.
"""

from __future__ import annotations

import json
from importlib import resources as importlib_resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.target_reference_authoring import (
    TargetReferenceAuthoringConfig,
    author_target_references,
)
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from tools.generate_uk_target_references import (
    POLICYENGINE_BINDING_KEYS,
    _annual_uc_award_band_token,
    _geography_pins,
    _value_operation_by_target_id,
)

ACTIVE_REFERENCE_COUNT = 387
UK_DATA_REPO = "policyengine-" + "uk-data"

FIXTURE_REFERENCE_NAMES = {
    "obr.income_tax",
    "dwp.uc.two_child_limit.households_affected",
    "ons.population.uk_total",
    "hmrc/employment_income_income_band_12_570_to_15_000",
    "slc.repayments.england_plan_2",
    "obr.esa",
    "hmrc.cgt.gains_total",
    "hmrc.cgt.taxpayers_total",
}

FIXTURE_FEED_ROWS = (
    Path(__file__).parent / "fixtures" / "uk_target_reference_feed_rows.jsonl"
)


def _load_uk_resource(name: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk").joinpath(name).read_text()
    )


def _contract_targets_by_id() -> dict[str, dict]:
    contract = _load_uk_resource("uk_national_targets.json")
    return {target["target_id"]: target for target in contract["targets"]}


def _expected_reference_entity(target: dict) -> str:
    binding = target["bindings"]["policyengine"]
    return binding.get("from_entity") or binding.get("map_to") or "household"


def _uc_benefit_units_fact(period: str, *, value: float) -> dict:
    normalized_period = period.replace("-", "_")
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:uc-{normalized_period}",
        "aggregation": {"method": "sum"},
        "assertion": "observation",
        "geography": {"level": "country", "id": "K03000001"},
        "observed_measure": {
            "source_name": "dwp",
            "source_concept": "dwp.uc_benefit_units",
            "source_measure_id": "total_units",
            "unit": "count",
        },
        "period": {"type": "month", "value": period},
        "value": value,
    }


def test_uk_target_references_load_as_typed_non_empty_resource() -> None:
    spec = load_country_spec("uk")

    assert len(spec.target_references) == ACTIVE_REFERENCE_COUNT
    assert {reference.name for reference in spec.target_references}


def test_uk_target_references_follow_contract_derivation_rules() -> None:
    resource = _load_uk_resource("target_references.json")
    targets_by_id = _contract_targets_by_id()
    names = [reference["name"] for reference in resource["target_references"]]

    assert resource["country"] == "uk"
    assert resource["allowed_value_operations"] == [
        "identity",
        "sum",
        "calendar_year_average",
        "latest_plateau",
    ]
    assert len(names) == len(set(names))

    for reference in resource["target_references"]:
        contract_target_id = reference["metadata"]["contract_target_id"]
        target = targets_by_id[contract_target_id]
        binding = target["bindings"]["policyengine"]

        for key, value in target["ledger_selector"].items():
            if key == "dimension_values":
                assert value.items() <= reference["ledger_selector"][key].items()
                continue
            assert reference["ledger_selector"][key] == value
        assert reference["ledger_selector"]["geography_level"] == "country"
        assert reference["ledger_selector"]["geography_id"]
        assert reference["entity"] == _expected_reference_entity(target)
        expected_measure = (
            binding["metric_name"]
            if reference["name"] == contract_target_id
            else reference["name"]
        )
        assert reference["measure"] == expected_measure
        assert reference["family"] == target["family"]
        assert reference["period"] == 2025
        assert reference["metadata"] == {
            "contract_target_id": contract_target_id,
            "measure_kind": "prepared_column",
        }
        # The measure is a prepared column, so the pointed-to contract binding
        # must carry what the microcosm#622 materializer needs to prepare it.
        assert (
            binding.get("value_variable")
            or binding.get("value_expression")
            or binding.get("kind")
        ), contract_target_id


def test_uk_target_references_do_not_bind_known_mismatched_property_amounts() -> None:
    resource = _load_uk_resource("target_references.json")

    assert not [
        reference["name"]
        for reference in resource["target_references"]
        if reference["metadata"]["contract_target_id"]
        == "hmrc.spi.property_income.amount_by_total_income_band"
    ]


def test_uk_target_references_do_not_emit_nan_uc_payment_bands() -> None:
    resource = _load_uk_resource("target_references.json")

    assert not [
        reference["name"]
        for reference in resource["target_references"]
        if "uc_payment_dist" in reference["name"] and "nan_to_nan" in reference["name"]
    ]


def test_uc_payment_fanout_excludes_source_only_unbounded_rows() -> None:
    assert _annual_uc_award_band_token("No payment") == ""
    assert _annual_uc_award_band_token("£1,500.01 or over") == ""
    assert _annual_uc_award_band_token("£2,500.01 or over") == ""


def test_ons_age_total_targets_pin_exact_age_dimension_set() -> None:
    references = {
        reference["name"]: reference
        for reference in _load_uk_resource("target_references.json")[
            "target_references"
        ]
    }

    assert references["ons.population.scotland_babies_under_1"][
        "ledger_selector"
    ]["dimensions"] == ["age"]
    assert references["ons.population.scotland_children_under_16"][
        "ledger_selector"
    ]["dimensions"] == ["age"]


def test_uk_target_reference_membership_report_is_packaged() -> None:
    membership = _load_uk_resource("target_reference_membership.json")

    assert membership["target_period"] == 2025
    assert membership["active_reference_count"] == ACTIVE_REFERENCE_COUNT
    assert membership["status_counts"] == {
        "active": 387,
        "multi_fact": 1,
        "no_fact_at_or_before_period": 28,
        "signed_excluded": 1,
    }
    assert membership["genuine_sum_residue"]
    assert membership["uprating_holds"]
    assert membership["fanout_family_outcomes"] == [
        {
            "family": "hmrc_spi",
            "status": "active_with_signed_property_amount_exclusion",
            "active_reference_count": 143,
            "signed_rationale": (
                "SPI income-band targets fan out by strict total-income-band "
                "dimension pins, except the HMRC property-income amount "
                "surface. Those 13 rows are signed out because Ledger carries "
                "the official SPI Table 3.7 net property-income amounts, "
                "while the incumbent target applies the populace-side x1.9 "
                "property-income undercount adjustment traced to "
                f"{UK_DATA_REPO} PR #311 / issue #230 and HMRC Property Rental "
                "Income Statistics."
            ),
        },
        {
            "family": "dwp_universal_credit",
            "status": "active_with_unmapped_vintage_residue_skipped",
            "active_reference_count": 100,
            "skipped_unmapped_fact_count": 12,
            "signed_rationale": (
                "UC payment-distribution targets fan out over family_type and "
                "monthly_award_amount_bands into incumbent-compatible "
                "annual-payment rows. The four source-only 'No payment' facts "
                "and eight overlapping source-only 'or over' facts are left "
                "out; no active reference may use the legacy nan_to_nan band "
                "name."
            ),
        },
        {
            "family": "council_tax_stock",
            "status": "active_declared_rows",
            "active_reference_count": 9,
            "signed_rationale": (
                "VOA council-tax stock bands are declared as nine explicit "
                "target rows, including total, and each resolves with its "
                "country-level geography and band pin."
            ),
        },
    ]
    assert membership["signed_exclusion_rationales"] == [
        {
            "family": "hmrc_spi",
            "target_id": "hmrc.spi.property_income.amount_by_total_income_band",
            "status": "signed_excluded",
            "signed_rationale": (
                "Signed out pending a first-class value-scaling operation or "
                "a Chronicle package for HMRC Property Rental Income "
                "Statistics with declared reconciliation: the Ledger facts "
                "are official HMRC SPI Table 3.7 net property-income amounts, "
                "while the incumbent calibration target applies the "
                "populace-side x1.9 property-income undercount adjustment. "
                f"The x1.9 trace is {UK_DATA_REPO} PR #311 / issue "
                f"{UK_DATA_REPO}#230: SPI covers only taxpayers with liability, "
                "and HMRC Property Rental Income Statistics show GBP 46.68bn "
                "versus SPI about GBP 24.5bn for 2020-21. Binding the raw SPI "
                "facts would knowingly calibrate to 10/19 of the incumbent "
                "surface."
            ),
        }
    ]
    assert membership["multi_fact_rationales"] == [
        {
            "family": "ons_population",
            "target_id": "ons.population.scotland_households_3plus_children",
            "candidate_name": "ons/scotland_households_3plus_children",
            "status": "adjudication_pending",
            "signed_rationale": (
                "Remaining multi_fact is genuine: the selector reaches ONS "
                "mid-year population age rows for Scotland across six eligible "
                "periods, while the contract target is a household count with "
                "three or more children. No Ledger household-composition fact "
                "at or before 2025 is selected by the current contract, so "
                "Microcosm must not adjudicate a replacement source here."
            ),
        }
    ]


def test_uk_fixture_b_signed_differences_carry_ruled_rationales() -> None:
    differences = {
        row["name"]: row
        for row in _load_uk_resource(
            "ledger_compile_parity_incumbent_2025_signed_differences.json"
        )["differences"]
    }

    assert "2023-24 outturn" in differences["hmrc.cgt.gains_total"]["reason"]
    assert "forecast/uprated value" in differences["hmrc.cgt.gains_total"]["reason"]
    assert "single-age-90 share" in differences[
        "ons.population.female_85_89"
    ]["reason"]
    assert "single-age-90 share" in differences[
        "ons.population.male_85_89"
    ]["reason"]
    assert "ages 91+ unconstrained" in differences[
        "ons.population.female_90_plus"
    ]["reason"]
    assert "ages 91+ unconstrained" in differences[
        "ons.population.male_90_plus"
    ]["reason"]


def test_uk_target_references_compile_from_real_staged_feed_rows() -> None:
    spec = load_country_spec("uk")
    references = [
        reference
        for reference in spec.target_references
        if reference.name in FIXTURE_REFERENCE_NAMES
    ]

    registry = compile_ledger_target_references(
        [
            json.loads(line)
            for line in FIXTURE_FEED_ROWS.read_text().splitlines()
            if line.strip()
        ],
        references,
        country="uk",
    )

    targets = {target.name: target for target in registry.specs}
    assert set(targets) == FIXTURE_REFERENCE_NAMES

    income_tax = targets["obr.income_tax"]
    assert income_tax.value == pytest.approx(331_437_583_074.4429)
    assert income_tax.period == 2025
    assert income_tax.metadata["ledger_assertion"] == "source_projection"
    assert income_tax.metadata["ledger_assertion_policy"] == ("allow_source_projection")

    tcl_households = targets["dwp.uc.two_child_limit.households_affected"]
    assert tcl_households.value == 469_780
    assert tcl_households.metadata["ledger_fact_period"] == "2025-04"

    population = targets["ons.population.uk_total"]
    assert population.value == 69_281_437
    assert population.metadata["ledger_value_operation"] == "sum"
    assert population.metadata["uprating_from_period"] == "2024"

    spi_band = targets["hmrc/employment_income_income_band_12_570_to_15_000"]
    assert spi_band.value == 16_900_000_000
    assert (
        spi_band.metadata["contract_target_id"]
        == "hmrc.spi.employment_income.amount_by_total_income_band"
    )

    slc_plan_2 = targets["slc.repayments.england_plan_2"]
    assert slc_plan_2.value == pytest.approx(2_778_253_361.64)
    assert slc_plan_2.metadata["ledger_member_fact_count"] == "2"

    assert targets["hmrc.cgt.gains_total"].value == 65_937_000_000
    assert targets["hmrc.cgt.taxpayers_total"].value == 378_000
    assert targets["hmrc.cgt.gains_total"].metadata["ledger_fact_period"] == "2023"


def test_uk_generator_assigns_calendar_average_to_uc_benefit_unit_targets() -> None:
    contract = _load_uk_resource("uk_national_targets.json")
    facts = [
        _uc_benefit_units_fact("2025-04", value=6_380_000.0),
        *(
            _uc_benefit_units_fact(f"2025-{month:02d}", value=6_600_000.0)
            for month in range(5, 9)
        ),
        *(
            _uc_benefit_units_fact(f"2025-{month:02d}", value=6_960_000.0)
            for month in range(9, 12)
        ),
        _uc_benefit_units_fact("2025-12", value=7_170_000.0),
    ]
    authored = author_target_references(
        contract,
        facts,
        TargetReferenceAuthoringConfig(
            target_period=2025,
            geography_pins=_geography_pins(contract),
            value_operation_by_target_id=_value_operation_by_target_id(contract),
            binding_vocabulary=POLICYENGINE_BINDING_KEYS,
            source_fact_feed="synthetic-uc-benefit-units",
        ),
    )
    references = {reference["name"]: reference for reference in authored.references}

    uc_reference = references["dwp.uc.households"]
    assert uc_reference["value_operation"] == "calendar_year_average"
    assert authored.membership_report["targets"]["dwp.uc.households"]["status"] == (
        "active"
    )

    registry = compile_ledger_target_references(
        facts,
        [LedgerTargetReference(**uc_reference)],
        country="uk",
    )
    assert registry.specs[0].value == pytest.approx(
        (6_380_000.0 + 4 * 6_600_000.0 + 3 * 6_960_000.0 + 7_170_000.0) / 9
    )


def test_uk_target_references_constrain_a_frame_with_prepared_columns() -> None:
    """The full active subset compiles into constraint rows, end to end.

    Measures name prepared columns (the US prepared-indicator-column doctrine):
    the microcosm#622 materializer owns building them from the contract
    bindings, so this test hand-prepares one column per measure on the
    reference's entity table and proves the compiled registry constrains a
    household-weighted frame with zero skipped targets and exact row
    aggregates.
    """

    spec = load_country_spec("uk")
    references = [
        reference
        for reference in spec.target_references
        if reference.name in FIXTURE_REFERENCE_NAMES
    ]
    feed_rows = [
        json.loads(line)
        for line in FIXTURE_FEED_ROWS.read_text().splitlines()
        if line.strip()
    ]

    registry = compile_ledger_target_references(feed_rows, references, country="uk")
    assert len(registry.specs) == len(FIXTURE_REFERENCE_NAMES)

    n_households = 3
    weights = np.array([10.0, 20.0, 30.0])
    household_columns: dict[str, np.ndarray] = {}
    person_columns: dict[str, np.ndarray] = {}
    expected_aggregates: dict[str, float] = {}
    for index, compiled in enumerate(registry.specs):
        column = np.array([index + 1.0, 2.0 * (index + 1.0), 0.0])
        columns = person_columns if compiled.entity == "person" else household_columns
        columns[compiled.measure] = column
        expected_aggregates[f"{compiled.name}@{compiled.period}"] = float(
            (column * weights).sum()
        )

    household_ids = np.arange(n_households, dtype="int64")
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": household_ids,
                    "person_household_id": household_ids,
                    **person_columns,
                }
            ),
            "household": pd.DataFrame(
                {"household_id": household_ids, **household_columns}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=weights, kind=WeightKind.DESIGN)},
    )

    problem = build_constraint_matrix(frame, registry.to_target_set())

    assert problem.skipped == ()
    assert len(problem.names) == len(FIXTURE_REFERENCE_NAMES)
    achieved = problem.matrix @ problem.initial_weights.values
    for name, estimate in zip(problem.names, achieved, strict=True):
        assert estimate == expected_aggregates[name], name
    fact_values_by_name = {
        f"{compiled.name}@{compiled.period}": compiled.value
        for compiled in registry.specs
    }
    for name, target_value in zip(problem.names, problem.target_vector, strict=True):
        assert target_value == fact_values_by_name[name], name


def _real_uk_consumer_fact_rows() -> list[dict]:
    """Public rows copied from .codex-work/consumer_facts_uk.jsonl."""

    return [
        {
            "aggregate_fact_key": "ledger.aggregate_fact.v2:a36696207826083b6dd7e878",
            "aggregation": {"method": "sum"},
            "assertion": "observation",
            "concept_alignment": {
                "authority": "isc",
                "canonical_concept": "isc.pupils_at_member_schools",
                "relation": "source_label",
                "source_concept": "isc.pupils_at_member_schools",
            },
            "dimension_set_key": "ledger.dimension_set.v2:798fe9e6b99defdb40a97be3",
            "dimensions": {"school_membership": "isc"},
            "entity": {"name": "person", "role": "private_school_pupil"},
            "geography": {
                "id": "K02000001",
                "level": "country",
                "vintage": "current",
            },
            "layout": {
                "groupby_dimension": "isc.census_line",
                "groupby_value_id": "pupils_at_isc_schools",
                "measure_id": "pupils",
                "record_set_id": "isc.census_2023.pupils",
            },
            "legacy_fact_key": "ledger.fact.v1:6862b35fc66e993fd1b3a496",
            "lineage": {
                "source_record_id": (
                    "isc.census_2023.pupils.pupils_at_isc_schools.pupils"
                )
            },
            "observed_measure": {
                "source_concept": "isc.pupils_at_member_schools",
                "source_measure_id": "pupils",
                "source_name": "isc",
                "source_table": "ISC annual Census 2023",
                "unit": "count",
            },
            "period": {"type": "month", "value": "2023-01"},
            "semantic_fact_key": "ledger.semantic_fact.v2:dcdb6fab7a97e0d16f5332a0",
            "source": {
                "source_file": "isc_census_2023_final.pdf",
                "source_name": "isc",
                "source_table": "ISC annual Census 2023",
                "url": "https://www.isc.co.uk/media/9316/isc_census_2023_final.pdf",
                "vintage": "census_2023",
            },
            "universe_constraint_set_key": (
                "ledger.universe_constraint_set.v2:6128dc3fab81d86b2f1e7d92"
            ),
            "universe_constraints": {"domain": "independent_school_education"},
            "value": 554_243,
            "value_type": "integer",
        },
        {
            "aggregate_fact_key": "ledger.aggregate_fact.v2:2d8306be080946696b35e1ee",
            "aggregation": {"method": "sum"},
            "assertion": "observation",
            "concept_alignment": {
                "authority": "ons",
                "canonical_concept": "ons.household_interest_resources",
                "relation": "source_label",
                "source_concept": "ons.ukea_haxv",
            },
            "dimension_set_key": "ledger.dimension_set.v2:44136fa355b3678a1146ad16",
            "dimensions": {},
            "entity": {"name": "household", "role": "resident_household"},
            "geography": {
                "id": "K02000001",
                "level": "country",
                "vintage": "current",
            },
            "layout": {
                "groupby_dimension": "ons.ukea_series",
                "groupby_value_id": "haxv",
                "measure_id": "amount",
                "record_set_id": "ons.ukea_haxv.savings_interest.cy2023",
            },
            "legacy_fact_key": "ledger.fact.v1:26472d1cd1730a29a966ad7c",
            "lineage": {
                "source_record_id": "ons.ukea_haxv.savings_interest.cy2023.haxv.amount"
            },
            "observed_measure": {
                "source_concept": "ons.ukea_haxv",
                "source_measure_id": "amount",
                "source_name": "ons",
                "source_table": (
                    "UKEA time series HAXV: Households (S.14) interest (D.41) "
                    "resources, current price GBP million NSA"
                ),
                "unit": "gbp",
            },
            "period": {"type": "calendar_year", "value": 2023},
            "semantic_fact_key": "ledger.semantic_fact.v2:9ac263134990690ff4d4aeca",
            "source": {
                "source_file": "haxv.csv",
                "source_name": "ons",
                "source_table": (
                    "UKEA time series HAXV: Households (S.14) interest (D.41) "
                    "resources, current price GBP million NSA"
                ),
                "url": (
                    "https://www.ons.gov.uk/generator?format=csv&uri=/economy/"
                    "grossdomesticproductgdp/timeseries/haxv/ukea"
                ),
                "vintage": "ukea_2026_06_30",
            },
            "universe_constraint_set_key": (
                "ledger.universe_constraint_set.v2:fa6e32cc24b49ed112373325"
            ),
            "universe_constraints": {"domain": "national_accounts"},
            "value": 86_040_000_000,
            "value_type": "integer",
        },
    ]
