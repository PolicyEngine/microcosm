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

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

ACTIVE_REFERENCE_COUNT = 15

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


def test_uk_target_references_load_as_typed_non_empty_resource() -> None:
    spec = load_country_spec("uk")

    assert len(spec.target_references) == ACTIVE_REFERENCE_COUNT
    assert {reference.name for reference in spec.target_references}


def test_uk_target_references_follow_contract_derivation_rules() -> None:
    resource = _load_uk_resource("target_references.json")
    targets_by_id = _contract_targets_by_id()
    names = [reference["name"] for reference in resource["target_references"]]

    assert resource["country"] == "uk"
    assert resource["allowed_value_operations"] == ["identity"]
    assert len(names) == len(set(names))

    for reference in resource["target_references"]:
        contract_target_id = reference["metadata"]["contract_target_id"]
        target = targets_by_id[contract_target_id]
        binding = target["bindings"]["policyengine"]

        assert reference["name"] == contract_target_id
        assert reference["ledger_selector"] == {
            **target["ledger_selector"],
            "geography_level": "country",
        }
        assert reference["entity"] == _expected_reference_entity(target)
        assert reference["measure"] == binding["metric_name"]
        assert reference["family"] == target["family"]
        expected_period = (
            2025
            if contract_target_id
            in {
                "dwp.uc.households",
                "obr.universal_credit_in_cap",
            }
            else 2023
        )
        assert reference["period"] == expected_period
        assert reference["metadata"] == {
            "contract_target_id": contract_target_id,
            "measure_kind": "prepared_column",
        }
        # The measure is a prepared column, so the pointed-to contract binding
        # must carry what the microcosm#622 materializer needs to prepare it.
        assert binding.get("value_variable"), contract_target_id


def test_uk_target_references_compile_from_real_staged_feed_rows() -> None:
    spec = load_country_spec("uk")
    references = [
        reference
        for reference in spec.target_references
        if reference.name
        in {"isc.private_school_students", "ons.savings_interest_income"}
    ]

    registry = compile_ledger_target_references(
        _real_uk_consumer_fact_rows(),
        references,
        country="uk",
    )

    targets = {target.name: target for target in registry.specs}
    assert set(targets) == {
        "isc.private_school_students",
        "ons.savings_interest_income",
    }

    private_school_students = targets["isc.private_school_students"]
    assert private_school_students.value == 554_243
    assert private_school_students.period == 2023
    assert private_school_students.entity == "person"
    assert private_school_students.measure == "obr/private_school_students"
    assert private_school_students.family == "isc"
    assert (
        private_school_students.metadata["contract_target_id"]
        == "isc.private_school_students"
    )
    assert (
        private_school_students.metadata["ledger_aggregate_fact_key"]
        == "ledger.aggregate_fact.v2:a36696207826083b6dd7e878"
    )

    savings_interest = targets["ons.savings_interest_income"]
    assert savings_interest.value == 86_040_000_000
    assert savings_interest.period == 2023
    assert savings_interest.entity == "person"
    assert savings_interest.measure == "ons/savings_interest_income"
    assert savings_interest.family == "ons_national_accounts"
    assert (
        savings_interest.metadata["contract_target_id"] == "ons.savings_interest_income"
    )
    assert (
        savings_interest.metadata["ledger_aggregate_fact_key"]
        == "ledger.aggregate_fact.v2:2d8306be080946696b35e1ee"
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
    references = tuple(
        reference for reference in spec.target_references if reference.period == 2023
    )
    feed_rows = [
        json.loads(line)
        for line in FIXTURE_FEED_ROWS.read_text().splitlines()
        if line.strip()
    ]

    registry = compile_ledger_target_references(feed_rows, references, country="uk")
    assert len(registry.specs) == 13

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
    assert len(problem.names) == 13
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
