from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from microcosm.build.target_reference_authoring import (
    AreaSignedDeferral,
    AreaTargetReferenceAuthoringConfig,
    TargetReferenceAuthoringConfig,
    author_area_target_references,
    author_target_references,
    target_references_resource,
)


def test_author_area_target_references_fans_out_roster_and_resolves_operations() -> (
    None
):
    contract = _contract()
    facts = [
        _area_fact("ons", "population", 10.0, area_id="A1", fact_key="a1-age-0"),
        _area_fact("ons", "population", 20.0, area_id="A1", fact_key="a1-age-1"),
        _area_fact("ons", "population", 30.0, area_id="A2", fact_key="a2-age-0"),
        _area_fact("ons", "population", 40.0, area_id="A2", fact_key="a2-age-1"),
        _area_fact(
            "hmrc",
            "employment_income_count",
            2.0,
            area_id="A1",
            fact_key="a1-employment-count",
        ),
        _area_fact(
            "hmrc",
            "employment_income_mean",
            100.0,
            area_id="A1",
            fact_key="a1-employment-mean",
            aggregation="mean",
        ),
        _area_fact(
            "hmrc",
            "employment_income_count",
            3.0,
            area_id="A2",
            fact_key="a2-employment-count",
        ),
        _area_fact(
            "hmrc",
            "employment_income_mean",
            200.0,
            area_id="A2",
            fact_key="a2-employment-mean",
            aggregation="mean",
        ),
    ]
    for fact in facts:
        fact["period"]["value"] = 2024

    authored = author_area_target_references(
        contract,
        facts,
        _area_config(
            areas=("A1", "A2"),
            value_operation_by_target_id={
                "ons.age.0_10": "sum",
                "hmrc.employment_income.amount": "count_x_mean",
            },
        ),
    )

    assert [reference["name"] for reference in authored.references] == [
        "ons.age.0_10@A1",
        "ons.age.0_10@A2",
        "hmrc.employment_income.amount@A1",
        "hmrc.employment_income.amount@A2",
    ]
    assert authored.status_counts == {"active": 4}
    assert (
        _candidate(authored.membership_report, "ons.age.0_10", "A1")["resolved_value"]
        == 30.0
    )
    assert (
        _candidate(
            authored.membership_report,
            "hmrc.employment_income.amount",
            "A2",
        )["resolved_value"]
        == 600.0
    )
    amount_reference = authored.references[2]
    assert amount_reference["measure"] == "hmrc/employment_income/amount"
    assert amount_reference["ledger_selector"]["geography_id"] == "A1"
    assert amount_reference["value_operation"] == "count_x_mean"
    assert amount_reference["hierarchy"]["target_label"] == "Employment income"
    assert authored.membership_report["uprating_holds"] == [
        {
            "name": "ons.age.0_10@A1",
            "target_id": "ons.age.0_10",
            "geography_level": "constituency",
            "geography_id": "A1",
            "from": "2024",
            "to": "2025",
        },
        {
            "name": "ons.age.0_10@A2",
            "target_id": "ons.age.0_10",
            "geography_level": "constituency",
            "geography_id": "A2",
            "from": "2024",
            "to": "2025",
        },
        {
            "name": "hmrc.employment_income.amount@A1",
            "target_id": "hmrc.employment_income.amount",
            "geography_level": "constituency",
            "geography_id": "A1",
            "from": "2024",
            "to": "2025",
        },
        {
            "name": "hmrc.employment_income.amount@A2",
            "target_id": "hmrc.employment_income.amount",
            "geography_level": "constituency",
            "geography_id": "A2",
            "from": "2024",
            "to": "2025",
        },
    ]
    assert authored.membership_report["holds_by_target"] == {
        "hmrc.employment_income.amount": 2,
        "ons.age.0_10": 2,
    }

    resource = target_references_resource(
        country="uk",
        description="synthetic",
        authored=authored,
        hierarchy=contract["hierarchy"],
    )
    assert resource["hierarchy"]["target_labels"] == {
        "hmrc.employment_income.amount": "Employment income",
        "ons.age.0_10": "People aged 0 to 9",
    }


def test_authoring_refuses_target_without_a_declared_category() -> None:
    contract = _single_age_contract()
    contract["targets"][0].pop("category_id")

    with pytest.raises(ValueError, match="references unknown hierarchy category"):
        author_area_target_references(
            contract,
            [],
            _area_config(areas=("A1",)),
        )


def test_authoring_refuses_target_without_a_chronicle_selector() -> None:
    contract = _single_age_contract()
    contract["targets"][0].pop("ledger_selector")

    with pytest.raises(ValueError, match="must declare a non-empty ledger_selector"):
        author_area_target_references(
            contract,
            [],
            _area_config(areas=("A1",)),
        )


def test_authoring_refuses_transformed_target_without_an_explicit_label() -> None:
    contract = _single_age_contract()
    contract["targets"][0].pop("label")
    fact = _area_fact(
        "ons",
        "population",
        10.0,
        area_id="A1",
        fact_key="a1-age-0",
    )

    with pytest.raises(ValueError, match="require an explicit Microcosm target label"):
        author_area_target_references(
            contract,
            [fact],
            _area_config(
                areas=("A1",),
                value_operation_by_target_id={"ons.age.0_10": "sum"},
            ),
        )


def test_authoring_refuses_target_materialization() -> None:
    contract = _single_age_contract()
    contract["targets"][0]["materialization"] = {"kind": "synthetic_runtime_value"}

    with pytest.raises(ValueError, match="declares unsupported materialization"):
        author_area_target_references(
            contract,
            [],
            _area_config(areas=("A1",)),
        )


def test_author_area_target_references_refuses_unsigned_absence() -> None:
    contract = _single_age_contract()
    facts = [
        _area_fact("ons", "population", 10.0, area_id="A1", fact_key="a1-age-0"),
    ]

    with pytest.raises(ValueError, match="Unsigned local target absence"):
        author_area_target_references(
            contract,
            facts,
            _area_config(
                areas=("A1", "A2"),
                value_operation_by_target_id={"ons.age.0_10": "sum"},
            ),
        )


def test_author_area_target_references_records_signed_no_fact_for_area() -> None:
    contract = _single_age_contract()
    facts = [
        _area_fact("ons", "population", 10.0, area_id="A1", fact_key="a1-age-0"),
    ]

    authored = author_area_target_references(
        contract,
        facts,
        _area_config(
            areas=("A1", "A2"),
            value_operation_by_target_id={"ons.age.0_10": "sum"},
            area_signed_deferrals=(
                AreaSignedDeferral(
                    target_id="ons.age.0_10",
                    geography_level="constituency",
                    reason_id="test_absence",
                    rationale="Synthetic area A2 has no fact by construction.",
                    area_ids=("A2",),
                ),
            ),
        ),
    )

    assert [reference["name"] for reference in authored.references] == [
        "ons.age.0_10@A1"
    ]
    assert authored.status_counts == {"active": 1, "no_fact_for_area": 1}
    candidate = _candidate(authored.membership_report, "ons.age.0_10", "A2")
    assert candidate["status"] == "no_fact_for_area"
    assert candidate["signed_reason_id"] == "test_absence"


def test_author_area_target_references_refuses_stale_signing() -> None:
    contract = _single_age_contract()
    facts = [
        _area_fact("ons", "population", 10.0, area_id="A1", fact_key="a1-age-0"),
    ]

    with pytest.raises(ValueError, match="Stale area signed deferral"):
        author_area_target_references(
            contract,
            facts,
            _area_config(
                areas=("A1",),
                value_operation_by_target_id={"ons.age.0_10": "sum"},
                area_signed_deferrals=(
                    AreaSignedDeferral(
                        target_id="ons.age.0_10",
                        geography_level="constituency",
                        reason_id="stale",
                        rationale="This signing should fail once the fact exists.",
                        area_ids=("A1",),
                    ),
                ),
            ),
        )


def test_author_area_target_references_records_signed_compilable_deferral() -> None:
    contract = _single_age_contract()
    facts = [
        _area_fact("ons", "population", 10.0, area_id="A1", fact_key="a1-age-0"),
    ]

    authored = author_area_target_references(
        contract,
        facts,
        _area_config(
            areas=("A1",),
            value_operation_by_target_id={"ons.age.0_10": "sum"},
            area_signed_deferrals=(
                AreaSignedDeferral(
                    target_id="ons.age.0_10",
                    geography_level="constituency",
                    reason_id="separate_adjudication",
                    rationale=(
                        "The local fact compiles but a separately adjudicated "
                        "constraint prevents it from binding."
                    ),
                    area_ids=("A1",),
                    defer_if_compiles=True,
                ),
            ),
        ),
    )

    assert authored.references == ()
    assert authored.status_counts == {"signed_deferred": 1}
    candidate = _candidate(authored.membership_report, "ons.age.0_10", "A1")
    assert candidate["status"] == "signed_deferred"
    assert candidate["signed_reason_id"] == "separate_adjudication"
    assert (
        authored.membership_report["signed_deferrals"][0]["defer_if_compiles"] is True
    )


def test_author_area_target_references_refuses_signed_area_outside_roster() -> None:
    contract = _single_age_contract()

    with pytest.raises(ValueError, match="outside the roster"):
        author_area_target_references(
            contract,
            [],
            _area_config(
                areas=("A1",),
                area_signed_deferrals=(
                    AreaSignedDeferral(
                        target_id="ons.age.0_10",
                        geography_level="constituency",
                        reason_id="outside",
                        rationale="A3 is not in the declared roster.",
                        area_ids=("A3",),
                    ),
                ),
            ),
        )


def test_generate_uk_local_target_references_cli_refuses_incomplete_contract(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[3]
    contract_path = tmp_path / "contract.json"
    facts_path = tmp_path / "facts.jsonl"
    crosswalk_path = tmp_path / "crosswalk.json"
    output_path = tmp_path / "local_target_references.json"
    membership_path = tmp_path / "membership.json"
    contract_path.write_text(json.dumps(_single_age_contract()), encoding="utf-8")
    facts_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _area_fact("ons", "population", 10.0, area_id="A1", fact_key="a1-0"),
                _area_fact("ons", "population", 20.0, area_id="A1", fact_key="a1-1"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    # The council-tax country masks refuse a local-authority roster that does
    # not carry the declared 32 Scottish, 11 Northern Irish, 22 Welsh, and
    # 296 English authorities
    # (covered by test_council_tax_country_masks_refuse_roster_count_drift).
    # This test is about the *contract* refusal, so give it a roster those
    # masks accept and let the missing target id be what fails.
    crosswalk_path.write_text(
        json.dumps(
            {
                "levels": {
                    "constituency": {"area_ids": ["A1"]},
                    "local_authority": {
                        "area_ids": [f"S{index:08d}" for index in range(32)]
                        + [f"N{index:08d}" for index in range(11)]
                        + [f"W{index:08d}" for index in range(22)]
                        + [f"E{index:08d}" for index in range(294)]
                        + ["E06000053", "E09000001"]
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/generate_uk_local_target_references.py",
            "--contract",
            str(contract_path),
            "--ledger-facts",
            str(facts_path),
            "--crosswalk",
            str(crosswalk_path),
            "--output",
            str(output_path),
            "--membership-report",
            str(membership_path),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "dwp.uc.households_by_area" in result.stderr
    assert "absent from the local contract" in result.stderr
    assert not output_path.exists()
    assert not membership_path.exists()


def _candidate(report: dict, target_id: str, area_id: str) -> dict:
    candidates = report["targets"][target_id]["geography_levels"]["constituency"][
        "candidates"
    ]
    for candidate in candidates:
        if candidate["geography_id"] == area_id:
            return candidate
    raise AssertionError(f"Missing candidate {target_id}@{area_id}.")


def _area_config(
    *,
    areas: tuple[str, ...],
    value_operation_by_target_id: dict[str, str] | None = None,
    area_signed_deferrals: tuple[AreaSignedDeferral, ...] = (),
) -> AreaTargetReferenceAuthoringConfig:
    return AreaTargetReferenceAuthoringConfig(
        target_period=2025,
        areas_by_geography_level={"constituency": areas},
        area_signed_deferrals=area_signed_deferrals,
        value_operation_by_target_id=value_operation_by_target_id or {},
        source_fact_feed="synthetic",
    )


def _contract() -> dict:
    contract = _single_age_contract()
    contract["targets"].append(
        {
            "target_id": "hmrc.employment_income.amount",
            "label": "Employment income",
            "family": "hmrc",
            "category_id": "hmrc.employment",
            "geography_levels": ["constituency"],
            "ledger_selector": {
                "source_name": "hmrc",
                "source_measure_id": [
                    "employment_income_count",
                    "employment_income_mean",
                ],
                "record_set_spec_id": "uk.local_geography.spi_income.by_constituency.v1",
            },
            "measurement": {"entity": "person", "concept": "uk.income.employment"},
            "bindings": {
                "policyengine": {
                    "metric_name": "hmrc/employment_income/amount",
                    "value_variable": "employment_income",
                    "from_entity": "person",
                }
            },
        }
    )
    return contract


def _single_age_contract() -> dict:
    return {
        "schema_version": 2,
        "country": "uk",
        "hierarchy": {
            "providers": {
                "hmrc": {"label": "HM Revenue and Customs"},
                "ons": {"label": "Office for National Statistics"},
            },
            "categories": {
                "hmrc.employment": {
                    "provider_id": "hmrc",
                    "label": "Employment income",
                },
                "ons.population": {
                    "provider_id": "ons",
                    "label": "Population",
                },
            },
        },
        "targets": [
            {
                "target_id": "ons.age.0_10",
                "label": "People aged 0 to 9",
                "family": "ons_population",
                "category_id": "ons.population",
                "geography_levels": ["constituency"],
                "ledger_selector": {
                    "source_name": "ons",
                    "source_measure_id": "population",
                    "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
                },
                "measurement": {"entity": "person", "concept": "uk.person.count"},
                "bindings": {
                    "policyengine": {
                        "metric_name": "age/0_10",
                        "value_variable": "person_count",
                        "from_entity": "person",
                    }
                },
            }
        ],
    }


def _area_fact(
    source_name: str,
    measure_id: str,
    value: float,
    *,
    area_id: str,
    fact_key: str,
    aggregation: str = "sum",
) -> dict:
    if source_name == "ons":
        record_set_spec_id = "uk.local_geography.population.age_0_10.v1"
        record_set_id = "ons.population.local_age"
        dimensions = {"synthetic_band": fact_key}
        groupby_dimension = "synthetic_band"
        groupby_value_id = fact_key
    else:
        record_set_spec_id = "uk.local_geography.spi_income.by_constituency.v1"
        record_set_id = "hmrc.spi.local_income"
        dimensions = {}
        groupby_dimension = ""
        groupby_value_id = ""
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{fact_key}",
        "legacy_fact_key": f"ledger.fact.v1:{fact_key}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{fact_key}",
        "lineage": {"source_record_id": f"{source_name}.{fact_key}"},
        "label": f"{source_name.upper()} {measure_id.replace('_', ' ')}",
        "value": value,
        "period": {"type": "calendar_year", "value": 2025},
        "geography": {
            "level": "constituency",
            "id": area_id,
            "name": f"Area {area_id}",
            "vintage": "pcon_2024",
        },
        "entity": {"name": "person"},
        "observed_measure": {
            "source_name": source_name,
            "source_measure_id": measure_id,
            "source_concept": f"{source_name}.{measure_id}",
            "unit": "count",
        },
        "aggregation": {"method": aggregation},
        "source": {
            "source_name": source_name,
            "source_table": "synthetic",
            "source_file": "synthetic.csv",
            "vintage": "synthetic",
        },
        "dimensions": dimensions,
        "dimension_labels": (
            {"synthetic_band": "Synthetic age band"} if dimensions else {}
        ),
        "dimension_value_labels": (
            {"synthetic_band": {fact_key: f"Synthetic band {fact_key}"}}
            if dimensions
            else {}
        ),
        "layout": {
            "record_set_id": record_set_id,
            "record_set_spec_id": record_set_spec_id,
            "groupby_dimension": groupby_dimension,
            "groupby_value_id": groupby_value_id,
            "groupby_dimension_label": (
                "Synthetic age band" if groupby_dimension else ""
            ),
            "groupby_value_label": (
                f"Synthetic band {fact_key}" if groupby_value_id else ""
            ),
            "measure_id": measure_id,
        },
    }


_REGION_TIER_CELLS = (
    ("region", "E12000007"),
    ("region", "E12000001"),
    ("country", "W92000004"),
)


def _tier_fact(level: str, area_id: str, value: float, fact_key: str) -> dict:
    fact = _area_fact("ons", "population", value, area_id=area_id, fact_key=fact_key)
    fact["geography"] = {"level": level, "id": area_id, "vintage": "gss_2024"}
    return fact


def _two_level_contract() -> dict:
    contract = _single_age_contract()
    target = contract["targets"][0]
    target["target_id"] = "ons.age.0_10_by_region"
    target["geography_levels"] = ["country", "region"]
    return contract


def _tier_metadata(
    target: dict, geography_level: str, geography_id: str, entity: str
) -> dict[str, str]:
    del target, geography_level
    return {
        "geography_predicate": json.dumps(
            {
                "variable": "region",
                "operator": "==",
                "value": geography_id,
                "map_to": entity,
            }
        ),
        "cross_grain_grain": "region",
    }


def _tier_facts() -> list[dict]:
    return [
        _tier_fact("region", "E12000007", 10.0, "london"),
        _tier_fact("region", "E12000001", 20.0, "north-east"),
        _tier_fact("country", "W92000004", 30.0, "wales"),
        _tier_fact("country", "K02000001", 60.0, "uk"),
    ]


def test_author_target_references_fans_out_over_a_geography_roster() -> None:
    config = TargetReferenceAuthoringConfig(
        target_period=2025,
        geography_fanout_by_target_id={"ons.age.0_10_by_region": _REGION_TIER_CELLS},
        geography_fanout_metadata=_tier_metadata,
        source_fact_feed="synthetic",
    )
    authored = author_target_references(_two_level_contract(), _tier_facts(), config)
    names = [row["name"] for row in authored.references]
    assert names == [
        "ons.age.0_10_by_region@E12000007",
        "ons.age.0_10_by_region@E12000001",
        "ons.age.0_10_by_region@W92000004",
    ]
    london = authored.references[0]
    # Each cell is its own measure column, never the shared metric name.
    assert london["measure"] == london["name"]
    assert london["entity"] == "person"
    assert london["ledger_selector"]["geography_level"] == "region"
    assert london["ledger_selector"]["geography_id"] == "E12000007"
    assert london["metadata"]["contract_target_id"] == "ons.age.0_10_by_region"
    assert london["metadata"]["geography_level"] == "region"
    assert london["metadata"]["geography_id"] == "E12000007"
    assert json.loads(london["metadata"]["geography_predicate"]) == {
        "variable": "region",
        "operator": "==",
        "value": "E12000007",
        "map_to": "person",
    }
    assert london["metadata"]["cross_grain_grain"] == "region"
    wales = authored.references[2]
    assert wales["ledger_selector"]["geography_level"] == "country"
    # The UK-wide fact is no cell of the roster: nothing binds it.
    assert "K02000001" not in json.dumps(authored.references)

    report = authored.membership_report
    assert report["active_reference_count"] == 3
    assert report["status_counts"] == {"active": 3}
    assert report["geography_pins"]["ons.age.0_10_by_region"] == {
        "geography_fanout": [
            {"geography_level": level, "geography_id": area_id}
            for level, area_id in _REGION_TIER_CELLS
        ]
    }
    candidates = report["targets"]["ons.age.0_10_by_region"]["candidates"]
    assert [
        (
            entry["geography_level"],
            entry["geography_id"],
            entry["status"],
            entry["resolved_value"],
        )
        for entry in candidates
    ] == [
        ("region", "E12000007", "active", 10.0),
        ("region", "E12000001", "active", 20.0),
        ("country", "W92000004", "active", 30.0),
    ]


def test_geography_fanout_refuses_an_absent_roster_cell() -> None:
    config = TargetReferenceAuthoringConfig(
        target_period=2025,
        geography_fanout_by_target_id={"ons.age.0_10_by_region": _REGION_TIER_CELLS},
        source_fact_feed="synthetic",
    )
    facts = [fact for fact in _tier_facts() if fact["geography"]["id"] != "W92000004"]
    with pytest.raises(ValueError, match="Unsigned geography fan-out absence") as info:
        author_target_references(_two_level_contract(), facts, config)
    assert "W92000004" in str(info.value)


def test_geography_fanout_refuses_a_pin_on_the_same_target() -> None:
    config = TargetReferenceAuthoringConfig(
        target_period=2025,
        geography_pins={
            "ons.age.0_10_by_region": {
                "geography_level": "country",
                "geography_id": "K02000001",
            }
        },
        geography_fanout_by_target_id={"ons.age.0_10_by_region": _REGION_TIER_CELLS},
        source_fact_feed="synthetic",
    )
    with pytest.raises(ValueError, match="both a geography pin"):
        author_target_references(_two_level_contract(), _tier_facts(), config)


def test_geography_fanout_refuses_duplicate_cells_and_non_string_metadata() -> None:
    duplicated = TargetReferenceAuthoringConfig(
        target_period=2025,
        geography_fanout_by_target_id={
            "ons.age.0_10_by_region": (("region", "E12000007"), ("region", "E12000007"))
        },
        source_fact_feed="synthetic",
    )
    with pytest.raises(ValueError, match="twice"):
        author_target_references(_two_level_contract(), _tier_facts(), duplicated)

    non_string = TargetReferenceAuthoringConfig(
        target_period=2025,
        geography_fanout_by_target_id={"ons.age.0_10_by_region": _REGION_TIER_CELLS},
        geography_fanout_metadata=lambda *args: {"cross_grain_grain": 1},
        source_fact_feed="synthetic",
    )
    with pytest.raises(ValueError, match="must be a string"):
        author_target_references(_two_level_contract(), _tier_facts(), non_string)


def test_area_scope_limits_a_target_to_its_declared_roster_areas() -> None:
    contract = _contract()
    facts = [
        _area_fact("ons", "population", 10.0, area_id="A1", fact_key="a1-age-0"),
        _area_fact("ons", "population", 30.0, area_id="A2", fact_key="a2-age-0"),
        _area_fact("hmrc", "employment_income_count", 2.0, area_id="A1", fact_key="c1"),
        _area_fact(
            "hmrc",
            "employment_income_mean",
            100.0,
            area_id="A1",
            fact_key="m1",
            aggregation="mean",
        ),
    ]
    config = AreaTargetReferenceAuthoringConfig(
        target_period=2025,
        areas_by_geography_level={"constituency": ("A1", "A2")},
        value_operation_by_target_id={
            "ons.age.0_10": "sum",
            "hmrc.employment_income.amount": "count_x_mean",
        },
        area_scope_by_target_id={
            "hmrc.employment_income.amount": {"constituency": frozenset({"A1"})}
        },
        source_fact_feed="synthetic",
    )

    authored = author_area_target_references(contract, facts, config)

    assert [row["name"] for row in authored.references] == [
        "ons.age.0_10@A1",
        "ons.age.0_10@A2",
        "hmrc.employment_income.amount@A1",
    ]
    report = authored.membership_report
    assert report["candidate_count"] == 3
    assert report["area_scope_by_target_id"] == {
        "hmrc.employment_income.amount": {"constituency": ["A1"]}
    }


def test_area_scope_naming_an_area_outside_the_roster_refuses() -> None:
    config = AreaTargetReferenceAuthoringConfig(
        target_period=2025,
        areas_by_geography_level={"constituency": ("A1",)},
        area_scope_by_target_id={"ons.age.0_10": {"constituency": frozenset({"Z9"})}},
        source_fact_feed="synthetic",
    )
    with pytest.raises(ValueError, match="names areas the roster does not carry"):
        author_area_target_references(_contract(), [], config)


def test_area_id_aliases_select_the_publisher_recoding_and_keep_the_roster_code() -> (
    None
):
    contract = _contract()
    contract["targets"] = [
        target
        for target in contract["targets"]
        if target["target_id"] == "ons.age.0_10"
    ]
    old = _area_fact("ons", "population", 10.0, area_id="A1", fact_key="a1-old")
    old["period"]["value"] = 2024
    new = _area_fact("ons", "population", 12.0, area_id="A1X", fact_key="a1-new")
    new["geography"]["name"] = "Area A1"
    other = _area_fact("ons", "population", 30.0, area_id="A2", fact_key="a2")
    config = AreaTargetReferenceAuthoringConfig(
        target_period=2025,
        areas_by_geography_level={"constituency": ("A1", "A2")},
        value_operation_by_target_id={"ons.age.0_10": "sum"},
        area_id_aliases={"constituency": {"A1": ("A1X",)}},
        source_fact_feed="synthetic",
    )

    authored = author_area_target_references(contract, [old, new, other], config)

    row = authored.references[0]
    assert row["name"] == "ons.age.0_10@A1"
    assert row["ledger_selector"]["geography_id"] == ["A1", "A1X"]
    assert row["metadata"]["geography_id"] == "A1"
    assert row["metadata"]["geography_id_aliases"] == "A1X"
    # The 2025 row filed under the alias wins over the 2024 row under the
    # roster code: the alias is a spelling of the same area, not a new one.
    assert (
        _candidate(authored.membership_report, "ons.age.0_10", "A1")["resolved_value"]
        == 12.0
    )
    assert authored.membership_report["uprating_holds"] == []


def test_geography_composition_sums_member_rows_into_the_declared_cell() -> None:
    contract = _two_level_contract()
    contract["targets"][0]["value_operation"] = "sum"
    facts = [
        _tier_fact("local_authority", "L1", 4.0, "l1"),
        _tier_fact("local_authority", "L2", 6.0, "l2"),
        _tier_fact("local_authority", "L2X", 7.0, "l2x"),
        _tier_fact("local_authority", "L3", 9.0, "l3"),
    ]
    facts[2]["period"]["value"] = 2025
    for fact in facts[:2] + facts[3:]:
        fact["period"]["value"] = 2025
    facts[1]["period"]["value"] = 2024  # L2's row under the roster code is stale
    config = TargetReferenceAuthoringConfig(
        target_period=2025,
        geography_fanout_by_target_id={
            "ons.age.0_10_by_region": (("region", "E12000007"),)
        },
        geography_fanout_metadata=_tier_metadata,
        geography_composition_by_target_id={
            "ons.age.0_10_by_region": {
                "E12000007": ("local_authority", ("L1", "L2", "L2X"))
            }
        },
        geography_composition_aliases={"local_authority": {"L2": ("L2X",)}},
        value_operation_by_target_id={"ons.age.0_10_by_region": "sum"},
        source_fact_feed="synthetic",
    )

    authored = author_target_references(contract, facts, config)

    row = authored.references[0]
    assert row["name"] == "ons.age.0_10_by_region@E12000007"
    assert row["ledger_selector"] == {
        "source_name": "ons",
        "source_measure_id": "population",
        "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
        "geography_level": "local_authority",
        "geography_id": ["L1", "L2", "L2X"],
    }
    assert row["expected_member_count"] == 2
    assert row["metadata"]["geography_level"] == "region"
    assert row["metadata"]["geography_id"] == "E12000007"
    assert row["metadata"]["composed_from_level"] == "local_authority"
    assert row["metadata"]["composed_member_count"] == "2"
    assert row["metadata"]["cross_grain_grain"] == "region"
    candidates = authored.membership_report["targets"]["ons.age.0_10_by_region"][
        "candidates"
    ]
    assert candidates[0]["status"] == "active"
    assert candidates[0]["resolved_value"] == 4.0 + 7.0
    assert authored.membership_report["geography_pins"]["ons.age.0_10_by_region"][
        "geography_composition"
    ] == {"E12000007": {"member_level": "local_authority", "member_count": 2}}


def test_geography_composition_refuses_a_non_aggregating_operation() -> None:
    config = TargetReferenceAuthoringConfig(
        target_period=2025,
        geography_fanout_by_target_id={
            "ons.age.0_10_by_region": (("region", "E12000007"),)
        },
        geography_fanout_metadata=_tier_metadata,
        geography_composition_by_target_id={
            "ons.age.0_10_by_region": {"E12000007": ("local_authority", ("L1",))}
        },
        source_fact_feed="synthetic",
    )
    with pytest.raises(ValueError, match="needs an aggregating operation"):
        author_target_references(
            _two_level_contract(),
            [_tier_fact("local_authority", "L1", 4.0, "l1")],
            config,
        )
