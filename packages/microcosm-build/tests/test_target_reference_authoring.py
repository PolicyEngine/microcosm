from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from microcosm.build.target_reference_authoring import (
    AreaSignedDeferral,
    AreaTargetReferenceAuthoringConfig,
    author_area_target_references,
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
    # not carry the declared 32 Scottish, 11 Northern Irish, and 22 Welsh
    # authorities
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
            "family": "hmrc",
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
        "country": "uk",
        "targets": [
            {
                "target_id": "ons.age.0_10",
                "family": "ons_population",
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
        "value": value,
        "period": {"type": "calendar_year", "value": 2025},
        "geography": {
            "level": "constituency",
            "id": area_id,
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
        "layout": {
            "record_set_id": record_set_id,
            "record_set_spec_id": record_set_spec_id,
            "groupby_dimension": groupby_dimension,
            "groupby_value_id": groupby_value_id,
            "measure_id": measure_id,
        },
    }
