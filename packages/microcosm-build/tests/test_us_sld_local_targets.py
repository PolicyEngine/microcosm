"""Tests for SLD fact ingestion and problem compilation (populace#625)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.sld_local_targets import (
    SLD_ACS_MONEY_INCOME_RECIPE,
    build_sld_district_problems,
    household_acs_money_income,
    load_sld_target_facts,
    resolve_money_income_recipe,
)


def _fact(
    *,
    geo_id: str = "610U900US49001",
    level: str = "state_legislative_district_upper",
    concept: str = "census_acs.household_count",
    value: float = 100.0,
    aggregation: str = "sum",
    entity: str = "household",
    constraints: list | None = None,
    consumer_shape: bool = False,
) -> dict:
    row = {
        "geography": {
            "id": geo_id,
            "level": level,
            "vintage": "2024_state_legislative_districts",
        },
        "entity": {"name": entity, "role": f"resident_{entity}"},
        "aggregation": {"method": aggregation, "denominator": None},
        "period": {"type": "calendar_year", "value": 2024},
        "value": value,
    }
    if consumer_shape:
        row["concept_alignment"] = {"canonical_concept": concept}
        row["universe_constraints"] = {
            "domain": "households",
            "constraints": constraints or [],
        }
    else:
        row["measure"] = {"concept": concept}
        row["constraints"] = constraints or []
    return row


def _write_facts(tmp_path, rows):
    path = tmp_path / "facts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _income_constraints(lower, upper):
    constraints = []
    if lower is not None:
        constraints.append(
            {"variable": "household_income", "operator": ">=", "value": lower}
        )
    if upper is not None:
        constraints.append(
            {"variable": "household_income", "operator": "<", "value": upper}
        )
    return constraints


def _age_constraints(lower, upper):
    constraints = [{"variable": "age", "operator": ">=", "value": lower}]
    if upper is not None:
        constraints.append({"variable": "age", "operator": "<", "value": upper})
    return constraints


def test_load_parses_both_ledger_shapes_and_derives_metrics(tmp_path):
    rows = [
        _fact(value=100.0),
        _fact(
            value=40.0,
            constraints=_income_constraints(None, 10_000),
            consumer_shape=True,
        ),
        _fact(value=25.0, constraints=_income_constraints(200_000, None)),
        _fact(
            concept="census_acs.person_count",
            entity="person",
            value=30.0,
            constraints=_age_constraints(0, 5),
        ),
        _fact(
            concept="census_acs.person_count",
            entity="person",
            value=12.0,
            constraints=_age_constraints(85, None),
        ),
        _fact(
            concept="census_acs.median_household_income",
            value=95_236.0,
            aggregation="median",
        ),
        # A non-SLD row rides along untouched.
        {
            "geography": {"id": "0400000US49", "level": "state", "vintage": "current"},
            "value": 1.0,
            "measure": {"concept": "census_acs.person_count"},
            "aggregation": {"method": "sum"},
            "entity": {"name": "person"},
            "period": {"type": "calendar_year", "value": 2024},
        },
    ]
    facts = load_sld_target_facts(_write_facts(tmp_path, rows))
    assert sorted(facts.calibration["metric"]) == [
        "age_0_to_4",
        "age_85_and_over",
        "households",
        "income_200000_and_over",
        "income_under_10000",
    ]
    assert facts.calibration["state_fips"].unique().tolist() == ["49"]
    assert facts.calibration["district_code"].unique().tolist() == ["001"]
    assert facts.validation["metric"].tolist() == ["median_household_income"]
    assert facts.validation["value"].tolist() == [95_236.0]
    assert facts.geography_vintages == ("2024_state_legislative_districts",)
    assert len(facts.source_sha256) == 64


def test_load_refuses_unknown_concepts_and_duplicate_surface(tmp_path):
    with pytest.raises(ValueError, match="unrecognized SLD fact concept"):
        load_sld_target_facts(
            _write_facts(
                tmp_path,
                [_fact(concept="census_acs.mystery_count")],
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        load_sld_target_facts(
            _write_facts(tmp_path, [_fact(value=1.0), _fact(value=2.0)])
        )


def test_load_refuses_median_summed_as_calibration(tmp_path):
    with pytest.raises(ValueError, match="median"):
        load_sld_target_facts(
            _write_facts(
                tmp_path,
                [
                    _fact(
                        concept="census_acs.median_household_income",
                        aggregation="sum",
                    )
                ],
            )
        )


def test_recipe_resolution_requires_core_components():
    with pytest.raises(ValueError, match="wages_salary_tips"):
        resolve_money_income_recipe(["age"], ["household_id"])
    columns = [
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "taxable_interest_income",
        "social_security_retirement",
    ]
    resolution = resolve_money_income_recipe(columns, [])
    assert set(resolution.person_columns) == set(columns)
    assert "retirement_pensions" in resolution.components_absent
    record = resolution.as_record()
    assert "supplemental_security_income" in record["declared_omissions"]
    assert "capital_gains" in record["declared_exclusions"]


def test_household_money_income_sums_person_and_household_columns():
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "rental_income": [5.0, 0.0],
        }
    )
    persons = pd.DataFrame(
        {
            "person_household_id": [1, 1, 2, 2],
            "age": [40.0, 35.0, 50.0, 10.0],
            "employment_income_before_lsr": [10.0, 20.0, 7.0, 99.0],
            "self_employment_income_before_lsr": [0.0, 1.0, 0.0, 0.0],
            "taxable_interest_income": [0.0, 0.0, 2.0, 0.0],
            "social_security_retirement": [0.0, 0.0, 3.0, 0.0],
        }
    )
    resolution = resolve_money_income_recipe(persons.columns, households.columns)
    income = household_acs_money_income(households, persons, resolution)
    # The 10-year-old's 99.0 is outside the ACS money-income universe.
    np.testing.assert_allclose(income, [36.0, 12.0])


def test_household_money_income_requires_age():
    households = pd.DataFrame({"household_id": [1]})
    persons = pd.DataFrame(
        {
            "person_household_id": [1],
            "employment_income_before_lsr": [1.0],
            "self_employment_income_before_lsr": [0.0],
            "taxable_interest_income": [0.0],
            "social_security_retirement": [0.0],
        }
    )
    resolution = resolve_money_income_recipe(persons.columns, households.columns)
    with pytest.raises(ValueError, match="aged 15 and over"):
        household_acs_money_income(households, persons, resolution)


def _frame_and_facts(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "state_fips": [49, 49, 49, 49],
            "sld_upper_code": ["001", "001", "002", "001"],
            "employment_income_before_lsr_hh_unused": [0.0] * 4,
        }
    )
    persons = pd.DataFrame(
        {
            "person_household_id": [1, 1, 2, 3, 4],
            "age": [30.0, 4.0, 90.0, 40.0, 50.0],
            "employment_income_before_lsr": [5_000.0, 0.0, 0.0, 9_000.0, 250_000.0],
            "self_employment_income_before_lsr": [0.0] * 5,
            "taxable_interest_income": [0.0] * 5,
            "social_security_retirement": [0.0, 0.0, 8_000.0, 0.0, 0.0],
        }
    )
    rows = [
        _fact(value=90.0),
        _fact(value=30.0, constraints=_income_constraints(None, 10_000)),
        _fact(value=60.0, constraints=_income_constraints(10_000, None)),
        _fact(
            concept="census_acs.person_count",
            entity="person",
            value=25.0,
            constraints=_age_constraints(0, 5),
        ),
        _fact(
            concept="census_acs.median_household_income",
            value=50_000.0,
            aggregation="median",
        ),
        # District 003 has the same fact surface but no assigned households.
        _fact(geo_id="610U900US49003", value=10.0),
        _fact(
            geo_id="610U900US49003",
            value=4.0,
            constraints=_income_constraints(None, 10_000),
        ),
        _fact(
            geo_id="610U900US49003",
            value=6.0,
            constraints=_income_constraints(10_000, None),
        ),
        _fact(
            geo_id="610U900US49003",
            concept="census_acs.person_count",
            entity="person",
            value=3.0,
            constraints=_age_constraints(0, 5),
        ),
    ]
    return households, persons, _write_facts(tmp_path, rows)


def test_build_problems_binds_hand_computable_measures(tmp_path):
    households, persons, path = _frame_and_facts(tmp_path)
    facts = load_sld_target_facts(path)
    build = build_sld_district_problems(
        households,
        persons,
        base_weights=np.array([10.0, 10.0, 10.0, 10.0]),
        facts=facts,
        area_type="sldu",
    )
    assert build.zero_support_districts == ("610U900US49003",)
    assert build.validation_facts["metric"].tolist() == ["median_household_income"]
    assert len(build.problems) == 1
    problem = build.problems[0]
    assert problem.area_code == "610U900US49001"
    # Canonical order: age band, households, then income brackets ascending.
    assert problem.target_frame["metric"].tolist() == [
        "age_0_to_4",
        "households",
        "income_under_10000",
        "income_10000_and_over",
    ]
    np.testing.assert_array_equal(problem.household_ids, [1, 2, 4])
    # Household incomes: hh1 = 5,000; hh2 = 8,000; hh4 = 250,000.
    np.testing.assert_allclose(
        problem.matrix,
        [
            [1.0, 0.0, 0.0],  # persons aged 0-4
            [1.0, 1.0, 1.0],  # households
            [1.0, 1.0, 0.0],  # income under 10k
            [0.0, 0.0, 1.0],  # income 10k and over
        ],
    )
    np.testing.assert_allclose(problem.targets, [25.0, 90.0, 30.0, 60.0])


def test_build_problems_requires_membership_column(tmp_path):
    households, persons, path = _frame_and_facts(tmp_path)
    facts = load_sld_target_facts(path)
    with pytest.raises(ValueError, match="sld_lower_code"):
        build_sld_district_problems(
            households,
            persons,
            base_weights=np.full(4, 10.0),
            facts=facts,
            area_type="sldl",
        )
    with pytest.raises(ValueError, match="area_type"):
        build_sld_district_problems(
            households,
            persons,
            base_weights=np.full(4, 10.0),
            facts=facts,
            area_type="county",
        )


def test_recipe_is_declared_and_ordered():
    names = [component.name for component in SLD_ACS_MONEY_INCOME_RECIPE]
    assert names[:4] == [
        "wages_salary_tips",
        "self_employment",
        "interest_dividends_rental_estates",
        "social_security",
    ]
    assert all(component.required for component in SLD_ACS_MONEY_INCOME_RECIPE[:4])


def test_group_quarters_rows_support_population_but_not_households(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "state_fips": [49, 49],
            "sld_upper_code": ["001", "001"],
            "TYPEHUGQ": [1.0, 3.0],  # row 2 is a noninstitutional GQ person
        }
    )
    persons = pd.DataFrame(
        {
            "person_household_id": [1, 2],
            "age": [30.0, 22.0],
            "employment_income_before_lsr": [5_000.0, 5_000.0],
            "self_employment_income_before_lsr": [0.0, 0.0],
            "taxable_interest_income": [0.0, 0.0],
            "social_security_retirement": [0.0, 0.0],
        }
    )
    rows = [
        _fact(value=1.0),
        _fact(value=1.0, constraints=_income_constraints(None, 10_000)),
        _fact(
            concept="census_acs.person_count",
            entity="person",
            value=2.0,
            constraints=_age_constraints(18, 65),
        ),
    ]
    facts = load_sld_target_facts(_write_facts(tmp_path, rows))
    build = build_sld_district_problems(
        households,
        persons,
        base_weights=np.array([10.0, 10.0]),
        facts=facts,
        area_type="sldu",
    )
    assert build.gq_marker_present
    assert build.n_group_quarters_rows == 1
    problem = build.problems[0]
    by_metric = dict(zip(problem.target_frame["metric"], problem.matrix, strict=True))
    # Age band counts both persons; household universes exclude the GQ row.
    np.testing.assert_allclose(by_metric["age_18_to_64"], [1.0, 1.0])
    np.testing.assert_allclose(by_metric["households"], [1.0, 0.0])
    np.testing.assert_allclose(by_metric["income_under_10000"], [1.0, 0.0])


def test_asymmetric_fact_surface_is_refused(tmp_path):
    rows = [
        _fact(value=1.0),
        _fact(geo_id="610U900US49002", value=1.0),
        _fact(
            geo_id="610U900US49002",
            value=1.0,
            constraints=_income_constraints(None, 10_000),
        ),
    ]
    facts = load_sld_target_facts(_write_facts(tmp_path, rows))
    households = pd.DataFrame(
        {
            "household_id": [1],
            "state_fips": [49],
            "sld_upper_code": ["001"],
        }
    )
    persons = pd.DataFrame(
        {
            "person_household_id": [1],
            "age": [30.0],
            "employment_income_before_lsr": [1.0],
            "self_employment_income_before_lsr": [0.0],
            "taxable_interest_income": [0.0],
            "social_security_retirement": [0.0],
        }
    )
    with pytest.raises(ValueError, match="asymmetric fact surface"):
        build_sld_district_problems(
            households,
            persons,
            base_weights=np.array([10.0]),
            facts=facts,
            area_type="sldu",
        )


def test_unknown_constraint_variables_are_refused(tmp_path):
    rows = [
        _fact(
            value=1.0,
            constraints=[{"variable": "race", "operator": "==", "value": "white"}],
        )
    ]
    with pytest.raises(ValueError, match="no binding rule"):
        load_sld_target_facts(_write_facts(tmp_path, rows))
