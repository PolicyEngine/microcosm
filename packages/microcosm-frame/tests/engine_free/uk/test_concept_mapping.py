"""The policyengine-uk concept mapping's UK-specific expectations.

The mapping is data importable without the engine; these tests pin what it
says about UK inputs. Whether each input exists in the installed engine is
checked in engine/uk.
"""

from dataclasses import replace

import pandas as pd
import pytest

from microcosm.frame.adapters.policyengine_uk import (
    POLICYENGINE_UK_CONCEPT_MAPPING,
    PolicyEngineUKEngine,
)
from microcosm.frame.concept_mapping import (
    ConceptMappedEngine,
    GroupRule,
    Identity,
    bind,
)
from microcosm.frame.concepts import AlignmentRelation

MAPPING = POLICYENGINE_UK_CONCEPT_MAPPING


def test_the_adapter_exposes_its_mapping_without_the_engine() -> None:
    adapter = PolicyEngineUKEngine()
    assert isinstance(adapter, ConceptMappedEngine)
    assert adapter.concept_mapping() is MAPPING
    assert MAPPING.engine == "policyengine-uk"


def test_formula_owned_variables_are_never_targets() -> None:
    targets = {binding.engine_input for binding in MAPPING.bindings}
    assert not targets & {
        "employment_income",
        "capital_gains",
        "state_pension_reported",
        "is_household_head",
        "marital_status",
        "current_education",
    }


def test_the_state_pension_has_no_input_route() -> None:
    assert "fact:person.public_pension_income" in MAPPING.unmapped


@pytest.mark.parametrize(
    ("group_rule", "error"),
    [
        (None, "its transform produces 'person' values"),
        (GroupRule.SUM_OVER_MEMBERS, "needs a group rule exactly when"),
    ],
)
def test_no_transform_puts_a_person_amount_on_a_household_input(
    group_rule, error
) -> None:
    # The liquid-asset reason rests on this: binding the person concept to
    # the household savings input is rejected with or without a group rule.
    liquid_assets = "fact:person.liquid_financial_assets"
    assert (
        "no transform puts a person amount on a household input"
        in MAPPING.unmapped[liquid_assets]
    )
    probe = bind(
        "savings",
        "household",
        liquid_assets,
        Identity(),
        AlignmentRelation.APPROXIMATE,
        "probe",
        group_rule=group_rule,
    )
    with pytest.raises(ValueError, match=error):
        replace(
            MAPPING,
            bindings=(*MAPPING.bindings, probe),
            unmapped={
                key: value
                for key, value in MAPPING.unmapped.items()
                if key != liquid_assets
            },
        )


def _person(**columns) -> dict[str, pd.DataFrame]:
    return {
        "person": pd.DataFrame(
            {"person_id": [1], "person_household_id": [1], **columns}
        ),
        "household": pd.DataFrame({"household_id": [1]}),
    }


def test_savings_interest_holds_all_interest_and_isa_interest_a_part() -> None:
    out = MAPPING.encode(
        _person(interest_income=[400.0]),
        shares={"uk.isa_interest_fraction": 0.25},
    ).tables["person"]
    assert out.loc[0, "savings_interest_income"] == 400.0
    assert out.loc[0, "individual_savings_account_interest_income"] == 100.0


def test_self_employment_sums_farm_and_non_farm_profit() -> None:
    out = MAPPING.encode(
        _person(
            nonfarm_self_employment_income=[100.0], farm_self_employment_income=[50.0]
        )
    ).tables["person"]
    assert out.loc[0, "self_employment_income"] == 150.0


def test_annual_hours_are_usual_weekly_hours_times_weeks() -> None:
    out = MAPPING.encode(_person(usual_weekly_hours=[40.0], weeks_worked=[52])).tables[
        "person"
    ]
    assert out.loc[0, "hours_worked"] == 2080.0
