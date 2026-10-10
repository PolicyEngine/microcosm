"""The policyengine-us concept mapping's US-specific expectations.

The mapping is data importable without the engine; these tests pin what it
says about US inputs. Whether each input exists in the installed engine is
checked in engine_contract/us and engine_scenario/us.
"""

import numpy as np
import pandas as pd
import pytest

from microcosm.frame.adapters.policyengine_us import (
    POLICYENGINE_US_CONCEPT_MAPPING,
    PolicyEngineUSEngine,
)
from microcosm.frame.concept_mapping import ConceptMappedEngine, coverage_report

MAPPING = POLICYENGINE_US_CONCEPT_MAPPING


def test_the_adapter_exposes_its_mapping_without_the_engine() -> None:
    adapter = PolicyEngineUSEngine()
    assert isinstance(adapter, ConceptMappedEngine)
    assert adapter.concept_mapping() is MAPPING
    assert MAPPING.engine == "policyengine-us"


def test_formula_owned_aggregates_are_never_targets() -> None:
    targets = {binding.engine_input for binding in MAPPING.bindings}
    assert not targets & {
        "employment_income",
        "self_employment_income",
        "interest_income",
        "dividend_income",
        "social_security",
        "weeks_worked",
        "rent",
        "is_male",
        "is_tax_unit_head",
    }


def test_every_concept_but_liquid_assets_reaches_the_us_engine() -> None:
    # The engine splits liquid financial assets across three person inputs,
    # and no transform splits one amount three ways.
    assert set(MAPPING.unmapped) == {"fact:person.liquid_financial_assets"}
    reason = MAPPING.unmapped["fact:person.liquid_financial_assets"]
    for name in ("bank_account_assets", "stock_assets", "bond_assets"):
        assert name in reason
    bound = {binding.engine_input for binding in MAPPING.bindings}
    assert not bound & {"bank_account_assets", "stock_assets", "bond_assets"}


def test_the_liquid_asset_reason_describes_each_donor_half() -> None:
    # The donor gives each household its whole vector from the SCF or the
    # SIPP. The halves differ in unit and in term deposits, and neither holds
    # each owner's share, so the reason must not say the columns hold a
    # household total on one member.
    reason = MAPPING.unmapped["fact:person.liquid_financial_assets"]
    for claim in (
        "SCF-sourced households carry a draw of an SCF family-unit total",
        "SIPP-sourced households carry a draw from a model trained on "
        "individual SIPP holdings",
        "includes certificates of deposit",
        "Neither half holds each owner's share",
    ):
        assert claim in reason, claim
    assert "household total on one member" not in reason


def test_the_donor_receipt_concept_feeds_social_security_retirement() -> None:
    assert [
        binding.engine_input
        for binding in MAPPING.bindings_for("fact:person.public_pension_income")
    ] == ["social_security_retirement"]


def test_the_reference_person_places_housing_costs() -> None:
    report = coverage_report(MAPPING, MAPPING.refs())
    fed = {
        ref.name for ref in report.concept_inputs["fact:household.reference_person_id"]
    }
    assert {"is_household_head", "pre_subsidy_rent", "real_estate_taxes"} <= fed
    assert "mortgage_payments" in fed


@pytest.mark.parametrize(
    ("enrollment", "full_time", "full_time_college", "part_time_college"),
    [
        ("tertiary", True, True, False),
        ("tertiary", False, False, True),
        ("upper_secondary", True, False, False),
        ("not_enrolled", False, False, False),
    ],
)
def test_college_flags_read_tertiary_enrollment(
    enrollment, full_time, full_time_college, part_time_college
) -> None:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": [1],
                "person_household_id": [1],
                "education_enrollment": [enrollment],
                "enrolled_full_time": [full_time],
            }
        ),
        "household": pd.DataFrame({"household_id": [1]}),
    }
    out = MAPPING.encode(tables).tables["person"]
    assert out.loc[0, "is_full_time_college_student"] == full_time_college
    assert out.loc[0, "is_part_time_college_student"] == part_time_college


def test_take_up_share_tracks_the_rate() -> None:
    rng = np.random.default_rng(7)
    n = 20_000
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.arange(n),
                "person_household_id": np.arange(n),
                "take_up_seed": rng.random(n),
            }
        ),
        "household": pd.DataFrame({"household_id": np.arange(n)}),
    }
    rates = dict.fromkeys(MAPPING.take_up_programs(), 0.5)
    rates.update({"us.medicaid": 0.3, "us.ssi": 0.7})
    flags = MAPPING.encode(tables, take_up_rates=rates).tables["person"]
    assert abs(flags["takes_up_medicaid_if_eligible"].mean() - 0.3) < 0.015
    assert abs(flags["takes_up_ssi_if_eligible"].mean() - 0.7) < 0.015
