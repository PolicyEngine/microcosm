"""The policyengine-uk concept mapping against the installed engine.

Every mapped input must be a real input on the stated entity with a dtype its
transform can produce; every recode must land in the engine's enum; the
take-up bindings must be exactly the engine's would-claim inputs; the
formula-ownership claims in the mapping's notes and unmapped reasons must
hold; the committed coverage report must match the engine; and concepts must
survive a round trip through an actual UK Microsimulation.
"""

import inspect
import json
from importlib.metadata import version

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.frame.adapters.policyengine_uk import (
    POLICYENGINE_UK_CONCEPT_MAPPING,
    UK_SCHEMA,
    PolicyEngineUKEngine,
)
from microcosm.frame.concept_mapping import (
    Recode,
    TakeUpThreshold,
    coverage_report,
    rules_engine_input_refs,
)
from test_support.microcosm_frame.concept_engine_frames import (
    assert_engine_round_trip,
    expected_input_dtypes,
)
from test_support.microcosm_frame.concept_frames import concept_frames, shares
from test_support.microcosm_frame.concept_mappings import coverage_golden

MAPPING = POLICYENGINE_UK_CONCEPT_MAPPING


@pytest.fixture(scope="module")
def engine() -> PolicyEngineUKEngine:
    return PolicyEngineUKEngine()


def test_mapping_was_reviewed_against_the_installed_engine() -> None:
    assert MAPPING.engine_version == version("policyengine-uk"), (
        "policyengine-uk moved: review the concept mapping against the new "
        "engine, then update engine_version and the coverage report."
    )


def test_every_mapped_input_is_an_engine_input_on_its_entity(engine) -> None:
    inputs = set(engine.variables())
    for binding in MAPPING.bindings:
        assert binding.engine_input in inputs, binding.engine_input
        metadata = engine.variable_metadata(binding.engine_input)
        assert metadata.entity == binding.engine_entity, binding.engine_input


def test_mapped_input_dtypes_fit_their_transforms(engine) -> None:
    for binding in MAPPING.bindings:
        dtype = engine.variable_metadata(binding.engine_input).dtype
        assert dtype in expected_input_dtypes(binding), (binding.engine_input, dtype)


def test_recodes_land_in_the_engine_enums(engine) -> None:
    for binding in MAPPING.bindings:
        if isinstance(binding.transform, Recode):
            members = {
                member.name for member in engine.enum_domain(binding.engine_input)
            }
            images = {value for _, value in binding.transform.pairs}
            assert images <= members, (binding.engine_input, images - members)


def test_take_up_bindings_are_the_engines_would_claim_inputs(engine) -> None:
    would_claim = {
        name for name in engine.variables() if name.startswith("would_claim_")
    }
    bound = {
        binding.engine_input
        for binding in MAPPING.bindings
        if isinstance(binding.transform, TakeUpThreshold)
        and binding.engine_input.startswith("would_claim_")
    }
    assert bound == would_claim


def test_structural_inputs_are_engine_inputs(engine) -> None:
    assert set(MAPPING.structural_inputs) <= set(engine.variables())


def test_the_mapping_notes_formula_ownership_claims_hold(engine) -> None:
    # Every variable the mapping's notes, unmapped reasons or module header
    # call formula-owned (or a loader override rather than an input).
    claimed = {
        "employment_income",
        "capital_gains",
        "state_pension",
        "state_pension_reported",
        "marital_status",
        "is_married",
        "is_benunit_head",
        "is_household_head",
        "weekly_hours",
        "tax_free_savings_income",
    }
    inputs = set(engine.variables())
    for name in claimed:
        engine.variable_metadata(name)  # a real engine variable
        assert name not in inputs, name


def test_the_liquid_asset_reason_holds(engine) -> None:
    # Financial wealth is a household input; the only person-level GBP stock
    # is a debt. uc_reported_capital is a benefit-unit input, unset at
    # -1, that the uc_assessable_capital formula reads in place of a household
    # proxy apportioned by benefit-unit adults.
    inputs = set(engine.variables())
    person_stocks = {
        name
        for name in inputs
        if engine._variable(name).quantity_type == "stock"
        and engine._variable(name).unit == "currency-GBP"
        and engine.variable_metadata(name).entity == "person"
    }
    assert person_stocks == {"student_loan_balance"}
    for name in ("savings", "gross_financial_wealth", "net_financial_wealth"):
        assert name in inputs, name
        assert engine.variable_metadata(name).entity == "household", name
    assert "uc_reported_capital" in inputs
    assert engine.variable_metadata("uc_reported_capital").entity == "benunit"
    assert engine._variable("uc_reported_capital").default_value == -1
    assert "uc_assessable_capital" not in inputs
    source = inspect.getsource(engine._variable("uc_assessable_capital").get_formula())
    assert "uc_reported_capital" in source
    assert 'add(benunit, period, ["is_adult"])' in source
    assert "loan" in engine._variable("student_loan_balance").label.lower()


def test_the_liquid_asset_reason_names_the_household_wealth_inputs(engine) -> None:
    # Shares and funds sit on the household too, in corporate_wealth, which
    # Universal Credit counts as capital alongside savings.
    reason = MAPPING.unmapped["fact:person.liquid_financial_assets"]
    assert engine.variable_metadata("corporate_wealth").entity == "household"
    assert engine._variable("corporate_wealth").quantity_type == "stock"
    assert engine._variable("corporate_wealth").unit == "currency-GBP"
    universal_credit = engine._tax_benefit_system().parameters.gov.dwp.universal_credit
    sources = universal_credit.means_test.capital.sources("2026-01-01")
    assert {"savings", "corporate_wealth"} <= set(sources)
    source = inspect.getsource(engine._variable("uc_assessable_capital").get_formula())
    assert "p.capital.sources" in source
    for name in (
        "savings",
        "corporate_wealth",
        "gross_financial_wealth",
        "net_financial_wealth",
    ):
        assert name in reason, name
    assert "one person-level GBP stock input is student_loan_balance" in reason


def test_weekly_hours_divides_annual_hours_by_52(engine) -> None:
    formula = engine._variable("weekly_hours").get_formula()
    assert "WEEKS_IN_YEAR" in inspect.getsource(formula)


def test_committed_coverage_report_matches_the_engine(engine) -> None:
    report = coverage_report(MAPPING, rules_engine_input_refs(engine))
    assert report.unknown_inputs == ()
    committed = json.loads(coverage_golden("policyengine-uk").read_text("utf-8"))
    assert committed == json.loads(json.dumps(report.to_dict())), (
        "Regenerate with: uv run --no-sync python tools/refresh_concept_coverage.py "
        "--engine policyengine-uk"
    )


@settings(
    max_examples=4,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(tables=concept_frames(min_households=2, max_households=4), data=st.data())
def test_concepts_round_trip_through_the_engine(engine, tables, data) -> None:
    assert_engine_round_trip(
        MAPPING,
        engine,
        tables,
        UK_SCHEMA,
        2025,
        shares={name: data.draw(shares) for name in MAPPING.share_parameters()},
        take_up_rates={name: data.draw(shares) for name in MAPPING.take_up_programs()},
        context={"household": {"region": "LONDON"}},
    )
