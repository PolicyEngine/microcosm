"""ASEC-measured CDCC adult-care inputs (PolicyEngine/populace#451 item 1)."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from populace.build.source_runtime import SourceRuntimeError
from populace.build.us_runtime.adult_care import (
    US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT,
    US_ADULT_CARE_EARNED_INCOME_SOURCES,
    US_ADULT_CARE_OUTPUT_COLUMNS,
    US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS,
    derive_us_adult_care_from_manifest,
    us_adult_care_signal_gate,
    us_adult_care_stage_spec,
    with_us_adult_care_inputs,
)
from populace.build.us_runtime.source_runtime import us_source_operation_handlers
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_FLAG, _EXPENSE = US_ADULT_CARE_OUTPUT_COLUMNS
_DONOR_CHILDCARE = 4_000.0


def _frame() -> Frame:
    """25 people: two statute-eligible units, donor units, and filler singles.

    Unit 201: married head (earned) + spouse with measured PEDISDRS == 1 and
    no earnings — the 21(b)(1)(C) prong whose work test passes only through
    the 21(d)(2) deeming rule.
    Unit 202: single head (earned) + disabled adult dependent — 21(b)(1)(B).
    Unit 203: two working parents + a child under 13 with positive measured
    childcare — the paid-usage donor.
    Unit 204: two working parents + a child under 13 with zero childcare —
    holds the measured usage rate strictly inside (0, 1).
    Units 205+: working singles diluting the weighted flag share into the
    plausibility band.
    """

    rows: list[dict[str, object]] = []

    def _person(
        unit: int,
        role: str,
        *,
        age: float,
        pedisdrs: float = 2.0,
        employment: float = 0.0,
    ) -> None:
        rows.append(
            {
                "person_tax_unit_id": unit,
                "tax_unit_role_input": role,
                "age": age,
                "PEDISDRS": pedisdrs,
                "employment_income_before_lsr": employment,
                "self_employment_income_before_lsr": 0.0,
                "sstb_self_employment_income_before_lsr": 0.0,
            }
        )

    _person(201, "HEAD", age=45, employment=80_000.0)
    _person(201, "SPOUSE", age=44, pedisdrs=1.0)
    _person(202, "HEAD", age=50, employment=40_000.0)
    _person(202, "DEPENDENT", age=25, pedisdrs=1.0)
    _person(203, "HEAD", age=35, employment=60_000.0)
    _person(203, "SPOUSE", age=34, employment=30_000.0)
    _person(203, "DEPENDENT", age=6)
    _person(204, "HEAD", age=36, employment=55_000.0)
    _person(204, "SPOUSE", age=35, employment=25_000.0)
    _person(204, "DEPENDENT", age=4)
    for offset in range(15):
        _person(205 + offset, "HEAD", age=30 + offset, employment=20_000.0)

    person = pd.DataFrame(rows)
    count = len(person)
    person.insert(0, "person_id", np.arange(1, count + 1, dtype="int64"))
    person["person_household_id"] = person["person_tax_unit_id"] + 800
    person["person_spm_unit_id"] = person["person_tax_unit_id"] + 400
    person["person_family_id"] = person["person_tax_unit_id"] + 600
    person["person_marital_unit_id"] = np.arange(701, 701 + count, dtype="int64")

    tax_unit_ids = person["person_tax_unit_id"].drop_duplicates().to_numpy()
    spm_unit_ids = person["person_spm_unit_id"].drop_duplicates().to_numpy()
    household_ids = person["person_household_id"].drop_duplicates().to_numpy()
    childcare = np.where(
        spm_unit_ids == 603,
        _DONOR_CHILDCARE,
        0.0,
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "state_fips": np.full(len(household_ids), 6),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": tax_unit_ids}),
        "spm_unit": pd.DataFrame(
            {
                "spm_unit_id": spm_unit_ids,
                "spm_unit_pre_subsidy_childcare_expenses": childcare,
            }
        ),
        "family": pd.DataFrame(
            {"family_id": person["person_family_id"].drop_duplicates().to_numpy()}
        ),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(household_ids), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def test_stage_manifest_pins_measured_flag_and_documented_donor_proxy() -> None:
    spec = us_adult_care_stage_spec()

    assert spec.stage == "adult_care_inputs"
    assert spec.survey == "Census CPS ASEC"
    assert spec.grain == "person"
    assert spec.outputs == US_ADULT_CARE_OUTPUT_COLUMNS
    assert spec.nonnegative_outputs == (_EXPENSE,)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_adult_care_inputs",
    ]
    assert spec.operations[1].parameters == {
        "self_care_difficulty_source": "PEDISDRS",
        "childcare_expense_source": "spm_unit_pre_subsidy_childcare_expenses",
        "child_qualifying_age_limit": US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT,
        "earned_income_sources": list(US_ADULT_CARE_EARNED_INCOME_SOURCES),
        "seed_from_build_config": True,
        "output_flag": _FLAG,
        "output_expenses": _EXPENSE,
    }
    # The source-decision receipts must stay in the shipped declaration.
    assert "PEDISDRS" in spec.notes
    assert "ESELFCARE" in spec.notes
    assert "TDPCAREAMT" in spec.notes
    assert "former household member" in spec.notes
    assert "21(d)(2)" in spec.notes

    handlers = us_source_operation_handlers()
    assert handlers["derive_adult_care_inputs"] is derive_us_adult_care_from_manifest


def test_flag_is_measured_and_expenses_bind_the_statute_structure() -> None:
    result = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    person = result.table("person")

    # The flag is exactly the measured PEDISDRS == 1 signal.
    assert person[_FLAG].dtype == bool
    np.testing.assert_array_equal(
        person[_FLAG].to_numpy(), person["PEDISDRS"].to_numpy() == 1.0
    )

    # The measured usage rate is 0.5 (one positive donor of two), so the
    # weight-targeted prefix selects exactly one of the two eligible units,
    # and the level draw reproduces the only positive donor value.
    expenses = person[_EXPENSE].to_numpy()
    positive = expenses > 0.0
    assert positive.sum() == 1
    assert expenses[positive][0] == pytest.approx(_DONOR_CHILDCARE)
    carrier = person.loc[positive].iloc[0]
    assert bool(carrier[_FLAG])
    assert int(carrier["person_tax_unit_id"]) in (201, 202)

    gate = us_adult_care_signal_gate(result)
    assert gate.passed, gate.failures


def test_derivation_is_deterministic_per_seed() -> None:
    first = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    second = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    pd.testing.assert_series_equal(
        first.table("person")[_EXPENSE], second.table("person")[_EXPENSE]
    )
    pd.testing.assert_series_equal(
        first.table("person")[_FLAG], second.table("person")[_FLAG]
    )


def test_taxpayer_alone_is_never_their_own_qualifying_individual() -> None:
    frame = _frame()
    person = frame.table("person")
    # Turn the married-spouse carrier into an unmarried disabled head: the
    # spouse prong must stop qualifying, leaving only unit 202 eligible.
    spouse_row = person.index[person["tax_unit_role_input"] == "SPOUSE"][0]
    assert int(person.loc[spouse_row, "person_tax_unit_id"]) == 201
    person.loc[spouse_row, "person_tax_unit_id"] = 299
    person.loc[spouse_row, "tax_unit_role_input"] = "HEAD"
    person.loc[spouse_row, "person_household_id"] = 1_099
    person.loc[spouse_row, "person_spm_unit_id"] = 699
    person.loc[spouse_row, "person_family_id"] = 899
    frame.table("tax_unit").loc[len(frame.table("tax_unit"))] = {"tax_unit_id": 299}
    frame.table("spm_unit").loc[len(frame.table("spm_unit"))] = {
        "spm_unit_id": 699,
        "spm_unit_pre_subsidy_childcare_expenses": 0.0,
    }
    frame.table("family").loc[len(frame.table("family"))] = {"family_id": 899}
    households = frame.table("household")
    households.loc[len(households)] = {"household_id": 1_099, "state_fips": 6}
    rebuilt = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(frame.table("household")), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )

    result = with_us_adult_care_inputs(rebuilt, seed=7, time_period=2024)
    person_out = result.table("person")
    expense_units = set(
        person_out.loc[person_out[_EXPENSE] > 0.0, "person_tax_unit_id"].tolist()
    )
    assert 299 not in expense_units
    assert 201 not in expense_units


def test_fails_closed_without_measured_sources() -> None:
    missing_flag_source = _frame()
    missing_flag_source.table("person").drop(columns=["PEDISDRS"], inplace=True)
    with pytest.raises(ValueError, match="cannot heal"):
        with_us_adult_care_inputs(missing_flag_source, seed=0, time_period=2024)

    missing_donor = _frame()
    missing_donor.table("spm_unit").drop(
        columns=["spm_unit_pre_subsidy_childcare_expenses"], inplace=True
    )
    with pytest.raises(ValueError, match="cannot heal"):
        with_us_adult_care_inputs(missing_donor, seed=0, time_period=2024)


def test_degenerate_measured_usage_rate_fails_closed() -> None:
    frame = _frame()
    spm = frame.table("spm_unit")
    # Making every kid-donor unit a payer collapses the measured usage rate
    # to 1.0, which must refuse rather than silently saturate.
    spm.loc[spm["spm_unit_id"] == 604, "spm_unit_pre_subsidy_childcare_expenses"] = (
        1_000.0
    )
    with pytest.raises(SourceRuntimeError, match="degenerate"):
        with_us_adult_care_inputs(frame, seed=0, time_period=2024)


def test_signal_gate_rejects_missing_default_and_invented_surfaces() -> None:
    valid = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    assert us_adult_care_signal_gate(valid).passed

    missing = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    missing.table("person").drop(columns=[_EXPENSE], inplace=True)
    assert not us_adult_care_signal_gate(missing).passed

    zeroed = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    zeroed.table("person")[_EXPENSE] = 0.0
    gate = us_adult_care_signal_gate(zeroed)
    assert not gate.passed
    assert any("structural zero" in failure for failure in gate.failures)

    unflagged_carrier = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    person = unflagged_carrier.table("person")
    unflagged_row = person.index[~person[_FLAG]][0]
    person.loc[unflagged_row, _EXPENSE] = 500.0
    gate = us_adult_care_signal_gate(unflagged_carrier)
    assert not gate.passed
    assert any("without the qualifying flag" in failure for failure in gate.failures)

    flagless = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    flagless.table("person")[_FLAG] = False
    assert not us_adult_care_signal_gate(flagless).passed


def test_release_frame_without_raw_source_preserves_valid_surface() -> None:
    built = with_us_adult_care_inputs(_frame(), seed=7, time_period=2024)
    built.table("person").drop(columns=["PEDISDRS"], inplace=True)

    healed = with_us_adult_care_inputs(
        built,
        seed=7,
        time_period=2024,
        allow_existing_without_source=True,
    )

    pd.testing.assert_frame_equal(healed.table("person"), built.table("person"))


def test_required_source_columns_are_the_measured_pair() -> None:
    assert US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS == (
        "PEDISDRS",
        "spm_unit_pre_subsidy_childcare_expenses",
    )


@requires_us
def test_policyengine_1_764_6_cdcc_adult_care_contract_and_live_binding() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "1.764.6"
    variables = CountryTaxBenefitSystem().variables
    flag = variables[_FLAG]
    assert flag.is_input_variable()
    assert flag.entity.key == "person"
    assert flag.value_type is bool
    expense = variables[_EXPENSE]
    assert expense.is_input_variable()
    assert expense.entity.key == "person"
    assert expense.value_type is float

    situation = {
        "people": {
            "head": {
                "age": {"2024": 45},
                "employment_income": {"2024": 80_000},
            },
            "spouse": {
                "age": {"2024": 44},
                _FLAG: {"2024": True},
                _EXPENSE: {"2024": 3_000},
            },
        },
        "tax_units": {
            "tax_unit": {
                "members": ["head", "spouse"],
                "filing_status": {"2024": "JOINT"},
            }
        },
        "spm_units": {"spm_unit": {"members": ["head", "spouse"]}},
        "households": {
            "household": {
                "members": ["head", "spouse"],
                "state_code": {"2024": "TX"},
            }
        },
    }

    baseline = Simulation(situation=situation)
    # 21(b)(1)(C) + 21(d)(2): the incapacitated spouse is the qualifying
    # individual and is deemed the earnings floor, so the credit binds.
    assert baseline.calculate("count_cdcc_eligible", 2024)[0] == 1
    assert baseline.calculate("cdcc_relevant_expenses", 2024)[0] == pytest.approx(
        3_000.0
    )
    assert baseline.calculate("cdcc", 2024)[0] > 0.0

    class NeutralizeAdultCare(Reform):
        def apply(self) -> None:
            self.neutralize_variable(_EXPENSE)

    neutralized = Simulation(situation=situation, reform=NeutralizeAdultCare)
    assert neutralized.calculate("cdcc_relevant_expenses", 2024)[0] == 0.0
    assert neutralized.calculate("cdcc", 2024)[0] == 0.0
    assert (
        neutralized.calculate("income_tax", 2024)[0]
        > baseline.calculate("income_tax", 2024)[0]
    )


def test_shipped_cdcc_adult_care_probe() -> None:
    from populace.build.us_runtime.release_input_coverage import (
        us_release_input_coverage_required_columns,
        us_release_reform_coverage_probes,
    )

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "cdcc_adult_care_expense_neutralization"
    )
    assert probe.neutralized_variable == _EXPENSE
    assert probe.binding_inputs == US_ADULT_CARE_OUTPUT_COLUMNS
    assert probe.budget_measure == "income_tax"
    assert probe.period == 2024
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "negative"
    assert probe.min_abs_effect >= 1_000_000.0
    assert "PolicyEngine/populace#451" in probe.issue

    required = us_release_input_coverage_required_columns()
    for column in US_ADULT_CARE_OUTPUT_COLUMNS:
        assert column in required
