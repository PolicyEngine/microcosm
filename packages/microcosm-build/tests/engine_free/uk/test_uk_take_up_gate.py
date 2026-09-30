from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.frs_brma import FRS_BRMA_OUTPUT_COLUMNS
from microcosm.build.uk_runtime.frs_household_draws import (
    FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS,
)
from microcosm.build.uk_runtime.frs_take_up import (
    FRS_TAKE_UP_OUTPUT_COLUMNS,
    UK_TAKE_UP_ENGINE_PREDICTORS,
    uk_take_up_signal_gate,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from test_support.microcosm_build.uk_take_up_population import WorkingAgeStubEngine


class _Contract:
    def rate(self, key: str, build_year: int | None = None) -> float:
        rates = {
            "child_benefit": 0.5,
            "child_benefit_opts_out_rate": 0.5,
            "pension_credit": 0.5,
            "universal_credit": 0.5,
            "tax_free_childcare": 0.5,
            "extended_childcare": 0.5,
            "universal_childcare": 0.5,
            "targeted_childcare": 0.5,
            "uc_childcare_single": 0.5,
            "uc_childcare_couple": 0.5,
            "marriage_allowance": 0.5,
            "scp_under_6": 0.5,
            "scp_6_plus": 0.5,
            "tv_ownership_rate": 0.5,
            "tv_licence_evasion_rate": 0.5,
            "first_time_buyer_rate": 0.5,
            "property_purchase_rate": 0.5,
        }
        return rates[key]


def _frame(*, brma_values=("LONDON_A", "LONDON_B")):
    n = 10
    household_ids = np.arange(1, n + 1)
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 101 + n),
            "person_benunit_id": np.arange(201, 201 + n),
            "person_household_id": household_ids,
            "age": [30, 40] * 5,
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": np.arange(201, 201 + n), "is_married": [False, True] * 5}
    )
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": np.ones(n),
            "brma": [brma_values[index % len(brma_values)] for index in range(n)],
        }
    )
    alternating = np.array([True, False] * 5)
    for column in FRS_TAKE_UP_OUTPUT_COLUMNS:
        if column != "maximum_extended_childcare_hours_usage":
            benunit[column] = alternating
    person["would_claim_marriage_allowance"] = alternating
    person["would_claim_scp"] = alternating
    person["attends_private_school_random_draw"] = np.linspace(0.05, 0.95, n)
    for column in FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS:
        household[column] = alternating
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


def test_take_up_gate_seeded_fixture_passes() -> None:
    result = uk_take_up_signal_gate(
        _frame(), contract=_Contract(), engine=WorkingAgeStubEngine()
    )

    assert result.passed is True
    assert "benunit.would_claim_child_benefit" in result.details


def test_take_up_gate_constant_column_fails() -> None:
    frame = _frame()
    frame.table("benunit")["would_claim_uc"] = True

    result = uk_take_up_signal_gate(
        frame, contract=_Contract(), engine=WorkingAgeStubEngine()
    )

    assert result.passed is False
    assert "constant column" in " ".join(result.failures)


def test_take_up_gate_out_of_band_share_fails() -> None:
    frame = _frame()
    frame.table("household")["property_purchased"] = [True] * 9 + [False]

    result = uk_take_up_signal_gate(
        frame, contract=_Contract(), engine=WorkingAgeStubEngine()
    )

    assert result.passed is False
    assert "property_purchased" in " ".join(result.failures)


def test_brma_enum_domain_binding_fails_off_domain() -> None:
    binding = UK_GATE_REGISTRY["enum_domain"]
    result = binding.evaluate(
        EvidenceContext(
            frame=_frame(brma_values=("LONDON_A", "OFF_DOMAIN")),
            artifacts={"brma_enum_domain": ("LONDON_A", "LONDON_B")},
        ),
        {"columns": ("brma",)},
    )

    assert result.passed is False
    assert "OFF_DOMAIN" in result.failures[0]


def test_student_loan_enum_domain_binding_resolves_person_column() -> None:
    frame = _frame()
    frame.table("person")["student_loan_plan"] = ["NONE"] * 9 + ["PLAN_4"]
    binding = UK_GATE_REGISTRY["enum_domain"]

    result = binding.evaluate(
        EvidenceContext(
            frame=frame,
            artifacts={
                "student_loan_plan_enum_domain": (
                    "NONE",
                    "PLAN_1",
                    "PLAN_2",
                    "PLAN_5",
                )
            },
        ),
        {"columns": ("student_loan_plan",)},
    )

    assert result.passed is False
    assert "PLAN_4" in result.failures[0]


def test_gate_registry_vocabulary_round_trip() -> None:
    assert "take_up_signal" in UK_GATE_REGISTRY
    assert "enum_domain" in UK_GATE_REGISTRY
    assert FRS_BRMA_OUTPUT_COLUMNS == ("brma",)


def test_take_up_gate_measures_uc_share_over_units_with_a_working_age_adult() -> None:
    """Units with every adult at or over State Pension age are outside the draw."""

    frame = _frame()
    person = frame.table("person")
    # Persons 101-110 sit one per benefit unit; make the last four benefit
    # units all-pension-age and mark them as claimants: they must not count.
    person.loc[person["person_id"] >= 107, "age"] = 70
    benunit = frame.table("benunit")
    benunit.loc[benunit["benunit_id"] >= 207, "would_claim_uc"] = True

    result = uk_take_up_signal_gate(
        frame, contract=_Contract(), engine=WorkingAgeStubEngine()
    )

    detail = result.details["benunit.would_claim_uc"]
    assert detail["population_units"] == 6
    assert detail["weighted_share"] == 0.5
    assert result.passed is True


def test_take_up_gate_fails_when_the_uc_population_is_empty() -> None:
    frame = _frame()
    frame.table("person")["age"] = 70

    result = uk_take_up_signal_gate(
        frame, contract=_Contract(), engine=WorkingAgeStubEngine()
    )

    assert result.passed is False
    assert "would_claim_uc: no unit in the draw's population" in " ".join(
        result.failures
    )


def test_take_up_gate_reads_the_uc_population_from_the_engine() -> None:
    """The gate asks the engine for is_WA_adult on the frame it measures.

    Auxiliary float columns the SPI channel leaves NaN by design are filled
    on the copy the engine reads, so the engine never refuses the frame.
    """

    frame = _frame()
    person = frame.table("person")
    person["other_investment_income"] = [np.nan, 1.0] * 5
    seen = []

    class _Recording(WorkingAgeStubEngine):
        def materialize(self, frame, variables, period):
            seen.append(frame.table("person")["other_investment_income"].isna().any())
            return super().materialize(frame, variables, period)

    engine = _Recording()
    result = uk_take_up_signal_gate(frame, contract=_Contract(), engine=engine)

    assert engine.calls == [(UK_TAKE_UP_ENGINE_PREDICTORS, "2023")]
    assert seen == [False]
    # The frame the gate measures keeps its NaN: only the engine's copy fills.
    assert person["other_investment_income"].isna().sum() == 5
    population = result.details["universal_credit_population"]
    assert population["variable"] == "is_WA_adult"
    assert population["period"] == "2023"
    assert population["working_age_adults"] == 10


def test_take_up_gate_follows_a_split_state_pension_age_cohort() -> None:
    """Units count by the engine's per-person answer, not by age.

    Every person here is 66; the engine says the first three are under State
    Pension age (as for 66-year-olds born from 6 April 1960 in 2026-27).
    """

    frame = _frame()
    frame.table("person")["age"] = 66
    engine = WorkingAgeStubEngine(working_age_adult=[True] * 3 + [False] * 7)

    result = uk_take_up_signal_gate(frame, contract=_Contract(), engine=engine)

    assert result.details["benunit.would_claim_uc"]["population_units"] == 3


def test_take_up_gate_refuses_to_guess_the_uc_population() -> None:
    with pytest.raises(ValueError, match="reads the Universal Credit population"):
        uk_take_up_signal_gate(_frame(), contract=_Contract())


def test_take_up_binding_passes_the_armed_rules_engine() -> None:
    binding = UK_GATE_REGISTRY["take_up_signal"]
    engine = WorkingAgeStubEngine()

    assert "rules_engine" in binding.artifact_keys
    result = binding.evaluator(
        EvidenceContext(frame=_frame(), artifacts={"rules_engine": engine}),
        {"maximum_share_deviation": 0.05},
    )

    assert engine.calls == [(UK_TAKE_UP_ENGINE_PREDICTORS, "2023")]
    assert "benunit.would_claim_uc" in result.details
