from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.frs_brma import FRS_BRMA_OUTPUT_COLUMNS
from microcosm.build.uk_runtime.frs_household_draws import (
    FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS,
)
from microcosm.build.uk_runtime.frs_take_up import (
    FRS_TAKE_UP_OUTPUT_COLUMNS,
    uk_take_up_signal_gate,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame


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
            "age": [5, 6] * 5,
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(201, 201 + n)})
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
    result = uk_take_up_signal_gate(_frame(), contract=_Contract())

    assert result.passed is True
    assert "benunit.would_claim_child_benefit" in result.details


def test_take_up_gate_constant_column_fails() -> None:
    frame = _frame()
    frame.table("benunit")["would_claim_uc"] = True

    result = uk_take_up_signal_gate(frame, contract=_Contract())

    assert result.passed is False
    assert "constant column" in " ".join(result.failures)


def test_take_up_gate_out_of_band_share_fails() -> None:
    frame = _frame()
    frame.table("household")["property_purchased"] = [True] * 9 + [False]

    result = uk_take_up_signal_gate(frame, contract=_Contract())

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
