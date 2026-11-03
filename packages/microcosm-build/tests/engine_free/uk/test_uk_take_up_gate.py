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
            "extended_childcare_age_0_usage": 0.5,
            "extended_childcare_age_1_usage": 0.5,
            # Conditional on not claiming the extended hours: the fixture's
            # universal flag is its extended flag, so the nested marginal is
            # 0.5 + 0.5 * 0.0.
            "universal_childcare": 0.0,
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
    # Ten adults, one per unit, and four children under 2: two in units that
    # claim the working-parent hours (201 and 203), two in units that do not.
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 101 + n + 4),
            "person_benunit_id": [*range(201, 201 + n), 201, 203, 202, 204],
            "person_household_id": [*household_ids, 1, 3, 2, 4],
            "age": [30, 40] * 5 + [0, 1, 0, 1],
            "months_since_last_birthday": [0.0] * (n + 4),
            "is_uc_claimant": [True] * n + [False] * 4,
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
    people = np.array([True, False] * 7)
    person["would_claim_marriage_allowance"] = people
    person["would_claim_scp"] = people
    person["attends_private_school_random_draw"] = np.linspace(0.05, 0.95, n + 4)
    # One of the two claiming children uses the hours.
    person["max_free_entitlement_hours_used"] = [30.0] * n + [30.0, 0.0, 30.0, 30.0]
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
    assert "benunit.would_claim_tfc" in result.details
    # The Child Benefit flags are measured by their own stage gate
    # (microcosm#1063), like would_claim_pc (microcosm#1069) and
    # would_claim_uc (microcosm#1334).
    assert "benunit.would_claim_child_benefit" not in result.details
    assert "benunit.child_benefit_opts_out" not in result.details
    assert "benunit.would_claim_uc" not in result.details


def test_take_up_gate_constant_column_fails() -> None:
    frame = _frame()
    frame.table("benunit")["would_claim_tfc"] = True

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
    frame.table("person")["student_loan_plan"] = ["NONE"] * 13 + ["PLAN_4"]
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


def test_take_up_gate_leaves_universal_credit_to_its_stage_gate() -> None:
    """would_claim_uc is set by award band and family type (microcosm#1334).

    Its share of the age population is no longer a contract rate, so even a
    constant flag is not this gate's to fail; uk_stage_uc_take_up measures it.
    """

    frame = _frame()
    frame.table("benunit")["would_claim_uc"] = True

    result = uk_take_up_signal_gate(frame, contract=_Contract())

    assert result.passed is True
    assert "benunit.would_claim_uc" not in result.details


def test_take_up_gate_measures_under_two_usage_over_claiming_children() -> None:
    result = uk_take_up_signal_gate(_frame(), contract=_Contract())

    details = result.details["person.max_free_entitlement_hours_used"]
    # Two children under 2 in claiming units, one of whom uses the hours; the
    # children in units that do not claim keep 30 hours and are not counted.
    assert details["population_units"] == 2
    assert details["weighted_share"] == 0.5
    assert details["target"] == 0.5

    frame = _frame()
    frame.table("person").drop(
        columns=["max_free_entitlement_hours_used"], inplace=True
    )
    missing = uk_take_up_signal_gate(frame, contract=_Contract())
    assert missing.passed is False
    assert "max_free_entitlement_hours_used: missing" in " ".join(missing.failures)


class _OffRateContract(_Contract):
    """The fixture's realised shares are 0.5; these rates sit 0.075 above them."""

    def rate(self, key: str, build_year: int | None = None) -> float:
        if key in (
            "extended_childcare_age_0_usage",
            "extended_childcare_age_1_usage",
            "property_purchase_rate",
        ):
            return 0.575
        return super().rate(key, build_year)


def test_take_up_gate_holds_under_two_usage_to_its_own_bound() -> None:
    usage = "person.max_free_entitlement_hours_used"

    shared = uk_take_up_signal_gate(_frame(), contract=_OffRateContract())
    assert any(usage in failure for failure in shared.failures)
    assert shared.details[usage]["maximum_share_deviation"] == 0.05

    own = uk_take_up_signal_gate(
        _frame(),
        contract=_OffRateContract(),
        under_two_usage_maximum_share_deviation=0.1,
    )
    # The under-2 share is inside its own bound; every other flag keeps the
    # shared one, so the household draw 0.075 off its rate still fails.
    assert not any(usage in failure for failure in own.failures)
    assert own.details[usage]["maximum_share_deviation"] == 0.1
    assert (
        own.details["household.property_purchased"]["maximum_share_deviation"] == 0.05
    )
    assert any("household.property_purchased" in failure for failure in own.failures)


# María's ruling of 2026-10-29 (microcosm#1095): every flag the terminal gate
# measures is checked at its drawing stage too, at the stage's weights.
_SIGNAL_STAGE_GATES = {
    "frs_take_up": "uk_stage_frs_take_up_signal",
    "frs_person_draws": "uk_stage_frs_person_draws_take_up_signal",
    "frs_household_draws": "uk_stage_frs_household_draws_take_up_signal",
    "childcare_copy_take_up": "uk_stage_childcare_copy_take_up_signal",
}


def _stage_signal_outputs() -> dict[str, tuple[tuple[str, str, str], ...]]:
    from microcosm.build.uk_runtime.frs_person_draws import (
        FRS_PERSON_DRAW_OUTPUT_COLUMNS,
    )
    from microcosm.build.uk_runtime.frs_take_up import (
        UKFRSTakeUpStageTransform,
        take_up_signal_outputs,
    )

    return {
        "frs_take_up": take_up_signal_outputs(
            UKFRSTakeUpStageTransform.output_columns()
        ),
        "frs_person_draws": take_up_signal_outputs(FRS_PERSON_DRAW_OUTPUT_COLUMNS),
        "frs_household_draws": take_up_signal_outputs(
            FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS
        ),
    }


def _stage_gate_parameters(stage: str) -> dict[str, object]:
    from microcosm.build import load_country_spec

    entries = {entry.id: entry for entry in load_country_spec("uk").gates.gates}
    entry = entries[_SIGNAL_STAGE_GATES[stage]]
    assert entry.gate == "stage_health"
    # The copy stage runs after the assembled boundary (frs_brma).
    assert entry.phase == (
        "transferred" if stage == "childcare_copy_take_up" else "assembled"
    )
    assert entry.criticality == "release_blocking"
    assert entry.evidence_absent_blocks is True
    return dict(entry.parameters)


def test_each_terminal_flag_is_checked_once_at_its_drawing_stage() -> None:
    from microcosm.build.uk_runtime.frs_take_up import UK_TAKE_UP_SIGNAL_OUTPUTS

    by_stage = _stage_signal_outputs()
    drawn = [row for rows in by_stage.values() for row in rows]
    assert sorted(drawn) == sorted(UK_TAKE_UP_SIGNAL_OUTPUTS)
    assert len(drawn) == len(set(drawn))
    for stage, rows in by_stage.items():
        parameters = _stage_gate_parameters(stage)
        assert parameters["stage"] == stage
        assert parameters["check"] == "take_up_signal"
        # The gate's declared flags are the stage's, in the terminal gate's
        # order, so a receipt that drops one fails closed.
        assert list(parameters["outputs"]) == [
            f"{entity}.{output}" for entity, output, _ in rows
        ]
        # The terminal gate's bounds until a spine receipt measures the stage.
        assert parameters["maximum_share_deviation"] == 0.05
    assert (
        _stage_gate_parameters("frs_take_up")["under_two_usage_maximum_share_deviation"]
        == 0.1
    )


def _receipt(frame, stage: str) -> dict[str, object]:
    from microcosm.build.uk_runtime.frs_take_up import uk_take_up_signal_receipt

    return uk_take_up_signal_receipt(
        frame, outputs=_stage_signal_outputs()[stage], contract=_Contract()
    )


def _stage_gate(stage: str, receipt: dict[str, object]):
    from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

    return uk_stage_health_gate(
        evidence={"stage": stage, "take_up_signal": receipt},
        stage=stage,
        check="take_up_signal",
        parameters=_stage_gate_parameters(stage),
    )


def test_stage_receipt_measures_drawn_flags_and_withholds_thin_populations() -> None:
    receipt = _receipt(_frame(), "frs_take_up")
    rows = {row["output"]: row for row in receipt["flags"]}

    tfc = rows["benunit.would_claim_tfc"]
    # Ten benefit units in ten households carry the flag's population.
    assert tfc["population_support"] == "10-29"
    assert tfc["weighted_share"] == 0.5
    assert tfc["target"] == 0.5
    assert tfc["distinct_values"] == 2
    assert (
        rows["benunit.would_claim_universal_childcare"][
            "extended_without_universal_units"
        ]
        == 0
    )
    usage = rows["person.max_free_entitlement_hours_used"]
    # Two children in two households: below the disclosure floor, so neither
    # the share nor the age-weighted target is given, and the gate flags the
    # flag unchecked rather than failing (María's ruling of 2026-10-26); the
    # terminal gate still measures it at the calibrated weights.
    assert usage["under_two_usage"] is True
    assert usage["weighted_share"] == "<3"
    assert usage["target"] == "<3"
    result = _stage_gate("frs_take_up", receipt)
    assert result.passed is True, result.failures
    assert result.details["flags_withheld_unchecked"] == [
        "person.max_free_entitlement_hours_used"
    ]
    # A share that is neither measured nor withheld still fails.
    broken = {
        **receipt,
        "flags": [
            {**row, "weighted_share": "n/a"}
            if row["output"] == "benunit.would_claim_tfc"
            else row
            for row in receipt["flags"]
        ],
    }
    assert any(
        "not a measured share" in f for f in _stage_gate("frs_take_up", broken).failures
    )


def test_stage_take_up_signal_gate_passes_and_names_each_defect() -> None:
    frame = _frame()
    passed = _stage_gate("frs_household_draws", _receipt(frame, "frs_household_draws"))
    assert passed.passed is True
    assert passed.details["household.property_purchased"]["absolute_deviation"] == 0

    frame.table("household")["property_purchased"] = [True] * 9 + [False]
    off = _stage_gate("frs_household_draws", _receipt(frame, "frs_household_draws"))
    assert off.passed is False
    assert any(
        "household.property_purchased's weighted share 0.9000" in f
        for f in off.failures
    )

    frame = _frame()
    frame.table("person")["would_claim_scp"] = True
    constant = _stage_gate("frs_person_draws", _receipt(frame, "frs_person_draws"))
    assert constant.passed is False
    assert any("would_claim_scp is constant" in f for f in constant.failures)

    receipt = _receipt(_frame(), "frs_household_draws")
    receipt["flags"] = receipt["flags"][1:]
    omitted = _stage_gate("frs_household_draws", receipt)
    assert omitted.passed is False
    assert any("omits the declared flags" in f for f in omitted.failures)


def test_household_draw_stage_records_its_take_up_receipt() -> None:
    from microcosm.build.uk_runtime.frs_household_draws import (
        UKFRSHouseholdDrawsStageTransform,
    )

    stage = UKFRSHouseholdDrawsStageTransform(contract=_Contract(), stage=None)
    with pytest.raises(RuntimeError, match="completed stage run"):
        stage.checkpoint_metadata()
    stage(_frame())
    evidence = stage.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == "frs_household_draws"
    assert [row["output"] for row in evidence["take_up_signal"]["flags"]] == list(
        _stage_gate_parameters("frs_household_draws")["outputs"]
    )


def test_the_copy_stage_checks_the_childcare_flags_it_redraws() -> None:
    """childcare_copy_take_up re-draws five of frs_take_up's flags for the
    copies, so its own stage gate measures them again after the re-draw."""
    from microcosm.build.uk_runtime.childcare_copy_take_up import (
        CHILDCARE_COPY_TAKE_UP_REWRITES,
    )
    from microcosm.build.uk_runtime.frs_take_up import take_up_signal_outputs

    redrawn = take_up_signal_outputs(CHILDCARE_COPY_TAKE_UP_REWRITES)
    assert set(redrawn) <= set(_stage_signal_outputs()["frs_take_up"])
    parameters = _stage_gate_parameters("childcare_copy_take_up")
    assert parameters["stage"] == "childcare_copy_take_up"
    assert parameters["check"] == "take_up_signal"
    assert list(parameters["outputs"]) == [
        f"{entity}.{output}" for entity, output, _ in redrawn
    ]
    assert parameters["maximum_share_deviation"] == 0.05
    assert parameters["under_two_usage_maximum_share_deviation"] == 0.1


def test_the_spine_battery_runs_each_take_up_stage_gate() -> None:
    """Each stage-time take-up gate is in the spine battery's scope and runs
    through the battery's stage_health binding on its stage's receipt."""
    from microcosm.build.uk_runtime.calibration_run import UK_SPINE_GATE_SCOPE

    frame = _frame()
    drawing = ("frs_take_up", "frs_person_draws", "frs_household_draws")
    evidence = {
        stage: {"stage": stage, "take_up_signal": _receipt(frame, stage)}
        for stage in drawing
    }
    binding = UK_GATE_REGISTRY["stage_health"]
    for stage, gate_id in _SIGNAL_STAGE_GATES.items():
        assert gate_id in UK_SPINE_GATE_SCOPE
        parameters = _stage_gate_parameters(stage)
        assert set(parameters) <= binding.parameter_keys
        if stage not in evidence:
            # The copy stage's receipt needs copies: its own test runs it.
            continue
        result = binding.evaluator(
            EvidenceContext(artifacts={"stage_evidence": evidence}), parameters
        )
        assert result.passed, (stage, result.failures)
