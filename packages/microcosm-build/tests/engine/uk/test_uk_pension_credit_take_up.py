"""Engine reads behind the Pension Credit take-up stage (microcosm#1095)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

#: The full new State Pension in 2024-25, a week.
_NEW_STATE_PENSION_2024 = 221.20


def _frame():
    """A single pensioner on the full new State Pension, and a mixed-age
    couple whose younger member cares and reports Income Support."""

    person = pd.DataFrame(
        {
            "person_id": [11, 21, 22],
            "person_benunit_id": [1, 2, 2],
            "person_household_id": [1, 2, 2],
            "age": [70.0, 72.0, 58.0],
            "state_pension_reported": [_NEW_STATE_PENSION_2024 * 52, 0.0, 0.0],
            "income_support_reported": [0.0, 0.0, 3_000.0],
            "carers_allowance_reported": [0.0, 0.0, 4_000.0],
            "is_carer_for_benefits": [False, False, True],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1, 2], "would_claim_pc": [True, True]})
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["WALES", "WALES"],
            "council_tax": [0.0, 0.0],
            "tenure_type": ["OWNED_OUTRIGHT", "OWNED_OUTRIGHT"],
            "rent": [0.0, 0.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(2),
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )


def test_entitlement_is_read_on_the_frame_the_engine_projects() -> None:
    # The stage reads calibration-year entitlement on a survey-year frame
    # (uk-data#510). Read as 2025 inputs, a 2024-25 full new State Pension sits
    # under the 2025-26 standard minimum guarantee and the pensioner is
    # entitled. Stamped with its survey year, the engine uprates the pension
    # as it does on the released file, and the pensioner is not.
    engine = PolicyEngineUKEngine()
    names = ["state_pension", "guarantee_credit"]

    relabelled = engine.materialize(_frame(), names, 2025)
    projected = engine.materialize(_frame(), names, 2025, dataset_period=2024)

    survey_pension = _NEW_STATE_PENSION_2024 * 52
    assert relabelled["state_pension"][0] == np.float32(survey_pension)
    assert relabelled["guarantee_credit"][0] > 0
    assert projected["state_pension"][0] > survey_pension * 1.03
    assert projected["guarantee_credit"][0] == 0


def test_a_dataset_period_after_the_period_is_refused() -> None:
    engine = PolicyEngineUKEngine()

    try:
        engine.materialize(_frame(), ["state_pension"], 2024, dataset_period=2025)
    except ValueError as error:
        assert "projects a dataset forward only" in str(error)
    else:
        raise AssertionError("a later dataset period must be refused")


def test_drawing_pension_credit_take_up_leaves_income_support_in_place() -> None:
    # policyengine-uk bars Income Support where a unit meets the Pension
    # Credit age conditions and would claim Pension Credit, with no means
    # test. The stage draws would_claim_pc for units without entitlement. A
    # mixed-age couple that reports Income Support and no Pension Credit has
    # no SI 2019/37 saving, so it does not meet those conditions and the draw
    # cannot take its Income Support away.
    engine = PolicyEngineUKEngine()

    values = engine.materialize(
        _frame(),
        [
            "is_mixed_age_couple",
            "has_mixed_age_couple_pension_credit_saving",
            "meets_pension_credit_age_conditions",
            "income_support_eligible",
        ],
        2025,
        dataset_period=2024,
    )

    assert bool(values["is_mixed_age_couple"][1])
    assert not bool(values["has_mixed_age_couple_pension_credit_saving"][1])
    assert not bool(values["meets_pension_credit_age_conditions"][1])
    assert bool(values["income_support_eligible"][1])
