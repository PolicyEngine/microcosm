"""Synthetic Child Benefit frames and engine inputs shared across test lanes."""

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.child_benefit_take_up import UKChildBenefitStatistics
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind


def _statistics(rates: dict[int, float] | None = None) -> UKChildBenefitStatistics:
    return UKChildBenefitStatistics(
        claim_rates=rates or {0: 0.7, 1: 0.8, 2: 0.9},
        all_ages_claim_rate=0.8,
        families_registered=1_000.0,
        families_in_payment=900.0,
        families_opted_out=100.0,
        children_in_payment=1_500.0,
        children_opted_out=150.0,
    )


class _StubEngine:
    country = "uk"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def materialize(self, frame, variables, period):
        self.calls.append((tuple(variables), str(period)))
        person = frame.table("person")
        return {
            "is_child_or_qualifying_young_person_for_child_benefit": (
                person["age"].to_numpy() < 16
            ),
            "adjusted_net_income": person["stub_income"].to_numpy(dtype=float),
        }


def _frame():
    # Unit 1: a reporter with a child. Unit 2: fully charged, with two
    # children. Unit 3: a low-income family. Unit 4: no child, reports.
    person = pd.DataFrame(
        {
            "person_id": [11, 12, 21, 22, 23, 31, 32, 41],
            "person_benunit_id": [1, 1, 2, 2, 2, 3, 3, 4],
            "person_household_id": [1, 1, 2, 2, 2, 3, 3, 4],
            "age": [40, 2, 45, 1, 2, 30, 0, 50],
            "child_benefit_reported": [1_300.0, 0, 0, 0, 0, 0, 0, 900.0],
            "stub_income": [20_000.0, 0, 95_000.0, 0, 0, 15_000.0, 0, 10_000.0],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [1, 2, 3, 4],
            "would_claim_child_benefit": [False, False, True, True],
            "child_benefit_opts_out": [True, False, True, True],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2, 3, 4], "region": ["WALES"] * 4})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([10.0, 10.0, 10.0, 10.0]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _taper_frame():
    """A scarce full-charge pool, a taper family and an independent nonclaimant."""
    frame = _frame()
    person = frame.table("person").copy()
    person.loc[person.person_id == 31, "stub_income"] = 70_000.0
    person = pd.concat(
        [
            person,
            pd.DataFrame(
                {
                    "person_id": [51, 52],
                    "person_benunit_id": [5, 5],
                    "person_household_id": [5, 5],
                    "age": [40, 3],
                    "child_benefit_reported": [0.0, 0.0],
                    "stub_income": [95_000.0, 0.0],
                }
            ),
        ],
        ignore_index=True,
    )
    benunit = pd.concat(
        [
            frame.table("benunit"),
            pd.DataFrame(
                {
                    "benunit_id": [5],
                    "would_claim_child_benefit": [False],
                    "child_benefit_opts_out": [True],
                }
            ),
        ],
        ignore_index=True,
    )
    household = pd.concat(
        [
            frame.table("household"),
            pd.DataFrame({"household_id": [5], "region": ["WALES"]}),
        ],
        ignore_index=True,
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([10.0, 2.0, 20.0, 10.0, 5.0]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2026",
    )
