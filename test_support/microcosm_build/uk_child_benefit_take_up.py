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
        benunit = frame.table("benunit")
        child = person["age"].to_numpy() < 16
        children = (
            pd.Series(child, index=person.index)
            .groupby(person["person_benunit_id"])
            .sum()
        )
        return {
            "is_child_or_qualifying_young_person_for_child_benefit": child,
            "adjusted_net_income": person["stub_income"].to_numpy(dtype=float),
            # A stand-in annual amount per eligible child, at benefit-unit grain.
            "child_benefit_entitlement": (
                benunit["benunit_id"].map(children).fillna(0).to_numpy(float)
                * STUB_ENTITLEMENT_PER_CHILD
            ),
        }


STUB_ENTITLEMENT_PER_CHILD = 1_000.0


def _with_frs_lineage(person, benunit, household):
    """The support-channel lineage of an FRS-only frame: every row its own twin."""

    person = person.assign(
        is_benunit_head=person.groupby("person_benunit_id").cumcount().eq(0),
        person_source_id=person["person_id"],
    )
    benunit = benunit.assign(
        benunit_support_channel="frs", benunit_source_id=benunit["benunit_id"]
    )
    household = household.assign(source_household_id=household["household_id"])
    return person, benunit, household


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
    person, benunit, household = _with_frs_lineage(person, benunit, household)
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
    person, benunit, household = _with_frs_lineage(person, benunit, household)
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([10.0, 2.0, 20.0, 10.0, 5.0]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2026",
    )


def _copy_frame():
    """FRS families and their SPI copies, one household per benefit unit.

    FRS units 1 to 4 are one-child families below the charge start; units 1
    and 2 report. Unit 5 is fully charged with one child and does not report.
    Unit 6 has no child and does not report. Copies: 101 of unit 1 (reporting
    twin, same cell); 102 of unit 3 (non-reporting twin, same cell, but the
    forests gave it a report); 105 of unit 1 moved into the fully charged
    cell; 106 of unit 6, childless, with a forest report of its own.
    """

    frs = [
        # unit, adult income, child, reported on the adult
        (1, 20_000.0, True, 1_300.0),
        (2, 25_000.0, True, 1_100.0),
        (3, 30_000.0, True, 0.0),
        (4, 35_000.0, True, 0.0),
        (5, 95_000.0, True, 0.0),
        (6, 40_000.0, False, 0.0),
    ]
    copies = [
        # copy unit, twin unit, adult income, reported on the adult (forests)
        (101, 1, 21_000.0, 0.0),
        (102, 3, 31_000.0, 900.0),
        (105, 1, 120_000.0, 1_300.0),
        (106, 6, 150_000.0, 700.0),
    ]
    people, units, households = [], [], []
    for unit, income, child, reported in frs:
        people.append(
            (unit * 10 + 1, unit, unit, 40, reported, income, True, unit * 10 + 1)
        )
        if child:
            people.append(
                (unit * 10 + 2, unit, unit, 5, 0.0, 0.0, False, unit * 10 + 2)
            )
        units.append((unit, "frs", unit))
        households.append((unit, unit))
    for unit, twin, income, reported in copies:
        people.append(
            (unit * 10 + 1, unit, unit, 40, reported, income, True, twin * 10 + 1)
        )
        if any(row[0] == twin and row[2] for row in frs):
            people.append(
                (unit * 10 + 2, unit, unit, 5, 0.0, 0.0, False, twin * 10 + 2)
            )
        units.append((unit, "spi", twin))
        households.append((unit, twin))
    person = pd.DataFrame(
        people,
        columns=[
            "person_id",
            "person_benunit_id",
            "person_household_id",
            "age",
            "child_benefit_reported",
            "stub_income",
            "is_benunit_head",
            "person_source_id",
        ],
    )
    benunit = pd.DataFrame(
        units, columns=["benunit_id", "benunit_support_channel", "benunit_source_id"]
    )
    benunit["would_claim_child_benefit"] = False
    benunit["child_benefit_opts_out"] = False
    household = pd.DataFrame(
        households, columns=["household_id", "source_household_id"]
    )
    household["region"] = "WALES"
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.full(len(household), 10.0),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )
