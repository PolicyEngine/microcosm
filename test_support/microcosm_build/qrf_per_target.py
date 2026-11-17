"""Fixtures for the per-target QRF fit-and-draw tests of two callers.

Small inputs for the CPS-ORG wage imputation and the ACS input transfer, with
the seeds and tree counts the tests use. See
:mod:`test_support.microcosm_build.qrf_chain_doubles` for the doubles and the
lifetime and byte checks.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from test_support.microcosm_build.us_org_wages import _donor, _frame, _person

# ---------------------------------------------------------------------------
# CPS-ORG fixture
# ---------------------------------------------------------------------------

#: Seed and tree count of the CPS-ORG per-target fixture.
ORG_FIXTURE_SEED = 11
ORG_FIXTURE_N_ESTIMATORS = 6


def org_wage_fixture() -> tuple[Frame, pd.DataFrame]:
    """A recipient frame and donor for the CPS-ORG wage imputation.

    The donor's ``hourly_wage`` is positive for every row (one forest, no
    sign classifier) and its ``is_paid_hourly`` is 0 or 1 (a sign classifier
    and a positive forest). Recipients vary in income, hours and weeks, and
    the first rows have no earnings, so the imputation zeroes them.
    """

    rng = np.random.default_rng(5)
    person = _person(160)
    workers = np.arange(len(person)) >= 24
    person["employment_income_before_lsr"] = np.where(
        workers, np.round(rng.uniform(6_000.0, 160_000.0, len(person)), 2), 0.0
    )
    person["weeks_worked"] = np.where(
        workers, rng.integers(20, 53, len(person)).astype(np.float64), 0.0
    )
    person["weekly_hours_worked_before_lsr"] = np.where(
        workers, rng.integers(8, 61, len(person)).astype(np.float64), 0.0
    )
    person["hours_worked_last_week"] = person["weekly_hours_worked_before_lsr"]
    return _frame(person), _donor(400)


# ---------------------------------------------------------------------------
# ACS transfer fixture
# ---------------------------------------------------------------------------

#: Seed and tree count of the ACS transfer per-target fixture.
ACS_FIXTURE_SEED = 29
ACS_FIXTURE_N_ESTIMATORS = 3

#: Families of the ACS fixture. ``rental_income`` and
#: ``short_term_capital_gains`` take negative, zero and positive donor
#: values; ``qualified_dividend_income`` and ``first_home_mortgage_balance``
#: zero and positive ones; ``taxable_private_pension_income`` positive ones
#: only; the two take-up flags are booleans.
ACS_FIXTURE_FAMILIES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "person": {
        "fixture_income": (
            "rental_income",
            "qualified_dividend_income",
            "takes_up_medicaid_if_eligible",
        ),
        "fixture_gains": (
            "short_term_capital_gains",
            "taxable_private_pension_income",
        ),
    },
    "tax_unit": {
        "fixture_tax_unit": ("takes_up_eitc", "first_home_mortgage_balance"),
    },
}

#: ``(entity, target)`` pairs whose fitted regimes the transfer checks.
ACS_FIXTURE_REGIME_EVIDENCE: tuple[tuple[str, str], ...] = tuple(
    ("person", target) for target in ACS_FIXTURE_FAMILIES["person"]["fixture_income"]
)

_ACS_SCHEMA = EntitySchema(group_entities=("household", "tax_unit"))


def _three_sign(
    rng: np.random.Generator, size: int, *, zero: float, negative: float
) -> np.ndarray:
    draw = rng.random(size)
    amount = np.round(rng.lognormal(7.0, 1.2, size), 2)
    return np.where(draw < zero, 0.0, np.where(draw < zero + negative, -amount, amount))


def _acs_people(
    rng: np.random.Generator, households: int, *, first_id: int
) -> pd.DataFrame:
    size = 2 * households
    household_ids = first_id + np.repeat(np.arange(households), 2)
    head = np.tile([True, False], households)
    age = np.where(head, rng.integers(20, 86, size), rng.integers(0, 86, size))
    employment = np.where(
        rng.random(size) < 0.35, 0.0, np.round(rng.lognormal(10.3, 0.8, size), 2)
    )
    self_employment = _three_sign(rng, size, zero=0.75, negative=0.05)
    return pd.DataFrame(
        {
            "person_id": np.arange(first_id, first_id + size, dtype=np.int64),
            "person_household_id": household_ids.astype(np.int64),
            "person_tax_unit_id": (household_ids + 50_000).astype(np.int64),
            "age": age.astype(np.float64),
            "is_female": rng.random(size) < 0.5,
            "is_household_head": head,
            "employment_income_before_lsr": employment,
            "self_employment_income_before_lsr": self_employment,
            "taxable_interest_income": np.where(
                rng.random(size) < 0.4, 0.0, np.round(rng.lognormal(5.0, 1.5, size), 2)
            ),
        }
    )


def _acs_group_tables(
    person: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    household_ids = np.unique(person["person_household_id"].to_numpy())
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "state_fips": rng.choice([6, 36, 48], len(household_ids)),
        }
    )
    tax_unit = pd.DataFrame(
        {"tax_unit_id": np.unique(person["person_tax_unit_id"].to_numpy())}
    )
    return household, tax_unit


def acs_transfer_fixture() -> tuple[Frame, Frame]:
    """A recipient and donor for the ACS transfer: ``(recipient, donor)``.

    The donor has 80 households of two people, one tax unit each, and every
    family target. The 24 recipient households lack the targets, and some of
    their people lack employment, self-employment or interest income, so
    each family's recipients fall into several availability patterns.
    """

    rng = np.random.default_rng(41)
    donor_person = _acs_people(rng, 80, first_id=1_000)
    size = len(donor_person)
    donor_person["rental_income"] = _three_sign(rng, size, zero=0.55, negative=0.15)
    donor_person["qualified_dividend_income"] = np.where(
        rng.random(size) < 0.5, 0.0, np.round(rng.lognormal(6.0, 1.3, size), 2)
    )
    donor_person["takes_up_medicaid_if_eligible"] = rng.random(size) < 0.6
    donor_person["short_term_capital_gains"] = _three_sign(
        rng, size, zero=0.6, negative=0.2
    )
    donor_person["taxable_private_pension_income"] = np.round(
        rng.lognormal(8.5, 0.9, size), 2
    )
    donor_household, donor_tax_unit = _acs_group_tables(donor_person, rng)
    units = len(donor_tax_unit)
    donor_tax_unit["takes_up_eitc"] = rng.random(units) < 0.3
    donor_tax_unit["first_home_mortgage_balance"] = np.where(
        rng.random(units) < 0.45, 0.0, np.round(rng.lognormal(11.5, 0.7, units), 2)
    )
    donor = Frame(
        {"person": donor_person, "household": donor_household, "tax_unit": donor_tax_unit},
        _ACS_SCHEMA,
        {
            "household": Weights(
                np.round(rng.uniform(50.0, 300.0, len(donor_household)), 2),
                WeightKind.CALIBRATED,
            )
        },
        pd.Series("asec_puf", index=donor_person.index, dtype=object),
    )

    recipient_person = _acs_people(rng, 24, first_id=9_000)
    rows = len(recipient_person)
    for column, share in (
        ("employment_income_before_lsr", 0.25),
        ("self_employment_income_before_lsr", 0.2),
        ("taxable_interest_income", 0.3),
    ):
        recipient_person.loc[rng.random(rows) < share, column] = np.nan
    recipient_household, recipient_tax_unit = _acs_group_tables(recipient_person, rng)
    recipient = Frame(
        {
            "person": recipient_person,
            "household": recipient_household,
            "tax_unit": recipient_tax_unit,
        },
        _ACS_SCHEMA,
        {
            "household": Weights(
                np.round(rng.uniform(40.0, 90.0, len(recipient_household)), 2),
                WeightKind.DESIGN,
            )
        },
        pd.Series("acs_2024_1yr", index=recipient_person.index, dtype=object),
    )
    return recipient, donor


__all__ = [name for name in globals() if not name.startswith("__")]
