"""Property tests for the proposed PUF-half wage gross-up (microcosm#1191).

Run from the repository root:
    uv run pytest docs/evidence/us-puf-wage-concept-1191/scripts/test_gross_up_properties.py

These test ``gross_up_reference.py``, a standalone statement of the proposal's
semantics. They do not exercise production imputation or the live build, and
the repository's ``testpaths = ["packages"]`` keeps them out of ordinary pytest
collection; tests for an implementation belong in the package test suites.

The last two tests state invariants 1 and 3 for today's behaviour in the same
reference and are expected to fail. They are marked xfail(strict=True) to
record the counterexamples; they do not monitor the build.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# The repository runs pytest in importlib mode, which leaves this folder off
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gross_up_reference import (
    allocate_unit_total,
    current_stored_wage,
    elective_deferral_limit,
    engine_pre_tax,
    float32_round_trip_bound,
    irs_employment_income,
    stored_gross_wage,
)

dollars = st.floats(min_value=0.0, max_value=5e7, allow_nan=False, allow_infinity=False)
small = st.floats(min_value=0.0, max_value=2e5, allow_nan=False, allow_infinity=False)
ages = st.integers(min_value=0, max_value=95)
# Wage bases are whole cents, as stored wages are; a subnormal basis would
# underflow the proportional split.
cents = st.integers(min_value=0, max_value=5_000_000_000).map(lambda c: c / 100)
member = st.tuples(cents, small, small, ages)
SETTINGS = settings(max_examples=2000, deadline=None)


def pre_tax_for(donor_wage, traditional, roth, age, premiums=0.0, hsa=0.0):
    """Engine pre-tax amount, with the build's rule of no deferral without wages.

    retirement_contributions.py lines 404-412 zero the 401(k) leaves where the
    wage is zero; the proposal extends that to every pre-tax payroll item.
    """

    if donor_wage <= 0:
        return 0.0
    return float(
        engine_pre_tax(
            traditional, roth, age, pre_tax_health_premiums=premiums, hsa_payroll=hsa
        )
    )


def taxed(donor_wage, pre_tax, store=stored_gross_wage):
    gross = float(store(donor_wage, pre_tax))
    return float(irs_employment_income(gross, pre_tax)), gross


# Invariant 1: the engine returns the carried donor wage, within float32 rounding.
@SETTINGS
@given(dollars, small, small, ages, small, small)
def test_round_trip_returns_the_donor_wage(wage, traditional, roth, age, prem, hsa):
    pre_tax = pre_tax_for(wage, traditional, roth, age, prem, hsa)
    result, gross = taxed(wage, pre_tax)
    assert abs(result - wage) <= float32_round_trip_bound(gross, pre_tax)


# Invariant 2: bounds, and the 401(k) part stays within the age-specific limit.
@SETTINGS
@given(dollars, small, small, ages, small, small)
def test_bounds(wage, traditional, roth, age, prem, hsa):
    pre_tax = pre_tax_for(wage, traditional, roth, age, prem, hsa)
    gross = float(stored_gross_wage(wage, pre_tax))
    assert gross >= wage >= 0
    # float64 rounding of the sum: one unit in the last place of the gross wage.
    assert abs((gross - wage) - pre_tax) <= 2 * float(np.spacing(gross))
    deferral = float(engine_pre_tax(traditional, roth, age))
    assert deferral <= float(elective_deferral_limit(age)) * (1 + 1e-12)


# Invariant 3: the engine's zero floor never binds on the PUF half.
@SETTINGS
@given(dollars, small, small, ages, small, small)
def test_zero_floor_never_binds(wage, traditional, roth, age, prem, hsa):
    pre_tax = pre_tax_for(wage, traditional, roth, age, prem, hsa)
    gross = float(stored_gross_wage(wage, pre_tax))
    assert np.float32(gross) - np.float32(pre_tax) >= np.float32(0.0)


# Invariant 4 (a modeling assumption): a zero donor wage means no employment,
# whatever pre-tax amount an upstream draw produced.
@SETTINGS
@given(small)
def test_zero_donor_wage_stores_zero(any_pre_tax):
    assert float(stored_gross_wage(0.0, any_pre_tax)) == 0.0
    assert pre_tax_for(0.0, any_pre_tax, 0.0, 40) == 0.0


@SETTINGS
@given(dollars, dollars, small, small, ages)
def test_monotone_in_the_donor_wage(wage_a, wage_b, traditional, roth, age):
    low, high = sorted((wage_a, wage_b))
    if low == 0:
        return
    pre_tax = float(engine_pre_tax(traditional, roth, age))
    assert float(stored_gross_wage(low, pre_tax)) <= float(
        stored_gross_wage(high, pre_tax)
    )


# Invariant 5: gross-up preserves the finalized allocation of the QRF total.
# Where the unit has a person aged 15 or over, the taxed wages sum to the
# clipped total; where it has none, the total is not stored (as today).
@SETTINGS
@given(
    st.floats(-1e5, 5e7, allow_nan=False),
    st.lists(member, min_size=1, max_size=6),
    st.floats(0.0, 1e5),
)
def test_unit_total_survives_allocation_and_gross_up(total, members, weight):
    basis = np.array([m[0] for m in members])
    age = np.array([m[3] for m in members])
    allocated = allocate_unit_total(total, basis, age)
    assert (allocated[age < 15] == 0).all()
    results, bound = [], 0.0
    for donor_wage, (_, traditional, roth, person_age) in zip(
        allocated, members, strict=True
    ):
        pre_tax = pre_tax_for(donor_wage, traditional, roth, person_age)
        result, gross = taxed(donor_wage, pre_tax)
        results.append(result)
        bound += float32_round_trip_bound(gross, pre_tax) + 1e-9 * gross
    expected = max(total, 0.0) if (age >= 15).any() else 0.0
    assert abs(weight * sum(results) - weight * expected) <= weight * bound + 1e-6


# Invariant 6: the release re-imputes contributions. Recomputing the stored
# wage from the carried donor wage after any later draw keeps invariant 1;
# recomputing from the previously stored gross wage carries the first draw
# along and misses by it.
@SETTINGS
@given(dollars, small, small, small, small, ages)
def test_reimputation_recomputes_from_the_carried_wage(wage, t1, r1, t2, r2, age):
    first = pre_tax_for(wage, t1, r1, age)
    stored_once = float(stored_gross_wage(wage, first))
    second = pre_tax_for(wage, t2, r2, age)
    result, gross = taxed(wage, second)
    assert abs(result - wage) <= float32_round_trip_bound(gross, second)
    if wage > 0 and first > 1.0:
        wrong, wrong_gross = taxed(stored_once, second)
        assert abs(wrong - (wage + first)) <= float32_round_trip_bound(
            wrong_gross, second
        ) + float32_round_trip_bound(stored_once, first)


@pytest.mark.xfail(
    strict=True,
    reason="Today's PUF-half wage is the donor wage, so the engine subtracts a second time.",
)
@SETTINGS
@given(dollars, small, small, ages)
def test_today_round_trip(wage, traditional, roth, age):
    pre_tax = pre_tax_for(wage, traditional, roth, age)
    result, stored = taxed(wage, pre_tax, store=current_stored_wage)
    assert abs(result - wage) <= float32_round_trip_bound(stored, pre_tax)


@pytest.mark.xfail(
    strict=True,
    reason="Today a deferral above the donor wage is lost to the engine's zero floor.",
)
@SETTINGS
@given(dollars, small, small, ages)
def test_today_zero_floor_never_binds(wage, traditional, roth, age):
    pre_tax = pre_tax_for(wage, traditional, roth, age)
    stored = float(current_stored_wage(wage, pre_tax))
    assert np.float32(stored) - np.float32(pre_tax) >= np.float32(0.0)
