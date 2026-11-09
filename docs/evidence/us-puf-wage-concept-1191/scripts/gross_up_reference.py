"""Reference semantics for the proposed PUF-half wage gross-up (microcosm#1191).

Not build code, and not imported by any build. It states what the proposal
means in a small executable form so its invariants can be property-tested
before any build change. It mirrors, without importing:

* ``allocate_unit_total``: how QRF finalization splits a predicted tax-unit
  wage total over the unit's PUF-clone persons
  (``puf_support.py::finalize_us_puf_tax_detail_predictions`` lines 2074-2124
  and ``_write_person_tax_unit_totals`` lines 3782-3830): persons under 15 get
  zero, persons 15 and over share the total in proportion to their copied
  nonnegative wage, and the first person aged 15 or over takes all of it when
  that basis sums to zero.
* ``engine_pre_tax``: what policyengine-us 2.2.1 subtracts in 2024
  (``pre_tax_contributions``): traditional 401(k) and 403(b) desired deferrals
  scaled by ``min(limit / max(total desired, 1), 1)``, where the limit is
  $23,000 plus $7,500 from age 50, plus pre-tax health premiums and HSA payroll
  contributions.
* ``irs_employment_income``: the engine formula, evaluated in float32 as
  policyengine-core stores float variables.
* ``stored_gross_wage``: the proposal. On the PUF half, store the carried
  donor wage plus that pre-tax amount, and treat a zero donor wage as no
  employment (an explicit modeling assumption).
"""

import numpy as np

LIMIT_401K_2024 = 23_000.0
CATCH_UP_401K_2024 = 7_500.0
CATCH_UP_AGE = 50
EARNINGS_MINIMUM_AGE = 15


def elective_deferral_limit(age):
    return LIMIT_401K_2024 + np.where(
        np.asarray(age) >= CATCH_UP_AGE, CATCH_UP_401K_2024, 0.0
    )


def engine_pre_tax(
    traditional_401k,
    roth_401k,
    age,
    traditional_403b=0.0,
    roth_403b=0.0,
    pre_tax_health_premiums=0.0,
    hsa_payroll=0.0,
):
    total = traditional_401k + roth_401k + traditional_403b + roth_403b
    scale = np.minimum(elective_deferral_limit(age) / np.maximum(total, 1.0), 1.0)
    return (
        (traditional_401k + traditional_403b) * scale
        + pre_tax_health_premiums
        + hsa_payroll
    )


def irs_employment_income(employment_income, pre_tax_contributions):
    gross = np.asarray(employment_income, dtype=np.float32)
    pre_tax = np.asarray(pre_tax_contributions, dtype=np.float32)
    return np.maximum(np.float32(0.0), gross - pre_tax)


def allocate_unit_total(total, basis, age):
    """Split one tax unit's clipped wage total over its persons."""

    basis = np.clip(np.asarray(basis, dtype=np.float64), 0.0, None)
    eligible = np.asarray(age) >= EARNINGS_MINIMUM_AGE
    total = max(float(total), 0.0)
    out = np.zeros(len(basis), dtype=np.float64)
    if not eligible.any():
        return out
    weights = np.where(eligible, basis, 0.0)
    if weights.sum() > 0:
        out[eligible] = total * weights[eligible] / weights.sum()
    else:
        out[np.flatnonzero(eligible)[0]] = total
    return out


def stored_gross_wage(donor_wage, pre_tax_contributions):
    """Proposed PUF-half stored wage, computed from the carried donor wage."""

    donor_wage = np.asarray(donor_wage, dtype=np.float64)
    return np.where(donor_wage > 0, donor_wage + pre_tax_contributions, 0.0)


def current_stored_wage(donor_wage, pre_tax_contributions):
    """What builds from main store today: the donor wage itself."""

    return np.asarray(donor_wage, dtype=np.float64)


def float32_round_trip_bound(gross, pre_tax):
    """Absolute bound on the float32 round-trip error.

    Storing the gross wage and the pre-tax amount in float32 rounds each by at
    most half a unit in the last place, and the subtraction rounds its result
    by at most half a unit of the gross wage. One unit of each input bounds
    the sum.
    """

    return float(np.spacing(np.float32(gross)) + np.spacing(np.float32(pre_tax)))
