"""Pension contributions that cohere with the pay they come from.

policyengine-uk relieves an employee's pension contributions through the net
pay arrangement, which cannot take more than the pay the contributions are
deducted from, and from 2.134.5 (pe-uk#2240) it counts every contribution, the
employer's included, as pension input for the annual allowance. Two readings
of the spine broke that link:

- the FRS records each job's pension deduction (JOB.DEDUC1) apart from the
  adult's usual earnings (ADULT.INEARNS), so a deduction can exceed the pay
  recorded beside it;
- an SPI-channel row redraws its pay from the tape, while a forest trained on
  FRS rows draws its contributions. The forest reads the drawn pay, so it
  draws whether the row has an employer contribution in line with that pay,
  but it can only draw amounts that FRS rows in the same leaf received, so it
  cannot scale them to pay above the FRS's range.

So employee contributions are capped at pay. An SPI-channel row keeps the
forest's decision on whether it has an employer contribution, and a row that
has one takes its FRS twin's rate, which the ASHE draw of microcosm#1069 set,
times its own pay. The twin's scheme record is not carried over: it describes
the twin's job at the twin's pay, and the row's pay is redrawn, so carrying it
put members at pay where few people are members (María's ruling of 2026-10-21,
after Vahid's review of microcosm#1332). Where the twin has no rate to read,
the forest's amount stays, bounded by ASHE's top employer rate times the row's
pay. Personal (relief-at-source) contributions are left as recorded: the
engine itself limits their relief.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from microcosm.build.uk_runtime.employer_pension_contributions import (
    ASHE_EMPLOYER_RATE_BANDS,
)
from microcosm.build.uk_runtime.receipt_counts import (
    UK_DISCLOSURE_FLOOR,
    UK_WITHHELD,
    distinct_count,
    support_band,
)

#: ASHE's highest employer contribution rate, the top of its open band as the
#: employer draw closes it (``employer_pension_contributions.py``).
ASHE_EMPLOYER_RATE_CEILING = ASHE_EMPLOYER_RATE_BANDS[-1][2]


def cap_net_pay_contributions(
    contributions: np.ndarray, pay: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Employee contributions capped at pay, and the rows the cap changed."""

    recorded = np.asarray(contributions, dtype=np.float64)
    ceiling = np.maximum(np.asarray(pay, dtype=np.float64), 0.0)
    if recorded.shape != ceiling.shape:
        raise ValueError("contributions and pay must have one value per person.")
    capped = np.minimum(recorded, ceiling)
    return capped, capped < recorded


def twin_rate_employer_contributions(
    *,
    drawn: np.ndarray,
    pay: np.ndarray,
    twin_pay: np.ndarray,
    twin_employer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Employer contributions at the FRS twin's rate times the row's own pay.

    The drawn contribution decides membership: a row the forest gave none
    keeps none. A member whose twin has pay and an employer contribution
    takes the twin's rate times its own pay; any other member keeps its drawn
    contribution, bounded by ASHE's top rate times its pay.

    Returns the contributions, the rows that took their twin's rate, and the
    rows whose drawn contribution the ASHE ceiling bounded.
    """

    drawn = np.maximum(np.asarray(drawn, dtype=np.float64), 0.0)
    pay = np.maximum(np.asarray(pay, dtype=np.float64), 0.0)
    twin_pay = np.asarray(twin_pay, dtype=np.float64)
    twin_employer = np.maximum(np.asarray(twin_employer, dtype=np.float64), 0.0)
    if not drawn.shape == pay.shape == twin_pay.shape == twin_employer.shape:
        raise ValueError("employer inputs must have one value per person.")
    has_rate = (twin_pay > 0.0) & (twin_employer > 0.0)
    rate = np.divide(
        twin_employer, twin_pay, out=np.zeros_like(twin_pay), where=has_rate
    )
    from_twin = (drawn > 0.0) & has_rate
    bounded = np.minimum(drawn, ASHE_EMPLOYER_RATE_CEILING * pay)
    employer = np.where(from_twin, rate * pay, bounded)
    return employer, from_twin, ~from_twin & (bounded < drawn)


def coherence_receipt(
    *,
    changed: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    weights: np.ndarray,
    source_ids: np.ndarray,
) -> Mapping[str, object]:
    """Banded support and weighted amounts for one coherence step.

    Support is counted on distinct source households (``support_band``), and
    the weighted amounts are withheld as "<3" where one or two of them carry
    the change, so the receipt gives back no small cell.
    """

    mask = np.asarray(changed, dtype=bool)
    weight = np.asarray(weights, dtype=np.float64)
    sources = distinct_count(source_ids, mask, weight)
    small = 0 < sources < UK_DISCLOSURE_FLOOR
    before_mass = float(np.dot(weight[mask], np.asarray(before, dtype=float)[mask]))
    after_mass = float(np.dot(weight[mask], np.asarray(after, dtype=float)[mask]))
    return {
        "support": support_band(sources),
        "weighted_before": UK_WITHHELD if small else before_mass,
        "weighted_after": UK_WITHHELD if small else after_mass,
    }
