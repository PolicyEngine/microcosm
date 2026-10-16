"""Counts and shares of licensed records for stage receipts (microcosm#1164).

Stage receipts are build output and can reach a release manifest, so a count
of survey records follows UKDS EUL disclosure control (CD137 clause 8,
CD171 section 5.2.1): nothing from fewer than ten records, and no exact count
that, beside a published sample size or another receipt, narrows a small
group by difference. A count a gate must read exactly, such as the zero a
check requires, stays a count of zero. A share follows the same rule: an
unweighted share is floored to two decimals, since an exact one gives back
its count, and a weighted share carried by one to nine records is "<10".

Ten is CD171's working minimum, and no study the build reads sets a higher
one. The UK Data Service catalogue entries were read for conditions beyond
the End User Licence (microcosm#1278): the FRS 2024-25 (SN 9563) has none
(checked 2026-10-16), and neither have the LCFS, ETB, WAS and NTS studies
(checked 2026-10-14). The SPI 2022-23 Public Use Tape (SN 9422, checked
2026-10-16) adds two: no attempt may be made to attribute a record to a named
individual, and material that includes analyses of the tape goes to HMRC at
least ten working days before it is distributed or published.
"""

from __future__ import annotations

import math


def rounded_count(count: int) -> int | str:
    """A count of donor records, rounded.

    Zero exact, one to nine "<10", then to the nearest ten below a thousand
    and to the nearest hundred from a thousand, so that a count near a
    survey's published sample size does not give back by difference how many
    records a step left out. Halves round up.
    """

    value = int(count)
    if value < 0:
        raise ValueError(f"a record count cannot be negative: {value}.")
    if value == 0:
        return 0
    if value < 10:
        return "<10"
    if value < 1000:
        return (value + 5) // 10 * 10
    return (value + 50) // 100 * 100


def extreme_count(count: int) -> int | str:
    """A count of records picked out by an extreme value, in bands.

    Zero exact, then "<10", "10-49" and "50-99", and from a hundred up the
    count to the nearest hundred: how many donors had a negative spend, a rate
    above its bound or VAT above their spending says something about the tail
    a nearby count would not.
    """

    value = int(count)
    if value < 0:
        raise ValueError(f"a record count cannot be negative: {value}.")
    if value == 0:
        return 0
    if value < 10:
        return "<10"
    if value < 50:
        return "10-49"
    if value < 100:
        return "50-99"
    return (value + 50) // 100 * 100


def floored_share(share: float) -> float:
    """An unweighted share of records, floored to two decimals.

    An exact share k/n of a few hundred or thousand records gives back k and
    n, so a receipt records the share only to the hundredth below it: 0.987
    is recorded 0.98, "at least 98 percent".
    """

    value = float(share)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"a share must lie in [0, 1]: {value}.")
    return math.floor(value * 100.0 + 1e-9) / 100.0


def small_cell_share(share: float, count: int) -> float | str:
    """A weighted share of records, or "<10" when one to nine records carry it."""

    value = int(count)
    if value < 0:
        raise ValueError(f"a record count cannot be negative: {value}.")
    if 0 < value < 10:
        return "<10"
    return float(share)
