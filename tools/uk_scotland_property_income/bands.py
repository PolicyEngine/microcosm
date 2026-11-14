"""The NSND bands of the two releases (microcosm#1326 A6).

Bands are on individual non-savings, non-dividend income before the personal
allowance, closed below and open above. Release 1 keeps the ten bands the
request names. Release 2's bins are unions of them, each edge the requested
edge nearest a 2023-24 Scottish rate threshold, so the outturn can be summed to
match and each bin mostly faces one marginal rate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PERSONAL_ALLOWANCE_2023 = 12_570

#: Lower edges of the ten requested bands; the last is open.
REQUESTED_EDGES: tuple[int, ...] = (
    12_570,
    15_000,
    20_000,
    25_000,
    30_000,
    35_000,
    40_000,
    50_000,
    65_000,
    90_000,
)

#: Lower edges of the Release 2 bins: starter and basic, intermediate, higher,
#: and higher with top.
INTENTIONAL_EDGES: tuple[int, ...] = (12_570, 25_000, 40_000, 90_000)

#: Where each 2023-24 Scottish rate starts, in NSND income before the
#: allowance: the allowance plus the band's start in taxable income. The top
#: rate starts at 125,140, where the allowance has tapered away.
SCOTTISH_RATE_STARTS_2023: dict[str, int] = {
    "starter": 12_571,
    "basic": 14_733,
    "intermediate": 25_689,
    "higher": 43_663,
    "top": 125_140,
}


def band_labels(edges: Sequence[int]) -> list[str]:
    """``PA–£14,999`` style labels, the last one open."""

    labels = []
    uppers = (*edges[1:], None)
    for lower, upper in zip(edges, uppers, strict=True):
        start = "PA" if lower == PERSONAL_ALLOWANCE_2023 else f"£{lower:,}"
        labels.append(f"{start}+" if upper is None else f"{start}–£{upper - 1:,}")
    return labels


def assign_bands(values: object, edges: Sequence[int]) -> np.ndarray:
    """Each value's band index, or -1 below the first edge."""

    boundaries = np.asarray(edges, dtype=float)
    return (
        np.searchsorted(boundaries, np.asarray(values, dtype=float), side="right") - 1
    )


def coarse_index_of_fine(fine: Sequence[int], coarse: Sequence[int]) -> np.ndarray:
    """The coarse band holding each fine band.

    Refuses unless every coarse band is a union of fine bands, which is what
    lets the outturn be summed to the coarse bins.
    """

    if coarse[0] != fine[0] or not set(coarse) <= set(fine):
        raise ValueError(
            f"Bins {list(coarse)} are not unions of the bands {list(fine)}."
        )
    return assign_bands(fine, coarse)
