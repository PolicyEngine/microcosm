"""Counts and shares of licensed records for stage receipts (microcosm#1164).

Stage receipts are build output and can reach a release manifest, so a count
of survey records follows UKDS EUL disclosure control (CD137 clause 8,
CD171 section 5.2.1): nothing from one or two records, and no exact count
that, beside a published sample size or another receipt, narrows a small
group by difference. A count a gate must read exactly, such as the zero a
check requires, stays a count of zero. A share follows the same rule: an
unweighted share is floored to two decimals, since an exact one gives back
its count, and a weighted share carried by one or two records is "<3".

Three is CD171's absolute floor. The build used ten, CD171's working minimum
against secondary disclosure across outputs, until María's ruling of
2026-10-29 moved every output to three (``UK_DISCLOSURE_FLOOR``). No study the
build reads sets a higher one. The UK Data Service catalogue entries were read for conditions beyond
the End User Licence (microcosm#1278): the FRS 2024-25 (SN 9563) has none
(checked 2026-10-16), and neither have the LCFS, ETB, WAS and NTS studies
(checked 2026-10-14). The SPI 2022-23 Public Use Tape (SN 9422, checked
2026-10-16) adds two: no attempt may be made to attribute a record to a named
individual, and material that includes analyses of the tape goes to HMRC at
least ten working days before it is distributed or published. The build
marks the outputs that rest on the tape
(:mod:`microcosm.build.uk_runtime.spi_dependency`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

#: The lineage a stage receipt counts distinct support by: support copies
#: repeat their source household, so a cell is counted in sources, not rows.
SOURCE_HOUSEHOLD_COLUMN = "household_source_id"
#: CD171's absolute floor in distinct source households: a figure resting on
#: fewer is withheld, with any figure that gives it back by difference
#: (María's ruling of 2026-10-29 moved every output from ten to it).
UK_DISCLOSURE_FLOOR = 3
#: How a withheld figure reads.
UK_WITHHELD = f"<{UK_DISCLOSURE_FLOOR}"
#: The band a banded count prints from the floor up to ten.
UK_BELOW_TEN_BAND = f"{UK_DISCLOSURE_FLOOR}-9"


def source_household_ids(household: Any, *, stage: str) -> np.ndarray:
    """Every household row's source household, for a receipt's support counts.

    A table without the column is refused, and so is one with a null in it: a
    null drops out of a distinct count or counts as one source, so a cell
    would read fewer sources than it rests on.
    """

    if SOURCE_HOUSEHOLD_COLUMN not in household.columns:
        raise ValueError(
            f"{stage} counts support by {SOURCE_HOUSEHOLD_COLUMN!r}, which the "
            "frame does not carry."
        )
    ids = household[SOURCE_HOUSEHOLD_COLUMN]
    if ids.isna().any():
        raise ValueError(
            f"{stage} counts support by {SOURCE_HOUSEHOLD_COLUMN!r}, which is null "
            "on some rows: every row needs its source household."
        )
    return ids.to_numpy()


def distinct_count(
    source_ids: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None
) -> int:
    """The distinct source households behind a cell, by the receipts' one rule.

    The cell's rows are counted as :func:`distinct_carriers` counts them: given
    ``weights``, a row of zero weight carries nothing (microcosm#1262).
    """

    selected = np.asarray(mask, dtype=bool)
    ids = np.asarray(source_ids)[selected]
    if weights is None:
        return distinct_carriers(ids)
    return distinct_carriers(ids, np.asarray(weights, dtype=float)[selected])


def rounded_count(count: int) -> int | str:
    """A count of donor records, rounded.

    Zero exact, one or two "<3", three to nine exact, then to the nearest ten
    below a thousand and to the nearest hundred from a thousand, so that a
    count near a survey's published sample size does not give back by
    difference how many records a step left out. Halves round up.
    """

    value = int(count)
    if value < 0:
        raise ValueError(f"a record count cannot be negative: {value}.")
    if value == 0:
        return 0
    if value < UK_DISCLOSURE_FLOOR:
        return UK_WITHHELD
    if value < 10:
        return value
    if value < 1000:
        return (value + 5) // 10 * 10
    return (value + 50) // 100 * 100


def extreme_count(count: int) -> int | str:
    """A count of records picked out by an extreme value, in bands.

    Zero exact, then "<3", "3-9", "10-49" and "50-99", and from a hundred up
    the count to the nearest hundred: how many donors had a negative spend, a
    rate above its bound or VAT above their spending says something about the
    tail a nearby count would not.
    """

    value = int(count)
    if value < 0:
        raise ValueError(f"a record count cannot be negative: {value}.")
    if value == 0:
        return 0
    if value < UK_DISCLOSURE_FLOOR:
        return UK_WITHHELD
    if value < 10:
        return UK_BELOW_TEN_BAND
    if value < 50:
        return "10-49"
    if value < 100:
        return "50-99"
    return (value + 50) // 100 * 100


def support_band(count: int) -> int | str:
    """A cell's survey support as a band, never as a count.

    Zero exact, as the other counts here, then "<3", "3-9", "10-29" and "30+":
    the band a stage receipt gives beside the cell's weighted figures, counted
    on distinct source households.
    """

    value = int(count)
    if value < 0:
        raise ValueError(f"a record count cannot be negative: {value}.")
    if value == 0:
        return 0
    if value < UK_DISCLOSURE_FLOOR:
        return UK_WITHHELD
    if value < 10:
        return UK_BELOW_TEN_BAND
    if value < 30:
        return "10-29"
    return "30+"


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
    """A weighted share of records, or "<3" when one or two records carry it."""

    value = int(count)
    if value < 0:
        raise ValueError(f"a record count cannot be negative: {value}.")
    if 0 < value < UK_DISCLOSURE_FLOOR:
        return UK_WITHHELD
    return float(share)


def distinct_carriers(
    keys: Iterable[object], weights: Iterable[float] | None = None
) -> int:
    """How many source households carry a figure: the receipts' one rule.

    A carrier is a distinct, non-blank source household key on a row of
    positive weight. A row of zero weight adds nothing to a figure, so it
    carries none; this is the rule the build's BRMA withholding counts by
    (microcosm#1262). Without ``weights`` every row counts as weighted.

    A missing key (None, NaN, NA or blank text) carries nothing, and a key
    is read one way however it is spelled, so ``1``, ``1.0`` and ``"1"`` are
    one household: an over-count is the unsafe direction for a floor.
    """

    keys = list(keys)
    if weights is not None:
        weights = [float(weight) for weight in weights]
        if len(weights) != len(keys):
            raise ValueError("carrier keys and weights must align.")
        keys = [key for key, weight in zip(keys, weights, strict=True) if weight > 0]
    spelled = (_carrier_key(key) for key in keys)
    return len({key for key in spelled if key is not None})


def _carrier_key(key: object) -> str | None:
    """A source household key's one spelling, or None for a missing key."""

    if key is None:
        return None
    if isinstance(key, str):
        return key.strip() or None
    if isinstance(key, (bool, np.bool_)):
        return str(bool(key))
    if isinstance(key, (int, np.integer)):
        return str(int(key))
    if isinstance(key, (float, np.floating)):
        value = float(key)
        if math.isnan(value):
            return None
        return str(int(value)) if value.is_integer() else str(value)
    import pandas as pd

    try:
        if pd.isna(key) is True:
            return None
    except (TypeError, ValueError):
        pass
    return str(key).strip() or None


def outside_work_trees(path: Path | str, *, what: str) -> Path:
    """Refuse a licensed output path that git could commit (CD137 clause 8).

    ``what`` names the output. Any ancestor of the resolved path that holds a
    ``.git`` entry, a checkout's directory or a worktree's file, makes the
    path committable: this checkout, another one, or this one spelled in
    another case on a case-insensitive volume. Returns the resolved path.
    """

    resolved = Path(path).expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            raise SystemExit(
                f"{resolved} lies inside the git work tree {parent}; {what} is "
                "derived from licensed microdata and stays with the run, outside "
                "every checkout (UKDS CD137 clause 8)."
            )
    return resolved
