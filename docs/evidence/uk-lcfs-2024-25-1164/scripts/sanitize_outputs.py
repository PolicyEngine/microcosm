"""Copy the #1164 stage-run outputs into the repository as aggregates only.

Usage: python sanitize_outputs.py <measurement dir> <outputs dir> <name> [<name> ...]

The outputs derive from UKDS-licensed microdata (LCFS 2023-24 and 2024-25, the
FRS 2024-25 spine), so they leave the measurement directory only as weighted
totals, shares and counts of ten or more cases (UKDS EUL CD137 clause 8,
CD171 section 5.2.1):

- exact extremes (``donor_min``, ``donor_max``, ``minimum_before``), the
  remainder minima and the base tree's donor-floor ``negative_mass`` (the sum
  over the few rows a floor raised) are dropped;
- every count of households or rows is rounded: one to nine is written
  "<10" (and the weighted quantity those households carry beside it, the
  share of kWh given up to a ceiling or the mass held at a maximum or floored
  at zero, is dropped with it), counts under a thousand go to the nearest ten
  and larger ones to the nearest hundred, so no exact count can be set beside
  a published sample size; a count of rows picked out by an extreme value (a
  cap, a clip, a floor, a rate above its bound) is banded further,
  "10-49", "50-99", then to the nearest hundred (Vahid's review of
  microcosm#1260);
- the energy-rake fit keeps each margin's largest deviation, not its cells;
- household dumps (``*.households.parquet``) are never copied;
- the ``spine`` field keeps only the file's name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DROPPED = (
    "donor_min",
    "donor_max",
    "minimum_before",
    "minimum_remainder",
    "negative_mass",
)
#: A small count suppresses the weighted quantities its households carry.
PAIRED = {
    "households_capped": ("capped_share_of_total",),
    "rows_held_at_maximum": ("weighted_held_mass",),
    "rows_floored": ("weighted_floored_mass",),
}
PATH_KEYS = ("spine",)


#: Counts of rows picked out by an extreme value: banded, not just rounded.
EXTREME_COUNTS = (
    "rows_capped_at_upper_bound",
    "rows_dropped_no_expenditure_after_vat",
    "rows_raised",
    "households_capped",
    "households_at_ceiling",
    "rows_held_at_maximum",
    "clipped_high_rows",
    "clipped_low_rows",
    "rows_floored",
    "households_above",
    "households_below",
)


def _rounded(key: str, value: int) -> int | str:
    if value < 10:
        return "<10"
    if key in EXTREME_COUNTS:
        if value < 50:
            return "10-49"
        if value < 100:
            return "50-99"
        return (value + 50) // 100 * 100
    if value < 1000:
        return (value + 5) // 10 * 10
    return (value + 50) // 100 * 100


def _small(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10


#: Counts of households whose names say neither "rows" nor "households".
OTHER_COUNTS = ("flagged_zero_before", "flagged_zero_after", "unflagged_with_fuel")


def _is_count(key: str) -> bool:
    return "rows" in key or "households" in key or key in OTHER_COUNTS


def _strip(value, key: str = ""):
    if isinstance(value, dict):
        dropped = set(DROPPED)
        for count, quantities in PAIRED.items():
            if _small(value.get(count)):
                dropped.update(quantities)
        result = {}
        for k, v in value.items():
            if k in dropped:
                continue
            if k in PATH_KEYS and isinstance(v, str):
                result[k] = Path(v).name
            elif k == "energy_rake_fit" and isinstance(v, dict):
                result[k] = {
                    margin: {
                        "max_abs_relative_deviation": block.get(
                            "max_abs_relative_deviation"
                        )
                    }
                    for margin, block in v.items()
                }
            elif (
                _is_count(str(k))
                and isinstance(v, int)
                and not isinstance(v, bool)
                and v > 0
            ):
                result[k] = _rounded(str(k), v)
            else:
                result[k] = _strip(v, str(k))
        return result
    if isinstance(value, list):
        return [_strip(item, key) for item in value]
    return value


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[3:]:
        if name.endswith(".parquet"):
            raise SystemExit(f"{name}: household dumps never leave {source}.")
        payload = _strip(json.loads((source / name).read_text(encoding="utf-8")))
        (target / name).write_text(
            json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n",
            encoding="utf-8",
        )
        print("wrote", name)


if __name__ == "__main__":
    main()
