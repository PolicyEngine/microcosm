"""Copy the #1164 stage-run outputs into the repository as aggregates only.

Usage: python sanitize_outputs.py <measurement dir> <outputs dir> <name> [<name> ...]

The outputs derive from UKDS-licensed microdata (LCFS 2023-24 and 2024-25, the
FRS 2024-25 spine), so they leave the measurement directory only as weighted
totals, shares and counts of ten or more cases (UKDS EUL CD137 clause 8,
CD171 section 5.2.1):

- exact extremes (``donor_min``, ``donor_max``, ``minimum_before``), the
  remainder minima and the base tree's donor-floor ``negative_mass`` (the sum
  over the few rows a floor raised) are dropped;
- a count of one to nine households or rows is written "<10", and the
  weighted quantity those households carry beside it (the share of kWh given
  up to a ceiling, the mass held at a maximum or floored at zero) is dropped
  with it;
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


def _small(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10


def _is_count(key: str) -> bool:
    return "rows" in key or "households" in key


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
            elif _is_count(str(k)) and _small(v):
                result[k] = "<10"
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
