"""The ACS local district ESS gate exactly as it stood before #1729.

A verbatim copy of the constants and the two functions
``tools/build_us_acs_local_release.py`` defined at ac7ab99ae (origin/main when
the gate's floors and comparisons moved to the shared
``microcosm.build.area_support``), under their original names, so ``git
diff`` against that commit's tool shows them unchanged. The differential test
holds the tool's gate entry equal to this copy's. Do not edit it; it is the
"before" of a pure move.
"""

from __future__ import annotations

import numpy as np

#: Finalize's district ESS gate. A congressional district collapses when
#: calibration leaves its Kish ESS below this share of its Kish ESS at the
#: design weights. At the release's ACS share (0.5), microcosm#1078's solves
#: (full surface and two holdout folds) leave 15-24 districts below a quarter
#: unpenalized, and none with the chi-square design-weight penalty.
DISTRICT_ESS_RELATIVE_FLOOR = 0.25
#: The absolute floor on a district's calibrated Kish ESS (0 turns it off).
#: Those solves' smallest district ESS is 7.8-11.7 unpenalized and at least
#: 18.5 penalized.
DISTRICT_ESS_FLOOR = 15.0
DISTRICT_ESS_GATE = "district_ess_collapse"


def _district_ess(weight_summary) -> dict[str, float] | None:
    """A weight summary's per-district Kish ESS as finite, nonnegative floats.

    None when the summary records none, or any value is not such a number.
    """

    by_district = (
        weight_summary.get("effective_sample_size_by_district")
        if isinstance(weight_summary, dict)
        else None
    )
    if not isinstance(by_district, dict) or not by_district:
        return None
    values = {}
    for district, ess in by_district.items():
        if isinstance(ess, bool) or not isinstance(ess, (int, float)):
            return None
        try:
            value = float(ess)
        except OverflowError:  # a JSON integer too large for a float
            return None
        if not np.isfinite(value) or value < 0:
            return None
        values[str(district)] = value
    return values


def district_ess_collapse_gate(
    weight_origin: dict | None,
    *,
    relative_floor: float = DISTRICT_ESS_RELATIVE_FLOOR,
    absolute_floor: float = DISTRICT_ESS_FLOOR,
    blocking: bool = False,
) -> dict:
    """The district ESS gate entry, from the calibration summary's weight origin.

    Reads ``effective_sample_size_by_district`` at the design and the
    calibrated weights. A district collapses when its calibrated Kish ESS is
    below ``relative_floor`` times its design-weight Kish ESS, and is below
    the floor when its calibrated Kish ESS is below ``absolute_floor``; a
    floor of 0 turns its check off. ``criteria_met`` says whether no district
    does either (None when the evidence is missing or malformed).

    The release contract publishes only gates that pass, so a report-only
    entry passes whatever it measured. Under ``blocking`` the entry passes
    only when the criteria are met.
    """

    if not 0.0 <= relative_floor <= 1.0:
        raise ValueError(f"relative_floor must be in [0, 1], got {relative_floor!r}.")
    if not (np.isfinite(absolute_floor) and absolute_floor >= 0.0):
        raise ValueError(
            f"absolute_floor must be finite and >= 0, got {absolute_floor!r}."
        )
    origin = weight_origin if isinstance(weight_origin, dict) else {}
    design = _district_ess(origin.get("design"))
    calibrated = _district_ess(origin.get("calibrated"))
    gate = {
        "blocking": bool(blocking),
        "report_only": not blocking,
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
        "note": (
            "A congressional district collapses when its calibrated Kish ESS "
            "is below relative_floor x its design-weight Kish ESS, and is "
            "below the floor when its calibrated Kish ESS is below "
            "absolute_floor (0 turns a check off); from "
            "calibration_summary.json weight_origin."
        ),
    }
    failures = []
    if design is None or calibrated is None:
        failures.append(
            "calibration_summary.json weight_origin records no finite, "
            "nonnegative effective_sample_size_by_district at both the design "
            "and the calibrated weights"
        )
    elif set(design) != set(calibrated):
        failures.append(
            "the design and calibrated per-district ESS cover different districts"
        )
    if failures:
        return {
            **gate,
            "passed": not blocking,
            "criteria_met": None,
            "failures": failures,
        }

    rows = [
        {
            "district": district,
            "calibrated_ess": calibrated[district],
            "design_ess": design[district],
            "ratio": (
                calibrated[district] / design[district]
                if design[district] > 0
                else None
            ),
        }
        for district in sorted(design)
    ]
    collapsed = sorted(
        (
            row
            for row in rows
            if row["calibrated_ess"] < relative_floor * row["design_ess"]
        ),
        key=lambda row: (row["ratio"], row["district"]),
    )
    below_floor = sorted(
        (row for row in rows if row["calibrated_ess"] < absolute_floor),
        key=lambda row: (row["calibrated_ess"], row["district"]),
    )
    smallest = min(rows, key=lambda row: (row["calibrated_ess"], row["district"]))
    rated = [row for row in rows if row["ratio"] is not None]
    lowest = (
        min(rated, key=lambda row: (row["ratio"], row["district"])) if rated else None
    )
    criteria_met = not collapsed and not below_floor
    return {
        **gate,
        "passed": criteria_met or not blocking,
        "criteria_met": criteria_met,
        "failures": [],
        "n_districts": len(rows),
        "n_collapsed": len(collapsed),
        "n_below_floor": len(below_floor),
        "min_calibrated_ess": smallest["calibrated_ess"],
        "min_calibrated_ess_district": smallest["district"],
        "min_ess_ratio": lowest["ratio"] if lowest else None,
        "min_ess_ratio_district": lowest["district"] if lowest else None,
        "collapsed": collapsed,
        "below_floor": below_floor,
    }
