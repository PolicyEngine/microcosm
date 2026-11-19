"""Fit theta, how a household's adults combine, on the LCFS 2024-25 donor.

Usage, from the tree: python theta_calibration.py <LCFS 2024-25 tab directory> <out.json>

A household plays when any adult does:
  q(theta) = 1 - (1 - p_max) * prod_{other adults} (1 - p_i) ** theta,
with each adult's p_i the committed top_up_gambling_participation step's
four-week rate for its sex and age band, read from the vendored GSGB rows
through the committed declaration. Theta 0 is one player per household, theta
1 adults who play independently. The diary under-records gambling, but if it
misses the same share of one- and two-adult households, the ratio of their
diary rates carries the household effect, so theta is set to give it. The
three-or-more-adult ratio is the check.

Writes aggregates only: weighted shares of households by number of adults,
each resting on 30 or more donors, which the script checks (microcosm#1359).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.gambling_participation import (
    REGULAR,
    gambling_participation_operation,
    household_participation,
    participation_rates,
)

KEYS = {
    "national_lottery": ("c94316", "c94319"),
    "betting_and_gaming": ("c94311", "c94312", "c94314"),
}
SUPPORT_FLOOR = 30


def main() -> None:
    lcfs, out = Path(sys.argv[1]), Path(sys.argv[2])
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    parameters = gambling_participation_operation(stage)
    rates = participation_rates(parameters)
    codes = sorted({code for lines in KEYS.values() for code in lines})
    household = pd.read_csv(
        lcfs / "dvhh_uk_anon_v3_2024.tab",
        sep="\t",
        usecols=["case", "cal_weight_annual", *codes],
    )
    household = household.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    raw = pd.read_csv(
        lcfs / "dvper_uk_anon_2024.tab", sep="\t", usecols=["case", "a004", "a005p"]
    )
    raw = raw.apply(pd.to_numeric, errors="coerce")
    ids = household["case"].to_numpy()
    person = pd.DataFrame(
        {
            "person_household_id": raw["case"].to_numpy(),
            "age": raw["a005p"].to_numpy(dtype=float),
            "gender": np.where(raw["a004"] == 1, "MALE", "FEMALE"),
        }
    )
    person = person[person["person_household_id"].isin(set(ids))]
    weight = household["cal_weight_annual"].to_numpy(dtype=float)
    adults = person[person["age"] >= 18].groupby("person_household_id").size()
    size = adults.reindex(ids).fillna(0).to_numpy()
    sizes = {"1": size == 1, "2": size == 2, "3+": size >= 3}
    report: dict = {"declared_theta": {}, "fitted": {}}
    for key, lines in KEYS.items():
        played = sum(household[code].clip(lower=0).to_numpy() for code in lines) > 0
        for name, mask in sizes.items():
            if (
                int(mask.sum()) < SUPPORT_FLOOR
                or int((mask & played).sum()) < SUPPORT_FLOOR
            ):
                raise SystemExit(
                    f"{key}: {name}-adult households rest on fewer than {SUPPORT_FLOOR} donors."
                )
        observed = {
            n: float(weight[m & played].sum() / weight[m].sum())
            for n, m in sizes.items()
        }
        target = observed["2"] / observed["1"]

        def predicted(theta: float) -> dict[str, float]:
            q = household_participation(
                person, ids, rates.rates[key][REGULAR], bands=rates.bands, theta=theta
            )
            return {
                n: float(weight[m] @ q[m] / weight[m].sum()) for n, m in sizes.items()
            }

        def ratio(shares: dict[str, float]) -> float:
            return shares["2"] / shares["1"]

        low, high = 0.0, 1.0
        at_low, at_high = predicted(low), predicted(high)
        if not ratio(at_low) <= target <= ratio(at_high):
            theta = low if target < ratio(at_low) else high
        else:
            for _ in range(40):
                middle = (low + high) / 2
                low, high = (
                    (middle, high)
                    if ratio(predicted(middle)) < target
                    else (low, middle)
                )
            theta = (low + high) / 2
        fitted = predicted(theta)
        report["declared_theta"][key] = parameters["keys"][key]["theta"]
        report["fitted"][key] = {
            "theta": round(theta, 3),
            "two_over_one": {
                "diary": round(target, 3),
                "theta_0": round(ratio(at_low), 3),
                "theta_1": round(ratio(at_high), 3),
                "fitted": round(ratio(fitted), 3),
            },
            "three_plus_over_one": {
                "diary": round(observed["3+"] / observed["1"], 3),
                "fitted": round(fitted["3+"] / fitted["1"], 3),
            },
            # The diary's household rate over GSGB's at the fitted theta: equal
            # for one and two adults by construction, a check for three or more.
            "diary_over_gsgb_four_week_rate": {
                n: round(observed[n] / fitted[n], 3) for n in sizes
            },
        }
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report["fitted"], indent=1))


if __name__ == "__main__":
    main()
