"""Who carries the excise inputs: multiples of spending and incidence, frame and donor.

Usage: python excise_incidence.py <head run JSON> <reconciliation JSON> <LCFS dvhh tab> <out.json>

Reads a sanitized stage run (``outputs/stage_dev_run.head-split.json``) and the
national reconciliation, and the LCFS 2024-25 household tab for the donor side.
Writes aggregates only:

- each duty over the spending that shares it out, at prior weights, and the
  share of households that carry it;
- the weighted share of donors, and of frame households, with each parent
  total and each part positive, and each part's rate among households with the
  parent positive, which shows where the frame loses incidence.

Every donor share rests on 30 or more donors; no count, value or extreme of a
donor is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOBACCO = ("c22111", "c22121", "c22131")
OFF_TRADE = ("c21111", "c21211", "c21212", "c21213", "c21214", "c21221", "c21311")
STAKES = ("c94311", "c94312", "c94313", "c94314", "c94315", "c94316", "c94319")
NATIONAL_LOTTERY = ("c94316", "c94319")
BETTING_AND_GAMING = ("c94311", "c94312", "c94314")
#: The smallest donor support a share here may rest on.
SUPPORT_FLOOR = 30


def _donor(path: Path) -> dict[str, float]:
    columns = (*TOBACCO, *OFF_TRADE, *STAKES, "p602", "p609", "cal_weight_annual")
    tab = pd.read_csv(path, sep="\t", usecols=list(columns))
    tab = tab.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weights = tab["cal_weight_annual"].to_numpy(dtype=float)

    def lines(codes) -> np.ndarray:
        return sum(tab[code].clip(lower=0.0).to_numpy(dtype=float) for code in codes)

    masks = {
        "alcohol_and_tobacco": tab["p602"].to_numpy(dtype=float) > 0,
        "tobacco": lines(TOBACCO) > 0,
        "recreation": tab["p609"].to_numpy(dtype=float) > 0,
        "gambling_stakes": lines(STAKES) > 0,
        "national_lottery": lines(NATIONAL_LOTTERY) > 0,
        "other_gambling_stakes": (lines(STAKES) - lines(NATIONAL_LOTTERY)) > 0,
        "betting_and_gaming": lines(BETTING_AND_GAMING) > 0,
    }
    for name, mask in masks.items():
        if int(mask.sum()) < SUPPORT_FLOOR:
            raise ValueError(f"{name} rests on fewer than {SUPPORT_FLOOR} donors.")
    return {
        name: float(weights[mask].sum() / weights.sum()) for name, mask in masks.items()
    }


def main() -> None:
    head = json.loads(Path(sys.argv[1]).read_text())
    reconciliation = json.loads(Path(sys.argv[2]).read_text())
    donor = _donor(Path(sys.argv[3]))
    components = head["excise_quantities"]["components"]
    positive = head["positive_shares"]
    duty = reconciliation["duties"]
    spending = {
        "alcohol": components["off_trade_alcohol_spending"]["weighted_total"]
        + components["on_trade_alcohol_spending"]["weighted_total"],
        "tobacco": components["tobacco_spending"]["weighted_total"],
        "national_lottery": components["national_lottery_stakes"]["weighted_total"],
        "betting_and_gaming": components["betting_and_gaming_stakes"]["weighted_total"],
    }
    keys = head["excise_quantities"]["keys"]
    charged = {
        "alcohol": duty["alcohol_duty"],
        "tobacco": duty["tobacco_duty"],
        "national_lottery": duty["lottery_duty"],
        "betting_and_gaming": duty["gambling_duty"] - duty["lottery_duty"],
    }
    bases = {key: 0.0 for key in ("national_lottery", "betting_and_gaming")}
    for output in head["excise_quantities"]["outputs"].values():
        if output["key"] in bases:
            bases[output["key"]] += output["annual_level"]
    multiples = {
        key: {
            "duty": charged[key],
            "spending": spending[key],
            "duty_over_spending": charged[key] / spending[key],
            "carrying_share": keys[key]["positive_share"],
            **(
                {"bases_over_spending": bases[key] / spending[key]}
                if key in bases
                else {}
            ),
        }
        for key in charged
    }
    frame = {
        "alcohol_and_tobacco": positive["alcohol_and_tobacco_consumption"],
        "tobacco": components["tobacco_spending"]["positive_share"],
        "recreation": positive["recreation_consumption"],
        "gambling_stakes": components["gambling_stakes"]["positive_share"],
        "national_lottery": components["national_lottery_stakes"]["positive_share"],
        "other_gambling_stakes": components["other_gambling_stakes"]["positive_share"],
        "betting_and_gaming": components["betting_and_gaming_stakes"]["positive_share"],
    }
    parents = {
        "tobacco": "alcohol_and_tobacco",
        "gambling_stakes": "recreation",
        "national_lottery": "gambling_stakes",
        "other_gambling_stakes": "gambling_stakes",
        "betting_and_gaming": "other_gambling_stakes",
    }
    incidence = {
        part: {
            "parent": parent,
            "donor_positive": donor[part],
            "frame_positive": frame[part],
            "donor_rate_within_parent": donor[part] / donor[parent],
            "frame_rate_within_parent": frame[part] / frame[parent],
        }
        for part, parent in parents.items()
    }
    report = {
        "weights": "frame at prior weights; donor at its survey weights",
        "multiples": multiples,
        "parents_positive": {
            name: {"donor": donor[name], "frame": frame[name]}
            for name in ("alcohol_and_tobacco", "recreation")
        },
        "incidence": incidence,
    }
    Path(sys.argv[4]).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report["multiples"], indent=1))


if __name__ == "__main__":
    main()
