"""National EITC child mix: each option against the other evidence.

1. The other EITC-by-children family on the national_state surface: SOI
   Historic Table 2 TY2022 state rows (51 states), summed to the nation.
2. The congressional-district file's US rows (full surface only; SOI TY2022
   data that Chronicle stamps TY2023, PolicyEngine/chronicle#117).
3. SOI's full-population state data for TY2022 and TY2023 (23in55cmcsv.csv,
   August 2026, not in the pinned feed): is Table 2.5's TY2023 mix (the mix
   the rake keeps) corroborated?

Shares are compared, not levels: the families differ in vintage and level.

Run: python3 -I consistency.py <results>/surface_differential.json <download_dir>
       <out.json>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

DIFF = json.loads(Path(sys.argv[1]).read_text())
HT2 = Path(sys.argv[2])
OUT = Path(sys.argv[3])

SPLITS = (
    "no_qualifying_children",
    "one_qualifying_child",
    "two_qualifying_children",
    "three_or_more_qualifying_children",
)
SHORT = ("childless", "one", "two", "three_plus")
SOI_CODES = ("1", "2", "3", "4")
HT2_MEASURE = {
    "eitc_total": (
        "eitc_no_children_amount",
        "eitc_one_child_amount",
        "eitc_two_children_amount",
        "eitc_three_or_more_children_amount",
    ),
    "eitc_returns": (
        "eitc_no_children_claims",
        "eitc_one_child_claims",
        "eitc_two_children_claims",
        "eitc_three_or_more_children_claims",
    ),
}


def shares(values):
    total = sum(values)
    return [v / total for v in values]


def tvd_pp(a, b):
    return 50 * sum(abs(x - y) for x, y in zip(a, b, strict=True))


# (1) HT2 TY2022 state rows on the surface, summed.
ht2_sum = DIFF["ht2_state_eitc_national_sums"]
# (2) CD US rows (as compiled).
cd = {k: v["cd_us"] for k, v in DIFF["cd_us_vs_t25"].items()}
cd_map = {
    "eitc_total": [
        cd[f"{p}_amount"]
        for p in (
            "eitc_no_children",
            "eitc_one_child",
            "eitc_two_children",
            "eitc_three_or_more_children",
        )
    ],
    "eitc_returns": [
        cd[f"{p}_claims"]
        for p in (
            "eitc_no_children",
            "eitc_one_child",
            "eitc_two_children",
            "eitc_three_or_more_children",
        )
    ],
}


# (3) Full-population state data, US row.
def soi_state(year: str):
    def num(x):
        return float(x.replace(",", "").strip() or 0)

    rows = list(csv.DictReader((HT2 / f"{year}in55cmcsv.csv").open(encoding="latin-1")))
    us = next(
        r for r in rows if r["STATE"].strip() == "US" and r["AGI_STUB"].strip() == "0"
    )
    return {
        "eitc_returns": [num(us[f"N5966{c}"]) for c in SOI_CODES],
        "eitc_total": [num(us[f"A5966{c}"]) * 1000 for c in SOI_CODES],
    }


soi = {"2022": soi_state("22"), "2023": soi_state("23")}

result = {}
for measure in ("eitc_total", "eitc_returns"):
    per = DIFF["options"]
    opt_shares = {}
    for name in "abcd":
        ps = per[name]["per_split"]
        bins = [ps[f"{measure}|{s}"]["bins_sum"] for s in SPLITS]
        opt_shares[f"{name}_bins"] = shares(bins)
        if ps[f"{measure}|{SPLITS[0]}"]["total_row"] is not None:
            opt_shares[f"{name}_total_rows"] = shares(
                [ps[f"{measure}|{s}"]["total_row"] for s in SPLITS]
            )
    others = {
        "ht2_ty2022_states_on_surface": shares(
            [ht2_sum[m] for m in HT2_MEASURE[measure]]
        ),
        "cd_ty2022_us_on_full": shares(cd_map[measure]),
        "soi_state_full_population_ty2022": shares(soi["2022"][measure]),
        "soi_state_full_population_ty2023": shares(soi["2023"][measure]),
    }
    rec = {
        "option_shares": {
            k: dict(zip(SHORT, v, strict=True)) for k, v in opt_shares.items()
        },
        "other_shares": {
            k: dict(zip(SHORT, v, strict=True)) for k, v in others.items()
        },
        "tvd_pp": {
            f"{opt}~{other}": tvd_pp(ov, xv)
            for opt, ov in opt_shares.items()
            for other, xv in others.items()
        },
        "ht2_states_on_surface_levels": dict(
            zip(SHORT, [ht2_sum[m] for m in HT2_MEASURE[measure]], strict=True)
        ),
        "soi_state_full_population_levels": {
            y: dict(zip(SHORT, soi[y][measure], strict=True)) for y in soi
        },
    }
    result[measure] = rec
    print(f"\n== {measure}: shares (childless, one, two, three+)")
    for k, v in {**opt_shares, **others}.items():
        print(f"  {k:36s} " + "  ".join(f"{100 * x:6.2f}%" for x in v))
    print("  TVD (pp) of each option from:")
    for other in others:
        print(
            f"    {other:36s} "
            + "  ".join(
                f"{opt}={tvd_pp(opt_shares[opt], others[other]):.2f}"
                for opt in opt_shares
            )
        )
OUT.write_text(json.dumps(result, indent=1))
