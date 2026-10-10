"""microcosm#1003 receipt Part F: the spine's WAS financial draws against the WAS donor, by age.

The stage-time spine is the candidate H5 without the CGT band donors (added after was_lisa);
the CGT incidence clones stay, because a clone and its original carry identical wealth and LISA
cells and their weights sum to the stage-time weight. The donor is the stage's own cleaned
person donor (non-dependent adults, credibility rule applied). Also records the ownership share
option (c) would imply: the donor's age-group shares applied to the spine's adult age mix.
Aggregates only (weighted medians rounded to 3 significant figures).

    .venv/bin/python -W ignore <this> [spine-lisa]      (from the populace-1003 worktree)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.was_lisa import WASLISAColumns, clean_was_lisa_donor

A = Path("/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/acceptance/1003-lisa")
U = Path("/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/was_2006_22")
LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]
HOUSEHOLD = ("gross_financial_wealth", "savings", "cash_isa", "stocks_and_shares_isa")


def sig3(value: float) -> float:
    return float(f"{value:.3g}")


def wmedian(values, weights) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values, kind="stable")
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, 0.5 * cw[-1])])


def by_group(frame: pd.DataFrame, weight: str) -> dict:
    out = {}
    for label, group in frame.groupby("group", observed=True):
        w = group[weight].to_numpy(float)
        row = {"weighted_adult_share": float(w.sum() / frame[weight].sum())}
        for column in (*HOUSEHOLD, "employment_income"):
            values = group[column].clip(lower=0).to_numpy(float)
            row[f"{column}_weighted_median"] = sig3(wmedian(values, w))
            row[f"{column}_weighted_mean_log1p"] = round(
                float(np.average(np.log1p(values), weights=w)), 2
            )
        row["private_renter_share"] = round(
            float(np.average(group["renter"].to_numpy(float), weights=w)), 3
        )
        out[str(label)] = row
    return out


def main(name: str) -> None:
    stage = load_country_spec("uk").sources.stage_map()["was_lisa"]
    params = {op.kind: dict(op.parameters) for op in stage.operations}
    artifacts = {a["role"]: a for a in stage.artifacts}
    columns = WASLISAColumns.from_parameters(params["clean_was_lisa_donor"])
    donor = clean_was_lisa_donor(
        read_pinned_tab(
            U / "was_round_8_person_eul_may_2025_230525.tab",
            artifacts["was_person_tab"],
            columns=columns.person_raw_columns(),
        ),
        read_pinned_tab(
            U / "was_round_8_hhold_eul_may_2025_230525.tab", artifacts["was_qrf_donor"]
        ),
        columns=columns,
    ).person
    donor["group"] = pd.cut(
        donor["age_floor"], [-1, 24, 34, 44, 54, 200], labels=LABELS
    )
    donor["renter"] = donor["is_private_renter"].astype(float)

    store = pd.HDFStore(A / name / f"{name}.h5", "r")
    person = store["/person"]
    household = store["/household"]
    store.close()
    household = household.loc[
        ~household["household_is_cgt_band_donor"].astype(bool)
    ].set_index("household_id")
    spine = person[person["person_household_id"].isin(household.index)].copy()
    hid = spine["person_household_id"].to_numpy()
    for column in (*HOUSEHOLD, "household_weight"):
        spine[column] = household[column].reindex(hid).to_numpy(float)
    spine["channel"] = (
        household["household_support_channel"].astype(str).reindex(hid).to_numpy()
    )
    spine["renter"] = (
        household["tenure_type"].astype(str).reindex(hid).str.contains("RENT_PRIVATELY")
    ).to_numpy(float)
    spine = spine[spine["age"] >= 18].copy()
    spine["group"] = pd.cut(spine["age"], [17, 24, 34, 44, 54, 200], labels=LABELS)

    result = {
        "spine": name,
        "spine_rows": "candidate H5 without CGT band donors (stage-time population; clones carry identical cells)",
        "donor": by_group(donor, "weight"),
        "spine_all": by_group(spine, "household_weight"),
        "spine_by_channel": {
            str(c): by_group(g, "household_weight") for c, g in spine.groupby("channel")
        },
    }
    donor_share = donor.groupby("group", observed=True).apply(
        lambda g: np.average(g["holds"].astype(float), weights=g["weight"])
    )
    spine_mix = (
        spine.groupby("group", observed=True)["household_weight"].sum()
        / spine["household_weight"].sum()
    )
    result["option_c_implied_adult_share"] = float((donor_share * spine_mix).sum())
    result["donor_age_group_shares"] = {
        str(k): float(v) for k, v in donor_share.items()
    }
    result["spine_adult_age_mix"] = {str(k): float(v) for k, v in spine_mix.items()}
    result["spine_weighted_adults"] = float(spine["household_weight"].sum())
    (A / "receipts").mkdir(exist_ok=True)
    json.dump(
        result,
        open(A / "receipts" / f"financial-wealth-probe-{name}.json", "w"),
        indent=1,
    )
    for key in ("donor", "spine_all"):
        print(key)
        print(
            pd.DataFrame(result[key])
            .T[
                [
                    "weighted_adult_share",
                    "gross_financial_wealth_weighted_median",
                    "gross_financial_wealth_weighted_mean_log1p",
                    "savings_weighted_mean_log1p",
                    "employment_income_weighted_mean_log1p",
                    "private_renter_share",
                ]
            ]
            .to_string()
        )
    for channel, rows in result["spine_by_channel"].items():
        print("channel", channel)
        print(
            pd.DataFrame(rows)
            .T[
                [
                    "weighted_adult_share",
                    "gross_financial_wealth_weighted_median",
                    "gross_financial_wealth_weighted_mean_log1p",
                ]
            ]
            .to_string()
        )
    print(
        "option (c) implied adult share:",
        round(result["option_c_implied_adult_share"] * 100, 3),
        "% on",
        round(result["spine_weighted_adults"] / 1e6, 2),
        "m weighted adults",
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spine-lisa")
