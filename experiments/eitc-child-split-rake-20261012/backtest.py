"""Backtest the EITC child-split options against published Table 2.5.

For each year pair (s -> t), uprate Table 2.5 TY s to TY t the way
``_uprate_cross_period_eitc_decompositions`` does after microcosm#1227, under
each option, and score the result against the published Table 2.5 TY t.

Two margin modes:

* ``fs``: the active AGI margins are SOI's mid-November filing-season Table 1
  for TY t, exactly as production uses the TY2024 one. Realistic.
* ``oracle``: the active margins are TY t Table 2.5 itself, coarsened to the
  same groups. Isolates the child-mix assumption from filing-season noise.

All options conserve the active total, which differs from the TY t Table 2.5
total (the mid-November file is not complete-year), so errors are scored
against the TY t table rescaled to the active total ("level-matched"): the
options are compared on how they distribute the active total, which is the
only thing they differ on.

Run (after ``fetch_sources.py <dir>``):
    python3 -I backtest.py <dir>/tsv <out.json> [<pinned consumer-facts feed>]

With the pinned feed (``consumer_facts_us_51aa40f.jsonl``) it first checks
that the parser reproduces every Table 2.5 and filing-season fact the feed
carries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eitc_core import (  # noqa: E402
    KINDS,
    SPLITS,
    collapse,
    group_margins,
    group_of,
    options,
    parse_filing_season,
    parse_t25,
)

TSV = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FEED = Path(sys.argv[3]) if len(sys.argv) > 3 else None

FS_FILE = {
    2017: "18inweek47expanded.xls",
    2018: "19inweek47expanded.xls",
    2019: "20inweek53.xls",
    2021: "22inweek47.xls",
    2022: "23inweek47.xls",
    2023: "24inweek47.xls",
    2024: "25inweek47.xlsx",
}
T25_YEARS = range(2013, 2024)
# Year pairs with no change to the EITC's structure, only indexation:
# ARPA (TY2021) widened and enlarged the childless credit and TY2022 reverted
# it; TY2020-21 allowed a prior-year earned-income election.
CLEAN = {
    (2013, 2014),
    (2014, 2015),
    (2015, 2016),
    (2016, 2017),
    (2017, 2018),
    (2018, 2019),
    (2022, 2023),
}
POLICY_BREAK = {(2019, 2020), (2020, 2021), (2021, 2022)}

t25 = {y: parse_t25(TSV / f"{y % 100:02d}in25ic.xls.tsv") for y in T25_YEARS}
fs = {y: parse_filing_season(TSV / f"{f}.tsv") for y, f in FS_FILE.items()}
BINS = t25[2023][0]
for y, (bins, _, _) in t25.items():
    assert bins == BINS, y

# --- Parser check against the pinned feed's own facts ----------------------
SPLIT_RS = {
    "no_qualifying_children": "childless",
    "one_qualifying_child": "one",
    "two_qualifying_children": "two",
    "three_or_more_qualifying_children": "three_plus",
}
MEASURE = {"eitc_total": "amount", "eitc_returns": "returns"}
checked = 0
for line in FEED.open() if FEED else ():
    if (
        "table_2_5.eitc_by_agi_children" not in line
        and "filing_season_week47" not in line
    ):
        continue
    fact = json.loads(line)
    layout = fact["layout"]
    rs = layout["record_set_id"]
    if ".table_2_5.eitc_by_agi_children." in rs:
        year = fact["period"]["value"]
        split = SPLIT_RS[rs.rsplit(".", 1)[1]]
        kind = MEASURE[layout["measure_id"]]
        bins, cells, totals = t25[year]
        row = layout["source_row_id"]
        i = SPLITS.index(split)
        if row == "total":
            mine = totals[kind][i]
        else:
            mine = cells[kind][i][layout["groupby_ordinal"]]
        assert abs(mine - float(fact["value"])) < 0.5, (
            rs,
            row,
            kind,
            mine,
            fact["value"],
        )
        checked += 1
    elif "filing_season_week47" in rs:
        kind = "amount" if layout["measure_id"].endswith("amount") else "returns"
        bins, values, totals = fs[2024]
        row = layout["source_row_id"]
        if row == "earned_income_credit":
            mine = totals[kind]
        else:
            mine = values[kind][layout["groupby_ordinal"]]
        assert abs(mine - float(fact["value"])) < 0.5, (
            rs,
            row,
            kind,
            mine,
            fact["value"],
        )
        checked += 1
if FEED:
    assert checked == 2 * 4 * 2 * 29 + 2 * 17, checked
    print(f"parser check: {checked} feed facts reproduced exactly")


def score(
    pred_bins, pred_totals, truth_bins, truth_totals, index, n_groups, active_total
):
    """Errors of one option's targets against level-matched truth."""

    scale = active_total / truth_bins.sum()
    tb = truth_bins * scale
    tt = truth_totals * scale
    pg = collapse(pred_bins, index, n_groups)
    tg = collapse(tb, index, n_groups)
    p_split = pred_bins.sum(axis=1)
    t_split = tb.sum(axis=1)
    nz = tb > 0
    rel_bins = pred_bins[nz] / tb[nz] - 1
    out = {
        # What the bins say about each split's national total.
        "bins_split_rel_error": dict(
            zip(SPLITS, (p_split / t_split - 1).tolist(), strict=True)
        ),
        "bins_split_share_error_pp": dict(
            zip(
                SPLITS,
                (100 * (p_split - t_split) / t_split.sum()).tolist(),
                strict=True,
            )
        ),
        # Split x group cells: total variation distance of the shares.
        "group_cells_tvd": float(0.5 * np.abs(pg - tg).sum() / tg.sum()),
        # Target rows as the capped relative-error loss sees them.
        "bin_targets_mean_capped_abs_rel_error": float(
            np.mean(np.minimum(np.abs(rel_bins), 1.0))
        ),
        "bin_targets_sqrtw_capped_abs_rel_error": float(
            np.sum(np.sqrt(tb[nz]) * np.minimum(np.abs(rel_bins), 1.0))
            / np.sum(np.sqrt(tb[nz]))
        ),
    }
    if pred_totals is not None:
        rel_tot = pred_totals / tt - 1
        out["total_rows_rel_error"] = dict(zip(SPLITS, rel_tot.tolist(), strict=True))
        out["total_rows_mean_abs_rel_error"] = float(np.mean(np.abs(rel_tot)))
        out["total_over_bins"] = dict(
            zip(SPLITS, (pred_totals / p_split).tolist(), strict=True)
        )
    out["bins_split_mean_abs_rel_error"] = float(np.mean(np.abs(p_split / t_split - 1)))
    return out


results = {"pairs": {}, "level_gap": {}}
for s, t in sorted(CLEAN | POLICY_BREAK):
    for mode in ("fs", "oracle"):
        if mode == "fs" and t not in fs:
            continue
        key = f"{s}->{t}|{mode}"
        rec = {"clean": (s, t) in CLEAN}
        for kind in KINDS:
            _, src_cells, src_totals = t25[s]
            _, tru_cells, tru_totals = t25[t]
            if mode == "fs":
                fs_bins, fs_values, fs_totals = fs[t]
                groups, index = group_of(BINS, fs_bins)
                margins = group_margins(fs_bins, fs_values[kind], groups)
            else:
                fs_bins = fs[2024][0]
                groups, index = group_of(BINS, fs_bins)
                margins = collapse(tru_cells[kind], index, len(groups)).sum(axis=0)
            opts = options(src_cells[kind], src_totals[kind], index, margins)
            active_total = margins.sum()
            rec[kind] = {
                "group_factors": opts["group_factors"].tolist(),
                "total_factor": opts["total_factor"],
                "ipf_iterations": opts["c"]["ipf_iterations"],
                "truth_child_share": dict(
                    zip(
                        SPLITS,
                        (tru_totals[kind] / tru_totals[kind].sum()).tolist(),
                        strict=True,
                    )
                ),
                "source_child_share": dict(
                    zip(
                        SPLITS,
                        (src_totals[kind] / src_totals[kind].sum()).tolist(),
                        strict=True,
                    )
                ),
                "options": {
                    name: score(
                        opts[name]["bins"],
                        opts[name]["totals"],
                        tru_cells[kind],
                        tru_totals[kind],
                        index,
                        len(groups),
                        active_total,
                    )
                    for name in "abcd"
                },
            }
        results["pairs"][key] = rec

# Filing-season mid-November vs complete-year Table 2.5, same tax year.
for y in sorted(set(fs) & set(t25)):
    fs_bins, fs_values, fs_totals = fs[y]
    groups, index = group_of(BINS, fs_bins)
    rec = {}
    for kind in KINDS:
        t25_groups = collapse(t25[y][1][kind], index, len(groups)).sum(axis=0)
        fs_groups = group_margins(fs_bins, fs_values[kind], groups)
        rec[kind] = {
            "fs_total": fs_totals[kind],
            "t25_total": float(t25[y][2][kind].sum()),
            "fs_over_t25_total": fs_totals[kind] / float(t25[y][2][kind].sum()),
            "fs_over_t25_by_group": {
                f"{lo:g}-{hi:g}": (f / g if g else None)
                for (lo, hi), f, g in zip(groups, fs_groups, t25_groups, strict=True)
            },
        }
    results["level_gap"][y] = rec

# Production: TY2023 Table 2.5 -> TY2024 filing season (no truth yet).
prod = {}
for kind in KINDS:
    _, src_cells, src_totals = t25[2023]
    fs_bins, fs_values, fs_totals = fs[2024]
    groups, index = group_of(BINS, fs_bins)
    margins = group_margins(fs_bins, fs_values[kind], groups)
    opts = options(src_cells[kind], src_totals[kind], index, margins)
    prod[kind] = {
        "groups": [f"{lo:g}-{hi:g}" for lo, hi in groups],
        "group_factors": opts["group_factors"].tolist(),
        "total_factor": opts["total_factor"],
        "options": {
            name: {
                "bins_split": dict(
                    zip(SPLITS, opts[name]["bins"].sum(axis=1).tolist(), strict=True)
                ),
                "total_rows": None
                if opts[name]["totals"] is None
                else dict(zip(SPLITS, opts[name]["totals"].tolist(), strict=True)),
            }
            for name in "abcd"
        },
    }
results["production_2023_to_2024"] = prod


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


summary = {}
for mode in ("fs", "oracle"):
    keys = [k for k, v in results["pairs"].items() if k.endswith(mode) and v["clean"]]
    for kind in KINDS:
        row = {"pairs": keys}
        for name in "abcd":
            o = [results["pairs"][k][kind]["options"][name] for k in keys]
            row[name] = {
                "bins_split_mean_abs_rel_error": mean(
                    x["bins_split_mean_abs_rel_error"] for x in o
                ),
                "group_cells_tvd": mean(x["group_cells_tvd"] for x in o),
                "bin_targets_mean_capped_abs_rel_error": mean(
                    x["bin_targets_mean_capped_abs_rel_error"] for x in o
                ),
                "bin_targets_sqrtw_capped_abs_rel_error": mean(
                    x["bin_targets_sqrtw_capped_abs_rel_error"] for x in o
                ),
                "total_rows_mean_abs_rel_error": None
                if name == "d"
                else mean(x["total_rows_mean_abs_rel_error"] for x in o),
                "worst_total_row_abs_rel_error": None
                if name == "d"
                else max(
                    max(abs(v) for v in x["total_rows_rel_error"].values()) for x in o
                ),
                "worst_bins_split_abs_rel_error": max(
                    max(abs(v) for v in x["bins_split_rel_error"].values()) for x in o
                ),
            }
        summary[f"{mode}|{kind}"] = row
results["summary_clean_pairs"] = summary
OUT.write_text(json.dumps(results, indent=1))

# --- Report -----------------------------------------------------------------
print("\nFiling-season mid-November / complete-year Table 2.5 (same tax year)")
for y, rec in results["level_gap"].items():
    for kind in KINDS:
        g = rec[kind]
        print(
            f"  TY{y} {kind:7s} total {g['fs_over_t25_total']:.4f}  by group",
            {
                k: (None if v is None else round(v, 3))
                for k, v in g["fs_over_t25_by_group"].items()
            },
        )

for key, rec in results["pairs"].items():
    print(f"\n=== {key}  ({'clean' if rec['clean'] else 'POLICY BREAK'})")
    for kind in KINDS:
        r = rec[kind]
        print(
            f"  {kind}: group factors {[round(f, 3) for f in r['group_factors']]}  all-AGI {r['total_factor']:.4f}"
        )
        print(
            f"    source share {dict((k, round(v, 4)) for k, v in r['source_child_share'].items())}"
        )
        print(
            f"    truth  share {dict((k, round(v, 4)) for k, v in r['truth_child_share'].items())}"
        )
        for name in "abcd":
            o = r["options"][name]
            line = f"    ({name}) bins split err " + " ".join(
                f"{100 * v:+6.2f}%" for v in o["bins_split_rel_error"].values()
            )
            if "total_rows_rel_error" in o:
                line += "  | total-row err " + " ".join(
                    f"{100 * v:+6.2f}%" for v in o["total_rows_rel_error"].values()
                )
            line += f"  | cells TVD {o['group_cells_tvd']:.4f}  bin capped |rel| {o['bin_targets_mean_capped_abs_rel_error']:.4f}"
            print(line)

print("\n=== Summary over clean pairs (mean)")
for key, row in summary.items():
    print(f"  {key}: {len(row['pairs'])} pairs")
    for name in "abcd":
        o = row[name]
        print(
            f"    ({name}) bins-split |err| {100 * o['bins_split_mean_abs_rel_error']:.2f}% (worst {100 * o['worst_bins_split_abs_rel_error']:.1f}%)"
            + (
                ""
                if o["total_rows_mean_abs_rel_error"] is None
                else f"  total-row |err| {100 * o['total_rows_mean_abs_rel_error']:.2f}% (worst {100 * o['worst_total_row_abs_rel_error']:.1f}%)"
            )
            + f"  cells TVD {o['group_cells_tvd']:.4f}  bins capped {o['bin_targets_mean_capped_abs_rel_error']:.4f}  sqrt-w {o['bin_targets_sqrtw_capped_abs_rel_error']:.4f}"
        )
