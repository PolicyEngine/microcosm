"""Support oracle for microcosm#940: state x AGI-band SOI targets vs frame support.

Analysis only. Reads the offline #982 predictions (pre-calibration, PUF-clone
half) and SOI Historic Table 2 (TY2023 shares, TY2022 controls), derives the
state x band targets the #940 compiler rule would produce, and measures the
pre-calibration support each cell has under the solver's hard weight cap.

AGI here is a PROXY everywhere: the sum of the 15 imputed income items on the
PUF half (no adjustments, no Social Security), and the saved survey income
total on the survey half. Nothing here is calibrated, aged, or engine-computed.

Usage (deterministic, no side effects outside OUT)::

    PYTHONDONTWRITEBYTECODE=1 python support_oracle.py

Inputs are absolute paths pinned below.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
# Inputs (override with environment variables):
# - the #982 offline predictions, archived with the tail lane's evidence;
# - SOI Historic Table 2 state files as registered in Chronicle
#   (packages/irs_soi/historic_table_2_state_agi_2023 and _2022 read them).
LANE = Path(
    os.environ.get(
        "MICROCOSM_940_PREDICTIONS_DIR",
        "/Users/maxghenis/PolicyEngine/_recovered/"
        "lane958-958-964-982-archive-20260926/982",
    )
)
SOI_DIR = Path(
    os.environ.get(
        "MICROCOSM_940_SOI_DIR",
        str(Path.home() / "PolicyEngine/chronicle/db/data/irs_soi/historic_table_2"),
    )
)
SOI_2023 = SOI_DIR / "23in55cmcsv.csv"
SOI_2022 = SOI_DIR / "22in55cmcsv.csv"

VARIANTS = {
    "current": "old design (QRF copied the unit's own survey income)",
    "demographic_midrank6_earn": "chosen #1033 design (demographics + mid-rank + earnings flag)",
}

# The 15 imputed income items whose sum is PROXY AGI (score_variants.py INCOME).
INCOME = [
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "taxable_private_pension_income",
    "taxable_ira_distributions",
    "rental_income",
    "estate_income",
    "farm_income",
    "miscellaneous_income",
    "partnership_income",
    "s_corp_income",
]
SIX_SURVEY = [
    "survey_puf_predictor_employment_income",
    "survey_puf_predictor_self_employment_income",
    "survey_puf_predictor_taxable_interest_income",
    "survey_puf_predictor_dividend_income",
    "survey_puf_predictor_short_term_capital_gains",
    "survey_puf_predictor_long_term_capital_gains",
]

# Historic Table 2 AGI_STUB -> band label for the four #940 bands.
BANDS = {
    "100k_200k": (7, 1e5, 2e5),
    "200k_500k": (8, 2e5, 5e5),
    "500k_1m": (9, 5e5, 1e6),
    "1m_plus": (10, 1e6, np.inf),
}
ALL_STUBS = list(range(1, 11))

# Solver contract, read from the worktree at
# packages/microcosm-build/src/microcosm/build/us/spec/calibration.yaml
# (hard_constraints.max_weight_ratio: 5.0, mass: conserve, solver.loss.params
# target_loss_cap: 1.0 with target_scale = max(abs(target), 1)) and
# packages/microcosm-calibrate/src/microcosm/calibrate/solve.py (log-weights,
# Adam, upper clamp only: log_w.clamp_(max=log(max_weight_ratio * w0))).
MAX_WEIGHT_RATIO = 5.0
TARGET_LOSS_CAP = 1.0

# CBO AGI 2022->2024 chained factor supplied by the caller (fiscal_targets.py
# documents "x1.1203 over 2022->2024" for the CBO AGI default).
AGI_AGING_2022_TO_2024 = 1.1203460

# SOI Table 1.1 TY2023 $1M+ sub-bands (score_variants.py SOI_TABLE_1_1):
# band -> (returns, AGI $bn).
SOI_TABLE_1_1_1M_PLUS = {
    "1m_to_1_5m": (368_931, 443.658),
    "1_5m_to_2m": (147_290, 252.903),
    "2m_to_5m": (203_229, 602.653),
    "5m_to_10m": (49_262, 336.335),
    "10m_plus": (30_382, 907.901),
}

# Standard state postal -> FIPS for the 50 states + DC.
STATE_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}
FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}
assert len(STATE_FIPS) == 51 and len(FIPS_STATE) == 51
# Spot checks requested by the brief.
assert STATE_FIPS["CO"] == "08" and STATE_FIPS["CA"] == "06"
assert STATE_FIPS["DC"] == "11" and STATE_FIPS["WY"] == "56"


def read_ht2(path: Path) -> pd.DataFrame:
    """Read Historic Table 2: STATE, AGI_STUB, N1 (returns), A00100 (AGI $)."""
    raw = pd.read_csv(
        path,
        encoding="latin-1",
        thousands=",",
        usecols=["STATE", "AGI_STUB", "N1", "A00100"],
        dtype={"STATE": str},
    )
    raw = raw[raw["STATE"].isin(STATE_FIPS)].copy()
    raw["AGI_STUB"] = raw["AGI_STUB"].astype(int)
    raw["returns"] = raw["N1"].astype(float)
    raw["agi"] = raw["A00100"].astype(float) * 1_000.0  # $ thousands -> $
    assert raw.groupby("STATE").size().eq(11).all(), "expected stubs 0..10 per state"
    return raw[["STATE", "AGI_STUB", "returns", "agi"]]


def build_targets() -> tuple[pd.DataFrame, dict]:
    """State x band targets at 2024 under the #940 compiler rule."""
    t23 = read_ht2(SOI_2023)
    t22 = read_ht2(SOI_2022)
    rows = []
    stub0_check = {}
    for st in sorted(STATE_FIPS):
        s23 = t23[t23["STATE"] == st].set_index("AGI_STUB")
        s22 = t22[t22["STATE"] == st].set_index("AGI_STUB")
        denom_n = float(s23.loc[ALL_STUBS, "returns"].sum())
        denom_a = float(s23.loc[ALL_STUBS, "agi"].sum())
        stub0_check[st] = {
            "ty2023_stub0_returns": float(s23.loc[0, "returns"]),
            "ty2023_sum_stubs_1_10_returns": denom_n,
            "ty2023_stub0_agi": float(s23.loc[0, "agi"]),
            "ty2023_sum_stubs_1_10_agi": denom_a,
        }
        control_n = float(s22.loc[0, "returns"])
        control_a = float(s22.loc[0, "agi"])
        for band, (stub, low, high) in BANDS.items():
            v_n = float(s23.loc[stub, "returns"])
            v_a = float(s23.loc[stub, "agi"])
            share_n = v_n / denom_n
            share_a = v_a / denom_a
            rows.append(
                {
                    "state": st,
                    "fips": STATE_FIPS[st],
                    "band": band,
                    "band_low": low,
                    "band_high": high,
                    "soi2023_returns": v_n,
                    "soi2023_agi": v_a,
                    "soi2023_share_returns": share_n,
                    "soi2023_share_agi": share_a,
                    "ty2022_control_returns": control_n,
                    "ty2022_control_agi": control_a,
                    "target_returns_2024": share_n * control_n,
                    "target_agi_2024": share_a * control_a * AGI_AGING_2022_TO_2024,
                }
            )
    targets = pd.DataFrame(rows)
    return targets, stub0_check


def load_variant(name: str) -> pd.DataFrame:
    d = pd.read_parquet(LANE / f"imputed_{name}.parquet")
    assert len(d) == 231_007, len(d)
    assert d["tax_unit_id"].is_unique
    out = pd.DataFrame(
        {
            "fips": d["state_fips"].astype(int).map(lambda x: f"{x:02d}"),
            "w": d["w"].astype(float),
            "proxy_agi": d[INCOME].sum(axis=1).astype(float),
            "survey_total": d["survey_total"].astype(float),
            "survey_six": d[SIX_SURVEY].sum(axis=1).astype(float),
        }
    )
    assert set(out["fips"]) == set(FIPS_STATE), "state_fips must be the 50 states + DC"
    assert (out["w"] > 0).all()
    return out


def band_mask(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return (values >= low) & (values < high)


def cell_stats(w: np.ndarray, a: np.ndarray, low: float) -> dict:
    """PUF-half style stats for one cell's records (weights w, income a)."""
    n = int(len(w))
    if n == 0:
        return {
            "n": 0,
            "N": 0.0,
            "A": 0.0,
            "mean_agi": np.nan,
            "max_agi": np.nan,
            "largest_share_of_A": np.nan,
            "largest_w": np.nan,
            "largest_share_above_low": np.nan,
            "min_agi": np.nan,
        }
    contrib = w * a
    amount = float(contrib.sum())
    i = int(np.argmax(contrib))  # largest single-record contribution to A
    above = w * np.maximum(a - low, 0.0) if np.isfinite(low) else contrib
    return {
        "n": n,
        "N": float(w.sum()),
        "A": amount,
        "mean_agi": amount / float(w.sum()),
        "max_agi": float(a.max()),
        "min_agi": float(a.min()),
        "largest_share_of_A": float(contrib[i] / amount) if amount else np.nan,
        "largest_w": float(w[i]),
        "largest_share_above_low": float(above.max() / above.sum())
        if above.sum() > 0
        else np.nan,
    }


def feasibility(w: np.ndarray, a: np.ndarray, target_n: float, target_a: float) -> dict:
    """Necessary conditions for (target_n, target_a) under the 5x upper cap.

    Weights may rise to MAX_WEIGHT_RATIO * w0 and fall to ~0 (no lower bound),
    so within one cell: N <= 5*sum(w0), A <= 5*sum(w0*a) (all a > 0 in these
    bands), and the target mean must lie inside [min(a), max(a)]. These are
    per-cell NECESSARY conditions only: every record's weight is shared across
    all targets it enters, and mass is conserved frame-wide.
    """
    n = int(len(w))
    cap_n = MAX_WEIGHT_RATIO * float(w.sum()) if n else 0.0
    cap_a = MAX_WEIGHT_RATIO * float((w * a).sum()) if n else 0.0
    target_mean = target_a / target_n if target_n > 0 else np.nan
    failing = []
    if n == 0:
        failing.append("empty_support")
        c1 = c2 = c3 = False
    else:
        c1 = target_n <= cap_n
        c2 = target_a <= cap_a
        c3 = bool(a.min() <= target_mean <= a.max())
        if not c1:
            failing.append("count_cap")
        if not c2:
            failing.append("amount_cap")
        if not c3:
            failing.append("mean_out_of_range")
    return {
        "cap_N_5x": cap_n,
        "cap_A_5x": cap_a,
        "frame_min_agi": float(a.min()) if n else np.nan,
        "frame_max_agi": float(a.max()) if n else np.nan,
        "target_mean_agi": target_mean,
        "c1_count_reachable": bool(c1),
        "c2_amount_reachable": bool(c2),
        "c3_mean_in_range": bool(c3),
        "feasible_necessary": bool(c1 and c2 and c3),
        "failing_conditions": "|".join(failing) if failing else "",
    }


def analyze_variant(name: str, targets: pd.DataFrame) -> pd.DataFrame:
    d = load_variant(name)
    rows = []
    for st in sorted(STATE_FIPS):
        fips = STATE_FIPS[st]
        s = d[d["fips"] == fips]
        w_all = s["w"].to_numpy()
        agi = s["proxy_agi"].to_numpy()
        svy = s["survey_total"].to_numpy()
        svy6 = s["survey_six"].to_numpy()
        for band, (_stub, low, high) in BANDS.items():
            tgt = targets[(targets["state"] == st) & (targets["band"] == band)].iloc[0]
            m_puf = band_mask(agi, low, high)
            m_svy = band_mask(svy, low, high)
            m_svy6 = band_mask(svy6, low, high)
            puf = cell_stats(w_all[m_puf], agi[m_puf], low)
            svy_c = cell_stats(w_all[m_svy], svy[m_svy], low)
            svy6_c = cell_stats(w_all[m_svy6], svy6[m_svy6], low)
            # Frame view: PUF clone at w with proxy AGI + survey twin at w with
            # survey income. Each channel carries half the population.
            fw = np.concatenate([w_all[m_puf], w_all[m_svy]])
            fa = np.concatenate([agi[m_puf], svy[m_svy]])
            frame_n = int(len(fw))
            frame_count = float(fw.sum())
            frame_amount = float((fw * fa).sum())
            tn, ta = float(tgt["target_returns_2024"]), float(tgt["target_agi_2024"])
            feas = feasibility(fw, fa, tn, ta)
            frame6_count = puf["N"] + svy6_c["N"]
            frame6_amount = puf["A"] + svy6_c["A"]
            row = {
                "variant": name,
                "state": st,
                "fips": fips,
                "band": band,
                **{k: tgt[k] for k in tgt.index if k not in ("state", "fips", "band")},
                # PUF half (weight w = half-population weight).
                "puf_n": puf["n"],
                "puf_N": puf["N"],
                "puf_A": puf["A"],
                "puf_mean_agi": puf["mean_agi"],
                "puf_min_agi": puf["min_agi"],
                "puf_max_agi": puf["max_agi"],
                "puf_largest_share_of_A": puf["largest_share_of_A"],
                "puf_largest_w": puf["largest_w"],
                "puf_largest_share_above_low": puf["largest_share_above_low"],
                # x2 view (as if the PUF half were the whole population).
                "puf_N_x2": 2 * puf["N"],
                "puf_A_x2": 2 * puf["A"],
                "puf_largest_w_x2": 2 * puf["largest_w"] if puf["n"] else np.nan,
                # Survey half, income = survey_total (broad saved survey total).
                "svy_n": svy_c["n"],
                "svy_N": svy_c["N"],
                "svy_A": svy_c["A"],
                "svy_max_income": svy_c["max_agi"],
                # Survey half, income = six predictor items only (sensitivity).
                "svy6_n": svy6_c["n"],
                "svy6_N": svy6_c["N"],
                "svy6_A": svy6_c["A"],
                # Frame view.
                "frame_n": frame_n,
                "frame_N": frame_count,
                "frame_A": frame_amount,
                "ratio_N": frame_count / tn if tn else np.nan,
                "ratio_A": frame_amount / ta if ta else np.nan,
                "frame6_N": frame6_count,
                "frame6_A": frame6_amount,
                "ratio6_N": frame6_count / tn if tn else np.nan,
                "ratio6_A": frame6_amount / ta if ta else np.nan,
                **feas,
                "zero_support": frame_n == 0,
                # Loss cap 1.0 on |est-target|/|target|: a start beyond 2x the
                # target (either direction past 0) sits on the clamp with zero
                # gradient. Only the over-2x side is reachable for a positive
                # estimate; under-target starts always have gradient.
                "over2x_A": bool(frame_amount > 2.0 * ta),
                "over2x_N": bool(frame_count > 2.0 * tn),
                "start_loss_A_capped": bool(
                    abs(frame_amount - ta) / max(abs(ta), 1) >= TARGET_LOSS_CAP
                ),
                "start_loss_N_capped": bool(
                    abs(frame_count - tn) / max(abs(tn), 1) >= TARGET_LOSS_CAP
                ),
            }
            # Sensitivity: 1m_plus with the single largest proxy-AGI record removed.
            if band == "1m_plus" and frame_n > 0:
                j = int(np.argmax(fa))
                keep = np.ones(frame_n, dtype=bool)
                keep[j] = False
                fw2, fa2 = fw[keep], fa[keep]
                feas2 = feasibility(fw2, fa2, tn, ta)
                row.update(
                    {
                        "drop1_removed_agi": float(fa[j]),
                        "drop1_removed_w": float(fw[j]),
                        "drop1_removed_channel": "puf"
                        if j < int(m_puf.sum())
                        else "survey",
                        "drop1_frame_n": int(len(fw2)),
                        "drop1_frame_N": float(fw2.sum()),
                        "drop1_frame_A": float((fw2 * fa2).sum()),
                        "drop1_ratio_A": float((fw2 * fa2).sum() / ta)
                        if ta
                        else np.nan,
                        "drop1_feasible_necessary": feas2["feasible_necessary"],
                        "drop1_failing_conditions": feas2["failing_conditions"],
                        "drop1_over2x_A": bool((fw2 * fa2).sum() > 2.0 * ta),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def colorado_reproduction(chosen: pd.DataFrame) -> dict:
    """Check the #982 README Colorado $1M+ figures against the x2 view."""
    co = chosen[(chosen["state"] == "CO") & (chosen["band"] == "1m_plus")].iloc[0]
    readme = {
        "records": 15,
        "returns_x2": 14_847,
        "largest_share_of_income_above_1m": 0.840,
        "largest_proxy_agi": 125.4e6,
        "largest_weight_x2": 867,
    }
    got = {
        "records": int(co["puf_n"]),
        "returns_x2": float(co["puf_N_x2"]),
        "largest_share_of_income_above_1m": float(co["puf_largest_share_above_low"]),
        "largest_proxy_agi": float(co["puf_max_agi"]),
        "largest_weight_x2": float(co["puf_largest_w_x2"]),
    }
    reproduces = (
        got["records"] == readme["records"]
        and abs(got["returns_x2"] - readme["returns_x2"]) < 1.0
        and abs(
            got["largest_share_of_income_above_1m"]
            - readme["largest_share_of_income_above_1m"]
        )
        < 0.0015
        and abs(got["largest_proxy_agi"] - readme["largest_proxy_agi"]) < 0.05e6
        and abs(got["largest_weight_x2"] - readme["largest_weight_x2"]) < 1.0
    )
    return {"readme_982": readme, "computed": got, "reproduces": bool(reproduces)}


def summarize(all_rows: pd.DataFrame, targets: pd.DataFrame, stub0_check: dict) -> dict:
    summary: dict = {
        "labels": {
            "agi": "PROXY AGI: sum of 15 imputed income items on the PUF half; "
            "saved survey_total (broad survey income list from compare_predictors.py "
            "SURVEY, up to 14 items, NOT the six predictor items) on the survey half",
            "status": "pre-calibration, offline #982 predictions, 8-tree fits",
            "targets": "share(TY2023 HT2 stub b / sum stubs 1..10) x TY2022 HT2 stub-0 "
            "control; AGI aged x1.1203460 to 2024; counts not aged",
            "solver": "max_weight_ratio 5.0 (upper clamp only), mass conserve, "
            "target_loss_cap 1.0 on |est-target|/max(|target|,1)",
        },
        "stub0_consistency": {
            "max_abs_rel_diff_returns": max(
                abs(v["ty2023_stub0_returns"] / v["ty2023_sum_stubs_1_10_returns"] - 1)
                for v in stub0_check.values()
            ),
            "max_abs_rel_diff_agi": max(
                abs(v["ty2023_stub0_agi"] / v["ty2023_sum_stubs_1_10_agi"] - 1)
                for v in stub0_check.values()
            ),
        },
        "targets_national_sums": {},
        "variants": {},
    }
    # National sums of the derived targets (sum over 51 states).
    for band in BANDS:
        t = targets[targets["band"] == band]
        summary["targets_national_sums"][band] = {
            "soi2023_returns_51": float(t["soi2023_returns"].sum()),
            "soi2023_agi_51": float(t["soi2023_agi"].sum()),
            "target_returns_2024_51": float(t["target_returns_2024"].sum()),
            "target_agi_2024_51": float(t["target_agi_2024"].sum()),
        }
    t11_returns = sum(v[0] for v in SOI_TABLE_1_1_1M_PLUS.values())
    t11_agi = sum(v[1] for v in SOI_TABLE_1_1_1M_PLUS.values()) * 1e9
    summary["soi_table_1_1_ty2023_1m_plus"] = {"returns": t11_returns, "agi": t11_agi}

    for name in VARIANTS:
        v = all_rows[all_rows["variant"] == name]
        per_band = {}
        for band in BANDS:
            b = v[v["band"] == band]
            per_band[band] = {
                "states": int(len(b)),
                "zero_support": int(b["zero_support"].sum()),
                "fail_count_cap": int((~b["c1_count_reachable"]).sum()),
                "fail_amount_cap": int((~b["c2_amount_reachable"]).sum()),
                "fail_mean_out_of_range": int((~b["c3_mean_in_range"]).sum()),
                "infeasible_any": int((~b["feasible_necessary"]).sum()),
                "over2x_A": int(b["over2x_A"].sum()),
                "over2x_N": int(b["over2x_N"].sum()),
                "start_loss_A_capped": int(b["start_loss_A_capped"].sum()),
                "start_loss_N_capped": int(b["start_loss_N_capped"].sum()),
                "ratio_A_median": float(b["ratio_A"].median()),
                "ratio_A_min": float(b["ratio_A"].min()),
                "ratio_A_max": float(b["ratio_A"].max()),
                "ratio_N_median": float(b["ratio_N"].median()),
                "ratio_N_min": float(b["ratio_N"].min()),
                "ratio_N_max": float(b["ratio_N"].max()),
                "puf_records_total": int(b["puf_n"].sum()),
                "puf_records_min_state": int(b["puf_n"].min()),
                "states_puf_n_lt_10": int((b["puf_n"] < 10).sum()),
                "states_largest_record_gt_50pct_of_puf_A": int(
                    (b["puf_largest_share_of_A"] > 0.5).sum()
                ),
                "national_frame_N": float(b["frame_N"].sum()),
                "national_frame_A": float(b["frame_A"].sum()),
                "national_puf_N_x2": float(b["puf_N_x2"].sum()),
                "national_puf_A_x2": float(b["puf_A_x2"].sum()),
            }
        top = v[v["band"] == "1m_plus"].copy()
        top["abs_log_ratio_A"] = np.abs(np.log(top["ratio_A"].replace(0, np.nan)))
        worst = top.sort_values("abs_log_ratio_A", ascending=False).head(10)
        cols = [
            "state",
            "puf_n",
            "svy_n",
            "frame_N",
            "target_returns_2024",
            "ratio_N",
            "frame_A",
            "target_agi_2024",
            "ratio_A",
            "puf_largest_share_of_A",
            "puf_max_agi",
            "failing_conditions",
            "over2x_A",
            "drop1_ratio_A",
            "drop1_feasible_necessary",
        ]
        drop1 = top.copy()
        # These columns are object-dtype (NaN outside 1m_plus); cast before `~`,
        # since bitwise-not of a Python bool is an int (~True == -2, truthy).
        for c in ("drop1_feasible_necessary", "drop1_over2x_A"):
            drop1[c] = drop1[c].fillna(False).astype(bool)
        sens = {
            "states_with_support": int((drop1["frame_n"] > 0).sum()),
            "feasible_before": int(drop1["feasible_necessary"].sum()),
            "feasible_after_drop1": int(drop1["drop1_feasible_necessary"].sum()),
            "over2x_A_before": int(drop1["over2x_A"].sum()),
            "over2x_A_after_drop1": int(drop1["drop1_over2x_A"].sum()),
            "ratio_A_median_before": float(drop1["ratio_A"].median()),
            "ratio_A_median_after_drop1": float(drop1["drop1_ratio_A"].median()),
            "states_flipping_feasible_to_infeasible": sorted(
                drop1[drop1["feasible_necessary"] & ~drop1["drop1_feasible_necessary"]][
                    "state"
                ].tolist()
            ),
            "states_flipping_infeasible_to_feasible": sorted(
                drop1[~drop1["feasible_necessary"] & drop1["drop1_feasible_necessary"]][
                    "state"
                ].tolist()
            ),
            "states_flipping_over2x_to_under": sorted(
                drop1[drop1["over2x_A"] & ~drop1["drop1_over2x_A"]]["state"].tolist()
            ),
        }
        co = v[v["state"] == "CO"].set_index("band")
        summary["variants"][name] = {
            "description": VARIANTS[name],
            "per_band": per_band,
            "worst10_1m_plus_by_abs_log_ratio_A": json.loads(
                worst[cols].to_json(orient="records")
            ),
            "highest10_1m_plus_ratio_A": json.loads(
                top.sort_values("ratio_A", ascending=False)
                .head(10)[cols]
                .to_json(orient="records")
            ),
            "lowest10_1m_plus_ratio_A": json.loads(
                top.sort_values("ratio_A", ascending=True)
                .head(10)[cols]
                .to_json(orient="records")
            ),
            "colorado": json.loads(co.to_json(orient="index")),
            "sensitivity_1m_plus_drop_largest_record": sens,
            "national_1m_plus_vs_table_1_1": {
                "puf_x2_returns": float(per_band["1m_plus"]["national_puf_N_x2"]),
                "puf_x2_agi": float(per_band["1m_plus"]["national_puf_A_x2"]),
                "frame_returns": float(per_band["1m_plus"]["national_frame_N"]),
                "frame_agi": float(per_band["1m_plus"]["national_frame_A"]),
                "targets_2024_returns_51": summary["targets_national_sums"]["1m_plus"][
                    "target_returns_2024_51"
                ],
                "targets_2024_agi_51": summary["targets_national_sums"]["1m_plus"][
                    "target_agi_2024_51"
                ],
                "table_1_1_ty2023_returns": t11_returns,
                "table_1_1_ty2023_agi": t11_agi,
            },
        }
    return summary


def main() -> None:
    os.chdir(OUT)
    targets, stub0_check = build_targets()
    frames = [analyze_variant(name, targets) for name in VARIANTS]
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(OUT / "support_by_state_band.csv", index=False)

    summary = summarize(all_rows, targets, stub0_check)
    chosen = all_rows[all_rows["variant"] == "demographic_midrank6_earn"]
    summary["colorado_reproduction_check"] = colorado_reproduction(chosen)

    # Mapping sanity: largest states by PUF-half weight should be CA, TX, FL, NY.
    d = load_variant("current")
    by_state = d.groupby("fips")["w"].sum().sort_values(ascending=False)
    summary["fips_mapping_sanity_top5_by_weight"] = [
        FIPS_STATE[f] for f in by_state.head(5).index
    ]
    summary["puf_half_weight_total"] = float(d["w"].sum())
    summary["survey_total_vs_six_items"] = {
        "rows_differing": int(
            (np.abs(d["survey_total"] - d["survey_six"]) > 1e-6).sum()
        ),
        "rows": int(len(d)),
        "note": "survey_total is compare_predictors.py's broad SURVEY list (up to 14 "
        "items incl. pension, IRA, rental, farm, UI, alimony, misc), not the six "
        "predictor items; the svy6_* columns use the six items.",
    }

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))

    (OUT / "support_summary.json").write_text(
        json.dumps(summary, indent=1, default=_default) + "\n"
    )
    print(
        json.dumps(summary["colorado_reproduction_check"], indent=1, default=_default)
    )
    print("top5 by weight:", summary["fips_mapping_sanity_top5_by_weight"])
    print("stub0 consistency:", summary["stub0_consistency"])
    for name in VARIANTS:
        print(name)
        print(
            json.dumps(
                summary["variants"][name]["per_band"], indent=1, default=_default
            )
        )


if __name__ == "__main__":
    main()
