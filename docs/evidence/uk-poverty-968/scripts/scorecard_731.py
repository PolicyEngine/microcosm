"""The microcosm#731 benchmark scorecard, replayed on a local H5 at 2026.

Vahid's scorecard (microcosm#731) compared populace_uk_2023 and
enhanced_frs_2024_25 against approximate admin benchmarks at 2026 through the
policyengine.py wrapper. This probe computes the same rows for any local H5
directly through policyengine_uk, so the v15 candidate can sit in the same
table. Run under the DRIVER TREE's venv (the publication engine):

    cd <populace worktree> && uv run --no-sync python scorecard_731.py \
        --h5 <dataset.h5> --label microcosm_uk_2024_v15 --year 2026 \
        --out out/v15/scorecard_731_candidate.json

Conventions, matching the #731 reproduction code: taxpayers = persons with
income_tax > 0; UC families = benunits with universal_credit > 0; Gini and the
top-1% share are household-weighted over household_net_income (Vahid's "net hh
income" label). All outputs are weighted aggregates.

Poverty (microcosm#968). The engine's ``in_poverty_bhc`` is the ABSOLUTE line
(60 % of the 2010/11 median uprated by CPI), not the relative line HBAI's
headline ~17 % refers to, so the scorecard reports the pair HBAI publishes:

* ``poverty_bhc_absolute_pct`` — the engine flag, person-weighted share of
  individuals (HBAI FYE 2024 absolute BHC: 15 %). ``poverty_bhc_pct`` is kept
  as the same value for older reports that read that key.
* ``poverty_bhc_relative_pct`` — individuals in households whose equivalised
  HBAI net income is under 60 % of the PERSON-weighted median of the same
  file (HBAI FYE 2024: 17 %). The engine's ``in_relative_poverty_bhc`` uses a
  household-weighted median, which understates this by ~2 points, so the
  median is taken here directly.
* ``poverty_ahc_absolute_pct`` / ``poverty_ahc_relative_pct`` — the AHC pair
  (HBAI FYE 2024: 18 % / 21 %).
* ``median_equiv_bhc_income_week`` — the person-weighted median, £/week for a
  couple with no children (HBAI FYE 2024: £650).
* ``child_poverty_bhc_relative_pct`` — children under 18 (HBAI FYE 2024: 23 %).

HBAI reference: DWP, Households Below Average Income FYE 2024, summary tables
1.2b and 1.3a–1.4a (March 2025).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def gini(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum_w = np.cumsum(w)
    cum_vw = np.cumsum(v * w)
    total_w, total_vw = cum_w[-1], cum_vw[-1]
    # Weighted Gini via the Lorenz-curve trapezoid rule.
    lorenz = cum_vw / total_vw
    prev_lorenz = np.concatenate(([0.0], lorenz[:-1]))
    return float(1.0 - ((w / total_w) * (lorenz + prev_lorenz)).sum())


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2.0)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--hbai-year",
        type=int,
        default=2024,
        help="year at which the HBAI-basis block is measured (HBAI FYE 2024 = April 2023 to March 2024; 0 disables)",
    )
    args = parser.parse_args()

    from importlib.metadata import version

    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=str(args.h5))
    year = args.year

    def measure(year):
        def person(var):
            return np.asarray(sim.calculate(var, year).values, dtype=float)

        def to_person(var):
            return np.asarray(
                sim.calculate(var, year, map_to="person").values, dtype=float
            )

        pw = np.asarray(sim.calculate("person_weight", year).values, dtype=float)
        bw = np.asarray(sim.calculate("benunit_weight", year).values, dtype=float)
        hw = np.asarray(sim.calculate("household_weight", year).values, dtype=float)

        income_tax = person("income_tax")
        uc = np.asarray(sim.calculate("universal_credit", year).values, dtype=float)
        net = np.asarray(
            sim.calculate("household_net_income", year).values, dtype=float
        )
        poverty_abs_bhc = to_person("in_poverty_bhc")
        poverty_abs_ahc = to_person("in_poverty_ahc")
        equiv_bhc = to_person("equiv_hbai_household_net_income")
        equiv_ahc = to_person("equiv_hbai_household_net_income_ahc")
        is_child = person("age") < 18

        median_bhc = weighted_median(equiv_bhc, pw)
        median_ahc = weighted_median(equiv_ahc, pw)
        rel_bhc = equiv_bhc < 0.6 * median_bhc
        rel_ahc = equiv_ahc < 0.6 * median_ahc

        def share(mask, weights=pw):
            return float(100.0 * weights[mask].sum() / weights.sum())

        def quantile(values, weights, q):
            order = np.argsort(values)
            cum = np.cumsum(weights[order]) / weights[order].sum()
            return float(values[order][np.searchsorted(cum, q)])

        order = np.argsort(net)[::-1]
        top1_cut = np.searchsorted(np.cumsum(hw[order]), hw.sum() * 0.01)
        top1_share = float(
            (net[order][: top1_cut + 1] * hw[order][: top1_cut + 1]).sum()
            / (net * hw).sum()
        )

        poverty_bhc_absolute = share(poverty_abs_bhc > 0)
        rows = {
            "population_m": float(pw.sum() / 1e6),
            "households_m": float(hw.sum() / 1e6),
            "state_pension_bn": float((person("state_pension") * pw).sum() / 1e9),
            "income_tax_bn": float((income_tax * pw).sum() / 1e9),
            "income_taxpayers_m": float(pw[income_tax > 0].sum() / 1e6),
            "universal_credit_bn": float((uc * bw).sum() / 1e9),
            "uc_families_m": float(bw[uc > 0].sum() / 1e6),
            "child_benefit_bn": float(
                (
                    np.asarray(sim.calculate("child_benefit", year).values, float) * bw
                ).sum()
                / 1e9
            ),
            "pension_credit_bn": float(
                (
                    np.asarray(sim.calculate("pension_credit", year).values, float) * bw
                ).sum()
                / 1e9
            ),
            "council_tax_bn": float(
                (
                    np.asarray(sim.calculate("council_tax", year).values, float) * hw
                ).sum()
                / 1e9
            ),
            # microcosm#968: the engine flag is the absolute line; keep the old key
            # (same value) for reports that read it, and add the HBAI pair.
            "poverty_bhc_pct": poverty_bhc_absolute,
            "poverty_bhc_absolute_pct": poverty_bhc_absolute,
            "poverty_bhc_relative_pct": share(rel_bhc),
            "poverty_ahc_absolute_pct": share(poverty_abs_ahc > 0),
            "poverty_ahc_relative_pct": share(rel_ahc),
            "median_equiv_bhc_income_week": median_bhc / 52.0,
            "child_poverty_bhc_relative_pct": float(
                100.0 * pw[is_child & rel_bhc].sum() / pw[is_child].sum()
            ),
            "p10_over_median_bhc": quantile(equiv_bhc, pw, 0.10) / median_bhc,
            "p90_over_median_bhc": quantile(equiv_bhc, pw, 0.90) / median_bhc,
            "zero_income_share_pct": share(equiv_bhc < 10.0 * 52.0),
            "top_1pct_net_income_share_pct": 100.0 * top1_share,
            "gini_net_hh_income": gini(net, hw),
        }
        return rows

    rows = measure(year)
    HBAI_KEYS = (
        "poverty_bhc_absolute_pct",
        "poverty_bhc_relative_pct",
        "poverty_ahc_absolute_pct",
        "poverty_ahc_relative_pct",
        "child_poverty_bhc_relative_pct",
        "median_equiv_bhc_income_week",
        "p10_over_median_bhc",
        "p90_over_median_bhc",
        "zero_income_share_pct",
    )
    hbai_basis = None
    if args.hbai_year:
        # HBAI FYE 2024 is April 2023 to March 2024; the closest engine year is 2024. The
        # files are measured at that year so the comparison does not conflate years
        # (microcosm#968, review point 4). p10/p90 are ratios to the person-weighted
        # median; zero_income_share is individuals under £10/week equivalised (HBAI's
        # £0-10 band, 1.0 % of individuals in FYE 2024).
        hb_rows = measure(args.hbai_year) if args.hbai_year != year else rows
        hbai_basis = {
            "year": args.hbai_year,
            "rows": {k: hb_rows[k] for k in HBAI_KEYS},
            "hbai": {
                "label": "HBAI FYE 2024 (April 2023 to March 2024), survey-reported incomes",
                "poverty_bhc_absolute_pct": 15,
                "poverty_bhc_relative_pct": 17,
                "poverty_ahc_absolute_pct": 18,
                "poverty_ahc_relative_pct": 21,
                "child_poverty_bhc_relative_pct": 23,
                "median_equiv_bhc_income_week": 650,
                "p10_over_median_bhc": 0.48,
                "p90_over_median_bhc": 1.955,
                "zero_income_share_pct": 1.0,
                "source": (
                    "DWP HBAI FYE 2024 (March 2025): summary tables 1.2b (quintile medians: "
                    "bottom 312, top 1271 on a 650 median) and 1.3a-1.4a; chart 2.4 income "
                    "bands (0.66m individuals at 0-10 GBP/week)"
                ),
            },
        }

    payload = {
        "probe": "microcosm#731 scorecard",
        "label": args.label,
        "h5": str(args.h5),
        "year": year,
        "engine": {"package": "policyengine-uk", "version": version("policyengine-uk")},
        "conventions": (
            "taxpayers: income_tax>0 persons; uc_families: universal_credit>0 "
            "benunits; poverty_bhc_absolute (= poverty_bhc_pct): person-weighted "
            "in_poverty_bhc, the engine's CPI-uprated 2010/11 line; "
            "poverty_*_relative: individuals under 60% of the person-weighted "
            "median equivalised HBAI net income of the same file (microcosm#968); "
            "median_equiv_bhc_income_week: person-weighted, couple-no-children "
            "equivalent; gini/top-1%: household-weighted household_net_income"
        ),
        "hbai_fye_2024_reference": {
            "poverty_bhc_absolute_pct": 15,
            "poverty_bhc_relative_pct": 17,
            "poverty_ahc_absolute_pct": 18,
            "poverty_ahc_relative_pct": 21,
            "median_equiv_bhc_income_week": 650,
            "child_poverty_bhc_relative_pct": 23,
            "year": "FYE 2024 (April 2023 to March 2024); compare with the hbai_basis block, not with the 2026 rows",
            "source": "DWP HBAI FYE 2024 summary tables 1.2b, 1.3a, 1.4a (March 2025)",
        },
        "rows": rows,
        "hbai_basis": hbai_basis,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({args.label: rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
