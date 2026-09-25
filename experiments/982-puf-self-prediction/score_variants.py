"""Score the offline predictor variants (microcosm#982).

Usage::

    python score_variants.py <dir with imputed_<variant>.parquet> <variant> [...]

Weights are the PUF-clone half's weights times two, i.e. "as if this half were
the whole population". AGI here is a PROXY: the sum of the 15 imputed income
items (no adjustments, no Social Security, no capital-gains distributions), so
the band table is an approximate, pre-calibration comparison with SOI.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

# SOI Table 1.1, TY2023, all returns: band -> (low, high, returns, AGI $bn), as
# compiled into irs_soi.ty2023.table_1_1.* targets on the #960 branch.
SOI_TABLE_1_1 = {
    "1m_to_1_5m": (1e6, 1.5e6, 368_931, 443.658),
    "1_5m_to_2m": (1.5e6, 2e6, 147_290, 252.903),
    "2m_to_5m": (2e6, 5e6, 203_229, 602.653),
    "5m_to_10m": (5e6, 1e7, 49_262, 336.335),
    "10m_plus": (1e7, np.inf, 30_382, 907.901),
}
# SOI Table 1.4 shares of AGI: band -> (wages %, net capital gains %), from
# experiments/958-us-top-income-tail/results/composition_*_vs_soi_table_1_4.json.
SOI_TABLE_1_4 = {
    "1m_to_1_5m": (47.6, 12.2),
    "1_5m_to_2m": (40.3, 14.4),
    "2m_to_5m": (33.1, 18.1),
    "5m_to_10m": (26.3, 24.2),
    "10m_plus": (17.0, 39.5),
}
# Colorado $1M+ reference from microcosm#940 (SOI Historic Table 2, TY2023).
COLORADO_SOI_MEAN_AGI = 3.03e6
COLORADO_SOI_INCOME_ABOVE_1M_BN = 31.7
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


def score(d: pd.DataFrame) -> dict[str, object]:
    w = d["w"].to_numpy() * 2
    agi = d[INCOME].sum(axis=1).to_numpy()
    report: dict[str, object] = {"total_proxy_agi_tn": float((w * agi).sum() / 1e12)}
    bands = {}
    for band, (low, high, returns, soi_agi) in SOI_TABLE_1_1.items():
        mask = (agi >= low) & (agi < high)
        total = (w * agi)[mask].sum()
        wages = (w * d["employment_income_before_lsr"].to_numpy())[mask].sum()
        gains = (
            w
            * (
                d["long_term_capital_gains_before_response"]
                + d["short_term_capital_gains"]
            ).to_numpy()
        )[mask].sum()
        bands[band] = {
            "returns": float(w[mask].sum()),
            "soi_returns": returns,
            "agi_bn": float(total / 1e9),
            "soi_agi_bn": soi_agi,
            "records": int(mask.sum()),
            "wages_pct": float(100 * wages / total) if total else None,
            "soi_wages_pct": SOI_TABLE_1_4[band][0],
            "capital_gains_pct": float(100 * gains / total) if total else None,
            "soi_capital_gains_pct": SOI_TABLE_1_4[band][1],
        }
    report["bands"] = bands

    top = agi >= 1e7
    contributions = np.sort((w * agi)[top])[::-1]
    report["top_concentration"] = {
        "records_10m_plus": int(top.sum()),
        "top1_share": float(contributions[:1].sum() / contributions.sum()),
        "top3_share": float(contributions[:3].sum() / contributions.sum()),
        "max_proxy_agi": float(agi.max()),
    }

    colorado = d["state_fips"].to_numpy() == 8
    colorado_top = colorado & (agi >= 1e6)
    above = w * np.maximum(agi - 1e6, 0)
    report["colorado_1m_plus"] = {
        "returns": float(w[colorado_top].sum()),
        "records": int(colorado_top.sum()),
        "mean_proxy_agi": float(
            (w * agi)[colorado_top].sum() / max(w[colorado_top].sum(), 1)
        ),
        "soi_mean_agi": COLORADO_SOI_MEAN_AGI,
        "income_above_1m_bn": float(above[colorado].sum() / 1e9),
        "soi_income_above_1m_bn": COLORADO_SOI_INCOME_ABOVE_1M_BN,
        "largest_record_share_of_income_above_1m": float(
            above[colorado].max() / max(above[colorado].sum(), 1)
        ),
    }

    survey_wages = d["survey_puf_predictor_employment_income"].to_numpy(float)
    survey_se = d["survey_puf_predictor_self_employment_income"].to_numpy(float)
    wages = d["employment_income_before_lsr"].to_numpy(float)
    positive = survey_wages > 0
    survey_zero = survey_wages == 0
    survey_non_earner = (survey_wages == 0) & (survey_se == 0)
    report["wages_self_prediction"] = {
        "rank_correlation_with_survey": float(
            pd.Series(survey_wages).rank().corr(pd.Series(wages).rank())
        ),
        "share_within_10pct_given_survey_positive": float(
            np.mean(
                np.abs(wages[positive] - survey_wages[positive])
                <= 0.10 * survey_wages[positive]
            )
        ),
        "survey_max": float(survey_wages.max()),
        "imputed_max": float(wages.max()),
        "records_above_survey_max": int((wages > survey_wages.max()).sum()),
    }
    report["wages_participation"] = {
        "survey_positive_share": float(w[positive].sum() / w.sum()),
        "imputed_positive_share": float(w[wages > 0].sum() / w.sum()),
        "survey_zero_imputed_positive": float(
            w[survey_zero & (wages > 0)].sum() / w[survey_zero].sum()
        ),
        "survey_non_earner_imputed_positive_wages": float(
            w[survey_non_earner & (wages > 0)].sum() / w[survey_non_earner].sum()
        ),
        "survey_positive_imputed_zero": float(
            w[positive & (wages == 0)].sum() / w[positive].sum()
        ),
        "survey_total_wages_tn": float((w * survey_wages).sum() / 1e12),
        "imputed_total_wages_tn": float((w * wages).sum() / 1e12),
    }
    report["positive_shares"] = {
        column: float(w[d[column].to_numpy() > 0].sum() / w.sum())
        for column in (
            "qualified_dividend_income",
            "taxable_interest_income",
            "long_term_capital_gains_before_response",
            "partnership_income",
        )
    }
    return report


def main() -> None:
    directory, variants = sys.argv[1], sys.argv[2:]
    scores = {
        variant: score(pd.read_parquet(f"{directory}/imputed_{variant}.parquet"))
        for variant in variants
    }
    print(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
