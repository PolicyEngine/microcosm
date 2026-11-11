"""Offline predictor-design comparison for the US PUF QRF (microcosm#982).

Fits the production QRF (``microcosm.fit.QRF``, seed 0, design weights) on the
saved 2026-09-12 genuine-build donor frame and predicts onto the saved PUF-clone
recipients, once per predictor variant. Output: one parquet of imputed
tax-unit totals per variant, plus the survey values the scorer compares with.

Usage::

    N_TREES=32 python compare_predictors.py <variant> <out.parquet>

Variants:

- ``current``: the pre-#982 production list (filing status, person count and
  the recipient's six survey income items), pinned literally below.
- ``demographic_midrank6``: filing status, person count, head age, spouse age,
  head sex, dependent count and the weighted mid-rank share of the six-item
  income total, each side ranked within its own population.
- ``demographic_midrank6_earn``: the above plus an earnings-participation
  indicator (nonzero wages or self-employment income), same definition on both
  sides.

Inputs are restricted build artifacts on the author's machine; the paths are
recorded here as provenance, not as a portable interface.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime import puf_support as support
from microcosm.fit import QRF

warnings.filterwarnings("ignore")

CHECKPOINTS = os.path.expanduser(
    "~/spm-rebuild-20260908/rollout/codex-continuation-20260912/"
    "genuine-build-76325eb7-20260912/base-checkpoints"
)
PUF = os.path.expanduser(
    "~/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5"
)
N_TREES = int(os.environ.get("N_TREES", "32"))

TARGETS = [
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
# The pre-#982 production predictor list, pinned so "current" means the same
# thing whichever checkout's environment runs this script.
INCOME8 = [
    "puf_predictor_filing_status_code",
    "puf_predictor_tax_unit_person_count",
    "puf_predictor_employment_income",
    "puf_predictor_self_employment_income",
    "puf_predictor_taxable_interest_income",
    "puf_predictor_dividend_income",
    "puf_predictor_short_term_capital_gains",
    "puf_predictor_long_term_capital_gains",
]
SIX = INCOME8[2:]
# Broad survey income total, used only by the scorer's coherence metric.
SURVEY = [
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "long_term_capital_gains_before_response",
    "short_term_capital_gains",
    "taxable_private_pension_income",
    "taxable_ira_distributions",
    "rental_income",
    "farm_operations_income",
    "unemployment_compensation",
    "alimony_income",
    "miscellaneous_income",
]
BASE = ["puf_predictor_filing_status_code", "puf_predictor_tax_unit_person_count"]
DEMO = [*BASE, "head_age", "head_male", "spouse_age", "n_dependents"]
VARIANTS = {
    "current": INCOME8,
    "demographic_midrank6": [*DEMO, "income_rank_share"],
    "demographic_midrank6_earn": [*DEMO, "income_rank_share", "has_earnings"],
    "demographic_stratrank6_earn": [*DEMO, "earnings_group_rank_share", "has_earnings"],
}


def mid_share(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mid-rank from the top: weight strictly above plus half the tie group."""

    order = np.argsort(-values, kind="stable")
    sorted_values, sorted_weights = values[order], weights[order]
    before = np.cumsum(sorted_weights) - sorted_weights
    first = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    group = np.cumsum(first) - 1
    tie_weight = np.bincount(group, weights=sorted_weights)[group]
    out = np.empty_like(values, dtype=float)
    out[order] = (before[first][group] + tie_weight / 2) / weights.sum()
    return out


def group_mid_share(
    values: np.ndarray, weights: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Mid-rank share computed separately within each group."""

    out = np.empty(len(values), dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        out[mask] = mid_share(values[mask], weights[mask])
    return out


def demographics(people: pd.DataFrame) -> pd.DataFrame:
    heads = people[people["is_head"]].groupby("tu")
    spouses = people[people["is_spouse"]].groupby("tu")
    dependents = people[people["is_dep"]].groupby("tu")
    return pd.DataFrame(
        {
            "head_age": heads["age"].first(),
            "head_male": heads["male"].first(),
            "spouse_age": spouses["age"].first(),
            "n_dependents": dependents.size(),
        }
    ).reindex(people.groupby("tu").size().index)


def main() -> None:
    variant, out = sys.argv[1], sys.argv[2]
    predictors = VARIANTS[variant]
    started = time.time()
    donor = load_frame_checkpoint(f"{CHECKPOINTS}/primary_qrf/donor.frame.h5").frame
    recipient = load_frame_checkpoint(
        f"{CHECKPOINTS}/primary_qrf/recipient.frame.h5"
    ).frame
    donor_units = donor.table("tax_unit").copy()
    donor_units["weight"] = donor.resolve_weights("tax_unit").values
    recipient_units = recipient.table("tax_unit").copy()
    outputs = [
        target
        for target in (
            *support.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
            *support.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
        )
        if target in TARGETS
    ]

    # Donor demographics from the processed PUF person arrays. The saved donor
    # frame renumbers tax units 1..N in PUF file order (wages, long-term gains
    # and filing status match by position exactly), so join by position.
    with h5py.File(PUF, "r") as puf:
        people = pd.DataFrame(
            {
                "tu": puf["person_tax_unit_id"][()],
                "age": puf["age"][()].astype(float),
                "male": puf["is_male"][()].astype(float),
                "is_head": puf["is_tax_unit_head"][()].astype(bool),
                "is_spouse": puf["is_tax_unit_spouse"][()].astype(bool),
                "is_dep": puf["is_tax_unit_dependent"][()].astype(bool),
            }
        )
        puf_tax_unit_ids = puf["tax_unit_id"][()]
    donor_demo = demographics(people).reindex(puf_tax_unit_ids).reset_index(drop=True)
    assert donor_demo["head_age"].notna().mean() > 0.99

    # Recipient demographics, survey income, weights and state from the clone
    # checkpoint.
    clone = load_frame_checkpoint(
        f"{CHECKPOINTS}/002_clone_feature_extraction.frame.h5"
    ).frame
    person = clone.table("person")
    household = clone.table("household").copy()
    household["w"] = clone.resolve_weights("household").values
    role = person["tax_unit_role_input"].astype(str).str.upper()
    recipient_people = pd.DataFrame(
        {
            "tu": person["person_tax_unit_id"].to_numpy(),
            "age": person["age"].astype(float).to_numpy(),
            "male": (1.0 - person["is_female"].astype(float)).to_numpy(),
            "is_head": (role == "HEAD").to_numpy(),
            "is_spouse": (role == "SPOUSE").to_numpy(),
            "is_dep": (role == "DEPENDENT").to_numpy(),
        }
    )
    recipient_ids = recipient_units["tax_unit_id"].to_numpy()
    recipient_demo = demographics(recipient_people).reindex(recipient_ids)
    have = [column for column in SURVEY if column in person.columns]
    survey_total = (
        person[have]
        .astype(float)
        .sum(axis=1)
        .groupby(person["person_tax_unit_id"])
        .sum()
        .reindex(recipient_ids)
        .fillna(0)
        .to_numpy()
    )
    unit_household = (
        person.groupby("person_tax_unit_id")["person_household_id"]
        .first()
        .reindex(recipient_ids)
        .to_numpy()
    )
    by_household = household.set_index("household_id")
    recipient_weight = by_household["w"].reindex(unit_household).to_numpy()
    state = by_household["state_fips"].reindex(unit_household).to_numpy()

    def features(demo: pd.DataFrame, units: pd.DataFrame, weight: np.ndarray):
        frame = units[BASE].reset_index(drop=True).copy()
        frame["head_age"] = demo["head_age"].fillna(0).to_numpy()
        frame["head_male"] = demo["head_male"].fillna(0).to_numpy()
        frame["spouse_age"] = demo["spouse_age"].fillna(0).to_numpy()
        frame["n_dependents"] = demo["n_dependents"].fillna(0).to_numpy()
        total = units[SIX].astype(float).sum(axis=1).to_numpy()
        frame["income_rank_share"] = mid_share(total, weight)
        # Earnings participation: nonzero wages or self-employment income.
        frame["has_earnings"] = (
            (units["puf_predictor_employment_income"].astype(float) != 0)
            | (units["puf_predictor_self_employment_income"].astype(float) != 0)
        ).to_numpy(dtype=float)
        frame["earnings_group_rank_share"] = group_mid_share(
            total, weight, frame["has_earnings"].to_numpy()
        )
        for column in SIX:
            frame[column] = units[column].to_numpy()
        return frame.set_index(units.index)

    donor_x = features(donor_demo, donor_units, donor_units["weight"].to_numpy(float))
    recipient_x = features(recipient_demo, recipient_units, recipient_weight)
    model = pd.concat(
        [
            donor_x[predictors],
            donor_units[outputs].astype(float).fillna(0.0),
            donor_units[["weight"]],
        ],
        axis=1,
    )
    model.columns = [*predictors, *outputs, "weight"]
    fitted = QRF(n_estimators=N_TREES, seed=0).fit(
        support._tax_unit_model_frame(model),
        list(predictors),
        list(outputs),
        weights="design",
    )
    predicted = fitted.predict(
        recipient_x[predictors].reset_index(drop=True), release_models=True
    )
    result = pd.DataFrame(
        {
            "tax_unit_id": recipient_ids,
            "w": recipient_weight,
            "state_fips": state,
            "survey_total": survey_total,
            **{
                f"survey_{column}": recipient_units[column].to_numpy() for column in SIX
            },
            **{column: predicted[column].to_numpy() for column in outputs},
        }
    )
    result.to_parquet(out)
    print(
        json.dumps(
            {
                "variant": variant,
                "n_trees": N_TREES,
                "predictors": predictors,
                "targets": outputs,
                "seconds": round(time.time() - started),
                "rows": len(result),
            }
        )
    )


if __name__ == "__main__":
    main()
