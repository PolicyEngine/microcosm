"""Real-pool proxy-AGI bands at frame weight (microcosm#982 correction).

Usage::

    python actual_pool_bands.py <dir with imputed_<variant>.parquet>

``score_variants.py`` weights the PUF-clone half x2 ("as if this half were the
whole population"). The real pool is the survey half (clone index 0, survey
values, half the weight) plus the PUF-clone half (clone index 1, imputed
values, the other half), before main's capital-gains tail stage. This script
combines both halves at frame weight on the same proxy AGI and compares the
bands with SOI Table 1.1. The survey half sums whichever of the 15 proxy-AGI
items its person table carries; ``clone0_items_used`` in the output lists them.

Writes ``actual_pool_bands.json`` into the input directory and prints a
summary. Inputs are restricted build artifacts on the author's machine; the
checkpoint path is recorded as provenance, not as a portable interface.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from score_variants import COLORADO_SOI_INCOME_ABOVE_1M_BN, INCOME, SOI_TABLE_1_1

from microcosm.build.frame_checkpoint import load_frame_checkpoint

warnings.filterwarnings("ignore")

CHECKPOINTS = os.path.expanduser(
    "~/spm-rebuild-20260908/rollout/codex-continuation-20260912/"
    "genuine-build-76325eb7-20260912/base-checkpoints"
)
VARIANTS = ("current", "demographic_midrank6_earn")
COLORADO_FIPS = 8


def survey_half(frame) -> tuple[pd.DataFrame, float, list[str]]:
    """Clone index 0 at frame weight: survey proxy AGI, weight and state per unit.

    Also returns clone 0's share of all tax-unit weight and the income items
    found on the person table.
    """

    person, tax_unit = frame.table("person"), frame.table("tax_unit")
    household = frame.table("household").copy()
    household["w"] = frame.resolve_weights("household").values
    clone = tax_unit["tax_unit_support_clone_index"].to_numpy()
    have = [column for column in INCOME if column in person.columns]
    survey = (
        person[have]
        .astype(float)
        .sum(axis=1)
        .groupby(person["person_tax_unit_id"])
        .sum()
    )
    unit_household = (
        person.groupby("person_tax_unit_id")["person_household_id"]
        .first()
        .reindex(tax_unit["tax_unit_id"])
        .to_numpy()
    )
    by_household = household.set_index("household_id")
    weight = by_household["w"].reindex(unit_household).to_numpy()
    state = by_household["state_fips"].reindex(unit_household).to_numpy()
    zero = clone == 0
    half = pd.DataFrame(
        {
            "agi": survey.reindex(tax_unit["tax_unit_id"][zero]).fillna(0).to_numpy(),
            "w": weight[zero],
            "state": state[zero],
            "clone": 0,
        }
    )
    return half, float(weight[zero].sum() / weight.sum()), have


def pool_report(pool: pd.DataFrame) -> dict[str, object]:
    """SOI Table 1.1 bands and Colorado's income above $1M for one pool."""

    bands = {}
    for band, (low, high, returns, soi_agi) in SOI_TABLE_1_1.items():
        mask = (pool.agi >= low) & (pool.agi < high)
        bands[band] = {
            "returns": float(pool.w[mask].sum()),
            "soi_returns": returns,
            "agi_bn": float((pool.w * pool.agi)[mask].sum() / 1e9),
            "soi_agi_bn": soi_agi,
            "records": int(mask.sum()),
            "clone1_returns": float(pool.w[mask & (pool.clone == 1)].sum()),
        }
    colorado = pool.state == COLORADO_FIPS
    above = pool.w * np.maximum(pool.agi - 1e6, 0)
    return {
        "bands": bands,
        "colorado_income_above_1m_bn": float(above[colorado].sum() / 1e9),
        "colorado_soi_bn": COLORADO_SOI_INCOME_ABOVE_1M_BN,
        "colorado_largest_record_share": float(
            above[colorado].max() / above[colorado].sum()
        ),
    }


def main() -> None:
    directory = sys.argv[1]
    frame = load_frame_checkpoint(
        f"{CHECKPOINTS}/002_clone_feature_extraction.frame.h5"
    ).frame
    survey, survey_share, items = survey_half(frame)
    out = {
        "clone0_weight_share": survey_share,
        "clone0_items_used": items,
        "variants": {},
    }
    for variant in VARIANTS:
        imputed = pd.read_parquet(f"{directory}/imputed_{variant}.parquet")
        clone1 = pd.DataFrame(
            {
                "agi": imputed[INCOME].sum(axis=1).to_numpy(),
                "w": imputed["w"].to_numpy(),
                "state": imputed["state_fips"].to_numpy(),
                "clone": 1,
            }
        )
        pool = pd.concat([survey, clone1], ignore_index=True)
        out["variants"][variant] = pool_report(pool)
    with open(f"{directory}/actual_pool_bands.json", "w") as handle:
        json.dump(out, handle, indent=1)

    print(f"clone0 weight share {out['clone0_weight_share']:.3f}")
    for variant, report in out["variants"].items():
        print("==", variant)
        for band, x in report["bands"].items():
            print(
                f"  {band:12} returns {x['returns']:>9,.0f} vs SOI "
                f"{x['soi_returns']:>8,} ({x['returns'] / x['soi_returns']:.2f}x)  "
                f"AGI ${x['agi_bn']:6.0f}B vs ${x['soi_agi_bn']:4.0f}B  "
                f"records {x['records']:5}  clone1 {x['clone1_returns']:>9,.0f}"
            )
        print(
            "  Colorado income above $1M "
            f"${report['colorado_income_above_1m_bn']:.1f}B vs SOI "
            f"${report['colorado_soi_bn']:.1f}B; largest record share "
            f"{report['colorado_largest_record_share']:.0%}"
        )


if __name__ == "__main__":
    main()
