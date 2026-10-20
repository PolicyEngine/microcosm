#!/usr/bin/env python
"""Host fit under each CPS construction of the SCF predictors (microcosm#1073).

Each variant changes one field of ``RecipientDefinition`` from the one before,
for a stated reason (``VARIANTS``), and imputes the published default with the
same seed and donor through the production code path. Writes
``experiments/receipts/1073-recipient-definition-variants.json``.

    .venv/Scripts/python.exe experiments/explore_1073_recipient_definitions.py \
        --h5 populace_us_2024.h5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.scf_education_debt import (
    SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS,
    SCF_EDUCATION_DEBT_FIRST_RECEIPT_DEFINITION,
    SCF_EDUCATION_DEBT_PREDICTORS,
    SCF_EDUCATION_DEBT_TOTAL_TARGET,
    RecipientDefinition,
    _recipients,
    impute_us_scf_education_debt,
    load_scf_2025_education_debt_donor,
    scf_education_debt_donor_fit_table,
    us_scf_education_debt_fit_table,
)
from microcosm.frame import Frame

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "receipt_1073", _HERE / "build_1073_scf_education_debt_receipt.py"
)
receipt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(receipt)

FIRST = SCF_EDUCATION_DEBT_FIRST_RECEIPT_DEFINITION
_V1 = replace(FIRST, partnered="spouse_or_partner")
_V2 = replace(_V1, children="of_reference_or_partner")
_V3 = replace(_V2, reference_person="householder_couple")
VARIANTS: tuple[tuple[str, RecipientDefinition, str], ...] = (
    ("v0_first_receipt", FIRST, "The first receipt's construction."),
    (
        "v1_partner_counts_as_married",
        _V1,
        "SCF MARRIED is married or living with a partner (X8023 in 1-2); "
        "A_MARITL 1-2 drops cohabiting partners.",
    ),
    (
        "v2_children_of_reference_or_partner",
        _V2,
        "SCF KIDS counts children of the reference person or spouse/partner, "
        "any age, anywhere in the household listing.",
    ),
    (
        "v3_householder_couple_reference_person",
        _V3,
        "The SCF reference person comes from the economically dominant "
        "individual or couple (the man, or the older partner); the CPS "
        "householder anchors that unit, where the oldest adult can be a "
        "co-resident parent or roommate.",
    ),
    (
        "v4a_spm_unit_income",
        replace(_V3, income_unit="spm_unit"),
        "SCF income covers the primary economic unit; the SPM unit is the CPS "
        "resource-sharing unit.",
    ),
    (
        "v4b_couple_and_children_income",
        replace(_V3, income_unit="reference_couple_and_children"),
        "The PEU narrowed to the couple and their children.",
    ),
)
_ALL = tuple(SCF_EDUCATION_DEBT_PREDICTORS)
_NO_INTDIV = tuple(p for p in _ALL if p != "interest_dividend_income")
_NO_INTDIV_SSRET = tuple(p for p in _NO_INTDIV if p != "social_security_pension_income")
#: Predictor-set remedies on top of the couple income unit. Each row also gets
#: an SCF hold-out so the dropped predictor's own contribution is measured.
PREDICTOR_VARIANTS: tuple[
    tuple[str, RecipientDefinition, tuple[str, ...], str], ...
] = (
    (
        "r1_couple_income_drop_interest_dividends",
        replace(_V3, income_unit="reference_couple_and_children"),
        _NO_INTDIV,
        "CPS interest and dividends are PUF-imputed at tax-unit level; at "
        "reference ages 25-34, 42% of host households have some against 16% "
        "of SCF families, so equal values do not mean equal positions.",
    ),
    (
        "r2_couple_income_drop_interest_dividends_and_ss_pension",
        replace(_V3, income_unit="reference_couple_and_children"),
        _NO_INTDIV_SSRET,
        "As r1, and Social Security or pension income: 18% of host households "
        "at 25-34 have some against 6% of SCF families.",
    ),
    (
        "r3_household_income_drop_interest_dividends_and_ss_pension",
        _V3,
        _NO_INTDIV_SSRET,
        "As r2 with the household income unit, to separate the two changes.",
    ),
)


def parent_plus_by_age(
    head_age: np.ndarray, parent_plus: np.ndarray, total: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    out = {}
    for label, low, high in SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS:
        inside = (head_age >= low) & (head_age <= high)
        denominator = float(np.dot(total[inside], weight[inside]))
        out[label] = (
            float(np.dot(parent_plus[inside], weight[inside])) / denominator
            if denominator > 0
            else float("nan")
        )
    return out


def host_parent_plus_by_age(frame: Frame, definition: RecipientDefinition) -> dict:
    person = frame.table("person")
    recipients = _recipients(frame, definition)
    household_ids = person["person_household_id"].to_numpy()
    order = frame.table("household")["household_id"].to_numpy()
    own = person["federal_student_loan_balance"].to_numpy(dtype=float)
    plus = person["federal_parent_plus_loan_balance"].to_numpy(dtype=float)
    total = (
        pd.Series(own + plus, index=household_ids)
        .groupby(level=0)
        .sum()
        .reindex(order)
        .to_numpy()
    )
    plus_hh = (
        pd.Series(plus, index=household_ids).groupby(level=0).sum().reindex(order)
    ).to_numpy()
    return parent_plus_by_age(
        recipients.features["age"].to_numpy(dtype=float),
        plus_hh,
        total,
        np.asarray(frame.weights_for("household").values, dtype=float),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument(
        "--scf-cache-dir", type=Path, default=Path.home() / ".cache/microcosm/scf"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/receipts/1073-recipient-definition-variants.json"),
    )
    args = parser.parse_args(argv)

    donor = load_scf_2025_education_debt_donor(
        args.scf_cache_dir / "rscfp2025.dta", args.scf_cache_dir / "p25i6.dta"
    )
    observed = scf_education_debt_donor_fit_table(donor)
    scf_parent_plus = parent_plus_by_age(
        donor["age"].to_numpy(float),
        donor["federal_parent_plus_loan_balance"].to_numpy(float),
        donor[SCF_EDUCATION_DEBT_TOTAL_TARGET].to_numpy(float),
        donor["scf_weight"].to_numpy(float),
    )
    base = receipt.load_host_frame(args.h5)
    results = {
        "scf_2025": {
            "share": observed["households"]["share_with_federal_education_debt"],
            "parent_plus_share": observed["households"]["parent_plus_share_of_total"],
            "parent_plus_by_reference_age": scf_parent_plus,
            "by_head_age_band": {
                k: v["share_with_federal_education_debt"]
                for k, v in observed["by_head_age_band"].items()
            },
            "by_income_quintile": {
                k: v["share_with_federal_education_debt"]
                for k, v in observed["by_income_quintile"].items()
            },
        },
        "variants": {},
    }
    holdout: dict[tuple[str, ...], dict] = {}
    runs = [(n, d, _ALL, r) for n, d, r in VARIANTS] + list(PREDICTOR_VARIANTS)
    for name, definition, predictors, reason in runs:
        if predictors not in holdout:
            drawn = receipt.holdout_draws(
                donor,
                predictors,
                folds=5,
                seed=args.seed,
                n_estimators=args.n_estimators,
            )
            holdout_fit = scf_education_debt_donor_fit_table(donor, imputed=drawn)
            holdout[predictors] = {
                "predictors": list(predictors),
                "share": holdout_fit["households"]["share_with_federal_education_debt"],
                "parent_plus_share": holdout_fit["households"][
                    "parent_plus_share_of_total"
                ],
                "metrics_vs_scf": receipt.fit_metrics(observed, holdout_fit),
            }
            print("holdout", json.dumps(holdout[predictors], default=float), flush=True)
        imputed = impute_us_scf_education_debt(
            base,
            donor,
            seed=args.seed,
            n_estimators=args.n_estimators,
            definition=definition,
            predictors=predictors,
        )
        tables = {entity: base.table(entity).copy() for entity in base.entities}
        for column in imputed.columns:
            tables["person"][column] = imputed[column].to_numpy()
        frame = Frame(
            tables,
            base.schema,
            {entity: base.weights_for(entity) for entity in base.weighted_entities},
        )
        fit = us_scf_education_debt_fit_table(frame, definition)
        metrics = receipt.fit_metrics(observed, fit)
        entry = {
            "definition": asdict(definition),
            "predictors": list(predictors),
            "scf_holdout": holdout[predictors],
            "reason": reason,
            "share": fit["households"]["share_with_federal_education_debt"],
            "parent_plus_share": fit["households"]["parent_plus_share_of_total"],
            "parent_plus_by_reference_age": host_parent_plus_by_age(frame, definition),
            "by_head_age_band": {
                k: v["share_with_federal_education_debt"]
                for k, v in fit["by_head_age_band"].items()
            },
            "by_income_quintile": {
                k: v["share_with_federal_education_debt"]
                for k, v in fit["by_income_quintile"].items()
            },
            "metrics_vs_scf": metrics,
            "fit": fit,
        }
        results["variants"][name] = entry
        print(
            name,
            json.dumps(
                {
                    "share": round(entry["share"], 4),
                    "pp": round(entry["parent_plus_share"], 4),
                    "age": {
                        k: round(v, 3) for k, v in entry["by_head_age_band"].items()
                    },
                    "inc": {
                        k: round(v, 3) for k, v in entry["by_income_quintile"].items()
                    },
                    "pp_age": {
                        k: round(v, 3)
                        for k, v in entry["parent_plus_by_reference_age"].items()
                    },
                    "age_mae": round(metrics["head_age_band_share_mae_pp"], 2),
                    "inc_mae": round(metrics["income_quintile_share_mae_pp"], 2),
                }
            ),
            flush=True,
        )
    print("scf", json.dumps(results["scf_2025"], default=float))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, indent=2, default=float) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
