#!/usr/bin/env python
"""Which predictor's SCF-vs-CPS shift drives the host borrower gap (microcosm#1073).

Fits the stage's household-total QRF once on the SCF 2025 donor, draws the CPS
host under the stage's recipient definition, then redraws it with one
predictor at a time replaced by its SCF counterpart at the same weighted rank
within the same five-year reference-person age bin (a quantile mapping), and
with all of them replaced. A predictor whose mapping closes the gap is one the
two surveys distribute differently at equal rank. Also writes each predictor's
weighted distribution by age band. Output:
``experiments/receipts/1073-predictor-shift-diagnosis.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.scf_education_debt import (
    SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS,
    SCF_EDUCATION_DEBT_TOTAL_TARGET,
    RecipientDefinition,
    _recipients,
    _stage_seed,
    load_scf_2025_education_debt_donor,
)

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "receipt_1073", _HERE / "build_1073_scf_education_debt_receipt.py"
)
receipt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(receipt)

#: The construction this diagnosis measured: the SCF-matching codings with
#: whole-household income, before the predictor and income-unit remedies it led
#: to (receipt run 20261020T150445Z-47949111f). Pinned so a rerun keeps its
#: meaning now that the stage default has moved on.
DIAGNOSED_DEFINITION = RecipientDefinition(income_unit="household")
BANDS = (("18_24", 18, 24), ("25_34", 25, 34), ("35_49", 35, 49),
         ("50_61", 50, 61), ("62_plus", 62, 200))  # fmt: skip
CONTINUOUS = (
    "employment_income",
    "interest_dividend_income",
    "social_security_pension_income",
    "household_income",
    "own_children_in_household",
)


def wq(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(values[order][min(index, len(values) - 1)])


def describe(values: np.ndarray, weights: np.ndarray, *, binary: bool) -> dict:
    if binary:
        return {"share": float(weights[values > 0].sum() / weights.sum())}
    return {
        "share_positive": float(weights[values > 0].sum() / weights.sum()),
        **{
            f"p{int(q * 100)}": wq(values, weights, q)
            for q in (0.25, 0.5, 0.75, 0.9, 0.99)
        },
    }


def weighted_rank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mid-rank of each value in its weighted distribution, ties sharing a rank."""

    frame = pd.DataFrame({"v": values, "w": weights})
    by_value = frame.groupby("v")["w"].sum().sort_index()
    upper = by_value.cumsum() / by_value.sum()
    mid = upper - by_value / by_value.sum() / 2.0
    return mid.reindex(values).to_numpy()


def quantile_map(
    host: np.ndarray,
    host_w: np.ndarray,
    host_age: np.ndarray,
    scf: np.ndarray,
    scf_w: np.ndarray,
    scf_age: np.ndarray,
) -> np.ndarray:
    """Replace each host value by the SCF value at its rank in its age bin."""

    out = host.copy()
    bins = np.arange(15, 100, 5)
    host_bin = np.digitize(host_age, bins)
    scf_bin = np.digitize(scf_age, bins)
    for b in np.unique(host_bin):
        h = host_bin == b
        s = scf_bin == b
        if s.sum() < 20:
            s = np.abs(scf_bin - b) <= 1
        ranks = weighted_rank(host[h], host_w[h])
        order = np.argsort(scf[s], kind="mergesort")
        cumulative = np.cumsum(scf_w[s][order]) / scf_w[s].sum()
        index = np.searchsorted(cumulative, ranks, side="left")
        out[h] = scf[s][order][np.minimum(index, s.sum() - 1)]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument(
        "--scf-cache-dir", type=Path, default=Path.home() / ".cache/microcosm/scf"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/receipts/1073-predictor-shift-diagnosis.json"),
    )
    args = parser.parse_args(argv)

    from microcosm.fit import QRF

    donor = load_scf_2025_education_debt_donor(
        args.scf_cache_dir / "rscfp2025.dta", args.scf_cache_dir / "p25i6.dta"
    )
    frame = receipt.load_host_frame(args.h5)
    host = _recipients(frame, DIAGNOSED_DEFINITION).features
    host_w = np.asarray(frame.weights_for("household").values, dtype=float)
    host_age = host["age"].to_numpy(float)
    scf_w = donor["scf_weight"].to_numpy(float)
    scf_age = donor["age"].to_numpy(float)

    report: dict[str, object] = {"distributions": {}, "counterfactual_shares": {}}
    for band, lo, hi in BANDS:
        h = (host_age >= lo) & (host_age <= hi)
        s = (scf_age >= lo) & (scf_age <= hi)
        entry = {}
        for column in SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS:
            if column == "age":
                continue
            binary = column in ("is_female", "is_married")
            if column == "cps_race":
                entry[column] = {
                    "scf": {
                        str(int(c)): float(scf_w[s][donor[column].to_numpy()[s] == c].sum() / scf_w[s].sum())
                        for c in (1, 2, 3, 4, 7)
                    },
                    "cps": {
                        str(int(c)): float(host_w[h][host[column].to_numpy()[h] == c].sum() / host_w[h].sum())
                        for c in (1, 2, 3, 4, 7)
                    },
                }  # fmt: skip
                continue
            entry[column] = {
                "scf": describe(donor[column].to_numpy(float)[s], scf_w[s], binary=binary),
                "cps": describe(host[column].to_numpy(float)[h], host_w[h], binary=binary),
            }  # fmt: skip
        report["distributions"][band] = entry

    predictors = list(SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS)
    fit = donor.loc[:, [*predictors, SCF_EDUCATION_DEBT_TOTAL_TARGET, "scf_weight"]]
    model = QRF(n_estimators=100, seed=_stage_seed(0, 1)).fit(
        fit,
        predictors=predictors,
        targets=[SCF_EDUCATION_DEBT_TOTAL_TARGET],
        weights="scf_weight",
    )

    def shares(features: pd.DataFrame) -> dict[str, float]:
        drawn = model.predict(features.loc[:, predictors])[
            SCF_EDUCATION_DEBT_TOTAL_TARGET
        ].to_numpy()
        positive = drawn > 0
        out = {"all": float(host_w[positive].sum() / host_w.sum())}
        for band, lo, hi in BANDS:
            b = (host_age >= lo) & (host_age <= hi)
            out[band] = float(host_w[b & positive].sum() / host_w[b].sum())
        return out

    scf_positive = donor[SCF_EDUCATION_DEBT_TOTAL_TARGET].to_numpy() > 0
    report["scf_observed_shares"] = {
        "all": float(scf_w[scf_positive].sum() / scf_w.sum()),
        **{
            band: float(
                scf_w[scf_positive & (scf_age >= lo) & (scf_age <= hi)].sum()
                / scf_w[(scf_age >= lo) & (scf_age <= hi)].sum()
            )
            for band, lo, hi in BANDS
        },
    }
    report["counterfactual_shares"]["host_as_is"] = shares(host)
    mapped_all = host.copy()
    for column in CONTINUOUS:
        mapped = host.copy()
        mapped[column] = quantile_map(
            host[column].to_numpy(float),
            host_w,
            host_age,
            donor[column].to_numpy(float),
            scf_w,
            scf_age,
        )
        mapped_all[column] = mapped[column]
        report["counterfactual_shares"][f"map_{column}"] = shares(mapped)
        print(
            column,
            json.dumps(report["counterfactual_shares"][f"map_{column}"]),
            flush=True,
        )
    report["counterfactual_shares"]["map_all_continuous"] = shares(mapped_all)
    print("host", json.dumps(report["counterfactual_shares"]["host_as_is"]))
    print("all", json.dumps(report["counterfactual_shares"]["map_all_continuous"]))
    print("scf", json.dumps(report["scf_observed_shares"]))
    for band in ("25_34", "35_49"):
        print(band, json.dumps(report["distributions"][band], default=float))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
