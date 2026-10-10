#!/usr/bin/env python3
"""Estimate an SSI receipt/own-liquid-assets association from pinned SIPP 2023.

This is an offline evidence tool, not a release builder.  It writes aggregate
estimates only.  The principal fit uses unmarried, apparently income-eligible
aged/disabled people with $0--$2,000 of own liquid assets.  A whole-range fit is
reported separately: above the limit receipt mixes claiming with statutory
eligibility and therefore cannot identify a claiming slope.  Extrapolating the
within-limit association beyond $2,000 remains an explicit modeling assumption.
No reform output, target recipient count, or CBPP outcome enters this estimator.

Run with the shared-machine memory admission wrapper, for example::

    heavy --mem 16 -- python tools/estimate_ssi_asset_gradient.py \
        --sipp /path/to/pu2023.csv --output /path/to/sipp_estimation.json \
        --packaged-output packages/microcosm-build/src/microcosm/build/us/ssi_asset_gradient.json

The 2023 survey describes calendar 2022.  December person records put the
resource observation and monthly receipt in the same period.  Receipt is
RSSI_MNYN == 1; RSSI_YRYN supplies an annual-receipt sensitivity.  SIPP assets
are collected only for people aged 15+, so the child coefficient is transferred
from the adult disabled band and is never represented as a child estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def source_specification() -> tuple[dict, dict]:
    """Read the existing donor pin/mappings without initializing the US engine."""
    path = (
        Path(__file__).resolve().parents[1]
        / "packages/microcosm-build/src/microcosm/build/us/source_stages.json"
    )
    stages = json.loads(path.read_text())["stages"]
    stage = next(row for row in stages if row["stage"] == "scf_wealth")
    artifact = next(row for row in stage["artifacts"] if row["member"] == "pu2023.csv")
    read = next(
        row for row in stage["operations"] if row.get("table") == "sipp_2023_person"
    )
    return artifact, read


DONOR_ARTIFACT, DONOR_READ = source_specification()
DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/sipp/tech-documentation/"
    "data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf"
)
ASSET_COLUMNS = tuple(DONOR_READ["targets"].values())
ASSET_ALLOCATION_COLUMNS = tuple(
    column
    for columns in DONOR_READ["target_allocation_status_columns"].values()
    for column in columns
)
DISABILITY_COLUMNS = ("EDISABL", "EJOBCANT", "EDISANY", "RDIS", "ESEEING")
DISABILITY_ALLOCATION_COLUMNS = ("ADISABL", "AJOBCANT", "ADISANY", "ADIS", "ASEEING")
EARNINGS_COLUMNS = tuple(f"TJB{i}_MSUM" for i in range(1, 8))
SOURCE_COLUMNS = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "WPFINWGT",
    "TAGE",
    "EMS",
    "TPTOTINC",
    "APTOTINC",
    "TSSI_AMT",
    "RSSI_MNYN",
    "ASSI_MNYN",
    "RSSI_YRYN",
    "ASSI_YRYN",
    *ASSET_COLUMNS,
    *ASSET_ALLOCATION_COLUMNS,
    *DISABILITY_COLUMNS,
    *DISABILITY_ALLOCATION_COLUMNS,
    *EARNINGS_COLUMNS,
)
OBSERVED_STATUSES = tuple(DONOR_READ["allocation_status_observed_values"])


def fit_weighted_logit(frame: pd.DataFrame, *, outcome: str = "receipt") -> dict:
    """Fit two-parameter survey-weighted logit; report household sandwich SEs.

    Weights normalize to mean one, which leaves coefficients and sandwich SEs
    unchanged.  The covariance clusters SSUID and uses a small-sample correction.
    It is not a full SIPP replicate-weight/design variance estimate.
    """
    if len(frame) < 3 or frame[outcome].nunique() != 2:
        return {"status": "not_estimable", "rows": int(len(frame))}
    assets = frame["liquid_assets"].to_numpy(dtype=float)
    x = np.column_stack((np.ones(len(frame)), np.log1p(assets)))
    if np.linalg.matrix_rank(x) != 2:
        return {"status": "not_estimable", "rows": int(len(frame))}
    y = frame[outcome].to_numpy(dtype=float)
    positive = assets[y == 1]
    negative = assets[y == 0]
    if negative.max() <= positive.min() or positive.max() <= negative.min():
        return {
            "status": "not_estimable",
            "rows": int(len(frame)),
            "reason": "complete or quasi-complete separation: one-predictor outcome ranges ordered; no finite unconstrained maximum likelihood estimate",
        }
    weights = frame["weight"].to_numpy(dtype=float, copy=True)
    weights /= weights.mean()

    def objective(beta):
        linear = x @ beta
        return float(np.sum(weights * (np.logaddexp(0.0, linear) - y * linear)))

    def jacobian(beta):
        return x.T @ (weights * (expit(x @ beta) - y))

    def hessian(beta):
        probability = expit(x @ beta)
        return x.T @ ((weights * probability * (1.0 - probability))[:, None] * x)

    start = np.array(
        [
            np.log(
                np.average(y, weights=weights) / (1 - np.average(y, weights=weights))
            ),
            0.0,
        ]
    )
    result = minimize(
        objective, start, jac=jacobian, hess=hessian, method="trust-exact"
    )
    if not result.success and np.max(np.abs(jacobian(result.x))) > 1e-5:
        raise ValueError(f"Logit did not converge: {result.message}")
    probability = expit(x @ result.x)
    bread = np.linalg.inv(hessian(result.x))
    score = x * (weights * (y - probability))[:, None]
    groups, group_ids = np.unique(frame["household"].astype(str), return_inverse=True)
    cluster_score = np.zeros((len(groups), 2))
    np.add.at(cluster_score, group_ids, score)
    correction = len(groups) / (len(groups) - 1) * (len(frame) - 1) / (len(frame) - 2)
    covariance = correction * bread @ (cluster_score.T @ cluster_score) @ bread
    standard_error = np.sqrt(np.diag(covariance))
    raw_weights = frame["weight"].to_numpy(dtype=float)
    return {
        "status": "estimated",
        "rows": int(len(frame)),
        "households": int(len(groups)),
        "receipt_rows": int(y.sum()),
        "weighted_population": float(raw_weights.sum()),
        "weighted_receipt_share": float(np.average(y, weights=raw_weights)),
        "effective_rows_kish": float(raw_weights.sum() ** 2 / np.sum(raw_weights**2)),
        "intercept": float(result.x[0]),
        "slope": float(result.x[1]),
        "intercept_standard_error": float(standard_error[0]),
        "slope_standard_error": float(standard_error[1]),
        "slope_95_percent_interval": [
            float(result.x[1] - 1.96 * standard_error[1]),
            float(result.x[1] + 1.96 * standard_error[1]),
        ],
        "covariance_method": "SSUID-clustered sandwich, finite cluster/row correction; no replicate weights",
        "minimum_assets": float(assets.min()),
        "maximum_assets": float(assets.max()),
        "positive_asset_rows": int((assets > 0).sum()),
        "score_max_abs": float(np.max(np.abs(jacobian(result.x)))),
    }


def prepare_sample(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Construct a conservative apparent-eligibility sample without a resource cut."""
    frame = raw.copy()
    for column in SOURCE_COLUMNS:
        if column != "SSUID":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["SSUID", "PNUM"]].duplicated().any():
        raise ValueError("December SIPP contains duplicate source-person identities")
    # SIPP TVAL_* assets are nonnegative but missing outside the age-15+ universe.
    # Missing assets never become measured zeros in the estimation sample.
    frame["liquid_assets"] = frame[list(ASSET_COLUMNS)].sum(axis=1, min_count=3)
    frame["household"] = frame["SSUID"].astype(str)
    frame["weight"] = frame["WPFINWGT"]
    frame["receipt"] = frame["RSSI_MNYN"].eq(1).astype(int)
    frame["annual_receipt"] = frame["RSSI_YRYN"].eq(1).astype(int)
    frame["age_band"] = np.select(
        [frame["TAGE"].lt(18), frame["TAGE"].lt(65)],
        ["under_18", "18_64"],
        default="65_plus",
    )
    # Broad independent disability proxy; the receipt label does not create ABD.
    disabled = frame[list(DISABILITY_COLUMNS)].eq(1).any(axis=1)
    aged_or_disabled = frame["TAGE"].ge(65) | disabled
    earned = frame[list(EARNINGS_COLUMNS)].fillna(0).clip(lower=0).sum(axis=1)
    # Remove SSI itself before the screen, so benefit receipt does not disqualify
    # otherwise low-income people.  $841 is the 2022 individual federal rate.
    non_ssi_income = (frame["TPTOTINC"] - frame["TSSI_AMT"].fillna(0)).clip(lower=0)
    unearned = (non_ssi_income - earned).clip(lower=0)
    general = np.minimum(unearned, 20.0)
    countable_income = (
        unearned - general + 0.5 * (earned - 65.0 - (20.0 - general)).clip(lower=0)
    )
    frame["countable_income"] = countable_income
    observed = frame["ASSI_MNYN"].isin(OBSERVED_STATUSES) & frame["RSSI_MNYN"].isin(
        (1, 2)
    )
    for column in (
        *ASSET_ALLOCATION_COLUMNS,
        *DISABILITY_ALLOCATION_COLUMNS,
        "APTOTINC",
    ):
        observed &= frame[column].isin(OBSERVED_STATUSES)
    finite = np.isfinite(
        frame[["liquid_assets", "weight", "TAGE", "countable_income"]]
    ).all(axis=1)
    unmarried = frame["EMS"].isin((3, 4, 5, 6))
    mask = (
        observed
        & finite
        & unmarried
        & aged_or_disabled
        & countable_income.le(841.0)
        & frame["TAGE"].ge(15)
        & frame["liquid_assets"].ge(0)
        & frame["weight"].gt(0)
    )
    diagnostics = {
        "december_rows": int(len(frame)),
        "observed_rows": int(observed.sum()),
        "aged_or_disabled_rows": int(aged_or_disabled.sum()),
        "apparent_income_eligible_rows": int(countable_income.le(841.0).sum()),
        "complete_unmarried_income_eligible_aged_or_disabled_rows": int(mask.sum()),
    }
    return frame.loc[mask].copy(), diagnostics


def estimate(raw: pd.DataFrame) -> dict:
    frame, sample_flow = prepare_sample(raw)
    fits = {}
    for band in ("under_18", "18_64", "65_plus"):
        band_frame = frame.loc[frame["age_band"].eq(band)]
        fits[band] = {
            "within_resource_limit": fit_weighted_logit(
                band_frame.loc[band_frame["liquid_assets"].le(2000)]
            ),
            "full_observed_range_confounded_by_eligibility": fit_weighted_logit(
                band_frame
            ),
            "positive_assets_within_limit": fit_weighted_logit(
                band_frame.loc[band_frame["liquid_assets"].between(1, 2000)]
            ),
            "within_limit_annual_receipt": fit_weighted_logit(
                band_frame.loc[band_frame["liquid_assets"].le(2000)],
                outcome="annual_receipt",
            ),
        }
    return {
        "schema_version": 1,
        "transform": "log1p(own bank + stock + bond assets, dollars)",
        "link": "logit",
        "sample_flow": sample_flow,
        "fits": fits,
        "sample_definition": "December 2022 SIPP person records age 15+, EMS in 3/4/5/6, independent reported aged/disability proxy, apparent countable non-SSI monthly income <= $841, observed receipt/income/asset/disability allocation codes 0/1/9, positive finite survey weight; primary own liquid assets <= $2,000",
        "identification_limits": [
            "Within-limit cross-sectional receipt association is not a causal claiming effect; broad disability proxy is not SSA medical eligibility.",
            "No spouse or parent deeming and no full non-liquid-resource screen; restrict primary sample to unmarried people.",
            "Extrapolation above $2,000 assumes the within-limit logit slope continues; full-range receipt slope is confounded with asset eligibility and is never used.",
            "Allocation recodes can conceal imputed components; explicit component asset flags are screened, but all income components are not.",
            "Household sandwich SEs do not reproduce SIPP stratified replicate-weight variance.",
            "Child assets are not measured under age 15; primary child slope transfers from disabled adults and child qualification remains fenced.",
        ],
    }


def packaged_summary(evidence: dict) -> dict:
    fits = evidence["fits"]
    # A transferred coefficient is explicit; it is not a fitted child result.
    bands = {}
    for band in ("under_18", "18_64", "65_plus"):
        source_band = "18_64" if band == "under_18" else band
        fit = fits[source_band]["within_resource_limit"]
        if fit["status"] != "estimated":
            raise ValueError(f"Primary coefficient is not estimable for {source_band}")
        bands[band] = {
            "slope": fit["slope"],
            "standard_error": fit["slope_standard_error"],
            "source_band": source_band,
            "sample_rows": fit["rows"],
            "role": "transferred disabled-adult coefficient; child qualification fence unchanged"
            if band == "under_18"
            else "within-limit SIPP association extrapolated above observed eligible range",
        }
    return {
        "schema_version": 1,
        "transform": evidence["transform"],
        "link": "logit",
        "source_sha256": DONOR_ARTIFACT["sha256"],
        "evidence_path": "docs/evidence/ssi-take-up-asset-gradient/sipp_estimation.json",
        "slopes": {band: row["slope"] for band, row in bands.items()},
        "bands": bands,
        "identification_limits": evidence["identification_limits"],
    }


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sipp", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--packaged-output", type=Path)
    parser.add_argument("--execution-note", default="")
    args = parser.parse_args()
    if args.sipp.stat().st_size != DONOR_ARTIFACT["size_bytes"]:
        raise ValueError("Input byte size differs from repository-pinned SIPP 2023")
    digest = hashlib.sha256()
    with args.sipp.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != DONOR_ARTIFACT["sha256"]:
        raise ValueError("Input digest differs from repository-pinned SIPP 2023")
    parts = []
    for chunk in pd.read_csv(
        args.sipp,
        sep=DONOR_READ["delimiter"],
        usecols=list(SOURCE_COLUMNS),
        chunksize=5000,
        low_memory=False,
        dtype={"SSUID": str},
    ):
        december = chunk.loc[
            pd.to_numeric(chunk[DONOR_READ["month_column"]], errors="coerce").eq(
                DONOR_READ["month"]
            )
        ]
        if len(december):
            parts.append(december)
    evidence = estimate(pd.concat(parts, ignore_index=True))
    evidence["source"] = {
        "survey": "Census SIPP 2023 (calendar 2022)",
        "sha256": DONOR_ARTIFACT["sha256"],
        "size_bytes": DONOR_ARTIFACT["size_bytes"],
        "mirror_revision": DONOR_ARTIFACT["revision"],
        "dictionary_url": DICTIONARY_URL,
        "dictionary_pages": {
            "TVAL_BANK": 190,
            "TVAL_BOND": 192,
            "TPTOTINC": 2871,
            "RSSI_MNYN": 3149,
        },
    }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    evidence["execution"] = {
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gib": rss / (2**30 if sys.platform == "darwin" else 2**20),
        "chunksize": 5000,
        "heavy_admitted": bool(os.environ.get("HEAVY_ADMITTED")),
        "note": args.execution_note,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    if args.packaged_output:
        args.packaged_output.parent.mkdir(parents=True, exist_ok=True)
        args.packaged_output.write_text(
            json.dumps(packaged_summary(evidence), indent=2, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                band: fit["within_resource_limit"]
                for band, fit in evidence["fits"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
