#!/usr/bin/env python
"""Fit receipts for the SCF 2025 federal student loan stage (microcosm#1073).

Three measurements, all weighted, all against the 2025 SCF:

1. **Source checks.** The loader's education debt against the Fed's published
   Alternate Table 13 (19.92% of families hold education debt, median
   $25,500), how the federal flag, X7179 and the whose-education codes split
   the balances the X7179 and holder rules act on, the X8000 degree test for
   loans 4 and 6, and the Parent PLUS share and borrowers under each reading
   of a loan for a child's education.
2. **Hold-out fit and predictor ablation.** Families are split into five folds
   by ``yy1`` (all implicates of a family stay together). Each fold is drawn
   from the stage's two QRFs fit on the other four, and the drawn table is
   compared with the observed one, for each predictor set.
3. **Host fit, before and after.** The stage imputed onto a US Microcosm H5
   (the published default unless ``--h5`` names another), tabulated at that
   file's weights, under three settings (``HOST_SETTINGS``): the first receipt's
   construction, the SCF-matching recipient definition with every candidate
   predictor, and the build (that definition without the two shifted income
   predictors).

Federal Student Aid's portfolio tables (``FSA``) are the benchmark for the
Parent PLUS share, borrower counts and borrower bands. Nothing here is a gate
or a calibration target. The committed receipts are
produced on Modal by ``experiments/modal_1073.py`` (step ``receipt``), which
clones a pushed commit and verifies every input's SHA-256; the receipt records
that commit and the run id. A local run needs about 4 GB:

    .venv/Scripts/python.exe experiments/build_1073_scf_education_debt_receipt.py \
        --scf-zip-dir ~/.cache/scf --h5 populace_us_2024.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import microcosm.build.us_runtime.scf_education_debt as education_debt
from microcosm.build.us_runtime.scf_education_debt import (
    SCF_2025_FULL_EXTRACT,
    SCF_2025_SUMMARY_EXTRACT,
    SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR,
    SCF_EDUCATION_DEBT_BALANCE_BANDS,
    SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS,
    SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS,
    SCF_EDUCATION_DEBT_FIRST_RECEIPT_DEFINITION,
    SCF_EDUCATION_DEBT_PREDICTORS,
    SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION,
    SCF_EDUCATION_DEBT_TOTAL_TARGET,
    SCF_EDUCATION_LOAN_AMOUNT_BORROWED_COLUMNS,
    SCF_EDUCATION_LOAN_BALANCE_COLUMNS,
    SCF_EDUCATION_LOAN_FEDERAL_COLUMNS,
    SCF_EDUCATION_LOAN_PROGRAM_COLUMNS,
    SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN,
    SCF_EDUCATION_LOAN_WHOSE_COLUMNS,
    SCF_EDUCATION_LOAN_YEAR_COLUMNS,
    SCF_HOUSEHOLD_LISTING_AGE_COLUMNS,
    SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS,
    SCF_REVERSED_COLUMN,
    RecipientDefinition,
    ScfPublicFile,
    _read_stata_columns,
    derive_scf_education_debt_targets,
    draw_scf_education_debt_households,
    fetch_scf_public_file,
    impute_us_scf_education_debt,
    load_scf_2025_education_debt_donor,
    scf_education_debt_donor_fit_table,
    us_scf_education_debt_fit_table,
    us_scf_education_debt_signal_gate,
    with_us_scf_education_debt_inputs,
)
from microcosm.build.us_runtime.scf_wealth import SCF_WEALTH_PREDICTORS
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

ISSUE = "PolicyEngine/microcosm#1073"
FED_ALT_TABLE_13 = {
    "source": (
        "Federal Reserve, 2025 SCF Chartbook tables, scf2025_tables_public_real_"
        "historical.xlsx, sheet 'Table 13 25 Alt %s & median', education loans"
    ),
    "share_of_families_with_education_debt_pct": 19.92,
    "median_among_holders": 25_500.0,
}
#: Federal Student Aid portfolio tables, as published (the chronicle#347 source
#: packages). Downloaded from
#: https://studentaid.gov/sites/default/files/fsawg/datacenter/library/<file>;
#: the SHA-256 of each file read is listed. Recipients are students, so a
#: Parent PLUS loan counts the child; the by-age and by-debt-size tables count
#: borrowers, so they count the parent.
FSA = {
    "files": {
        "PortfolioSummary.xls": (
            "11606d4b93ceea969fcae0f790f3a4e1c2999b62050559ce89bdc269810a62ee"
        ),
        "PortfoliobyLoanType.xls": (
            "e64e9d2945dfe703d42d383a7f65fbd4b044d134565ea77a0bd194f366becd33"
        ),
        "Portfolio-by-Age.xls": (
            "c2027d277bb88e31c5a8c0f8fc6d81422f45d60e0306f79e63a0c25c7286e2ef"
        ),
        "Portfolio-by-Debt-Size.xls": (
            "881bd8bbd257da5f5b906b0f0923cf246de6e3d28ee35e8be1cd23d841e88a4d"
        ),
    },
    "total_dollars_fy2026_q3": 1_724.5e9,
    "unduplicated_recipients_fy2026_q3": 42.3e6,
    "parent_plus_dollars_fy2026_q3": 115.0e9,
    "parent_plus_recipients_fy2026_q3": 3.5e6,
    "borrowers_by_age_fy2026_q2": {
        "24_and_younger": 6.2e6,
        "25_to_34": 14.0e6,
        "35_to_49": 15.0e6,
        "50_to_61": 6.4e6,
        "62_and_older": 3.2e6,
        "not_reported": 0.1e6,
    },
    "borrowers_by_debt_size_fy2026_q2": {
        "under_5k": 6.9e6,
        "5k_to_10k": 7.2e6,
        "10k_to_20k": 9.2e6,
        "20k_to_40k": 9.5e6,
        "40k_to_60k": 4.2e6,
        "60k_to_80k": 2.6e6,
        "80k_to_100k": 1.4e6,
        "100k_to_200k": 2.6e6,
        "200k_plus": 1.2e6,
    },
}
#: The build's host fit in the previous receipt (before loans for a child's
#: education were routed to co-resident children and Parent PLUS was drawn
#: for itself), for the before/after table. The SCF's own "before" column is
#: recomputed here from the old rule.
PREVIOUS_RECEIPT = {
    "commit": "2a6e65a739b79a8232e1177610d11928fe92e26c",
    "run": "20261022T153020Z-2a6e65a73",
    "host_build": {
        "parent_plus_share_of_total": 0.16758194869071996,
        "federal_education_debt_total": 858421208954.9309,
        "borrower_weight": 21223403.433088087,
        "age_bands": {
            "24_and_younger": 0.049988325664619956,
            "25_to_34": 0.2917790224346921,
            "35_to_49": 0.3780495021343625,
            "50_to_61": 0.20778027670943267,
            "62_and_older": 0.07240287305689257,
        },
    },
}
#: The published default the host fit imputes onto, and its release-manifest
#: SHA-256 (releases/populace-us-2024-spm-20260915/release_manifest.json).
DEFAULT_HOST_RELEASE = "populace-us-2024-spm-20260915"
DEFAULT_HOST_SHA256 = "6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84"
#: The build's predictor set, and the first receipt's (every candidate).
PRODUCTION_VARIANT = "build_without_interest_and_ss_pension"
FIRST_RECEIPT_VARIANT = "shared_8_plus_household_income"
VARIANTS: Mapping[str, tuple[str, ...]] = {
    "shared_8_without_age": tuple(p for p in SCF_WEALTH_PREDICTORS if p != "age"),
    "shared_8": SCF_WEALTH_PREDICTORS,
    FIRST_RECEIPT_VARIANT: SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS,
    "plus_household_income_and_education": (
        *SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS,
        "educ",
    ),
    PRODUCTION_VARIANT: SCF_EDUCATION_DEBT_PREDICTORS,
    "build_plus_education": (*SCF_EDUCATION_DEBT_PREDICTORS, "educ"),
}
#: Host settings for the before/after table: (label, definition, predictors).
BUILD_SETTING = "build"
HOST_SETTINGS: Mapping[str, tuple[str, RecipientDefinition, tuple[str, ...]]] = {
    "first_receipt": (
        "First receipt",
        SCF_EDUCATION_DEBT_FIRST_RECEIPT_DEFINITION,
        SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS,
    ),
    "scf_codings_all_candidates": (
        "SCF codings, all predictors",
        SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION,
        SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS,
    ),
    BUILD_SETTING: (
        "Build",
        SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION,
        SCF_EDUCATION_DEBT_PREDICTORS,
    ),
}
WHOSE_LABELS = {
    1: "1 respondent (reference person)",
    2: "2 spouse or partner",
    3: "3 child (incl. grandchild, other relative)",
    6: "6 multiple people",
    7: "7 unrelated person",
    -7: "-7 other",
}
_HOST_PERSON_COLUMNS = (
    "person_id",
    "person_household_id",
    "person_tax_unit_id",
    "person_spm_unit_id",
    "person_family_id",
    "person_marital_unit_id",
    "age",
    "is_female",
    "PRDTRACE",
    "PRDTHSP",
    "A_MARITL",
    "A_LINENO",
    "A_SPOUSE",
    "PECOHAB",
    "PH_SEQ",
    "PEPAR1",
    "PEPAR2",
)


def _git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_from_local_zip(
    public_file: ScfPublicFile, zip_dir: Path, cache_dir: Path
) -> Path:
    """Extract a pinned member from a locally cached Fed zip, verifying both pins."""

    archive = zip_dir / public_file.zip_name
    observed = _sha256(archive)
    if observed != public_file.zip_sha256:
        raise ValueError(f"{archive} sha-256 {observed} != {public_file.zip_sha256}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / public_file.member
    with zipfile.ZipFile(archive) as handle:
        target.write_bytes(handle.read(public_file.member))
    observed = _sha256(target)
    if observed != public_file.member_sha256:
        raise ValueError(f"{target} sha-256 {observed} != {public_file.member_sha256}")
    return target


def _pooled_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2, side="left"))
    return float(values[order][index])


def source_checks(donor: pd.DataFrame, full: pd.DataFrame) -> dict[str, object]:
    """The loader against the Fed's table, and the balances each rule acts on."""

    weight = donor["scf_weight"].to_numpy(dtype=np.float64)
    debt = donor["education_debt"].to_numpy(dtype=np.float64)
    holders = debt > 0
    per_implicate = [
        100.0
        * float(group.loc[group["education_debt"] > 0, "scf_weight"].sum())
        / float(group["scf_weight"].sum())
        for _, group in donor.groupby("implicate")
    ]
    full = full.reset_index(drop=True)
    full_weight = full["wgt"].to_numpy(dtype=np.float64)
    balances = np.column_stack(
        [
            np.maximum(full[c].to_numpy(dtype=np.float64), 0.0)
            for c in SCF_EDUCATION_LOAN_BALANCE_COLUMNS
        ]
    )
    federal = np.column_stack(
        [full[c].to_numpy(dtype=np.float64) for c in SCF_EDUCATION_LOAN_FEDERAL_COLUMNS]
    )
    whose = np.column_stack(
        [full[c].to_numpy(dtype=np.float64) for c in SCF_EDUCATION_LOAN_WHOSE_COLUMNS]
    )
    remaining = np.maximum(
        full[SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN].to_numpy(dtype=np.float64),
        0.0,
    )
    dollars = balances * full_weight[:, None]
    loans_1_6 = float(dollars.sum())
    all_education = loans_1_6 + float(np.dot(remaining, full_weight))
    federal_dollars = np.where(federal == 1, dollars, 0.0)
    by_whose = {}
    for code, label in WHOSE_LABELS.items():
        by_whose[label] = float(federal_dollars[whose == code].sum()) / float(
            federal_dollars.sum()
        )
    other_codes = sorted(
        {int(code) for code in np.unique(whose[balances > 0])} - set(WHOSE_LABELS)
    )
    return {
        "fed_alternate_table_13": FED_ALT_TABLE_13,
        "loader_share_of_families_with_education_debt_pct": float(
            np.mean(per_implicate)
        ),
        "loader_median_among_holders": _pooled_median(debt[holders], weight[holders]),
        "families_per_implicate": int(len(donor) // 5),
        "balance_shares": {
            "loans_1_6_of_all_education_debt": loans_1_6 / all_education,
            "x7179_remaining_loans_of_all_education_debt": 1.0
            - loans_1_6 / all_education,
            "federal_of_loans_1_6": float(federal_dollars.sum()) / loans_1_6,
            "non_federal_of_loans_1_6": float(np.where(federal == 5, dollars, 0).sum())
            / loans_1_6,
            "federal_status_blank_of_loans_1_6": float(
                np.where(federal == 0, dollars, 0).sum()
            )
            / loans_1_6,
        },
        "federal_balance_share_by_whose_education_code": by_whose,
        "whose_education_codes_outside_the_rule": other_codes,
        "federal_education_debt_after_x7179_rule": float(
            np.dot(donor[SCF_EDUCATION_DEBT_TOTAL_TARGET], weight)
        ),
        "remaining_loans_without_detail_dollars": float(
            np.dot(donor["remaining_loans_without_detail"], weight)
        ),
    }


_COMPLETED_COLUMNS = ("x7881", "x7886", "x7891", "x7896", "x7901", "x7996")
_CONSOLIDATED_COLUMNS = ("x9320", "x9321", "x9322", "x9323", "x9324", "x9325")
_DEGREE_LEVEL = {10: 1, 11: 1, 12: 2, 13: 3, 14: 4, 15: 4}
FULL_FILE_RECEIPT_COLUMNS = (
    "y1",
    "yy1",
    "x42001",
    "x5931",
    "x6111",
    SCF_REVERSED_COLUMN,
    *SCF_EDUCATION_LOAN_BALANCE_COLUMNS,
    SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN,
    *SCF_EDUCATION_LOAN_FEDERAL_COLUMNS,
    *SCF_EDUCATION_LOAN_WHOSE_COLUMNS,
    *SCF_EDUCATION_LOAN_YEAR_COLUMNS,
    *SCF_EDUCATION_LOAN_PROGRAM_COLUMNS,
    *SCF_EDUCATION_LOAN_AMOUNT_BORROWED_COLUMNS,
    *_COMPLETED_COLUMNS,
    *_CONSOLIDATED_COLUMNS,
    *SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS,
    *SCF_HOUSEHOLD_LISTING_AGE_COLUMNS,
)


def _matrix(full: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return np.column_stack(
        [pd.to_numeric(full[c], errors="coerce").fillna(0).to_numpy() for c in columns]
    )


def x8000_degree_check(full: pd.DataFrame) -> list[dict[str, object]]:
    """Do loans' whose-education codes match the coded person's degree?

    Completed loans coded 1 or 2 that financed a degree (programs 10-15), in
    families with a spouse or partner (X6111 recorded). A loan is consistent
    with a person whose highest degree (X5931 for the reference person, X6111
    for the spouse or partner) is at least the program's level; it
    discriminates when it is consistent with exactly one of them. Raw codes,
    before the stage's correction; counts pool the five implicates.
    """

    level = np.vectorize(lambda code: _DEGREE_LEVEL.get(int(code), 0))
    balance = _matrix(full, SCF_EDUCATION_LOAN_BALANCE_COLUMNS)
    whose = _matrix(full, SCF_EDUCATION_LOAN_WHOSE_COLUMNS)
    completed = _matrix(full, _COMPLETED_COLUMNS)
    program_level = level(_matrix(full, SCF_EDUCATION_LOAN_PROGRAM_COLUMNS))
    reference = level(pd.to_numeric(full["x5931"]).fillna(0).to_numpy())
    spouse_degree = pd.to_numeric(full["x6111"]).fillna(0).to_numpy()
    spouse = level(spouse_degree)
    reversed_family = pd.to_numeric(full[SCF_REVERSED_COLUMN]).to_numpy() == 1
    implicate = pd.to_numeric(full["y1"]).astype(int).to_numpy() % 10
    rows = []
    for loans, positions in (("1, 2, 3, 5", (0, 1, 2, 4)), ("4, 6", (3, 5))):
        for family, label in ((False, "not reversed"), (True, "reversed")):
            for code in (1, 2):
                counts = {"all": [0, 0], "implicate_1": [0, 0]}
                for position in positions:
                    loan = (
                        (balance[:, position] > 0)
                        & (whose[:, position] == code)
                        & (completed[:, position] == 1)
                        & (program_level[:, position] > 0)
                        & (spouse_degree != 0)
                        & (reversed_family == family)
                    )
                    fits_reference = reference >= program_level[:, position]
                    fits_spouse = spouse >= program_level[:, position]
                    discriminating = loan & (fits_reference != fits_spouse)
                    coded = fits_reference if code == 1 else fits_spouse
                    for key, mask in (
                        ("all", discriminating),
                        ("implicate_1", discriminating & (implicate == 1)),
                    ):
                        counts[key][0] += int(mask.sum())
                        counts[key][1] += int((mask & coded).sum())
                rows.append(
                    {
                        "loans": loans,
                        "x8000": label,
                        "code": code,
                        "discriminating": counts["all"][0],
                        "consistent_with_coded_person": counts["all"][1],
                        "discriminating_implicate_1": counts["implicate_1"][0],
                        "consistent_implicate_1": counts["implicate_1"][1],
                    }
                )
    return rows


def _derive_reading(full: pd.DataFrame, reading: str) -> pd.DataFrame:
    """Donor targets under one reading of the whose-education codes.

    ``before`` is the previous rule: no X8000 correction and every loan for a
    child's education the head's Parent PLUS loan (as if no child were
    listed). ``x8000_only`` adds the correction. ``build`` is the stage.
    ``within_limit_parent_plus`` reads an undergraduate loan within the
    dependent limit for a child of dependent age as Parent PLUS too.
    """

    frame = full.copy()
    if reading in {"before", "x8000_only"}:
        for column in SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS:
            frame[column] = 0
    if reading == "before":
        frame[SCF_REVERSED_COLUMN] = 5
    if reading != "within_limit_parent_plus":
        return derive_scf_education_debt_targets(frame)
    limit = education_debt._DEPENDENT_UNDERGRADUATE_AGGREGATE_LIMIT
    education_debt._DEPENDENT_UNDERGRADUATE_AGGREGATE_LIMIT = -1.0
    try:
        return derive_scf_education_debt_targets(frame)
    finally:
        education_debt._DEPENDENT_UNDERGRADUATE_AGGREGATE_LIMIT = limit


READINGS = ("before", "x8000_only", "build", "within_limit_parent_plus")


def child_education_readings(full: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Parent PLUS share and borrowers (people) under each reading."""

    weight = full["wgt"].to_numpy(dtype=np.float64)
    result = {}
    for reading in READINGS:
        targets = _derive_reading(full, reading)
        total = float(np.dot(targets[SCF_EDUCATION_DEBT_TOTAL_TARGET], weight))
        plus = targets["federal_parent_plus_loan_balance"].to_numpy()
        head = targets["head_federal_student_loan_balance"].to_numpy() + plus
        spouse = targets["spouse_federal_student_loan_balance"].to_numpy()
        child = targets["child_federal_student_loan_balance"].to_numpy()
        result[reading] = {
            "parent_plus_share_of_federal": float(np.dot(plus, weight)) / total,
            "child_own_share_of_federal": float(np.dot(child, weight)) / total,
            "parent_plus_dollars": float(np.dot(plus, weight)),
            "borrowers": float(
                weight[head > 0].sum()
                + weight[spouse > 0].sum()
                + weight[child > 0].sum()
            ),
            "parent_plus_holders": float(weight[plus > 0].sum()),
            "child_borrowers": float(weight[child > 0].sum()),
        }
    return result


def consolidated_share_of_new_loans(full: pd.DataFrame) -> float:
    """Share of federal balances dated 2014 or later on a consolidated loan."""

    weight = full["wgt"].to_numpy(dtype=np.float64)[:, None]
    balance = _matrix(full, SCF_EDUCATION_LOAN_BALANCE_COLUMNS)
    federal = (_matrix(full, SCF_EDUCATION_LOAN_FEDERAL_COLUMNS) == 1) & (balance > 0)
    new = _matrix(full, SCF_EDUCATION_LOAN_YEAR_COLUMNS) >= 2014
    consolidated = _matrix(full, _CONSOLIDATED_COLUMNS) == 1
    dated_new = np.where(federal & new, balance, 0.0) * weight
    return float(np.where(consolidated, dated_new, 0.0).sum() / dated_new.sum())


def donor_under_reading(
    donor: pd.DataFrame, full: pd.DataFrame, reading: str
) -> pd.DataFrame:
    """The donor with its targets derived under another reading."""

    targets = _derive_reading(full, reading)
    keys = pd.MultiIndex.from_arrays(
        [
            pd.to_numeric(full["yy1"]).to_numpy(),
            pd.to_numeric(full["y1"]).astype(int).to_numpy() % 10,
        ]
    )
    targets.index = keys
    aligned = targets.reindex(
        pd.MultiIndex.from_arrays(
            [donor["yy1"].to_numpy(), donor["implicate"].to_numpy()]
        )
    )
    result = donor.copy()
    for column in targets.columns:
        result[column] = aligned[column].to_numpy()
    return result


def parent_plus_draw_rate(donor: pd.DataFrame, drawn: pd.DataFrame) -> float:
    """Weighted share drawing Parent PLUS among held-out borrower families without it.

    Families observed with federal education debt but no Parent PLUS balance,
    and drawn with debt: how often the draw gives them Parent PLUS anyway.
    """

    weight = donor["scf_weight"].to_numpy(dtype=np.float64)
    observed_without = (donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] > 0) & (
        donor["federal_parent_plus_loan_balance"] <= 0
    )
    drawn_debt = drawn[SCF_EDUCATION_DEBT_TOTAL_TARGET] > 0
    base = (observed_without & drawn_debt).to_numpy()
    plus = (drawn["federal_parent_plus_loan_balance"] > 0).to_numpy()
    return float(weight[base & plus].sum() / weight[base].sum())


def _fold_of_family(families: np.ndarray, *, folds: int, seed: int) -> np.ndarray:
    unique = np.unique(families)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1073, 5]))
    assignment = pd.Series(rng.permutation(len(unique)) % folds, index=unique)
    return assignment.reindex(families).to_numpy()


def holdout_draws(
    donor: pd.DataFrame,
    predictors: Sequence[str],
    *,
    folds: int,
    seed: int,
    n_estimators: int,
) -> pd.DataFrame:
    """Draw every SCF row from models fit on the other folds' families."""

    fold = _fold_of_family(donor["yy1"].to_numpy(), folds=folds, seed=seed)
    parts = []
    for k in range(folds):
        held_out = fold == k
        drawn = draw_scf_education_debt_households(
            donor.loc[
                held_out, [*predictors, SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR]
            ],
            donor.loc[~held_out],
            seed=seed + k,
            predictors=predictors,
            n_estimators=n_estimators,
        )
        parts.append(drawn)
    return pd.concat(parts).reindex(donor.index)


def _flatten(table: Mapping[str, object], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in table.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{name}."))
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


def fit_metrics(
    observed: Mapping[str, object], drawn: Mapping[str, object]
) -> dict[str, float]:
    """Distributional distances between two fit tables (pp = percentage points)."""

    def share_mae(section: str) -> float:
        groups = observed[section]
        return float(
            np.nanmean(
                [
                    abs(
                        drawn[section][g]["share_with_federal_education_debt"]
                        - groups[g]["share_with_federal_education_debt"]
                    )
                    for g in groups
                ]
            )
            * 100
        )

    def tvd(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        return 0.5 * float(sum(abs(a[k] - b[k]) for k in a))

    households_o = observed["households"]
    households_d = drawn["households"]
    return {
        "share_with_debt_error_pp": 100
        * (
            households_d["share_with_federal_education_debt"]
            - households_o["share_with_federal_education_debt"]
        ),
        "mean_among_borrowers_error_pct": 100
        * (
            households_d["mean_among_borrowers"] / households_o["mean_among_borrowers"]
            - 1
        ),
        "median_among_borrowers_error_pct": 100
        * (
            households_d["median_among_borrowers"]
            / households_o["median_among_borrowers"]
            - 1
        ),
        "household_balance_band_tvd": tvd(
            observed["household_balance_bands"], drawn["household_balance_bands"]
        ),
        "head_age_band_share_mae_pp": share_mae("by_head_age_band"),
        "race_share_mae_pp": share_mae("by_race"),
        "income_quintile_share_mae_pp": share_mae("by_income_quintile"),
        "borrower_balance_band_tvd": tvd(
            observed["borrowers"]["balance_bands"], drawn["borrowers"]["balance_bands"]
        ),
        "borrower_age_band_tvd": tvd(
            observed["borrowers"]["age_bands"], drawn["borrowers"]["age_bands"]
        ),
    }


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a = pd.Series(a).rank().to_numpy()
    ranks_b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def load_host_frame(path: Path) -> Frame:
    """Read the columns the stage needs from a Microcosm US H5 (table format)."""

    from microcosm.build.us_runtime.scf_education_debt import (
        _RECIPIENT_HOUSEHOLD_INCOME_COLUMNS,
    )

    with pd.HDFStore(path, "r") as store:
        available = set(store.get_storer("/person").data_columns)
        columns = [c for c in _HOST_PERSON_COLUMNS if c in available] + [
            c for c in _RECIPIENT_HOUSEHOLD_INCOME_COLUMNS if c in available
        ]
        person = store.select("/person", columns=columns)
        household = store.select(
            "/household", columns=["household_id", "household_weight"]
        )
        tables = {
            "person": person.reset_index(drop=True),
            "household": household.loc[:, ["household_id"]].reset_index(drop=True),
        }
        for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
            tables[entity] = store.select(
                f"/{entity}", columns=[f"{entity}_id"]
            ).reset_index(drop=True)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=household["household_weight"].to_numpy(dtype=np.float64),
                kind=WeightKind.CALIBRATED,
            )
        },
    )


def _pct(value: float) -> str:
    return "" if value != value else f"{100 * value:.1f}%"


def _dollars(value: float) -> str:
    return "" if value != value else f"${value:,.0f}"


def markdown(receipt: Mapping[str, object]) -> str:
    """Render the receipt's tables as the committed Markdown note."""

    observed = receipt["scf_2025_observed"]
    holdout = receipt["holdout"]["variants"][PRODUCTION_VARIANT]["fit"]
    host = receipt.get("host", {}).get("fit")
    settings = receipt.get("host_settings", {})
    columns = [("SCF 2025", observed), ("SCF hold-out (build)", holdout)]
    for setting in settings.values():
        columns.append((f"Host: {setting['label']}", setting["fit"]))
    header = "| | " + " | ".join(label for label, _ in columns) + " |"
    rule = "|---|" + "---:|" * len(columns)

    def row(label: str, getter, formatter) -> str:
        return (
            f"| {label} | "
            + " | ".join(formatter(getter(table)) for _, table in columns)
            + " |"
        )

    lines = [
        "# microcosm#1073 receipts: SCF 2025 federal student loan balances",
        "",
        f"Generated {receipt['generated_at']} by "
        "`experiments/build_1073_scf_education_debt_receipt.py` on branch "
        f"`scf-2025-education-debt`, commit `{receipt['provenance']['commit']}`"
        + (
            f", Modal run `{receipt['provenance']['run']}` "
            "(`experiments/modal_1073.py`)"
            if receipt["provenance"].get("run")
            else ""
        )
        + ". Machine-readable copy: "
        "`experiments/receipts/1073-scf-education-debt-fit.json`.",
        "",
        "Host columns impute the published default "
        f"`{receipt.get('host', {}).get('release', DEFAULT_HOST_RELEASE)}` at its "
        "calibrated weights under three settings: the first receipt's construction "
        "(oldest adult, `A_MARITL` marriage, own children, whole-household income, "
        "every candidate predictor); the SCF-matching codings with every candidate "
        "predictor; and the build, which also leaves out interest and dividends and "
        "Social Security and pensions. The cause and the evidence are in the "
        "stage notes and `experiments/receipts/1073-predictor-shift-diagnosis.json`.",
        "",
        "Every figure is weighted. SCF rows pool the five implicates with `wgt` "
        "(X42001/5). Household figures group by the reference person's age and "
        "race and by weighted quintiles of income less capital gains of the "
        "setting's income unit, the income concept that setting conditions on. "
        "Borrower figures count "
        "people with a federal balance (own plus Parent PLUS), in Federal Student "
        "Aid's balance and age bands; SCF borrowers are the reference person, the "
        "spouse or partner and an adult child holding a loan for their own "
        "education, whose public-file age band (18-27 reads 25) is spread evenly "
        "over its years. Nothing here is a gate or a calibration target.",
        "",
        "## Source checks",
        "",
    ]
    checks = receipt["source_checks"]
    lines += [
        "| Check | Loader | Fed Alternate Table 13 |",
        "|---|---:|---:|",
        "| Families with any education debt | "
        f"{checks['loader_share_of_families_with_education_debt_pct']:.2f}% | "
        f"{FED_ALT_TABLE_13['share_of_families_with_education_debt_pct']:.2f}% |",
        "| Median education debt among them | "
        f"{_dollars(checks['loader_median_among_holders'])} | "
        f"{_dollars(FED_ALT_TABLE_13['median_among_holders'])} |",
        "",
        "| Share of education-loan balances | Value |",
        "|---|---:|",
    ]
    for key, value in checks["balance_shares"].items():
        lines.append(f"| {key.replace('_', ' ')} | {_pct(value)} |")
    lines += [
        "",
        "| Federal balances on loans 1-6, by whose education (codes as published) "
        "| Share |",
        "|---|---:|",
    ]
    for key, value in checks["federal_balance_share_by_whose_education_code"].items():
        lines.append(f"| {key} | {_pct(value)} |")
    lines += [
        "",
        "### X8000: are the whose-education codes reversed?",
        "",
        "Completed degree loans coded 1 or 2 in families with a spouse or partner. "
        "A loan discriminates when it is consistent with exactly one partner's "
        "highest degree (X5931, X6111: at least the program's level); the last "
        "columns count those consistent with the coded person (code 1 the "
        "reference person, code 2 the spouse or partner), as published, before "
        "the stage swaps codes 1 and 2 of loans 4 and 6 in reversed families. "
        "Counts pool the five implicates (implicate 1 in brackets).",
        "",
        "| Loans | X8000 | Code | Discriminating | Consistent with the coded person |",
        "|---|---|---:|---:|---:|",
    ]
    for entry in checks["x8000_degree_check"]:
        lines.append(
            f"| {entry['loans']} | {entry['x8000']} | {entry['code']} | "
            f"{entry['discriminating']} ({entry['discriminating_implicate_1']}) | "
            f"{entry['consistent_with_coded_person']} "
            f"({entry['consistent_implicate_1']}) |"
        )
    readings = checks["child_education_readings"]
    reading_labels = {
        "before": "Before: every code-3 loan Parent PLUS, codes as published",
        "x8000_only": "X8000 correction only",
        "build": "Build: co-resident adult child unless only a parent could borrow",
        "within_limit_parent_plus": (
            "Alternative: within-limit undergraduate loans for a dependent-age "
            "child also Parent PLUS"
        ),
    }
    lines += [
        "",
        "### Loans for a child's education (SCF 2025, weighted)",
        "",
        "FSA's Direct Parent PLUS balance is "
        f"{_pct(FSA['parent_plus_dollars_fy2026_q3'] / FSA['total_dollars_fy2026_q3'])}"
        " of its portfolio at FY2026 Q3, a floor, since consolidated Parent PLUS "
        "loans sit in the consolidation total.",
        "",
        "| Reading | Parent PLUS share | Child's own share | Parent PLUS holders "
        "| Child borrowers | Borrowers |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for reading, label in reading_labels.items():
        entry = readings[reading]
        lines.append(
            f"| {label} | {_pct(entry['parent_plus_share_of_federal'])} | "
            f"{_pct(entry['child_own_share_of_federal'])} | "
            f"{entry['parent_plus_holders'] / 1e6:,.1f}M | "
            f"{entry['child_borrowers'] / 1e6:,.1f}M | "
            f"{entry['borrowers'] / 1e6:,.1f}M |"
        )
    lines += [
        "",
        "IBR new-borrower status is an upper bound: the SCF sees only loans still "
        "owed, so paid-off pre-2014 loans are missed, and a consolidated loan is "
        "dated by its most recent underlying loan. "
        f"{_pct(checks['consolidated_share_of_federal_balances_dated_2014_or_later'])}"
        " of federal balances dated 2014 or later are on a loan that has been "
        "consolidated or refinanced (X9320-X9325).",
    ]
    before = receipt["scf_2025_observed_before"]
    previous = receipt["previous_receipt"]["host_build"]
    build = settings.get(BUILD_SETTING, {}).get("fit")
    by_age_fsa = {
        k: v
        for k, v in FSA["borrowers_by_age_fy2026_q2"].items()
        if k != "not_reported"
    }
    fsa_age_total = sum(by_age_fsa.values())
    fsa_borrowers = sum(FSA["borrowers_by_age_fy2026_q2"].values())
    compare = [
        ("SCF 2025, before", {
            "pp": before["households"]["parent_plus_share_of_total"],
            "pp_dollars": before["households"]["parent_plus_share_of_total"]
            * before["households"]["federal_education_debt_total"],
            "total": before["households"]["federal_education_debt_total"],
            "borrowers": before["borrowers"]["borrower_weight"],
            "ages": before["borrowers"]["age_bands"],
        }),
        ("SCF 2025", {
            "pp": observed["households"]["parent_plus_share_of_total"],
            "pp_dollars": observed["households"]["parent_plus_share_of_total"]
            * observed["households"]["federal_education_debt_total"],
            "total": observed["households"]["federal_education_debt_total"],
            "borrowers": observed["borrowers"]["borrower_weight"],
            "ages": observed["borrowers"]["age_bands"],
        }),
        ("Host: Build, before", {
            "pp": previous["parent_plus_share_of_total"],
            "pp_dollars": previous["parent_plus_share_of_total"]
            * previous["federal_education_debt_total"],
            "total": previous["federal_education_debt_total"],
            "borrowers": previous["borrower_weight"],
            "ages": previous["age_bands"],
        }),
    ]  # fmt: skip
    if build is not None:
        compare.append(
            (
                "Host: Build",
                {
                    "pp": build["households"]["parent_plus_share_of_total"],
                    "pp_dollars": build["households"]["parent_plus_share_of_total"]
                    * build["households"]["federal_education_debt_total"],
                    "total": build["households"]["federal_education_debt_total"],
                    "borrowers": build["borrowers"]["borrower_weight"],
                    "ages": build["borrowers"]["age_bands"],
                },
            )
        )
    lines += [
        "",
        "## Parent PLUS and borrowers against Federal Student Aid",
        "",
        "Before is the previous rule (every loan for a child's education the "
        "head's Parent PLUS loan, the residual of the composition draw); the "
        "host's before column is the build in the previous receipt (commit "
        f"`{receipt['previous_receipt']['commit'][:9]}`, Modal run "
        f"`{receipt['previous_receipt']['run']}`). FSA's Parent PLUS share is a "
        "floor; its dollars are the whole federal portfolio, of which household "
        "surveys capture part, so the shares compare and the levels do not. FSA "
        "borrowers come from its by-age table (FY2026 Q2), which counts the "
        "parent on a Parent PLUS loan; its "
        f"{FSA['unduplicated_recipients_fy2026_q3'] / 1e6:.1f}M unduplicated "
        "recipients (FY2026 Q3) count the student instead.",
        "",
        "| | FSA | " + " | ".join(label for label, _ in compare) + " |",
        "|---|---:|" + "---:|" * len(compare),
        "| Parent PLUS share of federal balances | "
        f"{_pct(FSA['parent_plus_dollars_fy2026_q3'] / FSA['total_dollars_fy2026_q3'])}"
        " (floor) | " + " | ".join(_pct(c["pp"]) for _, c in compare) + " |",
        f"| Parent PLUS balances | ${FSA['parent_plus_dollars_fy2026_q3'] / 1e9:,.1f}B | "
        + " | ".join(f"${c['pp_dollars'] / 1e9:,.1f}B" for _, c in compare)
        + " |",
        f"| Federal balances | ${FSA['total_dollars_fy2026_q3'] / 1e9:,.1f}B | "
        + " | ".join(f"${c['total'] / 1e9:,.1f}B" for _, c in compare)
        + " |",
        f"| Borrowers (people) | {fsa_borrowers / 1e6:,.1f}M | "
        + " | ".join(f"{c['borrowers'] / 1e6:,.1f}M" for _, c in compare)
        + " |",
    ]
    for label, _, _ in SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS:
        lines.append(
            f"| Borrowers aged {label.replace('_', ' ')} | "
            f"{_pct(by_age_fsa[label] / fsa_age_total)} | "
            + " | ".join(_pct(c["ages"][label]) for _, c in compare)
            + " |"
        )
    lines += [
        "",
        "## Fit",
        "",
        header,
        rule,
        row(
            "Households with federal education debt",
            lambda t: t["households"]["share_with_federal_education_debt"],
            _pct,
        ),
        row(
            "Mean among them",
            lambda t: t["households"]["mean_among_borrowers"],
            _dollars,
        ),
        row(
            "Median among them",
            lambda t: t["households"]["median_among_borrowers"],
            _dollars,
        ),
        row(
            "Weighted federal education debt",
            lambda t: t["households"]["federal_education_debt_total"],
            lambda v: f"${v / 1e12:,.3f}T",
        ),
        row(
            "Parent PLUS share of the total",
            lambda t: t["households"]["parent_plus_share_of_total"],
            _pct,
        ),
        row(
            "Borrowers (people)",
            lambda t: t["borrowers"]["borrower_weight"],
            lambda v: f"{v / 1e6:,.1f}M",
        ),
        row(
            "Mean balance per borrower",
            lambda t: t["borrowers"]["mean_balance"],
            _dollars,
        ),
        row(
            "Median balance per borrower",
            lambda t: t["borrowers"]["median_balance"],
            _dollars,
        ),
        row(
            "IBR new borrowers among borrowers (upper bound)",
            lambda t: t["borrowers"]["ibr_new_borrower_share"],
            _pct,
        ),
        row(
            "Graduate share of own original principal",
            lambda t: t["borrowers"]["graduate_share_of_own_original_principal"],
            _pct,
        ),
        "",
        "### Borrowers by FSA balance band",
        "",
        "FSA: borrowers by debt size, FY2026 Q2.",
        "",
        header + " FSA |",
        rule + "---:|",
    ]
    by_size = FSA["borrowers_by_debt_size_fy2026_q2"]
    for label, _, _ in SCF_EDUCATION_DEBT_BALANCE_BANDS:
        lines.append(
            row(label, lambda t, k=label: t["borrowers"]["balance_bands"][k], _pct)
            + f" {_pct(by_size[label] / sum(by_size.values()))} |"
        )
    by_age = {
        k: v
        for k, v in FSA["borrowers_by_age_fy2026_q2"].items()
        if k != "not_reported"
    }
    lines += [
        "",
        "### Borrowers by FSA age band",
        "",
        "FSA: borrowers by age, FY2026 Q2, of those with an age reported.",
        "",
        header + " FSA |",
        rule + "---:|",
    ]
    for label, _, _ in SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS:
        lines.append(
            row(label, lambda t, k=label: t["borrowers"]["age_bands"][k], _pct)
            + f" {_pct(by_age[label] / sum(by_age.values()))} |"
        )
    for section, title in (
        (
            "by_head_age_band",
            "Households with federal education debt, by age of reference person",
        ),
        ("by_race", "Households with federal education debt, by race (`racecl4`)"),
        (
            "by_income_quintile",
            "Households with federal education debt, by income quintile",
        ),
    ):
        lines += ["", f"### {title}", "", header, rule]
        for group in observed[section]:
            lines.append(
                row(
                    group,
                    lambda t, s=section, g=group: t[s][g][
                        "share_with_federal_education_debt"
                    ],
                    _pct,
                )
            )
        lines += ["", "Median among borrowers:", "", header, rule]
        for group in observed[section]:
            lines.append(
                row(
                    group,
                    lambda t, s=section, g=group: t[s][g]["median_among_borrowers"],
                    _dollars,
                )
            )
    lines += [
        "",
        "## Predictor ablation (five-fold hold-out on SCF families)",
        "",
        "Each variant refits both QRFs. Errors are drawn minus observed; TVD is "
        "total variation distance between band distributions; MAE averages the "
        "absolute error of the share with debt across the groups. `educ` is the "
        "SCF reference person's education, which the CPS side cannot supply until "
        "microcosm#297, so that row shows what the predictor would add. The last "
        "column is the weighted share of held-out families observed with federal "
        "debt but no Parent PLUS that are drawn some Parent PLUS anyway.",
        "",
    ]
    metrics = [
        "share_with_debt_error_pp",
        "mean_among_borrowers_error_pct",
        "median_among_borrowers_error_pct",
        "household_balance_band_tvd",
        "head_age_band_share_mae_pp",
        "race_share_mae_pp",
        "income_quintile_share_mae_pp",
        "borrower_balance_band_tvd",
        "borrower_age_band_tvd",
        "household_rank_correlation",
        "parent_plus_drawn_without_observed_parent_plus",
    ]
    lines.append("| Variant | " + " | ".join(metrics) + " |")
    lines.append("|---|" + "---:|" * len(metrics))
    for name, variant in receipt["holdout"]["variants"].items():
        values = variant["metrics"]
        marker = " (build)" if name == PRODUCTION_VARIANT else ""
        lines.append(
            f"| {name}{marker} | "
            + " | ".join(f"{values[m]:.3f}" for m in metrics)
            + " |"
        )
    if host is not None:
        host_metrics = [
            m
            for m in metrics
            if m
            not in {
                "household_rank_correlation",
                "parent_plus_drawn_without_observed_parent_plus",
            }
        ]
        lines += [
            "",
            "## Host fit, before and after",
            "",
            f"Imputed with build seed {receipt['host']['seed']} onto "
            f"`{receipt['host']['release']}` ({receipt['host']['households']:,} "
            f"households, sha256 `{receipt['host']['sha256'][:12]}...`) at its "
            "calibrated household weights; the published weights were calibrated "
            "without these columns. Build signal gate: "
            f"{'passed' if receipt['host']['gate_passed'] else 'FAILED'}. Errors "
            "are imputed minus SCF 2025.",
            "",
            "| Setting | Households with debt | Parent PLUS share | "
            + " | ".join(host_metrics)
            + " |",
            "|---|---:|---:|" + "---:|" * len(host_metrics),
        ]
        for setting in settings.values():
            fit = setting["fit"]
            lines.append(
                f"| {setting['label']} | "
                f"{_pct(fit['households']['share_with_federal_education_debt'])} | "
                f"{_pct(fit['households']['parent_plus_share_of_total'])} | "
                + " | ".join(
                    f"{setting['metrics_vs_scf'][m]:.3f}" for m in host_metrics
                )
                + " |"
            )
        lines.append(
            "| SCF 2025 | "
            f"{_pct(observed['households']['share_with_federal_education_debt'])} | "
            f"{_pct(observed['households']['parent_plus_share_of_total'])} | "
            + " | ".join("" for _ in host_metrics)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scf-zip-dir",
        type=Path,
        help="Directory holding scfp2025s.zip and scf2025s.zip (else fetch).",
    )
    parser.add_argument(
        "--scf-cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "microcosm" / "scf",
    )
    parser.add_argument("--h5", type=Path, help="US Microcosm H5 to impute onto.")
    parser.add_argument("--host-release", default=DEFAULT_HOST_RELEASE)
    parser.add_argument(
        "--run",
        default="",
        help="Label of the run that produced the receipt (the Modal run id).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(VARIANTS),
        choices=list(VARIANTS),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("experiments/receipts/1073-scf-education-debt-fit.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("experiments/1073-scf-education-debt-receipts.md"),
    )
    args = parser.parse_args(argv)

    if args.scf_zip_dir is not None:
        summary_path = _member_from_local_zip(
            SCF_2025_SUMMARY_EXTRACT, args.scf_zip_dir.expanduser(), args.scf_cache_dir
        )
        full_path = _member_from_local_zip(
            SCF_2025_FULL_EXTRACT, args.scf_zip_dir.expanduser(), args.scf_cache_dir
        )
    else:
        summary_path = fetch_scf_public_file(
            SCF_2025_SUMMARY_EXTRACT, args.scf_cache_dir
        )
        full_path = fetch_scf_public_file(SCF_2025_FULL_EXTRACT, args.scf_cache_dir)

    donor = load_scf_2025_education_debt_donor(summary_path, full_path)
    full = _read_stata_columns(
        full_path, FULL_FILE_RECEIPT_COLUMNS, label="2025 full file"
    )
    full["wgt"] = full["x42001"] / 5.0
    observed = scf_education_debt_donor_fit_table(donor)
    observed_before = scf_education_debt_donor_fit_table(
        donor_under_reading(donor, full, "before")
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "issue": ISSUE,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "provenance": {"commit": _git_commit(), "run": args.run},
        "scf_files": {
            public_file.member: {
                "url": public_file.url,
                "zip_sha256": public_file.zip_sha256,
                "member_sha256": public_file.member_sha256,
            }
            for public_file in (SCF_2025_SUMMARY_EXTRACT, SCF_2025_FULL_EXTRACT)
        },
        "source_checks": {
            **source_checks(donor, full),
            "x8000_degree_check": x8000_degree_check(full),
            "child_education_readings": child_education_readings(full),
            "consolidated_share_of_federal_balances_dated_2014_or_later": (
                consolidated_share_of_new_loans(full)
            ),
        },
        "fsa": FSA,
        "previous_receipt": PREVIOUS_RECEIPT,
        "scf_2025_observed": observed,
        "scf_2025_observed_before": observed_before,
        "holdout": {
            "folds": args.folds,
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "production_variant": PRODUCTION_VARIANT,
            "variants": {},
        },
    }
    observed_total = donor[SCF_EDUCATION_DEBT_TOTAL_TARGET].to_numpy(dtype=np.float64)
    for name in args.variants:
        predictors = VARIANTS[name]
        drawn = holdout_draws(
            donor,
            predictors,
            folds=args.folds,
            seed=args.seed,
            n_estimators=args.n_estimators,
        )
        fit = scf_education_debt_donor_fit_table(donor, imputed=drawn)
        metrics = fit_metrics(observed, fit)
        metrics["household_rank_correlation"] = _rank_correlation(
            observed_total,
            drawn[SCF_EDUCATION_DEBT_TOTAL_TARGET].to_numpy(dtype=np.float64),
        )
        metrics["parent_plus_drawn_without_observed_parent_plus"] = (
            parent_plus_draw_rate(donor, drawn)
        )
        receipt["holdout"]["variants"][name] = {
            "predictors": list(predictors),
            "fit": fit,
            "metrics": metrics,
        }
        print(name, json.dumps(metrics, indent=None, default=float))

    if args.h5 is not None:
        host_sha256 = _sha256(args.h5)
        base = load_host_frame(args.h5)
        receipt["host_settings"] = {}
        for name, (label, definition, predictors) in HOST_SETTINGS.items():
            if name == BUILD_SETTING:
                continue
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
            setting_frame = Frame(
                tables,
                base.schema,
                {entity: base.weights_for(entity) for entity in base.weighted_entities},
            )
            setting_fit = us_scf_education_debt_fit_table(setting_frame, definition)
            receipt["host_settings"][name] = {
                "label": label,
                "definition": asdict(definition),
                "predictors": list(predictors),
                "fit": setting_fit,
                "metrics_vs_scf": fit_metrics(observed, setting_fit),
            }
        # The build runs through the stage entry point and its signal gate.
        frame = with_us_scf_education_debt_inputs(
            base,
            seed=args.seed,
            time_period=2024,
            scf_education_debt_donor=donor,
            n_estimators=args.n_estimators,
        )
        gate = us_scf_education_debt_signal_gate(frame)
        host_fit = us_scf_education_debt_fit_table(frame)
        label, definition, predictors = HOST_SETTINGS[BUILD_SETTING]
        receipt["host_settings"][BUILD_SETTING] = {
            "label": label,
            "definition": asdict(definition),
            "predictors": list(predictors),
            "fit": host_fit,
            "metrics_vs_scf": fit_metrics(observed, host_fit),
        }
        receipt["host"] = {
            "release": args.host_release,
            "path": args.h5.name,
            "sha256": host_sha256,
            "is_the_published_default": host_sha256 == DEFAULT_HOST_SHA256,
            "households": int(len(frame.table("household"))),
            "people": int(len(frame.table("person"))),
            "seed": args.seed,
            "gate_passed": bool(gate.passed),
            "gate_failures": list(gate.failures),
            "household_income_columns_present": gate.details.get(
                "household_income_columns_present"
            ),
            "fit": host_fit,
            "metrics_vs_scf": fit_metrics(observed, host_fit),
        }
        print("host", json.dumps(receipt["host"]["metrics_vs_scf"], default=float))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(receipt, indent=2, sort_keys=False, default=float) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.out_md.write_text(markdown(receipt), encoding="utf-8", newline="\n")
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
