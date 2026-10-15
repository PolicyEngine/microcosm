#!/usr/bin/env python
"""Fit receipts for the SCF 2025 federal student loan stage (microcosm#1073).

Three measurements, all weighted, all against the 2025 SCF:

1. **Source checks.** The loader's education debt against the Fed's published
   Alternate Table 13 (19.92% of families hold education debt, median
   $25,500), and how the federal flag, X7179 and the whose-education codes
   split the balances the X7179 and holder rules act on.
2. **Hold-out fit and predictor ablation.** Families are split into five folds
   by ``yy1`` (all implicates of a family stay together). Each fold is drawn
   from the stage's two QRFs fit on the other four, and the drawn table is
   compared with the observed one, for each predictor set.
3. **Host fit.** The stage imputed onto a US Microcosm H5 (the published
   default unless ``--h5`` names another), tabulated at that file's weights.

Nothing here is a gate or a calibration target. Run with the repository's
engine-free environment:

    .venv/Scripts/python.exe experiments/build_1073_scf_education_debt_receipt.py \
        --scf-zip-dir ~/.cache/scf --h5 populace_us_2024.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.scf_education_debt import (
    SCF_2025_FULL_EXTRACT,
    SCF_2025_SUMMARY_EXTRACT,
    SCF_EDUCATION_DEBT_BALANCE_BANDS,
    SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS,
    SCF_EDUCATION_DEBT_PREDICTORS,
    SCF_EDUCATION_DEBT_TOTAL_TARGET,
    SCF_EDUCATION_LOAN_BALANCE_COLUMNS,
    SCF_EDUCATION_LOAN_FEDERAL_COLUMNS,
    SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN,
    SCF_EDUCATION_LOAN_WHOSE_COLUMNS,
    ScfPublicFile,
    _read_stata_columns,
    draw_scf_education_debt_households,
    fetch_scf_public_file,
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
#: The published default the host fit imputes onto, and its release-manifest
#: SHA-256 (releases/populace-us-2024-spm-20260915/release_manifest.json).
DEFAULT_HOST_RELEASE = "populace-us-2024-spm-20260915"
DEFAULT_HOST_SHA256 = "6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84"
PRODUCTION_VARIANT = "shared_8_plus_household_income"
VARIANTS: Mapping[str, tuple[str, ...]] = {
    "shared_8_without_age": tuple(p for p in SCF_WEALTH_PREDICTORS if p != "age"),
    "shared_8": SCF_WEALTH_PREDICTORS,
    PRODUCTION_VARIANT: SCF_EDUCATION_DEBT_PREDICTORS,
    "plus_household_income_and_education": (*SCF_EDUCATION_DEBT_PREDICTORS, "educ"),
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
            donor.loc[held_out, list(predictors)],
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
    columns = [("SCF 2025", observed), ("SCF hold-out draw", holdout)]
    if host is not None:
        columns.append((f"Imputed onto {receipt['host']['release']}", host))
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
        "`scf-2025-education-debt`. Machine-readable copy: "
        "`experiments/receipts/1073-scf-education-debt-fit.json`.",
        "",
        "Every figure is weighted. SCF rows pool the five implicates with `wgt` "
        "(X42001/5). Household figures group by the reference person's age and "
        "race and by weighted quintiles of household income less capital gains, "
        "the income concept the imputation conditions on. Borrower figures count "
        "people with a federal balance (own plus Parent PLUS), in Federal Student "
        "Aid's balance and age bands; SCF borrowers are the reference person and "
        "spouse. Nothing here is a gate or a calibration target.",
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
        "| Federal balances on loans 1-6, by whose education | Share |",
        "|---|---:|",
    ]
    for key, value in checks["federal_balance_share_by_whose_education_code"].items():
        lines.append(f"| {key} | {_pct(value)} |")
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
            "IBR new borrowers among borrowers",
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
        header,
        rule,
    ]
    for label, _, _ in SCF_EDUCATION_DEBT_BALANCE_BANDS:
        lines.append(
            row(label, lambda t, k=label: t["borrowers"]["balance_bands"][k], _pct)
        )
    lines += ["", "### Borrowers by FSA age band", "", header, rule]
    for label, _, _ in SCF_EDUCATION_DEBT_BORROWER_AGE_BANDS:
        lines.append(
            row(label, lambda t, k=label: t["borrowers"]["age_bands"][k], _pct)
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
        "microcosm#297, so that row shows what the predictor would add.",
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
        lines += [
            "",
            "## Host fit",
            "",
            f"Imputed with build seed {receipt['host']['seed']} onto "
            f"`{receipt['host']['release']}` ({receipt['host']['households']:,} "
            f"households, sha256 `{receipt['host']['sha256'][:12]}...`) at its "
            "calibrated household weights; the published weights were calibrated "
            "without these columns. Signal gate: "
            f"{'passed' if receipt['host']['gate_passed'] else 'FAILED'}.",
            "",
            "| Metric (imputed minus SCF 2025) | Value |",
            "|---|---:|",
        ]
        for key, value in receipt["host"]["metrics_vs_scf"].items():
            lines.append(f"| {key} | {value:.3f} |")
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
        full_path,
        (
            "y1",
            "yy1",
            "x42001",
            *SCF_EDUCATION_LOAN_BALANCE_COLUMNS,
            SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN,
            *SCF_EDUCATION_LOAN_FEDERAL_COLUMNS,
            *SCF_EDUCATION_LOAN_WHOSE_COLUMNS,
        ),
        label="2025 full file",
    )
    full["wgt"] = full["x42001"] / 5.0
    observed = scf_education_debt_donor_fit_table(donor)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "issue": ISSUE,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "scf_files": {
            public_file.member: {
                "url": public_file.url,
                "zip_sha256": public_file.zip_sha256,
                "member_sha256": public_file.member_sha256,
            }
            for public_file in (SCF_2025_SUMMARY_EXTRACT, SCF_2025_FULL_EXTRACT)
        },
        "source_checks": source_checks(donor, full),
        "scf_2025_observed": observed,
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
        receipt["holdout"]["variants"][name] = {
            "predictors": list(predictors),
            "fit": fit,
            "metrics": metrics,
        }
        print(name, json.dumps(metrics, indent=None, default=float))

    if args.h5 is not None:
        host_sha256 = _sha256(args.h5)
        frame = load_host_frame(args.h5)
        frame = with_us_scf_education_debt_inputs(
            frame,
            seed=args.seed,
            time_period=2024,
            scf_education_debt_donor=donor,
            n_estimators=args.n_estimators,
        )
        gate = us_scf_education_debt_signal_gate(frame)
        host_fit = us_scf_education_debt_fit_table(frame)
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
