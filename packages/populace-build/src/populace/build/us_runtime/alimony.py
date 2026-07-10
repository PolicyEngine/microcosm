"""ASEC/IRS PUF alimony inputs for the US build.

The retired eCPS pipeline measured recipient income from Census ASEC other-
income records and tax-return income/expense from two IRS PUF fields.  The
ASEC half keeps reported ``OI_VAL`` only when ``OI_OFF == 20``; the PUF support
half is populated from the direct ``E00800`` / ``E03500`` mappings through the
shared weighted PUF QRF.  ``miscellaneous_income`` must simultaneously exclude
alimony (and the retired pipeline's strike-benefit code 12), otherwise the ASEC
amount is counted twice in gross income.

This module owns only factual input leaves. PolicyEngine-US owns taxable-income
and above-the-line-deduction formulas.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.frame import Frame

__all__ = [
    "ALIMONY_ASEC_ARCHIVED_DERIVATION_URL",
    "ALIMONY_PUF_ARCHIVED_DERIVATION_URL",
    "US_ALIMONY_NONCONSTANT_PERSON_COLUMNS",
    "US_ALIMONY_OUTPUT_COLUMNS",
    "US_ALIMONY_STAGE_NAME",
    "derive_us_alimony_from_asec",
    "derive_us_alimony_from_puf",
    "us_alimony_signal_gate",
    "us_alimony_stage_spec",
    "us_alimony_summary",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
ALIMONY_ASEC_ARCHIVED_DERIVATION_URL = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/{_ARCHIVED_COMMIT}/"
    "policyengine_" + "us_data/datasets/cps/cps.py#L1481-L1492"
)
ALIMONY_PUF_ARCHIVED_DERIVATION_URL = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/{_ARCHIVED_COMMIT}/"
    "policyengine_" + "us_data/datasets/puf/puf.py#L640-L641"
)

US_ALIMONY_STAGE_NAME = "puf_tax_detail"
US_ALIMONY_OUTPUT_COLUMNS: tuple[str, ...] = (
    "alimony_income",
    "alimony_expense",
)
US_ALIMONY_NONCONSTANT_PERSON_COLUMNS = US_ALIMONY_OUTPUT_COLUMNS

_ASEC_ALIMONY_OTHER_INCOME_CODE = 20
_ASEC_SEPARATE_OTHER_INCOME_CODES = frozenset({12, 20})
_NONZERO_SHARE_BAND = (0.00001, 0.02)


def us_alimony_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_ALIMONY_STAGE_NAME]
    missing = sorted(set(US_ALIMONY_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_ALIMONY_STAGE_NAME!r} manifest stage does not declare "
            f"alimony output(s) {missing}."
        )
    return spec


def _strict_numeric_source(
    table: pd.DataFrame,
    column: str,
    *,
    label: str,
    nonnegative: bool,
) -> np.ndarray:
    if column not in table.columns:
        raise ValueError(f"{label} requires source column {column!r}.")
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = ~np.isfinite(values)
    if bool(nonfinite.any()):
        raise ValueError(
            f"{label} source column {column!r} contains "
            f"{int(np.count_nonzero(nonfinite))} nonnumeric or nonfinite value(s)."
        )
    if nonnegative:
        negative = values < 0.0
        if bool(negative.any()):
            raise ValueError(
                f"{label} source column {column!r} contains "
                f"{int(np.count_nonzero(negative))} negative value(s)."
            )
    return values


def derive_us_alimony_from_asec(person: pd.DataFrame) -> pd.DataFrame:
    """Carry reported ASEC alimony and remove it from miscellaneous income.

    An already materialized pair is preserved, which makes the CPS-carried
    transform idempotent on a staged base artifact. If either leaf is missing,
    both raw ASEC fields are mandatory; a missing source must not be healed with
    fabricated zeros.
    """

    raw_sources = {"OI_VAL", "OI_OFF"}
    present_sources = raw_sources.intersection(person.columns)
    if not present_sources:
        materialized = {"alimony_income", "miscellaneous_income"}
        if materialized.issubset(person.columns):
            return person.copy(deep=True)
        missing = sorted(materialized - set(person.columns))
        raise ValueError(
            "ASEC alimony derivation cannot reuse an unstaged table without "
            f"raw OI_VAL/OI_OFF; missing materialized column(s) {missing}."
        )
    if present_sources != raw_sources:
        missing = sorted(raw_sources - set(person.columns))
        raise ValueError(
            "ASEC alimony derivation requires both raw source columns; "
            f"missing {missing}."
        )

    label = "ASEC alimony derivation"
    amounts = _strict_numeric_source(
        person,
        "OI_VAL",
        label=label,
        nonnegative=True,
    )
    codes = _strict_numeric_source(
        person,
        "OI_OFF",
        label=label,
        nonnegative=True,
    )
    noninteger = codes != np.floor(codes)
    if bool(noninteger.any()):
        raise ValueError(
            "ASEC alimony derivation source column 'OI_OFF' contains "
            f"{int(np.count_nonzero(noninteger))} noninteger code(s)."
        )
    integer_codes = codes.astype(np.int64)

    result = person.copy(deep=True)
    result["alimony_income"] = np.where(
        integer_codes == _ASEC_ALIMONY_OTHER_INCOME_CODE,
        amounts,
        0.0,
    )
    result["miscellaneous_income"] = np.where(
        np.isin(integer_codes, list(_ASEC_SEPARATE_OTHER_INCOME_CODES)),
        0.0,
        amounts,
    )
    return result


def derive_us_alimony_from_puf(
    puf: pd.DataFrame,
    *,
    income_source_column: str = "E00800",
    income_output_column: str = "alimony_income",
    expense_source_column: str = "E03500",
    expense_output_column: str = "alimony_expense",
) -> pd.DataFrame:
    """Carry the archived PUF alimony income and expense fields exactly."""

    income = _strict_numeric_source(
        puf,
        income_source_column,
        label="PUF alimony derivation",
        nonnegative=True,
    )
    expense = _strict_numeric_source(
        puf,
        expense_source_column,
        label="PUF alimony derivation",
        nonnegative=True,
    )
    result = puf.copy(deep=True)
    result[income_output_column] = income
    result[expense_output_column] = expense
    return result


def us_alimony_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and validity diagnostics for both leaves."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    columns: dict[str, object] = {}
    for column in US_ALIMONY_OUTPUT_COLUMNS:
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(values)
        positive = finite & (values > 0.0)
        positive_share = (
            float(weights[positive].sum()) / total_weight if total_weight > 0.0 else 0.0
        )
        columns[column] = {
            "positive_share": positive_share,
            "positive_share_band": list(_NONZERO_SHARE_BAND),
            "unweighted_total": float(np.nansum(values)),
            "nonfinite": int(np.count_nonzero(~finite)),
            "negative": int(np.count_nonzero(finite & (values < 0.0))),
        }
    return {"columns": columns}


def us_alimony_signal_gate(frame: Frame) -> GateResult:
    """Require finite, nonnegative, plausibly sparse signal in both leaves."""

    person = frame.table("person")
    missing = [column for column in US_ALIMONY_OUTPUT_COLUMNS if column not in person]
    if missing:
        return GateResult(
            name="alimony_inputs_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_alimony_summary(frame)
    failures: list[str] = []
    column_summaries = summary["columns"]
    for column in US_ALIMONY_OUTPUT_COLUMNS:
        details = column_summaries[column]
        if details["nonfinite"]:
            failures.append(
                f"{column}: {int(details['nonfinite'])} nonfinite value(s)."
            )
        if details["negative"]:
            failures.append(f"{column}: {int(details['negative'])} negative value(s).")
        share = float(details["positive_share"])
        low, high = details["positive_share_band"]
        if not (low <= share <= high):
            failures.append(
                f"{column}: positive share {share:.6f} outside plausibility "
                f"band [{low}, {high}]."
            )
    raw_sources = {"OI_VAL", "OI_OFF"}
    if raw_sources.issubset(person.columns) and "miscellaneous_income" not in person:
        failures.append(
            "ASEC raw OI_VAL/OI_OFF are present but miscellaneous_income is missing."
        )
    elif raw_sources.issubset(person.columns):
        source_mask = np.ones(len(person), dtype=bool)
        if "person_support_channel" in person:
            source_mask = (
                person["person_support_channel"].to_numpy(dtype=object) == "asec"
            )
        amounts = pd.to_numeric(person["OI_VAL"], errors="coerce").to_numpy(
            dtype=np.float64
        )[source_mask]
        codes = pd.to_numeric(person["OI_OFF"], errors="coerce").to_numpy(
            dtype=np.float64
        )[source_mask]
        if bool(np.isfinite(amounts).all() and np.isfinite(codes).all()):
            integer_codes = codes.astype(np.int64)
            expected_alimony = np.where(
                integer_codes == _ASEC_ALIMONY_OTHER_INCOME_CODE,
                amounts,
                0.0,
            )
            expected_miscellaneous = np.where(
                np.isin(integer_codes, list(_ASEC_SEPARATE_OTHER_INCOME_CODES)),
                0.0,
                amounts,
            )
            actual_alimony = pd.to_numeric(
                person.loc[source_mask, "alimony_income"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            actual_miscellaneous = pd.to_numeric(
                person.loc[source_mask, "miscellaneous_income"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            alimony_mismatch = ~np.isclose(actual_alimony, expected_alimony)
            miscellaneous_mismatch = ~np.isclose(
                actual_miscellaneous,
                expected_miscellaneous,
            )
            if bool(alimony_mismatch.any()):
                failures.append(
                    "ASEC alimony_income disagrees with OI_OFF == 20 / OI_VAL "
                    f"on {int(np.count_nonzero(alimony_mismatch))} row(s)."
                )
            if bool(miscellaneous_mismatch.any()):
                failures.append(
                    "ASEC miscellaneous_income does not exclude alimony/strike "
                    f"codes on {int(np.count_nonzero(miscellaneous_mismatch))} row(s)."
                )
    return GateResult(
        name="alimony_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
