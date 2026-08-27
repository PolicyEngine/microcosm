"""ASEC other-income decomposition and IRS PUF alimony inputs.

The retired eCPS pipeline measured recipient income from Census ASEC other-
income records and tax-return income/expense from two IRS PUF fields.  The
ASEC half splits ``OI_VAL`` into reported alimony when ``OI_OFF == 20``, strike
benefits when ``OI_OFF == 12``, and miscellaneous income otherwise.  The PUF
support half is populated from the direct ``E00800`` / ``E03500`` mappings
through the shared weighted PUF QRF.  The three ASEC outputs are mutually
exclusive so every reported amount is carried exactly once.

This module owns only factual input leaves. PolicyEngine-US owns taxable-income
and above-the-line-deduction formulas.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    has_assembled_support_metadata,
    has_support_role_metadata,
    support_role_series,
    support_source_channel_series,
)
from microcosm.frame import Frame

__all__ = [
    "ALIMONY_ASEC_ARCHIVED_DERIVATION_URL",
    "ALIMONY_PUF_ARCHIVED_DERIVATION_URL",
    "STRIKE_BENEFITS_ASEC_ARCHIVED_DERIVATION_URL",
    "US_ASEC_OTHER_INCOME_OUTPUT_COLUMNS",
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
STRIKE_BENEFITS_ASEC_ARCHIVED_DERIVATION_URL = ALIMONY_ASEC_ARCHIVED_DERIVATION_URL
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
US_ASEC_OTHER_INCOME_OUTPUT_COLUMNS: tuple[str, ...] = (
    "alimony_income",
    "strike_benefits",
    "miscellaneous_income",
)

_ASEC_ALIMONY_OTHER_INCOME_CODE = 20
_ASEC_STRIKE_BENEFITS_OTHER_INCOME_CODE = 12
_ASEC_SEPARATE_OTHER_INCOME_CODES = frozenset(
    {
        _ASEC_ALIMONY_OTHER_INCOME_CODE,
        _ASEC_STRIKE_BENEFITS_OTHER_INCOME_CODE,
    }
)
_NONZERO_SHARE_BAND = (0.00001, 0.02)


def us_alimony_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
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
    """Split reported ASEC other income into three exhaustive input leaves.

    An already materialized split is preserved, which makes the CPS-carried
    transform idempotent on a staged base artifact. If any leaf is missing,
    both raw ASEC fields are mandatory; a missing source must not be healed
    with fabricated zeros.
    """

    raw_sources = {"OI_VAL", "OI_OFF"}
    present_sources = raw_sources.intersection(person.columns)
    if not present_sources:
        materialized = set(US_ASEC_OTHER_INCOME_OUTPUT_COLUMNS)
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
    result["strike_benefits"] = np.where(
        integer_codes == _ASEC_STRIKE_BENEFITS_OTHER_INCOME_CODE,
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
    if raw_sources.issubset(person.columns):
        missing_split = sorted(
            set(US_ASEC_OTHER_INCOME_OUTPUT_COLUMNS) - set(person.columns)
        )
        if missing_split:
            failures.append(
                "ASEC raw OI_VAL/OI_OFF are present but other-income split "
                f"column(s) are missing: {missing_split}."
            )
            return GateResult(
                name="alimony_inputs_signal",
                passed=False,
                failures=tuple(failures),
                details=summary,
            )
        source_mask = np.ones(len(person), dtype=bool)
        source_reconciliation_mask = source_mask.copy()
        if has_support_role_metadata(person, entity="person"):
            source_mask = (
                support_source_channel_series(person, entity="person")
                .eq(BASE_ASEC_SUPPORT_CHANNEL)
                .to_numpy()
            )
            source_reconciliation_mask = source_mask.copy()
            if has_assembled_support_metadata(person, entity="person"):
                source_reconciliation_mask &= (
                    support_role_series(person, entity="person")
                    .eq(BASE_ASEC_SUPPORT_CHANNEL)
                    .to_numpy()
                )
        raw_amounts = pd.to_numeric(person["OI_VAL"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        raw_codes = pd.to_numeric(person["OI_OFF"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        source_invalid = source_mask & (
            ~np.isfinite(raw_amounts) | ~np.isfinite(raw_codes)
        )
        summary["asec_source_rows"] = int(np.count_nonzero(source_mask))
        summary["asec_source_invalid"] = int(np.count_nonzero(source_invalid))
        summary["asec_source_reconciliation_rows"] = int(
            np.count_nonzero(source_reconciliation_mask)
        )
        if bool(source_invalid.any()):
            failures.append(
                "ASEC OI_VAL/OI_OFF contain nonfinite source values on "
                f"{int(np.count_nonzero(source_invalid))} row(s)."
            )
        amounts = raw_amounts[source_reconciliation_mask]
        codes = raw_codes[source_reconciliation_mask]
        if bool(np.isfinite(amounts).all() and np.isfinite(codes).all()):
            integer_codes = codes.astype(np.int64)
            expected_alimony = np.where(
                integer_codes == _ASEC_ALIMONY_OTHER_INCOME_CODE,
                amounts,
                0.0,
            )
            expected_strike_benefits = np.where(
                integer_codes == _ASEC_STRIKE_BENEFITS_OTHER_INCOME_CODE,
                amounts,
                0.0,
            )
            expected_miscellaneous = np.where(
                np.isin(integer_codes, list(_ASEC_SEPARATE_OTHER_INCOME_CODES)),
                0.0,
                amounts,
            )
            actual_alimony = pd.to_numeric(
                person.loc[source_reconciliation_mask, "alimony_income"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            actual_strike_benefits = pd.to_numeric(
                person.loc[source_reconciliation_mask, "strike_benefits"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            actual_miscellaneous = pd.to_numeric(
                person.loc[source_reconciliation_mask, "miscellaneous_income"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            alimony_mismatch = ~np.isclose(actual_alimony, expected_alimony)
            strike_benefits_mismatch = ~np.isclose(
                actual_strike_benefits,
                expected_strike_benefits,
            )
            miscellaneous_mismatch = ~np.isclose(
                actual_miscellaneous,
                expected_miscellaneous,
            )
            conservation_mismatch = ~np.isclose(
                actual_alimony + actual_strike_benefits + actual_miscellaneous,
                amounts,
            )
            if bool(alimony_mismatch.any()):
                failures.append(
                    "ASEC alimony_income disagrees with OI_OFF == 20 / OI_VAL "
                    f"on {int(np.count_nonzero(alimony_mismatch))} row(s)."
                )
            if bool(strike_benefits_mismatch.any()):
                failures.append(
                    "ASEC strike_benefits disagrees with OI_OFF == 12 / OI_VAL "
                    f"on {int(np.count_nonzero(strike_benefits_mismatch))} row(s)."
                )
            if bool(miscellaneous_mismatch.any()):
                failures.append(
                    "ASEC miscellaneous_income does not exclude alimony/strike "
                    f"codes on {int(np.count_nonzero(miscellaneous_mismatch))} row(s)."
                )
            if bool(conservation_mismatch.any()):
                failures.append(
                    "ASEC other-income split does not conserve OI_VAL on "
                    f"{int(np.count_nonzero(conservation_mismatch))} row(s)."
                )
    return GateResult(
        name="alimony_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
