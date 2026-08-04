"""IRS PUF miscellaneous-itemized input for the US build.

The retired eCPS pipeline used IRS PUF ``E20400`` (unreimbursed employee
business expenses) as its explicit proxy for all expenses eligible for the
miscellaneous itemized deduction. The immutable archived source coordinate is
exposed as ``MISC_ITEMIZED_ARCHIVED_DERIVATION_URL`` below.

There is no CPS amount and no synthetic fallback. The PUF tax-detail stage
carries the source exactly, then its existing weighted QRF places that
source-backed detail on the dedicated PUF support channel. PolicyEngine-US owns
the two-percent-of-AGI floor and deduction formula; this module persists only
the factual person input.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.frame import Frame

__all__ = [
    "MISC_ITEMIZED_ARCHIVED_DERIVATION_URL",
    "US_MISC_ITEMIZED_NONCONSTANT_PERSON_COLUMNS",
    "US_MISC_ITEMIZED_OUTPUT_COLUMNS",
    "US_MISC_ITEMIZED_STAGE_NAME",
    "derive_us_misc_itemized_from_puf",
    "us_misc_itemized_signal_gate",
    "us_misc_itemized_stage_spec",
    "us_misc_itemized_summary",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
MISC_ITEMIZED_ARCHIVED_DERIVATION_URL = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/datasets/puf/puf.py#L663-L665"
)

US_MISC_ITEMIZED_STAGE_NAME = "puf_tax_detail"
US_MISC_ITEMIZED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "unreimbursed_business_employee_expenses",
)
US_MISC_ITEMIZED_NONCONSTANT_PERSON_COLUMNS = US_MISC_ITEMIZED_OUTPUT_COLUMNS

_MISC_ITEMIZED_NONZERO_SHARE_BAND = (0.05, 0.55)


def us_misc_itemized_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_MISC_ITEMIZED_STAGE_NAME]
    missing = sorted(set(US_MISC_ITEMIZED_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_MISC_ITEMIZED_STAGE_NAME!r} manifest stage does not declare "
            f"miscellaneous-itemized output(s) {missing}."
        )
    return spec


def derive_us_misc_itemized_from_puf(
    puf: pd.DataFrame,
    *,
    source_column: str = "E20400",
    output_column: str = "unreimbursed_business_employee_expenses",
) -> pd.DataFrame:
    """Carry the archived IRS PUF miscellaneous-expense proxy exactly.

    Invalid or negative source values fail closed. Replacing them with zeros
    would invent observations and could silently erase the reform input this
    stage exists to restore.
    """

    if source_column not in puf.columns:
        raise ValueError(
            f"PUF miscellaneous-itemized derivation requires source column "
            f"{source_column!r}."
        )
    values = pd.to_numeric(puf[source_column], errors="coerce")
    numeric = values.to_numpy(dtype=np.float64)
    nonfinite = ~np.isfinite(numeric)
    if bool(nonfinite.any()):
        raise ValueError(
            f"PUF miscellaneous-itemized source column {source_column!r} contains "
            f"{int(np.count_nonzero(nonfinite))} nonnumeric or nonfinite value(s)."
        )
    negative = numeric < 0.0
    if bool(negative.any()):
        raise ValueError(
            f"PUF miscellaneous-itemized source column {source_column!r} contains "
            f"{int(np.count_nonzero(negative))} negative value(s)."
        )

    result = puf.copy(deep=True)
    result[output_column] = numeric
    return result


def us_misc_itemized_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and validity diagnostics for the proxy input."""

    person = frame.table("person")
    values = pd.to_numeric(
        person["unreimbursed_business_employee_expenses"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    total_weight = float(weights.sum())
    positive_share = (
        float(weights[positive].sum()) / total_weight if total_weight > 0.0 else 0.0
    )
    return {
        "positive_share": positive_share,
        "positive_share_band": list(_MISC_ITEMIZED_NONZERO_SHARE_BAND),
        "unweighted_total": float(np.nansum(values)),
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (values < 0.0))),
    }


def us_misc_itemized_signal_gate(frame: Frame) -> GateResult:
    """Require a finite, nonnegative, nondefault miscellaneous-expense input."""

    person = frame.table("person")
    missing = [
        column
        for column in US_MISC_ITEMIZED_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="misc_itemized_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_misc_itemized_summary(frame)
    failures: list[str] = []
    if summary["nonfinite"]:
        failures.append(
            "unreimbursed_business_employee_expenses nonfinite values: "
            f"{int(summary['nonfinite'])}."
        )
    if summary["negative"]:
        failures.append(
            "unreimbursed_business_employee_expenses negative values: "
            f"{int(summary['negative'])}."
        )
    share = float(summary["positive_share"])
    low, high = summary["positive_share_band"]
    if not (low <= share <= high):
        failures.append(
            f"miscellaneous-itemized positive share {share:.6f} outside "
            f"plausibility band [{low}, {high}]."
        )
    return GateResult(
        name="misc_itemized_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
