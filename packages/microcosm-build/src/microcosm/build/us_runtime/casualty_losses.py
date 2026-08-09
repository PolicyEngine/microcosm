"""IRS PUF casualty-loss inputs for the US build.

The retired eCPS pipeline carried ``casualty_loss`` directly from IRS PUF
field ``E20500``. The immutable archived source coordinate is exposed as
``CASUALTY_LOSS_ARCHIVED_DERIVATION_URL`` below.

There is no CPS analogue and no synthetic fallback.  The PUF tax-detail stage
therefore derives the source value exactly, then its existing weighted QRF
places that source-backed detail on the dedicated PUF support channel.  Because
casualty losses are rare, the shared PUF runtime sparsifies the predictions to
the donor's weighted positive rate while preserving their weighted total.

PolicyEngine-US owns the casualty-loss deduction formula and its AGI floor.
This module persists only the factual person input, never the computed
deduction.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.frame import Frame

__all__ = [
    "CASUALTY_LOSS_ARCHIVED_DERIVATION_URL",
    "US_CASUALTY_LOSS_NONCONSTANT_PERSON_COLUMNS",
    "US_CASUALTY_LOSS_OUTPUT_COLUMNS",
    "US_CASUALTY_LOSS_STAGE_NAME",
    "derive_us_casualty_loss_from_puf",
    "us_casualty_loss_signal_gate",
    "us_casualty_loss_stage_spec",
    "us_casualty_loss_summary",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
CASUALTY_LOSS_ARCHIVED_DERIVATION_URL = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/datasets/puf/puf.py#L636-L643"
)

US_CASUALTY_LOSS_STAGE_NAME = "puf_tax_detail"
US_CASUALTY_LOSS_OUTPUT_COLUMNS: tuple[str, ...] = ("casualty_loss",)
US_CASUALTY_LOSS_NONCONSTANT_PERSON_COLUMNS = US_CASUALTY_LOSS_OUTPUT_COLUMNS

_CASUALTY_LOSS_NONZERO_SHARE_BAND = (0.0001, 0.02)


def us_casualty_loss_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_CASUALTY_LOSS_STAGE_NAME]
    missing = sorted(set(US_CASUALTY_LOSS_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_CASUALTY_LOSS_STAGE_NAME!r} manifest stage does not declare "
            f"casualty-loss output(s) {missing}."
        )
    return spec


def derive_us_casualty_loss_from_puf(
    puf: pd.DataFrame,
    *,
    source_column: str = "E20500",
    output_column: str = "casualty_loss",
) -> pd.DataFrame:
    """Carry the archived IRS PUF casualty-loss field without alteration.

    Invalid or negative source values fail closed.  Replacing them with zeros
    would invent observations and could silently erase the reform input this
    stage exists to restore.
    """

    if source_column not in puf.columns:
        raise ValueError(
            f"PUF casualty-loss derivation requires source column {source_column!r}."
        )
    values = pd.to_numeric(puf[source_column], errors="coerce")
    numeric = values.to_numpy(dtype=np.float64)
    nonfinite = ~np.isfinite(numeric)
    if bool(nonfinite.any()):
        raise ValueError(
            f"PUF casualty-loss source column {source_column!r} contains "
            f"{int(np.count_nonzero(nonfinite))} nonnumeric or nonfinite value(s)."
        )
    negative = numeric < 0.0
    if bool(negative.any()):
        raise ValueError(
            f"PUF casualty-loss source column {source_column!r} contains "
            f"{int(np.count_nonzero(negative))} negative value(s)."
        )

    result = puf.copy(deep=True)
    result[output_column] = numeric
    return result


def us_casualty_loss_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and validity diagnostics for casualty loss."""

    person = frame.table("person")
    values = pd.to_numeric(person["casualty_loss"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    total_weight = float(weights.sum())
    positive_share = (
        float(weights[positive].sum()) / total_weight if total_weight > 0.0 else 0.0
    )
    return {
        "positive_share": positive_share,
        "positive_share_band": list(_CASUALTY_LOSS_NONZERO_SHARE_BAND),
        "unweighted_total": float(np.nansum(values)),
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (values < 0.0))),
    }


def us_casualty_loss_signal_gate(frame: Frame) -> GateResult:
    """Require a finite, nonnegative, plausibly sparse casualty-loss input."""

    person = frame.table("person")
    missing = [
        column
        for column in US_CASUALTY_LOSS_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="casualty_loss_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_casualty_loss_summary(frame)
    failures: list[str] = []
    if summary["nonfinite"]:
        failures.append(f"casualty_loss nonfinite values: {int(summary['nonfinite'])}.")
    if summary["negative"]:
        failures.append(f"casualty_loss negative values: {int(summary['negative'])}.")
    share = float(summary["positive_share"])
    low, high = summary["positive_share_band"]
    if not (low <= share <= high):
        failures.append(
            f"casualty-loss positive share {share:.6f} outside plausibility "
            f"band [{low}, {high}]."
        )
    return GateResult(
        name="casualty_loss_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
