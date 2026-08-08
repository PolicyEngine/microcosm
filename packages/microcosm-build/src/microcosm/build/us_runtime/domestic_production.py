"""IRS PUF domestic-production deduction input for the US build.

The retired eCPS pipeline carried ``domestic_production_ald`` directly from
IRS PUF field ``E03240``.  This is the former Section 199 domestic production
activities deduction, not the later Section 199A qualified-business-income
deduction.  The immutable archived derivation, export list, and versioned PUF
artifact coordinates are exposed below.

There is no CPS analogue and no synthetic fallback.  The PUF tax-detail stage
derives the source value exactly, then its weighted QRF places that source-backed
tax-unit input on the dedicated PUF support channel.  The support runtime
preserves the donor's sparse positive rate rather than smearing a rare PUF
amount across the population.  The versioned 2024 PUF is 2015-based and uprates
this legacy field; it is a frozen counterfactual source value, not observed
current-law 2024 deduction activity.

PolicyEngine-US owns which years include the input in above-the-line
deductions.  Microcosm persists the archived input even though current-law years
after 2017 exclude the former deduction from adjusted gross income.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
from microcosm.frame import Frame

__all__ = [
    "DOMESTIC_PRODUCTION_ALD_ARCHIVED_DERIVATION_URL",
    "DOMESTIC_PRODUCTION_ALD_ARCHIVED_EXPORT_URL",
    "DOMESTIC_PRODUCTION_ALD_ARCHIVED_IMPUTATION_URL",
    "DOMESTIC_PRODUCTION_ALD_ARCHIVED_PUF_ARTIFACT_URL",
    "US_DOMESTIC_PRODUCTION_ALD_NONCONSTANT_TAX_UNIT_COLUMNS",
    "US_DOMESTIC_PRODUCTION_ALD_OUTPUT_COLUMNS",
    "US_DOMESTIC_PRODUCTION_ALD_STAGE_NAME",
    "derive_us_domestic_production_ald_from_puf",
    "us_domestic_production_ald_signal_gate",
    "us_domestic_production_ald_stage_spec",
    "us_domestic_production_ald_summary",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
DOMESTIC_PRODUCTION_ALD_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L646"
)
DOMESTIC_PRODUCTION_ALD_ARCHIVED_EXPORT_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L808-L815"
)
DOMESTIC_PRODUCTION_ALD_ARCHIVED_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "calibration/puf_impute.py#L90-L198"
)
DOMESTIC_PRODUCTION_ALD_ARCHIVED_PUF_ARTIFACT_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L1655-L1660"
)

US_DOMESTIC_PRODUCTION_ALD_STAGE_NAME = "puf_tax_detail"
US_DOMESTIC_PRODUCTION_ALD_OUTPUT_COLUMNS: tuple[str, ...] = (
    "domestic_production_ald",
)
US_DOMESTIC_PRODUCTION_ALD_NONCONSTANT_TAX_UNIT_COLUMNS = (
    US_DOMESTIC_PRODUCTION_ALD_OUTPUT_COLUMNS
)

# The pinned full PUF donor has a 0.4808% weighted positive rate.  The lower
# bound rejects an all-default export; the upper bound catches QRF smearing
# while allowing sampling and support-channel composition differences.
_DOMESTIC_PRODUCTION_ALD_POSITIVE_SHARE_BAND = (0.0001, 0.02)


def us_domestic_production_ald_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_DOMESTIC_PRODUCTION_ALD_STAGE_NAME]
    missing = sorted(set(US_DOMESTIC_PRODUCTION_ALD_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_DOMESTIC_PRODUCTION_ALD_STAGE_NAME!r} manifest stage does not "
            f"declare domestic-production output(s) {missing}."
        )
    return spec


def derive_us_domestic_production_ald_from_puf(
    puf: pd.DataFrame,
    *,
    source_column: str = "E03240",
    output_column: str = "domestic_production_ald",
) -> pd.DataFrame:
    """Carry the archived IRS PUF E03240 field without alteration.

    Invalid or negative source values fail closed.  Replacing them with zeros
    would invent observations and could silently erase the input this stage
    exists to restore.
    """

    if source_column not in puf.columns:
        raise ValueError(
            "PUF domestic-production-ALD derivation requires source column "
            f"{source_column!r}."
        )
    values = pd.to_numeric(puf[source_column], errors="coerce")
    numeric = values.to_numpy(dtype=np.float64)
    nonfinite = ~np.isfinite(numeric)
    if bool(nonfinite.any()):
        raise ValueError(
            "PUF domestic-production-ALD source column "
            f"{source_column!r} contains "
            f"{int(np.count_nonzero(nonfinite))} nonnumeric or nonfinite value(s)."
        )
    negative = numeric < 0.0
    if bool(negative.any()):
        raise ValueError(
            "PUF domestic-production-ALD source column "
            f"{source_column!r} contains "
            f"{int(np.count_nonzero(negative))} negative value(s)."
        )

    result = puf.copy(deep=True)
    result[output_column] = numeric
    return result


def us_domestic_production_ald_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and validity diagnostics for the tax-unit input."""

    tax_unit = frame.table("tax_unit")
    values = pd.to_numeric(
        tax_unit["domestic_production_ald"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    weights = np.asarray(
        frame.resolve_weights("tax_unit").values,
        dtype=np.float64,
    )
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    total_weight = float(weights.sum())
    positive_share = (
        float(weights[positive].sum()) / total_weight if total_weight > 0.0 else 0.0
    )
    summary: dict[str, object] = {
        "positive_share": positive_share,
        "positive_share_band": list(_DOMESTIC_PRODUCTION_ALD_POSITIVE_SHARE_BAND),
        "unweighted_total": float(np.nansum(values)),
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (values < 0.0))),
    }
    if has_support_role_metadata(tax_unit, entity="tax_unit"):
        role_values = support_role_series(tax_unit, entity="tax_unit").to_numpy()
        channels: dict[str, dict[str, float | int]] = {}
        for channel in pd.unique(role_values):
            channel_mask = role_values == channel
            channel_weight = float(weights[channel_mask].sum())
            channel_positive = channel_mask & positive
            channels[str(channel)] = {
                "positive_rows": int(np.count_nonzero(channel_positive)),
                "positive_share": (
                    float(weights[channel_positive].sum()) / channel_weight
                    if channel_weight > 0.0
                    else 0.0
                ),
                "weighted_total": float(
                    (weights[channel_mask] * values[channel_mask]).sum()
                ),
            }
        summary["channels"] = channels
    return summary


def us_domestic_production_ald_signal_gate(frame: Frame) -> GateResult:
    """Require a finite, nonnegative, plausibly sparse tax-unit input."""

    tax_unit = frame.table("tax_unit")
    missing = [
        column
        for column in US_DOMESTIC_PRODUCTION_ALD_OUTPUT_COLUMNS
        if column not in tax_unit.columns
    ]
    if missing:
        return GateResult(
            name="domestic_production_ald_signal",
            passed=False,
            failures=(f"tax_unit columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_domestic_production_ald_summary(frame)
    failures: list[str] = []
    if summary["nonfinite"]:
        failures.append(
            f"domestic_production_ald nonfinite values: {int(summary['nonfinite'])}."
        )
    if summary["negative"]:
        failures.append(
            f"domestic_production_ald negative values: {int(summary['negative'])}."
        )
    share = float(summary["positive_share"])
    low, high = summary["positive_share_band"]
    if not (low <= share <= high):
        failures.append(
            f"domestic-production-ALD positive share {share:.6f} outside "
            f"plausibility band [{low}, {high}]."
        )
    channels = summary.get("channels")
    if isinstance(channels, dict):
        asec = channels.get("asec")
        puf = channels.get("puf_tax_detail")
        if isinstance(asec, dict) and int(asec.get("positive_rows", 0)):
            failures.append(
                "domestic_production_ald has positive values on the ASEC support "
                "channel, which has no E03240 source."
            )
        if not isinstance(puf, dict) or not int(puf.get("positive_rows", 0)):
            failures.append(
                "domestic_production_ald has no positive PUF tax-detail support values."
            )
    return GateResult(
        name="domestic_production_ald_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
