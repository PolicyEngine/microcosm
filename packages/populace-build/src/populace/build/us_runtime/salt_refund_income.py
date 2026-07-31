"""IRS PUF state-and-local-tax-refund income for the US build.

The retired eCPS pipeline carried ``salt_refund_income`` directly from IRS
PUF field ``E00700``.  The immutable archived derivation, exported field,
PUF QRF target/override lists, person-allocation path, and versioned PUF
artifact coordinates are exposed below.

There is no ASEC analogue and no synthetic fallback.  The shared PUF
tax-detail stage reduces the processed person array to the identifiable
tax-unit total, imputes that source-backed total only onto the dedicated PUF
support channel, and places it on the unit's first person.  PolicyEngine-US
sums this person input at tax-unit grain, so reproducing the retired randomized
filer/spouse ``EARNSPLIT`` allocation would add noise without changing policy
semantics.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
from populace.frame import Frame

__all__ = [
    "SALT_REFUND_ARCHIVED_DERIVATION_URL",
    "SALT_REFUND_ARCHIVED_EXPORT_URL",
    "SALT_REFUND_ARCHIVED_IMPUTATION_URL",
    "SALT_REFUND_ARCHIVED_PERSON_ALLOCATION_URL",
    "SALT_REFUND_ARCHIVED_PUF_ARTIFACT_URL",
    "US_SALT_REFUND_NONCONSTANT_PERSON_COLUMNS",
    "US_SALT_REFUND_OUTPUT_COLUMNS",
    "US_SALT_REFUND_STAGE_NAME",
    "derive_us_salt_refund_income_from_puf",
    "us_salt_refund_income_signal_gate",
    "us_salt_refund_income_stage_spec",
    "us_salt_refund_income_summary",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
SALT_REFUND_ARCHIVED_DERIVATION_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L707"
SALT_REFUND_ARCHIVED_EXPORT_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L804-L850"
SALT_REFUND_ARCHIVED_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "calibration/puf_impute.py#L90-L198"
)
SALT_REFUND_ARCHIVED_PERSON_ALLOCATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L1513-L1601"
)
SALT_REFUND_ARCHIVED_PUF_ARTIFACT_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L1655-L1660"
)

US_SALT_REFUND_STAGE_NAME = "puf_tax_detail"
US_SALT_REFUND_OUTPUT_COLUMNS: tuple[str, ...] = ("salt_refund_income",)
US_SALT_REFUND_NONCONSTANT_PERSON_COLUMNS = US_SALT_REFUND_OUTPUT_COLUMNS

_OUTPUT = US_SALT_REFUND_OUTPUT_COLUMNS[0]
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"

# The pinned processed PUF donor has a 10.74% weighted positive person share
# (13.58% at tax-unit grain).  The broad bounds reject a default-only column
# and QRF smearing while allowing support sampling and composition changes.
_OVERALL_POSITIVE_SHARE_BAND = (0.005, 0.20)
_PUF_POSITIVE_SHARE_BAND = (0.01, 0.40)


def us_salt_refund_income_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_SALT_REFUND_STAGE_NAME]
    missing = sorted(set(US_SALT_REFUND_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_SALT_REFUND_STAGE_NAME!r} manifest stage does not declare "
            f"SALT-refund output(s) {missing}."
        )
    return spec


def derive_us_salt_refund_income_from_puf(
    puf: pd.DataFrame,
    *,
    source_column: str = "E00700",
    output_column: str = _OUTPUT,
) -> pd.DataFrame:
    """Carry the archived IRS PUF E00700 field without alteration."""

    if source_column not in puf.columns:
        raise ValueError(
            f"PUF SALT-refund derivation requires source column {source_column!r}."
        )
    values = pd.to_numeric(puf[source_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    nonfinite = ~np.isfinite(values)
    if bool(nonfinite.any()):
        raise ValueError(
            f"PUF SALT-refund source column {source_column!r} contains "
            f"{int(np.count_nonzero(nonfinite))} nonnumeric or nonfinite value(s)."
        )
    negative = values < 0.0
    if bool(negative.any()):
        raise ValueError(
            f"PUF SALT-refund source column {source_column!r} contains "
            f"{int(np.count_nonzero(negative))} negative value(s)."
        )

    result = puf.copy(deep=True)
    result[output_column] = values
    return result


def us_salt_refund_income_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal, support-channel, and validity diagnostics."""

    person = frame.table("person")
    values = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(dtype=np.float64)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    total_weight = float(weights.sum())
    summary: dict[str, object] = {
        "positive_share": (
            float(weights[positive].sum()) / total_weight if total_weight > 0.0 else 0.0
        ),
        "positive_share_band": list(_OVERALL_POSITIVE_SHARE_BAND),
        "weighted_total": float((np.nan_to_num(values) * weights).sum()),
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (values < 0.0))),
    }
    if has_support_role_metadata(person, entity="person"):
        role_values = support_role_series(person, entity="person").to_numpy()
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
                    (np.nan_to_num(values[channel_mask]) * weights[channel_mask]).sum()
                ),
            }
        summary["channels"] = channels
    return summary


def us_salt_refund_income_signal_gate(frame: Frame) -> GateResult:
    """Require finite, nonnegative, source-aligned SALT-refund signal."""

    person = frame.table("person")
    missing = [
        column for column in US_SALT_REFUND_OUTPUT_COLUMNS if column not in person
    ]
    if missing:
        return GateResult(
            name="salt_refund_income_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_salt_refund_income_summary(frame)
    failures: list[str] = []
    if summary["nonfinite"]:
        failures.append(f"{_OUTPUT} nonfinite values: {int(summary['nonfinite'])}.")
    if summary["negative"]:
        failures.append(f"{_OUTPUT} negative values: {int(summary['negative'])}.")
    share = float(summary["positive_share"])
    low, high = summary["positive_share_band"]
    if not (low <= share <= high):
        failures.append(
            f"SALT-refund positive share {share:.6f} outside plausibility "
            f"band [{low}, {high}]."
        )

    channels = summary.get("channels")
    if isinstance(channels, dict):
        asec = channels.get(_BASE_ASEC_SUPPORT_CHANNEL)
        puf = channels.get(_PUF_TAX_DETAIL_SUPPORT_CHANNEL)
        if asec is None:
            failures.append("SALT-refund signal gate is missing the ASEC channel.")
        elif float(asec["weighted_total"]) != 0.0:
            failures.append(
                "salt_refund_income must remain zero on the source-unobserved "
                "ASEC support channel."
            )
        if puf is None:
            failures.append(
                "SALT-refund signal gate is missing the PUF tax-detail channel."
            )
        else:
            puf_share = float(puf["positive_share"])
            puf_low, puf_high = _PUF_POSITIVE_SHARE_BAND
            if not (puf_low <= puf_share <= puf_high):
                failures.append(
                    f"PUF SALT-refund positive share {puf_share:.6f} outside "
                    f"plausibility band [{puf_low}, {puf_high}]."
                )

    return GateResult(
        name="salt_refund_income_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
