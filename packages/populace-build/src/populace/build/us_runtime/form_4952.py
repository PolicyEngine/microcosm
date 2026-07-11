"""IRS PUF Form 4952 elected-investment-income input for the US build.

The retired eCPS pipeline carried
``investment_income_elected_form_4952`` directly from IRS PUF field
``E58990``. Immutable archived coordinates for that derivation, the exported
field, and its weighted PUF imputation are exposed below.

There is no CPS analogue and no synthetic fallback. The PUF tax-detail stage
carries the source exactly, then its shared weighted QRF places the input only
on the dedicated PUF support channel and sparsifies it to the donor's weighted
positive rate. PolicyEngine-US aggregates the person input to the tax unit and
subtracts it from net capital gain; this module persists only the factual
election amount.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.frame import Frame

__all__ = [
    "FORM_4952_ARCHIVED_DERIVATION_URL",
    "FORM_4952_ARCHIVED_EXPORT_URL",
    "FORM_4952_ARCHIVED_IMPUTATION_URL",
    "FORM_4952_ARCHIVED_PERSON_ALLOCATION_URL",
    "FORM_4952_ARCHIVED_PUF_ARTIFACT_URL",
    "US_FORM_4952_NONCONSTANT_PERSON_COLUMNS",
    "US_FORM_4952_OUTPUT_COLUMNS",
    "US_FORM_4952_STAGE_NAME",
    "derive_us_form_4952_election_from_puf",
    "us_form_4952_election_signal_gate",
    "us_form_4952_election_stage_spec",
    "us_form_4952_election_summary",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
FORM_4952_ARCHIVED_DERIVATION_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L708"
FORM_4952_ARCHIVED_EXPORT_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L804-L850"
FORM_4952_ARCHIVED_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "calibration/puf_impute.py#L90-L198"
)
FORM_4952_ARCHIVED_PERSON_ALLOCATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L477-L546"
)
FORM_4952_ARCHIVED_PUF_ARTIFACT_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L1655-L1660"

US_FORM_4952_STAGE_NAME = "puf_tax_detail"
US_FORM_4952_OUTPUT_COLUMNS: tuple[str, ...] = ("investment_income_elected_form_4952",)
US_FORM_4952_NONCONSTANT_PERSON_COLUMNS = US_FORM_4952_OUTPUT_COLUMNS

_OUTPUT = US_FORM_4952_OUTPUT_COLUMNS[0]
_PERSON_SUPPORT_CHANNEL_COLUMN = "person_support_channel"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
_OVERALL_POSITIVE_SHARE_BAND = (0.00005, 0.02)
_PUF_POSITIVE_SHARE_BAND = (0.0001, 0.04)


def us_form_4952_election_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_FORM_4952_STAGE_NAME]
    missing = sorted(set(US_FORM_4952_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_FORM_4952_STAGE_NAME!r} manifest stage does not declare "
            f"Form 4952 output(s) {missing}."
        )
    return spec


def derive_us_form_4952_election_from_puf(
    puf: pd.DataFrame,
    *,
    source_column: str = "E58990",
    output_column: str = _OUTPUT,
) -> pd.DataFrame:
    """Carry the archived IRS PUF E58990 field without alteration."""

    if source_column not in puf.columns:
        raise ValueError(
            f"PUF Form 4952 derivation requires source column {source_column!r}."
        )
    values = pd.to_numeric(puf[source_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    nonfinite = ~np.isfinite(values)
    if bool(nonfinite.any()):
        raise ValueError(
            f"PUF Form 4952 source column {source_column!r} contains "
            f"{int(np.count_nonzero(nonfinite))} nonnumeric or nonfinite value(s)."
        )
    negative = values < 0.0
    if bool(negative.any()):
        raise ValueError(
            f"PUF Form 4952 source column {source_column!r} contains "
            f"{int(np.count_nonzero(negative))} negative value(s)."
        )

    result = puf.copy(deep=True)
    result[output_column] = values
    return result


def us_form_4952_election_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal, channel, and validity diagnostics."""

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
    if _PERSON_SUPPORT_CHANNEL_COLUMN in person.columns:
        channels: dict[str, dict[str, float | int]] = {}
        channel_values = person[_PERSON_SUPPORT_CHANNEL_COLUMN].to_numpy()
        for channel in person[_PERSON_SUPPORT_CHANNEL_COLUMN].dropna().unique():
            channel_mask = channel_values == channel
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


def us_form_4952_election_signal_gate(frame: Frame) -> GateResult:
    """Require a finite, sparse, source-aligned Form 4952 input."""

    person = frame.table("person")
    missing = [
        column for column in US_FORM_4952_OUTPUT_COLUMNS if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="form_4952_election_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_form_4952_election_summary(frame)
    failures: list[str] = []
    if summary["nonfinite"]:
        failures.append(f"{_OUTPUT} nonfinite values: {int(summary['nonfinite'])}.")
    if summary["negative"]:
        failures.append(f"{_OUTPUT} negative values: {int(summary['negative'])}.")
    share = float(summary["positive_share"])
    low, high = summary["positive_share_band"]
    if not (low <= share <= high):
        failures.append(
            f"Form 4952 positive share {share:.6f} outside plausibility "
            f"band [{low}, {high}]."
        )

    channels = summary.get("channels")
    if isinstance(channels, dict):
        asec = channels.get(_BASE_ASEC_SUPPORT_CHANNEL)
        puf = channels.get(_PUF_TAX_DETAIL_SUPPORT_CHANNEL)
        if asec is None:
            failures.append(
                "Form 4952 signal gate is missing the ASEC support channel."
            )
        elif float(asec["weighted_total"]) != 0.0:
            failures.append(
                "Form 4952 input must remain zero on the source-unobserved "
                "ASEC support channel."
            )
        if puf is None:
            failures.append(
                "Form 4952 signal gate is missing the PUF-tax-detail support channel."
            )
        else:
            puf_share = float(puf["positive_share"])
            puf_low, puf_high = _PUF_POSITIVE_SHARE_BAND
            if not (puf_low <= puf_share <= puf_high):
                failures.append(
                    f"PUF Form 4952 positive share {puf_share:.6f} outside "
                    f"plausibility band [{puf_low}, {puf_high}]."
                )

    return GateResult(
        name="form_4952_election_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
