"""Signed IRS PUF farm-operations and farm-rent income inputs.

The retired eCPS PUF pipeline mapped Schedule F net income ``E02100`` to
``farm_operations_income`` and farm-rent net income ``E27200`` to
``farm_rent_income``.  Both sources are signed: losses are factual inputs and
must never be clipped or converted to positive-only support.  CPS ASEC
``FRSE_VAL`` independently measures ``farm_operations_income`` on the ASEC
channel.  The PUF's separate ``farm_income`` field comes from ``T27800`` and is
not a substitute for either Section 199A income-definition input.

The processed, versioned PUF artifact already carries the two archived leaves.
Populace's shared weighted QRF replaces them only on the dedicated PUF support
channel, preserving measured ``FRSE_VAL`` operations income and the default-zero
rent leaf on ASEC.  The qualification flags are a separate QBI family and do
not create or suppress either signed amount here.  This is deliberate Populace
hardening: the retired override replaced both extended-CPS halves, whereas the
source-preserving support design never overwrites measured ASEC observations.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.frame import Frame

__all__ = [
    "FARM_BUSINESS_INCOME_ARCHIVED_CPS_FARM_INCOME_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_DERIVATION_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_EXPORT_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_IMPUTATION_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_OVERRIDE_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_PUF_ARTIFACT_URL",
    "US_FARM_BUSINESS_INCOME_NONCONSTANT_PERSON_COLUMNS",
    "US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS",
    "US_FARM_BUSINESS_INCOME_STAGE_NAME",
    "derive_us_farm_business_income_from_puf",
    "us_farm_business_income_signal_gate",
    "us_farm_business_income_stage_spec",
    "us_farm_business_income_summary",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
FARM_BUSINESS_INCOME_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L636-L704"
)
FARM_BUSINESS_INCOME_ARCHIVED_CPS_FARM_INCOME_URL = (
    _ARCHIVED_ROOT + "datasets/cps/cps.py#L1363-L1382"
)
FARM_BUSINESS_INCOME_ARCHIVED_EXPORT_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L804-L875"
)
FARM_BUSINESS_INCOME_ARCHIVED_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "calibration/puf_impute.py#L80-L198"
)
FARM_BUSINESS_INCOME_ARCHIVED_OVERRIDE_URL = (
    _ARCHIVED_ROOT + "calibration/puf_impute.py#L513-L672"
)
FARM_BUSINESS_INCOME_ARCHIVED_PUF_ARTIFACT_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L1655-L1660"
)

US_FARM_BUSINESS_INCOME_STAGE_NAME = "puf_tax_detail"
US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS: tuple[str, ...] = (
    "farm_operations_income",
    "farm_rent_income",
)
US_FARM_BUSINESS_INCOME_NONCONSTANT_PERSON_COLUMNS = (
    US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS
)

_OPERATIONS_OUTPUT, _RENT_OUTPUT = US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS
_SOURCE_COLUMNS = ("E02100", "E27200")
_PERSON_SUPPORT_CHANNEL_COLUMN = "person_support_channel"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
_NONZERO_SHARE_BANDS: dict[str, tuple[float, float]] = {
    _OPERATIONS_OUTPUT: (0.0001, 0.20),
    _RENT_OUTPUT: (0.00001, 0.20),
}


def us_farm_business_income_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail output declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_FARM_BUSINESS_INCOME_STAGE_NAME]
    missing = sorted(set(US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_FARM_BUSINESS_INCOME_STAGE_NAME!r} manifest stage does not "
            f"declare farm-business output(s) {missing}."
        )
    forbidden_nonnegative = sorted(
        set(US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS) & set(spec.nonnegative_outputs)
    )
    if forbidden_nonnegative:
        raise ValueError(
            "Signed farm-business inputs must not be declared nonnegative: "
            f"{forbidden_nonnegative}."
        )
    return spec


def _strict_signed_values(puf: pd.DataFrame, column: str) -> np.ndarray:
    if column not in puf.columns:
        raise ValueError(
            f"PUF farm-business derivation requires source column {column!r}."
        )
    numeric = pd.to_numeric(puf[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(numeric)))
    if nonfinite:
        raise ValueError(
            f"PUF farm-business source column {column!r} contains {nonfinite} "
            "nonnumeric or nonfinite value(s)."
        )
    return numeric


def derive_us_farm_business_income_from_puf(
    puf: pd.DataFrame,
    *,
    operations_source_column: str = _SOURCE_COLUMNS[0],
    operations_output_column: str = _OPERATIONS_OUTPUT,
    rent_source_column: str = _SOURCE_COLUMNS[1],
    rent_output_column: str = _RENT_OUTPUT,
) -> pd.DataFrame:
    """Carry both archived signed PUF fields without clipping or mutation."""

    operations = _strict_signed_values(puf, operations_source_column)
    rent = _strict_signed_values(puf, rent_source_column)
    result = puf.copy(deep=True)
    result[operations_output_column] = operations
    result[rent_output_column] = rent
    return result


def us_farm_business_income_summary(frame: Frame) -> dict[str, object]:
    """Return signed, weighted, and support-channel signal diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    channel = (
        person[_PERSON_SUPPORT_CHANNEL_COLUMN].astype(str).to_numpy()
        if _PERSON_SUPPORT_CHANNEL_COLUMN in person
        else None
    )
    columns: dict[str, dict[str, object]] = {}
    for output in US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS:
        values = pd.to_numeric(person[output], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(values)
        nonzero = finite & (values != 0.0)
        positive = finite & (values > 0.0)
        negative = finite & (values < 0.0)
        detail: dict[str, object] = {
            "nonzero_share": (
                float(weights[nonzero].sum()) / total_weight
                if total_weight > 0.0
                else 0.0
            ),
            "nonzero_share_band": list(_NONZERO_SHARE_BANDS[output]),
            "positive_share": (
                float(weights[positive].sum()) / total_weight
                if total_weight > 0.0
                else 0.0
            ),
            "negative_share": (
                float(weights[negative].sum()) / total_weight
                if total_weight > 0.0
                else 0.0
            ),
            "weighted_total": float((np.nan_to_num(values) * weights).sum()),
            "weighted_absolute_total": float(
                (np.abs(np.nan_to_num(values)) * weights).sum()
            ),
            "nonfinite": int(np.count_nonzero(~finite)),
        }
        if channel is not None:
            channels: dict[str, dict[str, float | int]] = {}
            for name in (
                _BASE_ASEC_SUPPORT_CHANNEL,
                _PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            ):
                mask = channel == name
                channel_weight = float(weights[mask].sum())
                channels[name] = {
                    "nonzero_rows": int(np.count_nonzero(mask & nonzero)),
                    "nonzero_share": (
                        float(weights[mask & nonzero].sum()) / channel_weight
                        if channel_weight > 0.0
                        else 0.0
                    ),
                    "positive_rows": int(np.count_nonzero(mask & positive)),
                    "negative_rows": int(np.count_nonzero(mask & negative)),
                    "weighted_total": float(
                        (np.nan_to_num(values[mask]) * weights[mask]).sum()
                    ),
                    "weighted_absolute_total": float(
                        (np.abs(np.nan_to_num(values[mask])) * weights[mask]).sum()
                    ),
                }
            detail["channels"] = channels
        columns[output] = detail
    return {"columns": columns}


def us_farm_business_income_signal_gate(frame: Frame) -> GateResult:
    """Require finite signed signal on each source-backed support channel."""

    person = frame.table("person")
    missing = [
        output
        for output in US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS
        if output not in person.columns
    ]
    if missing:
        return GateResult(
            name="farm_business_income_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_farm_business_income_summary(frame)
    failures: list[str] = []
    columns = summary["columns"]
    for output in US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS:
        detail = columns[output]
        if detail["nonfinite"]:
            failures.append(f"{output}: {int(detail['nonfinite'])} nonfinite value(s).")
        share = float(detail["nonzero_share"])
        low, high = detail["nonzero_share_band"]
        if not (low <= share <= high):
            failures.append(
                f"{output}: nonzero share {share:.6f} outside plausibility "
                f"band [{low}, {high}]."
            )
        if float(detail["weighted_absolute_total"]) <= 0.0:
            failures.append(f"{output}: weighted absolute total is zero.")
        channels = detail.get("channels")
        if isinstance(channels, dict):
            asec = channels.get(_BASE_ASEC_SUPPORT_CHANNEL)
            puf = channels.get(_PUF_TAX_DETAIL_SUPPORT_CHANNEL)
            asec_nonzero = (
                int(asec.get("nonzero_rows", 0)) if isinstance(asec, dict) else 0
            )
            if output == _OPERATIONS_OUTPUT and not asec_nonzero:
                failures.append(
                    f"{output}: ASEC support has no measured FRSE_VAL signal."
                )
            if output == _OPERATIONS_OUTPUT and isinstance(asec, dict):
                if not int(asec.get("positive_rows", 0)):
                    failures.append(f"{output}: ASEC support has no positive values.")
                if not int(asec.get("negative_rows", 0)):
                    failures.append(f"{output}: ASEC support has no farm losses.")
            if output == _RENT_OUTPUT and asec_nonzero:
                failures.append(
                    f"{output}: ASEC support carries nonzero values despite no "
                    "farm-rent source."
                )
            if not isinstance(puf, dict) or not int(puf.get("nonzero_rows", 0)):
                failures.append(f"{output}: PUF tax-detail support has no signal.")
            elif not int(puf.get("positive_rows", 0)):
                failures.append(
                    f"{output}: PUF tax-detail support has no positive values."
                )
            elif not int(puf.get("negative_rows", 0)):
                failures.append(f"{output}: PUF tax-detail support has no farm losses.")
    return GateResult(
        name="farm_business_income_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
