"""IRS PUF educator-expense input for the US build.

The retired eCPS pipeline carried ``educator_expense`` directly from IRS PUF
field ``E03220`` and allocated the tax-unit amount between filer and spouse by
earnings share.  Immutable archived coordinates for the derivation, allocation,
export, and PUF-QRF treatment are exposed below.

There is no CPS analogue and no synthetic fallback.  Microcosm's established
PUF-detail topology therefore carries the processed, source-backed leaf into
the shared weighted QRF, places it only on the dedicated PUF support channel,
and sparsifies it to the donor's weighted positive rate.  This differs openly
from the retired pipeline's both-half override, while preserving population
mass and the build's source-channel contract.  PolicyEngine-US owns the
above-the-line deduction formula; this module persists only the factual input.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    has_support_role_metadata,
    support_role_series,
)
from microcosm.frame import Frame

__all__ = [
    "EDUCATOR_EXPENSE_ARCHIVED_ALLOCATION_URL",
    "EDUCATOR_EXPENSE_ARCHIVED_DERIVATION_URL",
    "EDUCATOR_EXPENSE_ARCHIVED_EXPORT_URL",
    "EDUCATOR_EXPENSE_ARCHIVED_PUF_IMPUTATION_URL",
    "US_EDUCATOR_EXPENSE_NONCONSTANT_PERSON_COLUMNS",
    "US_EDUCATOR_EXPENSE_OUTPUT_COLUMNS",
    "US_EDUCATOR_EXPENSE_STAGE_NAME",
    "derive_us_educator_expense_from_puf",
    "us_educator_expense_signal_gate",
    "us_educator_expense_stage_spec",
    "us_educator_expense_summary",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
EDUCATOR_EXPENSE_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L636-L649"
)
EDUCATOR_EXPENSE_ARCHIVED_ALLOCATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L617-L620"
)
EDUCATOR_EXPENSE_ARCHIVED_EXPORT_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L804-L815"
EDUCATOR_EXPENSE_ARCHIVED_PUF_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "calibration/puf_impute.py#L940-L1075"
)

US_EDUCATOR_EXPENSE_STAGE_NAME = "puf_tax_detail"
US_EDUCATOR_EXPENSE_OUTPUT_COLUMNS: tuple[str, ...] = ("educator_expense",)
US_EDUCATOR_EXPENSE_NONCONSTANT_PERSON_COLUMNS = US_EDUCATOR_EXPENSE_OUTPUT_COLUMNS

_OUTPUT = US_EDUCATOR_EXPENSE_OUTPUT_COLUMNS[0]
_OVERALL_POSITIVE_SHARE_BAND = (0.002, 0.04)
_PUF_POSITIVE_SHARE_BAND = (0.005, 0.08)


def us_educator_expense_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_EDUCATOR_EXPENSE_STAGE_NAME]
    missing = sorted(set(US_EDUCATOR_EXPENSE_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_EDUCATOR_EXPENSE_STAGE_NAME!r} manifest stage does not declare "
            f"educator-expense output(s) {missing}."
        )
    return spec


def derive_us_educator_expense_from_puf(
    puf: pd.DataFrame,
    *,
    source_column: str = "E03220",
    output_column: str = _OUTPUT,
) -> pd.DataFrame:
    """Carry the archived IRS PUF educator-expense field exactly."""

    if source_column not in puf.columns:
        raise ValueError(
            f"PUF educator-expense derivation requires source column {source_column!r}."
        )
    values = pd.to_numeric(puf[source_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    nonfinite = ~np.isfinite(values)
    if bool(nonfinite.any()):
        raise ValueError(
            f"PUF educator-expense source column {source_column!r} contains "
            f"{int(np.count_nonzero(nonfinite))} nonnumeric or nonfinite value(s)."
        )
    negative = values < 0.0
    if bool(negative.any()):
        raise ValueError(
            f"PUF educator-expense source column {source_column!r} contains "
            f"{int(np.count_nonzero(negative))} negative value(s)."
        )

    result = puf.copy(deep=True)
    result[output_column] = values
    return result


def us_educator_expense_summary(frame: Frame) -> dict[str, object]:
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
    if has_support_role_metadata(person, entity="person"):
        channel = support_role_series(person, entity="person").to_numpy()
        channels: dict[str, dict[str, float | int]] = {}
        for name in (
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        ):
            mask = channel == name
            channel_weight = float(weights[mask].sum())
            channels[name] = {
                "positive_rows": int(np.count_nonzero(mask & positive)),
                "positive_share": (
                    float(weights[mask & positive].sum()) / channel_weight
                    if channel_weight > 0.0
                    else 0.0
                ),
                "weighted_total": float(
                    (np.nan_to_num(values[mask]) * weights[mask]).sum()
                ),
            }
        summary["channels"] = channels
    return summary


def us_educator_expense_signal_gate(frame: Frame) -> GateResult:
    """Require source-backed, sparse PUF signal and a zero ASEC channel."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="educator_expense_signal",
            passed=False,
            failures=(f"person column missing: {_OUTPUT}.",),
            details={"missing": [_OUTPUT]},
        )

    summary = us_educator_expense_summary(frame)
    failures: list[str] = []
    if summary["nonfinite"]:
        failures.append(f"{_OUTPUT}: {int(summary['nonfinite'])} nonfinite values.")
    if summary["negative"]:
        failures.append(f"{_OUTPUT}: {int(summary['negative'])} negative values.")
    share = float(summary["positive_share"])
    low, high = summary["positive_share_band"]
    if not (low <= share <= high):
        failures.append(
            f"{_OUTPUT}: positive share {share:.4f} outside plausibility band "
            f"[{low}, {high}]."
        )

    channels = summary.get("channels")
    if isinstance(channels, dict):
        asec = channels.get(BASE_ASEC_SUPPORT_CHANNEL)
        puf = channels.get(PUF_TAX_DETAIL_SUPPORT_CHANNEL)
        if not isinstance(asec, dict) or not isinstance(puf, dict):
            failures.append(f"{_OUTPUT}: missing support-channel diagnostics.")
        else:
            asec_share = float(asec["positive_share"])
            if asec_share != 0.0:
                failures.append(
                    f"{_OUTPUT}: ASEC support channel must remain source-zero; "
                    f"positive share is {asec_share:.4f}."
                )
            puf_share = float(puf["positive_share"])
            puf_low, puf_high = _PUF_POSITIVE_SHARE_BAND
            if not (puf_low <= puf_share <= puf_high):
                failures.append(
                    f"{_OUTPUT}: PUF support-channel positive share "
                    f"{puf_share:.4f} outside plausibility band "
                    f"[{puf_low}, {puf_high}]."
                )
    return GateResult(
        name="educator_expense_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
