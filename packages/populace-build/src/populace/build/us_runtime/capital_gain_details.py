"""IRS PUF collectibles and unrecaptured-section-1250 gain inputs.

The retired eCPS pipeline mapped ``E24518`` directly to
``long_term_capital_gains_on_collectibles`` and ``E24515`` directly to
``unrecaptured_section_1250_gain``. The processed PUF carries the first leaf at
person grain and the second at tax-unit grain. Populace reduces the person leaf
to its identifiable tax-unit total for the shared PUF fit, then uses
first-person placement on the PUF support channel because PolicyEngine-US sums
it back to the tax unit. This avoids inventing the retired randomized
filer/spouse earnings split.

Both leaves are primary-source amounts. PolicyEngine-US owns their federal and
Massachusetts tax formulas; this module only carries and validates the inputs.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.frame import Frame

__all__ = [
    "CAPITAL_GAIN_DETAILS_ARCHIVED_DERIVATION_URL",
    "CAPITAL_GAIN_DETAILS_ARCHIVED_EXPORT_URL",
    "CAPITAL_GAIN_DETAILS_ARCHIVED_IMPUTATION_URL",
    "CAPITAL_GAIN_DETAILS_ARCHIVED_PERSON_ALLOCATION_URL",
    "CAPITAL_GAIN_DETAILS_ARCHIVED_PUF_ARTIFACT_URL",
    "US_CAPITAL_GAIN_DETAILS_NONCONSTANT_PERSON_COLUMNS",
    "US_CAPITAL_GAIN_DETAILS_NONCONSTANT_TAX_UNIT_COLUMNS",
    "US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS",
    "US_CAPITAL_GAIN_DETAILS_STAGE_NAME",
    "derive_us_capital_gain_details_from_puf",
    "us_capital_gain_details_signal_gate",
    "us_capital_gain_details_stage_spec",
    "us_capital_gain_details_summary",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
CAPITAL_GAIN_DETAILS_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L636-L702"
)
CAPITAL_GAIN_DETAILS_ARCHIVED_EXPORT_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L804-L850"
)
CAPITAL_GAIN_DETAILS_ARCHIVED_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "calibration/puf_impute.py#L90-L198"
)
CAPITAL_GAIN_DETAILS_ARCHIVED_PERSON_ALLOCATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L1513-L1601"
)
CAPITAL_GAIN_DETAILS_ARCHIVED_PUF_ARTIFACT_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L1655-L1660"
)

US_CAPITAL_GAIN_DETAILS_STAGE_NAME = "puf_tax_detail"
US_CAPITAL_GAIN_DETAILS_NONCONSTANT_PERSON_COLUMNS: tuple[str, ...] = (
    "long_term_capital_gains_on_collectibles",
)
US_CAPITAL_GAIN_DETAILS_NONCONSTANT_TAX_UNIT_COLUMNS: tuple[str, ...] = (
    "unrecaptured_section_1250_gain",
)
US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS = (
    *US_CAPITAL_GAIN_DETAILS_NONCONSTANT_PERSON_COLUMNS,
    *US_CAPITAL_GAIN_DETAILS_NONCONSTANT_TAX_UNIT_COLUMNS,
)

_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
_SIGNAL_BANDS = {
    "long_term_capital_gains_on_collectibles": {
        "overall": (0.00001, 0.005),
        "puf": (0.00002, 0.01),
    },
    "unrecaptured_section_1250_gain": {
        "overall": (0.0005, 0.03),
        "puf": (0.001, 0.06),
    },
}


def us_capital_gain_details_stage_spec() -> SourceStageSpec:
    """Load and validate the shared PUF tax-detail stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[US_CAPITAL_GAIN_DETAILS_STAGE_NAME]
    missing = sorted(set(US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_CAPITAL_GAIN_DETAILS_STAGE_NAME!r} manifest stage does not "
            f"declare capital-gain detail output(s) {missing}."
        )
    return spec


def derive_us_capital_gain_details_from_puf(
    puf: pd.DataFrame,
    *,
    collectibles_source_column: str = "E24518",
    collectibles_output_column: str = "long_term_capital_gains_on_collectibles",
    unrecaptured_source_column: str = "E24515",
    unrecaptured_output_column: str = "unrecaptured_section_1250_gain",
) -> pd.DataFrame:
    """Carry the archived E24518 and E24515 fields without alteration."""

    sources = {
        collectibles_output_column: collectibles_source_column,
        unrecaptured_output_column: unrecaptured_source_column,
    }
    missing = sorted(set(sources.values()) - set(puf.columns))
    if missing:
        raise ValueError(
            f"PUF capital-gain detail derivation requires source columns {missing}."
        )

    result = puf.copy(deep=True)
    for output, source in sources.items():
        values = pd.to_numeric(puf[source], errors="coerce").to_numpy(dtype=np.float64)
        nonfinite = ~np.isfinite(values)
        if bool(nonfinite.any()):
            raise ValueError(
                f"PUF capital-gain source column {source!r} contains "
                f"{int(np.count_nonzero(nonfinite))} nonnumeric or nonfinite "
                "value(s)."
            )
        negative = values < 0.0
        if bool(negative.any()):
            raise ValueError(
                f"PUF capital-gain source column {source!r} contains "
                f"{int(np.count_nonzero(negative))} negative value(s)."
            )
        result[output] = values
    return result


def _column_summary(
    frame: Frame,
    *,
    entity: str,
    output: str,
) -> dict[str, object]:
    table = frame.table(entity)
    values = pd.to_numeric(table[output], errors="coerce").to_numpy(dtype=np.float64)
    weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    total_weight = float(weights.sum())
    summary: dict[str, object] = {
        "positive_share": (
            float(weights[positive].sum()) / total_weight if total_weight > 0.0 else 0.0
        ),
        "positive_share_band": list(_SIGNAL_BANDS[output]["overall"]),
        "weighted_total": float((np.nan_to_num(values) * weights).sum()),
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (values < 0.0))),
    }
    support_channel = f"{entity}_support_channel"
    if support_channel in table.columns:
        channel_values = table[support_channel].to_numpy()
        channels: dict[str, dict[str, float | int]] = {}
        for channel in table[support_channel].dropna().unique():
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


def us_capital_gain_details_summary(frame: Frame) -> dict[str, dict[str, object]]:
    """Return weighted signal, support-channel, and validity diagnostics."""

    return {
        "long_term_capital_gains_on_collectibles": _column_summary(
            frame,
            entity="person",
            output="long_term_capital_gains_on_collectibles",
        ),
        "unrecaptured_section_1250_gain": _column_summary(
            frame,
            entity="tax_unit",
            output="unrecaptured_section_1250_gain",
        ),
    }


def us_capital_gain_details_signal_gate(frame: Frame) -> GateResult:
    """Require finite, nonnegative, source-aligned signal in both PUF leaves."""

    entity_outputs = {
        "person": US_CAPITAL_GAIN_DETAILS_NONCONSTANT_PERSON_COLUMNS,
        "tax_unit": US_CAPITAL_GAIN_DETAILS_NONCONSTANT_TAX_UNIT_COLUMNS,
    }
    missing = {
        entity: sorted(set(outputs) - set(frame.table(entity).columns))
        for entity, outputs in entity_outputs.items()
    }
    missing = {entity: outputs for entity, outputs in missing.items() if outputs}
    if missing:
        return GateResult(
            name="capital_gain_details_signal",
            passed=False,
            failures=(f"entity columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_capital_gain_details_summary(frame)
    failures: list[str] = []
    for output, details in summary.items():
        if details["nonfinite"]:
            failures.append(f"{output} nonfinite values: {int(details['nonfinite'])}.")
        if details["negative"]:
            failures.append(f"{output} negative values: {int(details['negative'])}.")
        share = float(details["positive_share"])
        low, high = details["positive_share_band"]
        if not (low <= share <= high):
            failures.append(
                f"{output} positive share {share:.6f} outside plausibility "
                f"band [{low}, {high}]."
            )
        channels = details.get("channels")
        if not isinstance(channels, dict):
            continue
        asec = channels.get(_BASE_ASEC_SUPPORT_CHANNEL)
        puf = channels.get(_PUF_TAX_DETAIL_SUPPORT_CHANNEL)
        if asec is None:
            failures.append(f"{output} is missing the ASEC support channel.")
        elif float(asec["weighted_total"]) != 0.0:
            failures.append(
                f"{output} must remain zero on the source-unobserved ASEC channel."
            )
        if puf is None:
            failures.append(f"{output} is missing the PUF tax-detail channel.")
        else:
            puf_share = float(puf["positive_share"])
            puf_low, puf_high = _SIGNAL_BANDS[output]["puf"]
            if not (puf_low <= puf_share <= puf_high):
                failures.append(
                    f"PUF {output} positive share {puf_share:.6f} outside "
                    f"plausibility band [{puf_low}, {puf_high}]."
                )

    return GateResult(
        name="capital_gain_details_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
