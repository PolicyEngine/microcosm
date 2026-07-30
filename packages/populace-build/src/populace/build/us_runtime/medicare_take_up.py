"""Measured CPS ASEC Medicare enrollment exported as the take-up input leaf.

The retired eCPS pipeline maps ``MCARE == 1`` to the formula-level
``medicare_enrolled`` value, duplicates non-PUF-imputed CPS variables onto the
PUF support half, and finally renames that measured value to the input leaf
``takes_up_medicare_if_eligible``.  This module restores the same source-backed
boolean directly, without applying a participation-rate draw.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from populace.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "MEDICARE_TAKE_UP_ARCHIVED_CLONE_URL",
    "MEDICARE_TAKE_UP_ARCHIVED_DERIVATION_URL",
    "MEDICARE_TAKE_UP_ARCHIVED_EXPORT_URL",
    "MEDICARE_TAKE_UP_ARCHIVED_SOURCE_COLUMNS_URL",
    "US_MEDICARE_TAKE_UP_NONCONSTANT_PERSON_COLUMNS",
    "US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS",
    "US_MEDICARE_TAKE_UP_REQUIRED_SOURCE_COLUMNS",
    "US_MEDICARE_TAKE_UP_STAGE_NAME",
    "derive_us_medicare_take_up_from_manifest",
    "us_medicare_take_up_signal_gate",
    "us_medicare_take_up_stage_spec",
    "us_medicare_take_up_summary",
    "with_us_medicare_take_up_input",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
MEDICARE_TAKE_UP_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/cps.py#L1579-L1585"
)
MEDICARE_TAKE_UP_ARCHIVED_SOURCE_COLUMNS_URL = (
    _ARCHIVED_ROOT + "datasets/cps/census_cps.py#L39-L58"
)
MEDICARE_TAKE_UP_ARCHIVED_CLONE_URL = (
    _ARCHIVED_ROOT + "calibration/puf_impute.py#L608-L629"
)
MEDICARE_TAKE_UP_ARCHIVED_EXPORT_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L1747-L1754"
)

US_MEDICARE_TAKE_UP_STAGE_NAME = "medicare_take_up_input"
US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS: tuple[str, ...] = ("takes_up_medicare_if_eligible",)
US_MEDICARE_TAKE_UP_NONCONSTANT_PERSON_COLUMNS = US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS
US_MEDICARE_TAKE_UP_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = ("MCARE",)

_OUTPUT = US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS[0]
_SOURCE = US_MEDICARE_TAKE_UP_REQUIRED_SOURCE_COLUMNS[0]
_PERSON_WEIGHT_COLUMN = "person_weight"
_VALID_SOURCE_CODES = frozenset({0, 1, 2})
_ENROLLED_CODE = 1
_EXPECTED_PARAMETERS = {
    "source": _SOURCE,
    "enrolled_code": _ENROLLED_CODE,
    "output": _OUTPUT,
}
_WEIGHTED_ENROLLED_SHARE_BAND = (0.15, 0.24)
_CHANNEL_ENROLLED_SHARE_BAND = (0.15, 0.24)


def us_medicare_take_up_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged Medicare take-up stage."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_MEDICARE_TAKE_UP_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_MEDICARE_TAKE_UP_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_MEDICARE_TAKE_UP_STAGE_NAME]
    if tuple(spec.outputs) != US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS:
        raise ValueError(
            f"{US_MEDICARE_TAKE_UP_STAGE_NAME!r} manifest outputs do not "
            "match the runtime-owned Medicare take-up family."
        )
    return spec


def _source_codes(person: pd.DataFrame, source: str) -> np.ndarray:
    if source not in person.columns:
        raise SourceRuntimeError(
            f"US Medicare take-up derivation requires ASEC source column {source!r}."
        )
    numeric = pd.to_numeric(person[source], errors="coerce").to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(numeric)
        & (numeric == np.floor(numeric))
        & np.isin(numeric, np.fromiter(_VALID_SOURCE_CODES, dtype=np.int64))
    )
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise SourceRuntimeError(
            f"US Medicare take-up source {source!r} must contain only integer "
            f"codes {sorted(_VALID_SOURCE_CODES)}; invalid row(s): {rows}."
        )
    return numeric.astype(np.int8)


def derive_us_medicare_take_up_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Map the exact retired ``MCARE == 1`` enrollment observation."""

    if operation.kind != "derive_medicare_take_up":
        raise SourceRuntimeError(
            "US Medicare take-up derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US Medicare take-up derivation requires the person table first."
        )
    parameters = dict(operation.parameters)
    if parameters != _EXPECTED_PARAMETERS:
        raise SourceRuntimeError(
            "US Medicare take-up derivation drifted from the archived method: "
            f"expected {_EXPECTED_PARAMETERS}, got {parameters}."
        )
    result = frame.copy(deep=True)
    result[_OUTPUT] = _source_codes(result, _SOURCE) == _ENROLLED_CODE
    return result


def _surface_matches_source(frame: Frame) -> bool:
    person = frame.table("person")
    if _OUTPUT not in person or person[_OUTPUT].dropna().nunique() <= 1:
        return False
    if _SOURCE not in person:
        return True
    try:
        expected = _source_codes(person, _SOURCE) == _ENROLLED_CODE
    except SourceRuntimeError:
        return False
    observed = person[_OUTPUT].fillna(False).astype(bool).to_numpy()
    return bool(np.array_equal(observed, expected))


def with_us_medicare_take_up_input(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
) -> Frame:
    """Materialize the measured Medicare take-up leaf on a US frame."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US Medicare take-up input requires the US schema.")
    if _surface_matches_source(frame):
        return frame

    person = frame.table("person")
    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_medicare_take_up_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_medicare_take_up": derive_us_medicare_take_up_from_manifest
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    if aligned[_OUTPUT].isna().any():
        raise ValueError(
            "US Medicare take-up stage output does not cover every person."
        )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][_OUTPUT] = aligned[_OUTPUT].to_numpy(dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_medicare_take_up_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and exact source-reconciliation diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    values = person[_OUTPUT].fillna(False).astype(bool).to_numpy()
    total_weight = float(weights.sum())
    summary: dict[str, object] = {
        "weighted_enrolled_share": (
            float(weights[values].sum()) / total_weight if total_weight > 0 else 0.0
        ),
        "weighted_enrolled_share_band": list(_WEIGHTED_ENROLLED_SHARE_BAND),
        "positive_count": int(np.count_nonzero(values)),
        "unique_count": int(person[_OUTPUT].dropna().nunique()),
        "missing_count": int(person[_OUTPUT].isna().sum()),
    }
    if _SOURCE in person:
        source_values = _source_codes(person, _SOURCE) == _ENROLLED_CODE
        summary["source_mismatch_count"] = int(
            np.count_nonzero(values != source_values)
        )
    if has_support_role_metadata(person, entity="person"):
        channel_shares: dict[str, float] = {}
        channels = support_role_series(person, entity="person").to_numpy()
        for channel in sorted(set(channels.tolist())):
            mask = channels == channel
            channel_weight = float(weights[mask].sum())
            channel_shares[str(channel)] = (
                float(weights[mask & values].sum()) / channel_weight
                if channel_weight > 0
                else 0.0
            )
        summary["channel_weighted_enrolled_shares"] = channel_shares
        summary["channel_weighted_enrolled_share_band"] = list(
            _CHANNEL_ENROLLED_SHARE_BAND
        )
    return summary


def us_medicare_take_up_signal_gate(frame: Frame) -> GateResult:
    """Require measured, nonconstant, source-consistent Medicare enrollment."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="medicare_take_up_input_signal",
            passed=False,
            failures=(f"{_OUTPUT}: missing",),
            details={},
        )
    try:
        summary = us_medicare_take_up_summary(frame)
    except SourceRuntimeError as exc:
        return GateResult(
            name="medicare_take_up_input_signal",
            passed=False,
            failures=(str(exc),),
            details={},
        )
    failures: list[str] = []
    if int(summary["missing_count"]):
        failures.append(f"{_OUTPUT}: missing values")
    if int(summary["unique_count"]) < 2:
        failures.append(f"{_OUTPUT}: constant")
    share = float(summary["weighted_enrolled_share"])
    low, high = _WEIGHTED_ENROLLED_SHARE_BAND
    if not low <= share <= high:
        failures.append(
            f"{_OUTPUT}: weighted enrolled share {share:.6f} outside "
            f"[{low:.3f}, {high:.3f}]"
        )
    mismatches = int(summary.get("source_mismatch_count", 0))
    if mismatches:
        failures.append(f"{_OUTPUT}: {mismatches} MCARE reconciliation mismatch(es)")
    channel_shares = summary.get("channel_weighted_enrolled_shares", {})
    channel_low, channel_high = _CHANNEL_ENROLLED_SHARE_BAND
    for channel, channel_share in dict(channel_shares).items():
        if not channel_low <= float(channel_share) <= channel_high:
            failures.append(
                f"{_OUTPUT}: {channel} weighted enrolled share "
                f"{float(channel_share):.6f} outside "
                f"[{channel_low:.3f}, {channel_high:.3f}]"
            )
    return GateResult(
        name="medicare_take_up_input_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
