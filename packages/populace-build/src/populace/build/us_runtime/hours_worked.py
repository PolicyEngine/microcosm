"""Hours-worked inputs from CPS ASEC reported hours.

Without this stage the published dataset stores no
``weekly_hours_worked_before_lsr`` (or stores it constant at the engine's
40-hour default), so PolicyEngine-US treats every person as working 40
hours per week and every hours-conditioned rule becomes a no-op — the
failure mode of populace issues #242/#248 (SNAP's 30-hour general and
20-hour ABAWD work requirements pass for everyone, erasing work-requirement
reform exposure entirely).

The stage writes two PolicyEngine-facing person input columns, each a
direct mapping of a measured CPS ASEC person variable — the same mappings
the retired enhanced-CPS pipeline used, so parity with the retired
enhanced-CPS hours surface is by construction rather than by imputation:

- ``weekly_hours_worked_before_lsr`` ← ``HRSWK`` (usual hours worked per
  week last year; zero for persons who did not work).
- ``hours_worked_last_week`` ← ``A_HRS1`` (hours worked at all jobs in the
  reference week).

``WKSWORK`` remains raw ASEC source evidence, but this stage deliberately does
not publish it as ``weeks_worked``. PolicyEngine-US 1.764.6 owns that canonical
name through a formula beginning in 2025, so Populace's period-agnostic input
leaf contract rejects it even for a 2024 pool.

Semantics note (documented, not hidden): SNAP's ABAWD standard is a monthly
80-hour test. ``HRSWK`` is an annual usual-hours measure, so a person who
worked half the year at 40 hours/week carries 40 here — the engine's
work-requirement variables read usual weekly hours, matching how the
retired enhanced-CPS pipeline fed them. Week-by-week volatility is not in
the CPS and is out of scope for this stage.

Healing behavior: a frame that already carries both input leaves *with signal*
passes through untouched (idempotent). A frame carrying a constant
``weekly_hours_worked_before_lsr`` — the published constant-40 landmine —
is recomputed from the raw columns rather than trusted, because a constant
hours column is indistinguishable from the engine default and masks the
missing derivation this stage exists to fix. A stale canonical
``weeks_worked`` column is removed on every path.
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
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "US_HOURS_WORKED_NONCONSTANT_PERSON_COLUMNS",
    "US_HOURS_WORKED_OUTPUT_COLUMNS",
    "US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS",
    "US_HOURS_WORKED_STAGE_NAME",
    "derive_us_hours_worked_from_manifest",
    "us_hours_worked_signal_gate",
    "us_hours_worked_summary",
    "us_hours_worked_stage_spec",
    "with_us_hours_worked_inputs",
]

US_HOURS_WORKED_STAGE_NAME = "hours_worked"

#: The PolicyEngine-facing person input columns this stage owns.
US_HOURS_WORKED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "weekly_hours_worked_before_lsr",
    "hours_worked_last_week",
)

_FORMULA_OWNED_HOURS_OUTPUTS = frozenset({"weeks_worked"})

#: Release gates require these person columns to carry signal (≥2 values).
US_HOURS_WORKED_NONCONSTANT_PERSON_COLUMNS: tuple[str, ...] = (
    US_HOURS_WORKED_OUTPUT_COLUMNS
)

#: Raw CPS ASEC person columns the derivation reads.
US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "HRSWK",
    "A_HRS1",
)

#: Weighted share of persons with positive usual weekly hours must land in
#: this band. CPS ASEC 2024: ~50% of all persons (children included) worked
#: last year; the retired enhanced-CPS surface measured 50.4%. A share
#: outside the band means the derivation read the wrong column or the raw
#: hours collapsed.
_WORKED_SHARE_BAND = (0.35, 0.62)

#: Weighted mean usual weekly hours among persons with positive hours must
#: land in this band (BLS/CPS average weekly hours for workers ≈ 38.5).
_MEAN_WEEKLY_HOURS_BAND = (30.0, 45.0)

_PERSON_WEIGHT_COLUMN = "person_weight"

_DERIVE_HOURS_WORKED_PARAMETER_KEYS = frozenset()


def us_hours_worked_stage_spec() -> SourceStageSpec:
    """Load the packaged ``hours_worked`` source-stage manifest entry."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_HOURS_WORKED_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_HOURS_WORKED_STAGE_NAME!r} stage."
        )
    return stage_map[US_HOURS_WORKED_STAGE_NAME]


def derive_us_hours_worked_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Map raw ASEC hours/weeks columns onto the PolicyEngine input names.

    The current frame must be the raw-column person table (from the stage's
    ``read_table`` operation).
    """

    if operation.kind != "derive_hours_worked":
        raise SourceRuntimeError(
            f"US hours derivation received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US hours derivation requires the person table to be read first."
        )
    unexpected = sorted(set(operation.parameters) - _DERIVE_HOURS_WORKED_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            f"US hours derivation received unsupported parameter(s): {unexpected}."
        )
    missing = [
        column
        for column in US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            f"US hours derivation requires raw ASEC column(s): {missing}."
        )

    def _nonnegative(column: str) -> np.ndarray:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        return np.maximum(values.to_numpy(dtype=np.float64), 0.0)

    result = frame.copy(deep=True)
    result["weekly_hours_worked_before_lsr"] = _nonnegative("HRSWK")
    result["hours_worked_last_week"] = _nonnegative("A_HRS1")
    return result


def _weekly_hours_carry_signal(person: pd.DataFrame) -> bool:
    """Whether the persisted weekly-hours column is trustworthy as-is.

    A constant column (one observed value) is indistinguishable from the
    engine's broadcast default and must be recomputed, not passed through.
    """

    values = pd.to_numeric(
        person["weekly_hours_worked_before_lsr"], errors="coerce"
    ).dropna()
    return values.nunique() > 1


def _without_formula_owned_hours_outputs(frame: Frame) -> Frame:
    """Remove stale engine-owned outputs while retaining raw ASEC evidence."""

    person = frame.table("person")
    stale = sorted(_FORMULA_OWNED_HOURS_OUTPUTS.intersection(person.columns))
    if not stale:
        return frame
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=stale)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def with_us_hours_worked_inputs(frame: Frame, *, seed: int, time_period: int) -> Frame:
    """Run the ``hours_worked`` manifest stage over a US frame.

    A frame already carrying both output columns with a non-constant
    weekly-hours distribution passes through untouched (idempotent). Any
    other surface — columns missing, or weekly hours constant at the
    engine default — is recomputed from the raw ASEC columns. A stale
    formula-owned ``weeks_worked`` column is removed before either path.

    Args:
        frame: A US-schema frame whose person table still carries the raw
            CPS ASEC source columns (unless the outputs already carry
            signal).
        seed: Build-wide imputation seed (recorded in the stage run; the
            derivation itself is deterministic).
        time_period: The dataset's time period.

    Returns:
        A new frame whose person table carries both hours input leaves and no
        canonical ``weeks_worked`` output.

    Raises:
        ValueError: If the frame is not US-schema or the stage output does
            not cover every person.
        SourceRuntimeError: If required raw ASEC columns are missing.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US hours-worked inputs require the US schema.")
    frame = _without_formula_owned_hours_outputs(frame)
    person = frame.table("person")
    have_all = all(
        column in person.columns for column in US_HOURS_WORKED_OUTPUT_COLUMNS
    )
    if have_all and _weekly_hours_carry_signal(person):
        return frame

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_hours_worked_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_hours_worked": derive_us_hours_worked_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_HOURS_WORKED_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                f"US hours-worked stage output does not cover every person for "
                f"{column!r}."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_HOURS_WORKED_OUTPUT_COLUMNS:
        tables["person"][column] = aligned[column].to_numpy(dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_hours_worked_summary(frame: Frame) -> dict[str, object]:
    """Weighted hours-distribution summary for gates and release manifests."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    weekly = pd.to_numeric(
        person["weekly_hours_worked_before_lsr"], errors="coerce"
    ).fillna(0.0)
    weekly_values = weekly.to_numpy(dtype=np.float64)
    total_weight = float(weights.sum())
    positive = weekly_values > 0
    positive_weight = float(weights[positive].sum())
    worked_share = positive_weight / total_weight if total_weight > 0 else 0.0
    mean_weekly_hours_workers = (
        float(np.average(weekly_values[positive], weights=weights[positive]))
        if positive_weight > 0
        else 0.0
    )
    unique_counts = {
        column: int(pd.to_numeric(person[column], errors="coerce").dropna().nunique())
        for column in US_HOURS_WORKED_OUTPUT_COLUMNS
        if column in person.columns
    }
    return {
        "worked_share": worked_share,
        "mean_weekly_hours_workers": mean_weekly_hours_workers,
        "worked_share_band": list(_WORKED_SHARE_BAND),
        "mean_weekly_hours_band": list(_MEAN_WEEKLY_HOURS_BAND),
        "unique_counts": unique_counts,
    }


def us_hours_worked_signal_gate(frame: Frame) -> GateResult:
    """Require the hours surface to carry a plausible worked distribution.

    Fails when a column is missing or constant, when the weighted share of
    persons with positive usual weekly hours leaves the plausibility band,
    or when mean weekly hours among workers does — each of which reproduces
    (or inverts) the everyone-works-40-hours failure of populace #242.
    """

    person = frame.table("person")
    failures: list[str] = []
    missing = [
        column
        for column in US_HOURS_WORKED_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="hours_worked_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_hours_worked_summary(frame)
    for column, count in summary["unique_counts"].items():
        if count < 2:
            failures.append(
                f"{column}: constant column (one observed value) — the hours "
                "surface carries no signal."
            )
    worked_share = float(summary["worked_share"])
    low, high = _WORKED_SHARE_BAND
    if not (low <= worked_share <= high):
        failures.append(
            f"worked share {worked_share:.3f} outside plausibility band "
            f"[{low}, {high}]."
        )
    mean_hours = float(summary["mean_weekly_hours_workers"])
    low, high = _MEAN_WEEKLY_HOURS_BAND
    if not (low <= mean_hours <= high):
        failures.append(
            f"mean weekly hours among workers {mean_hours:.1f} outside "
            f"plausibility band [{low}, {high}]."
        )
    return GateResult(
        name="hours_worked_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
