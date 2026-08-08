"""Measured CPS ASEC household-head and marital-status input leaves.

The archived eCPS construction at commit
``42ed5d45c56df80d754fbe24cce21cfeb8d05cbe`` derives these inputs directly
in ``datasets/cps/cps.py``:

- line 1074: ``is_household_head = P_SEQ == 1``;
- line 1212: ``is_surviving_spouse = A_MARITL == 4``; and
- line 1213: ``is_separated = A_MARITL == 6``.

All three SHA-locked ASEC vintages retain the required raw columns. Nothing is
imputed, and the stage fails closed if a source column is missing, malformed,
or does not identify exactly one household head per source household.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "US_RELATIONSHIP_INPUTS_NONCONSTANT_PERSON_COLUMNS",
    "US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS",
    "US_RELATIONSHIP_INPUTS_REQUIRED_SOURCE_COLUMNS",
    "US_RELATIONSHIP_INPUTS_STAGE_NAME",
    "derive_us_relationship_inputs_from_manifest",
    "us_relationship_inputs_signal_gate",
    "us_relationship_inputs_stage_spec",
    "us_relationship_inputs_summary",
    "with_us_relationship_inputs",
]

US_RELATIONSHIP_INPUTS_STAGE_NAME = "relationship_inputs"

US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS: tuple[str, ...] = (
    "is_household_head",
    "is_separated",
    "is_surviving_spouse",
)

US_RELATIONSHIP_INPUTS_NONCONSTANT_PERSON_COLUMNS = (
    US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS
)

US_RELATIONSHIP_INPUTS_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "PH_SEQ",
    "P_SEQ",
    "A_MARITL",
)

_PERSON_WEIGHT_COLUMN = "person_weight"
_HOUSEHOLD_HEAD_SHARE_BAND = (0.30, 0.55)
_SEPARATED_SHARE_BAND = (0.003, 0.04)
_SURVIVING_SPOUSE_SHARE_BAND = (0.02, 0.08)
_VALID_A_MARITL_CODES = frozenset(range(1, 8))
_DERIVE_RELATIONSHIP_INPUTS_PARAMETER_KEYS = frozenset()


def us_relationship_inputs_stage_spec() -> SourceStageSpec:
    """Load the packaged ``relationship_inputs`` stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_RELATIONSHIP_INPUTS_STAGE_NAME not in stage_map:
        raise ValueError(
            "US source manifest declares no "
            f"{US_RELATIONSHIP_INPUTS_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_RELATIONSHIP_INPUTS_STAGE_NAME]
    if tuple(spec.outputs) != US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS:
        raise ValueError(
            f"{US_RELATIONSHIP_INPUTS_STAGE_NAME!r} manifest outputs do not "
            "match the runtime-owned relationship input family."
        )
    return spec


def _strict_integer_source(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum: int,
    allowed: frozenset[int] | None = None,
) -> np.ndarray:
    """Return a source column as integers or fail on missing/invalid values."""

    numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric) & (numeric == np.floor(numeric))
    valid &= numeric >= float(minimum)
    if allowed is not None:
        valid &= np.isin(numeric, np.fromiter(allowed, dtype=np.int64))
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise SourceRuntimeError(
            f"US relationship-input derivation requires valid integer {column}; "
            f"invalid row(s): {rows}."
        )
    return numeric.astype(np.int64)


def derive_us_relationship_inputs_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Map exact ASEC head and marital codes to PolicyEngine input leaves."""

    if operation.kind != "derive_relationship_inputs":
        raise SourceRuntimeError(
            "US relationship-input derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US relationship-input derivation requires the person table to be "
            "read first."
        )
    unexpected = sorted(
        set(operation.parameters) - _DERIVE_RELATIONSHIP_INPUTS_PARAMETER_KEYS
    )
    if unexpected:
        raise SourceRuntimeError(
            "US relationship-input derivation received unsupported "
            f"parameter(s): {unexpected}."
        )
    missing = [
        column
        for column in US_RELATIONSHIP_INPUTS_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            f"US relationship-input derivation requires raw ASEC column(s): {missing}."
        )

    household = _strict_integer_source(frame, "PH_SEQ", minimum=1)
    person_sequence = _strict_integer_source(frame, "P_SEQ", minimum=1)
    marital_status = _strict_integer_source(
        frame,
        "A_MARITL",
        minimum=1,
        allowed=_VALID_A_MARITL_CODES,
    )
    is_head = person_sequence == 1
    grouping_household = (
        _strict_integer_source(frame, "person_household_id", minimum=1)
        if "person_household_id" in frame
        else household
    )
    head_counts = pd.Series(is_head).groupby(grouping_household, sort=False).sum()
    bad_households = head_counts.index[head_counts.to_numpy() != 1]
    if len(bad_households):
        examples = bad_households[:5].tolist()
        raise SourceRuntimeError(
            "US relationship-input derivation requires exactly one P_SEQ == 1 "
            f"person per frame household; invalid household(s): {examples}."
        )

    result = frame.copy(deep=True)
    result["is_household_head"] = is_head
    result["is_separated"] = marital_status == 6
    result["is_surviving_spouse"] = marital_status == 4
    return result


def _relationship_surface_carries_signal(frame: Frame) -> bool:
    person = frame.table("person")
    if any(column not in person for column in US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS):
        return False
    return all(
        person[column].dropna().nunique() > 1
        for column in US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS
    )


def with_us_relationship_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
) -> Frame:
    """Materialize measured ASEC relationship inputs on a US frame."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US relationship inputs require the US schema.")
    if _relationship_surface_carries_signal(frame):
        return frame

    person = frame.table("person")
    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_relationship_inputs_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_relationship_inputs": (derive_us_relationship_inputs_from_manifest)
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                "US relationship-input stage output does not cover every person "
                f"for {column!r}."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS:
        tables["person"][column] = aligned[column].to_numpy(dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_relationship_inputs_summary(frame: Frame) -> dict[str, object]:
    """Return weighted relationship shares and one-head invariants."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())

    def _share(column: str) -> float:
        values = person[column].fillna(False).astype(bool).to_numpy()
        return float(weights[values].sum()) / total_weight if total_weight > 0 else 0.0

    household_column = (
        "person_household_id" if "person_household_id" in person else "PH_SEQ"
    )
    head_counts = (
        person["is_household_head"]
        .fillna(False)
        .astype(bool)
        .groupby(person[household_column], sort=False)
        .sum()
    )
    separated = person["is_separated"].fillna(False).astype(bool).to_numpy()
    surviving = person["is_surviving_spouse"].fillna(False).astype(bool).to_numpy()
    return {
        "household_head_share": _share("is_household_head"),
        "separated_share": _share("is_separated"),
        "surviving_spouse_share": _share("is_surviving_spouse"),
        "household_head_share_band": list(_HOUSEHOLD_HEAD_SHARE_BAND),
        "separated_share_band": list(_SEPARATED_SHARE_BAND),
        "surviving_spouse_share_band": list(_SURVIVING_SPOUSE_SHARE_BAND),
        "households_without_exactly_one_head": int((head_counts != 1).sum()),
        "separated_and_surviving": int(np.count_nonzero(separated & surviving)),
        "unique_counts": {
            column: int(person[column].dropna().nunique())
            for column in US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS
        },
    }


def us_relationship_inputs_signal_gate(frame: Frame) -> GateResult:
    """Require plausible signal and exactly one ASEC head per household."""

    person = frame.table("person")
    missing = [
        column
        for column in US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS
        if column not in person
    ]
    if missing:
        return GateResult(
            name="relationship_inputs_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_relationship_inputs_summary(frame)
    failures: list[str] = []
    for share_key, band_key, label in (
        (
            "household_head_share",
            "household_head_share_band",
            "household-head weighted share",
        ),
        ("separated_share", "separated_share_band", "separated weighted share"),
        (
            "surviving_spouse_share",
            "surviving_spouse_share_band",
            "surviving-spouse weighted share",
        ),
    ):
        share = float(summary[share_key])
        lower, upper = summary[band_key]
        if not lower <= share <= upper:
            failures.append(f"{label} {share:.6f} outside [{lower:.6f}, {upper:.6f}].")
    invalid_heads = int(summary["households_without_exactly_one_head"])
    if invalid_heads:
        failures.append(
            f"{invalid_heads} household(s) do not carry exactly one "
            "is_household_head person."
        )
    overlap = int(summary["separated_and_surviving"])
    if overlap:
        failures.append(f"{overlap} person(s) are both separated and surviving spouse.")
    for column, count in summary["unique_counts"].items():
        if int(count) < 2:
            failures.append(f"{column} is degenerate with {count} distinct value(s).")
    return GateResult(
        name="relationship_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
