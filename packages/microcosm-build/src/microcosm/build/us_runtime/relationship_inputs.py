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

from collections.abc import Mapping
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
from microcosm.build.us_runtime._person_signal_summary import (
    validate_person_signal_summary,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "US_RELATIONSHIP_INPUTS_NONCONSTANT_PERSON_COLUMNS",
    "US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS",
    "US_RELATIONSHIP_INPUTS_REQUIRED_SOURCE_COLUMNS",
    "US_RELATIONSHIP_INPUTS_STAGE_NAME",
    "derive_us_relationship_inputs_from_manifest",
    "prepare_us_relationship_person",
    "us_relationship_inputs_person_carries_signal",
    "us_relationship_inputs_gate_from_summary",
    "us_relationship_inputs_person_gate",
    "us_relationship_inputs_person_summary",
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


def us_relationship_inputs_person_carries_signal(person: pd.DataFrame) -> bool:
    """Return the exact incumbent pass-through decision from the real person table.

    All three outputs must be present and each must have multiple observed
    values. Preserve the legacy null/constant rule; this does not validate raw
    source columns or resolve weights, and is not a scientific signal gate.
    """
    if any(column not in person for column in US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS):
        return False
    return all(
        person[column].dropna().nunique() > 1
        for column in US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS
    )


def _relationship_surface_carries_signal(frame: Frame) -> bool:
    return us_relationship_inputs_person_carries_signal(frame.table("person"))


def _validated_relationship_weights(
    person: pd.DataFrame, weights: np.ndarray
) -> np.ndarray:
    """Return ``weights`` as a float64 array aligned 1:1 with ``person``."""

    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or len(array) != len(person):
        raise ValueError(
            "US relationship-input weights must be a 1-D array aligned 1:1 "
            f"with the person table ({len(person)} row(s)); got shape "
            f"{array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("US relationship-input weights must be finite.")
    if (array < 0.0).any():
        raise ValueError("US relationship-input weights must be nonnegative.")
    return array


def prepare_us_relationship_person(
    person: pd.DataFrame,
    weights: np.ndarray,
    *,
    seed: int,
    time_period: int,
) -> pd.DataFrame:
    """Derive measured ASEC relationship inputs onto a copy of ``person``.

    The deterministic table-helper behind :func:`with_us_relationship_inputs`:
    it always runs the ``relationship_inputs`` source-manifest stage over
    ``person`` and ``weights`` and returns the aligned result. It does not
    decide whether the surface already carries signal — that pass-through
    decision belongs to the Frame wrapper.

    Args:
        person: The actual person table, carrying the raw ASEC source
            columns (``PH_SEQ``, ``P_SEQ``, ``A_MARITL``), in its own index
            and row order.
        weights: Person weights aligned 1:1 with ``person``'s rows (same
            length and row order; need not be reindexed by ``person_id``).
        seed: Build-wide imputation seed threaded to the source-stage
            runtime (the derivation itself is deterministic).
        time_period: The dataset's time period.

    Returns:
        A copy of ``person`` with ``is_household_head``, ``is_separated``,
        and ``is_surviving_spouse`` attached as ``bool`` columns, in
        ``person``'s original index and row order.

    Raises:
        ValueError: If ``weights`` does not align 1:1 with ``person``, is
            not finite/nonnegative, or the stage output does not cover
            every person.
        SourceRuntimeError: If required raw ASEC column(s) are missing or
            malformed.
    """

    weight_values = _validated_relationship_weights(person, weights)
    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = weight_values
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

    result = person.copy(deep=True)
    for column in US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS:
        result[column] = aligned[column].to_numpy(dtype=bool)
    return result


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
    new_person = prepare_us_relationship_person(
        person,
        frame.resolve_weights("person").values,
        seed=seed,
        time_period=time_period,
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = new_person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_relationship_inputs_person_summary(
    person: pd.DataFrame, weights: np.ndarray
) -> dict[str, object]:
    """Return weighted relationship shares and one-head invariants.

    The real-person-table counterpart of :func:`us_relationship_inputs_summary`,
    for callers (e.g. graph adapters) that hold a person table and an
    explicit weight vector without a :class:`~microcosm.frame.Frame`.

    Args:
        person: The actual person table, already carrying
            ``is_household_head``, ``is_separated``, and
            ``is_surviving_spouse``.
        weights: Person weights aligned 1:1 with ``person``'s rows.

    Returns:
        The same summary payload as :func:`us_relationship_inputs_summary`.
    """

    weight_values = _validated_relationship_weights(person, weights)
    total_weight = float(weight_values.sum())

    def _share(column: str) -> float:
        values = person[column].fillna(False).astype(bool).to_numpy()
        return (
            float(weight_values[values].sum()) / total_weight
            if total_weight > 0
            else 0.0
        )

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


def us_relationship_inputs_summary(frame: Frame) -> dict[str, object]:
    """Return weighted relationship shares and one-head invariants."""

    return us_relationship_inputs_person_summary(
        frame.table("person"), frame.resolve_weights("person").values
    )


def us_relationship_inputs_gate_from_summary(
    summary: Mapping[str, object],
) -> GateResult:
    """Check relationship-input plausibility bands and invariants from a summary.

    The pure decision core of :func:`us_relationship_inputs_signal_gate`,
    factored out so graph adapters can reuse the incumbent checks — same
    bands, order, and meaning — against a summary computed off the real
    person table (see :func:`us_relationship_inputs_person_summary`)
    without a :class:`~microcosm.frame.Frame`. Assumes the caller has
    already confirmed the three output columns are present; missing
    columns are a separate failure mode (see
    :func:`us_relationship_inputs_person_gate`).

    Raises:
        ValueError: If required fields/counts are missing, measurements are
            malformed, or supplied bands differ from the registered policy.
    """

    validate_person_signal_summary(
        summary,
        family="relationship",
        outputs=US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS,
        share_bands={
            "household_head_share": (
                "household_head_share_band",
                _HOUSEHOLD_HEAD_SHARE_BAND,
            ),
            "separated_share": ("separated_share_band", _SEPARATED_SHARE_BAND),
            "surviving_spouse_share": (
                "surviving_spouse_share_band",
                _SURVIVING_SPOUSE_SHARE_BAND,
            ),
        },
        invariants=("households_without_exactly_one_head", "separated_and_surviving"),
    )
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


def us_relationship_inputs_person_gate(
    person: pd.DataFrame, weights: np.ndarray
) -> GateResult:
    """Require plausible signal and exactly one ASEC head per household.

    The real-person-table counterpart of
    :func:`us_relationship_inputs_signal_gate`, composing
    :func:`us_relationship_inputs_person_summary` and
    :func:`us_relationship_inputs_gate_from_summary` exactly as the Frame
    wrapper does.
    """

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
    summary = us_relationship_inputs_person_summary(person, weights)
    return us_relationship_inputs_gate_from_summary(summary)


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
    return us_relationship_inputs_gate_from_summary(summary)
