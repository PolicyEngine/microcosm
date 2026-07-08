"""SNAP eligibility and exemption inputs from measured CPS ASEC variables.

Without this stage the published dataset stores none of the demographic
inputs that SNAP work-requirement and student-eligibility rules read, so
each one silently takes its engine default: ``is_disabled`` False (no
disability exemptions anywhere), ``is_full_time_college_student`` False
(the SNAP student rules never bind), ``own_children_in_household`` 0
(``is_parent`` is False for everyone, erasing parent/child exemption
logic), and ``veterans_benefits`` 0 (``is_veteran`` — a formula over this
input — is False for everyone, erasing the pre-HR1 veteran ABAWD
exemption). That is the failure mode of populace issues #244/#248.

The stage writes five PolicyEngine-facing person input columns, each a
direct mapping of measured CPS ASEC person variables, reproducing the
retired enhanced-CPS pipeline's constructions (cited per mapping below
against the retired enhanced-CPS repo @ 42ed5d45c5; nothing is imputed):

- ``is_disabled`` ← any of the six ASEC disability-difficulty items
  (PEDISDRS, PEDISEAR, PEDISEYE, PEDISOUT, PEDISPHY, PEDISREM) == 1,
  then aligned with reported SSI receipt: an under-65 person with
  positive ``SSI_VAL`` is disabled, since SSI requires under-65
  recipients to be disabled or blind. Both steps are retired-pipeline
  constructions: the battery is ``datasets/cps/cps.py`` (the
  ``CPS_SSI_DISABILITY_DIFFICULTY_COLUMNS`` any-of), and the alignment
  is ``align_reported_ssi_disability`` in ``datasets/cps/takeup.py``,
  applied to ``is_disabled`` in ``cps.py``'s take-up block.
- ``is_blind`` ← PEDISEYE == 1 ("serious difficulty seeing even when
  wearing glasses").
- ``is_full_time_college_student`` ← A_HSCOL == 2 (enrolled in college
  or university) & A_FTPT == 1 (enrolled full-time in school). This is
  the retired pipeline's primary branch — ``cps.py`` applies the A_FTPT
  filter whenever the column is loaded, and ``census_cps.py`` includes
  A_FTPT in the raw person load, so the strict form is what the
  published surface ran. The retired A_HSCOL-only fallback (for loads
  without A_FTPT) is deliberately not reproduced: this stage requires
  A_FTPT and fails loudly if it is missing.
- ``own_children_in_household`` ← count of household members whose
  parent pointers (PEPAR1/PEPAR2, parent line numbers within PH_SEQ)
  resolve to the person.
- ``veterans_benefits`` ← VET_VAL (veterans' payments received).

Ownership note: ``is_veteran`` is formula-owned in PolicyEngine-US
(derived from ``veterans_benefits``), so this stage exports the input
dollars and never the derived flag — storing formula-owned variables is
the over-export failure of populace #24.

Healing behavior: a frame that already carries all five columns *with
signal* passes through untouched (idempotent). A frame carrying a
constant ``is_disabled`` — indistinguishable from the engine's broadcast
default — is recomputed from the raw columns rather than trusted.
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
    "US_ELIGIBILITY_INPUTS_NONCONSTANT_PERSON_COLUMNS",
    "US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS",
    "US_ELIGIBILITY_INPUTS_REQUIRED_SOURCE_COLUMNS",
    "US_ELIGIBILITY_INPUTS_STAGE_NAME",
    "derive_us_eligibility_inputs_from_manifest",
    "us_eligibility_inputs_signal_gate",
    "us_eligibility_inputs_stage_spec",
    "us_eligibility_inputs_summary",
    "with_us_eligibility_inputs",
]

US_ELIGIBILITY_INPUTS_STAGE_NAME = "eligibility_inputs"

#: The PolicyEngine-facing person input columns this stage owns.
US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS: tuple[str, ...] = (
    "is_disabled",
    "is_blind",
    "is_full_time_college_student",
    "own_children_in_household",
    "veterans_benefits",
)

#: Release gates require these person columns to carry signal (≥2 values).
US_ELIGIBILITY_INPUTS_NONCONSTANT_PERSON_COLUMNS: tuple[str, ...] = (
    US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS
)

#: The six ASEC disability-difficulty items; 1 means "yes".
_DISABILITY_DIFFICULTY_COLUMNS: tuple[str, ...] = (
    "PEDISDRS",
    "PEDISEAR",
    "PEDISEYE",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
)

#: Raw CPS ASEC person columns the derivation reads. PH_SEQ/A_LINENO are
#: the household sequence and person line number the parent pointers
#: (PEPAR1/PEPAR2) resolve against; SSI_VAL and A_AGE feed the reported-SSI
#: disability alignment.
US_ELIGIBILITY_INPUTS_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    *_DISABILITY_DIFFICULTY_COLUMNS,
    "A_HSCOL",
    "A_FTPT",
    "PEPAR1",
    "PEPAR2",
    "PH_SEQ",
    "A_LINENO",
    "VET_VAL",
    "SSI_VAL",
    "A_AGE",
)

_SSI_DISABILITY_AGE_LIMIT = 65

#: Weighted share of all persons flagged disabled must land in this band.
#: The six-item ASEC difficulty battery yields ~10% of all persons
#: (children included); the SSI alignment adds a fraction of a point.
_DISABLED_SHARE_BAND = (0.05, 0.20)

#: Weighted share of all persons enrolled full-time in college must land
#: in this band (NCES: ~11M full-time postsecondary students ≈ 3-4% of
#: the population).
_FULL_TIME_STUDENT_SHARE_BAND = (0.01, 0.08)

#: Weighted share of all persons with at least one own child in the
#: household must land in this band (roughly a quarter of persons are
#: parents of a co-resident child).
_PARENT_SHARE_BAND = (0.12, 0.40)

#: Weighted share of all persons with positive veterans' payments must
#: land in this band (VA compensation/pension reaches ~2% of persons;
#: the ASEC measures somewhat less).
_VETERANS_BENEFITS_SHARE_BAND = (0.002, 0.04)

_PERSON_WEIGHT_COLUMN = "person_weight"

_DERIVE_ELIGIBILITY_INPUTS_PARAMETER_KEYS = frozenset()


def us_eligibility_inputs_stage_spec() -> SourceStageSpec:
    """Load the packaged ``eligibility_inputs`` source-stage manifest entry."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_ELIGIBILITY_INPUTS_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no "
            f"{US_ELIGIBILITY_INPUTS_STAGE_NAME!r} stage."
        )
    return stage_map[US_ELIGIBILITY_INPUTS_STAGE_NAME]


def _own_children_in_household(frame: pd.DataFrame) -> np.ndarray:
    """Count each person's own children in the household via parent pointers.

    PEPAR1/PEPAR2 carry the A_LINENO of the person's first and second
    parent within the same PH_SEQ household (or -1 when absent) — the
    retired enhanced-CPS pipeline's ``children_per_parent`` construction.
    """

    counts: dict[tuple[float, float], int] = {}
    ph_seq = pd.to_numeric(frame["PH_SEQ"], errors="coerce").to_numpy()
    for pointer_column in ("PEPAR1", "PEPAR2"):
        pointer = pd.to_numeric(frame[pointer_column], errors="coerce").to_numpy()
        for household, parent_line in zip(ph_seq, pointer, strict=True):
            if parent_line > 0:
                key = (household, parent_line)
                counts[key] = counts.get(key, 0) + 1
    line_number = pd.to_numeric(frame["A_LINENO"], errors="coerce").to_numpy()
    return np.array(
        [
            float(counts.get((household, line), 0))
            for household, line in zip(ph_seq, line_number, strict=True)
        ],
        dtype=np.float64,
    )


def derive_us_eligibility_inputs_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Map raw ASEC demographic columns onto the PolicyEngine input names.

    The current frame must be the raw-column person table (from the stage's
    ``read_table`` operation).
    """

    if operation.kind != "derive_eligibility_inputs":
        raise SourceRuntimeError(
            f"US eligibility-inputs derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US eligibility-inputs derivation requires the person table to "
            "be read first."
        )
    unexpected = sorted(
        set(operation.parameters) - _DERIVE_ELIGIBILITY_INPUTS_PARAMETER_KEYS
    )
    if unexpected:
        raise SourceRuntimeError(
            f"US eligibility-inputs derivation received unsupported "
            f"parameter(s): {unexpected}."
        )
    missing = [
        column
        for column in US_ELIGIBILITY_INPUTS_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            f"US eligibility-inputs derivation requires raw ASEC column(s): {missing}."
        )

    def _numeric(column: str) -> pd.Series:
        return pd.to_numeric(frame[column], errors="coerce")

    result = frame.copy(deep=True)
    difficulty_any = np.column_stack(
        [
            (_numeric(column) == 1).to_numpy()
            for column in _DISABILITY_DIFFICULTY_COLUMNS
        ]
    ).any(axis=1)
    # SSI alignment: under-65 reported-SSI recipients are disabled or blind
    # by program rule (the retired pipeline's align_reported_ssi_disability).
    reported_ssi = (_numeric("SSI_VAL").fillna(0.0) > 0).to_numpy()
    under_age_limit = (
        _numeric("A_AGE").fillna(0.0) < _SSI_DISABILITY_AGE_LIMIT
    ).to_numpy()
    result["is_disabled"] = difficulty_any | (reported_ssi & under_age_limit)
    result["is_blind"] = (_numeric("PEDISEYE") == 1).to_numpy()
    result["is_full_time_college_student"] = (
        (_numeric("A_HSCOL") == 2) & (_numeric("A_FTPT") == 1)
    ).to_numpy()
    result["own_children_in_household"] = _own_children_in_household(frame)
    result["veterans_benefits"] = np.maximum(
        _numeric("VET_VAL").fillna(0.0).to_numpy(dtype=np.float64), 0.0
    )
    return result


def _disabled_carries_signal(person: pd.DataFrame) -> bool:
    """Whether the persisted disability column is trustworthy as-is.

    A constant column (one observed value) is indistinguishable from the
    engine's broadcast default and must be recomputed, not passed through.
    """

    values = person["is_disabled"].dropna()
    return values.nunique() > 1


def with_us_eligibility_inputs(frame: Frame, *, seed: int, time_period: int) -> Frame:
    """Run the ``eligibility_inputs`` manifest stage over a US frame.

    A frame already carrying all five output columns with a non-constant
    disability distribution passes through untouched (idempotent). Any
    other surface — columns missing, or disability constant at the engine
    default — is recomputed from the raw ASEC columns.

    Args:
        frame: A US-schema frame whose person table still carries the raw
            CPS ASEC source columns (unless the outputs already carry
            signal).
        seed: Build-wide imputation seed (recorded in the stage run; the
            derivation itself is deterministic).
        time_period: The dataset's time period.

    Returns:
        A new frame whose person table carries all five eligibility
        columns.

    Raises:
        ValueError: If the frame is not US-schema or the stage output does
            not cover every person.
        SourceRuntimeError: If required raw ASEC columns are missing.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US eligibility inputs require the US schema.")
    person = frame.table("person")
    have_all = all(
        column in person.columns for column in US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS
    )
    if have_all and _disabled_carries_signal(person):
        return frame

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_eligibility_inputs_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_eligibility_inputs": derive_us_eligibility_inputs_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                f"US eligibility-inputs stage output does not cover every "
                f"person for {column!r}."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    bool_columns = ("is_disabled", "is_blind", "is_full_time_college_student")
    for column in US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS:
        if column in bool_columns:
            tables["person"][column] = aligned[column].to_numpy(dtype=bool)
        else:
            tables["person"][column] = aligned[column].to_numpy(dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def us_eligibility_inputs_summary(frame: Frame) -> dict[str, object]:
    """Weighted eligibility-share summary for gates and release manifests."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())

    def _share(mask: np.ndarray) -> float:
        return float(weights[mask].sum()) / total_weight if total_weight > 0 else 0.0

    disabled = person["is_disabled"].astype(bool).to_numpy()
    student = person["is_full_time_college_student"].astype(bool).to_numpy()
    children = pd.to_numeric(
        person["own_children_in_household"], errors="coerce"
    ).fillna(0.0)
    veterans = pd.to_numeric(person["veterans_benefits"], errors="coerce").fillna(0.0)
    unique_counts = {
        column: int(person[column].dropna().nunique())
        for column in US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS
        if column in person.columns
    }
    return {
        "disabled_share": _share(disabled),
        "full_time_college_student_share": _share(student),
        "parent_share": _share(children.to_numpy() > 0),
        "veterans_benefits_share": _share(veterans.to_numpy() > 0),
        "disabled_share_band": list(_DISABLED_SHARE_BAND),
        "full_time_student_share_band": list(_FULL_TIME_STUDENT_SHARE_BAND),
        "parent_share_band": list(_PARENT_SHARE_BAND),
        "veterans_benefits_share_band": list(_VETERANS_BENEFITS_SHARE_BAND),
        "unique_counts": unique_counts,
    }


def us_eligibility_inputs_signal_gate(frame: Frame) -> GateResult:
    """Require the eligibility surface to carry plausible distributions.

    Fails when a column is missing or constant, or when the weighted
    disabled, full-time-student, parent, or veterans'-payment shares leave
    their plausibility bands — each of which reproduces (or inverts) the
    everyone-defaults failure of populace #244.
    """

    person = frame.table("person")
    failures: list[str] = []
    missing = [
        column
        for column in US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="eligibility_inputs_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_eligibility_inputs_summary(frame)
    for column, count in summary["unique_counts"].items():
        if count < 2:
            failures.append(
                f"{column}: constant column (one observed value) — the "
                "eligibility surface carries no signal."
            )
    for share_key, band_key, label in (
        ("disabled_share", "disabled_share_band", "disabled share"),
        (
            "full_time_college_student_share",
            "full_time_student_share_band",
            "full-time college student share",
        ),
        ("parent_share", "parent_share_band", "parent share"),
        (
            "veterans_benefits_share",
            "veterans_benefits_share_band",
            "veterans' payments share",
        ),
    ):
        share = float(summary[share_key])
        low, high = summary[band_key]
        if not (low <= share <= high):
            failures.append(
                f"{label} {share:.3f} outside plausibility band [{low}, {high}]."
            )
    return GateResult(
        name="eligibility_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
