"""SSI and Medicaid take-up for the ACS rows of the retained ACS local lane.

The ACS transfer leaves take-up draws to the runtime, and the local lane runs
neither the fiscal lane's ``ssi_take_up`` nor its ``medicaid_take_up`` stage.
Every ACS person therefore reached the engine pass with
``takes_up_ssi_if_eligible`` and ``takes_up_medicaid_if_eligible`` missing,
and the reviewed-null fill gave both the engine default, ``True``
(microcosm#1022): every ACS person the engine found eligible received SSI and
enrolled in Medicaid. With real SSI disability criteria on the ACS rows
(:mod:`~microcosm.build.us_runtime.acs_local_ssi_disability`), universal SSI
take-up overstates ACS SSI, and universal Medicaid take-up makes the state CMS
enrollment targets shrink the weights of eligible ACS persons instead of
selecting enrollees among them (the microcosm#170 bias).

This stage assigns both flags on ACS rows with the donor stages' own methods,
against the donor stages' counts scaled to the ACS rows' share of the frame.
It fills only missing ACS cells; donor-spine values and any stored ACS value
are kept.

- **Engine pre-pass.** Like the donor stages, both assignments read the
  engine: SSI candidates are ``uncapped_ssi > 0`` in December 2024 and the
  Medicaid domain is ``is_medicaid_eligible``. The caller supplies both as
  callables over a view of the ACS households only (the builder owns the
  simulations and their household batching). In that view a missing take-up
  cell reads ``True``, neither variable depends on its own flag, and the
  Medicaid pass sees the SSI flags just assigned, because SSI receipt is a
  Medicaid eligibility category (``is_ssi_recipient_for_medicaid``), exactly
  as the fiscal lane assigns SSI before Medicaid.
- **Target scaling.** The frame pools the donor and ACS spines, each carrying
  part of the population weight, and the donor rows already carry flags drawn
  against the full national and state counts. Each ACS target is the donor
  stage's count times the ACS rows' share of the frame's person weight in the
  same SSA age band or state (pre-calibration weights), so the pooled frame
  carries the count before calibration moves any weight.
- **SSI** follows :mod:`~microcosm.build.us_runtime.ssi_take_up`. An ACS
  person with ``ssi_reported`` (the adjusted native ``SSIP``) above zero
  always takes SSI up, the ACS counterpart of the donor's ``SSI_VAL`` anchor.
  ``SSIP`` is asked from age 15, so a blank below 15 reads as no report and a
  blank at 15 or over is refused. Everyone else draws once against the band
  prior ``(target - reporter floor) / (capacity - reporter floor)`` of
  :func:`~microcosm.build.us_runtime.ssi_take_up._band_prior`, measured on the
  ACS rows' candidate weight, with its fallbacks. All three SSA bands draw, as
  on the donor, and the under-18 band is fenced from grading exactly as the
  donor's delivery gate fences it.
- **Medicaid** runs the donor's ``medicaid_take_up`` manifest stage unchanged
  on the ACS rows: ``HINS4 == 1`` anchors always take up, everyone else draws
  against the in-build state fill rate, and the assignment is greedily
  calibrated to each state's scaled CMS count among eligible non-anchored
  persons. ACS ``HINS4`` is "Medicaid, Medical Assistance, or any kind of
  government-assistance plan for those with low incomes or a disability", so
  it also covers CHIP and state-funded plans and is broader than the ASEC's
  current-Medicaid anchor. An anchor mass above a state's scaled count is that
  state's reachable floor (the donor gate's rule): anchors are never dropped.
- **Draws** come from seeded blake2b uniforms on the donor stages' streams,
  keyed on ``acs_2024_1yr:SERIALNO:SPORDER``, so a person's draw depends only
  on the build seed and their own record.

The receipt records each band's and state's count, ACS share, scaled target,
capacity and delivered weight on the pre-calibration frame weights, and a
digest of the ACS assignment.
:func:`acs_local_ssi_medicaid_take_up_signal_gate` grades both the flags and
the receipt.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.medicaid_take_up import (
    _DRAW_COLUMN as _MEDICAID_DRAW_COLUMN,
)
from microcosm.build.us_runtime.medicaid_take_up import (
    US_MEDICAID_ELIGIBILITY_COLUMN,
    US_MEDICAID_ENROLLMENT_TARGET_TABLE,
    US_MEDICAID_TAKE_UP_ANCHOR,
    US_MEDICAID_TAKE_UP_STAGE,
    US_MEDICAID_TAKE_UP_VARIABLE,
    _normalize_state_fips,
    _stable_person_draws,
    us_medicaid_take_up_diagnostics,
    us_medicaid_take_up_gate,
    with_us_medicaid_take_up_rate,
)
from microcosm.build.us_runtime.ssi_take_up import (
    US_SSI_TAKE_UP_AGE_TARGETS,
    US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE,
    US_SSI_TAKE_UP_ENFORCED_BAND_KEYS,
    US_SSI_TAKE_UP_OUTPUT_COLUMNS,
    _age_band_values,
    _band_prior,
    _normalize_targets,
    _stable_source_draw,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_MEDICAID_ELIGIBILITY_DEFINITION",
    "ACS_LOCAL_MEDICAID_TAKE_UP_COLUMN",
    "ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE",
    "ACS_LOCAL_SSI_CANDIDATE_DEFINITION",
    "ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS",
    "ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME",
    "ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE",
    "ACS_LOCAL_SSI_MEDICAID_TAKE_UP_METHOD",
    "ACS_LOCAL_SSI_TAKE_UP_COLUMN",
    "acs_local_ssi_medicaid_take_up_assignment",
    "acs_local_ssi_medicaid_take_up_signal_gate",
    "with_acs_local_ssi_medicaid_take_up",
    "with_recorded_acs_local_ssi_medicaid_take_up",
]

ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE = "microcosm#1022"
ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME = "acs_local_ssi_medicaid_take_up_signal"
ACS_LOCAL_SSI_MEDICAID_TAKE_UP_METHOD = (
    "donor_stage_methods_on_acs_rows_at_acs_share_scaled_targets"
)
ACS_LOCAL_SSI_TAKE_UP_COLUMN = US_SSI_TAKE_UP_OUTPUT_COLUMNS[0]
ACS_LOCAL_MEDICAID_TAKE_UP_COLUMN = US_MEDICAID_TAKE_UP_VARIABLE
#: The person flags this stage owns on ACS rows.
ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS: tuple[str, ...] = (
    ACS_LOCAL_SSI_TAKE_UP_COLUMN,
    ACS_LOCAL_MEDICAID_TAKE_UP_COLUMN,
)
#: An enforced SSI band whose prior was count-truthful (neither saturated nor
#: met by its anchors) must deliver its scaled count within this relative
#: tolerance: the donor's delivery envelope. The expected delivery equals the
#: count by construction, so only a broken draw misses it.
ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE = US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE
ACS_LOCAL_SSI_CANDIDATE_DEFINITION = "uncapped_ssi > 0 at 2024-12"
ACS_LOCAL_MEDICAID_ELIGIBILITY_DEFINITION = (
    "is_medicaid_eligible at 2024, with the ACS rows' SSI take-up assigned"
)

_SSI = ACS_LOCAL_SSI_TAKE_UP_COLUMN
_MEDICAID = ACS_LOCAL_MEDICAID_TAKE_UP_COLUMN
_AGE = "age"
#: Adjusted native ACS ``SSIP``: the SSI reporter anchor.
_REPORTED_SSI = "ssi_reported"
#: ``SSIP`` is asked of persons aged 15 and over.
_SSIP_MINIMUM_AGE = 15
#: ACS Medicaid/means-tested coverage item (1 yes, 2 no), asked of everyone.
_MEDICAID_COVERAGE = "HINS4"
_MEDICAID_COVERED = 1
_STATE = "state_fips"
_DRAW_KEY_FORMAT = f"{ACS_2024_1YR_SPINE}:SERIALNO:SPORDER"
_WEIGHTS_BASIS = "acs_rows_pre_calibration_frame_person_weights"
_BAND_KEYS = tuple(band.key for band in US_SSI_TAKE_UP_AGE_TARGETS)
_BAND_NUMBERS = (
    "scaled_target",
    "candidate_capacity",
    "reporter_candidate_floor",
    "selected_recipient_weight",
)

#: Person-aligned engine values over a view of the ACS households.
EngineValues = Callable[[Frame], np.ndarray]


def _boolean_cells(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(flags, present, invalid)`` of a nullable boolean column."""

    present = values.notna().to_numpy(dtype=bool)
    valid = present & values.isin([0, 1]).to_numpy(dtype=bool)
    flags = np.zeros(len(values), dtype=bool)
    flags[valid] = values[valid].astype(bool).to_numpy(dtype=bool)
    return flags, present, present & ~valid


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def _share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def _finite(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _filled(values: pd.Series, missing: np.ndarray, assigned: np.ndarray) -> pd.Series:
    if not missing.any():
        return values
    filled = values.astype(object)
    filled.loc[missing] = assigned[missing]
    return filled.astype(bool) if filled.notna().all() else filled


def _acs_draw_keys(household: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """``acs_2024_1yr:SERIALNO:SPORDER`` per ACS person; must be unique."""

    serial = rows["person_household_id"].map(
        household.set_index("household_id")["SERIALNO"]
    )
    order = pd.to_numeric(rows["SPORDER"], errors="coerce")
    if serial.isna().any() or order.isna().any():
        raise ValueError(
            "Every ACS person needs its household SERIALNO and its SPORDER to key "
            "the SSI and Medicaid take-up draws."
        )
    keys = (
        f"{ACS_2024_1YR_SPINE}:"
        + serial.astype(str)
        + ":"
        + order.astype(np.int64).astype(str)
    ).reset_index(drop=True)
    if keys.duplicated().any():
        raise ValueError(
            f"{int(keys.duplicated().sum())} ACS person draw key(s) repeat; "
            f"{_DRAW_KEY_FORMAT} must identify one person."
        )
    return keys.to_numpy(dtype=object)


def _ssi_reporters(rows: pd.DataFrame, age: np.ndarray) -> np.ndarray:
    """ACS persons with ``ssi_reported > 0``; a blank reads as none below 15."""

    raw = rows[_REPORTED_SSI]
    values = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & values.isna()
    if invalid.any() or not np.isfinite(values.fillna(0.0).to_numpy()).all():
        raise ValueError(
            f"ACS {_REPORTED_SSI} contains nonnumeric or nonfinite values; the SSI "
            "reporter anchor cannot be read."
        )
    blank = values.isna().to_numpy(dtype=bool) & (age >= _SSIP_MINIMUM_AGE)
    if blank.any():
        raise ValueError(
            f"{int(blank.sum())} ACS person(s) aged {_SSIP_MINIMUM_AGE} or over have "
            f"no {_REPORTED_SSI}; SSIP is asked from {_SSIP_MINIMUM_AGE}, so a blank "
            "there is not a report of no SSI."
        )
    return values.fillna(0.0).to_numpy(dtype=np.float64) > 0.0


def _checked_state_targets(state_targets: pd.DataFrame) -> pd.DataFrame:
    """One finite, nonnegative CMS count per zero-padded state code."""

    missing = [
        column for column in ("state_fips", "target") if column not in state_targets
    ]
    if missing:
        raise ValueError(
            "ACS Medicaid take-up state targets require columns "
            f"['state_fips', 'target']; missing {missing}."
        )
    if state_targets.empty:
        raise ValueError(
            "ACS Medicaid take-up requires CMS state enrollment targets; an empty "
            "table would leave ACS enrollment anchored-only."
        )
    table = pd.DataFrame(
        {
            "state_fips": _normalize_state_fips(state_targets["state_fips"].to_numpy()),
            "target": pd.to_numeric(state_targets["target"], errors="coerce"),
        }
    )
    if table["state_fips"].duplicated().any():
        raise ValueError(
            "ACS Medicaid take-up state targets repeat state(s) "
            f"{sorted(table.loc[table['state_fips'].duplicated(), 'state_fips'])}."
        )
    values = table["target"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("ACS Medicaid take-up state targets must be finite and >= 0.")
    return table


def _acs_weight_shares(
    groups: np.ndarray, acs: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """The ACS rows' share of the frame's person weight in each group."""

    table = pd.DataFrame(
        {
            "group": np.asarray(groups).astype(str),
            "weight": weights,
            "acs_weight": np.where(acs, weights, 0.0),
        }
    )
    sums = table.groupby("group", sort=True)[["weight", "acs_weight"]].sum()
    return {
        str(group): (
            float(row["acs_weight"]) / float(row["weight"])
            if row["weight"] > 0
            else 0.0
        )
        for group, row in sums.iterrows()
    }


def _engine_values(
    function: EngineValues, view: Frame, *, rows: int, label: str
) -> np.ndarray:
    values = np.asarray(function(view))
    if values.shape != (rows,):
        raise ValueError(
            f"The engine pre-pass returned {values.shape} {label} value(s) for "
            f"{rows} ACS person(s)."
        )
    if values.dtype != np.bool_ and not np.isfinite(values.astype(np.float64)).all():
        raise ValueError(f"The engine pre-pass returned nonfinite {label} values.")
    return values


def _ssi_assignment(
    *,
    keys: np.ndarray,
    bands: np.ndarray,
    weights: np.ndarray,
    candidate: np.ndarray,
    reporter: np.ndarray,
    flags: np.ndarray,
    present: np.ndarray,
    targets: Mapping[str, float],
    shares: Mapping[str, float],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """ACS-aligned SSI flags and one diagnostics row per SSA band.

    Stored cells are kept (a stored ``True`` counts toward the floor, a stored
    ``False`` leaves the capacity); a missing reporter takes up; every other
    missing cell draws once against its band prior.
    """

    missing = ~present
    fixed_true = (present & flags) | (missing & reporter)
    open_rows = missing & ~reporter
    assigned = np.where(present, flags, reporter)
    draws = np.ones(len(keys), dtype=np.float64)
    if open_rows.any():
        draws[open_rows] = [
            _stable_source_draw(str(key), seed=int(seed)) for key in keys[open_rows]
        ]
    rows: list[dict[str, Any]] = []
    for definition in US_SSI_TAKE_UP_AGE_TARGETS:
        key = definition.key
        in_band = bands == key
        target = float(targets[key])
        share = float(shares.get(key, 0.0))
        scaled = target * share
        candidates = in_band & candidate
        capacity = float(weights[candidates & (fixed_true | open_rows)].sum())
        floor = float(weights[candidates & fixed_true].sum())
        prior = _band_prior(scaled, capacity, floor)
        drawing = in_band & open_rows
        assigned[drawing] = draws[drawing] < prior
        delivered = float(weights[candidates & assigned].sum())
        rows.append(
            {
                "age_band": key,
                "label": definition.label,
                "enforced": key in US_SSI_TAKE_UP_ENFORCED_BAND_KEYS,
                "national_target": target,
                "acs_weight_share": share,
                "scaled_target": scaled,
                "acs_person_rows": int(in_band.sum()),
                "candidate_rows": int(candidates.sum()),
                "reporter_rows": int((in_band & reporter).sum()),
                "drawn_rows": int(drawing.sum()),
                "candidate_capacity": capacity,
                "reporter_candidate_floor": floor,
                "assignment_prior": float(prior),
                "selected_recipient_weight": delivered,
                "relative_error": (delivered - scaled) / scaled if scaled > 0 else None,
                "saturated": bool(capacity <= scaled),
                "anchor_excess": max(floor - scaled, 0.0),
            }
        )
    return assigned, rows


def _medicaid_assignment(
    *,
    person_ids: np.ndarray,
    keys: np.ndarray,
    state: np.ndarray,
    weights: np.ndarray,
    eligible: np.ndarray,
    anchor: np.ndarray,
    flags: np.ndarray,
    present: np.ndarray,
    state_targets: pd.DataFrame,
    shares: Mapping[str, float],
    seed: int,
    substitutions: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """ACS-aligned Medicaid flags and the donor stage's diagnostics.

    Runs the ``medicaid_take_up`` manifest stage on the ACS rows whose cell is
    missing or stored ``True`` (a stored ``False`` stays out of the stage),
    with ``HINS4 == 1`` or a stored ``True`` as the preserved anchor.
    """

    from microcosm.build.us_runtime import (  # local: package init owns the manifest
        US_SOURCE_MANIFEST,
        us_source_operation_handlers,
    )

    scaled = state_targets.assign(
        acs_weight_share=state_targets["state_fips"].map(
            lambda code: float(shares.get(str(code), 0.0))
        )
    )
    scaled["national_target"] = scaled["target"].astype(np.float64)
    scaled["target"] = scaled["national_target"] * scaled["acs_weight_share"]
    stage_targets = scaled[["state_fips", "target"]].reset_index(drop=True)

    missing = ~present
    fixed_true = (present & flags) | (missing & anchor)
    staged = missing | flags
    table = pd.DataFrame(
        {
            "person_id": person_ids[staged],
            US_MEDICAID_TAKE_UP_ANCHOR: fixed_true[staged],
            US_MEDICAID_ELIGIBILITY_COLUMN: eligible[staged],
            "state_fips": state[staged],
            "person_weight": weights[staged],
        }
    )
    table[_MEDICAID_DRAW_COLUMN] = _stable_person_draws(
        pd.DataFrame({"person_id": keys[staged]}), seed=int(seed)
    )
    table = with_us_medicaid_take_up_rate(table, stage_targets)
    stage = US_SOURCE_MANIFEST.stage_map()[US_MEDICAID_TAKE_UP_STAGE]
    output = run_source_stage(
        stage,
        tables={
            "person": table,
            US_MEDICAID_ENROLLMENT_TARGET_TABLE: stage_targets,
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=int(seed)),
    )
    staged_flags = (
        output.set_index("person_id")[US_MEDICAID_TAKE_UP_VARIABLE]
        .reindex(table["person_id"])
        .to_numpy()
    )
    if pd.isna(staged_flags).any():
        raise ValueError(
            "The medicaid_take_up stage output does not cover every staged ACS person."
        )
    assigned = np.where(present, flags, False)
    fill = np.zeros(len(assigned), dtype=bool)
    fill[staged] = missing[staged]
    assigned[fill] = staged_flags.astype(bool)[missing[staged]]
    diagnostics = us_medicaid_take_up_diagnostics(
        output,
        stage_targets,
        substitutions=[dict(record) for record in substitutions],
        weights_basis=_WEIGHTS_BASIS,
    )
    diagnostics["anchor"] = (
        f"ACS {_MEDICAID_COVERAGE} == {_MEDICAID_COVERED} (Medicaid, Medical "
        "Assistance, or any government-assistance plan for those with low incomes "
        "or a disability)"
    )
    diagnostics["issues"] = [
        *diagnostics.get("issues", []),
        ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE,
    ]
    return assigned, {
        "state_targets": [
            {
                "state_fips": str(row.state_fips),
                "national_target": float(row.national_target),
                "acs_weight_share": float(row.acs_weight_share),
                "scaled_target": float(row.target),
            }
            for row in scaled.itertuples(index=False)
        ],
        "staged_rows": int(staged.sum()),
        "stored_false_rows_kept": int((present & ~flags).sum()),
        "diagnostics": _json_ready(diagnostics),
    }


def _assignment_sha256(
    person_ids: np.ndarray, ssi: np.ndarray, medicaid: np.ndarray
) -> str:
    """Digest of the ACS persons' two take-up flags, to prove two passes agree."""

    table = pd.DataFrame(
        {
            "person_id": np.asarray(person_ids),
            _SSI: np.asarray(ssi, dtype=bool),
            _MEDICAID: np.asarray(medicaid, dtype=bool),
        }
    )
    hashed = pd.util.hash_pandas_object(table, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def with_acs_local_ssi_medicaid_take_up(
    frame: Frame,
    *,
    seed: int,
    ssi_band_targets: Mapping[str, float],
    medicaid_state_targets: pd.DataFrame,
    uncapped_ssi: EngineValues,
    medicaid_eligibility: EngineValues,
    medicaid_substitutions: Sequence[Mapping[str, object]] = (),
) -> tuple[Frame, dict[str, Any]]:
    """Fill missing ACS-row SSI and Medicaid take-up; keep every other cell.

    Args:
        frame: The local lane's multispine US frame, after the income, SSI
            disability-criteria and SNAP/TANF take-up stages. Its person
            table must carry complete origin tags, both take-up columns,
            ``age`` on every row, ``ssi_reported``, ``HINS4`` and ``SPORDER``;
            its household table ``SERIALNO`` and ``state_fips``. ACS persons
            must fill whole households.
        seed: The build seed for the draws.
        ssi_band_targets: The SSA federal-payment recipient counts per age
            band (``under_18``, ``18_64``, ``65_plus``), as the fiscal lane
            reads them from the calibration registry.
        medicaid_state_targets: CMS state Medicaid enrollment counts
            (``state_fips``, ``target``), after reviewed substitutions.
        uncapped_ssi: Returns December 2024 ``uncapped_ssi`` for every person
            of the view it is given (the ACS households, missing take-up
            cells reading ``True``).
        medicaid_eligibility: Returns 2024 ``is_medicaid_eligible`` for every
            person of the view it is given (SSI take-up now assigned).
        medicaid_substitutions: The reviewed CMS substitution records in
            effect, for the donor gate's staleness check.

    Returns:
        The frame with filled ACS cells and a JSON-ready receipt, including a
        digest of the ACS assignment.

    Raises:
        ValueError: If the frame is not US-schema, origin tags or a required
            column are missing, an ACS household holds a non-ACS person, an
            ACS row lacks what its draw or anchor reads, the targets are
            malformed, or an engine callable returns misaligned values.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS local SSI/Medicaid take-up requires the US schema.")
    person = frame.table("person")
    household = frame.table("household")
    tag = spine_column("person")
    if tag not in person or person[tag].isna().any():
        raise ValueError(
            f"ACS local SSI/Medicaid take-up requires complete origin tags: {tag}."
        )
    absent = [
        column
        for column in (
            _SSI,
            _MEDICAID,
            _AGE,
            _REPORTED_SSI,
            _MEDICAID_COVERAGE,
            "SPORDER",
        )
        if column not in person
    ]
    absent += [
        f"household.{column}"
        for column in ("SERIALNO", _STATE)
        if column not in household
    ]
    if absent:
        raise ValueError(
            f"ACS local SSI/Medicaid take-up requires column(s) {absent}; a "
            "missing take-up column would reach the engine as universal take-up."
        )
    targets = _normalize_targets(ssi_band_targets)
    state_targets = _checked_state_targets(medicaid_state_targets)
    acs = person[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    if not acs.any():
        raise ValueError("ACS local SSI/Medicaid take-up found no ACS person rows.")
    household_ids = person["person_household_id"]
    acs_households = pd.unique(household_ids[acs])
    mixed = household_ids.isin(acs_households).to_numpy(dtype=bool) & ~acs
    if mixed.any():
        raise ValueError(
            f"{int(mixed.sum())} non-ACS person(s) share a household with ACS "
            "persons; the engine pre-pass evaluates whole ACS households."
        )
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    age = _numeric(person[_AGE])
    if not np.isfinite(age).all():
        raise ValueError(
            "ACS local SSI/Medicaid take-up needs a finite age on every person: "
            "the ACS band shares read both spines."
        )
    bands = _age_band_values(age)
    state = _normalize_state_fips(frame.broadcast(_STATE).to_numpy())
    rows = person.loc[acs]
    person_ids = rows["person_id"].to_numpy()
    keys = _acs_draw_keys(household, rows)
    acs_age = age[acs]
    acs_weights = weights[acs]
    reporter = _ssi_reporters(rows, acs_age)
    coverage = _numeric(rows[_MEDICAID_COVERAGE])
    anchor = coverage == _MEDICAID_COVERED
    ssi_flags, ssi_present, ssi_invalid = _boolean_cells(rows[_SSI])
    medicaid_flags, medicaid_present, medicaid_invalid = _boolean_cells(rows[_MEDICAID])
    if ssi_invalid.any() or medicaid_invalid.any():
        raise ValueError(
            "Stored ACS SSI/Medicaid take-up cells must be boolean; "
            f"{int(ssi_invalid.sum())} SSI and {int(medicaid_invalid.sum())} "
            "Medicaid cell(s) are not."
        )
    band_shares = _acs_weight_shares(bands, acs, weights)
    state_shares = _acs_weight_shares(state, acs, weights)

    # Engine pre-pass over the ACS households only. ``select`` copied every
    # table, so the view is this stage's own: its missing take-up cells read
    # True, as the engine default would, and neither variable reads its flag.
    view = frame.select(acs)
    view_person = view.table("person")
    if not np.array_equal(view_person["person_id"].to_numpy(), person_ids):
        raise ValueError("The ACS engine view does not preserve the ACS person order.")
    view_person[_SSI] = np.where(ssi_present, ssi_flags, True)
    view_person[_MEDICAID] = np.where(medicaid_present, medicaid_flags, True)
    uncapped = _engine_values(
        uncapped_ssi, view, rows=len(rows), label="uncapped_ssi"
    ).astype(np.float64)
    candidate = uncapped > 0.0
    ssi_assigned, ssi_bands = _ssi_assignment(
        keys=keys,
        bands=bands[acs],
        weights=acs_weights,
        candidate=candidate,
        reporter=reporter,
        flags=ssi_flags,
        present=ssi_present,
        targets=targets,
        shares=band_shares,
        seed=int(seed),
    )
    # SSI receipt is a Medicaid eligibility category: evaluate it with the
    # SSI flags this stage just assigned.
    view_person[_SSI] = ssi_assigned
    eligible = _engine_values(
        medicaid_eligibility, view, rows=len(rows), label="is_medicaid_eligible"
    ).astype(bool)
    del view, view_person
    medicaid_assigned, medicaid = _medicaid_assignment(
        person_ids=person_ids,
        keys=keys,
        state=state[acs],
        weights=acs_weights,
        eligible=eligible,
        anchor=anchor,
        flags=medicaid_flags,
        present=medicaid_present,
        state_targets=state_targets,
        shares=state_shares,
        seed=int(seed),
        substitutions=medicaid_substitutions,
    )

    ssi_missing = np.zeros(len(person), dtype=bool)
    ssi_missing[acs] = ~ssi_present
    medicaid_missing = np.zeros(len(person), dtype=bool)
    medicaid_missing[acs] = ~medicaid_present
    result = frame
    if ssi_missing.any() or medicaid_missing.any():
        full_ssi = np.zeros(len(person), dtype=bool)
        full_ssi[acs] = ssi_assigned
        full_medicaid = np.zeros(len(person), dtype=bool)
        full_medicaid[acs] = medicaid_assigned
        updated = person.copy()
        updated[_SSI] = _filled(person[_SSI], ssi_missing, full_ssi)
        updated[_MEDICAID] = _filled(person[_MEDICAID], medicaid_missing, full_medicaid)
        result = _with_person(frame, updated)

    final = result.table("person").loc[acs]
    final_ssi, final_ssi_present, _ = _boolean_cells(final[_SSI])
    final_medicaid, final_medicaid_present, _ = _boolean_cells(final[_MEDICAID])
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE,
        "method": ACS_LOCAL_SSI_MEDICAID_TAKE_UP_METHOD,
        "spine": ACS_2024_1YR_SPINE,
        "seed": int(seed),
        "draw_key": _DRAW_KEY_FORMAT,
        "columns": list(ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS),
        "acs_persons": int(acs.sum()),
        "acs_households": int(len(acs_households)),
        "weights_basis": _WEIGHTS_BASIS,
        "target_scaling": (
            "each SSA band count and CMS state count times the ACS rows' share "
            "of the frame's person weight in that band or state"
        ),
        "engine_prepass": {
            "rows": "acs_households",
            "missing_take_up_cells_read_as": True,
            "ssi_candidate_definition": ACS_LOCAL_SSI_CANDIDATE_DEFINITION,
            "medicaid_eligibility_definition": (
                ACS_LOCAL_MEDICAID_ELIGIBILITY_DEFINITION
            ),
            "ssi_candidate_rows": int(candidate.sum()),
            "medicaid_eligible_rows": int(eligible.sum()),
        },
        "ssi": {
            "column": _SSI,
            "filled_rows": int((~ssi_present).sum()),
            "preserved_acs_rows": int(ssi_present.sum()),
            "unfilled_acs_rows": int((~final_ssi_present).sum()),
            "reporter_anchor": (
                f"{_REPORTED_SSI} (adjusted ACS SSIP) > 0; SSIP is asked from age "
                f"{_SSIP_MINIMUM_AGE}, so a blank below it reads as no report"
            ),
            "reporter_rows": int(reporter.sum()),
            "relative_tolerance": ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE,
            "enforced_band_keys": list(US_SSI_TAKE_UP_ENFORCED_BAND_KEYS),
            "bands": ssi_bands,
            "weighted_take_up_share": _share(acs_weights, final_ssi),
        },
        "medicaid": {
            "column": _MEDICAID,
            "filled_rows": int((~medicaid_present).sum()),
            "preserved_acs_rows": int(medicaid_present.sum()),
            "unfilled_acs_rows": int((~final_medicaid_present).sum()),
            "anchor": f"ACS {_MEDICAID_COVERAGE} == {_MEDICAID_COVERED}",
            "anchor_rows": int(anchor.sum()),
            "blank_coverage_rows_as_false": int(np.isnan(coverage).sum()),
            "weighted_take_up_share": _share(acs_weights, final_medicaid),
            **medicaid,
        },
        "assigned_sha256": _assignment_sha256(person_ids, final_ssi, final_medicaid),
    }
    return result, receipt


def acs_local_ssi_medicaid_take_up_assignment(frame: Frame) -> pd.DataFrame:
    """The ACS persons' ids and complete take-up flags, in person order.

    Raises:
        ValueError: If origin tags are missing or an ACS cell is missing or
            not boolean.
    """

    person = frame.table("person")
    tag = spine_column("person")
    if tag not in person or person[tag].isna().any():
        raise ValueError(f"Missing person origin tags: {tag}.")
    acs = person[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    rows = person.loc[acs]
    columns: dict[str, np.ndarray] = {"person_id": rows["person_id"].to_numpy()}
    for column in ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS:
        if column not in rows:
            raise ValueError(f"The frame has no {column}.")
        flags, present, invalid = _boolean_cells(rows[column])
        if not present.all() or invalid.any():
            raise ValueError(
                f"{column} is not a complete boolean on the ACS rows; run the "
                "ACS local SSI/Medicaid take-up stage first."
            )
        columns[column] = flags
    return pd.DataFrame(columns)


def with_recorded_acs_local_ssi_medicaid_take_up(
    frame: Frame, recorded: pd.DataFrame, *, assigned_sha256: str
) -> Frame:
    """Apply a recorded ACS assignment and prove it reproduces its digest.

    ``recorded`` is :func:`acs_local_ssi_medicaid_take_up_assignment` of the
    frame the stage ran on. It must cover exactly this frame's ACS persons, in
    order; stored ACS cells must agree with it, missing ones take its values,
    and the resulting ACS assignment must hash to ``assigned_sha256``.

    Raises:
        ValueError: On any mismatch.
    """

    person = frame.table("person")
    tag = spine_column("person")
    if tag not in person or person[tag].isna().any():
        raise ValueError(f"Missing person origin tags: {tag}.")
    acs = person[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    ids = person.loc[acs, "person_id"].to_numpy()
    required = ("person_id", *ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS)
    if (
        any(column not in recorded for column in required)
        or len(recorded) != len(ids)
        or not np.array_equal(recorded["person_id"].to_numpy(), ids)
    ):
        raise ValueError(
            "The recorded ACS SSI/Medicaid take-up does not cover this frame's "
            "ACS persons in order."
        )
    updated = person.copy()
    final: dict[str, np.ndarray] = {}
    for column in ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS:
        values = np.asarray(recorded[column].to_numpy(), dtype=bool)
        if column not in person:
            raise ValueError(f"The frame has no {column}.")
        flags, present, invalid = _boolean_cells(person.loc[acs, column])
        if invalid.any() or (flags[present] != values[present]).any():
            raise ValueError(
                f"Stored ACS {column} cells disagree with the recorded assignment."
            )
        missing = np.zeros(len(person), dtype=bool)
        missing[acs] = ~present
        full = np.zeros(len(person), dtype=bool)
        full[acs] = values
        updated[column] = _filled(person[column], missing, full)
        final[column] = values
    if _assignment_sha256(ids, final[_SSI], final[_MEDICAID]) != assigned_sha256:
        raise ValueError(
            "The recorded ACS SSI/Medicaid take-up does not reproduce its digest."
        )
    return _with_person(frame, updated)


def _grade_ssi_bands(bands: object) -> tuple[list[str], list[dict[str, Any]]]:
    """Grade each recorded SSA band; return failures and per-band statuses.

    Only an enforced band whose prior was count-truthful is held to the
    tolerance: under-18 is fenced as on the donor, a band whose anchors alone
    meet its count keeps them, and a saturated band cannot reach it.
    """

    if not isinstance(bands, Sequence) or [
        entry.get("age_band") if isinstance(entry, Mapping) else None for entry in bands
    ] != list(_BAND_KEYS):
        return [f"receipt: SSI bands must be exactly {list(_BAND_KEYS)}."], []
    tolerance = ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE
    failures: list[str] = []
    graded: list[dict[str, Any]] = []
    for entry in bands:
        key = str(entry["age_band"])
        numbers = [_finite(entry.get(name)) for name in _BAND_NUMBERS]
        if not all(np.isfinite(numbers)):
            failures.append(
                f"receipt: SSI band {key!r} has missing or nonfinite values."
            )
            continue
        scaled, capacity, floor, delivered = numbers
        error = (delivered - scaled) / scaled if scaled > 0 else None
        if key not in US_SSI_TAKE_UP_ENFORCED_BAND_KEYS:
            status = "fenced"
        elif scaled <= 0:
            status = "no_acs_target"
            failures.append(
                f"SSI band {key!r}: the ACS rows carry no weight in an enforced band."
            )
        elif floor >= scaled:
            status = "anchor_excess"
        elif capacity <= scaled:
            status = "saturated"
        elif abs(error) <= tolerance:
            status = "within_tolerance"
        else:
            status = "outside_tolerance"
            failures.append(
                f"SSI band {key!r}: delivered ACS candidate weight {delivered:.0f} "
                f"misses the ACS-scaled SSA count {scaled:.0f} by {error:+.1%}, "
                f"beyond {tolerance:.0%}; the prior was count-truthful there."
            )
        graded.append({"age_band": key, "status": status, "relative_error": error})
    return failures, graded


def _receipt_failures(
    receipt: object, acs_rows: int, details: dict[str, Any]
) -> list[str]:
    """The materialize receipt must show a complete, graded assignment."""

    if (
        not isinstance(receipt, Mapping)
        or receipt.get("issue") != ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE
        or receipt.get("method") != ACS_LOCAL_SSI_MEDICAID_TAKE_UP_METHOD
    ):
        details["receipt"] = {"present": False}
        return [
            "No acs_local_ssi_medicaid_take_up receipt "
            f"({ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE}); the ACS rows' SSI and "
            "Medicaid take-up cannot be shown to be assigned. Re-run --stage "
            "materialize."
        ]
    failures: list[str] = []
    if type(receipt.get("seed")) is not int:
        failures.append("receipt: no integer seed.")
    if receipt.get("acs_persons") != acs_rows:
        failures.append(
            f"receipt: records {receipt.get('acs_persons')!r} ACS person(s) but "
            f"the frame has {acs_rows}."
        )
    digest = receipt.get("assigned_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        failures.append("receipt: no assignment digest.")
    for program, column in (("ssi", _SSI), ("medicaid", _MEDICAID)):
        entry = receipt.get(program)
        if not isinstance(entry, Mapping) or entry.get("column") != column:
            failures.append(f"receipt: no {program} entry for {column}.")
            continue
        if type(entry.get("filled_rows")) is not int:
            failures.append(f"receipt: no {program} filled-row count.")
        if entry.get("unfilled_acs_rows") != 0:
            failures.append(
                f"receipt: {program} left {entry.get('unfilled_acs_rows')!r} ACS "
                "row(s) unfilled."
            )
    ssi = receipt.get("ssi")
    band_failures, graded = _grade_ssi_bands(
        ssi.get("bands") if isinstance(ssi, Mapping) else None
    )
    failures += band_failures
    details["ssi_bands"] = graded
    medicaid = receipt.get("medicaid")
    diagnostics = medicaid.get("diagnostics") if isinstance(medicaid, Mapping) else None
    if not isinstance(diagnostics, Mapping):
        failures.append("receipt: no Medicaid state diagnostics.")
    else:
        try:
            medicaid_gate = us_medicaid_take_up_gate(dict(diagnostics))
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"receipt: malformed Medicaid state diagnostics ({exc!r}).")
        else:
            failures += [f"medicaid: {failure}" for failure in medicaid_gate.failures]
            details["medicaid_saturated_states"] = list(
                diagnostics.get("saturated_states", [])
            )
    details["receipt"] = {"present": True, "failures": len(failures)}
    return failures


def acs_local_ssi_medicaid_take_up_signal_gate(
    frame: Frame, *, receipt: Mapping[str, Any] | None
) -> GateResult:
    """Require complete, anchored, count-faithful SSI and Medicaid take-up.

    Fails when either column is absent or has a missing or non-boolean cell on
    either spine (the reviewed-null fill would make it universal take-up), is
    constant on a spine, or when an ACS ``ssi_reported`` reporter lacks SSI
    take-up or an ACS ``HINS4 == 1`` person lacks Medicaid take-up. It also
    fails unless ``receipt`` shows the materialize assignment with no unfilled
    ACS row, each enforced SSI band (18-64, 65+) within
    :data:`ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE` of its ACS-scaled count
    wherever the prior was count-truthful, and the donor Medicaid gate passing
    on the ACS state diagnostics. The under-18 band, saturated and
    anchor-exceeded SSI bands, saturated Medicaid states and the weighted
    shares are reported, not failed. Donor cells are graded for completeness
    and variation only; the donor release's own gates graded their values.
    """

    person = frame.table("person")
    tag = spine_column("person")
    failures: list[str] = []
    per_spine: dict[str, Any] = {}
    details: dict[str, Any] = {
        "columns": list(ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS),
        "per_spine": per_spine,
        "ssi_relative_tolerance": ACS_LOCAL_SSI_BAND_RELATIVE_TOLERANCE,
    }
    if tag not in person or person[tag].isna().any():
        failures.append(f"Missing person origin tags: {tag}.")
        return GateResult(
            name=ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME,
            passed=False,
            failures=tuple(failures),
            details=details,
        )
    if set(person[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}:
        failures.append(
            "SSI/Medicaid take-up origin tags contain an unsupported spine."
        )
    acs = person[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    absent = [
        column
        for column in ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS
        if column not in person
    ]
    if absent:
        failures.append(
            f"Missing person column(s) {absent}; the engine default is universal "
            "take-up."
        )
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    cells = {
        column: _boolean_cells(person[column])
        for column in ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS
        if column in person
    }
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        selected = person[tag].eq(spine).to_numpy(dtype=bool)
        columns: dict[str, Any] = {}
        per_spine[spine] = {"rows": int(selected.sum()), "columns": columns}
        if not selected.any():
            failures.append(f"{spine}: no person rows.")
            continue
        for column, (flags, present, invalid) in cells.items():
            valid = selected & present & ~invalid
            entry = {
                "missing_rows": int((selected & ~present).sum()),
                "invalid_rows": int((selected & invalid).sum()),
                "unique_values": int(np.unique(flags[valid]).size),
                "weighted_take_up_share": _share(weights[valid], flags[valid]),
            }
            columns[column] = entry
            if entry["missing_rows"]:
                failures.append(
                    f"{spine}: {column} has {entry['missing_rows']} missing row(s); "
                    "the reviewed-null fill would make them universal take-up."
                )
            if entry["invalid_rows"]:
                failures.append(
                    f"{spine}: {column} has {entry['invalid_rows']} non-boolean row(s)."
                )
            if entry["unique_values"] < 2:
                failures.append(
                    f"{spine}: {column} is constant; universal take-up is the "
                    "engine-default landmine."
                )
    anchors: dict[str, Any] = {}
    details["acs_anchors"] = anchors
    for column, source, anchored in (
        (
            _SSI,
            _REPORTED_SSI,
            lambda: np.nan_to_num(_numeric(person[_REPORTED_SSI]), nan=0.0) > 0.0,
        ),
        (
            _MEDICAID,
            _MEDICAID_COVERAGE,
            lambda: _numeric(person[_MEDICAID_COVERAGE]) == _MEDICAID_COVERED,
        ),
    ):
        if column not in cells:
            continue
        if source not in person:
            failures.append(
                f"Missing person column {source}; the ACS {column} anchor cannot "
                "be verified."
            )
            continue
        flags, present, invalid = cells[column]
        reporters = acs & anchored()
        dropped = reporters & present & ~invalid & ~flags
        anchors[column] = {
            "source": source,
            "acs_anchor_rows": int(reporters.sum()),
            "dropped": int(dropped.sum()),
        }
        if dropped.any():
            failures.append(
                f"{ACS_2024_1YR_SPINE}: {int(dropped.sum())} person(s) anchored by "
                f"{source} do not carry {column}; anchors must take up."
            )
    failures += _receipt_failures(receipt, int(acs.sum()), details)
    return GateResult(
        name=ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
