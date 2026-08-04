"""SNAP take-up: anchored count-calibration to FNS state caseloads.

The ``snap_take_up`` stage (populace #243) anchors take-up on reported ASEC
receipt and fills non-reporters to the *national* FNS participation rate.
CPS SNAP receipt underreporting is strongly state-dependent, so a national
fill leaves the states with the worst underreporting short of taker
households no reweighting can recover: in the buildi-sparse release the
calibration undershoots ten state benefit targets by 7-43% while a
per-record feasibility audit shows every state but California is reachable
under the weight cap, and the state taker-household counts err from -38% to
+325% against FNS caseloads (populace #372).

This stage recalibrates the fill per state, following the Medicaid pattern
(populace #331) at SPM-unit grain:

1. **Anchor**: an SPM unit whose raw ASEC ``SPM_SNAPSUB`` subsidy is
   positive reported receiving SNAP and always takes up (survey
   measurement first) — the same anchor the national stage enforces.
2. **Fill**: non-reporters draw against an in-build state fill rate (the
   FNS state household caseload over weighted modeled-eligible units —
   computed transparently here, never cited as a sourced rate), then the
   assignment is greedily calibrated to the FNS state average-monthly
   household counts among eligible non-anchored units
   (``calibrate_binary_assignment``), so unsaturated states hit their
   administrative caseload up to unit-weight granularity.

Caseload semantics are **fiscal-year average monthly** stocks: the targets
are the FNS ``average_monthly_households`` facts (the same rows the
``snap_households`` calibration targets compile from), and the taker set is
the model counterpart of the average monthly caseload — the take-up flag
carries no within-year churn. Persons targets are deliberately absent: the
real SNAP assistance unit is often a subset of the SPM unit, so a person
count over taker-unit members overcounts FNS participants by roughly half.

Two deliberate carry-overs from the Medicaid stage:

- **Off-domain units keep their draw-based propensity** rather than being
  forced ``False``: the assign operation runs without an ``eligibility``
  mask, so a unit outside today's modeled eligibility carries a take-up
  value at the state fill rate. The engine's eligibility gate makes this
  invisible at baseline, but an eligibility-expanding reform then enrolls
  newly eligible units at a plausible propensity instead of a silent zero.
- **Saturation is recorded, not failed**: where the FNS count meets or
  exceeds the modeled eligible weight, every eligible unit takes up and the
  state is reported ``saturated`` — an eligibility-undercount symptom
  (California's feasibility ceiling of 1.05x its caseload), distinct from
  the all-True landmine the gates exist to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_runtime import SourceRuntimeConfig, run_source_stage
from populace.build.us_runtime.snap_take_up import (
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    US_SNAP_TAKE_UP_RAW_COLUMN,
)
from populace.build.us_runtime.take_up import (
    _stable_unit_draws,
    _units_with_source_identity,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "US_SNAP_CASELOAD_TOLERANCE",
    "US_SNAP_HOUSEHOLDS_TARGET_TABLE",
    "US_SNAP_STATE_TAKE_UP_ANCHOR",
    "US_SNAP_STATE_TAKE_UP_STAGE",
    "us_snap_source_spm_unit_table",
    "us_snap_state_take_up_diagnostics",
    "us_snap_state_take_up_gate",
    "with_us_snap_state_take_up",
    "with_us_snap_state_take_up_rate",
    "write_us_snap_state_take_up_diagnostics",
]

US_SNAP_STATE_TAKE_UP_STAGE = "snap_state_take_up"
US_SNAP_STATE_TAKE_UP_ANCHOR = "snap_reported_receipt"
US_SNAP_ELIGIBILITY_COLUMN = "is_snap_eligible"
US_SNAP_HOUSEHOLDS_TARGET_TABLE = "usda_snap_households_by_state"

#: Relative miss tolerance for an unsaturated state's calibrated caseload
#: against its FNS count. Greedy calibration overshoots by at most one unit
#: weight per state; the gate widens this to the state's largest unit weight
#: where that exceeds 2%, so small-state granularity never reads as a real
#: calibration failure.
US_SNAP_CASELOAD_TOLERANCE = 0.02

_DRAW_COLUMN = "stable_spm_unit_draw"
_RATE_COLUMN = "snap_take_up_rate"
_WEIGHT_COLUMN = "spm_unit_weight"


def us_snap_source_spm_unit_table(
    frame: Frame,
    *,
    is_snap_eligible: np.ndarray,
    state_fips: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Assemble the SPM-unit-grain stage table the manifest operations consume.

    Engine-derived columns (eligibility, state) are supplied by the caller —
    the builder owns the simulation, this module owns assembly. The
    reported-receipt anchor is derived here from the person table's raw ASEC
    ``SPM_SNAPSUB`` column, matching the national ``snap_take_up`` stage's
    anchor exactly.

    Args:
        frame: A US-schema frame whose person table still carries the raw
            ``SPM_SNAPSUB`` column.
        is_snap_eligible: SPM-unit-aligned boolean eligibility from the
            engine (``is_snap_eligible`` at spm_unit grain).
        state_fips: SPM-unit-aligned state FIPS codes (any numeric/str form;
            normalized to zero-padded two-char strings to match target rows).
        seed: Build-wide imputation seed for the stable draws.

    Raises:
        ValueError: If the frame is not US-schema, the raw anchor column is
            missing, or the supplied arrays do not align with the spm_unit
            table.
    """
    if frame.schema != US_SCHEMA:
        raise ValueError("US SNAP state take-up requires the US schema.")
    spm_unit = frame.table("spm_unit")
    count = len(spm_unit)
    if len(is_snap_eligible) != count or len(state_fips) != count:
        raise ValueError(
            "US SNAP state take-up inputs must align with the spm_unit table: "
            f"{count} units, {len(is_snap_eligible)} eligibility values, "
            f"{len(state_fips)} state codes."
        )
    person = frame.table("person")
    if US_SNAP_TAKE_UP_RAW_COLUMN not in person.columns:
        raise ValueError(
            f"US SNAP state take-up requires the raw {US_SNAP_TAKE_UP_RAW_COLUMN!r} "
            "reported-subsidy column on the person table."
        )

    table = _units_with_source_identity(frame, "spm_unit").rename(
        columns={"_weight": _WEIGHT_COLUMN}
    )
    subsidy = pd.to_numeric(person[US_SNAP_TAKE_UP_RAW_COLUMN], errors="coerce").fillna(
        0.0
    )
    reported = (
        person.assign(_subsidy=subsidy)
        .groupby("person_spm_unit_id")["_subsidy"]
        .max()
        .gt(0.0)
    )
    table[US_SNAP_STATE_TAKE_UP_ANCHOR] = (
        reported.reindex(table["spm_unit_id"]).fillna(False).to_numpy(dtype=bool)
    )
    table[US_SNAP_ELIGIBILITY_COLUMN] = np.asarray(is_snap_eligible, dtype=bool)
    table["state_fips"] = _normalize_state_fips(state_fips)
    table[_DRAW_COLUMN] = _stable_unit_draws(
        table,
        id_column="spm_unit_id",
        seed=seed,
        variable=US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    )
    return table


def _normalize_state_fips(values: np.ndarray) -> np.ndarray:
    """Zero-padded two-char state codes from numeric or string input.

    Missing values are refused rather than silently becoming the string
    ``'nan'`` and surfacing later as a states-without-targets gate failure.
    """
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("US SNAP state take-up state_fips contains missing values.")
    text = series.astype(str).str.replace(r"\.0$", "", regex=True)
    return text.str.zfill(2).to_numpy()


def with_us_snap_state_take_up_rate(
    table: pd.DataFrame, state_targets: pd.DataFrame
) -> pd.DataFrame:
    """Derive the in-build state fill rate: FNS caseload over weighted eligibles.

    This mirrors the Medicaid stage's rate derivation and is an assignment
    prior, not a sourced participation rate — the calibration operation is
    what makes unsaturated states hit their count. The ratio is computed
    transparently in-build so the contract never records it as provenance
    (the doctrine's model-relative-ratio rule).
    """
    result = table.copy(deep=True)
    if state_targets.empty:
        result[_RATE_COLUMN] = 0.0
        return result
    _require_target_columns(state_targets)
    eligible = result[US_SNAP_ELIGIBILITY_COLUMN].fillna(False).astype(bool)
    weighted_eligible = (
        result.loc[eligible].groupby("state_fips")[_WEIGHT_COLUMN].sum().astype(float)
    )
    targets = state_targets.groupby("state_fips")["target"].sum().astype(float)
    rates = (targets / weighted_eligible).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result[_RATE_COLUMN] = (
        result["state_fips"].map(rates).fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    return result


def _require_target_columns(state_targets: pd.DataFrame) -> None:
    missing = [
        column for column in ("state_fips", "target") if column not in state_targets
    ]
    if missing:
        raise ValueError(
            "US SNAP household caseload targets require columns "
            f"['state_fips', 'target']; missing {missing}."
        )


def with_us_snap_state_take_up(
    frame: Frame,
    *,
    is_snap_eligible: np.ndarray,
    state_fips: np.ndarray,
    state_targets: pd.DataFrame,
    seed: int,
) -> tuple[Frame, dict[str, object]]:
    """Assign ``takes_up_snap_if_eligible`` onto the frame's spm_unit table.

    Runs the ``snap_state_take_up`` manifest stage (anchor OR draw-under-rate,
    then greedy count-calibration among eligible non-anchored units) and
    writes the flag back. The column is always recomputed — this stage owns
    the final surface; the national ``snap_take_up`` fill it replaces is the
    assignment prior, not the answer.

    Returns:
        The new frame and the release diagnostics payload
        (:func:`us_snap_state_take_up_diagnostics` of the assigned table).

    Raises:
        ValueError: If ``state_targets`` is empty or malformed — an empty
            feed would silently revert every state to the national fill, so
            it refuses to run rather than degrade.
    """
    from populace.build.us_runtime import (  # local: package init owns the manifest
        US_SOURCE_MANIFEST,
        us_source_operation_handlers,
    )

    if state_targets.empty:
        raise ValueError(
            "US SNAP state take-up requires non-empty FNS state household "
            "caseload targets; an empty feed would ship the national fill."
        )
    _require_target_columns(state_targets)
    table = us_snap_source_spm_unit_table(
        frame,
        is_snap_eligible=is_snap_eligible,
        state_fips=state_fips,
        seed=seed,
    )
    table = with_us_snap_state_take_up_rate(table, state_targets)

    stage = US_SOURCE_MANIFEST.stage_map()[US_SNAP_STATE_TAKE_UP_STAGE]
    output = run_source_stage(
        stage,
        tables={
            "spm_unit": table,
            US_SNAP_HOUSEHOLDS_TARGET_TABLE: state_targets,
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=seed),
    )

    spm_unit = frame.table("spm_unit").copy()
    assigned = (
        output.set_index("spm_unit_id")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        .reindex(spm_unit["spm_unit_id"])
        .to_numpy()
    )
    if pd.isna(assigned).any():
        raise ValueError(
            "US SNAP state take-up stage output does not cover every SPM unit."
        )
    spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN] = assigned.astype(bool)

    tables = {
        entity: spm_unit if entity == "spm_unit" else frame.table(entity).copy()
        for entity in frame.entities
    }
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    diagnostics = us_snap_state_take_up_diagnostics(output, state_targets)
    return result, diagnostics


def us_snap_state_take_up_diagnostics(
    assigned: pd.DataFrame, state_targets: pd.DataFrame
) -> dict[str, object]:
    """The #372 caseload surface, by state and nationally.

    ``assigned`` is the stage output table (spm_unit grain, carrying the
    flag, eligibility, anchor, weight, and state). Each state row records
    eligible weight, anchored weight, the modeled caseload (flag AND
    eligible — the engine's gate), the FNS target, the caseload/eligibility
    ratio, and whether the state is ``saturated`` (target at or above
    eligible weight, so full take-up is the calibrated answer, not a
    landmine).
    """
    eligible = assigned[US_SNAP_ELIGIBILITY_COLUMN].fillna(False).astype(bool)
    anchored = assigned[US_SNAP_STATE_TAKE_UP_ANCHOR].fillna(False).astype(bool)
    takes_up = assigned[US_SNAP_TAKE_UP_OUTPUT_COLUMN].fillna(False).astype(bool)
    weights = pd.to_numeric(assigned[_WEIGHT_COLUMN], errors="coerce").fillna(0.0)
    caseload = takes_up & eligible
    # The anchor is unmasked by design, so EVERY reported recipient must
    # carry the flag — a unit-level invariant the gate checks even in
    # saturated states, where aggregate comparisons cannot see a dropped
    # anchor.
    anchor_violated = anchored & ~takes_up

    targets_by_state: dict[str, float] = {}
    if not state_targets.empty:
        _require_target_columns(state_targets)
        targets_by_state = (
            state_targets.groupby("state_fips")["target"].sum().astype(float).to_dict()
        )

    states = []
    for state, group in assigned.groupby("state_fips", sort=True):
        index = group.index
        eligible_weight = float(weights[index][eligible[index]].sum())
        caseload_weight = float(weights[index][caseload[index]].sum())
        anchored_eligible_weight = float(
            weights[index][(anchored & eligible)[index]].sum()
        )
        target = targets_by_state.get(str(state))
        saturated = target is not None and target >= eligible_weight
        states.append(
            {
                "state_fips": str(state),
                "target": target,
                "eligible_weight": eligible_weight,
                "anchored_eligible_weight": anchored_eligible_weight,
                "caseload_weight": caseload_weight,
                "caseload_to_eligibility_ratio": (
                    caseload_weight / eligible_weight if eligible_weight > 0 else None
                ),
                "saturated": bool(saturated),
                "anchored_not_taking_up_count": int(anchor_violated[index].sum()),
                # Greedy calibration lands within one unit of the count; the
                # gate uses this to size the granularity allowance.
                "max_unit_weight": float(weights[index].max()) if len(index) else 0.0,
            }
        )

    total_eligible = float(weights[eligible].sum())
    total_caseload = float(weights[caseload].sum())
    return {
        "schema_version": 1,
        "classification": "release_diagnostics",
        "issues": ["populace#372", "populace#243"],
        "variable": US_SNAP_TAKE_UP_OUTPUT_COLUMN,
        "anchor": US_SNAP_STATE_TAKE_UP_ANCHOR,
        "caseload_semantics": "fiscal_year_average_monthly_households",
        "target_table": US_SNAP_HOUSEHOLDS_TARGET_TABLE,
        # Stage-time surface: weights are the pre-calibration design weights.
        # Post-calibration weighted caseload is pulled to the same FNS counts
        # by the snap_households weight-calibration targets, but eligible/
        # anchored weights here have no post-calibration counterpart.
        "weights_basis": "pre_calibration_design_weights",
        "national": {
            "eligible_weight": total_eligible,
            "caseload_weight": total_caseload,
            "caseload_to_eligibility_ratio": (
                total_caseload / total_eligible if total_eligible > 0 else None
            ),
            "target_total": float(sum(targets_by_state.values()))
            if targets_by_state
            else None,
            "anchored_not_taking_up_count": int(anchor_violated.sum()),
        },
        "states": states,
        "states_without_targets": sorted(
            {str(s) for s in assigned["state_fips"].unique()} - set(targets_by_state)
        ),
        "saturated_states": sorted(
            row["state_fips"] for row in states if row["saturated"]
        ),
    }


def us_snap_state_take_up_gate(
    diagnostics: dict[str, object],
    *,
    tolerance: float = US_SNAP_CASELOAD_TOLERANCE,
) -> GateResult:
    """Require a non-degenerate, count-faithful SNAP take-up surface.

    Fails when:

    - any state has no FNS household caseload target (a feed hole would
      silently revert that state to the national fill);
    - any state carries a positive FNS count with zero modeled-eligible
      weight — an eligibility-feed collapse. Such a state classifies as
      ``saturated`` (the count trivially exceeds zero eligibility), so
      without this check it would skip the count checks and ship silently
      wrong;
    - any reported recipient lost the flag — a unit-level invariant checked
      in EVERY state, saturated or not;
    - an unsaturated state fully enrolls its eligibles when the reachable
      floor (FNS count or anchor mass, whichever is higher) sits meaningfully
      below eligibility — the universal-take-up landmine. Full take-up
      explained by anchor mass or by a floor within granularity of
      eligibility is a legitimate calibrated outcome, not a landmine;
    - an unsaturated state misses its floor by more than the granularity
      allowance (``tolerance`` relative, widened to one unit weight where
      that is larger — greedy calibration lands within one unit).

    Saturated states (FNS count at or above modeled eligible weight) skip
    the count checks: there, full take-up IS the calibrated answer and the
    shortfall is an eligibility-undercount question outside this stage's
    scope.
    """
    failures: list[str] = []
    states = diagnostics.get("states", [])
    missing = diagnostics.get("states_without_targets", [])
    if missing:
        failures.append(
            f"states without FNS household caseload targets: {missing} — the "
            "feed is incomplete; those states would ship the national fill."
        )
    for row in states:
        state = row["state_fips"]
        target = row["target"]
        violations = int(row.get("anchored_not_taking_up_count", 0))
        if violations:
            failures.append(
                f"state {state}: {violations} reported recipient(s) lost the "
                "flag; the reported-receipt anchor was not preserved."
            )
        if target is not None and target > 0 and float(row["eligible_weight"]) <= 0:
            failures.append(
                f"state {state}: FNS count {target:.0f} with zero "
                "modeled-eligible weight — the eligibility feed collapsed; "
                "its saturated classification is a division artifact, not a "
                "calibrated answer."
            )
            continue
        if target is None or row["saturated"]:
            continue
        eligible_weight = float(row["eligible_weight"])
        caseload_weight = float(row["caseload_weight"])
        anchored_weight = float(row["anchored_eligible_weight"])
        max_unit_weight = float(row.get("max_unit_weight", 0.0))
        # An unsaturated state's floor is its anchor mass: when reported
        # receipt already exceeds the FNS count, calibration cannot remove
        # anchored units (by design) and the count is unreachable downward.
        floor = max(float(target), anchored_weight)
        granularity = max(tolerance * eligible_weight, max_unit_weight)
        if (
            eligible_weight > 0
            and caseload_weight >= eligible_weight
            and floor < eligible_weight - granularity
        ):
            failures.append(
                f"state {state}: caseload equals eligibility while the "
                f"reachable floor {floor:.0f} sits below eligible weight "
                f"{eligible_weight:.0f} — the universal-take-up landmine."
            )
            continue
        allowed_miss = max(tolerance * floor, max_unit_weight)
        if floor > 0 and abs(caseload_weight - floor) > allowed_miss:
            failures.append(
                f"state {state}: caseload weight {caseload_weight:.0f} misses "
                f"the FNS count {target:.0f} (floor {floor:.0f}) by more than "
                f"the granularity allowance {allowed_miss:.0f}."
            )
    return GateResult(
        name="snap_state_take_up",
        passed=not failures,
        failures=tuple(failures),
        details=diagnostics,
    )


def write_us_snap_state_take_up_diagnostics(
    payload: dict[str, object], path: Path | str
) -> Path:
    """Write the SNAP state take-up diagnostics artifact as strict JSON."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path
