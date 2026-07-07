"""Medicaid take-up: anchored count-calibration to CMS state snapshots.

policyengine-us gates ``medicaid_enrolled`` on ``is_medicaid_eligible`` and the
data-seeded ``takes_up_medicaid_if_eligible`` flag, which defaults to ``True``:
without a dataset value, enrollment mechanically equals eligibility (72.30M ==
72.30M in the published 2024 release — populace #170). Worse, weight
calibration targets ``medicaid_enrolled`` against CMS counts, so an all-True
flag forces the solver to shrink the *eligible* population toward enrollment
counts, biasing weights for everything correlated with Medicaid eligibility.

No administrative participation *rate* clears the take-up contract's
provenance bar (us-data's state rates were MACPAC enrollment over modeled
eligibility — a model-relative ratio), so this stage cites no rate. It follows
the native ACA stage pattern instead (``count_calibrated`` in the contract):

1. **Anchor**: persons reporting Medicaid coverage at interview
   (``has_medicaid_health_coverage_at_interview``, CPS ASEC) always take up.
   The CPS undercounts Medicaid against administrative counts, so the anchor
   is a floor, not the answer.
2. **Fill**: everyone else draws against an in-build state fill rate (CMS
   state enrollment over weighted modeled eligibles — computed transparently
   here, never cited as a sourced rate), then the assignment is greedily
   calibrated to the CMS state enrollment counts among eligible non-anchored
   persons (``calibrate_binary_assignment``), so unsaturated states hit their
   administrative count exactly up to unit-weight granularity.

Enrollment semantics are **point-in-time** (average month): the anchor is
interview-point coverage and the targets are month-tagged CMS snapshots
(``cms_medicaid.month2024_12.state_enrollment.*``), never ever-enrolled-in-year
counts — the Urban HIPSM convention (populace #332). Within-year churn is out
of scope. The anchor and target are different months of the same year (ASEC
reports coverage as of the ~February-April interview; the snapshot is
December): both are point-in-time stocks, and with 2024 enrollment declining
through the unwinding, spring reporters can exceed a December count in some
states — the gate's anchor-floor rule absorbs exactly that direction rather
than reading it as a calibration miss.

Two deliberate asymmetries with the ACA stage:

- **Off-domain persons keep their draw-based propensity** rather than being
  forced ``False``: the assign operation runs without an ``eligibility`` mask,
  so a person outside today's modeled eligibility carries a take-up value at
  the state fill rate. The engine's eligibility gate makes this invisible at
  baseline, but a reform that expands eligibility then enrolls newly eligible
  people at a plausible propensity instead of a silent zero. (Not a behavioral
  model — that is future utility-layer work; this only avoids hard-coding "no
  reform response" into the data.)
- **Saturation is recorded, not failed**: where the CMS count exceeds the
  modeled eligible weight, every eligible person enrolls and the state is
  reported ``saturated`` — an eligibility-undercount symptom (#170's -2.4%
  combined shortfall), distinct from the all-True landmine the gate exists to
  catch.

CHIP is deliberately not assigned here: CMS ``total_chip_enrollment`` mixes
M-CHIP into a concept the model does not materialize (populace #321); CHIP
follows this pattern once the ledger splits the concepts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_runtime import SourceRuntimeConfig, run_source_stage
from populace.build.us_runtime.take_up import (
    _SOURCE_IDENTITY_COLUMNS,
    _stable_unit_draws,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "US_MEDICAID_ENROLLMENT_TARGET_TABLE",
    "US_MEDICAID_ENROLLMENT_TOLERANCE",
    "US_MEDICAID_TAKE_UP_ANCHOR",
    "US_MEDICAID_TAKE_UP_STAGE",
    "US_MEDICAID_TAKE_UP_VARIABLE",
    "us_medicaid_source_person_table",
    "us_medicaid_take_up_diagnostics",
    "us_medicaid_take_up_gate",
    "with_us_medicaid_take_up",
    "with_us_medicaid_take_up_rate",
    "write_us_medicaid_take_up_diagnostics",
]

US_MEDICAID_TAKE_UP_STAGE = "medicaid_take_up"
US_MEDICAID_TAKE_UP_VARIABLE = "takes_up_medicaid_if_eligible"
US_MEDICAID_TAKE_UP_ANCHOR = "has_medicaid_health_coverage_at_interview"
US_MEDICAID_ELIGIBILITY_COLUMN = "is_medicaid_eligible"
US_MEDICAID_ENROLLMENT_TARGET_TABLE = "cms_medicaid_enrollment_by_state"

#: Relative miss tolerance for an unsaturated state's calibrated enrollment
#: against its CMS count. Greedy calibration overshoots by at most one unit
#: weight per state; the gate widens this to the state's largest person
#: weight where that exceeds 2%, so small-state granularity never reads as a
#: real calibration failure.
US_MEDICAID_ENROLLMENT_TOLERANCE = 0.02

_DRAW_COLUMN = "stable_person_draw"
_RATE_COLUMN = "medicaid_take_up_rate"


def _stable_person_draws(person: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Seeded uniform draws keyed by stable source identity per person.

    Delegates to the seeded take-up stages' shared blake2b keying
    (:func:`populace.build.us_runtime.take_up._stable_unit_draws` at person
    grain) so support-channel clones of one source person draw together,
    reruns are bit-reproducible, and a future keying change lands in one
    place.
    """
    return _stable_unit_draws(
        person,
        id_column="person_id",
        seed=seed,
        variable=US_MEDICAID_TAKE_UP_VARIABLE,
    )


def us_medicaid_source_person_table(
    frame: Frame,
    *,
    is_medicaid_eligible: np.ndarray,
    state_fips: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Assemble the person-grain stage table the manifest operations consume.

    Engine-derived columns (eligibility, state) are supplied by the caller —
    the builder owns the simulation, this module owns assembly — and the
    reported-coverage anchor must already ride the person table (a CPS-carried
    column).

    Args:
        frame: A US-schema frame.
        is_medicaid_eligible: Person-aligned boolean eligibility from the
            engine (``is_medicaid_eligible`` materialized at person grain).
        state_fips: Person-aligned state FIPS codes (any numeric/str form;
            normalized to zero-padded two-char strings to match target rows).
        seed: Build-wide imputation seed for the stable draws.

    Raises:
        ValueError: If the frame is not US-schema, the anchor column is
            missing, or the supplied arrays do not align with the person
            table.
    """
    if frame.schema != US_SCHEMA:
        raise ValueError("US Medicaid take-up requires the US schema.")
    person = frame.table("person")
    count = len(person)
    if len(is_medicaid_eligible) != count or len(state_fips) != count:
        raise ValueError(
            "US Medicaid take-up inputs must align with the person table: "
            f"{count} persons, {len(is_medicaid_eligible)} eligibility values, "
            f"{len(state_fips)} state codes."
        )
    if US_MEDICAID_TAKE_UP_ANCHOR not in person.columns:
        raise ValueError(
            f"US Medicaid take-up requires the {US_MEDICAID_TAKE_UP_ANCHOR!r} "
            "reported-coverage column on the person table."
        )

    columns = {
        "person_id": person["person_id"].to_numpy(),
        US_MEDICAID_TAKE_UP_ANCHOR: person[US_MEDICAID_TAKE_UP_ANCHOR]
        .fillna(False)
        .to_numpy(dtype=bool),
        US_MEDICAID_ELIGIBILITY_COLUMN: np.asarray(is_medicaid_eligible, dtype=bool),
        "state_fips": _normalize_state_fips(state_fips),
        "person_weight": np.asarray(
            frame.resolve_weights("person").values, dtype=np.float64
        ),
    }
    for column in _SOURCE_IDENTITY_COLUMNS:
        if column in person.columns:
            columns[column] = person[column].to_numpy()
    table = pd.DataFrame(columns)
    table[_DRAW_COLUMN] = _stable_person_draws(table, seed=seed)
    return table


def _normalize_state_fips(values: np.ndarray) -> np.ndarray:
    """Zero-padded two-char state codes from numeric or string input.

    Deliberately NOT named like the builder's ``_state_fips_text`` (which has
    different semantics — int coercion, bytes handling) so the two cannot be
    confused; missing values are refused rather than silently becoming the
    string ``'nan'`` and surfacing later as a states-without-targets gate
    failure.
    """
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("US Medicaid take-up state_fips contains missing values.")
    text = series.astype(str).str.replace(r"\.0$", "", regex=True)
    return text.str.zfill(2).to_numpy()


def with_us_medicaid_take_up_rate(
    table: pd.DataFrame, state_targets: pd.DataFrame
) -> pd.DataFrame:
    """Derive the in-build state fill rate: CMS count over weighted eligibles.

    This mirrors the ACA stage's rate derivation and is an assignment prior,
    not a sourced participation rate — the calibration operation is what makes
    unsaturated states hit their count. The ratio is computed transparently
    in-build so the contract never records it as provenance (the doctrine's
    model-relative-ratio rule).
    """
    result = table.copy(deep=True)
    if state_targets.empty:
        result[_RATE_COLUMN] = 0.0
        return result
    _require_target_columns(state_targets)
    eligible = result[US_MEDICAID_ELIGIBILITY_COLUMN].fillna(False).astype(bool)
    weighted_eligible = (
        result.loc[eligible].groupby("state_fips")["person_weight"].sum().astype(float)
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
            "US Medicaid enrollment targets require columns "
            f"['state_fips', 'target']; missing {missing}."
        )


def with_us_medicaid_take_up(
    frame: Frame,
    *,
    is_medicaid_eligible: np.ndarray,
    state_fips: np.ndarray,
    state_targets: pd.DataFrame,
    seed: int,
) -> tuple[Frame, dict[str, object]]:
    """Assign ``takes_up_medicaid_if_eligible`` onto the frame's person table.

    Runs the ``medicaid_take_up`` manifest stage (anchor OR draw-under-rate,
    then greedy count-calibration among eligible non-anchored persons) and
    writes the flag back. The column is always recomputed — this stage owns
    it; a persisted landmine is repaired, not trusted.

    Returns:
        The new frame and the release diagnostics payload
        (:func:`us_medicaid_take_up_diagnostics` of the assigned table).

    Raises:
        ValueError: If ``state_targets`` is empty or malformed — an empty
            feed would silently revert every state to anchored-only
            enrollment, so it refuses to run rather than degrade.
    """
    from populace.build.us_runtime import (  # local: package init owns the manifest
        US_SOURCE_MANIFEST,
        us_source_operation_handlers,
    )

    if state_targets.empty:
        raise ValueError(
            "US Medicaid take-up requires non-empty CMS state enrollment "
            "targets; an empty feed would ship anchored-only enrollment."
        )
    _require_target_columns(state_targets)
    table = us_medicaid_source_person_table(
        frame,
        is_medicaid_eligible=is_medicaid_eligible,
        state_fips=state_fips,
        seed=seed,
    )
    table = with_us_medicaid_take_up_rate(table, state_targets)

    stage = US_SOURCE_MANIFEST.stage_map()[US_MEDICAID_TAKE_UP_STAGE]
    output = run_source_stage(
        stage,
        tables={
            "person": table,
            US_MEDICAID_ENROLLMENT_TARGET_TABLE: state_targets,
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=seed),
    )

    person = frame.table("person").copy()
    assigned = (
        output.set_index("person_id")[US_MEDICAID_TAKE_UP_VARIABLE]
        .reindex(person["person_id"])
        .to_numpy()
    )
    if pd.isna(assigned).any():
        raise ValueError(
            "US Medicaid take-up stage output does not cover every person."
        )
    person[US_MEDICAID_TAKE_UP_VARIABLE] = assigned.astype(bool)

    tables = {
        entity: person if entity == "person" else frame.table(entity).copy()
        for entity in frame.entities
    }
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )
    diagnostics = us_medicaid_take_up_diagnostics(output, state_targets)
    return result, diagnostics


def us_medicaid_take_up_diagnostics(
    assigned: pd.DataFrame, state_targets: pd.DataFrame
) -> dict[str, object]:
    """The #170 eligibility-to-enrollment surface, by state and nationally.

    ``assigned`` is the stage output table (person grain, carrying the flag,
    eligibility, anchor, weight, and state). Each state row records eligible
    weight, anchored-eligible weight, modeled enrollment (flag AND eligible —
    the engine's gate), the CMS target, the enrollment/eligibility ratio, and
    whether the state is ``saturated`` (target at or above eligible weight, so
    full enrollment is the calibrated answer, not a landmine).
    """
    eligible = assigned[US_MEDICAID_ELIGIBILITY_COLUMN].fillna(False).astype(bool)
    anchored = assigned[US_MEDICAID_TAKE_UP_ANCHOR].fillna(False).astype(bool)
    takes_up = assigned[US_MEDICAID_TAKE_UP_VARIABLE].fillna(False).astype(bool)
    weights = pd.to_numeric(assigned["person_weight"], errors="coerce").fillna(0.0)
    enrolled = takes_up & eligible
    # The anchor is unmasked by design, so EVERY anchored reporter must carry
    # the flag — a person-level invariant the gate checks even in saturated
    # states, where aggregate comparisons cannot see a dropped anchor.
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
        enrolled_weight = float(weights[index][enrolled[index]].sum())
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
                "enrolled_weight": enrolled_weight,
                "enrollment_to_eligibility_ratio": (
                    enrolled_weight / eligible_weight if eligible_weight > 0 else None
                ),
                "saturated": bool(saturated),
                "anchored_not_taking_up_count": int(anchor_violated[index].sum()),
                # Greedy calibration lands within one person of the count;
                # the gate uses this to size the granularity allowance.
                "max_person_weight": float(weights[index].max()) if len(index) else 0.0,
            }
        )

    total_eligible = float(weights[eligible].sum())
    total_enrolled = float(weights[enrolled].sum())
    return {
        "schema_version": 1,
        "classification": "release_diagnostics",
        "issues": ["populace#331", "populace#170", "populace#332"],
        "variable": US_MEDICAID_TAKE_UP_VARIABLE,
        "anchor": US_MEDICAID_TAKE_UP_ANCHOR,
        "enrollment_semantics": "point_in_time_monthly_snapshot",
        "target_table": US_MEDICAID_ENROLLMENT_TARGET_TABLE,
        # Stage-time surface: weights are the pre-calibration design weights.
        # Post-calibration weighted enrollment is pulled to the same CMS
        # counts by the medicaid_enrollment weight-calibration targets, but
        # eligible/anchored weights here have no post-calibration counterpart.
        "weights_basis": "pre_calibration_design_weights",
        "national": {
            "eligible_weight": total_eligible,
            "enrolled_weight": total_enrolled,
            "enrollment_to_eligibility_ratio": (
                total_enrolled / total_eligible if total_eligible > 0 else None
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


def us_medicaid_take_up_gate(
    diagnostics: dict[str, object],
    *,
    tolerance: float = US_MEDICAID_ENROLLMENT_TOLERANCE,
) -> GateResult:
    """Require a non-degenerate, count-faithful Medicaid take-up surface.

    Fails when:

    - any state has no CMS enrollment target (a feed hole would silently
      revert that state to anchored-only enrollment);
    - any anchored reporter lost the flag — a person-level invariant checked
      in EVERY state, saturated or not (aggregate comparisons cannot see a
      dropped anchor once calibration back-fills to the count);
    - an unsaturated state fully enrolls its eligibles when the reachable
      floor (CMS count or anchor mass, whichever is higher) sits meaningfully
      below eligibility — the #170 landmine. Full enrollment explained by
      anchor mass or by a floor within granularity of eligibility is a
      legitimate calibrated outcome, not a landmine;
    - an unsaturated state misses its floor by more than the granularity
      allowance (``tolerance`` relative, widened to one person weight where
      that is larger — greedy calibration lands within one person).

    Saturated states (CMS count at or above modeled eligible weight) skip the
    count checks: there, full enrollment IS the calibrated answer and the
    shortfall is an eligibility-undercount question outside this stage's
    scope.
    """
    failures: list[str] = []
    states = diagnostics.get("states", [])
    missing = diagnostics.get("states_without_targets", [])
    if missing:
        failures.append(
            f"states without CMS enrollment targets: {missing} — the feed is "
            "incomplete; those states would ship anchored-only enrollment."
        )
    for row in states:
        state = row["state_fips"]
        target = row["target"]
        violations = int(row.get("anchored_not_taking_up_count", 0))
        if violations:
            failures.append(
                f"state {state}: {violations} anchored reporter(s) lost the "
                "flag; the reported-coverage anchor was not preserved."
            )
        if target is None or row["saturated"]:
            continue
        eligible_weight = float(row["eligible_weight"])
        enrolled_weight = float(row["enrolled_weight"])
        anchored_weight = float(row["anchored_eligible_weight"])
        max_person_weight = float(row.get("max_person_weight", 0.0))
        # An unsaturated state's floor is its anchor mass: when reported
        # coverage already exceeds the CMS count, calibration cannot remove
        # anchored persons (by design) and the count is unreachable downward.
        floor = max(float(target), anchored_weight)
        granularity = max(tolerance * eligible_weight, max_person_weight)
        if (
            eligible_weight > 0
            and enrolled_weight >= eligible_weight
            and floor < eligible_weight - granularity
        ):
            failures.append(
                f"state {state}: enrollment equals eligibility while the "
                f"reachable floor {floor:.0f} sits below eligible weight "
                f"{eligible_weight:.0f} — the universal-take-up landmine "
                "(#170)."
            )
            continue
        allowed_miss = max(tolerance * floor, max_person_weight)
        if floor > 0 and abs(enrolled_weight - floor) > allowed_miss:
            failures.append(
                f"state {state}: enrolled weight {enrolled_weight:.0f} misses "
                f"the CMS count {target:.0f} (floor {floor:.0f}) by more than "
                f"the granularity allowance {allowed_miss:.0f}."
            )
    return GateResult(
        name="medicaid_take_up",
        passed=not failures,
        failures=tuple(failures),
        details=diagnostics,
    )


def write_us_medicaid_take_up_diagnostics(
    payload: dict[str, object], path: Path | str
) -> Path:
    """Write the Medicaid take-up diagnostics artifact as strict JSON."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path
