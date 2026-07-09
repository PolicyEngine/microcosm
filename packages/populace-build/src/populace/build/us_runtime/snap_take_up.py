"""State-conditional SNAP take-up from reported receipt and FNS rates.

PolicyEngine-US gates SNAP receipt on modeled eligibility and the
``takes_up_snap_if_eligible`` input, which defaults to ``True``. The original
Populace source stage repaired that default by anchoring reported CPS receipt
and topping non-reporters up to one national FNS participation rate. A
national top-up, however, preserves the CPS's uneven state under-reporting and
leaves too little SNAP-recipient support in the states that need the largest
administrative calibration adjustment (populace #372).

This stage instead uses USDA FNS's FY2022 state estimates of the share of
eligible *people* participating in SNAP:

1. SPM units reporting ``SPM_SNAPSUB > 0`` always take up, including units the
   current model does not classify as eligible. The survey anchor is a floor.
2. Within each state, eligible non-reporters are ordered by a stable seeded
   draw. A draw cutoff is chosen so the weighted number of eligible people in
   take-up units is closest to the FNS state rate, subject to the anchor.
3. The same cutoff is applied to currently ineligible non-reporters. They do
   not receive SNAP at baseline, but an eligibility-expanding reform therefore
   exposes a state-specific take-up propensity rather than a hard-coded zero.

The calibration weight is household design weight times modeled
``snap_unit_size`` for currently eligible SPM units. This matches the FNS
estimate's eligible-person denominator while persisting one SPM-unit boolean.
Stable-draw ties are calibrated together, so support-channel clones with the
same source identity always receive the same flag.

Rates and provenance live in the ``snap_take_up`` entry of
``populace/build/us/source_stages.json``. The release gate checks every state
against its cited rate up to source-unit granularity and rejects lost anchors,
missing rate rows, collapsed eligibility, and an unexplained universal-take-up
surface.
"""

from __future__ import annotations

import hashlib
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
    "US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN",
    "US_SNAP_TAKE_UP_OUTPUT_COLUMN",
    "US_SNAP_TAKE_UP_RAW_COLUMN",
    "US_SNAP_TAKE_UP_STAGE_NAME",
    "US_SNAP_TAKE_UP_TOLERANCE",
    "US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN",
    "derive_us_snap_take_up_from_manifest",
    "us_snap_take_up_diagnostics",
    "us_snap_take_up_signal_gate",
    "us_snap_take_up_stage_spec",
    "us_snap_take_up_summary",
    "with_us_snap_take_up_inputs",
]

US_SNAP_TAKE_UP_STAGE_NAME = "snap_take_up"
US_SNAP_TAKE_UP_OUTPUT_COLUMN = "takes_up_snap_if_eligible"
US_SNAP_TAKE_UP_RAW_COLUMN = "SPM_SNAPSUB"
US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN = "is_snap_eligible"
US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN = "snap_unit_size"

#: Relative rate tolerance. The gate widens this to the largest stable-draw
#: cluster in each state, because clone-consistent calibration cannot split a
#: source-identity cluster.
US_SNAP_TAKE_UP_TOLERANCE = 0.02

_PERSON_WEIGHT_COLUMN = "person_weight"
_SPM_MEMBERSHIP_COLUMN = "person_spm_unit_id"
_STATE_COLUMN = "state_fips"
_REPORTED_COLUMN = "reported_snap_receipt"
_ELIGIBLE_PERSON_WEIGHT_COLUMN = "eligible_person_weight"
_DRAW_COLUMN = "stable_spm_unit_draw"
_TARGET_RATE_COLUMN = "snap_take_up_target_rate"

_DERIVE_SNAP_TAKE_UP_PARAMETER_KEYS = frozenset(
    {"state_take_up_rates", "seed_from_build_config"}
)


def us_snap_take_up_stage_spec() -> SourceStageSpec:
    """Load the packaged ``snap_take_up`` source-stage manifest entry."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_SNAP_TAKE_UP_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_SNAP_TAKE_UP_STAGE_NAME!r} stage."
        )
    return stage_map[US_SNAP_TAKE_UP_STAGE_NAME]


def _state_take_up_rate_declaration(
    operation: SourceOperationSpec,
) -> tuple[dict[str, float], dict[str, object]]:
    declared = operation.parameters.get("state_take_up_rates")
    if not isinstance(declared, dict) or not isinstance(declared.get("values"), dict):
        raise SourceRuntimeError(
            "SNAP take-up requires state_take_up_rates with a values mapping."
        )
    source = str(declared.get("source") or "")
    if not source:
        raise SourceRuntimeError(
            "SNAP state take-up rates require a source citation in the manifest."
        )
    fiscal_year = declared.get("fiscal_year")
    if not isinstance(fiscal_year, int) or fiscal_year < 1969:
        raise SourceRuntimeError(
            "SNAP state take-up rates require a valid fiscal_year vintage."
        )
    measure = str(declared.get("measure") or "")
    if not measure:
        raise SourceRuntimeError(
            "SNAP state take-up rates require a declared participation measure."
        )
    rates: dict[str, float] = {}
    for state, raw_rate in declared["values"].items():
        state_fips = str(state).zfill(2)
        if state_fips in rates:
            raise SourceRuntimeError(
                "SNAP state take-up rates contain duplicate normalized state_fips "
                f"{state_fips}."
            )
        rate = float(raw_rate)
        if not (0.0 < rate <= 1.0):
            raise SourceRuntimeError(
                "SNAP state take-up rates must be in (0, 1]; "
                f"state {state_fips} has {rate!r}."
            )
        rates[state_fips] = rate
    if not rates:
        raise SourceRuntimeError("SNAP state take-up rates cannot be empty.")
    metadata = {
        "source": source,
        "fiscal_year": fiscal_year,
        "measure": measure,
    }
    return rates, metadata


def _normalize_state_fips(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    if series.isna().any():
        raise SourceRuntimeError("SNAP take-up state_fips contains missing values.")
    return (
        series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2).to_numpy()
    )


def _stable_unit_draws(units: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Seeded uniform draws keyed by stable source identity per SPM unit."""

    if {"source_year", "source_household_id", "source_person_id"} <= set(units.columns):
        keys = (
            units["source_year"].astype(str)
            + ":"
            + units["source_household_id"].astype(str)
            + ":"
            + units["source_person_id"].astype(str)
        )
    else:
        keys = units[_SPM_MEMBERSHIP_COLUMN].astype(str)
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:snap_take_up:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def _require_unit_constant_columns(
    person: pd.DataFrame, columns: tuple[str, ...]
) -> None:
    grouped = person.groupby(_SPM_MEMBERSHIP_COLUMN, sort=False)
    for column in columns:
        inconsistent = grouped[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            examples = inconsistent[inconsistent].index[:5].tolist()
            raise SourceRuntimeError(
                f"SNAP take-up requires {column!r} to be constant within each "
                f"SPM unit; inconsistent unit ids include {examples}."
            )


def _closest_draw_cutoff(
    draws: np.ndarray,
    weights: np.ndarray,
    *,
    target_weight: float,
) -> float:
    """Choose a clone-preserving draw cutoff closest to ``target_weight``."""

    if target_weight <= 0.0 or len(draws) == 0:
        return -np.inf
    by_draw = (
        pd.DataFrame({_DRAW_COLUMN: draws, "weight": weights})
        .groupby(_DRAW_COLUMN, sort=True)["weight"]
        .sum()
    )
    cumulative = by_draw.cumsum().to_numpy(dtype=np.float64)
    reachable = np.concatenate(([0.0], cumulative))
    misses = np.abs(reachable - target_weight)
    best_miss = float(misses.min())
    # Prefer the higher-weight prefix on an exact tie: this issue exists to
    # restore missing SNAP-carrying support, and both prefixes are equally
    # faithful to the rounded FNS rate.
    best = np.flatnonzero(np.isclose(misses, best_miss, rtol=0.0, atol=1e-12))[-1]
    if best == 0:
        return -np.inf
    return float(by_draw.index[best - 1])


def derive_us_snap_take_up_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Assign state-calibrated ``takes_up_snap_if_eligible`` at SPM grain."""

    if operation.kind != "derive_snap_take_up":
        raise SourceRuntimeError(
            f"SNAP take-up derivation received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "SNAP take-up derivation requires the person table to be read first."
        )
    unexpected = sorted(set(operation.parameters) - _DERIVE_SNAP_TAKE_UP_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            f"SNAP take-up derivation received unsupported parameter(s): {unexpected}."
        )
    required = (
        US_SNAP_TAKE_UP_RAW_COLUMN,
        _SPM_MEMBERSHIP_COLUMN,
        _PERSON_WEIGHT_COLUMN,
        _STATE_COLUMN,
        US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN,
        US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN,
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            f"SNAP take-up derivation requires person column(s): {missing}."
        )
    rates, _ = _state_take_up_rate_declaration(operation)

    subsidy = pd.to_numeric(frame[US_SNAP_TAKE_UP_RAW_COLUMN], errors="coerce").fillna(
        0.0
    )
    weight = pd.to_numeric(frame[_PERSON_WEIGHT_COLUMN], errors="coerce")
    unit_size = pd.to_numeric(frame[US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN], errors="coerce")
    if not np.isfinite(weight.to_numpy(dtype=np.float64)).all() or (weight < 0).any():
        raise SourceRuntimeError("SNAP take-up requires finite nonnegative weights.")
    if (
        not np.isfinite(unit_size.to_numpy(dtype=np.float64)).all()
        or (unit_size < 0).any()
    ):
        raise SourceRuntimeError(
            "SNAP take-up requires finite nonnegative snap_unit_size values."
        )
    person = frame.assign(
        _subsidy=subsidy,
        _weight=weight,
        **{
            _STATE_COLUMN: _normalize_state_fips(frame[_STATE_COLUMN]),
            US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN: unit_size,
        },
    )
    _require_unit_constant_columns(
        person,
        (
            _PERSON_WEIGHT_COLUMN,
            _STATE_COLUMN,
            US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN,
            US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN,
        ),
    )
    aggregates: dict[str, tuple[str, str]] = {
        "_subsidy": ("_subsidy", "max"),
        "_weight": ("_weight", "first"),
        _STATE_COLUMN: (_STATE_COLUMN, "first"),
        US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN: (
            US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN,
            "first",
        ),
        US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN: (
            US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN,
            "first",
        ),
    }
    for column in ("source_year", "source_household_id"):
        if column in person.columns:
            aggregates[column] = (column, "first")
    if "source_person_id" in person.columns:
        aggregates["source_person_id"] = ("source_person_id", "min")
    units = (
        person.groupby(_SPM_MEMBERSHIP_COLUMN, sort=True)
        .agg(**aggregates)
        .reset_index()
    )

    states = units[_STATE_COLUMN].astype(str).to_numpy()
    missing_states = sorted(set(states) - set(rates))
    if missing_states:
        raise SourceRuntimeError(
            "SNAP take-up has no cited FNS participation rate for represented "
            f"state_fips {missing_states}."
        )
    reported = units["_subsidy"].to_numpy(dtype=np.float64) > 0.0
    eligible = (
        units[US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN].fillna(False).to_numpy(dtype=bool)
    )
    household_weights = units["_weight"].to_numpy(dtype=np.float64)
    snap_unit_size = units[US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN].to_numpy(dtype=np.float64)
    eligible_person_weights = household_weights * snap_unit_size * eligible
    target_rates = np.asarray([rates[state] for state in states], dtype=np.float64)
    draws = _stable_unit_draws(units, seed=int(context.config.seed))
    takes_up = reported.copy()

    for state in sorted(set(states)):
        in_state = states == state
        eligible_weight = float(eligible_person_weights[in_state].sum())
        reporter_weight = float(eligible_person_weights[in_state & reported].sum())
        target_weight = rates[state] * eligible_weight
        residual = max(0.0, target_weight - reporter_weight)
        candidates = in_state & eligible & ~reported & (eligible_person_weights > 0)
        cutoff = _closest_draw_cutoff(
            draws[candidates],
            eligible_person_weights[candidates],
            target_weight=residual,
        )
        takes_up[in_state & ~reported] = draws[in_state & ~reported] <= cutoff

    return pd.DataFrame(
        {
            "spm_unit_id": units[_SPM_MEMBERSHIP_COLUMN].to_numpy(),
            _STATE_COLUMN: states,
            _PERSON_WEIGHT_COLUMN: household_weights,
            US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN: eligible,
            US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN: snap_unit_size,
            _ELIGIBLE_PERSON_WEIGHT_COLUMN: eligible_person_weights,
            _REPORTED_COLUMN: reported,
            _DRAW_COLUMN: draws,
            _TARGET_RATE_COLUMN: target_rates,
            US_SNAP_TAKE_UP_OUTPUT_COLUMN: takes_up,
        }
    )


def _align_spm_values_to_person(
    frame: Frame,
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    spm_unit = frame.table("spm_unit")
    if len(values) != len(spm_unit):
        raise ValueError(
            f"US SNAP take-up {name} must align with the spm_unit table: "
            f"{len(spm_unit)} units, {len(values)} values."
        )
    by_id = pd.Series(values, index=spm_unit["spm_unit_id"].to_numpy())
    aligned = by_id.reindex(frame.table("person")[_SPM_MEMBERSHIP_COLUMN]).to_numpy()
    if pd.isna(aligned).any():
        raise ValueError(
            f"US SNAP take-up could not align {name} to every person membership."
        )
    return aligned


def with_us_snap_take_up_inputs(
    frame: Frame,
    *,
    is_snap_eligible: np.ndarray,
    snap_unit_size: np.ndarray,
    state_fips: np.ndarray,
    seed: int,
    time_period: int,
) -> tuple[Frame, dict[str, object]]:
    """Assign state-conditional SNAP take-up and return release diagnostics.

    The three supplied arrays must align with ``frame.table("spm_unit")`` and
    come from the PolicyEngine-US baseline simulation. The owned output column
    is always recomputed; a persisted national assignment or constant engine
    default is not trusted.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US SNAP take-up inputs require the US schema.")
    stage_person = frame.table("person").copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    stage_person[US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN] = _align_spm_values_to_person(
        frame,
        np.asarray(is_snap_eligible, dtype=bool),
        name=US_SNAP_TAKE_UP_ELIGIBILITY_COLUMN,
    )
    stage_person[US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN] = _align_spm_values_to_person(
        frame,
        np.asarray(snap_unit_size, dtype=np.float64),
        name=US_SNAP_TAKE_UP_UNIT_SIZE_COLUMN,
    )
    stage_person[_STATE_COLUMN] = _align_spm_values_to_person(
        frame,
        np.asarray(state_fips),
        name=_STATE_COLUMN,
    )
    spec = us_snap_take_up_stage_spec()
    output = run_source_stage(
        spec,
        tables={"person": stage_person},
        operation_handlers={
            "derive_snap_take_up": derive_us_snap_take_up_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    spm_unit = frame.table("spm_unit")
    aligned = output.set_index("spm_unit_id").reindex(spm_unit["spm_unit_id"])
    if aligned[US_SNAP_TAKE_UP_OUTPUT_COLUMN].isna().any():
        raise ValueError("US SNAP take-up stage output does not cover every SPM unit.")

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["spm_unit"][US_SNAP_TAKE_UP_OUTPUT_COLUMN] = aligned[
        US_SNAP_TAKE_UP_OUTPUT_COLUMN
    ].to_numpy(dtype=bool)
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )
    derive = next(op for op in spec.operations if op.kind == "derive_snap_take_up")
    _, metadata = _state_take_up_rate_declaration(derive)
    return result, us_snap_take_up_diagnostics(output, rate_metadata=metadata)


def us_snap_take_up_diagnostics(
    assigned: pd.DataFrame,
    *,
    rate_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """State and national eligible-person participation diagnostics."""

    required = (
        _STATE_COLUMN,
        _ELIGIBLE_PERSON_WEIGHT_COLUMN,
        _REPORTED_COLUMN,
        _DRAW_COLUMN,
        _TARGET_RATE_COLUMN,
        US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    )
    missing = [column for column in required if column not in assigned.columns]
    if missing:
        raise ValueError(f"SNAP take-up diagnostics require column(s): {missing}.")
    takes_up = assigned[US_SNAP_TAKE_UP_OUTPUT_COLUMN].fillna(False).astype(bool)
    reported = assigned[_REPORTED_COLUMN].fillna(False).astype(bool)
    weights = pd.to_numeric(
        assigned[_ELIGIBLE_PERSON_WEIGHT_COLUMN], errors="coerce"
    ).fillna(0.0)
    participating = takes_up & (weights > 0)
    anchor_violated = reported & ~takes_up

    states: list[dict[str, object]] = []
    for state, group in assigned.groupby(_STATE_COLUMN, sort=True):
        index = group.index
        rates = pd.to_numeric(group[_TARGET_RATE_COLUMN], errors="coerce").dropna()
        if rates.nunique() > 1:
            raise ValueError(
                f"SNAP take-up diagnostics found multiple target rates for state "
                f"{state}."
            )
        target_rate = float(rates.iloc[0]) if not rates.empty else None
        eligible_weight = float(weights[index].sum())
        participating_weight = float(weights[index][participating[index]].sum())
        reported_eligible_weight = float(weights[index][reported[index]].sum())
        selectable = (~reported[index]) & (weights[index] > 0)
        draw_clusters = (
            pd.DataFrame(
                {
                    _DRAW_COLUMN: group.loc[selectable, _DRAW_COLUMN].to_numpy(),
                    "weight": weights[index][selectable].to_numpy(),
                }
            )
            .groupby(_DRAW_COLUMN)["weight"]
            .sum()
        )
        target_weight = (
            target_rate * eligible_weight if target_rate is not None else None
        )
        states.append(
            {
                "state_fips": str(state),
                "target_rate": target_rate,
                "eligible_person_weight": eligible_weight,
                "target_participating_person_weight": target_weight,
                "reported_eligible_person_weight": reported_eligible_weight,
                "participating_person_weight": participating_weight,
                "modeled_participation_rate": (
                    participating_weight / eligible_weight
                    if eligible_weight > 0
                    else None
                ),
                "anchor_floor_exceeds_target": bool(
                    target_weight is not None
                    and reported_eligible_weight > target_weight
                ),
                "reported_not_taking_up_count": int(anchor_violated[index].sum()),
                "max_selection_cluster_weight": (
                    float(draw_clusters.max()) if len(draw_clusters) else 0.0
                ),
            }
        )

    total_eligible = float(weights.sum())
    total_participating = float(weights[participating].sum())
    return {
        "schema_version": 2,
        "classification": "release_diagnostics",
        "issues": ["populace#372", "populace#243"],
        "variable": US_SNAP_TAKE_UP_OUTPUT_COLUMN,
        "anchor": f"{US_SNAP_TAKE_UP_RAW_COLUMN} > 0",
        "rate_semantics": "eligible_person_average_month_participation",
        "weights_basis": "pre_calibration_design_weight_x_snap_unit_size",
        "rate_source": dict(rate_metadata or {}),
        "national": {
            "eligible_person_weight": total_eligible,
            "participating_person_weight": total_participating,
            "modeled_participation_rate": (
                total_participating / total_eligible if total_eligible > 0 else None
            ),
            "reported_not_taking_up_count": int(anchor_violated.sum()),
            "take_up_unique_count": int(takes_up.nunique()),
        },
        "states": states,
        "states_without_rates": sorted(
            row["state_fips"] for row in states if row["target_rate"] is None
        ),
    }


def us_snap_take_up_summary(frame: Frame) -> dict[str, object]:
    """Backward-compatible output-column summary.

    State-rate fidelity requires engine eligibility inputs and therefore lives
    in :func:`us_snap_take_up_diagnostics`; this lightweight frame-only view is
    retained for callers that inspect persisted take-up signal and anchors.
    """
    spm_unit = frame.table("spm_unit")
    if US_SNAP_TAKE_UP_OUTPUT_COLUMN not in spm_unit:
        return {"missing": [US_SNAP_TAKE_UP_OUTPUT_COLUMN]}
    weights = np.asarray(frame.resolve_weights("spm_unit").values, dtype=np.float64)
    takes_up = spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN].to_numpy(dtype=bool)
    total_weight = float(weights.sum())
    summary: dict[str, object] = {
        "take_up_share": (
            float(weights[takes_up].sum()) / total_weight if total_weight > 0 else 0.0
        ),
        "unique_count": int(spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN].nunique()),
    }
    person = frame.table("person")
    if US_SNAP_TAKE_UP_RAW_COLUMN not in person:
        return summary
    subsidy = pd.to_numeric(person[US_SNAP_TAKE_UP_RAW_COLUMN], errors="coerce").fillna(
        0.0
    )
    reported = (
        person.assign(_subsidy=subsidy)
        .groupby(_SPM_MEMBERSHIP_COLUMN)["_subsidy"]
        .max()
        .gt(0.0)
        .reindex(spm_unit["spm_unit_id"])
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    summary["reported_share"] = (
        float(weights[reported].sum()) / total_weight if total_weight > 0 else 0.0
    )
    summary["reporters_not_taking_up"] = int(np.count_nonzero(reported & ~takes_up))
    return summary


def us_snap_take_up_signal_gate(
    diagnostics: dict[str, object],
    *,
    tolerance: float = US_SNAP_TAKE_UP_TOLERANCE,
) -> GateResult:
    """Require state-rate-faithful, anchor-preserving SNAP take-up."""

    failures: list[str] = []
    missing = diagnostics.get("states_without_rates", [])
    if missing:
        failures.append(
            f"states without cited FNS participation rates: {missing}; the "
            "state-conditional feed is incomplete."
        )
    unexplained_universal = False
    for row in diagnostics.get("states", []):
        state = row["state_fips"]
        violations = int(row.get("reported_not_taking_up_count", 0))
        if violations:
            failures.append(
                f"state {state}: {violations} reported SNAP recipient unit(s) "
                "lost the take-up flag; the survey anchor was not preserved."
            )
        target_rate = row.get("target_rate")
        eligible_weight = float(row["eligible_person_weight"])
        if target_rate is not None and float(target_rate) > 0 and eligible_weight <= 0:
            failures.append(
                f"state {state}: positive FNS participation rate with zero "
                "modeled-eligible person weight; the eligibility surface collapsed."
            )
            continue
        if target_rate is None:
            continue
        target_weight = float(row["target_participating_person_weight"])
        anchor_weight = float(row["reported_eligible_person_weight"])
        participating_weight = float(row["participating_person_weight"])
        floor = max(target_weight, anchor_weight)
        cluster_weight = float(row.get("max_selection_cluster_weight", 0.0))
        allowance = max(tolerance * max(floor, eligible_weight), cluster_weight)
        if abs(participating_weight - floor) > allowance:
            failures.append(
                f"state {state}: participating eligible-person weight "
                f"{participating_weight:.0f} misses the FNS/anchor floor "
                f"{floor:.0f} by more than the granularity allowance "
                f"{allowance:.0f}."
            )
        if floor < eligible_weight - allowance:
            unexplained_universal = True

    national = diagnostics.get("national", {})
    if int(national.get("take_up_unique_count", 0)) < 2 and unexplained_universal:
        failures.append(
            f"{US_SNAP_TAKE_UP_OUTPUT_COLUMN}: constant column despite at least "
            "one state target materially below full participation — the "
            "universal-take-up landmine remains."
        )
    return GateResult(
        name="snap_take_up_signal",
        passed=not failures,
        failures=tuple(failures),
        details=diagnostics,
    )
