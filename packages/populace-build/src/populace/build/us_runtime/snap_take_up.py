"""SNAP take-up assignment from reported receipt and the FNS participation rate.

Without this stage the published dataset stores no
``takes_up_snap_if_eligible`` (or stores it constant ``True`` — the engine
default), so PolicyEngine-US pays SNAP to 100% of eligible units. USDA FNS
measures participation among the eligible at roughly 82%, so universal
take-up misstates who receives SNAP and overstates the reach of any
eligibility-side reform (populace issue #243).

The stage writes one PolicyEngine-US SPM-unit input column,
``takes_up_snap_if_eligible``, with the same semantics the retired
enhanced-CPS pipeline used:

1. **Reported recipients always take up.** An SPM unit whose raw ASEC
   ``SPM_SNAPSUB`` subsidy is positive reported receiving SNAP; the flag is
   ``True`` for those units unconditionally (survey measurement first).
2. **Non-reporters are drawn to hit the published rate.** Among units with
   no reported receipt, a seeded draw grants take-up at exactly the rate
   needed for the overall weighted take-up share to land on the manifest's
   FNS participation rate. When reporters alone exceed the rate, no
   non-reporter is granted take-up and the share is emergent.

The engine intersects this flag with modeled eligibility, so the flag is
assigned across all units (matching the retired pipeline) rather than only
modeled-eligible ones — eligibility is the rules engine's job, not the
label stage's.

Selection draws are seeded blake2b hashes keyed by the unit's stable source
identity (``source_year`` / ``source_household_id`` / the unit's smallest
``source_person_id`` when present), so support-channel clones of one source
unit always receive the same flag and reruns are bit-reproducible.

The participation rate is data, not code: it lives in the ``snap_take_up``
stage of ``populace/build/us/source_stages.json`` with its FNS citation and
reaches this module as a manifest operation parameter.
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
    "US_SNAP_TAKE_UP_OUTPUT_COLUMN",
    "US_SNAP_TAKE_UP_RAW_COLUMN",
    "US_SNAP_TAKE_UP_STAGE_NAME",
    "derive_us_snap_take_up_from_manifest",
    "us_snap_take_up_signal_gate",
    "us_snap_take_up_summary",
    "us_snap_take_up_stage_spec",
    "with_us_snap_take_up_inputs",
]

US_SNAP_TAKE_UP_STAGE_NAME = "snap_take_up"

#: The PolicyEngine-US SPM-unit input column this stage owns.
US_SNAP_TAKE_UP_OUTPUT_COLUMN = "takes_up_snap_if_eligible"

#: Raw CPS ASEC person column carrying the SPM unit's reported SNAP subsidy.
US_SNAP_TAKE_UP_RAW_COLUMN = "SPM_SNAPSUB"

_PERSON_WEIGHT_COLUMN = "person_weight"
_SPM_MEMBERSHIP_COLUMN = "person_spm_unit_id"

_DERIVE_SNAP_TAKE_UP_PARAMETER_KEYS = frozenset(
    {"take_up_rate", "seed_from_build_config"}
)

#: Weighted take-up share must land in this band. FNS estimates ~82% of
#: eligible persons participate; the share here is over all SPM units (the
#: engine applies eligibility), so the assigned share tracks the manifest
#: rate directly. A share outside the band means the anchor or the draw
#: collapsed — or a constant-True surface (the published landmine).
_TAKE_UP_SHARE_BAND = (0.70, 0.95)


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


def _take_up_rate(operation: SourceOperationSpec) -> float:
    declared = operation.parameters.get("take_up_rate")
    if not isinstance(declared, dict) or "value" not in declared:
        raise SourceRuntimeError(
            "SNAP take-up requires a take_up_rate parameter with a cited value."
        )
    if not str(declared.get("source") or ""):
        raise SourceRuntimeError(
            "SNAP take-up rate requires a source citation in the manifest."
        )
    rate = float(declared["value"])
    if not (0.0 < rate <= 1.0):
        raise SourceRuntimeError(f"SNAP take-up rate must be in (0, 1], got {rate!r}.")
    return rate


def _stable_unit_draws(units: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Seeded uniform draws keyed by stable source identity per SPM unit.

    Support-channel clones share their source identity, so they always
    receive the same draw; frames without source columns key on the SPM
    unit id itself.
    """

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


def derive_us_snap_take_up_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Assign ``takes_up_snap_if_eligible`` at SPM-unit grain.

    The current frame must be the raw-column person table (from the stage's
    ``read_table`` operation) carrying a ``person_weight`` column. Returns
    one row per SPM unit.
    """

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
    required = [US_SNAP_TAKE_UP_RAW_COLUMN, _SPM_MEMBERSHIP_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            f"SNAP take-up derivation requires person column(s): {missing}."
        )
    if _PERSON_WEIGHT_COLUMN not in frame.columns:
        raise SourceRuntimeError(
            "SNAP take-up derivation requires a person_weight column."
        )
    rate = _take_up_rate(operation)

    subsidy = pd.to_numeric(frame[US_SNAP_TAKE_UP_RAW_COLUMN], errors="coerce").fillna(
        0.0
    )
    weight = pd.to_numeric(frame[_PERSON_WEIGHT_COLUMN], errors="coerce").fillna(0.0)
    person = frame.assign(_subsidy=subsidy, _weight=weight)
    aggregates = {
        "_subsidy": ("_subsidy", "max"),
        "_weight": ("_weight", "first"),
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

    reported = units["_subsidy"].to_numpy(dtype=np.float64) > 0.0
    weights = units["_weight"].to_numpy(dtype=np.float64)
    total_weight = float(weights.sum())
    reporter_weight = float(weights[reported].sum())
    non_reporter_weight = float(weights[~reported].sum())
    target_weight = rate * total_weight
    non_reporter_rate = (
        max(0.0, target_weight - reporter_weight) / non_reporter_weight
        if non_reporter_weight > 0.0
        else 0.0
    )
    draws = _stable_unit_draws(units, seed=int(context.config.seed))
    takes_up = reported | (~reported & (draws < non_reporter_rate))

    return pd.DataFrame(
        {
            "spm_unit_id": units[_SPM_MEMBERSHIP_COLUMN].to_numpy(),
            US_SNAP_TAKE_UP_OUTPUT_COLUMN: takes_up,
        }
    )


def _take_up_carries_signal(spm_unit: pd.DataFrame) -> bool:
    """Whether the persisted take-up column is trustworthy as-is.

    A constant column (all ``True`` — the engine default — or all
    ``False``) is the published landmine and must be recomputed.
    """

    values = spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN].dropna()
    return values.nunique() > 1


def with_us_snap_take_up_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
) -> Frame:
    """Run the ``snap_take_up`` manifest stage over a US frame.

    A frame already carrying a non-constant take-up column passes through
    untouched (idempotent). A missing or constant column is recomputed from
    the raw ASEC reported subsidy.

    Args:
        frame: A US-schema frame whose person table still carries the raw
            ``SPM_SNAPSUB`` column (unless the output already carries
            signal).
        seed: Build-wide imputation seed for the non-reporter draws.
        time_period: The dataset's time period.

    Returns:
        A new frame whose spm_unit table carries
        ``takes_up_snap_if_eligible``.

    Raises:
        ValueError: If the frame is not US-schema or the stage output does
            not cover every SPM unit.
        SourceRuntimeError: If required raw columns are missing.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US SNAP take-up inputs require the US schema.")
    spm_unit = frame.table("spm_unit")
    if US_SNAP_TAKE_UP_OUTPUT_COLUMN in spm_unit.columns and _take_up_carries_signal(
        spm_unit
    ):
        return frame

    stage_person = frame.table("person").copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_snap_take_up_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_snap_take_up": derive_us_snap_take_up_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("spm_unit_id").reindex(spm_unit["spm_unit_id"])
    if aligned[US_SNAP_TAKE_UP_OUTPUT_COLUMN].isna().any():
        raise ValueError("US SNAP take-up stage output does not cover every SPM unit.")

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["spm_unit"][US_SNAP_TAKE_UP_OUTPUT_COLUMN] = aligned[
        US_SNAP_TAKE_UP_OUTPUT_COLUMN
    ].to_numpy(dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _reported_by_spm_unit(frame: Frame) -> pd.Series | None:
    person = frame.table("person")
    if US_SNAP_TAKE_UP_RAW_COLUMN not in person.columns:
        return None
    subsidy = pd.to_numeric(person[US_SNAP_TAKE_UP_RAW_COLUMN], errors="coerce").fillna(
        0.0
    )
    return (
        person.assign(_subsidy=subsidy)
        .groupby(_SPM_MEMBERSHIP_COLUMN)["_subsidy"]
        .max()
        .gt(0.0)
    )


def us_snap_take_up_summary(frame: Frame) -> dict[str, object]:
    """Weighted take-up summary for gates and release manifests."""

    spm_unit = frame.table("spm_unit")
    weights = np.asarray(frame.resolve_weights("spm_unit").values, dtype=np.float64)
    takes_up = spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN].to_numpy(dtype=bool)
    total_weight = float(weights.sum())
    take_up_share = (
        float(weights[takes_up].sum()) / total_weight if total_weight > 0 else 0.0
    )
    summary: dict[str, object] = {
        "take_up_share": take_up_share,
        "take_up_share_band": list(_TAKE_UP_SHARE_BAND),
        "unique_count": int(spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN].nunique()),
    }
    reported = _reported_by_spm_unit(frame)
    if reported is not None:
        aligned = reported.reindex(spm_unit["spm_unit_id"]).fillna(False).to_numpy()
        reporter_weight = float(weights[aligned].sum())
        summary["reported_share"] = (
            reporter_weight / total_weight if total_weight > 0 else 0.0
        )
        summary["reporters_not_taking_up"] = int(np.count_nonzero(aligned & ~takes_up))
    return summary


def us_snap_take_up_signal_gate(frame: Frame) -> GateResult:
    """Require a plausible, anchor-respecting take-up surface.

    Fails when the column is missing or constant (the 100%-take-up
    landmine), when any SPM unit that reported SNAP receipt is assigned no
    take-up (the survey anchor must win), or when the weighted take-up
    share leaves the plausibility band around the FNS participation rate.
    """

    spm_unit = frame.table("spm_unit")
    if US_SNAP_TAKE_UP_OUTPUT_COLUMN not in spm_unit.columns:
        return GateResult(
            name="snap_take_up_signal",
            passed=False,
            failures=(f"spm_unit.{US_SNAP_TAKE_UP_OUTPUT_COLUMN}: missing.",),
            details={"missing": [US_SNAP_TAKE_UP_OUTPUT_COLUMN]},
        )

    summary = us_snap_take_up_summary(frame)
    failures: list[str] = []
    if int(summary["unique_count"]) < 2:
        failures.append(
            f"{US_SNAP_TAKE_UP_OUTPUT_COLUMN}: constant column — universal "
            "take-up is the engine-default landmine this stage exists to fix."
        )
    reporters_not_taking_up = summary.get("reporters_not_taking_up")
    if reporters_not_taking_up:
        failures.append(
            f"{reporters_not_taking_up} SPM unit(s) reported SNAP receipt but "
            "carry no take-up; reported recipients must always take up."
        )
    take_up_share = float(summary["take_up_share"])
    low, high = _TAKE_UP_SHARE_BAND
    if not (low <= take_up_share <= high):
        failures.append(
            f"take-up share {take_up_share:.3f} outside plausibility band "
            f"[{low}, {high}]."
        )
    return GateResult(
        name="snap_take_up_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
