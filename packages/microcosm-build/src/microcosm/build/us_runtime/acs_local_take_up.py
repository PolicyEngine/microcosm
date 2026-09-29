"""SNAP and TANF take-up for the ACS rows of the retained ACS local lane.

The ACS transfer leaves take-up draws to the runtime, and the local lane runs
neither ``snap_take_up`` nor ``snap_state_take_up``. Every ACS SPM unit
therefore reached the engine pass with ``takes_up_snap_if_eligible`` and
``takes_up_tanf_if_eligible`` missing, and the reviewed-null fill gave them
the engine default, ``True``: every eligible ACS unit took both programs up
(microcosm#1019, the take-up counterpart of the #765 hours defect).

This is a fresh-build repair for that lane. It fills only missing cells on
ACS-spine SPM units; donor-spine values and any stored ACS value are kept.

- SNAP follows :mod:`~microcosm.build.us_runtime.snap_take_up`: a unit with
  (QRF-imputed) ``receives_snap`` always takes up, and non-reporters draw at
  the rate that puts the weighted ACS take-up share on the national FNS
  participation rate of the ``snap_take_up`` manifest stage.
- TANF uses the take-up contract's seeded Bernoulli draw at its
  administrative rate, with no receipt anchor, exactly as the contract seeds
  it elsewhere.

Draws come from :func:`~microcosm.build.us_runtime.take_up._stable_unit_draws`,
which keys a unit without complete source identity on its own ids, so the
assignment depends only on ``seed``, the frame and its weights. The
per-state recalibration of the ASEC lane is not applied; the release's state
SNAP household targets reweight instead. Other ``takes_up_*`` flags on ACS
rows still ship at the engine default (microcosm#1022) and the gate here does
not grade them.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.snap_take_up import (
    _TAKE_UP_SHARE_BAND as _SNAP_TAKE_UP_SHARE_BAND,
)
from microcosm.build.us_runtime.snap_take_up import (
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    _take_up_rate,
    us_snap_take_up_stage_spec,
)
from microcosm.build.us_runtime.take_up import (
    US_TAKE_UP_SHARE_BAND,
    _seed_program,
    _stable_unit_draws,
    _units_with_source_identity,
)
from microcosm.build.us_runtime.take_up_contract import (
    TakeUpProgram,
    seeded_take_up_programs,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_TAKE_UP_COLUMNS",
    "ACS_LOCAL_TAKE_UP_GATE_NAME",
    "US_TANF_TAKE_UP_OUTPUT_COLUMN",
    "acs_local_take_up_signal_gate",
    "with_acs_local_take_up_inputs",
]

US_TANF_TAKE_UP_OUTPUT_COLUMN = "takes_up_tanf_if_eligible"

#: The SPM-unit take-up flags this stage owns on ACS rows.
ACS_LOCAL_TAKE_UP_COLUMNS: tuple[str, ...] = (
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    US_TANF_TAKE_UP_OUTPUT_COLUMN,
)
ACS_LOCAL_TAKE_UP_GATE_NAME = "acs_local_take_up_signal"

_ID_COLUMN = "spm_unit_id"
#: Reported (on ACS rows, QRF-transferred) SNAP receipt: the survey anchor.
_REPORTED_SNAP_COLUMN = "receives_snap"
_SNAP_OPERATION = "derive_snap_take_up"
_SHARE_BANDS: dict[str, tuple[float, float]] = {
    US_SNAP_TAKE_UP_OUTPUT_COLUMN: _SNAP_TAKE_UP_SHARE_BAND,
    US_TANF_TAKE_UP_OUTPUT_COLUMN: US_TAKE_UP_SHARE_BAND[US_TANF_TAKE_UP_OUTPUT_COLUMN],
}


def _flags(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Boolean values with missing cells as ``False``, and the present mask."""
    present = values.notna().to_numpy(dtype=bool)
    flags = np.zeros(len(values), dtype=bool)
    flags[present] = values[present].astype(bool).to_numpy(dtype=bool)
    return flags, present


def _snap_rate() -> tuple[float, str]:
    """The national rate and citation of the ``snap_take_up`` manifest stage."""
    stage = us_snap_take_up_stage_spec()
    operations = [op for op in stage.operations if op.kind == _SNAP_OPERATION]
    if len(operations) != 1:
        raise ValueError(
            f"The {stage.stage!r} stage must declare exactly one {_SNAP_OPERATION!r} "
            f"operation; found {len(operations)}."
        )
    rate = _take_up_rate(operations[0])
    return rate, str(operations[0].parameters["take_up_rate"]["source"])


def _tanf_program() -> TakeUpProgram:
    for program in seeded_take_up_programs():
        if program.variable == US_TANF_TAKE_UP_OUTPUT_COLUMN:
            return program
    raise ValueError(
        f"The take-up contract does not seed {US_TANF_TAKE_UP_OUTPUT_COLUMN!r}."
    )


def _filled(values: pd.Series, missing: np.ndarray, assigned: np.ndarray) -> pd.Series:
    if not missing.any():
        return values
    filled = values.astype(object)
    filled.loc[missing] = assigned[missing]
    return filled.astype(bool) if filled.notna().all() else filled


def _share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def _assignment_sha256(spm_unit: pd.DataFrame, rows: np.ndarray) -> str:
    """Digest of the ACS rows' ids and flags, to prove two passes agree."""
    selected = pd.DataFrame(
        {
            _ID_COLUMN: spm_unit.loc[rows, _ID_COLUMN].to_numpy(),
            **{
                column: _flags(spm_unit.loc[rows, column])[0]
                for column in ACS_LOCAL_TAKE_UP_COLUMNS
            },
        }
    )
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def with_acs_local_take_up_inputs(
    frame: Frame, *, seed: int
) -> tuple[Frame, dict[str, object]]:
    """Fill missing ACS-row SNAP and TANF take-up flags; keep every other cell.

    Args:
        frame: The local lane's multispine US frame. Its spm_unit table must
            carry complete origin tags, both take-up columns and
            ``receives_snap``.
        seed: The build seed for the draws.

    Returns:
        The frame with filled ACS cells (unchanged if none were missing) and
        a JSON-ready receipt, including a digest of the ACS assignment.

    Raises:
        ValueError: If the frame is not US-schema, origin tags are missing,
            or a required column is absent.
    """
    if frame.schema != US_SCHEMA:
        raise ValueError("ACS local take-up inputs require the US schema.")
    spm_unit = frame.table("spm_unit")
    tag = spine_column("spm_unit")
    if tag not in spm_unit or spm_unit[tag].isna().any():
        raise ValueError(f"ACS local take-up requires complete origin tags: {tag}.")
    absent = [
        column
        for column in (*ACS_LOCAL_TAKE_UP_COLUMNS, _REPORTED_SNAP_COLUMN)
        if column not in spm_unit
    ]
    if absent:
        raise ValueError(
            f"ACS local take-up requires spm_unit column(s) {absent}; a missing "
            "donor column would reach the engine as universal take-up."
        )
    acs = spm_unit[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    units = _units_with_source_identity(frame, "spm_unit")
    if not np.array_equal(units[_ID_COLUMN].to_numpy(), spm_unit[_ID_COLUMN]):
        raise ValueError("SPM-unit source identity does not align with the table.")
    weights = units["_weight"].to_numpy(dtype=np.float64)
    acs_weight = float(weights[acs].sum())

    # SNAP: reporters take up; non-reporters draw to the national rate.
    snap_flags, snap_present = _flags(spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN])
    snap_missing = acs & ~snap_present
    reported, reported_present = _flags(spm_unit[_REPORTED_SNAP_COLUMN])
    anchored = snap_missing & reported
    open_units = snap_missing & ~reported
    rate, rate_source = _snap_rate()
    # Weight the non-reporter draws must add for the ACS share to reach the
    # rate; when reporters and stored cells already exceed it, no one draws.
    shortfall = (
        rate * acs_weight
        - float(weights[acs & snap_present & snap_flags].sum())
        - float(weights[anchored].sum())
    )
    open_weight = float(weights[open_units].sum())
    draw_rate = (
        min(1.0, max(0.0, shortfall) / open_weight) if open_weight > 0.0 else 0.0
    )
    snap_assigned = anchored.copy()
    if open_units.any():
        draws = _stable_unit_draws(
            units.loc[open_units],
            id_column=_ID_COLUMN,
            seed=int(seed),
            variable=US_SNAP_TAKE_UP_OUTPUT_COLUMN,
        )
        snap_assigned[open_units] = draws < draw_rate

    # TANF: the contract's seeded Bernoulli draw, no receipt anchor.
    program = _tanf_program()
    seeded, _ = _seed_program(frame, program, seed=int(seed))
    if not np.array_equal(seeded[_ID_COLUMN], spm_unit[_ID_COLUMN].to_numpy()):
        raise ValueError("Seeded TANF take-up does not align with the spm_unit table.")
    tanf_assigned = np.asarray(seeded[US_TANF_TAKE_UP_OUTPUT_COLUMN], dtype=bool)
    tanf_missing = acs & ~_flags(spm_unit[US_TANF_TAKE_UP_OUTPUT_COLUMN])[1]

    result = frame
    if snap_missing.any() or tanf_missing.any():
        updated = spm_unit.copy()
        updated[US_SNAP_TAKE_UP_OUTPUT_COLUMN] = _filled(
            spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN], snap_missing, snap_assigned
        )
        updated[US_TANF_TAKE_UP_OUTPUT_COLUMN] = _filled(
            spm_unit[US_TANF_TAKE_UP_OUTPUT_COLUMN], tanf_missing, tanf_assigned
        )
        result = Frame(
            {
                entity: updated if entity == "spm_unit" else frame.table(entity)
                for entity in frame.entities
            },
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )
    final = result.table("spm_unit")
    acs_weights = weights[acs]
    receipt: dict[str, object] = {
        "issue": "microcosm#1019",
        "seed": int(seed),
        "spine": ACS_2024_1YR_SPINE,
        "acs_spm_units": int(acs.sum()),
        "programs": {
            US_SNAP_TAKE_UP_OUTPUT_COLUMN: {
                "rate": rate,
                "rate_source": rate_source,
                "filled_rows": int(snap_missing.sum()),
                "preserved_rows": int((acs & snap_present).sum()),
                "reported_receipt_rows": int(anchored.sum()),
                "missing_receipt_rows_as_non_reporters": int(
                    (snap_missing & ~reported_present).sum()
                ),
                "non_reporter_draw_rate": draw_rate,
                "weighted_take_up_share": _share(
                    acs_weights,
                    _flags(final.loc[acs, US_SNAP_TAKE_UP_OUTPUT_COLUMN])[0],
                ),
            },
            US_TANF_TAKE_UP_OUTPUT_COLUMN: {
                "rate": program.rate.get("value"),
                "rate_source": program.rate.get("source"),
                "filled_rows": int(tanf_missing.sum()),
                "preserved_rows": int((acs & ~tanf_missing).sum()),
                "weighted_take_up_share": _share(
                    acs_weights,
                    _flags(final.loc[acs, US_TANF_TAKE_UP_OUTPUT_COLUMN])[0],
                ),
            },
        },
        "assigned_sha256": _assignment_sha256(final, acs),
    }
    return result, receipt


def acs_local_take_up_signal_gate(frame: Frame) -> GateResult:
    """Require complete, anchored, plausible SNAP and TANF take-up per origin.

    For each spine, fails when a flag is missing, has missing cells, or is
    constant (the engine-default universal take-up this lane shipped). On the
    ACS spine, whose cells this stage assigns, it also fails when a share
    leaves its plausibility band or (SNAP) a ``receives_snap`` reporter does
    not take up. Donor-spine shares and anchors are reported but not graded:
    those cells come from the donor release, whose own take-up gates graded
    them. Other ``takes_up_*`` columns are not graded (microcosm#1022).
    """
    spm_unit = frame.table("spm_unit")
    tag = spine_column("spm_unit")
    failures: list[str] = []
    by_spine: dict[str, object] = {}
    if tag not in spm_unit or spm_unit[tag].isna().any():
        failures.append(f"Missing SPM-unit origin tags: {tag}.")
    else:
        weights = np.asarray(frame.resolve_weights("spm_unit").values, dtype=float)
        for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
            selected = spm_unit[tag].eq(spine).to_numpy(dtype=bool)
            # Only the ACS cells are this stage's output; grade their
            # plausibility and anchor. Donor cells are graded upstream.
            graded = spine == ACS_2024_1YR_SPINE
            columns: dict[str, object] = {}
            by_spine[spine] = {
                "rows": int(selected.sum()),
                "graded": graded,
                "columns": columns,
            }
            if not selected.any():
                failures.append(f"{spine}: no spm_unit rows.")
                continue
            for column, (low, high) in _SHARE_BANDS.items():
                if column not in spm_unit:
                    failures.append(f"{spine}: missing {column}.")
                    continue
                flags, present = _flags(spm_unit.loc[selected, column])
                share = _share(weights[selected], flags)
                unique = len(np.unique(flags[present]))
                entry: dict[str, object] = {
                    "missing_rows": int((~present).sum()),
                    "unique_values": unique,
                    "take_up_share": share,
                    "share_band": [low, high],
                }
                columns[column] = entry
                if not present.all():
                    failures.append(
                        f"{spine}: {column} has missing rows; an engine-default "
                        "fill is universal take-up."
                    )
                if unique < 2:
                    failures.append(
                        f"{spine}: {column} is constant; universal take-up is the "
                        "engine-default landmine."
                    )
                if graded and not (low <= share <= high):
                    failures.append(
                        f"{spine}: {column} take-up share {share:.3f} outside "
                        f"plausibility band [{low}, {high}]."
                    )
                if column != US_SNAP_TAKE_UP_OUTPUT_COLUMN:
                    continue
                if _REPORTED_SNAP_COLUMN not in spm_unit:
                    failures.append(
                        f"{spine}: missing {_REPORTED_SNAP_COLUMN}; the reported-"
                        "receipt anchor cannot be verified."
                    )
                    continue
                reported, _ = _flags(spm_unit.loc[selected, _REPORTED_SNAP_COLUMN])
                not_taking_up = int((reported & present & ~flags).sum())
                entry["reporters_not_taking_up"] = not_taking_up
                if graded and not_taking_up:
                    failures.append(
                        f"{spine}: {not_taking_up} SPM unit(s) report SNAP receipt "
                        "but carry no take-up; reported recipients must take up."
                    )
        unknown = set(spm_unit[tag].unique()) - {
            ASEC_PUF_DONOR_SPINE,
            ACS_2024_1YR_SPINE,
        }
        if unknown:
            failures.append("Local take-up origin tags contain an unsupported spine.")
    return GateResult(
        name=ACS_LOCAL_TAKE_UP_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details={"per_spine": by_spine},
    )
