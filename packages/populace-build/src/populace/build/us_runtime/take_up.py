"""Seed take-up flags for data-seeded programs from administrative rates.

policyengine-us gates several benefit programs on a ``takes_up_*`` input flag
that defaults to ``True``. When the dataset provides nothing the engine pays the
program to 100% of eligible units; the administrative participation rate is well
below 100% for most programs, so universal take-up overstates who receives the
benefit and the reach of any eligibility-side reform (populace #312).

This module seeds the flags whose rate clears the provenance bar. The take-up
contract inventory (:mod:`populace.build.us_runtime.take_up_contract`) records,
per program, whether populace seeds it and the administrative source of its
rate; only ``seed`` programs are assigned here, and only from a rate the
manifest cites (a rate without a source refuses to run — the rule ledger#77
established after fabricated SNAP values shipped).

Two programs clear the bar today, both assigned by **calibrated Bernoulli** at
the administrative rate (no survey-reported-receipt column is threaded through
the populace base spine, so the reported-receipt anchor the SNAP stage uses does
not apply without base-pool work):

- **TANF** (``takes_up_tanf_if_eligible``, SPM-unit): a single rate, 21.9%
  (HHS ASPE 24th Welfare Indicators Report, 2022). The ``tanf`` formula
  multiplies summed state benefits by the flag, exactly like SNAP.
- **EITC** (``takes_up_eitc``, tax-unit): a rate that varies by number of
  qualifying children (IRS National Taxpayer Advocate 2020, TY2016:
  0=65%, 1=86%, 2=85%, 3+=82%). The child count at stage time is approximated
  by tax-unit members under the EITC qualifying-child age (19); the rate is
  nearly flat above zero children, so the coarse count picks the right rate bin
  for all but edge cases (documented in the stage notes).

Draws are seeded blake2b hashes keyed by the assigned unit's stable source
identity, so support-channel clones of one source unit always receive the same
flag and reruns are bit-reproducible — the same keying the SNAP stage uses.

SNAP is seeded by its own source stage (``snap_take_up``, PR #294) and is not
touched here; ACA take-up is owned by the ACA workstream and its own source
stage. The remaining take-up flags are ``rate_unsourced`` in the inventory and
deliberately left unseeded until a rate is sourced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.us_runtime.take_up_contract import (
    TakeUpProgram,
    load_take_up_contract,
    seeded_take_up_programs,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "US_TAKE_UP_SHARE_BAND",
    "SeededTakeUpResult",
    "us_take_up_participation_diagnostics",
    "us_take_up_signal_gate",
    "us_take_up_summary",
    "with_us_take_up_inputs",
    "write_us_take_up_participation_diagnostics",
]

#: The EITC qualifying-child age test (26 USC § 32(c)(3)(A) / § 152(c)(3)):
#: under 19, or under 24 and a student. Only ``age`` is available at stage time,
#: so the student extension is not applied; the rate is nearly flat above zero
#: children, so the count only needs to distinguish 0 / 1 / 2 / 3+.
_EITC_QUALIFYING_CHILD_MAX_AGE = 19

#: Per-program weighted take-up share plausibility band. A share outside this
#: band means the draw or the rate collapsed, or the flag is constant at the
#: engine default (the universal-take-up landmine). The band is wide because the
#: assigned share is over all units of the flag's entity (the engine applies
#: eligibility), and TANF's rate (~0.22) and EITC's (~0.78) sit far apart.
US_TAKE_UP_SHARE_BAND: dict[str, tuple[float, float]] = {
    "takes_up_tanf_if_eligible": (0.10, 0.40),
    "takes_up_eitc": (0.60, 0.92),
}

_PERSON_AGE_COLUMN = "age"
_SOURCE_IDENTITY_COLUMNS = ("source_year", "source_household_id", "source_person_id")


@dataclass(frozen=True)
class SeededTakeUpResult:
    """The outcome of seeding one program's take-up flag onto a frame."""

    variable: str
    entity: str
    take_up_share: float
    target_rate: float
    unique_count: int


def _entity_id_column(entity: str) -> str:
    return f"{entity}_id"


def _person_membership_column(entity: str) -> str:
    return f"person_{entity}_id"


def _stable_unit_draws(
    units: pd.DataFrame, *, id_column: str, seed: int, variable: str
) -> np.ndarray:
    """Seeded uniform draws keyed by stable source identity per unit.

    When the frame carries source-identity columns (a unit's origin ASEC year,
    household, and smallest person id) the draw keys on them, so support-channel
    clones of one source unit — which share that identity — always draw
    together. Frames without source columns key on the entity id itself.
    """
    if set(_SOURCE_IDENTITY_COLUMNS) <= set(units.columns):
        keys = (
            units["source_year"].astype(str)
            + ":"
            + units["source_household_id"].astype(str)
            + ":"
            + units["source_person_id"].astype(str)
        )
    else:
        keys = units[id_column].astype(str)
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{variable}:{key}".encode(),
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


def _units_with_source_identity(frame: Frame, entity: str) -> pd.DataFrame:
    """One row per unit of ``entity`` carrying its resolved weight and, when
    available, the stable source identity aggregated from its members.

    Source identity lives on the person table; a group unit inherits its
    members' (year, household, min person id) so clones stay consistent. The
    person↔unit attachment is read from the frame's declared linkage, never
    re-derived from a flat frame.
    """
    id_column = _entity_id_column(entity)
    table = frame.table(entity)
    units = pd.DataFrame({id_column: table[id_column].to_numpy()})
    units["_weight"] = np.asarray(
        frame.resolve_weights(entity).values, dtype=np.float64
    )

    person = frame.table("person")
    membership = _person_membership_column(entity)
    if membership in person.columns and set(_SOURCE_IDENTITY_COLUMNS) <= set(
        person.columns
    ):
        aggregates = {
            "source_year": ("source_year", "first"),
            "source_household_id": ("source_household_id", "first"),
            "source_person_id": ("source_person_id", "min"),
        }
        identity = person.groupby(membership, sort=True).agg(**aggregates).reset_index()
        units = units.merge(
            identity, left_on=id_column, right_on=membership, how="left"
        )
        if membership != id_column:
            units = units.drop(columns=[membership])
    return units


def _eitc_child_count_by_tax_unit(frame: Frame) -> pd.Series:
    """Approximate EITC-qualifying-child count per tax unit from member ages.

    Counts tax-unit members under the qualifying-child age. Exact
    qualifying-child status (student-under-24, relationship, residency,
    disability) is engine logic not available at stage time; the count only
    needs to place a unit in the 0 / 1 / 2 / 3+ rate bin, and the rate is nearly
    flat above zero children.
    """
    person = frame.table("person")
    membership = _person_membership_column("tax_unit")
    ages = pd.to_numeric(person[_PERSON_AGE_COLUMN], errors="coerce")
    is_child = (ages < _EITC_QUALIFYING_CHILD_MAX_AGE).fillna(False)
    counts = (
        person.assign(_is_child=is_child.astype(int))
        .groupby(membership, sort=True)["_is_child"]
        .sum()
    )
    return counts


def _eitc_rate_by_unit(program: TakeUpProgram, frame: Frame) -> pd.Series:
    """Per-tax-unit EITC take-up rate from the child-count rate mapping."""
    rates = program.rate.get("values_by_num_children")
    if not isinstance(rates, dict):
        raise ValueError(
            "EITC take-up requires a values_by_num_children rate mapping in the "
            "take-up contract."
        )
    zero = float(rates["0"])
    one = float(rates["1"])
    two = float(rates["2"])
    three_plus = float(rates.get("3+", rates.get("3")))
    id_column = _entity_id_column("tax_unit")
    tax_units = frame.table("tax_unit")[id_column]
    counts = _eitc_child_count_by_tax_unit(frame).reindex(tax_units).fillna(0)
    mapping = {0: zero, 1: one, 2: two}
    rate = counts.map(lambda c: mapping.get(int(c), three_plus))
    rate.index = tax_units.to_numpy()
    return rate.astype(float)


def _seed_program(
    frame: Frame, program: TakeUpProgram, *, seed: int
) -> tuple[dict[str, np.ndarray], SeededTakeUpResult]:
    """Compute the take-up flag column for one program.

    Returns the new column keyed by entity id and a summary. Reported-receipt
    anchoring is not applied (no reported column is threaded through the base
    spine); assignment is calibrated Bernoulli at the administrative rate,
    which for EITC varies by number of children.
    """
    entity = program.entity
    id_column = _entity_id_column(entity)
    units = _units_with_source_identity(frame, entity)
    draws = _stable_unit_draws(
        units, id_column=id_column, seed=seed, variable=program.variable
    )

    if program.variable == "takes_up_eitc":
        rate_by_unit = _eitc_rate_by_unit(program, frame)
        rates = (
            rate_by_unit.reindex(units[id_column].to_numpy())
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        target_rate = float(rate_by_unit.mean()) if len(rate_by_unit) else 0.0
    else:
        rate_value = program.rate.get("value")
        if rate_value is None:
            raise ValueError(
                f"{program.variable} take-up requires a scalar rate value in the "
                "take-up contract."
            )
        rates = np.full(len(units), float(rate_value), dtype=np.float64)
        target_rate = float(rate_value)

    takes_up = draws < rates
    weights = units["_weight"].to_numpy(dtype=np.float64)
    total_weight = float(weights.sum())
    take_up_share = (
        float(weights[takes_up].sum()) / total_weight if total_weight > 0 else 0.0
    )
    column = {id_column: units[id_column].to_numpy(), program.variable: takes_up}
    result = SeededTakeUpResult(
        variable=program.variable,
        entity=entity,
        take_up_share=take_up_share,
        target_rate=target_rate,
        unique_count=int(np.unique(takes_up).size),
    )
    return column, result


def _column_carries_signal(table: pd.DataFrame, variable: str) -> bool:
    """Whether a persisted take-up column is trustworthy as-is.

    A constant column (all True — the engine default — or all False) is the
    published landmine and must be recomputed.
    """
    values = table[variable].dropna()
    return values.nunique() > 1


def with_us_take_up_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    programs: tuple[TakeUpProgram, ...] | None = None,
) -> Frame:
    """Seed every ``seed`` take-up flag from the contract onto a US frame.

    For each seeded program, assigns its ``takes_up_*`` flag across all units of
    the flag's entity at the administrative rate (per-child for EITC). A frame
    already carrying a non-constant column for a program is left untouched for
    that program (idempotent); a missing or constant column is (re)computed —
    the published all-True landmine is repaired, not trusted.

    Args:
        frame: A US-schema frame carrying ``age`` on the person table and the
            standard entity linkage.
        seed: Build-wide imputation seed for the draws.
        time_period: The dataset's time period (unused today; the rates are
            fixed vintages recorded in the contract, kept for signature parity
            with the other source stages and future year-varying rates).
        programs: Override the seeded program set (tests); defaults to the
            contract's seeded programs.

    Returns:
        A new frame whose entity tables carry the seeded take-up columns.

    Raises:
        ValueError: If the frame is not US-schema.
    """
    if frame.schema != US_SCHEMA:
        raise ValueError("US take-up inputs require the US schema.")
    seeded = programs if programs is not None else seeded_take_up_programs()

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    changed = False
    for program in seeded:
        entity = program.entity
        table = tables[entity]
        if program.variable in table.columns and _column_carries_signal(
            table, program.variable
        ):
            continue
        column, _ = _seed_program(frame, program, seed=seed)
        id_column = _entity_id_column(entity)
        assigned = (
            pd.DataFrame(column)
            .set_index(id_column)[program.variable]
            .reindex(table[id_column])
        )
        if assigned.isna().any():
            raise ValueError(
                f"US take-up stage output for {program.variable} does not cover "
                f"every {entity}."
            )
        table[program.variable] = assigned.to_numpy(dtype=bool)
        changed = True

    if not changed:
        return frame
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def us_take_up_summary(
    frame: Frame, programs: tuple[TakeUpProgram, ...] | None = None
) -> dict[str, dict[str, object]]:
    """Per-program weighted take-up summary for gates and release manifests.

    Reports, for each seeded program present on the frame, the weighted take-up
    share, the administrative target rate, the plausibility band, and the
    administrative source — the participation-vs-administrative surface the
    release diagnostics record so the class cannot regress silently.
    """
    seeded = programs if programs is not None else seeded_take_up_programs()
    summary: dict[str, dict[str, object]] = {}
    for program in seeded:
        entity = program.entity
        table = frame.table(entity)
        if program.variable not in table.columns:
            continue
        weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
        takes_up = table[program.variable].to_numpy(dtype=bool)
        total_weight = float(weights.sum())
        participation_count = float(weights[takes_up].sum())
        take_up_share = participation_count / total_weight if total_weight > 0 else 0.0
        rate = program.rate
        if program.variable == "takes_up_eitc":
            target = rate.get("values_by_num_children")
        else:
            target = rate.get("value")
        band = US_TAKE_UP_SHARE_BAND.get(program.variable)
        summary[program.variable] = {
            "entity": entity,
            "take_up_share": take_up_share,
            "weighted_participation_count": participation_count,
            "weighted_flag_universe": total_weight,
            "administrative_rate": target,
            "administrative_source": rate.get("source"),
            "administrative_agency": rate.get("agency"),
            "share_band": list(band) if band is not None else None,
            "unique_count": int(table[program.variable].nunique()),
        }
    return summary


def us_take_up_signal_gate(
    frame: Frame, programs: tuple[TakeUpProgram, ...] | None = None
) -> GateResult:
    """Require a plausible, non-constant take-up surface for every seeded flag.

    Fails when a seeded program's column is missing or constant (the
    universal-take-up landmine this class exists to fix) or when its weighted
    take-up share leaves the plausibility band around the administrative rate.
    """
    seeded = programs if programs is not None else seeded_take_up_programs()
    summary = us_take_up_summary(frame, seeded)
    failures: list[str] = []
    for program in seeded:
        variable = program.variable
        entity = program.entity
        table = frame.table(entity)
        if variable not in table.columns:
            failures.append(f"{variable}: missing seeded take-up column.")
            continue
        info = summary[variable]
        if int(info["unique_count"]) < 2:
            failures.append(
                f"{variable}: constant column — universal take-up is the "
                "engine-default landmine this stage exists to fix."
            )
        band = US_TAKE_UP_SHARE_BAND.get(variable)
        if band is not None:
            share = float(info["take_up_share"])
            low, high = band
            if not (low <= share <= high):
                failures.append(
                    f"{variable}: take-up share {share:.3f} outside plausibility "
                    f"band [{low}, {high}]."
                )
    return GateResult(
        name="take_up_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def us_take_up_participation_diagnostics(
    frame: Frame,
) -> dict[str, object]:
    """Return the standalone US take-up participation diagnostics payload.

    Records, per take-up flag, populace's weighted participation against the
    administrative rate the seed reproduces — the participation-vs-administrative
    surface #170 and #312 exist to expose. Every flag in the contract appears,
    not only the two seeded ones: for a seeded program the row carries the live
    weighted participation count, the weighted flag universe, the reproduced
    take-up share, and the administrative rate/source; for an unseeded, out-of-
    scope, or near-universal program the row records the curated treatment and
    the reason it carries no seeded participation surface (the honest record that
    the remaining flags still ship at the engine's universal-take-up default, so
    a reader is not misled into thinking the class is fully repaired).

    Standalone by design: it is written next to ``us_source_coverage.json`` and
    is not threaded through the release manifests, whose gate-assembly is a
    sibling workstream's zone (populace #313).
    """
    contract = load_take_up_contract()
    gate = us_take_up_signal_gate(frame)
    live = us_take_up_summary(frame)
    programs: list[dict[str, object]] = []
    for program in contract.programs:
        variable = program.variable
        rate = program.rate
        row: dict[str, object] = {
            "variable": variable,
            "entity": program.entity,
            "engine_class": program.engine_class,
            "populace_treatment": program.populace_treatment,
            "engine_default": program.default,
            "administrative_rate_status": rate.get("status"),
            "administrative_source": rate.get("source"),
            "administrative_agency": rate.get("agency"),
        }
        if program.is_seeded and variable in live:
            info = live[variable]
            row["seeded"] = True
            row["weighted_participation_count"] = info["weighted_participation_count"]
            row["weighted_flag_universe"] = info["weighted_flag_universe"]
            row["take_up_share"] = info["take_up_share"]
            row["administrative_rate"] = info["administrative_rate"]
            row["share_band"] = info["share_band"]
        else:
            row["seeded"] = False
            # Left at the engine's universal-take-up default: no seeded
            # participation surface exists for this flag yet. Record why.
            row["ships_at_engine_default"] = program.populace_treatment not in {
                "out_of_scope",
                "near_universal",
            }
            followup = program.raw.get("followup")
            if followup is not None:
                row["followup"] = followup
            scope_owner = program.raw.get("scope_owner")
            if scope_owner is not None:
                row["scope_owner"] = scope_owner
        programs.append(row)
    return {
        "schema_version": 1,
        "classification": "release_diagnostics",
        "issue": "populace#312",
        "take_up_contract": {
            "version": contract.version,
            "inventory_built_against": contract.inventory_built_against,
            "asserted_constraint": contract.asserted_constraint,
        },
        "gate": {
            "name": gate.name,
            "passed": gate.passed,
            "failures": list(gate.failures),
        },
        "seeded_program_count": sum(1 for row in programs if row["seeded"]),
        "programs": programs,
    }


def write_us_take_up_participation_diagnostics(
    payload: dict[str, object], path: Path | str
) -> Path:
    """Write the US take-up participation diagnostics artifact as strict JSON."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path
