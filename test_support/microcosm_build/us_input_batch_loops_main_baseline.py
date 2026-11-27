"""Frozen input-stage batch loops: the byte-identity reference for 1.6 · 2.

Copied from origin/main c58596526: the release tool's four input-stage loops
(ACA source inputs, Medicaid eligibility, December SSI amounts and SNAP
eligibility), the other-health-insurance premium loop and the PolicyEngine-US
adapter's ``materialize``. The release tool's loops take the loaded tool as
``builder`` and read its unchanged helpers from it. The household partition,
the household selection and the garbage-collection helpers, which the change
moves into ``microcosm.build.us_runtime.household_batches``, are copied here
as well, so this reference does not change with the code under test. Do not
edit these copies to follow production code.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.engine_lifecycle import release_engine_simulation
from microcosm.build.us_runtime.other_health_insurance import (
    US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES,
)
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE = 5_000


def _main_household_position_batches(
    n_households: int, batch_size: int | None
) -> Iterable[np.ndarray]:
    if n_households <= 0:
        return
    if batch_size is None or batch_size <= 0 or batch_size >= n_households:
        yield np.arange(n_households, dtype=np.int64)
        return
    for start in range(0, n_households, batch_size):
        stop = min(start + batch_size, n_households)
        yield np.arange(start, stop, dtype=np.int64)


def _main_select_households_by_position(frame: Frame, positions: np.ndarray) -> Frame:
    household_ids = frame.table("household")["household_id"].to_numpy()[positions]
    person_mask = frame.table("person")["person_household_id"].isin(household_ids)
    return frame.select(person_mask)


def _main_collect_batch_garbage() -> None:
    """Keep batch loops tidy without traversing the full object graph."""

    gc.collect(0)


def _main_collect_family_garbage() -> None:
    """Full collection at target-family boundaries (microcosm#456).

    ``release_engine_simulation`` frees each batch's array mass by refcount,
    but the residual simulation skeletons are cyclic, and anything promoted
    past generation 0 is invisible to ``_collect_batch_garbage`` forever —
    CPython's own full collections are throttled against the build's multi-GB
    long-lived heap, so without an explicit sweep the skeletons accumulate
    for the run's lifetime. One full collection per materialized family is
    cheap next to the family's engine work.
    """

    gc.collect()


@contextmanager
def _main_automatic_gc_suspended():
    """Avoid surprise full-graph cyclic GC inside memory-heavy PE batches."""

    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def main_aca_source_tax_unit_table_batched(
    builder,
    frame: Frame,
    target_tables: Mapping[str, pd.DataFrame],
    *,
    microsimulation_cls,
    maximum_microsim_batch_size: int | None,
) -> pd.DataFrame:
    builder._assert_no_formula_owned_columns(frame)
    tax_unit = frame.table("tax_unit").copy()
    household = frame.table("household")
    positions = builder._tax_unit_to_household_positions(frame)
    tax_unit["state_fips"] = builder._state_fips_text(
        np.asarray(household["state_fips"].to_numpy())[positions]
    )

    fill_columns = (
        "tax_unit_weight",
        "household_weight",
        "is_aca_ptc_eligible",
        "health_insurance_premiums_without_medicare_part_b",
        "assigned_aca_ptc",
        "weighted_assigned_aca_ptc",
        "slcsp",
    )
    tax_unit["tax_unit_weight"] = 0.0
    tax_unit["household_weight"] = frame.weights_for("household").values[positions]
    tax_unit["is_aca_ptc_eligible"] = False
    tax_unit["health_insurance_premiums_without_medicare_part_b"] = 0.0
    tax_unit["assigned_aca_ptc"] = 0.0
    tax_unit["weighted_assigned_aca_ptc"] = 0.0
    tax_unit["slcsp"] = 0.0

    tax_unit_positions = pd.Series(
        np.arange(len(tax_unit), dtype=np.int64),
        index=tax_unit["tax_unit_id"].to_numpy(),
    )
    n_households = frame.n("household")
    batches = tuple(
        _main_household_position_batches(n_households, maximum_microsim_batch_size)
    )
    if len(batches) > 1:
        print(
            "Materializing ACA source inputs in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )

    for household_positions in batches:
        with _main_automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else _main_select_households_by_position(frame, household_positions)
            )
            batch_simulation = microsimulation_cls(
                dataset=builder._dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                    period_only=True,
                )
            )
            batch_tax_unit = builder._aca_source_tax_unit_table_from_simulation(
                batch_frame,
                target_tables,
                simulation=batch_simulation,
            )
            full_positions = tax_unit_positions.reindex(
                batch_tax_unit["tax_unit_id"].to_numpy()
            ).to_numpy()
            if np.isnan(full_positions).any():
                raise RuntimeError(
                    "ACA source batch produced tax_unit_id values not present in "
                    "the full tax_unit table."
                )
            full_positions = full_positions.astype(np.int64)
            for column in fill_columns:
                tax_unit.iloc[
                    full_positions,
                    tax_unit.columns.get_loc(column),
                ] = batch_tax_unit[column].to_numpy()
            release_engine_simulation(batch_simulation)
            del batch_tax_unit, batch_simulation, batch_frame
        _main_collect_batch_garbage()
    _main_collect_family_garbage()
    return builder._with_state_take_up_rates(tax_unit, target_tables)


def main_medicaid_person_eligibility(
    builder,
    frame: Frame,
    *,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> np.ndarray:
    """Engine-computed person-level ``is_medicaid_eligible``, batched like ACA."""
    if simulation is not None:
        return (
            builder._calculate_array(
                simulation, "is_medicaid_eligible", map_to="person"
            )
            > 0
        )
    from policyengine_us import Microsimulation

    person_ids = frame.table("person")["person_id"].to_numpy()
    eligibility = np.zeros(len(person_ids), dtype=bool)
    person_positions = pd.Series(
        np.arange(len(person_ids), dtype=np.int64), index=person_ids
    )
    n_households = frame.n("household")
    batches = tuple(
        _main_household_position_batches(n_households, maximum_microsim_batch_size)
    )
    if len(batches) > 1:
        print(
            "Materializing Medicaid eligibility in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )
    for household_positions in batches:
        with _main_automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else _main_select_households_by_position(frame, household_positions)
            )
            batch_simulation = Microsimulation(
                dataset=builder._dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                    period_only=True,
                )
            )
            batch_eligible = (
                builder._calculate_array(
                    batch_simulation, "is_medicaid_eligible", map_to="person"
                )
                > 0
            )
            positions = person_positions.reindex(
                batch_frame.table("person")["person_id"].to_numpy()
            ).to_numpy()
            if np.isnan(positions).any():
                raise RuntimeError(
                    "Medicaid eligibility batch produced person_id values not "
                    "present in the full person table."
                )
            eligibility[positions.astype(np.int64)] = batch_eligible
            release_engine_simulation(batch_simulation)
            del batch_frame, batch_simulation
        _main_collect_batch_garbage()
    _main_collect_family_garbage()
    return eligibility


def main_ssi_person_uncapped_amount(
    builder,
    frame: Frame,
    *,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> np.ndarray:
    """December person-level potential federal SSI, batched like Medicaid.

    SSA's age-band recipient counts are December 2024 point-in-time stocks.
    ``uncapped_ssi > 0`` is the PolicyEngine-US 2.2.1 current-benefit
    candidate mask and does not depend on the take-up input being assigned.
    """

    period = f"{builder.PERIOD}-12"

    def calculate(active_simulation) -> np.ndarray:
        values = np.asarray(
            active_simulation.calculate(
                "uncapped_ssi",
                period=period,
                map_to="person",
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise RuntimeError(
                "SSI take-up materialization produced nonfinite uncapped_ssi values."
            )
        return values

    if simulation is not None:
        return calculate(simulation)

    from policyengine_us import Microsimulation

    person_ids = frame.table("person")["person_id"].to_numpy()
    uncapped = np.zeros(len(person_ids), dtype=np.float64)
    person_positions = pd.Series(
        np.arange(len(person_ids), dtype=np.int64), index=person_ids
    )
    n_households = frame.n("household")
    batches = tuple(
        _main_household_position_batches(n_households, maximum_microsim_batch_size)
    )
    if len(batches) > 1:
        print(
            "Materializing December SSI candidate amounts in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )
    for household_positions in batches:
        with _main_automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else _main_select_households_by_position(frame, household_positions)
            )
            batch_simulation = Microsimulation(
                dataset=builder._dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                    period_only=True,
                )
            )
            batch_uncapped = calculate(batch_simulation)
            positions = person_positions.reindex(
                batch_frame.table("person")["person_id"].to_numpy()
            ).to_numpy()
            if np.isnan(positions).any():
                raise RuntimeError(
                    "SSI candidate batch produced person_id values not present "
                    "in the full person table."
                )
            uncapped[positions.astype(np.int64)] = batch_uncapped
            release_engine_simulation(batch_simulation)
            del batch_frame, batch_simulation
        _main_collect_batch_garbage()
    _main_collect_family_garbage()
    return uncapped


def main_snap_spm_unit_eligibility(
    builder,
    frame: Frame,
    *,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> np.ndarray:
    """Engine-computed SPM-unit ``is_snap_eligible``, batched like Medicaid."""
    if simulation is not None:
        return (
            builder._calculate_array(simulation, "is_snap_eligible", map_to="spm_unit")
            > 0
        )
    from policyengine_us import Microsimulation

    spm_unit_ids = frame.table("spm_unit")["spm_unit_id"].to_numpy()
    eligibility = np.zeros(len(spm_unit_ids), dtype=bool)
    unit_positions = pd.Series(
        np.arange(len(spm_unit_ids), dtype=np.int64), index=spm_unit_ids
    )
    n_households = frame.n("household")
    batches = tuple(
        _main_household_position_batches(n_households, maximum_microsim_batch_size)
    )
    if len(batches) > 1:
        print(
            "Materializing SNAP eligibility in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )
    for household_positions in batches:
        with _main_automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else _main_select_households_by_position(frame, household_positions)
            )
            batch_simulation = Microsimulation(
                dataset=builder._dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                    period_only=True,
                )
            )
            batch_eligible = (
                builder._calculate_array(
                    batch_simulation, "is_snap_eligible", map_to="spm_unit"
                )
                > 0
            )
            positions = unit_positions.reindex(
                batch_frame.table("spm_unit")["spm_unit_id"].to_numpy()
            ).to_numpy()
            if np.isnan(positions).any():
                raise RuntimeError(
                    "SNAP eligibility batch produced spm_unit_id values not "
                    "present in the full spm_unit table."
                )
            eligibility[positions.astype(np.int64)] = batch_eligible
            release_engine_simulation(batch_simulation)
            del batch_frame, batch_simulation
        _main_collect_batch_garbage()
    _main_collect_family_garbage()
    return eligibility


def main_materialize(
    engine: PolicyEngineUSEngine,
    bundle: Frame,
    variables: Sequence[str],
    period: int | str,
) -> Mapping[str, np.ndarray]:
    """Compute ``variables`` for ``period`` with a Microsimulation.

    Builds a dataset for ``period`` from the bundle's entity tables
    (with the bundle's household weights as ``household_weight``), runs a
    ``Microsimulation`` over it, and calculates each variable. The
    simulation holds ``period`` alone, not policyengine-us's projection
    to later years (see :meth:`_read_year_dataset`).

    Args:
        bundle: A US-schema bundle.
        variables: PolicyEngine variable names to compute.
        period: Period to compute for (e.g. ``2026``).

    Returns:
        One array per variable, row-aligned to the variable's entity
        table in the bundle.

    Raises:
        ImportError: If ``policyengine_us`` is not installed.
        ValueError: If a computed array's length does not match its
            entity table (a structural mismatch the kernel refuses to
            pass through).
    """
    microsimulation_class = engine._import_policyengine_us().Microsimulation
    tables = engine._engine_tables(bundle)
    dataset = engine._read_year_dataset(tables, period)
    simulation = microsimulation_class(
        dataset=dataset,
        **({"spm": dict(engine._spm)} if engine._spm is not None else {}),
    )
    results: dict[str, np.ndarray] = {}
    for name in variables:
        entity = engine._entity_of(name)
        values = np.asarray(simulation.calculate(name, period=period))
        expected = bundle.n(entity)
        if values.shape != (expected,):
            raise ValueError(
                f"Materialized variable {name!r} has shape {values.shape} "
                f"but entity {entity!r} has {expected} row(s)."
            )
        results[name] = values
    return results


def main_materialize_modeled_premiums(
    frame: Frame,
    *,
    time_period: int,
    maximum_microsim_batch_size: int | None,
) -> dict[str, np.ndarray]:
    """Materialize tax-unit premiums in household batches and realign by ID."""

    household = frame.table("household")
    tax_unit = frame.table("tax_unit")
    person = frame.table("person")
    household_ids = household["household_id"].to_numpy()
    tax_unit_ids = tax_unit["tax_unit_id"].to_numpy()
    tax_unit_positions = pd.Series(
        np.arange(len(tax_unit_ids), dtype=np.int64),
        index=tax_unit_ids,
    )
    modeled = {
        variable: np.zeros(len(tax_unit_ids), dtype=np.float64)
        for variable in US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES
    }
    assigned = np.zeros(len(tax_unit_ids), dtype=bool)
    batch_size = (
        len(household_ids)
        if maximum_microsim_batch_size is None or maximum_microsim_batch_size <= 0
        else min(int(maximum_microsim_batch_size), len(household_ids))
    )
    if batch_size == 0:
        raise ValueError(
            "US other-health-insurance premium materialization requires at "
            "least one household."
        )

    engine = PolicyEngineUSEngine()
    for start in range(0, len(household_ids), batch_size):
        positions = np.arange(
            start,
            min(start + batch_size, len(household_ids)),
            dtype=np.int64,
        )
        full_batch = len(positions) == len(household_ids)
        batch_frame = (
            frame
            if full_batch
            else frame.select(
                person["person_household_id"].isin(household_ids[positions])
            )
        )
        batch_tax_unit_ids = batch_frame.table("tax_unit")["tax_unit_id"].to_numpy()
        full_positions = tax_unit_positions.reindex(batch_tax_unit_ids).to_numpy()
        if np.isnan(full_positions).any():
            raise ValueError(
                "US other-health-insurance premium batch produced tax-unit IDs "
                "outside the full frame."
            )
        full_positions = full_positions.astype(np.int64)
        if assigned[full_positions].any():
            raise ValueError(
                "US other-health-insurance premium batches overlap tax units; "
                "tax units must not span household batches."
            )
        materialized = main_materialize(
            engine,
            batch_frame,
            list(US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES),
            period=int(time_period),
        )
        for variable in US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES:
            if variable not in materialized:
                raise ValueError(
                    "PolicyEngine-US did not materialize required formula "
                    f"{variable!r}."
                )
            values = np.asarray(materialized[variable], dtype=np.float64)
            if values.shape != (len(batch_tax_unit_ids),):
                raise ValueError(
                    f"Materialized {variable!r} has shape {values.shape}; expected "
                    f"{(len(batch_tax_unit_ids),)} for the tax-unit batch."
                )
            modeled[variable][full_positions] = values
        assigned[full_positions] = True
        del materialized, batch_frame
        gc.collect()

    if not assigned.all():
        missing_ids = tax_unit_ids[~assigned].tolist()
        raise ValueError(
            "US other-health-insurance premium batches did not cover tax-unit "
            f"IDs {missing_ids}."
        )
    return modeled
