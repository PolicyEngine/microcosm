"""Household batches for the US release's policyengine-us passes.

The US release runs policyengine-us over its households one batch at a time:
the input stages (ACA source inputs, Medicaid eligibility, December SSI
amounts, SNAP eligibility and the other-health-insurance premiums), target
compilation and its reform passes, and post-export scoring. This module holds
what those passes share:

* :func:`household_position_batches` splits the household table into
  contiguous position ranges, and :func:`select_households_by_position` gives
  a batch frame holding those households, their people and their groups.
* :func:`household_batch_plan` makes the ranges and runs the frame checks that
  batching needs before any engine is built. A batch result can depend only on
  its own households when every group entity nests in one household
  (:func:`assert_group_entities_nest_in_households`) and every positive
  ``medicaid_claiming_tax_unit_id`` names a tax unit in the person's own
  household (:func:`assert_medicaid_claiming_tax_units_local`).
* :func:`checked_batch_engine` runs one batch on its engine and releases the
  engine afterwards, also when the batch raised. With the refusal on, it first
  refuses an engine that computed a population-aggregate variable
  (:func:`refuse_batch_population_aggregates`), including one computed on a
  branch the batch created through ``get_branch``
  (:func:`record_engine_branches`).
* :func:`automatic_gc_suspended`, :func:`collect_batch_garbage` and
  :func:`collect_pass_garbage` are the garbage-collection pattern of a batch
  loop: no automatic collection inside a batch, a youngest-generation
  collection after it, and one full collection when the pass ends.
  :func:`one_pass_garbage_collection` runs several calls of a pass function
  as one pass, with one full collection at the end.

A population-aggregate variable is a policyengine-us formula that sums or ranks
over every person or household in its engine
(:data:`microcosm.data.compatibility_recertification.POPULATION_AGGREGATE_VARIABLES`).
A batch engine holds one batch, so it would compute that batch's value instead
of the whole population's. The refusal is on by default when a pass has more
than one batch. A caller that hands a pass only part of the population switches
it on whatever the batch count (``refuse_population_aggregates=True``).
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.engine_lifecycle import release_engine_simulation
from microcosm.data.compatibility_recertification import (
    POPULATION_AGGREGATE_VARIABLES,
    engine_known_periods,
)
from microcosm.frame import Frame

__all__ = [
    "INPUT_PASS_POPULATION_AGGREGATE_REMEDY",
    "HouseholdBatchPlan",
    "assert_group_entities_nest_in_households",
    "assert_medicaid_claiming_tax_units_local",
    "automatic_gc_suspended",
    "checked_batch_engine",
    "collect_batch_garbage",
    "collect_pass_garbage",
    "group_to_household_positions",
    "household_batch_plan",
    "household_position_batches",
    "one_pass_garbage_collection",
    "record_engine_branches",
    "refuse_batch_population_aggregates",
    "select_households_by_position",
]

#: How an input-stage pass's population-aggregate refusal ends. An input pass
#: writes take-up and premium inputs for every household, so no batch size
#: makes such a variable correct there.
INPUT_PASS_POPULATION_AGGREGATE_REMEDY = (
    "so the values this pass computes would depend on which households share "
    "an engine. An input-stage pass must not request a variable that reaches "
    "them."
)


def household_position_batches(
    n_households: int, batch_size: int | None
) -> Iterable[np.ndarray]:
    """Contiguous household position ranges of at most ``batch_size``.

    ``None``, zero, a negative size or one of at least ``n_households`` gives
    one batch of every household; no households give no batches.
    """

    if n_households <= 0:
        return
    if batch_size is None or batch_size <= 0 or batch_size >= n_households:
        yield np.arange(n_households, dtype=np.int64)
        return
    for start in range(0, n_households, batch_size):
        stop = min(start + batch_size, n_households)
        yield np.arange(start, stop, dtype=np.int64)


def select_households_by_position(frame: Frame, positions: np.ndarray) -> Frame:
    """The households at ``positions``, with their people and groups."""

    household_ids = frame.table("household")["household_id"].to_numpy()[positions]
    person_mask = frame.table("person")["person_household_id"].isin(household_ids)
    return frame.select(person_mask)


def group_to_household_positions(frame: Frame, group_entity: str) -> np.ndarray:
    """Each ``group_entity`` row's household position in ``frame``.

    Refuses a group whose people live in more than one household, and a group
    with no people.
    """

    person = frame.table("person")
    household_ids = frame.table("household")["household_id"].to_numpy()
    group_ids = frame.table(group_entity)[f"{group_entity}_id"].to_numpy()
    person_group_column = f"person_{group_entity}_id"

    membership = person[[person_group_column, "person_household_id"]].drop_duplicates()
    counts = membership.groupby(person_group_column)["person_household_id"].nunique()
    ambiguous = counts[counts != 1]
    if not ambiguous.empty:
        raise ValueError(
            f"{group_entity} units must be nested in households; ambiguous ids "
            f"examples: {ambiguous.index[:5].tolist()}."
        )
    group_to_household = (
        membership.drop_duplicates(person_group_column)
        .set_index(person_group_column)["person_household_id"]
        .reindex(group_ids)
    )
    if group_to_household.isna().any():
        missing = group_ids[group_to_household.isna().to_numpy()][:5].tolist()
        raise ValueError(f"{group_entity} units with no person membership: {missing}.")
    household_positions = pd.Series(
        np.arange(len(household_ids), dtype=np.int64), index=household_ids
    )
    positions = household_positions.reindex(group_to_household.to_numpy()).to_numpy()
    if np.isnan(positions).any():
        raise ValueError(
            f"{group_entity} household ids are not present in household table."
        )
    return positions.astype(np.int64)


def assert_group_entities_nest_in_households(frame: Frame) -> None:
    """Refuse a frame whose group entities cross households.

    Batching selects persons by household and keeps every group they
    reference, so a group spanning two batches would be simulated twice, each
    time with only part of its members. ``Frame`` does not check nesting, and
    :func:`group_to_household_positions` raises on exactly this.
    """

    for entity in frame.entities:
        if entity in {"person", "household"}:
            continue
        group_to_household_positions(frame, entity)


def assert_medicaid_claiming_tax_units_local(frame: Frame, *, work: str) -> None:
    """Require positive Medicaid claiming IDs to resolve within the household.

    The engine's claiming-tax-unit helpers join IDs across persons. Requiring
    local references keeps those joins inside each household; an unresolved
    positive ID is refused too, since a pre-sliced caller may have omitted it.
    ``work`` names the pass in the refusal.
    """

    person = frame.table("person")
    column = "medicaid_claiming_tax_unit_id"
    if column not in person:
        return
    claims = person.loc[person[column] > 0]
    if claims.empty:
        return
    local_units = pd.MultiIndex.from_frame(
        person[["person_household_id", "person_tax_unit_id"]]
    )
    claimed_units = pd.MultiIndex.from_frame(claims[["person_household_id", column]])
    if not claimed_units.isin(local_units).all():
        raise ValueError(
            f"{work} is not batch-invariant: a positive "
            "medicaid_claiming_tax_unit_id does not resolve within the "
            "person's own household. Run the whole pool unbatched in one "
            "simulation with one slice or chunk."
        )


@dataclass(frozen=True)
class HouseholdBatchPlan:
    """A pass's household position batches and whether its refusal is on."""

    batches: tuple[np.ndarray, ...]
    refuse_population_aggregates: bool


def household_batch_plan(
    frame: Frame,
    maximum_batch_size: int | None,
    *,
    refuse_population_aggregates: bool | None = None,
    work: str,
) -> HouseholdBatchPlan:
    """Split ``frame``'s households and run the checks batching needs.

    The population-aggregate refusal is on when there is more than one batch,
    unless ``refuse_population_aggregates`` says otherwise: a caller that
    passes only part of the population sets it to ``True``. Before any engine
    is built, a frame whose group entities span households is refused when its
    households are split into more than one batch or the refusal is on, and a
    positive ``medicaid_claiming_tax_unit_id`` outside the person's household
    is refused when the refusal is on, as target compilation does.
    """

    batches = tuple(
        household_position_batches(frame.n("household"), maximum_batch_size)
    )
    batched = len(batches) > 1
    refusing = (
        batched
        if refuse_population_aggregates is None
        else bool(refuse_population_aggregates)
    )
    if batched or refusing:
        assert_group_entities_nest_in_households(frame)
    if refusing:
        assert_medicaid_claiming_tax_units_local(frame, work=work)
    return HouseholdBatchPlan(batches=batches, refuse_population_aggregates=refusing)


@contextmanager
def record_engine_branches(simulation: Any) -> Iterator[None]:
    """Retain branches created through get_branch until the batch is checked.

    Core's clone copies instance attributes, including a wrapped get_branch.
    Rebind that method to each child before it can create its own branches.
    Retained branches are released even if a formula detached them.
    """

    retained: list[object] = []
    originals: dict[int, tuple[object, object, bool]] = {}
    seen: set[int] = set()

    def get_branch(current, *args, **kwargs):
        branch = originals[id(current)][1](*args, **kwargs)
        retain(branch)
        return branch

    def retain(current):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        retained.append(current)
        original = getattr(current, "get_branch", None)
        if callable(original):
            had_instance_method = "get_branch" in vars(current)
            if getattr(original, "__func__", None) is get_branch:
                # clone copied the parent's bound wrapper; recover the
                # underlying method and bind it to the child.
                _, parent_original, had_instance_method = originals[
                    id(original.__self__)
                ]
                original = MethodType(parent_original.__func__, current)
            originals[id(current)] = (current, original, had_instance_method)
            current.get_branch = MethodType(get_branch, current)
        branches = getattr(current, "branches", None)
        if isinstance(branches, dict):
            for branch in tuple(branches.values()):
                retain(branch)
        retain(getattr(current, "baseline", None))

    retain(simulation)
    # engine_known_periods walks this attribute.
    simulation._target_materialization_branches = retained
    try:
        yield
    finally:
        for current, original, had_instance_method in originals.values():
            if had_instance_method:
                current.get_branch = original
            else:
                del current.get_branch
        for current in retained:
            # Core's shallow clone also copies the retained-list reference.
            if "_target_materialization_branches" in vars(current):
                del current._target_materialization_branches
            if current is not simulation:
                release_engine_simulation(current)


def refuse_batch_population_aggregates(
    simulation: Any,
    *,
    batch: int,
    batches: int,
    work: str,
    remedy: str,
) -> None:
    """Refuse any known population aggregate before releasing a batch engine.

    The check reads the periods the engine, its live branches, the branches
    :func:`record_engine_branches` kept and a reform engine's ``baseline``
    hold for each population-aggregate variable. ``work`` names the pass and
    ``remedy`` ends the message; it continues "here one household batch
    instead of the pool, ...".
    """

    computed = sorted(engine_known_periods(simulation, POPULATION_AGGREGATE_VARIABLES))
    if not computed:
        return
    raise ValueError(
        f"{work} is not batch-invariant: the engine for household batch "
        f"{batch}/{batches} computed "
        f"{', '.join(f'{variable}@{period}' for variable, period in computed)}. "
        "These policyengine-us formulas aggregate over the whole simulated "
        "population, here one household batch instead of the pool, "
        f"{remedy}"
    )


@contextmanager
def checked_batch_engine(
    simulation: Any,
    *,
    refuse_population_aggregates: bool,
    batch: int,
    batches: int,
    work: str,
    remedy: str,
) -> Iterator[Any]:
    """Run one batch on ``simulation``; check it, then release it.

    With the refusal on, branches the batch creates through ``get_branch`` are
    kept until the check, which runs once the batch's code has finished
    without an exception and before the engine is released. The engine is
    released on exit whether or not the batch raised (microcosm#456: an
    engine's rules keep a reference to it, so a finished engine can outlive
    the caller's last name for it).
    """

    try:
        if not refuse_population_aggregates:
            yield simulation
            return
        with record_engine_branches(simulation):
            yield simulation
            refuse_batch_population_aggregates(
                simulation,
                batch=batch,
                batches=batches,
                work=work,
                remedy=remedy,
            )
    finally:
        release_engine_simulation(simulation)


def collect_batch_garbage() -> None:
    """Keep batch loops tidy without traversing the full object graph."""

    gc.collect(0)


def collect_pass_garbage() -> None:
    """Full collection when a batched pass ends (microcosm#456).

    ``release_engine_simulation`` frees each batch's array mass by refcount,
    but the residual simulation skeletons are cyclic, and anything promoted
    past generation 0 is invisible to :func:`collect_batch_garbage` forever —
    CPython's own full collections are throttled against the build's multi-GB
    long-lived heap, so without an explicit sweep the skeletons accumulate
    for the run's lifetime. One full collection per pass (a target family,
    an input stage) is cheap next to the pass's engine work. Inside
    :func:`one_pass_garbage_collection` it leaves the collection to the end.
    """

    if _PASS_GARBAGE_DEFERRED:
        return
    gc.collect()


#: While set, :func:`collect_pass_garbage` leaves the full collection to
#: :func:`one_pass_garbage_collection`.
_PASS_GARBAGE_DEFERRED = False


@contextmanager
def one_pass_garbage_collection() -> Iterator[None]:
    """Run the pass calls inside as one pass: one full collection at the end.

    A reuse of an earlier pass (``microcosm.build.household_pass_reuse``)
    calls the pass function once per batch of households it reruns, and each
    call ends with :func:`collect_pass_garbage`; a full pass collects once.
    Nested uses collect once, when the outermost ends, also after an
    exception.
    """

    global _PASS_GARBAGE_DEFERRED
    already_deferred = _PASS_GARBAGE_DEFERRED
    _PASS_GARBAGE_DEFERRED = True
    try:
        yield
    finally:
        _PASS_GARBAGE_DEFERRED = already_deferred
        if not already_deferred:
            gc.collect()


@contextmanager
def automatic_gc_suspended() -> Iterator[None]:
    """Avoid surprise full-graph cyclic GC inside memory-heavy PE batches."""

    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
