"""Compose the six existing nominal pre-clone source stages.

This is a declaration seam: the real stage kernels keep their selection,
calculation, typed context, ownership and acceptance contracts. Housing runs its
actual independent donor fit and recipient apply, then hands the completed full
pool to eligibility. The historical prior-year stage retains its frozen parity
scope; this module adds no wage-history or money-restatement interface.
"""

from __future__ import annotations

from collections.abc import Sequence

from microcosm.graph import KernelRegistry, Node, Owned

from . import graph_cps_carried as cps
from . import graph_early_preparation as hours
from . import graph_housing_source as housing
from . import graph_person_preparation as person
from . import graph_prior_year as prior
from . import multispine_pool as pool
from .housing_inputs import _DEFAULT_N_ESTIMATORS

PRECLONE_FINAL_POPULATION = f"{person.ELIGIBILITY_NODE}.boundary"
PRECLONE_ACCEPTED_CONTEXT = person.ELIGIBILITY_NODE
PRECLONE_OPERATOR_ORDER = (
    "derive_us_cps_carried_inputs",
    "with_us_hours_worked_inputs",
    "with_us_prior_year_income_inputs",
    "with_us_relationship_inputs",
    "with_us_housing_inputs",
    "with_us_eligibility_inputs",
)


def _append_stage(
    nodes: list[Node],
    inventory: dict[tuple[str, str], Owned],
    stage: tuple[Node, ...],
    *,
    population: str,
) -> None:
    nodes.extend(stage)
    # Only the completed full-pool population supplies the next stage's
    # inventory. Housing's donor and selected-source outputs belong to other
    # populations and cannot silently become carried full-pool declarations.
    for node in stage:
        if node.population == population:
            for owned in node.outputs:
                inventory[owned.entity, owned.column] = owned


def us_preclone_source_nodes(
    columns: Sequence[Owned],
    *,
    base: str,
    context_producer: str,
    donor_descriptor: bytes,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> tuple[Node, ...]:
    """Declare CPS, accepted hours, historical prior year, relationship,
    actual fresh-source housing and eligibility, in their original order.

    The initial pool and context producer are explicit. The donor descriptor
    comes from the existing authenticated rent preflight; ``n_estimators`` only
    configures that existing rent fit. The final population/context identifiers
    are ``PRECLONE_FINAL_POPULATION`` and ``PRECLONE_ACCEPTED_CONTEXT``.
    """
    if tuple(pool.POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER) != PRECLONE_OPERATOR_ORDER:
        raise ValueError(
            "US pre-clone operator order differs from the registered six-stage recipe."
        )
    inventory = {(owned.entity, owned.column): owned for owned in columns}
    if len(inventory) != len(columns):
        raise ValueError("US pre-clone declarations repeat coordinates.")
    nodes: list[Node] = []
    before_housing = (
        (
            cps.us_cps_carried_nodes,
            f"{cps.CPS_CARRIED_NODE}.boundary",
            cps.CPS_CARRIED_NODE,
        ),
        (
            hours.us_hours_nodes,
            f"{hours.HOURS_PHASE}.boundary",
            f"{hours.HOURS_PHASE}.gate",
        ),
        (
            prior.us_prior_year_source_nodes,
            f"{prior.PRIOR_YEAR_NODE}.boundary",
            prior.PRIOR_YEAR_NODE,
        ),
        (
            person.us_relationship_source_nodes,
            f"{person.RELATIONSHIP_NODE}.boundary",
            person.RELATIONSHIP_NODE,
        ),
    )
    for factory, population, accepted_context in before_housing:
        stage = factory(
            tuple(inventory.values()), base=base, context_producer=context_producer
        )
        _append_stage(nodes, inventory, stage, population=population)
        base, context_producer = population, accepted_context
    stage = housing.us_housing_source_nodes(
        tuple(inventory.values()),
        base=base,
        context_producer=context_producer,
        donor_descriptor=donor_descriptor,
        seed=pool.POOL_RANDOM_SEED,
        n_estimators=n_estimators,
    )
    _append_stage(
        nodes, inventory, stage, population=housing.HOUSING_SOURCE_MERGE_POPULATION
    )
    stage = person.us_eligibility_source_nodes(
        tuple(inventory.values()),
        base=housing.HOUSING_SOURCE_MERGE_POPULATION,
        context_producer=housing.HOUSING_SOURCE_ACCEPTED_CONTEXT,
    )
    _append_stage(nodes, inventory, stage, population=PRECLONE_FINAL_POPULATION)
    return tuple(nodes)


def register_us_preclone_source_kernels(
    registry: KernelRegistry, *, source_codecs=None
) -> None:
    """Register each existing stage once, including the real rent kernels."""
    cps.register_us_cps_carried_kernels(registry)
    hours.register_us_hours_kernels(registry)
    prior.register_us_prior_year_source_kernels(registry)
    person.register_us_person_preparation_kernels(registry)
    housing.register_us_housing_source_kernels(registry, source_codecs=source_codecs)
