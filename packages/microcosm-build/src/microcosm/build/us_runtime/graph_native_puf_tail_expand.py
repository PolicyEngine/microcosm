"""Real selective EXPAND over explicitly invented domain/assignment declarations.

This structural slice copies whole clone1 households and halves their host
IMPORTANCE weights. Donor support weights are metadata, never household mass.
It neither authenticates source domains nor matches/qualifies donor roles,
places money, assigns geography, or makes a candidate release eligible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    StructuralDelta,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import support_provenance, survey_population_domains
from .support_provenance import (
    PUF_TAX_DETAIL_CLONE_INDEX,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from .survey_population_domains import Domain, Source

POLICY = "native_tail_matched_parent_half_importance_v1"
PROTOCOL = "microcosm.us.native-tail-invented-expansion/1"
TAIL_CLONE_INDEX = 2
ALPHA = (1, 2)
MAX_FIXTURE_HOUSEHOLDS = 10_000
MAX_FIXTURE_ENTITY_ROWS = 100_000
_INT64_MAX = np.iinfo(np.int64).max


def _require(condition, reason):
    if not condition:
        raise ValueError("NATIVE_TAIL_EXPAND_" + reason)


def _identifier(value):
    return type(value) is int and 0 <= value <= _INT64_MAX


@dataclass(frozen=True, slots=True)
class InventedTailAssignment:
    """Preselected fixture pair; no matching, person or source authority."""

    donor_id: int
    parent_household_id: int
    donor_support_weight: float


@dataclass(frozen=True, slots=True)
class InventedHouseholdDomain:
    """Complete supplied fixture roster bound to live origin coordinates."""

    household_id: int
    source: Source
    support_source_id: int
    spine_source_id: int
    clone_index: int
    domain: Domain
    statistical_unit: str


@dataclass(frozen=True, slots=True)
class InventedTailExpansion:
    """Immutable descriptive declaration, deliberately not an issued capability."""

    domains: tuple[InventedHouseholdDomain, ...]
    assignments: tuple[InventedTailAssignment, ...]

    @property
    def source_admission_issued(self) -> bool:
        return False

    @property
    def release_eligible(self) -> bool:
        return False

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(_declaration_payload(self))).hexdigest()


def _validate_declaration(declaration):
    _require(type(declaration) is InventedTailExpansion, "DECLARATION_TYPE")
    domains, assignments = declaration.domains, declaration.assignments
    _require(
        type(domains) is tuple and 0 < len(domains) <= MAX_FIXTURE_HOUSEHOLDS,
        "DOMAIN_ROSTER",
    )
    units = {
        row.domain: row.statistical_unit
        for row in survey_population_domains.declaration()
    }
    for row in domains:
        _require(type(row) is InventedHouseholdDomain, "DOMAIN_TYPE")
        _require(
            all(
                _identifier(value)
                for value in (
                    row.household_id,
                    row.support_source_id,
                    row.spine_source_id,
                )
            ),
            "DOMAIN_IDS",
        )
        _require(
            type(row.source) is Source and type(row.domain) is Domain, "DOMAIN_KNOWN"
        )
        _require(
            type(row.clone_index) is int
            and row.clone_index in (0, PUF_TAX_DETAIL_CLONE_INDEX),
            "DOMAIN_CLONE",
        )
        _require(
            type(row.statistical_unit) is str
            and row.statistical_unit == units[row.domain],
            "DOMAIN_UNIT",
        )
    _require(
        len({row.household_id for row in domains}) == len(domains), "DOMAIN_DUPLICATE"
    )
    origin_domains = {}
    for row in domains:
        origin = (row.source, row.support_source_id, row.spine_source_id)
        value = (row.domain, row.statistical_unit)
        _require(
            origin not in origin_domains or origin_domains[origin] == value,
            "DOMAIN_ORIGIN_DISAGREEMENT",
        )
        origin_domains[origin] = value
    _require(
        type(assignments) is tuple and 0 < len(assignments) <= len(domains),
        "ASSIGNMENTS",
    )
    for pair in assignments:
        _require(type(pair) is InventedTailAssignment, "ASSIGNMENT_TYPE")
        _require(
            _identifier(pair.donor_id)
            and pair.donor_id > 0
            and _identifier(pair.parent_household_id),
            "ASSIGNMENT_IDS",
        )
        _require(
            type(pair.donor_support_weight) is float
            and isfinite(pair.donor_support_weight)
            and pair.donor_support_weight > 0,
            "DONOR_SUPPORT_WEIGHT",
        )
    _require(
        len({p.donor_id for p in assignments}) == len(assignments), "DUPLICATE_DONOR"
    )
    _require(
        len({p.parent_household_id for p in assignments}) == len(assignments),
        "DUPLICATE_PARENT",
    )
    by_id = {row.household_id: row for row in domains}
    for pair in assignments:
        _require(pair.parent_household_id in by_id, "PARENT_MISSING")
        row = by_id[pair.parent_household_id]
        _require(row.clone_index == PUF_TAX_DETAIL_CLONE_INDEX, "PARENT_CLONE")
        _require(
            row.domain in (Domain.SHARED_HOUSING, Domain.RESIDUAL_HOUSING),
            "PARENT_HOUSING",
        )


def declare_invented_tail_expansion(
    domains: Iterable[InventedHouseholdDomain],
    assignments: Iterable[InventedTailAssignment],
) -> InventedTailExpansion:
    """Validate literal fixture declarations; no source qualification occurs."""
    result = InventedTailExpansion(tuple(domains), tuple(assignments))
    _validate_declaration(result)
    return InventedTailExpansion(
        tuple(sorted(result.domains, key=lambda row: row.household_id)),
        tuple(sorted(result.assignments, key=lambda row: row.parent_household_id)),
    )


def _declaration_payload(declaration):
    _validate_declaration(declaration)
    return (
        PROTOCOL,
        tuple(
            (
                r.household_id,
                r.source.value,
                r.support_source_id,
                r.spine_source_id,
                r.clone_index,
                r.domain.value,
                r.statistical_unit,
            )
            for r in declaration.domains
        ),
        tuple(
            (r.donor_id, r.parent_household_id, r.donor_support_weight)
            for r in declaration.assignments
        ),
    )


def _decode_declaration(payload):
    _require(
        type(payload) is tuple and len(payload) == 3 and payload[0] == PROTOCOL,
        "DECLARATION_PAYLOAD",
    )
    _require(
        type(payload[1]) is tuple and type(payload[2]) is tuple, "DECLARATION_ROWS"
    )
    try:
        _require(
            all(type(row) is tuple and len(row) == 7 for row in payload[1]),
            "DOMAIN_ROW",
        )
        _require(
            all(type(row) is tuple and len(row) == 3 for row in payload[2]),
            "ASSIGNMENT_ROW",
        )
        domains = tuple(
            InventedHouseholdDomain(
                r[0], Source(r[1]), r[2], r[3], r[4], Domain(r[5]), r[6]
            )
            for r in payload[1]
        )
        assignments = tuple(InventedTailAssignment(*r) for r in payload[2])
    except (ValueError, TypeError):
        raise ValueError("NATIVE_TAIL_EXPAND_DECLARATION_ROWS") from None
    result = declare_invented_tail_expansion(domains, assignments)
    _require(_declaration_payload(result) == payload, "DECLARATION_CANONICAL")
    return result


def provenance_columns(entity: str) -> tuple[str, ...]:
    """All origin/clone cells consumed on each entity's declared input slice."""
    return (
        support_source_id_column(entity),
        spine_source_id_column(entity),
        support_channel_column(entity),
        support_clone_index_column(entity),
    )


def _cells():
    return tuple(
        (entity, support_clone_index_column(entity), "int64")
        for entity in US_SCHEMA.entities
    )


def _params(declaration):
    return {
        "protocol": PROTOCOL,
        "policy": POLICY,
        "alpha": ALPHA,
        "basis": "invented_fixture",
        "source_admission_issued": False,
        "release_eligible": False,
        "declaration": _declaration_payload(declaration),
        "expand_cells": _cells(),
        "expand_weight_entity": "household",
        "expand_weight_kind": WeightKind.IMPORTANCE.value,
        "expand_require_sole_weight_entity": True,
    }


def _int64(table, column):
    _require(
        column in table and table[column].dtype == np.dtype("int64"), "INT64_COLUMN"
    )
    return table[column].to_numpy(copy=True)


def _validated_tables(context, declaration):
    _require(set(context.tables) == set(US_SCHEMA.entities), "ENTITY_SCOPE")
    tables = context.tables
    for entity in US_SCHEMA.entities:
        table = tables[entity]
        _require(0 < len(table) <= MAX_FIXTURE_ENTITY_ROWS, "ENTITY_SIZE")
        ids = _int64(table, US_SCHEMA.entity_id_column(entity))
        _require(len(set(ids)) == len(ids) and (ids >= 0).all(), "ENTITY_IDS")
        source = _int64(table, support_source_id_column(entity))
        native = _int64(table, spine_source_id_column(entity))
        clones = _int64(table, support_clone_index_column(entity))
        _require((source >= 0).all() and (native >= 0).all(), "ORIGIN_IDS")
        _require(np.isin(clones, (0, PUF_TAX_DETAIL_CLONE_INDEX)).all(), "BASE_CLONE")
        channel = table[support_channel_column(entity)]
        _require(
            not channel.isna().any()
            and channel.isin(tuple(s.value for s in Source)).all(),
            "SOURCE_CHANNEL",
        )
        _require(
            not table.duplicated(
                [support_source_id_column(entity), support_clone_index_column(entity)]
            ).any(),
            "ORIGIN_DUPLICATE",
        )
    household = tables["household"]
    _require(
        set(household.household_id) == {r.household_id for r in declaration.domains},
        "DOMAIN_COVERAGE",
    )
    by_id = household.set_index("household_id")
    for row in declaration.domains:
        live = by_id.loc[row.household_id]
        _require(
            (
                live[support_channel_column("household")],
                int(live[support_source_id_column("household")]),
                int(live[spine_source_id_column("household")]),
                int(live[support_clone_index_column("household")]),
            )
            == (
                row.source.value,
                row.support_source_id,
                row.spine_source_id,
                row.clone_index,
            ),
            "DOMAIN_ORIGIN",
        )
    person = tables[US_SCHEMA.person_entity]
    person_hh = _int64(person, US_SCHEMA.membership_column("household"))
    _require(set(person_hh) == set(household.household_id), "HOUSEHOLD_MEMBERS")
    group_households = {}
    for group in US_SCHEMA.group_entities:
        membership = _int64(person, US_SCHEMA.membership_column(group))
        group_id = US_SCHEMA.entity_id_column(group)
        group_table = tables[group].set_index(group_id)
        _require(set(membership) == set(group_table.index), "GROUP_MEMBERS")
        joined = group_table.reindex(membership)
        _require(
            np.array_equal(
                joined[support_clone_index_column(group)],
                person[support_clone_index_column("person")],
            ),
            "MEMBERSHIP_CLONE",
        )
        _require(
            np.array_equal(
                joined[support_channel_column(group)],
                person[support_channel_column("person")],
            ),
            "MEMBERSHIP_CHANNEL",
        )
        links = pd.DataFrame({"group": membership, "household": person_hh})
        count = links.groupby("group").household.nunique()
        _require((count == 1).all(), "GROUP_NOT_HOUSEHOLD_CLOSED")
        group_households[group] = links.groupby("group").household.first()
    for pair in declaration.assignments:
        members = person.loc[person_hh == pair.parent_household_id]
        _require(
            members[US_SCHEMA.membership_column("tax_unit")].nunique() == 1,
            "PARENT_TAX_UNITS",
        )
    return group_households


def _new_ids(table, entity, count):
    ids = _int64(table, US_SCHEMA.entity_id_column(entity))
    maximum = int(ids.max())
    _require(count <= _INT64_MAX - maximum, "ID_EXHAUSTION")
    # Python arithmetic avoids int64 overflow at the endpoint.
    return np.array([maximum + 1 + i for i in range(count)], dtype=np.int64)


class NativePufTailExpandKernel(KernelBase):
    ref = "us.native_puf_tail.invented_expand@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.EXPAND,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            support_provenance,
            survey_population_domains,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        declaration = _decode_declaration(context.params.get("declaration"))
        _require(dict(context.params) == _params(declaration), "PARAMS")
        _require(
            context.params["expand_require_sole_weight_entity"] is True
            and context.params["source_admission_issued"] is False
            and context.params["release_eligible"] is False,
            "PARAM_FLAGS",
        )
        node = context.node
        _require(
            node.kernel == self.ref
            and node.structural is StructuralDelta.EXPAND
            and node.mass == "conserve"
            and node.entrants is False
            and not node.sources
            and not node.artifact_inputs
            and not node.artifact_outputs
            and not node.outputs,
            "NODE_CONTRACT",
        )
        expected_inputs = tuple(
            Slice(entity, provenance_columns(entity)) for entity in US_SCHEMA.entities
        )
        _require(node.inputs == expected_inputs, "READSET")
        group_households = _validated_tables(context, declaration)
        incoming = context.weights.get("household")
        _require(
            incoming is not None and incoming.kind is WeightKind.IMPORTANCE,
            "WEIGHT_KIND",
        )
        household = context.tables["household"]
        weights = np.asarray(incoming.values, dtype=np.float64)
        _require(
            weights.shape == (len(household),)
            and np.isfinite(weights).all()
            and (weights >= 0).all(),
            "HOST_WEIGHTS",
        )
        parent_ids = tuple(pair.parent_household_id for pair in declaration.assignments)
        selected = set(parent_ids)
        person = context.tables[US_SCHEMA.person_entity]
        lineage, columns = {}, {}
        for entity in US_SCHEMA.entities:
            table = context.tables[entity]
            id_column = US_SCHEMA.entity_id_column(entity)
            if entity == US_SCHEMA.person_entity:
                mask = person[US_SCHEMA.membership_column("household")].isin(selected)
            else:
                mask = table[id_column].map(group_households[entity]).isin(selected)
            source_ids = table.loc[mask, id_column].to_numpy(dtype=np.int64, copy=True)
            target_ids = _new_ids(table, entity, len(source_ids))
            lineage[entity] = pd.Series(
                source_ids, index=pd.Index(target_ids, name=id_column), dtype="int64"
            )
            columns[(entity, support_clone_index_column(entity))] = pd.Series(
                np.concatenate(
                    [
                        _int64(table, support_clone_index_column(entity)),
                        np.full(len(source_ids), TAIL_CLONE_INDEX, dtype=np.int64),
                    ]
                ),
                index=pd.Index(
                    np.concatenate([_int64(table, id_column), target_ids]),
                    name=id_column,
                ),
                dtype="int64",
            )
        by_id = pd.Series(weights, index=household.household_id)
        copied_weights = by_id.reindex(lineage["household"].to_numpy()).to_numpy(
            dtype=np.float64
        )
        # Subnormal loss is tested explicitly below, never repaired or clipped.
        with np.errstate(under="ignore"):
            halves = copied_weights / 2
        _require(np.isfinite(halves).all() and (halves > 0).all(), "POSITIVE_HALF")
        _require(np.array_equal(halves + halves, copied_weights), "EXACT_HALF")
        retained = weights.copy()
        positions = pd.Index(household.household_id).get_indexer(
            lineage["household"].to_numpy()
        )
        retained[positions] = halves
        facts = []
        tail_households = {
            int(parent): int(target) for target, parent in lineage["household"].items()
        }
        for pair in declaration.assignments:
            parent = float(by_id.loc[pair.parent_household_id])
            facts.append(
                {
                    "donor_id": pair.donor_id,
                    "parent_household_id": pair.parent_household_id,
                    "tail_household_id": tail_households[pair.parent_household_id],
                    "donor_support_weight": pair.donor_support_weight,
                    "parent_importance_weight": parent,
                    "retained_importance_weight": parent / 2,
                    "tail_importance_weight": parent / 2,
                }
            )
        return KernelResult(
            expand=lineage,
            columns=columns,
            weights=Weights(np.concatenate([retained, halves]), WeightKind.IMPORTANCE),
            receipt={
                "protocol": PROTOCOL,
                "policy": POLICY,
                "alpha": list(ALPHA),
                "basis": "invented_fixture",
                "source_admission_issued": False,
                "matching_qualified": False,
                "person_projection_qualified": False,
                "monetary_placement": False,
                "geography_assigned": False,
                "release_eligible": False,
                "declaration_sha256": declaration.sha256,
                "donor_weights_used_as_host_mass": False,
                "assignments": facts,
                "lineage_sha256": hashlib.sha256(
                    canonical_json(
                        {
                            e: [
                                [int(target), int(source)]
                                for target, source in lineage[e].items()
                            ]
                            for e in US_SCHEMA.entities
                        }
                    )
                ).hexdigest(),
            },
        )


class NativePufTailCloneClaimKernel(KernelBase):
    """Own precisely the six clone-index overlays materialized by EXPAND."""

    ref = "us.native_puf_tail.invented_expand_claim@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self), support_provenance, dependencies=self.capabilities.dependencies
        )

    def run(self, context: KernelContext) -> KernelResult:
        _require(
            context.node.outputs
            == tuple(Owned(e, c, t, rewrite=True) for e, c, t in _cells()),
            "CLAIM_OUTPUTS",
        )
        _require(
            not context.node.inputs and dict(context.params) == {"protocol": PROTOCOL},
            "CLAIM_CONTRACT",
        )
        return KernelResult(
            columns={
                (e, c): pd.Series(
                    _int64(context.tables[e], c),
                    index=pd.Index(
                        _int64(context.tables[e], US_SCHEMA.entity_id_column(e)),
                        name=US_SCHEMA.entity_id_column(e),
                    ),
                    dtype=t,
                )
                for e, c, t in _cells()
            },
            receipt={"protocol": PROTOCOL, "source_admission_issued": False},
        )


def native_puf_tail_expand_nodes(
    columns: Sequence[Owned],
    *,
    base: str,
    declaration: InventedTailExpansion,
    prefix: str = "native_tail_fixture_expand",
) -> tuple[Node, ...]:
    """Declare structural lineage only; supplied fixture claims are not authority."""
    _validate_declaration(declaration)
    declaration = declare_invented_tail_expansion(
        declaration.domains, declaration.assignments
    )
    _require(type(base) is str and bool(base), "BASE")
    columns = tuple(columns)
    _require(all(type(column) is Owned for column in columns), "INVENTORY_TYPE")
    inventory = {(column.entity, column.column): column for column in columns}
    _require(len(inventory) == len(columns), "INVENTORY_DUPLICATE")
    for entity in US_SCHEMA.entities:
        for name in provenance_columns(entity):
            owned = inventory.get((entity, name))
            dtype = "string" if name == support_channel_column(entity) else "int64"
            _require(
                owned is not None and owned.dtype == dtype and owned.rows == "all",
                "INVENTORY_PROVENANCE",
            )
    return (
        Node(
            prefix,
            NativePufTailExpandKernel.ref,
            inputs=tuple(Slice(e, provenance_columns(e)) for e in US_SCHEMA.entities),
            structural=StructuralDelta.EXPAND,
            base=base,
            mass="conserve",
            params=_params(declaration),
            description="Invented matched housing support: clone whole households and split parent IMPORTANCE in half.",
        ),
        Node(
            prefix + ".owned",
            NativePufTailCloneClaimKernel.ref,
            outputs=tuple(Owned(e, c, t, rewrite=True) for e, c, t in _cells()),
            population=prefix,
            params={"protocol": PROTOCOL},
        ),
    )


def register_native_puf_tail_expand_kernels(registry: KernelRegistry) -> None:
    """Register only the structural fixture operator and its cell claimant."""
    registry.register(NativePufTailExpandKernel())
    registry.register(NativePufTailCloneClaimKernel())
