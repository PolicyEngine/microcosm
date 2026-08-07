"""The kernel datatype: a weighted, stratified frame of entity tables.

A :class:`Frame` holds one table per entity (person plus the group
entities declared by its :class:`~populace.frame.schema.EntitySchema`), typed
weights for the weighted entities, per-person stratum labels, and — as a
documented placeholder — link tables for the schema's declared associations.
Structure is established once, at assembly, and validated on every
construction:

- every group table's id column is unique, sorted ascending, and contains
  exactly the distinct ids referenced by the person membership column;
- the person table's id column is unique;
- weight vectors match their entity table lengths;
- strata are aligned to the person index;
- column names are globally unique across entity tables (the flattening
  rule rules engines rely on);
- every provided link table carries both linked entities' id columns, and
  every id it references exists in the linked entity's table.

Frame operations return new frames. Entity-table access follows a read-only
caller convention; integrity-critical receipts belong in the frame's private,
immutable metadata rather than in mutable table columns.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace.frame.schema import EntitySchema
from populace.frame.weights import (
    MassChange,
    MassChangeRecord,
    WeightKind,
    Weights,
    assert_kind_transition,
)

__all__ = ["Frame", "DEFAULT_STRATUM"]

#: Stratum label assigned when a frame is constructed without explicit strata.
DEFAULT_STRATUM = "default"

#: Kind rank used to resolve the union kind on concatenation.
_KIND_ORDER = ("design", "importance", "calibrated")

#: Literal accepted by :meth:`Frame.with_weights` to require mass conservation.
CONSERVE_MASS = "conserve"

#: Default relative tolerance for the ``mass="conserve"`` total check.
_MASS_CONSERVE_RTOL = 1e-9


class Frame:
    """A weighted sampling frame of entity tables, with kernel invariants.

    The name is the survey-statistics term: a *sampling frame* is the list
    of units a sample is drawn from and the thing weights refer back to. A
    ``Frame`` is that object made executable — entity tables with explicit
    linkage, typed weights with conservation invariants, and per-person
    strata recording how each record entered the frame. Every operator in
    the populace stack takes a frame and returns a frame.

    Args:
        tables: One :class:`pandas.DataFrame` per entity declared by
            ``schema`` (the person entity and every group entity), plus
            optionally one table per declared link, keyed by the link's
            name. Tables are copied; the caller's frames are never mutated
            or aliased.
        schema: The entity structure (see
            :class:`~populace.frame.schema.EntitySchema`).
        weights: Typed weight vectors keyed by entity name. At least one
            entity must carry weights (typically the household).
        strata: Per-person provenance labels, index-aligned to the person
            table. ``None`` assigns the single stratum
            :data:`DEFAULT_STRATUM` to every person.
        metadata: Optional stage receipts composed of mappings, sequences,
            sets, and scalar values. The structure is copied into immutable,
            pickle-safe containers and propagated by frame operations.

    Raises:
        TypeError: If a weight value is not a :class:`Weights`, or ``strata``
            is not a :class:`pandas.Series`.
        ValueError: If any kernel invariant fails. Messages name the exact
            entity, column, or ids involved.
    """

    __slots__ = (
        "_link_tables",
        "_mass_log",
        "_metadata",
        "_schema",
        "_strata",
        "_tables",
        "_weights",
    )

    def __init__(
        self,
        tables: Mapping[str, pd.DataFrame],
        schema: EntitySchema,
        weights: Mapping[str, Weights],
        strata: pd.Series | None = None,
        *,
        mass_log: tuple[MassChangeRecord, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._schema = schema
        self._mass_log = tuple(mass_log)
        self._metadata = _freeze_metadata(metadata)
        declared_links = {link.name for link in schema.links}
        self._tables: dict[str, pd.DataFrame] = {}
        self._link_tables: dict[str, pd.DataFrame] = {}
        for name, frame in tables.items():
            target = self._link_tables if name in declared_links else self._tables
            target[name] = frame.copy()
        self._weights = dict(weights)
        self._strata = self._validated_strata(strata)
        self._validate_tables()
        self._validate_linkage()
        self._validate_links()
        self._validate_global_columns()
        self._validate_weights()
        self._validate_reserved_weight_columns()

    def revalidate(self) -> None:
        """Re-run every constructor invariant against the current state.

        :meth:`table` and :attr:`person` return the stored tables, not
        copies, so a caller that mutates them in place can silently break
        the invariants construction proved. Seams that hand a long-lived
        bundle across trust boundaries (a build loop between stages, a
        writer before export) call this to fail closed on post-construction
        corruption instead of shipping it.

        Raises:
            TypeError, ValueError: Exactly as construction would on the
                same state — missing or extra tables, broken linkage,
                duplicated or unsorted ids, cross-entity column collisions,
                invalid weights or reserved weight columns, or strata that
                no longer align with the person index.
        """

        self._strata = self._validated_strata(self._strata)
        self._validate_tables()
        self._validate_linkage()
        self._validate_links()
        self._validate_global_columns()
        self._validate_weights()
        self._validate_reserved_weight_columns()

    # ------------------------------------------------------------------
    # Validation (constructor invariants)
    # ------------------------------------------------------------------

    def _validate_tables(self) -> None:
        expected = set(self._schema.entities)
        got = set(self._tables)
        missing = sorted(expected - got)
        unknown = sorted(got - expected)
        if missing:
            raise ValueError(
                f"Missing entity table(s): {missing}; schema declares "
                f"{list(self._schema.entities)}."
            )
        if unknown:
            message = (
                f"Unknown entity table(s): {unknown}; schema declares "
                f"{list(self._schema.entities)}"
            )
            link_names = [link.name for link in self._schema.links]
            if link_names:
                message += f" and link(s) {link_names}"
            raise ValueError(message + ".")
        person = self._tables[self._schema.person_entity]
        id_column = self._schema.person_id_column
        if id_column not in person.columns:
            raise ValueError(f"Person table must carry the id column {id_column!r}.")
        if person[id_column].isna().any():
            raise ValueError(f"Person id column {id_column!r} contains missing values.")
        if person[id_column].duplicated().any():
            duplicated = person[id_column][person[id_column].duplicated()]
            raise ValueError(
                f"Person id column {id_column!r} must be unique; duplicated ids "
                f"include {duplicated.head(5).tolist()}."
            )

    def _validate_linkage(self) -> None:
        person = self._tables[self._schema.person_entity]
        for group in self._schema.group_entities:
            membership_column = self._schema.membership_column(group)
            id_column = self._schema.id_column(group)
            if membership_column not in person.columns:
                raise ValueError(
                    f"Person table must carry the membership column "
                    f"{membership_column!r} linking to entity {group!r}."
                )
            table = self._tables[group]
            if id_column not in table.columns:
                raise ValueError(
                    f"Group table {group!r} must carry the id column {id_column!r}."
                )
            membership = person[membership_column]
            if membership.isna().any():
                raise ValueError(
                    f"Membership column {membership_column!r} contains missing "
                    "values; every person must reference exactly one "
                    f"{group} id."
                )
            ids = table[id_column]
            if ids.isna().any():
                raise ValueError(
                    f"Group table {group!r} id column {id_column!r} contains "
                    "missing values."
                )
            if ids.duplicated().any():
                duplicated = ids[ids.duplicated()]
                raise ValueError(
                    f"Group table {group!r} id column {id_column!r} must be "
                    f"unique; duplicated ids include {duplicated.head(5).tolist()}."
                )
            id_values = ids.to_numpy()
            if not np.array_equal(np.sort(id_values), id_values):
                raise ValueError(
                    f"Group table {group!r} id column {id_column!r} must be "
                    "sorted ascending."
                )
            referenced = np.unique(membership.to_numpy())
            if not np.array_equal(referenced, id_values):
                orphaned = np.setdiff1d(id_values, referenced)
                dangling = np.setdiff1d(referenced, id_values)
                parts = []
                if orphaned.size:
                    parts.append(
                        f"ids in the table referenced by no person: "
                        f"{orphaned[:5].tolist()}"
                    )
                if dangling.size:
                    parts.append(
                        f"ids referenced by persons but absent from the table: "
                        f"{dangling[:5].tolist()}"
                    )
                raise ValueError(
                    f"Group table {group!r} id column {id_column!r} must contain "
                    f"exactly the distinct values of {membership_column!r}; "
                    + "; ".join(parts)
                    + "."
                )

    def _validate_links(self) -> None:
        """Validate provided link tables against the schema's declared links.

        Placeholder scope: a provided link table must carry both linked
        entities' id columns (no missing values), and every id it references
        must exist in the linked entity's table. Link tables are join
        tables, not entity tables — they are exempt from the global
        column-uniqueness rule and are not yet carried through
        ``select``/``concat``.
        """
        for link in self._schema.links:
            table = self._link_tables.get(link.name)
            if table is None:
                continue
            for side in (link.left_entity, link.right_entity):
                column = self._schema.entity_id_column(side)
                if column not in table.columns:
                    raise ValueError(
                        f"Link table {link.name!r} must carry the id column "
                        f"{column!r} referencing entity {side!r}."
                    )
                values = table[column]
                if values.isna().any():
                    raise ValueError(
                        f"Link table {link.name!r} id column {column!r} "
                        "contains missing values."
                    )
                dangling = np.setdiff1d(values.to_numpy(), self._entity_ids(side))
                if dangling.size:
                    raise ValueError(
                        f"Link table {link.name!r} references {side} id(s) "
                        f"absent from the {side!r} table: "
                        f"{dangling[:5].tolist()}."
                    )

    def _validate_global_columns(self) -> None:
        owners: dict[str, list[str]] = {}
        for entity in self._schema.entities:
            for column in self._tables[entity].columns:
                owners.setdefault(column, []).append(entity)
        duplicated = {
            column: entities for column, entities in owners.items() if len(entities) > 1
        }
        if duplicated:
            described = ", ".join(
                f"{column!r} on {entities}" for column, entities in duplicated.items()
            )
            raise ValueError(
                "Column names must be globally unique across entity tables "
                f"(the flattening rule); duplicated: {described}."
            )

    def _validate_weights(self) -> None:
        if not self._weights:
            raise ValueError(
                "Bundle requires weights for at least one entity "
                "(typically the household)."
            )
        known = set(self._schema.entities)
        for entity, weights in self._weights.items():
            if entity not in known:
                raise ValueError(
                    f"Weights provided for unknown entity {entity!r}; schema "
                    f"declares {list(self._schema.entities)}."
                )
            if not isinstance(weights, Weights):
                raise TypeError(
                    f"weights[{entity!r}] must be a Weights instance, got "
                    f"{type(weights).__name__}."
                )
            n = len(self._tables[entity])
            if len(weights) != n:
                raise ValueError(
                    f"Weights for entity {entity!r} have length {len(weights)} "
                    f"but the {entity!r} table has {n} row(s)."
                )

    def _validate_reserved_weight_columns(self) -> None:
        """Reserve the ``{entity}_weight`` column names for the kernel.

        Weights are typed vectors stored off the tables; the rules-engine
        adapters materialize them into ``{entity}_weight`` columns at export
        from the bundle's typed weights. A ``{entity}_weight`` column sitting
        in an entity table the bundle does *not* carry typed weights for is an
        orphan the kernel would otherwise ignore silently while the engine
        consumes it — the calibration-loses-mass footgun. The name belongs to
        the kernel, so reject it.

        A ``{entity}_weight`` column on an entity that *does* carry typed
        weights is redundant but harmless: the export always overwrites it
        from the typed weights (see the adapter's ``_engine_tables``), so it
        is permitted for round-trip convenience.
        """
        for entity in self._schema.entities:
            reserved = f"{entity}_weight"
            for table_entity in self._schema.entities:
                if reserved not in self._tables[table_entity].columns:
                    continue
                if table_entity == entity and entity in self._weights:
                    continue
                raise ValueError(
                    f"Column {reserved!r} on the {table_entity!r} table is a "
                    "reserved weight-column name the kernel owns (it is "
                    f"materialized from {entity!r}'s typed weights at export). "
                    "Store weights as a typed Weights vector, not a table "
                    "column; drop this column before constructing the frame."
                )

    def _validated_strata(self, strata: pd.Series | None) -> pd.Series:
        person = self._tables.get(self._schema.person_entity)
        if person is None:
            raise ValueError(
                f"Missing entity table(s): [{self._schema.person_entity!r}]."
            )
        if strata is None:
            return pd.Series(
                DEFAULT_STRATUM, index=person.index, dtype=object, name="stratum"
            )
        if not isinstance(strata, pd.Series):
            raise TypeError(
                f"strata must be a pandas Series, got {type(strata).__name__}."
            )
        if len(strata) != len(person) or not strata.index.equals(person.index):
            raise ValueError(
                f"strata must be aligned to the person index: got length "
                f"{len(strata)} vs {len(person)} person row(s) "
                "(indexes must match exactly)."
            )
        if strata.isna().any():
            raise ValueError("strata contains missing labels.")
        out = strata.copy()
        out.name = "stratum"
        return out

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def schema(self) -> EntitySchema:
        """The entity schema this bundle was assembled with."""
        return self._schema

    @property
    def person(self) -> pd.DataFrame:
        """The person table. Treat as read-only; mutating it breaks invariants."""
        return self._tables[self._schema.person_entity]

    @property
    def entities(self) -> tuple[str, ...]:
        """All entity names: person first, then the group entities."""
        return self._schema.entities

    @property
    def strata(self) -> pd.Series:
        """Per-person stratum labels, index-aligned to the person table."""
        return self._strata

    @property
    def weighted_entities(self) -> tuple[str, ...]:
        """Entities that carry explicit weight vectors."""
        return tuple(entity for entity in self.entities if entity in self._weights)

    @property
    def links(self) -> tuple[str, ...]:
        """Names of the link tables this frame carries, in declaration order."""
        return tuple(
            link.name for link in self._schema.links if link.name in self._link_tables
        )

    @property
    def mass_log(self) -> tuple[MassChangeRecord, ...]:
        """Declared, intentional weight-mass changes applied to reach this frame.

        Each :class:`~populace.frame.weights.MassChangeRecord` is one
        :meth:`with_weights` call that passed ``mass=MassChange(...)`` — an
        importance resampling or a recalibration to a new control total —
        recorded in application order. Mass-conserving replacements
        (``mass="conserve"``) leave no entry. The log is carried forward
        through :meth:`select` and :meth:`concat`.
        """
        return self._mass_log

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Deeply immutable stage metadata carried with this frame.

        Metadata records integrity contracts that must survive ordinary frame
        transformations without becoming mutable through :meth:`table`.
        Keys and nested mapping keys are strings; mappings, sequences, and
        sets are frozen on construction.
        """

        return self._metadata

    def link(self, name: str) -> pd.DataFrame:
        """Return the link table named ``name``.

        Args:
            name: A link declared by the schema whose table this frame
                carries.

        Returns:
            The link table. Treat as read-only.

        Raises:
            ValueError: If ``name`` is not a declared link, or the frame
                carries no table for it.
        """
        declared = [link.name for link in self._schema.links]
        if name not in declared:
            raise ValueError(
                f"Unknown link {name!r}; schema declares links {declared}."
            )
        if name not in self._link_tables:
            raise ValueError(
                f"No table provided for link {name!r}; provided links: "
                f"{list(self.links)}."
            )
        return self._link_tables[name]

    def table(self, entity: str) -> pd.DataFrame:
        """Return the table for ``entity``.

        Args:
            entity: An entity declared by the schema.

        Returns:
            The entity's table. Treat as read-only.

        Raises:
            ValueError: If ``entity`` is not declared by the schema.
        """
        if entity not in self._tables:
            raise ValueError(
                f"Unknown entity {entity!r}; schema declares {list(self.entities)}."
            )
        return self._tables[entity]

    def n(self, entity: str) -> int:
        """Number of rows in the ``entity`` table."""
        return len(self.table(entity))

    def weights_for(self, entity: str) -> Weights:
        """Return the explicit weight vector stored for ``entity``.

        Args:
            entity: An entity declared by the schema.

        Returns:
            The stored :class:`~populace.frame.weights.Weights`.

        Raises:
            ValueError: If no weights are stored for ``entity``. The message
                names the entities that do carry weights.
        """
        self.table(entity)  # validates the entity name
        if entity not in self._weights:
            raise ValueError(
                f"No weights stored for entity {entity!r}; weighted entities: "
                f"{list(self.weighted_entities)}."
            )
        return self._weights[entity]

    def resolve_weights(self, entity: str) -> Weights:
        """Return ``entity``'s weights as a typed :class:`Weights`, resolving inheritance.

        Unlike :meth:`weights_for` — which returns *only* the explicit vector
        stored for ``entity`` and raises otherwise — this resolves the
        *effective* weights and preserves their kind: an entity without its own
        stored weights inherits the single weighted group entity's weights
        broadcast through membership, and the returned :class:`Weights` carries
        that source entity's :class:`~populace.frame.weights.WeightKind`. This
        is what lets a person-level fit run on a household-weighted frame: the
        person inherits the household's design/importance/calibrated kind, not a
        bare ndarray that has dropped the kind.

        The values are exactly those of :meth:`_effective_weights` (so weighted
        accounting and a fit weight the same rows identically); the kind is the
        stored kind when ``entity`` has its own weights, otherwise the source
        group entity's kind.

        Args:
            entity: An entity declared by the schema.

        Returns:
            The effective :class:`~populace.frame.weights.Weights` for
            ``entity``: the stored vector as-is when present, otherwise the
            inherited values tagged with the source entity's kind.

        Raises:
            ValueError: If ``entity`` is not declared by the schema, or its
                weights are ambiguous — the same zero/multiple-weighted-group
                conditions :meth:`_effective_weights` raises on (the message
                names them), or a group's members carry unequal person-level
                weights.
        """
        self.table(entity)  # validates the entity name
        if entity in self._weights:
            return self._weights[entity]
        values = self._effective_weights(entity)
        source_kind = self._inherited_kind(entity)
        return Weights(values=values, kind=source_kind)

    def _inherited_kind(self, entity: str) -> WeightKind:
        """Kind of the weights ``entity`` inherits when it stores none.

        Mirrors :meth:`_effective_weights`'s resolution *exactly*, so the kind
        always names the same source the values come from:

        - the person entity inherits the single weighted group entity's kind;
        - a group entity without its own weights derives from the person-level
          weights — which is the person's own kind when the person stores
          weights, otherwise (recursively) the single weighted group's kind.

        The earlier version always returned the single weighted *group*
        entity's kind, which (a) raised on person-weighted frames where
        ``_effective_weights`` succeeds via the person path, and (b) could tag
        person-derived values with a sibling group's kind.
        """
        person_entity = self._schema.person_entity
        if entity == person_entity:
            candidates = [
                group for group in self._schema.group_entities if group in self._weights
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot resolve weights for entity {entity!r}: no explicit "
                    f"weights and {len(candidates)} weighted group entities "
                    f"{candidates}; store weights for exactly one entity to "
                    "broadcast from."
                )
            return self._weights[candidates[0]].kind
        # A group entity without its own weights derives from the person-level
        # weights — the same source _effective_weights uses for its values.
        if person_entity in self._weights:
            return self._weights[person_entity].kind
        return self._inherited_kind(person_entity)

    def stratum_mass(self) -> pd.Series:
        """Weighted person mass per stratum.

        Persons are weighted by their effective person-level weights (explicit
        person weights when stored, otherwise the weighted group entity's
        weights broadcast through membership).

        Returns:
            Mass per stratum label, sorted by label, named ``"mass"``.
        """
        person_weights = self._effective_weights(self._schema.person_entity)
        mass = (
            pd.Series(person_weights, index=self._strata.to_numpy())
            .groupby(level=0)
            .sum()
        )
        mass.name = "mass"
        mass.index.name = "stratum"
        return mass

    # ------------------------------------------------------------------
    # Effective weights (used by accounting and stratum mass)
    # ------------------------------------------------------------------

    def _effective_weights(self, entity: str) -> np.ndarray:
        """Row-aligned weight values for ``entity``.

        Explicit weights win. Persons without explicit weights inherit the
        single weighted group entity's weights through membership. A group
        entity without explicit weights derives its weights from the
        person-level weights of its members, which must be constant within
        each group.

        Raises:
            ValueError: If person weights are ambiguous (zero or multiple
                weighted group entities) or a group's members carry unequal
                person-level weights.
        """
        if entity in self._weights:
            return self._weights[entity].values
        person_entity = self._schema.person_entity
        if entity == person_entity:
            candidates = [
                group for group in self._schema.group_entities if group in self._weights
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot resolve person-level weights: no explicit "
                    f"{person_entity!r} weights and {len(candidates)} weighted "
                    f"group entities {candidates}; store weights for exactly "
                    "one entity to broadcast from."
                )
            return self._group_values_to_person(
                candidates[0], self._weights[candidates[0]].values
            )
        person_weights = self._effective_weights(person_entity)
        return self._person_values_to_group(entity, person_weights)

    def _group_positions(self, group: str) -> np.ndarray:
        """Position of each person's group id within the (sorted) group table."""
        person = self.person
        membership = person[self._schema.membership_column(group)].to_numpy()
        ids = self._tables[group][self._schema.id_column(group)].to_numpy()
        return np.searchsorted(ids, membership)

    def _group_values_to_person(self, group: str, values: np.ndarray) -> np.ndarray:
        """Broadcast a group-aligned vector onto persons via membership."""
        return np.asarray(values)[self._group_positions(group)]

    def _person_values_to_group(self, group: str, values: np.ndarray) -> np.ndarray:
        """Collapse a person-aligned vector to one value per group.

        The vector must be constant within each group (true for weights of
        nested units); otherwise the collapse is ambiguous and refused.
        """
        positions = self._group_positions(group)
        n_groups = self.n(group)
        lo = np.full(n_groups, np.inf)
        hi = np.full(n_groups, -np.inf)
        np.minimum.at(lo, positions, values)
        np.maximum.at(hi, positions, values)
        unequal = lo != hi
        if unequal.any():
            ids = self._tables[group][self._schema.id_column(group)].to_numpy()
            raise ValueError(
                f"Cannot derive weights for entity {group!r}: members of "
                f"{group} id(s) {ids[unequal][:5].tolist()} carry unequal "
                f"person-level weights; store explicit weights for {group!r}."
            )
        return lo

    # ------------------------------------------------------------------
    # Operations (every operation returns a new, re-validated bundle)
    # ------------------------------------------------------------------

    def with_weights(
        self,
        entity: str,
        weights: Weights,
        *,
        mass: "str | MassChange",
    ) -> "Frame":
        """Return a new bundle with ``entity``'s weights replaced.

        Kind transitions are enforced: weights only move forward
        (``design -> importance -> calibrated``; same-kind replacement is
        allowed).

        The effect on total mass is an explicit, required decision — there is
        no silent default that could lose mass (calibration dropping a
        population from 300M to 3 must never pass unnoticed). Pass ``mass`` as
        one of:

        * :data:`CONSERVE_MASS` (the string ``"conserve"``) — the replacement
          must keep the existing total mass (within ``rtol=1e-9``). Requires
          existing weights to conserve against.
        * a :class:`~populace.frame.weights.MassChange` — declares an
          intentional change of mass (importance resampling, or calibration to
          a new control total), with a reason. If its ``factor`` is given, the
          realized ratio must match it (within ``rtol=1e-9``). The change is
          appended to the returned frame's :attr:`mass_log`.

        Args:
            entity: Entity whose weights to set.
            weights: The replacement vector (length must match the entity
                table).
            mass: The mass policy — :data:`CONSERVE_MASS` or a
                :class:`~populace.frame.weights.MassChange`. Required.

        Returns:
            A new validated bundle.

        Raises:
            TypeError: If ``weights`` is not a :class:`Weights`, or ``mass`` is
                neither :data:`CONSERVE_MASS` nor a :class:`MassChange`.
            ValueError: On a backward kind transition, a length mismatch,
                ``mass="conserve"`` without existing weights or against a
                changed total (the message carries both totals), or a
                :class:`MassChange` whose declared ``factor`` does not match
                the realized ratio.
        """
        self.table(entity)  # validates the entity name
        if not isinstance(weights, Weights):
            raise TypeError(
                f"weights must be a Weights instance, got {type(weights).__name__}."
            )
        if mass != CONSERVE_MASS and not isinstance(mass, MassChange):
            raise TypeError(
                "with_weights requires an explicit mass policy: pass "
                f"mass={CONSERVE_MASS!r} or mass=MassChange(factor=..., "
                f"reason=...); got {mass!r}."
            )
        existing = self._weights.get(entity)
        if existing is not None:
            assert_kind_transition(existing.kind, weights.kind)

        record: MassChangeRecord | None = None
        if mass == CONSERVE_MASS:
            if existing is None:
                raise ValueError(
                    f"mass='conserve' needs existing weights for entity "
                    f"{entity!r} to conserve against; none are stored. Pass a "
                    "MassChange to declare the initial mass intentionally."
                )
            try:
                existing.assert_mass_conserved(weights, rtol=_MASS_CONSERVE_RTOL)
            except ValueError as exc:
                raise ValueError(
                    f"with_weights({entity!r}, mass='conserve') violates mass "
                    f"conservation: {exc}"
                ) from None
        else:
            record = self._mass_change_record(entity, existing, weights, mass)

        new_weights = dict(self._weights)
        new_weights[entity] = weights
        new_mass_log = self._mass_log if record is None else (*self._mass_log, record)
        return Frame(
            {**self._tables, **self._link_tables},
            self._schema,
            new_weights,
            self._strata,
            mass_log=new_mass_log,
            metadata=self._metadata,
        )

    @staticmethod
    def _mass_change_record(
        entity: str,
        existing: Weights | None,
        weights: Weights,
        change: MassChange,
    ) -> MassChangeRecord:
        """Validate a declared :class:`MassChange` and build its log record."""
        old_total = existing.total if existing is not None else 0.0
        new_total = weights.total
        if change.factor is not None and existing is not None and old_total != 0.0:
            realized = new_total / old_total
            if not np.isclose(
                realized, change.factor, rtol=_MASS_CONSERVE_RTOL, atol=0.0
            ):
                raise ValueError(
                    f"with_weights({entity!r}) declared MassChange "
                    f"factor={change.factor!r} but the realized ratio is "
                    f"{realized!r} (old total {old_total!r} -> new total "
                    f"{new_total!r})."
                )
        return MassChangeRecord(
            entity=entity,
            old_total=old_total,
            new_total=new_total,
            declared_factor=change.factor,
            reason=change.reason,
        )

    def broadcast(self, column: str, to: str = "person") -> pd.Series:
        """Map a group-table column onto persons via membership.

        Args:
            column: A column on any entity table (column names are globally
                unique, so the owning entity is unambiguous). A person-level
                column broadcasts to itself.
            to: Target entity; only the person entity is supported.

        Returns:
            A person-aligned :class:`pandas.Series` named ``column``.

        Raises:
            ValueError: If ``column`` is unknown or ``to`` is not the person
                entity.
        """
        person_entity = self._schema.person_entity
        if to != person_entity:
            raise ValueError(
                f"broadcast target must be the person entity "
                f"{person_entity!r}, got {to!r}."
            )
        owner = self.column_entity(column)
        person = self.person
        if owner == person_entity:
            return person[column].copy()
        values = self._group_values_to_person(
            owner, self._tables[owner][column].to_numpy()
        )
        return pd.Series(values, index=person.index, name=column)

    def column_entity(self, column: str) -> str:
        """Return the entity whose table carries ``column``.

        Args:
            column: A column name.

        Returns:
            The owning entity (unambiguous: column names are globally unique).

        Raises:
            ValueError: If no entity table carries ``column``.
        """
        for entity in self.entities:
            if column in self._tables[entity].columns:
                return entity
        raise ValueError(
            f"Column {column!r} not found on any entity table {list(self.entities)}."
        )

    def place(
        self,
        column: str,
        entity: str,
        *,
        how: str = "sum",
        head_flag: str | None = None,
    ) -> "Frame":
        """Move ``column`` onto ``entity``, aggregating or carrying as needed.

        This is the single entity-placement operation builds use instead of
        hand-rolled groupby/map plumbing. Three directions are supported:

        * **person -> group**: aggregate person values to one per group.
          ``how`` is one of ``"sum"`` (numeric), ``"max"`` (numeric),
          ``"any"`` (boolean), or ``"first"`` (first member in person-table
          order; any dtype).
        * **group -> person**: ``how="broadcast"`` gives every member the
          group value; ``how="head"`` lands the value on one designated
          member per group (rows named by the boolean person column
          ``head_flag``, or the group's first member when ``head_flag`` is
          None) and the dtype's zero elsewhere.
        * **group -> group**: the source groups must nest inside the target
          groups (all members of a source group share one target group);
          source values aggregate into target rows with ``how`` as in
          person -> group.

        Weights, strata, links, and the mass log pass through untouched —
        placement moves a column, never mass.

        Args:
            column: Column to move (globally unique, so its owner is
                unambiguous). Structural columns (entity ids, membership
                columns) are refused.
            entity: Destination entity.
            how: Placement rule; see above. Defaults to ``"sum"``.
            head_flag: For ``how="head"`` only — boolean person column naming
                exactly one member per group.

        Returns:
            A new validated bundle with the column on ``entity``.

        Raises:
            ValueError: On unknown column/entity, structural columns, a
                ``how`` invalid for the direction, non-nested group -> group
                placement, an ambiguous ``head_flag``, or a dtype invalid
                for ``how``.
        """
        person_entity = self._schema.person_entity
        if entity not in self.entities:
            raise ValueError(
                f"Unknown destination entity {entity!r}; frame has "
                f"{list(self.entities)}."
            )
        structural = {self._schema.person_id_column} | {
            name
            for group in self._schema.group_entities
            for name in (
                self._schema.id_column(group),
                self._schema.membership_column(group),
            )
        }
        if column in structural:
            raise ValueError(
                f"Column {column!r} is structural (id/membership) and cannot be placed."
            )
        if head_flag is not None and how != "head":
            raise ValueError(
                f'head_flag is only meaningful with how="head", got how={how!r}.'
            )
        owner = self.column_entity(column)
        if owner == entity:
            return self
        # No destination-collision check is needed: the flattening rule
        # (global column-name uniqueness) makes a same-named destination
        # column impossible while the source still carries it.
        values = self._tables[owner][column].to_numpy()

        if owner == person_entity:
            placed = self._aggregate_to_group(column, entity, values, how)
        elif entity == person_entity:
            placed = self._carry_to_person(column, owner, values, how, head_flag)
        else:
            placed = self._aggregate_group_to_group(column, owner, entity, values, how)

        tables = {name: table for name, table in self._tables.items()}
        tables.update(self._link_tables)
        destination = tables[entity].copy()
        destination[column] = placed
        tables[entity] = destination
        tables[owner] = tables[owner].drop(columns=[column])
        return Frame(
            tables,
            self._schema,
            dict(self._weights),
            self._strata,
            mass_log=self._mass_log,
            metadata=self._metadata,
        )

    def _aggregate_positions(
        self,
        column: str,
        values: np.ndarray,
        positions: np.ndarray,
        n_rows: int,
        how: str,
    ) -> np.ndarray:
        """Aggregate ``values`` into ``n_rows`` slots at ``positions``."""
        if how == "first":
            out = np.zeros(n_rows, dtype=values.dtype)
            # first occurrence in table order wins
            order_first = np.unique(positions, return_index=True)
            out[order_first[0]] = values[order_first[1]]
            return out
        if how == "any":
            if values.dtype != np.bool_:
                raise ValueError(
                    f'place(how="any") needs a boolean column; {column!r} has '
                    f"dtype {values.dtype}."
                )
            out = np.zeros(n_rows, dtype=bool)
            np.logical_or.at(out, positions, values)
            return out
        if how in ("sum", "max"):
            if not np.issubdtype(values.dtype, np.number):
                raise ValueError(
                    f'place(how="{how}") needs a numeric column; {column!r} '
                    f"has dtype {values.dtype}."
                )
            if how == "sum":
                out = np.zeros(n_rows, dtype=np.float64)
                np.add.at(out, positions, values.astype(np.float64))
                return out
            out = np.full(n_rows, -np.inf)
            np.maximum.at(out, positions, values.astype(np.float64))
            out[np.isneginf(out)] = 0.0
            return out
        raise ValueError(
            f"Unknown aggregation {how!r}; expected one of "
            "'sum', 'max', 'any', 'first'."
        )

    def _aggregate_to_group(
        self, column: str, group: str, values: np.ndarray, how: str
    ) -> np.ndarray:
        """person -> group placement."""
        positions = self._group_positions(group)
        return self._aggregate_positions(column, values, positions, self.n(group), how)

    def _carry_to_person(
        self,
        column: str,
        group: str,
        values: np.ndarray,
        how: str,
        head_flag: str | None,
    ) -> np.ndarray:
        """group -> person placement (broadcast or head-carry)."""
        if how == "broadcast":
            return self._group_values_to_person(group, values)
        if how != "head":
            raise ValueError(
                f"group -> person placement supports how='broadcast' or "
                f"'head', got {how!r}."
            )
        positions = self._group_positions(group)
        person = self.person
        if head_flag is not None:
            if head_flag not in person.columns:
                raise ValueError(
                    f"head_flag column {head_flag!r} not on the person table."
                )
            flags = person[head_flag].to_numpy()
            if flags.dtype != np.bool_:
                raise ValueError(
                    f"head_flag column {head_flag!r} must be boolean, got "
                    f"dtype {flags.dtype}."
                )
            counts = np.zeros(self.n(group), dtype=np.int64)
            np.add.at(counts, positions, flags.astype(np.int64))
            if (counts != 1).any():
                ids = self._tables[group][self._schema.id_column(group)]
                bad = ids.to_numpy()[counts != 1]
                raise ValueError(
                    f"head_flag {head_flag!r} must name exactly one member "
                    f"per {group!r}; offending id(s) include "
                    f"{bad[:5].tolist()}."
                )
            head_rows = flags
        else:
            first_rows = np.unique(positions, return_index=True)[1]
            head_rows = np.zeros(len(person), dtype=bool)
            head_rows[first_rows] = True
        if values.dtype == np.bool_:
            zero: object = False
        elif np.issubdtype(values.dtype, np.number):
            zero = values.dtype.type(0)
        else:
            raise ValueError(
                f'place(how="head") needs a numeric or boolean column; '
                f"{column!r} has dtype {values.dtype}."
            )
        broadcast = self._group_values_to_person(group, values)
        return np.where(head_rows, broadcast, zero)

    def _aggregate_group_to_group(
        self,
        column: str,
        source: str,
        target: str,
        values: np.ndarray,
        how: str,
    ) -> np.ndarray:
        """group -> group placement; source groups must nest in target."""
        source_positions = self._group_positions(source)
        target_positions = self._group_positions(target)
        n_source = self.n(source)
        lo = np.full(n_source, np.iinfo(np.int64).max, dtype=np.int64)
        hi = np.full(n_source, -1, dtype=np.int64)
        np.minimum.at(lo, source_positions, target_positions)
        np.maximum.at(hi, source_positions, target_positions)
        unequal = lo != hi
        if unequal.any():
            ids = self._tables[source][self._schema.id_column(source)]
            bad = ids.to_numpy()[unequal]
            raise ValueError(
                f"Cannot place {column!r} from {source!r} onto {target!r}: "
                f"{source!r} id(s) {bad[:5].tolist()} span multiple "
                f"{target!r} rows ({source!r} does not nest in {target!r})."
            )
        return self._aggregate_positions(column, values, lo, self.n(target), how)

    def concat(self, other: "Frame") -> "Frame":
        """Union of two bundles; this is how pool strata assemble.

        The bundles must share the same schema, per-entity column sets, and
        weighted-entity set. Their strata must differ (disjoint label sets) or
        their id spaces must be disjoint for every entity; overlapping ids
        within overlapping strata would be ambiguous and are refused. When
        strata differ but integer id spaces collide, ``other``'s ids are
        shifted past this bundle's maximum per entity, preserving structure.

        Total weight mass is preserved exactly: each entity's union vector is
        the concatenation of the two vectors, with kind equal to the further
        of the two along ``design -> importance -> calibrated``.

        Args:
            other: The bundle to union with.

        Returns:
            A new validated bundle with ``ignore_index`` person rows
            (``self`` first), group tables re-sorted by id, and concatenated
            strata and weights.

        Raises:
            TypeError: If ``other`` is not a :class:`Frame`.
            NotImplementedError: If either frame carries link tables (links
                are a placeholder; link-aware concat comes with the full
                link operator).
            ValueError: On schema/column/weight-set mismatch, overlapping
                strata with overlapping id spaces, or non-integer colliding
                ids that cannot be shifted.
        """
        if not isinstance(other, Frame):
            raise TypeError(f"concat expects a Frame, got {type(other).__name__}.")
        if self._link_tables or other._link_tables:
            raise NotImplementedError(
                "concat does not yet carry link tables (links are a "
                "documented placeholder); drop link tables before "
                "concatenating."
            )
        if self._schema != other._schema:
            raise ValueError("Cannot concat bundles with different schemas.")
        if self._metadata != other._metadata:
            raise ValueError("Cannot concat bundles with different frame metadata.")
        for entity in self.entities:
            mine = set(self._tables[entity].columns)
            theirs = set(other._tables[entity].columns)
            if mine != theirs:
                raise ValueError(
                    f"Cannot concat: entity {entity!r} column sets differ "
                    f"(only here: {sorted(mine - theirs)}; only there: "
                    f"{sorted(theirs - mine)})."
                )
        if set(self._weights) != set(other._weights):
            raise ValueError(
                f"Cannot concat: weighted entities differ "
                f"({list(self.weighted_entities)} vs "
                f"{list(other.weighted_entities)})."
            )

        strata_overlap = sorted(
            set(self._strata.unique()) & set(other._strata.unique())
        )
        id_overlap = [
            entity
            for entity in self.entities
            if np.intersect1d(self._entity_ids(entity), other._entity_ids(entity)).size
            > 0
        ]
        if strata_overlap and id_overlap:
            raise ValueError(
                "Cannot concat: bundles share strata "
                f"{strata_overlap} and overlapping id spaces for entities "
                f"{id_overlap}; concatenated strata must differ or id spaces "
                "must be disjoint."
            )

        other_tables = {
            entity: other._tables[entity].copy() for entity in self.entities
        }
        for entity in id_overlap:
            self._shift_entity_ids(other_tables, entity)

        person_entity = self._schema.person_entity
        tables: dict[str, pd.DataFrame] = {}
        weights: dict[str, Weights] = {}
        tables[person_entity] = pd.concat(
            [self.person, other_tables[person_entity]], ignore_index=True
        )
        if person_entity in self._weights:
            weights[person_entity] = self._concat_weights(person_entity, other)
        for group in self._schema.group_entities:
            id_column = self._schema.id_column(group)
            combined = pd.concat(
                [self._tables[group], other_tables[group]], ignore_index=True
            )
            order = np.argsort(combined[id_column].to_numpy(), kind="stable")
            tables[group] = combined.iloc[order].reset_index(drop=True)
            if group in self._weights:
                union = self._concat_weights(group, other)
                weights[group] = union.with_values(union.values[order], kind=union.kind)
        strata = pd.concat([self._strata, other._strata], ignore_index=True)
        return Frame(
            tables,
            self._schema,
            weights,
            strata,
            mass_log=(*self._mass_log, *other._mass_log),
            metadata=self._metadata,
        )

    def select(self, person_mask: np.ndarray | pd.Series) -> "Frame":
        """Subset persons and prune group tables to the ids still referenced.

        Args:
            person_mask: Boolean mask over person rows. A plain array must
                have one element per person row; a :class:`pandas.Series`
                must be index-aligned to the person table.

        Returns:
            A new validated bundle containing the masked persons, the group
            rows they reference, the matching weight slices, and the matching
            strata.

        Raises:
            NotImplementedError: If the frame carries link tables (links are
                a placeholder; link-aware select comes with the full link
                operator).
            ValueError: If the mask is misaligned, non-boolean, or would
                select no persons.
        """
        if self._link_tables:
            raise NotImplementedError(
                "select does not yet prune link tables (links are a "
                "documented placeholder); drop link tables before selecting."
            )
        person = self.person
        if isinstance(person_mask, pd.Series):
            if not person_mask.index.equals(person.index):
                raise ValueError(
                    "person_mask Series must be index-aligned to the person table."
                )
            mask = person_mask.to_numpy()
        else:
            mask = np.asarray(person_mask)
        if mask.dtype != np.bool_:
            raise ValueError(f"person_mask must be boolean, got dtype {mask.dtype}.")
        if mask.shape != (len(person),):
            raise ValueError(
                f"person_mask must have one element per person row "
                f"({len(person)}), got shape {mask.shape}."
            )
        if not mask.any():
            raise ValueError(
                "select would produce an empty bundle; at least one person must remain."
            )

        person_entity = self._schema.person_entity
        tables: dict[str, pd.DataFrame] = {person_entity: person.loc[mask]}
        weights: dict[str, Weights] = {}
        if person_entity in self._weights:
            existing = self._weights[person_entity]
            weights[person_entity] = existing.with_values(
                existing.values[mask], kind=existing.kind
            )
        for group in self._schema.group_entities:
            membership = tables[person_entity][
                self._schema.membership_column(group)
            ].to_numpy()
            referenced = np.unique(membership)
            table = self._tables[group]
            keep = table[self._schema.id_column(group)].isin(referenced).to_numpy()
            tables[group] = table.loc[keep].reset_index(drop=True)
            if group in self._weights:
                existing = self._weights[group]
                weights[group] = existing.with_values(
                    existing.values[keep], kind=existing.kind
                )
        strata = self._strata.loc[mask]
        return Frame(
            tables,
            self._schema,
            weights,
            strata,
            mass_log=self._mass_log,
            metadata=self._metadata,
        )

    # ------------------------------------------------------------------
    # Concat helpers
    # ------------------------------------------------------------------

    def _entity_ids(self, entity: str) -> np.ndarray:
        """The id space of ``entity``: person ids or the group id column."""
        if entity == self._schema.person_entity:
            return self.person[self._schema.person_id_column].to_numpy()
        return self._tables[entity][self._schema.id_column(entity)].to_numpy()

    def _shift_entity_ids(
        self, other_tables: dict[str, pd.DataFrame], entity: str
    ) -> None:
        """Shift ``entity``'s integer ids in ``other_tables`` past this bundle's.

        Applied to the id column (group table or person table) and, for group
        entities, the person membership column, so structure is preserved.

        Raises:
            ValueError: If either id space is not integer-typed.
        """
        person_entity = self._schema.person_entity
        mine = self._entity_ids(entity)
        if entity == person_entity:
            id_frame = other_tables[person_entity]
            id_column = self._schema.person_id_column
        else:
            id_frame = other_tables[entity]
            id_column = self._schema.id_column(entity)
        theirs = id_frame[id_column].to_numpy()
        if not (
            np.issubdtype(mine.dtype, np.integer)
            and np.issubdtype(theirs.dtype, np.integer)
        ):
            raise ValueError(
                f"Cannot concat: id spaces for entity {entity!r} overlap and "
                f"are not integer-typed ({mine.dtype} vs {theirs.dtype}); "
                "remap ids to disjoint spaces before concatenating."
            )
        offset = int(mine.max()) + 1 - int(theirs.min())
        id_frame[id_column] = theirs + offset
        if entity != person_entity:
            membership_column = self._schema.membership_column(entity)
            person = other_tables[person_entity]
            person[membership_column] = person[membership_column] + offset

    def _concat_weights(self, entity: str, other: "Frame") -> Weights:
        """Concatenate two entities' weight vectors; kind is the further one."""
        mine = self._weights[entity]
        theirs = other._weights[entity]
        kinds = {kind.value: kind for kind in (mine.kind, theirs.kind)}
        union_kind = next(
            kinds[name] for name in reversed(_KIND_ORDER) if name in kinds
        )
        return Weights(
            values=np.concatenate([mine.values, theirs.values]),
            kind=union_kind,
        )

    def __repr__(self) -> str:
        sizes = ", ".join(f"{entity}={self.n(entity)}" for entity in self.entities)
        weighted = ", ".join(
            f"{entity}:{self._weights[entity].kind.value}"
            for entity in self.weighted_entities
        )
        n_strata = self._strata.nunique()
        links = f"; links={list(self.links)}" if self._link_tables else ""
        return f"Frame({sizes}; weights[{weighted}]; strata={n_strata}{links})"


@dataclass(frozen=True)
class _FrozenMapping(Mapping[str, Any]):
    """Small pickle-safe immutable mapping used for frame metadata."""

    _items: tuple[tuple[str, Any], ...]

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _freeze_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a recursively immutable copy of frame metadata."""

    if metadata is None:
        return _FrozenMapping(())
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"Frame metadata must be a mapping, got {type(metadata).__name__}."
        )
    frozen: list[tuple[str, Any]] = []
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Frame metadata keys must be non-empty strings.")
        frozen.append((key, _freeze_metadata_value(value, path=key)))
    return _FrozenMapping(tuple(frozen))


def _freeze_metadata_value(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        frozen: list[tuple[str, Any]] = []
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"Frame metadata mapping {path!r} has a non-string/empty key."
                )
            frozen.append(
                (
                    key,
                    _freeze_metadata_value(
                        nested,
                        path=f"{path}.{key}",
                    ),
                )
            )
        return _FrozenMapping(tuple(frozen))
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_metadata_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        return frozenset(
            _freeze_metadata_value(item, path=f"{path}[]") for item in value
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Frame metadata value at {path!r} must be composed of mappings, "
        "sequences, sets, and scalar values; got "
        f"{type(value).__name__}."
    )
