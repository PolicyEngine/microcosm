"""The kernel datatype: a weighted, stratified bundle of entity tables.

A :class:`WeightedBundle` holds one table per entity (person plus the group
entities declared by its :class:`~microframe.schema.EntitySchema`), typed
weights for the weighted entities, and per-person stratum labels. Structure
is established once, at assembly, and validated on every construction:

- every group table's id column is unique, sorted ascending, and contains
  exactly the distinct ids referenced by the person membership column;
- the person table's id column is unique;
- weight vectors match their entity table lengths;
- strata are aligned to the person index;
- column names are globally unique across entity tables (the flattening
  rule rules engines rely on).

All operations return new bundles; a bundle is never mutated in place.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from microframe.schema import EntitySchema
from microframe.weights import Weights, assert_kind_transition

__all__ = ["WeightedBundle", "DEFAULT_STRATUM"]

#: Stratum label assigned when a bundle is constructed without explicit strata.
DEFAULT_STRATUM = "default"

#: Kind rank used to resolve the union kind on concatenation.
_KIND_ORDER = ("design", "importance", "calibrated")


class WeightedBundle:
    """Entity tables + typed weights + strata, with kernel-enforced invariants.

    Args:
        tables: One :class:`pandas.DataFrame` per entity declared by
            ``schema`` (the person entity and every group entity). Tables are
            copied; the caller's frames are never mutated or aliased.
        schema: The entity structure (see
            :class:`~microframe.schema.EntitySchema`).
        weights: Typed weight vectors keyed by entity name. At least one
            entity must carry weights (typically the household).
        strata: Per-person provenance labels, index-aligned to the person
            table. ``None`` assigns the single stratum
            :data:`DEFAULT_STRATUM` to every person.

    Raises:
        TypeError: If a weight value is not a :class:`Weights`, or ``strata``
            is not a :class:`pandas.Series`.
        ValueError: If any kernel invariant fails. Messages name the exact
            entity, column, or ids involved.
    """

    __slots__ = ("_schema", "_strata", "_tables", "_weights")

    def __init__(
        self,
        tables: Mapping[str, pd.DataFrame],
        schema: EntitySchema,
        weights: Mapping[str, Weights],
        strata: pd.Series | None = None,
    ) -> None:
        self._schema = schema
        self._tables = {name: frame.copy() for name, frame in tables.items()}
        self._weights = dict(weights)
        self._strata = self._validated_strata(strata)
        self._validate_tables()
        self._validate_linkage()
        self._validate_global_columns()
        self._validate_weights()

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
            raise ValueError(
                f"Unknown entity table(s): {unknown}; schema declares "
                f"{list(self._schema.entities)}."
            )
        person = self._tables[self._schema.person_entity]
        id_column = self._schema.person_id_column
        if id_column not in person.columns:
            raise ValueError(
                f"Person table must carry the id column {id_column!r}."
            )
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

    def _validate_global_columns(self) -> None:
        owners: dict[str, list[str]] = {}
        for entity in self._schema.entities:
            for column in self._tables[entity].columns:
                owners.setdefault(column, []).append(entity)
        duplicated = {
            column: entities
            for column, entities in owners.items()
            if len(entities) > 1
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
            The stored :class:`~microframe.weights.Weights`.

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
                group
                for group in self._schema.group_entities
                if group in self._weights
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
        require_mass: bool = False,
    ) -> "WeightedBundle":
        """Return a new bundle with ``entity``'s weights replaced.

        Kind transitions are enforced: weights only move forward
        (``design -> importance -> calibrated``; same-kind replacement is
        allowed). With ``require_mass=True`` the replacement must conserve
        total mass against the existing vector.

        Args:
            entity: Entity whose weights to set.
            weights: The replacement vector (length must match the entity
                table).
            require_mass: When ``True``, raise unless the new total mass
                matches the existing total within ``rtol=1e-9``. Requires
                existing weights to compare against.

        Returns:
            A new validated bundle.

        Raises:
            TypeError: If ``weights`` is not a :class:`Weights`.
            ValueError: On a backward kind transition, a mass-conservation
                violation (the message carries both totals), a length
                mismatch, or ``require_mass=True`` without existing weights.
        """
        self.table(entity)  # validates the entity name
        if not isinstance(weights, Weights):
            raise TypeError(
                f"weights must be a Weights instance, got {type(weights).__name__}."
            )
        existing = self._weights.get(entity)
        if existing is not None:
            assert_kind_transition(existing.kind, weights.kind)
            if require_mass:
                try:
                    existing.assert_mass_conserved(weights)
                except ValueError as exc:
                    raise ValueError(
                        f"with_weights({entity!r}) violates mass conservation: "
                        f"{exc}"
                    ) from None
        elif require_mass:
            raise ValueError(
                f"require_mass=True needs existing weights for entity "
                f"{entity!r} to conserve against; none are stored."
            )
        new_weights = dict(self._weights)
        new_weights[entity] = weights
        return WeightedBundle(self._tables, self._schema, new_weights, self._strata)

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
            f"Column {column!r} not found on any entity table "
            f"{list(self.entities)}."
        )

    def concat(self, other: "WeightedBundle") -> "WeightedBundle":
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
            TypeError: If ``other`` is not a :class:`WeightedBundle`.
            ValueError: On schema/column/weight-set mismatch, overlapping
                strata with overlapping id spaces, or non-integer colliding
                ids that cannot be shifted.
        """
        if not isinstance(other, WeightedBundle):
            raise TypeError(
                f"concat expects a WeightedBundle, got {type(other).__name__}."
            )
        if self._schema != other._schema:
            raise ValueError("Cannot concat bundles with different schemas.")
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
            if np.intersect1d(
                self._entity_ids(entity), other._entity_ids(entity)
            ).size
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
                weights[group] = union.with_values(
                    union.values[order], kind=union.kind
                )
        strata = pd.concat([self._strata, other._strata], ignore_index=True)
        return WeightedBundle(tables, self._schema, weights, strata)

    def select(self, person_mask: np.ndarray | pd.Series) -> "WeightedBundle":
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
            ValueError: If the mask is misaligned, non-boolean, or would
                select no persons.
        """
        person = self.person
        if isinstance(person_mask, pd.Series):
            if not person_mask.index.equals(person.index):
                raise ValueError(
                    "person_mask Series must be index-aligned to the person "
                    "table."
                )
            mask = person_mask.to_numpy()
        else:
            mask = np.asarray(person_mask)
        if mask.dtype != np.bool_:
            raise ValueError(
                f"person_mask must be boolean, got dtype {mask.dtype}."
            )
        if mask.shape != (len(person),):
            raise ValueError(
                f"person_mask must have one element per person row "
                f"({len(person)}), got shape {mask.shape}."
            )
        if not mask.any():
            raise ValueError(
                "select would produce an empty bundle; at least one person "
                "must remain."
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
        return WeightedBundle(tables, self._schema, weights, strata)

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

    def _concat_weights(self, entity: str, other: "WeightedBundle") -> Weights:
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
        return (
            f"WeightedBundle({sizes}; weights[{weighted}]; "
            f"strata={n_strata})"
        )
