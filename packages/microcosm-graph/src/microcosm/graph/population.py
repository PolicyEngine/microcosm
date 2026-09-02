"""Immutable population versions and storage-preserving cell patches.

The :class:`~microcosm.frame.Frame` remains the carrier and the authority for
entity linkage, weights, and strata.  This module adds graph-version lineage,
cell ownership, strict declaration dtypes, and a per-node mass ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.frame import Frame, MassChangeRecord, WeightKind, Weights

from .decl import (
    MASS_POLICIES,
    ROWS_ALL,
    Node,
    Owned,
    Ownership,
    StructuralDelta,
)
from .kernel import KernelResult

__all__ = [
    "MassRecord",
    "Population",
    "PopulationError",
    "assert_dtype",
    "dtype_for_token",
    "dtype_matches",
    "owned_ids",
    "patch",
    "population_from_frame",
    "restore_cached_expand",
    "storage_equal",
    "token_for_dtype",
    "weight_cap_receipt",
]

_MASS_RTOL = 1e-9
_WEIGHT_ORDER = (
    WeightKind.DESIGN,
    WeightKind.IMPORTANCE,
    WeightKind.CALIBRATED,
)


class PopulationError(ValueError):
    """A kernel result cannot legally become a population version."""


def dtype_for_token(token: str) -> np.dtype | pd.api.extensions.ExtensionDtype:
    """Return the one pandas dtype represented by a declaration token."""

    if token == "boolean":
        return pd.BooleanDtype()
    if token == "Int64":
        return pd.Int64Dtype()
    if token == "string":
        # Pin Python storage so an optional pyarrow installation cannot change
        # a graph artifact's physical dtype.
        return pd.StringDtype(storage="python", na_value=pd.NA)
    if token in {"bool", "int32", "int64", "float32", "float64"}:
        return np.dtype(token)
    raise PopulationError(f"Unknown graph dtype token {token!r}.")


def token_for_dtype(dtype: object) -> str:
    """Return the graph token for a supported pandas/NumPy dtype."""

    if isinstance(dtype, pd.BooleanDtype):
        return "boolean"
    if isinstance(dtype, pd.Int64Dtype):
        return "Int64"
    if isinstance(dtype, pd.StringDtype):
        return "string"
    normalized = np.dtype(dtype)
    token = normalized.name
    if token in {"bool", "int32", "int64", "float32", "float64"}:
        return token
    raise PopulationError(f"Unsupported population dtype {dtype!s}.")


def dtype_matches(value: object, token: str) -> bool:
    """Whether a Series, array, or dtype exactly matches ``token``."""

    dtype = getattr(value, "dtype", value)
    try:
        return token_for_dtype(dtype) == token
    except (PopulationError, TypeError):
        return False


def assert_dtype(value: object, token: str, *, label: str) -> None:
    """Raise :class:`PopulationError` unless ``value`` has ``token``'s dtype."""

    dtype = getattr(value, "dtype", value)
    if not dtype_matches(dtype, token):
        raise PopulationError(
            f"{label} has dtype {dtype!s}; declaration requires {token!r}."
        )


@dataclass(frozen=True)
class MassRecord:
    """Mass before and after one population-changing graph node."""

    node_id: str
    operation: str
    policy: str
    before_total: float
    after_total: float
    before_by_stratum: tuple[tuple[object, float], ...]
    after_by_stratum: tuple[tuple[object, float], ...]
    entity: str | None = None

    @property
    def before_strata(self) -> Mapping[object, float]:
        return MappingProxyType(dict(self.before_by_stratum))

    @property
    def after_strata(self) -> Mapping[object, float]:
        return MappingProxyType(dict(self.after_by_stratum))

    @property
    def old_total(self) -> float:
        """Compatibility spelling used by :mod:`microcosm.frame`."""

        return self.before_total

    @property
    def new_total(self) -> float:
        """Compatibility spelling used by :mod:`microcosm.frame`."""

        return self.after_total


@dataclass(frozen=True)
class Population:
    """One immutable graph view over a validated :class:`Frame`."""

    frame: Frame
    version: str
    owners: Mapping[tuple[str, str], str]
    weight_kind: Mapping[str, WeightKind]
    mass_ledger: tuple[MassRecord, ...] = field(default_factory=tuple)
    design_weights: Mapping[str, np.ndarray] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError("Population.frame must be a Frame.")
        if not isinstance(self.version, str) or not self.version:
            raise PopulationError("Population.version must be a non-empty string.")

        expected_cells = {
            (entity, str(column))
            for entity in self.frame.entities
            for column in self.frame.table(entity).columns
        }
        copied_owners = dict(self.owners)
        if set(copied_owners) != expected_cells:
            missing = sorted(expected_cells - set(copied_owners))
            extra = sorted(set(copied_owners) - expected_cells)
            raise PopulationError(
                "Population ownership must cover every frame column exactly "
                f"(missing={missing}, extra={extra})."
            )
        if any(
            not isinstance(owner, str) or not owner for owner in copied_owners.values()
        ):
            raise PopulationError("Population owner ids must be non-empty strings.")

        expected_kinds = {
            entity: self.frame.weights_for(entity).kind
            for entity in self.frame.weighted_entities
        }
        copied_kinds = dict(self.weight_kind)
        if copied_kinds != expected_kinds:
            raise PopulationError(
                "Population.weight_kind must exactly describe the Frame's explicit "
                f"weights: expected {expected_kinds}, got {copied_kinds}."
            )
        object.__setattr__(self, "owners", MappingProxyType(copied_owners))
        object.__setattr__(self, "weight_kind", MappingProxyType(copied_kinds))
        object.__setattr__(self, "mass_ledger", tuple(self.mass_ledger))
        frozen_design: dict[str, np.ndarray] = {}
        for entity, raw_values in self.design_weights.items():
            if entity not in self.frame.entities:
                raise PopulationError(
                    f"Design-weight anchor names unknown entity {entity!r}."
                )
            values = np.asarray(raw_values, dtype=np.float64)
            expected_length = self.frame.n(entity)
            if values.shape != (expected_length,):
                raise PopulationError(
                    f"Design-weight anchor for {entity!r} has shape {values.shape}; "
                    f"expected ({expected_length},)."
                )
            if not np.isfinite(values).all() or (values < 0).any():
                raise PopulationError(
                    f"Design-weight anchor for {entity!r} must be finite and "
                    "non-negative."
                )
            # A bytes-backed array cannot be made writable again by a caller.
            frozen_design[entity] = np.frombuffer(values.tobytes(), dtype=np.float64)
        object.__setattr__(self, "design_weights", MappingProxyType(frozen_design))

    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        version: str,
        owners: Mapping[tuple[str, str], str] | None = None,
        *,
        mass_ledger: tuple[MassRecord, ...] = (),
        design_weights: Mapping[str, np.ndarray] | None = None,
    ) -> Population:
        """Create a version, assigning every loaded column to ``version`` by default."""

        resolved_owners = (
            {
                (entity, str(column)): version
                for entity in frame.entities
                for column in frame.table(entity).columns
            }
            if owners is None
            else dict(owners)
        )
        kinds = {
            entity: frame.weights_for(entity).kind for entity in frame.weighted_entities
        }
        if design_weights is None:
            design_weights = {
                entity: frame.weights_for(entity).values
                for entity in frame.weighted_entities
                if frame.weights_for(entity).kind is WeightKind.DESIGN
            }
        return cls(
            frame,
            version,
            resolved_owners,
            kinds,
            mass_ledger,
            design_weights,
        )


def population_from_frame(
    frame: Frame,
    version: str,
    owners: Mapping[tuple[str, str], str] | None = None,
    *,
    mass_ledger: tuple[MassRecord, ...] = (),
    design_weights: Mapping[str, np.ndarray] | None = None,
) -> Population:
    """Functional spelling of :meth:`Population.from_frame`."""

    return Population.from_frame(
        frame,
        version,
        owners,
        mass_ledger=mass_ledger,
        design_weights=design_weights,
    )


def restore_cached_expand(
    population: Population, node: Node, result: KernelResult
) -> Population:
    """Restore a previously validated EXPAND frame against its keyed base.

    A cache record stores the executor-materialized Frame, never a
    kernel-returned population.  Its content hash and node key already bind it
    to this base version.  This function re-establishes graph ownership,
    design-weight ancestry, and the executor mass ledger without relaxing the
    miss-path lineage validation.
    """

    if node.structural is not StructuralDelta.EXPAND or result.frame is None:
        raise PopulationError("restore_cached_expand requires an EXPAND Frame.")
    frame = result.frame
    if frame.schema != population.frame.schema:
        raise PopulationError(f"Cached EXPAND node {node.id!r} changed schema.")
    for entity in frame.entities:
        id_column = frame.schema.entity_id_column(entity)
        before_ids = pd.Index(population.frame.table(entity)[id_column])
        after_ids = pd.Index(frame.table(entity)[id_column])
        if not before_ids.isin(after_ids).all():
            raise PopulationError(
                f"Cached EXPAND node {node.id!r} dropped incumbent {entity!r} ids."
            )
    _assert_expand_weights(population, frame, node, result)

    design_weights: dict[str, np.ndarray] = {}
    for entity, old_anchor in population.design_weights.items():
        id_column = frame.schema.entity_id_column(entity)
        before_ids = pd.Index(population.frame.table(entity)[id_column])
        after_table = frame.table(entity)
        after_ids = pd.Index(after_table[id_column])
        positions = before_ids.get_indexer(after_ids)
        values = np.empty(len(after_ids), dtype=np.float64)
        retained = positions >= 0
        values[retained] = old_anchor[positions[retained]]
        if not retained.all():
            source_column = f"{entity}_source_id"
            if source_column in after_table:
                source_positions = before_ids.get_indexer(
                    after_table.loc[~retained, source_column]
                )
                if (source_positions < 0).any():
                    raise PopulationError(
                        f"Cached EXPAND node {node.id!r} has unknown design "
                        f"lineage in {source_column!r}."
                    )
                values[~retained] = old_anchor[source_positions]
            else:
                current = frame.weights_for(entity)
                if current.kind is not WeightKind.DESIGN:
                    raise PopulationError(
                        f"Cached EXPAND node {node.id!r} cannot restore design "
                        f"lineage for new {entity!r} ids."
                    )
                values[~retained] = current.values[~retained]
        design_weights[entity] = values

    ledger = (
        *population.mass_ledger,
        _mass_record(population.frame, frame, node, result, _mass_policy(node)),
    )
    owners = {
        (entity, str(column)): node.id
        for entity in frame.entities
        for column in frame.table(entity).columns
    }
    return Population.from_frame(
        frame,
        node.id,
        owners,
        mass_ledger=ledger,
        design_weights=design_weights,
    )


def owned_ids(population: Population, owned: Owned) -> pd.Index:
    """Entity ids at exactly the positions covered by an ``Owned`` declaration."""

    table = population.frame.table(owned.entity)
    id_column = population.frame.schema.entity_id_column(owned.entity)
    if owned.rows == ROWS_ALL:
        mask = np.ones(len(table), dtype=np.bool_)
    else:
        if owned.rows not in table:
            raise PopulationError(
                f"Node-owned mask {owned.entity}.{owned.rows} is absent from the frame."
            )
        mask_series = table[owned.rows]
        if not (
            pd.api.types.is_bool_dtype(mask_series.dtype)
            or isinstance(mask_series.dtype, pd.BooleanDtype)
        ):
            raise PopulationError(
                f"Owned-row mask {owned.entity}.{owned.rows} must be boolean, "
                f"got {mask_series.dtype!s}."
            )
        if mask_series.isna().any():
            raise PopulationError(
                f"Owned-row mask {owned.entity}.{owned.rows} contains nulls."
            )
        mask = mask_series.to_numpy(dtype=np.bool_, copy=True)
    return pd.Index(table.loc[mask, id_column].to_numpy(copy=True), name=id_column)


def storage_equal(
    left: pd.Series,
    right: pd.Series,
    positions: np.ndarray | pd.Series | None = None,
) -> bool:
    """Compare physical values and nullable masks exactly, including float bits."""

    if left.dtype != right.dtype or len(left) != len(right):
        return False
    if positions is None:
        selected = np.ones(len(left), dtype=np.bool_)
    else:
        selected = np.asarray(positions)
        if selected.dtype != np.bool_ or selected.shape != (len(left),):
            raise ValueError("storage_equal positions must be a row-aligned bool mask.")
    return _storage_parts(left, selected) == _storage_parts(right, selected)


def patch(population: Population, node: Node, result: KernelResult) -> Population:
    """Validate and apply one node result without mutating ``population``.

    ``EXPAND`` has one runtime convention beyond the frozen declaration
    surface.  A kernel still never receives or returns a population: its
    ``columns`` map carries one target-id-indexed source-id lineage Series per
    entity, the remapped person membership Series, and the cell overlays
    enumerated by ``params['expand_cells']``.  ``result.weights`` is the full
    target vector named by ``params['expand_weight_entity']`` and
    ``params['expand_weight_kind']``.  The executor validates those pieces and
    carries every other cell from the named source rows here.

    The convention is deliberately encoded in existing ``KernelResult``
    fields so the frozen ``decl.py`` and ``kernel.py`` interfaces remain
    unchanged.  It is content-addressed because every contract component is a
    normative node parameter.
    """

    if node.structural is StructuralDelta.CREATE:
        raise PopulationError(
            "CREATE has no incumbent Population; use Population.from_frame()."
        )
    if node.weights is not None and node.structural is StructuralDelta.NONE:
        raise PopulationError(
            f"Node {node.id!r} declares a weight transition without a structural "
            "population version."
        )
    _assert_no_ordinary_structural_outputs(population, node)
    expected_columns = {(owned.entity, owned.column) for owned in node.outputs}
    lineage_expand = node.structural is StructuralDelta.EXPAND and result.frame is None
    if not lineage_expand and set(result.columns) != expected_columns:
        raise PopulationError(
            f"Node {node.id!r} returned columns {sorted(result.columns)}; "
            f"expected exactly {sorted(expected_columns)}."
        )

    before = population.frame
    if node.structural is StructuralDelta.NONE:
        if result.frame is not None:
            raise PopulationError(f"Non-structural node {node.id!r} returned a Frame.")
        frame, owners = _patch_columns(population, node, result)
    elif lineage_expand:
        frame, owners = _patch_expand(population, node, result)
    else:
        frame, owners = _patch_structural(population, node, result)

    expand_weight_entity = _expand_weight_entity(node) if lineage_expand else None
    if expand_weight_entity is not None:
        _assert_carried_weights(
            population.frame,
            frame,
            node,
            transitioning=expand_weight_entity,
        )
        _assert_expand_weights(population, frame, node, result)
    elif node.weights is not None:
        _assert_carried_weights(
            population.frame,
            frame,
            node,
            transitioning=node.weights.entity,
        )
        frame = _apply_weight_transition(population, frame, node, result)
    elif result.weights is not None:
        raise PopulationError(
            f"Node {node.id!r} returned weights without declaring a transition."
        )
    else:
        _assert_carried_weights(population.frame, frame, node)

    frame = _append_frame_mass_log(population.frame, frame, node, result)

    design_weights = _carry_design_weights(population, frame, node, result)
    _assert_design_weight_cap(frame, design_weights, node)

    ledger = population.mass_ledger
    if node.structural is not StructuralDelta.NONE or node.weights is not None:
        policy = _mass_policy(node)
        record = _mass_record(before, frame, node, result, policy)
        ledger = (*ledger, record)

    return Population.from_frame(
        frame,
        node.id if node.structural is not StructuralDelta.NONE else population.version,
        owners,
        mass_ledger=ledger,
        design_weights=design_weights,
    )


def _expand_cells(node: Node) -> tuple[tuple[str, str, str], ...]:
    """Return the normative ``(entity, column, dtype)`` EXPAND overlays."""

    raw = node.params.get("expand_cells")
    if not isinstance(raw, tuple):
        raise PopulationError(
            f"EXPAND node {node.id!r} needs tuple params['expand_cells']."
        )
    cells: list[tuple[str, str, str]] = []
    for item in raw:
        if (
            not isinstance(item, tuple)
            or len(item) != 3
            or any(not isinstance(part, str) or not part for part in item)
        ):
            raise PopulationError(
                f"EXPAND node {node.id!r} has malformed expand_cells entry {item!r}."
            )
        entity, column, dtype = item
        # Reuse the declaration token validator without importing frozen
        # declaration internals into this runtime convention.
        dtype_for_token(dtype)
        cells.append((entity, column, dtype))
    if len({(entity, column) for entity, column, _ in cells}) != len(cells):
        raise PopulationError(f"EXPAND node {node.id!r} repeats an expanded cell.")
    return tuple(cells)


def _expand_weight_entity(node: Node) -> str | None:
    raw = node.params.get("expand_weight_entity")
    if node.structural is not StructuralDelta.EXPAND:
        return None
    if not isinstance(raw, str) or not raw:
        raise PopulationError(
            f"EXPAND node {node.id!r} needs params['expand_weight_entity']."
        )
    return raw


def _patch_expand(
    population: Population, node: Node, result: KernelResult
) -> tuple[Frame, dict[tuple[str, str], str]]:
    """Materialize a source-lineage EXPAND result in the executor."""

    before = population.frame
    if before.links:
        raise PopulationError(
            f"EXPAND node {node.id!r} cannot yet carry association link tables."
        )
    cells = _expand_cells(node)
    for entity, _, _ in cells:
        if entity not in before.entities:
            raise PopulationError(
                f"EXPAND node {node.id!r} names unknown entity {entity!r}."
            )

    structural_coordinates = {
        (entity, before.schema.entity_id_column(entity)) for entity in before.entities
    }
    person = before.schema.person_entity
    membership_coordinates = {
        (person, before.schema.membership_column(group))
        for group in before.schema.group_entities
    }
    cell_coordinates = {(entity, column) for entity, column, _ in cells}
    expected = structural_coordinates | membership_coordinates | cell_coordinates
    if set(result.columns) != expected:
        raise PopulationError(
            f"EXPAND node {node.id!r} returned columns {sorted(result.columns)}; "
            f"its lineage contract requires exactly {sorted(expected)}."
        )

    tables: dict[str, pd.DataFrame] = {}
    lineage_positions: dict[str, np.ndarray] = {}
    target_ids: dict[str, pd.Index] = {}
    for entity in before.entities:
        id_column = before.schema.entity_id_column(entity)
        lineage = result.columns[(entity, id_column)]
        if not isinstance(lineage, pd.Series):
            raise PopulationError(
                f"EXPAND node {node.id!r} lineage for {entity!r} is not a Series."
            )
        source_table = before.table(entity)
        source_ids = pd.Index(source_table[id_column].to_numpy(copy=True))
        targets = pd.Index(lineage.index, name=id_column)
        if not targets.is_unique:
            raise PopulationError(
                f"EXPAND node {node.id!r} repeats target {entity!r} ids."
            )
        if len(targets) < len(source_ids):
            raise PopulationError(
                f"EXPAND node {node.id!r} has fewer target than source {entity!r} rows."
            )
        positions = source_ids.get_indexer(lineage.to_numpy(copy=False))
        if (positions < 0).any():
            bad = lineage.iloc[np.flatnonzero(positions < 0)[:5]].tolist()
            raise PopulationError(
                f"EXPAND node {node.id!r} lineage names unknown {entity!r} "
                f"source ids {bad}."
            )
        target_position = targets.get_indexer(source_ids)
        if (target_position < 0).any() or not np.array_equal(
            lineage.iloc[target_position].to_numpy(copy=False),
            source_ids.to_numpy(copy=False),
        ):
            raise PopulationError(
                f"EXPAND node {node.id!r} must retain every incumbent "
                f"{entity!r} id with self-lineage."
            )
        carried = source_table.iloc[positions].reset_index(drop=True)
        replacement_ids = pd.Series(
            targets.to_numpy(copy=True), dtype=source_table[id_column].dtype
        )
        if len(replacement_ids) != len(carried):
            raise PopulationError(
                f"EXPAND node {node.id!r} lineage index/value lengths disagree "
                f"for {entity!r}."
            )
        carried[id_column] = replacement_ids.array
        tables[entity] = carried
        lineage_positions[entity] = positions
        target_ids[entity] = targets

    person_table = tables[person]
    for group in before.schema.group_entities:
        membership = before.schema.membership_column(group)
        incoming = result.columns[(person, membership)]
        if not isinstance(incoming, pd.Series):
            raise PopulationError(
                f"EXPAND node {node.id!r} membership {membership!r} is not a Series."
            )
        expected_ids = target_ids[person]
        if (
            not incoming.index.is_unique
            or len(incoming) != len(expected_ids)
            or set(incoming.index) != set(expected_ids)
        ):
            raise PopulationError(
                f"EXPAND node {node.id!r} membership {membership!r} must name "
                "every target person exactly once."
            )
        aligned = incoming.reindex(expected_ids)
        assert_dtype(
            aligned,
            token_for_dtype(before.table(person)[membership].dtype),
            label=f"EXPAND node {node.id!r} membership {membership}",
        )
        group_ids = set(tables[group][before.schema.entity_id_column(group)].tolist())
        unknown = set(aligned.tolist()) - group_ids
        if unknown:
            raise PopulationError(
                f"EXPAND node {node.id!r} membership {membership!r} names "
                f"unknown target ids {sorted(unknown)[:5]}."
            )
        person_table[membership] = aligned.array

    for entity, column, dtype in cells:
        incoming = result.columns[(entity, column)]
        if not isinstance(incoming, pd.Series):
            raise PopulationError(
                f"EXPAND node {node.id!r} cell {entity}.{column} is not a Series."
            )
        ids = target_ids[entity]
        if (
            not incoming.index.is_unique
            or len(incoming) != len(ids)
            or set(incoming.index) != set(ids)
        ):
            raise PopulationError(
                f"EXPAND node {node.id!r} cell {entity}.{column} must name "
                "every target id exactly once."
            )
        aligned = incoming.reindex(ids)
        assert_dtype(
            aligned,
            dtype,
            label=f"EXPAND node {node.id!r} cell {entity}.{column}",
        )
        tables[entity][column] = aligned.array

    weight_entity = _expand_weight_entity(node)
    assert weight_entity is not None
    if weight_entity not in before.weighted_entities:
        raise PopulationError(
            f"EXPAND node {node.id!r} cannot replace inherited weights for "
            f"{weight_entity!r}."
        )
    if result.weights is None:
        raise PopulationError(f"EXPAND node {node.id!r} returned no weights.")
    weights: dict[str, Weights] = {}
    for entity in before.weighted_entities:
        if entity == weight_entity:
            weights[entity] = result.weights
            continue
        old = before.weights_for(entity)
        weights[entity] = Weights(old.values[lineage_positions[entity]], kind=old.kind)

    person_positions = lineage_positions[person]
    strata = pd.Series(
        before.strata.iloc[person_positions].array.copy(),
        index=tables[person].index,
        name=before.strata.name,
        dtype=before.strata.dtype,
    )
    frame = Frame(
        tables,
        before.schema,
        weights,
        strata,
        mass_log=before.mass_log,
        metadata=before.metadata,
    )
    owners = {
        (entity, str(column)): node.id
        for entity in frame.entities
        for column in frame.table(entity).columns
    }
    return frame, owners


def _assert_expand_weights(
    population: Population, frame: Frame, node: Node, result: KernelResult
) -> None:
    entity = _expand_weight_entity(node)
    assert entity is not None
    raw_kind = node.params.get("expand_weight_kind")
    if not isinstance(raw_kind, str):
        raise PopulationError(
            f"EXPAND node {node.id!r} needs params['expand_weight_kind']."
        )
    try:
        declared_kind = WeightKind(raw_kind)
    except ValueError as error:
        raise PopulationError(
            f"EXPAND node {node.id!r} declares unknown weight kind {raw_kind!r}."
        ) from error
    assert result.weights is not None
    if result.weights.kind is not declared_kind:
        raise PopulationError(
            f"EXPAND node {node.id!r} returned {result.weights.kind.value!r} "
            f"weights, not declared {raw_kind!r}."
        )
    if len(result.weights.values) != frame.n(entity):
        raise PopulationError(
            f"EXPAND node {node.id!r} returned {len(result.weights.values)} "
            f"weights for {frame.n(entity)} target {entity!r} rows."
        )
    installed = frame.weights_for(entity)
    if installed.kind is not declared_kind or not np.array_equal(
        installed.values, result.weights.values
    ):
        raise PopulationError(
            f"EXPAND node {node.id!r} did not install its declared weights."
        )


def _patch_columns(
    population: Population, node: Node, result: KernelResult
) -> tuple[Frame, dict[tuple[str, str], str]]:
    tables = _copied_tables(population.frame)
    owners = dict(population.owners)
    for owned in node.outputs:
        table = tables[owned.entity]
        ids = owned_ids(population, owned)
        incoming = result.columns[(owned.entity, owned.column)]
        if not isinstance(incoming, pd.Series):
            raise PopulationError(
                f"Node {node.id!r} output {owned.entity}.{owned.column} is not a Series."
            )
        assert_dtype(
            incoming,
            owned.dtype,
            label=f"Node {node.id!r} output {owned.entity}.{owned.column}",
        )
        if (
            not incoming.index.is_unique
            or len(incoming) != len(ids)
            or set(incoming.index) != set(ids)
        ):
            raise PopulationError(
                f"Node {node.id!r} output {owned.entity}.{owned.column} ids must "
                "equal the declared owned ids exactly."
            )
        incoming = incoming.reindex(ids)
        if owned.ownership is Ownership.ABSENT and not incoming.isna().all():
            raise PopulationError(
                f"Node {node.id!r} declared {owned.entity}.{owned.column} ABSENT "
                "but returned a non-null value."
            )

        id_column = population.frame.schema.entity_id_column(owned.entity)
        entity_ids = table[id_column]
        owned_mask = entity_ids.isin(ids).to_numpy(dtype=np.bool_)
        if owned.column in table:
            incumbent = table[owned.column].copy(deep=True)
            assert_dtype(
                incumbent,
                owned.dtype,
                label=f"Incumbent {owned.entity}.{owned.column}",
            )
        else:
            incumbent = _empty_column(len(table), owned.dtype, owned_mask)
            table[owned.column] = incumbent

        positions = pd.Series(
            np.arange(len(table), dtype=np.int64), index=entity_ids.to_numpy()
        ).loc[ids]
        updated = table[owned.column].copy(deep=True)
        updated.iloc[positions.to_numpy(dtype=np.int64)] = incoming.array
        assert_dtype(
            updated,
            owned.dtype,
            label=f"Patched {owned.entity}.{owned.column}",
        )
        if not storage_equal(incumbent, updated, ~owned_mask):
            raise PopulationError(
                f"Node {node.id!r} changed non-owned storage in "
                f"{owned.entity}.{owned.column}."
            )
        table[owned.column] = updated
        owners[(owned.entity, owned.column)] = node.id

    return _rebuild_frame(population.frame, tables), owners


def _patch_structural(
    population: Population, node: Node, result: KernelResult
) -> tuple[Frame, dict[tuple[str, str], str]]:
    frame = result.frame
    if frame is None:
        if node.structural is StructuralDelta.REWEIGHT and result.weights is not None:
            frame = population.frame
        else:
            raise PopulationError(
                f"Structural node {node.id!r} ({node.structural.value}) must return "
                "a Frame."
            )
    if frame.schema != population.frame.schema:
        raise PopulationError(f"Structural node {node.id!r} changed the Frame schema.")
    for entity in population.frame.entities:
        before_table = population.frame.table(entity)
        after_table = frame.table(entity)
        if tuple(before_table.columns) != tuple(after_table.columns):
            raise PopulationError(
                f"Structural node {node.id!r} changed columns on {entity!r}."
            )
        for column in before_table:
            if before_table[column].dtype != after_table[column].dtype:
                raise PopulationError(
                    f"Structural node {node.id!r} changed dtype of {entity}.{column}."
                )
        id_column = frame.schema.entity_id_column(entity)
        old_ids = before_table[id_column].to_numpy(copy=False)
        new_ids = after_table[id_column].to_numpy(copy=False)
        old_set = set(old_ids.tolist())
        new_set = set(new_ids.tolist())
        if node.structural is StructuralDelta.FILTER and not new_set <= old_set:
            raise PopulationError(f"FILTER node {node.id!r} introduced {entity!r} ids.")
        if node.structural is StructuralDelta.EXPAND and not old_set <= new_set:
            raise PopulationError(
                f"EXPAND node {node.id!r} dropped original {entity!r} ids."
            )
        if node.structural is StructuralDelta.REWEIGHT and not np.array_equal(
            old_ids, new_ids
        ):
            raise PopulationError(f"REWEIGHT node {node.id!r} changed {entity!r} ids.")

        if node.structural is StructuralDelta.FILTER:
            retained_ids = new_ids
        else:
            # EXPAND must carry every incumbent row forward; REWEIGHT has the
            # same rows.  New EXPAND rows are structural output and therefore
            # have no incumbent storage to protect.
            retained_ids = old_ids
        before_index = pd.Index(old_ids)
        after_index = pd.Index(new_ids)
        before_positions = before_index.get_indexer(retained_ids)
        after_positions = after_index.get_indexer(retained_ids)
        if (before_positions < 0).any() or (after_positions < 0).any():
            raise PopulationError(
                f"Structural node {node.id!r} could not align retained {entity!r} ids."
            )
        for column in before_table:
            incumbent = (
                before_table[column].iloc[before_positions].reset_index(drop=True)
            )
            carried = after_table[column].iloc[after_positions].reset_index(drop=True)
            if not storage_equal(incumbent, carried):
                raise PopulationError(
                    f"Structural node {node.id!r} changed carried storage in "
                    f"{entity}.{column}."
                )
    owners = {
        (entity, str(column)): node.id
        for entity in frame.entities
        for column in frame.table(entity).columns
    }
    return frame, owners


def _apply_weight_transition(
    population: Population, frame: Frame, node: Node, result: KernelResult
) -> Frame:
    transition = node.weights
    assert transition is not None
    if result.weights is None:
        raise PopulationError(
            f"Node {node.id!r} declares a weight transition but returned no weights."
        )
    if transition.entity not in population.frame.weighted_entities:
        raise PopulationError(
            f"Node {node.id!r} cannot transition inherited weights for "
            f"{transition.entity!r}; explicit weights are required."
        )
    old = population.frame.weights_for(transition.entity)
    declared_kind = WeightKind(transition.to_kind)
    # Forward moves only, matching the Frame kernel's own rule: design may go
    # straight to calibrated (the UK pipeline calibrates design weights), and
    # nothing moves backwards or stays in place.
    if _WEIGHT_ORDER.index(declared_kind) <= _WEIGHT_ORDER.index(old.kind):
        raise PopulationError(
            f"Node {node.id!r} weight transition must move forward: "
            f"{old.kind.value!r} -> {transition.to_kind!r} does not."
        )
    if result.weights.kind is not declared_kind:
        raise PopulationError(
            f"Node {node.id!r} declared {transition.to_kind!r} weights but the "
            f"kernel returned {result.weights.kind.value!r}."
        )

    try:
        current = frame.weights_for(transition.entity)
    except ValueError:
        current = None
    if current is not None and current.kind is declared_kind:
        if not np.array_equal(current.values, result.weights.values):
            raise PopulationError(
                f"Node {node.id!r} returned a Frame and weights with different values."
            )
        return frame
    if current is not None and current.kind is not old.kind:
        raise PopulationError(
            f"Node {node.id!r} returned an unexpected intermediate weight kind "
            f"{current.kind.value!r}."
        )
    return _replace_weights(frame, transition.entity, result.weights)


def _replace_weights(frame: Frame, entity: str, replacement: Weights) -> Frame:
    weights = {
        weighted: frame.weights_for(weighted) for weighted in frame.weighted_entities
    }
    weights[entity] = replacement
    return Frame(
        _copied_tables(frame),
        frame.schema,
        weights,
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _assert_no_ordinary_structural_outputs(population: Population, node: Node) -> None:
    if node.structural is not StructuralDelta.NONE:
        return
    schema = population.frame.schema
    structural = {
        (entity, schema.entity_id_column(entity)) for entity in schema.entities
    }
    structural.update(
        (schema.person_entity, schema.membership_column(group))
        for group in schema.group_entities
    )
    for owned in node.outputs:
        if (owned.entity, owned.column) in structural:
            raise PopulationError(
                f"Node {node.id!r} cannot own structural column "
                f"{owned.entity}.{owned.column}."
            )


def _assert_carried_weights(
    before: Frame,
    after: Frame,
    node: Node,
    *,
    transitioning: str | None = None,
) -> None:
    """Protect explicit weight topology and retained values outside a transition."""

    excluded = set() if transitioning is None else {transitioning}
    before_entities = set(before.weighted_entities) - excluded
    after_entities = set(after.weighted_entities) - excluded
    if before_entities != after_entities:
        raise PopulationError(
            f"Node {node.id!r} changed explicit weighted entities without a "
            f"WeightTransition: {sorted(before_entities)} -> {sorted(after_entities)}."
        )
    for entity in sorted(before_entities):
        old = before.weights_for(entity)
        new = after.weights_for(entity)
        if old.kind is not new.kind:
            raise PopulationError(
                f"Node {node.id!r} changed weight kind for {entity!r} without "
                "declaring a WeightTransition."
            )
        id_column = before.schema.entity_id_column(entity)
        before_ids = pd.Index(before.table(entity)[id_column])
        after_ids = pd.Index(after.table(entity)[id_column])
        retained = (
            after_ids if node.structural is StructuralDelta.FILTER else before_ids
        )
        before_positions = before_ids.get_indexer(retained)
        after_positions = after_ids.get_indexer(retained)
        if (before_positions < 0).any() or (after_positions < 0).any():
            raise PopulationError(
                f"Node {node.id!r} could not align carried weights for {entity!r}."
            )
        before_values = np.ascontiguousarray(old.values[before_positions])
        after_values = np.ascontiguousarray(new.values[after_positions])
        if (
            before_values.dtype != after_values.dtype
            or before_values.shape != after_values.shape
            or before_values.tobytes() != after_values.tobytes()
        ):
            raise PopulationError(
                f"Node {node.id!r} changed carried weights for {entity!r} without "
                "declaring a WeightTransition."
            )


def _carry_design_weights(
    population: Population,
    frame: Frame,
    node: Node,
    result: KernelResult,
) -> dict[str, np.ndarray]:
    """Align original design weights to the new version by stable entity id."""

    carried: dict[str, np.ndarray] = {}
    for entity, old_anchor in population.design_weights.items():
        id_column = frame.schema.entity_id_column(entity)
        before_ids = pd.Index(population.frame.table(entity)[id_column])
        after_ids = pd.Index(frame.table(entity)[id_column])
        if node.structural is StructuralDelta.EXPAND and result.frame is None:
            lineage = result.columns.get((entity, id_column))
            if not isinstance(lineage, pd.Series):
                raise PopulationError(
                    f"EXPAND node {node.id!r} has no design lineage for {entity!r}."
                )
            before_positions = before_ids.get_indexer(
                lineage.reindex(after_ids).to_numpy(copy=False)
            )
        else:
            before_positions = before_ids.get_indexer(after_ids)
        values = np.empty(len(after_ids), dtype=np.float64)
        retained = before_positions >= 0
        values[retained] = old_anchor[before_positions[retained]]
        if not retained.all():
            if node.structural is not StructuralDelta.EXPAND:
                raise PopulationError(
                    f"Node {node.id!r} introduced {entity!r} ids outside EXPAND."
                )
            try:
                current = frame.weights_for(entity)
            except ValueError as error:
                raise PopulationError(
                    f"EXPAND node {node.id!r} has no design anchor for new "
                    f"{entity!r} ids."
                ) from error
            if current.kind is not WeightKind.DESIGN:
                raise PopulationError(
                    f"EXPAND node {node.id!r} cannot anchor new {entity!r} ids "
                    f"from {current.kind.value!r} weights; explicit design weights "
                    "are required."
                )
            values[~retained] = current.values[~retained]
        carried[entity] = values
    return carried


def _append_frame_mass_log(
    before: Frame,
    frame: Frame,
    node: Node,
    result: KernelResult,
) -> Frame:
    """Append transform-authored ``Frame.mass_log`` records from a receipt.

    Graph mass accounting remains executor-authored in ``Population``'s
    ledger.  This separate append preserves legacy Frame content identity for
    stages whose public Frame contract records a justified mass change or an
    explicit conservation check.
    """

    raw_records = result.receipt.get("frame_mass_log_append", ())
    if raw_records in (None, ()):  # normalized JSON cache receipts use lists
        return frame
    if not isinstance(raw_records, list | tuple):
        raise PopulationError(
            f"Node {node.id!r} receipt['frame_mass_log_append'] must be a list."
        )
    records: list[MassChangeRecord] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise PopulationError(
                f"Node {node.id!r} frame mass record {index} is not a mapping."
            )
        expected = {
            "entity",
            "old_total",
            "new_total",
            "declared_factor",
            "reason",
        }
        if set(raw) != expected:
            raise PopulationError(
                f"Node {node.id!r} frame mass record {index} fields are "
                f"{sorted(raw)}, not {sorted(expected)}."
            )
        entity = raw["entity"]
        reason = raw["reason"]
        factor = raw["declared_factor"]
        if not isinstance(entity, str) or entity not in frame.weighted_entities:
            raise PopulationError(
                f"Node {node.id!r} frame mass record {index} names unknown "
                f"weighted entity {entity!r}."
            )
        if not isinstance(reason, str) or not reason.strip():
            raise PopulationError(
                f"Node {node.id!r} frame mass record {index} needs a reason."
            )
        if factor is not None and (
            isinstance(factor, bool) or not isinstance(factor, int | float)
        ):
            raise PopulationError(
                f"Node {node.id!r} frame mass record {index} has invalid factor."
            )
        numeric: list[float] = []
        for field_name in ("old_total", "new_total"):
            value = raw[field_name]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise PopulationError(
                    f"Node {node.id!r} frame mass record {index}.{field_name} must "
                    "be numeric."
                )
            converted = float(value)
            if not np.isfinite(converted):
                raise PopulationError(
                    f"Node {node.id!r} frame mass record "
                    f"{index}.{field_name} is not finite."
                )
            numeric.append(converted)
        records.append(
            MassChangeRecord(
                entity=entity,
                old_total=numeric[0],
                new_total=numeric[1],
                declared_factor=None if factor is None else float(factor),
                reason=reason,
            )
        )

    # The legacy records describe the stage boundary, not arbitrary numbers.
    first = records[0]
    last = records[-1]
    before_total = before.weights_for(first.entity).total
    after_total = frame.weights_for(last.entity).total
    if not np.isclose(first.old_total, before_total, rtol=_MASS_RTOL, atol=0.0):
        raise PopulationError(
            f"Node {node.id!r} frame mass log starts at {first.old_total!r}; "
            f"the incumbent {first.entity!r} total is {before_total!r}."
        )
    if not np.isclose(last.new_total, after_total, rtol=_MASS_RTOL, atol=0.0):
        raise PopulationError(
            f"Node {node.id!r} frame mass log ends at {last.new_total!r}; "
            f"the resulting {last.entity!r} total is {after_total!r}."
        )
    return Frame(
        _copied_tables(frame),
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=(*frame.mass_log, *records),
        metadata=frame.metadata,
    )


def _design_cap(node: Node) -> tuple[str, float] | None:
    transition = node.weights
    if transition is None or transition.to_kind != WeightKind.CALIBRATED.value:
        return None
    raw_cap = node.params.get("max_weight_ratio")
    if raw_cap is None:
        return None
    if (
        isinstance(raw_cap, bool)
        or not isinstance(raw_cap, int | float)
        or not np.isfinite(float(raw_cap))
        or float(raw_cap) <= 0
    ):
        raise PopulationError(
            f"Node {node.id!r} max_weight_ratio must be finite and positive."
        )
    if node.params.get("weight_anchor") != WeightKind.DESIGN.value:
        raise PopulationError(
            f"Node {node.id!r} max_weight_ratio must declare weight_anchor='design'."
        )
    return transition.entity, float(raw_cap)


def _weight_ratios(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    ratios = np.empty(len(values), dtype=np.float64)
    positive_anchor = design > 0
    ratios[positive_anchor] = values[positive_anchor] / design[positive_anchor]
    zero_values = values[~positive_anchor] == 0
    ratios[~positive_anchor] = np.where(zero_values, 0.0, np.inf)
    return ratios


def _assert_design_weight_cap(
    frame: Frame,
    design_weights: Mapping[str, np.ndarray],
    node: Node,
) -> None:
    cap_spec = _design_cap(node)
    if cap_spec is None:
        return
    entity, cap = cap_spec
    try:
        design = design_weights[entity]
    except KeyError as error:
        raise PopulationError(
            f"Node {node.id!r} has no original design-weight anchor for {entity!r}."
        ) from error
    current = frame.weights_for(entity).values
    limits = design * cap
    positive_limits = limits > 0
    limits[positive_limits] = np.nextafter(limits[positive_limits], np.inf)
    violations = current > limits
    if violations.any():
        position = int(np.flatnonzero(violations)[0])
        id_column = frame.schema.entity_id_column(entity)
        entity_id = frame.table(entity)[id_column].iloc[position]
        raise PopulationError(
            f"Node {node.id!r} calibrated weight for {entity}.{entity_id!r} is "
            f"{current[position]!r}, above {cap!r} * original design weight "
            f"{design[position]!r}."
        )


def weight_cap_receipt(population: Population, node: Node) -> Mapping[str, object]:
    """Return executor-authored receipt fields for a validated design cap."""

    cap_spec = _design_cap(node)
    if cap_spec is None:
        return MappingProxyType({})
    entity, cap = cap_spec
    try:
        design = population.design_weights[entity]
    except KeyError as error:  # defended by patch(), useful for direct callers
        raise PopulationError(
            f"Node {node.id!r} has no original design-weight anchor for {entity!r}."
        ) from error
    current = population.frame.weights_for(entity).values
    realized = float(_weight_ratios(current, design).max())
    return MappingProxyType(
        {
            "weight_anchor": WeightKind.DESIGN.value,
            "max_weight_ratio": cap,
            "realized_max_weight_ratio": realized,
        }
    )


def _mass_policy(node: Node) -> str:
    if node.weights is None:
        return node.mass
    if node.structural is not StructuralDelta.NONE and node.mass != node.weights.mass:
        raise PopulationError(
            f"Node {node.id!r} has conflicting structural/weight mass policies "
            f"{node.mass!r} and {node.weights.mass!r}."
        )
    return node.weights.mass


def _mass_record(
    before: Frame,
    after: Frame,
    node: Node,
    result: KernelResult,
    policy: str,
) -> MassRecord:
    if policy not in MASS_POLICIES:
        raise PopulationError(f"Node {node.id!r} has unknown mass policy {policy!r}.")
    before_mass = before.stratum_mass()
    after_mass = after.stratum_mass()
    before_pairs = tuple((key, float(value)) for key, value in before_mass.items())
    after_pairs = tuple((key, float(value)) for key, value in after_mass.items())
    before_total = float(before_mass.sum())
    after_total = float(after_mass.sum())
    if policy == "conserve":
        _assert_mass_mapping(
            dict(before_pairs),
            dict(after_pairs),
            label=f"Node {node.id!r} mass='conserve'",
        )

    receipt_mass = result.receipt.get("mass")
    if receipt_mass is not None:
        _validate_mass_receipt(
            receipt_mass,
            policy=policy,
            before_total=before_total,
            after_total=after_total,
            before=dict(before_pairs),
            after=dict(after_pairs),
            node_id=node.id,
        )
    elif policy == "declared":
        raise PopulationError(
            f"Node {node.id!r} mass='declared' requires receipt['mass']."
        )

    return MassRecord(
        node_id=node.id,
        operation=(
            node.structural.value
            if node.structural is not StructuralDelta.NONE
            else "weights"
        ),
        policy=policy,
        before_total=before_total,
        after_total=after_total,
        before_by_stratum=before_pairs,
        after_by_stratum=after_pairs,
        entity=(
            node.weights.entity
            if node.weights is not None
            else _expand_weight_entity(node)
            if node.structural is StructuralDelta.EXPAND
            and "expand_weight_entity" in node.params
            else None
        ),
    )


def _validate_mass_receipt(
    raw: object,
    *,
    policy: str,
    before_total: float,
    after_total: float,
    before: Mapping[object, float],
    after: Mapping[object, float],
    node_id: str,
) -> None:
    if not isinstance(raw, Mapping):
        raise PopulationError(f"Node {node_id!r} receipt['mass'] must be a mapping.")
    if raw.get("policy") != policy:
        raise PopulationError(
            f"Node {node_id!r} mass receipt policy {raw.get('policy')!r} does not "
            f"match {policy!r}."
        )
    _assert_close(raw.get("before"), before_total, f"Node {node_id!r} mass.before")
    _assert_close(raw.get("after"), after_total, f"Node {node_id!r} mass.after")
    _assert_receipt_mapping(
        raw.get("stratum_before"), before, f"Node {node_id!r} mass.stratum_before"
    )
    _assert_receipt_mapping(
        raw.get("stratum_after"), after, f"Node {node_id!r} mass.stratum_after"
    )


def _assert_close(observed: object, expected: float, label: str) -> None:
    if isinstance(observed, bool) or not isinstance(observed, int | float):
        raise PopulationError(f"{label} must be numeric.")
    if not np.isclose(float(observed), expected, rtol=_MASS_RTOL, atol=0.0):
        raise PopulationError(
            f"{label} is {observed!r}; computed value is {expected!r}."
        )


def _assert_receipt_mapping(
    observed: object, expected: Mapping[object, float], label: str
) -> None:
    if not isinstance(observed, Mapping):
        raise PopulationError(f"{label} must be a mapping.")
    converted = {key: float(value) for key, value in observed.items()}
    _assert_mass_mapping(expected, converted, label=label)


def _assert_mass_mapping(
    expected: Mapping[object, float], observed: Mapping[object, float], *, label: str
) -> None:
    if set(expected) != set(observed):
        raise PopulationError(
            f"{label} changed strata: expected {list(expected)}, got {list(observed)}."
        )
    for stratum in expected:
        if not np.isclose(
            expected[stratum], observed[stratum], rtol=_MASS_RTOL, atol=0.0
        ):
            raise PopulationError(
                f"{label} changed stratum {stratum!r}: {expected[stratum]!r} -> "
                f"{observed[stratum]!r}."
            )


def _empty_column(length: int, token: str, owned_mask: np.ndarray) -> pd.Series:
    dtype = dtype_for_token(token)
    if token in {"boolean", "Int64", "string"}:
        return pd.Series(pd.array([pd.NA] * length, dtype=dtype))
    if token in {"float32", "float64"}:
        return pd.Series(np.full(length, np.nan, dtype=dtype))
    if not owned_mask.all():
        raise PopulationError(
            f"A new masked {token!r} column cannot be null outside its owned rows; "
            "use a nullable declaration dtype or own every row."
        )
    # No unowned position observes this temporary value; the incoming full-row
    # result replaces every element before the Frame is rebuilt.
    return pd.Series(np.zeros(length, dtype=dtype))


def _copied_tables(frame: Frame) -> dict[str, pd.DataFrame]:
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    tables.update({link: frame.link(link).copy(deep=True) for link in frame.links})
    return tables


def _rebuild_frame(frame: Frame, tables: Mapping[str, pd.DataFrame]) -> Frame:
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _storage_parts(series: pd.Series, selected: np.ndarray) -> tuple[bytes, bytes]:
    nulls = series.isna().to_numpy(dtype=np.bool_, copy=False)[selected]
    array = series.array
    data = getattr(array, "_data", None)
    mask = getattr(array, "_mask", None)
    if isinstance(data, np.ndarray) and isinstance(mask, np.ndarray):
        values = np.ascontiguousarray(data[selected]).tobytes()
        bitmap = np.ascontiguousarray(mask[selected]).tobytes()
        return values, bitmap
    if isinstance(series.dtype, pd.StringDtype):
        payload = bytearray()
        for value, is_null in zip(
            series.to_numpy(dtype=object, copy=False)[selected], nulls, strict=True
        ):
            if is_null:
                payload.extend((0).to_bytes(8, "little"))
            else:
                encoded = str(value).encode("utf-8")
                payload.extend((len(encoded) + 1).to_bytes(8, "little"))
                payload.extend(encoded)
        return bytes(payload), np.ascontiguousarray(nulls).tobytes()
    values = series.to_numpy(copy=False)[selected]
    return (
        np.ascontiguousarray(values).tobytes(),
        np.ascontiguousarray(nulls).tobytes(),
    )
