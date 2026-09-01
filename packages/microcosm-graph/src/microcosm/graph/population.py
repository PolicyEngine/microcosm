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

from microcosm.frame import Frame, WeightKind, Weights

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
    "storage_equal",
    "token_for_dtype",
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

    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        version: str,
        owners: Mapping[tuple[str, str], str] | None = None,
        *,
        mass_ledger: tuple[MassRecord, ...] = (),
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
        return cls(frame, version, resolved_owners, kinds, mass_ledger)


def population_from_frame(
    frame: Frame,
    version: str,
    owners: Mapping[tuple[str, str], str] | None = None,
    *,
    mass_ledger: tuple[MassRecord, ...] = (),
) -> Population:
    """Functional spelling of :meth:`Population.from_frame`."""

    return Population.from_frame(
        frame,
        version,
        owners,
        mass_ledger=mass_ledger,
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
    """Validate and apply one node result without mutating ``population``."""

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
    if set(result.columns) != expected_columns:
        raise PopulationError(
            f"Node {node.id!r} returned columns {sorted(result.columns)}; "
            f"expected exactly {sorted(expected_columns)}."
        )

    before = population.frame
    if node.structural is StructuralDelta.NONE:
        if result.frame is not None:
            raise PopulationError(f"Non-structural node {node.id!r} returned a Frame.")
        frame, owners = _patch_columns(population, node, result)
    else:
        frame, owners = _patch_structural(population, node, result)

    if node.weights is not None:
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
    expected_position = _WEIGHT_ORDER.index(old.kind) + 1
    if expected_position >= len(_WEIGHT_ORDER):
        raise PopulationError(
            f"Node {node.id!r} cannot transition terminal {old.kind.value!r} weights."
        )
    expected_kind = _WEIGHT_ORDER[expected_position]
    declared_kind = WeightKind(transition.to_kind)
    if declared_kind is not expected_kind or result.weights.kind is not expected_kind:
        raise PopulationError(
            f"Node {node.id!r} weight transition must be immediate: "
            f"{old.kind.value!r} -> {expected_kind.value!r}; declaration/result "
            f"requested {transition.to_kind!r}/{result.weights.kind.value!r}."
        )

    try:
        current = frame.weights_for(transition.entity)
    except ValueError:
        current = None
    if current is not None and current.kind is expected_kind:
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
        entity=node.weights.entity if node.weights is not None else None,
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
