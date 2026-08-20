"""Transactional execution of compiled spec-engine nodes.

Kernels never receive a live :class:`microcosm.frame.Frame`.  They receive an
immutable, detached projection and return a declarative patch.  The executor
applies that patch to a private copy, computes a full exact diff, and refuses
any effect outside the compiled capability, mutation, and cell-scope
contracts.  A refused kernel cannot partially mutate the caller's projection.

The module is deliberately country- and program-agnostic.  Runtime row-scope
meaning comes from :mod:`scope_algebra`'s finite compiler-owned atom space.
Ambient effects are represented as grants here and are enforced by the broker
layer; no broker receipt contributes to node reuse identity.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd

from .brokers import (
    BrokerContractError,
    BrokerReceipt,
    BrokerSession,
    KernelBrokerSession,
    RNGBehaviorIdentity,
    SourceBehaviorIdentity,
    deny_all_session_for_node,
)
from .canonical import sha256_json
from .compiler_ir import (
    CompiledNode,
    current_compiler_ir_abi,
    row_classifier_contract,
)
from .model import FrozenMap, freeze_json, thaw_json
from .scope_algebra import ClosedScopeRegistry, ScopeAlgebraError


class ExecutorError(ValueError):
    """A node, kernel patch, or resulting structural diff is invalid."""


class CapabilityError(ExecutorError):
    """A node's orthogonal capability contract is invalid before dispatch."""


class PatchScopeError(ExecutorError):
    """A patch writes outside its compiled entity/column/row authority."""


class StructuralDiffError(ExecutorError):
    """A patch's exact structural diff violates its declared delta."""


class NodeOrderingError(ExecutorError):
    """Compiled nodes cannot be placed in a safe deterministic total order."""


class Determinism(StrEnum):
    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    NONDETERMINISTIC = "nondeterministic"


class NumericReproducibility(StrEnum):
    BITWISE = "bitwise"
    TOLERANCE_BOUND = "tolerance_bound"
    UNSPECIFIED = "unspecified"


class Effect(StrEnum):
    NONE = "none"
    DECLARED_SOURCE_READ = "declared_source_read"
    DECLARED_SINK_WRITE = "declared_sink_write"


class StructuralDelta(StrEnum):
    NONE = "none"
    FILTER = "filter"
    EXPAND = "expand"
    JOIN = "join"
    RELINK = "relink"
    REORDER = "reorder"
    REWEIGHT = "reweight"


class RetrySafety(StrEnum):
    IDEMPOTENT = "idempotent"
    ATTEMPT_SCOPED = "attempt_scoped"
    NONRETRYABLE = "nonretryable"


class _InputCellState(StrEnum):
    SATISFIED = "satisfied"
    MISSING = "missing"
    INVALID = "invalid"


_CAPABILITY_FIELDS = frozenset(
    {
        "determinism",
        "numeric_reproducibility",
        "effects",
        "structural_delta",
        "retry_safety",
    }
)
_MUTATION_AXES = (
    "entity_keys",
    "cardinality",
    "links",
    "memberships",
    "order",
    "weights",
    "mass_history",
)
_PRESERVE_OPERATIONS = frozenset({"preserve", "preserve_absent"})
_TABLE_WRITE_MODES = frozenset({"column_cells", "structural_column"})
_WEIGHT_KINDS = frozenset({"design", "importance", "calibrated"})
_NODE_KEY_DOMAIN = "microcosm.spec-engine.static-node-key.v1"
_NODE_REUSE_KEY_DOMAIN = "microcosm.spec-engine.node-reuse-key.v1"
_BROKER_RECEIPT_DOMAIN = "microcosm.spec-engine.broker-access-receipt.v1"
_OPERATIONAL_SURFACE = "operational"
_RUN_PROVENANCE_FIELDS = frozenset(
    {
        "identity_generation",
        "source_grammar_receipt",
        "spec_binding",
        "authority_versions",
        "code_inventory_digest",
        "artifact_protocol_inventory",
        "run_request",
        "execution_receipt",
    }
)
_SOURCE_GRAMMAR_RECEIPT_FIELDS = frozenset(
    {"schema_version", "canonicalizer_version", "migration_chain"}
)
_SPEC_BINDING_FIELDS = frozenset(
    {
        "country",
        "schema_id",
        "schema_version",
        "canonicalizer_version",
        "spec_sha256",
        "attestation",
    }
)
_PROVENANCE_ONLY_FIELDS = frozenset(
    {
        "access_log",
        "artifact_protocol_inventory",
        "authority_versions",
        "broker_access_log",
        "broker_receipt",
        "code_inventory_digest",
        "config_authority",
        "execution_receipt",
        "identity_generation",
        "provenance",
        "run_provenance_identity",
        "run_request",
        "source_grammar_receipt",
        "spec_binding",
        "spec_sha256",
    }
)
_PROVENANCE_ONLY_VALUES = frozenset({_BROKER_RECEIPT_DOMAIN, _OPERATIONAL_SURFACE})
_VALID_MUTATION_CONTRACTS: Mapping[str, frozenset[tuple[str, str, str]]] = (
    MappingProxyType(
        {
            "entity_keys": frozenset(
                {
                    ("preserve", "entity_keys_valid", "entity_keys_unchanged"),
                    (
                        "append_remapped_clone_keys",
                        "native_entity_keys_unique",
                        "all_entity_keys_unique",
                    ),
                    (
                        "filter_entity_keys",
                        "entity_keys_valid",
                        "remaining_entity_keys_unique",
                    ),
                }
            ),
            "cardinality": frozenset(
                {
                    (
                        "preserve",
                        "entity_cardinality_valid",
                        "entity_cardinality_unchanged",
                    ),
                    (
                        "expand_complete_household_graphs",
                        "native_clone_index_zero",
                        "clone_roles_materialized",
                    ),
                    (
                        "filter_entity_rows",
                        "entity_cardinality_valid",
                        "entity_cardinality_filtered",
                    ),
                }
            ),
            "links": frozenset(
                {
                    ("preserve", "links_valid", "links_unchanged"),
                    ("preserve_absent", "link_tables_absent", "link_tables_absent"),
                    (
                        "append_relinked_clone_links",
                        "links_valid",
                        "clone_links_reference_remapped_keys",
                    ),
                    (
                        "filter_link_rows",
                        "links_valid",
                        "links_reference_surviving_keys",
                    ),
                    ("relink_references", "links_valid", "links_valid"),
                }
            ),
            "memberships": frozenset(
                {
                    ("preserve", "memberships_valid", "memberships_unchanged"),
                    (
                        "append_relinked_clone_memberships",
                        "native_memberships_valid",
                        "clone_memberships_reference_remapped_keys",
                    ),
                    (
                        "filter_membership_rows",
                        "memberships_valid",
                        "memberships_reference_surviving_keys",
                    ),
                    (
                        "relink_memberships",
                        "memberships_valid",
                        "memberships_valid",
                    ),
                }
            ),
            "order": frozenset(
                {
                    ("preserve", "entity_order_valid", "entity_order_unchanged"),
                    (
                        "append_clone_blocks_preserving_native_order",
                        "native_entity_order_valid",
                        "clone_blocks_follow_native_rows",
                    ),
                    (
                        "filter_rows_preserving_order",
                        "entity_order_valid",
                        "surviving_entity_order_preserved",
                    ),
                    (
                        "reorder_rows",
                        "entity_order_valid",
                        "entity_order_permuted",
                    ),
                }
            ),
            "weights": frozenset(
                {
                    ("preserve", "weights_valid", "weights_unchanged"),
                    (
                        "split_mass_across_clone_descendants",
                        "native_household_mass_finite",
                        "household_mass_conserved",
                    ),
                    (
                        "filter_row_weights",
                        "weights_valid",
                        "weights_aligned_to_surviving_keys",
                    ),
                    (
                        "realign_row_weights",
                        "weights_valid",
                        "weights_preserve_key_mapping",
                    ),
                    ("replace_weights", "weights_valid", "weights_valid"),
                }
            ),
            "mass_history": frozenset(
                {
                    (
                        "preserve",
                        "mass_history_valid",
                        "mass_history_unchanged",
                    ),
                    (
                        "append_mass_history",
                        "mass_history_valid",
                        "mass_history_extended",
                    ),
                }
            ),
        }
    )
)
_DELTA_OPERATION_NAMES: Mapping[StructuralDelta, frozenset[str]] = MappingProxyType(
    {
        StructuralDelta.NONE: _PRESERVE_OPERATIONS,
        StructuralDelta.JOIN: _PRESERVE_OPERATIONS,
        StructuralDelta.FILTER: _PRESERVE_OPERATIONS
        | {
            "filter_entity_keys",
            "filter_entity_rows",
            "filter_link_rows",
            "filter_membership_rows",
            "filter_rows_preserving_order",
            "filter_row_weights",
        },
        StructuralDelta.EXPAND: _PRESERVE_OPERATIONS
        | {
            "append_remapped_clone_keys",
            "expand_complete_household_graphs",
            "append_relinked_clone_links",
            "append_relinked_clone_memberships",
            "append_clone_blocks_preserving_native_order",
            "split_mass_across_clone_descendants",
        },
        StructuralDelta.RELINK: _PRESERVE_OPERATIONS
        | {"relink_references", "relink_memberships"},
        StructuralDelta.REORDER: _PRESERVE_OPERATIONS
        | {"reorder_rows", "realign_row_weights"},
        StructuralDelta.REWEIGHT: _PRESERVE_OPERATIONS
        | {"replace_weights", "append_mass_history"},
    }
)


def _wire(value: object) -> object:
    if isinstance(value, FrozenMap | tuple):
        return thaw_json(value)  # type: ignore[arg-type]
    return copy.deepcopy(value)


def _deep_copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame without pandas' object-cell aliasing escape hatch."""

    result = frame.copy(deep=True)
    for column_index, dtype in enumerate(result.dtypes):
        if not pd.api.types.is_object_dtype(dtype):
            continue
        for row_index in range(len(result)):
            result.iat[row_index, column_index] = copy.deepcopy(
                frame.iat[row_index, column_index]
            )
    result.index = copy.deepcopy(frame.index)
    result.columns = copy.deepcopy(frame.columns)
    result.attrs = copy.deepcopy(frame.attrs)
    return result


def _deep_copy_series(series: pd.Series) -> pd.Series:
    """Copy a series, including mutable values stored in object cells."""

    result = series.copy(deep=True)
    if result.dtype == object:
        for index in range(len(result)):
            result.iat[index] = copy.deepcopy(series.iat[index])
    result.index = copy.deepcopy(series.index)
    result.attrs = copy.deepcopy(series.attrs)
    return result


def _mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExecutorError(f"{location}: object required")
    return value


def _string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExecutorError(f"{location}: non-empty string required")
    return value


def _exact_object_equal(left: object, right: object) -> bool:
    """Return representation-exact equality for a sealed projection value.

    Python and pandas equality deliberately coalesce several byte-distinct
    values: ``1 == True``, ``+0.0 == -0.0``, every missing scalar, and Series
    values whose names or attrs differ.  Those semantics are unsuitable for a
    structural receipt.  The executor already seals projections with protocol
    5 pickle bytes, so use that same representation here and fail closed for
    values that cannot be sealed.
    """

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = tuple(left)
        right_keys = tuple(right)
        return _exact_object_equal(left_keys, right_keys) and all(
            _exact_object_equal(left[key], right[key]) for key in left_keys
        )
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, str | bytes)
    ):
        return len(left) == len(right) and all(
            _exact_object_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    try:
        return pickle.dumps(left, protocol=5) == pickle.dumps(right, protocol=5)
    except (AttributeError, pickle.PickleError, TypeError, ValueError):
        return False


class WeightState:
    """An immutable row-aligned weight vector plus its semantic kind."""

    __slots__ = ("_values", "kind")

    def __init__(self, values: Iterable[object] | np.ndarray, kind: str) -> None:
        if kind not in _WEIGHT_KINDS:
            raise ExecutorError(f"weight kind must be one of {sorted(_WEIGHT_KINDS)!r}")
        array = np.array(values, dtype=np.float64, copy=True)
        if array.ndim != 1:
            raise ExecutorError("weight values must be one-dimensional")
        if array.size == 0:
            raise ExecutorError("weight values cannot be empty")
        if not np.isfinite(array).all():
            raise ExecutorError("weight values must be finite")
        if (array < 0).any():
            raise ExecutorError("weight values must be non-negative")
        if not (array > 0).any():
            raise ExecutorError("weight values cannot be all zero")
        array.setflags(write=False)
        self._values = array
        self.kind = kind

    @property
    def values(self) -> np.ndarray:
        """Return a detached copy so callers cannot mutate stored state."""

        return self._values.copy()

    def _internal_values(self) -> np.ndarray:
        return self._values

    def copy(self) -> WeightState:
        return WeightState(self._values, self.kind)


class ImmutableFrameProjection:
    """A detached frame projection covering every executor-visible surface.

    Tables and link tables are copied on input and on access.  Stable entity
    keys make cell diffs independent of pandas positions.  ``row_atoms`` binds
    those stable ids to a closed predicate space for exact row authorization.
    """

    __slots__ = (
        "_entity_order",
        "_entity_keys",
        "_links",
        "_link_order",
        "_link_targets",
        "_mass_history",
        "_membership_columns",
        "_membership_targets",
        "_metadata",
        "_row_atoms",
        "_strata",
        "_strata_entity",
        "_tables",
        "_virtual_receipts",
        "_weight_order",
        "_weights",
    )

    def __init__(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        entity_keys: Mapping[str, str],
        membership_columns: Mapping[str, Iterable[str]] | None = None,
        membership_targets: Mapping[str, Mapping[str, str]] | None = None,
        links: Mapping[str, pd.DataFrame] | None = None,
        link_targets: Mapping[str, Mapping[str, str]] | None = None,
        weights: Mapping[str, WeightState] | None = None,
        strata: pd.Series | None = None,
        strata_entity: str | None = None,
        mass_history: Sequence[object] = (),
        metadata: Mapping[str, object] | None = None,
        virtual_receipts: Mapping[tuple[str, str], object] | None = None,
        row_atoms: Mapping[str, Mapping[Hashable, Iterable[str]]] | None = None,
    ) -> None:
        if not isinstance(tables, Mapping) or not tables:
            raise ExecutorError("projection tables must be a non-empty mapping")
        table_copies: dict[str, pd.DataFrame] = {}
        for entity, table in tables.items():
            if not isinstance(entity, str) or not entity:
                raise ExecutorError("projection entity names must be non-empty strings")
            if not isinstance(table, pd.DataFrame):
                raise ExecutorError(f"table {entity!r} must be a pandas DataFrame")
            if any(not isinstance(column, str) for column in table.columns):
                raise ExecutorError(f"table {entity!r} columns must be strings")
            table_copies[entity] = _deep_copy_frame(table)

        column_owners: dict[str, list[str]] = defaultdict(list)
        for entity, table in table_copies.items():
            for column in table.columns:
                column_owners[column].append(entity)
        repeated_columns = {
            column: owners
            for column, owners in column_owners.items()
            if len(owners) > 1
        }
        if repeated_columns:
            raise ExecutorError(
                "entity-table columns must be globally unique; repeated="
                f"{repeated_columns!r}"
            )

        if set(entity_keys) != set(table_copies):
            raise ExecutorError(
                "entity_keys must name exactly the projection's entity tables"
            )
        key_copy: dict[str, str] = {}
        for entity, key in entity_keys.items():
            if not isinstance(key, str) or key not in table_copies[entity].columns:
                raise ExecutorError(
                    f"entity key for {entity!r} must name an existing column"
                )
            values = table_copies[entity][key]
            if values.isna().any() or values.duplicated().any():
                raise ExecutorError(
                    f"entity key {entity}.{key} must be non-null and unique"
                )
            for value in values.tolist():
                try:
                    hash(value)
                except TypeError as error:
                    raise ExecutorError(
                        f"entity key {entity}.{key} contains an unhashable value"
                    ) from error
            key_copy[entity] = key

        memberships: dict[str, tuple[str, ...]] = {}
        for entity, columns in (membership_columns or {}).items():
            if entity not in table_copies:
                raise ExecutorError(f"membership entity {entity!r} is not projected")
            normalized = tuple(columns)
            if len(normalized) != len(set(normalized)):
                raise ExecutorError(f"membership columns for {entity!r} repeat")
            missing = [
                column
                for column in normalized
                if column not in table_copies[entity].columns
            ]
            if missing:
                raise ExecutorError(
                    f"membership columns for {entity!r} are absent: {missing!r}"
                )
            memberships[entity] = normalized

        unknown_membership_targets = set(membership_targets or {}) - set(memberships)
        if unknown_membership_targets:
            raise ExecutorError(
                "membership targets name entities without membership columns: "
                f"{sorted(unknown_membership_targets)!r}"
            )

        membership_target_copy: dict[str, Mapping[str, str]] = {}
        for entity, targets in (membership_targets or {}).items():
            if entity not in table_copies or not isinstance(targets, Mapping):
                raise ExecutorError(
                    f"membership targets for {entity!r} require a projected entity"
                )
            normalized_targets: dict[str, str] = {}
            for column, target_entity in targets.items():
                if column not in memberships.get(entity, ()):
                    raise ExecutorError(
                        f"membership target {entity}.{column} is not a membership column"
                    )
                if target_entity not in table_copies:
                    raise ExecutorError(
                        f"membership target {target_entity!r} is not projected"
                    )
                target_ids = set(
                    table_copies[target_entity][key_copy[target_entity]].tolist()
                )
                values = table_copies[entity][column]
                if values.isna().any() or not set(values.tolist()) <= target_ids:
                    raise ExecutorError(
                        f"membership {entity}.{column} contains dangling references"
                    )
                normalized_targets[column] = target_entity
            membership_target_copy[entity] = MappingProxyType(normalized_targets)

        for entity, columns in memberships.items():
            missing_targets = set(columns) - set(membership_target_copy.get(entity, {}))
            if missing_targets:
                raise ExecutorError(
                    f"membership columns for {entity!r} lack targets: "
                    f"{sorted(missing_targets)!r}"
                )
        for entity, targets in membership_target_copy.items():
            for column, target_entity in targets.items():
                target_values = table_copies[target_entity][key_copy[target_entity]]
                referenced = set(table_copies[entity][column].tolist())
                if set(target_values.tolist()) != referenced:
                    raise ExecutorError(
                        f"membership {entity}.{column} must reference every "
                        f"{target_entity!r} key exactly as a set"
                    )
                try:
                    ordered = sorted(target_values.tolist())
                except TypeError:
                    ordered = sorted(
                        target_values.tolist(),
                        key=lambda value: (type(value).__name__, repr(value)),
                    )
                if target_values.tolist() != ordered:
                    raise ExecutorError(
                        f"membership target table {target_entity!r} keys must be "
                        "in deterministic ascending order"
                    )

        link_copies: dict[str, pd.DataFrame] = {}
        for name, table in (links or {}).items():
            if not isinstance(name, str) or not name:
                raise ExecutorError("link names must be non-empty strings")
            if not isinstance(table, pd.DataFrame):
                raise ExecutorError(f"link {name!r} must be a pandas DataFrame")
            if any(not isinstance(column, str) for column in table.columns):
                raise ExecutorError(f"link {name!r} columns must be strings")
            link_copies[name] = _deep_copy_frame(table)

        if set(link_targets or {}) != set(link_copies):
            raise ExecutorError(
                "link_targets must name exactly the projected link tables"
            )

        link_target_copy: dict[str, Mapping[str, str]] = {}
        for name, targets in (link_targets or {}).items():
            if name not in link_copies or not isinstance(targets, Mapping):
                raise ExecutorError(
                    f"link targets for {name!r} require a projected link table"
                )
            normalized_targets: dict[str, str] = {}
            for column, target_entity in targets.items():
                if column not in link_copies[name].columns:
                    raise ExecutorError(f"link target column {name}.{column} is absent")
                if target_entity not in table_copies:
                    raise ExecutorError(
                        f"link target {target_entity!r} is not projected"
                    )
                target_ids = set(
                    table_copies[target_entity][key_copy[target_entity]].tolist()
                )
                values = link_copies[name][column]
                if values.isna().any() or not set(values.tolist()) <= target_ids:
                    raise ExecutorError(
                        f"link {name}.{column} contains dangling references"
                    )
                normalized_targets[column] = target_entity
            target_columns = tuple(normalized_targets)
            if not target_columns:
                raise ExecutorError(
                    f"link {name!r} must declare at least one target column"
                )
            if link_copies[name].duplicated(subset=list(target_columns)).any():
                raise ExecutorError(
                    f"link {name!r} target-column row keys must be unique"
                )
            link_target_copy[name] = MappingProxyType(normalized_targets)

        weight_copies: dict[str, WeightState] = {}
        for entity, weight in (weights or {}).items():
            if entity not in table_copies:
                raise ExecutorError(f"weights name unknown entity {entity!r}")
            if not isinstance(weight, WeightState):
                raise ExecutorError(f"weights[{entity!r}] must be WeightState")
            if len(weight._internal_values()) != len(table_copies[entity]):
                raise ExecutorError(
                    f"weights[{entity!r}] length differs from its entity table"
                )
            weight_copies[entity] = weight.copy()
        if not weight_copies:
            raise ExecutorError("projection requires at least one typed weight vector")

        for entity in table_copies:
            reserved = f"{entity}_weight"
            for table_entity, candidate in table_copies.items():
                if reserved in candidate.columns and not (
                    table_entity == entity and entity in weight_copies
                ):
                    raise ExecutorError(
                        f"reserved weight column {reserved!r} is on "
                        f"entity {table_entity!r} without matching typed weights"
                    )

        atom_copies: dict[str, Mapping[Hashable, frozenset[str]]] = {}
        raw_row_atoms = row_atoms or {}
        unknown_atom_entities = set(raw_row_atoms) - set(table_copies)
        if unknown_atom_entities:
            raise ExecutorError(
                f"row_atoms name unknown entities {sorted(unknown_atom_entities)!r}"
            )
        for entity, rows in raw_row_atoms.items():
            if not isinstance(rows, Mapping):
                raise ExecutorError(f"row_atoms[{entity!r}] must be a mapping")
            ids = set(table_copies[entity][key_copy[entity]].tolist())
            if set(rows) != ids:
                raise ExecutorError(
                    f"row_atoms[{entity!r}] must name every stable row id exactly"
                )
            normalized_rows: dict[Hashable, frozenset[str]] = {}
            for row_id, atoms in rows.items():
                atom_set = frozenset(atoms)
                if not atom_set or any(
                    not isinstance(atom, str) or not atom for atom in atom_set
                ):
                    raise ExecutorError(
                        f"row_atoms[{entity!r}][{row_id!r}] must be non-empty strings"
                    )
                normalized_rows[row_id] = atom_set
            atom_copies[entity] = MappingProxyType(normalized_rows)

        virtual_copy: dict[tuple[str, str], object] = {}
        for key, value in (virtual_receipts or {}).items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(not isinstance(part, str) or not part for part in key)
            ):
                raise ExecutorError(
                    "virtual receipt keys must be (entity, column) string pairs"
                )
            virtual_copy[key] = copy.deepcopy(value)

        if strata is None:
            if strata_entity is not None:
                raise ExecutorError("strata_entity requires a strata series")
        elif strata_entity not in table_copies:
            raise ExecutorError("strata require a projected strata_entity")
        elif len(strata) != len(table_copies[strata_entity]):
            raise ExecutorError("strata length differs from its entity table")
        elif not strata.index.equals(table_copies[strata_entity].index):
            raise ExecutorError("strata index differs from its entity table")
        elif strata.isna().any():
            raise ExecutorError("strata labels must be non-null")
        self._tables = table_copies
        self._entity_order = tuple(table_copies)
        self._entity_keys = key_copy
        self._membership_columns = memberships
        self._membership_targets = membership_target_copy
        self._links = link_copies
        self._link_order = tuple(link_copies)
        self._link_targets = link_target_copy
        self._weights = weight_copies
        self._weight_order = tuple(weight_copies)
        self._strata = None if strata is None else _deep_copy_series(strata)
        self._strata_entity = strata_entity
        self._mass_history = copy.deepcopy(tuple(mass_history))
        self._metadata = copy.deepcopy(dict(metadata or {}))
        self._virtual_receipts = virtual_copy
        self._row_atoms = atom_copies

    @property
    def entities(self) -> tuple[str, ...]:
        return self._entity_order

    def table(self, entity: str) -> pd.DataFrame:
        try:
            return _deep_copy_frame(self._tables[entity])
        except KeyError as error:
            raise ExecutorError(f"unknown projected entity {entity!r}") from error

    def link(self, name: str) -> pd.DataFrame:
        try:
            return _deep_copy_frame(self._links[name])
        except KeyError as error:
            raise ExecutorError(f"unknown projected link {name!r}") from error

    def weights_for(self, entity: str) -> WeightState:
        try:
            return self._weights[entity].copy()
        except KeyError as error:
            raise ExecutorError(
                f"unknown projected weight entity {entity!r}"
            ) from error

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(copy.deepcopy(self._metadata))

    @property
    def mass_history(self) -> tuple[object, ...]:
        return copy.deepcopy(self._mass_history)

    @property
    def strata(self) -> pd.Series | None:
        return None if self._strata is None else _deep_copy_series(self._strata)

    @property
    def virtual_receipts(self) -> Mapping[tuple[str, str], object]:
        return MappingProxyType(copy.deepcopy(self._virtual_receipts))

    def row_atoms_for(self, entity: str) -> Mapping[Hashable, frozenset[str]]:
        return MappingProxyType(dict(self._row_atoms.get(entity, {})))

    def _parts(self) -> dict[str, object]:
        return {
            "tables": {
                entity: _deep_copy_frame(table)
                for entity, table in self._tables.items()
            },
            "entity_keys": dict(self._entity_keys),
            "membership_columns": dict(self._membership_columns),
            "membership_targets": {
                entity: dict(targets)
                for entity, targets in self._membership_targets.items()
            },
            "links": {
                name: _deep_copy_frame(table) for name, table in self._links.items()
            },
            "link_targets": {
                name: dict(targets) for name, targets in self._link_targets.items()
            },
            "weights": {
                entity: weight.copy() for entity, weight in self._weights.items()
            },
            "strata": None if self._strata is None else _deep_copy_series(self._strata),
            "strata_entity": self._strata_entity,
            "mass_history": copy.deepcopy(self._mass_history),
            "metadata": copy.deepcopy(self._metadata),
            "virtual_receipts": copy.deepcopy(self._virtual_receipts),
            "row_atoms": {
                entity: dict(rows) for entity, rows in self._row_atoms.items()
            },
        }

    def detached_copy(self) -> ImmutableFrameProjection:
        return ImmutableFrameProjection(**self._parts())  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class KernelPatch:
    """Sparse surface replacements returned by a kernel."""

    structural_delta: StructuralDelta | str
    tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    drop_entities: frozenset[str] = frozenset()
    entity_keys: Mapping[str, str] = field(default_factory=dict)
    membership_columns: Mapping[str, Iterable[str]] = field(default_factory=dict)
    links: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    drop_links: frozenset[str] = frozenset()
    weights: Mapping[str, WeightState] = field(default_factory=dict)
    drop_weights: frozenset[str] = frozenset()
    strata: pd.Series | None = None
    replace_strata: bool = False
    mass_history: Sequence[object] | None = None
    metadata: Mapping[str, object] | None = None
    virtual_writes: Mapping[tuple[str, str], object] = field(default_factory=dict)
    virtual_deletes: frozenset[tuple[str, str]] = frozenset()
    row_atoms: Mapping[str, Mapping[Hashable, Iterable[str]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        try:
            delta = StructuralDelta(self.structural_delta)
        except ValueError as error:
            raise ExecutorError(
                f"patch structural_delta is unknown: {self.structural_delta!r}"
            ) from error
        object.__setattr__(self, "structural_delta", delta)
        object.__setattr__(
            self,
            "tables",
            MappingProxyType(
                {name: _deep_copy_frame(table) for name, table in self.tables.items()}
            ),
        )
        object.__setattr__(self, "drop_entities", frozenset(self.drop_entities))
        object.__setattr__(
            self, "entity_keys", MappingProxyType(dict(self.entity_keys))
        )
        object.__setattr__(
            self,
            "membership_columns",
            MappingProxyType(
                {
                    name: tuple(columns)
                    for name, columns in self.membership_columns.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "links",
            MappingProxyType(
                {name: _deep_copy_frame(table) for name, table in self.links.items()}
            ),
        )
        object.__setattr__(self, "drop_links", frozenset(self.drop_links))
        object.__setattr__(
            self,
            "weights",
            MappingProxyType(
                {name: value.copy() for name, value in self.weights.items()}
            ),
        )
        object.__setattr__(self, "drop_weights", frozenset(self.drop_weights))
        object.__setattr__(
            self,
            "strata",
            None if self.strata is None else _deep_copy_series(self.strata),
        )
        if self.mass_history is not None:
            object.__setattr__(
                self, "mass_history", copy.deepcopy(tuple(self.mass_history))
            )
        if self.metadata is not None:
            object.__setattr__(
                self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata)))
            )
        object.__setattr__(
            self,
            "virtual_writes",
            MappingProxyType(copy.deepcopy(dict(self.virtual_writes))),
        )
        object.__setattr__(self, "virtual_deletes", frozenset(self.virtual_deletes))
        object.__setattr__(
            self,
            "row_atoms",
            MappingProxyType(
                {
                    entity: MappingProxyType(
                        {row_id: frozenset(atoms) for row_id, atoms in rows.items()}
                    )
                    for entity, rows in self.row_atoms.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CellChange:
    entity: str
    column: str
    row_id: Hashable
    before_present: bool
    after_present: bool
    before: object
    after: object


@dataclass(frozen=True, slots=True)
class TableDiff:
    entity: str
    added: bool
    removed: bool
    before_row_ids: tuple[Hashable, ...]
    after_row_ids: tuple[Hashable, ...]
    before_columns: tuple[str, ...]
    after_columns: tuple[str, ...]
    added_row_ids: frozenset[Hashable]
    removed_row_ids: frozenset[Hashable]
    order_changed: bool
    index_changed: bool
    stable_index_changed_row_ids: frozenset[Hashable]
    column_order_changed: bool
    column_axis_name_changed: bool
    column_axis_contract_changed: bool
    attrs_changed: bool
    frame_contract_changed: bool
    added_columns: tuple[str, ...]
    removed_columns: tuple[str, ...]
    dtype_changes: tuple[str, ...]
    cell_changes: tuple[CellChange, ...]


@dataclass(frozen=True, slots=True)
class LinkCellChange:
    link: str
    column: str
    row_id: tuple[Hashable, ...]
    before_present: bool
    after_present: bool
    before: object
    after: object


@dataclass(frozen=True, slots=True)
class LinkDiff:
    name: str
    added: bool
    removed: bool
    before_row_ids: tuple[tuple[Hashable, ...], ...]
    after_row_ids: tuple[tuple[Hashable, ...], ...]
    before_columns: tuple[str, ...]
    after_columns: tuple[str, ...]
    added_row_ids: frozenset[tuple[Hashable, ...]]
    removed_row_ids: frozenset[tuple[Hashable, ...]]
    order_changed: bool
    index_changed: bool
    stable_index_changed_row_ids: frozenset[tuple[Hashable, ...]]
    column_order_changed: bool
    column_axis_name_changed: bool
    column_axis_contract_changed: bool
    attrs_changed: bool
    frame_contract_changed: bool
    added_columns: tuple[str, ...]
    removed_columns: tuple[str, ...]
    dtype_changes: tuple[str, ...]
    cell_changes: tuple[LinkCellChange, ...]


@dataclass(frozen=True, slots=True)
class WeightDiff:
    entity: str
    values_changed: bool
    kind_changed: bool
    storage_order_changed: bool
    changed_row_ids: frozenset[Hashable]


@dataclass(frozen=True, slots=True)
class VirtualChange:
    entity: str
    column: str
    before_present: bool
    after_present: bool


@dataclass(frozen=True, slots=True)
class FrameDiff:
    tables: tuple[TableDiff, ...]
    links: tuple[LinkDiff, ...]
    weights: tuple[WeightDiff, ...]
    strata_changed: bool
    mass_history_changed: bool
    metadata_changed: bool
    virtual_changes: tuple[VirtualChange, ...]
    entity_order_contract_changed: bool
    entity_key_contract_changed: bool
    membership_contract_changed: bool
    link_order_contract_changed: bool
    link_contract_changed: bool
    weight_order_contract_changed: bool
    strata_contract_changed: bool
    row_atoms_changed: bool

    @property
    def links_changed(self) -> tuple[str, ...]:
        """Compatibility view of the granular link-table diff."""

        return tuple(row.name for row in self.links)

    @property
    def empty(self) -> bool:
        return not (
            self.tables
            or self.links
            or self.weights
            or self.strata_changed
            or self.mass_history_changed
            or self.metadata_changed
            or self.virtual_changes
            or self.entity_order_contract_changed
            or self.entity_key_contract_changed
            or self.membership_contract_changed
            or self.link_order_contract_changed
            or self.link_contract_changed
            or self.weight_order_contract_changed
            or self.strata_contract_changed
            or self.row_atoms_changed
        )


@dataclass(frozen=True, slots=True)
class NodeCapabilities:
    determinism: Determinism
    numeric_reproducibility: NumericReproducibility
    effects: frozenset[Effect]
    structural_delta: StructuralDelta
    retry_safety: RetrySafety

    @classmethod
    def from_mapping(cls, value: object) -> NodeCapabilities:
        row = _mapping(value, location="capabilities")
        if set(row) != _CAPABILITY_FIELDS:
            missing = sorted(_CAPABILITY_FIELDS - set(row))
            extra = sorted(set(row) - _CAPABILITY_FIELDS)
            raise CapabilityError(
                f"capabilities must have exactly five orthogonal fields; "
                f"missing={missing!r}, extra={extra!r}"
            )
        try:
            determinism = Determinism(row["determinism"])
            reproducibility = NumericReproducibility(row["numeric_reproducibility"])
            structural_delta = StructuralDelta(row["structural_delta"])
            retry_safety = RetrySafety(row["retry_safety"])
        except (TypeError, ValueError) as error:
            raise CapabilityError(
                f"capabilities contain an unknown value: {error}"
            ) from error
        raw_effects = row["effects"]
        if not isinstance(raw_effects, Sequence) or isinstance(
            raw_effects, str | bytes
        ):
            raise CapabilityError("capabilities/effects must be a non-empty array")
        try:
            effects = frozenset(Effect(value) for value in raw_effects)
        except (TypeError, ValueError) as error:
            raise CapabilityError(
                f"capabilities/effects is unknown: {error}"
            ) from error
        if not effects or len(effects) != len(raw_effects):
            raise CapabilityError("capabilities/effects must be non-empty and unique")
        if Effect.NONE in effects and effects != {Effect.NONE}:
            raise CapabilityError("capabilities/effects 'none' is exclusive")
        return cls(
            determinism=determinism,
            numeric_reproducibility=reproducibility,
            effects=effects,
            structural_delta=structural_delta,
            retry_safety=retry_safety,
        )


@dataclass(frozen=True, slots=True)
class MutationContract:
    operation: str
    precondition: str
    postcondition: str


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    attempt: int = 0
    resumed: bool = False
    attempt_scope: str | None = None
    granted_effects: frozenset[Effect | str] = frozenset()
    require_byte_equivalence: bool = True
    brokers: BrokerSession | KernelBrokerSession | None = None
    run_provenance_identity: RunProvenanceIdentity | None = None

    def normalized_effects(self) -> frozenset[Effect]:
        try:
            return frozenset(Effect(effect) for effect in self.granted_effects)
        except ValueError as error:
            raise CapabilityError(
                f"execution context has unknown effect: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class RegisteredKernel:
    function: Callable[[ImmutableFrameProjection, ExecutionContext], KernelPatch]
    implementation_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("registered kernel function must be callable")
        if (
            not isinstance(self.implementation_sha256, str)
            or len(self.implementation_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.implementation_sha256
            )
        ):
            raise ValueError(
                "registered kernel implementation_sha256 must be lowercase sha256"
            )


@dataclass(frozen=True, slots=True)
class RowClassification:
    """Trusted scope atoms and one pre-patch source row for an added row."""

    atoms: frozenset[str]
    source_row_id: Hashable

    def __post_init__(self) -> None:
        if (
            not isinstance(self.atoms, frozenset)
            or not self.atoms
            or any(not isinstance(atom, str) or not atom for atom in self.atoms)
        ):
            raise ValueError("row classification atoms must be a non-empty frozenset")
        try:
            hash(self.source_row_id)
        except TypeError as error:
            raise ValueError(
                "row classification source_row_id must be hashable"
            ) from error


@dataclass(frozen=True, slots=True)
class RegisteredRowClassifier:
    """Trusted orchestration-side classifier for rows created by a kernel.

    The kernel never supplies row atoms.  The executor invokes this separately
    registered function after applying a structural patch and validates its
    output against the node's compiler-bound finite predicate space.
    """

    function: Callable[
        [str, pd.DataFrame, str, frozenset[Hashable], ClosedScopeRegistry],
        Mapping[Hashable, RowClassification],
    ]
    implementation_sha256: str
    predicate_space: str

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("row classifier function must be callable")
        if (
            not isinstance(self.implementation_sha256, str)
            or len(self.implementation_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.implementation_sha256
            )
        ):
            raise ValueError("row classifier implementation digest must be sha256")
        if not isinstance(self.predicate_space, str) or not self.predicate_space:
            raise ValueError("row classifier predicate_space must be non-empty")


class KernelRegistry(Protocol):
    def __getitem__(self, key: str) -> RegisteredKernel: ...


class RowClassifierRegistry(Protocol):
    def __getitem__(self, key: str) -> RegisteredRowClassifier: ...


@dataclass(frozen=True, slots=True)
class ValidatedPatch:
    """A patch proven against one exact base projection."""

    node_id: str
    node_key: str
    attempt: int
    attempt_scope: str | None
    patch: KernelPatch
    diff: FrameDiff
    broker_receipt: BrokerReceipt
    _base: ImmutableFrameProjection
    _result: ImmutableFrameProjection
    _base_sha256: str
    _result_sha256: str
    _patch_sha256: str
    _diff_sha256: str
    _envelope_sha256: str


def _provenance_sha256(value: object, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExecutorError(f"{location} must be a lowercase sha256")
    return value


def _provenance_positive_int(value: object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExecutorError(f"{location} must be a positive integer")
    return value


def _freeze_provenance_mapping(
    value: Mapping[str, object], *, location: str
) -> FrozenMap:
    if not isinstance(value, Mapping):
        raise ExecutorError(f"{location} must be an object")
    frozen = freeze_json(dict(value))
    if not isinstance(frozen, FrozenMap):  # pragma: no cover - guarded above
        raise AssertionError(f"{location} did not freeze to an object")
    return frozen


def _source_grammar_receipt(
    value: Mapping[str, object] | None,
) -> FrozenMap | None:
    if value is None:
        return None
    row = dict(value)
    if set(row) != _SOURCE_GRAMMAR_RECEIPT_FIELDS:
        raise ExecutorError(
            "source_grammar_receipt must contain exactly schema_version, "
            "canonicalizer_version, and migration_chain"
        )
    _provenance_positive_int(
        row["schema_version"], location="source_grammar_receipt/schema_version"
    )
    _provenance_positive_int(
        row["canonicalizer_version"],
        location="source_grammar_receipt/canonicalizer_version",
    )
    migration_chain = row["migration_chain"]
    if not isinstance(migration_chain, Sequence) or isinstance(
        migration_chain, str | bytes
    ):
        raise ExecutorError("source_grammar_receipt/migration_chain must be an array")
    for index, value in enumerate(migration_chain):
        if not isinstance(value, Mapping) or set(value) != {"id", "sha256"}:
            raise ExecutorError(
                "source_grammar_receipt migration rows require exactly id and sha256"
            )
        migration_id = value["id"]
        if not isinstance(migration_id, str) or not migration_id:
            raise ExecutorError(
                f"source_grammar_receipt/migration_chain/{index}/id must be non-empty"
            )
        _provenance_sha256(
            value["sha256"],
            location=f"source_grammar_receipt/migration_chain/{index}/sha256",
        )
    return _freeze_provenance_mapping(row, location="source_grammar_receipt")


def _provenance_spec_binding(
    value: Mapping[str, object] | None,
) -> FrozenMap | None:
    if value is None:
        return None
    row = dict(value)
    if set(row) != _SPEC_BINDING_FIELDS:
        raise ExecutorError(
            "spec_binding must contain exactly country, schema_id, schema_version, "
            "canonicalizer_version, spec_sha256, and attestation"
        )
    for field_name in ("country", "schema_id"):
        field_value = row[field_name]
        if not isinstance(field_value, str) or not field_value:
            raise ExecutorError(f"spec_binding/{field_name} must be non-empty")
    _provenance_positive_int(
        row["schema_version"], location="spec_binding/schema_version"
    )
    _provenance_positive_int(
        row["canonicalizer_version"], location="spec_binding/canonicalizer_version"
    )
    _provenance_sha256(row["spec_sha256"], location="spec_binding/spec_sha256")
    if row["attestation"] not in {"mirror-attested", "bundle-authoritative"}:
        raise ExecutorError(
            "spec_binding/attestation must be mirror-attested or bundle-authoritative"
        )
    return _freeze_provenance_mapping(row, location="spec_binding")


@dataclass(frozen=True, slots=True)
class RunProvenanceIdentity:
    """The closed D3 run identity, kept separate from semantic reuse keys."""

    identity_generation: int
    source_grammar_receipt: FrozenMap | None
    spec_binding: FrozenMap | None
    authority_versions: FrozenMap
    code_inventory_digest: str
    artifact_protocol_inventory: FrozenMap
    run_request: FrozenMap
    execution_receipt: FrozenMap

    def __post_init__(self) -> None:
        if (
            isinstance(self.identity_generation, bool)
            or not isinstance(self.identity_generation, int)
            or self.identity_generation not in {0, 1}
        ):
            raise ExecutorError(
                "identity_generation must be 0 (historic) or 1 (binding); "
                "unknown generations are refused"
            )
        for field_name in (
            "authority_versions",
            "artifact_protocol_inventory",
            "run_request",
            "execution_receipt",
        ):
            if not isinstance(getattr(self, field_name), FrozenMap):
                raise ExecutorError(f"run provenance {field_name} must be frozen")
        _provenance_sha256(
            self.code_inventory_digest,
            location="run_provenance_identity/code_inventory_digest",
        )
        if self.identity_generation == 0:
            if self.source_grammar_receipt is not None or self.spec_binding is not None:
                raise ExecutorError(
                    "generation 0 provenance cannot be retro-labeled with the D3 triad"
                )
            return
        if self.source_grammar_receipt is None or self.spec_binding is None:
            raise ExecutorError(
                "generation 1 provenance requires source_grammar_receipt and spec_binding"
            )
        grammar = _wire(self.source_grammar_receipt)
        binding = _wire(self.spec_binding)
        assert isinstance(grammar, dict) and isinstance(binding, dict)
        for field_name in ("schema_version", "canonicalizer_version"):
            if grammar[field_name] != binding[field_name]:
                raise ExecutorError(
                    f"run provenance {field_name} differs between grammar and binding"
                )

    @property
    def promotable(self) -> bool:
        return self.identity_generation == 1

    def to_wire(self) -> dict[str, object]:
        result = {
            "identity_generation": self.identity_generation,
            "source_grammar_receipt": (
                None
                if self.source_grammar_receipt is None
                else _wire(self.source_grammar_receipt)
            ),
            "spec_binding": (
                None if self.spec_binding is None else _wire(self.spec_binding)
            ),
            "authority_versions": _wire(self.authority_versions),
            "code_inventory_digest": self.code_inventory_digest,
            "artifact_protocol_inventory": _wire(self.artifact_protocol_inventory),
            "run_request": _wire(self.run_request),
            "execution_receipt": _wire(self.execution_receipt),
        }
        if set(result) != _RUN_PROVENANCE_FIELDS:  # pragma: no cover - literal map
            raise AssertionError("run provenance wire fields differ from the RFC")
        return result


def build_run_provenance_identity(
    *,
    identity_generation: int,
    source_grammar_receipt: Mapping[str, object] | None,
    spec_binding: Mapping[str, object] | None,
    authority_versions: Mapping[str, object],
    code_inventory_digest: str,
    artifact_protocol_inventory: Mapping[str, object],
    run_request: Mapping[str, object],
    execution_receipt: Mapping[str, object],
) -> RunProvenanceIdentity:
    """Build one closed provenance identity; none of it crosses into reuse."""

    if (
        isinstance(identity_generation, bool)
        or not isinstance(identity_generation, int)
        or identity_generation not in {0, 1}
    ):
        raise ExecutorError(
            "identity_generation must be 0 (historic) or 1 (binding); "
            "unknown generations are refused"
        )
    return RunProvenanceIdentity(
        identity_generation=identity_generation,
        source_grammar_receipt=_source_grammar_receipt(source_grammar_receipt),
        spec_binding=_provenance_spec_binding(spec_binding),
        authority_versions=_freeze_provenance_mapping(
            authority_versions, location="authority_versions"
        ),
        code_inventory_digest=_provenance_sha256(
            code_inventory_digest, location="code_inventory_digest"
        ),
        artifact_protocol_inventory=_freeze_provenance_mapping(
            artifact_protocol_inventory, location="artifact_protocol_inventory"
        ),
        run_request=_freeze_provenance_mapping(run_request, location="run_request"),
        execution_receipt=_freeze_provenance_mapping(
            execution_receipt, location="execution_receipt"
        ),
    )


@dataclass(frozen=True, slots=True)
class NodeReuseIdentity:
    """One semantic reuse key and the closed payload from which it was made."""

    key: str
    payload: FrozenMap

    def to_wire(self) -> dict[str, object]:
        return {"node_reuse_key": self.key, "payload": _wire(self.payload)}


def _reject_provenance_fields(
    value: object,
    *,
    location: str,
) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(set(value) & _PROVENANCE_ONLY_FIELDS)
        if forbidden:
            raise CapabilityError(
                f"node reuse semantic inputs contain provenance fields at "
                f"{location}: {forbidden!r}"
            )
        for key, child in value.items():
            _reject_provenance_fields(child, location=f"{location}/{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, child in enumerate(value):
            _reject_provenance_fields(child, location=f"{location}/{index}")
    elif isinstance(value, str) and value in _PROVENANCE_ONLY_VALUES:
        raise CapabilityError(
            f"node reuse semantic inputs contain provenance or operational receipt "
            f"markers at {location}: {value!r}"
        )


def node_reuse_identity(
    node: CompiledNode,
    *,
    behavior_relevant_run_inputs: Mapping[str, object],
    transitive_input_content_hashes: Mapping[str, str],
    implementation_dependency_sha256: str,
    rng_behavior_inputs: RNGBehaviorIdentity,
    source_behavior_inputs: SourceBehaviorIdentity,
    artifact_materializer_abis: Mapping[str, object],
    output_sensitive_backend_abi: Mapping[str, object] | None = None,
) -> NodeReuseIdentity:
    """Derive the runtime semantic reuse key, never a run-provenance key.

    ``run_provenance_identity``, binding generation, configuration-authority
    mode, and broker access receipts are intentionally impossible inputs to
    this API. Artifact content identities and actual RNG behavior inputs remain
    semantic.
    """

    if not isinstance(node, CompiledNode):
        raise TypeError("node_reuse_identity requires a CompiledNode")
    _compiled_contract_consistent(node)
    if not isinstance(rng_behavior_inputs, RNGBehaviorIdentity):
        raise TypeError(
            "node_reuse_identity requires a broker-issued RNGBehaviorIdentity"
        )
    if not isinstance(source_behavior_inputs, SourceBehaviorIdentity):
        raise TypeError(
            "node_reuse_identity requires a broker-issued SourceBehaviorIdentity"
        )
    try:
        rng_behavior_inputs.validate_issued()
    except BrokerContractError as error:
        raise CapabilityError("RNG behavior identity is not broker-issued") from error
    if rng_behavior_inputs.owner.kind != "producer_node" or (
        rng_behavior_inputs.owner.id != node.id
    ):
        raise CapabilityError("RNG behavior identity owner differs from compiled node")
    try:
        source_behavior_inputs.validate_issued()
    except BrokerContractError as error:
        raise CapabilityError(
            "source behavior identity is not broker-issued"
        ) from error
    if source_behavior_inputs.owner.kind != "producer_node" or (
        source_behavior_inputs.owner.id != node.id
    ):
        raise CapabilityError(
            "source behavior identity owner differs from compiled node"
        )
    if not source_behavior_inputs.same_session(rng_behavior_inputs):
        raise CapabilityError(
            "RNG and source behavior identities come from different broker sessions"
        )
    if (
        rng_behavior_inputs.protocol_id != "legacy-v1"
        or rng_behavior_inputs.protocol_sha256 != node.seed_protocol_sha256
    ):
        raise CapabilityError(
            "RNG behavior identity protocol differs from compiled node"
        )
    rng_wire = rng_behavior_inputs.to_wire()
    raw_rng_sites = rng_wire["sites"]
    if not isinstance(raw_rng_sites, list):  # pragma: no cover - typed broker output
        raise CapabilityError("RNG behavior identity sites must be an array")
    expected_rng_sites = [
        {
            "site_id": site.id,
            "stream": f"stream:{site.stream}",
            "contract_sha256": sha256_json(_wire(site.contract)),
        }
        for site in node.seed_sites
    ]
    actual_rng_sites = []
    for row in raw_rng_sites:
        if not isinstance(row, Mapping):
            raise CapabilityError("RNG behavior identity site must be an object")
        actual_rng_sites.append(
            {
                "site_id": row.get("site_id"),
                "stream": row.get("stream"),
                "contract_sha256": row.get("contract_sha256"),
            }
        )
    if actual_rng_sites != expected_rng_sites:
        raise CapabilityError("RNG behavior identity sites differ from compiled node")
    if len(implementation_dependency_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in implementation_dependency_sha256
    ):
        raise CapabilityError("implementation dependency digest must be sha256")
    input_hashes = dict(sorted(transitive_input_content_hashes.items()))
    for artifact_id, digest in input_hashes.items():
        if (
            not artifact_id
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CapabilityError(
                f"transitive input content hash is invalid for {artifact_id!r}"
            )
    source_wire = source_behavior_inputs.to_wire()
    raw_sources = source_wire["sources"]
    if not isinstance(raw_sources, list):  # pragma: no cover - typed broker output
        raise CapabilityError("source behavior identities must contain an array")
    for row in raw_sources:
        if not isinstance(row, Mapping):  # pragma: no cover - typed broker output
            raise CapabilityError("source behavior identity row must be an object")
        source_id = row.get("source_id")
        digest = row.get("content_sha256")
        assert isinstance(source_id, str) and isinstance(digest, str)
        identity_key = f"declared_source:{source_id}"
        existing = input_hashes.get(identity_key)
        if existing is not None and existing != digest:
            raise CapabilityError(
                f"transitive input hash conflicts with declared source {source_id!r}"
            )
        input_hashes[identity_key] = digest
    input_hashes = dict(sorted(input_hashes.items()))
    semantic_inputs = {
        "behavior_relevant_run_inputs": dict(behavior_relevant_run_inputs),
        "rng_behavior_inputs": rng_wire,
        "source_behavior_inputs": source_wire,
        "artifact_materializer_abis": dict(artifact_materializer_abis),
        "output_sensitive_backend_abi": None
        if output_sensitive_backend_abi is None
        else dict(output_sensitive_backend_abi),
    }
    _reject_provenance_fields(semantic_inputs, location="node_reuse")
    payload = {
        "domain": _NODE_REUSE_KEY_DOMAIN,
        "compiler_ir_abi": _wire(node.compiler_ir_abi),
        "resolved_transitive_node_slice": node.node_slice_wire(),
        "behavior_relevant_run_inputs": semantic_inputs["behavior_relevant_run_inputs"],
        "transitive_input_content_hashes": input_hashes,
        "declared_source_behavior": semantic_inputs["source_behavior_inputs"],
        "per_node_implementation_and_dependency": {
            "kernel": {
                "ref": node.kernel_ref,
                "implementation_sha256": node.kernel_implementation_sha256,
            },
            "dependency_sha256": implementation_dependency_sha256,
        },
        "rng_protocol_and_seed_material": {
            "seed_protocol_sha256": node.seed_protocol_sha256,
            "behavior_inputs": semantic_inputs["rng_behavior_inputs"],
        },
        "input_and_output_artifact_contracts": {
            "inputs": [_wire(value) for value in node.inputs],
            "outputs": [_wire(value) for value in node.outputs],
        },
        "per_artifact_materializer_abi": semantic_inputs["artifact_materializer_abis"],
        "output_sensitive_backend_abi": semantic_inputs["output_sensitive_backend_abi"],
    }
    _reject_provenance_fields(payload, location="node_reuse_payload")
    frozen = freeze_json(payload)
    if not isinstance(frozen, FrozenMap):  # pragma: no cover - root is literal map
        raise AssertionError("node reuse payload did not freeze to an object")
    return NodeReuseIdentity(key=sha256_json(payload), payload=frozen)


def _pickle_sha256(value: object) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _projection_sha256(projection: ImmutableFrameProjection) -> str:
    return _pickle_sha256(projection._parts())


def _patch_sha256(patch: KernelPatch) -> str:
    return _pickle_sha256(
        {
            "structural_delta": patch.structural_delta.value,
            "tables": {
                name: _deep_copy_frame(table) for name, table in patch.tables.items()
            },
            "drop_entities": tuple(sorted(patch.drop_entities)),
            "entity_keys": dict(patch.entity_keys),
            "membership_columns": {
                name: tuple(columns)
                for name, columns in patch.membership_columns.items()
            },
            "links": {
                name: _deep_copy_frame(table) for name, table in patch.links.items()
            },
            "drop_links": tuple(sorted(patch.drop_links)),
            "weights": {
                name: (weight.kind, weight.values)
                for name, weight in patch.weights.items()
            },
            "drop_weights": tuple(sorted(patch.drop_weights)),
            "strata": (
                None if patch.strata is None else _deep_copy_series(patch.strata)
            ),
            "replace_strata": patch.replace_strata,
            "mass_history": copy.deepcopy(patch.mass_history),
            "metadata": (
                None if patch.metadata is None else copy.deepcopy(dict(patch.metadata))
            ),
            "virtual_writes": copy.deepcopy(dict(patch.virtual_writes)),
            "virtual_deletes": tuple(sorted(patch.virtual_deletes)),
            "row_atoms": {
                entity: {row_id: tuple(sorted(atoms)) for row_id, atoms in rows.items()}
                for entity, rows in patch.row_atoms.items()
            },
        }
    )


def _validated_envelope_sha256(
    *,
    node_id: str,
    node_key: str,
    attempt: int,
    attempt_scope: str | None,
    base_sha256: str,
    result_sha256: str,
    patch_sha256: str,
    diff_sha256: str,
    broker_receipt_sha256: str,
) -> str:
    return sha256_json(
        {
            "domain": "microcosm.spec-engine.validated-patch-envelope.v1",
            "node_id": node_id,
            "node_key": node_key,
            "attempt": attempt,
            "attempt_scope": attempt_scope,
            "base_sha256": base_sha256,
            "result_sha256": result_sha256,
            "patch_sha256": patch_sha256,
            "diff_sha256": diff_sha256,
            # Operational binding: this does not enter patch bytes, artifact
            # bytes, or node reuse.  It prevents swapping an audit receipt
            # between otherwise identical executions.
            "broker_receipt_sha256": broker_receipt_sha256,
        }
    )


def _stable_sort(values: Iterable[Hashable]) -> tuple[Hashable, ...]:
    return tuple(sorted(values, key=lambda value: (type(value).__name__, repr(value))))


def _row_ids(table: pd.DataFrame, key: str) -> tuple[Hashable, ...]:
    return tuple(table[key].tolist())


def _axis_contract(index: pd.Index) -> dict[str, object]:
    """Describe axis metadata independently from its label values."""

    contract: dict[str, object] = {
        "class": f"{type(index).__module__}.{type(index).__qualname__}",
        "names": tuple(index.names),
        "dtype": repr(getattr(index, "dtype", None)),
        "nlevels": index.nlevels,
    }
    if isinstance(index, pd.CategoricalIndex):
        contract["categories"] = tuple(index.categories.tolist())
        contract["ordered"] = index.ordered
    if isinstance(index, pd.MultiIndex):
        contract["level_contracts"] = tuple(
            _axis_contract(level) for level in index.levels
        )
    return contract


def _frame_contract(frame: pd.DataFrame) -> dict[str, object]:
    """Return non-cell DataFrame state that kernels may never rewrite."""

    metadata_names = tuple(getattr(frame, "_metadata", ()))
    return {
        "class": f"{type(frame).__module__}.{type(frame).__qualname__}",
        "allows_duplicate_labels": frame.flags.allows_duplicate_labels,
        "index_axis": _axis_contract(frame.index),
        "subclass_metadata": {
            name: copy.deepcopy(getattr(frame, name, None)) for name in metadata_names
        },
    }


def _series_contract_equal(
    left: pd.Series | None,
    right: pd.Series | None,
) -> bool:
    """Compare stable Series metadata while excluding row labels and values."""

    if left is None or right is None:
        return left is right
    metadata_names = tuple(getattr(left, "_metadata", ()))
    return all(
        (
            type(left) is type(right),
            _exact_object_equal(left.dtype, right.dtype),
            _exact_object_equal(left.name, right.name),
            _exact_object_equal(left.attrs, right.attrs),
            left.flags.allows_duplicate_labels == right.flags.allows_duplicate_labels,
            _exact_object_equal(
                metadata_names,
                tuple(getattr(right, "_metadata", ())),
            ),
            _exact_object_equal(
                {
                    name: copy.deepcopy(getattr(left, name, None))
                    for name in metadata_names
                },
                {
                    name: copy.deepcopy(getattr(right, name, None))
                    for name in metadata_names
                },
            ),
            _exact_object_equal(
                _axis_contract(left.index),
                _axis_contract(right.index),
            ),
        )
    )


def _stable_index_changes(
    before_ids: Sequence[Hashable],
    before_index: pd.Index,
    after_ids: Sequence[Hashable],
    after_index: pd.Index,
) -> frozenset[Hashable]:
    before_by_id = dict(zip(before_ids, before_index.tolist(), strict=True))
    after_by_id = dict(zip(after_ids, after_index.tolist(), strict=True))
    return frozenset(
        row_id
        for row_id in set(before_by_id) & set(after_by_id)
        if not _exact_object_equal(before_by_id[row_id], after_by_id[row_id])
    )


def _table_diff(
    entity: str,
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
) -> TableDiff | None:
    before_table = before._tables.get(entity)
    after_table = after._tables.get(entity)
    if before_table is None:
        assert after_table is not None
        after_ids = _row_ids(after_table, after._entity_keys[entity])
        changes = tuple(
            CellChange(
                entity=entity,
                column=column,
                row_id=row_id,
                before_present=False,
                after_present=True,
                before=None,
                after=copy.deepcopy(value),
            )
            for row_id, row in after_table.set_index(
                after._entity_keys[entity], drop=False
            ).iterrows()
            for column, value in row.items()
        )
        return TableDiff(
            entity=entity,
            added=True,
            removed=False,
            before_row_ids=(),
            after_row_ids=after_ids,
            before_columns=(),
            after_columns=tuple(after_table.columns),
            added_row_ids=frozenset(after_ids),
            removed_row_ids=frozenset(),
            order_changed=False,
            index_changed=False,
            stable_index_changed_row_ids=frozenset(),
            column_order_changed=False,
            column_axis_name_changed=False,
            column_axis_contract_changed=False,
            attrs_changed=bool(after_table.attrs),
            frame_contract_changed=True,
            added_columns=tuple(after_table.columns),
            removed_columns=(),
            dtype_changes=(),
            cell_changes=changes,
        )
    if after_table is None:
        before_ids = _row_ids(before_table, before._entity_keys[entity])
        changes = tuple(
            CellChange(
                entity=entity,
                column=column,
                row_id=row_id,
                before_present=True,
                after_present=False,
                before=copy.deepcopy(value),
                after=None,
            )
            for row_id, row in before_table.set_index(
                before._entity_keys[entity], drop=False
            ).iterrows()
            for column, value in row.items()
        )
        return TableDiff(
            entity=entity,
            added=False,
            removed=True,
            before_row_ids=before_ids,
            after_row_ids=(),
            before_columns=tuple(before_table.columns),
            after_columns=(),
            added_row_ids=frozenset(),
            removed_row_ids=frozenset(before_ids),
            order_changed=False,
            index_changed=False,
            stable_index_changed_row_ids=frozenset(),
            column_order_changed=False,
            column_axis_name_changed=False,
            column_axis_contract_changed=False,
            attrs_changed=bool(before_table.attrs),
            frame_contract_changed=True,
            added_columns=(),
            removed_columns=tuple(before_table.columns),
            dtype_changes=(),
            cell_changes=changes,
        )

    before_key = before._entity_keys[entity]
    after_key = after._entity_keys[entity]
    before_ids = _row_ids(before_table, before_key)
    after_ids = _row_ids(after_table, after_key)
    before_id_set = frozenset(before_ids)
    after_id_set = frozenset(after_ids)
    before_columns = tuple(before_table.columns)
    after_columns = tuple(after_table.columns)
    common_columns = tuple(
        column for column in before_columns if column in after_columns
    )
    added_columns = tuple(
        column for column in after_columns if column not in before_columns
    )
    removed_columns = tuple(
        column for column in before_columns if column not in after_columns
    )
    dtype_changes = tuple(
        column
        for column in common_columns
        if not _exact_object_equal(
            before_table[column].dtype,
            after_table[column].dtype,
        )
    )
    before_rows = before_table.set_index(before_key, drop=False)
    after_rows = after_table.set_index(after_key, drop=False)
    cell_changes: list[CellChange] = []
    union_columns = tuple(dict.fromkeys((*before_columns, *after_columns)))
    for row_id in _stable_sort(before_id_set | after_id_set):
        before_row_present = row_id in before_id_set
        after_row_present = row_id in after_id_set
        for column in union_columns:
            before_present = before_row_present and column in before_columns
            after_present = after_row_present and column in after_columns
            left = before_rows.at[row_id, column] if before_present else None
            right = after_rows.at[row_id, column] if after_present else None
            if before_present == after_present and (
                not before_present or _exact_object_equal(left, right)
            ):
                continue
            cell_changes.append(
                CellChange(
                    entity=entity,
                    column=column,
                    row_id=row_id,
                    before_present=before_present,
                    after_present=after_present,
                    before=copy.deepcopy(left),
                    after=copy.deepcopy(right),
                )
            )
    shared_after_order = tuple(
        column for column in after_columns if column in set(common_columns)
    )
    column_order_changed = common_columns != shared_after_order
    index_changed = not _exact_object_equal(before_table.index, after_table.index)
    stable_index_changed_row_ids = _stable_index_changes(
        before_ids,
        before_table.index,
        after_ids,
        after_table.index,
    )
    column_axis_name_changed = before_table.columns.name != after_table.columns.name
    column_axis_contract_changed = not _exact_object_equal(
        _axis_contract(before_table.columns),
        _axis_contract(after_table.columns),
    )
    attrs_changed = not _exact_object_equal(before_table.attrs, after_table.attrs)
    frame_contract_changed = not _exact_object_equal(
        _frame_contract(before_table),
        _frame_contract(after_table),
    )
    changed = bool(
        before_key != after_key
        or before_ids != after_ids
        or index_changed
        or column_order_changed
        or column_axis_name_changed
        or column_axis_contract_changed
        or attrs_changed
        or frame_contract_changed
        or added_columns
        or removed_columns
        or dtype_changes
        or cell_changes
    )
    if not changed:
        return None
    return TableDiff(
        entity=entity,
        added=False,
        removed=False,
        before_row_ids=before_ids,
        after_row_ids=after_ids,
        before_columns=before_columns,
        after_columns=after_columns,
        added_row_ids=after_id_set - before_id_set,
        removed_row_ids=before_id_set - after_id_set,
        order_changed=(before_id_set == after_id_set and before_ids != after_ids),
        index_changed=index_changed,
        stable_index_changed_row_ids=stable_index_changed_row_ids,
        column_order_changed=column_order_changed,
        column_axis_name_changed=column_axis_name_changed,
        column_axis_contract_changed=column_axis_contract_changed,
        attrs_changed=attrs_changed,
        frame_contract_changed=frame_contract_changed,
        added_columns=added_columns,
        removed_columns=removed_columns,
        dtype_changes=dtype_changes,
        cell_changes=tuple(cell_changes),
    )


def _link_row_ids(
    table: pd.DataFrame,
    targets: Mapping[str, str],
) -> tuple[tuple[Hashable, ...], ...]:
    columns = tuple(targets)
    return tuple(
        tuple(row)
        for row in table.loc[:, list(columns)].itertuples(index=False, name=None)
    )


def _link_diff(
    name: str,
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
) -> LinkDiff | None:
    before_table = before._links.get(name)
    after_table = after._links.get(name)
    targets = before._link_targets.get(name) or after._link_targets.get(name)
    if targets is None:
        raise StructuralDiffError(f"link {name!r} has no immutable target contract")
    before_ids = () if before_table is None else _link_row_ids(before_table, targets)
    after_ids = () if after_table is None else _link_row_ids(after_table, targets)
    before_set = frozenset(before_ids)
    after_set = frozenset(after_ids)
    before_columns = () if before_table is None else tuple(before_table.columns)
    after_columns = () if after_table is None else tuple(after_table.columns)
    common_columns = tuple(
        column for column in before_columns if column in after_columns
    )
    added_columns = tuple(
        column for column in after_columns if column not in before_columns
    )
    removed_columns = tuple(
        column for column in before_columns if column not in after_columns
    )
    dtype_changes = tuple(
        column
        for column in common_columns
        if before_table is not None
        and after_table is not None
        and not _exact_object_equal(
            before_table[column].dtype,
            after_table[column].dtype,
        )
    )
    shared_after_order = tuple(
        column for column in after_columns if column in set(common_columns)
    )
    column_order_changed = common_columns != shared_after_order
    column_axis_name_changed = (
        None if before_table is None else before_table.columns.name
    ) != (None if after_table is None else after_table.columns.name)
    column_axis_contract_changed = (
        before_table is not None
        and after_table is not None
        and not _exact_object_equal(
            _axis_contract(before_table.columns),
            _axis_contract(after_table.columns),
        )
    )

    def keyed_rows(
        table: pd.DataFrame | None,
    ) -> Mapping[tuple[Hashable, ...], pd.Series]:
        if table is None:
            return {}
        ids = _link_row_ids(table, targets)
        return {row_id: table.iloc[index] for index, row_id in enumerate(ids)}

    before_rows = keyed_rows(before_table)
    after_rows = keyed_rows(after_table)
    union_columns = tuple(dict.fromkeys((*before_columns, *after_columns)))
    cell_changes: list[LinkCellChange] = []
    for row_id in sorted(
        before_set | after_set,
        key=lambda value: tuple((type(part).__name__, repr(part)) for part in value),
    ):
        before_row_present = row_id in before_set
        after_row_present = row_id in after_set
        for column in union_columns:
            before_present = before_row_present and column in before_columns
            after_present = after_row_present and column in after_columns
            left = before_rows[row_id][column] if before_present else None
            right = after_rows[row_id][column] if after_present else None
            if before_present == after_present and (
                not before_present or _exact_object_equal(left, right)
            ):
                continue
            cell_changes.append(
                LinkCellChange(
                    link=name,
                    column=column,
                    row_id=row_id,
                    before_present=before_present,
                    after_present=after_present,
                    before=copy.deepcopy(left),
                    after=copy.deepcopy(right),
                )
            )
    index_changed = not (
        before_table is not None
        and after_table is not None
        and _exact_object_equal(before_table.index, after_table.index)
    )
    stable_index_changed_row_ids = (
        frozenset()
        if before_table is None or after_table is None
        else frozenset(
            _stable_index_changes(
                before_ids,
                before_table.index,
                after_ids,
                after_table.index,
            )
        )
    )
    attrs_changed = not _exact_object_equal(
        {} if before_table is None else before_table.attrs,
        {} if after_table is None else after_table.attrs,
    )
    frame_contract_changed = not (
        before_table is not None
        and after_table is not None
        and _exact_object_equal(
            _frame_contract(before_table),
            _frame_contract(after_table),
        )
    )
    changed = bool(
        before_table is None
        or after_table is None
        or before_ids != after_ids
        or index_changed
        or column_order_changed
        or column_axis_name_changed
        or column_axis_contract_changed
        or attrs_changed
        or frame_contract_changed
        or added_columns
        or removed_columns
        or dtype_changes
        or cell_changes
    )
    if not changed:
        return None
    return LinkDiff(
        name=name,
        added=before_table is None,
        removed=after_table is None,
        before_row_ids=before_ids,
        after_row_ids=after_ids,
        before_columns=before_columns,
        after_columns=after_columns,
        added_row_ids=after_set - before_set,
        removed_row_ids=before_set - after_set,
        order_changed=before_set == after_set and before_ids != after_ids,
        index_changed=index_changed,
        stable_index_changed_row_ids=stable_index_changed_row_ids,
        column_order_changed=column_order_changed,
        column_axis_name_changed=column_axis_name_changed,
        column_axis_contract_changed=column_axis_contract_changed,
        attrs_changed=attrs_changed,
        frame_contract_changed=frame_contract_changed,
        added_columns=added_columns,
        removed_columns=removed_columns,
        dtype_changes=dtype_changes,
        cell_changes=tuple(cell_changes),
    )


def diff_projections(
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
) -> FrameDiff:
    """Return an exact, tolerance-free diff across every projected surface."""

    table_diffs = tuple(
        result
        for entity in sorted(set(before._tables) | set(after._tables))
        if (result := _table_diff(entity, before, after)) is not None
    )
    link_diffs = tuple(
        result
        for name in sorted(set(before._links) | set(after._links))
        if (result := _link_diff(name, before, after)) is not None
    )
    weight_diffs: list[WeightDiff] = []
    for entity in sorted(set(before._weights) | set(after._weights)):
        left = before._weights.get(entity)
        right = after._weights.get(entity)
        if left is None or right is None:
            ids = (
                _row_ids(before._tables[entity], before._entity_keys[entity])
                if right is None
                else _row_ids(after._tables[entity], after._entity_keys[entity])
            )
            weight_diffs.append(
                WeightDiff(
                    entity,
                    values_changed=True,
                    kind_changed=True,
                    storage_order_changed=False,
                    changed_row_ids=frozenset(ids),
                )
            )
            continue
        left_values = left._internal_values()
        right_values = right._internal_values()
        kind_changed = left.kind != right.kind
        before_ids = _row_ids(before._tables[entity], before._entity_keys[entity])
        after_ids = _row_ids(after._tables[entity], after._entity_keys[entity])
        changed_ids: set[Hashable] = set(before_ids) ^ set(after_ids)
        before_position = {row_id: index for index, row_id in enumerate(before_ids)}
        after_position = {row_id: index for index, row_id in enumerate(after_ids)}
        for row_id in set(before_ids) & set(after_ids):
            if not _exact_object_equal(
                left_values[before_position[row_id]],
                right_values[after_position[row_id]],
            ):
                changed_ids.add(row_id)
        if kind_changed:
            changed_ids.update(after_ids)
        values_changed = bool(changed_ids)
        storage_order_changed = before_ids != after_ids or not _exact_object_equal(
            left_values, right_values
        )
        if not values_changed and not kind_changed and not storage_order_changed:
            continue
        weight_diffs.append(
            WeightDiff(
                entity,
                values_changed,
                kind_changed,
                storage_order_changed,
                frozenset(changed_ids),
            )
        )

    virtual_changes = tuple(
        VirtualChange(
            entity=key[0],
            column=key[1],
            before_present=key in before._virtual_receipts,
            after_present=key in after._virtual_receipts,
        )
        for key in sorted(
            {
                key
                for key in set(before._virtual_receipts) | set(after._virtual_receipts)
                if key not in before._virtual_receipts
                or key not in after._virtual_receipts
                or not _exact_object_equal(
                    before._virtual_receipts[key], after._virtual_receipts[key]
                )
            }
        )
    )
    result = FrameDiff(
        tables=table_diffs,
        links=link_diffs,
        weights=tuple(weight_diffs),
        strata_changed=not _exact_object_equal(before._strata, after._strata),
        mass_history_changed=not _exact_object_equal(
            before._mass_history, after._mass_history
        ),
        metadata_changed=not _exact_object_equal(before._metadata, after._metadata),
        virtual_changes=virtual_changes,
        entity_order_contract_changed=before._entity_order != after._entity_order,
        entity_key_contract_changed=before._entity_keys != after._entity_keys,
        membership_contract_changed=(
            before._membership_columns != after._membership_columns
            or not _exact_object_equal(
                before._membership_targets, after._membership_targets
            )
        ),
        link_order_contract_changed=before._link_order != after._link_order,
        link_contract_changed=not _exact_object_equal(
            before._link_targets, after._link_targets
        ),
        weight_order_contract_changed=before._weight_order != after._weight_order,
        strata_contract_changed=(
            before._strata_entity != after._strata_entity
            or not _series_contract_equal(before._strata, after._strata)
        ),
        row_atoms_changed=not _exact_object_equal(before._row_atoms, after._row_atoms),
    )
    if result.empty and _projection_sha256(before) != _projection_sha256(after):
        raise StructuralDiffError(
            "projection fingerprint changed outside the modeled structural diff"
        )
    return result


def _apply_unchecked(
    projection: ImmutableFrameProjection,
    patch: KernelPatch,
    *,
    registry: ClosedScopeRegistry,
    row_classifier: RegisteredRowClassifier | None,
    broker_session: BrokerSession,
) -> tuple[
    ImmutableFrameProjection,
    dict[str, dict[Hashable, Hashable]],
]:
    if patch.row_atoms:
        raise PatchScopeError(
            "kernels cannot issue row-scope classifications; use the trusted "
            "executor row classifier"
        )
    parts = projection._parts()
    tables = parts["tables"]
    entity_keys = parts["entity_keys"]
    memberships = parts["membership_columns"]
    links = parts["links"]
    weights = parts["weights"]
    virtual = parts["virtual_receipts"]
    row_atoms = parts["row_atoms"]
    assert isinstance(tables, dict)
    assert isinstance(entity_keys, dict)
    assert isinstance(memberships, dict)
    assert isinstance(links, dict)
    assert isinstance(weights, dict)
    assert isinstance(virtual, dict)
    assert isinstance(row_atoms, dict)

    if patch.drop_entities & set(patch.tables):
        raise ExecutorError("patch cannot replace and drop the same entity")
    for entity in patch.drop_entities:
        tables.pop(entity, None)
        entity_keys.pop(entity, None)
        memberships.pop(entity, None)
        weights.pop(entity, None)
        row_atoms.pop(entity, None)
    tables.update(
        {name: _deep_copy_frame(value) for name, value in patch.tables.items()}
    )
    entity_keys.update(patch.entity_keys)
    memberships.update(
        {name: tuple(columns) for name, columns in patch.membership_columns.items()}
    )

    if patch.drop_links & set(patch.links):
        raise ExecutorError("patch cannot replace and drop the same link")
    for name in patch.drop_links:
        links.pop(name, None)
    links.update({name: _deep_copy_frame(value) for name, value in patch.links.items()})

    if patch.drop_weights & set(patch.weights):
        raise ExecutorError("patch cannot replace and drop the same weights")
    for entity in patch.drop_weights:
        weights.pop(entity, None)
    weights.update({name: value.copy() for name, value in patch.weights.items()})

    for key in patch.virtual_deletes:
        virtual.pop(key, None)
    virtual.update(copy.deepcopy(dict(patch.virtual_writes)))
    classified_rows: dict[str, dict[Hashable, frozenset[str]]] = {}
    added_row_sources: dict[str, dict[Hashable, Hashable]] = {}
    for entity, table in tables.items():
        key = entity_keys[entity]
        after_ids = frozenset(table[key].tolist())
        before_rows = projection._row_atoms.get(entity, {})
        retained = {
            row_id: frozenset(atoms)
            for row_id, atoms in before_rows.items()
            if row_id in after_ids
        }
        added_ids = after_ids - set(before_rows)
        if added_ids:
            if row_classifier is None:
                raise PatchScopeError(
                    f"added rows for {entity!r} require a trusted row classifier"
                )
            if row_classifier.predicate_space != registry.predicate_space:
                raise PatchScopeError(
                    "row classifier predicate space differs from compiled registry"
                )
            with broker_session.classifier_scope():
                raw_classification = row_classifier.function(
                    entity,
                    _deep_copy_frame(table),
                    key,
                    frozenset(added_ids),
                    registry,
                )
            if not isinstance(raw_classification, Mapping) or set(
                raw_classification
            ) != set(added_ids):
                raise PatchScopeError(
                    f"row classifier for {entity!r} must classify added ids exactly"
                )
            before_ids = set(projection._row_atoms.get(entity, {}))
            if any(
                not isinstance(value, RowClassification)
                for value in raw_classification.values()
            ):
                raise PatchScopeError(
                    f"row classifier for {entity!r} must return RowClassification values"
                )
            normalized = {
                row_id: value.atoms for row_id, value in raw_classification.items()
            }
            sources = {
                row_id: value.source_row_id
                for row_id, value in raw_classification.items()
            }
            invalid_sources = {
                row_id: source
                for row_id, source in sources.items()
                if source not in before_ids
            }
            if invalid_sources:
                raise PatchScopeError(
                    f"row classifier for {entity!r} returned non-native sources "
                    f"{invalid_sources!r}"
                )
            try:
                authorized = registry.authorized_row_ids(
                    {"atoms": list(registry.universe)}, normalized
                )
            except ScopeAlgebraError as error:
                raise PatchScopeError(str(error)) from error
            if authorized != frozenset(added_ids):
                raise PatchScopeError(
                    f"row classifier for {entity!r} returned incomplete atoms"
                )
            retained.update(normalized)
            added_row_sources[entity] = sources
        if retained:
            classified_rows[entity] = retained
    row_atoms = classified_rows
    parts.update(
        {
            "tables": tables,
            "entity_keys": entity_keys,
            "membership_columns": memberships,
            "links": links,
            "weights": weights,
            "virtual_receipts": virtual,
            "row_atoms": row_atoms,
        }
    )
    if patch.replace_strata:
        parts["strata"] = (
            None if patch.strata is None else _deep_copy_series(patch.strata)
        )
    if patch.mass_history is not None:
        parts["mass_history"] = copy.deepcopy(tuple(patch.mass_history))
    if patch.metadata is not None:
        parts["metadata"] = copy.deepcopy(dict(patch.metadata))
    result = ImmutableFrameProjection(**parts)  # type: ignore[arg-type]
    return result, added_row_sources


def _compiled_contract_consistent(node: CompiledNode) -> None:
    if _wire(node.compiler_ir_abi) != current_compiler_ir_abi().to_wire():
        raise CapabilityError(
            f"compiled node {node.id!r} targets a different executor/compiler ABI"
        )
    for param in node.resolved_params:
        if sha256_json(_wire(param.value)) != param.value_sha256:
            raise CapabilityError(
                f"compiled node {node.id!r} resolved-param digest differs at "
                f"{param.path!r}"
            )
    if sha256_json(node.node_slice_wire()) != node.node_slice_sha256:
        raise CapabilityError(f"compiled node {node.id!r} node-slice digest differs")
    expected_node_key = sha256_json(
        {
            "domain": _NODE_KEY_DOMAIN,
            "compiler_ir_abi": _wire(node.compiler_ir_abi),
            "node_slice_sha256": node.node_slice_sha256,
            "kernel": {
                "ref": node.kernel_ref,
                "implementation_sha256": node.kernel_implementation_sha256,
            },
            "seed_protocol_sha256": node.seed_protocol_sha256,
        }
    )
    if expected_node_key != node.node_key:
        raise CapabilityError(f"compiled node {node.id!r} node key differs")
    transitive_ids = tuple(row.id for row in node.transitive_nodes)
    if (
        len(set(transitive_ids)) != len(transitive_ids)
        or node.id in transitive_ids
        or not set(node.depends_on) <= set(transitive_ids)
    ):
        raise CapabilityError(
            f"compiled node {node.id!r} transitive slice does not cover its "
            "direct dependencies"
        )
    effective_streams = tuple(dict.fromkeys(site.stream for site in node.seed_sites))
    if effective_streams != node.seed_streams:
        raise CapabilityError(
            f"compiled node {node.id!r} seed-site and stream grants differ"
        )
    authored = [
        param
        for param in node.resolved_params
        if param.path.endswith(f"/nodes/{node.id}")
    ]
    scopes = [
        param
        for param in node.resolved_params
        if param.path.endswith(f"/nodes/{node.id}/write_scopes")
    ]
    registries = [
        param
        for param in node.resolved_params
        if param.path.endswith("/scope_registry")
    ]
    ranks = [
        param
        for param in node.resolved_params
        if param.path.endswith(f"/nodes/{node.id}/execution_rank")
    ]
    dependencies = [
        param
        for param in node.resolved_params
        if param.path == f"/compiled/producer_graph/nodes/{node.id}/depends_on"
    ]
    inputs = [
        param
        for param in node.resolved_params
        if param.path == f"/compiled/producer_graph/nodes/{node.id}/inputs"
    ]
    outputs = [
        param
        for param in node.resolved_params
        if param.path == f"/compiled/producer_graph/nodes/{node.id}/outputs"
    ]
    classifiers = [
        param
        for param in node.resolved_params
        if param.path == f"/compiled/producer_graph/nodes/{node.id}/row_classifier"
    ]
    kernels = [
        param
        for param in node.resolved_params
        if param.path == f"/compiled/producer_graph/nodes/{node.id}/kernel"
    ]
    grants = [
        param
        for param in node.resolved_params
        if param.path == f"/compiled/effective_seed_grant@producer={node.id}"
    ]
    if (
        len(authored) != 1
        or len(scopes) != 1
        or len(registries) != 1
        or len(ranks) != 1
        or len(dependencies) != 1
        or len(inputs) != 1
        or len(outputs) != 1
        or len(kernels) != 1
        or len(classifiers) != 1
        or len(grants) != 1
    ):
        raise CapabilityError(
            f"compiled node {node.id!r} lacks one closed executor contract projection"
        )
    authored_row = _mapping(_wire(authored[0].value), location="authored node")
    if authored_row.get("id") != node.id:
        raise CapabilityError(f"compiled node {node.id!r} authored-id lift differs")
    if authored_row.get("kernel") != node.kernel_ref:
        raise CapabilityError(f"compiled node {node.id!r} kernel lift differs")
    kernel = _mapping(_wire(kernels[0].value), location="compiled kernel")
    if kernel != {
        "ref": node.kernel_ref,
        "implementation_sha256": node.kernel_implementation_sha256,
    }:
        raise CapabilityError(f"compiled node {node.id!r} kernel lift differs")
    if _wire(node.capabilities) != authored_row.get("capabilities"):
        raise CapabilityError(f"compiled node {node.id!r} capability lift differs")
    if _wire(node.mutations) != authored_row.get("mutations"):
        raise CapabilityError(f"compiled node {node.id!r} mutation lift differs")
    if [_wire(scope) for scope in node.write_scopes] != _wire(scopes[0].value):
        raise CapabilityError(f"compiled node {node.id!r} write-scope lift differs")
    if _wire(node.scope_registry) != _wire(registries[0].value):
        raise CapabilityError(f"compiled node {node.id!r} scope-registry lift differs")
    if node.execution_rank != _wire(ranks[0].value):
        raise CapabilityError(f"compiled node {node.id!r} execution-rank lift differs")
    if _wire(node.depends_on) != _wire(dependencies[0].value):
        raise CapabilityError(f"compiled node {node.id!r} dependency lift differs")
    if _wire(node.inputs) != _wire(inputs[0].value):
        raise CapabilityError(f"compiled node {node.id!r} input lift differs")
    if _wire(node.outputs) != _wire(outputs[0].value):
        raise CapabilityError(f"compiled node {node.id!r} output lift differs")
    expected_classifier_ref, expected_classifier_sha256 = row_classifier_contract(
        current_compiler_ir_abi(),
        node.scope_registry,
    )
    classifier = _mapping(
        _wire(classifiers[0].value),
        location="compiled row classifier",
    )
    if (
        node.row_classifier_ref != expected_classifier_ref
        or node.row_classifier_implementation_sha256 != expected_classifier_sha256
        or classifier
        != {
            "ref": expected_classifier_ref,
            "implementation_sha256": expected_classifier_sha256,
        }
    ):
        raise CapabilityError(f"compiled node {node.id!r} row-classifier lift differs")
    grant = _mapping(_wire(grants[0].value), location="effective seed grant")
    if grant.get("sites") != [site.to_wire() for site in node.seed_sites]:
        raise CapabilityError(f"compiled node {node.id!r} seed-site lift differs")


def _node_scope_registry(node: CompiledNode) -> ClosedScopeRegistry:
    try:
        return ClosedScopeRegistry.from_wire(_wire(node.scope_registry))
    except ScopeAlgebraError as error:
        raise CapabilityError(
            f"compiled node {node.id!r} has an invalid scope registry: {error}"
        ) from error


def _physical_contract_columns(node: CompiledNode) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = defaultdict(set)

    def add(value: object, *, location: str) -> None:
        row = _mapping(value, location=location)
        entity = row.get("entity")
        column = row.get("column")
        if (
            isinstance(entity, str)
            and entity
            and isinstance(column, str)
            and column
            and not column.startswith("@")
        ):
            allowed[entity].add(column)

    for input_index, value in enumerate(node.inputs):
        row = _mapping(_wire(value), location=f"inputs/{input_index}")
        add(row, location=f"inputs/{input_index}")
        alternatives = row.get("alternatives", [])
        if not isinstance(alternatives, list):
            raise CapabilityError(
                f"compiled node {node.id!r} input alternatives must be an array"
            )
        for group_index, group in enumerate(alternatives):
            if not isinstance(group, list):
                raise CapabilityError(
                    f"compiled node {node.id!r} input alternative group must be an array"
                )
            for cell_index, cell in enumerate(group):
                add(
                    cell,
                    location=(
                        f"inputs/{input_index}/alternatives/{group_index}/{cell_index}"
                    ),
                )
    for output_index, value in enumerate(node.outputs):
        add(_wire(value), location=f"outputs/{output_index}")
    for scope_index, value in enumerate(node.write_scopes):
        add(_wire(value), location=f"write_scopes/{scope_index}")
    return allowed


def _required_input_rows(
    projection: ImmutableFrameProjection,
    entity: str,
    required_scope: str,
    registry: ClosedScopeRegistry,
) -> frozenset[Hashable]:
    if entity not in projection._tables:
        return frozenset()
    try:
        atoms = registry.atoms(required_scope)
        return _scope_rows(projection, entity, atoms, registry)
    except (PatchScopeError, ScopeAlgebraError) as error:
        raise CapabilityError(
            f"input scope {required_scope!r} cannot be resolved for {entity!r}: {error}"
        ) from error


def _physical_input_state(
    projection: ImmutableFrameProjection,
    *,
    entity: str,
    column: str,
    value_kind: str,
    required_scope: str,
    registry: ClosedScopeRegistry,
) -> _InputCellState:
    table = projection._tables.get(entity)
    if table is None or column not in table.columns:
        return _InputCellState.MISSING
    required_rows = _required_input_rows(
        projection,
        entity,
        required_scope,
        registry,
    )
    key = projection._entity_keys[entity]
    selected = table.loc[table[key].isin(required_rows), column]
    if value_kind == "column_present":
        return _InputCellState.SATISFIED
    if value_kind == "non_null":
        if selected.isna().any():
            return _InputCellState.MISSING
        return _InputCellState.SATISFIED
    if value_kind == "finite_numeric":
        missing = selected.isna()
        present = selected.loc[~missing]
        if present.empty and missing.any():
            return _InputCellState.MISSING
        if pd.api.types.is_bool_dtype(
            selected.dtype
        ) or not pd.api.types.is_numeric_dtype(selected.dtype):
            return _InputCellState.INVALID
        try:
            if not np.isfinite(present.to_numpy(dtype=np.float64)).all():
                return _InputCellState.INVALID
        except (TypeError, ValueError):
            return _InputCellState.INVALID
        if missing.any():
            return _InputCellState.MISSING
        return _InputCellState.SATISFIED
    raise CapabilityError(f"unknown producer input value_kind {value_kind!r}")


def _virtual_input_state(value: object, value_kind: str) -> _InputCellState:
    if value_kind == "column_present":
        return _InputCellState.SATISFIED
    if value_kind == "non_null":
        if value is None:
            return _InputCellState.MISSING
        if pd.api.types.is_scalar(value):
            try:
                if bool(pd.isna(value)):
                    return _InputCellState.MISSING
            except (TypeError, ValueError):
                return _InputCellState.INVALID
        return _InputCellState.SATISFIED
    if value_kind == "finite_numeric":
        if isinstance(value, bool):
            return _InputCellState.INVALID
        try:
            array = np.asarray(value)
            missing = pd.isna(array)
            if bool(np.asarray(missing).all()):
                return _InputCellState.MISSING
            if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
                array.dtype, np.bool_
            ):
                return _InputCellState.INVALID
            present = array[~missing]
            if not np.isfinite(present).all():
                return _InputCellState.INVALID
            if bool(np.asarray(missing).any()):
                return _InputCellState.MISSING
            return _InputCellState.SATISFIED
        except (TypeError, ValueError):
            return _InputCellState.INVALID
    raise CapabilityError(f"unknown producer input value_kind {value_kind!r}")


def _input_cell_state(
    projection: ImmutableFrameProjection,
    value: object,
    *,
    required_scope: str,
    registry: ClosedScopeRegistry,
    location: str,
) -> _InputCellState:
    row = _mapping(value, location=location)
    entity = _string(row.get("entity"), location=f"{location}/entity")
    column = _string(row.get("column"), location=f"{location}/column")
    value_kind = _string(row.get("value_kind"), location=f"{location}/value_kind")
    if value_kind not in {"column_present", "finite_numeric", "non_null"}:
        raise CapabilityError(f"unknown producer input value_kind {value_kind!r}")
    if column.startswith("@"):
        key = (entity, column)
        if key not in projection._virtual_receipts:
            return _InputCellState.MISSING
        return _virtual_input_state(projection._virtual_receipts[key], value_kind)
    return _physical_input_state(
        projection,
        entity=entity,
        column=column,
        value_kind=value_kind,
        required_scope=required_scope,
        registry=registry,
    )


def _absence_receipt_present(
    projection: ImmutableFrameProjection,
    receipt_id: str,
) -> bool:
    return any(column == receipt_id for _entity, column in projection._virtual_receipts)


def _validate_required_inputs(
    node: CompiledNode,
    projection: ImmutableFrameProjection,
    registry: ClosedScopeRegistry,
) -> None:
    for input_index, value in enumerate(node.inputs):
        location = f"inputs/{input_index}"
        row = _mapping(_wire(value), location=location)
        entity = _string(row.get("entity"), location=f"{location}/entity")
        column = _string(row.get("column"), location=f"{location}/column")
        required_scope = _string(
            row.get("required_scope"),
            location=f"{location}/required_scope",
        )
        try:
            registry.atoms(required_scope)
        except ScopeAlgebraError as error:
            raise CapabilityError(
                f"compiled node {node.id!r} has an unknown input scope "
                f"{required_scope!r}"
            ) from error
        alternatives = row.get("alternatives", [])
        if not isinstance(alternatives, list):
            raise CapabilityError(f"{location}/alternatives must be an array")
        satisfied = False
        present_but_invalid = False
        if alternatives:
            for group_index, group in enumerate(alternatives):
                if not isinstance(group, list) or not group:
                    raise CapabilityError(
                        f"{location}/alternatives/{group_index} must be a non-empty "
                        "AND group"
                    )
                group_satisfied = True
                for cell_index, cell in enumerate(group):
                    cell_location = (
                        f"{location}/alternatives/{group_index}/{cell_index}"
                    )
                    cell_state = _input_cell_state(
                        projection,
                        cell,
                        required_scope=required_scope,
                        registry=registry,
                        location=cell_location,
                    )
                    if cell_state is not _InputCellState.SATISFIED:
                        group_satisfied = False
                        if cell_state is _InputCellState.INVALID:
                            present_but_invalid = True
                if group_satisfied:
                    satisfied = True
                    break
        else:
            satisfied = (
                _input_cell_state(
                    projection,
                    {
                        "entity": entity,
                        "column": column,
                        "value_kind": "column_present",
                    },
                    required_scope=required_scope,
                    registry=registry,
                    location=location,
                )
                is _InputCellState.SATISFIED
            )
        receipts = row.get("tolerated_absence_receipts", [])
        if not isinstance(receipts, list) or any(
            not isinstance(receipt, str) or not receipt for receipt in receipts
        ):
            raise CapabilityError(
                f"{location}/tolerated_absence_receipts must be a string array"
            )
        tolerated_absence = not present_but_invalid and any(
            _absence_receipt_present(projection, receipt) for receipt in receipts
        )
        if not satisfied and not tolerated_absence:
            raise CapabilityError(
                f"compiled node {node.id!r} required input {entity}.{column} is "
                "not satisfied by any declared alternative or absence receipt"
            )


def _validate_projection_contract(
    node: CompiledNode,
    projection: ImmutableFrameProjection,
    registry: ClosedScopeRegistry,
) -> None:
    """Refuse missing inputs and projection columns outside the compiled read set."""

    _validate_required_inputs(node, projection, registry)
    catalog = _compiled_column_catalog(node)
    missing_output_contracts = [
        (entity, column)
        for index, value in enumerate(node.outputs)
        for row in [_mapping(_wire(value), location=f"outputs/{index}")]
        for entity in [_string(row.get("entity"), location=f"outputs/{index}/entity")]
        for column in [_string(row.get("column"), location=f"outputs/{index}/column")]
        if not column.startswith("@") and (entity, column) not in catalog
    ]
    if missing_output_contracts:
        raise CapabilityError(
            f"compiled node {node.id!r} physical outputs lack column contracts: "
            f"{missing_output_contracts!r}"
        )

    allowed = _physical_contract_columns(node)
    for entity, table in projection._tables.items():
        implicit = {
            projection._entity_keys[entity],
            *projection._membership_columns.get(entity, ()),
        }
        undeclared = set(table.columns) - allowed.get(entity, set()) - implicit
        if undeclared:
            raise CapabilityError(
                f"compiled node {node.id!r} projection exposes undeclared reads "
                f"for {entity!r}: {sorted(undeclared)!r}"
            )
    for name, table in projection._links.items():
        undeclared = set(table.columns) - set(projection._link_targets[name])
        if undeclared:
            raise CapabilityError(
                f"compiled node {node.id!r} projection exposes undeclared link "
                f"columns for {name!r}: {sorted(undeclared)!r}"
            )


def _compiled_column_catalog(
    node: CompiledNode,
) -> dict[tuple[str, str], Mapping[str, object]]:
    params = [
        param
        for param in node.resolved_params
        if param.path == f"/compiled/columns@producer={node.id}"
    ]
    if len(params) > 1:
        raise CapabilityError(
            f"compiled node {node.id!r} has duplicate output column catalogs"
        )
    if not params:
        return {}
    value = _wire(params[0].value)
    if not isinstance(value, list):
        raise CapabilityError(
            f"compiled node {node.id!r} output column catalog must be an array"
        )
    catalog: dict[tuple[str, str], Mapping[str, object]] = {}
    for index, raw_row in enumerate(value):
        row = _mapping(raw_row, location=f"compiled columns/{index}")
        entity = _string(row.get("entity"), location=f"compiled columns/{index}/entity")
        key = _string(row.get("key"), location=f"compiled columns/{index}/key")
        prefix = f"{entity}."
        if not key.startswith(prefix) or len(key) == len(prefix):
            raise CapabilityError(
                f"compiled node {node.id!r} output catalog key {key!r} is malformed"
            )
        column = key[len(prefix) :]
        if (entity, column) in catalog:
            raise CapabilityError(
                f"compiled node {node.id!r} repeats output catalog {entity}.{column}"
            )
        catalog[(entity, column)] = row
    return catalog


def _validate_outputs(
    node: CompiledNode,
    after: ImmutableFrameProjection,
    diff: FrameDiff,
) -> None:
    declared_outputs = [
        (
            _string(row.get("entity"), location=f"outputs/{index}/entity"),
            _string(row.get("column"), location=f"outputs/{index}/column"),
        )
        for index, value in enumerate(node.outputs)
        for row in [_mapping(_wire(value), location=f"outputs/{index}")]
        if isinstance(row.get("column"), str) and not str(row["column"]).startswith("@")
    ]
    added = [
        (table.entity, column)
        for table in diff.tables
        for column in table.added_columns
    ]
    added_set = set(added)
    if not added_set <= set(declared_outputs):
        raise PatchScopeError(
            f"join added columns outside compiled outputs: {sorted(added_set - set(declared_outputs))!r}"
        )
    expected_order = [item for item in declared_outputs if item in added_set]
    if added != expected_order:
        raise StructuralDiffError(
            "join added-column order differs from compiled output order"
        )
    catalog = _compiled_column_catalog(node)
    for table_diff in diff.tables:
        if table_diff.added_columns and table_diff.after_columns != (
            *table_diff.before_columns,
            *table_diff.added_columns,
        ):
            raise StructuralDiffError("join columns must append after existing columns")
    for entity, column in declared_outputs:
        table = after._tables.get(entity)
        if table is None or column not in table.columns:
            raise StructuralDiffError(
                f"compiled output {entity}.{column} is absent after execution"
            )
        contract = catalog.get((entity, column))
        if contract is None:
            raise CapabilityError(
                f"compiled output {entity}.{column} lacks a compiled column contract"
            )
        expected_dtype = _string(
            contract.get("dtype"),
            location=f"compiled column {entity}.{column}/dtype",
        )
        if str(table[column].dtype) != expected_dtype:
            raise StructuralDiffError(
                f"compiled output {entity}.{column} dtype differs: "
                f"{table[column].dtype!s} != {expected_dtype!r}"
            )
        nullable = contract.get("nullable")
        if not isinstance(nullable, bool):
            raise CapabilityError(
                f"compiled column {entity}.{column} nullable must be boolean"
            )
        if not nullable and table[column].isna().any():
            raise StructuralDiffError(
                f"compiled output {entity}.{column} violates non-nullability"
            )


def _mutation_contracts(node: CompiledNode) -> dict[str, MutationContract]:
    row = _mapping(_wire(node.mutations), location="mutations")
    if set(row) != set(_MUTATION_AXES):
        raise CapabilityError(
            "mutations must declare exactly entity_keys/cardinality/links/"
            "memberships/order/weights/mass_history"
        )
    contracts: dict[str, MutationContract] = {}
    for axis in _MUTATION_AXES:
        contract = _mapping(row[axis], location=f"mutations/{axis}")
        if set(contract) != {"operation", "precondition", "postcondition"}:
            raise CapabilityError(
                f"mutations/{axis} must declare operation/precondition/postcondition"
            )
        operation = _string(
            contract["operation"], location=f"mutations/{axis}/operation"
        )
        precondition = _string(
            contract["precondition"], location=f"mutations/{axis}/precondition"
        )
        postcondition = _string(
            contract["postcondition"], location=f"mutations/{axis}/postcondition"
        )
        triple = (operation, precondition, postcondition)
        if triple not in _VALID_MUTATION_CONTRACTS[axis]:
            raise CapabilityError(
                f"mutations/{axis} has an unknown or axis-incompatible "
                f"operation/precondition/postcondition tuple {triple!r}"
            )
        contracts[axis] = MutationContract(*triple)
    return contracts


def _validate_before_dispatch(
    node: CompiledNode,
    capabilities: NodeCapabilities,
    contracts: Mapping[str, MutationContract],
    context: ExecutionContext,
) -> None:
    _compiled_contract_consistent(node)
    if (
        isinstance(context.attempt, bool)
        or not isinstance(context.attempt, int)
        or context.attempt < 0
    ):
        raise CapabilityError("execution attempt must be non-negative")
    if capabilities.determinism is Determinism.SEEDED and not node.seed_streams:
        raise CapabilityError(
            f"seeded node {node.id!r} has no effective compiled RNG grant"
        )
    if capabilities.determinism is Determinism.DETERMINISTIC and node.seed_streams:
        raise CapabilityError(
            f"deterministic node {node.id!r} unexpectedly has RNG grants"
        )
    if (
        capabilities.determinism is Determinism.NONDETERMINISTIC
        and capabilities.numeric_reproducibility is NumericReproducibility.BITWISE
    ):
        raise CapabilityError("nondeterministic kernels cannot claim bitwise output")
    if (
        capabilities.determinism is Determinism.NONDETERMINISTIC
        and context.require_byte_equivalence
    ):
        raise CapabilityError(
            "nondeterministic kernel refused in byte-equivalence mode"
        )
    required_effects = capabilities.effects - {Effect.NONE}
    granted_effects = context.normalized_effects()
    if required_effects != granted_effects:
        raise CapabilityError(
            "kernel effect grants must exactly match its compiled declaration; "
            f"required={sorted(required_effects)!r}, granted={sorted(granted_effects)!r}"
        )
    if (
        capabilities.retry_safety is RetrySafety.ATTEMPT_SCOPED
        and not context.attempt_scope
    ):
        raise CapabilityError("attempt-scoped kernel requires an attempt scope")
    if capabilities.retry_safety is RetrySafety.NONRETRYABLE and (
        context.attempt != 0 or context.resumed
    ):
        raise CapabilityError("nonretryable kernel cannot be retried or resumed")

    changed_axes = {
        axis
        for axis, contract in contracts.items()
        if contract.operation not in _PRESERVE_OPERATIONS
    }
    delta = capabilities.structural_delta
    incompatible = {
        contract.operation
        for contract in contracts.values()
        if contract.operation not in _DELTA_OPERATION_NAMES[delta]
    }
    if incompatible:
        raise CapabilityError(
            f"structural_delta {delta.value} has incompatible mutation operations "
            f"{sorted(incompatible)!r}"
        )
    required_by_delta = {
        StructuralDelta.FILTER: {"entity_keys", "cardinality"},
        StructuralDelta.EXPAND: {"entity_keys", "cardinality"},
        StructuralDelta.RELINK: {"links", "memberships"},
        StructuralDelta.REORDER: {"order"},
        StructuralDelta.REWEIGHT: {"weights"},
    }
    if delta in {StructuralDelta.NONE, StructuralDelta.JOIN} and changed_axes:
        raise CapabilityError(
            f"structural_delta {delta.value} conflicts with mutation axes "
            f"{sorted(changed_axes)!r}"
        )
    required = required_by_delta.get(delta, set())
    if required and not required <= changed_axes:
        raise CapabilityError(
            f"structural_delta {delta.value} lacks required mutation axes "
            f"{sorted(required - changed_axes)!r}"
        )


def _scope_atoms(
    scope: Mapping[str, object],
    registry: ClosedScopeRegistry,
) -> frozenset[str]:
    segments = scope.get("cell_segments")
    if not isinstance(segments, Sequence) or isinstance(segments, str | bytes):
        return registry.atoms(scope.get("row_scope"))
    result: set[str] = set()
    for index, value in enumerate(segments):
        segment = _mapping(value, location=f"cell_segments/{index}")
        predicate = segment.get("predicate")
        if predicate == "coverage_scope":
            result.update(registry.atoms(segment.get("coverage_scope")))
        elif predicate == "origin_clone":
            origin = _string(segment.get("origin"), location="origin_clone/origin")
            clone_index = segment.get("clone_index")
            if isinstance(clone_index, bool) or not isinstance(clone_index, int):
                raise PatchScopeError("origin_clone/clone_index must be an integer")
            atom = f"origin:{origin}/clone:{clone_index}"
            if atom not in registry.universe:
                raise PatchScopeError(
                    f"compiled origin/clone atom {atom!r} is outside the scope universe"
                )
            result.add(atom)
        else:
            raise PatchScopeError(
                f"unknown compiled cell-segment predicate {predicate!r}"
            )
    if not result:
        raise PatchScopeError("compiled write scope resolves to no row atoms")
    return frozenset(result)


def _scope_rows(
    projection: ImmutableFrameProjection,
    entity: str,
    allowed_atoms: frozenset[str],
    registry: ClosedScopeRegistry,
) -> frozenset[Hashable]:
    rows = projection._row_atoms.get(entity)
    if rows is None:
        raise PatchScopeError(f"entity {entity!r} has no closed runtime row atoms")
    try:
        return registry.authorized_row_ids({"atoms": sorted(allowed_atoms)}, rows)
    except ScopeAlgebraError as error:
        raise PatchScopeError(str(error)) from error


def _write_scope_rows(
    node: CompiledNode,
    *,
    entity: str,
    column: str | None,
    modes: frozenset[str],
    basis: ImmutableFrameProjection,
    registry: ClosedScopeRegistry,
) -> frozenset[Hashable]:
    allowed: set[Hashable] = set()
    matched = False
    for value in node.write_scopes:
        scope = _mapping(_wire(value), location="write_scope")
        if scope.get("entity") != entity or scope.get("mode") not in modes:
            continue
        if column is not None and scope.get("column") != column:
            continue
        matched = True
        allowed.update(
            _scope_rows(basis, entity, _scope_atoms(scope, registry), registry)
        )
    if not matched:
        label = f"{entity}.{column}" if column is not None else entity
        raise PatchScopeError(f"no compiled write scope authorizes {label}")
    return frozenset(allowed)


def _membership_cells_changed(
    before: ImmutableFrameProjection,
    table_diff: TableDiff,
) -> bool:
    memberships = set(before._membership_columns.get(table_diff.entity, ()))
    return any(change.column in memberships for change in table_diff.cell_changes)


def _axis_changes(
    before: ImmutableFrameProjection,
    diff: FrameDiff,
) -> dict[str, bool]:
    row_structure = any(
        table.added or table.removed or table.added_row_ids or table.removed_row_ids
        for table in diff.tables
    )
    cardinality = any(
        len(table.before_row_ids) != len(table.after_row_ids) for table in diff.tables
    )
    return {
        "entity_keys": row_structure or diff.entity_key_contract_changed,
        "cardinality": cardinality,
        "links": bool(diff.links) or diff.link_contract_changed,
        "memberships": diff.membership_contract_changed
        or any(_membership_cells_changed(before, table) for table in diff.tables),
        "order": any(
            table.order_changed
            or table.added_row_ids
            or table.removed_row_ids
            or (
                table.index_changed
                and not table.added_row_ids
                and not table.removed_row_ids
            )
            for table in diff.tables
        ),
        "weights": any(
            weight.values_changed or weight.kind_changed for weight in diff.weights
        ),
        "mass_history": diff.mass_history_changed,
    }


def _is_subsequence(left: Sequence[Hashable], right: Sequence[Hashable]) -> bool:
    iterator = iter(right)
    return all(any(candidate == value for candidate in iterator) for value in left)


def _stable_strata_index_changes(
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
) -> frozenset[Hashable]:
    if (
        before._strata_entity is None
        or before._strata_entity != after._strata_entity
        or before._strata is None
        or after._strata is None
    ):
        return frozenset()
    entity = before._strata_entity
    return _stable_index_changes(
        _row_ids(before._tables[entity], before._entity_keys[entity]),
        before._strata.index,
        _row_ids(after._tables[entity], after._entity_keys[entity]),
        after._strata.index,
    )


def _validate_delta_semantics(
    delta: StructuralDelta,
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
    diff: FrameDiff,
) -> None:
    tables = diff.tables
    added_rows = any(table.added or table.added_row_ids for table in tables)
    removed_rows = any(table.removed or table.removed_row_ids for table in tables)
    order_changed = any(table.order_changed for table in tables)
    index_only_changed = any(
        table.index_changed and not table.added_row_ids and not table.removed_row_ids
        for table in tables
    )
    membership_changed = any(
        _membership_cells_changed(before, table) for table in tables
    )
    stable_cell_changes = tuple(
        change
        for table in tables
        for change in table.cell_changes
        if change.before_present and change.after_present
    )
    stable_key_changes = tuple(
        change
        for change in stable_cell_changes
        if change.column == before._entity_keys[change.entity]
    )
    if diff.entity_order_contract_changed or diff.entity_key_contract_changed:
        raise StructuralDiffError("entity schema and key authority are immutable")
    if diff.membership_contract_changed:
        raise StructuralDiffError("membership-column authority is immutable")
    if diff.link_order_contract_changed or diff.link_contract_changed:
        raise StructuralDiffError("link schema and target authority are immutable")
    if diff.weight_order_contract_changed:
        raise StructuralDiffError("typed-weight schema authority is immutable")
    if any(table.added or table.removed for table in tables):
        raise StructuralDiffError("the compiled entity-table inventory is immutable")
    if stable_key_changes:
        raise StructuralDiffError("stable entity-key values are immutable")
    if any(table.removed_columns for table in tables):
        raise StructuralDiffError(
            "column removal is not a declared kernel patch effect"
        )
    if any(table.dtype_changes for table in tables):
        raise StructuralDiffError("dtype changes are not writable cell effects")
    if any(table.column_order_changed for table in tables):
        raise StructuralDiffError("existing entity-table column order is immutable")
    if any(
        table.column_axis_name_changed
        or table.column_axis_contract_changed
        or table.attrs_changed
        or table.frame_contract_changed
        for table in tables
    ):
        raise StructuralDiffError("entity-table axis metadata and attrs are immutable")
    if delta is not StructuralDelta.JOIN and any(
        table.added_columns for table in tables
    ):
        raise StructuralDiffError(
            "only structural_delta join may add an entity-table column"
        )
    if delta is not StructuralDelta.REORDER and any(
        table.stable_index_changed_row_ids for table in tables
    ):
        raise StructuralDiffError(
            "stable entity rows changed index labels outside a reorder contract"
        )
    if delta is StructuralDelta.REORDER and any(
        table.index_changed and not table.order_changed for table in tables
    ):
        raise StructuralDiffError(
            "reorder may not rewrite an index without reordering that entity"
        )
    if delta not in {StructuralDelta.FILTER, StructuralDelta.EXPAND} and any(
        link.index_changed for link in diff.links
    ):
        raise StructuralDiffError(
            "link index labels changed outside a cardinality contract"
        )
    if any(link.stable_index_changed_row_ids for link in diff.links):
        raise StructuralDiffError("stable link rows changed index labels")
    if delta is not StructuralDelta.REORDER and _stable_strata_index_changes(
        before, after
    ):
        raise StructuralDiffError("stable strata rows changed index labels")
    if any(
        link.added
        or link.removed
        or link.added_columns
        or link.removed_columns
        or link.dtype_changes
        or link.column_order_changed
        or link.column_axis_name_changed
        or link.column_axis_contract_changed
        or link.attrs_changed
        or link.frame_contract_changed
        for link in diff.links
    ):
        raise StructuralDiffError(
            "link-table contracts and column structure are immutable"
        )
    if diff.metadata_changed:
        raise StructuralDiffError(
            "frame metadata is immutable; use a declared virtual receipt output"
        )
    if set(before._weights) != set(after._weights):
        raise StructuralDiffError("typed-weight entity authority is immutable")
    if delta is not StructuralDelta.REWEIGHT and any(
        weight.kind_changed for weight in diff.weights
    ):
        raise StructuralDiffError("only reweight may change a typed-weight kind")
    if diff.strata_contract_changed:
        raise StructuralDiffError("strata schema, dtype, and metadata are immutable")

    if before._strata_entity != after._strata_entity:
        raise StructuralDiffError("the strata entity contract is immutable")
    if before._strata_entity is not None:
        entity = before._strata_entity
        before_ids = _row_ids(before._tables[entity], before._entity_keys[entity])
        after_ids = _row_ids(after._tables[entity], after._entity_keys[entity])
        assert before._strata is not None
        assert after._strata is not None
        before_strata = dict(zip(before_ids, before._strata.tolist(), strict=True))
        after_strata = dict(zip(after_ids, after._strata.tolist(), strict=True))
        changed_existing = {
            row_id
            for row_id in set(before_ids) & set(after_ids)
            if not _exact_object_equal(before_strata[row_id], after_strata[row_id])
        }
        if changed_existing:
            raise StructuralDiffError(
                f"strata changed for stable rows {_stable_sort(changed_existing)!r}"
            )

    if delta is StructuralDelta.NONE:
        if (
            added_rows
            or removed_rows
            or order_changed
            or index_only_changed
            or diff.links
            or any(
                weight.values_changed
                or weight.kind_changed
                or weight.storage_order_changed
                for weight in diff.weights
            )
        ):
            raise StructuralDiffError("structural_delta none changed frame structure")
        if diff.strata_changed or diff.mass_history_changed:
            raise StructuralDiffError(
                "structural_delta none changed strata or mass history"
            )
    elif delta is StructuralDelta.FILTER:
        if added_rows:
            raise StructuralDiffError("filter may not add rows")
        if not removed_rows:
            raise StructuralDiffError("filter patch did not remove rows")
        for table in tables:
            if not _is_subsequence(table.after_row_ids, table.before_row_ids):
                raise StructuralDiffError("filter must preserve surviving row order")
        if stable_cell_changes:
            raise StructuralDiffError("filter may not rewrite surviving row cells")
    elif delta is StructuralDelta.EXPAND:
        if removed_rows:
            raise StructuralDiffError("expand may not remove rows")
        if not added_rows:
            raise StructuralDiffError("expand patch did not add rows")
        for table in tables:
            if not _is_subsequence(table.before_row_ids, table.after_row_ids):
                raise StructuralDiffError("expand must retain existing rows in order")
        if stable_cell_changes:
            raise StructuralDiffError("expand may not rewrite native row cells")
    elif delta is StructuralDelta.JOIN:
        if added_rows or removed_rows or order_changed or index_only_changed:
            raise StructuralDiffError(
                "join requires a row-preserving compiled join contract in F1"
            )
        if not any(table.added_columns for table in tables):
            raise StructuralDiffError("join patch did not add a declared column")
        added_by_entity = {table.entity: set(table.added_columns) for table in tables}
        if any(
            change.column not in added_by_entity.get(change.entity, set())
            for change in stable_cell_changes
        ):
            raise StructuralDiffError("join may not rewrite pre-existing columns")
    elif delta is StructuralDelta.RELINK:
        if (
            added_rows
            or removed_rows
            or order_changed
            or index_only_changed
            or any(
                weight.values_changed or weight.kind_changed for weight in diff.weights
            )
        ):
            raise StructuralDiffError(
                "relink changed keys, cardinality, order, or weights"
            )
        if not membership_changed and not diff.links:
            raise StructuralDiffError("relink patch changed no membership or link")
        if any(
            change.column not in set(before._membership_columns.get(change.entity, ()))
            for change in stable_cell_changes
        ):
            raise StructuralDiffError("relink may only rewrite membership columns")
    elif delta is StructuralDelta.REORDER:
        semantic_weight_change = any(
            weight.values_changed or weight.kind_changed for weight in diff.weights
        )
        if added_rows or removed_rows or diff.links or semantic_weight_change:
            raise StructuralDiffError(
                "reorder changed keys, cardinality, links, or weights"
            )
        if not order_changed:
            raise StructuralDiffError("reorder patch did not reorder a table")
        if stable_cell_changes:
            raise StructuralDiffError("reorder may not rewrite row cells")
    elif delta is StructuralDelta.REWEIGHT:
        table_structure = any(
            table.added
            or table.removed
            or table.added_row_ids
            or table.removed_row_ids
            or table.order_changed
            or table.index_changed
            or table.removed_columns
            or table.dtype_changes
            for table in tables
        )
        if table_structure or diff.links or diff.strata_changed:
            raise StructuralDiffError("reweight changed a non-weight frame surface")
        if any(table.cell_changes for table in tables):
            raise StructuralDiffError("reweight may not rewrite row cells")
        if not diff.weights:
            raise StructuralDiffError("reweight patch did not change weights")

    # Row atom changes are allowed only for the added/removed keys of a true
    # cardinality delta. Existing stable row classifications are immutable.
    for entity in set(before._row_atoms) & set(after._row_atoms):
        common = set(before._row_atoms[entity]) & set(after._row_atoms[entity])
        changed = {
            row_id
            for row_id in common
            if before._row_atoms[entity][row_id] != after._row_atoms[entity][row_id]
        }
        if changed:
            raise StructuralDiffError(
                f"stable row atoms changed for {entity}: {_stable_sort(changed)!r}"
            )


def _link_membership_mirror(
    projection: ImmutableFrameProjection,
    link_name: str,
) -> tuple[str, str, tuple[tuple[Hashable, Hashable], ...]] | None:
    """Resolve a link as an exact ordered mirror of one membership column."""

    link = projection._links[link_name]
    targets = projection._link_targets[link_name]
    matches: list[tuple[str, str, tuple[tuple[Hashable, Hashable], ...]]] = []
    for source_entity, membership_targets in projection._membership_targets.items():
        source_columns = [
            column for column, target in targets.items() if target == source_entity
        ]
        if len(source_columns) != 1:
            continue
        source_column = source_columns[0]
        source_key = projection._entity_keys[source_entity]
        for membership_column, target_entity in membership_targets.items():
            target_columns = [
                column for column, target in targets.items() if target == target_entity
            ]
            if membership_column in target_columns:
                target_column = membership_column
            elif len(target_columns) == 1:
                target_column = target_columns[0]
            else:
                continue
            if set(link.columns) != {source_column, target_column}:
                continue
            expected = tuple(
                zip(
                    projection._tables[source_entity][source_key].tolist(),
                    projection._tables[source_entity][membership_column].tolist(),
                    strict=True,
                )
            )
            actual = tuple(
                link.loc[:, [source_column, target_column]].itertuples(
                    index=False, name=None
                )
            )
            if _exact_object_equal(actual, expected):
                matches.append((source_entity, membership_column, expected))
    if len(matches) != 1:
        return None
    return matches[0]


def _validate_link_changes(
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
    diff: FrameDiff,
) -> None:
    """Allow changed link bytes only when they exactly mirror scoped membership."""

    for link in diff.links:
        before_mirror = _link_membership_mirror(before, link.name)
        after_mirror = _link_membership_mirror(after, link.name)
        if (
            before_mirror is None
            or after_mirror is None
            or before_mirror[:2] != after_mirror[:2]
            or _exact_object_equal(before_mirror[2], after_mirror[2])
        ):
            raise PatchScopeError(
                f"link {link.name!r} change is not an exact changed membership mirror"
            )


def _assert_projection_invariants(
    projection: ImmutableFrameProjection,
    *,
    label: str,
) -> None:
    try:
        ImmutableFrameProjection(**projection._parts())  # type: ignore[arg-type]
    except (ExecutorError, TypeError, ValueError) as error:  # pragma: no cover
        raise StructuralDiffError(f"{label} projection invariants are false") from error


def _assert_native_clone_index_zero(
    projection: ImmutableFrameProjection,
) -> None:
    malformed: list[tuple[str, Hashable, tuple[str, ...]]] = []
    for entity, rows in projection._row_atoms.items():
        for row_id, atoms in rows.items():
            clone_atoms = tuple(sorted(atom for atom in atoms if "/clone:" in atom))
            if len(clone_atoms) != 1 or not clone_atoms[0].endswith("/clone:0"):
                malformed.append((entity, row_id, clone_atoms))
    if malformed:
        raise StructuralDiffError(
            f"native_clone_index_zero precondition is false for {malformed[:5]!r}"
        )


def _assert_clone_roles_materialized(
    diff: FrameDiff,
    after: ImmutableFrameProjection,
    added_row_sources: Mapping[str, Mapping[Hashable, Hashable]],
) -> None:
    for table in diff.tables:
        if not table.added_row_ids:
            continue
        sources = added_row_sources.get(table.entity, {})
        if set(sources) != set(table.added_row_ids):
            raise StructuralDiffError(
                f"clone_roles_materialized lacks source lineage for {table.entity!r}"
            )
        for row_id in table.added_row_ids:
            atoms = after._row_atoms.get(table.entity, {}).get(row_id, frozenset())
            clone_atoms = tuple(atom for atom in atoms if "/clone:" in atom)
            if not clone_atoms or all(
                atom.endswith("/clone:0") for atom in clone_atoms
            ):
                raise StructuralDiffError(
                    "clone_roles_materialized lacks a non-native clone atom for "
                    f"{table.entity}.{row_id!r}"
                )


def _assert_clone_blocks_follow_native_rows(diff: FrameDiff) -> None:
    for table in diff.tables:
        if not table.added_row_ids:
            continue
        saw_added = False
        for row_id in table.after_row_ids:
            if row_id in table.added_row_ids:
                saw_added = True
            elif saw_added:
                raise StructuralDiffError(
                    "clone_blocks_follow_native_rows postcondition is false for "
                    f"{table.entity!r}"
                )


def _assert_clone_memberships_remapped(
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
    added_row_sources: Mapping[str, Mapping[Hashable, Hashable]],
) -> None:
    for source_entity, targets in before._membership_targets.items():
        source_lineage = added_row_sources.get(source_entity, {})
        if not source_lineage:
            continue
        before_key = before._entity_keys[source_entity]
        after_key = after._entity_keys[source_entity]
        before_rows = before._tables[source_entity].set_index(before_key, drop=False)
        after_rows = after._tables[source_entity].set_index(after_key, drop=False)
        for added_id, source_id in source_lineage.items():
            for column, target_entity in targets.items():
                native_target = before_rows.at[source_id, column]
                clone_target = after_rows.at[added_id, column]
                target_lineage = added_row_sources.get(target_entity, {})
                if clone_target != native_target and (
                    clone_target not in target_lineage
                    or not _exact_object_equal(
                        target_lineage[clone_target], native_target
                    )
                ):
                    raise StructuralDiffError(
                        "clone_memberships_reference_remapped_keys postcondition "
                        f"is false for {source_entity}.{column}/{added_id!r}"
                    )


def _assert_weights_aligned_to_surviving_keys(
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
    diff: FrameDiff,
) -> None:
    for weight in diff.weights:
        if weight.entity not in before._weights or weight.entity not in after._weights:
            continue
        stable_ids = set(
            _row_ids(before._tables[weight.entity], before._entity_keys[weight.entity])
        ) & set(
            _row_ids(after._tables[weight.entity], after._entity_keys[weight.entity])
        )
        changed_stable = stable_ids & set(weight.changed_row_ids)
        if changed_stable or weight.kind_changed:
            raise StructuralDiffError(
                "weights_aligned_to_surviving_keys postcondition is false for "
                f"{weight.entity!r}: {_stable_sort(changed_stable)!r}"
            )


def _assert_descendant_mass_conserved(
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
    added_row_sources: Mapping[str, Mapping[Hashable, Hashable]],
) -> None:
    for entity in set(before._weights) & set(after._weights):
        before_ids = _row_ids(before._tables[entity], before._entity_keys[entity])
        after_ids = _row_ids(after._tables[entity], after._entity_keys[entity])
        before_values = dict(
            zip(
                before_ids,
                before._weights[entity]._internal_values().tolist(),
                strict=True,
            )
        )
        after_values = dict(
            zip(
                after_ids,
                after._weights[entity]._internal_values().tolist(),
                strict=True,
            )
        )
        children: dict[Hashable, list[Hashable]] = defaultdict(list)
        for child, source in added_row_sources.get(entity, {}).items():
            children[source].append(child)
        for source_id, native_mass in before_values.items():
            descendant_mass = after_values[source_id] + sum(
                after_values[child] for child in children.get(source_id, ())
            )
            if not np.isclose(native_mass, descendant_mass, rtol=1e-9, atol=0.0):
                raise StructuralDiffError(
                    "household_mass_conserved postcondition is false for "
                    f"{entity}.{source_id!r}: {native_mass!r} != "
                    f"{descendant_mass!r}"
                )


def _validate_mutation_preconditions(
    contracts: Mapping[str, MutationContract],
    before: ImmutableFrameProjection,
) -> None:
    _assert_projection_invariants(before, label="pre-patch")
    invariant_preconditions = {
        "entity_keys_valid",
        "entity_cardinality_valid",
        "links_valid",
        "memberships_valid",
        "entity_order_valid",
        "weights_valid",
        "mass_history_valid",
        "native_entity_keys_unique",
        "native_memberships_valid",
        "native_entity_order_valid",
        "native_household_mass_finite",
    }
    for contract in contracts.values():
        if contract.precondition in invariant_preconditions:
            continue
        if contract.precondition == "link_tables_absent":
            if before._links:
                raise StructuralDiffError("link_tables_absent precondition is false")
            continue
        if contract.precondition == "native_clone_index_zero":
            _assert_native_clone_index_zero(before)
            continue
        raise CapabilityError(  # pragma: no cover - closed by the contract table
            f"mutation precondition {contract.precondition!r} is not executable"
        )


def _validate_mutation_contracts(
    contracts: Mapping[str, MutationContract],
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
    diff: FrameDiff,
    *,
    added_row_sources: Mapping[str, Mapping[Hashable, Hashable]],
) -> None:
    """Evaluate every declared mutation precondition and postcondition."""

    _assert_projection_invariants(after, label="post-patch")
    axis_changes = _axis_changes(before, diff)
    invariant_postconditions = {
        "entity_keys_unchanged",
        "entity_cardinality_unchanged",
        "links_unchanged",
        "memberships_unchanged",
        "entity_order_unchanged",
        "weights_unchanged",
        "mass_history_unchanged",
        "all_entity_keys_unique",
        "remaining_entity_keys_unique",
        "links_reference_surviving_keys",
        "memberships_reference_surviving_keys",
        "links_valid",
        "memberships_valid",
        "weights_valid",
    }
    for axis, contract in contracts.items():
        if contract.operation in _PRESERVE_OPERATIONS:
            if axis_changes[axis]:
                raise StructuralDiffError(
                    f"diff changed preserved mutation axis {axis!r}"
                )
        elif not axis_changes[axis] and contract.operation != "realign_row_weights":
            raise StructuralDiffError(
                f"mutation operation {contract.operation!r} produced no {axis} delta"
            )

        postcondition = contract.postcondition
        if postcondition in invariant_postconditions:
            pass
        elif postcondition == "link_tables_absent":
            if after._links:
                raise StructuralDiffError("link_tables_absent postcondition is false")
        elif postcondition == "entity_cardinality_filtered":
            before_sizes = {
                entity: len(table) for entity, table in before._tables.items()
            }
            after_sizes = {
                entity: len(table) for entity, table in after._tables.items()
            }
            if any(
                after_sizes[entity] > size for entity, size in before_sizes.items()
            ) or not any(
                after_sizes[entity] < size for entity, size in before_sizes.items()
            ):
                raise StructuralDiffError(
                    "entity_cardinality_filtered postcondition is false"
                )
        elif postcondition == "clone_roles_materialized":
            _assert_clone_roles_materialized(diff, after, added_row_sources)
        elif postcondition in {
            "clone_links_reference_remapped_keys",
            "clone_memberships_reference_remapped_keys",
        }:
            _assert_clone_memberships_remapped(
                before,
                after,
                added_row_sources,
            )
        elif postcondition == "clone_blocks_follow_native_rows":
            _assert_clone_blocks_follow_native_rows(diff)
        elif postcondition == "household_mass_conserved":
            _assert_descendant_mass_conserved(
                before,
                after,
                added_row_sources,
            )
        elif postcondition == "surviving_entity_order_preserved":
            if any(
                not _is_subsequence(table.after_row_ids, table.before_row_ids)
                for table in diff.tables
            ):
                raise StructuralDiffError(
                    "surviving_entity_order_preserved postcondition is false"
                )
        elif postcondition == "weights_aligned_to_surviving_keys":
            _assert_weights_aligned_to_surviving_keys(before, after, diff)
        elif postcondition == "entity_order_permuted":
            if not any(
                set(table.before_row_ids) == set(table.after_row_ids)
                and table.before_row_ids != table.after_row_ids
                for table in diff.tables
            ):
                raise StructuralDiffError(
                    "entity_order_permuted postcondition is false"
                )
        elif postcondition == "mass_history_extended":
            if len(after._mass_history) <= len(before._mass_history) or not (
                _exact_object_equal(
                    tuple(after._mass_history[: len(before._mass_history)]),
                    tuple(before._mass_history),
                )
            ):
                raise StructuralDiffError(
                    "mass_history_extended postcondition is false"
                )
        elif postcondition == "weights_preserve_key_mapping":
            changed = [
                weight.entity
                for weight in diff.weights
                if weight.values_changed or weight.kind_changed
            ]
            if changed:
                raise StructuralDiffError(
                    "weights_preserve_key_mapping postcondition is false for "
                    f"{sorted(changed)!r}"
                )
        else:  # pragma: no cover - closed by _VALID_MUTATION_CONTRACTS
            raise CapabilityError(
                f"mutation postcondition {postcondition!r} is not executable"
            )

    kind_rank = {"design": 0, "importance": 1, "calibrated": 2}
    for entity in set(before._weights) & set(after._weights):
        old = before._weights[entity].kind
        new = after._weights[entity].kind
        if kind_rank[new] < kind_rank[old]:
            raise StructuralDiffError(
                f"weight kind moved backward for {entity!r}: {old!r} -> {new!r}"
            )


def _validate_scopes(
    node: CompiledNode,
    before: ImmutableFrameProjection,
    after: ImmutableFrameProjection,
    diff: FrameDiff,
    registry: ClosedScopeRegistry,
) -> None:
    _validate_link_changes(before, after, diff)
    for table in diff.tables:
        for change in table.cell_changes:
            basis = before if change.before_present else after
            allowed = _write_scope_rows(
                node,
                entity=change.entity,
                column=change.column,
                modes=_TABLE_WRITE_MODES,
                basis=basis,
                registry=registry,
            )
            if change.row_id not in allowed:
                raise PatchScopeError(
                    f"write outside row scope: {change.entity}.{change.column} "
                    f"row {change.row_id!r}"
                )
        if table.order_changed:
            allowed = _write_scope_rows(
                node,
                entity=table.entity,
                column=None,
                modes=_TABLE_WRITE_MODES,
                basis=before,
                registry=registry,
            )
            outside = frozenset(table.before_row_ids) - allowed
            if outside:
                raise PatchScopeError(
                    f"row reorder outside scope: {table.entity} "
                    f"rows {_stable_sort(outside)!r}"
                )

    for weight in diff.weights:
        allowed: set[Hashable] = set()
        for basis in (before, after):
            if weight.entity not in basis._tables:
                continue
            allowed.update(
                _write_scope_rows(
                    node,
                    entity=weight.entity,
                    column="@resolved_weight",
                    modes=frozenset({"resolved_weight"}),
                    basis=basis,
                    registry=registry,
                )
            )
        outside = weight.changed_row_ids - allowed
        if outside:
            raise PatchScopeError(
                f"weight write outside row scope: {weight.entity} "
                f"rows {_stable_sort(outside)!r}"
            )

    for change in diff.virtual_changes:
        matched = False
        for value in node.write_scopes:
            scope = _mapping(_wire(value), location="write_scope")
            if (
                scope.get("entity") == change.entity
                and scope.get("column") == change.column
                and scope.get("mode") == "virtual_receipt"
            ):
                matched = True
                break
        if not matched:
            raise PatchScopeError(
                f"undeclared virtual receipt write {change.entity}.{change.column}"
            )


def execute_node(
    node: CompiledNode,
    projection: ImmutableFrameProjection,
    *,
    kernels: Mapping[str, RegisteredKernel],
    row_classifiers: Mapping[str, RegisteredRowClassifier] | None = None,
    context: ExecutionContext | None = None,
) -> ValidatedPatch:
    """Dispatch one compiled node and return a transactionally validated patch."""

    if not isinstance(node, CompiledNode):
        raise TypeError("execute_node requires a CompiledNode")
    if not isinstance(projection, ImmutableFrameProjection):
        raise TypeError("execute_node requires an ImmutableFrameProjection")
    requested_context = context or ExecutionContext()
    if not isinstance(requested_context.run_provenance_identity, RunProvenanceIdentity):
        raise CapabilityError(
            "execution context requires a closed run_provenance_identity"
        )
    run_provenance_wire = requested_context.run_provenance_identity.to_wire()
    capabilities = NodeCapabilities.from_mapping(_wire(node.capabilities))
    contracts = _mutation_contracts(node)
    _validate_before_dispatch(node, capabilities, contracts, requested_context)
    if requested_context.brokers is None:
        try:
            broker_session = deny_all_session_for_node(
                node,
                run_provenance_identity=run_provenance_wire,
                attempt=requested_context.attempt,
                attempt_scope=requested_context.attempt_scope,
                require_byte_equivalence=requested_context.require_byte_equivalence,
            )
        except BrokerContractError as error:
            raise CapabilityError(
                f"compiled node {node.id!r} requires an explicit bound broker session"
            ) from error
        selected_context = replace(
            requested_context, brokers=broker_session.kernel_view
        )
    elif isinstance(requested_context.brokers, BrokerSession):
        broker_session = requested_context.brokers
        selected_context = replace(
            requested_context, brokers=broker_session.kernel_view
        )
    else:
        raise CapabilityError("execution context brokers must be a BrokerSession")
    try:
        broker_session.validate_executor_binding(
            node=node,
            determinism=capabilities.determinism.value,
            effects=tuple(effect.value for effect in capabilities.effects),
            attempt=selected_context.attempt,
            attempt_scope=selected_context.attempt_scope,
            require_byte_equivalence=selected_context.require_byte_equivalence,
            run_provenance_identity=run_provenance_wire,
        )
    except BrokerContractError as error:
        raise CapabilityError(str(error)) from error
    scope_registry = _node_scope_registry(node)
    _validate_mutation_preconditions(contracts, projection)
    _validate_projection_contract(node, projection, scope_registry)
    try:
        registered = kernels[node.kernel_ref]
    except KeyError as error:
        raise CapabilityError(
            f"kernel {node.kernel_ref!r} is not registered"
        ) from error
    if registered.implementation_sha256 != node.kernel_implementation_sha256:
        raise CapabilityError(
            f"kernel {node.kernel_ref!r} implementation digest differs from compiled node"
        )
    registered_classifier: RegisteredRowClassifier | None = None
    if capabilities.structural_delta is StructuralDelta.EXPAND:
        try:
            registered_classifier = (row_classifiers or {})[node.row_classifier_ref]
        except KeyError as error:
            raise CapabilityError(
                f"row classifier {node.row_classifier_ref!r} is not registered"
            ) from error
        if (
            registered_classifier.implementation_sha256
            != node.row_classifier_implementation_sha256
        ):
            raise CapabilityError(
                f"row classifier {node.row_classifier_ref!r} implementation "
                "digest differs from compiled node"
            )
        if registered_classifier.predicate_space != scope_registry.predicate_space:
            raise CapabilityError(
                f"row classifier {node.row_classifier_ref!r} predicate space "
                "differs from compiled node"
            )

    try:
        broker_session.validate_callable(registered.function, role="kernel")
        if registered_classifier is not None:
            broker_session.validate_callable(
                registered_classifier.function, role="row_classifier"
            )
    except Exception:
        if not broker_session.sealed:
            broker_session.seal(status="aborted")
        raise

    try:
        immutable_before = projection.detached_copy()
        kernel_view = projection.detached_copy()
        with broker_session.activate():
            patch = registered.function(kernel_view, selected_context)
            if not isinstance(patch, KernelPatch):
                raise ExecutorError("kernel must return a KernelPatch")
            projection_mutation = diff_projections(projection, kernel_view)
            if not projection_mutation.empty:
                raise ExecutorError("kernel mutated its immutable input projection")
            if patch.structural_delta is not capabilities.structural_delta:
                raise StructuralDiffError(
                    f"patch delta {patch.structural_delta.value!r} differs from "
                    f"capability {capabilities.structural_delta.value!r}"
                )
            # The trusted row classifier executes inside _apply_unchecked.  It
            # remains under the same guard because ambient classification could
            # otherwise expand the kernel's write authority.
            result, added_row_sources = _apply_unchecked(
                immutable_before,
                patch,
                registry=scope_registry,
                row_classifier=registered_classifier,
                broker_session=broker_session,
            )
        diff = diff_projections(immutable_before, result)
        _validate_delta_semantics(
            capabilities.structural_delta, immutable_before, result, diff
        )
        _validate_outputs(node, result, diff)
        _validate_scopes(node, immutable_before, result, diff, scope_registry)
        _validate_mutation_contracts(
            contracts,
            immutable_before,
            result,
            diff,
            added_row_sources=added_row_sources,
        )
        try:
            broker_session.validate_executor_binding(
                node=node,
                determinism=capabilities.determinism.value,
                effects=tuple(effect.value for effect in capabilities.effects),
                attempt=selected_context.attempt,
                attempt_scope=selected_context.attempt_scope,
                require_byte_equivalence=selected_context.require_byte_equivalence,
                run_provenance_identity=run_provenance_wire,
            )
        except BrokerContractError as error:
            raise CapabilityError(
                "broker authority changed during kernel dispatch"
            ) from error
        base_sha256 = _projection_sha256(immutable_before)
        result_sha256 = _projection_sha256(result)
        patch_sha256 = _patch_sha256(patch)
        diff_sha256 = _pickle_sha256(diff)
        broker_receipt = broker_session.seal()
    except Exception:
        if not broker_session.sealed and not broker_session.active:
            broker_session.seal(status="aborted")
        raise
    return ValidatedPatch(
        node_id=node.id,
        node_key=node.node_key,
        attempt=selected_context.attempt,
        attempt_scope=selected_context.attempt_scope,
        patch=patch,
        diff=diff,
        broker_receipt=broker_receipt,
        _base=immutable_before,
        _result=result,
        _base_sha256=base_sha256,
        _result_sha256=result_sha256,
        _patch_sha256=patch_sha256,
        _diff_sha256=diff_sha256,
        _envelope_sha256=_validated_envelope_sha256(
            node_id=node.id,
            node_key=node.node_key,
            attempt=selected_context.attempt,
            attempt_scope=selected_context.attempt_scope,
            base_sha256=base_sha256,
            result_sha256=result_sha256,
            patch_sha256=patch_sha256,
            diff_sha256=diff_sha256,
            broker_receipt_sha256=broker_receipt.receipt_sha256,
        ),
    )


def apply_patch(
    projection: ImmutableFrameProjection,
    patch: ValidatedPatch,
) -> ImmutableFrameProjection:
    """Apply a validated patch only to the exact base it was checked against."""

    if not isinstance(patch.broker_receipt, BrokerReceipt):
        raise ExecutorError("validated patch broker receipt has an invalid type")
    try:
        patch.broker_receipt.validate()
    except BrokerContractError as error:
        raise ExecutorError("validated broker receipt was mutated") from error
    receipt = patch.broker_receipt
    if receipt.status != "complete":
        raise ExecutorError("validated patch requires a complete broker receipt")
    if receipt.owner.kind != "producer_node" or receipt.owner.id != patch.node_id:
        raise ExecutorError("validated broker receipt owner differs from its patch")
    if receipt.node_key != patch.node_key:
        raise ExecutorError("validated broker receipt node key differs from its patch")
    if receipt.attempt != patch.attempt:
        raise ExecutorError("validated broker receipt attempt differs from its patch")
    if receipt.attempt_scope != patch.attempt_scope:
        raise ExecutorError(
            "validated broker receipt attempt scope differs from its patch"
        )
    if (
        _validated_envelope_sha256(
            node_id=patch.node_id,
            node_key=patch.node_key,
            attempt=patch.attempt,
            attempt_scope=patch.attempt_scope,
            base_sha256=patch._base_sha256,
            result_sha256=patch._result_sha256,
            patch_sha256=patch._patch_sha256,
            diff_sha256=patch._diff_sha256,
            broker_receipt_sha256=receipt.receipt_sha256,
        )
        != patch._envelope_sha256
    ):
        raise ExecutorError("validated patch envelope seal was mutated")
    if _projection_sha256(patch._base) != patch._base_sha256:
        raise ExecutorError("validated patch base seal was mutated")
    if _projection_sha256(patch._result) != patch._result_sha256:
        raise ExecutorError("validated patch result seal was mutated")
    if _patch_sha256(patch.patch) != patch._patch_sha256:
        raise ExecutorError("validated kernel patch receipt was mutated")
    if _pickle_sha256(patch.diff) != patch._diff_sha256:
        raise ExecutorError("validated structural diff receipt was mutated")
    if _projection_sha256(projection) != patch._base_sha256:
        raise ExecutorError("validated patch base projection differs")
    return patch._result.detached_copy()


def _node_scope_atoms(
    node: CompiledNode,
    registry: ClosedScopeRegistry,
) -> dict[tuple[str, str], frozenset[str]]:
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for value in node.write_scopes:
        scope = _mapping(_wire(value), location=f"{node.id}/write_scope")
        key = (
            _string(scope.get("entity"), location="write_scope/entity"),
            _string(scope.get("column"), location="write_scope/column"),
        )
        result[key].update(_scope_atoms(scope, registry))
    return {key: frozenset(atoms) for key, atoms in result.items()}


def order_nodes(
    nodes: Iterable[CompiledNode],
) -> tuple[CompiledNode, ...]:
    """Return the compiled-rank Kahn order and reject unsafe incomparability."""

    rows = tuple(nodes)
    if not rows:
        return ()
    for node in rows:
        _compiled_contract_consistent(node)
    by_id = {node.id: node for node in rows}
    if len(by_id) != len(rows):
        raise NodeOrderingError("compiled node ids must be unique")
    ranks = [node.execution_rank for node in rows]
    if any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
        for rank in ranks
    ):
        raise NodeOrderingError(
            "compiled execution ranks must be non-negative integers"
        )
    if sorted(ranks) != list(range(len(rows))):
        raise NodeOrderingError(
            "compiled execution ranks must be exactly contiguous 0..n-1"
        )

    registries = {node.id: _node_scope_registry(node) for node in rows}
    registry_identities = {registry.identity_sha256 for registry in registries.values()}
    if len(registry_identities) != 1:
        raise NodeOrderingError("compiled nodes disagree on scope-registry authority")

    def order_key(node_id: str) -> tuple[int, str]:
        node = by_id[node_id]
        return node.execution_rank, node.id

    dependencies: dict[str, set[str]] = {}
    successors: dict[str, set[str]] = {node_id: set() for node_id in by_id}
    for node in rows:
        deps = set(node.depends_on)
        unknown = deps - set(by_id)
        if unknown:
            raise NodeOrderingError(
                f"node {node.id!r} has unknown dependencies {sorted(unknown)!r}"
            )
        dependencies[node.id] = deps
        for dependency in deps:
            if by_id[dependency].execution_rank >= node.execution_rank:
                raise NodeOrderingError(
                    f"dependency {dependency!r} must have a lower execution rank "
                    f"than {node.id!r}"
                )
            successors[dependency].add(node.id)
    local_slice_sha256 = {
        node.id: sha256_json([param.to_wire() for param in node.resolved_params])
        for node in rows
    }
    ancestor_ids: dict[str, set[str]] = {}
    ranked_nodes = sorted(rows, key=lambda node: (node.execution_rank, node.id))
    for node in ranked_nodes:
        ancestors = set(dependencies[node.id])
        for dependency in dependencies[node.id]:
            ancestors.update(ancestor_ids[dependency])
        ancestor_ids[node.id] = ancestors
        expected = tuple(
            (candidate.id, local_slice_sha256[candidate.id])
            for candidate in ranked_nodes
            if candidate.id in ancestors
        )
        observed = tuple(
            (candidate.id, candidate.local_slice_sha256)
            for candidate in node.transitive_nodes
        )
        if observed != expected:
            raise NodeOrderingError(
                f"node {node.id!r} transitive node slice differs from its "
                "dependency closure"
            )
    ready = sorted(
        (node_id for node_id, deps in dependencies.items() if not deps),
        key=order_key,
    )
    ordered_ids: list[str] = []
    while ready:
        node_id = ready.pop(0)
        ordered_ids.append(node_id)
        for successor in sorted(successors[node_id], key=order_key):
            dependencies[successor].remove(node_id)
            if not dependencies[successor]:
                ready.append(successor)
        ready.sort(key=order_key)
    if len(ordered_ids) != len(rows):
        cycle_nodes = sorted(set(by_id) - set(ordered_ids))
        raise NodeOrderingError(f"dependency cycle among {cycle_nodes!r}")

    reachable = {node_id: set(successors[node_id]) for node_id in by_id}
    changed = True
    while changed:
        changed = False
        for node_id in sorted(reachable):
            expanded = set(reachable[node_id])
            for child in tuple(reachable[node_id]):
                expanded.update(reachable[child])
            if expanded != reachable[node_id]:
                reachable[node_id] = expanded
                changed = True
    scope_cache = {
        node_id: _node_scope_atoms(node, registries[node_id])
        for node_id, node in by_id.items()
    }
    mutation_cache = {
        node_id: {
            axis
            for axis, contract in _mutation_contracts(node).items()
            if contract.operation not in _PRESERVE_OPERATIONS
        }
        for node_id, node in by_id.items()
    }
    topology_axes = frozenset({"entity_keys", "cardinality", "order"})

    def entity_scope_atoms(node_id: str) -> dict[str, frozenset[str]]:
        result: dict[str, set[str]] = defaultdict(set)
        for (entity, _column), atoms in scope_cache[node_id].items():
            result[entity].update(atoms)
        return {entity: frozenset(atoms) for entity, atoms in result.items()}

    entity_scope_cache = {node_id: entity_scope_atoms(node_id) for node_id in by_id}
    topology_cache = {
        node_id: (
            entity_scope_cache[node_id]
            if mutation_cache[node_id] & topology_axes
            else {}
        )
        for node_id in by_id
    }
    for index, left_id in enumerate(ordered_ids):
        for right_id in ordered_ids[index + 1 :]:
            if right_id in reachable[left_id] or left_id in reachable[right_id]:
                continue
            common = set(scope_cache[left_id]) & set(scope_cache[right_id])
            overlaps = [
                key
                for key in sorted(common)
                if not scope_cache[left_id][key].isdisjoint(scope_cache[right_id][key])
            ]
            if overlaps:
                raise NodeOrderingError(
                    "incomparable nodes have overlapping exact writes: "
                    f"{left_id!r}, {right_id!r}, {overlaps!r}"
                )
            topology_conflicts: list[tuple[str, tuple[str, ...]]] = []
            for entity in sorted(
                set(topology_cache[left_id]) & set(entity_scope_cache[right_id])
                | set(topology_cache[right_id]) & set(entity_scope_cache[left_id])
            ):
                atoms = (
                    topology_cache[left_id].get(entity, frozenset())
                    & entity_scope_cache[right_id].get(entity, frozenset())
                ) | (
                    topology_cache[right_id].get(entity, frozenset())
                    & entity_scope_cache[left_id].get(entity, frozenset())
                )
                if atoms:
                    topology_conflicts.append((entity, tuple(sorted(atoms))))
            if topology_conflicts:
                raise NodeOrderingError(
                    "incomparable structural and cell writes overlap: "
                    f"{left_id!r}, {right_id!r}, {topology_conflicts!r}"
                )
            structural_overlaps = mutation_cache[left_id] & mutation_cache[right_id]
            if structural_overlaps:
                raise NodeOrderingError(
                    "incomparable nodes have overlapping structural resources: "
                    f"{left_id!r}, {right_id!r}, "
                    f"{sorted(structural_overlaps)!r}"
                )
    return tuple(by_id[node_id] for node_id in ordered_ids)


__all__ = [
    "CapabilityError",
    "CellChange",
    "Determinism",
    "Effect",
    "ExecutionContext",
    "ExecutorError",
    "FrameDiff",
    "ImmutableFrameProjection",
    "KernelPatch",
    "LinkCellChange",
    "LinkDiff",
    "MutationContract",
    "NodeCapabilities",
    "NodeOrderingError",
    "NodeReuseIdentity",
    "NumericReproducibility",
    "PatchScopeError",
    "RegisteredKernel",
    "RegisteredRowClassifier",
    "RunProvenanceIdentity",
    "RowClassification",
    "RetrySafety",
    "StructuralDelta",
    "StructuralDiffError",
    "TableDiff",
    "ValidatedPatch",
    "VirtualChange",
    "WeightDiff",
    "WeightState",
    "apply_patch",
    "build_run_provenance_identity",
    "diff_projections",
    "execute_node",
    "node_reuse_identity",
    "order_nodes",
]
