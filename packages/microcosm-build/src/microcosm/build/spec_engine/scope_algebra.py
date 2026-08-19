"""Closed, finite row-scope predicates for compiler and executor checks.

The persisted bundle grammar permits named scopes and Boolean expressions over
compiler-provided primitives.  This module gives that grammar one executable
meaning: every valid expression is reduced to a canonical subset of a finite
atom universe.  Equality, overlap, and exhaustiveness are therefore exact set
operations rather than runtime guesses.

Nothing here knows a country, program, channel, support role, entity, or
column.  The compiler supplies the primitive-to-atom mappings for a predicate
space.  Runtime authorization accepts stable row ids whose non-empty atom set
is wholly contained by the declared scope; this conservative rule also works
for a row id representing a compound/grouped record.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from .canonical import canonical_json_bytes, sha256_json

_IDENTIFIER: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_BOOLEAN_FORMS: Final = frozenset({"all_of", "any_of", "not"})
_PRIMITIVE_FORMS: Final = frozenset(
    {"channel_is", "support_role_is", "column_predicate"}
)
_SCOPE_FORMS: Final = _BOOLEAN_FORMS | _PRIMITIVE_FORMS | {"atoms"}
_COLUMN_OPERATORS: Final = frozenset({"eq", "ne", "is_null", "not_null"})
_ABSENT: Final = object()


class ScopeAlgebraError(ValueError):
    """A row-scope predicate is outside its declared finite algebra."""


@dataclass(frozen=True, slots=True)
class CanonicalScope:
    """The stable semantic normal form of one row-scope expression."""

    predicate_space: str
    atom_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.predicate_space, location="predicate_space")
        if self.atom_ids != tuple(sorted(self.atom_ids)):
            raise ScopeAlgebraError("canonical scope atoms must be sorted")
        if len(self.atom_ids) != len(set(self.atom_ids)):
            raise ScopeAlgebraError("canonical scope atoms must be unique")
        for index, atom in enumerate(self.atom_ids):
            _require_atom(atom, location=f"atom_ids/{index}")


@dataclass(frozen=True, slots=True)
class _ColumnPredicateKey:
    entity: str
    column: str
    op: str
    value_bytes: bytes | None


@dataclass(frozen=True, slots=True, init=False)
class ClosedScopeRegistry:
    """One immutable Boolean algebra over a compiler-declared atom universe.

    ``scopes`` maps names to either atom collections or row-scope expressions.
    Named expressions may reference other named scopes, including forward
    references, but cycles are refused.

    ``primitives`` is a mapping with any of these compiler-owned spaces::

        {
            "channel_is": {"survey": {"atom_a"}},
            "support_role_is": {"donor": {"atom_b"}},
            "column_predicate": {
                ("person", "status", "eq", "eligible"): {"atom_a"},
                ("person", "income", "not_null"): {"atom_b"},
            },
        }

    For a column predicate with a non-hashable JSON value, the
    ``column_predicate`` value may instead be a sequence of
    ``(predicate_mapping, atoms)`` pairs.  Inputs are normalized and copied;
    the registry retains only immutable canonical keys and atom sets.
    """

    predicate_space: str
    universe: tuple[str, ...]
    _universe_atoms: frozenset[str]
    _scopes: Mapping[str, CanonicalScope]
    _primitives: Mapping[str, Mapping[object, frozenset[str]]]

    def __init__(
        self,
        predicate_space: str,
        universe: Iterable[str],
        scopes: Mapping[str, object],
        primitives: Mapping[str, object] | None = None,
    ) -> None:
        _require_identifier(predicate_space, location="predicate_space")
        universe_tuple = _atom_collection(universe, location="universe")
        if not universe_tuple:
            raise ScopeAlgebraError("universe: at least one finite atom required")
        universe_atoms = frozenset(universe_tuple)

        if not isinstance(scopes, Mapping):
            raise ScopeAlgebraError("scopes: mapping required")
        raw_scopes = dict(scopes)
        if not raw_scopes:
            raise ScopeAlgebraError("scopes: at least one named scope required")
        for name in raw_scopes:
            _require_identifier(name, location="scopes name")

        frozen_primitives = _normalize_primitives(
            {} if primitives is None else primitives,
            universe=universe_atoms,
        )

        resolved: dict[str, CanonicalScope] = {}
        resolving: list[str] = []

        def resolve_named(name: str) -> frozenset[str]:
            if name in resolved:
                return frozenset(resolved[name].atom_ids)
            if name not in raw_scopes:
                raise ScopeAlgebraError(f"unknown named scope {name!r}")
            if name in resolving:
                start = resolving.index(name)
                cycle = " -> ".join((*resolving[start:], name))
                raise ScopeAlgebraError(f"cyclic named scopes: {cycle}")

            resolving.append(name)
            definition = raw_scopes[name]
            try:
                if _is_atom_collection(definition):
                    atoms = _checked_atoms(
                        definition,
                        universe=universe_atoms,
                        location=f"scopes/{name}",
                    )
                else:
                    atoms = _evaluate_scope(
                        definition,
                        universe=universe_atoms,
                        primitives=frozen_primitives,
                        resolve_named=resolve_named,
                        location=f"scopes/{name}",
                        active_containers=set(),
                    )
            finally:
                resolving.pop()
            canonical = CanonicalScope(predicate_space, tuple(sorted(atoms)))
            resolved[name] = canonical
            return atoms

        for name in sorted(raw_scopes):
            resolve_named(name)

        object.__setattr__(self, "predicate_space", predicate_space)
        object.__setattr__(self, "universe", universe_tuple)
        object.__setattr__(self, "_universe_atoms", universe_atoms)
        object.__setattr__(
            self,
            "_scopes",
            MappingProxyType({name: resolved[name] for name in sorted(resolved)}),
        )
        object.__setattr__(self, "_primitives", frozen_primitives)

    @property
    def scopes(self) -> Mapping[str, CanonicalScope]:
        """The resolved named scopes, exposed through a read-only mapping."""

        return self._scopes

    @property
    def primitives(self) -> Mapping[str, Mapping[object, frozenset[str]]]:
        """The normalized primitive mappings, exposed read-only."""

        return self._primitives

    @classmethod
    def from_wire(cls, value: object) -> ClosedScopeRegistry:
        """Build the exact compiler-emitted finite registry.

        Persisted registries carry their named scopes in canonical atom form.
        Runtime callers therefore cannot substitute a different predicate
        interpretation under the same names.
        """

        if not isinstance(value, Mapping):
            raise ScopeAlgebraError("scope registry wire value must be an object")
        if set(value) != {"predicate_space", "universe", "scopes"}:
            raise ScopeAlgebraError(
                "scope registry wire value must contain exactly "
                "predicate_space/universe/scopes"
            )
        raw_scopes = value["scopes"]
        if not isinstance(raw_scopes, Sequence) or isinstance(raw_scopes, str | bytes):
            raise ScopeAlgebraError("scope registry scopes must be an array")
        scopes: dict[str, object] = {}
        for index, raw_scope in enumerate(raw_scopes):
            if not isinstance(raw_scope, Mapping) or set(raw_scope) != {
                "id",
                "atoms",
            }:
                raise ScopeAlgebraError(
                    f"scope registry scopes/{index} must contain exactly id/atoms"
                )
            name = _require_identifier(
                raw_scope["id"], location=f"scope registry scopes/{index}/id"
            )
            if name in scopes:
                raise ScopeAlgebraError(
                    f"scope registry scopes repeat named scope {name!r}"
                )
            scopes[name] = raw_scope["atoms"]
        return cls(
            predicate_space=value["predicate_space"],  # type: ignore[arg-type]
            universe=value["universe"],  # type: ignore[arg-type]
            scopes=scopes,
        )

    def to_wire(self) -> dict[str, object]:
        """Return the canonical compiler/runtime authority representation."""

        return {
            "predicate_space": self.predicate_space,
            "universe": list(self.universe),
            "scopes": [
                {"id": name, "atoms": list(scope.atom_ids)}
                for name, scope in self._scopes.items()
            ],
        }

    @property
    def identity_sha256(self) -> str:
        """Content identity used to seal a node to this finite algebra."""

        return sha256_json(self.to_wire())

    def canonical(self, scope: object) -> CanonicalScope:
        """Return the unique atom-set normal form for ``scope``."""

        if isinstance(scope, CanonicalScope):
            self._validate_canonical(scope)
            return scope
        atoms = _evaluate_scope(
            scope,
            universe=self._universe_atoms,
            primitives=self._primitives,
            resolve_named=self._resolve_named,
            location="scope",
            active_containers=set(),
        )
        return CanonicalScope(self.predicate_space, tuple(sorted(atoms)))

    def atoms(self, scope: object) -> frozenset[str]:
        """Return the immutable finite atom set denoted by ``scope``."""

        return frozenset(self.canonical(scope).atom_ids)

    def equal(self, left: object, right: object) -> bool:
        """Return whether two predicates denote exactly the same rows."""

        return self.atoms(left) == self.atoms(right)

    def overlaps(self, left: object, right: object) -> bool:
        """Return whether two predicates share at least one finite atom."""

        return not self.atoms(left).isdisjoint(self.atoms(right))

    def is_exhaustive(self, scopes: object) -> bool:
        """Return whether one predicate, or an iterable of them, covers all atoms."""

        if isinstance(scopes, (str, Mapping, CanonicalScope)):
            candidates = (scopes,)
        else:
            if not isinstance(scopes, Iterable):
                raise ScopeAlgebraError(
                    "exhaustiveness input must be a scope or iterable"
                )
            candidates = tuple(scopes)
        covered: set[str] = set()
        for scope in candidates:
            covered.update(self.atoms(scope))
        return covered == self._universe_atoms

    def authorized_row_ids(
        self,
        scope: object,
        row_atoms: Mapping[Hashable, AbstractSet[str]],
    ) -> frozenset[Hashable]:
        """Return stable row ids wholly covered by ``scope``.

        Each runtime row id must carry at least one known atom.  A compound row
        is authorized only if every atom it represents is in the scope, which
        prevents a partially covered structural write from escaping its row
        contract.
        """

        if not isinstance(row_atoms, Mapping):
            raise ScopeAlgebraError("row_atoms: mapping required")
        allowed = self.atoms(scope)
        authorized: set[Hashable] = set()
        for row_id, raw_atoms in row_atoms.items():
            try:
                hash(row_id)
            except TypeError as error:
                raise ScopeAlgebraError(
                    f"row_atoms: unhashable stable row id {row_id!r}"
                ) from error
            if not isinstance(raw_atoms, AbstractSet) or isinstance(
                raw_atoms, str | bytes
            ):
                raise ScopeAlgebraError(
                    f"row_atoms/{row_id!r}: non-empty atom set required"
                )
            atoms = _checked_atoms(
                raw_atoms,
                universe=self._universe_atoms,
                location=f"row_atoms/{row_id!r}",
            )
            if not atoms:
                raise ScopeAlgebraError(
                    f"row_atoms/{row_id!r}: non-empty atom set required"
                )
            if atoms <= allowed:
                authorized.add(row_id)
        return frozenset(authorized)

    def _resolve_named(self, name: str) -> frozenset[str]:
        try:
            canonical = self._scopes[name]
        except KeyError as error:
            raise ScopeAlgebraError(f"unknown named scope {name!r}") from error
        return frozenset(canonical.atom_ids)

    def _validate_canonical(self, scope: CanonicalScope) -> None:
        if scope.predicate_space != self.predicate_space:
            raise ScopeAlgebraError(
                "unknown predicate space "
                f"{scope.predicate_space!r}; expected {self.predicate_space!r}"
            )
        unknown = frozenset(scope.atom_ids) - self._universe_atoms
        if unknown:
            raise ScopeAlgebraError(
                f"canonical scope: atoms outside universe {sorted(unknown)!r}"
            )


def _require_identifier(value: object, *, location: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ScopeAlgebraError(
            f"{location}: identifier matching {_IDENTIFIER.pattern!r} required"
        )
    return value


def _require_atom(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScopeAlgebraError(f"{location}: non-empty string atom required")
    return value


def _is_atom_collection(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping)


def _atom_collection(value: object, *, location: str) -> tuple[str, ...]:
    if not _is_atom_collection(value):
        raise ScopeAlgebraError(f"{location}: finite atom collection required")
    atoms = tuple(
        _require_atom(atom, location=f"{location}/{index}")
        for index, atom in enumerate(value)
    )
    if len(atoms) != len(set(atoms)):
        raise ScopeAlgebraError(f"{location}: duplicate finite atom")
    return tuple(sorted(atoms))


def _checked_atoms(
    value: object,
    *,
    universe: frozenset[str],
    location: str,
) -> frozenset[str]:
    atoms = frozenset(_atom_collection(value, location=location))
    unknown = atoms - universe
    if unknown:
        raise ScopeAlgebraError(
            f"{location}: atoms outside universe {sorted(unknown)!r}"
        )
    return atoms


def _normalize_primitives(
    value: object,
    *,
    universe: frozenset[str],
) -> Mapping[str, Mapping[object, frozenset[str]]]:
    if not isinstance(value, Mapping):
        raise ScopeAlgebraError("primitives: mapping required")
    unknown_spaces = set(value) - _PRIMITIVE_FORMS
    if unknown_spaces:
        raise ScopeAlgebraError(
            f"primitives: unknown predicate spaces {_stable_values(unknown_spaces)!r}"
        )

    normalized: dict[str, Mapping[object, frozenset[str]]] = {}
    for primitive_name in sorted(_PRIMITIVE_FORMS):
        raw_space = value.get(primitive_name, {})
        rows: dict[object, frozenset[str]] = {}
        if primitive_name == "column_predicate":
            pairs = _column_primitive_pairs(raw_space)
            for index, (raw_key, raw_atoms) in enumerate(pairs):
                key = _column_key(
                    raw_key, location=f"primitives/{primitive_name}/{index}"
                )
                if key in rows:
                    raise ScopeAlgebraError(
                        f"primitives/{primitive_name}: duplicate canonical predicate"
                    )
                rows[key] = _checked_atoms(
                    raw_atoms,
                    universe=universe,
                    location=f"primitives/{primitive_name}/{index}/atoms",
                )
        else:
            if not isinstance(raw_space, Mapping):
                raise ScopeAlgebraError(
                    f"primitives/{primitive_name}: mapping required"
                )
            for raw_key, raw_atoms in raw_space.items():
                key = _require_identifier(
                    raw_key, location=f"primitives/{primitive_name} key"
                )
                rows[key] = _checked_atoms(
                    raw_atoms,
                    universe=universe,
                    location=f"primitives/{primitive_name}/{key}",
                )
        if primitive_name == "column_predicate":
            ordered_rows = {
                key: rows[key]
                for key in sorted(
                    rows,
                    key=lambda item: (
                        item.entity,
                        item.column,
                        item.op,
                        item.value_bytes or b"",
                    ),
                )
            }
        else:
            ordered_rows = {key: rows[key] for key in sorted(rows)}
        normalized[primitive_name] = MappingProxyType(ordered_rows)
    return MappingProxyType(normalized)


def _column_primitive_pairs(value: object) -> tuple[tuple[object, object], ...]:
    if isinstance(value, Mapping):
        return tuple(value.items())
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ScopeAlgebraError(
            "primitives/column_predicate: mapping or sequence of pairs required"
        )
    pairs: list[tuple[object, object]] = []
    for index, row in enumerate(value):
        if (
            not isinstance(row, Sequence)
            or isinstance(row, str | bytes)
            or len(row) != 2
        ):
            raise ScopeAlgebraError(
                f"primitives/column_predicate/{index}: predicate/atoms pair required"
            )
        pairs.append((row[0], row[1]))
    return tuple(pairs)


def _column_key(value: object, *, location: str) -> _ColumnPredicateKey:
    if isinstance(value, Mapping):
        raw = value
    elif isinstance(value, tuple) and len(value) in {3, 4}:
        raw = {
            "entity": value[0],
            "column": value[1],
            "op": value[2],
        }
        if len(value) == 4:
            raw["value"] = value[3]
    else:
        raise ScopeAlgebraError(
            f"{location}: column predicate mapping or 3/4-tuple required"
        )

    keys = set(raw)
    unknown = keys - {"entity", "column", "op", "value"}
    missing = {"entity", "column", "op"} - keys
    if unknown or missing:
        raise ScopeAlgebraError(
            f"{location}: malformed column predicate; "
            f"missing={sorted(missing)!r}, unknown={_stable_values(unknown)!r}"
        )
    entity = _require_identifier(raw["entity"], location=f"{location}/entity")
    column = _require_identifier(raw["column"], location=f"{location}/column")
    op = raw["op"]
    if not isinstance(op, str) or op not in _COLUMN_OPERATORS:
        raise ScopeAlgebraError(
            f"{location}/op: expected one of {sorted(_COLUMN_OPERATORS)!r}"
        )
    raw_value = raw.get("value", _ABSENT)
    if op in {"eq", "ne"} and raw_value is _ABSENT:
        raise ScopeAlgebraError(f"{location}/value: required for {op!r}")
    if op in {"is_null", "not_null"} and raw_value is not _ABSENT:
        raise ScopeAlgebraError(f"{location}/value: forbidden for {op!r}")
    if raw_value is _ABSENT:
        value_bytes = None
    else:
        try:
            value_bytes = canonical_json_bytes(raw_value)
        except (TypeError, ValueError) as error:
            raise ScopeAlgebraError(
                f"{location}/value: canonical JSON value required"
            ) from error
    return _ColumnPredicateKey(entity, column, op, value_bytes)


def _evaluate_scope(
    value: object,
    *,
    universe: frozenset[str],
    primitives: Mapping[str, Mapping[object, frozenset[str]]],
    resolve_named: Callable[[str], frozenset[str]],
    location: str,
    active_containers: set[int],
) -> frozenset[str]:
    if isinstance(value, str):
        return resolve_named(value)
    if not isinstance(value, Mapping):
        raise ScopeAlgebraError(
            f"{location}: named scope or single-form predicate mapping required"
        )

    marker = id(value)
    if marker in active_containers:
        raise ScopeAlgebraError(f"{location}: cyclic predicate container")
    active_containers.add(marker)
    try:
        keys = set(value)
        if len(keys) != 1 or not keys <= _SCOPE_FORMS:
            unknown = keys - _SCOPE_FORMS
            raise ScopeAlgebraError(
                f"{location}: exactly one predicate form required; "
                f"unknown={_stable_values(unknown)!r}"
            )
        form = next(iter(keys))
        operand = value[form]

        if form == "atoms":
            return _checked_atoms(
                operand, universe=universe, location=f"{location}/atoms"
            )
        if form in {"all_of", "any_of"}:
            if (
                not isinstance(operand, Sequence)
                or isinstance(operand, str | bytes)
                or not operand
            ):
                raise ScopeAlgebraError(
                    f"{location}/{form}: non-empty expression array required"
                )
            children = tuple(
                _evaluate_scope(
                    child,
                    universe=universe,
                    primitives=primitives,
                    resolve_named=resolve_named,
                    location=f"{location}/{form}/{index}",
                    active_containers=active_containers,
                )
                for index, child in enumerate(operand)
            )
            if form == "all_of":
                return frozenset.intersection(*children)
            return frozenset.union(*children)
        if form == "not":
            child = _evaluate_scope(
                operand,
                universe=universe,
                primitives=primitives,
                resolve_named=resolve_named,
                location=f"{location}/not",
                active_containers=active_containers,
            )
            return universe - child
        if form in {"channel_is", "support_role_is"}:
            key = _require_identifier(operand, location=f"{location}/{form}")
        else:
            key = _column_key(operand, location=f"{location}/column_predicate")
        try:
            return primitives[form][key]
        except KeyError as error:
            raise ScopeAlgebraError(
                f"{location}/{form}: unknown compiler-provided primitive {operand!r}"
            ) from error
    finally:
        active_containers.remove(marker)


def _stable_values(values: Iterable[object]) -> list[object]:
    """Order malformed heterogeneous keys without leaking a ``TypeError``."""

    return sorted(values, key=lambda value: (type(value).__name__, repr(value)))
