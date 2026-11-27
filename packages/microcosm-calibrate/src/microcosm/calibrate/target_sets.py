"""Declared target sets: which compiled targets a product calibrates to.

A release tool calibrates to a subset of the targets its country compiles.
Each country declares that subset as data, for each product and mode: the
families, the levels and the rule. This module reads the declarations and
selects by them (microcosm#1728). It holds no country code: a country passes
the named predicates and rules its declarations use as callables.

A declaration file has this form::

    {
      "version": 1,
      "country": "<country>",
      "level_keys": ["<metadata key>", ...],
      "products": {
        "<product>": {
          "tool": "<path of the tool>",
          "option": "<the option that picks the mode>",
          "default_mode": "<mode>",
          "modes": {
            "<mode>": {
              "description": "<what the product calibrates to>",
              "compiled": [<clause>, ...],
              "built": [<built rows>, ...]
            }
          }
        }
      }
    }

**Families and levels.** A target's family is ``spec.family``. Its level is
the first truthy value of the ``level_keys`` in its metadata, else
``"unspecified"``. These are the labels the shared target-loss rule reads
(``microcosm.calibrate.target_loss_rules``, microcosm#1729), so a target has
one family and one level everywhere.

A declaration can give a ``level_rule`` in place of ``level_keys``::

    "level_rule": {
      "keys": ["<metadata key>", ...],
      "values": ["<level>", ...] or "*",
      "conflict": "first" or "refuse",
      "rationale": "<why>"
    }

``values`` closes the set of levels: a target whose level is not one of them,
or that has no level, is refused. With ``"conflict": "refuse"``, a target
whose keys hold two different levels is refused; with ``"first"``, the first
key that has a value gives the level. ``level_keys`` is the short form of a
rule with ``"values": "*"`` and ``"conflict": "first"``.

**Compiled clauses.** A clause keeps the compiled specs that pass all of its
tests, in compile order:

- ``families``: family names, or ``"*"`` for any family;
- ``levels``: level names, or ``"*"`` for any level;
- ``metadata`` (optional): one test for each named key, one of
  ``{"present": true}`` (the key is in the metadata), ``{"equals": v}``,
  ``{"not_equals": v}`` (a missing key is not equal), or
  ``{"not_prefix": "text"}`` (the key holds a nonempty string that does not
  start with ``text``);
- ``exclude`` (optional): names of country predicates; a spec that any of
  them accepts is dropped;
- ``rule`` (optional): the name of a country rule. It takes the clause's
  specs and returns the specs it keeps (for example a reconciliation) and a
  receipt.

Each clause has an ``id``. A tool can turn on only some clauses by id (for
example from a ``--families`` option). The selection is the output of each
clause in declaration order. A spec that two clauses keep is an error in the
declaration.

**Built rows.** A tool also makes rows that are not compiled (for example
population marginals from a ladder). A ``built`` entry names their family and
the level that each value of its option gives, so that the declaration states
the whole target set. The selector does not make these rows.

**Selection by level.** A mode with a closed level set and one clause for
each level (any family, no tests, its own level) can also be selected by a
set of levels that the caller gives at run time. :func:`select_by_levels`
keeps the targets of those levels (all of them by default) in compile order.
It returns them as a registry, with a receipt that lists every target, kept
or not (schema ``microcosm.calibrate.target-selection.v1``). It refuses a
requested level that no target has, and an empty selection. The UK full
build selects its solve surface this way (microcosm#1759).

**Rationale.** Each clause and built entry has a ``rationale``: why the
product keeps those rows. The UK membership files give a signed rationale for
each exclusion in the same way.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .registry import TargetRegistry

__all__ = [
    "ANY",
    "DECLARATION_VERSION",
    "LEVEL_CONFLICTS",
    "SELECTION_RECEIPT_SCHEMA",
    "UNSPECIFIED_LEVEL",
    "BuiltRows",
    "LevelRule",
    "LevelSelection",
    "MetadataTest",
    "ProductDeclaration",
    "TargetSetClause",
    "TargetSetDeclaration",
    "TargetSetDeclarationError",
    "TargetSetDeclarations",
    "TargetSetSelection",
    "clause_predicate",
    "load_target_set_declarations",
    "select_by_levels",
    "select_target_set",
    "target_level",
    "target_set_census",
    "undeclared_cells",
]

ANY = "*"
DECLARATION_VERSION = 1
UNSPECIFIED_LEVEL = "unspecified"
#: How a level rule treats a target whose keys hold different levels.
LEVEL_CONFLICTS = ("first", "refuse")
#: The receipt schema of :func:`select_by_levels`.
SELECTION_RECEIPT_SCHEMA = "microcosm.calibrate.target-selection.v1"

_TOP_KEYS = frozenset(
    {"version", "country", "description", "level_keys", "level_rule", "products"}
)
_LEVEL_RULE_KEYS = frozenset({"keys", "values", "conflict", "rationale"})
_PRODUCT_KEYS = frozenset({"tool", "option", "default_mode", "modes"})
_MODE_KEYS = frozenset({"description", "compiled", "built"})
_CLAUSE_KEYS = frozenset(
    {"id", "families", "levels", "metadata", "exclude", "rule", "rationale"}
)
_BUILT_KEYS = frozenset({"id", "family", "levels", "option", "source", "rationale"})
_TEST_KINDS = frozenset({"present", "equals", "not_equals", "not_prefix"})

Predicate = Callable[[Any], bool]
Rule = Callable[[tuple[Any, ...]], tuple[Sequence[Any], Any]]


class TargetSetDeclarationError(ValueError):
    """A declaration that does not have the declared form."""


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def target_level(spec: Any, level_keys: Sequence[str]) -> str:
    """A spec's level: the first truthy ``level_keys`` value, else unspecified."""

    metadata = spec.metadata
    for key in level_keys:
        value = metadata.get(key)
        if value:
            return str(value)
    return UNSPECIFIED_LEVEL


@dataclass(frozen=True)
class LevelRule:
    """How a target's level is read from its metadata (see the module docstring).

    The level is the first truthy value of ``keys``. With ``conflict`` set to
    ``"refuse"``, two keys that hold different truthy values are refused.
    With ``values`` (a closed set), a level outside it is refused, and so is a
    target with no level; without it (``None``), a target with no level is
    ``"unspecified"``.
    """

    keys: tuple[str, ...]
    values: frozenset[str] | None = None
    conflict: str = "first"
    rationale: str = ""

    def level_of(self, spec: Any) -> str:
        if self.values is None and self.conflict == "first":
            return target_level(spec, self.keys)
        held = [spec.metadata.get(key) for key in self.keys]
        if self.conflict == "refuse":
            present = [value for value in held if value]
            if any(value != present[0] for value in present[1:]):
                raise ValueError(
                    f"Target {spec.name!r} has contradictory geography levels."
                )
        # The first truthy value, else the last key's value, as ``a or b`` does.
        level = next((value for value in held if value), held[-1])
        if self.values is None:
            return str(level) if level else UNSPECIFIED_LEVEL
        if level not in self.values:
            raise ValueError(
                f"Target {spec.name!r} has unsupported geography level {level!r}."
            )
        return str(level)


@dataclass(frozen=True)
class MetadataTest:
    """One test on one metadata key."""

    key: str
    kind: str
    value: Any = None

    def accepts(self, metadata: Mapping[str, Any]) -> bool:
        if self.kind == "present":
            return (self.key in metadata) is bool(self.value)
        if self.kind == "equals":
            return metadata.get(self.key) == self.value
        if self.kind == "not_equals":
            return metadata.get(self.key) != self.value
        held = metadata.get(self.key)
        return isinstance(held, str) and bool(held) and not held.startswith(self.value)


@dataclass(frozen=True)
class TargetSetClause:
    """Compiled specs of some families and levels that pass every test."""

    clause_id: str
    families: frozenset[str] | None
    levels: frozenset[str] | None
    metadata: tuple[MetadataTest, ...]
    exclude: tuple[str, ...]
    rule: str | None
    rationale: str

    def accepts(
        self, spec: Any, level: str, predicates: Mapping[str, Predicate]
    ) -> bool:
        if self.families is not None and spec.family not in self.families:
            return False
        return self.accepts_level_and_metadata(spec, level, predicates)

    def accepts_level_and_metadata(
        self, spec: Any, level: str, predicates: Mapping[str, Predicate]
    ) -> bool:
        """Every test but the family one, for a caller that selected the family."""

        if self.levels is not None and level not in self.levels:
            return False
        metadata = spec.metadata
        if not all(test.accepts(metadata) for test in self.metadata):
            return False
        return not any(predicates[name](spec) for name in self.exclude)


@dataclass(frozen=True)
class BuiltRows:
    """Rows a tool makes outside the compile, and the level each option gives."""

    built_id: str
    family: str
    levels: Mapping[str, str]
    option: str | None
    source: str
    rationale: str


@dataclass(frozen=True)
class TargetSetDeclaration:
    """What one product calibrates to in one mode."""

    country: str
    product: str
    mode: str
    description: str
    level_keys: tuple[str, ...]
    compiled: tuple[TargetSetClause, ...]
    built: tuple[BuiltRows, ...]
    #: The country's level rule; ``None`` reads ``level_keys`` alone.
    level_rule: LevelRule | None = None

    @property
    def clause_ids(self) -> tuple[str, ...]:
        return tuple(clause.clause_id for clause in self.compiled)

    def clause(self, clause_id: str) -> TargetSetClause:
        for clause in self.compiled:
            if clause.clause_id == clause_id:
                return clause
        raise KeyError(
            f"{self.product}/{self.mode} declares no compiled clause {clause_id!r}; "
            f"it declares {list(self.clause_ids)}."
        )

    def built_rows(self, built_id: str) -> BuiltRows:
        for built in self.built:
            if built.built_id == built_id:
                return built
        raise KeyError(
            f"{self.product}/{self.mode} declares no built rows {built_id!r}."
        )

    def level_of(self, spec: Any) -> str:
        if self.level_rule is not None:
            return self.level_rule.level_of(spec)
        return target_level(spec, self.level_keys)


@dataclass(frozen=True)
class ProductDeclaration:
    """One product: the option that picks its mode, and each mode's target set."""

    product: str
    tool: str
    option: str
    default_mode: str
    modes: Mapping[str, TargetSetDeclaration]

    @property
    def mode_names(self) -> tuple[str, ...]:
        return tuple(self.modes)

    def mode(self, name: str) -> TargetSetDeclaration:
        try:
            return self.modes[name]
        except KeyError:
            raise KeyError(
                f"{self.product} declares no mode {name!r}; it declares "
                f"{list(self.modes)}."
            ) from None


@dataclass(frozen=True)
class TargetSetDeclarations:
    """Every product's declared target sets for one country."""

    country: str
    level_keys: tuple[str, ...]
    products: Mapping[str, ProductDeclaration]
    level_rule: LevelRule | None = None

    def product(self, name: str) -> ProductDeclaration:
        try:
            return self.products[name]
        except KeyError:
            raise KeyError(
                f"{self.country} declares no product {name!r}; it declares "
                f"{list(self.products)}."
            ) from None

    def declaration(self, product: str, mode: str) -> TargetSetDeclaration:
        return self.product(product).mode(mode)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _fail(where: str, message: str) -> TargetSetDeclarationError:
    return TargetSetDeclarationError(f"{where}: {message}")


def _require_keys(
    where: str,
    raw: object,
    allowed: frozenset[str],
    required: Iterable[str],
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise _fail(where, "must be an object.")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _fail(where, f"unknown keys {unknown}.")
    missing = sorted(set(required) - set(raw))
    if missing:
        raise _fail(where, f"missing keys {missing}.")
    return raw


def _text(where: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _fail(where, "must be a nonempty string without outer spaces.")
    return value


def _names(where: str, value: object) -> frozenset[str] | None:
    if value == ANY:
        return None
    if not isinstance(value, list) or not value:
        raise _fail(where, f"must be {ANY!r} or a nonempty list of names.")
    names = [_text(where, item) for item in value]
    if len(set(names)) != len(names):
        raise _fail(where, "repeats a name.")
    if ANY in names:
        # A list holds literal names: ["*"] would match only a name "*".
        raise _fail(where, f"lists {ANY!r} as a name; give {ANY!r} alone, not a list.")
    return frozenset(names)


def _metadata_tests(where: str, raw: object) -> tuple[MetadataTest, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise _fail(where, "must be an object of key tests.")
    tests = []
    for key, test in raw.items():
        _text(f"{where}.{key}", key)
        if not isinstance(test, Mapping) or len(test) != 1:
            raise _fail(f"{where}.{key}", "must hold exactly one test.")
        ((kind, value),) = test.items()
        if kind not in _TEST_KINDS:
            raise _fail(
                f"{where}.{key}",
                f"unknown test {kind!r}; use one of {sorted(_TEST_KINDS)}.",
            )
        if kind == "present" and not isinstance(value, bool):
            raise _fail(f"{where}.{key}", "present takes true or false.")
        if kind == "not_prefix":
            _text(f"{where}.{key}", value)
        tests.append(MetadataTest(key, kind, value))
    return tuple(tests)


def _clause(where: str, raw: object) -> TargetSetClause:
    clause = _require_keys(
        where, raw, _CLAUSE_KEYS, ("id", "families", "levels", "rationale")
    )
    exclude = clause.get("exclude", [])
    if not isinstance(exclude, list):
        raise _fail(f"{where}.exclude", "must be a list of predicate names.")
    rule = clause.get("rule")
    return TargetSetClause(
        clause_id=_text(f"{where}.id", clause["id"]),
        families=_names(f"{where}.families", clause["families"]),
        levels=_names(f"{where}.levels", clause["levels"]),
        metadata=_metadata_tests(f"{where}.metadata", clause.get("metadata")),
        exclude=tuple(_text(f"{where}.exclude", name) for name in exclude),
        rule=None if rule is None else _text(f"{where}.rule", rule),
        rationale=_text(f"{where}.rationale", clause["rationale"]),
    )


def _level_rule(where: str, top: Mapping[str, Any]) -> LevelRule:
    """The declaration's level rule: ``level_rule``, or the ``level_keys`` form."""

    if ("level_keys" in top) == ("level_rule" in top):
        raise _fail(where, "give exactly one of level_keys and level_rule.")
    if "level_keys" in top:
        level_keys = top["level_keys"]
        if not isinstance(level_keys, list) or not level_keys:
            raise _fail(
                f"{where}.level_keys", "must be a nonempty list of metadata keys."
            )
        return LevelRule(tuple(_text(f"{where}.level_keys", key) for key in level_keys))
    at = f"{where}.level_rule"
    rule = _require_keys(at, top["level_rule"], _LEVEL_RULE_KEYS, _LEVEL_RULE_KEYS)
    keys = rule["keys"]
    if not isinstance(keys, list) or not keys:
        raise _fail(f"{at}.keys", "must be a nonempty list of metadata keys.")
    keys = tuple(_text(f"{at}.keys", key) for key in keys)
    if len(set(keys)) != len(keys):
        raise _fail(f"{at}.keys", "repeats a key.")
    if rule["conflict"] not in LEVEL_CONFLICTS:
        raise _fail(f"{at}.conflict", f"must be one of {list(LEVEL_CONFLICTS)}.")
    return LevelRule(
        keys=keys,
        values=_names(f"{at}.values", rule["values"]),
        conflict=rule["conflict"],
        rationale=_text(f"{at}.rationale", rule["rationale"]),
    )


def _built(where: str, raw: object) -> BuiltRows:
    built = _require_keys(
        where, raw, _BUILT_KEYS, ("id", "family", "levels", "source", "rationale")
    )
    levels = built["levels"]
    if not isinstance(levels, Mapping) or not levels:
        raise _fail(f"{where}.levels", "must map each option value to a level.")
    option = built.get("option")
    return BuiltRows(
        built_id=_text(f"{where}.id", built["id"]),
        family=_text(f"{where}.family", built["family"]),
        levels=MappingProxyType(
            {
                _text(f"{where}.levels", value): _text(f"{where}.levels", level)
                for value, level in levels.items()
            }
        ),
        option=None if option is None else _text(f"{where}.option", option),
        source=_text(f"{where}.source", built["source"]),
        rationale=_text(f"{where}.rationale", built["rationale"]),
    )


def _read_declaration_file(path: Path) -> object:
    """Read a declaration file; refuse a repeated key or a non-finite number.

    Plain ``json.loads`` keeps the last of two equal keys, so a second
    ``"state"`` mode or a second ``"levels"`` in a clause would replace the
    first one without a word.
    """

    where = str(path)

    def no_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _value in pairs]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            raise _fail(where, f"repeats the JSON key(s) {repeated}.")
        return dict(pairs)

    def no_constant(value: str) -> Any:
        raise _fail(where, f"holds the non-finite number {value}.")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_repeated_keys,
        parse_constant=no_constant,
    )


def load_target_set_declarations(
    source: str | Path | Mapping[str, Any],
) -> TargetSetDeclarations:
    """Read and check one country's declarations (a path or a parsed object).

    A file is read strictly: a repeated JSON key and a non-finite number are
    refused.
    """

    if isinstance(source, Mapping):
        raw, where = source, "target sets"
    else:
        path = Path(source)
        raw, where = _read_declaration_file(path), str(path)
    top = _require_keys(where, raw, _TOP_KEYS, ("version", "country", "products"))
    if type(top["version"]) is not int or top["version"] != DECLARATION_VERSION:
        raise _fail(where, f"version must be {DECLARATION_VERSION}.")
    country = _text(f"{where}.country", top["country"])
    level_rule = _level_rule(where, top)
    level_keys = level_rule.keys
    products_raw = top["products"]
    if not isinstance(products_raw, Mapping) or not products_raw:
        raise _fail(f"{where}.products", "must name at least one product.")
    products: dict[str, ProductDeclaration] = {}
    for product_name, product_raw in products_raw.items():
        at = f"{where}.products.{product_name}"
        product = _require_keys(at, product_raw, _PRODUCT_KEYS, _PRODUCT_KEYS)
        modes_raw = product["modes"]
        if not isinstance(modes_raw, Mapping) or not modes_raw:
            raise _fail(f"{at}.modes", "must name at least one mode.")
        modes: dict[str, TargetSetDeclaration] = {}
        for mode_name, mode_raw in modes_raw.items():
            mode_at = f"{at}.modes.{mode_name}"
            mode = _require_keys(
                mode_at, mode_raw, _MODE_KEYS, ("description", "compiled")
            )
            compiled_raw = mode["compiled"]
            if not isinstance(compiled_raw, list) or not compiled_raw:
                raise _fail(f"{mode_at}.compiled", "must list at least one clause.")
            compiled = tuple(
                _clause(f"{mode_at}.compiled[{index}]", clause)
                for index, clause in enumerate(compiled_raw)
            )
            built_raw = mode.get("built", [])
            if not isinstance(built_raw, list):
                raise _fail(f"{mode_at}.built", "must be a list.")
            built = tuple(
                _built(f"{mode_at}.built[{index}]", entry)
                for index, entry in enumerate(built_raw)
            )
            ids = [clause.clause_id for clause in compiled] + [
                b.built_id for b in built
            ]
            if len(set(ids)) != len(ids):
                raise _fail(mode_at, "repeats a clause or built id.")
            modes[_text(mode_at, mode_name)] = TargetSetDeclaration(
                country=country,
                product=product_name,
                mode=mode_name,
                description=_text(f"{mode_at}.description", mode["description"]),
                level_keys=level_keys,
                compiled=compiled,
                built=built,
                level_rule=level_rule,
            )
        default_mode = _text(f"{at}.default_mode", product["default_mode"])
        if default_mode not in modes:
            raise _fail(
                f"{at}.default_mode", f"{default_mode!r} is not a declared mode."
            )
        products[_text(at, product_name)] = ProductDeclaration(
            product=product_name,
            tool=_text(f"{at}.tool", product["tool"]),
            option=_text(f"{at}.option", product["option"]),
            default_mode=default_mode,
            modes=MappingProxyType(modes),
        )
    return TargetSetDeclarations(
        country=country,
        level_keys=level_keys,
        products=MappingProxyType(products),
        level_rule=level_rule,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSetSelection:
    """The selected specs, in order, with each clause's count and rule receipt."""

    declaration: TargetSetDeclaration
    specs: tuple[Any, ...]
    clause_counts: Mapping[str, int]
    rule_receipts: Mapping[str, Any]


def select_target_set(
    declaration: TargetSetDeclaration,
    specs: Iterable[Any],
    *,
    clause_ids: Iterable[str] | None = None,
    predicates: Mapping[str, Predicate] | None = None,
    rules: Mapping[str, Rule] | None = None,
) -> TargetSetSelection:
    """Select the declared target set from compiled ``specs``.

    Each compiled clause (only those in ``clause_ids``, when given) keeps the
    specs it accepts, in compile order; a clause with a rule passes them to
    the rule. The selection is the clauses' outputs in declaration order. A
    spec's level is read only when an active clause tests levels, so a
    closed level rule refuses only the specs that a selection places by
    level.

    Raises:
        KeyError: If ``clause_ids`` names a clause the declaration lacks.
        ValueError: If a clause names a predicate or rule not passed, or if
            two clauses keep the same spec.
    """

    specs = tuple(specs)
    predicates = dict(predicates or {})
    rules = dict(rules or {})
    if clause_ids is None:
        clauses = declaration.compiled
    else:
        wanted = set(clause_ids)
        for clause_id in sorted(wanted):
            declaration.clause(clause_id)
        clauses = tuple(c for c in declaration.compiled if c.clause_id in wanted)
    for clause in clauses:
        missing = [name for name in clause.exclude if name not in predicates]
        if missing:
            raise ValueError(
                f"{declaration.product}/{declaration.mode} clause "
                f"{clause.clause_id!r} excludes by {missing}, which the caller "
                "did not pass."
            )
        if clause.rule is not None and clause.rule not in rules:
            raise ValueError(
                f"{declaration.product}/{declaration.mode} clause "
                f"{clause.clause_id!r} names rule {clause.rule!r}, which the "
                "caller did not pass."
            )
    # A level is read only when a clause tests levels, so a closed level rule
    # does not refuse a target that no active clause places by level.
    if any(clause.levels is not None for clause in clauses):
        levels = [declaration.level_of(spec) for spec in specs]
    else:
        levels = [UNSPECIFIED_LEVEL] * len(specs)
    kept: list[Any] = []
    counts: dict[str, int] = {}
    receipts: dict[str, Any] = {}
    seen: dict[int, str] = {}
    for clause in clauses:
        picked: Sequence[Any] = [
            spec
            for spec, level in zip(specs, levels, strict=True)
            if clause.accepts(spec, level, predicates)
        ]
        if clause.rule is not None:
            picked, receipts[clause.clause_id] = rules[clause.rule](tuple(picked))
        for spec in picked:
            other = seen.setdefault(id(spec), clause.clause_id)
            if other != clause.clause_id:
                raise ValueError(
                    f"{declaration.product}/{declaration.mode}: clauses {other!r} "
                    f"and {clause.clause_id!r} both keep {spec.name!r}."
                )
        counts[clause.clause_id] = len(picked)
        kept.extend(picked)
    return TargetSetSelection(
        declaration=declaration,
        specs=tuple(kept),
        clause_counts=MappingProxyType(counts),
        rule_receipts=MappingProxyType(receipts),
    )


@dataclass(frozen=True)
class LevelSelection:
    """Selected targets in compile order, and a receipt detached from them."""

    registry: TargetRegistry
    _receipt_json: str

    @property
    def receipt(self) -> dict[str, object]:
        """A fresh receipt; a caller that changes it cannot change its bytes."""
        return json.loads(self._receipt_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_bytes(self) -> bytes:
        """The canonical receipt bytes."""
        return self._receipt_json.encode("utf-8")


def _require_level_mode(declaration: TargetSetDeclaration) -> None:
    """Refuse a mode that is not one clause for each level of a closed set."""

    at = f"{declaration.product}/{declaration.mode}"
    rule = declaration.level_rule
    if rule is None or rule.values is None:
        raise TargetSetDeclarationError(
            f"{at}: selection by level needs a level_rule with a closed set of values."
        )
    for clause in declaration.compiled:
        if (
            clause.families is not None
            or clause.levels != frozenset({clause.clause_id})
            or clause.metadata
            or clause.exclude
            or clause.rule is not None
        ):
            raise TargetSetDeclarationError(
                f"{at}: clause {clause.clause_id!r} is not a level clause (any "
                "family, levels [its own id], and no test, exclusion or rule)."
            )
    if set(declaration.clause_ids) != rule.values:
        raise TargetSetDeclarationError(
            f"{at}: the clauses {sorted(declaration.clause_ids)} are not the "
            f"levels {sorted(rule.values)}."
        )


def select_by_levels(
    registry: TargetRegistry,
    declaration: TargetSetDeclaration,
    levels: Sequence[str] | None = None,
) -> LevelSelection:
    """Keep the targets of the requested levels (all by default), in compile order.

    ``declaration`` is a mode with a closed level set and one clause for each
    level (:func:`_require_level_mode`). Each target's level comes from the
    declared rule, which refuses a target it cannot place. The kept targets
    are the ones the requested clauses keep (:func:`select_target_set`), in
    the registry's order. The receipt lists every target with its level and
    whether it was kept.

    Raises:
        TypeError: If ``registry`` is not a :class:`TargetRegistry`.
        TargetSetDeclarationError: If ``declaration`` is not a level mode.
        ValueError: If ``levels`` is empty, repeats a level or names one with
            outer spaces; if a requested level has no target; if a target's
            level cannot be read; or if nothing is kept.
    """

    if not isinstance(registry, TargetRegistry):
        raise TypeError("Target selection requires TargetRegistry.")
    _require_level_mode(declaration)
    requested = None
    if levels is not None:
        if isinstance(levels, str | bytes):
            raise ValueError("geography_levels must be a nonempty sequence of levels.")
        requested = tuple(levels)
        if not requested or any(
            not isinstance(level, str) or not level or level != level.strip()
            for level in requested
        ):
            raise ValueError("geography_levels must contain nonempty literal names.")
        if len(set(requested)) != len(requested):
            raise ValueError("geography_levels must not repeat levels.")
        requested = tuple(sorted(requested))
    resolved = [declaration.level_of(spec) for spec in registry.specs]
    if requested is not None and (unknown := set(requested) - set(resolved)):
        raise ValueError(
            f"Target selector contains unknown geography levels {sorted(unknown)}."
        )
    kept = {
        id(spec)
        for spec in select_target_set(
            declaration, registry.specs, clause_ids=requested
        ).specs
    }
    selected = registry.select(predicate=lambda spec: id(spec) in kept)
    if not selected.specs:
        raise ValueError("Target selection requires a nonempty selected problem.")
    included, excluded = [], []
    for spec, level in zip(registry.specs, resolved, strict=True):
        row = {
            "name": spec.name,
            "period": spec.period,
            "geography_level": level,
            "family": spec.family,
            "source": spec.source,
        }
        if id(spec) in kept:
            reason = "all_geographies" if requested is None else "geography_selected"
            included.append({**row, "reason": reason})
        else:
            excluded.append({**row, "reason": "geography_not_selected"})
    receipt = {
        "schema": SELECTION_RECEIPT_SCHEMA,
        "source_registry_version": registry.version,
        "selected_registry_version": selected.version,
        "selector": {"geography_levels": requested, "explicit": requested is not None},
        "included": included,
        "excluded": excluded,
    }
    return LevelSelection(
        selected,
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )


def clause_predicate(
    declaration: TargetSetDeclaration,
    clause_id: str,
    *,
    predicates: Mapping[str, Predicate] | None = None,
    test_family: bool = True,
) -> Predicate:
    """One compiled clause as a spec predicate; its rule is not applied.

    With ``test_family=False`` the predicate skips the family test, for a
    caller that has already selected the family.
    """

    clause = declaration.clause(clause_id)
    predicates = dict(predicates or {})
    missing = [name for name in clause.exclude if name not in predicates]
    if missing:
        raise ValueError(
            f"{declaration.product}/{declaration.mode} clause {clause_id!r} "
            f"excludes by {missing}, which the caller did not pass."
        )

    def accepted(spec: Any) -> bool:
        level = declaration.level_of(spec)
        if test_family:
            return clause.accepts(spec, level, predicates)
        return clause.accepts_level_and_metadata(spec, level, predicates)

    return accepted


def target_set_census(
    specs: Iterable[Any], level_keys: Sequence[str]
) -> dict[tuple[str, str], int]:
    """Spec counts by (family, level), sorted."""

    census = Counter(
        (str(spec.family), target_level(spec, level_keys)) for spec in specs
    )
    return dict(sorted(census.items()))


def undeclared_cells(
    declaration: TargetSetDeclaration, specs: Iterable[Any]
) -> dict[tuple[str, str], int]:
    """(family, level) cells of ``specs`` that no clause or built entry declares.

    Metadata tests, exclusions and rules are not applied: a cell is declared
    when some compiled clause names its family and level (or any), or some
    built entry gives its family that level.
    """

    def declared(family: str, level: str) -> bool:
        for clause in declaration.compiled:
            if (clause.families is None or family in clause.families) and (
                clause.levels is None or level in clause.levels
            ):
                return True
        return any(
            built.family == family and level in built.levels.values()
            for built in declaration.built
        )

    census = target_set_census(specs, declaration.level_keys)
    return {cell: n for cell, n in census.items() if not declared(*cell)}
