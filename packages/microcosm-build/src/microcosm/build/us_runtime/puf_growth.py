"""Pure per-field growth contract and transform for a raw-source PUF successor.

This is the first bounded slice of `microcosm#530
<https://github.com/PolicyEngine/microcosm/issues/530>`_ item 2: *pin the raw
PUF asset and implement PUF aging as a Microcosm stage, under the same uprating
discipline as the other sources*. Nothing here reads a file, acquires an
artifact, imports a country engine, or generates a factor. It compiles an
explicit contract and applies it to a table that the caller already holds.

What the contract makes explicit, per consumed field
----------------------------------------------------
Source reference year, target year, role (money / code / identifier / count /
design weight), growth kind (nominal price restatement versus real per-return
aging), sign branch, and the exact factor-series identity. A field with no
declared rule is refused; a field with two rules that could both select the
same row is refused; there is no implicit factor of one and no dispatch by
string substring. Every factor is resolved by naming a **variable row** and two
**year columns** of the factor table explicitly, and is applied to a given row
at most once.

Why the row/column axes are named rather than iterated
------------------------------------------------------
The archived producer looped ``for variable in uprating`` over a DataFrame
pivoted to year columns, so it iterated years while believing it iterated
variable names (:data:`ARCHIVED_PRODUCER_COMMIT`, ``datasets/puf/puf.py``
against ``utils/uprating.py``). Its 2015→2021 step separately listed two of its
three sign-branched fields in a fallback group as well, so for those two a
sign-specific factor was followed by a second growth factor. Both failure modes are structural
here: a series name is looked up in a variable-keyed mapping, a year is looked
up in a year-keyed mapping, and the sign-branch cover is validated to be exactly
disjoint before any array is touched.

Authority
---------
A factor table declares its own authority. ``invented_fixture_nonauthority`` is
the only authority this slice can compile: it is sufficient to prove the
contract and the transform, and :attr:`CompiledPufGrowth.release_eligible` is
``False`` for it. ``reviewed_resource`` is refused until a reviewed table is
generated, pinned, and added to :data:`REVIEWED_FACTOR_TABLE_DIGESTS`, which is
deliberately empty. An invented fixture can never be promoted into a release
recipe by passing a flag.

Scope
-----
Money growth and design-weight growth are separate entry points, so a money
transform cannot move a weight. Codes, identifiers and counts are declared but
never written, and preservation is verified rather than assumed. The raw
``RECID`` and the source-year AGI used for SOI band assignment are copied into
separate provenance columns before any growth, so a later change of AGI basis
cannot destroy the band input. Nothing here decides whether the pinned
processed artifact is comparable to a rebuilt one; that artifact stays an
opaque legacy input, outside this contract.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.monetary_targets import MonetaryBasis
from microcosm.build.us_runtime.puf_source_agi import PUF_SOURCE_YEAR

__all__ = [
    "DESIGN_WEIGHT_FIELD",
    "GROWN_MONEY_VALUATION",
    "MAX_EXACT_AMOUNT",
    "PROVENANCE_COLUMNS",
    "PROVENANCE_RECID_COLUMN",
    "PROVENANCE_SOURCE_AGI_COLUMN",
    "ARCHIVED_PRODUCER_COMMIT",
    "PROPOSED_TY2015_FIELD_ROSTER",
    "ROSTER_CITATION",
    "ROSTER_SOURCE_REFERENCE_YEAR",
    "REVIEWED_FACTOR_TABLE_DIGESTS",
    "REVIEWED_PROVENANCE_KEYS",
    "SOURCE_AGI_FIELD",
    "SOURCE_RECID_FIELD",
    "CompiledPufGrowth",
    "FactorAuthority",
    "FactorSeries",
    "FieldRole",
    "GrowthFactorTable",
    "GrowthKind",
    "GrowthRecipe",
    "GrowthRule",
    "GrownPufTable",
    "PufGrowthRefusalError",
    "ResolvedFactor",
    "RosterEntry",
    "UNDOCUMENTED_SOURCE_FIELDS",
    "SignBranch",
    "apply_puf_design_weight_growth",
    "apply_puf_growth",
    "compile_puf_growth",
    "parse_growth_recipe",
    "roster_role",
    "puf_growth_monetary_metadata",
    "require_transformable",
]

#: Schema version of the recipe identity this module emits.
RECIPE_SCHEMA_VERSION = 1

#: Schema version of the factor-table document this module parses.
FACTOR_TABLE_SCHEMA_VERSION = 1

#: Raw IRS PUF field carrying the return identifier.
SOURCE_RECID_FIELD = "RECID"

#: Raw IRS PUF field carrying adjusted gross income.
SOURCE_AGI_FIELD = "E00100"

#: Raw IRS PUF field carrying the sample weight (stored times 100).
DESIGN_WEIGHT_FIELD = "S006"

#: Copy of ``RECID``, written from the source before any growth. No rule may
#: target this column or take its name, but the column itself is an ordinary
#: pandas column: what protects it is the contract, and in the graph the fact
#: that no node declares it as a rewrite.
PROVENANCE_RECID_COLUMN = "puf_source_recid"

#: Copy of source-year AGI on the same terms, kept for the TY2015 SOI band
#: assignment in :mod:`microcosm.build.us_runtime.puf_source_agi`.
PROVENANCE_SOURCE_AGI_COLUMN = "puf_source_year_agi"

#: The provenance columns this transform adds, in output order.
PROVENANCE_COLUMNS = (PROVENANCE_RECID_COLUMN, PROVENANCE_SOURCE_AGI_COLUMN)

#: SHA-256 digests of factor tables that a reviewed authority has accepted.
#: Empty: no factor table has been generated or reviewed for this lineage, so
#: ``reviewed_resource`` authority cannot compile. Adding a digest here is the
#: reviewed act; it is not something a caller can supply.
REVIEWED_FACTOR_TABLE_DIGESTS: tuple[str, ...] = ()

#: What a ``reviewed_resource`` factor table must state about itself. Nothing
#: can satisfy this yet, because :data:`REVIEWED_FACTOR_TABLE_DIGESTS` is
#: empty; the set is here so the receipt a reviewed table owes is written down
#: executably rather than only in prose.
REVIEWED_PROVENANCE_KEYS = frozenset(
    {
        "dependency_versions",
        "generated_by",
        "generator_code_sha256",
        "index_or_total",
        "parameter_paths",
        "population_divisor",
        "rounding",
        "source_url",
    }
)

#: Largest integer amount float64 represents exactly. A larger integer money
#: value is refused rather than silently rounded by the working conversion.
MAX_EXACT_AMOUNT = 2**53

#: Largest table this transform will accept, so a mistake is a refusal rather
#: than an unbounded allocation.
MAX_ROWS = 2_000_000

#: Largest factor document this transform will parse.
MAX_DOCUMENT_BYTES = 512 * 1024

_INTEGER_KINDS = frozenset("iu")
_MONEY_KINDS_ALLOWED = frozenset("iuf")
_RESULT_TOKEN = object()


class PufGrowthRefusalError(ValueError):
    """Sanitized field/reason refusal; never carries row values or identifiers."""

    def __init__(self, reason: str, field: str = "contract") -> None:
        self.reason = reason
        self.field = field
        super().__init__(f"{field}: {reason}")


def _require(condition: bool, reason: str, field: str = "contract") -> None:
    if not condition:
        raise PufGrowthRefusalError(reason, field)


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("ascii")


def _parse(document: bytes) -> dict:
    """Parse a bounded JSON object, refusing duplicate keys and non-finite values."""
    _require(
        type(document) is bytes and 0 < len(document) <= MAX_DOCUMENT_BYTES,
        "DOCUMENT_SIZE_OR_TYPE",
    )

    def pairs(items):
        result: dict = {}
        for key, item in items:
            _require(key not in result, "DUPLICATE_DOCUMENT_KEY")
            result[key] = item
        return result

    def constant(_):
        raise PufGrowthRefusalError("NONFINITE_DOCUMENT")

    try:
        parsed = json.loads(document, object_pairs_hook=pairs, parse_constant=constant)
    except PufGrowthRefusalError:
        raise
    except (ValueError, RecursionError) as error:
        raise PufGrowthRefusalError("MALFORMED_DOCUMENT") from error
    _require(type(parsed) is dict, "DOCUMENT_OBJECT_REQUIRED")
    return parsed


def _digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier(value: object, reason: str, field: str = "contract") -> str:
    _require(
        type(value) is str and 0 < len(value) <= 128 and value.strip() == value,
        reason,
        field,
    )
    return value  # type: ignore[return-value]


def _year(value: object, reason: str, field: str = "contract") -> int:
    _require(type(value) is int and 1900 <= value <= 2200, reason, field)
    return value  # type: ignore[return-value]


class FieldRole(StrEnum):
    """What a consumed raw field is, which fixes what may happen to it."""

    MONEY = "money"
    CODE = "code"
    IDENTIFIER = "identifier"
    COUNT = "count"
    DESIGN_WEIGHT = "design_weight"


class GrowthKind(StrEnum):
    """How a declared factor is meant, which is not recoverable from its value."""

    #: Restates the same nominal amount into another year's purchasing power.
    NOMINAL_PRICE_RESTATEMENT = "nominal_price_restatement"
    #: Ages an amount per return: an aggregate total ratio divided by a return
    #: count ratio, so it carries real change as well as price change.
    REAL_PER_RETURN_AGING = "real_per_return_aging"
    #: Grows the design weight itself. Never applied to money.
    DESIGN_WEIGHT_GROWTH = "design_weight_growth"
    #: Declared and deliberately untouched.
    NONE = "none"


class SignBranch(StrEnum):
    """Which rows of a money field a rule may select."""

    ANY = "any"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class FactorAuthority(StrEnum):
    """What a factor table claims about itself."""

    INVENTED_FIXTURE = "invented_fixture_nonauthority"
    REVIEWED_RESOURCE = "reviewed_resource"


_MONEY_KINDS = frozenset(
    {GrowthKind.NOMINAL_PRICE_RESTATEMENT, GrowthKind.REAL_PER_RETURN_AGING}
)
_PRESERVED_ROLES = frozenset({FieldRole.CODE, FieldRole.IDENTIFIER, FieldRole.COUNT})

#: How a grown money column's valuation is named, per growth kind. The string
#: goes into :class:`~microcosm.build.monetary_targets.MonetaryBasis`, so a
#: later fit-boundary check compares a restated amount against an aged amount
#: as the different things they are.
GROWN_MONEY_VALUATION = MappingProxyType(
    {
        GrowthKind.NOMINAL_PRICE_RESTATEMENT: "nominal_price_restated_from_{source}",
        GrowthKind.REAL_PER_RETURN_AGING: "real_per_return_aged_from_{source}",
    }
)


@dataclass(frozen=True)
class GrowthRule:
    """One rule: which rows of which field grow by which factor series.

    Attributes:
        field: The raw source column this rule governs.
        role: What the field is. Only :attr:`FieldRole.MONEY` may carry a sign
            branch, and only money and the design weight may grow at all.
        source_reference_year: The year the observed amount belongs to.
        target_year: The year the output amount is stated in. Equal to
            ``source_reference_year`` only for a declared no-growth field.
        growth_kind: What the factor means. Fixed by the role for every role
            except money, which must choose restatement or aging explicitly.
        sign_branch: Which rows this rule selects. ``ANY`` must be the field's
            only rule; ``POSITIVE`` and ``NEGATIVE`` must both be present.
        factor_series: The exact variable-row name in the factor table. Empty
            only for a declared no-growth field.
        citation: Where the field's meaning was read. Descriptive.
    """

    field: str
    role: FieldRole
    source_reference_year: int
    target_year: int
    growth_kind: GrowthKind
    sign_branch: SignBranch = SignBranch.ANY
    factor_series: str = ""
    citation: str = ""

    def __post_init__(self) -> None:
        field = _identifier(self.field, "RULE_FIELD_NAME")
        _require(type(self.role) is FieldRole, "RULE_ROLE", field)
        _require(type(self.growth_kind) is GrowthKind, "RULE_GROWTH_KIND", field)
        _require(type(self.sign_branch) is SignBranch, "RULE_SIGN_BRANCH", field)
        _require(type(self.citation) is str, "RULE_CITATION", field)
        source = _year(self.source_reference_year, "RULE_SOURCE_YEAR", field)
        target = _year(self.target_year, "RULE_TARGET_YEAR", field)
        if self.role is FieldRole.MONEY:
            _require(self.growth_kind in _MONEY_KINDS, "MONEY_GROWTH_KIND", field)
        elif self.role is FieldRole.DESIGN_WEIGHT:
            _require(
                self.growth_kind is GrowthKind.DESIGN_WEIGHT_GROWTH,
                "WEIGHT_GROWTH_KIND",
                field,
            )
            _require(self.sign_branch is SignBranch.ANY, "WEIGHT_SIGN_BRANCH", field)
        else:
            _require(
                self.growth_kind is GrowthKind.NONE, "PRESERVED_GROWTH_KIND", field
            )
            _require(self.sign_branch is SignBranch.ANY, "PRESERVED_SIGN_BRANCH", field)
            _require(source == target, "PRESERVED_YEAR_SPAN", field)
        if self.growth_kind is GrowthKind.NONE:
            _require(self.factor_series == "", "PRESERVED_FACTOR_SERIES", field)
        else:
            _identifier(self.factor_series, "RULE_FACTOR_SERIES", field)
            _require(target >= source, "RULE_YEAR_ORDER", field)

    @property
    def grows(self) -> bool:
        """Whether this rule multiplies anything at all."""
        return self.growth_kind is not GrowthKind.NONE

    def as_document(self) -> dict:
        """Canonical, hashable description of this rule."""
        return {
            "field": self.field,
            "role": str(self.role),
            "source_reference_year": self.source_reference_year,
            "target_year": self.target_year,
            "growth_kind": str(self.growth_kind),
            "sign_branch": str(self.sign_branch),
            "factor_series": self.factor_series,
        }


@dataclass(frozen=True)
class GrowthRecipe:
    """A closed set of rules covering every consumed field exactly once.

    Coverage is validated here, before any table is seen: a money field is
    covered either by one ``ANY`` rule or by exactly one ``POSITIVE`` and one
    ``NEGATIVE`` rule, and never by both shapes. That is the structural refusal
    of the archived overlap, where a sign-branched field also sat in a fallback
    group and could receive a second factor.
    """

    recipe_id: str
    rules: tuple[GrowthRule, ...]

    def __post_init__(self) -> None:
        _identifier(self.recipe_id, "RECIPE_ID")
        _require(
            type(self.rules) is tuple
            and bool(self.rules)
            and all(type(rule) is GrowthRule for rule in self.rules),
            "RECIPE_RULES",
        )
        seen: set[tuple[str, SignBranch]] = set()
        branches: dict[str, set[SignBranch]] = {}
        shape: dict[str, tuple[FieldRole, int, int, GrowthKind]] = {}
        for rule in self.rules:
            key = (rule.field, rule.sign_branch)
            _require(key not in seen, "DUPLICATE_FIELD_RULE", rule.field)
            seen.add(key)
            branches.setdefault(rule.field, set()).add(rule.sign_branch)
            declaration = (
                rule.role,
                rule.source_reference_year,
                rule.target_year,
                rule.growth_kind,
            )
            if rule.field in shape:
                _require(
                    shape[rule.field] == declaration,
                    "AMBIGUOUS_FIELD_DECLARATION",
                    rule.field,
                )
            else:
                shape[rule.field] = declaration
        for name, present in branches.items():
            if present == {SignBranch.ANY}:
                continue
            _require(SignBranch.ANY not in present, "OVERLAPPING_SIGN_RULES", name)
            _require(
                present == {SignBranch.POSITIVE, SignBranch.NEGATIVE},
                "INCOMPLETE_SIGN_COVER",
                name,
            )
        for rule in self.rules:
            _require(
                rule.field not in PROVENANCE_COLUMNS,
                "PROVENANCE_COLUMN_NAME",
                rule.field,
            )
        money = [rule for rule in self.rules if rule.role is FieldRole.MONEY]
        _require(bool(money), "NO_MONEY_FIELD")
        identifiers = [
            rule.field for rule in self.rules if rule.role is FieldRole.IDENTIFIER
        ]
        _require(len(set(identifiers)) == 1, "PROVENANCE_RECID_FIELD")
        _require(
            any(rule.field == SOURCE_AGI_FIELD for rule in money),
            "PROVENANCE_AGI_FIELD",
            SOURCE_AGI_FIELD,
        )
        money_targets = {rule.target_year for rule in money}
        _require(len(money_targets) == 1, "MIXED_TARGET_YEAR")
        _require(
            len({rule.source_reference_year for rule in money}) == 1,
            "MIXED_SOURCE_REFERENCE_YEAR",
        )
        weights = [rule for rule in self.rules if rule.role is FieldRole.DESIGN_WEIGHT]
        _require(len(weights) <= 1, "MULTIPLE_DESIGN_WEIGHTS")
        if weights and money_targets:
            _require(
                weights[0].target_year == next(iter(money_targets)),
                "WEIGHT_TARGET_YEAR",
                weights[0].field,
            )
        object.__setattr__(self, "rules", tuple(sorted(self.rules, key=_rule_order)))

    @property
    def fields(self) -> tuple[str, ...]:
        """Every declared field name, once each, in the rules' sorted order.

        ``__post_init__`` sorts the rules by ``(field, sign_branch)``, so two
        recipes that declare the same rules in a different order are the same
        recipe and hash the same.
        """
        ordered: list[str] = []
        for rule in self.rules:
            if rule.field not in ordered:
                ordered.append(rule.field)
        return tuple(ordered)

    def rules_for(self, field: str) -> tuple[GrowthRule, ...]:
        """Every rule governing ``field``."""
        return tuple(rule for rule in self.rules if rule.field == field)

    @property
    def identity(self) -> bytes:
        """Canonical bytes identifying this recipe."""
        return _json(
            {
                "schema_version": RECIPE_SCHEMA_VERSION,
                "recipe_id": self.recipe_id,
                "rules": [rule.as_document() for rule in self.rules],
            }
        )


def parse_growth_recipe(document: bytes) -> GrowthRecipe:
    """Rebuild a recipe from its own :attr:`GrowthRecipe.identity` bytes.

    Round-tripping through the identity document is how a recipe travels as a
    graph node parameter without becoming an opaque blob: the bytes a node
    carries are exactly the bytes the recipe hashes to.

    Args:
        document: A recipe identity document.

    Returns:
        The recipe those bytes describe.

    Raises:
        PufGrowthRefusalError: If the document is malformed, is a different
            schema version, or does not round-trip to itself.
    """
    data = _parse(document)
    _require(
        set(data) == {"schema_version", "recipe_id", "rules"}, "RECIPE_DOCUMENT_SCHEMA"
    )
    _require(data["schema_version"] == RECIPE_SCHEMA_VERSION, "RECIPE_SCHEMA_VERSION")
    _require(
        type(data["rules"]) is list and bool(data["rules"]), "RECIPE_DOCUMENT_RULES"
    )
    rules = []
    for entry in data["rules"]:
        _require(
            type(entry) is dict
            and set(entry)
            == {
                "field",
                "role",
                "source_reference_year",
                "target_year",
                "growth_kind",
                "sign_branch",
                "factor_series",
            },
            "RECIPE_DOCUMENT_RULES",
        )
        for key, values in (
            ("role", FieldRole),
            ("growth_kind", GrowthKind),
            ("sign_branch", SignBranch),
        ):
            _require(
                entry[key] in tuple(str(value) for value in values),
                "RECIPE_DOCUMENT_RULES",
                entry["field"] if type(entry["field"]) is str else "contract",
            )
        rules.append(
            GrowthRule(
                entry["field"],
                FieldRole(entry["role"]),
                entry["source_reference_year"],
                entry["target_year"],
                GrowthKind(entry["growth_kind"]),
                SignBranch(entry["sign_branch"]),
                entry["factor_series"],
            )
        )
    recipe = GrowthRecipe(data["recipe_id"], tuple(rules))
    _require(recipe.identity == document, "RECIPE_DOCUMENT_ROUNDTRIP")
    return recipe


def _rule_order(rule: GrowthRule) -> tuple[str, str]:
    return (rule.field, str(rule.sign_branch))


@dataclass(frozen=True)
class FactorSeries:
    """One variable row of the factor table: a level per year column.

    Levels are index levels, not ratios. A factor is always a ratio of two
    explicitly named year columns of one explicitly named row, so a table
    cannot be read as if its year columns were variable names.
    """

    name: str
    kind: GrowthKind
    levels: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        name = _identifier(self.name, "SERIES_NAME")
        _require(type(self.kind) is GrowthKind, "SERIES_KIND", name)
        _require(self.kind is not GrowthKind.NONE, "SERIES_KIND", name)
        _require(
            type(self.levels) is tuple and bool(self.levels), "SERIES_LEVELS", name
        )
        years = [year for year, _ in self.levels]
        _require(len(set(years)) == len(years), "DUPLICATE_SERIES_YEAR", name)
        _require(years == sorted(years), "UNORDERED_SERIES_YEARS", name)
        for year, level in self.levels:
            _year(year, "SERIES_YEAR", name)
            _require(type(level) is str and level.strip() == level, "LEVEL_TEXT", name)
            _require(_finite_positive(level), "LEVEL_VALUE", name)

    def level_text(self, year: int) -> str:
        """The exact level text in the ``year`` column of this variable row."""
        for candidate, level in self.levels:
            if candidate == year:
                return level
        raise PufGrowthRefusalError("MISSING_FACTOR_YEAR", self.name)

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(year for year, _ in self.levels)


def _finite_positive(text: str) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    return bool(np.isfinite(value)) and value > 0.0


@dataclass(frozen=True)
class GrowthFactorTable:
    """A parsed factor document: variable rows by year columns, plus authority.

    The document is retained verbatim so the table's identity is the bytes a
    reviewer would read, not a re-serialization of this object's state.
    """

    document: bytes
    table_id: str
    authority: FactorAuthority
    years: tuple[int, ...]
    series: tuple[FactorSeries, ...]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        _identifier(self.table_id, "TABLE_ID")
        _require(type(self.authority) is FactorAuthority, "TABLE_AUTHORITY")
        _require(
            type(self.series) is tuple
            and bool(self.series)
            and all(type(item) is FactorSeries for item in self.series),
            "TABLE_SERIES",
        )
        names = [item.name for item in self.series]
        _require(len(set(names)) == len(names), "DUPLICATE_SERIES")
        _require(
            type(self.years) is tuple
            and bool(self.years)
            and list(self.years) == sorted(set(self.years)),
            "TABLE_YEARS",
        )
        for item in self.series:
            _require(item.years == self.years, "SERIES_YEAR_COVERAGE", item.name)
        # Re-serializing proves the retained bytes are the canonical document
        # for exactly this parsed state, so no field can drift from the bytes.
        _require(self.document == _json(self.as_document()), "TABLE_DOCUMENT")

    @classmethod
    def from_bytes(cls, document: bytes) -> GrowthFactorTable:
        """Parse a factor document, refusing anything it does not fully close."""
        data = _parse(document)
        _require(
            set(data)
            == {
                "authority",
                "provenance",
                "schema_version",
                "series",
                "table_id",
                "years",
            },
            "TABLE_SCHEMA",
        )
        _require(
            data["schema_version"] == FACTOR_TABLE_SCHEMA_VERSION,
            "TABLE_SCHEMA_VERSION",
        )
        _require(
            data["authority"] in tuple(str(value) for value in FactorAuthority),
            "TABLE_AUTHORITY",
        )
        _require(type(data["years"]) is list and bool(data["years"]), "TABLE_YEARS")
        years = tuple(_year(value, "TABLE_YEARS") for value in data["years"])
        _require(type(data["series"]) is dict and bool(data["series"]), "TABLE_SERIES")
        _require(type(data["provenance"]) is dict, "TABLE_PROVENANCE")
        series = []
        for name in sorted(data["series"]):
            entry = data["series"][name]
            _require(
                type(entry) is dict and set(entry) == {"kind", "levels"},
                "SERIES_SCHEMA",
                name,
            )
            _require(
                entry["kind"] in tuple(str(value) for value in GrowthKind),
                "SERIES_KIND",
                name,
            )
            levels = entry["levels"]
            _require(type(levels) is dict and bool(levels), "SERIES_LEVELS", name)
            _require(
                set(levels) == {str(year) for year in years},
                "SERIES_YEAR_COVERAGE",
                name,
            )
            series.append(
                FactorSeries(
                    name,
                    GrowthKind(entry["kind"]),
                    tuple((year, levels[str(year)]) for year in years),
                )
            )
        return cls(
            document,
            data["table_id"],
            FactorAuthority(data["authority"]),
            years,
            tuple(series),
            MappingProxyType(dict(data["provenance"])),
        )

    def as_document(self) -> dict:
        """The canonical document for this table's parsed state."""
        return {
            "authority": str(self.authority),
            "provenance": dict(self.provenance),
            "schema_version": FACTOR_TABLE_SCHEMA_VERSION,
            "series": {
                item.name: {
                    "kind": str(item.kind),
                    "levels": {str(year): level for year, level in item.levels},
                }
                for item in self.series
            },
            "table_id": self.table_id,
            "years": list(self.years),
        }

    def series_for(self, name: str) -> FactorSeries:
        """The variable row called ``name``; refuses a year column or a miss."""
        for item in self.series:
            if item.name == name:
                return item
        raise PufGrowthRefusalError("UNKNOWN_FACTOR_SERIES", name)

    @property
    def sha256(self) -> str:
        return _sha(self.document)


@dataclass(frozen=True)
class ResolvedFactor:
    """One rule's factor, resolved once from two named year columns."""

    field: str
    sign_branch: SignBranch
    series: str
    kind: GrowthKind
    source_year: int
    target_year: int
    source_level: str
    target_level: str
    ratio_bits: str

    def __post_init__(self) -> None:
        field = _identifier(self.field, "RESOLVED_FIELD")
        _require(type(self.sign_branch) is SignBranch, "RESOLVED_SIGN_BRANCH", field)
        _require(
            type(self.kind) is GrowthKind and self.kind is not GrowthKind.NONE,
            "RESOLVED_KIND",
            field,
        )
        _identifier(self.series, "RESOLVED_SERIES", field)
        _year(self.source_year, "RESOLVED_SOURCE_YEAR", field)
        _year(self.target_year, "RESOLVED_TARGET_YEAR", field)
        _require(
            _finite_positive(self.source_level) and _finite_positive(self.target_level),
            "RESOLVED_LEVEL",
            field,
        )
        # The stored bits are the factor. Recomputing them here means a
        # hand-built ResolvedFactor cannot claim a factor its own levels do not
        # produce, so the bits are evidence rather than an assertion.
        expected = struct.pack(
            "<d", float(self.target_level) / float(self.source_level)
        ).hex()
        _require(self.ratio_bits == expected, "RESOLVED_RATIO_BITS", field)

    @property
    def value(self) -> float:
        """The exact float64 this factor is, recovered from its bit pattern."""
        return struct.unpack("<d", bytes.fromhex(self.ratio_bits))[0]

    def as_document(self) -> dict:
        return {
            "field": self.field,
            "kind": str(self.kind),
            "ratio_bits": self.ratio_bits,
            "series": self.series,
            "sign_branch": str(self.sign_branch),
            "source_level": self.source_level,
            "source_year": self.source_year,
            "target_level": self.target_level,
            "target_year": self.target_year,
        }


@dataclass(frozen=True)
class CompiledPufGrowth:
    """A recipe bound to a factor table, with every factor already resolved."""

    recipe: GrowthRecipe
    factors: GrowthFactorTable
    resolved: tuple[ResolvedFactor, ...]
    identity: bytes

    def __post_init__(self) -> None:
        _require(type(self.recipe) is GrowthRecipe, "RECIPE_TYPE")
        _require(type(self.factors) is GrowthFactorTable, "FACTOR_TABLE_TYPE")
        # The authority gate lives here, not only in the compile function, so
        # that no construction path can produce a release-eligible contract the
        # compile chokepoint would have refused.
        if self.factors.authority is FactorAuthority.REVIEWED_RESOURCE:
            _require(
                self.factors.sha256 in REVIEWED_FACTOR_TABLE_DIGESTS,
                "UNREVIEWED_FACTOR_TABLE",
            )
            _require(
                set(self.factors.provenance) == REVIEWED_PROVENANCE_KEYS,
                "REVIEWED_PROVENANCE_SHAPE",
            )
        _require(
            type(self.resolved) is tuple
            and all(type(item) is ResolvedFactor for item in self.resolved),
            "RESOLVED_TYPE",
        )
        # Exactly one resolved factor per growing rule, agreeing with it, and
        # none left over: a directly constructed contract cannot describe a
        # different computation than its own recipe declares.
        expected = tuple(
            (
                rule.field,
                rule.sign_branch,
                rule.factor_series,
                rule.growth_kind,
                rule.source_reference_year,
                rule.target_year,
            )
            for rule in self.recipe.rules
            if rule.grows
        )
        actual = tuple(
            (
                factor.field,
                factor.sign_branch,
                factor.series,
                factor.kind,
                factor.source_year,
                factor.target_year,
            )
            for factor in self.resolved
        )
        _require(actual == expected, "RESOLVED_COVERAGE")
        _require(
            self.identity
            == _contract_identity(self.recipe, self.factors, self.resolved),
            "CONTRACT_BINDING",
        )

    @property
    def sha256(self) -> str:
        return _sha(self.identity)

    @property
    def release_eligible(self) -> bool:
        """Whether this contract may stand behind a release recipe.

        ``False`` for every invented fixture, and there is no override.
        """
        return self.factors.authority is FactorAuthority.REVIEWED_RESOURCE

    @property
    def money_fields(self) -> tuple[str, ...]:
        """Declared money fields, once each, in rule order."""
        return tuple(
            dict.fromkeys(
                rule.field for rule in self.recipe.rules if rule.role is FieldRole.MONEY
            )
        )

    @property
    def money_view_fields(self) -> tuple[str, ...]:
        """The columns :func:`apply_puf_growth` consumes: everything but weights.

        The design weight is excluded so a money transform neither reads nor
        depends on it. In the graph that keeps the money node and the
        design-weight node siblings rather than one ordering the other.
        """
        weight = self.design_weight_rule
        return tuple(
            name
            for name in self.recipe.fields
            if weight is None or name != weight.field
        )

    @property
    def preserved_fields(self) -> tuple[str, ...]:
        """Declared code/identifier/count fields, which this contract never writes."""
        return tuple(
            dict.fromkeys(
                rule.field
                for rule in self.recipe.rules
                if rule.role in _PRESERVED_ROLES
            )
        )

    @property
    def design_weight_rule(self) -> GrowthRule | None:
        for rule in self.recipe.rules:
            if rule.role is FieldRole.DESIGN_WEIGHT:
                return rule
        return None

    @property
    def target_year(self) -> int:
        for rule in self.recipe.rules:
            if rule.role is FieldRole.MONEY:
                return rule.target_year
        raise PufGrowthRefusalError("NO_MONEY_FIELD")

    def factor_for(self, field: str, sign_branch: SignBranch) -> ResolvedFactor:
        for factor in self.resolved:
            if factor.field == field and factor.sign_branch is sign_branch:
                return factor
        raise PufGrowthRefusalError("NO_RESOLVED_FACTOR", field)


def compile_puf_growth(
    recipe: GrowthRecipe, factors: GrowthFactorTable
) -> CompiledPufGrowth:
    """Bind a recipe to a factor table, resolving each factor exactly once.

    Args:
        recipe: The closed rule set. Already coverage-validated.
        factors: The factor table the rules name.

    Returns:
        An immutable compiled contract carrying its own identity bytes.

    Raises:
        PufGrowthRefusalError: If the table's authority is not compilable, if a
            rule names a series the table does not have, if a named series
            means a different growth kind than the rule declares, or if either
            declared year is not a column of that series.
    """
    _require(type(recipe) is GrowthRecipe, "RECIPE_TYPE")
    _require(type(factors) is GrowthFactorTable, "FACTOR_TABLE_TYPE")
    resolved = []
    for rule in recipe.rules:
        if not rule.grows:
            continue
        series = factors.series_for(rule.factor_series)
        _require(series.kind is rule.growth_kind, "FACTOR_KIND_MISMATCH", rule.field)
        source_level = series.level_text(rule.source_reference_year)
        target_level = series.level_text(rule.target_year)
        ratio = float(target_level) / float(source_level)
        _require(bool(np.isfinite(ratio)) and ratio > 0.0, "FACTOR_VALUE", rule.field)
        resolved.append(
            ResolvedFactor(
                rule.field,
                rule.sign_branch,
                series.name,
                rule.growth_kind,
                rule.source_reference_year,
                rule.target_year,
                source_level,
                target_level,
                struct.pack("<d", ratio).hex(),
            )
        )
    resolved = tuple(resolved)
    return CompiledPufGrowth(
        recipe, factors, resolved, _contract_identity(recipe, factors, resolved)
    )


def _contract_identity(
    recipe: GrowthRecipe,
    factors: GrowthFactorTable,
    resolved: tuple[ResolvedFactor, ...],
) -> bytes:
    """Canonical bytes identifying one recipe bound to one factor table."""
    return _json(
        {
            "schema_version": RECIPE_SCHEMA_VERSION,
            "recipe": json.loads(recipe.identity),
            "factor_table_sha256": factors.sha256,
            "factor_table_id": factors.table_id,
            "factor_authority": str(factors.authority),
            "resolved": [factor.as_document() for factor in resolved],
        }
    )


def puf_growth_monetary_metadata(
    compiled: CompiledPufGrowth,
) -> Mapping[str, MonetaryBasis]:
    """Declared output basis per money field, for a later fit-boundary check.

    The check itself is not in this slice. What is here is the metadata it
    needs: currency, target period, and a valuation string that distinguishes a
    nominal restatement from a real per-return aging, so a donor field and a
    recipient predictor cannot be silently matched across different bases.

    Args:
        compiled: A compiled contract.

    Returns:
        Money field name to its declared output
        :class:`~microcosm.build.monetary_targets.MonetaryBasis`.
    """
    _require(type(compiled) is CompiledPufGrowth, "COMPILED_TYPE")
    result: dict[str, MonetaryBasis] = {}
    for rule in compiled.recipe.rules:
        if rule.role is not FieldRole.MONEY or rule.field in result:
            continue
        result[rule.field] = MonetaryBasis(
            currency="USD",
            unit="base_currency",
            period=str(rule.target_year),
            temporal_basis="annual_flow",
            sector="households",
            perimeter="irs_individual_income_tax_returns",
            valuation=GROWN_MONEY_VALUATION[rule.growth_kind].format(
                source=rule.source_reference_year
            ),
        )
    return MappingProxyType(result)


@dataclass(frozen=True, eq=False)
class GrownPufTable:
    """A grown table and its receipt; only :func:`apply_puf_growth` builds one."""

    table: pd.DataFrame
    contract_sha256: str
    _receipt: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        _require(_token is _RESULT_TOKEN, "RESULT_CONSTRUCTOR_UNAVAILABLE")

    @property
    def receipt(self) -> dict:
        """The parsed receipt describing exactly what was applied."""
        return _parse(self._receipt)

    @property
    def receipt_bytes(self) -> bytes:
        return self._receipt


def _checked_source_table(
    table: pd.DataFrame, compiled: CompiledPufGrowth
) -> pd.DataFrame:
    _require(type(table) is pd.DataFrame, "SOURCE_TABLE_TYPE")
    rows = len(table)
    _require(0 < rows <= MAX_ROWS, "SOURCE_TABLE_ROWS")
    _require(all(type(name) is str for name in table.columns), "SOURCE_TABLE_COLUMNS")
    _require(len(set(table.columns)) == len(table.columns), "DUPLICATE_SOURCE_COLUMN")
    declared = set(compiled.money_view_fields)
    present = set(table.columns)
    weight = compiled.design_weight_rule
    if weight is not None and weight.field in present:
        raise PufGrowthRefusalError("DESIGN_WEIGHT_NOT_IN_MONEY_VIEW", weight.field)
    missing = sorted(declared - present)
    if missing:
        raise PufGrowthRefusalError("MISSING_DECLARED_FIELD", missing[0])
    undeclared = sorted(present - declared)
    if undeclared:
        raise PufGrowthRefusalError("UNDECLARED_SOURCE_COLUMN", undeclared[0])
    return table


def _money_values(table: pd.DataFrame, field: str) -> np.ndarray:
    """A money column as float64, refusing anything it would have to coerce."""
    series = table[field]
    # An explicit kind allow-list rather than ``is_numeric_dtype``, which also
    # admits complex, whose imaginary part a float64 cast would drop in silence.
    _require(
        series.dtype.kind in _MONEY_KINDS_ALLOWED
        or (
            pd.api.types.is_extension_array_dtype(series.dtype)
            and str(series.dtype).lower().startswith(("int", "uint", "float"))
        ),
        "FIELD_DTYPE",
        field,
    )
    _require(not series.isna().any(), "FIELD_MISSING_VALUE", field)
    integral = series.dtype.kind in _INTEGER_KINDS or (
        pd.api.types.is_extension_array_dtype(series.dtype)
        and str(series.dtype).lower().startswith(("int", "uint"))
    )
    if integral:
        # float64 is the working type, so an integer amount it cannot represent
        # exactly is refused rather than silently rounded.
        _require(
            bool((np.abs(series.to_numpy(dtype=object)) <= MAX_EXACT_AMOUNT).all()),
            "MONEY_MAGNITUDE",
            field,
        )
    values = series.to_numpy(dtype=np.float64, copy=True)
    _require(bool(np.isfinite(values).all()), "FIELD_NONFINITE", field)
    # A negative zero would pass every arithmetic guard untouched and then
    # compare equal to zero while carrying a different bit pattern.
    _require(not bool(np.signbit(values[values == 0.0]).any()), "NEGATIVE_ZERO", field)
    return values


def _integral_values(table: pd.DataFrame, field: str, reason: str) -> np.ndarray:
    series = table[field]
    _require(
        series.dtype.kind in _INTEGER_KINDS
        or (
            pd.api.types.is_extension_array_dtype(series.dtype)
            and str(series.dtype).lower().startswith(("int", "uint"))
        ),
        reason,
        field,
    )
    _require(not series.isna().any(), "FIELD_MISSING_VALUE", field)
    return series.to_numpy(copy=True)


def _selection(values: np.ndarray, branch: SignBranch) -> np.ndarray:
    """Rows this branch selects. Zero is never selected, so it never grows."""
    if branch is SignBranch.POSITIVE:
        return values > 0.0
    if branch is SignBranch.NEGATIVE:
        return values < 0.0
    return values != 0.0


def require_transformable(compiled: CompiledPufGrowth) -> None:
    """Check every precondition :func:`apply_puf_growth` needs, without a table.

    A caller that is about to declare work — a graph node factory, say — uses
    this so a contract that could never run is refused before anything is
    loaded, rather than at the moment the money transform reaches for a field
    that is not there.

    Args:
        compiled: The contract to check.

    Raises:
        PufGrowthRefusalError: If the contract declares no money field, no
            single identifier, or no source-year AGI field.
    """
    _require(type(compiled) is CompiledPufGrowth, "COMPILED_TYPE")
    _ = compiled.target_year
    _source_reference_year(compiled)
    _recid_field(compiled)
    _agi_field(compiled)


def apply_puf_growth(table: pd.DataFrame, compiled: CompiledPufGrowth) -> GrownPufTable:
    """Grow every declared money field once, preserving everything else.

    The input is never mutated: every refusal happens before any array is
    written, and the output is a new frame built from copies.

    Args:
        table: One row per raw source record, whose columns are exactly
            :attr:`CompiledPufGrowth.money_view_fields` — every declared field
            except the design weight.
        compiled: The compiled contract.

    Row correspondence is positional, not by pandas label: the output carries a
    fresh ``RangeIndex`` and row *i* of the output is row *i* of the input. A
    caller that needs the input's labels back must reattach them.

    The input is never mutated and no partial result is ever returned. Most
    refusals happen before any arithmetic; the product guards
    (``GROWN_NONFINITE``, ``SIGN_CHANGED``, ``FACTOR_APPLIED_TWICE``)
    necessarily run after a candidate array has been computed in local memory.

    Returns:
        A :class:`GrownPufTable` whose frame carries the grown money columns,
        the byte-identical code/identifier/count columns, and the two
        provenance columns copied from the source before any growth. The design
        weight is not part of this view at all; it grows only through
        :func:`apply_puf_design_weight_growth`.

    Raises:
        PufGrowthRefusalError: On any contract, dtype, value, or coverage
            violation. Nothing is written when this is raised.
    """
    _require(type(compiled) is CompiledPufGrowth, "COMPILED_TYPE")
    source = _checked_source_table(table, compiled)
    rows = len(source)

    # Validate every column and resolve every factor before writing anything.
    grown: dict[str, np.ndarray] = {}
    applications: dict[str, int] = {}
    for name in compiled.money_view_fields:
        rules = compiled.recipe.rules_for(name)
        role = rules[0].role
        if role is FieldRole.MONEY:
            values = _money_values(source, name)
            applied = np.zeros(rows, dtype=np.int64)
            result = values.copy()
            for rule in rules:
                factor = compiled.factor_for(name, rule.sign_branch)
                selected = _selection(values, rule.sign_branch)
                applied += selected.astype(np.int64)
                # The product is checked below; a warning here would only
                # duplicate a refusal that is already explicit.
                with np.errstate(over="ignore", under="ignore"):
                    result[selected] = values[selected] * factor.value
            _require(bool((applied <= 1).all()), "FACTOR_APPLIED_TWICE", name)
            _require(bool(np.isfinite(result).all()), "GROWN_NONFINITE", name)
            _require(
                bool((np.sign(result) == np.sign(values)).all()), "SIGN_CHANGED", name
            )
            grown[name] = result
            applications[name] = int(applied.sum())
        else:
            _integral_values(source, name, "PRESERVED_FIELD_DTYPE")

    recid_field = _recid_field(compiled)
    agi_field = _agi_field(compiled)
    provenance_recid = _integral_values(source, recid_field, "PROVENANCE_RECID_DTYPE")
    _require(len(np.unique(provenance_recid)) == rows, "DUPLICATE_RECID", recid_field)
    provenance_agi = _money_values(source, agi_field)

    output = pd.DataFrame(index=pd.RangeIndex(rows))
    for name in source.columns:
        output[name] = (
            pd.Series(grown[name], index=output.index, dtype="float64")
            if name in grown
            else source[name].reset_index(drop=True).copy(deep=True)
        )
    output[PROVENANCE_RECID_COLUMN] = pd.Series(
        provenance_recid, index=output.index, dtype=source[recid_field].dtype
    )
    output[PROVENANCE_SOURCE_AGI_COLUMN] = pd.Series(
        provenance_agi, index=output.index, dtype="float64"
    )

    receipt = _json(
        {
            "schema_version": RECIPE_SCHEMA_VERSION,
            "artifact_kind": "microcosm.us_puf_growth",
            "contract_sha256": _sha(compiled.identity),
            "factor_authority": str(compiled.factors.authority),
            "factor_table_sha256": compiled.factors.sha256,
            "release_eligible": compiled.release_eligible,
            "rows": rows,
            "source_reference_year": _source_reference_year(compiled),
            "target_year": compiled.target_year,
            "grown_money_fields": sorted(grown),
            "grown_row_counts": {name: applications[name] for name in sorted(grown)},
            "preserved_fields": sorted(compiled.preserved_fields),
            "design_weight_field": (
                compiled.design_weight_rule.field
                if compiled.design_weight_rule is not None
                else None
            ),
            "design_weight_in_money_view": False,
            "provenance_columns": list(PROVENANCE_COLUMNS),
            "provenance_recid_source": recid_field,
            "provenance_agi_source": agi_field,
        }
    )
    return GrownPufTable(output, _sha(compiled.identity), receipt, _token=_RESULT_TOKEN)


def apply_puf_design_weight_growth(
    weights: Sequence[float] | np.ndarray, compiled: CompiledPufGrowth
) -> np.ndarray:
    """Grow design weights by their own declared factor, and nothing else.

    Kept apart from :func:`apply_puf_growth` on purpose: money growth and
    weight growth are different operations with different factor series, and a
    single call that did both could silently move population with prices.

    Args:
        weights: The source design weights, one per record.
        compiled: The compiled contract, which must declare a design weight.

    Returns:
        A new float64 array of grown weights.

    Raises:
        PufGrowthRefusalError: If no design-weight rule is declared, or the
            weights are not finite and positive.
    """
    _require(type(compiled) is CompiledPufGrowth, "COMPILED_TYPE")
    rule = compiled.design_weight_rule
    _require(rule is not None, "NO_DESIGN_WEIGHT_RULE")
    values = np.asarray(weights, dtype=np.float64)
    _require(values.ndim == 1 and 0 < values.size <= MAX_ROWS, "WEIGHT_SHAPE")
    _require(bool(np.isfinite(values).all()), "WEIGHT_NONFINITE")
    _require(bool((values > 0.0).all()), "NONPOSITIVE_DESIGN_WEIGHT")
    factor = compiled.factor_for(rule.field, SignBranch.ANY)
    _require(
        factor.kind is GrowthKind.DESIGN_WEIGHT_GROWTH, "WEIGHT_FACTOR_KIND", rule.field
    )
    with np.errstate(over="ignore", under="ignore"):
        grown = values * factor.value
    _require(bool(np.isfinite(grown).all()), "GROWN_WEIGHT_NONFINITE", rule.field)
    _require(bool((grown > 0.0).all()), "GROWN_WEIGHT_NONPOSITIVE", rule.field)
    return grown


def _source_reference_year(compiled: CompiledPufGrowth) -> int:
    years = {
        rule.source_reference_year
        for rule in compiled.recipe.rules
        if rule.role is FieldRole.MONEY
    }
    _require(len(years) == 1, "MIXED_SOURCE_REFERENCE_YEAR")
    return years.pop()


def _recid_field(compiled: CompiledPufGrowth) -> str:
    """The one declared identifier. :class:`GrowthRecipe` already required it."""
    candidates = [
        rule.field
        for rule in compiled.recipe.rules
        if rule.role is FieldRole.IDENTIFIER
    ]
    _require(len(candidates) == 1, "PROVENANCE_RECID_FIELD")
    return candidates[0]


def _agi_field(compiled: CompiledPufGrowth) -> str:
    _require(
        SOURCE_AGI_FIELD in compiled.recipe.fields,
        "PROVENANCE_AGI_FIELD",
        SOURCE_AGI_FIELD,
    )
    return SOURCE_AGI_FIELD


@dataclass(frozen=True)
class RosterEntry:
    """One raw TY2015 PUF field this lineage already names, and what it is.

    A roster entry states a **role** and a documented **meaning**, never a
    factor. Which series a field should grow by is a producer decision that
    belongs to a reviewed recipe, not to this file.
    """

    field: str
    role: FieldRole
    meaning: str
    citation: str = ""

    def __post_init__(self) -> None:
        _identifier(self.field, "ROSTER_FIELD")
        _require(type(self.role) is FieldRole, "ROSTER_ROLE", self.field)
        _require(bool(self.meaning), "ROSTER_MEANING", self.field)


#: The archived producer commit every historical claim in this module is read
#: from. The retired US data package is named by commit rather than by package
#: name because this repository's launch contract forbids naming it in the live
#: tree (``test_no_incumbent_data_package_references_in_live_tree``); paths in
#: the citations below are relative to that repository's Python package
#: directory. Read a blob with ``git show <commit>:<package_dir>/<path>``.
ARCHIVED_PRODUCER_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"

#: Where the roster's field meanings come from. The archived file is a comment
#: block that itself cites an IRS booklet; that booklet was not read here, so a
#: meaning below is "the label the archived producer attached to this code",
#: not an independently authenticated IRS definition.
ROSTER_CITATION = (
    f"retired US data package @{ARCHIVED_PRODUCER_COMMIT}:"
    "datasets/puf/aggregate_record_totals.yaml"
    " (its own cited source: 2014 PUF General Description Booklet, pp. 33-36,"
    " June 2020)"
)

_STRUCTURAL_CITATION = (
    f"retired US data package @{ARCHIVED_PRODUCER_COMMIT}:"
    "datasets/puf/puf.py preprocess_puf"
)

#: Field code to the meaning the archived codebook records for it. Money only.
_DOCUMENTED_MONEY_MEANINGS: tuple[tuple[str, str], ...] = (
    ("E00100", "AGI"),
    ("E00200", "Wages and salaries"),
    ("E00300", "Taxable interest"),
    ("E00400", "Tax-exempt interest"),
    ("E00600", "Ordinary dividends"),
    ("E00650", "Qualified dividends"),
    ("E00700", "State/local tax refund"),
    ("E00800", "Alimony received"),
    ("E00900", "Business income/loss"),
    ("E01000", "Net capital gain/loss"),
    ("E01100", "Capital gain distributions"),
    ("E01200", "Other gains/losses"),
    ("E01400", "Taxable IRA distributions"),
    ("E01500", "Total pensions and annuities"),
    ("E01700", "Taxable pensions and annuities"),
    ("E02100", "Farm income/loss (Sch F)"),
    ("E02300", "Unemployment compensation"),
    ("E02400", "Social security benefits"),
    ("E03150", "IRA deduction"),
    ("E03210", "Student loan interest"),
    ("E03220", "Educator expenses"),
    ("E03230", "Tuition and fees"),
    ("E03240", "Domestic production deduction"),
    ("E03270", "Self-employed health insurance"),
    ("E03290", "HSA deduction"),
    ("E03300", "Self-employed SEP/SIMPLE"),
    ("E03400", "Early withdrawal penalty"),
    ("E03500", "Alimony paid"),
    ("E07240", "Saver's credit"),
    ("E07260", "Residential energy credit"),
    ("E07300", "Foreign tax credit"),
    ("E07400", "General business credit"),
    ("E07600", "Prior year min tax credit"),
    ("E09600", "AMT"),
    ("E09700", "Recapture of investment credit"),
    ("E09800", "Unreported SE tax"),
    ("E09900", "Penalty on early withdrawal"),
    ("E11200", "Excess FICA withheld"),
    ("E17500", "Medical expenses"),
    ("E18400", "State/local income tax"),
    ("E18500", "Real estate taxes"),
    ("E19200", "Interest paid"),
    ("E19800", "Cash contributions"),
    ("E20100", "Non-cash contributions"),
    ("E20400", "Misc itemized deductions"),
    ("E20500", "Casualty/theft loss"),
    ("E22250", "Short-term capital gain/loss"),
    ("E24515", "Unrecaptured Sec. 1250 gain"),
    ("E24518", "Collectibles gain/loss"),
    ("E25850", "Rental income (Sch E gross)"),
    ("E25860", "Rental loss (Sch E gross)"),
    ("E25940", "Partnership income (gross)"),
    ("E25960", "Partnership loss (gross)"),
    ("E25980", "Partnership net income"),
    ("E26180", "S-corp loss (gross)"),
    ("E26190", "S-corp net income"),
    ("E26270", "Partnership/S-corp net"),
    ("E26390", "Estate/trust income (gross)"),
    ("E26400", "Estate/trust loss (gross)"),
    ("E27200", "Farm rental income (Sch E)"),
    ("E30400", "SE income (taxpayer)"),
    ("E30500", "SE income (spouse)"),
    ("E32800", "Child care credit expenses"),
    ("E58990", "Investment interest (4952)"),
    ("E62900", "AMT foreign tax credit"),
    ("E87521", "American Opportunity Credit"),
    ("P08000", "Other credits"),
    ("P23250", "Long-term capital gain/loss"),
    ("T27800", "Farm income (Sch J)"),
)

_SOI_RENAME_CITATION = (
    f"retired US data package @{ARCHIVED_PRODUCER_COMMIT}:"
    "datasets/puf/uprate_puf.py SOI_TO_PUF_STRAIGHT_RENAMES"
)

_SCREENED_FIELD_CITATION = (
    f"retired US data package @{ARCHIVED_PRODUCER_COMMIT}:"
    "datasets/puf/aggregate_record_utils.py SCREENED_FIELDS"
)

#: The raw TY2015 PUF fields this lineage names, with the role each one plays.
#: A **proposal**, not an authenticated raw-source manifest: it is every money
#: code the archived codebook describes, plus the four structural codes and the
#: one screened code cited separately. It is not every code this repository
#: reads, and not every code the delivered file contains. It declares no factor
#: for any field. Adding or removing a field is a reviewed change, not something
#: a recipe may do implicitly.
PROPOSED_TY2015_FIELD_ROSTER: tuple[RosterEntry, ...] = (
    RosterEntry(
        SOURCE_RECID_FIELD,
        FieldRole.IDENTIFIER,
        "Record id (the archived producer copies it to household_id)",
        _STRUCTURAL_CITATION,
    ),
    RosterEntry(
        DESIGN_WEIGHT_FIELD,
        FieldRole.DESIGN_WEIGHT,
        "Sample weight, stored times 100 (the archived producer divides by 100)",
        _STRUCTURAL_CITATION,
    ),
    RosterEntry(
        "MARS",
        FieldRole.CODE,
        "Filing status: 1 single, 2 joint, 3 separate, 4 head of household; "
        "0 marks an aggregate record",
        _STRUCTURAL_CITATION,
    ),
    RosterEntry(
        "XTOT",
        FieldRole.COUNT,
        "Exemptions count",
        _STRUCTURAL_CITATION,
    ),
    RosterEntry(
        "P22250",
        FieldRole.MONEY,
        "Short-term capital gains",
        _SCREENED_FIELD_CITATION,
    ),
    RosterEntry(
        "E02500",
        FieldRole.MONEY,
        "Taxable social security (the archived SOI rename binds the SOI series"
        " taxable_social_security to this code; the codebook has no entry)",
        _SOI_RENAME_CITATION,
    ),
    *(
        RosterEntry(field, FieldRole.MONEY, meaning, ROSTER_CITATION)
        for field, meaning in _DOCUMENTED_MONEY_MEANINGS
    ),
)

#: Three codes the archived producer names in executable code while neither
#: archived commit says what they mean: ``DSI`` and ``EIC`` appear only in a
#: predictor list, ``E25920`` only as a subtracted term. They are deliberately
#: absent from the roster: a successor must resolve them against real source
#: documentation rather than inherit a guess. This is **not** a complete
#: inventory of undocumented codes -- it is the three this slice tripped over,
#: and closing it is a documented remaining obligation. ``E22250`` is in the roster because the
#: codebook documents it, but no archived code path reads it -- every path uses
#: ``P22250`` -- so whether both codes exist in the delivered file is open.
UNDOCUMENTED_SOURCE_FIELDS: tuple[str, ...] = ("DSI", "E25920", "EIC")

#: The source tax year those roster fields are observed in.
ROSTER_SOURCE_REFERENCE_YEAR = PUF_SOURCE_YEAR


def roster_role(field: str) -> FieldRole:
    """The proposed role of ``field``; refuses a field the roster does not name."""
    for entry in PROPOSED_TY2015_FIELD_ROSTER:
        if entry.field == field:
            return entry.role
    raise PufGrowthRefusalError("FIELD_NOT_IN_ROSTER", field)
