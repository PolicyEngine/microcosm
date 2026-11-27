"""Population targets from a geography ladder, for any country.

A geography ladder gives the population of each cell of each geography level
(a state, a district). A local build calibrates to those totals. This module
makes, for each declared level:

- the role record of each cell's target (:func:`population_rows`), with the
  record fields the country declares;
- the cell's row of the target matrix: each household's size where the
  household is in the cell, so the weighted row is the cell's population. A
  household of size zero supports its cell but adds nothing to the row;
- the cells that no household supports, by name. The caller decides whether
  a smaller surface is acceptable;
- the cell's target spec with its calibration hierarchy
  (:func:`population_target_specs`). A country gives a label lookup for each
  declared label kind, and each level's ``code_name``: what a refusal calls
  a code that has no label.

A country declares its levels as data, and :func:`load_ladder_population`
reads them (microcosm#1728). Nothing here reads country code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = [
    "CodeField",
    "LadderLevel",
    "LadderPopulation",
    "LadderPopulationError",
    "load_ladder_population",
    "population_rows",
    "population_target_specs",
]

#: A label lookup: ``(cell code as named, geography id) -> label or None``.
LabelLookup = Callable[[str, str], "str | None"]

_SECTION_KEYS = frozenset({"name_prefix", "record", "spec", "levels", "rationale"})
_RECORD_KEYS = frozenset({"source", "family"})
_SPEC_KEYS = frozenset(
    {"entity", "family", "source", "provider", "category", "target_label"}
)
_NODE_KEYS = frozenset({"id", "label"})
_LEVEL_KEYS = frozenset(
    {
        "geography",
        "level",
        "code_name",
        "household_column",
        "code_digits",
        "geography_id_prefix",
        "label",
        "record_codes",
    }
)
_CODE_KEYS = frozenset({"field", "divisor", "digits"})


class LadderPopulationError(ValueError):
    """A ladder population declaration or input is not valid."""


@dataclass(frozen=True)
class CodeField:
    """A record field made from the cell code: ``code // divisor``, zero-padded.

    With no divisor the field is None (the level has no such code).
    """

    field: str
    divisor: int | None
    digits: int | None

    def value(self, code: int) -> str | None:
        if self.divisor is None:
            return None
        return f"{int(code) // self.divisor:0{self.digits}d}"


@dataclass(frozen=True)
class LadderLevel:
    """One declared geography level of the ladder."""

    geography: str
    level: str
    #: What a refusal calls the level's code (for example ``US state FIPS``).
    code_name: str
    household_column: str
    code_digits: int
    geography_id_prefix: str
    label: str
    record_codes: tuple[CodeField, ...]


@dataclass(frozen=True)
class LadderPopulation:
    """A country's ladder population targets."""

    name_prefix: str
    record_source: str
    record_family: str
    entity: str
    spec_family: str
    spec_source: str
    provider: tuple[str, str]
    category: tuple[str, str]
    target_label: str
    levels: tuple[LadderLevel, ...]
    rationale: str

    def level(self, geography: str) -> LadderLevel:
        for level in self.levels:
            if level.geography == geography:
                return level
        raise LadderPopulationError(
            f"No ladder population level {geography!r}; declared: "
            f"{[level.geography for level in self.levels]}."
        )

    def name_prefix_of(self, level: LadderLevel) -> str:
        return f"{self.name_prefix}{level.geography}_"

    def target_name(self, geography: str, code: int) -> str:
        level = self.level(geography)
        return f"{self.name_prefix_of(level)}{int(code):0{level.code_digits}d}"

    def level_of_name(self, name: str) -> LadderLevel | None:
        for level in self.levels:
            if name.startswith(self.name_prefix_of(level)):
                return level
        return None


# ---------------------------------------------------------------------------
# Reading the declaration
# ---------------------------------------------------------------------------


def _mapping(value: Any, *, context: str, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LadderPopulationError(f"{context} must be an object.")
    unknown = sorted(set(value) - keys)
    missing = sorted(keys - set(value))
    if unknown or missing:
        raise LadderPopulationError(
            f"{context}: unknown keys {unknown}, missing keys {missing}."
        )
    return value


def _text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise LadderPopulationError(f"{context} must be a non-empty string.")
    return value


def _positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LadderPopulationError(f"{context} must be a positive integer.")
    return value


def _code_field(raw: Any, *, context: str) -> CodeField:
    raw = _mapping(raw, context=context, keys=_CODE_KEYS)
    field = _text(raw["field"], context=f"{context} field")
    if raw["divisor"] is None:
        if raw["digits"] is not None:
            raise LadderPopulationError(
                f"{context}: a field with no divisor has no digits."
            )
        return CodeField(field, None, None)
    return CodeField(
        field,
        _positive_int(raw["divisor"], context=f"{context} divisor"),
        _positive_int(raw["digits"], context=f"{context} digits"),
    )


def _level(raw: Any) -> LadderLevel:
    raw = _mapping(raw, context="ladder level", keys=_LEVEL_KEYS)
    geography = _text(raw["geography"], context="ladder level geography")
    context = f"ladder level {geography!r}"
    codes = raw["record_codes"]
    if not isinstance(codes, list):
        raise LadderPopulationError(f"{context} record_codes must be a list.")
    record_codes = tuple(
        _code_field(item, context=f"{context} record code") for item in codes
    )
    fields = [code.field for code in record_codes]
    if len(set(fields)) != len(fields):
        raise LadderPopulationError(f"{context} record_codes repeat a field.")
    return LadderLevel(
        geography=geography,
        level=_text(raw["level"], context=f"{context} level"),
        code_name=_text(raw["code_name"], context=f"{context} code_name"),
        household_column=_text(
            raw["household_column"], context=f"{context} household_column"
        ),
        code_digits=_positive_int(raw["code_digits"], context=f"{context} code_digits"),
        geography_id_prefix=_text(
            raw["geography_id_prefix"], context=f"{context} geography_id_prefix"
        ),
        label=_text(raw["label"], context=f"{context} label"),
        record_codes=record_codes,
    )


def load_ladder_population(section: Mapping[str, Any]) -> LadderPopulation:
    """Read a country's ``ladder_population`` declaration; refuse anything unknown."""

    raw = _mapping(section, context="ladder_population", keys=_SECTION_KEYS)
    record = _mapping(
        raw["record"], context="ladder_population record", keys=_RECORD_KEYS
    )
    spec = _mapping(raw["spec"], context="ladder_population spec", keys=_SPEC_KEYS)
    provider = _mapping(spec["provider"], context="spec provider", keys=_NODE_KEYS)
    category = _mapping(spec["category"], context="spec category", keys=_NODE_KEYS)
    target_label = _text(spec["target_label"], context="spec target_label")
    if "{label}" not in target_label:
        raise LadderPopulationError("spec target_label must contain {label}.")
    if not isinstance(raw["levels"], list) or not raw["levels"]:
        raise LadderPopulationError(
            "ladder_population levels must be a non-empty list."
        )
    levels = tuple(_level(item) for item in raw["levels"])
    geographies = [level.geography for level in levels]
    if len(set(geographies)) != len(geographies):
        raise LadderPopulationError(f"ladder_population levels repeat: {geographies}.")
    return LadderPopulation(
        name_prefix=_text(raw["name_prefix"], context="ladder_population name_prefix"),
        record_source=_text(record["source"], context="record source"),
        record_family=_text(record["family"], context="record family"),
        entity=_text(spec["entity"], context="spec entity"),
        spec_family=_text(spec["family"], context="spec family"),
        spec_source=_text(spec["source"], context="spec source"),
        provider=(
            _text(provider["id"], context="provider id"),
            _text(provider["label"], context="provider label"),
        ),
        category=(
            _text(category["id"], context="category id"),
            _text(category["label"], context="category label"),
        ),
        target_label=target_label,
        levels=levels,
        rationale=_text(raw["rationale"], context="ladder_population rationale"),
    )


# ---------------------------------------------------------------------------
# Rows and specs
# ---------------------------------------------------------------------------


def population_rows(
    households: pd.DataFrame,
    household_size: np.ndarray,
    ladder_populations: Mapping[str, Mapping[int, float]],
    geographies: Sequence[str],
    declaration: LadderPopulation,
) -> tuple[list[dict], sparse.csr_array, list[str]]:
    """Ladder population marginals as sparse rows (household size x 1[cell]).

    Returns (target records, CSR rows, dropped cell names), in the order of
    ``geographies`` and then of the cell codes. A ladder cell with no
    supporting household is dropped and named.
    """

    size32 = np.asarray(household_size, dtype=np.float32)
    records: list[dict] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    dropped: list[str] = []
    for geography in geographies:
        level = declaration.level(geography)
        codes = pd.to_numeric(households[level.household_column]).to_numpy()
        for value, population in sorted(ladder_populations[geography].items()):
            name = declaration.target_name(geography, value)
            present = np.flatnonzero(codes == value)
            if not len(present):
                dropped.append(name)
                continue
            present = present[size32[present] != 0]
            row = len(records)
            rows.append(np.full(len(present), row, dtype=np.int32))
            cols.append(present.astype(np.int32))
            data.append(size32[present])
            record = {
                "name": name,
                "value": float(population),
                "source": declaration.record_source,
                "family": declaration.record_family,
                "geography_level": level.level,
            }
            for code in level.record_codes:
                record[code.field] = code.value(value)
            records.append(record)
    matrix = sparse.csr_array(
        (
            np.concatenate(data) if data else np.empty(0, np.float32),
            (
                np.concatenate(rows) if rows else np.empty(0, np.int32),
                np.concatenate(cols) if cols else np.empty(0, np.int32),
            ),
        ),
        shape=(len(records), len(households)),
    )
    matrix.sort_indices()
    return records, matrix, dropped


def population_target_specs(
    names: Sequence[str],
    values: Sequence[float],
    declaration: LadderPopulation,
    *,
    period: int,
    labels: Mapping[str, LabelLookup],
) -> tuple:
    """The target spec of each named cell, with its calibration hierarchy.

    A name says its level (``<name_prefix><geography>_<code>``). The level's
    label lookup gives the geography's label from the code and the
    geography id; a code with no label is refused.
    """

    from microcosm.calibrate import TargetSpec
    from microcosm.calibrate.hierarchy import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )

    provider = HierarchyNode(*declaration.provider)
    category = HierarchyCategory(*declaration.category, provider.id)
    specs = []
    for name, value in zip(names, values, strict=True):
        level = declaration.level_of_name(name)
        if level is None:
            raise LadderPopulationError(f"Unknown population target name {name!r}.")
        code = name.removeprefix(declaration.name_prefix_of(level))
        geography_id = f"{level.geography_id_prefix}{code}"
        label = labels[level.label](code, geography_id)
        if label is None:
            raise LadderPopulationError(
                f"Population target {name!r} names no {level.code_name} code."
            )
        geography = HierarchyGeography(geography_id, label, level.level)
        specs.append(
            TargetSpec(
                name=name,
                entity=declaration.entity,
                measure=name,
                value=value,
                period=period,
                source=declaration.spec_source,
                family=declaration.spec_family,
                metadata={"geography_level": geography.level},
                hierarchy=CalibrationHierarchy(
                    provider=provider,
                    category=category,
                    geography=geography,
                    dimensions=(),
                    target=HierarchyNode(
                        name, declaration.target_label.format(label=label)
                    ),
                ),
            )
        )
    return tuple(specs)
