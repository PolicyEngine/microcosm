"""Extra columns an atomic-area support carries for a conditioned draw.

A country conditions its area draw on what its survey records about the real
address (microcosm#931): an exact-match code such as a deprivation decile, and
a weight chosen per household, such as the area's dwellings in the household's
own council-tax band. The shared operator already takes both
(:mod:`microcosm.build.atomic_geography`: code constraints, and ``weight_by``).
This module builds the columns, and names no country's sources or columns:

- :func:`modal_area_codes` gives each area the code most of its point records
  carry (postcodes, say), breaking a tie by the code its parent area's records
  favour, and giving an area without records its parent's code;
- :func:`apportion_weights` splits each area's count over categories by stated
  shares into exact integers, rounding half to even, and records the rounding;
- :func:`extend_atomic_support` adds declared columns to a support's bytes;
- :func:`zero_mass_cells` lists every constraint cell, at each fallback level,
  where a weight column has no sampling mass, so a draw can choose its
  fallback ahead of time rather than fail on an empty cell;
- :func:`choose_fallback_levels` makes that choice for each household: the
  first level of a declared ladder (:class:`FallbackLevel`) whose cell has
  mass in the weight the household samples by there.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import (
    AtomicSupport,
    decode_atomic_support,
    encode_atomic_support,
)

_U53 = 2**53


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Atomic support conditioning: " + message)


def _codes(array: np.ndarray, name: str) -> np.ndarray:
    _require(
        isinstance(array, np.ndarray) and array.ndim == 1 and array.dtype.kind == "U",
        f"{name} must be a one-dimensional Unicode array",
    )
    return array


@dataclass(frozen=True)
class ModalAreaCodes:
    """Each area's code, the rule that chose it, and the records' spread.

    ``rule`` is ``area`` (the area's own plurality), ``area_tie_parent`` (a
    tie broken by the parent's records), ``area_tie_order`` (a tie the parent
    leaves, broken by the domain order), ``parent`` (no record of its own:
    the parent's plurality) or ``parent_tie_order``. ``spanning`` marks areas
    whose own records carry more than one code.
    """

    codes: np.ndarray
    rule: np.ndarray
    spanning: np.ndarray
    records: int

    def receipt(self) -> dict[str, object]:
        """Counts only: areas by rule, areas spanning codes, records read."""
        rules, counts = np.unique(self.rule, return_counts=True)
        return {
            "areas": int(len(self.codes)),
            "records": int(self.records),
            "areas_by_rule": {
                str(rule): int(count) for rule, count in zip(rules, counts, strict=True)
            },
            "areas_whose_records_span_codes": int(self.spanning.sum()),
        }


def modal_area_codes(
    areas: np.ndarray,
    parents: np.ndarray,
    record_areas: np.ndarray,
    record_codes: np.ndarray,
    *,
    domain: Sequence[str],
) -> ModalAreaCodes:
    """Give each area the code most of its records carry.

    ``areas`` are a support's unique atomic codes and ``parents`` the area
    each nests in (one per area). Each record names its area and its code;
    a record outside ``areas`` or a code outside ``domain`` is refused, so the
    caller drops out-of-scope records explicitly. ``domain`` lists every code
    in tie-break order. An area whose parent has no record at all is refused.
    """
    areas = _codes(areas, "areas")
    parents = _codes(parents, "parents")
    record_areas = _codes(record_areas, "record areas")
    record_codes = _codes(record_codes, "record codes")
    _require(len(parents) == len(areas), "one parent per area")
    _require(len(record_areas) == len(record_codes), "one code per record")
    _require(len(np.unique(areas)) == len(areas), "areas must be unique")
    domain = tuple(domain)
    _require(
        bool(domain)
        and len(set(domain)) == len(domain)
        and all(isinstance(code, str) and code for code in domain),
        "domain must list distinct nonempty codes",
    )
    area_index = pd.Index(areas).get_indexer(record_areas)
    _require(
        bool(np.all(area_index >= 0)), "a record names an area outside the support"
    )
    code_index = pd.Index(domain).get_indexer(record_codes)
    _require(
        bool(np.all(code_index >= 0)), "a record carries a code outside the domain"
    )
    counts = np.zeros((len(areas), len(domain)), dtype=np.int64)
    np.add.at(counts, (area_index, code_index), 1)
    parent_codes, parent_index = np.unique(parents, return_inverse=True)
    parent_counts = np.zeros((len(parent_codes), len(domain)), dtype=np.int64)
    np.add.at(parent_counts, parent_index, counts)
    own = counts.sum(axis=1) > 0
    _require(
        bool(np.all(parent_counts[parent_index[~own]].sum(axis=1) > 0)),
        "an area and its parent have no record",
    )
    chosen = np.empty(len(areas), dtype=np.int64)
    rule = np.empty(len(areas), dtype=object)
    for row in range(len(areas)):
        parent = parent_counts[parent_index[row]]
        if own[row]:
            tied = np.flatnonzero(counts[row] == counts[row].max())
            if len(tied) == 1:
                chosen[row], rule[row] = tied[0], "area"
                continue
            best = tied[parent[tied] == parent[tied].max()]
            chosen[row] = best[0]
            rule[row] = "area_tie_parent" if len(best) == 1 else "area_tie_order"
        else:
            tied = np.flatnonzero(parent == parent.max())
            chosen[row] = tied[0]
            rule[row] = "parent" if len(tied) == 1 else "parent_tie_order"
    return ModalAreaCodes(
        codes=np.asarray(domain, dtype="U")[chosen],
        rule=np.asarray(rule.tolist(), dtype="U"),
        spanning=(counts > 0).sum(axis=1) > 1,
        records=len(record_areas),
    )


def _integers(array: np.ndarray, name: str) -> list[int]:
    _require(
        isinstance(array, np.ndarray) and array.ndim == 1 and array.dtype.kind in "iu",
        f"{name} must be a one-dimensional integer array",
    )
    _require(bool(np.all(array >= 0)), f"{name} must be nonnegative")
    return [int(value) for value in array]


def _round_half_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    if twice > denominator or (twice == denominator and quotient % 2):
        return quotient + 1
    return quotient


def apportion_weights(
    counts: np.ndarray,
    numerators: Mapping[str, np.ndarray],
    denominators: np.ndarray,
    *,
    scale: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Split each area's count over categories into exact integer weights.

    Category ``c`` of area ``i`` weighs ``count[i] × scale × numerators[c][i]
    / denominators[i]``, rounded half to even in exact integer arithmetic.
    Each area's numerators must sum to its denominator, so its weights sum to
    ``count × scale`` up to the rounding, which the receipt records. Every
    weight column's total must stay within the operator's exact sampling range.
    """
    _require(type(scale) is int and scale > 0, "scale must be a positive int")
    _require(
        isinstance(numerators, Mapping) and bool(numerators), "categories required"
    )
    count = _integers(counts, "counts")
    denominator = _integers(denominators, "denominators")
    _require(len(denominator) == len(count), "one denominator per area")
    shares = {name: _integers(array, name) for name, array in numerators.items()}
    _require(
        all(len(values) == len(count) for values in shares.values()),
        "one numerator per area and category",
    )
    for row in range(len(count)):
        stated = sum(values[row] for values in shares.values())
        _require(
            stated == denominator[row] and (denominator[row] > 0 or count[row] == 0),
            "each area's numerators must sum to a positive denominator",
        )
    weights = {}
    for name, values in shares.items():
        column = [
            0
            if count[row] == 0
            else _round_half_even(count[row] * scale * values[row], denominator[row])
            for row in range(len(count))
        ]
        total = sum(column)
        _require(total <= _U53, f"{name} total outside the exact sampling range")
        weights[name] = np.asarray(column, dtype=np.int64)
    deviation = [
        sum(int(weights[name][row]) for name in weights) - count[row] * scale
        for row in range(len(count))
    ]
    receipt = {
        "rounding": "round_half_even",
        "scale": scale,
        "areas_with_rounding": sum(1 for value in deviation if value),
        "largest_area_deviation": max((abs(value) for value in deviation), default=0),
        "total_deviation": sum(deviation),
        "column_totals": {
            name: int(sum(int(v) for v in w)) for name, w in weights.items()
        },
    }
    return weights, receipt


def extend_atomic_support(
    payload: bytes,
    *,
    arrays: Mapping[str, np.ndarray],
    columns: Mapping[str, Mapping[str, str]],
) -> bytes:
    """Add declared code or weight columns to a support's bytes.

    The support keeps its identity, rows and existing columns; a new column
    may not replace one. The result is validated by the shared decoder.
    """
    support = decode_atomic_support(payload)
    _require(set(arrays) == set(columns) and bool(arrays), "declare each new column")
    clash = set(arrays) & set(support.arrays)
    _require(not clash, f"columns already present: {sorted(clash)}")
    rows = len(support.arrays["area"])
    _require(
        all(isinstance(a, np.ndarray) and a.shape == (rows,) for a in arrays.values()),
        "one value per area",
    )
    metadata = {
        **dict(support.metadata),
        "columns": {
            **{name: dict(c) for name, c in support.metadata["columns"].items()},
            **{name: dict(columns[name]) for name in columns},
        },
    }
    merged = {name: np.asarray(array) for name, array in support.arrays.items()}
    merged.update({name: np.asarray(array) for name, array in arrays.items()})
    return encode_atomic_support(metadata, merged)


def zero_mass_cells(
    support: AtomicSupport,
    *,
    domains: Mapping[str, Sequence[str]],
    levels: Sequence[Sequence[str]],
    weights: Sequence[str],
) -> list[dict[str, object]]:
    """Every constraint cell, at each fallback level, with no sampling mass.

    ``levels`` lists the constraint columns of each fallback level, in order;
    ``domains`` gives every value a household may carry for each column, so a
    combination the support never shows is listed too. Each entry names the
    level's columns, a weight column and the cells (value tuples, in domain
    order) whose summed weight is zero.
    """
    columns = support.metadata["columns"]
    _require(
        all(columns.get(w, {}).get("kind") == "weight" for w in weights)
        and bool(weights),
        "weights must be support weight columns",
    )
    table = []
    for level in levels:
        level = tuple(level)
        _require(
            bool(level)
            and all(columns.get(c, {}).get("kind") == "code" for c in level)
            and all(c in domains and len(domains[c]) for c in level),
            "each level needs code columns with declared domains",
        )
        frame = pd.DataFrame(
            {
                **{c: np.asarray(support.arrays[c]) for c in level},
                **{w: np.asarray(support.arrays[w], dtype=np.int64) for w in weights},
            }
        )
        mass = frame.groupby(list(level), sort=False)[list(weights)].sum()
        cells = list(itertools.product(*(tuple(domains[c]) for c in level)))
        for weight in weights:
            empty = []
            for cell in cells:
                key = cell if len(level) > 1 else cell[0]
                if int(mass[weight].get(key, 0)) == 0:
                    empty.append(list(cell))
            table.append({"constraints": list(level), "weight": weight, "cells": empty})
    return table


def _name(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


@dataclass(frozen=True)
class FallbackLevel:
    """One level of a conditioned draw's fallback ladder.

    ``constraints`` names the household inputs the level keeps as exact-match
    constraints, each a key of :func:`choose_fallback_levels`'
    ``constraint_columns``. ``weight_by`` says whether the level samples by
    the weight column chosen for each household (``True``) or by the default
    weight (``False``). A draw at the level reads every input it does not keep
    as missing.
    """

    name: str
    constraints: tuple[str, ...]
    weight_by: bool

    def __post_init__(self) -> None:
        _require(_name(self.name), "a fallback level needs a name")
        _require(
            type(self.constraints) is tuple
            and all(_name(column) for column in self.constraints)
            and len(set(self.constraints)) == len(self.constraints),
            "a fallback level keeps a tuple of distinct input columns",
        )
        _require(type(self.weight_by) is bool, "a fallback level's weight_by is a bool")


def _observed(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Where a household input is present, and the input as Python objects."""
    present = ~series.isna().to_numpy(dtype=bool)
    codes = series.to_numpy(dtype=object)
    _require(
        all(_name(code) for code in codes[present]),
        "present input codes must be nonempty strings",
    )
    return present, codes


def _cell_mass(
    support: AtomicSupport, columns: Sequence[str], weights: Sequence[str]
) -> tuple[pd.MultiIndex | None, np.ndarray]:
    """The support's cells over ``columns``, each with its summed weights.

    Row ``i`` of the mass table is cell ``i``'s total in each of ``weights``,
    in order. Without columns the whole support is one cell (``None``).
    Totals stay exact: the decoder holds each weight column's total within
    the exact sampling range.
    """
    values = np.stack(
        [np.asarray(support.arrays[weight], dtype=np.int64) for weight in weights],
        axis=1,
    )
    if not columns:
        return None, values.sum(axis=0, keepdims=True)
    codes, cells = pd.MultiIndex.from_arrays(
        [support.arrays[column] for column in columns]
    ).factorize()
    mass = np.zeros((len(cells), len(weights)), dtype=np.int64)
    np.add.at(mass, codes, values)
    return cells, mass


def choose_fallback_levels(
    support: AtomicSupport,
    households: pd.DataFrame,
    *,
    constraint_columns: Mapping[str, str],
    weight_by: Mapping[str, object] | None,
    default_weight: str,
    levels: Sequence[FallbackLevel],
) -> np.ndarray:
    """Each household's first fallback level whose support cell has mass.

    The shared draw (:func:`~microcosm.build.atomic_geography.assign_atomic`)
    refuses a household whose constraint cell has no sampling mass in the
    weight column it samples by. A country that conditions its draw declares
    a fallback ladder, ``levels``, in order. A household takes the first level
    whose inputs are present (every constraint the level keeps and, where the
    level samples by the household's own weight, a code ``weight_by`` maps)
    and whose cell, the support's areas matching the household on every
    constraint the level keeps, has positive summed weight in the column the
    household samples by there. A household no level takes is refused; the
    refusal names no count, household or value.

    ``constraint_columns`` maps each household input a level may keep to the
    support code column it must match. ``weight_by`` is ``None`` or
    ``{"input": column, "map": {code: weight column}}``: a level whose
    ``weight_by`` is true samples by the column ``map`` names for the
    household's ``input`` code. Every other level samples by
    ``default_weight``, the draw's ``weight_by`` default or its one
    ``weight``. A present input must be a nonempty string without surrounding
    whitespace, as the draw requires. A draw that keeps a household's level's
    inputs, reads every other input as missing and samples each stage by this
    rule then meets no empty cell.

    Vectorised: one grouped mass table per level, read for the households no
    earlier level took. Returns each household's level name, in row order.
    """
    _require(
        isinstance(support, AtomicSupport), "support must be a decoded atomic support"
    )
    _require(isinstance(households, pd.DataFrame), "households must be a DataFrame")
    columns = support.metadata["columns"]

    def is_weight(name: object) -> bool:
        return _name(name) and columns.get(name, {}).get("kind") == "weight"

    _require(
        isinstance(constraint_columns, Mapping)
        and all(
            _name(column)
            and _name(matched)
            and matched != "area"
            and columns.get(matched, {}).get("kind") == "code"
            for column, matched in constraint_columns.items()
        )
        and len(set(constraint_columns.values())) == len(constraint_columns),
        "constraint_columns must map inputs to distinct support code columns "
        "other than the area",
    )
    _require(
        is_weight(default_weight), "default_weight must be a support weight column"
    )
    _require(
        weight_by is None
        or (
            isinstance(weight_by, Mapping)
            and set(weight_by) == {"input", "map"}
            and _name(weight_by["input"])
            and isinstance(weight_by["map"], Mapping)
            and bool(weight_by["map"])
            and all(
                _name(code) and is_weight(weight)
                for code, weight in weight_by["map"].items()
            )
        ),
        "weight_by must map codes of one input to support weight columns",
    )
    _require(
        isinstance(levels, Sequence)
        and bool(levels)
        and all(isinstance(level, FallbackLevel) for level in levels),
        "levels must be a nonempty sequence of FallbackLevel",
    )
    levels = tuple(levels)
    _require(
        len({level.name for level in levels}) == len(levels),
        "fallback level names must be distinct",
    )
    _require(
        all(set(level.constraints) <= set(constraint_columns) for level in levels),
        "a fallback level keeps an input constraint_columns does not map",
    )
    weighted = any(level.weight_by for level in levels)
    _require(
        weight_by is not None or not weighted,
        "a fallback level samples by weight_by, but no weight_by is given",
    )
    inputs = {column for level in levels for column in level.constraints}
    if weighted:
        inputs.add(weight_by["input"])
    missing = sorted(inputs - set(households.columns))
    _require(not missing, f"households lack input columns {missing}")
    observed = {column: _observed(households[column]) for column in sorted(inputs)}

    # Each household's own weight, as a position in the rule's weight columns
    # (-1 where its code is missing or unmapped).
    rows = len(households)
    rule_weights: tuple[str, ...] = ()
    own_weight = np.full(rows, -1, dtype=np.int64)
    if weighted:
        rule = weight_by["map"]
        rule_weights = tuple(dict.fromkeys(rule.values()))
        present, codes = observed[weight_by["input"]]
        mapped = pd.Index(list(rule)).get_indexer(codes[present])
        column_of_code = np.asarray(
            [rule_weights.index(rule[code]) for code in rule], dtype=np.int64
        )
        holders = np.flatnonzero(present)
        own_weight[holders[mapped >= 0]] = column_of_code[mapped[mapped >= 0]]

    chosen = np.full(rows, -1, dtype=np.int64)
    for position, level in enumerate(levels):
        open_rows = chosen < 0
        for column in level.constraints:
            open_rows &= observed[column][0]
        if level.weight_by:
            open_rows &= own_weight >= 0
        candidates = np.flatnonzero(open_rows)
        if not len(candidates):
            continue
        cells, mass = _cell_mass(
            support,
            [constraint_columns[column] for column in level.constraints],
            rule_weights if level.weight_by else (default_weight,),
        )
        if cells is None:
            cell = np.zeros(len(candidates), dtype=np.intp)
        else:
            cell = cells.get_indexer(
                pd.MultiIndex.from_arrays(
                    [observed[column][1][candidates] for column in level.constraints]
                )
            )
        weight = (
            own_weight[candidates]
            if level.weight_by
            else np.zeros(len(candidates), dtype=np.int64)
        )
        found = cell >= 0
        has_mass = np.zeros(len(candidates), dtype=bool)
        has_mass[found] = mass[cell[found], weight[found]] > 0
        chosen[candidates[has_mass]] = position
    _require(
        bool(np.all(chosen >= 0)),
        "a household has no fallback level whose inputs are present and whose "
        "cell has sampling mass",
    )
    return np.asarray([level.name for level in levels], dtype="U")[chosen]
