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
  fallback ahead of time rather than fail on an empty cell.
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
