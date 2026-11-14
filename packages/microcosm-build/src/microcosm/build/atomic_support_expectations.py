"""The exact expected value of an area code's value under a conditioned draw cell.

A conditioned draw (:func:`microcosm.build.atomic_geography.assign_atomic`)
draws each household's atomic area from its cell, the support's areas that
match the household on every constraint input it has, with probability
proportional to the weight column the household samples by. A stage that
later prices something by the area drawn, such as a local tax level, may need
that value's expectation over the household's own cell, exactly. Where the
expectation is not zero, the ratio of the drawn area's value to it averages
one when each area is drawn with probability its weight over the cell's mass,
which the draw realises on the 2**53 grid of its uniforms (microcosm#931).
This module computes it, and names no country's sources or columns:

- :func:`draw_cell_expectations` gives each household the mean of
  ``values``, keyed by a support code column, over its cell, weighted by the
  column the household samples by, as an exact
  :class:`fractions.Fraction`.

It takes :func:`~microcosm.build.atomic_support_conditioning.choose_fallback_levels`'
arguments but the ladder, with the household inputs already at each
household's level, and the values to average. Refusals name declared columns
only, never a count, a household, a code or a value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import AtomicSupport

__all__ = ["draw_cell_expectations"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Atomic support expectations: " + message)


def _name(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


def _exact(value: object) -> bool:
    """An int that is not a bool, or a Fraction: never a float."""
    return type(value) is int or isinstance(value, Fraction)


def _observed(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Where a household input is present, and the input as Python objects."""
    present = ~series.isna().to_numpy(dtype=bool)
    codes = series.to_numpy(dtype=object)
    _require(
        all(_name(code) for code in codes[present]),
        "present input codes must be nonempty strings",
    )
    return present, codes


def _cell_code_mass(
    support: AtomicSupport,
    columns: Sequence[str],
    value_column: str,
    weights: Sequence[str],
) -> tuple[pd.MultiIndex | None, np.ndarray, list[str], np.ndarray]:
    """The support's (cell, code) groups, each with its summed weights.

    A cell is a combination of ``columns``' codes (without columns the whole
    support is one cell, ``None``); a group is a cell's areas that share a
    ``value_column`` code. Groups are sorted by cell, then code, so cell
    ``i``'s groups are rows ``starts[i]`` to ``starts[i + 1]`` of the codes
    and of the mass table, whose row ``g`` is group ``g``'s total in each of
    ``weights``, in order. Totals stay exact: the decoder holds each weight
    column's total within the exact sampling range.
    """
    amounts = np.stack(
        [np.asarray(support.arrays[weight], dtype=np.int64) for weight in weights],
        axis=1,
    )
    code_of_row, codes = pd.factorize(np.asarray(support.arrays[value_column]))
    if columns:
        cell_of_row, cells = pd.MultiIndex.from_arrays(
            [support.arrays[column] for column in columns]
        ).factorize()
        cell_count = len(cells)
    else:
        cell_of_row = np.zeros(len(code_of_row), dtype=np.int64)
        cells, cell_count = None, 1
    pair = cell_of_row.astype(np.int64) * len(codes) + code_of_row.astype(np.int64)
    pairs, group_of_row = np.unique(pair, return_inverse=True)
    mass = np.zeros((len(pairs), len(weights)), dtype=np.int64)
    np.add.at(mass, group_of_row, amounts)
    starts = np.searchsorted(
        pairs, np.arange(cell_count + 1, dtype=np.int64) * len(codes)
    )
    group_codes = [str(code) for code in np.asarray(codes)[pairs % len(codes)]]
    return cells, starts, group_codes, mass


def draw_cell_expectations(
    support: AtomicSupport,
    households: pd.DataFrame,
    *,
    constraint_columns: Mapping[str, str],
    weight_by: Mapping[str, object] | None,
    default_weight: str,
    value_column: str,
    values: Mapping[str, int | Fraction],
) -> list[Fraction]:
    """Each household's exact expected value under its draw cell, in row order.

    ``households`` holds each household's inputs at the level it is drawn
    at, every input its level drops already missing, so the shared draw of
    the frame draws each household at its level. The inputs are read as the
    draw reads them: each input a key of ``constraint_columns`` names is,
    where present, an exact-match constraint on the support code column it
    maps to and, where missing, skipped, as the draw skips an input declared
    ``required: false``. The household samples by the weight column
    ``weight_by["map"]`` names for its ``weight_by["input"]`` code where that
    code is present, and by ``default_weight`` where it is missing or
    ``weight_by`` is ``None``. The arguments follow
    :func:`~microcosm.build.atomic_support_conditioning.choose_fallback_levels`:
    ``weight_by`` is ``None`` or ``{"input": column, "map": {code: weight
    column}}``, the draw's rule without its default, and its input may not
    also be a key of ``constraint_columns``.

    The household's cell is the support's areas that match it on every
    constraint it has, and its expectation is
    ``sum(w[a] * values[support[value_column][a]]) / sum(w[a])`` over the
    cell, ``w`` its weight column: the mean the drawn area's value takes
    when each area is drawn with probability ``w[a] / sum(w)``, the law of a
    draw of one stage at the atomic area, or of stages that all sample by
    that column. The draw realises that law on the 2**53 grid of its
    uniforms: its integer inverse CDF places each stage's probabilities
    within 2**-53 of their exact values. The expectation is exact: the
    integer weights are summed by cell and ``value_column`` code, each sum
    is multiplied by its code's int or :class:`~fractions.Fraction` value,
    and the total is divided by the cell's mass once, so the result is a
    ``Fraction``. A code without mass in the cell is not read. Vectorised:
    one grouped mass table per set of constraints present, and one
    expectation per distinct cell and weight column.

    Refused, with texts that name declared columns only, never a count, a
    household, a code or a value: ``constraint_columns``, ``weight_by`` or
    ``default_weight`` as the chooser refuses them; a ``value_column`` that
    is not a support code column; ``values`` that are not ints or Fractions
    (a float or a bool among them) keyed by codes; a frame that lacks or
    repeats an input column; a present input that is not a nonempty string
    without surrounding whitespace, as the draw requires; a present
    ``weight_by`` code the rule does not map, which the draw also refuses; a
    household whose cell has no support row, or no mass in the weight it
    samples by; and a code with mass in a household's cell that has no
    value.
    """
    _require(
        isinstance(support, AtomicSupport), "support must be a decoded atomic support"
    )
    _require(isinstance(households, pd.DataFrame), "households must be a DataFrame")
    columns = support.metadata["columns"]

    def kind(name: object) -> object:
        return columns.get(name, {}).get("kind") if _name(name) else None

    _require(
        isinstance(constraint_columns, Mapping)
        and all(
            _name(column) and matched != "area" and kind(matched) == "code"
            for column, matched in constraint_columns.items()
        )
        and len(set(constraint_columns.values())) == len(constraint_columns),
        "constraint_columns must map inputs to distinct support code columns "
        "other than the area",
    )
    _require(
        kind(default_weight) == "weight",
        "default_weight must be a support weight column",
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
                _name(code) and kind(weight) == "weight"
                for code, weight in weight_by["map"].items()
            )
        ),
        "weight_by must map codes of one input to support weight columns",
    )
    _require(
        weight_by is None or weight_by["input"] not in constraint_columns,
        "the weight_by input must not also be a constraint input",
    )
    _require(kind(value_column) == "code", "value_column must be a support code column")
    _require(
        isinstance(values, Mapping)
        and all(_name(code) and _exact(value) for code, value in values.items()),
        "values must map codes to ints or Fractions",
    )
    inputs = tuple(constraint_columns)
    read = (*inputs, weight_by["input"]) if weight_by is not None else inputs
    missing = sorted(set(read) - set(households.columns))
    _require(not missing, f"households lack input columns {missing}")
    _require(
        not households.columns[households.columns.isin(list(read))].duplicated().any(),
        "households repeat an input column",
    )
    observed = {column: _observed(households[column]) for column in read}

    # Each household's weight column, as a position in ``weights``.
    rows = len(households)
    weights: tuple[str, ...] = (default_weight,)
    weight_of_row = np.zeros(rows, dtype=np.int64)
    if weight_by is not None:
        rule = weight_by["map"]
        weights = tuple(dict.fromkeys((default_weight, *rule.values())))
        present, codes = observed[weight_by["input"]]
        mapped = pd.Index(list(rule)).get_indexer(codes[present])
        _require(
            bool(np.all(mapped >= 0)),
            "a present weight_by code has no declared weight column",
        )
        column_of_code = np.asarray(
            [weights.index(rule[code]) for code in rule], dtype=np.int64
        )
        weight_of_row[present] = column_of_code[mapped]

    # The constraints each household has, as a bit per input.
    pattern = np.zeros(rows, dtype=np.int64)
    for bit, column in enumerate(inputs):
        pattern |= observed[column][0].astype(np.int64) << bit

    result = np.empty(rows, dtype=object)
    for kept_bits in np.unique(pattern).tolist():
        kept = [column for bit, column in enumerate(inputs) if kept_bits >> bit & 1]
        members = np.flatnonzero(pattern == kept_bits)
        cells, starts, group_codes, mass = _cell_code_mass(
            support,
            [constraint_columns[column] for column in kept],
            value_column,
            weights,
        )
        if cells is None:
            cell = np.zeros(len(members), dtype=np.int64)
        else:
            cell = cells.get_indexer(
                pd.MultiIndex.from_arrays(
                    [observed[column][1][members] for column in kept]
                )
            ).astype(np.int64)
        _require(bool(np.all(cell >= 0)), "a household's cell has no support row")
        keys, group_of_member = np.unique(
            cell * len(weights) + weight_of_row[members], return_inverse=True
        )
        expectations = []
        for key in keys.tolist():
            at, weight = divmod(key, len(weights))
            first, end = int(starts[at]), int(starts[at + 1])
            amounts = mass[first:end, weight].tolist()
            total = sum(amounts)
            _require(
                total > 0,
                "a household's cell has no mass in the weight it samples by",
            )
            numerator: int | Fraction = 0
            for code, amount in zip(group_codes[first:end], amounts, strict=True):
                if amount:
                    _require(
                        code in values,
                        "a code with mass in a household's cell has no value",
                    )
                    numerator += amount * values[code]
            expectations.append(Fraction(numerator, total))
        table = np.empty(len(expectations), dtype=object)
        table[:] = expectations
        result[members] = table[group_of_member]
    return result.tolist()
