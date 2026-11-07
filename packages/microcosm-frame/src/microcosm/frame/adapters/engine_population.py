"""Engine blocks that stand for one pool (microcosm#931).

A rules engine run over part of a population, one engine block per clone in a
dense measure step or one batch per clone in a spine call, works out the
formulas that read beyond their own households over that part rather than
over the pool. This module holds the country-agnostic pieces that make such a
block stand for the pool; each country passes a declaration of which of its
formulas read beyond their households (:class:`PopulationDeclaration`).

Pool shares. A formula ``v = x / sum(x * w)``, optionally times a constant,
evaluated over a block whose engine weights are ``f_b * w`` returns
``x / (f_b * S_b)``, where ``S_b`` is the block's ``sum(x * w)`` at unscaled
weights; over the pool it returns ``x / S``. So the pool's value is the
block's own value times ``c_b = f_b * S_b / S``, one scalar for every row of
the block. An engine that carries each key and the weights to a later year by
one factor per row keeps ``c_b`` at every year, which
:func:`check_uniform_growth` checks. :func:`inject_pool_shares` sets the
block's own value times ``c_b`` as an input, at each period the block
computes, in the block's simulation and its baseline, before any branch is
made and before the guard is armed; :meth:`InjectedShares.apply` does the
same in every other simulation the block builds. :func:`pool_share_rtol`
bounds the result's float32 gap from one engine over the pool, and
:func:`pool_share_atol` the gap of a value that adds share terms to others.

The zero rules. A block whose keys are all zero has ``S_b = 0``; its rows
take zero, as they do in the pool, whatever ``S``. A block with
``S_b = 0`` but nonzero keys, or a pool with ``S = 0`` and a block with
``S_b != 0``, sends the formula down a fallback branch no scalar can
correct, so it is refused (:class:`PoolShareError`).

The guard. A :class:`FormulaRunRecorder` binds itself to each simulation's
``_run_formula`` (a per-instance method, never a class-level patch). A
policyengine-core clone rebinds instance-bound methods to the copy, so
branches made after arming record too, and the recorder sees values the
engine later drops and branches it pops. A walk of the holders' known periods
cross-checks it: a value a simulation holds is judged unless that simulation
ran it after arming. :func:`judge` holds both against the declaration under
a :class:`PassPolicy`: the corrected formulas must be inputs at every
period, the stored ones too, the zero-at-policy ones may run only under
their parameter predicate and only to all-zero values, an inert one may run
only through a sink that comes out all zero, the refused ones never run,
nothing runs in the declaration's position window (by the last year a
run's own period covers), and no structural reform the engine's
constructor applies is switched on. The verdict carries names, periods and
booleans only.

One block's life. :class:`EngineBlock` runs these on a block's own engine,
in order: it validates the declaration, refuses work in the window, runs
checks (a) and (b), injects the shares and arms the recorder; it gives every
simulation the caller builds afterwards the same arrays and recorder, and at
the block's end checks that each still holds the block's keys and weights and
judges them all. A country's caller passes its declaration and a pass.

Batches. :func:`household_batches` splits a frame by a household column
(whole households, groups nested in households) and
:func:`scatter_by_entity_ids` puts each batch's results back in the order one
engine over the frame returns them.

Simulations are duck-typed policyengine-core simulations (``calculate``,
``set_input``, ``get_holder``, ``baseline``, ``branches``, ``branch_name``,
``tax_benefit_system.parameters`` and the instance-bindable
``_run_formula``). No engine is imported when this module loads; a
declaration is checked against its engine only when a caller asks
(:func:`validate_declaration`).
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import operator
import types
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "FLOAT32_UNIT_ROUNDOFF",
    "MEASURE_PASS",
    "SPINE_BATCHED_PASS",
    "AllOf",
    "AnyOf",
    "EngineBlock",
    "FormulaRun",
    "FormulaRunRecorder",
    "HouseholdBatch",
    "InertRule",
    "InjectedShares",
    "ParameterTest",
    "PassPolicy",
    "PoolShare",
    "PoolShareError",
    "PopulationDeclaration",
    "PopulationDeclarationError",
    "PopulationGuardError",
    "PopulationVerdict",
    "PositionWindow",
    "Violation",
    "all_of",
    "any_of",
    "assert_groups_nest_in_households",
    "check_key_sums",
    "check_uniform_growth",
    "derived_watched_periods",
    "factor_spread",
    "float32_sum_roundings",
    "household_batches",
    "inject_pool_shares",
    "judge",
    "pool_share_atol",
    "pool_share_rtol",
    "pool_shares",
    "scatter_by_entity_ids",
    "source_sha256",
    "validate_declaration",
    "validate_pool_shares",
]


class PoolShareError(ValueError):
    """A block's pool-share correction cannot be applied exactly."""


class PopulationDeclarationError(ValueError):
    """An engine no longer matches its declaration of whole-population formulas."""


class PopulationGuardError(RuntimeError):
    """A block or batch engine computed what only the pool defines."""

    def __init__(self, verdict: PopulationVerdict) -> None:
        self.verdict = verdict
        super().__init__(verdict.describe())


# ---------------------------------------------------------------------------
# The float32 bound
# ---------------------------------------------------------------------------

#: The unit roundoff of float32, the precision policyengine-core stores and
#: computes variables in.
FLOAT32_UNIT_ROUNDOFF = 2.0**-24

#: Roundings in one share row besides the two sums and the adds that form a
#: key from several inputs: the row products ``x * w`` in the block's and the
#: pool's sums (2), the divisions by the sums (2), the products by a constant
#: such as an aggregate land value (2), the float32 storage of the key's
#: inputs (at most one rounding of a nonnegative key) and of ``w`` against
#: the float64 frame sums ``c_b`` is formed from, for ``S_b`` and ``S`` (4),
#: the product by ``c_b`` and its float32 cast (1), and float64 slack (2).
_SHARE_ROUNDINGS = 13


@functools.cache
def float32_sum_roundings(n: int) -> int:
    """The most roundings numpy's float32 sum of ``n`` terms chains into one term.

    numpy's ``pairwise_sum`` adds fewer than eight terms in order. Up to 128
    terms it keeps eight accumulators, each starting from one term and adding
    every eighth term after it, combines them in three levels and then adds
    the at most seven terms left over in order: up to 15 + 3 + 7 roundings.
    Above 128 it splits the array at half its length, rounded down to a
    multiple of eight, and adds the two halves' sums: one rounding per level.
    So the leaf alone contributes about 25 roundings whatever ``n`` is, and a
    bound of ``log2(n)`` roundings (which a plain recursive halving would
    give) is not valid.
    """

    n = int(n)
    if n < 8:
        return max(n - 1, 0)
    if n <= 128:
        return (n // 8 - 1) + 3 + (n % 8)
    half = n // 2
    half -= half % 8
    return 1 + max(float32_sum_roundings(half), float32_sum_roundings(n - half))


def pool_share_rtol(
    block_rows: int,
    pool_rows: int,
    *,
    key_inputs: int,
    projection_years: int = 0,
) -> float:
    """A valid bound on ``|v_block * c_b / v_pool - 1|`` for one row.

    For a share formula whose nonnegative key adds ``key_inputs`` inputs (the
    length of the key's ``adds``), evaluated in float32 over a block of
    ``block_rows`` households and over a pool of ``pool_rows``. It counts the
    two sums' roundings (:func:`float32_sum_roundings`) and the row's own
    roundings. It counts the engine's float32 adds that form each row's key
    from its inputs: ``key_inputs - 1`` roundings in every term of the
    block's sum and of the pool's, which the float64 frame sums ``c_b`` is
    formed from do not carry and which need not cancel between the two (so
    ``2 * (key_inputs - 1)``). And at a period the engine projects
    ``projection_years`` years from the dataset, by one factor per row
    (:func:`check_uniform_growth`), it counts one rounding per year in each
    row's key and weight, in the block's and the pool's sums. A first-order
    bound in the unit roundoff. A value that adds share-dependent terms to
    others (a household's taxes, say) is bounded per row by
    :func:`pool_share_atol`, not by this relative to itself.
    """

    if int(key_inputs) < 1:
        raise ValueError("A key adds at least one input.")
    roundings = (
        float32_sum_roundings(block_rows)
        + float32_sum_roundings(pool_rows)
        + _SHARE_ROUNDINGS
        + 2 * (int(key_inputs) - 1)
        + 4 * max(int(projection_years), 0)
    )
    return roundings * FLOAT32_UNIT_ROUNDOFF


def pool_share_atol(
    share_terms: Any,
    magnitude: Any,
    *,
    share_rtol: float,
    additions: int,
) -> np.ndarray:
    """A valid per-row bound on ``|v_block - v_pool|`` for a value that adds
    share-dependent terms to household-local ones.

    ``share_terms`` is each row's sum of the absolute share-dependent terms
    the value adds (a household's business rates and corporate SDLT, say)
    and ``share_rtol`` their bound from :func:`pool_share_rtol`.
    ``magnitude`` bounds the absolute value of every partial sum the value's
    additions form for the row; the sum of its terms' absolute values does.
    ``additions`` counts the additions, the final float32 storage included,
    at which the block's and the pool's partial sums can round apart. The
    household-local terms are equal in both engines, so the gap is the share
    terms' own, ``share_rtol * share_terms``, plus a rounding of up to
    ``u * magnitude`` in each engine at each such addition. Relative to the
    value itself the gap can be large, where the value nets large terms to a
    small one.
    """

    if int(additions) < 0:
        raise ValueError("additions cannot be negative.")
    share = np.abs(np.asarray(share_terms, dtype=np.float64))
    size = np.abs(np.asarray(magnitude, dtype=np.float64))
    return float(share_rtol) * share + 2 * int(additions) * FLOAT32_UNIT_ROUNDOFF * size


# ---------------------------------------------------------------------------
# Periods and values
# ---------------------------------------------------------------------------


def _period_key(period: Any) -> str:
    return str(period)


#: The years an eternity period covers: all of them.
_EVERY_YEAR = (-math.inf, math.inf)


def _period_span(period: Any) -> tuple[float, float] | None:
    """The first and last calendar years ``period`` covers, or None.

    A policyengine-core period gives its own start and stop; its eternity
    unit covers every year. A year as an integer covers itself. A string is
    read only in the forms ``YYYY``, ``YYYY-MM`` and ``YYYY-MM-DD`` (one
    year) and ``ETERNITY``; any other form gives None, which callers treat
    as covering every year.
    """

    if isinstance(period, bool):
        return None
    if isinstance(period, (int, np.integer)):
        return (int(period), int(period))
    if getattr(period, "unit", None) == "eternity":
        return _EVERY_YEAR
    try:
        first, last = period.start.year, period.stop.year
    except Exception:
        first = last = None
    if isinstance(first, (int, float)) and isinstance(last, (int, float)):
        return (first, last)
    text = str(period).strip()
    if text.lower() == "eternity":
        return _EVERY_YEAR
    parts = text.split("-")
    if (
        1 <= len(parts) <= 3
        and len(parts[0]) == 4
        and all(part.isdigit() for part in parts)
        and all(len(part) == 2 for part in parts[1:])
    ):
        return (int(parts[0]), int(parts[0]))
    return None


def _values(result: Any) -> np.ndarray:
    # A pandas or microdf series gives its array, in its own dtype, without
    # microdf's warning that ``.values`` drops the weights.
    return np.asarray(result)


def _all_zero(result: Any) -> bool | None:
    try:
        return bool(np.all(_values(result) == 0))
    except Exception:
        # A result that cannot be read as an array is not known to be zero.
        return None


# ---------------------------------------------------------------------------
# Pool shares
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolShare:
    """One block's correction of one share formula to the pool's denominator.

    ``block_key_sum`` is the block's ``sum(x * w)`` and ``pool_key_sum`` the
    pool's, both at unscaled weights; ``weight_scale`` is the factor ``f_b``
    the block's engine weights carry (1 when they are unscaled).
    """

    formula: str
    key_variable: str
    block_key_sum: float
    pool_key_sum: float
    weight_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("block_key_sum", "pool_key_sum", "weight_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise PoolShareError(f"{self.formula}: {name} must be finite.")
            object.__setattr__(self, name, value)
        if not self.weight_scale > 0.0:
            raise PoolShareError(f"{self.formula}: weight_scale must be positive.")

    @property
    def keys_must_be_zero(self) -> bool:
        """Whether the block's rows take zero: its key sum is zero."""

        return self.block_key_sum == 0.0

    def factor(self) -> float:
        """``c_b = f_b * S_b / S``, under the zero rules.

        A zero block sum gives 0: the block's rows take zero, which is the
        pool's value only if every key in the block is zero, as the injection
        and :func:`check_key_sums` require. A zero pool sum with a nonzero
        block sum is refused.
        """

        if self.block_key_sum == 0.0:
            return 0.0
        if self.pool_key_sum == 0.0:
            raise PoolShareError(
                f"{self.formula}: the pool's {self.key_variable} total is zero "
                "while this block's is not, so the pool's engine takes a "
                "fallback branch no factor reproduces."
            )
        return self.weight_scale * self.block_key_sum / self.pool_key_sum

    def as_receipt(self) -> dict[str, Any]:
        """The correction, as a ratio of block-weighted totals."""

        return {
            "formula": self.formula,
            "key_variable": self.key_variable,
            "correction": self.factor(),
        }


def pool_shares(
    block_key_sums: Mapping[Any, Mapping[str, float]],
    weight_scales: Mapping[Any, float] | None = None,
    *,
    key_variables: Mapping[str, str],
) -> dict[Any, dict[str, PoolShare]]:
    """Every block's :class:`PoolShare` for every corrected formula.

    ``block_key_sums[block][formula]`` is the block's ``sum(x * w)`` at
    unscaled weights; the pool's is their sum, since the blocks partition the
    pool. ``weight_scales[block]`` is the block's engine weight factor
    ``f_b``; without it every block's engine sees unscaled weights.
    ``key_variables`` maps each formula to the key variable it divides by.
    """

    formulas = sorted(key_variables)
    blocks = list(block_key_sums)
    if not blocks:
        raise PoolShareError("pool shares need at least one block.")
    for block in blocks:
        if sorted(block_key_sums[block]) != formulas:
            raise PoolShareError(
                f"block {block!r} carries key sums for "
                f"{sorted(block_key_sums[block])}, not {formulas}."
            )
    if weight_scales is not None and set(weight_scales) != set(blocks):
        raise PoolShareError("every block needs exactly one engine weight factor.")
    pool = {
        formula: math.fsum(float(block_key_sums[block][formula]) for block in blocks)
        for formula in formulas
    }
    return {
        block: {
            formula: PoolShare(
                formula=formula,
                key_variable=key_variables[formula],
                block_key_sum=float(block_key_sums[block][formula]),
                pool_key_sum=pool[formula],
                weight_scale=(
                    1.0 if weight_scales is None else float(weight_scales[block])
                ),
            )
            for formula in formulas
        }
        for block in blocks
    }


def check_key_sums(
    simulation: Any,
    shares: Mapping[str, PoolShare],
    period: Any,
    *,
    weight_variable: str,
    rtol: float = 1e-5,
) -> dict[str, float]:
    """Check (a): the engine's own ``sum(x * w)`` is ``f_b * S_b``.

    Read at ``period`` (the dataset year) from the engine's own key and
    weight arrays, so a key column the engine did not load, a stored total
    or a change to the key's inputs fails here. The tolerance is relative to
    ``sum(|x * w|)``. A share whose block sum is zero requires every key in
    the block to be zero. Returns each formula's relative gap, a ratio of
    block-weighted totals.
    """

    weights = _values(simulation.calculate(weight_variable, period)).astype(np.float64)
    gaps: dict[str, float] = {}
    keys: dict[str, np.ndarray] = {}
    for formula, share in sorted(shares.items()):
        if share.key_variable not in keys:
            keys[share.key_variable] = _values(
                simulation.calculate(share.key_variable, period)
            ).astype(np.float64)
        key = keys[share.key_variable]
        if key.shape != weights.shape:
            raise PoolShareError(
                f"{share.key_variable} and {weight_variable} do not align at {period}."
            )
        if share.keys_must_be_zero:
            if np.any(key != 0):
                raise PoolShareError(
                    f"{formula}: the block's {share.key_variable} total is zero "
                    "but not every key is, so the block's engine takes a "
                    "fallback branch no factor corrects."
                )
            gaps[formula] = 0.0
            continue
        products = key * weights
        scale = float(np.sum(np.abs(products)))
        expected = share.weight_scale * share.block_key_sum
        gap = abs(float(np.sum(products)) - expected) / max(scale, abs(expected))
        if not gap <= rtol:
            raise PoolShareError(
                f"{formula}: the engine's {share.key_variable} total at {period} "
                "is not the block's frame total, so the correction would not "
                "reach the pool's denominator."
            )
        gaps[formula] = gap
    return gaps


def check_uniform_growth(
    simulation: Any,
    names: Iterable[str],
    start: Any,
    end: Any,
    *,
    rtol: float = 1e-5,
) -> dict[str, float | None]:
    """Check (b): the engine carries each of ``names`` by one factor per row.

    From ``start`` (the dataset year) to ``end``: every row that is zero at
    ``start`` stays zero, and every other row grows by the same factor. That
    keeps ``S_b / S`` at ``end`` what it is at ``start``, so a ``c_b`` from
    dataset-year totals holds there. Pass each key itself, not only its
    parts, and the weight variable. Returns each name's factor, the ratio of
    its absolute totals (None where no row carries it).
    """

    factors: dict[str, float | None] = {}
    for name in names:
        then = _values(simulation.calculate(name, start)).astype(np.float64)
        now = _values(simulation.calculate(name, end)).astype(np.float64)
        if then.shape != now.shape:
            raise PoolShareError(f"{name} changes shape between {start} and {end}.")
        carried = then != 0
        if np.any(now[~carried] != 0):
            raise PoolShareError(
                f"{name} is zero at {start} on rows where it is not at {end}, so "
                "the engine does not carry it by one factor per row."
            )
        if not carried.any():
            factors[name] = None
            continue
        factor = float(np.sum(np.abs(now[carried])) / np.sum(np.abs(then[carried])))
        expected = factor * then[carried]
        if np.any(np.abs(now[carried] - expected) > rtol * np.abs(expected)):
            raise PoolShareError(
                f"The engine does not carry {name} from {start} to {end} by one "
                "factor per row, so a dataset-year correction does not hold there."
            )
        factors[name] = factor
    return factors


def factor_spread(
    factors_by_block: Mapping[Any, Mapping[str, float | None]],
) -> dict[str, float]:
    """Each name's spread of factors across blocks: ``(max - min) / |mean|``.

    Blocks whose factor is None (no row carries the name) are left out. The
    factors are the engine's parameter-driven growth, equal in every block
    up to float32 rounding.
    """

    names = sorted({name for factors in factors_by_block.values() for name in factors})
    spread: dict[str, float] = {}
    for name in names:
        values = [
            float(factors[name])
            for factors in factors_by_block.values()
            if factors.get(name) is not None
        ]
        if len(values) < 2:
            spread[name] = 0.0
            continue
        centre = abs(math.fsum(values) / len(values))
        spread[name] = (max(values) - min(values)) / centre if centre else math.inf
    return spread


class InjectedShares:
    """The pool's share arrays one block's simulations take, at each period."""

    def __init__(
        self,
        *,
        shares: Mapping[str, PoolShare],
        periods: Sequence[Any],
        own: Mapping[tuple[str, str], np.ndarray],
        corrected: Mapping[tuple[str, str], np.ndarray],
        references: Mapping[tuple[str, str], np.ndarray],
    ) -> None:
        self._shares = dict(shares)
        self._periods = tuple(periods)
        self._own = dict(own)
        self._corrected = dict(corrected)
        self._references = dict(references)

    @property
    def periods(self) -> tuple[str, ...]:
        return tuple(_period_key(period) for period in self._periods)

    @property
    def factors(self) -> dict[str, float]:
        return {formula: share.factor() for formula, share in self._shares.items()}

    def apply(self, simulation: Any) -> None:
        """Set the same arrays into ``simulation`` and its baseline.

        For another simulation the block builds (an annual-income or
        counterfactual sibling): its own share values must equal, bitwise,
        those the arrays were made from, so the same correction is the
        pool's value there too. As with :func:`inject_pool_shares`, apply
        before arming a recorder on the sibling and before it has branches.
        """

        for target in _with_baseline(simulation):
            _refuse_unready(target)
            _refuse_injected(target, self._shares, self._periods)
            own = _own_values(target, self._shares, self._periods)
            for key, values in own.items():
                if not np.array_equal(values, self._own[key]):
                    formula, period = key
                    raise PoolShareError(
                        f"{formula} at {period} differs in a simulation the "
                        "block builds, so the block's correction is not its "
                        "pool value there."
                    )
        for target in _with_baseline(simulation):
            _set_corrected(target, self._corrected, self._periods)

    def matches(self, simulation: Any) -> bool:
        """Whether ``simulation`` and its baseline still hold the block's keys
        and weights at every period, bitwise (checked when the block ends)."""

        for target in _with_baseline(simulation):
            for (name, period), reference in self._references.items():
                period_value = self._period_value(period)
                values = _values(target.calculate(name, period_value))
                if not np.array_equal(values, reference):
                    return False
        return True

    def as_receipt(self) -> dict[str, Any]:
        return {
            "periods": list(self.periods),
            "corrections": {
                formula: factor for formula, factor in sorted(self.factors.items())
            },
        }

    def _period_value(self, key: str) -> Any:
        for period in self._periods:
            if _period_key(period) == key:
                return period
        raise KeyError(key)


def _with_baseline(simulation: Any) -> list[Any]:
    targets = [simulation]
    baseline = getattr(simulation, "baseline", None)
    if baseline is not None and baseline is not simulation:
        targets.append(baseline)
    return targets


def _armed_by_a_recorder(simulation: Any) -> bool:
    bound = getattr(simulation, "__dict__", {}).get("_run_formula")
    function = getattr(bound, "__func__", None)
    owner = getattr(function, "_microcosm_recorder", None)
    recorder = owner() if owner is not None else None
    return recorder is not None and recorder._recording


def _refuse_unready(simulation: Any) -> None:
    """Refuse to set shares into a simulation with live branches, which keep
    their own values and would go on computing the block's own shares, or
    one a recorder is armed on, which would record the reads of its own
    values as runs."""

    branches = getattr(simulation, "branches", None)
    if isinstance(branches, Mapping) and branches:
        raise PoolShareError(
            "The simulation has live branches, which would keep computing "
            "the block's own shares; set the pool's shares before any branch "
            "is made."
        )
    if _armed_by_a_recorder(simulation):
        raise PoolShareError(
            "A recorder is armed on the simulation; set the pool's shares "
            "before arming it."
        )


def _input_keys(simulation: Any, name: str) -> set[str]:
    holder = simulation.get_holder(name)
    getter = getattr(holder, "get_input_periods", None)
    if getter is None:
        return set()
    branch = getattr(simulation, "branch_name", "default")
    return {_period_key(period) for period in getter(branch)}


def _refuse_injected(
    simulation: Any, shares: Mapping[str, PoolShare], periods: Sequence[Any]
) -> None:
    for formula in shares:
        inputs = _input_keys(simulation, formula)
        for period in periods:
            if _period_key(period) in inputs:
                raise PoolShareError(
                    f"{formula} is already an input at {period}; its own value "
                    "cannot be read, and correcting it twice would compound."
                )


def _own_values(
    simulation: Any, shares: Mapping[str, PoolShare], periods: Sequence[Any]
) -> dict[tuple[str, str], np.ndarray]:
    own: dict[tuple[str, str], np.ndarray] = {}
    for formula in sorted(shares):
        for period in periods:
            values = np.array(_values(simulation.calculate(formula, period)))
            if not np.all(np.isfinite(values)):
                raise PoolShareError(f"{formula} at {period} is not finite.")
            own[(formula, _period_key(period))] = values
    return own


def _set_corrected(
    simulation: Any,
    corrected: Mapping[tuple[str, str], np.ndarray],
    periods: Sequence[Any],
) -> None:
    by_key = {_period_key(period): period for period in periods}
    for (formula, key), values in corrected.items():
        simulation.set_input(formula, by_key[key], values.copy())


def inject_pool_shares(
    simulation: Any,
    shares: Mapping[str, PoolShare],
    periods: Iterable[Any],
    *,
    weight_variable: str,
) -> InjectedShares:
    """Set each share formula to its own value times ``c_b`` at every period.

    Into ``simulation`` and its baseline, before anything reads the shares,
    before any branch is made and before a recorder is armed (each is
    refused). The own values are the engine's (read first, for every formula
    and period, before any is set), so no row alignment is involved; a block
    whose key sum is zero takes zeros, and every key in it must be zero.
    The returned :class:`InjectedShares` sets the same arrays into the other
    simulations the block builds.
    """

    periods = tuple({_period_key(period): period for period in periods}.values())
    if not periods:
        raise PoolShareError("pool shares need at least one period.")
    targets = _with_baseline(simulation)
    for target in targets:
        _refuse_unready(target)
        _refuse_injected(target, shares, periods)
    own = _own_values(simulation, shares, periods)
    for target in targets[1:]:
        for key, values in _own_values(target, shares, periods).items():
            if not np.array_equal(values, own[key]):
                raise PoolShareError(
                    f"{key[0]} at {key[1]} differs between the simulation and "
                    "its baseline, so one correction cannot serve both."
                )
    references: dict[tuple[str, str], np.ndarray] = {}
    for period in periods:
        references[(weight_variable, _period_key(period))] = np.array(
            _values(simulation.calculate(weight_variable, period))
        )
        for share in shares.values():
            key = (share.key_variable, _period_key(period))
            if key not in references:
                references[key] = np.array(
                    _values(simulation.calculate(share.key_variable, period))
                )
            if share.keys_must_be_zero and np.any(references[key] != 0):
                raise PoolShareError(
                    f"{share.formula}: the block's {share.key_variable} total is "
                    f"zero but not every key is at {period}, so the block's "
                    "engine takes a fallback branch no factor corrects."
                )
    corrected: dict[tuple[str, str], np.ndarray] = {}
    for (formula, key), values in own.items():
        share = shares[formula]
        if share.keys_must_be_zero:
            corrected[(formula, key)] = np.zeros_like(values)
        else:
            corrected[(formula, key)] = (
                values.astype(np.float64) * share.factor()
            ).astype(values.dtype)
    for target in targets:
        _set_corrected(target, corrected, periods)
    return InjectedShares(
        shares=shares,
        periods=periods,
        own=own,
        corrected=corrected,
        references=references,
    )


# ---------------------------------------------------------------------------
# Parameter predicates
# ---------------------------------------------------------------------------

_OPERATORS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def _parameter_value(node: Any, path: str) -> Any:
    for part in path.split("."):
        try:
            node = getattr(node, part)
        except Exception:
            try:
                node = node[part]
            except Exception as error:
                raise AttributeError(f"No parameter {path}.") from error
    return node


@dataclass(frozen=True)
class ParameterTest:
    """A test of one parameter at a period: ``path op value``.

    ``op`` is one of ``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``, or
    ``eq_path``, which compares the parameter with the one at the path given
    as ``value`` (a reform's parameter with its baseline's, say).
    """

    path: str
    op: str
    value: bool | int | float | str | None = None

    def __post_init__(self) -> None:
        if self.op not in (*_OPERATORS, "eq_path"):
            raise ValueError(f"Unknown parameter test operator {self.op!r}.")
        if self.op == "eq_path" and not isinstance(self.value, str):
            raise ValueError("An eq_path test compares with a parameter path.")

    def holds(self, parameters: Any) -> bool:
        left = _parameter_value(parameters, self.path)
        if self.op == "eq_path":
            return bool(left == _parameter_value(parameters, str(self.value)))
        return bool(_OPERATORS[self.op](left, self.value))

    def paths(self) -> tuple[str, ...]:
        return (self.path, str(self.value)) if self.op == "eq_path" else (self.path,)

    def to_json(self) -> dict[str, Any]:
        return {"parameter": self.path, "op": self.op, "value": self.value}


@dataclass(frozen=True)
class AllOf:
    """Every test holds."""

    tests: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tests", tuple(self.tests))
        if not self.tests:
            raise ValueError("A predicate needs at least one test.")

    def holds(self, parameters: Any) -> bool:
        return all(test.holds(parameters) for test in self.tests)

    def paths(self) -> tuple[str, ...]:
        return tuple(path for test in self.tests for path in test.paths())

    def to_json(self) -> dict[str, Any]:
        return {"all_of": [test.to_json() for test in self.tests]}


@dataclass(frozen=True)
class AnyOf:
    """At least one test holds."""

    tests: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tests", tuple(self.tests))
        if not self.tests:
            raise ValueError("A predicate needs at least one test.")

    def holds(self, parameters: Any) -> bool:
        return any(test.holds(parameters) for test in self.tests)

    def paths(self) -> tuple[str, ...]:
        return tuple(path for test in self.tests for path in test.paths())

    def to_json(self) -> dict[str, Any]:
        return {"any_of": [test.to_json() for test in self.tests]}


#: A predicate over a simulation's parameters at a period.
Predicate = ParameterTest | AllOf | AnyOf


def all_of(*tests: Predicate) -> AllOf:
    return AllOf(tests)


def any_of(*tests: Predicate) -> AnyOf:
    return AnyOf(tests)


def _predicate_holds_in(simulation: Any, predicate: Predicate, period: Any) -> bool:
    try:
        parameters = simulation.tax_benefit_system.parameters(period)
        return bool(predicate.holds(parameters))
    except Exception:
        # A predicate whose parameters cannot be read does not hold.
        return False


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InertRule:
    """When a formula that reads beyond its households cannot reach a value.

    The formula's value reaches the population's totals only through
    ``sink``, a difference that is zero everywhere while ``predicate`` holds
    (a policy equal to its baseline). The formula may run in a block only if
    the sink ran wherever it did and came out all zero, the predicate held,
    none of ``carriers`` (which read a level the formula feeds) ran, and
    neither the formula, its ``levels`` nor a carrier was requested.
    """

    sink: str
    levels: tuple[str, ...]
    carriers: tuple[str, ...]
    predicate: Predicate

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "carriers", tuple(self.carriers))

    def unrequestable(self, name: str) -> frozenset[str]:
        return frozenset({name, *self.levels, *self.carriers})

    def to_json(self) -> dict[str, Any]:
        return {
            "sink": self.sink,
            "levels": sorted(self.levels),
            "carriers": sorted(self.carriers),
            "predicate": self.predicate.to_json(),
        }


@dataclass(frozen=True)
class PositionWindow:
    """Years in which an engine's values depend on a record's position.

    Any block or batch work in a year from ``first_year`` on is refused:
    a requested or measured period there, or any formula run there after
    the guard is armed. ``source`` names the engine's own statement of the
    years (``module:attribute``, a ``range`` starting at ``first_year``).
    """

    first_year: int
    source: str
    reason: str

    def contains(self, period: Any) -> bool:
        """Whether ``period`` reaches the window: its last year is
        ``first_year`` or later. An eternity period covers every year, so it
        does; so does a period whose years cannot be read."""

        return self.reaches(_period_span(period))

    def reaches(self, span: tuple[float, float] | None) -> bool:
        """Whether years ``(first, last)`` reach the window; None, years not
        known, does."""

        return span is None or span[1] >= self.first_year

    def to_json(self) -> dict[str, Any]:
        return {"first_year": self.first_year, "source": self.source}


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(sorted(dict(value).items())))


def _names(value: Iterable[str]) -> tuple[str, ...]:
    names = tuple(value)
    if len(set(names)) != len(names):
        raise ValueError(f"Names repeat in {names}.")
    return names


@dataclass(frozen=True, eq=False)
class PopulationDeclaration:
    """An engine's formulas that read beyond their own households, by class.

    - ``corrected``: share formulas (to the key variable each divides by),
      set to the pool's value in every block (:func:`inject_pool_shares`);
      ``key_adds`` names each key's inputs, which the frame sums.
    - ``stored``: formulas the data supplies as inputs at every period.
    - ``zero_at_policy``: formulas that are zero while their predicate holds.
    - ``inert``: stored formulas that may instead run under an
      :class:`InertRule`, in a pass that allows it.
    - ``refused``: formulas no block may run (ranks and medians).

    ``list_source`` names the engine's own list (``module:attribute``), which
    the classes must cover exactly; ``observed`` names formulas recorded for
    receipts but not judged; ``constructor_factories`` names the structural
    reform factories the engine's constructor applies, which must each return
    None; ``digests`` pins the sources the classification rests on
    (:func:`source_sha256`). ``reviewed_version`` is the engine release the
    classification was reviewed against; validation rests on the digests.
    """

    engine: str
    distribution: str
    reviewed_version: str
    list_source: str
    weight_variable: str
    corrected: Mapping[str, str]
    key_adds: Mapping[str, tuple[str, ...]]
    stored: tuple[str, ...]
    zero_at_policy: Mapping[str, Predicate]
    inert: Mapping[str, InertRule]
    refused: tuple[str, ...]
    window: PositionWindow
    observed: tuple[str, ...] = ()
    constructor_factories: tuple[str, ...] = ()
    digests: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "corrected", _frozen_mapping(self.corrected))
        object.__setattr__(
            self,
            "key_adds",
            _frozen_mapping(
                {key: tuple(adds) for key, adds in dict(self.key_adds).items()}
            ),
        )
        object.__setattr__(self, "zero_at_policy", _frozen_mapping(self.zero_at_policy))
        object.__setattr__(self, "inert", _frozen_mapping(self.inert))
        object.__setattr__(self, "digests", _frozen_mapping(self.digests))
        for name in ("stored", "refused", "observed", "constructor_factories"):
            object.__setattr__(self, name, _names(getattr(self, name)))
        classes = {
            "corrected": set(self.corrected),
            "stored": set(self.stored),
            "zero_at_policy": set(self.zero_at_policy),
            "refused": set(self.refused),
        }
        for first, first_names in classes.items():
            for second, second_names in classes.items():
                if first < second and first_names & second_names:
                    raise ValueError(
                        f"{sorted(first_names & second_names)} are both "
                        f"{first} and {second}."
                    )
        if not set(self.inert) <= set(self.stored):
            raise ValueError("An inert formula must also be declared stored.")
        missing_keys = set(self.corrected.values()) - set(self.key_adds)
        if missing_keys:
            raise ValueError(f"Keys {sorted(missing_keys)} name no inputs.")

    def classified(self) -> frozenset[str]:
        return frozenset(
            {
                *self.corrected,
                *self.stored,
                *self.zero_at_policy,
                *self.inert,
                *self.refused,
            }
        )

    def watched(self) -> frozenset[str]:
        """Every formula the guard records: the classified ones, the inert
        rules' sinks, levels and carriers, and the observed ones."""

        names = set(self.classified()) | set(self.observed)
        for rule in self.inert.values():
            names |= {rule.sink, *rule.levels, *rule.carriers}
        return frozenset(names)

    def correction_reads(self) -> frozenset[str]:
        """What a pool-share correction reads: the corrected formulas, their
        keys, the keys' inputs and the weight variable. No simulation a block
        builds may set any of them (:meth:`EngineBlock.refuse_substitution`)."""

        return frozenset(
            {
                *self.corrected,
                *self.corrected.values(),
                *(add for adds in self.key_adds.values() for add in adds),
                self.weight_variable,
            }
        )

    def run_predicates(self) -> dict[str, Predicate]:
        """The predicate checked at every run of each formula that has one."""

        predicates: dict[str, Predicate] = dict(self.zero_at_policy)
        for name, rule in self.inert.items():
            predicates[name] = rule.predicate
            predicates[rule.sink] = rule.predicate
        return predicates

    def to_json(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "distribution": self.distribution,
            "reviewed_version": self.reviewed_version,
            "list_source": self.list_source,
            "weight_variable": self.weight_variable,
            "corrected": dict(self.corrected),
            "key_adds": {key: list(adds) for key, adds in self.key_adds.items()},
            "stored": sorted(self.stored),
            "zero_at_policy": {
                name: predicate.to_json()
                for name, predicate in self.zero_at_policy.items()
            },
            "inert": {name: rule.to_json() for name, rule in self.inert.items()},
            "refused": sorted(self.refused),
            "observed": sorted(self.observed),
            "window": self.window.to_json(),
            "constructor_factories": sorted(self.constructor_factories),
            "digests": dict(self.digests),
        }

    @property
    def sha256(self) -> str:
        text = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_reference(reference: str) -> Any:
    module_name, _, qualified = reference.partition(":")
    if not qualified:
        raise ValueError(f"{reference!r} is not of the form 'module:attribute'.")
    value: Any = importlib.import_module(module_name)
    for part in qualified.split("."):
        value = getattr(value, part)
    return value


def source_sha256(reference: str) -> str:
    """The sha256 of the source a declaration pins.

    ``package/relative/path.py`` names a file inside an installed package (its
    bytes are hashed); ``module:qualified.name`` names a function or class
    (the text ``inspect.getsource`` returns is hashed).
    """

    if ":" in reference:
        source = inspect.getsource(_resolve_reference(reference))
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    package, _, relative = reference.partition("/")
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(f"No installed package {package!r}.")
    for location in spec.submodule_search_locations:
        path = Path(location) / relative
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    raise FileNotFoundError(f"{reference} is not in the installed package.")


def _is_formula(variable: Any) -> bool:
    return bool(getattr(variable, "formulas", None))


def _list_of(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def validate_declaration(declaration: PopulationDeclaration, system: Any) -> None:
    """Fail unless ``system`` (the engine's tax-benefit system) still matches.

    Its own list equals the declared classes; every named variable exists;
    every share formula is a formula over its key, every key a sum of plain
    inputs of the key's entity exactly as declared; the pinned sources are
    unchanged; the window's source starts at the window; every constructor
    factory resolves; every predicate's parameters exist. Callers validate
    when they first need the declaration, never at import.
    """

    problems: list[str] = []
    try:
        listed = frozenset(_resolve_reference(declaration.list_source))
    except Exception as error:
        problems.append(f"{declaration.list_source} cannot be read ({error}).")
    else:
        classified = declaration.classified()
        if listed != classified:
            problems.append(
                f"{declaration.list_source} lists {sorted(listed - classified)} "
                "the declaration does not classify, and the declaration "
                f"classifies {sorted(classified - listed)} it does not list."
            )
    variables = system.variables
    needed = (
        set(declaration.watched())
        | set(declaration.corrected.values())
        | {add for adds in declaration.key_adds.values() for add in adds}
        | {declaration.weight_variable}
    )
    absent = sorted(name for name in needed if name not in variables)
    if absent:
        problems.append(f"The engine has no variables {absent}.")
    weight = variables.get(declaration.weight_variable)
    for formula, key in declaration.corrected.items():
        formula_variable = variables.get(formula)
        key_variable = variables.get(key)
        if formula_variable is None or key_variable is None:
            continue
        if not _is_formula(formula_variable):
            problems.append(f"{formula} is no longer a formula.")
        entity = key_variable.entity.key
        if formula_variable.entity.key != entity or (
            weight is not None and weight.entity.key != entity
        ):
            problems.append(
                f"{formula}, {key} and {declaration.weight_variable} no longer "
                "share an entity."
            )
        if _is_formula(key_variable) or _list_of(
            getattr(key_variable, "subtracts", None)
        ):
            problems.append(f"{key} is no longer a plain sum of its inputs.")
        adds = _list_of(getattr(key_variable, "adds", None))
        if adds != declaration.key_adds[key]:
            problems.append(
                f"{key} adds {list(adds)}, not the declared "
                f"{list(declaration.key_adds[key])}."
            )
        for add in adds:
            leaf = variables.get(add)
            if leaf is None:
                continue
            if (
                _is_formula(leaf)
                or _list_of(getattr(leaf, "adds", None))
                or _list_of(getattr(leaf, "subtracts", None))
                or leaf.entity.key != entity
            ):
                problems.append(f"{add} is no longer a plain {entity} input.")
    for reference, expected in declaration.digests.items():
        try:
            actual = source_sha256(reference)
        except Exception as error:
            problems.append(f"{reference} cannot be read ({error}).")
            continue
        if actual != expected:
            problems.append(f"{reference} has changed since it was reviewed.")
    window = declaration.window
    try:
        years = _resolve_reference(window.source)
    except Exception as error:
        problems.append(f"{window.source} cannot be read ({error}).")
    else:
        if (
            not isinstance(years, range)
            or years.step != 1
            or years.start != window.first_year
        ):
            problems.append(
                f"{window.source} no longer starts the window at {window.first_year}."
            )
    for reference in declaration.constructor_factories:
        try:
            factory = _resolve_reference(reference)
        except Exception as error:
            problems.append(f"{reference} cannot be read ({error}).")
            continue
        if not callable(factory):
            problems.append(f"{reference} is not callable.")
    for name, predicate in sorted(declaration.run_predicates().items()):
        for path in predicate.paths():
            try:
                _parameter_value(system.parameters, path)
            except AttributeError:
                problems.append(f"{name}: the engine has no parameter {path}.")
    if problems:
        raise PopulationDeclarationError(
            f"{declaration.engine}'s whole-population declaration no longer "
            "matches the installed engine:\n  - " + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Passes and verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassPolicy:
    """How a pass treats the guard.

    ``inert_allowed``: whether an inert rule may stand in for a stored
    input. ``on_violation``: ``record`` returns a verdict that is not clean
    (the caller marks its result inexact); ``raise`` raises
    :class:`PopulationGuardError`.
    """

    name: str
    inert_allowed: bool
    on_violation: str

    def __post_init__(self) -> None:
        if self.on_violation not in ("record", "raise"):
            raise ValueError("on_violation is 'record' or 'raise'.")


#: The dense measure step: attendance and the other stored formulas must be
#: inputs, and a violation is recorded, leaving the blocks inexact (María's
#: ruling of 2026-11-06, microcosm#931).
MEASURE_PASS = PassPolicy("measure", inert_allowed=False, on_violation="record")
#: A spine engine call batched by clone: an inert formula may run under its
#: rule, and a violation raises (the same ruling).
SPINE_BATCHED_PASS = PassPolicy(
    "spine_batched", inert_allowed=True, on_violation="raise"
)


@dataclass(frozen=True)
class Violation:
    """One thing a block or batch did that only the pool defines."""

    kind: str
    name: str | None = None
    period: str | None = None

    def as_receipt(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "period": self.period}

    def describe(self) -> str:
        where = "" if self.period is None else f"@{self.period}"
        return f"{self.kind} {self.name or ''}{where}".strip()


#: The most violations a verdict lists; the rest are counted out as truncated.
_VIOLATION_LIMIT = 50


@dataclass(frozen=True)
class PopulationVerdict:
    """What one block or batch's simulations did, judged. Names, periods and
    booleans only: no value."""

    pass_name: str
    violations: tuple[Violation, ...]
    truncated: bool
    classes: Mapping[str, tuple[str, ...]]
    window_first_year: int
    declaration_sha256: str
    engine: str
    engine_version: str | None

    @property
    def clean(self) -> bool:
        return not self.violations

    def as_receipt(self) -> dict[str, Any]:
        window = [v for v in self.violations if v.kind.startswith("window_")]
        refused = sorted(
            {
                v.name
                for v in self.violations
                if v.kind.startswith("refused_") and v.name
            }
        )
        return {
            "pass": self.pass_name,
            "clean": self.clean,
            "declaration_sha256": self.declaration_sha256,
            "engine": {"package": self.engine, "version": self.engine_version},
            "classes": {name: list(names) for name, names in self.classes.items()},
            "refused": refused,
            "window": {
                "first_year": self.window_first_year,
                "violations": [violation.as_receipt() for violation in window],
            },
            "violations": [violation.as_receipt() for violation in self.violations],
            "truncated": self.truncated,
        }

    def describe(self) -> str:
        if self.clean:
            return f"pass {self.pass_name!r}: clean"
        listed = "; ".join(violation.describe() for violation in self.violations)
        more = " (and more)" if self.truncated else ""
        return (
            f"pass {self.pass_name!r}: a block or batch engine computed what "
            f"only the pool defines: {listed}{more}."
        )


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------

_TOKEN_ATTRIBUTE = "_microcosm_formula_run_token"


@dataclass(frozen=True)
class FormulaRun:
    """One run of a watched formula: in which simulation (the recorder's
    token for it), at which period, whether it produced a value, whether
    that value was all zero, and whether its predicate held, read at the run
    from that simulation's own parameters."""

    simulation: int
    name: str
    period: str
    produced: bool
    all_zero: bool | None
    predicate_holds: bool | None


class FormulaRunRecorder:
    """Records the formulas simulations run once it is armed on them.

    :meth:`arm` binds a recording ``_run_formula`` to a simulation, its
    baseline and its live branches, as instance methods. A policyengine-core
    clone rebinds instance-bound methods to the copy, so any branch made
    afterwards (``get_branch``, household subsets) records too. Watched
    formulas are recorded in full; every other run only by period, for the
    position window. Each simulation is told apart by a token checked
    against a weak reference, so a copy that inherits its source's token, or
    a simulation that reuses a collected one's id, gets its own.
    """

    def __init__(
        self,
        watched: Iterable[str],
        *,
        predicates: Mapping[str, Predicate] | None = None,
    ) -> None:
        self.watched = frozenset(watched)
        self._predicates = dict(predicates or {})
        unknown = set(self._predicates) - self.watched
        if unknown:
            raise ValueError(f"Predicates for unwatched formulas {sorted(unknown)}.")
        self._runs: list[FormulaRun] = []
        self._periods: dict[str, set[str]] = {}
        self._spans: dict[str, tuple[float, float] | None] = {}
        self._simulations: dict[int, weakref.ref] = {}
        self._next_token = 0
        self._recording = False
        recorder = self

        def run_formula(simulation, variable, population, period):
            result = type(simulation)._run_formula(
                simulation, variable, population, period
            )
            if recorder._recording:
                recorder._record(simulation, variable, period, result)
            return result

        run_formula._microcosm_recorder = weakref.ref(self)
        self._run_formula = run_formula

    @classmethod
    def for_declaration(cls, declaration: PopulationDeclaration) -> FormulaRunRecorder:
        return cls(declaration.watched(), predicates=declaration.run_predicates())

    @property
    def runs(self) -> tuple[FormulaRun, ...]:
        return tuple(self._runs)

    def periods_run(self) -> dict[str, frozenset[str]]:
        """Every formula that ran after arming, by period."""

        return {period: frozenset(names) for period, names in self._periods.items()}

    def period_span(self, period: str) -> tuple[float, float] | None:
        """The first and last years a recorded period covers, read from the
        period the engine ran at (None where they could not be read)."""

        return self._spans.get(period)

    def arm(self, simulation: Any) -> None:
        """Record from now on in ``simulation``, its baseline and its branches.

        Every one is checked before any is bound, so a refusal leaves them,
        and the recorder, as they were.
        """

        targets = list(_walk([simulation]))
        for current in targets:
            bound = vars(current).get("_run_formula")
            function = getattr(bound, "__func__", None)
            if bound is not None and function is not self._run_formula:
                owner = getattr(function, "_microcosm_recorder", None)
                previous = owner() if owner is not None else None
                if owner is None or (previous is not None and previous._recording):
                    raise ValueError(
                        "The simulation already carries its own _run_formula, "
                        "which arming would bypass."
                    )
        for current in targets:
            bound = vars(current).get("_run_formula")
            function = getattr(bound, "__func__", None)
            if function is not self._run_formula or bound.__self__ is not current:
                current._run_formula = types.MethodType(self._run_formula, current)
            self._token(current)
        self._recording = True

    def is_armed(self, simulation: Any) -> bool:
        bound = vars(simulation).get("_run_formula")
        return (
            self._recording
            and getattr(bound, "__func__", None) is self._run_formula
            and bound.__self__ is simulation
        )

    def disarm(self) -> None:
        """Stop recording and unbind from every simulation recorded so far.

        A copy made after arming that never ran a formula keeps the binding,
        which then only passes through.
        """

        self._recording = False
        for reference in self._simulations.values():
            simulation = reference()
            if simulation is None:
                continue
            state = vars(simulation)
            bound = state.get("_run_formula")
            if getattr(bound, "__func__", None) is self._run_formula:
                del state["_run_formula"]
            state.pop(_TOKEN_ATTRIBUTE, None)

    def token(self, simulation: Any) -> int | None:
        """The token runs in ``simulation`` carry, if it ran or was armed."""

        token = vars(simulation).get(_TOKEN_ATTRIBUTE)
        reference = self._simulations.get(token) if token is not None else None
        if reference is not None and reference() is simulation:
            return token
        return None

    def _token(self, simulation: Any) -> int:
        token = self.token(simulation)
        if token is not None:
            return token
        token = self._next_token
        self._next_token += 1
        self._simulations[token] = weakref.ref(simulation)
        vars(simulation)[_TOKEN_ATTRIBUTE] = token
        return token

    def _record(self, simulation: Any, variable: Any, period: Any, result: Any) -> None:
        name = variable.name
        key = _period_key(period)
        token = self._token(simulation)
        self._periods.setdefault(key, set()).add(name)
        if key not in self._spans:
            self._spans[key] = _period_span(period)
        if name not in self.watched:
            return
        produced = result is not None
        predicate = self._predicates.get(name)
        self._runs.append(
            FormulaRun(
                simulation=token,
                name=name,
                period=key,
                produced=produced,
                all_zero=_all_zero(result) if produced else None,
                predicate_holds=(
                    None
                    if predicate is None
                    else _predicate_holds_in(simulation, predicate, period)
                ),
            )
        )


# ---------------------------------------------------------------------------
# The known-periods walk and the judge
# ---------------------------------------------------------------------------


def _walk(simulations: Iterable[Any]) -> Iterator[Any]:
    """Each simulation once, with its baseline and live branches, depth first."""

    seen: set[int] = set()
    held: list[Any] = []
    stack = [simulation for simulation in simulations][::-1]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        held.append(current)
        yield current
        children: list[Any] = []
        branches = getattr(current, "branches", None)
        if isinstance(branches, Mapping):
            children.extend(branches.values())
        baseline = getattr(current, "baseline", None)
        if baseline is not None:
            children.append(baseline)
        stack.extend(reversed(children))


def _derived(
    simulations: Iterable[Any], names: Iterable[str]
) -> Iterator[tuple[Any, str, Any]]:
    """(simulation, name, period) for every value held that is not an input."""

    names = tuple(names)
    for simulation in _walk(simulations):
        for name in names:
            holder = simulation.get_holder(name)
            known = holder.get_known_periods()
            if not known:
                continue
            inputs = _input_keys(simulation, name)
            for period in known:
                if _period_key(period) not in inputs:
                    yield simulation, name, period


def derived_watched_periods(
    simulations: Iterable[Any], names: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    """The periods at which each of ``names`` holds a calculated value, in any
    of ``simulations``, their baselines or their live branches."""

    found: dict[str, set[str]] = {}
    for _simulation, name, period in _derived(simulations, names):
        found.setdefault(name, set()).add(_period_key(period))
    return {name: tuple(sorted(periods)) for name, periods in sorted(found.items())}


def _held_all_zero(simulation: Any, name: str, period: Any) -> bool:
    holder = simulation.get_holder(name)
    try:
        values = holder.get_array(period, getattr(simulation, "branch_name", "default"))
    except TypeError:
        values = holder.get_array(period)
    return values is not None and bool(_all_zero(values))


def _engine_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def judge(
    declaration: PopulationDeclaration,
    policy: PassPolicy,
    recorder: FormulaRunRecorder,
    simulations: Sequence[Any],
    periods: Iterable[Any],
    requested: Iterable[tuple[str, Any]] = (),
) -> PopulationVerdict:
    """Judge what a block or batch's simulations did against the declaration.

    ``simulations`` are the ones the block or batch built (its baselines and
    live branches are walked with them); every one must be armed with
    ``recorder``. ``periods`` are the periods the block computes at, where
    the corrected and stored formulas must be inputs. ``requested`` lists the
    ``(variable, period)`` pairs the caller asked for. Under a ``raise``
    policy a verdict that is not clean raises :class:`PopulationGuardError`.
    """

    tops = list(_walk_tops(simulations))
    targets = [target for top in tops for target in _with_baseline(top)]
    periods = tuple({_period_key(period): period for period in periods}.values())
    requested = tuple(requested)
    window = declaration.window
    violations: list[Violation] = []
    seen: set[tuple[str, str | None, str | None]] = set()

    def flag(kind: str, name: str | None = None, period: Any = None) -> None:
        key = (kind, name, None if period is None else _period_key(period))
        if key not in seen:
            seen.add(key)
            violations.append(Violation(*key))

    for target in targets:
        if not recorder.is_armed(target):
            flag("not_armed")
    for period in periods:
        if window.contains(period):
            flag("window_period", None, period)
    for name, period in requested:
        if window.contains(period):
            flag("window_request", name, period)
    for period, names in sorted(recorder.periods_run().items()):
        # The years the engine's own period covered, recorded at the run.
        if window.reaches(recorder.period_span(period)):
            for name in sorted(names):
                flag("window_run", name, period)
    for top in tops:
        parameters = top.tax_benefit_system.parameters
        for reference in declaration.constructor_factories:
            factory = _resolve_reference(reference)
            for period in periods:
                if factory(parameters, period) is not None:
                    flag("constructor_reform", reference, period)

    runs: dict[str, list[FormulaRun]] = {}
    recorded: set[tuple[int, str, str]] = set()
    for run in recorder.runs:
        if run.produced:
            runs.setdefault(run.name, []).append(run)
            recorded.add((run.simulation, run.name, run.period))
    derived: dict[str, list[tuple[Any, Any]]] = {}
    judged = sorted(declaration.watched())
    for simulation, name, period in _derived(tops, judged):
        derived.setdefault(name, []).append((simulation, period))

    def judged_at_run(simulation: Any, name: str, period: Any) -> bool:
        # A held value was judged when it ran only if this simulation ran it
        # at this period after arming; a run elsewhere says nothing of it.
        token = recorder.token(simulation)
        return token is not None and (token, name, _period_key(period)) in recorded

    def ran_at(name: str) -> set[str]:
        return {run.period for run in runs.get(name, ())} | {
            _period_key(period) for _simulation, period in derived.get(name, ())
        }

    def missing_inputs(name: str) -> list[Any]:
        return [
            period
            for period in periods
            if any(
                _period_key(period) not in _input_keys(target, name)
                for target in targets
            )
        ]

    classes: dict[str, set[str]] = {
        "corrected": set(),
        "stored": set(),
        "zero_at_policy": set(),
        "inert": set(),
    }
    for name in sorted(declaration.corrected):
        missing = missing_inputs(name)
        for period in missing:
            flag("corrected_not_input", name, period)
        for period in sorted(ran_at(name)):
            flag("corrected_derived", name, period)
        if not missing and not ran_at(name):
            classes["corrected"].add(name)
    for name in sorted(declaration.refused):
        for period in sorted(ran_at(name)):
            flag("refused_ran", name, period)
        for requested_name, period in requested:
            if requested_name == name:
                flag("refused_request", name, period)
    for name, predicate in sorted(declaration.zero_at_policy.items()):
        good = True
        for run in runs.get(name, ()):
            if not (run.predicate_holds and run.all_zero):
                good = False
                flag("zero_at_policy", name, run.period)
        for simulation, period in derived.get(name, ()):
            if judged_at_run(simulation, name, period):
                continue
            if not (
                _predicate_holds_in(simulation, predicate, period)
                and _held_all_zero(simulation, name, period)
            ):
                good = False
                flag("zero_at_policy", name, period)
        if good and ran_at(name):
            classes["zero_at_policy"].add(name)
    for name in sorted(declaration.stored):
        missing = missing_inputs(name)
        ran = sorted(ran_at(name))
        if not missing and not ran:
            classes["stored"].add(name)
            continue
        rule = declaration.inert.get(name) if policy.inert_allowed else None
        if rule is None:
            for period in missing:
                flag("stored_not_input", name, period)
            for period in ran:
                flag("stored_derived", name, period)
            continue
        before = len(violations)
        _judge_inert(name, rule, ran, runs, derived, requested, flag, judged_at_run)
        if len(violations) == before and ran:
            classes["inert"].add(name)

    truncated = len(violations) > _VIOLATION_LIMIT
    verdict = PopulationVerdict(
        pass_name=policy.name,
        violations=tuple(violations[:_VIOLATION_LIMIT]),
        truncated=truncated,
        classes={name: tuple(sorted(names)) for name, names in classes.items()},
        window_first_year=window.first_year,
        declaration_sha256=declaration.sha256,
        engine=declaration.distribution,
        engine_version=_engine_version(declaration.distribution),
    )
    if not verdict.clean and policy.on_violation == "raise":
        raise PopulationGuardError(verdict)
    return verdict


def _walk_tops(simulations: Iterable[Any]) -> Iterator[Any]:
    seen: set[int] = set()
    for simulation in simulations:
        if simulation is not None and id(simulation) not in seen:
            seen.add(id(simulation))
            yield simulation


def _judge_inert(
    name: str,
    rule: InertRule,
    ran: Sequence[str],
    runs: Mapping[str, Sequence[FormulaRun]],
    derived: Mapping[str, Sequence[tuple[Any, Any]]],
    requested: Sequence[tuple[str, Any]],
    flag: Any,
    judged_at_run: Any,
) -> None:
    """Flag every way an inert formula's runs could reach a value."""

    for run in runs.get(name, ()):
        if not run.predicate_holds:
            flag("inert_predicate", name, run.period)
    sink_runs = runs.get(rule.sink, ())
    # The periods the sink ran or is held at, in any simulation; each held
    # value is judged unless the simulation holding it ran it after arming.
    sink_periods = {run.period for run in sink_runs}
    for simulation, period in derived.get(rule.sink, ()):
        sink_periods.add(_period_key(period))
        if judged_at_run(simulation, rule.sink, period):
            continue
        if not (
            _predicate_holds_in(simulation, rule.predicate, period)
            and _held_all_zero(simulation, rule.sink, period)
        ):
            flag("inert_sink", rule.sink, period)
    for run in sink_runs:
        if run.period in ran and not (run.predicate_holds and run.all_zero):
            flag("inert_sink", rule.sink, run.period)
    for period in ran:
        if period not in sink_periods:
            flag("inert_without_sink", name, period)
    for carrier in rule.carriers:
        periods = {run.period for run in runs.get(carrier, ())} | {
            _period_key(period) for _simulation, period in derived.get(carrier, ())
        }
        for period in sorted(periods):
            flag("inert_carrier", carrier, period)
    unrequestable = rule.unrequestable(name)
    for requested_name, period in requested:
        if requested_name in unrequestable:
            flag("inert_request", requested_name, period)


# ---------------------------------------------------------------------------
# One block's life
# ---------------------------------------------------------------------------


def validate_pool_shares(
    declaration: PopulationDeclaration,
    shares: Mapping[str, PoolShare],
    *,
    weight_scale: float | None = None,
) -> None:
    """Refuse pool shares that are not one block's, for the declared formulas.

    One :class:`PoolShare` per formula the declaration corrects, over its
    declared key. With ``weight_scale`` (the factor the block's engine
    weights carry, 1 when they are unscaled) every share must carry it too.
    A caller's error, so it raises under any pass.
    """

    expected = dict(declaration.corrected)
    if set(shares) != set(expected):
        raise ValueError(
            f"engine pool shares must cover exactly {sorted(expected)}; got "
            f"{sorted(shares)}."
        )
    for formula, share in shares.items():
        if not isinstance(share, PoolShare):
            raise TypeError(f"{formula}: engine pool shares must be PoolShare records.")
        if share.formula != formula or share.key_variable != expected[formula]:
            raise ValueError(
                f"{formula}: the pool share is for {share.formula} over "
                f"{share.key_variable}, not over {expected[formula]}."
            )
        if weight_scale is not None and share.weight_scale != float(weight_scale):
            raise ValueError(
                f"{formula}: the pool share's weight factor is not the block's "
                "engine weight scale."
            )


class EngineBlock:
    """One engine block or batch standing for the pool, over its life.

    Built right after the block's simulation, before anything reads it, from
    the block's :class:`PoolShare` records (:func:`validate_pool_shares`),
    the declaration of the engine's whole-population formulas and the pass.
    ``periods`` are the years the block computes at and ``dataset_year`` the
    year its data is for. In order, on the block's own engine:

    - the declaration is validated against the simulation's tax-benefit
      system (:func:`validate_declaration`), here and never on one engine
      over the whole pool;
    - a period in the declaration's position window is refused, as the
      verdict will be: a block computing there is not the pool, corrected
      or not;
    - a period before the dataset's year is refused: the engine does not
      project back to it, so a correction from that year's totals does not
      hold there;
    - check (a) at the dataset's year (:func:`check_key_sums`);
    - check (b) from the dataset's year to every later period, on each key
      and on the weight variable (:func:`check_uniform_growth`);
    - every corrected formula is set at every period in the simulation and
      its baseline (:func:`inject_pool_shares`);
    - a :class:`FormulaRunRecorder` for the declaration is armed on both.

    Every simulation the caller builds afterwards takes the same arrays and
    the recorder through :meth:`wrap`; one that would set what the
    correction reads is refused by :meth:`refuse_substitution`.
    :meth:`note_request` records what the caller asks for. :meth:`close`, at
    the block's end, checks that every simulation still holds the block's
    keys and weights, bitwise, judges them all under the pass and disarms.
    Work after that is refused: nothing would judge it.

    Under a ``record`` pass a refusal leaves the block uncorrected or not
    judged, and says why (:attr:`refusals`), so the caller's result is not
    exact; under ``raise`` it raises. :meth:`receipt` holds names, periods,
    booleans, the corrections and the growth factors: ratios of
    block-weighted totals and parameter-driven factors, no record's value.
    """

    def __init__(
        self,
        simulation: Any,
        shares: Mapping[str, PoolShare],
        *,
        declaration: PopulationDeclaration,
        policy: PassPolicy,
        periods: Iterable[int],
        dataset_year: int,
    ) -> None:
        validate_pool_shares(declaration, shares)
        self.main = simulation
        self.shares = dict(shares)
        self.declaration = declaration
        self.policy = policy
        self.dataset_year = int(dataset_year)
        self.periods = tuple(sorted({int(period) for period in periods}))
        self.declaration_valid = False
        self.injected: InjectedShares | None = None
        self.recorder: FormulaRunRecorder | None = None
        self.key_sum_gaps: dict[str, float] = {}
        self.key_sums_verified = False
        self.growth_factors: dict[str, dict[str, float | None]] = {}
        self.uniform_growth_verified = False
        self.siblings: list[Any] = []
        self.siblings_match: bool | None = None
        self.requested: set[tuple[str, int]] = set()
        self.refusals: list[str] = []
        self.verdict: PopulationVerdict | None = None
        self.closed = False
        try:
            validate_declaration(declaration, simulation.tax_benefit_system)
        except PopulationDeclarationError as error:
            self._refuse(error)
            return
        self.declaration_valid = True
        window = declaration.window
        inside = [period for period in self.periods if window.contains(period)]
        if inside:
            self._refuse(
                PoolShareError(
                    f"the block computes at {inside}, from {window.first_year} "
                    f"on, where {window.reason}."
                )
            )
        try:
            self._correct(simulation)
        except PoolShareError as error:
            self._refuse(error)
        self.recorder = FormulaRunRecorder.for_declaration(declaration)
        self.recorder.arm(simulation)

    def _refuse(self, error: Exception) -> None:
        if self.policy.on_violation == "raise":
            raise error
        self.refusals.append(str(error))

    def _correct(self, simulation: Any) -> None:
        weight = self.declaration.weight_variable
        early = [period for period in self.periods if period < self.dataset_year]
        if early:
            raise PoolShareError(
                f"the block computes at {early}, before the dataset's year "
                f"{self.dataset_year}, which the engine does not project to, so "
                "a correction from that year's totals does not hold there."
            )
        self.key_sum_gaps = check_key_sums(
            simulation, self.shares, self.dataset_year, weight_variable=weight
        )
        self.key_sums_verified = True
        carried = (*sorted(set(self.declaration.corrected.values())), weight)
        for period in self.periods:
            if period == self.dataset_year:
                continue
            factors = check_uniform_growth(
                simulation, carried, self.dataset_year, period
            )
            for name, factor in factors.items():
                if factor is not None and not (math.isfinite(factor) and factor > 0):
                    raise PoolShareError(
                        f"The engine does not carry {name} from "
                        f"{self.dataset_year} to {period}."
                    )
            self.growth_factors[str(period)] = factors
        self.uniform_growth_verified = True
        self.injected = inject_pool_shares(
            simulation, self.shares, self.periods, weight_variable=weight
        )

    def wrap(self, factory: Callable[..., Any]) -> Callable[..., Any]:
        """``factory``, giving each simulation it builds the block's arrays
        and recorder before the caller sets any input on it."""

        def build(*args: Any, **kwargs: Any) -> Any:
            sibling = factory(*args, **kwargs)
            if self.closed:
                # Not armed: arming would record again, and nothing judges it.
                self._refuse(
                    PoolShareError(
                        "A simulation was built after the block was judged, "
                        "so nothing judges what it computes."
                    )
                )
                return sibling
            if self.injected is not None:
                try:
                    self.injected.apply(sibling)
                except PoolShareError as error:
                    self._refuse(error)
            if self.recorder is not None:
                self.recorder.arm(sibling)
            self.siblings.append(sibling)
            return sibling

        return build

    def note_request(self, name: str, period: int) -> None:
        """Record that the caller asked for ``name`` at ``period``, which
        the verdict judges (a refused formula, the window)."""

        if self.closed:
            self._refuse(
                PoolShareError(
                    f"{name} was asked for at {int(period)} after the block was "
                    "judged, so nothing judges it."
                )
            )
            return
        self.requested.add((str(name), int(period)))

    def refuse_substitution(self, names: Iterable[str | None], route: str) -> None:
        """Refuse a simulation that would set what the block's correction
        reads (:meth:`PopulationDeclaration.correction_reads`): its shares
        would no longer be the pool's. Raises under any pass, before the
        simulation is built."""

        reads = self.declaration.correction_reads()
        touched = sorted({str(name) for name in names if name} & reads)
        if touched:
            raise ValueError(
                f"{route} would set {touched}, which the block's pool-share "
                "correction reads (the share formulas, their allocation keys "
                "and the keys' inputs, and the weights), so its shares would no "
                "longer be the pool's."
            )

    def close(self) -> PopulationVerdict | None:
        """Judge the block's simulations once, at its end, and disarm.

        Returns the verdict, None when the declaration did not validate (so
        nothing was armed). Under a ``raise`` pass a simulation that no
        longer holds the block's keys or weights raises
        :class:`PoolShareError`, and a verdict that is not clean
        :class:`PopulationGuardError`.
        """

        if self.closed:
            return self.verdict
        self.closed = True
        if self.recorder is None:
            return None
        simulations = [self.main, *self.siblings]
        try:
            if self.injected is not None:
                self.siblings_match = all(
                    self.injected.matches(simulation) for simulation in simulations
                )
                if not self.siblings_match:
                    self._refuse_moved_keys()
            self.verdict = judge(
                self.declaration,
                self.policy,
                self.recorder,
                simulations,
                self.periods,
                sorted(self.requested),
            )
        finally:
            self.recorder.disarm()
        return self.verdict

    def _refuse_moved_keys(self) -> None:
        # Recorded by siblings_match under a record pass.
        if self.policy.on_violation == "raise":
            raise PoolShareError(
                "A simulation the block built no longer holds the block's keys "
                "or weights, so its shares are not the pool's."
            )

    def _corrections(self) -> dict[str, float | None]:
        corrections: dict[str, float | None] = {}
        for formula, share in sorted(self.shares.items()):
            try:
                corrections[formula] = share.factor()
            except PoolShareError:
                corrections[formula] = None
        return corrections

    def receipt(self) -> dict[str, Any]:
        """The correction: names, periods, booleans, ratios of block-weighted
        totals and parameter-driven growth factors."""

        return {
            "periods": [str(period) for period in self.periods],
            "dataset_year": self.dataset_year,
            "formulas": {
                formula: share.key_variable
                for formula, share in sorted(self.shares.items())
            },
            "corrections": self._corrections(),
            "declaration_valid": self.declaration_valid,
            "key_sums_verified": self.key_sums_verified,
            "largest_key_sum_gap": (
                max(self.key_sum_gaps.values()) if self.key_sum_gaps else None
            ),
            "uniform_growth_verified": self.uniform_growth_verified,
            "growth_factors": {
                period: dict(sorted(factors.items()))
                for period, factors in sorted(self.growth_factors.items())
            },
            "applied": self.injected is not None,
            "siblings_match": self.siblings_match,
            "refusals": list(self.refusals),
        }

    def receipt_entries(self) -> dict[str, Any]:
        """What the block adds to its caller's receipt: ``engine_pool_shares``
        (:meth:`receipt`) and, once judged, ``engine_population_reach`` (the
        verdict)."""

        entries: dict[str, Any] = {"engine_pool_shares": self.receipt()}
        if self.verdict is not None:
            entries["engine_population_reach"] = self.verdict.as_receipt()
        return entries


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class HouseholdBatch:
    """The households carrying one value of the batch column, and their people."""

    key: Any
    households: np.ndarray
    persons: np.ndarray


def assert_groups_nest_in_households(
    bundle: Any, *, household: str = "household"
) -> None:
    """Refuse a frame with a group that has members in two households.

    Batches take whole households; such a group would be split between
    batches and each batch would simulate it with part of its members.
    """

    schema = bundle.schema
    person = bundle.table(schema.person_entity)
    households = person[schema.membership_column(household)].to_numpy()
    for group in schema.group_entities:
        if group == household:
            continue
        pairs = pd.DataFrame(
            {
                "group": person[schema.membership_column(group)].to_numpy(),
                "household": households,
            }
        ).drop_duplicates()
        if pairs["group"].duplicated().any():
            raise ValueError(
                f"{group} units must nest in households: one with members in "
                "two households would be split between household batches."
            )


def household_batches(
    bundle: Any, column: str, *, household: str = "household"
) -> tuple[HouseholdBatch, ...]:
    """Whole-household batches of ``bundle``, one per value of ``column``.

    In ascending order of the value, each with boolean masks over the
    bundle's household and person rows (people keep the bundle's order).
    Groups must nest in households.
    """

    assert_groups_nest_in_households(bundle, household=household)
    schema = bundle.schema
    table = bundle.table(household)
    if column not in table.columns:
        raise ValueError(f"The {household} table carries no {column!r} column.")
    values = table[column].to_numpy()
    if pd.isna(values).any():
        raise ValueError(f"{column!r} is missing for some {household} rows.")
    ids = table[schema.id_column(household)].to_numpy()
    membership = (
        bundle.table(schema.person_entity)[schema.membership_column(household)]
    ).to_numpy()
    batches = []
    for key in np.unique(values):
        in_batch = values == key
        batches.append(
            HouseholdBatch(
                key=key.item() if hasattr(key, "item") else key,
                households=in_batch,
                persons=np.isin(membership, ids[in_batch]),
            )
        )
    return tuple(batches)


def scatter_by_entity_ids(
    bundle: Any,
    entity: str,
    parts: Iterable[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """One array over ``bundle``'s ``entity`` rows from per-batch arrays.

    ``parts`` holds each batch's ``(ids, values)``, the ids being the batch
    frame's own. The result follows the bundle's rows: people in the
    bundle's order and groups in ascending id order (a frame keeps its group
    tables sorted), which is the order one engine over the bundle returns.
    Every row must be filled exactly once.
    """

    target = np.asarray(bundle.table(entity)[bundle.schema.entity_id_column(entity)])
    parts = [(np.asarray(ids), np.asarray(values)) for ids, values in parts]
    if not parts:
        raise ValueError("Nothing to scatter.")
    for ids, values in parts:
        if ids.shape != values.shape:
            raise ValueError("A batch's ids and values do not align.")
    dtypes = {values.dtype for _ids, values in parts}
    if len(dtypes) != 1:
        raise ValueError(f"The batches return {entity} values of different types.")
    ids = np.concatenate([ids for ids, _values in parts])
    values = np.concatenate([values for _ids, values in parts])
    if ids.shape != target.shape:
        raise ValueError(
            f"The batches hold a different number of {entity} rows from the bundle."
        )
    order = np.argsort(target, kind="stable")
    ordered = target[order]
    if np.any(ordered[1:] == ordered[:-1]):
        raise ValueError(f"The bundle's {entity} ids repeat.")
    positions = np.searchsorted(ordered, ids)
    found = positions < len(ordered)
    found[found] = ordered[positions[found]] == ids[found]
    if not found.all():
        raise ValueError(f"A batch returns {entity} ids the bundle does not hold.")
    slots = order[positions]
    if len(np.unique(slots)) != len(slots):
        raise ValueError(f"Two batches return the same {entity} row.")
    result = np.empty(len(target), dtype=values.dtype)
    result[slots] = values
    return result
