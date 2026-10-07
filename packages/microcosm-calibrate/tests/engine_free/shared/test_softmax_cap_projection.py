"""Contracts for the softmax parametrization's per-step cap projection.

``_project_softmax_log_weights_`` maps log-weights onto the capped simplex
``{w : sum(w) == total, w <= cap}`` in log space: one common shift ``s``, then
``min(log_w + s, log_cap)``. Invariants pinned here, each for every input the
strategies draw (with records whose design weight, and so cap, is zero, and
with a cap of exactly 1, where the caps sum to the total):

1. It conserves the total and never passes a cap; a record with zero weight
   or a zero cap stays at zero.
2. It is idempotent, and invariant to a constant added to the input (the
   direction ``total * softmax(log_w)`` cannot see).
3. An input already within its caps after the shift moves by the shift alone.
4. It is the Kullback-Leibler projection: every record below its cap moved by
   one common shift, and every record at its cap would pass it under that
   shift (the KKT conditions of ``min KL(w || exp(log_w))`` on the set).
5. Differential: it equals a float64 water-fill written from the definition,
   and the active-set rounds and the sort that bounds them agree (an input
   built to need one round per record exercises the sort).
6. In float32, as the solver runs it, the clamp is exact in log space and the
   realized ``total * softmax(log_w)`` sits within the float64 caps up to the
   float32 error of the softmax.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.calibrate import solve as solve_module

_SETTINGS = settings(max_examples=300, deadline=None)

#: Float64 agreement in log space, for differences that involve no sum: the
#: shift added to every uncapped record, the clamp.
_EXACT = 1e-10
#: Float64 agreement in weight units, as a share of the total. Each path
#: computes the shift from a sum of up to 40 weights, whose rounding is a few
#: ulps of the total. When the caps nearly meet the total, the uncapped
#: records hold only the small remainder, so that absolute error is large in
#: log space for them; in weight units it stays a few ulps of the total.
_EXACT_MASS = 1e-12
#: Float32 tolerance for the realized weights over their float64 caps. Two
#: float32 errors reach them: rounding ``log_w`` (half an ulp of a log-weight
#: below 16 is 4.8e-7 relative) and the softmax's float32 normalizer. On these
#: sizes they stay below 1e-6; 1e-5 leaves an order of magnitude.
_FLOAT32 = 1e-5


@st.composite
def capped_problems(draw, *, max_size: int = 40):
    """Log-weights, log caps and a total the caps can hold.

    The caps are ``ratio * design`` and the total a ``fill`` share of their
    sum, so ``fill = 1 / ratio`` is calibrate's own total ``sum(design)``
    and ``fill = 1`` leaves no slack. Zero design weights have zero caps and
    zero weight.
    """
    n = draw(st.integers(1, max_size))
    design = np.array(
        draw(
            st.lists(
                st.one_of(st.just(0.0), st.floats(1e-3, 1e4)),
                min_size=n,
                max_size=n,
            )
        )
    )
    if design.sum() == 0.0:
        design[0] = 1.0
    moves = np.array(draw(st.lists(st.floats(-8.0, 8.0), min_size=n, max_size=n)))
    ratio = draw(st.one_of(st.just(1.0), st.floats(1.0, 10.0)))
    fill = draw(st.one_of(st.just(1.0), st.just(1.0 / ratio), st.floats(0.05, 1.0)))
    with np.errstate(divide="ignore"):
        log_weights = np.log(design) + moves
        log_caps = np.log(ratio * design)
    total = fill * float(np.exp(log_caps).sum())
    return log_weights, log_caps, total


def _project(log_weights: np.ndarray, log_caps: np.ndarray, total: float):
    """The solver's projection on a float64 copy."""
    log_w = torch.tensor(log_weights, dtype=torch.float64)
    solve_module._project_softmax_log_weights_(
        log_w, total, torch.tensor(log_caps, dtype=torch.float64)
    )
    return log_w.numpy()


def _reference_water_fill(
    log_weights: np.ndarray, log_caps: np.ndarray, total: float
) -> np.ndarray:
    """Float64 water-fill from the definition, independent of the solver.

    In weight space: scale the masses ``v`` by ``c``, holding at its cap
    ``u`` each record ``c * v`` would pass. Records reach their caps in
    order of ``u / v``; try each count ``k`` of capped records in that order
    and keep the one whose ``c`` caps exactly those ``k``.
    """
    mass = np.exp(log_weights - np.max(log_weights))
    caps = np.exp(log_caps)
    with np.errstate(divide="ignore", invalid="ignore"):
        breakpoints = np.where(mass > 0.0, caps / mass, np.inf)
    order = np.argsort(breakpoints, kind="stable")
    for k in range(len(order) + 1):
        held, scaled = order[:k], order[k:]
        free = mass[scaled].sum()
        room = total - caps[held].sum()
        if free == 0.0:
            # Nothing left to scale: only a total the held caps already meet.
            assert room == pytest.approx(0.0, abs=1e-9 * total)
            weights = np.where(np.isin(np.arange(len(mass)), held), caps, 0.0)
            break
        c = room / free
        below = k == 0 or c >= breakpoints[order[k - 1]] * (1.0 - 1e-12)
        above = k == len(order) or c <= breakpoints[order[k]] * (1.0 + 1e-12)
        if below and above:
            weights = np.minimum(c * mass, caps)
            break
    else:  # pragma: no cover - a feasible total always has a count
        raise AssertionError("no consistent capped count")
    with np.errstate(divide="ignore"):
        return np.log(weights)


def _assert_same_weights(
    actual: np.ndarray, expected: np.ndarray, total: float
) -> None:
    """Log-weight vectors with the same support and weights equal to rounding."""
    assert np.array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_allclose(
        np.exp(actual), np.exp(expected), rtol=1e-9, atol=_EXACT_MASS * total
    )


# --- 1: total, caps, support --------------------------------------------------


@_SETTINGS
@given(capped_problems())
def test_projection_conserves_the_total_and_respects_the_caps(problem) -> None:
    log_weights, log_caps, total = problem
    projected = _project(log_weights, log_caps, total)
    assert (projected <= log_caps).all()
    assert float(np.exp(projected).sum()) == pytest.approx(total, rel=1e-12)
    zero = np.isneginf(log_weights) | np.isneginf(log_caps)
    assert np.isneginf(projected[zero]).all()
    assert np.isfinite(projected[~zero]).all()


# --- 2: idempotence and shift invariance -------------------------------------


@_SETTINGS
@given(capped_problems(), st.floats(-30.0, 30.0))
def test_projection_is_idempotent_and_shift_invariant(problem, offset) -> None:
    log_weights, log_caps, total = problem
    projected = _project(log_weights, log_caps, total)
    _assert_same_weights(_project(projected, log_caps, total), projected, total)
    _assert_same_weights(
        _project(log_weights + offset, log_caps, total), projected, total
    )


# --- 3: a feasible input moves by the shift alone ----------------------------


@_SETTINGS
@given(
    st.lists(st.floats(1e-3, 1e4), min_size=1, max_size=40),
    st.data(),
    st.floats(0.05, 1.0),
    st.floats(-30.0, 30.0),
)
def test_a_feasible_input_moves_by_the_shift_alone(caps, data, fill, offset) -> None:
    """Below its caps once scaled to the total: the projection is the shift.

    Each record sits ``slack`` below its cap, and the total is at most the
    records' own sum, so scaling to it only lowers them.
    """
    log_caps = np.log(np.asarray(caps, dtype=np.float64))
    slack = np.array(
        data.draw(st.lists(st.floats(0.0, 8.0), min_size=len(caps), max_size=len(caps)))
    )
    log_weights = log_caps - slack
    total = fill * float(np.exp(log_weights).sum())
    shifted = log_weights + offset
    expected_shift = math.log(total) - float(np.logaddexp.reduce(shifted))
    _assert_same_weights(
        _project(shifted, log_caps, total), shifted + expected_shift, total
    )


# --- 4: the KKT conditions of the KL projection ------------------------------


@_SETTINGS
@given(capped_problems())
def test_projection_is_the_kullback_leibler_projection(problem) -> None:
    log_weights, log_caps, total = problem
    projected = _project(log_weights, log_caps, total)
    live = np.isfinite(projected)
    at_cap = live & (projected >= log_caps - _EXACT)
    below = live & ~at_cap
    if below.any():
        moved = projected[below] - log_weights[below]
        shift = float(np.median(moved))
        np.testing.assert_allclose(moved, shift, rtol=0.0, atol=_EXACT)
        # Every capped record would pass its cap under the common shift.
        assert (log_weights[at_cap] + shift >= log_caps[at_cap] - _EXACT).all()
    else:
        # Every live record is held: the caps alone meet the total.
        assert float(np.exp(log_caps[live]).sum()) == pytest.approx(total, rel=1e-12)


def test_a_cap_of_one_holds_every_record_at_its_design_weight() -> None:
    design = np.array([3.0, 0.0, 1.5, 40.0, 2.25])
    with np.errstate(divide="ignore"):
        log_caps = np.log(design)
    log_weights = log_caps + np.array([2.0, 0.0, -3.0, 0.5, -0.25])
    projected = _project(log_weights, log_caps, float(design.sum()))
    assert np.array_equal(projected, log_caps)


# --- 5: differential against the float64 water-fill --------------------------


@_SETTINGS
@given(capped_problems())
def test_projection_matches_the_reference_water_fill(problem) -> None:
    log_weights, log_caps, total = problem
    _assert_same_weights(
        _project(log_weights, log_caps, total),
        _reference_water_fill(log_weights, log_caps, total),
        total,
    )


@_SETTINGS
@given(capped_problems(), st.sampled_from([0, 1, 2]))
def test_active_set_rounds_and_the_sort_agree(problem, rounds) -> None:
    log_weights, log_caps, total = problem
    log_w = torch.tensor(log_weights, dtype=torch.float64)
    log_w = log_w + (math.log(total) - float(torch.logsumexp(log_w, dim=0)))
    log_upper = torch.tensor(log_caps, dtype=torch.float64)
    shift, taken = solve_module._capped_log_shift(log_w, log_upper, total)
    assert 1 <= taken <= solve_module._SOFTMAX_CAP_ACTIVE_SET_ROUNDS + 1
    bounded, bounded_taken = solve_module._capped_log_shift(
        log_w, log_upper, total, active_set_rounds=rounds
    )
    assert 1 <= bounded_taken <= rounds + 1
    _assert_same_weights(
        torch.minimum(log_w + bounded, log_upper).numpy(),
        torch.minimum(log_w + shift, log_upper).numpy(),
        total,
    )


def _active_set_chain(n: int, length: int):
    """Unit masses with caps that the active-set rounds take one at a time.

    Round ``r`` caps the records whose cap shift is at most the previous
    round's shift and solves for the next. Each record of the chain gets a
    cap just inside the next round's root, so every round adds exactly one;
    the margins shrink by more than the roots move, which keeps them in
    order. The records after the chain never bind.
    """
    caps = np.full(n, 10.0 * n)
    caps[0] = 0.5
    previous, root = 1.0, (n - 0.5) / (n - 1)
    margin = 0.5 * (root - previous)
    for r in range(1, length):
        caps[r] = root - margin
        previous, root = root, root + margin / (n - r - 1)
        margin = 0.5 * margin / (n - r - 1)
    return np.zeros(n), np.log(caps), float(n)


def test_a_long_active_set_chain_is_finished_by_the_sort() -> None:
    log_weights, log_caps, total = _active_set_chain(14, 12)
    log_w = torch.tensor(log_weights, dtype=torch.float64)
    log_upper = torch.tensor(log_caps, dtype=torch.float64)
    unbounded, unbounded_taken = solve_module._capped_log_shift(
        log_w, log_upper, total, active_set_rounds=10_000
    )
    assert unbounded_taken == 12
    shift, taken = solve_module._capped_log_shift(log_w, log_upper, total)
    assert taken == solve_module._SOFTMAX_CAP_ACTIVE_SET_ROUNDS + 1
    assert shift == pytest.approx(unbounded, abs=_EXACT)
    _assert_same_weights(
        _project(log_weights, log_caps, total),
        _reference_water_fill(log_weights, log_caps, total),
        total,
    )
    assert int(np.sum(_project(log_weights, log_caps, total) >= log_caps)) == 12


# --- 6: float32, as the solver runs it ----------------------------------------


@_SETTINGS
@given(capped_problems())
def test_float32_projection_holds_the_realized_weights_at_their_caps(problem) -> None:
    log_weights, log_caps, total = problem
    caps = np.exp(log_caps)
    upper = torch.tensor(caps, dtype=torch.float32)
    log_upper32 = torch.log(upper)
    log_w = torch.tensor(log_weights, dtype=torch.float32)
    solve_module._project_softmax_log_weights_(
        log_w, total, log_upper32.to(torch.float64)
    )
    assert (log_w <= log_upper32).all()
    realized = total * torch.softmax(log_w, dim=0)
    ratio = solve_module._max_cap_ratio(realized, torch.tensor(caps))
    assert ratio <= 1.0 + _FLOAT32
    assert float(realized.double().sum()) == pytest.approx(total, rel=_FLOAT32)


def test_max_cap_ratio_reads_zero_caps() -> None:
    caps = torch.tensor([2.0, 0.0, 4.0], dtype=torch.float64)
    assert solve_module._max_cap_ratio(
        torch.tensor([1.0, 0.0, 4.0]), caps
    ) == pytest.approx(1.0)
    assert solve_module._max_cap_ratio(torch.tensor([1.0, 1e-30, 1.0]), caps) == (
        math.inf
    )


def test_without_a_cap_the_projection_is_the_shift() -> None:
    log_weights = np.array([0.0, -2.0, 3.5, 1.0])
    log_w = torch.tensor(log_weights, dtype=torch.float64)
    assert solve_module._project_softmax_log_weights_(log_w, 50.0, None) == 0
    np.testing.assert_allclose(
        log_w.numpy() - log_weights,
        math.log(50.0) - float(np.logaddexp.reduce(log_weights)),
        rtol=0.0,
        atol=_EXACT,
    )
