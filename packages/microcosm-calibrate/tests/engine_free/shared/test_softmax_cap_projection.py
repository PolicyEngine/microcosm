"""Contracts for the softmax parametrization's per-step cap projection.

``_project_softmax_log_weights_`` maps log-weights onto the capped simplex
``{w : sum(w) == total, w <= cap}`` in log space: one common shift ``s``, then
``min(log_w + s, log_cap)``. Invariants pinned here, each for every input the
strategies draw. One strategy draws zero weights and zero caps independently
and includes a cap of exactly 1, where the caps sum to the total. The other
lets the caps of a held set meet the total, or nearly, while the other records
sit 45 to 1,100 nats below their caps: below float64 rounding of the total,
and for the deeper ones below float64's range.

1. It conserves the total and never passes a cap; a record with zero weight
   or a zero cap stays at zero, and every other record keeps a finite
   log-weight.
2. It is idempotent, and invariant to a constant added to the input (the
   direction ``total * softmax(log_w)`` cannot see).
3. An input already within its caps after the shift moves by the shift alone.
4. It is the Kullback-Leibler projection: every record below its cap moved by
   one common shift, and every record at its cap would pass it under that
   shift (the KKT conditions of ``min KL(w || exp(log_w))`` on the set).
5. Differential: it equals a water-fill written from the definition in
   60-digit decimal arithmetic, and the active-set rounds and the sort that
   bounds them agree (an input built to need one round per record exercises
   the sort). Wherever the cap step the full-scale measurement ran settles in
   its active-set rounds, with every nonzero weight in float64's normal range,
   the installed step returns the same shift and round count, bit for bit.
6. In float32, as the solver runs it, the clamp is exact in log space and the
   realized ``total * softmax(log_w)`` sits within the float64 caps up to the
   float32 error of the softmax.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.calibrate import solve as solve_module
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-calibrate")

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
#: Significant digits of the reference water-fill.
_REFERENCE_DIGITS = 60


@st.composite
def capped_problems(draw, *, max_size: int = 40):
    """Log-weights, log caps and a total the caps can hold.

    The caps are ``ratio * design`` and the total a ``fill`` share of the caps
    of the records that carry weight, so with every record carrying weight
    ``fill = 1 / ratio`` is calibrate's own total ``sum(design)``, and
    ``fill = 1`` leaves no slack. Zero caps (zero design weights) and zero
    weights are drawn independently: a record can carry weight under a zero
    cap, or have a cap and no weight.
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
    weightless = np.array(
        draw(
            st.lists(
                st.sampled_from((False, False, False, True)), min_size=n, max_size=n
            )
        )
    )
    if not (design[~weightless] > 0.0).any():
        design[0], weightless[0] = 1.0, False
    moves = np.array(draw(st.lists(st.floats(-8.0, 8.0), min_size=n, max_size=n)))
    ratio = draw(st.one_of(st.just(1.0), st.floats(1.0, 10.0)))
    fill = draw(st.one_of(st.just(1.0), st.just(1.0 / ratio), st.floats(0.05, 1.0)))
    with np.errstate(divide="ignore"):
        log_caps = np.log(ratio * design)
    log_weights = np.where(
        weightless, -np.inf, np.log(np.where(design > 0.0, design, 1.0)) + moves
    )
    total = fill * float(np.exp(log_caps[~weightless]).sum())
    return log_weights, log_caps, total


@st.composite
def sunk_problems(draw, *, max_size: int = 40):
    """The caps of a held set meet the total, or nearly; the rest sit far below.

    The held records start within 3 nats of their caps, above or below. The
    sunk records start 45 to 1,100 nats below theirs: their weight is tiny
    next to the total, for most draws below float64 rounding of it, and for
    the deeper ones below float64's range. The total is the held caps plus a
    ``room`` share of the sunk caps. At a share of 0 the held caps alone meet
    the total: the sunk records keep their tiny share, and the cap of
    whichever held record the rounds reach last fills what the others leave
    to within rounding. Above 0 the sunk records must fill a real room, which
    a weight below float64's range can only do in log space.
    """
    held = draw(st.integers(1, max_size // 2))
    sunk = draw(st.integers(1, max_size // 2))
    design = np.array(
        draw(st.lists(st.floats(1e-3, 1e4), min_size=held + sunk, max_size=held + sunk))
    )
    ratio = draw(st.one_of(st.just(1.0), st.floats(1.0, 10.0)))
    log_caps = np.log(ratio * design)
    above = draw(st.lists(st.floats(-3.0, 3.0), min_size=held, max_size=held))
    below = draw(st.lists(st.floats(45.0, 1100.0), min_size=sunk, max_size=sunk))
    log_weights = log_caps + np.concatenate((above, -np.asarray(below)))
    room = draw(st.one_of(st.just(0.0), st.floats(0.01, 1.0)))
    caps = np.exp(log_caps)
    total = float(caps[:held].sum()) + room * float(caps[held:].sum())
    order = np.array(draw(st.permutations(range(held + sunk))))
    return log_weights[order], log_caps[order], total


#: Every drawn input; most properties hold on both families.
problems = st.one_of(capped_problems(), sunk_problems())


def _project(log_weights: np.ndarray, log_caps: np.ndarray, total: float):
    """The solver's projection on a float64 copy."""
    log_w = torch.tensor(log_weights, dtype=torch.float64)
    solve_module._project_softmax_log_weights_(
        log_w, total, torch.tensor(log_caps, dtype=torch.float64)
    )
    return log_w.numpy()


def _normalized(log_weights: np.ndarray, log_caps: np.ndarray, total: float):
    """The float64 inputs the solver hands :func:`_capped_log_shift`."""
    log_w = torch.tensor(log_weights, dtype=torch.float64)
    log_w = log_w + (math.log(total) - float(torch.logsumexp(log_w, dim=0)))
    return log_w, torch.tensor(log_caps, dtype=torch.float64)


def _reference_water_fill(
    log_weights: np.ndarray, log_caps: np.ndarray, total: float
) -> np.ndarray:
    """Water-fill from the definition, in 60-digit decimal arithmetic.

    Independent of the solver and of float64 rounding. Each weight
    ``v = exp(log_weights)`` and cap ``u = exp(log_caps)`` is computed to 60
    digits, so a weight below float64's range still counts. Scale the weights
    by ``c``, holding at its cap each record ``c * v`` would pass. Records
    reach their caps in order of ``u / v``; try each count ``k`` of capped
    records in that order and keep the one whose ``c`` caps exactly those
    ``k``.
    """
    with localcontext() as context:
        context.prec = _REFERENCE_DIGITS
        weights = [
            Decimal(float(x)).exp() if math.isfinite(x) else Decimal(0)
            for x in log_weights
        ]
        caps = [
            Decimal(float(x)).exp() if math.isfinite(x) else Decimal(0)
            for x in log_caps
        ]
        target = Decimal(float(total))
        breakpoints = [
            cap / weight if weight else Decimal("Infinity")
            for weight, cap in zip(weights, caps, strict=True)
        ]
        order = sorted(range(len(weights)), key=breakpoints.__getitem__)
        for k in range(len(order) + 1):
            held, scaled = order[:k], order[k:]
            room = target - sum((caps[i] for i in held), Decimal(0))
            free = sum((weights[i] for i in scaled), Decimal(0))
            if not free:
                # Nothing left to scale: only a total the held caps already meet.
                assert abs(room) <= Decimal("1e-9") * target
                scale = Decimal(0)
                break
            if room <= 0:
                continue
            scale = room / free
            below = k == 0 or scale >= breakpoints[order[k - 1]]
            above = k == len(order) or scale <= breakpoints[order[k]]
            if below and above:
                break
        else:  # pragma: no cover - a feasible total always has a count
            raise AssertionError("no consistent capped count")
        capped = set(held)
        result = [
            caps[i] if i in capped else scale * weights[i] for i in range(len(weights))
        ]
        return np.array([float(w.ln()) if w > 0 else -np.inf for w in result])


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
@given(problems)
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
@given(problems, st.floats(-30.0, 30.0))
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
@given(problems)
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


# --- caps that hold the total, and weights below float64's range -------------


@pytest.mark.parametrize("total", [1.0, 1000.0])
@pytest.mark.parametrize("below", [46.0, 1000.0])
def test_caps_that_hold_the_total_leave_the_other_records_their_share(
    total: float, below: float
) -> None:
    """A cap of 1.05 where one record's cap holds the total, to float64 rounding.

    The first record sits at its cap, which is the total. In float64 that cap
    meets a total of 1 exactly and falls two ulps short of 1,000. The other
    record sits ``below`` nats under the total: below float64 rounding of it
    at 46 nats, below float64's range at 1,000. Its cap is 5% of the total.
    The first cap leaves it at most that rounding, so its weight stays below
    1e-12 of the total. The cap step the full-scale measurement ran lifted it
    to its cap instead, 5% over the total, in three of these four cases
    (microcosm#1096 review, S2).
    """
    log_caps = np.log([total, 0.05 * total])
    log_weights = np.array([math.log(total), math.log(total) - below])
    projected = _project(log_weights, log_caps, total)
    assert float(np.exp(projected).sum()) == pytest.approx(total, rel=1e-12)
    assert projected[0] == log_caps[0]
    assert math.exp(projected[1]) <= _EXACT_MASS * total


@pytest.mark.parametrize("below", [742.0, 1000.0])
def test_a_weight_below_float64s_range_still_fills_the_room(below: float) -> None:
    """A finite log-weight carries weight even where float64 underflows it.

    The first record's cap holds 60% of the total. The second record sits
    ``below`` nats lower, and its cap is 50% of the total; it alone can fill
    the other 40%. At 742 nats its weight is subnormal in float64, with about
    four significant digits; at 1,000 it is zero. The answer lifts it to
    exactly 40%, which only log space can compute. A float64 sum of the
    subnormal weight misses 40% by about 1e-5 of it; counting a zero weight
    as none leaves the room empty, and lifting the record to its cap puts the
    total 10% over (microcosm#1096 review, S2).
    """
    total = 1000.0
    log_caps = np.log([0.6 * total, 0.5 * total])
    log_weights = np.array([math.log(total), math.log(total) - below])
    projected = _project(log_weights, log_caps, total)
    np.testing.assert_allclose(
        np.exp(projected), [0.6 * total, 0.4 * total], rtol=1e-12
    )


# --- 5: differential against the reference water-fill ------------------------


@_SETTINGS
@given(problems)
def test_projection_matches_the_reference_water_fill(problem) -> None:
    log_weights, log_caps, total = problem
    _assert_same_weights(
        _project(log_weights, log_caps, total),
        _reference_water_fill(log_weights, log_caps, total),
        total,
    )


@_SETTINGS
@given(problems, st.sampled_from([0, 1, 2]))
def test_active_set_rounds_and_the_sort_agree(problem, rounds) -> None:
    log_weights, log_caps, total = problem
    log_w, log_upper = _normalized(log_weights, log_caps, total)
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


def test_the_sort_caps_a_record_that_fails_with_no_room_left() -> None:
    """The sort's rounding branch: no room is left after a record that failed.

    The first record's cap is the whole total and its weight sits below it;
    the second record's weight is 800 nats lower. In exact arithmetic the
    first record fails its own test only if room remains after its cap. For
    some of these ``u``, float64 rounds ``log(exp(u))`` above ``u``, so the
    first record fails with no room left, and the sort must cap it at its own
    cap shift. Its lower bound, 0, would leave it at its weight, 22% to 39%
    short of the total.
    """
    nothing_capped = torch.zeros(2, dtype=torch.bool)
    for u in np.linspace(0.25, 0.5, 64):
        log_w = torch.tensor([0.0, -800.0], dtype=torch.float64)
        log_upper = torch.tensor([u, math.log(10.0)], dtype=torch.float64)
        caps = torch.exp(log_upper)
        total = float(caps[0])
        shift = solve_module._sorted_capped_log_shift(
            log_upper - log_w, caps, log_w, total, nothing_capped, 0.0
        )
        projected = torch.minimum(log_w + shift, log_upper)
        assert float(projected.exp().sum()) == pytest.approx(total, rel=1e-15)


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


# The cap step the full-scale measurement ran, as a verbatim oracle.
_MEASURED_CAP_STEP = "cap_step_4bddde649.py"
_MEASURED_CAP_STEP_SHA256 = (
    "da94ee960c55e0c75bbac939ffd65bbcf8bdd57057cfb84e0c961279845f5c5f"
)
#: Calls the measured step made to its every-record cap: the exit microcosm#1096's
#: review (S2) replaced. Cleared before each comparison.
_MEASURED_CAPPED_EVERY_RECORD: list[float] = []


@functools.cache
def _measured_cap_step():
    path = _TEST_PATHS.tests / "fixtures/pre_cap_step_fix" / _MEASURED_CAP_STEP
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _MEASURED_CAP_STEP_SHA256
    spec = importlib.util.spec_from_file_location("measured_cap_step", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    cap_every_record = oracle._cap_every_record_shift

    def recorded(cap_shift: torch.Tensor) -> float:
        shift = cap_every_record(cap_shift)
        _MEASURED_CAPPED_EVERY_RECORD.append(shift)
        return shift

    oracle._cap_every_record_shift = recorded
    return oracle


@_SETTINGS
@given(capped_problems())
def test_the_measured_cap_step_is_unchanged_where_its_rounds_settle(problem) -> None:
    """Pins the arithmetic the full-scale measurement ran (review S1).

    ``experiments/us-acs-local-l2-basis-20260928/results/exact_cap.md``
    measured the cap step at 4bddde649. Each of its 1,600 steps settled in
    one to three active-set rounds, and none capped every record: after the
    caps first bound, no forward pass's largest weight-to-cap ratio fell more
    than 1% below the previous pass's, and that exit would have renormalized
    a 5x-capped vector to about a fifth of its caps. Wherever the measured
    step settles the same way, the installed one returns its shift and round
    count bit for bit. These inputs keep every nonzero weight in float64's
    normal range; below it the installed step hands its rounds over to the
    log-space sort, which the measured step lacked
    (``test_a_weight_below_float64s_range_still_fills_the_room``).
    """
    measured = _measured_cap_step()
    log_w, log_upper = _normalized(*problem)
    total = problem[2]
    _MEASURED_CAPPED_EVERY_RECORD.clear()
    expected = measured._capped_log_shift(log_w, log_upper, total)
    assume(
        not _MEASURED_CAPPED_EVERY_RECORD
        and expected[1] <= measured._SOFTMAX_CAP_ACTIVE_SET_ROUNDS
    )
    assert solve_module._capped_log_shift(log_w, log_upper, total) == expected


# --- 6: float32, as the solver runs it ----------------------------------------


@_SETTINGS
@given(problems)
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
