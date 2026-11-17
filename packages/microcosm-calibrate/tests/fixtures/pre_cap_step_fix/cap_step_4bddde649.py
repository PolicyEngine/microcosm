"""Immutable softmax cap step the full-scale measurement ran, from commit 4bddde649.

The three functions below are copied verbatim from
``packages/microcosm-calibrate/src/microcosm/calibrate/solve.py`` at
4bddde6499c5a9001700652999b00bd376aeace8, the head every receipt in
``experiments/us-acs-local-l2-basis-20260928/results/exact_cap/`` records
(solve.py's sha256 there is cf6d74a641a5...). Tests verify this module's SHA
before use and compare the installed cap step with it on the inputs where it
settles in its active-set rounds. This module is only an oracle; it never
replaces a solver function.
"""

from __future__ import annotations

import math

import torch

_SOFTMAX_CAP_ACTIVE_SET_ROUNDS = 8


def _cap_every_record_shift(cap_shift: torch.Tensor) -> float:
    """The smallest shift that puts every record able to reach its cap at it."""
    reachable = cap_shift[torch.isfinite(cap_shift)]
    return float(reachable.max()) if reachable.numel() else 0.0


def _sorted_capped_log_shift(
    cap_shift: torch.Tensor,
    caps: torch.Tensor,
    mass: torch.Tensor,
    total: float,
    capped: torch.Tensor,
    lower: float,
) -> float:
    """Finish :func:`_capped_log_shift` exactly with one sort.

    ``capped`` marks records known to be at their caps at the answer and
    ``lower`` is a lower bound on it. Sort the remaining records by the shift
    at which each meets its cap. With the first ``k`` of them capped too, the
    shift that meets ``total`` is ``log(room_k / free_k)``, ``room_k`` being
    the total less every capped record's cap and ``free_k`` the mass of the
    records after the ``k``-th. The answer is that shift at the smallest
    ``k`` where it does not pass the ``k``-th record's own cap shift: the
    ``k`` records before it reach their caps at a smaller shift, so they are
    capped, and every record from the ``k``-th on is not.
    """
    rest = ~capped & (mass > 0.0)
    order = torch.argsort(cap_shift[rest])
    rest_shift = cap_shift[rest][order]
    rest_caps = caps[rest][order]
    rest_mass = mass[rest][order]
    capped_before = torch.cat(
        (rest_caps.new_zeros(1), torch.cumsum(rest_caps, dim=0)[:-1])
    )
    room = (total - float(caps[capped].sum())) - capped_before
    free = torch.flip(torch.cumsum(torch.flip(rest_mass, (0,)), dim=0), (0,))
    # Once the capped records' caps reach the total, no shift meets it short
    # of capping every record (a cap of exactly 1, up to rounding).
    candidate = torch.where(
        room > 0.0,
        torch.log(room.clamp_min(torch.finfo(torch.float64).tiny)) - torch.log(free),
        math.inf,
    )
    meets = candidate <= rest_shift
    if not bool(meets.any()):
        return max(lower, _cap_every_record_shift(cap_shift))
    return max(lower, float(candidate[int(meets.to(torch.uint8).argmax())]))


def _capped_log_shift(
    log_weights: torch.Tensor,
    log_upper: torch.Tensor,
    total: float,
    *,
    active_set_rounds: int = _SOFTMAX_CAP_ACTIVE_SET_ROUNDS,
) -> tuple[float, int]:
    """The shift ``s`` with ``sum(min(exp(log_weights + s), exp(log_upper))) == total``.

    Float64 tensors; ``log_weights`` sums to ``total`` (``exp`` of it does),
    so ``s >= 0``. The left side is continuous and nondecreasing in ``s``, so
    the shift exists and the projection it defines is unique whenever the caps
    sum to at least ``total``.

    Active-set rounds: with ``C`` the records whose cap shift
    ``log_upper - log_weights`` is at most the current shift, the next shift
    solves ``sum(caps[C]) + exp(s) * sum(mass[~C]) == total``. That left side
    bounds the true one from above for every ``C`` (a record outside ``C``
    contributes ``exp(...)`` where the true sum has the smaller ``min``), so
    every round's shift is a lower bound on the answer and ``C`` only grows.
    When a round adds no record, the bound is attained and the shift is exact.
    If ``active_set_rounds`` pass without that, :func:`_sorted_capped_log_shift`
    finishes from the current capped set and bound. Records with a zero cap
    are capped at every shift; records with zero mass never reach a positive
    cap and contribute nothing.

    Returns the shift and the rounds taken (``active_set_rounds + 1`` when
    the sort finished it).
    """
    cap_shift = torch.where(
        torch.isneginf(log_upper),
        torch.full_like(log_upper, -math.inf),
        log_upper - log_weights,
    )
    caps = torch.exp(log_upper)
    mass = torch.exp(log_weights)
    zero = torch.zeros((), dtype=torch.float64)
    shift = 0.0
    capped = cap_shift <= shift
    for rounds in range(1, active_set_rounds + 1):
        room = total - float(torch.where(capped, caps, zero).sum())
        free = float(torch.where(capped, zero, mass).sum())
        if room <= 0.0 or free <= 0.0:
            # The capped records alone hold the total (a cap of exactly 1, up
            # to rounding): every record that can reach its cap sits at it.
            return max(shift, _cap_every_record_shift(cap_shift)), rounds
        shift = max(shift, math.log(room) - math.log(free))
        grown = capped | (cap_shift <= shift)
        if int(grown.sum()) == int(capped.sum()):
            return shift, rounds
        capped = grown
    return (
        _sorted_capped_log_shift(cap_shift, caps, mass, total, capped, shift),
        active_set_rounds + 1,
    )
