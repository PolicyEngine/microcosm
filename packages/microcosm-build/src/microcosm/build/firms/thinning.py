"""Stratified thinning of a candidate firm draw, with exact base weights.

A full firm census (millions of rows) is slow to calibrate. Thinning keeps
rows inside an analysis window (for example around a tax threshold) at one
rate and rows outside it at another, stratum by stratum, and gives each kept
row the base weight that restores its stratum's row count. Strata are the
caller's keys (for example industry and turnover band) crossed with the
window side.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["stratified_thin"]


def _as_1d(name: str, values: object, length: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} has {len(array)} entries; expected {length}")
    return array


def stratified_thin(
    window_values: np.ndarray,
    strata: Sequence[np.ndarray],
    *,
    window_lo: float,
    window_hi: float,
    window_fraction: float,
    tail_fraction: float,
    cell_floor: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin rows by stratum and return the kept rows and their base weights.

    Rows whose ``window_values`` lie inside ``[window_lo, window_hi]``
    (inclusive) are kept with probability ``window_fraction``; every other row
    with probability ``tail_fraction``. A stratum (the ``strata`` keys crossed
    with the window side) that keeps fewer than ``min(cell_floor, stratum
    size)`` rows is topped up with its unkept rows of lowest uniform draw, so
    small strata survive. Each kept row's base weight is its stratum's size
    divided by its kept count, so the base-weighted count of every stratum
    equals its row count (up to float64 rounding).

    One uniform draw per row comes from ``np.random.default_rng(rng_seed)``.
    Draws lie in ``[0, 1)``, so fractions of 1.0 keep every row at weight 1.0.

    Args:
        window_values: Float array, shape ``(n,)``: the value the window reads.
        strata: Integer arrays, each shape ``(n,)``: the stratum keys. An
            empty sequence stratifies by window side alone.
        window_lo: Lower window edge (inclusive).
        window_hi: Upper window edge (inclusive).
        window_fraction: Keep probability inside the window, in ``[0, 1]``.
        tail_fraction: Keep probability outside the window, in ``[0, 1]``.
        cell_floor: Minimum kept rows per stratum (capped at stratum size).
        rng_seed: Seed of the generator that draws the uniforms.

    Returns:
        ``(keep_idx, base_weights)``: ascending int64 positions of the kept
        rows and their float64 base weights.
    """

    values = _as_1d("window_values", window_values).astype(np.float64)
    n = len(values)
    keys = []
    for position, key in enumerate(strata):
        array = _as_1d(f"strata[{position}]", key, n)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"strata[{position}] must be an integer array")
        keys.append(array.astype(np.int64))
    for name, fraction in (
        ("window_fraction", window_fraction),
        ("tail_fraction", tail_fraction),
    ):
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {fraction}")
    if int(cell_floor) < 0:
        raise ValueError(f"cell_floor must be non-negative, got {cell_floor}")
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    in_window = (values >= window_lo) & (values <= window_hi)
    fraction = np.where(in_window, float(window_fraction), float(tail_fraction))
    rng = np.random.default_rng(int(rng_seed))
    u = rng.random(n)
    keep = u < fraction

    stacked = np.stack([*keys, in_window.astype(np.int64)], axis=1)
    _, stratum = np.unique(stacked, axis=0, return_inverse=True)
    stratum = stratum.reshape(-1)
    n_strata = int(stratum.max()) + 1
    sizes = np.bincount(stratum, minlength=n_strata)
    kept = np.bincount(stratum, weights=keep, minlength=n_strata).astype(np.int64)
    deficit = np.clip(np.minimum(int(cell_floor), sizes) - kept, 0, None)

    if deficit.any():
        # Top up each deficient stratum with its lowest-u unkept rows.
        candidates = np.flatnonzero(~keep & (deficit[stratum] > 0))
        order = candidates[np.lexsort((u[candidates], stratum[candidates]))]
        ordered_strata = stratum[order]
        first = np.searchsorted(ordered_strata, ordered_strata, side="left")
        rank = np.arange(len(order)) - first  # 0-based rank within stratum
        keep[order[rank < deficit[ordered_strata]]] = True
        kept = np.bincount(stratum, weights=keep, minlength=n_strata).astype(np.int64)

    base = sizes / np.maximum(kept, 1)
    keep_idx = np.flatnonzero(keep).astype(np.int64)
    return keep_idx, base[stratum[keep_idx]].astype(np.float64)
