"""UK firm universe: stratified thinning, registration propensity, VAT scope.

The UK uses of three country-neutral mechanics in ``microcosm.build.firms``,
which mirror the firm-microsim-paper generator (``firm_microsim`` at paper
commit aa1ce07):

* :func:`stratified_thin` mirrors ``generate.stratified_thin``;
* :func:`registration_propensity` mirrors ``calibration.registration_propensity``;
* :func:`assign_vat_scope_and_registration` mirrors ``generate.assign_vat_flags``
  together with its helper ``generate._select_to_weighted_target``.

Shared conventions
------------------
``band_idx`` is an integer array of HMRC VAT turnover-band indices, one per
row, using the paper's ``calibration.map_to_hmrc_bands`` layout::

    0 Negative_or_Zero      4 £300k_to_£500k
    1 £1_to_Threshold       5 £500k_to_£1m
    2 £Threshold_to_£150k   6 £1m_to_£10m
    3 £150k_to_£300k        7 Greater_than_£10m

``hmrc_bands`` is a one-dimensional float array of HMRC registered-trader
counts aligned with that index: ``hmrc_bands[b]`` is the HMRC count for band
``b``, so it has exactly :data:`N_HMRC_BANDS` (8) entries. Index 0 is accepted
but unused here, because appended negative/zero-turnover rows are HMRC traders
by construction (the paper's treatment).

``frame_mask`` is a boolean array, True for rows drawn from the ONS
VAT/PAYE enterprise frame and False for appended HMRC negative/zero-turnover
traders.

The VAT bands enter the shared functions through
:data:`UK_VAT_REGISTRATION_SPEC`: ``Negative_or_Zero`` is excluded (appended
traders live there, outside the frame), ``£1_to_Threshold`` is registrable and
the six bands above the threshold are selected.

The module does not depend on torch or on ``firm_generation``; callers convert
tensors to numpy before calling it.
"""

from __future__ import annotations

import numpy as np

from microcosm.build.firms import (
    RegistrationSpec,
    assign_scope_and_registration,
)
from microcosm.build.firms import (
    registration_propensity as _registration_propensity,
)
from microcosm.build.firms import (
    stratified_thin as _stratified_thin,
)

__all__ = [
    "HMRC_BAND_NAMES",
    "N_HMRC_BANDS",
    "UK_VAT_REGISTRATION_SPEC",
    "assign_vat_scope_and_registration",
    "registration_propensity",
    "stratified_thin",
]

#: HMRC VAT turnover bands, in ``band_idx`` order (paper ``map_to_hmrc_bands``).
HMRC_BAND_NAMES: tuple[str, ...] = (
    "Negative_or_Zero",
    "£1_to_Threshold",
    "£Threshold_to_£150k",
    "£150k_to_£300k",
    "£300k_to_£500k",
    "£500k_to_£1m",
    "£1m_to_£10m",
    "Greater_than_£10m",
)
N_HMRC_BANDS = len(HMRC_BAND_NAMES)

#: The HMRC VAT bands' registration roles (``microcosm.build.firms.registration``).
UK_VAT_REGISTRATION_SPEC = RegistrationSpec(
    HMRC_BAND_NAMES,
    ("excluded", "registrable", *("selected",) * (N_HMRC_BANDS - 2)),
)


def _as_1d(name: str, values: object, length: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} has {len(array)} entries; expected {length}")
    return array


def _hmrc_counts(hmrc_bands: object) -> np.ndarray:
    counts = _as_1d("hmrc_bands", hmrc_bands, N_HMRC_BANDS).astype(np.float64)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("hmrc_bands must hold finite, non-negative counts")
    return counts


def _band_indices(band_idx: object, length: int | None = None) -> np.ndarray:
    bands = _as_1d("band_idx", band_idx, length)
    if not np.issubdtype(bands.dtype, np.integer):
        raise TypeError("band_idx must be an integer array")
    if bands.size and (bands.min() < 0 or bands.max() >= N_HMRC_BANDS):
        raise ValueError(f"band_idx values must lie in [0, {N_HMRC_BANDS - 1}]")
    return bands.astype(np.int64)


def stratified_thin(
    turnover_k: np.ndarray,
    sic: np.ndarray,
    band_idx: np.ndarray,
    *,
    window_lo_k: float,
    window_hi_k: float,
    window_fraction: float,
    tail_fraction: float,
    cell_floor: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin a candidate firm draw by stratum and return exact base weights.

    Mirrors ``firm_microsim/generate.py::stratified_thin``. Rows whose
    turnover lies inside the analysis window ``[window_lo_k, window_hi_k]``
    (inclusive, £ thousands) are kept with probability ``window_fraction``;
    every other row is kept with probability ``tail_fraction``. Strata are
    SIC code x HMRC band x window side. A stratum that keeps fewer rows than
    ``min(cell_floor, stratum size)`` is topped up with its unkept rows of
    lowest uniform draw until it reaches that floor, so small strata survive.
    Each kept row's base weight is its stratum's size divided by its kept
    count, so the base-weighted count of every stratum equals its original
    row count (up to float64 rounding of ``kept * (size / kept)``).

    One uniform draw per row comes from ``np.random.default_rng(seed + 1)``,
    the paper's generator and offset. Because draws lie in ``[0, 1)``,
    fractions of 1.0 keep every row with base weight 1.0.

    Args:
        turnover_k: Float array, shape ``(n,)``: turnover in £ thousands.
        sic: Integer array, shape ``(n,)``: SIC code of each row.
        band_idx: Integer array, shape ``(n,)``: HMRC band index (0-7), as
            the paper computes from turnover and the VAT threshold.
        window_lo_k: Lower window edge (£ thousands, inclusive).
        window_hi_k: Upper window edge (£ thousands, inclusive).
        window_fraction: Keep probability inside the window, in ``[0, 1]``.
        tail_fraction: Keep probability outside the window, in ``[0, 1]``.
        cell_floor: Minimum kept rows per stratum (capped at stratum size).
        seed: Base seed; the generator is seeded with ``seed + 1``.

    Returns:
        ``(keep_idx, base_weights)``: ``keep_idx`` is an ascending int64 array
        of kept row positions; ``base_weights`` is a float64 array of the same
        length giving each kept row's base weight.
    """
    bands = _band_indices(band_idx, len(np.asarray(turnover_k)))
    return _stratified_thin(
        turnover_k,
        [sic, bands],
        window_lo=window_lo_k,
        window_hi=window_hi_k,
        window_fraction=window_fraction,
        tail_fraction=tail_fraction,
        cell_floor=cell_floor,
        rng_seed=int(seed) + 1,
    )


def registration_propensity(
    band_idx: np.ndarray,
    frame_mask: np.ndarray,
    base_weights: np.ndarray,
    hmrc_bands: np.ndarray,
) -> np.ndarray:
    """Per-row probability of being a VAT-registered trader, by HMRC band.

    Mirrors ``firm_microsim/calibration.py::registration_propensity``. For
    each positive-turnover band ``b`` (1-7) the propensity of frame rows in
    that band is ``hmrc_bands[b]`` divided by the band's base-weighted frame
    mass, capped at 1.0 when the frame cannot hold the HMRC population.

    * A band with zero frame mass is skipped, as in the paper: its frame rows
      (there are none with positive base weight) keep propensity 0.0.
    * Frame rows in band 0 (non-positive turnover) are not covered by any
      band loop in the paper and keep propensity 0.0.
    * Out-of-frame rows (``frame_mask`` False, appended HMRC negative/zero
      traders) receive propensity 1.0, as in the paper.

    Args:
        band_idx: Integer array, shape ``(n,)``: HMRC band index (0-7).
        frame_mask: Boolean array, shape ``(n,)``: True for ONS-frame rows.
        base_weights: Float array, shape ``(n,)``: base weight of each row.
        hmrc_bands: Float array, shape ``(8,)``: HMRC registered-trader count
            per band, aligned with ``band_idx``.

    Returns:
        Float64 array, shape ``(n,)``, of propensities in ``[0, 1]``.
    """
    return _registration_propensity(
        UK_VAT_REGISTRATION_SPEC,
        _band_indices(band_idx),
        frame_mask,
        base_weights,
        _hmrc_counts(hmrc_bands),
    )


def assign_vat_scope_and_registration(
    band_idx: np.ndarray,
    frame_mask: np.ndarray,
    weights: np.ndarray,
    hmrc_bands: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign VAT scope and registration flags matching HMRC trader counts.

    Mirrors ``firm_microsim/generate.py::assign_vat_flags``:

    * out-of-frame rows (``frame_mask`` False) are HMRC traders by
      construction: in scope and registered;
    * below the threshold (band 1), every frame row is in scope (registrable)
      and a seeded weighted selection whose cumulative weight reaches
      ``hmrc_bands[1]`` is registered;
    * above the threshold (bands 2-7), a seeded weighted selection per band
      reaching ``hmrc_bands[b]`` is in scope and registered; the rest of the
      band is out of scope (PAYE-only or exempt-sector enterprises);
    * frame rows in band 0 are neither in scope nor registered (the paper
      does not assign them).

    Each band's registered weight lies in ``[target, target + w)`` for the
    weight ``w`` of the last selected row, so it matches HMRC within one
    weight, unless the band's total weight is below the target, in which case
    every row in the band is selected.

    Selection uses one ``np.random.default_rng(seed)`` generator, drawing one
    permutation for band 1 and then one per band 2-7 in order. The paper
    draws with torch's global generator, so row-level selections differ from
    the paper; weighted totals do not.

    Args:
        band_idx: Integer array, shape ``(n,)``: HMRC band index (0-7).
        frame_mask: Boolean array, shape ``(n,)``: True for ONS-frame rows.
        weights: Float array, shape ``(n,)``: calibration (or base) weights.
        hmrc_bands: Float array, shape ``(8,)``: HMRC registered-trader count
            per band, aligned with ``band_idx``.
        seed: Seed for the selection generator.

    Returns:
        ``(scope, registered)``: boolean arrays of shape ``(n,)``.
    """
    return assign_scope_and_registration(
        UK_VAT_REGISTRATION_SPEC,
        _band_indices(band_idx),
        frame_mask,
        weights,
        _hmrc_counts(hmrc_bands),
        seed=seed,
    )
