"""Keyed nearest-donor choice within cells.

Some values are copied whole from a donor rather than modelled one by one, so
that combinations that occur together stay together: an industry division and
its sector, say. Each recipient takes one donor from its cell (the donors that
share its categorical keys), among the ``k`` nearest on a distance the caller
builds, chosen with probability proportional to the donor's weight at the
recipient's own uniform. A build passes uniforms keyed on stable identities, so
a recipient's donor does not depend on which other recipients are drawn with it.
A cell with too few donors falls back to the next, coarser cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

__all__ = ["donor_distance_features", "keyed_nearest_donors"]


def _reference_ranks(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    if len(ordered) == 0:
        return np.zeros(len(values))
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def donor_distance_features(
    recipients: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    rank_columns: Sequence[str] = (),
    presence_rank_columns: Sequence[str] = (),
    scaled_categoricals: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recipient and donor coordinates for :func:`keyed_nearest_donors`.

    - ``rank_columns``: each value's rank among the donors' values, in [0, 1].
    - ``presence_rank_columns``: for amounts that are often zero, whether the
      value is positive, and its rank among the donors' positive values
      (0 when it is not positive).
    - ``scaled_categoricals``: column -> scale; a one-hot on the donors'
      levels times the scale, so a level mismatch costs that much distance.

    Returns:
        ``(recipient_features, donor_features)``.
    """
    scaled = dict(scaled_categoricals or {})
    columns = [*rank_columns, *presence_rank_columns, *scaled]
    missing = [c for c in columns if c not in recipients or c not in donors]
    if missing:
        raise ValueError(
            f"Distance columns missing from recipients or donors: {missing}."
        )

    def coordinates(frame: pd.DataFrame) -> np.ndarray:
        parts = []
        for column in rank_columns:
            parts.append(
                _reference_ranks(
                    frame[column].to_numpy(np.float64),
                    donors[column].to_numpy(np.float64),
                )
            )
        for column in presence_rank_columns:
            values = frame[column].to_numpy(np.float64)
            reference = donors[column].to_numpy(np.float64)
            positive = values > 0
            parts.append(positive.astype(np.float64))
            parts.append(
                np.where(
                    positive, _reference_ranks(values, reference[reference > 0]), 0.0
                )
            )
        for column, scale in scaled.items():
            levels = np.unique(donors[column].to_numpy())
            one_hot = frame[column].to_numpy()[:, None] == levels[None, :]
            parts.extend(scale * one_hot.astype(np.float64).T)
        if not parts:
            raise ValueError("donor_distance_features needs at least one column.")
        return np.column_stack(parts)

    return coordinates(recipients), coordinates(donors)


def keyed_nearest_donors(
    recipient_features: np.ndarray,
    donor_features: np.ndarray,
    *,
    recipient_cells: Sequence[Sequence[object] | np.ndarray],
    donor_cells: Sequence[Sequence[object] | np.ndarray],
    donor_weights: np.ndarray,
    uniforms: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One donor per recipient: weighted, keyed, among its cell's ``k`` nearest.

    ``recipient_cells`` and ``donor_cells`` list the cell keys level by level,
    finest first (for example (major group, status group), then major group
    alone, then one cell for everyone). Keys are scalars: join several
    categorical keys into one string. A recipient uses the finest level whose
    cell holds at least ``k`` donors with positive weight; at the coarsest
    level any positive-weight donor will do. Within the cell it takes the
    ``k`` nearest donors (all, if fewer) and chooses one with probability
    proportional to weight, at its uniform.

    Returns:
        ``(donor_positions, levels)``: each recipient's donor, as a row
        position in the donor arrays, and the cell level it was drawn from.

    Raises:
        ValueError: If the inputs disagree in length, a uniform is outside
            ``[0, 1)``, ``k`` is below 1, or a recipient has no donor with
            positive weight at any level.
    """
    recipient_features = np.asarray(recipient_features, dtype=np.float64)
    donor_features = np.asarray(donor_features, dtype=np.float64)
    weights = np.asarray(donor_weights, dtype=np.float64)
    u = np.asarray(uniforms, dtype=np.float64)
    n, m = len(recipient_features), len(donor_features)
    if k < 1:
        raise ValueError("k must be at least 1.")
    if len(recipient_cells) != len(donor_cells) or not recipient_cells:
        raise ValueError("Recipient and donor cells need the same, non-empty levels.")
    if u.shape != (n,) or weights.shape != (m,):
        raise ValueError("uniforms and donor_weights need one value per row.")
    if not np.isfinite(u).all() or ((u < 0) | (u >= 1)).any():
        raise ValueError("Donor uniforms must be in [0, 1).")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Donor weights must be finite and non-negative.")
    positions = np.full(n, -1, dtype=np.int64)
    levels = np.full(n, -1, dtype=np.int64)
    usable = weights > 0
    last = len(recipient_cells) - 1
    for level, (r_keys, d_keys) in enumerate(
        zip(recipient_cells, donor_cells, strict=True)
    ):
        r_keys = np.asarray(r_keys, dtype=object)
        d_keys = np.asarray(d_keys, dtype=object)
        if r_keys.shape != (n,) or d_keys.shape != (m,):
            raise ValueError(f"Cell level {level} needs one key per row.")
        waiting = np.flatnonzero(positions < 0)
        if not len(waiting):
            break
        need = 1 if level == last else k
        # Positions into ``waiting``, grouped by each waiting recipient's key.
        for key, rows in pd.Series(waiting).groupby(r_keys[waiting]).indices.items():
            recipients = waiting[rows]
            pool = np.flatnonzero(usable & (d_keys == key))
            if len(pool) < need:
                continue
            count = min(k, len(pool))
            _, nearest = (
                NearestNeighbors(n_neighbors=count)
                .fit(donor_features[pool])
                .kneighbors(recipient_features[recipients])
            )
            cumulative = np.cumsum(weights[pool][nearest], axis=1)
            cumulative /= cumulative[:, -1:]
            pick = (cumulative > u[recipients, None]).argmax(axis=1)
            positions[recipients] = pool[nearest[np.arange(len(recipients)), pick]]
            levels[recipients] = level
    if (positions < 0).any():
        raise ValueError(
            "Some recipients have no donor with positive weight at any level."
        )
    return positions, levels
