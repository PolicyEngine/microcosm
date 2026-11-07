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

from microcosm.fit.model import duplicated_column_labels

__all__ = ["donor_distance_features", "keyed_nearest_donors"]

#: What :func:`donor_distance_features` does with a recipient category no
#: donor holds; the rules :class:`~microcosm.fit.CategoricalClassifier` names.
_UNSEEN_CATEGORY_RULES = ("refuse", "ignore")


def _reference_ranks(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    if len(ordered) == 0:
        return np.zeros(len(values))
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def _finite_values(frame: pd.DataFrame, column: str, side: str) -> np.ndarray:
    """A rank column as floats, refusing a missing or non-finite value by name."""
    values = frame[column].to_numpy(np.float64, na_value=np.nan)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise ValueError(
            f"Distance column {column!r} has {bad} missing or non-finite "
            f"value(s) among the {side}. A rank cannot place them (a missing "
            "recipient value would rank above every donor); fill or drop them "
            "before building distances."
        )
    return values


def _category_values(frame: pd.DataFrame, column: str, side: str) -> np.ndarray:
    """A categorical column's values, refusing a missing one by name."""
    values = frame[column].to_numpy()
    absent = int(pd.isna(values).sum())
    if absent:
        raise ValueError(
            f"Categorical distance column {column!r} has {absent} missing "
            f"value(s) among the {side}, and a missing value matches no level. "
            "Give missing values a level of their own, or drop those rows."
        )
    return values


def donor_distance_features(
    recipients: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    rank_columns: Sequence[str] = (),
    presence_rank_columns: Sequence[str] = (),
    scaled_categoricals: Mapping[str, float] | None = None,
    unseen_categories: str = "refuse",
) -> tuple[np.ndarray, np.ndarray]:
    """Recipient and donor coordinates for :func:`keyed_nearest_donors`.

    - ``rank_columns``: each value's rank among the donors' values, in [0, 1].
    - ``presence_rank_columns``: for amounts that are often zero, whether the
      value is positive, and its rank among the donors' positive values
      (0 when it is not positive).
    - ``scaled_categoricals``: column -> scale; a one-hot on the donors'
      levels times the scale, so a level mismatch costs that much distance.

    Every rank and presence-rank value must be finite, and no categorical
    value may be missing, among recipients and donors alike: a rank has no
    place for a missing value. ``unseen_categories`` says what happens to a
    recipient whose category no donor holds. ``"refuse"`` (the default, as in
    :class:`~microcosm.fit.CategoricalClassifier`) raises, naming the column
    and the categories. ``"ignore"`` sets none of that column's indicators,
    which puts the recipient ``scale`` from every donor on that column: nearer
    than a recipient of another seen level, at ``scale * sqrt(2)``.

    Returns:
        ``(recipient_features, donor_features)``.

    Raises:
        ValueError: If ``unseen_categories`` is neither ``"refuse"`` nor
            ``"ignore"``, a column is missing from either table or names more
            than one of its columns, a rank or presence-rank value is missing
            or non-finite, a categorical value is missing, a recipient holds a
            category no donor holds under ``"refuse"``, or the call names no
            column.
    """
    if unseen_categories not in _UNSEEN_CATEGORY_RULES:
        raise ValueError(
            f"unseen_categories must be one of {list(_UNSEEN_CATEGORY_RULES)}, "
            f"got {unseen_categories!r}."
        )
    scaled = dict(scaled_categoricals or {})
    columns = [*rank_columns, *presence_rank_columns, *scaled]
    missing = [c for c in columns if c not in recipients or c not in donors]
    if missing:
        raise ValueError(
            f"Distance columns missing from recipients or donors: {missing}."
        )
    tables = {"recipients": recipients, "donors": donors}
    for side, frame in tables.items():
        repeated = duplicated_column_labels(frame, columns)
        if repeated:
            raise ValueError(
                f"Distance column label(s) {repeated} each name more than one "
                f"column among the {side}; drop or rename the duplicates."
            )
    numeric = list(dict.fromkeys([*rank_columns, *presence_rank_columns]))
    numbers = {
        side: {column: _finite_values(frame, column, side) for column in numeric}
        for side, frame in tables.items()
    }
    categories = {
        side: {column: _category_values(frame, column, side) for column in scaled}
        for side, frame in tables.items()
    }
    levels = {column: np.unique(categories["donors"][column]) for column in scaled}
    indicators = {
        side: {
            column: values[:, None] == levels[column][None, :]
            for column, values in categories[side].items()
        }
        for side in tables
    }
    if unseen_categories == "refuse":
        for column, held in indicators["recipients"].items():
            seen = held.any(axis=1)
            if not seen.all():
                unseen = sorted(
                    {str(v) for v in categories["recipients"][column][~seen]}
                )
                raise ValueError(
                    f"Categorical distance column {column!r}: {int((~seen).sum())} "
                    f"recipient row(s) hold categories no donor holds: "
                    f"{unseen[:5]}. Pass unseen_categories='ignore' to place them "
                    "the same distance from every donor."
                )

    def coordinates(side: str) -> np.ndarray:
        parts = []
        for column in rank_columns:
            parts.append(
                _reference_ranks(numbers[side][column], numbers["donors"][column])
            )
        for column in presence_rank_columns:
            values = numbers[side][column]
            reference = numbers["donors"][column]
            positive = values > 0
            parts.append(positive.astype(np.float64))
            parts.append(
                np.where(
                    positive, _reference_ranks(values, reference[reference > 0]), 0.0
                )
            )
        for column, scale in scaled.items():
            parts.extend(scale * indicators[side][column].astype(np.float64).T)
        if not parts:
            raise ValueError("donor_distance_features needs at least one column.")
        return np.column_stack(parts)

    return coordinates("recipients"), coordinates("donors")


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
