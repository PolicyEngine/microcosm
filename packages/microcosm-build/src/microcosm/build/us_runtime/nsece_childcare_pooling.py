"""Experimental partial pooling of joint childcare schedule distributions.

Sparse cells borrow strength from broader cells, always retaining exact age.
Only measured complete calendars are donors; missing outcomes are not zeros.
This module does not change the production build or correct calendar selection.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
    _ids,
    _validate_attendance,
)

POOLED_CHILDCARE_MATCH_COLUMNS = (
    "age",
    "childcare_household_size",
    "parent_work_status",
    "income_band",
    "region",
)
POOLED_CHILDCARE_LEVELS = tuple(
    POOLED_CHILDCARE_MATCH_COLUMNS[:n]
    for n in range(1, len(POOLED_CHILDCARE_MATCH_COLUMNS) + 1)
)
POOLED_CHILDCARE_STRENGTH = 10.0
COMPOSITION_CHILDCARE_MATCH_COLUMNS = (
    "age",
    "childcare_household_size",
    "childcare_under6_count",
    "childcare_schoolage_count",
    "childcare_youngest_age_band",
    "childcare_has_younger_sibling",
    "parent_work_status",
    "income_band",
    "region",
)
COMPOSITION_CHILDCARE_LEVELS = tuple(
    COMPOSITION_CHILDCARE_MATCH_COLUMNS[:n]
    for n in range(1, len(COMPOSITION_CHILDCARE_MATCH_COLUMNS) + 1)
)


def with_childcare_household_size(children):
    """Count the full roster before selecting complete child calendars."""
    result = children.copy()
    _ids(result, "source_household_id", unique=False)
    age = result.age.to_numpy(dtype=float)
    if not np.isfinite(age).all() or (age < 0).any() or (age % 1 != 0).any():
        raise ValueError("Household-size predictors require finite whole-year ages.")
    result["childcare_household_size"] = (
        result.age.between(0, 12)
        .groupby(result.source_household_id)
        .transform("sum")
        .clip(upper=3)
    )
    for column, included in (
        ("childcare_under6_count", result.age.between(0, 5)),
        ("childcare_schoolage_count", result.age.between(6, 12)),
    ):
        result[column] = (
            included.groupby(result.source_household_id).transform("sum").clip(upper=3)
        )
    youngest = (
        result.age.where(result.age.between(0, 12))
        .groupby(result.source_household_id)
        .transform("min")
    )
    result["childcare_youngest_age_band"] = np.select(
        [youngest < 3, youngest < 6], [0, 1], default=2
    )
    result["childcare_has_younger_sibling"] = (result.age > youngest).astype(int)
    return result


class PooledScheduleDonors:
    """Weighted mixture of nested empirical donor distributions.

    The support statistic uses households as clusters, not child-row counts.
    Multiplying all survey weights by a common factor cannot change predictions.
    """

    def __init__(
        self, children, *, strength=POOLED_CHILDCARE_STRENGTH, composition=False
    ):
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("Pooling strength must be finite and positive.")
        self.strength = strength
        self.levels = (
            COMPOSITION_CHILDCARE_LEVELS if composition else POOLED_CHILDCARE_LEVELS
        )
        self.match_columns = self.levels[-1]
        required = [
            *self.match_columns,
            "source_household_id",
            "donor_id",
            "child_weight",
            "attendance_status",
            *US_CHILDCARE_ATTENDANCE_COLUMNS,
        ]
        if not set(required).issubset(children):
            raise ValueError("Pooled childcare donors need complete source predictors.")
        if children.attendance_status.eq("summary_bridge").any():
            raise ValueError("Experimental pooled donors require original calendars.")
        self.pool = children.loc[
            children.attendance_status.eq("complete") & children.age.between(0, 12)
        ].copy()
        if self.pool.empty or self.pool.reindex(columns=required).isna().any().any():
            raise ValueError("Pooled childcare donor fields must be complete.")
        if self.pool.donor_id.duplicated().any():
            raise ValueError("Pooled childcare donor IDs must be unique.")
        _ids(self.pool, "source_household_id", unique=False)
        _ids(self.pool, "donor_id", unique=True)
        predictors = self.pool.reindex(columns=self.match_columns).to_numpy(dtype=float)
        if (
            not np.isfinite(predictors).all()
            or (predictors % 1 != 0).any()
            or not self.pool.childcare_household_size.isin([1, 2, 3]).all()
        ):
            raise ValueError(
                "Pooled childcare predictors must be finite categorical values."
            )
        weights = self.pool.child_weight.to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Pooled childcare weights must be finite and positive.")
        _validate_attendance(self.pool, complete=True)
        _, days, hours = US_CHILDCARE_ATTENDANCE_COLUMNS
        self.pool = self.pool.sort_values([days, hours, "donor_id"]).reset_index(
            drop=True
        )
        self.values = np.column_stack(
            (self.pool[days] > 0, self.pool[days], self.pool[days] * self.pool[hours])
        ).astype(float)
        self.weights = self.pool.child_weight.to_numpy(dtype=float)
        self.households = self.pool.source_household_id.to_numpy()
        self.groups = [
            self.pool.groupby(list(level), sort=False).indices for level in self.levels
        ]
        self.cache = {}

    def distribution(self, child, *, exclude_household=None):
        """Return care-sorted values, cumulative probabilities and row masses."""
        key = tuple(child.reindex(self.match_columns))
        if not np.isfinite(np.asarray(key, dtype=float)).all():
            raise ValueError("Pooled target matching fields must be complete.")
        if exclude_household is None and key in self.cache:
            return self.cache[key]
        age_indices = np.asarray(self.groups[0].get(key[0], []), dtype=int)
        if exclude_household is not None:
            age_indices = age_indices[self.households[age_indices] != exclude_household]
        if not len(age_indices):
            raise ValueError("No exact-age donor household remains after exclusion.")
        probabilities = self.weights[age_indices].copy()
        probabilities /= probabilities.sum()
        for level, groups in zip(self.levels[1:], self.groups[1:], strict=True):
            indices = np.asarray(groups.get(key[: len(level)], []), dtype=int)
            if exclude_household is not None:
                indices = indices[self.households[indices] != exclude_household]
            if not len(indices):
                continue
            # Both arrays retain global care order, so local indices embed in
            # the exact-age root without a second sort or a donor cross-join.
            positions = np.searchsorted(age_indices, indices)
            weights = self.weights[indices]
            _, cluster = np.unique(self.households[indices], return_inverse=True)
            cluster_weights = np.bincount(cluster, weights=weights)
            effective = cluster_weights.sum() ** 2 / (cluster_weights @ cluster_weights)
            fraction = effective / (effective + self.strength)
            probabilities *= 1 - fraction
            probabilities[positions] += fraction * weights / weights.sum()
        probabilities /= probabilities.sum()
        cumulative = probabilities.cumsum()
        cumulative[-1] = 1.0
        result = (self.values[age_indices], cumulative, probabilities)
        if exclude_household is None:
            self.cache[key] = result
        return result


def fit_pooled_sibling_dependence(
    children,
    *,
    strength=POOLED_CHILDCARE_STRENGTH,
    objective="pair_squared_error",
    composition=False,
):
    """Fit all measured pair cross-products with whole-household donor exclusion."""
    from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
        _joint_product,
    )

    if objective not in ("pair_squared_error", "population_moments"):
        raise ValueError("Unknown pooled dependence objective.")
    model = PooledScheduleDonors(children, strength=strength, composition=composition)
    records = []
    households = 0
    incomplete_households = 0
    for household_id, roster in children.loc[children.age.between(0, 12)].groupby(
        "source_household_id", sort=True
    ):
        observed = roster.loc[roster.attendance_status.eq("complete")].sort_values(
            ["age", "donor_id"]
        )
        if len(observed) < 2:
            continue
        weight = roster.household_weight.to_numpy(dtype=float)
        if (
            not np.isfinite(weight).all()
            or (weight <= 0).any()
            or not (weight == weight[0]).all()
        ):
            raise ValueError("Pooled dependence requires consistent household weights.")
        distributions = [
            model.distribution(row, exclude_household=household_id)
            for _, row in observed.iterrows()
        ]
        means = [np.average(d[0], weights=d[2], axis=0) for d in distributions]
        _, days, hours = US_CHILDCARE_ATTENDANCE_COLUMNS
        actual = np.column_stack(
            (observed[days] > 0, observed[days], observed[days] * observed[hours])
        ).astype(float)
        pair_count = len(observed) * (len(observed) - 1) // 2
        for i, j in combinations(range(len(observed)), 2):
            independent = means[i] * means[j]
            shared = _joint_product(distributions[i][:2], distributions[j][:2])
            records.append(
                (
                    weight[0] / pair_count,
                    actual[i] * actual[j] - independent,
                    shared - independent,
                    (actual[i] ** 2 + actual[j] ** 2) / 2,
                )
            )
        households += 1
        incomplete_households += int(len(observed) != len(roster))
    if not records:
        raise ValueError("Pooled dependence requires observed sibling pairs.")
    weights = np.array([r[0] for r in records])
    residual = np.array([r[1] for r in records])
    direction = np.array([r[2] for r in records])
    scales = np.average(np.array([r[3] for r in records]), weights=weights, axis=0)
    active = scales > 0
    residual = residual[:, active] / scales[active]
    direction = direction[:, active] / scales[active]
    numerator = np.sum(weights[:, None] * residual * direction)
    denominator = np.sum(weights[:, None] * direction**2)
    mean_residual = np.average(residual, weights=weights, axis=0)
    mean_direction = np.average(direction, weights=weights, axis=0)
    if objective == "population_moments":
        numerator = mean_residual @ mean_direction
        denominator = mean_direction @ mean_direction
    unconstrained = float(numerator / denominator) if denominator > 0 else 0.0
    return {
        "rho": float(np.clip(unconstrained, 0, 1)),
        "unconstrained_rho": unconstrained,
        "households": households,
        "households_with_unresolved_siblings": incomplete_households,
        "observed_pairs": len(records),
        "outcome_second_moments": scales.tolist(),
        "objective": objective,
        "normalized_mean_residual": mean_residual.tolist(),
        "normalized_mean_shared_increment": mean_direction.tolist(),
        "donor_household_overlap": 0,
        "estimation": "least squares on participation/days/weekly-hours cross-products under the named objective; all observed pairs; household weight divided among pairs; leave whole fitting household out of donors",
    }
