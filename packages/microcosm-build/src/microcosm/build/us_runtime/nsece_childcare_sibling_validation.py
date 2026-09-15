"""Held-out joint schedule moments for the actual household rank mixture.

Integrate the finite donor CDFs exactly, rather than selecting a favorable
simulation seed. Whole households are held out, including all larger sibships.
"""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
)
from microcosm.build.us_runtime.nsece_childcare import (
    NSECE_CHILDCARE_FALLBACK_COLUMNS,
    NSECE_CHILDCARE_MATCH_COLUMNS,
)
from microcosm.build.us_runtime.nsece_childcare_dependence import (
    fit_nsece_sibling_dependence,
)


def _joint_product(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
):
    """E[XY] under one shared uniform, with arbitrary weighted discrete CDFs."""
    a, pa = first
    b, pb = second
    cuts = np.unique(np.r_[0, pa, pb, 1])
    midpoints = (cuts[:-1] + cuts[1:]) / 2
    x = a[np.minimum(np.searchsorted(pa, midpoints, side="right"), len(a) - 1)]
    y = b[np.minimum(np.searchsorted(pb, midpoints, side="right"), len(b) - 1)]
    return np.sum(x * y * np.diff(cuts)[:, None], axis=0)


def _moments(values, weights):
    return np.average(values, weights=weights, axis=0)


def _pair_summary(records):
    weights = np.array([r[0] for r in records])
    result = {}
    for arm, index in (("observed", 1), ("independent", 2), ("coupled", 3)):
        moments = _moments(np.array([r[index] for r in records]), weights)
        metrics = {}
        for j, name in enumerate(("participation", "days", "weekly_hours")):
            x, y, xx, yy, xy = moments[:, j]
            denominator = np.sqrt(max(0, xx - x * x) * max(0, yy - y * y))
            metrics[name] = {
                "joint_product": float(xy),
                "correlation": float((xy - x * y) / denominator)
                if denominator > 0
                else None,
            }
        result[arm] = metrics
    return {
        "households": len(records),
        "household_weight": float(weights.sum()),
        **result,
    }


def assess_sibling_schedules(source, *, seed=271828):
    """Evaluate youngest-pair days/hours and all-child totals for 3+ households."""
    children = source.children.loc[source.children.age.between(0, 12)].copy()
    match_columns = NSECE_CHILDCARE_MATCH_COLUMNS
    levels = (match_columns, *NSECE_CHILDCARE_FALLBACK_COLUMNS)
    folds = children.source_household_id.map(
        lambda x: (
            int.from_bytes(hashlib.sha256(f"{seed}:{x}".encode()).digest()[:8], "big")
            % 5
        )
    )
    pairs, large_pairs, larger = [], [], []
    month, days, hours = US_CHILDCARE_ATTENDANCE_COLUMNS
    del month
    fold_fits = []
    for fold in range(5):
        training = children.loc[folds != fold]
        rho = fit_nsece_sibling_dependence(training)["rho"]
        fold_fits.append(rho)
        train = training.loc[training.attendance_status.eq("complete")].sort_values(
            [days, hours, "donor_id"]
        )
        groups = [
            (level, train.groupby(list(level), sort=False).indices) for level in levels
        ]
        cache = {}

        def distribution(
            child: pd.Series,
            cache: dict = cache,
            groups: list = groups,
            train: pd.DataFrame = train,
        ):
            key = tuple(child.reindex(match_columns))
            if key not in cache:
                for level, indices in groups:
                    k = tuple(child.reindex(level))
                    pool = train.iloc[indices.get(k[0] if len(k) == 1 else k, [])]
                    pool = pool.loc[pool.child_weight > 0]
                    if len(pool):
                        break
                if pool.empty:
                    raise ValueError("Unsupported held-out sibling matching cell.")
                probabilities = pool.child_weight.to_numpy(dtype=float, copy=True)
                probabilities /= probabilities.sum()
                values = np.column_stack(
                    (pool[days] > 0, pool[days], pool[days] * pool[hours])
                ).astype(float)
                cumulative = probabilities.cumsum()
                cumulative[-1] = 1
                cache[key] = (values, cumulative, probabilities)
            return cache[key]

        for _, household in children.loc[folds == fold].groupby("source_household_id"):
            if (
                len(household) < 2
                or not household.attendance_status.eq("complete").all()
            ):
                continue
            household = household.sort_values(["age", "donor_id"])
            w = household.household_weight.to_numpy(dtype=float)
            if not np.isfinite(w).all() or (w <= 0).any() or not (w == w[0]).all():
                raise ValueError(
                    "Sibling assessment requires consistent household design weights."
                )
            distributions = [distribution(row) for _, row in household.iterrows()]
            means = np.array([_moments(d[0], d[2]) for d in distributions])
            seconds = np.array([_moments(d[0] ** 2, d[2]) for d in distributions])
            actual = np.column_stack(
                (
                    household[days] > 0,
                    household[days],
                    household[days] * household[hours],
                )
            ).astype(float)
            joint = _joint_product(distributions[0][:2], distributions[1][:2])
            independent = means[0] * means[1]
            observed = np.array(
                [
                    actual[0],
                    actual[1],
                    actual[0] ** 2,
                    actual[1] ** 2,
                    actual[0] * actual[1],
                ]
            )
            uncoupled = np.array(
                [means[0], means[1], seconds[0], seconds[1], independent]
            )
            coupled = np.array(
                [
                    means[0],
                    means[1],
                    seconds[0],
                    seconds[1],
                    (1 - rho) * independent + rho * joint,
                ]
            )
            record = (w[0], observed, uncoupled, coupled)
            pairs.append(record)
            if len(household) < 3:
                continue
            large_pairs.append(record)
            sums = means.sum(axis=0)
            sums_squared = {
                "independent": seconds.sum(axis=0).copy(),
                "coupled": seconds.sum(axis=0).copy(),
            }
            for i, j in combinations(range(len(household)), 2):
                independent = means[i] * means[j]
                joint = _joint_product(distributions[i][:2], distributions[j][:2])
                sums_squared["independent"] += 2 * independent
                sums_squared["coupled"] += 2 * ((1 - rho) * independent + rho * joint)
            independent_all = means[:, 0].prod()
            larger.append(
                (
                    w[0],
                    {
                        "observed": np.r_[
                            actual.sum(axis=0),
                            actual.sum(axis=0) ** 2,
                            actual[:, 0].prod(),
                        ],
                        "independent": np.r_[
                            sums, sums_squared["independent"], independent_all
                        ],
                        "coupled": np.r_[
                            sums,
                            sums_squared["coupled"],
                            (1 - rho) * independent_all + rho * means[:, 0].min(),
                        ],
                    },
                )
            )
    result = {
        "design": "five household-separated folds; exact integration of the implemented shared-rank donor CDFs",
        "seed": seed,
        "match_columns": match_columns,
        "training_rho": fold_fits,
        "youngest_pairs": _pair_summary(pairs) if pairs else None,
        "youngest_pairs_in_3plus_households": _pair_summary(large_pairs)
        if large_pairs
        else None,
        "larger_households": {"households": len(larger)},
        "production_ready": False,
    }
    if larger:
        weights = np.array([r[0] for r in larger])
        for arm in ("observed", "independent", "coupled"):
            values = _moments(np.array([r[1][arm] for r in larger]), weights)
            result["larger_households"][arm] = {
                "mean_total_days": float(values[1]),
                "mean_total_weekly_hours": float(values[2]),
                "sd_total_days": float(np.sqrt(max(0, values[4] - values[1] ** 2))),
                "sd_total_weekly_hours": float(
                    np.sqrt(max(0, values[5] - values[2] ** 2))
                ),
                "all_children_attend": float(values[6]),
            }
    result["diagnostic_screen"] = sibling_schedule_screen(result)
    return result


def sibling_schedule_screen(result):
    """Developmental screens declared before this expanded assessment was run.

    Passing would not certify the survey transport or authorize publication.
    Undefined metrics and empty subgroups are failures, not silent passes.
    """
    checks = []

    def check(name, observed, predicted, limit, relative=False):
        gap = None
        if observed is not None and predicted is not None:
            if not relative or observed != 0:
                gap = abs(predicted - observed) / (abs(observed) if relative else 1)
        checks.append(
            {
                "metric": name,
                "absolute_gap": gap,
                "relative": relative,
                "limit": limit,
                "passed": gap is not None and gap <= limit,
            }
        )

    for name in ("youngest_pairs", "youngest_pairs_in_3plus_households"):
        values = result.get(name) or {}
        for metric, moment, limit, relative in (
            ("participation", "joint_product", 0.05, False),
            ("days", "correlation", 0.10, False),
            ("weekly_hours", "correlation", 0.10, False),
            ("days", "joint_product", 0.20, True),
            ("weekly_hours", "joint_product", 0.20, True),
        ):
            observed = values.get("observed", {}).get(metric, {}).get(moment)
            predicted = values.get("coupled", {}).get(metric, {}).get(moment)
            check(f"{name}.{metric}.{moment}", observed, predicted, limit, relative)
    larger = result["larger_households"]
    for metric in (
        "mean_total_days",
        "mean_total_weekly_hours",
        "sd_total_days",
        "sd_total_weekly_hours",
        "all_children_attend",
    ):
        relative = metric != "all_children_attend"
        check(
            f"larger_households.{metric}",
            larger.get("observed", {}).get(metric),
            larger.get("coupled", {}).get(metric),
            0.20 if relative else 0.05,
            relative,
        )
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "interpretation": "provisional diagnostic screens; not publication authorization",
    }
