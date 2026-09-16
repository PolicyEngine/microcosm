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
    NSECEChildcareSource,
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


def _household_splits(children, *, seed, validation_seed, partition):
    """Keep the reserved households outside every development fit and donor pool."""
    if partition not in ("all", "development", "validation"):
        raise ValueError("Unknown sibling assessment partition.")
    if (partition == "all") != (validation_seed is None):
        raise ValueError("A reserved assessment needs a validation seed and partition.")
    households = children.source_household_id
    reserved = (
        households.map(
            lambda x: (
                int.from_bytes(
                    hashlib.sha256(f"{validation_seed}:reserved:{x}".encode()).digest()[
                        :8
                    ],
                    "big",
                )
                % 5
                == 0
            )
        )
        if validation_seed is not None
        else pd.Series(False, index=children.index)
    )
    if partition == "validation":
        return [(~reserved, reserved)]
    development = ~reserved
    folds = households.map(
        lambda x: (
            int.from_bytes(hashlib.sha256(f"{seed}:{x}".encode()).digest()[:8], "big")
            % 5
        )
    )
    return [
        (development & (folds != fold), development & (folds == fold))
        for fold in range(5)
    ]


def assess_sibling_schedules(
    source,
    *,
    seed=271828,
    match_columns=NSECE_CHILDCARE_MATCH_COLUMNS,
    fallback_match_columns=NSECE_CHILDCARE_FALLBACK_COLUMNS,
    validation_seed=None,
    partition="all",
):
    """Evaluate pair intensity and larger-household totals with disjoint training.

    A reserved split separates model development from the final comparison.
    Previously inspected source data do not become untouched external evidence
    merely because a new split is used.
    """
    children = source.children.loc[source.children.age.between(0, 12)].copy()
    if children.attendance_status.eq("summary_bridge").any():
        raise ValueError("Sibling validation requires original measured calendars.")
    levels = (match_columns, *fallback_match_columns)
    splits = _household_splits(
        children, seed=seed, validation_seed=validation_seed, partition=partition
    )
    pairs, large_pairs, larger = [], [], []
    month, days, hours = US_CHILDCARE_ATTENDANCE_COLUMNS
    del month
    fold_fits, split_counts, matching_counts = [], [], {}
    for training_mask, target_mask in splits:
        training = children.loc[training_mask]
        target = children.loc[target_mask]
        if training.empty or target.empty:
            raise ValueError(
                "Sibling assessment requires nonempty household partitions."
            )
        if not set(training.source_household_id).isdisjoint(target.source_household_id):
            raise ValueError("Sibling assessment leaks households across partitions.")
        split_counts.append(
            {
                "training_households": int(training.source_household_id.nunique()),
                "evaluation_households": int(target.source_household_id.nunique()),
                "household_overlap": 0,
            }
        )
        rho = fit_nsece_sibling_dependence(training, match_columns=match_columns)["rho"]
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
                cache[key] = (values, cumulative, probabilities, ",".join(level))
            *distribution_values, used_level = cache[key]
            matching_counts[used_level] = matching_counts.get(used_level, 0) + 1
            return distribution_values

        for _, household in target.groupby("source_household_id"):
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
        "design": "household-separated partitions; exact integration of the implemented shared-rank donor CDFs",
        "seed": seed,
        "validation_seed": validation_seed,
        "partition": partition,
        "splits": split_counts,
        "match_columns": match_columns,
        "fallback_match_columns": fallback_match_columns,
        "matching_counts": matching_counts,
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


def compare_household_size_matching(source, *, partition, validation_seed=20260916):
    """Evaluate one size-conditioned challenger without changing the build recipe.

    Count every rostered under-13 child before attendance selection. The
    selection diagnostic uses only development households even when the
    caller requests the separately reserved final comparison.
    """
    if partition not in ("development", "validation"):
        raise ValueError("Household-size comparison requires a reserved partition.")
    children = source.children.copy()
    children["childcare_household_size"] = (
        children.age.between(0, 12)
        .groupby(children.source_household_id)
        .transform("sum")
        .clip(upper=3)
    )
    revised_source = NSECEChildcareSource(
        children, source.weights, source.source_receipt
    )
    columns = ("age", "childcare_household_size", *NSECE_CHILDCARE_MATCH_COLUMNS[1:])
    fallbacks = tuple(
        ("age", "childcare_household_size", *level[1:])
        for level in NSECE_CHILDCARE_FALLBACK_COLUMNS
    ) + (("age",),)
    settings = {"partition": partition, "validation_seed": validation_seed}
    baseline = assess_sibling_schedules(source, **settings)
    challenger = assess_sibling_schedules(
        revised_source,
        match_columns=columns,
        fallback_match_columns=fallbacks,
        **settings,
    )
    return {
        "source": source.source_receipt,
        "partition": partition,
        "validation_seed": validation_seed,
        "legacy": baseline,
        "household_size": challenger,
        "development_selection": _household_selection_diagnostic(
            children, match_columns=columns, validation_seed=validation_seed
        ),
        "production_recipe_changed": False,
        "production_ready": False,
        "interpretation": (
            "Prospective internal comparison on previously inspected survey data; "
            "not untouched external acceptance evidence. Complete-household totals "
            "do not identify totals for families with missing calendars."
        ),
    }


def _household_selection_diagnostic(children, *, match_columns, validation_seed):
    children = children.loc[children.age.between(0, 12)]
    development, _ = _household_splits(
        children, seed=271828, validation_seed=validation_seed, partition="validation"
    )[0]
    children = children.loc[development].copy()
    children["complete_household"] = children.groupby(
        "source_household_id"
    ).attendance_status.transform(lambda status: status.eq("complete").all())
    records = []
    complete = children.loc[children.attendance_status.eq("complete")]
    for size, group in complete.groupby("childcare_household_size"):
        for sample, rows in (
            ("all_observed_children", group),
            ("children_in_complete_households", group.loc[group.complete_household]),
            (
                "observed_children_with_unresolved_siblings",
                group.loc[~group.complete_household],
            ),
        ):
            if rows.empty:
                continue
            weights = rows.child_weight
            records.append(
                {
                    "household_size_capped_at_3": int(size),
                    "sample": sample,
                    "children": len(rows),
                    "households": int(rows.source_household_id.nunique()),
                    "child_weight": float(weights.sum()),
                    "mean_days": float(
                        np.average(rows.childcare_days_per_week, weights=weights)
                    ),
                    "mean_weekly_hours": float(
                        np.average(
                            rows.childcare_days_per_week * rows.childcare_hours_per_day,
                            weights=weights,
                        )
                    ),
                }
            )
    support = {}
    for name, columns in (
        ("legacy", NSECE_CHILDCARE_MATCH_COLUMNS),
        ("household_size", match_columns),
    ):
        counts = complete.groupby(list(columns)).source_household_id.transform(
            "nunique"
        )
        support[name] = {
            "children": len(complete),
            "median_donor_households": float(counts.median()),
            "children_in_cells_below_10_households": int((counts < 10).sum()),
            "fraction_of_children_in_cells_below_10_households": float(
                (counts < 10).mean()
            ),
            "definition": "Exact matching cells in the whole development pool; fold-training support can be smaller. Ten is a descriptive cutoff, not a tuned fallback rule.",
        }
    return {
        "partition": "development only",
        "observed_child_comparisons": records,
        "donor_support": support,
        "interpretation": "Descriptive differences under calendar selection; not a causal effect of missingness and not corrected population totals.",
    }
