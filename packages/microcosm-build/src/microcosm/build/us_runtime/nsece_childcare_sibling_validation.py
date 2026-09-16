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
    pooled=False,
    include_observed_children=False,
    pooling_fit_objective="pair_squared_error",
    composition=False,
):
    """Evaluate pair intensity and larger-household totals with disjoint training.

    A reserved split separates model development from the final comparison.
    Previously inspected source data do not become untouched external evidence
    merely because a new split is used.
    """
    children = source.children.loc[source.children.age.between(0, 12)].copy()
    if composition and not pooled:
        raise ValueError("Composition matching requires the pooled model.")
    if pooled or include_observed_children:
        from microcosm.build.us_runtime.nsece_childcare_pooling import (
            PooledScheduleDonors,
            fit_pooled_sibling_dependence,
            with_childcare_household_size,
        )

        children = with_childcare_household_size(children)
    if children.attendance_status.eq("summary_bridge").any():
        raise ValueError("Sibling validation requires original measured calendars.")
    levels = (match_columns, *fallback_match_columns)
    splits = _household_splits(
        children, seed=seed, validation_seed=validation_seed, partition=partition
    )
    pairs, large_pairs, larger = [], [], []
    child_records, observed_pairs = [], []
    observed_pair_households = 0
    month, days, hours = US_CHILDCARE_ATTENDANCE_COLUMNS
    del month
    fold_fits, split_counts, matching_counts, dependence_fits = [], [], {}, []
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
        fitted = (
            fit_pooled_sibling_dependence(
                training, objective=pooling_fit_objective, composition=composition
            )
            if pooled
            else fit_nsece_sibling_dependence(training, match_columns=match_columns)
        )
        rho = fitted["rho"]
        dependence_fits.append(fitted)
        fold_fits.append(rho)
        pooled_model = (
            PooledScheduleDonors(training, composition=composition) if pooled else None
        )
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
            pooled_model=pooled_model,
        ):
            if pooled_model is not None:
                matching_counts["partially_pooled"] = (
                    matching_counts.get("partially_pooled", 0) + 1
                )
                return pooled_model.distribution(child)
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
            complete_household = bool(household.attendance_status.eq("complete").all())
            if not include_observed_children and (
                len(household) < 2 or not complete_household
            ):
                continue
            roster_size = len(household)
            household = household.loc[household.attendance_status.eq("complete")]
            if household.empty:
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
            if include_observed_children:
                for i, (_, child) in enumerate(household.iterrows()):
                    child_records.append(
                        (
                            float(child.child_weight),
                            min(roster_size, 3),
                            complete_household,
                            actual[i],
                            means[i],
                            seconds[i],
                        )
                    )
                if len(household) >= 2:
                    observed_pair_households += 1
                    pair_weight = w[0] / (len(household) * (len(household) - 1) / 2)
                    for i, j in combinations(range(len(household)), 2):
                        independent_product = means[i] * means[j]
                        shared_product = _joint_product(
                            distributions[i][:2], distributions[j][:2]
                        )
                        observed_pairs.append(
                            (
                                pair_weight,
                                np.array(
                                    [
                                        actual[i],
                                        actual[j],
                                        actual[i] ** 2,
                                        actual[j] ** 2,
                                        actual[i] * actual[j],
                                    ]
                                ),
                                np.array(
                                    [
                                        means[i],
                                        means[j],
                                        seconds[i],
                                        seconds[j],
                                        independent_product,
                                    ]
                                ),
                                np.array(
                                    [
                                        means[i],
                                        means[j],
                                        seconds[i],
                                        seconds[j],
                                        (1 - rho) * independent_product
                                        + rho * shared_product,
                                    ]
                                ),
                            )
                        )
            if len(household) < 2 or not complete_household:
                continue
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
    if pooled:
        from microcosm.build.us_runtime.nsece_childcare_pooling import (
            COMPOSITION_CHILDCARE_LEVELS,
            POOLED_CHILDCARE_LEVELS,
            POOLED_CHILDCARE_STRENGTH,
        )

        pooling_levels = (
            COMPOSITION_CHILDCARE_LEVELS if composition else POOLED_CHILDCARE_LEVELS
        )
        result["model"] = {
            "name": "partially_pooled_empirical_schedules",
            "levels": pooling_levels,
            "composition": composition,
            "strength": POOLED_CHILDCARE_STRENGTH,
            "dependence_objective": pooling_fit_objective,
            "dependence_fits": dependence_fits,
        }
        result["match_columns"] = pooling_levels[-1]
        result["fallback_match_columns"] = ()
    if include_observed_children:
        result["matching_count_universe"] = (
            "every held-out child with an observed complete calendar"
        )
        result["observed_child_validation"] = _observed_child_summary(child_records)
        if observed_pairs:
            summary = _pair_summary(observed_pairs)
            summary["observed_pairs"] = summary.pop("households")
            summary["households"] = observed_pair_households
            summary["interpretation"] = (
                "all observed pairs including households with unresolved other siblings; household weight divided among observed pairs"
            )
            result["all_observed_sibling_pairs"] = summary
        else:
            result["all_observed_sibling_pairs"] = None
    return result


def _observed_child_summary(records):
    if not records:
        raise ValueError("Observed-child validation requires measured calendars.")
    outcomes = ("participation", "days", "weekly_hours")
    comparisons, checks = [], []
    for label, rows in (
        ("all", records),
        *(
            (f"household_size_{size}", [r for r in records if r[1] == size])
            for size in (1, 2, 3)
        ),
        ("complete_household", [r for r in records if r[2]]),
        ("unresolved_siblings", [r for r in records if not r[2]]),
    ):
        if not rows:
            comparisons.append({"group": label, "children": 0})
            checks.append(
                {"group": label, "passed": False, "reason": "no observed children"}
            )
            continue
        weights = np.array([r[0] for r in rows])
        actual = np.array([r[3] for r in rows])
        means = np.array([r[4] for r in rows])
        seconds = np.array([r[5] for r in rows])
        observed = np.average(actual, weights=weights, axis=0)
        expected = np.average(means, weights=weights, axis=0)
        comparisons.append(
            {
                "group": label,
                "children": len(rows),
                "child_weight": float(weights.sum()),
                "observed": dict(zip(outcomes, observed.tolist(), strict=True)),
                "expected": dict(zip(outcomes, expected.tolist(), strict=True)),
                "conditional_mean_squared_error": dict(
                    zip(
                        outcomes,
                        np.average(
                            (actual - means) ** 2, weights=weights, axis=0
                        ).tolist(),
                        strict=True,
                    )
                ),
                "expected_draw_squared_error": dict(
                    zip(
                        outcomes,
                        np.average(
                            actual**2 - 2 * actual * means + seconds,
                            weights=weights,
                            axis=0,
                        ).tolist(),
                        strict=True,
                    )
                ),
            }
        )
        for index, name in enumerate(outcomes):
            relative = index != 0
            gap = float(
                abs(expected[index] - observed[index])
                / (abs(observed[index]) if relative and observed[index] != 0 else 1)
            )
            defined = bool(not relative or observed[index] != 0)
            limit = 0.20 if relative else 0.05
            checks.append(
                {
                    "group": label,
                    "metric": name,
                    "relative": relative,
                    "gap": gap if defined else None,
                    "limit": limit,
                    "passed": defined and gap <= limit,
                }
            )
    return {
        "comparisons": comparisons,
        "screen": {"passed": all(c["passed"] for c in checks), "checks": checks},
        "interpretation": "child-weighted predictions of observed calendars; missing children are not scored as zeros; previously used survey, not external validation",
    }


def coupling_screen_compatibility(result):
    """Check whether changing dependence alone could satisfy BOTH joint screens.

    Independent and coupled arms have identical marginal first/second moments.
    Therefore correlation is affine in E[XY], regardless of the chosen copula.
    Recover that line from the two arms and intersect the existing correlation
    and cross-product screen intervals. Disjoint intervals prove those screens
    require a change to marginal distributions, not merely a different rho.
    This is conditional on the evaluated marginals, not an impossibility result
    for other models or a claim that selected households represent the population.
    """
    comparisons = []
    screens = {row["metric"]: row for row in result["diagnostic_screen"]["checks"]}
    for population in ("youngest_pairs", "youngest_pairs_in_3plus_households"):
        group = result.get(population)
        for metric in ("days", "weekly_hours"):
            entry = {"population": population, "metric": metric}
            comparisons.append(entry)
            if not group:
                entry.update({"identified": False, "reason": "no evaluated pairs"})
                continue
            observed = group["observed"][metric]
            independent = group["independent"][metric]
            coupled = group["coupled"][metric]
            correlations = [
                arm["correlation"] for arm in (observed, independent, coupled)
            ]
            if any(c is None or not np.isfinite(c) for c in correlations) or np.isclose(
                correlations[1], correlations[2]
            ):
                entry.update(
                    {
                        "identified": False,
                        "reason": "marginal scale not recoverable from the two arms",
                    }
                )
                continue
            scale = (coupled["joint_product"] - independent["joint_product"]) / (
                correlations[2] - correlations[1]
            )
            if not np.isfinite(scale) or scale <= 0:
                entry.update({"identified": False, "reason": "invalid marginal scale"})
                continue
            product_of_means = independent["joint_product"] - correlations[1] * scale
            # Consume the reported criteria so a screen change cannot leave
            # this structural diagnostic silently using a different threshold.
            correlation_limit = screens[f"{population}.{metric}.correlation"]["limit"]
            product_limit = screens[f"{population}.{metric}.joint_product"]["limit"]
            correlation_range = [
                max(-1, correlations[0] - correlation_limit),
                min(1, correlations[0] + correlation_limit),
            ]
            required_product = [
                product_of_means + value * scale for value in correlation_range
            ]
            allowed_product = [
                (1 - product_limit) * observed["joint_product"],
                (1 + product_limit) * observed["joint_product"],
            ]
            lower, upper = (
                max(required_product[0], allowed_product[0]),
                min(required_product[1], allowed_product[1]),
            )
            entry.update(
                {
                    "identified": True,
                    "predicted_product_of_marginal_means": product_of_means,
                    "predicted_product_of_marginal_standard_deviations": scale,
                    "joint_product_required_by_correlation_screen": required_product,
                    "joint_product_allowed_by_product_screen": allowed_product,
                    "screen_intervals_overlap": lower <= upper,
                    "overlap_interval": [lower, upper] if lower <= upper else None,
                }
            )
    return {
        "comparisons": comparisons,
        "interpretation": "disjoint screen intervals rule out a dependence-only fix for these conditional marginals; overlapping intervals do not establish that a feasible joint distribution exists",
    }


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
