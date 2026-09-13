"""Fit a household shared-rank mixture from measured sibling participation.

The mixture retains each child's weighted conditional donor distribution. With
probability rho, siblings use a shared uniform rank in care-sorted donor pools;
otherwise they draw independently. It interpolates the pair's joint positive
probability between p1*p2 and min(p1,p2). This models dependence, not shared
provider identity, and must be checked on household-held-out pairs.
"""

from __future__ import annotations

import numpy as np

from microcosm.build.us_runtime.nsece_childcare import NSECE_CHILDCARE_MATCH_COLUMNS


def complete_sibling_pairs(children):
    children = children.loc[children.age.between(0, 12)].copy()
    counts = children.groupby("source_household_id").attendance_status.agg(
        lambda x: len(x) >= 2 and x.eq("complete").all()
    )
    selected = children.source_household_id.isin(counts.index[counts])
    # One pair per household avoids giving large sibships disproportionate mass.
    return (
        children.loc[selected]
        .sort_values(["source_household_id", "age", "donor_id"])
        .groupby("source_household_id", sort=False)
        .head(2)
    )


def fit_nsece_sibling_dependence(children) -> dict:
    pool = children.loc[children.attendance_status.eq("complete")].copy()
    pool["weighted_care"] = (pool.childcare_days_per_week > 0) * pool.child_weight
    cells = pool.groupby(list(NSECE_CHILDCARE_MATCH_COLUMNS))[
        ["weighted_care", "child_weight"]
    ].sum()
    cells["probability"] = cells.weighted_care / cells.child_weight
    pairs = complete_sibling_pairs(children)
    if pairs.empty:
        return {"rho": 0.0, "households": 0, "status": "no measured sibling pairs"}
    predicted = (
        pairs[list(NSECE_CHILDCARE_MATCH_COLUMNS)]
        .merge(
            cells[["probability"]],
            left_on=list(NSECE_CHILDCARE_MATCH_COLUMNS),
            right_index=True,
            how="left",
            validate="many_to_one",
        )
        .probability.to_numpy()
        .reshape(-1, 2)
    )
    actual = (pairs.childcare_days_per_week.to_numpy() > 0).reshape(-1, 2)
    weights = pairs.household_weight.to_numpy().reshape(-1, 2)
    if (
        not np.isfinite(weights).all()
        or (weights <= 0).any()
        or not np.array_equal(weights[:, 0], weights[:, 1])
    ):
        raise ValueError(
            "Sibling dependence requires consistent positive household design weights."
        )
    weights = weights[:, 0]
    independent = predicted.prod(axis=1)
    shared = predicted.min(axis=1)
    observed = actual.all(axis=1)
    denominator = float((shared - independent) @ weights)
    unconstrained = (
        float((observed - independent) @ weights) / denominator if denominator else 0.0
    )
    return {
        "rho": float(np.clip(unconstrained, 0, 1)),
        "unconstrained_rho": unconstrained,
        "households": len(weights),
        "observed_both_in_care": float(np.average(observed, weights=weights)),
        "independent_both_in_care": float(np.average(independent, weights=weights)),
        "shared_both_in_care": float(np.average(shared, weights=weights)),
        "estimation": "youngest pair in fully observed households; household design weights",
    }
