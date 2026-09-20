"""Household cross-validation and questionnaire transport checks for attendance.

These checks expose selection and sparse support. Conditional donor expectations
are used for prediction scoring so a single stochastic draw cannot hide bias.
They do not identify unobserved attendance under nonrandom calendar missingness.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.nsece_childcare import (
    NSECE_CHILDCARE_FALLBACK_COLUMNS,
    NSECE_CHILDCARE_MATCH_COLUMNS,
    NSECEChildcareSource,
)
from microcosm.build.us_runtime.nsece_childcare_bridge import (
    bridge_nsece_noncalendar_attendance,
)
from microcosm.build.us_runtime.nsece_childcare_dependence import (
    complete_sibling_pairs,
    fit_nsece_sibling_dependence,
)


def _conditional_expectations(train, target, outcomes):
    """Weighted empirical expectation with a declared, age-preserving hierarchy."""
    result = pd.DataFrame(np.nan, index=target.index, columns=outcomes)
    levels = pd.Series("unsupported", index=target.index, dtype="string")
    for columns in (NSECE_CHILDCARE_MATCH_COLUMNS, *NSECE_CHILDCARE_FALLBACK_COLUMNS):
        weighted = train.reindex(columns=columns).copy()
        weighted["mass"] = train.child_weight
        for outcome in outcomes:
            weighted[outcome] = (
                train.reindex(columns=[outcome]).iloc[:, 0] * train.child_weight
            )
        grouped = weighted.groupby(list(columns)).sum()
        means = grouped[outcomes].div(grouped.mass, axis=0)
        ids = levels.index[levels == "unsupported"]
        keys = (
            pd.MultiIndex.from_frame(target.reindex(index=ids, columns=columns))
            if len(columns) > 1
            else target.reindex(index=ids, columns=columns).iloc[:, 0]
        )
        values = means.reindex(keys).to_numpy()
        supported = np.isfinite(values).all(axis=1)
        result.loc[ids[supported]] = values[supported]
        levels.loc[ids[supported]] = ",".join(columns)
    return result, levels


def assess_nsece_childcare(source: NSECEChildcareSource, *, seed: int = 271828) -> dict:
    children = source.children.loc[source.children.age.between(0, 12)].copy()
    usable = children.attendance_status.eq("complete")
    coverage = []
    for grouping in (
        "questionnaire_version",
        "age",
        "region",
        "parent_work_status",
        "income_band",
    ):
        for label, indices in children.groupby(grouping).groups.items():
            g = children.loc[indices]
            coverage.append(
                {
                    "grouping": grouping,
                    "group": str(label),
                    "n": len(g),
                    "weight": float(g.child_weight.sum()),
                    "usable_weight_share": float(
                        g.loc[usable.loc[indices], "child_weight"].sum()
                        / g.child_weight.sum()
                    ),
                }
            )
    donors = children.loc[usable].copy()
    donors["participation"] = (donors.childcare_days_per_week > 0).astype(float)
    donors["days"] = donors.childcare_days_per_week
    donors["hours"] = donors.ece_hours_per_week
    outcomes = ["participation", "days", "hours"]
    folds = donors.source_household_id.map(
        lambda x: (
            int.from_bytes(hashlib.sha256(f"{seed}:{x}".encode()).digest()[:8], "big")
            % 5
        )
    )
    prediction = pd.DataFrame(np.nan, index=donors.index, columns=outcomes)
    levels = pd.Series("", index=donors.index, dtype="string")
    sibling_rows = []
    for fold in range(5):
        train, target = donors.loc[folds != fold], donors.loc[folds == fold]
        if train.empty or target.empty:
            raise ValueError("Five-fold assessment requires households in every fold")
        if not set(train.source_household_id).isdisjoint(target.source_household_id):
            raise ValueError("Childcare assessment leaks households across folds")
        predicted, matched = _conditional_expectations(train, target, outcomes)
        prediction.loc[target.index] = predicted
        levels.loc[target.index] = matched
        heldout_households = set(target.source_household_id)
        fitted = fit_nsece_sibling_dependence(
            children.loc[~children.source_household_id.isin(heldout_households)]
        )
        pairs = complete_sibling_pairs(
            children.loc[children.source_household_id.isin(heldout_households)]
        )
        if not pairs.empty:
            probabilities = (
                predicted.loc[pairs.index, "participation"].to_numpy().reshape(-1, 2)
            )
            actual = (
                (pairs.childcare_days_per_week.to_numpy() > 0)
                .reshape(-1, 2)
                .all(axis=1)
            )
            weights = pairs.household_weight.to_numpy().reshape(-1, 2)[:, 0]
            independent = probabilities.prod(axis=1)
            coupled = (1 - fitted["rho"]) * independent + fitted[
                "rho"
            ] * probabilities.min(axis=1)
            sibling_rows.append(
                {
                    "fold": fold,
                    "training_rho": fitted["rho"],
                    "households": len(weights),
                    "weight": float(weights.sum()),
                    "observed_both": float(np.average(actual, weights=weights)),
                    "independent_both": float(np.average(independent, weights=weights)),
                    "coupled_both": float(np.average(coupled, weights=weights)),
                }
            )
    comparisons = []
    for grouping in (None, "age", "region", "parent_work_status", "income_band"):
        groups = (
            [("all", donors.index)]
            if grouping is None
            else donors.groupby(grouping).groups.items()
        )
        for label, indices in groups:
            g = donors.loc[indices]
            comparisons.append(
                {
                    "grouping": grouping or "all",
                    "group": str(label),
                    "n": len(g),
                    "observed": {
                        c: float(np.average(g[c], weights=g.child_weight))
                        for c in outcomes
                    },
                    "expected": {
                        c: float(
                            np.average(
                                prediction.loc[indices, c], weights=g.child_weight
                            )
                        )
                        for c in outcomes
                    },
                }
            )
    # Independent questionnaire instruments have regular-care weekly hours but
    # no days. Compare the common regular-care estimand, not all-ECE totals.
    regular = children.loc[children.regular_hours_per_week.notna()].copy()
    regular["regular_participation"] = (regular.regular_hours_per_week > 0).astype(
        float
    )
    regular_outcomes = ["regular_participation", "regular_hours_per_week"]
    transport = []
    for questionnaire in (2, 3):
        target = regular.loc[regular.questionnaire_version == questionnaire]
        train = regular.loc[
            (regular.questionnaire_version == 1)
            & regular.attendance_status.eq("complete")
        ]
        if target.empty or train.empty:
            continue
        predicted, matched = _conditional_expectations(train, target, regular_outcomes)
        supported = matched != "unsupported"
        target, predicted = target.loc[supported], predicted.loc[supported]
        transport.append(
            {
                "questionnaire_version": questionnaire,
                "scored_children": len(target),
                "unsupported_children": int((~supported).sum()),
                "observed": {
                    c: float(np.average(target[c], weights=target.child_weight))
                    for c in regular_outcomes
                },
                "expected": {
                    c: float(np.average(predicted[c], weights=target.child_weight))
                    for c in regular_outcomes
                },
            }
        )
    return {
        "source": source.source_receipt,
        "seed": seed,
        "design": "five-fold household-separated conditional-expectation diagnostic",
        "coverage": coverage,
        "cross_validation": comparisons,
        "matching_levels": levels.value_counts().to_dict(),
        "sibling_validation": sibling_rows,
        "questionnaire_transport": transport,
        "assumptions": [
            "Calendar selection is ignorable conditional on matching fields; not identified from these data.",
            "Main reference-week schedules transfer to May/fall typical weeks; summer attendance is not observed.",
            "Regular-care comparisons cannot validate days or irregular care in instruments without calendars.",
        ],
        "production_ready": False,
    }


def assess_noncalendar_bridge(
    source: NSECEChildcareSource, *, seed: int = 161803
) -> dict:
    """Mask whole-household calendars, keeping only the observed regular hours."""
    children = source.children.copy()
    complete = children.attendance_status.eq("complete")
    holdout = complete & children.source_household_id.map(
        lambda x: (
            int.from_bytes(hashlib.sha256(f"{seed}:{x}".encode()).digest()[:8], "big")
            % 5
            == 0
        )
    )
    truth = children.loc[holdout].copy()
    children.loc[~complete, "regular_hours_per_week"] = np.nan
    children.loc[holdout, "attendance_status"] = "missing_calendar"
    children.loc[holdout, "questionnaire_version"] = 2
    children.loc[
        holdout,
        [
            "childcare_attending_days_per_month",
            "childcare_days_per_week",
            "childcare_hours_per_day",
            "ece_hours_per_week",
            "irregular_hours_per_week",
        ],
    ] = np.nan
    bridged = bridge_nsece_noncalendar_attendance(
        NSECEChildcareSource(children, source.weights, source.source_receipt), seed=seed
    )
    predicted = bridged.children.loc[holdout]
    if not predicted.attendance_status.eq("summary_bridge").all():
        raise ValueError("Masked-calendar bridge has unsupported records")
    errors = []
    for group in (None, "age", "parent_work_status"):
        groups = (
            [("all", truth.index)]
            if group is None
            else truth.groupby(group).groups.items()
        )
        for label, indices in groups:
            w = truth.loc[indices, "child_weight"]
            errors.append(
                {
                    "grouping": group or "all",
                    "group": str(label),
                    "n": len(indices),
                    "observed_days": float(
                        np.average(
                            truth.loc[indices, "childcare_days_per_week"], weights=w
                        )
                    ),
                    "imputed_days": float(
                        np.average(
                            predicted.loc[indices, "childcare_days_per_week"], weights=w
                        )
                    ),
                    "observed_hours": float(
                        np.average(truth.loc[indices, "ece_hours_per_week"], weights=w)
                    ),
                    "imputed_hours": float(
                        np.average(
                            predicted.loc[indices, "ece_hours_per_week"], weights=w
                        )
                    ),
                }
            )
    return {
        "seed": seed,
        "design": "whole-household masked-calendar test; regular weekly hours remain observed",
        "comparisons": errors,
        "regular_hours_preserved": bool(
            np.array_equal(
                predicted.regular_hours_per_week, truth.regular_hours_per_week
            )
        ),
        "limitations": "Tests reconstruction where calendars exist; not a direct test of unobserved days in the other questionnaire instruments.",
    }
