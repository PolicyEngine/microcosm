"""Diagnostic canonical-QRF matching of observed, exact-age joint schedules.

This experiment has no production integration or calendar nonresponse correction.
An ordinal schedule code is decoded to an observed day/hour pair; it is not a
new attendance input. All catalog construction and fits use training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
    _ids,
    _validate_attendance,
)
from microcosm.build.us_runtime.nsece_childcare_pooling import (
    COMPOSITION_CHILDCARE_MATCH_COLUMNS,
)
from microcosm.fit import fit

QRF_CHILDCARE_PREDICTORS = (
    *COMPOSITION_CHILDCARE_MATCH_COLUMNS,
    "household_income",
    "household_members",
    "resident_parent_count",
)
QRF_CHILDCARE_SCHEDULE_CODE = "childcare_schedule_code"
QRF_CHILDCARE_DAY_MULTIPLIER = 25.0  # Daily hours are bounded above by 24.
QRF_CHILDCARE_GRID_POWER = 7  # Fixed 128-point two-dimensional Sobol integration.
QRF_INTERVAL_DRAWS = 8
QRF_INTERVAL_EMPIRICAL_MIX = 0.1


def _interval_rows(children):
    return (
        children.age.between(0, 12)
        & children.questionnaire_version.eq(1)
        & children.attendance_status.isin(["ambiguous_calendar", "partial_calendar"])
    )


class QRFChildcareSchedules:
    """Canonical weighted QRF with an age-specific empirical schedule decoder."""

    def __init__(
        self, children, *, seed=915, n_estimators=100, allow_interval_training=False
    ):
        if children.attendance_status.eq("summary_bridge").any():
            raise ValueError("QRF diagnostics require original measured calendars.")
        self.training_households = frozenset(children.source_household_id)
        self.interval_ids = (
            frozenset(children.loc[_interval_rows(children), "donor_id"])
            if "questionnaire_version" in children
            else frozenset()
        )
        interval = children.attendance_status.eq("interval_model")
        if interval.any() and not allow_interval_training:
            raise ValueError(
                "Modeled interval rows require explicit experimental opt-in."
            )
        if interval.any():
            rows = children.loc[interval]
            days = rows.childcare_days_per_week
            weekly = days * rows.childcare_hours_per_day
            if not (
                days.between(rows.calendar_ece_days_lower, rows.calendar_ece_days_upper)
                & weekly.between(
                    rows.calendar_ece_hours_lower - 1e-10,
                    rows.calendar_ece_hours_upper + 1e-10,
                )
            ).all():
                raise ValueError("Modeled interval schedule violates measured bounds.")
        pool = (
            children.loc[
                children.age.between(0, 12)
                & (children.attendance_status.eq("complete") | interval)
            ]
            .sort_values("donor_id")
            .copy()
        )
        if pool.empty:
            raise ValueError("QRF attendance requires observed training calendars.")
        _ids(pool, "source_household_id", unique=False)
        _ids(pool, "donor_id", unique=True)
        _validate_attendance(pool, complete=True)
        if not np.isfinite(
            pool[list(QRF_CHILDCARE_PREDICTORS)].to_numpy(dtype=float)
        ).all():
            raise ValueError("QRF attendance predictors must be finite.")
        weights = pool.child_weight.to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("QRF attendance needs positive finite design weights.")
        _, days, hours = US_CHILDCARE_ATTENDANCE_COLUMNS
        pool[QRF_CHILDCARE_SCHEDULE_CODE] = (
            QRF_CHILDCARE_DAY_MULTIPLIER * pool[days] + pool[hours]
        )
        self.catalogs = {}
        self.empirical = {}
        self.ages = frozenset(pool.age)
        for age, group in pool.groupby("age"):
            schedules, inverse = np.unique(
                np.column_stack(
                    (
                        group[days].gt(0).astype(float),
                        group[days],
                        group[days] * group[hours],
                    )
                ),
                axis=0,
                return_inverse=True,
            )
            masses = np.bincount(inverse, weights=group.child_weight)
            self.empirical[age] = (schedules, masses / masses.sum())
        for age, group in pool.loc[pool[days] > 0].groupby("age"):
            group = group.sort_values(QRF_CHILDCARE_SCHEDULE_CODE).drop_duplicates(
                QRF_CHILDCARE_SCHEDULE_CODE
            )
            self.catalogs[age] = (
                group[QRF_CHILDCARE_SCHEDULE_CODE].to_numpy(),
                np.column_stack(
                    (np.ones(len(group)), group[days], group[days] * group[hours])
                ),
            )
        self.model = fit(
            pool,
            list(QRF_CHILDCARE_PREDICTORS),
            [QRF_CHILDCARE_SCHEDULE_CODE],
            weights="child_weight",
            seed=seed,
            n_estimators=n_estimators,
        )
        self.cache = {}
        self.decoder_movements = []

    def prepare(self, children):
        """Prepare held-out feature profiles in bounded, deterministic batches."""
        if self.training_households.intersection(children.source_household_id):
            raise ValueError("QRF evaluation households overlap training households.")
        self._prepare_profiles(children)

    def _prepare_profiles(self, children):
        """Internal prediction shared by held-out scoring and training completion."""
        profiles = children.reindex(columns=QRF_CHILDCARE_PREDICTORS).drop_duplicates()
        if not np.isfinite(profiles.to_numpy(dtype=float)).all():
            raise ValueError("QRF target predictors must be finite.")
        if not set(profiles.age).issubset(self.ages):
            raise ValueError("QRF target has no exact-age training support.")
        uniforms = qmc.Sobol(2, scramble=False).random_base2(QRF_CHILDCARE_GRID_POWER)
        count = len(uniforms)
        for start in range(0, len(profiles), 64):
            batch = profiles.iloc[start : start + 64]
            repeated = batch.loc[batch.index.repeat(count)].reset_index(drop=True)
            codes = (
                self.model.predict_from_uniforms(
                    repeated,
                    quantiles={
                        QRF_CHILDCARE_SCHEDULE_CODE: np.tile(uniforms[:, 1], len(batch))
                    },
                    sign_uniforms={
                        QRF_CHILDCARE_SCHEDULE_CODE: np.tile(uniforms[:, 0], len(batch))
                    },
                )[QRF_CHILDCARE_SCHEDULE_CODE]
                .to_numpy()
                .reshape(len(batch), count)
            )
            for row, draws in zip(
                batch.itertuples(index=False, name=None), codes, strict=True
            ):
                values = np.zeros((count, 3))
                positive = draws > 0
                if positive.any():
                    if row[0] not in self.catalogs:
                        raise ValueError(
                            "Positive QRF draw has no exact-age positive donor."
                        )
                    catalog, schedules = self.catalogs[row[0]]
                    right = np.clip(
                        np.searchsorted(catalog, draws[positive]), 0, len(catalog) - 1
                    )
                    left = np.maximum(right - 1, 0)
                    chosen = np.where(
                        np.abs(catalog[left] - draws[positive])
                        <= np.abs(catalog[right] - draws[positive]),
                        left,
                        right,
                    )
                    values[positive] = schedules[chosen]
                    self.decoder_movements.extend(
                        np.abs(catalog[chosen] - draws[positive]).tolist()
                    )
                values, counts = np.unique(values, axis=0, return_counts=True)
                probabilities = counts / count
                cumulative = probabilities.cumsum()
                cumulative[-1] = 1.0
                self.cache[row] = (values, cumulative, probabilities)

    def distribution(self, child):
        """Return the prepared empirical integration distribution."""
        key = tuple(child.reindex(QRF_CHILDCARE_PREDICTORS))
        if child.source_household_id in self.training_households:
            raise ValueError("QRF evaluation households overlap training households.")
        return self.cache[key]


def interval_training_rows(children, model, *, tilt=0):
    """One training-only interval completion pass; measured rows remain intact.

    The prior is a fixed QRF/empirical mixture, restricted to observed bounds.
    This assumes conditional coarsening at random. Tilts expose sensitivity to
    that assumption; completed rows must never enter observed-outcome scoring.
    """
    if tilt not in (-1, 0, 1):
        raise ValueError("Interval sensitivity tilt must be -1, 0 or 1.")
    target = children.loc[_interval_rows(children)].copy()
    bounds = target.reindex(
        columns=[
            "calendar_ece_days_lower",
            "calendar_ece_days_upper",
            "calendar_ece_hours_lower",
            "calendar_ece_hours_upper",
        ]
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(bounds).all()
        or (bounds < 0).any()
        or (bounds[:, :2] > 7).any()
        or (bounds[:, 2:] > 168).any()
        or (bounds[:, 0] > bounds[:, 1]).any()
        or (bounds[:, 2] > bounds[:, 3]).any()
    ):
        raise ValueError("Training calendar bounds must be finite and ordered.")
    if not set(target.donor_id).issubset(model.interval_ids):
        raise ValueError("Interval completion includes children outside training.")
    if not set(children.source_household_id).issubset(model.training_households):
        raise ValueError("Interval completion includes households outside training.")
    model._prepare_profiles(target)
    rows, completed, unsupported = [], set(), []
    quantiles = (np.arange(QRF_INTERVAL_DRAWS) + 0.5) / QRF_INTERVAL_DRAWS
    for _, child in target.iterrows():
        key = tuple(child.reindex(QRF_CHILDCARE_PREDICTORS))
        values, _, probability = model.cache[key]
        empirical, mass = model.empirical[child.age]
        combined, inverse = np.unique(
            np.concatenate([values, empirical]), axis=0, return_inverse=True
        )
        probability = np.bincount(
            inverse,
            weights=np.concatenate(
                [
                    (1 - QRF_INTERVAL_EMPIRICAL_MIX) * probability,
                    QRF_INTERVAL_EMPIRICAL_MIX * mass,
                ]
            ),
        )
        allowed = (
            (combined[:, 1] >= child.calendar_ece_days_lower)
            & (combined[:, 1] <= child.calendar_ece_days_upper)
            & (combined[:, 2] >= child.calendar_ece_hours_lower - 1e-10)
            & (combined[:, 2] <= child.calendar_ece_hours_upper + 1e-10)
        )
        if not allowed.any():
            unsupported.append(float(child.child_weight))
            continue
        schedules = combined[allowed]
        probability = probability[allowed] * np.exp(tilt * schedules[:, 2] / 40)
        probability /= probability.sum()
        cumulative = probability.cumsum()
        cumulative[-1] = 1
        chosen = schedules[np.searchsorted(cumulative, quantiles, side="right")]
        for draw, (_, days, weekly) in enumerate(chosen):
            row = child.copy()
            row["donor_id"] = f"{child.donor_id}:interval:{draw}"
            row["attendance_status"] = "interval_model"
            row["child_weight"] = child.child_weight / QRF_INTERVAL_DRAWS
            row["childcare_attending_days_per_month"] = np.floor(days * 52 / 12 + 0.5)
            row["childcare_days_per_week"] = days
            row["childcare_hours_per_day"] = weekly / days if days else 0.0
            row["ece_hours_per_week"] = weekly
            rows.append(row)
        completed.add(child.donor_id)
    retained = children.loc[~children.donor_id.isin(completed)]
    result = pd.concat([retained, pd.DataFrame(rows)], ignore_index=True)
    if not np.isclose(
        result.child_weight.sum(), children.child_weight.sum(), rtol=1e-12
    ):
        raise ValueError("Interval completion changed design-weight mass.")
    return result, {
        "tilt": tilt,
        "incomplete_training_children": len(target),
        "completed_training_children": len(completed),
        "unsupported_training_children": len(unsupported),
        "unsupported_child_weight": float(sum(unsupported)),
        "completed_child_weight": float(
            target.loc[target.donor_id.isin(completed), "child_weight"].sum()
        ),
        "draws_per_child": QRF_INTERVAL_DRAWS,
        "empirical_mixture_weight": QRF_INTERVAL_EMPIRICAL_MIX,
        "design_weight_conserved": True,
        "interpretation": "modeled training rows, not observations; conditional coarsening assumption",
    }
