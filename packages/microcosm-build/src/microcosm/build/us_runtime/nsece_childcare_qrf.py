"""Diagnostic canonical-QRF matching of observed, exact-age joint schedules.

This experiment has no production integration or calendar nonresponse correction.
An ordinal schedule code is decoded to an observed day/hour pair; it is not a
new attendance input. All catalog construction and fits use training data only.
"""

from __future__ import annotations

import numpy as np
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


class QRFChildcareSchedules:
    """Canonical weighted QRF with an age-specific empirical schedule decoder."""

    def __init__(self, children, *, seed=915, n_estimators=100):
        if children.attendance_status.eq("summary_bridge").any():
            raise ValueError("QRF diagnostics require original measured calendars.")
        self.training_households = frozenset(children.source_household_id)
        pool = (
            children.loc[
                children.age.between(0, 12) & children.attendance_status.eq("complete")
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
        self.ages = frozenset(pool.age)
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
