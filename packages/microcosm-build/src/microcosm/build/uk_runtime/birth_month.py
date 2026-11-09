"""Each person's months since their last birthday, stored once in the spine.

policyengine-uk places a date of birth from a whole age and
``months_since_last_birthday``: the months since the person's most recent
birthday on 6 October, the middle of the fiscal year. The date of birth
decides the two-child limit (born before or after 6 April 2017), the State
Pension scheme (reaching pension age before or after 6 April 2016) and the
Pension Credit qualifying age.

Left unset, the engine spreads each single year of age and sex evenly over
the year by weight, in an order set by a hash of ``person_id``. So the months
a file gets depend on its weights and its identifiers. The calibration
measures the engine at the spine's weights and the released file carries the
calibrated ones, so a release read a different spread of birth dates from the
one its own solve fitted (microcosm#1095).

The root stage therefore stores the months, drawn the engine's way once: at
the survey weights, in an order set by an identity-keyed draw on the person's
stable build identifier. Every later engine read (the stages, the solve and
the release) takes the stored value, and a copy of a person keeps their
months.
"""

from __future__ import annotations

import numpy as np

from microcosm.build.stochastic_assignment import stable_identity_uniforms

#: The engine variable the root stage stores.
UK_BIRTH_MONTH_COLUMN = "months_since_last_birthday"
UK_BIRTH_MONTH_SEED = 0
UK_BIRTH_MONTH_SALT = "uk.frs_spine.months_since_last_birthday"
#: The months in a year: positions on [0, 1) scale to [0, 12).
MONTHS_IN_YEAR = 12.0
#: The largest position below 1 that float32 keeps below 1 (the engine stores
#: the variable as float32, which would round a larger draw up to 12).
_LARGEST_POSITION = 1.0 - 2.0**-24


def stratified_positions(
    strata: np.ndarray, draws: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Spread records evenly over [0, 1) within each stratum, by weight.

    policyengine-uk's own ``utils.stochastic.stratified_uniform``, restated so
    the stage needs no engine: records in a stratum are ordered by their draw
    and each sits at the midpoint of its slice of the stratum's cumulative
    weight. The result does not depend on the order of the records. A stratum
    with no positive weight keeps the draws.
    """

    strata = np.asarray(strata)
    draws = np.asarray(draws, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if not (len(strata) == len(draws) == len(weights)):
        raise ValueError("Strata, draws and weights must have one value per record.")
    if len(strata) == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.lexsort((draws, strata))
    _, group = np.unique(strata[order], return_inverse=True)
    sorted_weights = weights[order]
    totals = np.bincount(group, weights=sorted_weights)
    starts = np.concatenate([[0.0], np.cumsum(totals)[:-1]])
    within = np.cumsum(sorted_weights) - starts[group] - sorted_weights / 2.0
    group_totals = totals[group]
    positions = np.where(
        group_totals > 0,
        within / np.where(group_totals > 0, group_totals, 1.0),
        draws[order],
    )
    result = np.empty_like(positions)
    result[order] = positions
    return np.clip(result, 0.0, _LARGEST_POSITION)


def uk_months_since_last_birthday(
    *,
    age: np.ndarray,
    is_male: np.ndarray,
    person_ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Months since the last birthday, spread evenly within each age and sex.

    ``age`` must be whole years, as the FRS records it; ``weights`` are the
    people's survey weights.
    """

    age = np.asarray(age, dtype=np.float64)
    if not np.array_equal(age, np.floor(age)):
        raise ValueError("Birth months are spread over whole ages only.")
    strata = age * 2.0 + np.asarray(is_male, dtype=bool).astype(np.float64)
    draws = stable_identity_uniforms(
        np.asarray(person_ids),
        seed=UK_BIRTH_MONTH_SEED,
        salt=UK_BIRTH_MONTH_SALT,
    )
    return MONTHS_IN_YEAR * stratified_positions(strata, draws, weights)


__all__ = [
    "MONTHS_IN_YEAR",
    "UK_BIRTH_MONTH_COLUMN",
    "UK_BIRTH_MONTH_SALT",
    "UK_BIRTH_MONTH_SEED",
    "stratified_positions",
    "uk_months_since_last_birthday",
]
