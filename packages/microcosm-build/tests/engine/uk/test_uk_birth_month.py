"""policyengine-uk reads the birth months the spine stores (microcosm#1095).

Left unset, the engine spreads each age and sex over the year by weight in an
order set by a hash of ``person_id``, so a file's birth dates depend on its
weights and identifiers. These tests hold the stored months against the
installed engine: the stage spreads them the engine's way, the engine returns
them as stored, and they no longer move with the weights.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.birth_month import (
    UK_BIRTH_MONTH_COLUMN,
    stratified_positions,
    uk_months_since_last_birthday,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_release_export_frame,
    write_uk_national_frame,
)
from microcosm.frame import WeightKind

N = 40


def _frame(weights: np.ndarray, *, store_months: bool):
    ids = np.arange(1, N + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": ids,
            "person_household_id": ids,
            "age": 66,
            "gender": "MALE",
        }
    )
    if store_months:
        person[UK_BIRTH_MONTH_COLUMN] = uk_months_since_last_birthday(
            age=person["age"].to_numpy(),
            is_male=np.ones(N, dtype=bool),
            person_ids=ids,
            weights=np.full(N, 1_000.0),
        )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.CALIBRATED,
        household_weights=weights,
    )


def _engine_months(path: Path) -> np.ndarray:
    import policyengine_uk
    from policyengine_uk.data import UKSingleYearDataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = policyengine_uk.Microsimulation(
            dataset=UKSingleYearDataset(file_path=path)
        )
        return np.asarray(simulation.calculate(UK_BIRTH_MONTH_COLUMN, 2025))


def test_the_stage_spreads_months_as_the_engine_does() -> None:
    from policyengine_uk.utils.stochastic import stratified_uniform

    rng = np.random.default_rng(11)
    strata = rng.integers(0, 6, 500)
    draws = rng.uniform(0.0, 1.0, 500)
    weights = rng.uniform(0.0, 5.0, 500)
    weights[strata == 5] = 0.0

    np.testing.assert_array_equal(
        stratified_positions(strata, draws, weights),
        stratified_uniform(strata, draws, weights),
    )


def test_stored_months_do_not_move_with_the_weights(tmp_path) -> None:
    pytest.importorskip("tables")
    even = np.full(N, 1_000.0)
    uneven = np.linspace(100.0, 5_000.0, N)

    stored = [
        _engine_months(
            write_uk_national_frame(
                uk_release_export_frame(_frame(weights, store_months=True)),
                tmp_path / f"stored-{name}.h5",
            )
        )
        for name, weights in (("even", even), ("uneven", uneven))
    ]
    np.testing.assert_allclose(stored[0], stored[1], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        stored[0],
        _frame(even, store_months=True).table("person")[UK_BIRTH_MONTH_COLUMN],
        rtol=0,
        atol=1e-5,
    )

    # The mutation: left to the engine, the same people's months follow the
    # weights the file carries.
    unset = [
        _engine_months(
            write_uk_national_frame(
                uk_release_export_frame(_frame(weights, store_months=False)),
                tmp_path / f"unset-{name}.h5",
            )
        )
        for name, weights in (("even", even), ("uneven", uneven))
    ]
    assert not np.allclose(unset[0], unset[1], rtol=0, atol=1e-3)
