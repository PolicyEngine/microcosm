"""policyengine-uk keeps a release's identifiers as written (microcosm#1194).

The engine stores its five identifier variables as 32-bit integers. The clone
stages number copies past that range, so the release boundary renumbers
(``national_frame.uk_release_identifier_tables``). These tests hold it
against the installed engine: the renumbered file reads back exactly, the
build's own identifiers would not have, and the calibration's engine file deals
the same birth months as the release.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import (
    UK_ENGINE_IDENTIFIER_MAX,
    uk_national_frame,
    uk_release_export_frame,
    write_uk_national_frame,
)
from microcosm.frame import WeightKind

IDENTIFIERS = (
    "person_id",
    "person_benunit_id",
    "person_household_id",
    "benunit_id",
    "household_id",
)
# One survey household and its copy, numbered as the clone stages number them.
COPY = 10**13
BUILD = np.asarray([1, 2, COPY + 1, COPY + 2], dtype=np.int64)


def _frame():
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": BUILD,
                "person_benunit_id": BUILD,
                "person_household_id": BUILD,
                "age": [40, 50, 40, 50],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": BUILD}),
        household=pd.DataFrame(
            {
                "household_id": BUILD,
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.CALIBRATED,
        household_weights=np.ones(len(BUILD)),
    )


def _engine_identifiers(path: Path) -> dict[str, np.ndarray]:
    import policyengine_uk
    from policyengine_uk.data import UKSingleYearDataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = policyengine_uk.Microsimulation(
            dataset=UKSingleYearDataset(file_path=path)
        )
        return {
            name: np.asarray(simulation.calculate(name, 2024)).astype(np.int64)
            for name in IDENTIFIERS
        }


def test_the_engine_returns_the_release_identifiers_as_written(tmp_path) -> None:
    pytest.importorskip("tables")
    path = write_uk_national_frame(
        uk_release_export_frame(_frame()), tmp_path / "release.h5"
    )

    with pd.HDFStore(path, mode="r") as store:
        stored = {
            "person_id": store["person"]["person_id"],
            "person_benunit_id": store["person"]["person_benunit_id"],
            "person_household_id": store["person"]["person_household_id"],
            "benunit_id": store["benunit"]["benunit_id"],
            "household_id": store["household"]["household_id"],
        }
    engine = _engine_identifiers(path)

    for name in IDENTIFIERS:
        written = stored[name].to_numpy(dtype=np.int64)
        assert written.tolist() == [1, 2, 3, 4], name
        np.testing.assert_array_equal(engine[name], written, err_msg=name)


def test_the_build_identifiers_do_not_survive_the_engine(tmp_path) -> None:
    """The mutation check: what the renumbering prevents."""
    pytest.importorskip("tables")
    assert BUILD.max() > UK_ENGINE_IDENTIFIER_MAX
    path = write_uk_national_frame(_frame(), tmp_path / "build-ids.h5")

    engine = _engine_identifiers(path)

    for name in IDENTIFIERS:
        assert not np.array_equal(engine[name], BUILD), name


def _same_age_frame():
    """Forty people of one age and sex, a copy of twenty: the engine spreads
    their birth months over the year by a hash of ``person_id``."""
    build = np.concatenate([np.arange(1, 21), COPY + np.arange(1, 21)]).astype(np.int64)
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": build,
                "person_benunit_id": build,
                "person_household_id": build,
                "age": 40,
                "gender": "MALE",
            }
        ),
        benunit=pd.DataFrame({"benunit_id": build}),
        household=pd.DataFrame(
            {
                "household_id": build,
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.CALIBRATED,
        household_weights=np.ones(len(build)),
    )


def _birth_months(simulation) -> np.ndarray:
    return np.asarray(simulation.calculate("months_since_last_birthday", 2025))


def test_the_calibration_and_the_release_deal_the_same_birth_months(tmp_path) -> None:
    """The date of birth decides the two-child limit, the State Pension scheme
    and the Pension Credit qualifying age, so the solve must measure the deal
    the released file has."""
    pytest.importorskip("tables")
    import policyengine_uk
    from policyengine_uk.data import UKSingleYearDataset

    frame = _same_age_frame()

    solve = UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path / "solve", year=2025
    )
    released = write_uk_national_frame(
        uk_release_export_frame(frame), tmp_path / "release.h5"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        release = policyengine_uk.Microsimulation(
            dataset=UKSingleYearDataset(file_path=released)
        )
        on_build_ids = policyengine_uk.Microsimulation(
            dataset=UKSingleYearDataset(
                file_path=write_uk_national_frame(frame, tmp_path / "build-ids.h5")
            )
        )

    months = _birth_months(release)
    np.testing.assert_array_equal(_birth_months(solve.simulation), months)
    assert len(np.unique(months)) > 20
    # The mutation: on the build's identifiers the engine deals them otherwise.
    assert not np.array_equal(_birth_months(on_build_ids), months)
    # The resolver's own frame keeps the build identifiers.
    assert solve.frame.table("person")["person_id"].tolist() == (
        frame.table("person")["person_id"].tolist()
    )
