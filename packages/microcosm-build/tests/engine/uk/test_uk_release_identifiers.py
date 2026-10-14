"""policyengine-uk keeps a release's identifiers as written (microcosm#1194).

The engine stores its five identifier variables as 32-bit integers. The clone
stages number copies past that range, so the release boundary renumbers
(``national_frame.uk_release_identifier_tables``). These tests hold the two
halves against the installed engine: the renumbered file reads back exactly,
and the build's own identifiers would not have.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
