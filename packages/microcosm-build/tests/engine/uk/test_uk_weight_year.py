"""policyengine-uk returns a release's solved weights at the solve year (microcosm#796 A19).

The release write divides the solved household weights by the engine's own
projection of ``household_weight`` from the dataset's year to the solve year
(``uk_runtime.weight_year``). These tests hold the three things that rests on
against the installed engine: the pinned growth, that ``household_weight`` is
the only weight the engine projects, and the round trip itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from microcosm.build.uk_runtime import hmrc_uprating
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_release_export_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.weight_year import (
    UK_WEIGHT_PROJECTION_PARAMETER,
    UK_WEIGHT_PROJECTION_PINS_BLOCK,
    UK_WEIGHT_PROJECTION_VARIABLE,
    uk_dataset_year_frame,
    uk_weight_projection_factor,
)
from microcosm.frame import WeightKind

SOLVED = np.asarray([1_000.0, 2_500.0, 400.0])


def _frame():
    ids = np.arange(3, dtype=np.int64)
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": [40, 50, 60],
            }
        ),
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
        household_weights=SOLVED,
    )


def _engine_weights(path: Path, year: int) -> np.ndarray:
    import policyengine_uk
    from policyengine_uk.data import UKSingleYearDataset

    simulation = policyengine_uk.Microsimulation(
        dataset=UKSingleYearDataset(file_path=path)
    )
    return np.asarray(
        simulation.calculate(UK_WEIGHT_PROJECTION_VARIABLE, year), dtype=float
    )


def test_the_pinned_growth_is_the_installed_engines() -> None:
    pins = hmrc_uprating.engine_parameter_pins()
    entry = pins[UK_WEIGHT_PROJECTION_PINS_BLOCK][UK_WEIGHT_PROJECTION_VARIABLE]
    assert entry["parameter"] == UK_WEIGHT_PROJECTION_PARAMETER
    assert list(entry["growth"]) == pins["instants"]
    for instant, value in entry["growth"].items():
        assert hmrc_uprating.live_engine_parameter_value(
            UK_WEIGHT_PROJECTION_PARAMETER, instant
        ) == float(value), instant


def test_household_weight_is_the_only_weight_the_engine_projects() -> None:
    import policyengine_uk
    from policyengine_uk import CountryTaxBenefitSystem

    path = Path(policyengine_uk.__file__).parent / "data" / "uprating_indices.yaml"
    indices = yaml.safe_load(path.read_text(encoding="utf-8"))
    projected = {
        variable: parameter
        for parameter, variables in indices.items()
        for variable in variables
        if "weight" in variable
    }
    assert projected == {UK_WEIGHT_PROJECTION_VARIABLE: UK_WEIGHT_PROJECTION_PARAMETER}
    # The person and benefit-unit weights are formulas of it, never stored.
    system = CountryTaxBenefitSystem()
    for name in ("person_weight", "benunit_weight"):
        assert not system.variables[name].is_input_variable(), name


def test_the_engine_returns_the_solved_weights_at_the_solve_year(tmp_path) -> None:
    pytest.importorskip("tables")
    frame = uk_release_export_frame(_frame())
    factor = uk_weight_projection_factor(2024, 2025)

    released = write_uk_national_frame(
        uk_dataset_year_frame(frame, solve_year=2025), tmp_path / "release.h5"
    )

    np.testing.assert_allclose(_engine_weights(released, 2025), SOLVED, rtol=1e-6)
    np.testing.assert_allclose(
        _engine_weights(released, 2024), SOLVED / factor, rtol=1e-6
    )
    # Beyond the solve year the engine carries on with its own growth.
    np.testing.assert_allclose(
        _engine_weights(released, 2026),
        SOLVED * uk_weight_projection_factor(2025, 2026),
        rtol=1e-6,
    )


def test_weights_written_as_solved_read_high_at_the_solve_year(tmp_path) -> None:
    """The mutation check: what the division prevents."""
    pytest.importorskip("tables")
    factor = uk_weight_projection_factor(2024, 2025)

    as_solved = write_uk_national_frame(
        uk_release_export_frame(_frame()), tmp_path / "as-solved.h5"
    )

    high = _engine_weights(as_solved, 2025)
    np.testing.assert_allclose(high, SOLVED * factor, rtol=1e-6)
    assert not np.allclose(high, SOLVED, rtol=1e-4, atol=0)
