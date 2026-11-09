"""The engine derives ``property_wealth`` at every engine boundary (microcosm#1106).

policyengine-uk defines ``property_wealth`` as ``residential_property_value``
(``main_residence_value`` + ``other_residential_property_value``) plus
``non_residential_property_value``. A persisted copy overrides that sum, and
the variable has no uprating index, so a saved WAS total would stay at its
base-year level while its components uprate. The release export and the
calibration resolver's scratch H5 both drop it (the uk-data#543 defect).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_release_export_frame,
    write_uk_national_frame,
)
from microcosm.frame import WeightKind

COMPONENTS = (
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
)


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
                "household_weight": [1.0, 1.0, 1.0],
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
                "main_residence_value": [300_000.0, 0.0, 250_000.0],
                "other_residential_property_value": [150_000.0, 0.0, 0.0],
                "non_residential_property_value": [0.0, 20_000.0, 0.0],
                "owned_land": [10_000.0, 0.0, 0.0],
                # The WAS identity adds owned land and the remainder too.
                "property_wealth": [500_000.0, 20_000.0, 260_000.0],
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


def _calculate(simulation, variable: str, year: int) -> np.ndarray:
    return np.asarray(simulation.calculate(variable, year), dtype=float)


def _component_sum(simulation, year: int) -> np.ndarray:
    return sum(_calculate(simulation, name, year) for name in COMPONENTS)


def test_calibration_engine_derives_property_wealth_from_its_components(tmp_path):
    frame = _frame()
    household = frame.table("household")
    persisted_sum = sum(household[name].to_numpy(dtype=float) for name in COMPONENTS)

    resolver = UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path, year=2025
    )

    for year in (2024, 2025):
        derived = _calculate(resolver.simulation, "property_wealth", year)
        np.testing.assert_allclose(derived, _component_sum(resolver.simulation, year))
        main_residence = _calculate(resolver.simulation, "main_residence_value", year)
        assert (derived >= main_residence).all()
    np.testing.assert_allclose(
        _calculate(resolver.simulation, "property_wealth", 2024), persisted_sum
    )
    # The components uprate, so the derived total grows into 2025.
    grown = _calculate(resolver.simulation, "property_wealth", 2025)
    assert (grown[persisted_sum > 0] > persisted_sum[persisted_sum > 0]).all()
    # The resolver's own frame keeps the WAS total for frame-column measures.
    assert "property_wealth" in resolver.frame.table("household").columns
    assert resolver.receipt()["engine_scratch_dropped_columns"] == ["property_wealth"]


def test_a_persisted_total_would_override_the_engine_sum(tmp_path):
    """The mutation check: what the drop prevents, and that the drop cures it."""

    import policyengine_uk
    from policyengine_uk.data import UKSingleYearDataset

    frame = _frame()
    saved = frame.table("household")["property_wealth"].to_numpy(dtype=float)

    with_total = write_uk_national_frame(frame, tmp_path / "with-total.h5")
    simulation = policyengine_uk.Microsimulation(
        dataset=UKSingleYearDataset(file_path=with_total)
    )
    np.testing.assert_allclose(_calculate(simulation, "property_wealth", 2024), saved)
    assert not np.allclose(
        _calculate(simulation, "property_wealth", 2024),
        _component_sum(simulation, 2024),
    )

    released = write_uk_national_frame(
        uk_release_export_frame(frame), tmp_path / "release.h5"
    )
    simulation = policyengine_uk.Microsimulation(
        dataset=UKSingleYearDataset(file_path=released)
    )
    np.testing.assert_allclose(
        _calculate(simulation, "property_wealth", 2024),
        _component_sum(simulation, 2024),
    )
