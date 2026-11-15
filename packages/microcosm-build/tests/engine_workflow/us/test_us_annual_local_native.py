"""The projected local fixed layout round-trips through the native US engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.build import us_annual_static_aging as annual
from microcosm.frame.adapters.policyengine_us import us_dataset_through
from microcosm.frame.materialize import engine_tables
from test_support.microcosm_build.us_annual_local import (
    default_households,
    local_frame,
    write_local_base,
)


@pytest.fixture(scope="module", params=(2026, 2027))
def projected_local_year(tmp_path_factory, request):
    """Project a tiny local frame and use the annual builder's fixed writer."""
    year = request.param
    root = tmp_path_factory.mktemp(f"local-annual-native-{year}")
    base_path = write_local_base(root / "base.h5", local_frame(default_households()))
    base = annual.read_local_base(base_path, 2024)
    tables = engine_tables(base.frame, weighted_entities=("household",))
    tables["household"]["household_weight"] *= 1.1
    tables["person"]["employment_income_before_lsr"] *= 1.05
    annual.check_carried_columns(
        base.frame, tables, factored=("employment_income_before_lsr",)
    )
    projected = annual.projected_local_frame(tables, base.frame)
    path = root / f"microcosm_us_{year}_acs_local.h5"
    receipt = annual.write_local_year(
        projected,
        path,
        year,
        root_attributes=base.root_attributes,
        columns=base.columns,
    )
    assert receipt["layout"] == annual.LOCAL_LAYOUT
    rows = {entity: len(table) for entity, table in tables.items()}
    return path, year, rows, tables


def test_projected_local_fixed_file_opens_at_its_own_year(projected_local_year):
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    from microcosm.build.us_runtime.engine_lifecycle import release_engine_simulation

    path, year, rows, tables = projected_local_year
    native = USSingleYearDataset(file_path=str(path))
    assert str(native.time_period) == str(year)
    for entity, expected in tables.items():
        pd.testing.assert_frame_equal(getattr(native, entity), expected)
    assert annual._verify_native_local(path, year, rows)

    # Bound the native engine to this artifact's year, as the production
    # adapter does, so its inputs cannot receive another year's uprating.
    dataset = us_dataset_through(native, year)
    assert set(dataset.datasets) == {year}
    simulation = Microsimulation(dataset=dataset)
    try:
        for entity, column in (
            ("person", "age"),
            ("person", "employment_income_before_lsr"),
            ("household", "household_weight"),
        ):
            np.testing.assert_allclose(
                np.asarray(simulation.calculate(column, year)),
                tables[entity][column].to_numpy(),
            )
    finally:
        release_engine_simulation(simulation)


@settings(max_examples=30, deadline=None)
@given(offset=st.one_of(st.integers(max_value=-1), st.integers(min_value=1)))
@example(offset=-1)
@example(offset=1)
@example(offset=-2)
@example(offset=9)
def test_projected_local_native_guard_refuses_every_other_year(
    projected_local_year, offset
):
    path, year, rows, _ = projected_local_year
    # USSingleYearDataset reads its stored period; the annual branch's guard
    # is the boundary that refuses a request for any different period.
    with pytest.raises(ValueError, match="Native loader did not retain year"):
        annual._verify_native_local(path, year + offset, rows)
