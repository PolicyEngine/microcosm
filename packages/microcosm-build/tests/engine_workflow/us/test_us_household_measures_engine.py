"""The household measures sidecar against the installed policyengine-us.

The engine-free tests pin batching, keying and the checks with a fake engine;
this one runs the real engine on the post-export scorer's tiny fixture H5
(eight households, written through ``PolicyEngineUSEngine.write_dataset`` at
2026) and checks that every measure from batches restricted to the file's year
equals an engine using the old full projection horizon. It also checks that
reusing the installed module-level system preserves definition checks and
receipt parameters, and that no batch reaches a population aggregate.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np

from test_support.microcosm_build.us_county_files import load_tool
from test_support.microcosm_build.us_post_export_scoring import (
    _GUARD_HOUSEHOLDS,
    _write_engine_h5,
)

YEAR = 2026


def test_batched_measures_equal_a_whole_file_engine(tmp_path) -> None:
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation
    from policyengine_us.system import system

    tool = load_tool("us_household_measures")
    h5 = _write_engine_h5(SimpleNamespace(PERIOD=YEAR), tmp_path, _GUARD_HOUSEHOLDS)
    _, dataset_factory, metadata_system = tool._installed_engine()
    assert metadata_system is system
    # Reusing the module system changes only checks and receipt metadata:
    # Microsimulation already uses it by default. Check both against the
    # separately constructed system used by the old metadata path.
    previous_metadata_system = CountryTaxBenefitSystem()
    tool.check_engine_definitions(previous_metadata_system)
    tool.check_engine_definitions(metadata_system)
    assert tool.resolved_engine_parameters(
        metadata_system, YEAR
    ) == tool.resolved_engine_parameters(previous_metadata_system, YEAR)
    del previous_metadata_system
    gc.collect()

    tables, year = tool.read_h5_tables(h5)
    bounded_dataset = dataset_factory(**tables, time_period=year)
    assert bounded_dataset.years == [YEAR]
    del bounded_dataset, tables
    receipt = tool.run_compute(
        h5, expect_year=YEAR, batch_households=3, log=lambda message: None
    )
    assert receipt["batching"]["batches"] == 3
    assert receipt["engine"]["aggregate_variables_computed"] == []
    assert receipt["checks"]["every_household_exactly_once"] is True
    resolved = receipt["engine"]["resolved_parameters"]
    assert resolved["gov.simulation.include_health_benefits_in_net_income"] is False
    assert "employment_income" in resolved["gov.household.market_income_sources"]
    assert "snap" in resolved["gov.household.household_benefits"]

    sidecar, _ = tool.sidecar_paths(h5)
    table = tool.read_sidecar(sidecar).set_index("household_id")
    # Passing the single-year H5 directly preserves the old constructor's
    # extension through the CPI-U horizon, independently of the batch factory.
    whole = Microsimulation(dataset=str(h5), spm=dict(tool.US_RELEASE_SPM_SELECTION))
    ids = np.asarray(whole.calculate("household_id", YEAR)).astype(np.int64)
    assert sorted(ids.tolist()) == sorted(table.index.tolist())
    table = table.loc[ids]

    def household(name: str) -> np.ndarray:
        return np.asarray(
            whole.calculate(name, YEAR, map_to="household"), dtype=np.float64
        )

    expected = {name: household(name) for name in tool.COMPONENT_VARIABLES}
    expected["net_income"] = household("household_net_income")
    expected["earnings"] = sum(household(name) for name in tool.EARNINGS_VARIABLES)
    expected["gross_income"] = (
        expected["household_market_income"] + expected["household_benefits"]
    )
    assert set(expected) == {name for name, _ in tool.SIDECAR_COLUMNS} - {
        "household_id",
        "year",
    }
    for name, values in expected.items():
        np.testing.assert_array_equal(table[name].to_numpy(), values, err_msg=name)
    # Every fixture household with wages has positive earnings and market
    # income; net income is below gross income for the highest earner.
    wages = household("employment_income")
    assert (table["earnings"].to_numpy()[wages > 0] > 0).all()
    top = int(np.argmax(wages))
    assert table["net_income"].iloc[top] < table["gross_income"].iloc[top]

    verified = tool.verify_sidecar(h5)
    assert verified["checks"]["households"] == len(_GUARD_HOUSEHOLDS)
