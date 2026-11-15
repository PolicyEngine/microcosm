"""The household measures sidecar against the installed policyengine-us.

The engine-free tests pin batching, keying and the checks with a fake engine;
this one runs the real engine on the post-export scorer's tiny fixture H5
(eight households, written through ``PolicyEngineUSEngine.write_dataset`` at
2026) and checks that batched measures equal one whole-file engine's, that the
installed engine still defines the variables as the tool documents, and that
no batch reaches a population aggregate.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from test_support.microcosm_build.us_county_files import load_tool
from test_support.microcosm_build.us_post_export_scoring import (
    _GUARD_HOUSEHOLDS,
    _write_engine_h5,
)

YEAR = 2026


def test_batched_measures_equal_a_whole_file_engine(tmp_path) -> None:
    from policyengine_us import Microsimulation

    tool = load_tool("us_household_measures")
    h5 = _write_engine_h5(SimpleNamespace(PERIOD=YEAR), tmp_path, _GUARD_HOUSEHOLDS)
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
    whole = Microsimulation(dataset=str(h5), spm=dict(tool.US_RELEASE_SPM_SELECTION))
    ids = np.asarray(whole.calculate("household_id", YEAR)).astype(np.int64)
    assert sorted(ids.tolist()) == sorted(table.index.tolist())
    table = table.loc[ids]

    def household(name: str) -> np.ndarray:
        return np.asarray(
            whole.calculate(name, YEAR, map_to="household"), dtype=np.float64
        )

    for name in tool.COMPONENT_VARIABLES:
        np.testing.assert_allclose(
            table[name].to_numpy(), household(name), rtol=1e-12, atol=1e-6
        )
    np.testing.assert_allclose(
        table["net_income"].to_numpy(),
        household("household_net_income"),
        rtol=1e-12,
        atol=1e-6,
    )
    earnings = sum(household(name) for name in tool.EARNINGS_VARIABLES)
    np.testing.assert_allclose(
        table["earnings"].to_numpy(), earnings, rtol=1e-12, atol=1e-6
    )
    # Every fixture household with wages has positive earnings and market
    # income; net income is below gross income for the highest earner.
    wages = household("employment_income")
    assert (table["earnings"].to_numpy()[wages > 0] > 0).all()
    top = int(np.argmax(wages))
    assert table["net_income"].iloc[top] < table["gross_income"].iloc[top]

    verified = tool.verify_sidecar(h5)
    assert verified["checks"]["households"] == len(_GUARD_HOUSEHOLDS)
