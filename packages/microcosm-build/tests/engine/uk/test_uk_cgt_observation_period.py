"""Tests split from packages/microcosm-build/tests/test_uk_cgt_observation_period.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_cgt_observation_period import *


def test_actual_engine_resolves_observed_cgt_without_relabelling_source(tmp_path):
    frame = _frame()
    resolver = UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path, year=2025
    )
    gains, route = resolver.compute("person", "cgt_2024_gains")
    tax, _ = resolver.compute("person", "cgt_2024_tax")
    np.testing.assert_array_equal(gains, frame.table("person").capital_gains)
    np.testing.assert_array_equal(
        tax, np.asarray(resolver.simulation.calculate("capital_gains_tax", 2024))
    )
    assert route == "engine_period:2024:capital_gains:native"
    assert (gains > 3000).tolist() == [False, True, True]
    assert tax[0] == 0 and np.all(tax[1:] > 0)
    # A separate forward calculation crosses the fixed AEA; it is not the fit.
    future, _ = resolver.compute("person", "cgt_2025_gains")
    assert future[0] > 3000
    assert frame.metadata["time_period"] == "2024"
    np.testing.assert_array_equal(
        frame.table("person").capital_gains, [3000, 5000, 20000]
    )
