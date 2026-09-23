"""Tests split from packages/microcosm-frame/tests/test_contracts.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_frame.contracts import *


def test_export_round_trips_through_the_rules_adapter(
    tmp_path, three_household_frame, make_household_weights
) -> None:
    """A bundle written by the adapter reloads with every column intact."""
    __import__("policyengine_us")
    pytest.importorskip("microunit")
    from microcosm.frame import assign_us_unit_structure

    bundle = assign_us_unit_structure(
        three_household_frame,
        year=2024,
        household_weights=make_household_weights(three_household_frame),
    )
    adapter = PolicyEngineUSEngine()
    path = tmp_path / "bundle.h5"
    adapter.write_dataset(bundle, path, period=2024)  # round-trip verified inside
    assert path.exists()
