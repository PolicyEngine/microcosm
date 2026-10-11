"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_uprating.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_hmrc_uprating import *


def test_pinned_engine_factors_are_the_ones_the_assessment_measured() -> None:
    def factor(path: str) -> float:
        return engine_parameter_value(path, "2025-01-01") / engine_parameter_value(
            path, "2023-01-01"
        )

    assert factor(
        "gov.economic_assumptions.indices.obr.average_earnings"
    ) == pytest.approx(1.1067, abs=2e-3)
    assert factor(
        "gov.economic_assumptions.indices.obr.per_capita.mixed_income"
    ) == pytest.approx(1.0344, abs=2e-3)
    assert factor("gov.dwp.state_pension.new_state_pension.amount") == pytest.approx(
        230.25 / 203.85, abs=1e-6
    )


def test_spi_band_targets_move_by_the_engines_dataset_projection() -> None:
    """An HMRC SPI amount family declares the index policyengine-uk's dataset
    projection grows its value variable with, so the targets move with the
    data the engine runs on (microcosm#1095, uk-data#541)."""

    import json
    from importlib.resources import files
    from pathlib import Path

    import policyengine_uk
    import yaml

    table = yaml.safe_load(
        (
            Path(policyengine_uk.__file__).parent / "data" / "uprating_indices.yaml"
        ).read_text()
    )
    projection = {
        variable: path.replace(".yoy_growth.", ".indices.")
        for path, variables in table.items()
        for variable in variables
    }
    contract = json.loads(
        files("microcosm.build.uk").joinpath("uk_population_targets.json").read_text()
    )
    checked = []
    for family in contract["targets"]:
        index = str(family.get("uprating_index") or "")
        path = index.removeprefix(UK_ENGINE_PARAMETER_INDEX_PREFIX)
        variable = family["bindings"]["policyengine"].get("value_variable")
        if (
            family.get("family") != "hmrc_spi"
            or path == index
            or not path.startswith("gov.economic_assumptions.indices.")
        ):
            continue
        assert path == projection[variable], family["target_id"]
        checked.append(family["target_id"])
    assert "hmrc.spi.savings_interest_income.amount_by_total_income_band" in checked
