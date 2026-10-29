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
    # 1.0344 until policyengine-uk 2.155.0 took population from the OBR's
    # March 2026 long-term determinants (PolicyEngine/policyengine-uk#2611).
    assert factor(
        "gov.economic_assumptions.indices.obr.per_capita.mixed_income"
    ) == pytest.approx(1.0384, abs=2e-3)
    assert factor("gov.dwp.state_pension.new_state_pension.amount") == pytest.approx(
        230.25 / 203.85, abs=1e-6
    )


def test_spi_band_targets_move_by_the_engines_dataset_projection() -> None:
    """An HMRC SPI amount family projects its value by the index
    policyengine-uk's dataset projection grows the variable with, so the
    targets move with the data the engine runs on (microcosm#1095,
    uk-data#541). Savings interest reaches the dataset's year by the index of
    its own history first."""

    import json
    from importlib.resources import files
    from pathlib import Path

    import policyengine_uk
    import yaml

    from microcosm.build.uk_runtime.hmrc_uprating import (
        UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER,
        UK_SPI_SAVINGS_INTEREST_INDEX_CONCEPT,
        UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER,
    )

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
    variables = policyengine_uk.CountryTaxBenefitSystem().variables
    contract = json.loads(
        files("microcosm.build.uk").joinpath("uk_population_targets.json").read_text()
    )
    checked = []
    for family in contract["targets"]:
        index = str(family.get("uprating_index") or "")
        variable = family["bindings"]["policyengine"].get("value_variable")
        if family.get("family") != "hmrc_spi":
            continue
        if index == UK_SPI_SAVINGS_INTEREST_INDEX_CONCEPT:
            assert variable == "savings_interest_income"
            assert UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER == projection[variable]
            assert UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER == (
                variables[variable].uprating
            )
            checked.append(family["target_id"])
            continue
        path = index.removeprefix(UK_ENGINE_PARAMETER_INDEX_PREFIX)
        if path == index or not path.startswith("gov.economic_assumptions.indices."):
            continue
        assert path == projection[variable], family["target_id"]
        checked.append(family["target_id"])
    assert "hmrc.spi.savings_interest_income.amount_by_total_income_band" in checked
    assert len(checked) > 1


def test_savings_interest_targets_take_the_donors_path() -> None:
    """The band targets reach 2025 by the path the SPI donors take: the
    history index to the frame's year, then the engine's projection."""

    history = "gov.economic_assumptions.indices.ons.household_interest_income"
    projection = "gov.economic_assumptions.indices.obr.per_capita.gdp"

    def ratio(path: str, start: int, end: int) -> float:
        return engine_parameter_value(path, f"{end}-01-01") / engine_parameter_value(
            path, f"{start}-01-01"
        )

    target_factor = ratio(history, 2023, 2024) * ratio(projection, 2024, 2025)
    donor_path = ratio(history, 2022, 2024) * ratio(projection, 2024, 2025)

    # 1.1920 until policyengine-uk 2.155.0 took population from the OBR's
    # March 2026 long-term determinants (PolicyEngine/policyengine-uk#2611).
    assert target_factor == pytest.approx(1.1961, abs=1e-4)
    # The donors start a year earlier and so carry one more year of history.
    assert donor_path / target_factor == pytest.approx(
        ratio(history, 2022, 2023), rel=1e-12
    )
