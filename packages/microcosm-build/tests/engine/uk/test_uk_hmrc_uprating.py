"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_uprating.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_hmrc_uprating import *


def test_pinned_engine_factors_are_the_ones_the_assessment_measured() -> None:
    def factor(path: str) -> float:
        return engine_parameter_value(path, "2025-01-01") / engine_parameter_value(
            path, "2023-01-01"
        )

    # 1.1067 until policyengine-uk 2.164.1 moved to the OBR's October 2026
    # outlook, which revises 2025 earnings down (microcosm#1415).
    assert factor(
        "gov.economic_assumptions.indices.obr.average_earnings"
    ) == pytest.approx(1.0943, abs=2e-3)
    # 1.0344 until policyengine-uk 2.155.0 took population from the OBR's
    # March 2026 long-term determinants (PolicyEngine/policyengine-uk#2611),
    # and 1.0384 until 2.164.1's October 2026 outlook.
    assert factor(
        "gov.economic_assumptions.indices.obr.per_capita.mixed_income"
    ) == pytest.approx(1.0327, abs=2e-3)
    assert factor("gov.dwp.state_pension.new_state_pension.amount") == pytest.approx(
        230.25 / 203.85, abs=1e-6
    )


def test_spi_band_targets_move_by_the_engines_dataset_projection() -> None:
    """An HMRC SPI amount family projects its value by the index
    policyengine-uk's dataset projection grows the variable with, so the
    targets move with the data the engine runs on (microcosm#1095,
    uk-data#541). The Table 3.7 interest and other income rows follow the
    OBR's savings income forecast instead, by María's ruling of 2026-10-29
    (microcosm#1418); Table 3.10's amounts move by their count index times the
    engine's index for self-employment income (the independent review of
    microcosm#1420); and a measure derived from an input moves by the input's
    index. The SPI donors reach the dataset's year with interest and other
    income on SPI's 2023-24 bands and the OBR's savings income growth (the
    microcosm#1418 savings follow-up) and run on the projection's index from
    there (``spi_income.SPI_HISTORY_INDEX_DIFFERS_FROM_PROJECTION``), so the
    projection's index is held to the engine's."""

    import json
    from importlib.resources import files
    from pathlib import Path

    import policyengine_uk
    import yaml

    from microcosm.build.uk_runtime.hmrc_uprating import (
        UK_OBR_SAVINGS_INCOME_GROWTH_INDEX_CONCEPT,
        UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER,
        UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER,
        UK_SPI_SELF_EMPLOYMENT_BAND_AMOUNT_INDEX_CONCEPT,
        UK_SPI_SELF_EMPLOYMENT_PROFIT_PARAMETER,
    )
    from microcosm.build.uk_runtime.spi_income import (
        SPI_HISTORY_INDEX_DIFFERS_FROM_PROJECTION,
    )

    derived_from = {
        "self_employment_income_after_loss_relief_annual": "self_employment_income"
    }

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
    assert SPI_HISTORY_INDEX_DIFFERS_FROM_PROJECTION == {
        "savings_interest_income": UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER,
        "other_investment_income": projection["other_investment_income"],
    }
    assert (
        UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER
        == (projection["savings_interest_income"])
    )
    assert UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER == (
        variables["savings_interest_income"].uprating
    )
    contract = json.loads(
        files("microcosm.build.uk").joinpath("uk_population_targets.json").read_text()
    )
    checked = []
    on_savings_path = set()
    for family in contract["targets"]:
        index = str(family.get("uprating_index") or "")
        variable = family["bindings"]["policyengine"].get("value_variable")
        if family.get("family") != "hmrc_spi":
            continue
        if index == UK_OBR_SAVINGS_INCOME_GROWTH_INDEX_CONCEPT:
            on_savings_path.add(family["target_id"])
            continue
        if index == UK_SPI_SELF_EMPLOYMENT_BAND_AMOUNT_INDEX_CONCEPT:
            assert (
                UK_SPI_SELF_EMPLOYMENT_PROFIT_PARAMETER
                == (projection[derived_from.get(variable, variable)])
            )
            checked.append(family["target_id"])
            continue
        path = index.removeprefix(UK_ENGINE_PARAMETER_INDEX_PREFIX)
        if path == index or not path.startswith("gov.economic_assumptions.indices."):
            continue
        assert path == projection[derived_from.get(variable, variable)], family[
            "target_id"
        ]
        checked.append(family["target_id"])
    assert on_savings_path == {
        "hmrc.spi.savings_interest_income.amount_by_total_income_band",
        "hmrc.spi.other_investment_income.amount_by_total_income_band",
    }
    assert "hmrc.spi.self_employment_income.amount_by_self_employment_income_band" in (
        checked
    )
    assert len(checked) > 1


def test_savings_interest_targets_take_the_donors_path() -> None:
    """The SPI donors reach 2025 by the history index to the frame's year,
    then the engine's projection. The band targets took the same path until
    they moved to the OBR's savings income forecast (microcosm#1418), so the
    gap between the two is now the calibration's to close."""

    history = "gov.economic_assumptions.indices.ons.household_interest_income"
    projection = "gov.economic_assumptions.indices.obr.per_capita.gdp"

    def ratio(path: str, start: int, end: int) -> float:
        return engine_parameter_value(path, f"{end}-01-01") / engine_parameter_value(
            path, f"{start}-01-01"
        )

    target_factor = ratio(history, 2023, 2024) * ratio(projection, 2024, 2025)
    donor_path = ratio(history, 2022, 2024) * ratio(projection, 2024, 2025)

    # 1.1920 until policyengine-uk 2.155.0 took population from the OBR's
    # March 2026 long-term determinants (PolicyEngine/policyengine-uk#2611),
    # and 1.1961 until 2.164.1's October 2026 outlook (microcosm#1415).
    assert target_factor == pytest.approx(1.1932, abs=1e-4)
    # The donors start a year earlier and so carry one more year of history.
    assert donor_path / target_factor == pytest.approx(
        ratio(history, 2022, 2023), rel=1e-12
    )
