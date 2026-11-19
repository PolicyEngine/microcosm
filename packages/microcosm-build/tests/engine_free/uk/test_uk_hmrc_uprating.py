"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_uprating.py."""

# ruff: noqa: F403, F405
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from test_support.microcosm_build.uk_hmrc_uprating import *


def test_every_declared_engine_index_and_the_count_index_have_an_applier() -> None:
    from microcosm.build.uk_runtime.hmrc_uprating import (
        UK_OBR_SAVINGS_INCOME_GROWTH_INDEX_CONCEPT,
        UK_SPI_SELF_EMPLOYMENT_BAND_AMOUNT_INDEX_CONCEPT,
    )

    appliers = hmrc_uprating_appliers()
    assert set(appliers) == {
        *UK_ENGINE_INDEX_CONCEPTS,
        UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT,
        "hmrc.itl_2026.total_income_growth_by_total_income_band",
        "hmrc.itl_2026.total_tax_growth_by_total_income_band",
        "hmrc.itl_2026.taxpayer_count_growth_all_bands",
        "hmrc.itl_2026.total_income_growth_all_bands",
        UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT,
        UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT,
        UK_SPI_SAVINGS_INTEREST_INDEX_CONCEPT,
        UK_OBR_SAVINGS_INCOME_GROWTH_INDEX_CONCEPT,
        UK_SPI_SELF_EMPLOYMENT_BAND_AMOUNT_INDEX_CONCEPT,
    }
    assert set(appliers) <= set(UK_UPRATING_APPLIERS)
    assert len(UK_ENGINE_INDEX_PARAMETERS) == 7


def test_engine_index_moves_the_amount_by_the_parameter_ratio_and_receipts_it() -> None:
    registry = TargetRegistry(
        (
            _spec(
                name="hmrc/employment_income_income_band_20_000_to_30_000@2025",
                value=172e9,
                lower_bound=20_000,
            ),
        ),
        country="uk",
    )

    aligned = align_hmrc_row_by_engine_index(
        _reference(EARNINGS_INDEX),
        registry,
        parameter_path=EARNINGS,
        parameter_value=_fake_parameter,
    )

    (spec,) = aligned.specs
    assert spec.value == pytest.approx(172e9 * 1.1)
    assert spec.metadata["uprating_index"] == EARNINGS_INDEX
    assert spec.metadata["uprating_index_parameter"] == EARNINGS
    assert spec.metadata["uprating_index_from_instant"] == "2023-01-01"
    assert spec.metadata["uprating_index_to_instant"] == "2025-01-01"
    assert float(spec.metadata["uprating_factor"]) == pytest.approx(1.1)
    assert spec.metadata["ledger_value_before_alignment"] == "172000000000"
    assert spec.metadata["uprating_index_engine"].startswith("policyengine-uk ")
    assert "2026-09-22" in spec.metadata["uprating_adjudication"]


def test_engine_index_applier_passes_through_a_reference_declaring_another_index() -> (
    None
):
    registry = TargetRegistry(
        (_spec(name="x", value=1.0, lower_bound=20_000),), country="uk"
    )
    other = f"{UK_ENGINE_PARAMETER_INDEX_PREFIX}gov.economic_assumptions.indices.obr.per_capita.gdp"

    assert (
        align_hmrc_row_by_engine_index(
            _reference(other),
            registry,
            parameter_path=EARNINGS,
            parameter_value=_fake_parameter,
        )
        is registry
    )


def test_a_fact_opening_in_the_calibration_year_binds_as_published_with_a_receipt() -> (
    None
):
    """A fact dated in the compile's own year, 2025 here, reads the index as an identity."""
    registry = TargetRegistry(
        (_spec(name="x", value=7.0, lower_bound=20_000, period="2025"),), country="uk"
    )

    (spec,) = align_hmrc_row_by_engine_index(
        _reference(EARNINGS_INDEX),
        registry,
        parameter_path=EARNINGS,
        parameter_value=_fake_parameter,
    ).specs
    assert spec.value == 7.0
    assert spec.metadata["uprating_factor"] == "1"
    assert spec.metadata["uprating_index_basis"].startswith("identity")

    (count,) = align_hmrc_count_row_by_taxpayer_growth(
        _reference(UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT), registry, count_rows=[]
    ).specs
    assert count.value == 7.0 and count.metadata["uprating_factor"] == "1"


def test_a_fact_opening_after_the_calibration_year_is_refused() -> None:
    registry = TargetRegistry(
        (_spec(name="x", value=1.0, lower_bound=20_000, period="2026"),), country="uk"
    )

    with pytest.raises(ValueError, match="never moves a value backwards"):
        align_hmrc_row_by_engine_index(
            _reference(EARNINGS_INDEX),
            registry,
            parameter_path=EARNINGS,
            parameter_value=_fake_parameter,
        )


def test_count_growth_uses_the_containing_table_2_5_band_and_the_calendar_window() -> (
    None
):
    registry = TargetRegistry(
        (
            _spec(
                name="hmrc/employment_income_count_income_band_40_000_to_50_000@2025",
                value=5.0e6,
                lower_bound=40_000,
            ),
            _spec(
                name="hmrc/employment_income_count_income_band_1_000_000_to_inf@2025",
                value=30_000,
                lower_bound=1_000_000,
            ),
        ),
        country="uk",
    )

    aligned = align_hmrc_count_row_by_taxpayer_growth(
        _reference(UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT),
        registry,
        count_rows=_count_rows(),
    )

    forty, million = aligned.specs
    expected_forty = (0.25 * 11.0e6 + 0.75 * 12.0e6) / 10.0e6
    assert forty.value == pytest.approx(5.0e6 * expected_forty)
    assert float(forty.metadata["uprating_factor"]) == pytest.approx(expected_forty)
    assert forty.metadata["uprating_index_itl_bands"] == "30000-50000"
    assert forty.metadata["uprating_index_window_weights"] == "2024=0.25;2025=0.75"
    assert forty.metadata["uprating_index_values_by_opening_year"] == (
        "2023=10000000;2024=11000000;2025=12000000"
    )
    expected_million = (0.25 * 28_500 + 0.75 * 31_000) / 26_000
    assert million.value == pytest.approx(30_000 * expected_million)
    assert million.metadata["uprating_index_itl_bands"] == "1000000-2000000;2000000-inf"
    assert CALENDAR_YEAR_WINDOW_WEIGHTS == {-1: 0.25, 0: 0.75}


def test_count_growth_refuses_a_band_without_a_vendored_count() -> None:
    registry = TargetRegistry(
        (_spec(name="x", value=1.0, lower_bound=12_570),), country="uk"
    )

    with pytest.raises(ValueError, match="matched 0 vendored rows"):
        align_hmrc_count_row_by_taxpayer_growth(
            _reference(UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT),
            registry,
            count_rows=_count_rows(),
        )


def test_every_spi_band_maps_to_table_2_5_bands_that_tile_the_range() -> None:
    assert sorted(SPI_BAND_TO_ITL_BANDS) == [
        12_570,
        15_000,
        20_000,
        30_000,
        40_000,
        50_000,
        70_000,
        100_000,
        150_000,
        200_000,
        300_000,
        500_000,
        1_000_000,
    ]
    for lower, bands in SPI_BAND_TO_ITL_BANDS.items():
        assert bands[0][0] <= lower
        for (_a_low, a_high), (b_low, _) in zip(bands, bands[1:], strict=False):
            assert a_high == b_low


def test_itl_bands_spanned_tiles_every_spi_band_and_the_regional_open_top_band() -> (
    None
):
    from microcosm.build.uk_runtime.hmrc_uprating import itl_bands_spanned

    # SPI Table 3.6/3.7 bands, spelled by their lower edge: the declared table.
    assert itl_bands_spanned(30_000, None) == ((30_000, 50_000),)
    assert itl_bands_spanned(40_000, None) == ((30_000, 50_000),)
    assert itl_bands_spanned(200_000, None) == ((200_000, 500_000),)
    assert itl_bands_spanned(1_000_000, None) == (
        (1_000_000, 2_000_000),
        (2_000_000, None),
    )
    # Bands spelled with both edges (Tables 3.3 and 3.11) sit inside the band
    # their lower edge declares.
    assert itl_bands_spanned(30_000, 40_000) == ((30_000, 50_000),)
    assert itl_bands_spanned(200_000, 300_000) == ((200_000, 500_000),)
    # A publisher's "and over" band takes every Table 2.5 band above it.
    assert itl_bands_spanned(200_000, None, open_top=True) == (
        (200_000, 500_000),
        (500_000, 1_000_000),
        (1_000_000, 2_000_000),
        (2_000_000, None),
    )
    with pytest.raises(ValueError, match="does not tile"):
        itl_bands_spanned(25_000, 35_000)
    with pytest.raises(ValueError, match="no Table 2.5 band opens"):
        itl_bands_spanned(250_000, None, open_top=True)
    with pytest.raises(ValueError, match="no declared Table 2.5 band"):
        itl_bands_spanned(250_000, None)


def test_a_regional_open_top_band_sums_every_table_2_5_band_above_it() -> None:
    rows = _count_rows()
    for year, value in ((2023, 100.0), (2024, 110.0), (2025, 120.0)):
        rows.append(_count_row(year=year, lower=200_000, upper=500_000, value=value))
        rows.append(_count_row(year=year, lower=500_000, upper=1_000_000, value=value))
    registry = TargetRegistry(
        (
            _spec(
                name="hmrc.spi_region.taxpayers_200000_plus@E12000007",
                value=1_000.0,
                lower_bound=200_000,
                value_id="london_all_band_200000_plus",
            ),
        ),
        country="uk",
    )

    (spec,) = align_hmrc_count_row_by_taxpayer_growth(
        _reference(UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT), registry, count_rows=rows
    ).specs
    y2023 = 100 + 100 + 20_000 + 6_000
    window = 0.25 * (110 + 110 + 22_000 + 6_500) + 0.75 * (120 + 120 + 24_000 + 7_000)
    assert float(spec.metadata["uprating_factor"]) == pytest.approx(window / y2023)
    assert spec.metadata["uprating_index_itl_bands"] == (
        "200000-500000;500000-1000000;1000000-2000000;2000000-inf"
    )


def test_state_pension_amount_moves_with_its_count_rows_recipients_and_the_rate() -> (
    None
):
    """microcosm#1069: amount = recipients (the count row's growth) x the rate."""

    registry = TargetRegistry(
        (
            _spec(
                name="hmrc/state_pension_income_band_40_000_to_50_000@2025",
                value=6.0e9,
                lower_bound=40_000,
            ),
        ),
        country="uk",
    )

    def rate(path: str, instant: str) -> float:
        assert path == UK_SPI_STATE_PENSION_RATE_PARAMETER
        return {"2023-01-01": 200.0, "2025-01-01": 230.0}[instant]

    (spec,) = align_hmrc_amount_by_recipient_growth_and_rate(
        _reference(TABLE_2_5_AMOUNT_INDEX),
        registry,
        index_concept=TABLE_2_5_AMOUNT_INDEX,
        parameter_path=UK_SPI_STATE_PENSION_RATE_PARAMETER,
        parameter_value=rate,
        rows=_count_rows(),
    ).specs

    # The count row's factor for the band (the 30-50k Table 2.5 band's window
    # over 2023) times the rate ratio: the mean per recipient moves by the rate.
    (count,) = align_hmrc_count_row_by_taxpayer_growth(
        _reference(UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT),
        registry,
        count_rows=_count_rows(),
    ).specs
    count_factor = float(count.metadata["uprating_factor"])
    assert count_factor == pytest.approx(1.175)
    assert spec.value == pytest.approx(6.0e9 * 1.175 * 1.15)
    assert spec.metadata["uprating_index"] == TABLE_2_5_AMOUNT_INDEX
    assert float(spec.metadata["uprating_count_factor"]) == pytest.approx(1.175)
    assert float(spec.metadata["uprating_rate_factor"]) == pytest.approx(1.15)
    assert (
        spec.metadata["uprating_rate_parameter"] == UK_SPI_STATE_PENSION_RATE_PARAMETER
    )
    assert spec.metadata["uprating_index_itl_bands"] == "30000-50000"
    assert spec.metadata["ledger_value_before_alignment"] == "6000000000"
    # Another index passes through untouched.
    assert (
        align_hmrc_amount_by_recipient_growth_and_rate(
            _reference(EARNINGS_INDEX),
            registry,
            index_concept=TABLE_2_5_AMOUNT_INDEX,
            parameter_path=UK_SPI_STATE_PENSION_RATE_PARAMETER,
            parameter_value=rate,
            rows=_count_rows(),
        )
        is registry
    )
    # Compiled at its own year, the row stays as published with a complete
    # receipt rather than dropping out.
    (own_year,) = align_hmrc_amount_by_recipient_growth_and_rate(
        _reference(TABLE_2_5_AMOUNT_INDEX, period=2023),
        registry,
        index_concept=TABLE_2_5_AMOUNT_INDEX,
        parameter_path=UK_SPI_STATE_PENSION_RATE_PARAMETER,
        parameter_value=rate,
        rows=_count_rows(),
    ).specs
    assert own_year.value == 6.0e9
    assert own_year.metadata["uprating_factor"] == "1"
    assert own_year.metadata["uprating_rate_from_instant"] == "2023-01-01"
    assert "uprating_rate_from_value" not in own_year.metadata


def _two_leg_parameter(path: str, instant: str) -> float:
    return {
        (UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER, "2023-01-01"): 5.0,
        (UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER, "2024-01-01"): 6.0,
        (UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER, "2024-01-01"): 2.0,
        (UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER, "2025-01-01"): 2.1,
    }[(path, instant)]


def _align_two_legs(registry, *, period: int = 2025, dataset_year: int = 2024):
    return align_hmrc_row_by_history_then_projection(
        _reference(UK_SPI_SAVINGS_INTEREST_INDEX_CONCEPT, period=period),
        registry,
        index_concept=UK_SPI_SAVINGS_INTEREST_INDEX_CONCEPT,
        history_path=UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER,
        projection_path=UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER,
        dataset_year=dataset_year,
        parameter_value=_two_leg_parameter,
    )


def test_savings_interest_moves_by_its_history_then_the_dataset_projection() -> None:
    # A 2023 fact reaches the dataset's year (2024) by the history index and
    # the calibration year by the projection's (microcosm#1095).
    registry = TargetRegistry(
        (_spec(name="x", value=100.0, lower_bound=20_000),), country="uk"
    )

    (spec,) = _align_two_legs(registry).specs

    assert spec.value == pytest.approx(100.0 * (6.0 / 5.0) * (2.1 / 2.0))
    assert spec.metadata["uprating_index"] == UK_SPI_SAVINGS_INTEREST_INDEX_CONCEPT
    assert spec.metadata["uprating_index_from_instant"] == "2023-01-01"
    assert spec.metadata["uprating_index_pivot_instant"] == "2024-01-01"
    assert spec.metadata["uprating_index_to_instant"] == "2025-01-01"
    assert float(spec.metadata["uprating_history_factor"]) == pytest.approx(1.2)
    assert float(spec.metadata["uprating_projection_factor"]) == pytest.approx(1.05)
    assert float(spec.metadata["uprating_factor"]) == pytest.approx(1.26)
    assert spec.metadata["ledger_value_before_alignment"] == "100"
    assert spec.metadata["uprating_history_parameter"] == (
        UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER
    )
    assert spec.metadata["uprating_projection_parameter"] == (
        UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER
    )


def test_the_two_leg_index_keeps_to_one_leg_outside_the_datasets_year() -> None:
    # A fact that opens in the dataset's year takes the projection leg alone.
    registry = TargetRegistry(
        (_spec(name="x", value=100.0, lower_bound=20_000, period="2024"),),
        country="uk",
    )
    (spec,) = _align_two_legs(registry).specs
    assert spec.value == pytest.approx(105.0)
    assert float(spec.metadata["uprating_history_factor"]) == 1.0

    # A fact compiled at its own year binds as published.
    registry = TargetRegistry(
        (_spec(name="x", value=7.0, lower_bound=20_000),), country="uk"
    )
    (spec,) = _align_two_legs(registry, period=2023).specs
    assert spec.value == 7.0
    assert spec.metadata["uprating_factor"] == "1"

    # Another index passes through, and a fact after the target is refused.
    assert (
        align_hmrc_row_by_history_then_projection(
            _reference(EARNINGS_INDEX),
            registry,
            index_concept=UK_SPI_SAVINGS_INTEREST_INDEX_CONCEPT,
            history_path=UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER,
            projection_path=UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER,
            dataset_year=2024,
        )
        is registry
    )
    late = TargetRegistry(
        (_spec(name="x", value=7.0, lower_bound=20_000, period="2026"),),
        country="uk",
    )
    with pytest.raises(ValueError, match="never moves a value backwards"):
        _align_two_legs(late)


def test_all_band_growth_moves_an_unbanded_area_row_by_the_all_ranges_window() -> None:
    """The SPI income-by-area cells carry no band: they move by HMRC's
    projected growth of all taxpayers (counts) and their total income
    (amounts), Table 2.5's all-ranges rows in the calendar-2025 window
    (microcosm#1123)."""

    from microcosm.build.uk_runtime.hmrc_uprating import (
        align_hmrc_row_by_itl_all_band_growth,
    )

    registry = TargetRegistry(
        (
            _spec(
                name="hmrc.employment_income.count@E14001063@2025",
                value=40_000.0,
                lower_bound=None,
            ),
        ),
        country="uk",
    )
    counted = align_hmrc_row_by_itl_all_band_growth(
        _reference("hmrc.itl_2026.taxpayer_count_growth_all_bands"),
        registry,
        index_concept="hmrc.itl_2026.taxpayer_count_growth_all_bands",
    )
    expected_count = (0.25 * 38.6e6 + 0.75 * 39.8e6) / 36.7e6
    (count,) = counted.specs
    assert float(count.metadata["uprating_factor"]) == pytest.approx(expected_count)
    assert count.value == pytest.approx(40_000.0 * expected_count)
    amounted = align_hmrc_row_by_itl_all_band_growth(
        _reference("hmrc.itl_2026.total_income_growth_all_bands"),
        registry,
        index_concept="hmrc.itl_2026.total_income_growth_all_bands",
    )
    (amount,) = amounted.specs
    assert float(amount.metadata["uprating_factor"]) == pytest.approx(
        (0.25 * 1.66e12 + 0.75 * 1.76e12) / 1.53e12
    )
    passthrough = align_hmrc_row_by_itl_all_band_growth(
        _reference(UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT),
        registry,
        index_concept="hmrc.itl_2026.taxpayer_count_growth_all_bands",
    )
    assert passthrough is registry


def test_table_3_10_amounts_move_by_the_count_index_times_mixed_income() -> None:
    """The independent review of microcosm#1420, finding 2: a Table 3.10 band's
    amount moves as its count does, times the engine's mixed income per head,
    so the band's mean moves as the engine moves incomes."""

    from microcosm.build.uk_runtime.hmrc_uprating import (
        UK_SPI_SELF_EMPLOYMENT_BAND_AMOUNT_INDEX_CONCEPT as INDEX,
    )
    from microcosm.build.uk_runtime.hmrc_uprating import (
        UK_SPI_SELF_EMPLOYMENT_PROFIT_PARAMETER as PARAMETER,
    )
    from microcosm.build.uk_runtime.hmrc_uprating import (
        align_hmrc_row_by_itl_all_band_growth,
    )

    count_index = "hmrc.itl_2026.taxpayer_count_growth_all_bands"
    registry = TargetRegistry(
        (
            _spec(
                name="hmrc/self_employment_income_self_employment_income_band@2025",
                value=2.0e7,
                lower_bound=1,
            ),
        ),
        country="uk",
    )

    def mixed_income(path: str, instant: str) -> float:
        assert path == PARAMETER
        return {"2023-01-01": 100.0, "2025-01-01": 104.0}[instant]

    (spec,) = align_hmrc_amount_by_recipient_growth_and_rate(
        _reference(INDEX),
        registry,
        index_concept=INDEX,
        parameter_path=PARAMETER,
        count_index_concept=count_index,
        parameter_value=mixed_income,
    ).specs
    (count,) = align_hmrc_row_by_itl_all_band_growth(
        _reference(count_index), registry, index_concept=count_index
    ).specs
    count_factor = float(count.metadata["uprating_factor"])
    assert spec.value == pytest.approx(2.0e7 * count_factor * 1.04)
    assert spec.metadata["uprating_index"] == INDEX
    assert float(spec.metadata["uprating_rate_factor"]) == pytest.approx(1.04)
    applier = hmrc_uprating_appliers()[INDEX]
    assert applier.keywords["count_index_concept"] == count_index
    assert applier.keywords["parameter_path"] == PARAMETER


def _savings_growth_rows(growth: dict[int, float]) -> list[dict]:
    return [
        {
            "concept": "obr.self_employed_income_growth.savings_income",
            "measure_id": "growth_rate",
            "period": {"type": "fiscal_year", "value": year},
            "value": rate,
            "source_record_id": f"obr.table_4_5.savings_income.fy{year}",
        }
        for year, rate in growth.items()
    ]


def test_savings_income_growth_chains_the_obr_rates_over_the_calendar_window() -> None:
    """The Table 3.7 interest and other income rows grow by the OBR's savings
    income forecast: the rates chained from the fact's year, the calendar
    year's two fiscal years weighted by their months (microcosm#1418)."""

    from microcosm.build.uk_runtime.hmrc_uprating import (
        UK_OBR_SAVINGS_INCOME_GROWTH_INDEX_CONCEPT as INDEX,
    )
    from microcosm.build.uk_runtime.hmrc_uprating import (
        UK_OBR_SAVINGS_INCOME_GROWTH_RESOURCE,
        align_hmrc_row_by_obr_savings_income_growth,
    )

    registry = TargetRegistry(
        (
            _spec(
                name="hmrc/savings_interest_income_income_band_12570@2025",
                value=1.0e9,
                lower_bound=12_570,
            ),
        ),
        country="uk",
    )
    rows = _savings_growth_rows({2024: 34.96, 2025: 0.65, 2026: -9.0})
    aligned = align_hmrc_row_by_obr_savings_income_growth(
        _reference(INDEX), registry, rows=rows
    )
    expected = 0.25 * 1.3496 + 0.75 * 1.3496 * 1.0065
    (spec,) = aligned.specs
    assert float(spec.metadata["uprating_factor"]) == pytest.approx(expected)
    assert spec.value == pytest.approx(1.0e9 * expected)
    assert spec.metadata["uprating_index"] == INDEX
    assert spec.metadata["uprating_index_resource"] == (
        UK_OBR_SAVINGS_INCOME_GROWTH_RESOURCE
    )
    # Only the years the window reaches are receipted.
    assert spec.metadata["uprating_index_growth_by_year"] == "2024=34.96;2025=0.65"
    assert spec.metadata["uprating_index_source_record_ids"] == (
        "obr.table_4_5.savings_income.fy2024,obr.table_4_5.savings_income.fy2025"
    )
    assert "2026-10-29" in spec.metadata["uprating_adjudication"]
    assert (
        align_hmrc_row_by_obr_savings_income_growth(
            _reference(UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT), registry, rows=rows
        )
        is registry
    )
    # A fact opening in the calibration year binds as published.
    same_year = TargetRegistry(
        (
            _spec(
                name="hmrc/savings_interest_income_income_band_12570@2025",
                value=1.0e9,
                lower_bound=12_570,
                period="2025",
            ),
        ),
        country="uk",
    )
    (published,) = align_hmrc_row_by_obr_savings_income_growth(
        _reference(INDEX), same_year, rows=rows
    ).specs
    assert published.value == pytest.approx(1.0e9)


def test_savings_income_growth_refuses_a_missing_or_repeated_year() -> None:
    from microcosm.build.uk_runtime.hmrc_uprating import (
        UK_OBR_SAVINGS_INCOME_GROWTH_INDEX_CONCEPT as INDEX,
    )
    from microcosm.build.uk_runtime.hmrc_uprating import (
        align_hmrc_row_by_obr_savings_income_growth,
    )

    registry = TargetRegistry(
        (
            _spec(
                name="hmrc/savings_interest_income_income_band_12570@2025",
                value=1.0e9,
                lower_bound=12_570,
            ),
        ),
        country="uk",
    )
    with pytest.raises(
        ValueError, match="no savings income growth for fiscal year 2025"
    ):
        align_hmrc_row_by_obr_savings_income_growth(
            _reference(INDEX), registry, rows=_savings_growth_rows({2024: 34.96})
        )
    repeated = _savings_growth_rows({2024: 34.96, 2025: 0.65})
    repeated.append(dict(repeated[0]))
    with pytest.raises(ValueError, match="more than one"):
        align_hmrc_row_by_obr_savings_income_growth(
            _reference(INDEX), registry, rows=repeated
        )


def _band_specs(*lower_bounds: int, period: str = "2023") -> TargetRegistry:
    return TargetRegistry(
        tuple(
            _spec(
                name=f"hmrc/state_pension_count_income_band_{lower}@2025",
                value=1.0e6 + lower,
                lower_bound=lower,
                period=period,
            )
            for lower in lower_bounds
        ),
        country="uk",
    )


def test_state_pension_age_growth_moves_every_band_by_the_resident_level() -> None:
    """microcosm#1069: the SPI State Pension rows hold their 2023-24 shares of
    the bound State Pension age level, which grows from Table 2.1's United
    Kingdom count to the window over Table 2.2's twelve region-tier areas."""

    registry = _band_specs(12_570, 1_000_000)
    aligned = align_hmrc_row_by_resident_state_pension_age_growth(
        _reference(UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT),
        registry,
        rows=_state_pension_age_rows(),
    )

    resident_window = 12 * (0.25 * 0.648e6 + 0.75 * 0.66744e6)
    uk_window = 0.25 * 8.6e6 + 0.75 * 8.9e6
    factor = resident_window / 8.0e6
    for before, spec in zip(registry.specs, aligned.specs, strict=True):
        assert spec.value == pytest.approx(before.value * factor)
        metadata = spec.metadata
        assert float(metadata["uprating_factor"]) == pytest.approx(factor)
        assert float(metadata["uprating_index_united_kingdom_growth"]) == (
            pytest.approx(uk_window / 8.0e6)
        )
        assert float(metadata["uprating_index_resident_share"]) == pytest.approx(
            resident_window / uk_window
        )
        assert metadata["uprating_index"] == (
            UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT
        )
        assert metadata["uprating_index_measure_id"] == "state_pension_age"
        assert metadata["uprating_index_resident_geographies"].split(";") == [
            code for _, code in UK_REGION_TIER
        ]
        assert metadata["uprating_index_united_kingdom_values_by_year"] == (
            "2023=8000000;2024=8600000;2025=8900000"
        )
        assert metadata["uprating_index_window_weights"] == "2024=0.25;2025=0.75"
        # The UK opening count, then per window year the UK row and the
        # twelve areas: England's own row and the other measures are not read.
        record_ids = metadata["uprating_index_source_record_ids"].split(",")
        assert len(record_ids) == 1 + 2 * 13
        assert not any("E92000001" in record for record in record_ids)
        assert metadata["ledger_value_before_alignment"] == f"{before.value:.15g}"
    # Every band moves by the same factor: the level's growth, not the band's.
    assert len({spec.metadata["uprating_factor"] for spec in aligned.specs}) == 1


def test_state_pension_age_growth_refuses_a_missing_area_and_passes_others() -> None:
    registry = _band_specs(20_000)
    rows = [
        row
        for row in _state_pension_age_rows()
        if not (
            row.get("geography", {}).get("id") == "N92000002"
            and row["period"]["value"] == 2025
        )
    ]
    with pytest.raises(ValueError, match="N92000002 in 2025 matched 0"):
        align_hmrc_row_by_resident_state_pension_age_growth(
            _reference(UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT),
            registry,
            rows=rows,
        )
    assert (
        align_hmrc_row_by_resident_state_pension_age_growth(
            _reference(UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT),
            registry,
            rows=rows,
        )
        is registry
    )
    # Compiled at the fact's own year the row binds as published; a fact
    # after the calibration year is refused.
    (own_year,) = align_hmrc_row_by_resident_state_pension_age_growth(
        _reference(UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT, period=2023),
        registry,
        rows=[],
    ).specs
    assert own_year.value == registry.specs[0].value
    assert own_year.metadata["uprating_factor"] == "1"
    with pytest.raises(ValueError, match="never moves a value backwards"):
        align_hmrc_row_by_resident_state_pension_age_growth(
            _reference(UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT),
            _band_specs(20_000, period="2026"),
            rows=_state_pension_age_rows(),
        )


def test_state_pension_amount_moves_with_the_state_pension_age_level_and_the_rate() -> (
    None
):
    registry = _band_specs(40_000)

    def rate(path: str, instant: str) -> float:
        assert path == UK_SPI_STATE_PENSION_RATE_PARAMETER
        return {"2023-01-01": 200.0, "2025-01-01": 230.0}[instant]

    appliers = hmrc_uprating_appliers()
    assert appliers[UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT].keywords[
        "count_index_concept"
    ] == (UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT)
    (spec,) = align_hmrc_amount_by_recipient_growth_and_rate(
        _reference(UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT),
        registry,
        index_concept=UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT,
        parameter_path=UK_SPI_STATE_PENSION_RATE_PARAMETER,
        count_index_concept=UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT,
        parameter_value=rate,
        rows=_state_pension_age_rows(),
    ).specs

    (count,) = align_hmrc_row_by_resident_state_pension_age_growth(
        _reference(UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT),
        registry,
        rows=_state_pension_age_rows(),
    ).specs
    count_factor = float(count.metadata["uprating_factor"])
    assert spec.value == pytest.approx(registry.specs[0].value * count_factor * 1.15)
    assert spec.metadata["uprating_index"] == (
        UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT
    )
    assert float(spec.metadata["uprating_count_factor"]) == pytest.approx(count_factor)
    assert float(spec.metadata["uprating_rate_factor"]) == pytest.approx(1.15)
    assert (
        spec.metadata["uprating_index_resident_share"]
        == (count.metadata["uprating_index_resident_share"])
    )


_counts = st.floats(min_value=1.0e3, max_value=1.0e8, allow_nan=False)
_shares = st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=12, max_size=12)
#: The twelve areas' total as a fraction of the United Kingdom's: above one is a
#: publisher table the applier must refuse.
_coverage = st.floats(min_value=0.5, max_value=1.1)


@settings(max_examples=200, deadline=None)
@given(
    uk=st.tuples(_counts, _counts, _counts),
    shares=st.tuples(_shares, _shares),
    coverage=st.tuples(_coverage, _coverage),
    values=st.dictionaries(
        st.sampled_from(sorted(SPI_BAND_TO_ITL_BANDS)),
        st.floats(min_value=1.0, max_value=1.0e7),
        min_size=1,
    ),
    rates=st.tuples(
        st.floats(min_value=100.0, max_value=400.0),
        st.floats(min_value=100.0, max_value=400.0),
    ),
)
def test_state_pension_age_factor_invariants(
    uk, shares, coverage, values, rates
) -> None:
    """For every positive Table 2.1 and 2.2 input, against a recompute from the
    raw inputs that shares only the list of areas (``UK_REGION_TIER``) with the
    applier (the window weights are written out, not read from
    ``CALENDAR_YEAR_WINDOW_WEIGHTS``), among rows the applier must ignore
    (England's own row, the aged 65 and over measure and 2026-27):

    * where each window year's twelve areas are within its United Kingdom
      count, every band moves by (1/4 areas 2024-25 + 3/4 areas 2025-26) over
      the United Kingdom's 2023-24 count; the receipt gives United Kingdom
      growth and the resident share, whose product is the factor; the share
      is at most one and the factor at most United Kingdom growth;
    * the bands keep their shares, so the rows sum to their opening total
      times the factor;
    * each amount row moves by its count row's factor times the rate ratio,
      and the two rows' ratio is the rate ratio;
    * where a window year's areas exceed its United Kingdom count, both
      appliers refuse.
    """

    opening, uk_2024, uk_2025 = uk
    areas = {}
    for year, uk_year, year_shares, fraction in zip(
        (2024, 2025), (uk_2024, uk_2025), shares, coverage, strict=True
    ):
        total = sum(year_shares)
        areas[year] = [uk_year * fraction * share / total for share in year_shares]
    rows = [
        _age_row(year=year, geography_id=UK_HMRC_ITL_UK_GEOGRAPHY_ID, value=value)
        for year, value in zip((2023, 2024, 2025), uk, strict=True)
    ]
    for year, year_areas in areas.items():
        rows.extend(
            _age_row(year=year, geography_id=code, value=value)
            for (_, code), value in zip(UK_REGION_TIER, year_areas, strict=True)
        )
    # Rows the applier must ignore. Each would move the factor if read, in
    # addition to or in place of the twelve areas. England's row is set 1% off
    # its regions' sum (HMRC's rounding leaves it off by less), so the four
    # nations do not stand in for the twelve areas.
    for year, year_areas in areas.items():
        england = 1.01 * math.fsum(
            value
            for (_, code), value in zip(UK_REGION_TIER, year_areas, strict=True)
            if code.startswith("E12")
        )
        rows.append(_age_row(year=year, geography_id="E92000001", value=england))
        rows.extend(
            _age_row(
                year=year,
                geography_id=code,
                value=value * 1.07,
                measure_id="age_65_and_over",
            )
            for (_, code), value in zip(UK_REGION_TIER, year_areas, strict=True)
        )
    rows.append(
        _age_row(
            year=2026, geography_id=UK_HMRC_ITL_UK_GEOGRAPHY_ID, value=uk_2025 * 1.3
        )
    )
    rows.extend(
        _age_row(year=2026, geography_id=code, value=value * 1.3)
        for (_, code), value in zip(UK_REGION_TIER, areas[2025], strict=True)
    )
    bands = sorted(values)
    counts = TargetRegistry(
        tuple(
            _spec(
                name=f"hmrc/state_pension_count_income_band_{lower}@2025",
                value=values[lower],
                lower_bound=lower,
            )
            for lower in bands
        ),
        country="uk",
    )

    def rate(path: str, instant: str) -> float:
        assert path == UK_SPI_STATE_PENSION_RATE_PARAMETER
        return {"2023-01-01": rates[0], "2025-01-01": rates[1]}[instant]

    def align_counts():
        return align_hmrc_row_by_resident_state_pension_age_growth(
            _reference(UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT),
            counts,
            rows=rows,
        )

    def align_amounts():
        return align_hmrc_amount_by_recipient_growth_and_rate(
            _reference(UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT),
            counts,
            index_concept=UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT,
            parameter_path=UK_SPI_STATE_PENSION_RATE_PARAMETER,
            count_index_concept=UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT,
            parameter_value=rate,
            rows=rows,
        )

    resident = {year: math.fsum(year_areas) for year, year_areas in areas.items()}
    if resident[2024] > uk_2024 * (1 + 1e-12) or resident[2025] > uk_2025 * (1 + 1e-12):
        for align in (align_counts, align_amounts):
            with pytest.raises(ValueError, match="more than the United Kingdom"):
                align()
        return
    if resident[2024] > uk_2024 * (1 - 1e-12) or resident[2025] > uk_2025 * (1 - 1e-12):
        # Within rounding of the United Kingdom count: either outcome is right.
        return
    resident_window = 0.25 * resident[2024] + 0.75 * resident[2025]
    uk_window = 0.25 * uk_2024 + 0.75 * uk_2025
    expected = resident_window / opening
    uk_growth = uk_window / opening
    aligned = align_counts()
    for before, spec in zip(counts.specs, aligned.specs, strict=True):
        assert spec.value == pytest.approx(before.value * expected, rel=1e-12)
        factor = float(spec.metadata["uprating_factor"])
        growth = float(spec.metadata["uprating_index_united_kingdom_growth"])
        share = float(spec.metadata["uprating_index_resident_share"])
        assert factor == pytest.approx(expected, rel=1e-12)
        assert growth == pytest.approx(uk_growth, rel=1e-12)
        assert share == pytest.approx(resident_window / uk_window, rel=1e-12)
        assert growth * share == pytest.approx(factor, rel=1e-12)
        assert share <= 1.0
        assert factor <= uk_growth * (1 + 1e-12)
    assert math.fsum(spec.value for spec in aligned.specs) == pytest.approx(
        math.fsum(values.values()) * expected, rel=1e-12
    )
    rate_ratio = rates[1] / rates[0]
    for count, amount in zip(aligned.specs, align_amounts().specs, strict=True):
        assert amount.value == pytest.approx(count.value * rate_ratio, rel=1e-12)
