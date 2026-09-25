"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_uprating.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_hmrc_uprating import *


def test_every_declared_engine_index_and_the_count_index_have_an_applier() -> None:
    appliers = hmrc_uprating_appliers()
    assert set(appliers) == {
        *UK_ENGINE_INDEX_CONCEPTS,
        UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT,
        "hmrc.itl_2026.total_income_growth_by_total_income_band",
        "hmrc.itl_2026.total_tax_growth_by_total_income_band",
    }
    assert set(appliers) <= set(UK_UPRATING_APPLIERS)
    assert len(UK_ENGINE_INDEX_PARAMETERS) == 6

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
    """The production-2023 parity surface compiles the same references at 2023."""
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
