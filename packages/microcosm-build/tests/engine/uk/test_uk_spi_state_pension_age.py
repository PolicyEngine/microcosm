"""State Pension age coherence of the SPI income stage with the engine's SPA (microcosm#1069)."""

# ruff: noqa: F403, F405
from microcosm.build.uk_runtime.frs_take_up import uk_take_up_population_policy
from microcosm.build.uk_runtime.spi_income import (
    SPI_DONOR_AGE_POPULATION_RESOURCE,
    UKSPIStatePensionAgeGuard,
    load_spi_donor_age_model,
)
from test_support.microcosm_build.uk_spi_income import *

_SPI_PENSION_AGE_COLUMNS = (
    "state_pension_reported",
    "pension_credit_reported",
    "winter_fuel_allowance_reported",
)


def test_donor_age_model_binds_the_vendored_ons_rows_and_the_engine_spa() -> None:
    model = load_spi_donor_age_model(2025)

    assert (
        model.state_pension_age == uk_take_up_population_policy(2025).state_pension_age
    )
    assert set(model.populations) == {
        (sex, age) for sex in ("MALE", "FEMALE") for age in range(0, 91)
    }
    # ONS mid-2023 UK resident population, every age summed.
    assert sum(model.populations.values()) == 68_265_209.0
    assert SPI_DONOR_AGE_POPULATION_RESOURCE in model.source


def test_spi_imputation_leaves_no_pension_age_reports_below_the_engine_spa(
    monkeypatch, tmp_path
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    _bypass_reviewed_donor_identity(monkeypatch)
    model = load_spi_donor_age_model(2025)
    guard = UKSPIStatePensionAgeGuard(
        state_pension_age=model.state_pension_age,
        spi_channel_columns=_SPI_PENSION_AGE_COLUMNS,
        base_channel_columns=("state_pension_reported",),
    )

    unguarded = impute_uk_spi_income_support(
        support, donor_path, seed=9, n_estimators=3, donor_sample_size=None
    )
    guarded = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
        donor_age_model=model,
        state_pension_age_guard=guard,
    )

    channel = support_channel_column("person")
    below = guarded.person["age"].lt(model.state_pension_age)
    spi_below = below & guarded.person[channel].eq("spi")
    # Every fixture person is below State Pension age, and the fake forest
    # draws a positive report for each of them without the guard.
    assert below.all()
    assert spi_below.any()
    assert (
        unguarded.person.loc[spi_below, list(_SPI_PENSION_AGE_COLUMNS)]
        .gt(0)
        .all()
        .all()
    )
    assert guarded.person.loc[below, "state_pension_reported"].eq(0).all()
    assert (
        guarded.person.loc[spi_below, list(_SPI_PENSION_AGE_COLUMNS)].eq(0).all().all()
    )
    assert (
        guarded.person.loc[spi_below, spi_income.SPI_HMRC_STATE_PENSION_INCOME_COLUMN]
        .eq(0)
        .all()
    )
    # Base-channel Pension Credit and Winter Fuel reports are FRS responses the
    # guard does not own.
    base = ~guarded.person[channel].eq("spi")
    pd.testing.assert_frame_equal(
        guarded.person.loc[
            base, ["pension_credit_reported", "winter_fuel_allowance_reported"]
        ],
        unguarded.person.loc[
            base, ["pension_credit_reported", "winter_fuel_allowance_reported"]
        ],
    )
    assert [receipt["step"] for receipt in guarded.state_pension_age_guard] == [
        "spi_state_pension_leaf_after_stage1",
        "base_channel_reports_before_stage2",
        "spi_channel_reports_after_stage2",
    ]
    assert guarded.donor_age_draw["state_pension_age"] == model.state_pension_age
    assert unguarded.state_pension_age_guard == ()
    assert unguarded.donor_age_draw["cells"] == []
