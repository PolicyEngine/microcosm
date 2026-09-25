"""The SPI housing shell imputes housing from the recipient's own incomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_housing_shell import (
    HOUSING_AMOUNT_COLUMNS,
    HOUSING_CATEGORICAL_COLUMNS,
    RENTED_TENURES,
    UK_SPI_HOUSING_SHELL_REWRITES,
    UKSPIHousingShellStageTransform,
    _assert_invariants,
    _assert_stage_parameters,
    impute_spi_housing_shell,
)
from microcosm.frame import WeightKind

_REGIONS = ("LONDON", "NORTH_EAST", "NORTHERN_IRELAND")
_TYPES = ("couple_no_children_households", "lone_households_under_65")


def _stage():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()["spi_housing_shell"]


def _tables(n_frs: int = 600, n_spi: int = 200, seed: int = 0):
    """FRS households whose housing follows income; SPI copies with a parent's housing."""

    rng = np.random.default_rng(seed)
    n = n_frs + n_spi
    household_id = np.arange(1, n + 1)
    channel = np.array(["frs"] * n_frs + ["spi"] * n_spi)
    region = rng.choice(_REGIONS, size=n)
    household_type = rng.choice(_TYPES, size=n)
    single = household_type == "lone_households_under_65"
    # FRS incomes are modest; SPI copies carry high incomes.
    income = np.where(
        channel == "frs", rng.lognormal(10.2, 0.8, n), rng.lognormal(12.5, 0.6, n)
    )
    high = income > np.quantile(income[:n_frs], 0.6)
    tenure = np.where(
        high,
        rng.choice(["OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE"], size=n),
        rng.choice(list(RENTED_TENURES), size=n),
    )
    accommodation = np.where(high, "HOUSE_DETACHED", "FLAT")
    rented = np.isin(tenure, RENTED_TENURES)
    mortgaged = tenure == "OWNED_WITH_MORTGAGE"
    ni = region == "NORTHERN_IRELAND"
    household = pd.DataFrame(
        {
            "household_id": household_id,
            "region": region,
            "ons_household_type": household_type,
            "council_tax_single_adult_raw": np.where(single, 1, 2),
            "household_support_channel": channel,
            "tenure_type": tenure,
            "accommodation_type": accommodation,
            "num_bedrooms": np.where(high, 4, 2).astype(np.int64),
            "council_tax_band": np.where(high, "F", "B"),
            "council_tax": np.where(ni, 0.0, np.where(high, 2800.0, 1300.0)),
            "council_tax_reported": np.where(ni, 0.0, np.where(high, 2800.0, 1300.0)),
            "rent": np.where(rented, 0.1 * income + 3000.0, 0.0),
            "mortgage_interest_repayment": np.where(mortgaged, 4000.0, 0.0),
            "mortgage_capital_repayment": np.where(mortgaged, 6000.0, 0.0),
            "structural_insurance_payments": np.where(rented, 0.0, 300.0),
            "housing_service_charges": np.zeros(n),
            "water_and_sewerage_charges": np.full(n, 400.0),
            "domestic_rates": np.where(ni, 1000.0, 0.0),
            "subrent": np.zeros(n),
            "council_tax_rebate": np.zeros(n),
        }
    )
    # One or two people per household; the first is the reference person.
    rows = []
    for index, hid in enumerate(household_id):
        rows.append((hid, 1, 45, income[index]))
        if not single[index]:
            rows.append((hid, 0, 43, 0.0))
    person = pd.DataFrame(
        rows,
        columns=[
            "person_household_id",
            "is_household_head",
            "age",
            "employment_income",
        ],
    )
    person["person_id"] = np.arange(1, len(person) + 1)
    person["person_benunit_id"] = person["person_household_id"]
    person["is_household_head"] = person["is_household_head"].astype(bool)
    for column in (
        "self_employment_income",
        "private_pension_income",
        "savings_interest_income",
        "dividend_income",
        "property_income",
        "other_investment_income",
        "state_pension_reported",
        "council_tax_benefit_reported",
    ):
        person[column] = 0.0
    hrp_rented = person["person_household_id"].map(
        pd.Series(rented, index=household_id)
    )
    person["housing_benefit_reported"] = np.where(
        person["is_household_head"] & hrp_rented, 2500.0, 0.0
    )
    benunit = pd.DataFrame({"benunit_id": household_id})
    weights = np.where(channel == "frs", 10.0, 5.0)
    return person, benunit, household, weights


def test_packaged_stage_matches_the_module_contract() -> None:
    _assert_stage_parameters(_stage())
    assert _stage().rewrites == UK_SPI_HOUSING_SHELL_REWRITES


def test_frs_rows_are_untouched_and_the_invariants_hold() -> None:
    person, _, household, weights = _tables()
    person_out, household_out, receipt = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    _assert_invariants(person, household, person_out, household_out)
    frs = household["household_support_channel"] == "frs"
    pd.testing.assert_frame_equal(household.loc[frs], household_out.loc[frs])
    assert receipt["training_households"] == int(frs.sum())
    assert receipt["recipient_households"] == int((~frs).sum())


def test_spi_housing_follows_the_recipients_own_income() -> None:
    person, _, household, weights = _tables()
    _, household_out, _ = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    spi = household_out[household_out["household_support_channel"] == "spi"]
    # SPI copies carry high incomes, so the planted relation puts them in
    # owned, detached housing; the parent's shell would not.
    owned = spi["tenure_type"].str.startswith("OWNED").mean()
    detached = (spi["accommodation_type"] == "HOUSE_DETACHED").mean()
    assert owned > 0.85
    assert detached > 0.85


def test_structural_rules_hold_on_recipients() -> None:
    person, _, household, weights = _tables()
    person_out, household_out, _ = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    spi = household_out[household_out["household_support_channel"] == "spi"]
    mortgage = spi["mortgage_interest_repayment"] + spi["mortgage_capital_repayment"]
    assert (mortgage[spi["tenure_type"] != "OWNED_WITH_MORTGAGE"] == 0).all()
    assert (spi.loc[spi["region"] != "NORTHERN_IRELAND", "domestic_rates"] == 0).all()
    assert (spi.loc[spi["region"] == "NORTHERN_IRELAND", "council_tax"] == 0).all()
    assert set(spi["num_bedrooms"].unique()) <= {2, 4}
    for column in HOUSING_CATEGORICAL_COLUMNS:
        assert set(spi[column].astype(str)) <= set(household[column].astype(str))
    spi_people = person_out[person_out["person_household_id"].isin(spi["household_id"])]
    tenure = spi.set_index("household_id")["tenure_type"]
    hb = spi_people[spi_people["housing_benefit_reported"] > 0]
    assert hb["person_household_id"].map(tenure).isin(RENTED_TENURES).all()
    assert hb["is_household_head"].all()


def test_draws_are_keyed_by_household_identity() -> None:
    person, _, household, weights = _tables()
    _, first, _ = impute_spi_housing_shell(
        person, household, weights, seed=0, n_estimators=20
    )
    order = np.random.default_rng(1).permutation(len(household))
    _, second, _ = impute_spi_housing_shell(
        person,
        household.iloc[order].reset_index(drop=True),
        weights[order],
        seed=0,
        n_estimators=20,
    )
    columns = [*HOUSING_CATEGORICAL_COLUMNS, *HOUSING_AMOUNT_COLUMNS]
    left = first.set_index("household_id")[columns].sort_index()
    right = second.set_index("household_id")[columns].sort_index()
    pd.testing.assert_frame_equal(left, right)


def test_stage_transform_runs_on_a_frame_and_records_its_receipt() -> None:
    person, benunit, household, weights = _tables()
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=weights,
    )
    transform = UKSPIHousingShellStageTransform(stage=_stage(), n_estimators=20)
    result = transform(frame)
    assert transform.output_columns() == ()
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == "spi_housing_shell"
    assert "rule_firings" in evidence
    np.testing.assert_array_equal(result.weights_for("household").values, weights)


def test_a_frame_without_frs_training_rows_is_refused() -> None:
    person, _, household, weights = _tables()
    household = household.assign(household_support_channel="spi")
    with pytest.raises(ValueError, match="FRS training rows"):
        impute_spi_housing_shell(person, household, weights, seed=0, n_estimators=20)
