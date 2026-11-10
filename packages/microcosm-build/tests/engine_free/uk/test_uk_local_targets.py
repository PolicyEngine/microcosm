from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    area_groups_from_codes,
    compute_household_metrics,
    local_targets,
    metric_names,
    metric_names_from_target_profile,
)
from microcosm.build.uk_runtime.domestic_energy_targets import (
    household_energy_kwh,
    household_energy_prices,
)
from microcosm.build.uk_runtime.local_targets import (
    AREA_TYPES,
    local_metric_bindings,
)
from test_support.microcosm_build.uk_local_metric_fixtures import (
    StubUKEngine,
    local_metric_resolver,
    three_household_engine,
    three_household_resolver,
)


def test_metric_names_match_expected_constituency_surface() -> None:
    names = metric_names("constituency")

    assert names[:4] == (
        "hmrc/self_employment_income/amount",
        "hmrc/self_employment_income/count",
        "hmrc/employment_income/amount",
        "hmrc/employment_income/count",
    )
    assert "age/0_10" in names
    assert "age/70_80" in names
    assert names[-14:] == (
        "uc_hh_0_children",
        "uc_hh_1_child",
        "uc_hh_2_children",
        "uc_hh_3plus_children",
        # Appended last so incumbent metric positions never renumber.
        "households",
        "hmrc/higher_rate_taxpayers/count",
        # microcosm#1240: the high-value homes family, after every earlier one.
        "high_value_homes/over_1m",
        "high_value_homes/over_1_5m",
        "high_value_homes/over_2m",
        # microcosm#1095: DESNZ's energy split onto constituencies, appended.
        "energy/electricity_kwh",
        "energy/gas_kwh",
        "energy/gas_connected_households",
        # microcosm#1418: SPI pension income by area, appended.
        "hmrc/pension_income/amount",
        "hmrc/pension_income/count",
    )


def test_metric_names_match_expected_la_surface() -> None:
    names = metric_names("la")

    assert "uc_households" in names
    assert "ons/equiv_net_income_bhc" in names
    assert "tenure/private_rent" in names
    assert "rent/private_rent" in names
    assert "uc_hh_0_children" not in names
    assert names[-20:] == (
        "households",
        "council_tax/band_a",
        "council_tax/band_b",
        "council_tax/band_c",
        "council_tax/band_d",
        "council_tax/band_e",
        "council_tax/band_f",
        "council_tax/band_g",
        "council_tax/band_h",
        "council_tax/band_i",
        "hmrc/higher_rate_taxpayers/count",
        "high_value_homes/over_1m",
        "high_value_homes/over_1_5m",
        "high_value_homes/over_2m",
        "energy/electricity_kwh",
        "energy/gas_kwh",
        "energy/gas_connected_households",
        # microcosm#1095: England's single adult discount by authority.
        "council_tax/single_adult_discount",
        # microcosm#1418: SPI pension income by area, both grains.
        "hmrc/pension_income/amount",
        "hmrc/pension_income/count",
    )
    # Both grains carry the energy family in the same order (microcosm#1095)
    # and end on pension income; the single adult discount stays
    # local-authority only.
    assert metric_names("constituency")[-5:-2] == names[-6:-3]
    assert metric_names("constituency")[-2:] == names[-2:]
    assert "council_tax/single_adult_discount" not in metric_names("constituency")


def test_metric_names_can_come_from_ledger_target_profile() -> None:
    profile = _ledger_target_profile(
        [
            _profile_target(
                "hmrc.employment_income.count",
                "hmrc/employment_income/count",
                ["constituency", "local_authority"],
            ),
            _profile_target(
                "ons.age.0_10",
                "age/0_10",
                ["constituency", "local_authority"],
            ),
            _profile_target(
                "dwp.uc.households_by_area_children_3plus",
                "uc_hh_3plus_children",
                ["constituency"],
            ),
            _profile_target(
                "ons.rent.private_rent",
                "rent/private_rent",
                ["local_authority"],
            ),
        ]
    )

    assert metric_names_from_target_profile(profile, "constituency") == (
        "hmrc/employment_income/count",
        "age/0_10",
        "uc_hh_3plus_children",
    )
    assert metric_names("la", target_profile=profile) == (
        "hmrc/employment_income/count",
        "age/0_10",
        "rent/private_rent",
    )


def test_compute_household_metrics_can_use_ledger_profile_subset(
    tmp_path, monkeypatch
) -> None:
    profile = _ledger_target_profile(
        [
            _profile_target(
                "hmrc.employment_income.count",
                "hmrc/employment_income/count",
                ["constituency"],
            ),
            _profile_target("ons.age.0_10", "age/0_10", ["constituency"]),
            _profile_target(
                "dwp.uc.households_by_area",
                "uc_households",
                ["constituency"],
            ),
        ]
    )

    metrics = compute_household_metrics(
        three_household_resolver(tmp_path, monkeypatch),
        "constituency",
        target_profile=profile,
    )

    assert metrics.columns.tolist() == [
        "hmrc/employment_income/count",
        "age/0_10",
        "uc_households",
    ]
    assert metrics["hmrc/employment_income/count"].tolist() == [1.0, 0.0, 1.0]
    assert metrics["age/0_10"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["uc_households"].tolist() == [0.0, 1.0, 1.0]


def test_metric_names_from_target_profile_rejects_duplicates() -> None:
    profile = _ledger_target_profile(
        [
            _profile_target("first", "age/0_10", ["constituency"]),
            _profile_target("second", "age/0_10", ["constituency"]),
        ]
    )

    with pytest.raises(ValueError, match="duplicate"):
        metric_names_from_target_profile(profile, "constituency")


def test_compute_constituency_household_metrics(tmp_path, monkeypatch) -> None:
    metrics = compute_household_metrics(
        three_household_resolver(tmp_path, monkeypatch), "constituency"
    )

    assert metrics.index.tolist() == [101, 102, 103]
    assert metrics.columns.tolist() == list(metric_names("constituency"))
    assert metrics["hmrc/self_employment_income/amount"].tolist() == [0.0, 0.0, 50.0]
    assert metrics["hmrc/employment_income/count"].tolist() == [1.0, 0.0, 1.0]
    # Higher rate taxpayers only: the second person is HIGHER but pays no tax.
    assert metrics["hmrc/higher_rate_taxpayers/count"].tolist() == [1.0, 0.0, 1.0]
    assert metrics["age/0_10"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["age/70_80"].tolist() == [0.0, 1.0, 0.0]
    assert metrics["uc_households"].tolist() == [0.0, 1.0, 1.0]
    assert metrics["uc_hh_0_children"].tolist() == [0.0, 1.0, 0.0]
    assert metrics["uc_hh_1_child"].tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("time_period", ["2024", "2025"])
def test_the_area_self_employment_rows_count_the_profit_after_loss_relief(
    tmp_path, monkeypatch, time_period
) -> None:
    # HMRC counts the profit after losses brought forward (microcosm#1095),
    # read from the frame at its own period and from the engine at any other.
    def metrics_with(loss_relief):
        engine = three_household_engine(loss_relief=loss_relief)
        resolver = three_household_resolver(
            tmp_path, monkeypatch, engine=engine, time_period=time_period
        )
        return compute_household_metrics(resolver, "constituency")

    metrics = metrics_with([0.0, 0.0, 0.0, 20.0, 0.0])
    assert metrics["hmrc/self_employment_income/amount"].tolist() == [0.0, 0.0, 30.0]
    assert metrics["hmrc/self_employment_income/count"].tolist() == [0.0, 0.0, 1.0]
    # Losses that absorb the whole profit leave no self-employment income.
    metrics = metrics_with([0.0, 0.0, 0.0, 80.0, 0.0])
    assert metrics["hmrc/self_employment_income/amount"].tolist() == [0.0, 0.0, 0.0]
    assert metrics["hmrc/self_employment_income/count"].tolist() == [0.0, 0.0, 0.0]


def test_uc_child_bands_sum_uc_benefit_units_to_household(
    tmp_path, monkeypatch
) -> None:
    # Three benefit units in one household: a claimant with one, three and
    # two children (the last two are 19 and 18, qualifying young people the
    # engine's age-18 count would drop). The bands read the FRS roles, as the
    # national UC-by-children rows do (microcosm#1095).
    people = 9
    engine = StubUKEngine(
        {
            "self_employment_income": [0.0] * people,
            "loss_relief": [0.0] * people,
            "employment_income": [0.0] * people,
            "benefits_in_kind": [0.0] * people,
            "total_pension_income": [0.0] * people,
            "income_tax": [0.0] * people,
            "income_tax_before_winter_fuel_payment_charge": [0.0] * people,
            "tax_band": ["NONE"] * people,
            "age": [35, 10, 40, 3, 6, 9, 45, 19, 18],
            "universal_credit": [100.0, 50.0, 0.0],
            "is_child_or_qualifying_young_person_for_universal_credit": [
                False,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
            ],
            "tenure_type": ["RENT_PRIVATELY"],
            # The constituency grain reads the energy spend too (microcosm#1095).
            "electricity_consumption": [0.0],
            "gas_consumption": [0.0],
            "region": ["LONDON"],
        },
        person_benunit=[0, 0, 1, 1, 1, 1, 2, 2, 2],
        person_household=[0] * people,
        benunit_household=[0, 0, 0],
    )
    heads = [True, False, True, False, False, False, True, False, False]
    resolver = local_metric_resolver(
        engine,
        household_ids=[101],
        is_benunit_head=heads,
        is_parent=heads,
        dependent_children=[1, 3, 2],
        main_residence_value=[0.0],
        scratch_dir=tmp_path,
        monkeypatch=monkeypatch,
    )

    metrics = compute_household_metrics(resolver, "constituency")

    assert metrics["uc_households"].tolist() == [2.0]
    assert metrics["uc_hh_0_children"].tolist() == [0.0]
    assert metrics["uc_hh_1_child"].tolist() == [1.0]
    assert metrics["uc_hh_2_children"].tolist() == [0.0]
    assert metrics["uc_hh_3plus_children"].tolist() == [1.0]


def test_compute_la_household_metrics(tmp_path, monkeypatch) -> None:
    metrics = compute_household_metrics(
        three_household_resolver(tmp_path, monkeypatch), "la"
    )

    assert metrics.index.tolist() == [101, 102, 103]
    assert metrics.columns.tolist() == list(metric_names("la"))
    assert metrics["ons/equiv_housing_costs"].tolist() == [20.0, 50.0, 40.0]
    assert metrics["tenure/owned_outright"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["tenure/private_rent"].tolist() == [0.0, 1.0, 0.0]
    assert metrics["tenure/social_rent"].tolist() == [0.0, 0.0, 1.0]
    assert metrics["rent/private_rent"].tolist() == [0.0, 10.0, 0.0]
    assert metrics["council_tax/band_a"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["council_tax/band_d"].tolist() == [0.0, 0.0, 1.0]
    assert metrics["council_tax/band_h"].tolist() == [0.0, 1.0, 0.0]
    # Every household takes the discount; only the first is in England.
    assert metrics["council_tax/single_adult_discount"].tolist() == [1.0, 0.0, 0.0]
    # The pensioner, the second household, pays income tax and so carries
    # their pension.
    assert metrics["hmrc/pension_income/amount"].tolist() == [0.0, 90.0, 0.0]
    assert metrics["hmrc/pension_income/count"].tolist() == [0.0, 1.0, 0.0]


def test_la_single_adult_discount_counts_english_households_at_the_rate(
    tmp_path, monkeypatch
) -> None:
    """The national row's two conditions, read on each household (#1095)."""

    engine = three_household_engine(
        country=["ENGLAND", "ENGLAND", "ENGLAND"],
        council_tax_discount_rate=[0.25, 0.0, 0.25],
    )
    metrics = compute_household_metrics(
        three_household_resolver(tmp_path, monkeypatch, engine=engine), "la"
    )

    assert metrics["council_tax/single_adult_discount"].tolist() == [1.0, 0.0, 1.0]


def test_la_energy_metrics_read_the_stored_spend_back_into_kwh(
    tmp_path, monkeypatch
) -> None:
    """The stage's calendar-2024 spend, at the dataset period (#1207)."""

    engine = three_household_engine()
    resolver = three_household_resolver(
        tmp_path, monkeypatch, engine=engine, time_period="2023"
    )
    metrics = compute_household_metrics(resolver, "la", period=2025, input_period=2023)

    kwh = household_energy_kwh(
        engine.values["electricity_consumption"],
        engine.values["gas_consumption"],
        region=engine.values["region"],
        prices=household_energy_prices(),
    )
    assert metrics["energy/electricity_kwh"].tolist() == pytest.approx(
        kwh["electricity_kwh"].tolist()
    )
    assert metrics["energy/gas_kwh"].tolist() == pytest.approx(kwh["gas_kwh"].tolist())
    assert metrics["energy/gas_connected_households"].tolist() == [1.0, 0.0, 1.0]
    assert metrics["energy/gas_kwh"].iloc[1] == 0.0
    # Read where they are stored, never where the engine would uprate them.
    for variable in ("electricity_consumption", "gas_consumption", "region"):
        assert engine.periods[variable] == {2023}, variable


def test_the_local_hmrc_rows_read_each_persons_tax_year() -> None:
    # As the national HMRC income rows do (microcosm#1334): the values, the
    # taxpayer test and the rate band on each person's income over the year.
    for area_type in ("constituency", "la"):
        bindings = local_metric_bindings(area_type)
        for name in metric_names(area_type):
            if not name.startswith("hmrc/"):
                continue
            binding = bindings[name]
            read = [
                binding["value_variable"],
                *(predicate["variable"] for predicate in binding["filters"]),
            ]
            assert "income_tax_before_winter_fuel_payment_charge_annual" in read, name
            assert all(
                variable == "person_count" or variable.endswith("_annual")
                for variable in read
            ), (name, read)


def test_every_local_metric_is_read_from_its_declaration() -> None:
    for area_type in AREA_TYPES:
        bindings = local_metric_bindings(area_type)
        assert set(metric_names(area_type)) <= set(bindings), area_type
        for name in metric_names(area_type):
            assert bindings[name]["metric_name"] == name


def _target(target_id, **binding):
    return {
        "target_id": target_id,
        "geography_levels": ["constituency"],
        "bindings": {"policyengine": {"metric_name": "households", **binding}},
    }


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (
            [
                _target("a", value_variable="household_count"),
                _target("b", value_variable="person_count"),
            ],
            "two different bindings",
        ),
        (
            [_target("a", value_variable="person_count", measurement_period=2024)],
            "measurement_period",
        ),
        (
            [
                _target(
                    "a",
                    value_variable="person_count",
                    from_entity="person",
                    map_to="benunit",
                )
            ],
            "benunit",
        ),
        *(
            (
                [
                    _target(
                        "a",
                        value_variable="person_count",
                        from_entity="person",
                        filters=[
                            {"variable": "age", "operator": ">", "value": 0, **extra}
                        ],
                    )
                ],
                # The refusal lists the key it does not honour.
                re.escape(repr([key])),
            )
            for key, extra in (
                ("entity", {"entity": "benunit"}),
                ("reduce", {"reduce": "any"}),
                ("concept", {"concept": "uk.person.age"}),
            )
        ),
    ],
)
def test_a_local_metric_the_local_line_cannot_read_refuses(
    monkeypatch, targets, message
) -> None:
    monkeypatch.setattr(
        local_targets, "load_uk_population_contract", lambda: {"targets": targets}
    )
    local_targets._local_metric_bindings.cache_clear()
    try:
        with pytest.raises(ValueError, match=message):
            local_metric_bindings("constituency")
    finally:
        local_targets._local_metric_bindings.cache_clear()


def test_compute_household_metrics_refuses_another_period(
    tmp_path, monkeypatch
) -> None:
    resolver = three_household_resolver(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="does not match target period"):
        compute_household_metrics(resolver, "constituency", period=2026)
    with pytest.raises(ValueError, match="is not the frame's period"):
        compute_household_metrics(
            resolver, "constituency", period=2025, input_period=2023
        )


def test_compute_household_metrics_validates_area_type(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="area_type"):
        compute_household_metrics(three_household_resolver(tmp_path, monkeypatch), "oa")


def test_area_groups_use_country_column_then_code_prefix() -> None:
    codes = pd.DataFrame({"code": ["E001", "S001"], "country": ["England", "Scotland"]})

    assert area_groups_from_codes(codes) == {"E001": "England", "S001": "Scotland"}
    assert area_groups_from_codes(["W001", "N001"]) == {
        "W001": "Wales",
        "N001": "Northern Ireland",
    }


def test_area_groups_reject_unknown_area_code_prefix() -> None:
    with pytest.raises(ValueError, match="Unknown UK local-area code prefix"):
        area_groups_from_codes(["X001"])


@pytest.mark.parametrize("code", [None, np.nan, ""])
def test_area_groups_reject_missing_or_blank_area_codes(code) -> None:
    with pytest.raises(ValueError, match="must not be"):
        area_groups_from_codes([code])


def _ledger_target_profile(targets: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "policyengine_ledger.target_profile.v1",
        "profile_id": "uk_local_geography",
        "country": "uk",
        "label": "UK local geography",
        "defaults": {
            "base_period_policy": "latest_not_after_build_base_period",
            "operation": "sum",
        },
        "targets": targets,
    }


def _profile_target(
    target_id: str,
    metric_name: str,
    geography_levels: list[str],
) -> dict[str, object]:
    return {
        "target_id": target_id,
        "geography_levels": geography_levels,
        "bindings": {"policyengine": {"metric_name": metric_name}},
    }


def test_brma_codes_and_metrics_cover_scotland_only(tmp_path, monkeypatch):
    """microcosm#1305 review: the partial BRMA grain's codes and its one metric."""
    from microcosm.build.uk_runtime.local_targets import uk_brma_area_codes

    lothian, cardiff = uk_brma_area_codes(["LOTHIAN", "CARDIFF"])
    assert lothian.startswith("S33") and cardiff == ""
    metrics = compute_household_metrics(
        three_household_resolver(tmp_path, monkeypatch), "brma"
    )
    assert metrics.columns.tolist() == ["rent/private_rent_brma"]
    assert metrics["rent/private_rent_brma"].tolist() == [0.0, 10.0, 0.0]
