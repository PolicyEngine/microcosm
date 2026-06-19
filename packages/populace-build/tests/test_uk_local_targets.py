from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk import (
    area_groups_from_codes,
    compute_household_metrics,
    metric_names,
)


class Result:
    def __init__(self, values):
        self.values = np.asarray(values)


class FakeUKSimulation:
    def __init__(self):
        self.person_household = np.asarray([0, 0, 1, 2])
        self.benunit_household = np.asarray([0, 1, 2])
        self.data = {
            "household_id": [101, 102, 103],
            "self_employment_income": [0.0, 100.0, 0.0, 50.0],
            "employment_income": [10.0, 20.0, 0.0, 30.0],
            "income_tax": [1.0, 0.0, 2.0, 3.0],
            "age": [5, 35, 72, 12],
            "universal_credit": [0.0, 100.0, 50.0],
            "is_child": [1.0, 0.0, 0.0, 1.0],
            "equiv_hbai_household_net_income": [100.0, 200.0, 300.0],
            "equiv_hbai_household_net_income_ahc": [80.0, 150.0, 260.0],
            "tenure_type": [
                "OWNED_OUTRIGHT",
                "RENT_PRIVATELY",
                "RENT_FROM_COUNCIL",
            ],
            "benunit_rent": [0.0, 10.0, 20.0],
        }

    def calculate(self, variable, **_kwargs):
        return Result(self.data[variable])

    def map_result(self, values, from_entity, to_entity):
        assert to_entity == "household"
        values = np.asarray(values, dtype=float)
        out = np.zeros(3, dtype=float)
        if from_entity == "person":
            for i, household_index in enumerate(self.person_household):
                out[household_index] += values[i]
            return out
        if from_entity == "benunit":
            for i, household_index in enumerate(self.benunit_household):
                out[household_index] += values[i]
            return out
        raise AssertionError(f"unexpected from_entity {from_entity}")


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
    assert names[-4:] == (
        "uc_hh_0_children",
        "uc_hh_1_child",
        "uc_hh_2_children",
        "uc_hh_3plus_children",
    )


def test_metric_names_match_expected_la_surface() -> None:
    names = metric_names("la")

    assert "uc_households" in names
    assert "ons/equiv_net_income_bhc" in names
    assert "tenure/private_rent" in names
    assert "rent/private_rent" in names
    assert "uc_hh_0_children" not in names


def test_compute_constituency_household_metrics() -> None:
    metrics = compute_household_metrics(FakeUKSimulation(), "constituency")

    assert metrics.index.tolist() == [101, 102, 103]
    assert metrics.columns.tolist() == list(metric_names("constituency"))
    assert metrics["hmrc/self_employment_income/amount"].tolist() == [0.0, 0.0, 50.0]
    assert metrics["hmrc/employment_income/count"].tolist() == [1.0, 0.0, 1.0]
    assert metrics["age/0_10"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["age/70_80"].tolist() == [0.0, 1.0, 0.0]
    assert metrics["uc_households"].tolist() == [0.0, 1.0, 1.0]
    assert metrics["uc_hh_0_children"].tolist() == [0.0, 1.0, 0.0]
    assert metrics["uc_hh_1_child"].tolist() == [0.0, 0.0, 1.0]


def test_compute_la_household_metrics() -> None:
    metrics = compute_household_metrics(FakeUKSimulation(), "la")

    assert metrics.index.tolist() == [101, 102, 103]
    assert metrics.columns.tolist() == list(metric_names("la"))
    assert metrics["ons/equiv_housing_costs"].tolist() == [20.0, 50.0, 40.0]
    assert metrics["tenure/owned_outright"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["tenure/private_rent"].tolist() == [0.0, 1.0, 0.0]
    assert metrics["tenure/social_rent"].tolist() == [0.0, 0.0, 1.0]
    assert metrics["rent/private_rent"].tolist() == [0.0, 10.0, 0.0]


def test_compute_household_metrics_validates_area_type() -> None:
    with pytest.raises(ValueError, match="area_type"):
        compute_household_metrics(FakeUKSimulation(), "oa")


def test_area_groups_use_country_column_then_code_prefix() -> None:
    codes = pd.DataFrame(
        {"code": ["E001", "S001"], "country": ["England", "Scotland"]}
    )

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
