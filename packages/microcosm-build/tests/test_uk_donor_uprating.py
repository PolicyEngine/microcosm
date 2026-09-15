"""Declared donor uprating to the FRS 2024-25 base year (microcosm#890 U)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import donor_uprating as module
from microcosm.build.uk_runtime.donor_uprating import (
    apply_donor_uprating,
    donor_uprating_factors,
    uprating_operation,
)
from microcosm.build.uk_runtime.etb_services import (
    etb_donor_uprating,
    rail_fare_index_denominator_key,
)
from microcosm.build.uk_runtime.lcfs_consumption import lcfs_donor_uprating
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

ROOT = Path(__file__).resolve().parents[3]
UK_PACKAGE = ROOT / "packages/microcosm-build/src/microcosm/build/uk"
CPI = "gov.economic_assumptions.indices.obr.consumer_price_index"

FAKE_INDICES = {
    CPI: {2023: 1.51964, 2024: 1.55763},
    "gov.economic_assumptions.indices.obr.average_earnings": {2023: 1.0, 2024: 1.051},
    "gov.economic_assumptions.indices.obr.per_capita.mixed_income": {
        2023: 2.0,
        2024: 2.0546,
    },
    "gov.economic_assumptions.indices.obr.private_pension_index": {
        2023: 1.0,
        2024: 1.05,
    },
    "gov.economic_assumptions.indices.obr.petrol_spending_litre_proxy": {
        2023: 1.61756,
        2024: 1.37799,
    },
    "gov.economic_assumptions.indices.obr.diesel_spending_litre_proxy": {
        2023: 1.34477,
        2024: 1.38733,
    },
}


def _fake_reader(path: str, year: int) -> float:
    try:
        return FAKE_INDICES[path][year]
    except KeyError as error:  # pragma: no cover - guards a typo in a test
        raise ValueError(f"unexpected read {path!r} at {year}") from error


def _stages():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()


def test_both_donor_stages_declare_uprating_to_the_base_year() -> None:
    stages = _stages()
    for name in ("lcfs_consumption", "etb_services"):
        operation = uprating_operation(stages[name])
        assert operation is not None, name
        assert (operation["from_period"], operation["to_period"]) == (2023, 2024)
        assert [op.kind for op in stages[name].operations][:2] == [
            "derive",
            "uprate_donor_columns",
        ]
    lcfs = uprating_operation(stages["lcfs_consumption"])
    assert lcfs["exempt"] == [
        "electricity_consumption",
        "gas_consumption",
        "domestic_energy_consumption",
    ]
    assert lcfs["columns"]["petrol_spending"]["basis"] == "vendored_litre_proxy"
    assert lcfs["columns"]["diesel_spending"]["basis"] == "vendored_litre_proxy"
    assert lcfs["columns"]["bus_fare_spending"] == {
        "basis": "engine_parameter",
        "parameter_path": CPI,
    }
    etb = uprating_operation(stages["etb_services"])
    assert etb["columns"]["rail_subsidy_spending"]["concept"] == (
        "orr.rail.total_government_support"
    )
    assert etb["columns"]["bus_subsidy_spending"]["geography_id"] == "E92000001"


def test_committed_json_mirror_matches_the_yaml_declaration() -> None:
    stages = _stages()
    mirror = json.loads((UK_PACKAGE / "source_stages.json").read_text("utf-8"))
    by_name = {entry["stage"]: entry for entry in mirror["stages"]}
    for name in ("lcfs_consumption", "etb_services"):
        declared = [op["kind"] for op in by_name[name]["operations"]]
        assert declared == [op.kind for op in stages[name].operations]
        mirrored = next(
            op
            for op in by_name[name]["operations"]
            if op["kind"] == "uprate_donor_columns"
        )
        assert {k: v for k, v in mirrored.items() if k != "kind"} == uprating_operation(
            stages[name]
        )


def test_engine_parameter_basis_is_the_ratio_of_the_two_instants() -> None:
    factors, receipt = donor_uprating_factors(
        {
            "from_period": 2023,
            "to_period": 2024,
            "columns": {
                "food": {"basis": "engine_parameter", "parameter_path": CPI},
            },
            "exempt": ["gas"],
        },
        parameter_reader=_fake_reader,
    )
    assert factors == {"food": pytest.approx(1.55763 / 1.51964)}
    assert receipt["operation"] == "uprate_donor_columns"
    assert receipt["instant"] == "january_first"
    assert receipt["exempt"] == ["gas"]
    assert receipt["columns"]["food"] == {
        "basis": "engine_parameter",
        "factor": pytest.approx(1.55763 / 1.51964),
        "parameter_path": CPI,
        "before": 1.51964,
        "after": 1.55763,
    }


def test_litre_proxy_basis_reads_the_vendored_desnz_hmrc_ons_rows() -> None:
    stages = _stages()
    spec = uprating_operation(stages["lcfs_consumption"])["columns"]["petrol_spending"]
    factors, receipt = donor_uprating_factors(
        {"from_period": 2023, "to_period": 2024, "columns": {"petrol_spending": spec}},
        parameter_reader=_fake_reader,
    )
    detail = receipt["columns"]["petrol_spending"]
    price = vendored_rows(
        "road_fuel_anchors.json",
        concept="desnz.road_fuel.annual_ulsp_pump_price",
        period_type="calendar_year",
    )
    by_year = {row["period"]["value"]: row["value"] for row in price}
    litres = {
        row["period_coverage"]["start_date"]: row["value"]
        for row in vendored_rows(
            "road_fuel_anchors.json",
            concept="hmrc.hydrocarbon_oils.total_petrol_quantity",
            period_type="fiscal_year",
        )
    }
    population = {
        row["period"]["value"]: row["value"]
        for row in vendored_rows(
            "road_fuel_anchors.json",
            concept="ons.mid_year_population_estimate",
            period_type="calendar_year",
        )
    }
    expected = (
        (by_year[2024] / by_year[2023])
        * (litres["2024-04-01"] / litres["2023-04-01"])
        / (population[2024] / population[2023])
    )
    assert factors["petrol_spending"] == pytest.approx(expected)
    assert detail["formula"] == "price_ratio * litres_ratio / population_ratio"
    assert detail["price"]["ratio"] == pytest.approx(by_year[2024] / by_year[2023])
    # The published outturn: petrol pump prices fell and litres per head rose.
    assert 0.97 < factors["petrol_spending"] < 1.0
    audit = detail["engine_audit"]
    assert audit["parameter_path"] == (
        "gov.economic_assumptions.indices.obr.petrol_spending_litre_proxy"
    )
    assert audit["engine_ratio"] == pytest.approx(1.37799 / 1.61756)
    assert audit["applied"] == "outturn"
    assert audit["outturn_over_engine"] == pytest.approx(
        factors["petrol_spending"] / (1.37799 / 1.61756)
    )


def test_diesel_outturn_falls_while_the_engine_proxy_rises() -> None:
    """The recorded divergence behind the policyengine-uk pump-price issue."""

    stages = _stages()
    columns = uprating_operation(stages["lcfs_consumption"])["columns"]
    factors, receipt = donor_uprating_factors(
        {
            "from_period": 2023,
            "to_period": 2024,
            "columns": {
                "petrol_spending": columns["petrol_spending"],
                "diesel_spending": columns["diesel_spending"],
            },
        },
        parameter_reader=_fake_reader,
    )
    assert factors["diesel_spending"] < 0.92
    assert factors["petrol_spending"] > factors["diesel_spending"]
    audits = {
        column: receipt["columns"][column]["engine_audit"]["engine_ratio"]
        for column in ("petrol_spending", "diesel_spending")
    }
    assert audits["petrol_spending"] < 0.86
    assert audits["diesel_spending"] > 1.03


def test_vendored_ratio_basis_selects_by_fiscal_start_and_geography() -> None:
    stages = _stages()
    columns = uprating_operation(stages["etb_services"])["columns"]
    factors, receipt = donor_uprating_factors(
        {
            "from_period": 2023,
            "to_period": 2024,
            "columns": {
                "rail_subsidy_spending": columns["rail_subsidy_spending"],
                "bus_subsidy_spending": columns["bus_subsidy_spending"],
            },
        },
        parameter_reader=_fake_reader,
    )
    rail = {
        row["period_coverage"]["start_date"]: row["value"]
        for row in vendored_rows(
            "orr_rail_facts.json", concept="orr.rail.total_government_support"
        )
    }
    assert factors["rail_subsidy_spending"] == pytest.approx(
        rail["2024-04-01"] / rail["2023-04-01"]
    )
    assert receipt["columns"]["rail_subsidy_spending"]["numerator"] == {
        "fiscal_start": "2024-04-01",
        "value": rail["2024-04-01"],
    }
    bus = {
        row["period_coverage"]["start_date"]: row["value"]
        for row in vendored_rows(
            "dft_bus_value_anchors.json",
            concept="dft.local_bus_total_estimated_net_support",
            geography_id="E92000001",
        )
    }
    assert factors["bus_subsidy_spending"] == pytest.approx(
        bus["2024-04-01"] / bus["2023-04-01"]
    )
    assert receipt["columns"]["bus_subsidy_spending"]["geography_id"] == "E92000001"
    # Without the geography the England row is not unique across BUS05 areas.
    without = dict(columns["bus_subsidy_spending"])
    del without["geography_id"]
    with pytest.raises(ValueError, match="expected exactly one row"):
        donor_uprating_factors(
            {
                "from_period": 2023,
                "to_period": 2024,
                "columns": {"bus_subsidy_spending": without},
            },
            parameter_reader=_fake_reader,
        )


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        (
            {"from_period": 2024, "to_period": 2023, "columns": {"a": {}}},
            "backwards",
        ),
        ({"from_period": 2023, "to_period": 2024, "columns": {}}, "no columns"),
        (
            {
                "from_period": 2023,
                "to_period": 2024,
                "columns": {"a": {"basis": "engine_parameter", "parameter_path": CPI}},
                "exempt": ["a"],
            },
            "exempt columns as uprated",
        ),
        (
            {
                "from_period": 2023,
                "to_period": 2024,
                "columns": {"a": {"basis": "made_up"}},
            },
            "unknown basis",
        ),
    ],
)
def test_declaration_refusals(parameters: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        donor_uprating_factors(parameters, parameter_reader=_fake_reader)


def test_non_positive_engine_index_is_refused() -> None:
    def zero_reader(path: str, year: int) -> float:
        return 0.0 if year == 2023 else 1.0

    with pytest.raises(ValueError, match="must be positive"):
        donor_uprating_factors(
            {
                "from_period": 2023,
                "to_period": 2024,
                "columns": {"a": {"basis": "engine_parameter", "parameter_path": CPI}},
            },
            parameter_reader=zero_reader,
        )


def test_apply_donor_uprating_scales_declared_columns_only() -> None:
    table = pd.DataFrame(
        {
            "petrol_spending": [10.0, np.nan, "3"],
            "gas_consumption": [5.0, 6.0, 7.0],
            "label": ["a", "b", "c"],
        }
    )
    result = apply_donor_uprating(table, {"petrol_spending": 0.5})
    assert result["petrol_spending"].tolist() == [5.0, 0.0, 1.5]
    assert result["gas_consumption"].tolist() == [5.0, 6.0, 7.0]
    assert result["label"].tolist() == ["a", "b", "c"]
    assert table["petrol_spending"].tolist()[0] == 10.0
    with pytest.raises(KeyError, match="lacks uprated column"):
        apply_donor_uprating(table, {"missing": 1.0})


def test_stage_helpers_return_nothing_for_an_undeclared_stage() -> None:
    class Stage:
        operations = ()

    assert lcfs_donor_uprating(Stage()) == ({}, None)
    assert etb_donor_uprating(Stage()) == ({}, None)


def test_rail_denominator_is_the_2024_fare_index_anchor() -> None:
    assert rail_fare_index_denominator_key() == "rail_fare_index_2024"
    anchors = json.loads((UK_PACKAGE / "etb_services_anchors.json").read_text("utf-8"))
    assert anchors["rail_fare_index_2024"]["period"] == 2024
    assert anchors["rail_fare_index_2024"]["parameter_path"] == (
        "gov.dft.rail.fare_index"
    )
    assert (
        anchors["rail_fare_index_2024"]["value"]
        > anchors["rail_fare_index_2023"]["value"]
    )


@pytest.mark.requires_uk
def test_declared_factors_lockstep_with_the_installed_engine() -> None:
    pytest.importorskip("policyengine_uk")
    stages = _stages()
    reader = module.engine_parameter_reader()

    lcfs_factors, lcfs_receipt = lcfs_donor_uprating(stages["lcfs_consumption"])
    cpi_ratio = reader(CPI, 2024) / reader(CPI, 2023)
    cpi_columns = [
        column
        for column, spec in lcfs_receipt["columns"].items()
        if spec.get("parameter_path") == CPI
    ]
    assert len(cpi_columns) == 13
    for column in cpi_columns:
        assert lcfs_factors[column] == cpi_ratio, column
    assert lcfs_factors["employment_income"] == reader(
        "gov.economic_assumptions.indices.obr.average_earnings", 2024
    ) / reader("gov.economic_assumptions.indices.obr.average_earnings", 2023)
    assert lcfs_factors["petrol_spending"] == pytest.approx(0.9846, abs=5e-4)
    assert lcfs_factors["diesel_spending"] == pytest.approx(0.8977, abs=5e-4)
    audits = {
        column: lcfs_receipt["columns"][column]["engine_audit"]
        for column in ("petrol_spending", "diesel_spending")
    }
    assert audits["petrol_spending"]["engine_ratio"] == pytest.approx(0.852, abs=1e-3)
    assert audits["diesel_spending"]["engine_ratio"] == pytest.approx(1.032, abs=1e-3)

    etb_factors, _receipt = etb_donor_uprating(stages["etb_services"])
    assert etb_factors["dfe_education_spending"] == cpi_ratio
    assert etb_factors["rail_subsidy_spending"] == pytest.approx(0.9351, abs=5e-4)
    assert etb_factors["bus_subsidy_spending"] == pytest.approx(1.1407, abs=5e-4)


@pytest.mark.requires_uk
def test_rail_fare_index_anchor_and_orr_series_lockstep_with_the_engine() -> None:
    pytest.importorskip("policyengine_uk")
    reader = module.engine_parameter_reader()
    anchors = json.loads((UK_PACKAGE / "etb_services_anchors.json").read_text("utf-8"))
    key = rail_fare_index_denominator_key()
    assert (
        reader(anchors[key]["parameter_path"], anchors[key]["period"])
        == (anchors[key]["value"])
    )
    engine_step = reader("gov.dft.rail.fare_index", 2025) / reader(
        "gov.dft.rail.fare_index", 2024
    )
    orr = {
        row["period"]["value"]: row["value"]
        for row in vendored_rows(
            "orr_rail_facts.json",
            concept="orr.rail.fare_index",
            dimensions={"fare_category": "regulated_standard"},
        )
    }
    assert abs(engine_step / (orr[2025] / orr[2024]) - 1.0) < 0.01
