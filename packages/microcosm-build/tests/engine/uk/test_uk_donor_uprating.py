"""Tests split from packages/microcosm-build/tests/test_uk_donor_uprating.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_donor_uprating import *


def test_declared_factors_lockstep_with_the_installed_engine() -> None:
    __import__("policyengine_uk")
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


def test_rail_fare_index_anchor_and_orr_series_lockstep_with_the_engine() -> None:
    __import__("policyengine_uk")
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
