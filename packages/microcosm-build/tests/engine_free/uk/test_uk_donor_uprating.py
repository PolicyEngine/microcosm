"""Tests split from packages/microcosm-build/tests/test_uk_donor_uprating.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_donor_uprating import *


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
