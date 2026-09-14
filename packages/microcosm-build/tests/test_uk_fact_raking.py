"""Rakes of imputed columns to vendored publisher facts (#890 D)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.etb_services import (
    UK_ETB_SERVICES_VENDORED_RESOURCES,
    etb_bus_support_rake,
)
from microcosm.build.uk_runtime.fact_raking import (
    FactCell,
    person_income_quintiles,
    rake_operations,
    rake_to_facts,
    resolve_cells,
)
from microcosm.build.uk_runtime.lcfs_consumption import UK_LCFS_VENDORED_RESOURCES
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

OUTSIDE_LONDON = (
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "SOUTH_EAST",
    "SOUTH_WEST",
)


def _stage(name: str):
    return load_country_spec("uk").sources.stage_map()[name]


def _fact(resource: str, **criteria) -> float:
    rows = vendored_rows(resource, **criteria)
    assert rows, criteria
    return float(sum(row["value"] for row in rows))


def test_lcfs_fare_cells_resolve_to_the_fy2024_25_published_receipts() -> None:
    (operation,) = rake_operations(_stage("lcfs_consumption"))
    assert operation["columns"] == ["bus_fare_spending"]
    assert operation["scope"] == "users_only"
    assert operation["quintile_income"] == "hbai_household_net_income"

    cells = {
        cell.label: cell
        for cell in resolve_cells(
            operation, allowed_resources=UK_LCFS_VENDORED_RESOURCES
        )
    }
    assert set(cells) == {
        "london",
        "england_outside_london",
        "scotland",
        "northern_ireland",
    }
    assert cells["london"].regions == ("LONDON",)
    assert cells["england_outside_london"].regions == OUTSIDE_LONDON
    assert cells["london"].value == _fact(
        "dft_bus_value_anchors.json",
        concept="dft.local_bus_passenger_fare_receipts",
        fiscal_start="2024-04-01",
        geography_id="E12000007",
    )
    assert cells["england_outside_london"].value == _fact(
        "dft_bus_value_anchors.json",
        concept="dft.local_bus_passenger_fare_receipts",
        fiscal_start="2024-04-01",
        geography_id="dft:england_outside_london",
    )
    assert cells["scotland"].value == 391_000_000.0
    assert cells["northern_ireland"].value == pytest.approx(
        49_584_434.28 + 100_498_383.21
    )
    assert cells["london"].allocation == "income_quintile_trips"
    assert set(cells["london"].quintile_trips) == {
        "lowest",
        "second",
        "third",
        "fourth",
        "highest",
    }
    assert (
        cells["london"].quintile_trips["lowest"]
        > cells["london"].quintile_trips["third"]
    )
    assert cells["scotland"].allocation == "uniform"
    assert cells["scotland"].quintile_trips is None
    receipt = cells["northern_ireland"].receipt
    assert receipt["selector"]["sum_over"] == {"service": ["ulsterbus", "metro_glider"]}
    assert len(receipt["source_record_ids"]) == 2
    # Wales publishes no fare receipts: no cell names it.
    assert not any("WALES" in cell.regions for cell in cells.values())


def test_etb_support_cells_cover_every_nation_with_the_ni_joint_cell() -> None:
    (operation,) = rake_operations(_stage("etb_services"))
    assert operation["columns"] == ["bus_subsidy_spending"]
    assert operation["scope"] == "all"

    cells = {
        cell.label: cell
        for cell in resolve_cells(
            operation, allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES
        )
    }
    assert set(cells) == {
        "london",
        "england_outside_london",
        "scotland",
        "wales",
        "northern_ireland",
    }
    assert cells["london"].value == 1_130_214_000.0
    assert cells["england_outside_london"].value == pytest.approx(1_894_690_320.84)
    assert cells["scotland"].value == 499_000_000.0
    assert cells["wales"].value == pytest.approx(62_443_460 + 69_046_510)
    assert cells["northern_ireland"].value == pytest.approx(61_800_000 + 49_900_000)
    assert cells["northern_ireland"].joint_columns == (
        "bus_subsidy_spending",
        "rail_subsidy_spending",
    )
    assert all(cell.allocation == "uniform" for cell in cells.values())
    covered = {region for cell in cells.values() for region in cell.regions}
    assert covered == {
        "LONDON",
        *OUTSIDE_LONDON,
        "SCOTLAND",
        "WALES",
        "NORTHERN_IRELAND",
    }


def test_resolve_cells_refuses_resources_the_stage_does_not_declare() -> None:
    (operation,) = rake_operations(_stage("lcfs_consumption"))
    with pytest.raises(ValueError, match="does not declare"):
        resolve_cells(operation, allowed_resources=("road_fuel_anchors.json",))
    import copy

    duplicated = copy.deepcopy(operation)
    duplicated["cells"][1]["regions"].append("LONDON")
    with pytest.raises(ValueError, match="repeats regions"):
        resolve_cells(duplicated, allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    wrong_period = copy.deepcopy(operation)
    wrong_period["cells"][0]["selector"]["fiscal_start"] = "1999-04-01"
    with pytest.raises(ValueError, match="expected exactly one row"):
        resolve_cells(wrong_period, allowed_resources=UK_LCFS_VENDORED_RESOURCES)


def test_person_income_quintiles_rank_people_not_households() -> None:
    income = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    persons = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    weights = np.ones(6)
    labels, edges = person_income_quintiles(
        income, persons, weights, mask=np.array([True] * 5 + [False])
    )
    assert labels.tolist() == ["lowest", "second", "third", "fourth", "highest", ""]
    assert edges == [20.0, 30.0, 40.0, 50.0]
    # The last household's 100 persons dominate once it is in the mask: it
    # spans every quintile boundary, so all four edges sit at its income and
    # the five one-person households all fall in the lowest quintile.
    labels, edges = person_income_quintiles(income, persons, weights, mask=np.ones(6, bool))
    assert edges == [60.0, 60.0, 60.0, 60.0]
    assert labels.tolist()[:5] == ["lowest"] * 5


def _synthetic(n: int = 600, seed: int = 5) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    regions = rng.choice(
        ["LONDON", "SOUTH_EAST", "NORTH_WEST", "SCOTLAND", "WALES", "NORTHERN_IRELAND"],
        n,
    )
    frame = pd.DataFrame(
        {
            "bus_fare_spending": rng.uniform(0.0, 500.0, n),
            "bus_subsidy_spending": rng.uniform(10.0, 300.0, n),
            "rail_subsidy_spending": rng.uniform(10.0, 300.0, n),
        }
    )
    frame.loc[rng.random(n) < 0.3, "bus_fare_spending"] = 0.0
    inputs = {
        "region": regions,
        "weights": rng.uniform(0.5, 3.0, n),
        "users": rng.random(n) < 0.6,
        "income": rng.uniform(5e3, 1e5, n),
        "persons": rng.integers(1, 5, n).astype(float),
    }
    return frame, inputs


def test_users_only_rake_hits_each_cell_total_and_leaves_the_rest() -> None:
    frame, inputs = _synthetic()
    quintile_trips = {
        "lowest": 40.0,
        "second": 30.0,
        "third": 20.0,
        "fourth": 15.0,
        "highest": 10.0,
    }
    cells = [
        FactCell(
            "london",
            ("LONDON",),
            1_000_000.0,
            "income_quintile_trips",
            quintile_trips,
            (),
            {},
        ),
        FactCell(
            "outside",
            ("SOUTH_EAST", "NORTH_WEST"),
            2_000_000.0,
            "income_quintile_trips",
            quintile_trips,
            (),
            {},
        ),
        FactCell("scotland", ("SCOTLAND",), 300_000.0, "uniform", None, (), {}),
    ]
    raked, receipt = rake_to_facts(
        frame,
        columns=["bus_fare_spending"],
        cells=cells,
        region=inputs["region"],
        weights=inputs["weights"],
        scope="users_only",
        users=inputs["users"],
        quintile_income=inputs["income"],
        household_persons=inputs["persons"],
    )
    fares = raked["bus_fare_spending"].to_numpy()
    w = inputs["weights"]
    region = inputs["region"]
    users = inputs["users"]

    def total(mask):
        return float(np.dot(fares[mask], w[mask]))

    assert total((region == "LONDON") & users) == pytest.approx(1_000_000.0)
    assert total(
        np.isin(region, ["SOUTH_EAST", "NORTH_WEST"]) & users
    ) == pytest.approx(2_000_000.0)
    assert total((region == "SCOTLAND") & users) == pytest.approx(300_000.0)
    # Non-users are outside the rake scope and Wales names no cell.
    assert (fares[~users] == frame["bus_fare_spending"].to_numpy()[~users]).all()
    assert (
        fares[region == "WALES"]
        == frame["bus_fare_spending"].to_numpy()[region == "WALES"]
    ).all()
    assert (raked["bus_subsidy_spending"] == frame["bus_subsidy_spending"]).all()
    # Within London the quintile shares follow persons x trips per person.
    london_fits = [fit for fit in receipt["fits"] if fit["label"] == "london"]
    assert len(london_fits) == 5
    assert sum(fit["share"] for fit in london_fits) == pytest.approx(1.0)
    assert all(
        fit["weighted_total_after"]["bus_fare_spending"]
        == pytest.approx(fit["target_total"])
        for fit in receipt["fits"]
    )
    assert len(receipt["quintile_edges"]) == 4
    assert receipt["scope"] == "users_only"
    assert receipt["joint_fits"] == []


def test_joint_cell_scales_both_columns_by_one_factor() -> None:
    frame, inputs = _synthetic()
    cells = [
        FactCell(
            "northern_ireland",
            ("NORTHERN_IRELAND",),
            50_000.0,
            "uniform",
            None,
            ("bus_subsidy_spending", "rail_subsidy_spending"),
            {},
        ),
        FactCell("wales", ("WALES",), 40_000.0, "uniform", None, (), {}),
    ]
    raked, receipt = rake_to_facts(
        frame,
        columns=["bus_subsidy_spending"],
        cells=cells,
        region=inputs["region"],
        weights=inputs["weights"],
        scope="all",
    )
    ni = inputs["region"] == "NORTHERN_IRELAND"
    w = inputs["weights"]
    joint_total = float(
        np.dot(raked["bus_subsidy_spending"].to_numpy()[ni], w[ni])
        + np.dot(raked["rail_subsidy_spending"].to_numpy()[ni], w[ni])
    )
    assert joint_total == pytest.approx(50_000.0)
    ratio_bus = (
        raked["bus_subsidy_spending"].to_numpy()[ni]
        / frame["bus_subsidy_spending"].to_numpy()[ni]
    )
    ratio_rail = (
        raked["rail_subsidy_spending"].to_numpy()[ni]
        / frame["rail_subsidy_spending"].to_numpy()[ni]
    )
    assert np.allclose(ratio_bus, ratio_bus[0]) and np.allclose(
        ratio_rail, ratio_bus[0]
    )
    (joint,) = receipt["joint_fits"]
    assert joint["factor"] == pytest.approx(ratio_bus[0])
    wales = inputs["region"] == "WALES"
    assert float(np.dot(raked["bus_subsidy_spending"].to_numpy()[wales], w[wales])) == (
        pytest.approx(40_000.0)
    )
    # Rail outside Northern Ireland is untouched by a joint cell elsewhere.
    assert (
        raked["rail_subsidy_spending"][~ni] == frame["rail_subsidy_spending"][~ni]
    ).all()


def test_rake_refuses_a_cell_with_published_mass_but_nobody_in_scope() -> None:
    frame, inputs = _synthetic()
    cells = [FactCell("london", ("LONDON",), 1_000.0, "uniform", None, (), {})]
    with pytest.raises(ValueError, match="no households in scope"):
        rake_to_facts(
            frame,
            columns=["bus_fare_spending"],
            cells=cells,
            region=inputs["region"],
            weights=inputs["weights"],
            scope="users_only",
            users=np.zeros(len(frame), dtype=bool),
        )
    zeroed = frame.copy()
    zeroed["bus_fare_spending"] = 0.0
    with pytest.raises(ValueError, match="current mean is zero"):
        rake_to_facts(
            zeroed,
            columns=["bus_fare_spending"],
            cells=cells,
            region=inputs["region"],
            weights=inputs["weights"],
            scope="all",
        )
    with pytest.raises(ValueError, match="unknown rake scope"):
        rake_to_facts(
            frame,
            columns=["bus_fare_spending"],
            cells=cells,
            region=inputs["region"],
            weights=None,
            scope="some",
        )


def test_etb_stage_helper_applies_the_declared_support_rake() -> None:
    frame, inputs = _synthetic(n=2000, seed=9)
    rng = np.random.default_rng(1)
    regions = rng.choice(
        ["LONDON", *OUTSIDE_LONDON, "SCOTLAND", "WALES", "NORTHERN_IRELAND"], len(frame)
    )
    raked, receipt = etb_bus_support_rake(
        _stage("etb_services"),
        frame,
        household=pd.DataFrame({"region": regions}),
        weights=inputs["weights"],
    )
    assert receipt is not None
    assert receipt["scope"] == "all"
    w = inputs["weights"]
    london = regions == "LONDON"
    assert float(
        np.dot(raked["bus_subsidy_spending"].to_numpy()[london], w[london])
    ) == (pytest.approx(1_130_214_000.0))
    ni = regions == "NORTHERN_IRELAND"
    assert float(
        np.dot(raked["bus_subsidy_spending"].to_numpy()[ni], w[ni])
        + np.dot(raked["rail_subsidy_spending"].to_numpy()[ni], w[ni])
    ) == pytest.approx(111_700_000.0)

    class Stage:
        operations = ()

    untouched, none = etb_bus_support_rake(
        Stage(), frame, household=pd.DataFrame({"region": regions}), weights=w
    )
    assert none is None and untouched is frame
