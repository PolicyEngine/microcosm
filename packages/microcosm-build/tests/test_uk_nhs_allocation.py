from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.etb_services import (
    allocate_nhs_by_age_gender,
    build_nhs_cell_table,
    load_etb_services_anchors,
    parse_nhs_age_bounds,
)


def _raw_nhs_rows() -> pd.DataFrame:
    rows = []
    for age_group, activity, cost in [
        ("80-84", 10.0, 100.0),
        ("85-89", 20.0, 300.0),
        ("90-94", 30.0, 600.0),
        ("95 years or older", 40.0, 1000.0),
    ]:
        for metric, total in (
            ("Activity Count", activity),
            ("Total Cost", cost),
        ):
            rows.append(
                {
                    "Age group": age_group,
                    "Gender": "Female",
                    "Service": "A&E",
                    "Metric": metric,
                    "Total": total,
                }
            )
    return pd.DataFrame(rows)


def test_nhs_age_bound_parsing_uses_half_open_top_code() -> None:
    assert parse_nhs_age_bounds("0 years") == (0, 1)
    assert parse_nhs_age_bounds("85-89") == (85, 90)
    assert parse_nhs_age_bounds("95 years or older") == (95, 120)


def test_nhs_native_top_bands_and_budget_normalization_use_full_table() -> None:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [1, 2, 3, 4],
            "age": [84, 85, 90, 95],
            "gender": ["female", "FEMALE", "FEMALE", "FEMALE"],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "household_weight": [2.0, 3.0, 4.0, 5.0],
        }
    )

    cells = build_nhs_cell_table(_raw_nhs_rows(), person, household)
    top = cells[cells["Lower age"] >= 85].sort_values("Lower age")

    assert list(zip(top["Lower age"], top["Upper age"], strict=True)) == [
        (85, 90),
        (90, 95),
        (95, 120),
    ]
    assert top["Activity Count"].tolist() == [20.0, 30.0, 40.0]
    assert top["Total Cost"].tolist() == [300.0, 600.0, 1000.0]
    assert top["Total people"].tolist() == [3.0, 4.0, 5.0]
    assert np.isclose(
        cells["Per-person average spending"].mul(cells["Total people"]).sum(),
        load_etb_services_anchors()["nhs_budget_2025_26"]["value"],
    )

    # The real frame's household table has no household_weight column (weights
    # live in the typed vector): the allocation must run from the passed array.
    allocated = allocate_nhs_by_age_gender(
        person,
        household_weights=household["household_weight"].to_numpy(dtype=float),
        household=household.drop(columns=["household_weight"]),
        nhs_table=_raw_nhs_rows(),
    )

    assert allocated.loc[0, "a_and_e_visits"] == 5.0
    assert allocated.loc[1, "a_and_e_visits"] == 20.0 / 3.0
    assert allocated.loc[2, "a_and_e_visits"] == 30.0 / 4.0
    assert allocated.loc[3, "a_and_e_visits"] == 40.0 / 5.0
    spending = allocated["nhs_a_and_e_spending"].to_numpy(dtype=float)
    assert np.all(np.diff(spending) > 0)


def test_an_empty_native_cell_is_named_in_the_receipt_and_allocates_nothing() -> None:
    # Nobody aged 95+ here: the donor cost for that cell reaches no recipient
    # and, because the budget factor normalizes the donor table rather than
    # the realized allocation, it is dropped. The receipt must say so rather
    # than let a placeholder count hide it.
    from microcosm.build.uk_runtime.etb_services import (
        allocate_nhs_by_age_gender,
        nhs_cell_receipt,
    )

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [1, 2, 3],
            "age": [84, 85, 90],
            "gender": ["FEMALE", "FEMALE", "FEMALE"],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_weight": [2.0, 3.0, 4.0],
        }
    )

    cells = build_nhs_cell_table(_raw_nhs_rows(), person, household)
    empty = cells[cells["Lower age"] == 95]
    receipt = nhs_cell_receipt(cells)

    assert (empty["Total people"] == 0).all()
    assert (empty["Per-person average spending"] == 0).all()
    assert (empty["Per-person average units"] == 0).all()
    assert receipt["empty_cells"] == [
        {
            "gender": "FEMALE",
            "lower_age": 95,
            "upper_age": 120,
            "services": sorted(empty["Service"].astype(str)),
        }
    ]
    assert receipt["empty_cell_count"] == len(empty)
    assert receipt["unallocated_cost_share"] == pytest.approx(
        float(empty["Total Cost"].sum()) / float(cells["Total Cost"].sum())
    )

    allocated, allocated_receipt = allocate_nhs_by_age_gender(
        person,
        household_weights=household["household_weight"].to_numpy(),
        household=household.drop(columns="household_weight"),
        nhs_table=_raw_nhs_rows(),
        with_receipt=True,
    )
    assert allocated_receipt == receipt
    assert allocated["nhs_a_and_e_spending"].gt(0).all()


def test_a_fully_occupied_table_reports_no_empty_cells() -> None:
    from microcosm.build.uk_runtime.etb_services import nhs_cell_receipt

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [1, 2, 3, 4],
            "age": [84, 85, 90, 95],
            "gender": ["FEMALE", "FEMALE", "FEMALE", "FEMALE"],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "household_weight": [2.0, 3.0, 4.0, 5.0],
        }
    )

    receipt = nhs_cell_receipt(build_nhs_cell_table(_raw_nhs_rows(), person, household))

    assert receipt == {
        "empty_cells": [],
        "empty_cell_count": 0,
        "unallocated_cost_share": 0.0,
    }


def test_nhs_age_bounds_parse_every_committed_resource_label() -> None:
    # Regression for the licensed-build crash on "01-04 years": the parser
    # must handle the committed resource's REAL labels, not just synthetic
    # fixtures.
    import json
    from pathlib import Path

    resource = (
        Path(__file__).resolve().parents[1]
        / "src/microcosm/build/uk/nhs_consumption_by_age_gender.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    labels = sorted({row["Age group"] for row in payload["rows"]})
    assert labels, "committed NHS resource has no rows"
    bounds = [parse_nhs_age_bounds(label) for label in labels]
    assert parse_nhs_age_bounds("01-04 years") == (1, 5)
    for lower, upper in bounds:
        assert 0 <= lower < upper <= 120
