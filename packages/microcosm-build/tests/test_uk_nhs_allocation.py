from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.etb_services import (
    NHS_BUDGET_2025_26,
    allocate_nhs_by_age_gender,
    build_nhs_cell_table,
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


def test_nhs_85_plus_fold_in_and_budget_normalization_use_full_table() -> None:
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_household_id": [1, 2],
            "age": [84, 85],
            "gender": ["female", "FEMALE"],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [2.0, 3.0],
        }
    )

    cells = build_nhs_cell_table(_raw_nhs_rows(), person, household)
    top = cells[cells["Lower age"] == 85].iloc[0]

    assert top["Upper age"] == 120
    assert top["Activity Count"] == 90.0
    assert top["Total Cost"] == 1900.0
    assert top["Total people"] == 3.0
    assert np.isclose(
        cells["Per-person average spending"].mul(cells["Total people"]).sum(),
        NHS_BUDGET_2025_26,
    )

    allocated = allocate_nhs_by_age_gender(
        person,
        household_weights=household["household_weight"].to_numpy(dtype=float),
        household=household,
        nhs_table=_raw_nhs_rows(),
    )

    assert allocated.loc[0, "a_and_e_visits"] == 5.0
    assert allocated.loc[1, "a_and_e_visits"] == 30.0
    assert allocated.loc[0, "nhs_a_and_e_spending"] < allocated.loc[
        1, "nhs_a_and_e_spending"
    ]


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
