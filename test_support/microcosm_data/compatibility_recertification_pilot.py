"""Public MOCK fixtures; no real records or runtime qualification evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mock_tables(sizes=(2, 1, 3), *, id_dtype="int64", offset=100):
    """Independent positional fixture with sparse IDs and unequal source weights."""
    dtype = np.dtype(id_dtype)
    household_ids = np.asarray(
        [offset + 11 * i for i in range(len(sizes))], dtype=dtype
    )
    people = []
    person_id = offset + 1000
    for position, (household, size) in enumerate(
        zip(household_ids, sizes, strict=True)
    ):
        for member in range(size):
            people.append(
                {
                    "person_id": person_id,
                    "person_household_id": household,
                    "person_tax_unit_id": int(household) + 100,
                    "person_spm_unit_id": int(household) + 200,
                    "person_family_id": int(household) + 300,
                    "person_marital_unit_id": person_id + 400,
                    "age": 40.0 if member == 0 else 10.0,
                    "employment_income": 45_000.0 + 1000 * position
                    if member == 0
                    else 0.0,
                    "tax_unit_role_input": "HEAD" if member == 0 else "DEPENDENT",
                }
            )
            person_id += 7
    person = pd.DataFrame(people)
    for column in person:
        if column.endswith("_id"):
            person[column] = person[column].astype(dtype)
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "household_weight": np.asarray(
                    [0.0 if i == 0 else i + 0.25 for i in range(len(sizes))]
                ),
                "state_fips": np.asarray(
                    [6 if i % 2 == 0 else 36 for i in range(len(sizes))], dtype="int32"
                ),
                "county_fips": np.asarray(
                    [6037 if i % 2 == 0 else 36061 for i in range(len(sizes))],
                    dtype="int32",
                ),
            }
        ),
    }
    for entity, increment in (("tax_unit", 100), ("spm_unit", 200), ("family", 300)):
        tables[entity] = pd.DataFrame(
            {
                f"{entity}_id": np.asarray(
                    [int(h) + increment for h in household_ids], dtype=dtype
                )
            }
        )
    tables["tax_unit"]["filing_status_input"] = "SINGLE"
    tables["marital_unit"] = pd.DataFrame(
        {"marital_unit_id": (person.person_id + 400).astype(dtype)}
    )
    return tables


def membership_inputs(tables):
    """Expose only the membership projection and original group-ID arrays."""
    columns = [c for c in tables["person"] if c.endswith("_id")]
    memberships = tables["person"].loc[:, columns].copy()
    groups = {
        entity: frame[f"{entity}_id"].to_numpy()
        for entity, frame in tables.items()
        if entity != "person"
    }
    return memberships, groups


def write_mock_h5(path, tables, *, year=2024):
    """Write synthetic six-table data for the engine-workflow category only."""
    with pd.HDFStore(path, "w") as store:
        store.put("_time_period", pd.Series([year]), format="table")
        for entity, frame in tables.items():
            store.put(entity, frame, format="table", data_columns=True)
