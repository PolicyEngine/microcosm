"""The UK firm export against the policyengine-uk firm schema snapshot.

``tests/fixtures/uk/policyengine_uk_firm_inputs.json`` is policyengine-uk's
firm variables as ``tools/snapshot_policyengine_uk_firm_inputs.py`` read them
from source (PolicyEngine/policyengine-uk#2940). These tests hold the export
to that snapshot without importing policyengine-uk: refresh the snapshot with
the tool when policyengine-uk's firm variables change.
"""

from __future__ import annotations

import json
from functools import cache

import numpy as np
import pandas as pd

from microcosm.build.firms import (
    EntityVariables,
    export_schema_problems,
    input_status_problems,
)
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_INPUT_STATUSES,
    UK_FIRM_MICROCOSM_ONLY_COLUMNS,
    UK_FIRM_POLICYENGINE_UK_COLUMNS,
    UK_FIRM_POLICYENGINE_UK_INPUT_STATUS,
    UK_FIRM_SIC_SECTIONS,
    UK_FIRM_TABLE_COLUMNS,
    to_policyengine_uk_firm_table,
)
from test_support.paths import paths_for

SNAPSHOT = (
    paths_for("microcosm-build").tests
    / "fixtures"
    / "uk"
    / "policyengine_uk_firm_inputs.json"
)


@cache
def _snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def _variables() -> EntityVariables:
    snapshot = _snapshot()
    return EntityVariables(
        variables=snapshot["firm_variables"],
        input_names=snapshot["firm_input_names"],
        variable_names=snapshot["policyengine_uk_variable_names"],
    )


def _variable(name: str) -> dict:
    return _snapshot()["firm_variables"][name]


def _firms() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firm_id": np.array([4, 2, 1, 3], dtype=np.int64),
            "source_firm_id": np.array([4, 2, 1, 3], dtype=np.int64),
            "source_firm_key": [
                f"2024-25:experimental_uk_firm:{i}" for i in (4, 2, 1, 3)
            ],
            "data_vintage": "2024-25",
            "sic_code": ["00011", "00047", "00004", "00086"],
            "annual_turnover_k": np.array([120.0, 90.0, 0.0, 40.0]),
            "annual_input_k": np.array([80.0, 30.0, 0.0, 10.0]),
            "employment": np.array([12, 3, 0, 1]),
            "firm_weight": [2.5, 10.0, 0.0, 4.0],
            "vat_scope": [True, True, False, False],
            "vat_registered": [True, True, False, True],
            "in_frame": [True, True, False, True],
            "base_weight": [1.0, 2.0, 1.0, 1.0],
        }
    )


def test_snapshot_records_the_policyengine_uk_checkout() -> None:
    snapshot = _snapshot()
    source = snapshot["policyengine_uk"]

    assert source["pull_request"] == "PolicyEngine/policyengine-uk#2940"
    assert len(source["head_commit"]) == 40
    # A dirty checkout's HEAD does not identify the source it was read from.
    assert source["dirty"] is False
    assert snapshot["firm_input_names"] == sorted(
        name
        for name, variable in snapshot["firm_variables"].items()
        if not variable["has_formula"]
    )
    assert set(snapshot["firm_variables"]) <= set(
        snapshot["policyengine_uk_variable_names"]
    )


def test_input_status_covers_exactly_the_snapshot_inputs() -> None:
    assert UK_FIRM_INPUT_STATUSES == ("set", "derived", "assumed", "not_set")
    assert (
        input_status_problems(
            UK_FIRM_POLICYENGINE_UK_INPUT_STATUS,
            _snapshot()["firm_input_names"],
            exported_columns=UK_FIRM_POLICYENGINE_UK_COLUMNS,
        )
        == []
    )


def test_sic_sections_are_the_sic_section_members() -> None:
    assert _variable("firm_sic_section")["enum"]["name"] == "SICSection"
    assert UK_FIRM_SIC_SECTIONS == tuple(_variable("firm_sic_section")["enum_members"])


def test_microcosm_only_columns_collide_with_no_policyengine_uk_variable() -> None:
    snapshot = _snapshot()
    microcosm_only = set(UK_FIRM_MICROCOSM_ONLY_COLUMNS)

    assert snapshot["collisions"] == []
    assert microcosm_only <= set(snapshot["microcosm_only_columns_checked"])
    assert microcosm_only.isdisjoint(snapshot["policyengine_uk_variable_names"])
    assert not any(column.startswith("firm_") for column in microcosm_only)


def test_export_loads_as_policyengine_uk_firm_inputs() -> None:
    table = to_policyengine_uk_firm_table(_firms(), threshold_k=90.0)
    assert tuple(table.columns) == UK_FIRM_TABLE_COLUMNS
    assert (
        export_schema_problems(
            table, _variables(), columns=UK_FIRM_POLICYENGINE_UK_COLUMNS
        )
        == []
    )
