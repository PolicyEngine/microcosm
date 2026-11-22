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
import pytest

from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS,
    UK_FIRM_EXTENDED_TABLE_COLUMNS,
    UK_FIRM_INPUT_STATUSES,
    UK_FIRM_LEGAL_FORMS,
    UK_FIRM_MICROCOSM_COLUMNS,
    UK_FIRM_MICROCOSM_ONLY_COLUMNS,
    UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM,
    UK_FIRM_PAYBILL_EXTENDED_POLICYENGINE_UK_COLUMNS,
    UK_FIRM_PAYBILL_TABLE_COLUMNS,
    UK_FIRM_POLICYENGINE_UK_COLUMNS,
    UK_FIRM_POLICYENGINE_UK_INPUT_STATUS,
    UK_FIRM_REGION_CODE_TO_REGION,
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


def _variable(name: str) -> dict:
    return _snapshot()["firm_variables"][name]


def _inputs() -> set[str]:
    return set(_snapshot()["firm_input_names"])


def _firms(*, attributes: bool, paybill: bool = False) -> pd.DataFrame:
    firms = pd.DataFrame(
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
    if attributes:
        firms["ons_legal_status"] = np.array([1, 2, 5, 7], dtype=np.int64)
        firms["ons_region_code"] = ["E12000007", "W92000004", "S92000003", "N92000002"]
    if paybill:
        # Headcount x a pay per head; the source is test_uk_firm_paybill's.
        firms["firm_paybill"] = firms["employment"].to_numpy(np.float64) * 30000.0
    return firms


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


@pytest.mark.parametrize(
    "columns",
    [
        UK_FIRM_TABLE_COLUMNS,
        UK_FIRM_EXTENDED_TABLE_COLUMNS,
        UK_FIRM_PAYBILL_TABLE_COLUMNS,
    ],
)
def test_exported_firm_columns_are_policyengine_uk_firm_inputs(columns) -> None:
    firm_columns = [column for column in columns if column.startswith("firm_")]

    assert set(firm_columns) <= _inputs(), sorted(set(firm_columns) - _inputs())


def test_input_status_covers_exactly_the_snapshot_inputs() -> None:
    assert set(UK_FIRM_POLICYENGINE_UK_INPUT_STATUS) == _inputs()
    assert set(UK_FIRM_POLICYENGINE_UK_INPUT_STATUS.values()) <= set(
        UK_FIRM_INPUT_STATUSES
    )
    exported = {
        name
        for name, status in UK_FIRM_POLICYENGINE_UK_INPUT_STATUS.items()
        if status in ("set", "derived", "assumed")
    }
    assert exported == set(UK_FIRM_PAYBILL_EXTENDED_POLICYENGINE_UK_COLUMNS)
    assert set(UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS) <= exported
    assert set(UK_FIRM_POLICYENGINE_UK_COLUMNS) <= exported


def test_sic_sections_are_the_sic_section_members() -> None:
    assert _variable("firm_sic_section")["enum"]["name"] == "SICSection"
    assert UK_FIRM_SIC_SECTIONS == tuple(_variable("firm_sic_section")["enum_members"])


def test_legal_forms_and_regions_are_enum_members() -> None:
    legal_form = _variable("firm_legal_form")
    region = _variable("firm_region")

    assert legal_form["enum"]["name"] == "FirmLegalForm"
    assert UK_FIRM_LEGAL_FORMS == tuple(legal_form["enum_members"])
    assert set(UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM.values()) <= set(
        legal_form["enum_members"]
    )
    assert region["enum"] == {
        "module": "policyengine_uk.variables.household.demographic.geography",
        "name": "Region",
    }
    assert set(UK_FIRM_REGION_CODE_TO_REGION.values()) <= set(region["enum_members"])


def test_microcosm_only_columns_collide_with_no_policyengine_uk_variable() -> None:
    snapshot = _snapshot()
    microcosm_only = {*UK_FIRM_MICROCOSM_COLUMNS, *UK_FIRM_MICROCOSM_ONLY_COLUMNS}

    assert snapshot["collisions"] == []
    assert microcosm_only <= set(snapshot["microcosm_only_columns_checked"])
    assert microcosm_only.isdisjoint(snapshot["policyengine_uk_variable_names"])
    assert not any(column.startswith("firm_") for column in microcosm_only)


_DTYPE_KINDS = {"float": "f", "int": "i", "bool": "b"}


@pytest.mark.parametrize(
    ("attributes", "paybill", "expected"),
    [
        (False, False, UK_FIRM_TABLE_COLUMNS),
        (True, False, UK_FIRM_EXTENDED_TABLE_COLUMNS),
        (True, True, UK_FIRM_PAYBILL_TABLE_COLUMNS),
    ],
)
def test_export_dtypes_match_policyengine_uk_value_types(
    attributes, paybill, expected
) -> None:
    table = to_policyengine_uk_firm_table(
        _firms(attributes=attributes, paybill=paybill), threshold_k=90.0
    )
    assert tuple(table.columns) == expected

    for column in (c for c in table.columns if c.startswith("firm_")):
        variable = _variable(column)
        value_type = variable["value_type"]
        if value_type == "Enum":
            values = table[column].to_numpy()
            assert all(isinstance(value, str) for value in values), column
            assert set(values) <= set(variable["enum_members"]), column
        else:
            assert table[column].dtype.kind == _DTYPE_KINDS[value_type], (
                column,
                table[column].dtype,
                value_type,
            )
