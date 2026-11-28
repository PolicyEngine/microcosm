"""Reading a country package's entity inputs from source, and checks against them.

The package here is a toy written to ``tmp_path``; no country model is
imported.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.firms import (
    EXPORTED_INPUT_STATUSES,
    INPUT_STATUSES,
    export_schema_problems,
    input_status_problems,
    read_entity_variables,
)

_FILES = {
    "__init__.py": "",
    "enums.py": """
        from enum import Enum

        class Region(Enum):
            NORTH = "North"
            SOUTH = "South"
            _PRIVATE = "hidden"
        """,
    "variables/__init__.py": "",
    "variables/firm/__init__.py": "",
    "variables/firm/inputs.py": """
        from toymodel.model_api import *
        from toymodel.enums import Region
        from toymodel import enums as e

        class firm_turnover(Variable):
            entity = Firm
            value_type = float
            unit = GBP
            definition_period = YEAR
            default_value = 0

        class firm_region(Variable):
            entity = Firm
            value_type = Enum
            possible_values = Region
            default_value = Region.SOUTH
            definition_period = YEAR

        class firm_home_region(Variable):
            entity = Firm
            value_type = Enum
            possible_values = e.Region
            definition_period = YEAR

        class firm_is_bank(Variable):
            entity = Firm
            value_type = bool
            definition_period = YEAR
        """,
    "variables/firm/outputs.py": """
        from toymodel.model_api import *

        class firm_tax(Variable):
            entity = Firm
            value_type = float
            definition_period = YEAR

            def formula(firm, period, parameters):
                return firm("firm_turnover", period) * 0.1

        class firm_total(Variable):
            entity = Firm
            value_type = float
            definition_period = YEAR
            adds = ["firm_tax"]

        class person_age(Variable):
            entity = Person
            value_type = int
            definition_period = YEAR
        """,
    "tests/test_ignored.py": """
        class firm_test_only(Variable):
            entity = Firm
            value_type = float
        """,
}


@pytest.fixture
def toy_package(tmp_path: Path) -> Path:
    package = tmp_path / "toymodel"
    for name, source in _FILES.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return package


def test_reader_finds_entity_variables_and_their_inputs(toy_package) -> None:
    read = read_entity_variables(toy_package, entity="Firm")
    assert sorted(read.variables) == [
        "firm_home_region",
        "firm_is_bank",
        "firm_region",
        "firm_tax",
        "firm_total",
        "firm_turnover",
    ]
    assert read.input_names == [
        "firm_home_region",
        "firm_is_bank",
        "firm_region",
        "firm_turnover",
    ]
    assert "person_age" in read.variable_names
    assert "firm_test_only" not in read.variable_names


def test_reader_records_types_units_defaults_and_enums(toy_package) -> None:
    variables = read_entity_variables(toy_package, entity="Firm").variables
    assert variables["firm_turnover"] == {
        "default_value": 0,
        "definition_period": "year",
        "enum": None,
        "enum_members": None,
        "file": "variables/firm/inputs.py",
        "has_formula": False,
        "name": "firm_turnover",
        "unit": "GBP",
        "value_type": "float",
    }
    region = variables["firm_region"]
    assert region["enum"] == {"name": "Region", "module": "toymodel.enums"}
    assert region["enum_members"] == ["NORTH", "SOUTH"]
    assert region["default_value"] == "SOUTH"
    assert variables["firm_home_region"]["enum_members"] == ["NORTH", "SOUTH"]
    assert variables["firm_tax"]["has_formula"]
    assert variables["firm_total"]["has_formula"]


def test_reader_refuses_an_entity_with_no_variables(toy_package) -> None:
    with pytest.raises(ValueError, match="defines no Household variables"):
        read_entity_variables(toy_package, entity="Household")


def test_input_status_must_cover_exactly_the_inputs() -> None:
    inputs = ["firm_turnover", "firm_region", "firm_is_bank"]
    status = {
        "firm_turnover": "set",
        "firm_region": "assumed",
        "firm_is_bank": "not_set",
    }
    assert input_status_problems(status, inputs) == []
    assert (
        input_status_problems(
            status, inputs, exported_columns=["firm_turnover", "firm_region"]
        )
        == []
    )
    problems = input_status_problems(
        {**status, "firm_is_bank": "guessed", "firm_other": "set"},
        [*inputs, "firm_new"],
        exported_columns=["firm_turnover"],
    )
    assert problems == [
        f"unknown input statuses ['guessed']; choose from {INPUT_STATUSES}",
        "engine inputs without a status: ['firm_new']",
        "statuses for names that are not engine inputs: ['firm_other']",
        "inputs marked exported that no export writes: ['firm_other', 'firm_region']",
    ]
    assert set(EXPORTED_INPUT_STATUSES) == set(INPUT_STATUSES) - {"not_set"}


def test_an_exported_column_marked_not_set_is_a_problem() -> None:
    problems = input_status_problems(
        {"firm_turnover": "not_set"},
        ["firm_turnover"],
        exported_columns=["firm_turnover"],
    )
    assert problems == ["exported inputs marked not_set or missing: ['firm_turnover']"]


def test_export_schema_accepts_a_loadable_table(toy_package) -> None:
    read = read_entity_variables(toy_package, entity="Firm")
    table = pd.DataFrame(
        {
            "firm_turnover": np.array([1.0, 2.0]),
            "firm_region": ["NORTH", "SOUTH"],
            "firm_is_bank": [False, True],
            "in_frame": [True, True],
        }
    )
    columns = [c for c in table.columns if c.startswith("firm_")]
    assert export_schema_problems(table, read, columns=columns) == []


def test_export_schema_names_each_unloadable_column(toy_package) -> None:
    read = read_entity_variables(toy_package, entity="Firm")
    table = pd.DataFrame(
        {
            "firm_turnover": np.array([1, 2]),
            "firm_region": ["NORTH", "EAST"],
            "firm_home_region": [1, 2],
            "firm_tax": [0.0, 0.0],
            "in_frame": [True, True],
        }
    )
    problems = export_schema_problems(
        table, read, columns=[*table.columns, "firm_gone"]
    )
    assert problems == [
        "firm_turnover: dtype int64 for value type float",
        "firm_region: values ['EAST'] are not enum members",
        "firm_home_region: enum values must be member names",
        "firm_tax: not a formula-less engine input",
        "in_frame: not a formula-less engine input",
        "firm_gone: not in the table",
    ]
