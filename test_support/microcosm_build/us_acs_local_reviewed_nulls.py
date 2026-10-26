"""Invented frames and engine stand-ins for the reviewed-null fill tests.

``fill_reviewed_nulls`` (``tools/build_us_acs_local_release.py``) takes its
engine adapter and tax-benefit system as keyword arguments, so these tests
run without policyengine-us. Every fixture here is invented.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.spm_role_source import NATIVE_SPM_ROLE
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

__all__ = [
    "RENT",
    "FakeEngine",
    "FakeSystem",
    "load_tool_module",
    "reviewed_null_frame",
    "write_summary",
]

#: An ordinary nullable engine input with a registered default (0.0).
RENT = "pre_subsidy_rent"


def load_tool_module():
    path = (
        paths_for("microcosm-build").repository
        / "tools"
        / "build_us_acs_local_release.py"
    )
    spec = importlib.util.spec_from_file_location("build_us_acs_local_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeEngine:
    """The two adapter methods the fill reads, and optionally its declaration."""

    def __init__(self, *, declares: frozenset[str] | None = None) -> None:
        self._declares = declares

    def variables(self) -> list[str]:
        return ["age", RENT, NATIVE_SPM_ROLE]

    def __getattr__(self, name: str):
        if name == "_dataset_source_inputs" and self._declares is not None:
            return lambda: self._declares
        raise AttributeError(name)


class FakeSystem:
    """``CountryTaxBenefitSystem.variables`` with the engine's own defaults."""

    variables = {
        "age": SimpleNamespace(default_value=40.0, value_type=float),
        RENT: SimpleNamespace(default_value=0.0, value_type=float),
        NATIVE_SPM_ROLE: SimpleNamespace(default_value=False, value_type=bool),
    }


def reviewed_null_frame(
    *, role: list[object], rent: list[float], spines: list[str]
) -> Frame:
    """A one-person-per-household US frame tagged with each row's spine."""

    n = len(role)
    ids = np.arange(1, n + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "age": np.full(n, 30.0),
            RENT: np.asarray(rent, dtype=np.float64),
            NATIVE_SPM_ROLE: pd.Series(role, dtype=object),
            spine_column("person"): spines,
        }
    )
    tables = {"person": person}
    for entity in ("household", "tax_unit", "spm_unit", "family", "marital_unit"):
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )


def write_summary(path: Path, register: dict[str, int]) -> Path:
    """A staging summary whose register names ``register`` person columns."""

    path.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {"entity": "person", "column": column, "missing_rows": rows}
                    for column, rows in register.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    return path
