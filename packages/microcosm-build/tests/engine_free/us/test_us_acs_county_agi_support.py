"""The pre-R2 check for counties whose SOI AGI row would be empty (microcosm#922).

``tools/check_us_acs_county_agi_support.py`` lists counties that have
households but none that can carry a nonzero AGI: R2's ``attach_surface``
refuses such a county after the engine pass. These tests use small fixture
H5 files and a stand-in engine index, so no country engine is imported.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.h5_io import write_nullable_us_h5
from microcosm.frame import US_SCHEMA, Frame, VariableMetadata, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


@pytest.fixture(scope="module")
def check():
    path = _TEST_PATHS.repository / "tools" / "check_us_acs_county_agi_support.py"
    spec = importlib.util.spec_from_file_location("check_us_county_agi", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Closure:
    input_leaves: tuple[str, ...]
    engine_version: str = "fixture"
    sha256: str = "c" * 64


class _Index:
    """Stand-in for PolicyEngineUSVariableMetadataIndex."""

    leaves = {
        "employment_income_before_lsr": ("person", "float", "currency-USD"),
        "taxable_interest_income": ("person", "float", "currency-USD"),
        "social_security_retirement": ("person", "float", "currency-USD"),
        "student_loan_interest": ("person", "float", "currency-USD"),
        "other_net_gain": ("tax_unit", "float", "currency-USD"),
        "age": ("person", "float", None),
        "is_blind": ("person", "bool", None),
    }

    def variable_dependency_closure(self, name):
        assert name == "irs_gross_income"
        return _Closure(tuple(sorted(self.leaves)))

    def variable_metadata(self, name):
        entity, dtype, _unit = self.leaves[name]
        return VariableMetadata(name=name, entity=entity, dtype=dtype, period="year")

    def variable_unit(self, name):
        return self.leaves[name][2]


def _frame(person_income: dict[str, list[float]], unit_income=None) -> Frame:
    """Six households in three counties of state 01, one person each.

    01001: two households with income. 01003: two households, the first with
    Social Security only, the second with a null wage. 01005: two ACS
    households, one a group-quarters record.
    """

    n = 6
    ids = np.arange(1, n + 1)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{f"person_{entity}_id": ids for entity in US_SCHEMA.group_entities},
            "age": np.full(n, 40.0),
        }
    )
    for column, values in person_income.items():
        person[column] = values
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    household = tables["household"]
    household["state_fips"] = np.full(n, 1)
    household["county_fips"] = [1001, 1001, 1003, 1003, 1005, 1005]
    household["county_fips_2023"] = household["county_fips"]
    household["household_spine"] = pd.Series(
        ["asec_puf"] * 4 + ["acs_2024_1yr"] * 2, dtype=object
    )
    household["TYPEHUGQ"] = [np.nan] * 4 + [1.0, 3.0]
    for column, values in (unit_income or {}).items():
        tables["tax_unit"][column] = values
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.CALIBRATED)},
    )


_INCOME = {
    "employment_income_before_lsr": [100.0, 0.0, 0.0, np.nan, 0.0, 0.0],
    "taxable_interest_income": [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    "social_security_retirement": [0.0, 0.0, 900.0, 0.0, 0.0, 0.0],
    "student_loan_interest": [0.0, 0.0, 0.0, 0.0, 300.0, 0.0],
}


def test_agi_income_columns_drop_social_security_and_deductions(check) -> None:
    sources = check.agi_income_columns(_Index())

    assert sources["person"] == [
        "employment_income_before_lsr",
        "taxable_interest_income",
    ]
    assert sources["tax_unit"] == ["other_net_gain"]
    assert sources["excluded"] == {
        "social_security_retirement": "social_security_taxable_share_only",
        "student_loan_interest": "deduction_expense_or_contribution",
    }
    assert sources["engine_version"] == "fixture"


def test_households_without_agi_income_are_found(check) -> None:
    frame = _frame(_INCOME)

    flags = check.household_income_flags(
        frame.table("household"),
        frame.table("person"),
        frame.table("tax_unit"),
        person_columns=["employment_income_before_lsr", "taxable_interest_income"],
    )

    # Social Security only, a null wage, a deduction only: no AGI income.
    assert flags.tolist() == [True, True, False, False, False, False]


def test_tax_unit_income_reaches_its_household(check) -> None:
    frame = _frame(_INCOME, unit_income={"other_net_gain": [0, 0, 0, 0, 0, 7.0]})

    flags = check.household_income_flags(
        frame.table("household"),
        frame.table("person"),
        frame.table("tax_unit"),
        person_columns=["employment_income_before_lsr"],
        tax_unit_columns=["other_net_gain"],
    )

    assert flags.tolist() == [True, False, False, False, False, True]


def test_report_lists_counties_with_households_but_no_agi_income(check) -> None:
    frame = _frame(_INCOME)
    households = frame.table("household")
    income = np.asarray([True, True, False, False, False, True])

    report = check.county_agi_report(
        households,
        income,
        county_column="county_fips_2023",
        county_populations={1001: 10.0, 1003: 5.0, 1005: 5.0, 1007: 3.0},
    )

    assert report["at_risk"] == ["01003"]
    assert report["at_risk_detail"]["01003"] == {
        "households": 2,
        "supporting_households": 2,
        "households_with_agi_income": 0,
    }
    # 1007 has ladder population but no household: minimum support's case.
    assert report["no_household"] == ["01007"]
    # A group-quarters record counts for AGI but not for support.
    assert report["by_county"]["01005"]["supporting_households"] == 1
    assert report["fewer_than_min_support_with_agi_income"] == ["01001", "01005"]
    assert report["counties_outside_ladder"] == []


@pytest.mark.parametrize("layout", ["fixed", "table"])
def test_read_columns_reads_only_the_named_columns(
    check, tmp_path: Path, layout: str
) -> None:
    frame = _frame(_INCOME)
    path = tmp_path / f"{layout}.h5"
    if layout == "fixed":
        write_nullable_us_h5(frame, path, period=2024, artifact_kind="fixture")
    else:
        with pd.HDFStore(path, mode="w") as store:
            store.put("person", frame.table("person"), format="table")

    read = check.read_columns(
        path,
        "person",
        ["person_household_id", "taxable_interest_income", "not_a_column"],
    )

    assert list(read.columns) == ["person_household_id", "taxable_interest_income"]
    np.testing.assert_array_equal(
        read["taxable_interest_income"].to_numpy(), _INCOME["taxable_interest_income"]
    )
    np.testing.assert_array_equal(
        read["person_household_id"].to_numpy(), np.arange(1, 7)
    )


def test_command_exits_2_and_reports_an_at_risk_county(
    check, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import microcosm.frame.adapters.policyengine_us as adapter

    monkeypatch.setattr(adapter, "PolicyEngineUSVariableMetadataIndex", _Index)
    staging = tmp_path / "staging.h5"
    write_nullable_us_h5(_frame(_INCOME), staging, period=2024, artifact_kind="fixture")
    out = tmp_path / "report.json"

    status = check.main(["--h5", str(staging), "--out", str(out)])

    report = json.loads(out.read_text())
    assert status == check.EXIT_AT_RISK
    assert report["at_risk"] == ["01003", "01005"]
    assert report["agi_income_columns"]["person_present"] == [
        "employment_income_before_lsr",
        "taxable_interest_income",
    ]
    assert report["agi_income_columns"]["tax_unit_present"] == []
    assert report["minimum_support"] is None

    healthy = dict(_INCOME, taxable_interest_income=[0.0, 5.0, 1.0, 0.0, 2.0, 0.0])
    write_nullable_us_h5(_frame(healthy), staging, period=2024, artifact_kind="fixture")
    assert check.main(["--h5", str(staging), "--county-column", "county_fips"]) == 0
