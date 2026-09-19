"""PolicyEngine-US adapter support for static aging: the series each
variable uprates by, and the multi-year export the engine reads as already
extended."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("policyengine_us")

from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights  # noqa: E402
from microcosm.frame.adapters.policyengine_us import (  # noqa: E402
    multi_year_dataset,
    uprating_series,
)

pytestmark = pytest.mark.requires_us

BASE_YEAR = 2024


@pytest.fixture(scope="module")
def system():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def test_uprating_series_classifies_totals_and_indices(system) -> None:
    totals, indices, column_series = uprating_series(
        [
            "employment_income_before_lsr",
            "other_medical_expenses",
            "child_support_expense",
            "household_weight",
            "age",
            "no_such_variable",
        ],
        (BASE_YEAR, 2025),
        system=system,
    )
    # A national total, whether the variable names it directly or through the
    # per-capita series newer engines derive.
    assert column_series["employment_income_before_lsr"].startswith(
        "calibration.gov.irs.soi.employment_income"
    )
    assert column_series["employment_income_before_lsr"] in totals
    # A per-person rate and a price index come back as indices.
    assert (
        column_series["other_medical_expenses"]
        == "calibration.gov.hhs.cms.moop_per_capita"
    )
    assert column_series["child_support_expense"] == "gov.bls.cpi.cpi_u"
    assert set(column_series.values()) - set(totals) == set(indices)
    # The weights' population series and unuprated inputs are skipped.
    assert "household_weight" not in column_series
    assert "age" not in column_series
    assert "no_such_variable" not in column_series
    for table in (totals, indices):
        for series in table.values():
            assert set(series) == {BASE_YEAR, 2025}
            assert all(np.isfinite(v) and v > 0 for v in series.values())


def test_uprating_series_includes_dataset_only_overrides(system) -> None:
    from policyengine_us.data.economic_assumptions import MICRODATA_UPRATING_OVERRIDES

    columns = ["taxable_pension_income", "long_term_capital_gains", "rent"]
    totals, indices, column_series = uprating_series(
        columns, (BASE_YEAR, 2025), system=system
    )
    for column in columns:
        declared = MICRODATA_UPRATING_OVERRIDES[column]
        # Newer engines point overrides at derived per-capita series; the
        # adapter must recover their source totals for demographic adjustment.
        assert column_series[column] == declared.removesuffix("_per_capita")
    assert column_series["taxable_pension_income"] in totals
    assert column_series["long_term_capital_gains"] in totals
    assert column_series["rent"] in indices


def test_uprating_series_override_precedence_and_derived_totals(monkeypatch) -> None:
    from policyengine_core.parameters import Parameter, ParameterNode
    from policyengine_us.data import economic_assumptions

    parameters = ParameterNode("root", data={})

    def add(path, values, metadata=None):
        node = parameters
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in node.children:
                node.add_child(part, ParameterNode(part, data={}))
            node = node.children[part]
        node.add_child(
            parts[-1],
            Parameter(
                path,
                data={
                    "values": {
                        f"{year}-01-01": {"value": value}
                        for year, value in values.items()
                    },
                    "metadata": metadata or {},
                },
            ),
        )

    total = "calibration.gov.irs.soi.taxable_pension_income"
    per_capita = f"{total}_per_capita"
    cms_rate = "calibration.gov.hhs.cms.moop_per_capita"
    other_calibration = "calibration.gov.hhs.cms.other_rate"
    cpi = "gov.bls.cpi.cpi_u"
    derived_index = "gov.bls.cpi.derived_index"
    add(total, {BASE_YEAR: 100.0, 2025: 120.0})
    add(per_capita, {BASE_YEAR: 10.0, 2025: 11.0}, {"derived_from": total})
    add(cms_rate, {BASE_YEAR: 5.0, 2025: 6.0})
    add(other_calibration, {BASE_YEAR: 2.0, 2025: 2.1})
    add(cpi, {BASE_YEAR: 10.0, 2025: 10.3})
    add(derived_index, {BASE_YEAR: 1.0, 2025: 1.03}, {"derived_from": cpi})
    monkeypatch.setitem(
        economic_assumptions.MICRODATA_UPRATING_OVERRIDES,
        "taxable_pension_income",
        per_capita,
    )
    mock_system = SimpleNamespace(
        parameters=parameters,
        variables={
            "taxable_pension_income": SimpleNamespace(uprating=cpi),
            "other_medical_expenses": SimpleNamespace(uprating=cms_rate),
            "other_rate": SimpleNamespace(uprating=other_calibration),
            "derived_index": SimpleNamespace(uprating=derived_index),
        },
    )
    totals, indices, column_series = uprating_series(
        mock_system.variables, (BASE_YEAR, 2025), system=mock_system
    )
    assert totals == {total: {BASE_YEAR: 100.0, 2025: 120.0}}
    assert column_series["taxable_pension_income"] == total
    assert indices == {
        cms_rate: {BASE_YEAR: 5.0, 2025: 6.0},
        other_calibration: {BASE_YEAR: 2.0, 2025: 2.1},
        derived_index: {BASE_YEAR: 1.0, 2025: 1.03},
    }


def _bundle() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [1, 1, 2],
            "person_tax_unit_id": [1, 1, 2],
            "person_spm_unit_id": [1, 1, 2],
            "person_family_id": [1, 1, 2],
            "person_marital_unit_id": [1, 1, 2],
            "age": [40, 8, 70],
            "is_male": [True, False, True],
            "employment_income_before_lsr": [50_000.0, 0.0, 0.0],
            "taxable_pension_income": [0.0, 0.0, 10_000.0],
            "rent": [12_000.0, 0.0, 0.0],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2], "state_fips": [6, 36]})
    tables = {
        "person": person,
        "household": household,
        "tax_unit": pd.DataFrame({"tax_unit_id": [1, 2]}),
        "spm_unit": pd.DataFrame({"spm_unit_id": [1, 2]}),
        "family": pd.DataFrame({"family_id": [1, 2]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [1, 2]}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.array([1_000.0, 2_000.0]), kind=WeightKind.CALIBRATED
            )
        },
    )


def test_multi_year_dataset_carries_weights_and_factored_columns() -> None:
    bundle = _bundle()
    dataset = multi_year_dataset(
        bundle,
        BASE_YEAR,
        {
            2025: (
                np.array([1_010.0, 2_050.0]),
                {"employment_income_before_lsr": 1.04},
            ),
            2026: (
                np.array([1_020.0, 2_100.0]),
                {
                    "employment_income_before_lsr": 1.08,
                    "taxable_pension_income": 1.06,
                    "rent": 1.03,
                },
            ),
        },
    )
    assert sorted(dataset.datasets) == [BASE_YEAR, 2025, 2026]
    base = dataset.datasets[BASE_YEAR]
    assert base.household["household_weight"].tolist() == [1_000.0, 2_000.0]
    assert base.person["employment_income_before_lsr"].tolist() == [50_000.0, 0.0, 0.0]
    year = dataset.datasets[2026]
    assert year.household["household_weight"].tolist() == [1_020.0, 2_100.0]
    assert year.person["employment_income_before_lsr"].tolist() == [54_000.0, 0.0, 0.0]
    assert year.person["taxable_pension_income"].tolist() == [0.0, 0.0, 10_600.0]
    assert year.person["rent"].tolist() == [12_360.0, 0.0, 0.0]
    assert year.person["age"].tolist() == [40, 8, 70]
    assert year.person["is_male"].tolist() == [True, False, True]
    assert bundle.person["rent"].tolist() == [12_000.0, 0.0, 0.0]


def test_multi_year_dataset_rejects_bad_inputs() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="household weights"):
        multi_year_dataset(bundle, BASE_YEAR, {2025: (np.array([1.0]), {})})
    with pytest.raises(ValueError, match="not in the bundle"):
        multi_year_dataset(
            bundle, BASE_YEAR, {2025: (np.array([1.0, 2.0]), {"missing": 1.1})}
        )


@pytest.mark.parametrize("year", [BASE_YEAR, BASE_YEAR - 1, 2024.5, "2024"])
def test_multi_year_dataset_rejects_non_projection_years(year) -> None:
    with pytest.raises(ValueError, match="Projection years"):
        multi_year_dataset(_bundle(), BASE_YEAR, {year: (np.array([1.0, 2.0]), {})})


@pytest.mark.parametrize(
    "weights",
    [[np.nan, 1.0], [np.inf, 1.0], [-1.0, 1.0], [0.0, 0.0], [[1.0], [2.0]]],
)
def test_multi_year_dataset_rejects_invalid_weights(weights) -> None:
    with pytest.raises(ValueError, match="household weights"):
        multi_year_dataset(_bundle(), BASE_YEAR, {2025: (np.array(weights), {})})


@pytest.mark.parametrize(
    "column",
    [
        "person_id",
        "person_household_id",
        "household_id",
        "household_weight",
        "age",
        "is_male",
        "state_fips",
    ],
)
def test_multi_year_dataset_rejects_non_upratable_factors(column) -> None:
    with pytest.raises(ValueError, match="cannot be factored"):
        multi_year_dataset(
            _bundle(), BASE_YEAR, {2025: (np.array([1.0, 2.0]), {column: 1.1})}
        )


@pytest.mark.parametrize("factor", [np.nan, np.inf, -np.inf, 1e308])
def test_multi_year_dataset_rejects_nonfinite_factored_values(factor) -> None:
    with pytest.raises(ValueError, match="finite"):
        multi_year_dataset(
            _bundle(),
            BASE_YEAR,
            {2025: (np.array([1.0, 2.0]), {"employment_income_before_lsr": factor})},
        )


def test_engine_reads_the_multi_year_dataset_as_already_extended() -> None:
    from policyengine_us import Microsimulation

    bundle = _bundle()
    dataset = multi_year_dataset(
        bundle,
        BASE_YEAR,
        {2025: (np.array([1_010.0, 2_050.0]), {"employment_income_before_lsr": 1.04})},
    )
    sim = Microsimulation(dataset=dataset)
    income_2025 = np.asarray(sim.calculate("employment_income_before_lsr", 2025))
    weights_2025 = np.asarray(sim.calculate("household_weight", 2025))
    assert income_2025.tolist() == pytest.approx([52_000.0, 0.0, 0.0])
    assert weights_2025.tolist() == pytest.approx([1_010.0, 2_050.0])
