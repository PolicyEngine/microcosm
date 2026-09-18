"""PolicyEngine-US adapter support for static aging: the series each
variable uprates by, and the multi-year export the engine reads as already
extended."""

from __future__ import annotations

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
            "employment_income_before_lsr": [50_000.0, 0.0, 0.0],
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
                {"employment_income_before_lsr": 1.08},
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
    assert year.person["age"].tolist() == [40, 8, 70]


def test_multi_year_dataset_rejects_bad_inputs() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="household weights"):
        multi_year_dataset(bundle, BASE_YEAR, {2025: (np.array([1.0]), {})})
    with pytest.raises(ValueError, match="not in the bundle"):
        multi_year_dataset(
            bundle, BASE_YEAR, {2025: (np.array([1.0, 2.0]), {"missing": 1.1})}
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
