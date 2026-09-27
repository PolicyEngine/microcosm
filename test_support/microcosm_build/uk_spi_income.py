# ruff: noqa: F401
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import frs_disability, spi_income
from microcosm.build.uk_runtime.frs_hmrc_leaves import (
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_RETAINED_LEAF_COLUMNS,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
)
from microcosm.build.uk_runtime.hmrc_income import (
    HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
)
from microcosm.build.uk_runtime.spi_income import (
    SPI_DONOR_FILENAME,
    SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
    impute_uk_spi_income_support,
)
from microcosm.build.uk_runtime.spi_support import (
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_HMRC_EMPLOYED_INCOME_COLUMN,
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_PAY_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
    SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN,
    SPI_INCOME_IMPUTATION_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    create_uk_spi_support_tables,
    replace_uk_spi_support_tables,
    support_channel_column,
)
from microcosm.frame import WeightKind
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

_REPO_ROOT = _TEST_PATHS.repository
_PINNED_SPI_DONOR_PATH = _REPO_ROOT / "inputs" / "spi" / "put2223uk.tab"


class _FakeFittedQRF:
    def __init__(self, targets: tuple[str, ...], weight_kind: str) -> None:
        self.targets = targets
        self.weight_kind = weight_kind

    def predict(self, predictors: pd.DataFrame) -> pd.DataFrame:
        values: dict[str, np.ndarray] = {}
        for position, target in enumerate(self.targets, start=1):
            value = float(position)
            if target == "savings_interest_income":
                value = 100.0
            elif target == "other_investment_income":
                value = 25.0
            elif target == "gift_aid":
                value = 10.0
            elif target == "charitable_investment_gifts":
                value = 2.0
            elif target == "tax_free_savings_income":
                value = 5.0
            elif target == SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN:
                value = -3.0
            values[target] = np.full(len(predictors), value, dtype=float)
        return pd.DataFrame(values, index=predictors.index)


class _FakeQRF:
    fit_weight_kinds: list[str] = []
    fit_weight_values: list[np.ndarray] = []

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        assert n_estimators > 0
        assert seed >= 0

    def fit(
        self,
        frame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: str,
    ) -> _FakeFittedQRF:
        assert predictors
        resolved = frame.resolve_weights("person")
        assert resolved.kind.value == weights
        assert (resolved.values > 0).all()
        self.fit_weight_kinds.append(resolved.kind.value)
        self.fit_weight_values.append(resolved.values.copy())
        return _FakeFittedQRF(tuple(targets), resolved.kind.value)


def _dead_support(
    *,
    drop_stage2: str | None = None,
    drop_hmrc_leaf: str | None = None,
    drop_income_component: str | None = None,
):
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, 5, dtype="int64"),
            "household_weight": [10.0, 20.0, 30.0, 40.0],
            "region": ["LONDON", "WALES", "LONDON", "WALES"],
            "clone_index": [0, 0, 1, 1],
            "household_is_capital_gains_clone": [False, True, False, True],
        }
    )
    person_columns: dict[str, object] = {
        "person_id": np.arange(101, 105, dtype="int64"),
        "person_household_id": np.arange(1, 5, dtype="int64"),
        "person_benunit_id": np.arange(201, 205, dtype="int64"),
        "age": [30, 40, 50, 60],
        "gender": ["MALE", "FEMALE", "MALE", "FEMALE"],
    }
    for position, column in enumerate(SPI_INCOME_IMPUTATION_COLUMNS, start=1):
        if column == drop_income_component:
            continue
        person_columns[column] = np.arange(
            position,
            position + 4,
            dtype=float,
        )
    for position, column in enumerate(FRS_HMRC_RETAINED_LEAF_COLUMNS, start=1):
        if column == drop_hmrc_leaf:
            continue
        person_columns[column] = np.arange(
            position,
            position + 4,
            dtype=float,
        )
    for position, column in enumerate(FRS_ONLY_SPI_FILL_PERSON_COLUMNS, start=1):
        if column in SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS or column == drop_stage2:
            continue
        person_columns[column] = np.arange(
            position,
            position + 4,
            dtype=float,
        )
    person_columns["tax_free_savings_income"] = np.zeros(4, dtype=float)
    person = pd.DataFrame(person_columns)
    benunit = pd.DataFrame({"benunit_id": np.arange(201, 205, dtype="int64")})
    dead = create_uk_spi_support_tables(
        person=person,
        benunit=benunit,
        household=household,
        selected_household_ids=(1, 2, 3, 4),
        source_year=2023,
    )
    return replace_uk_spi_support_tables(
        person=dead.person,
        benunit=dead.benunit,
        household=dead.household,
        seed=7,
        source_year=2023,
    )


def _write_donor(path: Path, *, drop: str | None = None) -> None:
    donor = pd.DataFrame(
        {
            "SEX": [1, 2, 1, 2],
            "FACT": [1.0, 2.0, 3.0, 4.0],
            "GORCODE": [7, 10, 7, 10],
            "AGERANGE": [2, 3, 4, 5],
            "PAY": [20_000.0, 30_000.0, 40_000.0, 50_000.0],
            "EPB": [0.0, 100.0, 0.0, 100.0],
            "EXPS": [0.0, 50.0, 100.0, 150.0],
            "TAXTERM": [0.0, 0.0, 200.0, 200.0],
            "INCPBEN": [0.0, 0.0, 0.0, 0.0],
            "OSSBEN": [0.0, 0.0, 0.0, 0.0],
            "UBISJA": [0.0, 0.0, 0.0, 0.0],
            "MOTHINC": [-25.0, 0.0, 75.0, -50.0],
            "OTHERINC": [0.0, 0.0, 0.0, 0.0],
            "PROFITS": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
            "CAPALL": [0.0, 100.0, 200.0, 300.0],
            "LOSSBF": [0.0, 0.0, 100.0, 100.0],
            "SRP": [0.0, 0.0, 500.0, 1_000.0],
            "INCBBS": [100.0, 200.0, 300.0, 400.0],
            "DIVIDENDS": [10.0, 20.0, 30.0, 40.0],
            "PENSION": [0.0, 0.0, 500.0, 1_000.0],
            "INCPROP": [0.0, 250.0, 500.0, 750.0],
            "OTHERINV": [5.0, 10.0, 15.0, 20.0],
            "GIFTAID": [10.0, 20.0, 30.0, 40.0],
            "GIFTINV": [1.0, 2.0, 3.0, 4.0],
        }
    )
    employment = (
        (donor["PAY"] + donor["EPB"] - donor["EXPS"]).clip(lower=0.0)
        + donor["INCPBEN"]
        + donor["OSSBEN"]
        + donor["TAXTERM"]
        + donor["UBISJA"]
        + donor["MOTHINC"]
    )
    self_employment = (donor["PROFITS"] - donor["CAPALL"] - donor["LOSSBF"]).clip(
        lower=0.0
    )
    donor["TEI"] = (
        employment
        + donor["OTHERINC"]
        + donor["SRP"]
        + donor["PENSION"]
        + self_employment
    )
    donor["TII"] = (
        donor["OTHERINV"] + donor["DIVIDENDS"] + donor["INCPROP"] + donor["INCBBS"]
    )
    donor["TI"] = donor["TEI"] + donor["TII"]
    if drop is not None:
        donor = donor.drop(columns=[drop])
    donor.to_csv(path, sep="\t", index=False)


def _bypass_reviewed_donor_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the synthetic donor's non-production identity explicit in tests."""

    monkeypatch.setattr(spi_income, "_verify_spi_donor_identity", lambda _: None)


__all__ = [name for name in globals() if not name.startswith("__")]
