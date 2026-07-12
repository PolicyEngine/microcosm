from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime import spi_income
from populace.build.uk_runtime.spi_income import (
    SPI_DONOR_FILENAME,
    SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
    impute_uk_spi_income_support,
)
from populace.build.uk_runtime.spi_support import (
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_INCOME_IMPUTATION_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    create_uk_spi_support_tables,
    replace_uk_spi_support_tables,
    support_channel_column,
)
from populace.frame import WeightKind


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


def _dead_support(*, drop_stage2: str | None = None):
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
            "MOTHINC": [0.0, 0.0, 0.0, 0.0],
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
    donor["TI"] = (
        employment
        + donor["OTHERINC"]
        + donor["SRP"]
        + donor["PENSION"]
        + self_employment
        + donor["OTHERINV"]
        + donor["DIVIDENDS"]
        + donor["INCPROP"]
        + donor["INCBBS"]
    )
    if drop is not None:
        donor = donor.drop(columns=[drop])
    donor.to_csv(path, sep="\t", index=False)


def test_spi_qrf_stages_use_typed_weights_and_restore_gross_savings(
    monkeypatch,
    tmp_path,
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    _FakeQRF.fit_weight_kinds = []
    _FakeQRF.fit_weight_values = []
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)

    before = support.person.copy()
    result = impute_uk_spi_income_support(
        support,
        donor_path,
        seed=9,
        n_estimators=3,
        donor_sample_size=None,
    )

    assert _FakeQRF.fit_weight_kinds == ["design", "importance"]
    assert [record.weight_kind for record in result.fit_weight_records] == [
        "design",
        "importance",
    ]
    assert result.reviewed_absent_stage2_outputs == (SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS)
    assert result.donor_rows == 4
    assert len(result.donor_sha256) == 64
    assert SPI_INCOME_QRF_OUTPUT_COLUMNS[-1] == "hmrc_spi_assessable_income"

    channel = support_channel_column("person")
    spi_people = result.person[channel] == "spi"
    base_people = ~spi_people
    assert result.person.loc[spi_people, "savings_interest_income"].eq(105.0).all()
    assert result.person.loc[spi_people, "other_investment_income"].eq(25.0).all()
    assert result.person.loc[spi_people, "gift_aid"].eq(10.0).all()
    assert result.person.loc[spi_people, "charitable_investment_gifts"].eq(2.0).all()
    assert result.person.loc[spi_people, "is_disabled_for_benefits"].all()
    assert {
        "aa_category",
        "dla_sc_category",
        "dla_m_category",
        "pip_m_category",
        "pip_dl_category",
    }.issubset(result.person.columns)
    pd.testing.assert_frame_equal(
        result.person.loc[base_people, before.columns],
        before.loc[base_people],
    )

    spi_households = support.household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN]
    assert support.household.loc[spi_households, "household_weight"].gt(0).all()
    assert support.household_weight_kind is WeightKind.IMPORTANCE


def test_spi_weighted_bootstrap_does_not_apply_fact_twice(
    monkeypatch,
    tmp_path,
) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    _FakeQRF.fit_weight_kinds = []
    _FakeQRF.fit_weight_values = []
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)

    result = impute_uk_spi_income_support(
        support,
        donor_path,
        donor_sample_size=8,
    )

    assert result.donor_rows == 8
    np.testing.assert_array_equal(_FakeQRF.fit_weight_values[0], np.ones(8))
    assert _FakeQRF.fit_weight_kinds[0] == "design"


def test_spi_qrf_fails_closed_on_missing_donor_component(monkeypatch, tmp_path) -> None:
    support = _dead_support()
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path, drop="OTHERINV")
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)

    with pytest.raises(ValueError, match="OTHERINV"):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
        )


def test_spi_donor_preserves_documented_unattributed_sex_code(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    raw.loc[0, "SEX"] = 0

    donor = spi_income._prepare_spi_donor(raw, seed=7)

    assert donor.loc[0, "gender"] == "UNKNOWN"
    assert set(donor["gender"]) == {"UNKNOWN", "MALE", "FEMALE"}


def test_spi_donor_rejects_undocumented_sex_code(tmp_path) -> None:
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    raw = pd.read_csv(donor_path, delimiter="\t")
    raw.loc[0, "SEX"] = 3

    with pytest.raises(ValueError, match="documented codes 0/1/2"):
        spi_income._prepare_spi_donor(raw, seed=7)


def test_spi_qrf_fails_closed_on_unreviewed_stage2_gap(monkeypatch, tmp_path) -> None:
    support = _dead_support(drop_stage2="universal_credit_reported")
    donor_path = tmp_path / SPI_DONOR_FILENAME
    _write_donor(donor_path)
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)

    with pytest.raises(ValueError, match="universal_credit_reported"):
        impute_uk_spi_income_support(
            support,
            donor_path,
            donor_sample_size=None,
        )


def test_spi_qrf_requires_current_donor_filename(tmp_path) -> None:
    support = _dead_support()
    donor_path = tmp_path / "put2021uk.tab"
    _write_donor(donor_path)

    with pytest.raises(ValueError, match=SPI_DONOR_FILENAME):
        impute_uk_spi_income_support(support, donor_path)
