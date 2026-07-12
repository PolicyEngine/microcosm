from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime import hmrc_income
from populace.build.uk_runtime.hmrc_income import (
    HMRC_SPI_BUILD_PERIOD,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_TARGET_RECORD_COUNT,
    materialize_hmrc_spi_income_band_targets,
)

TABLE_COMPONENT_POSITIONS = {
    "Table_3_6": {
        "self_employment_income": (1, 2),
        "employment_income": (4, 5),
        "state_pension": (7, 8),
        "private_pension_income": (10, 11),
    },
    "Table_3_7": {
        "property_income": (1, 2),
        "savings_interest_income": (4, 5),
        "dividend_income": (7, 8),
        "other_investment_income": (10, 11),
    },
}


def _source_table(sheet_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        np.full((19, 12), np.nan, dtype=object),
    )
    for band_index, lower_bound in enumerate(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS):
        row = 5 + band_index
        frame.iat[row, 0] = lower_bound
        for component_index, (count_position, amount_position) in enumerate(
            TABLE_COMPONENT_POSITIONS[sheet_name].values()
        ):
            frame.iat[row, count_position] = 10 + component_index + band_index
            frame.iat[row, amount_position] = 20 + component_index + band_index
    frame.iat[18, 0] = "All ranges"
    return frame


def _install_source_tables(monkeypatch, tables: dict[str, pd.DataFrame]) -> None:
    def fake_read_excel(
        path: Path,
        *,
        sheet_name: str,
        engine: str,
        header: None,
    ) -> pd.DataFrame:
        assert path.name == "spi.ods"
        assert engine == "odf"
        assert header is None
        return tables[sheet_name].copy()

    monkeypatch.setattr(hmrc_income.pd, "read_excel", fake_read_excel)


def _materialize(monkeypatch, tmp_path, tables=None, *, period="2023"):
    source_bytes = b"local HMRC SPI 2023-24 ODS bytes"
    ods_path = tmp_path / "spi.ods"
    ods_path.write_bytes(source_bytes)
    if tables is None:
        tables = {
            sheet_name: _source_table(sheet_name)
            for sheet_name in TABLE_COMPONENT_POSITIONS
        }
    _install_source_tables(monkeypatch, tables)
    result = materialize_hmrc_spi_income_band_targets(
        ods_path,
        build_period=period,
    )
    return result, source_bytes


def _target(result, *, component: str, measure: str, lower_bound: int):
    matches = [
        target
        for target in result.targets
        if target.component == component
        and target.measure == measure
        and target.total_income_lower_bound == lower_bound
    ]
    assert len(matches) == 1
    return matches[0]


def test_materializes_complete_surface_including_savings_and_other_income(
    monkeypatch, tmp_path
) -> None:
    result, _ = _materialize(monkeypatch, tmp_path)

    assert len(result.targets) == HMRC_SPI_TARGET_RECORD_COUNT == 208
    assert {target.component for target in result.targets} == set(
        HMRC_SPI_INCOME_COMPONENTS
    )
    assert {target.period for target in result.targets} == {HMRC_SPI_BUILD_PERIOD}

    savings_count = _target(
        result,
        component="savings_interest_income",
        measure="count",
        lower_bound=12_570,
    )
    other_amount = _target(
        result,
        component="other_investment_income",
        measure="amount",
        lower_bound=12_570,
    )
    assert savings_count.value == 11_000.0
    assert savings_count.unit == "people"
    assert other_amount.value == 23_000_000.0
    assert other_amount.unit == "GBP"
    assert other_amount.total_income_upper_bound == 15_000

    top_band = _target(
        result,
        component="other_investment_income",
        measure="count",
        lower_bound=1_000_000,
    )
    assert top_band.total_income_upper_bound is None


def test_source_provenance_hashes_supplied_ods_at_runtime(
    monkeypatch, tmp_path
) -> None:
    result, source_bytes = _materialize(monkeypatch, tmp_path)

    assert result.source.sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert result.source.local_path == (tmp_path / "spi.ods").resolve()
    assert result.source.source_vintage == "2023-24"
    assert result.source.source_tax_year == "2023-24"
    assert result.source.source_tax_year_start == 2023
    assert result.source.build_period == "2023"
    assert result.source.table_names == ("Table_3_6", "Table_3_7")


def test_rejects_period_other_than_tax_year_start_mapping(
    monkeypatch, tmp_path
) -> None:
    ods_path = tmp_path / "spi.ods"
    ods_path.write_bytes(b"source")
    called = False

    def should_not_read(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("period validation must precede ODS parsing")

    monkeypatch.setattr(hmrc_income.pd, "read_excel", should_not_read)

    with pytest.raises(ValueError, match="maps to build period '2023'"):
        materialize_hmrc_spi_income_band_targets(
            ods_path,
            build_period="2024",
        )
    assert called is False


def test_rejects_missing_published_band(monkeypatch, tmp_path) -> None:
    tables = {
        sheet_name: _source_table(sheet_name)
        for sheet_name in TABLE_COMPONENT_POSITIONS
    }
    tables["Table_3_7"] = tables["Table_3_7"].drop(index=9).reset_index(drop=True)

    with pytest.raises(ValueError, match=r"missing=\[40000\]"):
        _materialize(monkeypatch, tmp_path, tables)


def test_rejects_duplicate_published_band(monkeypatch, tmp_path) -> None:
    tables = {
        sheet_name: _source_table(sheet_name)
        for sheet_name in TABLE_COMPONENT_POSITIONS
    }
    tables["Table_3_6"].iat[6, 0] = 12_570

    with pytest.raises(ValueError, match="duplicate income lower band"):
        _materialize(monkeypatch, tmp_path, tables)


@pytest.mark.parametrize("sentinel_count", [0, 2])
def test_rejects_missing_or_duplicate_all_ranges_sentinel(
    monkeypatch,
    tmp_path,
    sentinel_count,
) -> None:
    tables = {
        sheet_name: _source_table(sheet_name)
        for sheet_name in TABLE_COMPONENT_POSITIONS
    }
    tables["Table_3_7"].iat[18, 0] = np.nan
    if sentinel_count == 2:
        extra = pd.DataFrame(np.full((2, 12), np.nan, dtype=object))
        extra.iat[0, 0] = "All ranges"
        extra.iat[1, 0] = "All ranges"
        tables["Table_3_7"] = pd.concat(
            [tables["Table_3_7"], extra],
            ignore_index=True,
        )

    with pytest.raises(ValueError, match="exactly one 'All ranges' sentinel"):
        _materialize(monkeypatch, tmp_path, tables)


def test_rejects_all_ranges_sentinel_before_last_band(monkeypatch, tmp_path) -> None:
    tables = {
        sheet_name: _source_table(sheet_name)
        for sheet_name in TABLE_COMPONENT_POSITIONS
    }
    last_band_row = 5 + len(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS) - 1
    tables["Table_3_7"].iat[last_band_row - 1, 0] = "All ranges"
    tables["Table_3_7"].iat[18, 0] = np.nan

    with pytest.raises(ValueError, match="sentinel must follow"):
        _materialize(monkeypatch, tmp_path, tables)


def test_rejects_missing_component_column(monkeypatch, tmp_path) -> None:
    tables = {
        sheet_name: _source_table(sheet_name)
        for sheet_name in TABLE_COMPONENT_POSITIONS
    }
    tables["Table_3_7"] = tables["Table_3_7"].iloc[:, :11]

    with pytest.raises(
        ValueError,
        match=r"other_investment_income amount \(position 11\)",
    ):
        _materialize(monkeypatch, tmp_path, tables)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
def test_rejects_nonpositive_or_nonfinite_component_values(
    monkeypatch, tmp_path, bad_value
) -> None:
    tables = {
        sheet_name: _source_table(sheet_name)
        for sheet_name in TABLE_COMPONENT_POSITIONS
    }
    tables["Table_3_7"].iat[5, 4] = bad_value

    with pytest.raises(
        ValueError,
        match="savings_interest_income.*finite positive number",
    ):
        _materialize(monkeypatch, tmp_path, tables)


def test_rejects_duplicate_component_layout(monkeypatch, tmp_path) -> None:
    duplicate_layouts = (
        *hmrc_income._TABLE_LAYOUTS,
        hmrc_income._TABLE_LAYOUTS[0],
    )
    monkeypatch.setattr(hmrc_income, "_TABLE_LAYOUTS", duplicate_layouts)
    ods_path = tmp_path / "spi.ods"
    ods_path.write_bytes(b"source")

    with pytest.raises(ValueError, match="duplicates=.*employment_income"):
        materialize_hmrc_spi_income_band_targets(
            ods_path,
            build_period="2023",
        )
