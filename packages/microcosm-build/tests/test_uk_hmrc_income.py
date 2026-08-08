from __future__ import annotations

from pathlib import Path

import pytest

from microcosm.build.uk_runtime import hmrc_income
from microcosm.build.uk_runtime.hmrc_income import (
    HMRC_SPI_BUILD_PERIOD,
    HMRC_SPI_COLLATED_ODS_SHA256,
    HMRC_SPI_COLLATED_ODS_SIZE_BYTES,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_TARGET_RECORD_COUNT,
    materialize_hmrc_spi_income_band_targets,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PINNED_ODS_PATH = (
    _REPO_ROOT / "inputs" / "hmrc" / "Collated_Tables_3_1_to_3_11_2324.ods"
)


def _layout(sheet_name: str) -> hmrc_income._TableLayout:
    return next(
        layout
        for layout in hmrc_income._TABLE_LAYOUTS
        if layout.sheet_name == sheet_name
    )


def _source_rows(layout: hmrc_income._TableLayout) -> list[list[object]]:
    rows: list[list[object]] = [[None] * 12 for _ in range(19)]
    rows[hmrc_income._HEADER_ROW][0] = hmrc_income._INCOME_RANGE_HEADER
    for component in layout.components:
        rows[hmrc_income._HEADER_ROW][component.count_position] = component.count_header
        rows[hmrc_income._HEADER_ROW][component.amount_position] = (
            component.amount_header
        )

    for band_index, lower_bound in enumerate(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS):
        row_position = hmrc_income._FIRST_DATA_ROW + band_index
        rows[row_position][0] = lower_bound
        for component_index, component in enumerate(layout.components):
            rows[row_position][component.count_position] = (
                10 + component_index + band_index
            )
            rows[row_position][component.amount_position] = (
                20 + component_index + band_index
            )
    rows[18][0] = hmrc_income._STOP_LABEL
    return rows


def _table(
    layout: hmrc_income._TableLayout,
    *,
    rows: list[list[object]] | None = None,
    significant_column_count: int = 12,
    repeats: dict[int, int] | None = None,
) -> hmrc_income._ODSTable:
    rows = _source_rows(layout) if rows is None else rows
    repeats = {} if repeats is None else repeats
    logical_position = 0
    runs: list[hmrc_income._ODSRowRun] = []
    for physical_position, values in enumerate(rows):
        repeat = repeats.get(physical_position, 1)
        runs.append(
            hmrc_income._ODSRowRun(
                start_position=logical_position,
                repeat=repeat,
                values=tuple(values),
            )
        )
        logical_position += repeat
    return hmrc_income._ODSTable(
        sheet_name=layout.sheet_name,
        rows=tuple(runs),
        significant_column_count=significant_column_count,
    )


def _records_from_synthetic_tables():
    records = []
    for layout in hmrc_income._TABLE_LAYOUTS:
        table = _table(layout)
        hmrc_income._validate_table_headers(table, layout=layout)
        records.extend(
            hmrc_income._records_from_table(
                table,
                layout=layout,
                period=HMRC_SPI_BUILD_PERIOD,
            )
        )
    targets = tuple(records)
    hmrc_income._validate_target_surface(targets)
    return targets


def _target(targets, *, component: str, measure: str, lower_bound: int):
    matches = [
        target
        for target in targets
        if target.component == component
        and target.measure == measure
        and target.total_income_lower_bound == lower_bound
    ]
    assert len(matches) == 1
    return matches[0]


def test_stdlib_ods_table_records_cover_complete_published_surface() -> None:
    targets = _records_from_synthetic_tables()

    assert len(targets) == HMRC_SPI_TARGET_RECORD_COUNT == 208
    assert {target.component for target in targets} == set(HMRC_SPI_INCOME_COMPONENTS)
    assert {target.period for target in targets} == {HMRC_SPI_BUILD_PERIOD}

    savings_count = _target(
        targets,
        component="savings_interest_income",
        measure="count",
        lower_bound=12_570,
    )
    other_amount = _target(
        targets,
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
        targets,
        component="other_investment_income",
        measure="count",
        lower_bound=1_000_000,
    )
    assert top_band.total_income_upper_bound is None


@pytest.mark.skipif(
    not _PINNED_ODS_PATH.is_file(),
    reason="reviewed HMRC ODS is an optional local input",
)
def test_real_pinned_ods_materializes_all_208_facts() -> None:
    result = materialize_hmrc_spi_income_band_targets(
        _PINNED_ODS_PATH,
        build_period=HMRC_SPI_BUILD_PERIOD,
    )

    assert _PINNED_ODS_PATH.stat().st_size == HMRC_SPI_COLLATED_ODS_SIZE_BYTES
    assert result.source.sha256 == HMRC_SPI_COLLATED_ODS_SHA256
    assert result.source.local_path == _PINNED_ODS_PATH.resolve()
    assert result.source.source_vintage == "2023-24"
    assert result.source.source_tax_year == "2023-24"
    assert result.source.source_tax_year_start == 2023
    assert result.source.build_period == HMRC_SPI_BUILD_PERIOD
    assert result.source.table_names == ("Table_3_6", "Table_3_7")

    expected_keys = {
        (lower_bound, component, measure)
        for lower_bound in HMRC_SPI_INCOME_BAND_LOWER_BOUNDS
        for component in HMRC_SPI_INCOME_COMPONENTS
        for measure in ("count", "amount")
    }
    actual_keys = {
        (target.total_income_lower_bound, target.component, target.measure)
        for target in result.targets
    }
    assert len(result.targets) == HMRC_SPI_TARGET_RECORD_COUNT == 208
    assert actual_keys == expected_keys
    assert all(target.value > 0 for target in result.targets)


def test_rejects_period_before_opening_ods(monkeypatch, tmp_path) -> None:
    ods_path = tmp_path / "spi.ods"
    ods_path.write_bytes(b"source")
    called = False

    def should_not_parse(_path: Path):
        nonlocal called
        called = True
        raise AssertionError("period validation must precede ODS parsing")

    monkeypatch.setattr(hmrc_income, "_read_ods_tables", should_not_parse)

    with pytest.raises(ValueError, match="maps to build period '2023'"):
        materialize_hmrc_spi_income_band_targets(
            ods_path,
            build_period="2024",
        )
    assert called is False


def test_rejects_ods_size_before_parsing(monkeypatch, tmp_path) -> None:
    ods_path = tmp_path / "spi.ods"
    ods_path.write_bytes(b"source")
    called = False

    def should_not_parse(_path: Path):
        nonlocal called
        called = True
        raise AssertionError("identity verification must precede ODS parsing")

    monkeypatch.setattr(hmrc_income, "_read_ods_tables", should_not_parse)

    with pytest.raises(ValueError, match="size does not match the reviewed source"):
        materialize_hmrc_spi_income_band_targets(
            ods_path,
            build_period=HMRC_SPI_BUILD_PERIOD,
        )
    assert called is False


def test_rejects_ods_sha256_before_parsing(monkeypatch, tmp_path) -> None:
    ods_path = tmp_path / "spi.ods"
    ods_path.write_bytes(b"x" * HMRC_SPI_COLLATED_ODS_SIZE_BYTES)
    called = False

    def should_not_parse(_path: Path):
        nonlocal called
        called = True
        raise AssertionError("identity verification must precede ODS parsing")

    monkeypatch.setattr(hmrc_income, "_read_ods_tables", should_not_parse)

    with pytest.raises(ValueError, match="SHA-256 does not match the reviewed source"):
        materialize_hmrc_spi_income_band_targets(
            ods_path,
            build_period=HMRC_SPI_BUILD_PERIOD,
        )
    assert called is False


def test_rejects_header_drift() -> None:
    layout = _layout("Table_3_6")
    rows = _source_rows(layout)
    rows[hmrc_income._HEADER_ROW][4] = "Employed Income"

    with pytest.raises(ValueError, match="header mismatch"):
        hmrc_income._validate_table_headers(_table(layout, rows=rows), layout=layout)


def test_rejects_missing_published_band() -> None:
    layout = _layout("Table_3_7")
    rows = _source_rows(layout)
    rows.pop(9)

    with pytest.raises(ValueError, match=r"missing=\[40000\]"):
        hmrc_income._records_from_table(
            _table(layout, rows=rows),
            layout=layout,
            period=HMRC_SPI_BUILD_PERIOD,
        )


def test_rejects_duplicate_published_band() -> None:
    layout = _layout("Table_3_6")
    rows = _source_rows(layout)
    rows[6][0] = 12_570

    with pytest.raises(ValueError, match="duplicate income lower band"):
        hmrc_income._records_from_table(
            _table(layout, rows=rows),
            layout=layout,
            period=HMRC_SPI_BUILD_PERIOD,
        )


@pytest.mark.parametrize("sentinel_count", [0, 2])
def test_rejects_missing_or_duplicate_all_ranges_sentinel(sentinel_count) -> None:
    layout = _layout("Table_3_7")
    rows = _source_rows(layout)
    repeats = None
    if sentinel_count == 0:
        rows[18][0] = None
    else:
        repeats = {18: 2}

    with pytest.raises(ValueError, match="exactly one 'All ranges' sentinel"):
        hmrc_income._records_from_table(
            _table(layout, rows=rows, repeats=repeats),
            layout=layout,
            period=HMRC_SPI_BUILD_PERIOD,
        )


def test_rejects_all_ranges_sentinel_before_last_band() -> None:
    layout = _layout("Table_3_7")
    rows = _source_rows(layout)
    last_band_row = (
        hmrc_income._FIRST_DATA_ROW + len(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS) - 1
    )
    rows[last_band_row - 1][0] = hmrc_income._STOP_LABEL
    rows[18][0] = None

    with pytest.raises(ValueError, match="sentinel must follow"):
        hmrc_income._records_from_table(
            _table(layout, rows=rows),
            layout=layout,
            period=HMRC_SPI_BUILD_PERIOD,
        )


def test_rejects_missing_component_column() -> None:
    layout = _layout("Table_3_7")

    with pytest.raises(
        ValueError,
        match=r"other_investment_income amount \(position 11\)",
    ):
        hmrc_income._records_from_table(
            _table(layout, significant_column_count=11),
            layout=layout,
            period=HMRC_SPI_BUILD_PERIOD,
        )


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_nonpositive_or_nonfinite_component_values(bad_value) -> None:
    layout = _layout("Table_3_7")
    rows = _source_rows(layout)
    rows[hmrc_income._FIRST_DATA_ROW][4] = bad_value

    with pytest.raises(
        ValueError,
        match="savings_interest_income.*finite positive number",
    ):
        hmrc_income._records_from_table(
            _table(layout, rows=rows),
            layout=layout,
            period=HMRC_SPI_BUILD_PERIOD,
        )


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
            build_period=HMRC_SPI_BUILD_PERIOD,
        )
