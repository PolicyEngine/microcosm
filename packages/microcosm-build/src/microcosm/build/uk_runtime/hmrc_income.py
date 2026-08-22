"""Strict HMRC SPI income-band targets for the UK national build.

The official 2023-24 collated ODS publishes counts in thousands of people and
amounts in GBP millions.  This module converts those source units to people
and GBP while retaining the exact 13-band source surface.  It deliberately
has no network or incumbent UK data-package dependency: callers must supply
the local ODS artifact whose runtime hash is recorded in the returned provenance.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree
from zipfile import ZIP_STORED, BadZipFile, ZipFile

__all__ = [
    "HMRCIncomeBandTargetRecord",
    "HMRCIncomeSourceProvenance",
    "HMRCIncomeTargetSet",
    "HMRC_SPI_BUILD_PERIOD",
    "HMRC_SPI_ASSESSABLE_INCOME_COLUMN",
    "HMRC_SPI_COLLATED_ODS_FILENAME",
    "HMRC_SPI_COLLATED_ODS_MIME_TYPE",
    "HMRC_SPI_COLLATED_ODS_SHA256",
    "HMRC_SPI_COLLATED_ODS_SIZE_BYTES",
    "HMRC_SPI_COLLATED_ODS_URL",
    "HMRC_SPI_INCOME_BAND_LOWER_BOUNDS",
    "HMRC_SPI_INCOME_COMPONENTS",
    "HMRC_SPI_PUBLICATION_URL",
    "HMRC_SPI_SOURCE_TAX_YEAR",
    "HMRC_SPI_SOURCE_TAX_YEAR_START",
    "HMRC_SPI_SOURCE_VINTAGE",
    "HMRC_SPI_TARGET_RECORD_COUNT",
    "VerifiedHMRCODSIdentity",
    "hmrc_spi_component_source_columns",
    "materialize_hmrc_spi_income_band_targets",
    "verify_hmrc_spi_collated_ods",
]

HMRC_SPI_PUBLICATION_URL = (
    "https://www.gov.uk/government/statistics/"
    "personal-incomes-statistics-for-the-tax-year-2023-to-2024"
)
HMRC_SPI_COLLATED_ODS_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "69f1f12d2fae53a03709682f/Collated_Tables_3_1_to_3_11_2324.ods"
)
HMRC_SPI_COLLATED_ODS_FILENAME = "Collated_Tables_3_1_to_3_11_2324.ods"
HMRC_SPI_COLLATED_ODS_SHA256 = (
    "ad063b06b2bdeef8600dbbb09d48153337a4966f8c7eea50df7a2e0304ebd73e"
)
HMRC_SPI_COLLATED_ODS_SIZE_BYTES = 166_693
HMRC_SPI_COLLATED_ODS_MIME_TYPE = "application/vnd.oasis.opendocument.spreadsheet"
HMRC_SPI_SOURCE_VINTAGE = "2023-24"
HMRC_SPI_SOURCE_TAX_YEAR = HMRC_SPI_SOURCE_VINTAGE
HMRC_SPI_SOURCE_TAX_YEAR_START = 2023
HMRC_SPI_BUILD_PERIOD = "2024"
HMRC_SPI_ASSESSABLE_INCOME_COLUMN = "hmrc_spi_assessable_income"

HMRC_SPI_INCOME_BAND_LOWER_BOUNDS = (
    12_570,
    15_000,
    20_000,
    30_000,
    40_000,
    50_000,
    70_000,
    100_000,
    150_000,
    200_000,
    300_000,
    500_000,
    1_000_000,
)
HMRC_SPI_INCOME_COMPONENTS = (
    "employment_income",
    "self_employment_income",
    "state_pension",
    "private_pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "other_investment_income",
)
HMRC_SPI_TARGET_RECORD_COUNT = (
    len(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS) * len(HMRC_SPI_INCOME_COMPONENTS) * 2
)
_HMRC_ODS_VERIFICATION_TOKEN = object()

HMRCIncomeMeasure = Literal["count", "amount"]
HMRCIncomeUnit = Literal["people", "GBP"]


@dataclass(frozen=True)
class HMRCIncomeSourceProvenance:
    """Runtime identity and explicit period mapping for one local HMRC ODS."""

    local_path: Path
    sha256: str
    publication_url: str
    ods_url: str
    source_vintage: str
    source_tax_year: str
    source_tax_year_start: int
    build_period: str
    table_names: tuple[str, ...]
    size_bytes: int | None = None
    mime_type: str | None = None


@dataclass(frozen=True)
class HMRCIncomeBandTargetRecord:
    """One source-backed count or amount for one total-income band."""

    name: str
    component: str
    measure: HMRCIncomeMeasure
    unit: HMRCIncomeUnit
    value: float
    period: str
    total_income_lower_bound: int
    total_income_upper_bound: int | None


@dataclass(frozen=True)
class HMRCIncomeTargetSet:
    """A complete HMRC target surface and its shared source provenance."""

    source: HMRCIncomeSourceProvenance
    targets: tuple[HMRCIncomeBandTargetRecord, ...]


@dataclass(frozen=True)
class _HMRCODSFileFingerprint:
    """Stable-file identity binding a reviewed hash to the parsed ODS bytes."""

    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class VerifiedHMRCODSIdentity:
    """Opaque proof that the local ODS matched the reviewed official file."""

    path: Path
    sha256: str
    size_bytes: int
    mime_type: str
    fingerprint: _HMRCODSFileFingerprint
    _verification_token: object


@dataclass(frozen=True)
class _ComponentColumns:
    component: str
    count_position: int
    amount_position: int
    count_header: str
    amount_header: str


@dataclass(frozen=True)
class _TableLayout:
    sheet_name: str
    components: tuple[_ComponentColumns, ...]


@dataclass(frozen=True)
class _ODSRowRun:
    """One physical ODS row and its logical repetition span."""

    start_position: int
    repeat: int
    values: tuple[object, ...]


@dataclass(frozen=True)
class _ODSTable:
    """Small, repeat-aware view of one ODS worksheet."""

    sheet_name: str
    rows: tuple[_ODSRowRun, ...]
    significant_column_count: int

    @property
    def row_count(self) -> int:
        if not self.rows:
            return 0
        last = self.rows[-1]
        return last.start_position + last.repeat

    def cell(self, row_position: int, column_position: int) -> object:
        if row_position < 0 or column_position < 0:
            raise IndexError("ODS row and column positions must be non-negative.")
        for row in self.rows:
            if row.start_position <= row_position < row.start_position + row.repeat:
                if column_position >= len(row.values):
                    return None
                return row.values[column_position]
        return None

    def first_column_runs(
        self,
        *,
        start_position: int,
    ) -> tuple[tuple[int, object, int], ...]:
        result: list[tuple[int, object, int]] = []
        for row in self.rows:
            end_position = row.start_position + row.repeat
            if end_position <= start_position:
                continue
            effective_start = max(row.start_position, start_position)
            repeat = end_position - effective_start
            value = row.values[0] if row.values else None
            if value not in (None, ""):
                result.append((effective_start, value, repeat))
        return tuple(result)


_TABLE_LAYOUTS = (
    _TableLayout(
        sheet_name="Table_3_6",
        components=(
            _ComponentColumns(
                "self_employment_income",
                1,
                2,
                "Self-employment income (Number of individuals) [Note 1]",
                "Self-employment income (Amount) [Note 1]",
            ),
            _ComponentColumns(
                "employment_income",
                4,
                5,
                "Employment income (Number of individuals)",
                "Employment income (Amount)",
            ),
            _ComponentColumns(
                "state_pension",
                7,
                8,
                (
                    "Pension Income from National Insurance contributions "
                    "(Number of individuals) [Note 2]"
                ),
                (
                    "Pension Income from National Insurance contributions "
                    "(Amount) [Note 2]"
                ),
            ),
            _ComponentColumns(
                "private_pension_income",
                10,
                11,
                "Pension income from other pensions (Number of individuals)",
                "Pension income from other pensions (Amount)",
            ),
        ),
    ),
    _TableLayout(
        sheet_name="Table_3_7",
        components=(
            _ComponentColumns(
                "property_income",
                1,
                2,
                "Net income from property (Number of individuals) [Note 2]",
                "Net income from property (Amount) [Note 2]",
            ),
            _ComponentColumns(
                "savings_interest_income",
                4,
                5,
                (
                    "Interest from building societies and banks "
                    "(Number of individuals) [Note 3, 4]"
                ),
                ("Interest from building societies and banks (Amount) [Note 3, 4]"),
            ),
            _ComponentColumns(
                "dividend_income",
                7,
                8,
                "Dividends (Number of individuals) [Note 5]",
                "Dividends (Amount) [Note 5]",
            ),
            # Table 3.7's fourth pair is "Other income".  The enhanced-FRS
            # parser previously stopped at dividends, silently omitting these
            # published positions 10/11 from the target family.
            _ComponentColumns(
                "other_investment_income",
                10,
                11,
                "Other income (Number of individuals) [Note 6]",
                "Other income (Amount) [Note 6]",
            ),
        ),
    ),
)

_HEADER_ROW = 4
_FIRST_DATA_ROW = 5
_STOP_LABEL = "All ranges"
_INCOME_RANGE_HEADER = "Range of total income (lower limit) £"
_ODS_COLUMN_LIMIT = 32
_ODF_OFFICE_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_ODF_TABLE_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_ODF_TEXT_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_ODF_TABLE_TAG = f"{{{_ODF_TABLE_NAMESPACE}}}table"
_ODF_ROW_TAG = f"{{{_ODF_TABLE_NAMESPACE}}}table-row"
_ODF_CELL_TAG = f"{{{_ODF_TABLE_NAMESPACE}}}table-cell"
_ODF_COVERED_CELL_TAG = f"{{{_ODF_TABLE_NAMESPACE}}}covered-table-cell"
_ODF_TABLE_NAME_ATTRIBUTE = f"{{{_ODF_TABLE_NAMESPACE}}}name"
_ODF_ROW_REPEAT_ATTRIBUTE = f"{{{_ODF_TABLE_NAMESPACE}}}number-rows-repeated"
_ODF_COLUMN_REPEAT_ATTRIBUTE = f"{{{_ODF_TABLE_NAMESPACE}}}number-columns-repeated"
_ODF_VALUE_TYPE_ATTRIBUTE = f"{{{_ODF_OFFICE_NAMESPACE}}}value-type"
_ODF_VALUE_ATTRIBUTE = f"{{{_ODF_OFFICE_NAMESPACE}}}value"
_ODF_BOOLEAN_VALUE_ATTRIBUTE = f"{{{_ODF_OFFICE_NAMESPACE}}}boolean-value"
_ODF_STRING_VALUE_ATTRIBUTE = f"{{{_ODF_OFFICE_NAMESPACE}}}string-value"
_ODF_PARAGRAPH_TAG = f"{{{_ODF_TEXT_NAMESPACE}}}p"
_SOURCE_SCALE: dict[HMRCIncomeMeasure, float] = {
    "count": 1_000.0,
    "amount": 1_000_000.0,
}
_UNIT_BY_MEASURE: dict[HMRCIncomeMeasure, HMRCIncomeUnit] = {
    "count": "people",
    "amount": "GBP",
}


def hmrc_spi_component_source_columns() -> dict[str, tuple[str, int, int]]:
    """Return the exact published sheet/count/amount layout used at runtime."""

    return {
        component.component: (
            layout.sheet_name,
            component.count_position,
            component.amount_position,
        )
        for layout in _TABLE_LAYOUTS
        for component in layout.components
    }


def materialize_hmrc_spi_income_band_targets(
    ods_path: str | Path | VerifiedHMRCODSIdentity,
    *,
    build_period: int | str,
) -> HMRCIncomeTargetSet:
    """Parse the complete 2023-24 HMRC SPI income-band target surface.

    The latest published tax year 2023-24 is replayed against build period
    ``"2024"`` (signed, microcosm#723). Any missing, duplicate, or malformed
    source record raises; this function never narrows the family.
    """

    period = str(build_period)
    if period != HMRC_SPI_BUILD_PERIOD:
        raise ValueError(
            f"HMRC SPI source tax year {HMRC_SPI_SOURCE_TAX_YEAR} is the "
            "latest published tax year 2023-24 replayed against build period "
            f"{HMRC_SPI_BUILD_PERIOD!r}; got {period!r}."
        )

    _validate_component_layouts()
    identity = (
        ods_path
        if isinstance(ods_path, VerifiedHMRCODSIdentity)
        else verify_hmrc_spi_collated_ods(ods_path)
    )
    _assert_verified_hmrc_ods_current(identity)
    path = identity.path
    size_bytes = identity.size_bytes
    source_sha256 = identity.sha256
    tables = _read_ods_tables(path)
    _assert_verified_hmrc_ods_current(identity)

    source = HMRCIncomeSourceProvenance(
        local_path=path,
        sha256=source_sha256,
        publication_url=HMRC_SPI_PUBLICATION_URL,
        ods_url=HMRC_SPI_COLLATED_ODS_URL,
        source_vintage=HMRC_SPI_SOURCE_VINTAGE,
        source_tax_year=HMRC_SPI_SOURCE_TAX_YEAR,
        source_tax_year_start=HMRC_SPI_SOURCE_TAX_YEAR_START,
        build_period=period,
        table_names=tuple(layout.sheet_name for layout in _TABLE_LAYOUTS),
        size_bytes=size_bytes,
        mime_type=HMRC_SPI_COLLATED_ODS_MIME_TYPE,
    )

    records: list[HMRCIncomeBandTargetRecord] = []
    for layout in _TABLE_LAYOUTS:
        table = tables[layout.sheet_name]
        _validate_table_headers(table, layout=layout)
        records.extend(_records_from_table(table, layout=layout, period=period))

    targets = tuple(records)
    _validate_target_surface(targets)
    return HMRCIncomeTargetSet(source=source, targets=targets)


def verify_hmrc_spi_collated_ods(
    ods_path: str | Path,
) -> VerifiedHMRCODSIdentity:
    """Verify the official ODS before parsing and bind the proof to its file."""

    path = Path(ods_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HMRC SPI collated ODS not found: {path}.")
    before = _hmrc_ods_fingerprint(path)
    if before.size_bytes != HMRC_SPI_COLLATED_ODS_SIZE_BYTES:
        raise ValueError(
            "HMRC SPI collated ODS size does not match the reviewed source "
            f"identity: expected {HMRC_SPI_COLLATED_ODS_SIZE_BYTES}, got "
            f"{before.size_bytes}."
        )
    source_sha256 = _sha256(path)
    after = _hmrc_ods_fingerprint(path)
    if after != before:
        raise RuntimeError(
            "HMRC SPI collated ODS changed while its reviewed identity was "
            "being verified."
        )
    if source_sha256 != HMRC_SPI_COLLATED_ODS_SHA256:
        raise ValueError(
            "HMRC SPI collated ODS SHA-256 does not match the reviewed source "
            f"identity: expected {HMRC_SPI_COLLATED_ODS_SHA256}, got "
            f"{source_sha256}."
        )
    return VerifiedHMRCODSIdentity(
        path=path,
        sha256=source_sha256,
        size_bytes=before.size_bytes,
        mime_type=HMRC_SPI_COLLATED_ODS_MIME_TYPE,
        fingerprint=before,
        _verification_token=_HMRC_ODS_VERIFICATION_TOKEN,
    )


def _read_ods_tables(path: Path) -> dict[str, _ODSTable]:
    """Read the two reviewed worksheets with the Python standard library."""

    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if names.count("mimetype") != 1:
                raise ValueError(
                    "HMRC SPI collated ODS must contain exactly one mimetype entry."
                )
            mimetype_info = archive.getinfo("mimetype")
            if mimetype_info.compress_type != ZIP_STORED:
                raise ValueError(
                    "HMRC SPI collated ODS mimetype entry must be stored "
                    "without compression."
                )
            try:
                mimetype = archive.read("mimetype").decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    "HMRC SPI collated ODS mimetype entry must be ASCII."
                ) from exc
            if mimetype != HMRC_SPI_COLLATED_ODS_MIME_TYPE:
                raise ValueError(
                    "HMRC SPI collated ODS MIME type does not match the reviewed "
                    f"source identity: expected {HMRC_SPI_COLLATED_ODS_MIME_TYPE!r}, "
                    f"got {mimetype!r}."
                )
            if names.count("content.xml") != 1:
                raise ValueError(
                    "HMRC SPI collated ODS must contain exactly one content.xml entry."
                )
            content = archive.read("content.xml")
    except BadZipFile as exc:
        raise ValueError("HMRC SPI collated ODS is not a valid ZIP container.") from exc

    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        raise ValueError("HMRC SPI collated ODS content.xml is malformed.") from exc

    required_sheets = {layout.sheet_name for layout in _TABLE_LAYOUTS}
    table_elements: dict[str, ElementTree.Element] = {}
    for element in root.iter(_ODF_TABLE_TAG):
        sheet_name = element.attrib.get(_ODF_TABLE_NAME_ATTRIBUTE)
        if sheet_name not in required_sheets:
            continue
        if sheet_name in table_elements:
            raise ValueError(
                f"HMRC SPI collated ODS contains duplicate sheet {sheet_name!r}."
            )
        table_elements[sheet_name] = element
    missing = sorted(required_sheets - set(table_elements))
    if missing:
        raise ValueError(f"HMRC SPI collated ODS is missing sheet(s): {missing}.")
    return {
        sheet_name: _ods_table_from_element(element, sheet_name=sheet_name)
        for sheet_name, element in table_elements.items()
    }


def _hmrc_ods_fingerprint(path: Path) -> _HMRCODSFileFingerprint:
    stat = path.stat()
    return _HMRCODSFileFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


def _assert_verified_hmrc_ods_current(identity: VerifiedHMRCODSIdentity) -> None:
    if identity._verification_token is not _HMRC_ODS_VERIFICATION_TOKEN:
        raise ValueError(
            "HMRC ODS identity must come from verify_hmrc_spi_collated_ods()."
        )
    if _hmrc_ods_fingerprint(identity.path) != identity.fingerprint:
        raise RuntimeError(
            "HMRC SPI collated ODS changed after identity verification; "
            "refusing to parse bytes not bound to the reviewed SHA-256."
        )


def _ods_table_from_element(
    element: ElementTree.Element,
    *,
    sheet_name: str,
) -> _ODSTable:
    rows: list[_ODSRowRun] = []
    logical_row_position = 0
    significant_column_count = 0
    for physical_row_position, row_element in enumerate(element.iter(_ODF_ROW_TAG)):
        repeat = _positive_repeat(
            row_element.attrib.get(_ODF_ROW_REPEAT_ATTRIBUTE),
            label=f"{sheet_name} row {physical_row_position} repeat",
        )
        values = _ods_row_values(
            row_element,
            sheet_name=sheet_name,
            logical_row_position=logical_row_position,
        )
        significant_column_count = max(significant_column_count, len(values))
        rows.append(
            _ODSRowRun(
                start_position=logical_row_position,
                repeat=repeat,
                values=values,
            )
        )
        logical_row_position += repeat
    return _ODSTable(
        sheet_name=sheet_name,
        rows=tuple(rows),
        significant_column_count=significant_column_count,
    )


def _ods_row_values(
    row_element: ElementTree.Element,
    *,
    sheet_name: str,
    logical_row_position: int,
) -> tuple[object, ...]:
    values: list[object] = []
    logical_column_position = 0
    for cell in row_element:
        if cell.tag not in {_ODF_CELL_TAG, _ODF_COVERED_CELL_TAG}:
            continue
        repeat = _positive_repeat(
            cell.attrib.get(_ODF_COLUMN_REPEAT_ATTRIBUTE),
            label=(
                f"{sheet_name} row {logical_row_position} column "
                f"{logical_column_position} repeat"
            ),
        )
        value = (
            None
            if cell.tag == _ODF_COVERED_CELL_TAG
            else _ods_cell_value(
                cell,
                sheet_name=sheet_name,
                row_position=logical_row_position,
                column_position=logical_column_position,
            )
        )
        if logical_column_position < _ODS_COLUMN_LIMIT:
            retained_repeat = min(
                repeat,
                _ODS_COLUMN_LIMIT - logical_column_position,
            )
            values.extend([value] * retained_repeat)
        logical_column_position += repeat

    while values and values[-1] in (None, ""):
        values.pop()
    return tuple(values)


def _ods_cell_value(
    cell: ElementTree.Element,
    *,
    sheet_name: str,
    row_position: int,
    column_position: int,
) -> object:
    value_type = cell.attrib.get(_ODF_VALUE_TYPE_ATTRIBUTE)
    label = f"{sheet_name} row {row_position} column {column_position}"
    if value_type in {"float", "currency", "percentage"}:
        raw_value = cell.attrib.get(_ODF_VALUE_ATTRIBUTE)
        try:
            return float(raw_value) if raw_value is not None else math.nan
        except ValueError as exc:
            raise ValueError(
                f"{label} has invalid numeric ODS value {raw_value!r}."
            ) from exc
    if value_type == "boolean":
        raw_value = cell.attrib.get(_ODF_BOOLEAN_VALUE_ATTRIBUTE)
        if raw_value not in {"true", "false"}:
            raise ValueError(f"{label} has invalid boolean ODS value {raw_value!r}.")
        return raw_value == "true"
    if value_type not in {None, "string"}:
        raise ValueError(f"{label} has unsupported ODS value type {value_type!r}.")

    paragraphs = [
        "".join(paragraph.itertext()) for paragraph in cell.iter(_ODF_PARAGRAPH_TAG)
    ]
    if paragraphs:
        return "\n".join(paragraphs)
    string_value = cell.attrib.get(_ODF_STRING_VALUE_ATTRIBUTE)
    if string_value is not None:
        return string_value
    return None


def _positive_repeat(raw_value: str | None, *, label: str) -> int:
    if raw_value is None:
        return 1
    try:
        repeat = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be a positive integer, got {raw_value!r}."
        ) from exc
    if repeat <= 0:
        raise ValueError(f"{label} must be a positive integer, got {raw_value!r}.")
    return repeat


def _validate_table_headers(table: _ODSTable, *, layout: _TableLayout) -> None:
    expected_headers = {0: _INCOME_RANGE_HEADER}
    for component in layout.components:
        expected_headers[component.count_position] = component.count_header
        expected_headers[component.amount_position] = component.amount_header
    for column_position, expected in sorted(expected_headers.items()):
        actual = table.cell(_HEADER_ROW, column_position)
        if actual != expected:
            raise ValueError(
                f"{layout.sheet_name} row {_HEADER_ROW} column {column_position} "
                f"header mismatch: expected {expected!r}, got {actual!r}."
            )


def _records_from_table(
    table: _ODSTable,
    *,
    layout: _TableLayout,
    period: str,
) -> list[HMRCIncomeBandTargetRecord]:
    if not isinstance(table, _ODSTable):
        raise TypeError(
            f"{layout.sheet_name} must load as an ODS table, "
            f"got {type(table).__name__}."
        )
    if table.significant_column_count == 0:
        raise ValueError(f"{layout.sheet_name} has no columns.")

    missing_positions = [
        f"{component.component} count (position {component.count_position})"
        for component in layout.components
        if component.count_position >= table.significant_column_count
    ]
    missing_positions.extend(
        f"{component.component} amount (position {component.amount_position})"
        for component in layout.components
        if component.amount_position >= table.significant_column_count
    )
    if missing_positions:
        raise ValueError(
            f"{layout.sheet_name} is missing component column(s): {missing_positions}."
        )

    band_rows = _strict_band_rows(table, sheet_name=layout.sheet_name)
    records: list[HMRCIncomeBandTargetRecord] = []
    upper_bounds = (*HMRC_SPI_INCOME_BAND_LOWER_BOUNDS[1:], None)
    for lower_bound, upper_bound in zip(
        HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
        upper_bounds,
        strict=True,
    ):
        row_position = band_rows[lower_bound]
        for component in layout.components:
            for measure, column_position in (
                ("count", component.count_position),
                ("amount", component.amount_position),
            ):
                source_value = _strict_positive_value(
                    table.cell(row_position, column_position),
                    sheet_name=layout.sheet_name,
                    component=component.component,
                    measure=measure,
                    lower_bound=lower_bound,
                )
                value = source_value * _SOURCE_SCALE[measure]
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"{layout.sheet_name} component {component.component!r} "
                        f"{measure} at lower band {lower_bound} does not convert "
                        "to a finite positive target."
                    )
                records.append(
                    HMRCIncomeBandTargetRecord(
                        name=_target_name(
                            component.component,
                            measure,
                            lower_bound,
                            upper_bound,
                        ),
                        component=component.component,
                        measure=measure,
                        unit=_UNIT_BY_MEASURE[measure],
                        value=value,
                        period=period,
                        total_income_lower_bound=lower_bound,
                        total_income_upper_bound=upper_bound,
                    )
                )
    return records


def _strict_band_rows(table: _ODSTable, *, sheet_name: str) -> dict[int, int]:
    if table.row_count <= _FIRST_DATA_ROW:
        raise ValueError(f"{sheet_name} has no income-band rows.")

    parsed: list[tuple[int, int]] = []
    stop_count = 0
    stop_position: int | None = None
    repeated_band_values: list[int] = []
    for row_position, raw_label, repeat in table.first_column_runs(
        start_position=_FIRST_DATA_ROW
    ):
        if isinstance(raw_label, str) and raw_label.strip() == _STOP_LABEL:
            stop_count += repeat
            if stop_position is None:
                stop_position = row_position
        lower_bound = _integer_or_none(raw_label)
        if lower_bound is not None:
            parsed.append((lower_bound, row_position))
            if repeat > 1:
                repeated_band_values.append(lower_bound)

    if stop_count != 1:
        raise ValueError(
            f"{sheet_name} must contain exactly one {_STOP_LABEL!r} sentinel; "
            f"found {stop_count}."
        )
    if parsed and stop_position is not None and stop_position <= parsed[-1][1]:
        raise ValueError(
            f"{sheet_name} {_STOP_LABEL!r} sentinel must follow all published "
            "income-band rows."
        )

    actual = tuple(lower_bound for lower_bound, _ in parsed)
    counts = Counter(actual)
    duplicates = sorted(
        set(repeated_band_values)
        | {value for value, count in counts.items() if count > 1}
    )
    if duplicates:
        raise ValueError(
            f"{sheet_name} contains duplicate income lower band(s): {duplicates}."
        )

    expected_set = set(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected or len(actual) != len(expected_set):
        raise ValueError(
            f"{sheet_name} must contain exactly the 13 published income lower "
            f"bands; missing={missing}, unexpected={unexpected}, "
            f"observed_count={len(actual)}."
        )
    if actual != HMRC_SPI_INCOME_BAND_LOWER_BOUNDS:
        raise ValueError(
            f"{sheet_name} income lower bands are out of published order: {actual}."
        )
    return dict(parsed)


def _validate_component_layouts() -> None:
    components = [
        component.component
        for layout in _TABLE_LAYOUTS
        for component in layout.components
    ]
    counts = Counter(components)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    missing = sorted(set(HMRC_SPI_INCOME_COMPONENTS) - set(components))
    unexpected = sorted(set(components) - set(HMRC_SPI_INCOME_COMPONENTS))
    if (
        duplicates
        or missing
        or unexpected
        or len(components) != len(HMRC_SPI_INCOME_COMPONENTS)
    ):
        raise ValueError(
            "HMRC SPI table layout must define each required component exactly "
            f"once; duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}."
        )


def _validate_target_surface(
    targets: tuple[HMRCIncomeBandTargetRecord, ...],
) -> None:
    keys = [
        (target.total_income_lower_bound, target.component, target.measure)
        for target in targets
    ]
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate HMRC SPI target record(s): {duplicates}.")

    expected_keys = {
        (lower_bound, component, measure)
        for lower_bound in HMRC_SPI_INCOME_BAND_LOWER_BOUNDS
        for component in HMRC_SPI_INCOME_COMPONENTS
        for measure in ("count", "amount")
    }
    actual_keys = set(keys)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected or len(targets) != HMRC_SPI_TARGET_RECORD_COUNT:
        raise ValueError(
            "HMRC SPI target surface is incomplete; "
            f"expected {HMRC_SPI_TARGET_RECORD_COUNT} records, got "
            f"{len(targets)}, missing={missing}, unexpected={unexpected}."
        )


def _strict_positive_value(
    value: object,
    *,
    sheet_name: str,
    component: str,
    measure: HMRCIncomeMeasure,
    lower_bound: int,
) -> float:
    if isinstance(value, bool):
        numeric = math.nan
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(
            f"{sheet_name} component {component!r} {measure} at lower band "
            f"{lower_bound} must be a finite positive number; got {value!r}."
        )
    return numeric


def _integer_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _target_name(
    component: str,
    measure: HMRCIncomeMeasure,
    lower_bound: int,
    upper_bound: int | None,
) -> str:
    formatted_lower = f"{lower_bound:_}"
    formatted_upper = "inf" if upper_bound is None else f"{upper_bound:_}"
    component_label = component if measure == "amount" else f"{component}_count"
    return f"hmrc/{component_label}_income_band_{formatted_lower}_to_{formatted_upper}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
