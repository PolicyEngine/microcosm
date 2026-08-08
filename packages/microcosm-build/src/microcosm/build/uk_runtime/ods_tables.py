"""Read OpenDocument spreadsheets with the Python standard library.

Published statistics arrive as ODS, and a build that reaches for a reader
dependency to open them takes on a transitive surface for what is a zip
holding one XML document. This module reads that document directly.

``hmrc_income`` carries its own reader with error messages naming the SPI
artifact. New source stages use this one, which takes the artifact label as an
argument; migrating that module onto this is left as a separate change so a
certified surface does not move inside a feature.

Repetition is the part worth care. ODS collapses runs of identical cells and
rows into ``table:number-columns-repeated`` and ``table:number-rows-repeated``
attributes, so a sheet's physical elements do not correspond to its logical
grid. Both are expanded on read, bounded so that a trailing run declaring a
million empty columns cannot exhaust memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZIP_STORED, BadZipFile, ZipFile

__all__ = [
    "ODS_MIME_TYPE",
    "ODSTable",
    "read_ods_tables",
]

ODS_MIME_TYPE = "application/vnd.oasis.opendocument.spreadsheet"

_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"

#: Bound on a single repeat run. Trailing runs in published sheets declare
#: very large spans to pad the grid, and expanding those literally would
#: allocate without limit.
_MAX_REPEAT = 4_096


@dataclass(frozen=True)
class ODSTable:
    """One worksheet as a dense grid of cell values."""

    sheet_name: str
    rows: tuple[tuple[object, ...], ...]

    def cell(self, row: int, column: int) -> object:
        """Return one cell, or None outside the populated grid."""
        if not 0 <= row < len(self.rows):
            return None
        values = self.rows[row]
        if not 0 <= column < len(values):
            return None
        return values[column]

    def column(self, column: int) -> tuple[object, ...]:
        """Return one column across every row."""
        return tuple(self.cell(row, column) for row in range(len(self.rows)))


def _repeat(element: ElementTree.Element, attribute: str, *, label: str) -> int:
    raw = element.get(f"{{{_TABLE_NS}}}{attribute}")
    if raw is None:
        return 1
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{label} declares a non-integer {attribute} of {raw!r}."
        ) from exc
    if value < 1:
        raise ValueError(f"{label} declares a {attribute} of {value}, below one.")
    return min(value, _MAX_REPEAT)


def _cell_value(cell: ElementTree.Element, *, label: str) -> object:
    """Return a cell's typed value, preferring the office value attributes.

    Suppressed cells in published tables carry text such as "[Fewer than 1]"
    with no office value, and come back as that string for the caller to
    handle rather than being coerced to a number or dropped.
    """
    value_type = cell.get(f"{{{_OFFICE_NS}}}value-type")
    if value_type in {"float", "percentage", "currency"}:
        raw = cell.get(f"{{{_OFFICE_NS}}}value")
        if raw is None:
            raise ValueError(f"{label} has a {value_type} cell with no value.")
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{label} has a {value_type} cell with a non-numeric value {raw!r}."
            ) from exc

    paragraphs = [
        "".join(paragraph.itertext()) for paragraph in cell.iterfind(f"{{{_TEXT_NS}}}p")
    ]
    if not paragraphs:
        return None
    return "\n".join(paragraphs).strip() or None


def _table_from_element(element: ElementTree.Element, *, label: str) -> ODSTable:
    sheet_name = element.get(f"{{{_TABLE_NS}}}name") or ""
    rows: list[tuple[object, ...]] = []
    for row_element in element.iterfind(f"{{{_TABLE_NS}}}table-row"):
        values: list[object] = []
        for cell in row_element.iterfind(f"{{{_TABLE_NS}}}table-cell"):
            value = _cell_value(cell, label=label)
            span = _repeat(cell, "number-columns-repeated", label=label)
            values.extend([value] * span)
        while values and values[-1] is None:
            values.pop()
        row_span = _repeat(row_element, "number-rows-repeated", label=label)
        rows.extend([tuple(values)] * row_span)
    while rows and not rows[-1]:
        rows.pop()
    return ODSTable(sheet_name=sheet_name, rows=tuple(rows))


def read_ods_tables(path: Path, *, label: str) -> dict[str, ODSTable]:
    """Read every worksheet in an ODS file, keyed by sheet name.

    Args:
        path: The local artifact to read.
        label: How to name the artifact in error messages, so a malformed
            file reports which source it came from.

    Raises:
        ValueError: If the file is not a well-formed ODS spreadsheet.
    """
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if names.count("mimetype") != 1:
                raise ValueError(f"{label} must contain exactly one mimetype entry.")
            if archive.getinfo("mimetype").compress_type != ZIP_STORED:
                raise ValueError(
                    f"{label} must store its mimetype entry without compression."
                )
            try:
                mimetype = archive.read("mimetype").decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{label} must have an ASCII mimetype entry.") from exc
            if mimetype != ODS_MIME_TYPE:
                raise ValueError(
                    f"{label} declares mimetype {mimetype!r}, not {ODS_MIME_TYPE!r}."
                )
            if names.count("content.xml") != 1:
                raise ValueError(f"{label} must contain exactly one content.xml entry.")
            content = archive.read("content.xml")
    except BadZipFile as exc:
        raise ValueError(f"{label} is not a readable zip archive.") from exc

    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        raise ValueError(f"{label} has unparseable content.xml.") from exc

    tables: dict[str, ODSTable] = {}
    for element in root.iter(f"{{{_TABLE_NS}}}table"):
        table = _table_from_element(element, label=label)
        if table.sheet_name in tables:
            raise ValueError(f"{label} repeats the sheet name {table.sheet_name!r}.")
        tables[table.sheet_name] = table
    if not tables:
        raise ValueError(f"{label} contains no worksheets.")
    return tables
