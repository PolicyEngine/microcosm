"""Shared fixtures: a small person+household frame, and an ODS writer."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime.ods_tables import ODS_MIME_TYPE
from populace.frame import EntitySchema, Frame, WeightKind, Weights


@pytest.fixture
def small_frame() -> Frame:
    """Four persons in two households, with an income column."""
    person = pd.DataFrame(
        {
            "person_id": np.arange(4, dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 2], dtype="int64"),
            "income": np.asarray([100.0, 0.0, 250.0, 50.0]),
        }
    )
    household = pd.DataFrame({"household_id": np.asarray([1, 2], dtype="int64")})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                values=np.asarray([1000.0, 2000.0]), kind=WeightKind.DESIGN
            )
        },
    )


# --- ODS fixtures -------------------------------------------------------

_ODS_CONTENT_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
  xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
  xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
  <office:body><office:spreadsheet>{tables}</office:spreadsheet></office:body>
</office:document-content>
"""


def _cell(value: object, *, repeat: int = 1) -> str:
    span = f' table:number-columns-repeated="{repeat}"' if repeat > 1 else ""
    if value is None:
        return f"<table:table-cell{span}/>"
    if isinstance(value, (int, float)):
        return (
            f'<table:table-cell office:value-type="float" '
            f'office:value="{value}"{span}><text:p>{value}</text:p>'
            "</table:table-cell>"
        )
    return (
        f'<table:table-cell office:value-type="string"{span}>'
        f"<text:p>{value}</text:p></table:table-cell>"
    )


def _sheet(name: str, rows: list[list[object]], *, row_repeat: int = 1) -> str:
    span = f' table:number-rows-repeated="{row_repeat}"' if row_repeat > 1 else ""
    body = "".join(
        f"<table:table-row{span}>{''.join(_cell(value) for value in row)}"
        "</table:table-row>"
        for row in rows
    )
    return f'<table:table table:name="{name}">{body}</table:table>'


def _write_ods(
    path: Path,
    tables_xml: str,
    *,
    mimetype: str = ODS_MIME_TYPE,
    store_mimetype: bool = True,
    include_content: bool = True,
) -> Path:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype",
            mimetype,
            compress_type=ZIP_STORED if store_mimetype else ZIP_DEFLATED,
        )
        if include_content:
            archive.writestr(
                "content.xml", _ODS_CONTENT_TEMPLATE.format(tables=tables_xml)
            )
    return path


class ODSBuilder:
    """Build small OpenDocument spreadsheets for tests."""

    cell = staticmethod(_cell)
    sheet = staticmethod(_sheet)
    write = staticmethod(_write_ods)


@pytest.fixture
def ods() -> ODSBuilder:
    """Helpers for writing a small ODS file to a temporary path."""
    return ODSBuilder()
