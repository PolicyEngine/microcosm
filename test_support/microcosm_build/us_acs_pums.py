"""Tiny ACS PUMS archive fixtures shared by the loader and pool tool tests."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def _write_csv_zip(
    path: Path,
    members: dict[str, list[dict[str, object]]],
) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, rows in members.items():
            archive.writestr(name, pd.DataFrame(rows).to_csv(index=False))


def _household(serialno: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "SERIALNO": serialno,
        "ST": "06",
        "PUMA": "12345",
        "WGTP": 10,
        "NP": 1,
        "ADJHSG": 1_000_000,
        "TEN": 1,
        "RNTP": None,
        "GRNTP": None,
        "TAXAMT": 2_400,
        "TYPEHUGQ": 1,
    }
    row.update(overrides)
    return row


def _person(
    serialno: str,
    sporder: int,
    relationship: int,
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "SERIALNO": serialno,
        "SPORDER": sporder,
        "RELSHIPP": relationship,
        "AGEP": 40,
        "SEX": 1,
        "MAR": 1,
        "ADJINC": 1_000_000,
        "WAGP": 50_000,
        "SEMP": 0,
        "SSP": 0,
        "SSIP": 0,
        "RETP": 0,
        "INTP": 0,
        "PWGTP": 10,
    }
    row.update(overrides)
    return row
