"""Toy UK atomic-area supports whose codes match the toy OA ladder.

The toy ladder (``test_uk_ladder_rowwise_clone._ladder_frame``) carries two
London output areas, one Welsh, one Scottish and one Northern Irish. The UK
adapter requires every FRS region of a nation system to carry household mass,
so the England & Wales support adds one filler area per remaining English
region; no toy household lives there. Every code the toy ladder's rosters,
target fixtures and gates know is therefore also an atomic-area code here.
No publisher file or country engine is involved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from test_uk_ladder_rowwise_clone import _ladder_frame

from microcosm.build.uk_runtime.atomic_area_support import (
    SOURCES,
    SYSTEMS,
    assemble_uk_atomic_area_support,
)
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE

EW, SCOT, NI = SYSTEMS
_COLUMNS = (
    "oa_code",
    "population",
    "households",
    "constituency_code",
    "region_code",
    "lsoa_code",
    "msoa_code",
    "local_authority_code",
    "ward_code",
    "itl3_code",
)


def _filler_rows() -> list[dict]:
    """One populated area per English region absent from the toy ladder."""
    present = set(_ladder_frame()["region_code"])
    rows = []
    letters = "CDEFGHJK"
    for offset, (_name, code) in enumerate(
        (n, c)
        for n, c in FRS_REGION_TO_REGION_CODE.items()
        if c.startswith("E") and c not in present
    ):
        serial = 100 + offset
        rows.append(
            {
                "oa_code": f"E00{serial:06d}",
                "population": 1.0,
                "households": 1.0,
                "constituency_code": f"E14{serial:06d}",
                "region_code": code,
                "lsoa_code": f"E01{serial:06d}",
                "msoa_code": f"E02{serial:06d}",
                "local_authority_code": f"E06{serial:06d}",
                "ward_code": f"E05{serial:06d}",
                "itl3_code": f"TL{letters[offset]}11",
            }
        )
    return rows


def toy_support_frames() -> dict[str, list[dict]]:
    ladder = _ladder_frame().to_dict("records")
    return {
        EW: [r for r in ladder if r["oa_code"][0] in "EW"] + _filler_rows(),
        SCOT: [r for r in ladder if r["oa_code"][0] == "S"],
        NI: [r for r in ladder if r["oa_code"][0] == "N"],
    }


def _metadata(system: str) -> dict:
    vintage = "2022_census" if system == SCOT else "2021_census"
    descriptions = {}
    for column in _COLUMNS:
        if column in {"population", "households"}:
            descriptions[column] = {
                "kind": "weight",
                "source": f"toy-{system}-{column}",
                "basis": "toy persons" if column == "population" else "toy households",
            }
        else:
            descriptions[column] = {
                "kind": "code",
                "source": f"toy-{system}-{column}",
                "vintage": vintage,
                "relation": "exact"
                if column in {"oa_code", "lsoa_code", "msoa_code", "region_code"}
                else "best_fit",
            }
    descriptions["constituency_code"]["vintage"] = "2024_pcon"
    if system == NI:
        descriptions["constituency_code"]["relation"] = "official_tabulation"
    return descriptions


def toy_support_payloads() -> dict[str, bytes]:
    """The three toy support artifacts, keyed by system."""
    payloads = {}
    for system, rows in toy_support_frames().items():
        arrays = {}
        for column in _COLUMNS:
            values = [row[column] for row in rows]
            arrays[column] = (
                np.asarray(values, dtype=np.float64)
                if column in {"population", "households"}
                else np.asarray([str(v) for v in values], dtype="U")
            )
        payloads[system] = assemble_uk_atomic_area_support(
            system=system, arrays=arrays, column_metadata=_metadata(system)
        )
    return payloads


def write_toy_supports(directory: Path) -> tuple[dict[str, bytes], dict[str, Path]]:
    """Write the toy supports as raw-bytes sources named by ``SOURCES``."""
    directory.mkdir(parents=True, exist_ok=True)
    payloads = toy_support_payloads()
    paths = {}
    for system, payload in payloads.items():
        path = directory / f"{SOURCES[system]}.npz"
        path.write_bytes(payload)
        paths[system] = path
    return payloads, paths


def toy_support_sources(paths: dict[str, Path]) -> dict[str, Path]:
    """Graph source bindings for the three toy supports."""
    return {SOURCES[system]: path for system, path in paths.items()}
