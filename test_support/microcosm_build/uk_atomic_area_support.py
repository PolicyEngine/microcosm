"""Invented three-system UK atomic-area support payloads.

Shared by the atomic-area support and household-lineage tests: a few areas per
FRS region with fixed population and household masses. No publisher file or
country engine.
"""

from __future__ import annotations

import numpy as np

from microcosm.build.uk_runtime.atomic_area_support import (
    SYSTEMS,
    assemble_uk_atomic_area_support,
)
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE


def parts(system):
    index = SYSTEMS.index(system)
    regions = (
        [
            r
            for r in FRS_REGION_TO_REGION_CODE
            if r not in {"SCOTLAND", "NORTHERN_IRELAND"}
        ]
        if index == 0
        else ["SCOTLAND"]
        if index == 1
        else ["NORTHERN_IRELAND"]
    )
    rows = []
    region_order = list(FRS_REGION_TO_REGION_CODE)
    for region in regions:
        prefix = FRS_REGION_TO_REGION_CODE[region][0]
        for offset in range(3):
            serial = region_order.index(region) * 3 + offset + 1
            area = prefix + ("20" if index == 2 else "00") + f"{serial:06d}"
            rows.append(
                {
                    "oa_code": area,
                    "population": [2.0, 6.0, 1000.0][offset],
                    "households": [0.0, 30.0, 10.0][offset],
                    "constituency_code": prefix
                    + "14"
                    + f"{serial - offset + (offset == 2):06d}",
                    "region_code": FRS_REGION_TO_REGION_CODE[region],
                    "lsoa_code": area
                    if index == 2
                    else prefix + "01" + f"{serial:06d}",
                    "msoa_code": prefix + "02" + f"{serial:06d}",
                    "local_authority_code": prefix + "06" + f"{serial:06d}",
                    "ward_code": prefix + "05" + f"{serial:06d}",
                    "itl3_code": "TL"
                    + "CDEFGHIJKLMN"[region_order.index(region)]
                    + "01",
                }
            )
    arrays = {column: np.asarray([r[column] for r in rows]) for column in rows[0]}
    descriptions = {}
    for column in arrays:
        if column in {"population", "households"}:
            descriptions[column] = {
                "kind": "weight",
                "source": "invented-" + system + "-" + column,
                "basis": "invented persons"
                if column == "population"
                else "invented occupied households",
            }
        else:
            descriptions[column] = {
                "kind": "code",
                "source": "invented-" + system + "-" + column,
                "vintage": "2022_census" if index == 1 else "2021_census",
                "relation": "exact"
                if column in {"oa_code", "lsoa_code", "msoa_code", "region_code"}
                else "best_fit",
            }
    descriptions["constituency_code"]["vintage"] = "2024_pcon"
    if index == 2:
        descriptions["constituency_code"]["relation"] = "official_tabulation"
    return arrays, descriptions


def payloads():
    return {
        system: assemble_uk_atomic_area_support(
            system=system, arrays=arrays, column_metadata=metadata
        )
        for system in SYSTEMS
        for arrays, metadata in [parts(system)]
    }


__all__ = [name for name in globals() if not name.startswith("__")]
