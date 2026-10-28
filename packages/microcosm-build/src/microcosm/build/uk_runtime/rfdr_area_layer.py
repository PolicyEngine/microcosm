"""The Rural Fuel Duty Relief area layer and each placed household's flag (microcosm#1418).

HMRC Notice 2001 opens the relief to retailers in the Inner and Outer
Hebrides, the Northern Isles, the Islands in the Clyde, the Isles of Scilly,
the post town of Hawes and sixteen mainland postcode districts.
policyengine-uk prices a household's petrol and diesel at the relieved rate
when ``in_rural_fuel_duty_relief_area`` is true.

The notice names the island groups without bounding them, so Microcosm
declares its own translation into postcode districts and sectors, beside the
notice's own districts (``tools/build_uk_rfdr_area_layer.py``). The tool reads
the ONS postcode lookup and flags every 2021 output area of England and Wales
and 2022 output area of Scotland whose postcodes mostly lie in the scheme;
``uk/rfdr_area_layer.json`` lists them. The post-geography node
``uk.full.geography.rfdr`` sets the flag on each placed household from its
assigned output area. Northern Ireland has no area in the scheme. The
national line has no areas, so it leaves the engine's default of false
(María's ruling of 2026-10-28).

Everything here is pure and depends on the standard library, numpy and
pandas only, so the build tool can load this module by path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "KIND",
    "METHOD_VERSION",
    "RFDR_AREA_LAYER_SOURCE",
    "RFDR_COLUMN",
    "RFDR_LOCALISATION_MODES",
    "SCHEMA_VERSION",
    "SYSTEMS",
    "RFDRAreaLayer",
    "RFDRPostcodeDefinition",
    "area_flags",
    "flag_output_areas",
    "load_rfdr_area_layer",
    "postcode_in_scheme",
]

KIND = "uk_rfdr_area_layer"
SCHEMA_VERSION = 1
METHOD_VERSION = "uk-rfdr-area-layer-v1"
#: Graph source name of the committed layer bytes.
RFDR_AREA_LAYER_SOURCE = "uk_rfdr_area_layer"
#: ``area`` sets each placed Great Britain household's flag from its assigned
#: output area; ``none`` leaves the engine's default on every clone.
RFDR_LOCALISATION_MODES = ("none", "area")
#: The policyengine-uk household input the node writes.
RFDR_COLUMN = "in_rural_fuel_duty_relief_area"
#: The support systems the layer covers, by the output-area code they read.
SYSTEMS: Mapping[str, str] = {
    "uk_ew_output_area_2021": r"[EW]00\d{6}",
    "uk_scotland_output_area_2022": r"S00\d{6}",
}
#: Northern Ireland's support system: no area of it is in the scheme.
NI_SYSTEM = "uk_ni_data_zone_2021"
#: An output area is in the scheme when more than this share of its postcodes
#: are.
MAJORITY_SHARE = 0.5
_DISTRICT = re.compile(r"[A-Z]{1,2}\d[A-Z\d]?")
_SECTOR = re.compile(r"[A-Z]{1,2}\d[A-Z\d]? \d")
_UNIT = re.compile(r"[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}")


def _strings(values: object, *, what: str) -> frozenset[str]:
    """A definition list, or a mapping of named lists flattened (its parts)."""

    if values is None:
        return frozenset()
    if isinstance(values, Mapping):
        return frozenset(str(v) for part in values.values() for v in part)
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise ValueError(f"RFDR definition {what} must be a list or named lists.")
    return frozenset(str(value) for value in values)


@dataclass(frozen=True)
class RFDRPostcodeDefinition:
    """The scheme's area as postcode districts, sectors and units.

    ``districts`` are outward codes (``EX35``); ``sectors`` are an outward
    code, a space and the inward code's first digit (``PA34 4``); ``units``
    are full postcodes (``PA34 4RU``), for an island that shares its sector
    with the mainland. ``excluded_units`` are postcodes of an included
    district or sector that lie outside the scheme.
    """

    districts: frozenset[str]
    sectors: frozenset[str]
    units: frozenset[str] = frozenset()
    excluded_units: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(cls, payload: Mapping) -> RFDRPostcodeDefinition:
        districts = _strings(payload.get("districts"), what="districts")
        sectors = _strings(payload.get("sectors"), what="sectors")
        units = _strings(payload.get("units"), what="units")
        excluded = _strings(payload.get("excluded_units"), what="excluded_units")
        bad = sorted(d for d in districts if not _DISTRICT.fullmatch(d))
        bad += sorted(s for s in sectors if not _SECTOR.fullmatch(s))
        bad += sorted(u for u in units | excluded if not _UNIT.fullmatch(u))
        if bad:
            raise ValueError(f"RFDR definition has malformed postcode units: {bad}.")
        if not districts:
            raise ValueError("RFDR definition names no postcode district.")
        overlap = sorted(s for s in sectors if s.split(" ")[0] in districts)
        overlap += sorted(u for u in units if _covered(u, districts, sectors))
        if overlap:
            raise ValueError(
                f"RFDR definition lists units its districts already hold: {overlap}."
            )
        stray = sorted(u for u in excluded if not _covered(u, districts, sectors))
        if stray or units & excluded:
            raise ValueError(
                "RFDR definition excludes units no included district or sector "
                f"holds: {stray or sorted(units & excluded)}."
            )
        return cls(
            districts=districts, sectors=sectors, units=units, excluded_units=excluded
        )


def _covered(unit: str, districts: frozenset[str], sectors: frozenset[str]) -> bool:
    outward, inward = unit.split(" ")
    return outward in districts or f"{outward} {inward[0]}" in sectors


def _normalised(postcodes: Iterable[str]) -> list[str]:
    """Each postcode as outward code, a space and inward code (``PA34 4RU``)."""

    codes = [str(code).replace(" ", "").upper() for code in postcodes]
    return [f"{code[:-3]} {code[-3:]}" if len(code) >= 5 else code for code in codes]


def postcode_in_scheme(
    postcodes: Iterable[str], definition: RFDRPostcodeDefinition
) -> np.ndarray:
    """Whether each postcode lies in a declared district, sector or unit."""

    inside = []
    for unit in _normalised(postcodes):
        if unit in definition.excluded_units:
            inside.append(False)
        elif unit in definition.units:
            inside.append(True)
        elif " " in unit:
            inside.append(_covered(unit, definition.districts, definition.sectors))
        else:
            inside.append(False)
    return np.asarray(inside, dtype=bool)


def flag_output_areas(
    postcodes: pd.DataFrame, definition: RFDRPostcodeDefinition
) -> tuple[dict[str, list[str]], dict[str, object]]:
    """The output areas whose postcodes mostly lie in the scheme, per system.

    ``postcodes`` carries ``postcode`` and ``oa21cd`` (the ONS lookup's
    best-fit 2021 output area, Scotland's 2022 output areas under the same
    column). Returns the flagged codes per support system and a receipt of
    postcode and area counts, with the areas the scheme only partly covers.
    """

    frame = pd.DataFrame(
        {
            "oa": postcodes["oa21cd"].astype("string").str.strip(),
            "inside": postcode_in_scheme(postcodes["postcode"], definition),
        }
    ).dropna(subset=["oa"])
    shares = frame.groupby("oa", sort=True)["inside"].mean()
    flagged: dict[str, list[str]] = {}
    receipt: dict[str, object] = {
        "majority_share": MAJORITY_SHARE,
        "postcodes_inside": int(frame["inside"].sum()),
        "by_system": {},
    }
    for system, pattern in SYSTEMS.items():
        mask = shares.index.to_series().str.fullmatch(pattern).to_numpy(dtype=bool)
        system_shares = shares[mask]
        inside = system_shares[system_shares > MAJORITY_SHARE]
        flagged[system] = sorted(str(code) for code in inside.index)
        receipt["by_system"][system] = {
            "areas_flagged": len(inside),
            "areas_partly_inside": int(
                ((system_shares > 0) & (system_shares < 1)).sum()
            ),
        }
    return flagged, receipt


@dataclass(frozen=True)
class RFDRAreaLayer:
    """The committed layer: the flagged output areas per support system."""

    method: str
    areas: Mapping[str, frozenset[str]]


def load_rfdr_area_layer(payload: bytes | str | Mapping) -> RFDRAreaLayer:
    """Parse and check the committed layer."""

    if isinstance(payload, bytes | str):
        payload = json.loads(payload)
    if payload.get("kind") != KIND or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Not a UK Rural Fuel Duty Relief area layer.")
    if payload.get("method") != METHOD_VERSION:
        raise ValueError(
            f"RFDR area layer method {payload.get('method')!r} is not "
            f"{METHOD_VERSION!r}."
        )
    declared = payload.get("areas")
    if not isinstance(declared, Mapping) or set(declared) != set(SYSTEMS):
        raise ValueError(
            f"RFDR area layer must list areas for exactly {sorted(SYSTEMS)}."
        )
    areas: dict[str, frozenset[str]] = {}
    for system, pattern in SYSTEMS.items():
        codes = [str(code) for code in declared[system]]
        bad = [code for code in codes if not re.fullmatch(pattern, code)]
        if bad:
            raise ValueError(
                f"RFDR area layer {system} carries malformed codes {bad[:3]}."
            )
        if len(set(codes)) != len(codes):
            raise ValueError(f"RFDR area layer {system} repeats a code.")
        areas[system] = frozenset(codes)
    if not any(areas.values()):
        raise ValueError("RFDR area layer flags no output area.")
    return RFDRAreaLayer(method=str(payload["method"]), areas=areas)


def area_flags(
    *, layer: RFDRAreaLayer, area_system: Iterable[object], area_code: Iterable[object]
) -> tuple[np.ndarray, dict[str, int]]:
    """Each household's flag from its assigned output area.

    A Great Britain household is in the scheme when the layer lists its area;
    a Northern Ireland household never is. Any other support system refuses,
    since the layer cannot say where it is.
    """

    systems = np.asarray(list(area_system), dtype=object)
    codes = np.asarray(list(area_code), dtype=object)
    if len(systems) != len(codes):
        raise ValueError("area_system and area_code must align.")
    unknown = sorted({str(system) for system in systems} - {*SYSTEMS, NI_SYSTEM})
    if unknown:
        raise ValueError(f"RFDR flags read no support system {unknown}.")
    flags = np.zeros(len(codes), dtype=bool)
    counts: dict[str, int] = {}
    for system, flagged in layer.areas.items():
        rows = systems == system
        flags[rows] = np.asarray([str(code) in flagged for code in codes[rows]])
        counts[system] = int(rows.sum())
    counts[NI_SYSTEM] = int((systems == NI_SYSTEM).sum())
    counts["flagged"] = int(flags.sum())
    return flags, counts
