"""The area-to-BRMA layer and each placed household's BRMA (microcosm#1262).

The spine's ``frs_brma`` stage draws a household's Broad Rental Market Area
within its FRS region before any household has an area, so every geographic
clone of a household used to carry the same BRMA wherever it was placed, and
its Local Housing Allowance rate described a different place from its local
authority and constituency.

``uk/brma_area_layer.json`` places every 2021 LSOA of England and Wales and
every 2022 output area of Scotland in one BRMA, by population-weighted
centroid in the published BRMA polygons (Scotland corrected by Rent Service
Scotland's postcode lookup); ``tools/build_uk_brma_area_layer.py`` rebuilds
it from Chronicle's raw registrations. The post-geography node
``uk.full.geography.brma`` reads a placed household's BRMA off its assigned
area, as its local authority and constituency are read. Northern Ireland has
no open BRMA geography and keeps the spine's draw. The national line has no
areas and keeps the spine's draw everywhere.

Everything here is pure and depends on the standard library, numpy and
pandas only, so the build tool can load this module by path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BRMA_AREA_LAYER_SOURCE",
    "BRMA_LOCALISATION_MODES",
    "GRAINS",
    "KIND",
    "METHOD_VERSION",
    "NI_SYSTEM",
    "SCHEMA_VERSION",
    "BRMAAreaLayer",
    "area_brmas",
    "english_welsh_brma_key",
    "list_brma_key",
    "load_brma_area_layer",
    "roster_sha256",
    "scottish_brma_key",
]

KIND = "uk_brma_area_layer"
SCHEMA_VERSION = 1
METHOD_VERSION = "uk-brma-area-layer-v1"
#: Graph source name of the committed layer bytes.
BRMA_AREA_LAYER_SOURCE = "uk_brma_area_layer"
#: ``area`` reads each placed Great Britain household's BRMA off its assigned
#: area; ``none`` leaves the spine's regional draw on every clone.
BRMA_LOCALISATION_MODES = ("none", "area")
NI_SYSTEM = "uk_ni_data_zone_2021"


@dataclass(frozen=True)
class GrainSpec:
    """One grain of the layer: the support system it serves and its key."""

    system: str
    key: str
    code_pattern: str


#: England and Wales are keyed by 2021 LSOA (each output area takes its
#: LSOA's BRMA, as policyengine-uk-data#516 places them); Scotland by 2022
#: output area.
GRAINS: Mapping[str, GrainSpec] = {
    "ew_lsoa21": GrainSpec(
        system="uk_ew_output_area_2021",
        key="lsoa_code",
        code_pattern=r"[EW]\d{8}",
    ),
    "scotland_oa22": GrainSpec(
        system="uk_scotland_output_area_2022",
        key="oa_code",
        code_pattern=r"S\d{8}",
    ),
}
_GRAIN_BY_SYSTEM = {spec.system: grain for grain, spec in GRAINS.items()}
_BRMA_NAME = re.compile(r"[A-Z0-9]+(?:_[A-Z0-9]+)*")
_SCOTTISH_CODE = re.compile(r"S33\d{6}")

#: Spellings in the lists of rents that differ from the engine's names.
_LIST_ALIASES = {
    "AYRSHIRE": "AYRSHIRES",
    "HIGHLANDS_AND_ISLANDS": "HIGHLAND_AND_ISLANDS",
    "RENFREWSHIRE_INVERCLYDE": "RENFREWSHIRE_AND_INVERCLYDE",
}


# ----------------------------------------------------------------------------
# Names


def english_welsh_brma_key(name: str) -> str:
    """The engine name of a VOA or Rent Officers Wales BRMA name.

    The rule policyengine-uk-data#516 applies to the same layers.
    """

    return re.sub(r"\W+", "_", str(name).replace("&", "").upper()).strip("_")


def scottish_brma_key(name: str) -> str:
    """The engine name of a Scottish Government or Rent Service Scotland name."""

    key = re.sub(r"\s*/\s*", " AND ", str(name).strip()).upper().replace(" ", "_")
    return "AYRSHIRES" if key == "AYRSHIRE" else key


def list_brma_key(name: str) -> str:
    """The engine name of a BRMA as the lists of rents spell it."""

    key = re.sub(r"[^A-Z0-9]+", "_", str(name).replace("&", " ").upper()).strip("_")
    return _LIST_ALIASES.get(key, key)


def roster_sha256(codes: Iterable[str]) -> str:
    """Digest of a roster of area codes, independent of their order."""

    text = "\n".join(sorted(str(code) for code in codes))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# The resource


@dataclass(frozen=True)
class BRMAAreaLayer:
    """The committed layer, keyed for lookup."""

    #: grain -> area code -> BRMA engine name.
    brma_by_area: Mapping[str, Mapping[str, str]]
    #: grain -> area code -> FRS region of the area.
    region_by_area: Mapping[str, Mapping[str, str]]
    #: Scottish BRMA engine name -> ONS code (``S33000001``-``S33000018``).
    scotland_codes: Mapping[str, str]
    method: str


def load_brma_area_layer(payload: bytes) -> BRMAAreaLayer:
    """Load and validate ``uk/brma_area_layer.json``."""

    data = json.loads(payload.decode("utf-8"))
    if data.get("kind") != KIND or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("The BRMA area layer is not a schema-1 layer resource.")
    if data.get("method") != METHOD_VERSION:
        raise ValueError(
            f"The BRMA area layer method is {data.get('method')!r}; "
            f"this build reads {METHOD_VERSION!r}."
        )
    grains = data.get("grains") or {}
    if set(grains) != set(GRAINS):
        raise ValueError(f"The BRMA area layer must carry the grains {sorted(GRAINS)}.")
    layout = data.get("brmas") or {}
    if set(layout) != set(GRAINS):
        raise ValueError("The BRMA area layer's BRMA lists do not match its grains.")
    brma_by_area: dict[str, dict[str, str]] = {}
    region_by_area: dict[str, dict[str, str]] = {}
    for grain, spec in GRAINS.items():
        declared = grains[grain]
        if (
            declared.get("area_system") != spec.system
            or declared.get("key") != spec.key
        ):
            raise ValueError(f"The BRMA area layer's {grain} grain is mislabelled.")
        pattern = re.compile(spec.code_pattern)
        brmas: dict[str, str] = {}
        regions: dict[str, str] = {}
        for region, by_brma in layout[grain].items():
            if not _BRMA_NAME.fullmatch(str(region)):
                raise ValueError(f"The BRMA area layer names an odd region {region!r}.")
            for brma, codes in by_brma.items():
                if not _BRMA_NAME.fullmatch(str(brma)):
                    raise ValueError(f"The BRMA area layer names an odd BRMA {brma!r}.")
                if not codes:
                    raise ValueError(f"BRMA {brma} in {region} lists no areas.")
                for code in codes:
                    code = str(code)
                    if not pattern.fullmatch(code):
                        raise ValueError(f"{code!r} is not a {grain} area code.")
                    if code in brmas:
                        raise ValueError(f"Area {code} sits in two BRMAs.")
                    brmas[code] = str(brma)
                    regions[code] = str(region)
        if len(brmas) != int(declared.get("areas", -1)):
            raise ValueError(
                f"The {grain} grain declares {declared.get('areas')} areas and lists "
                f"{len(brmas)}."
            )
        if roster_sha256(brmas) != declared.get("roster_sha256"):
            raise ValueError(f"The {grain} roster digest does not match its areas.")
        brma_by_area[grain] = brmas
        region_by_area[grain] = regions
    scotland_codes = {
        str(brma): str(code)
        for brma, code in (data.get("scotland_brma_codes") or {}).items()
    }
    scottish_brmas = set(brma_by_area["scotland_oa22"].values())
    if set(scotland_codes) != scottish_brmas:
        raise ValueError("Every Scottish BRMA needs exactly one ONS code.")
    if len(set(scotland_codes.values())) != len(scotland_codes) or not all(
        _SCOTTISH_CODE.fullmatch(code) for code in scotland_codes.values()
    ):
        raise ValueError("The Scottish BRMA codes must be distinct S33 codes.")
    return BRMAAreaLayer(
        brma_by_area=brma_by_area,
        region_by_area=region_by_area,
        scotland_codes=scotland_codes,
        method=str(data["method"]),
    )


# ----------------------------------------------------------------------------
# The lookup


def area_brmas(
    *,
    layer: BRMAAreaLayer,
    area_system: np.ndarray,
    lsoa_code: np.ndarray,
    oa_code: np.ndarray,
    region: np.ndarray,
    brma: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Each household's BRMA read off its assigned area.

    Great Britain rows take the layer's BRMA for their LSOA (England and
    Wales) or output area (Scotland); Northern Ireland rows keep ``brma``.
    Fails closed on an unknown support system, an area the layer does not
    hold, or an area the layer places in a different FRS region from the
    household's. Returns the BRMAs and exact row counts by grain; callers
    round counts before any receipt.
    """

    area_system = np.asarray(area_system).astype(str)
    lsoa_code = np.asarray(lsoa_code).astype(object)
    oa_code = np.asarray(oa_code).astype(object)
    region = np.asarray(region).astype(str)
    current = np.asarray(brma).astype(object)
    if not (
        len(area_system)
        == len(lsoa_code)
        == len(oa_code)
        == len(region)
        == len(current)
    ):
        raise ValueError("area_brmas inputs must align.")
    unknown = sorted(set(area_system) - set(_GRAIN_BY_SYSTEM) - {NI_SYSTEM})
    if unknown:
        raise ValueError(f"No BRMA rule for the support systems {unknown}.")
    out = current.copy()
    counts: dict[str, int] = {"northern_ireland": int((area_system == NI_SYSTEM).sum())}
    for grain, spec in GRAINS.items():
        rows = np.flatnonzero(area_system == spec.system)
        counts[grain] = int(len(rows))
        if not len(rows):
            continue
        codes = (lsoa_code if spec.key == "lsoa_code" else oa_code)[rows]
        lookup = layer.brma_by_area[grain]
        regions = layer.region_by_area[grain]
        missing = sorted({str(code) for code in codes if str(code) not in lookup})
        if missing:
            raise ValueError(
                f"{len(missing)} {spec.key} values have no BRMA in the layer, "
                f"e.g. {missing[:3]}."
            )
        layer_region = np.asarray([regions[str(code)] for code in codes], dtype=object)
        if np.any(layer_region != region[rows]):
            raise ValueError(
                f"Households in {grain} sit in an area the layer places in another "
                "FRS region."
            )
        out[rows] = [lookup[str(code)] for code in codes]
    counts["changed"] = int(np.sum(out != current))
    return out, counts
