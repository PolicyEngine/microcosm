"""Toy UK conditioned atomic-area supports over public area codes (microcosm#931).

The clone-first spine's area stages draw each household's area from the three
conditioned supports and then read committed resources keyed by the codes
the draw gives it: the local-authority stage the names roster, the BRMA stage
the area-to-BRMA layer, the fuel-relief stage the RFDR layer, the private-rent
stage the BRMAs' lists of rents and the home-value stage the constituency
knots. These toy supports carry, for all three systems, codes those resources
hold, so every stage resolves against the committed resources themselves:

- each toy area's local authority is on ``local_authority_names.json``;
- its England-and-Wales LSOA, or its Scottish output area, is on
  ``brma_area_layer.json`` in the area's own FRS region, under a BRMA that
  ``brma_rent_quantiles.json`` lists rents for;
- its England-and-Wales constituency has knots in
  ``high_value_home_value_knots.json``;
- two toy areas are output areas ``rfdr_area_layer.json`` flags: one of the
  Hawes output areas in Yorkshire, and one Highlands-and-Islands output area
  in Scotland. Every household drawn into their cells lands in them.

The other codes (England and Wales's other output areas, MSOAs, wards,
ITL3s, Scottish zones and constituencies, Northern Ireland's zones, electoral
areas and constituencies) are well-formed codes of the area's nation. Which codes sit
together in one toy area is chosen for the tests, not read from the ONS
lookups, and every count, decile and band weight is invented.

The fuel-relief stage refuses a layer that flags an output area outside its
system's support. So, unless a test asks for the toy areas alone, each Great
Britain support also carries every other output area the committed RFDR
layer flags, as a filler with no households, never drawn, on its system's
flagged toy area's codes.

:data:`TOY_HOUSEHOLDS` are invented households, each with the fallback level
the toy supports give it; :func:`toy_spine_frame` builds them into a frame of
``K`` geographic clones keyed as the spine keys them, through the spine's own
expansion and identity steps. :func:`with_invented_support_channel` stands in
for spi_support_channel and spi_income_band_donors, copying the households
:func:`invented_copy_picks` names at the support channel's two ordinals, so
the SPI-copy area stage has copies to re-place, in the engine-free tests and
in the engine tier's three-system test alike.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import AtomicSupport, decode_atomic_support
from microcosm.build.uk_runtime.atomic_area_support import (
    BAND_WEIGHT_COLUMNS,
    BAND_WEIGHT_SCALE,
    IMD_DECILE_COLUMN,
    SYSTEMS,
    assemble_uk_atomic_area_support,
    assemble_uk_conditioned_area_support,
)
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE
from microcosm.frame import Frame
from test_support.microcosm_build.uk_clone_first_frames import base_id_tables

__all__ = [
    "EW",
    "NI",
    "SCOT",
    "TOY_AREAS",
    "TOY_FLAGGED_AREAS",
    "TOY_HOUSEHOLDS",
    "ToyArea",
    "ToyHousehold",
    "committed_rfdr_areas",
    "invented_copy_picks",
    "toy_conditioned_support_payloads",
    "toy_conditioned_supports",
    "toy_household_table",
    "toy_spine_frame",
    "with_invented_support_channel",
]

EW, SCOT, NI = SYSTEMS


@dataclass(frozen=True)
class ToyArea:
    """One toy area of a support: its codes, its decile and its households.

    ``bands`` gives the households in each council-tax band, which sum to the
    area's households; Northern Ireland's areas give ``households`` instead.
    """

    system: str
    region: str
    oa_code: str
    lsoa_code: str
    msoa_code: str
    local_authority_code: str
    ward_code: str
    constituency_code: str
    itl3_code: str
    imd_decile: str
    bands: Mapping[str, int] = field(default_factory=dict)
    households: int = 0

    def household_count(self) -> int:
        return sum(dict(self.bands).values()) if self.bands else int(self.households)


def _ew(region, oa, lsoa, msoa, la, ward, pcon, itl3, decile, **bands) -> ToyArea:
    return ToyArea(EW, region, oa, lsoa, msoa, la, ward, pcon, itl3, decile, bands)


def _scot(oa, lsoa, msoa, la, ward, pcon, itl3, decile, **bands) -> ToyArea:
    return ToyArea(
        SCOT, "SCOTLAND", oa, lsoa, msoa, la, ward, pcon, itl3, decile, bands
    )


def _ni(zone, super_zone, district, dea, pcon, itl3, decile, households) -> ToyArea:
    return ToyArea(
        NI,
        "NORTHERN_IRELAND",
        zone,
        zone,
        super_zone,
        district,
        dea,
        pcon,
        itl3,
        decile,
        households=households,
    )


#: The toy areas. Each region has household mass; London's cells give every
#: fallback level a household (decile 1 holds no band H, decile 9 no area),
#: Wales holds the only band I, and the North East no band G or H at all.
# fmt: off
TOY_AREAS: tuple[ToyArea, ...] = (
    _ew("NORTH_EAST", "E00042101", "E01008162", "E02001700", "E08000021", "E05001101",
        "E14001063", "TLC22", "3", A=20, B=8, C=4, D=2),
    _ew("NORTH_EAST", "E00042102", "E01008163", "E02001700", "E08000021", "E05001101",
        "E14001063", "TLC22", "7", A=4, B=6, C=8, D=6, E=4, F=2),
    _ew("NORTH_WEST", "E00025101", "E01004969", "E02001001", "E08000003", "E05000801",
        "E14001064", "TLD33", "1", A=15, B=10, C=5),
    _ew("YORKSHIRE", "E00057101", "E01010646", "E02002301", "E08000035", "E05001401",
        "E14001065", "TLE42", "5", B=10, C=10, D=10),
    # A Hawes output area, which the RFDR layer flags.
    _ew("YORKSHIRE", "E00141336", "E01027587", "E02005701", "E06000065", "E05006001",
        "E14001066", "TLE22", "6", C=2, D=3, E=3, F=2),
    _ew("EAST_MIDLANDS", "E00069101", "E01013810", "E02002801", "E06000018", "E05001701",
        "E14001067", "TLF14", "3", A=10, B=10, C=5, D=5),
    _ew("WEST_MIDLANDS", "E00045101", "E01008881", "E02001801", "E08000025", "E05001201",
        "E14001068", "TLG31", "2", A=10, B=15, C=5),
    _ew("EAST_OF_ENGLAND", "E00092101", "E01017943", "E02003701", "E07000008",
        "E05002701", "E14001069", "TLH12", "9", C=5, D=10, E=10, F=5),
    _ew("LONDON", "E00000001", "E01000001", "E02000001", "E09000001", "E05009301",
        "E14001070", "TLI31", "1", A=5, B=10, C=10, D=10, E=5),
    _ew("LONDON", "E00000003", "E01000002", "E02000001", "E09000001", "E05009302",
        "E14001070", "TLI31", "1", C=5, D=10, E=10, F=5),
    _ew("LONDON", "E00000005", "E01000003", "E02000001", "E09000033", "E05013801",
        "E14001071", "TLI32", "4", E=5, F=5, G=10, H=20),
    _ew("LONDON", "E00009001", "E01001722", "E02000801", "E09000030", "E05009401",
        "E14001072", "TLI42", "2", A=10, B=20, C=10),
    _ew("SOUTH_EAST", "E00085101", "E01016849", "E02003501", "E06000043", "E05002401",
        "E14001073", "TLJ21", "6", B=5, C=10, D=10, E=5),
    _ew("SOUTH_WEST", "E00073101", "E01014395", "E02003001", "E06000023", "E05001901",
        "E14001074", "TLK11", "5", A=10, B=10, C=10),
    _ew("WALES", "W00009001", "W01001694", "W02000401", "W06000015", "W05001001",
        "W07000081", "TLL22", "2", A=5, B=5, C=10, D=5, I=5),
    _ew("WALES", "W00009002", "W01001695", "W02000401", "W06000015", "W05001002",
        "W07000082", "TLL22", "8", D=10, E=10, F=5, G=3, H=2),
    _scot("S00141165", "S01008417", "S02001575", "S12000036", "S13002922", "S14000021",
          "TLM75", "2", A=10, B=10, C=10),
    _scot("S00141166", "S01008418", "S02001575", "S12000036", "S13002923", "S14000021",
          "TLM75", "8", D=10, E=10, F=5, G=3, H=2),
    # A Highlands-and-Islands output area, which the RFDR layer flags.
    _scot("S00162038", "S01010001", "S02001900", "S12000013", "S13002608", "S14000022",
          "TLM64", "4", A=3, B=4, C=3),
    _ni("N20000001", "N21000001", "N09000003", "N10000301", "N05000001", "TLN06", "1", 30),
    _ni("N20000002", "N21000001", "N09000003", "N10000301", "N05000001", "TLN06", "1", 20),
    _ni("N20000100", "N21000020", "N09000011", "N10001101", "N05000002", "TLN09", "6", 25),
)
# fmt: on
#: The toy areas the committed RFDR layer flags, by system: the hosts of each
#: system's filler rows.
TOY_FLAGGED_AREAS: Mapping[str, str] = {EW: "E00141336", SCOT: "S00162038"}


@dataclass(frozen=True)
class ToyHousehold:
    """An invented household's draw inputs and the level the toys give it."""

    region: str
    imd_decile: int
    council_tax_band: str
    council_tax_band_observed: bool
    level: str


def _household(region, decile, band, observed, level) -> ToyHousehold:
    return ToyHousehold(region, decile, band, observed, level)


#: Invented households, one per case: L0 where the band has mass in the
#: household's decile cell, L1 where only households do (band H in London's
#: decile 1, band A in Wales's decile 8, no band observed), L2 where the decile
#: is missing or has no area and the band has mass in the region, L3 where
#: neither input holds (no band G or H in the North East). Northern Ireland
#: weighs no band: L1 or L3. Two households land in the flagged toy areas.
TOY_HOUSEHOLDS: tuple[ToyHousehold, ...] = (
    _household("LONDON", 1, "B", True, "L0"),
    _household("LONDON", 1, "H", True, "L1"),
    _household("LONDON", 0, "H", True, "L2"),
    _household("LONDON", 0, "D", False, "L3"),
    _household("LONDON", 9, "C", True, "L2"),
    _household("LONDON", 4, "G", False, "L1"),
    _household("LONDON", 2, "A", True, "L0"),
    _household("NORTH_EAST", 3, "H", True, "L1"),
    _household("NORTH_EAST", 0, "H", True, "L3"),
    _household("NORTH_EAST", 7, "E", True, "L0"),
    _household("NORTH_WEST", 1, "A", True, "L0"),
    _household("YORKSHIRE", 5, "B", True, "L0"),
    _household("YORKSHIRE", 6, "D", True, "L0"),
    _household("EAST_MIDLANDS", 3, "C", True, "L0"),
    _household("WEST_MIDLANDS", 0, "B", False, "L3"),
    _household("EAST_OF_ENGLAND", 9, "D", True, "L0"),
    _household("SOUTH_EAST", 6, "E", True, "L0"),
    _household("SOUTH_WEST", 5, "C", True, "L0"),
    _household("WALES", 2, "I", True, "L0"),
    _household("WALES", 8, "A", True, "L1"),
    _household("SCOTLAND", 2, "A", True, "L0"),
    _household("SCOTLAND", 4, "C", True, "L0"),
    _household("SCOTLAND", 0, "H", True, "L2"),
    _household("SCOTLAND", 3, "B", False, "L3"),
    _household("NORTHERN_IRELAND", 1, "D", False, "L1"),
    _household("NORTHERN_IRELAND", 3, "D", False, "L3"),
    _household("NORTHERN_IRELAND", 0, "D", False, "L3"),
    _household("NORTHERN_IRELAND", 6, "D", False, "L1"),
)

_CODE_COLUMNS = (
    "oa_code",
    "constituency_code",
    "region_code",
    "lsoa_code",
    "msoa_code",
    "local_authority_code",
    "ward_code",
    "itl3_code",
)


def committed_rfdr_areas() -> dict[str, tuple[str, ...]]:
    """The output areas the committed RFDR layer flags, by system, sorted."""

    payload = json.loads(
        files("microcosm.build.uk").joinpath("rfdr_area_layer.json").read_bytes()
    )
    return {system: tuple(sorted(codes)) for system, codes in payload["areas"].items()}


def _filler(host: ToyArea, oa_code: str) -> ToyArea:
    """A flagged output area with no households, on its host's other codes."""

    return ToyArea(
        host.system,
        host.region,
        oa_code,
        host.lsoa_code,
        host.msoa_code,
        host.local_authority_code,
        host.ward_code,
        host.constituency_code,
        host.itl3_code,
        host.imd_decile,
        households=0,
    )


def _areas(system: str, *, rfdr_roster: bool) -> list[ToyArea]:
    areas = [area for area in TOY_AREAS if area.system == system]
    if rfdr_roster and system in TOY_FLAGGED_AREAS:
        (host,) = [area for area in areas if area.oa_code == TOY_FLAGGED_AREAS[system]]
        held = {area.oa_code for area in areas}
        areas += [
            _filler(host, code)
            for code in committed_rfdr_areas()[system]
            if code not in held
        ]
    return areas


def _base_metadata(system: str) -> dict[str, dict[str, str]]:
    vintage = "2022_census" if system == SCOT else "2021_census"
    metadata = {}
    for column in (*_CODE_COLUMNS, "population", "households"):
        if column in {"population", "households"}:
            metadata[column] = {
                "kind": "weight",
                "source": f"toy-{system}-{column}",
                "basis": f"toy {column}",
            }
            continue
        metadata[column] = {
            "kind": "code",
            "source": f"toy-{system}-{column}",
            "vintage": vintage,
            "relation": "exact"
            if column in {"oa_code", "lsoa_code", "msoa_code", "region_code"}
            else "best_fit",
        }
    metadata["constituency_code"]["vintage"] = "2024_pcon"
    if system == NI:
        metadata["constituency_code"]["relation"] = "official_tabulation"
    return metadata


def _conditioning_metadata(system: str) -> dict[str, dict[str, str]]:
    metadata = {
        IMD_DECILE_COLUMN: {
            "kind": "code",
            "source": f"toy-{system}-imd-decile",
            "vintage": "toy_imd",
            "relation": "inferred_modal",
        }
    }
    for column in BAND_WEIGHT_COLUMNS[system]:
        metadata[column] = {
            "kind": "weight",
            "source": f"toy-{system}-{column}",
            "basis": "toy households in the band x1e6",
        }
    return metadata


def _payload(system: str, *, rfdr_roster: bool) -> bytes:
    areas = _areas(system, rfdr_roster=rfdr_roster)
    households = [area.household_count() for area in areas]
    rows = {
        "oa_code": [area.oa_code for area in areas],
        "constituency_code": [area.constituency_code for area in areas],
        "region_code": [FRS_REGION_TO_REGION_CODE[area.region] for area in areas],
        "lsoa_code": [area.lsoa_code for area in areas],
        "msoa_code": [area.msoa_code for area in areas],
        "local_authority_code": [area.local_authority_code for area in areas],
        "ward_code": [area.ward_code for area in areas],
        "itl3_code": [area.itl3_code for area in areas],
    }
    arrays = {column: np.asarray(values, dtype="U") for column, values in rows.items()}
    arrays["households"] = np.asarray(households, dtype=np.int64)
    arrays["population"] = np.asarray(
        [2 * count + 1 for count in households], dtype=np.int64
    )
    base = assemble_uk_atomic_area_support(
        system=system, arrays=arrays, column_metadata=_base_metadata(system)
    )
    # The adapter sorts its rows by area; the conditioning columns follow.
    order = {
        code: position
        for position, code in enumerate(
            decode_atomic_support(base).arrays["area"].tolist()
        )
    }
    ranked = sorted(areas, key=lambda area: order[area.oa_code])
    band_weights = {
        column: np.asarray(
            [
                dict(area.bands).get(column.removeprefix("households_band_").upper(), 0)
                * BAND_WEIGHT_SCALE
                for area in ranked
            ],
            dtype=np.int64,
        )
        for column in BAND_WEIGHT_COLUMNS[system]
    }
    return assemble_uk_conditioned_area_support(
        base=base,
        imd_decile=np.asarray([area.imd_decile for area in ranked], dtype="U"),
        band_weights=band_weights,
        column_metadata=_conditioning_metadata(system),
    )


@functools.cache
def _payloads(rfdr_roster: bool) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (system, _payload(system, rfdr_roster=rfdr_roster)) for system in SYSTEMS
    )


def toy_conditioned_support_payloads(*, rfdr_roster: bool = True) -> dict[str, bytes]:
    """The three toy conditioned supports' canonical bytes, keyed by system.

    With ``rfdr_roster`` (the default), each Great Britain support also holds
    every other output area the committed RFDR layer flags, as a filler with
    no households; without it, the toy areas alone.
    """

    return dict(_payloads(bool(rfdr_roster)))


def toy_conditioned_supports(*, rfdr_roster: bool = True) -> dict[str, AtomicSupport]:
    """The toy supports, decoded, keyed by system."""

    return {
        system: decode_atomic_support(payload)
        for system, payload in toy_conditioned_support_payloads(
            rfdr_roster=rfdr_roster
        ).items()
    }


def toy_household_table(households: Sequence[ToyHousehold]) -> pd.DataFrame:
    """The draw's raw inputs for ``households``, typed as the spine's root
    types them: an object region and band, an int64 decile (0 for none) and a
    boolean band flag."""

    return pd.DataFrame(
        {
            "region": pd.Series([h.region for h in households], dtype=object),
            "imd_decile": np.asarray(
                [h.imd_decile for h in households], dtype=np.int64
            ),
            "council_tax_band": pd.Series(
                [h.council_tax_band for h in households], dtype=object
            ),
            "council_tax_band_observed": np.asarray(
                [h.council_tax_band_observed for h in households], dtype=bool
            ),
        }
    )


def toy_spine_frame(
    *, clones: int = 1, households: Sequence[ToyHousehold] = TOY_HOUSEHOLDS
) -> Frame:
    """``households`` as the spine holds them after its identity step.

    Synthetic FRS-shaped ids (``uk_clone_first_frames.base_id_tables``, one
    to four people in one or two benefit units), the households' raw draw
    inputs, invented design weights and the FRS release's survey year as the
    time period; then ``clones`` geographic clones through the spine's own
    expansion, each keyed by the spine's identity step.
    """

    from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
    from microcosm.build.uk_runtime.geographic_expand import (
        expand_uk_geographic_clones,
    )
    from microcosm.build.uk_runtime.geographic_identity import (
        key_uk_households_by_clone_path,
    )
    from microcosm.build.uk_runtime.national_frame import uk_national_frame

    tables = base_id_tables(len(households))
    household = pd.concat(
        [
            tables.household[["household_id"]].reset_index(drop=True),
            toy_household_table(households),
        ],
        axis=1,
    )
    frame = uk_national_frame(
        person=tables.person,
        benunit=tables.benunit,
        household=household,
        household_weights=np.asarray(
            [800.0 + 25.0 * position for position in range(len(households))]
        ),
        time_period=str(load_uk_frs_release().survey_year),
    )
    expanded = expand_uk_geographic_clones(frame, clones=clones).frame
    return key_uk_households_by_clone_path(expanded).frame


def invented_copy_picks(source: int, clone: int) -> tuple[int, ...]:
    """The invented copies spi_support_channel and spi_income_band_donors
    make of a household, by ordinal.

    In clone 0 every household is copied (1) and every third one is a band
    donor too (2), by its source alone, so clone 0's copies are the same at
    every clone count; a later clone copies other households, so a union of
    copied sources over clones grows with K.
    """

    if clone == 0:
        return (1, 2) if source % 3 == 0 else (1,)
    return tuple(
        ordinal
        for ordinal, picked in (
            (1, (source + clone) % 2 == 0),
            (2, (source + clone) % 3 == 0),
        )
        if picked
    )


def with_invented_support_channel(
    frame: Frame, picks: Callable[[int, int], tuple[int, ...]] = invented_copy_picks
) -> Frame:
    """``frame`` as spi_support_channel and spi_income_band_donors leave it,
    for tests alone.

    Every row is on the support channel's FRS side at ordinal 0; each
    household ``picks`` names (by its source and clone) is copied whole at
    the channel's ordinals, every id minted in the copy-id table's
    ``spi_support_channel`` field, each copy at half its parent's invented
    weight, with a mass record for the copies. No other lineage column is
    added.
    """

    from microcosm.build.uk_runtime import copy_id_table
    from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        uk_time_period,
    )
    from microcosm.build.uk_runtime.spi_support import (
        BASE_FRS_SUPPORT_CHANNEL,
        SPI_SYNTHETIC_SUPPORT_CHANNEL,
        support_channel_column,
        support_clone_index_column,
    )
    from microcosm.frame import MassChangeRecord

    channel = support_channel_column("household")
    ordinal_column = support_clone_index_column("household")
    clone_column = ladder_clone_index_column("household")
    person, benunit = frame.table("person"), frame.table("benunit")
    household = frame.table("household").copy()
    household[channel] = BASE_FRS_SUPPORT_CHANNEL
    household[ordinal_column] = np.zeros(len(household), dtype=np.int64)
    weights = frame.weights_for("household").values
    persons, benunits, households = [person], [benunit], [household]
    masses = [weights]
    for ordinal in (1, 2):
        positions = [
            position
            for position, (source, clone) in enumerate(
                zip(
                    household["source_household_id"].tolist(),
                    household[clone_column].tolist(),
                    strict=True,
                )
            )
            if ordinal in picks(int(source), int(clone))
        ]
        if not positions:
            continue
        copied = household.iloc[positions].copy()
        members = person.loc[person["person_household_id"].isin(copied["household_id"])]
        members = members.copy()
        units = benunit.loc[benunit["benunit_id"].isin(members["person_benunit_id"])]
        units = units.copy()
        for column in ("person_id", "person_household_id", "person_benunit_id"):
            members[column] = copy_id_table.mint(
                members[column], "spi_support_channel", ordinal
            )
        units["benunit_id"] = copy_id_table.mint(
            units["benunit_id"], "spi_support_channel", ordinal
        )
        copied["household_id"] = copy_id_table.mint(
            copied["household_id"], "spi_support_channel", ordinal
        )
        copied[channel] = SPI_SYNTHETIC_SUPPORT_CHANNEL
        copied[ordinal_column] = np.int64(ordinal)
        persons.append(members)
        benunits.append(units)
        households.append(copied)
        masses.append(weights[positions] * 0.5)
    values = np.concatenate(masses)
    record = MassChangeRecord(
        entity="household",
        old_total=float(weights.sum()),
        new_total=float(values.sum()),
        declared_factor=None,
        reason="invented copies standing in for the SPI support channel",
    )
    return uk_national_frame(
        person=pd.concat(persons, ignore_index=True),
        benunit=pd.concat(benunits, ignore_index=True),
        household=pd.concat(households, ignore_index=True),
        time_period=uk_time_period(frame),
        household_weights=values,
        mass_log=(*frame.mass_log, record),
    )
