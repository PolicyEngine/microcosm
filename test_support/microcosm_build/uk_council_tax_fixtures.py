"""Invented council tax on the toy conditioned supports (microcosm#931).

The council-tax localisation (G6, ``council_tax_localise``) prices each Great
Britain household's council tax at the local authority the spine drew for it,
against the expected Band D of the household's own draw cell. Its tests run
on test_support's toy conditioned supports
(``uk_conditioned_support_fixtures``), with what this module invents:

- :data:`COUNCIL_TAX_CASES`: invented households, one source each (position
  + 1). Each holds its draw inputs (a ``ToyHousehold``; the first 28 are
  ``TOY_HOUSEHOLDS``, in order), the bill provenance stage 5 gives it, its
  council tax, the reported council tax reduction its household reference
  person carries (the rebate equals it, as the root writes both), its
  adults, and the path the localisation takes its FRS rows by. Every Great
  Britain path is taken in more than one region. Wales's band I and
  Scotland's bands E to H take the statute, with and without the
  single-adult discount; a missing bill and an unknown reduction take the
  ratio where the band was not observed; a reduction sits above its bill on
  a ratio row (a missing bill, and a zero bill) and above the statute on a
  statute row; a statute row in a cell of two authorities carries a
  reduction below its statutory bill; a reduction equals its bill on one
  row; and a Northern Ireland row's reduction sits above its zero bill.
- :func:`invented_band_d`: a Band D for every Great Britain toy authority,
  in thirds of a pound, so that no value is a published one.
- :func:`invented_shell_council_tax`: the council tax and reduction an
  SPI-channel row carries, as the housing shell draws them, invented by the
  row's source, clone and support-channel ordinal alone, so that clone 0's
  are the same at every clone count.
- :func:`council_tax_spine_frame`: the cases as the spine holds them at the
  localisation's place. They are cloned and keyed by the spine's own
  expansion and identity steps (``toy_spine_frame``) and placed by its area
  draw. The support channel's stand-in (``with_invented_support_channel``)
  copies them. The copies are re-keyed and re-placed at L3 as the SPI-copy
  area stage re-keys and re-places them (:func:`with_spi_copies_placed`).
  Then they take their council tax (:func:`with_toy_council_tax`).
- :func:`council_tax_cell_sweep`: one invented household for each Great
  Britain draw cell of the toy supports, at each fallback level.

Every amount, count and Band D here is invented.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.atomic_area_conditioning import (
    IDENTITY_COLUMN,
    UK_SPINE_AREA_COLUMNS,
    UK_SPINE_ATOMIC_AREA_SEED,
    uk_conditioned_assignment_definition,
)
from microcosm.build.uk_runtime.council_tax_band_d import UKCouncilTaxBandD
from microcosm.build.uk_runtime.frs_council_tax import (
    FRS_COUNCIL_TAX_BILL_MISSING,
    FRS_COUNCIL_TAX_BILL_NOT_LEVIED,
    FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS,
    FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
    FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION,
)
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    support_channel_column,
    support_clone_index_column,
)
from microcosm.build.uk_runtime.spine_geography import (
    assign_uk_spine_areas,
    draw_uk_spine_areas,
    uk_spi_copy_keys,
)
from microcosm.frame import Frame
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    NI,
    TOY_AREAS,
    TOY_HOUSEHOLDS,
    ToyHousehold,
    invented_copy_picks,
    toy_conditioned_support_payloads,
    toy_conditioned_supports,
    toy_spine_frame,
    with_invented_support_channel,
)

__all__ = [
    "COUNCIL_TAX_CASES",
    "ToyCouncilTax",
    "council_tax_cell_sweep",
    "council_tax_spine_frame",
    "invented_band_d",
    "invented_shell_council_tax",
    "toy_area_draw_definition",
    "with_spi_copies_placed",
    "with_toy_council_tax",
]

_STRING = pd.StringDtype("python")
_CLONE = ladder_clone_index_column("household")
_CHANNEL = support_channel_column("household")
_ORDINAL = support_clone_index_column("household")
_REDUCTION = "council_tax_benefit_reported"


@dataclass(frozen=True)
class ToyCouncilTax:
    """An invented household's draw inputs and council tax.

    ``provenance`` is stage 5's ``council_tax_bill_provenance``; ``bill`` its
    ``council_tax``; ``reduction`` the reported council tax reduction its
    household reference person carries, which its ``council_tax_rebate``
    equals; ``adults`` its ``council_tax_single_adult_raw``; and ``path`` the
    path the localisation takes its FRS rows by.
    """

    household: ToyHousehold
    provenance: str
    bill: float
    reduction: float
    adults: int
    path: str


def _case(
    household: ToyHousehold,
    provenance: str,
    bill: float,
    reduction: float = 0.0,
    *,
    adults: int = 2,
    path: str = "ratio_observed_gross",
) -> ToyCouncilTax:
    return ToyCouncilTax(household, provenance, bill, reduction, adults, path)


_GROSS = FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS
_MISSING = FRS_COUNCIL_TAX_BILL_MISSING
_UNKNOWN = FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION
_NOT_LEVIED = FRS_COUNCIL_TAX_BILL_NOT_LEVIED
_T = TOY_HOUSEHOLDS

#: The invented cases, one source each (position + 1). London's region-level
#: cells, Yorkshire's and Scotland's span more than one toy authority, so
#: their ratios differ from one; a cell of one authority has a ratio of one.
# fmt: off
COUNCIL_TAX_CASES: tuple[ToyCouncilTax, ...] = (
    _case(_T[0], _GROSS, 1612.37),
    _case(_T[1], _GROSS, 2988.60),
    _case(_T[2], _MISSING, 2410.00, path="formula_missing"),
    # A missing bill whose reduction sits above it, in a region-level cell.
    _case(_T[3], _MISSING, 905.40, 1210.00, adults=1, path="ratio_missing_unbanded"),
    _case(_T[4], _GROSS, 1450.25, 362.56),
    _case(_T[5], _UNKNOWN, 2675.00, adults=3, path="ratio_unknown_reduction_unbanded"),
    # A zero bill with a reduction.
    _case(_T[6], _GROSS, 0.0, 310.44, adults=1),
    # A reduction equal to its bill.
    _case(_T[7], _GROSS, 1180.00, 1180.00),
    _case(_T[8], _GROSS, 1333.33),
    _case(_T[9], _GROSS, 2100.00),
    _case(_T[10], _GROSS, 990.00, 247.50, adults=1),
    _case(_T[11], _UNKNOWN, 1187.00, adults=1, path="formula_unknown_reduction"),
    _case(_T[12], _GROSS, 1702.50, 425.00),
    _case(_T[13], _GROSS, 1500.00),
    _case(_T[14], _MISSING, 1395.80, path="ratio_missing_unbanded"),
    _case(_T[15], _GROSS, 1900.00),
    _case(_T[16], _GROSS, 2200.00, 550.00),
    _case(_T[17], _GROSS, 1610.00),
    # Wales's band I, one adult.
    _case(_T[18], _MISSING, 3150.00, adults=1, path="formula_missing"),
    _case(_T[19], _GROSS, 1300.00),
    _case(_T[20], _GROSS, 1250.00),
    _case(_T[21], _GROSS, 1410.00, 352.50),
    _case(_T[22], _GROSS, 3500.00),
    _case(_T[23], _UNKNOWN, 1290.00, path="ratio_unknown_reduction_unbanded"),
    # Northern Ireland: a reduction above its zero bill, and three others.
    _case(_T[24], _NOT_LEVIED, 0.0, 250.00, adults=1, path="unchanged_northern_ireland"),
    _case(_T[25], _NOT_LEVIED, 0.0, path="unchanged_northern_ireland"),
    _case(_T[26], _NOT_LEVIED, 512.00, 128.00, path="unchanged_northern_ireland"),
    _case(_T[27], _NOT_LEVIED, 0.0, path="unchanged_northern_ireland"),
    # Scotland's bands E to H by the statute; band E's reduction sits above
    # its statutory bill.
    _case(ToyHousehold("SCOTLAND", 8, "E", True, "L0"), _MISSING, 1500.00, 4000.00,
          path="formula_missing"),
    _case(ToyHousehold("SCOTLAND", 8, "F", True, "L0"), _UNKNOWN, 1720.00, adults=1,
          path="formula_unknown_reduction"),
    _case(ToyHousehold("SCOTLAND", 8, "G", True, "L0"), _UNKNOWN, 2049.70,
          path="formula_unknown_reduction"),
    _case(ToyHousehold("SCOTLAND", 8, "H", True, "L0"), _MISSING, 2600.00, adults=1,
          path="formula_missing"),
    # Wales's band I, two adults.
    _case(ToyHousehold("WALES", 2, "I", True, "L0"), _UNKNOWN, 3010.00,
          path="formula_unknown_reduction"),
    _case(ToyHousehold("SCOTLAND", 0, "B", True, "L2"), _GROSS, 1340.00),
    _case(ToyHousehold("YORKSHIRE", 0, "D", False, "L3"), _GROSS, 1555.00, 388.75),
    # The statute in London's band-C cell, which spans two authorities, one
    # adult, with a reduction below the statutory bill.
    _case(ToyHousehold("LONDON", 0, "C", True, "L2"), _MISSING, 1400.00, 300.00,
          adults=1, path="formula_missing"),
)
# fmt: on


def invented_band_d() -> UKCouncilTaxBandD:
    """A Band D for every Great Britain toy authority, invented.

    Each authority, in code order, takes a whole number of pounds plus a
    third, so that no value is a published Band D. The digests read
    ``"supplied"``, as for a payload a test supplies.
    """

    authorities = sorted(
        {area.local_authority_code for area in TOY_AREAS if area.system != NI}
    )
    return UKCouncilTaxBandD(
        band_d=MappingProxyType(
            {
                code: Fraction(3 * (1_000 + 97 * position) + 1, 3)
                for position, code in enumerate(authorities)
            }
        ),
        england_wales_sha256="supplied",
        scotland_sha256="supplied",
        scotland_source_fact_feed=MappingProxyType({}),
    )


def invented_shell_council_tax(
    source: int, clone: int, ordinal: int
) -> tuple[float, float]:
    """The council tax and reduction of an SPI-channel row, invented.

    The housing shell draws them for the rows the support channel copied.
    Here they follow the row's source, clone and ordinal alone. Every third
    row has a reduction, below its bill, as the shell caps it.
    """

    bill = round(1_050.0 + 41.5 * source + 17.25 * clone + 8.0 * ordinal, 2)
    reduction = round(0.3 * bill, 2) if (source + clone + ordinal) % 3 == 0 else 0.0
    return bill, reduction


def toy_area_draw_definition() -> dict:
    """G2's definition on the toy supports, at the area draw's own seed."""

    return uk_conditioned_assignment_definition(
        toy_conditioned_support_payloads(), seed=UK_SPINE_ATOMIC_AREA_SEED
    )


def _rebuilt(frame: Frame, household: pd.DataFrame, person: pd.DataFrame) -> Frame:
    return uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


def with_spi_copies_placed(frame: Frame) -> Frame:
    """``frame`` with the support channel's copies re-keyed and re-placed.

    As the SPI-copy area stage does: each copy takes its own geography key
    over its copy path (``uk_spi_copy_keys``) and its area and ladder from
    G2's draw on those keys (``draw_uk_spine_areas``), at L3, its region by
    households. The ``local_authority``, ``brma`` and
    ``in_rural_fuel_duty_relief_area`` columns, which the localisation does
    not read, are left out.
    """

    household = frame.table("household").copy()
    copies, keys = uk_spi_copy_keys(household)
    data = household[IDENTITY_COLUMN].to_numpy(dtype=object).copy()
    data[copies] = keys
    household[IDENTITY_COLUMN] = pd.Series(data, index=household.index, dtype=_STRING)
    if copies.any():
        draw = draw_uk_spine_areas(
            household.loc[copies],
            definition=toy_area_draw_definition(),
            supports=toy_conditioned_supports(),
        )
        if not (np.asarray(draw.levels, dtype=object) == "L3").all():
            raise AssertionError("every SPI copy draws at L3")
        for column in UK_SPINE_AREA_COLUMNS:
            values = household[column].to_numpy(dtype=object).copy()
            values[copies] = draw.columns[column].to_numpy(dtype=object)
            household[column] = pd.Series(
                values, index=household.index, dtype=household[column].dtype
            )
    return _rebuilt(frame, household, frame.table("person"))


def with_toy_council_tax(
    frame: Frame, cases: Sequence[ToyCouncilTax] = COUNCIL_TAX_CASES
) -> Frame:
    """``frame`` with the cases' council tax, as stage 5 and the shell leave it.

    ``frame`` holds the cases' households (source = position + 1) with the
    support channel's lineage. Every row takes its case's provenance and
    adults. An FRS-channel row takes its case's bill and reduction. A row the
    support channel copied takes :func:`invented_shell_council_tax`. The
    rebate equals the reduction, which the household's first person carries.
    """

    household = frame.table("household").copy()
    person = frame.table("person").copy()
    sources = household["source_household_id"].to_numpy(dtype=np.int64)
    rows = [cases[int(source) - 1] for source in sources]
    channel = household[_CHANNEL].to_numpy(dtype=object)
    clones = household[_CLONE].to_numpy(dtype=np.int64)
    ordinals = household[_ORDINAL].to_numpy(dtype=np.int64)
    bill = np.empty(len(household), dtype=np.float64)
    reduction = np.empty(len(household), dtype=np.float64)
    for position, case in enumerate(rows):
        if channel[position] == BASE_FRS_SUPPORT_CHANNEL:
            bill[position], reduction[position] = case.bill, case.reduction
        else:
            bill[position], reduction[position] = invented_shell_council_tax(
                int(sources[position]), int(clones[position]), int(ordinals[position])
            )
    household[FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN] = pd.Series(
        [case.provenance for case in rows], index=household.index, dtype=_STRING
    )
    household["council_tax_single_adult_raw"] = np.asarray(
        [float(case.adults) for case in rows], dtype=np.float64
    )
    household["council_tax"] = bill
    household["council_tax_rebate"] = reduction.copy()
    first = ~person["person_household_id"].duplicated().to_numpy(dtype=bool)
    by_household = pd.Series(reduction, index=household["household_id"].to_numpy())
    carried = person["person_household_id"].map(by_household).to_numpy(dtype=float)
    person[_REDUCTION] = np.where(first, carried, 0.0).astype(np.float64)
    return _rebuilt(frame, household, person)


def council_tax_spine_frame(
    *,
    clones: int = 1,
    cases: Sequence[ToyCouncilTax] = COUNCIL_TAX_CASES,
    picks: Callable[[int, int], tuple[int, ...]] = invented_copy_picks,
) -> Frame:
    """The cases at the localisation's place in the spine, at ``clones`` clones.

    Keyed and cloned by the spine's own steps, placed by G2's draw, copied by
    the support channel's stand-in at the ordinals ``picks`` names for each
    source and clone, the copies re-keyed and re-placed as the SPI-copy area
    stage re-places them, then given their council tax.
    """

    keyed = toy_spine_frame(
        clones=clones, households=[case.household for case in cases]
    )
    placed = assign_uk_spine_areas(
        keyed,
        definition=toy_area_draw_definition(),
        supports=toy_conditioned_supports(),
    ).frame
    copied = with_spi_copies_placed(with_invented_support_channel(placed, picks))
    return with_toy_council_tax(copied, cases)


def council_tax_cell_sweep() -> tuple[ToyHousehold, ...]:
    """One invented household for each Great Britain draw cell of the toys.

    For each Great Britain region of the toy areas: one household at L3
    (neither input), one at L2 for each band with households in the region
    (no decile), and, for each decile its areas hold, one at L1 (no band)
    and one at L0 for each band with households in that decile. Each states
    the level the toy supports give it.
    """

    great_britain = [area for area in TOY_AREAS if area.system != NI]
    households: list[ToyHousehold] = []
    for region in dict.fromkeys(area.region for area in great_britain):
        areas = [area for area in great_britain if area.region == region]
        households.append(ToyHousehold(region, 0, "D", False, "L3"))
        households += [
            ToyHousehold(region, 0, band, True, "L2")
            for band in sorted(
                {band for area in areas for band, count in area.bands.items() if count}
            )
        ]
        for decile in sorted({area.imd_decile for area in areas}, key=int):
            households.append(ToyHousehold(region, int(decile), "D", False, "L1"))
            households += [
                ToyHousehold(region, int(decile), band, True, "L0")
                for band in sorted(
                    {
                        band
                        for area in areas
                        if area.imd_decile == decile
                        for band, count in area.bands.items()
                        if count
                    }
                )
            ]
    return tuple(households)
