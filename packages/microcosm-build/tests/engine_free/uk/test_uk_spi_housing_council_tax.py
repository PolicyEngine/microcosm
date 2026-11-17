"""The housing shell's council tax at each SPI copy's reference Band D (microcosm#931).

``spi_housing_council_tax`` prices the bill the housing shell
(``spi_housing_shell``) writes on the rows the SPI support channel copied,
SPI copies and income band donors, and states what the shell's stage
declares for it. Nothing imports it yet, so these tests run it alone, and
with the council-tax localisation (G6, ``council_tax_localise``) after it,
as the spine is to run the two.

The frames are test_support's invented council-tax cases on the toy
conditioned supports (``uk_council_tax_fixtures``) at K = 1 and K = 3, the
support channel's copies re-placed at L3 as the SPI-copy area stage
re-places them, and the housing shell's invented tables
(``uk_spi_housing_shell``). Each toy authority has an invented Band D, and
every amount is invented. Each reference and bill expected here is worked
again from the toy supports, the invented Band D and the statute, or read
from the localisation's own references, never from this module's.

They hold:

- the constants are the copy stage's level, the shell's channel and the
  localisation's Great Britain regions; the declaration pieces are the
  localisation's artifacts and Scottish verify operation, declarative, and
  the derive's formula names the statute and live stages;
- the rows are the support channel's copies in table order, and each Great
  Britain copy's reference is the exact expected Band D of its region by
  households: the localisation's own reference for the row, one for each
  system and region;
- the references and the bills read the region, the support channel and
  its lineage alone: they are the same with every area column, the
  geography key, the IMD decile and the band flag dropped, on the household
  columns the shell consumes, and for a copy moved within its region;
- the narrow frame is the area draw's frame of a copy at its level, less the
  key, and the shared expectation refuses one without either draw input;
- each Great Britain bill is the 2024-25 statute at its reference, nation,
  drawn band and single-adult flag, bit for bit, over every band of every
  Great Britain nation, the flag given as ``council_tax_single_adult_raw``
  or as the shell's zero-or-one flag;
- Northern Ireland's rows keep the chained draw bit for bit (María's
  ruling of 2026-11-14), zero or positive, whatever their band and flag,
  and the branch is the region's nation alone;
- the bills do not depend on row order, and amounts out of the inputs'
  order are refused;
- the checks hold where the bills are written, each reduction capped at its
  written bill and the rebate set equal to it, the shell's order, and each
  check fails on its own breach: the statute check on a Great Britain row,
  the Northern Ireland check on a Northern Ireland row, a zero of the other
  sign included, and the reduction and rebate checks on a row of each;
- with the localisation after them, each Great Britain copy's bill is the
  statute at its own authority within four units in the last place, its
  kept bill is the statute at its reference less its rebate, the
  localisation's cap never binds on a copy, the FRS rows are as the
  localisation leaves them without the bills, and Northern Ireland's rows
  are left as written;
- the bills of each wrong design are caught: the reference at the drawn
  authority, the region's unweighted mean, Northern Ireland priced by the
  statute, the chained bill kept in Great Britain and the cap at the
  localised bill;
- the evidence has no number leaf on three Band Ds, the committed one among
  them, reads the Scottish feed's source commit and never the feed, and is
  the same across frames and at K = 1 and 3;
- every refusal names columns, dtypes and declared constants only.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import FrozenInstanceError, dataclass, replace
from fractions import Fraction
from importlib.resources import files
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import microcosm.build.uk_runtime.spi_housing_council_tax as shell_council_tax
from microcosm.build.atomic_geography import validate_assignment_spec
from microcosm.build.atomic_support_expectations import draw_cell_expectations
from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import (
    _reject_executable_parameter_keys,
    _reject_incumbent_dependencies,
)
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    BAND_INPUT_COLUMN,
    DECILE_INPUT_COLUMN,
    IDENTITY_COLUMN,
    REGION_COLUMN,
    SUPPORT_CHANNEL_BRANCH,
    UK_SPINE_AREA_COLUMNS,
    UK_SPINE_LADDER_COLUMNS,
    _system_rules,
    uk_draw_frame,
    uk_draw_inputs,
    uk_draw_levels,
    uk_support_channel_copies,
)
from microcosm.build.uk_runtime.council_tax_band_d import (
    SCOTLAND_BAND_D_RESOURCE,
    UK_BAND_D_BUILD_PERIOD,
    UK_BAND_D_ENGLAND_WALES_RESOURCE,
    UK_BAND_D_FISCAL_YEAR,
    UKCouncilTaxBandD,
    check_scotland_band_d_declaration,
    load_uk_council_tax_band_d,
    uk_statutory_council_tax,
)
from microcosm.build.uk_runtime.council_tax_constants import (
    UK_COUNCIL_TAX_BANDS_BY_NATION,
    UK_NATION_BY_REGION,
)
from microcosm.build.uk_runtime.frs_spine import (
    FRS_COUNCIL_TAX_BAND_OBSERVED_COLUMN,
    FRS_IMD_DECILE_COLUMN,
)
from microcosm.build.uk_runtime.graph import _STAGE_CONSUMES
from microcosm.build.uk_runtime.household_lineage import UK_COPY_LINEAGE
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
)
from microcosm.build.uk_runtime.spi_housing_council_tax import (
    UK_SPI_COPY_DRAW_LEVEL,
    UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS,
    UK_SPI_HOUSING_COUNCIL_TAX_CHANNEL_COLUMN,
    UK_SPI_HOUSING_COUNCIL_TAX_CHECKS,
    UK_SPI_HOUSING_COUNCIL_TAX_DERIVE_KIND,
    UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE,
    UK_SPI_HOUSING_COUNCIL_TAX_FORMULA,
    UK_SPI_HOUSING_COUNCIL_TAX_GREAT_BRITAIN_REGIONS,
    UK_SPI_HOUSING_COUNCIL_TAX_METHOD,
    UK_SPI_HOUSING_COUNCIL_TAX_OUTPUT,
    UK_SPI_HOUSING_COUNCIL_TAX_RECIPIENT_CHANNEL,
    UK_SPI_HOUSING_COUNCIL_TAX_REFERENCE,
    UK_SPI_HOUSING_COUNCIL_TAX_VERIFY_KIND,
    UKSPIHousingCouncilTaxInputs,
    uk_spi_copy_draw_frame,
    uk_spi_council_tax_bills,
    uk_spi_housing_council_tax_checks,
    uk_spi_housing_council_tax_derive_parameters,
    uk_spi_housing_council_tax_evidence,
    uk_spi_housing_council_tax_inputs,
    uk_spi_housing_council_tax_verify_parameters,
    uk_spi_statutory_council_tax,
)
from microcosm.build.uk_runtime.spi_housing_shell import (
    SPI_CHANNEL,
    UK_SPI_HOUSING_SHELL_STAGE_NAME,
)
from microcosm.build.uk_runtime.spine_council_tax import (
    COUNCIL_TAX_FRS_BILL_COLUMN,
    COUNCIL_TAX_LOCALISE_STAGE_NAME,
    UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS,
    UKSpineCouncilTax,
    council_tax_localise_operation_parameters,
    localise_uk_council_tax,
    uk_council_tax_references,
)
from microcosm.build.uk_runtime.spine_geography import (
    SPI_COPY_AREAS_OPERATION_KIND,
    SPI_COPY_AREAS_STAGE_NAME,
    spi_copy_areas_operation_parameters,
)
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    EW,
    SCOT,
    toy_conditioned_support_payloads,
    toy_conditioned_supports,
)
from test_support.microcosm_build.uk_council_tax_fixtures import (
    COUNCIL_TAX_CASES,
    council_tax_spine_frame,
    invented_band_d,
    toy_area_draw_definition,
)
from test_support.microcosm_build.uk_spi_housing_shell import housing_shell_tables

_HOUSEHOLD_ID = "household_id"
_PERSON_HOUSEHOLD_ID = "person_household_id"
_SOURCE = "source_household_id"
_SYSTEM = "atomic_area_system"
_BILL = UK_SPI_HOUSING_COUNCIL_TAX_OUTPUT
_REBATE = "council_tax_rebate"
_REDUCTION = "council_tax_benefit_reported"
_BAND = "council_tax_band"
_SINGLE_ADULT = "council_tax_single_adult_raw"
_NORTHERN_IRELAND = "NORTHERN_IRELAND"
#: The invented reductions the shell is said to draw, by the recipient's
#: source: above every bill of the toys on an odd source, so that the cap
#: binds there; below every positive bill of the toys on a source of two
#: modulo four; none on the others.
_ABOVE_EVERY_BILL = 10_000.0
_BELOW_EVERY_BILL = 125.25
#: The household columns that place a household, as the pin on the shell's
#: consumes lists them: the area draw's, and the local-authority, BRMA and
#: fuel-relief stages'.
_AREA_COLUMNS = (
    *UK_SPINE_AREA_COLUMNS,
    "local_authority",
    "brma",
    "in_rural_fuel_duty_relief_area",
)
#: How far, in units in the last place, the localisation's bill on a copy may
#: lie from the statute at its own authority: the ratio, the bill and their
#: product are each rounded once.
_ULPS = 4
#: The wrong designs the tests here catch.
_WRONG_DESIGNS = (
    "reference_at_the_drawn_authority",
    "unweighted_region_mean",
    "northern_ireland_priced_by_the_statute",
    "chained_bill_kept_in_great_britain",
    "cap_at_the_localised_bill",
)


# --- The frames ---------------------------------------------------------------------


@functools.cache
def _supports() -> dict:
    return toy_conditioned_supports()


@functools.cache
def _definition() -> dict:
    return toy_area_draw_definition()


@functools.cache
def _band_d() -> UKCouncilTaxBandD:
    return invented_band_d()


@functools.cache
def _committed_band_d() -> UKCouncilTaxBandD:
    return load_uk_council_tax_band_d()


def _fed_band_d() -> UKCouncilTaxBandD:
    """The invented Band D with an invented Scottish feed: string leaves and
    an integer row count, which the evidence never copies."""

    return replace(
        _band_d(),
        scotland_source_fact_feed=MappingProxyType(
            {
                "source_commit": "invented-commit",
                "feed": "invented-feed",
                "record_set": "invented.band_d",
                "fact_row_count": 7,
            }
        ),
    )


@functools.cache
def _tables(clones: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The council-tax cases' household and person tables at ``clones``."""

    frame = council_tax_spine_frame(clones=clones)
    return frame.table("household"), frame.table("person")


@functools.cache
def _shell_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The housing shell's invented household and person tables."""

    person, _, household, _ = housing_shell_tables()
    return household, person


def _table(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return {
        "one_clone": lambda: _tables(1),
        "three_clones": lambda: _tables(3),
        "housing_shell_tables": _shell_tables,
    }[name]()


def _inputs(
    household: pd.DataFrame, band_d: UKCouncilTaxBandD | None = None
) -> UKSPIHousingCouncilTaxInputs:
    return uk_spi_housing_council_tax_inputs(
        household,
        definition=_definition(),
        supports=_supports(),
        band_d=_band_d() if band_d is None else band_d,
    )


def _recipients(household: pd.DataFrame) -> np.ndarray:
    """The rows the housing shell redraws, by its own rule: its channel."""

    channel = household[UK_SPI_HOUSING_COUNCIL_TAX_CHANNEL_COLUMN].astype(str)
    return channel.to_numpy(dtype=object) == SPI_CHANNEL


def _great_britain(household: pd.DataFrame) -> np.ndarray:
    return household[REGION_COLUMN].to_numpy(dtype=object) != _NORTHERN_IRELAND


def _on(inputs: UKSPIHousingCouncilTaxInputs, values: object) -> pd.Series:
    """``values`` as a Series on the inputs' household ids, in their order."""

    return pd.Series(np.asarray(values), index=pd.Index(list(inputs.household_ids)))


@dataclass(frozen=True)
class _Draws:
    """What the shell hands the bills on its recipients, here read from the
    table, on the inputs' household ids: the chained bill, the drawn band
    and the single-adult flag."""

    chained: pd.Series
    bands: pd.Series
    single_adult: pd.Series


def _draws(household: pd.DataFrame, inputs: UKSPIHousingCouncilTaxInputs) -> _Draws:
    rows = household.loc[_recipients(household)]
    return _Draws(
        chained=_on(inputs, rows[_BILL].to_numpy(dtype=np.float64)),
        bands=_on(inputs, rows[_BAND].to_numpy(dtype=object)),
        single_adult=_on(inputs, rows[_SINGLE_ADULT].to_numpy(dtype=np.float64)),
    )


def _bills(
    inputs: UKSPIHousingCouncilTaxInputs,
    draws: _Draws,
    *,
    chained: pd.Series | None = None,
) -> np.ndarray:
    return uk_spi_council_tax_bills(
        draws.chained if chained is None else chained,
        inputs=inputs,
        bands=draws.bands,
        single_adult=draws.single_adult,
    )


def _bits(values: object) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.int64)


def _ulps(left: float, right: float) -> int:
    return abs(
        int(np.float64(left).view(np.int64)) - int(np.float64(right).view(np.int64))
    )


def _statute(band_d: Fraction, region: str, band: object, adults: object) -> float:
    """The 2024-25 statute's bill, from the statute itself."""

    return float(
        uk_statutory_council_tax(
            band_d,
            UK_NATION_BY_REGION[str(region)],
            str(band),
            single_adult=bool(float(adults) == 1.0),
        )
    )


def _region_mean(region: str, *, by_households: bool = True) -> Fraction:
    """The mean invented Band D of the region's toy support areas with
    households: by households, the expectation of the region's L3 cell, or
    each area once."""

    band_d = _band_d().band_d
    areas = [
        (int(households), band_d[str(code)])
        for support in _supports().values()
        for frs_region, code, households in zip(
            support.arrays["frs_region"],
            support.arrays[UK_LOCAL_AUTHORITY_CODE_COLUMN],
            support.arrays["households"],
            strict=True,
        )
        if str(frs_region) == region and int(households) > 0
    ]
    if by_households:
        return sum((count * value for count, value in areas), Fraction(0)) / sum(
            count for count, _ in areas
        )
    return sum((value for _, value in areas), Fraction(0)) / len(areas)


@functools.cache
def _references(clones: int):
    """The localisation's own references on the frame's every row."""

    household, _ = _tables(clones)
    return uk_council_tax_references(
        household, definition=_definition(), supports=_supports(), band_d=_band_d()
    )


def _localise(household: pd.DataFrame, person: pd.DataFrame) -> UKSpineCouncilTax:
    return localise_uk_council_tax(
        household,
        person,
        definition=_definition(),
        supports=_supports(),
        band_d=_band_d(),
    )


# --- The shell's composition --------------------------------------------------------


@dataclass(frozen=True)
class _Written:
    """The shell's council tax on a table's recipients, composed in the order
    the shell is to compose it: the bills, each recipient's drawn reduction
    capped at its written bill, the rebate equal to it, and the reduction on
    the household's first person, its reference person. ``bills``,
    ``reduction`` and ``rebate`` are on the inputs' household ids."""

    household: pd.DataFrame
    person: pd.DataFrame
    inputs: UKSPIHousingCouncilTaxInputs
    draws: _Draws
    drawn: np.ndarray
    bills: pd.Series
    reduction: pd.Series
    rebate: pd.Series


def _drawn_reductions(household: pd.DataFrame) -> np.ndarray:
    """Each recipient's invented drawn reduction, in table order, by its
    source alone."""

    sources = household[_SOURCE].to_numpy(dtype=np.int64)[_recipients(household)]
    return np.select(
        [sources % 2 == 1, sources % 4 == 2],
        [_ABOVE_EVERY_BILL, _BELOW_EVERY_BILL],
        0.0,
    )


def _written(
    household: pd.DataFrame,
    person: pd.DataFrame,
    *,
    bills: object = None,
    bound: object = None,
    band_d: UKCouncilTaxBandD | None = None,
) -> _Written:
    """The shell's composition on ``household``'s recipients: the module's
    bills, or ``bills`` in their place; the cap at the written bill, or at
    ``bound`` in its place."""

    household, person = household.copy(), person.copy()
    inputs = _inputs(household, band_d)
    draws = _draws(household, inputs)
    written = (
        _bills(inputs, draws) if bills is None else np.asarray(bills, dtype=np.float64)
    )
    assert not len(written) or written.max() < _ABOVE_EVERY_BILL
    drawn = _drawn_reductions(household)
    capped = np.minimum(drawn, written if bound is None else bound)
    rows = _recipients(household)
    household.loc[rows, _BILL] = written
    household.loc[rows, _REBATE] = capped
    ids = household.loc[rows, _HOUSEHOLD_ID].to_numpy()
    owners = person[_PERSON_HOUSEHOLD_ID]
    first = ~owners.duplicated().to_numpy(dtype=bool)
    recipient = owners.isin(ids).to_numpy(dtype=bool)
    carried = owners.map(pd.Series(capped, index=ids)).to_numpy(dtype=np.float64)
    reduction = person[_REDUCTION].to_numpy(dtype=np.float64).copy()
    reduction[recipient] = np.where(first, carried, 0.0)[recipient]
    person[_REDUCTION] = reduction
    return _Written(
        household=household,
        person=person,
        inputs=inputs,
        draws=draws,
        drawn=drawn,
        bills=_on(inputs, written),
        reduction=_on(inputs, capped),
        rebate=_on(inputs, household.loc[rows, _REBATE].to_numpy(dtype=np.float64)),
    )


def _checks(written: _Written, **changes: pd.Series) -> dict[str, bool]:
    """The module's checks on the composition, with ``changes`` in place."""

    arguments = {
        "chained": written.draws.chained,
        "bills": written.bills,
        "bands": written.draws.bands,
        "single_adult": written.draws.single_adult,
        "reduction": written.reduction,
        "rebate": written.rebate,
    }
    arguments.update(changes)
    chained = arguments.pop("chained")
    return uk_spi_housing_council_tax_checks(
        chained, inputs=written.inputs, **arguments
    )


def _own_statute_ulps(after: UKSpineCouncilTax, household: pd.DataFrame) -> np.ndarray:
    """For each Great Britain recipient, in table order, how far its
    localised bill lies from the statute at its own authority, in units in
    the last place."""

    rows = np.flatnonzero(_recipients(household) & _great_britain(household))
    bills = after.council_tax.to_numpy(dtype=np.float64)
    return np.asarray(
        [
            _ulps(
                bills[row],
                _statute(
                    after.references.authority_band_d[row],
                    household[REGION_COLUMN].iloc[row],
                    household[_BAND].iloc[row],
                    household[_SINGLE_ADULT].iloc[row],
                ),
            )
            for row in rows.tolist()
        ],
        dtype=np.int64,
    )


# --- The constants and the declaration ----------------------------------------------


def test_the_constants_are_the_copy_stages_the_shells_and_the_localisations() -> None:
    (parameters,) = [
        parameters
        for kind, parameters in spi_copy_areas_operation_parameters()
        if kind == SPI_COPY_AREAS_OPERATION_KIND
    ]
    # The level the copy stage declares its draw at, on the branch it re-places.
    assert parameters["area_draw"]["level"] == UK_SPI_COPY_DRAW_LEVEL == "L3"
    assert parameters["copies"]["branch"] == SUPPORT_CHANNEL_BRANCH
    branch = next(
        b for b in UK_COPY_LINEAGE.branches if b.name == SUPPORT_CHANNEL_BRANCH
    )
    assert (
        (
            UK_SPI_HOUSING_COUNCIL_TAX_CHANNEL_COLUMN,
            UK_SPI_HOUSING_COUNCIL_TAX_RECIPIENT_CHANNEL,
        )
        == (branch.marker, branch.copy)
        == ("household_support_channel", SPI_CHANNEL)
    )
    # The localisation's Great Britain: every region but those it leaves.
    assert UK_SPI_HOUSING_COUNCIL_TAX_GREAT_BRITAIN_REGIONS == tuple(
        sorted(set(UK_NATION_BY_REGION) - set(UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS))
    )
    assert UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS == (_NORTHERN_IRELAND,)
    assert UK_SPI_HOUSING_COUNCIL_TAX_METHOD == "uk-spi-housing-council-tax-v1"
    assert UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE == "council_tax"
    assert UK_SPI_HOUSING_COUNCIL_TAX_REFERENCE == "expected_band_d_of_copy_draw_cell"
    assert UK_SPI_HOUSING_COUNCIL_TAX_OUTPUT == "council_tax"
    assert UK_SPI_HOUSING_COUNCIL_TAX_CHECKS == (
        "bill_at_statute_of_reference",
        "northern_ireland_bill_as_chained",
        "reduction_within_bill",
        "rebate_equals_reduction",
    )


def test_the_declaration_pieces_are_the_localisations_and_declarative() -> None:
    (verify_kind, verify), (derive_kind, _) = (
        council_tax_localise_operation_parameters()[:2]
    )
    assert UK_SPI_HOUSING_COUNCIL_TAX_VERIFY_KIND == verify_kind
    assert UK_SPI_HOUSING_COUNCIL_TAX_DERIVE_KIND == derive_kind == "derive"
    mine = uk_spi_housing_council_tax_verify_parameters()
    # The localisation's Scottish verify operation, key order included, which
    # the runtime arm accepts for the shell's stage at the build period.
    assert mine == verify
    assert list(mine) == list(verify)
    check_scotland_band_d_declaration(
        mine, stage=UK_SPI_HOUSING_SHELL_STAGE_NAME, period=UK_BAND_D_BUILD_PERIOD
    )
    derive = uk_spi_housing_council_tax_derive_parameters()
    assert derive == {
        "output": UK_SPI_HOUSING_COUNCIL_TAX_OUTPUT,
        "formula": UK_SPI_HOUSING_COUNCIL_TAX_FORMULA,
    }
    for kind, parameters in ((verify_kind, mine), (derive_kind, derive)):
        _reject_executable_parameter_keys(parameters, context=kind)
        _reject_incumbent_dependencies(parameters, context=kind)
        assert json.loads(json.dumps(parameters)) == parameters
    # Each call gives a fresh declaration.
    mine["resource"] = derive["output"] = "edited"
    assert uk_spi_housing_council_tax_verify_parameters() == verify
    assert uk_spi_housing_council_tax_derive_parameters()["output"] == "council_tax"
    # The two artifacts, as the localisation's packaged stage declares them.
    spec = load_country_spec("uk")
    assert spec.sources is not None
    packaged = spec.sources.stage_map()[COUNCIL_TAX_LOCALISE_STAGE_NAME]
    assert (
        tuple(
            (artifact["role"], artifact["resource"]) for artifact in packaged.artifacts
        )
        == UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS
    )
    assert all(artifact["runtime_sha256_required"] for artifact in packaged.artifacts)


def test_the_formula_names_the_statute_and_live_stages() -> None:
    formula = UK_SPI_HOUSING_COUNCIL_TAX_FORMULA
    spec = load_country_spec("uk")
    assert spec.sources is not None
    stages = spec.sources.stage_map()
    for stage in (SPI_COPY_AREAS_STAGE_NAME, COUNCIL_TAX_LOCALISE_STAGE_NAME):
        assert stage in formula and stage in stages, stage
    assert f"council_tax_band_d.{uk_statutory_council_tax.__name__}" in formula
    # The flag it names is one the shell consumes.
    assert _SINGLE_ADULT in formula
    assert ("household", _SINGLE_ADULT) in _STAGE_CONSUMES[
        UK_SPI_HOUSING_SHELL_STAGE_NAME
    ]
    # No figure but the fiscal year and the flag's value.
    flag = f"{_SINGLE_ADULT} is 1"
    assert UK_BAND_D_FISCAL_YEAR in formula and flag in formula
    figures = formula.replace(UK_BAND_D_FISCAL_YEAR, "").replace(flag, "")
    assert not re.search(r"[0-9]", figures)


def test_the_module_imports_no_stage_module() -> None:
    """The shell is to import the module, so the module imports no stage's
    module: not the shell, the localisation, nor frs_council_tax's."""

    tree = ast.parse(inspect.getsource(shell_council_tax))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    stages = {"spi_housing_shell", "spine_council_tax", "frs_council_tax"}
    assert not {name.rsplit(".", 1)[-1] for name in imported} & stages


# --- The rows and their references --------------------------------------------------


@pytest.mark.parametrize("clones", [1, 3])
def test_the_rows_are_the_support_channels_copies_in_table_order(clones) -> None:
    household, _ = _tables(clones)
    inputs = _inputs(household)
    copies = uk_support_channel_copies(household)
    assert np.array_equal(_recipients(household), copies)
    # Both copy kinds, SPI copies and band donors, are among them.
    assert set(household.loc[copies, "household_support_clone_index"]) == {1, 2}
    assert list(inputs.household_ids) == household.loc[copies, _HOUSEHOLD_ID].tolist()
    assert inputs.regions == tuple(household.loc[copies, REGION_COLUMN])
    assert inputs.great_britain == tuple(
        region in UK_SPI_HOUSING_COUNCIL_TAX_GREAT_BRITAIN_REGIONS
        for region in inputs.regions
    )
    assert True in inputs.great_britain and False in inputs.great_britain
    assert all(
        (reference is None) is (not flag)
        for reference, flag in zip(
            inputs.reference_band_d, inputs.great_britain, strict=True
        )
    )
    # The digests are keyed by the supports given, in system order.
    assert list(inputs.support_sha256) == sorted(_supports())
    assert dict(inputs.support_sha256) == {
        system: support.sha256 for system, support in _supports().items()
    }
    # Frozen, read-only digests, and nothing of a row in the repr.
    with pytest.raises(FrozenInstanceError):
        inputs.regions = ()
    with pytest.raises(TypeError):
        inputs.support_sha256["extra"] = "edited"
    text = repr(inputs)
    assert "support_sha256" in text
    assert not any(str(value) in text for value in inputs.household_ids)
    assert not any(region in text for region in inputs.regions)
    # The Band D the references were taken at takes no part in a comparison.
    assert replace(inputs, band_d=_fed_band_d()) == inputs


@pytest.mark.parametrize("clones", [1, 3])
def test_each_reference_is_the_localisations_own_for_the_copy(clones) -> None:
    household, _ = _tables(clones)
    inputs = _inputs(household)
    references = _references(clones)
    rows = np.flatnonzero(_recipients(household))
    cells: dict[tuple[str, str], set[Fraction]] = {}
    for position, row in enumerate(rows.tolist()):
        mine = inputs.reference_band_d[position]
        if not inputs.great_britain[position]:
            assert mine is None and references.expected_band_d[row] is None
            continue
        region = inputs.regions[position]
        assert isinstance(mine, Fraction)
        assert mine == references.expected_band_d[row], position
        # Worked again: the region's toy areas' Band D, by households.
        assert mine == _region_mean(region), position
        cells.setdefault((str(references.systems[row]), region), set()).add(mine)
    # One reference for each system and region, in both Great Britain systems.
    assert all(len(values) == 1 for values in cells.values())
    assert {system for system, _ in cells} == {EW, SCOT}
    # Every copy is drawn at the copy stage's level.
    assert set(references.levels[rows]) == {UK_SPI_COPY_DRAW_LEVEL}


@pytest.mark.parametrize("clones", [1, 3])
def test_the_references_and_bills_read_no_area_column(clones) -> None:
    household, _ = _tables(clones)
    # The later stages' area columns too, invented, so that each is dropped.
    full = household.assign(
        local_authority=household[UK_LOCAL_AUTHORITY_CODE_COLUMN].astype(object),
        brma="invented-brma",
        in_rural_fuel_duty_relief_area=False,
    )
    inputs = _inputs(full)
    draws = _draws(full, inputs)
    bills = _bills(inputs, draws)
    dropped = full.drop(
        columns=[
            *_AREA_COLUMNS,
            IDENTITY_COLUMN,
            FRS_IMD_DECILE_COLUMN,
            FRS_COUNCIL_TAX_BAND_OBSERVED_COLUMN,
        ]
    )
    # The household columns the shell consumes, with the household id.
    consumed = sorted(
        column
        for entity, column in _STAGE_CONSUMES[UK_SPI_HOUSING_SHELL_STAGE_NAME]
        if entity == "household" and column in full.columns
    )
    narrow = full[[_HOUSEHOLD_ID, *consumed]]
    assert not set(narrow.columns) & set(_AREA_COLUMNS)
    for table in (dropped, narrow):
        mine = _inputs(table)
        assert mine == inputs
        assert mine.reference_band_d == inputs.reference_band_d
        assert np.array_equal(_bits(_bills(mine, draws)), _bits(bills))


@pytest.mark.parametrize("clones", [1, 3])
def test_the_narrow_frame_is_the_area_draws_frame_of_a_copy_at_its_level(
    clones,
) -> None:
    household, _ = _tables(clones)
    copies = _recipients(household)
    frame = uk_spi_copy_draw_frame(household.loc[copies])
    assert list(frame.columns) == [
        REGION_COLUMN,
        DECILE_INPUT_COLUMN,
        BAND_INPUT_COLUMN,
    ]
    string = pd.StringDtype(storage="python")
    assert all(frame[column].dtype == string for column in frame.columns)
    assert frame[DECILE_INPUT_COLUMN].isna().all()
    assert frame[BAND_INPUT_COLUMN].isna().all()
    # The area draw's own inputs at each copy's level, less the key.
    spec = validate_assignment_spec(_definition())
    draw_inputs = uk_draw_inputs(household)
    at_level = uk_draw_frame(
        draw_inputs, uk_draw_levels(draw_inputs, spec, _supports())
    )
    pd.testing.assert_frame_equal(frame, at_level.loc[copies, list(frame.columns)])


@pytest.mark.parametrize("column", [DECILE_INPUT_COLUMN, BAND_INPUT_COLUMN])
def test_a_narrow_frame_without_a_draw_input_is_refused_by_the_shared_expectation(
    column,
) -> None:
    household, _ = _tables(1)
    inputs = _inputs(household)
    rows = _recipients(household) & (household[_SYSTEM] == EW).to_numpy(dtype=bool)
    frame = uk_spi_copy_draw_frame(household.loc[rows])
    (system,) = [s for s in _definition()["systems"] if s["id"] == EW]
    weight_by, default = _system_rules(system)

    def expectations(households: pd.DataFrame) -> list[Fraction]:
        return draw_cell_expectations(
            _supports()[EW],
            households,
            constraint_columns={
                c["input"]: c["support"] for c in system["constraints"]
            },
            weight_by=weight_by,
            default_weight=default,
            value_column=UK_LOCAL_AUTHORITY_CODE_COLUMN,
            values=_band_d().band_d,
        )

    # The whole frame gives the module's references, ...
    positions = np.flatnonzero(rows[_recipients(household)])
    assert expectations(frame) == [inputs.reference_band_d[p] for p in positions]
    # ... and one without the input is refused, naming the column.
    with pytest.raises(ValueError, match="lack input columns") as caught:
        expectations(frame.drop(columns=[column]))
    assert column in str(caught.value)


# --- The bills ----------------------------------------------------------------------


@pytest.mark.parametrize("clones", [1, 3])
def test_each_great_britain_bill_is_the_statute_at_its_reference_bit_for_bit(
    clones,
) -> None:
    household, _ = _tables(clones)
    table = household.copy()
    inputs = _inputs(household)
    draws = _draws(household, inputs)
    given = {name: getattr(draws, name).copy() for name in ("chained", "bands")}
    bills = _bills(inputs, draws)
    assert bills.dtype == np.float64 and bills.shape == (len(inputs.household_ids),)
    # The housing shell hands the flag as its zero-or-one predictor, single
    # exactly where council_tax_single_adult_raw is 1: the same bills.
    flag = draws.single_adult.eq(1.0).astype(np.float64)
    assert (flag == 0.0).any() and (flag == 1.0).any()
    assert np.array_equal(
        _bits(_bills(inputs, replace(draws, single_adult=flag))), _bits(bills)
    )
    # Nothing it reads is changed.
    pd.testing.assert_frame_equal(household, table, check_exact=True)
    for name, series in given.items():
        pd.testing.assert_series_equal(getattr(draws, name), series, check_exact=True)
    references = _references(clones)
    rows = np.flatnonzero(_recipients(household))
    priced = set()
    for position, row in enumerate(rows.tolist()):
        if not inputs.great_britain[position]:
            continue
        region = inputs.regions[position]
        band = draws.bands.iloc[position]
        adults = draws.single_adult.iloc[position]
        expected = _statute(references.expected_band_d[row], region, band, adults)
        assert _bits(bills[position]) == _bits(expected), position
        priced.add((UK_NATION_BY_REGION[region], band, adults == 1.0))
    # Every Great Britain nation, Wales's band I, and both flags are priced.
    assert {nation for nation, _, _ in priced} == {"ENGLAND", "WALES", "SCOTLAND"}
    assert ("WALES", "I", True) in priced
    assert {single for _, _, single in priced} == {True, False}


def test_the_statute_helper_prices_every_band_of_every_great_britain_nation() -> None:
    reference = next(iter(_band_d().band_d.values()))
    regions, bands, flags, expected = [], [], [], []
    for region in UK_SPI_HOUSING_COUNCIL_TAX_GREAT_BRITAIN_REGIONS:
        nation = UK_NATION_BY_REGION[region]
        for band in UK_COUNCIL_TAX_BANDS_BY_NATION[nation]:
            for single in (False, True):
                one = uk_spi_statutory_council_tax(
                    regions=[region],
                    bands=[band],
                    single_adult=[single],
                    reference_band_d=[reference],
                )
                statute = float(
                    uk_statutory_council_tax(
                        reference, nation, band, single_adult=single
                    )
                )
                assert one.dtype == np.float64
                assert _bits(one[0]) == _bits(statute), (region, band, single)
                regions.append(region)
                bands.append(band)
                flags.append(np.bool_(single))
                expected.append(statute)
    # In one call, numpy flags among them, row by row.
    together = uk_spi_statutory_council_tax(
        regions=regions,
        bands=bands,
        single_adult=flags,
        reference_band_d=[reference] * len(regions),
    )
    assert np.array_equal(_bits(together), _bits(expected))


@pytest.mark.parametrize("table", ["one_clone", "housing_shell_tables"])
@pytest.mark.parametrize("given", ["zero", "positive"])
def test_northern_irelands_rows_keep_the_chained_draw_bit_for_bit(table, given) -> None:
    household, _ = _table(table)
    inputs = _inputs(household)
    draws = _draws(household, inputs)
    ni = ~np.asarray(inputs.great_britain, dtype=bool)
    assert ni.any() and (~ni).any()
    chained = draws.chained.copy()
    if given == "zero":
        # Zeros of either sign come back with their sign.
        chained.iloc[np.flatnonzero(ni)] = np.where(
            np.arange(ni.sum()) % 2 == 0, 0.0, -0.0
        )
    else:
        chained.iloc[np.flatnonzero(ni)] = 1_000.0 + 0.1 * np.arange(1, ni.sum() + 1)
    # A Great Britain row's chained bill does not choose its branch.
    chained.iloc[int(np.flatnonzero(~ni)[0])] = 0.0
    bills = _bills(inputs, draws, chained=chained)
    assert np.array_equal(_bits(bills[ni]), _bits(chained.to_numpy()[ni]))
    assert (bills[~ni] > 0.0).all()
    priced = _bills(inputs, draws)
    assert np.array_equal(_bits(bills[~ni]), _bits(priced[~ni]))
    # Northern Ireland's band and flag are not read.
    bands, single = draws.bands.copy(), draws.single_adult.copy()
    bands.iloc[np.flatnonzero(ni)] = "not-a-band"
    single.iloc[np.flatnonzero(ni)] = np.nan
    unread = uk_spi_council_tax_bills(
        chained, inputs=inputs, bands=bands, single_adult=single
    )
    assert np.array_equal(_bits(unread), _bits(bills))


@pytest.mark.parametrize("table", ["three_clones", "housing_shell_tables"])
def test_the_bills_do_not_depend_on_row_order(table) -> None:
    household, _ = _table(table)
    order = np.random.default_rng(931).permutation(len(household))
    permuted = household.iloc[order].reset_index(drop=True)
    by_id = {}
    for frame in (household, permuted):
        inputs = _inputs(frame)
        # The inputs hold their own table's order.
        assert list(inputs.household_ids) == (
            frame.loc[_recipients(frame), _HOUSEHOLD_ID].tolist()
        )
        bills = _bills(inputs, _draws(frame, inputs))
        by_id[len(by_id)] = (
            dict(
                zip(
                    inputs.household_ids,
                    zip(
                        inputs.reference_band_d,
                        _bits(bills).tolist(),
                        inputs.great_britain,
                        strict=True,
                    ),
                    strict=True,
                )
            ),
            dict(inputs.support_sha256),
        )
    assert list(by_id[0][0]) != list(by_id[1][0])
    assert by_id[0] == by_id[1]


def test_amounts_out_of_the_inputs_order_are_refused() -> None:
    household, _ = _tables(1)
    inputs = _inputs(household)
    draws = _draws(household, inputs)
    for name in ("chained", "bands", "single_adult"):
        for change in (lambda s: s.iloc[::-1], lambda s: s.iloc[:-1]):
            arguments = {
                "chained": draws.chained,
                "bands": draws.bands,
                "single_adult": draws.single_adult,
            }
            arguments[name] = change(arguments[name])
            with pytest.raises(ValueError, match="in their order") as caught:
                uk_spi_council_tax_bills(
                    arguments.pop("chained"), inputs=inputs, **arguments
                )
            assert name in str(caught.value)
            _assert_names_no_household(str(caught.value), household)


# --- The checks ---------------------------------------------------------------------


@pytest.mark.parametrize("table", ["one_clone", "three_clones", "housing_shell_tables"])
def test_the_checks_hold_on_the_shells_composition(table) -> None:
    written = _written(*_table(table))
    checks = _checks(written)
    assert tuple(checks) == UK_SPI_HOUSING_COUNCIL_TAX_CHECKS
    assert all(type(value) is bool for value in checks.values())
    assert all(checks.values())
    # The cap binds on some recipient, and some reduction sits below its bill.
    drawn, bills = written.drawn, written.bills.to_numpy()
    assert (drawn > bills).any()
    assert ((drawn > 0.0) & (drawn < bills)).any()


def test_each_check_fails_on_its_own_breach() -> None:
    written = _written(*_tables(1))
    great_britain = np.asarray(written.inputs.great_britain, dtype=bool)
    gb = int(np.flatnonzero(great_britain)[0])
    ni = int(np.flatnonzero(~great_britain)[0])
    bills = written.bills.to_numpy()

    def at(series: pd.Series, position: int, value: float) -> pd.Series:
        changed = series.copy()
        changed.iloc[position] = value
        return changed

    above = np.nextafter(bills[gb], np.inf)
    breaches = {
        # One unit in the last place above the statute.
        "bill_at_statute_of_reference": {"bills": at(written.bills, gb, above)},
        "northern_ireland_bill_as_chained": {
            "bills": at(written.bills, ni, np.nextafter(bills[ni], np.inf))
        },
        # A reduction, and the rebate with it, a unit above its bill.
        "reduction_within_bill": {
            "reduction": at(written.reduction, gb, above),
            "rebate": at(written.rebate, gb, above),
        },
        "rebate_equals_reduction": {
            "rebate": at(written.rebate, gb, written.rebate.iloc[gb] + 1.0)
        },
    }
    above_ni = np.nextafter(bills[ni], np.inf)
    northern_ireland = {
        # The other zero's sign: bit for bit, not equal in value.
        "northern_ireland_bill_as_chained": {
            "chained": at(written.draws.chained, ni, -0.0),
            "bills": at(written.bills, ni, 0.0),
            "reduction": at(written.reduction, ni, 0.0),
            "rebate": at(written.rebate, ni, 0.0),
        },
        "reduction_within_bill": {
            "reduction": at(written.reduction, ni, above_ni),
            "rebate": at(written.rebate, ni, above_ni),
        },
        "rebate_equals_reduction": {
            "rebate": at(written.rebate, ni, written.rebate.iloc[ni] + 1.0)
        },
    }
    assert bills[ni] > 0.0
    for check, changes in (*breaches.items(), *northern_ireland.items()):
        checks = _checks(written, **changes)
        assert checks == {name: name != check for name in checks}, check
    # Missing, infinite or negative amounts fail the reduction check.
    for value in (np.nan, np.inf, -1.0):
        assert not _checks(
            written,
            reduction=at(written.reduction, gb, value),
            rebate=at(written.rebate, gb, value),
        )["reduction_within_bill"]
    assert not _checks(written, bills=at(written.bills, ni, np.nan))[
        "reduction_within_bill"
    ]


# --- With the localisation after the bills ------------------------------------------


@pytest.mark.parametrize("clones", [1, 3])
def test_the_localisation_after_the_bills_prices_each_copy_at_its_own_authority(
    clones,
) -> None:
    household, person = _tables(clones)
    written = _written(household, person)
    after = _localise(written.household, written.person)
    before = _localise(household, person)
    rows = _recipients(household)
    great_britain = _great_britain(household)
    references = after.references
    # Each Great Britain copy's bill is the statute at its own authority.
    distances = _own_statute_ulps(after, written.household)
    assert len(distances) and (distances <= _ULPS).all()
    # Its kept bill, the paid bill WAS reads, is the statute at its reference
    # less its rebate; every copy's is its written bill less its rebate.
    kept = after.council_tax_frs_bill.to_numpy(dtype=np.float64)
    for position, row in enumerate(np.flatnonzero(rows).tolist()):
        bill = written.bills.iloc[position]
        if great_britain[row]:
            statute = _statute(
                references.expected_band_d[row],
                household[REGION_COLUMN].iloc[row],
                household[_BAND].iloc[row],
                household[_SINGLE_ADULT].iloc[row],
            )
            assert _bits(bill) == _bits(statute), position
        expected = np.maximum(bill - written.rebate.iloc[position], 0.0)
        assert _bits(kept[row]) == _bits(expected), position
    # The localisation's cap never binds on a copy: its rebate and reduction
    # are the ratio times the shell's.
    assert not after.capped[rows].any()
    copies = np.flatnonzero(rows & great_britain)
    ratios = references.ratios
    assert np.array_equal(
        _bits(after.council_tax_rebate.to_numpy()[copies]),
        _bits(ratios[copies] * written.household[_REBATE].to_numpy()[copies]),
    )
    owner = pd.Index(household[_HOUSEHOLD_ID]).get_indexer(person[_PERSON_HOUSEHOLD_ID])
    carriers = np.isin(owner, copies) & (written.person[_REDUCTION].to_numpy() > 0.0)
    assert carriers.any()
    assert np.array_equal(
        _bits(after.council_tax_benefit_reported.to_numpy()[carriers]),
        _bits(
            ratios[owner[carriers]] * written.person[_REDUCTION].to_numpy()[carriers]
        ),
    )
    # The FRS rows are as the localisation leaves them without the bills.
    frs = ~rows
    for column in (_BILL, _REBATE, COUNCIL_TAX_FRS_BILL_COLUMN):
        assert np.array_equal(
            _bits(getattr(after, column).to_numpy()[frs]),
            _bits(getattr(before, column).to_numpy()[frs]),
        ), column
    frs_people = frs[owner]
    assert np.array_equal(
        _bits(after.council_tax_benefit_reported.to_numpy()[frs_people]),
        _bits(before.council_tax_benefit_reported.to_numpy()[frs_people]),
    )
    # Northern Ireland's rows, copies among them, are left as written.
    ni = ~great_britain
    assert (ni & rows).any()
    for column in (_BILL, _REBATE):
        assert np.array_equal(
            _bits(getattr(after, column).to_numpy()[ni]),
            _bits(written.household[column].to_numpy()[ni]),
        ), column
    ni_people = ni[owner]
    assert np.array_equal(
        _bits(after.council_tax_benefit_reported.to_numpy()[ni_people]),
        _bits(written.person[_REDUCTION].to_numpy()[ni_people]),
    )


def _a_move(household: pd.DataFrame) -> tuple[int, int]:
    """The first Great Britain copy whose region's toy areas span more than
    one authority, and the first area of its region, with households, in
    another authority."""

    for row in np.flatnonzero(_recipients(household) & _great_britain(household)):
        arrays = _supports()[str(household[_SYSTEM].iloc[row])].arrays
        for area in range(len(arrays["area"])):
            if (
                str(arrays["frs_region"][area]) == household[REGION_COLUMN].iloc[row]
                and int(arrays["households"][area]) > 0
                and str(arrays[UK_LOCAL_AUTHORITY_CODE_COLUMN][area])
                != household[UK_LOCAL_AUTHORITY_CODE_COLUMN].iloc[row]
            ):
                return int(row), area
    raise AssertionError("no Great Britain copy can move to another authority")


def _moved_to(household: pd.DataFrame, row: int, area: int) -> pd.DataFrame:
    """``household`` with the row placed in another area of its system's
    support: its area code and every ladder code that area's."""

    moved = household.copy()
    arrays = _supports()[str(moved[_SYSTEM].iloc[row])].arrays
    codes = {"atomic_area_code": arrays["area"][area]}
    codes.update(
        {
            column: arrays["area" if column == "oa_code" else column][area]
            for column in UK_SPINE_LADDER_COLUMNS
        }
    )
    for column, code in codes.items():
        values = moved[column].to_numpy(dtype=object).copy()
        values[row] = str(code)
        moved[column] = pd.Series(values, index=moved.index, dtype=moved[column].dtype)
    return moved


@pytest.mark.parametrize("clones", [1, 3])
def test_a_copy_moved_within_its_region_keeps_the_shells_side_bit_for_bit(
    clones,
) -> None:
    household, person = _tables(clones)
    row, area = _a_move(household)
    moved = _moved_to(household, row, area)
    authority = UK_LOCAL_AUTHORITY_CODE_COLUMN
    assert moved[authority].iloc[row] != household[authority].iloc[row]
    assert moved[REGION_COLUMN].iloc[row] == household[REGION_COLUMN].iloc[row]
    assert _inputs(moved) == _inputs(household)
    written, moved_written = _written(household, person), _written(moved, person)
    for name in ("bills", "reduction", "rebate"):
        assert np.array_equal(
            _bits(getattr(moved_written, name)), _bits(getattr(written, name))
        ), name
    after = _localise(written.household, written.person)
    moved_after = _localise(moved_written.household, moved_written.person)
    # The localisation alone reads the move: the copy's bill becomes the
    # statute at its new authority, ...
    assert _bits(moved_after.council_tax.iloc[row]) != _bits(
        after.council_tax.iloc[row]
    )
    assert (_own_statute_ulps(moved_after, moved_written.household) <= _ULPS).all()
    # ... and every kept bill is as before.
    assert np.array_equal(
        _bits(moved_after.council_tax_frs_bill), _bits(after.council_tax_frs_bill)
    )


# --- The wrong designs --------------------------------------------------------------


def _wrong_reference(wrong: str, authority: object, region: str) -> Fraction:
    """A copy's reference under a wrong design: its drawn authority's Band D,
    or its region's toy areas' Band D, each area once."""

    if wrong == "reference_at_the_drawn_authority":
        return _band_d().band_d[str(authority)]
    return _region_mean(region, by_households=False)


@pytest.mark.parametrize("wrong", _WRONG_DESIGNS)
def test_the_bills_of_each_wrong_design_are_caught(wrong) -> None:
    household, person = _tables(1)
    right = _written(household, person)
    inputs, draws = right.inputs, right.draws
    rows = np.flatnonzero(_recipients(household))
    great_britain = np.asarray(inputs.great_britain, dtype=bool)
    references = _references(1)
    if wrong == "northern_ireland_priced_by_the_statute":
        ni = np.flatnonzero(~great_britain)
        reference = inputs.reference_band_d[int(np.flatnonzero(great_britain)[0])]
        statute = float(
            uk_statutory_council_tax(reference, "ENGLAND", "D", single_adult=False)
        )
        zero = draws.chained.copy()
        zero.iloc[ni] = 0.0
        # Priced where the chained bill is zero, or where it is positive, or
        # zeroed where it is positive: the check catches each.
        for chained, value in (
            (zero, statute),
            (draws.chained, statute),
            (draws.chained, 0.0),
        ):
            bills = _bills(inputs, draws, chained=chained)
            assert (chained.iloc[ni] != value).all()
            bills[ni] = value
            checks = _checks(right, chained=chained, bills=_on(inputs, bills))
            assert checks["northern_ireland_bill_as_chained"] is False
        return
    if wrong == "cap_at_the_localised_bill":
        # The cap at the bill the localisation is to write, its ratio times
        # the shell's, in place of the shell's own bill.
        bills = right.bills.to_numpy()
        ratios = references.ratios[rows]
        localised = np.where(great_britain, ratios * bills, bills)
        capped = _written(household, person, bound=localised)
        assert _checks(capped)["reduction_within_bill"] is False
        after = _localise(capped.household, capped.person)
        assert after.capped[rows].any()
        return
    if wrong == "chained_bill_kept_in_great_britain":
        bills = draws.chained.to_numpy()
    else:
        authorities = household[UK_LOCAL_AUTHORITY_CODE_COLUMN].to_numpy(dtype=object)
        reference = tuple(
            _wrong_reference(wrong, code, region) if flag else None
            for code, region, flag in zip(
                authorities[rows], inputs.regions, great_britain, strict=True
            )
        )
        # Not the localisation's reference on some copy, ...
        assert any(
            mine != references.expected_band_d[row]
            for mine, row, flag in zip(reference, rows, great_britain, strict=True)
            if flag
        )
        bills = _bills(replace(inputs, reference_band_d=reference), draws)
    # ... so its bills fail the statute check, and once localised some copy's
    # bill is not the statute at its own authority.
    wrong_written = _written(household, person, bills=bills)
    assert _checks(wrong_written)["bill_at_statute_of_reference"] is False
    after = _localise(wrong_written.household, wrong_written.person)
    assert (_own_statute_ulps(after, wrong_written.household) > _ULPS).any()


# --- The evidence -------------------------------------------------------------------


def _leaves(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _keys(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _keys(item)


def _sha256(resource: str) -> str:
    return hashlib.sha256(
        files("microcosm.build.uk").joinpath(resource).read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("which", ["invented", "fed", "committed"])
def test_the_evidence_has_no_number_leaf(which) -> None:
    band_d = {
        "invented": _band_d,
        "fed": _fed_band_d,
        "committed": _committed_band_d,
    }[which]()
    written = _written(*_tables(1), band_d=band_d)
    checks = _checks(written)
    assert all(checks.values())
    evidence = uk_spi_housing_council_tax_evidence(written.inputs, checks=checks)
    assert list(evidence) == [
        "method",
        "reference",
        "copy_draw_level",
        "band_d",
        "support_sha256",
        "checks",
    ]
    assert (
        evidence["method"],
        evidence["reference"],
        evidence["copy_draw_level"],
    ) == (
        UK_SPI_HOUSING_COUNCIL_TAX_METHOD,
        UK_SPI_HOUSING_COUNCIL_TAX_REFERENCE,
        UK_SPI_COPY_DRAW_LEVEL,
    )
    # The source commit is read from the feed as the Band D's own evidence
    # reads it.
    source_commit = band_d.evidence()["source_commit"]
    assert evidence["band_d"] == {
        "england_wales": {
            "resource": UK_BAND_D_ENGLAND_WALES_RESOURCE,
            "sha256": band_d.england_wales_sha256,
        },
        "scotland": {
            "resource": SCOTLAND_BAND_D_RESOURCE,
            "sha256": band_d.scotland_sha256,
            "source_commit": source_commit,
        },
    }
    if which == "invented":
        assert source_commit is None
    else:
        assert isinstance(source_commit, str) and source_commit
    if which == "committed":
        assert band_d.england_wales_sha256 == _sha256(UK_BAND_D_ENGLAND_WALES_RESOURCE)
        assert band_d.scotland_sha256 == _sha256(SCOTLAND_BAND_D_RESOURCE)
    assert evidence["support_sha256"] == {
        system: support.sha256 for system, support in sorted(_supports().items())
    }
    assert evidence["checks"] == dict.fromkeys(UK_SPI_HOUSING_COUNCIL_TAX_CHECKS, True)
    # No number leaf: strings and booleans, and the absent commit alone.
    leaves = list(_leaves(evidence))
    assert all(leaf is None or type(leaf) in (str, bool) for leaf in leaves)
    assert sum(leaf is None for leaf in leaves) == (which == "invented")
    # Nothing of the feed but its source commit, and no clone count.
    keys = set(_keys(evidence))
    assert "fact_row_count" not in keys and "clone_count" not in keys
    assert "invented-feed" not in leaves
    assert json.loads(json.dumps(evidence)) == evidence


def test_the_evidence_is_the_same_across_frames_and_clone_counts() -> None:
    def cases(region: str) -> list:
        return [case for case in COUNCIL_TAX_CASES if case.household.region != region]

    def tables(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = council_tax_spine_frame(clones=1, **kwargs)
        return frame.table("household"), frame.table("person")

    frames = {
        "one_clone": _tables(1),
        "three_clones": _tables(3),
        "no_scottish_copy": tables(cases=cases("SCOTLAND")),
        "no_northern_ireland_copy": tables(cases=cases(_NORTHERN_IRELAND)),
        "no_copy": tables(picks=lambda source, clone: ()),
        "housing_shell_tables": _shell_tables(),
    }
    blocks = {}
    regions = {}
    for name, (household, person) in frames.items():
        written = _written(household, person)
        checks = _checks(written)
        assert all(checks.values()), name
        blocks[name] = uk_spi_housing_council_tax_evidence(
            written.inputs, checks=checks
        )
        regions[name] = set(written.inputs.regions)
    # The frames differ in the copies they hold.
    assert {"SCOTLAND", _NORTHERN_IRELAND} <= regions["one_clone"]
    assert "SCOTLAND" not in regions["no_scottish_copy"]
    assert _NORTHERN_IRELAND not in regions["no_northern_ireland_copy"]
    assert not regions["no_copy"]
    # The block is the same on each, key order included.
    text = json.dumps(blocks["one_clone"])
    for name, block in blocks.items():
        assert block == blocks["one_clone"], name
        assert json.dumps(block) == text, name


# --- The refusals -------------------------------------------------------------------

#: The digits a refusal may print: the Band D's fiscal year and build
#: period, the issue, and dtype names, as the localisation's tests declare
#: them. Copied: under pytest's importlib mode one test module cannot import
#: another.
_DECLARED_DIGITS = ("microcosm#931", "2024-25", "2024", "float64", "int64")


def _assert_names_no_household(message: str, household: pd.DataFrame) -> None:
    text = message
    for token in _DECLARED_DIGITS:
        text = text.replace(token, "")
    assert not re.search(r"[0-9]", text), message
    for column in (IDENTITY_COLUMN, "atomic_area_code", UK_LOCAL_AUTHORITY_CODE_COLUMN):
        if column not in household.columns:
            continue
        values = {
            value for value in household[column] if isinstance(value, str) and value
        }
        assert not any(value in message for value in values), column


def _assert_names_no_band_or_region(message: str) -> None:
    assert not re.search(r"\b[A-I]\b", message), message
    names = {*UK_NATION_BY_REGION, *UK_NATION_BY_REGION.values()}
    assert not any(name in message for name in names), message


def _k1() -> tuple[pd.DataFrame, UKSPIHousingCouncilTaxInputs, _Draws]:
    household, _ = _tables(1)
    inputs = _inputs(household)
    return household, inputs, _draws(household, inputs)


def _with_household(change: Callable[[pd.DataFrame], pd.DataFrame]):
    def call() -> object:
        return _inputs(change(_tables(1)[0].copy()))

    return call


def _at_first_copy(
    column: str, value: object
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def change(household: pd.DataFrame) -> pd.DataFrame:
        values = household[column].to_numpy(dtype=object).copy()
        values[int(np.flatnonzero(_recipients(household))[0])] = value
        household[column] = pd.Series(values, index=household.index, dtype=object)
        return household

    return change


def _inputs_with(**fields: object):
    def call() -> object:
        return replace(_k1()[1], **fields)

    return call


def _on_every_row(name: str, value: Callable[[object], object]):
    def call() -> object:
        inputs = _k1()[1]
        return replace(inputs, **{name: value(inputs)})

    return call


def _statute_with(**changes: object):
    def call() -> object:
        arguments = {
            "regions": ["LONDON"],
            "bands": ["B"],
            "single_adult": [False],
            "reference_band_d": [next(iter(_band_d().band_d.values()))],
        }
        arguments.update(changes)
        return uk_spi_statutory_council_tax(**arguments)

    return call


def _bills_with(**changes: Callable[[_Draws], object]):
    def call() -> object:
        _, inputs, draws = _k1()
        arguments = {
            "chained": draws.chained,
            "bands": draws.bands,
            "single_adult": draws.single_adult,
            "inputs": inputs,
        }
        arguments.update({name: change(draws) for name, change in changes.items()})
        return uk_spi_council_tax_bills(arguments.pop("chained"), **arguments)

    return call


def _checks_with(**changes: Callable[[_Written], object]):
    def call() -> object:
        written = _written(*_tables(1))
        return _checks(
            written, **{name: change(written) for name, change in changes.items()}
        )

    return call


def _evidence_with(
    *, band_d: Callable[[], UKCouncilTaxBandD] | None = None, checks=None
):
    def call() -> object:
        household, _ = _tables(1)
        inputs = _inputs(household, None if band_d is None else band_d())
        given = dict.fromkeys(UK_SPI_HOUSING_COUNCIL_TAX_CHECKS, True)
        return uk_spi_housing_council_tax_evidence(
            inputs, checks=given if checks is None else checks
        )

    return call


def _band_d_without(region: str) -> UKCouncilTaxBandD:
    """The invented Band D without the authorities of ``region``'s toy areas."""

    codes = {
        str(code)
        for support in _supports().values()
        for frs_region, code in zip(
            support.arrays["frs_region"],
            support.arrays[UK_LOCAL_AUTHORITY_CODE_COLUMN],
            strict=True,
        )
        if str(frs_region) == region
    }
    return replace(
        _band_d(),
        band_d=MappingProxyType(
            {
                code: value
                for code, value in _band_d().band_d.items()
                if code not in codes
            }
        ),
    )


def _int64(series: pd.Series) -> pd.Series:
    return series.round().astype(np.int64)


_EXACTLY = "these do not"
_REFUSALS = [
    # The inputs.
    pytest.param(
        TypeError,
        "prices bills at a UKCouncilTaxBandD",
        lambda: uk_spi_housing_council_tax_inputs(
            _tables(1)[0],
            definition=_definition(),
            supports=_supports(),
            band_d=dict(_band_d().band_d),
        ),
        id="inputs_band_d_not_a_band_d",
    ),
    pytest.param(
        TypeError,
        "reads the household table as a DataFrame",
        lambda: _inputs(_tables(1)[0].to_dict()),
        id="inputs_household_not_a_frame",
    ),
    *(
        pytest.param(
            ValueError,
            "the table lacks them",
            _with_household(lambda h, c=column: h.drop(columns=[c])),
            id=f"inputs_without_{column}",
        )
        for column in (
            _HOUSEHOLD_ID,
            UK_SPI_HOUSING_COUNCIL_TAX_CHANNEL_COLUMN,
            REGION_COLUMN,
        )
    ),
    *(
        pytest.param(
            ValueError,
            "reads a decoded conditioned support for every system",
            lambda supports=supports: uk_spi_housing_council_tax_inputs(
                _tables(1)[0],
                definition=_definition(),
                supports=supports(),
                band_d=_band_d(),
            ),
            id=f"inputs_supports_{name}",
        )
        for name, supports in (
            (
                "without_a_system",
                lambda: {k: v for k, v in _supports().items() if k != SCOT},
            ),
            (
                "with_another_system",
                lambda: {**_supports(), "invented": _supports()[EW]},
            ),
            ("undecoded", toy_conditioned_support_payloads),
        )
    ),
    *(
        pytest.param(
            ValueError,
            "reads a nonempty 'region' on every row it prices",
            _with_household(_at_first_copy(REGION_COLUMN, value)),
            id=f"inputs_region_{name}",
        )
        for name, value in (("missing", None), ("blank", ""), ("padded", " LONDON"))
    ),
    pytest.param(
        ValueError,
        "every household must fall in exactly one area system by its region",
        _with_household(_at_first_copy(REGION_COLUMN, "INVENTED_REGION")),
        id="inputs_region_no_system_selects",
    ),
    pytest.param(
        ValueError,
        "takes the expected 2024-25 Band D of each copy's draw cell",
        lambda: _inputs(_tables(1)[0], _band_d_without("LONDON")),
        id="inputs_band_d_without_a_cell_authority",
    ),
    # The narrow frame.
    pytest.param(
        TypeError,
        "reads the household table as a DataFrame",
        lambda: uk_spi_copy_draw_frame(list(_tables(1)[0][REGION_COLUMN])),
        id="frame_not_a_frame",
    ),
    pytest.param(
        ValueError,
        "the table lacks them",
        lambda: uk_spi_copy_draw_frame(_tables(1)[0].drop(columns=[REGION_COLUMN])),
        id="frame_without_region",
    ),
    # The inputs' own form.
    pytest.param(
        ValueError,
        _EXACTLY,
        _on_every_row("regions", lambda inputs: inputs.regions[:-1]),
        id="inputs_of_unequal_lengths",
    ),
    pytest.param(
        ValueError,
        _EXACTLY,
        _on_every_row("household_ids", lambda inputs: list(inputs.household_ids)),
        id="inputs_of_a_list",
    ),
    pytest.param(
        ValueError,
        _EXACTLY,
        _on_every_row(
            "great_britain",
            lambda inputs: tuple(not flag for flag in inputs.great_britain),
        ),
        id="inputs_flag_not_the_regions",
    ),
    pytest.param(
        ValueError,
        _EXACTLY,
        _on_every_row(
            "reference_band_d",
            lambda inputs: tuple(
                reference if flag else Fraction(1)
                for reference, flag in zip(
                    inputs.reference_band_d, inputs.great_britain, strict=True
                )
            ),
        ),
        id="inputs_reference_on_northern_ireland",
    ),
    pytest.param(
        ValueError,
        _EXACTLY,
        _on_every_row(
            "reference_band_d",
            lambda inputs: tuple(
                None if reference is None else float(reference)
                for reference in inputs.reference_band_d
            ),
        ),
        id="inputs_float_reference",
    ),
    pytest.param(
        TypeError,
        "prices bills at a UKCouncilTaxBandD",
        _inputs_with(band_d=None),
        id="inputs_band_d_field",
    ),
    pytest.param(
        ValueError,
        "takes the supports' digests as strings keyed by system",
        _inputs_with(support_sha256={EW: None}),
        id="inputs_digest_not_text",
    ),
    # The statute.
    pytest.param(
        ValueError,
        "differ in length",
        _statute_with(bands=[]),
        id="statute_of_unequal_lengths",
    ),
    *(
        pytest.param(
            ValueError,
            "some flag is not a bool",
            _statute_with(single_adult=[flag]),
            id=f"statute_flag_{name}",
        )
        for name, flag in (("number", 1.0), ("missing", None), ("text", "True"))
    ),
    *(
        pytest.param(
            ValueError,
            "some row's region is not",
            _statute_with(regions=[region]),
            id=f"statute_region_{name}",
        )
        for name, region in (
            ("northern_ireland", _NORTHERN_IRELAND),
            ("unknown", "INVENTED_REGION"),
        )
    ),
    *(
        pytest.param(
            ValueError,
            "some row's band is not one of them",
            _statute_with(regions=[region], bands=[band]),
            id=f"statute_band_{name}",
        )
        for name, region, band in (
            ("i_in_england", "LONDON", "I"),
            ("i_in_scotland", "SCOTLAND", "I"),
            ("unknown", "WALES", "Z"),
            ("missing", "LONDON", None),
        )
    ),
    pytest.param(
        ValueError,
        "must be an exact rational",
        _statute_with(reference_band_d=[1_000.5]),
        id="statute_float_reference",
    ),
    pytest.param(
        ValueError,
        "must be positive",
        _statute_with(reference_band_d=[Fraction(0)]),
        id="statute_zero_reference",
    ),
    # The bills.
    pytest.param(
        TypeError,
        "reads inputs built by uk_spi_housing_council_tax_inputs",
        _bills_with(inputs=lambda draws: object()),
        id="bills_of_other_inputs",
    ),
    pytest.param(
        TypeError,
        "reads chained as a Series indexed by household_id",
        _bills_with(chained=lambda draws: draws.chained.tolist()),
        id="bills_chained_not_a_series",
    ),
    pytest.param(
        ValueError,
        "reads chained as float64, bit for bit; it holds int64",
        _bills_with(chained=lambda draws: _int64(draws.chained)),
        id="bills_chained_int64",
    ),
    pytest.param(
        ValueError,
        "reads chained as float64, bit for bit; it holds object",
        _bills_with(chained=lambda draws: draws.chained.astype(object)),
        id="bills_chained_object",
    ),
    pytest.param(
        ValueError,
        "reads single_adult as a numeric flag",
        _bills_with(single_adult=lambda draws: draws.single_adult.astype(str)),
        id="bills_single_adult_text",
    ),
    pytest.param(
        ValueError,
        "some row's band is not one of them",
        _bills_with(
            bands=lambda draws: pd.Series("I", index=draws.bands.index, dtype=object)
        ),
        id="bills_band_i_in_england",
    ),
    # The checks.
    *(
        pytest.param(
            ValueError,
            f"reads {name} as float64, bit for bit; it holds int64",
            _checks_with(**{name: lambda written, n=name: _int64(getattr(written, n))}),
            id=f"checks_{name}_int64",
        )
        for name in ("bills", "reduction", "rebate")
    ),
    pytest.param(
        ValueError,
        "reads rebate on the inputs' rows",
        _checks_with(rebate=lambda written: written.rebate.iloc[::-1]),
        id="checks_rebate_out_of_order",
    ),
    # The evidence.
    pytest.param(
        TypeError,
        "reads inputs built by uk_spi_housing_council_tax_inputs",
        lambda: uk_spi_housing_council_tax_evidence(
            object(), checks=dict.fromkeys(UK_SPI_HOUSING_COUNCIL_TAX_CHECKS, True)
        ),
        id="evidence_of_other_inputs",
    ),
    *(
        pytest.param(
            ValueError,
            "and no other, each a bool",
            _evidence_with(checks=checks),
            id=f"evidence_checks_{name}",
        )
        for name, checks in (
            ("missing", {UK_SPI_HOUSING_COUNCIL_TAX_CHECKS[0]: True}),
            (
                "another",
                {
                    **dict.fromkeys(UK_SPI_HOUSING_COUNCIL_TAX_CHECKS, True),
                    "other": True,
                },
            ),
            ("not_bool", dict.fromkeys(UK_SPI_HOUSING_COUNCIL_TAX_CHECKS, 1)),
        )
    ),
    pytest.param(
        ValueError,
        "records each Band D resource's digest as a string",
        _evidence_with(band_d=lambda: replace(_band_d(), england_wales_sha256=None)),
        id="evidence_digest_not_text",
    ),
    pytest.param(
        ValueError,
        "records the Scottish Band D feed's source_commit as a string",
        _evidence_with(
            band_d=lambda: replace(
                _band_d(),
                scotland_source_fact_feed=MappingProxyType({"source_commit": 7}),
            )
        ),
        id="evidence_source_commit_not_text",
    ),
]


#: The refusals the module makes before the statute is called, whose texts
#: name no band letter, region or nation either.
_BEFORE_THE_STATUTE = (
    "statute_band_",
    "statute_region_",
    "bills_band_",
    "statute_of_unequal_lengths",
    "statute_flag_",
)


@pytest.mark.parametrize(("kind", "text", "call"), _REFUSALS)
def test_each_refusal_names_columns_dtypes_and_declared_constants_only(
    kind, text, call, request
) -> None:
    with pytest.raises(kind, match=re.escape(text)) as caught:
        call()
    message = str(caught.value)
    _assert_names_no_household(message, _tables(1)[0])
    if request.node.callspec.id.startswith(_BEFORE_THE_STATUTE):
        _assert_names_no_band_or_region(message)


def test_rows_of_the_channel_that_are_not_the_lineages_copies_are_refused(
    monkeypatch,
) -> None:
    household, _ = _tables(1)
    # The lineage's reader agrees with the channel wherever it reads a table,
    # so a reader that marks no row stands in for one that disagrees.
    monkeypatch.setattr(
        shell_council_tax,
        "uk_support_channel_copies",
        lambda table: np.zeros(len(table), dtype=bool),
    )
    with pytest.raises(ValueError, match="the two differ") as caught:
        _inputs(household)
    _assert_names_no_household(str(caught.value), household)
