"""The housing shell's council tax on the rows the SPI support channel copied (microcosm#931).

The housing shell (``spi_housing_shell``) draws the housing of the SPI copies
and income band donors from their region, composition and incomes: tenure,
dwelling type, bedrooms and council-tax band from its classifiers, then one
chained QRF of amounts whose first target is ``council_tax``. This module
prices that bill by the statute on the Great Britain rows, and states what
the shell's stage declares for it. It draws nothing and calls no engine.

- **The reference.** A copy's draw inputs are missing by rule, so the
  SPI-copy area stage (``spi_copy_areas``) draws its area at L3
  (:data:`UK_SPI_COPY_DRAW_LEVEL`), its region by households. Its reference
  Band D is the exact expected 2024-25 Band D of that cell under the draw's
  own weights: ``E = sum(households x D) / sum(households)`` over the
  region's areas, ``D`` each area's local authority's Band D.
  :func:`uk_spi_housing_council_tax_inputs` takes it from
  :func:`~microcosm.build.atomic_support_expectations.draw_cell_expectations`
  with each system's constraints, weight rule and default, the arguments the
  council-tax localisation (``council_tax_localise``) passes for the same
  rows, on a narrow frame of the region alone whose decile and band inputs
  are present and missing (:func:`uk_spi_copy_draw_frame`). It reads each
  row's region, support channel and copy lineage, and no area column,
  geography key, decile or band flag.
- **The bill.** On a Great Britain row, one whose region's nation
  :data:`~microcosm.build.uk_runtime.council_tax_band_d.UK_COUNCIL_TAX_BAND_RATIOS`
  prices, :func:`uk_spi_statutory_council_tax` gives the 2024-25 statute
  (:func:`~microcosm.build.uk_runtime.council_tax_band_d.uk_statutory_council_tax`)
  at the reference for the row's nation and drawn band, less the
  single-adult discount where the household has a single adult: one
  correctly rounded conversion of an exact rational, made once for each
  distinct reference, nation, band and flag. :func:`uk_spi_council_tax_bills`
  returns it there, and on a Northern Ireland row the chained draw as given,
  bit for bit (María's ruling of 2026-11-14). The branch is the nation of the
  row's region, never a data value.
- **With the localisation.** The localisation later multiplies a Great
  Britain copy's bill, rebate and reduction by its drawn authority's Band D
  over the same reference, so the bill becomes the statute at that authority
  within a few units in the last place, and the localisation stays the one
  step that localises the bill. The paid bill it keeps for WAS's bill
  predictor (``council_tax_frs_bill``, from the bill and rebate before it
  rewrites them) is the statute at the reference less the rebate: it reads
  the band and the region's reference, never the drawn area or its
  authority.
- **The checks** (:func:`uk_spi_housing_council_tax_checks`, in the order of
  :data:`UK_SPI_HOUSING_COUNCIL_TAX_CHECKS`), as booleans: every Great
  Britain bill at the statute of its reference, every Northern Ireland bill
  as chained, every reduction within its bill, finite and nonnegative, and
  every rebate equal to its reduction.
- **The evidence** (:func:`uk_spi_housing_council_tax_evidence`): the method,
  the reference, the copy level, the two Band D resources with their digests
  and the Scottish feed's source commit, the supports' digests and the
  checks. Its leaves are strings and booleans (the source commit is ``None``
  for a Band D a test supplies without a feed), never a number, and its keys
  are the same whatever systems, nations or clones a frame holds.
- **The declaration.** The two Band D artifacts
  (:data:`UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS`) and the Scottish resource's
  ``verify_vendored_fact_resource`` operation, as the localisation declares
  them (:func:`uk_spi_housing_council_tax_verify_parameters`), and the
  ``derive`` of ``council_tax``
  (:func:`uk_spi_housing_council_tax_derive_parameters`).

This module's refusals name columns, dtypes and declared constants only,
never a household's id, area code, authority code, band, value or count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import AtomicSupport, validate_assignment_spec
from microcosm.build.atomic_support_expectations import draw_cell_expectations
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    BAND_INPUT_COLUMN,
    DECILE_INPUT_COLUMN,
    REGION_COLUMN,
    SUPPORT_CHANNEL_BRANCH,
    UK_ATOMIC_FALLBACK_LEVELS,
    _system_rules,
    uk_draw_systems,
    uk_support_channel_copies,
)
from microcosm.build.uk_runtime.council_tax_band_d import (
    SCOTLAND_BAND_D_ARTIFACT_ROLE,
    SCOTLAND_BAND_D_RECORD_SETS,
    SCOTLAND_BAND_D_RESOURCE,
    UK_BAND_D_ARTIFACT_ROLE,
    UK_BAND_D_BUILD_PERIOD,
    UK_BAND_D_ENGLAND_WALES_RESOURCE,
    UK_BAND_D_FISCAL_YEAR,
    UK_COUNCIL_TAX_BAND_RATIOS,
    UKCouncilTaxBandD,
    uk_statutory_council_tax,
)
from microcosm.build.uk_runtime.council_tax_constants import UK_NATION_BY_REGION
from microcosm.build.uk_runtime.household_lineage import UK_COPY_LINEAGE
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
)

__all__ = [
    "UK_SPI_COPY_DRAW_LEVEL",
    "UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS",
    "UK_SPI_HOUSING_COUNCIL_TAX_CHANNEL_COLUMN",
    "UK_SPI_HOUSING_COUNCIL_TAX_CHECKS",
    "UK_SPI_HOUSING_COUNCIL_TAX_DERIVE_KIND",
    "UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE",
    "UK_SPI_HOUSING_COUNCIL_TAX_FORMULA",
    "UK_SPI_HOUSING_COUNCIL_TAX_GREAT_BRITAIN_REGIONS",
    "UK_SPI_HOUSING_COUNCIL_TAX_METHOD",
    "UK_SPI_HOUSING_COUNCIL_TAX_OUTPUT",
    "UK_SPI_HOUSING_COUNCIL_TAX_RECIPIENT_CHANNEL",
    "UK_SPI_HOUSING_COUNCIL_TAX_REFERENCE",
    "UK_SPI_HOUSING_COUNCIL_TAX_VERIFY_KIND",
    "UKSPIHousingCouncilTaxInputs",
    "uk_spi_copy_draw_frame",
    "uk_spi_council_tax_bills",
    "uk_spi_housing_council_tax_checks",
    "uk_spi_housing_council_tax_derive_parameters",
    "uk_spi_housing_council_tax_evidence",
    "uk_spi_housing_council_tax_inputs",
    "uk_spi_housing_council_tax_verify_parameters",
    "uk_spi_statutory_council_tax",
]

UK_SPI_HOUSING_COUNCIL_TAX_METHOD = "uk-spi-housing-council-tax-v1"
#: The key under which the housing shell's receipt carries the evidence.
UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE = "council_tax"
#: What a Great Britain row's bill is priced at: the expected Band D of the
#: cell the SPI-copy area stage drew the row's area from.
UK_SPI_HOUSING_COUNCIL_TAX_REFERENCE = "expected_band_d_of_copy_draw_cell"
#: The level every copy is drawn at: the last of the fallback ladder, its
#: region by households.
UK_SPI_COPY_DRAW_LEVEL = UK_ATOMIC_FALLBACK_LEVELS[-1].name
#: The support channel's step of the copy lineage: its marker column and the
#: marker's value on the rows the channel copied, the rows the housing shell
#: redraws.
_SUPPORT_CHANNEL = next(
    branch
    for branch in UK_COPY_LINEAGE.branches
    if branch.name == SUPPORT_CHANNEL_BRANCH
)
UK_SPI_HOUSING_COUNCIL_TAX_CHANNEL_COLUMN: str = _SUPPORT_CHANNEL.marker
UK_SPI_HOUSING_COUNCIL_TAX_RECIPIENT_CHANNEL: str = _SUPPORT_CHANNEL.copy
#: The regions whose nation levies council tax, so the statute prices their
#: rows: every region but Northern Ireland's, which pays domestic rates.
UK_SPI_HOUSING_COUNCIL_TAX_GREAT_BRITAIN_REGIONS: tuple[str, ...] = tuple(
    sorted(
        region
        for region, nation in UK_NATION_BY_REGION.items()
        if nation in UK_COUNCIL_TAX_BAND_RATIOS
    )
)
#: The two Band D artifacts the stage declares, (role, resource), as the
#: council-tax localisation declares them: England's and Wales's, and
#: Scotland's vendored facts.
UK_SPI_HOUSING_COUNCIL_TAX_ARTIFACTS: tuple[tuple[str, str], ...] = (
    (UK_BAND_D_ARTIFACT_ROLE, UK_BAND_D_ENGLAND_WALES_RESOURCE),
    (SCOTLAND_BAND_D_ARTIFACT_ROLE, SCOTLAND_BAND_D_RESOURCE),
)
UK_SPI_HOUSING_COUNCIL_TAX_VERIFY_KIND = "verify_vendored_fact_resource"
UK_SPI_HOUSING_COUNCIL_TAX_DERIVE_KIND = "derive"
#: The column the stage's ``derive`` writes.
UK_SPI_HOUSING_COUNCIL_TAX_OUTPUT = "council_tax"
#: The ``derive``'s formula, as the stage declares it.
UK_SPI_HOUSING_COUNCIL_TAX_FORMULA = (
    "After the band draw and the chained amounts, on the Great Britain rows the "
    "SPI support channel copied (SPI copies and income band donors): the 2024-25 "
    "statute (council_tax_band_d.uk_statutory_council_tax) at the copy's "
    "reference Band D, the exact expected 2024-25 Band D of the cell "
    "spi_copy_areas drew its area from (its region, by households, as "
    "council_tax_localise takes it), for its nation and drawn band, less the "
    "single-adult discount where council_tax_single_adult_raw is 1. Northern "
    "Ireland's rows keep the chained draw. The bill reads neither the drawn area "
    "nor its local authority, so council_tax_localise localises it once."
)
#: The checks, in the evidence's order. Each must hold on every row the
#: support channel copied.
UK_SPI_HOUSING_COUNCIL_TAX_CHECKS = (
    "bill_at_statute_of_reference",
    "northern_ireland_bill_as_chained",
    "reduction_within_bill",
    "rebate_equals_reduction",
)
_HOUSEHOLD_ID_COLUMN = "household_id"
_BAND_COLUMN = "council_tax_band"
_FLOAT64 = np.dtype("float64")
_WHAT = "The housing shell's council tax"


# --- The declaration ----------------------------------------------------------------


def uk_spi_housing_council_tax_verify_parameters() -> dict[str, object]:
    """The Scottish Band D resource's ``verify_vendored_fact_resource``
    parameters, as the council-tax localisation declares them."""

    return {
        "artifact_role": SCOTLAND_BAND_D_ARTIFACT_ROLE,
        "resource": SCOTLAND_BAND_D_RESOURCE,
        "feed_pin": "chronicle_feed.json",
        "record_sets": list(SCOTLAND_BAND_D_RECORD_SETS),
        "source_vintage": "2024-25 (ScotGov CTAS 2026)",
        "mapped_build_period": UK_BAND_D_BUILD_PERIOD,
        "period_mapping": "fiscal_year_2024_25_is_the_build_period",
        "require_before_source_read": True,
        "runtime_sha256_required": True,
        "fail_on_mismatch": True,
    }


def uk_spi_housing_council_tax_derive_parameters() -> dict[str, object]:
    """The ``derive`` the stage declares: ``council_tax`` and its formula."""

    return {
        "output": UK_SPI_HOUSING_COUNCIL_TAX_OUTPUT,
        "formula": UK_SPI_HOUSING_COUNCIL_TAX_FORMULA,
    }


# --- The inputs ---------------------------------------------------------------------


_GREAT_BRITAIN_REGIONS = frozenset(UK_SPI_HOUSING_COUNCIL_TAX_GREAT_BRITAIN_REGIONS)


def _levies_council_tax(region: object) -> bool:
    """Whether a region's nation levies council tax: a Great Britain region."""

    return isinstance(region, str) and region in _GREAT_BRITAIN_REGIONS


@dataclass(frozen=True)
class UKSPIHousingCouncilTaxInputs:
    """The rows the SPI support channel copied, and each one's reference.

    In the household table's row order: ``household_ids``, ``regions``,
    ``great_britain`` (whether the region's nation levies council tax) and
    ``reference_band_d``, the exact expected 2024-25 Band D of the row's copy
    draw cell on a Great Britain row and ``None`` elsewhere. ``band_d`` is
    the Band D the references were taken at, and ``support_sha256`` the
    digests of the supports given, by system. Nothing but the digests prints
    in the repr, and ``band_d`` takes no part in a comparison.
    """

    household_ids: tuple[object, ...] = field(repr=False)
    regions: tuple[str, ...] = field(repr=False)
    great_britain: tuple[bool, ...] = field(repr=False)
    reference_band_d: tuple[Fraction | None, ...] = field(repr=False)
    band_d: UKCouncilTaxBandD = field(repr=False, compare=False)
    support_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        rows = (
            self.household_ids,
            self.regions,
            self.great_britain,
            self.reference_band_d,
        )
        held = (
            all(isinstance(values, tuple) for values in rows)
            and len({len(values) for values in rows}) == 1
            and all(type(flag) is bool for flag in self.great_britain)
            and self.great_britain
            == tuple(_levies_council_tax(region) for region in self.regions)
            and all(
                isinstance(reference, Fraction) if flag else reference is None
                for reference, flag in zip(
                    self.reference_band_d, self.great_britain, strict=True
                )
            )
        )
        if not held:
            raise ValueError(
                f"{_WHAT} takes inputs holding one household_id, region, Great "
                "Britain flag and reference per row, a Fraction reference on "
                "every row whose region's nation levies council tax and none "
                "elsewhere; these do not."
            )
        if not isinstance(self.band_d, UKCouncilTaxBandD):
            raise TypeError(f"{_WHAT} prices bills at a UKCouncilTaxBandD.")
        if not (
            isinstance(self.support_sha256, Mapping)
            and all(
                isinstance(system, str) and isinstance(digest, str)
                for system, digest in self.support_sha256.items()
            )
        ):
            raise ValueError(
                f"{_WHAT} takes the supports' digests as strings keyed by system."
            )


def _require_columns(household: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in household.columns]
    if missing:
        raise ValueError(
            f"{_WHAT} reads household {missing}, which earlier stages write; the "
            "table lacks them (microcosm#931)."
        )


def uk_spi_copy_draw_frame(household: pd.DataFrame) -> pd.DataFrame:
    """The draw inputs of ``household``'s rows at the copy level.

    The narrow frame the SPI-copy area stage's draw leaves a copy at L3, one
    row per household in the table's row order and index: the region as the
    table holds it, and :data:`~microcosm.build.uk_runtime.atomic_area_conditioning.DECILE_INPUT_COLUMN`
    and :data:`~microcosm.build.uk_runtime.atomic_area_conditioning.BAND_INPUT_COLUMN`
    present and missing on every row, each in the graph's string dtype
    (``StringDtype("python")``). It reads ``region`` alone.

    Refused, naming the column: a table without ``region``, and a missing,
    blank or padded region.
    """

    if not isinstance(household, pd.DataFrame):
        raise TypeError(f"{_WHAT} reads the household table as a DataFrame.")
    _require_columns(household, (REGION_COLUMN,))
    regions = household[REGION_COLUMN].to_numpy(dtype=object)
    if not all(
        isinstance(region, str) and region and region.strip() == region
        for region in regions
    ):
        raise ValueError(
            f"{_WHAT} reads a nonempty {REGION_COLUMN!r} on every row it prices; "
            "some row lacks one."
        )
    string = pd.StringDtype(storage="python")
    missing = [pd.NA] * len(household)
    return pd.DataFrame(
        {
            REGION_COLUMN: pd.Series(
                [str(region) for region in regions], index=household.index, dtype=string
            ),
            DECILE_INPUT_COLUMN: pd.Series(
                missing, index=household.index, dtype=string
            ),
            BAND_INPUT_COLUMN: pd.Series(missing, index=household.index, dtype=string),
        },
        index=household.index,
    )


def uk_spi_housing_council_tax_inputs(
    household: pd.DataFrame,
    *,
    definition: Mapping[str, object],
    supports: Mapping[str, AtomicSupport],
    band_d: UKCouncilTaxBandD,
) -> UKSPIHousingCouncilTaxInputs:
    """The rows the SPI support channel copied, and each one's reference Band D.

    ``definition`` is the area draw's (G2,
    :func:`~microcosm.build.uk_runtime.atomic_area_conditioning.uk_conditioned_assignment_definition`)
    and ``supports`` the decoded conditioned supports of its systems. The
    rows are those whose ``household_support_channel`` is ``"spi"``, the
    housing shell's rule; they must be exactly the rows the support channel's
    lineage marks as its copies
    (:func:`~microcosm.build.uk_runtime.atomic_area_conditioning.uk_support_channel_copies`).
    Their systems come from the definition's region selectors on
    :func:`uk_spi_copy_draw_frame`, and each Great Britain system's rows take
    :func:`~microcosm.build.atomic_support_expectations.draw_cell_expectations`
    on that frame, with the system's constraints, its ``weight_by`` rule
    without the default and the default, over the local authority codes'
    2024-25 Band D: the call the council-tax localisation makes for the same
    rows, so each reference is the region's, by households. The digests are
    keyed by the supports given, never by the systems the rows fall in.

    Refused, naming columns and declared constants only: a ``band_d`` that is
    not a :class:`~microcosm.build.uk_runtime.council_tax_band_d.UKCouncilTaxBandD`;
    a missing column; supports that are not a decoded support for each system
    of the definition and no other; rows of the channel that are not exactly
    the lineage's copies, and what the lineage's reader refuses; every refusal
    of :func:`uk_spi_copy_draw_frame`; a region no system selects; and what
    the expectation refuses, a code with mass in a copy's cell and no Band D
    among them.
    """

    if not isinstance(band_d, UKCouncilTaxBandD):
        raise TypeError(f"{_WHAT} prices bills at a UKCouncilTaxBandD.")
    if not isinstance(household, pd.DataFrame):
        raise TypeError(f"{_WHAT} reads the household table as a DataFrame.")
    _require_columns(
        household,
        (
            _HOUSEHOLD_ID_COLUMN,
            UK_SPI_HOUSING_COUNCIL_TAX_CHANNEL_COLUMN,
            REGION_COLUMN,
        ),
    )
    spec = validate_assignment_spec(definition)
    if not (
        isinstance(supports, Mapping)
        and set(supports) == {system["id"] for system in spec["systems"]}
        and all(isinstance(support, AtomicSupport) for support in supports.values())
    ):
        raise ValueError(
            f"{_WHAT} reads a decoded conditioned support for every system of the "
            "area draw's definition, and no other."
        )
    channel = household[UK_SPI_HOUSING_COUNCIL_TAX_CHANNEL_COLUMN].astype(str)
    recipients = (
        channel.to_numpy(dtype=object) == UK_SPI_HOUSING_COUNCIL_TAX_RECIPIENT_CHANNEL
    )
    if not np.array_equal(recipients, uk_support_channel_copies(household)):
        raise ValueError(
            f"{_WHAT} prices the rows whose "
            f"{UK_SPI_HOUSING_COUNCIL_TAX_CHANNEL_COLUMN!r} is "
            f"{UK_SPI_HOUSING_COUNCIL_TAX_RECIPIENT_CHANNEL!r}, which must be "
            "exactly the rows the support channel's lineage marks as its copies; "
            "the two differ."
        )
    rows = np.flatnonzero(recipients)
    frame = uk_spi_copy_draw_frame(household.iloc[rows])
    regions = tuple(
        str(region) for region in frame[REGION_COLUMN].to_numpy(dtype=object)
    )
    great_britain = np.asarray(
        [_levies_council_tax(region) for region in regions], dtype=bool
    )
    systems = uk_draw_systems(frame, spec)
    reference = np.full(len(rows), None, dtype=object)
    for system in spec["systems"]:
        members = np.flatnonzero(great_britain & (systems == system["id"]))
        if not len(members):
            continue
        weight_by, default = _system_rules(system)
        try:
            means = draw_cell_expectations(
                supports[system["id"]],
                frame.iloc[members],
                constraint_columns={
                    constraint["input"]: constraint["support"]
                    for constraint in system["constraints"]
                },
                weight_by=weight_by,
                default_weight=default,
                value_column=UK_LOCAL_AUTHORITY_CODE_COLUMN,
                values=band_d.band_d,
            )
        except ValueError as error:
            raise ValueError(
                f"{_WHAT} takes the expected {UK_BAND_D_FISCAL_YEAR} Band D of each "
                f"copy's draw cell: {error}"
            ) from error
        table = np.empty(len(means), dtype=object)
        table[:] = means
        reference[members] = table
    return UKSPIHousingCouncilTaxInputs(
        household_ids=tuple(
            household[_HOUSEHOLD_ID_COLUMN].to_numpy(dtype=object)[rows].tolist()
        ),
        regions=regions,
        great_britain=tuple(bool(flag) for flag in great_britain.tolist()),
        reference_band_d=tuple(reference.tolist()),
        band_d=band_d,
        support_sha256=MappingProxyType(
            {system: supports[system].sha256 for system in sorted(supports)}
        ),
    )


# --- The bill -----------------------------------------------------------------------


def uk_spi_statutory_council_tax(
    *,
    regions: Sequence[str],
    bands: Sequence[str],
    single_adult: Sequence[bool],
    reference_band_d: Sequence[Fraction],
) -> np.ndarray:
    """The 2024-25 statute's bill on Great Britain rows, as float64.

    Row by row: ``float(uk_statutory_council_tax(reference, nation, band,
    single_adult=flag))``, the nation the region's, one correctly rounded
    conversion of the statute's exact rational, made once for each distinct
    reference, nation, band and flag. ``single_adult`` holds one bool per
    row. Before it calls the statute it refuses, with its own messages, a row
    whose region's nation levies no council tax and a band the region's
    nation does not price, so the statute's own refusals, which name a nation
    or a band, never fire on a row.

    Refused too: inputs of different lengths, and a flag that is not a bool;
    a reference the statute does not take (not an exact positive rational) is
    refused by the statute, naming its type only.
    """

    region_list = list(regions)
    band_list = list(bands)
    flags = list(single_adult)
    references = list(reference_band_d)
    if not len(region_list) == len(band_list) == len(flags) == len(references):
        raise ValueError(
            f"{_WHAT} prices one bill per row; the regions, bands, single-adult "
            "flags and references it is given differ in length."
        )
    if not all(isinstance(flag, bool | np.bool_) for flag in flags):
        raise ValueError(
            f"{_WHAT} reads the single-adult flag as one bool per row; some flag "
            "is not a bool."
        )
    if not all(_levies_council_tax(region) for region in region_list):
        raise ValueError(
            f"{_WHAT} prices by the statute only rows whose {REGION_COLUMN!r} is in "
            "a nation that levies council tax; some row's region is not."
        )
    nations = [UK_NATION_BY_REGION[region] for region in region_list]
    if not all(
        isinstance(band, str) and band in UK_COUNCIL_TAX_BAND_RATIOS[nation]
        for nation, band in zip(nations, band_list, strict=True)
    ):
        raise ValueError(
            f"{_WHAT} prices by the statute a {_BAND_COLUMN!r} its region's nation "
            "prices; some row's band is not one of them."
        )
    bills: dict[tuple[object, str, str, bool], float] = {}
    result = np.empty(len(region_list), dtype=np.float64)
    for position, key in enumerate(
        zip(
            references,
            nations,
            (str(band) for band in band_list),
            (bool(flag) for flag in flags),
            strict=True,
        )
    ):
        if key not in bills:
            bills[key] = float(
                uk_statutory_council_tax(key[0], key[1], key[2], single_adult=key[3])
            )
        result[position] = bills[key]
    return result


def _require_inputs(inputs: object) -> UKSPIHousingCouncilTaxInputs:
    if not isinstance(inputs, UKSPIHousingCouncilTaxInputs):
        raise TypeError(
            f"{_WHAT} reads inputs built by uk_spi_housing_council_tax_inputs."
        )
    return inputs


def _on_inputs(
    values: object, inputs: UKSPIHousingCouncilTaxInputs, name: str
) -> pd.Series:
    """``values``, a Series indexed by the inputs' household ids in their order."""

    if not isinstance(values, pd.Series):
        raise TypeError(f"{_WHAT} reads {name} as a Series indexed by household_id.")
    if values.index.tolist() != list(inputs.household_ids):
        raise ValueError(
            f"{_WHAT} reads {name} on the inputs' rows; its index is not the "
            "inputs' household_id values in their order."
        )
    return values


def _amounts_on_inputs(
    values: object, inputs: UKSPIHousingCouncilTaxInputs, name: str
) -> np.ndarray:
    """A float64 amount per row of the inputs, read bit for bit."""

    series = _on_inputs(values, inputs, name)
    if series.dtype != _FLOAT64:
        raise ValueError(
            f"{_WHAT} reads {name} as float64, bit for bit; it holds {series.dtype}."
        )
    return series.to_numpy(dtype=np.float64)


def _single_adults(values: object, inputs: UKSPIHousingCouncilTaxInputs) -> np.ndarray:
    """Where the flag is 1, as the statute's single-adult discount reads
    ``council_tax_single_adult_raw``."""

    series = _on_inputs(values, inputs, "single_adult")
    if not pd.api.types.is_numeric_dtype(series.dtype):
        raise ValueError(
            f"{_WHAT} reads single_adult as a numeric flag, one where the "
            f"household has a single adult; it holds {series.dtype}."
        )
    return series.to_numpy(dtype=float) == 1.0


def uk_spi_council_tax_bills(
    chained: pd.Series,
    *,
    inputs: UKSPIHousingCouncilTaxInputs,
    bands: pd.Series,
    single_adult: pd.Series,
) -> np.ndarray:
    """The bill the housing shell writes on each row of ``inputs``, as float64.

    ``chained`` is the shell's chained ``council_tax`` draw, ``bands`` its
    drawn ``council_tax_band`` and ``single_adult`` a numeric flag, 1 where
    the household has a single adult (as ``council_tax_single_adult_raw``
    reads), each a Series indexed by the inputs' household ids in their
    order. A Great Britain row takes :func:`uk_spi_statutory_council_tax` at
    its reference, nation, drawn band and flag; a Northern Ireland row takes
    its chained draw as given, bit for bit. The branch is the nation of the
    row's region, never a value of the draw. Returns the bills in the inputs'
    order.

    Refused, naming parameters, dtypes and declared constants only: inputs
    not built by :func:`uk_spi_housing_council_tax_inputs`; ``chained``,
    ``bands`` or ``single_adult`` not indexed by the inputs' household ids in
    their order; a ``chained`` that is not float64; a ``single_adult`` that is
    not numeric; and every refusal of :func:`uk_spi_statutory_council_tax`.
    """

    _require_inputs(inputs)
    amounts = _amounts_on_inputs(chained, inputs, "chained")
    drawn = _on_inputs(bands, inputs, "bands").to_numpy(dtype=object)
    single = _single_adults(single_adult, inputs)
    bills = amounts.copy()
    rows = np.flatnonzero(np.asarray(inputs.great_britain, dtype=bool))
    if len(rows):
        bills[rows] = uk_spi_statutory_council_tax(
            regions=[inputs.regions[row] for row in rows.tolist()],
            bands=drawn[rows].tolist(),
            single_adult=single[rows].tolist(),
            reference_band_d=[inputs.reference_band_d[row] for row in rows.tolist()],
        )
    return bills


# --- The checks and the evidence ----------------------------------------------------


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.array_equal(
            np.ascontiguousarray(left, dtype=np.float64).view(np.uint64),
            np.ascontiguousarray(right, dtype=np.float64).view(np.uint64),
        )
    )


def uk_spi_housing_council_tax_checks(
    chained: pd.Series,
    *,
    inputs: UKSPIHousingCouncilTaxInputs,
    bills: pd.Series,
    bands: pd.Series,
    single_adult: pd.Series,
    reduction: pd.Series,
    rebate: pd.Series,
) -> dict[str, bool]:
    """The checks on the bills written, as booleans, in the order of
    :data:`UK_SPI_HOUSING_COUNCIL_TAX_CHECKS`.

    ``chained``, ``bands`` and ``single_adult`` are what
    :func:`uk_spi_council_tax_bills` read; ``bills`` the ``council_tax``
    written, ``reduction`` each household's reported council tax reduction
    after the cap and ``rebate`` its ``council_tax_rebate``, each a Series
    indexed by the inputs' household ids in their order, the amounts float64:

    - ``bill_at_statute_of_reference``: every Great Britain bill is the
      statute at its reference, bit for bit;
    - ``northern_ireland_bill_as_chained``: every Northern Ireland bill is its
      chained draw, bit for bit;
    - ``reduction_within_bill``: every bill and reduction is finite and
      nonnegative, and every reduction is at most its bill;
    - ``rebate_equals_reduction``: every rebate equals its reduction.

    Refused as :func:`uk_spi_council_tax_bills` refuses, and an amount that
    is not float64 or not indexed by the inputs' household ids in their order.
    """

    _require_inputs(inputs)
    drawn = _amounts_on_inputs(chained, inputs, "chained")
    written = _amounts_on_inputs(bills, inputs, "bills")
    reduced = _amounts_on_inputs(reduction, inputs, "reduction")
    rebated = _amounts_on_inputs(rebate, inputs, "rebate")
    expected = uk_spi_council_tax_bills(
        chained, inputs=inputs, bands=bands, single_adult=single_adult
    )
    great_britain = np.asarray(inputs.great_britain, dtype=bool)
    return {
        "bill_at_statute_of_reference": _bitwise_equal(
            written[great_britain], expected[great_britain]
        ),
        "northern_ireland_bill_as_chained": _bitwise_equal(
            written[~great_britain], drawn[~great_britain]
        ),
        "reduction_within_bill": bool(
            np.all(
                np.isfinite(written)
                & (written >= 0.0)
                & np.isfinite(reduced)
                & (reduced >= 0.0)
                & (reduced <= written)
            )
        ),
        "rebate_equals_reduction": bool(np.all(rebated == reduced)),
    }


def uk_spi_housing_council_tax_evidence(
    inputs: UKSPIHousingCouncilTaxInputs, *, checks: Mapping[str, bool]
) -> dict[str, object]:
    """The block the housing shell's receipt carries under
    :data:`UK_SPI_HOUSING_COUNCIL_TAX_EVIDENCE`.

    The method, the reference, the copy level, the two Band D resources with
    their digests and the Scottish feed's ``source_commit`` (read from the
    feed as
    :meth:`~microcosm.build.uk_runtime.council_tax_band_d.UKCouncilTaxBandD.evidence`
    reads it, a string, or ``None`` for a Band D supplied without a feed;
    never the feed itself), the supports' digests as the inputs hold them,
    and ``checks`` as booleans. Every leaf is a string, a boolean or that
    absent commit: no number, no count, ratio, Band D, bill, region or
    nation. The keys are the same whatever systems, nations or clones a frame
    holds.

    Refused: inputs not built by :func:`uk_spi_housing_council_tax_inputs`;
    ``checks`` whose keys are not those of
    :data:`UK_SPI_HOUSING_COUNCIL_TAX_CHECKS`, or a check that is not a bool;
    a Band D digest that is not a string; and a feed's ``source_commit`` that
    is neither a string nor absent.
    """

    _require_inputs(inputs)
    if not (
        isinstance(checks, Mapping)
        and set(checks) == set(UK_SPI_HOUSING_COUNCIL_TAX_CHECKS)
        and all(isinstance(value, bool | np.bool_) for value in checks.values())
    ):
        raise ValueError(
            f"{_WHAT} records the checks {list(UK_SPI_HOUSING_COUNCIL_TAX_CHECKS)} "
            "and no other, each a bool."
        )
    band_d = inputs.band_d
    if not (
        isinstance(band_d.england_wales_sha256, str)
        and isinstance(band_d.scotland_sha256, str)
    ):
        raise ValueError(f"{_WHAT} records each Band D resource's digest as a string.")
    source_commit = band_d.scotland_source_fact_feed.get("source_commit")
    if source_commit is not None and not isinstance(source_commit, str):
        raise ValueError(
            f"{_WHAT} records the Scottish Band D feed's source_commit as a string; "
            "the feed holds another type."
        )
    return {
        "method": UK_SPI_HOUSING_COUNCIL_TAX_METHOD,
        "reference": UK_SPI_HOUSING_COUNCIL_TAX_REFERENCE,
        "copy_draw_level": UK_SPI_COPY_DRAW_LEVEL,
        "band_d": {
            "england_wales": {
                "resource": UK_BAND_D_ENGLAND_WALES_RESOURCE,
                "sha256": band_d.england_wales_sha256,
            },
            "scotland": {
                "resource": SCOTLAND_BAND_D_RESOURCE,
                "sha256": band_d.scotland_sha256,
                "source_commit": source_commit,
            },
        },
        "support_sha256": dict(sorted(inputs.support_sha256.items())),
        "checks": {
            check: bool(checks[check]) for check in UK_SPI_HOUSING_COUNCIL_TAX_CHECKS
        },
    }
