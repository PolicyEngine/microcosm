"""Assign a main asset type, a residential-property flag and BADR claims to CGT taxpayers.

The amounts redraw (``cgt_imputation``) places net gains on persons; nothing
says what was sold. Three HMRC surfaces do, and all are vendored verbatim
from the pinned Chronicle feed into ``hmrc_cgt_asset_type_facts.json``:

- **Table 8 (administrative, 2024-25).** Table 8a counts UK-resident
  taxpayers reporting residential property disposals, with their gains and
  tax, across the CGT on UK Property service and Self Assessment; Table 8b
  splits the service channel by taxpayer type (individuals, trusts). The
  national CGT targets are individuals only, so the Table 8a totals are
  restated on the individuals basis by the Table 8b individuals/all share
  for the same measure — the same declared assumption the Table 5 region
  targets make — and those two numbers are what the residential flag is
  solved to and what the residential targets bind (microcosm#725).
- **Table 7 (sample-based, 2023-24).** Disposals, proceeds and gains by asset
  type. It counts disposals rather than taxpayers, is a year older, and
  includes trusts and non-UK assets, so it is fenced from calibration; it
  seeds the composition of the non-residential remainder and the stage
  reports achieved against it as a diagnostic only.
- **Table 4.1 (administrative, 2024-25).** Individuals claiming Business
  Asset Disposal Relief or Investors' Relief, with the gains they claimed
  on and the tax charged at the relief rate, by band of qualifying gain
  (microcosm#1014). The engine charges ``capital_gains_badr`` at the relief
  rate (PolicyEngine/policyengine-uk#1861); without it every such gain is
  charged at the main rates.

Three declared mechanisms, in order:

1. **Residential flag.** Among liable gainers (net gains above the annual
   exempt amount, the national taxpayer proxy) the probability of holding
   a residential disposal is logistic in centred log gains with a stock
   shift, ``p = 1 / (1 + exp(-(a + b (log g - c) + s h)))`` with ``c`` the
   weighted mean log gain of the liable population, ``h`` one for a gainer
   who shows residential stock (another residential property in the
   household, or property income) and ``s`` the declared stock log-odds
   (microcosm#1014). ``(a, b)`` are solved by nested bisection so the
   household-weighted expected count and gains of flagged persons equal the
   individuals-basis Table 8a totals: residential gainers are many and
   small relative to the whole (a fifth of the gains on a third of the
   taxpayers), which the negative slope carries. Flags are then realised by
   a weighted systematic walk in ascending gain order with one seeded
   offset: the walk flags a person whenever the expected weight owed so far
   reaches that person's weight, so the flagged weight tracks the expected
   weight within one person's weight everywhere along the gains axis, and
   both the realised count and the realised gains sit close to their
   expectations on frames whose liable rows are few and heavy. The
   Bernoulli sigma of each is reported as the envelope a gate can bound
   with. A flagged person's whole net
   gain is attributed to residential property (``capital_gains_residential_property``);
   a taxpayer with both residential and other disposals is not split.
2. **BADR and Investors' Relief claims.** Among liable gainers not flagged
   residential, each Table 4.1 band draws its claimants from the persons
   whose net gain falls inside the band's range. Below the lifetime limit a
   claimant's whole net gain qualifies, so the band's claimants are solved
   like the residential flag: a logistic in centred log gains whose
   intercept matches the band's published claimant count and whose slope
   matches its published qualifying gains, realised by the same weighted
   systematic walk with one seeded offset per band. Each band's logistic
   carries the same stock shift for a gainer who shows business stock
   (household corporate wealth, or self-employment income). The open top
   band holds claimants at the lifetime limit: its probability is flat in
   gains and varies only with the stock shift, its intercept matches the
   count, the count is the published gains over the limit (the published
   count is rounded to the nearest thousand), and the qualifying amount is
   the limit. The whole-gain assumption reproduces
   Table 4.1's tax column to within half a percent; Table 4.1 counts gains
   before losses while the frame carries net gains, and Investors' Relief
   is merged with BADR as in the engine's input.
3. **Main asset type.** Every liable gainer not flagged residential draws
   one of five Table 7 types from a size-tilted categorical: the type
   weight times a log-normal kernel in log gains centred on the type's
   Table 7 mean gain per disposal (listed shares near £5,500, unlisted
   shares near £115,000) with one declared log-scale width, times the
   declared stock floor where the type's stock signal is absent (listed
   shares: a stocks-and-shares ISA or dividends; unlisted shares: business
   stock; other financial assets: financial wealth). A BADR or
   Investors' Relief claimant draws only from the three types the reliefs
   apply to (unlisted shares, business land and buildings, other
   non-financial assets), so the type follows the claim. The type weights
   are fitted, with that restriction, by multiplicative updates until the
   household-weighted gains shares by type reproduce Table 7's
   non-residential gains shares, and the fit refuses if it does not
   converge; one seeded uniform per person realises the draw. The
   composition by gain band is reported for the evidence note; nothing is
   fitted to it.

Non-gainers carry ``none``; positive gains at or below the annual exempt
amount carry ``sub_aea`` (Table 8 describes taxpayers with a liability and
the sub-AEA remainder is never a taxpayer). Household weights pass through
untouched and the stage appends a mass-conservation receipt the terminal
family gate requires.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.cgt_imputation import (
    UKCGTPolicyParameters,
    uk_cgt_policy_parameters,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    _band_bounds_from_value_id,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import (
    load_vendored_resource,
    verify_vendored_resource_feed_identity,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "CGT_ASSET_TYPE_COLUMN",
    "CGT_ASSET_TYPE_DOMAIN",
    "CGT_ASSET_TYPE_LOG_SIGMA",
    "CGT_ASSET_TYPE_NONE",
    "CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES",
    "CGT_ASSET_TYPE_RESIDENTIAL",
    "CGT_ASSET_TYPE_SUB_AEA",
    "CGT_BADR_ELIGIBLE_TYPES",
    "CGT_BADR_FLAG_SEED",
    "CGT_BADR_GAINS_COLUMN",
    "CGT_CLAIMANT_STATUSES",
    "CGT_RESIDENTIAL_GAINS_COLUMN",
    "HMRCCGTAssetTypeFacts",
    "HMRCCGTBADRBand",
    "HMRC_CGT_ASSET_TYPE_RECORD_SETS",
    "HMRC_CGT_ASSET_TYPE_RESOURCE",
    "HMRC_CGT_TABLE4_BAND_LOWER_BOUNDS",
    "HMRC_CGT_TABLE4_SOURCE_SHA256",
    "HMRC_CGT_TABLE7_SOURCE_SHA256",
    "HMRC_CGT_TABLE8_SOURCE_SHA256",
    "UKCGTAssetTypeStageTransform",
    "UKCGTAssetTypeSummary",
    "UKCGTBADRParameters",
    "UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON",
    "UK_CGT_ASSET_TYPE_STAGE_NAME",
    "assign_uk_cgt_asset_types",
    "claimant_status_share_targets",
    "load_hmrc_cgt_asset_type_facts",
    "uk_cgt_asset_type_stage_transform",
    "uk_cgt_badr_parameters",
]

UK_CGT_ASSET_TYPE_STAGE_NAME = "hmrc_cgt_asset_type_spine"
UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON = (
    "CGT asset-type assignment on the source spine: household weights pass "
    "through unchanged and total household mass is conserved."
)

#: The per-concern vendored resource (Tables 4.1, 7, 8a and 8b). Regenerated
#: from the pinned Chronicle feed by ``tools/vendor_uk_ledger_facts.py``.
HMRC_CGT_ASSET_TYPE_RESOURCE = "hmrc_cgt_asset_type_facts.json"

#: Chronicle record sets the stage reads, as the manifest declares them.
_TABLE8A_RECORD_SET_ID = "hmrc.cgt_table8_2026.table8a.ty2024"
_TABLE8B_RECORD_SET_ID = "hmrc.cgt_table8_2026.table8b.ty2024"
_TABLE7_CATEGORY_RECORD_SET_ID = "hmrc.cgt_table7_2026.asset_category.ty2023"
_TABLE7_TOTAL_RECORD_SET_ID = "hmrc.cgt_table7_2026.asset_category_total.ty2023"
_TABLE7_FINANCIAL_RECORD_SET_ID = "hmrc.cgt_table7_2026.financial_asset_type.ty2023"
_TABLE7_NON_FINANCIAL_RECORD_SET_ID = (
    "hmrc.cgt_table7_2026.non_financial_asset_type.ty2023"
)
_TABLE4_BANDS_RECORD_SET_ID = "hmrc.cgt_badr_ir_2026.table4_1.ty2024.main"
_TABLE4_TRUSTS_RECORD_SET_ID = "hmrc.cgt_badr_ir_2026.table4_1.ty2024.trusts_total"
_TABLE4_ALL_RECORD_SET_ID = "hmrc.cgt_badr_ir_2026.table4_1.ty2024.all_total"
HMRC_CGT_ASSET_TYPE_RECORD_SETS: tuple[str, ...] = (
    _TABLE8A_RECORD_SET_ID,
    _TABLE8B_RECORD_SET_ID,
    _TABLE7_CATEGORY_RECORD_SET_ID,
    _TABLE7_TOTAL_RECORD_SET_ID,
    _TABLE7_FINANCIAL_RECORD_SET_ID,
    _TABLE7_NON_FINANCIAL_RECORD_SET_ID,
    _TABLE4_BANDS_RECORD_SET_ID,
    _TABLE4_TRUSTS_RECORD_SET_ID,
    _TABLE4_ALL_RECORD_SET_ID,
)

#: The publisher workbooks every vendored row must trace back to.
HMRC_CGT_TABLE8_SOURCE_SHA256 = (
    "fefe621cab478ca14b06aaeabe0ac2db95a4912b1b40ac029fbbd5ea3fa34952"
)
HMRC_CGT_TABLE7_SOURCE_SHA256 = (
    "a27ad79c3c67178e8b808c0561201dd6585b472d7919c0043fdc44142460b2bd"
)
HMRC_CGT_TABLE4_SOURCE_SHA256 = (
    "7bd0098913e44e3c79a533690cd25ee08f627cd72935c46a874a7eab5442d5ca"
)

#: Table 4.1's bands of qualifying gain (lower limits, GBP); the last is open.
HMRC_CGT_TABLE4_BAND_LOWER_BOUNDS: tuple[int, ...] = (
    0,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
)
_TABLE4_BAND_DIMENSION = "cgt_badr_ir_gain_band"
_TABLE4_TOTAL_BAND = "total"

CGT_ASSET_TYPE_COLUMN = "capital_gains_asset_type"
CGT_RESIDENTIAL_GAINS_COLUMN = "capital_gains_residential_property"
CGT_BADR_GAINS_COLUMN = "capital_gains_badr"

#: Value ids shared with Chronicle's Table 7 ``cgt_asset_type`` dimension,
#: plus the two states Table 7 does not describe.
CGT_ASSET_TYPE_NONE = "none"
CGT_ASSET_TYPE_SUB_AEA = "sub_aea"
CGT_ASSET_TYPE_RESIDENTIAL = "residential_land_buildings"
CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES: tuple[str, ...] = (
    "listed_shares",
    "unlisted_shares",
    "other_financial_assets",
    "agricultural_commercial_industrial_land_buildings",
    "other_non_financial_assets",
)
CGT_ASSET_TYPE_DOMAIN: tuple[str, ...] = (
    CGT_ASSET_TYPE_NONE,
    CGT_ASSET_TYPE_SUB_AEA,
    CGT_ASSET_TYPE_RESIDENTIAL,
    *CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES,
)

#: The Table 7 types Business Asset Disposal Relief and Investors' Relief
#: apply to: shares in a trading company, and business assets. A claimant's
#: main asset type is drawn from these only.
CGT_BADR_ELIGIBLE_TYPES: tuple[str, ...] = (
    "unlisted_shares",
    "agricultural_commercial_industrial_land_buildings",
    "other_non_financial_assets",
)

#: Log-scale width of the size kernel around each type's mean gain per
#: disposal. Wide enough that every type reaches every band; the fitted
#: type weights, not the width, carry the composition.
CGT_ASSET_TYPE_LOG_SIGMA = 1.5

#: The BADR seed is combined with the build period, as the amounts stage
#: does; distinct from the amounts stage's 552 so the two stages never share
#: a stream. The residential flag draws nothing since microcosm#1063: the
#: ``cgt_residential_split`` stage carries its probability as weight.
CGT_BADR_FLAG_SEED = 555

_BISECTION_ITERATIONS = 200
_SHARE_FIT_ITERATIONS = 200
_SHARE_FIT_TOLERANCE = 1e-6
_SOLVE_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Vendored facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HMRCCGTTable7Type:
    """One Table 7 asset-type row: disposals, proceeds and gains (2023-24)."""

    asset_type: str
    category: str
    disposals: float
    proceeds: float
    gains: float

    @property
    def mean_gain_per_disposal(self) -> float:
        return self.gains / self.disposals


@dataclass(frozen=True)
class HMRCCGTBADRBand:
    """One Table 4.1 row: individuals claiming BADR or Investors' Relief.

    ``upper_bound`` is the exclusive edge (one past the published inclusive
    limit) and ``None`` for the open top band; ``tax`` is ``None`` where the
    publisher suppressed the cell.
    """

    lower_bound: int
    upper_bound: int | None
    taxpayers: float
    gains: float
    tax: float | None

    @property
    def label(self) -> str:
        if self.upper_bound is None:
            return f"GBP {self.lower_bound:,} and over"
        return f"GBP {self.lower_bound:,} to {self.upper_bound - 1:,}"


@dataclass(frozen=True)
class HMRCCGTAssetTypeFacts:
    """The vendored Table 4.1, 7, 8a and 8b rows, typed."""

    table8a_taxpayers_total: float
    table8a_gains_total: float
    table8a_disposals_total: float
    table8a_tax_total: float
    table8b_individuals_taxpayers: float
    table8b_individuals_gains: float
    table8b_all_taxpayers: float
    table8b_all_gains: float
    table7_types: tuple[HMRCCGTTable7Type, ...]
    table7_total_gains: float
    table7_total_disposals: float
    table4_bands: tuple[HMRCCGTBADRBand, ...]
    table4_individuals_taxpayers: float
    table4_individuals_gains: float
    table4_individuals_tax: float
    table4_trusts_gains: float
    table4_trusts_tax: float
    table4_all_taxpayers: float
    table4_all_gains: float
    table4_all_tax: float
    resource: str
    resource_sha256: str
    source_commit: str

    def individuals_share(self, measure: str) -> float:
        """Table 8b individuals ÷ all for ``taxpayers`` or ``gains``."""

        numerator = getattr(self, f"table8b_individuals_{measure}")
        denominator = getattr(self, f"table8b_all_{measure}")
        if denominator <= 0:
            raise ValueError(f"Table 8b all-taxpayer {measure} must be positive.")
        return numerator / denominator

    @property
    def residential_taxpayers_individuals_basis(self) -> float:
        return self.table8a_taxpayers_total * self.individuals_share("taxpayers")

    @property
    def residential_gains_individuals_basis(self) -> float:
        return self.table8a_gains_total * self.individuals_share("gains")

    def table7_type(self, asset_type: str) -> HMRCCGTTable7Type:
        for row in self.table7_types:
            if row.asset_type == asset_type:
                return row
        raise KeyError(f"No Table 7 row for asset type {asset_type!r}.")

    def non_residential_gains_shares(self) -> dict[str, float]:
        """Table 7 gains shares over the five non-residential types."""

        gains = {
            asset_type: self.table7_type(asset_type).gains
            for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        }
        total = sum(gains.values())
        return {asset_type: value / total for asset_type, value in gains.items()}


def _record_set_id(row: Mapping[str, Any]) -> str:
    return str((row.get("layout") or {}).get("record_set_id") or "")


def _value(row: Mapping[str, Any]) -> float:
    value = row.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"Vendored CGT asset-type row {row.get('aggregate_fact_key')!r} has no "
            "numeric value."
        )
    return float(value)


def _source_sha256(row: Mapping[str, Any]) -> str:
    return str((row.get("source") or {}).get("source_sha256") or "")


def load_hmrc_cgt_asset_type_facts(
    resource: str = HMRC_CGT_ASSET_TYPE_RESOURCE,
    *,
    verify_feed_identity: bool = True,
) -> HMRCCGTAssetTypeFacts:
    """Type the vendored rows; refuse a stale feed, source or roster."""

    payload = load_vendored_resource(resource)
    if verify_feed_identity:
        verify_vendored_resource_feed_identity(payload)
    digest = hashlib.sha256(
        files("microcosm.build.uk").joinpath(resource).read_bytes()
    ).hexdigest()
    rows = list(payload["rows"])

    table8a: dict[str, float] = {}
    table8b: dict[tuple[str, str], float] = {}
    table7: dict[tuple[str, str], dict[str, float]] = {}
    table7_total: dict[str, float] = {}
    table4_bands: dict[str, dict[str, float]] = {}
    table4_totals: dict[tuple[str, str], float] = {}
    for row in rows:
        record_set = _record_set_id(row)
        measure = str(row.get("measure_id") or "")
        dimensions = row.get("dimensions") or {}
        if record_set == _TABLE8A_RECORD_SET_ID:
            if _source_sha256(row) != HMRC_CGT_TABLE8_SOURCE_SHA256:
                raise ValueError(
                    "Table 8a rows trace to a workbook other than the pinned one."
                )
            table8a[measure] = _value(row)
        elif record_set == _TABLE8B_RECORD_SET_ID:
            if _source_sha256(row) != HMRC_CGT_TABLE8_SOURCE_SHA256:
                raise ValueError(
                    "Table 8b rows trace to a workbook other than the pinned one."
                )
            if dimensions.get("channel") != "uk_property_service":
                raise ValueError("Table 8b rows must describe the UK Property service.")
            taxpayer_type = str(dimensions.get("taxpayer_type") or "all")
            table8b[(taxpayer_type, measure)] = _value(row)
        elif record_set in (
            _TABLE7_FINANCIAL_RECORD_SET_ID,
            _TABLE7_NON_FINANCIAL_RECORD_SET_ID,
        ):
            if _source_sha256(row) != HMRC_CGT_TABLE7_SOURCE_SHA256:
                raise ValueError(
                    "Table 7 rows trace to a workbook other than the pinned one."
                )
            asset_type = dimensions.get("cgt_asset_type")
            if asset_type is None:
                continue  # the category subtotal restates the Table 7_1 rows
            category = str(dimensions.get("cgt_asset_category") or "")
            table7.setdefault((str(asset_type), category), {})[measure] = _value(row)
        elif record_set == _TABLE7_TOTAL_RECORD_SET_ID:
            if _source_sha256(row) != HMRC_CGT_TABLE7_SOURCE_SHA256:
                raise ValueError(
                    "Table 7 rows trace to a workbook other than the pinned one."
                )
            table7_total[measure] = _value(row)
        elif record_set == _TABLE7_CATEGORY_RECORD_SET_ID:
            continue
        elif record_set in (
            _TABLE4_BANDS_RECORD_SET_ID,
            _TABLE4_TRUSTS_RECORD_SET_ID,
            _TABLE4_ALL_RECORD_SET_ID,
        ):
            if _source_sha256(row) != HMRC_CGT_TABLE4_SOURCE_SHA256:
                raise ValueError(
                    "Table 4 rows trace to a workbook other than the pinned one."
                )
            band = str(dimensions.get(_TABLE4_BAND_DIMENSION) or "")
            taxpayer_type = str(dimensions.get("taxpayer_type") or "")
            if band == _TABLE4_TOTAL_BAND:
                table4_totals[(taxpayer_type, measure)] = _value(row)
            elif record_set == _TABLE4_BANDS_RECORD_SET_ID and (
                taxpayer_type == "individuals"
            ):
                table4_bands.setdefault(band, {})[measure] = _value(row)
            else:
                raise ValueError(
                    f"Table 4 row {row.get('aggregate_fact_key')!r} is neither an "
                    "individuals band nor a total."
                )
        else:
            raise ValueError(
                f"Vendored CGT asset-type resource carries an undeclared record set "
                f"{record_set!r}."
            )

    try:
        facts = HMRCCGTAssetTypeFacts(
            table8a_taxpayers_total=table8a["taxpayers_total"],
            table8a_gains_total=table8a["gains_total"],
            table8a_disposals_total=table8a["disposals_total"],
            table8a_tax_total=table8a["tax_total"],
            table8b_individuals_taxpayers=table8b[("individuals", "taxpayers")],
            table8b_individuals_gains=table8b[("individuals", "gains")],
            table8b_all_taxpayers=table8b[("all", "taxpayers")],
            table8b_all_gains=table8b[("all", "gains")],
            table7_types=tuple(
                HMRCCGTTable7Type(
                    asset_type=asset_type,
                    category=category,
                    disposals=cells["disposals"],
                    proceeds=cells["disposal_proceeds"],
                    gains=cells["gains"],
                )
                for (asset_type, category), cells in sorted(table7.items())
            ),
            table7_total_gains=table7_total["gains"],
            table7_total_disposals=table7_total["disposals"],
            table4_bands=_table4_bands(table4_bands),
            table4_individuals_taxpayers=table4_totals[("individuals", "taxpayers")],
            table4_individuals_gains=table4_totals[("individuals", "gains")],
            table4_individuals_tax=table4_totals[("individuals", "tax")],
            table4_trusts_gains=table4_totals[("trusts", "gains")],
            table4_trusts_tax=table4_totals[("trusts", "tax")],
            table4_all_taxpayers=table4_totals[("all", "taxpayers")],
            table4_all_gains=table4_totals[("all", "gains")],
            table4_all_tax=table4_totals[("all", "tax")],
            resource=resource,
            resource_sha256=digest,
            source_commit=str(payload["source_fact_feed"]["source_commit"]),
        )
    except KeyError as missing:
        raise ValueError(f"Vendored CGT asset-type resource lacks {missing}.") from None
    expected_types = {CGT_ASSET_TYPE_RESIDENTIAL, *CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES}
    actual_types = {row.asset_type for row in facts.table7_types}
    if actual_types != expected_types:
        raise ValueError(
            "Table 7 asset-type roster drifted: expected "
            f"{sorted(expected_types)}, got {sorted(actual_types)}."
        )
    for row in facts.table7_types:
        if row.disposals <= 0 or row.gains <= 0:
            raise ValueError(f"Table 7 {row.asset_type} must have positive rows.")
    if facts.table8b_individuals_taxpayers > facts.table8b_all_taxpayers:
        raise ValueError("Table 8b individuals cannot exceed all taxpayers.")
    if facts.table8b_individuals_gains > facts.table8b_all_gains:
        raise ValueError("Table 8b individual gains cannot exceed all gains.")
    _validate_table4(facts)
    return facts


#: Table 4.1 publishes taxpayers to the nearest thousand and amounts to the
#: nearest million, so each published row carries half a unit of rounding.
_TABLE4_COUNT_ROUNDING = 500.0
_TABLE4_AMOUNT_ROUNDING = 500_000.0


def _table4_bands(
    cells: Mapping[str, Mapping[str, float]],
) -> tuple[HMRCCGTBADRBand, ...]:
    bands = []
    for value_id, measures in cells.items():
        lower, upper = _band_bounds_from_value_id(value_id, prefix="gain")
        try:
            bands.append(
                HMRCCGTBADRBand(
                    lower_bound=lower,
                    upper_bound=upper,
                    taxpayers=measures["taxpayers"],
                    gains=measures["gains"],
                    tax=measures.get("tax"),
                )
            )
        except KeyError as missing:
            raise ValueError(
                f"Table 4 band {value_id!r} lacks its published {missing}."
            ) from None
    return tuple(sorted(bands, key=lambda band: band.lower_bound))


def _validate_table4(facts: HMRCCGTAssetTypeFacts) -> None:
    """Refuse a Table 4.1 roster, edge set or total the stage was not built on."""

    bands = facts.table4_bands
    lowers = tuple(band.lower_bound for band in bands)
    if lowers != HMRC_CGT_TABLE4_BAND_LOWER_BOUNDS:
        raise ValueError(
            "Table 4.1 band roster drifted: expected lower limits "
            f"{HMRC_CGT_TABLE4_BAND_LOWER_BOUNDS}, got {lowers}."
        )
    for band, following in zip(bands, bands[1:], strict=False):
        if band.upper_bound != following.lower_bound:
            raise ValueError(
                f"Table 4.1 band {band.label} does not meet the next band's edge."
            )
    if bands[-1].upper_bound is not None:
        raise ValueError("Table 4.1's top band must be open.")
    for band in bands:
        if band.taxpayers <= 0.0 or band.gains <= 0.0:
            raise ValueError(f"Table 4.1 band {band.label} must have positive rows.")
    for label, published, total, rounding in (
        (
            "taxpayers",
            sum(band.taxpayers for band in bands),
            facts.table4_individuals_taxpayers,
            _TABLE4_COUNT_ROUNDING,
        ),
        (
            "gains",
            sum(band.gains for band in bands),
            facts.table4_individuals_gains,
            _TABLE4_AMOUNT_ROUNDING,
        ),
    ):
        if abs(published - total) > rounding * (len(bands) + 1):
            raise ValueError(
                f"Table 4.1 band {label} sum {published} does not reconcile with "
                f"the individuals total {total} within publication rounding."
            )


# ---------------------------------------------------------------------------
# BADR schedule parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UKCGTBADRParameters:
    """The BADR schedule the claims are sized to: lifetime limit and rate.

    Read from the policyengine-uk parameter tree by
    :func:`uk_cgt_badr_parameters`, or constructed directly in tests and the
    engine-free parity fixture. The rate is reported, not used by the draw.
    """

    lifetime_limit: float
    rate: float
    instant: str
    source: str


def uk_cgt_badr_parameters(build_period: int | str) -> UKCGTBADRParameters:
    """Read the BADR lifetime limit and rate from the policyengine-uk tree.

    Read from the raw dated parameter files at 1 June of the tax year
    starting in ``build_period``, the instant the stage's other policy
    parameters use. The schedule first appears in policyengine-uk 2.99.2
    (PolicyEngine/policyengine-uk#1861). Requires the ``uk`` extra; the import
    is deferred so the base package does not import policyengine-uk.
    """

    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "uk_cgt_badr_parameters requires the microcosm-build 'uk' extra "
            "(policyengine-uk)."
        ) from exc

    parameters_dir = Path(policyengine_uk.__file__).parent / "parameters"
    parameters = ParameterNode(directory_path=str(parameters_dir))
    instant = f"{int(build_period)}-06-01"
    badr = parameters.gov.hmrc.cgt.badr
    return UKCGTBADRParameters(
        lifetime_limit=float(badr.lifetime_limit(instant)),
        rate=float(badr.rate(instant)),
        instant=instant,
        source="policyengine-uk parameters "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


# ---------------------------------------------------------------------------
# Residential flag
# ---------------------------------------------------------------------------


def _logistic(
    log_gains: np.ndarray, a: float, b: float, offset: np.ndarray | float = 0.0
) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a + b * log_gains + offset)))


#: A flag or a type implies the gainer holds the stock disposed of
#: (microcosm#1014). The residential flag and each BADR band add this shift
#: (odds times five) to the log-odds of a gainer who shows the implied stock;
#: the solves still meet every published total, so the shift moves who is
#: drawn, not how many or how much. On the licensed spine it took BADR
#: claimants with neither corporate wealth nor self-employment income from
#: 51.1 to 27.6 percent, and residential gainers with neither another
#: residential property nor property income from 82.6 to 74.1 percent.
CGT_STOCK_LOG_ODDS = 1.6

#: The main-type kernel is scaled by this factor for a type whose declared
#: stock signal the gainer lacks. On the licensed spine, with the shift above,
#: it took unlisted-shares gainers without business stock from 49.9 to 26.1
#: percent and listed-shares gainers with neither an ISA nor dividends from
#: 62.0 to 47.9 percent. Business land and buildings and other non-financial
#: assets declare no signal and are not scaled: no gainer typed business land
#: holds non-residential property or land on the spine, so that stock cannot
#: be read there.
CGT_STOCK_TYPE_FLOOR = 0.2

CGT_RESIDENTIAL_STOCK_SIGNAL = (
    "household other residential property value above zero, or the person's "
    "property income above zero"
)
CGT_BUSINESS_STOCK_SIGNAL = (
    "household corporate wealth above zero, or the person's self-employment "
    "income above zero"
)
CGT_TYPE_STOCK_SIGNALS: dict[str, str] = {
    "listed_shares": (
        "household stocks-and-shares ISA above zero, or the person's dividend "
        "income above zero"
    ),
    "unlisted_shares": CGT_BUSINESS_STOCK_SIGNAL,
    "other_financial_assets": "household gross financial wealth above zero",
}
CGT_STOCK_HOUSEHOLD_COLUMNS: tuple[str, ...] = (
    "other_residential_property_value",
    "corporate_wealth",
    "stocks_and_shares_isa",
    "gross_financial_wealth",
)
CGT_STOCK_PERSON_COLUMNS: tuple[str, ...] = (
    "property_income",
    "self_employment_income",
    "dividend_income",
)


def _stock_signals(person: pd.DataFrame, household: pd.DataFrame) -> dict[str, object]:
    """Which persons show the stock each flag or type implies.

    A household column is shared by every member, so a signal read from one
    marks all of them; a person column marks only that person.
    """

    missing = [c for c in CGT_STOCK_HOUSEHOLD_COLUMNS if c not in household.columns] + [
        c for c in CGT_STOCK_PERSON_COLUMNS if c not in person.columns
    ]
    if missing:
        raise ValueError(
            "The CGT asset-type stage conditions flags and types on the stocks "
            f"they imply; the frame lacks {missing}."
        )
    indexed = household.set_index("household_id")

    def finite(column: str, values: np.ndarray) -> np.ndarray:
        if not np.isfinite(values).all():
            raise ValueError(
                f"The stock signal column {column} must be finite for every person."
            )
        return values

    def hh(column: str) -> np.ndarray:
        values = pd.to_numeric(indexed[column], errors="raise")
        return finite(
            column, person["person_household_id"].map(values).to_numpy(dtype=float)
        )

    def pp(column: str) -> np.ndarray:
        return finite(
            column, pd.to_numeric(person[column], errors="raise").to_numpy(dtype=float)
        )

    residential = (hh("other_residential_property_value") > 0) | (
        pp("property_income") > 0
    )
    business = (hh("corporate_wealth") > 0) | (pp("self_employment_income") > 0)
    by_type = {
        "listed_shares": (hh("stocks_and_shares_isa") > 0)
        | (pp("dividend_income") > 0),
        "unlisted_shares": business,
        "other_financial_assets": hh("gross_financial_wealth") > 0,
    }
    factor = np.column_stack(
        [
            np.where(by_type[name], 1.0, CGT_STOCK_TYPE_FLOOR)
            if name in by_type
            else np.ones(len(person))
            for name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        ]
    )
    return {
        "residential": residential,
        "business": business,
        "by_type": by_type,
        "type_factor": factor,
    }


def _weighted_share(
    weights: np.ndarray, rows: np.ndarray, holds: np.ndarray
) -> float | None:
    mass = float(weights[rows].sum())
    return float(weights[rows & holds].sum() / mass) if mass > 0 else None


def _centred_log_gains(
    gains: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    """Log gains less their weighted mean, so the intercept stays O(1)."""

    log_gains = np.log(gains)
    centre = float((weights * log_gains).sum() / weights.sum())
    return log_gains - centre, centre


def _solve_a(
    log_gains: np.ndarray,
    weights: np.ndarray,
    *,
    b: float,
    count_target: float,
    offset: np.ndarray | float = 0.0,
) -> float:
    """The intercept that puts the expected weighted count on target at slope b."""

    low, high = -400.0, 400.0
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        expected = float((weights * _logistic(log_gains, mid, b, offset)).sum())
        if expected < count_target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def solve_residential_logistic(
    gains: np.ndarray,
    weights: np.ndarray,
    *,
    count_target: float,
    gains_target: float,
    offset: np.ndarray | float = 0.0,
) -> tuple[float, float, float]:
    """Solve (a, b, c) so expected weighted count and gains hit both targets.

    ``c`` is the weighted mean log gain the logistic is centred on. For each
    trial slope the intercept is bisected onto the count; the slope is then
    bisected onto the gains, which rise monotonically in the slope at a
    fixed count. Refuses targets the frame cannot reach: more taxpayers or
    gains than the liable population holds, or a mean gain outside what a
    logistic tilt of these persons can produce.
    """

    return _solve_logistic(
        gains,
        weights,
        count_target=count_target,
        gains_target=gains_target,
        label="Residential",
        population="liable",
        offset=offset,
    )


def _solve_logistic(
    gains: np.ndarray,
    weights: np.ndarray,
    *,
    count_target: float,
    gains_target: float,
    label: str,
    population: str,
    offset: np.ndarray | float = 0.0,
) -> tuple[float, float, float]:
    """The logistic solve behind the residential flag and each BADR band."""

    if gains.size == 0:
        raise ValueError(f"No {population} gainers to flag for {label}.")
    total_weight = float(weights.sum())
    total_gains = float((weights * gains).sum())
    if count_target >= total_weight:
        raise ValueError(
            f"{label} count target {count_target} is not below the {population} "
            f"taxpayer mass {total_weight}."
        )
    if gains_target >= total_gains:
        raise ValueError(
            f"{label} gains target {gains_target} is not below the {population} "
            f"gains mass {total_gains}."
        )
    log_gains, centre = _centred_log_gains(gains, weights)
    low, high = -12.0, 12.0

    def expected_gains(b: float) -> float:
        a = _solve_a(log_gains, weights, b=b, count_target=count_target, offset=offset)
        return float((weights * gains * _logistic(log_gains, a, b, offset)).sum())

    if not expected_gains(low) <= gains_target <= expected_gains(high):
        raise ValueError(
            f"{label} gains target lies outside the range a logistic tilt of "
            f"the {population} gainers can reach at count {count_target}: "
            f"[{expected_gains(low)}, {expected_gains(high)}] versus {gains_target}."
        )
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        if expected_gains(mid) < gains_target:
            low = mid
        else:
            high = mid
        if high - low < _SOLVE_TOLERANCE:
            break
    b = 0.5 * (low + high)
    a = _solve_a(log_gains, weights, b=b, count_target=count_target, offset=offset)
    return a, b, centre


def _weighted_systematic_flags(
    probabilities: np.ndarray,
    weights: np.ndarray,
    order: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Weighted systematic sampling along ``order``.

    Walking the persons in order, the expected weight ``w * p`` accrues to a
    running balance; a person is flagged when the balance reaches
    ``(1 - offset)`` of their own weight, and their whole weight is then
    drawn down. The balance therefore stays within one person's weight of
    zero at every step, so the flagged weight tracks the expected weight
    along the ordering rather than only in total.
    """

    flags = np.zeros(probabilities.shape, dtype=bool)
    owed = 0.0
    for index in order:
        weight = float(weights[index])
        if weight <= 0.0:
            # A row without weight carries no mass to flag or to owe.
            continue
        owed += weight * float(probabilities[index])
        if owed >= (1.0 - offset) * weight:
            flags[index] = True
            owed -= weight
    return flags


def _solve_band_logistic(
    gains: np.ndarray,
    weights: np.ndarray,
    *,
    count_target: float,
    gains_target: float,
    label: str,
    offset: np.ndarray | float = 0.0,
) -> tuple[float, float, float]:
    """A band's logistic, with zero slope when the pool's own mean fits.

    A flat probability gives the band pool's own weighted mean gain; where
    that already equals the published mean the slope is exactly zero, which
    also covers a pool of one distinct gain, where a slope is not
    identified. Otherwise the residential-style nested bisection runs.
    """

    total_weight = float(weights.sum())
    if count_target >= total_weight:
        raise ValueError(
            f"{label} count target {count_target} is not below the band pool "
            f"taxpayer mass {total_weight}."
        )
    log_gains, centre = _centred_log_gains(gains, weights)
    a_flat = _solve_a(
        log_gains, weights, b=0.0, count_target=count_target, offset=offset
    )
    flat_gains = float(
        (weights * gains * _logistic(log_gains, a_flat, 0.0, offset)).sum()
    )
    if abs(flat_gains - gains_target) <= _SOLVE_TOLERANCE * max(abs(gains_target), 1.0):
        return a_flat, 0.0, centre
    return _solve_logistic(
        gains,
        weights,
        count_target=count_target,
        gains_target=gains_target,
        label=label,
        population="band pool",
        offset=offset,
    )


def _relief_rate_tax(
    qualifying: np.ndarray,
    gains: np.ndarray,
    *,
    rate: float,
    annual_exempt_amount: float,
) -> np.ndarray:
    """Baseline tax at the relief rate, as the engine allocates the exempt amount.

    The engine sets the exempt amount against the highest-rate schedule
    first, so the relief-rate gain takes it last; a claimant carries no
    residential or carried-interest gains, so only the main-rate remainder
    precedes it. Reported against Table 4.1's tax column, never fitted.
    """

    main = np.maximum(gains, 0.0) - qualifying
    exempt_left = np.maximum(annual_exempt_amount - main, 0.0)
    return rate * (qualifying - np.minimum(qualifying, exempt_left))


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _assign_badr_claims(
    *,
    gains: np.ndarray,
    person_weight: np.ndarray,
    person_id: np.ndarray,
    pool_index: np.ndarray,
    facts: HMRCCGTAssetTypeFacts,
    badr_parameters: UKCGTBADRParameters,
    annual_exempt_amount: float,
    time_period: int,
    seed: int,
    offset_by_person: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Draw each Table 4.1 band's claimants and their qualifying gains.

    Returns the qualifying gain of every person (0 where not a claimant) and
    one receipt row per band. One seeded offset per declared band is drawn
    in band order whether or not the band is skipped, so the stream does
    not depend on which bands carry a target.
    """

    qualifying = np.zeros(gains.shape)
    limit = float(badr_parameters.lifetime_limit)
    if not limit > 0.0:
        raise ValueError(f"BADR lifetime limit must be positive, got {limit}.")
    bands = facts.table4_bands
    rng = np.random.default_rng((seed, int(time_period)))
    offsets = rng.random(len(bands))
    pool_gains = gains[pool_index]
    receipts: list[dict[str, object]] = []
    for position, band in enumerate(bands):
        label = f"BADR band {band.label}"
        open_top = band.upper_bound is None
        if open_top:
            if band.lower_bound != limit:
                raise ValueError(
                    f"{label}: the open top band is the claims at the lifetime "
                    f"limit, so it must start at the limit ({limit}), not at "
                    f"{band.lower_bound}."
                )
            in_band = pool_gains >= band.lower_bound
            count_target = band.gains / limit
        else:
            in_band = (pool_gains >= band.lower_bound) & (pool_gains < band.upper_bound)
            count_target = band.taxpayers
        rows = pool_index[in_band]
        receipt: dict[str, object] = {
            "lower_bound": band.lower_bound,
            "upper_bound": band.upper_bound,
            "published_taxpayers": band.taxpayers,
            "published_gains": band.gains,
            "published_tax": band.tax,
            "count_target": float(count_target),
            "gains_target": float(band.gains),
            "qualifying_amount": "lifetime_limit" if open_top else "net_gain",
            "pool_rows": int(rows.size),
        }
        if count_target <= 0.0 or band.gains <= 0.0:
            receipt["skipped"] = True
            receipts.append(receipt)
            continue
        if rows.size == 0:
            raise ValueError(
                f"{label} has a positive target but no liable non-residential "
                "gainer in its range."
            )
        band_weights = person_weight[rows]
        band_gains = gains[rows]
        band_offset = 0.0 if offset_by_person is None else offset_by_person[rows]
        if open_top:
            mass = float(band_weights.sum())
            if count_target >= mass:
                raise ValueError(
                    f"{label} count target {count_target} is not below the band "
                    f"pool taxpayer mass {mass}."
                )
            intercept, slope, centre = float("nan"), 0.0, float("nan")
            if not np.any(band_offset):
                probabilities = np.full(rows.size, count_target / mass)
            else:
                # Zero slope: the probability varies only with the stock shift.
                flat = np.zeros(rows.size)
                intercept = _solve_a(
                    flat,
                    band_weights,
                    b=0.0,
                    count_target=count_target,
                    offset=band_offset,
                )
                probabilities = _logistic(flat, intercept, 0.0, band_offset)
            amounts = np.full(rows.size, limit)
        else:
            intercept, slope, centre = _solve_band_logistic(
                band_gains,
                band_weights,
                count_target=count_target,
                gains_target=band.gains,
                label=label,
                offset=band_offset,
            )
            probabilities = _logistic(
                np.log(band_gains) - centre, intercept, slope, band_offset
            )
            amounts = band_gains
        order = np.lexsort((person_id[rows], band_gains))
        flags = _weighted_systematic_flags(
            probabilities, band_weights, order, float(offsets[position])
        )
        claimants = rows[flags]
        qualifying[claimants] = amounts[flags]
        bernoulli = probabilities * (1.0 - probabilities)
        tax = _relief_rate_tax(
            qualifying[claimants],
            gains[claimants],
            rate=badr_parameters.rate,
            annual_exempt_amount=annual_exempt_amount,
        )
        receipt.update(
            {
                "skipped": False,
                "pool_mass": float(band_weights.sum()),
                "pool_gains": float((band_weights * band_gains).sum()),
                "pool_min_gain": float(band_gains.min()),
                "pool_max_gain": float(band_gains.max()),
                "max_pool_weight": float(band_weights.max()),
                "logistic_intercept": _json_number(intercept),
                "logistic_slope": float(slope),
                "log_gain_centre": _json_number(centre),
                "expected_count": float((band_weights * probabilities).sum()),
                "expected_gains": float((band_weights * probabilities * amounts).sum()),
                "achieved_count": float(person_weight[claimants].sum()),
                "achieved_gains": float(
                    (person_weight[claimants] * qualifying[claimants]).sum()
                ),
                "achieved_rows": int(claimants.size),
                "count_bernoulli_sigma": float(
                    np.sqrt((band_weights**2 * bernoulli).sum())
                ),
                "gains_bernoulli_sigma": float(
                    np.sqrt(((band_weights * amounts) ** 2 * bernoulli).sum())
                ),
                "relief_rate_tax": float((person_weight[claimants] * tax).sum()),
            }
        )
        receipts.append(receipt)
    return qualifying, receipts


# ---------------------------------------------------------------------------
# Main asset type
# ---------------------------------------------------------------------------


def _type_kernels(gains: np.ndarray, medians: Mapping[str, float]) -> np.ndarray:
    """Log-normal size kernels, one column per non-residential type."""

    log_gains = np.log(gains)[:, None]
    centres = np.log(
        np.asarray(
            [medians[asset_type] for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES]
        )
    )[None, :]
    return np.exp(-0.5 * ((log_gains - centres) / CGT_ASSET_TYPE_LOG_SIGMA) ** 2)


def fit_type_weights(
    gains: np.ndarray,
    weights: np.ndarray,
    *,
    medians: Mapping[str, float],
    share_targets: Mapping[str, float],
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit the type weights so expected gains shares match the targets.

    Returns the type weights, the per-person type probabilities and the
    number of iterations used. Multiplicative updates on the weights are a
    one-dimensional rake per type; with a wide kernel every type reaches
    every person, so the fixed point exists and the loop converges quickly.
    ``allowed`` (persons x types) multiplies each kernel: zero for a type a
    person may not take, as a BADR claimant may take only the types the
    relief applies to, and the stock floor for a type whose stock the person
    does not show; a person must keep at least one type. The fit refuses if
    it has not converged within the iteration limit rather than returning an
    unfitted draw.
    """

    kernels = _type_kernels(gains, medians)
    if allowed is not None:
        if allowed.shape != kernels.shape:
            raise ValueError(
                f"Type restriction has shape {allowed.shape}, expected {kernels.shape}."
            )
        kernels = kernels * allowed
        if not (kernels.sum(axis=1) > 0.0).all():
            raise ValueError("Every person must keep at least one asset type.")
    targets = np.asarray(
        [
            share_targets[asset_type]
            for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        ]
    )
    type_weights = np.ones(len(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES))
    mass = weights * gains
    iterations = 0
    probabilities = kernels
    gap = np.inf
    for step in range(1, _SHARE_FIT_ITERATIONS + 1):
        iterations = step
        scaled = kernels * type_weights[None, :]
        probabilities = scaled / scaled.sum(axis=1, keepdims=True)
        achieved = (mass[:, None] * probabilities).sum(axis=0) / mass.sum()
        gap = float(np.max(np.abs(achieved - targets)))
        if gap < _SHARE_FIT_TOLERANCE:
            break
        type_weights = type_weights * (targets / np.maximum(achieved, 1e-300))
        type_weights = type_weights / type_weights.sum()
    if gap >= _SHARE_FIT_TOLERANCE:
        raise ValueError(
            "Main asset-type weights did not reach the Table 7 gains shares "
            f"within {_SHARE_FIT_ITERATIONS} iterations (largest share gap {gap})."
        )
    return type_weights, probabilities, iterations


CGT_CLAIMANT_STATUSES: tuple[str, ...] = ("claimants", "non_claimants")


def claimant_status_share_targets(
    share_targets: Mapping[str, float], claimant_gains_share: float
) -> dict[str, dict[str, float]]:
    """Table 7's composition split by claimant status (microcosm#1063).

    Claimants take Table 7's shares among the types the reliefs apply to,
    ``c_t = s_t / S_E`` with ``S_E`` those types' summed share; non-claimants
    take the shares that restore the published composition over everyone,
    ``n_t = (s_t - k c_t) / (1 - k)`` with ``k`` the claimants' share of the
    non-residential gains. A non-claimant share is negative exactly when ``k``
    exceeds ``S_E``, which the stage refuses before it calls this.
    """

    eligible_share = float(sum(share_targets[name] for name in CGT_BADR_ELIGIBLE_TYPES))
    if not eligible_share > 0.0:
        raise ValueError("Table 7 gives the BADR-eligible types no gains share.")
    if not 0.0 <= claimant_gains_share < 1.0:
        raise ValueError(
            f"The claimants' share of the non-residential gains must lie in "
            f"[0, 1), got {claimant_gains_share}."
        )
    claimants = {
        name: (
            float(share_targets[name]) / eligible_share
            if name in CGT_BADR_ELIGIBLE_TYPES
            else 0.0
        )
        for name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
    }
    non_claimants = {
        name: (float(share_targets[name]) - claimant_gains_share * claimants[name])
        / (1.0 - claimant_gains_share)
        for name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
    }
    if min(non_claimants.values()) < -_SHARE_FIT_TOLERANCE:
        raise ValueError(
            "The non-claimant composition that restores Table 7 has a negative "
            f"share: {non_claimants}."
        )
    return {
        "claimants": claimants,
        "non_claimants": {
            name: max(share, 0.0) for name, share in non_claimants.items()
        },
    }


def _diffuse_types(
    probabilities: np.ndarray, stakes: np.ndarray, order: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign one type per person so the realised gains by type track the fit.

    Walking the persons in ``order``, each adds its stake (weight times gain)
    times its type probabilities to a running balance per type and takes the
    type it may hold whose balance is largest, which is then drawn down by the
    whole stake. An independent draw lets a handful of heavy rows decide the
    realised composition; here every row's assignment repays what the rows
    before it left owing, so the realised gains by type stay within a few of
    the largest stakes of the fitted composition throughout. Returns the type
    index per person and the final balances (expected less realised gains).
    """

    choice = np.zeros(stakes.shape, dtype=int)
    owed = np.zeros(probabilities.shape[1])
    for index in order:
        row = probabilities[index]
        stake = float(stakes[index])
        owed += stake * row
        chosen = int(np.argmax(np.where(row > 0.0, owed, -np.inf)))
        choice[index] = chosen
        owed[chosen] -= stake
    return choice, owed


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0 or not float(weights.sum()) > 0.0:
        return None
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values[order][min(position, values.size - 1)])


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UKCGTAssetTypeSummary:
    """What the stage was asked for and what it realised."""

    residential: Mapping[str, object]
    badr: Mapping[str, object]
    asset_type: Mapping[str, object]
    composition_by_band: tuple[dict[str, object], ...]
    value_counts: Mapping[str, int]
    facts: Mapping[str, object]
    seeds: Mapping[str, int]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_CGT_ASSET_TYPE_STAGE_NAME,
            "residential": dict(self.residential),
            "badr": dict(self.badr),
            "asset_type": dict(self.asset_type),
            "composition_by_band": [dict(row) for row in self.composition_by_band],
            "value_counts": dict(self.value_counts),
            "facts": dict(self.facts),
            "seeds": dict(self.seeds),
        }


def assign_uk_cgt_asset_types(
    frame: Frame,
    facts: HMRCCGTAssetTypeFacts,
    parameters: UKCGTPolicyParameters,
    badr_parameters: UKCGTBADRParameters,
    *,
    badr_seed: int = CGT_BADR_FLAG_SEED,
    mass_change_reason: str = UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON,
) -> tuple[Frame, UKCGTAssetTypeSummary]:
    """Write the asset-type and BADR qualifying gains columns.

    The residential arms come from ``cgt_residential_split`` (microcosm#1063):
    a person whose ``capital_gains_residential_property`` is positive is a
    residential gainer on this arm; the stage types the rest.
    """

    validate_uk_national_frame(frame)
    time_period = uk_time_period(frame)
    person = frame.table("person").reset_index(drop=True)
    if "capital_gains" not in person.columns:
        raise ValueError("Person table has no capital_gains column to classify.")
    if CGT_RESIDENTIAL_GAINS_COLUMN not in person.columns:
        raise ValueError(
            f"Person table has no {CGT_RESIDENTIAL_GAINS_COLUMN} column; run "
            "cgt_residential_split before the asset-type stage."
        )
    for column in (CGT_ASSET_TYPE_COLUMN, CGT_BADR_GAINS_COLUMN):
        if column in person.columns:
            raise ValueError(f"Person table already carries {column}.")
    household = frame.table("household")
    weights_by_household = pd.Series(
        frame.weights_for("household").values, index=household["household_id"]
    )
    person_weight = (
        person["person_household_id"].map(weights_by_household).to_numpy(dtype=float)
    )
    if not np.isfinite(person_weight).all():
        raise ValueError("Every person must map to a weighted household.")
    gains = pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(dtype=float)
    person_id = person["person_id"].to_numpy()

    liable = gains > parameters.annual_exempt_amount
    positive = gains > 0
    asset_type = np.full(len(person), CGT_ASSET_TYPE_NONE, dtype=object)
    asset_type[positive & ~liable] = CGT_ASSET_TYPE_SUB_AEA

    stocks = _stock_signals(person, household)

    # 1. Residential arms, as the residential split wrote them.
    liable_index = np.flatnonzero(liable)
    liable_gains = gains[liable_index]
    liable_weights = person_weight[liable_index]
    residential_gains = pd.to_numeric(
        person[CGT_RESIDENTIAL_GAINS_COLUMN], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(residential_gains).all() or (residential_gains < 0.0).any():
        raise ValueError(
            f"{CGT_RESIDENTIAL_GAINS_COLUMN} must be finite and non-negative."
        )
    flags = residential_gains[liable_index] > 0.0
    if (residential_gains[~liable] > 0.0).any():
        raise ValueError("A residential gain sits on a person who is not liable.")
    if not np.array_equal(residential_gains[liable_index][flags], liable_gains[flags]):
        raise ValueError(
            "A residential arm's residential gain differs from the person's net "
            "gain; the split attributes the whole gain."
        )
    residential_rows = liable_index[flags]
    asset_type[residential_rows] = CGT_ASSET_TYPE_RESIDENTIAL
    count_target = facts.residential_taxpayers_individuals_basis
    gains_target = facts.residential_gains_individuals_basis

    # 2. BADR and Investors' Relief claims on the non-residential remainder.
    remainder_index = liable_index[~flags]
    qualifying, badr_bands = _assign_badr_claims(
        gains=gains,
        person_weight=person_weight,
        person_id=person_id,
        pool_index=remainder_index,
        facts=facts,
        badr_parameters=badr_parameters,
        annual_exempt_amount=parameters.annual_exempt_amount,
        time_period=int(time_period),
        seed=badr_seed,
        offset_by_person=CGT_STOCK_LOG_ODDS * stocks["business"].astype(float),
    )
    claimant = qualifying > 0.0
    in_pool = np.zeros(len(person), dtype=bool)
    in_pool[remainder_index] = True
    limit = float(badr_parameters.lifetime_limit)
    band_lowers = np.asarray(
        [band.lower_bound for band in facts.table4_bands], dtype=float
    )
    band_uppers = np.asarray(
        [
            np.inf if band.upper_bound is None else band.upper_bound
            for band in facts.table4_bands
        ],
        dtype=float,
    )
    claimant_band = np.searchsorted(band_lowers, gains[claimant], side="right") - 1
    within_band = np.zeros(int(claimant.sum()), dtype=bool)
    if claimant.any():
        top = band_uppers[claimant_band] == np.inf
        claimed = qualifying[claimant]
        within_band = np.where(
            top,
            claimed == limit,
            (claimed >= band_lowers[claimant_band])
            & (claimed < band_uppers[claimant_band]),
        )
    invariants = {
        "claimants_outside_pool": int((claimant & ~in_pool).sum()),
        "qualifying_above_gain": int((claimant & (qualifying > gains)).sum()),
        "qualifying_above_limit": int((qualifying > limit).sum()),
        "qualifying_outside_band": int((~within_band).sum()),
        "residential_overlap": int((claimant & (residential_gains > 0.0)).sum()),
        "sub_aea_claimants": int((claimant & ~liable).sum()),
    }
    if any(invariants.values()):
        raise ValueError(f"BADR claims broke a declared invariant: {invariants}.")

    # 3. Main asset type for the non-residential liable remainder; a claimant
    # draws only from the types the reliefs apply to, so the type follows the
    # claim.
    medians = {
        asset_type_name: facts.table7_type(asset_type_name).mean_gain_per_disposal
        for asset_type_name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
    }
    share_targets = facts.non_residential_gains_shares()
    eligible_type = np.isin(
        np.asarray(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES, dtype=object),
        np.asarray(CGT_BADR_ELIGIBLE_TYPES, dtype=object),
    )
    eligible_share_target = float(
        sum(share_targets[name] for name in CGT_BADR_ELIGIBLE_TYPES)
    )
    remainder_mass_values = person_weight[remainder_index] * gains[remainder_index]
    claimant_in_remainder = claimant[remainder_index]
    claimant_gains_share = (
        float(remainder_mass_values[claimant_in_remainder].sum())
        / float(remainder_mass_values.sum())
        if remainder_index.size
        else 0.0
    )
    if claimant_gains_share > eligible_share_target:
        raise ValueError(
            "BADR claimants hold a share of the non-residential gains "
            f"({claimant_gains_share}) above the Table 7 share of the types the "
            f"reliefs apply to ({eligible_share_target}); the restricted type "
            "fit cannot reach the Table 7 shares."
        )
    type_names = CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
    status_targets = claimant_status_share_targets(share_targets, claimant_gains_share)
    status_reports: dict[str, dict[str, object]] = {}
    remainder_types = np.zeros(remainder_index.size, dtype=int)
    for status in CGT_CLAIMANT_STATUSES:
        in_status = (
            claimant_in_remainder if status == "claimants" else ~claimant_in_remainder
        )
        rows = remainder_index[in_status]
        targets = status_targets[status]
        report: dict[str, object] = {
            "rows": int(rows.size),
            "target_gains_share": dict(targets),
        }
        status_reports[status] = report
        if rows.size == 0:
            continue
        allowed = stocks["type_factor"][rows].copy()
        if status == "claimants":
            allowed = allowed * eligible_type
        fitted_weights, type_probabilities, iterations = fit_type_weights(
            gains[rows],
            person_weight[rows],
            medians=medians,
            share_targets=targets,
            allowed=allowed,
        )
        stakes = person_weight[rows] * gains[rows]
        # Largest gain first, so the rows that could move the composition most
        # are placed while every later row can still repay the balance.
        order = np.lexsort((person_id[rows], -gains[rows]))
        choice, balances = _diffuse_types(type_probabilities, stakes, order)
        remainder_types[in_status] = choice
        mass = float(stakes.sum())
        expected = (stakes[:, None] * type_probabilities).sum(axis=0)
        realised = np.asarray(
            [float(stakes[choice == index].sum()) for index in range(len(type_names))]
        )
        assigned_probability = type_probabilities[np.arange(rows.size), choice]
        report.update(
            {
                "gains_mass": mass,
                "type_weights": {
                    name: float(fitted_weights[index])
                    for index, name in enumerate(type_names)
                },
                "share_fit_iterations": int(iterations),
                "expected_gains_share": {
                    name: float(expected[index] / mass) if mass > 0.0 else 0.0
                    for index, name in enumerate(type_names)
                },
                "achieved_gains_share": {
                    name: float(realised[index] / mass) if mass > 0.0 else 0.0
                    for index, name in enumerate(type_names)
                },
                "final_balance": {
                    name: float(balances[index])
                    for index, name in enumerate(type_names)
                },
                "largest_stake": float(stakes.max()),
                # How far the walk departs from the fit row by row: the
                # stake-weighted mean fitted probability of the type taken,
                # beside what an independent draw gives in expectation.
                "mean_assigned_probability": (
                    float((stakes * assigned_probability).sum() / mass)
                    if mass > 0.0
                    else 0.0
                ),
                "independent_draw_mean_probability": (
                    float((stakes * (type_probabilities**2).sum(axis=1)).sum() / mass)
                    if mass > 0.0
                    else 0.0
                ),
            }
        )
    if remainder_index.size:
        asset_type[remainder_index] = np.asarray(type_names, dtype=object)[
            remainder_types
        ]

    if (asset_type[liable] == CGT_ASSET_TYPE_NONE).any() or (
        asset_type[liable] == CGT_ASSET_TYPE_SUB_AEA
    ).any():
        raise ValueError("Every liable gainer must carry an asset type.")
    if set(asset_type) - set(CGT_ASSET_TYPE_DOMAIN):
        raise ValueError("Asset-type draw produced a value outside the domain.")

    # Reporting: the residential identities the split left, by gain band.
    achieved_count = float(person_weight[residential_rows].sum())
    achieved_gains = float(
        residential_gains[residential_rows].dot(person_weight[residential_rows])
    )
    bounds = np.asarray(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS, dtype=float)
    liable_band = np.searchsorted(bounds, liable_gains, side="right") - 1
    residential_bands = []
    for position, lower in enumerate(bounds):
        in_band = liable_band == position
        on_arm = in_band & flags
        residential_bands.append(
            {
                "gain_lower_bound": float(lower),
                "gain_upper_bound": (
                    float(bounds[position + 1]) if position + 1 < bounds.size else None
                ),
                "liable_rows": int(in_band.sum()),
                "achieved_rows": int(on_arm.sum()),
                "achieved_count": float(liable_weights[on_arm].sum()),
                "achieved_gains": float(
                    (liable_weights[on_arm] * liable_gains[on_arm]).sum()
                ),
            }
        )
    residential = {
        "source_stage": "cgt_residential_split",
        "count_target_individuals_basis": count_target,
        "gains_target_individuals_basis": gains_target,
        "achieved_count": achieved_count,
        "achieved_gains": achieved_gains,
        "achieved_rows": int(residential_rows.size),
        "count_relative_error": (
            abs(achieved_count - count_target) / count_target
            if count_target > 0.0
            else None
        ),
        "gains_relative_error": (
            abs(achieved_gains - gains_target) / gains_target
            if gains_target > 0.0
            else None
        ),
        "max_liable_weight": float(liable_weights.max())
        if liable_weights.size
        else 0.0,
        "bands": residential_bands,
        "liable_taxpayer_mass": float(liable_weights.sum()),
        "liable_gains_mass": float((liable_weights * liable_gains).sum()),
        "table8a_taxpayers_total": facts.table8a_taxpayers_total,
        "table8a_gains_total": facts.table8a_gains_total,
        "table8b_individuals_taxpayer_share": facts.individuals_share("taxpayers"),
        "table8b_individuals_gains_share": facts.individuals_share("gains"),
        "stock_signal": CGT_RESIDENTIAL_STOCK_SIGNAL,
        "stock_share_flagged": _weighted_share(
            person_weight,
            asset_type == CGT_ASSET_TYPE_RESIDENTIAL,
            stocks["residential"],
        ),
        "stock_share_liable": _weighted_share(
            person_weight, liable, stocks["residential"]
        ),
    }
    remainder_mass = float(
        (person_weight[remainder_index] * gains[remainder_index]).sum()
    )
    achieved_type_gains = {
        asset_type_name: float(
            (person_weight * gains)[asset_type == asset_type_name].sum()
        )
        for asset_type_name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
    }
    asset_type_report = {
        "log_sigma": CGT_ASSET_TYPE_LOG_SIGMA,
        "median_anchor_gbp": medians,
        "composition_by_claimant_status": status_reports,
        "realization": "error diffusion in descending (gain, person_id) order",
        "achieved_median_gain_gbp": {
            asset_type_name: _weighted_median(
                gains[asset_type == asset_type_name],
                person_weight[asset_type == asset_type_name],
            )
            for asset_type_name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        },
        "target_gains_share": dict(share_targets),
        "achieved_gains_share": {
            asset_type_name: (value / remainder_mass if remainder_mass > 0 else 0.0)
            for asset_type_name, value in achieved_type_gains.items()
        },
        "achieved_people": {
            asset_type_name: float(person_weight[asset_type == asset_type_name].sum())
            for asset_type_name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        },
        "achieved_gains": achieved_type_gains,
        "non_residential_gains_mass": remainder_mass,
        "share_fit_converged": True,
        "claimant_categories": list(CGT_BADR_ELIGIBLE_TYPES),
        "claimant_gains_share": claimant_gains_share,
        "claimant_categories_target_share": eligible_share_target,
        "stock_type_floor": CGT_STOCK_TYPE_FLOOR,
        "stock_share_by_type": {
            asset_type_name: {
                "typed": _weighted_share(
                    person_weight,
                    asset_type == asset_type_name,
                    stocks["by_type"][asset_type_name],
                ),
                "non_residential_liable": _weighted_share(
                    person_weight, in_pool, stocks["by_type"][asset_type_name]
                ),
            }
            for asset_type_name in CGT_TYPE_STOCK_SIGNALS
        },
    }
    claimant_types = {
        asset_type_name: {
            "people": float(
                person_weight[claimant & (asset_type == asset_type_name)].sum()
            ),
            "net_gains": float(
                (person_weight * gains)[
                    claimant & (asset_type == asset_type_name)
                ].sum()
            ),
            "qualifying_gains": float(
                (person_weight * qualifying)[
                    claimant & (asset_type == asset_type_name)
                ].sum()
            ),
        }
        for asset_type_name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
    }
    badr_tax = _relief_rate_tax(
        qualifying[claimant],
        gains[claimant],
        rate=badr_parameters.rate,
        annual_exempt_amount=parameters.annual_exempt_amount,
    )
    badr = {
        "population": "liable gainers not flagged residential",
        "lifetime_limit": limit,
        "rate": float(badr_parameters.rate),
        "parameters_instant": badr_parameters.instant,
        "parameters_source": badr_parameters.source,
        "bands": badr_bands,
        "totals": {
            "count_target": float(
                sum(float(row["count_target"]) for row in badr_bands)
            ),
            "gains_target": float(
                sum(float(row["gains_target"]) for row in badr_bands)
            ),
            "achieved_count": float(person_weight[claimant].sum()),
            "achieved_gains": float((person_weight * qualifying)[claimant].sum()),
            "achieved_rows": int(claimant.sum()),
            "relief_rate_tax": float((person_weight[claimant] * badr_tax).sum()),
            "published_individuals_taxpayers": facts.table4_individuals_taxpayers,
            "published_individuals_gains": facts.table4_individuals_gains,
            "published_individuals_tax": facts.table4_individuals_tax,
            "published_trusts_gains": facts.table4_trusts_gains,
            "published_trusts_tax": facts.table4_trusts_tax,
            "published_all_taxpayers": facts.table4_all_taxpayers,
            "published_all_gains": facts.table4_all_gains,
            "published_all_tax": facts.table4_all_tax,
        },
        "invariants": invariants,
        "claimant_types": claimant_types,
        "stock_log_odds": CGT_STOCK_LOG_ODDS,
        "stock_signal": CGT_BUSINESS_STOCK_SIGNAL,
        "stock_share_claimants": _weighted_share(
            person_weight, claimant, stocks["business"]
        ),
        "stock_share_pool": _weighted_share(person_weight, in_pool, stocks["business"]),
    }
    bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    uppers = (*bounds[1:], np.inf)
    composition = []
    for lower, upper in zip(bounds, uppers, strict=True):
        in_band = liable & (gains >= lower) & (gains < upper)
        row: dict[str, object] = {"gain_lower_bound": lower}
        for asset_type_name in (
            CGT_ASSET_TYPE_RESIDENTIAL,
            *CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES,
        ):
            mask = in_band & (asset_type == asset_type_name)
            row[f"{asset_type_name}_people"] = float(person_weight[mask].sum())
            row[f"{asset_type_name}_gains"] = float(
                (person_weight[mask] * gains[mask]).sum()
            )
        composition.append(row)
    value_counts = {
        value: int((asset_type == value).sum()) for value in CGT_ASSET_TYPE_DOMAIN
    }
    summary = UKCGTAssetTypeSummary(
        residential=residential,
        badr=badr,
        asset_type=asset_type_report,
        composition_by_band=tuple(composition),
        value_counts=value_counts,
        facts={
            "resource": facts.resource,
            "resource_sha256": facts.resource_sha256,
            "source_commit": facts.source_commit,
            "annual_exempt_amount": parameters.annual_exempt_amount,
        },
        seeds={"badr_flag": badr_seed},
    )

    new_person = person.copy()
    new_person[CGT_ASSET_TYPE_COLUMN] = asset_type
    new_person[CGT_BADR_GAINS_COLUMN] = qualifying
    weights = frame.weights_for("household")
    household_mass = float(weights.total)
    receipt = MassChangeRecord(
        entity="household",
        old_total=household_mass,
        new_total=household_mass,
        declared_factor=1.0,
        reason=mass_change_reason,
    )
    result = uk_national_frame(
        person=new_person,
        benunit=frame.table("benunit"),
        household=household,
        time_period=time_period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=weights.values,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result)
    return result, summary


def uk_cgt_asset_type_stage_transform(
    stage: SourceStageSpec,
    *,
    facts: HMRCCGTAssetTypeFacts | None = None,
    parameters: UKCGTPolicyParameters | None = None,
    badr_parameters: UKCGTBADRParameters | None = None,
):
    """Bind the spine manifest, then run the reviewed assignment."""

    _assert_cgt_asset_type_stage_parameters(stage)
    return UKCGTAssetTypeStageTransform(
        stage=stage,
        facts=facts,
        parameters=parameters,
        badr_parameters=badr_parameters,
    )


@dataclass(frozen=True)
class UKCGTAssetTypeStageTransform:
    """Source-plan asset-type assignment with a stage-time summary receipt."""

    stage: SourceStageSpec
    facts: HMRCCGTAssetTypeFacts | None = None
    parameters: UKCGTPolicyParameters | None = None
    badr_parameters: UKCGTBADRParameters | None = None
    last_result: UKCGTAssetTypeSummary | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_cgt_asset_type_stage_parameters(self.stage)
        facts = self.facts
        if facts is None:
            facts = load_hmrc_cgt_asset_type_facts()
        parameters = self.parameters
        if parameters is None:
            parameters = uk_cgt_policy_parameters(uk_time_period(frame))
        badr_parameters = self.badr_parameters
        if badr_parameters is None:
            badr_parameters = uk_cgt_badr_parameters(uk_time_period(frame))
        result, summary = assign_uk_cgt_asset_types(
            frame, facts, parameters, badr_parameters
        )
        object.__setattr__(self, "last_result", summary)
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (CGT_ASSET_TYPE_COLUMN, CGT_BADR_GAINS_COLUMN)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def cgt_asset_type_operation_parameters() -> dict[str, dict[str, Any]]:
    """The reviewed operation payloads the manifests must carry verbatim."""

    return {
        "verify_vendored_fact_resource": {
            "artifact_role": "cgt_asset_type_facts",
            "resource": HMRC_CGT_ASSET_TYPE_RESOURCE,
            "feed_pin": "chronicle_feed.json",
            "record_sets": list(HMRC_CGT_ASSET_TYPE_RECORD_SETS),
            "source_vintage": "2024-25",
            "mapped_build_period": 2024,
            "period_mapping": "published_tax_year_equals_build_period",
            "require_before_source_read": True,
            "runtime_sha256_required": True,
            "fail_on_mismatch": True,
        },
        "assign_badr_qualifying_gains": {
            "population": (
                "liable gainers whose arm is not residential "
                "(capital_gains_residential_property == 0, cgt_residential_split)"
            ),
            "bands": (
                "HMRC Table 4.1 2024-25 individuals claiming Business Asset "
                "Disposal Relief or Investors' Relief by band of qualifying gain "
                "(lower limits 0, 10,000, 25,000, 50,000, 100,000, 250,000, "
                "500,000 and 1,000,000)"
            ),
            "band_pool": (
                "the population with net gains inside the band's range; the open "
                "top band takes net gains at or above the lifetime limit"
            ),
            "model": (
                "per band below the lifetime limit, a logistic probability in "
                "centred log gains with the stock shift, solved like the "
                "residential flag: the intercept matches the band's published "
                "claimant count and the slope its published qualifying gains, with "
                "zero slope where the band pool's own mean already matches; the "
                "open top band has zero slope, so its probability varies only with "
                "the stock shift, and its intercept matches its count"
            ),
            "stock_log_odds": CGT_STOCK_LOG_ODDS,
            "stock_signal": CGT_BUSINESS_STOCK_SIGNAL,
            "qualifying_amount": (
                "below the lifetime limit a claimant's whole net gain qualifies; "
                "in the open top band the qualifying gain is the lifetime limit"
            ),
            "top_band_count": (
                "the published top-band qualifying gains divided by the lifetime "
                "limit; the published count is rounded to the nearest thousand"
            ),
            "basis_assumption": (
                "Table 4.1 counts qualifying gains before losses and the annual "
                "exempt amount while the frame carries net gains after losses; "
                "Investors' Relief is merged with BADR, as in the engine's "
                "capital_gains_badr input; the whole-gain assumption reproduces "
                "Table 4.1's tax column to within half a percent"
            ),
            "realization": (
                "weighted systematic sampling within each band in ascending "
                "(gain, person_id) order with one seeded offset per band, drawn "
                "in band order whether or not the band carries a target"
            ),
            "seed": CGT_BADR_FLAG_SEED,
            "seed_mixing": "seed combined with the build period",
            "policy_parameters": [
                "gov.hmrc.cgt.badr.lifetime_limit",
                "gov.hmrc.cgt.badr.rate",
            ],
            "output_column": CGT_BADR_GAINS_COLUMN,
            "output_semantics": (
                "the qualifying gain where the person claims BADR or Investors' "
                "Relief, 0 otherwise; never above the person's net gain or the "
                "lifetime limit"
            ),
            "type_coupling": (
                "a claimant's main asset type is drawn only from "
                + ", ".join(CGT_BADR_ELIGIBLE_TYPES)
            ),
            "reported_not_fitted": (
                "tax at the relief rate, with the annual exempt amount set "
                "against the main-rate gains first as the engine allocates it, "
                "against Table 4.1's tax column"
            ),
        },
        "assign_main_asset_type": {
            "population": "liable gainers not flagged residential",
            "categories": list(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES),
            "claimant_categories": list(CGT_BADR_ELIGIBLE_TYPES),
            "claimant_restriction": (
                "a BADR or Investors' Relief claimant draws only from the "
                "claimant categories, so the type follows the claim"
            ),
            "kernel": (
                "log-normal density in log gains centred on each type's Table 7 "
                "2023-24 mean gain per disposal, one common log-scale width, "
                "multiplied by the stock floor where the type declares a stock "
                "signal the gainer does not show; types without a declared signal "
                "are not scaled"
            ),
            "log_sigma": CGT_ASSET_TYPE_LOG_SIGMA,
            "stock_type_floor": CGT_STOCK_TYPE_FLOOR,
            "stock_signals": dict(CGT_TYPE_STOCK_SIGNALS),
            "share_targets": (
                "Table 7 2023-24 gains by asset type excluding residential land "
                "and buildings, as shares"
            ),
            "claimant_share_targets": (
                "the share targets among the claimant categories, each over their sum"
            ),
            "non_claimant_share_targets": (
                "(share target - k x claimant share target) / (1 - k) with k the "
                "claimants' share of the non-residential gains, the composition "
                "that restores the share targets over claimants and non-claimants "
                "together"
            ),
            "basis_assumption": (
                "Table 7 does not split gains by relief, so claimants are given "
                "its composition among the claimant categories"
            ),
            "weight_fitting": (
                "one fit per claimant status: multiplicative updates of the type "
                "weights until the household-weighted expected gains shares by "
                "type match that status's share targets; a fit refuses if it has "
                "not converged within the iteration limit, and the stage refuses "
                "up front when claimants hold more of the gains than the claimant "
                "categories' share"
            ),
            "weight_fitting_iterations": _SHARE_FIT_ITERATIONS,
            "weight_fitting_tolerance": _SHARE_FIT_TOLERANCE,
            "realization": (
                "error diffusion per claimant status in descending (gain, "
                "person_id) order: each person adds weight x gain x type "
                "probability to a running balance per type and takes the type it "
                "may hold with the largest balance, which is drawn down by the "
                "whole weight x gain, so the realised gains by type track the "
                "fitted composition instead of following the heaviest rows; "
                "deterministic, no seed"
            ),
            "output_column": CGT_ASSET_TYPE_COLUMN,
            "value_domain": list(CGT_ASSET_TYPE_DOMAIN),
            "none_semantics": "no positive net gains",
            "sub_aea_semantics": (
                "positive net gains at or below the annual exempt amount; never a "
                "taxpayer, so never classified"
            ),
            "diagnostic_only": True,
        },
        "record_mass_conservation_receipt": {
            "entity": "household",
            "reason": UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON,
            "declared_factor": 1.0,
            "gate_coupling": (
                "The terminal family gate requires a valid mass-conserving "
                "MassChangeRecord carrying exactly this stage-specific reason."
            ),
        },
        "classify_cgt_asset_type_facts_with_reviewed_fence": {
            "calibration_permitted": False,
            "fact_fence_id": "cgt_asset_type_facts_sample_based_prior_vintage",
            "fenced_fact_count": 49,
            "fenced_fact_composition": (
                "Table 7 2023-24 rows: 6 asset-category rows, 3 category totals, "
                "12 financial asset-type rows, 12 non-financial asset-type rows; "
                "Table 4.1 2024-25 rows: 8 individuals tax rows by qualifying-gain "
                "band, 3 individuals totals, 2 trusts totals, 3 all-taxpayer totals"
            ),
            "classification_rationale": (
                "Table 7 is sample-based, published for 2023-24 only, counts "
                "disposals rather than taxpayers and includes trusts and non-UK "
                "assets; it seeds the asset-type draw and is reported against, "
                "never fitted (microcosm#725). Table 4.1 conditions the BADR "
                "draw, and its 16 individuals band rows of claimants and "
                "qualifying gains are also bound; the tax column has no engine "
                "variable to measure (the stage reports its relief-rate tax "
                "against it), the individuals totals restate the bands, and the "
                "trusts and all-taxpayer totals only reconcile the individuals "
                "rows (microcosm#1014)."
            ),
            "calibrated_facts": (
                "The Table 8a 2024-25 total residential property taxpayers and "
                "gains restated on the individuals basis "
                "(hmrc.cgt.residential_property_taxpayers, "
                "hmrc.cgt.residential_property_gains in uk_population_targets.json), "
                "and the Table 4.1 2024-25 individuals claimants and qualifying "
                "gains in each of the eight qualifying-gain bands "
                "(hmrc.cgt.badr_ir_taxpayers_by_band, "
                "hmrc.cgt.badr_ir_qualifying_gains_by_band)."
            ),
            "adjudication": "https://github.com/PolicyEngine/microcosm/issues/1014",
        },
    }


def _assert_cgt_asset_type_stage_parameters(stage: SourceStageSpec) -> None:
    """Arm 1 of the #730/#684 two-arm rule: the manifest restates the code."""

    expected = cgt_asset_type_operation_parameters()
    kinds = tuple(operation.kind for operation in stage.operations)
    if kinds != tuple(expected):
        raise ValueError(
            "CGT asset-type stage operation order drifted: expected "
            f"{tuple(expected)}, got {kinds}."
        )
    for operation in stage.operations:
        actual = dict(operation.parameters)
        declared = expected[operation.kind]
        if actual != declared:
            drifted = sorted(
                key
                for key in {*actual, *declared}
                if actual.get(key) != declared.get(key)
            )
            raise ValueError(
                f"CGT asset-type stage {operation.kind} declaration drifted from "
                f"the reviewed mapping on parameter(s) {drifted}."
            )
    if tuple(stage.outputs) != UKCGTAssetTypeStageTransform.output_columns():
        raise ValueError(
            "CGT asset-type stage outputs drifted: expected "
            f"{UKCGTAssetTypeStageTransform.output_columns()}, got {stage.outputs}."
        )
