"""Annual purchases of additional dwellings and non-residential property
(uk-data#556, ported from uk-data#573; microcosm#1095).

policyengine-uk 2.134.1 (pe-uk#2241) turned
``additional_residential_property_purchased`` and
``non_residential_property_purchased`` from formulas into inputs that default
to nil. Before it, a household that drew a main-home purchase was taken to buy
its whole ``other_residential_property_value`` at the higher rates and its whole
``non_residential_property_value`` as well; since it, a purchasing household
pays SDLT on its main home only unless the data supplies the two purchases.

This stage ports uk-data#573 (a draft, at ``07f6b460f``) at the spine's design
weights, after the regional uprating and the support copies, so every copy
draws on its own:

* **Who can buy.** A household whose stock of the kind is at least the
  GBP 40,000 HMRC's tables cover, with positive weight; at most one purchase of
  each kind a year.
* **Who buys.** A purchase hazard proportional to the stock over the weighted
  mean stock of eligible England-and-Northern-Ireland owners (exponent 1, so
  larger portfolios trade more often). Each household's exponential event
  score ``-log(1 - u) / hazard`` reads its own identity-keyed uniform; the
  score threshold is the one whose weighted count of
  England-and-Northern-Ireland purchasers comes closest to HMRC's
  transactions. The same threshold applies in Scotland and Wales, whose
  transaction taxes have no target here.
* **At what price.** A band-mean price drawn at HMRC's band shares on a second
  identity-keyed stream, times one common scale fitted by bisection so that the
  weighted England-and-Northern-Ireland purchase value equals HMRC's; every
  price is at least GBP 40,000 and never above the household's stock.

The targets are HMRC's Annual Stamp Tax Statistics 2024 to 2025 (corrected),
vendored from the pinned Chronicle feed (``hmrc_stamp_tax_facts.json``):

* additional dwellings: Table 6b's higher-rates (HRAD) transactions and value
  for 2024-25, which cover every buyer, allocated to households by
  individuals' shares of Table 8a's residential transactions and value (what
  other buyers would take is companies'); the price bands are Table 6b's, with
  the five above GBP 2m combined, as uk-data#573 does;
* non-residential: Table 8b's individuals for 2024-25, and its six bands.

Households' HRAD is not published, so the allocation by individuals' shares
of all Table 8a residential transactions is an assumption, as uk-data#573
says, not an observed household series. uk-data#573 gives the household
lower bounds Tables 6b and 8a imply, 87,200 transactions and GBP 33.28bn;
the allocation probably overstates households' purchases, since nearly all
receipts at or below the GBP 250,000 nil-rate band are surcharge, and other
buyers' receipts there imply more higher-rates transactions than their share
of all transactions gives them.

The stage fits transactions and value only. Households' SDLT is fitted at
build level through ``obr.sdlt``; uk-data#573 also holds its file's
higher-rates SDLT within 30% of HMRC's in a test, which this port does not
carry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    extreme_count,
    small_cell_share,
    source_household_ids,
    support_band,
)
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_COUNTRY
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "HMRC_SDLT_MINIMUM_CONSIDERATION",
    "PROPERTY_PURCHASE_BAND_IDS",
    "PROPERTY_PURCHASE_KINDS",
    "PROPERTY_PURCHASES_OPERATION_KIND",
    "PROPERTY_PURCHASES_OUTPUT_COLUMNS",
    "PROPERTY_PURCHASES_RECORD_SETS",
    "PROPERTY_PURCHASES_RESOURCE",
    "PROPERTY_PURCHASES_STAGE_NAME",
    "PURCHASE_MINIMUM_VALUE_SHARE",
    "SDLT_REGIONS",
    "UKPropertyPurchaseImputation",
    "UKPropertyPurchaseKind",
    "UKPropertyPurchaseTarget",
    "UKPropertyPurchaseTargets",
    "UKPropertyPurchasesResult",
    "UKPropertyPurchasesStageTransform",
    "impute_property_purchase_prices",
    "impute_property_purchases",
    "load_property_purchase_targets",
    "property_purchase_targets_from_rows",
    "property_purchase_uniforms",
    "property_purchases_operation_parameters",
    "sdlt_region_mask",
    "select_property_purchasers",
]

PROPERTY_PURCHASES_STAGE_NAME = "property_purchases"
PROPERTY_PURCHASES_OPERATION_KIND = "impute_annual_property_purchases"
PROPERTY_PURCHASES_RESOURCE = "hmrc_stamp_tax_facts.json"
#: The build period the 2024-25 Stamp Tax facts are mapped to (the verify
#: operation's ``mapped_build_period``); the stage refuses a frame at any other.
PROPERTY_PURCHASES_BUILD_PERIOD = 2024
_HRAD_RECORD_SET = "hmrc.stamp_taxes_2024_25.table_6b.hrad.fy2024"
_RESIDENTIAL_INDIVIDUALS_RECORD_SET = (
    "hmrc.stamp_taxes_2024_25.table_8a.individuals.fy2024"
)
_RESIDENTIAL_OTHER_BUYERS_RECORD_SET = (
    "hmrc.stamp_taxes_2024_25.table_8a.other_buyers.fy2024"
)
_NON_RESIDENTIAL_INDIVIDUALS_RECORD_SET = (
    "hmrc.stamp_taxes_2024_25.table_8b.individuals.fy2024"
)
#: The record sets the stage reads, in the order the manifest pins them.
PROPERTY_PURCHASES_RECORD_SETS = (
    _HRAD_RECORD_SET,
    _RESIDENTIAL_INDIVIDUALS_RECORD_SET,
    _RESIDENTIAL_OTHER_BUYERS_RECORD_SET,
    _NON_RESIDENTIAL_INDIVIDUALS_RECORD_SET,
)
#: HMRC's tables cover SDLT, which applies in England and Northern Ireland.
SDLT_REGIONS = tuple(
    region
    for region, country in FRS_REGION_TO_COUNTRY.items()
    if country in {"England", "Northern Ireland"}
)
#: Larger portfolios are taken to trade more often (uk-data#573's assumption).
PURCHASE_STOCK_EXPONENT = 1.0
#: Where the purchasers' stock cannot carry HMRC's purchase value, every
#: purchaser buys at its stock and the receipt records the shortfall (María's
#: ruling of 2026-10-16, microcosm#1095); below this share of the value the
#: stage refuses, as uk-data#573 refuses any shortfall.
PURCHASE_MINIMUM_VALUE_SHARE = 0.8
#: Table 6b's bands above GBP 2m, combined into one as uk-data#573 does.
_HRAD_TOP_BANDS = (
    "gbp_2m_to_3m",
    "gbp_3m_to_4m",
    "gbp_4m_to_5m",
    "gbp_5m_to_10m",
    "over_gbp_10m",
)
_TOTAL_BAND = "all_price_bands"
_BAND_ORDER = (
    "gbp_250k_or_less",
    "gbp_250k_to_500k",
    "gbp_500k_to_1m",
    "gbp_1m_to_1_5m",
    "gbp_1_5m_to_2m",
    "over_gbp_2m",
)
#: The price bands every target carries, in order.
PROPERTY_PURCHASE_BAND_IDS = _BAND_ORDER
#: HMRC's Stamp Tax tables count transactions of GBP 40,000 or above (each
#: sheet's note), the floor on every purchase and on the stock that can buy.
#: Chronicle carries it as a universe constraint, which vendored rows drop.
HMRC_SDLT_MINIMUM_CONSIDERATION = 40_000.0
_BISECTION_STEPS = 80


@dataclass(frozen=True)
class UKPropertyPurchaseKind:
    """One kind of purchase: the stock it is drawn from and its streams."""

    name: str
    stock: str
    output: str
    event_salt: str
    price_salt: str


PROPERTY_PURCHASE_KINDS = (
    UKPropertyPurchaseKind(
        name="additional_residential",
        stock="other_residential_property_value",
        output="additional_residential_property_purchased",
        event_salt="additional_residential_property_purchased",
        price_salt="additional_residential_property_purchased_price",
    ),
    UKPropertyPurchaseKind(
        name="non_residential",
        stock="non_residential_property_value",
        output="non_residential_property_purchased",
        event_salt="non_residential_property_purchased",
        price_salt="non_residential_property_purchased_price",
    ),
)
PROPERTY_PURCHASES_OUTPUT_COLUMNS = tuple(
    kind.output for kind in PROPERTY_PURCHASE_KINDS
)
_STREAM_SEED = 0


def property_purchase_uniforms(
    household_ids: np.ndarray, kind: UKPropertyPurchaseKind
) -> tuple[np.ndarray, np.ndarray]:
    """The event and price uniforms of one kind, keyed by household identity."""

    return (
        stable_identity_uniforms(
            household_ids, seed=_STREAM_SEED, salt=kind.event_salt
        ),
        stable_identity_uniforms(
            household_ids, seed=_STREAM_SEED, salt=kind.price_salt
        ),
    )


PROPERTY_PURCHASES_MASS_CHANGE_REASON = (
    "Annual property purchases imputed on the source spine: household weights "
    "pass through unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)


@dataclass(frozen=True)
class UKPropertyPurchaseTarget:
    """England-and-Northern-Ireland household purchases of one kind, one year."""

    property_type: str
    transactions: float
    value: float
    band_ids: tuple[str, ...]
    band_transactions: tuple[float, ...]
    band_values: tuple[float, ...]
    minimum_price: float

    def __post_init__(self) -> None:
        if (
            not (
                len(self.band_ids)
                == len(self.band_transactions)
                == len(self.band_values)
            )
            or not self.band_ids
        ):
            raise ValueError(
                f"{self.property_type} purchase bands must align and be non-empty."
            )
        numbers = (
            self.transactions,
            self.value,
            self.minimum_price,
            *self.band_transactions,
            *self.band_values,
        )
        if not all(np.isfinite(number) and number > 0 for number in numbers):
            raise ValueError(
                f"{self.property_type} purchase targets and bands must be positive."
            )

    def band_prices(self) -> np.ndarray:
        return np.asarray(self.band_values, dtype=float) / np.asarray(
            self.band_transactions, dtype=float
        )

    def band_shares(self) -> np.ndarray:
        """Band shares normalised by their own sum: bands are rounded."""
        counts = np.asarray(self.band_transactions, dtype=float)
        return counts / counts.sum()

    def payload(self) -> dict[str, object]:
        return {
            "property_type": self.property_type,
            "transactions": float(self.transactions),
            "value": float(self.value),
            "band_ids": list(self.band_ids),
            "band_transactions": [float(value) for value in self.band_transactions],
            "band_values": [float(value) for value in self.band_values],
            "minimum_price": float(self.minimum_price),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> UKPropertyPurchaseTarget:
        return cls(
            property_type=str(payload["property_type"]),
            transactions=float(payload["transactions"]),
            value=float(payload["value"]),
            band_ids=tuple(str(value) for value in payload["band_ids"]),
            band_transactions=tuple(
                float(value) for value in payload["band_transactions"]
            ),
            band_values=tuple(float(value) for value in payload["band_values"]),
            minimum_price=float(payload["minimum_price"]),
        )


@dataclass(frozen=True)
class UKPropertyPurchaseTargets:
    """The two kinds' targets and the facts they were derived from.

    The vendored HMRC facts always give both kinds. A kind without a target is
    not imputed, and its purchases stay nil: only the synthetic parity fixture,
    whose small frame holds no stock of a kind, passes one.
    """

    additional_residential: UKPropertyPurchaseTarget | None
    non_residential: UKPropertyPurchaseTarget | None
    #: How the household HRAD targets were allocated (the published inputs).
    derivation: Mapping[str, float] = field(default_factory=dict)

    def for_kind(self, name: str) -> UKPropertyPurchaseTarget | None:
        if name == "additional_residential":
            return self.additional_residential
        if name == "non_residential":
            return self.non_residential
        raise KeyError(name)

    def payload(self) -> dict[str, object]:
        return {
            "additional_residential": (
                None
                if self.additional_residential is None
                else self.additional_residential.payload()
            ),
            "non_residential": (
                None if self.non_residential is None else self.non_residential.payload()
            ),
            "derivation": {key: float(value) for key, value in self.derivation.items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> UKPropertyPurchaseTargets:
        def kind(value: Mapping[str, Any] | None) -> UKPropertyPurchaseTarget | None:
            return (
                None if value is None else UKPropertyPurchaseTarget.from_payload(value)
            )

        return cls(
            additional_residential=kind(payload["additional_residential"]),
            non_residential=kind(payload["non_residential"]),
            derivation={
                str(key): float(value)
                for key, value in dict(payload.get("derivation") or {}).items()
            },
        )


def _record_set(row: Mapping[str, Any]) -> str:
    return str((row.get("layout") or {}).get("record_set_id") or "")


def _band(row: Mapping[str, Any]) -> str:
    return str((row.get("layout") or {}).get("groupby_value_id") or "")


def _cells(
    rows: Sequence[Mapping[str, Any]], *, record_set: str, concept: str
) -> dict[str, float]:
    """Values by band of one record set and concept, refused if duplicated."""

    cells: dict[str, float] = {}
    for row in rows:
        if _record_set(row) != record_set or row.get("concept") != concept:
            continue
        band = _band(row)
        if band in cells:
            raise ValueError(f"{record_set} {concept} repeats band {band!r}.")
        cells[band] = float(row["value"])
    if _TOTAL_BAND not in cells:
        raise ValueError(f"{record_set} carries no {concept} total ({_TOTAL_BAND}).")
    return cells


def _bands(
    counts: Mapping[str, float],
    values: Mapping[str, float],
    *,
    combine: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    detail_counts = {band: n for band, n in counts.items() if band != _TOTAL_BAND}
    detail_values = {band: v for band, v in values.items() if band != _TOTAL_BAND}
    if combine:
        missing = [band for band in combine if band not in detail_counts]
        if missing:
            raise ValueError(f"The bands to combine are missing: {missing}.")
        detail_counts["over_gbp_2m"] = sum(detail_counts.pop(band) for band in combine)
        detail_values["over_gbp_2m"] = sum(detail_values.pop(band) for band in combine)
    if set(detail_counts) != set(_BAND_ORDER) or set(detail_values) != set(_BAND_ORDER):
        raise ValueError(
            "The purchase price bands must be exactly "
            f"{_BAND_ORDER}; got {sorted(detail_counts)} / {sorted(detail_values)}."
        )
    return (
        _BAND_ORDER,
        tuple(float(detail_counts[band]) for band in _BAND_ORDER),
        tuple(float(detail_values[band]) for band in _BAND_ORDER),
    )


def property_purchase_targets_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> UKPropertyPurchaseTargets:
    """Household purchase targets from the vendored HMRC Stamp Tax rows."""

    minimum = HMRC_SDLT_MINIMUM_CONSIDERATION
    hrad_count = _cells(
        rows, record_set=_HRAD_RECORD_SET, concept="hmrc.sdlt.hrad_transactions"
    )
    hrad_value = _cells(
        rows, record_set=_HRAD_RECORD_SET, concept="hmrc.sdlt.hrad_property_value"
    )
    individuals_count = _cells(
        rows,
        record_set=_RESIDENTIAL_INDIVIDUALS_RECORD_SET,
        concept="hmrc.sdlt.transactions",
    )[_TOTAL_BAND]
    individuals_value = _cells(
        rows,
        record_set=_RESIDENTIAL_INDIVIDUALS_RECORD_SET,
        concept="hmrc.sdlt.property_value",
    )[_TOTAL_BAND]
    other_count = _cells(
        rows,
        record_set=_RESIDENTIAL_OTHER_BUYERS_RECORD_SET,
        concept="hmrc.sdlt.transactions",
    )[_TOTAL_BAND]
    other_value = _cells(
        rows,
        record_set=_RESIDENTIAL_OTHER_BUYERS_RECORD_SET,
        concept="hmrc.sdlt.property_value",
    )[_TOTAL_BAND]
    count_share = individuals_count / (individuals_count + other_count)
    value_share = individuals_value / (individuals_value + other_value)
    band_ids, band_counts, band_values = _bands(
        hrad_count, hrad_value, combine=_HRAD_TOP_BANDS
    )
    additional = UKPropertyPurchaseTarget(
        property_type="additional_residential",
        transactions=hrad_count[_TOTAL_BAND] * count_share,
        value=hrad_value[_TOTAL_BAND] * value_share,
        band_ids=band_ids,
        band_transactions=band_counts,
        band_values=band_values,
        minimum_price=minimum,
    )
    non_residential_count = _cells(
        rows,
        record_set=_NON_RESIDENTIAL_INDIVIDUALS_RECORD_SET,
        concept="hmrc.sdlt.transactions",
    )
    non_residential_value = _cells(
        rows,
        record_set=_NON_RESIDENTIAL_INDIVIDUALS_RECORD_SET,
        concept="hmrc.sdlt.property_value",
    )
    band_ids, band_counts, band_values = _bands(
        non_residential_count, non_residential_value, combine=()
    )
    non_residential = UKPropertyPurchaseTarget(
        property_type="non_residential",
        transactions=non_residential_count[_TOTAL_BAND],
        value=non_residential_value[_TOTAL_BAND],
        band_ids=band_ids,
        band_transactions=band_counts,
        band_values=band_values,
        minimum_price=minimum,
    )
    return UKPropertyPurchaseTargets(
        additional_residential=additional,
        non_residential=non_residential,
        derivation={
            "hrad_transactions_all_buyers": hrad_count[_TOTAL_BAND],
            "hrad_property_value_all_buyers": hrad_value[_TOTAL_BAND],
            "residential_individuals_transaction_share": count_share,
            "residential_individuals_value_share": value_share,
        },
    )


def load_property_purchase_targets() -> UKPropertyPurchaseTargets:
    """The targets from the committed vendored resource, checked against the pin."""

    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    return property_purchase_targets_from_rows(
        vendored_rows(PROPERTY_PURCHASES_RESOURCE)
    )


@dataclass(frozen=True)
class UKPropertyPurchaseImputation:
    """One kind's purchase prices and its aggregate receipt."""

    prices: np.ndarray
    receipt: dict[str, object]


def select_property_purchasers(
    *,
    stock: np.ndarray,
    weights: np.ndarray,
    sdlt_mask: np.ndarray,
    event_uniforms: np.ndarray,
    transactions: float,
    minimum_price: float,
    property_type: str,
) -> tuple[np.ndarray, float]:
    """Who buys: the purchasers' mask and the eligible fit-region owner weight.

    A stock-proportional hazard turns each household's uniform into an
    exponential event score; the threshold is the one whose weighted count of
    purchasers in the fit regions is closest to ``transactions``, and it applies
    to every eligible owner, inside the fit regions or not.
    """

    owners = (stock >= minimum_price) & (weights > 0)
    eligible = np.flatnonzero(owners & sdlt_mask)
    eligible_weight = float(weights[eligible].sum())
    if eligible_weight < transactions:
        raise ValueError(
            f"{property_type}: eligible England-and-Northern-Ireland owners carry "
            "less weight than the purchase target."
        )
    reference_stock = float(np.average(stock[eligible], weights=weights[eligible]))
    hazard = (
        np.maximum(stock, minimum_price) / reference_stock
    ) ** PURCHASE_STOCK_EXPONENT
    events = -np.log1p(-event_uniforms) / hazard
    ranked = eligible[np.argsort(events[eligible], kind="stable")]
    cumulative = np.concatenate(([0.0], np.cumsum(weights[ranked])))
    n_purchases = int(np.argmin(np.abs(cumulative - transactions)))
    if n_purchases == len(ranked):
        threshold = np.inf
    elif n_purchases == 0:
        threshold = 0.0
    else:
        threshold = (events[ranked[n_purchases - 1]] + events[ranked[n_purchases]]) / 2
    return owners & (events < threshold), eligible_weight


def impute_property_purchase_prices(
    *,
    stock: np.ndarray,
    weights: np.ndarray,
    sdlt_mask: np.ndarray,
    event_uniforms: np.ndarray,
    price_uniforms: np.ndarray,
    target: UKPropertyPurchaseTarget,
    source_ids: np.ndarray,
) -> UKPropertyPurchaseImputation:
    """Purchase prices for one kind, fitting the weighted count and value.

    uk-data#573's ``impute_property_purchase_values``, reading identity-keyed
    uniforms instead of positional seeded streams.
    """

    stock = np.asarray(stock, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sdlt_mask = np.asarray(sdlt_mask, dtype=bool)
    event_uniforms = np.asarray(event_uniforms, dtype=float)
    price_uniforms = np.asarray(price_uniforms, dtype=float)
    shape = stock.shape
    if (
        stock.ndim != 1
        or weights.shape != shape
        or sdlt_mask.shape != shape
        or event_uniforms.shape != shape
        or price_uniforms.shape != shape
    ):
        raise ValueError(
            "Property stocks, weights, the SDLT mask and the uniforms must be "
            "aligned vectors."
        )
    if (
        not np.all(np.isfinite(stock))
        or not np.all(np.isfinite(weights))
        or np.any(stock < 0)
        or np.any(weights < 0)
    ):
        raise ValueError("Property stocks and weights must be finite and non-negative.")
    if np.any((event_uniforms < 0) | (event_uniforms >= 1)) or np.any(
        (price_uniforms < 0) | (price_uniforms >= 1)
    ):
        raise ValueError("Purchase uniforms must lie in [0, 1).")

    minimum = target.minimum_price
    purchased, eligible_weight = select_property_purchasers(
        stock=stock,
        weights=weights,
        sdlt_mask=sdlt_mask,
        event_uniforms=event_uniforms,
        transactions=target.transactions,
        minimum_price=minimum,
        property_type=target.property_type,
    )
    fitted = purchased & sdlt_mask
    fitted_weights = weights * fitted
    capacity = float(np.dot(stock, fitted_weights))
    if capacity < PURCHASE_MINIMUM_VALUE_SHARE * target.value:
        raise ValueError(
            f"{target.property_type}: the purchasers' stock carries less than "
            f"{PURCHASE_MINIMUM_VALUE_SHARE:.0%} of the purchase value target."
        )
    if minimum * fitted_weights.sum() > target.value:
        raise ValueError(
            f"{target.property_type}: the purchase value target is below the "
            "minimum transaction price."
        )

    cumulative_shares = np.cumsum(target.band_shares())
    band_index = np.minimum(
        np.searchsorted(cumulative_shares, price_uniforms, side="right"),
        len(cumulative_shares) - 1,
    )
    draws = target.band_prices()[band_index]

    def capped(scale: float) -> np.ndarray:
        return np.minimum(stock, np.maximum(minimum, draws * scale))

    def weighted_value(scale: float) -> float:
        return float(np.dot(capped(scale), fitted_weights))

    scale: float | None
    if capacity < target.value:
        # The stock cannot carry the value: every purchaser buys at its
        # stock, the most the cap allows, and the receipt records the
        # shortfall (María's ruling of 2026-10-16).
        scale = None
        final = stock.copy()
    else:
        lower, upper = 0.0, 1.0
        while weighted_value(upper) < target.value:
            upper *= 2.0
        for _ in range(_BISECTION_STEPS):
            midpoint = (lower + upper) / 2.0
            if weighted_value(midpoint) < target.value:
                lower = midpoint
            else:
                upper = midpoint
        scale = upper
        final = capped(upper)
    prices = np.where(purchased, final, 0.0)
    fitted_count = float(fitted_weights.sum())
    at_cap = fitted & (final >= stock)
    outside = purchased & ~sdlt_mask
    ids = np.asarray(source_ids)

    def sources(mask: np.ndarray) -> int:
        return distinct_count(ids, mask)

    # UKDS disclosure control (receipt_counts): a weighted figure one to nine
    # source households carry is "<10", and so is anything derived from it.
    fitted_sources = sources(fitted)
    at_cap_sources = sources(at_cap)
    # Copies of one source can sit on both sides, so each side is counted on
    # its own distinct sources, not as the fit's less the other side's.
    uncapped_sources = sources(fitted & ~at_cap)
    eligible = (stock >= minimum) & (weights > 0) & sdlt_mask
    eligible_sources = sources(eligible)
    # The eligible owner weight less the fitted transactions is the eligible
    # owners who do not buy, and so is that weight times one less the event
    # rate: both are given only where neither the buyers nor the others rest
    # on one to nine sources.
    eligible_small = 0 < min(fitted_sources, sources(eligible & ~fitted)) < 10
    fitted_value = float(np.dot(prices, weights * sdlt_mask))
    at_cap_share = float(weights[at_cap].sum()) / fitted_count if fitted_count else 0.0
    # The share at the cap is withheld where the fit or either side of it
    # rests on one to nine sources: beside the fitted weight a small side
    # would come back, and a share of one beside a small fit still describes it.
    at_cap_small = (
        0 < fitted_sources < 10 or 0 < min(at_cap_sources, uncapped_sources) < 10
    )
    receipt: dict[str, object] = {
        "property_type": target.property_type,
        "target_transactions": float(target.transactions),
        "target_value": float(target.value),
        "fitted_transactions": small_cell_share(fitted_count, fitted_sources),
        "fitted_value": small_cell_share(fitted_value, fitted_sources),
        "value_fit": "scaled_band_draws" if scale is not None else "stock_cap",
        "value_shortfall_share": small_cell_share(
            max(0.0, 1.0 - fitted_value / target.value), fitted_sources
        ),
        "eligible_owner_weight": (
            "<10"
            if eligible_small
            else small_cell_share(eligible_weight, eligible_sources)
        ),
        "event_rate": (
            "<10"
            if eligible_small
            else small_cell_share(fitted_count / eligible_weight, fitted_sources)
        ),
        "price_scale": (
            None if scale is None else small_cell_share(scale, fitted_sources)
        ),
        "weighted_share_at_stock_cap": "<10" if at_cap_small else at_cap_share,
        "stock_cap_support": extreme_count(at_cap_sources),
        "purchases_outside_sdlt_regions_weight": small_cell_share(
            float(weights[outside].sum()), sources(outside)
        ),
        "outside_sdlt_regions_support": support_band(sources(outside)),
        "purchaser_support": support_band(fitted_sources),
    }
    return UKPropertyPurchaseImputation(prices=prices, receipt=receipt)


def property_purchases_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            "verify_vendored_fact_resource",
            {
                "artifact_role": "stamp_tax_facts",
                "resource": PROPERTY_PURCHASES_RESOURCE,
                "feed_pin": "chronicle_feed.json",
                "record_sets": list(PROPERTY_PURCHASES_RECORD_SETS),
                "source_vintage": "2024-25",
                "mapped_build_period": PROPERTY_PURCHASES_BUILD_PERIOD,
                "period_mapping": "published_tax_year_equals_build_period",
                "require_before_source_read": True,
                "runtime_sha256_required": True,
                "fail_on_mismatch": True,
            },
        ),
        (
            PROPERTY_PURCHASES_OPERATION_KIND,
            {
                "outputs": list(PROPERTY_PURCHASES_OUTPUT_COLUMNS),
                "stocks": [kind.stock for kind in PROPERTY_PURCHASE_KINDS],
                "port_of": "uk-data#573 at 07f6b460f",
                "minimum_price": HMRC_SDLT_MINIMUM_CONSIDERATION,
                "eligibility": (
                    "stock at least minimum_price, the chargeable consideration "
                    "HMRC's Stamp Tax tables cover, and positive weight; at most "
                    "one purchase of each kind a year"
                ),
                "fit_regions": list(SDLT_REGIONS),
                "hazard": "(max(stock, minimum) / weighted mean eligible stock) ** 1",
                "event_score": "-log1p(-u) / hazard",
                "threshold": (
                    "the score threshold whose weighted count of purchasers in the "
                    "fit regions is closest to the transactions target, applied "
                    "everywhere"
                ),
                "price": (
                    "min(stock, max(minimum, scale * band mean)), band drawn at "
                    "the band shares, scale bisected to the value target; where "
                    "the purchasers' stock cannot carry the value, every "
                    "purchaser buys at its stock and the shortfall is recorded"
                ),
                "minimum_value_share": PURCHASE_MINIMUM_VALUE_SHARE,
                "weights": "spine_design_weights",
                "streams": {
                    kind.output: {
                        "event_salt": kind.event_salt,
                        "price_salt": kind.price_salt,
                        "seed": _STREAM_SEED,
                    }
                    for kind in PROPERTY_PURCHASE_KINDS
                },
                "additional_residential_targets": (
                    "Table 6b HRAD transactions and value, 2024-25, times "
                    "individuals' shares of Table 8a residential transactions and "
                    "value; Table 6b bands with the five above GBP 2m combined"
                ),
                "non_residential_targets": (
                    "Table 8b individuals' transactions, value and bands, 2024-25"
                ),
            },
        ),
        (
            "record_mass_conservation_receipt",
            {
                "entity": "household",
                "reason": PROPERTY_PURCHASES_MASS_CHANGE_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )


@dataclass(frozen=True)
class UKPropertyPurchasesResult:
    """The frame with both purchases, and the stage's aggregate receipt."""

    frame: Frame
    receipts: tuple[dict[str, object], ...]
    targets: UKPropertyPurchaseTargets

    def evidence(self) -> dict[str, object]:
        return {
            "stage": PROPERTY_PURCHASES_STAGE_NAME,
            "port_of": "uk-data#573 at 07f6b460f",
            "kinds": [dict(receipt) for receipt in self.receipts],
            "derivation": dict(self.targets.derivation),
        }


def sdlt_region_mask(regions: pd.Series) -> np.ndarray:
    """Households where SDLT applies (England and Northern Ireland)."""

    values = regions.astype(str)
    unknown = sorted(set(values) - set(FRS_REGION_TO_COUNTRY))
    if unknown:
        raise ValueError(f"Property purchases met unknown FRS regions: {unknown}.")
    return values.isin(SDLT_REGIONS).to_numpy()


def impute_property_purchases(
    frame: Frame, *, targets: UKPropertyPurchaseTargets
) -> UKPropertyPurchasesResult:
    """Add both annual purchases at the frame's household weights."""

    validate_uk_national_frame(frame)
    period = int(uk_time_period(frame))
    if period != PROPERTY_PURCHASES_BUILD_PERIOD:
        raise ValueError(
            "The 2024-25 Stamp Tax facts are mapped to build period "
            f"{PROPERTY_PURCHASES_BUILD_PERIOD}, but the frame's period is {period}."
        )
    household = frame.table("household").copy()
    for kind in PROPERTY_PURCHASE_KINDS:
        if kind.output in household.columns:
            raise ValueError(
                f"{kind.output!r} is already stored: this stage is its only writer."
            )
        if kind.stock not in household.columns:
            raise ValueError(f"Property purchases need the stock {kind.stock!r}.")
    weights = frame.weights_for("household")
    weight_values = np.asarray(weights.values, dtype=float)
    regions = household["region"].astype(str)
    sdlt_mask = sdlt_region_mask(regions)
    source_ids = source_household_ids(household, stage="The property purchases stage")
    ids = household["household_id"].to_numpy()
    receipts = []
    for kind in PROPERTY_PURCHASE_KINDS:
        target = targets.for_kind(kind.name)
        if target is None:
            household[kind.output] = 0.0
            receipts.append({"property_type": kind.name, "imputed": False})
            continue
        stock = pd.to_numeric(household[kind.stock], errors="raise").to_numpy(
            dtype=float
        )
        event_uniforms, price_uniforms = property_purchase_uniforms(ids, kind)
        imputation = impute_property_purchase_prices(
            stock=stock,
            weights=weight_values,
            sdlt_mask=sdlt_mask,
            event_uniforms=event_uniforms,
            price_uniforms=price_uniforms,
            target=target,
            source_ids=source_ids,
        )
        household[kind.output] = imputation.prices
        receipts.append(imputation.receipt)
    mass = MassChangeRecord(
        entity="household",
        old_total=weights.total,
        new_total=weights.total,
        declared_factor=1.0,
        reason=PROPERTY_PURCHASES_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=frame.table("person").copy(),
        benunit=frame.table("benunit").copy(),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=weights.values,
        mass_log=(*frame.mass_log, mass),
    )
    validate_uk_national_frame(result)
    return UKPropertyPurchasesResult(
        frame=result, receipts=tuple(receipts), targets=targets
    )


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind the manifest to the reviewed code constants (closed world)."""

    _assert_closed_world_operations(stage, property_purchases_operation_parameters())
    if stage.stage != PROPERTY_PURCHASES_STAGE_NAME:
        raise ValueError(
            f"Expected stage {PROPERTY_PURCHASES_STAGE_NAME!r}, got {stage.stage!r}."
        )
    if tuple(stage.outputs) != PROPERTY_PURCHASES_OUTPUT_COLUMNS or tuple(
        stage.rewrites
    ):
        raise ValueError(
            "The property purchases stage writes exactly "
            f"{PROPERTY_PURCHASES_OUTPUT_COLUMNS} and rewrites nothing."
        )
    if stage.grain != "household":
        raise ValueError(
            f"The property purchases stage is household grain, got {stage.grain!r}."
        )


@dataclass(frozen=True)
class UKPropertyPurchasesStageTransform:
    """Whole-stage callable for the annual property purchases."""

    stage: SourceStageSpec
    #: The targets; read from the vendored HMRC facts when not given (the
    #: synthetic parity fixture sizes its own to its frame).
    targets: UKPropertyPurchaseTargets | None = field(default=None, repr=False)
    last_result: UKPropertyPurchasesResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        targets = (
            self.targets
            if self.targets is not None
            else load_property_purchase_targets()
        )
        result = impute_property_purchases(frame, targets=targets)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return PROPERTY_PURCHASES_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
