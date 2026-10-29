"""ISA subscriptions per adult, imputed from HMRC's ISA tables (microcosm#1415).

The FRS and the Wealth and Assets Survey record ISA balances, not what people pay
in during a year, so no survey donor exists for subscriptions. This stage, the
last in the UK roster, draws them from HMRC's published tables and writes five
person inputs policyengine-uk reads: ``cash_isa_balance``,
``stocks_and_shares_isa_balance``, ``cash_isa_subscriptions``,
``stocks_and_shares_isa_subscriptions`` and ``lifetime_isa_payments``. It rewrites
no household column, so every earlier reader of the ``was_wealth`` household
totals and every means test is unchanged.

The steps, every draw keyed on identity, so reordering rows moves none. Adding
rows can move others' amounts: the rakes, the band walks and the growth factors
read the whole population.

1. **Per-person balances.** Each household's ``cash_isa`` and
   ``stocks_and_shares_isa`` (the ``was_wealth`` draws) are split across its adults
   by the pattern of a WAS round-8 donor household with the same number of adults
   and the same types held. Who in a donor household holds what is read jointly
   from the person tab (``DVCISAvR8``, ``DVIISAvR8``), whose values must sum to the
   household aggregates; person values sum exactly to the household draw.
2. **Who subscribes (2023-24).** For the types paid into (both, then stocks and
   shares only, then cash only), eligible adults (18 or over, holding what they
   pay into) are raked over income band x age band x sex x region to Tables 9.7,
   9.8 and 9.9, and selected inside each cell by a keyed systematic walk. A cell
   with fewer eligible adults than its target is reported, and its excess moves to
   cells of the same income band with room; nobody is made a holder.
3. **How much (2023-24).** Within each (type, income band) the subscribers walk
   Table 9.7's band masses from the top, ranked by a key that blends their balance
   rank with a keyed uniform, so larger holders tend to pay in more. Amounts are
   spread evenly inside each band on weight-proportional strata; GBP 20,000 is a
   point mass.
4. **The split for people paying into both** (María's ruling, 28 October 2026).
   Each person's cash share follows their own balances; one common logit shift sets
   the group's stocks-and-shares total to Table 9.4's stocks-and-shares amount less
   the stocks-and-shares-only payments. Table 9.7's cash bands imply more cash than
   Table 9.4 records, so the cash residual cannot anchor it; Table 9.7's people
   are kept and the cash excess over Table 9.4 is reported.
5. **Growth to 2024-25.** Subscriber counts grow by Table 9.4's account growth by
   type, an upper bound on people because from 6 April 2024 one person may pay into
   several ISAs of one type. Amounts then scale by one factor per type to Table
   9.4's 2024-25 amounts; subscribers already at GBP 20,000 stay there, amounts
   between GBP 18,000 and the limit move on a line up to it so no range below it
   empties, and the rest are capped at it (María's rulings of 29 October 2026).
6. **Lifetime ISA payments (2024-25).** Holders aged 18 to 47, the ages that can
   hold one in 2024-25, pay in amounts on (0, GBP 4,000], tilted so the total
   equals Table 9.4's Lifetime ISA amount and ranked like step 3, within each
   person's room under GBP 20,000; a holder with no room pays nothing.

Receipts carry weighted aggregates and rounded counts only (microcosm#1164).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.cgt_imputation import (
    _truncated_exponential_quantile,
    _walk_bands,
    uk_cgt_component_sum_income,
)
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.hmrc_isa_statistics import (
    HMRC_ISA_AGE_BAND_IDS,
    HMRC_ISA_BUILD_TAX_YEAR,
    HMRC_ISA_INCOME_BAND_IDS,
    HMRC_ISA_INDIVIDUAL_TAX_YEAR,
    HMRC_ISA_REGION_IDS,
    HMRC_ISA_SUBSCRIPTION_BAND_IDS,
    HMRC_ISA_TYPES,
    HMRCISAStatistics,
    check_isa_statistics_declaration,
    isa_band_bounds,
    lifetime_isa_paying_ages,
    load_hmrc_isa_statistics,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import rounded_count
from microcosm.calibrate.geography_constants import UK_REGION_TIER_ENUM
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

__all__ = [
    "UK_ISA_SUBSCRIPTIONS_MASS_CONSERVATION_REASON",
    "UK_ISA_SUBSCRIPTIONS_NONNEGATIVE_OUTPUT_COLUMNS",
    "UK_ISA_SUBSCRIPTIONS_OUTPUT_COLUMNS",
    "UK_ISA_SUBSCRIPTIONS_STAGE_NAME",
    "ISASubscriptionsError",
    "ISASubscriptionsSpec",
    "UKISASubscriptionsStageTransform",
    "WASISADonor",
    "clean_was_isa_person_donor",
    "impute_isa_subscriptions",
    "split_household_isa_balances",
]

UK_ISA_SUBSCRIPTIONS_STAGE_NAME = "isa_subscriptions"
#: The household-mass receipt this stage records (the manifest's
#: ``record_mass_conservation_receipt`` operation repeats it).
UK_ISA_SUBSCRIPTIONS_MASS_CONSERVATION_REASON = (
    "ISA subscriptions per adult from HMRC's ISA tables on the source spine: "
    "household weights pass through unchanged and total household mass is conserved."
)
UK_WAS_HOUSEHOLD_TAB_ROLE = "was_qrf_donor"
UK_WAS_PERSON_TAB_ROLE = "was_person_tab"

CLEAN_WAS_ISA_PERSON_DONOR_KIND = "clean_was_isa_person_donor"
SPLIT_HOUSEHOLD_ISA_BALANCES_KIND = "split_household_isa_balances"
ALLOCATE_ISA_SUBSCRIBERS_KIND = "allocate_isa_subscribers"
DRAW_ISA_SUBSCRIPTION_AMOUNTS_KIND = "draw_isa_subscription_amounts"
SPLIT_BOTH_TYPE_SUBSCRIPTIONS_KIND = "split_both_type_subscriptions"
GROW_ISA_SUBSCRIPTIONS_KIND = "grow_isa_subscriptions"
DRAW_LIFETIME_ISA_PAYMENTS_KIND = "draw_lifetime_isa_payments"
VERIFY_VENDORED_FACT_RESOURCE_KIND = "verify_vendored_fact_resource"
CLASSIFY_ISA_FACTS_KIND = "classify_isa_facts_with_reviewed_fence"

#: The reviewed fence over every vendored ISA fact (the manifest's
#: ``classify_isa_facts_with_reviewed_fence`` operation repeats it exactly).
UK_ISA_FACT_FENCE: dict[str, Any] = {
    "calibration_permitted": False,
    "fact_fence_id": "isa_facts_imputation_seeds_measured_only",
    "fenced_fact_count": 1074,
    "fenced_fact_composition": (
        "Table 9.7 2023-24 subscribers by income and subscription band and holders "
        "making no further subscription, for all subscribers and each type paid "
        "into (624); Table 9.8 by age and sex (325); Table 9.9 by region (70); "
        "Table 9.4 accounts and amounts for 2023-24 and 2024-25 (24); Table 9.6 "
        "2024-25 market values (19); the impact note's shares of under-65 cash "
        "subscribers either side of GBP 12,000 (2); Lifetime ISA Table 1a for "
        "2024-25 and 2025-26 (10)"
    ),
    "classification_rationale": (
        "The stage draws subscribers and amounts to these cells, so binding them in "
        "the weight solve would be circular. The national build reports the "
        "subscription cells after calibration, with the holders (Tables 9.10 to "
        "9.12) and market values (Table 9.6) as independent checks; none blocks a "
        "build."
    ),
    "adjudication": "https://github.com/PolicyEngine/microcosm/issues/1415",
}

CASH_BALANCE = "cash_isa_balance"
STOCKS_AND_SHARES_BALANCE = "stocks_and_shares_isa_balance"
CASH_SUBSCRIPTIONS = "cash_isa_subscriptions"
STOCKS_AND_SHARES_SUBSCRIPTIONS = "stocks_and_shares_isa_subscriptions"
LIFETIME_ISA_PAYMENTS = "lifetime_isa_payments"
UK_ISA_SUBSCRIPTIONS_OUTPUT_COLUMNS: tuple[str, ...] = (
    CASH_BALANCE,
    STOCKS_AND_SHARES_BALANCE,
    CASH_SUBSCRIPTIONS,
    STOCKS_AND_SHARES_SUBSCRIPTIONS,
    LIFETIME_ISA_PAYMENTS,
)
UK_ISA_SUBSCRIPTIONS_NONNEGATIVE_OUTPUT_COLUMNS = UK_ISA_SUBSCRIPTIONS_OUTPUT_COLUMNS

HOUSEHOLD_CASH = "cash_isa"
HOUSEHOLD_STOCKS_AND_SHARES = "stocks_and_shares_isa"
LIFETIME_ISA_HOLDER = "has_lifetime_isa"
LIFETIME_ISA_BALANCE = "lifetime_isa_balance"

BOTH = "cash_and_stocks_and_shares"
STOCKS_AND_SHARES_ONLY = "stocks_and_shares_only"
CASH_ONLY = "cash_only"
#: Where 2024-25 amounts start to taper to the overall limit: the floor of Table
#: 9.7's last subscription band below it (GBP 18,000, María's ruling of 29
#: October 2026).
TAPER_FROM_GBP = float(isa_band_bounds(HMRC_ISA_SUBSCRIPTION_BAND_IDS[-2])[0] - 1)

#: Selection order: the scarcest holding pattern first, so a person who holds
#: both types is offered the 'both' class before a single-type one.
CLASS_ORDER: tuple[str, ...] = (BOTH, STOCKS_AND_SHARES_ONLY, CASH_ONLY)
if set(CLASS_ORDER) != set(HMRC_ISA_TYPES):
    raise RuntimeError("CLASS_ORDER must cover exactly HMRC_ISA_TYPES.")

#: Spine region enum name -> Table 9.9's row id.
_REGION_ROW: Mapping[str, str] = {
    name: code for code, name in UK_REGION_TIER_ENUM.items()
}
_OVERSEAS = "overseas_and_unknown"
if set(_REGION_ROW.values()) | {_OVERSEAS} != set(HMRC_ISA_REGION_IDS):
    raise RuntimeError("Table 9.9's rows must be the region tier plus overseas.")


class ISASubscriptionsError(ValueError):
    """Raised when the inputs or the declaration do not carry what the stage needs."""


class _KeyedOffset:
    """One identity-keyed offset in the shape ``_walk_bands`` reads (``random()``)."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def random(self) -> float:
        return self._value


def _keyed_offset(seed: int, salt: str, key: str) -> _KeyedOffset:
    return _KeyedOffset(stable_identity_uniforms([key], seed=seed, salt=salt)[0])


# --------------------------------------------------------------------------
# Declaration
# --------------------------------------------------------------------------


def _operation(stage: SourceStageSpec, kind: str) -> Mapping[str, Any]:
    for operation in stage.operations:
        if operation.kind == kind:
            return dict(operation.parameters)
    raise ISASubscriptionsError(f"{stage.stage} declares no {kind!r} operation.")


def _artifact(stage: SourceStageSpec, role: str) -> Mapping[str, Any]:
    for artifact in stage.artifacts:
        if artifact.get("role") == role:
            return artifact
    raise ISASubscriptionsError(f"{stage.stage} declares no {role!r} artifact.")


@dataclass(frozen=True)
class WASISAPersonColumns:
    """The declared WAS person-tab names and codes the balance split reads."""

    household_key: str
    household_weight: str
    person_number: str
    is_dependent_child: str
    age_band: str
    cash_isa_value: str
    stocks_and_shares_isa_value: str
    cash_isa_aggregate: str
    stocks_and_shares_isa_aggregate: str
    aggregate_tolerance_gbp: float
    non_dependent_code: int

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> WASISAPersonColumns:
        person = parameters.get("person_columns")
        aggregates = parameters.get("household_aggregates")
        if not isinstance(person, Mapping) or not isinstance(aggregates, Mapping):
            raise ISASubscriptionsError(
                "clean_was_isa_person_donor needs person_columns and household_aggregates."
            )
        try:
            return cls(
                household_key=str(parameters["household_key"]),
                household_weight=str(parameters["household_weight"]),
                person_number=str(person["person_number"]),
                is_dependent_child=str(person["is_dependent_child"]),
                age_band=str(person["age_band"]),
                cash_isa_value=str(person["cash_isa_value"]),
                stocks_and_shares_isa_value=str(person["stocks_and_shares_isa_value"]),
                cash_isa_aggregate=str(aggregates["cash_isa"]),
                stocks_and_shares_isa_aggregate=str(
                    aggregates["stocks_and_shares_isa"]
                ),
                aggregate_tolerance_gbp=float(parameters["aggregate_tolerance_gbp"]),
                non_dependent_code=int(parameters["non_dependent_code"]),
            )
        except KeyError as error:
            raise ISASubscriptionsError(
                f"clean_was_isa_person_donor lacks {error.args[0]!r}."
            ) from None

    def person_raw_columns(self) -> tuple[str, ...]:
        return (
            self.household_key,
            self.person_number,
            self.is_dependent_child,
            self.age_band,
            self.cash_isa_value,
            self.stocks_and_shares_isa_value,
        )


@dataclass(frozen=True)
class ISASubscriptionsSpec:
    """Every declared parameter of the stage, read once from the manifest."""

    columns: WASISAPersonColumns
    split_seed: int
    split_salt: str
    maximum_donor_adults: int
    minimum_age: int
    select_seed: int
    select_salt: str
    amount_seed: int
    amount_salt: str
    balance_rank_weight: float
    share_floor: float
    base_tax_year: int
    build_tax_year: int
    overall_limit_gbp: float
    taper_from_gbp: float
    lifetime_seed: int
    lifetime_salt: str
    lifetime_minimum_age: int
    lifetime_maximum_age: int
    lifetime_limit_gbp: float

    @classmethod
    def from_stage(cls, stage: SourceStageSpec) -> ISASubscriptionsSpec:
        clean = _operation(stage, CLEAN_WAS_ISA_PERSON_DONOR_KIND)
        split = _operation(stage, SPLIT_HOUSEHOLD_ISA_BALANCES_KIND)
        select = _operation(stage, ALLOCATE_ISA_SUBSCRIBERS_KIND)
        amounts = _operation(stage, DRAW_ISA_SUBSCRIPTION_AMOUNTS_KIND)
        both = _operation(stage, SPLIT_BOTH_TYPE_SUBSCRIPTIONS_KIND)
        grow = _operation(stage, GROW_ISA_SUBSCRIPTIONS_KIND)
        lifetime = _operation(stage, DRAW_LIFETIME_ISA_PAYMENTS_KIND)
        if _operation(stage, CLASSIFY_ISA_FACTS_KIND) != UK_ISA_FACT_FENCE:
            raise ISASubscriptionsError(
                "the declared ISA fact fence differs from the reviewed one: no ISA "
                "fact is a calibration target."
            )
        for operation, outputs in (
            (split, [CASH_BALANCE, STOCKS_AND_SHARES_BALANCE]),
            (amounts, [CASH_SUBSCRIPTIONS, STOCKS_AND_SHARES_SUBSCRIPTIONS]),
            (lifetime, [LIFETIME_ISA_PAYMENTS]),
        ):
            if list(operation.get("outputs", ())) != outputs:
                raise ISASubscriptionsError(
                    f"the declaration writes {operation.get('outputs')!r}, the stage "
                    f"writes {outputs!r}."
                )
        if select.get("selection_order") != list(CLASS_ORDER):
            raise ISASubscriptionsError(
                f"allocate_isa_subscribers must select in the order {list(CLASS_ORDER)}."
            )
        if both.get("anchor") != "table_9_4_stocks_and_shares_residual":
            raise ISASubscriptionsError(
                "the both-type split is anchored on Table 9.4's stocks-and-shares residual."
            )
        if amounts.get("within_band") != "uniform":
            raise ISASubscriptionsError("amounts are spread evenly inside each band.")
        if (
            grow.get("people") != "table_9_4_account_growth"
            or grow.get("amounts") != "table_9_4_amount_level_tapered_to_limit"
        ):
            raise ISASubscriptionsError(
                "growth is people by Table 9.4's account growth and amounts to its "
                "build-year amounts, holding and tapering to the overall limit."
            )
        taper_from = float(grow["taper_from_gbp"])
        if taper_from != TAPER_FROM_GBP:
            raise ISASubscriptionsError(
                f"the taper starts at Table 9.7's last band floor below the limit, "
                f"GBP {TAPER_FROM_GBP:,.0f}, not GBP {taper_from:,.0f}."
            )
        years = (int(grow["from_tax_year"]), int(grow["to_tax_year"]))
        if years != (HMRC_ISA_INDIVIDUAL_TAX_YEAR, HMRC_ISA_BUILD_TAX_YEAR):
            raise ISASubscriptionsError(
                f"growth runs from the individual tables' {HMRC_ISA_INDIVIDUAL_TAX_YEAR} "
                f"to Table 9.4's {HMRC_ISA_BUILD_TAX_YEAR}, not {years}."
            )
        paying_ages = lifetime_isa_paying_ages(HMRC_ISA_BUILD_TAX_YEAR)
        declared_ages = (int(lifetime["minimum_age"]), int(lifetime["maximum_age"]))
        if declared_ages != paying_ages:
            raise ISASubscriptionsError(
                f"Lifetime ISA payments in {HMRC_ISA_BUILD_TAX_YEAR}-"
                f"{(HMRC_ISA_BUILD_TAX_YEAR + 1) % 100:02d} come from holders aged "
                f"{paying_ages[0]} to {paying_ages[1]}, not {declared_ages}."
            )
        weight = float(amounts["balance_rank_weight"])
        if not 0.0 <= weight <= 1.0:
            raise ISASubscriptionsError("balance_rank_weight must lie in [0, 1].")
        floor = float(both["share_floor"])
        if not 0.0 < floor < 0.5:
            raise ISASubscriptionsError("share_floor must lie in (0, 0.5).")
        return cls(
            columns=WASISAPersonColumns.from_parameters(clean),
            split_seed=int(split["seed"]),
            split_salt=str(split["salt"]),
            maximum_donor_adults=int(split["maximum_donor_adults"]),
            minimum_age=int(select["minimum_age"]),
            select_seed=int(select["seed"]),
            select_salt=str(select["salt"]),
            amount_seed=int(amounts["seed"]),
            amount_salt=str(amounts["salt"]),
            balance_rank_weight=weight,
            share_floor=floor,
            base_tax_year=int(grow["from_tax_year"]),
            build_tax_year=int(grow["to_tax_year"]),
            overall_limit_gbp=float(grow["overall_limit_gbp"]),
            taper_from_gbp=taper_from,
            lifetime_seed=int(lifetime["seed"]),
            lifetime_salt=str(lifetime["salt"]),
            lifetime_minimum_age=int(lifetime["minimum_age"]),
            lifetime_maximum_age=int(lifetime["maximum_age"]),
            lifetime_limit_gbp=float(lifetime["annual_limit_gbp"]),
        )


# --------------------------------------------------------------------------
# Donor: who in a WAS household holds what
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WASISADonor:
    """Donor households: adults in age order, their shares and the household weight."""

    adult_count: np.ndarray
    holds_cash: np.ndarray
    holds_stocks_and_shares: np.ndarray
    weight: np.ndarray
    cash_shares: tuple[np.ndarray, ...]
    stocks_and_shares_shares: tuple[np.ndarray, ...]
    receipt: Mapping[str, Any]


def _lowered(table: pd.DataFrame, label: str) -> dict[str, str]:
    lowered: dict[str, str] = {}
    for column in table.columns:
        key = str(column).lower()
        if key in lowered:
            raise ISASubscriptionsError(f"the WAS {label} tab repeats column {key!r}.")
        lowered[key] = column
    return lowered


def _numeric(
    table: pd.DataFrame, lowered: Mapping[str, str], name: str, label: str
) -> pd.Series:
    actual = lowered.get(name.lower())
    if actual is None:
        raise ISASubscriptionsError(
            f"the WAS {label} tab lacks the declared column {name!r}."
        )
    return pd.to_numeric(table[actual], errors="coerce")


def clean_was_isa_person_donor(
    person_raw: pd.DataFrame,
    household_raw: pd.DataFrame,
    *,
    columns: WASISAPersonColumns,
    maximum_adults: int,
) -> WASISADonor:
    """Read each WAS household's adults and their cash and stocks-and-shares shares.

    The person values must sum to the household aggregates on every household or
    the tab is refused, and a missing or negative value is refused rather than read
    as zero. Adults are ordered oldest first (age band, then person number), the
    order recipients are matched in.
    """

    person_lower = _lowered(person_raw, "person")
    household_lower = _lowered(household_raw, "household")
    keys = _numeric(household_raw, household_lower, columns.household_key, "household")
    if keys.isna().any() or keys.duplicated().any():
        raise ISASubscriptionsError(
            "the WAS household tab has a missing or repeated case."
        )
    aggregates = pd.DataFrame(
        {
            "household_key": keys.to_numpy(),
            "weight": _numeric(
                household_raw, household_lower, columns.household_weight, "household"
            ).to_numpy(float),
            "cash": _numeric(
                household_raw, household_lower, columns.cash_isa_aggregate, "household"
            )
            .fillna(0.0)
            .to_numpy(float),
            "stocks_and_shares": _numeric(
                household_raw,
                household_lower,
                columns.stocks_and_shares_isa_aggregate,
                "household",
            )
            .fillna(0.0)
            .to_numpy(float),
        }
    ).set_index("household_key")
    person = pd.DataFrame(
        {
            "household_key": _numeric(
                person_raw, person_lower, columns.household_key, "person"
            ),
            "person_number": _numeric(
                person_raw, person_lower, columns.person_number, "person"
            ),
            "dependent": _numeric(
                person_raw, person_lower, columns.is_dependent_child, "person"
            ),
            "age_band": _numeric(person_raw, person_lower, columns.age_band, "person"),
            "cash": _numeric(
                person_raw, person_lower, columns.cash_isa_value, "person"
            ),
            "stocks_and_shares": _numeric(
                person_raw, person_lower, columns.stocks_and_shares_isa_value, "person"
            ),
        }
    )
    if person["household_key"].isna().any():
        raise ISASubscriptionsError("the WAS person tab has a missing case number.")
    if person.duplicated(["household_key", "person_number"]).any():
        raise ISASubscriptionsError("the WAS person tab repeats a (case, person) pair.")
    if not person["household_key"].isin(aggregates.index).all():
        raise ISASubscriptionsError("every WAS person must map to a household row.")
    for column in ("cash", "stocks_and_shares"):
        values = person[column]
        if values.isna().any() or (values < 0).any():
            raise ISASubscriptionsError(
                f"the WAS person {column} ISA values must be present and non-negative."
            )
        totals = values.groupby(person["household_key"]).sum()
        published = aggregates[column].reindex(totals.index).fillna(0.0)
        if float((totals - published).abs().max()) > columns.aggregate_tolerance_gbp:
            raise ISASubscriptionsError(
                f"the WAS person {column} ISA values do not sum to the household aggregate."
            )
    adults = person.loc[person["dependent"] == columns.non_dependent_code].copy()
    adults = adults.sort_values(
        ["household_key", "age_band", "person_number"],
        ascending=[True, False, True],
        kind="stable",
    )
    counts, cash_flags, stock_flags, weights, cash_shares, stock_shares = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for key, group in adults.groupby("household_key", sort=True):
        household_total_cash = float(group["cash"].sum())
        household_total_stocks = float(group["stocks_and_shares"].sum())
        if household_total_cash <= 0 and household_total_stocks <= 0:
            continue
        group = group.head(maximum_adults)
        counts.append(len(group))
        cash_flags.append(household_total_cash > 0)
        stock_flags.append(household_total_stocks > 0)
        weights.append(float(aggregates.at[key, "weight"]))
        cash_shares.append(_shares(group["cash"].to_numpy(float)))
        stock_shares.append(_shares(group["stocks_and_shares"].to_numpy(float)))
    weight_array = np.asarray(weights, dtype=float)
    if (weight_array <= 0).any() or not np.isfinite(weight_array).all():
        raise ISASubscriptionsError(
            "every WAS donor household needs a positive weight."
        )
    count_array = np.asarray(counts, dtype=np.int64)
    cash_array = np.asarray(cash_flags, dtype=bool)
    stock_array = np.asarray(stock_flags, dtype=bool)
    receipt = {
        "donor_households": rounded_count(len(count_array)),
        "by_pattern": {
            pattern: rounded_count(int(mask.sum()))
            for pattern, mask in (
                ("cash_only", cash_array & ~stock_array),
                ("stocks_and_shares_only", ~cash_array & stock_array),
                ("both", cash_array & stock_array),
            )
        },
        "by_adult_count": {
            str(count): rounded_count(int((count_array == count).sum()))
            for count in range(1, maximum_adults + 1)
        },
        "aggregate_pin_tolerance_gbp": columns.aggregate_tolerance_gbp,
    }
    return WASISADonor(
        adult_count=count_array,
        holds_cash=cash_array,
        holds_stocks_and_shares=stock_array,
        weight=weight_array,
        cash_shares=tuple(cash_shares),
        stocks_and_shares_shares=tuple(stock_shares),
        receipt=receipt,
    )


def _shares(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.zeros(len(values), dtype=float)
    return values / total


# --------------------------------------------------------------------------
# Step 1: per-person balances
# --------------------------------------------------------------------------


def split_household_isa_balances(
    *,
    person: pd.DataFrame,
    household: pd.DataFrame,
    donor: WASISADonor,
    seed: int,
    salt: str,
    minimum_age: int,
    maximum_adults: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split each household's two ISA totals over its adults, exactly.

    A household holding a type takes a donor household with the same number of
    adults (capped) and the same types held, chosen by its keyed uniform in
    proportion to donor weight; with none, the adult count steps down. The donor's
    shares go to the recipient's adults oldest first; any remainder lands on the
    adult with the largest share, so person values sum to the household value.
    """

    person_ids = person["person_id"].to_numpy()
    person_households = person["person_household_id"].to_numpy()
    ages = pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy(float)
    cash_total = pd.to_numeric(household[HOUSEHOLD_CASH], errors="coerce").fillna(0.0)
    stock_total = pd.to_numeric(
        household[HOUSEHOLD_STOCKS_AND_SHARES], errors="coerce"
    ).fillna(0.0)
    if (cash_total < 0).any() or (stock_total < 0).any():
        raise ISASubscriptionsError("household ISA totals must be non-negative.")
    household_ids = household["household_id"].to_numpy()
    uniforms = stable_identity_uniforms(household_ids, seed=seed, salt=salt)
    cash = np.zeros(len(person), dtype=float)
    stocks = np.zeros(len(person), dtype=float)
    positions = pd.Series(np.arange(len(person)), index=person.index)
    members = pd.Series(positions.to_numpy()).groupby(person_households).apply(list)
    pools: dict[tuple[int, bool, bool], np.ndarray] = {}
    for count in range(1, maximum_adults + 1):
        for holds_cash in (False, True):
            for holds_stocks in (False, True):
                mask = (
                    (donor.adult_count == count)
                    & (donor.holds_cash == holds_cash)
                    & (donor.holds_stocks_and_shares == holds_stocks)
                )
                pools[(count, holds_cash, holds_stocks)] = np.flatnonzero(mask)
    stepped_down = 0
    without_adult = 0
    without_donor = 0
    split = 0
    for index, household_id in enumerate(household_ids):
        cash_value = float(cash_total.iloc[index])
        stock_value = float(stock_total.iloc[index])
        if cash_value <= 0 and stock_value <= 0:
            continue
        rows = np.asarray(members.get(household_id, []), dtype=np.int64)
        if rows.size == 0:
            raise ISASubscriptionsError("a household holding ISAs has no persons.")
        adults = rows[ages[rows] >= minimum_age]
        if adults.size == 0:
            without_adult += 1
            adults = rows
        order = np.lexsort((person_ids[adults], -ages[adults]))
        adults = adults[order]
        key = (min(len(adults), maximum_adults), cash_value > 0, stock_value > 0)
        pool = pools.get(key, np.empty(0, dtype=np.int64))
        while pool.size == 0 and key[0] > 1:
            key = (key[0] - 1, key[1], key[2])
            pool = pools[key]
            stepped_down += 1
        if pool.size == 0:
            without_donor += 1
            cash_shares = np.zeros(len(adults))
            stock_shares = np.zeros(len(adults))
            cash_shares[0] = stock_shares[0] = 1.0
        else:
            cumulative = np.cumsum(donor.weight[pool])
            chosen = pool[
                min(
                    int(np.searchsorted(cumulative, uniforms[index] * cumulative[-1])),
                    pool.size - 1,
                )
            ]
            cash_shares = _fit_shares(donor.cash_shares[chosen], len(adults))
            stock_shares = _fit_shares(
                donor.stocks_and_shares_shares[chosen], len(adults)
            )
        cash[adults] = _exact_split(cash_value, cash_shares)
        stocks[adults] = _exact_split(stock_value, stock_shares)
        split += 1
    receipt = {
        "households_split": rounded_count(split),
        "households_without_adult_split_over_all_persons": rounded_count(without_adult),
        "households_matched_on_fewer_adults": rounded_count(stepped_down),
        "households_without_donor_given_to_eldest": rounded_count(without_donor),
        "maximum_donor_adults": maximum_adults,
    }
    return cash, stocks, receipt


def _fit_shares(shares: np.ndarray, adults: int) -> np.ndarray:
    fitted = np.zeros(adults, dtype=float)
    take = min(adults, len(shares))
    fitted[:take] = shares[:take]
    total = float(fitted.sum())
    if total <= 0:
        fitted[:] = 0.0
        fitted[0] = 1.0
        return fitted
    return fitted / total


def _exact_split(total: float, shares: np.ndarray) -> np.ndarray:
    values = total * shares
    if total > 0:
        largest = int(np.argmax(shares))
        values[largest] = total - (values.sum() - values[largest])
    return values


# --------------------------------------------------------------------------
# Recipients
# --------------------------------------------------------------------------


def _income_band_ids(total_income: np.ndarray) -> np.ndarray:
    edges = [isa_band_bounds(band)[1] for band in HMRC_ISA_INCOME_BAND_IDS[:-1]]
    index = np.searchsorted(np.asarray(edges, dtype=float), total_income, side="left")
    return np.asarray(HMRC_ISA_INCOME_BAND_IDS, dtype=object)[index]


def _age_band_ids(ages: np.ndarray) -> np.ndarray:
    uppers = []
    for band in HMRC_ISA_AGE_BAND_IDS[:-1]:
        uppers.append(23 if band == "age_under_24" else isa_band_bounds(band)[1])
    index = np.searchsorted(
        np.asarray(uppers, dtype=float), np.floor(ages), side="left"
    )
    return np.asarray(HMRC_ISA_AGE_BAND_IDS, dtype=object)[index]


def _enum_name(value: object) -> str:
    name = getattr(value, "name", value)
    return str(name)


def recipient_table(frame: Frame) -> pd.DataFrame:
    """One row per person: identity, weight, the cell attributes and LISA holding."""

    person = frame.table("person")
    household = frame.table("household")
    for column in ("person_id", "person_household_id", "age", "gender"):
        if column not in person:
            raise ISASubscriptionsError(f"the person table lacks {column!r}.")
    for column in (
        "household_id",
        "region",
        HOUSEHOLD_CASH,
        HOUSEHOLD_STOCKS_AND_SHARES,
    ):
        if column not in household:
            raise ISASubscriptionsError(f"the household table lacks {column!r}.")
    for column in (LIFETIME_ISA_HOLDER, LIFETIME_ISA_BALANCE):
        if column not in person:
            raise ISASubscriptionsError(
                f"the person table lacks {column!r} (was_lisa runs first)."
            )
    weights = pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=household["household_id"].to_numpy(),
    )
    regions = household.set_index("household_id")["region"].map(_enum_name)
    unknown = sorted(set(regions) - set(_REGION_ROW))
    if unknown:
        raise ISASubscriptionsError(f"unrecognised household regions {unknown}.")
    household_ids = person["person_household_id"].to_numpy()
    ages = pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy(float)
    total_income = uk_cgt_component_sum_income(person)
    sex = person["gender"].map(_enum_name).str.upper()
    if not sex.isin(("MALE", "FEMALE")).all():
        raise ISASubscriptionsError("person gender must be MALE or FEMALE.")
    return pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            "household_id": household_ids,
            "weight": weights.reindex(household_ids).to_numpy(float),
            "age": ages,
            "sex": sex.str.lower().to_numpy(),
            "total_income": total_income,
            "income_band": _income_band_ids(total_income),
            "age_band": _age_band_ids(ages),
            "region": regions.reindex(household_ids).map(_REGION_ROW).to_numpy(),
            "holds_lifetime_isa": person[LIFETIME_ISA_HOLDER].astype(bool).to_numpy(),
            "lifetime_isa_balance": pd.to_numeric(
                person[LIFETIME_ISA_BALANCE], errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(float),
        },
        index=person.index,
    )


# --------------------------------------------------------------------------
# Step 2: who subscribes
# --------------------------------------------------------------------------


def _margins(
    statistics: HMRCISAStatistics, isa_type: str, scale: float = 1.0
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, float]]:
    """Table 9.7's income, 9.8's age-by-sex and 9.9's region margins, on 9.7's total.

    Table 9.9's overseas-and-unknown row has no spine cell and is dropped; 9.8 and
    9.9 are scaled to Table 9.7's total, so the three margins agree.
    """

    income = {
        band: statistics.subscribers(isa_type, band, "all") * scale
        for band in HMRC_ISA_INCOME_BAND_IDS
    }
    total = sum(income.values())
    age_sex = {
        (age, sex): statistics.subscribers_by_age_sex[(isa_type, sex, age)]
        for age in HMRC_ISA_AGE_BAND_IDS
        for sex in ("male", "female")
    }
    age_total = sum(age_sex.values())
    region = {
        row: statistics.subscribers_by_region[(isa_type, row)]
        for row in HMRC_ISA_REGION_IDS
        if row != _OVERSEAS
    }
    region_total = sum(region.values())
    return (
        income,
        {key: value * total / age_total for key, value in age_sex.items()},
        {key: value * total / region_total for key, value in region.items()},
    )


def _rake(
    seed_mass: np.ndarray,
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    margins: tuple[Mapping[Any, float], Mapping[Any, float], Mapping[Any, float]],
    *,
    iterations: int = 500,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Iterative proportional fitting of cell masses to three margins."""

    mass = seed_mass.astype(float).copy()
    for _ in range(iterations):
        worst = 0.0
        for label, margin in zip(labels, margins, strict=True):
            current = pd.Series(mass).groupby(label).sum()
            for key, target in margin.items():
                achieved = float(current.get(key, 0.0))
                if achieved <= 0:
                    continue
                factor = target / achieved
                worst = max(worst, abs(factor - 1.0))
                mass[label == key] *= factor
        if worst < tolerance:
            break
    return mass


def select_subscribers(
    recipients: pd.DataFrame,
    eligible: np.ndarray,
    margins: tuple[Mapping[Any, float], Mapping[Any, float], Mapping[Any, float]],
    *,
    seed: int,
    salt: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rake eligible mass to the margins and walk each cell to its share."""

    selected = np.zeros(len(recipients), dtype=bool)
    if not eligible.any():
        return selected, {
            "selected_weight": 0.0,
            "shortfall_weight": sum(margins[0].values()),
        }
    frame = recipients.loc[eligible]
    weights = frame["weight"].to_numpy(float)
    cells = frame.groupby(["income_band", "age_band", "sex", "region"], sort=True)[
        "weight"
    ].sum()
    index = cells.index
    seed_mass = cells.to_numpy(float)
    income = np.asarray(index.get_level_values(0), dtype=object)
    age_sex = np.asarray(
        list(zip(index.get_level_values(1), index.get_level_values(2), strict=True)),
        dtype=object,
    )
    age_sex_keys = np.asarray([f"{age}|{sex}" for age, sex in age_sex], dtype=object)
    region = np.asarray(index.get_level_values(3), dtype=object)
    age_sex_margin = {f"{age}|{sex}": value for (age, sex), value in margins[1].items()}
    raked = _rake(
        seed_mass,
        (income, age_sex_keys, region),
        (margins[0], age_sex_margin, margins[2]),
    )
    target = np.minimum(raked, seed_mass)
    shortfall = raked - target
    # Move each income band's excess onto its cells with room, in proportion to room.
    moved = 0.0
    for band in np.unique(income):
        in_band = income == band
        excess = float(shortfall[in_band].sum())
        room = seed_mass[in_band] - target[in_band]
        if excess <= 0 or float(room.sum()) <= 0:
            continue
        take = min(excess, float(room.sum()))
        target[in_band] += room / float(room.sum()) * take
        moved += take
    unmet = float(shortfall.sum()) - moved
    uniforms = stable_identity_uniforms(
        frame["person_id"].to_numpy(), seed=seed, salt=salt
    )
    positions = np.flatnonzero(eligible)
    cell_of = pd.MultiIndex.from_frame(
        frame[["income_band", "age_band", "sex", "region"]]
    )
    codes = index.get_indexer(cell_of)
    person_ids = frame["person_id"].to_numpy()
    # Each cell's rows in ascending position, as np.flatnonzero(codes == cell) would
    # give them, from one stable sort instead of a scan per cell.
    by_cell = np.argsort(codes, kind="stable")
    cell_ids = np.arange(len(index))
    starts = np.searchsorted(codes[by_cell], cell_ids, side="left")
    ends = np.searchsorted(codes[by_cell], cell_ids, side="right")
    for cell in range(len(index)):
        if target[cell] <= 0:
            continue
        rows = by_cell[starts[cell] : ends[cell]]
        ranked = rows[np.lexsort((person_ids[rows], uniforms[rows]))]
        members: dict[int, list[np.ndarray]] = {}
        assigned, _ = _walk_bands(
            ranked=ranked,
            weights=weights[ranked],
            boundaries=[(0, float(target[cell]))],
            rng=_keyed_offset(seed, f"{salt}:cell", "|".join(map(str, index[cell]))),
            members=members,
        )
        selected[positions[ranked[assigned]]] = True
    achieved_income = recipients.loc[selected].groupby("income_band")["weight"].sum()
    receipt = {
        "target_weight": float(sum(margins[0].values())),
        "selected_weight": float(recipients.loc[selected, "weight"].sum()),
        "eligible_weight": float(weights.sum()),
        "cells": int(len(index)),
        "cells_short_of_eligible_adults": int((shortfall > 1e-9).sum()),
        "shortfall_weight_moved_within_income_band": moved,
        "shortfall_weight_unmet": max(unmet, 0.0),
        "income_margin_max_abs_relative_error": _max_relative_error(
            margins[0], achieved_income
        ),
        "selected_records": rounded_count(int(selected.sum())),
    }
    return selected, receipt


def _max_relative_error(
    targets: Mapping[str, float], achieved: pd.Series
) -> float | None:
    errors = [
        abs(float(achieved.get(key, 0.0)) - value) / value
        for key, value in targets.items()
        if value > 0
    ]
    return max(errors) if errors else None


# --------------------------------------------------------------------------
# Step 3: how much
# --------------------------------------------------------------------------


def _subscription_bands() -> list[tuple[float, float]]:
    """(exclusive lower, upper) of each band; the top band is the point 20,000."""

    bands = []
    previous = 0.0
    for band in HMRC_ISA_SUBSCRIPTION_BAND_IDS:
        low, high = isa_band_bounds(band)
        if low == high:
            bands.append((float(low), float(high)))
        else:
            upper = float(high) + 1.0 if high == 19999 else float(high)
            bands.append((previous, upper))
        previous = float(high)
    return bands


SUBSCRIPTION_BANDS = _subscription_bands()


def _weighted_rank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Each value's weighted mid-rank in [0, 1] (ties share a rank)."""

    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    ranks = np.empty(len(values), dtype=float)
    if total <= 0:
        ranks[:] = 0.5
        return ranks
    ranks[order] = (cumulative - weights[order] / 2.0) / total
    return ranks


def draw_band_amounts(
    rows: np.ndarray,
    *,
    weights: np.ndarray,
    balance: np.ndarray,
    person_ids: np.ndarray,
    band_shares: np.ndarray,
    balance_rank_weight: float,
    seed: int,
    salt: str,
    key: str,
) -> np.ndarray:
    """Amounts for one (type, income band): Table 9.7's band masses, evenly within bands."""

    amounts = np.zeros(len(rows), dtype=float)
    if len(rows) == 0:
        return amounts
    row_weights = weights[rows]
    total = float(row_weights.sum())
    shares = np.asarray(band_shares, dtype=float)
    if total <= 0 or shares.sum() <= 0:
        return amounts
    masses = shares / shares.sum() * total
    uniforms = stable_identity_uniforms(
        person_ids[rows], seed=seed, salt=f"{salt}:rank"
    )
    rank_key = (
        balance_rank_weight * _weighted_rank(balance[rows], row_weights)
        + (1.0 - balance_rank_weight) * uniforms
    )
    ranked = np.lexsort((person_ids[rows], -rank_key))
    members: dict[int, list[np.ndarray]] = {}
    boundaries = [
        (band, float(masses[band])) for band in range(len(masses) - 1, -1, -1)
    ]
    assigned, _ = _walk_bands(
        ranked=ranked,
        weights=row_weights[ranked],
        boundaries=boundaries,
        rng=_keyed_offset(seed, f"{salt}:walk", key),
        members=members,
    )
    leftover = ranked[~assigned]
    if leftover.size:
        members.setdefault(0, []).append(leftover)
    for band, groups in members.items():
        positions = np.concatenate(groups)
        lower, upper = SUBSCRIPTION_BANDS[band]
        if lower == upper:
            amounts[positions] = upper
            continue
        ordered = positions[np.argsort(rank_key[positions], kind="stable")]
        cumulative = np.cumsum(row_weights[ordered])
        tops = cumulative / cumulative[-1]
        bottoms = np.concatenate(([0.0], tops[:-1]))
        jitter = stable_identity_uniforms(
            [f"{key}:{band}"], seed=seed, salt=f"{salt}:jitter"
        )[0]
        quantiles = bottoms + (0.05 + 0.9 * jitter) * (tops - bottoms)
        amounts[ordered] = lower + quantiles * (upper - lower)
    return amounts


# --------------------------------------------------------------------------
# Step 4: the split for people paying into both
# --------------------------------------------------------------------------


def _share_logit(balance_share: np.ndarray, share_floor: float) -> np.ndarray:
    base = np.clip(balance_share, share_floor, 1.0 - share_floor)
    return np.log(base / (1.0 - base))


def _shifted_share(logit: np.ndarray, shift: float, share_floor: float) -> np.ndarray:
    """The shifted cash share, kept inside [floor, 1 - floor] so both parts stay positive."""

    return np.clip(
        1.0 / (1.0 + np.exp(-(logit + shift))), share_floor, 1.0 - share_floor
    )


def split_both_type_amounts(
    combined: np.ndarray,
    cash_share_of_balance: np.ndarray,
    weights: np.ndarray,
    *,
    stocks_and_shares_target: float,
    share_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Each person's cash share: their balance share plus one common logit shift."""

    logit = _share_logit(cash_share_of_balance, share_floor)
    weighted = combined * weights

    def stocks_total(shift: float) -> float:
        share = _shifted_share(logit, shift, share_floor)
        return float((weighted * (1.0 - share)).sum())

    low, high = -40.0, 40.0
    lowest, highest = stocks_total(high), stocks_total(low)
    saturated = None
    if stocks_and_shares_target <= lowest:
        shift, saturated = high, "stocks_and_shares_residual_below_reach"
    elif stocks_and_shares_target >= highest:
        shift, saturated = low, "stocks_and_shares_residual_above_reach"
    else:
        for _ in range(200):
            mid = (low + high) / 2.0
            if stocks_total(mid) > stocks_and_shares_target:
                low = mid
            else:
                high = mid
        shift = (low + high) / 2.0
    share = _shifted_share(logit, shift, share_floor)
    total = float(weighted.sum())
    receipt = {
        "logit_shift": shift,
        "saturated": saturated,
        "stocks_and_shares_target_gbp": stocks_and_shares_target,
        "stocks_and_shares_achieved_gbp": stocks_total(shift),
        "stocks_and_shares_share_of_group": (
            stocks_total(shift) / total if total > 0 else None
        ),
    }
    return share, receipt


# --------------------------------------------------------------------------
# Step 5: growth and step 6: Lifetime ISA payments
# --------------------------------------------------------------------------


def _capped(
    cash: np.ndarray, stocks: np.ndarray, limit: float
) -> tuple[np.ndarray, np.ndarray]:
    combined = cash + stocks
    over = combined > limit
    if over.any():
        factor = np.ones(len(cash))
        factor[over] = limit / combined[over]
        return cash * factor, stocks * factor
    return cash, stocks


def scale_to_build_year_amounts(
    cash: np.ndarray,
    stocks: np.ndarray,
    weights: np.ndarray,
    *,
    cash_target: float,
    stocks_target: float,
    limit: float,
    taper_from: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One factor per type, solved to both totals, holding and tapering to the limit.

    Subscribers already paying the overall limit keep their amounts, so a factor
    below one moves nobody off it (María's ruling of 29 October 2026). Everyone
    else scales by the type's factor, except that a person whose combined amount
    lies between ``taper_from`` and the limit moves on a line from their scaled
    ``taper_from`` to the limit (María's ruling of 29 October 2026). One factor
    up to the limit would empty the range just below it whenever it fell under
    one (review of microcosm#1421). The scaled amounts are capped at the limit,
    pro rata across the two types.
    """

    if not 0 < taper_from < limit:
        raise ISASubscriptionsError("the taper starts inside (0, limit).")
    combined = cash + stocks
    at_limit = combined >= limit - 1e-6
    tapered = (combined > taper_from) & ~at_limit

    def scaled(cash_factor: float, stocks_factor: float) -> tuple[np.ndarray, ...]:
        moved_cash, moved_stocks = cash * cash_factor, stocks * stocks_factor
        if tapered.any():
            moved = moved_cash + moved_stocks
            factor = moved[tapered] / combined[tapered]
            start = factor * taper_from
            target = start + (combined[tapered] - taper_from) * (limit - start) / (
                limit - taper_from
            )
            ratio = np.ones(len(cash))
            ratio[tapered] = np.divide(
                target,
                moved[tapered],
                out=np.ones(int(tapered.sum())),
                where=moved[tapered] > 0,
            )
            moved_cash, moved_stocks = moved_cash * ratio, moved_stocks * ratio
        moved_cash, moved_stocks = _capped(moved_cash, moved_stocks, limit)
        return (
            np.where(at_limit, cash, moved_cash),
            np.where(at_limit, stocks, moved_stocks),
        )

    def totals(cash_factor: float, stocks_factor: float) -> tuple[float, float]:
        new_cash, new_stocks = scaled(cash_factor, stocks_factor)
        return float((new_cash * weights).sum()), float((new_stocks * weights).sum())

    def solve(target: float, other: float, which: int) -> float:
        low, high = 0.0, 50.0
        for _ in range(100):
            mid = (low + high) / 2.0
            achieved = totals(mid, other)[0] if which == 0 else totals(other, mid)[1]
            if achieved < target:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    cash_factor, stocks_factor = 1.0, 1.0
    for _ in range(50):
        new_cash_factor = solve(cash_target, stocks_factor, 0)
        new_stocks_factor = solve(stocks_target, new_cash_factor, 1)
        converged = (
            abs(new_cash_factor - cash_factor) < 1e-10
            and abs(new_stocks_factor - stocks_factor) < 1e-10
        )
        cash_factor, stocks_factor = new_cash_factor, new_stocks_factor
        if converged:
            break
    scaled_cash, scaled_stocks = scaled(cash_factor, stocks_factor)
    achieved = totals(cash_factor, stocks_factor)
    receipt = {
        "cash_factor": cash_factor,
        "stocks_and_shares_factor": stocks_factor,
        "cash_target_gbp": cash_target,
        "cash_achieved_gbp": achieved[0],
        "stocks_and_shares_target_gbp": stocks_target,
        "stocks_and_shares_achieved_gbp": achieved[1],
        "held_at_overall_limit_records": rounded_count(int(at_limit.sum())),
        "taper_from_gbp": taper_from,
        "tapered_records": rounded_count(int(tapered.sum())),
        "at_overall_limit_records": rounded_count(
            int(((scaled_cash + scaled_stocks) >= limit - 1e-6).sum())
        ),
    }
    return scaled_cash, scaled_stocks, receipt


def draw_lifetime_isa_payments(
    recipients: pd.DataFrame,
    *,
    headroom: np.ndarray,
    target_gbp: float,
    spec: ISASubscriptionsSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Lifetime ISA payments on (0, limit], tilted to Table 9.4's amount, in headroom."""

    payments = np.zeros(len(recipients), dtype=float)
    ages = recipients["age"].to_numpy(float)
    payers = (
        recipients["holds_lifetime_isa"].to_numpy(bool)
        & (ages >= spec.lifetime_minimum_age)
        & (ages <= spec.lifetime_maximum_age)
    )
    rows = np.flatnonzero(payers)
    weights = recipients["weight"].to_numpy(float)
    receipt: dict[str, Any] = {
        # Holders at the paying ages; those without room under the overall
        # limit pay nothing, so the paying weight below is the payer count.
        "eligible_holder_weight": float(weights[rows].sum()),
        "eligible_holder_records": rounded_count(int(rows.size)),
        "target_gbp": target_gbp,
    }
    if rows.size == 0:
        receipt.update({"achieved_gbp": 0.0, "paying_weight": 0.0, "paying_records": 0})
        return payments, receipt
    payer_weights = weights[rows]
    person_ids = recipients["person_id"].to_numpy()[rows]
    uniforms = stable_identity_uniforms(
        person_ids, seed=spec.lifetime_seed, salt=f"{spec.lifetime_salt}:rank"
    )
    key = (
        spec.balance_rank_weight
        * _weighted_rank(
            recipients["lifetime_isa_balance"].to_numpy(float)[rows], payer_weights
        )
        + (1.0 - spec.balance_rank_weight) * uniforms
    )
    order = np.lexsort((person_ids, key))
    cumulative = np.cumsum(payer_weights[order])
    tops = cumulative / cumulative[-1]
    bottoms = np.concatenate(([0.0], tops[:-1]))
    quantiles = np.empty(rows.size)
    quantiles[order] = (bottoms + tops) / 2.0
    room = np.clip(headroom[rows], 0.0, spec.lifetime_limit_gbp)
    limit = spec.lifetime_limit_gbp

    def drawn(mean: float) -> np.ndarray:
        return np.minimum(
            _truncated_exponential_quantile(quantiles, 0.0, limit, mean), room
        )

    low, high = limit * 1e-6, limit * (1 - 1e-6)
    if float((drawn(high) * payer_weights).sum()) <= target_gbp:
        mean, receipt["saturated"] = high, "target_above_reach"
    else:
        for _ in range(100):
            mid = (low + high) / 2.0
            if float((drawn(mid) * payer_weights).sum()) < target_gbp:
                low = mid
            else:
                high = mid
        mean = (low + high) / 2.0
        receipt["saturated"] = None
    payments[rows] = drawn(mean)
    paying = payments[rows] > 0
    receipt.update(
        {
            "uncapped_mean_gbp": mean,
            "achieved_gbp": float((payments[rows] * payer_weights).sum()),
            "paying_weight": float(payer_weights[paying].sum()),
            "paying_records": rounded_count(int(paying.sum())),
            "trimmed_to_overall_limit_records": rounded_count(
                int(
                    (
                        room
                        < _truncated_exponential_quantile(quantiles, 0.0, limit, mean)
                    ).sum()
                )
            ),
        }
    )
    return payments, receipt


# --------------------------------------------------------------------------
# The whole imputation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ISASubscriptionsResult:
    cash_balance: np.ndarray
    stocks_and_shares_balance: np.ndarray
    cash_subscriptions: np.ndarray
    stocks_and_shares_subscriptions: np.ndarray
    lifetime_isa_payments: np.ndarray
    receipt: Mapping[str, Any]


def _class_balance(isa_type: str, cash: np.ndarray, stocks: np.ndarray) -> np.ndarray:
    if isa_type == CASH_ONLY:
        return cash
    if isa_type == STOCKS_AND_SHARES_ONLY:
        return stocks
    return cash + stocks


def _eligible(isa_type: str, cash: np.ndarray, stocks: np.ndarray) -> np.ndarray:
    if isa_type == CASH_ONLY:
        return cash > 0
    if isa_type == STOCKS_AND_SHARES_ONLY:
        return stocks > 0
    return (cash > 0) & (stocks > 0)


def impute_isa_subscriptions(
    recipients: pd.DataFrame,
    *,
    cash_balance: np.ndarray,
    stocks_balance: np.ndarray,
    statistics: HMRCISAStatistics,
    spec: ISASubscriptionsSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Steps 2 to 6 on recipients that already carry per-person balances."""

    weights = recipients["weight"].to_numpy(float)
    person_ids = recipients["person_id"].to_numpy()
    adult = recipients["age"].to_numpy(float) >= spec.minimum_age
    taken = np.zeros(len(recipients), dtype=bool)
    class_of = np.full(len(recipients), "", dtype=object)
    extra = np.zeros(len(recipients), dtype=bool)
    base_year, build_year = spec.base_tax_year, spec.build_tax_year
    growth = {
        CASH_ONLY: statistics.accounts_subscribed[("cash", build_year)]
        / statistics.accounts_subscribed[("cash", base_year)],
        STOCKS_AND_SHARES_ONLY: statistics.accounts_subscribed[
            ("stocks_and_shares", build_year)
        ]
        / statistics.accounts_subscribed[("stocks_and_shares", base_year)],
    }
    growth[BOTH] = (
        statistics.accounts_subscribed[("cash", build_year)]
        + statistics.accounts_subscribed[("stocks_and_shares", build_year)]
    ) / (
        statistics.accounts_subscribed[("cash", base_year)]
        + statistics.accounts_subscribed[("stocks_and_shares", base_year)]
    )
    selection: dict[str, Any] = {}
    for isa_type in CLASS_ORDER:
        eligible = adult & ~taken & _eligible(isa_type, cash_balance, stocks_balance)
        chosen, receipt = select_subscribers(
            recipients,
            eligible,
            _margins(statistics, isa_type),
            seed=spec.select_seed,
            salt=f"{spec.select_salt}:{isa_type}",
        )
        taken |= chosen
        class_of[chosen] = isa_type
        # Step 5's people: the account growth, drawn from those still eligible.
        more = adult & ~taken & _eligible(isa_type, cash_balance, stocks_balance)
        added, growth_receipt = select_subscribers(
            recipients,
            more,
            _margins(statistics, isa_type, scale=growth[isa_type] - 1.0),
            seed=spec.select_seed,
            salt=f"{spec.select_salt}:{isa_type}:growth",
        )
        taken |= added
        class_of[added] = isa_type
        extra |= added
        selection[isa_type] = {
            "base": receipt,
            "growth": {**growth_receipt, "account_growth": growth[isa_type]},
        }

    combined = np.zeros(len(recipients), dtype=float)
    for isa_type in CLASS_ORDER:
        balance = _class_balance(isa_type, cash_balance, stocks_balance)
        for income in HMRC_ISA_INCOME_BAND_IDS:
            shares = np.asarray(
                [
                    statistics.subscribers(isa_type, income, band)
                    for band in HMRC_ISA_SUBSCRIPTION_BAND_IDS
                ]
            )
            for cohort, mask in (("base", ~extra), ("growth", extra)):
                rows = np.flatnonzero(
                    (class_of == isa_type)
                    & mask
                    & (recipients["income_band"].to_numpy() == income)
                )
                combined[rows] = draw_band_amounts(
                    rows,
                    weights=weights,
                    balance=balance,
                    person_ids=person_ids,
                    band_shares=shares,
                    balance_rank_weight=spec.balance_rank_weight,
                    seed=spec.amount_seed,
                    salt=f"{spec.amount_salt}:{isa_type}",
                    key=f"{isa_type}|{income}|{cohort}",
                )

    cash = np.where(class_of == CASH_ONLY, combined, 0.0)
    stocks = np.where(class_of == STOCKS_AND_SHARES_ONLY, combined, 0.0)
    both = class_of == BOTH
    base = ~extra
    stocks_only_base = float((stocks[base] * weights[base]).sum())
    residual = (
        statistics.amount_subscribed[("stocks_and_shares", base_year)]
        - stocks_only_base
    )
    holding = cash_balance + stocks_balance
    balance_share = np.divide(
        cash_balance, holding, out=np.full(len(holding), 0.5), where=holding > 0
    )
    share = np.zeros(len(recipients))
    share[both & base], split_receipt = split_both_type_amounts(
        combined[both & base],
        balance_share[both & base],
        weights[both & base],
        stocks_and_shares_target=residual,
        share_floor=spec.share_floor,
    )
    shift = split_receipt["logit_shift"]
    if (both & extra).any():
        share[both & extra] = _shifted_share(
            _share_logit(balance_share[both & extra], spec.share_floor),
            shift,
            spec.share_floor,
        )
    cash[both] = combined[both] * share[both]
    stocks[both] = combined[both] * (1.0 - share[both])

    base_cash = float((cash[base] * weights[base]).sum())
    base_stocks = float((stocks[base] * weights[base]).sum())
    # Table 9.4's build-year amounts are the 2024-25 targets, not its growth on
    # the stage's own 2023-24 totals. Table 9.7's cash bands sit about a third
    # above Table 9.4, and growing that excess by Table 9.4's ratio, with the
    # limit holding the top, scaled everyone below the limit by about 2.05 on
    # populations built from HMRC's own tables (María's ruling of 29 October
    # 2026, review of microcosm#1421). Stocks and shares meet Table 9.4 in
    # 2023-24 through the both class's residual, so the two rules agree for them.
    cash, stocks, scale_receipt = scale_to_build_year_amounts(
        cash,
        stocks,
        weights,
        cash_target=statistics.amount_subscribed[("cash", build_year)],
        stocks_target=statistics.amount_subscribed[("stocks_and_shares", build_year)],
        limit=spec.overall_limit_gbp,
        taper_from=spec.taper_from_gbp,
    )
    headroom = spec.overall_limit_gbp - cash - stocks
    lifetime, lifetime_receipt = draw_lifetime_isa_payments(
        recipients,
        headroom=headroom,
        target_gbp=statistics.amount_subscribed[("lifetime", build_year)],
        spec=spec,
    )
    lifetime_receipt["hmrc_accounts_subscribed"] = statistics.accounts_subscribed[
        ("lifetime", build_year)
    ]
    receipt = {
        "selection": selection,
        "both_type_split": split_receipt,
        "base_year": {
            "tax_year": base_year,
            "cash_gbp": base_cash,
            "stocks_and_shares_gbp": base_stocks,
            "table_9_4_cash_gbp": statistics.amount_subscribed[("cash", base_year)],
            "table_9_4_stocks_and_shares_gbp": statistics.amount_subscribed[
                ("stocks_and_shares", base_year)
            ],
            "cash_excess_over_table_9_4_gbp": base_cash
            - statistics.amount_subscribed[("cash", base_year)],
        },
        "growth": {
            "to_tax_year": build_year,
            "people": "table_9_4_account_growth (upper bound on people)",
            "account_growth": growth,
            "amounts": "table_9_4_amount_level_tapered_to_limit",
            **scale_receipt,
        },
        "lifetime_isa": lifetime_receipt,
        "checks": _checks(
            recipients,
            cash,
            stocks,
            lifetime,
            class_of,
            cash_balance,
            stocks_balance,
            spec,
        ),
    }
    return cash, stocks, lifetime, receipt


def _checks(
    recipients: pd.DataFrame,
    cash: np.ndarray,
    stocks: np.ndarray,
    lifetime: np.ndarray,
    class_of: np.ndarray,
    cash_balance: np.ndarray,
    stocks_balance: np.ndarray,
    spec: ISASubscriptionsSpec,
) -> dict[str, Any]:
    weights = recipients["weight"].to_numpy(float)
    ages = recipients["age"].to_numpy(float)
    under_65_cash = (cash > 0) & (ages < 65)
    cash_only_under_65 = (class_of == CASH_ONLY) & (ages < 65)
    # NaN compares false, so every comparison below would pass it: count it first.
    finite = np.ones(len(cash), dtype=bool)
    for values in (cash, stocks, lifetime, cash_balance, stocks_balance):
        finite &= np.isfinite(values)
    violations = {
        "non_finite_values": int((~finite).sum()),
        "over_overall_limit": int(
            (cash + stocks + lifetime > spec.overall_limit_gbp + 1e-6).sum()
        ),
        "lifetime_over_annual_limit": int(
            (lifetime > spec.lifetime_limit_gbp + 1e-6).sum()
        ),
        "subscriber_under_minimum_age": int(
            (((cash + stocks) > 0) & (ages < spec.minimum_age)).sum()
        ),
        "cash_without_cash_holding": int(((cash > 0) & (cash_balance <= 0)).sum()),
        "stocks_without_stocks_holding": int(
            ((stocks > 0) & (stocks_balance <= 0)).sum()
        ),
        "both_class_missing_a_part": int(
            ((class_of == BOTH) & ((cash <= 0) | (stocks <= 0))).sum()
        ),
        "negative_amounts": int(((cash < 0) | (stocks < 0) | (lifetime < 0)).sum()),
    }
    return {
        "violations": violations,
        "impact_note_under_65_cash_share_over_12000": {
            "cash_part": _share(cash[under_65_cash] > 12000, weights[under_65_cash]),
            "cash_only_subscribers": _share(
                cash[cash_only_under_65] > 12000, weights[cash_only_under_65]
            ),
            "published_2022_23_percent": 22,
        },
        "subscribers_above_their_balance_records": rounded_count(
            int(
                (
                    ((cash > cash_balance) & (cash > 0))
                    | ((stocks > stocks_balance) & (stocks > 0))
                ).sum()
            )
        ),
    }


def _share(mask: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    return float(weights[mask].sum() / total) if total > 0 else None


# --------------------------------------------------------------------------
# Stage transform
# --------------------------------------------------------------------------


@dataclass
class UKISASubscriptionsStageTransform:
    """Whole-stage callable for the ISA subscriptions imputation."""

    stage: SourceStageSpec
    engine: object
    was_tab_path: str | Path | None = None
    was_person_tab_path: str | Path | None = None
    donor_household: pd.DataFrame | None = None
    donor_person: pd.DataFrame | None = None
    #: Supplied facts (tests and the synthetic fixture); else the vendored resource.
    statistics: HMRCISAStatistics | None = None
    last_receipt: Mapping[str, Any] | None = field(default=None, init=False)

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        return ()

    def _table(
        self,
        supplied: pd.DataFrame | None,
        path: str | Path | None,
        role: str,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if supplied is not None:
            return supplied
        if path is None:
            raise ISASubscriptionsError(
                f"{UK_ISA_SUBSCRIPTIONS_STAGE_NAME} requires the caller-supplied {role} tab."
            )
        return read_pinned_tab(
            Path(path).expanduser().resolve(),
            _artifact(self.stage, role),
            columns=columns,
        )

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        spec = ISASubscriptionsSpec.from_stage(self.stage)
        check_isa_statistics_declaration(
            _operation(self.stage, VERIFY_VENDORED_FACT_RESOURCE_KIND),
            stage=UK_ISA_SUBSCRIPTIONS_STAGE_NAME,
            period=int(uk_time_period(frame)),
        )
        statistics = self.statistics or load_hmrc_isa_statistics()
        donor = clean_was_isa_person_donor(
            self._table(
                self.donor_person,
                self.was_person_tab_path,
                UK_WAS_PERSON_TAB_ROLE,
                columns=spec.columns.person_raw_columns(),
            ),
            self._table(
                self.donor_household, self.was_tab_path, UK_WAS_HOUSEHOLD_TAB_ROLE
            ),
            columns=spec.columns,
            maximum_adults=spec.maximum_donor_adults,
        )
        person = frame.table("person").copy()
        household = frame.table("household")
        cash_balance, stocks_balance, split_receipt = split_household_isa_balances(
            person=person,
            household=household,
            donor=donor,
            seed=spec.split_seed,
            salt=spec.split_salt,
            minimum_age=spec.minimum_age,
            maximum_adults=spec.maximum_donor_adults,
        )
        recipients = recipient_table(frame)
        cash, stocks, lifetime, receipt = impute_isa_subscriptions(
            recipients,
            cash_balance=cash_balance,
            stocks_balance=stocks_balance,
            statistics=statistics,
            spec=spec,
        )
        person[CASH_BALANCE] = cash_balance
        person[STOCKS_AND_SHARES_BALANCE] = stocks_balance
        person[CASH_SUBSCRIPTIONS] = cash
        person[STOCKS_AND_SHARES_SUBSCRIPTIONS] = stocks
        person[LIFETIME_ISA_PAYMENTS] = lifetime
        household_weights = np.asarray(
            frame.weights_for("household").values, dtype=float
        )
        sums = (
            pd.DataFrame(
                {
                    "cash": cash_balance,
                    "stocks": stocks_balance,
                    "household": person["person_household_id"].to_numpy(),
                }
            )
            .groupby("household")[["cash", "stocks"]]
            .sum()
        )
        published = household.set_index("household_id")[
            [HOUSEHOLD_CASH, HOUSEHOLD_STOCKS_AND_SHARES]
        ]
        published = published.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        aligned = sums.reindex(published.index).fillna(0.0)
        exact = float(
            max(
                (aligned["cash"] - published[HOUSEHOLD_CASH]).abs().max(),
                (aligned["stocks"] - published[HOUSEHOLD_STOCKS_AND_SHARES])
                .abs()
                .max(),
            )
        )
        result = uk_national_frame(
            person=person,
            benunit=frame.table("benunit").copy(),
            household=household.copy(),
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=household_weights,
            mass_log=(
                *frame.mass_log,
                uk_household_mass_conservation_receipt(
                    frame, UK_ISA_SUBSCRIPTIONS_MASS_CONSERVATION_REASON
                ),
            ),
        )
        validate_uk_national_frame(result)
        self.last_receipt = {
            "stage": UK_ISA_SUBSCRIPTIONS_STAGE_NAME,
            "facts": statistics.evidence(),
            "donor": donor.receipt,
            "balance_split": {
                **split_receipt,
                "max_abs_household_sum_error_gbp": exact,
            },
            **receipt,
        }
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_ISA_SUBSCRIPTIONS_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_receipt is None:
            return {}
        return {"evidence": dict(self.last_receipt)}
