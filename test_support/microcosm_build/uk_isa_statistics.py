"""Synthetic ISA facts, WAS person tabs and the stage declaration (microcosm#1415).

Values are invented and internally consistent (the 'all' blocks are the sums of the
type blocks, Tables 9.8 and 9.9 restate Table 9.7's totals); they exercise the code,
not HMRC's numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.uk_runtime.hmrc_isa_statistics import (
    HMRC_ISA_AGE_BAND_IDS,
    HMRC_ISA_INCOME_BAND_IDS,
    HMRC_ISA_REGION_IDS,
    HMRC_ISA_STATISTICS_RECORD_SETS,
    HMRC_ISA_SUBSCRIPTION_BAND_IDS,
    HMRC_ISA_TYPES,
)
from microcosm.build.uk_runtime.isa_subscriptions import (
    UK_ISA_FACT_FENCE,
    UK_ISA_SUBSCRIPTIONS_MASS_CONSERVATION_REASON,
)

PREFIX = "hmrc.isa_statistics_2026."
#: Invented 2023-24 subscriber totals by type, people.
TYPE_TOTALS = {
    "cash_only": 9_000.0,
    "stocks_and_shares_only": 3_000.0,
    "cash_and_stocks_and_shares": 1_000.0,
}
_INCOME_SHARES = np.linspace(1.0, 0.2, len(HMRC_ISA_INCOME_BAND_IDS))
_BAND_SHARES = np.asarray(
    [0.3, 0.12, 0.07, 0.04, 0.05, 0.03, 0.02, 0.03, 0.02, 0.04, 0.28]
)
_AGE_SHARES = np.linspace(0.6, 1.0, len(HMRC_ISA_AGE_BAND_IDS))
_REGION_SHARES = np.asarray(
    [0.5, 1.4, 1.0, 0.9, 1.1, 1.3, 1.8, 2.0, 1.2, 0.6, 0.9, 0.3, 0.02]
)

#: The vendored-facts artifact and its check, which both stages declare.
ISA_FACTS_ARTIFACT: dict[str, Any] = {
    "role": "isa_statistics_facts",
    "kind": "public_aggregate_reference",
    "resource": "hmrc_isa_statistics_facts.json",
    "format": "json",
    "runtime_sha256_required": True,
}
ISA_FACTS_VERIFY_OPERATION: dict[str, Any] = {
    "kind": "verify_vendored_fact_resource",
    "artifact_role": "isa_statistics_facts",
    "resource": "hmrc_isa_statistics_facts.json",
    "feed_pin": "chronicle_feed.json",
    "record_sets": list(HMRC_ISA_STATISTICS_RECORD_SETS),
    "source_vintage": "2023-24 individuals; Table 9.4 for 2023-24 and 2024-25",
    "mapped_build_period": 2024,
    "period_mapping": "individual_tables_grown_to_the_table_9_4_build_year",
    "require_before_source_read": True,
    "runtime_sha256_required": True,
    "fail_on_mismatch": True,
}
#: The ``was_lisa`` anchor to HMRC's 2024-25 Lifetime ISA payers.
LISA_ANCHOR_OPERATION: dict[str, Any] = {
    "kind": "anchor_lifetime_isa_ownership",
    "output": "has_lifetime_isa",
    "target": "table_9_4_lifetime_isa_accounts_subscribed",
    "tax_year": 2024,
    "ages": {"minimum": 18, "maximum": 47},
    "counted_ages": {"minimum": 18, "maximum": 47},
}

ISA_STAGE_OPERATIONS: tuple[dict[str, Any], ...] = (
    ISA_FACTS_VERIFY_OPERATION,
    {
        "kind": "clean_was_isa_person_donor",
        "household_key": "CASER8",
        "household_weight": "R8xshhwgt",
        "person_columns": {
            "person_number": "PersonR8",
            "is_dependent_child": "IsDepR8",
            "age_band": "DVAge17R8",
            "cash_isa_value": "DVCISAvR8",
            "stocks_and_shares_isa_value": "DVIISAvR8",
        },
        "household_aggregates": {
            "cash_isa": "DVCISAVR8_aggr",
            "stocks_and_shares_isa": "DVIISAVR8_aggr",
        },
        "aggregate_tolerance_gbp": 1.0,
        "non_dependent_code": 2,
    },
    {
        "kind": "split_household_isa_balances",
        "outputs": ["cash_isa_balance", "stocks_and_shares_isa_balance"],
        "seed": 0,
        "salt": "isa_subscriptions:balance_split",
        "maximum_donor_adults": 4,
    },
    {
        "kind": "allocate_isa_subscribers",
        "seed": 0,
        "salt": "isa_subscriptions:subscribers",
        "minimum_age": 18,
        "selection_order": [
            "cash_and_stocks_and_shares",
            "stocks_and_shares_only",
            "cash_only",
        ],
    },
    {
        "kind": "draw_isa_subscription_amounts",
        "outputs": ["cash_isa_subscriptions", "stocks_and_shares_isa_subscriptions"],
        "seed": 0,
        "salt": "isa_subscriptions:amounts",
        "within_band": "uniform",
        "balance_rank_weight": 0.5,
    },
    {
        "kind": "split_both_type_subscriptions",
        "anchor": "table_9_4_stocks_and_shares_residual",
        "share_floor": 0.02,
    },
    {
        "kind": "grow_isa_subscriptions",
        "people": "table_9_4_account_growth",
        "amounts": "table_9_4_amount_level_limit_held",
        "from_tax_year": 2023,
        "to_tax_year": 2024,
        "overall_limit_gbp": 20000.0,
    },
    {
        "kind": "draw_lifetime_isa_payments",
        "outputs": ["lifetime_isa_payments"],
        "seed": 0,
        "salt": "isa_subscriptions:lifetime_isa_payments",
        "minimum_age": 18,
        "maximum_age": 47,
        "annual_limit_gbp": 4000.0,
    },
    {
        "kind": "record_mass_conservation_receipt",
        "entity": "household",
        "reason": UK_ISA_SUBSCRIPTIONS_MASS_CONSERVATION_REASON,
        "declared_factor": 1.0,
        "gate_coupling": (
            "The terminal family gate requires a valid mass-conserving "
            "MassChangeRecord carrying exactly this stage-specific reason."
        ),
    },
    {"kind": "classify_isa_facts_with_reviewed_fence", **UK_ISA_FACT_FENCE},
)


def isa_stage_spec() -> SourceStageSpec:
    """The stage declaration the tests run against."""

    return SourceStageSpec(
        stage="isa_subscriptions",
        survey="HMRC ISA statistics; Wealth and Assets Survey round 8",
        source="synthetic",
        grain="person",
        artifacts=(
            ISA_FACTS_ARTIFACT,
            {"role": "was_qrf_donor", "filename": "was_household.tab"},
            {"role": "was_person_tab", "filename": "was_person.tab"},
        ),
        operations=tuple(
            SourceOperationSpec(
                kind=operation["kind"],
                parameters={
                    key: value for key, value in operation.items() if key != "kind"
                },
            )
            for operation in ISA_STAGE_OPERATIONS
        ),
        outputs=(
            "cash_isa_balance",
            "stocks_and_shares_isa_balance",
            "cash_isa_subscriptions",
            "stocks_and_shares_isa_subscriptions",
            "lifetime_isa_payments",
        ),
    )


def synthetic_isa_statistics_payload(scale: float = 1.0) -> dict[str, Any]:
    """Every record the loader reads, with invented consistent values."""

    rows: list[dict[str, Any]] = []

    def add(record_id: str, value: float) -> None:
        rows.append({"source_record_id": record_id, "value": float(value)})

    income = _INCOME_SHARES / _INCOME_SHARES.sum()
    bands = _BAND_SHARES / _BAND_SHARES.sum()
    blocks: dict[str, np.ndarray] = {}
    for isa_type, total in TYPE_TOTALS.items():
        blocks[isa_type] = np.outer(income, bands) * total * scale
    blocks["all"] = sum(blocks[t] for t in HMRC_ISA_TYPES)
    for isa_type, cells in blocks.items():
        record_set = f"{PREFIX}table9_7.ty2023.{isa_type}"
        rows_with_total = np.vstack([cells, cells.sum(axis=0)])
        for i, band_row in enumerate((*HMRC_ISA_INCOME_BAND_IDS, "all")):
            for b, band in enumerate(HMRC_ISA_SUBSCRIPTION_BAND_IDS):
                add(
                    f"{record_set}.subscribers.{band_row}.{band}", rows_with_total[i, b]
                )
            subscribers = rows_with_total[i].sum()
            add(f"{record_set}.subscribers.{band_row}.all_subscriptions", subscribers)
            add(
                f"{record_set}.holders.{band_row}.no_further_subscription",
                subscribers * 0.8,
            )
    measures = {
        "cash_only": "cash_only_subscribers",
        "stocks_and_shares_only": "stocks_and_shares_only_subscribers",
        "cash_and_stocks_and_shares": "cash_and_stocks_and_shares_subscribers",
        "all": "all_subscribers",
    }
    ages = _AGE_SHARES / _AGE_SHARES.sum()
    for sex, sex_share in (("male", 0.48), ("female", 0.52), ("all", 1.0)):
        record_set = f"{PREFIX}table9_8.ty2023.{sex}"
        for a, age in enumerate(HMRC_ISA_AGE_BAND_IDS):
            age_share = ages[a]
            for isa_type, measure in measures.items():
                total = (
                    sum(TYPE_TOTALS.values())
                    if isa_type == "all"
                    else TYPE_TOTALS[isa_type]
                )
                add(
                    f"{record_set}.subscribers.{age}.{measure}",
                    total * scale * sex_share * age_share,
                )
    regions = _REGION_SHARES / _REGION_SHARES.sum()
    england = [
        index for index, row in enumerate(HMRC_ISA_REGION_IDS) if row.startswith("E12")
    ]
    for isa_type, measure in measures.items():
        total = (
            sum(TYPE_TOTALS.values()) if isa_type == "all" else TYPE_TOTALS[isa_type]
        )
        for r, row in enumerate(HMRC_ISA_REGION_IDS):
            add(
                f"{PREFIX}table9_9.ty2023.subscribers.{row}.{measure}",
                total * scale * regions[r],
            )
        add(
            f"{PREFIX}table9_9.ty2023.subscribers.E92000001.{measure}",
            total * scale * regions[england].sum(),
        )
    accounts = {
        2023: {
            "cash": 10_000.0,
            "stocks_and_shares": 4_000.0,
            "innovative_finance": 10.0,
            "lifetime": 900.0,
        },
        2024: {
            "cash": 10_800.0,
            "stocks_and_shares": 4_800.0,
            "innovative_finance": 9.0,
            "lifetime": 1_100.0,
        },
    }
    amounts = {
        2023: {
            "cash": 60e6,
            "stocks_and_shares": 30e6,
            "innovative_finance": 0.1e6,
            "lifetime": 2.2e6,
        },
        2024: {
            "cash": 80e6,
            "stocks_and_shares": 36e6,
            "innovative_finance": 0.1e6,
            "lifetime": 2.7e6,
        },
    }
    for year in (2023, 2024):
        for component in (
            "cash",
            "stocks_and_shares",
            "innovative_finance",
            "lifetime",
        ):
            add(
                f"{PREFIX}table9_4.accounts.ty{year}.{component}.accounts",
                accounts[year][component] * scale,
            )
            add(
                f"{PREFIX}table9_4.amounts.ty{year}.{component}.amounts",
                amounts[year][component] * scale,
            )
        add(
            f"{PREFIX}table9_4.accounts.ty{year}.all.accounts",
            sum(accounts[year].values()) * scale,
        )
        add(
            f"{PREFIX}table9_4.amounts.ty{year}.all.amounts",
            sum(amounts[year].values()) * scale,
        )
    for component, value in (
        ("stocks_and_shares", 500e6),
        ("cash", 400e6),
        ("innovative_finance", 0.5e6),
        ("all", 900.5e6),
    ):
        add(f"{PREFIX}table9_6.ty2024.{component}.all.market_value", value * scale)
    add(
        f"{PREFIX}table9_6.ty2024.stocks_and_shares.cash_on_deposit.market_value",
        20e6 * scale,
    )
    note = "hmrc.cash_isa_limit_reduction_2026.cash_subscribers_under_65.ty2022"
    add(f"{note}.under_12000.share_of_cash_subscribers", 78)
    add(f"{note}.over_12000.share_of_cash_subscribers", 22)
    for year in (2024, 2025):
        add(
            f"hmrc.lisa_statistics_2026.table1a.ty{year}.tax_year_{year}.house_purchase_withdrawers",
            90.0 * scale,
        )
    return {"schema_version": 1, "kind": "uk_vendored_ledger_facts", "rows": rows}


def synthetic_was_isa_tabs(
    households: int = 120, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """WAS round-8 household and person tabs whose person ISA values pin to the aggregates.

    The household tab carries the case number, the weight and the ISA aggregates.
    """

    rng = np.random.default_rng(seed)
    people = []
    house_rows = []
    for case in range(1, households + 1):
        adults = int(rng.integers(1, 4))
        cash = np.where(
            rng.random(adults) < 0.6, rng.integers(500, 30_000, adults), 0
        ).astype(float)
        stocks = np.where(
            rng.random(adults) < 0.3, rng.integers(500, 60_000, adults), 0
        ).astype(float)
        for person in range(adults):
            people.append(
                {
                    "CASER8": case,
                    "PersonR8": person + 1,
                    "IsDepR8": 2,
                    "DVAge17R8": int(rng.integers(4, 15)),
                    "DVCISAvR8": cash[person],
                    "DVIISAvR8": stocks[person],
                }
            )
        people.append(
            {
                "CASER8": case,
                "PersonR8": adults + 1,
                "IsDepR8": 1,
                "DVAge17R8": 1,
                "DVCISAvR8": 0.0,
                "DVIISAvR8": 0.0,
            }
        )
        house_rows.append(
            {
                "CASER8": case,
                "R8xshhwgt": float(rng.integers(500, 2_000)),
                "DVCISAVR8_aggr": float(cash.sum()),
                "DVIISAVR8_aggr": float(stocks.sum()),
            }
        )
    return pd.DataFrame(house_rows), pd.DataFrame(people)
