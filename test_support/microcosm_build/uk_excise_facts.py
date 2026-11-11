"""Synthetic excise facts for the LCFS excise step's tests (microcosm#1359).

The values are made up, not publisher rows: they exercise the step's
arithmetic and its checks, so no test copies a published figure. The real
rows come from the vendored Chronicle resource (PolicyEngine/chronicle#381).
"""

from __future__ import annotations

import copy
from typing import Any

EXCISE_RESOURCE = "excise_facts.json"
ALCOHOL = "synthetic.alcohol_clearances"
ALCOHOL_BY_RELIEF = "synthetic.alcohol_clearances_by_relief"
TOBACCO = "synthetic.tobacco_clearances"
ON_TRADE = "synthetic.alcohol_on_trade_share"
RECEIPTS = "synthetic.betting_and_gaming_duty_receipts"
LIABILITIES = "synthetic.machine_games_duty_liabilities"
YIELD = "synthetic.gross_gambling_yield"


def _row(
    concept: str,
    dimensions: dict[str, str],
    value: float,
    unit: str,
    geography: str = "K02000001",
) -> dict[str, Any]:
    key = ".".join(dimensions.values())
    return {
        "concept": concept,
        "unit": unit,
        "period": {"type": "fiscal_year", "value": 2024},
        "period_coverage": {"start_date": "2024-04-01", "end_date": "2025-03-31"},
        "geography": {"id": geography},
        "dimensions": dict(dimensions),
        "value": value,
        "source_record_id": f"{concept}.{key}.fy2024",
    }


def _yield(sector: str, activity: str, event: str, value: float) -> dict[str, Any]:
    return _row(
        YIELD,
        {
            "gambling_sector": sector,
            "gambling_activity": activity,
            "betting_event": event,
        },
        value,
        "gbp",
        geography="K03000001",
    )


#: In the units Chronicle stores (litres of pure alcohol, sticks, kilograms,
#: pounds) and a share: round numbers no publisher printed.
EXCISE_FACT_ROWS: list[dict[str, Any]] = [
    _row(ALCOHOL, {"product": "beer_and_cider"}, 2.0e5, "litre_of_pure_alcohol"),
    _row(ALCOHOL, {"product": "wine"}, 1.0e5, "litre_of_pure_alcohol"),
    _row(ALCOHOL, {"product": "spirits"}, 0.5e5, "litre_of_pure_alcohol"),
    *(
        _row(
            ALCOHOL_BY_RELIEF,
            {"product": "beer_and_cider", "relief": relief},
            value,
            "litre_of_pure_alcohol",
        )
        for relief, value in (
            ("draught_relief_only", 0.4e5),
            ("draught_and_small_producer_relief", 0.1e5),
            ("all", 2.0e5),
        )
    ),
    _row(TOBACCO, {"product": "cigarettes"}, 1.0e8, "stick"),
    _row(TOBACCO, {"product": "hand_rolling"}, 500.0, "kilogram"),
    _row(TOBACCO, {"product": "other"}, 100.0, "kilogram"),
    _row(ON_TRADE, {"product": "all_alcohol"}, 0.25, "ratio"),
    *(
        _row(
            RECEIPTS, {"accounting_basis": "cash", "gambling_duty": duty}, value, "gbp"
        )
        for duty, value in (
            ("remote_gaming_duty", 2.1e6),
            ("general_betting_duty", 1.5e6),
            ("pool_betting_duty", 1.5e4),
            ("bingo_duty", 1.0e4),
            ("lottery_duty", 1.2e6),
            ("gaming_duty", 2.0e5),
        )
    ),
    *(
        _row(LIABILITIES, {"mgd_rate": band}, value, "gbp")
        for band, value in (
            ("lower_rate", 5.0e3),
            ("standard_rate", 6.0e5),
            ("higher_rate", 2.5e4),
        )
    ),
    _yield("betting_non_remote", "off_course_over_the_counter", "all", 4.0e6),
    _yield("betting_remote", "betting", "all", 4.0e6),
    _yield("betting_remote", "betting_exchange", "all", 1.0e6),
    _yield("betting_non_remote", "pool_betting", "all", 0.6e6),
    _yield("betting_remote", "pool_betting", "all", 0.4e6),
    _yield("betting_remote", "all", "horses", 1.5e6),
    _yield("betting_remote", "all", "all", 6.0e6),
]


def _alcohol(product: str, strength: float, parameter: str) -> dict[str, Any]:
    return {
        "concept": ALCOHOL,
        "dimensions": {"product": product},
        "unit": "litre_of_pure_alcohol",
        "strength": {
            "value": strength,
            "parameter_path": f"household.consumption.alcohol.{parameter}",
        },
    }


def _tobacco(product: str, unit: str) -> dict[str, Any]:
    return {"concept": TOBACCO, "dimensions": {"product": product}, "unit": unit}


def _rate(value: float, path: str) -> dict[str, Any]:
    return {"value": value, "parameter_path": path}


EXCISE_DECLARATION: dict[str, Any] = {
    "kind": "level_excise_quantities",
    "resource": EXCISE_RESOURCE,
    "period_type": "fiscal_year",
    "period_value": 2024,
    "fiscal_start": "2024-04-01",
    "geography_id": "K02000001",
    "weeks_per_year": 52,
    "alcohol_volume_weights": {"rule": "spending"},
    "outputs": {
        "beer_litres": _alcohol("beer_and_cider", 0.04, "beer_abv"),
        "wine_litres": _alcohol("wine", 0.12, "wine_abv"),
        "spirits_litres": _alcohol("spirits", 0.40, "spirits_abv"),
        "cigarettes_per_week": _tobacco("cigarettes", "stick"),
        "hand_rolling_tobacco_grams_per_week": _tobacco("hand_rolling", "kilogram"),
        "other_tobacco_grams_per_week": _tobacco("other", "kilogram"),
    },
    "draught_shares": {
        "beer_draught_share": {
            "concept": ALCOHOL_BY_RELIEF,
            "unit": "litre_of_pure_alcohol",
            "numerator": [
                {"product": "beer_and_cider", "relief": "draught_relief_only"},
                {
                    "product": "beer_and_cider",
                    "relief": "draught_and_small_producer_relief",
                },
            ],
            "denominator": [{"product": "beer_and_cider", "relief": "all"}],
        },
    },
    "gambling": {
        "receipts": {"concept": RECEIPTS, "dimensions": {"accounting_basis": "cash"}},
        "machine_games_liabilities": {"concept": LIABILITIES},
        "yield": {"concept": YIELD, "geography_id": "K03000001"},
        "rates": {
            "general_betting": _rate(
                0.15, "gov.hmrc.gambling_duty.general_betting.rates.fixed_odds"
            ),
            "pool_betting": _rate(0.15, "gov.hmrc.gambling_duty.pool_betting.rate"),
            "remote_gaming": _rate(0.21, "gov.hmrc.gambling_duty.remote_gaming.rate"),
            "machine_games_lower": _rate(
                0.05, "gov.hmrc.gambling_duty.machine_games.rates.lower"
            ),
            "machine_games_standard": _rate(
                0.20, "gov.hmrc.gambling_duty.machine_games.rates.standard"
            ),
            "machine_games_higher": _rate(
                0.25, "gov.hmrc.gambling_duty.machine_games.rates.higher"
            ),
            "bingo": _rate(0.10, "gov.hmrc.gambling_duty.bingo.rate"),
            "lottery": _rate(0.12, "gov.hmrc.gambling_duty.lottery.rate"),
            "casino": _rate(
                0.235487, "household.consumption.gambling.casino_effective_rate"
            ),
        },
    },
}

#: The on-trade channel's volume as a published share, for the channel rule.
ON_TRADE_VOLUME_WEIGHTS: dict[str, Any] = {
    "rule": "published_on_trade_share",
    "concept": ON_TRADE,
    "dimensions": {"product": "all_alcohol"},
}


def excise_declaration(**changes: Any) -> dict[str, Any]:
    """A copy of the synthetic declaration with top-level keys replaced."""

    declaration = copy.deepcopy(EXCISE_DECLARATION)
    declaration.update(copy.deepcopy(changes))
    return declaration


def patch_excise_facts(monkeypatch, declaration: dict[str, Any] | None = None) -> None:
    """Serve the synthetic rows for the excise resource, the real ones otherwise.

    With ``declaration`` the stage also reads it as its declared step, for the
    gate, which reads the committed spec's.
    """

    from microcosm.build.uk_runtime import lcfs_consumption
    from microcosm.build.uk_runtime.ledger_fact_vendoring import rows_matching

    real = lcfs_consumption.vendored_rows

    def vendored_rows(name, *, fiscal_start=None, **criteria):
        if name != EXCISE_RESOURCE:
            return real(name, fiscal_start=fiscal_start, **criteria)
        rows = rows_matching({"rows": EXCISE_FACT_ROWS}, **criteria)
        return [
            row
            for row in rows
            if fiscal_start is None
            or row["period_coverage"]["start_date"] == str(fiscal_start)
        ]

    monkeypatch.setattr(lcfs_consumption, "vendored_rows", vendored_rows)
    if declaration is not None:
        monkeypatch.setattr(
            lcfs_consumption,
            "excise_quantities_operation",
            lambda stage: copy.deepcopy(declaration),
        )
