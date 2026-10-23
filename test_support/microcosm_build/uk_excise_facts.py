"""Synthetic excise facts for the LCFS excise step's tests (microcosm#1359).

The values are made up, not publisher rows: they exercise the step's
arithmetic and its checks, so no test copies a published figure. The real
rows come from the vendored Chronicle resource once PolicyEngine/chronicle#381
lands.
"""

from __future__ import annotations

import copy
from typing import Any

EXCISE_RESOURCE = "obr_excise_facts.json"
ALCOHOL = "synthetic.alcohol_clearances"
TOBACCO = "synthetic.tobacco_clearances"
ON_TRADE = "synthetic.alcohol_on_trade_share"


def _row(concept: str, product: str, value: float, unit: str) -> dict[str, Any]:
    return {
        "concept": concept,
        "unit": unit,
        "period": {"type": "fiscal_year", "value": 2024},
        "period_coverage": {"start_date": "2024-04-01", "end_date": "2025-03-31"},
        "geography": {"id": "K02000001"},
        "dimensions": {"product": product},
        "value": value,
        "source_record_id": f"{concept}.{product}.fy2024",
    }


#: In the units Chronicle stores (litres of pure alcohol, sticks, kilograms) and a
#: share: round numbers no publisher printed.
EXCISE_FACT_ROWS: list[dict[str, Any]] = [
    _row(ALCOHOL, "beer_and_cider", 2.0e5, "litre_of_pure_alcohol"),
    _row(ALCOHOL, "wine", 1.0e5, "litre_of_pure_alcohol"),
    _row(ALCOHOL, "spirits", 0.5e5, "litre_of_pure_alcohol"),
    _row(TOBACCO, "cigarettes", 1.0e8, "stick"),
    _row(TOBACCO, "hand_rolling", 500.0, "kilogram"),
    _row(TOBACCO, "other", 100.0, "kilogram"),
    _row(ON_TRADE, "all_alcohol", 0.25, "ratio"),
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
