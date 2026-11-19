"""The Microcosm-derived fact resources compiled beside the pinned Chronicle feed.

Two committed resources carry facts Microcosm computes from public data, in
consumer-fact row shape under the ``microcosm.derived_fact.`` key namespace:
the high-value home stock (``uk/high_value_home_stock_facts.json``,
microcosm#1240) and DESNZ's domestic energy split onto constituencies
(``uk/desnz_constituency_energy_facts.json``, microcosm#1095). Every compile of
the UK target references reads both beside the feed, so they are listed here
once; a compile that left one out would find its family's cells absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from microcosm.build.uk_runtime.desnz_constituency_energy import (
    load_uk_desnz_constituency_energy_facts,
)
from microcosm.build.uk_runtime.high_value_homes import (
    load_uk_high_value_home_stock_facts,
)

__all__ = [
    "load_uk_derived_facts",
    "uk_derived_fact_resource_pins",
    "uk_derived_fact_rows",
]


def load_uk_derived_facts() -> dict[str, Any]:
    """Each committed derived resource, validated, keyed by its file name's stem."""

    return {
        "high_value_home_stock_facts": load_uk_high_value_home_stock_facts(),
        "desnz_constituency_energy_facts": load_uk_desnz_constituency_energy_facts(),
    }


def uk_derived_fact_rows(
    derived: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Every committed derived fact row, the high-value home stock's first."""

    resources = load_uk_derived_facts() if derived is None else derived
    return tuple(row for facts in resources.values() for row in facts.rows)


def uk_derived_fact_resource_pins(
    derived: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str | int]]:
    """Each committed derived resource's identity, keyed by its file name."""

    resources = load_uk_derived_facts() if derived is None else derived
    return {
        f"{stem}.json": {
            "resource": f"{stem}.json",
            "rows": len(facts.rows),
            "rows_sha256": facts.rows_sha256,
            "inputs_sha256": facts.inputs_sha256,
        }
        for stem, facts in resources.items()
    }
