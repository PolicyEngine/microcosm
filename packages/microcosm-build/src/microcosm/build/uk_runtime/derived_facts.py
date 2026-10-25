"""The Microcosm-derived fact resources every UK national compile binds.

Two committed resources carry ``uk_derived_consumer_facts`` rows that contract
targets select with ``source_name: microcosm`` beside the Chronicle feed: the
high-value home stock (``uk/high_value_home_stock_facts.json``; local cells
microcosm#1240, region rows microcosm#1301) and CenTax's tabulations of parents
of 3- and 4-year-olds (``uk/centax_childcare_parent_facts.json``,
microcosm#1399). A compile of the committed national references that misses
either reports its rows unsupported, so every national compile site merges them
through :func:`uk_derived_fact_rows`, and every receipt names both resources
through :func:`uk_derived_fact_provenance`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from microcosm.build.uk_runtime.centax_childcare_parents import (
    load_uk_centax_childcare_parent_facts,
)
from microcosm.build.uk_runtime.high_value_homes import (
    load_uk_high_value_home_stock_facts,
)


def uk_derived_fact_rows() -> tuple[Mapping[str, Any], ...]:
    """Both committed derived-fact resources' rows, high-value homes first."""

    return (
        *load_uk_high_value_home_stock_facts().rows,
        *load_uk_centax_childcare_parent_facts().rows,
    )


def uk_derived_fact_provenance() -> dict[str, dict[str, Any]]:
    """Each committed derived-fact resource's identity, keyed by its stem."""

    high_value = load_uk_high_value_home_stock_facts()
    centax = load_uk_centax_childcare_parent_facts()
    return {
        "high_value_home_stock_facts": {
            "rows": len(high_value.rows),
            "rows_sha256": high_value.rows_sha256,
            "inputs_sha256": high_value.inputs_sha256,
            "generator": dict(high_value.generator),
        },
        "centax_childcare_parent_facts": {
            "rows": len(centax.rows),
            "rows_sha256": centax.rows_sha256,
            "inputs_sha256": centax.inputs_sha256,
            "generator": dict(centax.generator),
        },
    }
