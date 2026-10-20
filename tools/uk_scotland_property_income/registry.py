"""The SPI 2023-24 facts the national build binds, read back to 2023-24 (microcosm#1326 A4).

The national target registry holds each SPI fact uprated to the calibration
period. Its metadata records the factor (``uprating_factor``) and, for the
regional rows, the value before the cross-grain control
(``cross_grain_value_before``), so the published 2023-24 value is the value
before the control divided by the factor.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The measures this tool evaluates, as ``measures.contribution`` defines them.
MEASURES = (
    "property_amount",
    "property_count",
    "taxpayers",
    "total_income",
    "income_tax",
)

_PROPERTY = re.compile(
    r"^hmrc/property_income_(?P<count>count_)?income_band_"
    r"(?P<lower>[0-9_]+)_to_(?P<upper>[0-9_]+|inf)$"
)
_REGION = re.compile(
    r"^hmrc\.spi_region\.(?P<measure>taxpayers|total_income|income_tax)_by_region_"
    r"(?P<lower>\d+)_(?P<upper>\d+|plus)@"
)


@dataclass(frozen=True)
class SpiFact:
    """One SPI 2023-24 fact at its published value."""

    name: str
    measure: str
    lower: float
    upper: float
    region: str | None
    value: float


def load_registry(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def published_value(spec: Mapping[str, Any]) -> float:
    """The fact's 2023-24 value, before uprating and any cross-grain control."""

    metadata = spec.get("metadata") or {}
    if str(metadata.get("uprating_from_period")) != "2023":
        raise ValueError(f"{spec['name']} is not a 2023-24 fact.")
    if "uprating_factor" not in metadata:
        raise ValueError(f"{spec['name']} records no uprating factor.")
    base = float(metadata.get("cross_grain_value_before", spec["value"]))
    return base / float(metadata["uprating_factor"])


def _amount(text: str) -> float:
    return float("inf") if text in {"inf", "plus"} else float(text.replace("_", ""))


def _region(spec: Mapping[str, Any]) -> str:
    predicate = json.loads((spec.get("metadata") or {})["geography_predicate"])
    if predicate.get("variable") != "region" or predicate.get("operator") != "==":
        raise ValueError(f"{spec['name']} has an unexpected geography predicate.")
    return str(predicate["value"])


def spi_2023_facts(registry: Mapping[str, Any]) -> list[SpiFact]:
    """Every SPI property and regional fact in the registry, at 2023-24 values."""

    facts = []
    for spec in registry["specs"]:
        name = str(spec["name"])
        if match := _PROPERTY.match(name):
            measure = "property_count" if match["count"] else "property_amount"
            region = None
        elif match := _REGION.match(name):
            measure = match["measure"]
            region = _region(spec)
        else:
            continue
        facts.append(
            SpiFact(
                name=name,
                measure=measure,
                lower=_amount(match["lower"]),
                upper=_amount(match["upper"]),
                region=region,
                value=published_value(spec),
            )
        )
    return facts
