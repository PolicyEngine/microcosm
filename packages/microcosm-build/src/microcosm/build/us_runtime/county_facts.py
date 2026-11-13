"""Pinned county facts for the ACS-local county rung (microcosm#922).

The county rung needs county-grain Chronicle facts: IRS SOI county returns and
AGI, Census PEP county population, and the ACS 5-year county tables. They live
in their own pinned file, ``us/chronicle_county_facts.json`` with scope
``us/chronicle_county_facts_scope.json``, beside the main US feed pin
(``us/chronicle_feed.json``), because carrying them in the main feed would grow
it 6.5x for every US build while only county builds read them. No fiscal
compile reads this file.

:func:`load_us_county_facts` reads the file in one pass: it hashes the exact
bytes, parses every line, refuses a row on an unpinned row contract, at any
geography level but ``county`` or outside the scope's (record set, period)
pairs, drops Puerto Rico (state 72, outside the 3,144 county-equivalents the
County Files cover) with a count, and projects each kept row to the fields the
county rung reads. It refuses the file unless its digest and row count are the
pin's. The file on disk keeps Chronicle's own bytes; only the in-memory rows
are projected.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from microcosm.build.us_runtime.chronicle_feed import (
    US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE,
    USChronicleFeed,
    load_us_chronicle_county_facts_pin,
)

__all__ = [
    "COUNTY_FACT_FIELDS",
    "DROPPED_STATE_FIPS",
    "USCountyFacts",
    "load_us_county_facts",
    "us_county_facts_scope_pairs",
]

#: Top-level row fields kept in memory. Everything else (source, lineage,
#: concept alignment, period coverage, labels) stays in the pinned bytes.
COUNTY_FACT_FIELDS = (
    "aggregate_fact_key",
    "schema_version",
    "geography",
    "observed_measure",
    "dimensions",
    "dimension_value_labels",
    "period",
    "value",
    "value_type",
    "entity",
    "aggregation",
    "universe_constraints",
    "layout",
)
#: Puerto Rico municipios: in the ACS county tables, outside the County Files.
DROPPED_STATE_FIPS = frozenset({"72"})
_COUNTY_GEOID_PREFIX = "0500000US"

Pair = tuple[str, str, str]


@dataclass(frozen=True)
class USCountyFacts:
    """The kept, projected county rows and how they were read."""

    rows: tuple[dict[str, Any], ...]
    path: str
    facts_sha256: str
    row_count: int
    dropped: Mapping[str, int] = field(default_factory=dict)
    pin: USChronicleFeed | None = None

    def receipt(self) -> dict[str, object]:
        return {
            "path": self.path,
            "facts_sha256": self.facts_sha256,
            "row_count": self.row_count,
            "kept_rows": len(self.rows),
            "dropped": dict(self.dropped),
            "pin": None if self.pin is None else self.pin.to_dict(),
        }


def us_county_facts_scope_pairs() -> frozenset[Pair]:
    """The (record set, period type, period value) pairs the county pin keeps."""

    raw = json.loads(
        files("microcosm.build.us")
        .joinpath(US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE)
        .read_bytes()
    )
    return frozenset(
        (entry["record_set_id"], entry["period_type"], str(entry["period_value"]))
        for entry in raw["pairs"]
    )


def _county_state(row: Mapping[str, Any]) -> str:
    geography = row.get("geography")
    if not isinstance(geography, Mapping) or geography.get("level") != "county":
        raise ValueError(
            "US county facts must all be county rows; got geography "
            f"{geography!r} in {row.get('aggregate_fact_key')!r}."
        )
    geoid = str(geography.get("id", ""))
    code = geoid[len(_COUNTY_GEOID_PREFIX) :]
    if (
        not geoid.startswith(_COUNTY_GEOID_PREFIX)
        or len(code) != 5
        or not code.isdigit()
    ):
        raise ValueError(f"US county facts: invalid county geography id {geoid!r}.")
    return code[:2]


def _pair(row: Mapping[str, Any]) -> Pair:
    period = row.get("period") or {}
    layout = row.get("layout") or {}
    return (
        str(layout.get("record_set_id", "")),
        str(period.get("type", "")),
        str(period.get("value", "")),
    )


def load_us_county_facts(
    path: str | Path, *, pin: USChronicleFeed | None = None
) -> USCountyFacts:
    """Read the pinned county facts file once and keep the projected rows."""

    pin = pin or load_us_chronicle_county_facts_pin()
    pairs = us_county_facts_scope_pairs()
    contracts = set(pin.consumer_fact_schema_versions)
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    count = 0
    with open(path, "rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            count += 1
            row = json.loads(line)
            if row.get("schema_version") not in contracts:
                raise ValueError(
                    "US county facts row carries an unpinned row contract "
                    f"{row.get('schema_version')!r}; the pin declares "
                    f"{sorted(contracts)}."
                )
            pair = _pair(row)
            if pair not in pairs:
                raise ValueError(
                    f"US county facts row {row.get('aggregate_fact_key')!r} is "
                    f"outside the county scope: {pair}."
                )
            if _county_state(row) in DROPPED_STATE_FIPS:
                dropped[f"state_{_county_state(row)}"] += 1
                continue
            rows.append({key: row[key] for key in COUNTY_FACT_FIELDS if key in row})
    facts_sha256 = digest.hexdigest()
    if facts_sha256 != pin.facts_sha256:
        raise ValueError(
            f"US county facts {path} has sha256 {facts_sha256}; the county pin "
            f"names {pin.facts_sha256}."
        )
    if count != pin.fact_row_count:
        raise ValueError(
            f"US county facts {path} has {count} rows; the county pin names "
            f"{pin.fact_row_count}."
        )
    return USCountyFacts(
        rows=tuple(rows),
        path=str(path),
        facts_sha256=facts_sha256,
        row_count=count,
        dropped=dict(sorted(dropped.items())),
        pin=pin,
    )
