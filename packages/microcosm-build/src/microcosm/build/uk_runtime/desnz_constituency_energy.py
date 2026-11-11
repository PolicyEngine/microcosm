"""DESNZ domestic energy by constituency, split from the authorities' totals.

PolicyEngine/microcosm#1095 ("Constituency grain"). DESNZ publishes Great
Britain's domestic electricity and gas consumption and meter counts by local
authority and by small area (2021 LSOAs in England and Wales, 2022 data zones
in Scotland), but no constituency cut. The committed resource
``uk/desnz_constituency_energy_facts.json`` carries Microcosm's split of each
authority's published total over the July 2024 constituencies its output areas
fall in, in proportion to DESNZ's own small-area figures for the same year
(María's ruling of 2026-10-25: a census household split makes use per
household uniform within each authority, and moved seats' gas by up to a
third against the small-area split). ``tools/build_uk_desnz_constituency_energy_facts.py``
builds it from the pinned Chronicle feed and the committed atomic supports.
The functions here are pure, so the tool and its tests share them, and the
loader is the one place that knows the resource's shape.

Method, per fuel and measure (consumption, meter count):

1. Each output area's key is its small area's published value times the
   output area's share of that small area's census households (the support's
   ``households``). A small area DESNZ publishes no row for carries no key.
2. The pieces, one per authority and constituency, sum their output areas'
   keys, and each takes its authority's published total times its share of
   the authority's key. Every authority keeps its total, and DESNZ's
   unallocated meters (one Great Britain row) follow the split.
3. A constituency's value is the sum of its pieces.

The rows ride in consumer-fact shape under the Microcosm-owned
``microcosm.derived_fact.`` key namespace and keep DESNZ's own concept, period
and dimensions, so the energy appliers treat seat and authority cells alike.
Northern Ireland is out of scope: DESNZ publishes no small areas for it, and
the family binds Great Britain only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "DERIVED_FACT_KEY_PREFIX",
    "EnergySeries",
    "FACTS_RESOURCE",
    "SeatSplit",
    "UKDesnzConstituencyEnergyFacts",
    "load_uk_desnz_constituency_energy_facts",
    "rows_sha256",
    "seat_fact_rows",
    "seat_meter_counts",
    "select_series",
    "split_series",
    "support_frame",
]

DERIVED_FACT_KEY_PREFIX = "microcosm.derived_fact.desnz_constituency_energy.v1:"
SEMANTIC_FACT_KEY_PREFIX = "microcosm.semantic_fact.desnz_constituency_energy.v1:"
RESOURCE_KIND = "uk_derived_consumer_facts"
ROW_SCHEMA_VERSION = "microcosm.derived_consumer_fact.v1"
FACTS_RESOURCE = "desnz_constituency_energy_facts.json"
METHOD_VERSION = "uk-desnz-constituency-energy-v1"
SUPPORT_PROVENANCE_RESOURCE = "uk_atomic_area_supports.provenance.json"
CROSSWALK_RESOURCE = "local_area_crosswalk.json"
#: The atomic supports the split reads: Great Britain's output areas.
SUPPORTS: tuple[str, ...] = ("uk_ew_output_area_2021", "uk_scotland_output_area_2022")
SUPPORT_COLUMNS: tuple[str, ...] = (
    "area",
    "lsoa_code",
    "local_authority_code",
    "constituency_code",
    "region_code",
    "households",
)
CONSTITUENCY_LEVEL = "constituency"
CONSTITUENCY_VINTAGE = "pcon_2024"
SMALL_AREA_VINTAGES = frozenset({"lsoa_2021", "dz_2022"})
UNALLOCATED_GEOGRAPHY_ID = "desnz:great_britain_unallocated"
RECORD_SET_SPEC_PREFIX = "uk.local_geography.desnz_constituency_energy."
#: A small area's published values within one authority may differ from the
#: authority's total by DESNZ's unallocated and unpublished meters, which the
#: split hands back pro rata; past this share the two publications disagree
#: about the authority and the split is refused (2024: within 0.7%).
SMALL_AREA_SUM_TRIPWIRE = 0.02
FUELS: tuple[str, ...] = ("electricity", "gas")
METRICS: tuple[str, ...] = ("consumption", "meter_count")
CONCEPTS: Mapping[tuple[str, str], str] = {
    (
        "electricity",
        "consumption",
    ): "desnz.subnational.domestic_electricity_consumption",
    (
        "electricity",
        "meter_count",
    ): "desnz.subnational.domestic_electricity_meter_count",
    ("gas", "consumption"): "desnz.subnational.domestic_gas_consumption",
    ("gas", "meter_count"): "desnz.subnational.domestic_gas_meter_count",
}
UNITS: Mapping[str, str] = {"consumption": "kwh", "meter_count": "count"}
MEASURE_LABELS: Mapping[tuple[str, str], str] = {
    ("electricity", "consumption"): "Domestic electricity consumption",
    ("electricity", "meter_count"): "Domestic electricity meters",
    ("gas", "consumption"): "Domestic gas consumption",
    ("gas", "meter_count"): "Domestic gas meters",
}


def authority_record_set_id(fuel: str, year: int) -> str:
    """DESNZ's authority record set of one fuel and year in the feed."""

    return f"desnz.subnational.{fuel}.cy{year}"


def small_area_record_set_id(fuel: str, year: int) -> str:
    """DESNZ's LSOA and data zone record set of one fuel and year in the feed.

    Electricity years are calendar years and gas years run from mid-May; the
    feed labels both by their opening year.
    """

    return f"desnz.lsoa.{fuel}.{'gy' if fuel == 'gas' else 'cy'}{year}"


@dataclass(frozen=True)
class EnergySeries:
    """One fuel and measure: the authorities' published totals and the small areas'."""

    fuel: str
    metric: str
    authority_totals: Mapping[str, float]
    small_area_values: Mapping[str, float]
    unallocated: float | None
    #: One published authority row, whose concept, period and dimensions the
    #: seat rows keep.
    template: Mapping[str, Any]


@dataclass(frozen=True)
class SeatSplit:
    """The split of one series: seat values, the pieces and the receipt."""

    seats: Mapping[str, float]
    pieces: pd.DataFrame
    receipt: Mapping[str, Any]


def _metric_of(row: Mapping[str, Any]) -> str:
    return str((row.get("dimensions") or {}).get("metric") or "")


def _unit_of(row: Mapping[str, Any]) -> str:
    return str((row.get("observed_measure") or {}).get("unit") or "")


def _finite_value(row: Mapping[str, Any], what: str) -> float:
    value = row.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{what} has an invalid value {value!r}.")
    return float(value)


def select_series(
    facts: Iterable[Mapping[str, Any]],
    *,
    year: int,
    aliases: Mapping[str, str],
) -> dict[tuple[str, str], EnergySeries]:
    """The four series of one year from DESNZ's authority and small-area rows.

    ``aliases`` maps a publisher's recoded authority to the roster code
    (``local_area_crosswalk.json``'s ``code_aliases``). Two rows for one area,
    a row in another unit and a series with no authority or no small area are
    refused.
    """

    record_sets = {
        authority_record_set_id(fuel, year): ("authority", fuel) for fuel in FUELS
    } | {small_area_record_set_id(fuel, year): ("small_area", fuel) for fuel in FUELS}
    totals: dict[tuple[str, str], dict[str, float]] = {}
    small: dict[tuple[str, str], dict[str, float]] = {}
    unallocated: dict[tuple[str, str], float] = {}
    templates: dict[tuple[str, str], tuple[str, Mapping[str, Any]]] = {}
    for row in facts:
        record_set = str((row.get("layout") or {}).get("record_set_id") or "")
        if record_set not in record_sets:
            continue
        kind, fuel = record_sets[record_set]
        metric = _metric_of(row)
        if metric not in METRICS:
            continue
        key = (fuel, metric)
        geography = row.get("geography") or {}
        code = str(geography.get("id") or "")
        level = str(geography.get("level") or "")
        what = f"DESNZ {record_set} {metric} row for {code!r}"
        if _unit_of(row) != UNITS[metric]:
            raise ValueError(f"{what} is not in {UNITS[metric]}.")
        concept = str((row.get("observed_measure") or {}).get("source_concept") or "")
        if concept != CONCEPTS[key]:
            raise ValueError(f"{what} carries concept {concept!r}.")
        if kind == "authority":
            if level != "local_authority":
                continue
            area = aliases.get(code, code)
            bucket = totals.setdefault(key, {})
            if area in bucket:
                raise ValueError(f"{what}: two authority rows for {area!r}.")
            bucket[area] = _finite_value(row, what)
            if key not in templates or area < templates[key][0]:
                templates[key] = (area, row)
            continue
        if code == UNALLOCATED_GEOGRAPHY_ID:
            if key in unallocated:
                raise ValueError(f"{what}: two unallocated rows.")
            unallocated[key] = _finite_value(row, what)
            continue
        if level != "statistical_scope" or (
            str(geography.get("vintage") or "") not in SMALL_AREA_VINTAGES
        ):
            continue
        bucket = small.setdefault(key, {})
        if code in bucket:
            raise ValueError(f"{what}: two small-area rows.")
        bucket[code] = _finite_value(row, what)
    series: dict[tuple[str, str], EnergySeries] = {}
    for fuel in FUELS:
        for metric in METRICS:
            key = (fuel, metric)
            if not totals.get(key) or not small.get(key):
                raise ValueError(
                    f"DESNZ {fuel} {metric} for {year}: the feed carries "
                    f"{len(totals.get(key, {}))} authority rows and "
                    f"{len(small.get(key, {}))} small-area rows; the split needs both."
                )
            series[key] = EnergySeries(
                fuel=fuel,
                metric=metric,
                authority_totals=dict(sorted(totals[key].items())),
                small_area_values=dict(sorted(small[key].items())),
                unallocated=unallocated.get(key),
                template=templates[key][1],
            )
    return series


def support_frame(supports: Sequence[Mapping[str, np.ndarray]]) -> pd.DataFrame:
    """The output areas of the given atomic supports, one row each."""

    frames = []
    for arrays in supports:
        missing = [column for column in SUPPORT_COLUMNS if column not in arrays]
        if missing:
            raise ValueError(f"atomic support lacks columns {missing}.")
        frames.append(
            pd.DataFrame(
                {
                    column: (
                        np.asarray(arrays[column], dtype=np.int64)
                        if column == "households"
                        else np.asarray(arrays[column]).astype(str)
                    )
                    for column in SUPPORT_COLUMNS
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    if frame["area"].duplicated().any():
        raise ValueError("an output area appears in two atomic supports.")
    if (frame["households"] < 0).any():
        raise ValueError("an output area has negative households.")
    return frame


def split_series(
    support: pd.DataFrame,
    series: EnergySeries,
    *,
    tripwire: float = SMALL_AREA_SUM_TRIPWIRE,
) -> SeatSplit:
    """Split each authority's total over its constituencies by the small areas.

    Refused: an authority with a total but no output area or no key, a
    published small area outside the support or in an authority without a
    total, a small area with a value but no households, a constituency whose
    output areas carry two region codes, and an authority whose small areas
    sum further from its total than ``tripwire``.
    """

    what = f"DESNZ {series.fuel} {series.metric}"
    totals = series.authority_totals
    values = series.small_area_values
    known_areas = set(support["lsoa_code"])
    outside = sorted(set(values) - known_areas)
    if outside:
        raise ValueError(
            f"{what}: {len(outside)} published small areas are not in the atomic "
            f"supports (first {outside[:3]}): the vintages disagree."
        )
    in_support = set(support["local_authority_code"])
    absent = sorted(set(totals) - in_support)
    if absent:
        raise ValueError(
            f"{what}: authorities {absent[:5]} carry a total but no output area."
        )
    covered = support[support["local_authority_code"].isin(totals)].copy()
    uncovered = support[~support["local_authority_code"].isin(totals)]
    stray = sorted(set(uncovered["lsoa_code"]) & set(values))
    if stray:
        raise ValueError(
            f"{what}: small areas {stray[:3]} carry a value in an authority with "
            "no published total."
        )
    small_area_households = covered.groupby("lsoa_code")["households"].transform("sum")
    covered["value"] = covered["lsoa_code"].map(values)
    published = covered["value"].notna()
    empty = sorted(
        set(covered.loc[published & (small_area_households == 0), "lsoa_code"])
    )
    if empty:
        raise ValueError(
            f"{what}: small areas {empty[:3]} carry a value but no households."
        )
    share = np.where(
        small_area_households > 0,
        covered["households"]
        / small_area_households.where(small_area_households > 0, 1),
        0.0,
    )
    covered["key"] = covered["value"].fillna(0.0).to_numpy() * share
    seat_regions = covered.groupby("constituency_code")["region_code"].nunique()
    straddling_regions = sorted(seat_regions[seat_regions > 1].index)
    if straddling_regions:
        raise ValueError(
            f"{what}: constituencies {straddling_regions[:3]} carry output areas "
            "of two regions."
        )
    authority_key = covered.groupby("local_authority_code")["key"].sum()
    keyless = sorted(
        code
        for code, total in totals.items()
        if total > 0 and float(authority_key.get(code, 0.0)) <= 0
    )
    if keyless:
        raise ValueError(f"{what}: authorities {keyless[:5]} carry a total but no key.")
    small_area_authority = covered.drop_duplicates("lsoa_code").set_index("lsoa_code")[
        "local_authority_code"
    ]
    published_by_authority = (
        pd.Series(values)
        .groupby(small_area_authority.reindex(list(values)).to_numpy())
        .sum()
    )
    ratio = published_by_authority.reindex(list(totals)).fillna(0.0) / pd.Series(
        totals
    ).replace(0.0, np.nan)
    ratio = ratio.dropna()
    off = ratio[(ratio - 1.0).abs() > tripwire]
    if not off.empty:
        raise ValueError(
            f"{what}: the small areas of {sorted(off.index)[:5]} sum more than "
            f"{tripwire:.0%} away from the authority's total."
        )
    pieces = (
        covered.groupby(["local_authority_code", "constituency_code"], sort=True)["key"]
        .sum()
        .reset_index()
    )
    pieces["value"] = [
        totals[authority] * key / float(authority_key[authority])
        if float(authority_key[authority]) > 0
        else 0.0
        for authority, key in zip(
            pieces["local_authority_code"], pieces["key"], strict=True
        )
    ]
    seats = pieces.groupby("constituency_code", sort=True)["value"].sum()
    seats = seats[seats > 0]
    small_area_seats = (
        covered[published].groupby("lsoa_code")["constituency_code"].nunique()
    )
    straddling = set(small_area_seats[small_area_seats > 1].index)
    all_households = float(covered["households"].sum())
    authority_seats = pieces.groupby("local_authority_code")[
        "constituency_code"
    ].nunique()
    seat_authorities = pieces.groupby("constituency_code")[
        "local_authority_code"
    ].unique()
    whole = [
        seat
        for seat, authorities in seat_authorities.items()
        if all(authority_seats[authority] == 1 for authority in authorities)
    ]
    nation = ratio.index.str[0]
    receipt: dict[str, Any] = {
        "authorities_with_total": len(totals),
        "authorities_in_support_without_total": sorted(
            set(uncovered["local_authority_code"])
        ),
        "small_areas_published": len(values),
        "small_areas_in_support_without_row": int(
            covered.loc[~published, "lsoa_code"].nunique()
        ),
        "small_areas_straddling_seats": len(straddling),
        "households_share_in_straddling_small_areas": round(
            float(
                covered.loc[covered["lsoa_code"].isin(straddling), "households"].sum()
            )
            / all_households,
            5,
        )
        if all_households
        else 0.0,
        "small_area_sum_over_authority_total": {
            str(prefix): {
                "min": round(float(group.min()), 5),
                "max": round(float(group.max()), 5),
            }
            for prefix, group in ratio.groupby(nation)
        },
        "unallocated_share_of_authority_total": (
            round(series.unallocated / float(sum(totals.values())), 5)
            if series.unallocated is not None and sum(totals.values()) > 0
            else None
        ),
        "pieces": len(pieces),
        "authorities_split_over_seats": int((authority_seats > 1).sum()),
        "seats": len(seats),
        "seats_of_whole_authorities": len(whole),
        "authority_total": float(sum(totals.values())),
        "seat_total": float(seats.sum()),
    }
    return SeatSplit(seats=seats.to_dict(), pieces=pieces, receipt=receipt)


def _digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]


def seat_fact_rows(
    series: Mapping[tuple[str, str], EnergySeries],
    splits: Mapping[tuple[str, str], SeatSplit],
    *,
    year: int,
    seat_names: Mapping[str, str],
    inputs_sha256: str,
) -> list[dict[str, Any]]:
    """The seat rows of every series, sorted by key.

    Each row keeps its series' published concept, period, dimensions and
    entity, so it compiles through the same selector and appliers as the
    authority rows; a constituency with no piece of a fuel carries no row.
    """

    rows: list[dict[str, Any]] = []
    for key in sorted(splits):
        fuel, metric = key
        template = series[key].template
        concept = CONCEPTS[key]
        unit = UNITS[metric]
        period = dict(template["period"])
        measure_id = str(
            (template.get("observed_measure") or {}).get("source_measure_id")
        )
        dimensions = dict(template.get("dimensions") or {})
        labels = template.get("dimension_labels") or {}
        value_labels = template.get("dimension_value_labels") or {}
        record_set_id = f"{RECORD_SET_SPEC_PREFIX}{fuel}.{metric}"
        for code, value in sorted(splits[key].seats.items()):
            name = str(seat_names.get(code) or "")
            if not name:
                raise ValueError(f"No name for constituency {code}.")
            identity = {
                "record_set_spec_id": f"{record_set_id}.v1",
                "level": CONSTITUENCY_LEVEL,
                "id": code,
                "period": period,
                "inputs_sha256": inputs_sha256,
                "method": METHOD_VERSION,
            }
            digest = _digest(identity)
            rows.append(
                {
                    "schema_version": ROW_SCHEMA_VERSION,
                    "aggregate_fact_key": DERIVED_FACT_KEY_PREFIX + digest,
                    "semantic_fact_key": SEMANTIC_FACT_KEY_PREFIX + digest,
                    "value": float(round(value, 3)),
                    "value_type": "float",
                    "assertion": "observation",
                    "provenance_class": "derived_administrative",
                    "period": period,
                    "period_coverage": dict(template.get("period_coverage") or {}),
                    "geography": {
                        "level": CONSTITUENCY_LEVEL,
                        "id": code,
                        "vintage": CONSTITUENCY_VINTAGE,
                        "name": name,
                    },
                    "entity": dict(template.get("entity") or {}),
                    "aggregation": {"method": "sum"},
                    "observed_measure": {
                        "source_concept": concept,
                        "source_measure_id": measure_id,
                        "unit": unit,
                    },
                    "measure": {"unit": unit},
                    "concept_alignment": {
                        "source_concept": concept,
                        "canonical_concept": concept,
                        "relation": "derived",
                    },
                    "dimensions": dimensions,
                    "dimension_labels": {
                        dimension: labels[dimension] for dimension in dimensions
                    },
                    "dimension_value_labels": {
                        dimension: {
                            str(value_id): value_labels[dimension][str(value_id)]
                        }
                        for dimension, value_id in dimensions.items()
                    },
                    "source": {
                        "source_name": "microcosm",
                        "source_table": (
                            "DESNZ sub-national and LSOA / data zone domestic "
                            "energy statistics, Microcosm derivation"
                        ),
                        "source_file": FACTS_RESOURCE,
                        "vintage": str((template.get("source") or {}).get("vintage")),
                        "source_sha256": inputs_sha256,
                        "url": "tools/build_uk_desnz_constituency_energy_facts.py",
                    },
                    "lineage": {
                        "source_record_id": (
                            f"microcosm.desnz_constituency_energy.{fuel}.{metric}."
                            f"{period['value']}.{CONSTITUENCY_LEVEL}.{code}"
                        )
                    },
                    "layout": {
                        "record_set_id": record_set_id,
                        "record_set_spec_id": f"{record_set_id}.v1",
                        "measure_id": measure_id,
                    },
                    "label": (
                        f"{MEASURE_LABELS[key]}, {period['value']}, {name}: DESNZ's "
                        "authority totals split by its small-area figures"
                    ),
                }
            )
    rows.sort(key=lambda row: row["aggregate_fact_key"])
    if len({row["aggregate_fact_key"] for row in rows}) != len(rows):
        raise ValueError("Derived fact keys collide.")
    return rows


def rows_sha256(rows: Iterable[Mapping[str, object]]) -> str:
    """Digest of consumer-fact rows, independent of their order."""

    digest = hashlib.sha256()
    for line in sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in rows
    ):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# ----------------------------------------------------------------------------
# The committed resource


@dataclass(frozen=True)
class UKDesnzConstituencyEnergyFacts:
    """The validated derived-facts resource."""

    rows: tuple[Mapping[str, Any], ...]
    inputs_sha256: str
    rows_sha256: str
    parameters: Mapping[str, Any]
    inputs: Mapping[str, Any]
    generator: Mapping[str, Any]
    receipts: Mapping[str, Any] = field(default_factory=dict)

    def values(
        self, *, fuel: str, metric: str, period_value: int | str
    ) -> dict[str, float]:
        """Values by constituency for one fuel, measure and period."""

        return {
            str(row["geography"]["id"]): float(row["value"])
            for row in self.rows
            if row["observed_measure"]["source_concept"] == CONCEPTS[(fuel, metric)]
            and str(row["period"]["value"]) == str(period_value)
        }


def _package_text(name: str) -> str:
    from importlib import resources

    return (
        resources.files("microcosm.build.uk").joinpath(name).read_text(encoding="utf-8")
    )


def _validate_row(row: Mapping[str, Any], *, accepted_vintages: frozenset[str]) -> None:
    key = str(row.get("aggregate_fact_key") or "")
    if not key.startswith(DERIVED_FACT_KEY_PREFIX):
        raise ValueError(
            f"constituency energy fact {key!r} is outside the derived-fact namespace."
        )
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError(
            f"constituency energy fact {key!r} has schema {row.get('schema_version')!r}."
        )
    geography = row.get("geography")
    if (
        not isinstance(geography, Mapping)
        or geography.get("level") != CONSTITUENCY_LEVEL
    ):
        raise ValueError(f"constituency energy fact {key!r} is not a constituency row.")
    code = str(geography.get("id") or "")
    if code[:1] not in {"E", "W", "S"} or not str(geography.get("name") or "").strip():
        raise ValueError(
            f"constituency energy fact {key!r} lacks a Great Britain code or a name."
        )
    vintage = str(geography.get("vintage") or "")
    if vintage not in accepted_vintages:
        raise ValueError(
            f"constituency energy fact {key!r} at {code!r} carries vintage {vintage!r}; "
            f"the crosswalk accepts {sorted(accepted_vintages)} for constituencies."
        )
    value = row.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"constituency energy fact {key!r} has invalid value {value!r}."
        )
    if not str(row.get("label") or "").strip():
        raise ValueError(f"constituency energy fact {key!r} requires a label.")
    concept = str((row.get("observed_measure") or {}).get("source_concept") or "")
    if concept not in set(CONCEPTS.values()):
        raise ValueError(f"constituency energy fact {key!r} names concept {concept!r}.")
    dimensions = row.get("dimensions") or {}
    labels = row.get("dimension_labels") or {}
    value_labels = row.get("dimension_value_labels") or {}
    for dimension_id, value_id in dimensions.items():
        if not str(labels.get(dimension_id) or "").strip():
            raise ValueError(
                f"constituency energy fact {key!r} lacks a label for {dimension_id!r}."
            )
        if not str(
            (value_labels.get(dimension_id) or {}).get(str(value_id)) or ""
        ).strip():
            raise ValueError(
                f"constituency energy fact {key!r} lacks a value label for "
                f"{dimension_id!r}={value_id!r}."
            )
    layout = row.get("layout") or {}
    if not str(layout.get("record_set_spec_id") or "").startswith(
        RECORD_SET_SPEC_PREFIX
    ):
        raise ValueError(
            f"constituency energy fact {key!r} lacks its record set spec id."
        )
    if (row.get("aggregation") or {}).get("method") != "sum":
        raise ValueError(f"constituency energy fact {key!r} must aggregate by sum.")


def load_uk_desnz_constituency_energy_facts(
    path: Path | None = None,
    *,
    support_provenance: Mapping[str, Any] | None = None,
    crosswalk: Mapping[str, Any] | None = None,
    feed_facts_sha256: str | None = None,
) -> UKDesnzConstituencyEnergyFacts:
    """Load and validate the derived-facts resource.

    Refused when its ``rows_sha256`` does not match its rows, when it was
    split through atomic supports other than the committed ones, when it was
    built on another Chronicle feed than ``uk/chronicle_feed.json`` pins (the
    rule for every resource taken from the feed: rebuild it on a re-pin), or
    when a row is malformed or carries a boundary vintage the crosswalk does
    not accept for constituencies.
    """

    text = (
        path.read_text(encoding="utf-8")
        if path is not None
        else _package_text(FACTS_RESOURCE)
    )
    payload = json.loads(text)
    if payload.get("kind") != RESOURCE_KIND or payload.get("schema_version") != 1:
        raise ValueError(f"{FACTS_RESOURCE} is not a schema-1 derived-facts resource.")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{FACTS_RESOURCE} carries no rows.")
    if payload.get("row_count") != len(rows):
        raise ValueError(f"{FACTS_RESOURCE} row_count disagrees with its rows.")
    digest = rows_sha256(rows)
    if digest != payload.get("rows_sha256"):
        raise ValueError(f"{FACTS_RESOURCE} rows differ from their recorded digest.")
    inputs = payload.get("inputs") or {}
    provenance = support_provenance or json.loads(
        _package_text(SUPPORT_PROVENANCE_RESOURCE)
    )
    committed = {name: provenance["supports"][name]["sha256"] for name in SUPPORTS}
    if dict(inputs.get("support_sha256") or {}) != committed:
        raise ValueError(
            f"{FACTS_RESOURCE} was split through atomic supports other than the "
            "committed ones; rebuild it with "
            "tools/build_uk_desnz_constituency_energy_facts.py."
        )
    if feed_facts_sha256 is None:
        from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed

        feed_facts_sha256 = load_uk_chronicle_feed().facts_sha256
    built_on_feed = str((inputs.get("ledger_feed") or {}).get("facts_sha256") or "")
    if built_on_feed != feed_facts_sha256:
        raise ValueError(
            f"{FACTS_RESOURCE} was built on another Chronicle feed than "
            "uk/chronicle_feed.json pins; rebuild it with "
            "tools/build_uk_desnz_constituency_energy_facts.py on the pinned feed."
        )
    levels = (crosswalk or json.loads(_package_text(CROSSWALK_RESOURCE)))["levels"]
    expected = levels[CONSTITUENCY_LEVEL].get("expected_vintage")
    accepted = frozenset(
        [expected] if isinstance(expected, str) else list(expected or ())
    )
    keys: set[str] = set()
    for row in rows:
        _validate_row(row, accepted_vintages=accepted)
        key = str(row["aggregate_fact_key"])
        if key in keys:
            raise ValueError(f"constituency energy fact key {key!r} is duplicated.")
        keys.add(key)
    parameters = payload.get("parameters") or {}
    if "year" not in parameters:
        raise ValueError(f"{FACTS_RESOURCE} lacks parameter 'year'.")
    return UKDesnzConstituencyEnergyFacts(
        rows=tuple(rows),
        inputs_sha256=str(payload.get("inputs_sha256") or ""),
        rows_sha256=digest,
        parameters=parameters,
        inputs=inputs,
        generator=payload.get("generator") or {},
        receipts=payload.get("receipts") or {},
    )


def seat_meter_counts(
    facts: UKDesnzConstituencyEnergyFacts,
    *,
    year: int,
    period_types: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """Domestic meter counts per fuel by constituency, for one meter year.

    ``period_types`` names each fuel's period type (calendar years for
    electricity, gas years for gas); a meter row on another period type is
    refused rather than read as a missing fuel.
    """

    meters: dict[str, dict[str, float]] = {}
    for fuel in FUELS:
        concept = CONCEPTS[(fuel, "meter_count")]
        for row in facts.rows:
            if row["observed_measure"]["source_concept"] != concept:
                continue
            period = row.get("period") or {}
            if str(period.get("value")) != str(year):
                continue
            if str(period.get("type")) != period_types[fuel]:
                raise ValueError(
                    f"{FACTS_RESOURCE}: {year} {fuel} meter counts are stamped "
                    f"{period.get('type')!r}, where the {fuel} lookups read "
                    f"{period_types[fuel]!r}."
                )
            code = str(row["geography"]["id"])
            by_fuel = meters.setdefault(code, {})
            if fuel in by_fuel:
                raise ValueError(
                    f"{FACTS_RESOURCE}: two {fuel} meter rows for {code!r}."
                )
            by_fuel[fuel] = float(row["value"])
    return meters
