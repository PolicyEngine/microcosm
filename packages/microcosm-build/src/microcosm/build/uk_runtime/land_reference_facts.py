"""The reference facts and constants the UK stock cell builder reads.

Everything here is public: Tax Policy Associates' (TPA) model constants and
output, which Chronicle registers raw-only (``tpa-lvt-model-2026``), and
facts of the pinned Chronicle consumer feed. Each reader returns plain
values with what it read, so the builder can record the inputs it used and
fingerprint them (:func:`feed_inputs_fingerprint`): a test then fails when
the pinned feed carries other values than the committed resources were
built from (microcosm#1524).

- **TPA** (:func:`parse_tpa_constants`, :func:`parse_tpa_cells`): the cost
  per m² by built form and the regional factors, read from ``config.py`` as
  syntax and never executed; and TPA's own land and value by authority from
  ``cells.json``, a ratio of totals, for the ranking check.
- **MHCLG's land per new home** (:func:`land_per_new_home`): £ per hectare
  ÷ dwellings per hectare, by sheet, density and price percentile.
- **The Market Value Survey** (:func:`market_value_means`): mean market
  values by region and tenure at July 2024.
- **Council tax** (:func:`band_d_by_authority`, :func:`tax_base_by_authority`):
  each area's Band D, England and Wales, and England's tax base.
- **EPC D3** (:func:`epc_mean_floor_area`): mean floor area per authority
  over every quarter, for a sensitivity.
- **ONS's level inputs** (:func:`resolve_level_input`): each resolves as a
  feed fact, else a value the command line states, else absent.

This module is loaded by file path from the builder tool, so it imports no
``uk_runtime`` sibling.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean

from microcosm.build.derived_resource import payload_sha256

__all__ = [
    "FEED_MARKERS",
    "band_d_by_authority",
    "epc_mean_floor_area",
    "fact_value",
    "feed_inputs_fingerprint",
    "land_per_new_home",
    "market_value_means",
    "parse_tpa_cells",
    "parse_tpa_constants",
    "read_feed_facts",
    "resolve_level_input",
    "select_facts",
    "series_by_period",
    "tax_base_by_authority",
]

#: Strings that mark every feed row the builder reads. A row is parsed only
#: when it holds one, so a 10 GB feed streams in a minute.
FEED_MARKERS: tuple[str, ...] = (
    "uk.local_geography.tenure.",
    "mhclg.residential_land_value_floor_adjusted",
    "mhclg.residential_land_value_before_floor",
    "mhclg.residential_site_density",
    "mhclg.dwelling_stock_density",
    "mhclg.mvs_market_value_mean",
    "mhclg.council_tax.average_band_d_billing_authority_area",
    "mhclg.council_tax.council_tax_base",
    "welshgov.council_tax.average_band_d",
    "mhclg.epc_domestic_total_floor_area",
    "mhclg.epc_domestic_lodgements",
    "ons.land_article_2022.land_underlying_dwellings",
    '"ons.e44',
    '"ons.e46',
    "capital_stock",
    "lps.housing_stock",
)
TPA_COST_NAME = "BASE_COST_PER_M2"
TPA_REGION_NAME = "REGION_FACTOR"
ENGLISH_REGIONS: tuple[str, ...] = tuple(f"E1200000{i}" for i in range(1, 10))


# ----------------------------------------------------------------------------
# TPA's model


def _positive(value: object, what: str) -> float:
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{what} must be finite and positive.")
    return number


def parse_tpa_constants(source: str) -> tuple[dict[str, float], dict[str, float]]:
    """TPA's cost per m² by built form and its regional factors.

    ``config.py`` is parsed as syntax and the two assignments are read by
    literal evaluation, so nothing in it runs. The costs must cover D, S, T
    and F and the factors the nine English regions. TPA's national
    calibration constants in the same file are not read: the level comes
    from ONS.
    """
    found: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in (
                TPA_COST_NAME,
                TPA_REGION_NAME,
            ):
                if target.id in found:
                    raise ValueError(f"TPA's config assigns {target.id} twice.")
                found[target.id] = ast.literal_eval(node.value)
    if set(found) != {TPA_COST_NAME, TPA_REGION_NAME}:
        raise ValueError("TPA's config lacks its rebuilding-cost constants.")
    costs = {
        str(key): _positive(value, f"TPA's cost for {key}")
        for key, value in found[TPA_COST_NAME].items()
    }
    factors = {
        str(key): _positive(value, f"TPA's factor for {key}")
        for key, value in found[TPA_REGION_NAME].items()
    }
    if set(costs) != {"D", "S", "T", "F"}:
        raise ValueError("TPA's costs must cover D, S, T and F.")
    if set(factors) != set(ENGLISH_REGIONS):
        raise ValueError("TPA's regional factors must cover the nine English regions.")
    return costs, factors


def parse_tpa_cells(payload: bytes) -> tuple[dict, dict[str, dict[str, float]]]:
    """TPA's run settings and its land and value per authority.

    Each authority's cells give a count and value and land deciles per
    band; the authority's land share is the ratio of its totals, count ×
    mean of the deciles, not a mean of ratios.
    """
    data = json.loads(payload.decode("utf-8"))
    meta = dict(data.get("meta") or {})
    out: dict[str, dict[str, float]] = {}
    for code, authority in sorted((data.get("las") or {}).items()):
        dwellings = value = land = 0.0
        for band in (authority.get("bands") or {}).values():
            count = float(band["count"])
            values = [float(v) for v in band["value_deciles"]]
            lands = [float(v) for v in band["land_deciles"]]
            if not values or len(values) != len(lands) or count < 0:
                raise ValueError(f"TPA's cells for {code} are malformed.")
            if any(not math.isfinite(v) or v < 0 for v in [*values, *lands]):
                raise ValueError(f"TPA's deciles for {code} are malformed.")
            dwellings += count
            value += count * fmean(values)
            land += count * fmean(lands)
        if not value > 0 or land > value:
            raise ValueError(f"TPA's land and value for {code} are inconsistent.")
        out[str(code)] = {
            "dwellings": dwellings,
            "value_gbp": value,
            "land_gbp": land,
            "land_share": land / value,
        }
    if not out:
        raise ValueError("TPA's cells carry no authority.")
    return meta, out


# ----------------------------------------------------------------------------
# The feed


def read_feed_facts(path: Path, markers: Sequence[str] = FEED_MARKERS) -> list[dict]:
    """The feed rows that hold one of ``markers``, parsed; the rest skipped."""
    needles = tuple(marker.encode("utf-8") for marker in markers)
    rows = []
    with path.open("rb") as stream:
        for line in stream:
            if any(needle in line for needle in needles):
                rows.append(json.loads(line))
    return rows


def fact_value(fact: Mapping) -> float:
    value = fact.get("value")
    if value is None:
        raise ValueError(f"Fact {fact.get('aggregate_fact_key')} has no value.")
    return float(value)


def select_facts(
    facts: Iterable[Mapping],
    *,
    concept: str,
    level: str | None = None,
    spec_prefix: str | None = None,
    dimensions: Mapping[str, str] | None = None,
) -> list[Mapping]:
    """The facts of one source concept, optionally at a level, under a spec
    prefix, and with exactly the given dimensions."""
    out = []
    for fact in facts:
        if (fact.get("observed_measure") or {}).get("source_concept") != concept:
            continue
        if level is not None and (fact.get("geography") or {}).get("level") != level:
            continue
        spec = str((fact.get("layout") or {}).get("record_set_spec_id") or "")
        if spec_prefix is not None and not spec.startswith(spec_prefix):
            continue
        if dimensions is not None and dict(fact.get("dimensions") or {}) != dict(
            dimensions
        ):
            continue
        out.append(fact)
    return out


def _by_area(facts: Iterable[Mapping], what: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for fact in facts:
        code = str((fact.get("geography") or {}).get("id"))
        if code in out:
            raise ValueError(f"{what}: area {code} appears twice.")
        out[code] = fact_value(fact)
    return out


def series_by_period(facts: Iterable[Mapping], what: str) -> dict[str, float]:
    """One value per period, as text, refusing a repeated period."""
    out: dict[str, float] = {}
    for fact in facts:
        period = str((fact.get("period") or {}).get("value"))
        if period in out:
            raise ValueError(f"{what}: period {period} appears twice.")
        out[period] = fact_value(fact)
    return out


def land_per_new_home(
    facts: Iterable[Mapping],
    *,
    concept: str,
    density: str,
    percentile: int,
    density_factor: float = 1.0,
) -> dict[str, float]:
    """MHCLG's land value per new home, by authority: V ÷ (f × d).

    ``V`` is the £ per hectare of ``concept`` (the floor-adjusted sheet, or
    the one before the floor) at the density scenario and new-build price
    percentile; ``d`` is MHCLG's dwellings per hectare for the same density;
    ``f`` scales the density. An authority with no value has no entry.
    """
    facts = list(facts)
    if not density_factor > 0:
        raise ValueError("The density factor must be positive.")
    value = _by_area(
        select_facts(
            facts,
            concept=concept,
            level="local_authority",
            dimensions={
                "new_build_price_percentile": f"p{percentile}",
                "site_density_scenario": density,
            },
        ),
        f"{concept} at p{percentile}, {density} density",
    )
    homes = _by_area(
        select_facts(
            facts,
            concept="mhclg.residential_site_density",
            level="local_authority",
            dimensions={"site_density_scenario": density},
        ),
        f"MHCLG's {density} site density",
    )
    out = {}
    for code, per_hectare in value.items():
        dwellings = homes.get(code)
        if dwellings is None or not dwellings > 0:
            raise ValueError(f"MHCLG gives {code} a land value but no density.")
        if per_hectare < 0:
            raise ValueError(f"MHCLG's land value for {code} is negative.")
        out[code] = per_hectare / (density_factor * dwellings)
    return out


def market_value_means(
    facts: Iterable[Mapping], *, spec_prefix: str
) -> dict[str, dict[str, float]]:
    """Market Value Survey means by area and tenure (``"all"`` for every tenure)."""
    out: dict[str, dict[str, float]] = {}
    for fact in select_facts(
        facts, concept="mhclg.mvs_market_value_mean", spec_prefix=spec_prefix
    ):
        code = str((fact.get("geography") or {}).get("id"))
        dimensions = dict(fact.get("dimensions") or {})
        extra = set(dimensions) - {"ehs_tenure"}
        if extra:
            raise ValueError(f"MVS mean under {spec_prefix} has dimensions {extra}.")
        tenure = dimensions.get("ehs_tenure", "all")
        if tenure in out.setdefault(code, {}):
            raise ValueError(f"MVS mean for {code}, {tenure} appears twice.")
        out[code][tenure] = fact_value(fact)
    if not out:
        raise ValueError(f"The feed has no MVS means under {spec_prefix}.")
    return out


def band_d_by_authority(
    facts: Iterable[Mapping], *, england_spec: str, wales_spec: str
) -> dict[str, float]:
    """Each billing authority's area Band D: England's and Wales's tables."""
    facts = list(facts)
    england = _by_area(
        select_facts(
            facts,
            concept="mhclg.council_tax.average_band_d_billing_authority_area",
            level="local_authority",
            spec_prefix=england_spec,
        ),
        "England's area Band D",
    )
    wales = _by_area(
        select_facts(
            facts,
            concept="welshgov.council_tax.average_band_d",
            level="local_authority",
            spec_prefix=wales_spec,
        ),
        "Wales's Band D",
    )
    if set(england) & set(wales):
        raise ValueError("An authority has a Band D in both tables.")
    return {**england, **wales}


def tax_base_by_authority(facts: Iterable[Mapping], *, spec: str) -> dict[str, float]:
    """England's council tax base (Band D equivalents) by billing authority."""
    return _by_area(
        select_facts(
            facts,
            concept="mhclg.council_tax.council_tax_base",
            level="local_authority",
            spec_prefix=spec,
        ),
        "England's tax base",
    )


def epc_mean_floor_area(
    facts: Iterable[Mapping],
    *,
    spec_prefix: str,
    code_aliases: Mapping[str, str] | None = None,
    predecessors: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Mean floor area per authority: total floor area ÷ lodgements, all quarters.

    ``code_aliases`` folds a recoded authority onto its roster code, so a
    council that changed codes counts once in each quarter. ``predecessors``
    maps a council merged away to its successor, whose totals add up its
    predecessors' quarters.
    """
    facts = list(facts)
    aliases = dict(code_aliases or {})
    merged = dict(predecessors or {})
    totals: dict[str, list[float]] = {}
    seen: set[tuple[str, str, str]] = set()
    for index, concept in enumerate(
        ("mhclg.epc_domestic_total_floor_area", "mhclg.epc_domestic_lodgements")
    ):
        for fact in select_facts(
            facts, concept=concept, level="local_authority", spec_prefix=spec_prefix
        ):
            raw = str((fact.get("geography") or {}).get("id"))
            council = aliases.get(raw, raw)
            period = str((fact.get("period") or {}).get("value"))
            if (concept, council, period) in seen:
                continue  # the old and new code of one council in one quarter
            seen.add((concept, council, period))
            code = merged.get(council, council)
            totals.setdefault(code, [0.0, 0.0])[index] += fact_value(fact)
    return {
        code: area / lodgements
        for code, (area, lodgements) in sorted(totals.items())
        if lodgements > 0
    }


# ----------------------------------------------------------------------------
# Level inputs and the fingerprint


def resolve_level_input(
    name: str,
    *,
    feed_value: float | None,
    feed_key: str | None,
    parameter: float | None,
) -> dict[str, object]:
    """One ONS level input: the feed's fact, else a stated parameter, else absent.

    The status says which: a resource built on a parameter records it as a
    parameter, never as a fact. A feed input names its fact as
    ``aggregate_fact_key``, the field every package resource uses for a
    Chronicle fact's identity.
    """
    if feed_value is not None:
        if parameter is not None and not math.isclose(feed_value, parameter):
            raise ValueError(
                f"{name}: the feed carries {feed_value}, but the command line "
                f"states {parameter}; drop the parameter."
            )
        return {
            "name": name,
            "status": "feed",
            "value": feed_value,
            "aggregate_fact_key": feed_key,
        }
    if parameter is not None:
        return {"name": name, "status": "parameter", "value": float(parameter)}
    return {"name": name, "status": "absent", "value": None}


def feed_inputs_fingerprint(inputs: Mapping[str, object]) -> str:
    """The sha256 of everything the builder read from the feed, canonically."""
    return payload_sha256(inputs)
