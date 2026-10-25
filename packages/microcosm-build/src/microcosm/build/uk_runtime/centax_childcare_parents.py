"""CenTax's HMRC tabulations of parents of 3- and 4-year-olds, as derived facts.

Advani, Flew, Pepin-Hall and Summers (CenTax, September 2026), *Removing the
childcare cliff-edge*, tabulate HMRC's administrative data for tax year
2021-22 (their "2022"): the adjusted net income (ANI) of every identified
parent in England with a child age-eligible for the free hours in at least
one term of the year (Figure 3.1), split by main source of income (3.3), sex
(3.6) and region (3.7, pooled 2019-2022). The committed resource
``uk/centax_childcare_parent_facts.json`` turns those tabulations into nine
counts at England grain, built by ``tools/build_uk_centax_childcare_parent_facts.py``
from the workbook the authors released (its SHA-256 is pinned):

- four ANI bands, (0, £50k], (£50k, £80k], (£80k, £120k] and (£120k, £200k]
  in 2021-22 pounds, each Figure 3.1's share of the parents in (0, £200k];
- five splits of the parents in (£80k, £150k], the widest band every split
  figure covers: female (3.6), main source other than employment (3.3), and
  London, the South excluding London and the North and Midlands (3.7).

Microcosm targets are sums, so each share becomes a count: the share times
``parent_count``, the model's own number of such parents in (0, £200k] at the
calibration year, read off the baseline build named in the resource. The
targets set the distribution's shape and leave its level where the DfE
caseloads put it. Only the published tabulations are used. CenTax's model
outputs (bunching estimates, frictions, elasticities, marginal bunchers,
costs) never enter the resource.

The band edges are in 2021-22 pounds. The contract filters age them to the
calibration year by the OBR average earnings index policyengine-uk carries
(``ani_uprating``), rounded to whole pounds, and the loader refuses a contract
whose edges differ. ANI is the year's annual concept
(``adjusted_net_income_annual``): HMRC's data count what each parent received
in the tax year.

This module is the one place that knows the resource's shape and the two
measures the targets read: ``centax_parent_of_eligible_3_4_year_old`` (an
adult in a benefit unit with such a child, as policyengine-uk dates its
terms) and ``centax_employment_main_income_source`` (employment is the
largest income source in the year).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from microcosm.calibrate.geography_constants import UK_REGION_TIER_ENUM

__all__ = [
    "CENTAX_ANI_BANDS_2022",
    "CENTAX_ANI_MEASURE",
    "CENTAX_DATA_ENGINE_YEAR",
    "CENTAX_DATA_TAX_YEAR",
    "CENTAX_DERIVED_FACT_KEY_PREFIX",
    "CENTAX_ELIGIBLE_DFE_AGES",
    "CENTAX_EMPLOYEE_MAIN_SOURCE",
    "CENTAX_EMPLOYEE_SOURCE_ENGINE_NAMES",
    "CENTAX_EMPLOYEE_SOURCE_INPUTS",
    "CENTAX_FACTS_RESOURCE",
    "CENTAX_FILTER_CONCEPTS",
    "CENTAX_PARENT_ENGINE_INPUTS",
    "CENTAX_PARENT_MEASURES",
    "CENTAX_PARENT_OF_ELIGIBLE_3_4",
    "CENTAX_REGION_GROUPS",
    "CENTAX_REGION_GROUP_CODES",
    "CENTAX_REPORT_URL",
    "CENTAX_SPLIT_BAND_2022",
    "CENTAX_SPLITS",
    "CENTAX_WORKBOOK_SHA256",
    "UKCenTaxChildcareParentFacts",
    "centax_ani_uprating",
    "centax_cell_filters",
    "centax_cell_measurement_filters",
    "centax_cell_ids",
    "employment_main_income_source",
    "load_uk_centax_childcare_parent_facts",
    "model_ani_edges",
    "parent_of_eligible_3_4_year_old",
    "rows_sha256",
    "uk_centax_childcare_parent_facts",
]

CENTAX_FACTS_RESOURCE = "centax_childcare_parent_facts.json"
CENTAX_DERIVED_FACT_KEY_PREFIX = "microcosm.derived_fact.centax_childcare_parents.v1:"
RESOURCE_KIND = "uk_derived_consumer_facts"
ROW_SCHEMA_VERSION = "microcosm.derived_consumer_fact.v1"
#: The workbook the authors released with the report (Arun Advani to Vahid
#: Ahmadi, 21 October 2026): AdvaniFlewPepinhallSummers2026_ChildcareThreshold_data.xlsx.
CENTAX_WORKBOOK_SHA256 = (
    "8c23855745154a601064e766552c64bc2e88de51e043b4df1397c580e969354b"
)
CENTAX_REPORT_URL = (
    "https://centax.org.uk/wp-content/uploads/2026/09/"
    "AdvaniFlewPepin-HallSummers2026_Removing-the-childcare-cliff-edge.pdf"
)
#: Tax year of the HMRC data behind Figures 3.1, 3.3 and 3.6; policyengine-uk
#: dates it 2021, the year the tax year starts in.
CENTAX_DATA_TAX_YEAR = "2021-22"
CENTAX_DATA_ENGINE_YEAR = 2021

#: ANI bands in 2021-22 pounds, lower edge exclusive and upper edge inclusive,
#: as the workbook's bins are labelled by their upper edge. Wide around £100k
#: on purpose (see the contract's selector notes).
CENTAX_ANI_BANDS_2022: tuple[tuple[str, float, float], ...] = (
    ("ani_0_to_50k", 0.0, 50_000.0),
    ("ani_50k_to_80k", 50_000.0, 80_000.0),
    ("ani_80k_to_120k", 80_000.0, 120_000.0),
    ("ani_120k_to_200k", 120_000.0, 200_000.0),
)
#: The band every split figure covers (Figures 3.3, 3.6 and 3.7 stop at £150k).
CENTAX_SPLIT_BAND_2022 = (80_000.0, 150_000.0)
#: Region groups of Figure 3.7, by English region GSS code. The figure does
#: not list its regions; the East of England sits in the South because only
#: then do the figure's low-income shares match the ONS child-population shares
#: of the three groups.
CENTAX_REGION_GROUP_CODES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "london": ("E12000007",),
        "south_excl_london": ("E12000006", "E12000008", "E12000009"),
        "north_and_midlands": (
            "E12000001",
            "E12000002",
            "E12000003",
            "E12000004",
            "E12000005",
        ),
    }
)
#: The same groups by the spine's ``region`` enum name, the value the
#: contract filters compare against.
CENTAX_REGION_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        group: tuple(UK_REGION_TIER_ENUM[code] for code in codes)
        for group, codes in CENTAX_REGION_GROUP_CODES.items()
    }
)
#: The split cells, each with the workbook figure it reads.
CENTAX_SPLITS: tuple[tuple[str, str], ...] = (
    ("female", "Fig. 3.6"),
    ("non_employee", "Fig. 3.3"),
    ("london", "Fig. 3.7"),
    ("south_excl_london", "Fig. 3.7"),
    ("north_and_midlands", "Fig. 3.7"),
)

CENTAX_PARENT_OF_ELIGIBLE_3_4 = "centax_parent_of_eligible_3_4_year_old"
CENTAX_EMPLOYEE_MAIN_SOURCE = "centax_employment_main_income_source"
CENTAX_PARENT_MEASURES = (CENTAX_PARENT_OF_ELIGIBLE_3_4, CENTAX_EMPLOYEE_MAIN_SOURCE)
#: Engine variables the parent measure reads at person grain. The spine sets
#: ``is_hbai_dependent_child`` for the FRS child-table members of each benefit
#: unit, so the others are its adults, the claimant and any partner; SPI
#: cloning keeps the roles.
CENTAX_PARENT_ENGINE_INPUTS = (
    "birth_year",
    "country",
    "early_years_term_share_before_school",
    "is_hbai_dependent_child",
)
#: The year's income sources the main-source measure compares, employment
#: first: every component of the engine's ``total_income`` (ITA 2007 s. 23),
#: with ``social_security_income`` taken benefit by benefit. The first four
#: are the annual measures. The investment, property and miscellaneous incomes
#: are annual amounts already. The taxable benefits other than the State
#: Pension are read as the engine computes them for the year.
CENTAX_EMPLOYEE_SOURCE_INPUTS = (
    "employment_income_annual",
    "self_employment_income_after_loss_relief_annual",
    "private_pension_income_annual",
    "state_pension_annual",
    "property_income",
    "dividend_income",
    "savings_interest_income",
    "other_investment_income",
    "taxable_miscellaneous_income",
    "incapacity_benefit",
    "jsa_contrib_reported",
    "esa_contrib_reported",
    "carers_allowance",
    "carer_support_payment",
    "scottish_carer_supplement",
    "widowed_parents_allowance",
)
#: The engine variable each compared source stands for, where the name
#: differs: the annual measures and the loss-relieved self-employment income.
CENTAX_EMPLOYEE_SOURCE_ENGINE_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "employment_income_annual": "employment_income",
        "self_employment_income_after_loss_relief_annual": "self_employment_income",
        "private_pension_income_annual": "private_pension_income",
        "state_pension_annual": "state_pension",
    }
)
#: DfE ages (``Y - birth_year``, the age on 31 December of model year Y) of a
#: child age-eligible for the universal hours in at least one of the year's
#: terms (1 April, 1 September, 1 January). The hours start the term after the
#: third birthday, so a DfE age 2 child is never eligible before April Y + 1;
#: every DfE age 3 and 4 child is; a DfE age 5 child is only while it has not
#: started school, which ``early_years_term_share_before_school`` dates.
CENTAX_ELIGIBLE_DFE_AGES = (3, 4, 5)
#: ``early_years_term_share_before_school`` is 0, 1/3 or 1 in float32, so it
#: is tested against a sixth, never for equality.
_TERM_SHARE_FLOOR = 1.0 / 6.0
_ENGLAND_GSS = "E92000001"


def parent_of_eligible_3_4_year_old(
    *,
    birth_year: np.ndarray,
    country: np.ndarray,
    term_share_before_school: np.ndarray,
    dependent_child: np.ndarray,
    benunit_id: np.ndarray,
    year: int,
) -> np.ndarray:
    """One for an adult whose benefit unit has an eligible child in England.

    A child is eligible when it is a dependent child of its benefit unit in
    England, its DfE age is 3, 4 or 5 and it is before school in at least one
    of the year's terms. The parent is any adult of the child's benefit unit,
    the FRS's family: CenTax's identified parents and partners. Every array
    is at person grain.
    """

    dfe_age = float(year) - np.asarray(birth_year, dtype=float)
    child = np.asarray(dependent_child).astype(bool)
    eligible = (
        child
        & (np.asarray(country).astype(str) == "ENGLAND")
        & np.isin(dfe_age, CENTAX_ELIGIBLE_DFE_AGES)
        & (np.asarray(term_share_before_school, dtype=float) > _TERM_SHARE_FLOOR)
    )
    units = np.asarray(benunit_id)
    with_child = np.isin(units, units[eligible])
    return (with_child & ~child).astype(float)


def employment_main_income_source(sources: Mapping[str, np.ndarray]) -> np.ndarray:
    """One where employment income is positive and at least every other source.

    ``sources`` maps each name in ``CENTAX_EMPLOYEE_SOURCE_INPUTS`` to a person
    array. CenTax's employees are taxpayers whose main source of income is
    employment; a tie goes to employment.
    """

    missing = [name for name in CENTAX_EMPLOYEE_SOURCE_INPUTS if name not in sources]
    if missing:
        raise ValueError(f"the main-source measure needs {missing}.")
    employment = np.asarray(sources[CENTAX_EMPLOYEE_SOURCE_INPUTS[0]], dtype=float)
    others = np.vstack(
        [
            np.asarray(sources[name], dtype=float)
            for name in CENTAX_EMPLOYEE_SOURCE_INPUTS[1:]
        ]
    )
    return ((employment > 0) & (employment >= others.max(axis=0))).astype(float)


def centax_ani_uprating(calibration_year: int) -> float:
    """The OBR average earnings index at ``calibration_year`` over 2021.

    Read from the installed policyengine-uk, which carries the index the
    engine uprates earnings by; 2021 is the engine's year of tax year 2021-22.
    """

    from policyengine_uk import CountryTaxBenefitSystem

    index = CountryTaxBenefitSystem().parameters.gov.economic_assumptions.indices.obr.average_earnings
    return float(index(f"{int(calibration_year)}-01-01")) / float(
        index(f"{CENTAX_DATA_ENGINE_YEAR}-01-01")
    )


def centax_cell_ids() -> tuple[str, ...]:
    """Every cell of the resource: the four bands, then the five splits."""

    return tuple(band for band, _, _ in CENTAX_ANI_BANDS_2022) + tuple(
        f"{split}_ani_80k_to_150k" for split, _ in CENTAX_SPLITS
    )


#: The annual ANI measure every cell bands on.
CENTAX_ANI_MEASURE = "adjusted_net_income_annual"
#: Each split's extra predicate, in the contract's binding filter vocabulary.
_SPLIT_FILTERS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "female": {"variable": "gender", "equals": "FEMALE"},
        "non_employee": {"variable": CENTAX_EMPLOYEE_MAIN_SOURCE, "equals": 0},
        **{
            group: {"variable": "region", "in": list(regions)}
            for group, regions in CENTAX_REGION_GROUPS.items()
        },
    }
)


def centax_cell_filters(cell_id: str, edges: Sequence[int]) -> list[dict[str, Any]]:
    """The policyengine binding filters of a cell at its model ANI edges.

    The parent flag, then ANI above the lower edge and at most the upper one,
    then the split's own predicate. The contract carries exactly these, which
    the tests check.
    """

    lower, upper = (int(edge) for edge in edges)
    filters: list[dict[str, Any]] = [
        {"variable": CENTAX_PARENT_OF_ELIGIBLE_3_4, "equals": 1},
        {"variable": CENTAX_ANI_MEASURE, "operator": ">", "value": lower},
        {"variable": CENTAX_ANI_MEASURE, "operator": "<=", "value": upper},
    ]
    split = cell_id.removesuffix("_ani_80k_to_150k")
    if split != cell_id:
        filters.append(dict(_SPLIT_FILTERS[split]))
    elif cell_id not in {band for band, _, _ in CENTAX_ANI_BANDS_2022}:
        raise KeyError(cell_id)
    return filters


#: The contract's measurement concept for each variable a cell filters on.
CENTAX_FILTER_CONCEPTS: Mapping[str, str] = MappingProxyType(
    {
        CENTAX_PARENT_OF_ELIGIBLE_3_4: (
            "uk.childcare.centax_parent_of_eligible_3_4_year_old"
        ),
        CENTAX_ANI_MEASURE: "uk.income.adjusted_net.amount",
        CENTAX_EMPLOYEE_MAIN_SOURCE: "uk.income.employment.main_source",
        "gender": "uk.demographics.gender",
        "region": "uk.geography.region",
    }
)


def centax_cell_measurement_filters(
    cell_id: str, edges: Sequence[int]
) -> list[dict[str, Any]]:
    """The cell's filters in the contract's measurement (concept) vocabulary."""

    filters = []
    for predicate in centax_cell_filters(cell_id, edges):
        concept = dict(predicate)
        concept = {
            "concept": CENTAX_FILTER_CONCEPTS[concept.pop("variable")],
            **concept,
        }
        filters.append(concept)
    return filters


@dataclass(frozen=True)
class UKCenTaxChildcareParentFacts:
    """The validated derived-facts resource."""

    rows: tuple[Mapping[str, Any], ...]
    rows_sha256: str
    inputs_sha256: str
    parameters: Mapping[str, Any]
    inputs: Mapping[str, Any]
    generator: Mapping[str, Any]
    shares: Mapping[str, float]

    @property
    def ani_uprating(self) -> float:
        return float(self.parameters["ani_uprating"])

    @property
    def parent_count(self) -> float:
        return float(self.parameters["parent_count"])

    def model_edges(self, cell_id: str) -> tuple[int, int]:
        """A cell's ANI edges at the calibration year, in whole pounds."""

        return tuple(self.parameters["model_ani_edges"][cell_id])  # type: ignore[return-value]

    def value(self, cell_id: str) -> float:
        for row in self.rows:
            if row["dimensions"]["microcosm.centax_cell"] == cell_id:
                return float(row["value"])
        raise KeyError(cell_id)


def _package_text(name: str) -> str:
    return (
        resources.files("microcosm.build.uk").joinpath(name).read_text(encoding="utf-8")
    )


def rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Digest of the rows, independent of their order."""

    digest = hashlib.sha256()
    for line in sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in rows
    ):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def model_ani_edges(uprating: float) -> dict[str, list[int]]:
    """Each cell's 2021-22 edges aged by ``uprating`` and rounded to whole pounds."""

    edges: dict[str, list[int]] = {}
    for band, lower, upper in CENTAX_ANI_BANDS_2022:
        edges[band] = [round(lower * uprating), round(upper * uprating)]
    split_lower, split_upper = CENTAX_SPLIT_BAND_2022
    for split, _ in CENTAX_SPLITS:
        edges[f"{split}_ani_80k_to_150k"] = [
            round(split_lower * uprating),
            round(split_upper * uprating),
        ]
    return edges


def _validate_row(row: Mapping[str, Any]) -> str:
    key = str(row.get("aggregate_fact_key") or "")
    if not key.startswith(CENTAX_DERIVED_FACT_KEY_PREFIX):
        raise ValueError(f"CenTax fact {key!r} is outside the derived-fact namespace.")
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError(
            f"CenTax fact {key!r} has schema {row.get('schema_version')!r}."
        )
    geography = row.get("geography") or {}
    if geography.get("level") != "country" or geography.get("id") != _ENGLAND_GSS:
        raise ValueError(f"CenTax fact {key!r} must sit at England (country) grain.")
    value = row.get("value")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"CenTax fact {key!r} has invalid value {value!r}.")
    if (row.get("aggregation") or {}).get("method") != "sum":
        raise ValueError(f"CenTax fact {key!r} must aggregate by sum.")
    if (row.get("source") or {}).get("source_name") != "microcosm":
        raise ValueError(f"CenTax fact {key!r} must name source microcosm.")
    cell = str((row.get("dimensions") or {}).get("microcosm.centax_cell") or "")
    if cell not in centax_cell_ids():
        raise ValueError(f"CenTax fact {key!r} names an unknown cell {cell!r}.")
    spec = str((row.get("layout") or {}).get("record_set_spec_id") or "")
    if spec != f"uk.centax_childcare_parents.{cell}.v1":
        raise ValueError(f"CenTax fact {key!r} has record set spec {spec!r}.")
    if not str(row.get("label") or "").strip():
        raise ValueError(f"CenTax fact {key!r} requires a label.")
    return cell


def load_uk_centax_childcare_parent_facts(
    path: Path | None = None,
) -> UKCenTaxChildcareParentFacts:
    """Load and validate the derived-facts resource.

    Refused when the rows differ from their digest, the workbook is not the
    pinned one, a cell is missing or repeated, a band share falls outside
    (0, 1) or the bands' shares do not sum to one, a value is not its share
    times the parent count, or the recorded edges are not the 2021-22 edges
    aged by the recorded uprating.
    """

    text = (
        path.read_text(encoding="utf-8")
        if path is not None
        else _package_text(CENTAX_FACTS_RESOURCE)
    )
    payload = json.loads(text)
    if payload.get("kind") != RESOURCE_KIND or payload.get("schema_version") != 1:
        raise ValueError(
            f"{CENTAX_FACTS_RESOURCE} is not a schema-1 derived-facts resource."
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{CENTAX_FACTS_RESOURCE} carries no rows.")
    if payload.get("row_count") != len(rows):
        raise ValueError(f"{CENTAX_FACTS_RESOURCE} row_count disagrees with its rows.")
    digest = rows_sha256(rows)
    if digest != payload.get("rows_sha256"):
        raise ValueError(f"{CENTAX_FACTS_RESOURCE} rows differ from their digest.")
    inputs = payload.get("inputs") or {}
    if (inputs.get("workbook") or {}).get("sha256") != CENTAX_WORKBOOK_SHA256:
        raise ValueError(
            f"{CENTAX_FACTS_RESOURCE} was not built from the pinned CenTax workbook."
        )
    parameters = payload.get("parameters") or {}
    for required in (
        "ani_uprating",
        "calibration_year",
        "model_ani_edges",
        "parent_count",
        "shares",
    ):
        if required not in parameters:
            raise ValueError(f"{CENTAX_FACTS_RESOURCE} lacks parameter {required!r}.")
    uprating = float(parameters["ani_uprating"])
    if not 1.0 < uprating < 2.0:
        raise ValueError(f"implausible ANI uprating {uprating!r}.")
    if parameters["model_ani_edges"] != model_ani_edges(uprating):
        raise ValueError(
            f"{CENTAX_FACTS_RESOURCE} edges are not the 2021-22 edges aged by "
            f"{uprating!r}."
        )
    parent_count = float(parameters["parent_count"])
    if not parent_count > 0:
        raise ValueError("the parent count must be positive.")
    shares = {str(cell): float(share) for cell, share in parameters["shares"].items()}
    if set(shares) != set(centax_cell_ids()):
        raise ValueError(f"{CENTAX_FACTS_RESOURCE} shares do not cover every cell.")
    if any(not 0.0 < share < 1.0 for share in shares.values()):
        raise ValueError(f"{CENTAX_FACTS_RESOURCE} has a share outside (0, 1).")
    band_total = sum(shares[band] for band, _, _ in CENTAX_ANI_BANDS_2022)
    if abs(band_total - 1.0) > 1e-9:
        raise ValueError(f"the ANI band shares sum to {band_total!r}, not 1.")
    region_total = sum(
        shares[f"{group}_ani_80k_to_150k"] for group in CENTAX_REGION_GROUPS
    )
    split_share = float(parameters.get("split_band_share", 0.0))
    if not 0.0 < split_share < 1.0 or abs(region_total - split_share) > 1e-9:
        raise ValueError(
            "the region cells must partition the (£80k, £150k] parents: "
            f"{region_total!r} vs {split_share!r}."
        )
    seen: set[str] = set()
    for row in rows:
        cell = _validate_row(row)
        if cell in seen:
            raise ValueError(f"CenTax cell {cell!r} is repeated.")
        seen.add(cell)
        expected = round(shares[cell] * parent_count, 3)
        if abs(float(row["value"]) - expected) > 1e-6:
            raise ValueError(
                f"CenTax cell {cell!r} value {row['value']!r} is not its share "
                f"times the parent count ({expected!r})."
            )
    if seen != set(centax_cell_ids()):
        raise ValueError(
            f"{CENTAX_FACTS_RESOURCE} lacks cells {sorted(set(centax_cell_ids()) - seen)}."
        )
    return UKCenTaxChildcareParentFacts(
        rows=tuple(rows),
        rows_sha256=digest,
        inputs_sha256=str(payload.get("inputs_sha256") or ""),
        parameters=parameters,
        inputs=inputs,
        generator=payload.get("generator") or {},
        shares=shares,
    )


@cache
def uk_centax_childcare_parent_facts() -> UKCenTaxChildcareParentFacts:
    """The committed resource, loaded once."""

    return load_uk_centax_childcare_parent_facts()
