"""CenTax's HMRC tabulations of parents of 3- and 4-year-olds, as derived facts.

Advani, Flew, Pepin-Hall and Summers (CenTax, September 2026), *Removing the
£100,000 childcare cliff-edge*, tabulate HMRC's administrative data for tax
year 2021-22 (their "2022"): the adjusted net income (ANI) of every identified
parent in England with a child age-eligible for the free hours in at least
one term of the year (Figure 3.1), split by main source of income (3.3), sex
(3.6) and region (3.7, pooled 2018-19 to 2021-22). Their data for the charts
is registered raw-only in Chronicle as ``centax-childcare-cliff-edge-2026``;
``tools/build_uk_centax_childcare_parent_facts.py`` checks the workbook's bytes
against that registration and writes the committed resource
``uk/centax_childcare_parent_facts.json``: five counts at England grain.

- Three ANI bands, (£50k, £80k], (£80k, £120k] and (£120k, £200k] in 2021-22
  pounds, each Figure 3.1's share of the parents in (£50k, £200k].
- Two splits of the parents in (£80k, £150k], the widest band every split
  figure covers: female (3.6) and London (3.7).

Microcosm targets are sums, so each share becomes a count: the share times
``parent_count``, the baseline build's own parents with ANI in the aged
(£50k, £200k] at the calibration year. The targets set the shape of the
distribution above £50k and leave its level where the baseline put it.

Three cells of the first design are diagnostics, reported by
``tools/diagnose_uk_centax_parents.py`` and never bound (Vahid's reviews of
microcosm#1399):

- ANI up to £50k. CenTax counts taxpayers that HMRC's PAYE and Self Assessment
  data hold. A model adult whose only income is savings interest or a taxable
  benefit has positive ANI but no such record, and almost all of them sit in
  that band, so a share of (0, £200k] times the model's count of it would
  scale every cell up with them. Above £50k a parent is in HMRC's data.
- The South excluding London and the North and Midlands. Figure 3.7 does not
  list its regions; the East of England's place in the South is inferred
  (``CENTAX_REGION_GROUP_CODES``). London binds; the diagnostic reports the
  other two groups under both placements of the East.
- Main source other than employment (Figure 3.3). The cell rests on fewer than
  3 FRS source households, too few to bind (María's ruling of 24 November
  2026).

Only the published tabulations are used. CenTax's model outputs (bunching
estimates, frictions, elasticities, marginal bunchers, costs) never enter the
resource, and neither do the figures kept as diagnostics (Table A.1's levels,
Figure 3.12's non-working partners): the diagnostic reads those from the
registered workbook.

The band edges are in 2021-22 pounds. The contract filters age them to the
calibration year by the OBR average earnings index policyengine-uk carries
(``ani_uprating``), rounded to whole pounds, and the loader refuses a contract
whose edges differ. ANI is the year's annual concept
(``adjusted_net_income_annual``): HMRC's data count what each parent received
in the tax year.

This module is the one place that knows the resource's shape and the two
measures the cells read: ``centax_parent_of_eligible_3_4_year_old`` (an
adult in a benefit unit with such a child, as policyengine-uk dates its
terms) and ``centax_employment_main_income_source`` (employment is the
largest income source in the year), which only the diagnostic's
non-employee cell reads.
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
    "CENTAX_BOUND_REGION_GROUPS",
    "CENTAX_CHRONICLE_PACKAGE_ID",
    "CENTAX_CHRONICLE_SOURCE_ID",
    "CENTAX_DATA_ENGINE_YEAR",
    "CENTAX_DATA_TAX_YEAR",
    "CENTAX_DERIVED_FACT_KEY_PREFIX",
    "CENTAX_DIAGNOSTIC_BAND_2022",
    "CENTAX_DIAGNOSTIC_SPLITS",
    "CENTAX_EAST_OF_ENGLAND",
    "CENTAX_ELIGIBLE_DFE_AGES",
    "CENTAX_EMPLOYEE_MAIN_SOURCE",
    "CENTAX_EMPLOYEE_SOURCE_COMPONENTS",
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
    "CENTAX_SHARE_BASE_2022",
    "CENTAX_SPLIT_BAND_2022",
    "CENTAX_SPLITS",
    "CENTAX_WORKBOOK_FILENAME",
    "CENTAX_WORKBOOK_SHA256",
    "RESOURCE_SCHEMA_VERSION",
    "UKCenTaxChildcareParentFacts",
    "aged_edges",
    "centax_ani_uprating",
    "centax_cell_filters",
    "centax_cell_ids",
    "centax_cell_measurement_filters",
    "centax_diagnostic_cell_ids",
    "centax_region_codes",
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
#: Schema 2 (microcosm#1399's review): the shares are taken within the share
#: base, the workbook is named by its Chronicle registration, and the
#: diagnostics-only figures left the resource.
RESOURCE_SCHEMA_VERSION = 2
ROW_SCHEMA_VERSION = "microcosm.derived_consumer_fact.v1"
#: The Chronicle registration of CenTax's data for the charts (raw only).
CENTAX_CHRONICLE_SOURCE_ID = "centax"
CENTAX_CHRONICLE_PACKAGE_ID = "centax-childcare-cliff-edge-2026"
CENTAX_WORKBOOK_FILENAME = "AdvaniFlewPepinhallSummers2026_ChildcareThreshold_data.xlsx"
#: The workbook's bytes, as the registration pins them (the copy CenTax shared
#: on 21 October 2026 and the one arunadvani.com publishes are the same).
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

#: The 2021-22 ANI range the shares are taken within, lower edge exclusive.
#: Above £50k a parent is in HMRC's PAYE or Self Assessment data, so the
#: model's parents there are the ones CenTax can count (see the module notes).
CENTAX_SHARE_BASE_2022 = (50_000.0, 200_000.0)
#: The bound ANI bands in 2021-22 pounds, lower edge exclusive and upper edge
#: inclusive, as the workbook's bins are labelled by their upper edge. They
#: partition the share base. Wide around £100k on purpose (see the contract's
#: selector notes).
CENTAX_ANI_BANDS_2022: tuple[tuple[str, float, float], ...] = (
    ("ani_50k_to_80k", 50_000.0, 80_000.0),
    ("ani_80k_to_120k", 80_000.0, 120_000.0),
    ("ani_120k_to_200k", 120_000.0, 200_000.0),
)
#: The band below the share base: a diagnostic, never a target.
CENTAX_DIAGNOSTIC_BAND_2022 = ("ani_0_to_50k", 0.0, 50_000.0)
#: The band every split figure covers (Figures 3.3, 3.6 and 3.7 stop at £150k).
CENTAX_SPLIT_BAND_2022 = (80_000.0, 150_000.0)
#: Region groups of Figure 3.7, by English region GSS code. The figure does
#: not list its regions; the East of England sits in the South because only
#: then do the figure's low-income shares match the ONS child-population shares
#: of the three groups. That placement is inferred, so only London binds.
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
#: The region whose group the figure leaves open.
CENTAX_EAST_OF_ENGLAND = "E12000006"
#: The same groups by the spine's ``region`` enum name, the value the
#: contract filters compare against.
CENTAX_REGION_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        group: tuple(UK_REGION_TIER_ENUM[code] for code in codes)
        for group, codes in CENTAX_REGION_GROUP_CODES.items()
    }
)
#: The region groups the targets bind: London, the one group the figure names
#: without doubt (Vahid's review, item 3).
CENTAX_BOUND_REGION_GROUPS: tuple[str, ...] = ("london",)
#: The split cells the targets bind, each with the workbook figure it reads.
CENTAX_SPLITS: tuple[tuple[str, str], ...] = (
    ("female", "Fig. 3.6"),
    ("london", "Fig. 3.7"),
)
#: The split the diagnostic reports and the targets never bind: its cell rests
#: on fewer than 3 FRS source households (María's ruling of 24 November 2026).
CENTAX_DIAGNOSTIC_SPLITS: tuple[tuple[str, str], ...] = (("non_employee", "Fig. 3.3"),)

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
#: Each component of the engine's adjusted net income
#: (``gov.hmrc.income_tax.adjusted_net_income_components``, the sources ITA
#: 2007 s. 23 totals) and the sources the main-source measure compares for it:
#: the social security income benefit by benefit, and savings interest apart
#: from other investment income. Employment is pay plus the company car and
#: car fuel benefits, which HMRC's employment income counts and
#: policyengine-uk keeps apart from pay since 2.165.0, as the HMRC rows on
#: employment income read it (PolicyEngine/uk-autumn-budget-dashboard-2026#316).
CENTAX_EMPLOYEE_SOURCE_COMPONENTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "taxable_employment_income": (
            "employment_income_with_benefits_in_kind_annual",
        ),
        "taxable_self_employment_income": (
            "self_employment_income_after_loss_relief_annual",
        ),
        "taxable_pension_income": ("private_pension_income_annual",),
        "taxable_social_security_income": (
            "state_pension_annual",
            "incapacity_benefit",
            "jsa_contrib_reported",
            "esa_contrib_reported",
            "carers_allowance",
            "carer_support_payment",
            "scottish_carer_supplement",
            "widowed_parents_allowance",
        ),
        "taxable_property_income": ("property_income",),
        "taxable_savings_interest_income": (
            "savings_interest_income",
            "other_investment_income",
        ),
        "taxable_dividend_income": ("dividend_income",),
        "taxable_miscellaneous_income": ("taxable_miscellaneous_income",),
    }
)
#: The year's income sources the main-source measure compares, employment
#: first. The first four are the annual measures. The investment, property and
#: miscellaneous incomes are annual amounts already. The taxable benefits other
#: than the State Pension are read as the engine computes them for the year.
CENTAX_EMPLOYEE_SOURCE_INPUTS: tuple[str, ...] = tuple(
    name for sources in CENTAX_EMPLOYEE_SOURCE_COMPONENTS.values() for name in sources
)
#: The engine variable each compared social security source stands for,
#: where the name differs: the State Pension's annual measure.
CENTAX_EMPLOYEE_SOURCE_ENGINE_NAMES: Mapping[str, str] = MappingProxyType(
    {"state_pension_annual": "state_pension"}
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
    employment (Figure 3.3's notes). The report does not say whether that is
    the largest single source or more than half of income; this reads the
    largest single source, and a tie goes to employment.
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
    """Every bound cell of the resource: the three bands, then the bound splits."""

    return tuple(band for band, _, _ in CENTAX_ANI_BANDS_2022) + tuple(
        f"{split}_ani_80k_to_150k" for split, _ in CENTAX_SPLITS
    )


def centax_diagnostic_cell_ids() -> tuple[str, ...]:
    """The split cells the diagnostic reports and the targets never bind."""

    return tuple(f"{split}_ani_80k_to_150k" for split, _ in CENTAX_DIAGNOSTIC_SPLITS)


def centax_region_codes(
    *, east_in: str = "south_excl_london"
) -> dict[str, tuple[str, ...]]:
    """Figure 3.7's region groups with the East of England in ``east_in``.

    ``south_excl_london`` is the inferred placement the module binds against;
    ``north_and_midlands`` is the other one, which the diagnostic also reports.
    London never moves.
    """

    if east_in not in {"south_excl_london", "north_and_midlands"}:
        raise ValueError(f"the East of England cannot sit in {east_in!r}.")
    groups = {
        group: tuple(code for code in codes if code != CENTAX_EAST_OF_ENGLAND)
        for group, codes in CENTAX_REGION_GROUP_CODES.items()
    }
    groups[east_in] = tuple(sorted((*groups[east_in], CENTAX_EAST_OF_ENGLAND)))
    return groups


#: The annual ANI measure every cell bands on.
CENTAX_ANI_MEASURE = "adjusted_net_income_annual"
#: Each split's extra predicate, in the contract's binding filter vocabulary.
_SPLIT_FILTERS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "female": {"variable": "gender", "equals": "FEMALE"},
        "non_employee": {"variable": CENTAX_EMPLOYEE_MAIN_SOURCE, "equals": 0},
        **{
            group: {"variable": "region", "in": list(CENTAX_REGION_GROUPS[group])}
            for group in CENTAX_BOUND_REGION_GROUPS
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

    @property
    def baseline(self) -> Mapping[str, Any]:
        """The national build the parent count was read off."""

        return self.inputs["baseline"]

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


def aged_edges(lower: float, upper: float, uprating: float) -> list[int]:
    """A 2021-22 band's edges aged by ``uprating``, rounded to whole pounds."""

    return [round(lower * uprating), round(upper * uprating)]


def model_ani_edges(uprating: float) -> dict[str, list[int]]:
    """Each bound cell's 2021-22 edges aged by ``uprating``."""

    edges: dict[str, list[int]] = {}
    for band, lower, upper in CENTAX_ANI_BANDS_2022:
        edges[band] = aged_edges(lower, upper, uprating)
    for split, _ in CENTAX_SPLITS:
        edges[f"{split}_ani_80k_to_150k"] = aged_edges(
            *CENTAX_SPLIT_BAND_2022, uprating
        )
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


def _validate_registration(workbook: Mapping[str, Any]) -> None:
    """The workbook must be the one Chronicle registers, with its pinned bytes."""

    if workbook.get("chronicle_package_id") != CENTAX_CHRONICLE_PACKAGE_ID or (
        workbook.get("chronicle_source_id") != CENTAX_CHRONICLE_SOURCE_ID
    ):
        raise ValueError(
            f"{CENTAX_FACTS_RESOURCE} must name the Chronicle registration "
            f"{CENTAX_CHRONICLE_PACKAGE_ID}."
        )
    if workbook.get("registration_kind") != "raw_only":
        raise ValueError(f"{CENTAX_CHRONICLE_PACKAGE_ID} is a raw-only registration.")
    files = list((workbook.get("files") or {}).values())
    if (
        len(files) != 1
        or files[0].get("filename") != CENTAX_WORKBOOK_FILENAME
        or files[0].get("sha256") != CENTAX_WORKBOOK_SHA256
    ):
        raise ValueError(
            f"{CENTAX_FACTS_RESOURCE} was not built from the pinned CenTax workbook."
        )


def load_uk_centax_childcare_parent_facts(
    path: Path | None = None,
) -> UKCenTaxChildcareParentFacts:
    """Load and validate the derived-facts resource.

    Refused when the rows differ from their digest, the workbook is not the
    one Chronicle registers with the pinned bytes, a cell is missing or
    repeated, a share falls outside (0, 1), the bands' shares do not sum to
    one or a split's exceeds the split band's, a value is not its share times
    the parent count, or the recorded edges are not the 2021-22 edges aged by
    the recorded uprating.
    """

    text = (
        path.read_text(encoding="utf-8")
        if path is not None
        else _package_text(CENTAX_FACTS_RESOURCE)
    )
    payload = json.loads(text)
    if (
        payload.get("kind") != RESOURCE_KIND
        or payload.get("schema_version") != RESOURCE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{CENTAX_FACTS_RESOURCE} is not a schema-{RESOURCE_SCHEMA_VERSION} "
            "derived-facts resource."
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
    _validate_registration(inputs.get("workbook") or {})
    baseline = inputs.get("baseline") or {}
    if not baseline.get("dataset") or len(str(baseline.get("sha256") or "")) != 64:
        raise ValueError(f"{CENTAX_FACTS_RESOURCE} must name its baseline build.")
    parameters = payload.get("parameters") or {}
    for required in (
        "ani_uprating",
        "calibration_year",
        "model_ani_edges",
        "parent_count",
        "shares",
        "share_base_2022",
        "split_band_share",
    ):
        if required not in parameters:
            raise ValueError(f"{CENTAX_FACTS_RESOURCE} lacks parameter {required!r}.")
    if parameters["share_base_2022"] != list(CENTAX_SHARE_BASE_2022):
        raise ValueError(
            f"{CENTAX_FACTS_RESOURCE} shares are not taken within "
            f"{CENTAX_SHARE_BASE_2022}."
        )
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
    split_share = float(parameters["split_band_share"])
    if not 0.0 < split_share < 1.0:
        raise ValueError(f"implausible split band share {split_share!r}.")
    for split, _ in CENTAX_SPLITS:
        if shares[f"{split}_ani_80k_to_150k"] >= split_share:
            raise ValueError(
                f"the {split} share exceeds the (£80k, £150k] parents' share."
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
