"""Labels chosen for identifiers that a country's sources spell differently.

Calibration diagnostics schema 9 declares each geography and dimension-value
label once, so the writer refuses target rows that label one identifier two
ways (``microcosm.diagnostics.catalog_calibration_diagnostics``), unless a
label is chosen here. A choice applies only where the rows disagree, so an
identifier that every row labels one way keeps that label.

The choices cover the spellings found in published releases:

- UK: Chronicle's local facts carry their area as a dimension named
  ``geography`` whose value IDs are lower-case GSS codes, and each publisher
  names the area its own way (HMRC "Hartlepool UA", ONS "Hartlepool"; 88 areas
  in the October 2026 dense release, and "Yorkshire And The Humber" beside
  "Yorkshire and The Humber" in the national releases). Such a value takes
  its area's label: the payload's own geography label for that code, else
  ``UK_GEOGRAPHY_ID_TO_LABEL``. The UK build's writer adds the ONS
  local-authority names (``microcosm.build.uk_runtime.diagnostics``).
- US: IRS SOI and Census tables word three dimension values differently
  (release ``populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z``, built
  from the pinned feed at Chronicle ``51aa40f``, which still produces them).
  Chronicle registers the two spellings of each as reviewed variants
  (``REVIEWED_VALUE_LABEL_VARIANTS``, chronicle#377). One label for each was
  chosen in review on 2026-10-21; a US feed re-pin brings the same labels from
  Chronicle, so releases built after it carry one spelling and these choices
  then apply only to older feeds. Each choice covers only the two reviewed
  spellings, so a third spelling is still refused.
- Belgium: each Belgian Chronicle package states its national figure as the
  group-by value ``measure`` = ``country_total`` and labels it with its own
  measure name (SPF Finances, ONSS, ONEM and NBB; Chronicle registers the four
  as reviewed variants, chronicle#265). The value is written as "Country
  total", the label chosen in review on 2026-10-22 and the one Chronicle's
  canonical register gives it (chronicle#377). Without it the transport
  graph's diagnostics step would raise and stop a Belgian build. The choice
  covers only the four package spellings, so another is still refused.

Every disagreement a choice settles is logged as a warning.

Any other spelling difference still stops the writer; correct the source or
add a reviewed choice here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from microcosm.calibrate.geography_constants import UK_GEOGRAPHY_ID_TO_LABEL
from microcosm.diagnostics import HierarchyLabelChoices

__all__ = [
    "BE_DIMENSION_VALUE_LABEL_CHOICES",
    "BE_DIMENSION_VALUE_REVIEWED_SPELLINGS",
    "BE_HIERARCHY_LABEL_CHOICES",
    "UK_HIERARCHY_LABEL_CHOICES",
    "US_DIMENSION_VALUE_LABEL_CHOICES",
    "US_DIMENSION_VALUE_REVIEWED_SPELLINGS",
    "US_HIERARCHY_LABEL_CHOICES",
    "hierarchy_label_choices",
]

_ADJUSTED_GROSS_INCOME = "us:statutes/26/62#adjusted_gross_income"

#: ``(dimension ID, value ID)`` -> label, for the US values two sources word
#: differently. Counts are targets in release 0fb05b6.
US_DIMENSION_VALUE_LABEL_CHOICES: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        # Seven SOI packages, including Table 1.1 (106 targets); the
        # filing-season week-47 EITC table's column header adds "and deficit"
        # (2 targets). chronicle#377 changes that package's label to this one;
        # the variant leaves the feed at its next re-pin.
        (_ADJUSTED_GROSS_INCOME, "under_1"): "No adjusted gross income",
        # SOI Tables 1.2, 1.4, 2.1 and 4.3 (53 targets); Historic Table 2 and
        # Table 1.1 write "All returns" (47).
        (_ADJUSTED_GROSS_INCOME, "all"): "All returns, total",
        # Census PEP state estimates by age (51 targets); the national table
        # writes "Age 85 and over" (1 target).
        ("age", "85_plus"): "Age 85 plus",
    }
)
#: The spellings each US choice was reviewed for; another is refused.
US_DIMENSION_VALUE_REVIEWED_SPELLINGS: Mapping[tuple[str, str], frozenset[str]] = (
    MappingProxyType(
        {
            (_ADJUSTED_GROSS_INCOME, "under_1"): frozenset(
                {"No adjusted gross income", "No adjusted gross income and deficit"}
            ),
            (_ADJUSTED_GROSS_INCOME, "all"): frozenset(
                {"All returns", "All returns, total"}
            ),
            ("age", "85_plus"): frozenset({"Age 85 plus", "Age 85 and over"}),
        }
    )
)

_COUNTRY_TOTAL = ("measure", "country_total")

#: ``(dimension ID, value ID)`` -> label, for the Belgian value each source
#: package labels its own way.
BE_DIMENSION_VALUE_LABEL_CHOICES: Mapping[tuple[str, str], str] = MappingProxyType(
    {_COUNTRY_TOTAL: "Country total"}
)
#: The spellings the Belgian choice was reviewed for; another is refused.
BE_DIMENSION_VALUE_REVIEWED_SPELLINGS: Mapping[tuple[str, str], frozenset[str]] = (
    MappingProxyType(
        {
            _COUNTRY_TOTAL: frozenset(
                {
                    # spf_finances/pit_2023 (calibration target
                    # spf_finances_pit_total)
                    "Belgium PIT federal and local tax before withholding",
                    # onss/contributions_2024 (onss_employee_contribution_total)
                    "Worker-borne personal social security contributions",
                    # onem_rva/unemployment_2024 (onem_unemployment_caseload)
                    "Complete unemployment benefit recipients",
                    # nbb/national_accounts_household_disposable_income_2024
                    # (validation target nbb_household_disposable_income)
                    "Household disposable income gross",
                }
            ),
        }
    )
)

UK_HIERARCHY_LABEL_CHOICES = HierarchyLabelChoices(
    geographies=UK_GEOGRAPHY_ID_TO_LABEL,
    geography_dimensions=frozenset({"geography"}),
)
US_HIERARCHY_LABEL_CHOICES = HierarchyLabelChoices(
    dimension_values=US_DIMENSION_VALUE_LABEL_CHOICES,
    dimension_value_spellings=US_DIMENSION_VALUE_REVIEWED_SPELLINGS,
)
BE_HIERARCHY_LABEL_CHOICES = HierarchyLabelChoices(
    dimension_values=BE_DIMENSION_VALUE_LABEL_CHOICES,
    dimension_value_spellings=BE_DIMENSION_VALUE_REVIEWED_SPELLINGS,
)
_CHOICES_BY_COUNTRY: Mapping[str, HierarchyLabelChoices] = MappingProxyType(
    {
        "be": BE_HIERARCHY_LABEL_CHOICES,
        "uk": UK_HIERARCHY_LABEL_CHOICES,
        "us": US_HIERARCHY_LABEL_CHOICES,
    }
)


def hierarchy_label_choices(country: str | None) -> HierarchyLabelChoices:
    """The label choices for a registry's country; none for other countries."""

    return _CHOICES_BY_COUNTRY.get(
        (country or "").strip().lower(), HierarchyLabelChoices()
    )
