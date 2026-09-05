"""Country-owned display labels for calibration data providers.

The identifiers remain the stable values used for grouping, filtering, and
joins.  These mappings provide presentation text only; they deliberately live
outside Chronicle facts and target-reference metadata so every target from the
same provider is serialized consistently.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "CALIBRATION_PROVIDER_LABELS_BY_COUNTRY",
    "UK_CALIBRATION_PROVIDER_LABELS",
    "US_CALIBRATION_PROVIDER_LABELS",
    "calibration_provider_label",
]


US_CALIBRATION_PROVIDER_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "bea": "BEA",
        "bea_nipa": "BEA · National Income and Product Accounts",
        "cbo": "CBO",
        "census_acs": "Census · American Community Survey",
        "census_pep": "Census · Population Estimates Program",
        "census_population": "Census population",
        "census_population_projections": "Census · Population Projections",
        "census_stc": "Census · State Tax Collections",
        "cms_aca": "CMS · ACA marketplace",
        "cms_medicaid": "CMS · Medicaid / CHIP",
        "cms_medicare": "CMS · Medicare",
        "cms_nhe": "CMS · National Health Expenditure Accounts",
        "federal_reserve": "Federal Reserve",
        "federal_reserve_z1": "Federal Reserve · Financial Accounts (Z.1)",
        "hhs_acf_liheap": "HHS · LIHEAP",
        "hhs_acf_tanf": "HHS · TANF",
        "ici": "Investment Company Institute",
        "irs_soi": "IRS Statistics of Income",
        "jct": "JCT",
        "kff": "KFF",
        "ssa": "SSA",
        "ssa_ssi_monthly": "SSA · SSI Monthly Statistics",
        "ssa_supplement": "SSA · Annual Statistical Supplement",
        "state_income_tax": "State income tax",
        "unspecified": "Internal calibration constraint",
        "usda_snap": "USDA · SNAP",
    }
)


UK_CALIBRATION_PROVIDER_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "dwp": "Department for Work and Pensions",
        "hmrc": "HM Revenue and Customs",
        "isc": "Independent Schools Council",
        "obr": "Office for Budget Responsibility",
        "ons": "Office for National Statistics",
        "scotgov": "Scottish Government",
        "slc": "Student Loans Company",
        "voa": "Valuation Office Agency",
    }
)


CALIBRATION_PROVIDER_LABELS_BY_COUNTRY: Mapping[str, Mapping[str, str]] = (
    MappingProxyType(
        {
            "uk": UK_CALIBRATION_PROVIDER_LABELS,
            "us": US_CALIBRATION_PROVIDER_LABELS,
        }
    )
)


def calibration_provider_label(country: str, source_id: str) -> str | None:
    """Return a country-owned display label for a structured source id."""

    labels = CALIBRATION_PROVIDER_LABELS_BY_COUNTRY.get(country.strip().lower())
    if labels is None:
        return None
    return labels.get(source_id.strip())
