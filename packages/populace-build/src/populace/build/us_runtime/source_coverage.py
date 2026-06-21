"""US source-family coverage for US release gates.

This is a pinned Populace-side copy of the Ledger source coverage contract
merged in ``PolicyEngine/ledger-data`` commit
``e2fc882c35f9203c788a7159e7b08b0b5e5ceb58``. Ledger owns source packages;
Populace owns whether source families are active hard targets, validation-only
diagnostics, or explicit source gaps in a release profile.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from populace.build.gates import GateResult, source_coverage_gate

__all__ = [
    "LEDGER_US_SOURCE_COVERAGE_CONTRACT_COMMIT",
    "CoverageRole",
    "SourceCoverageEntry",
    "US_SOURCE_COVERAGE",
    "hard_target_package_aliases",
    "source_gap_family_ids",
    "us_source_coverage_diagnostics",
    "us_source_coverage_gate",
    "write_us_source_coverage_diagnostics",
    "validation_only_family_ids",
]

LEDGER_US_SOURCE_COVERAGE_CONTRACT_COMMIT = "e2fc882c35f9203c788a7159e7b08b0b5e5ceb58"

CoverageRole = Literal["hard_target", "validation_only", "source_gap"]


@dataclass(frozen=True)
class SourceCoverageEntry:
    """Coverage status for one source family relevant to US release work."""

    family_id: str
    label: str
    role: CoverageRole
    package_aliases: tuple[str, ...] = ()
    missing_source_packages: tuple[str, ...] = ()
    notes: str = ""


US_SOURCE_COVERAGE: tuple[SourceCoverageEntry, ...] = (
    SourceCoverageEntry(
        "population_age_sex",
        "Population by age, sex, state, and congressional district",
        "hard_target",
        (
            "census-pep-2024-national-age-sex",
            "census-pep-2024-state-age-sex",
            "census-acs-s0101-national-age-2024",
            "census-acs-s0101-state-age-2024",
            "census-acs-s0101-congressional-district-age-2024",
        ),
    ),
    SourceCoverageEntry(
        "nipa_personal_income",
        "NIPA personal income, transfers, taxes, and pensions",
        "hard_target",
        (
            "bea-nipa-total-wages-salaries",
            "bea-nipa-personal-income-components",
            "bea-nipa-personal-income-disposition",
            "bea-nipa-pension-contributions",
        ),
    ),
    SourceCoverageEntry(
        "irs_soi_filer_income_tax_credits",
        "SOI filer income, taxes, deductions, and credits",
        "hard_target",
        (
            "soi-table-1-1",
            "soi-table-1-2",
            "soi-table-1-4",
            "soi-table-2-1",
            "soi-table-2-5",
            "soi-table-2-5-eitc-agi-children-2023",
            "soi-filing-season-week47-2024-eitc-total",
            "soi-table-4-3",
            "soi-state-2022",
            "soi-historic-table-2",
            "soi-historic-table-2-state-agi-2022",
            "soi-historic-table-2-state-broad-2022",
            "soi-historic-table-2-state-eitc-2022",
            "soi-w2-statistics-2020",
        ),
    ),
    SourceCoverageEntry(
        "social_security_ssi",
        "Social Security and SSI payments",
        "hard_target",
        ("ssa-annual-statistical-supplement-2025", "ssa-ssi-table-7b1-2024"),
    ),
    SourceCoverageEntry(
        "snap_admin",
        "SNAP participation and benefit cost",
        "hard_target",
        ("usda-snap-fy69-to-current",),
    ),
    SourceCoverageEntry(
        "tanf_admin",
        "TANF caseload and financial data",
        "hard_target",
        ("hhs-acf-tanf-caseload-2024", "hhs-acf-tanf-financial-2024"),
    ),
    SourceCoverageEntry(
        "liheap_admin",
        "LIHEAP households and benefits",
        "hard_target",
        (
            "hhs-acf-liheap-fy2023-national-profile",
            "hhs-acf-liheap-fy2024-national-profile",
        ),
    ),
    SourceCoverageEntry(
        "health_programs",
        "Medicaid, CHIP, ACA, Medicare, and NHE controls",
        "hard_target",
        (
            "cms-medicaid-chip-monthly-enrollment-dataset",
            "cms-medicaid-chip-monthly-enrollment-december-2024",
            "cms-nhe-historical-service-source",
            "cms-aca-oep-state-level",
            "cms-aca-oep-state-level-2022",
            "cms-aca-oep-state-level-2025",
            "cms-aca-effectuated-enrollment-2022",
            "cms-medicare-trustees-report-2025-part-b-premium-income",
        ),
    ),
    SourceCoverageEntry(
        "state_income_tax_collections",
        "State individual income tax collections",
        "hard_target",
        ("census-stc-individual-income-tax",),
    ),
    SourceCoverageEntry(
        "tax_expenditures",
        "JCT tax expenditure revenue-loss estimates",
        "hard_target",
        ("jct-tax-expenditures-2024",),
    ),
    SourceCoverageEntry(
        "snap_local_proxy",
        "SNAP congressional district household estimates",
        "validation_only",
        ("census-acs-s2201-congressional-district-snap-2024",),
    ),
    SourceCoverageEntry(
        "cbo_income_revenue_projection",
        "CBO income and revenue projections",
        "validation_only",
        ("cbo-revenue-projections-income-by-source-2026-02",),
    ),
    SourceCoverageEntry(
        "wealth_balance_sheet",
        "Household net worth balance-sheet checks",
        "validation_only",
        ("federal-reserve-z1-household-net-worth",),
    ),
    SourceCoverageEntry(
        "census_cps_spm",
        "Census CPS ASEC SPM resources and thresholds",
        "validation_only",
        notes="Validation only; not a hard calibration target.",
    ),
    SourceCoverageEntry(
        "dina_distributional_accounts",
        "Distributional national accounts",
        "validation_only",
        notes="Validation only; not a hard calibration target.",
    ),
    SourceCoverageEntry(
        "acs_income_distribution",
        "ACS income distributions",
        "validation_only",
        notes="Validation only; not an SPM hard target.",
    ),
    SourceCoverageEntry(
        "hud_assisted_housing",
        "Housing assistance and subsidy controls",
        "source_gap",
        missing_source_packages=(
            "HUD Picture of Subsidized Households",
            "HUD assisted-housing expenditure or unit-count tables",
        ),
    ),
    SourceCoverageEntry(
        "usda_wic",
        "WIC participation and benefits",
        "source_gap",
        missing_source_packages=("USDA FNS WIC program data",),
    ),
    SourceCoverageEntry(
        "usda_school_meals",
        "School lunch and breakfast benefits",
        "source_gap",
        missing_source_packages=(
            "USDA FNS National School Lunch Program data",
            "USDA FNS School Breakfast Program data",
        ),
    ),
    SourceCoverageEntry(
        "ocse_child_support",
        "Child support received and paid",
        "source_gap",
        missing_source_packages=("HHS OCSE child support annual report tables",),
    ),
    SourceCoverageEntry(
        "dol_workers_compensation",
        "Workers' compensation benefits",
        "source_gap",
        missing_source_packages=(
            "DOL or NASI workers' compensation benefit totals",
            "State workers' compensation benefit totals",
        ),
    ),
    SourceCoverageEntry(
        "moop_work_childcare_costs",
        "MOOP, work expenses, and childcare expense validation",
        "source_gap",
        missing_source_packages=(
            "MEPS out-of-pocket medical spending tables",
            "BLS Consumer Expenditure work-related expense tables",
            "Childcare expense validation source",
        ),
    ),
)


def hard_target_package_aliases() -> tuple[str, ...]:
    """Ledger package aliases required for hard-target source coverage."""
    return tuple(
        sorted(
            {
                alias
                for entry in US_SOURCE_COVERAGE
                if entry.role == "hard_target"
                for alias in entry.package_aliases
            }
        )
    )


def validation_only_family_ids() -> tuple[str, ...]:
    """Source families that can be diagnostics but must not be hard targets."""
    return tuple(
        entry.family_id
        for entry in US_SOURCE_COVERAGE
        if entry.role == "validation_only"
    )


def source_gap_family_ids() -> tuple[str, ...]:
    """Source families the release must report as currently unsupported."""
    return tuple(
        entry.family_id for entry in US_SOURCE_COVERAGE if entry.role == "source_gap"
    )


def us_source_coverage_gate(
    *,
    active_target_aliases: Iterable[str] = (),
    active_target_families: Iterable[str] = (),
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Build the named US source-coverage release gate."""
    return source_coverage_gate(
        US_SOURCE_COVERAGE,
        active_target_aliases=active_target_aliases,
        active_target_families=active_target_families,
        reviewed_exclusions=reviewed_exclusions,
        name="us_source_coverage",
    )


def _json_ready(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False))


def us_source_coverage_diagnostics(
    *,
    active_target_aliases: Iterable[str] = (),
    active_target_families: Iterable[str] = (),
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return the US source-coverage diagnostics artifact payload."""
    result = us_source_coverage_gate(
        active_target_aliases=active_target_aliases,
        active_target_families=active_target_families,
        reviewed_exclusions=reviewed_exclusions,
    )
    details = _json_ready(result.details)
    if not isinstance(details, dict):  # pragma: no cover - defensive
        raise TypeError("source coverage details must be a JSON object.")
    return {
        "schema_version": 1,
        "classification": "release_gate",
        "source_contract": {
            "name": "us_source_coverage",
            "ledger_commit": LEDGER_US_SOURCE_COVERAGE_CONTRACT_COMMIT,
        },
        "gate": {
            "name": result.name,
            "passed": result.passed,
            "failures": list(result.failures),
        },
        "coverage_summary": details["coverage_summary"],
        "hard_target_families": details["hard_target_families"],
        "validation_only_families": details["validation_only_families"],
        "source_gap_families": details["source_gap_families"],
        "active_target_aliases": details["active_target_aliases"],
        "active_target_families": details["active_target_families"],
        "missing_hard_targets": details["missing_hard_targets"],
        "reviewed_exclusions": details["reviewed_exclusions"],
        "validation_only_activated": details["validation_only_activated"],
    }


def write_us_source_coverage_diagnostics(
    payload: Mapping[str, object], path: Path | str
) -> Path:
    """Write a US source-coverage diagnostics artifact as strict JSON."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path
