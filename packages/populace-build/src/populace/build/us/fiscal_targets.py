"""US fiscal target-profile requirements and declared target facts.

JCT tax-expenditure rows must be computed from simple neutralization reforms:
run baseline income tax, neutralize one provision, run income tax again, and
use ``reform_income_tax - baseline_income_tax`` as the per-household
calibration row. This avoids treating tax expenditures as ordinary aggregate
columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from populace.build.gates import TargetCoverageRequirement
from populace.build.ledger_targets import (
    LedgerTargetParityReport,
    LedgerTargetReference,
    compile_ledger_target_references,
    ledger_target_registry_parity_report,
)
from populace.calibrate import TargetRegistry, TargetSpec

__all__ = [
    "US_FISCAL_MACRO_REALISM_BANDS",
    "US_FISCAL_TARGET_REGISTRY",
    "US_FISCAL_TARGET_SPECS",
    "US_FISCAL_TARGET_COVERAGE_REQUIREMENTS",
    "US_FISCAL_TARGET_LEDGER_REFERENCES",
    "US_FISCAL_LEDGER_PARITY_REGISTRY",
    "US_FISCAL_LEDGER_PARITY_REPORT",
    "US_JCT_TAX_EXPENDITURE_REFORMS",
    "US_JCT_TAX_EXPENDITURE_TARGET_SPECS",
    "US_SOI_FISCAL_TARGET_SPECS",
    "US_STATE_INCOME_TAX_TARGET_SPECS",
    "SimpleTaxExpenditureReform",
]

TaxExpenditureReformKind = Literal["neutralize_variable"]
TaxExpenditureMatrixRow = Literal["reform_minus_baseline_income_tax"]


@dataclass(frozen=True)
class SimpleTaxExpenditureReform:
    """A JCT target row backed by one simple neutralization reform.

    The builder implementation should construct a PolicyEngine reform that
    calls ``neutralize_variable(neutralized_variable)``, calculate
    ``income_tax`` under both baseline and reform, and calibrate to the
    household-level delta ``reform - baseline``.
    """

    target_name: str
    neutralized_variable: str
    value: float
    source: str
    measure: str
    period: int | str
    kind: TaxExpenditureReformKind = "neutralize_variable"
    output_variable: str = "income_tax"
    matrix_row: TaxExpenditureMatrixRow = "reform_minus_baseline_income_tax"

    @classmethod
    def from_target_spec(cls, spec: TargetSpec) -> SimpleTaxExpenditureReform:
        """Build the reform contract from a declared JCT target row."""
        return cls(
            target_name=spec.name,
            neutralized_variable=spec.metadata.get("neutralized_variable", ""),
            value=spec.value,
            source=spec.source,
            measure=spec.measure or "",
            period=spec.period,
            kind=spec.metadata.get("kind", ""),
            output_variable=spec.metadata.get("output_variable", ""),
            matrix_row=spec.metadata.get("matrix_row", ""),
        )

    def __post_init__(self) -> None:
        if not self.target_name:
            raise ValueError("target_name is required.")
        if not self.neutralized_variable:
            raise ValueError(f"{self.target_name}: neutralized_variable is required.")
        if not self.value > 0:
            raise ValueError(f"{self.target_name}: value must be positive.")
        if not self.source:
            raise ValueError(f"{self.target_name}: source is required.")
        if self.measure != self.target_name:
            raise ValueError(
                f"{self.target_name}: JCT targets must use a precomputed "
                "measure column named after the target row, not an ordinary "
                "income_tax column."
            )
        if self.kind != "neutralize_variable":
            raise ValueError(
                f"{self.target_name}: JCT targets must use a simple "
                "neutralize_variable reform."
            )
        if self.output_variable != "income_tax":
            raise ValueError(
                f"{self.target_name}: JCT targets must fit the income_tax delta."
            )
        if self.matrix_row != "reform_minus_baseline_income_tax":
            raise ValueError(
                f"{self.target_name}: JCT matrix row must be reform income_tax "
                "minus baseline income_tax."
            )

    def coverage_requirement(self) -> TargetCoverageRequirement:
        """The target-profile requirement satisfied by this JCT row."""
        return TargetCoverageRequirement(
            requirement_id=f"jct_tax_expenditure:{self.neutralized_variable}",
            label=f"JCT tax expenditure for {self.neutralized_variable}",
            accepted_names=(self.target_name,),
            required_measures=(self.target_name,),
            required_metadata=(
                ("kind", self.kind),
                ("output_variable", self.output_variable),
                ("matrix_row", self.matrix_row),
                ("neutralized_variable", self.neutralized_variable),
            ),
            notes=(
                "Must be computed as a simple neutralize_variable reform and "
                "calibrated to income_tax(reform) - income_tax(baseline)."
            ),
        )


def _load_us_fiscal_target_specs() -> tuple[TargetSpec, ...]:
    payload = json.loads(files(__package__).joinpath("fiscal_targets.json").read_text())
    if payload.get("country") != "us":
        raise ValueError("US fiscal target manifest must declare country='us'.")
    return tuple(TargetSpec(**raw) for raw in payload["target_specs"])


def _ledger_fact_key_for_spec(spec: TargetSpec) -> str:
    return f"populace.us.fiscal_target.v1:{spec.name}@{spec.period}"


def _ledger_reference_from_spec(spec: TargetSpec) -> LedgerTargetReference:
    ledger_fact_key = _ledger_fact_key_for_spec(spec)
    return LedgerTargetReference(
        name=spec.name,
        ledger_fact_key=ledger_fact_key,
        ledger_source_record_id=ledger_fact_key,
        entity=spec.entity,
        measure=spec.measure,
        aggregation=spec.aggregation,
        filter=spec.filter,
        period=spec.period,
        source=spec.source,
        family=spec.family,
        signed=spec.signed,
        se=spec.se,
        tolerance=spec.tolerance,
        notes=spec.notes,
        metadata=spec.metadata,
    )


def _ledger_compat_fact_from_spec(spec: TargetSpec) -> dict[str, object]:
    """Represent the current manifest as Ledger-shaped facts for parity only.

    This is the no-behavior-change bridge: it validates the Populace target
    registry model against the Ledger reference compiler before any production
    target loads directly from the Ledger database.
    """

    ledger_fact_key = _ledger_fact_key_for_spec(spec)
    return {
        "aggregate_fact_key": ledger_fact_key,
        "lineage": {"source_record_id": ledger_fact_key},
        "value": spec.value,
        "period": {
            "type": "tax_year" if isinstance(spec.period, int) else "period",
            "value": spec.period,
        },
        "entity": {"name": spec.entity},
        "aggregation": {"method": spec.aggregation},
        "observed_measure": {
            "source_name": spec.family,
            "source_table": spec.source,
            "source_measure_id": spec.measure or spec.name,
        },
        "source": {
            "source_name": spec.family,
            "source_table": spec.source,
        },
    }


US_FISCAL_TARGET_SPECS: tuple[TargetSpec, ...] = _load_us_fiscal_target_specs()
US_FISCAL_TARGET_REGISTRY = TargetRegistry(US_FISCAL_TARGET_SPECS, country="us")
US_FISCAL_TARGET_LEDGER_REFERENCES: tuple[LedgerTargetReference, ...] = tuple(
    _ledger_reference_from_spec(spec) for spec in US_FISCAL_TARGET_SPECS
)
US_FISCAL_LEDGER_PARITY_REGISTRY = compile_ledger_target_references(
    (_ledger_compat_fact_from_spec(spec) for spec in US_FISCAL_TARGET_SPECS),
    US_FISCAL_TARGET_LEDGER_REFERENCES,
    country="us",
)
US_FISCAL_LEDGER_PARITY_REPORT: LedgerTargetParityReport = (
    ledger_target_registry_parity_report(
        US_FISCAL_TARGET_REGISTRY,
        US_FISCAL_LEDGER_PARITY_REGISTRY,
    )
)
US_JCT_TAX_EXPENDITURE_TARGET_SPECS: tuple[TargetSpec, ...] = tuple(
    spec for spec in US_FISCAL_TARGET_SPECS if spec.family == "jct"
)
US_JCT_TAX_EXPENDITURE_REFORMS: tuple[SimpleTaxExpenditureReform, ...] = tuple(
    SimpleTaxExpenditureReform.from_target_spec(spec)
    for spec in US_JCT_TAX_EXPENDITURE_TARGET_SPECS
)
US_STATE_INCOME_TAX_TARGET_SPECS: tuple[TargetSpec, ...] = tuple(
    spec for spec in US_FISCAL_TARGET_SPECS if spec.family == "state_income_tax"
)
US_SOI_FISCAL_TARGET_SPECS: tuple[TargetSpec, ...] = tuple(
    spec for spec in US_FISCAL_TARGET_SPECS if spec.family == "irs_soi"
)

US_FISCAL_TARGET_COVERAGE_REQUIREMENTS: tuple[TargetCoverageRequirement, ...] = (
    TargetCoverageRequirement(
        requirement_id="federal_income_tax_total",
        label="Federal individual income tax total",
        accepted_names=(
            "nation/cbo/individual_income_tax",
            "nation/treasury/individual_income_tax",
            "nation/treasury/individual income tax",
            "nation/irs/income_tax_total",
            "nation/irs/total_income_tax",
            "nation/irs/total income tax",
        ),
        accepted_name_prefixes=(
            "nation/cbo/individual_income_tax/",
            "nation/treasury/individual_income_tax/",
            "nation/treasury/individual income tax/",
            "nation/irs/income_tax_total/",
            "nation/irs/total_income_tax/",
            "nation/irs/total income tax/",
        ),
        notes=(
            "A positive-income-tax diagnostic is not enough; the profile needs "
            "the total federal income tax aggregate used by downstream netting."
        ),
    ),
    TargetCoverageRequirement(
        requirement_id="irs_agi_distribution",
        label="SOI AGI distribution and top-tail controls",
        accepted_name_substrings=("/irs/adjusted gross income/",),
        min_matches=20,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_wages_distribution",
        label="SOI wages by AGI bracket",
        accepted_name_substrings=(
            "/irs/salaries and wages/",
            "/irs/employment income/",
        ),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_business_income_distribution",
        label="SOI business income by AGI bracket",
        accepted_name_substrings=("/irs/business net ",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_partnership_s_corp_distribution",
        label="SOI partnership and S-corp income by AGI bracket",
        accepted_name_substrings=("/irs/partnership and s corp income/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_capital_gains_distribution",
        label="SOI capital gains by AGI bracket",
        accepted_name_substrings=("/irs/capital gains gross/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_dividends_distribution",
        label="SOI dividends by AGI bracket",
        accepted_name_substrings=("/irs/ordinary dividends/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_interest_distribution",
        label="SOI taxable interest by AGI bracket",
        accepted_name_substrings=("/irs/taxable interest income/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_pension_distribution",
        label="SOI pension income by AGI bracket",
        accepted_name_substrings=("/irs/total pension income/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_social_security_distribution",
        label="SOI Social Security by AGI bracket",
        accepted_name_substrings=("/irs/total social security/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="state_income_tax",
        label="State individual income tax collections",
        accepted_name_prefixes=("state/",),
        accepted_families=("state_income_tax",),
        required_metadata=(("target_role", "state_income_tax"),),
        min_matches=51,
    ),
    *(spec.coverage_requirement() for spec in US_JCT_TAX_EXPENDITURE_REFORMS),
)

US_FISCAL_MACRO_REALISM_BANDS: dict[str, tuple[float, float]] = {
    "federal_income_tax_to_gdp": (0.07, 0.11),
    "agi_to_gdp": (0.50, 0.70),
    "spm_below_threshold_rate": (0.06, 0.18),
}
