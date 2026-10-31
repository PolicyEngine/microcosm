"""Regenerate UK Ledger compile-parity signed-difference resources."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path

import numpy as np

from microcosm.build.gates import ledger_compile_parity_signed_differences
from microcosm.build.uk_runtime.derived_facts import uk_derived_fact_rows
from microcosm.build.uk_runtime.ledger_targets import (
    LOCAL_REGISTRY_PARITY_FIXTURE_RESOURCE,
    align_uk_local_registry_parity_fixture,
    align_uk_national_registry_parity_fixture,
    compile_uk_local_target_registry,
    compile_uk_target_registry,
)

UK_PACKAGE = "microcosm.build.uk"
UK_PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "uk"
)


@dataclass(frozen=True)
class ParityReceiptSpec:
    fixture_resource: str
    output_resource: str
    target_period: int
    surface: str = "national"


RECEIPTS = (
    ParityReceiptSpec(
        fixture_resource="registry_parity_fixture_2025.json",
        output_resource="ledger_compile_parity_incumbent_2025_signed_differences.json",
        target_period=2025,
    ),
    ParityReceiptSpec(
        fixture_resource=LOCAL_REGISTRY_PARITY_FIXTURE_RESOURCE,
        output_resource=(
            "ledger_compile_parity_local_incumbent_2025_signed_differences.json"
        ),
        target_period=2025,
        surface="local",
    ),
)

_SPI_UPRATED_DRIFT_RATIONALE = (
    "Declared uprating (PolicyEngine/chronicle#280 lane, María's ruling of 2026-09-22): ours "
    "moves the SPI 2023-24 band value to the 2025 calibration year by the index "
    "the reference declares (amounts by the pinned engine's per-variable OBR and "
    "ONS indices between 1 January 2023 and 1 January 2025, counts by HMRC's projected "
    "taxpayer growth in the containing Table 2.5 band as the calendar-2025 "
    "window); the frozen incumbent fixture holds the 2023-24 values it projects "
    "one year forward with its own income-projection ratios."
)

_SPI_PROPERTY_AMOUNT_DRIFT_RATIONALE = (
    "The x1.9 adjustment is retired (microcosm#1106, María's ruling of "
    "2026-10-05): ours binds the SPI 2023-24 Table 3.7 net income from "
    "property, landlords' profit after allowable expenses and before "
    "residential finance costs, moved to the 2025 calibration year by HMRC's "
    "projected total income in the band (hmrc.itl_2026."
    "total_income_growth_by_total_income_band, the index the count rows' "
    "growth belongs with; María's ruling of 2026-10-11); the frozen incumbent "
    "fixture holds "
    "1.9 times the SPI amounts (uk-data#311), a factor that set HMRC Property "
    "Rental Income Statistics receipts, which are before expenses, against "
    "this net concept."
)

_INCOME_ANCHOR_LEDGER_ONLY_RATIONALE = (
    "Coverage the incumbent lacks (PolicyEngine/chronicle#280 lane): HMRC's Income Tax "
    "liabilities Table 2.5 taxpayers, total income and tax by band as the "
    "calendar-2025 window of the 2024-25 and 2025-26 projections, and the SPI "
    "2023-24 Table 3.7 savings-interest and other-income amounts by band, grown "
    "to the calibration year by the OBR's savings income forecast (October "
    "2026 EFO Table 4.5; María's ruling of 2026-10-29, microcosm#1418; until "
    "then the interest amounts moved to the dataset's year by the household "
    "interest index and on by GDP per head, microcosm#1095), with the "
    "savings-interest counts by band grown by HMRC's projected taxpayers in "
    "the same Table 2.5 band; the incumbent registry binds neither "
    "(chronicle#282)."
)

_SPI_REGION_LEDGER_ONLY_RATIONALE = (
    "Coverage the incumbent lacks (PolicyEngine/chronicle#280 lane, María's ruling of "
    "2026-09-22 that the regions bind uprated): the SPI 2023-24 Table 3.11 "
    "Income Tax payers, total income and Income Tax liabilities by ten regional "
    "total-income bands, one row per region-tier area (microcosm#905), each "
    "moved to the 2025 calibration year by HMRC's projected growth for the same "
    "measure in the Table 2.5 band(s) the regional band spans; the incumbent "
    "registry has no regional SPI band metric (chronicle#282)."
)

_OBR_WINDOW_DRIFT_RATIONALE = (
    "Period basis (PolicyEngine/chronicle#280 lane, María's rule of 2026-09-22 that "
    "fiscal-year facts take only the months to the end of the calibration "
    "calendar year): ours binds the OBR March 2026 line at the calendar-2025 "
    "window, three twelfths of FY2024-25 and nine twelfths of FY2025-26, on "
    "the same series; the frozen incumbent fixture holds the FY2025-26 value "
    "alone."
)

_OBR_COUNCIL_TAX_GB_RECEIPTS_RATIONALE = (
    "Series and period basis (microcosm#1095, María's ruling of 2026-10-09): ours "
    "binds OBR table 4.1 row 15, the council tax receipts of England, Scotland "
    "and Wales, over Great Britain's households, at the calendar-2025 window "
    "(three twelfths of FY2024-25 and nine twelfths of FY2025-26, María's rule "
    "of 2026-09-22, PolicyEngine/chronicle#280 lane), scaled to main residences "
    "by England's chargeable dwellings less its second homes and empty "
    "dwellings, over its chargeable dwellings (MHCLG Council Taxbase, October "
    "2025; María's ruling of 2026-10-29, microcosm#1095 D14f); the frozen incumbent "
    "fixture holds the FY2025-26 value of row 19 alone, which adds Northern "
    "Ireland domestic rates, the council tax accruals adjustment and the "
    "high-value council tax surcharge."
)

_OBR_SDLT_HOUSEHOLDS_RATIONALE = (
    "Payer and period basis (uk-data#564, microcosm#1095): ours binds households' "
    "SDLT, the OBR March 2026 line at the calendar-2025 window (three twelfths of "
    "FY2024-25 and nine twelfths of FY2025-26, María's rule of 2026-09-22, "
    "PolicyEngine/chronicle#280 lane) times individuals' share of HMRC's 2024-25 "
    "SDLT receipts, residential (Table 8a) and non-residential (Table 8b) "
    "together; the frozen incumbent fixture holds the FY2025-26 value of the "
    "whole line alone, which companies, partnerships and collective investment "
    "schemes pay about a third of."
)

_ESA_CUBE_DRIFT_RATIONALE = (
    "Source class (PolicyEngine/chronicle#280 lane): ours binds DWP's Stat-Xplore ESA "
    "caseload by payment type as the mean of the four quarterly points inside "
    "calendar 2025 summed over the paid types (credits-only claimants left out; "
    "dual claimants counted under both types they hold); the incumbent binds the "
    "August 2025 headline and single-type counts from the benefit statistics "
    "summary (chronicle#282)."
)

_ESA_CUBE_ROWS = (
    "dwp.esa_claimants",
    "dwp.esa_contrib_claimants",
    "dwp.esa_income_claimants",
)

#: microcosm#1069: the resident State Pension level replaces the OBR line in
#: the fit (ruling R4); the exact OBR forecast stays as diagnostic provenance.
_OBR_STATE_PENSION_DIAGNOSTIC_RATIONALE = (
    "microcosm#1069 (ruling R4) moves the OBR FY2025-26 State Pension forecast "
    "out of the fit: it counts Great Britain plus pensioners paid abroad and "
    "leaves out Northern Ireland, a different quantity from the resident level "
    "the DWP Stat-Xplore and DfC rows bind. The exact forecast is retained as "
    "diagnostic provenance on dwp.state_pension.amount, so the frozen fixture's "
    "fitted row is fixture-only on the current surface."
)
_OBR_NI_DIAGNOSTIC_RATIONALE = (
    "microcosm#1095 (María's ruling of 2026-10-09; uk-data#537 drops the row) moves "
    "OBR's total NICs receipts out of the fit: a cash total against the three "
    "accrued class rows, carrying statutory-payment recoveries, Class 1A, 1B and 3 "
    "and settlements that no household carries. The exact FY2025-26 figure is "
    "retained as diagnostic provenance on obr.ni_employee, so the frozen fixture's "
    "fitted row is fixture-only on the current surface."
)
#: The next policyengine-uk bump (microcosm#1095, María's ruling of
#: 2026-10-22) retires these rows from the targets to registry_parity.excluded.
_OBR_UC_CAP_SPLIT_ROWS = frozenset(
    {"obr.universal_credit_in_cap", "obr.universal_credit_outside_cap"}
)
_OBR_UC_CAP_SPLIT_RETIRED_RATIONALE = (
    "microcosm#1095 (María's ruling of 2026-10-22) retires OBR's split of "
    "Universal Credit spending at the welfare cap from the targets: the welfare "
    "cap is a spending limit, no household-level concept in the model marks the "
    "boundary, and the benefit-cap status the incumbent maps the split to is "
    "another concept (microcosm#757). obr.universal_credit binds the sum of the "
    "two cells, so the frozen fixture's fitted row is fixture-only on the current "
    "surface."
)
_SLC_POSTGRADUATE_RETIRED_RATIONALE = (
    "microcosm#1095 (María's ruling of 2026-10-22) retires SLC's England "
    "postgraduate repayments from the targets: the student-loans stage assigns no "
    "postgraduate plan and policyengine-uk holds one plan a person, so no record "
    "carries the repayments (microcosm#757). The frozen fixture's fitted row is "
    "fixture-only on the current surface."
)
_DWP_PIP_CASELOAD_DIAGNOSTIC_ROWS = frozenset(
    {
        "dwp.pip.daily_living_standard_claimants",
        "dwp.pip.daily_living_enhanced_claimants",
    }
)
_DWP_PIP_CASELOAD_DIAGNOSTIC_RATIONALE = (
    "microcosm#1095 (María's ruling of 2026-10-10) keeps DWP's England-and-Wales "
    "PIP daily living caseload (FOI2025/24990, end of January 2025) out of the "
    "fit: the two rows resolve and fit, but they push the heaviest support "
    "family's weight over the uk_weight_ratio fence. The exact figures are "
    "retained as diagnostic provenance on obr.pip, so the frozen fixture's "
    "fitted rows are fixture-only on the current surface."
)
_DWP_UC_FIVE_OR_MORE_DIAGNOSTIC_RATIONALE = (
    "microcosm#1095 (María's ruling of 2026-10-16) keeps DWP's count of Universal "
    "Credit households with five or more children out of the fit: it includes "
    "families outside the two-child limit, which the bound two-child-limit rows "
    "leave out and too few FRS source households carry. The April 2025 figure is "
    "retained as diagnostic provenance on "
    "dwp.uc.two_child_limit.households_5_children, so the frozen fixture's fitted "
    "row is fixture-only on the current surface."
)
_STATE_PENSION_PREFIXES = ("dwp.state_pension.", "dfc_ni.state_pension.")
_STATE_PENSION_LEDGER_ONLY_RATIONALE = (
    "microcosm#1069 binds the resident State Pension from DWP Stat-Xplore (Great "
    "Britain) and DfC (Northern Ireland) on the four quarterly points inside "
    "calendar 2025 (PolicyEngine/chronicle#302 via #305): the level, age band by "
    "sex by type, area by type and weekly amount band by type. The frozen fixture "
    "has no State Pension caseload or resident-level row."
)
_INCOME_ANCHOR_PREFIXES = (
    "hmrc.itl.",
    "hmrc.spi.savings_interest_income.",
    "hmrc.spi.other_investment_income.",
)
#: microcosm#1069: Income Tax payers by age (ITL Table 2.2), a family the
#: incumbent registry has no row of.
_ITL_AGE_PREFIXES = (
    "hmrc.itl.state_pension_age_taxpayers@",
    "hmrc.itl.taxpayers_aged_65_and_over@",
)
#: microcosm#1418: Income Tax on savings at each rate (ITL Table 2.6), which
#: the incumbent registry has no row of.
_ITL_SAVINGS_TAX_PREFIX = "hmrc.itl.savings_tax_at_"
_ITL_SAVINGS_TAX_LEDGER_ONLY_RATIONALE = (
    "Coverage the incumbent lacks (microcosm#1418): HMRC's Income Tax liabilities "
    "Table 2.6 (July 2026) Income Tax on savings at the basic, higher and additional "
    "rates, all payers, United Kingdom, as the calendar-2025 window of the 2024-25 "
    "and 2025-26 projections: the base the Budget 2026 savings income rates are "
    "costed on. The incumbent registry binds no tax by income source or rate."
)
_ITL_AGE_LEDGER_ONLY_RATIONALE = (
    "Coverage the incumbent lacks (microcosm#1069): HMRC's Income Tax liabilities "
    "Table 2.2 (July 2026) Income Tax payers of State Pension age and aged 65 and "
    "over, one row per region-tier area as the calendar-2025 window of the 2024-25 "
    "and 2025-26 projections. Table 2.2 counts residents; Table 2.1's United "
    "Kingdom row, which adds taxpayers abroad or of unknown region, is vendored "
    "but not bound. The incumbent registry binds no taxpayer count by age."
)
#: The two SPI State Pension band families the State Pension age index moves,
#: named exactly so no other ``hmrc/state_pension_*`` band row takes their
#: rationale.
_SPI_STATE_PENSION_BAND_PREFIXES = (
    "hmrc/state_pension_count_income_band_",
    "hmrc/state_pension_income_band_",
)
_SPI_STATE_PENSION_UPRATED_DRIFT_RATIONALE = (
    "Declared uprating (microcosm#1069, after María's ruling of 2026-09-22): ours "
    "moves the SPI 2023-24 State Pension band rows to the 2025 calibration year "
    "by one factor for every band, the calendar-2025 window of HMRC's resident "
    "Income Tax payers of State Pension age (Income Tax liabilities Table 2.2, "
    "the twelve region-tier areas) over Table 2.1's 2023-24 United Kingdom count, "
    "and the amount rows also by the new State Pension rate the engine pays; the "
    "frozen incumbent fixture holds the 2023-24 values it projects one year "
    "forward with its own income-projection ratios."
)

_CGT_OBSERVED_RATIONALES = {
    "hmrc.cgt.gains_total": (
        "PR #889 selects HMRC FY2024-25 individuals-only observed gains of "
        "GBP 119,258,000,000, measured in 2024 at calibration index 2025. "
        "The frozen incumbent fixture retains its historical forecast/uprated "
        "GBP 67,727,478,991.60 value. This explicit observation-year and "
        "population-scope change does not rewrite the historical fixture."
    ),
    "hmrc.cgt.taxpayers_total": (
        "PR #889 selects the HMRC FY2024-25 individuals-only observation of "
        "551,000 taxpayers, measured in 2024 at calibration index 2025. "
        "The frozen incumbent fixture retains its historical 378,000 count."
    ),
    "hmrc.cgt.liability_total": (
        "PR #889 adds HMRC FY2024-25 individuals-only liability of "
        "GBP 22,503,000,000 to the observed-year fit. The historical fixture "
        "has no equivalent liability row; this is a declared ledger-only "
        "observation, not a cash-receipts reconciliation."
    ),
    "obr.capital_gains_tax": (
        "PR #889 retains the exact March 2026 OBR FY2025-26 cash forecast "
        "as diagnostic provenance outside the FY2024-25 observation fit. "
        "The frozen incumbent fixture keeps its fitted cash row, so it is "
        "fixture-only on the current surface. This scoped change does not "
        "settle the general liability-to-cash translation tracked by #875."
    ),
}

#: The nine incumbent banded CGT names per measure now map onto the Table
#: 2.1a fan-out (microcosm#467): the incumbent read uk-data's 2023-24
#: size-of-gain CSV uprated by a national ratio; the current row binds the
#: published FY2024-25 individuals cell at identity.
_CGT_BAND_INCUMBENT_PREFIXES = ("hmrc/cgt_taxpayers_band_", "hmrc/capital_gains_band_")
_CGT_BAND_DRIFT_RATIONALE = (
    "Vintage and basis class: the incumbent banded CGT row reads uk-data's "
    "2023-24 HMRC Table 2.1a size-of-gain CSV uprated by a national ratio, "
    "while the current reference binds the published FY2024-25 individuals "
    "cell for the same band at identity (Chronicle cgt_size_of_gain_2026), "
    "measured in 2024 at calibration index 2025 like the national CGT "
    "observations (microcosm#467, #725). The frozen incumbent fixture keeps "
    "its historical value."
)
_CGT_NEW_ROW_PREFIXES = (
    "hmrc.cgt.taxpayers_by_age_band.",
    "hmrc.cgt.gains_by_age_band.",
    "hmrc.cgt.tax_by_age_band.",
    "hmrc.cgt.taxpayers_by_region@",
    "hmrc.cgt.gains_by_region@",
    "hmrc/cgt_taxpayers_band_3000",
    "hmrc/cgt_taxpayers_band_6000",
    "hmrc/cgt_taxpayers_band_10000",
    "hmrc/capital_gains_band_3000",
    "hmrc/capital_gains_band_6000",
    "hmrc/capital_gains_band_10000",
    "hmrc.cgt.residential_property_taxpayers",
    "hmrc.cgt.residential_property_gains",
    "hmrc.cgt.badr_ir_taxpayers_by_band.",
    "hmrc.cgt.badr_ir_qualifying_gains_by_band.",
    "hmrc.cgt.taxpayers_by_taxable_income_band.",
    "hmrc.cgt.gains_by_taxable_income_band.",
)
_CGT_NEW_ROW_LEDGER_ONLY_RATIONALE = (
    "Declared ledger-only observation: HMRC CGT statistics 2026 release, "
    "FY2024-25 individual rows by age band (Table 6), by country and region "
    "on the individuals basis via the Table 1 share (Table 5), and the "
    "size-of-gain bands the incumbent never carried (Table 2.1a), the "
    "residential property taxpayers and gains on the individuals basis via "
    "the Table 8b share (Table 8a), the Business Asset Disposal Relief and "
    "Investors' Relief claimants and qualifying gains by band of qualifying "
    "gain (Table 4.1), and the all-gains taxpayers and gains by band of the "
    "engine's CGT taxable income (Table 3). The frozen incumbent fixture has "
    "no equivalent row (microcosm#725, #467, #1014)."
)

_UC_PAID_WINDOW_NAMES = frozenset(
    {"dwp.uc.households"}
    | {f"dwp.uc.households_children_{i}" for i in (1, 2, 3, 4, "5_or_more")}
    | {
        f"dwp.uc.households_{family}"
        for family in (
            "couple_no_children",
            "couple_with_children",
            "single_no_children",
            "single_with_children",
        )
    }
)

_ONS_TERMINAL_BAND_RATIONALES = {
    "ons.population.female_85_89": (
        "María ruling 2026-08-21: incumbent ons/female_85_90 maps to the "
        "uniform female 85_89 declaration. The value gap is the single-age-90 "
        "share; ages 90+ are constrained separately."
    ),
    "ons.population.male_85_89": (
        "María ruling 2026-08-21: incumbent ons/male_85_90 maps to the "
        "uniform male 85_89 declaration. The value gap is the single-age-90 "
        "share; ages 90+ are constrained separately."
    ),
    "ons.population.female_90_plus": (
        "María ruling 2026-08-21: new female 90_plus tail constrains ages 90+; "
        "the incumbent six-year terminal band left ages 91+ unconstrained."
    ),
    "ons.population.male_90_plus": (
        "María ruling 2026-08-21: new male 90_plus tail constrains ages 90+; "
        "the incumbent six-year terminal band left ages 91+ unconstrained."
    ),
}

_ONS_15_19_BAND_SPLIT_FIXTURE_ONLY_RATIONALE = (
    "microcosm#1095 (María's ruling of 2026-10-13; uk-data#563): the five-year "
    "15_19 sex band is split into ages 15, 16-17 and 18-19, because a fit to the "
    "band left the ages inside it free to drift. The three rows partition this "
    "row's population and read the same ONS mid-year estimate by single year of "
    "age, so they sum to it. No single current row has this definition, so the "
    "frozen fixture's row is fixture-only on the current surface."
)
_ONS_15_19_BAND_SPLIT_LEDGER_ONLY_RATIONALE = (
    "microcosm#1095 (María's ruling of 2026-10-13; uk-data#563): one of the three "
    "rows (ages 15, 16-17 and 18-19) that partition the incumbent five-year 15_19 "
    "sex band, bound so the fit holds the age mix inside the band. The incumbent "
    "binds only the band, so this row has no fixture counterpart."
)
_ONS_15_19_BAND_ROWS = frozenset(
    f"ons.population.{sex}_15_19" for sex in ("female", "male")
)
_ONS_15_19_BAND_SPLIT_ROWS = frozenset(
    f"ons.population.{sex}_{ages}"
    for sex in ("female", "male")
    for ages in ("15", "16_17", "18_19")
)
_ONS_SINGLE_YEAR_AGE_LEDGER_ONLY_RATIONALE = (
    "microcosm#1278 (the pre-Budget PR's plan, approved by María on 2026-10-20; "
    "microcosm#1095): one of the three all-persons rows for ages 0, 1 and 2, bound "
    "so the fit holds the single years inside the 0_4 sex bands, which stay bound "
    "beside them. The incumbent binds only the five-year bands, so this row has no "
    "fixture counterpart."
)
_ONS_SINGLE_YEAR_AGE_ROWS = frozenset(f"ons.population.age_{age}" for age in (0, 1, 2))
_ONS_SINGLE_YEAR_AGE_0_6_LEDGER_ONLY_RATIONALE = (
    "Requested by María on 2026-10-21 for the £100k childcare costing fixes "
    "(item 6; microcosm#1278, #1095): one of the eleven all-persons rows that "
    "extend the single years of age, the UK's ages 3 to 6 and England's ages 0 to "
    "6, bound so the fit holds the single years inside the 0_4 and 5_9 sex bands "
    "and the region 0_9 cells, which stay bound beside them. The incumbent binds "
    "only the bands, so this row has no fixture counterpart."
)
_ONS_SINGLE_YEAR_AGE_0_6_ROWS = frozenset(
    (
        *(f"ons.population.age_{age}" for age in (3, 4, 5, 6)),
        *(f"ons.population.england_age_{age}" for age in range(7)),
    )
)

_DFE_SINGLE_AGE_LEDGER_ONLY_RATIONALE = (
    "Requested by María on 2026-10-21 for the £100k childcare costing fixes "
    "(microcosm#1278, #1095): one of the five DfE funded-childcare counts by "
    "single age from the January 2026 census, England (working parent "
    "registrations at 9 to 11 months and at ages 1, 2 and 3, and universal "
    "nursery registrations at age 3), bound beside the working parent 2 to 4, "
    "early learning and universal-only rows and, with them, measured at the "
    "spring term by the age on 31 December (María's ruling of 2026-10-22), the "
    "model year 2025's spring term being January 2026 (microcosm#1095 D5). The "
    "age-4 cells are signed out, as the bound rows fix them. The incumbent binds "
    "no DfE row, so this row has no fixture counterpart."
)
_DFE_SINGLE_AGE_ROWS = frozenset(
    (
        "dfe.funded_childcare.working_parent_children_9_to_11_months",
        *(
            f"dfe.funded_childcare.working_parent_children_age_{age}"
            for age in (1, 2, 3)
        ),
        "dfe.funded_childcare.universal_nursery_children_age_3",
    )
)

_DEVOLVED_RENT_FIXTURE_ONLY_RATIONALE = (
    "Ruled signed exclusion: the incumbent y-values for devolved private rent "
    "are hardcoded constants in devolved_housing.py with no source or year, "
    "then allocated by population share. Adopting them would violate the "
    "facts-only doctrine; a Chronicle facts request covers sourced Scottish "
    "Government and StatsWales private-rent statistics."
)

_COUNCIL_TAX_BAND_DRIFT_RATIONALE = (
    "Signed basis translation (microcosm#929, ruled by María on 2026-09-11 and "
    "2026-09-15): the incumbent binds the VOA CTSOP valuation-list stock, every "
    "banded dwelling including exempt, empty and second homes; Microcosm binds "
    "the MHCLG Council Taxbase return (chronicle#264), CTB line 7 chargeable "
    "dwellings after the disabled-relief adjustment (band A adds the A- "
    "column) minus line 11 second homes minus line 15 empty dwellings, the "
    "dwellings a household file can carry. England's valuation list runs "
    "6.3 % over the household rows, so the taxbase cells sit below the VOA "
    "cells by roughly that margin, authority by authority (Westminster band A "
    "1,660 VOA against 1,328 taxbase). Receipts: "
    "experiments/929-council-tax-taxbase-receipts.md Parts C and D."
)

_COUNCIL_TAX_BAND_FIXTURE_ONLY_RATIONALE = (
    "Signed deferrals on the taxbase basis (microcosm#929): the incumbent's "
    "English band-H cells stay deferred under "
    "council_tax_band_h_spine_support_absent (microcosm#762 A14; 296 "
    "authorities), and the City of London and Isles of Scilly cells sit under "
    "the local-authority support floor. Scotland and Wales now bind their own "
    "returns, so no coverage gap remains on the incumbent's rows."
)

_COUNCIL_TAX_SCOTLAND_LEDGER_ONLY_RATIONALE = (
    "No incumbent counterpart: the incumbent bound VOA local rows for England "
    "only. Microsm binds the 32 Scottish councils' CTAXBASE chargeable "
    "dwellings by band A-H (chronicle#264, the concept the "
    "scotgov.council_tax_stock country rows already bind; Shetland band H is "
    "signed deferred for support), microcosm#929."
).replace("Microsm", "Microcosm")

_COUNCIL_TAX_WALES_LEDGER_ONLY_RATIONALE = (
    "No incumbent counterpart: the incumbent bound VOA local rows for England "
    "only, and Wales alone has a band I. Microcosm binds the 22 Welsh "
    "authorities' StatsWales CT1 chargeable dwellings by band A-I (a1 minus "
    "h7 empty minus h8 second homes; chronicle#264), microcosm#929."
)

_MHCLG_REGION_BASIS_RATIONALE = (
    "Signed basis translation (microcosm#929): the incumbent's "
    "voa/council_tax/<REGION>/<band> row is the VOA valuation-list stock, all "
    "banded dwellings; Microcosm's region cell is composed from the 296 "
    "billing authorities' MHCLG Council Taxbase rows (CTB line 7 plus A- for "
    "band A, minus lines 11 and 15: occupied chargeable dwellings, "
    "chronicle#264) and sums to the publisher's England row to the unit. The "
    "valuation list runs 6.3 % over England's household rows, so the taxbase "
    "cell sits below the VOA value (North East band A 667,540 VOA against "
    "621,096 taxbase). Ruled by María 2026-09-11 / 2026-09-15; receipts "
    "experiments/929-council-tax-taxbase-receipts.md."
)

_WELSHGOV_COUNTRY_BASIS_RATIONALE = (
    "Signed basis translation (microcosm#929): the incumbent's "
    "voa/council_tax/WALES/<band> row is the VOA valuation-list stock; "
    "Microcosm binds the StatsWales CT1 2025-26 return's chargeable dwellings "
    "by band A-I minus h7 empty and h8 second homes (chronicle#264; Wales A "
    "215,380 VOA against 194,973 taxbase). Ruled by María 2026-09-11 / "
    "2026-09-15; receipts experiments/929-council-tax-taxbase-receipts.md."
)

_WELSH_BAND_I_LEDGER_ONLY_RATIONALE = (
    "No incumbent counterpart: Wales alone has a council tax band I and the "
    "incumbent's VOA rows stop at band H; Microcosm binds welshgov.council_tax_stock.band_i "
    "from the StatsWales CT1 return (microcosm#929)."
)

_COUNCIL_TAX_NET_FIXTURE_ONLY_RATIONALE = (
    "Signed source deferral: the pinned feed carries no comparable 2025 "
    "full-roster local-authority council-tax net series. The incumbent fixture "
    "row remains non-binding until Chronicle supplies the publisher facts."
)

_DEVOLVED_RENT_METRICS = frozenset(
    {
        "housing/wales_private_renter_households",
        "housing/wales_private_rent_amount",
        "housing/scotland_private_renter_households",
        "housing/scotland_private_rent_amount",
    }
)

_COUNCIL_TAX_BAND_METRICS = frozenset(
    {f"voa/council_tax/{band}" for band in "ABCDEFGHI"}
    | {f"council_tax/band_{band.lower()}" for band in "ABCDEFGHI"}
)

_UC_METRICS = frozenset(
    {
        "uc_households",
        "uc_hh_0_children",
        "uc_hh_1_child",
        "uc_hh_2_children",
        "uc_hh_3plus_children",
    }
)

_UC_CHILDREN_METRICS = _UC_METRICS - {"uc_households"}

# Cite the issue in the repo's short form throughout: the live tree may not
# name the retired data package (test_us_plan's incumbent-reference guard).
# The fixture is the frozen incumbent surface and ours the Ledger compile.
# Each UC drift rationale quotes the spread of fixture over ours across the
# rows it covers, computed from those rows when the receipt is written
# (_uc_drift_rationales), so the quoted range moves with the values.
_UC_TOTALS_DRIFT_RATIONALE = (
    "Fix-and-signed (Maria's ruling, PR #795): the fixture corrects "
    "uk-data#468 at extraction - the incumbent attaches local UC counts by "
    "row position against unrelated orderings, so its live y-frame is a "
    "permutation; the fixture joins the same publisher counts to their areas "
    "by name instead. The residual per-area gap is one of basis: the "
    "incumbent scales every area's count by one national payment-distribution "
    "factor, while ours counts each area's own paid claims (Stat-Xplore's "
    "Payment Indicator 'Yes', bound since the abf0921 re-pin, microcosm#1300), "
    "so fixture/ours moves with each area's paid share (P5-P95 {spread} "
    "across the {rows} rows), plus extract-vintage noise."
)

_UC_CHILDREN_DRIFT_RATIONALE = (
    "Fix-and-signed (Maria's ruling, PR #795), imputation-vs-published "
    "class: on top of the uk-data#468 alignment correction, the incumbent's "
    "child splits are a COUNTRY-share imputation (GB proportions applied to "
    "each area's nationally rescaled total), while ours are the published "
    "Stat-Xplore per-area child-count buckets of each area's paid claims "
    "(Payment Indicator 'Yes' since the abf0921 re-pin, microcosm#1300). The "
    "per-area scatter (P5-P95 fixture/ours {spread} across the {rows} rows) "
    "is the real geographic variation in family size and in paid shares that "
    "the imputation and the national rescale flatten."
)

_UC_NI_FIXTURE_ONLY_RATIONALE = (
    "Signed deferral: DWP Stat-Xplore publishes UC for Great Britain only, so "
    "the Northern Ireland areas have no UC facts in the pinned feed. The "
    "incumbent fills NI from the DfC May-2025 supplementary tables, which are "
    "requested as a Chronicle port in chronicle#200; the child splits stay "
    "deferred regardless (no NI child-count source; the incumbent imputes them "
    "from GB proportions)."
)

_AGE_DRIFT_RATIONALE = (
    "Vintage and scaling class: ours holds the mid-2024 per-area publisher "
    "estimates at identity; the incumbent scales its age columns by "
    "uk_total_population / targets_total_pop * 0.9 (a bare literal) and, at "
    "constituency grain, boundary-maps 2010-vintage rows to PCON24."
)

_HMRC_DRIFT_RATIONALE = (
    "Scaling and operation class: both sides read the same SPI tables for tax "
    "year 2023-24; ours resolves counts at identity and amounts as the "
    "declared count_x_mean operation, while the incumbent multiplies every "
    "column by a national income-projection ratio for the target year."
)

_TENURE_DRIFT_RATIONALE = (
    "Vintage class: ours holds the per-nation census tenure counts (E&W 2021, "
    "Scotland 2022, NI 2021) at identity with uprating declared as holds; the "
    "incumbent derives counts as percentage shares times a separate "
    "household-count workbook."
)

_EQUIV_INCOME_FIXTURE_ONLY_RATIONALE = (
    "Kept out: ONS models its small-area income estimates (MSOA means, before "
    "and after housing costs, and housing costs) from the FRS, so binding them "
    "would partly fit the FRS to itself; the feed also carries no "
    "local-authority rows (María's ruling of 2026-10-29, microcosm#1095 D18a)."
)

_PRIVATE_RENT_FIXTURE_ONLY_RATIONALE = (
    "Signed deferral: PIPR publishes no rent for these authorities. Scotland's "
    "rents are published for its 18 Broad Rental Market Areas only and bind at "
    "that grain (ons.rent.private_rent_brma, microcosm#1090, #1262); Northern "
    "Ireland has a country mean only; and two English authorities are not "
    "published. The incumbent's rows here are not published cells."
)

#: microcosm#1262: Scotland's PIPR rents bind at the 18 BRMAs, a grain the
#: incumbent never fitted.
_PRIVATE_RENT_BRMA_LEDGER_ONLY_RATIONALE = (
    "ledger_only by construction: PIPR publishes Scotland's private rents for "
    "its 18 Broad Rental Market Areas only, and Microcosm binds them at that "
    "grain (ons.rent.private_rent_brma, microcosm#1090, #1262); the incumbent "
    "fits no BRMA rows, so it carries no row to compare. The value is the "
    "compiled calendar-2025 mean, which the bound total composes with the "
    "BRMA's private renters."
)

#: microcosm#1095: DESNZ publishes no constituency cut, so a seat's domestic
#: energy cell is Microcosm's split of its authorities' totals.
_DESNZ_SEAT_SPLIT_LEDGER_ONLY_RATIONALE = (
    "ledger_only by construction: DESNZ publishes no constituency cut, so the "
    "cell is Microcosm's split of its authorities' published totals by DESNZ's "
    "LSOA and data zone figures for the same year "
    "(uk/desnz_constituency_energy_facts.json, microcosm#1095), compiled beside "
    "the Chronicle feed; the incumbent fits no domestic energy rows, so it "
    "carries no row to compare."
)
#: GSS prefixes of Great Britain's July 2024 constituencies.
_GB_CONSTITUENCY_PREFIXES = ("E14", "W07", "S14")

_LEDGER_ONLY_RATIONALE = (
    "Coverage the incumbent lacks: our side compiles a published per-area fact "
    "the incumbent's surface NaN-masks or never carries (for example Northern "
    "Ireland SPI and tenure rows). Confirmed by roster diff (PR #795 round-3 "
    "review): the incumbent's 360-row LA file carries N09000001-N09000010 "
    "only, omitting N09000011 (Newry, Mourne and Down) entirely - the 361st "
    "crosswalk area is a genuine incumbent roster gap, not a dropped row."
)

_CENSUS_HOUSEHOLDS_LEDGER_ONLY_RATIONALE = (
    "Coverage the incumbent lacks after microcosm#887: the contract binds the "
    "published disclosure-controlled census household cell compiled from the "
    "pinned Chronicle feed; the retired fixture supplied this family from OA-"
    "ladder sums outside the contract instead."
)

#: The UC metrics' drift rationales quote their rows' spread, so they are
#: formatted per receipt (_uc_drift_rationales) rather than declared here.
_LOCAL_DRIFT_RATIONALES = {
    **{f"age/{lower}_{lower + 10}": _AGE_DRIFT_RATIONALE for lower in range(0, 80, 10)},
    **{
        f"hmrc/{variable}/{measure}": _HMRC_DRIFT_RATIONALE
        for variable in ("employment_income", "self_employment_income")
        for measure in ("amount", "count")
    },
    **{
        f"tenure/{key}": _TENURE_DRIFT_RATIONALE
        for key in ("owned_outright", "owned_mortgage", "private_rent", "social_rent")
    },
}

_SPI_CELL_FIXTURE_ONLY_RATIONALE = (
    "Signed deferral: the publisher's SPI area tables lack the cell upstream — "
    "E14001416 publishes a self-employment count with no mean (so count_x_mean "
    "cannot resolve), and two local authorities carry no SPI target measures. "
    "The incumbent's fixture row for these cells is a boundary-mapped blend, "
    "not a published cell."
)

_LOCAL_FIXTURE_ONLY_RATIONALES = {
    **{metric: _UC_NI_FIXTURE_ONLY_RATIONALE for metric in _UC_METRICS},
    **{
        f"hmrc/{variable}/{measure}": _SPI_CELL_FIXTURE_ONLY_RATIONALE
        for variable in ("employment_income", "self_employment_income")
        for measure in ("amount", "count")
    },
    **{
        f"ons/{key}": _EQUIV_INCOME_FIXTURE_ONLY_RATIONALE
        for key in (
            "equiv_net_income_bhc",
            "equiv_net_income_ahc",
            "equiv_housing_costs",
        )
    },
    "rent/private_rent": _PRIVATE_RENT_FIXTURE_ONLY_RATIONALE,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help="Chronicle consumer JSONL artifact used to compile UK references.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UK_PACKAGE_DIR,
        help="Directory receiving the signed-difference JSON resources.",
    )
    parser.add_argument(
        "--surface",
        choices=("all", "national", "local"),
        default="all",
        help="Receipt surface to regenerate. Use 'local' to avoid national receipt churn.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    # Stream the lines: the feed is several GB, and reading it whole held its
    # text and every line beside the parsed rows. Both compiles read every
    # row, so the parsed rows are still kept.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_fixture(resource: str) -> dict:
    return json.loads(
        importlib_resources.files(UK_PACKAGE).joinpath(resource).read_text()
    )


def _load_crosswalk() -> dict:
    return json.loads(
        importlib_resources.files(UK_PACKAGE)
        .joinpath("local_area_crosswalk.json")
        .read_text()
    )


def _local_metric_by_target_id() -> dict[str, str]:
    contract = json.loads(
        importlib_resources.files(UK_PACKAGE)
        .joinpath("uk_population_targets.json")
        .read_text()
    )
    mapping: dict[str, str] = {}
    for target in contract.get("targets", ()):
        binding = target.get("bindings", {}).get("policyengine", {})
        metric = binding.get("metric_name")
        if metric:
            mapping[str(target["target_id"])] = str(metric)
    return mapping


def _aligned_fixture(fixture: dict) -> dict:
    """Apply current contract-row names to retired-fixture rows before comparison."""

    rows = []
    for row in fixture.get("rows", ()):
        if not isinstance(row, dict):
            rows.append(row)
            continue
        updated = dict(row)
        if updated.get("name") == "ons/female_85_90":
            updated["contract_target_id"] = "ons.population.female_85_89"
            updated["measure"] = "ons.population.female_85_89"
        elif updated.get("name") == "ons/male_85_90":
            updated["contract_target_id"] = "ons.population.male_85_89"
            updated["measure"] = "ons.population.male_85_89"
        rows.append(updated)
    aligned = dict(fixture)
    aligned["rows"] = rows
    return aligned


_REGION_TIER_ONS_DRIFT_RATIONALE = (
    "Region-tier ONS population cell (microcosm#905): the Ledger row binds the "
    "ONS mid-year estimate by single year of age at Chronicle's region or "
    "country stamp, held to the comparison period as identity, while the "
    "incumbent's regional row is an ONS subnational projection rounded to the "
    "nearest thousand. The difference is projection-versus-estimate vintage "
    "and rounding, never area or band definition: the nine English regions "
    "and three nations sum exactly to the retired UK-wide row of the same "
    "publication."
)


#: microcosm#1301: the region rows of the high-value homes family (the high
#: value council tax surcharge, HVCTS) bind the same Microcosm-derived estimate
#: as its local cells, on the local line only, so the incumbent national
#: register has nothing to compare them with either.
_HIGH_VALUE_HOMES_REGION_PREFIX = "microcosm.high_value_homes."
_HIGH_VALUE_HOMES_REGION_LEDGER_ONLY_RATIONALE = (
    "ledger_only by construction: the row is a Microcosm-derived estimate of "
    "the owner-occupied dwelling stock above a high value council tax "
    "surcharge (HVCTS) threshold in an English region or Wales "
    "(uk/high_value_home_stock_facts.json, fence "
    "high_value_home_stock_derived_estimate), compiled beside the Chronicle "
    "feed. No publisher counts such dwellings by region and the incumbent "
    "registry carries no row to compare, so the difference is the derived "
    "target itself. The row is the cross-grain control for the family's "
    "constituency and local-authority cells and binds on the local line only "
    "(calibration_lines; microcosm#1301)."
)


def _add_signed_rationale_notes(
    report: dict[str, object],
    *,
    fixture_resource: str,
) -> None:
    if fixture_resource != "registry_parity_fixture_2025.json":
        return
    for row in report.get("differences", ()):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        if (
            name == "dwp.uc.households_children_5_or_more"
            and row.get("kind") == "fixture_only"
        ):
            row["reason"] = _DWP_UC_FIVE_OR_MORE_DIAGNOSTIC_RATIONALE
        elif (
            name.startswith(_HIGH_VALUE_HOMES_REGION_PREFIX)
            and row.get("kind") == "ledger_only"
        ):
            row["reason"] = _HIGH_VALUE_HOMES_REGION_LEDGER_ONLY_RATIONALE
        elif name in _UC_PAID_WINDOW_NAMES:
            inherited_change = (
                "PR #891 contract with #892 source-window averaging"
                if name.startswith("dwp.uc.households_children_")
                else "PR #891 contract"
            )
            row["reason"] = (
                f"Inherited {inherited_change}: current paid-UC counts use the "
                "explicit January-December 2025 source windows on the ec7169 "
                "national feed. This compares that declaration with the frozen "
                "2025 incumbent baseline; it does not redatum the observations to the comparison "
                "year or create a new year ruling."
            )
        elif "_by_region@" in name and row.get("kind") == "calibration_drift":
            row["reason"] = _REGION_TIER_ONS_DRIFT_RATIONALE
        elif (
            name.startswith("mhclg.council_tax_stock.")
            and row.get("kind") == "calibration_drift"
        ):
            row["reason"] = _MHCLG_REGION_BASIS_RATIONALE
        elif (
            name.startswith("welshgov.council_tax_stock.")
            and row.get("kind") == "calibration_drift"
        ):
            row["reason"] = _WELSHGOV_COUNTRY_BASIS_RATIONALE
        elif name.startswith("welshgov.council_tax_stock.band_i"):
            row["reason"] = _WELSH_BAND_I_LEDGER_ONLY_RATIONALE
        elif name == "obr.state_pension" and row.get("kind") == "fixture_only":
            row["reason"] = _OBR_STATE_PENSION_DIAGNOSTIC_RATIONALE
        elif name == "obr.ni" and row.get("kind") == "fixture_only":
            row["reason"] = _OBR_NI_DIAGNOSTIC_RATIONALE
        elif name in _OBR_UC_CAP_SPLIT_ROWS and row.get("kind") == "fixture_only":
            row["reason"] = _OBR_UC_CAP_SPLIT_RETIRED_RATIONALE
        elif (
            name == "slc.repayments.england_postgraduate"
            and row.get("kind") == "fixture_only"
        ):
            row["reason"] = _SLC_POSTGRADUATE_RETIRED_RATIONALE
        elif (
            name in _DWP_PIP_CASELOAD_DIAGNOSTIC_ROWS
            and row.get("kind") == "fixture_only"
        ):
            row["reason"] = _DWP_PIP_CASELOAD_DIAGNOSTIC_RATIONALE
        elif (
            name.startswith(_STATE_PENSION_PREFIXES)
            and row.get("kind") == "ledger_only"
        ):
            row["reason"] = _STATE_PENSION_LEDGER_ONLY_RATIONALE
        elif fixture_resource != "registry_parity_fixture_2025.json":
            continue
        elif name in _CGT_OBSERVED_RATIONALES:
            row["reason"] = _CGT_OBSERVED_RATIONALES[name]
        elif (
            name.startswith("hmrc/property_income_income_band_")
            and row.get("kind") == "calibration_drift"
        ):
            row["reason"] = _SPI_PROPERTY_AMOUNT_DRIFT_RATIONALE
        elif (
            name.startswith(_SPI_STATE_PENSION_BAND_PREFIXES)
            and row.get("kind") == "calibration_drift"
        ):
            row["reason"] = _SPI_STATE_PENSION_UPRATED_DRIFT_RATIONALE
        elif (
            name.startswith("hmrc/")
            and "_income_band_" in name
            and row.get("kind") == "calibration_drift"
        ):
            row["reason"] = _SPI_UPRATED_DRIFT_RATIONALE
        elif name.startswith(_ITL_AGE_PREFIXES) and row.get("kind") == "ledger_only":
            row["reason"] = _ITL_AGE_LEDGER_ONLY_RATIONALE
        elif (
            name.startswith(_ITL_SAVINGS_TAX_PREFIX)
            and row.get("kind") == "ledger_only"
        ):
            row["reason"] = _ITL_SAVINGS_TAX_LEDGER_ONLY_RATIONALE
        elif (
            name.startswith(_INCOME_ANCHOR_PREFIXES)
            and row.get("kind") == "ledger_only"
        ):
            row["reason"] = _INCOME_ANCHOR_LEDGER_ONLY_RATIONALE
        elif name in _ESA_CUBE_ROWS and row.get("kind") == "calibration_drift":
            row["reason"] = _ESA_CUBE_DRIFT_RATIONALE
        elif name.startswith("hmrc.spi_region.") and row.get("kind") == "ledger_only":
            row["reason"] = _SPI_REGION_LEDGER_ONLY_RATIONALE
        elif name == "obr.council_tax" and row.get("kind") == "calibration_drift":
            row["reason"] = _OBR_COUNCIL_TAX_GB_RECEIPTS_RATIONALE
        elif name == "obr.sdlt" and row.get("kind") == "calibration_drift":
            row["reason"] = _OBR_SDLT_HOUSEHOLDS_RATIONALE
        elif (
            name.startswith("obr.")
            and row.get("kind") == "calibration_drift"
            and name not in _CGT_OBSERVED_RATIONALES
        ):
            row["reason"] = _OBR_WINDOW_DRIFT_RATIONALE
        elif (
            name.startswith(_CGT_BAND_INCUMBENT_PREFIXES)
            and row.get("kind") == "calibration_drift"
        ):
            row["reason"] = _CGT_BAND_DRIFT_RATIONALE
        elif (
            name.startswith(_CGT_NEW_ROW_PREFIXES) and row.get("kind") == "ledger_only"
        ):
            row["reason"] = _CGT_NEW_ROW_LEDGER_ONLY_RATIONALE
        elif name in _ONS_TERMINAL_BAND_RATIONALES:
            row["reason"] = _ONS_TERMINAL_BAND_RATIONALES[name]
        elif name in _ONS_15_19_BAND_ROWS and row.get("kind") == "fixture_only":
            row["reason"] = _ONS_15_19_BAND_SPLIT_FIXTURE_ONLY_RATIONALE
        elif name in _ONS_15_19_BAND_SPLIT_ROWS and row.get("kind") == "ledger_only":
            row["reason"] = _ONS_15_19_BAND_SPLIT_LEDGER_ONLY_RATIONALE
        elif name in _ONS_SINGLE_YEAR_AGE_ROWS and row.get("kind") == "ledger_only":
            row["reason"] = _ONS_SINGLE_YEAR_AGE_LEDGER_ONLY_RATIONALE
        elif name in _ONS_SINGLE_YEAR_AGE_0_6_ROWS and row.get("kind") == "ledger_only":
            row["reason"] = _ONS_SINGLE_YEAR_AGE_0_6_LEDGER_ONLY_RATIONALE
        elif name in _DFE_SINGLE_AGE_ROWS and row.get("kind") == "ledger_only":
            row["reason"] = _DFE_SINGLE_AGE_LEDGER_ONLY_RATIONALE


#: microcosm#1240: the high_value_homes family binds a Microcosm-derived
#: estimate (HM Land Registry stock counts), not a published per-area fact,
#: so the incumbent register has nothing to compare it with.
_DERIVED_STOCK_LEDGER_ONLY_RATIONALE = (
    "ledger_only by construction: the target is a Microcosm-derived estimate "
    "of the owner-occupied dwelling stock above a value threshold "
    "(uk/high_value_home_stock_facts.json, fence "
    "high_value_home_stock_derived_estimate), compiled beside the Chronicle "
    "feed; no publisher counts such dwellings by area, so the incumbent "
    "registry carries no row to compare and the difference is the derived "
    "target itself (microcosm#1240)."
)


def _small_cell_deferral_rationales() -> dict[tuple[str, str], str]:
    """(target id, area id) -> the signed rationale of each small-cell deferral.

    A cell the incumbent carries but a signed small-cell rule defers is
    fixture-only here; its rationale is the one its deferral was signed with,
    which names the rule (microcosm#1278).
    """

    def committed(resource: str) -> dict:
        return json.loads(
            importlib_resources.files(UK_PACKAGE)
            .joinpath(resource)
            .read_text(encoding="utf-8")
        )

    # The support floor's reviewed exclusions defer their areas' cells as
    # signed deferrals too; their rationale, not a per-metric one, is the cause
    # (#1305 review: two English UC cells read as Northern Ireland's).
    rule_ids = {
        str(rule["rule_id"])
        for rule in committed("local_small_cell_rule.json")["rules"]
    } | {"local_authority_support_floor_excluded"}
    return {
        (str(deferral["target_id"]), str(area)): str(deferral["rationale"])
        for deferral in committed("local_target_reference_membership.json")[
            "signed_deferrals"
        ]
        if deferral["reason_id"] in rule_ids
        for area in deferral["area_ids"]
    }


def _fixture_over_ours_spread(
    rows: list[dict[str, object]],
) -> dict[str, str] | None:
    """P5-P95 of fixture over ours across ``rows``, to three decimals."""

    ratios = [
        float(row["fixture_value"]) / float(row["ledger_value"])
        for row in rows
        if float(row["ledger_value"]) != 0.0
    ]
    if not ratios:
        return None
    low, high = np.percentile(ratios, [5, 95])
    return {"spread": f"{low:.3f}-{high:.3f}", "rows": f"{len(ratios):,}"}


def _uc_drift_rationales(
    report: dict[str, object], metric_by_target_id: dict[str, str]
) -> dict[str, str]:
    """Metric -> the UC drift rationale, quoting the spread of the rows it covers.

    The totals rationale covers the ``uc_households`` drift rows and the
    children rationale the four child bands' drift rows together.
    """

    totals: list[dict[str, object]] = []
    children: list[dict[str, object]] = []
    for row in report.get("differences", ()):
        if not isinstance(row, dict) or row.get("kind") != "calibration_drift":
            continue
        prefix = str(row.get("name", "")).partition("@")[0]
        metric = metric_by_target_id.get(prefix, prefix)
        if metric == "uc_households":
            totals.append(row)
        elif metric in _UC_CHILDREN_METRICS:
            children.append(row)
    rationales: dict[str, str] = {}
    totals_spread = _fixture_over_ours_spread(totals)
    if totals_spread is not None:
        rationales["uc_households"] = _UC_TOTALS_DRIFT_RATIONALE.format(**totals_spread)
    children_spread = _fixture_over_ours_spread(children)
    if children_spread is not None:
        rationales.update(
            dict.fromkeys(
                _UC_CHILDREN_METRICS,
                _UC_CHILDREN_DRIFT_RATIONALE.format(**children_spread),
            )
        )
    return rationales


def _add_local_signed_rationale_notes(report: dict[str, object]) -> None:
    """Replace generic classifier reasons with the ruled per-family rationales.

    Every rationale names its cause (an adjudicated exclusion, a signed
    deferral, a diagnosed incumbent defect, or a declared scaling/vintage
    class). A row whose (metric, kind) has no ruled rationale keeps the
    classifier's generic reason — never invent one.
    """

    metric_by_target_id = _local_metric_by_target_id()
    small_cells = _small_cell_deferral_rationales()
    drift_rationales = {
        **_LOCAL_DRIFT_RATIONALES,
        **_uc_drift_rationales(report, metric_by_target_id),
    }
    for row in report.get("differences", ()):
        if not isinstance(row, dict):
            continue
        prefix, _, area = str(row.get("name", "")).partition("@")
        metric = metric_by_target_id.get(prefix, prefix)
        kind = str(row.get("kind", ""))
        if kind == "fixture_only" and (prefix, area) in small_cells:
            row["reason"] = small_cells[(prefix, area)]
        elif metric in _DEVOLVED_RENT_METRICS:
            row["reason"] = _DEVOLVED_RENT_FIXTURE_ONLY_RATIONALE
        elif kind == "calibration_drift" and metric in _COUNCIL_TAX_BAND_METRICS:
            row["reason"] = _COUNCIL_TAX_BAND_DRIFT_RATIONALE
        elif kind == "fixture_only" and metric in _COUNCIL_TAX_BAND_METRICS:
            row["reason"] = _COUNCIL_TAX_BAND_FIXTURE_ONLY_RATIONALE
        elif kind == "ledger_only" and prefix.startswith("scotgov.council_tax_stock."):
            row["reason"] = _COUNCIL_TAX_SCOTLAND_LEDGER_ONLY_RATIONALE
        elif kind == "ledger_only" and prefix.startswith("welshgov.council_tax_stock."):
            row["reason"] = _COUNCIL_TAX_WALES_LEDGER_ONLY_RATIONALE
        elif metric == "housing/council_tax_net":
            row["reason"] = _COUNCIL_TAX_NET_FIXTURE_ONLY_RATIONALE
        elif kind == "calibration_drift" and metric in drift_rationales:
            row["reason"] = drift_rationales[metric]
        elif kind == "fixture_only" and metric in _LOCAL_FIXTURE_ONLY_RATIONALES:
            row["reason"] = _LOCAL_FIXTURE_ONLY_RATIONALES[metric]
        elif kind == "ledger_only" and metric == "households":
            row["reason"] = _CENSUS_HOUSEHOLDS_LEDGER_ONLY_RATIONALE
        elif kind == "ledger_only" and metric.startswith("high_value_homes/"):
            row["reason"] = _DERIVED_STOCK_LEDGER_ONLY_RATIONALE
        elif kind == "ledger_only" and metric == "rent/private_rent_brma":
            row["reason"] = _PRIVATE_RENT_BRMA_LEDGER_ONLY_RATIONALE
        elif (
            kind == "ledger_only"
            and metric.startswith("energy/")
            and area.startswith(_GB_CONSTITUENCY_PREFIXES)
        ):
            row["reason"] = _DESNZ_SEAT_SPLIT_LEDGER_ONLY_RATIONALE
        elif kind == "ledger_only":
            row["reason"] = _LEDGER_ONLY_RATIONALE


def _compile_for_receipt(spec: ParityReceiptSpec, facts: list[dict]):
    if spec.surface == "national":
        # The high-value homes region rows (microcosm#1301) select the
        # derived facts too; the national compile carries every reference.
        return compile_uk_target_registry(
            (*facts, *uk_derived_fact_rows()),
            target_period=spec.target_period,
        )
    if spec.surface == "local":
        return compile_uk_local_target_registry(
            (*facts, *uk_derived_fact_rows()),
            target_period=spec.target_period,
            crosswalk=_load_crosswalk(),
        )
    raise ValueError(f"unknown parity receipt surface {spec.surface!r}.")


def _fixture_for_receipt(spec: ParityReceiptSpec) -> dict:
    fixture = _aligned_fixture(_load_fixture(spec.fixture_resource))
    if spec.surface == "local":
        return align_uk_local_registry_parity_fixture(fixture)
    return align_uk_national_registry_parity_fixture(fixture)


def main() -> None:
    args = _parse_args()
    facts = _load_jsonl(args.ledger_facts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in RECEIPTS:
        if args.surface != "all" and spec.surface != args.surface:
            continue
        compilation = _compile_for_receipt(spec, facts)
        report = ledger_compile_parity_signed_differences(
            compilation.registry,
            _fixture_for_receipt(spec),
            unsupported=compilation.unsupported,
        )
        _add_signed_rationale_notes(report, fixture_resource=spec.fixture_resource)
        if spec.surface == "local":
            _add_local_signed_rationale_notes(report)
        output_path = args.output_dir / spec.output_resource
        output_path.write_text(
            json.dumps(report, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "compiled_count": report["compiled_count"],
                    "counts_by_kind": report["counts_by_kind"],
                    "difference_count": report["difference_count"],
                    "fixture": spec.fixture_resource,
                    "fixture_count": report["fixture_count"],
                    "output": spec.output_resource,
                    "surface": spec.surface,
                    "target_period": spec.target_period,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
