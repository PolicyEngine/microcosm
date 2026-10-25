"""Current-vintage SPI QRF stages for the UK national support channel."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.stochastic_assignment import (
    integer_record_keys,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.frs_annual_income import (
    ANNUAL_SHARE_COLUMNS,
    EMPLOYMENT_INCOME_OTHER_SPELLS,
    FRS_ANNUAL_INCOME_OUTPUT_COLUMNS,
    SELF_EMPLOYMENT_INCOME_OTHER_SPELLS,
)
from microcosm.build.uk_runtime.frs_disability import (
    UKDWPDisabilityCategoryRates,
    UKDWPDisabilityFlagRates,
)
from microcosm.build.uk_runtime.frs_hmrc_source import (
    FRS_HMRC_INCPBEN_COLUMN,
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
    FRS_HMRC_UBISJA_COLUMN,
)
from microcosm.build.uk_runtime.frs_spine import (
    FRS_LANDLORD_CARRIER_COLUMN,
    uc_gainful_self_employment,
)
from microcosm.build.uk_runtime.hmrc_income import (
    LOSS_RELIEF_COLUMN,
    self_employment_income_after_loss_relief,
)
from microcosm.build.uk_runtime.pension_coherence import (
    cap_net_pay_contributions,
    coherence_receipt,
    twin_rate_employer_contributions,
)
from microcosm.build.uk_runtime.spi_rank_transport import (
    SPI_STAGE1_ANALOGUE_COLUMNS,
    SPI_TRANSPORT_FIT_NAME,
    SPI_TRANSPORT_FOLDS,
    SPI_TRANSPORT_KEY_COLUMN,
    SPI_TRANSPORT_SEED_OFFSET,
    analogue_targets,
    assert_analogue_map_covers,
    cell_fold_ids,
    cross_fit_transport,
    observed_analogues,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS,
    FRS_ONLY_SPI_FILL_OWNER_MANAGER_FLAGS,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_HMRC_EMPLOYED_INCOME_COLUMN,
    SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS,
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_PAY_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
    SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    SPI_LOSS_DRAW_COLUMNS,
    SPI_LOSSES_BROUGHT_FORWARD_DRAW_COLUMN,
    SPI_PROFIT_BEFORE_LOSSES_DRAW_COLUMN,
    SPI_PROPERTY_FINANCE_COSTS_COLUMN,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UKSPISupportResult,
    support_channel_column,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

QRF: Any | None = None

SPI_DONOR_RELEASE = "spi_2022_23"
SPI_DONOR_FILENAME = "put2223uk.tab"
SPI_DONOR_VINTAGE = "2022-23"
SPI_DONOR_UKDS_STUDY = "SN 9422"
SPI_DONOR_DOI = "10.5255/UKDA-SN-9422-1"
SPI_DONOR_DOCUMENTATION_URL = (
    "https://doc.ukdataservice.ac.uk/doc/9422/mrdoc/pdf/"
    "9422_put_2223_full_documentation.pdf"
)
SPI_DONOR_FIT_NAME = "uk_spi_2022_23_income"
FRS_ONLY_FIT_NAME = "uk_frs_only_spi_fill"
DEFAULT_SPI_DONOR_SAMPLE_SIZE = 100_000
# Recipient guard matching the existing constructed SPI age support below.
# SN 9422 Annex A labels the youngest band "Under 25"; it does not establish
# a survey age minimum. Keep observed FRS inputs below the constructed lower
# bound instead of extrapolating this coarsened donor age to younger children.
# FRS dependent-child status can extend to ages 16-19; this guard is an age
# support boundary, not a crosswalk to that household-composition definition.
SPI_MINIMUM_RECIPIENT_AGE = 16
#: The tape samples taxpayers in their own right, so its draws go to the FRS
#: benefit unit's claimant or partner; a dependent child aged 16 to 19 is
#: neither and keeps every value of its FRS twin (uk-data#504, microcosm#1095).
#: The column persists the FRS adult-file role from
#: ``uc_relationships.frs_uc_claimant_mask``.
SPI_RECIPIENT_ROLE_COLUMN = "is_uc_claimant"
#: Person flag the stage writes: True exactly for the rows stage 1 drew from
#: the tape (claimants and partners aged 16 and over with taxable income under
#: the recipient income rule, and the reserved band carriers), False for the
#: undrawn copies, the channel's dependants and every FRS row. A later stage
#: that tells tape-concept rows from FRS-concept ones reads it instead of
#: re-deriving the recipient rule (property_components; microcosm#840).
SPI_INCOME_RECIPIENT_COLUMN = "person_is_spi_income_recipient"
#: The tape's bridge codes (SPI 2022-23 Public Use Tape documentation, Annex
#: A): MAINSRCE "main source indicator", SEINC_NUM "indicator for
#: self-employed cases (those submitting pages in their tax return for income
#: from trades or partnerships)" and SPA "State Pension age indicator".
#: Composite records (-1) carry none and derive theirs from their leaves.
SPI_MAINSRCE_CODES: Mapping[int, str] = {
    -1: "composite",
    1: "pay",
    2: "pension",
    3: "trade",
    4: "trade",
    5: "other",
    6: "other",
}
SPI_MAINSRCE_CODE_LABELS: Mapping[int, str] = {
    -1: "not classified (composite record)",
    1: "main source pay",
    2: "main source occupational pension",
    3: "main source sole trader",
    4: "main source partnership",
    5: "main source other",
    6: "claims case",
}
SPI_INDICATOR_CODES = (-1, 0, 1)
#: The bridge predictors (microcosm#840): four facts about a record's income
#: that the tape and the FRS twin each carry, so a recipient draws from the
#: tape records that share them. main_source and has_pay come from the same
#: leaf formulas on both sides. has_trade and at_or_over_spa take the tape's
#: published indicator codes (SEINC_NUM, self-employment pages filed; SPA, at
#: State Pension age), the leaf formulas serving only a composite record
#: without codes, and on the twin its trade leaf or self-employed status and
#: its age against State Pension age. The published MAINSRCE code validates
#: the leaf-defined main source, and the receipt reports the indicator codes
#: against the leaves both ways.
SPI_MAIN_SOURCE_LEVELS = ("pay", "pension", "trade", "other")
SPI_BRIDGE_PREDICTOR_COLUMNS = ("main_source", "has_pay", "has_trade", "at_or_over_spa")
SPI_STAGE1_CATEGORICAL_PREDICTORS = ("gender", "region", "main_source")
#: The stage-1 forests' age predictor: the SPI AGERANGE band, not an age
#: (uk-data#628). See SPI_STAGE1_AGE_PREDICTOR_RULE.
SPI_STAGE1_AGE_BAND_COLUMN = "spi_age_band"
SPI_STAGE1_PREDICTOR_COLUMNS = (
    SPI_STAGE1_AGE_BAND_COLUMN,
    "gender",
    "region",
    *SPI_BRIDGE_PREDICTOR_COLUMNS,
)
#: The leaf-defined main source must translate the published MAINSRCE code
#: on at least this weighted share of the classified tape records. The stage
#: evidence's bridge_validation measured 0.969 on the full 2022-23 tape with
#: this stage's preparation (pay 0.983, occupational pension 0.987, sole
#: trader 0.851, partnership 0.744, other 0.916; composites 0.002).
SPI_MAIN_SOURCE_MINIMUM_AGREEMENT = 0.95
#: The no-record rule (microcosm#840): the tape holds people with an income
#: tax record, so a twin with no taxable income has no population on it to
#: condition on and its copy keeps the twin's values. Unearned income counts
#: only above this floor, the order of the savings, dividend and property
#: allowances: a declared modelling boundary like the under-16 one.
SPI_RECIPIENT_UNEARNED_INCOME_FLOOR_GBP = 1_000.0
SPI_RECIPIENT_INCOME_RULE = (
    "drawn when the twin reports pay, a trade (self-employment income or a "
    "self-employed main job) or a pension, or more than the floor of taxable "
    "interest (savings interest less the tax-free part), dividends, property "
    "and miscellaneous income together, and always for a reserved band "
    "carrier; otherwise the twin has no tax record for the tape to condition "
    "on and the copy keeps the twin's values at stage 1 and stage 2; a twin "
    "whose only record is the State Pension is drawn, against a tape that "
    "mostly holds taxpayers, which can lift its pension leaves"
)
SPI_DONOR_INCOME_YEAR = 2022
#: The donor rebase moves each SPI component from the tape's year to the
#: frame's survey year. Those years are history, so a component moves by the
#: index its engine variable names (``Variable.uprating``). For every
#: component but savings interest that is also the index policyengine-uk's
#: dataset projection applies from the survey year on (``policyengine_uk/data/
#: uprating_indices.yaml``). Savings interest's own index is ONS household
#: interest income, x2.27 from 2022 to 2024 as rates rose, where the
#: projection's GDP per head gives x1.09. The projection's index applies from
#: the survey year on, where the engine runs it and the SPI band targets follow
#: it (``hmrc_uprating.UK_SPI_SAVINGS_INTEREST_INDEX_CONCEPT``, uk-data#541,
#: microcosm#1095). The engine lockstep test holds both statements.
SPI_HISTORY_INDEX_DIFFERS_FROM_PROJECTION = {
    "savings_interest_income": "gov.economic_assumptions.indices.obr.per_capita.gdp",
}
# Use the pinned engine's variable-specific indices. None explicitly retains
# nominal amounts where no same-scope indexed mapping has been reviewed;
# broader indexed benefit inputs do not establish a taxable SPI crosswalk.
SPI_INCOME_UPRATING_VARIABLES = {
    "self_employment_income": "self_employment_income",
    "savings_interest_income": "savings_interest_income",
    "dividend_income": "dividend_income",
    "private_pension_income": "private_pension_income",
    "property_income": "property_income",
    "other_investment_income": "other_investment_income",
    "gift_aid": None,
    "charitable_investment_gifts": None,
    "hmrc_spi_pay": "employment_income_before_lsr",
    "hmrc_spi_employment_benefits": "employment_income_before_lsr",
    "hmrc_spi_employment_expenses": "employment_income_before_lsr",
    "hmrc_spi_incapacity_benefit_income": None,
    "hmrc_spi_other_social_security_income": None,
    "hmrc_spi_taxable_termination_pay": "employment_income_before_lsr",
    "hmrc_spi_unemployment_benefit_income": None,
    "hmrc_spi_miscellaneous_employment_income": "employment_income_before_lsr",
    "hmrc_spi_other_income": "miscellaneous_income",
    "hmrc_spi_state_pension_income": None,
    SPI_PROPERTY_FINANCE_COSTS_COLUMN: SPI_PROPERTY_FINANCE_COSTS_COLUMN,
    # The year's profit and its losses brought forward move with profits.
    SPI_PROFIT_BEFORE_LOSSES_DRAW_COLUMN: "self_employment_income",
    SPI_LOSSES_BROUGHT_FORWARD_DRAW_COLUMN: "self_employment_income",
}
SPI_DONOR_SHA256 = "5ef829461060c91a2a47be59ad541d9b519fc3976d66ca80d4920f711bb96f66"
SPI_DONOR_SIZE_BYTES = 141_323_762
_SPI_DONOR_VERIFICATION_TOKEN = object()
# The pinned donor's published TI field differs from TEI + TII by at most £5
# because the public-use fields are rounded. Synthetic draws never inherit
# that discrepancy: their accounting aggregates are derived after the draw.
SPI_TI_IDENTITY_ABS_TOLERANCE_GBP = 5.0
# Annex A gives the exact leaf formulas. The pinned PUT first rounds source
# fields, then averages composite records, then rounds remaining income fields
# to £5. The two max() operations do not commute with composite averaging, so
# source-leaf reconciliation has separate observed envelopes for the exact
# sha-pinned ordinary and composite records. Synthetic identities remain exact.
SPI_SOURCE_TEI_FORMULA = (
    "max(0, PAY + EPB - EXPS) + INCPBEN + OSSBEN + TAXTERM + UBISJA + "
    "MOTHINC + OTHERINC + SRP + PENSION + "
    "max(0, PROFITS - CAPALL - LOSSBF)"
)
SPI_SOURCE_TII_FORMULA = "OTHERINV + DIVIDENDS + INCPROP + INCBBS"
SPI_SOURCE_TI_FORMULA = "TEI + TII"
SPI_SOURCE_COMPOSITE_INDICATOR = "AGERANGE == -1"
SPI_SOURCE_LEAF_RECONCILIATION_ABS_TOLERANCE_GBP = {
    "ordinary": {"TEI": 15.0, "TII": 10.0, "TI": 20.0},
    "composite": {"TEI": 180.0, "TII": 10.0, "TI": 180.0},
}

SPI_POLICYENGINE_EMPLOYMENT_SOURCE_COLUMNS = ("PAY", "EPB", "TAXTERM")
SPI_POLICYENGINE_EMPLOYMENT_FORMULA = (
    "hmrc_spi_pay + hmrc_spi_employment_benefits + hmrc_spi_taxable_termination_pay"
)
SPI_HMRC_EMPLOYED_INCOME_SOURCE_COLUMN_MAP = {
    SPI_HMRC_PAY_COLUMN: ("PAY",),
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN: ("EPB",),
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN: ("EXPS",),
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN: ("INCPBEN",),
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN: ("OSSBEN",),
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN: ("TAXTERM",),
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN: ("UBISJA",),
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN: ("MOTHINC",),
}
SPI_HMRC_EMPLOYED_INCOME_FORMULA = (
    "max(0, hmrc_spi_pay + hmrc_spi_employment_benefits - "
    "hmrc_spi_employment_expenses) + hmrc_spi_incapacity_benefit_income + "
    "hmrc_spi_other_social_security_income + hmrc_spi_taxable_termination_pay + "
    "hmrc_spi_unemployment_benefit_income + "
    "hmrc_spi_miscellaneous_employment_income"
)
# The FRS instrument can source three full concepts and two explicitly named
# subsets. The remaining full SPI concepts are donor-only: they must be absent
# before the QRF rather than represented by zeroes or by the two partial leaves.
FRS_HMRC_AUXILIARY_SOURCE_COLUMNS = {
    FRS_HMRC_PAY_COLUMN: (FRS_HMRC_PAY_COLUMN,),
    FRS_HMRC_UBISJA_COLUMN: (FRS_HMRC_UBISJA_COLUMN,),
    FRS_HMRC_INCPBEN_COLUMN: (FRS_HMRC_INCPBEN_COLUMN,),
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN: (
        FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    ),
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN: (FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,),
}
FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS = (
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
)

SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS = {
    "incapacity_benefit_reported": (
        "Absent/all-default on the pinned enhanced-FRS export and certified "
        "Microcosm UK base; not a populated loader layer."
    ),
    "maternity_allowance_reported": (
        "Absent from the pinned enhanced-FRS export and certified Microcosm UK "
        "base; no training source can be materialized for this stage."
    ),
}

SPI_DONOR_REQUIRED_COLUMNS = (
    "SEX",
    "FACT",
    "GORCODE",
    "AGERANGE",
    "PAY",
    "EPB",
    "EXPS",
    "TAXTERM",
    "INCPBEN",
    "OSSBEN",
    "UBISJA",
    "MOTHINC",
    "OTHERINC",
    "PROFITS",
    "CAPALL",
    "LOSSBF",
    "SRP",
    "INCBBS",
    "DIVIDENDS",
    "PENSION",
    "INCPROP",
    "OTHERINV",
    "GIFTAID",
    "GIFTINV",
    # Restricted residential finance costs and the tax credits that carry
    # their 20% relief (microcosm#1106).
    "LLIR_RESTRICT_AMT_TOT",
    "TAX_CRED",
    "TEI",
    "TII",
    "TI",
    "MAINSRCE",
    "SEINC_NUM",
    "SPA",
)
SPI_INCOME_SOURCE_COLUMNS = {
    # Keep the PolicyEngine input on the pinned enhanced-FRS precedent. The
    # broader published HMRC employment measure is a separate auxiliary.
    "employment_income": SPI_POLICYENGINE_EMPLOYMENT_SOURCE_COLUMNS,
    "self_employment_income": ("PROFITS", "CAPALL", "LOSSBF"),
    "savings_interest_income": ("INCBBS",),
    "dividend_income": ("DIVIDENDS",),
    "private_pension_income": ("PENSION",),
    "property_income": ("INCPROP",),
    "other_investment_income": ("OTHERINV",),
    "gift_aid": ("GIFTAID",),
    "charitable_investment_gifts": ("GIFTINV",),
    **SPI_HMRC_EMPLOYED_INCOME_SOURCE_COLUMN_MAP,
    SPI_HMRC_OTHER_INCOME_COLUMN: ("OTHERINC",),
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN: ("SRP",),
    SPI_PROPERTY_FINANCE_COSTS_COLUMN: ("LLIR_RESTRICT_AMT_TOT",),
    SPI_PROFIT_BEFORE_LOSSES_DRAW_COLUMN: ("PROFITS", "CAPALL"),
    SPI_LOSSES_BROUGHT_FORWARD_DRAW_COLUMN: ("LOSSBF",),
}
SPI_QRF_SOURCE_COLUMNS = {
    output: sources
    for output, sources in SPI_INCOME_SOURCE_COLUMNS.items()
    if output != "employment_income"
}
SPI_DERIVED_POLICYENGINE_SOURCE_COLUMNS = {
    "employment_income": SPI_POLICYENGINE_EMPLOYMENT_SOURCE_COLUMNS,
}
_DIRECT_SPI_OUTPUT_SOURCE_COLUMNS = {
    output: sources
    for output, sources in SPI_INCOME_SOURCE_COLUMNS.items()
    if output
    not in {"employment_income", "self_employment_income", *SPI_LOSS_DRAW_COLUMNS}
}
#: SPI 2022-23 AGERANGE codes as whole-year bands ``[low, high)``: code 6 is
#: "65 to 74" and code 7 "75 and over", drawn over 75 to 89 and the ONS
#: 90-and-over cell (ages 90 up to 91). Composite records (-1) carry no age.
_SPI_AGE_RANGES = {
    -1: (16, 70),
    1: (16, 25),
    2: (25, 35),
    3: (35, 45),
    4: (45, 55),
    5: (55, 65),
    6: (65, 75),
    7: (75, 91),
}
#: The vendored ONS single-year population resource the donor age draw reads.
SPI_DONOR_AGE_POPULATION_RESOURCE = "ons_single_year_age_populations.json"
#: The ONS mid-year estimate period the resource is vendored at (mid-2023,
#: the nearest estimate to the 2022-23 donor tape).
SPI_DONOR_AGE_POPULATION_PERIOD = 2023
SPI_DONOR_AGE_DRAW_METHOD = (
    "whole-year age drawn within the AGERANGE band in proportion to the ONS "
    "single-year population of the donor's sex, then a uniform fraction of a "
    "year; State Pension recipients (SRP > 0) are drawn at or above State "
    "Pension age, and the band's below-State-Pension-age population share is "
    "drawn from its non-recipients"
)
#: The tape gives age only as a band, so a donor's drawn age carries no income
#: pattern within its band: a forest trained on it learns none, and twins of
#: one band faced conditionals that differed only by that noise
#: (uk-data#628). The stage-1 tape and reference forests are trained and
#: queried on the band code instead: the tape's own AGERANGE code, composite
#: records (-1) as a band of their own, and the band of a frame person's
#: whole-year age. The State Pension split inside a band is at_or_over_spa's.
#: The drawn ages stay on the donor for their other readers (the band pools'
#: State Pension side).
SPI_STAGE1_AGE_PREDICTOR_RULE = (
    "the SPI 2022-23 AGERANGE band code: the tape's own code, composite "
    "records (-1) as a band of their own, and the band of a frame person's "
    "whole-year age"
)
_SPI_REGION_MAP = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    3: "YORKSHIRE",
    4: "EAST_MIDLANDS",
    5: "WEST_MIDLANDS",
    6: "EAST_OF_ENGLAND",
    7: "LONDON",
    8: "SOUTH_EAST",
    9: "SOUTH_WEST",
    10: "WALES",
    11: "SCOTLAND",
    12: "NORTHERN_IRELAND",
}


@dataclass(frozen=True)
class SPIDonorAgeModel:
    """Single-year populations and State Pension age for the donor age draw.

    ``populations`` maps ``("MALE" | "FEMALE", whole-year age)`` to the ONS
    resident population, the 90-and-over cell at age 90. A donor of unknown
    sex draws on the two sexes summed.
    """

    populations: Mapping[tuple[str, int], float]
    state_pension_age: int
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.state_pension_age, int) or self.state_pension_age <= 0:
            raise ValueError("SPI donor age model needs an integer State Pension age.")
        for (sex, age), value in self.populations.items():
            if sex not in {"MALE", "FEMALE"} or not isinstance(age, int):
                raise ValueError(
                    f"SPI donor age model has an invalid population key {(sex, age)!r}."
                )
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"SPI donor age model population for {(sex, age)!r} must be "
                    "finite and non-negative."
                )

    def weights(self, sex: str, ages: range) -> np.ndarray:
        """Population weights over ``ages`` for ``sex`` (both sexes if unknown)."""

        sexes = (sex,) if sex in {"MALE", "FEMALE"} else ("MALE", "FEMALE")
        values = np.asarray(
            [
                sum(float(self.populations.get((name, age), 0.0)) for name in sexes)
                for age in ages
            ],
            dtype=np.float64,
        )
        if values.sum() <= 0.0:
            raise ValueError(
                f"SPI donor age model has no population for {sex} at ages "
                f"{ages.start} to {ages.stop - 1}."
            )
        return values


def load_spi_donor_age_model(build_period: int | str) -> SPIDonorAgeModel:
    """Read the vendored ONS single-year populations and the engine's SPA."""

    from microcosm.build.uk_runtime.frs_take_up import uk_take_up_population_policy
    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    rows = vendored_rows(
        SPI_DONOR_AGE_POPULATION_RESOURCE,
        period_value=str(SPI_DONOR_AGE_POPULATION_PERIOD),
    )
    populations: dict[tuple[str, int], float] = {}
    for row in rows:
        dimensions = row.get("dimensions") or {}
        sex = str(dimensions.get("sex", "")).upper()
        raw_age = str(dimensions.get("age", ""))
        age = 90 if raw_age == "90_plus" else int(raw_age)
        key = (sex, age)
        if key in populations:
            raise ValueError(f"{SPI_DONOR_AGE_POPULATION_RESOURCE} repeats {key!r}.")
        populations[key] = float(row["value"])
    expected = {(sex, age) for sex in ("MALE", "FEMALE") for age in range(0, 91)}
    if set(populations) != expected:
        missing = sorted(expected - set(populations))[:5]
        raise ValueError(
            f"{SPI_DONOR_AGE_POPULATION_RESOURCE} must carry ages 0 to 90 for both "
            f"sexes at mid-{SPI_DONOR_AGE_POPULATION_PERIOD}; missing {missing}."
        )
    policy = uk_take_up_population_policy(build_period)
    return SPIDonorAgeModel(
        populations=populations,
        state_pension_age=policy.state_pension_age,
        source=(
            f"{SPI_DONOR_AGE_POPULATION_RESOURCE} (ONS MYE2 mid-"
            f"{SPI_DONOR_AGE_POPULATION_PERIOD}); State Pension age from "
            f"{policy.source} at {policy.instant}"
        ),
    )


def _draw_spi_donor_ages(
    age_codes: np.ndarray,
    sex: np.ndarray,
    receives_state_pension: np.ndarray,
    fact: np.ndarray,
    rng: np.random.Generator,
    *,
    model: SPIDonorAgeModel,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Draw donor ages by ONS single-year shares, State Pension age aware.

    Within each (AGERANGE band, sex) cell, a band that straddles State Pension
    age splits into a below segment and an at-or-above segment. Recipients
    (``SRP > 0``) always draw from the at-or-above segment. Non-recipients
    draw from the below segment with the probability that puts the band's
    ONS below-segment population share of the cell's FACT mass there (capped
    at one, the shortfall receipted), and otherwise from at-or-above. The
    whole-year age inside a segment follows the ONS single-year populations,
    and a uniform fraction of a year keeps the predictor continuous.
    """

    spa = model.state_pension_age
    ages = np.empty(len(age_codes), dtype=np.float64)
    receipt: list[dict[str, object]] = []
    for code in sorted(set(int(value) for value in age_codes)):
        low, high = _SPI_AGE_RANGES[code]
        for sex_label in ("MALE", "FEMALE", "UNKNOWN"):
            cell = np.flatnonzero((age_codes == code) & (sex == sex_label))
            if cell.size == 0:
                continue
            recipients = receives_state_pension[cell]
            segments: list[range] = []
            if low < spa < high:
                segments = [range(low, spa), range(spa, high)]
            else:
                segments = [range(low, high)]
            cell_mass = float(fact[cell].sum())
            below_share = 0.0
            p_below = 0.0
            unfilled_below_mass = 0.0
            if len(segments) == 2:
                all_weights = model.weights(sex_label, range(low, high))
                below_share = float(all_weights[: spa - low].sum() / all_weights.sum())
                non_recipient_mass = float(fact[cell][~recipients].sum())
                wanted = below_share * cell_mass
                if non_recipient_mass > 0.0:
                    p_below = min(1.0, wanted / non_recipient_mass)
                unfilled_below_mass = max(0.0, wanted - non_recipient_mass)
                choose_below = (~recipients) & (rng.random(cell.size) < p_below)
            else:
                choose_below = np.zeros(cell.size, dtype=bool)
            for position, segment in enumerate(segments):
                if len(segments) == 2:
                    members = choose_below if position == 0 else ~choose_below
                else:
                    members = np.ones(cell.size, dtype=bool)
                count = int(members.sum())
                if count == 0:
                    continue
                weights = model.weights(sex_label, segment)
                whole = rng.choice(
                    np.arange(segment.start, segment.stop),
                    size=count,
                    p=weights / weights.sum(),
                )
                ages[cell[members]] = whole + rng.random(count)
            drawn = ages[cell]
            receipt.append(
                {
                    "agerange": code,
                    "sex": sex_label,
                    "band": [low, high],
                    "donors": int(cell.size),
                    "state_pension_recipients": int(recipients.sum()),
                    "fact_mass": cell_mass,
                    "ons_below_state_pension_age_share": below_share,
                    "non_recipient_below_probability": p_below,
                    "unfilled_below_state_pension_age_mass": unfilled_below_mass,
                    "recipients_below_state_pension_age": int(
                        (recipients & (drawn < spa)).sum()
                    ),
                }
            )
    return ages, receipt


@dataclass(frozen=True)
class UKSPIStatePensionAgeGuard:
    """Pension-age reports the income stage zeroes below State Pension age.

    ``spi_channel_columns`` are imputed on the synthetic channel (the SPI
    State Pension leaf is always zeroed there before the stage-2 receipt
    bridge reads it); ``base_channel_columns`` are FRS reports on the base
    channel, zeroed before stage 2 trains on them (microcosm#1069, ruling R8:
    zero with a receipt).
    """

    state_pension_age: int
    spi_channel_columns: tuple[str, ...]
    base_channel_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state_pension_age, int) or self.state_pension_age <= 0:
            raise ValueError("State Pension age guard needs an integer age.")
        for columns in (self.spi_channel_columns, self.base_channel_columns):
            if len(set(columns)) != len(columns) or not all(
                isinstance(column, str) and column for column in columns
            ):
                raise ValueError(
                    "State Pension age guard columns must be distinct names."
                )


def _zero_below_state_pension_age(
    person: pd.DataFrame,
    *,
    rows: pd.Series,
    columns: Sequence[str],
    state_pension_age: int,
    person_weights: pd.Series,
    step: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Zero ``columns`` on ``rows`` whose age is below State Pension age."""

    below = rows.astype(bool) & person["age"].lt(state_pension_age)
    removed: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in person.columns:
            raise ValueError(
                f"State Pension age guard names {column!r}, absent from the frame."
            )
        values = person.loc[below, column].to_numpy(dtype=np.float64)
        positive = values > 0.0
        weights = person_weights.loc[below].to_numpy(dtype=np.float64)
        removed[column] = {
            "rows": int(positive.sum()),
            "weighted_people": float(weights[positive].sum()),
            "unweighted_amount": float(values[positive].sum()),
            "weighted_amount": float((values * weights)[positive].sum()),
        }
        person.loc[below, column] = 0.0
    return person, {
        "step": step,
        "state_pension_age": state_pension_age,
        "rows_below_state_pension_age": int(below.sum()),
        "removed": removed,
    }


@dataclass(frozen=True)
class UKSPIIncomeImputationResult:
    """SPI-filled person table and auditable fit/source evidence."""

    person: pd.DataFrame
    fit_weight_records: tuple[FitWeightRecord, ...]
    donor_path: Path
    donor_sha256: str
    donor_size_bytes: int
    donor_rows: int
    stage2_training_rows: int
    spi_prediction_rows: int
    reviewed_absent_stage2_outputs: dict[str, str]
    pension_receipt_bridge: Mapping[str, object] | None = None
    income_uprating: Mapping[str, object] | None = None
    band_donor_resample: Mapping[str, object] | None = None
    donor_age_draw: Mapping[str, object] | None = None
    donor_property_finance_costs: Mapping[str, object] | None = None
    state_pension_age_guard: tuple[Mapping[str, object], ...] = ()
    recipient_domain: Mapping[str, object] | None = None
    stage1_transport: Mapping[str, object] | None = None
    bridge_validation: Mapping[str, object] | None = None
    reference_training_rows: int = 0
    state_pension_age: int | None = None
    gainful_self_employment_refresh: Mapping[str, object] | None = None
    pension_coherence: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _SPIDonorFingerprint:
    """Stable-file identity binding a reviewed hash to the parsed donor bytes."""

    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class VerifiedSPIDonorIdentity:
    """Opaque proof that the local licensed donor matched the reviewed bytes."""

    path: Path
    sha256: str
    size_bytes: int
    fingerprint: _SPIDonorFingerprint
    _verification_token: object


def verify_spi_donor_identity(path: str | Path) -> VerifiedSPIDonorIdentity:
    """Verify the licensed donor before parsing and bind the proof to its file."""

    donor = Path(path).expanduser().resolve()
    if donor.name != SPI_DONOR_FILENAME:
        raise ValueError(
            f"Current SPI donor must be named {SPI_DONOR_FILENAME!r}, got "
            f"{donor.name!r}."
        )
    if not donor.is_file():
        raise FileNotFoundError(f"SPI 2022-23 donor not found: {donor}.")
    before = _spi_donor_fingerprint(donor)
    if before.size_bytes != SPI_DONOR_SIZE_BYTES:
        raise ValueError(
            "SPI 2022-23 donor size does not match the reviewed UKDS source "
            f"identity: expected {SPI_DONOR_SIZE_BYTES}, got {before.size_bytes}."
        )
    digest = _sha256(donor)
    after = _spi_donor_fingerprint(donor)
    if after != before:
        raise RuntimeError(
            "SPI 2022-23 donor changed while its reviewed identity was being verified."
        )
    if digest != SPI_DONOR_SHA256:
        raise ValueError(
            "SPI 2022-23 donor SHA-256 does not match the reviewed UKDS source "
            f"identity: expected {SPI_DONOR_SHA256}, got {digest}."
        )
    return VerifiedSPIDonorIdentity(
        path=donor,
        sha256=digest,
        size_bytes=before.size_bytes,
        fingerprint=before,
        _verification_token=_SPI_DONOR_VERIFICATION_TOKEN,
    )


def assert_frs_hmrc_auxiliary_crosswalk_available(person: pd.DataFrame) -> None:
    """Require the adjudicated FRS leaves and forbid unavailable full concepts."""

    required = tuple(FRS_HMRC_AUXILIARY_SOURCE_COLUMNS)
    missing = sorted(set(required) - set(person.columns))
    if missing:
        raise ValueError(
            "Certified UK FRS channel is missing adjudicated retained HMRC "
            f"leaf column(s): {missing}. Retain PAY, UBISJA, INCPBEN, "
            "ossben_identifiable_subset, and srp_regular_code5 from their "
            "reviewed raw FRS sources before the SPI QRF."
        )
    unavailable = sorted(
        set(FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS) & set(person.columns)
    )
    if unavailable:
        raise ValueError(
            "Pre-QRF FRS candidate must not materialize source-absent full "
            f"SPI concept column(s): {unavailable}. EPB, EXPS, TAXTERM, "
            "MOTHINC, OTHERINC, full OSSBEN, and full SRP cannot be zero-filled "
            "or inferred from their named subsets."
        )
    numeric = person[list(required)].apply(pd.to_numeric, errors="coerce")
    _require_finite_numeric(numeric, label="FRS HMRC retained leaves")
    if (numeric.to_numpy(dtype=float) < 0.0).any():
        raise ValueError("FRS HMRC retained leaves contain negative values.")


def impute_uk_spi_income_support(
    support: UKSPISupportResult,
    spi_tab_path: str | Path,
    *,
    seed: int = 42,
    n_estimators: int = 100,
    donor_sample_size: int | None = DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    build_period: int | str = 2023,
    verified_donor: VerifiedSPIDonorIdentity | None = None,
    donor_table: pd.DataFrame | None = None,
    initialize_frs_channel_columns: Mapping[str, float] | None = None,
    condition_on_state_pension_receipt: bool = False,
    rebase_income_to_build_period: bool = False,
    band_donor_resample: Mapping[str, object] | None = None,
    donor_age_model: SPIDonorAgeModel | None = None,
    state_pension_age_guard: UKSPIStatePensionAgeGuard | None = None,
    state_pension_age: int | None = None,
    allow_in_sample_reference: bool = False,
) -> UKSPIIncomeImputationResult:
    """Run strict SPI-income and FRS-only QRFs on rebuilt positive support.

    ``band_donor_resample`` (``lower_bounds``, ``regional_pool_minimum``,
    ``seed`` and optionally ``age_pool_minimum``) gives the reserved band
    carriers (PolicyEngine/chronicle#280 lane) a band-conditional draw from
    the full prepared tape after the stage-1 forest draw; ``None`` leaves
    every synthetic adult on the forest draw. ``donor_age_model`` draws donor
    ages by ONS single-year shares, State Pension age aware, and
    ``state_pension_age_guard`` zeroes pension-age reports below State Pension
    age with receipts (microcosm#1069). ``state_pension_age`` is the boundary
    the bridge predictor ``at_or_over_spa`` reads on the twin; it defaults to
    the donor age model's and is required without one (microcosm#840).
    """

    if support.household_weight_kind is not WeightKind.IMPORTANCE:
        raise ValueError(
            "SPI income imputation requires rebuilt importance-weight support."
        )
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer.")
    if donor_sample_size is not None and (
        not isinstance(donor_sample_size, int) or donor_sample_size <= 0
    ):
        raise ValueError("donor_sample_size must be a positive integer or None.")

    donor_path = Path(spi_tab_path).expanduser().resolve()
    if donor_table is None:
        if verified_donor is None:
            # Keep the private helper call as the narrow test seam used by the
            # synthetic-donor unit tests. Production receives an opaque identity.
            verified_donor = _verify_spi_donor_identity(donor_path)
        elif verified_donor.path != donor_path:
            raise ValueError(
                "Verified SPI donor identity does not match the requested donor path."
            )
        if isinstance(verified_donor, VerifiedSPIDonorIdentity):
            _assert_verified_spi_donor_current(verified_donor)
        raw_donor = pd.read_csv(donor_path, delimiter="\t")
        if isinstance(verified_donor, VerifiedSPIDonorIdentity):
            _assert_verified_spi_donor_current(verified_donor)
    else:
        if verified_donor is not None:
            raise ValueError("donor_table and verified_donor are mutually exclusive.")
        if not isinstance(donor_table, pd.DataFrame):
            raise TypeError("donor_table must be a pandas DataFrame.")
        raw_donor = donor_table.copy(deep=True)
    donor = _prepare_spi_donor(raw_donor, seed=seed, age_model=donor_age_model)
    donor_age_draw = {
        **donor.attrs.get("age_draw", {}),
        "stage1_age_predictor": SPI_STAGE1_AGE_PREDICTOR_RULE,
    }
    donor_full = donor
    donor_fit_weights = donor["FACT"].to_numpy(dtype=np.float64)
    if donor_sample_size is not None:
        donor = donor.sample(
            n=donor_sample_size,
            replace=True,
            weights="FACT",
            random_state=seed,
        ).reset_index(drop=True)
        # Mirroring the enhanced-FRS pipeline, the FACT-weighted bootstrap
        # itself represents the SPI design. Reapplying FACT to the sampled
        # rows would square the survey weights. Uniform typed DESIGN weights
        # keep the fit auditable without double-weighting the donor.
        donor_fit_weights = np.ones(len(donor), dtype=np.float64)

    person = support.person.copy()
    household = support.household
    person_channel = support_channel_column("person")
    household_channel = support_channel_column("household")
    _require_columns(person, (person_channel,), label="person support")
    _require_columns(
        household,
        (
            "household_id",
            "household_weight",
            household_channel,
            HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
        ),
        label="household support",
    )
    spi_household = household[household_channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    if (
        not spi_household.any()
        or not household.loc[spi_household, "household_weight"].gt(0.0).all()
    ):
        raise ValueError("SPI support rows must all carry positive effective mass.")
    spi_channel_people = person[person_channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    if not spi_channel_people.any():
        raise ValueError("SPI support has no person rows to impute.")
    _require_columns(
        person, ("age", SPI_RECIPIENT_ROLE_COLUMN), label="SPI recipient domain"
    )
    _require_finite_numeric(person[["age"]], label="SPI recipient ages")
    if not pd.api.types.is_bool_dtype(person[SPI_RECIPIENT_ROLE_COLUMN].dtype):
        raise ValueError(
            f"SPI recipient role column {SPI_RECIPIENT_ROLE_COLUMN!r} must be boolean."
        )
    # Both forests are queried over the age domain, the recipient rule before
    # uk-data#504, so each recipient keeps the draw it has always had; the
    # rows of dependants aged 16 to 19 are then discarded.
    spi_age_domain = spi_channel_people & person.age.ge(SPI_MINIMUM_RECIPIENT_AGE)
    role_domain = spi_age_domain & person[SPI_RECIPIENT_ROLE_COLUMN]
    if not role_domain.any():
        raise ValueError("SPI support has no recipients in the donor age domain.")
    if state_pension_age is None and donor_age_model is not None:
        state_pension_age = donor_age_model.state_pension_age
    if state_pension_age is None:
        raise ValueError(
            "SPI stage-1 bridge predictors need State Pension age: pass "
            "state_pension_age or a donor age model."
        )
    person_weights = _person_household_weights(person, household)
    # The no-record rule (microcosm#840): a twin with no taxable income has no
    # tax record for the tape to condition on, so its copy keeps the twin's
    # values; a reserved band carrier is always drawn, since its leaves come
    # from the band pools after the forest draw.
    taxable = _taxable_income_mask(person.loc[role_domain])
    carrier = (
        person.loc[role_domain, SPI_INCOME_BAND_CARRIER_COLUMN].astype(bool).to_numpy()
        if SPI_INCOME_BAND_CARRIER_COLUMN in person.columns
        else np.zeros(int(role_domain.sum()), dtype=bool)
    )
    spi_people = role_domain.copy()
    spi_people.loc[role_domain] = taxable | carrier
    undrawn = role_domain & ~spi_people
    if not spi_people.any():
        raise ValueError("SPI support has no recipients with taxable income.")
    # The FRS twin's pay and employer contribution, before the tape redraws
    # the pay: the twin's employer rate is carried onto the drawn pay below.
    twin_pay = person.loc[spi_people, "employment_income"].to_numpy(
        dtype=np.float64, copy=True
    )
    twin_employer = person.loc[spi_people, "employer_pension_contributions"].to_numpy(
        dtype=np.float64, copy=True
    )
    recipient_domain = {
        "minimum_age": SPI_MINIMUM_RECIPIENT_AGE,
        "role_column": SPI_RECIPIENT_ROLE_COLUMN,
        "recipient_rows": int(spi_people.sum()),
        "dependant_rows_kept_on_twin_values": int(
            (spi_age_domain & ~role_domain).sum()
        ),
        "dependant_weight_kept_on_twin_values": float(
            person_weights.loc[spi_age_domain & ~role_domain].sum()
        ),
        "income_rule": SPI_RECIPIENT_INCOME_RULE,
        "unearned_income_floor_gbp": SPI_RECIPIENT_UNEARNED_INCOME_FLOOR_GBP,
        "undrawn_no_taxable_income_rows": int(undrawn.sum()),
        "undrawn_no_taxable_income_weight": float(person_weights.loc[undrawn].sum()),
        "carriers_drawn_without_taxable_income": int((carrier & ~taxable).sum()),
    }
    base_people = person[person_channel] == BASE_FRS_SUPPORT_CHANNEL
    if not base_people.any():
        raise ValueError("SPI support has no FRS base rows.")
    if initialize_frs_channel_columns is not None:
        person = _initialize_frs_channel_columns(
            person,
            base_people=base_people,
            columns=initialize_frs_channel_columns,
        )
    person = _seed_frs_hmrc_auxiliary_leaves(person, spi_people=spi_people)

    base_household = household[household_channel] == BASE_FRS_SUPPORT_CHANNEL
    if "clone_index" in household:
        base_household &= household["clone_index"] == 0
    training_household_ids = set(household.loc[base_household, "household_id"])
    training_people = base_people & person["person_household_id"].isin(
        training_household_ids
    )
    if not training_people.any():
        raise ValueError("FRS-only stage has no canonical base training rows.")
    # The reference forest's population mirrors the draw's: FRS claimants and
    # partners aged 16 and over with taxable income under the same rule.
    reference_people = (
        training_people
        & person.age.ge(SPI_MINIMUM_RECIPIENT_AGE)
        & person[SPI_RECIPIENT_ROLE_COLUMN]
    )
    reference_people.loc[reference_people] = _taxable_income_mask(
        person.loc[reference_people]
    )
    if not reference_people.any():
        raise ValueError(
            "SPI reference forest has no FRS training rows with taxable income."
        )

    recipient_predictors = _stage1_query_predictors(
        person.loc[spi_people],
        household,
        state_pension_age=state_pension_age,
    )
    reference_predictors = _stage1_query_predictors(
        person.loc[reference_people],
        household,
        state_pension_age=state_pension_age,
    )
    donor_predictors, encoded_reference, encoded_recipient = _encode_predictor_frames(
        (
            donor[list(SPI_STAGE1_PREDICTOR_COLUMNS)],
            reference_predictors,
            recipient_predictors,
        ),
        categorical=SPI_STAGE1_CATEGORICAL_PREDICTORS,
    )
    donor_frame = _person_fit_frame(
        predictors=donor_predictors,
        targets=donor[list(SPI_INCOME_QRF_OUTPUT_COLUMNS)],
        weights=donor_fit_weights,
        weight_kind=WeightKind.DESIGN,
    )
    qrf_cls = _qrf_class()
    stage1 = qrf_cls(n_estimators=n_estimators, seed=seed).fit(
        donor_frame,
        list(donor_predictors.columns),
        list(SPI_INCOME_QRF_OUTPUT_COLUMNS),
        weights="design",
    )
    # The twin-conditional transport (microcosm#840): reference forests on
    # the FRS rows, with the stage-1 predictors and chain order, locate each
    # twin's observed leaves; the stage-1 forest draws at those uniforms. The
    # forests are cross-fitted by predictor cell, so the forest that locates
    # a copy never saw its twin (module docstring). The reference seed derives
    # from the stage seed like stage 2's and the band resample's; changing the
    # offset edits the derivation convention.
    assert_analogue_map_covers(SPI_INCOME_QRF_OUTPUT_COLUMNS)
    reference_weights = person_weights.loc[reference_people]
    reference_targets = list(analogue_targets(SPI_INCOME_QRF_OUTPUT_COLUMNS))
    reference_weight_kind: dict[str, str] = {}

    def fit_reference(encoded: pd.DataFrame, observed: pd.DataFrame):
        frame = _person_fit_frame(
            predictors=encoded,
            targets=observed,
            weights=reference_weights.loc[encoded.index].to_numpy(dtype=np.float64),
            weight_kind=WeightKind.IMPORTANCE,
        )
        fitted = qrf_cls(
            n_estimators=n_estimators, seed=seed + SPI_TRANSPORT_SEED_OFFSET
        ).fit(frame, list(encoded.columns), reference_targets, weights="importance")
        reference_weight_kind.setdefault("kind", fitted.weight_kind)
        return fitted

    training_folds, used_folds = cell_fold_ids(
        reference_predictors, folds=SPI_TRANSPORT_FOLDS, seed=seed
    )
    if SPI_TRANSPORT_KEY_COLUMN not in person.columns:
        raise ValueError(
            f"SPI stage-1 identity uniforms need {SPI_TRANSPORT_KEY_COLUMN!r} on "
            "the person table (the support lineage column the manifest declares)."
        )
    identity_column = SPI_TRANSPORT_KEY_COLUMN
    transport = cross_fit_transport(
        fit_reference,
        training_predictors=reference_predictors,
        encoded_training=encoded_reference,
        training_observed=observed_analogues(person.loc[reference_people]),
        training_folds=training_folds,
        training_identities=person.loc[reference_people, identity_column].to_numpy(),
        used=used_folds,
        recipient_predictors=recipient_predictors,
        encoded_recipients=encoded_recipient,
        observed=observed_analogues(person.loc[spi_people]),
        identities=person.loc[spi_people, identity_column].to_numpy(),
        groups=recipient_predictors["main_source"].to_numpy(),
        seed=seed,
        folds=SPI_TRANSPORT_FOLDS,
        allow_in_sample=allow_in_sample_reference,
        outputs=SPI_INCOME_QRF_OUTPUT_COLUMNS,
    )
    # The tape forest reads a located sign uniform as the reference's gate
    # wrote it only when both gates order the same classes (microcosm#840): a
    # full build refuses a mismatch; a sampled rung or a synthetic fixture,
    # whose tiny training support can drop a sign, records it in the receipt.
    sign_class_mismatches = _sign_class_mismatches(transport.receipt, stage1)
    if sign_class_mismatches and not allow_in_sample_reference:
        raise ValueError(
            "SPI stage-1 targets whose FRS reference sign classes differ from the "
            f"tape forest's, so a located sign uniform would change meaning: "
            f"{sign_class_mismatches}."
        )
    transport.receipt["sign_class_mismatches"] = sign_class_mismatches
    stage1_draws = stage1.predict_from_uniforms(
        encoded_recipient,
        quantiles=transport.quantiles,
        sign_uniforms=transport.sign_uniforms,
    )
    _validate_predictions(
        stage1_draws,
        expected=SPI_INCOME_QRF_OUTPUT_COLUMNS,
        label="SPI stage-1",
    )
    # FRS-channel rows keep their own reported incomes, dividends included
    # (uk-data#498, microcosm#1095); dependants aged 16 to 19 and undrawn
    # copies keep their twin's values (uk-data#504, microcosm#840).
    income_uprating = None
    uprating_factors = dict.fromkeys(SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0)
    if rebase_income_to_build_period:
        uprating_factors, income_uprating = _spi_income_uprating_factors(
            int(build_period)
        )
        stage1_draws = stage1_draws.mul(pd.Series(uprating_factors), axis="columns")
    nonnegative_stage1 = [
        column
        for column in SPI_INCOME_QRF_OUTPUT_COLUMNS
        if column != SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN
    ]
    if (stage1_draws[nonnegative_stage1].to_numpy(dtype=np.float64) < 0.0).any():
        raise ValueError("SPI stage-1 produced negative non-negative outputs.")
    for column in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        if column not in person:
            # These full concepts are unavailable on the FRS instrument, so
            # the FRS channel carries the adjudicated stage-time semantics:
            # zero, the value every consumer fills at read time. Shipping NaN
            # here was assessment-era honesty that made the artifact
            # unloadable — the engine's validate() refuses NaN inputs, and
            # the calibration seam's finiteness fence rightly refuses the
            # frame — while the auxiliary crosswalk guard above already
            # protects the QRF from mistaking the fill for measured data.
            person[column] = 0.0
        person.loc[spi_people, column] = stage1_draws[column].to_numpy()
    band_donor_receipt = None
    if band_donor_resample is not None:
        person, band_donor_receipt = _resample_band_donor_leaves(
            person,
            donor_full,
            household=household,
            spi_people=spi_people,
            uprating_factors=uprating_factors,
            state_pension_age=(
                None if donor_age_model is None else donor_age_model.state_pension_age
            ),
            **dict(band_donor_resample),
        )
    guard_receipts: list[Mapping[str, object]] = []
    if state_pension_age_guard is not None:
        # The SPI State Pension leaf decides the stage-2 receipt bridge, so it
        # is zeroed below State Pension age before the bridge reads it.
        person, receipt = _zero_below_state_pension_age(
            person,
            rows=spi_people,
            columns=(SPI_HMRC_STATE_PENSION_INCOME_COLUMN,),
            state_pension_age=state_pension_age_guard.state_pension_age,
            person_weights=_person_household_weights(person, household),
            step="spi_state_pension_leaf_after_stage1",
        )
        guard_receipts.append(receipt)
    person = _derive_policyengine_employment_input(person, spi_people=spi_people)

    taxable_interest_draw = person.loc[spi_people, "savings_interest_income"].to_numpy(
        dtype=np.float64, copy=True
    )
    stage2_outputs, reviewed_absent = _stage2_outputs(person)
    person = _owner_manager_flags_as_numbers(person)
    if (
        state_pension_age_guard is not None
        and state_pension_age_guard.base_channel_columns
    ):
        person, receipt = _zero_below_state_pension_age(
            person,
            rows=base_people,
            columns=state_pension_age_guard.base_channel_columns,
            state_pension_age=state_pension_age_guard.state_pension_age,
            person_weights=_person_household_weights(person, household),
            step="base_channel_reports_before_stage2",
        )
        guard_receipts.append(receipt)

    income_predictors = FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS
    train_predictors = _person_predictors(
        person.loc[training_people],
        household,
        income_predictors=income_predictors,
    )
    # Queried over the age domain like stage 1; a dependant's row conditions
    # on its twin's values and is discarded below, and each row draws from
    # uniforms keyed on its own person_id, so no recipient's draw depends on
    # it. Stage 1 keys on the twin's person_source_id on purpose and counts
    # the shared identities; stage 2 does not, because the SPI channel and
    # the band donors copy one FRS person more than once, and each copy
    # draws its stage-2 values apart, as under the positional stream.
    target_predictors = _person_predictors(
        person.loc[spi_age_domain],
        household,
        income_predictors=income_predictors,
    )
    stage2_recipient_positions = np.flatnonzero(
        spi_people.loc[spi_age_domain].to_numpy(dtype=bool)
    )
    pension_bridge = None
    if condition_on_state_pension_receipt:
        # A receipt indicator is a predictive bridge, not an assertion that
        # FRS regular pension and the full taxable SPI SRP amount coincide.
        # Without it, stage 2 can attach a second, unrelated pension regime
        # to a stage-1 vector whose HMRC pension leaf is already determined.
        train_receipt = person.loc[training_people, "state_pension_reported"].gt(0)
        target_receipt = person.loc[
            spi_age_domain, SPI_HMRC_STATE_PENSION_INCOME_COLUMN
        ].gt(0)
        train_predictors["state_pension_receipt"] = train_receipt.astype(float)
        target_predictors["state_pension_receipt"] = target_receipt.astype(float)
        pension_bridge = {
            "training_source": "state_pension_reported > 0",
            "recipient_source": "hmrc_spi_state_pension_income > 0",
            "training_positive_rows": int(train_receipt.sum()),
            "recipient_positive_rows": int(
                target_receipt.iloc[stage2_recipient_positions].sum()
            ),
        }
    encoded_train, encoded_target = _encode_predictor_pair(
        train_predictors,
        target_predictors,
    )
    training_targets = person.loc[training_people, list(stage2_outputs)].copy()
    _require_finite_numeric(training_targets, label="FRS-only training outputs")
    if (training_targets.to_numpy(dtype=np.float64) < 0.0).any():
        raise ValueError("FRS-only training outputs must be non-negative.")
    person_weights = _person_household_weights(person, household)
    stage2_frame = _person_fit_frame(
        predictors=encoded_train,
        targets=training_targets,
        weights=person_weights.loc[training_people].to_numpy(dtype=np.float64),
        weight_kind=WeightKind.IMPORTANCE,
    )
    # Stage 2 deliberately derives a distinct RNG stream from the single
    # reviewed stage seed; changing this edits the derivation convention.
    stage2 = qrf_cls(n_estimators=n_estimators, seed=seed + 1).fit(
        stage2_frame,
        list(encoded_train.columns),
        list(stage2_outputs),
        weights="importance",
    )
    stage2_keys = integer_record_keys(
        person.loc[spi_age_domain, "person_id"].to_numpy(),
        label="SPI stage-2 person_id",
    )
    stage2_draws = stage2.predict_from_uniforms(
        encoded_target,
        quantiles={
            target: stable_identity_uniforms(
                stage2_keys, seed=seed + 1, salt=f"{FRS_ONLY_FIT_NAME}:{target}"
            )
            for target in stage2_outputs
        },
        sign_uniforms={
            target: stable_identity_uniforms(
                stage2_keys, seed=seed + 1, salt=f"{FRS_ONLY_FIT_NAME}:{target}:sign"
            )
            for target in stage2_outputs
        },
    )
    _validate_predictions(
        stage2_draws,
        expected=stage2_outputs,
        label="FRS-only stage-2",
    )
    stage2_draws = stage2_draws.iloc[stage2_recipient_positions].copy()
    if (stage2_draws.to_numpy(dtype=np.float64) < 0.0).any():
        raise ValueError("FRS-only stage-2 produced negative non-negative outputs.")
    for column in stage2_outputs:
        person.loc[spi_people, column] = stage2_draws[column].to_numpy()
    person, pension_coherence = _spi_pension_coherence(
        person,
        household,
        spi_people=spi_people,
        twin_pay=twin_pay,
        twin_employer=twin_employer,
    )
    person = _owner_manager_flags_from_draws(person, spi_people=spi_people)
    if (
        state_pension_age_guard is not None
        and state_pension_age_guard.spi_channel_columns
    ):
        person, receipt = _zero_below_state_pension_age(
            person,
            rows=spi_people,
            columns=state_pension_age_guard.spi_channel_columns,
            state_pension_age=state_pension_age_guard.state_pension_age,
            person_weights=_person_household_weights(person, household),
            step="spi_channel_reports_after_stage2",
        )
        guard_receipts.append(receipt)

    tax_free = person.loc[spi_people, "tax_free_savings_income"].to_numpy(
        dtype=np.float64
    )
    person.loc[spi_people, "savings_interest_income"] = taxable_interest_draw + tax_free
    person = derive_hmrc_income_auxiliaries(person, row_mask=spi_people)
    person = _move_losses_brought_forward_to_loss_relief(person, spi_people=spi_people)
    person = _refresh_disability_derived_inputs(person, spi_people=spi_people)
    person = _refresh_carer_take_up_input(person, spi_people=spi_people)
    person = _refresh_esa_support_component(person, spi_people=spi_people)
    person = _clear_trading_loss_beside_profit(person, spi_people=spi_people)
    person = _refresh_uc_start_up_period(person, spi_people=spi_people)
    person, gainful_self_employment = _refresh_uc_gainful_self_employment(
        person, spi_people=spi_people
    )
    person = _refresh_landlord_carrier(person, spi_people=spi_people)
    person = _refresh_annual_income_descriptors(person, spi_people=spi_people)
    # The flag is the stage's last new column, where the graph's typed cells
    # place it (the legacy oracle and the graph must agree on column order).
    person[SPI_INCOME_RECIPIENT_COLUMN] = (
        spi_people.reindex(person.index).fillna(False).to_numpy(dtype=bool)
    )
    return UKSPIIncomeImputationResult(
        person=person,
        fit_weight_records=(
            FitWeightRecord(SPI_DONOR_FIT_NAME, stage1.weight_kind),
            FitWeightRecord(SPI_TRANSPORT_FIT_NAME, reference_weight_kind["kind"]),
            FitWeightRecord(FRS_ONLY_FIT_NAME, stage2.weight_kind),
        ),
        donor_path=donor_path,
        donor_sha256=(
            verified_donor.sha256
            if isinstance(verified_donor, VerifiedSPIDonorIdentity)
            else _sha256(donor_path)
        ),
        donor_size_bytes=(
            verified_donor.size_bytes
            if isinstance(verified_donor, VerifiedSPIDonorIdentity)
            else donor_path.stat().st_size
        ),
        donor_rows=len(donor),
        stage2_training_rows=int(training_people.sum()),
        spi_prediction_rows=int(spi_people.sum()),
        reviewed_absent_stage2_outputs=reviewed_absent,
        pension_receipt_bridge=pension_bridge,
        income_uprating=income_uprating,
        band_donor_resample=band_donor_receipt,
        donor_age_draw=donor_age_draw,
        donor_property_finance_costs=donor_full.attrs.get("property_finance_costs"),
        state_pension_age_guard=tuple(guard_receipts),
        recipient_domain=recipient_domain,
        stage1_transport=transport.receipt,
        bridge_validation=donor_full.attrs.get("bridge"),
        reference_training_rows=int(reference_people.sum()),
        state_pension_age=int(state_pension_age),
        gainful_self_employment_refresh=gainful_self_employment,
        pension_coherence=pension_coherence,
    )


SPI_INCOME_BAND_CARRIER_COLUMN = "person_is_spi_income_band_carrier"
SPI_INCOME_BAND_LOWER_BOUND_COLUMN = "spi_income_band_donor_lower_bound"
#: The carrier's leaf draw is keyed on the FRS person it copies (the support
#: lineage column), falling back to ``person_id`` on a frame without lineage.
SPI_INCOME_BAND_CARRIER_KEY_COLUMN = "person_source_id"
SPI_INCOME_BAND_DONOR_LEAF_DRAW_SALT = "spi_income_band_donor_leaf_draw"


def _spi_agerange_codes(ages: np.ndarray) -> np.ndarray:
    """The SPI AGERANGE code (1 to 7) of each age; under-16s map to code 1."""

    codes = np.ones(len(ages), dtype=np.int64)
    for code in sorted(code for code in _SPI_AGE_RANGES if code > 0):
        low = _SPI_AGE_RANGES[code][0]
        codes[np.asarray(ages, dtype=float) >= low] = code
    return codes


def _band_mean_total_income(
    realized: np.ndarray, drawn: np.ndarray
) -> float | str | None:
    """A reserved band's mean realised total income, for its receipt row.

    A carrier's realised total income is its drawn tape record's own, so a
    band whose carriers drew fewer than ten distinct records, as any band with
    fewer than ten carriers does, would give back one record's value or come
    close to it. Its mean is "<10" (UKDS disclosure control; see
    ``receipt_counts``), and a band without carriers has none. No single
    record's value, such as the band's lowest or highest, is recorded.
    """

    if not realized.size:
        return None
    if np.unique(drawn).size < 10:
        return "<10"
    return float(realized.mean())


def _resample_band_donor_leaves(
    person: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    household: pd.DataFrame,
    spi_people: pd.Series,
    uprating_factors: Mapping[str, float],
    lower_bounds: Sequence[int],
    regional_pool_minimum: int,
    seed: int,
    age_pool_minimum: int | None = None,
    state_pension_age: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Give every reserved band carrier a band-conditional tape draw.

    The pool for a carrier is every prepared tape record whose total income
    *as it will be realised* lies in the carrier's band: the record's stage-1
    leaves at the stage's uprating factors plus the published remainder of its
    total income (TEI + TII) held nominal. The bands are the calibration
    year's Table 2.5 bands and the reserved weights are that year's taxpayer
    counts, so membership is asserted on the uprated amount, not on the
    tape's 2022-23 nominal one. The pool narrows to the carrier's region when
    that regional pool holds at least ``regional_pool_minimum`` records. With
    ``age_pool_minimum`` the band pool first narrows to the records in the
    carrier's published SPI age band (composites, which carry none, never
    age-match) when that pool holds at least ``age_pool_minimum`` records,
    and the regional narrowing then applies within it (microcosm#1069). With
    ``state_pension_age`` the band pool first keeps only the records whose
    drawn age is on the carrier's side of State Pension age, since the
    published age band 65 to 74 straddles it and a carrier at or over it must
    not inherit the zero State Pension leaf of a donor drawn below it. One
    record is drawn FACT-weighted with replacement and all stage-1 leaves are
    copied from it, then uprated exactly as the forest draws were; the
    accounting aggregates derive after the draw as usual. The draw is keyed on
    the carrier's identity (the id of the FRS person it copies, where the frame
    carries it), so a carrier draws the same record whatever the row order and
    whoever else is seated (microcosm#1063). Composite records
    stay in the pools as published. Every carrier's realised total income is
    checked against its band after the draw and any breach refuses the stage.
    """

    if not isinstance(regional_pool_minimum, int) or regional_pool_minimum <= 0:
        raise ValueError("regional_pool_minimum must be a positive integer.")
    if not isinstance(seed, int):
        raise ValueError("band donor resample seed must be an integer.")
    if age_pool_minimum is not None and (
        not isinstance(age_pool_minimum, int) or age_pool_minimum <= 0
    ):
        raise ValueError("age_pool_minimum must be a positive integer or None.")
    if state_pension_age is not None and (
        not isinstance(state_pension_age, int) or state_pension_age <= 0
    ):
        raise ValueError("state_pension_age must be a positive integer or None.")
    lowers = sorted(int(value) for value in lower_bounds)
    if not lowers or len(set(lowers)) != len(lowers):
        raise ValueError("band donor lower bounds must be distinct.")
    if SPI_INCOME_BAND_CARRIER_COLUMN not in person.columns:
        # A frame built without the band-donor stage (a synthetic fixture or
        # a test support) carries no reserved carriers; the receipt says so
        # rather than the stage inventing any.
        return person, {
            "carriers": 0,
            "skipped": (
                f"no {SPI_INCOME_BAND_CARRIER_COLUMN} column on the frame; "
                "spi_income_band_donors did not run before this stage"
            ),
            "bands": [],
        }
    _require_columns(
        household,
        ("household_id", "region", SPI_INCOME_BAND_LOWER_BOUND_COLUMN),
        label="band donor households",
    )
    _require_columns(
        person, ("person_id", "person_household_id"), label="band donor carriers"
    )
    for column in ("total_income", "is_composite", "region", "FACT"):
        if column not in donor.columns:
            raise ValueError(f"band donor resample needs donor column {column!r}.")
    flagged = person[SPI_INCOME_BAND_CARRIER_COLUMN].astype(bool)
    if (flagged & ~spi_people.astype(bool)).any():
        raise ValueError(
            "band donor carriers must be SPI recipients; a carrier outside the "
            "recipient domain would keep its twin's values and leave its band "
            "unfilled."
        )
    carriers = flagged
    if not carriers.any():
        raise ValueError("band donor resample found no reserved carriers.")
    by_household = household.set_index("household_id")
    carrier_rows = person.loc[carriers]
    bands = (
        carrier_rows["person_household_id"]
        .map(by_household[SPI_INCOME_BAND_LOWER_BOUND_COLUMN])
        .to_numpy(dtype=float)
    )
    regions = (
        carrier_rows["person_household_id"]
        .map(by_household["region"])
        .astype(str)
        .to_numpy()
    )
    unknown = sorted({int(value) for value in bands} - set(lowers))
    if unknown:
        raise ValueError(f"band donor carriers name undeclared band(s) {unknown}.")
    columns = list(SPI_INCOME_QRF_OUTPUT_COLUMNS)
    factors = np.asarray([float(uprating_factors[column]) for column in columns])
    # Restricted finance costs are a deduction the published total never held,
    # so they take no part in a record's band: the pools are cut on the income
    # leaves alone, and a drawn record's finance costs travel with its leaves
    # (microcosm#1106). The loss draws travel the same way: the published
    # total holds the profit net of losses brought forward, which
    # self_employment_income already carries (microcosm#1095).
    income_columns = [
        column
        for column in columns
        if column not in (SPI_PROPERTY_FINANCE_COSTS_COLUMN, *SPI_LOSS_DRAW_COLUMNS)
    ]
    income_factors = np.asarray(
        [float(uprating_factors[column]) for column in income_columns]
    )
    published_total = donor["total_income"].to_numpy(dtype=float)
    leaf_values = donor[columns].to_numpy(dtype=np.float64)
    income_values = donor[income_columns].to_numpy(dtype=np.float64)
    # The total income a drawn record realises on the frame: its income leaves
    # at the stage's factors plus whatever of the published total the leaves do
    # not carry, held nominal.
    total_income = income_values @ income_factors + np.maximum(
        published_total - income_values.sum(axis=1), 0.0
    )
    donor_region = donor["region"].astype(str).to_numpy()
    fact = donor["FACT"].to_numpy(dtype=float)
    composite = donor["is_composite"].to_numpy(dtype=bool)
    edges = [*lowers, np.inf]
    pools: dict[int, np.ndarray] = {}
    for position, lower in enumerate(lowers):
        pool = np.flatnonzero(
            (total_income >= lower) & (total_income < edges[position + 1])
        )
        if pool.size == 0:
            raise ValueError(
                "the SPI donor tape carries no record whose uprated total income "
                f"lies in the band from {lower}."
            )
        pools[lower] = pool
    if age_pool_minimum is not None and "agerange" not in donor.columns:
        raise ValueError("age-band pooling needs the prepared donor's agerange column.")
    donor_agerange = (
        donor["agerange"].to_numpy(dtype=np.int64)
        if age_pool_minimum is not None
        else None
    )
    # Ages are read only when age pooling is declared; a frame without it
    # keeps the income-band and regional pools unchanged.
    carrier_agerange = (
        _spi_agerange_codes(carrier_rows["age"].to_numpy(dtype=float))
        if donor_agerange is not None
        else np.zeros(len(carrier_rows), dtype=np.int64)
    )
    if state_pension_age is not None and "age" not in donor.columns:
        raise ValueError("State Pension age pooling needs the prepared donor's age.")
    donor_pension_age = (
        donor["age"].to_numpy(dtype=float) >= state_pension_age
        if state_pension_age is not None
        else None
    )
    carrier_pension_age = (
        carrier_rows["age"].to_numpy(dtype=float) >= state_pension_age
        if state_pension_age is not None
        else np.zeros(len(carrier_rows), dtype=bool)
    )
    key_column = (
        SPI_INCOME_BAND_CARRIER_KEY_COLUMN
        if SPI_INCOME_BAND_CARRIER_KEY_COLUMN in carrier_rows.columns
        else "person_id"
    )
    uniforms = stable_identity_uniforms(
        carrier_rows[key_column].to_numpy(),
        seed=seed,
        salt=SPI_INCOME_BAND_DONOR_LEAF_DRAW_SALT,
    )
    drawn = np.empty(len(carrier_rows), dtype=np.int64)
    matched = np.zeros(len(carrier_rows), dtype=bool)
    age_matched = np.zeros(len(carrier_rows), dtype=bool)
    side_matched = np.zeros(len(carrier_rows), dtype=bool)
    pool_cache: dict[
        tuple[int, str, int, bool], tuple[np.ndarray, bool, bool, bool]
    ] = {}
    for index, (band, region, code, pension_age) in enumerate(
        zip(bands, regions, carrier_agerange, carrier_pension_age, strict=True)
    ):
        lower = int(band)
        key = (lower, region, int(code), bool(pension_age))
        if key not in pool_cache:
            base = pools[lower]
            base_is_sided = False
            if donor_pension_age is not None:
                sided = base[donor_pension_age[base] == bool(pension_age)]
                if sided.size:
                    base, base_is_sided = sided, True
            base_is_aged = False
            if donor_agerange is not None:
                aged = base[donor_agerange[base] == int(code)]
                if aged.size >= age_pool_minimum:
                    base, base_is_aged = aged, True
            regional = base[donor_region[base] == region]
            regional_ok = regional.size >= regional_pool_minimum
            pool_cache[key] = (
                regional if regional_ok else base,
                regional_ok,
                base_is_aged,
                base_is_sided,
            )
        pool, matched[index], age_matched[index], side_matched[index] = pool_cache[key]
        # Inverse CDF over the pool in tape order at the carrier's own uniform.
        cumulative = np.cumsum(fact[pool])
        position = int(
            np.searchsorted(cumulative, uniforms[index] * cumulative[-1], side="right")
        )
        drawn[index] = int(pool[min(position, pool.size - 1)])
    leaves = leaf_values[drawn] * factors
    person = person.copy()
    person.loc[carriers, columns] = leaves
    realised_all = total_income[drawn]
    uppers = np.asarray([edges[position + 1] for position in range(len(lowers))])
    upper_by_carrier = uppers[np.searchsorted(np.asarray(lowers), bands)]
    outside = (realised_all < bands) | (realised_all >= upper_by_carrier)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} reserved band carrier(s) realise a total income "
            "outside their band after the uprated draw; the pools are cut on the "
            "same uprated total, so this is a stage defect."
        )
    band_rows = []
    for lower in lowers:
        mask = bands == lower
        pool = pools[lower]
        realized = realised_all[mask] if mask.any() else np.asarray([])
        band_rows.append(
            {
                "lower_bound": lower,
                "carriers": int(mask.sum()),
                "carriers_outside_band": int(outside[mask].sum()),
                "pool_records": int(pool.size),
                "pool_composite_records": int(composite[pool].sum()),
                "pool_weighted_taxpayers": float(fact[pool].sum()),
                "regional_matched_carriers": int(matched[mask].sum()),
                "age_matched_carriers": int(age_matched[mask].sum()),
                "state_pension_age_matched_carriers": int(side_matched[mask].sum()),
                "realized_mean_total_income": _band_mean_total_income(
                    realized, drawn[mask]
                ),
            }
        )
    receipt = {
        "carriers": int(carriers.sum()),
        "carriers_outside_band": int(outside.sum()),
        "pool_basis": (
            "uprated total income: stage-1 income leaves at the stage's uprating "
            "factors plus the published remainder held nominal; restricted "
            "finance costs cut no pool and are copied with the drawn record"
        ),
        "regional_pool_minimum": regional_pool_minimum,
        "age_pool_minimum": age_pool_minimum,
        "state_pension_age": state_pension_age,
        "seed": seed,
        "salt": SPI_INCOME_BAND_DONOR_LEAF_DRAW_SALT,
        "draw_key": key_column,
        "weighting": "FACT",
        "with_replacement": True,
        "total_income_basis": "published TEI + TII, 2022-23 terms",
        "bands": band_rows,
    }
    return person, receipt


@cache
def _spi_income_uprating_factors(year: int):
    """Rebase SPI money flows before conditioning on build-year FRS incomes."""
    from policyengine_uk import CountryTaxBenefitSystem

    if year < SPI_DONOR_INCOME_YEAR:
        raise ValueError("SPI income cannot be rebased before its source year.")
    if set(SPI_INCOME_UPRATING_VARIABLES) != set(SPI_INCOME_QRF_OUTPUT_COLUMNS):
        raise ValueError("Every SPI income output needs an explicit uprating rule.")
    system = CountryTaxBenefitSystem()
    source = system.parameters(f"{SPI_DONOR_INCOME_YEAR}-01-01")
    target = system.parameters(f"{year}-01-01")
    factors = {}
    columns = {}
    for column, variable_name in SPI_INCOME_UPRATING_VARIABLES.items():
        path = None
        factor = 1.0
        if variable_name is not None:
            variable = system.variables.get(variable_name)
            if variable is None:
                raise ValueError(
                    f"Missing SPI uprating variable {variable_name!r} for {column}."
                )
            path = getattr(variable, "uprating", None)
            if not isinstance(path, str) or not path:
                raise ValueError(f"Unsupported SPI uprating index for {column}.")
            before, after = source, target
            try:
                for part in path.split("."):
                    before, after = getattr(before, part), getattr(after, part)
            except AttributeError as error:
                raise ValueError(
                    f"Missing SPI uprating parameter {path!r} for {column}."
                ) from error
            if not np.isfinite([before, after]).all() or before <= 0 or after <= 0:
                raise ValueError(f"SPI uprating index must be positive for {column}.")
            factor = float(after / before)
        factors[column] = factor
        columns[column] = {
            "variable": variable_name,
            "index": path,
            "factor": factor,
            "basis": "model_index" if path else "held_nominal",
        }
        if variable_name in SPI_HISTORY_INDEX_DIFFERS_FROM_PROJECTION:
            columns[column]["dataset_projection_index"] = (
                SPI_HISTORY_INDEX_DIFFERS_FROM_PROJECTION[variable_name]
            )
    return factors, {
        "from_period": SPI_DONOR_INCOME_YEAR,
        "to_period": year,
        "columns": columns,
    }


def _initialize_frs_channel_columns(
    person: pd.DataFrame,
    *,
    base_people: pd.Series,
    columns: Mapping[str, float],
) -> pd.DataFrame:
    if not isinstance(columns, Mapping):
        raise TypeError("initialize_frs_channel_columns must be a mapping.")
    if not base_people.index.equals(person.index):
        raise ValueError("FRS base mask must align to the person table.")
    result = person.copy()
    for column, value in columns.items():
        if not isinstance(column, str) or not column:
            raise ValueError("initialize_frs_channel_columns keys must be strings.")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(
                f"initialize_frs_channel_columns[{column!r}] must be finite."
            )
        if column not in result:
            # Belt-and-braces zero: every initialized column has its base
            # rows overwritten on the next line, but a NaN default here would
            # silently survive any future wiring change.
            result[column] = 0.0
        result.loc[base_people, column] = numeric
    return result


def prepare_spi_donor_table(
    raw: pd.DataFrame,
    *,
    seed: int,
    age_model: SPIDonorAgeModel | None = None,
) -> pd.DataFrame:
    """Prepare a raw SPI 2022-23 tape exactly as the income stage does.

    The band-donor stage shares this preparation so its propensity table and
    the income stage's band pools read the same ages, regions and leaves.
    """

    return _prepare_spi_donor(raw, seed=seed, age_model=age_model)


def _prepare_spi_donor(
    raw: pd.DataFrame,
    *,
    seed: int,
    age_model: SPIDonorAgeModel | None = None,
) -> pd.DataFrame:
    """Prepare the donor tape; ``age_model`` draws ages by ONS shares.

    Without a model (synthetic tapes that ship no vendored population
    resource) each donor's age is uniform within its AGERANGE band; the
    production stages always pass the model the manifest declares.
    """

    _require_columns(raw, SPI_DONOR_REQUIRED_COLUMNS, label="SPI 2022-23 donor")
    numeric = pd.DataFrame(
        {
            column: pd.to_numeric(raw[column], errors="coerce")
            for column in SPI_DONOR_REQUIRED_COLUMNS
        }
    )
    _require_finite_numeric(numeric, label="SPI 2022-23 donor")
    if not (numeric["FACT"] > 0.0).all():
        raise ValueError("SPI 2022-23 FACT weights must be strictly positive.")
    sex = numeric["SEX"]
    if not sex.isin([0, 1, 2]).all():
        raise ValueError("SPI 2022-23 SEX must contain only documented codes 0/1/2.")
    age_codes = numeric["AGERANGE"].astype(int)
    unknown_age = sorted(set(age_codes) - set(_SPI_AGE_RANGES))
    if unknown_age:
        raise ValueError(f"SPI 2022-23 has unknown AGERANGE code(s): {unknown_age}.")
    rng = np.random.default_rng(seed)
    gender = np.select((sex == 1, sex == 2), ("MALE", "FEMALE"), default="UNKNOWN")
    age_draw_receipt: list[dict[str, object]] | None = None
    if age_model is None:
        bounds = np.asarray([_SPI_AGE_RANGES[code] for code in age_codes])
        ages = bounds[:, 0] + rng.random(len(raw)) * (bounds[:, 1] - bounds[:, 0])
    else:
        ages, age_draw_receipt = _draw_spi_donor_ages(
            age_codes.to_numpy(dtype=np.int64),
            np.asarray(gender, dtype=object),
            numeric["SRP"].to_numpy(dtype=np.float64) > 0.0,
            numeric["FACT"].to_numpy(dtype=np.float64),
            rng,
            model=age_model,
        )
    donor = pd.DataFrame(
        {
            "age": ages,
            "gender": gender,
            "region": numeric["GORCODE"]
            .astype(int)
            .map(_SPI_REGION_MAP)
            .fillna("UNKNOWN"),
            "FACT": numeric["FACT"],
            # PolicyEngine's persisted employment input follows the exact
            # enhanced-FRS precedent. It must not be widened to HMRC's
            # published employed-income measure.
            "employment_income": (numeric["PAY"] + numeric["EPB"] + numeric["TAXTERM"]),
            "self_employment_income": np.maximum(
                numeric["PROFITS"] - numeric["CAPALL"] - numeric["LOSSBF"],
                0.0,
            ),
            # The year's profit before losses brought forward and the losses
            # themselves, which the stage moves into loss_relief
            # (microcosm#1095).
            SPI_PROFIT_BEFORE_LOSSES_DRAW_COLUMN: np.maximum(
                numeric["PROFITS"] - numeric["CAPALL"], 0.0
            ),
            SPI_LOSSES_BROUGHT_FORWARD_DRAW_COLUMN: np.maximum(numeric["LOSSBF"], 0.0),
        }
    )
    for output, sources in _DIRECT_SPI_OUTPUT_SOURCE_COLUMNS.items():
        donor[output] = numeric[list(sources)].sum(axis=1)
    # The published total income (TEI + TII) and the composite indicator stay
    # on the prepared tape for the band donors' propensity and band pools;
    # they are never QRF inputs or outputs.
    donor["total_income"] = numeric["TI"].to_numpy(dtype=float)
    donor["is_composite"] = numeric["AGERANGE"].astype(int).eq(-1).to_numpy()
    # The published age band travels with the record so band pools can match
    # a carrier's age band; composites (-1) have none and never age-match.
    donor["agerange"] = age_codes.to_numpy(dtype=np.int64)
    # The stage-1 forests read the band itself, a composite's -1 included
    # (uk-data#628).
    donor[SPI_STAGE1_AGE_BAND_COLUMN] = age_codes.to_numpy(dtype=np.float64)
    # The bridge predictors (microcosm#840): main_source and has_pay from the
    # leaves, as on the twin; has_trade and at_or_over_spa from the published
    # indicator codes, the leaves serving a composite record without codes;
    # the receipt validates the leaf-defined main source against MAINSRCE and
    # reports the indicator codes against the leaves both ways.
    _validate_spi_bridge_codes(numeric)
    seinc = numeric["SEINC_NUM"].astype(int).to_numpy()
    spa = numeric["SPA"].astype(int).to_numpy()
    donor["main_source"] = _main_source(
        donor["employment_income"].to_numpy(dtype=np.float64),
        (
            donor["private_pension_income"]
            + donor[SPI_HMRC_STATE_PENSION_INCOME_COLUMN]
        ).to_numpy(dtype=np.float64),
        donor["self_employment_income"].to_numpy(dtype=np.float64),
    )
    donor["has_pay"] = donor["employment_income"].gt(0.0).to_numpy(dtype=np.float64)
    donor["has_trade"] = np.where(
        seinc == -1, donor["self_employment_income"].to_numpy() > 0.0, seinc == 1
    ).astype(np.float64)
    donor["at_or_over_spa"] = np.where(
        spa == -1, numeric["SRP"].to_numpy(dtype=np.float64) > 0.0, spa == 1
    ).astype(np.float64)
    donor.attrs["bridge"] = _spi_bridge_receipt(numeric, donor)
    donor.attrs["age_draw"] = {
        "method": (
            SPI_DONOR_AGE_DRAW_METHOD
            if age_model is not None
            else "uniform within the AGERANGE band (no population model)"
        ),
        "source": None if age_model is None else age_model.source,
        "state_pension_age": (
            None if age_model is None else age_model.state_pension_age
        ),
        "cells": age_draw_receipt or [],
    }
    donor.attrs["property_finance_costs"] = _check_spi_property_finance_costs(numeric)
    _require_finite_numeric(
        donor[["age", "FACT", *SPI_INCOME_QRF_OUTPUT_COLUMNS]],
        label="SPI 2022-23 derived donor",
    )
    nonnegative_outputs = [
        column
        for column in SPI_INCOME_QRF_OUTPUT_COLUMNS
        if column != SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN
    ]
    if (donor[nonnegative_outputs].to_numpy(dtype=float) < 0.0).any():
        raise ValueError(
            "SPI 2022-23 non-negative donor outputs contain negative values."
        )

    # TI, TEI and TII remain source-validation fields only. The QRF draws the
    # documented leaves, and the synthetic aggregates are derived after every
    # draw so the accounting identity holds by construction.
    _validate_spi_source_leaf_reconciliation(numeric)
    ti_error = np.abs(numeric["TI"] - (numeric["TEI"] + numeric["TII"]))
    if (ti_error > SPI_TI_IDENTITY_ABS_TOLERANCE_GBP).any():
        worst = float(ti_error.max())
        raise ValueError(
            "SPI 2022-23 TI disagrees with the published TEI + TII identity; "
            f"worst absolute difference {worst:.6g} exceeds the reviewed "
            f"£{SPI_TI_IDENTITY_ABS_TOLERANCE_GBP:.0f} rounding tolerance."
        )
    for column in ("gift_aid", "charitable_investment_gifts"):
        if (donor[column] < 0.0).any():
            raise ValueError(f"SPI donor {column} must be non-negative.")
    return donor


#: The tape's TAX_CRED carries the 20% basic-rate reduction on restricted
#: residential finance costs (ITTOIA 2005 s. 274A; 2022-23) together with the
#: other tax credits, so on landlords with restricted costs its median ratio
#: to LLIR_RESTRICT_AMT_TOT sits at the relief rate. The band refuses a tape
#: whose column means something else (microcosm#1106).
SPI_PROPERTY_FINANCE_COST_RELIEF_RATIO_BOUNDS = (0.195, 0.205)


def _check_spi_property_finance_costs(numeric: pd.DataFrame) -> dict[str, object]:
    """Refuse a tape whose restricted finance costs do not carry 20% relief."""

    costs = numeric["LLIR_RESTRICT_AMT_TOT"].to_numpy(dtype=np.float64)
    credits = numeric["TAX_CRED"].to_numpy(dtype=np.float64)
    weights = numeric["FACT"].to_numpy(dtype=np.float64)
    if (costs < 0.0).any():
        raise ValueError("SPI 2022-23 LLIR_RESTRICT_AMT_TOT contains negative costs.")
    restricted = costs > 0.0
    low, high = SPI_PROPERTY_FINANCE_COST_RELIEF_RATIO_BOUNDS
    receipt: dict[str, object] = {
        "column": "LLIR_RESTRICT_AMT_TOT",
        "relief_column": "TAX_CRED",
        "rows_with_restricted_costs": int(restricted.sum()),
        "weighted_landlords_with_restricted_costs": float(weights[restricted].sum()),
        "weighted_restricted_costs_gbp": float(np.dot(costs, weights)),
        "median_relief_ratio_bounds": [low, high],
    }
    if not restricted.any():
        # A tape without restricted costs (the synthetic test tapes) donates no
        # finance costs; property_components refuses such a donor, so the
        # build fails closed where the costs are consumed.
        receipt["median_relief_ratio"] = None
        return receipt
    ratio = float(np.median(credits[restricted] / costs[restricted]))
    receipt["median_relief_ratio"] = ratio
    if not low <= ratio <= high:
        raise ValueError(
            "SPI 2022-23 TAX_CRED does not carry the 20% finance-cost relief on "
            f"LLIR_RESTRICT_AMT_TOT: median ratio {ratio:.4f} outside "
            f"[{low}, {high}]."
        )
    return receipt


def _validate_spi_source_leaf_reconciliation(numeric: pd.DataFrame) -> None:
    """Check Annex A leaf formulas within the pinned PUT anonymization envelope."""

    source_tei = (
        np.maximum(0.0, numeric["PAY"] + numeric["EPB"] - numeric["EXPS"])
        + numeric["INCPBEN"]
        + numeric["OSSBEN"]
        + numeric["TAXTERM"]
        + numeric["UBISJA"]
        + numeric["MOTHINC"]
        + numeric["OTHERINC"]
        + numeric["SRP"]
        + numeric["PENSION"]
        + np.maximum(
            0.0,
            numeric["PROFITS"] - numeric["CAPALL"] - numeric["LOSSBF"],
        )
    )
    source_tii = (
        numeric["OTHERINV"]
        + numeric["DIVIDENDS"]
        + numeric["INCPROP"]
        + numeric["INCBBS"]
    )
    derived = {"TEI": source_tei, "TII": source_tii, "TI": source_tei + source_tii}
    composite = numeric["AGERANGE"].eq(-1).to_numpy(dtype=bool)
    groups = {"ordinary": ~composite, "composite": composite}
    for group, mask in groups.items():
        if not mask.any():
            continue
        for field, tolerance in SPI_SOURCE_LEAF_RECONCILIATION_ABS_TOLERANCE_GBP[
            group
        ].items():
            error = np.abs(
                numeric.loc[mask, field].to_numpy(dtype=float)
                - derived[field].loc[mask].to_numpy(dtype=float)
            )
            if (error > tolerance).any():
                worst = float(error.max())
                raise ValueError(
                    "SPI 2022-23 source-leaf reconciliation failed for "
                    f"{group} {field}; worst absolute difference {worst:.6g} "
                    f"exceeds the reviewed £{tolerance:.0f} PUT anonymization "
                    "envelope."
                )


def _seed_frs_hmrc_auxiliary_leaves(
    person: pd.DataFrame,
    *,
    spi_people: pd.Series,
) -> pd.DataFrame:
    """Validate retained FRS leaves without manufacturing unavailable concepts."""

    if len(spi_people) != len(person):
        raise ValueError("SPI person mask must align to the person table.")
    assert_frs_hmrc_auxiliary_crosswalk_available(person)
    return person.copy()


def _derive_policyengine_employment_input(
    person: pd.DataFrame,
    *,
    spi_people: pd.Series,
) -> pd.DataFrame:
    """Set SPI employment_income to PAY + EPB + TAXTERM by construction."""

    required = (
        SPI_HMRC_PAY_COLUMN,
        SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
        SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    )
    _require_columns(person, required, label="PolicyEngine SPI employment crosswalk")
    leaves = person.loc[spi_people, list(required)].apply(
        pd.to_numeric, errors="coerce"
    )
    _require_finite_numeric(leaves, label="PolicyEngine SPI employment leaves")
    employment = leaves.sum(axis=1).to_numpy(dtype=float)
    if (employment < 0.0).any():
        raise ValueError("Derived PolicyEngine SPI employment_income is negative.")
    result = person.copy()
    if "employment_income" not in result:
        result["employment_income"] = 0.0
    result.loc[spi_people, "employment_income"] = employment
    return result


def derive_hmrc_income_auxiliaries(
    person: pd.DataFrame,
    *,
    row_mask: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Derive TEI, TII and TI exactly, optionally only on selected rows.

    The whole-frame default preserves the exact synthetic/calibration contract.
    A row mask is used by the mixed FRS/SPI stage: full concepts absent from the
    FRS instrument remain unavailable there, while SPI rows receive identities
    derived deterministically from their joint donor draws.
    """

    required = (
        *SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS,
        SPI_HMRC_OTHER_INCOME_COLUMN,
        SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
        "self_employment_income",
        "private_pension_income",
        "savings_interest_income",
        "tax_free_savings_income",
        "dividend_income",
        "property_income",
        "other_investment_income",
    )
    _require_columns(person, required, label="HMRC accounting identity")
    mask = _hmrc_auxiliary_row_mask(person, row_mask)
    numeric = person.loc[mask, list(required)].apply(pd.to_numeric, errors="coerce")
    _require_finite_numeric(numeric, label="HMRC accounting identity leaves")
    nonnegative = [
        column
        for column in required
        if column != SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN
    ]
    if (numeric[nonnegative].to_numpy(dtype=float) < 0.0).any():
        raise ValueError(
            "HMRC non-negative accounting identity leaves contain negative values."
        )

    if LOSS_RELIEF_COLUMN in person.columns:
        # HMRC's earned income holds the profit after losses brought forward,
        # which policyengine-uk keeps apart in loss_relief (microcosm#1095).
        relief = pd.to_numeric(person.loc[mask, LOSS_RELIEF_COLUMN], errors="coerce")
        numeric["self_employment_income"] = self_employment_income_after_loss_relief(
            numeric["self_employment_income"].to_numpy(dtype=float),
            relief.fillna(0.0).to_numpy(dtype=float),
        )
    taxable_interest = (
        numeric["savings_interest_income"] - numeric["tax_free_savings_income"]
    )
    if (taxable_interest < 0.0).any():
        raise ValueError(
            "HMRC accounting identity requires gross savings_interest_income "
            "to be at least tax_free_savings_income on every person row."
        )
    employed_income = (
        np.maximum(
            numeric[SPI_HMRC_PAY_COLUMN]
            + numeric[SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN]
            - numeric[SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN],
            0.0,
        )
        + numeric[SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN]
        + numeric[SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN]
        + numeric[SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN]
        + numeric[SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN]
        + numeric[SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN]
    ).to_numpy(dtype=float)
    total_earned = (
        employed_income
        + numeric[SPI_HMRC_OTHER_INCOME_COLUMN]
        + numeric[SPI_HMRC_STATE_PENSION_INCOME_COLUMN]
        + numeric["self_employment_income"]
        + numeric["private_pension_income"]
    ).to_numpy(dtype=float)
    total_investment = (
        taxable_interest
        + numeric["dividend_income"]
        + numeric["property_income"]
        + numeric["other_investment_income"]
    ).to_numpy(dtype=float)
    assessable = total_earned + total_investment

    result = person.copy()
    derived = {
        SPI_HMRC_EMPLOYED_INCOME_COLUMN: employed_income,
        SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN: total_earned,
        SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN: total_investment,
        "hmrc_spi_assessable_income": assessable,
    }
    if row_mask is None:
        for column, values in derived.items():
            result[column] = values
    else:
        # A prior full-frame auxiliary is not valid evidence for the partial
        # FRS channel. Clear every derived output to the stage-time zero,
        # then materialize SPI only — the mask-restricted arithmetic below
        # never reads the cleared rows, and the artifact must not ship NaN.
        for column, values in derived.items():
            result[column] = 0.0
            result.loc[mask, column] = values
    if not np.array_equal(
        result.loc[mask, "hmrc_spi_assessable_income"].to_numpy(dtype=float),
        result.loc[mask, SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN].to_numpy(dtype=float)
        + result.loc[mask, SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN].to_numpy(
            dtype=float
        ),
    ):
        raise RuntimeError("HMRC TI must equal derived TEI + TII exactly.")
    return result


def _hmrc_auxiliary_row_mask(
    person: pd.DataFrame,
    row_mask: pd.Series | np.ndarray | None,
) -> np.ndarray:
    """Return a strict row-aligned boolean mask for optional HMRC derivation."""

    if row_mask is None:
        return np.ones(len(person), dtype=bool)
    if isinstance(row_mask, pd.Series):
        if not row_mask.index.equals(person.index):
            raise ValueError("HMRC auxiliary row mask index must align to person.")
        if row_mask.isna().any() or not pd.api.types.is_bool_dtype(row_mask.dtype):
            raise ValueError("HMRC auxiliary row mask must contain only booleans.")
        mask = row_mask.to_numpy(dtype=bool)
    else:
        raw = np.asarray(row_mask)
        if raw.ndim != 1 or raw.dtype.kind != "b":
            raise ValueError("HMRC auxiliary row mask must be one-dimensional bool.")
        mask = raw.astype(bool, copy=False)
    if len(mask) != len(person):
        raise ValueError("HMRC auxiliary row mask must align to the person table.")
    return mask


def _person_predictors(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    income_predictors: tuple[str, ...],
) -> pd.DataFrame:
    required = ("person_household_id", "age", "gender", *income_predictors)
    _require_columns(person, required, label="SPI QRF person predictors")
    _require_columns(household, ("household_id", "region"), label="household")
    region = household.set_index("household_id")["region"]
    mapped_region = person["person_household_id"].map(region)
    if mapped_region.isna().any():
        raise ValueError("SPI QRF cannot map every person to a household region.")
    result = person[["age", "gender", *income_predictors]].copy()
    result["region"] = mapped_region.to_numpy()
    result = result[["age", "gender", "region", *income_predictors]]
    _require_finite_numeric(
        result[["age", *income_predictors]],
        label="SPI QRF numeric predictors",
    )
    if result[["gender", "region"]].isna().any().any():
        raise ValueError("SPI QRF categorical predictors contain missing values.")
    return result


def _stage1_query_predictors(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    state_pension_age: int,
) -> pd.DataFrame:
    """Stage-1 query predictors: the SPI band of the whole-year age, and the
    bridge predictors derived from the row's own (the twin's) leaves."""

    predictors = _person_predictors(person, household, income_predictors=())
    predictors[SPI_STAGE1_AGE_BAND_COLUMN] = _spi_agerange_codes(
        np.floor(predictors["age"].to_numpy(dtype=np.float64))
    ).astype(np.float64)
    bridge = _person_bridge(person, state_pension_age=state_pension_age)
    for column in SPI_BRIDGE_PREDICTOR_COLUMNS:
        predictors[column] = bridge[column].to_numpy()
    return predictors[list(SPI_STAGE1_PREDICTOR_COLUMNS)]


def _main_source(pay: np.ndarray, pension: np.ndarray, trade: np.ndarray) -> np.ndarray:
    """The largest of pay, pension and trade; ``other`` when none is positive."""

    stacked = np.vstack([pay, pension, trade])
    labels = np.asarray(SPI_MAIN_SOURCE_LEVELS[:3], dtype=object)[
        stacked.argmax(axis=0)
    ]
    return np.where(
        stacked.max(axis=0) > 0.0, labels, SPI_MAIN_SOURCE_LEVELS[3]
    ).astype(object)


_BRIDGE_PERSON_COLUMNS = (
    "age",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "state_pension_reported",
    "employment_status",
)
_UNEARNED_INCOME_COLUMNS = (
    "savings_interest_income",
    "tax_free_savings_income",
    "dividend_income",
    "property_income",
    "miscellaneous_income",
)


def _person_bridge(person: pd.DataFrame, *, state_pension_age: int) -> pd.DataFrame:
    """The bridge predictors of FRS persons, from their own leaves.

    The formulas mirror the tape's in :func:`_prepare_spi_donor`: the main
    source is the largest of pay, pension (private plus State Pension) and
    trade; ``has_pay`` is pay above zero; ``has_trade`` is self-employment
    income above zero or a self-employed main job, the twin's analogue of the
    tape's self-employment pages; ``at_or_over_spa`` reads the engine's State
    Pension age at the build instant.
    """

    _require_columns(person, _BRIDGE_PERSON_COLUMNS, label="SPI bridge predictors")
    numeric = person[list(_BRIDGE_PERSON_COLUMNS[:-1])]
    _require_finite_numeric(numeric, label="SPI bridge predictors")
    pay = numeric["employment_income"].to_numpy(dtype=np.float64)
    pension = (
        numeric["private_pension_income"] + numeric["state_pension_reported"]
    ).to_numpy(dtype=np.float64)
    trade = numeric["self_employment_income"].to_numpy(dtype=np.float64)
    status = person["employment_status"].astype(str)
    bridge = pd.DataFrame(index=person.index)
    bridge["main_source"] = _main_source(pay, pension, trade)
    bridge["has_pay"] = (pay > 0.0).astype(np.float64)
    bridge["has_trade"] = (
        (trade > 0.0) | status.isin(SELF_EMPLOYED_STATUSES).to_numpy()
    ).astype(np.float64)
    bridge["at_or_over_spa"] = (
        numeric["age"].to_numpy(dtype=np.float64) >= float(state_pension_age)
    ).astype(np.float64)
    return bridge


def _taxable_income_mask(person: pd.DataFrame) -> np.ndarray:
    """Rows whose twin reports taxable income under the no-record rule."""

    _require_columns(
        person,
        (*_BRIDGE_PERSON_COLUMNS, *_UNEARNED_INCOME_COLUMNS),
        label="SPI recipient income rule",
    )
    numeric = person[[*_BRIDGE_PERSON_COLUMNS[1:-1], *_UNEARNED_INCOME_COLUMNS]]
    _require_finite_numeric(numeric, label="SPI recipient income rule")
    values = {column: numeric[column].to_numpy(dtype=np.float64) for column in numeric}
    status = person["employment_status"].astype(str)
    earned = (
        (values["employment_income"] > 0.0)
        | (values["self_employment_income"] > 0.0)
        | status.isin(SELF_EMPLOYED_STATUSES).to_numpy()
        | (values["private_pension_income"] + values["state_pension_reported"] > 0.0)
    )
    unearned = (
        np.maximum(
            values["savings_interest_income"] - values["tax_free_savings_income"], 0.0
        )
        + values["dividend_income"]
        + values["property_income"]
        + values["miscellaneous_income"]
    )
    return earned | (unearned > SPI_RECIPIENT_UNEARNED_INCOME_FLOOR_GBP)


def _validate_spi_bridge_codes(numeric: pd.DataFrame) -> None:
    """Refuse a tape whose bridge codes are not the documented ones, by code."""

    for column, allowed in (
        ("MAINSRCE", set(SPI_MAINSRCE_CODES)),
        ("SEINC_NUM", set(SPI_INDICATOR_CODES)),
        ("SPA", set(SPI_INDICATOR_CODES)),
    ):
        values = numeric[column].to_numpy(dtype=np.float64)
        if (values != np.round(values)).any():
            raise ValueError(f"SPI 2022-23 {column} must hold integer codes.")
        unknown = sorted(set(values.astype(int).tolist()) - allowed)
        if unknown:
            raise ValueError(
                f"SPI 2022-23 {column} has undocumented code(s): {unknown}."
            )


def _spi_bridge_receipt(numeric: pd.DataFrame, donor: pd.DataFrame) -> dict[str, Any]:
    """Validate the leaf-defined bridge against the published codes.

    Weighted shares over the whole tape; the leaf-defined main source must
    agree with the published MAINSRCE code on at least
    :data:`SPI_MAIN_SOURCE_MINIMUM_AGREEMENT` of the classified weight.
    """

    fact = numeric["FACT"].to_numpy(dtype=np.float64)
    codes = numeric["MAINSRCE"].astype(int).to_numpy()
    published = np.asarray(
        [SPI_MAINSRCE_CODES[int(code)] for code in codes], dtype=object
    )
    classified = codes != -1
    derived = donor["main_source"].to_numpy()
    agree = (derived == published) & classified

    def share(mask: np.ndarray, within: np.ndarray) -> float | None:
        denominator = float(fact[within].sum())
        if denominator <= 0.0:
            return None
        return float(fact[mask & within].sum() / denominator)

    by_code = {}
    for code in sorted(set(codes.tolist())):
        rows = codes == code
        by_code[SPI_MAINSRCE_CODE_LABELS[int(code)]] = {
            "code": int(code),
            "weighted_share": share(rows, np.ones(len(codes), dtype=bool)),
            "agreement": share(agree, rows) if code != -1 else None,
        }
    overall = share(agree, classified)
    pay = numeric["PAY"].to_numpy(dtype=np.float64) > 0.0
    profits = numeric["PROFITS"].to_numpy(dtype=np.float64) > 0.0
    seinc = numeric["SEINC_NUM"].astype(int).to_numpy() == 1
    srp = numeric["SRP"].to_numpy(dtype=np.float64) > 0.0
    spa = numeric["SPA"].astype(int).to_numpy() == 1
    receipt = {
        "predictors": list(SPI_BRIDGE_PREDICTOR_COLUMNS),
        "main_source": {
            "definition": (
                "largest of employment_income, private_pension_income + "
                "hmrc_spi_state_pension_income and self_employment_income; "
                "other when none is positive"
            ),
            "validation_column": "MAINSRCE",
            "minimum_agreement": SPI_MAIN_SOURCE_MINIMUM_AGREEMENT,
            "weighted_agreement": overall,
            "by_published_code": by_code,
        },
        "has_pay": {
            "definition": "employment_income > 0",
            "pay_share_among_main_source_pay": share(pay, codes == 1),
        },
        "has_trade": {
            "definition": "SEINC_NUM = 1; composites self_employment_income > 0",
            "seinc_num_share_among_positive_profits": share(
                seinc, profits & classified
            ),
            "positive_profits_share_among_seinc_num": share(
                profits, seinc & classified
            ),
        },
        "at_or_over_spa": {
            "definition": "SPA = 1; composites SRP > 0",
            "srp_share_among_spa": share(srp, spa),
            "spa_share_among_srp": share(spa, srp & classified),
        },
        "composite_weighted_share": share(~classified, np.ones(len(codes), dtype=bool)),
    }
    if overall is not None and overall < SPI_MAIN_SOURCE_MINIMUM_AGREEMENT:
        raise ValueError(
            "the leaf-defined main source no longer translates the published "
            f"MAINSRCE code: weighted agreement {overall:.4f} is below "
            f"{SPI_MAIN_SOURCE_MINIMUM_AGREEMENT}."
        )
    return receipt


def _sign_class_mismatches(
    receipt: Mapping[str, Any], tape: Any
) -> dict[str, dict[str, list[int]]]:
    """Located targets whose reference sign classes differ from the tape's."""

    mismatches: dict[str, dict[str, list[int]]] = {}
    for target in analogue_targets(SPI_INCOME_QRF_OUTPUT_COLUMNS):
        reference_classes = receipt["targets"][target].get("reference_classes")
        if reference_classes is None:
            continue
        tape_classes = [int(value) for value in tape.sign_classes(target)]
        if list(reference_classes) != tape_classes:
            mismatches[target] = {
                "reference": [int(value) for value in reference_classes],
                "tape": tape_classes,
            }
    return mismatches


def _encode_predictor_frames(
    frames: Sequence[pd.DataFrame],
    *,
    categorical: Sequence[str],
) -> tuple[pd.DataFrame, ...]:
    """One-hot the categorical predictors identically across several frames."""

    lengths = [len(frame) for frame in frames]
    combined = pd.concat(
        [frame.reset_index(drop=True) for frame in frames], ignore_index=True
    )
    present = [column for column in categorical if column in combined.columns]
    encoded = pd.get_dummies(combined, columns=present, drop_first=False, dtype=float)
    encoded = encoded.reindex(sorted(encoded.columns), axis=1)
    out = []
    start = 0
    for frame, length in zip(frames, lengths, strict=True):
        part = encoded.iloc[start : start + length].copy()
        part.index = frame.index
        _require_finite_numeric(part, label="encoded QRF predictors")
        out.append(part)
        start += length
    return tuple(out)


def _encode_predictor_pair(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded, target_encoded = _encode_predictor_frames(
        (train, target), categorical=("gender", "region")
    )
    return train_encoded, target_encoded


def _person_fit_frame(
    *,
    predictors: pd.DataFrame,
    targets: pd.DataFrame,
    weights: np.ndarray,
    weight_kind: WeightKind,
) -> Frame:
    if len(predictors) != len(targets) or len(predictors) != len(weights):
        raise ValueError("QRF predictors, targets, and weights must align.")
    ids = np.arange(1, len(predictors) + 1, dtype=np.int64)
    person = pd.concat(
        [predictors.reset_index(drop=True), targets.reset_index(drop=True)],
        axis=1,
    )
    person.insert(0, "person_household_id", ids)
    person.insert(0, "person_id", ids)
    household = pd.DataFrame({"household_id": ids})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray(weights), weight_kind)},
    )


def _stage2_outputs(person: pd.DataFrame) -> tuple[tuple[str, ...], dict[str, str]]:
    outputs: list[str] = []
    reviewed: dict[str, str] = {}
    missing_unreviewed: list[str] = []
    for column in FRS_ONLY_SPI_FILL_PERSON_COLUMNS:
        if column in SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS:
            if column in person:
                values = pd.to_numeric(person[column], errors="coerce").to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Reviewed-absent FRS-only output {column!r} contains "
                        "non-finite values."
                    )
                if (values != 0.0).any():
                    raise ValueError(
                        f"Reviewed-absent FRS-only output {column!r} now carries "
                        "non-default source signal; update and review the source "
                        "manifest before adding it to the QRF surface."
                    )
            reviewed[column] = SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS[column]
        elif column in person:
            outputs.append(column)
        else:
            missing_unreviewed.append(column)
    if missing_unreviewed:
        raise ValueError(
            "FRS-only stage cannot silently narrow missing output(s): "
            f"{missing_unreviewed}."
        )
    if not outputs:
        raise ValueError("FRS-only stage has no materializable outputs.")
    return tuple(outputs), reviewed


def _spi_pension_coherence(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    spi_people: pd.Series,
    twin_pay: np.ndarray,
    twin_employer: np.ndarray,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """Tie an SPI row's contributions to the pay the tape drew for it.

    Stage 2 draws the contributions from a forest trained on FRS rows. It reads
    the drawn pay, so whether the row has an employer contribution follows that
    pay, but it cannot scale amounts beyond the FRS's pay. So the employee
    (net-pay) contribution is capped at the row's pay, and a row the forest
    gave an employer contribution takes the FRS twin's rate times its own pay
    where the twin has one (``uk_runtime.pension_coherence``).
    """

    rows = spi_people.to_numpy(dtype=bool)
    pay = person.loc[spi_people, "employment_income"].to_numpy(dtype=np.float64)
    weights = _person_household_weights(person, household).to_numpy(dtype=float)[rows]
    sources = (
        person["person_household_id"]
        .map(household.set_index("household_id")["household_source_id"])
        .to_numpy()[rows]
    )
    employee_before = person.loc[spi_people, "employee_pension_contributions"].to_numpy(
        dtype=np.float64
    )
    employee, capped = cap_net_pay_contributions(employee_before, pay)
    employer_before = person.loc[spi_people, "employer_pension_contributions"].to_numpy(
        dtype=np.float64
    )
    employer, from_twin, bounded = twin_rate_employer_contributions(
        drawn=employer_before,
        pay=pay,
        twin_pay=twin_pay,
        twin_employer=twin_employer,
    )
    person.loc[spi_people, "employee_pension_contributions"] = employee
    person.loc[spi_people, "employer_pension_contributions"] = employer
    receipt = {
        "employee_capped_at_pay": coherence_receipt(
            changed=capped,
            before=employee_before,
            after=employee,
            weights=weights,
            source_ids=sources,
        ),
        "employer_at_twin_rate": coherence_receipt(
            changed=from_twin,
            before=employer_before,
            after=employer,
            weights=weights,
            source_ids=sources,
        ),
        "employer_bounded_by_ashe_ceiling": coherence_receipt(
            changed=bounded,
            before=employer_before,
            after=employer,
            weights=weights,
            source_ids=sources,
        ),
    }
    return person, receipt


def _person_household_weights(
    person: pd.DataFrame,
    household: pd.DataFrame,
) -> pd.Series:
    weights = household.set_index("household_id")["household_weight"]
    mapped = person["person_household_id"].map(weights)
    if mapped.isna().any() or not np.isfinite(mapped.to_numpy(dtype=float)).all():
        raise ValueError("Cannot resolve finite household weights for every person.")
    if not (mapped > 0.0).any():
        raise ValueError("Resolved person weights contain no positive mass.")
    return mapped.astype(float)


def _validate_predictions(
    predictions: pd.DataFrame,
    *,
    expected: tuple[str, ...],
    label: str,
) -> None:
    if tuple(predictions.columns) != tuple(expected):
        raise ValueError(
            f"{label} prediction surface mismatch: expected {list(expected)}, "
            f"got {list(predictions.columns)}."
        )
    _require_finite_numeric(predictions, label=f"{label} predictions")


def _require_finite_numeric(frame: pd.DataFrame, *, label: str) -> None:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if not finite.all():
        non_finite_counts = {
            str(column): int((~finite[:, position]).sum())
            for position, column in enumerate(numeric.columns)
            if (~finite[:, position]).any()
        }
        raise ValueError(
            f"{label} must contain only finite numeric values; non-finite "
            f"counts by column: {non_finite_counts}."
        )


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing column(s): {missing}.")


def _qrf_class():
    if QRF is not None:
        return QRF
    return import_module("microcosm.fit").QRF


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spi_donor_fingerprint(path: Path) -> _SPIDonorFingerprint:
    stat = path.stat()
    return _SPIDonorFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


def _assert_verified_spi_donor_current(identity: VerifiedSPIDonorIdentity) -> None:
    if identity._verification_token is not _SPI_DONOR_VERIFICATION_TOKEN:
        raise ValueError(
            "SPI donor identity must come from verify_spi_donor_identity()."
        )
    if _spi_donor_fingerprint(identity.path) != identity.fingerprint:
        raise RuntimeError(
            "SPI 2022-23 donor changed after identity verification; refusing "
            "to parse bytes not bound to the reviewed SHA-256."
        )


def _verify_spi_donor_identity(path: Path) -> VerifiedSPIDonorIdentity:
    """Backward-compatible private seam for synthetic-donor unit tests."""

    return verify_spi_donor_identity(path)


def _refresh_disability_derived_inputs(
    person: pd.DataFrame,
    *,
    spi_people: pd.Series,
    category_rates: UKDWPDisabilityCategoryRates | None = None,
    flag_rates: UKDWPDisabilityFlagRates | None = None,
) -> pd.DataFrame:
    """Keep category/flag inputs coherent with stage-2 reported amounts.

    Re-derives the eight disability columns for the SPI-redrawn rows with the
    FRS stage's own derivation and, unless rates are injected, the FRS stage's
    declared year rule, so the two paths cannot disagree on the parameter
    tree or the week constant (uk-data#475, uk-data#476). The engine is only
    touched when rates are not supplied.
    """

    from microcosm.build.uk_runtime import frs_disability

    if category_rates is None or flag_rates is None:
        from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule

        year = resolve_uk_year_rule(frs_disability.YEAR_RULE)
        if category_rates is None:
            category_rates = frs_disability.uk_dwp_disability_category_rates(year)
        if flag_rates is None:
            flag_rates = frs_disability.uk_dwp_disability_flag_rates(year)
    derived = frs_disability.derive_frs_disability(
        person.loc[spi_people],
        category_rates=category_rates,
        flag_rates=flag_rates,
    )
    for column in frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS:
        default = "NONE" if column.endswith("_category") else False
        _assign_spi_values(
            person,
            spi_people,
            column,
            derived[column].to_numpy(),
            default=default,
        )
    return person


def _refresh_carer_take_up_input(
    person: pd.DataFrame, *, spi_people: pd.Series
) -> pd.DataFrame:
    """Keep would_claim_carers_allowance coherent with the refilled receipt.

    The root stage sets the flag from reported Carer's Allowance; the SPI
    fill re-imputes that receipt on the redrawn rows, so the flag is
    re-derived here from the refilled amount. A stale flag would pay the
    allowance to a redrawn 35-hour carer whose refilled receipt is zero, or
    withhold it from one whose refilled receipt is positive (#882). This is
    the anchor ``carers_allowance_take_up`` reads after the SPI income chain:
    it refuses a flag that no longer follows reported receipt, and opens it
    to the carers the earnings limit alone keeps from the allowance
    (microcosm#1319).
    """

    if "would_claim_carers_allowance" not in person.columns:
        return person
    refilled = (
        pd.to_numeric(
            person.loc[spi_people, "carers_allowance_reported"], errors="coerce"
        )
        .fillna(0.0)
        .to_numpy()
        > 0.0
    )
    _assign_spi_values(
        person, spi_people, "would_claim_carers_allowance", refilled, default=False
    )
    return person


def _owner_manager_flags_as_numbers(person: pd.DataFrame) -> pd.DataFrame:
    """Hold the owner-manager flags as 0 or 1 for the FRS-only fill to draw."""

    for column in FRS_ONLY_SPI_FILL_OWNER_MANAGER_FLAGS:
        if column in person.columns:
            person[column] = person[column].astype(bool).astype(np.float64)
    return person


def _owner_manager_flags_from_draws(
    person: pd.DataFrame, *, spi_people: pd.Series
) -> pd.DataFrame:
    """Store the drawn owner-manager inputs as the root stage does.

    The fill's zero regime gate draws each flag as 0 or 1 on a redrawn row. The
    root stage's rules then hold on every row: the person who stands as an
    owner works in the company, its other inputs need an owner, and a company
    carries on a trade or a property business, never both (microcosm#1095).
    """

    present = [c for c in FRS_ONLY_SPI_FILL_OWNER_MANAGER_FLAGS if c in person]
    if not present:
        return person
    if len(present) != len(FRS_ONLY_SPI_FILL_OWNER_MANAGER_FLAGS):
        raise ValueError(
            "The owner-manager inputs are set together; the frame carries "
            f"only {present}."
        )
    flags = {
        column: pd.to_numeric(person[column], errors="coerce").to_numpy() > 0.5
        for column in present
    }
    redrawn = spi_people.to_numpy(dtype=bool)
    owner = flags["stands_as_sole_owner_or_partner_of_company"]
    main = flags["owned_company_is_main_employment"] & owner
    property_business = flags["owned_company_carries_on_property_business"] & main
    trade = flags["owned_company_carries_on_trade"] & main & ~property_business
    held = {
        "stands_as_sole_owner_or_partner_of_company": owner,
        "is_engaged_in_owned_company_trade": owner,
        "owned_company_is_main_employment": main,
        "owned_company_carries_on_trade": trade,
        "owned_company_carries_on_property_business": property_business,
    }
    for column, values in held.items():
        # FRS rows keep their root-stage answers; only redrawn rows move.
        person[column] = np.where(redrawn, values, flags[column])
    if "owned_company_income_share" in person:
        share = pd.to_numeric(
            person["owned_company_income_share"], errors="coerce"
        ).to_numpy(dtype=float)
        person["owned_company_income_share"] = np.where(redrawn & ~owner, 0.0, share)
    return person


def _refresh_esa_support_component(
    person: pd.DataFrame, *, spi_people: pd.Series
) -> pd.DataFrame:
    """Return a redrawn row's ESA group to the engine's default.

    The root stage reads the ESA group from the FRS benefits tab
    (``esa_includes_support_component``); the SPI fill re-imputes the ESA
    receipts on the redrawn rows from other FRS donors, so the twin's answer no
    longer describes them. The flag returns to true, policyengine-uk's own
    default for an award of one's own (microcosm#1095).
    """

    if "esa_includes_support_component" not in person.columns:
        return person
    _assign_spi_values(
        person,
        spi_people,
        "esa_includes_support_component",
        np.ones(int(spi_people.sum()), dtype=bool),
        default=True,
    )
    return person


#: The inputs the tape's losses brought forward move into: income tax relieves
#: them against the profit (ITA 2007 s.83), Class 4 against its profits (SSCBA
#: 1992 Sch 2 para 3).
SPI_LOSS_RELIEF_COLUMNS = (LOSS_RELIEF_COLUMN, "ni_class_4_losses_brought_forward")


def _move_losses_brought_forward_to_loss_relief(
    person: pd.DataFrame, *, spi_people: pd.Series
) -> pd.DataFrame:
    """Move the tape's losses brought forward out of the profit, into loss_relief.

    The chain draws ``self_employment_income`` net of losses brought forward,
    as the tape's profits are, and then the year's profit before them and the
    losses (LOSSBF). policyengine-uk relieves earlier losses itself: income
    tax against ``loss_relief`` and Class 4 against
    ``ni_class_4_losses_brought_forward``, while Universal Credit and HBAI read
    the year's profit. So a drawn row's profit becomes the year's profit and
    both inputs take its losses, held so that the profit after the relief is
    the profit the chain drew: where that is positive the losses were all
    relieved, so they are the gap; where it is zero they are at least the
    profit. Runs after the HMRC aggregates, which read the profit net of the
    relief either way, and drops the two draws (uk-data#528, microcosm#1095).
    """

    if SPI_LOSSES_BROUGHT_FORWARD_DRAW_COLUMN not in person.columns:
        return person
    rows = spi_people.reindex(person.index).fillna(False).to_numpy(dtype=bool)
    net = (
        pd.to_numeric(person.loc[rows, "self_employment_income"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    profit = np.maximum(
        pd.to_numeric(
            person.loc[rows, SPI_PROFIT_BEFORE_LOSSES_DRAW_COLUMN], errors="coerce"
        )
        .fillna(0.0)
        .to_numpy(dtype=float),
        net,
    )
    losses = (
        pd.to_numeric(
            person.loc[rows, SPI_LOSSES_BROUGHT_FORWARD_DRAW_COLUMN], errors="coerce"
        )
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    relief = np.where(net > 0.0, profit - net, np.maximum(losses, profit))
    person.loc[rows, "self_employment_income"] = profit
    for column in SPI_LOSS_RELIEF_COLUMNS:
        if column not in person.columns:
            person[column] = 0.0
        person.loc[rows, column] = relief
    return person.drop(columns=list(SPI_LOSS_DRAW_COLUMNS))


def _clear_trading_loss_beside_profit(
    person: pd.DataFrame, *, spi_people: pd.Series
) -> pd.DataFrame:
    """A redrawn row with a positive profit keeps no drawn trading loss.

    policyengine-uk takes a trade's profit in ``self_employment_income`` and
    the year's loss in ``trading_loss``, and the FRS rows never hold both
    (uk-data#528). Stage 2 draws the loss from the FRS rows, and its forests
    can mix a loss into a row the tape gives a profit, so the tape's profit
    holds and the loss goes (microcosm#1095).
    """

    if "trading_loss" not in person.columns:
        return person
    profit = pd.to_numeric(
        person.loc[spi_people, "self_employment_income"], errors="coerce"
    ).fillna(0.0)
    loss = pd.to_numeric(person.loc[spi_people, "trading_loss"], errors="coerce")
    _assign_spi_values(
        person,
        spi_people,
        "trading_loss",
        np.where(profit.to_numpy() > 0.0, 0.0, loss.fillna(0.0).to_numpy()),
        default=0.0,
    )
    return person


#: employment_status values of the self-employed (FRS EMPSTATI 3 and 4).
SELF_EMPLOYED_STATUSES = ("FT_SELF_EMPLOYED", "PT_SELF_EMPLOYED")


def _refresh_uc_start_up_period(
    person: pd.DataFrame, *, spi_people: pd.Series
) -> pd.DataFrame:
    """Clear the UC start-up period where a redrawn row is no longer self-employed.

    The root stage dates the period from the donor's claim and trade; a
    redrawn row keeps the donor's employment status but takes new imputed
    income, so the flag survives only while that status or the refilled
    self-employment income still shows self-employment (uk-data#527).
    """

    if "uc_is_in_startup_period" not in person.columns:
        return person
    rows = person.loc[spi_people]
    still_self_employed = rows["employment_status"].isin(
        SELF_EMPLOYED_STATUSES
    ).to_numpy() | (
        pd.to_numeric(rows["self_employment_income"], errors="coerce")
        .fillna(0.0)
        .to_numpy()
        != 0.0
    )
    _assign_spi_values(
        person,
        spi_people,
        "uc_is_in_startup_period",
        rows["uc_is_in_startup_period"].astype(bool).to_numpy() & still_self_employed,
        default=False,
    )
    return person


#: The drawn income whose absence keeps a twin's other-spell amount.
_OTHER_SPELLS_DRAWN_INCOME = {
    EMPLOYMENT_INCOME_OTHER_SPELLS: "employment_income",
    SELF_EMPLOYMENT_INCOME_OTHER_SPELLS: "self_employment_income",
}


def _refresh_annual_income_descriptors(
    person: pd.DataFrame, *, spi_people: pd.Series
) -> pd.DataFrame:
    """Hold a drawn row's annual income descriptors to its drawn incomes.

    The tape records income received over the tax year, so a drawn row's pay,
    profit and private pension already cover the whole year and each share is
    1. Its contributory benefits are stage-2 draws on the current concept with
    no spell of their own, so their share is 1 too. A drawn row keeps its
    twin's other-spell pay only where it drew no pay, and its twin's ceased
    self-employment only where it drew no profit: the bridge draws a record
    without pay for a twin who left a job in the year, and the twin's spell is
    then the only record of that year's pay. An undrawn copy keeps every
    descriptor of its twin, with its twin's incomes.
    """

    present = [
        column for column in FRS_ANNUAL_INCOME_OUTPUT_COLUMNS if column in person
    ]
    if not present:
        return person
    if len(present) != len(FRS_ANNUAL_INCOME_OUTPUT_COLUMNS):
        missing = sorted(set(FRS_ANNUAL_INCOME_OUTPUT_COLUMNS) - set(present))
        raise ValueError(
            f"SPI rows carry only some annual income descriptors; missing {missing}."
        )
    rows = person.loc[spi_people]
    for column in ANNUAL_SHARE_COLUMNS:
        _assign_spi_values(person, spi_people, column, np.ones(len(rows)), default=1.0)
    for column, drawn_column in _OTHER_SPELLS_DRAWN_INCOME.items():
        drew_income = (
            pd.to_numeric(rows[drawn_column], errors="coerce").fillna(0.0).to_numpy()
            != 0.0
        )
        twin_spells = rows[column].to_numpy(dtype=np.float64)
        _assign_spi_values(
            person,
            spi_people,
            column,
            np.where(drew_income, 0.0, twin_spells),
            default=0.0,
        )
    return person


def _refresh_uc_gainful_self_employment(
    person: pd.DataFrame, *, spi_people: pd.Series
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Derive a drawn row's UC gainful self-employment from the root's rule.

    A drawn row keeps its twin's employment status and takes pay and profit
    drawn at the twin's rank given that status (``has_trade`` conditions the
    draw; microcosm#840), so the root's rule applies whole: a self-employed
    main job whatever its profit, or a side trade whose profit is above zero
    and above the row's pay (uk-data#525, ADM H4034). An undrawn copy keeps
    its twin's flag with its twin's incomes. The receipt counts the rows each
    route flags.
    """

    if "uc_is_in_gainful_self_employment" not in person.columns:
        return person, None
    rows = person.loc[spi_people]
    main_job = rows["employment_status"].astype(str).isin(SELF_EMPLOYED_STATUSES)
    main_job = main_job.to_numpy(dtype=bool)
    profit = pd.to_numeric(rows["self_employment_income"], errors="coerce").fillna(0.0)
    pay = pd.to_numeric(rows["employment_income"], errors="coerce").fillna(0.0)
    earnings_route = uc_gainful_self_employment(
        np.zeros(len(rows), dtype=bool), profit, pay
    )
    flagged = uc_gainful_self_employment(main_job, profit, pay)
    _assign_spi_values(
        person,
        spi_people,
        "uc_is_in_gainful_self_employment",
        flagged,
        default=False,
    )
    receipt = {
        "rule": (
            "self-employed main job (FT_SELF_EMPLOYED or PT_SELF_EMPLOYED) whatever "
            "its profit, or a profit above zero and above pay; undrawn copies keep "
            "their twin's flag"
        ),
        "rows": int(len(rows)),
        "flagged_rows": int(flagged.sum()),
        "main_job_route_rows": int(main_job.sum()),
        "earnings_route_rows": int(earnings_route.sum()),
        "main_job_route_only_rows": int((flagged & ~earnings_route).sum()),
    }
    return person, receipt


def _refresh_landlord_carrier(
    person: pd.DataFrame, *, spi_people: pd.Series
) -> pd.DataFrame:
    """Clear the FRS landlord carrier on a redrawn row.

    The carrier records that the FRS respondent reported rent, at a profit or
    a loss. A redrawn row takes its property income from an SPI donor that did
    not read it, so the twin's answer says nothing about the row's letting,
    and the tape has no letting-loss field to rebuild it from. The CGT stages
    read the row's drawn ``property_income`` instead (microcosm#1095).
    """

    if FRS_LANDLORD_CARRIER_COLUMN not in person.columns:
        return person
    rows = person.loc[spi_people]
    _assign_spi_values(
        person,
        spi_people,
        FRS_LANDLORD_CARRIER_COLUMN,
        np.zeros(len(rows), dtype=bool),
        default=False,
    )
    return person


def _assign_spi_values(
    person: pd.DataFrame,
    spi_people: pd.Series,
    column: str,
    values: np.ndarray,
    *,
    default: object,
) -> None:
    if column not in person:
        person[column] = default
    elif isinstance(person[column].dtype, pd.CategoricalDtype):
        person[column] = person[column].astype(object)
    person.loc[spi_people, column] = values


__all__ = [
    "DEFAULT_SPI_DONOR_SAMPLE_SIZE",
    "FRS_HMRC_AUXILIARY_SOURCE_COLUMNS",
    "FRS_HMRC_UNAVAILABLE_FULL_CONCEPT_COLUMNS",
    "FRS_ONLY_FIT_NAME",
    "SPI_DONOR_DOI",
    "SPI_DONOR_FILENAME",
    "SPI_DONOR_FIT_NAME",
    "SPI_DONOR_RELEASE",
    "SPI_DONOR_SHA256",
    "SPI_DONOR_SIZE_BYTES",
    "SPI_DONOR_UKDS_STUDY",
    "SPI_DONOR_VINTAGE",
    "SPI_DONOR_REQUIRED_COLUMNS",
    "SPI_INCOME_RECIPIENT_COLUMN",
    "SPI_INCOME_SOURCE_COLUMNS",
    "SPI_DERIVED_POLICYENGINE_SOURCE_COLUMNS",
    "SPI_DONOR_DOCUMENTATION_URL",
    "SPI_HMRC_EMPLOYED_INCOME_FORMULA",
    "SPI_HMRC_EMPLOYED_INCOME_SOURCE_COLUMN_MAP",
    "SPI_POLICYENGINE_EMPLOYMENT_FORMULA",
    "SPI_POLICYENGINE_EMPLOYMENT_SOURCE_COLUMNS",
    "SPI_QRF_SOURCE_COLUMNS",
    "SPI_SOURCE_COMPOSITE_INDICATOR",
    "SPI_INCOME_BAND_CARRIER_COLUMN",
    "SPI_INCOME_BAND_CARRIER_KEY_COLUMN",
    "SPI_INCOME_BAND_DONOR_LEAF_DRAW_SALT",
    "SPI_INCOME_BAND_LOWER_BOUND_COLUMN",
    "prepare_spi_donor_table",
    "SPI_SOURCE_LEAF_RECONCILIATION_ABS_TOLERANCE_GBP",
    "SPI_SOURCE_TEI_FORMULA",
    "SPI_SOURCE_TII_FORMULA",
    "SPI_SOURCE_TI_FORMULA",
    "SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS",
    "SPI_TI_IDENTITY_ABS_TOLERANCE_GBP",
    "SPI_BRIDGE_PREDICTOR_COLUMNS",
    "SPI_INDICATOR_CODES",
    "SPI_MAINSRCE_CODES",
    "SPI_MAIN_SOURCE_LEVELS",
    "SPI_MAIN_SOURCE_MINIMUM_AGREEMENT",
    "SPI_RECIPIENT_INCOME_RULE",
    "SPI_RECIPIENT_UNEARNED_INCOME_FLOOR_GBP",
    "SPI_STAGE1_ANALOGUE_COLUMNS",
    "SPI_STAGE1_CATEGORICAL_PREDICTORS",
    "SPI_STAGE1_PREDICTOR_COLUMNS",
    "UKSPIIncomeImputationResult",
    "VerifiedSPIDonorIdentity",
    "assert_frs_hmrc_auxiliary_crosswalk_available",
    "derive_hmrc_income_auxiliaries",
    "impute_uk_spi_income_support",
    "verify_spi_donor_identity",
]
