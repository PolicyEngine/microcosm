"""Benefit reports and take-up on SPI-channel rows (microcosm#1095, uk-data#514).

The SPI support channel copies whole FRS households, replaces their adults'
incomes with SPI draws and re-imputes each adult's benefit reports with the
FRS-only stage-2 forests of ``hmrc_spi_income_spine``, which condition on age,
gender, region and the new incomes and see nothing of the benefit unit's
entitlement or claim history. On the 30 September national build the SPI rows
reported Income Support, Working and Child Tax Credit and income-based JSA at
2.6 to 9.1 times the FRS rate and contributory JSA at 17 times (two thirds of
its SPI reporters earning above the permitted-work limit), held 64%, 72% and
67% of the reported IIDB, AFCS and bereavement support, kept the FRS twin's
``receives_benefits_in_own_right`` (3.04m SPI persons disagreed with their own
reports) and kept the twin's early Universal Credit take-up anchor (1.51m SPI
benefit units claimed only because the twin reported Universal Credit).

This stage runs on SPI-channel rows only, after ``uc_reporter_redraw`` (the
last writer of Universal Credit reports) and before ``uc_capital_coherence``
and the Pension Credit and Child Benefit take-up stages, which read
qualifying-young-person status and the taxable benefits:

- it zeroes the reports whose award needs an existing legacy claim, an
  out-of-work test, or a claim the copied household's new incomes cannot
  support: income-based JSA, income-related ESA, Income Support, Working and
  Child Tax Credit, Severe Disablement Allowance and the Sure Start Maternity
  Grant. policyengine-uk pays income-related ESA as
  reported behind a capital test alone, so a kept draw paid the twin's award
  whatever the copy's new incomes: on the microcosm#1095 arm A8 spine the SPI
  rows held 29% of the income-related ESA reporters at design weights, and
  56% of their benefit units had incomes above GBP 12,570;
- it restores Industrial Injuries Disablement Benefit, the Armed Forces
  Compensation Scheme and bereavement support from the FRS twin, since they
  follow an injury, service or a death and not income;
- it gives contributory JSA and ESA the FRS twin's award where the copy's own
  work stays within each benefit's limits, and none outside them (María's
  ruling of 29 October 2026, microcosm#1095 B10). A contributory award
  follows the person's National Insurance record and, for ESA, an incapacity
  the copy keeps; what the copy changes is its work. ESA's is permitted work
  (ESA Regs 2008 reg 45(4)): under 16 hours a week, and earnings up to
  policyengine-uk's own ``esa_exempt_work_higher_earnings_limit`` at the
  frame's year (GBP 183.50 a week in 2024-25). JSA's is its own (Jobseekers
  Act 1995 ss. 1(2)(e), 2(1)(c) and 4(1)(b), Vahid's review of
  microcosm#1426): under 16 hours a week (JSA Regs 1996 reg 51(1)(a)),
  earnings no higher than the prescribed amount, the age-related amount plus
  the earnings disregard less 1p (reg 56; GBP 95.49 a week from 25 in
  2024-25), and an award of at most the age-related amount less the earnings
  above the disregard (reg 80), all read from policyengine-uk's parameters
  at the frame's year. policyengine-uk pays both awards as reported;
- it re-derives the disability flags from the final reports with the FRS
  stage's rule (the categories, which read only AA, DLA and PIP, are
  unchanged and asserted so);
- it rebuilds ``receives_benefits_in_own_right`` from each person's own
  Universal Credit, JSA and ESA reports with ``frs_education``'s rule;
- it redraws ``would_claim_uc`` for SPI benefit units as ``frs_take_up`` drew
  it for the FRS units: a unit with a Universal Credit reporter claims, and the
  other units in the Universal Credit age population claim as the residual of
  the take-up contract's rate, on draws keyed on the SPI unit's own identity.
  That draw stands for a unit with no award: ``uc_take_up`` redraws every
  entitled unit of both channels by award band and family type once
  ``uc_capital_coherence`` has set the capital the engine assesses
  (microcosm#1334).

The disability benefits, Carer's Allowance, State Pension and Winter Fuel
Payment keep their draws, and FRS base rows are never modified. Before B10,
contributory ESA kept its forest draw, which on the A8 spine sat at half the
FRS rate per working-age adult with 9.9% of its SPI reporters by weight earning
above the permitted-work limit (uk-data#514 zeroes it on a build where 41%
did), and contributory JSA was zeroed, two thirds of its SPI reporters on the
30 September build earning above the limit. B10 first gave contributory JSA
ESA's limit too, about twice JSA's own. The limits are tested on the copy's
gross earnings, its employment income plus any self-employment profit over 52
weeks. The benefits' own tests read net earnings, but up to the limits pay and
profit sit under the personal allowance and the National Insurance
thresholds, so the two differ only by half of any pension contributions,
which the gross test leaves in. The SPI tape has no hours, so the copy keeps
its twin's usual hours (``hours_worked`` over 52 weeks), which the SPI pay
floor also reads as the copy's (uk-data#629), and the hours tests read them.
The tape's pay covers the tax year, so averaging it over 52 weeks is strict
for anyone who worked part of the year: a copy whose pay was earned in 26
weeks and which claimed in the other 26 is tested on half its weekly pay in
work, not on the nil it earned while claiming. The tests therefore understate
part-year claimants, above all of contribution-based JSA, which lasts at most
182 days (Jobseekers Act 1995 s. 5(1)).

Neither award is reduced for the copy's pension payments, though the copy
carries the tape's pension income (``private_pension_income``, from its
PENSION): JSA Regs 1996 reg 81 takes every pound of pension payments above
policyengine-uk's ``gov.dwp.JSA.contrib.pension_disregard`` (GBP 50 a week)
off contribution-based JSA, and ESA Regs 2008 reg 74 half of every pound
above GBP 85 a week off contributory ESA. B10 tests the copy's work alone, so
the receipt shows how much rests on the omission: for each award, the support
and weighted award of the copies whose pension payments are above its
threshold, the most a reduction could take off (Vahid's review of
microcosm#1426).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    assign_binary_with_anchored_residual,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime import frs_disability
from microcosm.build.uk_runtime.frs_education import (
    BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS,
)
from microcosm.build.uk_runtime.frs_take_up import (
    UK_UC_AGE_ELIGIBLE_AGGREGATE,
    UK_UC_AGE_ELIGIBLE_METHOD,
    UK_UC_AGE_ELIGIBLE_ROLE,
    UK_UC_AGE_ELIGIBLE_SOURCE,
    UK_UC_TAKE_UP_OUTPUT,
    UKTakeUpPopulationPolicy,
    assert_take_up_stage_population_declaration,
    uc_age_eligible_benunits,
    uk_take_up_population_policy,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    UK_DISCLOSURE_FLOOR,
    UK_WITHHELD,
    distinct_count,
    record_source_households,
    support_band,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    fill_support_channel_from_source,
    support_channel_column,
    support_source_id_column,
)
from microcosm.build.uk_runtime.take_up_contract import (
    UKTakeUpContract,
    load_uk_take_up_contract,
)
from microcosm.build.uk_runtime.uc_capital_coherence import (
    _household_to_benunit_weights,
)
from microcosm.frame import Frame

SPI_BENEFIT_COHERENCE_STAGE_NAME = "spi_benefit_coherence"
SPI_BENEFIT_COHERENCE_ROWS = "spi_synthetic_support_channel"
#: Reports whose award needs an existing legacy claim, an out-of-work test, or
#: a claim the copied household's new incomes cannot carry. Contributory JSA
#: and ESA follow the twin under the permitted-work limit instead (below).
SPI_ZEROED_REPORT_COLUMNS = (
    "jsa_income_reported",
    "esa_income_reported",
    "income_support_reported",
    "working_tax_credit_reported",
    "child_tax_credit_reported",
    "sda_reported",
    "ssmg_reported",
)
SPI_ZEROED_REPORT_REASON = (
    "award needs an existing legacy claim, an out-of-work test, or a claim the "
    "copied household's new incomes cannot carry"
)
#: Reports that follow an injury, service or a death, not income: the SPI row
#: keeps the FRS twin's value.
SPI_RESTORED_REPORT_COLUMNS = (
    "iidb_reported",
    "afcs_reported",
    "bsp_reported",
    "war_widows_pension_reported",
    "widowed_parents_allowance_reported",
)
SPI_RESTORED_REPORT_REASON = "follows an injury, service or a death, not income"
#: Contributory awards the SPI row takes from the FRS twin where the copy's own
#: work stays within each benefit's limits, and none outside them (María's
#: ruling of 29 October 2026, microcosm#1095 B10; JSA's own limits from
#: Vahid's review of microcosm#1426).
SPI_JSA_CONTRIB_COLUMN = "jsa_contrib_reported"
SPI_ESA_CONTRIB_COLUMN = "esa_contrib_reported"
SPI_WITHIN_LIMIT_REPORT_COLUMNS = (SPI_JSA_CONTRIB_COLUMN, SPI_ESA_CONTRIB_COLUMN)
SPI_WITHIN_LIMIT_EARNINGS_COLUMNS = ("employment_income", "self_employment_income")
#: The twin's usual weekly hours, annualised at 52 weeks by the FRS spine
#: (uk-data#531), which the copy keeps and the SPI pay floor reads.
SPI_WITHIN_LIMIT_HOURS_COLUMN = "hours_worked"
SPI_WITHIN_LIMIT_WEEKS_IN_YEAR = 52
SPI_WITHIN_LIMIT_EARNINGS = (
    "(employment_income + max(self_employment_income, 0)) / 52, the SPI row's own"
)
SPI_WITHIN_LIMIT_HOURS = (
    "hours_worked / 52, the twin's usual hours, which the SPI row keeps"
)
SPI_PERMITTED_WORK_LIMIT_VARIABLE = "esa_exempt_work_higher_earnings_limit"
#: JSA's age-related amount is its higher one from this age.
SPI_JSA_HIGHER_AMOUNT_FROM_AGE = 25
SPI_PERMITTED_WORK_LIMIT = (
    f"{SPI_ESA_CONTRIB_COLUMN}: up to policyengine-uk "
    f"{SPI_PERMITTED_WORK_LIMIT_VARIABLE} (ESA Regs 2008 reg 45(4)); "
    f"{SPI_JSA_CONTRIB_COLUMN}: below the age-related amount "
    "(gov.dwp.JSA.contrib.amount_18_24 under 25, amount_over_25 from 25) plus "
    "gov.dwp.JSA.contrib.earn_disregard, the prescribed amount being that sum "
    "less 1p (JSA Regs 1996 reg 56); weekly, at the frame's year"
)
SPI_WITHIN_LIMIT_HOURS_LIMIT = (
    f"{SPI_ESA_CONTRIB_COLUMN}: below gov.dwp.ESA.exempt_work.hours_limit (ESA "
    f"Regs 2008 reg 45(4)); {SPI_JSA_CONTRIB_COLUMN}: below "
    "gov.dwp.JSA.remunerative_work.claimant_hours (JSA Regs 1996 reg 51(1)(a)); "
    "weekly, at the frame's year"
)
SPI_WITHIN_LIMIT_AWARD = (
    f"{SPI_ESA_CONTRIB_COLUMN}: the twin's; {SPI_JSA_CONTRIB_COLUMN}: the lower "
    "of the twin's and 52 times the age-related amount less the earnings above "
    "the disregard (Jobseekers Act 1995 s. 4(1)(b), JSA Regs 1996 reg 80); nil "
    "outside the limits"
)
SPI_WITHIN_LIMIT_REASON = (
    "a contributory award follows the person's National Insurance record and, "
    "for ESA, incapacity, which the copy keeps, but not work outside each "
    "benefit's own earnings and hours limits"
)
#: The copy's pension payments, the tape's PENSION, which neither award is
#: reduced for; the receipt reads them against each award's threshold.
SPI_PENSION_PAYMENTS_COLUMN = "private_pension_income"
#: Half of every pound of pension payments above GBP 85 a week comes off
#: contributory ESA (ESA Regs 2008 reg 74(1)), an amount policyengine-uk has
#: no parameter for. JSA's threshold is policyengine-uk's
#: ``gov.dwp.JSA.contrib.pension_disregard`` (JSA Regs 1996 reg 81(1)).
SPI_ESA_PENSION_PAYMENTS_THRESHOLD = 85.0
SPI_TWIN_LINKAGE = "person_source_id -> FRS-channel person_id"
SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT = "receives_benefits_in_own_right"
SPI_BENEFITS_IN_OWN_RIGHT_RULE = (
    "universal_credit_reported + jsa_contrib_reported + jsa_income_reported + "
    "esa_contrib_reported + esa_income_reported > 0 (frs_education), SPI-channel "
    "rows; FRS rows asserted unchanged"
)
SPI_DISABILITY_REFRESH_SCOPE = (
    "disability categories and flags re-derived on SPI-channel rows from the "
    "final reports with the frs_disability rule at its declared year; the "
    "categories, which read only AA, DLA and PIP, asserted unchanged"
)
SPI_UC_REPORTED_ANCHOR = "universal_credit_reported_anchor"
SPI_UC_TAKE_UP_RATE_KEY = "universal_credit"
SPI_UC_TAKE_UP_SEED = 0
#: The disability flags the refresh writes; the categories read only AA, DLA
#: and PIP, which the stage keeps, so they are asserted unchanged instead.
SPI_DISABILITY_FLAG_COLUMNS = tuple(
    column
    for column in frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS
    if not column.endswith("_category")
)
SPI_BENEFIT_COHERENCE_REWRITES = (
    *SPI_ZEROED_REPORT_COLUMNS,
    *SPI_RESTORED_REPORT_COLUMNS,
    *SPI_WITHIN_LIMIT_REPORT_COLUMNS,
    *SPI_DISABILITY_FLAG_COLUMNS,
    SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT,
    UK_UC_TAKE_UP_OUTPUT,
)


def spi_benefit_coherence_operations() -> tuple[tuple[str, dict[str, object]], ...]:
    """The stage's operations, in order, as ``(kind, parameters)`` pairs."""

    return (
        (
            "zero_spi_channel_reports",
            {
                "rows": SPI_BENEFIT_COHERENCE_ROWS,
                "columns": list(SPI_ZEROED_REPORT_COLUMNS),
                "reason": SPI_ZEROED_REPORT_REASON,
            },
        ),
        (
            "restore_spi_channel_reports_from_source",
            {
                "rows": SPI_BENEFIT_COHERENCE_ROWS,
                "columns": list(SPI_RESTORED_REPORT_COLUMNS),
                "linkage": SPI_TWIN_LINKAGE,
                "reason": SPI_RESTORED_REPORT_REASON,
            },
        ),
        (
            "restore_spi_channel_reports_from_source_within_earnings_limit",
            {
                "rows": SPI_BENEFIT_COHERENCE_ROWS,
                "columns": list(SPI_WITHIN_LIMIT_REPORT_COLUMNS),
                "linkage": SPI_TWIN_LINKAGE,
                "earnings": SPI_WITHIN_LIMIT_EARNINGS,
                "weekly_limit": SPI_PERMITTED_WORK_LIMIT,
                "hours": SPI_WITHIN_LIMIT_HOURS,
                "weekly_hours_limit": SPI_WITHIN_LIMIT_HOURS_LIMIT,
                "award": SPI_WITHIN_LIMIT_AWARD,
                "reason": SPI_WITHIN_LIMIT_REASON,
            },
        ),
        ("derive", {"scope": SPI_DISABILITY_REFRESH_SCOPE}),
        (
            "derive",
            {
                "output": SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT,
                "formula": SPI_BENEFITS_IN_OWN_RIGHT_RULE,
            },
        ),
        (
            "aggregate_person_to_benunit",
            {
                "method": "any_positive",
                "consumed_only": True,
                "aggregates": {SPI_UC_REPORTED_ANCHOR: "universal_credit_reported"},
            },
        ),
        (
            "aggregate_person_to_benunit",
            {
                "method": UK_UC_AGE_ELIGIBLE_METHOD,
                "consumed_only": True,
                "aggregates": {UK_UC_AGE_ELIGIBLE_AGGREGATE: UK_UC_AGE_ELIGIBLE_SOURCE},
                "role_column": UK_UC_AGE_ELIGIBLE_ROLE,
            },
        ),
        (
            "assign_binary_with_anchored_residual",
            {
                "output": UK_UC_TAKE_UP_OUTPUT,
                "draw": UK_UC_TAKE_UP_OUTPUT,
                "rate_key": SPI_UC_TAKE_UP_RATE_KEY,
                "anchor": SPI_UC_REPORTED_ANCHOR,
                "population": UK_UC_AGE_ELIGIBLE_AGGREGATE,
                "seed": SPI_UC_TAKE_UP_SEED,
                "rows": SPI_BENEFIT_COHERENCE_ROWS,
            },
        ),
    )


@dataclass(frozen=True)
class UKSPIBenefitCoherenceResult:
    """Output frame and executed-effect receipt for the SPI benefit pass."""

    frame: Frame
    spi_person_rows: int
    spi_benefit_units: int
    zeroed: Mapping[str, Mapping[str, float]]
    restored: Mapping[str, Mapping[str, float]]
    restored_within_earnings_limit: Mapping[str, Mapping[str, object]]
    permitted_work_limit: Mapping[str, object]
    disability_flags_changed: Mapping[str, int]
    benefits_in_own_right: Mapping[str, float]
    universal_credit_take_up: Mapping[str, float]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": SPI_BENEFIT_COHERENCE_STAGE_NAME,
            "rows": SPI_BENEFIT_COHERENCE_ROWS,
            "spi_person_rows": self.spi_person_rows,
            "spi_benefit_units": self.spi_benefit_units,
            "zeroed": {key: dict(value) for key, value in self.zeroed.items()},
            "restored": {key: dict(value) for key, value in self.restored.items()},
            "restored_within_earnings_limit": {
                key: dict(value)
                for key, value in self.restored_within_earnings_limit.items()
            },
            "permitted_work_limit": dict(self.permitted_work_limit),
            "disability_flags_changed": dict(self.disability_flags_changed),
            "benefits_in_own_right": dict(self.benefits_in_own_right),
            "universal_credit_take_up": dict(self.universal_credit_take_up),
            "base_rows_unchanged": True,
        }


@dataclass(frozen=True)
class UKSPIPermittedWorkLimit:
    """The contributory awards' weekly work limits as policyengine-uk reads a year.

    ESA's permitted work: earnings up to ``esa_weekly_limit`` in under
    ``esa_hours_limit`` hours. JSA: under ``jsa_remunerative_hours`` hours and
    earnings below the age-related amount plus ``jsa_earnings_disregard``, and
    an award of at most that amount less the earnings above the disregard.
    ``jsa_pension_disregard`` is the weekly pension payments above which JSA
    is reduced, which only the receipt reads.
    """

    esa_weekly_limit: float
    esa_hours_limit: float
    jsa_amount_under_25: float
    jsa_amount_from_25: float
    jsa_earnings_disregard: float
    jsa_remunerative_hours: float
    jsa_pension_disregard: float
    year: int
    source: str

    def jsa_amount(self, age: np.ndarray) -> np.ndarray:
        """JSA's weekly age-related amount at each age."""

        return np.where(
            np.asarray(age, dtype=float) < SPI_JSA_HIGHER_AMOUNT_FROM_AGE,
            self.jsa_amount_under_25,
            self.jsa_amount_from_25,
        )

    def pension_payments_thresholds(self) -> dict[str, float]:
        """Each award's weekly pension payments threshold (reg 81, reg 74)."""

        return {
            SPI_JSA_CONTRIB_COLUMN: self.jsa_pension_disregard,
            SPI_ESA_CONTRIB_COLUMN: SPI_ESA_PENSION_PAYMENTS_THRESHOLD,
        }

    def evidence(self) -> dict[str, object]:
        return {
            "esa_weekly_limit": self.esa_weekly_limit,
            "esa_hours_limit": self.esa_hours_limit,
            "jsa_amount_under_25": self.jsa_amount_under_25,
            "jsa_amount_from_25": self.jsa_amount_from_25,
            "jsa_earnings_disregard": self.jsa_earnings_disregard,
            "jsa_remunerative_hours": self.jsa_remunerative_hours,
            "jsa_pension_disregard": self.jsa_pension_disregard,
            "esa_pension_payments_threshold": SPI_ESA_PENSION_PAYMENTS_THRESHOLD,
            "year": self.year,
            "source": self.source,
        }


def uk_spi_permitted_work_limit(year: int | str) -> UKSPIPermittedWorkLimit:
    """Read the limits from the engine's own variable and parameters for ``year``.

    ESA's earnings limit is a variable that reads parameters alone (16 hours at
    the national living wage, rounded up to the next 50p), so one person in an
    empty household gives it; the rest are the parameters at the year, as the
    engine's formulas read them.
    """

    try:
        from policyengine_uk import Simulation
    except ImportError as exc:
        raise ImportError(
            "The SPI permitted-work limit requires `uv sync --all-packages --extra uk`."
        ) from exc

    period = int(year)
    simulation = Simulation(
        situation={
            "people": {"person": {}},
            "benunits": {"benunit": {"members": ["person"]}},
            "households": {"household": {"members": ["person"]}},
        }
    )
    values = np.asarray(
        simulation.calculate(SPI_PERMITTED_WORK_LIMIT_VARIABLE, period), dtype=float
    )
    if values.shape != (1,) or not (np.isfinite(values[0]) and values[0] > 0.0):
        raise ValueError(
            f"policyengine-uk {SPI_PERMITTED_WORK_LIMIT_VARIABLE} at {period} is "
            f"not one positive amount: {values.tolist()}."
        )
    dwp = simulation.tax_benefit_system.parameters(period).gov.dwp
    amounts = {
        "esa_hours_limit": float(dwp.ESA.exempt_work.hours_limit),
        "jsa_amount_under_25": float(dwp.JSA.contrib.amount_18_24),
        "jsa_amount_from_25": float(dwp.JSA.contrib.amount_over_25),
        "jsa_earnings_disregard": float(dwp.JSA.contrib.earn_disregard),
        "jsa_remunerative_hours": float(dwp.JSA.remunerative_work.claimant_hours),
        "jsa_pension_disregard": float(dwp.JSA.contrib.pension_disregard),
    }
    invalid = {
        name: value
        for name, value in amounts.items()
        if not (np.isfinite(value) and value > 0.0)
    }
    if invalid:
        raise ValueError(
            f"policyengine-uk's contributory work limits at {period} are not all "
            f"positive amounts: {invalid}."
        )
    return UKSPIPermittedWorkLimit(
        esa_weekly_limit=float(values[0]),
        **amounts,
        year=period,
        source=(
            f"policyengine-uk {metadata.version('policyengine-uk')} "
            f"{SPI_PERMITTED_WORK_LIMIT_VARIABLE}, gov.dwp.ESA.exempt_work."
            "hours_limit, gov.dwp.JSA.contrib and gov.dwp.JSA.remunerative_work."
            f"claimant_hours at {period}"
        ),
    )


@dataclass(frozen=True)
class UKSPIBenefitCoherenceStageTransform:
    """Whole-stage callable for the SPI-channel benefit coherence pass."""

    stage: SourceStageSpec
    contract: UKTakeUpContract | None = None
    population_policy: UKTakeUpPopulationPolicy | None = None
    category_rates: object | None = None
    flag_rates: object | None = None
    permitted_work_limit: UKSPIPermittedWorkLimit | None = None
    last_result: UKSPIBenefitCoherenceResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        assert_spi_benefit_coherence_stage_parameters(self.stage)
        result = apply_spi_benefit_coherence(
            frame,
            contract=self.contract or load_uk_take_up_contract(),
            population_policy=(
                self.population_policy
                or uk_take_up_population_policy(uk_time_period(frame))
            ),
            category_rates=self.category_rates,
            flag_rates=self.flag_rates,
            permitted_work_limit=self.permitted_work_limit,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def apply_spi_benefit_coherence(
    frame: Frame,
    *,
    contract: UKTakeUpContract,
    population_policy: UKTakeUpPopulationPolicy,
    category_rates: object | None = None,
    flag_rates: object | None = None,
    permitted_work_limit: UKSPIPermittedWorkLimit | None = None,
) -> UKSPIBenefitCoherenceResult:
    """Zero, restore and re-derive SPI-channel reports; redraw their UC take-up."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    person_channel = support_channel_column("person")
    benunit_channel = support_channel_column("benunit")
    required_person = {
        "person_id",
        "person_benunit_id",
        "person_household_id",
        "age",
        person_channel,
        support_source_id_column("person"),
        *SPI_ZEROED_REPORT_COLUMNS,
        *SPI_RESTORED_REPORT_COLUMNS,
        *SPI_WITHIN_LIMIT_REPORT_COLUMNS,
        *SPI_WITHIN_LIMIT_EARNINGS_COLUMNS,
        SPI_WITHIN_LIMIT_HOURS_COLUMN,
        SPI_PENSION_PAYMENTS_COLUMN,
        *BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS,
        *frs_disability.UK_DISABILITY_FLAG_REPORTED_COLUMNS,
        *frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS,
        SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT,
    }
    missing = sorted(required_person - set(person.columns))
    if missing:
        raise ValueError(f"SPI benefit coherence person columns missing: {missing}.")
    missing = sorted(
        {"benunit_id", benunit_channel, UK_UC_TAKE_UP_OUTPUT} - set(benunit.columns)
    )
    if missing:
        raise ValueError(f"SPI benefit coherence benunit columns missing: {missing}.")
    channels = set(person[person_channel].unique())
    unexpected = sorted(
        channels - {BASE_FRS_SUPPORT_CHANNEL, SPI_SYNTHETIC_SUPPORT_CHANNEL}
    )
    if unexpected:
        raise ValueError(
            "SPI benefit coherence runs before any other support channel exists; "
            f"found {unexpected}."
        )
    spi = (person[person_channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy()
    base = ~spi
    if not spi.any():
        raise ValueError("SPI benefit coherence found no SPI-channel person rows.")
    rewritten = list(
        dict.fromkeys(
            (
                *SPI_ZEROED_REPORT_COLUMNS,
                *SPI_RESTORED_REPORT_COLUMNS,
                *SPI_WITHIN_LIMIT_REPORT_COLUMNS,
                *frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS,
                SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT,
            )
        )
    )
    # The categories are in the base-row check too: the refresh must leave
    # them untouched everywhere.
    before = person[rewritten].copy()
    weights = _person_weights(person, household, frame.weights_for("household").values)

    zeroed: dict[str, dict[str, float]] = {}
    for column in SPI_ZEROED_REPORT_COLUMNS:
        values = pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
        reporting = spi & (values > 0.0)
        zeroed[column] = {
            "rows_reporting_before": int(reporting.sum()),
            "weighted_reporters_before": float(weights[reporting].sum()),
            "weighted_amount_before": float((values * weights)[spi].sum()),
        }
        person.loc[spi, column] = 0.0
        after = pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
        zeroed[column]["rows_reporting_after"] = int((spi & (after > 0.0)).sum())

    twin = person.loc[base, ["person_id", *SPI_RESTORED_REPORT_COLUMNS]]
    restored_frame = fill_support_channel_from_source(
        person,
        twin,
        entity="person",
        columns=SPI_RESTORED_REPORT_COLUMNS,
    )
    restored: dict[str, dict[str, float]] = {}
    for column in SPI_RESTORED_REPORT_COLUMNS:
        old = pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
        new = (
            pd.to_numeric(restored_frame[column], errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        restored[column] = {
            "rows_changed": int((spi & ~np.isclose(old, new)).sum()),
            "weighted_amount_before": float((old * weights)[spi].sum()),
            "weighted_amount_after": float((new * weights)[spi].sum()),
            "weighted_reporters_before": float(weights[spi & (old > 0.0)].sum()),
            "weighted_reporters_after": float(weights[spi & (new > 0.0)].sum()),
        }
        person[column] = restored_frame[column].to_numpy()
    restored_check = fill_support_channel_from_source(
        person,
        person.loc[base, ["person_id", *SPI_RESTORED_REPORT_COLUMNS]],
        entity="person",
        columns=SPI_RESTORED_REPORT_COLUMNS,
    )
    for column in SPI_RESTORED_REPORT_COLUMNS:
        restored[column]["rows_differing_from_twin_after"] = int(
            (
                spi
                & ~np.isclose(
                    pd.to_numeric(person[column], errors="coerce").fillna(0.0),
                    pd.to_numeric(restored_check[column], errors="coerce").fillna(0.0),
                )
            ).sum()
        )

    limit = permitted_work_limit or uk_spi_permitted_work_limit(uk_time_period(frame))
    within_limits, award_caps = _contributory_work_limits(person, limit)
    weekly_pension = (
        _numeric(person[SPI_PENSION_PAYMENTS_COLUMN]) / SPI_WITHIN_LIMIT_WEEKS_IN_YEAR
    )
    pension_thresholds = limit.pension_payments_thresholds()
    sources = _person_source_households(person, household)
    twin_awards = fill_support_channel_from_source(
        person,
        person.loc[base, ["person_id", *SPI_WITHIN_LIMIT_REPORT_COLUMNS]],
        entity="person",
        columns=SPI_WITHIN_LIMIT_REPORT_COLUMNS,
    )
    restored_within_limit: dict[str, dict[str, object]] = {}
    for column in SPI_WITHIN_LIMIT_REPORT_COLUMNS:
        old = _numeric(person[column])
        twin_award = _numeric(twin_awards[column])
        within = within_limits[column]
        award = np.where(within, np.minimum(twin_award, award_caps[column]), 0.0)
        new = np.where(spi, award, old)
        twin_reports = spi & (twin_award > 0.0)
        restored_within_limit[column] = _within_limit_receipt(
            old=old,
            new=new,
            spi=spi,
            twin_reports=twin_reports,
            outside=twin_reports & ~within,
            reduced=twin_reports & within & (award < twin_award),
            pension_above=weekly_pension > pension_thresholds[column],
            weights=weights,
            sources=sources,
        )
        person[column] = new
    twin_check = fill_support_channel_from_source(
        person,
        person.loc[base, ["person_id", *SPI_WITHIN_LIMIT_REPORT_COLUMNS]],
        entity="person",
        columns=SPI_WITHIN_LIMIT_REPORT_COLUMNS,
    )
    for column in SPI_WITHIN_LIMIT_REPORT_COLUMNS:
        expected = np.where(
            within_limits[column],
            np.minimum(_numeric(twin_check[column]), award_caps[column]),
            0.0,
        )
        restored_within_limit[column]["rows_breaking_rule_after"] = int(
            (spi & ~np.isclose(_numeric(person[column]), expected)).sum()
        )
    permitted_work = {
        **limit.evidence(),
        "earnings": SPI_WITHIN_LIMIT_EARNINGS,
        "hours": SPI_WITHIN_LIMIT_HOURS,
        "spi_source_households_within_limits": {
            column: support_band(
                distinct_count(sources, spi & within_limits[column], weights)
            )
            for column in SPI_WITHIN_LIMIT_REPORT_COLUMNS
        },
    }

    flags_changed = _refresh_disability(
        person, spi=spi, category_rates=category_rates, flag_rates=flag_rates
    )

    own_right_rule = (
        person[list(BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
        .gt(0.0)
        .to_numpy()
    )
    stored_own_right = person[SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT].astype(bool).to_numpy()
    drifted = base & (own_right_rule != stored_own_right)
    if drifted.any():
        raise ValueError(
            f"{int(drifted.sum())} FRS-channel person(s) carry "
            f"{SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT} that disagrees with their own "
            "reports; the frs_education rule and an upstream stage have drifted."
        )
    own_right = {
        "spi_rows_changed": int((spi & (own_right_rule != stored_own_right)).sum()),
        "spi_true_before": int((spi & stored_own_right).sum()),
        "spi_true_after": int((spi & own_right_rule).sum()),
        "weighted_spi_changed": float(
            weights[spi & (own_right_rule != stored_own_right)].sum()
        ),
    }
    person.loc[spi, SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT] = own_right_rule[spi]
    person[SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT] = person[
        SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT
    ].astype(bool)
    own_right["mismatches_after"] = int(
        (
            person[SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT].to_numpy(dtype=bool)
            != own_right_rule
        ).sum()
    )

    for column in rewritten:
        unchanged = before.loc[base, column].equals(person.loc[base, column])
        if not unchanged:
            raise ValueError(
                f"SPI benefit coherence changed FRS-channel {column!r}; base rows "
                "are never modified."
            )

    benunit, take_up = _redraw_spi_uc_take_up(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=frame.weights_for("household").values,
        contract=contract,
        population_policy=population_policy,
    )
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return UKSPIBenefitCoherenceResult(
        frame=result,
        spi_person_rows=int(spi.sum()),
        spi_benefit_units=int(take_up["spi_benefit_units"]),
        zeroed=zeroed,
        restored=restored,
        restored_within_earnings_limit=restored_within_limit,
        permitted_work_limit=permitted_work,
        disability_flags_changed=flags_changed,
        benefits_in_own_right=own_right,
        universal_credit_take_up=take_up,
    )


def _redraw_spi_uc_take_up(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    contract: UKTakeUpContract,
    population_policy: UKTakeUpPopulationPolicy,
) -> tuple[pd.DataFrame, dict[str, float]]:
    benunit = benunit.copy()
    spi_units = (
        benunit[support_channel_column("benunit")] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    ).to_numpy()
    reported = (
        pd.to_numeric(person["universal_credit_reported"], errors="coerce").fillna(0.0)
        > 0.0
    )
    reporter = (
        benunit["benunit_id"]
        .isin(set(person.loc[reported, "person_benunit_id"]))
        .to_numpy(dtype=bool)
    )
    population = uc_age_eligible_benunits(person, benunit, population_policy)
    rate = contract.rate(SPI_UC_TAKE_UP_RATE_KEY)
    draws = stable_identity_uniforms(
        benunit.loc[spi_units, "benunit_id"].to_numpy(),
        seed=SPI_UC_TAKE_UP_SEED,
        salt=UK_UC_TAKE_UP_OUTPUT,
    )
    redrawn = assign_binary_with_anchored_residual(
        draws,
        rate,
        anchor=reporter[spi_units],
        population=population[spi_units],
    )
    previous = benunit[UK_UC_TAKE_UP_OUTPUT].fillna(False).astype(bool).to_numpy()
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=household_weights,
    )
    after = previous.copy()
    after[spi_units] = redrawn
    benunit[UK_UC_TAKE_UP_OUTPUT] = after
    spi_population = spi_units & population
    receipt = {
        "rate_key": SPI_UC_TAKE_UP_RATE_KEY,
        "rate": float(rate),
        "seed": SPI_UC_TAKE_UP_SEED,
        "population_state_pension_age": int(population_policy.state_pension_age),
        "spi_benefit_units": int(spi_units.sum()),
        "spi_population_units": int(spi_population.sum()),
        "spi_reporter_units": int((spi_units & reporter).sum()),
        "spi_claiming_before": int((spi_units & previous).sum()),
        "spi_claiming_after": int((spi_units & after).sum()),
        "set_true": int((spi_units & ~previous & after).sum()),
        "set_false": int((spi_units & previous & ~after).sum()),
        "weighted_spi_claiming_before": float(weights[spi_units & previous].sum()),
        "weighted_spi_claiming_after": float(weights[spi_units & after].sum()),
        "weighted_spi_population": float(weights[spi_population].sum()),
        "reporters_not_claiming": int((spi_units & reporter & ~after).sum()),
        "outside_population_non_reporters_claiming": int(
            (spi_units & ~population & ~reporter & after).sum()
        ),
    }
    return benunit, receipt


def _refresh_disability(
    person: pd.DataFrame,
    *,
    spi: np.ndarray,
    category_rates: object | None,
    flag_rates: object | None,
) -> dict[str, int]:
    if category_rates is None or flag_rates is None:
        from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule

        year = resolve_uk_year_rule(frs_disability.YEAR_RULE)
        if category_rates is None:
            category_rates = frs_disability.uk_dwp_disability_category_rates(year)
        if flag_rates is None:
            flag_rates = frs_disability.uk_dwp_disability_flag_rates(year)
    rows = person.loc[spi]
    derived = frs_disability.derive_frs_disability(
        rows, category_rates=category_rates, flag_rates=flag_rates
    )
    changed: dict[str, int] = {}
    for column in frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS:
        new = derived[column].to_numpy()
        old = rows[column].to_numpy()
        moved = int((new != old).sum())
        if column.endswith("_category"):
            if moved:
                raise ValueError(
                    f"SPI benefit coherence moved {moved} {column!r} value(s); the "
                    "categories read only AA, DLA and PIP, which this stage keeps."
                )
            continue
        changed[column] = moved
        person.loc[spi, column] = new.astype(bool)
    return changed


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _weekly_earnings(person: pd.DataFrame) -> np.ndarray:
    """Employment income plus any self-employment profit, over 52 weeks.

    Annual pay averaged over 52 weeks, which understates part-year claimants
    (see the module docstring).
    """

    employment, profit = (
        _numeric(person[column]) for column in SPI_WITHIN_LIMIT_EARNINGS_COLUMNS
    )
    return (employment + np.maximum(profit, 0.0)) / SPI_WITHIN_LIMIT_WEEKS_IN_YEAR


def _contributory_work_limits(
    person: pd.DataFrame, limit: UKSPIPermittedWorkLimit
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Each contributory award's within-limits mask and its annual award cap.

    JSA's earnings test is the strict "below the age-related amount plus the
    disregard", which on amounts in pence is reg 56's "no more than that sum
    less 1p". ESA's award is the twin's, uncapped, within permitted work.
    """

    earnings = _weekly_earnings(person)
    hours = _numeric(person[SPI_WITHIN_LIMIT_HOURS_COLUMN])
    hours = hours / SPI_WITHIN_LIMIT_WEEKS_IN_YEAR
    jsa_amount = limit.jsa_amount(_numeric(person["age"]))
    disregard = limit.jsa_earnings_disregard
    within = {
        SPI_JSA_CONTRIB_COLUMN: (hours < limit.jsa_remunerative_hours)
        & (earnings < jsa_amount + disregard),
        SPI_ESA_CONTRIB_COLUMN: (hours < limit.esa_hours_limit)
        & (earnings <= limit.esa_weekly_limit),
    }
    caps = {
        SPI_JSA_CONTRIB_COLUMN: SPI_WITHIN_LIMIT_WEEKS_IN_YEAR
        * np.maximum(jsa_amount - np.maximum(earnings - disregard, 0.0), 0.0),
        SPI_ESA_CONTRIB_COLUMN: np.full(len(person), np.inf),
    }
    return within, caps


def _within_limit_receipt(
    *,
    old: np.ndarray,
    new: np.ndarray,
    spi: np.ndarray,
    twin_reports: np.ndarray,
    outside: np.ndarray,
    reduced: np.ndarray,
    pension_above: np.ndarray,
    weights: np.ndarray,
    sources: np.ndarray,
) -> dict[str, object]:
    """One award's receipt on distinct source households (microcosm#1164).

    Support is banded. A weighted figure is "<3" when one or two source
    households carry it, or carry the change between its before and after,
    which the two figures would otherwise give back by difference. The award
    on the copies whose pension payments are above its threshold is "<3" too
    when one or two carry the rest of the award, which it would give back
    beside the whole.
    """

    changed = distinct_count(sources, spi & ~np.isclose(old, new), weights)

    def weighted(values: np.ndarray, mask: np.ndarray) -> float | str:
        support = distinct_count(sources, mask, weights)
        if 0 < support < UK_DISCLOSURE_FLOOR or 0 < changed < UK_DISCLOSURE_FLOOR:
            return UK_WITHHELD
        return float((values * weights)[mask].sum())

    before = spi & (old > 0.0)
    after = spi & (new > 0.0)
    above = after & np.asarray(pension_above, dtype=bool)
    rest = distinct_count(sources, after & ~above, weights)

    def weighted_above(values: np.ndarray) -> float | str:
        if 0 < rest < UK_DISCLOSURE_FLOOR:
            return UK_WITHHELD
        return weighted(values, above)

    return {
        "twin_reporters": support_band(distinct_count(sources, twin_reports, weights)),
        "twin_reporters_outside_limits": support_band(
            distinct_count(sources, outside, weights)
        ),
        "twin_reporters_reduced": support_band(
            distinct_count(sources, reduced, weights)
        ),
        "changed": support_band(changed),
        "weighted_amount_before": weighted(old, before),
        "weighted_amount_after": weighted(new, after),
        "weighted_reporters_before": weighted(np.ones_like(old), before),
        "weighted_reporters_after": weighted(np.ones_like(new), after),
        "pension_payments_above_threshold": {
            "source_households": support_band(distinct_count(sources, above, weights)),
            "weighted_amount_after": weighted_above(new),
            "weighted_reporters_after": weighted_above(np.ones_like(new)),
        },
    }


def _person_source_households(
    person: pd.DataFrame, household: pd.DataFrame
) -> np.ndarray:
    """Each person's true source household, the lineage the receipt counts by."""

    return record_source_households(
        {"person": person, "household": household},
        "person",
        stage=SPI_BENEFIT_COHERENCE_STAGE_NAME,
    )


def _person_weights(
    person: pd.DataFrame, household: pd.DataFrame, household_weights: np.ndarray
) -> np.ndarray:
    by_household = pd.Series(
        np.asarray(household_weights, dtype=float),
        index=household["household_id"].to_numpy(),
    )
    weights = person["person_household_id"].map(by_household)
    if weights.isna().any():
        raise ValueError("SPI benefit coherence persons must map to a household.")
    return weights.to_numpy(dtype=float)


def assert_spi_benefit_coherence_stage_parameters(stage: SourceStageSpec) -> None:
    """Refuse a manifest whose operations differ from the code's."""

    if stage.stage != SPI_BENEFIT_COHERENCE_STAGE_NAME:
        raise ValueError(
            f"SPI benefit coherence received stage {stage.stage!r}, expected "
            f"{SPI_BENEFIT_COHERENCE_STAGE_NAME!r}."
        )
    expected = [
        (kind, dict(parameters))
        for kind, parameters in spi_benefit_coherence_operations()
    ]
    actual = [
        (operation.kind, dict(operation.parameters)) for operation in stage.operations
    ]
    if actual != expected:
        raise ValueError(
            f"{SPI_BENEFIT_COHERENCE_STAGE_NAME} operations drifted: expected "
            f"{expected}, got {actual}."
        )
    assert_take_up_stage_population_declaration(stage)
    if stage.outputs != () or tuple(stage.rewrites) != SPI_BENEFIT_COHERENCE_REWRITES:
        raise ValueError(
            f"{SPI_BENEFIT_COHERENCE_STAGE_NAME} must declare no new outputs and "
            f"exactly the rewrites {SPI_BENEFIT_COHERENCE_REWRITES}."
        )


__all__ = [
    "SPI_BENEFIT_COHERENCE_REWRITES",
    "SPI_DISABILITY_FLAG_COLUMNS",
    "SPI_BENEFIT_COHERENCE_STAGE_NAME",
    "SPI_RESTORED_REPORT_COLUMNS",
    "SPI_WITHIN_LIMIT_REPORT_COLUMNS",
    "SPI_ZEROED_REPORT_COLUMNS",
    "UKSPIBenefitCoherenceResult",
    "UKSPIBenefitCoherenceStageTransform",
    "UKSPIPermittedWorkLimit",
    "apply_spi_benefit_coherence",
    "assert_spi_benefit_coherence_stage_parameters",
    "spi_benefit_coherence_operations",
    "uk_spi_permitted_work_limit",
]
