"""Annual income descriptors on the UK FRS spine.

HMRC's income statistics (the Survey of Personal Incomes and the Income Tax
liabilities statistics) count the income each taxpayer received over a tax
year, banded by what they received. They cover individuals liable to UK
income tax, so a person who left a job counts only where the year's income
makes them liable. The spine's income columns are current, annualised
incomes, the concept HBAI and policyengine-uk use: the usual weekly pay of
the job held at interview times 365.25 / 7. A person who started a job during
the year, left one, changed jobs or had a gap sits in another HMRC band than
their current pay puts them in, or in none, and a recent leaver who paid tax
in the year is no taxpayer at all.

This stage writes what the FRS records about the 12 months before the
interview, for every adult, as descriptors beside the current columns, which
it never changes:

* ``employment_income_annual_share``: the share of the 12 months paid at the
  current pay rate (1 for a full-year employee, 0 with no current pay);
* ``employment_income_other_spells``: annual employee pay from spells outside
  the current job, chiefly the job a recent leaver left (FRS basis, GBP a year
  at the interview's prices);
* ``self_employment_income_annual_share`` and
  ``self_employment_income_other_spells``: the same for self-employment;
* ``private_pension_income_annual_share``: the share of the year a private
  pension was in payment for someone who retired during it;
* ``contributory_benefits_annual_share``: the share of the year contributory
  JSA and ESA and Incapacity Benefit were in payment for someone who left work
  during it.

An annual amount is the current amount times its share plus the other spells.
So the year is a proxy for HMRC's tax year, not the tax year itself: the 12
months before each interview (interviews from April 2024 to March 2025),
applied as shares and spells to the engine's amounts at the calibration year
(2025), and fitted to HMRC targets for calendar 2025 (the SPI 2023-24 rows
uprated to it, and the Income Tax liabilities rows' calendar-2025 window).
Only the calibration's HMRC measures read the descriptors; poverty, benefits
and the published engine's tax keep reading current income. The State Pension
needs no descriptor: policyengine-uk's ``months_since_state_pension_age``
gives the share of the year it was in payment.

What the survey records (UKDS SN 9563, the 2024-25 question instructions and
variable listing; every amount is GBP a week in the EUL):

* ``ADULT.CURRJOBM`` (DV): months in the current job; ``JOB.WRKPREV``: what the
  person did just before it, asked when the job is under 12 months old. Code 1
  ("in paid employment") includes a break of under three months.
* ``ADULT.SAMESIT`` (all adults) and ``SDEMP01``-``SDEMP12``, the self-defined
  activity one to twelve months before the interview, asked when SAMESIT is 1.
  The calendar under-records change: many people who started or left a job
  in the year answer SAMESIT 2, so it decides months only where it records a
  change, and never outranks CURRJOBM, WRKPREV or LSTYR.
* ``ADULT.NUMJOB2`` and the JOB rows of people out of work who worked in the
  last 12 months: their last job's type and its usual pay, ``JOB.UGROSS``
  ("people currently out of work who last worked in the previous 12 months
  ... in respect of that employment", question instructions p. 220), for
  employee last jobs. ``ADULT.LSTYR``: weeks of regular paid work in the last
  12 months; ``ADULT.LSTWRK1``/``LSTWRK2``: when they left;
  ``ADULT.RETIRE``/``RETIRE1``: retired in the last 12 months, and the month.
* ``JOB.SEJBLONG``/``SEJBMTHS``: time in the current self-employed job;
  ``JOB.SEINCAMT`` (else ``PROFIT1``) and ``JOB.SELWKS`` on a ceased
  self-employed last job. ``ADULT.SEINCAM2`` is already a 12-month concept
  (accounts profit or a 12-month average) the FRS uprates by average earnings
  to the interview, so continuing self-employment keeps its whole year and its
  amount is not uprated again. A ceased business's SEINCAMT (else PROFIT1) is
  the amount when it ceased, so it is uprated by average earnings for the
  months since, as a leaver's last pay is.

The rules, in order of precedence (a person in work at interview is never a
leaver):

* A current earner (``employment_income > 0``) is paid at today's rate for
  every month of the window that the current job, or an earlier spell, covers:
  12 months when CURRJOBM is 12 or more; for a job changer (WRKPREV 1) the
  months in the job plus, where SAMESIT is 1, the calendar's employee months
  before it started, and otherwise the whole year (the respondent reports no
  change of situation); for an entrant the months in the job plus the
  calendar's employee months before it, where SAMESIT is 1.
* An employee in work without recorded pay keeps none for the current job, as
  the current columns do, and is paid a donor's rate for the calendar's
  employee months before the job started. Anyone else in work (chiefly people
  who moved from employment to self-employment) is paid a donor's rate for the
  calendar's employee months.
* A leaver whose last job was an employee job works LSTYR x 12 / 52 months
  where LSTYR is positive (LSTYR counts *regular* work, so zero falls
  through), else the calendar's employee months, else ``LEAVING_DATE_SHARE`` of
  the months between the window's start and leaving (LSTWRK1/2, else RETIRE1);
  a leaving date 12 months or more before the interview gives none. Their pay
  is the last job's usual gross pay, scaled to the DV concept by the weighted
  job-level UGRSPAY / UGROSS ratio on current employees who report both, and
  uprated by average earnings for the months since they left; without it, a
  donor's rate.
* A person out of work outside the last-job block takes the calendar's
  employee months at a donor's rate.
* Donors are current employee earners in their first ``DONOR_RECENT_START_MONTHS``
  months in the job, closer to the pay of part-year workers than long-tenure
  employees, drawn by GROSS4 weight within age band x sex x full- or part-time
  cells, which drop sex below ``MINIMUM_DONORS``. The draw is an inverse-CDF
  read of one identity-keyed uniform per person.
* Self-employment: the whole year where SEJBLONG is one or more, (SEJBMTHS + 1)
  / 12 for a business under a year old, and at most the calendar's
  self-employed months where SAMESIT is 1 and self-employment is the activity
  today (the calendar records the main activity only, so a side business
  beside a job keeps its whole year). A ceased self-employed last job
  pays SEINCAMT (else PROFIT1, a loss as zero), uprated by average earnings
  for the months since it ceased, for SELWKS x 12 / 52 months, else LSTYR, the
  calendar or the leaving date.
* A private pension is in payment for (months since RETIRE1 + 0.5) / 12 of the
  year after retiring in it; contributory benefits for the months not worked
  by a leaver or a ceased self-employed person.

The receipt is build output and follows UKDS disclosure control
(:mod:`receipt_counts`): every cell gives its support as a band of distinct
source households, and a weighted figure only where ten or more carry it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_spine import (
    WEEKS_IN_YEAR,
    frs_interview_dates,
    normalize_ids,
    read_pinned_tab,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_carriers,
    extreme_count,
    small_cell_share,
    support_band,
)
from microcosm.frame import Frame

FRS_ANNUAL_INCOME_STAGE_NAME = "frs_annual_income_spells"
FRS_ANNUAL_INCOME_OPERATION_KIND = "derive_annual_income_spells"

EMPLOYMENT_INCOME_ANNUAL_SHARE = "employment_income_annual_share"
EMPLOYMENT_INCOME_OTHER_SPELLS = "employment_income_other_spells"
SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE = "self_employment_income_annual_share"
SELF_EMPLOYMENT_INCOME_OTHER_SPELLS = "self_employment_income_other_spells"
PRIVATE_PENSION_INCOME_ANNUAL_SHARE = "private_pension_income_annual_share"
CONTRIBUTORY_BENEFITS_ANNUAL_SHARE = "contributory_benefits_annual_share"
#: The descriptors in output order. A share is 1 for a whole year; an other
#: spells amount is GBP a year on the FRS basis.
FRS_ANNUAL_INCOME_OUTPUT_COLUMNS = (
    EMPLOYMENT_INCOME_ANNUAL_SHARE,
    EMPLOYMENT_INCOME_OTHER_SPELLS,
    SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE,
    SELF_EMPLOYMENT_INCOME_OTHER_SPELLS,
    PRIVATE_PENSION_INCOME_ANNUAL_SHARE,
    CONTRIBUTORY_BENEFITS_ANNUAL_SHARE,
)
ANNUAL_SHARE_COLUMNS = (
    EMPLOYMENT_INCOME_ANNUAL_SHARE,
    SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE,
    PRIVATE_PENSION_INCOME_ANNUAL_SHARE,
    CONTRIBUTORY_BENEFITS_ANNUAL_SHARE,
)
OTHER_SPELLS_COLUMNS = (
    EMPLOYMENT_INCOME_OTHER_SPELLS,
    SELF_EMPLOYMENT_INCOME_OTHER_SPELLS,
)
#: The current columns each descriptor qualifies; the stage never writes them.
CURRENT_INCOME_COLUMNS = ("employment_income", "self_employment_income")

WINDOW_MONTHS = 12
#: The survey's activity codes (SELFDEMP, SDEMP01-12; variable listing).
SELF_DEFINED_EMPLOYEE_CODES = (1, 2)
SELF_DEFINED_PART_TIME_EMPLOYEE_CODE = 2
SELF_DEFINED_SELF_EMPLOYED_CODES = (3, 4)
#: ADULT.SAMESIT 1: the respondent's situation changed in the last 12 months.
SITUATION_CHANGED_CODE = 1
#: ADULT.EMPSTATI full- and part-time employee; 2 is part-time.
ILO_EMPLOYEE_CODES = (1, 2)
ILO_PART_TIME_EMPLOYEE_CODE = 2
#: ADULT.WORKING and JOBAWAY 1: did paid work last week, or was away from a job.
WORKING_CODE = 1
#: JOB.WRKPREV 1: in paid employment just before the current job.
WRKPREV_PAID_EMPLOYMENT_CODE = 1
#: JOB.ETYPE 1 is an employee; 2 to 7 every self-employed description.
ETYPE_EMPLOYEE_CODE = 1
ETYPE_SELF_EMPLOYED_CODES = (2, 3, 4, 5, 6, 7)
#: JOB.FTPT 2: part time.
JOB_PART_TIME_CODE = 2
#: JOB.PROFIT2 2: PROFIT1 is a loss.
PROFIT_LOSS_CODE = 2
#: ADULT.RETIRE 1: retired in the last 12 months.
RETIRED_IN_WINDOW_CODE = 1

#: The share of the months between the window's start and the leaving date a
#: leaver with no weeks or calendar months is taken to have worked: leavers
#: with LSTYR work about three quarters of the months open to them.
LEAVING_DATE_SHARE = 0.75
#: Months open to a leaver whose leaving date is unknown, before the share.
UNKNOWN_LEAVING_MONTHS = 6.0
DONOR_RECENT_START_MONTHS = 24
MINIMUM_DONORS = 30
DONOR_AGE_EDGES = (25, 35, 50, 65)
DRAW_SEED = 0
DRAW_SALT = EMPLOYMENT_INCOME_OTHER_SPELLS
#: Leavers' usual pay is from when they left; it is uprated by the pinned OBR
#: average earnings index over the survey window (2024 over 2023), as many
#: months as they have been gone, at most two years.
EARNINGS_INDEX_RESOURCE = "hmrc_uprating_engine_pins.json"
EARNINGS_INDEX_PARAMETER = "gov.economic_assumptions.indices.obr.average_earnings"
EARNINGS_INDEX_FROM = "2023-01-01"
EARNINGS_INDEX_TO = "2024-01-01"
MAXIMUM_UPRATING_MONTHS = 24
#: The ratio check: the job-level DV-to-usual pay ratio must lie in this range.
PAY_CONCEPT_RATIO_BOUNDS = (0.95, 1.10)

_ADULT_COLUMNS = (
    "sernum",
    "benunit",
    "person",
    "samesit",
    "selfdemp",
    *(f"sdemp{month:02d}" for month in range(1, 13)),
    "working",
    "jobaway",
    "numjob2",
    "currjobm",
    "prevjobm",
    "lstwrk1",
    "lstwrk2",
    "lstyr",
    "retire",
    "retire1",
    "empstati",
)
_JOB_COLUMNS = (
    "sernum",
    "benunit",
    "person",
    "jobtype",
    "etype",
    "ftpt",
    "wrkprev",
    "ugross",
    "ugrspay",
    "sejblong",
    "sejbmths",
    "selwks",
    "seincamt",
    "profit1",
    "profit2",
)
_HOUSEHOLD_COLUMNS = ("sernum", "intdate")
_FRS_TABLES = ("adult", "job", "househol")

#: Spell sources, for the receipt.
SOURCE_FULL_YEAR = "current, 12 or more months in the job"
SOURCE_CHANGER_CALENDAR = "current, job changer, months from the calendar"
SOURCE_CHANGER_CONTINUOUS = "current, job changer, situation unchanged"
SOURCE_ENTRANT = "current, entrant"
SOURCE_UNPAID_EARLIER = "in work without recorded pay, earlier spell"
SOURCE_SWITCHER = "in work, employee months before the current activity"
SOURCE_LEAVER_LSTYR = "leaver, weeks of regular work"
SOURCE_LEAVER_CALENDAR = "leaver, months from the calendar"
SOURCE_LEAVER_LEAVING_DATE = "leaver, months from the leaving date"
SOURCE_CALENDAR_ONLY = "out of work, months from the calendar only"
SOURCE_CEASED_SELF_EMPLOYED = "ceased self-employed"
SPELL_SOURCES = (
    SOURCE_FULL_YEAR,
    SOURCE_CHANGER_CALENDAR,
    SOURCE_CHANGER_CONTINUOUS,
    SOURCE_ENTRANT,
    SOURCE_UNPAID_EARLIER,
    SOURCE_SWITCHER,
    SOURCE_LEAVER_LSTYR,
    SOURCE_LEAVER_CALENDAR,
    SOURCE_LEAVER_LEAVING_DATE,
    SOURCE_CALENDAR_ONLY,
    SOURCE_CEASED_SELF_EMPLOYED,
)


@dataclass(frozen=True)
class UKFRSAnnualIncomeResult:
    """Output frame and aggregate receipt for the annual income descriptors."""

    frame: Frame
    receipt: dict[str, object]

    def evidence(self) -> dict[str, object]:
        return {"stage": FRS_ANNUAL_INCOME_STAGE_NAME, **self.receipt}


@dataclass(frozen=True)
class UKFRSAnnualIncomeStageTransform:
    """Whole-stage callable for the FRS annual income descriptors."""

    raw_dir: str | Path
    stage: SourceStageSpec
    last_result: UKFRSAnnualIncomeResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = add_frs_annual_income_spells(frame, self.raw_dir, stage=self.stage)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_ANNUAL_INCOME_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def frs_annual_income_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            "read_tables",
            {
                "format": "tab",
                "delimiter": "\t",
                "lowercase_columns": True,
                "numeric_errors": "coerce",
                "runtime_sha256_required": True,
            },
        ),
        (
            FRS_ANNUAL_INCOME_OPERATION_KIND,
            {
                "outputs": list(FRS_ANNUAL_INCOME_OUTPUT_COLUMNS),
                "window_months": WINDOW_MONTHS,
                "leaving_date_share": LEAVING_DATE_SHARE,
                "unknown_leaving_months": UNKNOWN_LEAVING_MONTHS,
                "donor_recent_start_months": DONOR_RECENT_START_MONTHS,
                "minimum_donors": MINIMUM_DONORS,
                "donor_age_edges": list(DONOR_AGE_EDGES),
                "seed": DRAW_SEED,
                "salt": DRAW_SALT,
                "earnings_index": EARNINGS_INDEX_PARAMETER,
                "earnings_index_from": EARNINGS_INDEX_FROM,
                "earnings_index_to": EARNINGS_INDEX_TO,
                "maximum_uprating_months": MAXIMUM_UPRATING_MONTHS,
            },
        ),
    )


def add_frs_annual_income_spells(
    frame: Frame, raw_dir: str | Path, *, stage: SourceStageSpec
) -> UKFRSAnnualIncomeResult:
    """Add the annual income descriptors from the pinned FRS tabs."""

    artifacts = _artifact_by_table(stage)
    tables = {
        name: normalize_ids(
            read_pinned_tab(
                Path(raw_dir) / str(artifacts[name]["locator"]),
                artifacts[name],
                columns=columns,
            )
        )
        for name, columns in (
            ("adult", _ADULT_COLUMNS),
            ("job", _JOB_COLUMNS),
            ("househol", _HOUSEHOLD_COLUMNS),
        )
    }
    person = frame.table("person")
    household = frame.table("household")
    household_weights = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"].to_numpy(),
    )
    person_weights = (
        person["person_household_id"].map(household_weights).to_numpy(dtype=float)
    )
    if not np.isfinite(person_weights).all():
        raise ValueError(
            f"{FRS_ANNUAL_INCOME_STAGE_NAME} needs a household weight for every "
            "person row."
        )
    interview = frs_interview_dates(tables["househol"].set_index("household_id"))
    descriptors, receipt = derive_frs_annual_income_spells(
        person,
        person_weights=person_weights,
        adult=tables["adult"],
        job=tables["job"],
        interview=interview,
        earnings_growth=load_earnings_growth(),
    )
    output = person.copy()
    for column in FRS_ANNUAL_INCOME_OUTPUT_COLUMNS:
        output[column] = descriptors[column].to_numpy(dtype=np.float64)
    result = uk_national_frame(
        person=output,
        benunit=frame.table("benunit"),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return UKFRSAnnualIncomeResult(frame=result, receipt=receipt)


def load_earnings_growth() -> float:
    """The pinned OBR average earnings index's growth over the survey window."""

    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(EARNINGS_INDEX_RESOURCE)
        .read_text()
    )
    values = payload["values"][EARNINGS_INDEX_PARAMETER]
    growth = float(values[EARNINGS_INDEX_TO]) / float(values[EARNINGS_INDEX_FROM]) - 1
    if not np.isfinite(growth) or not 0.0 <= growth < 0.25:
        raise ValueError(
            f"{EARNINGS_INDEX_PARAMETER} grows {growth!r} from {EARNINGS_INDEX_FROM} "
            f"to {EARNINGS_INDEX_TO}; expected a finite rate in [0, 0.25)."
        )
    return growth


def derive_frs_annual_income_spells(
    person: pd.DataFrame,
    *,
    person_weights: np.ndarray,
    adult: pd.DataFrame,
    job: pd.DataFrame,
    interview: pd.Series,
    earnings_growth: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Person-aligned descriptors and their aggregate receipt.

    ``person`` is the spine's person table at the raw FRS identity
    (``person_id`` = sernum * 1000 + person). ``adult`` and ``job`` are the
    pinned tabs after :func:`normalize_ids`; ``interview`` is the interview date
    by ``household_id``. ``person_weights`` are the household weights each
    person row carries.
    """

    person_ids = person["person_id"].to_numpy()
    rows = len(person)
    weights = np.asarray(person_weights, dtype=float)
    if weights.shape != (rows,):
        raise ValueError("person weights must align with the person table.")
    raw = adult.drop_duplicates("person_id").set_index("person_id")
    if len(raw) != len(adult):
        raise ValueError("FRS adult.tab carries a person twice.")
    is_adult = np.isin(person_ids, raw.index.to_numpy())
    a = raw.reindex(person_ids)
    main = (
        job.loc[_numeric(job, "jobtype") == 1]
        .drop_duplicates("person_id")
        .set_index("person_id")
        .reindex(person_ids)
    )

    current = _amount(person, "employment_income")
    pay = (current > 0) & is_adult
    se_current = _amount(person, "self_employment_income")
    working = (_numeric(a, "working") == WORKING_CODE) | (
        _numeric(a, "jobaway") == WORKING_CODE
    )
    ilo = _numeric(a, "empstati")
    ilo_employee = np.isin(ilo, ILO_EMPLOYEE_CODES)

    # Twelve months ending at the interview: month 0 is now (SELFDEMP), months
    # 1-11 are SDEMP01-SDEMP11. Only a reported change of situation carries a
    # calendar: SAMESIT 2 copies today's status into every month.
    changed = _numeric(a, "samesit") == SITUATION_CHANGED_CODE
    now = _numeric(a, "selfdemp")
    calendar = np.column_stack(
        [now, *(_numeric(a, f"sdemp{month:02d}") for month in range(1, 12))]
    )
    calendar = np.where(changed[:, None], calendar, now[:, None])
    calendar_employee = (
        np.isin(calendar, SELF_DEFINED_EMPLOYEE_CODES) & changed[:, None]
    )
    calendar_self_employed = (
        np.isin(calendar, SELF_DEFINED_SELF_EMPLOYED_CODES) & changed[:, None]
    )
    calendar_employee_months = calendar_employee.sum(axis=1).astype(float)
    calendar_self_employed_months = calendar_self_employed.sum(axis=1).astype(float)

    months_in_job = _numeric(a, "currjobm")
    in_job = np.where(
        np.isfinite(months_in_job),
        np.minimum(np.floor(np.maximum(months_in_job, 0.0)) + 1.0, WINDOW_MONTHS),
        float(WINDOW_MONTHS),
    )
    before_job = (
        calendar_employee & (np.arange(WINDOW_MONTHS)[None, :] >= in_job[:, None])
    ).sum(axis=1)
    wrkprev = _numeric(main, "wrkprev")

    source = np.full(rows, "", dtype=object)
    employment_share = np.zeros(rows)
    other_months = np.zeros(rows)

    full_year = pay & (in_job >= WINDOW_MONTHS)
    changer = pay & ~full_year & (wrkprev == WRKPREV_PAID_EMPLOYMENT_CODE)
    entrant = pay & ~full_year & ~changer
    employment_share[full_year] = 1.0
    employment_share[changer & changed] = (
        np.minimum(in_job + before_job, WINDOW_MONTHS) / WINDOW_MONTHS
    )[changer & changed]
    employment_share[changer & ~changed] = 1.0
    employment_share[entrant] = (
        np.minimum(in_job + before_job, WINDOW_MONTHS) / WINDOW_MONTHS
    )[entrant]
    source[full_year] = SOURCE_FULL_YEAR
    source[changer & changed] = SOURCE_CHANGER_CALENDAR
    source[changer & ~changed] = SOURCE_CHANGER_CONTINUOUS
    source[entrant] = SOURCE_ENTRANT

    unpaid = is_adult & working & ilo_employee & ~pay
    unpaid_months = np.where(in_job < WINDOW_MONTHS, before_job, 0).astype(float)
    other_months = np.where(unpaid, unpaid_months, other_months)
    source[unpaid & (unpaid_months > 0)] = SOURCE_UNPAID_EARLIER
    switcher = is_adult & working & ~pay & ~unpaid & (calendar_employee_months > 0)
    other_months = np.where(switcher, calendar_employee_months, other_months)
    source[switcher] = SOURCE_SWITCHER

    # Out of work: the last-job block (NUMJOB2 and its JOB rows).
    etype = _numeric(main, "etype")
    last_job = is_adult & ~working & np.isfinite(_numeric(a, "numjob2"))
    employee_leaver = last_job & (etype == ETYPE_EMPLOYEE_CODE)
    ceased_self_employed = last_job & np.isin(etype, ETYPE_SELF_EMPLOYED_CODES)
    interview_dates = pd.to_datetime(
        pd.Series(person["person_household_id"].to_numpy()).map(interview)
    )
    if interview_dates.isna().any():
        raise ValueError(
            f"{FRS_ANNUAL_INCOME_STAGE_NAME} needs an interview date for every "
            "person's household."
        )
    interview_year = interview_dates.dt.year.to_numpy(dtype=float)
    interview_month = interview_dates.dt.month.to_numpy(dtype=float)
    left_year = _numeric(a, "lstwrk2")
    left_month = _numeric(a, "lstwrk1")
    months_since_left = np.where(
        np.isfinite(left_month),
        (interview_year - left_year) * 12.0 + (interview_month - left_month),
        (interview_year - left_year) * 12.0,
    )
    retired = _numeric(a, "retire") == RETIRED_IN_WINDOW_CODE
    months_since_retiring = np.mod(interview_month - _numeric(a, "retire1"), 12.0)
    months_since_left = np.where(
        np.isfinite(months_since_left),
        months_since_left,
        np.where(retired, months_since_retiring, np.nan),
    )
    leaving_date_months = np.where(
        np.isfinite(months_since_left),
        np.where(
            months_since_left >= WINDOW_MONTHS,
            0.0,
            LEAVING_DATE_SHARE
            * (WINDOW_MONTHS - np.clip(months_since_left, 0.0, WINDOW_MONTHS)),
        ),
        LEAVING_DATE_SHARE * UNKNOWN_LEAVING_MONTHS,
    )
    weeks_worked = _numeric(a, "lstyr")
    regular_weeks = np.isfinite(weeks_worked) & (weeks_worked > 0)
    leaver_months = np.where(
        regular_weeks,
        np.minimum(weeks_worked, 52.0) * WINDOW_MONTHS / 52.0,
        np.where(
            calendar_employee_months > 0, calendar_employee_months, leaving_date_months
        ),
    )
    other_months = np.where(employee_leaver, leaver_months, other_months)
    source[employee_leaver & regular_weeks] = SOURCE_LEAVER_LSTYR
    source[employee_leaver & ~regular_weeks & (calendar_employee_months > 0)] = (
        SOURCE_LEAVER_CALENDAR
    )
    source[employee_leaver & ~regular_weeks & ~(calendar_employee_months > 0)] = (
        SOURCE_LEAVER_LEAVING_DATE
    )
    calendar_only = (
        is_adult
        & ~working
        & ~employee_leaver
        & ~ceased_self_employed
        & (calendar_employee_months > 0)
    )
    other_months = np.where(calendar_only, calendar_employee_months, other_months)
    source[calendar_only] = SOURCE_CALENDAR_ONLY

    # Pay for those months: the last job's usual pay, else a donor's rate.
    uprating = (1.0 + float(earnings_growth)) ** (
        np.clip(
            np.nan_to_num(months_since_left, nan=UNKNOWN_LEAVING_MONTHS),
            0.0,
            MAXIMUM_UPRATING_MONTHS,
        )
        / WINDOW_MONTHS
    )
    ratio, ratio_support = pay_concept_ratio(person_ids, weights, job, working)
    usual = _numeric(main, "ugross")
    observed = employee_leaver & np.isfinite(usual) & (usual > 0)
    weekly_rate = np.zeros(rows)
    weekly_rate[observed] = (usual * ratio * uprating)[observed]
    needs_donor = (other_months > 0) & ~observed
    age = _amount(person, "age")
    male = (person["gender"].astype(str) == "MALE").to_numpy()
    part_time = np.isin(calendar, (SELF_DEFINED_PART_TIME_EMPLOYEE_CODE,)).any(
        axis=1
    ) & ~np.isin(calendar, (SELF_DEFINED_EMPLOYEE_CODES[0],)).any(axis=1)
    ftpt = _numeric(main, "ftpt")
    part_time = np.where(
        employee_leaver & np.isfinite(ftpt), ftpt == JOB_PART_TIME_CODE, part_time
    )
    donor = (
        pay
        & ilo_employee
        & np.isfinite(months_in_job)
        & (months_in_job <= DONOR_RECENT_START_MONTHS)
        & (weights > 0)
    )
    donor_part_time = ilo == ILO_PART_TIME_EMPLOYEE_CODE
    draws = stable_identity_uniforms(person_ids, seed=DRAW_SEED, salt=DRAW_SALT)
    donor_rates, donor_cells = _donor_rates(
        needs_donor,
        donor=donor,
        current_weekly=current / WEEKS_IN_YEAR,
        weights=weights,
        age=age,
        male=male,
        part_time=part_time,
        donor_part_time=donor_part_time,
        draws=draws,
    )
    weekly_rate[needs_donor] = donor_rates[needs_donor]
    employment_other = weekly_rate * WEEKS_IN_YEAR * other_months / WINDOW_MONTHS

    # Self-employment.
    se_now = (se_current > 0) & is_adult
    years_in_business = _numeric(main, "sejblong")
    months_in_business = _numeric(main, "sejbmths")
    se_share = np.ones(rows)
    new_business = se_now & (years_in_business == 0) & np.isfinite(months_in_business)
    se_share[new_business] = (
        np.minimum(months_in_business + 1.0, WINDOW_MONTHS) / WINDOW_MONTHS
    )[new_business]
    # The calendar records the main activity only, so it bounds a business
    # only where self-employment is the activity today: a side business beside
    # a job or a pension keeps its whole year.
    se_switcher = (
        se_now
        & changed
        & np.isin(now, SELF_DEFINED_SELF_EMPLOYED_CODES)
        & (calendar_self_employed_months < WINDOW_MONTHS)
    )
    se_share[se_switcher] = np.minimum(
        se_share, np.maximum(calendar_self_employed_months, 1.0) / WINDOW_MONTHS
    )[se_switcher]
    se_income = _numeric(main, "seincamt")
    profit = np.where(
        _numeric(main, "profit2") == PROFIT_LOSS_CODE, 0.0, _numeric(main, "profit1")
    )
    se_rate = np.where(
        np.isfinite(se_income) & (se_income > 0),
        se_income,
        np.where(np.isfinite(profit), np.maximum(profit, 0.0), 0.0),
    )
    se_weeks = _numeric(main, "selwks")
    se_months = np.where(
        np.isfinite(se_weeks),
        np.minimum(np.maximum(se_weeks, 0.0), 52.0) * WINDOW_MONTHS / 52.0,
        np.where(
            regular_weeks,
            np.minimum(weeks_worked, 52.0) * WINDOW_MONTHS / 52.0,
            np.where(
                calendar_self_employed_months > 0,
                calendar_self_employed_months,
                leaving_date_months,
            ),
        ),
    )
    se_other = np.where(
        ceased_self_employed,
        se_rate * uprating * WEEKS_IN_YEAR * se_months / WINDOW_MONTHS,
        0.0,
    )
    source[ceased_self_employed & (se_other > 0) & (source == "")] = (
        SOURCE_CEASED_SELF_EMPLOYED
    )

    # Pensions started on retiring, and contributory benefits on leaving.
    pension_share = np.ones(rows)
    retired_in_window = is_adult & retired & np.isfinite(months_since_retiring)
    pension_share[retired_in_window] = (
        np.minimum(months_since_retiring + 0.5, WINDOW_MONTHS) / WINDOW_MONTHS
    )[retired_in_window]
    months_worked = np.maximum(
        np.where(employee_leaver | calendar_only, other_months, 0.0),
        np.where(ceased_self_employed, se_months, 0.0),
    )
    benefits_share = np.ones(rows)
    left_work = (employee_leaver | calendar_only | ceased_self_employed) & (
        months_worked > 0
    )
    benefits_share[left_work] = (
        np.clip(WINDOW_MONTHS - months_worked, 0.0, WINDOW_MONTHS) / WINDOW_MONTHS
    )[left_work]

    descriptors = pd.DataFrame(
        {
            EMPLOYMENT_INCOME_ANNUAL_SHARE: employment_share,
            EMPLOYMENT_INCOME_OTHER_SPELLS: employment_other,
            SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE: np.where(is_adult, se_share, 1.0),
            SELF_EMPLOYMENT_INCOME_OTHER_SPELLS: se_other,
            PRIVATE_PENSION_INCOME_ANNUAL_SHARE: pension_share,
            CONTRIBUTORY_BENEFITS_ANNUAL_SHARE: benefits_share,
        },
        index=person.index,
    )
    # Anyone outside adult.tab, and any adult with no current pay and no
    # spell, keeps the current columns as they are: a share of 1 on a zero
    # current amount and nothing from other spells.
    no_employee_pay = ~pay
    descriptors.loc[no_employee_pay, EMPLOYMENT_INCOME_ANNUAL_SHARE] = 1.0
    _check_descriptors(descriptors)
    receipt = _receipt(
        person,
        weights=weights,
        descriptors=descriptors,
        current=current,
        se_current=se_current,
        pay=pay,
        source=source,
        observed=observed,
        needs_donor=needs_donor,
        donor_cells=donor_cells,
        ratio=ratio,
        ratio_support=ratio_support,
        earnings_growth=earnings_growth,
        employment_share=employment_share,
    )
    return descriptors, receipt


def pay_concept_ratio(
    person_ids: np.ndarray,
    weights: np.ndarray,
    job: pd.DataFrame,
    working: np.ndarray,
) -> tuple[float, int]:
    """Weighted UGRSPAY / UGROSS on current employee main jobs reporting both.

    UGRSPAY is the DV pay the spine's ``employment_income`` sums; UGROSS is the
    usual gross pay the survey asks of a current job whose last pay was not
    usual and of a recent leaver's last job. Measured job by job on one job per
    person, so a second job's pay never enters it.
    """

    weight_by_person = pd.Series(weights, index=person_ids)
    working_by_person = pd.Series(np.asarray(working, dtype=bool), index=person_ids)
    jobs = job.loc[
        (_numeric(job, "jobtype") == 1)
        & (_numeric(job, "etype") == ETYPE_EMPLOYEE_CODE)
        & (_numeric(job, "ugross") > 0)
        & (_numeric(job, "ugrspay") > 0)
    ]
    jobs = jobs.loc[
        working_by_person.reindex(jobs["person_id"].to_numpy())
        .fillna(False)
        .to_numpy(dtype=bool)
    ]
    job_weights = weight_by_person.reindex(jobs["person_id"].to_numpy()).to_numpy(
        dtype=float
    )
    usual = _numeric(jobs, "ugross")
    dv = _numeric(jobs, "ugrspay")
    support = distinct_carriers(jobs["household_id"].to_numpy(), job_weights)
    denominator = float(np.nansum(job_weights * usual))
    if support < 10 or denominator <= 0:
        raise ValueError(
            f"{FRS_ANNUAL_INCOME_STAGE_NAME} cannot measure the usual-to-DV pay "
            "ratio: fewer than ten households report both."
        )
    ratio = float(np.nansum(job_weights * dv)) / denominator
    lower, upper = PAY_CONCEPT_RATIO_BOUNDS
    if not lower <= ratio <= upper:
        raise ValueError(
            f"{FRS_ANNUAL_INCOME_STAGE_NAME}: the DV-to-usual pay ratio {ratio:.4f} "
            f"lies outside {PAY_CONCEPT_RATIO_BOUNDS}."
        )
    return ratio, support


def _donor_rates(
    needs_donor: np.ndarray,
    *,
    donor: np.ndarray,
    current_weekly: np.ndarray,
    weights: np.ndarray,
    age: np.ndarray,
    male: np.ndarray,
    part_time: np.ndarray,
    donor_part_time: np.ndarray,
    draws: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """A donor's weekly pay for each recipient, by weighted inverse CDF."""

    rates = np.zeros(len(needs_donor))
    band = np.digitize(np.nan_to_num(age), DONOR_AGE_EDGES)
    cells = {"cells": 0, "cells_without_sex": 0}
    if not needs_donor.any():
        return rates, cells
    if not donor.any():
        raise ValueError(f"{FRS_ANNUAL_INCOME_STAGE_NAME} has no pay donors.")
    keys = {
        (int(b), bool(m), bool(p))
        for b, m, p in zip(
            band[needs_donor], male[needs_donor], part_time[needs_donor], strict=True
        )
    }
    for age_band, is_male, is_part_time in sorted(keys):
        recipients = (
            needs_donor
            & (band == age_band)
            & (male == is_male)
            & (part_time == is_part_time)
        )
        pool = donor & (band == age_band) & (donor_part_time == is_part_time)
        with_sex = pool & (male == is_male)
        cells["cells"] += 1
        if with_sex.sum() >= MINIMUM_DONORS:
            pool = with_sex
        else:
            cells["cells_without_sex"] += 1
        if pool.sum() < MINIMUM_DONORS:
            pool = donor & (donor_part_time == is_part_time)
        if not pool.any():
            pool = donor
        order = np.flatnonzero(pool)
        cumulative = np.cumsum(weights[order])
        cumulative /= cumulative[-1]
        picks = np.searchsorted(cumulative, draws[recipients], side="right")
        picks = np.minimum(picks, len(order) - 1)
        rates[recipients] = current_weekly[order[picks]]
    return rates, cells


def _check_descriptors(descriptors: pd.DataFrame) -> None:
    values = descriptors.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{FRS_ANNUAL_INCOME_STAGE_NAME} wrote a non-finite value.")
    for column in ANNUAL_SHARE_COLUMNS:
        share = descriptors[column].to_numpy(dtype=float)
        if ((share < 0) | (share > 1)).any():
            raise ValueError(f"{column} must lie in [0, 1].")
    for column in OTHER_SPELLS_COLUMNS:
        if (descriptors[column].to_numpy(dtype=float) < 0).any():
            raise ValueError(f"{column} must not be negative.")


def _receipt(
    person: pd.DataFrame,
    *,
    weights: np.ndarray,
    descriptors: pd.DataFrame,
    current: np.ndarray,
    se_current: np.ndarray,
    pay: np.ndarray,
    source: np.ndarray,
    observed: np.ndarray,
    needs_donor: np.ndarray,
    donor_cells: Mapping[str, int],
    ratio: float,
    ratio_support: int,
    earnings_growth: float,
    employment_share: np.ndarray,
) -> dict[str, object]:
    households = person["person_household_id"].to_numpy()

    def carriers(mask: np.ndarray) -> int:
        return distinct_carriers(households[mask], weights[mask])

    def weighted(values: np.ndarray, mask: np.ndarray) -> float | str:
        return small_cell_share(float((weights * values)[mask].sum()), carriers(mask))

    other = descriptors[EMPLOYMENT_INCOME_OTHER_SPELLS].to_numpy(dtype=float)
    annual = current * descriptors[EMPLOYMENT_INCOME_ANNUAL_SHARE].to_numpy() + other
    se_annual = se_current * descriptors[
        SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE
    ].to_numpy() + descriptors[SELF_EMPLOYMENT_INCOME_OTHER_SPELLS].to_numpy(
        dtype=float
    )
    annual_earner = annual > 0
    outside = (other > 0) & ~pay
    # Each source gives its support band only: weighted counts by source beside
    # the totals below would give back a suppressed source by difference.
    sources = {
        label: support_band(carriers(source == label)) for label in SPELL_SOURCES
    }
    # Structural checks the stage gate reads: each is zero by construction,
    # and a count of rows that break one is banded as the other stages band
    # theirs (zero exact), so a failing build never records 1 to 9.
    full_year = source == SOURCE_FULL_YEAR
    return {
        "window_months": WINDOW_MONTHS,
        "weights": "household weights at the stage, carried by each person",
        "pay_concept_ratio": ratio,
        "pay_concept_ratio_support": support_band(ratio_support),
        "earnings_growth": float(earnings_growth),
        "spell_sources": sources,
        "donor_cells": dict(donor_cells),
        "annual_employee_earners": weighted(np.ones(len(pay)), annual_earner),
        "current_employee_earners": weighted(np.ones(len(pay)), pay),
        "annual_over_current_earners": (
            "<10"
            if carriers(annual_earner) < 10 or carriers(pay) < 10
            else float(weights[annual_earner].sum()) / float(weights[pay].sum())
        ),
        "annual_over_current_employment_income": (
            float((weights * annual).sum()) / float((weights * current).sum())
            if carriers(pay) >= 10
            else "<10"
        ),
        "annual_over_current_self_employment_income": (
            float((weights * se_annual).sum()) / float((weights * se_current).sum())
            if carriers(se_current > 0) >= 10
            else "<10"
        ),
        "other_spells_pay_observed_share": (
            "<10"
            if carriers(outside) < 10
            else float(weights[outside & observed].sum())
            / float(weights[outside].sum())
        ),
        "other_spells_pay_donor_support": support_band(carriers(needs_donor)),
        "full_year_rows_not_whole": extreme_count(
            int((full_year & (employment_share != 1.0)).sum())
        ),
        "rows_outside_share_bounds": extreme_count(
            int(
                sum(
                    ((descriptors[column] < 0) | (descriptors[column] > 1))
                    .to_numpy()
                    .sum()
                    for column in ANNUAL_SHARE_COLUMNS
                )
            )
        ),
        "rows_negative_other_spells": extreme_count(
            int(
                sum(
                    (descriptors[column] < 0).to_numpy().sum()
                    for column in OTHER_SPELLS_COLUMNS
                )
            )
        ),
    }


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {
        str(artifact.get("table")): artifact
        for artifact in stage.artifacts
        if artifact.get("table")
    }
    missing = [name for name in _FRS_TABLES if name not in by_table]
    if missing:
        raise ValueError(
            f"{FRS_ANNUAL_INCOME_STAGE_NAME} manifest is missing the FRS tables "
            f"{missing}."
        )
    return by_table


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    if stage.stage != FRS_ANNUAL_INCOME_STAGE_NAME:
        raise ValueError(
            f"FRS annual income received stage {stage.stage!r}, expected "
            f"{FRS_ANNUAL_INCOME_STAGE_NAME!r}."
        )
    kinds = tuple(operation.kind for operation in stage.operations)
    expected = frs_annual_income_operation_parameters()
    if kinds != tuple(kind for kind, _ in expected):
        raise ValueError(
            f"{FRS_ANNUAL_INCOME_STAGE_NAME} operation order drifted: expected "
            f"{tuple(kind for kind, _ in expected)}, got {kinds}."
        )
    for operation, (kind, parameters) in zip(stage.operations, expected, strict=True):
        if dict(operation.parameters) != parameters:
            raise ValueError(
                f"{FRS_ANNUAL_INCOME_STAGE_NAME} operation {kind!r} parameters "
                "drifted from the reviewed values."
            )
    if (
        tuple(stage.outputs) != FRS_ANNUAL_INCOME_OUTPUT_COLUMNS
        or tuple(stage.rewrites) != ()
    ):
        raise ValueError(
            f"{FRS_ANNUAL_INCOME_STAGE_NAME} must declare exactly the outputs "
            f"{FRS_ANNUAL_INCOME_OUTPUT_COLUMNS} and no rewrites."
        )
    if stage.grain != "person":
        raise ValueError(
            f"{FRS_ANNUAL_INCOME_STAGE_NAME} is person grain, got {stage.grain!r}."
        )
    resources = {
        str(artifact["resource"])
        for artifact in stage.artifacts
        if "resource" in artifact
    }
    if resources != {EARNINGS_INDEX_RESOURCE}:
        raise ValueError(
            f"{FRS_ANNUAL_INCOME_STAGE_NAME} must declare exactly the resource "
            f"{EARNINGS_INDEX_RESOURCE!r}."
        )


def _numeric(table: pd.DataFrame, column: str) -> np.ndarray:
    """A raw FRS column as floats; blanks, text and negative codes are missing."""

    if column not in table.columns:
        raise KeyError(f"{FRS_ANNUAL_INCOME_STAGE_NAME} needs FRS column {column!r}.")
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
    return np.where(values < 0, np.nan, values)


def _amount(table: pd.DataFrame, column: str) -> np.ndarray:
    if column not in table.columns:
        raise KeyError(f"{FRS_ANNUAL_INCOME_STAGE_NAME} needs spine column {column!r}.")
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"spine column {column!r} must be finite.")
    return values


__all__ = [
    "ANNUAL_SHARE_COLUMNS",
    "CONTRIBUTORY_BENEFITS_ANNUAL_SHARE",
    "CURRENT_INCOME_COLUMNS",
    "EMPLOYMENT_INCOME_ANNUAL_SHARE",
    "EMPLOYMENT_INCOME_OTHER_SPELLS",
    "FRS_ANNUAL_INCOME_OPERATION_KIND",
    "FRS_ANNUAL_INCOME_OUTPUT_COLUMNS",
    "FRS_ANNUAL_INCOME_STAGE_NAME",
    "OTHER_SPELLS_COLUMNS",
    "PRIVATE_PENSION_INCOME_ANNUAL_SHARE",
    "SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE",
    "SELF_EMPLOYMENT_INCOME_OTHER_SPELLS",
    "SPELL_SOURCES",
    "UKFRSAnnualIncomeResult",
    "UKFRSAnnualIncomeStageTransform",
    "add_frs_annual_income_spells",
    "derive_frs_annual_income_spells",
    "frs_annual_income_operation_parameters",
    "load_earnings_growth",
    "pay_concept_ratio",
]
