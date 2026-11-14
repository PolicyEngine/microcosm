"""Immigration inputs for the ACS rows of the retained ACS local lane.

The immigration stage (:mod:`~microcosm.build.us_runtime.immigration`) reads
raw CPS ASEC columns, so it labels only the donor spine. The ACS transfer
leaves immigration to the runtime, and the local lane never ran it on ACS
rows: every ACS person reached the engine pass with ``immigration_status_str``
and ``ssn_card_type`` missing and the reviewed-null fill wrote the engine
default, ``CITIZEN``. ``years_since_us_entry`` was never stored at all, so every
person in the file got the engine default of 5 (microcosm#1020).

This fresh-build stage runs the same cited method on the ACS rows through a
CPS-named view of each ACS person record, and fills only missing cells:
donor-spine immigration values and any stored ACS value are kept unchanged.

- **Citizenship is measured.** ACS ``CIT`` uses the ``PRCITSHP`` coding
  (1-4 citizens, 5 non-citizens).
- **Arrival is the exact year of entry.** ``YOEP`` replaces the ASEC
  ``PEINUSYR`` bins, and the kernel reads the IRCA cohort and the residence
  windows from the year itself.
- **Legal-status indicators** map to their nearest ACS fields
  (:data:`ACS_TO_CPS_INDICATOR_MAPPING`). Three ASEC indicators have no ACS
  field (:data:`ACS_UNMAPPED_INDICATORS`) and are set to explicit
  non-indicator values. ACS coverage is measured at interview, while the CPS
  fields cover any time last year. Both gaps can only leave the ACS residual
  (likely undocumented) pool the same size or larger than the ASEC method
  would find for the same people.
- **Labor-force status is measured.** The kernel's worker test is CPS
  ``A_LFSR`` (employed or unemployed) at age 16 or older, the universe of
  Pew's unauthorized labor force. ACS ``ESR`` carries the same current
  status (:data:`ACS_TO_CPS_INDICATOR_MAPPING`). The donor's delivered
  workers are measured the same way when the donor carries ``A_LFSR``; the
  support base drops that column after its immigration stage, so such a
  donor falls back to the earnings pair, and the receipt names the test it
  used (microcosm#1554).
- **The ACS rows target the donor residual.** The Pew worker and Higher Ed
  student controls are national counts, and so is every cited humanitarian
  stock (microcosm#767). The donor spine was labelled against the full
  controls on its own mass and then pooled at rescaled household weights, so
  it already delivers part of each count, and not necessarily the share its
  person weight suggests. The stage measures that part on the frame it sees
  and targets the ACS rows at ``max(0, control - donor delivered)``. The
  worker and student controls bind Pew's broad unauthorized universe
  (residual ``NONE`` rows, DACA, the temporary protections Pew retains and
  EAD Cuban/Haitian entrants), so their delivered counts are the donor rows
  in that universe who work or attend college; a humanitarian draw's
  delivered count is the donor rows already carrying its status inside its
  origin, arrival and SSN cohort. Donor plus ACS then pools to each control
  whenever the ACS rows have the capacity (microcosm#1052 review, following
  the #1060 residual targets). The receipt records, per control and per
  humanitarian draw, the national count, the donor delivered count, the
  residual, the ACS assigned count and the pooled total, plus the ACS
  capacity for the worker and student controls.
- **Draws** are the kernel's seeded blake2b draws, keyed on
  ``acs_2024_1yr:SERIALNO:SPORDER``. ``SPORDER`` alone (the ACS
  ``source_person_id``) repeats in every household.

``years_since_us_entry`` is filled on both spines. It is an **arrival-based
status-duration proxy**. For the foreign-born it is the time period minus the
measured arrival year (the exact ACS ``YOEP``, or the ASEC ``PEINUSYR`` bin
year the immigration stage itself uses,
:func:`~microcosm.build.us_runtime.immigration._asec_arrival_year`), clipped
at zero. For the US-born (``CIT`` / ``PRCITSHP`` 1-3) it is their age: years in
the US since birth. The engine reads it as the five-year-bar clock, which
8 U.S.C. 1613(a) starts at entry *with qualified status*.

**Known limitation: the clock runs from arrival.** Neither the ACS nor the
ASEC donor records the date qualified status was granted. The proxy therefore
runs long, and can clear the five-year bar early, for anyone who adjusted
status after arriving: a person who arrived in 2014 and obtained qualified
status in 2023 gets 10 modeled years in 2024 but has one qualifying year
(policyengine-us#9658). That matters for the policyengine-us 2.2.1 tests
that read the clock. The Medicaid immigration-status test (CHIP follows it,
because CHIP excludes Medicaid-eligible children) and Washington TANF read
it against a five-year bar, so the long clock can clear the bar early.
Washington RCA reads it the other way, as a window that closes
(``years_since_us_entry < window_years``), so the long clock can wrongly
exclude a recently qualified person from RCA. Under the statute the SNAP
five-year bar also runs from entry with
qualified status, but the pinned engine's federal SNAP status test reads
the status, not this clock, so any SNAP effect is indirect. The receipt's
``adjustment_lag_sensitivity`` block reports the size of the affected group
under each lag in :data:`ADJUSTMENT_LAG_YEARS` (one to ten, 15 and 20 years):
the ACS weight assigned a status the bar applies to
(:data:`FIVE_YEAR_BAR_STATUSES`) whose proxy falls in ``[5, 5 + lag)``, which
is the mass whose bar status would flip under that lag or any shorter one; that
mass as a cumulative share of all bar-status weight; and the largest observed
proxy. The real-data check is microcosm#1093.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    ACS_SOURCE_COLUMN_RENAMES,
)
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.immigration import (
    _ARRIVAL_YEAR_MIDPOINTS,
    _CPS_LABOR_FORCE_STATUS_CODES,
    _NON_CITIZEN_SHARE_BAND,
    _PEW_INCLUDED_HUMANITARIAN_STATUS_VALUES,
    _SSN_CODE_TO_NAME,
    IMMIGRATION_STATUS_VALUES,
    SSN_CARD_TYPE_VALUES,
    US_IMMIGRATION_OUTPUT_COLUMNS,
    HumanitarianDraw,
    ImmigrationControls,
    _asec_arrival_year,
    _assign_ssn_card_codes,
    _derive_immigration_status,
    _humanitarian_emitted_mask,
    _pew_unauthorized_rows,
    _source_aware_immigration_profile,
    us_immigration_composition_gate,
    us_immigration_controls,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_IMMIGRATION_SOURCE_COLUMNS",
    "ACS_LOCAL_IMMIGRATION_COLUMNS",
    "ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD",
    "ACS_LOCAL_IMMIGRATION_GATE_NAME",
    "ACS_LOCAL_IMMIGRATION_ISSUE",
    "ACS_TO_CPS_INDICATOR_MAPPING",
    "ACS_UNMAPPED_INDICATORS",
    "ADJUSTMENT_LAG_YEARS",
    "ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY",
    "FIVE_YEAR_BAR_STATUSES",
    "FIVE_YEAR_BAR_YEARS",
    "YEARS_SINCE_US_ENTRY_COLUMN",
    "acs_local_immigration_receipt_failures",
    "acs_local_immigration_signal_gate",
    "require_acs_local_immigration_donor",
    "with_acs_local_immigration_inputs",
]

ACS_LOCAL_IMMIGRATION_ISSUE = "microcosm#1020"
#: The engine's five-year-bar clock. This stage fills it with an
#: arrival-based status-duration proxy (years since measured arrival), which
#: can clear the bar early for people who gained qualified status later.
YEARS_SINCE_US_ENTRY_COLUMN = "years_since_us_entry"
#: How the ACS targets are set: each national control minus the donor rows'
#: pooled delivered count, floored at zero. The microcosm#1052 review set the
#: rule for the worker and student controls; v2 (microcosm#767) measures their
#: delivered counts in Pew's broad unauthorized universe and gives every
#: humanitarian stock draw its own residual. A staging receipt with another
#: method predates one of the two and is refused.
ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD = (
    "national_control_minus_donor_pooled_delivered_v2"
)
#: Statuses the engine's five-year bar applies to: policyengine-us 2.2.1
#: ``gov.hhs.medicaid.eligibility.eligible_immigration_statuses`` (the
#: 2018-01-01 list) less ``CITIZEN`` and less
#: ``bar_exempt_immigration_statuses`` (8 U.S.C. 1613(b)). In 2.2.1 Medicaid
#: and Washington TANF read the clock against this bar, and Washington RCA
#: reads it as a closing window; the federal SNAP status test does not. Of
#: these, the stage assigns ``LEGAL_PERMANENT_RESIDENT`` and, through the
#: humanitarian parole draw, ``PAROLED_ONE_YEAR`` (microcosm#767); the kernel
#: never emits ``CONDITIONAL_ENTRANT``.
FIVE_YEAR_BAR_STATUSES: tuple[str, ...] = (
    "LEGAL_PERMANENT_RESIDENT",
    "CONDITIONAL_ENTRANT",
    "PAROLED_ONE_YEAR",
)
#: ``gov.hhs.medicaid.eligibility.five_year_bar_years`` (8 U.S.C. 1613(a)).
FIVE_YEAR_BAR_YEARS = 5.0
#: Arrival-to-qualified-status lags the sensitivity block reports: one to ten
#: years, then 15 and 20. Lags past five years exist (arrival in 2014 and
#: status in 2023 is a nine-year lag), so five windows were not enough
#: (microcosm#1052 review, round 4).
ADJUSTMENT_LAG_YEARS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20)
#: The person inputs this stage owns on ACS rows (and, for the entry clock, on
#: donor rows too).
ACS_LOCAL_IMMIGRATION_COLUMNS: tuple[str, ...] = (
    *US_IMMIGRATION_OUTPUT_COLUMNS,
    YEARS_SINCE_US_ENTRY_COLUMN,
)
ACS_LOCAL_IMMIGRATION_GATE_NAME = "acs_local_immigration_signal"
#: policyengine-us's ``years_since_us_entry`` default, an engine modelling
#: convention rather than a statute value. A spine carrying it on every row
#: has no entry-clock signal.
ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY = 5.0

_ACS_MIL = ACS_SOURCE_COLUMN_RENAMES["MIL"]
_AGE = "age"

#: Raw ACS person columns the CPS-named view reads. ``A_MARITL``/``A_SPOUSE``
#: are the ACS loader's CPS-named structural columns (``MAR`` with the
#: ``RELSHIPP`` spouse pairing); ``SPORDER`` keys the draws.
ACS_IMMIGRATION_SOURCE_COLUMNS: tuple[str, ...] = (
    "CIT",
    "YOEP",
    "POBP",
    "AGEP",
    "A_MARITL",
    "A_SPOUSE",
    "WAGP",
    "SEMP",
    "SSP",
    "SSIP",
    "HINS3",
    "HINS4",
    "HINS5",
    "HINS6",
    "HINS7",
    "COW",
    "SCHG",
    "ESR",
    _ACS_MIL,
    "SPORDER",
)
#: Donor person columns the entry clock reads: the ASEC citizenship and
#: entry-year codes.
_ASEC_ENTRY_SOURCE_COLUMNS: tuple[str, ...] = ("PRCITSHP", "PEINUSYR")
#: The two counts the kernel forces: non-citizens in Pew's broad
#: unauthorized universe (residual ``NONE``, DACA, the temporary protections
#: Pew retains and EAD Cuban/Haitian entrants) who work, and who attend
#: college.
_CONTROLLED_STATUSES: tuple[str, ...] = (
    "undocumented_workers",
    "undocumented_students",
)
#: The kernel's worker test: ASEC ``A_LFSR`` in the labor force at 16 or
#: older. The ASEC-by-PUF support base attaches ``A_LFSR`` only while its own
#: immigration stage runs and drops it afterwards
#: (``tools/build_us_puf_support_base.py``), so a donor built there carries
#: none and falls back to :data:`_DONOR_WORKER_COLUMNS`.
_DONOR_LABOR_FORCE_COLUMN = "A_LFSR"
_LABOR_FORCE_MIN_AGE = 16
#: Earnings-proxy donor columns for the worker test (positive wages or
#: self-employment), used when the donor carries no ``A_LFSR``: the raw ASEC
#: fields first, then harmonized inputs with the same sign. Prior-year
#: earnings and current labor-force status differ at the margins (a worker
#: laid off this year, a new entrant without last year's earnings), so the
#: worker residual is then approximate; the receipt names the test used.
_DONOR_WORKER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("WSAL_VAL", "SEMP_VAL"),
    ("employment_income_before_lsr", "self_employment_income_before_lsr"),
    ("employment_income", "self_employment_income"),
)
#: Donor columns for the kernel's student test: the raw ASEC ``A_HSCOL == 2``
#: the kernel reads, else the harmonized full-time college flag (``A_HSCOL ==
#: 2`` and full time, so narrower: the donor count can only be understated).
_DONOR_STUDENT_COLUMNS: tuple[str, ...] = ("A_HSCOL", "is_full_time_college_student")

_CITIZENSHIP_DOMAIN = (1, 2, 3, 4, 5)
_FOREIGN_BORN = (4, 5)
_MEASURED_CITIZEN = (1, 2, 3, 4)
_YES = (1,)
#: SCHG 15/16: college undergraduate, graduate or professional school.
_COLLEGE_GRADES = (15, 16)
#: ESR 4/5: Armed Forces, at work or with a job.
_ARMED_FORCES = (4, 5)
#: ACS MIL 1/2: now on, or in the past on, active duty. 3 (Reserves/National
#: Guard training only) and 4 (never served) are not active-duty service,
#: which is what CPS PEAFEVER asks about.
_ACTIVE_DUTY = (1, 2)
#: ACS COW government codes -> CPS PEIO1COW government codes.
_COW_TO_PEIO1COW: Mapping[int, int] = MappingProxyType({5: 1, 4: 2, 3: 3})
#: ACS ESR -> CPS A_LFSR. ESR 1/2 (civilian employed, at work / with a job
#: but not at work) and 3 (unemployed) keep their codes; 4/5 (Armed Forces)
#: are CPS code 0, outside the civilian universe; 6 (not in the labor force)
#: is CPS 7. A blank ESR (under 16) is CPS 0.
_ESR_TO_A_LFSR: Mapping[int, int] = MappingProxyType(
    {1: 1, 2: 2, 3: 3, 4: 0, 5: 0, 6: 7}
)
_CPS_ARMED_FORCES_OCCUPATION = 11
_CPS_COLLEGE = 2
_CPS_YES, _CPS_NO = 1, 2

#: CPS ASEC field the immigration kernel reads <- the ACS expression the view
#: builds it from.
ACS_TO_CPS_INDICATOR_MAPPING: Mapping[str, str] = MappingProxyType(
    {
        "PRCITSHP": "CIT (the same 1-5 coding)",
        "arrival year": (
            "YOEP, the exact year of entry (1938 = 1938 or earlier, 1939 = "
            "1939-1944), in place of the PEINUSYR bins"
        ),
        "PENATVTY": "POBP (Cuba 327 and Haiti 332 in both codebooks)",
        "A_AGE": "AGEP",
        "A_MARITL, A_SPOUSE": (
            "the ACS loader's CPS-named structural columns (MAR with the "
            "RELSHIPP spouse pairing)"
        ),
        "A_HSCOL": "2 (college) when SCHG is 15 or 16",
        "A_LFSR": (
            "ESR 1/2/3 (employed at work, with a job not at work, unemployed) "
            "keep their codes; ESR 4/5 (Armed Forces) and a blank ESR (under "
            "16) are 0; ESR 6 (not in the labor force) is 7"
        ),
        "WSAL_VAL": "WAGP (sign only; blank = under 15, no wages)",
        "SEMP_VAL": "SEMP (sign only; blank = under 15)",
        "MCARE": "1 when HINS3 == 1 (Medicare at interview)",
        "CAID": "1 when HINS4 == 1 (Medicaid or means-tested plan at interview)",
        "MIL": "1 when HINS5 == 1 (TRICARE or other military care at interview)",
        "CHAMPVA": "1 when HINS6 == 1 (VA health care at interview; nearest field)",
        "IHSFLG": "1 when HINS7 == 1 (Indian Health Service at interview)",
        "PEIO1COW": "1/2/3 (federal/state/local government) when COW is 5/4/3",
        "A_MJOCC": "11 (Armed Forces) when ESR is 4 or 5",
        "PEAFEVER": (
            "1 when ACS MIL (loaded as ACS_MIL) is 1 or 2, ever on active "
            "duty; 3 (Reserves/National Guard training only) and 4 are not"
        ),
        "SS_YN": "1 when SSP > 0",
        "SSI_YN": "1 when SSIP > 0",
    }
)
#: ASEC indicators with no ACS field, and the explicit value the view uses.
ACS_UNMAPPED_INDICATORS: Mapping[str, str] = MappingProxyType(
    {
        "PEN_SC1, PEN_SC2": (
            "0: ACS RETP combines every retirement source, so a federal "
            "pension cannot be told apart"
        ),
        "RESNSS1, RESNSS2": (
            "0: ACS has no reason-for-receipt field; any SSP > 0 already sets "
            "SS_YN, so no one loses an indicator"
        ),
        "SPM_CAPHOUSESUB": "0: ACS carries no housing-subsidy field",
    }
)
_DRAW_KEY_FORMAT = f"{ACS_2024_1YR_SPINE}:SERIALNO:SPORDER"
_CONTROLS_DECISION = (
    "each national control minus the donor rows' pooled delivered count, "
    "floored at 0: for the worker and student controls, donor rows in Pew's "
    "broad unauthorized universe who work or attend college; for each "
    "humanitarian stock draw, donor rows carrying its status inside its "
    "origin, arrival and SSN cohort; all on the frame's pooled weights. "
    "Donor plus ACS pools to each control whenever the ACS rows have the "
    "capacity"
)
_ENTRY_CLOCK = {
    "label": "arrival-based status-duration proxy",
    "foreign_born": (
        "time period minus the measured arrival year: the ACS YOEP year, or "
        "the ASEC PEINUSYR bin year of immigration._asec_arrival_year; "
        "clipped at 0"
    ),
    "us_born": "age (years in the US since birth)",
    "caveat": (
        "Known limitation: neither the ACS nor the ASEC donor records the "
        "date qualified status was granted, so the clock runs from arrival. "
        "8 U.S.C. 1613(a) starts the five-year clock at entry with qualified "
        "status, so for a person who adjusted status after arriving the "
        "proxy runs long and can clear the bar early (arrival in 2014, "
        "status in 2023: 10 modeled years in 2024, one qualifying year; "
        "policyengine-us#9658). That affects the policyengine-us 2.2.1 "
        "tests that read the clock: Medicaid (CHIP follows it) and "
        "Washington TANF read it against a five-year bar, so the long "
        "clock can clear the bar early; Washington RCA reads it as a "
        "window that closes (years_since_us_entry < window_years), so "
        "the long clock can wrongly exclude a recently qualified person "
        "from RCA. Under the statute the SNAP five-year "
        "bar also runs from entry with qualified status, but the pinned "
        "engine's federal SNAP status test reads the status, not this "
        "clock, so any SNAP effect is indirect. adjustment_lag_sensitivity "
        "reports the size of the affected group under each lag; the "
        "real-data check is microcosm#1093."
    ),
    "engine_readers": (
        "policyengine-us 2.2.1 reads the clock in the Medicaid "
        "immigration-status test (CHIP follows it, because CHIP excludes "
        "Medicaid-eligible children) and in Washington TANF, as a "
        "five-year bar, and in Washington RCA, as a closing window; its "
        "federal SNAP status test reads the status alone"
    ),
}


@dataclass(frozen=True)
class _ResidualTargets:
    """The worker and student counts the kernel forces, as ACS residuals.

    :class:`~microcosm.build.us_runtime.immigration.UndocumentedControls`
    holds positive national counts; a residual can be zero (the donor already
    delivers the control), which the kernel reads as "spill every ACS
    candidate". The kernel's SSN assignment reads only these two attributes
    of ``ImmigrationControls.undocumented``.
    """

    workers: float
    students: float


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def _is_any(values: pd.Series, codes: Sequence[int]) -> np.ndarray:
    """``values`` is one of ``codes``; a blank (out-of-universe) cell is not."""
    return np.isin(_numeric(values), np.asarray(codes, dtype=np.float64))


def _positive(values: pd.Series) -> np.ndarray:
    """Strictly positive amounts; a blank (under-15) amount is not."""
    return np.nan_to_num(_numeric(values), nan=0.0) > 0.0


def _cps_yes_no(values: pd.Series) -> np.ndarray:
    return np.where(_is_any(values, _YES), _CPS_YES, _CPS_NO).astype(np.int64)


def _cps_labor_force_status(esr: pd.Series) -> np.ndarray:
    """CPS ``A_LFSR`` from ACS ``ESR``; a blank (under 16) is code 0.

    Raises:
        ValueError: If any non-blank ``ESR`` (an infinity included) is outside
            the ACS codebook.
    """

    codes = _numeric(esr)
    outside = ~np.isnan(codes) & ~np.isin(
        codes, np.asarray(list(_ESR_TO_A_LFSR), dtype=np.float64)
    )
    if outside.any():
        raise ValueError(
            f"ACS ESR must be one of {sorted(_ESR_TO_A_LFSR)} or blank; "
            f"{int(outside.sum())} row(s) are outside the domain."
        )
    status = np.zeros(len(codes), dtype=np.int64)
    for esr_code, cps_code in _ESR_TO_A_LFSR.items():
        status[codes == esr_code] = cps_code
    return status


def _require_codes(values: pd.Series, *, name: str, where: str) -> np.ndarray:
    codes = _numeric(values)
    bad = ~np.isin(codes, np.asarray(_CITIZENSHIP_DOMAIN, dtype=np.float64))
    if bad.any():
        raise ValueError(
            f"{name} must be one of {_CITIZENSHIP_DOMAIN} on every {where} "
            f"person; {int(bad.sum())} row(s) are blank or outside the domain."
        )
    return codes


def _acs_cps_view(
    person: pd.DataFrame, *, time_period: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """The CPS-named kernel input for ACS persons, and their exact entry year.

    Every indicator is built from its ACS field; the three with no ACS field
    are explicit zeros, never a blank read as zero. The entry year is NaN
    where ``YOEP`` is blank (US-born).
    """

    missing = [
        column for column in ACS_IMMIGRATION_SOURCE_COLUMNS if column not in person
    ]
    if missing:
        raise ValueError(
            f"ACS local immigration requires ACS person column(s) {missing}; "
            "rebuild staging with the current ACS loader."
        )
    citizenship = _require_codes(person["CIT"], name="ACS CIT", where="ACS")
    entry_year = _numeric(person["YOEP"])
    known_entry = np.isfinite(entry_year) & (entry_year <= time_period)
    unknown = np.isin(citizenship, _FOREIGN_BORN) & ~known_entry
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} foreign-born ACS person(s) (CIT 4/5) have a "
            f"blank or post-{time_period} YOEP; the stage never invents an "
            "entry year."
        )
    for column in ("POBP", "AGEP"):
        if person[column].isna().any():
            raise ValueError(f"ACS {column} must not be blank on an ACS person.")
    cow = _numeric(person["COW"])
    government = np.zeros(len(person), dtype=np.int64)
    for acs_code, cps_code in _COW_TO_PEIO1COW.items():
        government[cow == acs_code] = cps_code
    view = pd.DataFrame(
        {
            "PRCITSHP": citizenship.astype(np.int64),
            "PENATVTY": _numeric(person["POBP"]).astype(np.int64),
            "A_AGE": _numeric(person["AGEP"]).astype(np.int64),
            # Structural loader columns; read only by the naturalization test,
            # which can move no citizen, so a blank stays the ASEC "none".
            "A_MARITL": np.nan_to_num(_numeric(person["A_MARITL"])).astype(np.int64),
            "A_SPOUSE": np.nan_to_num(_numeric(person["A_SPOUSE"])).astype(np.int64),
            "A_HSCOL": np.where(
                _is_any(person["SCHG"], _COLLEGE_GRADES), _CPS_COLLEGE, 0
            ).astype(np.int64),
            "A_LFSR": _cps_labor_force_status(person["ESR"]),
            "WSAL_VAL": np.nan_to_num(_numeric(person["WAGP"]), nan=0.0),
            "SEMP_VAL": np.nan_to_num(_numeric(person["SEMP"]), nan=0.0),
            "MCARE": _cps_yes_no(person["HINS3"]),
            "CAID": _cps_yes_no(person["HINS4"]),
            "MIL": _cps_yes_no(person["HINS5"]),
            "CHAMPVA": _cps_yes_no(person["HINS6"]),
            "IHSFLG": _cps_yes_no(person["HINS7"]),
            # No ACS field (ACS_UNMAPPED_INDICATORS): explicit non-indicators.
            "PEN_SC1": np.zeros(len(person), dtype=np.int64),
            "PEN_SC2": np.zeros(len(person), dtype=np.int64),
            "RESNSS1": np.zeros(len(person), dtype=np.int64),
            "RESNSS2": np.zeros(len(person), dtype=np.int64),
            "SPM_CAPHOUSESUB": np.zeros(len(person), dtype=np.float64),
            "SS_YN": np.where(_positive(person["SSP"]), _CPS_YES, _CPS_NO),
            "SSI_YN": np.where(_positive(person["SSIP"]), _CPS_YES, _CPS_NO),
            "PEIO1COW": government,
            "A_MJOCC": np.where(
                _is_any(person["ESR"], _ARMED_FORCES), _CPS_ARMED_FORCES_OCCUPATION, 0
            ).astype(np.int64),
            "PEAFEVER": np.where(
                _is_any(person[_ACS_MIL], _ACTIVE_DUTY), _CPS_YES, _CPS_NO
            ).astype(np.int64),
        }
    ).reset_index(drop=True)
    return view, np.where(np.isfinite(entry_year), entry_year, np.nan)


def _acs_draw_keys(household: pd.DataFrame, person: pd.DataFrame) -> pd.Series:
    """``acs_2024_1yr:SERIALNO:SPORDER`` per ACS person, unique per person."""

    if "SERIALNO" not in household or "SPORDER" not in person:
        raise ValueError(
            "ACS local immigration draws are keyed on household SERIALNO and "
            "person SPORDER; the frame lacks one of them."
        )
    serial = person["person_household_id"].map(
        household.set_index("household_id")["SERIALNO"]
    )
    order = pd.to_numeric(person["SPORDER"], errors="coerce")
    if serial.isna().any() or order.isna().any():
        raise ValueError(
            "Every ACS person needs its household SERIALNO and its SPORDER to "
            "key the immigration draws."
        )
    keys = (
        f"{ACS_2024_1YR_SPINE}:"
        + serial.astype(str)
        + ":"
        + order.astype(np.int64).astype(str)
    )
    return keys.reset_index(drop=True)


def _donor_status_masks(
    person: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """The kernel's worker and student tests on donor rows, and their columns.

    The worker test is the kernel's own (``A_LFSR`` in the labor force at 16
    or older) when the donor carries ``A_LFSR``, else the earnings proxy of
    :data:`_DONOR_WORKER_COLUMNS`.

    Raises:
        ValueError: If neither ``A_LFSR`` nor a worker pair of
            :data:`_DONOR_WORKER_COLUMNS`, or no student column of
            :data:`_DONOR_STUDENT_COLUMNS`, is present: the donor's delivered
            counts could not be measured.
    """

    labor_force = _DONOR_LABOR_FORCE_COLUMN in person
    worker_columns = next(
        (pair for pair in _DONOR_WORKER_COLUMNS if all(c in person for c in pair)),
        None,
    )
    student_column = next((c for c in _DONOR_STUDENT_COLUMNS if c in person), None)
    if (not labor_force and worker_columns is None) or student_column is None:
        raise ValueError(
            f"The donor spine needs {_DONOR_LABOR_FORCE_COLUMN} or a worker pair "
            f"from {[list(pair) for pair in _DONOR_WORKER_COLUMNS]} and a "
            f"student column from {list(_DONOR_STUDENT_COLUMNS)} to measure the "
            "undocumented worker and student counts it already delivers."
        )
    if labor_force:
        worker = _is_any(
            person[_DONOR_LABOR_FORCE_COLUMN], _CPS_LABOR_FORCE_STATUS_CODES
        ) & (_numeric(person[_AGE]) >= _LABOR_FORCE_MIN_AGE)
        worker_test = (
            f"{_DONOR_LABOR_FORCE_COLUMN} in "
            f"{list(_CPS_LABOR_FORCE_STATUS_CODES)} and {_AGE} >= "
            f"{_LABOR_FORCE_MIN_AGE}"
        )
    else:
        worker = _positive(person[worker_columns[0]]) | _positive(
            person[worker_columns[1]]
        )
        worker_test = f"{worker_columns[0]} > 0 or {worker_columns[1]} > 0"
    if student_column == "A_HSCOL":
        student = _is_any(person["A_HSCOL"], (_CPS_COLLEGE,))
        student_test = "A_HSCOL == 2"
    else:
        student = (
            person[student_column].astype(object).fillna(False).astype(bool).to_numpy()
        )
        student_test = (
            f"{student_column} (full time only: narrower than A_HSCOL == 2, so "
            "the donor student count can only be understated)"
        )
    return worker, student, {"worker": worker_test, "student": student_test}


def _donor_delivered(
    person: pd.DataFrame, weights: np.ndarray, donor: np.ndarray
) -> tuple[dict[str, float], dict[str, object] | None]:
    """Pooled weight of donor rows in each controlled status, and the columns.

    The donor rows' stored labels on the frame's (pooled) weights: what the
    donor already delivers toward each national control. The controls bind
    Pew's broad unauthorized universe, so a donor DACA, TPS or parole worker
    counts as well as a ``NONE`` one.
    """

    if not donor.any():
        return dict.fromkeys(_CONTROLLED_STATUSES, 0.0), None
    rows = person.loc[donor]
    worker, student, columns = _donor_status_masks(rows)
    unauthorized = _pew_unauthorized_rows(rows)
    donor_weights = weights[donor]
    return (
        {
            "undocumented_workers": float(donor_weights[unauthorized & worker].sum()),
            "undocumented_students": float(donor_weights[unauthorized & student].sum()),
        },
        columns,
    )


def _evidence_profile(person: pd.DataFrame, *, time_period: int, where: str):
    """The kernel's source-aware evidence for ``person``, refusals as ValueError."""

    try:
        return _source_aware_immigration_profile(person, time_period=time_period)
    except SourceRuntimeError as exc:
        raise ValueError(f"{where} immigration evidence is unusable: {exc}") from exc


def _humanitarian_delivered(
    person: pd.DataFrame,
    weights: np.ndarray,
    rows: np.ndarray,
    draws: Sequence[HumanitarianDraw],
    *,
    time_period: int,
    where: str,
) -> dict[str, float]:
    """Pooled weight of ``rows`` emitted as each draw within its cohort.

    The same contract the whole-file composition gate grades: the draw's
    status on a row whose origin, arrival and SSN evidence support it.
    """

    if not draws or not rows.any():
        return {draw.label: 0.0 for draw in draws}
    subset = person.loc[rows]
    profile = _evidence_profile(subset, time_period=time_period, where=where)
    selected = weights[rows]
    return {
        draw.label: float(
            selected[
                _humanitarian_emitted_mask(subset, draw=draw, profile=profile)
            ].sum()
        )
        for draw in draws
    }


def _humanitarian_reconciliation(
    draw: HumanitarianDraw, donor: float, assigned: float
) -> dict[str, object]:
    """One humanitarian draw: target, donor, residual, ACS and pooled mass."""

    pooled = donor + assigned
    return {
        "category": draw.category,
        "origin": draw.origin,
        "status": draw.status,
        "target": draw.target,
        "donor_delivered": donor,
        "residual": max(0.0, draw.target - donor),
        "acs_assigned": assigned,
        "pooled_total": pooled,
        "pooled_relative": (pooled / draw.target) if draw.target > 0 else None,
        "donor_excess": max(0.0, donor - draw.target),
        "source": draw.source,
    }


def _status_reconciliation(
    control: float, donor: float, capacity: float, assigned: float
) -> dict[str, float]:
    """One controlled status: control, donor, residual, ACS and pooled counts."""

    residual = max(0.0, control - donor)
    pooled = donor + assigned
    return {
        "control": control,
        "donor_delivered": donor,
        "donor_share_of_control": donor / control,
        "residual": residual,
        "acs_capacity": capacity,
        "acs_assigned": assigned,
        "pooled_total": pooled,
        "pooled_relative_error": (pooled - control) / control,
        "donor_excess": max(0.0, donor - control),
        "acs_shortfall": max(0.0, residual - capacity),
    }


def _adjustment_lag_sensitivity(
    person: pd.DataFrame, weights: np.ndarray
) -> dict[str, object]:
    """ACS mass whose five-year-bar status would flip under an adjustment lag.

    If qualified status came ``lag`` years after arrival, a person assigned a
    bar status (:data:`FIVE_YEAR_BAR_STATUSES`) whose arrival-based proxy is
    in ``[5, 5 + lag)`` would still be inside the bar. The windows nest, so
    each lag's entry is also everyone who would flip under any lag up to it;
    ``cumulative_share_of_bar_status_weight`` divides that by all bar-status
    weight. ``max_years_since_us_entry`` is the largest proxy among
    bar-status persons: only a lag past it less five flips everyone already
    past the bar. Lags are :data:`ADJUSTMENT_LAG_YEARS`. Informational, never
    graded.
    """

    subject = (
        person["immigration_status_str"]
        .astype(object)
        .isin(FIVE_YEAR_BAR_STATUSES)
        .to_numpy()
    )
    years = pd.to_numeric(
        person[YEARS_SINCE_US_ENTRY_COLUMN], errors="coerce"
    ).to_numpy(dtype=np.float64)
    past_bar = subject & (years >= FIVE_YEAR_BAR_YEARS)
    past_weight = float(weights[past_bar].sum())
    bar_weight = float(weights[subject].sum())
    flips: dict[str, dict[str, object]] = {}
    for lag in ADJUSTMENT_LAG_YEARS:
        flipped = past_bar & (years < FIVE_YEAR_BAR_YEARS + lag)
        weight = float(weights[flipped].sum())
        flips[str(lag)] = {
            "persons": int(flipped.sum()),
            "weight": weight,
            "share_of_past_bar_weight": (
                weight / past_weight if past_weight > 0 else 0.0
            ),
            "cumulative_share_of_bar_status_weight": (
                weight / bar_weight if bar_weight > 0 else 0.0
            ),
        }
    return {
        "graded": False,
        "statuses": list(FIVE_YEAR_BAR_STATUSES),
        "bar_years": FIVE_YEAR_BAR_YEARS,
        "lags_years": list(ADJUSTMENT_LAG_YEARS),
        "definition": (
            "ACS persons assigned a bar status whose years_since_us_entry is "
            "in [bar, bar + lag): past the bar on arrival, inside it if "
            "qualified status came lag years after arrival. The windows "
            "nest, so each entry also counts everyone who would flip under "
            "any shorter lag; cumulative_share_of_bar_status_weight divides "
            "it by all bar-status weight. max_years_since_us_entry is the "
            "largest clock among bar-status persons."
        ),
        "bar_status_persons": int(subject.sum()),
        "bar_status_weight": bar_weight,
        "max_years_since_us_entry": (
            float(years[subject].max()) if subject.any() else None
        ),
        "past_bar_weight": past_weight,
        "within_bar_weight": float(weights[subject & ~past_bar].sum()),
        "flips_by_lag_years": flips,
    }


def acs_local_immigration_receipt_failures(
    receipt: Mapping[str, object],
) -> list[str]:
    """Why a staging receipt's ACS targets are not the donor residuals.

    Empty when the receipt carries
    :data:`ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD`; for each controlled status,
    ``residual == max(0, control - donor_delivered)`` and
    ``pooled_total == donor_delivered + acs_assigned``; and the same two
    identities (``target`` for ``control``) for exactly the packaged
    humanitarian draws.
    """

    controls = receipt.get("controls") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(controls, Mapping)
        or controls.get("method") != ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD
    ):
        return [
            f"controls.method is not {ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD!r} "
            "(a staging run from before the current donor-residual targets)"
        ]
    statuses = controls.get("statuses")
    if not isinstance(statuses, Mapping):
        return ["controls.statuses is missing"]
    failures: list[str] = []
    for status in _CONTROLLED_STATUSES:
        entry = statuses.get(status)
        try:
            control, donor, residual, assigned, pooled = (
                float(entry[key])
                for key in (
                    "control",
                    "donor_delivered",
                    "residual",
                    "acs_assigned",
                    "pooled_total",
                )
            )
        except (KeyError, TypeError, ValueError):
            failures.append(
                f"controls.statuses.{status} lacks a numeric control, "
                "donor_delivered, residual, acs_assigned or pooled_total"
            )
            continue
        expected = max(0.0, control - donor)
        if not np.isclose(residual, expected, rtol=1e-9, atol=1e-6):
            failures.append(
                f"{status}: residual {residual:,.1f} is not max(0, control - "
                f"donor delivered) = {expected:,.1f}"
            )
        if not np.isclose(pooled, donor + assigned, rtol=1e-9, atol=1e-6):
            failures.append(
                f"{status}: pooled total {pooled:,.1f} is not donor delivered + "
                f"ACS assigned = {donor + assigned:,.1f}"
            )
    humanitarian = controls.get("humanitarian")
    if not isinstance(humanitarian, Mapping):
        return [*failures, "controls.humanitarian is missing"]
    expected_labels = {draw.label for draw in us_immigration_controls().humanitarian}
    if set(humanitarian) != expected_labels:
        failures.append(
            "controls.humanitarian draws differ from the packaged draws: "
            f"missing {sorted(expected_labels - set(humanitarian))}, "
            f"unexpected {sorted(set(humanitarian) - expected_labels)}"
        )
    for label in sorted(set(humanitarian) & expected_labels):
        entry = humanitarian[label]
        try:
            target, donor, residual, assigned, pooled = (
                float(entry[key])
                for key in (
                    "target",
                    "donor_delivered",
                    "residual",
                    "acs_assigned",
                    "pooled_total",
                )
            )
        except (KeyError, TypeError, ValueError):
            failures.append(
                f"controls.humanitarian.{label} lacks a numeric target, "
                "donor_delivered, residual, acs_assigned or pooled_total"
            )
            continue
        expected = max(0.0, target - donor)
        if not np.isclose(residual, expected, rtol=1e-9, atol=1e-6):
            failures.append(
                f"{label}: residual {residual:,.1f} is not max(0, target - "
                f"donor delivered) = {expected:,.1f}"
            )
        if not np.isclose(pooled, donor + assigned, rtol=1e-9, atol=1e-6):
            failures.append(
                f"{label}: pooled total {pooled:,.1f} is not donor delivered + "
                f"ACS assigned = {donor + assigned:,.1f}"
            )
    return failures


def _asec_entry_year(
    person: pd.DataFrame, *, time_period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Checked ASEC citizenship codes and bin arrival year (NaN if US-born)."""

    missing = [column for column in _ASEC_ENTRY_SOURCE_COLUMNS if column not in person]
    if missing:
        raise ValueError(
            f"The donor spine lacks ASEC person column(s) {missing}; the "
            "years_since_us_entry clock cannot be read."
        )
    citizenship = _require_codes(
        person["PRCITSHP"], name="ASEC PRCITSHP", where="donor"
    )
    codes = _numeric(person["PEINUSYR"])
    foreign_born = np.isin(citizenship, _FOREIGN_BORN)
    unknown = foreign_born & ~np.isin(
        codes, np.asarray(list(_ARRIVAL_YEAR_MIDPOINTS), dtype=np.float64)
    )
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} foreign-born donor person(s) (PRCITSHP 4/5) "
            "carry a PEINUSYR code outside the ASEC codebook; the stage never "
            "invents an entry year."
        )
    arrival = _asec_arrival_year(person, time_period=time_period).astype(np.float64)
    return citizenship, np.where(foreign_born, arrival, np.nan)


def _years_since_entry(
    citizenship: np.ndarray,
    entry_year: np.ndarray,
    age: np.ndarray,
    *,
    time_period: int,
) -> np.ndarray:
    """Time period minus entry for the foreign-born, age for the US-born."""

    foreign_born = np.isin(citizenship, _FOREIGN_BORN)
    unknown_age = ~foreign_born & ~np.isfinite(age)
    if unknown_age.any():
        raise ValueError(
            f"{int(unknown_age.sum())} US-born person(s) have no age; their "
            "entry clock is their age."
        )
    since_entry = time_period - np.where(foreign_born, entry_year, time_period)
    return np.maximum(np.where(foreign_born, since_entry, age), 0.0)


def _filled_labels(
    existing: pd.Series, missing: np.ndarray, assigned: np.ndarray
) -> pd.Series:
    """``existing`` with ``missing`` cells replaced, keeping a string dtype."""

    values = existing.to_numpy(dtype=object, copy=True)
    values[missing] = assigned[missing]
    filled = pd.Series(values, index=existing.index, name=existing.name)
    if isinstance(existing.dtype, pd.StringDtype):
        return filled.astype(existing.dtype)
    return filled


def _share(weights: np.ndarray, mask: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[mask].sum()) / total if total > 0 else 0.0


def _composition(person: pd.DataFrame, weights: np.ndarray) -> dict[str, object]:
    """Weighted SSN/status composition of complete immigration cells."""

    ssn = person["ssn_card_type"].astype(object)
    status = person["immigration_status_str"].astype(object)
    return {
        "person_population": float(weights.sum()),
        "non_citizen_share": _share(
            weights, (ssn.notna() & ssn.ne("CITIZEN")).to_numpy()
        ),
        "undocumented_population": float(weights[ssn.eq("NONE").to_numpy()].sum()),
        "immigration_status_population": {
            str(value): float(weights[status.eq(value).to_numpy()].sum())
            for value in sorted(status.dropna().unique())
        },
    }


def _assignment_sha256(person: pd.DataFrame) -> str:
    """Digest of every person's immigration labels and entry clock."""

    selected = pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            **{
                column: person[column].astype(object).fillna("").to_numpy(dtype=object)
                for column in US_IMMIGRATION_OUTPUT_COLUMNS
            },
            YEARS_SINCE_US_ENTRY_COLUMN: pd.to_numeric(
                person[YEARS_SINCE_US_ENTRY_COLUMN], errors="coerce"
            ).to_numpy(dtype=np.float64),
        }
    )
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def require_acs_local_immigration_donor(base: Frame, *, time_period: int) -> None:
    """Fail before the ACS transfer when the donor cannot seed this stage.

    The stage runs after hours of transfer fitting; its donor-side inputs are
    checked here first: a complete immigration surface, the ASEC citizenship
    and entry-year codes, age, the birth-country evidence the humanitarian
    cohorts read, and the worker and student columns that measure the
    controlled counts the donor already delivers.
    """

    person = base.table("person")
    missing = [
        column
        for column in (*US_IMMIGRATION_OUTPUT_COLUMNS, _AGE)
        if column not in person
    ]
    if missing:
        raise ValueError(
            f"The donor lacks person column(s) {missing} the ACS local "
            "immigration stage needs."
        )
    incomplete = [
        column
        for column in US_IMMIGRATION_OUTPUT_COLUMNS
        if person[column].isna().any()
    ]
    if incomplete:
        raise ValueError(
            f"The donor's immigration surface has missing cells in {incomplete}."
        )
    citizenship, entry_year = _asec_entry_year(person, time_period=time_period)
    _years_since_entry(
        citizenship, entry_year, _numeric(person[_AGE]), time_period=time_period
    )
    _evidence_profile(person, time_period=time_period, where="The donor's")
    _donor_status_masks(person)


def with_acs_local_immigration_inputs(
    frame: Frame, *, seed: int, time_period: int
) -> tuple[Frame, dict[str, object]]:
    """Fill missing ACS-row immigration labels and every missing entry clock.

    Args:
        frame: The local lane's multispine US frame. Its person table must
            carry complete origin tags, the donor's two immigration columns,
            the raw ACS fields of :data:`ACS_IMMIGRATION_SOURCE_COLUMNS` on ACS
            rows, the ASEC ``PRCITSHP``/``PENATVTY``/``PEINUSYR`` codes and a
            worker and a student column (:data:`_DONOR_WORKER_COLUMNS`,
            :data:`_DONOR_STUDENT_COLUMNS`) on donor rows, and ``age``; the
            household table carries ``SERIALNO``.
        seed: The build seed for the humanitarian and EAD draws.
        time_period: The dataset year, against which entry years are read.

    Returns:
        The frame with filled cells (the same object if none were missing)
        and a JSON-ready receipt, including a digest of every person's labels
        and entry clock.

    Raises:
        ValueError: If the frame is not US-schema, origin tags or a required
            column are missing, an ACS row carries only one of the two labels,
            or a foreign-born person has no entry year.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS local immigration inputs require the US schema.")
    time_period = int(time_period)
    person = frame.table("person")
    tag = spine_column("person")
    if tag not in person or person[tag].isna().any():
        raise ValueError(f"ACS local immigration requires complete origin tags: {tag}.")
    unknown = set(person[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}
    if unknown:
        raise ValueError("ACS local immigration origin tags contain an unknown spine.")
    acs = person[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    if not acs.any():
        raise ValueError("ACS local immigration found no ACS person rows.")
    absent = [
        column
        for column in (*US_IMMIGRATION_OUTPUT_COLUMNS, _AGE)
        if column not in person
    ]
    if absent:
        raise ValueError(
            f"ACS local immigration requires person column(s) {absent}; a "
            "missing donor column would reach the engine as every person a "
            "citizen."
        )
    status_missing = person["immigration_status_str"].isna().to_numpy()
    ssn_missing = person["ssn_card_type"].isna().to_numpy()
    partial = acs & (status_missing != ssn_missing)
    if partial.any():
        raise ValueError(
            f"{int(partial.sum())} ACS person row(s) carry one immigration "
            "label without the other; a partial surface would default the "
            "missing one."
        )
    fill = acs & status_missing

    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    acs_person = person.loc[acs]
    view, entry_year = _acs_cps_view(acs_person, time_period=time_period)
    national = us_immigration_controls()
    total_weight = float(weights.sum())
    acs_weight = float(weights[acs].sum())
    if not (total_weight > 0 and acs_weight > 0):
        raise ValueError("ACS local immigration needs positive ACS person weight.")
    share = acs_weight / total_weight
    # The donor was labelled against the full controls on its own mass and
    # pooled at rescaled weights; measure what it delivers on this frame and
    # target the ACS rows at the nonnegative residual (#1052 review), for the
    # worker and student controls and for every humanitarian stock (#767).
    donor_delivered, donor_columns = _donor_delivered(person, weights, ~acs)
    donor_humanitarian = _humanitarian_delivered(
        person,
        weights,
        ~acs,
        national.humanitarian,
        time_period=time_period,
        where="The donor's",
    )
    undocumented = national.undocumented
    targets = ImmigrationControls(
        undocumented=_ResidualTargets(
            workers=max(
                0.0, undocumented.workers - donor_delivered["undocumented_workers"]
            ),
            students=max(
                0.0, undocumented.students - donor_delivered["undocumented_students"]
            ),
        ),
        humanitarian=tuple(
            HumanitarianDraw(
                category=draw.category,
                origin=draw.origin,
                status=draw.status,
                target=max(0.0, draw.target - donor_humanitarian[draw.label]),
                source=draw.source,
            )
            for draw in national.humanitarian
        ),
    )
    try:
        codes, marks = _assign_ssn_card_codes(
            view,
            weights[acs],
            seed=int(seed),
            controls=targets,
            arrival_year=entry_year,
            time_period=time_period,
            draw_keys=_acs_draw_keys(frame.table("household"), acs_person),
        )
    except SourceRuntimeError as exc:
        raise ValueError(f"ACS local immigration evidence is unusable: {exc}") from exc
    derived_status = np.full(len(person), None, dtype=object)
    derived_status[acs] = _derive_immigration_status(
        view, codes, marks, time_period=time_period, arrival_year=entry_year
    )
    derived_ssn = np.full(len(person), None, dtype=object)
    derived_ssn[acs] = np.asarray(
        [_SSN_CODE_TO_NAME[code] for code in range(len(_SSN_CODE_TO_NAME))],
        dtype=object,
    )[codes]

    if YEARS_SINCE_US_ENTRY_COLUMN in person:
        stored_years = pd.to_numeric(
            person[YEARS_SINCE_US_ENTRY_COLUMN], errors="coerce"
        ).to_numpy(dtype=np.float64, copy=True)
    else:
        stored_years = np.full(len(person), np.nan)
    years_missing = np.isnan(stored_years)
    age = _numeric(person[_AGE])
    derived_years = np.full(len(person), np.nan)
    derived_years[acs] = _years_since_entry(
        view["PRCITSHP"].to_numpy(),
        entry_year,
        age[acs],
        time_period=time_period,
    )
    donor_fill = ~acs & years_missing
    if donor_fill.any():
        citizenship, donor_entry = _asec_entry_year(
            person.loc[donor_fill], time_period=time_period
        )
        derived_years[donor_fill] = _years_since_entry(
            citizenship, donor_entry, age[donor_fill], time_period=time_period
        )

    result = frame
    if fill.any() or years_missing.any():
        updated = person.copy(deep=False)
        if fill.any():
            for column, derived in (
                ("immigration_status_str", derived_status),
                ("ssn_card_type", derived_ssn),
            ):
                updated[column] = _filled_labels(person[column], fill, derived)
        stored_years[years_missing] = derived_years[years_missing]
        updated[YEARS_SINCE_US_ENTRY_COLUMN] = pd.Series(
            stored_years, index=person.index, dtype=np.float64
        )
        result = Frame(
            {
                entity: updated if entity == "person" else frame.table(entity)
                for entity in frame.entities
            },
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )

    final = result.table("person")
    acs_weights = weights[acs]
    noncitizen = view["PRCITSHP"].to_numpy() == 5
    acs_status = {
        "undocumented_workers": (view["A_AGE"].to_numpy() >= _LABOR_FORCE_MIN_AGE)
        & np.isin(view["A_LFSR"].to_numpy(), _CPS_LABOR_FORCE_STATUS_CODES),
        "undocumented_students": view["A_HSCOL"].to_numpy() == _CPS_COLLEGE,
    }
    # Capacity: ACS non-citizens inside Pew's broad universe before the EAD
    # spills (the residual pool, NONE or EAD, and the Pew-retained
    # humanitarian marks); assigned: final labels inside that universe,
    # stored ones included.
    in_universe_before_spill = noncitizen & (
        np.isin(codes, (0, 2))
        | np.isin(marks, _PEW_INCLUDED_HUMANITARIAN_STATUS_VALUES)
    )
    acs_unauthorized = _pew_unauthorized_rows(final.loc[acs])
    statuses = {
        status: _status_reconciliation(
            getattr(undocumented, status.removeprefix("undocumented_")),
            donor_delivered[status],
            float(acs_weights[in_universe_before_spill & in_status].sum()),
            float(acs_weights[acs_unauthorized & in_status].sum()),
        )
        for status, in_status in acs_status.items()
    }
    acs_humanitarian = _humanitarian_delivered(
        final,
        weights,
        acs,
        national.humanitarian,
        time_period=time_period,
        where="The ACS rows'",
    )
    humanitarian = {
        draw.label: _humanitarian_reconciliation(
            draw, donor_humanitarian[draw.label], acs_humanitarian[draw.label]
        )
        for draw in national.humanitarian
    }
    receipt: dict[str, object] = {
        "issue": ACS_LOCAL_IMMIGRATION_ISSUE,
        "seed": int(seed),
        "time_period": time_period,
        "spine": ACS_2024_1YR_SPINE,
        "acs_persons": int(acs.sum()),
        "filled_rows": {
            "immigration_labels": int(fill.sum()),
            YEARS_SINCE_US_ENTRY_COLUMN: {
                ASEC_PUF_DONOR_SPINE: int(donor_fill.sum()),
                ACS_2024_1YR_SPINE: int((acs & years_missing).sum()),
            },
        },
        "preserved_acs_label_rows": int((acs & ~fill).sum()),
        "controls": {
            "method": ACS_LOCAL_IMMIGRATION_CONTROLS_METHOD,
            "decision": _CONTROLS_DECISION,
            "donor_status_columns": donor_columns,
            "acs_person_weight_share": share,
            "national": {
                "undocumented_workers": undocumented.workers,
                "undocumented_students": undocumented.students,
                "undocumented_population_anchor": undocumented.population_anchor,
            },
            "statuses": statuses,
            "humanitarian": humanitarian,
            "sources": dict(undocumented.sources),
        },
        "indicator_mapping": dict(ACS_TO_CPS_INDICATOR_MAPPING),
        "unmapped_indicators": dict(ACS_UNMAPPED_INDICATORS),
        "draw_key": _DRAW_KEY_FORMAT,
        "entry_clock": {
            **_ENTRY_CLOCK,
            "adjustment_lag_sensitivity": _adjustment_lag_sensitivity(
                final.loc[acs], acs_weights
            ),
        },
        "acs_composition": _composition(final.loc[acs], weights[acs]),
        "assigned_sha256": _assignment_sha256(final),
    }
    return result, receipt


def _label_failures(
    spine: str, person: pd.DataFrame, columns: dict[str, object]
) -> list[str]:
    """Missing, incomplete, constant or out-of-domain labels on one spine."""

    failures: list[str] = []
    for column, domain in (
        ("ssn_card_type", SSN_CARD_TYPE_VALUES),
        ("immigration_status_str", IMMIGRATION_STATUS_VALUES),
    ):
        if column not in person:
            failures.append(
                f"{spine}: missing {column}; the engine default makes every "
                "person a citizen with a valid SSN."
            )
            continue
        present = person[column].notna().to_numpy()
        values = sorted(map(str, person.loc[present, column].unique()))
        columns[column] = {
            "missing_rows": int((~present).sum()),
            "unique_values": len(values),
        }
        if not present.all():
            failures.append(
                f"{spine}: {column} has {int((~present).sum())} missing row(s); "
                "the reviewed-null fill would make them CITIZEN."
            )
        if len(values) < 2:
            only = values[0] if values else None
            failures.append(
                f"{spine}: {column} is constant {only!r}; a spine of citizens "
                "only is the #1020 engine-default signature."
            )
        outside = sorted(set(values) - set(domain))
        if outside:
            failures.append(
                f"{spine}: {column} value(s) outside the engine enum domain: {outside}."
            )
    return failures


def _entry_clock_failures(
    spine: str, person: pd.DataFrame, columns: dict[str, object]
) -> list[str]:
    if YEARS_SINCE_US_ENTRY_COLUMN not in person:
        return [
            f"{spine}: missing {YEARS_SINCE_US_ENTRY_COLUMN}; every person "
            "would get the engine default of "
            f"{ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY:g} years."
        ]
    years = pd.to_numeric(person[YEARS_SINCE_US_ENTRY_COLUMN], errors="coerce")
    present = years.notna().to_numpy()
    observed = years.to_numpy(dtype=np.float64)[present]
    at_default = observed == ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY
    columns[YEARS_SINCE_US_ENTRY_COLUMN] = {
        "missing_rows": int((~present).sum()),
        "share_of_rows_at_engine_default": (
            float(at_default.mean()) if observed.size else 0.0
        ),
    }
    failures: list[str] = []
    if not present.all():
        failures.append(
            f"{spine}: {YEARS_SINCE_US_ENTRY_COLUMN} has "
            f"{int((~present).sum())} missing row(s)."
        )
    if observed.size and at_default.all():
        failures.append(
            f"{spine}: {YEARS_SINCE_US_ENTRY_COLUMN} is the engine default "
            f"{ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY:g} on every row."
        )
    if observed.size and (~np.isfinite(observed) | (observed < 0)).any():
        failures.append(
            f"{spine}: {YEARS_SINCE_US_ENTRY_COLUMN} has negative or "
            "non-finite value(s)."
        )
    return failures


def _measured_citizenship_failures(
    spine: str,
    person: pd.DataFrame,
    weights: np.ndarray,
    states: pd.Series | None,
    entry: dict[str, object],
) -> list[str]:
    """Grade the stage's own output: CIT agreement and the non-citizen band."""

    if "ssn_card_type" not in person:
        return []
    ssn = person["ssn_card_type"].astype(object)
    non_citizen = (ssn.notna() & ssn.ne("CITIZEN")).to_numpy()
    share = _share(weights, non_citizen)
    low, high = _NON_CITIZEN_SHARE_BAND
    entry["non_citizen_share"] = share
    entry["non_citizen_share_band"] = [low, high]
    failures: list[str] = []
    if not (low <= share <= high):
        failures.append(
            f"{spine}: non-citizen weighted share {share:.4f} outside [{low}, {high}]."
        )
    if states is not None:
        by_state: dict[str, float] = {}
        codes = pd.to_numeric(states, errors="coerce").to_numpy(dtype=np.float64)
        for state in np.unique(codes[np.isfinite(codes)]):
            in_state = codes == state
            by_state[f"{int(state):02d}"] = _share(
                weights[in_state], non_citizen[in_state]
            )
        entry["non_citizen_share_by_state"] = by_state
    if "CIT" not in person:
        failures.append(
            f"{spine}: missing CIT; measured citizenship cannot be verified."
        )
        return failures
    measured = _is_any(person["CIT"], _MEASURED_CITIZEN)
    labelled = ssn.eq("CITIZEN").to_numpy()
    disagreements = int((ssn.notna().to_numpy() & (measured != labelled)).sum())
    entry["cit_disagreements"] = disagreements
    if disagreements:
        failures.append(
            f"{spine}: {disagreements} person(s) are labelled against their "
            "measured CIT citizenship."
        )
    return failures


def acs_local_immigration_signal_gate(
    frame: Frame,
    *,
    time_period: int,
    controls: ImmigrationControls | None = None,
) -> GateResult:
    """Require complete, measured, plausible immigration inputs per origin.

    For each spine, fails when a label or the entry clock is missing or has
    missing cells, when a label is constant (the all-``CITIZEN`` signature of
    microcosm#1020) or outside the engine enum domain, or when the entry clock
    is the engine default 5 on every row. On the ACS spine, whose labels this
    stage assigns, it also fails when a label contradicts measured ``CIT``
    citizenship or the weighted non-citizen share leaves the immigration
    stage's band. Donor-spine composition is reported but not graded; the
    donor release graded it. The whole file must pass
    :func:`~microcosm.build.us_runtime.immigration.us_immigration_composition_gate`
    at ``time_period`` against ``controls`` (default: the packaged controls):
    label agreement, the source-aware cohort contracts, the non-citizen band,
    the undocumented anchor band and every humanitarian stock band.
    """

    person = frame.table("person")
    tag = spine_column("person")
    failures: list[str] = []
    by_spine: dict[str, object] = {}
    details: dict[str, object] = {"per_spine": by_spine}
    if tag not in person or person[tag].isna().any():
        failures.append(f"Missing person origin tags: {tag}.")
        return GateResult(
            name=ACS_LOCAL_IMMIGRATION_GATE_NAME,
            passed=False,
            failures=tuple(failures),
            details=details,
        )
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    household = frame.table("household")
    states = None
    if "state_fips" in household:
        states = person["person_household_id"].map(
            household.set_index("household_id")["state_fips"]
        )
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        selected = person[tag].eq(spine).to_numpy(dtype=bool)
        graded = spine == ACS_2024_1YR_SPINE
        columns: dict[str, object] = {}
        entry: dict[str, object] = {
            "rows": int(selected.sum()),
            "graded": graded,
            "columns": columns,
        }
        by_spine[spine] = entry
        if not selected.any():
            failures.append(f"{spine}: no person rows.")
            continue
        rows = person.loc[selected]
        failures += _label_failures(spine, rows, columns)
        failures += _entry_clock_failures(spine, rows, columns)
        if all(column in rows for column in US_IMMIGRATION_OUTPUT_COLUMNS):
            entry["composition"] = _composition(rows, weights[selected])
        if graded:
            failures += _measured_citizenship_failures(
                spine,
                rows,
                weights[selected],
                None if states is None else states.loc[selected],
                entry,
            )
    if set(person[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}:
        failures.append("Immigration origin tags contain an unsupported spine.")
    complete = all(
        column in person and person[column].notna().all()
        for column in US_IMMIGRATION_OUTPUT_COLUMNS
    )
    if complete:
        whole = us_immigration_composition_gate(
            frame,
            time_period=int(time_period),
            controls=us_immigration_controls() if controls is None else controls,
        )
        details["file"] = {
            "passed": whole.passed,
            "failures": list(whole.failures),
            "details": dict(whole.details),
        }
        failures += [f"file: {failure}" for failure in whole.failures]
    else:
        # The per-spine checks above name the gaps; the whole-file anchor is
        # meaningless until every person is labelled.
        details["file"] = {"passed": False, "evaluated": False}
        failures.append(
            "file: immigration composition not evaluated; label cells are missing."
        )
    return GateResult(
        name=ACS_LOCAL_IMMIGRATION_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
