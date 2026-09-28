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
- **Controls scale with the ACS mass.** The Pew worker and Higher Ed student
  controls are national counts, and the ACS rows carry only part of the
  staging mass, so each is multiplied by the ACS rows' share of weighted
  persons in the frame (equivalently, the national rate is applied to the ACS
  rows). The donor spine was labelled against the full controls on its own
  mass before pooling.
- **Draws** are the kernel's seeded blake2b draws, keyed on
  ``acs_2024_1yr:SERIALNO:SPORDER``. ``SPORDER`` alone (the ACS
  ``source_person_id``) repeats in every household.

``years_since_us_entry`` is filled on both spines. For the foreign-born it is
the time period minus the exact ACS entry year, or minus the ASEC ``PEINUSYR``
bin year the immigration stage itself uses
(:func:`~microcosm.build.us_runtime.immigration._asec_arrival_year`), clipped
at zero. For the US-born (``CIT`` / ``PRCITSHP`` 1-3) it is their age: years in
the US since birth. The engine reads it as the five-year-bar clock
(8 USC 1613), which runs from the grant of qualified status. Neither survey
measures that date, so this clock starts at entry and runs long for people who
adjusted status after arriving (policyengine-us#9658).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    ACS_SOURCE_COLUMN_RENAMES,
)
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.immigration import (
    _ARRIVAL_YEAR_MIDPOINTS,
    _NON_CITIZEN_SHARE_BAND,
    _SSN_CODE_TO_NAME,
    IMMIGRATION_STATUS_VALUES,
    SSN_CARD_TYPE_VALUES,
    US_IMMIGRATION_OUTPUT_COLUMNS,
    UndocumentedControls,
    _asec_arrival_year,
    _assign_ssn_card_codes,
    _derive_immigration_status,
    _packaged_controls,
    us_immigration_composition_gate,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_IMMIGRATION_SOURCE_COLUMNS",
    "ACS_LOCAL_IMMIGRATION_COLUMNS",
    "ACS_LOCAL_IMMIGRATION_GATE_NAME",
    "ACS_LOCAL_IMMIGRATION_ISSUE",
    "ACS_TO_CPS_INDICATOR_MAPPING",
    "ACS_UNMAPPED_INDICATORS",
    "ENGINE_DEFAULT_YEARS_SINCE_US_ENTRY",
    "YEARS_SINCE_US_ENTRY_COLUMN",
    "acs_local_immigration_signal_gate",
    "require_acs_local_immigration_donor",
    "with_acs_local_immigration_inputs",
]

ACS_LOCAL_IMMIGRATION_ISSUE = "microcosm#1020"
YEARS_SINCE_US_ENTRY_COLUMN = "years_since_us_entry"
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
    "each national control times the ACS rows' share of weighted persons in "
    "the frame (the national rate applied to the ACS rows)"
)
_ENTRY_CLOCK = {
    "foreign_born": (
        "time period minus the ACS YOEP year, or minus the ASEC PEINUSYR bin "
        "year of immigration._asec_arrival_year; clipped at 0"
    ),
    "us_born": "age (years in the US since birth)",
    "caveat": (
        "years since entry, not years in qualified status: neither survey "
        "measures the status-grant date (policyengine-us#9658)"
    ),
}


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


def _scaled_controls(
    national: UndocumentedControls, share: float
) -> UndocumentedControls:
    return UndocumentedControls(
        workers=national.workers * share,
        students=national.students * share,
        population_anchor=national.population_anchor * share,
        sources=national.sources,
    )


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
    and entry-year codes, and age.
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


def with_acs_local_immigration_inputs(
    frame: Frame, *, seed: int, time_period: int
) -> tuple[Frame, dict[str, object]]:
    """Fill missing ACS-row immigration labels and every missing entry clock.

    Args:
        frame: The local lane's multispine US frame. Its person table must
            carry complete origin tags, the donor's two immigration columns,
            the raw ACS fields of :data:`ACS_IMMIGRATION_SOURCE_COLUMNS` on ACS
            rows, the ASEC ``PRCITSHP``/``PEINUSYR`` codes on donor rows, and
            ``age``; the household table carries ``SERIALNO``.
        seed: The build seed for the EAD draws.
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
    national = _packaged_controls()
    total_weight = float(weights.sum())
    acs_weight = float(weights[acs].sum())
    if not (total_weight > 0 and acs_weight > 0):
        raise ValueError("ACS local immigration needs positive ACS person weight.")
    share = acs_weight / total_weight
    controls = _scaled_controls(national, share)
    codes = _assign_ssn_card_codes(
        view,
        weights[acs],
        seed=int(seed),
        controls=controls,
        arrival_year=entry_year,
        time_period=time_period,
        draw_keys=_acs_draw_keys(frame.table("household"), acs_person),
    )
    derived_status = np.full(len(person), None, dtype=object)
    derived_status[acs] = _derive_immigration_status(
        view, codes, time_period=time_period, arrival_year=entry_year
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
            "decision": _CONTROLS_DECISION,
            "acs_person_weight_share": share,
            "national": {
                "undocumented_workers": national.workers,
                "undocumented_students": national.students,
                "undocumented_population_anchor": national.population_anchor,
            },
            "scaled": {
                "undocumented_workers": controls.workers,
                "undocumented_students": controls.students,
            },
            "sources": dict(national.sources),
        },
        "indicator_mapping": dict(ACS_TO_CPS_INDICATOR_MAPPING),
        "unmapped_indicators": dict(ACS_UNMAPPED_INDICATORS),
        "draw_key": _DRAW_KEY_FORMAT,
        "entry_clock": dict(_ENTRY_CLOCK),
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


def acs_local_immigration_signal_gate(frame: Frame) -> GateResult:
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
    (agreement, non-citizen band and the undocumented anchor band).
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
        whole = us_immigration_composition_gate(frame)
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
