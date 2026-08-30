"""Axiom-first US immigration facts and a lossy PolicyEngine projection.

The source surveys measure citizenship, origin, arrival cohort, and a small
set of benefit/work indicators. They do *not* measure one exhaustive current
immigration status.  This module therefore generates a neutral fact surface:
current facts and status histories are separate boolean columns, and their
clocks are retained as event years and month indices. In particular, later
LPR adjustment does not erase refugee or asylum history, and parole, TPS, and DACA may
overlap.

Pre-calibration assignment never reads weights or population targets.  Exact
survey facts pass through.  Where the survey only establishes a plausible
cohort, stable blake2b draws create maximum-entropy support within that cohort.
Those 50/50 splits are pool-design choices, not prevalence estimates;
``microcosm-calibrate`` owns representation against independently sourced
government facts.

Recent ACS YOEP values identify an entry year, but the historical 1938/1939
censored or grouped codes are still retained as supplied; their interval
bounds have not yet been expanded in this prototype.

``immigration_status_str`` and ``ssn_card_type`` remain temporary,
deterministic one-way projections for PolicyEngine-US.  They are derived from
the neutral facts after assignment and never feed facts back into the model.
The enum necessarily discards overlaps; Axiom consumes the fact columns
directly through input bindings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "IMMIGRATION_STATUS_VALUES",
    "SSN_CARD_TYPE_VALUES",
    "US_IMMIGRATION_ALL_OUTPUT_COLUMNS",
    "US_IMMIGRATION_FACT_COLUMNS",
    "US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS",
    "US_IMMIGRATION_NONCONSTANT_PERSON_COLUMNS",
    "US_IMMIGRATION_OUTPUT_COLUMNS",
    "US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS",
    "US_IMMIGRATION_STAGE_NAME",
    "derive_us_immigration_facts",
    "derive_us_immigration_status_from_manifest",
    "us_immigration_composition_gate",
    "us_immigration_composition_summary",
    "us_immigration_evidence_features",
    "us_immigration_stage_spec",
    "with_us_immigration_inputs",
]

US_IMMIGRATION_STAGE_NAME = "immigration_status"

#: Lossy compatibility columns consumed by PolicyEngine-US.
US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS: tuple[str, ...] = (
    "ssn_card_type",
    "immigration_status_str",
)

#: Neutral person facts consumed by Axiom through an explicit binding map.
#: Boolean facts are intentionally nonexclusive where legally compatible.
#: ``immigration_has_lawful_status_evidence`` records suggestive survey
#: indicators only, not a legal finding or a prerequisite for status support.
#: Employment authorization means legal permission to work (including for
#: citizens), not possession of a physical Employment Authorization Document.
#: Year/count/month-index facts use zero when the corresponding event-history
#: fact is false. Parole event facts remain populated after current parole ends.
#: Month indices use
#: ``year * 12 + month - 1`` and are generated independently of any policy
#: evaluation period so Axiom can compute anniversaries at materialization.
US_IMMIGRATION_FACT_COLUMNS: tuple[str, ...] = (
    "immigration_is_us_citizen",
    "immigration_is_us_national",
    "immigration_is_us_resident",
    "immigration_is_alien",
    "immigration_has_lawful_status_evidence",
    "immigration_has_other_lawful_status",
    "immigration_is_lawful_permanent_resident",
    "immigration_was_admitted_as_refugee",
    "immigration_has_been_granted_asylum",
    "immigration_was_paroled",
    "immigration_is_currently_paroled",
    "immigration_has_deportation_or_removal_withheld",
    "immigration_was_granted_conditional_entry",
    "immigration_is_current_cuban_haitian_entrant",
    "immigration_is_current_cofa_resident",
    "immigration_is_current_tps",
    "immigration_is_current_daca",
    "immigration_is_amerasian_immigrant",
    "immigration_has_iraqi_or_afghan_special_immigrant_history",
    "immigration_has_qualifying_afghan_humanitarian_parole_history",
    "immigration_has_qualifying_ukrainian_humanitarian_parole_history",
    "immigration_is_battered_alien_or_qualifying_relative",
    "immigration_is_severe_trafficking_victim_or_qualifying_relative",
    "immigration_adjusted_to_lpr_from_refugee_status",
    "immigration_adjusted_to_lpr_from_asylum_status",
    "immigration_adjusted_to_lpr_from_withholding_status",
    "immigration_adjusted_to_lpr_from_conditional_entry_status",
    "immigration_adjusted_to_lpr_from_cuban_haitian_entrant_status",
    "immigration_adjusted_to_lpr_from_cofa_status",
    "immigration_adjusted_to_lpr_from_amerasian_status",
    "immigration_adjusted_to_lpr_from_iraqi_afghan_special_immigrant_status",
    "immigration_adjusted_to_lpr_from_qualifying_afghan_parole",
    "immigration_adjusted_to_lpr_from_qualifying_ukrainian_parole",
    "immigration_adjusted_to_lpr_from_battered_alien_status",
    "immigration_adjusted_to_lpr_from_severe_trafficking_victim_status",
    "immigration_adjusted_to_lpr_from_parole_for_at_least_one_year",
    "immigration_has_employment_authorization",
    "immigration_has_ssn_evidence",
    "immigration_has_ssn",
    "immigration_ssn_is_valid_for_employment",
    "immigration_us_entry_year",
    "immigration_us_entry_year_lower_bound",
    "immigration_us_entry_year_upper_bound",
    "immigration_entry_year_is_interval_midpoint",
    "immigration_us_entry_month_index",
    "immigration_status_start_year_is_imputed",
    "immigration_status_start_month_is_imputed",
    "immigration_lpr_status_start_year",
    "immigration_lpr_status_start_month_index",
    "immigration_refugee_admission_year",
    "immigration_refugee_admission_month_index",
    "immigration_asylum_grant_year",
    "immigration_asylum_grant_month_index",
    "immigration_parole_start_year",
    "immigration_parole_start_month_index",
    "immigration_parole_period_years",
    "immigration_withholding_start_year",
    "immigration_withholding_start_month_index",
    "immigration_conditional_entry_year",
    "immigration_conditional_entry_month_index",
    "immigration_cuban_haitian_entrant_start_year",
    "immigration_cuban_haitian_entrant_start_month_index",
    "immigration_amerasian_admission_year",
    "immigration_amerasian_admission_month_index",
    "immigration_first_qualified_status_start_year",
    "immigration_first_qualified_status_start_month_index",
    "immigration_first_qualified_status_start_year_is_imputed",
    "immigration_first_qualified_status_start_month_is_imputed",
    "immigration_first_qualified_status_started_on_or_after_1996_08_22",
)

US_IMMIGRATION_ALL_OUTPUT_COLUMNS: tuple[str, ...] = (
    *US_IMMIGRATION_FACT_COLUMNS,
    *US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS,
)

# Public stage-output contract.  Every build surface must carry the neutral
# facts as well as the temporary PolicyEngine compatibility projection.
US_IMMIGRATION_OUTPUT_COLUMNS = US_IMMIGRATION_ALL_OUTPUT_COLUMNS

#: Release gates require these person columns to carry signal (≥2 values).
US_IMMIGRATION_NONCONSTANT_PERSON_COLUMNS: tuple[str, ...] = (
    US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS
)

#: PolicyEngine-US ``SSNCardType`` enum member names.
SSN_CARD_TYPE_VALUES: tuple[str, ...] = (
    "CITIZEN",
    "NON_CITIZEN_VALID_EAD",
    "OTHER_NON_CITIZEN",
    "NONE",
)

#: PolicyEngine-US ``ImmigrationStatus`` enum member names the projection may
#: emit.  This is not the fact domain and must never be projected back.
IMMIGRATION_STATUS_VALUES: tuple[str, ...] = (
    "CITIZEN",
    "LEGAL_PERMANENT_RESIDENT",
    "CUBAN_HAITIAN_ENTRANT",
    "DACA",
    "UNDOCUMENTED",
    "PAROLED_ONE_YEAR",
    "REFUGEE",
    "ASYLEE",
    "DEPORTATION_WITHHELD",
    "CONDITIONAL_ENTRANT",
    "TPS",
)

#: Raw CPS ASEC person columns the derivation reads. All of them ship in the
#: raw Census ASEC person table (the ASEC person file carries the SPM unit
#: variables, so ``SPM_CAPHOUSESUB`` is person-grain here).
US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "PRCITSHP",
    "PEINUSYR",
    "PENATVTY",
    "A_AGE",
    "MCARE",
    "CAID",
    "IHSFLG",
    "CHAMPVA",
    "MIL",
    "PEN_SC1",
    "PEN_SC2",
    "RESNSS1",
    "RESNSS2",
    "SS_YN",
    "SSI_YN",
    "PEIO1COW",
    "A_MJOCC",
    "PEAFEVER",
    "SPM_CAPHOUSESUB",
)

_DERIVE_IMMIGRATION_STATUS_PARAMETER_KEYS = frozenset(
    {
        "seed_from_build_config",
        "time_period_from_build_config",
    }
)

#: PEINUSYR code → arrival-year midpoint (Census ASEC codebook intervals).
_ARRIVAL_YEAR_MIDPOINTS: Mapping[int, int] = {
    1: 1945,
    2: 1955,
    3: 1962,
    4: 1967,
    5: 1972,
    6: 1977,
    7: 1981,
    8: 1983,
    9: 1985,
    10: 1987,
    11: 1989,
    12: 1991,
    13: 1993,
    14: 1995,
    15: 1997,
    16: 1999,
    17: 2001,
    18: 2003,
    19: 2005,
    20: 2007,
    21: 2009,
    22: 2011,
    23: 2013,
    24: 2015,
    25: 2017,
    26: 2019,
    27: 2021,
    28: 2023,
}

#: PEINUSYR code -> inclusive arrival-year interval. Keeping the interval
#: alongside its conventional midpoint prevents an ASEC cohort such as
#: 1996-1997 from becoming false exact evidence at the August 22, 1996
#: qualified-alien cutoff.
_ARRIVAL_YEAR_BOUNDS: Mapping[int, tuple[int, int]] = {
    1: (1900, 1949),
    2: (1950, 1959),
    3: (1960, 1964),
    4: (1965, 1969),
    5: (1970, 1974),
    6: (1975, 1979),
    7: (1980, 1981),
    8: (1982, 1983),
    9: (1984, 1985),
    10: (1986, 1987),
    11: (1988, 1989),
    12: (1990, 1991),
    13: (1992, 1993),
    14: (1994, 1995),
    15: (1996, 1997),
    16: (1998, 1999),
    17: (2000, 2001),
    18: (2002, 2003),
    19: (2004, 2005),
    20: (2006, 2007),
    21: (2008, 2009),
    22: (2010, 2011),
    23: (2012, 2013),
    24: (2014, 2015),
    25: (2016, 2017),
    26: (2018, 2019),
    27: (2020, 2021),
    28: (2022, 2024),
}

#: PENATVTY codes for Cuba and Haiti; the Cuban/Haitian entrant class exists
#: for arrivals after the Refugee Education Assistance Act of 1980.
_CUBAN_HAITIAN_BIRTH_CODES = (327, 332)
_CUBAN_HAITIAN_ARRIVAL_CUTOFF = 1980

#: DACA source-supported cohort tests, 8 CFR 236.22(b)(1), (2), and (7).
#: The source age/year and arrival interval do not identify an exact DOB or
#: within-year arrival date, so these bounds retain feasible cutoff cases.
#: https://www.ecfr.gov/current/title-8/section-236.22
_DACA_LATEST_ARRIVAL_YEAR = 2007
_DACA_EARLIEST_BIRTH_YEAR = 1981
_DACA_MAX_AGE_AT_ENTRY = 16
_DACA_MIN_CURRENT_AGE = 15

#: PEINUSYR codes for 2020+ arrivals. The 2024 ASEC distinguishes 2020-2021
#: (27) from 2022-2024 (28). Program-specific rules below do not treat those
#: windows as interchangeable: U4U and CHNV cannot draw from code 27, while
#: Operation Allies Welcome can.
_RECENT_ARRIVAL_CODES = (27, 28)
_PAROLE_ASEC_ARRIVAL_CODES: Mapping[str, tuple[int, ...]] = {
    "afghanistan": _RECENT_ARRIVAL_CODES,
    "ukraine": (28,),
    "nicaragua": (28,),
    "venezuela": (28,),
}
_PAROLE_ACS_MIN_ARRIVAL_YEAR: Mapping[str, int] = {
    "afghanistan": 2021,
    "ukraine": 2022,
    "nicaragua": 2023,
    "venezuela": 2022,
}
#: Asylum grants lag arrival by filing queues plus court backlog; grantees
#: in 2022-2024 overwhelmingly arrived 2016+ (OHSS asylee flow reports).
_ASYLEE_ARRIVAL_CODES = (25, 26, 27, 28)
#: Withholding of removal follows years of proceedings: settled arrivals only.
_WITHHELD_MAX_ARRIVAL_CODE = 26

# Refugee admission/asylum are not limited to leading nationalities in an
# administrative flow table. Birthplace alone cannot establish a legal zero:
# it need not equal nationality or the relevant prior country of residence.
# 8 USC 1157(c), 1158(a)-(b), and 1101(a)(42).
# https://www.govinfo.gov/content/pkg/USCODE-2024-title8/html/USCODE-2024-title8-chap12-subchapII-partI-sec1158.htm

#: TPS per-country keys: birth codes plus the latest PEINUSYR arrival code
#: compatible with the designation's required continuous-residence date
#: (CRS RS20844 Table 1). Legacy designations bind hard (El Salvador
#: 2001-02-13 -> code 17; Honduras/Nicaragua 1998-12-30 -> code 16; Nepal
#: 2015-06-24 -> code 24); 2021+ designations reach the top code, where the
#: 2022-2024 bin unavoidably includes some post-cutoff arrivals.
#: Ukraine and Afghanistan TPS registrants are carried by the parole draw
#: (same U4U/OAW people); Haiti TPS is carried by CUBAN_HAITIAN_ENTRANT.
_TPS_ORIGIN_CODES: Mapping[str, tuple[tuple[int, ...], int]] = {
    "venezuela": ((373,), 28),
    "el_salvador": ((312,), 17),
    "honduras": ((314,), 16),
    "nicaragua": ((315,), 16),
    "nepal": ((229,), 24),
    # Burma, Syria, Yemen, Lebanon, Cameroon, Ethiopia, Somalia, Sudan
    # (South Sudan shares 451 in the CPS codebook).
    "other_designated": ((205, 239, 248, 224, 407, 416, 448, 451), 28),
}

#: Recent ACS YOEP preserves an exact year, unlike the ASEC arrival bins. Use that
#: extra information for continuous-residence compatibility rather than
#: deliberately coarsening YOEP back to PEINUSYR. The per-country cutoffs are
#: calendar-year approximations of the designation dates; ASEC retains the
#: published coarse-bin contract above.
_TPS_ACS_MAX_ARRIVAL_YEAR_BY_BIRTH: Mapping[int, int] = {
    373: 2023,  # Venezuela (2021 and 2023 designations)
    312: 2001,  # El Salvador
    314: 1998,  # Honduras
    315: 1998,  # Nicaragua
    229: 2015,  # Nepal
    205: 2024,  # Burma
    239: 2024,  # Syria
    248: 2024,  # Yemen
    224: 2024,  # Lebanon
    407: 2023,  # Cameroon
    416: 2024,  # Ethiopia
    448: 2024,  # Somalia
    451: 2023,  # Sudan / South Sudan (shared survey code)
}

# Neutral support generation includes statuses that the former exclusive PE
# draw intentionally collapsed into another enum.  These tables define only
# source-compatible cohorts; they contain no population totals.
_PAROLE_SUPPORT_ASEC_ARRIVAL_CODES_BY_BIRTH: Mapping[int, tuple[int, ...]] = {
    200: _RECENT_ARRIVAL_CODES,  # Afghanistan / Operation Allies Welcome
    164: (28,),  # Ukraine / Uniting for Ukraine
    315: (28,),  # Nicaragua / CHNV
    373: (28,),  # Venezuela / CHNV
    327: (28,),  # Cuba / CHNV
    332: (28,),  # Haiti / CHNV
}
_PAROLE_SUPPORT_ACS_MIN_ARRIVAL_YEAR_BY_BIRTH: Mapping[int, int] = {
    200: 2021,
    164: 2022,
    315: 2023,
    373: 2022,
    327: 2023,
    332: 2023,
}

# Primary-cohort parole-into-the-US windows, at the retained month precision.
# The boundary months contain an admissible day (July 31 / February 24);
# drawing that month does not assert an observed or exact day. These are
# historical grant-support constraints, not handwritten benefit eligibility.
# Afghan: PL117-43 §2502(a)(1)(A), amended by PL117-328 divM §1501.
# Ukrainian: PL117-128 §401(a)(1)(A), amended by PL118-50 divB §301.
# https://www.govinfo.gov/content/pkg/USCODE-2024-title8/html/USCODE-2024-title8-chap12-subchapI-sec1101.htm
_AFGHAN_PRIMARY_PAROLE_FIRST_MONTH = 2021 * 12 + 6  # July 31, 2021
_AFGHAN_PRIMARY_PAROLE_LAST_MONTH = 2023 * 12 + 8  # September 30, 2023
_UKRAINIAN_PRIMARY_PAROLE_FIRST_MONTH = 2022 * 12 + 1  # February 24, 2022
_UKRAINIAN_PRIMARY_PAROLE_LAST_MONTH = 2024 * 12 + 8  # September 30, 2024
_TPS_SUPPORT_ASEC_MAX_ARRIVAL_CODE_BY_BIRTH: Mapping[int, int] = {
    **{
        birth_code: max_arrival_code
        for codes, max_arrival_code in _TPS_ORIGIN_CODES.values()
        for birth_code in codes
    },
    164: 28,  # Ukraine; code 28 contains the 2022 designation cohort.
    200: 28,  # Afghanistan; code 28 contains the 2022 designation cohort.
    332: 28,  # Haiti; preserved separately from CHE on the fact surface.
}
_TPS_SUPPORT_ACS_MAX_ARRIVAL_YEAR_BY_BIRTH: Mapping[int, int] = {
    **_TPS_ACS_MAX_ARRIVAL_YEAR_BY_BIRTH,
    164: 2022,
    200: 2022,
    332: 2022,
}

# CPS/ACS country-code support for the Federated States of Micronesia and the
# Marshall Islands.  The shared survey codebook does not expose Palau as a
# distinct country value on this surface, so it must not be fabricated.
_COFA_BIRTH_CODES = (511, 512)

# A support split is deliberately balanced because the source survey provides
# no prevalence information within a compatible cohort.  Calibration, not
# this constant, determines the represented population.
_MAXIMUM_ENTROPY_SUPPORT_THRESHOLD = 0.5


@dataclass(frozen=True)
class _ImmigrationEvidenceProfile:
    """Canonical immigration evidence shared by ASEC and ACS rows."""

    source_is_acs: np.ndarray
    is_citizen: np.ndarray
    birth_country: np.ndarray
    arrival_code: np.ndarray
    arrival_year: np.ndarray
    arrival_year_lower_bound: np.ndarray
    arrival_year_upper_bound: np.ndarray
    age: np.ndarray
    age_reference_year: np.ndarray
    age_at_entry: np.ndarray
    tps_acs_max_arrival_year: np.ndarray


def us_immigration_stage_spec() -> SourceStageSpec:
    """Load the packaged ``immigration_status`` source-stage manifest entry."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_IMMIGRATION_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_IMMIGRATION_STAGE_NAME!r} stage."
        )
    return stage_map[US_IMMIGRATION_STAGE_NAME]


def derive_us_immigration_status_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Derive neutral immigration facts, then the one-way PE projection.

    The operation accepts only the build seed and target period. Population
    targets and materialized weights have no route into person assignment.
    """

    if operation.kind != "derive_immigration_status":
        raise SourceRuntimeError(
            "US immigration derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US immigration derivation requires the person table to be read first."
        )
    params = operation.parameters
    unexpected = sorted(set(params) - _DERIVE_IMMIGRATION_STATUS_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            f"US immigration derivation received unsupported parameter(s): "
            f"{unexpected}."
        )
    if params.get("seed_from_build_config") is not True:
        raise SourceRuntimeError(
            "US immigration derivation requires seed_from_build_config=true."
        )
    if params.get("time_period_from_build_config") is not True:
        raise SourceRuntimeError(
            "US immigration derivation requires time_period_from_build_config=true."
        )
    if context.config.target_year is None:
        raise SourceRuntimeError(
            "US immigration derivation requires a target year in the runtime config."
        )
    return derive_us_immigration_facts(
        frame,
        seed=int(context.config.seed),
        time_period=int(context.config.target_year),
    )


def _integer_column(person: pd.DataFrame, column: str) -> np.ndarray:
    return (
        pd.to_numeric(person[column], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )


def _float_column(person: pd.DataFrame, column: str) -> np.ndarray:
    return (
        pd.to_numeric(person[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _numeric_evidence_column(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person.columns:
        return np.full(len(person), np.nan, dtype=np.float64)
    return pd.to_numeric(person[column], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )


def _source_aware_immigration_profile(
    person: pd.DataFrame,
    *,
    time_period: int,
) -> _ImmigrationEvidenceProfile:
    """Normalize ASEC PRCITSHP/PENATVTY/PEINUSYR and ACS CIT/POBP/YOEP.

    A stacked recipient carries the union of both raw survey schemas, with the
    non-owning source null on each row. Exactly one citizenship source must be
    present. Relevant POBP country codes share the Census country codebook with
    PENATVTY, so the normalized origin is lossless. Recent ACS YOEP values
    retain their entry year, while ASEC uses the documented interval midpoint.
    Historical ACS 1938/1939 censored or grouped codes are not yet expanded
    into their source interval bounds and remain a prototype limitation.
    """

    if isinstance(time_period, bool) or int(time_period) != time_period:
        raise ValueError(
            f"US immigration time_period must be an integer, got {time_period!r}."
        )
    time_period = int(time_period)
    asec_citizenship = _numeric_evidence_column(person, "PRCITSHP")
    acs_citizenship = _numeric_evidence_column(person, "CIT")
    has_asec = np.isfinite(asec_citizenship)
    has_acs = np.isfinite(acs_citizenship)
    ambiguous = has_asec & has_acs
    missing = ~has_asec & ~has_acs
    if ambiguous.any() or missing.any():
        raise SourceRuntimeError(
            "US immigration evidence requires exactly one row-level citizenship "
            "source (ASEC PRCITSHP or ACS CIT); "
            f"ambiguous_rows={int(ambiguous.sum())}, missing_rows={int(missing.sum())}."
        )

    for label, values, present in (
        ("PRCITSHP", asec_citizenship, has_asec),
        ("CIT", acs_citizenship, has_acs),
    ):
        invalid = present & (
            ~np.equal(values, np.rint(values)) | ~np.isin(values, [1, 2, 3, 4, 5])
        )
        if invalid.any():
            bad = sorted(set(values[invalid].tolist()))[:5]
            raise SourceRuntimeError(
                f"{label} carries value(s) outside the Census domain 1..5: {bad}."
            )

    citizenship = np.where(has_acs, acs_citizenship, asec_citizenship).astype(np.int64)
    is_citizen = np.isin(citizenship, [1, 2, 3, 4])

    asec_birth = _numeric_evidence_column(person, "PENATVTY")
    acs_birth = _numeric_evidence_column(person, "POBP")
    birth = np.where(has_acs, acs_birth, asec_birth)
    invalid_birth = ~np.isfinite(birth) | ~np.equal(birth, np.rint(birth)) | (birth < 1)
    if invalid_birth.any():
        raise SourceRuntimeError(
            "US immigration evidence requires a positive integral PENATVTY/POBP "
            f"country code on every row; invalid_rows={int(invalid_birth.sum())}."
        )
    birth_country = birth.astype(np.int64)

    arrival_code_values = _numeric_evidence_column(person, "PEINUSYR")
    invalid_asec_arrival = has_asec & (
        ~np.isfinite(arrival_code_values)
        | ~np.equal(arrival_code_values, np.rint(arrival_code_values))
        | (arrival_code_values < 0)
        | (arrival_code_values > max(_ARRIVAL_YEAR_MIDPOINTS))
    )
    if invalid_asec_arrival.any():
        bad = sorted(set(arrival_code_values[invalid_asec_arrival].tolist()))[:5]
        raise SourceRuntimeError(
            "PEINUSYR carries value(s) outside the 2024 ASEC domain "
            f"0..{max(_ARRIVAL_YEAR_MIDPOINTS)}: {bad}."
        )
    arrival_code = np.zeros(len(person), dtype=np.int64)
    arrival_code[has_asec] = arrival_code_values[has_asec].astype(np.int64)
    arrival_year = np.full(len(person), time_period, dtype=np.int64)
    arrival_year_lower_bound = np.full(len(person), time_period, dtype=np.int64)
    arrival_year_upper_bound = np.full(len(person), time_period, dtype=np.int64)
    for code, midpoint in _ARRIVAL_YEAR_MIDPOINTS.items():
        cohort = has_asec & (arrival_code == code)
        lower, upper = _ARRIVAL_YEAR_BOUNDS[code]
        arrival_year[cohort] = min(midpoint, time_period)
        arrival_year_lower_bound[cohort] = min(lower, time_period)
        arrival_year_upper_bound[cohort] = min(upper, time_period)

    acs_arrival = _numeric_evidence_column(person, "YOEP")
    acs_foreign_born = has_acs & np.isin(citizenship, [4, 5])
    invalid_acs_arrival = acs_foreign_born & (
        ~np.isfinite(acs_arrival)
        | ~np.equal(acs_arrival, np.rint(acs_arrival))
        | (acs_arrival < 1900)
        | (acs_arrival > time_period)
    )
    if invalid_acs_arrival.any():
        raise SourceRuntimeError(
            "ACS YOEP must be an integral 1900..time_period year for every "
            f"foreign-born CIT=4/5 row; invalid_rows={int(invalid_acs_arrival.sum())}."
        )
    observed_acs_arrival = has_acs & np.isfinite(acs_arrival)
    arrival_year[observed_acs_arrival] = acs_arrival[observed_acs_arrival].astype(
        np.int64
    )
    arrival_year_lower_bound[observed_acs_arrival] = arrival_year[observed_acs_arrival]
    arrival_year_upper_bound[observed_acs_arrival] = arrival_year[observed_acs_arrival]

    age_values = _numeric_evidence_column(person, "A_AGE")
    if "age" in person.columns:
        mapped_age = _numeric_evidence_column(person, "age")
        age_values = np.where(np.isfinite(age_values), age_values, mapped_age)
    invalid_age = (
        ~np.isfinite(age_values)
        | ~np.equal(age_values, np.rint(age_values))
        | (age_values < 0)
    )
    if invalid_age.any():
        raise SourceRuntimeError(
            "US immigration evidence requires a finite non-negative integral "
            f"A_AGE/age on every row; invalid_rows={int(invalid_age.sum())}."
        )
    age = age_values.astype(np.int64)
    source_year = _numeric_evidence_column(person, "source_year")
    observed_source_year = np.isfinite(source_year)
    invalid_source_year = observed_source_year & (
        ~np.equal(source_year, np.rint(source_year)) | (source_year < 1900)
    )
    if invalid_source_year.any():
        raise SourceRuntimeError(
            "US immigration source_year must be an integral observation year "
            f"from 1900 onward; invalid_rows={int(invalid_source_year.sum())}."
        )
    # Pooled sources retain ages at their observation year. A later build
    # target must not change the possible birth years implied by those ages.
    age_reference_year = np.where(
        observed_source_year, source_year, time_period
    ).astype(np.int64)
    age_at_entry = np.maximum(0, age - (age_reference_year - arrival_year))
    tps_acs_max_arrival_year = np.full(len(person), -1, dtype=np.int64)
    for birth_code, cutoff in _TPS_ACS_MAX_ARRIVAL_YEAR_BY_BIRTH.items():
        tps_acs_max_arrival_year[birth_country == birth_code] = cutoff
    return _ImmigrationEvidenceProfile(
        source_is_acs=has_acs,
        is_citizen=is_citizen,
        birth_country=birth_country,
        arrival_code=arrival_code,
        arrival_year=arrival_year,
        arrival_year_lower_bound=arrival_year_lower_bound,
        arrival_year_upper_bound=arrival_year_upper_bound,
        age=age,
        age_reference_year=age_reference_year,
        age_at_entry=age_at_entry,
        tps_acs_max_arrival_year=tps_acs_max_arrival_year,
    )


def us_immigration_evidence_features(
    frame: Frame,
    *,
    time_period: int = 2024,
) -> pd.DataFrame:
    """Return finite, source-harmonized immigration predictors per person."""

    person = frame.table(frame.schema.person_entity)
    profile = _source_aware_immigration_profile(person, time_period=time_period)
    return pd.DataFrame(
        {
            "is_us_citizen": profile.is_citizen.astype(np.float64),
            "birth_country_code": profile.birth_country.astype(np.float64),
            "arrival_year": profile.arrival_year.astype(np.float64),
        },
        index=person.index,
    )


def _stable_person_draws(person: pd.DataFrame, *, seed: int, salt: str) -> np.ndarray:
    """Deterministic uniform draws keyed by stable person identity.

    Support-channel clones carry their source person's ``source_year`` /
    ``source_household_id`` / ``source_person_id``, so keying on those gives
    every clone of one source person the same draw; frames without full source
    ids fall back to the legacy source pair or ``person_id``.
    """

    def complete(columns: tuple[str, ...]) -> bool:
        return set(columns).issubset(person.columns) and bool(
            person.loc[:, list(columns)].notna().to_numpy(dtype=bool).all()
        )

    full_lineage = ("source_year", "source_household_id", "source_person_id")
    legacy_lineage = ("source_year", "source_person_id")
    if complete(full_lineage):
        keys = (
            person["source_year"].astype(str)
            + ":"
            + person["source_household_id"].astype(str)
            + ":"
            + person["source_person_id"].astype(str)
        )
    elif complete(legacy_lineage):
        keys = (
            person["source_year"].astype(str)
            + ":"
            + person["source_person_id"].astype(str)
        )
    elif complete(("person_id",)):
        keys = person["person_id"].astype(str)
    else:
        raise SourceRuntimeError(
            "US immigration deterministic draws require one complete stable "
            "person-lineage alternative: source_year/source_household_id/"
            "source_person_id, source_year/source_person_id, or person_id."
        )
    denominator = float(2**64)
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{salt}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ),
        dtype=np.float64,
        count=len(keys),
    )


def _source_code_is(
    person: pd.DataFrame,
    column: str,
    values: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Return a false-on-missing code-membership mask."""

    return np.isin(_numeric_evidence_column(person, column), values)


def _positive_source_value(person: pd.DataFrame, column: str) -> np.ndarray:
    """Return a false-on-missing positive-value mask."""

    values = _numeric_evidence_column(person, column)
    return np.isfinite(values) & (values > 0)


def _lawful_status_evidence(
    person: pd.DataFrame,
    profile: _ImmigrationEvidenceProfile,
) -> np.ndarray:
    """Suggestive survey indicators, not a finding of lawful status.

    Benefit receipt, covered work, and household assistance do not establish
    an individual's status. Their presence or absence must not force or
    exclude any status/history draw. For example, emergency Medicaid can
    cover people without lawful status (42 USC 1396b(v)(2)); pre-1982 arrival
    alone does not establish an IRCA adjustment (8 USC 1255a(a)).
    """

    federal_pension = _source_code_is(person, "PEN_SC1", [3]) | _source_code_is(
        person, "PEN_SC2", [3]
    )
    social_security_disability = _source_code_is(
        person, "RESNSS1", [2]
    ) | _source_code_is(person, "RESNSS2", [2])
    government_employee = _source_code_is(
        person, "PEIO1COW", [1, 2, 3]
    ) | _source_code_is(person, "A_MJOCC", [11])
    military_connected = _source_code_is(person, "PEAFEVER", [1]) | _source_code_is(
        person, "A_MJOCC", [11]
    )
    evidence = (
        _source_code_is(person, "MCARE", [1])
        | _source_code_is(person, "CAID", [1])
        | _source_code_is(person, "IHSFLG", [1])
        | _source_code_is(person, "CHAMPVA", [1])
        | _source_code_is(person, "MIL", [1])
        | federal_pension
        | social_security_disability
        | government_employee
        | military_connected
        | _source_code_is(person, "SS_YN", [1])
        | _source_code_is(person, "SSI_YN", [1])
        | _positive_source_value(person, "SPM_CAPHOUSESUB")
        # ACS-native analogues for the two cash-benefit indicators.
        | _positive_source_value(person, "SSP")
        | _positive_source_value(person, "SSIP")
    )
    return np.asarray(evidence & ~profile.is_citizen, dtype=bool)


def _maximum_entropy_support(
    person: pd.DataFrame,
    candidates: np.ndarray,
    *,
    seed: int,
    salt: str,
) -> np.ndarray:
    """Stable balanced support inside a survey-compatible candidate cohort.

    A large cohort uses the direct Bernoulli(0.5) maximum-entropy draw. When
    two or more distinct source people form a thin cohort and chance puts all
    of them on one side, the extreme draw is flipped as a deterministic
    support floor. Clone rows share a draw and therefore remain consistent.
    """

    candidates = np.asarray(candidates, dtype=bool)
    draws = _stable_person_draws(person, seed=seed, salt=salt)
    selected = candidates & (draws < _MAXIMUM_ENTROPY_SUPPORT_THRESHOLD)
    candidate_draws = np.unique(draws[candidates])
    if candidate_draws.size >= 2:
        if not selected[candidates].any():
            selected |= candidates & (draws == candidate_draws.min())
        elif selected[candidates].all():
            selected &= ~(candidates & (draws == candidate_draws.max()))
    return np.asarray(selected, dtype=bool)


def _parole_support_candidates(
    profile: _ImmigrationEvidenceProfile,
) -> np.ndarray:
    candidates = np.zeros(len(profile.is_citizen), dtype=bool)
    for (
        birth_code,
        arrival_codes,
    ) in _PAROLE_SUPPORT_ASEC_ARRIVAL_CODES_BY_BIRTH.items():
        same_origin = profile.birth_country == birth_code
        candidates |= same_origin & (
            ((~profile.source_is_acs) & np.isin(profile.arrival_code, arrival_codes))
            | (
                profile.source_is_acs
                & (
                    profile.arrival_year
                    >= _PAROLE_SUPPORT_ACS_MIN_ARRIVAL_YEAR_BY_BIRTH[birth_code]
                )
            )
        )
    return np.asarray(candidates & ~profile.is_citizen, dtype=bool)


def _tps_support_candidates(profile: _ImmigrationEvidenceProfile) -> np.ndarray:
    candidates = np.zeros(len(profile.is_citizen), dtype=bool)
    for (
        birth_code,
        max_arrival_code,
    ) in _TPS_SUPPORT_ASEC_MAX_ARRIVAL_CODE_BY_BIRTH.items():
        same_origin = profile.birth_country == birth_code
        candidates |= same_origin & (
            (
                (~profile.source_is_acs)
                & (profile.arrival_code >= 1)
                & (profile.arrival_code <= max_arrival_code)
            )
            | (
                profile.source_is_acs
                & (
                    profile.arrival_year
                    <= _TPS_SUPPORT_ACS_MAX_ARRIVAL_YEAR_BY_BIRTH[birth_code]
                )
            )
        )
    return np.asarray(candidates & ~profile.is_citizen, dtype=bool)


def _stable_imputed_year(
    person: pd.DataFrame,
    flag: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    *,
    seed: int,
    salt: str,
) -> np.ndarray:
    """Draw a clone-stable integral year from each feasible interval."""

    lower = np.asarray(lower_bound, dtype=np.int64)
    upper = np.maximum(lower, np.asarray(upper_bound, dtype=np.int64))
    span = upper - lower + 1
    draws = _stable_person_draws(person, seed=seed, salt=salt)
    sampled = lower + np.minimum((draws * span).astype(np.int64), span - 1)
    return np.where(flag, sampled, 0).astype(np.int64)


def _stable_imputed_month_index(
    person: pd.DataFrame,
    flag: np.ndarray,
    year: np.ndarray,
    *,
    seed: int,
    salt: str,
    lower_bound: np.ndarray | None = None,
    upper_bound: np.ndarray | None = None,
) -> np.ndarray:
    """Draw a clone-stable month inside each active event year.

    Census exposes entry years or year intervals, not event month/day. The
    imputed month is a neutral evidence completion, not an eligibility result;
    Axiom compares it with the requested evaluation period to resolve five-
    and seven-year anniversaries. Inactive events retain the integer sentinel
    zero used by the rest of this fact surface.
    """

    active = np.asarray(flag, dtype=bool)
    event_year = np.asarray(year, dtype=np.int64)
    lower = event_year * 12
    upper = lower + 11
    if lower_bound is not None:
        lower = np.maximum(lower, lower_bound)
    if upper_bound is not None:
        upper = np.minimum(upper, upper_bound)
    if np.any(active & (lower > upper)):
        raise SourceRuntimeError(
            "US immigration event month has no source-compatible interval."
        )
    span = np.maximum(1, upper - lower + 1)
    draws = _stable_person_draws(person, seed=seed, salt=salt)
    month_index = lower + np.minimum((draws * span).astype(np.int64), span - 1)
    return np.where(active, month_index, 0).astype(np.int64)


def _started_on_or_after_august_22_1996(
    person: pd.DataFrame,
    flag: np.ndarray,
    start_year: np.ndarray,
    start_month_index: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Resolve the within-August-1996 cutoff without inventing exactness.

    The event month comes from the retained month-index imputation. Only rows
    assigned August 1996 need a second stable day draw: August 22 through 31
    covers 10 of that month's 31 days. The adjacent imputation flags prevent
    this legal input from masquerading as observed evidence.
    """

    result = np.asarray(flag & (start_year > 1996), dtype=bool)
    start_month = np.asarray(start_month_index, dtype=np.int64) % 12 + 1
    within_1996 = flag & (start_year == 1996)
    result[within_1996 & (start_month > 8)] = True
    within_august_1996 = within_1996 & (start_month == 8)
    draws = _stable_person_draws(
        person,
        seed=seed,
        salt="immigration:first_qualified_status_1996_cutoff_day",
    )
    result[within_august_1996] = draws[within_august_1996] < (10.0 / 31.0)
    return result


def _ssn_evidence(person: pd.DataFrame) -> np.ndarray:
    """Direct survey evidence that a person already has an SSN."""

    return np.asarray(
        _source_code_is(person, "SS_YN", [1])
        | _source_code_is(person, "SSI_YN", [1])
        | _positive_source_value(person, "SSP")
        | _positive_source_value(person, "SSIP"),
        dtype=bool,
    )


def _legacy_immigration_projection(
    facts: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Project nonexclusive facts into the two legacy PE-US enum inputs.

    Assignment order is documented precedence, not a claim that a person has
    only one status.  Current LPR wins over status history; CHE wins over
    other humanitarian history; parole wins over TPS, which wins over DACA.
    COFA and other-lawful-status assignments map to LPR because PE has no
    matching enum value. That substitution is not eligibility-preserving
    and is not a faithful PE migration. No downstream code may use this
    projection to rebuild facts.
    """

    citizen = facts["immigration_is_us_citizen"]
    lpr = facts["immigration_is_lawful_permanent_resident"]
    refugee = facts["immigration_was_admitted_as_refugee"]
    asylum = facts["immigration_has_been_granted_asylum"]
    parole = facts["immigration_is_currently_paroled"]
    withholding = facts["immigration_has_deportation_or_removal_withheld"]
    conditional = facts["immigration_was_granted_conditional_entry"]
    che = facts["immigration_is_current_cuban_haitian_entrant"]
    cofa = facts["immigration_is_current_cofa_resident"]
    tps = facts["immigration_is_current_tps"]
    daca = facts["immigration_is_current_daca"]
    other_lawful = facts["immigration_has_other_lawful_status"]

    has_ssn = facts["immigration_has_ssn"]
    valid_for_employment = facts["immigration_ssn_is_valid_for_employment"]
    ssn = np.full(len(citizen), "NONE", dtype="U24")
    ssn[has_ssn] = "OTHER_NON_CITIZEN"
    ssn[valid_for_employment] = "NON_CITIZEN_VALID_EAD"
    ssn[citizen] = "CITIZEN"

    status = np.full(len(citizen), "UNDOCUMENTED", dtype="U32")
    status[other_lawful | cofa] = "LEGAL_PERMANENT_RESIDENT"
    status[daca] = "DACA"
    status[tps] = "TPS"
    status[parole] = "PAROLED_ONE_YEAR"
    status[conditional] = "CONDITIONAL_ENTRANT"
    status[withholding] = "DEPORTATION_WITHHELD"
    status[refugee] = "REFUGEE"
    status[asylum] = "ASYLEE"
    status[che] = "CUBAN_HAITIAN_ENTRANT"
    status[lpr] = "LEGAL_PERMANENT_RESIDENT"
    status[citizen] = "CITIZEN"
    return ssn, status


def derive_us_immigration_facts(
    person: pd.DataFrame,
    *,
    seed: int,
    time_period: int,
) -> pd.DataFrame:
    """Generate a stable, weight- and target-invariant immigration trajectory.

    Survey-compatible cohort membership defines support.  Stable balanced
    draws resolve facts that the survey does not observe, while independent
    salts preserve legally possible overlaps.  The returned frame retains all
    source columns and appends the neutral facts plus the PE bridge.
    """

    result = person.copy(deep=True)
    profile = _source_aware_immigration_profile(result, time_period=time_period)
    alien = ~profile.is_citizen
    entry_year = np.where(alien, profile.arrival_year, 0).astype(np.int64)
    years_in_us = np.where(alien, np.maximum(0, time_period - entry_year), 0)
    lawful_evidence = _lawful_status_evidence(result, profile)

    refugee_candidate = alien & (entry_year >= 1980)
    asylum_candidate = alien & (entry_year >= 1980)
    refugee = _maximum_entropy_support(
        result,
        refugee_candidate,
        seed=seed,
        salt="immigration:refugee_admission_history_support",
    )
    asylum = _maximum_entropy_support(
        result,
        asylum_candidate,
        seed=seed,
        salt="immigration:asylum_grant_history_support",
    )

    # Retain the grant as history independently of whether its temporary
    # authorization is still current. Reuse the established draw salt so a
    # compatible person's grant does not change merely because it is renamed.
    parole_history = _maximum_entropy_support(
        result,
        _parole_support_candidates(profile),
        seed=seed,
        salt="immigration:current_parole_support",
    )
    parole = parole_history & (years_in_us <= 2)
    tps = _maximum_entropy_support(
        result,
        _tps_support_candidates(profile),
        seed=seed,
        salt="immigration:current_tps_support",
    )
    daca = _maximum_entropy_support(
        result,
        alien
        & _daca_statutory_cohort(
            profile.arrival_year,
            profile.age_at_entry,
            profile.age,
            age_reference_year=profile.age_reference_year,
        ),
        seed=seed,
        salt="immigration:current_daca_support",
    )
    che = _maximum_entropy_support(
        result,
        alien
        & np.isin(profile.birth_country, _CUBAN_HAITIAN_BIRTH_CODES)
        & (entry_year >= _CUBAN_HAITIAN_ARRIVAL_CUTOFF),
        seed=seed,
        salt="immigration:current_cuban_haitian_entrant_support",
    )
    cofa = _maximum_entropy_support(
        result,
        alien & np.isin(profile.birth_country, _COFA_BIRTH_CODES),
        seed=seed,
        salt="immigration:current_cofa_resident_support",
    )

    conditional = _maximum_entropy_support(
        result,
        alien & (entry_year > 0) & (entry_year < 1980),
        seed=seed,
        salt="immigration:conditional_entry_history_support",
    )
    withholding = _maximum_entropy_support(
        result,
        alien & (years_in_us >= 2) & ~refugee & ~asylum,
        seed=seed,
        salt="immigration:withholding_support",
    )
    # These histories are not survey fields. Broad official-program cohorts
    # define support; independent balanced draws keep both outcomes available
    # for calibration without imposing administrative totals at generation.
    # Pub. L. 100-202 section 584(a)(1)(B), as amended, sets a March 22,
    # 1988 departure floor, not an empirical 2000 upper-admission cutoff.
    # https://www.govinfo.gov/content/pkg/USCODE-2024-title8/html/USCODE-2024-title8-chap12-subchapI-sec1101.htm
    amerasian = _maximum_entropy_support(
        result,
        alien
        & (profile.birth_country == 247)  # Vietnam
        & (entry_year >= 1988),
        seed=seed,
        salt="immigration:amerasian_admission_history_support",
    )
    special_immigrant = _maximum_entropy_support(
        result,
        alien
        & np.isin(profile.birth_country, [200, 213])  # Afghanistan, Iraq
        & (entry_year >= 2007),
        seed=seed,
        salt="immigration:iraqi_afghan_special_immigrant_history_support",
    )
    source_entry_first_month = profile.arrival_year_lower_bound * 12
    source_entry_last_month = profile.arrival_year_upper_bound * 12 + 11
    # Do not apply the April 24, 2024 extension to a pre-2024 build. The
    # month representation deliberately does not invent a within-2024
    # assessment day; Axiom owns time-specific benefit determinations.
    ukrainian_primary_last_month = (
        _UKRAINIAN_PRIMARY_PAROLE_LAST_MONTH if time_period >= 2024 else 2023 * 12 + 8
    )
    # Birthplace is the available programme-origin proxy, not proof of
    # citizenship/last habitual residence. Later-entry qualifying relatives
    # and people born elsewhere need those additional facts; this generator
    # currently supports only the explicit primary-cohort branch and must
    # not treat false here as proof that the statutory family branch failed.
    afghan_parole_history = _maximum_entropy_support(
        result,
        alien
        & (profile.birth_country == 200)
        & (source_entry_first_month <= _AFGHAN_PRIMARY_PAROLE_LAST_MONTH)
        & (source_entry_last_month >= _AFGHAN_PRIMARY_PAROLE_FIRST_MONTH),
        seed=seed,
        salt="immigration:qualifying_afghan_parole_history_support",
    )
    ukrainian_parole_history = _maximum_entropy_support(
        result,
        alien
        & (profile.birth_country == 164)
        & (source_entry_first_month <= ukrainian_primary_last_month)
        & (source_entry_last_month >= _UKRAINIAN_PRIMARY_PAROLE_FIRST_MONTH),
        seed=seed,
        salt="immigration:qualifying_ukrainian_parole_history_support",
    )
    # A specific parole programme is also a generic parole history; neither
    # the programme flag nor the generic flag should invent a second event.
    qualifying_parole_history = afghan_parole_history | ukrainian_parole_history
    parole_history |= qualifying_parole_history
    parole = parole_history & (years_in_us <= 2)
    battered_alien = _maximum_entropy_support(
        result,
        alien,
        seed=seed,
        salt="immigration:battered_alien_or_qualifying_relative_support",
    )
    severe_trafficking = _maximum_entropy_support(
        result,
        alien,
        seed=seed,
        salt="immigration:severe_trafficking_victim_or_relative_support",
    )

    # Recent immigrants may already have an immigrant visa or have adjusted
    # status; no five-year residence rule governs obtaining LPR status.
    # Citizenship/origin/arrival alone cannot rule it out. Absence of a
    # benefit or work indicator is not evidence of absence of lawful status.
    base_lpr_candidate = alien
    lpr = _maximum_entropy_support(
        result,
        base_lpr_candidate,
        seed=seed,
        salt="immigration:current_lpr_support",
    )
    adjustment_candidate = (
        refugee
        | asylum
        | withholding
        | conditional
        | che
        | cofa
        | amerasian
        | special_immigrant
        | afghan_parole_history
        | ukrainian_parole_history
        | battered_alien
        | severe_trafficking
        | parole_history
    ) & (years_in_us >= 1)
    humanitarian_lpr_adjustment = _maximum_entropy_support(
        result,
        adjustment_candidate,
        seed=seed,
        salt="immigration:lpr_adjustment_from_humanitarian_history_support",
    )
    lpr |= humanitarian_lpr_adjustment
    # The retained humanitarian parole event does not remain the person's
    # current admission basis after the modeled LPR transition (8 USC 1255(b)).
    # https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title8-section1255
    # USCIS's active-DACA stock likewise excludes people who obtained LPR
    # status; retain current DACA only on the non-LPR branch. This is not a
    # claim that a historical DACA grant never existed.
    # https://www.uscis.gov/sites/default/files/document/data/active_daca_recipients_fy2026_q2_v1.xlsx
    parole &= ~lpr
    daca &= ~lpr

    explicit_lawful = (
        lpr
        | refugee
        | asylum
        | parole
        | withholding
        | conditional
        | che
        | cofa
        | tps
        | daca
        | amerasian
        | battered_alien
        | severe_trafficking
    )
    # Census proxies do not establish a legal status or rule one out. Keep
    # both other-lawful and residual support when no explicit lawful fact
    # was assigned, independently of those suggestive observed indicators.
    other_lawful = _maximum_entropy_support(
        result,
        alien & ~explicit_lawful,
        seed=seed,
        salt="immigration:other_lawful_status_support",
    )
    # 8 CFR 274a.12(a)(1),(3),(5),(8),(10),(12) authorize these modeled
    # statuses incident to status. Retained refugee/asylum grants have no
    # modeled termination other than LPR adjustment, also authorized here.
    # Ordinary parole and DACA instead require a separate authorization
    # grant under (c)(11)/(33); generic lawful status, CHE, and the broad
    # trafficking/relative facts likewise do not establish an EAD grant.
    # https://www.govinfo.gov/content/pkg/CFR-2024-title8-vol1/pdf/CFR-2024-title8-vol1-sec274a-12.pdf
    # This fact means legal authorization to work, not possession of an EAD
    # document; citizens are also authorized and require no DHS grant.
    incident_employment_authorization = (
        profile.is_citizen | lpr | refugee | asylum | cofa | withholding | tps
    )
    # USCIS's November 21, 2022 policy additionally authorizes the qualifying
    # Afghan/Ukrainian parole cohorts incident to *current* parole. Apply
    # that categorical completion in full years after the policy began;
    # 2022 retains grant uncertainty without inventing an assessment day.
    # https://www.uscis.gov/newsroom/alerts/certain-afghan-and-ukrainian-parolees-are-employment-authorized-incident-to-parole
    # USCIS confirms the distinction from ordinary parole in its 2024
    # response to the 2023 Ombudsman report, p.14:
    # https://www.uscis.gov/sites/default/files/document/reports/Signed_OMB_Cleared_USCIS_Response_to_CISOMB_2023_Annual_Report-7-25-24.pdf
    if time_period >= 2023:
        incident_employment_authorization |= parole & qualifying_parole_history
    authorization_grant = _maximum_entropy_support(
        result,
        (explicit_lawful | other_lawful) & ~incident_employment_authorization,
        seed=seed,
        salt="immigration:employment_authorization_grant_support",
    )
    employment_authorization = incident_employment_authorization | authorization_grant

    ssn_evidence = _ssn_evidence(result)
    valid_work_ssn = profile.is_citizen | _maximum_entropy_support(
        result,
        alien & employment_authorization,
        seed=seed,
        salt="immigration:ssn_valid_for_employment_support",
    )
    other_ssn = _maximum_entropy_support(
        result,
        alien & ~valid_work_ssn & (lawful_evidence | ssn_evidence),
        seed=seed,
        salt="immigration:ssn_not_valid_for_employment_support",
    )
    has_ssn = profile.is_citizen | ssn_evidence | valid_work_ssn | other_ssn

    entry_first_month = source_entry_first_month.copy()
    entry_last_month = source_entry_last_month.copy()
    for selected, first, last in (
        (
            afghan_parole_history,
            _AFGHAN_PRIMARY_PAROLE_FIRST_MONTH,
            _AFGHAN_PRIMARY_PAROLE_LAST_MONTH,
        ),
        (
            ukrainian_parole_history,
            _UKRAINIAN_PRIMARY_PAROLE_FIRST_MONTH,
            ukrainian_primary_last_month,
        ),
    ):
        entry_first_month[selected] = np.maximum(entry_first_month[selected], first)
        entry_last_month[selected] = np.minimum(entry_last_month[selected], last)
    imputed_entry_year = _stable_imputed_year(
        result,
        alien,
        entry_first_month // 12,
        entry_last_month // 12,
        seed=seed,
        salt="immigration:entry_year_within_reported_interval",
    )
    refugee_year = np.where(refugee, imputed_entry_year, 0).astype(np.int64)
    parole_year = np.where(parole_history, imputed_entry_year, 0).astype(np.int64)
    conditional_year = np.where(conditional, imputed_entry_year, 0).astype(np.int64)
    che_year = np.where(che, imputed_entry_year, 0).astype(np.int64)
    amerasian_year = np.where(amerasian, imputed_entry_year, 0).astype(np.int64)
    cofa_year = np.where(cofa, imputed_entry_year, 0).astype(np.int64)
    special_immigrant_year = np.where(special_immigrant, imputed_entry_year, 0).astype(
        np.int64
    )
    afghan_parole_year = np.where(afghan_parole_history, imputed_entry_year, 0).astype(
        np.int64
    )
    ukrainian_parole_year = np.where(
        ukrainian_parole_history, imputed_entry_year, 0
    ).astype(np.int64)
    asylum_year = _stable_imputed_year(
        result,
        asylum,
        imputed_entry_year,
        np.full(len(result), time_period, dtype=np.int64),
        seed=seed,
        salt="immigration:asylum_grant_year_after_entry",
    )
    withholding_year = _stable_imputed_year(
        result,
        withholding,
        np.minimum(imputed_entry_year + 1, time_period),
        np.full(len(result), time_period, dtype=np.int64),
        seed=seed,
        salt="immigration:withholding_grant_year_after_entry",
    )
    trafficking_year = _stable_imputed_year(
        result,
        severe_trafficking,
        imputed_entry_year,
        np.full(len(result), time_period, dtype=np.int64),
        seed=seed,
        salt="immigration:trafficking_qualified_status_start_year",
    )
    battered_year = _stable_imputed_year(
        result,
        battered_alien,
        imputed_entry_year,
        np.full(len(result), time_period, dtype=np.int64),
        seed=seed,
        salt="immigration:battered_alien_qualified_status_start_year",
    )
    history_starts = (
        (refugee, refugee_year),
        (asylum, asylum_year),
        (withholding, withholding_year),
        (conditional, conditional_year),
        (che, che_year),
        (cofa, cofa_year),
        (parole_history, parole_year),
        (amerasian, amerasian_year),
        (special_immigrant, special_immigrant_year),
        (afghan_parole_history, afghan_parole_year),
        (ukrainian_parole_history, ukrainian_parole_year),
        (battered_alien, battered_year),
        (severe_trafficking, trafficking_year),
    )
    sentinel = time_period + 1
    # Every history marked as preceding this adjustment must actually precede
    # it. Bounding by only the earliest grant would allow a later asylum or
    # trafficking grant to be labeled as the basis of an earlier LPR event.
    last_history_start = np.max(
        np.vstack([np.where(flag, start, 0) for flag, start in history_starts]),
        axis=0,
    )
    lpr_lower_bound = np.where(
        humanitarian_lpr_adjustment & (last_history_start > 0),
        np.minimum(last_history_start + 1, time_period),
        imputed_entry_year,
    )
    lpr_year = _stable_imputed_year(
        result,
        lpr,
        lpr_lower_bound,
        np.full(len(result), time_period, dtype=np.int64),
        seed=seed,
        salt="immigration:lpr_status_acquisition_year",
    )
    entry_month_index = _stable_imputed_month_index(
        result,
        alien,
        imputed_entry_year,
        seed=seed,
        salt="immigration:entry_month_within_imputed_year",
        lower_bound=entry_first_month,
        upper_bound=entry_last_month,
    )

    def event_month(
        flag: np.ndarray,
        year: np.ndarray,
        name: str,
        *,
        no_earlier_than: np.ndarray | None = None,
    ) -> np.ndarray:
        sampled = _stable_imputed_month_index(
            result,
            flag,
            year,
            seed=seed,
            salt=f"immigration:{name}_month_within_year",
        )
        if no_earlier_than is not None:
            sampled = np.where(
                flag,
                np.maximum(sampled, no_earlier_than),
                0,
            ).astype(np.int64)
        return sampled

    refugee_month = event_month(
        refugee,
        refugee_year,
        "refugee_admission",
        no_earlier_than=entry_month_index,
    )
    asylum_month = event_month(
        asylum,
        asylum_year,
        "asylum_grant",
        no_earlier_than=entry_month_index,
    )
    parole_month = event_month(
        parole_history,
        parole_year,
        "parole_start",
        no_earlier_than=entry_month_index,
    )
    # This branch asserts parole *into* the United States in the programme
    # window, so its grant and modeled entry share the same unknown month.
    # Keep the generic history clock aligned with the specific history.
    parole_month = np.where(
        qualifying_parole_history, entry_month_index, parole_month
    ).astype(np.int64)
    withholding_month = event_month(
        withholding,
        withholding_year,
        "withholding_start",
        no_earlier_than=entry_month_index,
    )
    conditional_month = event_month(
        conditional,
        conditional_year,
        "conditional_entry",
        no_earlier_than=entry_month_index,
    )
    che_month = event_month(
        che,
        che_year,
        "cuban_haitian_entrant_start",
        no_earlier_than=entry_month_index,
    )
    amerasian_month = event_month(
        amerasian,
        amerasian_year,
        "amerasian_admission",
        no_earlier_than=entry_month_index,
    )
    cofa_month = event_month(
        cofa,
        cofa_year,
        "cofa_status_start",
        no_earlier_than=entry_month_index,
    )
    special_immigrant_month = event_month(
        special_immigrant,
        special_immigrant_year,
        "special_immigrant_status_start",
        no_earlier_than=entry_month_index,
    )
    afghan_parole_month = np.where(afghan_parole_history, parole_month, 0).astype(
        np.int64
    )
    ukrainian_parole_month = np.where(ukrainian_parole_history, parole_month, 0).astype(
        np.int64
    )
    battered_month = event_month(
        battered_alien,
        battered_year,
        "battered_alien_qualified_status_start",
        no_earlier_than=entry_month_index,
    )
    trafficking_month = event_month(
        severe_trafficking,
        trafficking_year,
        "trafficking_qualified_status_start",
        no_earlier_than=entry_month_index,
    )
    last_history_month = np.max(
        np.vstack(
            [
                np.where(flag, start, 0)
                for flag, start in (
                    (refugee, refugee_month),
                    (asylum, asylum_month),
                    (withholding, withholding_month),
                    (conditional, conditional_month),
                    (che, che_month),
                    (cofa, cofa_month),
                    (parole_history, parole_month),
                    (amerasian, amerasian_month),
                    (special_immigrant, special_immigrant_month),
                    (afghan_parole_history, afghan_parole_month),
                    (ukrainian_parole_history, ukrainian_parole_month),
                    (battered_alien, battered_month),
                    (severe_trafficking, trafficking_month),
                )
            ]
        ),
        axis=0,
    )
    lpr_month = event_month(
        lpr,
        lpr_year,
        "lpr_status_acquisition",
        no_earlier_than=np.where(
            humanitarian_lpr_adjustment & (last_history_month > 0),
            last_history_month,
            entry_month_index,
        ),
    )
    parole_period_years = np.where(parole_history, 2, 0).astype(np.int64)

    qualified_statuses = (
        (lpr, lpr_year),
        (refugee, refugee_year),
        (asylum, asylum_year),
        (parole_history & (parole_period_years >= 1), parole_year),
        (withholding, withholding_year),
        (conditional, conditional_year),
        (che, che_year),
        (cofa, cofa_year),
        (amerasian, amerasian_year),
        (special_immigrant, special_immigrant_year),
        (afghan_parole_history, afghan_parole_year),
        (ukrainian_parole_history, ukrainian_parole_year),
        (battered_alien, battered_year),
        (severe_trafficking, trafficking_year),
    )
    qualified_start = np.min(
        np.vstack(
            [np.where(flag, start, sentinel) for flag, start in qualified_statuses]
        ),
        axis=0,
    )
    qualified = qualified_start != sentinel
    qualified_start = np.where(qualified, qualified_start, 0).astype(np.int64)
    qualified_start_month = np.min(
        np.vstack(
            [
                np.where(flag, start, sentinel * 12)
                for flag, start in (
                    (lpr, lpr_month),
                    (refugee, refugee_month),
                    (asylum, asylum_month),
                    (parole_history & (parole_period_years >= 1), parole_month),
                    (withholding, withholding_month),
                    (conditional, conditional_month),
                    (che, che_month),
                    (cofa, cofa_month),
                    (amerasian, amerasian_month),
                    (special_immigrant, special_immigrant_month),
                    (afghan_parole_history, afghan_parole_month),
                    (ukrainian_parole_history, ukrainian_parole_month),
                    (battered_alien, battered_month),
                    (severe_trafficking, trafficking_month),
                )
            ]
        ),
        axis=0,
    )
    qualified_start_month = np.where(qualified, qualified_start_month, 0).astype(
        np.int64
    )
    qualified_after_cutoff = _started_on_or_after_august_22_1996(
        result,
        qualified,
        qualified_start,
        qualified_start_month,
        seed=seed,
    )

    facts: dict[str, np.ndarray] = {
        "immigration_is_us_citizen": profile.is_citizen.astype(bool),
        # Every U.S. citizen is also a U.S. national; the surveys do not
        # separately identify the small non-citizen-national population.
        "immigration_is_us_national": profile.is_citizen.astype(bool),
        "immigration_is_us_resident": np.ones(len(result), dtype=bool),
        "immigration_is_alien": alien.astype(bool),
        "immigration_has_lawful_status_evidence": lawful_evidence,
        "immigration_has_other_lawful_status": other_lawful,
        "immigration_is_lawful_permanent_resident": lpr,
        "immigration_was_admitted_as_refugee": refugee,
        "immigration_has_been_granted_asylum": asylum,
        "immigration_was_paroled": parole_history,
        "immigration_is_currently_paroled": parole,
        "immigration_has_deportation_or_removal_withheld": withholding,
        "immigration_was_granted_conditional_entry": conditional,
        "immigration_is_current_cuban_haitian_entrant": che,
        "immigration_is_current_cofa_resident": cofa,
        "immigration_is_current_tps": tps,
        "immigration_is_current_daca": daca,
        "immigration_is_amerasian_immigrant": amerasian,
        "immigration_has_iraqi_or_afghan_special_immigrant_history": (
            special_immigrant
        ),
        "immigration_has_qualifying_afghan_humanitarian_parole_history": (
            afghan_parole_history
        ),
        "immigration_has_qualifying_ukrainian_humanitarian_parole_history": (
            ukrainian_parole_history
        ),
        "immigration_is_battered_alien_or_qualifying_relative": battered_alien,
        "immigration_is_severe_trafficking_victim_or_qualifying_relative": (
            severe_trafficking
        ),
        "immigration_adjusted_to_lpr_from_refugee_status": (
            humanitarian_lpr_adjustment & refugee
        ),
        "immigration_adjusted_to_lpr_from_asylum_status": (
            humanitarian_lpr_adjustment & asylum
        ),
        "immigration_adjusted_to_lpr_from_withholding_status": (
            humanitarian_lpr_adjustment & withholding
        ),
        "immigration_adjusted_to_lpr_from_conditional_entry_status": (
            humanitarian_lpr_adjustment & conditional
        ),
        "immigration_adjusted_to_lpr_from_cuban_haitian_entrant_status": (
            humanitarian_lpr_adjustment & che
        ),
        "immigration_adjusted_to_lpr_from_cofa_status": (
            humanitarian_lpr_adjustment & cofa
        ),
        "immigration_adjusted_to_lpr_from_amerasian_status": (
            humanitarian_lpr_adjustment & amerasian
        ),
        "immigration_adjusted_to_lpr_from_iraqi_afghan_special_immigrant_status": (
            humanitarian_lpr_adjustment & special_immigrant
        ),
        "immigration_adjusted_to_lpr_from_qualifying_afghan_parole": (
            humanitarian_lpr_adjustment & afghan_parole_history
        ),
        "immigration_adjusted_to_lpr_from_qualifying_ukrainian_parole": (
            humanitarian_lpr_adjustment & ukrainian_parole_history
        ),
        "immigration_adjusted_to_lpr_from_battered_alien_status": (
            humanitarian_lpr_adjustment & battered_alien
        ),
        "immigration_adjusted_to_lpr_from_severe_trafficking_victim_status": (
            humanitarian_lpr_adjustment & severe_trafficking
        ),
        "immigration_adjusted_to_lpr_from_parole_for_at_least_one_year": (
            humanitarian_lpr_adjustment & parole_history
        ),
        "immigration_has_employment_authorization": employment_authorization,
        "immigration_has_ssn_evidence": ssn_evidence,
        "immigration_has_ssn": has_ssn,
        "immigration_ssn_is_valid_for_employment": valid_work_ssn,
        "immigration_us_entry_year": entry_year,
        "immigration_us_entry_year_lower_bound": np.where(
            alien, profile.arrival_year_lower_bound, 0
        ).astype(np.int64),
        "immigration_us_entry_year_upper_bound": np.where(
            alien, profile.arrival_year_upper_bound, 0
        ).astype(np.int64),
        "immigration_entry_year_is_interval_midpoint": (alien & ~profile.source_is_acs),
        "immigration_us_entry_month_index": entry_month_index,
        "immigration_status_start_year_is_imputed": (
            explicit_lawful
            | parole_history
            | special_immigrant
            | afghan_parole_history
            | ukrainian_parole_history
        ),
        "immigration_status_start_month_is_imputed": (
            explicit_lawful
            | parole_history
            | special_immigrant
            | afghan_parole_history
            | ukrainian_parole_history
        ),
        "immigration_lpr_status_start_year": lpr_year,
        "immigration_lpr_status_start_month_index": lpr_month,
        "immigration_refugee_admission_year": refugee_year,
        "immigration_refugee_admission_month_index": refugee_month,
        "immigration_asylum_grant_year": asylum_year,
        "immigration_asylum_grant_month_index": asylum_month,
        "immigration_parole_start_year": parole_year,
        "immigration_parole_start_month_index": parole_month,
        "immigration_parole_period_years": parole_period_years,
        "immigration_withholding_start_year": withholding_year,
        "immigration_withholding_start_month_index": withholding_month,
        "immigration_conditional_entry_year": conditional_year,
        "immigration_conditional_entry_month_index": conditional_month,
        "immigration_cuban_haitian_entrant_start_year": che_year,
        "immigration_cuban_haitian_entrant_start_month_index": che_month,
        "immigration_amerasian_admission_year": amerasian_year,
        "immigration_amerasian_admission_month_index": amerasian_month,
        "immigration_first_qualified_status_start_year": qualified_start,
        "immigration_first_qualified_status_start_month_index": (qualified_start_month),
        "immigration_first_qualified_status_start_year_is_imputed": qualified,
        "immigration_first_qualified_status_start_month_is_imputed": qualified,
        "immigration_first_qualified_status_started_on_or_after_1996_08_22": (
            qualified_after_cutoff
        ),
    }
    ssn, status = _legacy_immigration_projection(facts)
    for column in US_IMMIGRATION_FACT_COLUMNS:
        result[column] = facts[column]
    result["ssn_card_type"] = ssn
    result["immigration_status_str"] = status
    return result


def _daca_statutory_cohort(
    arrival_year: np.ndarray,
    age_at_entry: np.ndarray,
    age: np.ndarray,
    *,
    age_reference_year: np.ndarray,
) -> np.ndarray:
    """Return the source-supported DACA cohort without inventing exact dates.

    8 CFR 236.22(b)(7) requires birth on or after June 16, 1981. Age measured
    on an unknown date in ``age_reference_year`` permits birth in either
    ``age_reference_year - age - 1`` or ``age_reference_year - age``. Exclude a row only if
    its latest possible birth year predates 1981; the within-1981 boundary
    remains support uncertainty, not an imputed exact DOB.
    """

    return (
        (arrival_year <= _DACA_LATEST_ARRIVAL_YEAR)
        & (age_reference_year - age >= _DACA_EARLIEST_BIRTH_YEAR)
        & (age_at_entry < _DACA_MAX_AGE_AT_ENTRY)
        & (age >= _DACA_MIN_CURRENT_AGE)
    )


def with_us_immigration_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    person_weight_scale: float = 1.0,
) -> Frame:
    """Append neutral immigration facts and the one-way PE projection.

    A complete neutral surface is idempotent.  A frame carrying only the old
    two-column PE surface is upgraded from its raw evidence rather than being
    mistaken for a completed Axiom-first assignment.

    Args:
        frame: A US-schema frame whose person table still carries the raw
            CPS ASEC source columns.
        seed: Build-wide imputation seed.
        time_period: The dataset's time period (arrival-year arithmetic).
        person_weight_scale: Deprecated compatibility argument.  Assignment
            deliberately ignores it; representation belongs to calibration.

    Returns:
        A new frame whose person table carries neutral nonexclusive facts and
        the two legacy projection columns.

    Raises:
        ValueError: If the frame is not US-schema, one output column exists
            without the other, or the stage output does not cover every
            person.
        SourceRuntimeError: If source citizenship/origin/arrival evidence or
            stable identity is missing.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US immigration inputs require the US schema.")
    # Kept in the public signature until generated callers migrate.  Do not
    # inspect, validate, or apply it: even a stage-only weight scale must have
    # no route into support assignment.
    del person_weight_scale
    person = frame.table("person")
    fact_present = [
        column for column in US_IMMIGRATION_FACT_COLUMNS if column in person.columns
    ]
    legacy_present = [
        column
        for column in US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS
        if column in person.columns
    ]
    if len(fact_present) == len(US_IMMIGRATION_FACT_COLUMNS) and len(
        legacy_present
    ) == len(US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS):
        return frame
    if fact_present:
        missing = sorted(set(US_IMMIGRATION_FACT_COLUMNS) - set(fact_present))
        if not missing:
            missing = sorted(
                set(US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS) - set(legacy_present)
            )
        raise ValueError(
            f"US frame carries a partial neutral immigration surface without "
            f"{missing}; "
            "immigration surface would silently default the missing column."
        )
    if 0 < len(legacy_present) < len(US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS):
        missing = sorted(
            set(US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS) - set(legacy_present)
        )
        raise ValueError(
            f"US frame carries {legacy_present} without {missing}; a partial "
            "legacy immigration projection is invalid."
        )

    output = run_source_stage(
        us_immigration_stage_spec(),
        tables={"person": person},
        operation_handlers={
            "derive_immigration_status": derive_us_immigration_status_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_IMMIGRATION_ALL_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                f"US immigration stage output does not cover every person for "
                f"{column!r}."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_IMMIGRATION_ALL_OUTPUT_COLUMNS:
        tables["person"][column] = aligned[column].to_numpy()
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_immigration_composition_summary(frame: Frame) -> dict[str, object]:
    """Weighted SSN-card-type and immigration-status composition of a frame."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total = float(weights.sum())
    summary: dict[str, object] = {"person_population": total}
    for column in US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS:
        if column not in person.columns:
            summary[column] = None
            continue
        values = person[column].astype(str)
        populations = {
            str(value): float(weights[(values == value).to_numpy()].sum())
            for value in sorted(values.unique())
        }
        summary[column] = {
            "population": populations,
            "share": {
                value: (population / total if total else None)
                for value, population in populations.items()
            },
        }
    return summary


def us_immigration_composition_gate(frame: Frame) -> GateResult:
    """Validate the neutral fact surface and its one-way PE projection.

    This gate checks structure and internal consistency only. It deliberately
    applies no weighted prevalence, stock, flow, or population target; those
    belong to calibration against separately versioned government facts.
    """

    person = frame.table("person")
    failures: list[str] = []
    details: dict[str, object] = {
        "summary": us_immigration_composition_summary(frame),
        "neutral_fact_count": len(US_IMMIGRATION_FACT_COLUMNS),
        "legacy_projection_columns": list(US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS),
        "target_free": True,
    }
    missing = sorted(set(US_IMMIGRATION_ALL_OUTPUT_COLUMNS) - set(person.columns))
    if missing:
        failures.append(
            f"missing person column(s) {missing}: the neutral immigration "
            "surface or its PolicyEngine projection is incomplete."
        )
        return GateResult(
            name="immigration_composition",
            passed=False,
            failures=tuple(failures),
            details=details,
        )

    integer_facts = {
        column
        for column in US_IMMIGRATION_FACT_COLUMNS
        if column.endswith("_year")
        or column.endswith("_month_index")
        or column
        in {
            "immigration_us_entry_year_lower_bound",
            "immigration_us_entry_year_upper_bound",
            "immigration_parole_period_years",
        }
    }
    boolean_facts = set(US_IMMIGRATION_FACT_COLUMNS) - integer_facts
    for column in sorted(boolean_facts):
        values = person[column]
        invalid = values.isna() | ~values.isin((False, True))
        if invalid.any():
            failures.append(
                f"{column}: {int(invalid.sum())} row(s) are outside the "
                "non-null boolean fact domain."
            )
    for column in sorted(integer_facts):
        numeric = pd.to_numeric(person[column], errors="coerce")
        values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        invalid = ~np.isfinite(values) | (values < 0) | (values != np.floor(values))
        if invalid.any():
            failures.append(
                f"{column}: {int(invalid.sum())} row(s) are outside the "
                "non-negative integer fact domain."
            )
    if failures:
        return GateResult(
            name="immigration_composition",
            passed=False,
            failures=tuple(failures),
            details=details,
        )

    ssn = person["ssn_card_type"].astype(str)
    status = person["immigration_status_str"].astype(str)
    for column, values, domain in (
        ("ssn_card_type", ssn, SSN_CARD_TYPE_VALUES),
        ("immigration_status_str", status, IMMIGRATION_STATUS_VALUES),
    ):
        outside = sorted(set(values.unique()) - set(domain))
        if outside:
            failures.append(
                f"{column}: value(s) outside the PolicyEngine enum domain: {outside}."
            )
        if len(values) and values.nunique() < 2:
            failures.append(
                f"{column}: constant value {values.iloc[0]!r}; the legacy "
                "projection carries no immigration signal."
            )

    facts = {
        column: person[column].to_numpy(
            dtype=bool if column in boolean_facts else np.int64,
            copy=False,
        )
        for column in US_IMMIGRATION_FACT_COLUMNS
    }
    lpr = facts["immigration_is_lawful_permanent_resident"]
    for label, invalid in (
        ("current DACA and LPR", lpr & facts["immigration_is_current_daca"]),
        ("current parole and LPR", lpr & facts["immigration_is_currently_paroled"]),
        (
            "current parole without retained parole history",
            facts["immigration_is_currently_paroled"]
            & ~facts["immigration_was_paroled"],
        ),
    ):
        if invalid.any():
            failures.append(f"{label}: {int(invalid.sum())} inconsistent row(s).")

    transition_histories = (
        ("refugee_status", "was_admitted_as_refugee", "refugee_admission"),
        ("asylum_status", "has_been_granted_asylum", "asylum_grant"),
        (
            "withholding_status",
            "has_deportation_or_removal_withheld",
            "withholding_start",
        ),
        (
            "conditional_entry_status",
            "was_granted_conditional_entry",
            "conditional_entry",
        ),
        (
            "cuban_haitian_entrant_status",
            "is_current_cuban_haitian_entrant",
            "cuban_haitian_entrant_start",
        ),
        ("cofa_status", "is_current_cofa_resident", None),
        ("amerasian_status", "is_amerasian_immigrant", "amerasian_admission"),
        (
            "iraqi_afghan_special_immigrant_status",
            "has_iraqi_or_afghan_special_immigrant_history",
            None,
        ),
        (
            "qualifying_afghan_parole",
            "has_qualifying_afghan_humanitarian_parole_history",
            None,
        ),
        (
            "qualifying_ukrainian_parole",
            "has_qualifying_ukrainian_humanitarian_parole_history",
            None,
        ),
        ("battered_alien_status", "is_battered_alien_or_qualifying_relative", None),
        (
            "severe_trafficking_victim_status",
            "is_severe_trafficking_victim_or_qualifying_relative",
            None,
        ),
        ("parole_for_at_least_one_year", "was_paroled", "parole_start"),
    )
    for transition_suffix, history_suffix, event_suffix in transition_histories:
        transition_name = f"immigration_adjusted_to_lpr_from_{transition_suffix}"
        transition = facts[transition_name]
        invalid = transition & (~lpr | ~facts[f"immigration_{history_suffix}"])
        if event_suffix is not None:
            event = facts[f"immigration_{event_suffix}_month_index"]
            invalid |= transition & (
                (event <= 0)
                | (event > facts["immigration_lpr_status_start_month_index"])
            )
        if invalid.any():
            failures.append(
                f"{transition_name}: {int(invalid.sum())} row(s) lack the LPR/"
                "history facts or have a history event after LPR acquisition."
            )
    expected_ssn, expected_status = _legacy_immigration_projection(facts)
    ssn_mismatch = ssn.to_numpy() != expected_ssn
    status_mismatch = status.to_numpy() != expected_status
    details["legacy_projection_consistency"] = {
        "ssn_card_type_mismatch_rows": int(ssn_mismatch.sum()),
        "immigration_status_str_mismatch_rows": int(status_mismatch.sum()),
    }
    if ssn_mismatch.any():
        failures.append(
            f"ssn_card_type differs from the neutral-fact projection on "
            f"{int(ssn_mismatch.sum())} row(s)."
        )
    if status_mismatch.any():
        failures.append(
            f"immigration_status_str differs from the neutral-fact projection "
            f"on {int(status_mismatch.sum())} row(s)."
        )

    return GateResult(
        name="immigration_composition",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
