"""SSN card type and immigration status from CPS ASEC citizenship inputs.

Without this stage the published dataset stores no SSN/immigration columns at
all, so the rules engine defaults every person to a citizen with a valid SSN
and every SSN- or immigration-conditioned policy becomes a no-op — the
failure mode of microcosm issue #225 (OBBBA's CTC SSN tightening scored ≈ $0
against PolicyEngine's official +$3.3B).

The stage writes two PolicyEngine-US person input columns:

- ``ssn_card_type``: ``CITIZEN`` / ``NON_CITIZEN_VALID_EAD`` /
  ``OTHER_NON_CITIZEN`` / ``NONE`` (likely undocumented, i.e. ITIN-filer
  territory for tax purposes).
- ``immigration_status_str``: ``CITIZEN`` / ``LEGAL_PERMANENT_RESIDENT`` /
  ``CUBAN_HAITIAN_ENTRANT`` / ``DACA`` / ``UNDOCUMENTED`` plus the supported
  humanitarian and temporary-protection statuses listed below.

Method — survey measurement first, published residual method second, and one
narrowly-scoped control where the survey carries no signal at all:

1. **Citizenship is measured, not imputed.** ``PRCITSHP`` in {1..4} →
   citizens; 5 → non-citizens.
2. **Legal-status indicators (ASEC-UA residual method).** Non-citizens with
   any indicator of authorized status — pre-1982 IRCA-cohort arrival,
   naturalization-eligibility, Medicare/Medicaid/SSI/Social Security receipt,
   federal pensions, IHS/CHAMPVA/military coverage, government employment,
   subsidized housing, veteran status — move to ``OTHER_NON_CITIZEN``
   (Van Hook et al., "Are Estimates of Unauthorized Immigrants Based on
   Residual Methods Robust?", SSRN 4662801).
3. **Work/study authorization split.** The CPS carries no work-authorization
   variable, so among residual non-citizens the split between EAD holders and
   the unauthorized is unidentified from the survey alone. Workers (measured
   with ASEC ``A_LFSR`` at age 16+, not prior-year earnings) and students
   spill to ``NON_CITIZEN_VALID_EAD`` in deterministic seeded order until the
   broad reported universes match their controls. Pew retains DACA, parole,
   TPS, and similar temporary protections in its unauthorized estimates, so
   those statuses continue to count after receiving EADs. Residual
   Cuban/Haitian rows that receive an EAD also stay in that universe because
   the engine's broader ``CUBAN_HAITIAN_ENTRANT`` label carries the modeled
   CHNV-parole/Haiti-TPS subset.
4. **The total Pew-defined unauthorized population is emergent, not forced.**
   Representation is calibration's job, not the label stage's: the release
   gate checks the emergent broad-universe total against Pew's cited 2023
   anchor with a coarse plausibility band. The engine's narrower
   ``UNDOCUMENTED`` enum remains exactly paired with ``ssn_card_type=NONE``.
5. **Humanitarian/temporary-protection statuses are drawn to published
   stocks, never blanketed (microcosm #767).** Blanket ``REFUGEE``/``TPS``
   labels for recent arrivals were originally refused because LPR treatment
   gave near-identical means-tested eligibility; H.R.1 repealed exactly that
   equivalence (Medicaid §71109 effective 2026-10-01, SNAP §10108 effective
   2025-07-01, ACA §71301/§71302 effective 2026/2027 — all keyed to these
   enum values in policyengine-us parameters), so an LPR-only file silently
   zeroes every one of those channels. Instead of blankets, the stage now
   draws ``REFUGEE`` / ``ASYLEE`` / ``DEPORTATION_WITHHELD`` /
   ``PAROLED_ONE_YEAR`` / ``TPS`` — in that order, mutually exclusive —
   from candidates whose survey signature supports the status (origin
   country x arrival window x legal-status indicators), each to a
   manifest-cited weighted stock target. Cuba/Haiti-born persons are
   excluded from every humanitarian draw (the ``CUBAN_HAITIAN_ENTRANT``
   class is the better statutory label and keeps H.R.1 eligibility), as is
   the DACA statutory cohort. Draws from the residual pool (TPS and parole
   only) flip ``ssn_card_type`` ``NONE`` to ``NON_CITIZEN_VALID_EAD`` —
   both programs grant employment authorization — so the two output columns
   never disagree. ``CONDITIONAL_ENTRANT`` is deliberately not emitted: the
   INA 203(a)(7) class closed in 1980 and surviving holders are
   indistinguishable from LPRs at this granularity.
6. **Status tags carry only statutory tests the data can support.**
   ``DACA`` applies the statutory cohort test (arrived by 2007 before age
   16, aged 15+) to EAD holders; ``CUBAN_HAITIAN_ENTRANT`` applies the
   nationality-plus-arrival class to documented non-citizens; every other
   documented non-citizen stays ``LEGAL_PERMANENT_RESIDENT`` (the modal true
   status).

Selection draws are seeded blake2b hashes keyed by the person's stable source
identity (``source_year``/``source_household_id``/``source_person_id`` when
present), so
support-channel clones of one source person always receive the same status
and reruns are bit-reproducible without global RNG state.

Control totals and the gate anchor are data, not code: they live in the
``immigration_status`` stage of ``microcosm/build/us/source_stages.json`` with
one source citation per number, and reach this module as manifest operation
parameters.
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
    "HUMANITARIAN_STATUS_CATEGORIES",
    "IMMIGRATION_STATUS_VALUES",
    "SSN_CARD_TYPE_VALUES",
    "US_IMMIGRATION_NONCONSTANT_PERSON_COLUMNS",
    "US_IMMIGRATION_OUTPUT_COLUMNS",
    "US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS",
    "US_IMMIGRATION_STAGE_NAME",
    "HumanitarianDraw",
    "ImmigrationControls",
    "UndocumentedControls",
    "derive_us_immigration_status_from_manifest",
    "reconcile_us_immigration_humanitarian_transfer",
    "us_immigration_composition_gate",
    "us_immigration_composition_summary",
    "us_immigration_controls",
    "us_immigration_evidence_features",
    "us_immigration_humanitarian_draw_mask",
    "us_immigration_humanitarian_transfer_selection_masks",
    "us_immigration_stage_spec",
    "with_us_immigration_inputs",
]

US_IMMIGRATION_STAGE_NAME = "immigration_status"

#: The PolicyEngine-US person input columns this stage owns.
US_IMMIGRATION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "ssn_card_type",
    "immigration_status_str",
)

#: Release gates require these person columns to carry signal (≥2 values).
US_IMMIGRATION_NONCONSTANT_PERSON_COLUMNS: tuple[str, ...] = (
    US_IMMIGRATION_OUTPUT_COLUMNS
)

#: PolicyEngine-US ``SSNCardType`` enum member names.
SSN_CARD_TYPE_VALUES: tuple[str, ...] = (
    "CITIZEN",
    "NON_CITIZEN_VALID_EAD",
    "OTHER_NON_CITIZEN",
    "NONE",
)

#: PolicyEngine-US ``ImmigrationStatus`` enum member names this stage emits
#: (a deliberate subset of the engine's full enum domain; see module
#: docstring for why ``CONDITIONAL_ENTRANT`` is not fabricated).
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
    "A_MARITL",
    "A_SPOUSE",
    "A_HSCOL",
    "A_LFSR",
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

_PERSON_WEIGHT_COLUMN = "person_weight"

_FORCED_CONTROL_KEYS = ("undocumented_workers", "undocumented_students")
_ANCHOR_KEY = "undocumented_population_anchor"
_HUMANITARIAN_KEY = "humanitarian_status_stocks"

_DERIVE_IMMIGRATION_STATUS_PARAMETER_KEYS = frozenset(
    {
        *_FORCED_CONTROL_KEYS,
        _ANCHOR_KEY,
        _HUMANITARIAN_KEY,
        "seed_from_build_config",
        "time_period_from_build_config",
    }
)

#: PEINUSYR codes that mean arrival before 1982 (IRCA amnesty eligible).
_PRE_1982_ARRIVAL_CODES = (1, 2, 3, 4, 5, 6, 7)

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

#: PENATVTY codes for Cuba and Haiti; the Cuban/Haitian entrant class exists
#: for arrivals after the Refugee Education Assistance Act of 1980.
_CUBAN_HAITIAN_BIRTH_CODES = (327, 332)
_CUBAN_HAITIAN_ARRIVAL_CUTOFF = 1980

#: DACA statutory cohort tests (USCIS): arrived before June 15 2007 and
#: before their 16th birthday, and at least 15 to request.
_DACA_LATEST_ARRIVAL_YEAR = 2007
_DACA_MAX_AGE_AT_ENTRY = 16
_DACA_MIN_CURRENT_AGE = 15

#: Humanitarian draw order: decreasing target precision (exact per-origin
#: program admissions first, national flow aggregates next, per-country TPS
#: registrations last). Draws are sequential and mutually exclusive.
HUMANITARIAN_STATUS_CATEGORIES: tuple[str, ...] = (
    "paroled_one_year",
    "refugee",
    "asylee",
    "deportation_withheld",
    "tps",
)

_HUMANITARIAN_STATUS_BY_CATEGORY: Mapping[str, str] = {
    "paroled_one_year": "PAROLED_ONE_YEAR",
    "refugee": "REFUGEE",
    "asylee": "ASYLEE",
    "deportation_withheld": "DEPORTATION_WITHHELD",
    "tps": "TPS",
}

# Pew's residual estimates use a deliberately broader universe than the
# engine's ``UNDOCUMENTED`` enum.  In particular, Pew retains DACA recipients
# and people with temporary protection (including parole and TPS) in its
# unauthorized-immigrant totals. Refugees and people already granted asylum
# are lawfully admitted statuses and therefore stay outside this universe.
# ``CUBAN_HAITIAN_ENTRANT`` is conditional: only EAD rows that came from the
# residual pool count, not assumed-documented entrants sharing that broad
# engine label.
_PEW_UNAUTHORIZED_STATUS_VALUES: tuple[str, ...] = (
    "UNDOCUMENTED",
    "DACA",
    "PAROLED_ONE_YEAR",
    "DEPORTATION_WITHHELD",
    "TPS",
)
_PEW_INCLUDED_HUMANITARIAN_STATUS_VALUES: tuple[str, ...] = (
    "PAROLED_ONE_YEAR",
    "DEPORTATION_WITHHELD",
    "TPS",
)

# CPS ASEC A_LFSR: 1 working, 2 with a job/not at work, 3 unemployed/looking,
# 4 unemployed/on layoff.  Code 0 covers children and Armed Forces; 7 is not
# in the labor force.  Pew's labor-force total is for people age 16 or older.
_CPS_LABOR_FORCE_STATUS_CODES = (1, 2, 3, 4)
_CPS_LABOR_FORCE_STATUS_DOMAIN = (0, 1, 2, 3, 4, 7)

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

#: PENATVTY birth-country codes, verified against Census CPS technical
#: documentation (cpsmar24.pdf) Appendix I "Countries and Areas of the
#: World". Cuba (327) and Haiti (332) stay out of every humanitarian draw:
#: the Cuban/Haitian-entrant class is the better statutory label.
#:
#: Parole per-origin keys: Operation Allies Welcome (Afghanistan), Uniting
#: for Ukraine, and the non-Cuban/Haitian CHNV nationalities.
_PAROLE_ORIGIN_CODES: Mapping[str, tuple[int, ...]] = {
    "afghanistan": (200,),
    "ukraine": (164,),
    "nicaragua": (315,),
    "venezuela": (373,),
}

#: Top refugee-resettlement nationalities, OHSS Refugees Annual Flow Report
#: FY2023 Table 3 and the OHSS FY-24 report (DR Congo, Syria, Afghanistan,
#: Burma, Venezuela lead; Congo appears as both 412 Congo and 459 Zaire).
_REFUGEE_ORIGIN_CODES: tuple[int, ...] = (
    412,  # Congo
    459,  # Zaire (Democratic Republic of the Congo)
    239,  # Syria
    200,  # Afghanistan
    205,  # Myanmar (Burma)
    373,  # Venezuela
    448,  # Somalia
    451,  # Sudan
    417,  # Eritrea
    213,  # Iraq
    313,  # Guatemala
    164,  # Ukraine
)

#: Top asylum-grant nationalities, OHSS Asylees flow reports FY2022-FY2024
#: (Afghanistan, China, Venezuela, Russia lead; Central/South America and
#: Egypt/Cameroon/Turkey round out the recurring top grant countries).
_ASYLEE_ORIGIN_CODES: tuple[int, ...] = (
    200,  # Afghanistan
    207,  # China
    373,  # Venezuela
    163,  # Russia
    312,  # El Salvador
    313,  # Guatemala
    314,  # Honduras
    315,  # Nicaragua
    210,  # India
    414,  # Egypt
    407,  # Cameroon
    364,  # Colombia
    365,  # Ecuador
    370,  # Peru
    243,  # Turkey
    239,  # Syria
)

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

#: ACS YOEP preserves an exact year, unlike the ASEC arrival bins. Use that
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

#: Categories whose flat manifest block carries one national target.
_FLAT_HUMANITARIAN_CATEGORIES = ("refugee", "asylee", "deportation_withheld")
#: Categories whose manifest block carries per-origin targets, and the
#: module table each block's keys must match exactly.
_PER_ORIGIN_HUMANITARIAN_CATEGORIES: Mapping[str, tuple[str, ...]] = {
    "paroled_one_year": tuple(_PAROLE_ORIGIN_CODES),
    "tps": tuple(_TPS_ORIGIN_CODES),
}

#: Weighted share of persons whose SSN card type is not ``CITIZEN`` must land
#: in this band (Census counts ~22–27M non-citizens of ~336M residents; a
#: share outside it means the imputation collapsed or exploded).
_NON_CITIZEN_SHARE_BAND = (0.03, 0.12)
#: Emergent Pew-defined unauthorized population relative to the cited
#: published anchor. This includes DACA and specified temporary protections,
#: not only the engine's ``NONE``/``UNDOCUMENTED`` pair. Coarse by design — a
#: release-blocking backstop against collapse or explosion, not a calibration
#: objective; the level belongs to the calibration lane.
_UNDOCUMENTED_ANCHOR_RELATIVE_BAND = (0.5, 1.6)
#: Emitted humanitarian mass per category relative to its cited target.
#: The draw forces the target when candidates suffice, so the band only
#: bites on pool exhaustion (the ASEC undercovers 2022-24 arrivals — the
#: Ukraine parole pool saturates below its admin count) or collapse. A
#: category with an explicit zero target must emit exactly zero.
_HUMANITARIAN_TARGET_RELATIVE_BAND = (0.35, 1.5)


@dataclass(frozen=True)
class UndocumentedControls:
    """Published totals the stage forces or gates against.

    Attributes:
        workers: Pew unauthorized-immigrant labor-force control the EAD spill
            enforces, including DACA and temporary protections Pew retains.
        students: Broad undocumented-student control the EAD spill enforces.
        population_anchor: Published Pew unauthorized population the
            composition gate checks the *emergent broad-universe* total
            against (never forced in the labels).
        sources: Citation URL per key, straight from the manifest.
    """

    workers: float
    students: float
    population_anchor: float
    sources: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in ("workers", "students", "population_anchor"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Undocumented control {name!r} must be positive, got {value!r}."
                )
        missing = [
            key
            for key in (*_FORCED_CONTROL_KEYS, _ANCHOR_KEY)
            if not self.sources.get(key)
        ]
        if missing:
            raise ValueError(
                f"Undocumented control(s) missing a source citation: {missing}."
            )


@dataclass(frozen=True)
class HumanitarianDraw:
    """One seeded humanitarian draw: a cited stock the stage selects toward.

    Attributes:
        category: Manifest category key (``paroled_one_year`` … ``tps``).
        origin: Per-origin key inside the category, or ``None`` for a
            national draw.
        status: ``ImmigrationStatus`` enum member name the draw emits.
        target: Cited weighted stock. Zero is allowed and means the
            category is explicitly not imputed (the citation documents why).
        source: Citation URL or reference for the target.
    """

    category: str
    origin: str | None
    status: str
    target: float
    source: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.target) or self.target < 0:
            raise ValueError(
                f"Humanitarian target {self.label!r} must be non-negative, "
                f"got {self.target!r}."
            )
        if not self.source:
            raise ValueError(
                f"Humanitarian target {self.label!r} requires a source citation."
            )

    @property
    def label(self) -> str:
        return (
            self.category if self.origin is None else f"{self.category}:{self.origin}"
        )

    @property
    def salt(self) -> str:
        return f"immigration:{self.label}"


@dataclass(frozen=True)
class ImmigrationControls:
    """Every manifest-sourced control the immigration stage consumes.

    Attributes:
        undocumented: The legacy forced margins and the composition-gate
            anchor for the residual pool.
        humanitarian: The humanitarian draws in assignment order — the
            per-origin draws of a category are adjacent, categories follow
            ``HUMANITARIAN_STATUS_CATEGORIES``.
    """

    undocumented: UndocumentedControls
    humanitarian: tuple[HumanitarianDraw, ...]

    def humanitarian_target(self, category: str) -> float:
        """Summed cited target for one category across its origins."""

        return float(
            sum(draw.target for draw in self.humanitarian if draw.category == category)
        )


@dataclass(frozen=True)
class _ImmigrationEvidenceProfile:
    """Canonical immigration evidence shared by ASEC and ACS rows."""

    source_is_acs: np.ndarray
    is_citizen: np.ndarray
    birth_country: np.ndarray
    arrival_code: np.ndarray
    arrival_year: np.ndarray
    age: np.ndarray
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
    """Assign ``ssn_card_type`` and ``immigration_status_str`` to persons.

    The current frame must be the raw-column person table (from the stage's
    ``read_table`` operation) carrying a ``person_weight`` column materialized
    from the bundle's household weights.
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
    controls = _controls_from_parameters(params)

    missing = [
        column
        for column in (*US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS, _PERSON_WEIGHT_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            "US immigration derivation is missing required person column(s): "
            f"{missing}."
        )

    result = frame.copy(deep=True)
    weights = pd.to_numeric(result[_PERSON_WEIGHT_COLUMN], errors="coerce")
    if weights.isna().any() or (weights < 0).any():
        raise SourceRuntimeError(
            "US immigration derivation requires finite non-negative person weights."
        )
    ssn_codes, humanitarian_marks = _assign_ssn_card_codes(
        result,
        weights.to_numpy(dtype=np.float64),
        seed=int(context.config.seed),
        controls=controls,
        time_period=int(context.config.target_year),
    )
    status = _derive_immigration_status(
        result,
        ssn_codes,
        humanitarian_marks,
        time_period=int(context.config.target_year),
    )
    code_to_name = {
        0: "NONE",
        1: "CITIZEN",
        2: "NON_CITIZEN_VALID_EAD",
        3: "OTHER_NON_CITIZEN",
    }
    result["ssn_card_type"] = pd.Series(ssn_codes, index=result.index).map(code_to_name)
    result["immigration_status_str"] = status
    return result


def _target_and_source(
    block: object,
    *,
    label: str,
    value_key: str,
    allow_zero: bool,
) -> tuple[float, str]:
    if not isinstance(block, Mapping):
        raise SourceRuntimeError(
            f"US immigration control {label!r} requires an object with "
            f"{value_key!r} and 'source'."
        )
    unexpected = sorted(set(block) - {value_key, "source"})
    if unexpected:
        raise SourceRuntimeError(
            f"US immigration control {label!r} has unsupported key(s): {unexpected}."
        )
    value = block.get(value_key)
    source = block.get("source")
    valid_number = isinstance(value, int | float) and not isinstance(value, bool)
    if not valid_number or (float(value) <= 0 and not (allow_zero and value == 0)):
        requirement = "non-negative" if allow_zero else "positive"
        raise SourceRuntimeError(
            f"US immigration control {label!r} requires a {requirement} numeric "
            f"{value_key!r}."
        )
    if not isinstance(source, str) or not source:
        raise SourceRuntimeError(
            f"US immigration control {label!r} requires a source citation."
        )
    return float(value), source


def _humanitarian_draws_from_parameters(
    params: Mapping[str, object],
) -> tuple[HumanitarianDraw, ...]:
    block = params.get(_HUMANITARIAN_KEY)
    if not isinstance(block, Mapping):
        raise SourceRuntimeError(
            f"US immigration derivation requires a {_HUMANITARIAN_KEY!r} object "
            f"with one block per category {list(HUMANITARIAN_STATUS_CATEGORIES)}."
        )
    unexpected = sorted(set(block) - set(HUMANITARIAN_STATUS_CATEGORIES))
    if unexpected:
        raise SourceRuntimeError(
            f"US immigration {_HUMANITARIAN_KEY} has unsupported category(ies): "
            f"{unexpected}."
        )
    missing = sorted(set(HUMANITARIAN_STATUS_CATEGORIES) - set(block))
    if missing:
        raise SourceRuntimeError(
            f"US immigration {_HUMANITARIAN_KEY} is missing category(ies): {missing}."
        )
    draws: list[HumanitarianDraw] = []
    for category in HUMANITARIAN_STATUS_CATEGORIES:
        status = _HUMANITARIAN_STATUS_BY_CATEGORY[category]
        category_block = block[category]
        if category in _PER_ORIGIN_HUMANITARIAN_CATEGORIES:
            origin_keys = _PER_ORIGIN_HUMANITARIAN_CATEGORIES[category]
            if not isinstance(category_block, Mapping):
                raise SourceRuntimeError(
                    f"US immigration control {category!r} requires per-origin "
                    f"blocks {list(origin_keys)}."
                )
            unexpected_origins = sorted(set(category_block) - set(origin_keys))
            if unexpected_origins:
                raise SourceRuntimeError(
                    f"US immigration control {category!r} has origin(s) outside "
                    f"the stage's codebook table: {unexpected_origins}."
                )
            missing_origins = sorted(set(origin_keys) - set(category_block))
            if missing_origins:
                raise SourceRuntimeError(
                    f"US immigration control {category!r} is missing origin(s): "
                    f"{missing_origins}."
                )
            for origin in origin_keys:
                target, source = _target_and_source(
                    category_block[origin],
                    label=f"{category}:{origin}",
                    value_key="target",
                    allow_zero=True,
                )
                draws.append(
                    HumanitarianDraw(
                        category=category,
                        origin=origin,
                        status=status,
                        target=target,
                        source=source,
                    )
                )
        else:
            target, source = _target_and_source(
                category_block,
                label=category,
                value_key="target",
                allow_zero=True,
            )
            draws.append(
                HumanitarianDraw(
                    category=category,
                    origin=None,
                    status=status,
                    target=target,
                    source=source,
                )
            )
    return tuple(draws)


def _controls_from_parameters(params: Mapping[str, object]) -> ImmigrationControls:
    values: dict[str, float] = {}
    sources: dict[str, str] = {}
    for key, value_key in (
        ("undocumented_workers", "target"),
        ("undocumented_students", "target"),
        (_ANCHOR_KEY, "value"),
    ):
        value, source = _target_and_source(
            params.get(key), label=key, value_key=value_key, allow_zero=False
        )
        values[key] = value
        sources[key] = source
    undocumented = UndocumentedControls(
        workers=values["undocumented_workers"],
        students=values["undocumented_students"],
        population_anchor=values[_ANCHOR_KEY],
        sources=sources,
    )
    return ImmigrationControls(
        undocumented=undocumented,
        humanitarian=_humanitarian_draws_from_parameters(params),
    )


def us_immigration_controls() -> ImmigrationControls:
    """Return the controls bound to the packaged immigration manifest stage."""

    derive = [
        operation
        for operation in us_immigration_stage_spec().operations
        if operation.kind == "derive_immigration_status"
    ]
    if len(derive) != 1:
        raise ValueError(
            "US immigration stage must declare exactly one "
            "derive_immigration_status operation."
        )
    return _controls_from_parameters(derive[0].parameters)


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
    PENATVTY, so the normalized origin is lossless; ACS YOEP retains its exact
    year while ASEC necessarily uses the documented interval midpoint.
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
    for code, midpoint in _ARRIVAL_YEAR_MIDPOINTS.items():
        arrival_year[has_asec & (arrival_code == code)] = midpoint

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
    age_at_entry = np.maximum(0, age - (time_period - arrival_year))
    tps_acs_max_arrival_year = np.full(len(person), -1, dtype=np.int64)
    for birth_code, cutoff in _TPS_ACS_MAX_ARRIVAL_YEAR_BY_BIRTH.items():
        tps_acs_max_arrival_year[birth_country == birth_code] = cutoff
    return _ImmigrationEvidenceProfile(
        source_is_acs=has_acs,
        is_citizen=is_citizen,
        birth_country=birth_country,
        arrival_code=arrival_code,
        arrival_year=arrival_year,
        age=age,
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


def _select_weight_to_target(
    candidates: np.ndarray,
    weights: np.ndarray,
    draws: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Boolean mask selecting ~``amount`` of weighted mass from candidates.

    Selection is threshold-based: candidates are ranked by draw, and every
    candidate at or below the draw where the cumulative weight first reaches
    ``amount`` is selected. Records sharing a draw — support-channel clones
    of one source person share theirs by construction — land on the same
    side of the threshold, so a clone pair is never split; the final tie
    group may overshoot the amount. A non-positive ``amount`` selects
    nothing — a count already at or below its control spills nothing.
    """

    selected = np.zeros(len(weights), dtype=bool)
    if amount <= 0:
        return selected
    indices = np.flatnonzero(candidates)
    if indices.size == 0:
        return selected
    order = np.argsort(draws[indices], kind="stable")
    ordered = indices[order]
    cumulative = np.cumsum(weights[ordered])
    crossing = int(np.searchsorted(cumulative, amount))
    if crossing >= ordered.size:
        selected[ordered] = True
        return selected
    threshold = draws[ordered[crossing]]
    chosen = indices[draws[indices] <= threshold]
    selected[chosen] = True
    return selected


def _arrival_profile(
    person: pd.DataFrame, *, time_period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(PEINUSYR code, arrival-year midpoint, age at entry) per person."""

    arrival_code = _integer_column(person, "PEINUSYR")
    arrival_year = np.full(len(person), time_period, dtype=np.int64)
    for code, midpoint in _ARRIVAL_YEAR_MIDPOINTS.items():
        arrival_year[arrival_code == code] = midpoint
    years_in_us = time_period - arrival_year
    age = _integer_column(person, "A_AGE")
    age_at_entry = np.maximum(0, age - years_in_us)
    return arrival_code, arrival_year, age_at_entry


def _daca_statutory_cohort(
    arrival_year: np.ndarray, age_at_entry: np.ndarray, age: np.ndarray
) -> np.ndarray:
    return (
        (arrival_year <= _DACA_LATEST_ARRIVAL_YEAR)
        & (age_at_entry < _DACA_MAX_AGE_AT_ENTRY)
        & (age >= _DACA_MIN_CURRENT_AGE)
    )


def _special_status_masks(
    *,
    ssn_codes: np.ndarray,
    birth_country: np.ndarray,
    arrival_year: np.ndarray,
    age_at_entry: np.ndarray,
    age: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return final Cuban/Haitian-entrant and DACA cohort masks.

    The source derivation applies DACA after the Cuban/Haitian entrant rule,
    so a qualifying EAD holder in both cohorts retains DACA. Keeping that
    precedence in one helper lets source derivation, recipient reconciliation,
    and release validation share the same exact contract.
    """

    documented_noncitizen = np.isin(ssn_codes, [2, 3])
    daca = (ssn_codes == 2) & _daca_statutory_cohort(
        arrival_year,
        age_at_entry,
        age,
    )
    cuban_haitian = (
        documented_noncitizen
        & np.isin(birth_country, _CUBAN_HAITIAN_BIRTH_CODES)
        & (arrival_year >= _CUBAN_HAITIAN_ARRIVAL_CUTOFF)
        & ~daca
    )
    return cuban_haitian, daca


def _pew_unauthorized_projection_mask(
    *,
    ssn_codes: np.ndarray,
    humanitarian_marks: np.ndarray,
    retained_ead_cohort: np.ndarray,
) -> np.ndarray:
    """Project final membership in Pew's broad unauthorized universe.

    This helper runs while EAD assignment is still in progress. Residual
    ``NONE`` rows count directly; DACA and residual Cuban/Haitian cohort rows
    continue to count if an EAD spill gives them a protected engine label;
    and the temporary-protection statuses Pew explicitly retains count
    regardless of their non-``NONE`` SSN code.
    """

    return (
        (ssn_codes == 0)
        | ((ssn_codes == 2) & retained_ead_cohort)
        | np.isin(
            humanitarian_marks,
            _PEW_INCLUDED_HUMANITARIAN_STATUS_VALUES,
        )
    )


def _spill_pew_unauthorized_excess(
    person: pd.DataFrame,
    ssn_codes: np.ndarray,
    humanitarian_marks: np.ndarray,
    weights: np.ndarray,
    *,
    noncitizens: np.ndarray,
    scope: np.ndarray,
    preserve_scope: np.ndarray,
    retained_ead_cohort: np.ndarray,
    target: float,
    seed: int,
    salt: str,
) -> None:
    """Spill a broad Pew-universe margin to EAD in deterministic order.

    The first draw preserves the prior EAD/DACA allocation behavior by
    selecting from the whole residual scope. A selected DACA or residual
    Cuban/Haitian cohort row still belongs to Pew's universe, so a second pass
    spills additional rows outside those retained cohorts when needed. Within
    each pass, rows outside the other controlled margin are selected first;
    this keeps the student and worker controls from disturbing one another
    unless their overlap makes that unavoidable.
    """

    draws = _stable_person_draws(person, seed=seed, salt=salt)

    def current_excess() -> float:
        included = _pew_unauthorized_projection_mask(
            ssn_codes=ssn_codes,
            humanitarian_marks=humanitarian_marks,
            retained_ead_cohort=retained_ead_cohort,
        )
        return float(weights[scope & included].sum()) - target

    # Preserve the existing seeded EAD allocation surface. DACA selections
    # do not reduce the Pew count and are compensated by the corrective pass.
    initial_candidates = (ssn_codes == 0) & noncitizens & scope
    initial_excess = current_excess()
    if initial_excess > 0:
        for priority in (~preserve_scope, preserve_scope):
            selected = _select_weight_to_target(
                initial_candidates & priority,
                weights,
                draws,
                current_excess(),
            )
            ssn_codes[selected] = 2
            if current_excess() <= 0:
                return

    # DACA and residual Cuban/Haitian cohort members retain protected engine
    # labels after EAD assignment and therefore remain inside Pew's estimate.
    # Only rows outside those cohorts can close any remaining gap.
    for priority in (~preserve_scope, preserve_scope):
        corrective_candidates = (
            (ssn_codes == 0) & noncitizens & scope & ~retained_ead_cohort & priority
        )
        selected = _select_weight_to_target(
            corrective_candidates,
            weights,
            draws,
            current_excess(),
        )
        ssn_codes[selected] = 2
        if current_excess() <= 0:
            return


def _humanitarian_draw_candidates(
    draw: HumanitarianDraw,
    *,
    ssn_codes: np.ndarray,
    profile: _ImmigrationEvidenceProfile,
) -> np.ndarray:
    """Candidate mask for one draw, before exclusions shared by all draws.

    ``REFUGEE``/``ASYLEE``/``DEPORTATION_WITHHELD`` draw from the
    indicator-documented pool only (code 3: those statuses carry immediate
    federal benefits access, the same signals the ASEC-UA method reads);
    parole and TPS may also draw from the still-unspilled residual pool
    (code 0), whose members then receive EAD-backed SSN cards.
    """

    noncitizen = ~profile.is_citizen
    excluded = np.isin(profile.birth_country, _CUBAN_HAITIAN_BIRTH_CODES)
    excluded |= _daca_statutory_cohort(
        profile.arrival_year,
        profile.age_at_entry,
        profile.age,
    )

    if draw.category == "paroled_one_year":
        origins = (
            (draw.origin,) if draw.origin is not None else tuple(_PAROLE_ORIGIN_CODES)
        )
        cohort = np.zeros(len(ssn_codes), dtype=bool)
        for origin in origins:
            if origin not in _PAROLE_ORIGIN_CODES:
                raise SourceRuntimeError(
                    f"US immigration has no parole origin rule for {origin!r}."
                )
            origin_match = np.isin(
                profile.birth_country,
                _PAROLE_ORIGIN_CODES[origin],
            )
            asec_window = (~profile.source_is_acs) & np.isin(
                profile.arrival_code,
                _PAROLE_ASEC_ARRIVAL_CODES[origin],
            )
            acs_window = profile.source_is_acs & (
                profile.arrival_year >= _PAROLE_ACS_MIN_ARRIVAL_YEAR[origin]
            )
            cohort |= origin_match & (asec_window | acs_window)
        return noncitizen & np.isin(ssn_codes, (0, 3)) & cohort & ~excluded
    if draw.category == "refugee":
        return (
            noncitizen
            & (ssn_codes == 3)
            & np.isin(profile.birth_country, _REFUGEE_ORIGIN_CODES)
            & (
                ((~profile.source_is_acs) & (profile.arrival_code == 28))
                | (profile.source_is_acs & (profile.arrival_year >= 2022))
            )
            & ~excluded
        )
    if draw.category == "asylee":
        return (
            noncitizen
            & (ssn_codes == 3)
            & np.isin(profile.birth_country, _ASYLEE_ORIGIN_CODES)
            & (
                (
                    (~profile.source_is_acs)
                    & np.isin(profile.arrival_code, _ASYLEE_ARRIVAL_CODES)
                )
                | (profile.source_is_acs & (profile.arrival_year >= 2016))
            )
            & ~excluded
        )
    if draw.category == "deportation_withheld":
        return (
            noncitizen
            & (ssn_codes == 3)
            & (
                (
                    (~profile.source_is_acs)
                    & (profile.arrival_code >= 1)
                    & (profile.arrival_code <= _WITHHELD_MAX_ARRIVAL_CODE)
                )
                | (profile.source_is_acs & (profile.arrival_year <= 2019))
            )
            & ~excluded
        )
    if draw.category == "tps":
        origins = (
            (draw.origin,) if draw.origin is not None else tuple(_TPS_ORIGIN_CODES)
        )
        cohort = np.zeros(len(ssn_codes), dtype=bool)
        for origin in origins:
            if origin not in _TPS_ORIGIN_CODES:
                raise SourceRuntimeError(
                    f"US immigration has no TPS origin rule for {origin!r}."
                )
            codes, max_arrival_code = _TPS_ORIGIN_CODES[origin]
            origin_match = np.isin(profile.birth_country, codes)
            asec_window = (
                (~profile.source_is_acs)
                & (profile.arrival_code >= 1)
                & (profile.arrival_code <= max_arrival_code)
            )
            acs_window = (
                profile.source_is_acs
                & (profile.tps_acs_max_arrival_year >= 0)
                & (profile.arrival_year <= profile.tps_acs_max_arrival_year)
            )
            cohort |= origin_match & (asec_window | acs_window)
        return noncitizen & np.isin(ssn_codes, (0, 3)) & cohort & ~excluded
    raise SourceRuntimeError(
        f"US immigration derivation has no candidate rule for draw {draw.label!r}."
    )


def us_immigration_humanitarian_draw_mask(
    frame: Frame,
    draw: HumanitarianDraw,
    *,
    time_period: int = 2024,
) -> np.ndarray:
    """Rows emitted as ``draw`` whose source evidence supports that label.

    This is the shared hard-cohort contract for transfer reconciliation,
    calibration, and the release gate. It intentionally includes the emitted
    status test: callers receive the achieved population for one manifest
    draw, not the larger pool of possible candidates.
    """

    person = frame.table(frame.schema.person_entity)
    missing = sorted(set(US_IMMIGRATION_OUTPUT_COLUMNS) - set(person.columns))
    if missing:
        raise ValueError(
            f"US humanitarian draw compatibility requires output column(s) {missing}."
        )
    profile = _source_aware_immigration_profile(person, time_period=time_period)
    return _humanitarian_emitted_mask(person, draw=draw, profile=profile)


def us_immigration_humanitarian_transfer_selection_masks(
    frame: Frame,
    *,
    mutable_rows: np.ndarray,
    seed: int,
    time_period: int = 2024,
    controls: ImmigrationControls | None = None,
) -> dict[str, np.ndarray]:
    """Replay the exact deterministic mutable-row selection for every draw.

    The post-transfer frame retains everything needed to reconstruct the
    reconciliation decision: source evidence, final SSN codes, stable person
    lineage, resolved person weights, manifest controls, and draw order.  For
    parole and TPS, final EAD codes are mapped back to the residual-pool code
    exactly as they are during reconciliation.  Earlier expected selections,
    rather than observed status labels, exclude rows from later draws, so a
    downstream equal-mass status swap cannot redefine the replayed candidate
    surface.
    """

    person = frame.table(frame.schema.person_entity)
    missing = sorted(set(US_IMMIGRATION_OUTPUT_COLUMNS) - set(person.columns))
    if missing:
        raise ValueError(
            "Humanitarian transfer selection replay requires person column(s) "
            f"{missing}."
        )
    mutable = np.asarray(mutable_rows, dtype=bool)
    if mutable.shape != (len(person),):
        raise ValueError(
            "Humanitarian transfer selection replay mutable_rows must align "
            "one-to-one with the person table."
        )
    weights = np.asarray(
        frame.resolve_weights(frame.schema.person_entity).values,
        dtype=np.float64,
    )
    if (
        weights.shape != (len(person),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError(
            "Humanitarian transfer selection replay requires finite "
            "non-negative person weights."
        )
    controls = us_immigration_controls() if controls is None else controls
    profile = _source_aware_immigration_profile(person, time_period=time_period)
    ssn_codes = _ssn_name_codes(person)
    immutable = ~mutable
    selected_once = np.zeros(len(person), dtype=bool)
    selections: dict[str, np.ndarray] = {}
    for draw in controls.humanitarian:
        immutable_draw = (
            _humanitarian_emitted_mask(
                person,
                draw=draw,
                profile=profile,
            )
            & immutable
        )
        residual_target = max(
            0.0,
            float(draw.target) - float(weights[immutable_draw].sum()),
        )
        candidate_ssn = ssn_codes
        if draw.category in {"paroled_one_year", "tps"}:
            candidate_ssn = np.where(candidate_ssn == 2, 0, candidate_ssn)
        candidates = (
            _humanitarian_draw_candidates(
                draw,
                ssn_codes=candidate_ssn,
                profile=profile,
            )
            & mutable
            & ~selected_once
        )
        stable_draws = _stable_person_draws(
            person,
            seed=seed,
            salt=f"acs_transfer:{draw.salt}",
        )
        selected = _select_weight_to_target(
            candidates,
            weights,
            stable_draws,
            residual_target,
        )
        selections[draw.label] = selected
        selected_once |= selected
    return selections


def _ssn_name_codes(person: pd.DataFrame) -> np.ndarray:
    ssn_lookup = {
        "NONE": 0,
        "CITIZEN": 1,
        "NON_CITIZEN_VALID_EAD": 2,
        "OTHER_NON_CITIZEN": 3,
    }
    return (
        person["ssn_card_type"]
        .astype(str)
        .map(ssn_lookup)
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )


def _humanitarian_emitted_mask(
    person: pd.DataFrame,
    *,
    draw: HumanitarianDraw,
    profile: _ImmigrationEvidenceProfile,
) -> np.ndarray:
    ssn_codes = _ssn_name_codes(person)
    candidates = _humanitarian_draw_candidates(
        draw,
        ssn_codes=ssn_codes,
        profile=profile,
    )
    # Parole and TPS selections from the residual pool are upgraded from NONE
    # to EAD after candidacy is evaluated. Their emitted rows may therefore
    # carry code 2 even though only code 0/3 was eligible before selection.
    if draw.category in {"paroled_one_year", "tps"}:
        candidates |= _humanitarian_draw_candidates(
            draw,
            ssn_codes=np.where(ssn_codes == 2, 0, ssn_codes),
            profile=profile,
        ) & (ssn_codes == 2)
    status = person["immigration_status_str"].astype(str).to_numpy()
    return np.asarray(candidates & (status == draw.status), dtype=bool)


def reconcile_us_immigration_humanitarian_transfer(
    person: pd.DataFrame,
    *,
    weights: np.ndarray,
    mutable_rows: np.ndarray,
    seed: int,
    time_period: int = 2024,
    controls: ImmigrationControls | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reconcile a transferred immigration pair to manifest draw targets.

    Immutable rows are the ASEC source contribution. Only rows whose paired
    target cells were imputed may be repaired or selected. Each manifest draw
    receives the residual of its national target after its compatible,
    immutable contribution, in manifest order; candidate exhaustion is a hard
    error rather than permission to ship a wrong-origin label.
    """

    result = person.copy()
    missing = sorted(set(US_IMMIGRATION_OUTPUT_COLUMNS) - set(result.columns))
    if missing:
        raise ValueError(
            f"Humanitarian transfer reconciliation requires person column(s) {missing}."
        )
    weights = np.asarray(weights, dtype=np.float64)
    mutable = np.asarray(mutable_rows, dtype=bool)
    if weights.shape != (len(result),) or mutable.shape != (len(result),):
        raise ValueError(
            "Humanitarian transfer weights and mutable_rows must align one-to-one "
            "with the person table."
        )
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(
            "Humanitarian transfer reconciliation requires finite non-negative "
            "person weights."
        )
    controls = us_immigration_controls() if controls is None else controls
    profile = _source_aware_immigration_profile(result, time_period=time_period)
    original_ssn = result["ssn_card_type"].astype(str).to_numpy(copy=True)
    original_status = result["immigration_status_str"].astype(str).to_numpy(copy=True)
    ssn = original_ssn.copy()
    status = original_status.copy()
    valid_ssn = np.isin(ssn, SSN_CARD_TYPE_VALUES)
    valid_status = np.isin(status, IMMIGRATION_STATUS_VALUES)
    if (~valid_ssn).any() or (~valid_status).any():
        raise ValueError(
            "Humanitarian transfer reconciliation received values outside the "
            "PolicyEngine SSN/immigration enum domains."
        )

    humanitarian_statuses = set(_HUMANITARIAN_STATUS_BY_CATEGORY.values())
    evidence_derived_statuses = {"CUBAN_HAITIAN_ENTRANT", "DACA"}
    constrained_statuses = humanitarian_statuses | evidence_derived_statuses
    baseline_constrained = mutable & np.isin(status, list(constrained_statuses))
    if baseline_constrained.any():
        raise ValueError(
            "ACS joint-QRF baseline emitted evidence-constrained labels before "
            "reconciliation."
        )

    # Source citizenship is observed on both surveys. Normalize only mutable
    # pairs; immutable ASEC disagreements remain an error below.
    citizen_mutable = mutable & profile.is_citizen
    ssn[citizen_mutable] = "CITIZEN"
    status[citizen_mutable] = "CITIZEN"
    noncitizen_mutable = mutable & ~profile.is_citizen
    ssn[noncitizen_mutable & (ssn == "CITIZEN")] = "OTHER_NON_CITIZEN"
    status[noncitizen_mutable & (status == "CITIZEN")] = "LEGAL_PERMANENT_RESIDENT"
    status[noncitizen_mutable & (ssn == "NONE")] = "UNDOCUMENTED"
    status[noncitizen_mutable & (ssn != "NONE") & (status == "UNDOCUMENTED")] = (
        "LEGAL_PERMANENT_RESIDENT"
    )
    citizenship_repairs = int(
        (
            mutable
            & ((ssn != original_ssn) | (status != original_status))
            & (
                profile.is_citizen
                | (original_ssn == "CITIZEN")
                | (original_status == "CITIZEN")
            )
        ).sum()
    )
    pair_repairs = int(
        (mutable & ((ssn != original_ssn) | (status != original_status))).sum()
    )
    result["ssn_card_type"] = ssn
    result["immigration_status_str"] = status

    # CHE and DACA have hard source-evidence rules just like the manifest
    # draws, but no stock target. Rebuild them deterministically after the
    # unconstrained codec and citizenship/pair repair rather than accepting a
    # donor label on an incompatible ACS cohort.
    special_ssn_codes = _ssn_name_codes(result)
    cuban_haitian, daca = _special_status_masks(
        ssn_codes=special_ssn_codes,
        birth_country=profile.birth_country,
        arrival_year=profile.arrival_year,
        age_at_entry=profile.age_at_entry,
        age=profile.age,
    )
    special_before = status.copy()
    status[mutable & cuban_haitian] = "CUBAN_HAITIAN_ENTRANT"
    status[mutable & daca] = "DACA"
    special_status_assignments = int((mutable & (status != special_before)).sum())
    result["immigration_status_str"] = status

    immutable = ~mutable
    immutable_pair_invalid = immutable & (
        ((ssn == "CITIZEN") != (status == "CITIZEN"))
        | ((ssn == "NONE") != (status == "UNDOCUMENTED"))
        | ((ssn == "CITIZEN") != profile.is_citizen)
    )
    if immutable_pair_invalid.any():
        raise ValueError(
            "Immutable ASEC immigration rows violate citizenship/paired-status "
            f"invariants; invalid_rows={int(immutable_pair_invalid.sum())}."
        )
    immutable_special_invalid = immutable & (
        ((status == "CUBAN_HAITIAN_ENTRANT") != cuban_haitian)
        | ((status == "DACA") != daca)
    )
    if immutable_special_invalid.any():
        raise ValueError(
            "Immutable ASEC immigration rows violate Cuban/Haitian entrant or "
            f"DACA cohort invariants; invalid_rows={int(immutable_special_invalid.sum())}."
        )

    selected_once = np.zeros(len(result), dtype=bool)
    compatible_humanitarian = np.zeros(len(result), dtype=bool)
    draw_receipts: dict[str, object] = {}
    largest_target = max((draw.target for draw in controls.humanitarian), default=0.0)
    tolerance = max(1e-6, largest_target * 1e-12)
    for draw in controls.humanitarian:
        immutable_draw = (
            _humanitarian_emitted_mask(
                result,
                draw=draw,
                profile=profile,
            )
            & immutable
        )
        compatible_humanitarian |= immutable_draw
        immutable_population = float(weights[immutable_draw].sum())
        if draw.target <= 0 and immutable_population > tolerance:
            raise ValueError(
                f"Immutable ASEC rows emit {immutable_population:,.6f} weighted "
                f"persons for explicit-zero humanitarian draw {draw.label!r}."
            )
        residual_target = max(0.0, float(draw.target) - immutable_population)

        candidate_ssn = _ssn_name_codes(result)
        if draw.category in {"paroled_one_year", "tps"}:
            candidate_ssn = np.where(candidate_ssn == 2, 0, candidate_ssn)
        candidates = (
            _humanitarian_draw_candidates(
                draw,
                ssn_codes=candidate_ssn,
                profile=profile,
            )
            & mutable
            & ~selected_once
        )
        available_population = float(weights[candidates].sum())
        if available_population + tolerance < residual_target:
            raise ValueError(
                "Humanitarian transfer candidate shortfall for "
                f"{draw.label!r}: residual_target={residual_target:,.6f}, "
                f"eligible_recipient_population={available_population:,.6f}, "
                f"immutable_population={immutable_population:,.6f}."
            )
        stable_draws = _stable_person_draws(
            result,
            seed=seed,
            salt=f"acs_transfer:{draw.salt}",
        )
        selected = _select_weight_to_target(
            candidates,
            weights,
            stable_draws,
            residual_target,
        )
        selected_population = float(weights[selected].sum())
        if selected_population + tolerance < residual_target:
            raise RuntimeError(
                f"Humanitarian transfer selection underfilled {draw.label!r}."
            )
        status[selected] = draw.status
        if draw.category in {"paroled_one_year", "tps"}:
            ssn[selected & (ssn == "NONE")] = "NON_CITIZEN_VALID_EAD"
        selected_once |= selected
        result["ssn_card_type"] = ssn
        result["immigration_status_str"] = status
        achieved_mask = _humanitarian_emitted_mask(
            result,
            draw=draw,
            profile=profile,
        )
        compatible_humanitarian |= achieved_mask
        achieved_population = float(weights[achieved_mask].sum())
        threshold_tie_population = 0.0
        if selected.any():
            threshold = float(np.max(stable_draws[selected]))
            threshold_tie_population = float(
                weights[candidates & (stable_draws == threshold)].sum()
            )
        residual_selection_error = abs(selected_population - residual_target)
        if residual_selection_error > threshold_tie_population + tolerance:
            raise RuntimeError(
                f"Humanitarian transfer residual selection for {draw.label!r} "
                "missed its target by more than the threshold tie mass."
            )
        absolute_error = abs(achieved_population - float(draw.target))
        immutable_overshoot = max(0.0, immutable_population - float(draw.target))
        discrete_bound = immutable_overshoot + threshold_tie_population + tolerance
        draw_receipts[draw.label] = {
            "target": float(draw.target),
            "immutable_population": immutable_population,
            "residual_target": residual_target,
            "eligible_recipient_population": available_population,
            "selected_recipient_population": selected_population,
            "achieved_population": achieved_population,
            "absolute_error": absolute_error,
            "selection_threshold_tie_population": threshold_tie_population,
            "residual_selection_error": residual_selection_error,
            "within_residual_discrete_weight_bound": True,
            "immutable_overshoot": immutable_overshoot,
            "within_discrete_weight_bound": absolute_error <= discrete_bound,
        }
        if absolute_error > discrete_bound:
            raise RuntimeError(
                f"Humanitarian transfer reconciliation for {draw.label!r} "
                "missed its target by more than the discrete selection bound."
            )

    result["ssn_card_type"] = ssn
    result["immigration_status_str"] = status
    all_humanitarian = np.isin(status, list(humanitarian_statuses))
    incompatible_humanitarian = all_humanitarian & ~compatible_humanitarian
    if incompatible_humanitarian.any():
        raise ValueError(
            "Humanitarian transfer output contains status/origin/arrival/SSN "
            f"incompatible rows: {int(incompatible_humanitarian.sum())}."
        )
    final_ssn_codes = _ssn_name_codes(result)
    final_cuban_haitian, final_daca = _special_status_masks(
        ssn_codes=final_ssn_codes,
        birth_country=profile.birth_country,
        arrival_year=profile.arrival_year,
        age_at_entry=profile.age_at_entry,
        age=profile.age,
    )
    special_status_invalid = (
        (status == "CUBAN_HAITIAN_ENTRANT") != final_cuban_haitian
    ) | ((status == "DACA") != final_daca)
    if special_status_invalid.any():
        raise RuntimeError(
            "Humanitarian transfer reconciliation left Cuban/Haitian entrant "
            "or DACA cohort invariants invalid on "
            f"{int(special_status_invalid.sum())} row(s)."
        )
    final_pair_invalid = (
        ((ssn == "CITIZEN") != (status == "CITIZEN"))
        | ((ssn == "NONE") != (status == "UNDOCUMENTED"))
        | ((ssn == "CITIZEN") != profile.is_citizen)
    )
    if final_pair_invalid.any():
        raise RuntimeError(
            "Humanitarian transfer reconciliation left citizenship/paired-status "
            f"invariants invalid on {int(final_pair_invalid.sum())} row(s)."
        )
    if not np.array_equal(
        result.loc[immutable, "ssn_card_type"].astype(str).to_numpy(),
        original_ssn[immutable],
    ) or not np.array_equal(
        result.loc[immutable, "immigration_status_str"].astype(str).to_numpy(),
        original_status[immutable],
    ):
        raise RuntimeError(
            "Humanitarian transfer reconciliation changed immutable ASEC rows."
        )
    receipt: dict[str, object] = {
        "kind": "deterministic_humanitarian_residual_target",
        "seed": int(seed),
        "time_period": int(time_period),
        "mutable_rows": int(mutable.sum()),
        "immutable_rows": int(immutable.sum()),
        "citizenship_repairs": citizenship_repairs,
        "pair_repairs": pair_repairs,
        "special_status_assignments": special_status_assignments,
        "floating_tolerance": tolerance,
        "selection_order": [draw.label for draw in controls.humanitarian],
        "draws": draw_receipts,
    }
    return result, receipt


def _assign_humanitarian_statuses(
    person: pd.DataFrame,
    ssn_codes: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    controls: ImmigrationControls,
    time_period: int,
) -> np.ndarray:
    """Mark humanitarian statuses and upgrade residual draws' SSN codes.

    Runs after the legal-status indicators and before the EAD worker/student
    spill. Pew-included temporary protections remain inside the broad control
    universe even after receiving EADs. Draws are sequential over
    ``controls.humanitarian``; a person marked by an earlier draw is out of
    every later candidate pool. Residual-pool selections (parole/TPS only)
    move to ``NON_CITIZEN_VALID_EAD`` — both programs confer employment
    authorization — keeping the ``NONE`` ⇔ ``UNDOCUMENTED`` invariant.
    """

    marks = np.full(len(person), "", dtype="U24")
    profile = _source_aware_immigration_profile(person, time_period=time_period)
    for draw in controls.humanitarian:
        if draw.target <= 0:
            continue
        candidates = _humanitarian_draw_candidates(
            draw,
            ssn_codes=ssn_codes,
            profile=profile,
        ) & (marks == "")
        draws = _stable_person_draws(person, seed=seed, salt=draw.salt)
        selected = _select_weight_to_target(candidates, weights, draws, draw.target)
        marks[selected] = draw.status
        ssn_codes[selected & (ssn_codes == 0)] = 2
    return marks


def _assign_ssn_card_codes(
    person: pd.DataFrame,
    weights: np.ndarray,
    *,
    seed: int,
    controls: ImmigrationControls,
    time_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    citizenship = _integer_column(person, "PRCITSHP")
    unknown = ~np.isin(citizenship, [1, 2, 3, 4, 5])
    if unknown.any():
        bad = sorted(set(citizenship[unknown].tolist()))[:5]
        raise SourceRuntimeError(
            f"PRCITSHP carries value(s) outside the ASEC domain 1..5: {bad}."
        )
    citizens = np.isin(citizenship, [1, 2, 3, 4])
    noncitizens = citizenship == 5

    ssn_codes = np.zeros(len(person), dtype=np.int64)
    ssn_codes[citizens] = 1

    # ASEC-UA legal-status indicators (Van Hook et al., SSRN 4662801): any
    # one of them moves a non-citizen out of the likely-undocumented pool.
    arrival_code = _integer_column(person, "PEINUSYR")
    arrived_before_1982 = np.isin(arrival_code, _PRE_1982_ARRIVAL_CODES)
    age = _integer_column(person, "A_AGE")
    marital = _integer_column(person, "A_MARITL")
    spouse = _integer_column(person, "A_SPOUSE")
    is_naturalized = citizenship == 4
    has_five_plus_years = np.isin(arrival_code, list(range(8, 27)))
    has_three_plus_years = np.isin(arrival_code, list(range(8, 28)))
    is_married = np.isin(marital, [1, 2]) & (spouse > 0)
    eligible_naturalized = (
        is_naturalized
        & (age >= 18)
        & (has_five_plus_years | (has_three_plus_years & is_married))
    )
    federal_pension = (_integer_column(person, "PEN_SC1") == 3) | (
        _integer_column(person, "PEN_SC2") == 3
    )
    ss_disability = (_integer_column(person, "RESNSS1") == 2) | (
        _integer_column(person, "RESNSS2") == 2
    )
    government_employee = np.isin(_integer_column(person, "PEIO1COW"), [1, 2, 3]) | (
        _integer_column(person, "A_MJOCC") == 11
    )
    military_connected = (_integer_column(person, "PEAFEVER") == 1) | (
        _integer_column(person, "A_MJOCC") == 11
    )
    assumed_documented = (
        arrived_before_1982
        | eligible_naturalized
        | (_integer_column(person, "MCARE") == 1)
        | federal_pension
        | ss_disability
        | (_integer_column(person, "IHSFLG") == 1)
        | (_integer_column(person, "CAID") == 1)
        | (_integer_column(person, "CHAMPVA") == 1)
        | (_integer_column(person, "MIL") == 1)
        | government_employee
        | (_integer_column(person, "SS_YN") == 1)
        | (_float_column(person, "SPM_CAPHOUSESUB") > 0)
        | military_connected
        | (_integer_column(person, "SSI_YN") == 1)
    )
    ssn_codes[(ssn_codes == 0) & assumed_documented] = 3

    # Humanitarian draws run between the indicators and the EAD spill. Pew
    # retains temporary protection in its unauthorized estimate, so those
    # rows contribute to the broad controls even though they receive EADs.
    humanitarian_marks = _assign_humanitarian_statuses(
        person,
        ssn_codes,
        weights,
        seed=seed,
        controls=controls,
        time_period=time_period,
    )

    labor_force_status = _integer_column(person, "A_LFSR")
    invalid_labor_force_status = ~np.isin(
        labor_force_status,
        _CPS_LABOR_FORCE_STATUS_DOMAIN,
    )
    if invalid_labor_force_status.any():
        bad = sorted(set(labor_force_status[invalid_labor_force_status].tolist()))[:5]
        raise SourceRuntimeError(
            "A_LFSR carries value(s) outside the CPS ASEC domain "
            f"{list(_CPS_LABOR_FORCE_STATUS_DOMAIN)}: {bad}."
        )
    is_worker = (age >= 16) & np.isin(
        labor_force_status,
        _CPS_LABOR_FORCE_STATUS_CODES,
    )
    is_student = _integer_column(person, "A_HSCOL") == 2

    _, arrival_year, age_at_entry = _arrival_profile(person, time_period=time_period)
    daca_cohort = _daca_statutory_cohort(arrival_year, age_at_entry, age)
    cuban_haitian_cohort = (
        np.isin(_integer_column(person, "PENATVTY"), _CUBAN_HAITIAN_BIRTH_CODES)
        & (arrival_year >= _CUBAN_HAITIAN_ARRIVAL_CUTOFF)
        & ~daca_cohort
    )
    retained_ead_cohort = daca_cohort | cuban_haitian_cohort

    # The CPS has no work-authorization variable, so the EAD split inside the
    # residual pool is unidentified. The two controls bind the same broad
    # unauthorized universe their publishers report: residual undocumented
    # people plus DACA and Pew-included temporary protections. Student spill
    # runs first; the worker spill runs last and is therefore authoritative
    # for Pew's published labor-force total. Each pass prefers rows outside
    # the other margin so their overlap is disturbed only when unavoidable.
    _spill_pew_unauthorized_excess(
        person,
        ssn_codes,
        humanitarian_marks,
        weights,
        noncitizens=noncitizens,
        scope=is_student,
        preserve_scope=is_worker,
        retained_ead_cohort=retained_ead_cohort,
        target=controls.undocumented.students,
        seed=seed,
        salt="immigration:ead_students",
    )
    _spill_pew_unauthorized_excess(
        person,
        ssn_codes,
        humanitarian_marks,
        weights,
        noncitizens=noncitizens,
        scope=is_worker,
        preserve_scope=is_student,
        retained_ead_cohort=retained_ead_cohort,
        target=controls.undocumented.workers,
        seed=seed,
        salt="immigration:ead_workers",
    )
    return ssn_codes, humanitarian_marks


def _derive_immigration_status(
    person: pd.DataFrame,
    ssn_codes: np.ndarray,
    humanitarian_marks: np.ndarray,
    *,
    time_period: int,
) -> np.ndarray:
    _, arrival_year, age_at_entry = _arrival_profile(person, time_period=time_period)
    age = _integer_column(person, "A_AGE")
    birth_country = _integer_column(person, "PENATVTY")

    # Documented non-citizens default to LPR — the modal true status — and
    # only statutory tests the data can support narrow it. Citizens stay
    # CITIZEN throughout and undocumented is exactly the NONE SSN pool, so
    # the two columns can never disagree.
    status = np.full(len(person), "LEGAL_PERMANENT_RESIDENT", dtype="U32")
    status[ssn_codes == 1] = "CITIZEN"
    status[ssn_codes == 0] = "UNDOCUMENTED"

    cuban_haitian, daca = _special_status_masks(
        ssn_codes=ssn_codes,
        birth_country=birth_country,
        arrival_year=arrival_year,
        age_at_entry=age_at_entry,
        age=age,
    )
    status[cuban_haitian] = "CUBAN_HAITIAN_ENTRANT"
    status[daca] = "DACA"

    # Humanitarian draws exclude Cuba/Haiti-born and the DACA cohort, so
    # the marks are disjoint from both tags; every marked person carries a
    # non-NONE SSN code, so the columns still agree about the undocumented.
    marked = humanitarian_marks != ""
    status[marked] = humanitarian_marks[marked]
    return status


def with_us_immigration_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    person_weight_scale: float = 1.0,
) -> Frame:
    """Run the ``immigration_status`` manifest stage over a US frame.

    Existing output columns are preserved, making the transform idempotent —
    a base H5 built with this stage passes through untouched at release time.

    Args:
        frame: A US-schema frame whose person table still carries the raw
            CPS ASEC source columns.
        seed: Build-wide imputation seed.
        time_period: The dataset's time period (arrival-year arithmetic).
        person_weight_scale: Stage-only multiplier for person design weights.
            Pooled source projections use the inverse of their share of the
            full pool's person mass so absolute national controls contribute
            only that source's mass share. Standalone frames keep the default
            multiplier of one.

    Returns:
        A new frame whose person table carries ``ssn_card_type`` and
        ``immigration_status_str``.

    Raises:
        ValueError: If the frame is not US-schema, one output column exists
            without the other, or the stage output does not cover every
            person.
        SourceRuntimeError: If required raw ASEC columns are missing.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US immigration inputs require the US schema.")
    if (
        isinstance(person_weight_scale, bool)
        or not np.isfinite(person_weight_scale)
        or person_weight_scale <= 0
    ):
        raise ValueError(
            "US immigration person_weight_scale must be a finite positive "
            f"number, got {person_weight_scale!r}."
        )
    person = frame.table("person")
    present = [
        column for column in US_IMMIGRATION_OUTPUT_COLUMNS if column in person.columns
    ]
    if len(present) == len(US_IMMIGRATION_OUTPUT_COLUMNS):
        return frame
    if present:
        missing = sorted(set(US_IMMIGRATION_OUTPUT_COLUMNS) - set(present))
        raise ValueError(
            f"US frame carries {present} without {missing}; a partial "
            "immigration surface would silently default the missing column."
        )

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = np.asarray(
        frame.resolve_weights("person").values, dtype=np.float64
    ) * float(person_weight_scale)
    output = run_source_stage(
        us_immigration_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_immigration_status": derive_us_immigration_status_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_IMMIGRATION_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                f"US immigration stage output does not cover every person for "
                f"{column!r}."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_IMMIGRATION_OUTPUT_COLUMNS:
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
    for column in US_IMMIGRATION_OUTPUT_COLUMNS:
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


def us_immigration_composition_gate(
    frame: Frame,
    *,
    controls: ImmigrationControls | None = None,
) -> GateResult:
    """Release gate: the SSN/immigration surface exists and is plausible.

    Fails when either output column is missing or constant (the #225 failure
    mode: everyone a citizen with a valid SSN), when a value falls outside
    the engine enum domain, when the two columns disagree about citizenship
    or undocumented status, when the weighted non-citizen share leaves its
    plausibility band, when the emergent Pew-defined unauthorized population
    strays outside a coarse band around its cited published anchor, or when a
    humanitarian category's emitted mass leaves the coarse band around its
    cited stock target (microcosm #767 — the H.R.1 §71109/§71301/§71302 and
    SNAP §10108 eligibility channels all bind through these categories; an
    explicit zero target must emit exactly zero).
    """

    if controls is None:
        controls = us_immigration_controls()

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total = float(weights.sum())
    failures: list[str] = []
    details: dict[str, object] = {
        "summary": us_immigration_composition_summary(frame),
        "controls": {
            "undocumented_workers": controls.undocumented.workers,
            "undocumented_students": controls.undocumented.students,
            "undocumented_population_anchor": (controls.undocumented.population_anchor),
            "sources": dict(controls.undocumented.sources),
            "humanitarian_status_stocks": {
                draw.label: {"target": draw.target, "source": draw.source}
                for draw in controls.humanitarian
            },
        },
        "non_citizen_share_band": list(_NON_CITIZEN_SHARE_BAND),
        "undocumented_anchor_relative_band": list(_UNDOCUMENTED_ANCHOR_RELATIVE_BAND),
        "humanitarian_target_relative_band": list(_HUMANITARIAN_TARGET_RELATIVE_BAND),
    }

    missing = [
        column
        for column in US_IMMIGRATION_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        failures.append(
            f"missing person column(s) {missing}: the engine would default "
            "every person to a citizen with a valid SSN."
        )
        return GateResult(
            name="immigration_composition",
            passed=False,
            failures=tuple(failures),
            details=details,
        )

    ssn = person["ssn_card_type"].astype(str)
    status = person["immigration_status_str"].astype(str)
    try:
        evidence = _source_aware_immigration_profile(person, time_period=2024)
    except (SourceRuntimeError, ValueError) as exc:
        evidence = None
        failures.append(f"source-aware immigration evidence is invalid: {exc}")
    for column, values, domain in (
        ("ssn_card_type", ssn, SSN_CARD_TYPE_VALUES),
        ("immigration_status_str", status, IMMIGRATION_STATUS_VALUES),
    ):
        outside = sorted(set(values.unique()) - set(domain))
        if outside:
            failures.append(
                f"{column}: value(s) outside the engine enum domain: {outside}."
            )
        if values.nunique() < 2:
            failures.append(
                f"{column}: constant value {values.iloc[0]!r} — the #225 "
                "failure mode (no SSN/immigration signal)."
            )

    ssn_citizen = (ssn == "CITIZEN").to_numpy()
    status_citizen = (status == "CITIZEN").to_numpy()
    citizen_disagreements = int((ssn_citizen != status_citizen).sum())
    if citizen_disagreements:
        failures.append(
            f"{citizen_disagreements} person(s) have ssn_card_type and "
            "immigration_status_str disagreeing about citizenship."
        )
    ssn_none = (ssn == "NONE").to_numpy()
    status_undocumented = (status == "UNDOCUMENTED").to_numpy()
    undocumented_disagreements = int((ssn_none != status_undocumented).sum())
    if undocumented_disagreements:
        failures.append(
            f"{undocumented_disagreements} person(s) have ssn_card_type NONE "
            "and immigration_status_str UNDOCUMENTED disagreeing."
        )
    if evidence is not None:
        evidence_citizen_disagreements = int((ssn_citizen != evidence.is_citizen).sum())
        if evidence_citizen_disagreements:
            failures.append(
                f"{evidence_citizen_disagreements} person(s) have emitted "
                "citizenship disagreeing with their source PRCITSHP/CIT evidence."
            )
        expected_cuban_haitian, expected_daca = _special_status_masks(
            ssn_codes=_ssn_name_codes(person),
            birth_country=evidence.birth_country,
            arrival_year=evidence.arrival_year,
            age_at_entry=evidence.age_at_entry,
            age=evidence.age,
        )
        special_status_invalid = (
            (status.to_numpy() == "CUBAN_HAITIAN_ENTRANT") != expected_cuban_haitian
        ) | ((status.to_numpy() == "DACA") != expected_daca)
        details["evidence_derived_status_compatibility"] = {
            "cuban_haitian_entrant_rows": int(expected_cuban_haitian.sum()),
            "daca_rows": int(expected_daca.sum()),
            "invalid_rows": int(special_status_invalid.sum()),
        }
        if special_status_invalid.any():
            failures.append(
                f"{int(special_status_invalid.sum())} person(s) violate the "
                "source-aware Cuban/Haitian entrant or DACA cohort contract."
            )

    non_citizen_share = (
        float(weights[~ssn_citizen].sum()) / total if total else float("nan")
    )
    low, high = _NON_CITIZEN_SHARE_BAND
    if not np.isfinite(non_citizen_share) or not (low <= non_citizen_share <= high):
        failures.append(
            f"non-citizen weighted share {non_citizen_share:.4f} outside "
            f"[{low}, {high}]."
        )

    status_values = status.to_numpy()
    ssn_values = ssn.to_numpy()
    pew_unauthorized_rows = np.isin(
        status_values,
        _PEW_UNAUTHORIZED_STATUS_VALUES,
    ) | (
        (status_values == "CUBAN_HAITIAN_ENTRANT")
        & (ssn_values == "NON_CITIZEN_VALID_EAD")
    )
    pew_unauthorized = float(weights[pew_unauthorized_rows].sum())
    details["pew_unauthorized_population"] = pew_unauthorized
    details["pew_unauthorized_statuses"] = list(_PEW_UNAUTHORIZED_STATUS_VALUES)
    details["pew_unauthorized_paired_status"] = {
        "immigration_status_str": "CUBAN_HAITIAN_ENTRANT",
        "ssn_card_type": "NON_CITIZEN_VALID_EAD",
    }
    relative = pew_unauthorized / controls.undocumented.population_anchor
    rel_low, rel_high = _UNDOCUMENTED_ANCHOR_RELATIVE_BAND
    if not (rel_low <= relative <= rel_high):
        failures.append(
            f"emergent Pew-defined unauthorized population "
            f"{pew_unauthorized:,.0f} is "
            f"{relative:.2f}x the published anchor "
            f"{controls.undocumented.population_anchor:,.0f} "
            f"(band [{rel_low}, {rel_high}], "
            f"{controls.undocumented.sources[_ANCHOR_KEY]})."
        )

    hum_low, hum_high = _HUMANITARIAN_TARGET_RELATIVE_BAND
    humanitarian_statuses = set(_HUMANITARIAN_STATUS_BY_CATEGORY.values())
    humanitarian_rows = np.isin(status_values, list(humanitarian_statuses))
    draw_achieved: dict[str, dict[str, object]] = {}
    compatible_humanitarian = np.zeros(len(person), dtype=bool)
    if evidence is not None:
        for draw in controls.humanitarian:
            emitted_mask = _humanitarian_emitted_mask(
                person,
                draw=draw,
                profile=evidence,
            )
            compatible_humanitarian |= emitted_mask
            emitted = float(weights[emitted_mask].sum())
            relative = (emitted / draw.target) if draw.target > 0 else None
            draw_achieved[draw.label] = {
                "category": draw.category,
                "origin": draw.origin,
                "status": draw.status,
                "target": draw.target,
                "population": emitted,
                "relative": relative,
                "source": draw.source,
            }
            if draw.target <= 0:
                if emitted > 0:
                    failures.append(
                        f"{draw.label}: {emitted:,.0f} compatible weighted "
                        "persons emitted against an explicit zero target."
                    )
                continue
            if relative is None or not (hum_low <= relative <= hum_high):
                failures.append(
                    f"{draw.label}: compatible emitted population {emitted:,.0f} "
                    f"is {relative:.2f}x the cited stock target {draw.target:,.0f} "
                    f"(band [{hum_low}, {hum_high}])."
                )
        incompatible = humanitarian_rows & ~compatible_humanitarian
        if incompatible.any():
            incompatible_population = float(weights[incompatible].sum())
            examples = sorted(set(status_values[incompatible].tolist()))
            failures.append(
                f"{int(incompatible.sum())} humanitarian row(s) "
                f"({incompatible_population:,.0f} weighted persons) violate "
                "their source-aware status/origin/arrival/SSN cohort contract; "
                f"statuses={examples}."
            )
    details["humanitarian_draw_achieved"] = draw_achieved

    achieved_by_category: dict[str, dict[str, object]] = {}
    for category in HUMANITARIAN_STATUS_CATEGORIES:
        status_name = _HUMANITARIAN_STATUS_BY_CATEGORY[category]
        target = controls.humanitarian_target(category)
        emitted = float(weights[status_values == status_name].sum())
        achieved_by_category[category] = {
            "status": status_name,
            "target": target,
            "population": emitted,
            "relative": (emitted / target) if target > 0 else None,
        }
        if target <= 0:
            if emitted > 0:
                failures.append(
                    f"{status_name}: {emitted:,.0f} weighted persons emitted "
                    "against an explicit zero target — the manifest documents "
                    "this category as not imputed."
                )
            continue
        category_relative = emitted / target
        if not (hum_low <= category_relative <= hum_high):
            failures.append(
                f"{status_name}: emitted population {emitted:,.0f} is "
                f"{category_relative:.2f}x the cited stock target "
                f"{target:,.0f} (band [{hum_low}, {hum_high}]) — the H.R.1 "
                "eligibility channels through this category are degenerate."
            )
    details["humanitarian_achieved"] = achieved_by_category

    return GateResult(
        name="immigration_composition",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
