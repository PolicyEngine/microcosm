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
  ``CUBAN_HAITIAN_ENTRANT`` / ``DACA`` / ``UNDOCUMENTED``.

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
   the unauthorized is unidentified from the survey alone. Workers and
   students spill to ``NON_CITIZEN_VALID_EAD`` — in deterministic seeded
   order — until the *remaining* undocumented worker/student counts match
   their published control totals (Pew Research Center; Higher Ed
   Immigration Portal). This is the only forced margin.
4. **The total undocumented population is emergent, not forced.**
   Representation is calibration's job, not the label stage's: the release
   gate checks the emergent total against a cited published anchor with a
   coarse plausibility band, and a follow-up calibration target (Ledger
   facts lane) can reconcile the level in the weights. On the 2024 ASEC the
   emergent total is ≈13M — inside the range of published 2023–24 estimates
   — with 26.6M non-citizens, 18.0M residual after the indicators, and 4.8M
   spilled to EAD.
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
identity (``source_year``/``source_person_id`` when present), so
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
    "us_immigration_composition_gate",
    "us_immigration_composition_summary",
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
    "WSAL_VAL",
    "SEMP_VAL",
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
    29: 2024,
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

#: PEINUSYR codes for 2020+ arrivals (2024 ASEC data dictionary: 27 =
#: 2020-2021, 28 = 2022-2024) — the humanitarian-parole and recent-refugee
#: cohorts. The Census codebook rebins this variable across vintages, so
#: these windows are survey-vintage facts, not statutory dates.
_RECENT_ARRIVAL_CODES = (27, 28)
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
#: Emergent weighted ``NONE`` (undocumented) population relative to the cited
#: published anchor. Coarse by design — a release-blocking backstop against
#: collapse or explosion, not a calibration objective; the level belongs to
#: the calibration lane.
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
        workers: Undocumented-worker control the EAD spill enforces.
        students: Undocumented-student control the EAD spill enforces.
        population_anchor: Published total undocumented population the
            composition gate checks the *emergent* total against (never
            forced in the labels).
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


def _stable_person_draws(person: pd.DataFrame, *, seed: int, salt: str) -> np.ndarray:
    """Deterministic uniform draws keyed by stable person identity.

    Support-channel clones carry their source person's ``source_year`` /
    ``source_person_id``, so keying on those gives every clone of one source
    person the same draw; frames without source ids fall back to
    ``person_id``.
    """

    if {"source_year", "source_person_id"}.issubset(person.columns):
        keys = (
            person["source_year"].astype(str)
            + ":"
            + person["source_person_id"].astype(str)
        )
    else:
        keys = person["person_id"].astype(str)
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


def _humanitarian_draw_candidates(
    draw: HumanitarianDraw,
    *,
    ssn_codes: np.ndarray,
    birth_country: np.ndarray,
    arrival_code: np.ndarray,
) -> np.ndarray:
    """Candidate mask for one draw, before exclusions shared by all draws.

    ``REFUGEE``/``ASYLEE``/``DEPORTATION_WITHHELD`` draw from the
    indicator-documented pool only (code 3: those statuses carry immediate
    federal benefits access, the same signals the ASEC-UA method reads);
    parole and TPS may also draw from the still-unspilled residual pool
    (code 0), whose members then receive EAD-backed SSN cards.
    """

    if draw.category == "paroled_one_year":
        assert draw.origin is not None
        return (
            np.isin(ssn_codes, (0, 3))
            & np.isin(birth_country, _PAROLE_ORIGIN_CODES[draw.origin])
            & np.isin(arrival_code, _RECENT_ARRIVAL_CODES)
        )
    if draw.category == "refugee":
        return (
            (ssn_codes == 3)
            & np.isin(birth_country, _REFUGEE_ORIGIN_CODES)
            & np.isin(arrival_code, _RECENT_ARRIVAL_CODES)
        )
    if draw.category == "asylee":
        return (
            (ssn_codes == 3)
            & np.isin(birth_country, _ASYLEE_ORIGIN_CODES)
            & np.isin(arrival_code, _ASYLEE_ARRIVAL_CODES)
        )
    if draw.category == "deportation_withheld":
        return (
            (ssn_codes == 3)
            & (arrival_code >= 1)
            & (arrival_code <= _WITHHELD_MAX_ARRIVAL_CODE)
        )
    if draw.category == "tps":
        assert draw.origin is not None
        codes, max_arrival_code = _TPS_ORIGIN_CODES[draw.origin]
        return (
            np.isin(ssn_codes, (0, 3))
            & np.isin(birth_country, codes)
            & (arrival_code >= 1)
            & (arrival_code <= max_arrival_code)
        )
    raise SourceRuntimeError(
        f"US immigration derivation has no candidate rule for draw {draw.label!r}."
    )


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
    spill, so the Pew worker/student controls still bind the remaining
    residual pool exactly. Draws are sequential over
    ``controls.humanitarian``; a person marked by an earlier draw is out of
    every later candidate pool. Residual-pool selections (parole/TPS only)
    move to ``NON_CITIZEN_VALID_EAD`` — both programs confer employment
    authorization — keeping the ``NONE`` ⇔ ``UNDOCUMENTED`` invariant.
    """

    marks = np.full(len(person), "", dtype="U24")
    arrival_code, arrival_year, age_at_entry = _arrival_profile(
        person, time_period=time_period
    )
    age = _integer_column(person, "A_AGE")
    birth_country = _integer_column(person, "PENATVTY")
    # Cuba/Haiti-born keep the statutorily-favorable entrant class; the DACA
    # cohort keeps its statutory tag (dual TPS/DACA holders stay DACA).
    excluded = np.isin(birth_country, _CUBAN_HAITIAN_BIRTH_CODES)
    excluded |= _daca_statutory_cohort(arrival_year, age_at_entry, age)
    for draw in controls.humanitarian:
        if draw.target <= 0:
            continue
        candidates = (
            _humanitarian_draw_candidates(
                draw,
                ssn_codes=ssn_codes,
                birth_country=birth_country,
                arrival_code=arrival_code,
            )
            & ~excluded
            & (marks == "")
        )
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

    # Humanitarian draws run between the indicators and the EAD spill: the
    # spill controls then bind the *remaining* residual pool, so the final
    # undocumented worker/student counts still land on the Pew anchors.
    humanitarian_marks = _assign_humanitarian_statuses(
        person,
        ssn_codes,
        weights,
        seed=seed,
        controls=controls,
        time_period=time_period,
    )

    is_worker = (_float_column(person, "WSAL_VAL") > 0) | (
        _float_column(person, "SEMP_VAL") > 0
    )
    is_student = _integer_column(person, "A_HSCOL") == 2

    # The CPS has no work-authorization variable, so the EAD-vs-unauthorized
    # split inside the residual pool is unidentified from the survey. Spill
    # excess undocumented workers/students to EAD so the *remaining* counts
    # match their published controls; counts already at or below a control
    # spill nothing. The total undocumented population is emergent — the
    # composition gate checks it against a cited anchor, and reconciling the
    # level is the calibration lane's job, not this label stage's.
    worker_candidates = (ssn_codes == 0) & noncitizens & is_worker
    worker_excess = (
        float(weights[worker_candidates].sum()) - controls.undocumented.workers
    )
    worker_draws = _stable_person_draws(
        person, seed=seed, salt="immigration:ead_workers"
    )
    ssn_codes[
        _select_weight_to_target(
            worker_candidates, weights, worker_draws, worker_excess
        )
    ] = 2

    student_candidates = (ssn_codes == 0) & noncitizens & is_student
    student_excess = (
        float(weights[student_candidates].sum()) - controls.undocumented.students
    )
    student_draws = _stable_person_draws(
        person, seed=seed, salt="immigration:ead_students"
    )
    ssn_codes[
        _select_weight_to_target(
            student_candidates, weights, student_draws, student_excess
        )
    ] = 2
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

    documented_noncitizen = np.isin(ssn_codes, [2, 3])
    cuban_haitian = (
        documented_noncitizen
        & np.isin(birth_country, _CUBAN_HAITIAN_BIRTH_CODES)
        & (arrival_year >= _CUBAN_HAITIAN_ARRIVAL_CUTOFF)
    )
    status[cuban_haitian] = "CUBAN_HAITIAN_ENTRANT"

    daca = (ssn_codes == 2) & _daca_statutory_cohort(arrival_year, age_at_entry, age)
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
) -> Frame:
    """Run the ``immigration_status`` manifest stage over a US frame.

    Existing output columns are preserved, making the transform idempotent —
    a base H5 built with this stage passes through untouched at release time.

    Args:
        frame: A US-schema frame whose person table still carries the raw
            CPS ASEC source columns.
        seed: Build-wide imputation seed.
        time_period: The dataset's time period (arrival-year arithmetic).

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
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
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
    plausibility band, when the emergent undocumented population strays
    outside a coarse band around its cited published anchor, or when a
    humanitarian category's emitted mass leaves the coarse band around its
    cited stock target (microcosm #767 — the H.R.1 §71109/§71301/§71302 and
    SNAP §10108 eligibility channels all bind through these categories; an
    explicit zero target must emit exactly zero).
    """

    if controls is None:
        stage = us_immigration_stage_spec()
        derive = [
            operation
            for operation in stage.operations
            if operation.kind == "derive_immigration_status"
        ]
        if len(derive) != 1:
            raise ValueError(
                "US immigration stage must declare exactly one "
                "derive_immigration_status operation."
            )
        controls = _controls_from_parameters(derive[0].parameters)

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

    non_citizen_share = (
        float(weights[~ssn_citizen].sum()) / total if total else float("nan")
    )
    low, high = _NON_CITIZEN_SHARE_BAND
    if not np.isfinite(non_citizen_share) or not (low <= non_citizen_share <= high):
        failures.append(
            f"non-citizen weighted share {non_citizen_share:.4f} outside "
            f"[{low}, {high}]."
        )

    undocumented = float(weights[ssn_none].sum())
    relative = undocumented / controls.undocumented.population_anchor
    rel_low, rel_high = _UNDOCUMENTED_ANCHOR_RELATIVE_BAND
    if not (rel_low <= relative <= rel_high):
        failures.append(
            f"emergent undocumented population {undocumented:,.0f} is "
            f"{relative:.2f}x the published anchor "
            f"{controls.undocumented.population_anchor:,.0f} "
            f"(band [{rel_low}, {rel_high}], "
            f"{controls.undocumented.sources[_ANCHOR_KEY]})."
        )

    status_values = status.to_numpy()
    hum_low, hum_high = _HUMANITARIAN_TARGET_RELATIVE_BAND
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
