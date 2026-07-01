"""SSN card type and immigration status from CPS ASEC citizenship inputs.

Without this stage the published dataset stores no SSN/immigration columns at
all, so the rules engine defaults every person to a citizen with a valid SSN
and every SSN- or immigration-conditioned policy becomes a no-op — the
failure mode of populace issue #225 (OBBBA's CTC SSN tightening scored ≈ $0
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
5. **Status tags carry only statutory tests the data can support.**
   ``DACA`` applies the statutory cohort test (arrived by 2007 before age
   16, aged 15+) to EAD holders; ``CUBAN_HAITIAN_ENTRANT`` applies the
   nationality-plus-arrival class to documented non-citizens; every other
   documented non-citizen stays ``LEGAL_PERMANENT_RESIDENT`` (the modal true
   status). Blanket ``REFUGEE``/``TPS`` labels for recent arrivals or
   leftover EAD holders are deliberately not emitted: they would mislabel
   millions (true stocks are under a million each) and over-grant
   refugee-class exemptions in benefit rules, while LPR treatment gives
   near-identical means-tested eligibility for those populations.

Selection draws are seeded blake2b hashes keyed by the person's stable source
identity (``source_year``/``source_person_id`` when present), so
support-channel clones of one source person always receive the same status
and reruns are bit-reproducible without global RNG state.

Control totals and the gate anchor are data, not code: they live in the
``immigration_status`` stage of ``populace/build/us/source_stages.json`` with
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

from populace.build.gates import GateResult
from populace.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "IMMIGRATION_STATUS_VALUES",
    "SSN_CARD_TYPE_VALUES",
    "US_IMMIGRATION_NONCONSTANT_PERSON_COLUMNS",
    "US_IMMIGRATION_OUTPUT_COLUMNS",
    "US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS",
    "US_IMMIGRATION_STAGE_NAME",
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
#: docstring for why blanket REFUGEE/TPS labels are not fabricated).
IMMIGRATION_STATUS_VALUES: tuple[str, ...] = (
    "CITIZEN",
    "LEGAL_PERMANENT_RESIDENT",
    "CUBAN_HAITIAN_ENTRANT",
    "DACA",
    "UNDOCUMENTED",
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

_DERIVE_IMMIGRATION_STATUS_PARAMETER_KEYS = frozenset(
    {
        *_FORCED_CONTROL_KEYS,
        _ANCHOR_KEY,
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

#: Weighted share of persons whose SSN card type is not ``CITIZEN`` must land
#: in this band (Census counts ~22–27M non-citizens of ~336M residents; a
#: share outside it means the imputation collapsed or exploded).
_NON_CITIZEN_SHARE_BAND = (0.03, 0.12)
#: Emergent weighted ``NONE`` (undocumented) population relative to the cited
#: published anchor. Coarse by design — a release-blocking backstop against
#: collapse or explosion, not a calibration objective; the level belongs to
#: the calibration lane.
_UNDOCUMENTED_ANCHOR_RELATIVE_BAND = (0.5, 1.6)


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


def us_immigration_stage_spec() -> SourceStageSpec:
    """Load the packaged ``immigration_status`` source-stage manifest entry."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
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
    ssn_codes = _assign_ssn_card_codes(
        result,
        weights.to_numpy(dtype=np.float64),
        seed=int(context.config.seed),
        controls=controls,
    )
    status = _derive_immigration_status(
        result,
        ssn_codes,
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


def _controls_from_parameters(params: Mapping[str, object]) -> UndocumentedControls:
    values: dict[str, float] = {}
    sources: dict[str, str] = {}
    for key, value_key in (
        ("undocumented_workers", "target"),
        ("undocumented_students", "target"),
        (_ANCHOR_KEY, "value"),
    ):
        block = params.get(key)
        if not isinstance(block, Mapping):
            raise SourceRuntimeError(
                f"US immigration derivation requires a {key!r} object with "
                f"{value_key!r} and 'source'."
            )
        unexpected = sorted(set(block) - {value_key, "source"})
        if unexpected:
            raise SourceRuntimeError(
                f"US immigration control {key!r} has unsupported key(s): {unexpected}."
            )
        value = block.get(value_key)
        source = block.get("source")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or float(value) <= 0
        ):
            raise SourceRuntimeError(
                f"US immigration control {key!r} requires a positive numeric "
                f"{value_key!r}."
            )
        if not isinstance(source, str) or not source:
            raise SourceRuntimeError(
                f"US immigration control {key!r} requires a source citation."
            )
        values[key] = float(value)
        sources[key] = source
    return UndocumentedControls(
        workers=values["undocumented_workers"],
        students=values["undocumented_students"],
        population_anchor=values[_ANCHOR_KEY],
        sources=sources,
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


def _assign_ssn_card_codes(
    person: pd.DataFrame,
    weights: np.ndarray,
    *,
    seed: int,
    controls: UndocumentedControls,
) -> np.ndarray:
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
    worker_excess = float(weights[worker_candidates].sum()) - controls.workers
    worker_draws = _stable_person_draws(
        person, seed=seed, salt="immigration:ead_workers"
    )
    ssn_codes[
        _select_weight_to_target(
            worker_candidates, weights, worker_draws, worker_excess
        )
    ] = 2

    student_candidates = (ssn_codes == 0) & noncitizens & is_student
    student_excess = float(weights[student_candidates].sum()) - controls.students
    student_draws = _stable_person_draws(
        person, seed=seed, salt="immigration:ead_students"
    )
    ssn_codes[
        _select_weight_to_target(
            student_candidates, weights, student_draws, student_excess
        )
    ] = 2
    return ssn_codes


def _derive_immigration_status(
    person: pd.DataFrame,
    ssn_codes: np.ndarray,
    *,
    time_period: int,
) -> np.ndarray:
    arrival_code = _integer_column(person, "PEINUSYR")
    arrival_year = np.full(len(person), time_period, dtype=np.int64)
    for code, midpoint in _ARRIVAL_YEAR_MIDPOINTS.items():
        arrival_year[arrival_code == code] = midpoint
    years_in_us = time_period - arrival_year
    age = _integer_column(person, "A_AGE")
    age_at_entry = np.maximum(0, age - years_in_us)
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

    daca = (
        (ssn_codes == 2)
        & (arrival_year <= _DACA_LATEST_ARRIVAL_YEAR)
        & (age_at_entry < _DACA_MAX_AGE_AT_ENTRY)
        & (age >= _DACA_MIN_CURRENT_AGE)
    )
    status[daca] = "DACA"
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
    controls: UndocumentedControls | None = None,
) -> GateResult:
    """Release gate: the SSN/immigration surface exists and is plausible.

    Fails when either output column is missing or constant (the #225 failure
    mode: everyone a citizen with a valid SSN), when a value falls outside
    the engine enum domain, when the two columns disagree about citizenship
    or undocumented status, when the weighted non-citizen share leaves its
    plausibility band, or when the emergent undocumented population strays
    outside a coarse band around its cited published anchor.
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
            "undocumented_workers": controls.workers,
            "undocumented_students": controls.students,
            "undocumented_population_anchor": controls.population_anchor,
            "sources": dict(controls.sources),
        },
        "non_citizen_share_band": list(_NON_CITIZEN_SHARE_BAND),
        "undocumented_anchor_relative_band": list(_UNDOCUMENTED_ANCHOR_RELATIVE_BAND),
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
    relative = undocumented / controls.population_anchor
    rel_low, rel_high = _UNDOCUMENTED_ANCHOR_RELATIVE_BAND
    if not (rel_low <= relative <= rel_high):
        failures.append(
            f"emergent undocumented population {undocumented:,.0f} is "
            f"{relative:.2f}x the published anchor "
            f"{controls.population_anchor:,.0f} (band [{rel_low}, {rel_high}], "
            f"{controls.sources[_ANCHOR_KEY]})."
        )

    return GateResult(
        name="immigration_composition",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
