"""Live source-owned domain projection on the exact raw survey clone prefix.

Rows and receipts are descriptive. Only the retained issued capsule can
requalify them against its actual preparation and populations. No financial
descendant, donor role, matching, amount placement or geography is admitted.
"""

from __future__ import annotations

import hashlib
import json
import sys
import weakref
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import FunctionType, ModuleType

from microcosm.graph import Population

from . import support_provenance as provenance
from . import survey_catalogue_selection as selection
from . import survey_origin_budget as budget
from . import survey_population_domains as domains
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.current-survey-household-domains/1"
# These are the maintained complete-source/transport bounds, not the separate
# tiny invented tail-matching caps. Two ordinary copies follow each original.
MAX_ORIGINS = budget.MAX_GROUPS
MAX_ROSTER_BYTES = source.MAX_ROSTER_BYTES
_ISSUED = {}
_SOURCE_MODULE = source
_BUDGET_MODULE = budget


def _require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_HOUSEHOLD_DOMAINS_" + reason)


@dataclass(frozen=True, slots=True)
class SurveyHouseholdDomainRow:
    """Private source coordinates on clone0/1; not a receiving capability."""

    household_id: int
    source: domains.Source
    source_year: int
    survey_year: int
    raw_native_id: str
    support_source_id: int
    spine_source_id: int
    clone_index: int
    domain: domains.Domain
    statistical_unit: str
    original_design_anchor: Fraction
    occupied_housing_tail_eligible: bool


def _row_value(row):
    _require(type(row) is SurveyHouseholdDomainRow, "ROW_TYPE")
    return (
        row.household_id,
        row.source.value,
        row.source_year,
        row.survey_year,
        row.raw_native_id,
        row.support_source_id,
        row.spine_source_id,
        row.clone_index,
        row.domain.value,
        row.statistical_unit,
        (row.original_design_anchor.numerator, row.original_design_anchor.denominator),
        row.occupied_housing_tail_eligible,
    )


def _roster_digest(rows):
    _require(type(rows) is tuple and 0 < len(rows) <= 2 * MAX_ORIGINS, "ROSTER_BOUND")
    digest, size = hashlib.sha256(), 0
    for row in rows:
        payload = source._encode(_row_value(row))
        size += len(payload) + 8
        _require(size <= MAX_ROSTER_BYTES, "ROSTER_BYTES")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _modules():
    return (sys.modules[__name__], provenance, selection, domains)


def _live():
    _require(
        type(source) is ModuleType
        and source is _SOURCE_MODULE
        and type(budget) is ModuleType
        and budget is _BUDGET_MODULE,
        "IMPLEMENTATION_BINDING_CHANGED",
    )
    result = {}
    for module in _modules():
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[module.__name__, name] = source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[module.__name__, name] = value
                for method, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[module.__name__, name, method] = source._function_seal(
                            function
                        )
    result["contract"] = (
        PROTOCOL,
        MAX_ORIGINS,
        MAX_ROSTER_BYTES,
        # declaration() returns fresh DomainShare objects. Bind their exact
        # primitive content, not the opaque object identities used by the
        # source module's general dependency marker.
        tuple(
            (
                share.domain.value,
                share.statistical_unit,
                (share.acs_share.numerator, share.acs_share.denominator),
                (share.asec_share.numerator, share.asec_share.denominator),
            )
            for share in domains.declaration()
        ),
    )
    return result


def _code_bytes():
    return tuple(
        (
            module.__name__,
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        )
        for module in _modules()
    )


def _producer():
    _require(_live() == _LIVE and _code_bytes() == _BYTES, "IMPLEMENTATION_CHANGED")
    # Bind the actual maintained no-geography reconstruction implementation,
    # not just the local call site's name or a caller's supplied digest.
    budget._producer()
    return _BYTES


def _raw_key(record):
    return (
        record["source"],
        record["source_year"],
        record["survey_year"],
        record["raw_native_id"],
    )


def _project(view, clone_population):
    selected = view.selection_plan.selected
    _require(
        type(selected) is tuple and 0 < len(selected) <= MAX_ORIGINS, "SELECTED_BOUND"
    )
    by_raw = {}
    for row in selected:
        _require(type(row) is selection.SelectedHousehold, "SELECTED_TYPE")
        key = (
            row.key.source.value,
            row.key.source_year,
            row.key.survey_year,
            row.key.native_id,
        )
        _require(key not in by_raw, "DUPLICATE_RAW_IDENTITY")
        by_raw[key] = row
    origins = view.receipt["origins"]["households"]
    _require(type(origins) is list and len(origins) == len(selected), "ORIGIN_COVERAGE")
    prepared = view.frame.table("household").set_index("household_id")
    _require(
        prepared.index.is_unique and len(prepared) == len(selected), "PREPARED_AXIS"
    )
    source_column = provenance.support_source_id_column("household")
    spine_column = provenance.spine_source_id_column("household")
    channel_column = provenance.support_channel_column("household")
    clone_column = provenance.support_clone_index_column("household")
    by_support, used_households = {}, set()
    for origin in origins:
        key = _raw_key(origin)
        chosen = by_raw.pop(key, None)
        _require(chosen is not None, "RAW_IDENTITY_COVERAGE")
        household_id = origin["household_id"]
        _require(
            type(household_id) is int
            and household_id in prepared.index
            and household_id not in used_households,
            "ORIGIN_AXIS",
        )
        used_households.add(household_id)
        row = prepared.loc[household_id]
        _require(
            row[channel_column] == chosen.key.source.value
            and int(row[spine_column]) == origin["selected_receiving_household_id"],
            "PREPARED_ORIGIN",
        )
        anchor = chosen.original_design_weight
        _require(
            type(anchor) is Fraction
            and anchor >= 0
            and origin["original_anchor"] == [anchor.numerator, anchor.denominator],
            "ORIGINAL_ANCHOR",
        )
        coordinate = (
            chosen.key.source.value,
            int(row[source_column]),
            int(row[spine_column]),
        )
        _require(coordinate not in by_support, "DUPLICATE_SUPPORT_ORIGIN")
        by_support[coordinate] = chosen
    _require(not by_raw and used_households == set(prepared.index), "COMPLETE_ORIGINS")
    units = {row.domain: row.statistical_unit for row in domains.declaration()}
    projected, copies = [], set()
    clone = clone_population.frame.table("household")
    _require(len(clone) == 2 * len(selected), "ORDINARY_CLONE_ROSTER")
    for household_id, channel, support_id, spine_id, clone_index in clone.loc[
        :, ["household_id", channel_column, source_column, spine_column, clone_column]
    ].itertuples(index=False, name=None):
        coordinate = (channel, int(support_id), int(spine_id))
        _require(coordinate in by_support and clone_index in (0, 1), "CLONE_ORIGIN")
        copy = (coordinate, int(clone_index))
        _require(copy not in copies, "DUPLICATE_CLONE_ORIGIN")
        copies.add(copy)
        chosen = by_support[coordinate]
        _require(
            type(chosen.domain) is domains.Domain
            and chosen.statistical_unit == units[chosen.domain],
            "DOMAIN_UNIT",
        )
        eligible = chosen.domain in (
            domains.Domain.SHARED_HOUSING,
            domains.Domain.RESIDUAL_HOUSING,
        )
        projected.append(
            SurveyHouseholdDomainRow(
                int(household_id),
                chosen.key.source,
                chosen.key.source_year,
                chosen.key.survey_year,
                chosen.key.native_id,
                int(support_id),
                int(spine_id),
                int(clone_index),
                chosen.domain,
                chosen.statistical_unit,
                chosen.original_design_weight,
                eligible,
            )
        )
    _require(
        copies
        == {
            (coordinate, clone_index)
            for coordinate in by_support
            for clone_index in (0, 1)
        },
        "COMPLETE_CLONE_ORIGINS",
    )
    return tuple(sorted(projected, key=lambda row: row.household_id))


@dataclass(frozen=True)
class _State:
    preparation: object
    preparation_entry: tuple
    preparation_objects: tuple
    preparation_context: bytes
    allocated: Population
    cloned: Population
    population_frames: tuple
    populations: tuple
    rows: tuple[SurveyHouseholdDomainRow, ...]
    rows_sha256: str
    payload: bytes


def _source_entry(preparation):
    _require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    return source.AuthenticatedSurveyPopulationPreparation._checked(preparation)


def _view(entry):
    state = entry[2]
    return source.CheckedSurveyPopulationView(
        entry[1], state.context, state.frame, state.plan, json.loads(entry[1])
    )


def _population_seals(allocated, cloned):
    return tuple(budget._population_identity(p) for p in (allocated, cloned))


def _preparation_objects(entry):
    state = entry[2]
    return (state.frame, state.plan, *state.source_frames)


def _pure_final(state):
    # No I/O follows these fences: closing-source or implementation-read
    # callbacks cannot change the previously captured baseline unnoticed.
    entry = source._ISSUED.get(id(state.preparation))
    _require(
        entry is state.preparation_entry
        and entry[0]() is state.preparation
        and type(state.preparation.payload) is bytes
        and state.preparation.payload == entry[1],
        "FINAL_PREPARATION_IDENTITY",
    )
    _require(_roster_digest(state.rows) == state.rows_sha256, "FINAL_ROSTER_SEAL")
    _require(
        _project(_view(entry), state.cloned) == state.rows, "FINAL_ROSTER_PROJECTION"
    )
    # Decode/projection precede the final source and population fences, so a
    # callback while borrowing a view cannot establish a new object baseline.
    _require(
        all(
            current is retained
            for current, retained in zip(
                _preparation_objects(entry), state.preparation_objects, strict=True
            )
        )
        and entry[2].context == state.preparation_context,
        "FINAL_PREPARATION_OBJECTS",
    )
    _require(
        state.allocated.frame is state.population_frames[0]
        and state.cloned.frame is state.population_frames[1],
        "FINAL_POPULATION_OBJECTS",
    )
    source._pure_final(entry[2])
    _require(
        _population_seals(state.allocated, state.cloned) == state.populations,
        "FINAL_POPULATION_SEAL",
    )
    _require(_live() == _LIVE, "FINAL_IMPLEMENTATION_CHANGED")


def _validate(state):
    with source.verification_epoch():
        _producer()
        _pure_final(state)
        entry = _source_entry(state.preparation)
        _require(entry is state.preparation_entry, "PREPARATION_IDENTITY")
        budget._initial(
            _view(entry),
            state.allocated,
            state.cloned,
            preparation=state.preparation,
            geography_config=None,
        )
        _require(_source_entry(state.preparation) is entry, "FINAL_SOURCE_IDENTITY")
        _pure_final(state)
    # Close performs the actual unconditional source revalidation. Only then
    # do the final code-read and pure-state fences permit a value to escape.
    _producer()
    _pure_final(state)


def _entry(value):
    entry = _ISSUED.get(id(value))
    _require(
        type(value) is QualifiedSurveyHouseholdDomains
        and entry is not None
        and entry[0]() is value
        and type(value.payload) is bytes
        and value.payload == entry[1] == entry[2].payload,
        "UNISSUED_OR_CHANGED",
    )
    return entry


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class QualifiedSurveyHouseholdDomains:
    """Retained live raw-prefix projection owner; no public constructor."""

    payload: bytes

    def __post_init__(self):
        raise ValueError("CURRENT_SURVEY_HOUSEHOLD_DOMAINS_NO_PUBLIC_CONSTRUCTOR")

    def checked_rows(self):
        try:
            entry = _entry(self)
            _validate(entry[2])
            _require(_entry(self) is entry, "FINAL_ISSUANCE")
            return entry[2].rows
        except BaseException:
            _ISSUED.pop(id(self), None)
            raise

    @property
    def release_eligible(self):
        return False


def qualify_current_survey_household_domains(
    preparation, allocated_population, clone_population
):
    """Qualify the exact source allocation/ordinary clone prefix, without a fit.

    Actual source files are accepted only through the existing preparation
    issuer. Raw source keys stay in the private checked rows; the small receipt
    binds their digest. No matching or downstream receiving authority follows.
    """
    with source.verification_epoch():
        _producer()
        entry = _source_entry(preparation)
        preparation_objects = _preparation_objects(entry)
        preparation_context = entry[2].context
        before = _population_seals(allocated_population, clone_population)
        population_frames = (allocated_population.frame, clone_population.frame)
        view = _view(entry)
        budget._initial(
            view,
            allocated_population,
            clone_population,
            preparation=preparation,
            geography_config=None,
        )
        rows = _project(view, clone_population)
        rows_sha256 = _roster_digest(rows)
        payload = source._encode(
            dict(
                protocol=PROTOCOL,
                preparation_sha256=hashlib.sha256(entry[1]).hexdigest(),
                roster_sha256=rows_sha256,
                roster_rows=len(rows),
                occupied_housing_rows=sum(
                    row.occupied_housing_tail_eligible for row in rows
                ),
                original_households=len(rows) // 2,
                raw_prefix_only=True,
                source_window_equivalence_claim=False,
                matching_qualified=False,
                donor_role_qualified=False,
                geography_assigned=False,
                release_eligible=False,
                implementation=dict(_BYTES),
            )
        )
        state = _State(
            preparation,
            entry,
            preparation_objects,
            preparation_context,
            allocated_population,
            clone_population,
            population_frames,
            before,
            rows,
            rows_sha256,
            payload,
        )
        _require(_source_entry(preparation) is entry, "FINAL_SOURCE_IDENTITY")
        _pure_final(state)
    _producer()
    _pure_final(state)
    value = object.__new__(QualifiedSurveyHouseholdDomains)
    object.__setattr__(value, "payload", payload)
    identifier = id(value)

    def forget(reference):
        current = _ISSUED.get(identifier)
        if current is not None and current[0] is reference:
            _ISSUED.pop(identifier, None)

    reference = weakref.ref(value, forget)
    _ISSUED[identifier] = (reference, payload, state)
    return value


_LIVE = _live()
_BYTES = _code_bytes()
