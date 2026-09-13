"""Canonical household reference-person role, qualified from retained owners.

Scope fence. This module is descriptive. It issues no source, no population and
no release authority, and it is deliberately not wired into the accepted country
enrichment host. It borrows maintained owners and re-states none of their
contracts:

* The canonical requirement is the one the composed binding already records.
  :mod:`.graph_composed_asec_binding` carries ``is_household_head`` in
  ``UNBOUND_DEMOGRAPHIC_COLUMNS`` as an open column, with the ASEC requirement
  ``A_EXPRRP`` printed codes 1/2 and exactly one such person per household, the
  ACS requirement ``RELSHIPP == 20`` on housing units only, and three signals
  recorded as *not adopted*. That entry is pinned here and every declaration in
  this module is checked against it rather than restated.
* ASEC roles come from :mod:`.asec_demographic_source`, which reconstructs the
  printed ``A_EXPRRP`` code from the pinned CSV members and publishes
  ``asec_household_reference_state`` and its unbound reason after proving the
  grouping column is a bijection with the published household key.
* ACS roles come from the pinned ACS person member's own ``RELSHIPP`` token,
  classified by that same owner's :func:`acs_household_reference_states`.

What this module refuses to assume, stated once:

* ``P_SEQ == 1`` is **not** headship. It is a within-household sequence number
  whose dictionary entry labels no code as reference person, and the maintained
  owner already records it as ``diagnostic_crosscheck``. The incumbent ASEC
  ``is_household_head`` leaf produced by :mod:`.relationship_inputs` is that
  signal; this module compares values with it and never adopts its rule.
* A prepared arm's ``A_EXPRRP`` column is **not** a source. ``asec_pool``
  derives it from ``A_LINENO == 1`` when a locked input omits it, and
  ``acs_pums`` derives it from ``RELSHIPP`` — writing ``14`` for every member of
  a group-quarters-only household. Both arms are therefore read from their own
  original code, never from the harmonized column.
* Relationship-code allocation provenance is **unresolved on both arms**. The
  ASEC readset carries no allocation flag for ``A_EXPRRP`` and the ACS readset
  carries none for ``RELSHIPP``, so no receipt written here may call a value
  respondent-reported or claim the two arms carry equivalent evidence.
* A group-quarters ACS household has no observed housing-unit reference person.
  The maintained owner reports that as its own state rather than as "not the
  reference person", and this module keeps it unbound. An absent reference
  person is not a false one.
* Age, family relationship, tax-unit role, line number and row order fill
  nothing. Unresolved stays unresolved.

Returned values are detached. A graph host must requalify the live owners and
compare materialized descendants, including after replay.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, MappingProxyType

import numpy as np
import pandas as pd

from microcosm.graph.population import PopulationError, dtype_for_token, token_for_dtype

from . import acs_housing_universe_source as housing
from . import acs_person_coverage_authentication as records
from . import asec_demographic_source as demographic
from . import graph_composed_asec_binding as composed
from . import support_provenance as provenance
from . import survey_population_preparation as source
from .support_provenance import (
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)

PROTOCOL = "microcosm.us.current-survey-household-roles.v1"
#: The one canonical leaf this seam may write on a receiving population.
CANONICAL_COLUMN = "is_household_head"
#: Every other observation is namespaced so it cannot impersonate that leaf.
SOURCE_PREFIX = "household_role_source_"
SURVEYS = ("acs", "asec")
MAX_PERSONS = 64 * 1024**2 // 32
#: At most one ACS household roster of the printed line-number domain.
MAX_HOUSEHOLD_MEMBERS = 20
MAX_ROSTER_ROWS = MAX_PERSONS
MAX_RECEIPT_BYTES = 64 * 1024

SURVEY_COLUMN = SOURCE_PREFIX + "survey"
NATIVE_ID_COLUMN = SOURCE_PREFIX + "native_person_id"
CODE_COLUMN = SOURCE_PREFIX + "relationship_code"
ROLE_STATE_COLUMN = SOURCE_PREFIX + "role_state"
UNBOUND_REASON_COLUMN = SOURCE_PREFIX + "unbound_reason"
HOUSEHOLD_VERDICT_COLUMN = SOURCE_PREFIX + "household_verdict"
CODE_KNOWN_COLUMN = SOURCE_PREFIX + "code_known"
VALUE_KNOWN_COLUMN = SOURCE_PREFIX + "value_known"
ALLOCATION_KNOWN_COLUMN = SOURCE_PREFIX + "allocation_known"
ORIGIN_COLUMN = SOURCE_PREFIX + "origin"
UNIVERSE_COLUMN = SOURCE_PREFIX + "universe"
OBSERVATION_YEAR_COLUMN = SOURCE_PREFIX + "observation_year"
VALUE_COLUMN = SOURCE_PREFIX + CANONICAL_COLUMN

#: The private rowwise projection, in body order, with its declared tokens.
COLUMN_TOKENS: tuple[tuple[str, str], ...] = (
    (SURVEY_COLUMN, "string"),
    (NATIVE_ID_COLUMN, "int64"),
    (CODE_COLUMN, "Int64"),
    (ROLE_STATE_COLUMN, "int64"),
    (UNBOUND_REASON_COLUMN, "int64"),
    (HOUSEHOLD_VERDICT_COLUMN, "int64"),
    (CODE_KNOWN_COLUMN, "bool"),
    (VALUE_KNOWN_COLUMN, "bool"),
    (ALLOCATION_KNOWN_COLUMN, "bool"),
    (ORIGIN_COLUMN, "string"),
    (UNIVERSE_COLUMN, "string"),
    (OBSERVATION_YEAR_COLUMN, "int64"),
    (VALUE_COLUMN, "boolean"),
)
COLUMNS: tuple[str, ...] = tuple(name for name, _token in COLUMN_TOKENS)

#: The harmonized person state. Both arms' owners publish their own state
#: codes; these are the shared meanings, mapped from theirs and never replacing
#: them. Code 0 is the only state that leaves the canonical leaf unwritten.
ROLE_STATES: dict[int, str] = MappingProxyType(
    {
        0: "unbound",
        1: "bound_household_reference_person",
        2: "bound_not_household_reference_person",
    }
)
#: Why a person is unbound. Codes 1-4 are the ASEC owner's own per-row reasons,
#: carried through unchanged. Codes 5-6 are the ACS arm, whose owner publishes a
#: per-row state and aggregate diagnostics rather than a per-row reason, so only
#: the distinction its states actually carry is claimed here.
ROLE_UNBOUND_REASONS: dict[int, str] = MappingProxyType(
    {
        0: "bound",
        1: "relationship_code_outside_printed_codes",
        2: "household_reference_absent",
        3: "household_reference_duplicated",
        4: "household_relationship_code_indeterminate",
        5: "group_quarters_no_observed_reference_person",
        6: "acs_housing_unit_unbound_reason_not_separately_observed",
    }
)
#: The household-level verdict behind each person's state.
HOUSEHOLD_VERDICTS: dict[int, str] = MappingProxyType(
    {
        1: "exactly_one_reference",
        2: "no_reference",
        3: "multiple_references",
        4: "indeterminate_relationship_code",
        5: "group_quarters_no_observed_reference_person",
        6: "acs_housing_unit_unbound_reason_not_separately_observed",
    }
)
#: ASEC per-row unbound reason -> the household verdict it implies. Taken from
#: the owner's classifier, which sets reason 1 on the offending rows and 4 on
#: the rest of an indeterminate household.
_ASEC_VERDICT_BY_REASON: dict[int, int] = MappingProxyType(
    {0: 1, 1: 4, 2: 2, 3: 3, 4: 4}
)
#: ACS per-row state -> (harmonized state, unbound reason, household verdict).
_ACS_STATE_MAP: dict[int, tuple[int, int, int]] = MappingProxyType(
    {0: (0, 6, 6), 1: (1, 0, 1), 2: (2, 0, 1), 3: (0, 5, 5)}
)

#: The ACS relationship domain the maintained owner admits. A token outside it
#: unbinds its whole housing-unit household there; the sentinel below stands for
#: a token this module could not read as an integer at all, and is deliberately
#: outside the domain and outside the group-quarters codes.
ACS_PRINTED_RELATIONSHIP_DOMAIN: tuple[int, ...] = tuple(range(20, 39))
ACS_UNREADABLE_RELATIONSHIP_SENTINEL = -1
#: Neither arm reaches an allocation flag for its relationship code here, so
#: allocation knownness is false on both. The two reasons are different and are
#: recorded separately rather than collapsed into one claim: the ASEC readset
#: (:data:`asec_demographic_source.ASEC_DEMOGRAPHIC_SOURCE_COLUMNS`) publishes
#: no allocation flag for ``A_EXPRRP`` at all, while the ACS PUMS dictionary
#: does print ``FRELSHIPP`` and this seam simply does not read it. Supplying
#: real ACS relationship-allocation evidence is a separate, separately reviewed
#: source projection; until then neither arm may be called respondent-reported.
RELATIONSHIP_ALLOCATION_PROVENANCE = "unresolved"
ASEC_ALLOCATION_EVIDENCE = "asec_readset_publishes_no_allocation_flag_for_a_exprrp"
ACS_ALLOCATION_EVIDENCE = "acs_frelshipp_allocation_flag_outside_this_seam_readset"
ASEC_ORIGIN = "census_published_asec_relationship_allocation_unresolved"
ACS_ORIGIN = "census_published_acs_relationship_allocation_unresolved"
#: The ASEC universe is the printed one the maintained field carries. The ACS
#: universe is this seam's own required person readset, not a printed quote.
ASEC_UNIVERSE = "asec_a_exprrp_printed_universe"
ACS_UNIVERSE = "acs_required_person_readset"
ASEC_OBSERVATION_YEAR = 2025
ACS_OBSERVATION_YEAR = 2024
INCOME_YEAR = 2024

ASEC_KEYS = ("PERIDNUM", "PH_SEQ", "A_LINENO")
ACS_KEYS = ("SERIALNO", "SPORDER")
ACS_ROSTER_COLUMNS = (*ACS_KEYS, "RELSHIPP")

#: The only keys a public graph receipt may carry. Every value is an aggregate
#: count, a digest, a contract reference or a closed flag; no row, identity or
#: source code belongs here.
PUBLIC_RECEIPT_KEYS = frozenset(
    {
        "protocol",
        "canonical_column",
        "canonical_dtype",
        "declared_rewrite",
        "contract_sha256",
        "projection_sha256",
        "receiving_version",
        "receiving_rows",
        "original_persons",
        "clone_indices",
        "preserved_cells",
        "preserved_unbound_cells",
        "filled_cells",
        "unresolved_cells",
        "agreeing_cells",
        "acs_originals",
        "asec_originals",
        "code_known_originals",
        "value_known_originals",
        "reference_person_originals",
        "acs_group_quarters_originals",
        "relationship_allocation_provenance",
        "equivalent_allocation_evidence_claim",
        "unallocated_observation_claim",
        "positional_sequence_headship_claim",
        "source_admission_issued",
        "population_admission_issued",
        "release_eligible",
    }
)


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_HOUSEHOLD_ROLES_" + reason)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _implementation_modules():
    return (sys.modules[__name__], demographic, composed, provenance)


def _source_bytes():
    """Source identity is checked before the final pure owner/output seals."""
    return tuple(
        (module.__name__, _sha(Path(module.__file__).read_bytes()))
        for module in _implementation_modules()
    )


def _live():
    """Extend the maintained preparation fence with this qualifier's closure."""
    result = dict(source._live())
    for module in _implementation_modules():
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
    result["household_role_contract"] = source._runtime_marker(
        (
            PROTOCOL,
            CANONICAL_COLUMN,
            SOURCE_PREFIX,
            SURVEYS,
            MAX_PERSONS,
            MAX_HOUSEHOLD_MEMBERS,
            MAX_ROSTER_ROWS,
            MAX_RECEIPT_BYTES,
            COLUMN_TOKENS,
            COLUMNS,
            tuple(sorted(ROLE_STATES.items())),
            tuple(sorted(ROLE_UNBOUND_REASONS.items())),
            tuple(sorted(HOUSEHOLD_VERDICTS.items())),
            tuple(sorted(_ASEC_VERDICT_BY_REASON.items())),
            tuple(sorted(_ACS_STATE_MAP.items())),
            ACS_PRINTED_RELATIONSHIP_DOMAIN,
            ACS_UNREADABLE_RELATIONSHIP_SENTINEL,
            RELATIONSHIP_ALLOCATION_PROVENANCE,
            ASEC_ALLOCATION_EVIDENCE,
            ACS_ALLOCATION_EVIDENCE,
            ASEC_ORIGIN,
            ACS_ORIGIN,
            ASEC_UNIVERSE,
            ACS_UNIVERSE,
            ASEC_OBSERVATION_YEAR,
            ACS_OBSERVATION_YEAR,
            INCOME_YEAR,
            ASEC_KEYS,
            ACS_KEYS,
            ACS_ROSTER_COLUMNS,
            tuple(sorted(PUBLIC_RECEIPT_KEYS)),
        )
    )
    return result


def _canonical_requirement():
    """Bind the composed binding's own recorded requirement for this leaf.

    The requirement is not restated here. It is read from the module that
    records ``is_household_head`` as an open column and refuses any node of that
    stage from owning it, so a change to the recorded contract refuses this seam
    instead of letting it drift into an unreviewed second definition.
    """
    entries = [
        entry
        for entry in composed._UNBOUND_DEMOGRAPHICS
        if entry[0] == CANONICAL_COLUMN
    ]
    require(len(entries) == 1, "CANONICAL_REQUIREMENT_ROSTER")
    column, required, diagnostic, asec_rule, not_adopted, caveats, acs_rule = entries[0]
    require(
        CANONICAL_COLUMN in composed.UNBOUND_DEMOGRAPHIC_COLUMNS
        and tuple(required) == ("A_EXPRRP",)
        and tuple(diagnostic) == ("P_SEQ",)
        and "printed codes 1" in asec_rule
        and "exactly one such person per household" in asec_rule
        and "RELSHIPP == 20" in acs_rule
        and "housing units only" in acs_rule
        and "37 and 38" in acs_rule
        and "no observed reference person" in acs_rule
        and any("P_SEQ == 1" in signal for signal, _reason in not_adopted)
        and any("A_FAMREL == 1" in signal for signal, _reason in not_adopted)
        and any("A_LINENO == 1" in signal for signal, _reason in not_adopted)
        and len(caveats) == 2,
        "CANONICAL_REQUIREMENT_CHANGED",
    )
    return {
        "column": column,
        "required_source_columns": list(required),
        "diagnostic_only_source_columns": list(diagnostic),
        "asec_rule": asec_rule,
        "acs_rule": acs_rule,
        "not_adopted": [
            {"signal": signal, "refusal": reason} for signal, reason in not_adopted
        ],
        "prepared_column_caveats": list(caveats),
    }


def _shared_code_tables():
    """Bind the maintained owners' code meanings instead of restating them.

    A change to any of these tables must be reviewed against this seam rather
    than silently reinterpreted here, so the pin refuses instead of drifting.
    """
    states = dict(demographic.HOUSEHOLD_REFERENCE_STATES)
    reasons = dict(demographic.HOUSEHOLD_REFERENCE_UNBOUND_REASONS)
    acs_states = dict(demographic.ACS_REFERENCE_STATES)
    require(
        states
        == {
            0: "unbound",
            1: "bound_household_reference_person",
            2: "bound_not_household_reference_person",
        }
        and reasons
        == {
            0: "bound",
            1: "relationship_code_outside_printed_codes",
            2: "household_reference_absent",
            3: "household_reference_duplicated",
            4: "household_relationship_code_indeterminate",
        }
        and acs_states
        == {
            0: "unbound",
            1: "observed_housing_unit_reference_person",
            2: "observed_not_reference_person",
            3: "group_quarters_no_observed_reference_person",
        }
        and tuple(demographic.HOUSEHOLD_REFERENCE_CODES) == (1, 2)
        and demographic.ACS_OBSERVED_REFERENCE_CODE == 20
        and tuple(demographic.ACS_GROUP_QUARTERS_CODES) == (37, 38)
        and demographic.ACS_SYNTHETIC_GROUP_QUARTERS_RELATIONSHIP_CODE == 14
        and demographic.A_EXPRRP.printed_range == (1, 14)
        and dict(demographic.A_EXPRRP.named_codes)[1]
        == "Reference person with relatives"
        and dict(demographic.A_EXPRRP.named_codes)[2]
        == "Reference person without relatives"
        and 6 not in dict(demographic.A_EXPRRP.named_codes)
        and demographic.A_EXPRRP.universe_as_printed == "All Persons"
        and demographic.A_EXPRRP.role == "restored_observation"
        and demographic.P_SEQ.role == "diagnostic_crosscheck"
        and dict(demographic.P_SEQ.named_codes) == {}
        and demographic.HOUSEHOLD_GROUPING_COLUMN == "person_household_id"
        and any(
            signal["signal"] == "P_SEQ == 1"
            and signal["carried_as"] == "diagnostic_crosscheck"
            for signal in demographic.NON_ADOPTED_HEAD_SIGNALS
        )
        and any(
            signal["signal"] == "A_FAMREL == 1" and signal["carried_as"] == "not_read"
            for signal in demographic.NON_ADOPTED_HEAD_SIGNALS
        ),
        "SHARED_CODE_TABLES_CHANGED",
    )
    # The harmonized states must stay a faithful renaming of the ASEC owner's.
    require(
        {code: ROLE_STATES[code] for code in states} == states
        and {code: ROLE_UNBOUND_REASONS[code] for code in reasons} == reasons
        and set(_ASEC_VERDICT_BY_REASON) == set(reasons)
        and set(_ACS_STATE_MAP) == set(acs_states),
        "HARMONIZED_STATES_DIVERGED",
    )
    return states, reasons, acs_states


def household_role_contract_document():
    """The full declared contract, as it is recorded in every receipt."""
    _shared_code_tables()
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "canonical_column": CANONICAL_COLUMN,
        "concept": "household reference person of the surveyed household",
        "canonical_requirement": _canonical_requirement(),
        "income_year": INCOME_YEAR,
        "surveys": {
            "asec": {
                "source_column": "A_EXPRRP",
                "observation_year": ASEC_OBSERVATION_YEAR,
                "reference_codes": list(demographic.HOUSEHOLD_REFERENCE_CODES),
                "named_codes": {
                    str(code): demographic.A_EXPRRP.named_codes[code]
                    for code in sorted(demographic.A_EXPRRP.named_codes)
                },
                "owner": demographic.__name__,
                "owner_states": "asec_household_reference_state",
                "owner_reasons": "asec_household_reference_unbound_reason",
                "household_grouping_column": demographic.HOUSEHOLD_GROUPING_COLUMN,
                "household_membership_proof": "verify_household_membership",
                "universe": ASEC_UNIVERSE,
                "universe_as_printed": demographic.A_EXPRRP.universe_as_printed,
                "origin": ASEC_ORIGIN,
                "allocation_evidence": ASEC_ALLOCATION_EVIDENCE,
                "quoted_locator": demographic.A_EXPRRP.quoted_locator,
            },
            "acs": {
                "source_column": "RELSHIPP",
                "observation_year": ACS_OBSERVATION_YEAR,
                "reference_code": demographic.ACS_OBSERVED_REFERENCE_CODE,
                "group_quarters_codes": list(demographic.ACS_GROUP_QUARTERS_CODES),
                "admitted_domain": list(ACS_PRINTED_RELATIONSHIP_DOMAIN),
                "owner": demographic.__name__,
                "owner_states": "acs_household_reference_states",
                "roster_scope": "published_household_roster_read_from_pinned_member",
                "universe": ACS_UNIVERSE,
                "origin": ACS_ORIGIN,
                "allocation_evidence": ACS_ALLOCATION_EVIDENCE,
                "synthetic_group_quarters_relationship_code": (
                    demographic.ACS_SYNTHETIC_GROUP_QUARTERS_RELATIONSHIP_CODE
                ),
            },
        },
        "role_states": {str(k): v for k, v in sorted(ROLE_STATES.items())},
        "unbound_reasons": {str(k): v for k, v in sorted(ROLE_UNBOUND_REASONS.items())},
        "household_verdicts": {
            str(k): v for k, v in sorted(HOUSEHOLD_VERDICTS.items())
        },
        "relationship_allocation_provenance": RELATIONSHIP_ALLOCATION_PROVENANCE,
        "equivalent_allocation_evidence_claim": False,
        "positional_sequence_headship_claim": False,
        "harmonized_relationship_column_read": False,
        "age_or_family_role_fallback": False,
        "group_quarters_reference_person_manufactured": False,
        "columns": list(COLUMNS),
        "column_tokens": {name: token for name, token in COLUMN_TOKENS},
        "source_admission_issued": False,
        "release_eligible": False,
    }


def contract_sha256():
    return _sha(
        json.dumps(
            household_role_contract_document(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )


def _code(value, reason):
    """Accept a plain integer code; a bool or float is not a source code."""
    require(type(value) is int or type(value) is np.int64, reason)
    return int(value)


def asec_household_role_observation(state, reason):
    """Classify one ASEC person from the maintained owner's two published codes.

    The owner has already read the printed ``A_EXPRRP`` token, required every
    relationship code in the published household to be printed, and required
    exactly one reference-person code there. This restates none of that: it
    renames the owner's state and carries its reason through unchanged.
    """
    states, reasons, _acs = _shared_code_tables()
    state = _code(state, "ASEC_STATE_TYPE")
    reason = _code(reason, "ASEC_REASON_TYPE")
    require(state in states and reason in reasons, "ASEC_STATE_DOMAIN")
    require((state == 0) == (reason != 0), "ASEC_STATE_REASON_DISAGREE")
    verdict = _ASEC_VERDICT_BY_REASON[reason]
    value = None if state == 0 else state == 1
    return {
        ROLE_STATE_COLUMN: state,
        UNBOUND_REASON_COLUMN: reason,
        HOUSEHOLD_VERDICT_COLUMN: verdict,
        VALUE_COLUMN: value,
        VALUE_KNOWN_COLUMN: state != 0,
        ORIGIN_COLUMN: ASEC_ORIGIN,
        UNIVERSE_COLUMN: ASEC_UNIVERSE,
        OBSERVATION_YEAR_COLUMN: ASEC_OBSERVATION_YEAR,
    }


def acs_household_role_observation(state, relshipp):
    """Classify one ACS person from the maintained owner's state and its code.

    A group-quarters household reports no observed housing-unit reference
    person. That is the owner's own state, and it stays unbound here: an absent
    reference person is not a false one, and the synthetic nonrelative code the
    unit builder writes for such a household is never read as evidence.
    """
    _states, _reasons, acs_states = _shared_code_tables()
    state = _code(state, "ACS_STATE_TYPE")
    relshipp = _code(relshipp, "ACS_CODE_TYPE")
    require(state in acs_states, "ACS_STATE_DOMAIN")
    role, reason, verdict = _ACS_STATE_MAP[state]
    require(
        state != 1 or relshipp == demographic.ACS_OBSERVED_REFERENCE_CODE,
        "ACS_REFERENCE_CODE_DISAGREE",
    )
    require(
        state != 3 or relshipp in demographic.ACS_GROUP_QUARTERS_CODES,
        "ACS_GROUP_QUARTERS_CODE_DISAGREE",
    )
    require(
        relshipp in ACS_PRINTED_RELATIONSHIP_DOMAIN or role == 0,
        "ACS_UNPRINTED_CODE_BOUND",
    )
    value = None if role == 0 else role == 1
    return {
        ROLE_STATE_COLUMN: role,
        UNBOUND_REASON_COLUMN: reason,
        HOUSEHOLD_VERDICT_COLUMN: verdict,
        VALUE_COLUMN: value,
        VALUE_KNOWN_COLUMN: role != 0,
        ORIGIN_COLUMN: ACS_ORIGIN,
        UNIVERSE_COLUMN: ACS_UNIVERSE,
        OBSERVATION_YEAR_COLUMN: ACS_OBSERVATION_YEAR,
    }


def acs_relationship_token(token):
    """Preserve an unreadable literal rather than interpreting it as a code.

    An empty or non-integer token is not a relationship. It becomes a sentinel
    outside both the admitted domain and the group-quarters codes, so the
    maintained classifier unbinds its household instead of admitting a guess.
    """
    require(type(token) is str and len(token) <= 64, "ACS_TOKEN_BOUND")
    if re.fullmatch(r"[0-9]{1,2}", token, re.ASCII) is None:
        return ACS_UNREADABLE_RELATIONSHIP_SENTINEL, False
    code = int(token)
    return code, code in ACS_PRINTED_RELATIONSHIP_DOMAIN


def _observation_table(rows, index):
    """Assemble the rowwise projection with exactly its declared storage."""
    require(len(rows) == len(index), "OBSERVATION_AXIS")
    table = pd.DataFrame(index=pd.Index(index, name="person_id", dtype="int64"))
    for name, token in COLUMN_TOKENS:
        values = [row[name] for row in rows]
        if token == "string":
            table[name] = pd.array(values, dtype=pd.StringDtype(storage="python"))
        elif token in ("boolean", "Int64"):
            table[name] = pd.array(values, dtype=dtype_for_token(token))
        elif token == "bool":
            table[name] = np.asarray(values, dtype=bool)
        else:
            table[name] = np.asarray(values, dtype=np.int64)
    require(tuple(table.columns) == COLUMNS, "OBSERVATION_ROSTER")
    return table


def _projection_bytes(table):
    return table.reset_index().to_json(orient="table", index=False).encode()


def _table_stamp(table):
    return tuple(
        (name, str(table[name].dtype), _sha(table[name].to_json().encode()))
        for name in table
    )


def _origins(preparation_frame, document):
    """Rebuild the original-person roster the preparation itself issued.

    The roster is not trusted as a document: every coordinate is compared with
    the retained Frame's own provenance columns before a key is formed.
    """
    payload = document["origins"]["persons"]
    original = pd.DataFrame(payload["rows"], columns=payload["columns"])
    required = {
        "person_id",
        "source",
        "source_year",
        "survey_year",
        "raw_native_household_id",
        "raw_native_person_id",
        "native_line_numeric_original",
        "selected_receiving_person_id",
    }
    require(
        required <= set(original)
        and original.person_id.dtype == np.dtype("int64")
        and original.person_id.is_unique
        and 0 < len(original) <= MAX_PERSONS,
        "ORIGIN_ROSTER",
    )
    original = original.set_index("person_id", drop=True)
    people = preparation_frame.person
    require(
        np.array_equal(original.index.to_numpy(), people.person_id.to_numpy())
        and np.array_equal(
            original.source.to_numpy(),
            people[support_channel_column("person")].to_numpy(),
        )
        and np.array_equal(
            original.selected_receiving_person_id.to_numpy(),
            people[spine_source_id_column("person")].to_numpy(),
        ),
        "ORIGIN_FRAME_IDENTITY",
    )
    require(
        original.source.isin(SURVEYS).all()
        and original.source_year.eq(INCOME_YEAR).all()
        and original.survey_year.eq(
            original.source.map(
                {"acs": ACS_OBSERVATION_YEAR, "asec": ASEC_OBSERVATION_YEAR}
            )
        ).all(),
        "ORIGIN_PERIOD",
    )
    original["native_person_id"] = original.selected_receiving_person_id
    return original


def _acs_keys(origins):
    """Form the exact ACS person coordinates, and the household roster scope."""
    keys, serials = {}, {}
    arm = origins.loc[origins.source.eq("acs")]
    for person_id, row in arm.iterrows():
        serial = row.raw_native_household_id
        line = row.native_line_numeric_original
        require(
            type(serial) is str
            and type(row.raw_native_person_id) is str
            and type(line) is str,
            "ACS_COORDINATE_TYPE",
        )
        require(
            re.fullmatch(r"2024(?:HU|GQ)[0-9]{7}", serial, re.ASCII) is not None,
            "ACS_HOUSEHOLD_KEY",
        )
        require(
            re.fullmatch(r"[0-9]{1,2}", line, re.ASCII) is not None
            and 1 <= int(line) <= MAX_HOUSEHOLD_MEMBERS
            and row.raw_native_person_id == str(int(line)),
            "ACS_LINE_KEY",
        )
        key = (serial, int(line))
        require(key not in keys, "DUPLICATE_ACS_ORIGIN")
        keys[key] = int(person_id)
        serials.setdefault(serial, set()).add(int(line))
    return keys, serials


def _asec_keys(origins):
    """Form the exact ASEC person coordinates used for the owner-artifact join."""
    keys = {}
    arm = origins.loc[origins.source.eq("asec")]
    for person_id, row in arm.iterrows():
        household = row.raw_native_household_id
        person_key = row.raw_native_person_id
        line = row.native_line_numeric_original
        require(
            type(household) is str and type(person_key) is str and type(line) is str,
            "ASEC_COORDINATE_TYPE",
        )
        require(
            re.fullmatch(r"[0-9]{22}", person_key, re.ASCII) is not None,
            "ASEC_PERSON_KEY",
        )
        require(
            re.fullmatch(r"[0-9]{1,5}", household, re.ASCII) is not None
            and int(household) > 0,
            "ASEC_HOUSEHOLD_KEY",
        )
        require(
            re.fullmatch(r"[0-9]{1,2}", line, re.ASCII) is not None
            and 1 <= int(line) <= MAX_HOUSEHOLD_MEMBERS,
            "ASEC_LINE_KEY",
        )
        key = (int(household), person_key, int(line))
        require(key not in keys, "DUPLICATE_ASEC_ORIGIN")
        keys[key] = int(person_id)
    return keys


def _scan_acs_relationships(stream, *, serials, selected, maximum):
    """Exhaust a byte-bounded literal member, retaining whole household rosters.

    Selection is by household, not by retained person: the reference-person
    cardinality of a published household is only evidence when every one of its
    members is present, so a retained subset of a household may never decide it.
    This helper grants no authority to a file, row count or caller key list.
    """
    header, count = None, 0
    for raw in records._records(stream):
        values = records._decode_record(raw, first=header is None)
        if header is None:
            header = values
            require(
                bool(header)
                and all(header)
                and len(set(header)) == len(header)
                and set(ACS_ROSTER_COLUMNS) <= set(header),
                "ACS_SOURCE_HEADER",
            )
            positions = [header.index(column) for column in ACS_ROSTER_COLUMNS]
            continue
        count += 1
        require(count <= maximum and len(values) == len(header), "ACS_SOURCE_ROW_SHAPE")
        row = dict(zip(ACS_ROSTER_COLUMNS, (values[p] for p in positions), strict=True))
        require(
            all(len(row[column]) <= 64 for column in ACS_ROSTER_COLUMNS),
            "ACS_SOURCE_TOKEN_BOUND",
        )
        serial = row["SERIALNO"]
        if serial not in serials:
            continue
        require(
            re.fullmatch(r"[0-9]{1,2}", row["SPORDER"], re.ASCII) is not None
            and 1 <= int(row["SPORDER"]) <= MAX_HOUSEHOLD_MEMBERS,
            "ACS_SOURCE_LINE",
        )
        key = (serial, int(row["SPORDER"]))
        require(key not in selected, "DUPLICATE_SELECTED_ACS_ROW")
        require(len(selected) < MAX_ROSTER_ROWS, "ACS_ROSTER_BOUND")
        selected[key] = row["RELSHIPP"]
    require(header is not None, "ACS_SOURCE_HEADER")
    return count


def _acs_states(serials, roster):
    """Classify every retained ACS household through the maintained owner.

    Households are numbered densely in a deterministic order; the owner reads
    only that grouping and the printed codes, and never a line number. Every
    retained person's own coordinate must appear in the roster the member
    supplied, so a retained line the published household does not carry refuses.
    """
    require(
        {serial for serial, _line in roster} == set(serials),
        "ACS_ROSTER_HOUSEHOLD_SCOPE",
    )
    for serial, lines in serials.items():
        require(all((serial, line) in roster for line in lines), "ACS_ROSTER_COVERAGE")
    ordered = sorted(roster)
    household_index = {serial: index for index, serial in enumerate(sorted(serials))}
    household_id = np.asarray(
        [household_index[serial] + 1 for serial, _line in ordered], dtype=np.int64
    )
    codes, known = [], []
    for key in ordered:
        code, is_known = acs_relationship_token(roster[key])
        codes.append(code)
        known.append(is_known)
    relshipp = np.asarray(codes, dtype=np.int64)
    states, diagnostics = demographic.acs_household_reference_states(
        household_id=household_id, relshipp=relshipp
    )
    code_known = np.asarray(known, dtype=bool)
    require(bool((states[~code_known] == 0).all()), "ACS_UNPRINTED_CODE_BOUND")
    resolved = {
        key: (int(states[index]), int(relshipp[index]), bool(code_known[index]))
        for index, key in enumerate(ordered)
    }
    return resolved, diagnostics


def _seal_inputs(rows):
    """Every projected row must carry exactly the declared observation keys."""
    declared = {
        ROLE_STATE_COLUMN,
        UNBOUND_REASON_COLUMN,
        HOUSEHOLD_VERDICT_COLUMN,
        VALUE_COLUMN,
        VALUE_KNOWN_COLUMN,
        ORIGIN_COLUMN,
        UNIVERSE_COLUMN,
        OBSERVATION_YEAR_COLUMN,
    }
    for row in rows:
        require(set(row) == declared, "OBSERVATION_KEYS")


def _project(origins, asec_selected, acs_selected):
    """Assemble the rowwise projection on the complete original-person axis."""
    require(
        set(asec_selected) | set(acs_selected) == set(origins.index)
        and not set(asec_selected) & set(acs_selected)
        and len(asec_selected) + len(acs_selected) == len(origins),
        "PROJECTION_ROSTER",
    )
    rows = []
    for person_id in origins.index:
        survey = origins.at[person_id, "source"]
        if survey == "asec":
            code, state, reason = asec_selected[int(person_id)]
            observation = asec_household_role_observation(state, reason)
            code_known = code in demographic.A_EXPRRP.named_codes
        else:
            state, code, code_known = acs_selected[int(person_id)]
            observation = acs_household_role_observation(state, code)
        _seal_inputs([observation])
        require(observation[UNIVERSE_COLUMN] is not None, "OBSERVATION_UNIVERSE")
        rows.append(
            {
                SURVEY_COLUMN: survey,
                NATIVE_ID_COLUMN: int(origins.at[person_id, "native_person_id"]),
                CODE_COLUMN: None if code < 0 else int(code),
                CODE_KNOWN_COLUMN: bool(code_known),
                ALLOCATION_KNOWN_COLUMN: False,
                **observation,
            }
        )
    table = _observation_table(rows, origins.index.to_numpy(dtype=np.int64))
    require(
        bool(
            (
                table[VALUE_KNOWN_COLUMN].to_numpy()
                == ~np.asarray(pd.isna(table[VALUE_COLUMN]), dtype=bool)
            ).all()
        ),
        "PROJECTION_KNOWNNESS",
    )
    unprinted = ~table[CODE_KNOWN_COLUMN].to_numpy()
    require(
        bool((table[ROLE_STATE_COLUMN].to_numpy()[unprinted] == 0).all()),
        "PROJECTION_UNPRINTED_CODE_BOUND",
    )
    return table


def _counts(table):
    survey = table[SURVEY_COLUMN].astype(object).to_numpy()
    state = table[ROLE_STATE_COLUMN].to_numpy()
    return {
        "original_persons": int(len(table)),
        "acs_originals": int((survey == "acs").sum()),
        "asec_originals": int((survey == "asec").sum()),
        "code_known_originals": int(table[CODE_KNOWN_COLUMN].to_numpy().sum()),
        "value_known_originals": int(table[VALUE_KNOWN_COLUMN].to_numpy().sum()),
        "reference_person_originals": int((state == 1).sum()),
        "acs_group_quarters_originals": int(
            (table[UNBOUND_REASON_COLUMN].to_numpy() == 5).sum()
        ),
    }


@dataclass(frozen=True)
class QualifiedCurrentSurveyHouseholdRoles:
    """A detached projection, not a substitute for the retained source owner."""

    source_frame: object
    origins: pd.DataFrame
    rows: pd.DataFrame
    projection: bytes
    receipt: bytes


def household_role_projection_seal(value):
    """The complete in-process identity of one qualified projection."""
    require(type(value) is QualifiedCurrentSurveyHouseholdRoles, "QUALIFIED_TYPE")
    require(
        value.projection == _projection_bytes(value.rows)
        and json.loads(value.receipt)["projection_sha256"] == _sha(value.projection),
        "PROJECTION_BINDING",
    )
    return (
        source._frame_identity(value.source_frame),
        _table_stamp(value.origins),
        _table_stamp(value.rows),
        value.projection,
        value.receipt,
    )


def _asec_observed(state, preparation):
    """Reconstruct the maintained ASEC demographic owner from its pinned members.

    The parent is the retained native population's own current-money source, as
    the maintained ASEC demographic qualifier beside this one already binds it.
    Nothing here supplies member bytes, a row count or a key list.
    """
    native = state.native[1]
    entry = source.asec_native._ISSUED.get(id(native))
    require(
        entry is not None and entry[0]() is native and native.payload == entry[1],
        "ASEC_NATIVE_ISSUANCE",
    )
    document = json.loads(entry[1])
    require(
        document["source_year"] == document["income_year"] == INCOME_YEAR
        and document["survey_year"] == ASEC_OBSERVATION_YEAR,
        "ASEC_NATIVE_PERIOD",
    )
    parent = entry[2].parent
    observed = demographic.load_authenticated_asec_demographic_source(
        parent,
        member_paths={
            year: state.root / "asec" / f"pppub{year - 1999}.csv"
            for year in sorted(demographic.COHORTS)
        },
    )
    require(
        observed.receipt["source_identity"] == parent.source.identity.decode(),
        "ASEC_PARENT_IDENTITY",
    )
    require(
        observed.receipt["household_membership"]["bijective"] is True
        and observed.receipt["household_membership"]["equivalence"]
        == "bidirectional_partition_bijection",
        "ASEC_HOUSEHOLD_MEMBERSHIP",
    )
    require(
        observed.receipt["diagnostics"]["p_seq_crosscheck"][
            "is_semantic_proof_of_headship"
        ]
        is False
        and observed.receipt["diagnostics"]["line_number_used_for_headship"] is False
        and observed.receipt["diagnostics"]["family_relationship_field_read"] is False,
        "ASEC_HEADSHIP_EVIDENCE",
    )
    require(preparation._checked() is not None, "PREPARATION_CHANGED")
    return observed, entry


def _asec_selected(origins, keys, observed):
    """Select the retained ASEC originals from the owner's numeric buffers.

    The join key is the preparation's own selected source id, which is the ASEC
    owner's ``person_id``; the published household id and roster line the
    preparation recorded are then compared with the owner's own coordinates, so
    a silent re-identification refuses instead of producing a plausible row.
    """
    ids = observed.array("person_id")
    years = observed.array("income_year")
    require(
        len(set(zip(years.tolist(), ids.tolist(), strict=True))) == len(ids),
        "ASEC_SOURCE_AXIS",
    )
    lookup = pd.MultiIndex.from_arrays((years, ids))
    arm = origins.loc[origins.source.eq("asec")]
    require(len(arm) == len(keys), "ASEC_ARM_ROSTER")
    person_ids = arm.index.to_numpy(dtype=np.int64)
    natives = arm.native_person_id.to_numpy(dtype=np.int64)
    positions = lookup.get_indexer(
        pd.MultiIndex.from_arrays(
            (np.full(len(arm), INCOME_YEAR, dtype=np.int64), natives)
        )
    )
    require(
        bool((positions >= 0).all()) and len(set(positions.tolist())) == len(positions),
        "ASEC_NATIVE_JOIN",
    )
    expected_line = np.asarray(
        [int(value) for value in arm.native_line_numeric_original], dtype=np.int64
    )
    expected_household = np.asarray(
        [int(value) for value in arm.raw_native_household_id], dtype=np.int64
    )
    require(
        np.array_equal(observed.array("A_LINENO")[positions], expected_line)
        and np.array_equal(
            observed.array("source_household_id")[positions], expected_household
        ),
        "ASEC_COORDINATE_DISAGREE",
    )
    code = observed.array("asec_A_EXPRRP")[positions]
    state = observed.array("asec_household_reference_state")[positions]
    reason = observed.array("asec_household_reference_unbound_reason")[positions]
    return {
        int(person_ids[index]): (
            int(code[index]),
            int(state[index]),
            int(reason[index]),
        )
        for index in range(len(person_ids))
    }


def _acs_roster(state, serials):
    """Re-read the pinned ACS person member for the retained household rosters.

    The catalogue is the retained ACS issuer; its pinned archive identity is
    re-verified before and after the read. Auxiliary members are exhausted so
    their CRC is checked, and the member's own row count must match the
    catalogue's, so a truncated read cannot look like an empty household.
    """
    owned = source.acs_catalogue._lookup(state.catalogues[0])
    document = json.loads(owned.receipt)
    person_pins = tuple(pin for pin in owned.pins if pin[0] == "person")
    require(
        len(person_pins) == 1
        and document["source_year"] == document["survey_year"] == ACS_OBSERVATION_YEAR,
        "ACS_PIN_OR_PERIOD",
    )
    _role, name, digest, size = person_pins[0]
    roster, count = {}, 0
    with tempfile.TemporaryDirectory(prefix="microcosm-household-roles-") as temporary:
        captured = Path(temporary) / name
        require(
            housing._copy(state.root / "acs" / name, captured, size, exact_size=size)
            == digest,
            "ACS_CAPTURE_DIGEST",
        )
        with zipfile.ZipFile(captured) as archive:
            members, prefix = records._members(archive, "person")
            for item in members:
                with archive.open(item) as stream:
                    if item.filename.casefold().startswith(prefix):
                        count += _scan_acs_relationships(
                            stream,
                            serials=serials,
                            selected=roster,
                            maximum=document["counts"]["people"] - count,
                        )
                    else:
                        while stream.read(65536):
                            pass
        require(
            count == document["counts"]["people"]
            and housing._persisted_sha(captured, size) == digest,
            "ACS_CAPTURE_CHANGED",
        )
    require(
        source.acs_catalogue._lookup(state.catalogues[0]) is owned, "ACS_OWNER_CHANGED"
    )
    return roster, owned


def qualify_current_survey_household_roles(preparation):
    """Qualify both arms' household reference-person roles from their owners.

    This entry point does read original microdata when called by an authorized
    build: the ASEC arm reconstructs the maintained demographic owner from its
    pinned CSV members and the ACS arm re-reads the pinned person archive for
    ``RELSHIPP``. Tests exercise the same path with privately pinned invented
    bytes. No engine, native Frame or harmonized relationship column is read.
    """
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    _shared_code_tables()
    requirement = _canonical_requirement()
    entry = preparation._checked()
    state = entry[2]
    document = json.loads(entry[1])
    origins = _origins(state.frame, document)
    acs_keys, serials = _acs_keys(origins)
    asec_keys = _asec_keys(origins)
    require(
        len(acs_keys) + len(asec_keys) == len(origins),
        "ORIGIN_ARM_PARTITION",
    )
    observed, native_entry = _asec_observed(state, preparation)
    roster, acs_owned = _acs_roster(state, serials)
    acs_states, acs_diagnostics = _acs_states(serials, roster)
    acs_selected = {person_id: acs_states[key] for key, person_id in acs_keys.items()}
    asec_selected = _asec_selected(origins, asec_keys, observed)
    table = _project(origins, asec_selected, acs_selected)
    projection = _projection_bytes(table)
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "canonical_column": CANONICAL_COLUMN,
            "canonical_requirement": requirement,
            "contract_sha256": contract_sha256(),
            "preparation_sha256": _sha(entry[1]),
            "asec_native_sha256": _sha(native_entry[1]),
            "asec_demographic_content_sha256": observed.content_sha256,
            "acs_catalogue_sha256": _sha(acs_owned.receipt),
            "acs_relationship_diagnostics": acs_diagnostics,
            "acs_roster_rows": int(len(roster)),
            "acs_roster_households": int(len(serials)),
            "projection_sha256": _sha(projection),
            "relationship_allocation_provenance": RELATIONSHIP_ALLOCATION_PROVENANCE,
            "asec_allocation_evidence": ASEC_ALLOCATION_EVIDENCE,
            "acs_allocation_evidence": ACS_ALLOCATION_EVIDENCE,
            "equivalent_allocation_evidence_claim": False,
            "unallocated_observation_claim": False,
            "positional_sequence_headship_claim": False,
            "harmonized_relationship_column_read": False,
            **_counts(table),
            "source_admission_issued": False,
            "population_admission_issued": False,
            "release_eligible": False,
        },
        maximum=MAX_RECEIPT_BYTES,
    )
    observed.validate()
    # The original issuers check source bytes after the last capture/cleanup I/O.
    require(preparation._checked() is entry, "PREPARATION_CHANGED")
    source._pure_final(state)
    require(
        source._ISSUED.get(id(preparation)) is entry
        and preparation.payload == entry[1]
        and source.asec_native._ISSUED.get(id(state.native[1])) is native_entry
        and state.native[1].payload == native_entry[1]
        and source.acs_catalogue._lookup(state.catalogues[0]) is acs_owned,
        "FINAL_OWNER",
    )
    return QualifiedCurrentSurveyHouseholdRoles(
        state.frame, origins, table, projection, receipt
    )


def _person_table(receiving):
    people = getattr(receiving, "person", None)
    require(type(people) is pd.DataFrame, "RECEIVING_PERSON_TABLE")
    return people


def canonical_dtype_token(receiving):
    """The token the canonical leaf must be declared with on this population.

    An absent leaf is created as a nullable Boolean. An incumbent keeps its own
    declared storage, because a rewrite must match the incumbent exactly.
    """
    people = _person_table(receiving)
    if CANONICAL_COLUMN not in people:
        return "boolean"
    try:
        token = token_for_dtype(people[CANONICAL_COLUMN].dtype)
    except (PopulationError, TypeError):
        raise ValueError(
            "CURRENT_SURVEY_HOUSEHOLD_ROLES_INCUMBENT_DTYPE_NOT_DECLARABLE"
        ) from None
    require(token in ("bool", "boolean"), "INCUMBENT_DTYPE")
    return token


def _incumbent(people, count):
    """Normalize the incumbent leaf to an explicit known mask and bool values."""
    if CANONICAL_COLUMN not in people:
        return np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
    values = pd.array(people[CANONICAL_COLUMN], dtype=dtype_for_token("boolean"))
    known = ~np.asarray(pd.isna(values), dtype=bool)
    resolved = np.zeros(count, dtype=bool)
    resolved[known] = np.asarray(values[known], dtype=bool)
    return known, resolved


def _aligned(qualified, receiving):
    """Fan each qualified original onto exactly its two unchanged clones."""
    require(type(qualified) is QualifiedCurrentSurveyHouseholdRoles, "QUALIFIED_TYPE")
    rows = qualified.rows
    require(
        tuple(rows.columns) == COLUMNS
        and rows.index.name == "person_id"
        and str(rows.index.dtype) == "int64"
        and rows.index.is_unique,
        "QUALIFIED_AXIS",
    )
    people = _person_table(receiving)
    require("person_id" in people, "RECEIVING_PERSON_AXIS")
    ids = people.person_id
    original = people[support_source_id_column("person")]
    clone = people[support_clone_index_column("person")]
    native = people[spine_source_id_column("person")]
    channel = people[support_channel_column("person")].astype(object)
    require(str(ids.dtype) == "int64" and ids.is_unique, "RECEIVING_ID_AXIS")
    require(
        str(original.dtype) == str(clone.dtype) == str(native.dtype) == "int64",
        "CLONE_AXIS_DTYPE",
    )
    original = original.to_numpy(dtype=np.int64)
    require(
        len(original) == 2 * len(rows) and set(original) == set(rows.index),
        "CLONE_SOURCE_ROSTER",
    )
    expected = rows.reindex(original)
    require(
        np.array_equal(
            native.to_numpy(dtype=np.int64),
            expected[NATIVE_ID_COLUMN].to_numpy(dtype=np.int64),
        )
        and list(channel) == list(expected[SURVEY_COLUMN].astype(object)),
        "CLONE_SOURCE_IDENTITY",
    )
    clone = clone.to_numpy(dtype=np.int64)
    pairs = pd.MultiIndex.from_arrays((original, clone))
    require(pairs.is_unique and bool(np.isin(clone, (0, 1)).all()), "CLONE_PAIR")
    require(
        bool(pd.Series(original).groupby(original).size().eq(2).all()),
        "WHOLE_CLONE_PAIRS",
    )
    return ids, clone, expected


def _bind(qualified, receiving):
    """Bind the source-backed role, preserving and verifying every known cell.

    An incumbent cell that the qualified projection also binds must agree: a
    contradiction refuses for explicit adjudication rather than being rewritten
    silently, because the ASEC incumbent shipped by ``relationship_inputs`` is
    the positional ``P_SEQ == 1`` signal the canonical requirement records as
    *not adopted*. An incumbent cell the projection leaves unbound is preserved
    unchanged and counted, because an unresolved observation is not evidence
    that the incumbent is wrong. Only a null cell whose binding is known is
    filled.
    """
    ids, clone, expected = _aligned(qualified, receiving)
    people = _person_table(receiving)
    count = len(people)
    token = canonical_dtype_token(receiving)
    qualified_known = expected[VALUE_KNOWN_COLUMN].to_numpy(dtype=bool)
    qualified_value = np.zeros(count, dtype=bool)
    values = expected[VALUE_COLUMN]
    present = ~np.asarray(pd.isna(values), dtype=bool)
    require(bool(np.array_equal(present, qualified_known)), "QUALIFIED_KNOWNNESS")
    qualified_value[present] = np.asarray(values[present], dtype=bool)
    incumbent_known, incumbent_value = _incumbent(people, count)
    both = incumbent_known & qualified_known
    conflict = both & (incumbent_value != qualified_value)
    require(not bool(conflict.any()), "CANONICAL_CONFLICT")
    resolved = incumbent_known | qualified_known
    bound = np.where(qualified_known, qualified_value, incumbent_value)
    require(token == "boolean" or bool(resolved.all()), "UNRESOLVED_UNDER_BOOL_LEAF")
    if token == "boolean":
        column = pd.array(bound, dtype=dtype_for_token("boolean"))
        column[~resolved] = pd.NA
    else:
        column = bound
    series = pd.Series(
        column,
        index=pd.Index(ids.to_numpy(dtype=np.int64), name="person_id"),
        name=CANONICAL_COLUMN,
    )
    summary = {
        "canonical_dtype": token,
        "receiving_rows": int(count),
        "clone_indices": sorted(int(value) for value in set(clone.tolist())),
        "preserved_cells": int(both.sum()),
        "preserved_unbound_cells": int((incumbent_known & ~qualified_known).sum()),
        "filled_cells": int((~incumbent_known & qualified_known).sum()),
        "unresolved_cells": int((~resolved).sum()),
        "agreeing_cells": int((both & ~conflict).sum()),
    }
    return series, summary


def household_role_columns_for_population(value, receiving_frame):
    """Return the one canonical leaf this fragment writes on a receiving frame."""
    series, _summary = _bind(value, receiving_frame)
    return {("person", CANONICAL_COLUMN): series}


def household_role_binding_receipt(
    value, receiving_frame, *, receiving_version, declared_rewrite
):
    """Closed public aggregate metadata; no row, identity or code may appear."""
    require(
        type(receiving_version) is str
        and 0 < len(receiving_version) <= 256
        and type(declared_rewrite) is bool,
        "RECEIPT_INPUT",
    )
    _series, summary = _bind(value, receiving_frame)
    document = json.loads(value.receipt)
    receipt = {
        "protocol": PROTOCOL,
        "canonical_column": CANONICAL_COLUMN,
        "declared_rewrite": declared_rewrite,
        "contract_sha256": document["contract_sha256"],
        "projection_sha256": document["projection_sha256"],
        "receiving_version": receiving_version,
        "relationship_allocation_provenance": RELATIONSHIP_ALLOCATION_PROVENANCE,
        "equivalent_allocation_evidence_claim": False,
        "unallocated_observation_claim": False,
        "positional_sequence_headship_claim": False,
        "source_admission_issued": False,
        "population_admission_issued": False,
        "release_eligible": False,
        **summary,
        **{
            key: document[key]
            for key in (
                "original_persons",
                "acs_originals",
                "asec_originals",
                "code_known_originals",
                "value_known_originals",
                "reference_person_originals",
                "acs_group_quarters_originals",
            )
        },
    }
    require(set(receipt) == PUBLIC_RECEIPT_KEYS, "PUBLIC_RECEIPT_ROSTER")
    require(
        all(
            type(v) is bool
            or type(v) is int
            or type(v) is str
            or (type(v) is list and all(type(i) is int for i in v))
            for v in receipt.values()
        ),
        "PUBLIC_RECEIPT_AGGREGATES",
    )
    return receipt


def household_role_reconciliation(value, receiving_frame):
    """Describe the incumbent/qualified relation without binding or refusing.

    A reviewer needs the disagreement counts *before* the binding gate trips on
    them, because the shipped ASEC incumbent is the positional signal the
    canonical requirement refuses. This is a pure aggregate description: it
    writes no column, claims no source admission, and adjudicates nothing.
    """
    ids, _clone, expected = _aligned(value, receiving_frame)
    people = _person_table(receiving_frame)
    count = len(people)
    qualified_known = expected[VALUE_KNOWN_COLUMN].to_numpy(dtype=bool)
    qualified_value = np.zeros(count, dtype=bool)
    present = ~np.asarray(pd.isna(expected[VALUE_COLUMN]), dtype=bool)
    qualified_value[present] = np.asarray(expected[VALUE_COLUMN][present], dtype=bool)
    incumbent_known, incumbent_value = _incumbent(people, count)
    channel = np.asarray(
        [str(v) for v in expected[SURVEY_COLUMN].astype(object)], dtype=object
    )
    both = incumbent_known & qualified_known
    conflict = both & (incumbent_value != qualified_value)
    result = {
        "protocol": PROTOCOL,
        "canonical_column": CANONICAL_COLUMN,
        "incumbent_present": bool(CANONICAL_COLUMN in people),
        "receiving_rows": int(count),
        "receiving_person_ids": int(len(ids)),
        "compared_cells": int(both.sum()),
        "agreeing_cells": int((both & ~conflict).sum()),
        "conflicting_cells": int(conflict.sum()),
        "incumbent_head_qualified_not_head": int((conflict & incumbent_value).sum()),
        "qualified_head_incumbent_not_head": int((conflict & ~incumbent_value).sum()),
        "incumbent_known_qualified_unbound": int(
            (incumbent_known & ~qualified_known).sum()
        ),
        "qualified_known_incumbent_unknown": int(
            (~incumbent_known & qualified_known).sum()
        ),
        "unresolved_cells": int((~(incumbent_known | qualified_known)).sum()),
        "binding_would_refuse": bool(conflict.any()),
        "by_survey": {
            survey: {
                "rows": int((channel == survey).sum()),
                "compared_cells": int((both & (channel == survey)).sum()),
                "conflicting_cells": int((conflict & (channel == survey)).sum()),
                "qualified_unbound_cells": int(
                    (~qualified_known & (channel == survey)).sum()
                ),
            }
            for survey in SURVEYS
        },
        "unbound_reason_counts": {
            ROLE_UNBOUND_REASONS[code]: int(
                (expected[UNBOUND_REASON_COLUMN].to_numpy() == code).sum()
            )
            for code in sorted(ROLE_UNBOUND_REASONS)
        },
        "household_verdict_counts": {
            HOUSEHOLD_VERDICTS[code]: int(
                (expected[HOUSEHOLD_VERDICT_COLUMN].to_numpy() == code).sum()
            )
            for code in sorted(HOUSEHOLD_VERDICTS)
        },
        "positional_sequence_headship_claim": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    return result


_SOURCE_BYTES = _source_bytes()
_LIVE = _live()
