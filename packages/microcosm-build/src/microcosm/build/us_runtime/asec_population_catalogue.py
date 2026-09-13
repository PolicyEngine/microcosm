"""Authenticate the complete 2024 ASEC catalogue before population construction.

The closed parent still authenticates three cohorts. Only original 2024 records
are exposed here. No descendant Frame, unit assignment, domain classification,
selection, or weight allocation is performed. Household literals are retained;
A_AGE and A_LINENO are canonical strings of authenticated numeric originals.
"""

from __future__ import annotations

import hashlib
import json
import operator
import sys
import weakref
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import NamedTuple

from . import asec_2024_native_population as native
from . import survey_population_domains as domains

ARTIFACT_KIND = "microcosm.us.asec-source-catalogue.v1"
_MAX_ENVELOPE_BYTES = 1024**2
_MAX_RECORD_BYTES = 128 * 1024**2
_ISSUED: dict[int, tuple[weakref.ReferenceType, bytes, object]] = {}


class AsecSourceCatalogueError(ValueError):
    """Static refusal codes, without source paths or source values."""


def _require(condition, code):
    if not condition:
        raise AsecSourceCatalogueError(code)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _encode(value):
    # Only compact identities and individual records use this encoder. The
    # complete catalogue is hashed incrementally, never encoded as another body.
    result = bytearray()
    encoder = json.JSONEncoder(
        ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    for part in encoder.iterencode(value):
        part = part.encode("ascii")
        _require(len(result) + len(part) <= _MAX_ENVELOPE_BYTES, "ENVELOPE_SIZE")
        result.extend(part)
    return bytes(result)


def _implementation():
    modules = (sys.modules[__name__], domains)
    runtime = {m.__name__: native.coverage_owner._runtime_code(m) for m in modules}
    code = {m.__name__: _sha(Path(m.__file__).read_bytes()) for m in modules}
    _require(
        runtime == _RUNTIME_AUTHORITY and code == _BYTE_AUTHORITY,
        "PRODUCER_CODE_CHANGED",
    )
    return {
        "kind": ARTIFACT_KIND,
        "runtime": runtime,
        "code": code,
        "source_owner": native._implementation(),
        "limits": [_MAX_ENVELOPE_BYTES, _MAX_RECORD_BYTES],
        "periods": [2024, 2025],
    }


def _source_authority():
    """Snapshot exact code-owned source pin values without file/resource I/O.

    Canonical bytes detach the seal from mutable registry objects. The original
    household registry helper validates its records but reads no source files.
    """
    return _encode(
        {
            "parent": native.parent_owner._SOURCE_PINS,
            "restoration": native.restoration._MEMBER_PINS,
            "coverage": native.coverage_owner._MEMBER_PINS,
            "household": [asdict(pin) for pin in native.anchor_owner._registry()],
        }
    )


@dataclass(frozen=True, slots=True)
class UnrepresentedAsecHousehold:
    """Original noninterview household, absent from the authenticated roster.

    This is a custody ledger, not a scientific coverage classification. No empty
    person list is supplied as a claim about the household's actual membership.
    """

    key: domains.HouseholdKey
    h_hhtype: str
    hrhtype: str
    h_livqrt: str
    h_numper: str
    hsup_wgt: str
    reason: str = "original_noninterview_not_in_authenticated_person_roster"


def _key_record(key):
    _require(type(key) is domains.HouseholdKey, "RECORD_TYPE")
    _require(key.source is domains.Source.ASEC, "RECORD_SOURCE")
    return [key.source.value, key.source_year, key.survey_year, key.native_id]


def _record(row):
    _require(
        type(row) in (domains.AsecHousehold, UnrepresentedAsecHousehold),
        "RECORD_TYPE",
    )
    result = {
        "key": _key_record(row.key),
        "H_HHTYPE": row.h_hhtype,
        "HRHTYPE": row.hrhtype,
        "H_LIVQRT": row.h_livqrt,
        "H_NUMPER": row.h_numper,
        "HSUP_WGT": row.hsup_wgt,
    }
    if type(row) is UnrepresentedAsecHousehold:
        result["unrepresented_reason"] = row.reason
    else:
        _require(type(row.persons) is tuple, "RECORD_TYPE")
        persons = []
        for person in row.persons:
            _require(type(person) is domains.AsecPerson, "RECORD_TYPE")
            persons.append(
                [
                    person.peridnum,
                    person.a_lineno,
                    person.age,
                    person.prpertyp,
                    person.prpertyp_state,
                    _key_record(person.household_key),
                ]
            )
        result["persons"] = persons
    return result


class _RecordsMemo(NamedTuple):
    """The issued record roots, their projected leaves, and the identity streamed."""

    households: tuple
    ledger: tuple
    household_leaves: tuple
    ledger_leaves: tuple
    identity: tuple


def _records_identity(households, ledger, *, memo=False):
    """Stream every canonical record; optionally bind a leaf snapshot memo.

    With ``memo=True`` the result is ``(identity, memo)``. The memo's leaves are
    projected from the exact rows encoded in this same pass, so it can only
    stand for this identity. It is ``None`` when any record carries a leaf
    that is not an exact ``str``/``int`` or the ASEC source member.
    """
    _require(type(households) is tuple and type(ledger) is tuple, "RECORD_TYPE")
    digest, count, size, persons = hashlib.sha256(), 0, 0, 0
    household_leaves, ledger_leaves, eligible = [], [], memo
    for kind, rows in (("household", households), ("unrepresented", ledger)):
        for row in rows:
            data = _encode([kind, _record(row)]) + b"\n"
            size += len(data)
            _require(size <= _MAX_RECORD_BYTES, "RECORD_SIZE")
            digest.update(data)
            count += 1
            if kind == "household":
                persons += len(row.persons)
            if eligible:
                projected = _eligible_leaves(row, kind)
                if projected is None:
                    eligible = False
                elif kind == "household":
                    household_leaves.append(projected)
                else:
                    ledger_leaves.append(projected)
    identity = (digest.hexdigest(), size, count, persons)
    if not memo:
        return identity
    if not eligible:
        return identity, None
    return identity, _RecordsMemo(
        households, ledger, tuple(household_leaves), tuple(ledger_leaves), identity
    )


_KEY_LEAVES = operator.attrgetter("source", "source_year", "survey_year", "native_id")
_HOUSEHOLD_LITERALS = operator.attrgetter(
    "h_hhtype", "hrhtype", "h_livqrt", "h_numper", "hsup_wgt"
)
_HOUSEHOLD_LEAVES = operator.attrgetter(
    "key", "h_hhtype", "hrhtype", "h_livqrt", "h_numper", "hsup_wgt", "persons"
)
_PERSON_LITERALS = operator.attrgetter(
    "peridnum", "a_lineno", "age", "prpertyp", "prpertyp_state"
)
_PERSON_LEAVES = operator.attrgetter(
    "peridnum", "a_lineno", "age", "prpertyp", "prpertyp_state", "household_key"
)
_LEDGER_LITERALS = operator.attrgetter(
    "h_hhtype", "hrhtype", "h_livqrt", "h_numper", "hsup_wgt", "reason"
)
_LEDGER_LEAVES = operator.attrgetter(
    "key", "h_hhtype", "hrhtype", "h_livqrt", "h_numper", "hsup_wgt", "reason"
)


def _key_leaves(key):
    # The encoder writes ``source.value``; an enum member is a mutable object,
    # so its value object is a leaf in its own right. Classes are leaves too:
    # ``__class__`` can be reassigned in place, and the encoder refuses
    # anything but the exact record types on every full pass.
    source, *rest = _KEY_LEAVES(key)
    return (type(key), source, source.value, *rest)


def _household_leaves(row):
    key, *literals, persons = _HOUSEHOLD_LEAVES(row)
    return (
        type(row),
        key,
        *_key_leaves(key),
        *literals,
        type(persons),
        persons,
        *chain.from_iterable(
            (type(person), *_PERSON_LEAVES(person), *_key_leaves(person.household_key))
            for person in persons
        ),
    )


def _ledger_leaves(row):
    key, *literals = _LEDGER_LEAVES(row)
    return (type(row), key, *_key_leaves(key), *literals)


def _eligible_key(key):
    return (
        type(key) is domains.HouseholdKey
        and key.source is domains.Source.ASEC
        and type(key.source.value) is str
        and type(key.source_year) is int
        and type(key.survey_year) is int
        and type(key.native_id) is str
    )


def _eligible_leaves(row, kind):
    """Project one record's leaves when every one is an exact immutable value."""
    if not _eligible_key(row.key):
        return None
    if kind == "household":
        if type(row) is not domains.AsecHousehold or type(row.persons) is not tuple:
            return None
        literals = list(_HOUSEHOLD_LITERALS(row))
        for person in row.persons:
            if type(person) is not domains.AsecPerson or not _eligible_key(
                person.household_key
            ):
                return None
            literals.extend(_PERSON_LITERALS(person))
        leaves = _household_leaves(row)
    else:
        if type(row) is not UnrepresentedAsecHousehold:
            return None
        literals = _LEDGER_LITERALS(row)
        leaves = _ledger_leaves(row)
    if any(type(item) is not str for item in literals):
        return None
    return leaves


def _leaves_unchanged(rows, snapshot, project):
    if len(rows) != len(snapshot):
        return False
    for row, expected in zip(rows, snapshot, strict=True):
        try:
            current = project(row)
        except (AttributeError, TypeError):
            return False
        if len(current) != len(expected) or not all(
            map(operator.is_, current, expected)
        ):
            return False
    return True


def _memoized_records_identity(
    households, ledger, memo=None, *, expected_identity=None
):
    """Reuse the issued records identity only over the same unchanged leaves.

    A hit needs the exact issued household and ledger tuples, a memo identity
    equal to the owner's retained one, and every projected leaf (literals,
    keys, person rosters and their members' fields) to be the very object seen
    while the identity streamed. Frozen dataclasses stay writable through
    ``object.__setattr__``, so the leaves carry the proof, not the records.
    Anything else falls back to the full canonical encoding and its refusals.
    """
    if (
        type(households) is tuple
        and type(ledger) is tuple
        and type(memo) is _RecordsMemo
        and households is memo.households
        and ledger is memo.ledger
        and type(memo.household_leaves) is tuple
        and type(memo.ledger_leaves) is tuple
        and type(memo.identity) is tuple
        and memo.identity == expected_identity
        and _leaves_unchanged(households, memo.household_leaves, _household_leaves)
        and _leaves_unchanged(ledger, memo.ledger_leaves, _ledger_leaves)
    ):
        return memo.identity
    return _records_identity(households, ledger)


def _bound_records_identity(rows, roster):
    """Stream complete receiving/native keys, source positions and exact anchors."""
    digest, count, size = hashlib.sha256(), 0, 0
    for kind, values in (("household", rows), ("person", roster)):
        for row in values:
            data = _encode([kind, row]) + b"\n"
            size += len(data)
            _require(size <= _MAX_RECORD_BYTES, "RECORD_SIZE")
            digest.update(data)
            count += 1
    return {"sha256": digest.hexdigest(), "canonical_bytes": size, "records": count}


def _rows(rows, roster, anchors, fields):
    people = {}
    for person in roster:
        people.setdefault(int(person["source_household_id"]), []).append(person)
    households = []
    for row in sorted(rows, key=lambda r: r["source"]["member_row_1based"]):
        original = row["household_fields"]
        key = domains.HouseholdKey(domains.Source.ASEC, 2024, 2025, original["H_SEQ"])
        members = tuple(
            domains.AsecPerson(
                p["PERIDNUM"],
                str(int(p["A_LINENO"])),
                str(int(p["A_AGE"])),
                p["PRPERTYP"],
                p["PRPERTYP_state"],
                key,
            )
            for p in people[row["native_key"][1]]
        )
        households.append(
            domains.AsecHousehold(
                key,
                original["H_HHTYPE"],
                original["HRHTYPE"],
                original["H_LIVQRT"],
                original["H_NUMPER"],
                row["HSUP_WGT"],
                members,
            )
        )
    # _roster(None) already authenticates every original member row and refuses
    # interview rows absent from the represented roster or noninterviews in it.
    weights = {r["H_SEQ"]: r for r in anchors.document["records"]}
    ledger = tuple(
        UnrepresentedAsecHousehold(
            domains.HouseholdKey(domains.Source.ASEC, 2024, 2025, r["H_SEQ"]),
            r["H_HHTYPE"],
            r["HRHTYPE"],
            r["H_LIVQRT"],
            r["H_NUMPER"],
            weights[r["H_SEQ"]]["HSUP_WGT"],
        )
        for r in fields.document["records"]
        if r["H_HHTYPE"] != "1"
    )
    return tuple(households), ledger


class _State(NamedTuple):
    parent: object
    parent_frame: object
    coverage: object
    anchors: object
    fields: object
    source_files: tuple
    producer: bytes
    source_authority: bytes
    parent_identity: str
    attached_evidence: tuple
    households: tuple
    ledger: tuple
    records_identity: tuple
    records_memo: _RecordsMemo | None = None


def _current_records_identity(state):
    return _memoized_records_identity(
        state.households,
        state.ledger,
        state.records_memo,
        expected_identity=state.records_identity,
    )


def _validate_state(state):
    _require(_source_authority() == state.source_authority, "SOURCE_AUTHORITY_CHANGED")
    _require(_encode(_implementation()) == state.producer, "PRODUCER_CHANGED")
    _require(
        state.parent.frame is state.parent_frame
        and native._frame_identity(state.parent.frame) == state.parent_identity,
        "PARENT_CHANGED",
    )
    _require(
        _current_records_identity(state) == state.records_identity, "RECORDS_CHANGED"
    )
    state.parent.validate()
    native.coverage_owner.verify_asec_coverage_parent(state.coverage, state.parent)
    native.anchor_owner.verify_asec_household_weights_source(state.anchors)
    native.field_owner.verify_asec_household_coverage_fields(state.fields)
    for path, expected, maximum, size in state.source_files:
        native._file_identity(path, expected, maximum, size)
    _require(_encode(_implementation()) == state.producer, "PRODUCER_CHANGED")
    _require(
        state.parent.frame is state.parent_frame
        and native._frame_identity(state.parent.frame) == state.parent_identity,
        "PARENT_CHANGED",
    )
    _require(
        _current_records_identity(state) == state.records_identity, "RECORDS_CHANGED"
    )
    # Pure final seals after source/resource I/O; no recursively repeated reads.
    current = native._attached_evidence(
        state.parent, state.coverage, state.anchors, state.fields
    )
    _require(
        current[0] is state.attached_evidence[0]
        and current[1:] == state.attached_evidence[1:],
        "ATTACHED_EVIDENCE_CHANGED",
    )
    # Producer helpers read registry values before later helper source-code I/O.
    # A current borrow must refuse a late registry change, without waiting for
    # the next producer pass. This final comparison is pure and bounded.
    _require(_source_authority() == state.source_authority, "SOURCE_AUTHORITY_CHANGED")


@dataclass(frozen=True, slots=True, weakref_slot=True)
class AuthenticatedAsecSourceCatalogue:
    """Process-issued immutable records with checked borrows and compact evidence."""

    payload: bytes

    def _checked(self):
        try:
            entry = _ISSUED.get(id(self))
            _require(
                type(self) is AuthenticatedAsecSourceCatalogue
                and entry is not None
                and entry[0]() is self
                and type(self.payload) is bytes
                and self.payload == entry[1],
                "UNISSUED_OR_CHANGED",
            )
            _validate_state(entry[2])
            _require(
                _ISSUED.get(id(self)) is entry
                and type(self.payload) is bytes
                and self.payload == entry[1],
                "UNISSUED_OR_CHANGED",
            )
            return entry
        except AsecSourceCatalogueError:
            raise
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
        ):
            raise AsecSourceCatalogueError("CATALOGUE_BINDING_REFUSAL") from None

    @property
    def households(self) -> tuple[domains.AsecHousehold, ...]:
        return self._checked()[2].households

    @property
    def exclusion_ledger(self) -> tuple[UnrepresentedAsecHousehold, ...]:
        return self._checked()[2].ledger

    @property
    def receipt(self) -> dict:
        return json.loads(self._checked()[1])

    def to_bytes(self) -> bytes:
        return self._checked()[1]

    def validate(self) -> None:
        self._checked()


def verify_asec_source_catalogue(value: object) -> AuthenticatedAsecSourceCatalogue:
    _require(type(value) is AuthenticatedAsecSourceCatalogue, "UNISSUED_OR_CHANGED")
    value.validate()
    return value


def issue_asec_source_catalogue(
    parent_path: str | Path,
    household_attachment_path: str | Path,
    person_income_attachment_path: str | Path,
    *,
    person_member_paths: Mapping[int, str | Path],
    household_member_path: str | Path,
    candidate: bytes | None = None,
) -> AuthenticatedAsecSourceCatalogue:
    """Reconstruct the complete 2024 source catalogue from authenticated paths.

    Candidate bytes are compared only after actual reconstruction. No caller
    Frame, record list, weights or receipt can grant source authority.
    """
    try:
        _require(
            candidate is None
            or (type(candidate) is bytes and len(candidate) <= _MAX_ENVELOPE_BYTES),
            "CANDIDATE_SIZE_OR_TYPE",
        )
        parent_path, household_path, restored_path, persons, member, _, candidate = (
            native._inputs(
                parent_path,
                household_attachment_path,
                person_income_attachment_path,
                person_member_paths,
                household_member_path,
                None,
                candidate,
            )
        )
        source_authority = _source_authority()
        producer = _encode(_implementation())
        parent = native.restoration.load_authenticated_restored_current_money_source(
            parent_path, household_path, restored_path, member_paths=persons
        )
        parent_identity = native._frame_identity(parent.frame)
        coverage = native.coverage_owner.authenticate_asec_coverage(
            parent, member_paths=persons
        )
        anchors = native.anchor_owner.load_authenticated_asec_household_weights(
            {2024: member}
        )
        fields = native.field_owner.load_authenticated_asec_household_coverage_fields(
            {2024: member}
        )
        _, _, rows, roster = native._roster(parent, coverage, anchors, fields, None)
        households, ledger = _rows(rows, roster, anchors, fields)
        records_identity, records_memo = _records_identity(
            households, ledger, memo=True
        )
        identity = json.loads(parent.source.identity)
        files = [
            (
                parent_path,
                identity["parent_sha256"],
                native._MAX_CHECKPOINT_BYTES,
                None,
            ),
            (
                household_path,
                identity["attachment_sha256"],
                native._MAX_CHECKPOINT_BYTES,
                None,
            ),
            (
                restored_path,
                identity["person_income_attachment_sha256"],
                native._MAX_CHECKPOINT_BYTES,
                None,
            ),
        ]
        sources = coverage.receipt["sources"]
        for source in sources:
            files.append(
                (
                    persons[source["source_year"]],
                    source["member_sha256"],
                    source["member_bytes"],
                    source["member_bytes"],
                )
            )
        pin = anchors.document["members"][0]
        files.append(
            (member, pin["member_sha256"], pin["size_bytes"], pin["size_bytes"])
        )
        attached = native._attached_evidence(parent, coverage, anchors, fields)
        state = _State(
            parent,
            parent.frame,
            coverage,
            anchors,
            fields,
            tuple(files),
            producer,
            source_authority,
            parent_identity,
            attached,
            households,
            ledger,
            records_identity,
            records_memo,
        )
        payload = _encode(
            {
                "kind": ARTIFACT_KIND,
                "source_year": 2024,
                "survey_year": 2025,
                "counts": {
                    "households": len(households),
                    "persons": records_identity[3],
                    "unrepresented_households": len(ledger),
                },
                "records": {
                    "sha256": records_identity[0],
                    "canonical_bytes": records_identity[1],
                    "count": records_identity[2],
                },
                "native_binding": _bound_records_identity(rows, roster),
                "producer_sha256": _sha(producer),
                "source_authority_sha256": _sha(source_authority),
                "parent_custody": {
                    "cohorts": [2022, 2023, 2024],
                    "catalogue_cohorts": [2024],
                    "identity": identity,
                    "frame_sha256": parent_identity,
                },
                "source_members": sources,
                "household_member": pin,
                "attached_capsules_sha256": [_sha(value) for value in attached[1:]],
                "age_representation": "canonical_integer_string_of_authenticated_original_A_AGE_not_retained_CSV_lexeme",
                "line_representation": "canonical_integer_string_of_authenticated_original_A_LINENO_not_retained_CSV_lexeme",
                "household_key_representation": "original_H_SEQ_CSV_lexeme",
                "classification_performed": False,
                "weights_applied": False,
                "selection_performed": False,
                "release_eligible": False,
            }
        )
        _require(candidate is None or candidate == payload, "CANDIDATE_MISMATCH")
        _validate_state(state)
        result = AuthenticatedAsecSourceCatalogue(payload)
        key = id(result)

        def discard(reference):
            entry = _ISSUED.get(key)
            if entry is not None and entry[0] is reference:
                del _ISSUED[key]

        _ISSUED[key] = (weakref.ref(result, discard), payload, state)
        result.validate()
        return result
    except AsecSourceCatalogueError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise AsecSourceCatalogueError("CATALOGUE_SOURCE_REFUSAL") from None


_RUNTIME_AUTHORITY = {
    m.__name__: native.coverage_owner._runtime_code(m)
    for m in (sys.modules[__name__], domains)
}

_BYTE_AUTHORITY = {
    m.__name__: _sha(Path(m.__file__).read_bytes())
    for m in (sys.modules[__name__], domains)
}
