"""Issue selected 2024 ASEC observations with original household DESIGN anchors.

The closed parent and person-coverage owners still authenticate all three source
cohorts. Only the selected 2024 observations become this new population. Exact
key selection is an engineering operation with no inclusion-probability claim.
No pooling coefficient, age adjustment, coverage classification or allocation is
performed here. Source evidence stays outside the selected Frame's metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from microcosm.build.cd_benchmark import origin
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame import bundle as frame_owner
from microcosm.frame import weights as weight_owner

from . import asec_coverage_authentication as coverage_owner
from . import asec_current_money_source as parent_owner
from . import asec_household_coverage_fields as field_owner
from . import asec_original_household_weights as anchor_owner
from . import asec_person_income_source as restoration
from . import graph_context

ARTIFACT_KIND = "microcosm.us.asec-2024-native-population.v1"
_MAX_HOUSEHOLDS = 100_000
_MAX_PERSONS = 600_000
_MAX_EVIDENCE_BYTES = 128 * 1024**2
_MAX_CHECKPOINT_BYTES = 8 * 1024**3
_ISSUED: dict[int, tuple[weakref.ReferenceType, bytes, object]] = {}


class AsecNativePopulationError(ValueError):
    """Static refusal codes; no source paths, native keys or source values."""


def _require(condition, code):
    if not condition:
        raise AsecNativePopulationError(code)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _encode(value):
    result = bytearray()
    for part in json.JSONEncoder(
        ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).iterencode(value):
        part = part.encode("ascii")
        _require(len(result) + len(part) <= _MAX_EVIDENCE_BYTES, "EVIDENCE_SIZE")
        result.extend(part)
    return bytes(result)


def _modules():
    return (
        sys.modules[__name__],
        parent_owner,
        restoration,
        coverage_owner,
        anchor_owner,
        field_owner,
        graph_context,
        origin,
        frame_owner,
        weight_owner,
    )


def _implementation():
    runtime = {m.__name__: coverage_owner._runtime_code(m) for m in _modules()}
    _require(runtime == _RUNTIME_AUTHORITY, "PRODUCER_CODE_CHANGED")
    return {
        "kind": ARTIFACT_KIND,
        "runtime": runtime,
        "code": {m.__name__: _sha(Path(m.__file__).read_bytes()) for m in _modules()},
        "parent": parent_owner._verification_identity(),
        "restoration": restoration._implementation(),
        "coverage": coverage_owner._implementation(),
        "anchors": anchor_owner._implementation(),
        "household_fields": field_owner._producer(),
        "limits": [
            _MAX_HOUSEHOLDS,
            _MAX_PERSONS,
            _MAX_EVIDENCE_BYTES,
            _MAX_CHECKPOINT_BYTES,
        ],
        "periods": [2024, 2025, 2024],
    }


def _path(value):
    _require(type(value) is str or isinstance(value, Path), "SOURCE_PATH")
    return Path(value).absolute()


def _inputs(parent, household, restored, persons, member, selected, candidate):
    # Snapshot every caller-owned lookup before fingerprinting or source I/O.
    _require(isinstance(persons, Mapping) and len(persons) == 3, "PERSON_PATHS")
    paths = {}
    for year in persons:
        _require(
            type(year) is int
            and year in (2022, 2023, 2024)
            and year not in paths
            and len(paths) < 3,
            "PERSON_PATHS",
        )
        paths[year] = _path(persons[year])
    _require(set(paths) == {2022, 2023, 2024} and len(persons) == 3, "PERSON_PATHS")
    if selected is not None:
        _require(
            type(selected) is tuple and 0 < len(selected) <= _MAX_HOUSEHOLDS,
            "SELECTION",
        )
        for key in selected:
            _require(
                type(key) is tuple
                and len(key) == 2
                and all(type(v) is int for v in key)
                and key[0] == 2024
                and 1 <= key[1] <= 99999,
                "SELECTION",
            )
        _require(len(set(selected)) == len(selected), "SELECTION")
    _require(
        candidate is None
        or (type(candidate) is bytes and len(candidate) <= _MAX_EVIDENCE_BYTES),
        "CANDIDATE",
    )
    return (
        _path(parent),
        _path(household),
        _path(restored),
        paths,
        _path(member),
        selected,
        candidate,
    )


def _file_identity(path, expected, maximum, size=None):
    """Check original input bytes on every borrow, with bounded regular-file I/O."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
            "SOURCE_FILE_BOUNDS",
        )
        _require(size is None or before.st_size == size, "SOURCE_FILE_SIZE")
        digest, count = hashlib.sha256(), 0
        while block := os.read(descriptor, min(1024**2, before.st_size - count + 1)):
            count += len(block)
            _require(count <= before.st_size, "SOURCE_FILE_CHANGED")
            digest.update(block)
        after = os.fstat(descriptor)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _require(
            count == before.st_size
            and digest.hexdigest() == expected
            and all(getattr(before, k) == getattr(after, k) for k in keys),
            "SOURCE_FILE_CHANGED",
        )
    finally:
        os.close(descriptor)


def _frame_identity(frame):
    _require(
        type(frame) is Frame and frame.schema == US_SCHEMA and not frame.links,
        "FRAME_SHAPE",
    )
    _require(
        set(frame._tables) == set(US_SCHEMA.entities)
        and not frame._link_tables
        and set(frame._weights) == set(frame.weighted_entities),
        "FRAME_STORAGE",
    )
    # The parent signature binds typed cells, ordered axes, metadata, mass log,
    # schema and stored weights. Bind additional mutable pandas/ndarray state.
    decorations = []
    for name in frame.entities:
        table = frame.table(name)
        _require(type(table) is pd.DataFrame, "FRAME_TABLE_TYPE")
        decorations.append([name, table.attrs, table.flags.allows_duplicate_labels])
    decorations.append(
        ["strata", frame.strata.attrs, frame.strata.flags.allows_duplicate_labels]
    )
    weights = []
    for entity in frame.weighted_entities:
        vector = frame.weights_for(entity)
        _require(
            type(vector) is Weights and type(vector.values) is np.ndarray,
            "FRAME_WEIGHT_TYPE",
        )
        weights.append(
            [
                entity,
                vector.kind.value,
                str(vector.values.dtype),
                list(vector.values.shape),
                vector.values.flags.writeable,
                _sha(vector.values.tobytes()),
            ]
        )
    return _sha(
        _encode(
            {
                "base": parent_owner._frame_signature(frame),
                "decorations": decorations,
                "weights": weights,
            }
        )
    )


def _roster(parent, coverage, anchors, fields, selected):
    projection = coverage.table()
    people = parent.frame.person
    _require(len(projection) == len(people), "PARENT_ROSTER")
    for column in coverage_owner._PARENT_COLUMNS:
        _require(
            projection[column].tolist() == people[column].tolist(), "PARENT_ROSTER"
        )
    anchor_doc, field_doc = anchors.document, fields.document
    _require(
        anchor_doc["members"] == field_doc["members"]
        and len(anchor_doc["members"]) == 1,
        "HOUSEHOLD_MEMBER_BINDING",
    )
    member = anchor_doc["members"][0]
    source = next(s for s in coverage.receipt["sources"] if s["source_year"] == 2024)
    _require(
        member["income_year"] == 2024
        and member["survey_year"] == 2025
        and source["survey_year"] == 2025
        and member["archive_sha256"] == source["archive_sha256"],
        "COHORT_ARCHIVE_BINDING",
    )
    anchor_rows = {r["native_household_id"]: r for r in anchor_doc["records"]}
    field_rows = {r["native_household_id"]: r for r in field_doc["records"]}
    _require(
        len(anchor_rows) == len(anchor_doc["records"])
        and len(field_rows) == len(field_doc["records"])
        and anchor_rows.keys() == field_rows.keys(),
        "HOUSEHOLD_ROSTER",
    )
    native_to_receiving, receiving_to_native, person_keys = {}, {}, set()
    counts = {}
    for row in projection.itertuples(index=False):
        key = (int(row.source_year), int(row.source_household_id))
        receiving = int(row.person_household_id)
        _require(
            native_to_receiving.setdefault(key, receiving) == receiving
            and receiving_to_native.setdefault(receiving, key) == key,
            "HOUSEHOLD_PARTITION",
        )
        person_key = (key[0], row.PERIDNUM, int(row.A_LINENO))
        _require(person_key not in person_keys, "PERSON_ROSTER")
        person_keys.add(person_key)
        counts[key] = counts.get(key, 0) + 1
    all_keys = {key for key in native_to_receiving if key[0] == 2024}
    _require(bool(all_keys), "EMPTY_COHORT")
    for native, row in field_rows.items():
        anchor = anchor_rows[native]
        for column in (
            "income_year",
            "survey_year",
            "member_id",
            "member_sha256",
            "member_row_1based",
            "origin_key",
            "H_SEQ",
            "H_HHTYPE",
        ):
            _require(row[column] == anchor[column], "HOUSEHOLD_PROJECTION_BINDING")
        _require(
            row["income_year"] == 2024 and row["survey_year"] == 2025,
            "HOUSEHOLD_COHORT",
        )
        key = (2024, native)
        if row["H_HHTYPE"] == "1":
            _require(
                key in all_keys and row["reported_person_count"] == counts[key],
                "HOUSEHOLD_MEMBER_COUNT",
            )
        else:
            _require(key not in all_keys, "HOUSEHOLD_INTERVIEW_STATUS")
    _require(all(k[1] in field_rows for k in all_keys), "MISSING_HOUSEHOLD")
    chosen = all_keys if selected is None else set(selected)
    _require(chosen <= all_keys and len(chosen) <= _MAX_HOUSEHOLDS, "SELECTION_UNKNOWN")
    mask = np.array(
        [receiving_to_native[int(h)] in chosen for h in people.person_household_id],
        dtype=bool,
    )
    _require(0 < int(mask.sum()) <= _MAX_PERSONS, "SELECTED_PERSON_BOUNDS")
    selected_ids = set(people.loc[mask, "person_household_id"].tolist())
    ordered_ids = [
        int(h)
        for h in parent.frame.table("household").household_id
        if h in selected_ids
    ]
    ordered_keys = tuple(receiving_to_native[h] for h in ordered_ids)
    _require(set(ordered_keys) == chosen, "RECEIVING_HOUSEHOLDS")
    exact = anchors.original_weight_fractions(ordered_keys)
    rows = []
    for receiving, key, fraction in zip(ordered_ids, ordered_keys, exact, strict=True):
        row = anchor_rows[key[1]]
        expected_origin = origin.origin_key(
            origin.AsecHouseholdOrigin(
                origin.SourceMember(
                    member["canonical_member_id"], member["member_sha256"]
                ),
                2024,
                key[1],
            )
        )
        _require(row["origin_key"] == expected_origin, "ORIGIN_BINDING")
        rows.append(
            {
                "household_id": receiving,
                "native_key": list(key),
                "origin_key": expected_origin,
                "HSUP_WGT": row["HSUP_WGT"],
                "numerator": row["weight_integer_units"],
                "denominator": 100,
                "fraction": [fraction.numerator, fraction.denominator],
                "source": row,
                "household_fields": field_rows[key[1]],
            }
        )
    roster = projection.loc[mask].to_dict(orient="records")
    return mask, np.array([float(f) for f in exact], dtype=np.float64), rows, roster


def _descendant(parent, mask, anchors):
    """Select observations without using legacy weight slices as source anchors."""
    frame = parent.frame
    _require(frame.schema == US_SCHEMA and not frame.links, "PARENT_SCHEMA")
    person = frame.person.loc[mask].copy(deep=True)
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        column = US_SCHEMA.membership_column(entity)
        ids = set(person[column].tolist())
        # Referenced groups cannot silently retain only part of their members.
        _require(not frame.person.loc[~mask, column].isin(ids).any(), "SPLIT_GROUP")
        table = frame.table(entity)
        tables[entity] = table.loc[
            table[US_SCHEMA.id_column(entity)].isin(ids)
        ].reset_index(drop=True)
    descendant = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(anchors, WeightKind.DESIGN)},
        frame.strata.loc[mask].copy(deep=True),
        metadata={},
        mass_log=(),
    )
    parent_owner._detach_frame_axes(descendant)
    return descendant


class _State(NamedTuple):
    """Tuple storage keeps the external seal immutable even to object.__setattr__."""

    frame: Frame
    parent: object
    coverage: object
    anchors: object
    fields: object
    source_files: tuple
    producer: bytes
    parent_identity: str
    frame_identity: str
    context: bytes
    attached_evidence: tuple


def _attached_evidence(parent, coverage, anchors, fields):
    """Capture immutable issued bytes; the parent authority also has identity."""
    values = (
        parent.source._validate(),
        coverage._header,
        coverage._body,
        anchor_owner._checked_capsule_payload(anchors),
        anchor_owner._checked_capsule_payload(fields),
    )
    _require(all(type(value) is bytes for value in values), "ATTACHED_EVIDENCE_CHANGED")
    return (parent.source, *values)


def _validate_state(state):
    _require(_encode(_implementation()) == state.producer, "PRODUCER_CHANGED")
    _require(_frame_identity(state.frame) == state.frame_identity, "FRAME_CHANGED")
    _require(
        _frame_identity(state.parent.frame) == state.parent_identity, "PARENT_CHANGED"
    )
    state.parent.validate()
    coverage_owner.verify_asec_coverage_parent(state.coverage, state.parent)
    anchor_owner.verify_asec_household_weights_source(state.anchors)
    field_owner.verify_asec_household_coverage_fields(state.fields)
    for path, expected, maximum, size in state.source_files:
        _file_identity(path, expected, maximum, size)
    # File checks/foreign owner calls may yield. Recheck the actual receiving
    # frame and producer after them, not only before final source verification.
    _require(_encode(_implementation()) == state.producer, "PRODUCER_CHANGED")
    _require(
        _frame_identity(state.parent.frame) == state.parent_identity, "PARENT_CHANGED"
    )
    _require(
        _frame_identity(state.frame) == state.frame_identity
        and graph_context.encode_us_frame_context(state.frame) == state.context,
        "FRAME_CHANGED",
    )
    # These final seal checks perform no source/resource I/O. The immutable
    # tuple belongs to the external population seal, not mutable capsule state.
    current = _attached_evidence(
        state.parent, state.coverage, state.anchors, state.fields
    )
    _require(
        current[0] is state.attached_evidence[0]
        and current[1:] == state.attached_evidence[1:],
        "ATTACHED_EVIDENCE_CHANGED",
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class AuthenticatedAsec2024NativePopulation:
    """Process-issued population; every accessor verifies its full live binding."""

    payload: bytes

    def _checked(self):
        try:
            entry = _ISSUED.get(id(self))
            _require(
                type(self) is AuthenticatedAsec2024NativePopulation
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
        except AsecNativePopulationError:
            raise
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
        ):
            raise AsecNativePopulationError("NATIVE_BINDING_REFUSAL") from None

    def _state(self):
        return self._checked()[2]

    def validate(self):
        self._state()

    @property
    def frame(self):
        return self._state().frame

    @property
    def context(self):
        return self._state().context

    @property
    def receipt(self):
        return json.loads(self._checked()[1])

    def to_bytes(self):
        return self._checked()[1]


def load_authenticated_asec_2024_native_population(
    parent_path,
    household_attachment_path,
    person_income_attachment_path,
    *,
    person_member_paths,
    household_member_path,
    selected_households=None,
    candidate=None,
):
    """Reconstruct from closed source paths; candidate bytes never grant authority.

    ``selected_households`` is a nonempty unique tuple of ``(2024, H_SEQ)``
    integer keys, or None for the complete 2024 source cohort. This is not a
    probability sample. The authenticated parent/coverage still read all cohorts.
    """
    try:
        (
            parent_path,
            household_path,
            restored_path,
            persons,
            member,
            selection,
            candidate,
        ) = _inputs(
            parent_path,
            household_attachment_path,
            person_income_attachment_path,
            person_member_paths,
            household_member_path,
            selected_households,
            candidate,
        )
        producer = _encode(_implementation())
        parent = restoration.load_authenticated_restored_current_money_source(
            parent_path, household_path, restored_path, member_paths=persons
        )
        parent_identity = _frame_identity(parent.frame)
        coverage = coverage_owner.authenticate_asec_coverage(
            parent, member_paths=persons
        )
        anchors = anchor_owner.load_authenticated_asec_household_weights({2024: member})
        fields = field_owner.load_authenticated_asec_household_coverage_fields(
            {2024: member}
        )
        mask, weights, rows, roster = _roster(
            parent, coverage, anchors, fields, selection
        )
        frame = _descendant(parent, mask, weights)
        context = graph_context.encode_us_frame_context(frame)
        identity = json.loads(parent.source.identity)
        source_files = [
            (parent_path, identity["parent_sha256"], _MAX_CHECKPOINT_BYTES, None),
            (
                household_path,
                identity["attachment_sha256"],
                _MAX_CHECKPOINT_BYTES,
                None,
            ),
            (
                restored_path,
                identity["person_income_attachment_sha256"],
                _MAX_CHECKPOINT_BYTES,
                None,
            ),
        ]
        for source in coverage.receipt["sources"]:
            source_files.append(
                (
                    persons[source["source_year"]],
                    source["member_sha256"],
                    source["member_bytes"],
                    source["member_bytes"],
                )
            )
        pin = anchors.document["members"][0]
        source_files.append(
            (member, pin["member_sha256"], pin["size_bytes"], pin["size_bytes"])
        )
        state = _State(
            frame,
            parent,
            coverage,
            anchors,
            fields,
            tuple(source_files),
            producer,
            parent_identity,
            _frame_identity(frame),
            context,
            _attached_evidence(parent, coverage, anchors, fields),
        )
        payload = _encode(
            {
                "kind": ARTIFACT_KIND,
                "source_year": 2024,
                "income_year": 2024,
                "survey_year": 2025,
                "analysis_year": 2024,
                "selection": {
                    "kind": "complete_2024_cohort"
                    if selection is None
                    else "exact_engineering_keys",
                    "keys": [row["native_key"] for row in rows],
                    "inclusion_probabilities_known": False,
                },
                "households": rows,
                "persons": roster,
                "raw_age_relation": "original_A_AGE_identity_no_adjustment",
                "original_weight_conversion": "float64(Fraction(HSUP_WGT_integer,100))",
                "weight_kind": "design",
                "frame_sha256": state.frame_identity,
                "context_sha256": _sha(context),
                "producer_sha256": _sha(producer),
                "parent_custody": {
                    "cohorts": [2022, 2023, 2024],
                    "population_cohorts": [2024],
                    "parent_identity": identity,
                    "frame_sha256": parent_identity,
                },
                "person_coverage": coverage.receipt,
                "household_weights_sha256": _sha(anchors.payload),
                "household_fields_sha256": _sha(fields.payload),
                "coverage_classification_performed": False,
                "source_share_allocation_performed": False,
                "release_eligible": False,
            }
        )
        _require(candidate is None or candidate == payload, "CANDIDATE_MISMATCH")
        _validate_state(state)
        result = AuthenticatedAsec2024NativePopulation(payload)
        key = id(result)

        def discard(reference):
            entry = _ISSUED.get(key)
            if entry is not None and entry[0] is reference:
                del _ISSUED[key]

        _ISSUED[key] = (weakref.ref(result, discard), payload, state)
        result.validate()
        return result
    except AsecNativePopulationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise AsecNativePopulationError("NATIVE_SOURCE_REFUSAL") from None


# Pins can be substituted explicitly by invented tests. Callable implementation
# changes cannot silently establish a new producer merely by rebuilding receipts.
_RUNTIME_AUTHORITY = {m.__name__: coverage_owner._runtime_code(m) for m in _modules()}
