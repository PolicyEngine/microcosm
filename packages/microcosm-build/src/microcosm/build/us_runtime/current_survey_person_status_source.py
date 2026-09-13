"""Descriptive person status from retained original ACS and ASEC source owners.

No new source issuer, candidate reader, Frame mutation, eligibility mapping or
host is introduced. Consumers must retain this object and its preparation,
validate after their last relevant I/O, and independently bind any descendants.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FunctionType

import numpy as np
import pandas as pd

from . import asec_current_money_source as physical
from . import asec_student_controls as student
from . import current_survey_health_source as original
from . import current_survey_person_status as status
from . import source_csv_builtin
from . import survey_population_preparation as source

require = status.require
STRING = pd.StringDtype(storage="python", na_value=pd.NA)


def _live():
    functions = {}
    for module in (sys.modules[__name__], status, student, original):
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                functions[module.__name__, name] = source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                functions[module.__name__, name] = value
                for key, function in vars(value).items():
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        functions[module.__name__, name, key] = source._function_seal(
                            function
                        )
    return functions, (
        status.PROTOCOL,
        status.ASEC_PERIOD,
        status.ACS_PERIOD,
        status.ASEC_STUDENT_PERIOD,
        status.ACS_STUDENT_PERIOD,
        tuple(tuple(vars(item).items()) for item in status.ITEMS),
        tuple(status.PX_CODES.items()),
        tuple(status.AX_CODES.items()),
        tuple(status.ACS_FLAG_CODES.items()),
        status.ASEC_COLUMNS,
        status.ACS_COLUMNS,
        status.RAW_COLUMNS,
        status.OBSERVATIONS,
        student.COLUMNS,
        student.CONTROLS,
        student.COORDINATES,
        student._READ_COLUMNS,
        _REVALIDATE_CODE,
        repr(STRING),
    )


def _scan(stream, *, survey, wanted, selected, maximum):
    """Exhaust an authenticated bounded capture, retaining exact source literals."""
    require(survey in ("asec", "acs"), "SURVEY")
    require(source_csv_builtin.capture_csv_reader(csv) is not None, "CSV_BINDING")
    columns = status.ASEC_COLUMNS if survey == "asec" else status.ACS_COLUMNS
    header, count = None, 0
    for raw in original.records._records(stream):
        values = original.records._decode_record(raw, first=header is None)
        if header is None:
            header = values
            require(
                bool(header)
                and all(header)
                and len(set(header)) == len(header)
                and set(columns) <= set(header),
                "SOURCE_HEADER",
            )
            positions = [header.index(c) for c in columns]
            continue
        count += 1
        require(count <= maximum and len(values) == len(header), "SOURCE_ROW_SHAPE")
        row = dict(zip(columns, (values[p] for p in positions), strict=True))
        require(all(len(row[c]) <= 64 for c in columns), "SOURCE_TOKEN_BOUND")
        key = original._key(row, survey)
        if key in wanted:
            require(key not in selected, "DUPLICATE_SELECTED_SOURCE_KEY")
            selected[key] = row
    require(header is not None, "SOURCE_HEADER")
    return count


def _table_seal(table):
    require(
        type(table) is pd.DataFrame
        and table.index.is_unique
        and table.columns.is_unique,
        "TABLE_AXES",
    )
    digest = hashlib.sha256(b"microcosm/person-status-table/1\0")
    digest.update(physical._index_identity(table.index))
    digest.update(
        source._encode(
            (
                type(table.columns).__name__,
                str(table.columns.dtype),
                table.columns.name,
                tuple(table.columns),
            )
        )
    )
    for column in table:
        physical._series_digest(digest, table[column])
    return digest.hexdigest()


def _table(records, index):
    columns = tuple(dict.fromkeys(c for row in records for c in row))
    result = pd.DataFrame(index=index)
    for column in columns:
        values = [row.get(column) for row in records]
        known = [v for v in values if v is not None]
        if known and all(type(v) is bool for v in known):
            result[column] = pd.array(values, dtype="boolean")
        elif known and all(type(v) is int for v in known):
            result[column] = pd.array(values, dtype="Int64")
        else:
            result[column] = pd.array(values, dtype=STRING)
    # All-null observations remain typed nullable booleans, never empty strings.
    for column in result:
        if column in status.OBSERVATIONS or column.endswith(
            ("__applicable", "__agrees_with_battery")
        ):
            result[column] = pd.array(result[column], dtype="boolean")
    return result


def _combine(frame, origins, keys, selected):
    require(set(selected) == {"asec", "acs"}, "SOURCE_ARMS")
    require(
        all(set(selected[s]) == {k for arm, k in keys if arm == s} for s in selected),
        "SELECTED_SOURCE_ROSTER",
    )
    rows, observations = {}, {}
    frame_people = frame.person.set_index("person_id", drop=False)
    require(
        frame_people.index.equals(origins.index) and "A_AGE" in frame_people,
        "ORIGINAL_AGE_AXIS",
    )
    for (survey, key), pid in keys.items():
        row = selected[survey][key]
        require(original._key(row, survey) == key, "SOURCE_COORDINATE_CHANGED")
        age, parsed = status.literal_code(
            row["A_AGE" if survey == "asec" else "AGEP"], range(100)
        )
        require(
            parsed == "named"
            and not pd.isna(frame_people.at[pid, "A_AGE"])
            and age == frame_people.at[pid, "A_AGE"],
            "ORIGINAL_AGE_MISMATCH",
        )
        rows[pid] = {c: row.get(c) for c in status.RAW_COLUMNS}
        derived = status.recode_person_status(row, survey=survey)
        derived["survey_any_applicable_difficulty__applicable_fields"] = json.dumps(
            derived["survey_any_applicable_difficulty__applicable_fields"],
            separators=(",", ":"),
        )
        observations[pid] = derived
    return (
        _table([rows[int(pid)] for pid in origins.index], origins.index.copy()),
        _table([observations[int(pid)] for pid in origins.index], origins.index.copy()),
    )


def _student_join(parent, controls, origins, keys, selected):
    """Join issuer-owned numeric controls through exact full-parent coordinates."""
    controls.validate()
    require(type(controls) is student.AuthenticatedStudentControls, "STUDENT_TYPE")
    receipt = controls.receipt
    require(
        receipt["source_identity"] == parent.source.identity.decode()
        and receipt["annual_five_month_student_status_validated"] is False,
        "STUDENT_PARENT_OR_PERIOD",
    )
    arrays = {name: controls.array(name) for name in student.COLUMNS}
    ids = np.asarray(parent.scope.person_ids)
    years = np.asarray(parent.scope.person_years)
    native_keys = tuple(parent.scope.person_native_keys)
    require(
        np.array_equal(arrays["person_id"], ids)
        and np.array_equal(arrays["income_year"], years)
        and len(native_keys) == len(ids),
        "STUDENT_FULL_PARENT_AXIS",
    )
    lookup = {}
    for position, (year, pid) in enumerate(zip(years, ids, strict=True)):
        coordinate = (int(year), int(pid))
        require(coordinate not in lookup, "STUDENT_DUPLICATE_PARENT_ID")
        lookup[coordinate] = position
    for (survey, key), pid in keys.items():
        if survey != "asec":
            continue
        original_id = int(origins.at[pid, "native_person_id"])
        position = lookup.get((2024, original_id))
        require(
            position is not None and native_keys[position] == key[1],
            "STUDENT_NATIVE_KEY_JOIN",
        )
        row = selected["asec"][key]
        expected = {
            "source_household_id": key[0],
            "A_LINENO": key[2],
            "A_AGE": int(row["A_AGE"]),
        }
        require(
            all(
                int(arrays[name][position]) == value for name, value in expected.items()
            ),
            "STUDENT_NATIVE_COORDINATE_OR_AGE",
        )
        for name in student.CONTROLS:
            code, state = status.literal_code(row[name], (0, 1, 2), width=1)
            require(
                state == "named" and code == int(arrays[name][position]),
                "STUDENT_AUTHENTICATED_CONTROL_CONFLICT",
            )


@dataclass(frozen=True, eq=False)
class QualifiedSurveyPersonStatus:
    """Private detached values with a retained-owner revalidation callback.

    Construction/copying does not confer admission. The callback binds this
    exact value object; consumers cannot replace its owner with receipt JSON.
    """

    source_frame: object
    origins: pd.DataFrame
    raw: pd.DataFrame
    observations: pd.DataFrame
    receipt: bytes
    _revalidate: object = field(default=None, repr=False, compare=False)

    def validate(self):
        require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
        require(
            type(self) is QualifiedSurveyPersonStatus
            and type(self._revalidate) is FunctionType
            and self._revalidate.__code__ is _REVALIDATE_CODE,
            "RETAINED_OWNER_REQUIRED",
        )
        self._revalidate(self)


def qualify_current_survey_person_status(preparation):
    """Borrow actual preparation and existing three-cohort student authority.

    This callable reads microdata only inside an authorized build. It has no
    user-provided source pins or fallback to the preparation's carried cells.
    """
    require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state = entry[2]
    origins, keys = original._origins(state.frame, json.loads(entry[1]))
    native = state.native[1]
    issued = source.asec_native._ISSUED.get(id(native))
    require(
        issued is not None and issued[0]() is native and issued[1] == native.payload,
        "ASEC_NATIVE_ISSUANCE",
    )
    native_document = json.loads(issued[1])
    require(
        native_document["source_year"] == native_document["income_year"] == 2024
        and native_document["survey_year"] == 2025,
        "ASEC_NATIVE_PERIOD",
    )
    parent = issued[2].parent
    ready = parent.ready()
    # This is the maintained issuer's complete three-cohort read contract.
    student_pins = tuple(student._MEMBER_PINS)
    require(
        tuple(p[0] for p in student_pins) == (2022, 2023, 2024)
        and student_pins == tuple(original.asec._MEMBER_PINS),
        "STUDENT_SOURCE_PIN_ROSTER",
    )
    controls = student.load_authenticated_student_controls(
        parent,
        ready,
        member_paths={
            year: state.root / "asec" / name for year, name, *_ in student_pins
        },
    )
    controls_header, controls_body = controls._header, controls._body
    ready_header = ready.header
    ready_content = student.current_money_content_sha256(ready)
    pin = tuple(p for p in original.asec._MEMBER_PINS if p[0] == 2024)
    require(len(pin) == 1, "ASEC_PIN_ROSTER")
    year, member, archive_digest, digest, rows, size = pin[0]
    retained = tuple(
        s for s in issued[2].coverage.receipt["sources"] if s["source_year"] == year
    )
    require(
        len(retained) == 1
        and all(
            retained[0][k] == v
            for k, v in (
                ("member", member),
                ("archive_sha256", archive_digest),
                ("member_sha256", digest),
                ("rows", rows),
                ("member_bytes", size),
            )
        ),
        "ASEC_MEMBER_BINDING",
    )
    acs_owned = source.acs_catalogue._lookup(state.catalogues[0])
    acs_doc = json.loads(acs_owned.receipt)
    person_pins = tuple(p for p in acs_owned.pins if p[0] == "person")
    require(
        len(person_pins) == 1
        and acs_doc["source_year"] == acs_doc["survey_year"] == 2024,
        "ACS_PIN_OR_PERIOD",
    )
    _role, acs_name, acs_digest, acs_size = person_pins[0]
    selected = {"asec": {}, "acs": {}}
    with tempfile.TemporaryDirectory(
        prefix="microcosm-person-status-source-"
    ) as temporary:
        directory = Path(temporary)
        captured = directory / member
        identity = original.asec._capture(
            state.root / "asec" / member,
            captured,
            size=size,
            digest=digest,
            budget=[original.asec._BODY_MAX],
        )
        with captured.open("rb") as stream:
            count = _scan(
                stream,
                survey="asec",
                wanted={k for s, k in keys if s == "asec"},
                selected=selected["asec"],
                maximum=rows,
            )
        require(
            count == rows
            and original.asec._identity(captured.stat(follow_symlinks=False))
            == identity
            and original.housing._persisted_sha(captured, size) == digest,
            "ASEC_CAPTURE_CHANGED",
        )
        captured = directory / acs_name
        require(
            original.housing._copy(
                state.root / "acs" / acs_name, captured, acs_size, exact_size=acs_size
            )
            == acs_digest,
            "ACS_CAPTURE_DIGEST",
        )
        count = 0
        with zipfile.ZipFile(captured) as archive:
            members, prefix = original.records._members(archive, "person")
            for item in members:
                with archive.open(item) as stream:
                    if item.filename.casefold().startswith(prefix):
                        count += _scan(
                            stream,
                            survey="acs",
                            wanted={k for s, k in keys if s == "acs"},
                            selected=selected["acs"],
                            maximum=acs_doc["counts"]["people"] - count,
                        )
                    else:
                        while stream.read(65536):
                            pass  # Exhaust auxiliary members to check their CRC.
        require(
            count == acs_doc["counts"]["people"]
            and original.housing._persisted_sha(captured, acs_size) == acs_digest,
            "ACS_CAPTURE_CHANGED",
        )
    _student_join(parent, controls, origins, keys, selected)
    raw, observations = _combine(state.frame, origins, keys, selected)
    seals = (_table_seal(origins), _table_seal(raw), _table_seal(observations))
    receipt = source._encode(
        {
            "protocol": status.PROTOCOL,
            "preparation_sha256": source._sha(entry[1]),
            "asec_native_sha256": source._sha(issued[1]),
            "acs_catalogue_sha256": source._sha(acs_owned.receipt),
            "asec_person_member_sha256": digest,
            "acs_person_archive_sha256": acs_digest,
            "student_controls_sha256": controls.content_sha256,
            "student_controls_income_cohorts": [2022, 2023, 2024],
            "student_controls_selected_income_cohort": 2024,
            "asec_observation_year": 2025,
            "acs_observation_year": 2024,
            "asec_difficulty_period": status.ASEC_PERIOD,
            "acs_difficulty_period": status.ACS_PERIOD,
            "asec_student_period": status.ASEC_STUDENT_PERIOD,
            "acs_student_period": status.ACS_STUDENT_PERIOD,
            "read_columns": {"asec": status.ASEC_COLUMNS, "acs": status.ACS_COLUMNS},
            "origins_sha256": seals[0],
            "raw_sha256": seals[1],
            "observations_sha256": seals[2],
            "selected_rows": len(raw),
            "canonical_eligibility_assigned": False,
            "statutory_blindness_validated": False,
            "program_disability_validated": False,
            "annual_five_month_student_status_validated": False,
            "acs_full_time_workload_measured": False,
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    result = QualifiedSurveyPersonStatus(
        state.frame, origins, raw, observations, receipt
    )

    def revalidate(candidate):
        # Foreign owners may perform I/O; all retained values are checked after it.
        require(
            candidate is result and candidate._revalidate is revalidate,
            "PROJECTION_OBJECT_CHANGED",
        )
        require(preparation._checked() is entry, "PREPARATION_CHANGED")
        student._parent(parent, ready)
        controls.validate()
        require(
            _live() == _LIVE
            and tuple(student._MEMBER_PINS) == student_pins
            and tuple(original.asec._MEMBER_PINS) == student_pins,
            "IMPLEMENTATION_OR_PIN_CHANGED",
        )
        source._pure_final(state)
        require(
            source._ISSUED.get(id(preparation)) is entry
            and preparation.payload == entry[1]
            and source.asec_native._ISSUED.get(id(native)) is issued
            and native.payload == issued[1]
            and issued[2].parent is parent
            and source.acs_catalogue._lookup(state.catalogues[0]) is acs_owned,
            "FINAL_OWNER",
        )
        require(
            controls._header == controls_header
            and controls._body == controls_body
            and ready.header == ready_header
            and student.current_money_content_sha256(ready) == ready_content,
            "STUDENT_OR_READY_CHANGED",
        )
        require(
            candidate.source_frame is state.frame
            and candidate.receipt == receipt
            and candidate.origins is origins
            and candidate.raw is raw
            and candidate.observations is observations
            and (_table_seal(origins), _table_seal(raw), _table_seal(observations))
            == seals,
            "PROJECTION_CHANGED",
        )

    object.__setattr__(result, "_revalidate", revalidate)
    result.validate()
    return result


_REVALIDATE_CODE = next(
    c
    for c in qualify_current_survey_person_status.__code__.co_consts
    if type(c) is CodeType and c.co_name == "revalidate"
)
_LIVE = _live()
