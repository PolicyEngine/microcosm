"""Exact categorical demographics from retained original ACS and ASEC members.

A descriptive projection borrows genuine preparation authority. Raw literals and
unresolved categories survive; no coarse donor-model representative is presented
as a full CPS race code. Hosts retain this object and requalify after final I/O.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType

import numpy as np
import pandas as pd

from . import current_survey_health_coverage as attachment
from . import current_survey_health_source as original
from . import current_survey_person_status as literals
from . import graph_full_puf_enrichment as physical
from . import source_csv_builtin
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.current-survey-race-hispanic-source.v1"
STRING = pd.StringDtype(storage="python", na_value=pd.NA)
OUTPUTS = ("cps_race", "is_hispanic")
ASEC_FIELDS = ("PRDTRACE", "PXRACE1", "PEHSPNON", "PXHSPNON", "PRDTHSP")
ACS_INDICATORS = ("RACWHT", "RACBLK", "RACAIAN", "RACASN", "RACNH", "RACPI", "RACSOR")
ACS_FIELDS = ("RAC1P", "FRACP", "HISP", "FHISP", *ACS_INDICATORS, "RACNUM")
ASEC_COLUMNS = (*original.ASEC_KEYS, *ASEC_FIELDS)
ACS_COLUMNS = (*original.ACS_KEYS, *ACS_FIELDS)
LITERAL_FIELDS = (*ASEC_FIELDS, *ACS_FIELDS)
PREFIX = "survey_demographic_"
RAW_COLUMNS = (
    PREFIX + "source",
    PREFIX + "observation_year",
    *(PREFIX + c for c in LITERAL_FIELDS),
    PREFIX + "race_allocation",
    PREFIX + "hispanic_allocation",
    PREFIX + "race_mapping_status",
    PREFIX + "hispanic_mapping_status",
)
# The shared PX labels target disability items, whose code0 wording differs.
# These printed demographic flag entries explicitly call0 not allocated and
# omit -1; reuse the common labels only where the source definitions coincide.
PX_CODES = MappingProxyType(
    {
        code: ("not_allocated" if code == 0 else label)
        for code, label in literals.PX_CODES.items()
        if code >= 0
    }
)
PX_VALUE_CODES = frozenset(
    (0, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43)
)
# W White, B Black, I American Indian/Alaska Native, A Asian, H NH/Pacific.
# All five groups and all combinations are covered by the published26 codes.
CPS_COMBINATIONS = MappingProxyType(
    {
        "W": 1,
        "B": 2,
        "I": 3,
        "A": 4,
        "H": 5,
        "WB": 6,
        "WI": 7,
        "WA": 8,
        "WH": 9,
        "BI": 10,
        "BA": 11,
        "BH": 12,
        "IA": 13,
        "IH": 14,
        "AH": 15,
        "WBI": 16,
        "WBA": 17,
        "WBH": 18,
        "WIA": 19,
        "WIH": 20,
        "WAH": 21,
        "BIA": 22,
        "BIH": 25,
        "BAH": 25,
        "IAH": 25,
        "WBIA": 23,
        "WBIH": 26,
        "WBAH": 26,
        "WIAH": 24,
        "BIAH": 26,
        "WBIAH": 26,
    }
)
RAC1_SINGLE = MappingProxyType(
    {1: "W", 2: "B", 3: "I", 4: "I", 5: "I", 6: "A", 7: "H", 8: "S"}
)
DICTIONARY = MappingProxyType(
    {
        "asec_url": "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf",
        "asec_pages_1based": (27, 28, 31, 32),
        "acs_url": "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf",
        "acs_pages_1based": (63, 106, 111, 127, 129),
        "universe": "all persons; PRDTHSP detail only PEHSPNON=1",
        "units": "categorical person observation; no currency or period conversion",
    }
)


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_RACE_HISPANIC_" + reason)


def _live():
    functions = []
    for module in (
        sys.modules[__name__],
        original,
        original.asec,
        original.housing,
        original.records,
        literals,
        source_csv_builtin,
        attachment,
        physical,
    ):
        for name, value in vars(module).items():
            if type(value) is FunctionType:
                functions.append((module.__name__, name, source._function_seal(value)))
            elif isinstance(value, type) and value.__module__ == module.__name__:
                functions.append((module.__name__, name, value))
                for member, function in vars(value).items():
                    if isinstance(function, (classmethod, staticmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if type(function) is FunctionType:
                        functions.append(
                            (
                                module.__name__,
                                name,
                                member,
                                source._function_seal(function),
                            )
                        )
    return tuple(functions), (
        PROTOCOL,
        OUTPUTS,
        ASEC_FIELDS,
        ACS_INDICATORS,
        ACS_FIELDS,
        ASEC_COLUMNS,
        ACS_COLUMNS,
        LITERAL_FIELDS,
        PREFIX,
        RAW_COLUMNS,
        repr(STRING),
        type(PX_CODES),
        type(PX_VALUE_CODES),
        type(CPS_COMBINATIONS),
        type(RAC1_SINGLE),
        type(DICTIONARY),
        type(literals.PX_CODES),
        type(literals.ACS_FLAG_CODES),
        tuple(PX_CODES.items()),
        PX_VALUE_CODES,
        tuple(CPS_COMBINATIONS.items()),
        tuple(RAC1_SINGLE.items()),
        tuple(DICTIONARY.items()),
        tuple(literals.PX_CODES.items()),
        tuple(literals.ACS_FLAG_CODES.items()),
        _REVALIDATE_CODE,
    )


def _code(row, name, codes, width=2):
    return literals.literal_code(row[name], codes, width=width)[0]


def _allocation(row, name, survey):
    named = PX_CODES if survey == "asec" else literals.ACS_FLAG_CODES
    code, status = literals.literal_code(
        row[name], named, width=2 if survey == "asec" else 1
    )
    return code, named[code] if status == "named" else status


def _map_row(row, survey):
    """Pure exact mapping; source authority belongs to the surrounding capture."""
    require(survey in ("asec", "acs"), "SURVEY")
    if survey == "asec":
        race_flag, race_allocation = _allocation(row, "PXRACE1", survey)
        hisp_flag, hisp_allocation = _allocation(row, "PXHSPNON", survey)
        race = _code(row, "PRDTRACE", range(1, 27))
        race_status = (
            "exact_cps_code" if race is not None else "unresolved_race_literal"
        )
        if race_flag not in PX_VALUE_CODES:
            race_status = (
                "unresolved_race_allocation"
                if race_flag is None
                else "conflicting_race_allocation"
                if race is not None
                else "missing_race_per_allocation"
            )
            race = None
        answer = _code(row, "PEHSPNON", (1, 2), 1)
        detail = _code(row, "PRDTHSP", range(9), 1)
        consistent = (answer == 1 and detail is not None and 1 <= detail <= 8) or (
            answer == 2 and detail == 0
        )
        hispanic = answer == 1 if consistent else None
        hisp_status = (
            "exact_hispanic_item"
            if consistent
            else "unresolved_hispanic_item"
            if answer is None
            else "unavailable_hispanic_detail"
            if detail is None
            else "inconsistent_hispanic_universe"
        )
        if hisp_flag not in PX_VALUE_CODES:
            hisp_status = (
                "unresolved_hispanic_allocation"
                if hisp_flag is None
                else "conflicting_hispanic_allocation"
                if answer is not None
                else "missing_hispanic_per_allocation"
            )
            hispanic = None
    else:
        race_flag, race_allocation = _allocation(row, "FRACP", survey)
        hisp_flag, hisp_allocation = _allocation(row, "FHISP", survey)
        code = _code(row, "RAC1P", range(1, 10), 1)
        bits = {c: _code(row, c, (0, 1), 1) for c in ACS_INDICATORS}
        count = _code(row, "RACNUM", range(1, 7), 1)
        race, race_status = None, "unresolved_race_literal"
        if (
            code is not None
            and count is not None
            and all(v is not None for v in bits.values())
        ):
            groups = (
                bits["RACWHT"],
                bits["RACBLK"],
                bits["RACAIAN"],
                bits["RACASN"],
                int(bool(bits["RACNH"] or bits["RACPI"])),
                bits["RACSOR"],
            )
            key = "".join(
                c for c, present in zip("WBIAHS", groups, strict=True) if present
            )
            consistent = count == sum(groups) and (
                (code == 9 and count >= 2)
                or (code in RAC1_SINGLE and count == 1 and key == RAC1_SINGLE[code])
            )
            if not consistent:
                race_status = "inconsistent_race_detail"
            elif bits["RACSOR"]:
                race_status = "no_exact_cps_category_some_other_race"
            else:
                require(key in CPS_COMBINATIONS, "MAPPING_INCOMPLETE")
                race, race_status = CPS_COMBINATIONS[key], "exact_category_crosswalk"
        if race_flag not in (0, 1):
            race, race_status = None, "unresolved_race_allocation"
        hisp = _code(row, "HISP", range(1, 25))
        hispanic = None if hisp is None else hisp != 1
        hisp_status = (
            "exact_hispanic_category"
            if hisp is not None
            else "unresolved_hispanic_literal"
        )
        if hisp_flag not in (0, 1):
            hispanic, hisp_status = None, "unresolved_hispanic_allocation"
    return race, hispanic, race_allocation, hisp_allocation, race_status, hisp_status


def recode(raw):
    require(
        type(raw) is pd.DataFrame
        and tuple(raw) == RAW_COLUMNS
        and raw.index.is_unique
        and raw.index.dtype == np.dtype("int64")
        and raw.index.name == "person_id",
        "RAW_AXIS",
    )
    rows = []
    for _, row in raw.iterrows():
        survey = row[PREFIX + "source"]
        fields = ASEC_FIELDS if survey == "asec" else ACS_FIELDS
        require(
            survey in ("asec", "acs")
            and row[PREFIX + "observation_year"]
            == (2025 if survey == "asec" else 2024),
            "RAW_PERIOD",
        )
        values = _map_row({c: row[PREFIX + c] for c in fields}, survey)
        require(
            tuple(
                row[PREFIX + c]
                for c in (
                    "race_allocation",
                    "hispanic_allocation",
                    "race_mapping_status",
                    "hispanic_mapping_status",
                )
            )
            == values[2:],
            "RAW_MAPPING_STATUS",
        )
        rows.append(values[:2])
    return pd.DataFrame(
        {
            OUTPUTS[0]: pd.array([r[0] for r in rows], dtype="Int64"),
            OUTPUTS[1]: pd.array([r[1] for r in rows], dtype="boolean"),
        },
        index=raw.index.copy(),
    )


def _combine(origins, keys, selected):
    require(
        set(selected) == {"acs", "asec"}
        and all(
            set(selected[s]) == {k for arm, k in keys if arm == s} for s in selected
        ),
        "SELECTED_SOURCE_ROSTER",
    )
    raw = pd.DataFrame(index=origins.index.copy())
    raw[RAW_COLUMNS[0]] = pd.array(origins.source, dtype=STRING)
    raw[RAW_COLUMNS[1]] = origins.source.map({"asec": 2025, "acs": 2024}).to_numpy(
        dtype="int64"
    )
    for column in RAW_COLUMNS[2:]:
        raw[column] = pd.array([None] * len(raw), dtype=STRING)
    for (survey, key), pid in keys.items():
        row = selected[survey][key]
        require(original._key(row, survey) == key, "SOURCE_COORDINATE_CHANGED")
        for name in ASEC_FIELDS if survey == "asec" else ACS_FIELDS:
            raw.at[pid, PREFIX + name] = row[name]
        values = _map_row(row, survey)
        for name, value in zip(RAW_COLUMNS[-4:], values[2:], strict=True):
            raw.at[pid, name] = value
    recode(raw)
    return raw


def _scan(stream, *, survey, wanted, selected, maximum):
    """Exhaust a byte-bounded literal member; selected records remain strings.

    This helper grants no authority to a file, row count or caller key list.
    A complete authenticated source/roster is bound by the enclosing qualifier.
    """
    require(
        survey in ("asec", "acs")
        and source_csv_builtin.capture_csv_reader(csv) is not None,
        "SCANNER_OR_CSV_BINDING",
    )
    columns = ASEC_COLUMNS if survey == "asec" else ACS_COLUMNS
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
        key = original._key(row, survey)
        require(all(len(row[c]) <= 64 for c in columns), "SOURCE_TOKEN_BOUND")
        if key in wanted:
            require(key not in selected, "DUPLICATE_SELECTED_SOURCE_KEY")
            selected[key] = row
    require(header is not None, "SOURCE_HEADER")
    return count


def _capture(preparation, expected_entry=None):
    """Re-read pinned originals without native Frame or engine reconstruction.

    This entry point does read original microdata when called by an authorized
    build. Tests can exercise the same path with privately pinned invented bytes.
    """
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    require(expected_entry is None or entry is expected_entry, "PREPARATION_CHANGED")
    state = entry[2]
    document = json.loads(entry[1])
    origins, keys = original._origins(state.frame, document)
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
        prefix="microcosm-demographic-source-"
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
                            pass  # Exhaust auxiliary members so their CRC is checked.
        require(
            count == acs_doc["counts"]["people"]
            and original.housing._persisted_sha(captured, acs_size) == acs_digest,
            "ACS_CAPTURE_CHANGED",
        )
    raw = _combine(origins, keys, selected)
    evidence = source._encode(
        {
            "protocol": PROTOCOL,
            "preparation_sha256": source._sha(entry[1]),
            "asec_native_sha256": source._sha(issued[1]),
            "acs_catalogue_sha256": source._sha(acs_owned.receipt),
            "asec_person_member_sha256": digest,
            "acs_person_archive_sha256": acs_digest,
            "asec_observation_year": 2025,
            "acs_observation_year": 2024,
            "income_year": 2024,
            "dictionary": dict(DICTIONARY),
            "acs_race_crosswalk": dict(CPS_COMBINATIONS),
            "some_other_race_policy": "unresolved; no representative CPS code",
            "unknown_policy": "nullable canonical values; raw literals and mapping status retained",
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    # Full original-owner check after all capture/cleanup I/O; pure final seal last.
    require(preparation._checked() is entry, "PREPARATION_CHANGED")
    source._pure_final(state)
    require(
        source._ISSUED.get(id(preparation)) is entry
        and source.asec_native._ISSUED.get(id(native)) is issued
        and source.acs_catalogue._lookup(state.catalogues[0]) is acs_owned,
        "FINAL_OWNER",
    )
    return entry, origins, raw, evidence


@dataclass(frozen=True, eq=False)
class QualifiedSurveyRaceHispanic:
    source_frame: object
    origins: pd.DataFrame
    raw: pd.DataFrame
    evidence: bytes
    _revalidate: object = field(default=None, repr=False, compare=False)

    def validate(self):
        retained(self)
        self._revalidate(self)


def seal(value):
    require(type(value) is QualifiedSurveyRaceHispanic, "QUALIFIED_TYPE")
    return (
        source._frame_identity(value.source_frame),
        physical._table_stamp(value.origins),
        physical._table_stamp(value.raw),
        value.evidence,
    )


def retained(value):
    """Pure check of the exact projection closed over by the real qualifier."""
    require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    require(
        type(value) is QualifiedSurveyRaceHispanic
        and type(value._revalidate) is FunctionType
        and value._revalidate.__code__ is _REVALIDATE_CODE,
        "RETAINED_OWNER_REQUIRED",
    )
    cells = dict(
        zip(
            value._revalidate.__code__.co_freevars,
            (cell.cell_contents for cell in value._revalidate.__closure__),
            strict=True,
        )
    )
    require(
        cells["result"] is value and cells["revalidate"] is value._revalidate,
        "PROJECTION_OBJECT_CHANGED",
    )
    preparation, entry = cells["preparation"], cells["entry"]
    require(
        source._ISSUED.get(id(preparation)) is entry
        and entry[0]() is preparation
        and preparation.payload == entry[1]
        and value.source_frame is entry[2].frame
        and value.origins is cells["origins"]
        and value.raw is cells["raw"]
        and seal(value) == cells["stamp"],
        "RETAINED_PROJECTION_CHANGED",
    )


def qualify_current_survey_race_hispanic(preparation):
    require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    entry, origins, raw, evidence = _capture(preparation)
    result = QualifiedSurveyRaceHispanic(entry[2].frame, origins, raw, evidence)
    stamp = seal(result)

    def revalidate(candidate):
        require(
            candidate is result and candidate._revalidate is revalidate,
            "PROJECTION_OBJECT_CHANGED",
        )
        _entry, current_origins, current_raw, current_evidence = _capture(
            preparation, entry
        )
        require(
            _live() == _LIVE
            and candidate.source_frame is entry[2].frame
            and candidate.origins is origins
            and candidate.raw is raw
            and seal(candidate) == stamp
            and physical._table_stamp(current_origins) == physical._table_stamp(origins)
            and physical._table_stamp(current_raw) == physical._table_stamp(raw)
            and current_evidence == evidence,
            "PROJECTION_CHANGED",
        )

    object.__setattr__(result, "_revalidate", revalidate)
    retained(result)
    return result


_REVALIDATE_CODE = next(
    c
    for c in qualify_current_survey_race_hispanic.__code__.co_consts
    if type(c) is CodeType and c.co_name == "revalidate"
)
_LIVE = _live()
