"""Borrow original immigration literals without assigning legal-status labels.

The full current ASEC roster is retained for a later pre-selection stock draw.
This qualifier does not admit those stocks, their year alignment, or the rules
consumer. ACS has its own literal profile, never filled from ASEC defaults.
Only the retained preparation supplies source authority; a receipt cannot.
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
from types import CodeType, FunctionType, SimpleNamespace

import pandas as pd

from . import asec_current_money_source as physical
from . import current_survey_health_coverage as attachment
from . import current_survey_health_source as original
from . import source_csv_builtin
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.current-survey-immigration-source.v1"
RULES_HEAD = "7ee36ac1bea125218912996b1030b77a208db963"
PREFIX = "immigration_source_"
# Exact source readset of the future #779 consumer at RULES_HEAD. These are
# source literals, not a second enum or immigration-rule implementation. The
# incumbent immigration.py has the older wage-proxy interface and is unchanged.
ASEC_VALUE_COLUMNS = (
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
ACS_VALUE_COLUMNS = ("CIT", "POBP", "YOEP", "AGEP")
ASEC_COLUMNS = (*original.ASEC_KEYS, *ASEC_VALUE_COLUMNS)
ACS_COLUMNS = (*original.ACS_KEYS, *ACS_VALUE_COLUMNS)
STRING = pd.StringDtype(storage="python", na_value=pd.NA)


def require(condition, reason):
    """Refuse without printing an observation, native identifier or source path."""
    if not condition:
        raise ValueError("NATIVE_IMMIGRATION_SOURCE_" + reason)


def _live():
    functions = {}
    for module in (sys.modules[__name__], attachment, original):
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
        PROTOCOL,
        RULES_HEAD,
        PREFIX,
        ASEC_COLUMNS,
        ACS_COLUMNS,
        ASEC_VALUE_COLUMNS,
        ACS_VALUE_COLUMNS,
        _REVALIDATE_CODE,
        repr(STRING),
    )


def _implementation():
    return {
        module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (sys.modules[__name__], attachment, original, physical)
    }


def _scan(stream, *, survey, wanted, selected, complete, maximum):
    """Exhaust pinned bytes; retain the full ASEC readset before selection."""
    require(survey in ("asec", "acs"), "SURVEY")
    require(source_csv_builtin.capture_csv_reader(csv) is not None, "CSV_BINDING")
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
        require(all(len(row[c]) <= 64 for c in columns), "SOURCE_TOKEN_BOUND")
        key = original._key(row, survey)
        if key in wanted:
            require(key not in selected, "DUPLICATE_SELECTED_SOURCE_KEY")
            selected[key] = row
        if survey == "asec":
            require(key not in complete, "DUPLICATE_ASEC_SOURCE_KEY")
            complete[key] = row
    require(header is not None, "SOURCE_HEADER")
    return count


def _table(rows, columns, index):
    return pd.DataFrame(
        {c: pd.array([r[c] for r in rows], dtype=STRING) for c in columns},
        index=index.copy(),
    )


def _table_seal(table):
    require(
        type(table) is pd.DataFrame
        and table.index.is_unique
        and table.columns.is_unique,
        "TABLE_AXES",
    )
    digest = hashlib.sha256(b"microcosm/native-immigration-table/1\0")
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


def _full_roster(coverage):
    table = coverage.table()
    current = table.loc[table.source_year.eq(2024)].copy()
    keys = [
        (int(r.source_household_id), r.PERIDNUM, int(r.A_LINENO))
        for r in current.itertuples(index=False)
    ]
    require(
        len(keys) == len(set(keys))
        and len({k[1] for k in keys}) == len(keys)
        and len({(k[0], k[2]) for k in keys}) == len(keys),
        "FULL_ASEC_ROSTER",
    )
    return current, keys


def _combine(origins, keys, selected, complete, roster, roster_keys):
    require(
        set(selected) == {"asec", "acs"}
        and all(
            set(selected[s]) == {k for arm, k in keys if arm == s} for s in selected
        ),
        "SELECTED_SOURCE_ROSTER",
    )
    require(set(complete) == set(roster_keys), "FULL_ASEC_COVERAGE_ROSTER")
    for row, key in zip(roster.itertuples(index=False), roster_keys, strict=True):
        age = complete[key]["A_AGE"]
        require(
            age.isascii() and age.isdecimal() and int(age) == row.A_AGE,
            "FULL_ASEC_AGE_CROSSCHECK",
        )
    full = _table(
        [complete[k] for k in roster_keys],
        ASEC_COLUMNS,
        pd.Index(roster.person_id.to_numpy(), name="source_person_id"),
    )
    tables = {}
    evidence = pd.DataFrame(index=origins.index.copy())
    for survey, columns in (("asec", ASEC_COLUMNS), ("acs", ACS_COLUMNS)):
        ordered = [(key, pid) for (arm, key), pid in keys.items() if arm == survey]
        rows = [selected[survey][key] for key, _ in ordered]
        for (key, _), row in zip(ordered, rows, strict=True):
            require(original._key(row, survey) == key, "SOURCE_COORDINATE_CHANGED")
        table = _table(
            rows, columns, pd.Index([pid for _, pid in ordered], name="person_id")
        )
        tables[survey] = table
        for column in columns:
            require(PREFIX + column not in evidence, "SOURCE_FIELD_OVERLAP")
            # Missing means the other survey owns this person; an owning-source
            # empty literal remains an empty string. Neither becomes zero.
            evidence[PREFIX + column] = table[column].reindex(origins.index)
    evidence[PREFIX + "survey"] = pd.array(origins.source, dtype=STRING)
    evidence[PREFIX + "observation_year"] = origins.survey_year.to_numpy(copy=True)
    return full, tables["asec"], tables["acs"], evidence


@dataclass(frozen=True, eq=False)
class QualifiedSurveyImmigrationSources:
    """Private literal projections borrowed from one retained preparation.

    The receipt is descriptive. It neither authorizes stock draws nor substitutes
    for the source owner. Consumers validate after their final relevant I/O.
    """

    source_frame: object
    origins: pd.DataFrame
    asec_full_raw: pd.DataFrame
    asec_selected_raw: pd.DataFrame
    acs_raw: pd.DataFrame
    person_evidence: pd.DataFrame
    receipt: bytes
    _revalidate: object = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
        require(
            type(self) is QualifiedSurveyImmigrationSources
            and type(self._revalidate) is FunctionType
            and self._revalidate.__code__ is _REVALIDATE_CODE,
            "RETAINED_OWNER_REQUIRED",
        )
        self._revalidate(self)


def qualify_current_survey_immigration(
    preparation: source.AuthenticatedSurveyPopulationPreparation,
) -> QualifiedSurveyImmigrationSources:
    """Capture current original literals using the genuine retained source pins.

    Calling this reads microdata in a separately authorized build. Invented tests
    use the same preparation issuers and capture path with private fixture pins.
    No source archive, country model, or national-stock controls are downloaded.
    """
    require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    implementation = _implementation()
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
    parent, coverage = issued[2].parent, issued[2].coverage
    roster, roster_keys = _full_roster(coverage)
    coverage_header, coverage_body = coverage._header, coverage._body
    pins = tuple(original.asec._MEMBER_PINS)
    pin = tuple(p for p in pins if p[0] == 2024)
    require(len(pin) == 1, "ASEC_PIN_ROSTER")
    year, member, archive_digest, digest, rows, size = pin[0]
    retained = tuple(s for s in coverage.receipt["sources"] if s["source_year"] == year)
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
    selected, complete = {"asec": {}, "acs": {}}, {}
    with tempfile.TemporaryDirectory(
        prefix="microcosm-immigration-source-"
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
                complete=complete,
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
                            complete=complete,
                            maximum=acs_doc["counts"]["people"] - count,
                        )
                    else:
                        while stream.read(65536):
                            pass
        require(
            count == acs_doc["counts"]["people"]
            and original.housing._persisted_sha(captured, acs_size) == acs_digest,
            "ACS_CAPTURE_CHANGED",
        )
    full, asec_selected, acs_raw, person_evidence = _combine(
        origins, keys, selected, complete, roster, roster_keys
    )
    tables = (origins, full, asec_selected, acs_raw, person_evidence)
    seals = tuple(_table_seal(table) for table in tables)
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "rules_head": RULES_HEAD,
            "implementation_sha256": implementation,
            "preparation_sha256": source._sha(entry[1]),
            "asec_native_sha256": source._sha(issued[1]),
            "asec_coverage_sha256": coverage.content_sha256,
            "acs_catalogue_sha256": source._sha(acs_owned.receipt),
            "asec_person_member_sha256": digest,
            "acs_person_archive_sha256": acs_digest,
            "asec_income_year": 2024,
            "asec_observation_year": 2025,
            "acs_observation_year": 2024,
            "labor_force_evidence": "original_A_LFSR_current_at_ASEC2025_interview_not_prior_income",
            "projection_scope": "full_current_ASEC_literals_and_selected_original_survey_people",
            "raw_columns": {"asec": ASEC_COLUMNS, "acs": ACS_COLUMNS},
            "tables_sha256": dict(
                zip(
                    (
                        "origins",
                        "asec_full_raw",
                        "asec_selected_raw",
                        "acs_raw",
                        "person_evidence",
                    ),
                    seals,
                    strict=True,
                )
            ),
            "full_asec_rows": len(full),
            "selected_asec_rows": len(asec_selected),
            "selected_acs_rows": len(acs_raw),
            "status_assignment_performed": False,
            "national_stock_alignment_qualified": False,
            "unallocated_observation_claim": False,
            "source_admission_issued": False,
            "missing_rule_admissions": [
                "ASEC_observation_year_and_arrival_code_semantics",
                "national_stock_year_and_full_roster_design_weight_alignment",
                "ACS_status_transfer_or_assignment",
                "native_status_graph_attachment",
                "country_model_and_export_gates",
            ],
            "prior_income_columns_consumed": False,
        }
    )
    result = QualifiedSurveyImmigrationSources(
        state.frame, origins, full, asec_selected, acs_raw, person_evidence, receipt
    )

    def revalidate(candidate):
        require(
            candidate is result and candidate._revalidate is revalidate,
            "PROJECTION_OBJECT_CHANGED",
        )
        # File and foreign-owner checks come before the final pure seals.
        require(preparation._checked() is entry, "PREPARATION_CHANGED")
        coverage.validate()
        require(_implementation() == implementation, "IMPLEMENTATION_BYTES_CHANGED")
        require(
            _live() == _LIVE and tuple(original.asec._MEMBER_PINS) == pins,
            "IMPLEMENTATION_OR_PIN_CHANGED",
        )
        source._pure_final(state)
        require(
            source._ISSUED.get(id(preparation)) is entry
            and preparation.payload == entry[1]
            and source.asec_native._ISSUED.get(id(native)) is issued
            and native.payload == issued[1]
            and issued[2].parent is parent
            and issued[2].coverage is coverage
            and coverage._header == coverage_header
            and coverage._body == coverage_body
            and source.acs_catalogue._lookup(state.catalogues[0]) is acs_owned,
            "FINAL_OWNER",
        )
        require(
            candidate.source_frame is state.frame
            and candidate.receipt == receipt
            and candidate.origins is origins
            and candidate.asec_full_raw is full
            and candidate.asec_selected_raw is asec_selected
            and candidate.acs_raw is acs_raw
            and candidate.person_evidence is person_evidence
            and tuple(_table_seal(table) for table in tables) == seals,
            "PROJECTION_CHANGED",
        )

    object.__setattr__(result, "_revalidate", revalidate)
    result.validate()
    return result


def borrow_cloned_immigration_evidence(
    qualified: QualifiedSurveyImmigrationSources,
    receiving_people: pd.DataFrame,
) -> dict[tuple[str, str], pd.Series]:
    """Transport literals through exact two-clone ancestry, with no draws or fills.

    This authenticates no receiving population; a graph host must retain and
    verify that separate owner. Existing columns are never overwritten.
    """
    require(
        type(qualified) is QualifiedSurveyImmigrationSources, "RETAINED_OWNER_REQUIRED"
    )
    qualified.validate()
    require(type(receiving_people) is pd.DataFrame, "RECEIVING_TABLE")
    before = _table_seal(receiving_people)
    columns = attachment.attach_columns(
        qualified.origins,
        SimpleNamespace(person=receiving_people),
        qualified.person_evidence,
    )

    def seal():
        return _table_seal(
            pd.concat([s.rename(name) for (_, name), s in columns.items()], axis=1)
        )

    output_seal = seal()
    qualified.validate()
    require(_table_seal(receiving_people) == before, "RECEIVING_CHANGED")
    require(seal() == output_seal, "BORROWED_COLUMNS_CHANGED")
    return columns


_REVALIDATE_CODE = next(
    c
    for c in qualify_current_survey_immigration.__code__.co_consts
    if type(c) is CodeType and c.co_name == "revalidate"
)
_LIVE = _live()
