"""ACS-only hours proposals borrowed from actual retained survey source owners.

This is not an all-person engine input, a source issuer, or release admission.
Selected ASEC literals remain uninterpreted for a later own-arm qualification.
Consumers retain this object and validate after their final relevant I/O.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import CodeType, FunctionType

import pandas as pd

from . import asec_current_money_source as physical
from . import current_survey_health_source as original
from . import current_survey_hours as hours
from . import source_csv_builtin
from . import survey_population_preparation as source

require = hours._require
PROTOCOL = "microcosm.us.native-usual-hours-source.v1"
_EXPECTED_DONORS = 2174
STRING = pd.StringDtype(storage="python", na_value=pd.NA)


def _live():
    functions = {}
    for module in (sys.modules[__name__], hours, original):
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
        _EXPECTED_DONORS,
        hours.PROTOCOL,
        hours.AGE15_POLICY,
        hours.UNDER15_POLICY,
        hours.SEED,
        hours.TARGET,
        hours.ACS_FIELDS,
        hours.ASEC_FIELDS,
        hours.ALLOCATION_FLAGS,
        _REVALIDATE_CODE,
        repr(STRING),
    )


def _implementation():
    return {
        module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (sys.modules[__name__], hours, original, physical)
    }


def _scan(stream, *, survey, wanted, selected, donors, maximum):
    """Exhaust every record; donors are never filtered by selected support."""
    require(survey in ("asec", "acs"), "SURVEY")
    require(source_csv_builtin.capture_csv_reader(csv) is not None, "CSV_BINDING")
    columns = hours.ASEC_FIELDS if survey == "asec" else hours.ACS_FIELDS
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
            age = hours._integer(row["A_AGE"], low=0, high=99)
            if age == 15:
                require(key not in donors, "DUPLICATE_DONOR_KEY")
                donors[key] = row
                require(len(donors) <= _EXPECTED_DONORS, "DONOR_COUNT")
    require(header is not None, "SOURCE_HEADER")
    return count


def _donor_keys(coverage):
    """Use full issued coverage, before native household selection."""
    table = coverage.table()
    cohort = table.loc[table.source_year.eq(2024) & table.A_AGE.eq(15)]
    keys = []
    for row in cohort.itertuples(index=False):
        key = hours.NativeHoursKey(
            "asec", 2025, 2024, str(row.source_household_id), row.PERIDNUM, row.A_LINENO
        )
        keys.append((int(key.household), key.person, key.line))
    require(
        len(keys) == _EXPECTED_DONORS
        and len(set(keys)) == len(keys)
        and len({k[1] for k in keys}) == len(keys)
        and len({(k[0], k[2]) for k in keys}) == len(keys),
        "AUTHENTICATED_DONOR_ROSTER",
    )
    return frozenset(keys)


def _table(records, columns, index):
    return pd.DataFrame(
        {name: pd.array([r[name] for r in records], dtype=STRING) for name in columns},
        index=index,
    )


def _table_seal(table):
    require(
        type(table) is pd.DataFrame
        and table.index.is_unique
        and table.columns.is_unique,
        "TABLE_AXES",
    )
    digest = hashlib.sha256(b"microcosm/native-hours-table/1\0")
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


def _proposal_seal(batch):
    require(type(batch) is hours.HoursProposalBatch, "PROPOSAL_TYPE")
    digest = hashlib.sha256(b"microcosm/native-hours-proposals/1\0")
    digest.update(
        source._encode(
            (
                batch.age15_policy,
                batch.under15_policy,
                batch.supplied_donors,
                batch.source_authenticated,
                batch.complete_donor_cohort_authenticated,
                batch.release_qualified,
            )
        )
    )
    require(type(batch.proposals) is tuple, "PROPOSAL_TYPE")
    for proposal in batch.proposals:
        require(type(proposal) is hours.HoursProposal, "PROPOSAL_TYPE")
        digest.update(source._encode(asdict(proposal)))
        digest.update(b"\0")
    return digest.hexdigest()


def _combine(frame, origins, keys, selected, donors, *, age15_policy, under15_policy):
    require(
        set(selected) == {"asec", "acs"}
        and all(
            set(selected[s]) == {k for arm, k in keys if arm == s} for s in selected
        ),
        "SELECTED_SOURCE_ROSTER",
    )
    people = frame.person.set_index("person_id", drop=False)
    require(
        people.index.equals(origins.index) and "A_AGE" in people, "ORIGINAL_AGE_AXIS"
    )
    by_id = {}
    for (survey, key), pid in keys.items():
        row = selected[survey][key]
        require(original._key(row, survey) == key, "SOURCE_COORDINATE_CHANGED")
        age = hours._integer(
            row["A_AGE" if survey == "asec" else "AGEP"], low=0, high=99
        )
        require(
            not pd.isna(people.at[pid, "A_AGE"]) and age == people.at[pid, "A_AGE"],
            "ORIGINAL_AGE_MISMATCH",
        )
        by_id[pid] = row
    acs_index = origins.index[origins.source.eq("acs")].copy()
    asec_index = origins.index[origins.source.eq("asec")].copy()
    acs_rows = [by_id[int(pid)] for pid in acs_index]
    asec_rows = [by_id[int(pid)] for pid in asec_index]
    donor_rows = [donors[key] for key in sorted(donors)]
    batch = hours.propose_acs_usual_hours(
        acs_rows,
        donors=donor_rows,
        age15_policy=age15_policy,
        under15_policy=under15_policy,
    )
    return (
        _table(acs_rows, hours.ACS_FIELDS, acs_index),
        _table(asec_rows, hours.ASEC_FIELDS, asec_index),
        _table(
            donor_rows, hours.ASEC_FIELDS, pd.RangeIndex(len(donor_rows), name="donor")
        ),
        batch,
    )


@dataclass(frozen=True, eq=False)
class QualifiedAcsHoursProposals:
    """Borrowed custody of ACS proposals, private literals and the full donor cohort.

    Exact object/callback identity is required. Neither a copied object nor
    this object's receipt is a source capability. There is no ASEC hours leaf.
    """

    source_frame: object
    origins: pd.DataFrame
    acs_raw: pd.DataFrame
    asec_selected_raw: pd.DataFrame
    donor_raw: pd.DataFrame
    proposals: hours.HoursProposalBatch
    receipt: bytes
    _revalidate: object = field(default=None, repr=False, compare=False)

    def validate(self):
        require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
        require(
            type(self) is QualifiedAcsHoursProposals
            and type(self._revalidate) is FunctionType
            and self._revalidate.__code__ is _REVALIDATE_CODE,
            "RETAINED_OWNER_REQUIRED",
        )
        self._revalidate(self)


def qualify_current_survey_hours(preparation, *, age15_policy, under15_policy):
    """Read only an authorized preparation's pinned original survey members.

    The complete current ASEC age15 roster is authenticated independently of
    selected native support. Both model policies require an explicit choice.
    """
    require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    require(age15_policy == hours.AGE15_POLICY, "AGE15_POLICY")
    require(under15_policy == hours.UNDER15_POLICY, "UNDER15_POLICY")
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
    expected_donors = _donor_keys(coverage)
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
    selected, donors = {"asec": {}, "acs": {}}, {}
    with tempfile.TemporaryDirectory(prefix="microcosm-hours-source-") as temporary:
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
                donors=donors,
                maximum=rows,
            )
        require(
            count == rows
            and original.asec._identity(captured.stat(follow_symlinks=False))
            == identity
            and original.housing._persisted_sha(captured, size) == digest,
            "ASEC_CAPTURE_CHANGED",
        )
        require(frozenset(donors) == expected_donors, "DONOR_COVERAGE_ROSTER")
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
                            donors=donors,
                            maximum=acs_doc["counts"]["people"] - count,
                        )
                    else:
                        while stream.read(65536):
                            pass  # Verify auxiliary member CRC as well.
        require(
            count == acs_doc["counts"]["people"]
            and original.housing._persisted_sha(captured, acs_size) == acs_digest,
            "ACS_CAPTURE_CHANGED",
        )
    acs_raw, asec_raw, donor_raw, proposals = _combine(
        state.frame,
        origins,
        keys,
        selected,
        donors,
        age15_policy=age15_policy,
        under15_policy=under15_policy,
    )
    tables = (origins, acs_raw, asec_raw, donor_raw)
    seals = tuple(_table_seal(table) for table in tables)
    proposal_seal = _proposal_seal(proposals)
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "proposal_protocol": hours.PROTOCOL,
            "implementation_sha256": implementation,
            "preparation_sha256": source._sha(entry[1]),
            "asec_native_sha256": source._sha(issued[1]),
            "asec_coverage_sha256": coverage.content_sha256,
            "acs_catalogue_sha256": source._sha(acs_owned.receipt),
            "asec_person_member_sha256": digest,
            "acs_person_archive_sha256": acs_digest,
            "income_year": 2024,
            "acs_survey_year": 2024,
            "asec_survey_year": 2025,
            "age15_policy": age15_policy,
            "under15_policy": under15_policy,
            "seed": hours.SEED,
            "projection_scope": "selected_original_acs_only",
            "selected_acs_rows": len(acs_raw),
            "selected_asec_literal_rows": len(asec_raw),
            "full_age15_donor_rows": len(donor_raw),
            "donor_roster_relation": "exact_full_income2024_age15_coverage_native_keys",
            "donor_keys_sha256": source._digest(sorted(expected_donors)),
            "raw_columns": {"acs": hours.ACS_FIELDS, "asec": hours.ASEC_FIELDS},
            "tables_sha256": dict(
                zip(
                    ("origins", "acs_raw", "asec_selected_raw", "donor_raw"),
                    seals,
                    strict=True,
                )
            ),
            "proposals_sha256": proposal_seal,
            "all_person_engine_input_qualified": False,
            "asec_own_arm_hours_qualified": False,
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    result = QualifiedAcsHoursProposals(
        state.frame, origins, acs_raw, asec_raw, donor_raw, proposals, receipt
    )

    def revalidate(candidate):
        require(
            candidate is result and candidate._revalidate is revalidate,
            "PROJECTION_OBJECT_CHANGED",
        )
        # All file/foreign-owner checks precede the pure trailing seals.
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
            and candidate.acs_raw is acs_raw
            and candidate.asec_selected_raw is asec_raw
            and candidate.donor_raw is donor_raw
            and candidate.proposals is proposals
            and tuple(_table_seal(table) for table in tables) == seals
            and _proposal_seal(proposals) == proposal_seal,
            "PROJECTION_CHANGED",
        )

    object.__setattr__(result, "_revalidate", revalidate)
    result.validate()
    return result


_REVALIDATE_CODE = next(
    c
    for c in qualify_current_survey_hours.__code__.co_consts
    if type(c) is CodeType and c.co_name == "revalidate"
)
_LIVE = _live()
