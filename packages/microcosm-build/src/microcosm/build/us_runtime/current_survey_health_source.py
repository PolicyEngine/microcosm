"""Health-code projection from the retained original ACS and ASEC sources.

The preparation issues source identity. This module only borrows it, captures
its pinned original members and returns descriptive values. A host must retain
that owner and requalify after its last relevant I/O before returning output.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import acs_housing_universe_source as housing
from . import acs_person_coverage_authentication as records
from . import asec_coverage_authentication as asec
from . import current_survey_health_coverage as health
from . import source_csv_builtin
from . import survey_population_preparation as source

require = health.require
ASEC_KEYS = ("PERIDNUM", "PH_SEQ", "A_LINENO")
ACS_KEYS = ("SERIALNO", "SPORDER")
ASEC_COLUMNS = (*ASEC_KEYS, *health.ASEC_VALUE_COLUMNS, *health.ASEC_ALLOCATION_COLUMNS)
ACS_COLUMNS = (
    *ACS_KEYS,
    *health.ACS_VALUE_COLUMNS,
    *health.ACS_ALLOCATION_COLUMNS,
    *health.ACS_EDIT_COLUMNS,
)


def _key(row, survey):
    if survey == "asec":
        require(
            re.fullmatch(r"[0-9]{22}", row["PERIDNUM"], re.ASCII) is not None,
            "ASEC_PERSON_KEY",
        )
        require(
            re.fullmatch(r"[0-9]{1,5}", row["PH_SEQ"], re.ASCII) is not None,
            "ASEC_HOUSEHOLD_KEY",
        )
        require(
            re.fullmatch(r"[0-9]{1,2}", row["A_LINENO"], re.ASCII) is not None
            and 1 <= int(row["A_LINENO"]) <= 20
            and int(row["PH_SEQ"]) > 0,
            "ASEC_LINE_KEY",
        )
        return (int(row["PH_SEQ"]), row["PERIDNUM"], int(row["A_LINENO"]))
    require(
        re.fullmatch(r"2024(?:HU|GQ)[0-9]{7}", row["SERIALNO"], re.ASCII) is not None,
        "ACS_HOUSEHOLD_KEY",
    )
    require(
        re.fullmatch(r"[0-9]{1,2}", row["SPORDER"], re.ASCII) is not None
        and 1 <= int(row["SPORDER"]) <= 20,
        "ACS_LINE_KEY",
    )
    return (row["SERIALNO"], str(int(row["SPORDER"])), int(row["SPORDER"]))


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
    for raw in records._records(stream):
        values = records._decode_record(raw, first=header is None)
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
        key = _key(row, survey)
        require(all(len(row[c]) <= 64 for c in columns), "SOURCE_TOKEN_BOUND")
        if key in wanted:
            require(key not in selected, "DUPLICATE_SELECTED_SOURCE_KEY")
            selected[key] = row
    require(header is not None, "SOURCE_HEADER")
    return count


def _origins(preparation_frame, document):
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
        and original.person_id.is_unique,
        "ORIGIN_ROSTER",
    )
    original = original.set_index("person_id", drop=True)
    people = preparation_frame.person
    require(
        np.array_equal(original.index.to_numpy(), people.person_id.to_numpy())
        and np.array_equal(
            original.source.to_numpy(),
            people[health.provenance.support_channel_column("person")].to_numpy(),
        )
        and np.array_equal(
            original.selected_receiving_person_id.to_numpy(),
            people[health.provenance.spine_source_id_column("person")].to_numpy(),
        ),
        "ORIGIN_FRAME_IDENTITY",
    )
    require(
        original.source.isin(("acs", "asec")).all()
        and original.source_year.eq(2024).all()
        and original.survey_year.eq(
            original.source.map({"acs": 2024, "asec": 2025})
        ).all(),
        "ORIGIN_PERIOD",
    )
    original["native_person_id"] = original.selected_receiving_person_id
    keys = {}
    for pid, row in original.iterrows():
        household = row.raw_native_household_id
        require(
            type(household) is str
            and type(row.raw_native_person_id) is str
            and type(row.native_line_numeric_original) is str,
            "ORIGIN_COORDINATE_TYPE",
        )
        line = row.native_line_numeric_original
        if row.source == "asec":
            key = _key(
                {
                    "PH_SEQ": household,
                    "PERIDNUM": row.raw_native_person_id,
                    "A_LINENO": line,
                },
                "asec",
            )
        else:
            key = _key({"SERIALNO": household, "SPORDER": line}, "acs")
            require(row.raw_native_person_id == key[1], "ORIGIN_PERSON_KEY")
        require((row.source, key) not in keys, "DUPLICATE_ORIGIN")
        keys[row.source, key] = pid
    return original, keys


def _combine(origins, keys, selected):
    require(set(selected) == {"acs", "asec"}, "SOURCE_ARMS")
    require(
        all(set(selected[s]) == {k for arm, k in keys if arm == s} for s in selected),
        "SELECTED_SOURCE_ROSTER",
    )
    raw = origins.loc[:, ["source", "source_year", "survey_year"]].copy()
    for column in health.RAW_COLUMNS:
        raw[column] = pd.Series([None] * len(raw), index=raw.index, dtype=object)
    for (survey, key), pid in keys.items():
        row = selected[survey][key]
        require(_key(row, survey) == key, "SOURCE_COORDINATE_CHANGED")
        for column in health.RAW_COLUMNS:
            if column in row:
                raw.at[pid, column] = row[column]
    return raw


@dataclass(frozen=True)
class QualifiedSurveyHealthCoverage:
    """A detached projection, not a substitute for the retained source owner."""

    source_frame: object
    origins: pd.DataFrame
    raw: pd.DataFrame
    projection: bytes
    evidence: dict


def qualify_current_survey_health(preparation):
    """Re-read pinned originals without native Frame or engine reconstruction.

    This entry point does read original microdata when called by an authorized
    build. Tests can exercise the same path with privately pinned invented bytes.
    """
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state = entry[2]
    document = json.loads(entry[1])
    origins, keys = _origins(state.frame, document)
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
    pin = tuple(p for p in asec._MEMBER_PINS if p[0] == 2024)
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
    with tempfile.TemporaryDirectory(prefix="microcosm-health-source-") as temporary:
        directory = Path(temporary)
        captured = directory / member
        identity = asec._capture(
            state.root / "asec" / member,
            captured,
            size=size,
            digest=digest,
            budget=[asec._BODY_MAX],
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
            and asec._identity(captured.stat(follow_symlinks=False)) == identity
            and housing._persisted_sha(captured, size) == digest,
            "ASEC_CAPTURE_CHANGED",
        )
        captured = directory / acs_name
        require(
            housing._copy(
                state.root / "acs" / acs_name, captured, acs_size, exact_size=acs_size
            )
            == acs_digest,
            "ACS_CAPTURE_DIGEST",
        )
        count = 0
        with zipfile.ZipFile(captured) as archive:
            members, prefix = records._members(archive, "person")
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
            and housing._persisted_sha(captured, acs_size) == acs_digest,
            "ACS_CAPTURE_CHANGED",
        )
    raw = _combine(origins, keys, selected)
    projection = raw.reset_index().to_json(orient="table", index=False).encode()
    evidence = {
        "protocol": health.PROTOCOL,
        "preparation_sha256": hashlib.sha256(entry[1]).hexdigest(),
        "asec_native_sha256": hashlib.sha256(issued[1]).hexdigest(),
        "acs_catalogue_sha256": hashlib.sha256(acs_owned.receipt).hexdigest(),
        "asec_person_member_sha256": digest,
        "acs_person_archive_sha256": acs_digest,
        "asec_coverage_observation_year": 2025,
        "acs_coverage_observation_year": 2024,
        "projection_sha256": hashlib.sha256(projection).hexdigest(),
        "selected_rows": len(raw),
        "source_admission_issued": False,
        "unallocated_observation_claim": False,
        "release_eligible": False,
    }
    # The original issuer checks source bytes after the last capture/cleanup I/O.
    require(preparation._checked() is entry, "PREPARATION_CHANGED")
    source._pure_final(state)
    require(
        source._ISSUED.get(id(preparation)) is entry
        and source.asec_native._ISSUED.get(id(native)) is issued
        and source.acs_catalogue._lookup(state.catalogues[0]) is acs_owned,
        "FINAL_OWNER",
    )
    return QualifiedSurveyHealthCoverage(
        state.frame, origins, raw, projection, evidence
    )
