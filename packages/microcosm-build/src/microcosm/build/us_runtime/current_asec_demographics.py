"""Current survey source projection for ASEC sex and household state.

Reuse the maintained demographic source owner for A_SEX/AXSEX. Add only the
missing GESTFIPS household projection through the original weight owner's
bounded capture/CSV primitives and closed member registry. Returned numerical
values are not source authority; a graph host must re-run this live qualifier
and compare materialized descendants, including after replay.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import asec_demographic_source as demographic
from . import asec_original_household_weights as household
from . import source_csv_builtin
from . import survey_population_preparation as preparation_owner
from .support_provenance import spine_source_id_column, support_channel_column

PROTOCOL = "microcosm.us.current-asec-demographic-projection.v1"
STATE_COLUMNS = ("H_SEQ", "GESTFIPS")
STATE_CONTRACT = {
    "field": "GESTFIPS",
    "concept": "State FIPS code",
    "entity": "household",
    "survey_year": 2025,
    "income_year": 2024,
    "dictionary_url": "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf",
    "dictionary_sha256": "5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f",
    "pdf_page_1based": 8,
    "printed_page": "6A-1",
    "printed_length": 2,
    "printed_position": 44,
    "printed_range": [1, 56],
    "universe": "All Households",
    "canonical_mapping": "numeric_identity",
    "unassigned_jurisdiction_validation": "separate_geography_gate",
    "unknown_policy": "preserve_token_and_unbound_state; never_use_carried_state_as_fallback",
}


def _require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_ASEC_DEMOGRAPHICS_" + reason)


def _state_token(token):
    """Preserve an unknown literal rather than interpreting it as a state."""
    _require(
        type(token) is str and len(token) <= household._MAX_FIELD_CHARS,
        "STATE_TOKEN_BOUND",
    )
    if token == "":
        return None, "missing"
    if re.fullmatch(r"[0-9]{1,2}", token, re.ASCII) is None:
        return None, "malformed"
    code = int(token)
    return (
        (code, "in_printed_range")
        if 1 <= code <= 56
        else (None, "outside_printed_range")
    )


def _read_state_capture(path, pin):
    """Low-level literal reader; a caller-supplied pin grants no authority."""
    reader = source_csv_builtin.capture_csv_reader(csv)
    _require(reader is not None, "CSV_READER_CHANGED")
    result = {}
    with open(path, "rb", buffering=0, opener=household._regular_opener) as raw:
        before = os.fstat(raw.fileno())
        _require(before.st_size == pin.size_bytes, "CAPTURE_SIZE")
        digest = household._DigestReader(raw, pin.size_bytes)
        with io.TextIOWrapper(
            io.BufferedReader(digest), encoding="utf-8-sig", newline=""
        ) as text:
            lines = household._RecordLines(text)
            records = reader(lines, strict=True)
            header = next(records, [])
            _require(
                bool(header)
                and all(header)
                and len(header) == len(set(header))
                and set(STATE_COLUMNS) <= set(header),
                "STATE_HEADER",
            )
            positions = [header.index(c) for c in STATE_COLUMNS]
            while True:
                lines.characters = 0
                try:
                    row = next(records)
                except StopIteration:
                    break
                _require(
                    len(row) == len(header) and len(result) < pin.rows,
                    "STATE_ROW_SHAPE",
                )
                key, token = (row[i] for i in positions)
                _require(
                    re.fullmatch(r"[0-9]{1,5}", key, re.ASCII) is not None
                    and 1 <= int(key) <= 99999,
                    "HOUSEHOLD_KEY",
                )
                native = int(key)
                _require(native not in result, "HOUSEHOLD_DUPLICATE_KEY")
                code, status = _state_token(token)
                result[native] = {
                    "H_SEQ": key,
                    "GESTFIPS": token,
                    "state_code": code,
                    "status": status,
                    "member_row_1based": len(result) + 1,
                }
            _require(len(result) == pin.rows, "STATE_ROW_COUNT")
            _require(
                digest.count == pin.size_bytes
                and digest.digest.hexdigest() == pin.member_sha256,
                "STATE_CAPTURE_CHANGED",
            )
            after = os.fstat(raw.fileno())
            _require(
                all(
                    getattr(before, k) == getattr(after, k)
                    for k in (
                        "st_dev",
                        "st_ino",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                ),
                "STATE_CAPTURE_CHANGED",
            )
    return result


def _load_current_state(member_path, expected_member):
    """Closed registry lookup; expected member is retained native-owner evidence."""
    pins = household._registry()
    selected = [pin for pin in pins if pin.income_year == 2024]
    _require(len(selected) == 1, "CURRENT_STATE_REGISTRY")
    pin = selected[0]
    _require(
        pin.survey_year == 2025
        and pin.canonical_member_id == expected_member["canonical_member_id"]
        and pin.member_sha256 == expected_member["member_sha256"]
        and pin.archive_sha256 == expected_member["archive_sha256"],
        "STATE_NATIVE_MEMBER_BINDING",
    )
    registry = household._encode(household._implementation())
    paths = household._member_path_snapshot({2024: member_path}, pins)
    with tempfile.TemporaryDirectory(prefix="asec-current-state-") as directory:
        capture = Path(directory) / "hhpub25.csv"
        _require(
            household._capture_owner._snapshot(
                paths[2024], capture, size=pin.size_bytes
            )
            == pin.member_sha256,
            "STATE_SOURCE_SHA256",
        )
        rows = _read_state_capture(capture, pin)
    _require(
        household._encode(household._implementation()) == registry,
        "STATE_REGISTRY_CHANGED",
    )
    return rows, {
        "canonical_member_id": pin.canonical_member_id,
        "member_sha256": pin.member_sha256,
        "archive_sha256": pin.archive_sha256,
        "bytes": pin.size_bytes,
        "rows": pin.rows,
    }


@dataclass(frozen=True)
class CurrentAsecDemographicValues:
    """Source-derived selected arrays; no independent authority or launch verdict."""

    person: pd.DataFrame
    household: pd.DataFrame
    receipt: bytes


def qualify_current_asec_demographics(preparation):
    """Reconstruct the real parent sex owner and current household state source.

    The selected current survey is joined after full-source reconstruction.
    Missing/unlabelled sex allocation or state codes remain explicitly unknown;
    no non-2→male or carried-state fallback is permitted.
    """
    _require(
        type(preparation) is preparation_owner.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state = entry[2]
    native = state.native[1]
    native_entry = preparation_owner.asec_native._ISSUED.get(id(native))
    _require(
        native_entry is not None
        and native_entry[0]() is native
        and native.payload == native_entry[1],
        "NATIVE_ISSUANCE",
    )
    parent = native_entry[2].parent
    native_receipt = json.loads(native_entry[1])
    _require(
        native_receipt["source_year"] == native_receipt["income_year"] == 2024
        and native_receipt["survey_year"] == 2025,
        "NATIVE_PERIOD",
    )
    observed = demographic.load_authenticated_asec_demographic_source(
        parent,
        member_paths={
            year: state.root / "asec" / f"pppub{year - 1999}.csv"
            for year in (2022, 2023, 2024)
        },
    )
    sex_receipt = observed.receipt
    _require(
        sex_receipt["source_identity"] == parent.source.identity.decode(),
        "DEMOGRAPHIC_PARENT_IDENTITY",
    )
    fields = native_entry[2].fields.document
    _require(len(fields["members"]) == 1, "STATE_MEMBER_ROSTER")
    states, state_pin = _load_current_state(
        state.root / "asec" / "hhpub25.csv", fields["members"][0]
    )
    source_households = {
        row["household_id"]: row for row in native_receipt["households"]
    }
    households = state.frame.table("household")
    selected_hh = households.loc[
        households[support_channel_column("household")].eq("asec")
    ]
    hrows = []
    for stacked, source_hh in selected_hh[
        ["household_id", spine_source_id_column("household")]
    ].itertuples(index=False, name=None):
        _require(source_hh in source_households, "HOUSEHOLD_NATIVE_JOIN")
        current = source_households[source_hh]
        year, key = current["native_key"]
        _require(year == 2024 and key in states, "HOUSEHOLD_SOURCE_JOIN")
        row = states[key]
        _require(
            row["H_SEQ"] == current["household_fields"]["H_SEQ"],
            "HOUSEHOLD_LITERAL_KEY",
        )
        hrows.append(
            (
                int(stacked),
                int(source_hh),
                key,
                row["GESTFIPS"],
                row["state_code"],
                row["status"],
            )
        )
    htable = pd.DataFrame(
        hrows,
        columns=(
            "household_id",
            "native_household_id",
            "H_SEQ_integer",
            "GESTFIPS",
            "state_fips",
            "state_status",
        ),
    ).set_index("household_id", drop=False)
    htable["state_fips"] = pd.array(htable.state_fips, dtype="Int64")
    htable["state_known"] = htable.state_fips.notna()
    full_ids, years = observed.array("person_id"), observed.array("income_year")
    _require(
        len(set(zip(years, full_ids, strict=True))) == len(full_ids), "SEX_SOURCE_AXIS"
    )
    lookup = pd.MultiIndex.from_arrays((years, full_ids))
    person = state.frame.person
    selected = person.loc[person[support_channel_column("person")].eq("asec")]
    original_ids = selected[spine_source_id_column("person")].to_numpy(dtype=np.int64)
    positions = lookup.get_indexer(
        pd.MultiIndex.from_arrays(
            (np.full(len(selected), 2024, dtype=np.int64), original_ids)
        )
    )
    _require(
        (positions >= 0).all() and len(set(positions)) == len(positions),
        "SEX_NATIVE_JOIN",
    )
    ptable = pd.DataFrame(
        {
            "person_id": selected.person_id.to_numpy(dtype=np.int64),
            "native_person_id": original_ids,
        },
        index=pd.Index(selected.person_id.to_numpy(), name="person_id"),
    )
    for name in (
        "asec_A_SEX",
        "asec_AXSEX",
        "asec_sex_binding_state",
        "asec_sex_allocation_state",
    ):
        ptable[name] = observed.array(name)[positions]
    binding = ptable.asec_sex_binding_state.to_numpy()
    _require(np.isin(binding, (0, 1, 2)).all(), "SEX_BINDING_DOMAIN")
    ptable["is_female"] = pd.array(
        [False if code == 1 else True if code == 2 else None for code in binding],
        dtype="boolean",
    )
    ptable["sex_known"] = binding != 0
    ptable["sex_universe"] = "All Persons"
    ptable["sex_origin"] = np.where(
        ptable.asec_sex_allocation_state.eq(2),
        "census_allocated",
        np.where(
            ptable.asec_sex_allocation_state.eq(1),
            "source_no_change",
            "unresolved_allocation",
        ),
    )
    receipt = preparation_owner._encode(
        {
            "protocol": PROTOCOL,
            "preparation_sha256": preparation_owner._sha(entry[1]),
            "native_population_sha256": preparation_owner._sha(native_entry[1]),
            "demographic_source_content_sha256": observed.content_sha256,
            "demographic_source_receipt": sex_receipt,
            "state_source": state_pin,
            "state_contract": STATE_CONTRACT,
            "selected_person_ids": ptable.person_id.tolist(),
            "selected_household_ids": htable.household_id.tolist(),
            "sex_known_persons": int(ptable.sex_known.sum()),
            "sex_unknown_persons": int((~ptable.sex_known).sum()),
            "state_known_households": int(htable.state_known.sum()),
            "state_unknown_households": int((~htable.state_known).sum()),
            "sex_observation_year": 2025,
            "state_observation_year": 2025,
            "income_year": 2024,
            "state_is_income_year_residence_claim": False,
            "selected_household_state_tokens": htable.loc[
                :,
                [
                    "household_id",
                    "native_household_id",
                    "H_SEQ_integer",
                    "GESTFIPS",
                    "state_status",
                ],
            ].values.tolist(),
            "person_projection_sha256": preparation_owner._sha(
                ptable.reset_index(drop=True).to_json(orient="table").encode()
            ),
            "household_projection_sha256": preparation_owner._sha(
                htable.reset_index(drop=True).to_json(orient="table").encode()
            ),
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    observed.validate()
    preparation_owner._pure_final(state)
    _require(
        preparation_owner._ISSUED.get(id(preparation)) is entry
        and preparation.payload == entry[1]
        and preparation_owner.asec_native._ISSUED.get(id(native)) is native_entry
        and native.payload == native_entry[1]
        and native_entry[2].parent is parent,
        "FINAL_ISSUANCE",
    )
    return CurrentAsecDemographicValues(ptable, htable, receipt)
