"""Qualified observed geography and stable household draw keys for both surveys.

This additive, values-only projection borrows current authenticated sources.
It does not assign a location, mutate a Frame, or issue source or Population
authority. Hosts must requalify the projection before execution and replay.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import pandas as pd

from microcosm.graph.population import dtype_for_token

from . import current_asec_demographics as demographics
from . import survey_population_preparation as source
from .support_provenance import spine_source_id_column, support_channel_column

PROTOCOL = "microcosm.us.current-survey-geography.v1"
COLUMNS = (
    "survey_geography_origin_key",
    "survey_observed_state",
    "survey_observed_puma",
)
MAX_HOUSEHOLDS = 64 * 1024**2 // 128
MAX_RECEIPT_BYTES = 64 * 1024


def _require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_GEOGRAPHY_" + reason)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _origin_key(row):
    channel = row["source"]
    _require(type(channel) is str and channel in {"acs", "asec"}, "SOURCE")
    _require(
        type(row["source_year"]) is int
        and row["source_year"] == 2024
        and type(row["survey_year"]) is int
        and row["survey_year"] == (2024 if channel == "acs" else 2025),
        "SOURCE_PERIOD",
    )
    raw = row["raw_native_id"]
    _require(type(raw) is str and 0 < len(raw) <= 128, "NATIVE_KEY")
    # Preserve the source literal, including ASEC's leading zeroes. Receiving
    # integer ids are coordinates only and never enter the keyed draw stream.
    return json.dumps(
        [channel, row["source_year"], row["survey_year"], raw],
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _project(origins, households, acs, asec):
    """Pure alignment from just-qualified values, with no authority of its own."""
    _require(
        type(origins) is list
        and 0 < len(origins) <= MAX_HOUSEHOLDS
        and len(origins) == len(households),
        "HOUSEHOLD_COUNT",
    )
    _require(
        str(households.household_id.dtype) == "int64"
        and households.household_id.is_unique
        and households.household_id.ge(0).all(),
        "HOUSEHOLD_AXIS",
    )
    channel_column = support_channel_column("household")
    native_column = spine_source_id_column("household")
    _require(set(households[channel_column]) == {"acs", "asec"}, "CHANNEL_ROSTER")
    _require(
        {"household_id", "SERIALNO", "ST", "PUMA", "puma_geoid", "puma"} <= set(acs)
        and acs.household_id.is_unique,
        "ACS_SOURCE_COLUMNS",
    )
    _require(
        {
            "household_id",
            "native_household_id",
            "H_SEQ_integer",
            "GESTFIPS",
            "state_fips",
            "state_status",
            "state_known",
        }
        <= set(asec)
        and asec.household_id.is_unique,
        "ASEC_SOURCE_COLUMNS",
    )
    acs_by_id = acs.set_index("household_id", drop=False)
    asec_by_id = asec.set_index("household_id", drop=False)
    origin_by_id, keys = {}, set()
    for row in origins:
        _require(type(row) is dict, "ORIGIN_RECORD")
        household_id = row["household_id"]
        _require(
            type(household_id) is int
            and household_id >= 0
            and household_id not in origin_by_id
            and type(row["selected_receiving_household_id"]) is int
            and row["selected_receiving_household_id"] >= 0,
            "ORIGIN_COORDINATE",
        )
        key = _origin_key(row)
        _require(key not in keys, "ORIGIN_DUPLICATE")
        keys.add(key)
        origin_by_id[household_id] = (row, key)
    _require(set(origin_by_id) == set(households.household_id), "ORIGIN_ROSTER")
    rows, used_acs, used_asec = [], set(), set()
    for household_id, channel, native_id in households[
        ["household_id", channel_column, native_column]
    ].itertuples(index=False, name=None):
        row, key = origin_by_id[household_id]
        _require(
            channel == row["source"]
            and native_id == row["selected_receiving_household_id"],
            "RECEIVING_ORIGIN_JOIN",
        )
        puma = None
        if channel == "acs":
            _require(native_id in acs_by_id.index, "ACS_NATIVE_JOIN")
            original = acs_by_id.loc[native_id]
            _require(original.SERIALNO == row["raw_native_id"], "ACS_LITERAL_KEY")
            state, local_puma = original.ST, original.PUMA
            _require(
                type(state) is str
                and re.fullmatch(r"[0-9]{2}", state, re.ASCII) is not None
                and int(state) > 0
                and type(local_puma) is str
                and re.fullmatch(r"[0-9]{5}", local_puma, re.ASCII) is not None
                and int(local_puma) > 0,
                "ACS_LITERAL_GEOGRAPHY",
            )
            puma = state + local_puma
            _require(
                original.puma_geoid == puma and original.puma == puma,
                "ACS_PUMA_IDENTITY",
            )
            used_acs.add(native_id)
        else:
            _require(household_id in asec_by_id.index, "ASEC_RECEIVING_JOIN")
            original = asec_by_id.loc[household_id]
            raw = row["raw_native_id"]
            _require(
                re.fullmatch(r"[0-9]+", raw, re.ASCII) is not None
                and int(raw) == original.H_SEQ_integer
                and native_id == original.native_household_id,
                "ASEC_LITERAL_KEY",
            )
            code, status = demographics._state_token(original.GESTFIPS)
            _require(
                original.state_status == status
                and bool(original.state_known) == (code is not None)
                and (
                    pd.isna(original.state_fips)
                    if code is None
                    else original.state_fips == code
                ),
                "ASEC_STATE_IDENTITY",
            )
            state = None if code is None else str(code).zfill(2)
            used_asec.add(household_id)
        rows.append((int(household_id), key, state, puma))
    _require(
        used_acs == set(acs.household_id) and used_asec == set(asec.household_id),
        "COMPLETE_NATIVE_ROSTER",
    )
    return tuple(rows)


def _projection_digest(household):
    """Stream bounded row encodings; the receipt never contains source keys."""
    _require(
        tuple(household.columns) == COLUMNS
        and household.index.name == "household_id"
        and str(household.index.dtype) == "int64"
        and household.index.is_unique
        and 0 < len(household) <= MAX_HOUSEHOLDS
        and all(household[c].dtype == dtype_for_token("string") for c in COLUMNS),
        "PROJECTION_STORAGE",
    )
    digest = hashlib.sha256()
    for row in household.itertuples(index=True, name=None):
        values = [int(row[0]), *(None if pd.isna(v) else v for v in row[1:])]
        encoded = source._encode(values, maximum=4096)
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _check_asec_projection(values, receipt):
    """Bind the detached household values to their producer's exact receipt."""
    _require(type(receipt) is bytes and values.receipt == receipt, "ASEC_RECEIPT")
    document = json.loads(receipt)
    _require(
        _sha(values.household.reset_index(drop=True).to_json(orient="table").encode())
        == document.get("household_projection_sha256"),
        "ASEC_PROJECTION_DIGEST",
    )


@dataclass(frozen=True)
class CurrentSurveyGeographyValues:
    """Detached values; retained output bytes grant no downstream admission."""

    household: pd.DataFrame
    receipt: bytes


def qualify_current_survey_geography(preparation):
    """Read current source geography, aligned to selected receiving households."""
    _require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state = entry[2]
    origins = json.loads(entry[1])["origins"]["households"]
    acs_native, acs_frame = state.native[0], state.source_frames[0]
    source.acs_native.verify_acs_native_coverage(acs_native, acs_frame)
    acs_owned = source.acs_native._owned(acs_native)
    acs_payload = acs_owned.payload
    _require(json.loads(acs_payload)["vintage"] == 2024, "ACS_NATIVE_PERIOD")
    asec = demographics.qualify_current_asec_demographics(preparation)
    asec_receipt = asec.receipt
    _check_asec_projection(asec, asec_receipt)
    projected = _project(
        origins,
        state.frame.table("household"),
        acs_frame.table("household"),
        asec.household,
    )
    table = pd.DataFrame(
        {
            column: pd.array(
                [row[i + 1] for row in projected], dtype=dtype_for_token("string")
            )
            for i, column in enumerate(COLUMNS)
        },
        index=pd.Index(
            [row[0] for row in projected], name="household_id", dtype="int64"
        ),
    )
    projection_sha256 = _projection_digest(table)
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "preparation_sha256": _sha(entry[1]),
            "acs_native_sha256": _sha(acs_payload),
            "asec_demographic_sha256": _sha(asec_receipt),
            "projection_sha256": projection_sha256,
            "columns": list(COLUMNS),
            "households": len(table),
            "acs_households": len(acs_frame.table("household")),
            "asec_households": len(asec.household),
            "state_unknown_households": int(table[COLUMNS[1]].isna().sum()),
            "puma_observed_households": int(table[COLUMNS[2]].notna().sum()),
            "draw_identity": ["source", "source_year", "survey_year", "raw_native_id"],
            "draw_key_encoding": "compact ASCII JSON array; source literal unchanged",
            "acs_survey_year": 2024,
            "asec_survey_year": 2025,
            "asec_income_year": 2024,
            "state_is_income_year_residence_claim": False,
            "observed_state": "ACS ST; current ASEC GESTFIPS; two-digit strings",
            "observed_puma": "ACS ST+PUMA; ASEC remains unknown",
            "unknown_state": "preserved; no carried-state fallback",
            "unassigned_jurisdiction_validation": "separate geography gate",
            "source_admission_issued": False,
            "population_admission_issued": False,
            "release_eligible": False,
        },
        maximum=MAX_RECEIPT_BYTES,
    )
    result = CurrentSurveyGeographyValues(table, receipt)
    # Complete owner I/O before pure comparisons. A callback during the last
    # source check must not mutate a retained input or detached output unnoticed.
    source.acs_native.verify_acs_native_coverage(acs_native, acs_frame)
    final_entry = preparation._checked()
    _require(
        final_entry is entry
        and source._ISSUED.get(id(preparation)) is entry
        and preparation.payload == entry[1]
        and source.acs_native._ISSUED.get(acs_native) is acs_owned
        and acs_native.payload == acs_payload
        and acs_owned.frame is acs_frame
        and asec.receipt == asec_receipt,
        "FINAL_ISSUANCE",
    )
    source._pure_final(state)
    _check_asec_projection(asec, asec_receipt)
    _require(
        _project(
            origins,
            state.frame.table("household"),
            acs_frame.table("household"),
            asec.household,
        )
        == projected
        and result.receipt == receipt
        and _projection_digest(result.household) == projection_sha256,
        "FINAL_PROJECTION",
    )
    return result
