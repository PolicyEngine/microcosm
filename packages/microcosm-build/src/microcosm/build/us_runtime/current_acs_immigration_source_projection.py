"""Selected original ACS immigration inputs, without status or stock admission.

The existing preparation, literal reader and native issuer supply all source
authority. This retained view binds only their selected ACS observations and
records public-code ambiguity rather than inferring a country or entry year.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import weakref
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import FunctionType
from typing import NamedTuple

import numpy as np
import pandas as pd

from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL

from . import current_survey_immigration_source as literals
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.current-acs-immigration-source-projection.v1"
DICTIONARY = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.txt"
CODE_LIST = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/code_lists/ACSPUMS2024CodeLists.xls"
# 2024 one-year PUMS POBP public codes, not the ASEC PENATVTY domain.
# The detailed code-list sheet groups South Sudan with Tunisia/Western Sahara
# under 464. The separate public 463 code was removed for confidentiality.
_POBP_CODES = (
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    53,
    54,
    55,
    56,
    60,
    66,
    69,
    72,
    78,
    100,
    102,
    103,
    104,
    105,
    106,
    108,
    109,
    110,
    116,
    117,
    118,
    119,
    120,
    126,
    127,
    128,
    129,
    130,
    132,
    134,
    136,
    137,
    138,
    139,
    140,
    142,
    147,
    148,
    149,
    150,
    151,
    152,
    154,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    200,
    202,
    203,
    205,
    206,
    207,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    222,
    223,
    224,
    226,
    228,
    229,
    231,
    233,
    235,
    236,
    238,
    239,
    240,
    242,
    243,
    245,
    246,
    247,
    248,
    249,
    253,
    254,
    300,
    301,
    303,
    310,
    311,
    312,
    313,
    314,
    315,
    316,
    321,
    323,
    324,
    327,
    328,
    329,
    330,
    332,
    333,
    338,
    339,
    340,
    341,
    343,
    344,
    360,
    361,
    362,
    363,
    364,
    365,
    368,
    369,
    370,
    372,
    373,
    374,
    399,
    400,
    407,
    408,
    412,
    414,
    416,
    417,
    420,
    421,
    423,
    425,
    427,
    429,
    430,
    436,
    440,
    442,
    444,
    447,
    448,
    449,
    451,
    453,
    454,
    457,
    459,
    460,
    461,
    462,
    464,
    467,
    468,
    469,
    501,
    508,
    511,
    512,
    515,
    523,
    527,
    554,
)
_RETAINED = {}


def _require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_ACS_IMMIGRATION_" + reason)


def _live():
    functions = {}
    for module in (sys.modules[__name__], literals):
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                functions[module.__name__, name] = source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                functions[module.__name__, name] = value
                for key, function in vars(value).items():
                    if isinstance(function, FunctionType):
                        functions[module.__name__, name, key] = source._function_seal(
                            function
                        )
    return functions, (
        PROTOCOL,
        DICTIONARY,
        CODE_LIST,
        _POBP_CODES,
        literals.ACS_COLUMNS,
        tuple(US_STATE_NUMERIC_FIPS_TO_POSTAL),
    )


def _implementation():
    return source._encode(
        {
            module.__name__: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (sys.modules[__name__], literals)
        }
    )


def _token(token, domain, field):
    _require(type(token) is str, "TOKEN_TYPE_" + field)
    _require(
        re.fullmatch(r"[0-9]+", token.strip(), re.ASCII) is not None, "TOKEN_" + field
    )
    value = int(token)
    _require(value in domain, "DOMAIN_" + field)
    return value


def _numeric_literals(raw):
    """Pure dictionary qualification; its result has no source authority."""
    _require(
        tuple(raw.columns) == literals.ACS_COLUMNS and raw.index.is_unique, "READSET"
    )
    rows = []
    for row in raw.itertuples(index=False):
        cit = _token(row.CIT, range(1, 6), "CIT")
        birth = _token(row.POBP, _POBP_CODES, "POBP")
        age = _token(row.AGEP, range(100), "AGEP")
        _require(type(row.YOEP) is str, "TOKEN_TYPE_YOEP")
        niu = row.YOEP.strip() == ""
        _require(not niu or cit == 1, "YOEP_REQUIRED")
        _require(cit != 1 or niu, "YOEP_OUTSIDE_UNIVERSE")
        year, lower, upper, precision = None, None, None, "not_in_universe"
        if not niu:
            year = _token(row.YOEP, (1938, 1939, *range(1945, 2025)), "YOEP")
            if year == 1938:
                upper, precision = 1938, "bottom_coded"
            elif year == 1939:
                lower, upper, precision = 1939, 1944, "interval"
            else:
                lower = upper = year
                precision = "calendar_year"
        rows.append(
            (
                cit,
                birth,
                year,
                age,
                niu,
                precision,
                lower,
                upper,
                "Tunisia_Western_Sahara_South_Sudan_group"
                if birth == 464
                else "published_category",
            )
        )
    result = pd.DataFrame(
        rows,
        index=raw.index.copy(),
        columns=(
            "CIT",
            "POBP",
            "YOEP",
            "AGEP",
            "YOEP_is_niu",
            "YOEP_precision",
            "YOEP_lower",
            "YOEP_upper",
            "POBP_precision",
        ),
    )
    for name in ("YOEP", "YOEP_lower", "YOEP_upper"):
        result[name] = pd.array(result[name], dtype="Int64")
    return result


def _coordinate(value):
    _require(
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and value >= 0,
        "COORDINATE",
    )
    return int(value)


def _project(
    raw, origins, household_origins, native_people, native_households, receiving_people
):
    """Pure identity joins; all source authority stays with the calling owners."""
    selected = origins.loc[origins.source.eq("acs")]
    _require(
        raw.index.equals(selected.index) and raw.index.is_unique, "SELECTED_ROSTER"
    )
    _require(
        selected.source_year.eq(2024).all() and selected.survey_year.eq(2024).all(),
        "SOURCE_PERIOD",
    )
    native = {r.person_id: r for r in native_people.itertuples(index=False)}
    houses = {r.household_id: r for r in native_households.itertuples(index=False)}
    receiving = dict(
        receiving_people[["person_id", "person_household_id"]].itertuples(
            index=False, name=None
        )
    )
    household_keys = {r["household_id"]: r for r in household_origins}
    _require(
        len(native) == len(native_people)
        and len(houses) == len(native_households)
        and len(receiving) == len(receiving_people)
        and len(household_keys) == len(household_origins),
        "DUPLICATE_ID",
    )
    _require(
        selected.native_person_id.is_unique
        and set(selected.native_person_id) <= set(native),
        "NATIVE_PERSON_JOIN",
    )
    numeric = _numeric_literals(raw)
    coordinates, household_rows, seen_keys = [], {}, set()
    for pid, origin in zip(
        selected.index, selected.itertuples(index=False), strict=True
    ):
        _coordinate(pid)
        native_id = _coordinate(origin.native_person_id)
        person = native[native_id]
        native_hid = _coordinate(person.person_household_id)
        _require(native_hid in houses and pid in receiving, "MEMBERSHIP")
        house = houses[native_hid]
        receiving_hid = _coordinate(receiving[pid])
        _require(receiving_hid in household_keys, "HOUSEHOLD_ORIGIN")
        house_origin = household_keys[receiving_hid]
        row = raw.loc[pid]
        key = literals.original._key(row, "acs")
        _require(key not in seen_keys, "DUPLICATE_SOURCE_COORDINATE")
        seen_keys.add(key)
        _require(
            origin.raw_native_household_id == key[0] == house.SERIALNO
            and origin.raw_native_person_id == key[1] == str(person.source_person_id)
            and _token(origin.native_line_numeric_original, range(1, 21), "LINE")
            == key[2]
            and person.SPORDER == key[2]
            and person.source_household_id == native_hid
            and person.source_year == 2024,
            "PERSON_SOURCE_COORDINATE",
        )
        _require(
            house_origin["source"] == "acs"
            and house_origin["source_year"] == house_origin["survey_year"] == 2024
            and house_origin["selected_receiving_household_id"] == native_hid
            and house_origin["raw_native_id"] == key[0],
            "HOUSEHOLD_SOURCE_COORDINATE",
        )
        age = numeric.at[pid, "AGEP"]
        _require(person.AGEP == person.A_AGE == person.age == age, "OBSERVED_AGE")
        _require(
            not isinstance(person.SEX, (bool, np.bool_))
            and person.SEX in (1, 2)
            and isinstance(person.is_female, (bool, np.bool_))
            and person.is_female == (person.SEX == 2),
            "OBSERVED_SEX",
        )
        _require(
            type(house.ST) is str
            and re.fullmatch(r"[0-9]{2}", house.ST, re.ASCII) is not None,
            "STATE_TOKEN",
        )
        state = int(house.ST)
        _require(
            state in US_STATE_NUMERIC_FIPS_TO_POSTAL and house.state_fips == state,
            "OBSERVED_STATE",
        )
        coordinates.append(
            (
                receiving_hid,
                native_id,
                native_hid,
                key[0],
                key[1],
                int(person.SEX),
                bool(person.is_female),
                state,
            )
        )
        household_rows[receiving_hid] = (
            receiving_hid,
            native_hid,
            key[0],
            house.ST,
            state,
        )
    output = numeric.copy(deep=True)
    columns = (
        "person_household_id",
        "native_person_id",
        "native_household_id",
        "raw_native_household_id",
        "raw_native_person_id",
        "SEX",
        "is_female",
        "state_fips",
    )
    for pos, name in enumerate(columns):
        # Never pass heterogeneous rows or large coordinates through float.
        output[name] = pd.Series(
            [r[pos] for r in coordinates],
            index=output.index,
            dtype=object if pos < 3 else None,
        )
    output["source"] = "acs"
    output["source_year"] = 2024
    output["observation_year"] = 2024
    households = pd.DataFrame(
        list(household_rows.values()),
        columns=("household_id", "native_household_id", "SERIALNO", "ST", "state_fips"),
    )
    return output, households


class _State(NamedTuple):
    preparation: object
    preparation_entry: tuple
    native: object
    native_owned: object
    qualified: object
    qualified_receipt: bytes
    literal_tables: tuple
    literal_seals: tuple
    tables: tuple
    seals: tuple
    implementation: bytes


def _literal_tables(value):
    return (
        value.origins,
        value.asec_full_raw,
        value.asec_selected_raw,
        value.acs_raw,
        value.person_evidence,
    )


def _final(value, entry):
    state = entry[2]
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    source._pure_final(state.preparation_entry[2])
    _require(
        source._ISSUED.get(id(state.preparation)) is state.preparation_entry
        and state.preparation.payload == state.preparation_entry[1]
        and source.acs_native._owned(state.native) is state.native_owned,
        "RETAINED_SOURCE_CHANGED",
    )
    _require(
        state.qualified.source_frame is state.preparation_entry[2].frame
        and state.qualified.receipt == state.qualified_receipt
        and all(
            a is b
            for a, b in zip(
                _literal_tables(state.qualified), state.literal_tables, strict=True
            )
        )
        and tuple(literals._table_seal(t) for t in state.literal_tables)
        == state.literal_seals,
        "LITERAL_CHANGED",
    )
    _require(
        _RETAINED.get(id(value)) is entry
        and entry[0]() is value
        and value.receipt == entry[1]
        and all(
            a is b
            for a, b in zip(
                (value.raw, value.person, value.households), state.tables, strict=True
            )
        )
        and tuple(literals._table_seal(t) for t in state.tables) == state.seals,
        "PROJECTION_CHANGED",
    )


@dataclass(frozen=True, eq=False)
class CurrentAcsImmigrationProjection:
    """Borrowed source-only view; consumers validate after their final I/O."""

    raw: pd.DataFrame
    person: pd.DataFrame
    households: pd.DataFrame
    receipt: bytes

    def validate(self) -> None:
        entry = _RETAINED.get(id(self))
        _require(
            type(self) is CurrentAcsImmigrationProjection
            and entry is not None
            and entry[0]() is self,
            "RETAINED_VIEW",
        )
        _final(self, entry)
        state = entry[2]
        state.qualified.validate()
        source.acs_native.verify_acs_native_coverage(
            state.native, state.preparation_entry[2].source_frames[0]
        )
        _require(
            state.preparation._checked() is state.preparation_entry,
            "PREPARATION_CHANGED",
        )
        _require(
            _implementation() == state.implementation, "IMPLEMENTATION_BYTES_CHANGED"
        )
        _final(self, entry)


def borrow_current_acs_immigration_projection(
    preparation: source.AuthenticatedSurveyPopulationPreparation,
) -> CurrentAcsImmigrationProjection:
    """Internally qualify selected original ACS evidence, without rule admission.

    An authorized build may read original sources through this opt-in operation.
    Nothing is downloaded; tests exercise these same issuers on invented bytes.
    """
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    _require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    implementation = _implementation()
    entry = preparation._checked()
    state = entry[2]
    native = state.native[0]
    source.acs_native.verify_acs_native_coverage(native, state.source_frames[0])
    owned = source.acs_native._owned(native)
    _require(json.loads(owned.payload)["vintage"] == 2024, "NATIVE_PERIOD")
    qualified = literals.qualify_current_survey_immigration(preparation)
    qualified.validate()
    _require(
        qualified.source_frame is state.frame
        and json.loads(qualified.receipt)["preparation_sha256"]
        == source._sha(entry[1]),
        "LITERAL_OWNER",
    )
    # Independent immutable seals survive edits to the literal owner's closure.
    literal_tables = _literal_tables(qualified)
    literal_seals = tuple(literals._table_seal(t) for t in literal_tables)
    raw = qualified.acs_raw.copy(deep=True)
    person, households = _project(
        raw,
        qualified.origins,
        json.loads(entry[1])["origins"]["households"],
        state.source_frames[0].person,
        state.source_frames[0].table("household"),
        state.frame.person,
    )
    tables = (raw, person, households)
    seals = tuple(literals._table_seal(t) for t in tables)
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "preparation_sha256": source._sha(entry[1]),
            "acs_native_sha256": source._sha(owned.payload),
            "literal_receipt_sha256": source._sha(qualified.receipt),
            "implementation": json.loads(implementation),
            "source_dictionary": DICTIONARY,
            "country_code_list": CODE_LIST,
            "projection_scope": "selected_original_ACS_people",
            "source": "acs",
            "source_year": 2024,
            "observation_year": 2024,
            "person_rows": len(person),
            "household_rows": len(households),
            "tables_sha256": dict(
                zip(("raw", "person", "households"), seals, strict=True)
            ),
            "birthplace_code_system": "ACS_2024_POBP_not_ASEC_PENATVTY",
            "POBP_464": ["Tunisia", "Western Sahara", "South Sudan"],
            "South_Sudan_separately_identifiable": False,
            "YOEP_convention": "latest_entry_to_live_in_US; repeat_entry_reporting_ambiguous; retain_public_coarsening",
            "age_convention": "observed_AGEP; source_topcoding_retained",
            "unallocated_observation_claim": False,
            "status_assignment_performed": False,
            "national_stock_alignment_qualified": False,
            "prior_income_columns_consumed": False,
            "person_weight_authority": "none",
            "source_admission_issued": False,
        }
    )
    result = CurrentAcsImmigrationProjection(raw, person, households, receipt)
    retained = _State(
        preparation,
        entry,
        native,
        owned,
        qualified,
        qualified.receipt,
        literal_tables,
        literal_seals,
        tables,
        seals,
        implementation,
    )
    key = id(result)
    _RETAINED[key] = (
        weakref.ref(result, lambda _: _RETAINED.pop(key, None)),
        receipt,
        retained,
    )
    result.validate()
    return result


_LIVE = _live()
