"""Checked full-original-ASEC donor projection; no immigration status assignment.

Literal evidence and exact HSUP_WGT anchors are retained through their existing
owners. Selection does not define this donor universe. No receipt or detached
Frame can issue the retained view, and no prior-wage leaf enters its projection.
"""

from __future__ import annotations

import hashlib
import re
import sys
import weakref
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import FunctionType, MappingProxyType
from typing import NamedTuple

import numpy as np
import pandas as pd

from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL
from microcosm.frame import US_SCHEMA, Frame, WeightKind

from . import current_asec_demographics as demographics
from . import current_survey_immigration_source as literals
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.full-asec-immigration-donor.v1"
DICTIONARY = "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf"
COUNTRY_DICTIONARY = (
    "https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf"
)
# 2025 technical documentation, Appendix J (the person dictionary's Appendix H
# cross-reference is stale). Header-only negative response codes are unresolved
# for this rule-ready projection, not defaulted to any nationality.
_COUNTRIES = (
    57,
    60,
    66,
    69,
    73,
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
    155,
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
    168,
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
    220,
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
    421,
    423,
    425,
    427,
    429,
    430,
    436,
    440,
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
    501,
    508,
    511,
    512,
    515,
    523,
    527,
    555,
)
_DOMAINS = MappingProxyType(
    {
        "PRCITSHP": tuple(range(1, 6)),
        "PEINUSYR": tuple(range(29)),
        "A_AGE": (*range(81), 85),
        "A_MARITL": tuple(range(1, 8)),
        "A_SPOUSE": tuple(range(17)),
        "A_HSCOL": (0, 1, 2),
        "A_LFSR": (0, 1, 2, 3, 4, 7),
        **{
            name: (0, 1, 2)
            for name in ("MCARE", "CAID", "IHSFLG", "CHAMPVA", "MIL", "SS_YN", "SSI_YN")
        },
        **{
            name: tuple(range(9))
            for name in ("PEN_SC1", "PEN_SC2", "RESNSS1", "RESNSS2")
        },
        "PEIO1COW": tuple(range(9)),
        "A_MJOCC": tuple(range(12)),
        "PEAFEVER": (-1, 1, 2),
    }
)
_ISSUED = {}


def _require(condition, reason):
    if not condition:
        raise ValueError("FULL_ASEC_IMMIGRATION_DONOR_" + reason)


def _modules():
    return (sys.modules[__name__], literals, demographics, demographics.demographic)


def _live():
    functions = {}
    for module in _modules():
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
        _COUNTRIES,
        tuple(_DOMAINS.items()),
        literals.ASEC_COLUMNS,
    )


def _implementation():
    return source._encode(
        {
            module.__name__: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in _modules()
        }
    )


def _numeric_literals(raw):
    """Admit printed source codes, preserving raw blanks/NIU in the raw owner."""
    _require(tuple(raw.columns) == literals.ASEC_COLUMNS, "LITERAL_READSET")
    result = pd.DataFrame(index=raw.index.copy())
    for name in literals.ASEC_VALUE_COLUMNS:
        values = []
        for token in raw[name]:
            _require(
                type(token) is str and bool(token.strip()), "MISSING_TOKEN_" + name
            )
            text = token.strip()
            if name == "SPM_CAPHOUSESUB":
                _require(
                    re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text, re.ASCII) is not None,
                    "MALFORMED_TOKEN_" + name,
                )
                number = Decimal(text)
                _require(
                    number == number.to_integral_value() and 0 <= number <= 99999,
                    "UNRESOLVED_DOMAIN_" + name,
                )
                value = int(number)
            else:
                _require(
                    re.fullmatch(r"-?[0-9]+", text, re.ASCII) is not None,
                    "MALFORMED_TOKEN_" + name,
                )
                value = int(text)
                domain = _COUNTRIES if name == "PENATVTY" else _DOMAINS[name]
                _require(value in domain, "UNRESOLVED_DOMAIN_" + name)
            values.append(value)
        result[name] = np.array(values, dtype=np.int64)
    return result


def _check_demographic_coordinates(observed, positions, numeric, raw):
    """Confirm every independent original coordinate at the person-id join."""
    _require(
        np.array_equal(observed.array("A_AGE")[positions], numeric.A_AGE.to_numpy())
        and np.array_equal(
            observed.array("A_LINENO")[positions], raw.A_LINENO.map(int).to_numpy()
        )
        and np.array_equal(
            observed.array("source_household_id")[positions],
            raw.PH_SEQ.map(int).to_numpy(),
        ),
        "SEX_SOURCE_COORDINATE_DISAGREE",
    )


def _state_column(household_projection, households, state_codes):
    _require(
        np.array_equal(household_projection.household_id, households.household_id)
        and len(state_codes) == len(households),
        "STATE_HOUSEHOLD_ALIGNMENT",
    )
    return np.array(state_codes, dtype=np.int64)


class _State(NamedTuple):
    preparation: object
    preparation_entry: tuple
    native: object
    native_entry: tuple
    qualified: object
    qualified_seal: str
    observed: object
    observed_header: bytes
    observed_body: bytes
    frame: Frame
    frame_seal: str
    raw: pd.DataFrame
    raw_seal: str
    households: pd.DataFrame
    household_seal: str
    implementation: bytes


def _final(value, entry):
    state = entry[2]
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    source._pure_final(state.preparation_entry[2])
    native_state = state.native_entry[2]
    _require(
        source._ISSUED.get(id(state.preparation)) is state.preparation_entry
        and state.preparation.payload == state.preparation_entry[1]
        and source.asec_native._ISSUED.get(id(state.native)) is state.native_entry
        and state.native.payload == state.native_entry[1]
        and source.asec_native._frame_identity(native_state.parent.frame)
        == native_state.parent_identity
        and source.asec_native._attached_evidence(
            native_state.parent,
            native_state.coverage,
            native_state.anchors,
            native_state.fields,
        )
        == native_state.attached_evidence,
        "RETAINED_SOURCE_CHANGED",
    )
    _require(
        state.observed._header == state.observed_header
        and state.observed._body == state.observed_body
        and literals._table_seal(state.qualified.asec_full_raw) == state.qualified_seal,
        "SOURCE_PROJECTION_CHANGED",
    )
    _require(
        _ISSUED.get(id(value)) is entry
        and entry[0]() is value
        and value.receipt == entry[1]
        and value.frame is state.frame
        and value.raw is state.raw
        and value.households is state.households
        and source.asec_native._frame_identity(state.frame) == state.frame_seal
        and literals._table_seal(state.raw) == state.raw_seal
        and literals._table_seal(state.households) == state.household_seal,
        "DONOR_PROJECTION_CHANGED",
    )


@dataclass(frozen=True, eq=False)
class CurrentAsecImmigrationDonor:
    """Retained view: validate again after the consumer's final relevant I/O."""

    frame: Frame
    raw: pd.DataFrame
    households: pd.DataFrame
    receipt: bytes

    def validate(self):
        entry = _ISSUED.get(id(self))
        _require(
            type(self) is CurrentAsecImmigrationDonor
            and entry is not None
            and entry[0]() is self,
            "ISSUED_OWNER_REQUIRED",
        )
        _final(self, entry)
        state = entry[2]
        state.qualified.validate()
        state.observed.validate()
        _require(state.native._checked() is state.native_entry, "NATIVE_CHANGED")
        _require(
            state.preparation._checked() is state.preparation_entry,
            "PREPARATION_CHANGED",
        )
        _require(
            _implementation() == state.implementation, "IMPLEMENTATION_BYTES_CHANGED"
        )
        _final(self, entry)


def borrow_full_asec_immigration_donor(preparation):
    """Internally qualify full original literals and predictors, never selected mass."""
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    _require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    implementation = _implementation()
    entry = preparation._checked()
    native = entry[2].native[1]
    native_entry = native._checked()
    retained = native_entry[2]
    # Existing owner logic authenticates full coverage, interview universe and
    # exact HSUP_WGT fractions. The selected public frame is not used as donor.
    mask, weights, household_rows, roster = source.asec_native._roster(
        retained.parent, retained.coverage, retained.anchors, retained.fields, None
    )
    full = source.asec_native._descendant(retained.parent, mask, weights)
    normalized = source._normalized_source_copy(full)
    qualified = literals.qualify_current_survey_immigration(preparation)
    raw = qualified.asec_full_raw.copy(deep=True)
    person_ids = normalized.person.person_id
    _require(
        raw.index.is_unique and set(raw.index) == set(person_ids),
        "FULL_PERSON_BIJECTION",
    )
    raw = raw.reindex(pd.Index(person_ids.to_numpy(), name="source_person_id"))
    coverage = pd.DataFrame(roster).set_index("person_id")
    for pid, row in raw.iterrows():
        key = literals.original._key(row, "asec")
        expected = coverage.loc[pid]
        _require(
            key
            == (
                int(expected.source_household_id),
                expected.PERIDNUM,
                int(expected.A_LINENO),
            ),
            "SOURCE_KEY_BIJECTION",
        )
    numeric = _numeric_literals(raw)
    _require(
        np.array_equal(numeric.A_AGE, normalized.person.A_AGE)
        and np.array_equal(numeric.A_AGE, normalized.person.age),
        "OBSERVED_AGE_CHANGED",
    )
    root = entry[2].root / "asec"
    observed = demographics.demographic.load_authenticated_asec_demographic_source(
        retained.parent,
        member_paths={
            year: root / f"pppub{year - 1999}.csv" for year in (2022, 2023, 2024)
        },
    )
    _require(
        observed.receipt["source_identity"] == retained.parent.source.identity.decode(),
        "SEX_PARENT_BINDING",
    )
    ids, years = observed.array("person_id"), observed.array("income_year")
    lookup = pd.MultiIndex.from_arrays((years, ids))
    _require(lookup.is_unique, "SEX_SOURCE_BIJECTION")
    positions = lookup.get_indexer(
        pd.MultiIndex.from_arrays(
            (np.full(len(person_ids), 2024), person_ids.to_numpy())
        )
    )
    _require(
        (positions >= 0).all() and len(set(positions)) == len(positions),
        "SEX_FULL_ROSTER",
    )
    _check_demographic_coordinates(observed, positions, numeric, raw)
    sex = observed.array("asec_sex_binding_state")[positions]
    _require(np.isin(sex, (1, 2)).all(), "SEX_UNRESOLVED")
    states, state_pin = demographics._load_current_state(
        root / "hhpub25.csv", retained.fields.document["members"][0]
    )
    household_projection = pd.DataFrame(
        [
            {
                "household_id": row["household_id"],
                "income_year": row["native_key"][0],
                "source_household_id": row["native_key"][1],
                "origin_key": row["origin_key"],
                "HSUP_WGT": row["HSUP_WGT"],
                "numerator": row["numerator"],
                "denominator": row["denominator"],
                "fraction_numerator": row["fraction"][0],
                "fraction_denominator": row["fraction"][1],
            }
            for row in household_rows
        ]
    )
    state_codes = []
    for row in household_rows:
        year, native_id = row["native_key"]
        _require(
            year == 2024
            and native_id in states
            and states[native_id]["H_SEQ"] == row["household_fields"]["H_SEQ"],
            "STATE_SOURCE_BIJECTION",
        )
        code = states[native_id]["state_code"]
        _require(code in US_STATE_NUMERIC_FIPS_TO_POSTAL, "STATE_UNRESOLVED")
        state_codes.append(code)
    # Project rather than restore the old parent input surface.
    person_columns = (
        US_SCHEMA.person_id_column,
        *(US_SCHEMA.membership_column(e) for e in US_SCHEMA.group_entities),
    )
    person = normalized.person.loc[:, person_columns].copy(deep=True)
    for name in literals.ASEC_VALUE_COLUMNS:
        person[name] = numeric[name].to_numpy(copy=True)
    for name in literals.original.ASEC_KEYS:
        person[name] = raw[name].to_numpy(copy=True)
    person["age"] = normalized.person.age.copy(deep=True)
    person["is_female"] = sex == 2
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = (
            normalized.table(entity)
            .loc[:, [US_SCHEMA.id_column(entity)]]
            .copy(deep=True)
        )
    tables["household"]["state_fips"] = _state_column(
        household_projection, tables["household"], state_codes
    )
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": normalized.weights_for("household")},
        normalized.strata.copy(deep=True),
        metadata=normalized.metadata,
        mass_log=normalized.mass_log,
    )
    _require(
        frame.weights_for("household").kind is WeightKind.DESIGN,
        "DESIGN_WEIGHT_AUTHORITY",
    )
    raw_seal, household_seal = (
        literals._table_seal(raw),
        literals._table_seal(household_projection),
    )
    frame_seal = source.asec_native._frame_identity(frame)
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "preparation_sha256": source._sha(entry[1]),
            "asec_native_sha256": source._sha(native_entry[1]),
            "literal_receipt_sha256": source._sha(qualified.receipt),
            "demographic_source_sha256": observed.content_sha256,
            "state_source": state_pin,
            "income_year": 2024,
            "observation_year": 2025,
            "age_convention": source.observed_age.AGE_CONVENTION,
            "source_dictionary": DICTIONARY,
            "country_dictionary": COUNTRY_DICTIONARY,
            "persons": len(person),
            "households": frame.n("household"),
            "weight_kind": "design",
            "weight_source": "original_HSUP_WGT/100",
            "person_weight_authority": "none",
            "readset": literals.ASEC_VALUE_COLUMNS,
            "frame_sha256": frame_seal,
            "raw_sha256": raw_seal,
            "household_projection_sha256": household_seal,
            "status_assignment_performed": False,
            "national_stock_alignment_qualified": False,
            "prior_income_columns_consumed": False,
            "source_admission_issued": False,
        }
    )
    result = CurrentAsecImmigrationDonor(frame, raw, household_projection, receipt)
    state = _State(
        preparation,
        entry,
        native,
        native_entry,
        qualified,
        literals._table_seal(qualified.asec_full_raw),
        observed,
        observed._header,
        observed._body,
        frame,
        frame_seal,
        raw,
        raw_seal,
        household_projection,
        household_seal,
        implementation,
    )
    owner_id = id(result)
    _ISSUED[owner_id] = (
        weakref.ref(result, lambda _: _ISSUED.pop(owner_id, None)),
        receipt,
        state,
    )
    result.validate()
    return result


_LIVE = _live()
