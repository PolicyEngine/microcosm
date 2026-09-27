"""Original-person sex observations borrowed from genuine survey preparation.

ASEC uses the maintained A_SEX/AXSEX qualifier. ACS uses the native coverage
owner's SEX mapping. Unknown ASEC codes/allocation remain nullable, and ACS
allocation provenance is not inferred. Detached values issue no source or
release authority; consumers retain this object and validate at their I/O fence.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from types import CodeType, FunctionType

import numpy as np
import pandas as pd

from . import current_asec_demographics as asec
from . import current_survey_health_coverage as attachment
from . import current_survey_health_source as original
from . import graph_full_puf_enrichment as physical
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.current-survey-sex-source.v1"
STRING = pd.StringDtype(storage="python", na_value=pd.NA)
RAW_COLUMNS = (
    "survey_sex_source",
    "survey_sex_observation_year",
    "survey_sex_A_SEX",
    "survey_sex_AXSEX",
    "survey_sex_SEX",
    "survey_sex_binding_state",
    "survey_sex_allocation",
)
OUTPUT = "is_female"


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_SEX_" + reason)


def _live():
    functions = []
    for module in (
        sys.modules[__name__],
        asec,
        asec.demographic,
        asec.household,
        asec.source_csv_builtin,
        source.acs_native,
        original,
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
        RAW_COLUMNS,
        OUTPUT,
        repr(STRING),
        _REVALIDATE_CODE,
        tuple(
            tuple(vars(item).items())
            for item in asec.demographic.ASEC_DEMOGRAPHIC_FIELDS
        ),
        tuple(
            (
                name,
                type(getattr(asec.demographic, name)),
                tuple(vars(getattr(asec.demographic, name)).items()),
            )
            for name in ("A_SEX", "AXSEX")
        ),
        asec.demographic.ASEC_DEMOGRAPHIC_SOURCE_COLUMNS,
        tuple(asec.demographic.SEX_BINDING_STATES.items()),
        tuple(asec.demographic.SEX_ALLOCATION_STATES.items()),
    )


def recode(raw):
    """Map qualified binding states; never interpret an unknown as male."""
    require(
        type(raw) is pd.DataFrame
        and tuple(raw) == RAW_COLUMNS
        and raw.index.is_unique
        and raw.index.name == "person_id",
        "RAW_AXIS",
    )
    binding = raw.survey_sex_binding_state
    require(
        binding.dtype == np.dtype("int64") and binding.isin((0, 1, 2)).all(),
        "BINDING_DOMAIN",
    )
    return pd.DataFrame(
        {
            OUTPUT: pd.array(
                [False if v == 1 else True if v == 2 else None for v in binding],
                dtype="boolean",
            )
        },
        index=raw.index.copy(),
    )


def _join(origins, asec_people, acs_people):
    """Pure exact original/native-key join over already-qualified projections."""
    require(
        origins.index.is_unique
        and origins.index.dtype == np.dtype("int64")
        and origins.index.name == "person_id"
        and origins.source.isin(("asec", "acs")).all()
        and origins.native_person_id.dtype == np.dtype("int64"),
        "ORIGIN_AXIS",
    )
    selected = origins.source.eq("asec")
    asec_ids = origins.index[selected]
    require(
        asec_people.index.is_unique
        and set(asec_people.index) == set(asec_ids)
        and asec_people.native_person_id.dtype == np.dtype("int64"),
        "ASEC_ROSTER",
    )
    asec_rows = asec_people.reindex(asec_ids)
    require(
        np.array_equal(
            asec_rows.native_person_id.to_numpy(),
            origins.loc[selected, "native_person_id"].to_numpy(),
        ),
        "ASEC_NATIVE_JOIN",
    )
    require(
        acs_people.person_id.dtype == np.dtype("int64")
        and acs_people.person_id.is_unique
        and set(acs_people.person_id)
        == set(origins.loc[~selected, "native_person_id"]),
        "ACS_ROSTER",
    )
    acs_rows = acs_people.set_index("person_id").reindex(
        origins.loc[~selected, "native_person_id"].to_numpy()
    )
    # Native ACS issues complete printed SEX codes. Compare its supplied mapping
    # as a consistency check, but derive the canonical result from those codes.
    sex = acs_rows.SEX
    require(
        pd.api.types.is_integer_dtype(sex.dtype)
        and not pd.api.types.is_bool_dtype(sex.dtype)
        and sex.isin((1, 2)).all()
        and pd.api.types.is_bool_dtype(acs_rows.is_female.dtype)
        and acs_rows.is_female.notna().all()
        and np.array_equal(acs_rows.is_female.to_numpy(), sex.eq(2).to_numpy()),
        "ACS_SEX_IDENTITY",
    )
    binding = asec_rows.asec_sex_binding_state
    require(
        pd.api.types.is_integer_dtype(binding.dtype)
        and not pd.api.types.is_bool_dtype(binding.dtype)
        and binding.isin((0, 1, 2)).all(),
        "ASEC_BINDING_DOMAIN",
    )
    expected = pd.array(
        [False if v == 1 else True if v == 2 else None for v in binding],
        dtype="boolean",
    )
    require(
        pd.api.types.is_bool_dtype(asec_rows.is_female.dtype)
        and pd.Series(expected).equals(asec_rows.is_female.reset_index(drop=True))
        and np.array_equal(asec_rows.sex_known.to_numpy(), binding.ne(0).to_numpy()),
        "ASEC_SEX_IDENTITY",
    )
    raw = pd.DataFrame(index=origins.index.copy())
    raw[RAW_COLUMNS[0]] = pd.array(origins.source, dtype=STRING)
    raw[RAW_COLUMNS[1]] = origins.source.map({"asec": 2025, "acs": 2024}).to_numpy(
        dtype="int64"
    )
    for name in RAW_COLUMNS[2:5]:
        raw[name] = pd.array([None] * len(raw), dtype="Int64")
    raw.loc[selected, "survey_sex_A_SEX"] = asec_rows.asec_A_SEX.to_numpy()
    raw.loc[selected, "survey_sex_AXSEX"] = asec_rows.asec_AXSEX.to_numpy()
    raw.loc[~selected, "survey_sex_SEX"] = sex.to_numpy()
    raw[RAW_COLUMNS[5]] = np.zeros(len(raw), dtype="int64")
    raw.loc[selected, RAW_COLUMNS[5]] = binding.to_numpy()
    raw.loc[~selected, RAW_COLUMNS[5]] = sex.to_numpy()
    raw[RAW_COLUMNS[6]] = pd.array(["unresolved"] * len(raw), dtype=STRING)
    raw.loc[selected, RAW_COLUMNS[6]] = asec_rows.sex_origin.to_numpy()
    recode(raw)
    return raw


def _capture(preparation, expected_entry=None):
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    require(expected_entry is None or entry is expected_entry, "PREPARATION_CHANGED")
    state = entry[2]
    origins, _keys = original._origins(state.frame, json.loads(entry[1]))
    asec_values = asec.qualify_current_asec_demographics(preparation)
    asec_document = json.loads(asec_values.receipt)
    require(
        asec_document["preparation_sha256"] == source._sha(entry[1])
        and asec_document["income_year"] == 2024
        and asec_document["sex_observation_year"] == 2025,
        "ASEC_PARENT_PERIOD",
    )
    acs_frame = state.source_frames[0]
    source.acs_native.verify_acs_native_coverage(state.native[0], acs_frame)
    acs_entry = source.acs_native._owned(state.native[0])
    require(json.loads(acs_entry.payload)["vintage"] == 2024, "ACS_PERIOD")
    require(
        source._sha(
            asec_values.person.reset_index(drop=True).to_json(orient="table").encode()
        )
        == asec_document["person_projection_sha256"],
        "ASEC_PROJECTION_CHANGED",
    )
    raw = _join(origins, asec_values.person, acs_frame.person)
    evidence = source._encode(
        {
            "protocol": PROTOCOL,
            "preparation_sha256": source._sha(entry[1]),
            "asec_projection_sha256": source._sha(asec_values.receipt),
            "acs_native_sha256": source._sha(acs_entry.payload),
            "asec_observation_year": 2025,
            "acs_observation_year": 2024,
            "income_year": 2024,
            "asec_mapping": "A_SEX 1=male, 2=female only with AXSEX 0=no_change or 4=allocated",
            "acs_mapping": "SEX 1=male, 2=female; native coverage verified",
            "acs_allocation_provenance": "unresolved; no_unallocated_claim",
            "unknown_policy": "preserve nullable boolean; never unknown-to-false",
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    source._pure_final(state)
    require(
        source._ISSUED.get(id(preparation)) is entry
        and preparation.payload == entry[1]
        and source.acs_native._owned(state.native[0]) is acs_entry,
        "FINAL_PARENT",
    )
    return entry, origins, raw, evidence


@dataclass(frozen=True, eq=False)
class QualifiedSurveySex:
    source_frame: object
    origins: pd.DataFrame
    raw: pd.DataFrame
    evidence: bytes
    _revalidate: object = field(default=None, repr=False, compare=False)

    def validate(self):
        retained(self)
        self._revalidate(self)


def seal(value):
    require(type(value) is QualifiedSurveySex, "QUALIFIED_TYPE")
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
        type(value) is QualifiedSurveySex
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


def qualify_current_survey_sex(preparation):
    require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    entry, origins, raw, evidence = _capture(preparation)
    result = QualifiedSurveySex(entry[2].frame, origins, raw, evidence)
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
    for c in qualify_current_survey_sex.__code__.co_consts
    if type(c) is CodeType and c.co_name == "revalidate"
)
_LIVE = _live()
