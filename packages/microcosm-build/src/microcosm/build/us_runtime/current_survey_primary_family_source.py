"""Detached primary-family evidence borrowed from the existing survey owners.

This source-only projection does not issue source/population admission and cannot
authorize graph attachment. A future consumer must requalify against the live
preparation after relevant I/O, as with the existing householder projection.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import FunctionType

import pandas as pd

from . import current_survey_household_roles as roles
from . import current_survey_housing as reader
from . import current_survey_primary_family as classifier

source = roles.source
require = classifier._require
PROTOCOL = "microcosm.us.native-primary-family-source.v1"
REFERENCES = (
    "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.txt",
    "https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf",
    "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf",
    "https://www.census.gov/programs-surveys/cps/technical-documentation/subject-definitions.html",
)


def _live():
    functions = []
    for module in (sys.modules[__name__], classifier):
        for name, item in vars(module).items():
            if type(item) is FunctionType:
                functions.append((module.__name__, name, source._function_seal(item)))
            elif isinstance(item, type) and item.__module__ == module.__name__:
                functions.append((module.__name__, name, item))
    return (
        tuple(functions),
        PROTOCOL,
        REFERENCES,
        classifier.PROTOCOL,
        tuple(
            (name, value) for name, value in vars(classifier).items() if name.isupper()
        ),
        reader.live(),
        roles._live(),
    )


def _projection(table):
    return source._encode(
        {
            "columns": list(table.columns),
            "household_ids": list(table.index),
            "rows": table.astype(object).where(table.notna(), None).values.tolist(),
        }
    )


@dataclass(frozen=True)
class QualifiedCurrentSurveyPrimaryFamily:
    source_frame: object
    origins: pd.DataFrame
    household: pd.DataFrame
    projection: bytes
    receipt: bytes


def primary_family_projection_seal(value):
    require(type(value) is QualifiedCurrentSurveyPrimaryFamily, "QUALIFIED_TYPE")
    return (
        source._frame_identity(value.source_frame),
        roles._table_stamp(value.origins),
        roles._table_stamp(value.household),
        value.projection,
        value.receipt,
    )


def _classify(frame, origins, person_origins, keys, selected, householder):
    require(
        householder.source_frame is frame
        and householder.rows.index.equals(person_origins.index),
        "HOUSEHOLDER_SOURCE_AXIS",
    )
    people = frame.person.set_index("person_id")
    require(people.index.equals(person_origins.index), "PERSON_SOURCE_AXIS")
    grouped = {hid: [] for hid in origins.index}
    heads = {hid: [] for hid in origins.index}
    for (survey, key), pid in keys.items():
        hid = people.at[pid, "person_household_id"]
        require(hid in grouped and origins.at[hid, "source"] == survey, "SOURCE_JOIN")
        raw = selected[survey + "_person"][key]
        require(reader.source_reader._key(raw, survey) == key, "ORIGINAL_KEY")
        grouped[hid].append(raw)
        known = householder.rows.at[pid, roles.VALUE_KNOWN_COLUMN]
        if known and householder.rows.at[pid, roles.VALUE_COLUMN]:
            heads[hid].append(int(raw["A_LINENO" if survey == "asec" else "SPORDER"]))
    values = []
    for hid, origin in origins.iterrows():
        survey = origin.source
        raw_key = (
            int(origin.raw_native_id) if survey == "asec" else origin.raw_native_id
        )
        raw = selected[survey + "_household"][raw_key]
        values.append(
            classifier.classify_primary_household(
                survey,
                raw,
                grouped[hid],
                qualified_head_line=heads[hid][0] if len(heads[hid]) == 1 else None,
            )
        )
    table = pd.DataFrame(index=origins.index.copy())
    for name in classifier.METRICS:
        table[name] = pd.array([row[name] for row in values], dtype="Int64")
    for name in classifier.STATUSES:
        table[name] = pd.array(
            [row[name] for row in values], dtype=pd.StringDtype(storage="python")
        )
    return table


def qualify_current_survey_primary_family(preparation):
    """Use genuine current-ASEC/ACS members and the existing householder owner."""
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    live = _live()
    require(live == _LIVE, "IMPLEMENTATION_CHANGED")
    entry = preparation._checked()
    require(_live() == live, "IMPLEMENTATION_CHANGED")
    state = entry[2]
    frame, origins, person_origins, keys, selected, evidence = reader._original_columns(
        preparation, readsets=classifier.READSETS
    )
    require(_live() == live and frame is state.frame, "ORIGINAL_OWNER_CHANGED")
    # Freeze literal records before another issuer callback; never retain a
    # caller-authored receipt as authority for the classification.
    literals = source._encode(
        {
            name: [
                [list(key) if type(key) is tuple else key, row]
                for key, row in rows.items()
            ]
            for name, rows in selected.items()
        }
    )
    axes = (
        roles._table_stamp(origins),
        roles._table_stamp(person_origins),
        tuple(keys.items()),
    )
    householder = roles.qualify_current_survey_household_roles(preparation)
    require(_live() == live, "IMPLEMENTATION_CHANGED")
    householder_seal = roles.household_role_projection_seal(householder)
    table = _classify(frame, origins, person_origins, keys, selected, householder)
    projection = _projection(table)
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "classification_protocol": classifier.PROTOCOL,
            **evidence,
            "householder_receipt_sha256": roles._sha(householder.receipt),
            "readsets": classifier.READSETS,
            "references": REFERENCES,
            "projection_sha256": roles._sha(projection),
            "source_households": len(table),
            "source_periods": [
                {
                    "survey": survey,
                    "source_year": int(year),
                    "survey_year": int(interview),
                }
                for survey, year, interview in sorted(
                    set(
                        origins[["source", "source_year", "survey_year"]].itertuples(
                            index=False, name=None
                        )
                    )
                )
            ],
            "status_counts": {
                name: table[name].value_counts().to_dict()
                for name in classifier.STATUSES
            },
            "own_child_definition": "never_married_biological_adopted_or_step_child_of_householder_under_18",
            "asec_universe_fields": ["H_HHTYPE", "H_LIVQRT", "HRHTYPE"],
            "asec_resident_spouse_pairs": "reciprocal_A_SPOUSE_by_PH_SEQ_and_A_LINENO",
            "acs_resident_spouse_pairs": "unavailable_no_secondary_spouse_line_pointer",
            "relationship_allocation_evidence": "published_values_allocation_unresolved",
            "unallocated_observation_claim": False,
            "source_admission_issued": False,
            "population_admission_issued": False,
            "graph_attachment_qualified": False,
            "release_eligible": False,
        }
    )
    result = QualifiedCurrentSurveyPrimaryFamily(
        frame, origins, table, projection, receipt
    )
    result_seal = primary_family_projection_seal(result)
    require(preparation._checked() is entry, "PREPARATION_CHANGED")
    require(_live() == live, "IMPLEMENTATION_CHANGED")
    source._pure_final(state)
    require(
        source._ISSUED.get(id(preparation)) is entry
        and preparation.payload == entry[1]
        and (
            roles._table_stamp(origins),
            roles._table_stamp(person_origins),
            tuple(keys.items()),
        )
        == axes
        and roles.household_role_projection_seal(householder) == householder_seal,
        "FINAL_OWNER",
    )
    # Pure reconstruction follows the last owner/source I/O. It must reproduce
    # all nullable metrics and statuses from the retained original literals.
    frozen = {
        name: {tuple(key) if type(key) is list else key: row for key, row in rows}
        for name, rows in json.loads(literals).items()
    }
    rebuilt = _classify(frame, origins, person_origins, keys, frozen, householder)
    require(
        _projection(rebuilt) == projection
        and primary_family_projection_seal(result) == result_seal
        and _live() == live,
        "FINAL_PROJECTION",
    )
    return result


_LIVE = _live()
