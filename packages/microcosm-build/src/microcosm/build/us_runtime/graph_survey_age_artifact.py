"""Unweighted age counts as an ordered artifact, without population columns.

The actual NationalAgeCountKernel performs the measurement. Its household
roster is the sorted unique membership of the declared person projection;
Frame requires every household to be referenced. Downstream admission must
also compare the artifact's ordered IDs with the complete receiving population.
This artifact is a measurement, not source authority or target activation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    KernelBase,
    KernelResult,
    source_hash,
)

from . import graph_national_age_counts as ages
from . import national_age_activation as activation

COUNTS_TYPE = ArtifactType("microcosm.us.survey_household_age_counts", 1)
MAGIC = b"MCUSAGE1\n"
MAX_BYTES = 64 * 1024**2
MAX_HEADER_BYTES = 65_536
MAX_PEOPLE = 10_000_000
_COLUMNS = tuple(b.column for b in activation.NATIONAL_AGE_ACTIVATION.bands)
_WIDTH = 1 + len(_COLUMNS)
_FIELDS = frozenset(
    {"protocol", "population", "columns", "rows", "people", "age_convention"}
)


def _require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_AGE_ARTIFACT_" + reason)


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _shape(rows, people):
    _require(type(rows) is int and type(people) is int, "INTEGER_SHAPE")
    _require(0 < rows <= people <= MAX_PEOPLE, "SHAPE")
    _require(
        len(MAGIC) + 4 + MAX_HEADER_BYTES + rows * _WIDTH * 8 <= MAX_BYTES, "LIMIT"
    )


def _validate(header: dict, values: np.ndarray):
    _require(type(header) is dict and set(header) == _FIELDS, "HEADER")
    _require(
        header["protocol"] == "microcosm.us.survey-household-age-counts.v1", "PROTOCOL"
    )
    _require(
        type(header["population"]) is str and 0 < len(header["population"]) <= 256,
        "POPULATION",
    )
    _require(header["columns"] == list(_COLUMNS), "COLUMNS")
    _require(header["age_convention"] == ages.SUPPORTED_AGE_CONVENTION, "CONVENTION")
    _shape(header["rows"], header["people"])
    _require(
        type(values) is np.ndarray
        and values.dtype == np.dtype("<i8")
        and values.shape == (header["rows"], _WIDTH),
        "ARRAY",
    )
    ids, counts = values[:, 0], values[:, 1:]
    _require(np.all(ids[1:] > ids[:-1]), "ORDERED_IDS")
    _require(np.all(counts >= 0) and np.all(counts <= header["people"]), "COUNTS")
    _require(np.all(np.any(counts > 0, axis=1)), "EMPTY_HOUSEHOLD")
    # The shape and per-cell limits above bound this sum well below int64 max.
    _require(int(counts.sum(dtype=np.int64)) == header["people"], "PEOPLE_TOTAL")


@dataclass(frozen=True)
class SurveyAgeCountValues:
    """Decoded values only. Typed ancestry and current-population checks are separate."""

    population: str
    household_ids: np.ndarray
    counts: pd.DataFrame
    people: int


def decode_survey_age_counts(payload: bytes) -> SurveyAgeCountValues:
    _require(
        type(payload) is bytes and len(MAGIC) + 4 < len(payload) <= MAX_BYTES, "PAYLOAD"
    )
    _require(payload.startswith(MAGIC), "MAGIC")
    offset = len(MAGIC)
    length = int.from_bytes(payload[offset : offset + 4], "big")
    _require(
        0 < length <= MAX_HEADER_BYTES and offset + 4 + length <= len(payload),
        "HEADER_LIMIT",
    )
    offset += 4
    raw = payload[offset : offset + length]
    try:
        header = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ValueError("SURVEY_AGE_ARTIFACT_HEADER_JSON") from None
    _require(type(header) is dict and set(header) == _FIELDS, "HEADER")
    _require(_json(header) == raw, "CANONICAL_HEADER")
    _shape(header["rows"], header["people"])
    offset += length
    _require(len(payload) - offset == header["rows"] * _WIDTH * 8, "BYTE_COUNT")
    values = np.frombuffer(payload, dtype="<i8", offset=offset).reshape(
        header["rows"], _WIDTH
    )
    _validate(header, values)
    ids = values[:, 0].copy()
    counts = pd.DataFrame(
        values[:, 1:].copy(),
        columns=_COLUMNS,
        index=pd.Index(ids.copy(), name="household_id"),
    )
    return SurveyAgeCountValues(header["population"], ids, counts, header["people"])


def survey_age_count_artifact_node(*, population, node_id="survey.age_count_matrix"):
    original = ages.national_age_count_node(population=population, node_id=node_id)
    return replace(
        original,
        kernel=SurveyAgeCountArtifactKernel.ref,
        outputs=(),
        artifact_outputs=(ArtifactOutput("counts", COUNTS_TYPE),),
    )


class SurveyAgeCountArtifactKernel(KernelBase):
    ref = "us.survey_age_count_artifact@1"
    capabilities = ages.NationalAgeCountKernel.capabilities

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            ages,
            activation,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        expected = survey_age_count_artifact_node(
            population=context.node.population, node_id=context.node.id
        )
        _require(context.node.normative() == expected.normative(), "DECLARATION")
        _require(dict(context.params) == dict(expected.params), "PARAMETERS")
        _require(
            set(context.tables) == {"person"}
            and not context.sources
            and not context.artifacts,
            "CONTEXT",
        )
        person = context.tables["person"]
        _require(0 < len(person) <= MAX_PEOPLE, "PEOPLE_LIMIT")
        ids = np.unique(
            ages._ids(person, "person_household_id", what="household membership")
        )
        _shape(len(ids), len(person))
        original = ages.national_age_count_node(
            population=context.node.population, node_id=context.node.id
        )
        measured = ages.NationalAgeCountKernel().run(
            replace(
                context,
                node=original,
                params=original.params,
                tables={
                    "person": person,
                    "household": pd.DataFrame({"household_id": ids}),
                },
                weights={},
            )
        )
        header = {
            "protocol": "microcosm.us.survey-household-age-counts.v1",
            "population": context.node.population,
            "columns": list(_COLUMNS),
            "rows": len(ids),
            "people": len(person),
            "age_convention": ages.SUPPORTED_AGE_CONVENTION,
        }
        values = np.empty((len(ids), _WIDTH), dtype="<i8")
        values[:, 0] = ids
        for position, column in enumerate(_COLUMNS, 1):
            result = measured.columns[("household", column)]
            _require(np.array_equal(result.index.to_numpy(), ids), "MEASUREMENT_ORDER")
            values[:, position] = result.to_numpy()
        _validate(header, values)
        raw_header = _json(header)
        _require(len(raw_header) <= MAX_HEADER_BYTES, "HEADER_LIMIT")
        payload = (
            MAGIC
            + len(raw_header).to_bytes(4, "big")
            + raw_header
            + values.tobytes(order="C")
        )
        _require(len(payload) <= MAX_BYTES, "LIMIT")
        return KernelResult(
            artifacts={"counts": payload},
            receipt={
                **measured.receipt,
                "scope": "survey_age_count_artifact_development",
                "counts_sha256": hashlib.sha256(payload).hexdigest(),
                "count_columns_added": False,
                "household_order": "ascending unique person household membership; consumer verifies full roster",
            },
        )
