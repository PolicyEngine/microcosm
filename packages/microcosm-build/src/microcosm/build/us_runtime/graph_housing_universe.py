"""Non-authoritative HU transport bound to CREATE and selected graph ancestry.

Parsing these bytes never issues AuthenticatedHousingUniverse or Frame authority.
The graph supplies the producer and preparation-receipt custody; this module
checks its native bytes and the exact int64 promotion of selected carried cells.
"""

from __future__ import annotations

import struct
from dataclasses import InitVar, dataclass

import numpy as np
import pandas as pd

from microcosm.graph import ArtifactType

from . import asec_housing_universe as hu
from . import asec_housing_universe_source as source

US_ASEC_HOUSING_UNIVERSE_TYPE = ArtifactType(
    "microcosm.us.asec_bound_housing_universe", 1
)
HOUSEHOLD_EVIDENCE_COLUMNS = (
    *("asec_" + name for name in ("H_SEQ",) + hu.CONTEXT_COLUMNS),
    *source.ATTACHED_COLUMNS,
)
_TOKEN = object()
_HEADER_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "source_authentication",
        "release_eligible",
        "source_period_kind",
        "zero_origin_policy",
        "zero_origin_evidence",
        "household_rows",
        "columns",
        "codebook",
        "dictionaries",
        "definition_sha256",
        "source",
        "implementation",
        "limitations",
    }
)


ACCEPTED_PREPARED_RECEIPT_SCHEMAS = ("microcosm.us.asec_prepared_receipt.v3",)
ACCEPTED_PREPARED_SOURCE_KINDS = ("us_asec_prepared_current_money_v3",)


@dataclass(frozen=True)
class BoundHousingUniverse:
    """Immutable native buffers with graph bindings, no full-source authority."""

    header: bytes
    buffers: tuple[bytes, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        hu._require(_token is _TOKEN, "BOUND_HU_CONSTRUCTOR")

    def array(self, name: str) -> np.ndarray:
        hu._require(name in hu.COLUMNS, "BOUND_HU_COLUMN")
        return np.frombuffer(
            self.buffers[hu.COLUMNS.index(name)],
            dtype="<i8" if name in hu.RAW_COLUMNS else "u1",
        )


def bind_housing_universe(
    payload: bytes, *, prepared_receipt: dict
) -> BoundHousingUniverse:
    """Validate transport against a graph-provided receipt, not a source decoder."""
    try:
        hu._require(
            type(payload) is bytes
            and len(hu.MAGIC) + 4 + 32 < len(payload) <= hu.PAYLOAD_MAX_BYTES,
            "BOUND_HU_SIZE",
        )
        binding = prepared_receipt["housing_universe"]
        hu._require(
            prepared_receipt["schema"] in ACCEPTED_PREPARED_RECEIPT_SCHEMAS
            and prepared_receipt["source_kind"] in ACCEPTED_PREPARED_SOURCE_KINDS
            and prepared_receipt["release_eligible"] is False
            and binding["parent_kind"] == "HousingStatusAttachedAsec"
            and binding["source_period_kind"] == "interview_household_universe"
            and binding["native_dtype"] == "uint8"
            and binding["graph_dtype"] == "int64"
            and binding["aliases"] == list(source.ATTACHED_COLUMNS),
            "BOUND_HU_PREPARATION_SCHEMA",
        )
        hu._require(hu._sha(payload) == binding["payload_sha256"], "BOUND_HU_PAYLOAD")
        hu._require(payload.startswith(hu.MAGIC), "BOUND_HU_MAGIC")
        hu._require(hu._sha(payload[:-32]) == payload[-32:].hex(), "BOUND_HU_CHECKSUM")
        size = struct.unpack_from("<I", payload, len(hu.MAGIC))[0]
        start = len(hu.MAGIC) + 4
        hu._require(
            0 < size <= hu.HEADER_MAX_BYTES and start + size < len(payload) - 32,
            "BOUND_HU_HEADER_SIZE",
        )
        header = payload[start : start + size]
        data = hu._parse(header)
        hu._require(set(data) == _HEADER_KEYS, "BOUND_HU_HEADER_SCHEMA")
        hu._require(hu._sha(header) == binding["header_sha256"], "BOUND_HU_HEADER")
        hu._require(
            hu._sha(b"microcosm/asec-housing-universe-content/1\0" + payload[:-32])
            == binding["content_sha256"],
            "BOUND_HU_CONTENT",
        )
        hu._require(
            type(data["schema_version"]) is int
            and data["schema_version"] == 1
            and data["artifact_kind"] == "microcosm.asec_housing_universe.v1"
            and data["source_authentication"] == "checkpoint_bytes_verified"
            and data["release_eligible"] is False
            and data["source_period_kind"] == "interview_household_universe"
            and data["columns"] == list(hu.COLUMNS)
            and data["zero_origin_policy"] == hu.ZERO_POLICY
            and hu._json(data["zero_origin_evidence"])
            == hu._json(hu.ZERO_ORIGIN_EVIDENCE)
            and hu._json(data["dictionaries"]) == hu._json(hu.DICTIONARIES)
            and hu._json(data["codebook"]) == hu._json(hu._codebook()),
            "BOUND_HU_DEFINITION",
        )
        hu._require(
            data["definition_sha256"]
            == binding["definition_sha256"]
            == hu._definition_identity()
            and hu._json(data["implementation"]) == hu._json(source._implementation()),
            "BOUND_HU_IMPLEMENTATION",
        )
        hu._require(
            set(data["source"]) == {"identity", "carried_columns", "income_year_origin"}
            and data["source"]["carried_columns"]
            == ["household_id", *("asec_" + n for n in ("H_SEQ",) + hu.CONTEXT_COLUMNS)]
            and data["source"]["income_year_origin"] == "authenticated_money_scope"
            and hu._sha(data["source"]["identity"].encode())
            == prepared_receipt["source_evidence_sha256"],
            "BOUND_HU_SOURCE",
        )
        rows = data["household_rows"]
        hu._require(
            type(rows) is int
            and 0 < rows <= hu.MAX_HOUSEHOLDS
            and rows == prepared_receipt["entity_rows"]["household"],
            "BOUND_HU_ROWS",
        )
        cursor, buffers = start + size, []
        expected_size = (
            cursor + rows * (8 * len(hu.RAW_COLUMNS) + len(hu.DERIVED_COLUMNS)) + 32
        )
        hu._require(len(payload) == expected_size, "BOUND_HU_LENGTH")
        for name in hu.COLUMNS:
            width = rows * (8 if name in hu.RAW_COLUMNS else 1)
            buffers.append(payload[cursor : cursor + width])
            cursor += width
        body = BoundHousingUniverse(header, tuple(buffers), _token=_TOKEN)
        derived = hu._derive({name: body.array(name) for name in hu.RAW_COLUMNS})
        hu._require(
            all(
                derived[name].tobytes() == body.array(name).tobytes()
                for name in hu.DERIVED_COLUMNS
            ),
            "BOUND_HU_DERIVED",
        )
        return body
    except hu.HousingUniverseRefusalError:
        raise
    except (
        KeyError,
        ValueError,
        TypeError,
        OverflowError,
        AttributeError,
        struct.error,
    ):
        raise hu.HousingUniverseRefusalError("BOUND_HU_CONTRACT") from None


def verify_graph_housing_rows(
    household: pd.DataFrame,
    body: BoundHousingUniverse,
    *,
    positions: np.ndarray,
    income_years: np.ndarray | None = None,
) -> None:
    """Compare graph int64 rows with native evidence; no cast or authority upgrade."""
    hu._require(type(body) is BoundHousingUniverse, "BOUND_HU_TYPE")
    hu._require(
        type(positions) is np.ndarray
        and positions.dtype == np.dtype("int64")
        and positions.shape == (len(household),)
        and bool(
            ((positions >= 0) & (positions < len(body.array("household_id")))).all()
        )
        and (len(positions) < 2 or bool((np.diff(positions) > 0).all())),
        "BOUND_HU_POSITIONS",
    )
    hu._require(household.columns.is_unique, "BOUND_HU_TABLE_COLUMNS")
    for name, column in (
        ("household_id", "household_id"),
        *((n, "asec_" + n) for n in ("H_SEQ",) + hu.CONTEXT_COLUMNS),
        *zip(hu.DERIVED_COLUMNS, source.ATTACHED_COLUMNS, strict=True),
    ):
        hu._require(column in household, "BOUND_HU_CARRIED_COLUMN")
        values = household[column]
        hu._require(
            values.dtype == np.dtype("int64")
            and np.array_equal(values.to_numpy(), body.array(name)[positions]),
            "BOUND_HU_CARRIED_VALUES",
        )
    if income_years is not None:
        hu._require(
            type(income_years) is np.ndarray
            and income_years.dtype == np.dtype("int64")
            and np.array_equal(income_years, body.array("income_year")[positions]),
            "BOUND_HU_COHORTS",
        )


__all__ = [
    "BoundHousingUniverse",
    "HOUSEHOLD_EVIDENCE_COLUMNS",
    "US_ASEC_HOUSING_UNIVERSE_TYPE",
    "bind_housing_universe",
    "verify_graph_housing_rows",
]
