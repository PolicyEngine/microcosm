"""Additive full-return PUF source projection, before canonical modeling.

This is not the older 13-column monetary envelope. It preserves every source
return, raw period, integer S006 and missing demographic state. Only ordinary
returns carry numeric amounts; disclosure aggregates retain sparse lexical
tokens and use an explicit int64 sentinel. The unchanged raw decoder owns CSV
layout, status-code and join validation. This owner authenticates both exact
delivery buffers before calling it and binds the subsequent projection to its
complete RECID sequence. No tax engine, growth, fit or donor admission occurs.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from . import puf_raw_source as raw

FULL_SOURCE_VERSION = "microcosm.us.puf_2015_full_return_source/3"
FULL_SOURCE_MAX_BYTES = 128 * 1024 * 1024
OUTSIDE_AMOUNT_UNIVERSE = -(2**63)

# Explicit publisher amount-field roster. No role is inferred from a prefix.
# E30400/E30500 deliberately excluded: capped Schedule SE amounts cannot
# reconstruct uncapped partnership earnings.
MONEY_COLUMNS = (
    "E00100",
    "E00200",
    "E00300",
    "E00400",
    "E00600",
    "E00650",
    "E00700",
    "E00800",
    "E00900",
    "E01100",
    "E01200",
    "E01400",
    "E01700",
    "E02100",
    "E02400",
    "E03150",
    "E03210",
    "E03220",
    "E03230",
    "E03240",
    "E03290",
    "E03300",
    "E03500",
    "E18500",
    "E19200",
    "E19800",
    "E20100",
    "E20400",
    "E20500",
    "P22250",
    "P23250",
    "E24515",
    "E24518",
    "E25850",
    "E25860",
    "E25940",
    "E25980",
    "E25920",
    "E25960",
    "E26110",
    "E26170",
    "E26190",
    "E26160",
    "E26180",
    "E26270",
    "E26100",
    "E26390",
    "E26400",
    "E27200",
    "T27800",
    "E58990",
    "E87530",
)
COUNT_COLUMNS = ("XFPT", "XFST", "XOCAH", "XOCAWH", "XOODEP", "XOPAR", "XTOT")
DEPENDENT_COLUMNS = ("XOCAH", "XOCAWH", "XOODEP", "XOPAR")
PROJECTED_COLUMNS = (*MONEY_COLUMNS, *COUNT_COLUMNS)
_MONEY = re.compile(r"-?[0-9]{1,12}\Z", re.ASCII)
_COUNT = re.compile(r"[0-9]{1,2}\Z", re.ASCII)
# Operational preservation bound for four out-of-universe source rows.
# Empty, scientific notation and literal formatting are retained, not parsed.
AGGREGATE_LEXEME_MAX_CHARACTERS = 64
_AGGREGATE = re.compile(r"[\x20-\x7e]{0,64}\Z", re.ASCII)


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _validate_count_fields(values, status):
    """Publisher disclosure-coded counts; never a household/person roster."""
    ordinary = status["disclosure_aggregate"] == 0
    mars = status["MARS"][ordinary]
    caps = np.select([mars == 1, mars == 2, mars == 3, mars == 4], [2, 3, 1, 3], -1)
    _require((caps >= 0).all(), "FULL_COUNT_MARS")
    for name in ("XFPT", "XFST"):
        _require(
            np.isin(values[name][ordinary], (0, 1)).all(), "FULL_EXEMPTION_FLAG:" + name
        )
    cumulative = np.zeros(len(mars), dtype=np.int64)
    for name in DEPENDENT_COLUMNS:
        count = values[name][ordinary]
        _require(((count >= 0) & (count <= 3)).all(), "FULL_DEPENDENT_DOMAIN:" + name)
        cumulative += count
        _require((cumulative <= caps).all(), "FULL_DEPENDENT_SEQUENTIAL_CAP:" + name)
    xtot = values["XTOT"][ordinary]
    _require(((xtot >= 0) & (xtot <= 5)).all(), "FULL_EXEMPTIONS_DOMAIN")
    _require(
        np.array_equal(
            xtot, values["XFPT"][ordinary] + values["XFST"][ordinary] + cumulative
        ),
        "FULL_EXEMPTIONS_COMPONENT_IDENTITY",
    )


def puf_2015_receiver_size_measurement(filing_status_code, dependent_count):
    """Match the PUF's coarse status and censored return-size predictor.

    Generic model status codes are single1/joint2/separate3/HOH4/widow5.
    PUF code2 combines joint and surviving-spouse returns. Its baseline of two
    is a predictor convention for that combined class, not an observed number
    of living filers. Physical survey membership must remain unchanged.
    """
    status = np.asarray(filing_status_code)
    dependents = np.asarray(dependent_count)
    _require(
        status.ndim == 1 and dependents.shape == status.shape,
        "FULL_RECEIVER_COUNT_SHAPE",
    )
    _require(
        status.dtype.kind in "iu" and dependents.dtype.kind in "iu",
        "FULL_RECEIVER_COUNT_TYPE",
    )
    _require(np.isin(status, (1, 2, 3, 4, 5)).all(), "FULL_RECEIVER_STATUS_DOMAIN")
    _require(
        ((dependents >= 0) & (dependents <= 1000)).all(),
        "FULL_RECEIVER_DEPENDENT_BOUND",
    )
    mars = np.where(status == 5, 2, status).astype(np.int64)
    cap = np.select([mars == 1, mars == 2, mars == 3, mars == 4], [2, 3, 1, 3])
    return {
        "puf_2015_filing_status_code": mars,
        "puf_2015_capped_return_size": 1
        + (mars == 2).astype(np.int64)
        + np.minimum(dependents, cap),
    }


def _frozen_array(values):
    values = np.asarray(values)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True)
class FullPufSource:
    """Read-only source values; available is not an admission or truth flag."""

    definition_sha256: str
    source_sha256: Mapping[str, str]
    status: Mapping[str, np.ndarray]
    values: Mapping[str, np.ndarray]
    aggregate_tokens: Mapping[int, Mapping[str, str]]
    status_payload: bytes

    @property
    def ordinary(self):
        return self.status["disclosure_aggregate"] == 0


def decode_full_puf_source(main_bytes, demographic_bytes, definition):
    """Authenticate, decode, and align a complete declared delivery in memory."""
    _require(isinstance(definition, raw.PufRawSourceDefinition), "FULL_DEFINITION")
    for data, pin in (
        (main_bytes, definition.main),
        (demographic_bytes, definition.demographic),
    ):
        _require(type(data) is bytes and len(data) == pin.bytes, "FULL_SOURCE_SIZE")
        _require(hashlib.sha256(data).hexdigest() == pin.sha256, "FULL_SOURCE_SHA256")
    status = raw.decode_puf_raw_source(main_bytes, demographic_bytes, definition)
    rows = len(status.typed["RECID"])
    _require(
        rows * len(PROJECTED_COLUMNS) * 8 <= FULL_SOURCE_MAX_BYTES,
        "FULL_PROJECTION_BODY_LIMIT",
    )
    profile = definition.document["csv_profile"]
    reader = raw._reader(main_bytes, profile)
    header = raw._check_header(reader, definition.main, profile=profile)
    _require(set(PROJECTED_COLUMNS) <= set(header), "FULL_PROJECTED_HEADER")
    positions = {name: header.index(name) for name in PROJECTED_COLUMNS}
    recid_position = header.index("RECID")
    values = {
        name: np.full(rows, OUTSIDE_AMOUNT_UNIVERSE, dtype="<i8")
        for name in PROJECTED_COLUMNS
    }
    aggregate_tokens = {}
    count = 0
    for row_index, record in enumerate(
        raw._records(reader, definition.main, profile=profile)
    ):
        _require(row_index < rows, "FULL_EXCESS_RECORD")
        recid = int(record[recid_position])  # Already grammar-checked by raw owner.
        _require(recid == int(status.typed["RECID"][row_index]), "FULL_RECID_SEQUENCE")
        aggregate = bool(status.typed["disclosure_aggregate"][row_index])
        if aggregate:
            tokens = {name: record[index] for name, index in positions.items()}
            _require(
                all(_AGGREGATE.fullmatch(token) for token in tokens.values()),
                "FULL_AGGREGATE_LEXICAL_GRAMMAR",
            )
            aggregate_tokens[recid] = MappingProxyType(tokens)
        else:
            for name, index in positions.items():
                token = record[index]
                grammar = _COUNT if name in COUNT_COLUMNS else _MONEY
                _require(
                    grammar.fullmatch(token) is not None,
                    "FULL_ORDINARY_TOKEN_GRAMMAR:" + name,
                )
                values[name][row_index] = int(token)
        count += 1
    _require(count == rows, "FULL_RECORD_COUNT")
    _validate_count_fields(values, status.typed)
    return FullPufSource(
        definition_sha256=definition.sha256,
        source_sha256=MappingProxyType(
            {
                "main": definition.main.sha256,
                "demographic": definition.demographic.sha256,
            }
        ),
        status=MappingProxyType(
            {name: _frozen_array(v) for name, v in status.typed.items()}
        ),
        values=MappingProxyType({name: _frozen_array(v) for name, v in values.items()}),
        aggregate_tokens=MappingProxyType(aggregate_tokens),
        status_payload=raw.encode_return_status(status, definition),
    )


_MAGIC = (FULL_SOURCE_VERSION + "\n").encode("ascii")
_HEADER_LIMIT = 128 * 1024


def encode_full_puf_source(source):
    """Encode a deterministic private typed artifact with independent digests."""
    _require(isinstance(source, FullPufSource), "FULL_ENCODE_TYPE")
    blocks = [source.status_payload]
    entries = []
    for name in PROJECTED_COLUMNS:
        data = source.values[name].astype("<i8", copy=False).tobytes()
        entries.append(
            {
                "name": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        blocks.append(data)
    header = json.dumps(
        {
            "definition_sha256": source.definition_sha256,
            "source_sha256": dict(source.source_sha256),
            "status_bytes": len(source.status_payload),
            "status_sha256": hashlib.sha256(source.status_payload).hexdigest(),
            "rows": len(source.status["RECID"]),
            "columns": entries,
            "aggregate_tokens": {
                str(k): dict(v) for k, v in source.aggregate_tokens.items()
            },
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    _require(len(header) <= _HEADER_LIMIT, "FULL_HEADER_LIMIT")
    _require(sum(len(b) for b in blocks) <= FULL_SOURCE_MAX_BYTES, "FULL_BODY_LIMIT")
    return _MAGIC + struct.pack("<Q", len(header)) + header + b"".join(blocks)


def decode_full_puf_artifact(payload, definition):
    """Decode the closed additive artifact against its exact source definition."""
    _require(type(payload) is bytes and payload.startswith(_MAGIC), "FULL_MAGIC")
    _require(
        len(payload) <= len(_MAGIC) + 8 + _HEADER_LIMIT + FULL_SOURCE_MAX_BYTES,
        "FULL_ARTIFACT_LIMIT",
    )
    offset = len(_MAGIC)
    _require(len(payload) >= offset + 8, "FULL_HEADER_LENGTH")
    size = struct.unpack("<Q", payload[offset : offset + 8])[0]
    _require(size <= _HEADER_LIMIT and size > 0, "FULL_HEADER_LIMIT")
    offset += 8
    _require(offset + size <= len(payload), "FULL_TRUNCATED_HEADER")
    header_bytes = payload[offset : offset + size]
    try:
        h = json.loads(header_bytes)
    except (UnicodeError, ValueError) as error:
        raise ValueError("FULL_HEADER_JSON") from error
    _require(
        type(h) is dict
        and set(h)
        == {
            "definition_sha256",
            "source_sha256",
            "status_bytes",
            "status_sha256",
            "rows",
            "columns",
            "aggregate_tokens",
        },
        "FULL_HEADER_KEYS",
    )
    _require(
        json.dumps(
            h, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        == header_bytes,
        "FULL_HEADER_CANONICAL",
    )
    _require(
        h["definition_sha256"] == definition.sha256
        and h["source_sha256"]
        == {
            "main": definition.main.sha256,
            "demographic": definition.demographic.sha256,
        },
        "FULL_ARTIFACT_SOURCE_BINDING",
    )
    rows = h["rows"]
    _require(
        type(rows) is int and rows == definition.main.data_records, "FULL_ARTIFACT_ROWS"
    )
    _require(
        type(h["status_bytes"]) is int
        and 0 < h["status_bytes"] <= FULL_SOURCE_MAX_BYTES,
        "FULL_STATUS_LENGTH",
    )
    offset += size
    status_payload = payload[offset : offset + h["status_bytes"]]
    _require(
        len(status_payload) == h["status_bytes"]
        and hashlib.sha256(status_payload).hexdigest() == h["status_sha256"],
        "FULL_STATUS_SHA256",
    )
    status = raw.decode_return_status(status_payload, definition=definition)
    offset += h["status_bytes"]
    _require(
        type(h["columns"]) is list and len(h["columns"]) == len(PROJECTED_COLUMNS),
        "FULL_ARTIFACT_COLUMNS",
    )
    values = {}
    for name, entry in zip(PROJECTED_COLUMNS, h["columns"], strict=True):
        _require(
            type(entry) is dict
            and set(entry) == {"name", "bytes", "sha256"}
            and entry["name"] == name
            and type(entry["bytes"]) is int
            and entry["bytes"] == rows * 8,
            "FULL_COLUMN_DECLARATION",
        )
        data = payload[offset : offset + entry["bytes"]]
        _require(
            len(data) == entry["bytes"]
            and hashlib.sha256(data).hexdigest() == entry["sha256"],
            "FULL_COLUMN_SHA256",
        )
        values[name] = np.frombuffer(data, dtype="<i8")
        offset += entry["bytes"]
    _require(offset == len(payload), "FULL_TRAILING_BYTES")
    aggregate = status.typed["disclosure_aggregate"] == 1
    tokens = h["aggregate_tokens"]
    ids = status.typed["RECID"][aggregate]
    _require(
        type(tokens) is dict and set(tokens) == {str(int(i)) for i in ids},
        "FULL_AGGREGATE_TOKEN_KEYS",
    )
    for fields in tokens.values():
        _require(
            type(fields) is dict
            and set(fields) == set(PROJECTED_COLUMNS)
            and all(
                type(v) is str and _AGGREGATE.fullmatch(v) for v in fields.values()
            ),
            "FULL_AGGREGATE_LEXICAL_GRAMMAR",
        )
    for name, value in values.items():
        _require(
            (value[aggregate] == OUTSIDE_AMOUNT_UNIVERSE).all(),
            "FULL_AGGREGATE_SENTINEL",
        )
        cap = 99 if name in COUNT_COLUMNS else 999_999_999_999
        _require(
            ((value[~aggregate] >= -cap) & (value[~aggregate] <= cap)).all()
            and (name not in COUNT_COLUMNS or (value[~aggregate] >= 0).all()),
            "FULL_ORDINARY_TYPED_DOMAIN",
        )
    _validate_count_fields(values, status.typed)
    return FullPufSource(
        definition.sha256,
        MappingProxyType(h["source_sha256"]),
        status.typed,
        MappingProxyType(values),
        MappingProxyType({int(k): MappingProxyType(v) for k, v in tokens.items()}),
        status_payload,
    )


DIRECT_MAPPINGS = {
    "employment_income_before_lsr": "E00200",
    "self_employment_income_before_lsr": "E00900",
    "taxable_interest_income": "E00300",
    "qualified_dividend_income": "E00650",
    "tax_exempt_interest_income": "E00400",
    "short_term_capital_gains": "P22250",
    "long_term_capital_gains_before_response": "P23250",
    "long_term_capital_gains_on_collectibles": "E24518",
    "non_sch_d_capital_gains": "E01100",
    "taxable_private_pension_income": "E01700",
    "taxable_ira_distributions": "E01400",
    "alimony_income": "E00800",
    "alimony_expense": "E03500",
    "salt_refund_income": "E00700",
    "charitable_cash_donations": "E19800",
    "charitable_non_cash_donations": "E20100",
    "real_estate_taxes": "E18500",
    "investment_income_elected_form_4952": "E58990",
    "student_loan_interest": "E03210",
    "educator_expense": "E03220",
    "casualty_loss": "E20500",
    "farm_income": "T27800",
    "farm_operations_income": "E02100",
    "farm_rent_income": "E27200",
    "miscellaneous_income": "E01200",
    "domestic_production_ald": "E03240",
    "unrecaptured_section_1250_gain": "E24515",
    "health_savings_account_ald": "E03290",
}


def observed_and_derived_return_columns(source, *, selected_recids=None):
    """Return canonical observed/derived quantities plus explicit model inputs.

    This stage intentionally retains total SS, total interest, realized IRA/
    Keogh deductions and total miscellaneous itemized deductions under honest
    auxiliary names. The next declared model owner supplies component/proxy
    outputs; their absence is never silently converted to a known zero here.
    """
    _require(isinstance(source, FullPufSource), "FULL_SOURCE_TYPE")
    mask = source.ordinary.copy()
    if selected_recids is not None:
        ids = tuple(selected_recids)
        _require(
            len(ids) > 0
            and all(type(i) is int for i in ids)
            and len(set(ids)) == len(ids),
            "FULL_SELECTION_IDS",
        )
        _require(
            set(ids) <= set(source.status["RECID"][mask].tolist()),
            "FULL_SELECTION_UNIVERSE",
        )
        mask &= np.isin(source.status["RECID"], ids)
    a = {name: value[mask] for name, value in source.values.items()}
    out = {name: a[field].copy() for name, field in DIRECT_MAPPINGS.items()}
    out.update(
        {
            "non_qualified_dividend_income": a["E00600"] - a["E00650"],
            "rental_income": a["E25850"] - a["E25860"],
            "estate_income": a["E26390"] - a["E26400"],
            "partnership_income": (
                a["E25940"] + a["E25980"] - a["E25920"] - a["E25960"] - a["E26110"]
            ),
            "s_corp_income": (
                a["E26170"] + a["E26190"] - a["E26160"] - a["E26180"] - a["E26100"]
            ),
        }
    )
    auxiliary = {
        "raw_adjusted_gross_income": a["E00100"],
        "raw_total_social_security": a["E02400"],
        "raw_total_interest_deduction": a["E19200"],
        "raw_realized_ira_deduction": a["E03150"],
        "raw_realized_keogh_deduction": a["E03300"],
        "raw_tuition_fees_deduction": a["E03230"],
        "raw_lifetime_learning_qualified_expenses": a["E87530"],
        "raw_miscellaneous_itemized_deductions": a["E20400"],
        "raw_partnership_nonpassive_net": a["E25980"] - a["E25960"],
        "raw_partnership_s_corp_combined": a["E26270"],
        "raw_exemptions_count": a["XTOT"],
        **{"raw_" + name.lower(): a[name] for name in COUNT_COLUMNS[:-1]},
        "reported_dependent_count": sum(a[name] for name in DEPENDENT_COLUMNS),
        "puf_2015_filing_status_code": source.status["MARS"][mask].copy(),
        "puf_2015_capped_return_size": (
            1
            + (source.status["MARS"][mask] == 2).astype(np.int64)
            + sum(a[name] for name in DEPENDENT_COLUMNS)
        ),
        "raw_ordinary_dividend_income": a["E00600"],
    }
    auxiliary["partnership_component_reconciliation_residual"] = (
        out["partnership_income"] + out["s_corp_income"] - a["E26270"]
    )
    # Canonical intermediates stay integer dollars until the explicit model
    # boundary. S006 stays integer hundredths here too.
    status = {name: value[mask].copy() for name, value in source.status.items()}
    return out, auxiliary, status
