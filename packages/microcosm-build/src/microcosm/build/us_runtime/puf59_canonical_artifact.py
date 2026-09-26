"""Bounded deterministic private canonical59 array envelope; no source admission."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping

import numpy as np

from .puf59_canonical import PREFIX_NAMES, VERSION, prefix_values_digest
from .puf_target2024_growth import INCIDENCE_FIELDS, OUTPUTS, RECIPE_SHA256, _digest

MAGIC = b"MCPUF59\x02"
MAX_HEADER = 128 * 1024
MAX_BODY = 128 * 1024 * 1024
PREFIX = PREFIX_NAMES
NAMES = (*PREFIX, *OUTPUTS)
INTEGERS = set(PREFIX) - {"weight"} | set(INCIDENCE_FIELDS)


def _require(value, code):
    if not value:
        raise ValueError(code)


def _receipt(receipt, expected_growth_scheme):
    _require(
        expected_growth_scheme in ("family_observed", "cpi_only"),
        "PUF59_ARTIFACT_EXPECTED_SCHEME",
    )
    _require(isinstance(receipt, Mapping), "PUF59_ARTIFACT_RECEIPT_TYPE")
    r = dict(receipt)
    _require(
        r.get("schema") == VERSION
        and type(r.get("input_money_year")) is int
        and r["input_money_year"] == 2015
        and type(r.get("output_money_year")) is int
        and r["output_money_year"] == 2024
        and type(r.get("rows")) is int
        and 0 < r["rows"] <= MAX_BODY // (len(NAMES) * 8)
        and r.get("release_eligible") is False,
        "PUF59_ARTIFACT_RECEIPT_CONTRACT",
    )
    gr = r.get("growth")
    _require(
        isinstance(gr, Mapping) and gr.get("scheme") == expected_growth_scheme,
        "PUF59_ARTIFACT_GROWTH_SCHEME",
    )
    _require(gr.get("recipe_sha256") == RECIPE_SHA256, "PUF59_ARTIFACT_GROWTH_RECIPE")
    h = r.pop("sha256", None)
    _require(
        h
        == hashlib.sha256(
            json.dumps(
                r, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        "PUF59_ARTIFACT_RECEIPT_HASH",
    )
    r["sha256"] = h
    return r


def _validate(arrays, n):
    _require(
        type(n) is int and 0 < n <= MAX_BODY // (len(NAMES) * 8), "PUF59_ARTIFACT_ROWS"
    )
    _require(
        isinstance(arrays, Mapping) and set(arrays) == set(NAMES),
        "PUF59_ARTIFACT_COLUMNS",
    )
    for name in NAMES:
        a = np.asarray(arrays[name])
        kind = "iu" if name in INTEGERS else "fiu"
        _require(
            a.shape == (n,) and a.dtype.kind in kind,
            "PUF59_ARTIFACT_COLUMN_TYPE:" + name,
        )
        _require(bool(np.isfinite(a).all()), "PUF59_ARTIFACT_NONFINITE:" + name)
        if a.dtype.kind in "iu":
            _require(
                bool(((a >= -(2**53)) & (a <= 2**53)).all()),
                "PUF59_ARTIFACT_INTEGER_RANGE:" + name,
            )
    ids = arrays["RECID"]
    w = arrays["weight"]
    _require(
        bool((ids > 0).all())
        and len(np.unique(ids)) == n
        and not np.isin(ids, [999996, 999997, 999998, 999999]).any(),
        "PUF59_ARTIFACT_ID_DOMAIN",
    )
    _require(
        bool((w >= 0).all()) and np.isfinite(w.sum()) and w.sum() > 0,
        "PUF59_ARTIFACT_WEIGHT",
    )
    _require(
        bool((arrays["puf_person_incidence_capacity"] == 1).all()),
        "PUF59_ARTIFACT_CAPACITY",
    )
    _require(
        bool(np.isin(arrays["puf_2015_filing_status_code"], [1, 2, 3, 4]).all()),
        "PUF59_ARTIFACT_STATUS",
    )
    size = arrays["puf_2015_capped_return_size"]
    _require(bool(((size >= 1) & (size <= 5)).all()), "PUF59_ARTIFACT_RETURN_SIZE")
    for name in INCIDENCE_FIELDS:
        _require(bool(np.isin(arrays[name], [0, 1]).all()), "PUF59_ARTIFACT_INCIDENCE")


def _bindings(arrays, receipt):
    _require(
        prefix_values_digest(arrays) == receipt.get("prefix_values_sha256"),
        "PUF59_ARTIFACT_PREFIX_BINDING",
    )
    _require(
        _digest(arrays) == receipt["growth"].get("output_values_sha256"),
        "PUF59_ARTIFACT_GROWTH_BINDING",
    )


def _encode(arrays, receipt, expected_growth_scheme):
    _require(
        isinstance(arrays, Mapping) and set(arrays) == set(NAMES),
        "PUF59_ARTIFACT_COLUMNS",
    )
    n = len(arrays["RECID"])
    _validate(arrays, n)
    r = _receipt(receipt, expected_growth_scheme)
    _bindings(arrays, r)
    _require(r["rows"] == n, "PUF59_ARTIFACT_RECEIPT_ROWS")
    blocks = [
        np.asarray(arrays[name], dtype="<i8" if name in INTEGERS else "<f8").tobytes()
        for name in NAMES
    ]
    body = b"".join(blocks)
    header = {
        "schema": "microcosm.us.puf59_canonical_artifact/2",
        "rows": n,
        "names": list(NAMES),
        "dtypes": ["<i8" if name in INTEGERS else "<f8" for name in NAMES],
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "receipt": r,
    }
    h = json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    _require(len(h) <= MAX_HEADER and len(body) <= MAX_BODY, "PUF59_ARTIFACT_SIZE")
    return MAGIC + struct.pack("<I", len(h)) + h + body


def encode_canonical_puf59(result, *, expected_growth_scheme="family_observed"):
    arrays = {
        "RECID": result.recids,
        "weight": result.design_weight,
        "puf_person_incidence_capacity": result.person_incidence_capacity,
        **result.source_predictors,
        **result.columns,
    }
    _require(result.money_year == 2024, "PUF59_ARTIFACT_MONEY_YEAR")
    return _encode(arrays, result.receipt, expected_growth_scheme)


def decode_canonical_puf59(payload, *, expected_growth_scheme="family_observed"):
    _require(
        type(payload) is bytes
        and len(MAGIC) + 4 < len(payload) <= len(MAGIC) + 4 + MAX_HEADER + MAX_BODY,
        "PUF59_ARTIFACT_SIZE",
    )
    _require(payload[: len(MAGIC)] == MAGIC, "PUF59_ARTIFACT_MAGIC")
    length = struct.unpack("<I", payload[len(MAGIC) : len(MAGIC) + 4])[0]
    _require(
        0 < length <= MAX_HEADER and len(payload) > len(MAGIC) + 4 + length,
        "PUF59_ARTIFACT_HEADER_SIZE",
    )
    start = len(MAGIC) + 4

    def pairs(items):
        d = {}
        for k, v in items:
            _require(k not in d, "PUF59_ARTIFACT_DUPLICATE_JSON_KEY")
            d[k] = v
        return d

    h = json.loads(
        payload[start : start + length],
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("PUF59_ARTIFACT_JSON_NONFINITE")
        ),
    )
    _require(
        type(h) is dict
        and set(h) == {"schema", "rows", "names", "dtypes", "body_sha256", "receipt"},
        "PUF59_ARTIFACT_HEADER_CONTRACT",
    )
    _require(
        json.dumps(h, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        == payload[start : start + length],
        "PUF59_ARTIFACT_HEADER_CANONICAL",
    )
    _require(
        h["schema"] == "microcosm.us.puf59_canonical_artifact/2"
        and h["names"] == list(NAMES)
        and h["dtypes"] == ["<i8" if name in INTEGERS else "<f8" for name in NAMES],
        "PUF59_ARTIFACT_SCHEMA",
    )
    n = h["rows"]
    _require(
        type(n) is int and 0 < n <= MAX_BODY // (len(NAMES) * 8), "PUF59_ARTIFACT_ROWS"
    )
    body = memoryview(payload)[start + length :]
    _require(
        len(body) == n * 8 * len(NAMES)
        and hashlib.sha256(body).hexdigest() == h["body_sha256"],
        "PUF59_ARTIFACT_BODY_IDENTITY",
    )
    arrays = {
        name: np.frombuffer(body, dtype=h["dtypes"][i], count=n, offset=i * n * 8)
        for i, name in enumerate(NAMES)
    }
    _validate(arrays, n)
    receipt = _receipt(h["receipt"], expected_growth_scheme)
    _bindings(arrays, receipt)
    _require(receipt["rows"] == n, "PUF59_ARTIFACT_RECEIPT_ROWS")
    return arrays, receipt


def reencode_canonical_puf59(
    arrays, receipt, *, expected_growth_scheme="family_observed"
):
    """Round-trip verification; this does not assign execution/source status."""
    return _encode(arrays, receipt, expected_growth_scheme)
