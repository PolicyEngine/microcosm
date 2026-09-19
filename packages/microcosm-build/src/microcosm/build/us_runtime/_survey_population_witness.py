"""Private immutable observations for the compact native receiving profile.

These bytes confer no authority. Only the financial issuer's retained capsule
may supply an expected observation to completion. Unsupported/malformed actual
state can refuse at observation time, before the full replay oracle would run.
The original replay comparator remains unchanged and is the test oracle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from microcosm.frame import EntitySchema, Frame, MassChangeRecord, WeightKind
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import MassRecord, Population
from microcosm.graph.store import _encode_frame_metadata, _encode_object_scalar

PROTOCOL = "microcosm.private-survey-observation.v1"
_MASKED = tuple(
    cls()
    for cls in (
        pd.Int8Dtype,
        pd.Int16Dtype,
        pd.Int32Dtype,
        pd.Int64Dtype,
        pd.UInt8Dtype,
        pd.UInt16Dtype,
        pd.UInt32Dtype,
        pd.UInt64Dtype,
        pd.BooleanDtype,
    )
)


def _admit(condition, reason):
    if not condition:
        raise ValueError("SURVEY_POPULATION_WITNESS_" + reason)


def _same(condition, reason):
    if not condition:
        raise ValueError("SURVEY_POPULATION_REPLAY_" + reason)


def _dtype(dtype):
    # Tokens are selected only after exact reviewed type/value admission.
    for number, admitted in enumerate(_MASKED):
        if type(dtype) is type(admitted) and dtype == admitted:
            return ["masked", number]
    if type(dtype) is pd.StringDtype:
        _admit(dtype.storage in ("python", "pyarrow"), "STRING_STORAGE")
        na = "NA" if dtype.na_value is pd.NA else "nan"
        _admit(
            na == "NA" or (type(dtype.na_value) is float and np.isnan(dtype.na_value)),
            "STRING_NA",
        )
        _admit(
            dtype == pd.StringDtype(storage=dtype.storage, na_value=dtype.na_value),
            "STRING_DTYPE",
        )
        return ["string", dtype.storage, na]
    _admit(isinstance(dtype, np.dtype), "DTYPE")
    _admit(
        dtype.fields is None
        and dtype.subdtype is None
        and dtype.metadata is None
        and type(dtype) is type(np.dtype(dtype.str))
        and dtype == np.dtype(dtype.str)
        and (
            (dtype.kind in "biuf" and dtype.itemsize in (1, 2, 4, 8))
            or dtype == np.dtype(object)
        ),
        "DTYPE",
    )
    return ["object" if dtype.hasobject else "native", dtype.str]


def _array(array):
    _admit(type(array) is np.ndarray and not array.dtype.hasobject, "ARRAY")
    token = _dtype(array.dtype)
    digest = hashlib.sha256()
    # Bounded temporary buffers, including strided/Fortran-order arrays.
    for chunk in np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=65536,
    ):
        digest.update(chunk.tobytes())
    return [token, list(array.shape), digest.hexdigest()]


def _tokens(values, encode):
    digest = hashlib.sha256()
    for value in values:
        try:
            payload = encode(value)
        except (TypeError, ValueError, OverflowError, UnicodeError):
            raise ValueError("SURVEY_POPULATION_WITNESS_SCALAR") from None
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _text(value):
    _admit(type(value) is str, "STRING_VALUE")
    return value.encode("utf-8")


def _series(series):
    _admit(type(series) is pd.Series, "SERIES_TYPE")
    token = _dtype(series.dtype)
    result = {"dtype": token, "length": len(series)}
    if token[0] == "masked":
        expected_array_type = (
            pd.arrays.BooleanArray
            if type(series.dtype) is pd.BooleanDtype
            else pd.arrays.IntegerArray
        )
        _admit(type(series.array) is expected_array_type, "MASKED_ARRAY_TYPE")
        values, mask = series.array._data, series.array._mask
        _admit(
            type(mask) is np.ndarray
            and mask.dtype == np.dtype(bool)
            and mask.shape == (len(series),)
            and type(values) is np.ndarray
            and values.shape == mask.shape,
            "MASKED_STORAGE",
        )
        result.update(
            mask=_array(mask),
            present=_array(values[~mask]),
            full=_array(values),
            null_zero=bool(np.all(values[mask] == 0)),
        )
    elif token[0] == "string":
        admitted = (
            pd.arrays.StringArray
            if token[1] == "python"
            else pd.arrays.ArrowStringArray
        )
        _admit(type(series.array) is admitted, "STRING_ARRAY_TYPE")
        mask = series.isna().to_numpy(dtype=bool, copy=False)
        result.update(
            mask=_array(mask),
            values=_tokens(
                (
                    value
                    for missing, value in zip(mask, series, strict=True)
                    if not missing
                ),
                _text,
            ),
        )
    elif token[0] == "object":
        result["values"] = _tokens(series, _encode_object_scalar)
    else:
        result["values"] = _array(series.to_numpy(copy=False))
    return result


def _name(name):
    _admit(name is None or type(name) is str, "AXIS_NAME")
    return name


def _axis(axis):
    _admit(type(axis) in (pd.Index, pd.RangeIndex), "AXIS_TYPE")
    # Reviewed pandas Index.identical also checks its _comparables. Exact
    # plain/RangeIndex currently have only name; refuse any new semantics.
    _admit(tuple(axis._comparables) == ("name",), "AXIS_COMPARATORS")
    sequence = None
    if type(axis) is pd.RangeIndex:
        _admit(type(axis._range) is range, "RANGE_STORAGE")
        # RangeIndex.equals uses _range, whereas .array may be cached. Bind
        # both independently so stale cached values cannot hide a changed range.
        size = len(axis._range)
        sequence = [
            size,
            axis._range.start if size else None,
            axis._range.step if size > 1 else None,
        ]
    token = _dtype(axis.dtype)
    _admit(
        (token[0] == "native" and axis.dtype.kind in "iu")
        or (
            token[0] in ("object", "string")
            and all(type(value) is str for value in axis)
        ),
        "AXIS_VALUES",
    )
    return {
        "kind": "range" if type(axis) is pd.RangeIndex else "index",
        "name": _name(axis.name),
        "range_sequence": sequence,
        "values": _series(pd.Series(axis.array, copy=False)),
    }


def _frame(frame):
    _admit(type(frame) is Frame and type(frame.schema) is EntitySchema, "FRAME_TYPE")
    _admit(frame.links == frame.schema.links == (), "FRAME_LINKS")
    _admit(
        type(frame.schema.person_entity) is str
        and type(frame.schema.group_entities) is tuple
        and all(type(name) is str for name in frame.schema.group_entities),
        "SCHEMA",
    )
    _admit(all(type(row) is MassChangeRecord for row in frame.mass_log), "MASS_LOG")
    tables = []
    for entity in frame.entities:
        table = frame.table(entity)
        _admit(type(table) is pd.DataFrame, "TABLE_TYPE")
        _admit(type(table.flags.allows_duplicate_labels) is bool, "TABLE_FLAGS")
        tables.append(
            {
                "flags": table.flags.allows_duplicate_labels,
                "index": _axis(table.index),
                "columns": _axis(table.columns),
                "series": [_series(table[column]) for column in table],
            }
        )
    weights = []
    for entity in frame.weighted_entities:
        item = frame.weights_for(entity)
        _admit(type(item.kind) is WeightKind, "WEIGHT_KIND")
        weights.append([entity, item.kind.value, _array(item.values)])
    return {
        "context": {
            "schema": asdict(frame.schema),
            "entities": list(frame.entities),
            "metadata": _encode_frame_metadata(frame.metadata),
            "mass_log": [asdict(row) for row in frame.mass_log],
            "weighted_entities": list(frame.weighted_entities),
        },
        "tables": tables,
        "strata_index": _axis(frame.strata.index),
        "strata_name": _name(frame.strata.name),
        "strata": _series(frame.strata),
        "weights": weights,
    }


def population_witness(population):
    """Capture a detached observation; caller supplies issuance, never these bytes."""
    _admit(type(population) is Population, "POPULATION_TYPE")
    _admit(type(population.version) is str, "VERSION")
    _admit(
        all(
            type(key) is tuple
            and len(key) == 2
            and all(type(value) is str for value in key)
            and type(writer) is str
            for key, writer in population.owners.items()
        ),
        "OWNERS",
    )
    _admit(
        all(type(kind) is WeightKind for kind in population.weight_kind.values()),
        "WEIGHT_KINDS",
    )
    _admit(
        type(population.mass_ledger) is tuple
        and all(type(row) is MassRecord for row in population.mass_ledger),
        "MASS_LEDGER",
    )
    return canonical_json(
        {
            "protocol": PROTOCOL,
            "frame": _frame(population.frame),
            "context": {
                "version": population.version,
                "owners": sorted(population.owners.items()),
                "weight_kind": [
                    [entity, kind.value]
                    for entity, kind in population.weight_kind.items()
                ],
                "mass_ledger": [asdict(row) for row in population.mass_ledger],
                "design_entities": list(population.design_weights),
            },
            "design_weights": [
                _array(values) for values in population.design_weights.values()
            ],
        }
    )


def _decode(witness):
    _admit(type(witness) is bytes, "BYTES")
    result = json.loads(witness)
    _admit(
        result.get("protocol") == PROTOCOL and canonical_json(result) == witness,
        "PROTOCOL",
    )
    return result


def _same_series(expected, actual):
    _same(
        expected["dtype"] == actual["dtype"] and expected["length"] == actual["length"],
        "SERIES_DTYPE_OR_LENGTH",
    )
    branch = expected["dtype"][0]
    if branch == "masked":
        _same(
            expected["mask"] == actual["mask"]
            and expected["full"][:2] == actual["full"][:2],
            "MASKED_STORAGE",
        )
        _same(expected["present"] == actual["present"], "PRESENT_BITS")
        _same(
            expected["full"] == actual["full"] or actual["null_zero"],
            "NONCANONICAL_NULL_BACKING",
        )
    elif branch == "string":
        _same(expected["mask"] == actual["mask"], "STRING_MASK")
        _same(expected["values"] == actual["values"], "STRING_VALUE")
    else:
        _same(
            expected["values"] == actual["values"],
            "OBJECT_VALUE" if branch == "object" else "NATIVE_BITS",
        )


def _same_axis(expected, actual):
    # On the closed plain-Index/RangeIndex profile, exact dtype,
    # name and supported value equality imply pandas' identical predicate.
    _same(
        expected["kind"] == actual["kind"]
        and expected["name"] == actual["name"]
        and expected["range_sequence"] == actual["range_sequence"],
        "AXIS",
    )
    try:
        _same_series(expected["values"], actual["values"])
    except ValueError:
        raise ValueError("SURVEY_POPULATION_REPLAY_AXIS") from None


def same_witness(expected, actual):
    """Directional comparison of two issuer-owned observations, not equality."""
    left, right = _decode(expected), _decode(actual)
    a, b = left["frame"], right["frame"]
    _same(
        canonical_json(a["context"]) == canonical_json(b["context"]),
        "FRAME_CONTEXT",
    )
    for x, y in zip(a["tables"], b["tables"], strict=True):
        _same(x["flags"] == y["flags"], "TABLE_TYPE_OR_FLAGS")
        _same_axis(x["index"], y["index"])
        _same_axis(x["columns"], y["columns"])
        for first, second in zip(x["series"], y["series"], strict=True):
            _same_series(first, second)
    _same_axis(a["strata_index"], b["strata_index"])
    _same(a["strata_name"] == b["strata_name"], "STRATA_NAME")
    _same_series(a["strata"], b["strata"])
    _same(a["weights"] == b["weights"], "WEIGHT_BYTES")
    _same(
        canonical_json(left["context"]) == canonical_json(right["context"]),
        "POPULATION_CONTEXT",
    )
    _same(left["design_weights"] == right["design_weights"], "DESIGN_BYTES")


def same_population_witness(expected, actual):
    same_witness(population_witness(expected), actual)
