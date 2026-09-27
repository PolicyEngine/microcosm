"""Whole-series selection parity at the remaining US in-process seal sites.

Invented series and frames only: no genuine microdata, no source admission, no
store, archive or native read, and no tax engine.  Two of the converted seals
are pure enough to compare byte for byte against verbatim copies of their
pre-change bodies; every converted site — including the three whose inputs are
not cheaply invented — is held by the static guard at the end, which fails if
any call here goes back to building an all-True selector.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import struct

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_predictors as predictors
from microcosm.build.us_runtime import population_input_coverage as coverage
from microcosm.build.us_runtime import puf55_survey_recipients as recipients
from microcosm.build.us_runtime import survey_atomic_geography as geography
from microcosm.build.us_runtime import survey_origin_budget as budget
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import _storage_parts
from microcosm.graph.store import _axis_name_payload, _encode_object_scalar

CONVERTED_MODULES = (
    coverage,
    geography,
    budget,
    predictors,
    recipients,
)


def _reference_series_stamp(series):
    """The exact pre-change body of ``population_input_coverage._series_stamp``."""

    digest = hashlib.sha256()
    digest.update(canonical_json((str(series.dtype), _axis_name_payload(series.name))))
    if pd.api.types.is_object_dtype(series.dtype):
        # Typed scalar encoding, never process-specific PyObject addresses.
        for value in series:
            part = _encode_object_scalar(value)
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    else:
        for part in _storage_parts(series, np.ones(len(series), dtype=np.bool_)):
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return digest.hexdigest()


def _reference_physical_nonweight_identity(frame):
    """The exact pre-change body of ``survey_origin_budget``'s storage seal."""

    digest = hashlib.sha256()
    for series in (
        *(
            frame.table(entity)[column]
            for entity in frame.entities
            for column in frame.table(entity)
        ),
        frame.strata,
    ):
        selected = np.ones(len(series), dtype=np.bool_)
        for part in _storage_parts(series, selected):
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return digest.hexdigest()


def _nan(payload):
    return struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000 | payload))[0]


_NAN_ONE = _nan(0x1)
_NAN_TWO = _nan(0x2)


def _object_series(values, name=None):
    array = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        array[position] = value
    return pd.Series(array, name=name)


def _string_series(values, name=None):
    return pd.Series(
        pd.array(list(values), dtype=pd.StringDtype(storage="python", na_value=pd.NA)),
        name=name,
    )


_SERIES = {
    "empty-float64": lambda: pd.Series(np.array([], dtype=np.float64), name="amount"),
    "float64-specials": lambda: pd.Series(
        np.array(
            [-0.0, 0.0, float("inf"), float("-inf"), 5e-324, _NAN_ONE, _NAN_TWO],
            dtype=np.float64,
        ),
        name="amount",
    ),
    "int64": lambda: pd.Series(np.array([-1, 0, 2**40], dtype=np.int64), name="count"),
    "bool": lambda: pd.Series(np.array([True, False], dtype=np.bool_), name="flag"),
    "Int64-masked": lambda: pd.Series(
        pd.arrays.IntegerArray(
            np.array([1, 999, 3], dtype=np.int64),
            np.array([False, True, False], dtype=np.bool_),
        ),
        name="nullable",
    ),
    "Float64-masked": lambda: pd.Series(
        pd.arrays.FloatingArray(
            np.array([1.0, _NAN_ONE], dtype=np.float64),
            np.array([False, True], dtype=np.bool_),
        ),
        name="nullable_amount",
    ),
    "string-python": lambda: _string_series(("a", None, "ccc"), name="text"),
    "object-mixed": lambda: _object_series(("a", 5, 2.5, None), name="leaf"),
    "unnamed": lambda: pd.Series(np.arange(3, dtype=np.int64)),
    "strided": lambda: pd.Series(np.arange(12, dtype=np.int64), name="c")[::3],
    "reversed": lambda: pd.Series(np.arange(6, dtype=np.float64), name="c")[::-1],
}


@pytest.mark.parametrize("name", sorted(_SERIES))
def test_coverage_series_stamp_matches_the_all_true_reference(name):
    series = _SERIES[name]()
    assert coverage._series_stamp(series) == _reference_series_stamp(series)


def _series_mutations():
    yield (
        "hidden-masked-data",
        pd.Series(
            pd.arrays.IntegerArray(
                np.array([1, 999, 3], dtype=np.int64),
                np.array([False, True, False], dtype=np.bool_),
            ),
            name="nullable",
        ),
        pd.Series(
            pd.arrays.IntegerArray(
                np.array([1, 1000, 3], dtype=np.int64),
                np.array([False, True, False], dtype=np.bool_),
            ),
            name="nullable",
        ),
    )
    yield (
        "signed-zero",
        pd.Series(np.array([-0.0], dtype=np.float64), name="amount"),
        pd.Series(np.array([0.0], dtype=np.float64), name="amount"),
    )
    yield (
        "nan-payload",
        pd.Series(np.array([_NAN_ONE], dtype=np.float64), name="amount"),
        pd.Series(np.array([_NAN_TWO], dtype=np.float64), name="amount"),
    )
    yield (
        "series-name",
        pd.Series(np.array([1.0], dtype=np.float64), name="amount"),
        pd.Series(np.array([1.0], dtype=np.float64), name="other"),
    )
    yield (
        "object-leaf-type",
        _object_series((5,), name="leaf"),
        _object_series((5.0,), name="leaf"),
    )


@pytest.mark.parametrize("label,left,right", list(_series_mutations()), ids=lambda v: v)
def test_every_series_mutation_moves_the_coverage_stamp(label, left, right):
    assert coverage._series_stamp(left) != coverage._series_stamp(right)
    assert _reference_series_stamp(left) != _reference_series_stamp(right)
    assert coverage._series_stamp(left) == _reference_series_stamp(left)


def _frame(
    *, amount=(-0.0, 1.5), hidden=(1, 999), mask=(False, True), labels=("a", "b")
):
    person = pd.DataFrame(
        {
            "person_id": np.array([1, 2], dtype=np.int64),
            "person_household_id": np.array([10, 20], dtype=np.int64),
            "amount": np.array(amount, dtype=np.float64),
            "nullable": pd.arrays.IntegerArray(
                np.array(hidden, dtype=np.int64), np.array(mask, dtype=np.bool_)
            ),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.array([10, 20], dtype=np.int64),
            "rent": np.array([1.0, 2.0], dtype=np.float64),
        }
    )
    strata = _object_series(labels)
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0, 2.0]), WeightKind.DESIGN)},
        strata,
    )


def test_origin_budget_storage_identity_matches_the_all_true_reference():
    frame = _frame()
    assert budget._physical_nonweight_identity(
        frame
    ) == _reference_physical_nonweight_identity(frame)


@pytest.mark.parametrize(
    "mutated",
    [
        _frame(amount=(0.0, 1.5)),
        _frame(hidden=(1, 1000)),
        _frame(mask=(False, False)),
        _frame(labels=("a", "c")),
    ],
    ids=["visible-value", "hidden-masked-data", "null-mask", "strata-label"],
)
def test_every_frame_mutation_moves_the_origin_budget_storage_identity(mutated):
    base = _frame()
    baseline = budget._physical_nonweight_identity(base)
    assert baseline == _reference_physical_nonweight_identity(base)
    moved = budget._physical_nonweight_identity(mutated)
    assert moved == _reference_physical_nonweight_identity(mutated)
    assert moved != baseline


def _storage_parts_calls(module):
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.attr
            if isinstance(function, ast.Attribute)
            else getattr(function, "id", None)
        )
        if name == "_storage_parts":
            yield node


@pytest.mark.parametrize(
    "module", CONVERTED_MODULES, ids=lambda module: module.__name__.rsplit(".", 1)[-1]
)
def test_every_converted_site_still_selects_the_whole_series_by_slice(module):
    calls = list(_storage_parts_calls(module))
    assert calls, f"{module.__name__} no longer calls the storage helper"
    for call in calls:
        assert len(call.args) == 2, ast.unparse(call)
        assert ast.unparse(call.args[1]) == "slice(None)", ast.unparse(call)
