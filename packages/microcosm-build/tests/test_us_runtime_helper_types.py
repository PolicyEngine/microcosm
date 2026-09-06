"""The declared source-helper parameter types are documentation, not enforcement.

Each helper touched by the US runtime helper-type lane carries an annotation that
restates the concrete type its own body already validates at runtime. These tests
fix three properties the annotations must never quietly change:

* the annotation is never executed (every module declares postponed evaluation),
  so an annotation can neither import, coerce, nor validate anything;
* the runtime refusal that *proves* the declared type still fires on the wrong
  input, so enforcement did not move into the annotation; and
* no annotated value is a Frame, DataFrame or column name -- the declared types
  are stat results, one-dimensional arrays, row-count and path maps, and bytes.

No engine, no network and no genuine source file is touched here.
"""

from __future__ import annotations

import ast
import inspect
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import acs_housing_universe_source as acs_source
from microcosm.build.us_runtime import asec_current_money_selection as selection
from microcosm.build.us_runtime import asec_housing_status_source as housing_source
from microcosm.build.us_runtime import asec_income_observations as observations
from microcosm.build.us_runtime import asec_person_income_source as person_source
from microcosm.build.us_runtime import asec_student_controls as student_controls
from microcosm.build.us_runtime import graph_acs_housing_universe as acs_graph
from microcosm.build.us_runtime.asec_current_money import MoneyRefusalError
from microcosm.build.us_runtime.asec_housing_status import HousingStatusRefusalError

_MODULES = (
    acs_source,
    selection,
    housing_source,
    observations,
    person_source,
    student_controls,
    acs_graph,
)

# (module, qualified helper, parameter, exact declared annotation)
_DECLARED = (
    (acs_source, "_identity", "value", "os.stat_result"),
    (selection, "_positions", "values", "np.ndarray"),
    (selection, "_split", "counts", "dict[str, int]"),
    (selection, "_field_shapes", "counts", "dict[str, int]"),
    (housing_source, "_load", "cohort_paths", "Mapping[int, str | Path]"),
    (observations, "_validate_observations", "age", "np.ndarray"),
    (observations, "_validate_observations", "receipt_code", "np.ndarray"),
    (observations, "_validate_observations", "nominal", "np.ndarray"),
    (observations, "_reconstruct", "member_paths", "Mapping[int, str | Path]"),
    (person_source, "_reconstruct", "member_paths", "Mapping[int, str | Path]"),
    (student_controls, "_reconstruct", "member_paths", "Mapping[int, str | Path]"),
    (acs_graph, "bind_acs_housing_evidence", "payload", "bytes"),
)


def test_declared_helper_parameter_types_are_exactly_the_reviewed_set() -> None:
    """The annotations are present, exact, and nothing else was annotated."""

    for module, helper, parameter, expected in _DECLARED:
        annotation = inspect.get_annotations(getattr(module, helper))[parameter]
        assert annotation == expected, (module.__name__, helper, parameter, annotation)


def test_source_helper_annotations_are_never_executed() -> None:
    """Postponed evaluation keeps every annotation a string, so none can run."""

    for module in _MODULES:
        tree = ast.parse(Path(module.__file__).read_text())
        futures = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
            for alias in node.names
        }
        assert "annotations" in futures, module.__name__

    for module, helper, parameter, expected in _DECLARED:
        raw = getattr(module, helper).__annotations__[parameter]
        assert isinstance(raw, str) and raw == expected, (module.__name__, helper)


def test_no_declared_helper_type_is_a_frame_dataframe_or_column_name() -> None:
    """A source-spine carrier must never be declared as a benign value."""

    for module, helper, parameter, expected in _DECLARED:
        assert not any(
            marker in expected for marker in ("Frame", "DataFrame", "Series", "column")
        ), (module.__name__, helper, parameter, expected)


def test_stat_identity_still_requires_real_stat_fields() -> None:
    """``_identity`` reads stat fields; a non-stat object still fails loudly."""

    identity = acs_source._identity(os.stat(acs_source.__file__))
    assert len(identity) == 5 and all(isinstance(item, int) for item in identity)

    class NotAStatResult:
        st_dev = 1

    with pytest.raises(AttributeError):
        acs_source._identity(NotAStatResult())


def test_position_arrays_still_refuse_non_ndarray_input() -> None:
    """The declared ``np.ndarray`` restates a check that still refuses lists."""

    values = np.array([0, 2, 5], dtype=np.int64)
    assert selection._positions(values, rows=10, reason="POSITIONS") is values

    for wrong in (
        [0, 2, 5],
        np.array([0.0, 2.0], dtype=np.float64),
        np.zeros((2, 2), dtype=np.int64),
    ):
        with pytest.raises(MoneyRefusalError):
            selection._positions(wrong, rows=10, reason="POSITIONS")

    with pytest.raises(MoneyRefusalError):  # unsorted positions
        selection._positions(np.array([5, 2], dtype=np.int64), rows=10, reason="X")


def test_observation_arrays_still_refuse_non_ndarray_input() -> None:
    """``_validate_observations`` enforces int64 1-D arrays, not the annotation."""

    age = np.array([20, 30], dtype=np.int64)
    receipt = np.array([1, 0], dtype=np.int64)
    nominal = np.array([500, 0], dtype=np.int64)
    observations._validate_observations(age, receipt, nominal)

    with pytest.raises(MoneyRefusalError):
        observations._validate_observations([20, 30], receipt, nominal)
    with pytest.raises(MoneyRefusalError):
        observations._validate_observations(age.astype(np.float64), receipt, nominal)


def test_member_path_maps_still_refuse_non_mapping_and_wrong_key_types() -> None:
    """``_paths`` -- the shared validator behind three helpers -- still refuses."""

    person_source._paths({2022: "a", 2023: Path("b"), 2024: "c"})

    for wrong in (
        [(2022, "a"), (2023, "b"), (2024, "c")],  # not a Mapping
        {"2022": "a", "2023": "b", "2024": "c"},  # string keys
        {2022: "a", 2023: "b"},  # incomplete year set
        {2022: 1, 2023: 2, 2024: 3},  # non str/Path values
    ):
        with pytest.raises(MoneyRefusalError):
            person_source._paths(wrong)


def test_declared_path_map_type_matches_the_shared_runtime_validator() -> None:
    """The declared ``Mapping[int, str | Path]`` is what ``_paths`` enforces."""

    accepted = {2022: "a", 2023: Path("b"), 2024: "c"}
    assert isinstance(accepted, Mapping)
    assert all(isinstance(year, int) for year in accepted)
    assert all(isinstance(value, (str, Path)) for value in accepted.values())
    person_source._paths(accepted)

    for module in (observations, student_controls):
        assert module.restoration._paths is person_source._paths


def test_graph_payload_still_refuses_a_non_bytes_transport() -> None:
    """``bind_acs_housing_evidence`` refuses str/bytearray despite the annotation."""

    for wrong in ("x" * 200, bytearray(b"x" * 200), memoryview(b"x" * 200), None):
        with pytest.raises(acs_source.ACSHousingSourceError) as raised:
            acs_graph.bind_acs_housing_evidence(
                wrong, prepared_receipt=b"{}", frame_context=b"{}"
            )
        assert "GRAPH_ACS_PAYLOAD_SIZE" in str(raised.value)


def test_entity_count_maps_are_row_counts_rather_than_column_names() -> None:
    """``counts`` is keyed by entity and valued by row count, never by column."""

    assert selection._ENTITIES == ("person", "household")
    counts = {"person": 3, "household": 2}
    assert all(isinstance(key, str) for key in counts)
    assert all(isinstance(value, int) for value in counts.values())

    # Buffers sized for three persons, a count map declaring two: the widths
    # check reads counts[entity] as a row count and refuses the mismatch.
    field = selection.MoneyField(
        selection.FIELDS[0], b"\0" * 24, b"\0" * 3, b"\0" * 3, b"\0" * 3
    )
    with pytest.raises(MoneyRefusalError) as raised:
        selection._field_shapes((field,), ("person",), {"person": 2})
    assert "FIELD_BUFFER_SHAPE" in str(raised.value)

    # A roster/entity length mismatch is refused by zip(strict=True), unchanged.
    with pytest.raises(ValueError):
        selection._field_shapes((), ("person",), counts)


def test_housing_status_load_still_refuses_an_unauthenticated_source() -> None:
    """``_load``'s first refusal is unchanged; the ``source`` beside the annotated
    ``cohort_paths`` is Frame-bearing and was deliberately left unannotated."""

    paths = {2022: "a", 2023: Path("b"), 2024: "c"}
    for wrong_source in (object(), None, "source"):
        with pytest.raises(HousingStatusRefusalError) as raised:
            housing_source._load(wrong_source, paths)
        assert "AUTHENTICATED_SOURCE" in str(raised.value)

    signature = inspect.signature(housing_source._load)
    assert signature.parameters["source"].annotation is inspect.Parameter.empty
    assert signature.parameters["cohort_paths"].annotation == "Mapping[int, str | Path]"
