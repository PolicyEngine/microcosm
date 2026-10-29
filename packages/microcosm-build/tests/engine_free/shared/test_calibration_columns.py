"""Simple prepared columns use the declared domains and retain multiplicities."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.calibration_columns import prepare_calibration_columns
from microcosm.build.country_spec import _validate_target_references
from microcosm.build.reference_profiles import resolve_reference_profiles
from microcosm.calibrate import Target, TargetSet
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.concepts import TENURE_DOMAIN
from test_support.paths import paths_for


def _bindings():
    return json.loads(
        (
            paths_for("microcosm-build").package
            / "src/microcosm/build/nz/calibration_bindings.json"
        ).read_text()
    )


def _tables():
    return {
        "person": pd.DataFrame(
            {
                "person_id": range(8),
                "person_household_id": [0, 1, 2, 3, 4, 5, 0, 1],
                "age": [0, 14, 15, 39, 40, 64, 65, 120],
                "region_code": ["01"] * 6 + ["99"] * 2,
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": range(6),
                "household_region_code": ["01"] * 6,
                "tenure": list(TENURE_DOMAIN),
            }
        ),
    }


def _frame(tables):
    return Frame(
        tables,
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(len(tables["household"])), WeightKind.DESIGN)},
    )


def _replace_people(tables, ages):
    tables["household"] = tables["household"].iloc[:1].copy()
    tables["person"] = pd.DataFrame(
        {
            "person_id": range(len(ages)),
            "person_household_id": [0] * len(ages),
            "age": ages,
            "region_code": ["01"] * len(ages),
        }
    )


def test_prepared_columns_partition_age_and_canonical_tenure_on_a_real_frame():
    tables = _tables()
    prepared = prepare_calibration_columns(tables, _bindings())
    frame = _frame(prepared)
    assert prepared["person"]["people"].tolist() == [1] * 8
    assert prepared["household"]["households"].tolist() == [1] * 6
    ages = [
        name for name in prepared["person"] if name.startswith("population_region_")
    ]
    assert len(ages) == 64
    np.testing.assert_array_equal(
        prepared["person"][ages].sum(axis=1), [1] * 6 + [0, 0]
    )
    expected = {"0_14": [0, 1], "15_39": [2, 3], "40_64": [4, 5]}
    for band, positions in expected.items():
        assert (
            np.flatnonzero(
                prepared["person"][f"population_region_01_age_{band}_indicator"]
            ).tolist()
            == positions
        )
    owned = prepared["household"]["household_region_01_owned_indicator"]
    other = prepared["household"]["household_region_01_not_owned_indicator"]
    assert owned.tolist() == [True, True, False, False, False, False]
    np.testing.assert_array_equal(owned.astype(int) + other.astype(int), np.ones(6))
    assert frame.n("person") == 8
    assert "people" not in tables["person"]
    assert not any(
        column.startswith("receives_")
        for table in prepared.values()
        for column in table
    )


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=200), min_size=1, max_size=30))
def test_nonnegative_integer_ages_have_exactly_one_declared_band(ages):
    tables = _tables()
    _replace_people(tables, ages)
    person = prepare_calibration_columns(tables, _bindings())["person"]
    indicators = [
        f"population_region_01_age_{band}_indicator"
        for band in ("0_14", "15_39", "40_64", "65_plus")
    ]
    np.testing.assert_array_equal(person[indicators].sum(axis=1), np.ones(len(ages)))


@pytest.mark.parametrize(
    ("entity", "column", "value"),
    [
        ("person", "age", -1),
        ("person", "age", 1.5),
        ("person", "age", float("nan")),
        ("person", "age", float("inf")),
        ("person", "age", True),
        ("person", "age", "18"),
        ("person", "region_code", None),
        ("person", "region_code", "999"),
        ("household", "household_region_code", "19"),
        ("household", "tenure", None),
        ("household", "tenure", "unknown"),
    ],
)
def test_missing_or_unknown_age_region_and_tenure_refuse(entity, column, value):
    tables = _tables()
    tables[entity][column] = tables[entity][column].astype(object)
    tables[entity].loc[0, column] = value
    with pytest.raises(ValueError, match="Calibration input"):
        prepare_calibration_columns(tables, _bindings())


@pytest.mark.parametrize(
    ("entity", "column"),
    [
        ("person", "age"),
        ("person", "region_code"),
        ("household", "tenure"),
        ("household", "household_region_code"),
    ],
)
def test_missing_required_columns_refuse(entity, column):
    tables = _tables()
    tables[entity] = tables[entity].drop(columns=column)
    with pytest.raises(ValueError, match="preparation needs"):
        prepare_calibration_columns(tables, _bindings())


def test_stale_indicators_are_recomputed_after_inputs_change():
    tables = prepare_calibration_columns(_tables(), _bindings())
    tables["person"].loc[0, "age"] = 65
    fresh = prepare_calibration_columns(tables, _bindings())
    assert not fresh["person"].loc[0, "population_region_01_age_0_14_indicator"]
    assert fresh["person"].loc[0, "population_region_01_age_65_plus_indicator"]


def test_receipt_age_indicators_retain_the_published_strata_boundaries():
    tables = _tables()
    ages = [17, 18, 24, 25, 39, 40, 54, 55, 64, 65]
    _replace_people(tables, ages)
    person = prepare_calibration_columns(tables, _bindings())["person"]
    for code, positions in (
        ("18_24", [1, 2]),
        ("25_39", [3, 4]),
        ("40_54", [5, 6]),
        ("55_64", [7, 8]),
    ):
        assert np.flatnonzero(person[f"age_{code}_indicator"]).tolist() == positions


def test_all_active_calibration_bindings_enter_the_small_frame_matrix():
    frame = _frame(prepare_calibration_columns(_tables(), _bindings()))
    root = paths_for("microcosm-build").package / "src/microcosm/build/nz"
    profiles = json.loads((root / "reference_profiles.json").read_text())
    effective, _, _ = resolve_reference_profiles(
        json.loads((root / "target_references.json").read_text()),
        profiles,
        {},
        factor_context=json.loads(
            (root / profiles["factor_context_resource"]).read_text()
        ),
    )
    references = [
        ref
        for ref in _validate_target_references(effective, country="nz")
        if ref.metadata["activation_status"] == "active"
    ]
    # Synthetic positive right-hand sides isolate column readiness; the public
    # artifact audit independently checks every observed right-hand side.
    targets = TargetSet(
        Target(
            name=ref.name,
            entity=ref.entity,
            measure=ref.measure,
            filter=ref.filter,
            period=ref.period,
            value=1,
        )
        for ref in references
    )
    problem = build_constraint_matrix(frame, targets, "household")
    assert not problem.skipped
    assert len(problem.names) == len(references) == 234


def test_household_size_indicators_partition_all_regions_and_include_nine_people():
    bindings = _bindings()
    regions = bindings["input_domains"]["household"]["household_region_code"]["values"]
    sizes = list(range(1, 10)) * len(regions)
    household_regions = [region for region in regions for _ in range(9)]
    memberships = [index for index, size in enumerate(sizes) for _ in range(size)]
    tables = {
        "household": pd.DataFrame(
            {
                "household_id": range(len(sizes)),
                "household_region_code": household_regions,
                "tenure": [TENURE_DOMAIN[0]] * len(sizes),
            }
        ),
        "person": pd.DataFrame(
            {
                "person_id": range(len(memberships)),
                "person_household_id": memberships,
                "age": [30] * len(memberships),
                "region_code": [household_regions[index] for index in memberships],
            }
        ),
    }
    prepared = prepare_calibration_columns(tables, bindings)["household"]
    columns = [row["name"] for row in bindings["columns"] if "_size_" in row["name"]]
    assert len(columns) == 136
    assert prepared["household_size"].tolist() == sizes
    np.testing.assert_array_equal(prepared[columns].sum(axis=1), np.ones(len(sizes)))
    for region in regions:
        eight_plus = prepared[f"household_region_{region}_size_8_indicator"]
        assert eight_plus.sum() == 2


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_household",
        "duplicate_person",
        "duplicate_boxed_person",
        "orphan",
        "empty",
        "missing",
        "missing_person",
        "missing_household",
        "blank",
        "wrong_type",
    ],
)
def test_household_roster_refuses_invalid_membership(defect):
    tables = _tables()
    if defect == "duplicate_household":
        tables["household"].loc[1, "household_id"] = 0
    elif defect == "duplicate_person":
        tables["person"].loc[1, "person_id"] = 0
    elif defect == "duplicate_boxed_person":
        tables["person"]["person_id"] = tables["person"]["person_id"].astype(object)
        tables["person"].loc[1, "person_id"] = np.int64(0)
    elif defect == "orphan":
        tables["person"].loc[0, "person_household_id"] = 999
    elif defect == "empty":
        tables["person"] = tables["person"].loc[
            lambda rows: rows.person_household_id != 5
        ]
    elif defect in {"missing", "missing_person", "missing_household", "blank"}:
        entity, column = {
            "missing": ("person", "person_household_id"),
            "missing_person": ("person", "person_id"),
            "missing_household": ("household", "household_id"),
            "blank": ("person", "person_household_id"),
        }[defect]
        tables[entity][column] = tables[entity][column].astype(object)
        tables[entity].loc[0, column] = "" if defect == "blank" else None
    else:
        tables["person"]["person_household_id"] = tables["person"][
            "person_household_id"
        ].astype(object)
        tables["person"].loc[0, "person_household_id"] = "0"
    with pytest.raises(ValueError, match="missing values|identit|orphan|empty"):
        prepare_calibration_columns(tables, _bindings())


def test_prepared_columns_cannot_overwrite_a_validated_roster_count():
    bindings = _bindings()
    bindings["columns"].append(
        {"entity": "household", "name": "household_size", "constant": 8}
    )
    with pytest.raises(ValueError, match="repeated or structural"):
        prepare_calibration_columns(_tables(), bindings)


def test_family_multiplicity_is_a_measure_while_nucleus_activation_stays_blocked():
    tables = prepare_calibration_columns(_tables(), _bindings())
    tables["household"]["census_families_type_1"] = [2, 0, 0, 0, 0, 0]
    frame = _frame(tables)
    target = Target(
        name="two nuclei",
        entity="household",
        measure="census_families_type_1",
        filter="household_region_01_owned_indicator",
    )
    assert target.constraint_row(frame).tolist() == [2, 0, 0, 0, 0, 0]
    prepared = prepare_calibration_columns(_tables(), _bindings())
    assert "census_families_type_1" not in prepared["household"]


@pytest.mark.parametrize(
    "defect",
    ["operator", "unvalidated", "duplicate", "structural", "empty", "nonfinite"],
)
def test_malformed_preparation_contracts_refuse(defect):
    bindings = copy.deepcopy(_bindings())
    row = next(row for row in bindings["columns"] if "predicates" in row)
    if defect == "operator":
        row["predicates"][0]["operator"] = "guess"
    elif defect == "unvalidated":
        row["predicates"][0]["column"] = "missing"
    elif defect == "duplicate":
        bindings["columns"].append(copy.deepcopy(row))
    elif defect == "structural":
        row["name"] = "person_id"
    elif defect == "empty":
        row["predicates"] = []
    else:
        bindings["columns"][0]["constant"] = float("nan")
    with pytest.raises(ValueError):
        prepare_calibration_columns(_tables(), bindings)
