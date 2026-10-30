"""Simple prepared columns use the declared domains and retain multiplicities."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.calibration_columns import prepare_calibration_columns
from microcosm.build.calibration_predicates import calibration_predicate_mask
from microcosm.build.country_spec import _validate_target_references
from microcosm.build.reference_profiles import resolve_reference_profiles
from microcosm.calibrate import Target, TargetSet
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.concepts import TENURE_DOMAIN
from test_support.microcosm_build.reference_intervals import (
    interval_bindings,
    interval_table_fixture,
)
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
        json.loads((root / profiles["interval_resource"]).read_text()),
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


# ---------------------------------------------------------------------------
# Interval outputs: their own binding kind beside the predicate outputs
# ---------------------------------------------------------------------------


def _interval_contract(entity="person"):
    table = interval_table_fixture(entity=entity, source_period=2002)
    bindings = interval_bindings(table)
    bindings["columns"].append({"entity": entity, "name": "counted", "constant": 1})
    return {"schema_version": 1, "tables": [table]}, bindings


def _interval_tables(amounts, statuses, entity="person"):
    return {
        entity: pd.DataFrame(
            {
                f"{entity}_id": range(len(amounts)),
                "toy_income_cents": amounts,
                "toy_income_universe": statuses,
            }
        )
    }


def test_interval_outputs_follow_the_universe_and_recompute_stale_values():
    intervals, bindings = _interval_contract()
    tables = _interval_tables(
        [-50, 0, 0, 100_000, 100_001, 5_000_000],
        ["included", "included", "excluded", "included", "included", "excluded"],
    )
    tables["person"]["toy_income_band_nil"] = [True] * 6
    prepared = prepare_calibration_columns(tables, bindings, intervals=intervals)
    person = prepared["person"]
    # A nonfiler with no income is excluded: never an all-person nil indicator.
    assert person["toy_income_band_nil"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert person["toy_income_band_b0"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert person["toy_income_band_b1"].tolist() == [False] * 4 + [True, False]
    assert person["toy_income_band_b2"].tolist() == [False] * 6
    assert person["toy_income_band_b0"].dtype == bool
    assert person["counted"].tolist() == [1] * 6
    assert tables["person"]["toy_income_band_nil"].tolist() == [True] * 6
    tables["person"].loc[3, "toy_income_cents"] = 100_001
    fresh = prepare_calibration_columns(tables, bindings, intervals=intervals)
    assert fresh["person"].loc[3, "toy_income_band_b1"]
    assert not fresh["person"].loc[3, "toy_income_band_b0"]


@pytest.mark.parametrize(
    ("amount", "status"),
    [
        (10, "unresolved"),
        (10, "unknown"),
        (True, "included"),
        (10.0, "included"),
        ("10", "included"),
        (pd.NA, "included"),
    ],
)
def test_unresolved_universe_or_invalid_amount_refuses_before_any_output(
    amount, status
):
    intervals, bindings = _interval_contract()
    tables = _interval_tables(pd.array([5, amount], dtype=object), ["included", status])
    before = copy.deepcopy(tables)
    with pytest.raises(ValueError, match="refused before emitting indicators"):
        prepare_calibration_columns(tables, bindings, intervals=intervals)
    pd.testing.assert_frame_equal(tables["person"], before["person"])


def test_household_entity_interval_outputs_are_entity_generic():
    intervals, bindings = _interval_contract(entity="household")
    prepared = prepare_calibration_columns(
        _interval_tables([0, 100_000], ["excluded", "included"], entity="household"),
        bindings,
        intervals=intervals,
    )
    assert prepared["household"]["toy_income_band_b0"].tolist() == [False, True]
    assert prepared["household"]["toy_income_band_nil"].tolist() == [False, False]


@pytest.mark.parametrize(
    "defect",
    [
        "no_document",
        "unknown_table",
        "unknown_band",
        "wrong_entity",
        "not_the_band_indicator",
        "mixed_with_predicates",
        "mixed_with_constant",
        "extra_interval_field",
        "unaged_source_cuts",
        "missing_universe_column",
        "missing_income_column",
        "invalid_document",
    ],
)
def test_malformed_interval_outputs_refuse(defect):
    intervals, bindings = _interval_contract()
    entry = bindings["columns"][1]
    tables = _interval_tables([1, 2], ["included", "included"])
    if defect == "no_document":
        intervals = None
    elif defect == "unknown_table":
        entry["interval"]["table"] = "other"
    elif defect == "unknown_band":
        entry["interval"]["band"] = "b9"
    elif defect == "wrong_entity":
        bindings["required_inputs"]["household"] = ["household_id"]
        bindings["input_domains"]["household"] = {"household_id": {"kind": "identity"}}
        tables["household"] = pd.DataFrame({"household_id": [0]})
        entry["entity"] = "household"
    elif defect == "not_the_band_indicator":
        entry["name"] = "renamed_indicator"
    elif defect == "mixed_with_predicates":
        entry["predicates"] = [{"column": "person_id", "operator": "ge", "value": 0}]
    elif defect == "mixed_with_constant":
        entry["constant"] = 1
    elif defect == "extra_interval_field":
        entry["interval"]["closure"] = "left"
    elif defect == "unaged_source_cuts":
        intervals["tables"][0]["source_period"] = 2001
    elif defect == "missing_universe_column":
        tables["person"] = tables["person"].drop(columns="toy_income_universe")
    elif defect == "missing_income_column":
        tables["person"] = tables["person"].drop(columns="toy_income_cents")
    else:
        intervals["tables"][0].pop("positive_band_closure")
    with pytest.raises(ValueError):
        prepare_calibration_columns(tables, bindings, intervals=intervals)


def test_interval_outputs_enter_a_constraint_row_through_their_filter():
    intervals, bindings = _interval_contract()
    tables = _interval_tables([0, 50_000, 150_000], ["included"] * 3)
    tables["household"] = pd.DataFrame({"household_id": [0, 1, 2]})
    tables["person"]["person_household_id"] = [0, 1, 2]
    prepared = prepare_calibration_columns(tables, bindings, intervals=intervals)
    frame = _frame(prepared)
    target = Target(
        name="first band",
        entity="person",
        measure="toy_income_cents",
        filter="toy_income_band_b0",
    )
    assert target.constraint_row(frame).tolist() == [0, 50_000, 0]


def test_signed_integer_domain_admits_losses_and_refuses_booleans_and_floats():
    bindings = {
        "schema_version": 1,
        "required_inputs": {"person": ["person_id", "net_cents"]},
        "input_domains": {
            "person": {
                "person_id": {"kind": "identity"},
                "net_cents": {"kind": "signed_integer"},
            }
        },
        "columns": [
            {
                "entity": "person",
                "name": "loss_indicator",
                "predicates": [{"column": "net_cents", "operator": "lt", "value": 0}],
            }
        ],
    }
    tables = {"person": pd.DataFrame({"person_id": [0, 1], "net_cents": [-5, 7]})}
    prepared = prepare_calibration_columns(tables, bindings)
    assert prepared["person"]["loss_indicator"].tolist() == [True, False]
    for values in ([True, False], [-5.0, 7.0], pd.array([-5, None], dtype="Int64")):
        tables["person"]["net_cents"] = values
        with pytest.raises(ValueError):
            prepare_calibration_columns(tables, bindings)


def _shared_predicate_bindings():
    return {
        "schema_version": 1,
        "required_inputs": {"person": ["age", "category"]},
        "input_domains": {
            "person": {
                "age": {"kind": "nonnegative_integer"},
                "category": {"values": ["a", "b", "c"]},
            }
        },
        "columns": [
            {
                "entity": "person",
                "name": "young_a",
                "predicates": [
                    {"column": "age", "operator": "ge", "value": 18},
                    {"column": "age", "operator": "lt", "value": 25},
                    {"column": "category", "operator": "eq", "value": "a"},
                ],
            },
            {
                "entity": "person",
                "name": "older_ab",
                "predicates": [
                    {"column": "age", "operator": "ge", "value": 25},
                    {"column": "category", "operator": "in", "value": ["a", "b"]},
                ],
            },
        ],
    }


def _shared_predicate_tables():
    return {
        "person": pd.DataFrame(
            {"age": [17, 18, 24, 25, 64], "category": ["a", "a", "b", "b", "c"]}
        )
    }


def test_receipt_predicate_uses_declared_filter_and_ignores_stale_indicators():
    tables, bindings = _shared_predicate_tables(), _shared_predicate_bindings()
    tables["person"]["young_a"] = True
    young = calibration_predicate_mask(
        tables, bindings, entity="person", name="young_a"
    )
    older = calibration_predicate_mask(
        tables, bindings, entity="person", name="older_ab"
    )
    assert young.tolist() == [False, True, False, False, False]
    assert older.tolist() == [False, False, False, True, False]
    assert tables["person"]["young_a"].all()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("age", None, "contains missing values"),
        ("age", -1, "outside its declared domain"),
        ("age", 18.5, "outside its declared domain"),
        ("age", True, "outside its declared domain"),
        ("age", "18", "outside its declared domain"),
        ("category", None, "contains missing values"),
        ("category", "unknown", "outside its declared domain"),
    ],
)
def test_receipt_predicate_validates_inputs_before_mask(column, value, message):
    tables, bindings = _shared_predicate_tables(), _shared_predicate_bindings()
    tables["person"][column] = tables["person"][column].astype(object)
    tables["person"].loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        calibration_predicate_mask(tables, bindings, entity="person", name="young_a")


def test_receipt_predicate_validates_even_required_inputs_absent_from_its_mask():
    tables, bindings = _shared_predicate_tables(), _shared_predicate_bindings()
    bindings["columns"][0]["predicates"] = [
        {"column": "age", "operator": "ge", "value": 18}
    ]
    tables["person"].loc[0, "category"] = "unknown"
    with pytest.raises(ValueError, match="Calibration input person.category"):
        calibration_predicate_mask(tables, bindings, entity="person", name="young_a")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("schema", "require schema_version 1"),
        ("required", "need required_inputs and input_domains"),
        ("domains", "need required_inputs and input_domains"),
        ("columns", "need nonempty columns"),
        ("empty_required", "required_inputs.*invalid"),
        ("missing_domain", "needs a domain"),
        ("unsupported_domain", "unsupported domain"),
        ("missing_column", "preparation needs person.age"),
        ("missing_table", "needs the 'person' table"),
    ],
)
def test_receipt_predicate_validates_complete_binding_inputs(change, message):
    tables, bindings = _shared_predicate_tables(), _shared_predicate_bindings()
    if change == "schema":
        bindings["schema_version"] = 2
    elif change in {"required", "domains"}:
        bindings.pop("required_inputs" if change == "required" else "input_domains")
    elif change == "columns":
        bindings["columns"] = []
    elif change == "empty_required":
        bindings["required_inputs"]["person"] = []
    elif change == "missing_domain":
        bindings["input_domains"]["person"].pop("age")
    elif change == "unsupported_domain":
        bindings["input_domains"]["person"]["age"] = {"kind": "unsupported"}
    elif change == "missing_column":
        tables["person"] = tables["person"].drop(columns="age")
    elif change == "missing_table":
        tables = {}
    with pytest.raises(ValueError, match=message):
        calibration_predicate_mask(tables, bindings, entity="person", name="young_a")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "exactly one declared predicate column"),
        ("repeated", "exactly one declared predicate column"),
        ("constant", "must use predicates"),
        ("interval", "must use predicates"),
        ("empty", "needs predicates"),
        ("not_object", "predicates must be objects"),
        ("unvalidated", "needs a validated person input"),
        ("missing_value", "needs a validated person input"),
        ("operator", "Unsupported calibration predicate"),
        ("empty_in", "Unsupported calibration predicate"),
    ],
)
def test_receipt_predicate_refuses_undeclared_or_invalid_filter(change, message):
    tables, bindings = _shared_predicate_tables(), _shared_predicate_bindings()
    entry = bindings["columns"][0]
    if change == "missing":
        bindings["columns"].pop(0)
    elif change == "repeated":
        bindings["columns"].append(copy.deepcopy(entry))
    elif change in {"constant", "interval"}:
        entry[change] = 1
    elif change == "empty":
        entry["predicates"] = []
    elif change == "not_object":
        entry["predicates"] = [None]
    elif change == "unvalidated":
        tables["person"]["undeclared"] = 18
        entry["predicates"][0]["column"] = "undeclared"
    elif change == "missing_value":
        entry["predicates"][0].pop("value")
    elif change == "operator":
        entry["predicates"][0]["operator"] = "unsupported"
    elif change == "empty_in":
        entry["predicates"][0].update(operator="in", value=[])
    with pytest.raises(ValueError, match=message):
        calibration_predicate_mask(tables, bindings, entity="person", name="young_a")


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=120), st.sampled_from(["a", "b", "c"])
        ),
        min_size=1,
        max_size=30,
    )
)
def test_receipt_predicate_masks_match_preparation_for_validated_raw_inputs(rows):
    tables = {"person": pd.DataFrame(rows, columns=["age", "category"])}
    bindings = _shared_predicate_bindings()
    prepared = prepare_calibration_columns(tables, bindings)["person"]
    for name in ("young_a", "older_ab"):
        np.testing.assert_array_equal(
            calibration_predicate_mask(tables, bindings, entity="person", name=name),
            prepared[name].to_numpy(),
        )


def test_prepare_matches_pinned_legacy_output_digest():
    # Captured from the pre-refactor implementation at 4f333f2c2ee70b6b758586f402d966df7297a2f0,
    # run on these NZ bindings, which #1398 extended after the capture.
    tables = prepare_calibration_columns(_tables(), _bindings())
    document = {
        entity: {
            "columns": list(table),
            "dtypes": [str(dtype) for dtype in table.dtypes],
            "index": table.index.tolist(),
            "values": table.to_dict(orient="list"),
        }
        for entity, table in tables.items()
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == (
        "1b22f4e11e6e6c901866010d709acadaaad638e0774e80751e0ff23257a2bde7"
    )
