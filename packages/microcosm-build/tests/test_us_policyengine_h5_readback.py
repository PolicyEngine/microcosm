"""Invented logical comparisons and mocked I/O lifecycle, with no country engine.

The tiny writer below creates ordinary bytes, not HDF. These tests do not
replace the separately recorded real maintained-adapter codec probe.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from microcosm.build.spm_input_contract import ROLE_INPUT, UNIVERSE_INPUT
from microcosm.build.us_runtime import policyengine_h5_readback as readback
from microcosm.build.us_runtime.common_frame_export_contract import (
    RetainedFrameExportError,
)
from microcosm.frame import Frame, MassChange, WeightKind, Weights
from microcosm.frame.units import US_SCHEMA


def _strings(values):
    return pd.Series(values, dtype=pd.StringDtype(storage="python", na_value=pd.NA))


def _case(view="full", *, role_dtype="boolean"):
    # Deliberately nonmonotonic person IDs, beyond float64's exact integer range.
    person_ids = np.array([6, 1, 5, 2, 4, 3], dtype=np.int64) + 2**53
    memberships = {
        "household": [10, 10, 20, 20, 30, 30],
        "tax_unit": [100, 100, 200, 201, 300, 300],
        "spm_unit": [100, 100, 200, 201, 300, 300],
        "family": [11, 11, 21, 21, 31, 31],
        "marital_unit": [12, 12, 22, 23, 32, 32],
    }
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": person_ids,
                **{
                    f"person_{entity}_id": np.array(ids, dtype=np.int64)
                    for entity, ids in memberships.items()
                },
                "employment_income_before_lsr": np.array(
                    [-0.0, np.nan, 3.25, 4.5, 0.0, 9.0], dtype=np.float64
                ),
                ROLE_INPUT: pd.Series(
                    [True, False, False, True, False, True], dtype=role_dtype
                ),
            }
        ),
        **{
            entity: pd.DataFrame(
                {f"{entity}_id": np.unique(np.array(ids, dtype=np.int64))}
            )
            for entity, ids in memberships.items()
        },
    }
    tables["household"]["county_fips"] = _strings(["36061", "36061", "06037"])
    tables["spm_unit"][UNIVERSE_INPUT] = _strings(
        ["INCLUDED", "OUTSIDE", "INCLUDED", "UNRESOLVED"]
    )
    tables["spm_unit"]["takes_up_snap_if_eligible"] = np.array(
        [True, False, False, True], dtype=np.bool_
    )
    parent = Frame(
        {entity: tables[entity] for entity in US_SCHEMA.entities},
        US_SCHEMA,
        {"household": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.IMPORTANCE)},
        strata=_strings(["invented-a"] * 2 + ["invented-b"] * 4),
        metadata={"basis": "invented supplied parent; no source authority"},
    )
    calibrated = np.array([0.0, 9.0, 10.0], dtype=np.float64)
    weighted = parent.with_weights(
        "household",
        Weights(calibrated, WeightKind.CALIBRATED),
        mass=MassChange(factor=None, reason="Invented comparison weights; no solve"),
    )
    keep = {
        "full": [True] * 6,
        "pruned": [False, False, True, True, True, True],
        "local": [False, False, False, False, True, True],
    }[view]
    candidate = weighted.select(np.array(keep, dtype=np.bool_))
    arguments = {
        "period": 2024,
        "parent_reference": "invented projected parent, not a graph credential",
        "ordered_household_ids": np.array([10, 20, 30], dtype=np.int64),
        "calibrated_weights": calibrated.copy(),
        "calibration_specification": b'{"basis":"invented","year":2024}',
        "prune_zero_weight": view != "full",
    }
    if view == "local":
        arguments["scope_household_ids"] = np.array([30], dtype=np.int64)
    return parent, candidate, arguments


def _logical_tables(candidate):
    tables = {e: candidate.table(e).copy(deep=True) for e in candidate.entities}
    # This one conversion is the observed complete-BooleanDtype codec boundary.
    tables["person"][ROLE_INPUT] = tables["person"][ROLE_INPUT].to_numpy(dtype=np.bool_)
    tables["household"]["household_weight"] = candidate.weights_for(
        "household"
    ).values.copy()
    return tables


def _period():
    return pd.Series([2024], dtype=np.int64)


@pytest.mark.parametrize("view", ["full", "pruned", "local"])
@pytest.mark.parametrize("role_dtype", ["boolean", "bool"])
def test_complete_logical_readback_preserves_selected_six_entity_frame(
    view, role_dtype
):
    _, candidate, _ = _case(view, role_dtype=role_dtype)
    before = {e: candidate.table(e).copy(deep=True) for e in candidate.entities}
    actual, normalizations = readback.verify_policyengine_h5_readback(
        candidate, _logical_tables(candidate), _period(), period=2024
    )
    assert actual is not candidate and actual.schema == US_SCHEMA
    assert actual.entities == candidate.entities and len(actual.entities) == 6
    assert isinstance(normalizations, tuple)
    if role_dtype == "boolean":
        assert len(normalizations) == 1 and ROLE_INPUT in normalizations[0]
    else:
        assert normalizations == ()
    for entity in candidate.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity), before[entity], check_exact=True
        )
        pd.testing.assert_frame_equal(candidate.table(entity), before[entity])
    assert actual.person.person_id.to_numpy().tobytes() == (
        candidate.person.person_id.to_numpy().tobytes()
    )
    assert actual.weights_for("household").kind is WeightKind.CALIBRATED
    assert actual.weights_for("household").values.tobytes() == (
        candidate.weights_for("household").values.tobytes()
    )
    assert "household_weight" not in actual.table("household")
    assert actual.metadata == candidate.metadata
    pd.testing.assert_series_equal(actual.strata, candidate.strata)
    if view == "full":
        assert np.signbit(actual.person.employment_income_before_lsr.iloc[0])
        assert actual.table("spm_unit")[UNIVERSE_INPUT].tolist() == [
            "INCLUDED",
            "OUTSIDE",
            "INCLUDED",
            "UNRESOLVED",
        ]


@pytest.mark.parametrize(
    "change",
    [
        "missing_person",
        "row_order",
        "id_value",
        "float_id",
        "id_width",
        "membership",
        "orphan",
        "unreferenced_group",
        "value",
        "mask",
        "float_width",
        "zero_sign",
        "boolean_value",
        "numeric_boolean",
        "unknown_boolean",
        "status",
        "weight",
        "weight_width",
        "missing_column",
        "extra_column",
        "column_order",
        "missing_table",
        "extra_table",
        "object_string",
        "string_policy",
    ],
)
def test_readback_corruption_cannot_be_hidden_by_codec_restoration(change):
    _, candidate, _ = _case()
    tables = _logical_tables(candidate)
    person = tables["person"]
    money = "employment_income_before_lsr"
    if change == "missing_person":
        tables["person"] = person.iloc[1:].copy()
    elif change == "row_order":
        tables["person"] = person.iloc[::-1].reset_index(drop=True)
    elif change == "id_value":
        person.loc[0, "person_id"] += 100
    elif change == "float_id":
        person["person_id"] = person.person_id.astype(np.float64)
    elif change == "id_width":
        tables["household"]["household_id"] = tables["household"].household_id.astype(
            np.int32
        )
    elif change == "membership":
        person.loc[0, "person_household_id"] = 20
    elif change == "orphan":
        person.loc[0, "person_family_id"] = 999
    elif change == "unreferenced_group":
        tables["family"].loc[len(tables["family"])] = [999]
    elif change == "value":
        person.loc[2, money] += 1
    elif change == "mask":
        person.loc[1, money] = 0.0
    elif change == "float_width":
        person[money] = person[money].astype(np.float32)
    elif change == "zero_sign":
        person.loc[0, money] = 0.0
    elif change == "boolean_value":
        person.loc[0, ROLE_INPUT] = False
    elif change == "numeric_boolean":
        person[ROLE_INPUT] = person[ROLE_INPUT].astype(np.int8)
    elif change == "unknown_boolean":
        person[ROLE_INPUT] = person[ROLE_INPUT].astype("boolean")
        person.loc[0, ROLE_INPUT] = pd.NA
    elif change == "status":
        tables["spm_unit"].loc[0, UNIVERSE_INPUT] = "OUTSIDE"
    elif change == "weight":
        tables["household"].loc[1, "household_weight"] += 1.0
    elif change == "weight_width":
        tables["household"]["household_weight"] = tables["household"][
            "household_weight"
        ].astype(np.float32)
    elif change == "missing_column":
        person.drop(columns=money, inplace=True)
    elif change == "extra_column":
        person["default_created_input"] = True
    elif change == "column_order":
        tables["person"] = person.loc[:, list(reversed(person.columns))]
    elif change == "missing_table":
        del tables["marital_unit"]
    elif change == "extra_table":
        tables["unexpected"] = pd.DataFrame({"value": [1]})
    elif change == "object_string":
        tables["spm_unit"][UNIVERSE_INPUT] = tables["spm_unit"][UNIVERSE_INPUT].astype(
            object
        )
    else:
        tables["spm_unit"][UNIVERSE_INPUT] = tables["spm_unit"][UNIVERSE_INPUT].astype(
            pd.StringDtype(storage="python", na_value=np.nan)
        )
    with pytest.raises(readback.PolicyEngineH5ReadbackError):
        readback.verify_policyengine_h5_readback(
            candidate, tables, _period(), period=2024
        )


def test_only_nan_payload_under_unchanged_null_mask_may_normalize():
    _, candidate, _ = _case()
    tables = _logical_tables(candidate)
    name = "employment_income_before_lsr"
    payload = np.array([0x7FF8000000001234], dtype=np.uint64).view(np.float64)[0]
    tables["person"].loc[1, name] = payload
    actual, _ = readback.verify_policyengine_h5_readback(
        candidate, tables, _period(), period=2024
    )
    expected_values = candidate.person[name].to_numpy()
    actual_values = actual.person[name].to_numpy()
    known = ~np.isnan(expected_values)
    assert np.array_equal(np.isnan(actual_values), ~known)
    assert actual_values[known].tobytes() == expected_values[known].tobytes()


@pytest.mark.parametrize("view", ["full", "pruned", "local"])
def test_arrow_string_readback_compares_content_and_reports_storage(view):
    _, candidate, _ = _case(view)
    tables = _logical_tables(candidate)
    storage_changes = []
    option_before = pd.options.mode.string_storage
    for entity, column in (("household", "county_fips"), ("spm_unit", UNIVERSE_INPUT)):
        tables[entity][column] = tables[entity][column].astype(
            pd.StringDtype(storage="pyarrow", na_value=pd.NA)
        )
        storage_changes.append(f"{entity}.{column}:string[pyarrow]->string[python]")
    snapshots = {entity: table.copy(deep=True) for entity, table in tables.items()}
    actual, normalizations = readback.verify_policyengine_h5_readback(
        candidate, tables, _period(), period=2024
    )
    assert normalizations == (
        f"person.{ROLE_INPUT}:bool->boolean",
        *storage_changes,
    )
    assert pd.options.mode.string_storage == option_before
    for entity in candidate.entities:
        pd.testing.assert_frame_equal(actual.table(entity), candidate.table(entity))
        pd.testing.assert_frame_equal(tables[entity], snapshots[entity])


@pytest.mark.parametrize(
    "change",
    [
        "value",
        "mask",
        "order",
        "object",
        "category",
        "numeric",
        "nan_policy",
        "arrow_dtype",
    ],
)
def test_arrow_storage_does_not_hide_changed_strings_or_wrong_dtype(change):
    _, candidate, _ = _case()
    tables = _logical_tables(candidate)
    column = tables["spm_unit"][UNIVERSE_INPUT].astype(
        pd.StringDtype(storage="pyarrow", na_value=pd.NA)
    )
    if change == "value":
        column.iloc[0] = "OUTSIDE"
    elif change == "mask":
        column.iloc[0] = pd.NA
    elif change == "order":
        column = column.iloc[::-1].reset_index(drop=True)
    elif change in ("object", "category"):
        column = column.astype(change)
    elif change == "numeric":
        column = pd.Series([0, 1, 2, 3], dtype=np.int64)
    elif change == "arrow_dtype":
        column = column.astype(pd.ArrowDtype(pa.string()))
    else:
        column = column.astype(pd.StringDtype(storage="pyarrow", na_value=np.nan))
    tables["spm_unit"][UNIVERSE_INPUT] = column
    with pytest.raises(readback.PolicyEngineH5ReadbackError):
        readback.verify_policyengine_h5_readback(
            candidate, tables, _period(), period=2024
        )


def test_arrow_source_is_still_refused_before_writer(tmp_path):
    parent, candidate, arguments = _case()
    for frame in (parent, candidate):
        frame.table("spm_unit")[UNIVERSE_INPUT] = frame.table("spm_unit")[
            UNIVERSE_INPUT
        ].astype(pd.StringDtype(storage="pyarrow", na_value=pd.NA))
    writer = _InventedWriter()
    with pytest.raises(
        readback.PolicyEngineH5ReadbackError, match="H5_UNSUPPORTED_DTYPE"
    ):
        readback.write_verified_policyengine_h5_export(
            parent, candidate, writer, tmp_path / "export.h5", **arguments
        )
    assert writer.calls == []


@pytest.mark.parametrize("view", ["full", "pruned", "local"])
def test_mocked_arrow_readback_retains_parent_comparison(tmp_path, monkeypatch, view):
    parent, candidate, arguments = _case(view)
    tables = _mock_reader(monkeypatch, candidate)
    tables["spm_unit"][UNIVERSE_INPUT] = tables["spm_unit"][UNIVERSE_INPUT].astype(
        pd.StringDtype(storage="pyarrow", na_value=pd.NA)
    )
    receipt = readback.write_verified_policyengine_h5_export(
        parent, candidate, _InventedWriter(), tmp_path / "export.h5", **arguments
    )
    assert receipt.normalizations == (
        f"person.{ROLE_INPUT}:bool->boolean",
        f"spm_unit.{UNIVERSE_INPUT}:string[pyarrow]->string[python]",
    )
    assert tables["spm_unit"][UNIVERSE_INPUT].dtype.storage == "pyarrow"
    assert (
        receipt.source_ancestry_verified is False and receipt.release_eligible is False
    )


@pytest.mark.parametrize(
    "stored",
    [
        pd.Series([2025], dtype=np.int64),
        pd.Series([True], dtype=np.bool_),
        pd.Series([2024.0], dtype=np.float64),
        pd.Series(["2024"], dtype="string"),
        pd.Series([], dtype=np.int64),
        pd.Series([2024, 2024], dtype=np.int64),
    ],
    ids=["wrong_year", "boolean", "float", "string", "empty", "multiple"],
)
def test_period_is_exactly_one_integer_year(stored):
    _, candidate, _ = _case()
    with pytest.raises(readback.PolicyEngineH5ReadbackError):
        readback.verify_policyengine_h5_readback(
            candidate, _logical_tables(candidate), stored, period=2024
        )


@pytest.mark.parametrize(
    "unsupported",
    ["nullable_integer", "missing_boolean", "missing_string", "object", "categorical"],
)
def test_unprobed_source_codecs_are_refused(unsupported):
    _, candidate, _ = _case()
    tables = _logical_tables(candidate)
    if unsupported == "nullable_integer":
        candidate.person["person_id"] = candidate.person.person_id.astype("Int64")
    elif unsupported == "missing_boolean":
        candidate.person.loc[0, ROLE_INPUT] = pd.NA
    elif unsupported == "missing_string":
        candidate.table("spm_unit").loc[0, UNIVERSE_INPUT] = pd.NA
    elif unsupported == "object":
        candidate.table("spm_unit")[UNIVERSE_INPUT] = candidate.table("spm_unit")[
            UNIVERSE_INPUT
        ].astype(object)
    else:
        candidate.table("spm_unit")[UNIVERSE_INPUT] = candidate.table("spm_unit")[
            UNIVERSE_INPUT
        ].astype("category")
    # Match the unsupported representation on both sides. A dtype mismatch
    # alone must not be the reason these unprobed source codecs are refused.
    for entity in candidate.entities:
        for column in candidate.table(entity):
            if column != ROLE_INPUT or unsupported == "missing_boolean":
                tables[entity][column] = candidate.table(entity)[column].copy()
    with pytest.raises(readback.PolicyEngineH5ReadbackError):
        readback.verify_policyengine_h5_readback(
            candidate, tables, _period(), period=2024
        )


class _InventedWriter:
    """Mock only the foreign writer call; this never writes a real HDF file."""

    def __init__(self, callback=None):
        self.callback = callback
        self.calls = []

    def write_dataset(self, bundle, path, *, period):
        self.calls.append((bundle, Path(path), period))
        Path(path).write_bytes(b"invented-control-flow-only-not-hdf")
        if self.callback is not None:
            self.callback()


def _mock_reader(monkeypatch, candidate):
    tables = _logical_tables(candidate)
    monkeypatch.setattr(readback, "_read_h5", lambda path: (tables, _period()))
    return tables


@pytest.mark.parametrize("view", ["full", "pruned", "local"])
def test_mocked_writer_lifecycle_returns_only_supplied_parent_claims(
    tmp_path, monkeypatch, view
):
    parent, candidate, arguments = _case(view)
    snapshots = [
        {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
        for frame in (parent, candidate)
    ]
    _mock_reader(monkeypatch, candidate)
    writer = _InventedWriter()
    path = tmp_path / "export.h5"
    receipt = readback.write_verified_policyengine_h5_export(
        parent, candidate, writer, path, **arguments
    )
    assert writer.calls == [(candidate, path, 2024)]
    assert receipt.source_ancestry_verified is False
    assert receipt.release_eligible is False and receipt.period == 2024
    assert type(receipt.binding) is bytes and receipt.binding
    assert len(receipt.sha256) == 64 and receipt.nonserialized_context
    assert receipt.normalizations and ROLE_INPUT in receipt.normalizations[0]
    for frame, snapshot in zip((parent, candidate), snapshots, strict=True):
        for entity, table in snapshot.items():
            pd.testing.assert_frame_equal(frame.table(entity), table, check_exact=True)


def test_existing_output_is_preserved_before_any_writer_call(tmp_path, monkeypatch):
    parent, candidate, arguments = _case()
    _mock_reader(monkeypatch, candidate)
    path = tmp_path / "export.h5"
    path.write_bytes(b"pre-existing unrelated output")
    writer = _InventedWriter()
    with pytest.raises((readback.PolicyEngineH5ReadbackError, FileExistsError)):
        readback.write_verified_policyengine_h5_export(
            parent, candidate, writer, path, **arguments
        )
    assert writer.calls == []
    assert path.read_bytes() == b"pre-existing unrelated output"


@pytest.mark.parametrize(
    "unsupported",
    [
        "missing_boolean",
        "source_weight",
        "uncalibrated_weight",
        "missing_role_column",
        "integer_role",
        "missing_scope_column",
        "noncanonical_scope",
    ],
)
def test_unsupported_candidate_refuses_before_writer(tmp_path, unsupported):
    parent, candidate, arguments = _case()
    if unsupported == "missing_boolean":
        # Change both sides so the ordinary parent comparison alone could pass.
        parent.person.loc[0, ROLE_INPUT] = pd.NA
        candidate.person.loc[0, ROLE_INPUT] = pd.NA
    elif unsupported == "source_weight":
        for frame in (parent, candidate):
            frame.table("household")["household_weight"] = frame.weights_for(
                "household"
            ).values
    elif unsupported == "uncalibrated_weight":
        candidate._weights["household"] = Weights(
            candidate.weights_for("household").values, WeightKind.IMPORTANCE
        )
    else:
        # Paired edits preserve parent/candidate equality, so these controls
        # specifically require the wrapper's native role/scope admission.
        for frame in (parent, candidate):
            if unsupported == "missing_role_column":
                frame.person.drop(columns=ROLE_INPUT, inplace=True)
            elif unsupported == "integer_role":
                frame.person[ROLE_INPUT] = frame.person[ROLE_INPUT].astype(np.int8)
            elif unsupported == "missing_scope_column":
                frame.table("spm_unit").drop(columns=UNIVERSE_INPUT, inplace=True)
            else:
                frame.table("spm_unit").loc[0, UNIVERSE_INPUT] = "UNKNOWN"
    writer = _InventedWriter()
    path = tmp_path / "export.h5"
    with pytest.raises(readback.PolicyEngineH5ReadbackError):
        readback.write_verified_policyengine_h5_export(
            parent, candidate, writer, path, **arguments
        )
    assert writer.calls == [] and not path.exists()


@pytest.mark.parametrize("phase", ["writer", "final_file_read"])
@pytest.mark.parametrize(
    "target", ["parent", "candidate", "excluded_weight", "scope", "readback"]
)
def test_foreign_io_cannot_mutate_previously_compared_inputs(
    tmp_path, monkeypatch, phase, target
):
    scenarios = ["value"]
    arrow_dtype = None
    if target == "parent":
        try:
            arrow_dtype = pd.StringDtype(storage="pyarrow", na_value=pd.NA)
        except ImportError:
            if os.environ.get("SPM_REQUIRE_ARROW_POLICY_CONTROLS") == "1":
                raise
            # Optional backend coverage must not skip the original value check.
            pass
        else:
            scenarios.extend(["strata_backend", "column_axis_backend"])
    expected_count = 3 if arrow_dtype is not None else 1
    completed = []
    original_seal = readback._file_seal
    for scenario in scenarios:
        # Fresh inputs/output make each refusal independent of the others.
        parent, candidate, arguments = _case("local")
        if scenario == "column_axis_backend":
            for frame in (parent, candidate):
                frame.person.columns = pd.Index(
                    frame.person.columns,
                    dtype=pd.StringDtype(storage="python", na_value=pd.NA),
                )
        tables = _logical_tables(candidate)
        state = {"read": False, "mutated": False}

        def mutate(
            state=state,
            scenario=scenario,
            parent=parent,
            candidate=candidate,
            arguments=arguments,
            tables=tables,
        ):
            if state["mutated"]:
                return
            if scenario == "strata_backend":
                for frame in (parent, candidate):
                    before = frame.strata.tolist()
                    assert frame.strata.dtype.storage == "python"
                    frame._strata = frame.strata.astype(arrow_dtype)
                    assert frame.strata.dtype.storage == "pyarrow"
                    assert frame.strata.tolist() == before
            elif scenario == "column_axis_backend":
                for frame in (parent, candidate):
                    before = frame.person.columns.tolist()
                    assert frame.person.columns.dtype.storage == "python"
                    frame.person.columns = frame.person.columns.astype(arrow_dtype)
                    assert frame.person.columns.dtype.storage == "pyarrow"
                    assert frame.person.columns.tolist() == before
            elif target == "parent":
                # This cell lies outside the retained local household.
                parent.person.loc[0, "employment_income_before_lsr"] = 17.0
            elif target == "candidate":
                candidate.person.loc[candidate.person.index[0], ROLE_INPUT] = True
            elif target == "excluded_weight":
                arguments["calibrated_weights"][0] = 1.0
            elif target == "scope":
                arguments["scope_household_ids"][0] = 20
            else:
                tables["person"].loc[tables["person"].index[0], ROLE_INPUT] = True
            state["mutated"] = True

        def read_tables(path, state=state, tables=tables):
            state["read"] = True
            return tables, _period()

        def seal(path, state=state, mutate=mutate):
            result = original_seal(path)
            if phase == "final_file_read" and state["read"]:
                mutate()
            return result

        output_dir = tmp_path / scenario
        output_dir.mkdir()
        with monkeypatch.context() as io_patch:
            io_patch.setattr(readback, "_read_h5", read_tables)
            io_patch.setattr(readback, "_file_seal", seal)
            writer = _InventedWriter(mutate if phase == "writer" else None)
            with pytest.raises(
                (readback.PolicyEngineH5ReadbackError, RetainedFrameExportError)
            ):
                readback.write_verified_policyengine_h5_export(
                    parent, candidate, writer, output_dir / "export.h5", **arguments
                )
        assert state["mutated"]
        completed.append(scenario)
    assert len(completed) == expected_count


def test_readback_added_default_and_changed_file_are_not_success(tmp_path, monkeypatch):
    parent, candidate, arguments = _case()
    tables = _logical_tables(candidate)
    tables["person"]["default_created_input"] = False
    monkeypatch.setattr(readback, "_read_h5", lambda path: (tables, _period()))
    with pytest.raises(readback.PolicyEngineH5ReadbackError):
        readback.write_verified_policyengine_h5_export(
            parent, candidate, _InventedWriter(), tmp_path / "default.h5", **arguments
        )

    tables = _logical_tables(candidate)

    def changed_file(path):
        Path(path).write_bytes(b"changed-during-invented-reader")
        return tables, _period()

    monkeypatch.setattr(readback, "_read_h5", changed_file)
    with pytest.raises(readback.PolicyEngineH5ReadbackError):
        readback.write_verified_policyengine_h5_export(
            parent, candidate, _InventedWriter(), tmp_path / "changed.h5", **arguments
        )
