"""US SNAP eligibility/exemption inputs stage tests (microcosm #244/#248)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import (
    US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS,
    US_ELIGIBILITY_INPUTS_PARENT_ID_COLUMNS,
    US_ELIGIBILITY_INPUTS_REQUIRED_SOURCE_COLUMNS,
    US_ELIGIBILITY_INPUTS_STAGE_NAME,
    derive_us_eligibility_inputs_from_manifest,
    us_eligibility_inputs_signal_gate,
    us_eligibility_inputs_stage_spec,
    us_eligibility_inputs_summary,
    with_us_eligibility_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.build.us_runtime.support_provenance import support_clone_index_column
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024

_RAW_BASELINE = {
    "PEDISDRS": 2,
    "PEDISEAR": 2,
    "PEDISEYE": 2,
    "PEDISOUT": 2,
    "PEDISPHY": 2,
    "PEDISREM": 2,
    "A_HSCOL": 0,
    "A_FTPT": 0,
    "PEPAR1": -1,
    "PEPAR2": -1,
    "PH_SEQ": 1,
    "VET_VAL": 0,
    "SSI_VAL": 0,
    "A_AGE": 40,
}


def _person_table(rows: list[dict]) -> pd.DataFrame:
    """A raw-ASEC person table: baseline is a no-flag adult."""

    records = []
    for index, row in enumerate(rows):
        record = dict(_RAW_BASELINE)
        record["A_LINENO"] = index + 1
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_household_id", record["PH_SEQ"])
        records.append(record)
    return pd.DataFrame(records)


def _us_frame(
    person_rows: list[dict],
    *,
    household_weights: list[float] | None = None,
) -> Frame:
    person = _person_table(person_rows)
    n = len(person)
    household_ids = person["person_household_id"].to_numpy()
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_spm_unit_id"] = person["person_household_id"] + 2_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype="int64") + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unique_households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": unique_households + 2_000}),
        "family": pd.DataFrame({"family_id": unique_households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(n, dtype="int64") + 4_000}
        ),
    }
    weights = household_weights or [1.0] * len(unique_households)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray(weights, dtype=np.float64),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    """The frame with a hand-edited person table, for gate-failure cases."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _plausible_rows() -> list[dict]:
    """Fifty persons whose weighted shares land inside every band.

    Layout (single household, lines 1-50): parents on lines 1-10, their
    children on lines 11-20 (one child per parent), four disabled persons
    (three difficulty items, one via the reported-SSI alignment), two
    full-time college students, one veteran with payments, and no-flag
    adult fillers. Shares: disabled 0.08, students 0.04, parents 0.20,
    veterans' payments 0.02.
    """

    rows: list[dict] = [{} for _ in range(10)]  # lines 1-10: parents
    rows += [
        # lines 11-20: one child per parent. The first child names a second
        # parent as well, so parent_2_id carries signal without moving the
        # parent share: line 2 already parents line 12.
        {"PEPAR1": parent_line, "PEPAR2": 2, "A_AGE": 8}
        if parent_line == 1
        else {"PEPAR1": parent_line, "A_AGE": 8}
        for parent_line in range(1, 11)
    ]
    rows += [
        {"PEDISPHY": 1},
        {"PEDISREM": 1},
        {"PEDISEYE": 1},
        {"SSI_VAL": 6_000, "A_AGE": 45},
    ]  # 4 disabled (one also blind, keeping is_blind non-constant)
    rows += [{"A_HSCOL": 2, "A_FTPT": 1, "A_AGE": 20} for _ in range(2)]  # 2 students
    rows += [{"VET_VAL": 12_000}]  # 1 veteran
    rows += [{} for _ in range(23)]  # fillers to 50
    assert len(rows) == 50
    return rows


_PARENT_INDEX = 0
_CHILD_INDEX = 10
_DISABLED_INDEX = 20
_SSI_ALIGNED_INDEX = 23
_STUDENT_INDEX = 24
_VETERAN_INDEX = 26


class TestManifestDeclaration:
    def test_stage_is_declared_with_its_output_columns(self) -> None:
        spec = us_eligibility_inputs_stage_spec()
        assert spec.stage == US_ELIGIBILITY_INPUTS_STAGE_NAME
        assert tuple(spec.outputs) == US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS
        assert set(spec.nonnegative_outputs) == {
            "own_children_in_household",
            *US_ELIGIBILITY_INPUTS_PARENT_ID_COLUMNS,
            "veterans_benefits",
        }

    def test_parent_ids_are_declared_outputs(self) -> None:
        assert US_ELIGIBILITY_INPUTS_PARENT_ID_COLUMNS == (
            "parent_1_id",
            "parent_2_id",
        )
        assert set(US_ELIGIBILITY_INPUTS_PARENT_ID_COLUMNS) <= set(
            US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS
        )

    def test_handler_is_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers["derive_eligibility_inputs"]
            is derive_us_eligibility_inputs_from_manifest
        )


class TestDerivation:
    def _derive(self, table: pd.DataFrame) -> pd.DataFrame:
        spec = us_eligibility_inputs_stage_spec()
        operation = next(
            op for op in spec.operations if op.kind == "derive_eligibility_inputs"
        )
        return derive_us_eligibility_inputs_from_manifest(table, operation, None)

    def test_any_difficulty_item_flags_disability(self) -> None:
        table = _person_table(
            [{column: 1} for column in ("PEDISDRS", "PEDISEAR", "PEDISREM")] + [{}]
        )
        result = self._derive(table)
        assert result["is_disabled"].tolist() == [True, True, True, False]

    def test_reported_ssi_under_65_flags_disability(self) -> None:
        table = _person_table(
            [
                {"SSI_VAL": 5_000, "A_AGE": 40},
                {"SSI_VAL": 5_000, "A_AGE": 70},
                {"SSI_VAL": 0, "A_AGE": 40},
            ]
        )
        result = self._derive(table)
        assert result["is_disabled"].tolist() == [True, False, False]

    def test_blindness_maps_the_eye_item_only(self) -> None:
        table = _person_table([{"PEDISEYE": 1}, {"PEDISPHY": 1}])
        result = self._derive(table)
        assert result["is_blind"].tolist() == [True, False]

    def test_full_time_college_requires_both_enrollment_and_full_time(self) -> None:
        table = _person_table(
            [
                {"A_HSCOL": 2, "A_FTPT": 1},
                {"A_HSCOL": 2, "A_FTPT": 2},
                {"A_HSCOL": 1, "A_FTPT": 1},
                {},
            ]
        )
        result = self._derive(table)
        assert result["is_full_time_college_student"].tolist() == [
            True,
            False,
            False,
            False,
        ]

    def test_children_counted_per_parent_through_both_pointers(self) -> None:
        table = _person_table(
            [
                {},  # line 1: parent of lines 3 and 4
                {},  # line 2: second parent of line 3 only
                {"PEPAR1": 1, "PEPAR2": 2, "A_AGE": 9},
                {"PEPAR1": 1, "A_AGE": 12},
            ]
        )
        result = self._derive(table)
        assert result["own_children_in_household"].tolist() == [2.0, 1.0, 0.0, 0.0]

    def test_parent_pointers_do_not_cross_households(self) -> None:
        table = _person_table(
            [
                {"PH_SEQ": 1, "A_LINENO": 1},
                {"PH_SEQ": 2, "A_LINENO": 1},
                {"PH_SEQ": 2, "A_LINENO": 2, "PEPAR1": 1, "A_AGE": 8},
            ]
        )
        result = self._derive(table)
        assert result["own_children_in_household"].tolist() == [0.0, 1.0, 0.0]

    def test_two_parent_household_resolves_both_pointers(self) -> None:
        table = _person_table(
            [
                {},  # line 1 / person_id 1: parent
                {},  # line 2 / person_id 2: parent
                {"PEPAR1": 1, "PEPAR2": 2, "A_AGE": 9},
            ]
        )
        result = self._derive(table)
        assert result["parent_1_id"].tolist() == [0, 0, 1]
        assert result["parent_2_id"].tolist() == [0, 0, 2]
        assert result["own_children_in_household"].tolist() == [1.0, 1.0, 0.0]

    def test_single_parent_leaves_the_second_pointer_unset(self) -> None:
        table = _person_table([{}, {"PEPAR1": 1, "A_AGE": 4}])
        result = self._derive(table)
        assert result["parent_1_id"].tolist() == [0, 1]
        assert result["parent_2_id"].tolist() == [0, 0]

    def test_three_generation_household_points_children_at_their_own_parent(
        self,
    ) -> None:
        # Line 1 grandparent, line 2 their adult child, line 3 the grandchild.
        # The grandchild's pointer names line 2 only: the count-based proxy
        # would treat the grandparent as a second parent (microcosm#884).
        table = _person_table(
            [
                {"A_AGE": 62},
                {"PEPAR1": 1, "A_AGE": 30},
                {"PEPAR1": 2, "A_AGE": 6},
            ]
        )
        result = self._derive(table)
        assert result["parent_1_id"].tolist() == [0, 1, 2]
        assert result["parent_2_id"].tolist() == [0, 0, 0]
        assert result["own_children_in_household"].tolist() == [1.0, 1.0, 0.0]

    def test_unresolved_pointer_reads_zero(self) -> None:
        # Line 9 does not exist in the household, and line 1 of the other
        # household is not reachable from household 2.
        table = _person_table(
            [
                {"PH_SEQ": 1, "A_LINENO": 1},
                {"PH_SEQ": 2, "A_LINENO": 1},
                {"PH_SEQ": 2, "A_LINENO": 2, "PEPAR1": 9, "PEPAR2": 1, "A_AGE": 8},
            ]
        )
        result = self._derive(table)
        assert result["parent_1_id"].tolist() == [0, 0, 0]
        assert result["parent_2_id"].tolist() == [0, 0, 2]

    def test_absent_pointer_sentinels_read_zero(self) -> None:
        table = _person_table([{}, {"PEPAR1": 0, "PEPAR2": -1, "A_AGE": 8}])
        result = self._derive(table)
        assert result["parent_1_id"].tolist() == [0, 0]
        assert result["parent_2_id"].tolist() == [0, 0]

    def test_parent_ids_are_int64_not_float(self) -> None:
        table = _person_table([{}, {"PEPAR1": 1, "A_AGE": 8}])
        result = self._derive(table)
        for column in US_ELIGIBILITY_INPUTS_PARENT_ID_COLUMNS:
            assert result[column].dtype == np.int64

    def test_parent_ids_use_person_id_not_the_line_number(self) -> None:
        table = _person_table([{"person_id": 5_000}, {"PEPAR1": 1, "A_AGE": 8}])
        result = self._derive(table)
        assert result["parent_1_id"].tolist() == [0, 5_000]

    def test_missing_person_id_is_named(self) -> None:
        table = _person_table([{}, {"PEPAR1": 1, "A_AGE": 8}]).drop(
            columns=["person_id"]
        )
        with pytest.raises(SourceRuntimeError, match="person_id"):
            self._derive(table)

    def test_repeated_household_line_is_refused(self) -> None:
        table = _person_table(
            [
                {"PH_SEQ": 1, "A_LINENO": 1},
                {"PH_SEQ": 1, "A_LINENO": 1},
            ]
        )
        with pytest.raises(SourceRuntimeError, match="repeats line"):
            self._derive(table)

    def test_veterans_benefits_map_vet_val_and_clip_negatives(self) -> None:
        table = _person_table([{"VET_VAL": 8_400}, {"VET_VAL": -1}, {}])
        result = self._derive(table)
        assert result["veterans_benefits"].tolist() == [8_400.0, 0.0, 0.0]

    def test_missing_raw_column_is_named(self) -> None:
        table = _person_table([{}]).drop(columns=["VET_VAL"])
        with pytest.raises(SourceRuntimeError, match="VET_VAL"):
            self._derive(table)

    def test_requires_person_table_first(self) -> None:
        spec = us_eligibility_inputs_stage_spec()
        operation = next(
            op for op in spec.operations if op.kind == "derive_eligibility_inputs"
        )
        with pytest.raises(SourceRuntimeError, match="person table"):
            derive_us_eligibility_inputs_from_manifest(None, operation, None)

    def test_unexpected_parameters_are_refused(self) -> None:
        operation_spec = SourceStageSpec.from_mapping(
            {
                "stage": US_ELIGIBILITY_INPUTS_STAGE_NAME,
                "survey": "test ASEC",
                "source": "https://example.com",
                "grain": "person",
                "operations": [
                    {"kind": "read_table", "table": "person"},
                    {"kind": "derive_eligibility_inputs", "surprise": True},
                ],
                "outputs": list(US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS),
            }
        )
        operation = operation_spec.operations[1]
        with pytest.raises(SourceRuntimeError, match="surprise"):
            derive_us_eligibility_inputs_from_manifest(
                _person_table([{}]), operation, None
            )


class TestFrameIntegration:
    def test_with_us_eligibility_inputs_writes_every_output_column(self) -> None:
        frame = with_us_eligibility_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person")
        for column in US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS:
            assert column in person.columns
        assert bool(person["is_disabled"].iloc[_DISABLED_INDEX])
        assert bool(person["is_disabled"].iloc[_SSI_ALIGNED_INDEX])
        assert bool(person["is_full_time_college_student"].iloc[_STUDENT_INDEX])
        assert person["own_children_in_household"].iloc[_PARENT_INDEX] == 1.0
        assert person["own_children_in_household"].iloc[_CHILD_INDEX] == 0.0
        assert person["veterans_benefits"].iloc[_VETERAN_INDEX] == 12_000.0

    def test_parent_pointer_count_is_derived_once_then_cloned(self) -> None:
        direct = with_us_eligibility_inputs(
            _us_frame([{}, {"PEPAR1": 1, "A_AGE": 8}]),
            seed=0,
            time_period=TIME_PERIOD,
        )
        expanded = clone_us_frame_for_puf_support(direct)
        person = expanded.table("person")
        parent = person["A_LINENO"].eq(1)

        assert set(person.loc[parent, support_clone_index_column("person")]) == {0, 1}
        assert person.loc[parent, "own_children_in_household"].tolist() == [1.0, 1.0]

    def test_parent_ids_name_the_parent_row_person_id(self) -> None:
        frame = with_us_eligibility_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person")
        assert person["parent_1_id"].dtype == np.int64
        assert (
            person["parent_1_id"].iloc[_CHILD_INDEX]
            == person["person_id"].iloc[_PARENT_INDEX]
        )
        # The first child names a second parent; every other row is unset.
        assert person["parent_2_id"].iloc[_CHILD_INDEX] == person["person_id"].iloc[1]
        assert person["parent_1_id"].iloc[_PARENT_INDEX] == 0
        assert person["parent_2_id"].iloc[_PARENT_INDEX] == 0

    def test_parent_ids_are_remapped_onto_each_support_clone(self) -> None:
        direct = with_us_eligibility_inputs(
            _us_frame([{}, {"PEPAR1": 1, "A_AGE": 8}]),
            seed=0,
            time_period=TIME_PERIOD,
        )
        expanded = clone_us_frame_for_puf_support(direct)
        person = expanded.table("person")
        clone_column = support_clone_index_column("person")
        for clone_index in sorted(person[clone_column].unique()):
            arm = person[person[clone_column] == clone_index]
            child = arm[arm["A_LINENO"] == 2].iloc[0]
            parent = arm[arm["A_LINENO"] == 1].iloc[0]
            # The pointer must name this arm's parent, not the other arm's.
            assert child["parent_1_id"] == parent["person_id"]
            assert child["parent_1_id"] in set(arm["person_id"])
            assert parent["parent_1_id"] == 0

    def test_absent_parent_sentinel_is_not_shifted_by_the_clone(self) -> None:
        direct = with_us_eligibility_inputs(
            _us_frame([{}, {"PEPAR1": 1, "A_AGE": 8}]),
            seed=0,
            time_period=TIME_PERIOD,
        )
        expanded = clone_us_frame_for_puf_support(direct)
        person = expanded.table("person")
        parents = person["A_LINENO"].eq(1)
        # 0 means "no parent"; shifting it would mint a valid-looking id.
        assert person.loc[parents, "parent_1_id"].tolist() == [0, 0]
        assert person.loc[parents, "parent_2_id"].tolist() == [0, 0]

    def test_frame_with_signal_passes_through_untouched(self) -> None:
        derived = with_us_eligibility_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        again = with_us_eligibility_inputs(derived, seed=1, time_period=TIME_PERIOD)
        assert again is derived

    def test_constant_default_landmine_is_recomputed_from_raw(self) -> None:
        # The published failure mode: all five columns present but broadcast
        # at their engine defaults (False/0), masking the raw signal.
        constants = {
            "is_disabled": False,
            "is_blind": False,
            "is_full_time_college_student": False,
            "own_children_in_household": 0.0,
            "parent_1_id": 0,
            "parent_2_id": 0,
            "veterans_benefits": 0.0,
        }
        rows = [
            {"PEDISPHY": 1, **constants},
            {**constants},
        ]
        frame = with_us_eligibility_inputs(
            _us_frame(rows), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person")
        assert person["is_disabled"].tolist() == [True, False]

    def test_missing_raw_columns_without_signal_raise(self) -> None:
        frame = _us_frame([{"person_id": 1}])
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["person"] = tables["person"].drop(
            columns=list(US_ELIGIBILITY_INPUTS_REQUIRED_SOURCE_COLUMNS)
        )
        stripped = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )
        with pytest.raises(SourceRuntimeError, match="PEDIS"):
            with_us_eligibility_inputs(stripped, seed=0, time_period=TIME_PERIOD)


class TestGate:
    def test_plausible_distribution_passes(self) -> None:
        frame = with_us_eligibility_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        gate = us_eligibility_inputs_signal_gate(frame)
        assert gate.passed, gate.failures
        summary = us_eligibility_inputs_summary(frame)
        assert summary["disabled_share"] == pytest.approx(0.08)
        assert summary["full_time_college_student_share"] == pytest.approx(0.04)
        assert summary["parent_share"] == pytest.approx(0.20)
        assert summary["veterans_benefits_share"] == pytest.approx(0.02)

    def test_missing_columns_fail(self) -> None:
        gate = us_eligibility_inputs_signal_gate(_us_frame([{}]))
        assert not gate.passed
        assert "missing" in gate.failures[0]

    def test_constant_default_surface_fails(self) -> None:
        constants = {
            "is_disabled": False,
            "is_blind": False,
            "is_full_time_college_student": False,
            "own_children_in_household": 0.0,
            "parent_1_id": 0,
            "parent_2_id": 0,
            "veterans_benefits": 0.0,
        }
        frame = _us_frame([{**constants} for _ in range(4)])
        gate = us_eligibility_inputs_signal_gate(frame)
        assert not gate.passed
        assert any("constant" in failure for failure in gate.failures)

    def test_parent_id_naming_no_person_fails(self) -> None:
        # The failure a missed id remap produces: a pointer that survives
        # into a frame whose person ids have moved.
        frame = with_us_eligibility_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person").copy()
        person.loc[person.index[_CHILD_INDEX], "parent_1_id"] = 10_000_000
        gate = us_eligibility_inputs_signal_gate(_replace_person(frame, person))
        assert not gate.passed
        assert any("name no person" in failure for failure in gate.failures)

    def test_parent_id_outside_the_household_fails(self) -> None:
        frame = with_us_eligibility_inputs(
            _us_frame(
                [
                    {"PH_SEQ": 1, "A_LINENO": 1},
                    {"PH_SEQ": 1, "A_LINENO": 2, "PEPAR1": 1, "A_AGE": 8},
                    {"PH_SEQ": 2, "A_LINENO": 1},
                    {"PH_SEQ": 2, "A_LINENO": 2, "PEPAR1": 1, "A_AGE": 8},
                ]
            ),
            seed=0,
            time_period=TIME_PERIOD,
        )
        person = frame.table("person").copy()
        # Point household 1's child at household 2's parent.
        person.loc[person.index[1], "parent_1_id"] = int(person["person_id"].iloc[2])
        gate = us_eligibility_inputs_signal_gate(_replace_person(frame, person))
        assert not gate.passed
        assert any(
            "outside the pointing person's household" in failure
            for failure in gate.failures
        )

    def test_self_parenting_fails(self) -> None:
        frame = with_us_eligibility_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person").copy()
        index = person.index[_CHILD_INDEX]
        person.loc[index, "parent_1_id"] = int(person.loc[index, "person_id"])
        gate = us_eligibility_inputs_signal_gate(_replace_person(frame, person))
        assert not gate.passed
        assert any("their own parent" in failure for failure in gate.failures)

    def test_count_disagreeing_with_the_pointers_fails(self) -> None:
        frame = with_us_eligibility_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person").copy()
        person.loc[person.index[_PARENT_INDEX], "own_children_in_household"] = 7.0
        gate = us_eligibility_inputs_signal_gate(_replace_person(frame, person))
        assert not gate.passed
        assert any(
            "disagrees with the parent pointers" in failure for failure in gate.failures
        )

    def test_null_parent_id_fails(self) -> None:
        frame = with_us_eligibility_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person").copy()
        person["parent_1_id"] = person["parent_1_id"].astype("float64")
        person.loc[person.index[_CHILD_INDEX], "parent_1_id"] = np.nan
        gate = us_eligibility_inputs_signal_gate(_replace_person(frame, person))
        assert not gate.passed
        assert any("never as null" in failure for failure in gate.failures)

    def test_person_id_zero_cannot_be_named_and_is_reported(self) -> None:
        # person_id is 0-based in production, so one person can never be
        # named by a 0-means-unknown pointer. The gate must not fail on it,
        # and the cost must be visible.
        frame = with_us_eligibility_inputs(
            _us_frame(
                [
                    {"person_id": 0},
                    {"PEPAR1": 1, "A_AGE": 8, "person_id": 1},
                    {"PEDISPHY": 1, "person_id": 2},
                    {"VET_VAL": 9_000, "person_id": 3},
                    {"A_HSCOL": 2, "A_FTPT": 1, "A_AGE": 20, "person_id": 4},
                ]
            ),
            seed=0,
            time_period=TIME_PERIOD,
        )
        person = frame.table("person")
        assert person["own_children_in_household"].iloc[0] == 1.0
        assert person["parent_1_id"].iloc[1] == 0
        summary = us_eligibility_inputs_summary(frame)
        assert summary["pointers_unnameable_at_person_id_zero"] == 1
        failures = [
            failure
            for failure in us_eligibility_inputs_signal_gate(frame).failures
            if "parent pointers" in failure
        ]
        assert failures == []

    def test_everyone_disabled_fails_the_share_band(self) -> None:
        rows = [{"PEDISPHY": 1} for _ in range(10)]
        frame = with_us_eligibility_inputs(
            _us_frame(rows), seed=0, time_period=TIME_PERIOD
        )
        gate = us_eligibility_inputs_signal_gate(frame)
        assert not gate.passed
        assert any("disabled share" in failure for failure in gate.failures)
