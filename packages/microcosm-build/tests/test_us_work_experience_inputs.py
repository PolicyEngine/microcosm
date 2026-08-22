"""US work-experience industry and worked-last-year input stage tests.

The stage carries the ASEC work-experience industry recodes (``WEIND``
detailed groups, ``WEMIND`` major groups) and derives ``worked_last_year`` as
``WKSWORK > 0`` — the official universe condition of the recode block.  The
tests exercise the manifest declaration, the derive handler's fail-closed
identities, the operator's sidecar restore, idempotency, and the signal gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_WORK_EXPERIENCE_NONCONSTANT_PERSON_COLUMNS,
    US_WORK_EXPERIENCE_OUTPUT_COLUMNS,
    US_WORK_EXPERIENCE_REQUIRED_SOURCE_COLUMNS,
    US_WORK_EXPERIENCE_STAGE_NAME,
    clone_us_frame_for_puf_support,
    derive_us_work_experience_inputs_from_manifest,
    us_work_experience_signal_gate,
    us_work_experience_stage_spec,
    us_work_experience_summary,
    with_us_work_experience_inputs,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024
_OUTPUTS = ("detailed_industry_recode", "major_industry_recode", "worked_last_year")


def _person_table(rows: list[dict]) -> pd.DataFrame:
    records: list[dict] = []
    for index, row in enumerate(rows):
        record = {"WEIND": 0, "WEMIND": 0, "WKSWORK": 0}
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_household_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _us_frame(person_rows: list[dict]) -> Frame:
    person = _person_table(person_rows)
    n = len(person)
    household_ids = person["person_household_id"].to_numpy(dtype=np.int64)
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = household_ids + 1_000
    person["person_spm_unit_id"] = household_ids + 2_000
    person["person_family_id"] = household_ids + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype=np.int64) + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unique_households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": unique_households + 2_000}),
        "family": pd.DataFrame({"family_id": unique_households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(n, dtype=np.int64) + 4_000}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.ones(len(unique_households), dtype=np.float64),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _plausible_rows() -> list[dict]:
    """100 people: 52 workers across industries, 30 never-worked, 18 NIU."""

    rows: list[dict] = []
    for index in range(52):
        detailed = 1 + index % 21
        rows.append(
            {"WEIND": detailed, "WEMIND": 1 + detailed % 14, "WKSWORK": 10 + index % 43}
        )
    rows.extend({"WEIND": 23, "WEMIND": 15} for _ in range(30))
    rows.extend({} for _ in range(18))
    return rows


def _operation():
    spec = us_work_experience_stage_spec()
    return next(
        op for op in spec.operations if op.kind == "derive_work_experience_inputs"
    )


class TestStageSpec:
    def test_manifest_declares_the_outputs(self) -> None:
        spec = us_work_experience_stage_spec()
        assert spec.stage == US_WORK_EXPERIENCE_STAGE_NAME == "work_experience_inputs"
        assert tuple(spec.outputs) == US_WORK_EXPERIENCE_OUTPUT_COLUMNS == _OUTPUTS
        assert tuple(spec.nonnegative_outputs) == (
            "detailed_industry_recode",
            "major_industry_recode",
        )
        assert US_WORK_EXPERIENCE_NONCONSTANT_PERSON_COLUMNS == _OUTPUTS
        assert US_WORK_EXPERIENCE_REQUIRED_SOURCE_COLUMNS == (
            "WEIND",
            "WEMIND",
            "WKSWORK",
        )

    def test_stage_reads_person_then_derives(self) -> None:
        kinds = [op.kind for op in us_work_experience_stage_spec().operations]
        assert kinds == ["read_table", "derive_work_experience_inputs"]

    def test_manifest_pins_every_pooled_archive_and_the_dictionary(self) -> None:
        artifacts = us_work_experience_stage_spec().artifacts
        microdata = [item for item in artifacts if item["kind"] == "public_microdata"]
        assert [item["member"] for item in microdata] == [
            "pppub23.csv",
            "pppub24.csv",
            "pppub25.csv",
        ]
        assert all(len(item["sha256"]) == 64 for item in microdata)
        dictionary = [
            item for item in artifacts if item["kind"] == "official_data_dictionary"
        ]
        assert len(dictionary) == 1 and "WEIND" in dictionary[0]["lines"]

    def test_handler_is_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers["derive_work_experience_inputs"]
            is derive_us_work_experience_inputs_from_manifest
        )


class TestDerivation:
    def _derive(self, table: pd.DataFrame) -> pd.DataFrame:
        return derive_us_work_experience_inputs_from_manifest(table, _operation(), None)

    def test_carries_recodes_and_derives_worked_from_weeks(self) -> None:
        table = _person_table(
            [
                {"WEIND": 7, "WEMIND": 5, "WKSWORK": 52},
                {"WEIND": 22, "WEMIND": 15, "WKSWORK": 40},
                {"WEIND": 23, "WEMIND": 15, "WKSWORK": 0},
                {},
            ]
        )
        result = self._derive(table)
        assert result["detailed_industry_recode"].tolist() == [7, 22, 23, 0]
        assert result["major_industry_recode"].tolist() == [5, 15, 15, 0]
        assert result["worked_last_year"].tolist() == [True, True, False, False]
        assert result["detailed_industry_recode"].dtype == np.int16
        assert result["major_industry_recode"].dtype == np.int16
        assert result["worked_last_year"].dtype == bool

    def test_rejects_unexpected_operation_kind(self) -> None:
        spec = us_work_experience_stage_spec()
        read = next(op for op in spec.operations if op.kind == "read_table")
        with pytest.raises(SourceRuntimeError, match="unexpected operation"):
            derive_us_work_experience_inputs_from_manifest(
                _person_table([{}]), read, None
            )

    def test_requires_the_person_table(self) -> None:
        with pytest.raises(SourceRuntimeError, match="read first"):
            derive_us_work_experience_inputs_from_manifest(None, _operation(), None)

    def test_rejects_missing_source_columns(self) -> None:
        table = _person_table([{}]).drop(columns=["WEMIND"])
        with pytest.raises(SourceRuntimeError, match=r"source column\(s\).*WEMIND"):
            self._derive(table)

    @pytest.mark.parametrize(
        ("row", "message"),
        (
            ({"WEIND": 24, "WEMIND": 1, "WKSWORK": 1}, r"WEIND.*\[0, 23\]"),
            ({"WEIND": 1, "WEMIND": 16, "WKSWORK": 1}, r"WEMIND.*\[0, 15\]"),
            ({"WEIND": 1, "WEMIND": 1, "WKSWORK": 53}, r"WKSWORK.*\[0, 52\]"),
            ({"WEIND": 1, "WEMIND": 1, "WKSWORK": 0}, "universe identity"),
            ({"WEIND": 0, "WEMIND": 0, "WKSWORK": 5}, "universe identity"),
            ({"WEIND": 23, "WEMIND": 0, "WKSWORK": 0}, "disagreeing"),
        ),
    )
    def test_fails_closed_on_official_identities(self, row: dict, message: str) -> None:
        with pytest.raises(SourceRuntimeError, match=message):
            self._derive(_person_table([row]))


class TestOperator:
    def test_materializes_outputs_from_present_source_columns(self) -> None:
        frame = with_us_work_experience_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person")
        assert person["detailed_industry_recode"].dtype == np.int16
        assert person["major_industry_recode"].dtype == np.int16
        assert person["worked_last_year"].dtype == bool
        assert int(person["worked_last_year"].sum()) == 52
        assert int(person["detailed_industry_recode"].eq(23).sum()) == 30
        assert int(person["detailed_industry_recode"].eq(0).sum()) == 18
        worker = person["detailed_industry_recode"].between(1, 22)
        assert worker.equals(person["worked_last_year"])

    def test_requires_the_frozen_weeks_column(self) -> None:
        source = _us_frame(_plausible_rows())
        tables = {entity: source.table(entity).copy() for entity in source.entities}
        tables["person"] = tables["person"].drop(columns=["WKSWORK"])
        stripped = Frame(
            tables,
            source.schema,
            {entity: source.weights_for(entity) for entity in source.weighted_entities},
            source.strata,
        )
        with pytest.raises(SourceRuntimeError, match="WKSWORK"):
            with_us_work_experience_inputs(stripped, seed=0, time_period=TIME_PERIOD)

    def test_requires_the_sidecar_when_recodes_are_absent(self) -> None:
        source = _us_frame(_plausible_rows())
        tables = {entity: source.table(entity).copy() for entity in source.entities}
        tables["person"] = tables["person"].drop(columns=["WEIND", "WEMIND"])
        stripped = Frame(
            tables,
            source.schema,
            {entity: source.weights_for(entity) for entity in source.weighted_entities},
            source.strata,
        )
        with pytest.raises(SourceRuntimeError, match="pinned ASEC work-experience"):
            with_us_work_experience_inputs(stripped, seed=0, time_period=TIME_PERIOD)

    def test_restores_recodes_from_the_sidecar_by_identity(self) -> None:
        rows = _plausible_rows()
        source = _us_frame(rows)
        tables = {entity: source.table(entity).copy() for entity in source.entities}
        person = tables["person"]
        n = len(person)
        person["source_year"] = 2023
        person["PERIDNUM"] = [f"{index:022d}" for index in range(n)]
        sidecar = pd.DataFrame(
            {
                "source_year": np.full(n, 2023, dtype=np.int64),
                "PH_SEQ": np.arange(1, n + 1, dtype=np.int64),
                "P_SEQ": np.ones(n, dtype=np.int64),
                "A_LINENO": np.ones(n, dtype=np.int64),
                "PERIDNUM": person["PERIDNUM"].to_numpy(),
                "WEIND": person["WEIND"].to_numpy(),
                "WEMIND": person["WEMIND"].to_numpy(),
            }
        )
        tables["person"] = person.drop(columns=["WEIND", "WEMIND"])
        stripped = Frame(
            tables,
            source.schema,
            {entity: source.weights_for(entity) for entity in source.weighted_entities},
            source.strata,
        )
        restored = with_us_work_experience_inputs(
            stripped,
            seed=0,
            time_period=TIME_PERIOD,
            asec_work_experience_source=sidecar,
        )
        direct = with_us_work_experience_inputs(source, seed=0, time_period=TIME_PERIOD)
        for column in _OUTPUTS:
            assert restored.table("person")[column].tolist() == (
                direct.table("person")[column].tolist()
            )

    def test_passes_a_healthy_surface_through_unchanged(self) -> None:
        first = with_us_work_experience_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        second = with_us_work_experience_inputs(first, seed=1, time_period=TIME_PERIOD)
        assert second is first

    def test_both_support_clones_carry_identical_values(self) -> None:
        expanded = clone_us_frame_for_puf_support(_us_frame(_plausible_rows()))
        materialized = with_us_work_experience_inputs(
            expanded, seed=0, time_period=TIME_PERIOD
        )
        summary = us_work_experience_summary(materialized)
        channels = summary["channels"]
        assert set(channels) == {
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        }
        asec = channels[BASE_ASEC_SUPPORT_CHANNEL]
        puf = channels[PUF_TAX_DETAIL_SUPPORT_CHANNEL]
        assert asec["rows"] == puf["rows"] == 100
        assert asec["worked_share"] == pytest.approx(puf["worked_share"])
        assert asec["recode_positive_share"] == pytest.approx(
            puf["recode_positive_share"]
        )


class TestSignalGate:
    def test_plausible_surface_passes_with_zero_identity_breaks(self) -> None:
        frame = with_us_work_experience_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        gate = us_work_experience_signal_gate(frame)
        assert gate.passed, gate.failures
        details = gate.details
        assert details["worked_share"] == pytest.approx(0.52)
        assert details["recode_positive_share"] == pytest.approx(0.82)
        assert details["never_worked_share"] == pytest.approx(0.30)
        assert details["universe_identity_breaks"] == 0
        assert details["recode_zero_breaks"] == 0

    def test_missing_outputs_fail_the_gate(self) -> None:
        gate = us_work_experience_signal_gate(_us_frame(_plausible_rows()))
        assert not gate.passed
        assert "person columns missing" in gate.failures[0]

    def test_collapsed_surface_fails_the_plausibility_bands(self) -> None:
        frame = with_us_work_experience_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["person"]["worked_last_year"] = False
        tables["person"]["detailed_industry_recode"] = np.int16(0)
        tables["person"]["major_industry_recode"] = np.int16(0)
        collapsed = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
        )
        gate = us_work_experience_signal_gate(collapsed)
        assert not gate.passed
        assert any("worked-last-year share" in failure for failure in gate.failures)
        assert any("industry-recode share" in failure for failure in gate.failures)

    def test_identity_breaks_fail_the_gate(self) -> None:
        frame = with_us_work_experience_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        person = tables["person"]
        person.loc[person.index[0], "worked_last_year"] = False
        broken = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
        )
        gate = us_work_experience_signal_gate(broken)
        assert not gate.passed
        assert any("disagreeing with worked" in failure for failure in gate.failures)
