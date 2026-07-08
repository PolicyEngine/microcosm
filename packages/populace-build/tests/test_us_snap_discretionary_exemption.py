"""US SNAP ABAWD discretionary-exemption seeding tests (populace #323/#248)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceStageSpec
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from populace.build.us_runtime import (
    US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN,
    US_SNAP_DISCRETIONARY_EXEMPTION_REQUIRED_SOURCE_COLUMNS,
    US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME,
    derive_us_snap_discretionary_exemption_from_manifest,
    us_snap_discretionary_exemption_signal_gate,
    us_snap_discretionary_exemption_stage_spec,
    us_snap_discretionary_exemption_summary,
    with_us_snap_discretionary_exemption_inputs,
)
from populace.build.us_runtime.source_runtime import us_source_operation_handlers
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024


def _person_table(rows: list[dict]) -> pd.DataFrame:
    """A raw-ASEC person table: baseline is a child (outside the band)."""

    records = []
    for index, row in enumerate(rows):
        record = {"A_AGE": 10}
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_household_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _us_frame(person_rows: list[dict]) -> Frame:
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


def _context(seed: int = 7) -> SourceRuntimeContext:
    return SourceRuntimeContext(
        config=SourceRuntimeConfig(seed=seed, target_year=TIME_PERIOD),
        tables={},
    )


def _operation():
    spec = us_snap_discretionary_exemption_stage_spec()
    return next(
        op
        for op in spec.operations
        if op.kind == "derive_snap_abawd_discretionary_exemption"
    )


class TestManifestDeclaration:
    def test_stage_is_declared_with_the_output_column(self) -> None:
        spec = us_snap_discretionary_exemption_stage_spec()
        assert spec.stage == US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME
        assert tuple(spec.outputs) == (US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN,)

    def test_rate_is_declared_with_a_source(self) -> None:
        parameters = _operation().parameters
        declared = parameters["exemption_rate"]
        assert 0.0 < float(declared["value"]) < 1.0
        assert declared["source"].startswith("https://")

    def test_handler_is_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers["derive_snap_abawd_discretionary_exemption"]
            is derive_us_snap_discretionary_exemption_from_manifest
        )


class TestDerivation:
    def _derive(self, table: pd.DataFrame, *, seed: int = 7) -> pd.DataFrame:
        return derive_us_snap_discretionary_exemption_from_manifest(
            table, _operation(), _context(seed)
        )

    def test_only_adults_in_the_covered_band_can_draw_exempt(self) -> None:
        rows = [{"A_AGE": 30, "person_id": i + 1} for i in range(200)] + [
            {"A_AGE": 17, "person_id": 201},
            {"A_AGE": 65, "person_id": 202},
            {"A_AGE": 10, "person_id": 203},
        ]
        result = self._derive(_person_table(rows))
        exempt = result[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN]
        assert exempt.iloc[:200].any()  # ~200 draws at 8% ≈ 16 expected
        assert not exempt.iloc[200:].any()

    def test_band_edges_are_inclusive(self) -> None:
        rows = [{"A_AGE": 18, "person_id": i + 1} for i in range(60)] + [
            {"A_AGE": 64, "person_id": 61 + i} for i in range(60)
        ]
        result = self._derive(_person_table(rows))
        assert result[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].any()

    def test_draws_are_deterministic_per_seed(self) -> None:
        rows = [{"A_AGE": 25, "person_id": i + 1} for i in range(100)]
        first = self._derive(_person_table(rows), seed=3)
        second = self._derive(_person_table(rows), seed=3)
        assert first[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].tolist() == (
            second[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].tolist()
        )

    def test_source_identity_keys_the_draw_when_present(self) -> None:
        rows = [
            {
                "A_AGE": 25,
                "person_id": i + 1,
                "source_year": 2024,
                "source_household_id": 1,
                "source_person_id": 42,
            }
            for i in range(2)
        ]
        result = self._derive(_person_table(rows))
        values = result[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].tolist()
        assert values[0] == values[1]  # clones of one source person agree

    def test_missing_raw_column_is_named(self) -> None:
        table = _person_table([{}]).drop(columns=["A_AGE"])
        with pytest.raises(SourceRuntimeError, match="A_AGE"):
            self._derive(table)

    def test_requires_person_table_first(self) -> None:
        with pytest.raises(SourceRuntimeError, match="person table"):
            derive_us_snap_discretionary_exemption_from_manifest(
                None, _operation(), _context()
            )

    def test_unexpected_parameters_are_refused(self) -> None:
        operation_spec = SourceStageSpec.from_mapping(
            {
                "stage": US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME,
                "survey": "test ASEC",
                "source": "https://example.com",
                "grain": "person",
                "operations": [
                    {"kind": "read_table", "table": "person"},
                    {
                        "kind": "derive_snap_abawd_discretionary_exemption",
                        "exemption_rate": {"value": 0.08, "source": "https://x"},
                        "surprise": True,
                    },
                ],
                "outputs": [US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN],
            }
        )
        with pytest.raises(SourceRuntimeError, match="surprise"):
            derive_us_snap_discretionary_exemption_from_manifest(
                _person_table([{}]), operation_spec.operations[1], _context()
            )

    def test_missing_rate_parameter_is_refused(self) -> None:
        operation_spec = SourceStageSpec.from_mapping(
            {
                "stage": US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME,
                "survey": "test ASEC",
                "source": "https://example.com",
                "grain": "person",
                "operations": [
                    {"kind": "read_table", "table": "person"},
                    {"kind": "derive_snap_abawd_discretionary_exemption"},
                ],
                "outputs": [US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN],
            }
        )
        with pytest.raises(SourceRuntimeError, match="exemption_rate"):
            derive_us_snap_discretionary_exemption_from_manifest(
                _person_table([{}]), operation_spec.operations[1], _context()
            )


#: 600 adults 18-64 + 400 others: at the 8% cap the expected exempt
#: share of all persons is ~4.8%, inside the plausibility band, and 600
#: draws make an all-False fluke vanishingly unlikely.
def _plausible_rows() -> list[dict]:
    rows = [{"A_AGE": 20 + (i % 45), "person_id": i + 1} for i in range(600)]
    rows += [{"A_AGE": 10, "person_id": 601 + i} for i in range(400)]
    return rows


class TestFrameIntegration:
    def test_with_us_snap_discretionary_exemption_inputs_writes_the_column(
        self,
    ) -> None:
        frame = with_us_snap_discretionary_exemption_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        person = frame.table("person")
        assert US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN in person.columns
        assert person[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].any()
        assert (
            not person[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].iloc[600:].any()
        )

    def test_frame_with_signal_passes_through_untouched(self) -> None:
        derived = with_us_snap_discretionary_exemption_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        again = with_us_snap_discretionary_exemption_inputs(
            derived, seed=1, time_period=TIME_PERIOD
        )
        assert again is derived

    def test_constant_default_landmine_is_reseeded(self) -> None:
        rows = [
            {**row, "is_snap_abawd_discretionary_exempt": False}
            for row in _plausible_rows()
        ]
        frame = with_us_snap_discretionary_exemption_inputs(
            _us_frame(rows), seed=0, time_period=TIME_PERIOD
        )
        assert frame.table("person")[
            US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN
        ].any()

    def test_missing_raw_columns_without_signal_raise(self) -> None:
        frame = _us_frame([{"person_id": 1}])
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["person"] = tables["person"].drop(
            columns=list(US_SNAP_DISCRETIONARY_EXEMPTION_REQUIRED_SOURCE_COLUMNS)
        )
        stripped = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )
        with pytest.raises(SourceRuntimeError, match="A_AGE"):
            with_us_snap_discretionary_exemption_inputs(
                stripped, seed=0, time_period=TIME_PERIOD
            )


class TestGate:
    def test_plausible_seeded_share_passes(self) -> None:
        frame = with_us_snap_discretionary_exemption_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        gate = us_snap_discretionary_exemption_signal_gate(frame)
        assert gate.passed, gate.failures
        summary = us_snap_discretionary_exemption_summary(frame)
        assert 0.015 <= summary["exempt_share"] <= 0.09

    def test_missing_column_fails(self) -> None:
        gate = us_snap_discretionary_exemption_signal_gate(_us_frame([{}]))
        assert not gate.passed
        assert "missing" in gate.failures[0]

    def test_constant_false_fails(self) -> None:
        rows = [{"is_snap_abawd_discretionary_exempt": False} for _ in range(4)]
        gate = us_snap_discretionary_exemption_signal_gate(_us_frame(rows))
        assert not gate.passed
        assert any("constant" in failure for failure in gate.failures)

    def test_everyone_exempt_fails_the_share_band(self) -> None:
        rows = [
            {"A_AGE": 25, "is_snap_abawd_discretionary_exempt": True} for _ in range(9)
        ] + [{"is_snap_abawd_discretionary_exempt": False}]
        gate = us_snap_discretionary_exemption_signal_gate(_us_frame(rows))
        assert not gate.passed
        assert any("exempt share" in failure for failure in gate.failures)
