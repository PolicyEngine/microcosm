"""US education-assistance and education-credit input stage tests.

The retired eCPS pipeline sourced educational assistance directly from CPS
ASEC ``ED_VAL`` and imputed ``qualified_tuition_expenses`` from the PUF.  The
PUF transformation has already combined the applicable source fields before
this stage runs; this stage preserves that tuition amount and turns its
positive-support mask into the five factual AOTC eligibility inputs exported
by the reference eCPS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceStageSpec
from populace.build.source_runtime import SourceRuntimeError
from populace.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS,
    US_EDUCATION_INPUTS_NONCONSTANT_PERSON_COLUMNS,
    US_EDUCATION_INPUTS_OUTPUT_COLUMNS,
    US_EDUCATION_INPUTS_REQUIRED_SOURCE_COLUMNS,
    US_EDUCATION_INPUTS_STAGE_NAME,
    clone_us_frame_for_puf_support,
    derive_us_education_inputs_from_manifest,
    impute_us_puf_tax_detail_support,
    support_channel_column,
    us_education_inputs_signal_gate,
    us_education_inputs_stage_spec,
    us_education_inputs_summary,
    with_us_education_inputs,
)
from populace.build.us_runtime.source_runtime import us_source_operation_handlers
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024

_EXPECTED_AOTC_COLUMNS = (
    "is_pursuing_credential_for_american_opportunity_credit",
    "attends_eligible_educational_institution_for_american_opportunity_credit",
    "is_enrolled_at_least_half_time_for_american_opportunity_credit",
    "has_american_opportunity_credit_1098_t_or_exception",
    "has_american_opportunity_credit_institution_ein",
)
_EXPECTED_OUTPUT_COLUMNS = (
    "qualified_tuition_expenses",
    "educational_assistance",
    *_EXPECTED_AOTC_COLUMNS,
)


def _person_table(rows: list[dict]) -> pd.DataFrame:
    """Return a person table with both education source columns present."""

    records: list[dict] = []
    for index, row in enumerate(rows):
        record = {
            "ED_VAL": 0.0,
            "qualified_tuition_expenses": 0.0,
        }
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_household_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _us_frame(
    person_rows: list[dict],
    *,
    household_weights: list[float] | None = None,
) -> Frame:
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
    weights = (
        [1.0] * len(unique_households)
        if household_weights is None
        else household_weights
    )
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


def _plausible_rows() -> list[dict]:
    """100 people: 2% with tuition and 4% with assistance."""

    rows = [
        {"qualified_tuition_expenses": 1_000.0},
        {"qualified_tuition_expenses": 4_000.0},
        {"ED_VAL": 2_500.0},
        {"ED_VAL": 5_000.0},
        {"ED_VAL": 7_500.0},
        {"ED_VAL": 10_000.0},
    ]
    rows.extend({} for _ in range(94))
    return rows


def _operation():
    spec = us_education_inputs_stage_spec()
    return next(op for op in spec.operations if op.kind == "derive_education_inputs")


class TestManifestDeclaration:
    def test_stage_declares_the_complete_seven_column_family(self) -> None:
        spec = us_education_inputs_stage_spec()

        assert spec.stage == US_EDUCATION_INPUTS_STAGE_NAME == "education_inputs"
        assert US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS == _EXPECTED_AOTC_COLUMNS
        assert US_EDUCATION_INPUTS_OUTPUT_COLUMNS == _EXPECTED_OUTPUT_COLUMNS
        assert (
            US_EDUCATION_INPUTS_NONCONSTANT_PERSON_COLUMNS
            == _EXPECTED_OUTPUT_COLUMNS
        )
        assert tuple(spec.outputs) == _EXPECTED_OUTPUT_COLUMNS
        assert US_EDUCATION_INPUTS_REQUIRED_SOURCE_COLUMNS == (
            "ED_VAL",
            "qualified_tuition_expenses",
        )
        assert set(spec.nonnegative_outputs) == {
            "qualified_tuition_expenses",
            "educational_assistance",
        }

    def test_stage_reads_person_then_derives_education_inputs(self) -> None:
        kinds = [
            operation.kind for operation in us_education_inputs_stage_spec().operations
        ]
        assert kinds == ["read_table", "derive_education_inputs"]

    def test_handler_is_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers["derive_education_inputs"]
            is derive_us_education_inputs_from_manifest
        )


class TestDerivation:
    def _derive(self, table: pd.DataFrame) -> pd.DataFrame:
        return derive_us_education_inputs_from_manifest(table, _operation(), None)

    def test_preserves_upstream_tuition_and_flags_exactly_positive_tuition(
        self,
    ) -> None:
        # E03230/E87530 have already been reconciled by the PUF transformation;
        # this stage must not reinterpret or rescale their combined result.
        table = _person_table(
            [
                {"qualified_tuition_expenses": 0.0},
                {"qualified_tuition_expenses": 1_250.0},
                {"qualified_tuition_expenses": 4_000.0},
            ]
        )

        result = self._derive(table)

        assert result["qualified_tuition_expenses"].tolist() == [0.0, 1_250.0, 4_000.0]
        expected = [False, True, True]
        for column in _EXPECTED_AOTC_COLUMNS:
            assert result[column].tolist() == expected

    def test_maps_ed_val_to_educational_assistance(self) -> None:
        result = self._derive(_person_table([{"ED_VAL": 750.0}, {"ED_VAL": 0.0}]))
        assert result["educational_assistance"].tolist() == [750.0, 0.0]

    def test_amounts_are_numeric_finite_and_clipped_nonnegative(self) -> None:
        result = self._derive(
            _person_table(
                [
                    {"ED_VAL": -10.0, "qualified_tuition_expenses": -20.0},
                    {"ED_VAL": np.nan, "qualified_tuition_expenses": np.nan},
                    {"ED_VAL": "300", "qualified_tuition_expenses": "1200"},
                ]
            )
        )

        assert result["educational_assistance"].tolist() == [0.0, 0.0, 300.0]
        assert result["qualified_tuition_expenses"].tolist() == [0.0, 0.0, 1_200.0]
        for column in _EXPECTED_AOTC_COLUMNS:
            assert result[column].tolist() == [False, False, True]

    @pytest.mark.parametrize("missing", US_EDUCATION_INPUTS_REQUIRED_SOURCE_COLUMNS)
    def test_missing_source_column_is_named(self, missing: str) -> None:
        table = _person_table([{}]).drop(columns=[missing])
        with pytest.raises(SourceRuntimeError, match=missing):
            self._derive(table)

    def test_requires_person_table_first(self) -> None:
        with pytest.raises(SourceRuntimeError, match="person table"):
            derive_us_education_inputs_from_manifest(None, _operation(), None)

    def test_unexpected_parameters_are_refused(self) -> None:
        operation_spec = SourceStageSpec.from_mapping(
            {
                "stage": US_EDUCATION_INPUTS_STAGE_NAME,
                "survey": "test ASEC + PUF",
                "source": "https://example.com",
                "grain": "person",
                "operations": [
                    {"kind": "read_table", "table": "person"},
                    {"kind": "derive_education_inputs", "surprise": True},
                ],
                "outputs": list(US_EDUCATION_INPUTS_OUTPUT_COLUMNS),
            }
        )
        with pytest.raises(SourceRuntimeError, match="surprise"):
            derive_us_education_inputs_from_manifest(
                _person_table([{}]), operation_spec.operations[1], None
            )


class TestFrameIntegration:
    def test_with_inputs_writes_the_complete_family(self) -> None:
        frame = with_us_education_inputs(
            _us_frame(
                [
                    {"ED_VAL": 600.0, "qualified_tuition_expenses": 0.0},
                    {"ED_VAL": 0.0, "qualified_tuition_expenses": 2_000.0},
                ]
            ),
            seed=0,
            time_period=TIME_PERIOD,
        )
        person = frame.table("person")

        assert set(_EXPECTED_OUTPUT_COLUMNS).issubset(person.columns)
        assert person["educational_assistance"].tolist() == [600.0, 0.0]
        for column in _EXPECTED_AOTC_COLUMNS:
            assert person[column].tolist() == [False, True]

    def test_frame_with_coherent_signal_passes_through_untouched(self) -> None:
        derived = with_us_education_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        again = with_us_education_inputs(derived, seed=1, time_period=TIME_PERIOD)
        assert again is derived

    def test_incoherent_default_surface_is_healed_from_sources(self) -> None:
        constants = {
            "educational_assistance": 0.0,
            **{column: False for column in _EXPECTED_AOTC_COLUMNS},
        }
        frame = with_us_education_inputs(
            _us_frame(
                [
                    {
                        "ED_VAL": 900.0,
                        "qualified_tuition_expenses": 1_500.0,
                        **constants,
                    },
                    {**constants},
                ]
            ),
            seed=0,
            time_period=TIME_PERIOD,
        )
        person = frame.table("person")

        assert person["educational_assistance"].tolist() == [900.0, 0.0]
        for column in _EXPECTED_AOTC_COLUMNS:
            assert person[column].tolist() == [True, False]

    def test_missing_sources_without_a_complete_surface_raise(self) -> None:
        frame = _us_frame([{}])
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["person"] = tables["person"].drop(
            columns=list(US_EDUCATION_INPUTS_REQUIRED_SOURCE_COLUMNS)
        )
        stripped = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )

        with pytest.raises(SourceRuntimeError, match="ED_VAL"):
            with_us_education_inputs(stripped, seed=0, time_period=TIME_PERIOD)

    def test_puf_support_to_education_stage_preserves_real_source_signal(
        self,
    ) -> None:
        rows = [{"ED_VAL": 1_000.0} for _ in range(4)]
        rows.extend({} for _ in range(96))
        expanded = clone_us_frame_for_puf_support(_us_frame(rows))
        donor = pd.DataFrame(
            {
                "puf_predictor_tax_unit_person_count": np.ones(100),
                "qualified_tuition_expenses": [1_000.0, 4_000.0, *([0.0] * 98)],
                "weight": np.ones(100),
            }
        )

        imputed = impute_us_puf_tax_detail_support(
            expanded,
            donor,
            predictors=("puf_predictor_tax_unit_person_count",),
            person_outputs=("qualified_tuition_expenses",),
            tax_unit_outputs=(),
            seed=0,
            n_estimators=20,
        )
        person = imputed.table("person")
        channel = person[support_channel_column("person")]
        assert not person.loc[
            channel == BASE_ASEC_SUPPORT_CHANNEL,
            "qualified_tuition_expenses",
        ].any()
        assert person.loc[
            channel == PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            "qualified_tuition_expenses",
        ].gt(0).any()

        result = with_us_education_inputs(
            imputed,
            seed=0,
            time_period=TIME_PERIOD,
        )

        gate = us_education_inputs_signal_gate(result)
        assert gate.passed, gate.failures
        result_person = result.table("person")
        tuition_positive = result_person["qualified_tuition_expenses"] > 0
        for column in _EXPECTED_AOTC_COLUMNS:
            assert result_person[column].equals(tuition_positive)


class TestGate:
    def test_plausible_surface_passes(self) -> None:
        frame = with_us_education_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )

        summary = us_education_inputs_summary(frame)
        gate = us_education_inputs_signal_gate(frame)

        assert gate.passed, gate.failures
        assert gate.details == summary

    def test_missing_columns_fail(self) -> None:
        gate = us_education_inputs_signal_gate(_us_frame([{}]))
        assert not gate.passed
        assert any("missing" in failure for failure in gate.failures)

    def test_all_zero_surface_fails(self) -> None:
        rows = [
            {
                "educational_assistance": 0.0,
                **{column: False for column in _EXPECTED_AOTC_COLUMNS},
            }
            for _ in range(20)
        ]
        gate = us_education_inputs_signal_gate(_us_frame(rows))

        assert not gate.passed
        assert any(
            token in " ".join(gate.failures).lower()
            for token in ("constant", "zero", "share")
        )

    def test_flag_disagreeing_with_positive_tuition_fails(self) -> None:
        frame = with_us_education_inputs(
            _us_frame(_plausible_rows()), seed=0, time_period=TIME_PERIOD
        )
        frame.table("person").loc[
            0, "has_american_opportunity_credit_institution_ein"
        ] = False

        gate = us_education_inputs_signal_gate(frame)

        assert not gate.passed
        message = " ".join(gate.failures).lower()
        assert any(token in message for token in ("coher", "match", "disagree"))
