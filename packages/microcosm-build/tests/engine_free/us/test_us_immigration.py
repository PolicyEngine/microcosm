"""US SSN-card-type / immigration-status stage tests (microcosm #225)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeError,
    run_source_stage,
)
from microcosm.build.us_runtime import (
    IMMIGRATION_STATUS_VALUES,
    SSN_CARD_TYPE_VALUES,
    US_DONORS,
    US_IMMIGRATION_OUTPUT_COLUMNS,
    US_IMMIGRATION_STAGE_NAME,
    US_SOURCE_MANIFEST,
    US_STAGE_NAMES,
    UndocumentedControls,
    derive_us_immigration_status_from_manifest,
    us_immigration_composition_gate,
    us_immigration_composition_summary,
    us_immigration_stage_spec,
    with_us_immigration_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_HANDLERS = {"derive_immigration_status": derive_us_immigration_status_from_manifest}

TIME_PERIOD = 2024


def _stage_spec(
    *,
    workers: float,
    students: float,
    anchor: float,
) -> SourceStageSpec:
    return SourceStageSpec.from_mapping(
        {
            "stage": US_IMMIGRATION_STAGE_NAME,
            "survey": "test ASEC",
            "source": "https://example.com",
            "grain": "person",
            "operations": [
                {"kind": "read_table", "table": "person"},
                {
                    "kind": "derive_immigration_status",
                    "seed_from_build_config": True,
                    "time_period_from_build_config": True,
                    "undocumented_workers": {
                        "target": workers,
                        "source": "https://example.com/workers",
                    },
                    "undocumented_students": {
                        "target": students,
                        "source": "https://example.com/students",
                    },
                    "undocumented_population_anchor": {
                        "value": anchor,
                        "source": "https://example.com/population",
                    },
                },
            ],
            "outputs": list(US_IMMIGRATION_OUTPUT_COLUMNS),
        }
    )


def _person_table(rows: list[dict]) -> pd.DataFrame:
    """A raw-ASEC person table: baseline is a US-born citizen adult."""

    baseline = {
        "PRCITSHP": 1,
        "PEINUSYR": 0,
        "PENATVTY": 57,
        "A_AGE": 30,
        "A_MARITL": 7,
        "A_SPOUSE": 0,
        "A_HSCOL": 0,
        "WSAL_VAL": 0.0,
        "SEMP_VAL": 0.0,
        "MCARE": 2,
        "CAID": 2,
        "IHSFLG": 2,
        "CHAMPVA": 2,
        "MIL": 2,
        "PEN_SC1": 0,
        "PEN_SC2": 0,
        "RESNSS1": 0,
        "RESNSS2": 0,
        "SS_YN": 2,
        "SSI_YN": 2,
        "PEIO1COW": 0,
        "A_MJOCC": 0,
        "PEAFEVER": 2,
        "SPM_CAPHOUSESUB": 0.0,
        "person_weight": 1.0,
    }
    records = []
    for index, row in enumerate(rows):
        record = dict(baseline)
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_household_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _noncitizen(**overrides) -> dict:
    """A non-citizen row with no legal-status indicators (PEINUSYR 24 = 2015
    arrival, so an adult's age at entry is over the DACA threshold)."""

    row = {"PRCITSHP": 5, "PEINUSYR": 24, "PENATVTY": 303}
    row.update(overrides)
    return row


def _run(
    person: pd.DataFrame,
    *,
    workers: float = 100.0,
    students: float = 100.0,
    anchor: float = 10.0,
    seed: int = 0,
) -> pd.DataFrame:
    return run_source_stage(
        _stage_spec(workers=workers, students=students, anchor=anchor),
        tables={"person": person},
        operation_handlers=_HANDLERS,
        config=SourceRuntimeConfig(seed=seed, target_year=TIME_PERIOD),
    )


class TestSSNCardAssignment:
    def test_citizens_keep_citizen_regardless_of_indicators(self) -> None:
        person = _person_table(
            [
                {"PRCITSHP": 1},
                {"PRCITSHP": 2, "CAID": 1},
                {"PRCITSHP": 3, "WSAL_VAL": 50_000.0},
                {"PRCITSHP": 4, "PEINUSYR": 24},
            ]
        )
        output = _run(person)
        assert (output["ssn_card_type"] == "CITIZEN").all()
        assert (output["immigration_status_str"] == "CITIZEN").all()

    @pytest.mark.parametrize(
        "indicator",
        [
            {"PEINUSYR": 5},  # pre-1982 IRCA cohort arrival
            {"MCARE": 1},
            {"CAID": 1},
            {"IHSFLG": 1},
            {"CHAMPVA": 1},
            {"MIL": 1},
            {"PEN_SC1": 3},
            {"RESNSS2": 2},
            {"SS_YN": 1},
            {"SSI_YN": 1},
            {"PEIO1COW": 2},
            {"A_MJOCC": 11},
            {"PEAFEVER": 1},
            {"SPM_CAPHOUSESUB": 1_000.0},
        ],
    )
    def test_legal_status_indicators_mark_other_non_citizen(
        self, indicator: dict
    ) -> None:
        output = _run(_person_table([_noncitizen(**indicator)]))
        assert output.loc[0, "ssn_card_type"] == "OTHER_NON_CITIZEN"

    def test_noncitizen_without_indicators_is_undocumented(self) -> None:
        output = _run(_person_table([_noncitizen()]))
        assert output.loc[0, "ssn_card_type"] == "NONE"
        assert output.loc[0, "immigration_status_str"] == "UNDOCUMENTED"

    def test_worker_spill_leaves_undocumented_workers_at_control(self) -> None:
        person = _person_table(
            [_noncitizen(WSAL_VAL=10_000.0) for _ in range(10)]
            + [_noncitizen(), _noncitizen()]
        )
        output = _run(person, workers=4.0)
        workers = output["WSAL_VAL"] > 0
        undocumented_workers = ((output["ssn_card_type"] == "NONE") & workers).sum()
        ead_workers = (
            (output["ssn_card_type"] == "NON_CITIZEN_VALID_EAD") & workers
        ).sum()
        assert undocumented_workers == 4
        assert ead_workers == 6
        # Non-workers are untouched by the worker spill.
        assert (output.loc[~workers, "ssn_card_type"] == "NONE").all()

    def test_below_control_counts_spill_nothing(self) -> None:
        person = _person_table([_noncitizen(WSAL_VAL=10_000.0), _noncitizen(A_HSCOL=2)])
        output = _run(person, workers=50.0, students=50.0)
        assert (output["ssn_card_type"] == "NONE").all()

    def test_student_spill_leaves_undocumented_students_at_control(self) -> None:
        person = _person_table([_noncitizen(A_HSCOL=2) for _ in range(6)])
        output = _run(person, students=2.0)
        assert (output["ssn_card_type"] == "NONE").sum() == 2
        assert (output["ssn_card_type"] == "NON_CITIZEN_VALID_EAD").sum() == 4

    def test_weights_drive_the_spill_amounts(self) -> None:
        person = _person_table(
            [
                _noncitizen(WSAL_VAL=10_000.0, person_weight=6.0),
                _noncitizen(WSAL_VAL=10_000.0, person_weight=6.0),
            ]
        )
        output = _run(person, workers=6.0)
        assert set(output["ssn_card_type"]) == {"NON_CITIZEN_VALID_EAD", "NONE"}

    def test_indicator_holders_never_flip_to_undocumented(self) -> None:
        # The total undocumented population is emergent: a short count is
        # never topped up from people with legal-status indicators.
        person = _person_table(
            [_noncitizen(), _noncitizen(CAID=1, person_household_id=1)]
        )
        output = _run(person, anchor=1_000_000.0)
        assert output.loc[1, "ssn_card_type"] == "OTHER_NON_CITIZEN"

    def test_prcitshp_outside_domain_raises(self) -> None:
        with pytest.raises(SourceRuntimeError, match="PRCITSHP"):
            _run(_person_table([{"PRCITSHP": 7}]))

    def test_missing_required_column_raises(self) -> None:
        person = _person_table([_noncitizen()]).drop(columns=["PEINUSYR"])
        with pytest.raises(SourceRuntimeError, match="PEINUSYR"):
            _run(person)

    def test_missing_person_weight_raises(self) -> None:
        person = _person_table([_noncitizen()]).drop(columns=["person_weight"])
        with pytest.raises(SourceRuntimeError, match="person_weight"):
            _run(person)


class TestImmigrationStatusTags:
    def test_daca_statutory_cohort_among_ead_holders(self) -> None:
        # Arrived 2005 (code 19) aged 10 → age at entry < 16, now 29, EAD via
        # worker spill with a zero control.
        person = _person_table([_noncitizen(PEINUSYR=19, A_AGE=29, WSAL_VAL=20_000.0)])
        output = _run(person, workers=0.001)
        assert output.loc[0, "ssn_card_type"] == "NON_CITIZEN_VALID_EAD"
        assert output.loc[0, "immigration_status_str"] == "DACA"

    def test_ead_outside_daca_cohort_is_lpr(self) -> None:
        # Arrived 2015 as an adult: fails the DACA arrival test.
        person = _person_table([_noncitizen(PEINUSYR=24, A_AGE=40, WSAL_VAL=20_000.0)])
        output = _run(person, workers=0.001)
        assert output.loc[0, "ssn_card_type"] == "NON_CITIZEN_VALID_EAD"
        assert output.loc[0, "immigration_status_str"] == "LEGAL_PERMANENT_RESIDENT"

    def test_cuban_haitian_entrant_for_documented_noncitizens(self) -> None:
        person = _person_table(
            [
                _noncitizen(PENATVTY=327, CAID=1),
                _noncitizen(PENATVTY=332, PEINUSYR=5, MCARE=1),
            ]
        )
        output = _run(person)
        # Post-1980 arrival from Cuba qualifies; a pre-1980 arrival does not.
        assert output.loc[0, "immigration_status_str"] == "CUBAN_HAITIAN_ENTRANT"
        assert output.loc[1, "immigration_status_str"] == "LEGAL_PERMANENT_RESIDENT"

    def test_citizens_born_in_cuba_stay_citizen(self) -> None:
        person = _person_table([{"PRCITSHP": 4, "PENATVTY": 327, "PEINUSYR": 24}])
        output = _run(person)
        assert output.loc[0, "immigration_status_str"] == "CITIZEN"

    def test_undocumented_tag_matches_none_ssn_exactly(self) -> None:
        person = _person_table(
            [
                _noncitizen(),
                _noncitizen(CAID=1),
                _noncitizen(WSAL_VAL=10_000.0),
                {"PRCITSHP": 1},
            ]
        )
        output = _run(person, workers=0.001)
        none_ssn = output["ssn_card_type"] == "NONE"
        undocumented = output["immigration_status_str"] == "UNDOCUMENTED"
        assert (none_ssn == undocumented).all()

    def test_emitted_values_stay_inside_engine_enum_domains(self) -> None:
        person = _person_table(
            [
                _noncitizen(**overrides)
                for overrides in (
                    {},
                    {"CAID": 1},
                    {"WSAL_VAL": 10_000.0},
                    {"A_HSCOL": 2},
                    {"PENATVTY": 327},
                    {"PEINUSYR": 19, "A_AGE": 25, "WSAL_VAL": 5_000.0},
                )
            ]
            + [{"PRCITSHP": 1}]
        )
        output = _run(person, workers=0.001, students=0.001)
        assert set(output["ssn_card_type"]) <= set(SSN_CARD_TYPE_VALUES)
        assert set(output["immigration_status_str"]) <= set(IMMIGRATION_STATUS_VALUES)


class TestDeterminism:
    def _worker_pool(self) -> pd.DataFrame:
        return _person_table([_noncitizen(WSAL_VAL=10_000.0) for _ in range(20)])

    def test_same_seed_is_bit_reproducible(self) -> None:
        first = _run(self._worker_pool(), workers=10.0, seed=7)
        second = _run(self._worker_pool(), workers=10.0, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_select_different_ead_holders(self) -> None:
        first = _run(self._worker_pool(), workers=10.0, seed=0)
        second = _run(self._worker_pool(), workers=10.0, seed=1)
        assert not first["ssn_card_type"].equals(second["ssn_card_type"])

    def test_source_identity_keys_make_clones_consistent(self) -> None:
        rows = [
            _noncitizen(
                WSAL_VAL=10_000.0,
                person_id=index + 1,
                source_year=2024,
                source_person_id=f"P{index % 10}",
            )
            for index in range(20)
        ]
        person = _person_table(rows)
        output = _run(person, workers=5.0)
        by_source = output.groupby("source_person_id")["ssn_card_type"].nunique()
        assert (by_source == 1).all()


class TestManifestStage:
    def test_packaged_stage_spec_loads(self) -> None:
        stage = us_immigration_stage_spec()
        assert stage.stage == US_IMMIGRATION_STAGE_NAME
        assert tuple(stage.outputs) == US_IMMIGRATION_OUTPUT_COLUMNS
        kinds = [operation.kind for operation in stage.operations]
        assert kinds == ["read_table", "derive_immigration_status"]

    def test_stage_is_in_plan_and_donor_graph(self) -> None:
        assert US_IMMIGRATION_STAGE_NAME in US_STAGE_NAMES
        assert US_IMMIGRATION_STAGE_NAME in US_DONORS
        assert US_IMMIGRATION_STAGE_NAME in US_SOURCE_MANIFEST.stage_map()

    def test_manifest_controls_carry_citations(self) -> None:
        stage = us_immigration_stage_spec()
        derive = stage.operations[1]
        for key, value_key in (
            ("undocumented_workers", "target"),
            ("undocumented_students", "target"),
            ("undocumented_population_anchor", "value"),
        ):
            block = derive.parameters[key]
            assert float(block[value_key]) > 0
            assert str(block["source"]).startswith("https://")

    def test_unexpected_parameter_is_refused(self) -> None:
        spec = SourceStageSpec.from_mapping(
            {
                "stage": US_IMMIGRATION_STAGE_NAME,
                "survey": "test",
                "source": "https://example.com",
                "grain": "person",
                "operations": [
                    {"kind": "read_table", "table": "person"},
                    {
                        "kind": "derive_immigration_status",
                        "seed_from_build_config": True,
                        "time_period_from_build_config": True,
                        "mystery_knob": 1,
                        "undocumented_workers": {
                            "target": 1,
                            "source": "https://example.com",
                        },
                        "undocumented_students": {
                            "target": 1,
                            "source": "https://example.com",
                        },
                        "undocumented_population_anchor": {
                            "value": 1,
                            "source": "https://example.com",
                        },
                    },
                ],
                "outputs": list(US_IMMIGRATION_OUTPUT_COLUMNS),
            }
        )
        with pytest.raises(SourceRuntimeError, match="mystery_knob"):
            run_source_stage(
                spec,
                tables={"person": _person_table([_noncitizen()])},
                operation_handlers=_HANDLERS,
                config=SourceRuntimeConfig(seed=0, target_year=TIME_PERIOD),
            )

    def test_control_without_citation_is_refused(self) -> None:
        spec = SourceStageSpec.from_mapping(
            {
                "stage": US_IMMIGRATION_STAGE_NAME,
                "survey": "test",
                "source": "https://example.com",
                "grain": "person",
                "operations": [
                    {"kind": "read_table", "table": "person"},
                    {
                        "kind": "derive_immigration_status",
                        "seed_from_build_config": True,
                        "time_period_from_build_config": True,
                        "undocumented_workers": {"target": 1},
                        "undocumented_students": {
                            "target": 1,
                            "source": "https://example.com",
                        },
                        "undocumented_population_anchor": {
                            "value": 1,
                            "source": "https://example.com",
                        },
                    },
                ],
                "outputs": list(US_IMMIGRATION_OUTPUT_COLUMNS),
            }
        )
        with pytest.raises(SourceRuntimeError, match="source citation"):
            run_source_stage(
                spec,
                tables={"person": _person_table([_noncitizen()])},
                operation_handlers=_HANDLERS,
                config=SourceRuntimeConfig(seed=0, target_year=TIME_PERIOD),
            )


def _us_frame(
    person_rows: list[dict],
    *,
    household_weights: list[float] | None = None,
) -> Frame:
    person = _person_table(person_rows).drop(columns=["person_weight"])
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


class TestFrameIntegration:
    def test_with_us_immigration_inputs_writes_both_columns(self) -> None:
        # Production manifest controls are in persons; weight each household
        # at 1M persons so the Pew-scale controls bind sensibly.
        rows = (
            [{"PRCITSHP": 1} for _ in range(93)]
            + [_noncitizen(CAID=1) for _ in range(2)]
            + [_noncitizen(WSAL_VAL=10_000.0) for _ in range(12)]
            + [_noncitizen() for _ in range(5)]
        )
        frame = _us_frame(rows, household_weights=[1e6] * len(rows))
        result = with_us_immigration_inputs(frame, seed=0, time_period=TIME_PERIOD)
        person = result.table("person")
        for column in US_IMMIGRATION_OUTPUT_COLUMNS:
            assert column in person.columns
        assert set(person["ssn_card_type"]) <= set(SSN_CARD_TYPE_VALUES)
        assert (person.loc[person["PRCITSHP"] == 1, "ssn_card_type"] == "CITIZEN").all()
        # 12M weighted undocumented workers against the 8.3M Pew control:
        # some spill to EAD, the rest stay undocumented.
        assert (person["ssn_card_type"] == "NON_CITIZEN_VALID_EAD").any()
        assert (person["ssn_card_type"] == "NONE").any()

    def test_idempotent_when_columns_already_present(self) -> None:
        rows = [{"PRCITSHP": 1}, _noncitizen()]
        frame = _us_frame(rows)
        first = with_us_immigration_inputs(frame, seed=0, time_period=TIME_PERIOD)
        second = with_us_immigration_inputs(first, seed=99, time_period=TIME_PERIOD)
        pd.testing.assert_frame_equal(first.table("person"), second.table("person"))

    def test_partial_surface_is_refused(self) -> None:
        frame = _us_frame([{"PRCITSHP": 1, "ssn_card_type": "CITIZEN"}])
        with pytest.raises(ValueError, match="partial"):
            with_us_immigration_inputs(frame, seed=0, time_period=TIME_PERIOD)

    def test_missing_raw_columns_raise_loudly(self) -> None:
        frame = _us_frame([{"PRCITSHP": 1}])
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["person"] = tables["person"].drop(columns=["PRCITSHP"])
        stripped = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )
        with pytest.raises(SourceRuntimeError, match="PRCITSHP"):
            with_us_immigration_inputs(stripped, seed=0, time_period=TIME_PERIOD)


def _plausible_controls() -> UndocumentedControls:
    return UndocumentedControls(
        workers=20.0,
        students=5.0,
        population_anchor=30.0,
        sources={
            "undocumented_workers": "https://example.com/workers",
            "undocumented_students": "https://example.com/students",
            "undocumented_population_anchor": "https://example.com/population",
        },
    )


def _composition_frame(
    *,
    citizens: int = 930,
    other: int = 30,
    ead: int = 10,
    none: int = 30,
) -> Frame:
    rows: list[dict] = []
    values: list[tuple[str, str]] = (
        [("CITIZEN", "CITIZEN")] * citizens
        + [("OTHER_NON_CITIZEN", "LEGAL_PERMANENT_RESIDENT")] * other
        + [("NON_CITIZEN_VALID_EAD", "LEGAL_PERMANENT_RESIDENT")] * ead
        + [("NONE", "UNDOCUMENTED")] * none
    )
    for ssn, status in values:
        rows.append(
            {
                "PRCITSHP": 1 if ssn == "CITIZEN" else 5,
                "ssn_card_type": ssn,
                "immigration_status_str": status,
            }
        )
    return _us_frame(rows)


class TestCompositionGate:
    def test_passes_on_plausible_composition(self) -> None:
        gate = us_immigration_composition_gate(
            _composition_frame(), controls=_plausible_controls()
        )
        assert gate.passed, gate.failures

    def test_fails_when_columns_missing(self) -> None:
        gate = us_immigration_composition_gate(
            _us_frame([{"PRCITSHP": 1}]), controls=_plausible_controls()
        )
        assert not gate.passed
        assert any("missing person column" in failure for failure in gate.failures)

    def test_fails_on_the_225_failure_mode_all_citizens(self) -> None:
        rows = [
            {
                "PRCITSHP": 1,
                "ssn_card_type": "CITIZEN",
                "immigration_status_str": "CITIZEN",
            }
            for _ in range(50)
        ]
        gate = us_immigration_composition_gate(
            _us_frame(rows), controls=_plausible_controls()
        )
        assert not gate.passed
        assert any("constant" in failure for failure in gate.failures)

    def test_fails_on_values_outside_engine_enum_domain(self) -> None:
        frame = _composition_frame()
        person = frame.table("person")
        person.loc[0, "ssn_card_type"] = "5"
        gate = us_immigration_composition_gate(frame, controls=_plausible_controls())
        assert not gate.passed
        assert any("enum domain" in failure for failure in gate.failures)

    def test_fails_when_columns_disagree_about_citizenship(self) -> None:
        frame = _composition_frame()
        person = frame.table("person")
        person.loc[0, "immigration_status_str"] = "LEGAL_PERMANENT_RESIDENT"
        gate = us_immigration_composition_gate(frame, controls=_plausible_controls())
        assert not gate.passed
        assert any("citizenship" in failure for failure in gate.failures)

    def test_fails_when_undocumented_far_from_anchor(self) -> None:
        gate = us_immigration_composition_gate(
            _composition_frame(citizens=930, other=48, ead=10, none=2),
            controls=_plausible_controls(),
        )
        assert not gate.passed
        assert any("anchor" in failure for failure in gate.failures)
        assert all("weighted share" not in failure for failure in gate.failures)

    def test_fails_when_non_citizen_share_implausible(self) -> None:
        gate = us_immigration_composition_gate(
            _composition_frame(citizens=40, other=20, ead=20, none=20),
            controls=UndocumentedControls(
                workers=8.0,
                students=1.0,
                population_anchor=20.0,
                sources=_plausible_controls().sources,
            ),
        )
        assert not gate.passed
        assert any("non-citizen weighted share" in failure for failure in gate.failures)

    def test_gate_reads_packaged_controls_by_default(self) -> None:
        gate = us_immigration_composition_gate(_us_frame([{"PRCITSHP": 1}]))
        assert not gate.passed
        controls = gate.details["controls"]
        assert controls["undocumented_workers"] == 8_300_000
        assert controls["undocumented_population_anchor"] == 11_000_000

    def test_summary_reports_weighted_composition(self) -> None:
        summary = us_immigration_composition_summary(
            _composition_frame(citizens=3, other=1, ead=0, none=1)
        )
        ssn = summary["ssn_card_type"]
        assert ssn["population"]["CITIZEN"] == 3.0
        assert ssn["population"]["NONE"] == 1.0
        assert summary["person_population"] == 5.0
