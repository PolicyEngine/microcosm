"""US SSN-card-type / immigration-status stage tests (microcosm #225, #776)."""

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
    US_IMMIGRATION_OWNED_PERSON_COLUMNS,
    US_IMMIGRATION_STAGE_NAME,
    US_IMMIGRATION_YEARS_SINCE_ENTRY_COLUMN,
    US_SOURCE_MANIFEST,
    US_STAGE_NAMES,
    UndocumentedControls,
    derive_us_immigration_status_from_manifest,
    us_immigration_composition_gate,
    us_immigration_composition_summary,
    us_immigration_stage_spec,
    with_us_immigration_inputs,
)
from microcosm.build.us_runtime.immigration import (
    _asec_years_since_us_entry,
    _assign_ssn_card_codes,
    _controls_from_parameters,
    _derive_immigration_status,
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
            "outputs": list(US_IMMIGRATION_OWNED_PERSON_COLUMNS),
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
        assert tuple(stage.outputs) == US_IMMIGRATION_OWNED_PERSON_COLUMNS
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


YEARS = US_IMMIGRATION_YEARS_SINCE_ENTRY_COLUMN


def _clock(
    rows: list[dict],
    *,
    seed: int = 0,
    time_period: int = TIME_PERIOD,
) -> np.ndarray:
    """``years_since_us_entry`` of raw ASEC rows (US-born citizen baseline)."""

    return _asec_years_since_us_entry(
        _person_table(rows), seed=seed, time_period=time_period
    )


def _foreign_born_cohort(
    code: int,
    *,
    income_year: int,
    size: int = 400,
    age: int = 40,
    citizenship: int = 5,
) -> list[dict]:
    """``size`` distinct source persons sharing one PEINUSYR band."""

    return [
        _noncitizen(
            PRCITSHP=citizenship,
            PEINUSYR=code,
            A_AGE=age,
            source_year=income_year,
            source_person_id=f"P{index}",
        )
        for index in range(size)
    ]


class TestEntryClockBands:
    """PEINUSYR band -> arrival-year interval, per ASEC data dictionary."""

    @pytest.mark.parametrize(
        ("code", "first", "last"),
        [
            (2, 1950, 1959),
            (3, 1960, 1964),
            (7, 1980, 1981),
            (8, 1982, 1983),
            (19, 2004, 2005),
            (24, 2014, 2015),
            (26, 2018, 2019),
        ],
    )
    @pytest.mark.parametrize("income_year", [2022, 2023, 2024, 2025])
    def test_fixed_bands_are_the_codebook_interval_in_every_vintage(
        self, code: int, first: int, last: int, income_year: int
    ) -> None:
        years = _clock(_foreign_born_cohort(code, income_year=income_year, age=80))
        arrival = TIME_PERIOD - years
        assert set(arrival.tolist()) == set(range(first, last + 1))

    @pytest.mark.parametrize(
        ("income_year", "code", "first", "last"),
        [
            # cpsmar23.pdf (income 2022): 27 = 2020-2023, the top code.
            (2022, 27, 2020, 2023),
            # cpsmar24.pdf (income 2023): 27 = 2020-2021, 28 = 2022-2024.
            (2023, 27, 2020, 2021),
            (2023, 28, 2022, 2024),
            # cpsmar25.pdf (income 2024): 28 widens to 2022-2025.
            (2024, 27, 2020, 2021),
            (2024, 28, 2022, 2025),
            # cpsmar26.pdf (income 2025): 28 = 2022-2023, new 29 = 2024-2026.
            (2025, 27, 2020, 2021),
            (2025, 28, 2022, 2023),
            (2025, 29, 2024, 2026),
        ],
    )
    def test_top_codes_follow_each_vintages_own_codebook(
        self, income_year: int, code: int, first: int, last: int
    ) -> None:
        # Read the arrival year against a late period so no draw is clipped.
        period = 2030
        years = _clock(
            _foreign_born_cohort(code, income_year=income_year),
            time_period=period,
        )
        assert set((period - years).tolist()) == set(range(first, last + 1))

    def test_years_are_period_minus_arrival_clipped_at_zero(self) -> None:
        # ASEC 2026 code 29 (2024-2026) read against a 2024 period: arrivals
        # after the period are the most recent arrivals, zero years.
        years = _clock(_foreign_born_cohort(29, income_year=2025))
        assert set(years.tolist()) == {0.0}
        assert years.dtype == np.float64

    def test_open_before_1950_band_is_bounded_by_birth_year(self) -> None:
        # Aged 80 at the ASEC 2024 interview: born 1943 or 1944, so arrived
        # 1943-1949.
        years = _clock(_foreign_born_cohort(1, income_year=2023, age=80))
        assert set((TIME_PERIOD - years).tolist()) == set(range(1943, 1950))

    def test_band_is_bounded_below_by_birth_year(self) -> None:
        # Aged 0 at the ASEC 2024 interview: born 2023 or 2024, so not 2022.
        years = _clock(_foreign_born_cohort(28, income_year=2023, age=0))
        assert set((TIME_PERIOD - years).tolist()) == {2023, 2024}

    def test_an_age_contradicting_the_band_takes_the_bands_last_year(self) -> None:
        # Aged 1 at the ASEC 2026 interview (born 2024-2025) but reporting
        # 2022-2023: the reported band wins, at its latest year.
        years = _clock(_foreign_born_cohort(28, income_year=2025, age=1, size=20))
        assert set((TIME_PERIOD - years).tolist()) == {2023}

    def test_frame_without_source_year_reads_the_periods_income_year(self) -> None:
        # No source_year: one ASEC file of the period's income year. With a
        # 2024 period that is the 2025 survey, whose code 28 is 2022-2025.
        rows = [_noncitizen(PEINUSYR=28, person_id=index + 1) for index in range(400)]
        years = _clock(rows)
        assert set((TIME_PERIOD - years).tolist()) == {2022, 2023, 2024}
        assert set(years.tolist()) == {0.0, 1.0, 2.0}

    @pytest.mark.parametrize(
        ("row", "match"),
        [
            # NIU code 0 on a foreign-born person.
            ({"PEINUSYR": 0, "source_year": 2024}, "PEINUSYR"),
            # Code 29 does not exist in ASEC 2025 (income 2024).
            ({"PEINUSYR": 29, "source_year": 2024}, r"\(2025, 29\)"),
            # Income 2026 (ASEC 2027) has no reviewed codebook yet.
            ({"PEINUSYR": 24, "source_year": 2026}, r"\(2027, 24\)"),
            ({"PEINUSYR": 24, "source_year": np.nan}, "source_year"),
        ],
    )
    def test_unreviewed_codes_and_vintages_are_refused(
        self, row: dict, match: str
    ) -> None:
        with pytest.raises(SourceRuntimeError, match=match):
            _clock([_noncitizen(source_person_id="P1", **row)])

    def test_naturalized_citizens_get_the_entry_clock(self) -> None:
        years = _clock(_foreign_born_cohort(24, income_year=2023, citizenship=4))
        assert set(years.tolist()) == {9.0, 10.0}


class TestEntryClockStraddleRule:
    def test_a_band_straddling_five_years_lands_on_both_sides(self) -> None:
        # In 2025, code 27 (2020-2021) is 4 or 5 years: the band straddles the
        # five-year bar, and a midpoint would put it wholly on one side.
        years = _clock(
            _foreign_born_cohort(27, income_year=2024, size=1_000),
            time_period=2025,
        )
        assert set(years.tolist()) == {4.0, 5.0}
        assert 0.45 < float((years >= 5).mean()) < 0.55

    def test_2024_period_puts_no_two_year_band_across_five(self) -> None:
        clear = _clock(_foreign_born_cohort(26, income_year=2023))
        barred = _clock(_foreign_born_cohort(27, income_year=2023))
        assert set(clear.tolist()) == {5.0, 6.0}
        assert set(barred.tolist()) == {3.0, 4.0}

    def test_wide_top_band_straddles_the_one_year_window(self) -> None:
        # ASEC 2025 code 28 (2022-2025) in 2024: 0, 1 or 2 years.
        years = _clock(_foreign_born_cohort(28, income_year=2024, size=1_000))
        assert set(years.tolist()) == {0.0, 1.0, 2.0}

    def test_draws_are_seeded_and_bit_reproducible(self) -> None:
        cohort = _foreign_born_cohort(27, income_year=2024, size=200)
        first = _clock(cohort, seed=3, time_period=2025)
        np.testing.assert_array_equal(first, _clock(cohort, seed=3, time_period=2025))
        assert not np.array_equal(first, _clock(cohort, seed=4, time_period=2025))

    def test_support_clones_share_one_entry_year(self) -> None:
        rows = [
            _noncitizen(
                PEINUSYR=27,
                source_year=2024,
                source_person_id=f"P{index % 25}",
                person_id=index + 1,
            )
            for index in range(100)
        ]
        person = _person_table(rows)
        person[YEARS] = _asec_years_since_us_entry(person, seed=0, time_period=2025)
        assert (person.groupby("source_person_id")[YEARS].nunique() == 1).all()


class TestEntryClockUsBorn:
    @pytest.mark.parametrize("citizenship", [1, 2, 3])
    def test_us_born_clock_is_age_whatever_peinusyr_says(
        self, citizenship: int
    ) -> None:
        rows = [
            {"PRCITSHP": citizenship, "A_AGE": 7, "PEINUSYR": 0},
            {"PRCITSHP": citizenship, "A_AGE": 64, "PEINUSYR": 24},
            {"PRCITSHP": citizenship, "A_AGE": 85, "PEINUSYR": 99},
        ]
        np.testing.assert_array_equal(_clock(rows), [7.0, 64.0, 85.0])


class TestEntryClockStageOutput:
    def _mixed_population(self) -> pd.DataFrame:
        codes = [5, 19, 24, 26, 27, 28, 20, 27, 28]
        rows = (
            [{"PRCITSHP": 1, "A_AGE": 30 + index} for index in range(5)]
            + [{"PRCITSHP": 4, "PEINUSYR": 19, "A_AGE": 50, "source_year": 2023}]
            + [
                _noncitizen(
                    PEINUSYR=code,
                    WSAL_VAL=10_000.0 * (index % 2),
                    A_HSCOL=2 if index % 3 == 0 else 0,
                    CAID=1 if index % 4 == 0 else 2,
                    source_year=2023 + index % 3,
                    source_person_id=f"P{index}",
                )
                for index, code in enumerate(codes)
            ]
            + [_noncitizen(PENATVTY=327, CAID=1, PEINUSYR=27, source_year=2024)]
        )
        return _person_table(rows)

    def test_asec_labels_and_columns_are_otherwise_byte_identical(self) -> None:
        person = self._mixed_population()
        output = _run(person.copy(deep=True), workers=2.0, students=1.0)

        # The kernels that write the two labels are untouched, so running them
        # directly reproduces the stage's output before the clock.
        controls = _controls_from_parameters(
            _stage_spec(workers=2.0, students=1.0, anchor=10.0).operations[1].parameters
        )
        codes = _assign_ssn_card_codes(
            person,
            person["person_weight"].to_numpy(dtype=np.float64),
            seed=0,
            controls=controls,
        )
        expected = person.copy(deep=True)
        expected["ssn_card_type"] = pd.Series(codes, index=person.index).map(
            {
                0: "NONE",
                1: "CITIZEN",
                2: "NON_CITIZEN_VALID_EAD",
                3: "OTHER_NON_CITIZEN",
            }
        )
        expected["immigration_status_str"] = _derive_immigration_status(
            person, codes, time_period=TIME_PERIOD
        )
        pd.testing.assert_frame_equal(output.drop(columns=[YEARS]), expected)
        assert list(output.columns) == [*expected.columns, YEARS]
        assert output[YEARS].dtype == np.float64

    def test_stage_clock_is_the_clock_kernel_for_every_seed(self) -> None:
        person = self._mixed_population()
        for seed in (0, 11):
            output = _run(person.copy(deep=True), workers=2.0, students=1.0, seed=seed)
            np.testing.assert_array_equal(
                output[YEARS].to_numpy(),
                _asec_years_since_us_entry(person, seed=seed, time_period=TIME_PERIOD),
            )

    def test_stage_clock_is_complete_and_non_negative(self) -> None:
        output = _run(self._mixed_population(), workers=2.0, students=1.0)
        years = output[YEARS].to_numpy()
        assert np.isfinite(years).all()
        assert (years >= 0).all()
        us_born = output["PRCITSHP"].isin([1, 2, 3]).to_numpy()
        np.testing.assert_array_equal(
            years[us_born], output.loc[us_born, "A_AGE"].to_numpy(dtype=float)
        )


class TestEntryClockFrameIntegration:
    def test_with_us_immigration_inputs_writes_the_clock(self) -> None:
        rows = [{"PRCITSHP": 1, "A_AGE": 44}, _noncitizen(PEINUSYR=26)]
        result = with_us_immigration_inputs(
            _us_frame(rows), seed=0, time_period=TIME_PERIOD
        )
        person = result.table("person")
        assert list(person.columns[-3:]) == list(US_IMMIGRATION_OWNED_PERSON_COLUMNS)
        assert person.loc[0, YEARS] == 44.0
        assert person.loc[1, YEARS] in {5.0, 6.0}

    def test_a_labelled_base_without_the_clock_passes_through(self) -> None:
        rows = [
            {
                "PRCITSHP": 1,
                "ssn_card_type": "CITIZEN",
                "immigration_status_str": "CITIZEN",
            }
        ]
        frame = _us_frame(rows)
        result = with_us_immigration_inputs(frame, seed=0, time_period=TIME_PERIOD)
        assert result is frame
        assert YEARS not in result.table("person").columns

    def test_a_clock_without_labels_is_refused(self) -> None:
        frame = _us_frame([{"PRCITSHP": 1, YEARS: 30.0}])
        with pytest.raises(ValueError, match="partial"):
            with_us_immigration_inputs(frame, seed=0, time_period=TIME_PERIOD)


def _clocked_composition_frame(lpr_years: list[float]) -> Frame:
    frame = _composition_frame(citizens=930, other=len(lpr_years), ead=10, none=30)
    person = frame.table("person")
    years = np.full(len(person), 40.0)
    other = (person["ssn_card_type"] == "OTHER_NON_CITIZEN").to_numpy()
    years[other] = lpr_years
    person[YEARS] = years
    return frame


def _set_first_nan(years: np.ndarray) -> None:
    years[0] = np.nan


def _set_first_negative(years: np.ndarray) -> None:
    years[0] = -1.0


def _set_all_default(years: np.ndarray) -> None:
    years[:] = 5.0


class TestEntryClockSummaryAndGate:
    def test_summary_reports_the_lpr_share_under_five_years(self) -> None:
        frame = _clocked_composition_frame([0.0, 3.0, 4.0, 5.0] + [12.0] * 26)
        summary = us_immigration_composition_summary(frame)[YEARS]
        lpr = summary["by_status"]["LEGAL_PERMANENT_RESIDENT"]
        # 30 OTHER_NON_CITIZEN rows (clock as given) and 10 EAD holders
        # (clock 40) carry the LPR label.
        assert lpr["population"] == 40.0
        assert lpr["under_5_years_population"] == 3.0
        assert lpr["under_5_years_share"] == pytest.approx(3.0 / 40.0)
        assert lpr["bands"] == {"0-4": 3.0, "5-9": 1.0, "10-19": 26.0, "20+": 10.0}
        assert summary["non_citizen"]["population"] == 70.0
        assert summary["bar_years"] == 5.0

    def test_summary_is_none_without_the_clock(self) -> None:
        summary = us_immigration_composition_summary(_composition_frame())
        assert summary[YEARS] is None

    def test_gate_passes_with_a_derived_clock(self) -> None:
        gate = us_immigration_composition_gate(
            _clocked_composition_frame([2.0] * 10 + [15.0] * 20),
            controls=_plausible_controls(),
        )
        assert gate.passed, gate.failures
        assert gate.details["summary"][YEARS]["non_citizen"]["population"] == 70.0

    def test_gate_tolerates_a_pre_776_frame_without_the_clock(self) -> None:
        gate = us_immigration_composition_gate(
            _composition_frame(), controls=_plausible_controls()
        )
        assert gate.passed, gate.failures

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            (_set_first_nan, "no finite value"),
            (_set_first_negative, "negative"),
            (_set_all_default, "engine default"),
        ],
    )
    def test_gate_fails_on_an_incomplete_negative_or_default_clock(
        self, change, match: str
    ) -> None:
        frame = _clocked_composition_frame([2.0] * 10 + [15.0] * 20)
        person = frame.table("person")
        years = person[YEARS].to_numpy(dtype=np.float64, copy=True)
        change(years)
        person[YEARS] = years
        gate = us_immigration_composition_gate(frame, controls=_plausible_controls())
        assert not gate.passed
        assert any(match in failure for failure in gate.failures)
