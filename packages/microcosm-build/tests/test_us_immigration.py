"""US SSN-card-type / immigration-status stage tests (microcosm #225)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.immigration as immigration_module
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
    US_IMMIGRATION_ALL_OUTPUT_COLUMNS,
    US_IMMIGRATION_FACT_COLUMNS,
    US_IMMIGRATION_OUTPUT_COLUMNS,
    US_IMMIGRATION_STAGE_NAME,
    US_SOURCE_MANIFEST,
    US_STAGE_NAMES,
    derive_us_immigration_facts,
    derive_us_immigration_status_from_manifest,
    us_immigration_composition_gate,
    us_immigration_composition_summary,
    us_immigration_stage_spec,
    with_us_immigration_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_HANDLERS = {"derive_immigration_status": derive_us_immigration_status_from_manifest}

TIME_PERIOD = 2024


def _stage_spec(**parameters: object) -> SourceStageSpec:
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
                    **parameters,
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
        "A_LFSR": 7,
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
    seed: int = 0,
) -> pd.DataFrame:
    return run_source_stage(
        _stage_spec(),
        tables={"person": person},
        operation_handlers=_HANDLERS,
        config=SourceRuntimeConfig(seed=seed, target_year=TIME_PERIOD),
    )


class TestAxiomFirstImmigrationFacts:
    def test_person_weights_cannot_change_assignment(
        self,
    ) -> None:
        rows = [
            _noncitizen(
                person_id=index + 1,
                PENATVTY=(200 if index % 2 else 373),
                PEINUSYR=(28 if index % 3 else 20),
                A_AGE=30,
                CAID=(1 if index % 4 else 2),
                person_weight=(1.0 if index % 2 else 1_000_000.0),
            )
            for index in range(96)
        ]
        person = _person_table(rows)
        original = _run(person, seed=37)

        reweighted = person.copy()
        reweighted["person_weight"] = np.linspace(1e-9, 1e12, len(reweighted))
        reweighted_result = _run(reweighted, seed=37)

        pd.testing.assert_frame_equal(
            original.loc[:, US_IMMIGRATION_ALL_OUTPUT_COLUMNS],
            reweighted_result.loc[:, US_IMMIGRATION_ALL_OUTPUT_COLUMNS],
        )

    def test_lpr_adjustment_preserves_refugee_and_asylum_history(self) -> None:
        refugee_rows = [
            _noncitizen(
                person_id=index + 1,
                PENATVTY=412,
                PEINUSYR=24,
                A_AGE=45,
                CAID=1,
            )
            for index in range(128)
        ]
        asylum_rows = [
            _noncitizen(
                person_id=1_000 + index,
                PENATVTY=207,
                PEINUSYR=24,
                A_AGE=45,
                CAID=1,
            )
            for index in range(128)
        ]
        output = derive_us_immigration_facts(
            _person_table(refugee_rows + asylum_rows),
            seed=19,
            time_period=TIME_PERIOD,
        )
        lpr = output["immigration_is_lawful_permanent_resident"]
        refugee = output["immigration_was_admitted_as_refugee"]
        asylum = output["immigration_has_been_granted_asylum"]

        assert (lpr & refugee).any()
        assert (lpr & asylum).any()
        assert (
            output.loc[lpr & refugee, "immigration_status_str"]
            == "LEGAL_PERMANENT_RESIDENT"
        ).all()
        assert (
            output.loc[lpr & asylum, "immigration_status_str"]
            == "LEGAL_PERMANENT_RESIDENT"
        ).all()

        refugee_adjustment = output["immigration_adjusted_to_lpr_from_refugee_status"]
        asylum_adjustment = output["immigration_adjusted_to_lpr_from_asylum_status"]
        assert refugee_adjustment.any()
        assert asylum_adjustment.any()
        assert (lpr[refugee_adjustment] & refugee[refugee_adjustment]).all()
        assert (lpr[asylum_adjustment] & asylum[asylum_adjustment]).all()

    def test_temporary_facts_can_overlap(self) -> None:
        parole_tps_rows = [
            _noncitizen(
                person_id=index + 1,
                PENATVTY=164,
                PEINUSYR=28,
                A_AGE=30,
            )
            for index in range(128)
        ]
        tps_daca_rows = [
            _noncitizen(
                person_id=1_000 + index,
                PENATVTY=373,
                PEINUSYR=20,
                A_AGE=30,
            )
            for index in range(128)
        ]
        output = derive_us_immigration_facts(
            _person_table(parole_tps_rows + tps_daca_rows),
            seed=23,
            time_period=TIME_PERIOD,
        )
        parole = output["immigration_is_currently_paroled"]
        tps = output["immigration_is_current_tps"]
        daca = output["immigration_is_current_daca"]

        assert (parole & tps).any()
        assert (tps & daca).any()
        durable = (
            output["immigration_is_lawful_permanent_resident"]
            | output["immigration_was_admitted_as_refugee"]
            | output["immigration_has_been_granted_asylum"]
            | output["immigration_has_deportation_or_removal_withheld"]
            | output["immigration_was_granted_conditional_entry"]
            | output["immigration_is_current_cuban_haitian_entrant"]
            | output["immigration_is_current_cofa_resident"]
        )
        parole_tps_only = parole & tps & ~durable
        assert parole_tps_only.any()
        assert (
            output.loc[parole_tps_only, "immigration_status_str"] == "PAROLED_ONE_YEAR"
        ).all()

    def test_frame_weights_and_legacy_weight_scale_cannot_change_facts(
        self,
    ) -> None:
        rows = [
            _noncitizen(
                person_id=index + 1,
                PENATVTY=(164 if index % 2 else 373),
                PEINUSYR=(28 if index % 2 else 20),
                A_AGE=30,
            )
            for index in range(64)
        ]
        light = _us_frame(rows, household_weights=[1.0] * len(rows))
        heavy = _us_frame(rows, household_weights=[1e9] * len(rows))
        light_result = with_us_immigration_inputs(
            light,
            seed=29,
            time_period=TIME_PERIOD,
            person_weight_scale=1e-12,
        )
        heavy_result = with_us_immigration_inputs(
            heavy,
            seed=29,
            time_period=TIME_PERIOD,
            person_weight_scale=1e12,
        )

        pd.testing.assert_frame_equal(
            light_result.table("person").loc[:, US_IMMIGRATION_FACT_COLUMNS],
            heavy_result.table("person").loc[:, US_IMMIGRATION_FACT_COLUMNS],
        )

    def test_legacy_projection_is_deterministic_and_one_way(self) -> None:
        person = _person_table(
            [
                _noncitizen(
                    person_id=index + 1,
                    PENATVTY=164,
                    PEINUSYR=28,
                    A_AGE=30,
                )
                for index in range(64)
            ]
        )
        poisoned = person.copy()
        poisoned["immigration_status_str"] = "REFUGEE"
        poisoned["ssn_card_type"] = "CITIZEN"
        expected = derive_us_immigration_facts(
            person,
            seed=41,
            time_period=TIME_PERIOD,
        )
        actual = derive_us_immigration_facts(
            poisoned,
            seed=41,
            time_period=TIME_PERIOD,
        )

        pd.testing.assert_frame_equal(
            expected.loc[:, US_IMMIGRATION_ALL_OUTPUT_COLUMNS],
            actual.loc[:, US_IMMIGRATION_ALL_OUTPUT_COLUMNS],
        )
        assert set(actual["immigration_status_str"]) <= set(IMMIGRATION_STATUS_VALUES)
        assert set(actual["ssn_card_type"]) <= set(SSN_CARD_TYPE_VALUES)

    def test_event_month_indices_stay_inside_their_event_years(self) -> None:
        person = _person_table(
            [
                _noncitizen(
                    person_id=index + 1,
                    PENATVTY=(412 if index % 2 else 207),
                    PEINUSYR=24,
                    A_AGE=45,
                    CAID=1,
                )
                for index in range(128)
            ]
        )
        output = derive_us_immigration_facts(
            person,
            seed=43,
            time_period=TIME_PERIOD,
        )
        event_pairs = (
            (
                "immigration_is_lawful_permanent_resident",
                "immigration_lpr_status_start_year",
                "immigration_lpr_status_start_month_index",
            ),
            (
                "immigration_was_admitted_as_refugee",
                "immigration_refugee_admission_year",
                "immigration_refugee_admission_month_index",
            ),
            (
                "immigration_has_been_granted_asylum",
                "immigration_asylum_grant_year",
                "immigration_asylum_grant_month_index",
            ),
            (
                "immigration_has_deportation_or_removal_withheld",
                "immigration_withholding_start_year",
                "immigration_withholding_start_month_index",
            ),
        )
        for flag_name, year_name, month_name in event_pairs:
            active = output[flag_name].to_numpy(dtype=bool)
            year = output[year_name].to_numpy(dtype=np.int64)
            month_index = output[month_name].to_numpy(dtype=np.int64)
            assert (month_index[~active] == 0).all()
            np.testing.assert_array_equal(month_index[active] // 12, year[active])
            assert ((month_index[active] % 12) <= 11).all()

        qualified = (
            output["immigration_first_qualified_status_start_year"] > 0
        ).to_numpy(dtype=bool)
        np.testing.assert_array_equal(
            output.loc[
                qualified, "immigration_first_qualified_status_start_month_index"
            ].to_numpy(dtype=np.int64)
            // 12,
            output.loc[
                qualified, "immigration_first_qualified_status_start_year"
            ].to_numpy(dtype=np.int64),
        )

    def test_status_month_imputation_is_clone_stable(self) -> None:
        rows = [
            _noncitizen(
                person_id=index + 1,
                source_year=2024,
                source_household_id="H1",
                source_person_id="P1",
                PENATVTY=412,
                PEINUSYR=24,
                A_AGE=45,
                CAID=1,
            )
            for index in range(4)
        ]
        output = derive_us_immigration_facts(
            _person_table(rows),
            seed=47,
            time_period=TIME_PERIOD,
        )
        for column in US_IMMIGRATION_FACT_COLUMNS:
            assert output[column].nunique(dropna=False) == 1

    def test_long_residence_does_not_force_lpr_clock_to_physical_entry(self) -> None:
        output = derive_us_immigration_facts(
            _person_table(
                [
                    _noncitizen(
                        person_id=index + 1,
                        PENATVTY=303,
                        PEINUSYR=5,
                        A_AGE=60,
                        CAID=1,
                    )
                    for index in range(256)
                ]
            ),
            seed=53,
            time_period=TIME_PERIOD,
        )
        lpr = output["immigration_is_lawful_permanent_resident"].to_numpy(dtype=bool)
        assert lpr.any()
        assert (
            output.loc[lpr, "immigration_lpr_status_start_year"]
            > output.loc[lpr, "immigration_us_entry_year"]
        ).any()

    def test_august_1996_cutoff_uses_imputed_month_before_day(self) -> None:
        person = _person_table([{"person_id": 1}, {"person_id": 2}])
        flag = np.array([True, True])
        start_year = np.array([1996, 1996])
        start_month_index = np.array(
            [1996 * 12 + 6, 1996 * 12 + 8],
            dtype=np.int64,
        )
        result = immigration_module._started_on_or_after_august_22_1996(
            person,
            flag,
            start_year,
            start_month_index,
            seed=59,
        )
        assert not result[0]  # July is before the cutoff regardless of day.
        assert result[1]  # September is after the cutoff regardless of day.


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
    def test_proxy_indicators_are_preserved_as_evidence_not_a_status(
        self, indicator: dict
    ) -> None:
        output = _run(_person_table([_noncitizen(**indicator)]))
        assert output.loc[0, "immigration_has_lawful_status_evidence"]
        assert output.loc[0, "immigration_is_alien"]
        assert output.loc[0, "immigration_status_str"] in IMMIGRATION_STATUS_VALUES
        assert output.loc[0, "ssn_card_type"] in SSN_CARD_TYPE_VALUES

    def test_pre_1982_arrival_does_not_observe_a_legal_status(self) -> None:
        output = _run(_person_table([_noncitizen(PEINUSYR=5, A_AGE=85)]))
        assert not output.loc[0, "immigration_has_lawful_status_evidence"]
        assert output.loc[0, "immigration_is_alien"]

    def test_absent_survey_indicators_do_not_observe_a_legal_status(self) -> None:
        output = _run(_person_table([_noncitizen()]))
        assert not output.loc[0, "immigration_has_ssn_evidence"]
        assert output.loc[0, "immigration_is_alien"]
        assert output.loc[0, "ssn_card_type"] in SSN_CARD_TYPE_VALUES
        assert output.loc[0, "immigration_status_str"] in IMMIGRATION_STATUS_VALUES

    def test_proxy_indicators_do_not_exclude_residual_support(self) -> None:
        person = _person_table(
            [_noncitizen(CAID=1, person_id=index + 1) for index in range(256)]
        )
        output = _run(person)
        assert output["immigration_has_lawful_status_evidence"].all()
        assert output["immigration_has_other_lawful_status"].any()
        status_facts = (
            "immigration_has_other_lawful_status",
            "immigration_is_lawful_permanent_resident",
            "immigration_was_admitted_as_refugee",
            "immigration_has_been_granted_asylum",
            "immigration_was_paroled",
            "immigration_has_deportation_or_removal_withheld",
            "immigration_was_granted_conditional_entry",
            "immigration_is_current_cuban_haitian_entrant",
            "immigration_is_current_cofa_resident",
            "immigration_is_current_tps",
            "immigration_is_current_daca",
            "immigration_is_amerasian_immigrant",
            "immigration_has_iraqi_or_afghan_special_immigrant_history",
            "immigration_has_qualifying_afghan_humanitarian_parole_history",
            "immigration_has_qualifying_ukrainian_humanitarian_parole_history",
            "immigration_is_battered_alien_or_qualifying_relative",
            "immigration_is_severe_trafficking_victim_or_qualifying_relative",
        )
        # The lossy PE enum alone cannot establish genuinely residual support.
        assert (~output.loc[:, status_facts].any(axis=1)).any()

    def test_prcitshp_outside_domain_raises(self) -> None:
        with pytest.raises(SourceRuntimeError, match="PRCITSHP"):
            _run(_person_table([{"PRCITSHP": 7}]))

    def test_2024_asec_rejects_nonexistent_peinusyr_29(self) -> None:
        with pytest.raises(SourceRuntimeError, match=r"PEINUSYR.*0\.\.28"):
            _run(_person_table([_noncitizen(PEINUSYR=29)]))

    def test_missing_required_column_raises(self) -> None:
        person = _person_table([_noncitizen()]).drop(columns=["PEINUSYR"])
        with pytest.raises(SourceRuntimeError, match="PEINUSYR"):
            _run(person)


class TestImmigrationStatusTags:
    def test_cuban_haitian_support_respects_arrival_bounds(self) -> None:
        person = _person_table(
            [
                _noncitizen(person_id=index + 1, PENATVTY=327, CAID=1)
                for index in range(128)
            ]
            + [
                _noncitizen(
                    person_id=index + 1_000,
                    PENATVTY=332,
                    PEINUSYR=5,
                    A_AGE=85,
                    MCARE=1,
                )
                for index in range(128)
            ]
        )
        output = _run(person)
        # Post-1980 arrival from Cuba supports CHE, but does not prove it.
        che = output["immigration_is_current_cuban_haitian_entrant"]
        assert che.iloc[:128].any()
        assert not che.iloc[128:].any()

    def test_citizens_born_in_cuba_stay_citizen(self) -> None:
        person = _person_table([{"PRCITSHP": 4, "PENATVTY": 327, "PEINUSYR": 24}])
        output = _run(person)
        assert output.loc[0, "immigration_status_str"] == "CITIZEN"

    def test_ssn_projection_is_independent_of_legal_status_projection(self) -> None:
        from microcosm.build.us_runtime.immigration import (
            _legacy_immigration_projection,
        )

        person = _person_table(
            [
                _noncitizen(),
                _noncitizen(CAID=1),
                _noncitizen(A_LFSR=1, WSAL_VAL=10_000.0),
                {"PRCITSHP": 1},
            ]
        )
        output = _run(person)
        facts = {
            column: output[column].to_numpy() for column in US_IMMIGRATION_FACT_COLUMNS
        }
        for has_ssn in (False, True):
            facts["immigration_has_ssn"] = np.full(len(output), has_ssn, dtype=bool)
            facts["immigration_ssn_is_valid_for_employment"] = np.full(
                len(output), has_ssn, dtype=bool
            )
            ssn, status = _legacy_immigration_projection(facts)
            np.testing.assert_array_equal(status, output["immigration_status_str"])
            citizen = output["immigration_is_us_citizen"].to_numpy()
            assert (
                ssn[~citizen] == ("NON_CITIZEN_VALID_EAD" if has_ssn else "NONE")
            ).all()

    def test_emitted_values_stay_inside_engine_enum_domains(self) -> None:
        person = _person_table(
            [
                _noncitizen(**overrides)
                for overrides in (
                    {},
                    {"CAID": 1},
                    {"A_LFSR": 1, "WSAL_VAL": 10_000.0},
                    {"A_HSCOL": 2},
                    {"PENATVTY": 327},
                    {
                        "PEINUSYR": 19,
                        "A_AGE": 25,
                        "A_LFSR": 1,
                        "WSAL_VAL": 5_000.0,
                    },
                )
            ]
            + [{"PRCITSHP": 1}]
        )
        output = _run(person)
        assert set(output["ssn_card_type"]) <= set(SSN_CARD_TYPE_VALUES)
        assert set(output["immigration_status_str"]) <= set(IMMIGRATION_STATUS_VALUES)


class TestDeterminism:
    def _worker_pool(self) -> pd.DataFrame:
        return _person_table(
            [_noncitizen(A_LFSR=1, WSAL_VAL=10_000.0) for _ in range(20)]
        )

    def test_partial_household_lineage_uses_complete_legacy_alternative(self) -> None:
        person = _person_table([_noncitizen(), _noncitizen()])
        person["source_year"] = [2024, 2024]
        person["source_household_id"] = [100, np.nan]
        person["source_person_id"] = [1, 2]

        partial_full = immigration_module._stable_person_draws(
            person,
            seed=17,
            salt="lineage-regression",
        )
        legacy_only = immigration_module._stable_person_draws(
            person.drop(columns="source_household_id"),
            seed=17,
            salt="lineage-regression",
        )

        np.testing.assert_array_equal(partial_full, legacy_only)

    def test_same_seed_is_bit_reproducible(self) -> None:
        first = _run(self._worker_pool(), seed=7)
        second = _run(self._worker_pool(), seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_select_different_ead_holders(self) -> None:
        first = _run(self._worker_pool(), seed=0)
        second = _run(self._worker_pool(), seed=1)
        assert not first["ssn_card_type"].equals(second["ssn_card_type"])

    def test_source_identity_keys_make_clones_consistent(self) -> None:
        rows = [
            _noncitizen(
                A_LFSR=1,
                WSAL_VAL=10_000.0,
                person_id=index + 1,
                source_year=2024,
                source_person_id=f"P{index % 10}",
            )
            for index in range(20)
        ]
        person = _person_table(rows)
        output = _run(person)
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

    def test_unexpected_parameter_is_refused(self) -> None:
        with pytest.raises(SourceRuntimeError, match="mystery_knob"):
            run_source_stage(
                _stage_spec(mystery_knob=1),
                tables={"person": _person_table([_noncitizen()])},
                operation_handlers=_HANDLERS,
                config=SourceRuntimeConfig(seed=0, target_year=TIME_PERIOD),
            )

    @pytest.mark.parametrize(
        "parameter",
        (
            "undocumented_workers",
            "undocumented_students",
            "undocumented_population_anchor",
            "humanitarian_status_stocks",
        ),
    )
    def test_population_controls_are_rejected(self, parameter: str) -> None:
        with pytest.raises(SourceRuntimeError, match=parameter):
            run_source_stage(
                _stage_spec(**{parameter: {"target": 1, "source": "test"}}),
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
        # Deliberately large survey weights do not affect support assignment.
        rows = (
            [{"PRCITSHP": 1} for _ in range(93)]
            + [_noncitizen(CAID=1) for _ in range(2)]
            + [_noncitizen(A_LFSR=1, WSAL_VAL=10_000.0) for _ in range(12)]
            + [_noncitizen() for _ in range(5)]
        )
        frame = _us_frame(rows, household_weights=[1e6] * len(rows))
        result = with_us_immigration_inputs(frame, seed=0, time_period=TIME_PERIOD)
        person = result.table("person")
        for column in US_IMMIGRATION_OUTPUT_COLUMNS:
            assert column in person.columns
        assert set(person["ssn_card_type"]) <= set(SSN_CARD_TYPE_VALUES)
        assert (person.loc[person["PRCITSHP"] == 1, "ssn_card_type"] == "CITIZEN").all()
        # The source-compatible residual cohort carries both SSN alternatives.
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


def _composition_frame() -> Frame:
    frame = _us_frame(
        [{"PRCITSHP": 1} for _ in range(20)] + [_noncitizen() for _ in range(20)]
    )
    return with_us_immigration_inputs(frame, seed=0, time_period=TIME_PERIOD)


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
    )


class TestCompositionGate:
    def test_valid_projection_passes_without_population_controls(self) -> None:
        gate = us_immigration_composition_gate(_composition_frame())

        assert gate.passed, gate.failures
        assert gate.details["target_free"] is True
        assert "controls" not in gate.details

    def test_missing_neutral_fact_is_refused(self) -> None:
        frame = _composition_frame()
        person = frame.table("person").drop(columns="immigration_has_ssn")
        gate = us_immigration_composition_gate(_replace_person(frame, person))

        assert not gate.passed
        assert any("immigration_has_ssn" in failure for failure in gate.failures)

    def test_legacy_columns_must_equal_the_fact_projection(self) -> None:
        frame = _composition_frame()
        person = frame.table("person").copy()
        person.loc[0, "ssn_card_type"] = "NONE"
        gate = us_immigration_composition_gate(_replace_person(frame, person))

        assert not gate.passed
        assert any("neutral-fact projection" in failure for failure in gate.failures)

    def test_invalid_boolean_fact_fails_without_evaluating_projection(self) -> None:
        frame = _composition_frame()
        person = frame.table("person").copy()
        person["immigration_has_ssn"] = person["immigration_has_ssn"].astype(object)
        person.loc[0, "immigration_has_ssn"] = None
        gate = us_immigration_composition_gate(_replace_person(frame, person))

        assert not gate.passed
        assert any("boolean fact domain" in failure for failure in gate.failures)

    def test_summary_is_descriptive_only(self) -> None:
        summary = us_immigration_composition_summary(_composition_frame())

        assert summary["person_population"] == 40.0
        assert summary["ssn_card_type"]["population"]["CITIZEN"] == 20.0
        assert set(summary) == {
            "person_population",
            "ssn_card_type",
            "immigration_status_str",
        }
