from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.was_wealth import (
    REGIONS,
    UK_WAS_ENGINE_PREDICTOR_ENTITIES,
    UK_WAS_WEALTH_OUTPUT_COLUMNS,
    UKWASWealthStageTransform,
    allocate_student_loan_balance_to_people,
    clean_was_household_table,
    recipient_predictors,
    support_clip_to_donor,
)


class _FakeEngine:
    """Returns engine variables at their native entity, like the real adapter."""

    country = "uk"

    def variable_metadata(self, name):
        return SimpleNamespace(entity=UK_WAS_ENGINE_PREDICTOR_ENTITIES[name])

    def materialize(self, frame, variables, period):
        assert period == "2023"
        tables = {
            entity: frame.table(entity) for entity in ("person", "benunit", "household")
        }
        return {
            variable: tables[UK_WAS_ENGINE_PREDICTOR_ENTITIES[variable]][
                variable
            ].to_numpy()
            for variable in variables
        }


def _stage() -> SourceStageSpec:
    return SourceStageSpec.from_mapping(
        {
            "stage": "was_wealth",
            "survey": "test",
            "source": "test",
            "grain": "household",
            "artifacts": [],
            "operations": [
                {"kind": "fit_weighted_qrf_chain", "seed": 0, "n_estimators": 2}
            ],
            "outputs": list(UK_WAS_WEALTH_OUTPUT_COLUMNS),
            "nonnegative_outputs": [
                name
                for name in UK_WAS_WEALTH_OUTPUT_COLUMNS
                if name != "net_financial_wealth"
            ],
        }
    )


def _raw_was() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "R8xshhwgt": [2.0, 3.0],
            "DVLUKValR8_sum": [10.0, 100.0],
            "DVPropertyR8": [20.0, 200.0],
            "DVFESHARESR8_aggr": [1.0, 2.0],
            "DVFShUKVR8_aggr": [3.0, 4.0],
            "DVIISAVR8_aggR": [5.0, 6.0],
            "DVCISAVR8_aggr": [7.0, 8.0],
            "DVFCollVR8_aggr": [9.0, 10.0],
            "totalpenr8_aggr": [100.0, 200.0],
            "dvvaldbt_scaper8_aggr": [40.0, 50.0],
            "NumAdultR8": [2, 1],
            "NumCh18R8": [1, 0],
            "DVGIPPENR8_AGGR": [11.0, 12.0],
            "DVGISER8_AGGR": [13.0, 14.0],
            "DVGIINVR8_aggr": [15.0, 16.0],
            "DVGIEMPR8_AGGR": [17.0, 18.0],
            "HBedRmR8": [3, 4],
            "GORR8": [8, 12],
            "DVPriRntR8": [1, 2],
            "CTAmtR8": [1000.0, 1200.0],
            "HFINWNTR8_Sum": [-5.0, 50.0],
            "HFINWR8_SUM": [30.0, 40.0],
            "DVhvalueR8": [100000.0, 200000.0],
            "DVHseValR8_sum": [1000.0, 2000.0],
            "DVBlDValR8_sum": [3000.0, 4000.0],
            "DVTotinc_bhcR8": [50000.0, 60000.0],
            "DVSaValR8_aggr": [500.0, 600.0],
            "vcarnr8": [1.2, 2.8],
            "Tot_LosR8_aggr": [9000.0, 5000.0],
            "Tot_los_exc_SLCR8_aggr": [4000.0, 3000.0],
        }
    )


def _frame() -> object:
    person = pd.DataFrame(
        {
            "person_id": [101, 102, 201, 202],
            "person_benunit_id": [10, 10, 20, 20],
            "person_household_id": [1, 1, 2, 2],
            "age": [22, 45, 19, 70],
            "student_loan_repayments": [20.0, 10.0, 0.0, 0.0],
            "student_loans": [0.0, 0.0, 1.0, 0.0],
            "highest_education": ["UPPER_SECONDARY", "TERTIARY", "TERTIARY", "LOW"],
            "current_education": [
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "TERTIARY",
                "NOT_IN_EDUCATION",
            ],
            "private_pension_income": [0.5, 0.5, 2.0, 0.0],
            "employment_income": [4.0, 6.0, 12.0, 8.0],
            "self_employment_income": [0.0, 0.0, 1.0, 0.0],
            "capital_income": [1.0, 2.0, 4.0, 0.0],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [10, 20],
            "num_adults": [2, 2],
            "num_children": [0, 0],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [2.0, 3.0],
            "region": ["NORTHERN_IRELAND", "SCOTLAND"],
            "num_bedrooms": [3, 4],
            "council_tax": [1000.0, 1200.0],
            "household_net_income": [50000.0, 60000.0],
            "is_renting": [False, True],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


def test_was_donor_cleaning_arithmetic_and_exact_case_insensitive_columns() -> None:
    donor = clean_was_household_table(_raw_was())

    assert donor["stocks_and_shares_isa"].tolist() == [5.0, 6.0]
    assert donor["cash_isa"].tolist() == [7.0, 8.0]
    assert donor["corporate_wealth_excl_isa"].tolist() == [73.0, 166.0]
    assert donor["corporate_wealth"].tolist() == [78.0, 172.0]
    assert donor["student_loan_balance"].tolist() == [5000.0, 2000.0]
    assert donor["region"].tolist() == ["LONDON", "SCOTLAND"]
    assert donor["is_renting"].tolist() == [True, False]
    assert 3 not in REGIONS


def test_was_donor_missing_cash_isa_fails_closed() -> None:
    raw = _raw_was().drop(columns=["DVCISAVR8_aggr"])

    with pytest.raises(ValueError, match="DVCISAVR8_aggr"):
        clean_was_household_table(raw)


def test_recipient_predictors_remap_ni_to_wales_for_prediction_only() -> None:
    predictors = recipient_predictors(_frame(), _FakeEngine())

    assert predictors["region"].tolist() == ["WALES", "SCOTLAND"]
    assert _frame().table("household")["region"].tolist()[0] == "NORTHERN_IRELAND"


def test_recipient_predictors_sum_person_and_benunit_variables_to_household() -> None:
    predictors = recipient_predictors(_frame(), _FakeEngine())

    assert predictors["employment_income"].tolist() == [10.0, 20.0]
    assert predictors["private_pension_income"].tolist() == [1.0, 2.0]
    assert predictors["self_employment_income"].tolist() == [0.0, 1.0]
    assert predictors["capital_income"].tolist() == [3.0, 4.0]
    assert predictors["num_adults"].tolist() == [2.0, 2.0]
    assert predictors["num_children"].tolist() == [0.0, 0.0]
    assert predictors["household_net_income"].tolist() == [50000.0, 60000.0]
    assert predictors["is_renting"].tolist() == [False, True]


def test_recipient_predictors_fail_loud_on_entity_mismatch() -> None:
    class _WrongEntityEngine(_FakeEngine):
        def variable_metadata(self, name):
            if name == "employment_income":
                return SimpleNamespace(entity="household")
            return super().variable_metadata(name)

    with pytest.raises(ValueError, match="employment_income"):
        recipient_predictors(_frame(), _WrongEntityEngine())


def test_student_loan_waterfall_uses_household_ids_not_positions() -> None:
    person = _frame().table("person").copy()
    balances = pd.Series([900.0, 300.0], index=[1, 2])

    allocated = allocate_student_loan_balance_to_people(
        household_balances=balances,
        household_ids=[1, 2],
        person=person,
    )

    assert allocated.tolist() == [600.0, 300.0, 300.0, 0.0]


def test_student_loan_waterfall_exercises_equal_split_tiers() -> None:
    person = pd.DataFrame(
        {
            "person_household_id": [10, 10, 20, 20, 30, 30],
            "age": [30, 40, 16, 17, 60, 70],
            "student_loan_repayments": [0, 0, 0, 0, 0, 0],
            "student_loans": [0, 0, 0, 0, 0, 0],
            "highest_education": ["TERTIARY", "LOW", "LOW", "LOW", "LOW", "LOW"],
            "current_education": ["NONE", "NONE", "TERTIARY", "LOW", "LOW", "LOW"],
        }
    )

    allocated = allocate_student_loan_balance_to_people(
        household_balances=pd.Series([100.0, 80.0, 60.0], index=[10, 20, 30]),
        household_ids=[10, 20, 30],
        person=person,
    )

    assert allocated.tolist() == [100.0, 0.0, 80.0, 0.0, 30.0, 30.0]


def test_support_clip_and_integer_vehicle_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = clean_was_household_table(_raw_was())
    draws = pd.DataFrame(
        {column: [999999.0, -999999.0] for column in UK_WAS_WEALTH_OUTPUT_COLUMNS}
    )

    clipped = support_clip_to_donor(draws, donor)

    assert clipped["owned_land"].tolist() == [100.0, 10.0]
    assert clipped["net_financial_wealth"].tolist() == [50.0, -5.0]

    def fake_impute(*args, **kwargs):
        result = donor.loc[:, UK_WAS_WEALTH_OUTPUT_COLUMNS].reset_index(drop=True)
        result = result.copy()
        result["num_vehicles"] = [1.2, 2.8]
        return result

    import microcosm.build.uk_runtime.was_wealth as module

    monkeypatch.setattr(module, "impute_was_wealth", fake_impute)
    transformed = UKWASWealthStageTransform(
        stage=_stage(),
        engine=_FakeEngine(),
        donor=_raw_was(),
    )(_frame())

    assert transformed.table("household")["num_vehicles"].tolist() == [1, 3]
    assert "student_loan_balance" not in transformed.table("household").columns
    assert transformed.table("person")["student_loan_balance"].sum() == pytest.approx(
        7000.0
    )


def test_stage_transform_is_deterministic_with_fast_synthetic_imputer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import microcosm.build.uk_runtime.was_wealth as module

    donor = clean_was_household_table(_raw_was())
    monkeypatch.setattr(
        module,
        "impute_was_wealth",
        lambda *args, **kwargs: donor.loc[:, UK_WAS_WEALTH_OUTPUT_COLUMNS].reset_index(
            drop=True
        ),
    )
    transform = UKWASWealthStageTransform(
        stage=_stage(),
        engine=_FakeEngine(),
        donor=_raw_was(),
    )

    a = transform(_frame())
    b = transform(_frame())

    pd.testing.assert_frame_equal(a.table("household"), b.table("household"))
    pd.testing.assert_frame_equal(a.table("person"), b.table("person"))


def test_stage_transform_requires_a_tab_path_or_donor() -> None:
    transform = UKWASWealthStageTransform(stage=_stage(), engine=_FakeEngine())

    with pytest.raises(ValueError, match="caller-supplied WAS tab path"):
        transform(_frame())


def test_stage_transform_refuses_sha_mismatched_tab(tmp_path) -> None:
    tab = tmp_path / "was.tab"
    tab.write_text("\t".join(_raw_was().columns) + "\n")
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "was_wealth",
            "survey": "test",
            "source": "test",
            "grain": "household",
            "artifacts": [
                {
                    "role": "qrf_donor",
                    "kind": "private_microdata",
                    "filename": "was.tab",
                    "sha256": "0" * 64,
                    "size_bytes": tab.stat().st_size,
                    "runtime_sha256_required": True,
                }
            ],
            "operations": [
                {"kind": "fit_weighted_qrf_chain", "seed": 0, "n_estimators": 2}
            ],
            "outputs": list(UK_WAS_WEALTH_OUTPUT_COLUMNS),
        }
    )
    transform = UKWASWealthStageTransform(
        stage=stage,
        engine=_FakeEngine(),
        was_tab_path=tab,
    )

    with pytest.raises(ValueError, match="hashes to"):
        transform(_frame())


def test_was_imputer_uses_checkpointed_chain_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import microcosm.build.uk_runtime.was_wealth as module
    import microcosm.fit

    calls = []

    class FakeQRF:
        def __init__(self, *, n_estimators, seed):
            assert n_estimators == 7
            assert seed == 0

        def start_chain(self, donor, predictors, targets, *, weights):
            assert weights == "weight"
            calls.append((tuple(predictors), tuple(targets)))
            return SimpleNamespace(targets=tuple(targets), position=0)

        def fit_draw_next(
            self,
            donor,
            recipient_predictors,
            raw_prior_draws,
            *,
            state,
            weights,
        ):
            assert weights == "weight"
            target = state.targets[state.position]
            if state.position:
                previous = state.targets[state.position - 1]
                assert previous in raw_prior_draws
            return SimpleNamespace(
                target=target,
                raw_draw=np.full(len(recipient_predictors), float(state.position + 1)),
                state=SimpleNamespace(
                    targets=state.targets,
                    position=state.position + 1,
                ),
            )

    monkeypatch.setattr(microcosm.fit, "RegimeGatedQRF", FakeQRF)
    donor = clean_was_household_table(_raw_was())
    recipient = recipient_predictors(_frame(), _FakeEngine())

    result = module.impute_was_wealth(donor, recipient, seed=0, n_estimators=7)

    assert result.columns.tolist() == list(UK_WAS_WEALTH_OUTPUT_COLUMNS)
    assert calls[0][1] == ("owned_land", "property_wealth")
    assert calls[1][1] == ("corporate_wealth_excl_isa", "stocks_and_shares_isa")
    assert "corporate_wealth" in calls[2][0]
    assert calls[2][1][-1] == "cash_isa"
