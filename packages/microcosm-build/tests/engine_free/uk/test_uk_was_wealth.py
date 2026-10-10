from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.tenure_constants import UK_TENURE_CATEGORIES
from microcosm.build.uk_runtime.was_wealth import (
    REGIONS,
    UK_WAS_DERIVED_TOTALS,
    UK_WAS_DRAWN_ONLY_COLUMNS,
    UK_WAS_ENGINE_PREDICTOR_ENTITIES,
    UK_WAS_ENGINE_PREDICTORS,
    UK_WAS_NET_FINANCIAL_LIABILITIES,
    UK_WAS_STRATIFIED_TARGETS,
    UK_WAS_TENURE_CATEGORY_COLUMN,
    UK_WAS_TENURE_PREDICTORS,
    UK_WAS_WEALTH_OUTPUT_COLUMNS,
    UK_WAS_WEALTH_PREDICTORS,
    UKWASWealthStageTransform,
    allocate_student_loan_balance_to_people,
    cap_derived_totals_to_donor_range,
    clean_was_household_table,
    impute_was_wealth,
    recipient_predictors,
    recipient_tenure_category,
    support_clip_to_donor,
    tenure_coherence_receipt,
    was_tenure_category,
    wealth_identity_violations,
)

_DRAW_COLUMNS = [*UK_WAS_WEALTH_OUTPUT_COLUMNS, *UK_WAS_DRAWN_ONLY_COLUMNS]


def _chain_operation(n_estimators: int = 2) -> dict[str, object]:
    return {
        "kind": "fit_weighted_qrf_chain",
        "seed": 0,
        "n_estimators": n_estimators,
        "predictors": list(UK_WAS_WEALTH_PREDICTORS),
        "stratified_targets": {
            target: list(categories)
            for target, categories in UK_WAS_STRATIFIED_TARGETS.items()
        },
        "derived_totals": {
            **{
                total: list(components)
                for total, components in UK_WAS_DERIVED_TOTALS.items()
            },
            "net_financial_wealth": [
                "gross_financial_wealth",
                *(f"-{column}" for column in UK_WAS_NET_FINANCIAL_LIABILITIES),
            ],
        },
    }


def _materialize_operation() -> dict[str, object]:
    return {
        "kind": "materialize_rules_engine_predictors",
        "predictors": list(UK_WAS_ENGINE_PREDICTORS),
        "consumed_only": True,
    }


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
            "operations": [_materialize_operation(), _chain_operation()],
            "outputs": list(UK_WAS_WEALTH_OUTPUT_COLUMNS),
            "nonnegative_outputs": [
                name
                for name in UK_WAS_WEALTH_OUTPUT_COLUMNS
                if name != "net_financial_wealth"
            ],
        }
    )


def _raw_was() -> pd.DataFrame:
    """Two donors that satisfy the round-8 identities: a mortgaged owner and a
    social renter. Property is the sum of the property values plus 90 / 400 of
    other property; gross financial wealth is the listed assets plus 5 / 14."""

    return pd.DataFrame(
        {
            "R8xshhwgt": [2.0, 3.0],
            "DVLUKValR8_sum": [10.0, 100.0],
            "DVPropertyR8": [104_100.0, 6_500.0],
            "DVFESHARESR8_aggr": [1.0, 2.0],
            "DVFShUKVR8_aggr": [3.0, 4.0],
            "DVIISAVR8_aggR": [5.0, 6.0],
            "DVCISAVR8_aggr": [7.0, 8.0],
            "DVFCollVR8_aggr": [9.0, 10.0],
            "totalpenr8_aggr": [100.0, 200.0],
            "dvvaldbt_scaper8_aggr": [40.0, 50.0],
            "NumAdultR8": [2, 1],
            "NumChildR8": [1, 0],
            "DVGIPPENR8_AGGR": [11.0, 12.0],
            "DVGISER8_AGGR": [13.0, 14.0],
            "DVGIINVR8_aggr": [15.0, 16.0],
            "DVGIEMPR8_AGGR": [17.0, 18.0],
            "HBedRmR8": [3, 4],
            "GORR8": [8, 12],
            "DVPriRntR8": [-9, 2],
            "DVCTaxAmtAnnualR8": [1000.0, 1200.0],
            "DVNetRentAmtAnnualR8_aggr": [0.0, 300.0],
            "HFINWNTR8_Sum": [-4_480.0, -1_356.0],
            "HFINWNTR8_exSLC_Sum": [520.0, 644.0],
            "HFINWR8_SUM": [530.0, 644.0],
            "HMortGR8": [1000.0, 0.0],
            "TotMortR8": [1000.0, 0.0],
            "OthMortR8_sum": [0.0, 0.0],
            "Ten1R8": [2, 4],
            "DVhvalueR8": [100_000.0, 0.0],
            "DVHseValR8_sum": [1000.0, 2000.0],
            "DVBltValR8_sum": [0.0, 300.0],
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
            "property_income": [0.0, 100.0, 50.0, 0.0],
            # The 19-year-old is a dependent child of the benefit unit.
            "is_uc_claimant": [True, True, False, True],
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
            "council_tax_rebate": [250.0, 0.0],
            "household_net_income": [50000.0, 60000.0],
            "tenure_type": ["OWNED_WITH_MORTGAGE", "OWNED_OUTRIGHT"],
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
    # Private pension wealth (total pensions less current DB) is its own
    # output; corporate_wealth keeps only the share-like holdings.
    assert donor["private_pension_wealth"].tolist() == [60.0, 150.0]
    assert donor["corporate_wealth_excl_isa"].tolist() == [13.0, 16.0]
    assert donor["corporate_wealth"].tolist() == [18.0, 22.0]
    assert donor["student_loan_balance"].tolist() == [5000.0, 2000.0]
    assert donor["mortgage_debt"].tolist() == [1000.0, 0.0]
    assert donor["consumer_debt"].tolist() == [10.0, 0.0]
    assert donor[UK_WAS_TENURE_CATEGORY_COLUMN].tolist() == [
        "owned_with_mortgage",
        "social_rent",
    ]
    assert donor["tenure_owned_with_mortgage"].tolist() == [True, False]
    assert donor["tenure_social_rent"].tolist() == [False, True]
    assert donor["tenure_private_rent"].tolist() == [False, False]
    # The remainders that make each total the sum of its drawn components.
    # Buy-to-let joins the other houses in other residential property, so
    # the undrawn remainder shrinks by it (microcosm#1095).
    assert donor["other_residential_property_value"].tolist() == [1000.0, 2300.0]
    assert donor["other_property_value"].tolist() == [90.0, 100.0]
    assert donor["other_financial_assets"].tolist() == [5.0, 14.0]
    assert donor["main_residence_mortgage"].tolist() == [1000.0, 0.0]
    assert donor["other_mortgage"].tolist() == [0.0, 0.0]
    assert donor["region"].tolist() == ["LONDON", "SCOTLAND"]
    # The private-renter flag the Lifetime ISA stage reads on both sides.
    assert donor["is_renting"].tolist() == [False, False]
    assert 3 not in REGIONS


def test_was_tenure_category_reads_the_round_8_codes() -> None:
    codes = pd.Series([1, 2, 3, 4, 4, 5, 5, 6, -8])
    private = pd.Series([-9, -9, -9, 1, 2, 1, 2, -9, -9])

    assert was_tenure_category(codes, private).tolist() == [
        "owned_outright",
        "owned_with_mortgage",
        "owned_with_mortgage",
        "private_rent",
        "social_rent",
        "private_rent",
        "social_rent",
        # Codes outside 1-5 take the FRS spine's own fallback and are counted.
        "private_rent",
        "private_rent",
    ]
    raw = pd.concat([_raw_was()] * 2, ignore_index=True)
    raw.loc[3, "Ten1R8"] = 6
    donor = clean_was_household_table(raw)
    assert donor["tenure_code_unclassified"].tolist() == [False, False, False, True]


def test_recipient_tenure_category_groups_council_and_housing_association() -> None:
    tenure = pd.Series(
        [
            "OWNED_OUTRIGHT",
            "OWNED_WITH_MORTGAGE",
            "RENT_PRIVATELY",
            "RENT_FROM_COUNCIL",
            "RENT_FROM_HA",
        ]
    )

    assert recipient_tenure_category(tenure).tolist() == [
        "owned_outright",
        "owned_with_mortgage",
        "private_rent",
        "social_rent",
        "social_rent",
    ]
    with pytest.raises(ValueError, match="outside the engine's enum"):
        recipient_tenure_category(pd.Series(["RENT_FREE"]))


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("DVPropertyR8", 1.0, "property_wealth is below the sum"),
        ("HFINWR8_SUM", 1.0, "gross_financial_wealth is below the sum"),
        ("HFINWNTR8_Sum", 0.0, "net financial wealth is not gross"),
    ],
)
def test_was_donor_breaking_an_identity_is_refused(column, value, match) -> None:
    raw = _raw_was()
    raw.loc[0, column] = value

    with pytest.raises(ValueError, match=match):
        clean_was_household_table(raw)


def test_was_donor_sentinel_codes_recode_to_zero_for_nonnegative_domains() -> None:
    raw = _raw_was()
    raw.loc[0, "vcarnr8"] = -8
    raw.loc[1, "HBedRmR8"] = -8
    # -8 is a legal net financial wealth value; the liabilities move with it
    # so the row still satisfies the identity.
    raw.loc[0, "HFINWNTR8_Sum"] = -8.0
    raw.loc[0, "Tot_LosR8_aggr"] = raw.loc[0, "Tot_los_exc_SLCR8_aggr"] + 528.0

    donor = clean_was_household_table(raw)

    assert donor["num_vehicles"].tolist()[0] == 0
    assert donor["num_bedrooms"].tolist()[1] == 0
    # Genuinely negative domains are never recoded.
    assert donor["net_financial_wealth"].tolist()[0] == -8.0


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
    # Adults and children are the FRS family roles, as WAS counts them, so
    # the 19-year-old dependant is a child (microcosm#1095).
    assert predictors["num_adults"].tolist() == [2.0, 1.0]
    assert predictors["num_children"].tolist() == [0.0, 1.0]
    assert predictors["rental_income"].tolist() == [100.0, 50.0]
    assert predictors["household_net_income"].tolist() == [50000.0, 60000.0]
    # The donor records the bill paid, so the reduction comes off the
    # liability before council tax reduction (microcosm#1095).
    assert predictors["council_tax"].tolist() == [750.0, 1200.0]
    assert predictors[UK_WAS_TENURE_CATEGORY_COLUMN].tolist() == [
        "owned_with_mortgage",
        "owned_outright",
    ]
    assert predictors["tenure_owned_with_mortgage"].tolist() == [True, False]
    assert predictors["tenure_private_rent"].tolist() == [False, False]
    assert predictors["tenure_social_rent"].tolist() == [False, False]
    # The stage reads no engine tenure formula.
    assert "is_renting" not in predictors
    assert "is_renting" not in UK_WAS_ENGINE_PREDICTOR_ENTITIES


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
        {column: [9_999_999.0, -9_999_999.0] for column in UK_WAS_WEALTH_OUTPUT_COLUMNS}
    )

    clip_result = support_clip_to_donor(draws, donor)
    clipped = clip_result.clipped

    assert clipped["owned_land"].tolist() == [100.0, 10.0]
    assert clipped["net_financial_wealth"].tolist() == [-1_356.0, -4_480.0]
    receipt = clip_result.receipt.evidence()["columns"]
    assert receipt["owned_land"] == {
        "donor_min": 10.0,
        "donor_max": 100.0,
        "clipped_low_rows": 1,
        "clipped_high_rows": 1,
        "rows_considered": 2,
    }
    assert receipt["net_financial_wealth"] == {
        "donor_min": -4_480.0,
        "donor_max": -1_356.0,
        "clipped_low_rows": 1,
        "clipped_high_rows": 1,
        "rows_considered": 2,
    }

    import microcosm.build.uk_runtime.was_wealth as module

    def fake_impute(*args, **kwargs):
        result = donor.loc[:, _DRAW_COLUMNS].reset_index(drop=True)
        result = result.copy()
        result["num_vehicles"] = [1.2, 2.8]
        return module.UKWASWealthImputationResult(
            draws=result,
            fit_weight_records=(
                module.FitWeightRecord("uk_was_2018_20_wealth:test", "explicit"),
            ),
        )

    monkeypatch.setattr(module, "impute_was_wealth", fake_impute)
    transform = UKWASWealthStageTransform(
        stage=_stage(),
        engine=_FakeEngine(),
        donor=_raw_was(),
    )
    assert transform.fit_weight_records == ()
    transformed = transform(_frame())

    assert transform.fit_weight_records == (
        module.FitWeightRecord("uk_was_2018_20_wealth:test", "explicit"),
    )
    assert transformed.table("household")["num_vehicles"].tolist() == [1, 3]
    # The stage records its household-mass conservation receipt: the terminal
    # family gate asserts positively that the weights passed through.
    receipt = transformed.mass_log[-1]
    assert receipt.reason == module.UK_WAS_WEALTH_MASS_CONSERVATION_REASON
    assert receipt.entity == "household"
    assert receipt.old_total == receipt.new_total > 0
    assert receipt.declared_factor == 1.0
    assert transform.last_result is not None
    assert transform.checkpoint_metadata()["evidence"]["support_clip"]["columns"][
        "num_vehicles"
    ] == {
        "donor_min": 1.2,
        "donor_max": 2.8,
        "clipped_low_rows": 0,
        "clipped_high_rows": 0,
        "rows_considered": 2,
    }
    evidence = transform.checkpoint_metadata()["evidence"]
    # The drawn-only components never reach the frame.
    assert not set(UK_WAS_DRAWN_ONLY_COLUMNS) & set(
        transformed.table("household").columns
    )
    assert evidence["identities"] == {
        "property_wealth_violation_rows": 0,
        "corporate_wealth_violation_rows": 0,
        "gross_financial_wealth_violation_rows": 0,
        "mortgage_debt_violation_rows": 0,
        "net_financial_wealth_violation_rows": 0,
        "property_wealth_capped_rows": 0,
        "corporate_wealth_capped_rows": 0,
        "gross_financial_wealth_capped_rows": 0,
        "mortgage_debt_capped_rows": 0,
        "net_financial_wealth_capped_high_rows": 0,
        "net_financial_wealth_capped_low_rows": 0,
    }
    tenure = evidence["tenure_coherence"]
    assert tenure["main_residence_mortgage_off_mortgaged_tenure_rows"] == 0
    assert tenure["main_residence_value_off_owner_tenure_rows"] == 0
    # The fake imputer hands the outright owner the renter donor's row.
    assert tenure["owner_rows_without_main_residence_value"] == 1
    assert tenure["donor_rows_by_tenure"] == {
        "owned_outright": 0,
        "owned_with_mortgage": 1,
        "private_rent": 0,
        "social_rent": 1,
    }
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
        lambda *args, **kwargs: module.UKWASWealthImputationResult(
            draws=donor.loc[:, _DRAW_COLUMNS].reset_index(drop=True),
            fit_weight_records=(),
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
                    "role": "was_qrf_donor",
                    "kind": "private_microdata",
                    "filename": "was.tab",
                    "sha256": "0" * 64,
                    "size_bytes": tab.stat().st_size,
                    "runtime_sha256_required": True,
                }
            ],
            "operations": [_chain_operation()],
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
    donors = []
    seeds = []

    class FakeQRF:
        def __init__(self, *, n_estimators, seed):
            assert n_estimators == 7
            seeds.append(seed)

        def start_chain(self, donor, predictors, targets, *, weights):
            assert weights == "weight"
            calls.append((tuple(predictors), tuple(targets)))
            donors.append(len(donor))
            return SimpleNamespace(
                targets=tuple(targets), position=0, weight_kind="explicit"
            )

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
            assert raw_prior_draws.index.equals(recipient_predictors.index)
            return SimpleNamespace(
                target=target,
                raw_draw=np.full(len(recipient_predictors), float(state.position + 1)),
                weight_kind="explicit",
                state=SimpleNamespace(
                    targets=state.targets,
                    position=state.position + 1,
                    weight_kind="explicit",
                ),
            )

    monkeypatch.setattr(microcosm.fit, "RegimeGatedQRF", FakeQRF)
    donor = clean_was_household_table(_raw_was())
    recipient = recipient_predictors(_frame(), _FakeEngine())

    result = module.impute_was_wealth(donor, recipient, seed=0, n_estimators=7)

    assert result.draws.columns.tolist() == _DRAW_COLUMNS
    assert [targets for _, targets in calls] == [
        ("owned_land",),
        ("main_residence_value",),
        (
            "other_residential_property_value",
            "non_residential_property_value",
            "other_property_value",
        ),
        (
            "private_pension_wealth",
            "corporate_wealth_excl_isa",
            "stocks_and_shares_isa",
        ),
        (
            "savings",
            "cash_isa",
            "other_financial_assets",
            "num_vehicles",
            "student_loan_balance",
        ),
        ("main_residence_mortgage",),
        ("other_mortgage",),
        ("consumer_debt",),
    ]
    base = calls[0][0]
    assert set(UK_WAS_TENURE_PREDICTORS) <= set(base)
    assert "is_renting" not in base
    assert calls[1][0] == (*base, "owned_land")
    assert calls[2][0] == (*base, "owned_land", "main_residence_value")
    assert calls[3][0] == (*base, "owned_land", "property_wealth")
    assert "private_pension_wealth" in calls[4][0]
    assert "corporate_wealth" in calls[4][0]
    prior = tuple(
        column
        for column in UK_WAS_WEALTH_OUTPUT_COLUMNS
        if column not in ("mortgage_debt", "consumer_debt", "net_financial_wealth")
    )
    assert calls[5][0] == (*base, *prior)
    assert calls[6][0] == (*base, *prior, "main_residence_mortgage")
    assert calls[7][0] == (*base, *prior, "mortgage_debt")
    # The stratified targets are fitted on their stratum's donors only: the one
    # mortgaged owner of the two donors; the other mortgages on both.
    assert donors == [2, 1, 2, 2, 2, 1, 2, 2]
    # Both recipients are owners, one of them mortgaged: the outright owner
    # holds no main-residence mortgage by rule but may hold other mortgages,
    # and mortgage_debt is the sum of the two.
    assert result.draws["main_residence_value"].tolist() == [1.0, 1.0]
    assert result.draws["main_residence_mortgage"].tolist() == [1.0, 0.0]
    assert result.draws["other_mortgage"].tolist() == [1.0, 1.0]
    assert result.draws["mortgage_debt"].tolist() == [2.0, 1.0]
    # Totals are sums of the drawn components, never draws of their own.
    assert result.draws["property_wealth"].tolist() == [8.0, 8.0]
    assert result.draws["corporate_wealth"].tolist() == [5.0, 5.0]
    assert result.draws["gross_financial_wealth"].tolist() == [11.0, 11.0]
    assert result.draws["net_financial_wealth"].tolist() == [5.0, 5.0]
    fitted_targets = [name for _, targets in calls for name in targets]
    assert [record.fit_name for record in result.fit_weight_records] == [
        f"uk_was_2018_20_wealth:{target}" for target in fitted_targets
    ]
    assert {record.weight_kind for record in result.fit_weight_records} == {"explicit"}
    # One independent RNG root per segment, derived from the declared seed.
    assert seeds == list(module.was_wealth_segment_seeds(0))
    assert len(set(seeds)) == module.UK_WAS_CHAIN_SEGMENTS == 8
    assert result.segment_seeds == tuple(seeds)


def test_stratified_target_refuses_a_donor_without_the_stratum() -> None:
    raw = _raw_was()
    raw.loc[0, ["Ten1R8", "HMortGR8", "TotMortR8"]] = [1, 0.0, 0.0]
    donor = clean_was_household_table(raw)
    recipient = recipient_predictors(_frame(), _FakeEngine())

    with pytest.raises(ValueError, match="no household in the tenure stratum"):
        impute_was_wealth(donor, recipient, seed=0, n_estimators=2)


def test_chain_declaration_drift_is_refused() -> None:
    import microcosm.build.uk_runtime.was_wealth as module

    operation = _chain_operation()
    operation["stratified_targets"] = {"mortgage_debt": ["owned_with_mortgage"]}
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "was_wealth",
            "survey": "test",
            "source": "test",
            "grain": "household",
            "artifacts": [],
            "operations": [_materialize_operation(), operation],
            "outputs": list(UK_WAS_WEALTH_OUTPUT_COLUMNS),
        }
    )

    with pytest.raises(ValueError, match="stratified_targets drifted"):
        module._assert_chain_declaration(stage)
    operation = _chain_operation()
    operation["derived_totals"].pop("net_financial_wealth")
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "was_wealth",
            "survey": "test",
            "source": "test",
            "grain": "household",
            "artifacts": [],
            "operations": [_materialize_operation(), operation],
            "outputs": list(UK_WAS_WEALTH_OUTPUT_COLUMNS),
        }
    )
    with pytest.raises(ValueError, match="derived_totals drifted"):
        module._assert_chain_declaration(stage)
    operation = _chain_operation()
    operation["predictors"] = [
        name for name in UK_WAS_WEALTH_PREDICTORS if name != "rental_income"
    ]
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "was_wealth",
            "survey": "test",
            "source": "test",
            "grain": "household",
            "artifacts": [],
            "operations": [_materialize_operation(), operation],
            "outputs": list(UK_WAS_WEALTH_OUTPUT_COLUMNS),
        }
    )
    with pytest.raises(ValueError, match="predictors drifted"):
        module._assert_chain_declaration(stage)


def test_private_pension_wealth_split_preserves_the_old_corporate_wealth_identity() -> (
    None
):
    """The pension component plus the new corporate_wealth reproduces the
    pre-split corporate_wealth row for row, and the pension component is
    exactly total pensions less current defined-benefit wealth."""
    raw = _raw_was()
    donor = clean_was_household_table(raw)

    old_corporate_wealth = (
        raw["totalpenr8_aggr"]
        - raw["dvvaldbt_scaper8_aggr"]
        + raw["DVFESHARESR8_aggr"]
        + raw["DVFShUKVR8_aggr"]
        + raw["DVFCollVR8_aggr"]
        + raw["DVIISAVR8_aggR"]
    )
    assert (
        donor["corporate_wealth"] + donor["private_pension_wealth"]
    ).tolist() == old_corporate_wealth.tolist()
    assert "private_pension_wealth" in UK_WAS_WEALTH_OUTPUT_COLUMNS
    assert (
        UK_WAS_WEALTH_OUTPUT_COLUMNS.index("private_pension_wealth")
        == UK_WAS_WEALTH_OUTPUT_COLUMNS.index("corporate_wealth") + 1
    )


def test_segment_seeds_are_distinct_and_deterministic() -> None:
    """Each chain segment starts its own fit/draw streams (start_chain respawns
    from the model seed on every call), so the roots must differ and derive
    deterministically from the declared stage seed."""
    import microcosm.build.uk_runtime.was_wealth as module

    seeds = module.was_wealth_segment_seeds(0)

    # Golden pin: the production roots for the declared stage seed 0. Moving
    # them is a spec-visible RNG change, never an accident. The first four are
    # the roots the four-segment chain used before microcosm#1063.
    assert seeds == (
        3757552657,
        673228719,
        3241444873,
        3685993406,
        1216546553,
        2078861726,
        2471122328,
        23012616,
    )
    assert len(seeds) == 8
    assert len(set(seeds)) == 8
    assert seeds[:3] == module.was_wealth_segment_seeds(0, segments=3)
    assert seeds == module.was_wealth_segment_seeds(0)
    assert seeds != module.was_wealth_segment_seeds(1)
    assert all(isinstance(seed, int) for seed in seeds)


def _synthetic_recipients(n: int = 80, seed: int = 1063) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    category = np.resize(np.asarray(UK_TENURE_CATEGORIES), n)
    recipient = pd.DataFrame(
        {
            "household_net_income": rng.uniform(5_000, 90_000, n),
            "num_adults": rng.integers(1, 4, n),
            "num_children": rng.integers(0, 3, n),
            "private_pension_income": rng.uniform(0, 20_000, n),
            "employment_income": rng.uniform(0, 60_000, n),
            "self_employment_income": rng.uniform(0, 10_000, n),
            "capital_income": rng.uniform(0, 5_000, n),
            "rental_income": np.where(rng.uniform(size=n) < 0.1, 8_000.0, 0.0),
            "num_bedrooms": rng.integers(1, 6, n),
            "council_tax": rng.uniform(800, 3_000, n),
        }
    )
    for predictor in UK_WAS_TENURE_PREDICTORS:
        recipient[predictor] = category == predictor.removeprefix("tenure_")
    recipient["region"] = np.resize(
        np.asarray(["LONDON", "SCOTLAND", "WALES", "NORTH_EAST", "SOUTH_WEST"]), n
    )
    assert list(recipient.columns) == list(UK_WAS_WEALTH_PREDICTORS)
    recipient[UK_WAS_TENURE_CATEGORY_COLUMN] = category
    return recipient


def _fixture_donor() -> pd.DataFrame:
    from tools.graph_uk_spine_fixture import _was_donor

    return clean_was_household_table(_was_donor())


def test_later_segments_leave_the_earlier_draws_byte_equal() -> None:
    """A later chain segment cannot move an earlier segment's draws.

    Runs the real chain on the hermetic H2 fixture donor, stopping after each
    segment in turn: every segment draws from its own child seed and only
    appends columns.
    """

    donor = _fixture_donor()
    recipient = _synthetic_recipients(48, seed=685)

    full = impute_was_wealth(donor, recipient, seed=0, n_estimators=2)
    assert list(full.draws.columns) == _DRAW_COLUMNS
    for segments in range(1, 8):
        partial = impute_was_wealth(
            donor, recipient, seed=0, n_estimators=2, segments=segments
        )
        pd.testing.assert_frame_equal(
            partial.draws, full.draws.loc[:, list(partial.draws.columns)]
        )
        assert partial.segment_seeds == full.segment_seeds
    # One fit record per drawn target; the derived totals are never fitted.
    derived = {*UK_WAS_DERIVED_TOTALS, "net_financial_wealth"}
    assert len(full.fit_weight_records) == len(_DRAW_COLUMNS) - len(derived)


def test_real_chain_holds_the_tenure_rules_and_the_identities() -> None:
    donor = _fixture_donor()
    recipient = _synthetic_recipients()
    category = recipient[UK_WAS_TENURE_CATEGORY_COLUMN].to_numpy()

    draws = impute_was_wealth(donor, recipient, seed=0, n_estimators=3).draws

    owner = np.isin(category, ["owned_outright", "owned_with_mortgage"])
    mortgaged = category == "owned_with_mortgage"
    # The two structural columns live in their tenure stratum and nowhere else;
    # every fixture donor of the stratum holds a positive value.
    assert (draws.loc[~owner, "main_residence_value"] == 0.0).all()
    assert (draws.loc[owner, "main_residence_value"] > 0.0).all()
    assert (draws.loc[~mortgaged, "main_residence_mortgage"] == 0.0).all()
    assert (draws.loc[mortgaged, "main_residence_mortgage"] > 0.0).all()
    # The other mortgages are carried on every tenure and mortgage_debt is
    # the sum of the two, so a renter's buy-to-let keeps its mortgage.
    assert (draws["other_mortgage"] >= 0.0).all()
    assert (
        draws["mortgage_debt"]
        == draws["main_residence_mortgage"] + draws["other_mortgage"]
    ).all()
    assert wealth_identity_violations(draws) == {
        "property_wealth_violation_rows": 0,
        "corporate_wealth_violation_rows": 0,
        "gross_financial_wealth_violation_rows": 0,
        "mortgage_debt_violation_rows": 0,
        "net_financial_wealth_violation_rows": 0,
    }

    capped, fired = cap_derived_totals_to_donor_range(draws, donor)
    clipped = support_clip_to_donor(capped, donor)
    # The cap keeps every total inside the donor's range, so the support clip
    # moves nothing and the identities survive it.
    assert all(
        column["clipped_low_rows"] == 0 and column["clipped_high_rows"] == 0
        for column in clipped.receipt.evidence()["columns"].values()
    )
    assert set(wealth_identity_violations(clipped.clipped).values()) == {0}
    assert set(fired) == {
        "property_wealth_capped_rows",
        "corporate_wealth_capped_rows",
        "gross_financial_wealth_capped_rows",
        "mortgage_debt_capped_rows",
        "net_financial_wealth_capped_high_rows",
        "net_financial_wealth_capped_low_rows",
    }

    receipt = tenure_coherence_receipt(
        donor=donor,
        recipient_category=recipient[UK_WAS_TENURE_CATEGORY_COLUMN],
        recipient_draws=clipped.clipped,
        recipient_weights=np.ones(len(recipient)),
    )
    assert receipt["main_residence_mortgage_off_mortgaged_tenure_rows"] == 0
    assert receipt["main_residence_value_off_owner_tenure_rows"] == 0
    assert receipt["owner_rows_without_main_residence_value"] == 0
    shares = receipt["positive_share_by_tenure"]["main_residence_value"]
    assert shares["social_rent"] == {"donor": 0.0, "recipient": 0.0}
    assert shares["owned_outright"] == {"donor": 1.0, "recipient": 1.0}
    # The fixture donor carries mortgages on other property off a mortgaged
    # tenure; the chain carries them too, and the receipt says how much on
    # each side.
    assert receipt["donor_mortgage_debt_mass_off_mortgaged_tenure"] > 0.0
    assert receipt["mortgage_debt_rows_off_mortgaged_tenure"] >= 0
    assert receipt["donor_other_property_mortgage_mass_off_mortgaged_tenure"] == (
        pytest.approx(receipt["donor_mortgage_debt_mass_off_mortgaged_tenure"])
    )
    assert receipt["donor_main_residence_mortgage_mass_off_mortgaged_tenure"] == 0.0


def test_total_cap_takes_the_excess_from_the_remainder_first() -> None:
    donor = clean_was_household_table(_raw_was())
    draws = donor.loc[[0], _DRAW_COLUMNS].reset_index(drop=True)
    # Push the property total 500 past the donor's largest (104,100) with
    # donor-plausible components: the remainder (90) goes first, then the
    # buildings, and the main residence is never touched.
    draws.loc[0, "non_residential_property_value"] += 500.0
    draws.loc[0, "property_wealth"] += 500.0

    capped, fired = cap_derived_totals_to_donor_range(draws, donor)

    assert fired["property_wealth_capped_rows"] == 1
    assert capped.loc[0, "property_wealth"] == 104_100.0
    assert capped.loc[0, "other_property_value"] == 0.0
    assert capped.loc[0, "non_residential_property_value"] == 3_090.0
    assert capped.loc[0, "main_residence_value"] == 100_000.0
    assert set(wealth_identity_violations(capped).values()) == {0}
