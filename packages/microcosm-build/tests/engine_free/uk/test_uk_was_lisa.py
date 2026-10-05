"""Engine-free tests for the WAS Lifetime ISA stage (microcosm#1003)."""

from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.build.uk_runtime.was_lisa import (
    BALANCE_COLUMN,
    HOUSEHOLD_BALANCE_COLUMN,
    OWNERSHIP_COLUMN,
    RULE_IMPOSSIBLE_HOLDER,
    UK_WAS_LISA_MASS_CONSERVATION_REASON,
    UK_WAS_LISA_OUTPUT_COLUMNS,
    UK_WAS_LISA_STAGE_NAME,
    UK_WAS_LISA_SUPPORT_CLIP_COLUMNS,
    BalanceModelSpec,
    OwnershipModelSpec,
    UKWASLISAStageTransform,
    WASLISAColumns,
    WASLISAError,
    cap_lifetime_isa_to_financial_wealth,
    clean_was_lisa_donor,
    fit_ownership_model,
    impute_lifetime_isa_balance,
    impute_lifetime_isa_ownership,
    recipient_predictors,
)
from microcosm.build.uk_runtime.was_wealth import UK_WAS_ENGINE_PREDICTOR_ENTITIES
from test_support.paths import paths_for
from tools.graph_uk_spine_fixture import _was_donor, _was_person_donor

_PATHS = paths_for("microcosm-build")

#: The declared operation parameters (the committed manifest carries these).
CLEAN = {
    "household_key": "CASER8",
    "person_columns": {
        "person_number": "PersonR8",
        "is_dependent_child": "IsDepR8",
        "age_band": "DVAge17R8",
        "sex": "SexR8",
        "employment_income": "DVGIEmpR8",
        "holds_lifetime_isa": "fisa_binary3r8_i",
        "holding_imputed": "fisa_binary3r8_iflag",
        "reported_value": "FLISAVR8",
        "value_imputed": "flisavr8_iflag",
        "reported_band": "FLISABR8",
        "band_imputed": "flisabr8_iflag",
        "lifetime_isa_balance": "DVFLISAvR8",
    },
    "household_aggregate": "DVFLISAVR8_aggr",
    "aggregate_tolerance_gbp": 1.0,
    "non_dependent_code": 2,
    "female_code": 2,
    "age_band_edges": list(range(0, 85, 5)),
    "sentinel_codes": [-9, -8, -7, -6],
    "sentinel_recode_columns": ["employment_income"],
    "household_predictors": [
        "household_net_income",
        "num_adults",
        "num_children",
        "gross_financial_wealth",
        "savings",
        "cash_isa",
        "stocks_and_shares_isa",
    ],
    "credibility_rule": {
        "maximum_holder_age_band_lower": 45,
        "balance_ceiling_gbp": 40000,
    },
}
OWNERSHIP = {
    "output": "has_lifetime_isa",
    "seed": 0,
    "salt": "was_lisa:has_lifetime_isa",
    "model": "weighted_ridge_logistic",
    "penalty_c": 1.0,
    "solver": "lbfgs",
    "max_iter": 5000,
    "standardise": "donor_unweighted_mean_sd",
    "age_group_lower_bounds": [0, 25, 35, 45, 55],
    "log1p_predictors": [
        "employment_income",
        "household_net_income",
        "gross_financial_wealth",
        "cash_isa",
        "stocks_and_shares_isa",
        "savings",
    ],
    "indicator_predictors": ["is_private_renter", "is_female"],
    "count_predictors": ["num_children"],
    "population": {"minimum_age": 18},
}
BALANCE = {
    "output": "lifetime_isa_balance",
    "seed": 0,
    "salt": "was_lisa:lifetime_isa_balance",
    "model": "regime_gated_qrf",
    "n_estimators": 4,
    "training_rows": "credible_holders",
    "expected_regime": "positive_only",
    "predictors": [
        "age_band",
        "is_female",
        "employment_income",
        "household_net_income",
        "gross_financial_wealth",
        "savings",
        "cash_isa",
        "stocks_and_shares_isa",
        "is_private_renter",
        "num_children",
    ],
}
CAP = {
    "column": "lifetime_isa_balance",
    "cap_column": "gross_financial_wealth",
    "method": "pro_rata_within_household",
    "ownership_output": "has_lifetime_isa",
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
            "stage": UK_WAS_LISA_STAGE_NAME,
            "survey": "test",
            "source": "test",
            "grain": "person",
            "artifacts": [],
            "operations": [
                {"kind": "clean_was_lisa_donor", **CLEAN},
                {"kind": "impute_lifetime_isa_ownership", **OWNERSHIP},
                {"kind": "impute_lifetime_isa_balance", **BALANCE},
                {"kind": "cap_lifetime_isa_to_financial_wealth", **CAP},
            ],
            "outputs": list(UK_WAS_LISA_OUTPUT_COLUMNS),
            "nonnegative_outputs": [BALANCE_COLUMN, HOUSEHOLD_BALANCE_COLUMN],
        }
    )


def _committed_stage(n_estimators: int = 4) -> SourceStageSpec:
    """The packaged ``was_lisa`` declaration, with a small balance forest."""

    spec = load_country_spec("uk")
    assert spec.sources is not None
    committed = spec.sources.stage_map()[UK_WAS_LISA_STAGE_NAME]
    operations = [
        {
            "kind": operation.kind,
            **dict(operation.parameters),
            **(
                {"n_estimators": n_estimators}
                if operation.kind == "impute_lifetime_isa_balance"
                else {}
            ),
        }
        for operation in committed.operations
    ]
    return SourceStageSpec.from_mapping(
        {
            "stage": committed.stage,
            "survey": committed.survey,
            "source": committed.source,
            "grain": committed.grain,
            "artifacts": [dict(artifact) for artifact in committed.artifacts],
            "operations": operations,
            "outputs": list(committed.outputs),
            "nonnegative_outputs": list(committed.nonnegative_outputs),
        }
    )


def _columns() -> WASLISAColumns:
    return WASLISAColumns.from_parameters(CLEAN)


def _donor():
    return clean_was_lisa_donor(_was_person_donor(), _was_donor(), columns=_columns())


def _recipient_frame(households: int = 64):
    """FRS-like recipients in the synthetic donor's ranges (the fixture's scale)."""

    persons: list[dict[str, object]] = []
    benunits: list[dict[str, object]] = []
    homes: list[dict[str, object]] = []
    person_id = 1
    tenures = [
        "RENT_PRIVATELY",
        "OWNED_WITH_MORTGAGE",
        "OWNED_OUTRIGHT",
        "RENT_FROM_HA",
    ]
    for household in range(1, households + 1):
        adults = 1 + household % 2
        children = household % 3
        benunit = household * 10
        for index in range(adults + children):
            adult = index < adults
            persons.append(
                {
                    "person_id": person_id,
                    "person_benunit_id": benunit,
                    "person_household_id": household,
                    "age": (18 + (household * 7 + index * 11) % 60)
                    if adult
                    else (3 + household % 14),
                    "gender": "FEMALE" if (household + index) % 2 else "MALE",
                    "employment_income": 15_000.0 + 1_000.0 * ((household + index) % 40)
                    if adult
                    else 0.0,
                    "self_employment_income": 0.0,
                    "private_pension_income": 0.0,
                    "capital_income": 0.0,
                    "property_income": 0.0,
                    "is_uc_claimant": adult,
                }
            )
            person_id += 1
        benunits.append(
            {"benunit_id": benunit, "num_adults": adults, "num_children": children}
        )
        wealth = 30.0 + household % 64
        if household == 2:
            wealth = 0.0
        if household == 3:
            wealth = 5.0
        homes.append(
            {
                "household_id": household,
                "household_weight": 1.0 + household % 5,
                "region": "LONDON" if household % 2 else "WALES",
                "num_bedrooms": 1 + household % 4,
                "council_tax": 1_000.0 + 10.0 * household,
                "council_tax_rebate": 0.0,
                "household_net_income": 20_000.0 + 1_000.0 * (household % 60),
                "is_renting": bool(household % 4 == 0),
                "tenure_type": tenures[household % 4],
                "gross_financial_wealth": wealth,
                "savings": 500.0 + 20.0 * (household % 60),
                "cash_isa": 7.0 + household % 60,
                "stocks_and_shares_isa": 5.0 + household % 60,
            }
        )
    return uk_national_frame(
        person=pd.DataFrame(persons),
        benunit=pd.DataFrame(benunits),
        household=pd.DataFrame(homes),
        time_period="2023",
    )


# --------------------------------------------------------------------------
# Donor cleaning
# --------------------------------------------------------------------------


def test_clean_joins_persons_to_households_and_applies_the_credibility_rule() -> None:
    donor = _donor()
    receipt = donor.receipt

    assert receipt["persons_read"] == 190
    assert receipt["donor_adults"] == 127
    assert receipt["dependent_children_excluded"] == 63
    assert receipt["holders_released"] == 15
    # The household-6 holder is recorded in the 60-64 band: no LISA holder can
    # be that old, so the rule recodes them to a non-holder.
    assert receipt["holders_after_rule"] == 14
    assert receipt["credible_holders_in_balance_fit"] == 14
    assert receipt["credibility_rule"]["holders_recoded_for_age"] == 1
    assert receipt["aggregate_check"]["max_abs_difference_gbp"] == 0.0
    assert receipt["sentinel_recodes"] == {"employment_income": 1}
    assert receipt["released_flag_value_disagreements"] == 0
    classes = {k: v["persons"] for k, v in receipt["response_classes"].items()}
    assert classes[RULE_IMPOSSIBLE_HOLDER] == 1
    assert classes["holder_banded_value"] == 1
    assert classes["imputed_holder"] == 1
    assert classes["observed_holder_exact_value"] == 12
    assert classes["imputed_non_holder"] > 0
    assert sum(classes.values()) == receipt["donor_adults"]
    recoded = donor.person.loc[donor.person["household_key"] == 6].iloc[0]
    assert recoded["holds"] == 0
    assert recoded[BALANCE_COLUMN] == 0.0
    assert recoded["lisa_response_class"] == RULE_IMPOSSIBLE_HOLDER
    # A missing answer never becomes a zero: holders keep their released value.
    holders = donor.person["holds"] == 1
    assert (donor.person.loc[holders, BALANCE_COLUMN] > 0).all()
    assert (donor.person.loc[~holders, BALANCE_COLUMN] == 0).all()


def test_clean_keeps_holders_above_the_ceiling_as_owners_outside_the_balance_fit() -> (
    None
):
    person = _was_person_donor()
    household = _was_donor()
    row = person.index[(person["CASER8"] == 1) & (person["PersonR8"] == 1)][0]
    person.loc[row, "DVFLISAvR8"] = 60_000.0
    household.loc[household["CASER8"] == 1, "DVFLISAVR8_aggr"] = 60_000.0

    donor = clean_was_lisa_donor(person, household, columns=_columns())

    above = donor.person.loc[donor.person["household_key"] == 1].iloc[0]
    assert above["holds"] == 1
    assert np.isnan(above[BALANCE_COLUMN])
    assert donor.receipt["credibility_rule"]["holders_above_ceiling"] == 1
    assert donor.receipt["credible_holders_in_balance_fit"] == 13


def test_clean_refuses_a_person_total_that_misses_the_household_aggregate() -> None:
    household = _was_donor()
    household.loc[0, "DVFLISAVR8_aggr"] += 50.0

    with pytest.raises(WASLISAError, match="does not sum to the household"):
        clean_was_lisa_donor(_was_person_donor(), household, columns=_columns())


def test_clean_refuses_persons_without_a_household() -> None:
    person = _was_person_donor()
    person.loc[0, "CASER8"] = 9_999.0

    with pytest.raises(WASLISAError, match="have no household row"):
        clean_was_lisa_donor(person, _was_donor(), columns=_columns())


def test_clean_refuses_a_missing_released_value_instead_of_reading_a_zero() -> None:
    person = _was_person_donor()
    adult = person.index[person["IsDepR8"] == 2][3]
    person.loc[adult, "DVFLISAvR8"] = np.nan

    with pytest.raises(WASLISAError, match="never read as a zero balance"):
        clean_was_lisa_donor(person, _was_donor(), columns=_columns())


def test_clean_refuses_a_repeated_person() -> None:
    person = pd.concat([_was_person_donor(), _was_person_donor().iloc[:1]])

    with pytest.raises(WASLISAError, match=r"repeats a \(case, person\) pair"):
        clean_was_lisa_donor(person, _was_donor(), columns=_columns())


def test_clean_refuses_an_undeclared_column() -> None:
    person = _was_person_donor().drop(columns=["FLISABR8"])

    with pytest.raises(WASLISAError, match="FLISABR8"):
        clean_was_lisa_donor(person, _was_donor(), columns=_columns())


def test_declaration_drift_is_refused() -> None:
    with pytest.raises(WASLISAError, match="household_predictors drifted"):
        WASLISAColumns.from_parameters(
            {**CLEAN, "household_predictors": ["household_net_income"]}
        )
    with pytest.raises(WASLISAError, match="weighted_ridge_logistic"):
        OwnershipModelSpec.from_parameters(
            {**OWNERSHIP, "model": "hist_gradient_boosting"}
        )
    with pytest.raises(WASLISAError, match="credible holders"):
        BalanceModelSpec.from_parameters({**BALANCE, "training_rows": "all_adults"})
    donor = _donor()
    recipient = recipient_predictors(
        _recipient_frame(), _FakeEngine(), age_band_edges=CLEAN["age_band_edges"]
    )
    with pytest.raises(WASLISAError, match="regime"):
        impute_lifetime_isa_balance(
            donor.person,
            recipient,
            holds=np.zeros(len(recipient), dtype=bool),
            spec=BalanceModelSpec.from_parameters(
                {**BALANCE, "expected_regime": "zero_inflated_positive"}
            ),
        )


# --------------------------------------------------------------------------
# Ownership
# --------------------------------------------------------------------------


def test_ownership_logistic_reproduces_the_weighted_donor_share() -> None:
    donor = _donor()
    spec = OwnershipModelSpec.from_parameters(OWNERSHIP)

    _, receipt = fit_ownership_model(donor.person, spec)

    # The unpenalised intercept's score equation: the model's weighted mean
    # probability on the donor is the donor's weighted ownership share.
    assert receipt["donor_weighted_mean_probability"] == pytest.approx(
        receipt["donor_weighted_ownership_share"], rel=1e-3
    )
    assert list(receipt["coefficients"]) == list(spec.feature_names())
    assert receipt["iterations"] < spec.max_iter
    assert receipt["donor_holders"] == 14


def test_ownership_draw_is_keyed_on_person_id_and_skips_minors() -> None:
    donor = _donor()
    spec = OwnershipModelSpec.from_parameters(OWNERSHIP)
    recipient = recipient_predictors(
        _recipient_frame(), _FakeEngine(), age_band_edges=CLEAN["age_band_edges"]
    )
    shuffled = recipient.sample(frac=1.0, random_state=3)

    forward = impute_lifetime_isa_ownership(donor.person, recipient, spec=spec)
    backward = impute_lifetime_isa_ownership(donor.person, shuffled, spec=spec)

    by_id = pd.Series(backward.holds, index=shuffled["person_id"].to_numpy())
    assert np.array_equal(
        forward.holds, by_id.reindex(recipient["person_id"].to_numpy()).to_numpy()
    )
    minors = recipient["age"].to_numpy() < 18
    assert minors.any()
    assert not forward.holds[minors].any()
    assert (forward.probability[minors] == 0.0).all()
    assert forward.holds.any()


# --------------------------------------------------------------------------
# Balance
# --------------------------------------------------------------------------


def test_balance_is_positive_exactly_for_owners_and_inside_credible_support() -> None:
    donor = _donor()
    recipient = recipient_predictors(
        _recipient_frame(), _FakeEngine(), age_band_edges=CLEAN["age_band_edges"]
    )
    holds = impute_lifetime_isa_ownership(
        donor.person, recipient, spec=OwnershipModelSpec.from_parameters(OWNERSHIP)
    ).holds
    spec = BalanceModelSpec.from_parameters(BALANCE)

    drawn = impute_lifetime_isa_balance(donor.person, recipient, holds=holds, spec=spec)

    assert np.array_equal(drawn.balance > 0, holds)
    credible = donor.person.loc[donor.person["holds"] == 1, BALANCE_COLUMN].dropna()
    assert drawn.balance[holds].min() >= credible.min()
    assert drawn.balance[holds].max() <= credible.max()
    assert drawn.receipt["regime"] == "positive_only"
    assert drawn.receipt["training_rows"] == 14

    shuffled = recipient.sample(frac=1.0, random_state=5)
    order = shuffled.index
    again = impute_lifetime_isa_balance(
        donor.person,
        shuffled,
        holds=pd.Series(holds, index=recipient.index).reindex(order).to_numpy(),
        spec=spec,
    )
    by_id = pd.Series(again.balance, index=shuffled["person_id"].to_numpy())
    assert np.array_equal(
        drawn.balance, by_id.reindex(recipient["person_id"].to_numpy()).to_numpy()
    )


# --------------------------------------------------------------------------
# Coherence cap
# --------------------------------------------------------------------------


def test_cap_scales_a_household_over_its_financial_wealth_pro_rata() -> None:
    capped, receipt = cap_lifetime_isa_to_financial_wealth(
        person_household_ids=np.asarray([1, 1, 2, 3, 3]),
        balances=np.asarray([60.0, 60.0, 10.0, 10.0, 0.0]),
        household_ids=np.asarray([1, 2, 3]),
        household_financial_wealth=np.asarray([100.0, 0.0, 1_000.0]),
        household_weights=np.asarray([2.0, 1.0, 1.0]),
    )

    assert capped.tolist() == [50.0, 50.0, 0.0, 10.0, 0.0]
    assert receipt["households_over_financial_wealth"] == 2
    assert receipt["households_over_with_zero_financial_wealth"] == 1
    assert receipt["persons_scaled"] == 2
    assert receipt["owners_cleared"] == 1
    assert receipt["gbp_removed_unweighted"] == pytest.approx(30.0)
    assert receipt["gbp_removed_weighted"] == pytest.approx(2 * 20.0 + 10.0)


# --------------------------------------------------------------------------
# Stage transform
# --------------------------------------------------------------------------


def _transform() -> UKWASLISAStageTransform:
    return UKWASLISAStageTransform(
        stage=_committed_stage(),
        engine=_FakeEngine(),
        donor_household=_was_donor(),
        donor_person=_was_person_donor(),
    )


def test_stage_writes_the_declared_outputs_with_receipts() -> None:
    frame = _recipient_frame()
    transform = _transform()

    result = transform(frame)

    person = result.table("person")
    household = result.table("household")
    assert list(person.columns[-2:]) == [OWNERSHIP_COLUMN, BALANCE_COLUMN]
    assert person[OWNERSHIP_COLUMN].dtype == bool
    assert np.array_equal(
        person[OWNERSHIP_COLUMN].to_numpy(), person[BALANCE_COLUMN].to_numpy() > 0
    )
    assert not person.loc[person["age"] < 18, OWNERSHIP_COLUMN].any()
    totals = person.groupby("person_household_id")[BALANCE_COLUMN].sum()
    np.testing.assert_allclose(
        household[HOUSEHOLD_BALANCE_COLUMN].to_numpy(),
        totals.reindex(household["household_id"]).fillna(0.0).to_numpy(),
    )
    assert (
        household[HOUSEHOLD_BALANCE_COLUMN].to_numpy()
        <= household["gross_financial_wealth"].to_numpy() + 1e-9
    ).all()
    assert person[OWNERSHIP_COLUMN].any()
    receipt = result.mass_log[-1]
    assert receipt.reason == UK_WAS_LISA_MASS_CONSERVATION_REASON
    assert receipt.entity == "household"
    assert receipt.old_total == receipt.new_total > 0
    assert receipt.declared_factor == 1.0
    np.testing.assert_array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )

    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == UK_WAS_LISA_STAGE_NAME
    assert set(evidence) == {
        "stage",
        "support_clip",
        "donor",
        "ownership_model",
        "balance_model",
        "realised",
        "financial_wealth_coherence",
        "population",
    }
    assert evidence["support_clip"]["columns"][BALANCE_COLUMN]["clipped_high_rows"] == 0
    assert (
        evidence["financial_wealth_coherence"]["households_over_financial_wealth"] >= 0
    )
    gate = uk_stage_health_gate(
        evidence=evidence,
        stage=UK_WAS_LISA_STAGE_NAME,
        check="support_clip",
        parameters={
            "columns": [BALANCE_COLUMN],
            "max_clipped_low_rows_by_column": {BALANCE_COLUMN: 0},
            "max_clipped_high_rows_by_column": {BALANCE_COLUMN: 0},
        },
    )
    assert gate.passed, gate.failures
    assert [record.fit_name for record in transform.fit_weight_records] == [
        "uk_was_round_8_lifetime_isa:has_lifetime_isa",
        "uk_was_round_8_lifetime_isa:lifetime_isa_balance",
    ]
    assert {record.weight_kind for record in transform.fit_weight_records} == {
        "explicit"
    }


def test_stage_is_deterministic() -> None:
    first = _transform()(_recipient_frame()).table("person")
    second = _transform()(_recipient_frame()).table("person")

    pd.testing.assert_series_equal(first[BALANCE_COLUMN], second[BALANCE_COLUMN])
    pd.testing.assert_series_equal(first[OWNERSHIP_COLUMN], second[OWNERSHIP_COLUMN])


def test_stage_requires_its_tabs() -> None:
    transform = UKWASLISAStageTransform(stage=_stage(), engine=_FakeEngine())

    with pytest.raises(WASLISAError, match="was_person_tab"):
        transform(_recipient_frame())


def test_committed_declaration_carries_the_tested_parameters() -> None:
    """The packaged manifest declares exactly what these tests exercise."""

    committed = load_country_spec("uk").sources.stage_map()[UK_WAS_LISA_STAGE_NAME]
    by_kind = {
        operation.kind: dict(operation.parameters) for operation in committed.operations
    }

    clean = dict(by_kind["clean_was_lisa_donor"])
    clean.pop("scope")
    clean["credibility_rule"] = {
        key: value for key, value in clean["credibility_rule"].items() if key != "basis"
    }
    assert clean == CLEAN
    assert by_kind["impute_lifetime_isa_ownership"] == OWNERSHIP
    assert by_kind["impute_lifetime_isa_balance"] == {**BALANCE, "n_estimators": 100}
    assert by_kind["cap_lifetime_isa_to_financial_wealth"] == CAP
    assert by_kind["aggregate_person_to_household"] == {
        "method": "sum",
        "aggregates": {HOUSEHOLD_BALANCE_COLUMN: BALANCE_COLUMN},
    }
    receipts = by_kind["record_mass_conservation_receipt"]
    assert receipts["reason"] == UK_WAS_LISA_MASS_CONSERVATION_REASON
    assert committed.outputs == UK_WAS_LISA_OUTPUT_COLUMNS
    roles = {artifact["role"]: artifact for artifact in committed.artifacts}
    assert roles["was_person_tab"]["sha256"] == (
        "1ca6fd37d9c677242112e0d7839df8b406eb1611db8322b20771f054e329d71a"
    )
    was_wealth = load_country_spec("uk").sources.stage_map()["was_wealth"]
    assert roles["was_qrf_donor"] == next(
        artifact
        for artifact in was_wealth.artifacts
        if artifact["role"] == "was_qrf_donor"
    )


def _bounds_tool():
    path = _PATHS.repository / "tools/build_uk_was_lisa_support_bounds.py"
    spec = importlib.util.spec_from_file_location(
        "build_uk_was_lisa_support_bounds", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_support_bounds_round_the_credible_range_outward() -> None:
    payload = _bounds_tool().support_bounds_payload(
        _was_person_donor(),
        _was_donor(),
        columns=_columns(),
        household_tab_sha256="a" * 64,
        person_tab_sha256="b" * 64,
    )

    # The synthetic credible holders hold GBP 5-24 (the household-6 holder is
    # recoded by the age rule), so the one-significant-figure bounds are 0-30.
    assert payload["bounds"] == {BALANCE_COLUMN: [0.0, 30]}
    assert payload["source"]["person_tab_sha256"] == "b" * 64


def test_committed_support_bounds_match_the_stage_pins() -> None:
    resource = (
        _PATHS.package / "src/microcosm/build/uk" / "was_lisa_support_bounds.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    stage = load_country_spec("uk").sources.stage_map()[UK_WAS_LISA_STAGE_NAME]
    pins = {artifact["role"]: artifact["sha256"] for artifact in stage.artifacts}

    assert set(payload["bounds"]) == set(UK_WAS_LISA_SUPPORT_CLIP_COLUMNS)
    assert payload["source"]["household_tab_sha256"] == pins["was_qrf_donor"]
    assert payload["source"]["person_tab_sha256"] == pins["was_person_tab"]
    low, high = payload["bounds"][BALANCE_COLUMN]
    assert low == 0.0
    # Outward-rounded to one significant figure: no unit-record value.
    assert high == float(f"{high:.0e}")
