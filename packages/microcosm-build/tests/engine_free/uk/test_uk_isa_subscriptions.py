"""The ISA subscriptions stage's pieces on synthetic data (microcosm#1415)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime import isa_subscriptions as stage
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
)
from microcosm.build.uk_runtime.hmrc_isa_statistics import (
    HMRC_ISA_INCOME_BAND_IDS,
    HMRCISAStatisticsError,
    isa_band_bounds,
    lifetime_isa_paying_ages,
    load_hmrc_isa_statistics,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_isa_statistics import (
    isa_stage_spec,
    synthetic_isa_statistics_payload,
    synthetic_was_isa_tabs,
)


@pytest.fixture(scope="module")
def statistics():
    return load_hmrc_isa_statistics(
        payload=synthetic_isa_statistics_payload(), verify_feed_identity=False
    )


@pytest.fixture(scope="module")
def spec():
    return stage.ISASubscriptionsSpec.from_stage(isa_stage_spec())


@pytest.fixture(scope="module")
def donor(spec):
    household, person = synthetic_was_isa_tabs()
    return stage.clean_was_isa_person_donor(
        person,
        household,
        columns=spec.columns,
        maximum_adults=spec.maximum_donor_adults,
    )


def _population(seed: int = 1, households: int = 800):
    rng = np.random.default_rng(seed)
    household = pd.DataFrame(
        {
            "household_id": np.arange(households) + 1000,
            "cash_isa": np.where(
                rng.random(households) < 0.75, rng.integers(100, 80_000, households), 0
            ).astype(float),
            "stocks_and_shares_isa": np.where(
                rng.random(households) < 0.4, rng.integers(100, 150_000, households), 0
            ).astype(float),
            "region": rng.choice(sorted(stage._REGION_ROW), households),
        }
    )
    rows = []
    for household_id in household["household_id"]:
        for _ in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "person_id": 10_000 + len(rows),
                    "person_household_id": household_id,
                    "age": float(rng.integers(16, 90)),
                    "gender": rng.choice(["MALE", "FEMALE"]),
                }
            )
    person = pd.DataFrame(rows)
    weights = pd.Series(rng.uniform(5, 30, households), index=household["household_id"])
    income = rng.lognormal(10.3, 0.9, len(person))
    ages = person["age"].to_numpy(float)
    recipients = pd.DataFrame(
        {
            "person_id": person["person_id"].to_numpy(),
            "household_id": person["person_household_id"].to_numpy(),
            "weight": weights.reindex(person["person_household_id"]).to_numpy(),
            "age": ages,
            "sex": person["gender"].str.lower().to_numpy(),
            "total_income": income,
            "income_band": stage._income_band_ids(income),
            "age_band": stage._age_band_ids(ages),
            "region": household.set_index("household_id")["region"]
            .reindex(person["person_household_id"])
            .map(stage._REGION_ROW)
            .to_numpy(),
            "holds_lifetime_isa": (rng.random(len(person)) < 0.06)
            & (ages < 45)
            & (ages >= 18),
            "lifetime_isa_balance": rng.uniform(100, 20_000, len(person)),
        }
    )
    return household, person, recipients


def _run(spec, donor, statistics, seed: int = 1):
    household, person, recipients = _population(seed)
    cash_balance, stocks_balance, receipt = stage.split_household_isa_balances(
        person=person,
        household=household,
        donor=donor,
        seed=spec.split_seed,
        salt=spec.split_salt,
        minimum_age=spec.minimum_age,
        maximum_adults=spec.maximum_donor_adults,
    )
    cash, stocks, lifetime, result = stage.impute_isa_subscriptions(
        recipients,
        cash_balance=cash_balance,
        stocks_balance=stocks_balance,
        statistics=statistics,
        spec=spec,
    )
    return (
        household,
        person,
        recipients,
        cash_balance,
        stocks_balance,
        cash,
        stocks,
        lifetime,
        result,
    )


def test_the_declaration_is_read_and_refuses_drift():
    spec = stage.ISASubscriptionsSpec.from_stage(isa_stage_spec())
    assert spec.minimum_age == 18 and spec.overall_limit_gbp == 20_000
    assert (spec.base_tax_year, spec.build_tax_year) == (2023, 2024)
    for kind, key, value in (
        ("allocate_isa_subscribers", "selection_order", ["cash_only"]),
        ("draw_isa_subscription_amounts", "within_band", "tilted"),
        ("split_both_type_subscriptions", "anchor", "table_9_4_cash_residual"),
        ("grow_isa_subscriptions", "people", "table_9_4_amount_growth"),
        ("grow_isa_subscriptions", "amounts", "table_9_4_amount_growth_capped"),
        # Nobody aged 48 or 49 can hold a Lifetime ISA in 2024-25.
        ("draw_lifetime_isa_payments", "maximum_age", 49),
        ("draw_lifetime_isa_payments", "outputs", ["lifetime_isa_bonus"]),
        ("grow_isa_subscriptions", "to_tax_year", 2025),
        ("classify_isa_facts_with_reviewed_fence", "calibration_permitted", True),
        ("classify_isa_facts_with_reviewed_fence", "fenced_fact_count", 1000),
    ):
        declared = isa_stage_spec()
        operations = tuple(
            SourceOperationSpec(op.kind, {**op.parameters, key: value})
            if op.kind == kind
            else op
            for op in declared.operations
        )
        with pytest.raises(stage.ISASubscriptionsError):
            stage.ISASubscriptionsSpec.from_stage(
                dataclasses.replace(declared, operations=operations)
            )


def test_the_donor_pins_person_values_to_the_household_aggregates(spec):
    household, person = synthetic_was_isa_tabs()
    donor = stage.clean_was_isa_person_donor(
        person, household, columns=spec.columns, maximum_adults=4
    )
    for shares, holds in (
        (donor.cash_shares, donor.holds_cash),
        (donor.stocks_and_shares_shares, donor.holds_stocks_and_shares),
    ):
        totals = np.asarray([share.sum() for share in shares])
        assert np.allclose(totals[holds], 1.0) and np.allclose(totals[~holds], 0.0)
    broken = person.copy()
    broken.loc[broken.index[0], "DVCISAvR8"] += 50.0
    with pytest.raises(stage.ISASubscriptionsError, match="do not sum"):
        stage.clean_was_isa_person_donor(
            broken, household, columns=spec.columns, maximum_adults=4
        )
    negative = person.copy()
    negative.loc[negative.index[0], "DVIISAvR8"] = -1.0
    with pytest.raises(stage.ISASubscriptionsError, match="non-negative"):
        stage.clean_was_isa_person_donor(
            negative, household, columns=spec.columns, maximum_adults=4
        )


def test_person_balances_sum_exactly_to_the_household_draws(spec, donor):
    household, person, *_ = _population()
    cash, stocks, receipt = stage.split_household_isa_balances(
        person=person,
        household=household,
        donor=donor,
        seed=0,
        salt="split",
        minimum_age=18,
        maximum_adults=4,
    )
    sums = (
        pd.DataFrame(
            {"cash": cash, "stocks": stocks, "household": person["person_household_id"]}
        )
        .groupby("household")[["cash", "stocks"]]
        .sum()
        .reindex(household["household_id"])
        .fillna(0.0)
    )
    assert np.allclose(sums["cash"], household["cash_isa"], rtol=0, atol=1e-6)
    assert np.allclose(
        sums["stocks"], household["stocks_and_shares_isa"], rtol=0, atol=1e-6
    )
    assert (cash >= 0).all() and (stocks >= 0).all()
    minors = person["age"].to_numpy(float) < 18
    has_adult = person.groupby("person_household_id")["age"].transform("max") >= 18
    assert not ((cash > 0) & minors & has_adult.to_numpy()).any()


def test_the_imputation_keeps_every_invariant(spec, donor, statistics):
    *_, recipients, cash_balance, stocks_balance, cash, stocks, lifetime, result = _run(
        spec, donor, statistics
    )
    assert all(count == 0 for count in result["checks"]["violations"].values()), result[
        "checks"
    ]
    combined = cash + stocks
    assert ((combined + lifetime) <= 20_000 + 1e-6).all()
    assert (lifetime <= 4_000 + 1e-6).all()
    assert ((cash > 0) <= (cash_balance > 0)).all() and (
        (stocks > 0) <= (stocks_balance > 0)
    ).all()
    ages = recipients["age"].to_numpy(float)
    assert not ((combined > 0) & (ages < 18)).any()
    assert not (
        (lifetime > 0) & ((ages > 47) | ~recipients["holds_lifetime_isa"].to_numpy())
    ).any()
    split = result["both_type_split"]
    assert split["saturated"] is None
    assert split["stocks_and_shares_achieved_gbp"] == pytest.approx(
        split["stocks_and_shares_target_gbp"]
    )
    growth = result["growth"]
    # 2024-25 meets Table 9.4's own amounts (María's ruling of 29 October 2026).
    assert growth["cash_target_gbp"] == statistics.amount_subscribed[("cash", 2024)]
    assert (
        growth["stocks_and_shares_target_gbp"]
        == (statistics.amount_subscribed[("stocks_and_shares", 2024)])
    )
    assert growth["cash_achieved_gbp"] == pytest.approx(
        growth["cash_target_gbp"], rel=1e-9
    )
    assert growth["stocks_and_shares_achieved_gbp"] == pytest.approx(
        growth["stocks_and_shares_target_gbp"], rel=1e-9
    )


def test_band_amounts_sit_inside_their_bands_with_an_exact_top(statistics):
    rng = np.random.default_rng(3)
    rows = np.arange(400)
    weights = rng.uniform(1, 5, 400)
    shares = np.asarray(
        [
            statistics.subscribers("cash_only", HMRC_ISA_INCOME_BAND_IDS[3], band)
            for band in stage.HMRC_ISA_SUBSCRIPTION_BAND_IDS
        ]
    )
    amounts = stage.draw_band_amounts(
        rows,
        weights=weights,
        balance=rng.uniform(0, 50_000, 400),
        person_ids=rows + 1,
        band_shares=shares,
        balance_rank_weight=0.5,
        seed=0,
        salt="t",
        key="k",
    )
    assert (amounts > 0).all() and (amounts <= 20_000).all()
    top = amounts == 20_000
    assert weights[top].sum() / weights.sum() == pytest.approx(
        shares[-1] / shares.sum(), abs=0.02
    )
    assert not ((amounts > 18_000) & (amounts < 20_000) & (amounts >= 19_999.999)).any()


class _UKEngine:
    country = "uk"


def _stage_frame(seed: int = 1, order: np.ndarray | None = None):
    """The synthetic population as the 2024 frame the whole transform reads."""

    household, person, recipients = _population(seed)
    holds = recipients["holds_lifetime_isa"].to_numpy()
    person = person.assign(
        person_benunit_id=person["person_household_id"].to_numpy() * 10,
        **dict.fromkeys(UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS, 0.0),
        has_lifetime_isa=holds,
        lifetime_isa_balance=np.where(
            holds, recipients["lifetime_isa_balance"].to_numpy(), 0.0
        ),
    )
    person["employment_income"] = recipients["total_income"].to_numpy()
    if order is not None:
        person = person.iloc[order].reset_index(drop=True)
    weights = (
        recipients.groupby("household_id")["weight"]
        .first()
        .reindex(household["household_id"])
        .to_numpy()
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": household["household_id"] * 10}),
        household=household,
        household_weights=weights,
        weight_kind=WeightKind.DESIGN,
        time_period="2024",
    )


def _transform(statistics):
    household, person = synthetic_was_isa_tabs()
    return stage.UKISASubscriptionsStageTransform(
        stage=isa_stage_spec(),
        engine=_UKEngine(),
        donor_household=household,
        donor_person=person,
        statistics=statistics,
    )


def test_the_whole_stage_writes_its_columns_within_every_invariant(statistics):
    from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

    frame = _stage_frame()
    transform = _transform(statistics)
    result = transform(frame)
    person = result.table("person")
    for column in stage.UK_ISA_SUBSCRIPTIONS_OUTPUT_COLUMNS:
        values = person[column].to_numpy(float)
        assert np.isfinite(values).all() and (values >= 0).all(), column
    assert (person[stage.CASH_SUBSCRIPTIONS] > 0).any()
    assert (person[stage.LIFETIME_ISA_PAYMENTS] > 0).any()
    receipt = transform.last_receipt
    assert all(count == 0 for count in receipt["checks"]["violations"].values())
    assert receipt["balance_split"]["max_abs_household_sum_error_gbp"] <= 0.01
    # Households, weights and every earlier person column pass through.
    pd.testing.assert_frame_equal(result.table("household"), frame.table("household"))
    assert np.array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )
    pd.testing.assert_frame_equal(
        person[list(frame.table("person").columns)], frame.table("person")
    )
    gate = uk_stage_health_gate(
        evidence=receipt,
        stage="isa_subscriptions",
        check="isa_subscriptions",
        parameters={
            "maximum_relative_total_deviation": 1e-6,
            "maximum_household_sum_error_gbp": 0.01,
        },
    )
    assert gate.passed, gate.failures


def test_reordering_the_rows_moves_no_draw(statistics):
    """Every draw is keyed on identity, so row order moves none.

    Adding or removing rows can move others' amounts (the rakes, band walks and
    growth factors read the whole population), so only order is invariant.
    """

    columns = list(stage.UK_ISA_SUBSCRIPTIONS_OUTPUT_COLUMNS)
    transform = _transform(statistics)
    first = transform(_stage_frame()).table("person")
    order = np.random.default_rng(7).permutation(len(first))
    second = transform(_stage_frame(order=order)).table("person")
    pd.testing.assert_frame_equal(
        first.set_index("person_id")[columns].sort_index(),
        second.set_index("person_id")[columns].sort_index(),
        check_exact=False,
        rtol=0,
        atol=1e-6,
    )


def test_the_both_split_reports_saturation():
    combined = np.full(10, 10_000.0)
    weights = np.ones(10)
    share, receipt = stage.split_both_type_amounts(
        combined,
        np.full(10, 0.5),
        weights,
        stocks_and_shares_target=-1.0,
        share_floor=0.02,
    )
    assert receipt["saturated"] == "stocks_and_shares_residual_below_reach"
    assert (share > 0).all() and (share < 1).all()
    share, receipt = stage.split_both_type_amounts(
        combined,
        np.full(10, 0.5),
        weights,
        stocks_and_shares_target=30_000.0,
        share_floor=0.02,
    )
    assert receipt["saturated"] is None
    assert float((combined * (1 - share)).sum()) == pytest.approx(30_000.0)


@pytest.mark.parametrize(
    ("cash_target", "stocks_target"),
    [
        # Below the starting totals: the factor falls under one.
        (30_000.0, 12_000.0),
        # Above them: the rest rise, capped at the limit.
        (52_000.0, 18_000.0),
    ],
)
def test_amounts_scale_to_the_build_year_holding_the_limit(cash_target, stocks_target):
    cash = np.asarray([19_000.0, 5_000.0, 0.0, 10_000.0, 8_000.0])
    stocks = np.asarray([1_000.0, 0.0, 8_000.0, 5_000.0, 2_000.0])
    weights = np.ones(5)
    scaled_cash, scaled_stocks, receipt = stage.scale_to_build_year_amounts(
        cash,
        stocks,
        weights,
        cash_target=cash_target,
        stocks_target=stocks_target,
        limit=20_000.0,
    )
    # The subscriber at the limit stays exactly where they were.
    assert (scaled_cash[0], scaled_stocks[0]) == (19_000.0, 1_000.0)
    assert ((scaled_cash + scaled_stocks) <= 20_000 + 1e-6).all()
    assert scaled_cash.sum() == pytest.approx(cash_target)
    assert scaled_stocks.sum() == pytest.approx(stocks_target)
    rising = bool(cash_target > cash.sum())
    assert (receipt["cash_factor"] > 1) is rising


def test_a_non_finite_amount_is_a_violation(spec):
    recipients = pd.DataFrame({"weight": np.ones(3), "age": [30.0, 40.0, 50.0]})
    cash = np.asarray([1_000.0, np.nan, 0.0])
    zeros = np.zeros(3)
    checks = stage._checks(
        recipients,
        cash,
        zeros,
        zeros,
        np.asarray([stage.CASH_ONLY, stage.CASH_ONLY, ""], dtype=object),
        np.full(3, 5_000.0),
        zeros,
        spec,
    )
    # NaN compares false, so only the finiteness count catches it.
    assert checks["violations"]["non_finite_values"] == 1
    assert checks["violations"]["over_overall_limit"] == 0


def test_lifetime_isa_paying_ages_follow_the_2017_opening():
    # Opened at 18 to 39 from 6 April 2017; payments stop at 50.
    assert lifetime_isa_paying_ages(2017) == (18, 40)
    assert lifetime_isa_paying_ages(2024) == (18, 47)
    assert lifetime_isa_paying_ages(2026) == (18, 49)
    assert lifetime_isa_paying_ages(2030) == (18, 49)
    with pytest.raises(ValueError):
        lifetime_isa_paying_ages(2016)


def test_band_ids_parse_and_the_loader_refuses_gaps():
    assert isa_band_bounds("subscription_20000_to_20000") == (20000, 20000)
    assert isa_band_bounds("income_150001_plus") == (150001, None)
    with pytest.raises(HMRCISAStatisticsError):
        isa_band_bounds("income_banded")
    payload = synthetic_isa_statistics_payload()
    payload["rows"] = [
        row for row in payload["rows"] if ".table9_9." not in row["source_record_id"]
    ]
    with pytest.raises(HMRCISAStatisticsError, match="lack"):
        load_hmrc_isa_statistics(payload=payload, verify_feed_identity=False)
    payload = synthetic_isa_statistics_payload()
    for row in payload["rows"]:
        if row["source_record_id"].endswith(
            "table9_7.ty2023.all.subscribers.all.all_subscriptions"
        ):
            row["value"] += 1_000.0
    with pytest.raises(HMRCISAStatisticsError):
        load_hmrc_isa_statistics(payload=payload, verify_feed_identity=False)


def test_the_committed_declaration_carries_the_tested_operations():
    """The packaged manifest declares exactly what these tests exercise."""

    from microcosm.build.country_spec import load_country_spec
    from test_support.microcosm_build.uk_isa_statistics import (
        ISA_FACTS_ARTIFACT,
        ISA_STAGE_OPERATIONS,
    )

    stages = load_country_spec("uk").sources.stage_map()
    committed = stages["isa_subscriptions"]
    assert [
        {"kind": operation.kind, **dict(operation.parameters)}
        for operation in committed.operations
    ] == list(ISA_STAGE_OPERATIONS)
    assert committed.outputs == stage.UK_ISA_SUBSCRIPTIONS_OUTPUT_COLUMNS
    assert (
        committed.nonnegative_outputs
        == stage.UK_ISA_SUBSCRIPTIONS_NONNEGATIVE_OUTPUT_COLUMNS
    )
    roles = {artifact["role"]: artifact for artifact in committed.artifacts}
    assert roles["isa_statistics_facts"] == ISA_FACTS_ARTIFACT
    lisa = {artifact["role"]: artifact for artifact in stages["was_lisa"].artifacts}
    for role in ("was_qrf_donor", "was_person_tab"):
        assert roles[role] == lisa[role]
    assert list(stages)[-1] == "isa_subscriptions"


def test_no_calibration_target_reads_an_isa_fact():
    """The reviewed fence keeps every vendored ISA fact measured only.

    The stage draws subscribers and amounts to these cells, so a target compiled
    from them would be circular (María's ruling of 28 October 2026). The fence is
    declared on the stage; this holds the contract and the compiled references to
    it, so a target that read one would fail here, not only in review.
    """

    from importlib.resources import files

    prefixes = (
        "hmrc.isa_statistics_2026",
        "hmrc.lisa_statistics_2026",
        "hmrc.cash_isa_limit_reduction_2026",
    )
    for resource in (
        "uk_population_targets.json",
        "target_references.json",
        "local_target_references.json",
    ):
        text = files("microcosm.build.uk").joinpath(resource).read_text("utf-8")
        for prefix in prefixes:
            assert prefix not in text, (resource, prefix)
