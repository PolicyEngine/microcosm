"""The ISA subscriptions stage's pieces on synthetic data (microcosm#1415)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.stage_evidence import snapshot_stage_evidence
from microcosm.build.stochastic_assignment import binomial_share_tolerance
from microcosm.build.uk_runtime import copy_id_table
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


#: Five balance atoms, as repeated QRF draws give them: holders tie on them.
BALANCE_ATOMS = (1_000.0, 5_000.0, 12_000.0, 20_000.0, 45_000.0)


def _population(
    seed: int = 1, households: int = 800, atoms: tuple[float, ...] | None = None
):
    """The synthetic population; with ``atoms``, every positive household ISA
    total and every Lifetime ISA balance is one of them, drawn from a second
    generator so the other draws stay as they are."""

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
            # Enough holders that the payments reach the synthetic Table 9.4
            # amount without saturating.
            "holds_lifetime_isa": (rng.random(len(person)) < 0.10)
            & (ages < 45)
            & (ages >= 18),
            "lifetime_isa_balance": rng.uniform(100, 20_000, len(person)),
        }
    )
    if atoms is not None:
        pick = np.random.default_rng(seed + 1_000)
        for column in ("cash_isa", "stocks_and_shares_isa"):
            held = household[column].to_numpy() > 0
            household[column] = np.where(held, pick.choice(atoms, households), 0.0)
        recipients["lifetime_isa_balance"] = pick.choice(atoms, len(person))
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


def _row_position_rank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The balance rank before microcosm#931: tied values ranked in row order."""

    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    ranks = np.empty(len(values), dtype=float)
    if total <= 0:
        ranks[:] = 0.5
        return ranks
    ranks[order] = (cumulative - weights[order] / 2.0) / total
    return ranks


def _tie_heavy(rng: np.random.Generator, n: int):
    """Values half on the five atoms and half distinct, weights, unique ids."""

    values = np.where(
        rng.random(n) < 0.5,
        rng.choice(BALANCE_ATOMS, n),
        rng.uniform(100.0, 60_000.0, n),
    )
    weights = rng.uniform(1.0, 50.0, n)
    ids = rng.choice(10**9, n, replace=False).astype(np.int64)
    return values, weights, ids


def test_tied_balances_share_one_weighted_mid_rank():
    # The three rows tied on 5 take the group's mid-rank: the weight below it
    # (6) plus half the group's (6), over the total (12). The two singletons
    # keep their own. By row position the tied rows ranked 6.5, 8.5 and 11
    # twelfths (microcosm#931).
    values = np.asarray([5.0, 1.0, 5.0, 3.0, 5.0])
    weights = np.asarray([1.0, 2.0, 3.0, 4.0, 2.0])
    ids = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)
    np.testing.assert_array_equal(
        stage._weighted_rank(values, weights, ids),
        [9 / 12, 1 / 12, 9 / 12, 4 / 12, 9 / 12],
    )
    np.testing.assert_array_equal(
        _row_position_rank(values, weights)[[0, 2, 4]], [6.5 / 12, 8.5 / 12, 11 / 12]
    )
    # On tie-heavy input every member of a group shares one rank,
    # (F(x-) + F(x)) / 2 over the total, which for a singleton is its own.
    values, weights, ids = _tie_heavy(np.random.default_rng(20261110), 3_000)
    ranks = stage._weighted_rank(values, weights, ids)
    total = weights.sum()
    sizes = []
    for value in np.unique(values):
        group = values == value
        sizes.append(int(group.sum()))
        expected = (weights[values < value].sum() + weights[group].sum() / 2) / total
        assert len(set(ranks[group].tolist())) == 1
        assert ranks[group][0] == pytest.approx(expected, rel=0, abs=1e-12)
    tie_groups = [size for size in sizes if size > 1]
    assert len(tie_groups) == len(BALANCE_ATOMS) and min(tie_groups) > 200
    # No weight: every rank is the middle, as before.
    assert (stage._weighted_rank(values, np.zeros_like(weights), ids) == 0.5).all()


def test_singleton_ranks_are_the_row_position_ranks_on_id_sorted_rows():
    # The spine's tables arrive in id order, so a balance nobody else holds
    # keeps its rank bit for bit and only tied rows move, to their group's
    # mid-rank (microcosm#931).
    values, weights, ids = _tie_heavy(np.random.default_rng(1110), 3_000)
    in_id_order = np.argsort(ids)
    values, weights, ids = values[in_id_order], weights[in_id_order], ids[in_id_order]
    tied = pd.Series(values).duplicated(keep=False).to_numpy()
    assert 0.4 < tied.mean() < 0.6

    ranks = stage._weighted_rank(values, weights, ids)
    before = _row_position_rank(values, weights)

    np.testing.assert_array_equal(ranks[~tied], before[~tied])
    assert (ranks[tied] != before[tied]).mean() > 0.99
    # Without ties every rank is the row-position rank, bit for bit.
    np.testing.assert_array_equal(
        stage._weighted_rank(values[~tied], weights[~tied], ids[~tied]),
        _row_position_rank(values[~tied], weights[~tied]),
    )


def test_the_ranks_do_not_depend_on_row_order():
    # The weights accumulate in (value, id) order, so each row's rank is the
    # same bit for bit however the rows arrive; by row position, tied rows'
    # ranks followed their rows (microcosm#931).
    values, weights, ids = _tie_heavy(np.random.default_rng(11), 2_000)
    tied = pd.Series(values).duplicated(keep=False).to_numpy()
    ranks = pd.Series(stage._weighted_rank(values, weights, ids), index=ids)
    before = pd.Series(_row_position_rank(values, weights), index=ids)
    for seed in range(5):
        rows = np.random.default_rng(seed).permutation(len(values))
        permuted = stage._weighted_rank(values[rows], weights[rows], ids[rows])
        np.testing.assert_array_equal(permuted, ranks.loc[ids[rows]].to_numpy())
        old = _row_position_rank(values[rows], weights[rows])
        followed = old != before.loc[ids[rows]].to_numpy()
        assert followed[tied[rows]].mean() > 0.99


def _clone_pool(*, sources: int, clone_count: int, seed: int = 931):
    """``sources`` holders and their geographic clones as one walk's payers:
    clone-major, as their ids sort, each clone carrying its source's balance
    (on the five atoms; both draws rank it) and an equal share of its weight."""

    rng = np.random.default_rng(seed)
    roots = 10_000 + np.arange(sources, dtype=np.int64)
    ids = np.concatenate(
        [
            roots,
            *(
                copy_id_table.mint(roots, "geographic_support", k)
                for k in range(1, clone_count)
            ),
        ]
    )
    balance = np.tile(rng.choice(BALANCE_ATOMS, sources), clone_count)
    weights = np.tile(rng.uniform(5.0, 30.0, sources) / clone_count, clone_count)
    return pd.DataFrame(
        {
            "person_id": ids,
            "weight": weights,
            "age": np.full(len(ids), 30.0),
            "holds_lifetime_isa": np.ones(len(ids), dtype=bool),
            "lifetime_isa_balance": balance,
        }
    )


def _clone_zero_least(amounts: np.ndarray, clone_count: int) -> tuple[float, int]:
    """The share of sources whose clone 0 gets strictly the least of its
    clones, over the sources whose clones' amounts all differ, and their count."""

    by_clone = amounts.reshape(clone_count, -1)
    distinct = (np.diff(np.sort(by_clone, axis=0), axis=0) > 0).all(axis=0)
    least = (by_clone[0] < by_clone[1:]).all(axis=0)
    return float(least[distinct].mean()), int(distinct.sum())


def test_clone_zero_is_not_ranked_below_its_clones(spec, monkeypatch):
    # Three geographic clones of 2,000 holders, clone-major as their ids sort,
    # each carrying its source's balance on five atoms. Clone 0 gets the least
    # ISA amount, and the least Lifetime ISA payment, of its three in about a
    # third of the sources, inside the four-sigma binomial band. Ranked by row
    # position, each clone 0 sat in the bottom third of its tie group and got
    # the least in about 43 and 40 per cent of the sources, outside the band
    # (microcosm#931).
    clone_count = 3
    pool = _clone_pool(sources=2_000, clone_count=clone_count)
    shares = np.full(len(stage.HMRC_ISA_SUBSCRIPTION_BAND_IDS), 1.0)

    def isa_amounts() -> np.ndarray:
        return stage.draw_band_amounts(
            np.arange(len(pool)),
            weights=pool["weight"].to_numpy(),
            balance=pool["lifetime_isa_balance"].to_numpy(),
            person_ids=pool["person_id"].to_numpy(),
            band_shares=shares,
            balance_rank_weight=spec.balance_rank_weight,
            seed=spec.amount_seed,
            salt=spec.amount_salt,
            key="clones",
        )

    def lifetime_isa_payments() -> np.ndarray:
        payments, receipt = stage.draw_lifetime_isa_payments(
            pool,
            headroom=np.full(len(pool), 20_000.0),
            target_gbp=float(pool["weight"].sum()) * 1_000.0,
            spec=spec,
        )
        assert receipt["saturated"] is None
        return payments

    def clone_zero_least() -> list[tuple[float, float]]:
        found = []
        for draw in (isa_amounts, lifetime_isa_payments):
            share, count = _clone_zero_least(draw(), clone_count)
            band = binomial_share_tolerance(
                target=1 / clone_count, rows=count, sigma=4.0
            )
            found.append((share, band))
        return found

    for share, band in clone_zero_least():
        assert abs(share - 1 / clone_count) <= band
    monkeypatch.setattr(
        stage,
        "_weighted_rank",
        lambda values, weights, ids: _row_position_rank(values, weights),
    )
    for share, band in clone_zero_least():
        assert share > 1 / clone_count + band


class _UKEngine:
    country = "uk"


def _stage_frame(
    seed: int = 1,
    order: np.ndarray | None = None,
    atoms: tuple[float, ...] | None = None,
):
    """The synthetic population as the 2024 frame the whole transform reads."""

    household, person, recipients = _population(seed, atoms=atoms)
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


def _declared_gate_parameters(gate_id: str) -> dict:
    from microcosm.build.country_spec import load_country_spec

    entry = next(
        gate for gate in load_country_spec("uk").gates.gates if gate.id == gate_id
    )
    return dict(entry.parameters)


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
    # Every class is selected, and the Lifetime ISA total is reached, not named.
    for isa_type in stage.CLASS_ORDER:
        assert receipt["selection"][isa_type]["base"]["selected_weight"] > 0, isa_type
    assert receipt["lifetime_isa"]["saturated"] is None
    assert (
        0
        < receipt["lifetime_isa"]["paying_weight"]
        <= (receipt["lifetime_isa"]["eligible_holder_weight"])
    )
    # Households, weights and every earlier person column pass through.
    pd.testing.assert_frame_equal(result.table("household"), frame.table("household"))
    assert np.array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )
    pd.testing.assert_frame_equal(
        person[list(frame.table("person").columns)], frame.table("person")
    )
    # Both declared entries pass at their declared tolerances.
    for gate_id in (
        "uk_stage_isa_subscriptions_invariants",
        "uk_stage_isa_subscriptions",
    ):
        parameters = _declared_gate_parameters(gate_id)
        gate = uk_stage_health_gate(
            evidence=receipt,
            stage=parameters["stage"],
            check=parameters["check"],
            parameters=parameters,
        )
        assert gate.passed, (gate_id, gate.failures)


def test_the_stage_declares_no_fit_weight_records(statistics):
    """The stage fits no model. A declared empty list read as a fitting stage
    that recorded nothing, which failed the release-cut weights audit
    (microcosm#1428)."""

    transform = _transform(statistics)
    transform(_stage_frame())
    evidence = snapshot_stage_evidence("isa_subscriptions", transform)
    assert "fit_weight_records" not in evidence


@pytest.mark.parametrize(
    "atoms", [None, BALANCE_ATOMS], ids=["distinct_balances", "five_balance_atoms"]
)
def test_reordering_the_rows_moves_no_draw(statistics, atoms):
    """Every draw is keyed on identity, so row order moves none.

    On five atoms most holders tie on their balance, as copies, clone halves
    and repeated QRF draws do. Tied balances share one rank, so the keyed
    uniform orders them, not their rows (microcosm#931); ranked by row
    position, most of their amounts moved under this permutation. Adding or
    removing rows can move others' amounts (the rakes, band walks and growth
    factors read the whole population), so only order is invariant.
    """

    columns = list(stage.UK_ISA_SUBSCRIPTIONS_OUTPUT_COLUMNS)
    transform = _transform(statistics)
    first = transform(_stage_frame(atoms=atoms)).table("person")
    if atoms is not None:
        # The ties are there: most subscribers and Lifetime ISA payers share
        # their balance with another.
        paying = (
            first[stage.CASH_SUBSCRIPTIONS]
            + first[stage.STOCKS_AND_SHARES_SUBSCRIPTIONS]
            > 0
        )
        held = first[stage.CASH_BALANCE] + first[stage.STOCKS_AND_SHARES_BALANCE]
        assert held[paying].duplicated(keep=False).mean() > 0.6
        lifetime = first.loc[
            first[stage.LIFETIME_ISA_PAYMENTS] > 0, stage.LIFETIME_ISA_BALANCE
        ]
        assert lifetime.duplicated(keep=False).mean() > 0.9
    order = np.random.default_rng(7).permutation(len(first))
    second = transform(_stage_frame(order=order, atoms=atoms)).table("person")
    pd.testing.assert_frame_equal(
        first.set_index("person_id")[columns].sort_index(),
        second.set_index("person_id")[columns].sort_index(),
        check_exact=False,
        rtol=0,
        atol=1e-6,
    )


def test_the_walk_offsets_and_jitters_are_the_uniforms_they_were(spec):
    """The keyed offset moved to the shared scalar and the redraw's adapter
    (microcosm#931); every walk offset and band jitter is the uniform the
    stage keyed before the move, bit for bit."""

    from microcosm.build import stochastic_assignment
    from microcosm.build.uk_runtime import cgt_imputation

    assert stage._KeyedOffset is cgt_imputation._KeyedOffset
    assert (
        stage.stable_identity_uniform is stochastic_assignment.stable_identity_uniform
    )
    before = stochastic_assignment.stable_identity_uniforms
    for isa_type in stage.CLASS_ORDER:
        # The selection's cell walks, then the amounts' walk and band jitter,
        # salted as the stage salts them.
        select = f"{spec.select_salt}:{isa_type}"
        amount = f"{spec.amount_salt}:{isa_type}"
        for seed, salt, key in (
            (spec.select_seed, f"{select}:cell", "income_3|age_2|female|london"),
            (spec.amount_seed, f"{amount}:walk", "income_3"),
        ):
            offset = stage._keyed_offset(seed, salt, key)
            assert isinstance(offset, cgt_imputation._KeyedOffset)
            assert offset.random() == before([key], seed=seed, salt=salt)[0]
        salt = f"{amount}:jitter"
        jitter = stage.stable_identity_uniform(
            "income_3:2", seed=spec.amount_seed, salt=salt
        )
        assert jitter == before(["income_3:2"], seed=spec.amount_seed, salt=salt)[0]


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
        (75_000.0, 18_000.0),
    ],
)
def test_amounts_scale_to_the_build_year_holding_the_limit(cash_target, stocks_target):
    cash = np.asarray([19_000.0, 5_000.0, 0.0, 10_000.0, 8_000.0, 19_000.0])
    stocks = np.asarray([1_000.0, 0.0, 8_000.0, 5_000.0, 2_000.0, 0.0])
    weights = np.ones(6)
    scaled_cash, scaled_stocks, receipt = stage.scale_to_build_year_amounts(
        cash,
        stocks,
        weights,
        cash_target=cash_target,
        stocks_target=stocks_target,
        limit=20_000.0,
        taper_from=18_000.0,
    )
    # The subscriber at the limit stays exactly where they were.
    assert (scaled_cash[0], scaled_stocks[0]) == (19_000.0, 1_000.0)
    assert ((scaled_cash + scaled_stocks) <= 20_000 + 1e-6).all()
    assert scaled_cash.sum() == pytest.approx(cash_target)
    assert scaled_stocks.sum() == pytest.approx(stocks_target)
    factor = receipt["cash_factor"]
    rising = bool(cash_target > cash.sum())
    assert (factor > 1) is rising
    # Above GBP 18,000 the amount moves on the line from the scaled GBP 18,000
    # to the limit, so a factor below one leaves no empty range under it.
    start = factor * 18_000.0
    expected = start + (19_000.0 - 18_000.0) * (20_000.0 - start) / 2_000.0
    assert scaled_cash[5] == pytest.approx(min(expected, 20_000.0))
    if not rising:
        assert factor * 19_999.0 < scaled_cash[5] < 20_000.0


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
    # Only the self-employed's gross receipts run after it (microcosm#1428).
    assert list(stages)[-2:] == ["isa_subscriptions", "trading_allowance_receipts"]


def test_no_calibration_target_reads_an_isa_fact():
    """The reviewed fence keeps every vendored ISA fact measured only.

    The stage draws subscribers and amounts to these cells, so a target compiled
    from them would be circular (María's ruling of 28 October 2026). Targets
    select facts by concept, so the contract and both compiled reference surfaces
    must name none of the vendored ISA concepts, nor any concept under their
    prefixes, nor the ISA record sets (review of microcosm#1421).
    """

    import json
    from importlib.resources import files

    package = files("microcosm.build.uk")
    vendored = json.loads(
        package.joinpath("hmrc_isa_statistics_facts.json").read_text("utf-8")
    )
    concepts = {row["concept"] for row in vendored["rows"]}
    assert {"hmrc.isa_amount_subscribed", "hmrc.isa_subscribers"} <= concepts
    prefixes = ("hmrc.isa_", "hmrc.lisa_", "hmrc.cash_isa_")
    assert all(concept.startswith(prefixes) for concept in concepts)
    record_sets = (
        "hmrc.isa_statistics_2026",
        "hmrc.lisa_statistics_2026",
        "hmrc.cash_isa_limit_reduction_2026",
    )
    for resource in (
        "uk_population_targets.json",
        "target_references.json",
        "local_target_references.json",
    ):
        text = package.joinpath(resource).read_text("utf-8")
        for name in (*prefixes, *record_sets):
            # As the start of a JSON string: a concept or record-set id.
            assert f'"{name}' not in text, (resource, name)
