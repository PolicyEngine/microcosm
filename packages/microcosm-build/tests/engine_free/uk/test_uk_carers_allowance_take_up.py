"""Carer's Allowance claims by the carers the earnings limit keeps out
(microcosm#1319), among the carers of a severely disabled person
(microcosm#1418)."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.stochastic_assignment import (
    stable_identity_uniforms,
    weight_balanced_uniforms,
)
from microcosm.build.uk_runtime import carers_allowance_take_up as ca
from microcosm.build.uk_runtime.frs_take_up import UK_TAKE_UP_SIGNAL_OUTPUTS
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
from microcosm.frame import WeightKind

_LIMIT = 151.0
_RATE = 81.90
_POLICY = ca.UKCarersAllowancePolicy(
    min_hours=35.0,
    earnings_limit=_LIMIT,
    weekly_rate=_RATE,
    instant="2024-04-30",
    source="test",
)
_OUTPUT = ca.CARERS_ALLOWANCE_TAKE_UP_OUTPUT
_INSIDE = ca.CARERS_ALLOWANCE_IN_HOUSEHOLD_HOURS
#: The stub engine's disability test reads this person column.
_DISABLED = "stub_severely_disabled"


class _Contract:
    def __init__(self, rate: float = 0.6) -> None:
        self._rate = rate
        self.keys: list[str] = []

    def rate(self, key: str, build_year: int | None = None) -> float:
        self.keys.append(key)
        return self._rate


class _StubEngine:
    """The engine's earnings test, weekly pay above the limit, and its
    disability test, the frame's stub column."""

    country = "uk"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str, object]] = []

    def materialize(self, frame, variables, period, *, dataset_period=None):
        self.calls.append((tuple(variables), str(period), dataset_period))
        person = frame.table("person")
        pay = person["employment_income"].to_numpy(dtype=float)
        return {
            ca.CARERS_ALLOWANCE_EARNINGS_TEST: pay / 52.0 > _LIMIT,
            ca.CARERS_ALLOWANCE_DISABILITY_TEST: person[_DISABLED].to_numpy(bool),
            ca.CARERS_ALLOWANCE_EARNINGS: pay,
        }


def _means_tested_columns(receives) -> dict[str, np.ndarray]:
    """The reported means-tested amounts: universal credit where ``receives``."""

    receives = np.asarray(receives, dtype=bool)
    columns = {
        name: np.zeros(receives.size)
        for name in ca.CARERS_ALLOWANCE_MEANS_TESTED_REPORTED
    }
    columns["universal_credit_reported"] = np.where(receives, 5_000.0, 0.0)
    return columns


def _frame(
    *,
    care_hours,
    reported,
    weekly_pay,
    regions=None,
    weights=None,
    source_ids=None,
    lineage: bool = True,
    flag=None,
    time_period: str = "2024",
    in_household_hours=None,
    member_disabled=None,
    means_tested=None,
):
    """A carer and the member they may care for in each household, in a
    benefit unit each; ``source_ids`` makes support copies (each household's
    own id by default), ``lineage=False`` drops them.

    The carers come first, in order, then the members. A carer gives their
    hours inside the household (``in_household_hours``, all of ``care_hours``
    by default), and the member is severely disabled (``member_disabled``,
    true by default), so by default every carer cares for a severely disabled
    member (microcosm#1418). ``means_tested`` gives a carer's benefit unit a
    reported universal credit amount (none by default)."""

    n = len(care_hours)
    ids = np.arange(1, n + 1, dtype="int64")
    reported = np.asarray(reported, dtype=float)
    care_hours = np.asarray(care_hours, dtype=float)
    carer = pd.DataFrame(
        {
            "person_id": ids * 10 + 1,
            "person_household_id": ids,
            "person_benunit_id": ids,
            "age": np.full(n, 45, dtype="int64"),
            "care_hours": care_hours,
            _INSIDE: care_hours
            if in_household_hours is None
            else np.asarray(in_household_hours, dtype=float),
            "carers_allowance_reported": reported,
            "employment_income": np.asarray(weekly_pay, dtype=float) * 52.0,
            _OUTPUT: reported > 0 if flag is None else np.asarray(flag, dtype=bool),
            _DISABLED: np.zeros(n, dtype=bool),
            **_means_tested_columns(
                np.zeros(n, dtype=bool) if means_tested is None else means_tested
            ),
        }
    )
    member = pd.DataFrame(
        {
            "person_id": ids * 10 + 2,
            "person_household_id": ids,
            "person_benunit_id": ids + n,
            "age": np.full(n, 80, dtype="int64"),
            "care_hours": np.zeros(n),
            _INSIDE: np.zeros(n),
            "carers_allowance_reported": np.zeros(n),
            "employment_income": np.zeros(n),
            _OUTPUT: np.zeros(n, dtype=bool),
            _DISABLED: np.ones(n, dtype=bool)
            if member_disabled is None
            else np.asarray(member_disabled, dtype=bool),
            **_means_tested_columns(np.zeros(n, dtype=bool)),
        }
    )
    person = pd.concat([carer, member], ignore_index=True)
    benunit = pd.DataFrame({"benunit_id": np.r_[ids, ids + n]})
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": pd.array(
                ["WALES"] * n if regions is None else list(regions), dtype="string"
            ),
        }
    )
    if lineage:
        household["household_source_id"] = (
            ids if source_ids is None else np.asarray(source_ids, dtype="int64")
        )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=time_period,
        weight_kind=WeightKind.DESIGN,
        household_weights=np.ones(n) if weights is None else np.asarray(weights),
    )


def _run(frame, *, rate: float = 0.6, engine: _StubEngine | None = None):
    return ca.redraw_carers_allowance_take_up(
        frame,
        engine=engine or _StubEngine(),
        contract=_Contract(rate),
        policy=_POLICY,
    )


def _stage():
    return load_country_spec("uk").sources.stage_map()[
        ca.CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME
    ]


def test_eligibility_is_the_pool_no_report_and_over_the_limit() -> None:
    rows = list(itertools.product([False, True], [False, True], [False, True]))
    pool, reported, gainful = (np.asarray(column) for column in zip(*rows, strict=True))
    eligible = ca.carers_allowance_eligible(
        pool=pool, reported=reported, gainfully_employed=gainful
    )
    np.testing.assert_array_equal(eligible, pool & ~reported & gainful)
    assert int(eligible.sum()) == 1


def test_the_pool_holds_carers_of_a_severely_disabled_person() -> None:
    # Six carers: in-household carers with and without a disabled member,
    # short hours, and three whose hours are not all inside the household.
    care = np.array([35.0, 50.0, 35.0, 20.0, 35.0, 100.0])
    inside = np.array([35.0, 50.0, 35.0, 20.0, 0.0, 20.0])
    disabled_member = np.array([True, True, False, True, False, True])
    reported = np.zeros(6, dtype=bool)
    weights = np.ones(6)
    # The in-household carers reporting nothing: two of three have a
    # disabled member, so the other carers join below two thirds.
    draws = np.array([0.9, 0.9, 0.9, 0.9, 0.5, 0.7])
    result = ca.carers_allowance_pool(
        care_hours=care,
        in_household_hours=inside,
        other_member_severely_disabled=disabled_member,
        reported=reported,
        weights=weights,
        pool_draws=draws,
        min_hours=35.0,
    )
    assert result.share == pytest.approx(2 / 3)
    np.testing.assert_array_equal(
        result.in_household, [True, True, True, False, False, False]
    )
    np.testing.assert_array_equal(
        result.other, [False, False, False, False, True, True]
    )
    # A disabled member alone does not bring short hours into the pool, and an
    # other carer's own household plays no part.
    np.testing.assert_array_equal(result.pool, [True, True, False, False, True, False])


def test_the_share_is_measured_on_in_household_carers_reporting_nothing() -> None:
    # A reporter with no disabled member leaves the share alone, weights
    # count, and with no in-household carer no other carer joins.
    result = ca.carers_allowance_pool(
        care_hours=np.array([35.0, 35.0, 35.0, 35.0]),
        in_household_hours=np.array([35.0, 35.0, 35.0, 0.0]),
        other_member_severely_disabled=np.array([True, False, False, False]),
        reported=np.array([False, False, True, False]),
        weights=np.array([3.0, 1.0, 5.0, 1.0]),
        pool_draws=np.zeros(4),
        min_hours=35.0,
    )
    assert result.share == pytest.approx(0.75)
    none = ca.carers_allowance_pool(
        care_hours=np.array([35.0]),
        in_household_hours=np.array([0.0]),
        other_member_severely_disabled=np.array([True]),
        reported=np.array([False]),
        weights=np.ones(1),
        pool_draws=np.zeros(1),
        min_hours=35.0,
    )
    assert none.share is None
    assert not none.pool.any()


def test_another_member_is_anyone_in_the_household_but_the_carer() -> None:
    household = np.array([1, 1, 2, 2, 3])
    disabled = np.array([True, False, True, True, True])
    np.testing.assert_array_equal(
        ca.other_member_severely_disabled(household, disabled),
        [False, True, True, True, False],
    )


def _groups():
    """Carers on every side of each condition, a hundred or more in each."""

    groups = {
        # Over the limit and no report: the drawn groups, Scotland's included.
        "eligible": dict(care=35.0, reported=0.0, pay=200.0, region="WALES", n=400),
        "eligible_ni": dict(
            care=50.0, reported=0.0, pay=250.0, region="NORTHERN_IRELAND", n=100
        ),
        "reporter": dict(care=35.0, reported=81.9, pay=100.0, region="WALES", n=100),
        "reporter_over_limit": dict(
            care=35.0, reported=81.9, pay=200.0, region="LONDON", n=100
        ),
        "under_limit": dict(care=35.0, reported=0.0, pay=120.0, region="WALES", n=100),
        "at_limit": dict(care=35.0, reported=0.0, pay=_LIMIT, region="WALES", n=100),
        "scotland": dict(care=35.0, reported=0.0, pay=200.0, region="SCOTLAND", n=100),
        "scotland_reporter": dict(
            care=35.0, reported=81.9, pay=100.0, region="SCOTLAND", n=100
        ),
        "short_hours": dict(care=20.0, reported=0.0, pay=200.0, region="WALES", n=100),
    }
    labels = np.concatenate([[name] * spec["n"] for name, spec in groups.items()])

    def column(key):
        return np.concatenate([[spec[key]] * spec["n"] for spec in groups.values()])

    frame = _frame(
        care_hours=column("care"),
        reported=column("reported"),
        weekly_pay=column("pay"),
        regions=column("region"),
    )
    return frame, np.concatenate([labels, ["member"] * len(labels)])


def test_the_draw_opens_the_flag_to_the_eligible_carers_only() -> None:
    frame, labels = _groups()
    result = _run(frame)
    person = result.frame.table("person")
    flag = person[_OUTPUT].to_numpy()
    for group in ("reporter", "reporter_over_limit", "scotland_reporter"):
        assert flag[labels == group].all(), group
    # Under or at the limit the survey records them entitled and not claiming;
    # 20 hours do not meet the condition.
    for group in ("under_limit", "at_limit", "short_hours", "member"):
        assert not flag[labels == group].any(), group
    # Scotland is drawn too: there the flag is Carer Support Payment's. The
    # uniforms are balanced within strata: GBP 200 a week is under one weekly
    # rate over the limit and GBP 250 over one, so the two pay levels draw
    # apart, each six tenths of its weight to within one carer (#1418).
    eligible = np.isin(labels, ["eligible", "eligible_ni", "scotland"])
    uniforms = stable_identity_uniforms(
        person["person_id"].to_numpy(), seed=0, salt=_OUTPUT
    )
    band_one = labels == "eligible_ni"
    positions = weight_balanced_uniforms(
        band_one[eligible], uniforms[eligible], np.ones(int(eligible.sum()))
    )
    np.testing.assert_array_equal(flag[eligible], positions < 0.6)
    for stratum in (eligible & ~band_one, band_one):
        assert abs(flag[stratum].sum() - 0.6 * stratum.sum()) <= 1
    assert person[_OUTPUT].dtype == bool
    # A flag only: every other cell, the weights and the mass log pass through.
    before = frame.table("person").drop(columns=[_OUTPUT])
    pd.testing.assert_frame_equal(person.drop(columns=[_OUTPUT]), before)
    assert result.frame.mass_log == frame.mass_log
    np.testing.assert_array_equal(
        result.frame.weights_for("household").values,
        frame.weights_for("household").values,
    )
    evidence = result.evidence()
    assert evidence["stage"] == ca.CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME
    assert evidence["scotland_eligible_support"] == "30+"
    assert evidence["scotland_eligible_weighted_persons"] == pytest.approx(100.0)
    assert evidence["scotland_drawn_support"] == "30+"
    assert evidence["eligible_support"] == "30+"
    assert evidence["eligible_weighted_persons"] == pytest.approx(600.0)
    assert evidence["drawn_share_of_over_limit_pool"] == pytest.approx(
        flag[eligible].mean()
    )
    # Every carer here cares for a disabled member inside the household.
    assert evidence["in_household_disabled_member_share"] == 1.0
    assert evidence["over_limit_carers_support"] == "30+"
    assert evidence["over_limit_carers_weighted_persons"] == pytest.approx(600.0)
    assert evidence["eligible_in_household_support"] == "30+"
    assert evidence["eligible_other_support"] == 0


def test_the_draw_leaves_out_carers_of_no_disabled_member_and_draws_the_others() -> (
    None
):
    # Two hundred carers over the limit give their hours inside the household,
    # half of them to a severely disabled member; two hundred more give none
    # inside it and join at that half share on their pool uniform, whoever
    # they live with.
    n_in, n_out = 200, 200
    n = n_in + n_out
    member_disabled = np.r_[np.arange(n_in) % 2 == 0, np.ones(n_out, dtype=bool)]
    frame = _frame(
        care_hours=np.full(n, 35.0),
        reported=np.zeros(n),
        weekly_pay=np.full(n, 300.0),
        in_household_hours=np.r_[np.full(n_in, 35.0), np.zeros(n_out)],
        member_disabled=member_disabled,
    )
    result = _run(frame, rate=1.0)
    person = result.frame.table("person")
    pool_draws = stable_identity_uniforms(
        person["person_id"].to_numpy()[:n], seed=0, salt=ca.CARERS_ALLOWANCE_POOL_SALT
    )
    # The outside carers are one stratum (no means-tested benefit, all paid one
    # to two weekly rates over the limit), so exactly half of them join.
    balanced = weight_balanced_uniforms(
        np.zeros(n_out), pool_draws[n_in:], np.ones(n_out)
    )
    np.testing.assert_array_equal(
        person[_OUTPUT].to_numpy()[:n],
        np.r_[member_disabled[:n_in], balanced < 0.5],
    )
    assert int(person[_OUTPUT].to_numpy()[n_in:n].sum()) == n_out // 2
    evidence = result.evidence()
    assert evidence["in_household_disabled_member_share"] == pytest.approx(0.5)
    assert evidence["pool_salt"] == ca.CARERS_ALLOWANCE_POOL_SALT
    assert evidence["over_limit_carers_weighted_persons"] == pytest.approx(400.0)
    assert evidence["over_limit_in_household_support"] == "30+"
    assert evidence["over_limit_other_support"] == "30+"
    assert evidence["eligible_in_household_support"] == "30+"
    assert evidence["eligible_other_support"] == "30+"


def test_the_engine_reads_the_earnings_test_on_the_frame_at_the_survey_year() -> None:
    frame, _ = _groups()
    engine = _StubEngine()
    _run(frame, engine=engine)
    # One consumed-only read on the frame's own values, never projected.
    assert engine.calls == [(ca.CARERS_ALLOWANCE_ENGINE_VARIABLES, "2024", None)]


def test_the_weighted_share_drawn_is_the_rate() -> None:
    rng = np.random.default_rng(7)
    n = 6_000
    pay = rng.uniform(80.0, 400.0, n)
    care = rng.choice([0.0, 20.0, 35.0, 50.0, 100.0], n)
    reported = np.where(rng.random(n) < 0.1, 81.9, 0.0)
    regions = rng.choice(["WALES", "LONDON", "SCOTLAND", "NORTHERN_IRELAND"], n)
    weights = rng.uniform(0.5, 3.0, n)
    frame = _frame(
        care_hours=care,
        reported=reported,
        weekly_pay=pay,
        regions=regions,
        weights=weights,
    )
    result = _run(frame)
    flag = result.frame.table("person")[_OUTPUT].to_numpy()[:n]
    eligible = (care >= 35.0) & (reported == 0.0) & (pay > _LIMIT)
    share = weights[eligible & flag].sum() / weights[eligible].sum()
    assert share == pytest.approx(0.6, abs=0.03)
    evidence = result.evidence()
    assert evidence["drawn_share_of_over_limit_pool"] == pytest.approx(share)
    assert evidence["eligible_weighted_persons"] == pytest.approx(
        weights[eligible].sum()
    )
    assert evidence["rate"] == 0.6
    assert evidence["weekly_earnings_limit"] == _LIMIT
    assert evidence["min_hours"] == 35.0


def test_draws_are_keyed_by_person_identity() -> None:
    frame, _ = _groups()
    first = _run(frame).frame.table("person")[_OUTPUT]
    again = _run(frame).frame.table("person")[_OUTPUT]
    pd.testing.assert_series_equal(first, again)
    # A person's uniform reads its own id, so the stream does not depend on
    # which other people the frame carries.
    ids = frame.table("person")["person_id"].to_numpy()
    subset = ids[np.random.default_rng(3).random(len(ids)) < 0.5]
    full = pd.Series(stable_identity_uniforms(ids, seed=0, salt=_OUTPUT), index=ids)
    np.testing.assert_array_equal(
        full.loc[subset].to_numpy(),
        stable_identity_uniforms(subset, seed=0, salt=_OUTPUT),
    )


def test_the_receipt_bands_support_copies_on_their_sources() -> None:
    # UKDS disclosure control: twelve support copies of three source
    # households carry nothing above "<10", however many rows copy them.
    n = 12
    frame = _frame(
        care_hours=np.full(n, 35.0),
        reported=np.zeros(n),
        weekly_pay=np.full(n, 300.0),
        source_ids=np.arange(n) % 3,
    )
    evidence = _run(frame, rate=1.0).evidence()
    assert evidence["eligible_support"] == "<10"
    assert evidence["eligible_weighted_persons"] == "<10"
    assert evidence["drawn_support"] == "<10"
    assert evidence["drawn_share_of_over_limit_pool"] == "<10"
    assert evidence["in_household_disabled_member_share"] == "<10"


def test_the_share_hides_a_small_side() -> None:
    # Forty eligible carers on forty sources, three of them drawn: the share
    # times the eligible weight would give the three carers' weight back.
    n = 40
    frame = _frame(
        care_hours=np.full(n, 35.0),
        reported=np.zeros(n),
        weekly_pay=np.full(n, 300.0),
    )
    # One stratum of equal weights: the k-th uniform sits at (k + 1/2) / 40.
    result = _run(frame, rate=3 / n)
    assert int(result.frame.table("person")[_OUTPUT].sum()) == 3
    evidence = result.evidence()
    assert evidence["eligible_support"] == "30+"
    assert evidence["eligible_weighted_persons"] == pytest.approx(40.0)
    assert evidence["drawn_support"] == "<10"
    assert evidence["drawn_share_of_over_limit_pool"] == "<10"
    # All but three drawn: the share would give the three undrawn carers'
    # weight back the same way.
    result = _run(frame, rate=37 / n)
    assert int(result.frame.table("person")[_OUTPUT].sum()) == 37
    evidence = result.evidence()
    assert evidence["drawn_support"] == "30+"
    assert evidence["drawn_share_of_over_limit_pool"] == "<10"


def test_earnings_bands_read_the_weekly_earnings_the_engine_tests() -> None:
    weekly = np.array(
        [
            100.0,
            _LIMIT,
            _LIMIT + 0.004,
            _LIMIT + 0.01,
            _LIMIT + _RATE - 0.05,
            _LIMIT + _RATE + 0.05,
            _LIMIT + 2.5 * _RATE,
            _LIMIT + 9 * _RATE,
        ]
    )
    bands = ca.carers_allowance_earnings_band(
        weekly * 52, earnings_limit=_LIMIT, weekly_rate=_RATE
    )
    # At or under the limit, a fraction of a penny over included, is -1; the
    # last band is open.
    assert bands.tolist() == [-1, -1, -1, 0, 0, 1, 2, 3]
    with pytest.raises(ValueError, match="weekly rate"):
        ca.carers_allowance_earnings_band(weekly, earnings_limit=_LIMIT, weekly_rate=0)


def test_a_means_tested_report_marks_the_whole_benefit_unit() -> None:
    reported = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [0.0, 3.0]])
    marks = ca.means_tested_benefit_unit(np.array([1, 1, 2, 3]), reported)
    assert marks.tolist() == [True, True, False, True]


def _strata_frame(n_per: int = 40, *, outside: bool = False):
    """Eligible carers in eight strata of unequal weights: on and off a
    means-tested benefit, by four pay levels a band apart."""

    pays = (_LIMIT + 0.5 * _RATE, _LIMIT + 1.5 * _RATE, _LIMIT + 2.5 * _RATE, 600.0)
    cells = list(itertools.product((False, True), pays))
    n = n_per * len(cells)
    rng = np.random.default_rng(5)
    frame = _frame(
        care_hours=np.full(n, 35.0),
        reported=np.zeros(n),
        weekly_pay=np.repeat([pay for _, pay in cells], n_per),
        means_tested=np.repeat([on for on, _ in cells], n_per),
        weights=rng.uniform(0.5, 3.0, n),
        in_household_hours=np.zeros(n) if outside else None,
    )
    stratum = np.repeat(np.arange(len(cells)), n_per)
    return frame, stratum, n


def test_each_stratum_claims_the_rate_of_its_eligible_weight() -> None:
    frame, stratum, n = _strata_frame()
    result = _run(frame, rate=0.6)
    flag = result.frame.table("person")[_OUTPUT].to_numpy()[:n]
    weights = frame.weights_for("household").values
    for code in np.unique(stratum):
        members = stratum == code
        claimed = float(weights[members & flag].sum())
        assert abs(claimed - 0.6 * weights[members].sum()) <= weights[members].max()
    rows = result.evidence()["take_up_strata"]
    assert len(rows) == 8
    assert {row["route"] for row in rows} == {"in_household"}
    assert {row["means_tested"] for row in rows} == {False, True}
    assert {row["earnings_band"] for row in rows} == {
        "0 to 1 weekly rates over",
        "1 to 2 weekly rates over",
        "2 to 3 weekly rates over",
        "3 or more weekly rates over",
    }
    for row in rows:
        assert row["support"] == "30+"
        assert row["target"] == 0.6
        assert row["realised_share"] == pytest.approx(0.6, abs=0.06)
    assert result.evidence()["pool_strata"] == []
    # Identity-keyed, so a rerun draws the same carers.
    rerun = _run(frame, rate=0.6).frame.table("person")[_OUTPUT].to_numpy()[:n]
    np.testing.assert_array_equal(rerun, flag)


def test_the_stage_refuses_a_stratum_off_its_share_by_more_than_a_record(
    monkeypatch,
) -> None:
    # Anthony's review of microcosm#1420: the balance is enforced, not only
    # reported. Every uniform at zero draws each stratum whole.
    frame, _, _ = _strata_frame()
    assert _run(frame, rate=0.6).evidence()["strata_within_one_record"] is True
    monkeypatch.setattr(
        ca,
        "weight_balanced_uniforms",
        lambda strata, draws, weights: np.zeros(len(draws)),
    )
    with pytest.raises(ValueError, match="take-up stratum .* more than one record"):
        _run(frame, rate=0.6)


def test_a_stratum_may_miss_its_share_by_up_to_one_record() -> None:
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    members = np.ones(4, dtype=bool)
    strata = np.zeros(4, dtype=np.int64)
    # Half of 10 is 5: taking 3 + 4 misses by 2, under the heaviest record.
    ca.assert_strata_within_one_record(
        members,
        strata,
        np.array([False, False, True, True]),
        weights,
        0.5,
        label="pool",
    )
    with pytest.raises(ValueError, match="pool stratum 0 misses its share"):
        ca.assert_strata_within_one_record(
            members, strata, np.ones(4, dtype=bool), weights, 0.5, label="pool"
        )


def test_the_outside_pool_draws_its_share_within_each_stratum() -> None:
    # Every carer gives the hours outside the household: none is in the
    # in-household base, so the share is that of a separate in-household half.
    frame, stratum, n = _strata_frame(outside=True)
    inside, _, n_in = _strata_frame(n_per=10)
    person = pd.concat(
        [
            frame.table("person").iloc[:n],
            inside.table("person")
            .iloc[:n_in]
            .assign(
                person_id=lambda df: df["person_id"] + 100_000,
                person_household_id=lambda df: df["person_household_id"] + 10_000,
                person_benunit_id=lambda df: df["person_benunit_id"] + 10_000,
            ),
            frame.table("person").iloc[n:],
            inside.table("person")
            .iloc[n_in:]
            .assign(
                person_id=lambda df: df["person_id"] + 100_000,
                person_household_id=lambda df: df["person_household_id"] + 10_000,
                person_benunit_id=lambda df: df["person_benunit_id"] + 10_000,
                **{_DISABLED: lambda df: np.arange(len(df)) % 2 == 0},
            ),
        ],
        ignore_index=True,
    )
    household = pd.concat(
        [
            frame.table("household"),
            inside.table("household").assign(
                household_id=lambda df: df["household_id"] + 10_000,
                household_source_id=lambda df: df["household_source_id"] + 10_000,
            ),
        ],
        ignore_index=True,
    )
    weights = np.r_[
        frame.weights_for("household").values, inside.weights_for("household").values
    ]
    combined = uk_national_frame(
        person=person,
        benunit=pd.DataFrame(
            {"benunit_id": np.sort(person["person_benunit_id"].unique())}
        ),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
        household_weights=weights,
    )
    result = _run(combined, rate=1.0)
    out = result.frame.table("person")
    flag = out.set_index("person_id")[_OUTPUT]
    drawn = flag.loc[frame.table("person")["person_id"].to_numpy()[:n]].to_numpy()
    share = result.evidence()["in_household_disabled_member_share"]
    outside_weights = frame.weights_for("household").values
    for code in np.unique(stratum):
        members = stratum == code
        claimed = float(outside_weights[members & drawn].sum())
        assert (
            abs(claimed - share * outside_weights[members].sum())
            <= outside_weights[members].max()
        )
    assert len(result.evidence()["pool_strata"]) == 8


def test_households_are_read_through_membership_not_position() -> None:
    # Two people to a household, the person table shuffled against the
    # households (group tables stay sorted by id), and a region, a weight and
    # a lineage of each household's own: a positional read puts a carer in
    # the wrong household.
    rng = np.random.default_rng(11)
    household_ids = np.arange(1001, 1025)
    regions = np.asarray(["SCOTLAND", "WALES", "NORTHERN_IRELAND"] * 8)
    weights = 1.0 + np.arange(24, dtype=float)
    pay = np.r_[np.full(24, 300.0), np.full(24, 100.0)]
    person = pd.DataFrame(
        {
            "person_id": np.r_[household_ids * 10 + 1, household_ids * 10 + 2],
            "person_household_id": np.r_[household_ids, household_ids],
            "person_benunit_id": np.r_[household_ids, household_ids],
            "age": np.full(48, 45, dtype="int64"),
            # The first member cares 35 hours over the limit for the second,
            # who is severely disabled and does not care.
            "care_hours": np.r_[np.full(24, 35.0), np.zeros(24)],
            _INSIDE: np.r_[np.full(24, 35.0), np.zeros(24)],
            "carers_allowance_reported": np.zeros(48),
            "employment_income": pay * 52.0,
            _OUTPUT: np.zeros(48, dtype=bool),
            _DISABLED: np.r_[np.zeros(24, dtype=bool), np.ones(24, dtype=bool)],
            **_means_tested_columns(np.zeros(48, dtype=bool)),
        }
    ).iloc[rng.permutation(48)]
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "region": pd.array(regions, dtype="string"),
            "household_source_id": household_ids * 10,
        }
    )
    frame = uk_national_frame(
        person=person.reset_index(drop=True),
        benunit=pd.DataFrame({"benunit_id": household_ids}),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
        household_weights=weights,
    )
    result = _run(frame, rate=1.0)
    out = result.frame.table("person")
    region_of = dict(zip(household_ids, regions, strict=True))
    scottish = out["person_household_id"].map(region_of).eq("SCOTLAND").to_numpy()
    carer = out["care_hours"].to_numpy() >= 35.0
    np.testing.assert_array_equal(out[_OUTPUT].to_numpy(), carer)
    evidence = result.evidence()
    # Twenty-four households, one eligible carer in each, eight in Scotland.
    assert evidence["eligible_weighted_persons"] == pytest.approx(weights.sum())
    assert evidence["eligible_support"] == "10-29"
    assert evidence["drawn_share_of_over_limit_pool"] == 1.0
    assert evidence["scotland_eligible_support"] == "<10"
    assert evidence["scotland_eligible_weighted_persons"] == "<10"
    assert int(carer[scottish].sum()) == 8


def test_a_frame_without_eligible_carers_draws_nothing() -> None:
    frame = _frame(
        care_hours=np.full(3, 35.0),
        reported=np.array([81.9, 0.0, 0.0]),
        weekly_pay=np.array([100.0, 100.0, 120.0]),
    )
    result = _run(frame)
    assert result.frame.table("person")[_OUTPUT].tolist()[:3] == [True, False, False]
    evidence = result.evidence()
    assert evidence["eligible_support"] == 0
    assert evidence["eligible_weighted_persons"] == 0.0
    assert evidence["drawn_share_of_over_limit_pool"] is None


@pytest.mark.parametrize("period", ["2023", "2025", "2024-25"])
def test_the_stage_refuses_a_frame_off_the_survey_year(period) -> None:
    frame = _frame(
        care_hours=np.full(3, 35.0),
        reported=np.zeros(3),
        weekly_pay=np.full(3, 300.0),
        time_period=period,
    )
    engine = _StubEngine()
    # The earnings test is read at the survey year's limit on the frame's own
    # values: another year, or a fiscal-year label, is refused by name.
    with pytest.raises(ValueError, match="survey_year 2024"):
        _run(frame, engine=engine)
    assert engine.calls == []


def test_the_stage_refuses_a_frame_without_lineage() -> None:
    frame = _frame(
        care_hours=np.full(3, 35.0),
        reported=np.zeros(3),
        weekly_pay=np.full(3, 300.0),
        lineage=False,
    )
    with pytest.raises(ValueError, match="household_source_id"):
        _run(frame)


def test_the_stage_refuses_a_null_lineage() -> None:
    frame = _frame(
        care_hours=np.full(3, 35.0),
        reported=np.zeros(3),
        weekly_pay=np.full(3, 300.0),
    )
    frame.table("household")["household_source_id"] = pd.array(
        [1, None, 3], dtype="Int64"
    )
    # A null would drop out of a distinct count or count as one source.
    with pytest.raises(ValueError, match="null"):
        _run(frame)


def test_the_stage_refuses_a_flag_that_left_reported_receipt() -> None:
    frame = _frame(
        care_hours=np.full(3, 35.0),
        reported=np.array([81.9, 0.0, 0.0]),
        weekly_pay=np.full(3, 100.0),
        flag=np.array([True, True, False]),
    )
    engine = _StubEngine()
    # The stage rebuilds the flag from reported receipt, so it refuses to
    # overwrite another writer's flag.
    with pytest.raises(ValueError, match="must arrive as reported receipt"):
        _run(frame, engine=engine)
    assert engine.calls == []


def test_the_stage_refuses_missing_inputs() -> None:
    frame = _frame(
        care_hours=np.full(3, 35.0),
        reported=np.zeros(3),
        weekly_pay=np.full(3, 300.0),
    )
    person = frame.table("person").drop(columns=["care_hours"])
    reduced = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
        household_weights=np.ones(3),
    )
    with pytest.raises(ValueError, match=r"person columns missing: \['care_hours'\]"):
        _run(reduced)


def test_the_transform_runs_the_packaged_stage_at_the_contract_rate() -> None:
    transform = ca.UKCarersAllowanceTakeUpStageTransform(
        stage=_stage(), engine=_StubEngine(), policy=_POLICY
    )
    with pytest.raises(RuntimeError, match="completed stage run"):
        transform.checkpoint_metadata()
    frame, _ = _groups()
    result = transform(frame)
    assert transform.output_columns() == ()
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == ca.CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME
    assert evidence["rate_key"] == ca.CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY
    assert evidence["rate"] == 0.6
    assert evidence["eligibility"] == ca.CARERS_ALLOWANCE_ELIGIBILITY
    assert result.table("person")[_OUTPUT].any()


def test_the_packaged_stage_carries_the_reviewed_operations_in_order() -> None:
    stage = _stage()
    # Closed world: the transform refuses any drift from the code's constants.
    ca._assert_stage_parameters(stage)
    assert stage.grain == "person"
    assert stage.outputs == ()
    assert stage.rewrites == (_OUTPUT,)
    materialize, assign = stage.operations
    assert tuple(materialize.parameters["predictors"]) == (
        ca.CARERS_ALLOWANCE_ENGINE_VARIABLES
    )
    assert assign.parameters["rate_key"] == ca.CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY
    assert assign.parameters["seed"] == 0
    order = [stage.stage for stage in load_country_spec("uk").sources.stages]
    position = order.index(ca.CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME)
    # After the SPI income chain and the other take-up redraws, before the
    # deduction attributes and every CGT clone.
    assert order[position - 1] == "child_benefit_take_up"
    assert order[position + 1] == "uc_deduction_attributes"
    assert position > order.index("hmrc_spi_income_spine")
    assert position < order.index("cgt_support_split")


def test_a_drifted_manifest_is_refused() -> None:
    stage = _stage()
    operations = tuple(
        replace(
            operation,
            parameters={**operation.parameters, "rate_key": "pension_credit"},
        )
        if operation.kind == "assign_binary_from_rate"
        else operation
        for operation in stage.operations
    )
    transform = ca.UKCarersAllowanceTakeUpStageTransform(
        stage=replace(stage, operations=operations),
        engine=_StubEngine(),
        policy=_POLICY,
    )
    frame, _ = _groups()
    with pytest.raises(ValueError, match="parameters drifted"):
        transform(frame)
    with pytest.raises(ValueError, match="exactly the rewrite"):
        ca._assert_stage_parameters(replace(stage, rewrites=()))


def test_the_contract_carries_the_newly_entitled_rate() -> None:
    contract = load_uk_take_up_contract()
    entry = contract.entry(ca.CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY)
    assert contract.rate(entry.key) == 0.6
    assert entry.raw["output"] == _OUTPUT
    assert entry.raw["entity"] == "person"
    # No published take-up rate exists for newly entitled carers: the value
    # is a ruling, and the back-test against the Independent Review's 61,000
    # is its follow-up.
    assert entry.source["status"] == "frozen_by_adjudication"
    assert entry.source["freeze"]["followup"] == "microcosm#1319"


def test_the_flag_stays_out_of_the_terminal_signal_gate() -> None:
    assert all(output != _OUTPUT for _, output, _ in UK_TAKE_UP_SIGNAL_OUTPUTS)


def test_the_graph_node_reads_the_lineage_and_rewrites_the_flag_on_its_own_boundary() -> (
    None
):
    from microcosm.build.uk_runtime.graph import uk_spine_graph

    graph = uk_spine_graph(source_mode="split")
    node = graph.node(ca.CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME)
    inputs = {
        (slice_.entity, column) for slice_ in node.inputs for column in slice_.columns
    }
    # An open surface: the receipt's lineage and the eligibility's inputs
    # reach the stage in a graph build.
    assert {
        ("household", "household_source_id"),
        ("household", "region"),
        ("person", "care_hours"),
        ("person", _INSIDE),
        ("person", "carers_allowance_reported"),
    } <= inputs
    assert [
        (owned.entity, owned.column, owned.dtype, owned.rewrite)
        for owned in node.outputs
    ] == [("person", _OUTPUT, "bool", True)]
    # The open-surface readers before it bind the incumbent flag, so the
    # rewrite runs on a boundary of its own.
    assert node.population == f"{ca.CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME}.boundary"
