"""Tests split from packages/microcosm-build/tests/test_uk_frs_stochastic.py."""

# ruff: noqa: F403, F405
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.uk_runtime.frs_legacy_proxies import (
    UK_LEGACY_PROXY_PREDICTORS,
    derive_frs_legacy_proxies,
)
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR
from microcosm.build.uk_runtime.frs_take_up import (
    uk_take_up_signal_gate,
)
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from test_support.microcosm_build.uk_frs_stochastic import *

_ENGINE = PolicyEngineUKEngine()


def _engine_frame(
    ages,
    *,
    period: str,
    person_ids=None,
    household_weights=None,
    genders=None,
    benunit_of=None,
    reported_uc=None,
):
    """A frame the engine can simulate: one household per benefit unit."""

    ages = np.asarray(ages)
    n = len(ages)
    ids = np.asarray(person_ids if person_ids is not None else np.arange(n) + 1)
    benunit_of = np.asarray(benunit_of if benunit_of is not None else np.arange(n))
    units = np.unique(benunit_of)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": 1000 + benunit_of,
            "person_household_id": 1000 + benunit_of,
            "age": ages,
            "gender": genders
            if genders is not None
            else np.where(np.arange(n) % 2 == 0, "MALE", "FEMALE"),
            "child_benefit_reported": np.zeros(n),
            "pension_credit_reported": np.zeros(n),
            "universal_credit_reported": np.asarray(
                reported_uc if reported_uc is not None else np.zeros(n), dtype=float
            ),
        }
    )
    weights = np.asarray(
        household_weights if household_weights is not None else np.ones(len(units)),
        dtype=float,
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": 1000 + units, "is_married": False}),
        household=pd.DataFrame(
            {
                "household_id": 1000 + units,
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        time_period=period,
        household_weights=weights,
    )


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    period=st.sampled_from(["2024", "2025"]),
    ages=st.lists(st.integers(0, 100), min_size=1, max_size=40),
    id_offset=st.integers(1, 10**9),
    data=st.data(),
)
def test_engine_working_age_is_the_whole_age_rule_at_the_build_periods(
    period, ages, id_offset, data
) -> None:
    """At 2024-25 and 2025-26 is_WA_adult is exactly ``18 <= age < 66``.

    Whatever the sexes, person ids and weights, so the per-person read draws
    exactly the population the whole-age rule drew before (the current build
    period is 2024): no take-up draw moves.
    """

    n = len(ages)
    weights = data.draw(
        st.lists(st.floats(0.0, 5000.0), min_size=n, max_size=n), label="weights"
    )
    genders = data.draw(
        st.lists(st.sampled_from(["MALE", "FEMALE"]), min_size=n, max_size=n),
        label="genders",
    )
    # Zero weights are allowed (the SPI channel's support copies carry none);
    # one positively weighted adult keeps the frame's total mass positive.
    ages = [*ages, 40]
    frame = _engine_frame(
        ages,
        period=period,
        person_ids=id_offset + np.arange(n + 1),
        household_weights=[*weights, 1.0],
        genders=[*genders, "FEMALE"],
    )

    population = uk_take_up_population(frame, _ENGINE)

    assert population.working_age_adult.tolist() == (
        whole_age_working_age_adult(ages).tolist()
    )


def _split_cohort_frame(period: str):
    # 600 people aged 66 in their own benefit units, both sexes, with weights
    # 1-3, plus control ages either side.
    ages = np.repeat([30, 65, 66, 67, 70], [4, 4, 600, 4, 4])
    return _engine_frame(
        ages,
        period=period,
        household_weights=1.0 + np.arange(len(ages)) % 3,
    ), ages


@pytest.mark.parametrize(
    ("period", "share_under"),
    [
        # People born 6 April 1960 to 5 March 1961 reach State Pension age at
        # 66 years and 1 to 11 months, and from 6 March 1961 at 67 (Pensions
        # Act 1995 Sch 4 para 1, as amended by the Pensions Act 2014 s.26).
        # On 6 October 2026 about a quarter of 66-year-olds are still under
        # it, and on 6 October 2027 about three quarters.
        ("2026", 0.25),
        ("2027", 0.75),
    ],
)
def test_take_up_population_splits_66_year_olds_by_date_of_birth(
    period, share_under
) -> None:
    frame, ages = _split_cohort_frame(period)
    weights = frame.weights_for("household").values

    population = uk_take_up_population(frame, _ENGINE)
    working = population.working_age_adult

    # The engine spreads birthdays over the year by weight within each age
    # and sex, so the weighted share is the statutory one to within about a
    # month of births.
    at_66 = ages == 66
    assert np.average(working[at_66], weights=weights[at_66]) == pytest.approx(
        share_under, abs=1 / 12
    )
    assert working[ages == 30].all() and working[ages == 65].all()
    assert not working[ages == 67].any() and not working[ages == 70].any()
    # The population is exactly the engine's own is_WA_adult on this frame.
    direct = _ENGINE.materialize(frame, ("is_WA_adult",), period)["is_WA_adult"]
    assert working.tolist() == np.asarray(direct, dtype=bool).tolist()


@pytest.mark.parametrize("period", ["2026", "2027"])
def test_take_up_stage_and_gate_use_the_engine_population(period) -> None:
    """End to end: the stage draws over the engine's population, the gate
    measures on it.

    A unit whose only adult is a 66-year-old over State Pension age is never
    drawn unless it reported Universal Credit; one whose 66-year-old is under
    it is in the population and can be. The gate reads the same population
    from a NaN-bearing frame, as the SPI channel leaves one.
    """

    frame, ages = _split_cohort_frame(period)
    working = uk_take_up_population(frame, _ENGINE).working_age_adult
    over_66 = np.flatnonzero((ages == 66) & ~working)
    reported = np.zeros(len(ages))
    reported[over_66[0]] = 100.0  # an anchor outside the population
    frame.table("person")["universal_credit_reported"] = reported

    transformed = UKFRSTakeUpStageTransform(
        contract=_Contract(), stage=_take_up_stage(), engine=_ENGINE
    )(frame)

    # One person per benefit unit, in person order.
    claim = transformed.table("benunit")["would_claim_uc"].to_numpy(dtype=bool)
    outside = ~working & (ages >= 18)
    assert not claim[outside & (reported == 0)].any()
    assert claim[over_66[0]]
    # Inside the population the residual is drawn unit by unit at the
    # contract rate (0.5), so the share is near it, not exactly it.
    assert claim[working].mean() == pytest.approx(0.5, abs=0.1)

    # The real adapter refuses NaN; the gate reads a filled copy and still
    # measures the frame it was given.
    transformed.table("person")["other_investment_income"] = np.where(
        ages == 66, np.nan, 0.0
    )
    with pytest.raises(ValueError, match="NaN"):
        _ENGINE.materialize(transformed, ("is_WA_adult",), period)
    result = uk_take_up_signal_gate(transformed, contract=_Contract(), engine=_ENGINE)

    detail = result.details["benunit.would_claim_uc"]
    assert detail["population_units"] == int(working.sum())
    evidence = result.details["universal_credit_population"]
    assert evidence["period"] == period
    assert evidence["working_age_adults"] == int(working.sum())
    assert evidence["source"] == (
        f"policyengine-uk {metadata.version('policyengine-uk')} is_WA_adult"
    )


@pytest.mark.parametrize("period", ["2024", "2025", "2026"])
def test_legacy_proxies_read_state_pension_age_status_from_the_engine(period) -> None:
    """The declared predictor gives the engine's per-person status.

    The engine's per-person ``state_pension_age`` is a fractional age, so a
    whole-year ``age < state_pension_age`` would put 66-year-olds who are over
    State Pension age under it. ``is_SP_age`` agrees with ``is_WA_adult``
    for every adult, and 66-year-olds over it get no legacy proxy.
    """

    frame, ages = _split_cohort_frame(period)
    materialized = _ENGINE.materialize(
        frame, (*UK_LEGACY_PROXY_PREDICTORS, "is_WA_adult"), period
    )
    over = np.asarray(materialized["is_SP_age"])
    assert over.dtype.kind == "b"
    adults = ages >= 18
    assert (~over[adults] == np.asarray(materialized["is_WA_adult"])[adults]).all()

    person = frame.table("person").assign(
        employment_status="UNEMPLOYED",
        hours_worked=0.0,
        current_education="NOT_IN_EDUCATION",
    )
    proxies = derive_frs_legacy_proxies(
        person,
        employment_status_reported=np.ones(len(ages), dtype=bool),
        over_state_pension_age=over,
        max_annual_hours=16 * WEEKS_IN_YEAR,
    )
    jobseeker = proxies["legacy_jobseeker_proxy"].to_numpy(dtype=bool)
    assert (jobseeker == (adults & ~over)).all()
    if period in ("2024", "2025"):
        # Every 66-year-old is over State Pension age before 2026-27.
        assert not jobseeker[ages == 66].any()
