from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import cgt_imputation
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_AGE_GROUP_LOWER_BOUNDS,
    UK_CGT_IMPUTATION_STAGE_NAME,
    UK_CGT_MASS_CONSERVATION_REASON,
    UK_CGT_REGION_GROUP_LABELS,
    UK_CGT_REGION_GROUPS,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
    _band_plans,
    _joint_plans,
    _pareto_quantile,
    _rake_allocation_targets,
    _truncated_exponential_quantile,
    impute_uk_capital_gains,
    impute_uk_capital_gains_with_report,
    summarize_uk_cgt_imputation,
    uk_capital_gains_imputation_stage,
    uk_cgt_spine_stage_transform,
    uk_cgt_taxable_income_proxy,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRCCapitalGainsBandTotal,
    HMRCCapitalGainsCell,
    HMRCCapitalGainsIncomeTotal,
    HMRCCapitalGainsJointDistribution,
    HMRCCapitalGainsSourceProvenance,
    load_hmrc_cgt_conditioning_facts,
    load_hmrc_cgt_joint_distribution,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import Frame

PARAMETERS = UKCGTPolicyParameters(
    personal_allowance=12_570.0,
    personal_allowance_taper_threshold=100_000.0,
    personal_allowance_taper_rate=0.5,
    annual_exempt_amount=6_000.0,
    instant="2023-06-01",
    source="test",
)


def _distribution(
    *, cell_people: float = 1_000.0, suppress_top_low_income: bool = False
) -> HMRCCapitalGainsJointDistribution:
    """A synthetic joint distribution with in-band cell means."""
    bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    uppers = (*bounds[1:], None)
    cells = []
    band_totals = []
    column_people = dict.fromkeys(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, 0.0)
    column_gains = dict.fromkeys(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, 0.0)
    for lower, upper in zip(bounds, uppers, strict=True):
        if upper is None:
            mean = lower * 2.0
        else:
            mean = lower + (upper - lower) * 0.4
        band_people = 0.0
        band_gains = 0.0
        for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS:
            suppressed = (
                suppress_top_low_income
                and upper is None
                and income_lower == HMRC_CGT_INCOME_BAND_LOWER_BOUNDS[0]
            )
            # Real suppression withholds the count and keeps the amount in
            # all but one published cell, so the fixture matches that shape.
            people = None if suppressed else cell_people
            gains = cell_people * mean
            cells.append(
                HMRCCapitalGainsCell(
                    gain_lower_bound=lower,
                    income_lower_bound=income_lower,
                    individuals=people,
                    gains=gains,
                )
            )
            band_people += cell_people
            band_gains += cell_people * mean
            column_people[income_lower] += cell_people
            column_gains[income_lower] += cell_people * mean
        band_totals.append(
            HMRCCapitalGainsBandTotal(
                gain_lower_bound=lower,
                individuals=band_people,
                gains=band_gains,
            )
        )
    income_totals = tuple(
        HMRCCapitalGainsIncomeTotal(
            income_lower_bound=income_lower,
            individuals=column_people[income_lower],
            gains=column_gains[income_lower],
        )
        for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    )
    return HMRCCapitalGainsJointDistribution(
        cells=tuple(cells),
        band_totals=tuple(band_totals),
        income_totals=income_totals,
        source=HMRCCapitalGainsSourceProvenance(
            resource="synthetic.json",
            resource_sha256="synthetic",
            source_commit="synthetic",
            record_set_prefix="synthetic.",
            source_file="synthetic.ods",
            source_sha256="synthetic",
            source_vintage="2024-25",
            build_period="2024",
        ),
        total_individuals=sum(t.individuals for t in band_totals),
        total_gains=sum(t.gains for t in band_totals),
    )


def _frame(
    person_rows: int,
    *,
    gains,
    incomes,
    ages=None,
    regions=None,
    weights=None,
) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.arange(person_rows, dtype="int64"),
            "person_household_id": np.arange(person_rows, dtype="int64"),
            "person_benunit_id": np.arange(person_rows, dtype="int64"),
            "capital_gains": np.asarray(gains, dtype=float),
            "employment_income": np.asarray(incomes, dtype=float),
            "age": (
                np.full(person_rows, 45, dtype="int64")
                if ages is None
                else np.asarray(ages, dtype="int64")
            ),
        }
    )
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        if column not in person.columns:
            person[column] = 0.0
    household = pd.DataFrame(
        {
            "household_id": np.arange(person_rows, dtype="int64"),
            "household_weight": (
                np.full(person_rows, 100.0)
                if weights is None
                else np.asarray(weights, dtype=float)
            ),
            "region": (
                np.full(person_rows, "LONDON", dtype=object)
                if regions is None
                else np.asarray(regions, dtype=object)
            ),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(person_rows, dtype="int64")})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


class TestTaxableIncomeProxy:
    def test_subtracts_the_personal_allowance(self) -> None:
        person = pd.DataFrame(
            {column: [0.0] for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS}
        )
        person["employment_income"] = [30_000.0]

        proxy = uk_cgt_taxable_income_proxy(person, PARAMETERS)

        assert proxy[0] == pytest.approx(30_000.0 - 12_570.0)

    def test_tapers_the_allowance_above_the_threshold(self) -> None:
        person = pd.DataFrame(
            {column: [0.0, 0.0] for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS}
        )
        # £120,000 tapers the allowance to £2,570; £130,000 removes it.
        person["employment_income"] = [120_000.0, 130_000.0]

        proxy = uk_cgt_taxable_income_proxy(person, PARAMETERS)

        assert proxy[0] == pytest.approx(120_000.0 - 2_570.0)
        assert proxy[1] == pytest.approx(130_000.0)

    def test_rejects_a_missing_component(self) -> None:
        person = pd.DataFrame({"employment_income": [1.0]})

        with pytest.raises(ValueError, match="components missing"):
            uk_cgt_taxable_income_proxy(person, PARAMETERS)


class TestWithinBandDraws:
    def test_truncated_exponential_matches_the_target_mean(self) -> None:
        rng = np.random.default_rng(0)
        quantiles = rng.random(200_000)

        for target in (30_000.0, 50_000.0, 70_000.0):
            draws = _truncated_exponential_quantile(
                quantiles, 25_000.0, 100_000.0, target
            )
            assert draws.min() >= 25_000.0
            assert draws.max() <= 100_000.0
            assert draws.mean() == pytest.approx(target, rel=5e-3)

    def test_midpoint_mean_degenerates_to_uniform(self) -> None:
        quantiles = np.asarray([0.0, 0.25, 0.5, 1.0])

        draws = _truncated_exponential_quantile(quantiles, 0.0, 100.0, 50.0)

        assert draws == pytest.approx([0.0, 25.0, 50.0, 100.0])

    def test_pareto_matches_the_published_mean(self) -> None:
        rng = np.random.default_rng(0)
        quantiles = rng.random(500_000)

        draws = _pareto_quantile(quantiles, 5_000_000.0, 11_357_000.0)

        assert draws.min() >= 5_000_000.0
        # alpha ~ 1.79 has a heavy tail; the sample mean converges slowly,
        # so hold it loosely and the median (analytic) tightly.
        assert draws.mean() == pytest.approx(11_357_000.0, rel=0.15)
        alpha = 11_357_000.0 / (11_357_000.0 - 5_000_000.0)
        assert np.median(draws) == pytest.approx(
            5_000_000.0 * 2.0 ** (1.0 / alpha), rel=5e-3
        )

    def test_rejects_a_mean_outside_the_band(self) -> None:
        with pytest.raises(ValueError, match="does not sit inside"):
            _truncated_exponential_quantile(np.asarray([0.5]), 0.0, 10.0, 20.0)
        with pytest.raises(ValueError, match="must exceed"):
            _pareto_quantile(np.asarray([0.5]), 5_000_000.0, 4_000_000.0)


class TestImputation:
    def test_reaches_the_top_band_and_is_deterministic(self) -> None:
        distribution = _distribution()
        rows = 2_000
        rng = np.random.default_rng(1)
        gains = np.where(rng.random(rows) < 0.5, rng.lognormal(10, 1, rows), 0.0)
        incomes = rng.lognormal(10.5, 0.8, rows)
        frame = _frame(rows, gains=gains, incomes=incomes)

        first = impute_uk_capital_gains(frame, distribution, PARAMETERS)
        second = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = first.table("person")["capital_gains"].to_numpy()
        assert (drawn == second.table("person")["capital_gains"].to_numpy()).all()
        assert drawn.max() >= 5_000_000.0, "no draw reached the open top band"

    def test_non_gainers_stay_at_zero_and_weights_pass_through(self) -> None:
        distribution = _distribution()
        frame = _frame(
            6,
            gains=[0.0, 0.0, 50_000.0, 20_000.0, 10_000.0, 5_000.0],
            incomes=[10_000.0] * 6,
        )

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert (drawn[:2] == 0.0).all()
        assert (drawn[2:] > 0.0).all()
        pd.testing.assert_frame_equal(
            result.table("household"), frame.table("household")
        )
        # The appended record is a conservation receipt for the terminal
        # family gate, not a mass change.
        assert result.mass_log[:-1] == frame.mass_log
        receipt = result.mass_log[-1]
        assert receipt.entity == "household"
        assert receipt.reason == UK_CGT_MASS_CONSERVATION_REASON
        assert receipt.old_total == receipt.new_total
        assert receipt.declared_factor == 1.0

    def test_loss_makers_pass_through_byte_identical(self) -> None:
        # The certified candidate carries net losses: 49,640 of 1,157,100
        # person rows hold negative capital_gains (min -£72,374). The first
        # gated build failed on a blanket non-negativity guard over the whole
        # column; losses are legitimate pass-through content the published
        # surface says nothing about.
        distribution = _distribution()
        losses = [-72_374.0, -1_000.0, -0.5]
        frame = _frame(
            6,
            gains=[*losses, 0.0, 50_000.0, 20_000.0],
            incomes=[10_000.0] * 6,
        )

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert drawn[:3] == pytest.approx(losses)
        assert drawn[3] == 0.0
        assert (drawn[4:] > 0.0).all()
        receipt = result.mass_log[-1]
        assert receipt.reason == UK_CGT_MASS_CONSERVATION_REASON

    def test_remainder_keeps_existing_amounts_capped_at_the_aea(self) -> None:
        # One income band holds far more gainer mass than the published
        # taxpayers, so the ranking's tail lands in the sub-AEA remainder.
        distribution = _distribution(cell_people=10.0)
        rows = 3_000
        rng = np.random.default_rng(2)
        gains = rng.lognormal(9, 1.5, rows)
        frame = _frame(rows, gains=gains, incomes=np.full(rows, 20_000.0))

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        remainder = drawn <= PARAMETERS.annual_exempt_amount
        assert remainder.any()
        expected = np.minimum(gains, PARAMETERS.annual_exempt_amount)
        assert drawn[remainder] == pytest.approx(expected[remainder])

    def test_allocation_preserves_the_existing_ranking(self) -> None:
        # Everyone sits in one income band, whose cells hold 100 people each
        # against person weights of 100 — so the £5m+ band holds exactly one
        # person of mass, and it must be the largest existing gainer.
        distribution = _distribution(cell_people=100.0)
        rows = 4_000
        rng = np.random.default_rng(3)
        gains = rng.lognormal(9, 2, rows)
        frame = _frame(rows, gains=gains, incomes=np.full(rows, 20_000.0))

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)
        drawn = result.table("person")["capital_gains"].to_numpy()

        ranked = np.argsort(-gains)
        assert drawn[ranked[0]] >= 5_000_000.0
        # Walking down the existing ranking, assigned gain bands never rise.
        taxpayer_mass = int(
            sum(100.0 for _ in HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
        )  # people of mass in the band, at weight 100 each -> 10 persons
        assigned = drawn[ranked[: taxpayer_mass // 100]]
        bands = np.digitize(assigned, HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
        assert (np.diff(bands) <= 0).all()

    def test_suppressed_cell_falls_back_to_the_band_mean(self) -> None:
        distribution = _distribution(suppress_top_low_income=True)
        rows = 4_000
        rng = np.random.default_rng(4)
        gains = rng.lognormal(9, 2, rows)
        frame = _frame(rows, gains=gains, incomes=np.full(rows, 20_000.0))

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert drawn.max() >= 5_000_000.0

    def test_summary_reports_achieved_against_published(self) -> None:
        distribution = _distribution()
        rows = 2_000
        rng = np.random.default_rng(5)
        gains = np.where(rng.random(rows) < 0.5, rng.lognormal(10, 1, rows), 0.0)
        frame = _frame(rows, gains=gains, incomes=np.full(rows, 20_000.0))

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)
        summary = summarize_uk_cgt_imputation(frame, result, distribution, PARAMETERS)

        conditioning = load_hmrc_cgt_conditioning_facts()
        assert len(summary.rows) == len(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
        # Levels are the 2024-25 individual observations, not the joint's
        # own 2023-24 total.
        assert summary.published_taxpayer_mass == (
            conditioning.table1.individuals_taxpayers
        )
        assert summary.taxpayer_mass > 0
        assert (summary.rows["published_gains"] > 0).all()
        assert summary.rows["published_people"].sum() == pytest.approx(
            conditioning.table1.individuals_taxpayers
        )
        assert len(summary.age_rows) == len(conditioning.age_bands)
        assert len(summary.region_rows) == len(conditioning.regions)
        assert len(summary.age_by_band_rows) == len(conditioning.age_bands) * len(
            HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
        )
        evidence = summary.evidence()
        assert {"rows", "age_rows", "region_rows", "age_by_band_rows"} <= set(evidence)


class TestStage:
    def test_stage_runs_end_to_end_on_the_vendored_surface(self) -> None:
        """The factory's own transform path, on the committed resource.

        Regression test for the transform keeping a retired carrier type in
        its signature: with postponed annotation evaluation, only running the
        stage exercises the closure.
        """
        stage = uk_capital_gains_imputation_stage(parameters=PARAMETERS)
        incomes = [20_000.0, 55_000.0, 80_000.0, 120_000.0, 180_000.0, 400_000.0]
        frame = _frame(
            60,
            gains=[float(5_000 * (i + 1)) for i in range(60)],
            incomes=[incomes[i % 6] for i in range(60)],
        )

        result = stage.run(frame)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert (drawn >= 0).all()
        assert drawn.max() > 0

    def test_stage_carries_the_reviewed_name(self) -> None:
        stage = uk_capital_gains_imputation_stage(parameters=PARAMETERS)

        assert stage.name == UK_CGT_IMPUTATION_STAGE_NAME

    def test_stage_checks_the_feed_pin_before_reading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import copy

        from microcosm.build.uk_runtime import hmrc_capital_gains
        from microcosm.build.uk_runtime.ledger_fact_vendoring import (
            load_vendored_resource,
        )

        stale = copy.deepcopy(
            load_vendored_resource("hmrc_cgt_conditioning_facts.json")
        )
        stale["source_fact_feed"]["facts_sha256"] = "0" * 64
        monkeypatch.setattr(
            hmrc_capital_gains, "load_vendored_resource", lambda _name: stale
        )
        stage = uk_capital_gains_imputation_stage(parameters=PARAMETERS)
        frame = _frame(1, gains=[10_000.0], incomes=[20_000.0])

        with pytest.raises(ValueError, match="differs from the committed UK pin"):
            stage.run(frame)


def test_cgt_spine_parsed_inputs_match_the_resource_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    stage = spec.sources.stage_map()["hmrc_cgt_gains_spine"]
    distribution = _distribution(cell_people=10.0)
    frame = _frame(
        6,
        gains=[5_000.0, 20_000.0, 75_000.0, 300_000.0, 2_500_000.0, 0.0],
        incomes=[20_000.0, 40_000.0, 60_000.0, 120_000.0, 250_000.0, 20_000.0],
    )
    resolved: list[str] = []

    def load_distribution():
        resolved.append("distribution")
        return distribution

    def load_parameters(period):
        assert period == "2023"
        resolved.append("parameters")
        return PARAMETERS

    monkeypatch.setattr(
        cgt_imputation, "load_hmrc_cgt_joint_distribution", load_distribution
    )
    monkeypatch.setattr(cgt_imputation, "uk_cgt_policy_parameters", load_parameters)
    resource_transform = uk_cgt_spine_stage_transform(stage)
    from_resource = resource_transform(frame)
    assert resolved == ["distribution", "parameters"]

    def unexpected_loader(*_args, **_kwargs):
        raise AssertionError("parsed inputs must bypass source resolution")

    monkeypatch.setattr(
        cgt_imputation, "load_hmrc_cgt_joint_distribution", unexpected_loader
    )
    monkeypatch.setattr(
        cgt_imputation,
        "uk_cgt_policy_parameters",
        unexpected_loader,
    )
    seam_transform = uk_cgt_spine_stage_transform(
        stage,
        distribution=distribution,
        parameters=PARAMETERS,
    )
    from_seam = seam_transform(frame)

    assert uk_frame_content_identity(from_resource) == uk_frame_content_identity(
        from_seam
    )
    assert (
        resource_transform.checkpoint_metadata() == seam_transform.checkpoint_metadata()
    )


def _real_distribution() -> HMRCCapitalGainsJointDistribution:
    """The published 2024-25 joint, from the committed vendored rows.

    The rows are copied verbatim from the pinned Chronicle feed, so CI
    exercises the real surface without hand-copied values.
    """
    return load_hmrc_cgt_joint_distribution()


class TestRealPublishedSurface:
    """The published 2024-25 values, vendored so CI cannot mask them.

    A synthetic distribution with feasible means hid that real cells imply
    a mean outside their band through rounding, crashing the stage on any
    population reaching those income bands.
    """

    def test_every_income_band_plans_without_raising(self) -> None:
        distribution = _real_distribution()

        for income in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS:
            plans = _band_plans(
                distribution,
                income,
                annual_exempt_amount=PARAMETERS.annual_exempt_amount,
            )
            for plan in plans:
                if np.isinf(plan.gain_upper_bound):
                    assert plan.mean > plan.effective_lower_bound
                else:
                    assert (
                        plan.effective_lower_bound < plan.mean < plan.gain_upper_bound
                    )

    def test_infeasible_rounded_means_are_repaired_toward_the_boundary(self) -> None:
        distribution = _real_distribution()

        plans = _band_plans(
            distribution, 37700, annual_exempt_amount=PARAMETERS.annual_exempt_amount
        )
        by_band = {plan.gain_lower_bound: plan for plan in plans}

        # 1,000 people (rounded) holding £980m in the £1m-£2m band imply a
        # mean of £980,000, below the band; it clamps just inside.
        repaired = by_band[1_000_000]
        assert repaired.mean_repaired
        assert 1_000_000 < repaired.mean <= 1_020_000
        by_band_100k = {
            plan.gain_lower_bound: plan
            for plan in _band_plans(
                distribution,
                100_000,
                annual_exempt_amount=PARAMETERS.annual_exempt_amount,
            )
        }
        assert by_band_100k[1_000_000].mean_repaired
        assert not by_band[500_000].mean_repaired

    def test_imputation_runs_across_every_income_band(self) -> None:
        distribution = _real_distribution()
        rows = 3_000
        rng = np.random.default_rng(7)
        gains = np.where(rng.random(rows) < 0.6, rng.lognormal(10, 1.5, rows), 0.0)
        incomes = rng.choice(
            [20_000.0, 55_000.0, 80_000.0, 120_000.0, 180_000.0, 400_000.0], rows
        )
        frame = _frame(rows, gains=gains, incomes=incomes)

        proxy = uk_cgt_taxable_income_proxy(frame.table("person"), PARAMETERS)
        visited = set(
            np.asarray(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)[
                np.digitize(proxy, HMRC_CGT_INCOME_BAND_LOWER_BOUNDS[1:])
            ]
        )
        assert visited == set(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS), (
            "fixture incomes must reach every published income band after "
            f"the Personal Allowance; missing {set(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS) - visited}"
        )

        result = impute_uk_capital_gains(frame, distribution, PARAMETERS)

        drawn = result.table("person")["capital_gains"].to_numpy()
        assert np.isfinite(drawn).all()
        assert (drawn >= 0).all()
        redrawn = drawn[gains > 0]
        liable = redrawn[redrawn > PARAMETERS.annual_exempt_amount]
        # Every draw allocated to the liability distribution clears the AEA.
        assert liable.size > 0
        below = redrawn[(redrawn > 0) & (redrawn <= PARAMETERS.annual_exempt_amount)]
        # The remainder keeps capped existing amounts, never band draws.
        assert below.size == 0 or below.max() <= PARAMETERS.annual_exempt_amount
        assert drawn.max() >= 5_000_000.0


class TestPolicyParameters:
    @pytest.mark.requires_uk
    def test_reads_the_2023_values_from_the_parameter_tree(self) -> None:
        from microcosm.build.uk_runtime.cgt_imputation import uk_cgt_policy_parameters

        parameters = uk_cgt_policy_parameters(2023)

        assert parameters.personal_allowance == 12_570.0
        assert parameters.personal_allowance_taper_threshold == 100_000.0
        assert parameters.personal_allowance_taper_rate == 0.5
        assert parameters.annual_exempt_amount == 6_000.0
        assert parameters.instant == "2023-06-01"


class TestConditionedAllocation:
    """Age and region conditioning on the 2024-25 band vintage (microcosm#725)."""

    def test_joint_reconciles_with_the_table_2_1a_bands(self) -> None:
        conditioning = load_hmrc_cgt_conditioning_facts()
        distribution = _real_distribution()
        joint, band_rows = _joint_plans(
            distribution,
            conditioning,
            annual_exempt_amount=3_000.0,
        )
        folded = conditioning.size_bands_aggregated(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
        for row in band_rows:
            lower = row["gain_lower_bound"]
            assert row["published_people"] == distribution.band_total(lower).individuals
            assert row["table2_1a_people"] == folded[lower][0]
            assert row["people_difference"] == 0.0
            assert abs(row["gains_difference"]) <= 1_000_000.0
            # Column reconciliation keeps each band near its published row.
            assert row["target_people"] == pytest.approx(
                row["published_people"], rel=0.15
            )
            for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS:
                plan = joint[(lower, income_lower)]
                assert plan.effective_lower_bound < plan.mean
                assert plan.mean < plan.gain_upper_bound
        assert sum(row["means_repaired"] for row in band_rows) == 2
        # The joint's mass is the published taxpayer count, column by column.
        assert sum(row["target_people"] for row in band_rows) == pytest.approx(
            conditioning.table1.individuals_taxpayers, rel=1e-9
        )

    def test_rake_meets_feasible_margins(self) -> None:
        conditioning = load_hmrc_cgt_conditioning_facts()
        joint, _ = _joint_plans(
            _real_distribution(), conditioning, annual_exempt_amount=3_000.0
        )
        rows = 6 * len(UK_CGT_AGE_GROUP_LOWER_BOUNDS) * len(UK_CGT_REGION_GROUP_LABELS)
        rng = np.random.default_rng(7)
        # Every (income band, age group, region group) cell has gainers, so
        # the seed is strictly positive and every margin is attainable.
        income_band = np.repeat(
            np.asarray(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS), rows // 6
        )
        age_group = np.tile(
            np.repeat(
                np.arange(len(UK_CGT_AGE_GROUP_LOWER_BOUNDS)),
                len(UK_CGT_REGION_GROUP_LABELS),
            ),
            6,
        )
        region_group = np.tile(
            np.arange(len(UK_CGT_REGION_GROUP_LABELS)),
            rows // len(UK_CGT_REGION_GROUP_LABELS),
        )
        targets, report = _rake_allocation_targets(
            joint,
            conditioning,
            is_gainer=np.ones(rows, dtype=bool),
            person_weight=rng.uniform(50.0, 150.0, rows),
            income_band=income_band,
            age_group=age_group,
            region_group=region_group,
        )
        assert report["ipf_max_abs_margin_error"] < 1e-6
        assert report["ipf_zero_seed_cells"] == 0
        assert targets.sum() == pytest.approx(report["joint_total_people"])
        raked_by_age = targets.sum(axis=(0, 1, 3))
        for index, lower in enumerate(UK_CGT_AGE_GROUP_LOWER_BOUNDS):
            assert raked_by_age[index] == pytest.approx(
                report["age_margin"][str(lower)]["normalised"], rel=1e-6
            )

    def test_rake_reports_an_income_band_without_gainers(self) -> None:
        conditioning = load_hmrc_cgt_conditioning_facts()
        joint, _ = _joint_plans(
            _real_distribution(), conditioning, annual_exempt_amount=3_000.0
        )
        rows = 40
        targets, report = _rake_allocation_targets(
            joint,
            conditioning,
            is_gainer=np.ones(rows, dtype=bool),
            person_weight=np.full(rows, 100.0),
            income_band=np.full(rows, 0),
            age_group=np.arange(rows) % len(UK_CGT_AGE_GROUP_LOWER_BOUNDS),
            region_group=np.arange(rows) % len(UK_CGT_REGION_GROUP_LABELS),
        )
        assert report["income_bands_without_gainers"] == list(
            HMRC_CGT_INCOME_BAND_LOWER_BOUNDS[1:]
        )
        assert report["ipf_zero_seed_cells"] > 0
        # Nothing is invented for the empty bands.
        assert targets[:, 1:, :, :].sum() == 0.0

    def test_age_conditioning_moves_taxpayer_mass_toward_table_6(self) -> None:
        # Six ages x twelve regions x six income bands, every combination
        # present with the same prior, so the margins are attainable and any
        # age shape in the result comes from the rake, not from the priors.
        # 24 persons per (age, region, income) combination at weight 500:
        # every cell holds more support than its raked target, and one
        # person is small next to the smallest age group (32,000).
        rows = 6 * 12 * 6 * 24
        rng = np.random.default_rng(11)
        gains = rng.lognormal(10, 1, rows)
        incomes = np.asarray(
            # After the personal allowance these land in the six Table 3
            # taxable-income bands (45,000 would share band 0 with 20,000).
            [20_000.0, 55_000.0, 75_000.0, 120_000.0, 180_000.0, 300_000.0]
        )
        index = np.arange(rows)
        ages = np.asarray([20, 40, 50, 60, 70, 80])[index % 6]
        regions = np.asarray(sorted(UK_CGT_REGION_GROUPS))[(index // 6) % 12]
        frame = _frame(
            rows,
            gains=gains,
            incomes=incomes[(index // 72) % 6],
            ages=ages,
            regions=regions,
            # Enough support in every cell that no cell scales down and
            # the pooled walk is never needed: the achieved shape is the rake's.
            weights=np.full(rows, 500.0),
        )
        conditioning = load_hmrc_cgt_conditioning_facts()

        result, report = impute_uk_capital_gains_with_report(
            frame, _real_distribution(), PARAMETERS, conditioning=conditioning
        )

        assert report.rake["ipf_max_abs_margin_error"] < 1e-3
        assert report.rake["ipf_zero_seed_cells"] == 0
        drawn = result.table("person")["capital_gains"].to_numpy()
        liable = drawn > PARAMETERS.annual_exempt_amount
        weight = 500.0
        share = {
            lower: float(liable[ages_group == index].sum() * weight)
            for index, lower in enumerate(UK_CGT_AGE_GROUP_LOWER_BOUNDS)
            for ages_group in [np.digitize(ages, UK_CGT_AGE_GROUP_LOWER_BOUNDS[1:])]
        }
        total = sum(share.values())
        published = {
            band.lower_bound: band.taxpayers for band in conditioning.age_bands
        }
        # Uniform support would give each group a sixth; Table 6 gives
        # 16-34 under 6 % of taxpayers and 55-64 over a quarter.
        assert share[16] / total < 0.10
        assert share[55] / total > 0.20
        older = (share[65] + share[75]) / total
        published_older = (
            published[65] + published[75] + published[85]
        ) / conditioning.table1.individuals_taxpayers
        assert abs(older - published_older) < 0.10
        # The tail follows the same shape: no longer a working-age monopoly
        # (the #725 finding was 2 % at 65+ above GBP 2m). Read at GBP 500k
        # and above so the check rests on dozens of persons, not a handful.
        tail = drawn >= 500_000.0
        assert tail.sum() >= 30
        assert float((tail & (ages >= 65)).sum() / tail.sum()) > 0.2

    def test_fallback_meets_the_joint_where_a_cell_has_no_support(self) -> None:
        rows = 3_000
        rng = np.random.default_rng(13)
        gains = rng.lognormal(10, 1, rows)
        # Everyone is a 45-year-old Londoner: five age groups and three
        # region groups have no support, so pass 1 can only fill one cell
        # per income band and the pooled walk must carry the rest. Weights
        # of 100 keep one person smaller than the top band's low-income
        # target (a suppressed cell of roughly 130 people).
        frame = _frame(
            rows,
            gains=gains,
            incomes=np.full(rows, 20_000.0),
            weights=np.full(rows, 100.0),
        )

        result, report = impute_uk_capital_gains_with_report(
            frame,
            _real_distribution(),
            PARAMETERS,
            conditioning=load_hmrc_cgt_conditioning_facts(),
        )

        assert report.rake["ipf_zero_seed_cells"] > 0
        assert report.fallback_released_mass > 0.0
        joint_rows = pd.DataFrame(report.joint_rows)
        low_income = joint_rows[joint_rows["income_lower_bound"] == 0]
        achieved = low_income["achieved_pass1"] + low_income["achieved_fallback"]
        # The band's pooled support (300,000 people) exceeds the low-income
        # column's 2024-25 target (292,000), so the joint is met in full
        # despite the empty conditioning cells; a person never splits, so
        # allow one weight of slack per band.
        assert (achieved + 100.0 >= low_income["target_people"]).all()
        assert abs(achieved.sum() - low_income["target_people"].sum()) < 100.0 * len(
            low_income
        )
        drawn = result.table("person")["capital_gains"].to_numpy()
        assert drawn.max() >= 5_000_000.0

    def test_unknown_region_and_missing_age_refuse(self) -> None:
        frame = _frame(
            4,
            gains=[1_000.0, 2_000.0, 3_000.0, 4_000.0],
            incomes=[20_000.0] * 4,
            regions=["LONDON", "LONDON", "MARS", "LONDON"],
        )
        with pytest.raises(ValueError, match="Unknown region name"):
            impute_uk_capital_gains(frame, _distribution(), PARAMETERS)

        person = frame.table("person").drop(columns=["age"])
        stripped = uk_national_frame(
            person=person,
            benunit=frame.table("benunit"),
            household=frame.table("household"),
            time_period="2023",
            household_weights=frame.weights_for("household").values,
        )
        with pytest.raises(ValueError, match="no age column"):
            impute_uk_capital_gains(stripped, _distribution(), PARAMETERS)

    def test_region_groups_cover_the_tier_once(self) -> None:
        assert set(UK_CGT_REGION_GROUPS.values()) == set(UK_CGT_REGION_GROUP_LABELS)
        assert UK_CGT_REGION_GROUPS["LONDON"] == "london"
        assert (
            UK_CGT_REGION_GROUPS["SOUTH_EAST"]
            == UK_CGT_REGION_GROUPS["EAST_OF_ENGLAND"]
        )

    def test_report_records_the_conditioning_resource(self) -> None:
        conditioning = load_hmrc_cgt_conditioning_facts()
        frame = _frame(
            12,
            gains=[float(10_000 * (i + 1)) for i in range(12)],
            incomes=[20_000.0] * 12,
        )
        _, report = impute_uk_capital_gains_with_report(
            frame, _distribution(), PARAMETERS, conditioning=conditioning
        )
        assert report.conditioning["resource_sha256"] == conditioning.resource_sha256
        assert report.conditioning["vintage_tax_year"] == 2024
        assert report.conditioning["joint_resource"] == "synthetic.json"
        assert report.conditioning["joint_vintage"] == "2024-25"
        assert report.conditioning["fallback_policy"] == "pooled_income_band_rank_walk"
        evidence = report.evidence()
        assert set(evidence) == {
            "band_rows",
            "joint_rows",
            "rake",
            "fallback_released_mass",
            "fallback_share_by_band",
            "conditioning",
        }
