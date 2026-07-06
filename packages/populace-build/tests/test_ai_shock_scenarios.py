from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime.ai_shock_scenarios import (
    AGE_BANDS,
    COMPLEMENTARITY_COLUMN,
    DISPLACED_COLUMN,
    PRE_SHOCK_EMPLOYMENT_INCOME_COLUMN,
    PRESETS,
    ShockScenario,
    apply_capital_shock,
    apply_employment_shock,
    apply_wage_shock,
    run_scenario,
)

N_PERSONS = 12_000


def synthetic_persons(seed: int = 7, n: int = N_PERSONS) -> pd.DataFrame:
    """Person table with age, incomes, exposure/complementarity and weights."""

    rng = np.random.default_rng(seed)
    age = rng.integers(16, 86, size=n)
    employed = rng.uniform(size=n) < np.where(age < 65, 0.75, 0.1)
    occupation_group = rng.choice(
        ["managers", "clerical", "trades", "care", "elementary"], size=n
    )
    exposure_by_group = {
        "managers": 0.9,
        "clerical": 1.4,
        "trades": 0.2,
        "care": 0.4,
        "elementary": 0.7,
    }
    exposure = np.array([exposure_by_group[g] for g in occupation_group])
    exposure = exposure + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame(
        {
            "age": age,
            "employment_income": np.where(
                employed, rng.lognormal(10.0, 0.5, size=n), 0.0
            ),
            "savings_interest_income": rng.exponential(120.0, size=n),
            "dividend_income": rng.exponential(300.0, size=n),
            "property_income": rng.exponential(500.0, size=n),
            "ai_exposure": exposure,
            "ai_complementarity": rng.uniform(0.1, 1.0, size=n),
            "occupation_group": occupation_group,
            "person_weight": rng.uniform(0.5, 3.0, size=n),
        }
    )


def scenario(**overrides) -> ShockScenario:
    parameters = {
        "name": "test",
        "displacement_rate": 0.07,
        "wage_uplift": 0.026,
        "n_draws": 10,
        "seed": 11,
    }
    parameters.update(overrides)
    return ShockScenario(**parameters)


def employed_mask(persons: pd.DataFrame) -> pd.Series:
    return (persons["employment_income"] > 0) & (persons["age"] >= 16)


class TestEmploymentShock:
    def test_aggregate_weighted_displacement_matches_rate(self):
        persons = synthetic_persons()
        shocked, summary = apply_employment_shock(persons, scenario())

        employed = employed_mask(persons)
        employed_weight = persons.loc[employed, "person_weight"].sum()
        realised = (
            shocked.loc[shocked[DISPLACED_COLUMN], "person_weight"].sum()
            / employed_weight
        )
        # The realised draw honours the per-group weighted quotas up to the
        # granularity of one person weight per group.
        assert realised == pytest.approx(0.07, abs=0.005)
        # The draw-averaged summary matches at least as tightly.
        assert summary["displaced_weighted_share"] == pytest.approx(0.07, abs=0.005)

    def test_eq_3_4_group_totals_honoured(self):
        persons = synthetic_persons()
        shocked, summary = apply_employment_shock(persons, scenario())

        employed = employed_mask(persons)
        weights = persons["person_weight"]
        total_job_loss = 0.07 * weights[employed].sum()
        scores = {}
        for label, block in persons[employed].groupby("occupation_group"):
            emp = block["person_weight"].sum()
            mean_exposure = (block["ai_exposure"] * block["person_weight"]).sum() / emp
            scores[label] = emp * mean_exposure
        total_score = sum(scores.values())

        max_weight = weights.max()
        for label, score in scores.items():
            expected_quota = total_job_loss * score / total_score
            group = summary["by_occupation_group"][label]
            assert group["job_loss_quota_weighted"] == pytest.approx(expected_quota)
            # Every draw meets its quota up to one boundary person's weight,
            # so the realised table and the draw average both honour eq 3.4.
            assert group["displaced_weighted"] == pytest.approx(
                expected_quota, abs=max_weight
            )
            realised = shocked.loc[
                shocked[DISPLACED_COLUMN] & (shocked["occupation_group"] == label),
                "person_weight",
            ].sum()
            assert realised == pytest.approx(expected_quota, abs=max_weight)

    def test_displacement_concentrates_in_high_exposure_group(self):
        persons = synthetic_persons()
        _, summary = apply_employment_shock(persons, scenario())
        by_group = summary["by_occupation_group"]
        # clerical has the highest exposure, trades the lowest.
        assert (
            by_group["clerical"]["displaced_weighted_share"]
            > by_group["trades"]["displaced_weighted_share"]
        )

    def test_multi_draw_averaging_converges(self):
        persons = synthetic_persons()
        single = scenario(n_draws=1)
        many = scenario(n_draws=50)
        _, single_summary = apply_employment_shock(persons, single)
        _, many_summary = apply_employment_shock(persons, many)

        # Averaging over draws must not move the aggregate away from the
        # quota-implied share, and per-band averages stabilise: the
        # youngest band's draw-averaged share over 50 draws sits within the
        # spread of individual draws.
        target = single_summary["displaced_weighted_share"]
        assert many_summary["displaced_weighted_share"] == pytest.approx(
            target, rel=0.02
        )
        band_shares = []
        for draw in range(20):
            _, draw_summary = apply_employment_shock(
                persons, scenario(n_draws=1, seed=100 + draw)
            )
            band_shares.append(
                draw_summary["by_age_band"]["16-24"]["displaced_weighted_share"]
            )
        averaged = many_summary["by_age_band"]["16-24"]["displaced_weighted_share"]
        assert min(band_shares) <= averaged <= max(band_shares)

    def test_youth_multiplier_shifts_displacement_to_16_24(self):
        persons = synthetic_persons()
        _, neutral = apply_employment_shock(persons, scenario())
        _, tilted = apply_employment_shock(
            persons, scenario(youth_displacement_multiplier=3.0)
        )

        assert (
            tilted["by_age_band"]["16-24"]["displaced_weighted_share"]
            > neutral["by_age_band"]["16-24"]["displaced_weighted_share"]
        )
        # The group-level eq 3.4 totals are preserved under the tilt.
        for label, group in tilted["by_occupation_group"].items():
            assert group["displaced_weighted"] == pytest.approx(
                neutral["by_occupation_group"][label]["job_loss_quota_weighted"],
                abs=persons["person_weight"].max(),
            )
        # And the aggregate stays at the scenario rate.
        assert tilted["displaced_weighted_share"] == pytest.approx(0.07, abs=0.005)

    def test_displaced_income_zeroed_and_flagged(self):
        persons = synthetic_persons()
        shocked, summary = apply_employment_shock(persons, scenario())

        displaced = shocked[DISPLACED_COLUMN]
        assert displaced.any()
        assert (shocked.loc[displaced, "employment_income"] == 0.0).all()
        assert (shocked.loc[displaced, PRE_SHOCK_EMPLOYMENT_INCOME_COLUMN] > 0.0).all()
        # Non-displaced incomes are untouched by the employment shock.
        untouched = ~displaced
        pd.testing.assert_series_equal(
            shocked.loc[untouched, "employment_income"],
            persons.loc[untouched, "employment_income"],
        )
        assert summary["returned_draw"] == 0

    def test_fallback_exposure_groups_without_occupation_column(self):
        persons = synthetic_persons().drop(columns=["occupation_group"])
        _, summary = apply_employment_shock(persons, scenario())
        assert set(summary["by_occupation_group"]) <= {
            f"exposure_q{i}" for i in range(1, 6)
        }
        assert summary["displaced_weighted_share"] == pytest.approx(0.07, abs=0.005)
        # Higher exposure quintiles bear a larger displaced share.
        by_group = summary["by_occupation_group"]
        assert (
            by_group["exposure_q5"]["displaced_weighted_share"]
            > by_group["exposure_q1"]["displaced_weighted_share"]
        )


class TestWageShock:
    def test_distributed_by_complementarity_with_target_mean(self):
        persons = synthetic_persons()
        shocked, summary = apply_wage_shock(persons, scenario())

        recipients = employed_mask(persons)
        uplift = (
            shocked.loc[recipients, "employment_income"]
            / persons.loc[recipients, "employment_income"]
            - 1.0
        )
        theta = persons.loc[recipients, COMPLEMENTARITY_COLUMN]
        weights = persons.loc[recipients, "person_weight"]
        mean_theta = (theta * weights).sum() / weights.sum()

        # Eq 3.5: person uplift proportional to theta, employment-weighted
        # mean equal to the aggregate wage change.
        np.testing.assert_allclose(uplift, 0.026 * theta / mean_theta)
        weighted_mean = (uplift * weights).sum() / weights.sum()
        assert weighted_mean == pytest.approx(0.026)
        assert summary["weighted_mean_uplift"] == pytest.approx(0.026)
        assert not summary["uniform_fallback"]

    def test_uniform_fallback_without_complementarity_column(self):
        persons = synthetic_persons().drop(columns=[COMPLEMENTARITY_COLUMN])
        with pytest.warns(UserWarning, match="uniform wage uplift"):
            shocked, summary = apply_wage_shock(persons, scenario())

        recipients = employed_mask(persons)
        uplift = (
            shocked.loc[recipients, "employment_income"]
            / persons.loc[recipients, "employment_income"]
            - 1.0
        )
        np.testing.assert_allclose(uplift, 0.026)
        assert summary["uniform_fallback"]

    def test_wage_shock_leaves_displaced_at_zero(self):
        persons = synthetic_persons()
        shocked, _ = apply_employment_shock(persons, scenario())
        reshocked, _ = apply_wage_shock(shocked, scenario())

        displaced = reshocked[DISPLACED_COLUMN]
        assert displaced.any()
        assert (reshocked.loc[displaced, "employment_income"] == 0.0).all()


class TestCapitalShock:
    def test_scales_interest_and_dividends_by_return_ratio(self):
        persons = synthetic_persons()
        shocked, summary = apply_capital_shock(persons, scenario())

        # ESRI anchor: 0.004 / 0.01005 ~= +39.8% capital income.
        uplift = 0.004 / 0.01005
        assert uplift == pytest.approx(0.398, abs=0.001)
        np.testing.assert_allclose(
            shocked["savings_interest_income"],
            persons["savings_interest_income"] * (1.0 + uplift),
        )
        np.testing.assert_allclose(
            shocked["dividend_income"],
            persons["dividend_income"] * (1.0 + uplift),
        )
        assert summary["capital_uplift"] == pytest.approx(uplift)
        expected_change = (
            persons["person_weight"]
            * (persons["savings_interest_income"] + persons["dividend_income"])
            * uplift
        ).sum()
        assert summary["weighted_capital_income_change"] == pytest.approx(
            expected_change
        )

    def test_rent_excluded_by_default(self):
        persons = synthetic_persons()
        shocked, summary = apply_capital_shock(persons, scenario())
        pd.testing.assert_series_equal(
            shocked["property_income"], persons["property_income"]
        )
        assert "property_income" not in summary["capital_columns"]

    def test_errors_without_any_capital_column(self):
        persons = synthetic_persons().drop(
            columns=["savings_interest_income", "dividend_income"]
        )
        with pytest.raises(ValueError, match="capital income column"):
            apply_capital_shock(persons, scenario())


class TestRunScenario:
    def test_same_seed_identical_results(self):
        persons = synthetic_persons()
        first, first_summary = run_scenario(persons, scenario())
        second, second_summary = run_scenario(persons, scenario())

        pd.testing.assert_frame_equal(first, second)
        assert first_summary == second_summary

    def test_different_seed_differs(self):
        persons = synthetic_persons()
        first, _ = run_scenario(persons, scenario(seed=1))
        second, _ = run_scenario(persons, scenario(seed=2))
        assert not first[DISPLACED_COLUMN].equals(second[DISPLACED_COLUMN])

    def test_preset_names_and_anchors(self):
        assert set(PRESETS) == {"central", "low", "high"}
        central = PRESETS["central"]
        assert central.displacement_rate == pytest.approx(0.07)
        assert central.wage_uplift == pytest.approx(0.026)
        assert central.capital_uplift == pytest.approx(0.398, abs=0.001)
        assert central.youth_displacement_multiplier == 1.0
        assert central.n_draws == 50

    def test_composes_all_three_shocks(self):
        persons = synthetic_persons()
        shocked, summary = run_scenario(persons, scenario(), seed=99, draw_index=2)

        assert summary["parameters"]["seed"] == 99
        assert summary["employment"]["returned_draw"] == 2
        assert {"employment", "wage", "capital"} <= set(summary)
        displaced = shocked[DISPLACED_COLUMN]
        assert displaced.any()
        assert (shocked.loc[displaced, "employment_income"] == 0.0).all()
        survivors = employed_mask(persons) & ~displaced
        assert (
            shocked.loc[survivors, "employment_income"]
            > persons.loc[survivors, "employment_income"]
        ).all()
        assert (shocked["dividend_income"] > persons["dividend_income"]).all()
        # Age-band summaries cover every band.
        expected_labels = {
            f"{lower}-{upper - 1}" if upper else f"{lower}+"
            for lower, upper in AGE_BANDS
        }
        for shock in ("employment", "wage", "capital"):
            assert set(summary[shock]["by_age_band"]) == expected_labels

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario preset"):
            run_scenario(synthetic_persons(), "sideways")

    def test_input_table_not_mutated(self):
        persons = synthetic_persons()
        original = persons.copy()
        run_scenario(persons, scenario())
        pd.testing.assert_frame_equal(persons, original)


class TestFrameInput:
    def test_frame_person_table_is_extracted_with_weights(self):
        pytest.importorskip("populace.frame")
        from populace.frame import (
            EntitySchema,
            Frame,
            WeightKind,
            Weights,
        )

        persons = synthetic_persons(n=400).drop(columns=["person_weight"])
        persons["person_id"] = np.arange(len(persons))
        persons["person_household_id"] = np.arange(len(persons))
        households = pd.DataFrame({"household_id": np.arange(len(persons))})
        schema = EntitySchema(group_entities=("household",))
        weights = Weights(np.full(len(persons), 1.5), WeightKind.CALIBRATED)
        frame = Frame(
            tables={"person": persons, "household": households},
            schema=schema,
            weights={"household": weights},
        )

        shocked, summary = apply_employment_shock(frame, scenario())
        assert isinstance(shocked, pd.DataFrame)
        assert "person_weight" in shocked.columns
        assert summary["displaced_weighted_share"] == pytest.approx(0.07, abs=0.01)
