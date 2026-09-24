"""Behavioral contracts for static aging.

Weights carry demographics only: each projected year's weights reproduce that
year's population by cell and nothing else. Dollar columns follow their series
through factors: a national total is hit exactly under the year's weights, an
index applies as a ratio, an unmapped column carries over. Program counts are
predictions the projected frame makes and are never fed back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.calibrate import (
    DemographicProjection,
    SeriesProjection,
    score_predictions,
    ssa_population_projection,
    static_aging,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

BASE_YEAR = 2024
SCHEMA = EntitySchema(group_entities=("household",))
AGE_TOP = 85


def _frame(n_households: int = 240, seed: int = 7) -> Frame:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, size=n_households)
    household_ids = np.arange(n_households, dtype="int64")
    person_household = np.repeat(household_ids, sizes)
    n_persons = len(person_household)
    age = rng.integers(0, 95, size=n_persons)
    is_female = rng.random(n_persons) < 0.51
    old = age >= 65
    employment_income = np.where(
        (age >= 18) & (age < 65), rng.lognormal(10.5, 0.6, size=n_persons), 0.0
    )
    receives_benefit = old & (rng.random(n_persons) < 0.9)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n_persons, dtype="int64"),
            "person_household_id": person_household,
            "age": np.minimum(age, AGE_TOP),
            "is_female": is_female,
            "employment_income": employment_income,
            "receives_benefit": receives_benefit,
            "benefit": np.where(receives_benefit, 18_000.0, 0.0),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "rent": rng.uniform(6_000, 24_000, size=n_households),
        }
    )
    weights = rng.uniform(500, 1_500, size=n_households)
    return Frame(
        {"person": person, "household": household},
        SCHEMA,
        {"household": Weights(values=weights, kind=WeightKind.CALIBRATED)},
    )


def _cell_counts(frame: Frame) -> pd.DataFrame:
    person = frame.person.copy()
    person["w"] = frame.resolve_weights("person").values
    return (
        person.groupby(["age", "is_female"], as_index=False)["w"]
        .sum()
        .rename(columns={"w": "count"})
    )


def _projection(frame: Frame, growth_old: float = 1.10, growth_young: float = 1.0):
    """Base-year cells at the frame's own counts; later years grow 65+ faster."""
    base = _cell_counts(frame)
    rows = []
    for year, power in ((BASE_YEAR, 0), (2025, 1), (2026, 2)):
        cells = base.copy()
        growth = np.where(cells["age"] >= 65, growth_old, growth_young) ** power
        cells["count"] = cells["count"] * growth
        cells["year"] = year
        rows.append(cells)
    return DemographicProjection(
        counts=pd.concat(rows), cells=("age", "is_female"), source="synthetic"
    )


def _series(frame: Frame) -> tuple[SeriesProjection, dict[str, str]]:
    base_income = float(
        (
            frame.resolve_weights("person").values * frame.person["employment_income"]
        ).sum()
    )
    # The series sits at three times the frame's own total, as a broader
    # concept would: only its growth may reach the factor, never its level.
    level = 3.0 * base_income
    series = SeriesProjection(
        totals={
            "soi.employment_income": {
                BASE_YEAR: level,
                2025: level * 1.05,
                2026: level * 1.10,
            }
        },
        indices={"cpi_u": {BASE_YEAR: 300.0, 2025: 309.0, 2026: 318.27}},
    )
    return series, {"employment_income": "soi.employment_income", "rent": "cpi_u"}


@pytest.fixture(scope="module")
def aged():
    frame = _frame()
    series, mapping = _series(frame)
    result = static_aging(
        frame,
        base_year=BASE_YEAR,
        years=(2025, 2026),
        demographics=_projection(frame),
        series=series,
        column_series=mapping,
        epochs=400,
        max_weight_ratio=3.0,
    )
    return frame, result


def _support(frame: Frame) -> pd.DataFrame:
    return (
        frame.person.groupby(["age", "is_female"])
        .size()
        .rename("n_records")
        .reset_index()
    )


def test_weights_reproduce_each_years_population_by_cell(aged) -> None:
    """Cells with real support are hit; the few single-record cells whose
    household also holds flat-cell members cannot be, and the aggregate is."""
    frame, result = aged
    support = _support(frame)
    for projection in result.years:
        fit = projection.demographic_fit.merge(
            support, on=["age", "is_female"], how="left"
        )
        supported = fit["base"] > 0
        error = (fit["achieved"] / fit["target"] - 1).abs()
        target = fit.loc[supported, "target"]
        weighted_error = float((error[supported] * target).sum() / target.sum())
        assert weighted_error < 0.01
        well_supported = supported & (fit["n_records"] >= 3)
        assert error[well_supported].max() < 0.03
        assert projection.weights.kind is WeightKind.CALIBRATED
        assert projection.fraction_within_10pct > 0.95
        old = fit["age"] >= 65
        for group in (old, ~old):
            ratio = fit.loc[group, "achieved"].sum() / fit.loc[group, "target"].sum()
            assert ratio == pytest.approx(1.0, abs=0.01)
    # The 65-and-over population grew 10 percent a year, the rest stayed flat.
    fit_2026 = result.year(2026).demographic_fit
    old = fit_2026["age"] >= 65
    assert fit_2026.loc[old, "target"].sum() == pytest.approx(
        fit_2026.loc[old, "base"].sum() * 1.21
    )
    assert fit_2026.loc[~old, "target"].sum() == pytest.approx(
        fit_2026.loc[~old, "base"].sum()
    )


def test_weights_stay_within_the_ratio_bound(aged) -> None:
    frame, result = aged
    base = frame.weights_for("household").values
    for projection in result.years:
        ratio = projection.weights.values / base
        assert ratio.max() <= 3.0 + 1e-9
        assert ratio.min() > 0


def test_total_series_factor_carries_the_projected_growth_not_the_level(
    aged,
) -> None:
    frame, result = aged
    for projection in result.years:
        aged_frame = result.frame_for(frame, projection.year)
        total = float(
            (
                aged_frame.resolve_weights("person").values
                * aged_frame.person["employment_income"]
            ).sum()
        )
        base_total = float(
            (
                frame.resolve_weights("person").values
                * frame.person["employment_income"]
            ).sum()
        )
        expected = base_total * {2025: 1.05, 2026: 1.10}[projection.year]
        assert total == pytest.approx(expected, rel=1e-9)


def test_index_series_factor_is_the_index_ratio(aged) -> None:
    _, result = aged
    assert result.year(2025).factors["rent"] == pytest.approx(309.0 / 300.0)
    assert result.year(2026).factors["rent"] == pytest.approx(318.27 / 300.0)


def test_unmapped_columns_carry_over_unchanged(aged) -> None:
    frame, result = aged
    assert "benefit" not in result.year(2025).factors
    aged_frame = result.frame_for(frame, 2025)
    assert np.array_equal(aged_frame.person["benefit"], frame.person["benefit"])
    assert np.array_equal(aged_frame.person["age"], frame.person["age"])


def test_frame_for_records_the_mass_change(aged) -> None:
    frame, result = aged
    aged_frame = result.frame_for(frame, 2025)
    assert np.array_equal(
        aged_frame.weights_for("household").values, result.year(2025).weights.values
    )
    record = aged_frame.mass_log[-1]
    assert record.entity == "household"
    assert "2025" in record.reason
    assert aged_frame.metadata["static_aging_year"] == 2025


def test_program_counts_are_predictions_not_targets(aged) -> None:
    frame, result = aged

    def recipients(f: Frame) -> float:
        return float(
            (f.resolve_weights("person").values * f.person["receives_benefit"]).sum()
        )

    base = recipients(frame)
    predicted = recipients(result.frame_for(frame, 2026))
    # Recipients live in the 65+ cells, which grew 21 percent; nothing targeted them.
    assert predicted / base == pytest.approx(1.21, rel=0.03)
    external = {"recipients": {2025: base * 1.5, 2026: base * 1.5}}
    scored = score_predictions(frame, result, {"recipients": recipients}, external)
    assert list(scored["year"]) == [2025, 2026]
    assert scored.loc[scored["year"] == 2026, "predicted"].item() == pytest.approx(
        predicted
    )
    assert scored.loc[scored["year"] == 2026, "ratio"].item() == pytest.approx(
        predicted / (base * 1.5)
    )
    # A different external number changes the score, never the prediction.
    other = score_predictions(
        frame, result, {"recipients": recipients}, {"recipients": {2026: base}}
    )
    assert other["predicted"].item() == pytest.approx(predicted)


def test_projection_anchor_targets_absolute_counts() -> None:
    frame = _frame(n_households=120, seed=3)
    projection = _projection(frame)
    scaled = DemographicProjection(
        counts=projection.counts.assign(count=projection.counts["count"] * 1.5),
        cells=projection.cells,
    )
    result = static_aging(
        frame,
        base_year=BASE_YEAR,
        years=(2025,),
        demographics=scaled,
        anchor="projection",
        epochs=400,
        max_weight_ratio=None,
    )
    fit = result.year(2025).demographic_fit
    supported = fit["base"] > 0
    assert fit.loc[supported, "target"].sum() == pytest.approx(
        scaled.for_year(2025).loc[supported.to_numpy(), "count"].sum()
    )
    ratio = fit.loc[supported, "achieved"].sum() / fit.loc[supported, "target"].sum()
    assert ratio == pytest.approx(1.0, abs=0.01)


def test_unsupported_cells_are_skipped_not_targeted() -> None:
    frame = _frame(n_households=60, seed=11)
    projection = _projection(frame)
    empty = pd.DataFrame(
        {
            "year": [BASE_YEAR, 2025, 2026],
            "age": [AGE_TOP, AGE_TOP, AGE_TOP],
            "is_female": [True, True, True],
            "count": [0.0, 0.0, 0.0],
        }
    )
    counts = projection.counts
    already = ((counts["age"] == AGE_TOP) & counts["is_female"]).any()
    if already:
        counts = counts[~((counts["age"] == AGE_TOP) & counts["is_female"])]
    with_empty = DemographicProjection(
        counts=pd.concat([counts, empty]), cells=projection.cells
    )
    frame_no_cell = frame.select(
        ~((frame.person["age"] == AGE_TOP) & frame.person["is_female"]).to_numpy()
    )
    result = static_aging(
        frame_no_cell,
        base_year=BASE_YEAR,
        years=(2025,),
        demographics=with_empty,
        epochs=50,
    )
    names = [d.name for d in result.year(2025).calibration.diagnostics]
    assert not any(f"age={AGE_TOP},is_female=True" in name for name in names)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p.counts.drop(columns=["age"]), "missing columns"),
        (lambda p: pd.concat([p.counts, p.counts.head(1)]), "duplicate"),
        (
            lambda p: p.counts[~((p.counts["year"] == 2026) & (p.counts["age"] == 3))],
            "same cells",
        ),
        (lambda p: p.counts.assign(count=-1.0), "non-negative"),
    ],
)
def test_projection_validation(mutate, match) -> None:
    projection = _projection(_frame(n_households=40))
    with pytest.raises(ValueError, match=match):
        DemographicProjection(counts=mutate(projection), cells=projection.cells)


def test_static_aging_validation() -> None:
    frame = _frame(n_households=40)
    projection = _projection(frame)
    with pytest.raises(ValueError, match="follow base year"):
        static_aging(frame, base_year=BASE_YEAR, years=(2024,), demographics=projection)
    with pytest.raises(ValueError, match="does not carry"):
        static_aging(
            frame,
            base_year=BASE_YEAR,
            years=(2025,),
            demographics=projection,
            series=SeriesProjection(),
            column_series={"rent": "cpi_u"},
        )
    with pytest.raises(ValueError, match="not found on any entity"):
        static_aging(
            frame,
            base_year=BASE_YEAR,
            years=(2025,),
            demographics=projection,
            series=SeriesProjection(indices={"cpi_u": {BASE_YEAR: 1, 2025: 1}}),
            column_series={"no_such_column": "cpi_u"},
        )
    with pytest.raises(ValueError, match="anchor"):
        static_aging(
            frame,
            base_year=BASE_YEAR,
            years=(2025,),
            demographics=projection,
            anchor="x",
        )
    with pytest.raises(ValueError, match="both totals and indices"):
        SeriesProjection(totals={"a": {2024: 1}}, indices={"a": {2024: 1}})


def test_ssa_population_projection_parses_and_top_codes(tmp_path) -> None:
    rows = []
    for year in (2024, 2025):
        for age in range(0, 91):
            rows.append(
                {
                    "Year": year,
                    "Age": age,
                    "Total": 1_000 + age + year - 2024,
                    "M Tot": 500 + age,
                    "F Tot": 500 + (year - 2024),
                }
            )
    path = tmp_path / "SSPopJul_TR2024.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    projection = ssa_population_projection(path, age_top=85)
    assert projection.cells == ("age", "is_female")
    assert projection.years == (2024, 2025)
    cells = projection.for_year(2025)
    assert cells["age"].max() == 85
    top = cells[(cells["age"] == 85) & ~cells["is_female"]]["count"].item()
    assert top == pytest.approx(sum(500 + age for age in range(85, 91)))
    assert projection.total(2025) == pytest.approx(
        sum(1_000 + age + 1 for age in range(0, 91))
    )
    assert "SSPopJul_TR2024.csv" in projection.source


def test_ssa_population_projection_matches_cps_age_bands(tmp_path):
    # Code 80 represents five ages in CPS. Their growth differs, so the
    # projection must sum their counts before taking the group's growth.
    table = pd.DataFrame(
        {
            "Year": np.repeat([2024, 2025], 8),
            "Age": list(range(79, 87)) * 2,
            "M Tot": [100] * 8 + [100, 200, 300, 400, 500, 600, 700, 800],
            "F Tot": [200] * 8 + [200, 300, 400, 500, 600, 700, 800, 900],
        }
    )
    path = tmp_path / "SSA.csv"
    table.to_csv(path, index=False)
    projection = ssa_population_projection(path, age_top=85, age_bands={80: 84})
    base = projection.for_year(2024)
    future = projection.for_year(2025)
    assert sorted(future["age"].unique()) == [79, 80, 85]
    male = ~future["is_female"]
    assert future.loc[male & future["age"].eq(79), "count"].item() == 100
    assert future.loc[male & future["age"].eq(80), "count"].item() == 2000
    assert future.loc[male & future["age"].eq(85), "count"].item() == 1500
    assert base.loc[~base["is_female"] & base["age"].eq(80), "count"].item() == 500
    for year in (2024, 2025):
        expected = table.loc[table["Year"].eq(year), ["M Tot", "F Tot"]].sum().sum()
        assert projection.total(year) == expected


@pytest.mark.parametrize(
    "bands", [{80: 85}, {80: 79}, {-1: 4}, {70: 75, 75: 79}, {80: 84.5}]
)
def test_ssa_population_projection_rejects_invalid_age_bands(tmp_path, bands):
    with pytest.raises(ValueError, match="disjoint integer ranges"):
        ssa_population_projection(tmp_path / "unused.csv", age_bands=bands)


@pytest.mark.parametrize(
    "values, year_weights",
    [
        ([100.0, -100.0], [1.0, 2.0]),  # Base net cancels; aged net does not.
        ([100.0, -50.0], [1.0, 2.0]),  # Aged net cancels; base net does not.
        ([100.0, -50.0], [1.0, 3.0]),  # Reweighting reverses the net sign.
        ([100.0, -50.0], [2.0, 1.0]),  # Positive net stays positive.
        ([100.0, -100.0], [2.0, 2.0]),  # Both nets cancel.
    ],
)
def test_signed_totals_preserve_gross_growth_and_record_signs(values, year_weights):
    from microcosm.calibrate.static_aging import _factors
    from microcosm.frame import MassChange

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "person_household_id": [1, 2],
                    "business_income": values,
                }
            ),
            "household": pd.DataFrame({"household_id": [1, 2]}),
        },
        SCHEMA,
        {"household": Weights(np.ones(2), WeightKind.CALIBRATED)},
    )
    aged = frame.with_weights(
        "household",
        Weights(np.array(year_weights), WeightKind.CALIBRATED),
        mass=MassChange(factor=None, reason="test demographic change"),
    )
    series = SeriesProjection(totals={"income": {2024: 1000.0, 2025: 1100.0}})
    factors = _factors(aged, frame, 2024, 2025, series, {"business_income": "income"})
    factor = factors["business_income"]
    # Each gross component follows the same 10% growth even when the net
    # total cancels to zero or reweighting reverses its sign.
    projected = np.array([values[0] * factor.positive, values[1] * factor.negative])
    assert np.sign(projected).tolist() == np.sign(values).tolist()
    assert projected[0] * year_weights[0] == pytest.approx(values[0] * 1.1)
    assert projected[1] * year_weights[1] == pytest.approx(values[1] * 1.1)
    assert np.dot(projected, year_weights) == pytest.approx(sum(values) * 1.1)


@pytest.mark.parametrize(
    "base_weights, year_weights",
    [([0.0, 0.0], [1.0, 1.0]), ([1.0, 1.0], [0.0, 0.0])],
)
def test_component_factor_rejects_gaining_or_losing_weighted_support(
    base_weights, year_weights
):
    from microcosm.calibrate.static_aging import _component_factor

    with pytest.raises(ValueError, match="zero weighted support"):
        _component_factor(
            np.array([-100.0, -50.0]),
            np.array(base_weights),
            np.array(year_weights),
            1.1,
            "losses",
        )


def test_component_without_weighted_support_stays_unchanged():
    from microcosm.calibrate.static_aging import _component_factor

    assert (
        _component_factor(
            np.array([-100.0, -50.0]), np.zeros(2), np.zeros(2), 1.1, "losses"
        )
        == 1.0
    )


def test_projected_frame_preserves_signed_gross_totals_and_index_ratios():
    from microcosm.frame import SignedScale

    frame = _frame(n_households=20)
    values = np.where(frame.person["age"] >= 65, -100.0, 200.0).astype(np.float32)
    values[0] = 0.0
    frame.person["business_income"] = values
    frame.person["indexed_amount"] = values
    original = frame.person.copy(deep=True)
    result = static_aging(
        frame,
        base_year=2024,
        years=(2025,),
        demographics=_projection(frame),
        series=SeriesProjection(
            totals={"business": {2024: 1000, 2025: 1100}},
            indices={"prices": {2024: 100, 2025: 105}},
        ),
        column_series={"business_income": "business", "indexed_amount": "prices"},
        epochs=50,
    )
    assert isinstance(result.year(2025).factors["business_income"], SignedScale)
    projected = result.frame_for(frame, 2025)
    base_weights = frame.resolve_weights("person").values
    year_weights = projected.resolve_weights("person").values
    after = projected.person["business_income"].to_numpy()
    assert after.dtype == np.float64
    np.testing.assert_array_equal(np.sign(after), np.sign(values))
    for mask in (values > 0, values < 0, np.ones(len(values), dtype=bool)):
        assert np.dot(after[mask], year_weights[mask]) == pytest.approx(
            np.dot(values[mask], base_weights[mask]) * 1.1, rel=1e-10
        )
    np.testing.assert_allclose(
        projected.person["indexed_amount"], values.astype(float) * 1.05
    )
    pd.testing.assert_frame_equal(frame.person, original, check_exact=True)


def test_zero_signal_still_validates_series():
    from microcosm.calibrate.static_aging import _factors

    frame = _frame(n_households=2)
    frame.person["employment_income"] = 0.0
    mapping = {"employment_income": "income"}
    valid = SeriesProjection(totals={"income": {2024: 1000.0, 2025: 1100.0}})
    assert _factors(frame, frame, 2024, 2025, valid, mapping) == {
        "employment_income": 1.0
    }
    missing = SeriesProjection(totals={"income": {2024: 1000.0}})
    with pytest.raises(KeyError, match="no value for 2025"):
        _factors(frame, frame, 2024, 2025, missing, mapping)


def test_static_aging_rejects_stale_secondary_weights():
    frame = _frame(n_households=4)
    both = Frame(
        {entity: frame.table(entity).copy() for entity in frame.entities},
        frame.schema,
        {
            "household": frame.weights_for("household"),
            "person": frame.resolve_weights("person"),
        },
    )
    with pytest.raises(ValueError, match="sole stored weight entity"):
        static_aging(
            both, base_year=2024, years=(2025,), demographics=_projection(frame)
        )


def test_person_weighted_frame_can_age_demographics():
    frame = _frame(n_households=4)
    person_frame = Frame(
        {entity: frame.table(entity).copy() for entity in frame.entities},
        frame.schema,
        {"person": frame.resolve_weights("person")},
    )
    result = static_aging(
        person_frame,
        base_year=2024,
        years=(2025,),
        demographics=_projection(person_frame, growth_old=1.1, growth_young=1.1),
        weight_entity="person",
        epochs=400,
    )
    fit = result.year(2025).demographic_fit
    np.testing.assert_allclose(fit["achieved"], fit["target"], rtol=0.01)
    np.testing.assert_allclose(
        result.frame_for(person_frame, 2025).weights_for("person").values,
        person_frame.weights_for("person").values * 1.1,
        rtol=0.01,
    )


def test_projected_frame_preserves_link_tables():
    from microcosm.frame import LinkSpec

    schema = EntitySchema(
        group_entities=("household", "firm"),
        links=(LinkSpec(name="jobs", left_entity="person", right_entity="firm"),),
    )
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [0, 1],
                    "person_household_id": [1, 1],
                    "person_firm_id": [1, 2],
                    "age": [30, 70],
                }
            ),
            "household": pd.DataFrame({"household_id": [1]}),
            "firm": pd.DataFrame({"firm_id": [1, 2]}),
            "jobs": pd.DataFrame({"person_id": [0, 0, 1], "firm_id": [1, 2, 2]}),
        },
        schema,
        {"household": Weights(np.array([100.0]), WeightKind.DESIGN)},
    )
    demographics = DemographicProjection(
        pd.DataFrame(
            {
                "year": [2024, 2024, 2025, 2025],
                "age": [30, 70, 30, 70],
                "count": [100.0, 100.0, 110.0, 110.0],
            }
        ),
        ("age",),
    )
    result = static_aging(
        frame, base_year=2024, years=(2025,), demographics=demographics, epochs=50
    )
    pd.testing.assert_frame_equal(
        result.frame_for(frame, 2025).link("jobs"), frame.link("jobs")
    )
