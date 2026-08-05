"""AI shock runner: pure summary math, grid construction, dataclass I/O.

Everything here runs without policyengine_uk installed or real data — the
runner's pure parts are the unit under test. The single engine-facing smoke
test is skipped when policyengine_uk is unavailable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from populace.build.uk_runtime.ai_shock_runner import (
    GRID_CAPITAL_RETURN_INCREASE,
    AgeBandDelta,
    DecileDelta,
    PopulationMetrics,
    ScenarioResult,
    age_band_deltas,
    build_scenario_grid,
    decile_deltas,
    decile_ids,
    gini,
    scenario_result_from_dict,
    weighted_mean,
    write_grid_csv,
    write_grid_json,
)

# ----------------------------------------------------------------------
# Grid construction
# ----------------------------------------------------------------------


def test_grid_is_the_jr16_robustness_grid():
    grid = build_scenario_grid()
    assert len(grid) == 50  # 10 displacement x 5 wage points
    rates = sorted({scenario.displacement_rate for scenario in grid})
    uplifts = sorted({scenario.wage_uplift for scenario in grid})
    assert rates == pytest.approx([r / 100 for r in range(1, 11)])
    assert uplifts == pytest.approx([u / 100 for u in range(1, 6)])
    # The +0.4pp capital shock is always on.
    assert all(
        scenario.capital_return_increase == GRID_CAPITAL_RETURN_INCREASE
        for scenario in grid
    )
    assert len({scenario.name for scenario in grid}) == 50
    assert any(scenario.name == "grid_emp7pct_wage2pct" for scenario in grid)


def test_grid_custom_margins_and_seed():
    grid = build_scenario_grid(
        displacement_rates=(0.02,), wage_uplifts=(0.01, 0.03), seed=7, n_draws=3
    )
    assert [scenario.name for scenario in grid] == [
        "grid_emp2pct_wage1pct",
        "grid_emp2pct_wage3pct",
    ]
    assert all(s.seed == 7 and s.n_draws == 3 for s in grid)


# ----------------------------------------------------------------------
# Pure summary math
# ----------------------------------------------------------------------


def test_weighted_mean():
    assert weighted_mean(np.array([1.0, 3.0]), np.array([1.0, 3.0])) == 2.5
    assert weighted_mean(np.array([]), np.array([])) == 0.0


def test_gini_known_values():
    # Perfect equality.
    assert gini(np.full(5, 10.0), np.ones(5)) == pytest.approx(0.0)
    # One person holds everything: G -> (n-1)/n with equal weights.
    values = np.array([0.0, 0.0, 0.0, 100.0])
    assert gini(values, np.ones(4)) == pytest.approx(0.75)
    # Weight-replication invariance: weight 2 equals a duplicated record.
    a = gini(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0]))
    b = gini(np.array([1.0, 2.0, 2.0, 3.0]), np.ones(4))
    assert a == pytest.approx(b)


def test_gini_rejects_bad_input():
    with pytest.raises(ValueError, match="finite"):
        gini(np.array([1.0, np.nan]), np.ones(2))
    with pytest.raises(ValueError, match="non-negative"):
        gini(np.array([1.0, 2.0]), np.array([1.0, -1.0]))


def test_decile_ids_equal_weights():
    income = np.arange(100, dtype=float)
    deciles = decile_ids(income, np.ones(100))
    assert deciles.min() == 1 and deciles.max() == 10
    counts = np.bincount(deciles)[1:]
    assert counts.tolist() == [10] * 10
    # Monotone in income.
    assert (np.diff(deciles[np.argsort(income)]) >= 0).all()


def test_decile_ids_respects_weights():
    # One heavy low-income record should fill the bottom deciles.
    income = np.array([1.0, 2.0, 3.0])
    deciles = decile_ids(income, np.array([8.0, 1.0, 1.0]))
    assert deciles[0] == 1 and deciles[2] == 10


def test_decile_deltas_fixed_at_baseline():
    baseline = np.arange(1.0, 101.0)
    shocked = baseline.copy()
    shocked[-10:] = 0.0  # top decile wiped out, JR16-style displacement
    deltas = decile_deltas(baseline, shocked, np.ones(100))
    assert len(deltas) == 10
    assert all(d.mean_income_change == 0.0 for d in deltas[:-1])
    top = deltas[-1]
    assert top.decile == 10
    # Reported within the *baseline* decile even though income collapsed.
    assert top.baseline_mean_income == pytest.approx(95.5)
    assert top.mean_income_change == pytest.approx(-95.5)
    assert top.weighted_population == 10.0


def test_age_band_deltas():
    age = np.array([20.0, 30.0, 70.0])
    baseline = np.array([100.0, 200.0, 300.0])
    shocked = np.array([50.0, 200.0, 330.0])
    deltas = age_band_deltas(age, baseline, shocked, np.ones(3))
    by_band = {d.age_band: d for d in deltas}
    assert set(by_band) == {"16-24", "25-34", "35-44", "45-54", "55-64", "65+"}
    assert by_band["16-24"].mean_income_change == pytest.approx(-50.0)
    assert by_band["25-34"].mean_income_change == pytest.approx(0.0)
    assert by_band["65+"].mean_income_change == pytest.approx(30.0)
    assert by_band["35-44"].weighted_population == 0.0


# ----------------------------------------------------------------------
# Dataclass round-trip and writers
# ----------------------------------------------------------------------


def _result(name: str = "central") -> ScenarioResult:
    baseline = PopulationMetrics(
        gov_balance=100.0,
        poverty_rate_bhc=0.15,
        poverty_rate_ahc=None,
        gini_equivalised_income=0.32,
    )
    shocked = PopulationMetrics(
        gov_balance=90.0,
        poverty_rate_bhc=0.17,
        poverty_rate_ahc=None,
        gini_equivalised_income=0.34,
    )
    return ScenarioResult(
        scenario=name,
        displacement_rate=0.07,
        wage_uplift=0.026,
        capital_return_increase=0.004,
        baseline=baseline,
        shocked=shocked,
        exchequer_cost=10.0,
        poverty_rate_change_bhc=0.02,
        poverty_rate_change_ahc=None,
        gini_change=0.02,
        by_decile=(
            DecileDelta(1, 10.0, 9.0, -1.0, 100.0),
            DecileDelta(2, 20.0, 21.0, 1.0, 100.0),
        ),
        by_age_band=(AgeBandDelta("16-24", 15.0, 12.0, -3.0, 50.0),),
        employment_status_applied=True,
        shock_summary={"scenario": name, "employment": {"displaced_count": 3.0}},
    )


def test_scenario_result_json_round_trip():
    result = _result()
    payload = json.loads(json.dumps(result.to_dict()))
    assert scenario_result_from_dict(payload) == result


def test_write_grid_json_and_csv(tmp_path):
    results = [_result("central"), _result("low")]
    json_path = tmp_path / "out" / "results.json"
    write_grid_json(results, json_path)
    loaded = json.loads(json_path.read_text())
    assert [scenario_result_from_dict(item) for item in loaded] == results

    paths = write_grid_csv(results, tmp_path / "out")
    assert set(paths) == {"overall", "by_decile", "by_age_band"}
    import pandas as pd

    overall = pd.read_csv(paths["overall"])
    assert overall["scenario"].tolist() == ["central", "low"]
    assert overall["exchequer_cost"].tolist() == [10.0, 10.0]
    by_decile = pd.read_csv(paths["by_decile"])
    assert len(by_decile) == 4  # 2 scenarios x 2 deciles
    by_age = pd.read_csv(paths["by_age_band"])
    assert by_age["age_band"].tolist() == ["16-24", "16-24"]


# ----------------------------------------------------------------------
# Engine-facing path (skip without policyengine_uk)
# ----------------------------------------------------------------------


def test_run_ai_shock_analysis_requires_exposure_or_major_group():
    import pandas as pd

    from populace.build.uk_runtime.ai_shock_runner import _attach_exposure_columns

    with pytest.raises(ValueError, match="soc_major_group"):
        _attach_exposure_columns(
            pd.DataFrame({"age": [30.0]}),
            exposure=None,
            soc_major_group=None,
            complementarity=None,
        )


def test_attach_exposure_from_major_group_uses_crosswalk():
    import pandas as pd

    from populace.build.uk_runtime.ai_exposure import (
        load_major_group_ai_exposure_table,
    )
    from populace.build.uk_runtime.ai_shock_runner import _attach_exposure_columns

    persons = _attach_exposure_columns(
        pd.DataFrame({"age": [30.0, 40.0]}),
        exposure=None,
        soc_major_group=[2000, 9000],
        complementarity=None,
    )
    table = load_major_group_ai_exposure_table()
    assert persons["ai_exposure"].iloc[0] == pytest.approx(table.loc["2", "c_aioe"])
    assert persons["ai_exposure"].iloc[1] == pytest.approx(table.loc["9", "c_aioe"])
    assert np.isfinite(persons["ai_complementarity"]).all()


def test_engine_smoke():
    pytest.importorskip("policyengine_uk")
    # A full engine run needs a real UK H5 dataset (licensed, not in repo);
    # the importable surface is asserted instead.
    from populace.build.uk_runtime.ai_shock_runner import run_ai_shock_analysis

    assert callable(run_ai_shock_analysis)
