"""End-to-end driver for ESRI-style UK AI shock scenario analysis.

Pipeline (modeled on :mod:`populace.build.uk_runtime.local_runner`'s runner
conventions — frozen result dataclasses, lazy ``policyengine_uk`` imports
behind the ``uk`` extra, JSON/CSV writers):

1. Load a PolicyEngine-UK single-year H5 dataset
   (:func:`~populace.build.uk_runtime.local_runner.load_uk_dataset`).
2. Extract the person shock table (age, employment/self-employment income,
   interest/dividend income, person weight) from a baseline simulation and
   attach AI exposure — either draws from
   :mod:`~populace.build.uk_runtime.exposure_imputation`, or the zero-model
   crosswalk baseline via
   :func:`~populace.build.uk_runtime.ai_exposure.exposure_for_major_group`
   from an observed ``soc_major_group`` column (FRS ``1000``-``9000``
   coding, normalised with
   :func:`~populace.build.uk_runtime.frs_occupation.frs_major_group_to_digit`).
3. Run scenarios — the :data:`~populace.build.uk_runtime.ai_shock_scenarios.PRESETS`
   and/or the ESRI JR16 robustness grid (:func:`build_scenario_grid`:
   employment loss 1-10% crossed with wage +1-5%, the +0.4pp capital-return
   shock always on) — through
   :func:`~populace.build.uk_runtime.ai_shock_scenarios.run_scenario`.
4. Write the shocked person table back into a fresh ``policyengine_uk``
   Simulation with ``set_input``: ``employment_income`` (zero for displaced,
   complementarity-graded uplift for the rest), ``savings_interest_income``
   and ``dividend_income`` (capital-return uplift).

   **Displaced-worker expressibility note.** ESRI treat displaced workers as
   long-term unemployed. PolicyEngine-UK exposes ``employment_status`` as an
   input enum (set to ``UNEMPLOYED`` for displaced persons here), but the
   full ESRI contract — 9+ months unemployed, contributory benefits
   exhausted — is **only partially expressible**: PE-UK models entitlement
   from current-period inputs and does not carry an unemployment-duration
   input, so contributory JSA/ESA exhaustion cannot be encoded. The runner
   sets ``employment_income`` and ``employment_status`` and records in the
   result whether the status input was accepted
   (``employment_status_applied``); benefit-exhaustion nuances beyond that
   are a documented scope limitation.
5. Compute deltas versus baseline: Exchequer cost (change in ``gov_balance``),
   poverty rates (BHC and AHC, as the installed PE-UK model exposes them),
   and the Gini of equivalised household disposable income — overall, by
   baseline income decile (the true JR16 replication view: deciles are fixed
   at their baseline assignment), and by age band (the extension, using
   :data:`~populace.build.uk_runtime.ai_shock_scenarios.AGE_BANDS`).

Everything summary-mathematical (:func:`gini`, :func:`weighted_mean`,
:func:`decile_ids`, :func:`decile_deltas`, :func:`age_band_deltas`,
:func:`build_scenario_grid`, the dataclasses and writers) is a pure function
importable and unit-testable without ``policyengine_uk`` installed or real
data; only :func:`run_ai_shock_analysis` / :func:`run_grid` touch the engine.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.uk_runtime.ai_shock_scenarios import (
    AGE_BANDS,
    DEFAULT_CAPITAL_COLUMNS,
    DEFAULT_WEIGHT_COLUMN,
    DISPLACED_COLUMN,
    EMPLOYMENT_INCOME_COLUMN,
    PRESETS,
    ShockScenario,
    run_scenario,
)
from populace.build.uk_runtime.local_runner import load_uk_dataset

__all__ = [
    "AgeBandDelta",
    "DecileDelta",
    "GRID_CAPITAL_RETURN_INCREASE",
    "GRID_DISPLACEMENT_RATES",
    "GRID_WAGE_UPLIFTS",
    "PopulationMetrics",
    "ScenarioResult",
    "age_band_deltas",
    "build_scenario_grid",
    "decile_deltas",
    "decile_ids",
    "gini",
    "run_ai_shock_analysis",
    "run_grid",
    "scenario_result_from_dict",
    "weighted_mean",
    "write_grid_csv",
    "write_grid_json",
]

#: JR16 robustness grid margins: employment displacement 1-10% crossed with
#: wage changes of +1-5%; the +0.4pp return-to-capital shock is always on.
GRID_DISPLACEMENT_RATES = tuple(rate / 100 for rate in range(1, 11))
GRID_WAGE_UPLIFTS = tuple(uplift / 100 for uplift in range(1, 6))
GRID_CAPITAL_RETURN_INCREASE = 0.004

#: Person-level inputs written back into the shocked simulation.
SHOCKED_INPUT_COLUMNS = (EMPLOYMENT_INCOME_COLUMN, *DEFAULT_CAPITAL_COLUMNS)


# ----------------------------------------------------------------------
# Pure summary math (no engine, no data files)
# ----------------------------------------------------------------------


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean, 0.0 on zero total weight."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total else 0.0


def gini(
    values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray
) -> float:
    """Weighted Gini coefficient of ``values``.

    Standard weighted formulation: sort by value, then
    ``G = 1 - 2 * sum(w_i * (S_i - y_i*w_i/2)) / (W * S_n)`` with ``S_i`` the
    cumulative weighted income. Returns 0.0 for empty/zero-mass input.

    Raises:
        ValueError: On mismatched lengths, non-finite entries, or negative
            weights.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("values and weights must be 1-D arrays of equal length.")
    if not (np.isfinite(values).all() and np.isfinite(weights).all()):
        raise ValueError("values and weights must be finite.")
    if (weights < 0).any():
        raise ValueError("weights must be non-negative.")
    total_weight = float(weights.sum())
    if len(values) == 0 or total_weight == 0.0:
        return 0.0
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_values * sorted_weights)
    total_income = float(cumulative[-1])
    if total_income == 0.0:
        return 0.0
    # Trapezoid area under the weighted Lorenz curve.
    lorenz_mass = float(
        (sorted_weights * (cumulative - sorted_values * sorted_weights / 2.0)).sum()
    )
    return float(1.0 - 2.0 * lorenz_mass / (total_weight * total_income))


def decile_ids(
    income: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    *,
    n_quantiles: int = 10,
) -> np.ndarray:
    """Weighted decile assignment (1..``n_quantiles``) per record.

    Each record is placed by the lower edge of its cumulative weighted-income
    rank (so a heavy record spanning several decile boundaries reports in the
    lowest decile it touches). This is the *baseline* assignment used for the
    JR16 replication view — the shocked deltas are reported within these
    fixed deciles.

    Raises:
        ValueError: On empty input, non-finite entries, or non-positive
            total weight.
    """

    income = np.asarray(income, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if income.ndim != 1 or income.shape != weights.shape or len(income) == 0:
        raise ValueError("income and weights must be equal-length, non-empty 1-D.")
    if not (np.isfinite(income).all() and np.isfinite(weights).all()):
        raise ValueError("income and weights must be finite.")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be non-negative with positive total.")

    order = np.argsort(income, kind="stable")
    cumulative = np.cumsum(weights[order])
    lower_edge = (cumulative - weights[order]) / cumulative[-1]
    interior = np.arange(1, n_quantiles) / n_quantiles
    bands = np.searchsorted(interior, lower_edge, side="right") + 1
    out = np.empty(len(income), dtype=int)
    out[order] = bands
    return out


@dataclass(frozen=True)
class DecileDelta:
    """Per-baseline-income-decile change in mean equivalised income."""

    decile: int
    baseline_mean_income: float
    shocked_mean_income: float
    mean_income_change: float
    weighted_population: float


@dataclass(frozen=True)
class AgeBandDelta:
    """Per-age-band change in mean equivalised income (the JR16 extension)."""

    age_band: str
    baseline_mean_income: float
    shocked_mean_income: float
    mean_income_change: float
    weighted_population: float


def decile_deltas(
    baseline_income: np.ndarray,
    shocked_income: np.ndarray,
    weights: np.ndarray,
    *,
    deciles: np.ndarray | None = None,
) -> tuple[DecileDelta, ...]:
    """Mean-income deltas by baseline income decile (JR16 replication view).

    Deciles are computed on (or supplied fixed at) the *baseline* income, so
    a household displaced into a lower income still reports within its
    pre-shock decile — exactly the JR16 incidence table shape.
    """

    baseline_income = np.asarray(baseline_income, dtype=float)
    shocked_income = np.asarray(shocked_income, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if deciles is None:
        deciles = decile_ids(baseline_income, weights)
    deciles = np.asarray(deciles, dtype=int)
    out = []
    for decile in range(1, int(deciles.max()) + 1 if len(deciles) else 1):
        mask = deciles == decile
        base = weighted_mean(baseline_income[mask], weights[mask])
        shocked = weighted_mean(shocked_income[mask], weights[mask])
        out.append(
            DecileDelta(
                decile=decile,
                baseline_mean_income=base,
                shocked_mean_income=shocked,
                mean_income_change=shocked - base,
                weighted_population=float(weights[mask].sum()),
            )
        )
    return tuple(out)


def age_band_deltas(
    age: np.ndarray,
    baseline_income: np.ndarray,
    shocked_income: np.ndarray,
    weights: np.ndarray,
) -> tuple[AgeBandDelta, ...]:
    """Mean-income deltas by :data:`AGE_BANDS` (the extension beyond JR16)."""

    age = np.asarray(age, dtype=float)
    baseline_income = np.asarray(baseline_income, dtype=float)
    shocked_income = np.asarray(shocked_income, dtype=float)
    weights = np.asarray(weights, dtype=float)
    out = []
    for lower, upper in AGE_BANDS:
        if upper is None:
            label, mask = f"{lower}+", age >= lower
        else:
            label, mask = f"{lower}-{upper - 1}", (age >= lower) & (age < upper)
        base = weighted_mean(baseline_income[mask], weights[mask])
        shocked = weighted_mean(shocked_income[mask], weights[mask])
        out.append(
            AgeBandDelta(
                age_band=label,
                baseline_mean_income=base,
                shocked_mean_income=shocked,
                mean_income_change=shocked - base,
                weighted_population=float(weights[mask].sum()),
            )
        )
    return tuple(out)


def build_scenario_grid(
    *,
    displacement_rates: Sequence[float] = GRID_DISPLACEMENT_RATES,
    wage_uplifts: Sequence[float] = GRID_WAGE_UPLIFTS,
    capital_return_increase: float = GRID_CAPITAL_RETURN_INCREASE,
    n_draws: int = 50,
    seed: int = 0,
    youth_displacement_multiplier: float = 1.0,
) -> tuple[ShockScenario, ...]:
    """The ESRI JR16 robustness grid as :class:`ShockScenario` objects.

    Defaults reproduce JR16's own grid: employment displacement 1-10%
    crossed with wage changes +1-5%, the +0.4pp capital shock always on.
    Scenario names encode the point, e.g. ``"grid_emp7pct_wage2.6pct"``.
    """

    def _pct(value: float) -> str:
        percent = value * 100
        return f"{percent:g}pct"

    return tuple(
        ShockScenario(
            name=f"grid_emp{_pct(displacement)}_wage{_pct(wage)}",
            displacement_rate=float(displacement),
            wage_uplift=float(wage),
            capital_return_increase=float(capital_return_increase),
            youth_displacement_multiplier=youth_displacement_multiplier,
            n_draws=n_draws,
            seed=seed,
        )
        for displacement in displacement_rates
        for wage in wage_uplifts
    )


# ----------------------------------------------------------------------
# Result dataclasses and writers (pure; local_runner style)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationMetrics:
    """One simulation's headline distributional metrics.

    ``poverty_rate_bhc``/``poverty_rate_ahc`` are None where the installed
    PolicyEngine-UK model does not expose the corresponding poverty flag.
    """

    gov_balance: float
    poverty_rate_bhc: float | None
    poverty_rate_ahc: float | None
    gini_equivalised_income: float


@dataclass(frozen=True)
class ScenarioResult:
    """Baseline-versus-shock deltas for one AI shock scenario.

    ``exchequer_cost`` is ``baseline.gov_balance - shocked.gov_balance``: a
    positive number is a cost to the Exchequer.
    """

    scenario: str
    displacement_rate: float
    wage_uplift: float
    capital_return_increase: float
    baseline: PopulationMetrics
    shocked: PopulationMetrics
    exchequer_cost: float
    poverty_rate_change_bhc: float | None
    poverty_rate_change_ahc: float | None
    gini_change: float
    by_decile: tuple[DecileDelta, ...]
    by_age_band: tuple[AgeBandDelta, ...]
    employment_status_applied: bool
    shock_summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (round-trips via
        :func:`scenario_result_from_dict`)."""

        return asdict(self) | {"shock_summary": dict(self.shock_summary)}


def scenario_result_from_dict(payload: Mapping[str, Any]) -> ScenarioResult:
    """Rebuild a :class:`ScenarioResult` from :meth:`ScenarioResult.to_dict`."""

    data = dict(payload)
    data["baseline"] = PopulationMetrics(**data["baseline"])
    data["shocked"] = PopulationMetrics(**data["shocked"])
    data["by_decile"] = tuple(DecileDelta(**row) for row in data["by_decile"])
    data["by_age_band"] = tuple(AgeBandDelta(**row) for row in data["by_age_band"])
    return ScenarioResult(**data)


def _overall_rows(results: Sequence[ScenarioResult]) -> list[dict[str, Any]]:
    return [
        {
            "scenario": result.scenario,
            "displacement_rate": result.displacement_rate,
            "wage_uplift": result.wage_uplift,
            "capital_return_increase": result.capital_return_increase,
            "exchequer_cost": result.exchequer_cost,
            "poverty_rate_change_bhc": result.poverty_rate_change_bhc,
            "poverty_rate_change_ahc": result.poverty_rate_change_ahc,
            "gini_change": result.gini_change,
            "baseline_gini": result.baseline.gini_equivalised_income,
            "shocked_gini": result.shocked.gini_equivalised_income,
            "employment_status_applied": result.employment_status_applied,
        }
        for result in results
    ]


def write_grid_json(
    results: Sequence[ScenarioResult],
    path: str | Path,
) -> None:
    """Write full scenario results (deciles, age bands, summaries) as JSON."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True)
    )


def write_grid_csv(
    results: Sequence[ScenarioResult],
    output_dir: str | Path,
    *,
    prefix: str = "ai_shock",
) -> dict[str, Path]:
    """Write overall / by-decile / by-age-band CSVs, local_runner style.

    Returns the written paths keyed ``overall``, ``by_decile``,
    ``by_age_band``.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "overall": out / f"{prefix}_overall.csv",
        "by_decile": out / f"{prefix}_by_decile.csv",
        "by_age_band": out / f"{prefix}_by_age_band.csv",
    }
    pd.DataFrame(_overall_rows(results)).to_csv(paths["overall"], index=False)
    pd.DataFrame(
        [
            {"scenario": result.scenario, **asdict(row)}
            for result in results
            for row in result.by_decile
        ]
    ).to_csv(paths["by_decile"], index=False)
    pd.DataFrame(
        [
            {"scenario": result.scenario, **asdict(row)}
            for result in results
            for row in result.by_age_band
        ]
    ).to_csv(paths["by_age_band"], index=False)
    return paths


# ----------------------------------------------------------------------
# Engine-facing pipeline (lazy policyengine_uk imports; `uk` extra)
# ----------------------------------------------------------------------

#: Person-level columns the shock table is built from (plus the weight).
_BASELINE_PERSON_VARIABLES = (
    "age",
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
)


def _default_simulation_factory(dataset: Any) -> Any:
    try:
        from policyengine_uk import Microsimulation
    except ImportError as exc:  # pragma: no cover - engine-absent path
        raise ImportError(
            "AI shock analysis requires policyengine-uk. Install the UK "
            "engine (the `uk` extra) before calling run_ai_shock_analysis()."
        ) from exc
    return Microsimulation(dataset=dataset)


def _values(result: Any) -> np.ndarray:
    if hasattr(result, "values"):
        return np.asarray(result.values)
    return np.asarray(result)


def _calculate(sim: Any, variable: str, period: int | str, map_to: str) -> np.ndarray:
    return _values(sim.calculate(variable, period=period, map_to=map_to)).astype(float)


def _try_poverty_rate(
    sim: Any, variable: str, period: int | str, weights: np.ndarray
) -> float | None:
    """Person-weighted poverty rate, or None if the model lacks the flag."""

    try:
        flags = _calculate(sim, variable, period, "person")
    except Exception:
        return None
    return weighted_mean(flags, weights)


def _population_metrics(
    sim: Any,
    period: int | str,
    *,
    person_weights: np.ndarray,
    household_weights: np.ndarray,
) -> tuple[PopulationMetrics, np.ndarray]:
    """Headline metrics plus person-level equivalised household income."""

    gov_balance = float(
        (_calculate(sim, "gov_balance", period, "household") * household_weights).sum()
    )
    equivalised_household = _calculate(
        sim, "equiv_household_net_income", period, "household"
    )
    equivalised_person = _calculate(sim, "equiv_household_net_income", period, "person")
    metrics = PopulationMetrics(
        gov_balance=gov_balance,
        poverty_rate_bhc=_try_poverty_rate(
            sim, "in_poverty_bhc", period, person_weights
        ),
        poverty_rate_ahc=_try_poverty_rate(
            sim, "in_poverty_ahc", period, person_weights
        ),
        gini_equivalised_income=gini(equivalised_household, household_weights),
    )
    return metrics, equivalised_person


def _apply_shocked_inputs(sim: Any, shocked: pd.DataFrame, period: int | str) -> bool:
    """Write the shocked person table into ``sim`` via ``set_input``.

    Sets the income inputs, then attempts the displaced persons'
    ``employment_status`` (PE-UK's employment-status input enum). Returns
    whether the status input was accepted — see the module docstring's
    expressibility note; contributory-benefit exhaustion is not encodable.
    """

    for column in SHOCKED_INPUT_COLUMNS:
        if column in shocked.columns:
            sim.set_input(column, period, shocked[column].to_numpy(dtype=float))
    displaced = shocked[DISPLACED_COLUMN].to_numpy(dtype=bool)
    if not displaced.any():
        return True
    try:
        current = np.asarray(
            _values(sim.calculate("employment_status", period=period, map_to="person")),
            dtype=object,
        )
        status = current.copy()
        status[displaced] = "UNEMPLOYED"
        sim.set_input("employment_status", period, status)
        return True
    except Exception:
        return False


def run_ai_shock_analysis(
    dataset: Any | str | Path,
    scenario: ShockScenario | str,
    *,
    exposure: Sequence[float] | np.ndarray | pd.Series | None = None,
    soc_major_group: Sequence[float] | np.ndarray | pd.Series | None = None,
    complementarity: Sequence[float] | np.ndarray | pd.Series | None = None,
    period: int | str | None = None,
    draw_index: int = 0,
    simulation_factory: Callable[[Any], Any] | None = None,
) -> ScenarioResult:
    """Run one AI shock scenario end-to-end against a UK H5 dataset.

    Args:
        dataset: A ``UKSingleYearDataset`` (or its H5 path; loaded via
            :func:`~populace.build.uk_runtime.local_runner.load_uk_dataset`).
        scenario: A :class:`ShockScenario` or a preset name from
            :data:`~populace.build.uk_runtime.ai_shock_scenarios.PRESETS`.
        exposure: One AI-exposure score (C-AIOE) per person, e.g. QRF draws
            from :mod:`~populace.build.uk_runtime.exposure_imputation`. When
            omitted, ``soc_major_group`` must be given and the zero-model
            crosswalk baseline
            (:func:`~populace.build.uk_runtime.ai_exposure.exposure_for_major_group`)
            is used.
        soc_major_group: Observed FRS major groups (``1000``-``9000``; see
            :mod:`~populace.build.uk_runtime.frs_occupation`), used to derive
            exposure — and complementarity, when not supplied — from the
            shipped crosswalk.
        complementarity: Optional per-person theta for the eq 3.5 wage
            distribution; the scenario falls back per
            :mod:`~populace.build.uk_runtime.ai_shock_scenarios`.
        period: Simulation period; inferred from the dataset when omitted.
        draw_index: Which employment-shock draw the shocked simulation
            realises.
        simulation_factory: Override for tests; defaults to
            ``policyengine_uk.Microsimulation``.

    Returns:
        A frozen :class:`ScenarioResult`.
    """

    dataset_obj = (
        load_uk_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset
    )
    run_period = _infer_period(dataset_obj, period)
    factory = simulation_factory or _default_simulation_factory

    baseline_sim = factory(dataset_obj)
    person_weights = _calculate(baseline_sim, "person_weight", run_period, "person")
    household_weights = _calculate(
        baseline_sim, "household_weight", run_period, "household"
    )
    persons = pd.DataFrame(
        {
            variable: _calculate(baseline_sim, variable, run_period, "person")
            for variable in _BASELINE_PERSON_VARIABLES
        }
    )
    persons[DEFAULT_WEIGHT_COLUMN] = person_weights
    persons = _attach_exposure_columns(
        persons,
        exposure=exposure,
        soc_major_group=soc_major_group,
        complementarity=complementarity,
    )

    baseline_metrics, baseline_equivalised = _population_metrics(
        baseline_sim,
        run_period,
        person_weights=person_weights,
        household_weights=household_weights,
    )
    age = persons["age"].to_numpy(dtype=float)

    shocked_table, shock_summary = run_scenario(
        persons, scenario, draw_index=draw_index
    )

    shocked_sim = factory(dataset_obj)
    employment_status_applied = _apply_shocked_inputs(
        shocked_sim, shocked_table, run_period
    )
    shocked_metrics, shocked_equivalised = _population_metrics(
        shocked_sim,
        run_period,
        person_weights=person_weights,
        household_weights=household_weights,
    )

    scenario_name = str(shock_summary["scenario"])
    parameters = shock_summary["parameters"]
    return ScenarioResult(
        scenario=scenario_name,
        displacement_rate=float(parameters["displacement_rate"]),
        wage_uplift=float(parameters["wage_uplift"]),
        capital_return_increase=float(parameters["capital_return_increase"]),
        baseline=baseline_metrics,
        shocked=shocked_metrics,
        exchequer_cost=baseline_metrics.gov_balance - shocked_metrics.gov_balance,
        poverty_rate_change_bhc=_optional_delta(
            baseline_metrics.poverty_rate_bhc, shocked_metrics.poverty_rate_bhc
        ),
        poverty_rate_change_ahc=_optional_delta(
            baseline_metrics.poverty_rate_ahc, shocked_metrics.poverty_rate_ahc
        ),
        gini_change=shocked_metrics.gini_equivalised_income
        - baseline_metrics.gini_equivalised_income,
        by_decile=decile_deltas(
            baseline_equivalised, shocked_equivalised, person_weights
        ),
        by_age_band=age_band_deltas(
            age, baseline_equivalised, shocked_equivalised, person_weights
        ),
        employment_status_applied=employment_status_applied,
        shock_summary=shock_summary,
    )


def run_grid(
    dataset: Any | str | Path,
    scenarios: Sequence[ShockScenario | str] | None = None,
    *,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> tuple[ScenarioResult, ...]:
    """Run many scenarios (default: presets plus the JR16 robustness grid).

    Loads the dataset once and forwards ``kwargs`` (exposure attachment,
    period, factory, ...) to :func:`run_ai_shock_analysis` per scenario.
    With ``output_dir`` set, writes the JSON and CSV outputs via
    :func:`write_grid_json` / :func:`write_grid_csv`.
    """

    dataset_obj = (
        load_uk_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset
    )
    if scenarios is None:
        scenarios = (*PRESETS.values(), *build_scenario_grid())
    results = tuple(
        run_ai_shock_analysis(dataset_obj, scenario, **kwargs) for scenario in scenarios
    )
    if output_dir is not None:
        write_grid_json(results, Path(output_dir) / "ai_shock_results.json")
        write_grid_csv(results, output_dir)
    return results


def _optional_delta(baseline: float | None, shocked: float | None) -> float | None:
    if baseline is None or shocked is None:
        return None
    return shocked - baseline


def _infer_period(dataset: Any, period: int | str | None) -> int | str:
    if period is not None:
        return period
    for attr in ("time_period", "fiscal_year", "default_calculation_period"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return value
    raise ValueError("period is required when it cannot be inferred from the dataset.")


def _attach_exposure_columns(
    persons: pd.DataFrame,
    *,
    exposure: Any,
    soc_major_group: Any,
    complementarity: Any,
) -> pd.DataFrame:
    """Attach ``ai_exposure`` (and ``ai_complementarity`` when derivable)."""

    persons = persons.copy()
    if exposure is None and soc_major_group is None:
        raise ValueError(
            "Provide either per-person exposure scores (e.g. QRF draws from "
            "exposure_imputation) or the observed soc_major_group column "
            "(frs_occupation) to derive them from the crosswalk."
        )
    if exposure is not None:
        persons["ai_exposure"] = np.asarray(exposure, dtype=float)
    if soc_major_group is not None:
        from populace.build.uk_runtime.ai_exposure import exposure_for_major_group
        from populace.build.uk_runtime.frs_occupation import frs_major_group_to_digit

        digits = frs_major_group_to_digit(np.asarray(soc_major_group))
        if exposure is None:
            persons["ai_exposure"] = exposure_for_major_group(digits, "c_aioe")
        if complementarity is None:
            theta = exposure_for_major_group(digits, "complementarity")
            finite = np.isfinite(theta)
            if finite.any() and not finite.all():
                # Persons without an observed major group (children, missing
                # SOC) take the mean theta so the eq 3.5 factors stay finite.
                theta = np.where(finite, theta, float(theta[finite].mean()))
            persons["ai_complementarity"] = theta
    if complementarity is not None:
        persons["ai_complementarity"] = np.asarray(complementarity, dtype=float)
    if not np.isfinite(persons["ai_exposure"].to_numpy(dtype=float)).all():
        raise ValueError(
            "ai_exposure contains non-finite entries; persons without a "
            "resolvable exposure score cannot enter the employment shock. "
            "Impute or fill exposure upstream (exposure_imputation's blind "
            "fallback covers persons without an observed major group)."
        )
    return persons
