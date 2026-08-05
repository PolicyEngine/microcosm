"""ESRI-style AI adoption shock scenarios for UK person-level microdata.

Implements the scenario-microsimulation method of ESRI JR16 ("AI and income
inequality in Ireland", Doorley et al. 2026, Chapter 3): three stylised
shocks applied to a person table carrying AI exposure scores —

1. **Employment shock** (eq 3.4) — total job loss is allocated across
   occupation groups in proportion to employment-weighted complementarity-
   adjusted AI occupational exposure (C-AIOE):
   ``JobLoss_l = TotalJobLoss * (EMP_l * C_AIOE_l) / sum_l(EMP_l * C_AIOE_l)``.
   Displaced individuals are selected *randomly within group*, and summary
   statistics are averaged over ``n_draws`` random draws (ESRI use 50).
2. **Wage shock** (eq 3.5) — the aggregate wage change is distributed by a
   *separate* complementarity score theta (Pizzinelli et al. 2023), not by
   the exposure score:
   ``WageChange_l = TotalWageChange * theta_l / (sum_l theta_l*EMP_l / sum_l EMP_l)``.
3. **Capital shock** — the return to capital rises (ESRI: 1.005% -> 1.405%,
   a ~+39.8% increase in capital income), applied uniformly to recipients of
   interest/dividend-type income. Rental income is *excluded*, following
   ESRI.

This is *scenario analysis*, not causal estimation: parameters are
calibrated-from-literature anchors the analyst is expected to override (see
:data:`PRESETS`). The central anchors are the ones ESRI JR16 (Doorley,
O'Connor, O'Shea & Tuda 2026, doi:10.26504/jr16) adopts from Briggs &
Kodnani (2023): 7% job loss, and a +2.6pp wage change that is *not* a
Goldman Sachs headline number but "the median wage change estimate of a
number of studies surveyed by the authors" (JR16 fn.3, §3.2). JR16 notes
this scenario is an upper bound relative to Acemoglu (2025), "The Simple
Macroeconomics of AI", Economic Policy 40(121), 13–58 (employment effect
0.9–1.1%, with *no* corresponding wage change estimated — JR16 fn.8) and
McKinsey (2025) (wage +0.35%). The +0.4pp return-to-capital increase is
from Cazzaniga, Pizzinelli et al. (2024, IMF), applied uniformly across
households with non-rental capital income.

Scenario grid for downstream drivers
------------------------------------
JR16's own robustness grid — the intended scenario-grid shape for
downstream drivers of this module — spans employment displacement of
1–10% crossed with wage changes of +1–5%, with the +0.4pp capital shock
always on. Note the grid tops out at 10% displacement; the "high" preset
below deliberately exceeds it (see the preset notes).

Scope limitation: self-employed excluded
----------------------------------------
All shocks operate on *employees only*: the employed mask is
``employment_income > 0`` and ``age >= 16``, so persons whose labour
income is self-employment income receive no employment or wage shock.
This mirrors ESRI JR16's treatment, but it is a real scope limitation —
self-employed workers are exposed to AI too, and their exclusion is
silent in the microdata. The employment-shock summary therefore reports
``excluded_self_employed_weighted``, the weighted count of persons aged
16+ with positive ``self_employment_income`` who fall outside the
employed mask, so the omitted population is visible to downstream
consumers.

Displaced-worker contract
-------------------------
ESRI treat transitioned workers as unemployed for 9+ months: contributory
benefits exhausted, no re-employment within the scenario horizon. This module
does **not** model benefit entitlements. It sets ``employment_income`` to 0,
stores the pre-shock wage in ``pre_shock_employment_income`` and flags
``ai_displaced=True``; the downstream PolicyEngine-UK run is responsible for
modelling flagged persons as long-term unemployed.

Extension beyond ESRI JR16
--------------------------
ESRI note as a limitation that their within-group selection is random and
"does not account for the higher transition probability of younger workers".
This module's novelty is twofold:

* every shock summary is resolved **by age band** (16-24, 25-34, ..., 65+);
* the employment shock takes an optional ``youth_displacement_multiplier``
  that tilts within-group selection toward 16-24 year olds (seniority-biased
  displacement) while *preserving the group-level job-loss totals of eq 3.4*.
  The default 1.0 reproduces ESRI's age-neutral random selection.

Data surface
------------
Functions operate on a **pandas person table** and return a modified copy.
This is deliberate: :class:`populace.frame.Frame` tables are documented
read-only and rebuilding a frame requires schema/weights plumbing that shock
scenarios do not own, while the ESRI-style pipeline (shock, then re-simulate
taxes and benefits) consumes a person table. A ``Frame`` is accepted as
*input* for convenience — its person table is extracted with resolved person
weights attached — but the output is always a DataFrame.

Person-table contract: ``age``, ``employment_income``, ``ai_exposure``
(C-AIOE, drives job loss), optionally ``ai_complementarity`` (theta, drives
wage gains; uniform fallback with a warning if absent), optionally
``occupation_group`` (2-digit occupation groups for eq 3.4; weighted exposure
quintiles are used as pseudo-groups if absent), a person-weight column, and
the capital income columns to be scaled.

Determinism: the only randomness is the within-group selection of displaced
workers, seeded per draw from ``(scenario.seed, draw_index)`` (the repo
forbids unseeded randomness in builds). Wage and capital shocks are fully
deterministic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

AGE_COLUMN = "age"
EXPOSURE_COLUMN = "ai_exposure"
COMPLEMENTARITY_COLUMN = "ai_complementarity"
OCCUPATION_GROUP_COLUMN = "occupation_group"
EMPLOYMENT_INCOME_COLUMN = "employment_income"
DISPLACED_COLUMN = "ai_displaced"
PRE_SHOCK_EMPLOYMENT_INCOME_COLUMN = "pre_shock_employment_income"
DEFAULT_WEIGHT_COLUMN = "person_weight"

#: Capital income scaled by the capital shock: interest/dividend-type income
#: only. ESRI exclude rental income from the returns-to-capital gain, so
#: property/rent columns are deliberately absent.
DEFAULT_CAPITAL_COLUMNS = ("savings_interest_income", "dividend_income")

#: Default working-age bands the outputs are resolved by. Half-open
#: [lower, upper); the final band is 65+. Drivers may pass an alternative
#: scheme via the ``age_bands`` parameter of the shock functions.
AGE_BANDS = ((16, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, None))
#: Default youth band targeted by ``youth_displacement_multiplier``. Kept
#: separate from ``age_bands`` so overriding the reporting bands never
#: changes which workers the multiplier tilts toward.
YOUTH_BAND = (16, 25)

#: Column identifying self-employed persons for the scope-limitation count
#: in the employment-shock summary (see the module docstring).
SELF_EMPLOYMENT_INCOME_COLUMN = "self_employment_income"

AgeBands = tuple[tuple[int, int | None], ...]

#: Number of pseudo exposure-quintile groups used when the person table has
#: no ``occupation_group`` column.
N_FALLBACK_EXPOSURE_GROUPS = 5


@dataclass(frozen=True)
class ShockScenario:
    """One named AI-adoption shock scenario.

    Attributes:
        name: Scenario label carried into summaries.
        displacement_rate: TotalJobLoss as a share of aggregate weighted
            employment (0-1), allocated across occupation groups by eq 3.4.
        wage_uplift: Aggregate proportional wage change for non-displaced
            workers (0.026 = +2.6%), distributed by complementarity (eq 3.5).
        base_capital_return: Baseline return to capital (ESRI: 1.005%).
        capital_return_increase: Scenario increase in the return to capital
            in the same units (ESRI central: +0.4pp), so
            ``capital_uplift = capital_return_increase / base_capital_return``
            (~+39.8% capital income at the ESRI anchors).
        youth_displacement_multiplier: Relative within-group selection
            probability for 16-24 year olds versus older workers. 1.0
            reproduces ESRI's age-neutral random selection; >1 encodes
            seniority-biased displacement. Group-level job-loss totals from
            eq 3.4 are preserved either way.
        n_draws: Number of random within-group selection draws averaged in
            the employment-shock summary (ESRI use 50).
        seed: Base seed; draw ``d`` uses generator seed ``(seed, d)``. Same
            seed and data give identical results.
    """

    name: str
    displacement_rate: float
    wage_uplift: float
    base_capital_return: float = 0.01005
    capital_return_increase: float = 0.004
    youth_displacement_multiplier: float = 1.0
    n_draws: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.displacement_rate < 1.0:
            raise ValueError(
                f"displacement_rate must be in [0, 1), got {self.displacement_rate}."
            )
        if self.wage_uplift < 0.0:
            raise ValueError(f"wage_uplift must be >= 0, got {self.wage_uplift}.")
        if self.base_capital_return <= 0.0:
            raise ValueError(
                f"base_capital_return must be > 0, got {self.base_capital_return}."
            )
        if self.capital_return_increase < 0.0:
            raise ValueError(
                "capital_return_increase must be >= 0, got "
                f"{self.capital_return_increase}."
            )
        if self.youth_displacement_multiplier <= 0.0:
            raise ValueError(
                "youth_displacement_multiplier must be > 0, got "
                f"{self.youth_displacement_multiplier}."
            )
        if self.n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {self.n_draws}.")

    @property
    def capital_uplift(self) -> float:
        """Proportional capital income change implied by the return shift."""

        return self.capital_return_increase / self.base_capital_return


#: Klein & Teeselink (2025, SSRN 5516798) find that highly AI-exposed firms
#: cut total employment by -4.5% and junior employment by -5.8%. The ratio
#: 5.8 / 4.5 ~= 1.29 is a UK-evidence-calibrated value for
#: ``youth_displacement_multiplier`` (juniors displaced ~1.3x as intensely
#: as workers overall). The default presets keep the multiplier at 1.0 —
#: pure ESRI replication with age-neutral within-group selection — and the
#: ``"central_youth_tilted"`` preset applies this calibration.
KLEIN_TEESELINK_YOUTH_MULTIPLIER = 5.8 / 4.5

#: Scenario presets — calibrated-from-literature anchors, not estimates;
#: analysts should override them. JR16's own robustness grid (employment
#: 1-10% x wages +1-5%, +0.4pp capital always on) is the intended
#: scenario-grid shape for downstream drivers.
#:
#: Central: 7% job loss and +2.6pp wages, the Briggs & Kodnani (2023)
#: anchors adopted by ESRI JR16 (Doorley, O'Connor, O'Shea & Tuda 2026,
#: doi:10.26504/jr16). The +2.6pp is "the median wage change estimate of a
#: number of studies surveyed by the authors" (JR16 fn.3, §3.2) — not a
#: Goldman headline — and JR16 notes this scenario is an upper bound versus
#: Acemoglu (2025) employment 0.9-1.1% and McKinsey (2025) wage +0.35%.
#: Capital: +0.4pp return from Cazzaniga/Pizzinelli et al. (2024, IMF)
#: (1.005% -> 1.405%, ~+39.8% capital income), uniform across households
#: with non-rental capital income.
#:
#: Low: ~1% job loss, following Acemoglu (2025), "The Simple Macroeconomics
#: of AI", Economic Policy 40(121), 13-58 — employment effect 0.9-1.1%.
#: Note Acemoglu estimates *no* corresponding wage change (JR16 fn.8); the
#: preset's small wage/capital gains are this module's own muted variants,
#: not his estimates.
#:
#: High: 13% job loss, from Brynjolfsson, Chandar & Chen (2025), "Canaries
#: in the Coal Mine?" (Stanford Digital Economy Lab). Caveat: that 13% is a
#: *cohort-specific* relative employment decline for early-career workers
#: (aged 22-25) in the most-exposed occupations, not an aggregate
#: displacement estimate; it is used here deliberately as an aggressive
#: aggregate scenario. JR16's own robustness grid tops out at 10%
#: displacement, so this preset sits above JR16's upper bound.
PRESETS: dict[str, ShockScenario] = {
    "central": ShockScenario(
        name="central",
        displacement_rate=0.07,
        wage_uplift=0.026,
        capital_return_increase=0.004,
    ),
    "low": ShockScenario(
        name="low",
        displacement_rate=0.01,
        wage_uplift=0.01,
        capital_return_increase=0.002,
    ),
    "high": ShockScenario(
        name="high",
        displacement_rate=0.13,
        wage_uplift=0.04,
        capital_return_increase=0.008,
    ),
    # Central parameters with the Klein & Teeselink (2025) UK-evidence
    # youth calibration (~1.29x) applied; see
    # KLEIN_TEESELINK_YOUTH_MULTIPLIER above.
    "central_youth_tilted": ShockScenario(
        name="central_youth_tilted",
        displacement_rate=0.07,
        wage_uplift=0.026,
        capital_return_increase=0.004,
        youth_displacement_multiplier=KLEIN_TEESELINK_YOUTH_MULTIPLIER,
    ),
}


def age_band_label(age: float, age_bands: AgeBands = AGE_BANDS) -> str:
    """Human-readable band label for ``age`` (e.g. ``"25-34"``, ``"65+"``)."""

    for lower, upper in age_bands:
        if upper is None:
            if age >= lower:
                return f"{lower}+"
        elif lower <= age < upper:
            return f"{lower}-{upper - 1}"
    return "under_16"


def apply_employment_shock(
    frame_or_df: Any,
    scenario: ShockScenario,
    *,
    weight_column: str = DEFAULT_WEIGHT_COLUMN,
    draw_index: int = 0,
    age_bands: AgeBands = AGE_BANDS,
    youth_band: tuple[int, int] = YOUTH_BAND,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transition a fraction of employed persons to unemployment (eq 3.4).

    TotalJobLoss (``scenario.displacement_rate`` times aggregate weighted
    employment) is allocated to occupation groups in proportion to
    ``EMP_l * C_AIOE_l`` with employment-weighted group mean exposure, then
    displaced individuals are selected randomly within each group until the
    group's weighted quota is met. Summary statistics are averaged over
    ``scenario.n_draws`` seeded draws; the returned table is the single
    realisation ``draw_index`` (for downstream tax-benefit simulation).

    Displaced persons get ``employment_income = 0``, their pre-shock wage in
    ``pre_shock_employment_income`` and ``ai_displaced = True``. Benefit
    treatment (ESRI: long-term unemployed, contributory benefits exhausted)
    is the downstream simulation's contract, not this function's.

    If ``occupation_group`` is absent, weighted exposure quintiles of the
    employed serve as pseudo-groups.

    ``age_bands`` controls the reporting bands of the summary only;
    ``youth_band`` (half-open, default 16-24) controls which workers the
    ``youth_displacement_multiplier`` targets and is deliberately
    independent of ``age_bands`` so overriding the reporting scheme never
    changes the shock itself.

    Returns:
        ``(shocked, summary)`` — the ``draw_index`` realisation and a summary
        with draw-averaged displaced counts, weighted shares, per-age-band
        and per-group breakdowns (including the eq 3.4 quotas), plus
        ``excluded_self_employed_weighted``: the weighted count of persons
        aged 16+ with positive self-employment income outside the employed
        mask, who receive no shock (see the module docstring).
    """

    persons = _person_table(frame_or_df, weight_column)
    _require_columns(
        persons, (AGE_COLUMN, EXPOSURE_COLUMN, EMPLOYMENT_INCOME_COLUMN, weight_column)
    )
    if not 0 <= draw_index < scenario.n_draws:
        raise ValueError(
            f"draw_index must be in [0, {scenario.n_draws}), got {draw_index}."
        )
    shocked = persons.copy()
    if PRE_SHOCK_EMPLOYMENT_INCOME_COLUMN not in shocked.columns:
        shocked[PRE_SHOCK_EMPLOYMENT_INCOME_COLUMN] = shocked[
            EMPLOYMENT_INCOME_COLUMN
        ].astype(float)
    shocked[DISPLACED_COLUMN] = False

    employed = _employed_mask(shocked).to_numpy()
    weights = shocked[weight_column].to_numpy(dtype=float)
    age = shocked[AGE_COLUMN].to_numpy(dtype=float)
    exposure = shocked[EXPOSURE_COLUMN].to_numpy(dtype=float)
    groups = _occupation_groups(shocked, exposure, weights, employed)
    quotas = _group_job_loss_quotas(
        exposure, groups, weights, employed, scenario.displacement_rate
    )

    draws = [
        _draw_displaced(
            employed=employed,
            weights=weights,
            age=age,
            groups=groups,
            quotas=quotas,
            youth_multiplier=scenario.youth_displacement_multiplier,
            youth_band=youth_band,
            rng=np.random.default_rng(np.random.SeedSequence((scenario.seed, draw))),
        )
        for draw in range(scenario.n_draws)
    ]
    displaced = draws[draw_index]
    shocked.loc[displaced, DISPLACED_COLUMN] = True
    shocked.loc[displaced, EMPLOYMENT_INCOME_COLUMN] = 0.0

    employed_weight = float(weights[employed].sum())
    summary = {
        "shock": "employment",
        "scenario": scenario.name,
        "displacement_rate": scenario.displacement_rate,
        "youth_displacement_multiplier": scenario.youth_displacement_multiplier,
        "n_draws": scenario.n_draws,
        "returned_draw": draw_index,
        "employed_weighted": employed_weight,
        "displaced_count": float(np.mean([d.sum() for d in draws])),
        "displaced_weighted": float(np.mean([weights[d].sum() for d in draws])),
        "displaced_weighted_share": _safe_share(
            float(np.mean([weights[d].sum() for d in draws])), employed_weight
        ),
        "excluded_self_employed_weighted": _excluded_self_employed_weighted(
            shocked, employed, weights, age
        ),
        "by_occupation_group": _group_summary(draws, quotas, groups, weights, employed),
        "by_age_band": _age_band_summary(draws, age, weights, employed, age_bands),
    }
    return shocked, summary


def apply_wage_shock(
    frame_or_df: Any,
    scenario: ShockScenario,
    *,
    weight_column: str = DEFAULT_WEIGHT_COLUMN,
    age_bands: AgeBands = AGE_BANDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Uplift wages of non-displaced workers by complementarity (eq 3.5).

    Each employed, non-displaced worker's ``employment_income`` is scaled by
    ``1 + scenario.wage_uplift * theta_i / mean_theta`` where ``theta`` is
    the ``ai_complementarity`` score (Pizzinelli et al. 2023) and
    ``mean_theta`` its employment-weighted mean over recipients — so the
    weighted mean uplift equals ``scenario.wage_uplift`` while more
    AI-complementary workers gain more. Note this deliberately uses the
    complementarity score, *not* the exposure score that drives job loss.

    If ``ai_complementarity`` is absent (or non-positive on average), a
    uniform uplift is applied and a warning is emitted.

    Fully deterministic. Displaced workers (flagged by
    :func:`apply_employment_shock`) keep zero employment income.

    Returns:
        ``(shocked, summary)`` with the weighted mean uplift and per-age-band
        wage-bill changes.
    """

    persons = _person_table(frame_or_df, weight_column)
    _require_columns(persons, (AGE_COLUMN, EMPLOYMENT_INCOME_COLUMN, weight_column))
    shocked = persons.copy()
    if DISPLACED_COLUMN not in shocked.columns:
        shocked[DISPLACED_COLUMN] = False

    recipients = (
        _employed_mask(shocked) & ~shocked[DISPLACED_COLUMN].astype(bool)
    ).to_numpy()
    weights = shocked[weight_column].to_numpy(dtype=float)
    uplift = np.zeros(len(shocked), dtype=float)
    uniform_fallback = False
    if recipients.any() and scenario.wage_uplift > 0.0:
        relative, uniform_fallback = _relative_complementarity(
            shocked, recipients, weights
        )
        uplift[recipients] = scenario.wage_uplift * relative

    before = shocked[EMPLOYMENT_INCOME_COLUMN].to_numpy(dtype=float)
    shocked[EMPLOYMENT_INCOME_COLUMN] = before * (1.0 + uplift)

    recipient_weights = weights[recipients]
    age = shocked[AGE_COLUMN].to_numpy(dtype=float)
    wage_bill_change = weights * before * uplift
    summary = {
        "shock": "wage",
        "scenario": scenario.name,
        "wage_uplift": scenario.wage_uplift,
        "uniform_fallback": uniform_fallback,
        "recipient_count": int(recipients.sum()),
        "weighted_mean_uplift": _safe_share(
            float((uplift[recipients] * recipient_weights).sum()),
            float(recipient_weights.sum()),
        ),
        "weighted_wage_bill_change": float(wage_bill_change.sum()),
        "by_age_band": {
            label: {
                "recipient_count": int((recipients & band).sum()),
                "weighted_wage_bill_change": float(wage_bill_change[band].sum()),
            }
            for label, band in _age_band_masks(age, age_bands)
        },
    }
    return shocked, summary


def apply_capital_shock(
    frame_or_df: Any,
    scenario: ShockScenario,
    *,
    capital_columns: tuple[str, ...] = DEFAULT_CAPITAL_COLUMNS,
    weight_column: str = DEFAULT_WEIGHT_COLUMN,
    age_bands: AgeBands = AGE_BANDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale interest/dividend-type income by the return-to-capital gain.

    Every present column in ``capital_columns`` is scaled uniformly by
    ``1 + scenario.capital_uplift`` where the uplift is the transparent ratio
    ``capital_return_increase / base_capital_return`` (ESRI: 0.004 / 0.01005
    ~= +39.8%). Rental income is excluded by default, following ESRI. It is
    an error if none of the columns are present. Fully deterministic.

    Returns:
        ``(shocked, summary)`` with the weighted capital-income change
        overall and per age band.
    """

    persons = _person_table(frame_or_df, weight_column)
    _require_columns(persons, (AGE_COLUMN, weight_column))
    present = tuple(column for column in capital_columns if column in persons.columns)
    if not present:
        raise ValueError(
            f"None of the capital income column(s) {list(capital_columns)} are "
            f"present in the person table."
        )
    shocked = persons.copy()
    weights = shocked[weight_column].to_numpy(dtype=float)
    change = np.zeros(len(shocked), dtype=float)
    for column in present:
        before = shocked[column].to_numpy(dtype=float)
        shocked[column] = before * (1.0 + scenario.capital_uplift)
        change += before * scenario.capital_uplift

    age = shocked[AGE_COLUMN].to_numpy(dtype=float)
    summary = {
        "shock": "capital",
        "scenario": scenario.name,
        "base_capital_return": scenario.base_capital_return,
        "capital_return_increase": scenario.capital_return_increase,
        "capital_uplift": scenario.capital_uplift,
        "capital_columns": list(present),
        "weighted_capital_income_change": float((weights * change).sum()),
        "by_age_band": {
            label: {
                "weighted_capital_income_change": float((weights * change)[band].sum()),
            }
            for label, band in _age_band_masks(age, age_bands)
        },
    }
    return shocked, summary


def run_scenario(
    frame_or_df: Any,
    scenario: ShockScenario | str,
    *,
    capital_columns: tuple[str, ...] = DEFAULT_CAPITAL_COLUMNS,
    weight_column: str = DEFAULT_WEIGHT_COLUMN,
    seed: int | None = None,
    draw_index: int = 0,
    age_bands: AgeBands = AGE_BANDS,
    youth_band: tuple[int, int] = YOUTH_BAND,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply employment, wage and capital shocks in sequence.

    The employment summary averages over ``scenario.n_draws`` within-group
    selection draws (ESRI average over 50); the returned microdata is the
    single realisation ``draw_index``, with the wage and capital shocks
    applied on top of it deterministically.

    Args:
        frame_or_df: Person table or :class:`~populace.frame.Frame`.
        scenario: A :class:`ShockScenario`, or a preset name from
            :data:`PRESETS` (``"central"``, ``"low"``, ``"high"``,
            ``"central_youth_tilted"``).
        capital_columns: Capital income columns to scale (rent excluded by
            default, following ESRI).
        weight_column: Person-weight column name.
        seed: Optional override of the scenario base seed.
        draw_index: Which employment-shock draw the returned table realises.
        age_bands: Reporting age bands for every shock summary
            (default :data:`AGE_BANDS`).
        youth_band: Half-open band targeted by the youth displacement
            multiplier (default :data:`YOUTH_BAND`); independent of
            ``age_bands``.

    Returns:
        ``(shocked, summary)`` — the fully shocked person table (always a
        DataFrame) and a summary dict with one entry per shock plus the
        scenario parameters.
    """

    if isinstance(scenario, str):
        try:
            scenario = PRESETS[scenario]
        except KeyError:
            raise ValueError(
                f"Unknown scenario preset {scenario!r}; presets: {sorted(PRESETS)}."
            ) from None
    if seed is not None:
        scenario = replace(scenario, seed=seed)

    shocked, employment_summary = apply_employment_shock(
        frame_or_df,
        scenario,
        weight_column=weight_column,
        draw_index=draw_index,
        age_bands=age_bands,
        youth_band=youth_band,
    )
    shocked, wage_summary = apply_wage_shock(
        shocked, scenario, weight_column=weight_column, age_bands=age_bands
    )
    shocked, capital_summary = apply_capital_shock(
        shocked,
        scenario,
        capital_columns=capital_columns,
        weight_column=weight_column,
        age_bands=age_bands,
    )
    summary = {
        "scenario": scenario.name,
        "parameters": {
            "displacement_rate": scenario.displacement_rate,
            "wage_uplift": scenario.wage_uplift,
            "base_capital_return": scenario.base_capital_return,
            "capital_return_increase": scenario.capital_return_increase,
            "capital_uplift": scenario.capital_uplift,
            "youth_displacement_multiplier": scenario.youth_displacement_multiplier,
            "n_draws": scenario.n_draws,
            "seed": scenario.seed,
        },
        "employment": employment_summary,
        "wage": wage_summary,
        "capital": capital_summary,
    }
    return shocked, summary


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _person_table(frame_or_df: Any, weight_column: str) -> pd.DataFrame:
    """Normalise the input to a person DataFrame carrying ``weight_column``."""

    if isinstance(frame_or_df, pd.DataFrame):
        return frame_or_df
    person = getattr(frame_or_df, "person", None)
    resolve_weights = getattr(frame_or_df, "resolve_weights", None)
    if isinstance(person, pd.DataFrame) and callable(resolve_weights):
        table = person.copy()
        if weight_column not in table.columns:
            person_entity = frame_or_df.schema.person_entity
            table[weight_column] = np.asarray(
                resolve_weights(person_entity).values, dtype=float
            )
        return table
    raise TypeError(
        "frame_or_df must be a pandas DataFrame or a populace Frame, got "
        f"{type(frame_or_df).__name__}."
    )


def _safe_share(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or 0.0 for an empty denominator."""

    return float(numerator) / float(denominator) if denominator else 0.0


def _require_columns(persons: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in persons.columns]
    if missing:
        raise ValueError(f"Person table is missing required column(s): {missing}.")


def _employed_mask(persons: pd.DataFrame) -> pd.Series:
    """Employed persons eligible for labour shocks: earning and aged 16+."""

    return (persons[EMPLOYMENT_INCOME_COLUMN].astype(float) > 0.0) & (
        persons[AGE_COLUMN].astype(float) >= 16
    )


def _excluded_self_employed_weighted(
    persons: pd.DataFrame,
    employed: np.ndarray,
    weights: np.ndarray,
    age: np.ndarray,
) -> float:
    """Weighted count of self-employed persons outside the employed mask.

    Persons aged 16+ with positive ``self_employment_income`` who are not in
    the employee mask receive no employment or wage shock (mirroring ESRI
    JR16's employees-only treatment); this makes the omitted population
    visible in the summary. Returns 0.0 when the column is absent.
    """

    if SELF_EMPLOYMENT_INCOME_COLUMN not in persons.columns:
        return 0.0
    self_employed = (
        persons[SELF_EMPLOYMENT_INCOME_COLUMN].to_numpy(dtype=float) > 0.0
    ) & (age >= 16)
    return float(weights[self_employed & ~employed].sum())


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Inverse-CDF weighted quantiles (matches populace.frame.accounting)."""

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    proportion = cumulative / cumulative[-1]
    indices = np.searchsorted(proportion, quantiles, side="left")
    return sorted_values[np.clip(indices, 0, len(sorted_values) - 1)]


def _occupation_groups(
    persons: pd.DataFrame,
    exposure: np.ndarray,
    weights: np.ndarray,
    employed: np.ndarray,
) -> np.ndarray:
    """Occupation-group label per person for the eq 3.4 allocation.

    Uses the ``occupation_group`` column when present; otherwise weighted
    exposure quintiles of the employed serve as pseudo-groups (labelled
    ``exposure_q1`` ... ``exposure_q5``).
    """

    if OCCUPATION_GROUP_COLUMN in persons.columns:
        return persons[OCCUPATION_GROUP_COLUMN].astype(str).to_numpy()
    interior = np.arange(1, N_FALLBACK_EXPOSURE_GROUPS) / N_FALLBACK_EXPOSURE_GROUPS
    if employed.any():
        edges = _weighted_quantiles(exposure[employed], weights[employed], interior)
    else:
        edges = np.quantile(exposure, interior)
    indices = np.searchsorted(edges, exposure, side="right")
    labels = np.array(
        [f"exposure_q{index + 1}" for index in range(N_FALLBACK_EXPOSURE_GROUPS)]
    )
    return labels[indices]


def _group_job_loss_quotas(
    exposure: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    employed: np.ndarray,
    displacement_rate: float,
) -> dict[str, float]:
    """Weighted job-loss quota per occupation group (eq 3.4).

    ``JobLoss_l = TotalJobLoss * (EMP_l * C_AIOE_l) / sum_l(EMP_l * C_AIOE_l)``
    with ``EMP_l`` the weighted employment of group ``l`` and ``C_AIOE_l`` its
    employment-weighted mean exposure. Negative group scores (possible with
    standardised C-AIOE) are floored at zero for the allocation. Quotas are
    capped at group employment.
    """

    total_employment = float(weights[employed].sum())
    total_job_loss = displacement_rate * total_employment
    if total_job_loss <= 0.0 or not employed.any():
        return {}
    group_employment: dict[str, float] = {}
    group_score: dict[str, float] = {}
    for label in np.unique(groups[employed]):
        in_group = employed & (groups == label)
        emp = float(weights[in_group].sum())
        mean_exposure = _safe_share(
            float((exposure[in_group] * weights[in_group]).sum()), emp
        )
        group_employment[label] = emp
        group_score[label] = emp * max(mean_exposure, 0.0)
    total_score = sum(group_score.values())
    if total_score <= 0.0:
        raise ValueError(
            "Cannot allocate job loss: all employment-weighted group exposure "
            "scores are non-positive."
        )
    return {
        label: min(total_job_loss * score / total_score, group_employment[label])
        for label, score in group_score.items()
    }


def _draw_displaced(
    *,
    employed: np.ndarray,
    weights: np.ndarray,
    age: np.ndarray,
    groups: np.ndarray,
    quotas: dict[str, float],
    youth_multiplier: float,
    youth_band: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """One random within-group selection meeting each group's weighted quota.

    Selection order within a group is a weighted random permutation
    (Efraimidis-Spirakis keys) with ``youth_band`` workers' selection
    intensity scaled by ``youth_multiplier``; persons are taken in order
    until the cumulative weight is as close as possible to the group's
    quota. With ``youth_multiplier == 1`` this is ESRI's uniform random
    selection.
    """

    displaced = np.zeros(len(weights), dtype=bool)
    youth_lower, youth_upper = youth_band
    for label, quota in quotas.items():
        if quota <= 0.0:
            continue
        candidates = np.flatnonzero(employed & (groups == label))
        intensity = np.where(
            (age[candidates] >= youth_lower) & (age[candidates] < youth_upper),
            youth_multiplier,
            1.0,
        )
        keys = rng.exponential(size=len(candidates)) / intensity
        order = candidates[np.argsort(keys, kind="stable")]
        cumulative = np.cumsum(weights[order])
        cut = int(np.searchsorted(cumulative, quota, side="left"))
        if cut < len(order):
            # Include the boundary person only if that lands closer to the
            # quota than stopping short of it.
            below = cumulative[cut - 1] if cut > 0 else 0.0
            if abs(cumulative[cut] - quota) <= abs(quota - below):
                cut += 1
        displaced[order[:cut]] = True
    return displaced


def _relative_complementarity(
    persons: pd.DataFrame,
    recipients: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Eq 3.5 relative wage-gain factors over the recipients.

    Returns ``(theta_i / mean_theta, uniform_fallback)`` with ``mean_theta``
    the employment-weighted mean complementarity over recipients, so the
    factors have weighted mean 1. Falls back to uniform factors (with a
    warning) when the ``ai_complementarity`` column is absent or its weighted
    mean is non-positive.
    """

    n_recipients = int(recipients.sum())
    if COMPLEMENTARITY_COLUMN not in persons.columns:
        warnings.warn(
            f"Person table has no {COMPLEMENTARITY_COLUMN!r} column; applying a "
            "uniform wage uplift instead of the eq 3.5 complementarity-graded "
            "distribution.",
            stacklevel=3,
        )
        return np.ones(n_recipients), True
    theta = persons[COMPLEMENTARITY_COLUMN].to_numpy(dtype=float)[recipients]
    recipient_weights = weights[recipients]
    mean_theta = _safe_share(
        float((theta * recipient_weights).sum()), float(recipient_weights.sum())
    )
    if mean_theta <= 0.0:
        warnings.warn(
            f"Weighted mean {COMPLEMENTARITY_COLUMN!r} over wage-shock "
            "recipients is non-positive; applying a uniform wage uplift.",
            stacklevel=3,
        )
        return np.ones(n_recipients), True
    return theta / mean_theta, False


def _age_band_masks(
    age: np.ndarray, age_bands: AgeBands
) -> list[tuple[str, np.ndarray]]:
    """``(label, boolean mask)`` per age band, in band order."""

    masks: list[tuple[str, np.ndarray]] = []
    for lower, upper in age_bands:
        if upper is None:
            masks.append((f"{lower}+", age >= lower))
        else:
            masks.append((f"{lower}-{upper - 1}", (age >= lower) & (age < upper)))
    return masks


def _age_band_summary(
    draws: list[np.ndarray],
    age: np.ndarray,
    weights: np.ndarray,
    employed: np.ndarray,
    age_bands: AgeBands,
) -> dict[str, dict[str, float]]:
    """Draw-averaged employment-shock breakdown per age band."""

    result: dict[str, dict[str, float]] = {}
    for label, band in _age_band_masks(age, age_bands):
        employed_weight = float(weights[employed & band].sum())
        displaced_weight = float(np.mean([weights[d & band].sum() for d in draws]))
        result[label] = {
            "employed_weighted": employed_weight,
            "displaced_count": float(np.mean([(d & band).sum() for d in draws])),
            "displaced_weighted": displaced_weight,
            "displaced_weighted_share": _safe_share(displaced_weight, employed_weight),
        }
    return result


def _group_summary(
    draws: list[np.ndarray],
    quotas: dict[str, float],
    groups: np.ndarray,
    weights: np.ndarray,
    employed: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Draw-averaged employment-shock breakdown per occupation group."""

    result: dict[str, dict[str, float]] = {}
    for label, quota in sorted(quotas.items()):
        in_group = employed & (groups == label)
        employed_weight = float(weights[in_group].sum())
        displaced_weight = float(np.mean([weights[d & in_group].sum() for d in draws]))
        result[label] = {
            "employed_weighted": employed_weight,
            "job_loss_quota_weighted": float(quota),
            "displaced_weighted": displaced_weight,
            "displaced_weighted_share": _safe_share(displaced_weight, employed_weight),
        }
    return result
