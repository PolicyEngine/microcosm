"""Reform validation: score policy reforms on the calibrated dataset and
compare the budget effect to the authority's (JCT's) official score.

Where ``calibration_diagnostics.json`` reports how well the calibrated weights
reproduce their *calibration targets*, ``reform_validation.json`` reports a
downstream property the calibration did not directly optimize: how closely the
dataset reproduces the *budget effects of scored policy reforms*. Two kinds of
reform are validated, and each row is labelled so a consumer can tell them
apart:

* **in-sample** — the JCT tax-expenditure reforms that are themselves
  calibration targets (``US_JCT_TAX_EXPENDITURE_REFORMS``). The dataset was
  tuned to hit these, so agreement is expected; the row is published for
  completeness and provenance, flagged ``in_sample=True``.
* **out-of-sample** — reforms the calibration never saw (e.g. provisions of
  the 2025 One Big Beautiful Bill Act), curated in ``obbba_reforms.json`` with
  their JCT scores. These are the genuine test of dataset fidelity.

The simulation is isolated behind an injected ``simulate`` callable so the
payload assembly is unit-testable without policyengine-us; the default factory
builds a ``Microsimulation`` over the freshly written release H5.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from populace.build.us_runtime.fiscal_targets import (
    US_JCT_TAX_EXPENDITURE_REFORMS,
    SimpleTaxExpenditureReform,
)

__all__ = [
    "REFORM_VALIDATION_SCHEMA_VERSION",
    "ReformValidationSpec",
    "in_sample_reform_specs",
    "out_of_sample_reform_specs",
    "tax_expenditure_reform_specs",
    "load_default_reform_specs",
    "reform_validation_payload",
    "write_reform_validation",
]

#: Schema version of reform_validation.json. The calibration-diagnostics
#: dashboard keys its reader on it; bump with any shape change.
REFORM_VALIDATION_SCHEMA_VERSION = 1

#: The budget effect of a reform is the weighted-sum change of this variable
#: between the reform and baseline simulations, unless a spec overrides it. For
#: income-tax provisions this is the simulated federal income-tax revenue change
#: — the quantity JCT scores (− = revenue loss / cost; + = revenue raised).
DEFAULT_BUDGET_MEASURE = "income_tax"


def _build_parameter_reform(parameter_changes: dict[str, Any]) -> Any:
    from policyengine_core.reforms import Reform

    return Reform.from_dict(parameter_changes, country_id="us")


@dataclass(frozen=True)
class ReformValidationSpec:
    """One reform to score on the dataset and compare to its JCT figure.

    Exactly one of ``neutralized_variable`` (an in-sample tax-expenditure
    neutralization) or ``parameter_changes`` (an out-of-sample
    ``Reform.from_dict`` payload) defines the reform.
    """

    id: str
    name: str
    category: str
    in_sample: bool
    period: int
    # The published figure to compare against. None for provisions neither JCT
    # nor Treasury score (e.g. the regular standard deduction, which both treat
    # as baseline) — those rows publish the simulated magnitude with no error.
    jct_score: float | None
    jct_window: str
    jct_source: str
    jct_source_url: str
    jct_score_type: str = "conventional"
    budget_measure: str = DEFAULT_BUDGET_MEASURE
    description: str = ""
    neutralized_variable: str | None = None
    parameter_changes: dict[str, Any] | None = None
    # How the budget effect is signed relative to the simulations. JCT scores a
    # *tax expenditure* as the revenue raised by repeal (reform − baseline, the
    # neutralize convention), but scores an *enacted provision* as the effect of
    # enacting it. OBBBA is already in the policyengine-us baseline, so its
    # provisions are validated by a counterfactual *revert* reform — there the
    # provision's effect is baseline − reform (JCT enactment sign).
    effect_direction: str = "reform_minus_baseline"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ReformValidationSpec.id is required.")
        has_neutralize = bool(self.neutralized_variable)
        has_params = bool(self.parameter_changes)
        if has_neutralize == has_params:
            raise ValueError(
                f"{self.id}: provide exactly one of neutralized_variable or "
                "parameter_changes."
            )
        if self.effect_direction not in {
            "reform_minus_baseline",
            "baseline_minus_reform",
        }:
            raise ValueError(
                f"{self.id}: effect_direction must be 'reform_minus_baseline' or "
                "'baseline_minus_reform'."
            )

    def build_reform(self) -> Any:
        """Construct the policyengine reform object for this spec.

        Imports are lazy so the module (and its unit tests) load without
        policyengine-us installed.
        """
        if self.neutralized_variable:
            from policyengine_core.reforms import Reform

            variable = self.neutralized_variable

            class _Neutralize(Reform):
                def apply(self) -> None:
                    self.neutralize_variable(variable)

            return _Neutralize

        return _build_parameter_reform(self.parameter_changes or {})


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _is_obbba_spec(spec: ReformValidationSpec) -> bool:
    return (
        not spec.in_sample
        and spec.parameter_changes is not None
        and spec.category.strip().lower() == "obbba"
    )


def _merged_parameter_changes(specs: Iterable[ReformValidationSpec]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    owner: dict[str, str] = {}
    for spec in specs:
        for path, change in (spec.parameter_changes or {}).items():
            if path in merged and merged[path] != change:
                raise ValueError(
                    f"{spec.id}: OBBBA parameter path {path!r} conflicts with "
                    f"{owner[path]}."
                )
            merged[path] = change
            owner[path] = spec.id
    return merged


def in_sample_reform_specs(
    reforms: Iterable[SimpleTaxExpenditureReform] = US_JCT_TAX_EXPENDITURE_REFORMS,
    *,
    period: int,
) -> tuple[ReformValidationSpec, ...]:
    """The JCT tax-expenditure calibration targets as validation specs."""
    specs: list[ReformValidationSpec] = []
    for reform in reforms:
        specs.append(
            ReformValidationSpec(
                id=reform.target_name,
                name=reform.target_name,
                category="JCT tax expenditure",
                in_sample=True,
                period=int(period),
                # The JCT figure for an in-sample reform is the calibration
                # target's own value, supplied at payload time via
                # in_sample_targets — it lives in the ledger now, not on the
                # reform object.
                jct_score=None,
                jct_window="annual",
                jct_source=reform.source or "JCT tax-expenditure (calibration target)",
                jct_source_url="",
                budget_measure=reform.output_variable or DEFAULT_BUDGET_MEASURE,
                neutralized_variable=reform.neutralized_variable,
            )
        )
    return tuple(specs)


def _obbba_config_path() -> Path:
    return Path(str(files("populace.build.us").joinpath("obbba_reforms.json")))


def out_of_sample_reform_specs(
    path: Path | None = None,
    *,
    period: int,
) -> tuple[ReformValidationSpec, ...]:
    """Curated out-of-sample reforms (OBBBA provisions) from JSON config."""
    config_path = path or _obbba_config_path()
    if not config_path.exists():
        return ()
    payload = json.loads(config_path.read_text())
    specs: list[ReformValidationSpec] = []
    for raw in payload.get("reforms", ()):
        jct = raw.get("jct", {})
        specs.append(
            ReformValidationSpec(
                id=raw["id"],
                name=raw["name"],
                category=raw.get("category", "OBBBA"),
                in_sample=False,
                period=int(raw.get("period", period)),
                jct_score=(
                    float(jct["score"]) if jct.get("score") is not None else None
                ),
                jct_window=str(jct.get("window", "")),
                jct_source=str(jct.get("source", "")),
                jct_source_url=str(jct.get("source_url", "")),
                jct_score_type=str(jct.get("score_type", "conventional")),
                budget_measure=str(raw.get("budget_measure", DEFAULT_BUDGET_MEASURE)),
                description=str(raw.get("description", "")),
                parameter_changes=raw["parameter_changes"],
                # OBBBA provisions are baked into the baseline, so the config
                # encodes a revert; the provision's effect is baseline − reform.
                effect_direction=str(
                    raw.get("effect_direction", "baseline_minus_reform")
                ),
            )
        )
    return tuple(specs)


def _tax_expenditure_config_path() -> Path:
    return Path(
        str(files("populace.build.us").joinpath("tax_expenditure_reforms.json"))
    )


def tax_expenditure_reform_specs(
    path: Path | None = None,
    *,
    period: int,
) -> tuple[ReformValidationSpec, ...]:
    """Big-provision tax-expenditure reforms from JSON config.

    Each is a ``neutralize_variable`` reform (repeal the provision) whose
    simulated revenue change is compared to a published tax-expenditure figure
    (JCT where it scores the provision, Treasury otherwise — recorded per row).
    The ``in_sample`` flag reflects whether the dataset is calibrated to that
    provision (e.g. EITC is, the standard deduction is not), so a consumer can
    weight the result accordingly.
    """
    config_path = path or _tax_expenditure_config_path()
    if not config_path.exists():
        return ()
    payload = json.loads(config_path.read_text())
    specs: list[ReformValidationSpec] = []
    for raw in payload.get("reforms", ()):
        bench = raw.get("benchmark", {})
        specs.append(
            ReformValidationSpec(
                id=raw["id"],
                name=raw["name"],
                category=raw.get("category", "Tax expenditure"),
                in_sample=bool(raw.get("in_sample", False)),
                period=int(raw.get("period", period)),
                jct_score=(
                    float(bench["score"]) if bench.get("score") is not None else None
                ),
                jct_window=str(bench.get("window", "")),
                jct_source=str(bench.get("source", "")),
                jct_source_url=str(bench.get("source_url", "")),
                jct_score_type=str(bench.get("score_type", "tax_expenditure")),
                budget_measure=str(raw.get("budget_measure", DEFAULT_BUDGET_MEASURE)),
                description=str(raw.get("description", "")),
                neutralized_variable=raw["neutralized_variable"],
                # Neutralizing the provision raises tax by the expenditure amount
                # (positive), matching the positive published figure.
                effect_direction="reform_minus_baseline",
            )
        )
    return tuple(specs)


def load_default_reform_specs(
    *,
    period: int,
    obbba_path: Path | None = None,
    tax_expenditure_path: Path | None = None,
) -> tuple[ReformValidationSpec, ...]:
    """In-sample JCT tax expenditures + out-of-sample OBBBA provisions + the
    big-provision tax-expenditure reforms (CTC/EITC/CDCC/standard/itemized)."""
    return (
        *in_sample_reform_specs(period=period),
        *out_of_sample_reform_specs(obbba_path, period=period),
        *tax_expenditure_reform_specs(tax_expenditure_path, period=period),
    )


# A simulate(reform_or_None) -> object with .calculate(measure, period).sum().
SimulateFn = Callable[[Any], Any]


def _weighted_total(simulation: Any, measure: str, period: int) -> float:
    """Weighted population total of ``measure`` (MicroSeries .sum() is
    weight-aware in policyengine-us)."""
    return float(simulation.calculate(measure, period).sum())


def default_simulate_factory(dataset_path: Path) -> SimulateFn:
    """Build a simulate() that runs a Microsimulation over the release H5."""

    def simulate(reform: Any) -> Any:
        from policyengine_us import Microsimulation
        from policyengine_us.data import USSingleYearDataset

        dataset = USSingleYearDataset(file_path=str(dataset_path))
        if reform is None:
            return Microsimulation(dataset=dataset)
        return Microsimulation(dataset=dataset, reform=reform)

    return simulate


def reform_validation_payload(
    specs: Sequence[ReformValidationSpec],
    *,
    period: int,
    simulate: SimulateFn | None = None,
    in_sample_estimates: dict[str, float] | None = None,
    in_sample_targets: dict[str, float] | None = None,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Score each reform on the dataset and render the JSON-stable payload.

    In-sample reforms are JCT tax-expenditure *calibration targets*: their
    populace budget effect is the calibrated ``final_estimate`` the calibration
    already produced — passed in via ``in_sample_estimates`` (keyed by spec id),
    so no extra simulation is run for them. Out-of-sample reforms (OBBBA
    provisions) are simulated: a baseline is built once and each reform's budget
    effect is the weighted-sum change of its budget measure (reform − baseline).

    ``simulate`` is required only if any out-of-sample spec is present (or an
    in-sample estimate is missing); when absent, those rows publish a null
    budget effect rather than failing the build. OBBBA rows are special because
    policyengine-us already carries OBBBA in its baseline, while JCX-35-25
    scores enactment relative to a pre-OBBBA present-law baseline. For those
    rows, each spec's ``parameter_changes`` remains the counterfactual repeal
    patch, but scoring first merges all OBBBA repeal patches into a pre-OBBBA
    baseline and then adds the row's provision back. The shape matches the
    calibration-diagnostics dashboard's reform_validation reader.
    """
    estimates = in_sample_estimates or {}
    targets = in_sample_targets or {}
    baseline: Any = None
    baseline_totals: dict[tuple[int, str], float] = {}
    parameter_reform_sims: dict[str, Any] = {}
    obbba_specs = tuple(spec for spec in specs if _is_obbba_spec(spec))
    obbba_pre_baseline_changes = _merged_parameter_changes(obbba_specs)

    def parameter_changes_key(changes: dict[str, Any]) -> str:
        return json.dumps(changes, sort_keys=True, separators=(",", ":"))

    def baseline_total(measure: str, at_period: int) -> float:
        nonlocal baseline
        if baseline is None:
            baseline = simulate(None)  # type: ignore[misc]
        key = (at_period, measure)
        if key not in baseline_totals:
            baseline_totals[key] = _weighted_total(baseline, measure, at_period)
        return baseline_totals[key]

    def simulation_for_parameter_changes(changes: dict[str, Any]) -> Any:
        key = parameter_changes_key(changes)
        if key not in parameter_reform_sims:
            reform = None if not changes else _build_parameter_reform(changes)
            parameter_reform_sims[key] = simulate(reform)  # type: ignore[misc]
        return parameter_reform_sims[key]

    def obbba_component_effect(
        spec: ReformValidationSpec,
    ) -> tuple[float | None, float | None, float | None]:
        if simulate is None:
            return None, None, None
        component_paths = set((spec.parameter_changes or {}).keys())
        component_on_changes = {
            path: change
            for path, change in obbba_pre_baseline_changes.items()
            if path not in component_paths
        }
        pre_obbba = simulation_for_parameter_changes(obbba_pre_baseline_changes)
        component_on = simulation_for_parameter_changes(component_on_changes)
        base = _weighted_total(pre_obbba, spec.budget_measure, spec.period)
        reform_total = _weighted_total(component_on, spec.budget_measure, spec.period)
        return reform_total - base, base, reform_total

    def simulated_effect(
        spec: ReformValidationSpec,
    ) -> tuple[float | None, float | None, float | None]:
        if simulate is None:
            return None, None, None
        if _is_obbba_spec(spec):
            return obbba_component_effect(spec)
        base = baseline_total(spec.budget_measure, spec.period)
        reform_total = _weighted_total(
            simulate(spec.build_reform()), spec.budget_measure, spec.period
        )
        raw = reform_total - base
        # A counterfactual revert measures the provision as baseline − reform.
        effect = raw if spec.effect_direction == "reform_minus_baseline" else -raw
        return effect, base, reform_total

    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.in_sample and spec.id in estimates:
            effect: float | None = float(estimates[spec.id])
            base_total: float | None = None
            reform_total: float | None = None
        else:
            effect, base_total, reform_total = simulated_effect(spec)
        # In-sample reforms get their JCT figure from the calibration target.
        effective_jct = spec.jct_score
        if effective_jct is None and spec.in_sample and spec.id in targets:
            effective_jct = targets[spec.id]
        rows.append(
            {
                "id": spec.id,
                "name": spec.name,
                "category": spec.category,
                "in_sample": spec.in_sample,
                "period": spec.period,
                "description": spec.description or None,
                "jct": {
                    "score": None if effective_jct is None else _finite(effective_jct),
                    "score_type": spec.jct_score_type,
                    "window": spec.jct_window or None,
                    "source": spec.jct_source or None,
                    "source_url": spec.jct_source_url or None,
                },
                "populace": {
                    "budget_effect": None if effect is None else _finite(effect),
                    "period": spec.period,
                    "window": spec.jct_window or None,
                    "measure": spec.budget_measure,
                    "baseline_total": None
                    if base_total is None
                    else _finite(base_total),
                    "reform_total": None
                    if reform_total is None
                    else _finite(reform_total),
                },
            }
        )

    # Self-describing flag so a null out-of-sample row can be told apart from a
    # genuinely-zero one, and so a release built with simulation skipped is
    # never mistaken for one where the dataset simply failed the fidelity test.
    # False only when out-of-sample reforms exist but no simulate() was given.
    has_out_of_sample = any(not spec.in_sample for spec in specs)
    out_of_sample_simulated = simulate is not None or not has_out_of_sample
    payload: dict[str, Any] = {
        "schema_version": REFORM_VALIDATION_SCHEMA_VERSION,
        "baseline_period": int(period),
        "scoring_window": "see per-reform jct.window",
        "out_of_sample_simulated": out_of_sample_simulated,
        "reforms": rows,
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def write_reform_validation(payload: dict[str, Any], path: Path | str) -> Path:
    """Write the reform-validation payload as ``reform_validation.json``."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path
