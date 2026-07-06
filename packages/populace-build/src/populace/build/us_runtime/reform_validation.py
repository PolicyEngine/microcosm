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
    "BaselineLevelSpec",
    "ReformValidationSpec",
    "in_sample_reform_specs",
    "out_of_sample_reform_specs",
    "tax_expenditure_reform_specs",
    "soi_baseline_level_specs",
    "state_program_level_specs",
    "state_program_reform_specs",
    "state_reform_specs",
    "default_baseline_level_specs",
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
    # JCT's first full fiscal year (FY2027). FY2026 is a partial ramp year for
    # provisions effective 1/1/2026, so it understates the annual effect; the
    # dashboard shows this as the fairer like-for-like against calendar-year
    # populace liability.
    jct_score_fy2027: float | None = None
    budget_measure: str = DEFAULT_BUDGET_MEASURE
    description: str = ""
    neutralized_variable: str | list[str] | None = None
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

            # A str neutralizes one variable; a list neutralizes each. The
            # list form exists for credits whose umbrella total is a derived
            # reporting variable the tax formula never reads (e.g. md_eitc =
            # md_refundable_eitc + md_non_refundable_eitc): repeal must
            # neutralize the components on the tax path, not the umbrella.
            variables = (
                [self.neutralized_variable]
                if isinstance(self.neutralized_variable, str)
                else list(self.neutralized_variable)
            )

            class _Neutralize(Reform):
                def apply(self) -> None:
                    for variable in variables:
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
                jct_score_fy2027=(
                    float(jct["score_fy2027"])
                    if jct.get("score_fy2027") is not None
                    else None
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


def _state_program_reforms_config_path() -> Path:
    return Path(str(files("populace.build.us").joinpath("state_program_reforms.json")))


def state_program_reform_specs(
    path: Path | None = None,
    *,
    period: int,
) -> tuple[ReformValidationSpec, ...]:
    """State tax-credit repeal reforms scored against official state outlays.

    Each is a ``neutralize_variable`` repeal of a state credit, with the
    simulated change in ``state_income_tax`` compared to the state's published
    program cost. State credit variables are ``defined_for`` their state, so no
    geography filtering is needed — the national change IS the state change.
    For a refundable credit the repeal effect equals total payments; for a
    nonrefundable one it equals the amount actually used against liability,
    matching how revenue departments report claims. State program totals are
    not calibration targets, so these rows are out-of-sample.
    """
    config_path = path or _state_program_reforms_config_path()
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
                category=raw.get("category", "State program"),
                in_sample=bool(raw.get("in_sample", False)),
                period=int(raw.get("period", period)),
                jct_score=(
                    float(bench["score"]) if bench.get("score") is not None else None
                ),
                jct_window=str(bench.get("window", "")),
                jct_source=str(bench.get("source", "")),
                jct_source_url=str(bench.get("source_url", "")),
                jct_score_type=str(bench.get("score_type", "actual")),
                budget_measure=str(raw.get("budget_measure", "state_income_tax")),
                description=str(raw.get("description", "")),
                neutralized_variable=raw["neutralized_variable"],
                # Repealing a credit raises state tax by the program's cost
                # (positive), matching the positive published figure.
                effect_direction="reform_minus_baseline",
            )
        )
    return tuple(specs)


def _state_reforms_config_path() -> Path:
    return Path(str(files("populace.build.us").joinpath("state_reforms.json")))


def state_reform_specs(
    path: Path | None = None,
    *,
    period: int,
) -> tuple[ReformValidationSpec, ...]:
    """State legislative reforms scored against official state fiscal notes.

    Each is a parametric bill from the PolicyEngine state-legislative-tracker:
    the reform's parameter changes are applied on top of current law and the
    change in the state's own income-tax variable (reform − baseline) is
    compared to the state's published fiscal note. State income-tax variables
    are ``defined_for`` their state, so the national delta IS the state delta.
    Unlike OBBBA provisions these bills are NOT in the policyengine-us
    baseline, so no revert/stacking applies — each is a plain enactment.
    State income taxes are not calibration targets, so every row is
    out-of-sample.
    """
    config_path = path or _state_reforms_config_path()
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
                category=raw.get("category", "State reform"),
                in_sample=False,
                period=int(raw.get("period", period)),
                jct_score=(
                    float(bench["score"]) if bench.get("score") is not None else None
                ),
                jct_window=str(bench.get("window", "")),
                jct_source=str(bench.get("source", "")),
                jct_source_url=str(bench.get("source_url", "")),
                jct_score_type=str(bench.get("score_type", "fiscal_note")),
                budget_measure=str(raw.get("budget_measure", "state_income_tax")),
                description=str(raw.get("description", "")),
                parameter_changes=raw["parameter_changes"],
                # An enacted bill's effect is reform − baseline (a rate cut
                # yields a negative delta, matching the fiscal note's sign).
                # A bill that has since become baseline law (e.g. GA HB1001 via
                # HB463) is validated as a counterfactual revert instead and
                # declares baseline_minus_reform, like the OBBBA provisions.
                effect_direction=str(
                    raw.get("effect_direction", "reform_minus_baseline")
                ),
            )
        )
    return tuple(specs)


@dataclass(frozen=True)
class BaselineLevelSpec:
    """A baseline total compared to a published actual (no counterfactual).

    Unlike a reform, a level row needs no reform simulation: the populace value
    is the plain baseline weighted total of ``variable`` at ``period``, so
    every level shares the one baseline simulation. Levels are only meaningful
    as validation when the variable is NOT a calibration target — the shipped
    config is restricted to SOI tax items outside the target set.
    """

    id: str
    name: str
    variable: str
    period: int
    benchmark_value: float
    benchmark_year: str
    source: str
    source_url: str
    description: str = ""
    # SOI credit columns report the amount USED to offset tax (liability-
    # limited), while PE per-credit variables are the amount AVAILABLE. For
    # nonrefundable-credit lines, set cap_variable (income_tax_before_credits)
    # to compare min(available, cap) per tax unit — an approximation that
    # ignores IRS credit-ordering, so the populace side remains a slight upper
    # bound.
    cap_variable: str | None = None
    # Row grouping shown on the dashboard. SOI lines keep the historical
    # default; state-program lines override it per config file.
    category: str = "IRS SOI actual"
    # 'actual' for published administrative totals; 'approximation' for derived
    # benchmarks (e.g. state match-rate × IRS federal EITC in the state, or
    # eligible children × credit amount).
    benchmark_score_type: str = "actual"

    def __post_init__(self) -> None:
        if not self.id or not self.variable:
            raise ValueError("BaselineLevelSpec requires id and variable.")


def _soi_baseline_levels_config_path() -> Path:
    return Path(str(files("populace.build.us").joinpath("soi_baseline_levels.json")))


def _baseline_level_specs_from_config(
    config_path: Path, *, default_category: str
) -> tuple[BaselineLevelSpec, ...]:
    if not config_path.exists():
        return ()
    payload = json.loads(config_path.read_text())
    specs: list[BaselineLevelSpec] = []
    for raw in payload.get("levels", ()):
        bench = raw.get("benchmark", {})
        specs.append(
            BaselineLevelSpec(
                id=raw["id"],
                name=raw["name"],
                variable=raw["variable"],
                period=int(raw["period"]),
                benchmark_value=float(bench["value"]),
                benchmark_year=str(bench.get("year", "")),
                source=str(bench.get("source", "")),
                source_url=str(bench.get("source_url", "")),
                description=str(raw.get("description", "")),
                cap_variable=raw.get("cap_variable") or None,
                category=str(raw.get("category", default_category)),
                benchmark_score_type=str(bench.get("score_type", "actual")),
            )
        )
    return tuple(specs)


def soi_baseline_level_specs(
    path: Path | None = None,
) -> tuple[BaselineLevelSpec, ...]:
    """Curated SOI-actual baseline levels from JSON config."""
    return _baseline_level_specs_from_config(
        path or _soi_baseline_levels_config_path(),
        default_category="IRS SOI actual",
    )


def _state_program_levels_config_path() -> Path:
    return Path(str(files("populace.build.us").joinpath("state_program_levels.json")))


def state_program_level_specs(
    path: Path | None = None,
) -> tuple[BaselineLevelSpec, ...]:
    """State tax-credit baseline levels vs official state statistics.

    Each line compares the populace baseline total of a state credit variable
    (defined_for its state, so the national weighted sum IS the state total) to
    a published state figure — a Department of Revenue statistical report where
    one exists (score_type 'actual'), or a documented approximation such as the
    state match-rate × IRS federal EITC claimed in the state (score_type
    'approximation'). None of these totals are calibration targets, so every
    row is genuinely out-of-sample.
    """
    return _baseline_level_specs_from_config(
        path or _state_program_levels_config_path(),
        default_category="State program actual",
    )


def default_baseline_level_specs() -> tuple[BaselineLevelSpec, ...]:
    """All shipped baseline-level backtests: IRS SOI + state programs."""
    return (*soi_baseline_level_specs(), *state_program_level_specs())


def load_default_reform_specs(
    *,
    period: int,
    obbba_path: Path | None = None,
    tax_expenditure_path: Path | None = None,
    state_program_path: Path | None = None,
    state_reforms_path: Path | None = None,
) -> tuple[ReformValidationSpec, ...]:
    """In-sample JCT tax expenditures + out-of-sample OBBBA provisions + the
    big-provision tax-expenditure reforms (CTC/EITC/CDCC/standard/itemized) +
    state-program repeal reforms scored against official state outlays +
    state legislative reforms scored against official state fiscal notes."""
    return (
        *in_sample_reform_specs(period=period),
        *out_of_sample_reform_specs(obbba_path, period=period),
        *tax_expenditure_reform_specs(tax_expenditure_path, period=period),
        *state_program_reform_specs(state_program_path, period=period),
        *state_reform_specs(state_reforms_path, period=period),
    )


# A simulate(reform_or_None) -> object with .calculate(measure, period).sum().
SimulateFn = Callable[[Any], Any]


def _weighted_total(simulation: Any, measure: str, period: int) -> float:
    """Weighted population total of ``measure`` (MicroSeries .sum() is
    weight-aware in policyengine-us)."""
    return float(simulation.calculate(measure, period).sum())


def _capped_weighted_total(
    simulation: Any, measure: str, cap_measure: str, period: int
) -> float:
    """Weighted total of ``min(measure, cap_measure)`` per unit.

    Used for SOI credit columns, which report the amount USED to offset tax
    rather than the amount available. Weights are applied explicitly because
    elementwise numpy ops do not reliably preserve MicroSeries weights.
    """
    import numpy as np

    values = simulation.calculate(measure, period)
    caps = simulation.calculate(cap_measure, period)
    capped = np.minimum(np.asarray(values), np.asarray(caps))
    weights = getattr(values, "weights", None)
    if weights is None:
        return float(capped.sum())
    return float((capped * np.asarray(weights)).sum())


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
    baseline_levels: Sequence[BaselineLevelSpec] = (),
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
    rows, each spec's ``parameter_changes`` is the counterfactual repeal patch;
    scoring merges all repeals into a pre-OBBBA baseline, then enacts the
    provisions one at a time in JCX order and scores each *stacked* — the
    incremental effect given the lines above it — so the per-line effects sum to
    the bill total, matching JCT (see ``stacked_obbba_effects``). The shape
    matches the calibration-diagnostics dashboard's reform_validation reader.
    """
    estimates = in_sample_estimates or {}
    targets = in_sample_targets or {}
    baseline: Any = None
    baseline_totals: dict[tuple[int, str], float] = {}
    obbba_specs = tuple(spec for spec in specs if _is_obbba_spec(spec))
    obbba_pre_baseline_changes = _merged_parameter_changes(obbba_specs)
    obbba_stacked: dict[str, tuple[float, float, float]] | None = None

    def baseline_total(measure: str, at_period: int) -> float:
        nonlocal baseline
        if baseline is None:
            baseline = simulate(None)  # type: ignore[misc]
        key = (at_period, measure)
        if key not in baseline_totals:
            baseline_totals[key] = _weighted_total(baseline, measure, at_period)
        return baseline_totals[key]

    def simulation_for_parameter_changes(changes: dict[str, Any]) -> Any:
        reform = None if not changes else _build_parameter_reform(changes)
        return simulate(reform)  # type: ignore[misc]

    def stacked_obbba_effects() -> dict[str, tuple[float, float, float]]:
        """Score the OBBBA provisions *stacked* in their JCX-35-25 order.

        JCT presents each provision's budget effect incrementally — given the
        provisions above it in the document — so the line items sum to the
        bill's total. We mirror that: starting from the pre-OBBBA baseline (all
        provisions reverted), enact the provisions one at a time in ``specs``
        order, and score each as the change in its budget measure from enacting
        it on top of the lines already enacted. The per-line effects then
        telescope to the true total OBBBA effect, rather than each being
        measured in isolation against pre-OBBBA law (which ignores the
        interactions between provisions, e.g. the standard deduction and the
        personal-exemption repeal).

        Provisions are stacked per (budget_measure, period) group: the
        income-tax provisions form one cumulative stack in JCX order, while a
        provision scored on a different measure — e.g. the estate exemption,
        whose budget_measure is estate_tax because estate tax is not a
        component of income_tax — stacks within its own group (a single-member
        group reduces to enacting that provision alone). The pre-OBBBA baseline
        always merges every provision's revert, so each group is scored in the
        context of the whole bill; measures are disjoint taxes, so cross-group
        interactions are nil by construction.
        """
        if simulate is None or not obbba_specs:
            return {}
        groups: dict[tuple[str, int], list[ReformValidationSpec]] = {}
        for spec in obbba_specs:
            groups.setdefault((spec.budget_measure, spec.period), []).append(spec)
        effects: dict[str, tuple[float, float, float]] = {}
        for (measure, period), group in groups.items():
            # state 0: pre-OBBBA (every provision reverted, across all groups).
            prev_total = _weighted_total(
                simulation_for_parameter_changes(obbba_pre_baseline_changes),
                measure,
                period,
            )
            enacted: set[str] = set()
            for spec in group:
                # Enacting a provision means dropping its repeal from the
                # baseline; other groups' provisions stay reverted throughout.
                enacted |= set((spec.parameter_changes or {}).keys())
                state_changes = {
                    path: change
                    for path, change in obbba_pre_baseline_changes.items()
                    if path not in enacted
                }
                cur_total = _weighted_total(
                    simulation_for_parameter_changes(state_changes), measure, period
                )
                effects[spec.id] = (cur_total - prev_total, prev_total, cur_total)
                prev_total = cur_total
        return effects

    def simulated_effect(
        spec: ReformValidationSpec,
    ) -> tuple[float | None, float | None, float | None]:
        nonlocal obbba_stacked
        if simulate is None:
            return None, None, None
        if _is_obbba_spec(spec):
            if obbba_stacked is None:
                obbba_stacked = stacked_obbba_effects()
            return obbba_stacked.get(spec.id, (None, None, None))
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
                    "score_fy2027": (
                        None
                        if spec.jct_score_fy2027 is None
                        else _finite(spec.jct_score_fy2027)
                    ),
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
    # Baseline-level backtest rows (SOI actuals): no counterfactual — every
    # level reads the shared baseline simulation, so together they cost one
    # extra measure per line, not one simulation per line.
    def _level_total(level: BaselineLevelSpec) -> float:
        nonlocal baseline
        if level.cap_variable:
            if baseline is None:
                baseline = simulate(None)  # type: ignore[misc]
            return _capped_weighted_total(
                baseline, level.variable, level.cap_variable, level.period
            )
        return baseline_total(level.variable, level.period)

    for level in baseline_levels:
        total = None if simulate is None else _level_total(level)
        rows.append(
            {
                "id": level.id,
                "name": level.name,
                "category": level.category,
                "in_sample": False,
                "period": level.period,
                "description": level.description or None,
                "jct": {
                    "score": _finite(level.benchmark_value),
                    "score_fy2027": None,
                    "score_type": level.benchmark_score_type,
                    "window": level.benchmark_year or None,
                    "source": level.source or None,
                    "source_url": level.source_url or None,
                },
                "populace": {
                    "budget_effect": None if total is None else _finite(total),
                    "period": level.period,
                    "window": level.benchmark_year or None,
                    "measure": level.variable,
                    "baseline_total": None if total is None else _finite(total),
                    "reform_total": None,
                },
            }
        )

    has_out_of_sample = any(not spec.in_sample for spec in specs) or bool(
        baseline_levels
    )
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
