"""Reform-coverage smoke: a bound reform scoring $0 fails the release (#368).

The column-coverage gate proves the required input *keys* are present and carry
signal. This is the end-to-end companion: a pinned set of reforms that
mechanically bind through named input leaves must move their budget measure on
the export. If a probe scores ~$0, the leaves it binds through are absent or
degenerate — the silent-zero failure proven through the engine, not just the
column surface.

First probe: the SSI resource limit raised to $10k individual / $20k couple. It
binds only through ``ssi_countable_resources`` (= ``bank_account_assets`` +
``stock_assets`` + ``bond_assets``); with those absent, countable resources are
0 for every record, everyone already passes the resource test, and the
relaxation scores exactly $0. Until Deliverable 2 restores the asset stage this
probe fails by design — that is the gate doing its job.

The gate takes an injected ``simulate(reform) -> simulation`` (the same seam as
:mod:`populace.build.us_runtime.reform_validation`), so it unit-tests without
policyengine-us and runs live against the written release H5 in the build via
:func:`populace.build.us_runtime.reform_validation.default_simulate_factory`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from populace.build.gates import GateResult
from populace.build.us_runtime.release_input_coverage import (
    ReformCoverageProbe,
    us_release_reform_coverage_probes,
)

__all__ = [
    "us_reform_coverage_smoke_gate",
]

# simulate(reform_or_None) -> object with .calculate(measure, period).sum()
SimulateFn = Callable[[Any], Any]


def _weighted_total(simulation: Any, measure: str, period: int) -> float:
    """Weighted population total (policyengine-us MicroSeries .sum() is weighted)."""
    return float(simulation.calculate(measure, period).sum())


def _build_reform(parameter_changes: dict[str, Any]) -> Any:
    from policyengine_core.reforms import Reform

    return Reform.from_dict(parameter_changes, country_id="us")


def us_reform_coverage_smoke_gate(
    *,
    simulate: SimulateFn,
    probes: Iterable[ReformCoverageProbe] | None = None,
    period: int = 2024,
    name: str = "us_reform_coverage_smoke",
) -> GateResult:
    """Score each pinned probe on the export; a ~$0 bound reform fails.

    For each probe, the budget-measure change (reform vs baseline, signed by
    ``effect_direction``) must have magnitude at least ``min_abs_effect``. A
    smaller magnitude means the reform did not bind — its ``binding_inputs`` are
    absent/degenerate on the export — and fails the release.

    Args:
        simulate: ``simulate(None)`` builds the baseline; ``simulate(reform)``
            builds the reformed simulation. Each result answers
            ``.calculate(measure, period).sum()`` as a weighted total.
        probes: Probes to run. Defaults to the shipped manifest's probe set.
        period: The period to score at.
        name: Gate name for the manifest.

    Returns:
        The reform-coverage smoke gate result. Passes iff every probe scores at
        least its ``min_abs_effect`` in magnitude.
    """
    probes = tuple(
        probes if probes is not None else us_release_reform_coverage_probes()
    )
    if not probes:
        raise ValueError(
            "reform-coverage smoke gate needs at least one probe; a probe-less "
            "gate would pass vacuously."
        )

    baseline = simulate(None)
    failures: list[str] = []
    results: dict[str, Any] = {}
    for probe in probes:
        reform = _build_reform(dict(probe.parameter_changes))
        reformed = simulate(reform)
        baseline_total = _weighted_total(baseline, probe.budget_measure, period)
        reform_total = _weighted_total(reformed, probe.budget_measure, period)
        if probe.effect_direction == "baseline_minus_reform":
            effect = baseline_total - reform_total
        else:
            effect = reform_total - baseline_total
        results[probe.id] = {
            "name": probe.name,
            "budget_measure": probe.budget_measure,
            "baseline_total": baseline_total,
            "reform_total": reform_total,
            "effect": effect,
            "min_abs_effect": probe.min_abs_effect,
            "binding_inputs": list(probe.binding_inputs),
            "issue": probe.issue,
            "passed": abs(effect) >= probe.min_abs_effect,
        }
        if abs(effect) < probe.min_abs_effect:
            failures.append(
                f"{probe.id}: '{probe.name}' scores {effect:+,.0f} on "
                f"{probe.budget_measure} (|effect| < ${probe.min_abs_effect:,.0f}) "
                "— the reform did not bind, so its input leaves "
                f"{list(probe.binding_inputs)} are absent or degenerate on the "
                f"export. {probe.reason} Restore them ({probe.issue})."
            )

    return GateResult(
        name=name,
        passed=not failures,
        failures=tuple(failures),
        details={
            "period": int(period),
            "probes": len(probes),
            "results": results,
        },
    )
