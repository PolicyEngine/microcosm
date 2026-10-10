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
:mod:`microcosm.build.us_runtime.reform_validation`), so it unit-tests without
policyengine-us. In the build it runs live against the written release H5
through the release tool's household-batched post-export scorer
(``tools/build_us_fiscal_refresh_release.py``, microcosm#956), which serves the
same seam from one engine per household batch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.release_input_coverage import (
    ListParameterEdit,
    ReformCoverageProbe,
    installed_list_parameter_baseline,
    list_parameter_baseline,
    resolve_probe_parameter_changes,
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


def _build_reform(probe: ReformCoverageProbe) -> Any:
    """The probe's reform, its list edits resolved on the installed engine.

    A list edit resolves against the installed PolicyEngine-US baseline here,
    so an edit that no longer fits that baseline fails before anything is
    scored. The returned reform also re-reads each edited list from the system
    it is applied to, over the edit's whole period, and refuses to apply
    unless that list is the baseline the edit was resolved against or already
    the reform's own resolution, so the resolved list can only replace the
    baseline it was derived from.

    Raises:
        ValueError: Naming the probe, if a list edit does not fit the installed
            baseline.
    """
    from policyengine_core.reforms import Reform

    if probe.neutralized_variable:
        variable = probe.neutralized_variable

        class _Neutralize(Reform):
            def apply(self) -> None:
                self.neutralize_variable(variable)
                neutralized = self.variables[variable]
                neutralized.default_value = (
                    False if neutralized.value_type is bool else 0
                )

        return _Neutralize
    if not probe.list_edits:
        return Reform.from_dict(dict(probe.parameter_changes), country_id="us")

    baselines: dict[str, list[str]] = {}

    def installed_baseline(path: str, edit: ListParameterEdit) -> list[str]:
        baselines[path] = installed_list_parameter_baseline(path, edit)
        return baselines[path]

    changes = resolve_probe_parameter_changes(probe, installed_baseline)
    resolved = Reform.from_dict(changes, country_id="us")
    probe_id, list_edits = probe.id, dict(probe.list_edits)
    resolved_lists = {
        path: list(changes[path][edit.period]) for path, edit in list_edits.items()
    }

    class _ResolvedListEdits(resolved):
        resolved_list_edits = resolved_lists

        def apply(self) -> None:
            # PolicyEngine-US applies a reform set more than once while it
            # builds a system (CountryTaxBenefitSystem applies it before and
            # after backdating parameters), so a list that already holds this
            # reform's resolution is re-applied as a no-op.
            for path, edit in list_edits.items():
                try:
                    applied_to = list_parameter_baseline(self.parameters, path, edit)
                except ValueError as error:
                    raise ValueError(
                        f"reform-coverage probe {probe_id!r}: {error}"
                    ) from error
                if applied_to not in (baselines[path], resolved_lists[path]):
                    raise ValueError(
                        f"reform-coverage probe {probe_id!r}: {path} was resolved "
                        f"against the baseline {baselines[path]} but the system "
                        f"it is applied to holds {applied_to} over "
                        f"{edit.period}; the resolved list would revert that "
                        "difference."
                    )
            resolved.apply(self)

    return _ResolvedListEdits


def us_reform_coverage_smoke_gate(
    *,
    simulate: SimulateFn,
    probes: Iterable[ReformCoverageProbe] | None = None,
    period: int = 2024,
    name: str = "us_reform_coverage_smoke",
) -> GateResult:
    """Score each pinned probe on the export; a ~$0 bound reform fails.

    For each probe, the budget-measure change (reform vs baseline, signed by
    ``effect_direction``) must have the declared ``expected_sign`` and magnitude
    at least ``min_abs_effect``. A zero, undersized, or wrong-signed effect means
    the reform did not bind as declared and fails the release. A probe's list
    edits resolve against the installed engine as its reform is built; one that
    no longer fits raises rather than scoring a different reform, and each
    result records the resolved lists under ``list_edits``.

    Args:
        simulate: ``simulate(None)`` builds the baseline; ``simulate(reform)``
            builds the reformed simulation. Each result answers
            ``.calculate(measure, period).sum()`` as a weighted total.
        probes: Probes to run. Defaults to the shipped manifest's probe set.
        period: Default period for probes without a probe-specific period.
        name: Gate name for the manifest.

    Returns:
        The reform-coverage smoke gate result. Passes iff every probe scores at
        its expected sign and at least its ``min_abs_effect`` in magnitude.
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
        reform = _build_reform(probe)
        reformed = simulate(reform)
        probe_period = int(probe.period if probe.period is not None else period)
        baseline_total = _weighted_total(baseline, probe.budget_measure, probe_period)
        reform_total = _weighted_total(reformed, probe.budget_measure, probe_period)
        if probe.effect_direction == "baseline_minus_reform":
            effect = baseline_total - reform_total
        else:
            effect = reform_total - baseline_total
        # "either" proves BINDING without a directional claim: the probe
        # passes when the reform moves the measure by the floor in either
        # direction. For a signed, two-channel input (measured ASEC leg plus
        # donor-pinned PUF leg, e.g. farm_operations_income) the aggregate
        # sign is a property of the frame mix, not of coverage — pinning a
        # direction rots when the frame's honest composition changes.
        if probe.expected_sign == "either":
            signed_magnitude = abs(effect)
        else:
            signed_magnitude = effect if probe.expected_sign == "positive" else -effect
        passed = signed_magnitude >= probe.min_abs_effect
        results[probe.id] = {
            "name": probe.name,
            "period": probe_period,
            "budget_measure": probe.budget_measure,
            "baseline_total": baseline_total,
            "reform_total": reform_total,
            "effect": effect,
            "expected_sign": probe.expected_sign,
            "min_abs_effect": probe.min_abs_effect,
            "binding_inputs": list(probe.binding_inputs),
            "issue": probe.issue,
            "passed": passed,
        }
        if probe.list_edits:
            # Record the list each edit resolved to, so the smoke evidence
            # names exactly what was scored against the installed engine.
            resolved_lists = getattr(reform, "resolved_list_edits", None) or {}
            results[probe.id]["list_edits"] = {
                path: {
                    "period": edit.period,
                    "remove": list(edit.remove),
                    "add": list(edit.add),
                    "resolved": (
                        None
                        if resolved_lists.get(path) is None
                        else list(resolved_lists[path])
                    ),
                }
                for path, edit in probe.list_edits.items()
            }
        if not passed:
            expectation = (
                "an effect in either direction"
                if probe.expected_sign == "either"
                else f"a {probe.expected_sign} effect"
            )
            scored = (
                f"{probe.id}: '{probe.name}' scores {effect:+,.0f} on "
                f"{probe.budget_measure} for {probe_period}; expected "
                f"{expectation} with magnitude at least "
                f"${probe.min_abs_effect:,.0f}."
            )
            # A wrong-signed effect that clears the floor proves the reform
            # binds, so the inputs are carried; what disagrees is the probe's
            # definition and the installed engine (the 2026-09-28 Route A
            # failure: parameter lists pinned before a PolicyEngine-US change).
            if abs(effect) >= probe.min_abs_effect:
                failures.append(
                    f"{scored} The reform binds, but in the opposite "
                    "direction, so the probe's definition disagrees with the "
                    "installed engine: check its parameter change against "
                    "the engine's current baseline value before suspecting "
                    f"the export. {probe.reason} ({probe.issue})"
                )
            else:
                failures.append(
                    f"{scored} The reform did not bind as declared, so its "
                    f"input leaves {list(probe.binding_inputs)} are absent or "
                    f"degenerate on the export. {probe.reason} Restore them "
                    f"({probe.issue})."
                )

    return GateResult(
        name=name,
        passed=not failures,
        failures=tuple(failures),
        details={
            "default_period": int(period),
            "probes": len(probes),
            "results": results,
        },
    )
