"""US validation-input coverage: every scored provision's critical input leaf
is produced by a source stage, or is an allowlisted known gap.

The reform-validation configs (``obbba_reforms.json``,
``tax_expenditure_reforms.json``, ``soi_baseline_levels.json``) score policy
provisions on the calibrated dataset. A provision's simulated effect is driven
by one or more *pure-input leaves* — PolicyEngine-US variables with no formula,
whose values must come from a source stage. When a provision's critical leaf is
produced by no stage, the provision is a structural zero on the dataset and its
validation row reads as a silent zero until an external benchmark exposes it.
Two shipped instances:

- education credits (``soi_baseline_levels`` row ``soi_education_credits``)
  depend on ``qualified_tuition_expenses``; the PUF education stage now
  produces that leaf and its five AOTC factual inputs (populace#253); and
- the OBBBA auto-loan-interest deduction depends on
  ``qualified_passenger_vehicle_loan_interest``; the SCF auto-loan stage now
  derives a documented qualifying-share proxy so the provision binds
  (populace#252).

Both were invisible because nothing checked that a validation row's inputs are
actually produced. :func:`us_validation_input_coverage_gate` is that check.

Scope — curated critical leaves, not the transitive closure. The full input-leaf
closure of a broad measure such as ``income_tax`` is the whole tax base
(hundreds of leaves, the great majority legitimately defaulted: energy-credit
expenditures, clean-vehicle flags, state-specific earnings). Gating the closure
would drown the real signal behind a ~150-entry allowlist and would not have
singled out #252/#253. This registry instead names, per validation concern, the
input leaf whose *absence zeroes that provision* — the small, high-signal set
the issue's evidence table describes. :func:`assert_validation_leaf_registry_current`
proves each registered leaf against the live PolicyEngine-US graph (it must be
an input leaf, and a dependency of the provision it is registered under), so the
registry cannot silently rot as the engine evolves.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from populace.build.gates import GateResult, source_stage_input_coverage_gate
from populace.build.source_manifest import SourceManifest

__all__ = [
    "US_VALIDATION_PROVISION_INPUT_LEAVES",
    "ValidationInputLeaf",
    "assert_validation_leaf_registry_current",
    "us_source_stage_outputs",
    "us_validation_input_coverage_gate",
    "us_validation_input_leaf_requirements",
    "us_validation_input_leaf_reviewed_exclusions",
]


@dataclass(frozen=True)
class ValidationInputLeaf:
    """A provision-critical input leaf a validation row structurally needs.

    Attributes:
        leaf: The PolicyEngine-US input variable (no formula) the provision
            depends on. Its absence makes the provision a structural zero.
        provision_variables: The engine variable(s) whose computation reads
            this leaf — the measured/neutralized/level variable of the rows
            below. Used by :func:`assert_validation_leaf_registry_current` to
            prove, against the live engine graph, that the leaf really is a
            dependency of the provision (so the registry cannot rot).
        validation_rows: The validation-config row ids that score the
            provision (from ``obbba_reforms.json`` / ``tax_expenditure_reforms.json``
            / ``soi_baseline_levels.json``). Named in the gate failure so the
            build points at the affected rows.
        reason: Non-empty when this leaf is a *known, tracked* imputation gap
            allowed to be missing for now (a reviewed exclusion). Empty string
            means the leaf must be a declared source-stage output.
    """

    leaf: str
    provision_variables: tuple[str, ...]
    validation_rows: tuple[str, ...]
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.leaf:
            raise ValueError("ValidationInputLeaf.leaf is required.")
        if not self.provision_variables:
            raise ValueError(
                f"ValidationInputLeaf {self.leaf!r}: provision_variables is required "
                "so the registry can be checked against the engine graph."
            )
        if not self.validation_rows:
            raise ValueError(
                f"ValidationInputLeaf {self.leaf!r}: validation_rows is required so "
                "a failure can name the affected rows."
            )


#: The provision-critical input leaves the shipped US validation configs depend
#: on. Seeded with the two invisible-gap instances the check exists to catch;
#: extend it as new validation rows are added whose provision keys on an input
#: leaf. Each entry is proven against the live engine graph by
#: :func:`assert_validation_leaf_registry_current`.
US_VALIDATION_PROVISION_INPUT_LEAVES: tuple[ValidationInputLeaf, ...] = (
    ValidationInputLeaf(
        leaf="tip_income",
        provision_variables=("tip_income_deduction",),
        validation_rows=("obbba_no_tax_on_tips",),
    ),
    ValidationInputLeaf(
        leaf="treasury_tipped_occupation_code",
        provision_variables=("tip_income_deduction",),
        validation_rows=("obbba_no_tax_on_tips",),
    ),
    ValidationInputLeaf(
        leaf="fsla_overtime_premium",
        provision_variables=("overtime_income_deduction",),
        validation_rows=("obbba_no_tax_on_overtime",),
    ),
    ValidationInputLeaf(
        leaf="qualified_tuition_expenses",
        provision_variables=("education_tax_credits",),
        validation_rows=("soi_education_credits",),
    ),
    ValidationInputLeaf(
        leaf="traditional_401k_contributions_desired",
        provision_variables=("savers_credit",),
        validation_rows=("soi_savers_credit",),
    ),
    ValidationInputLeaf(
        leaf="roth_401k_contributions_desired",
        provision_variables=("savers_credit",),
        validation_rows=("soi_savers_credit",),
    ),
    ValidationInputLeaf(
        leaf="traditional_ira_contributions_desired",
        provision_variables=("savers_credit",),
        validation_rows=("soi_savers_credit",),
    ),
    ValidationInputLeaf(
        leaf="roth_ira_contributions_desired",
        provision_variables=("savers_credit",),
        validation_rows=("soi_savers_credit",),
    ),
    ValidationInputLeaf(
        leaf="self_employed_pension_contributions_desired",
        provision_variables=("savers_credit",),
        validation_rows=("soi_savers_credit",),
    ),
    ValidationInputLeaf(
        leaf="casualty_loss",
        provision_variables=("casualty_loss_deduction",),
        validation_rows=("obbba_casualty_loss_limit",),
    ),
    ValidationInputLeaf(
        leaf="qualified_passenger_vehicle_loan_interest",
        provision_variables=("auto_loan_interest_deduction",),
        validation_rows=("obbba_auto_loan_interest",),
    ),
)


def _source_stages_path() -> Path:
    return Path(str(files("populace.build.us").joinpath("source_stages.json")))


def us_source_stage_outputs(path: Path | None = None) -> frozenset[str]:
    """Every variable name some US source stage declares as an output.

    Reads the typed ``source_stages.json`` manifest, so a malformed stage
    declaration fails here rather than being silently skipped. This is the
    "declared" surface the coverage gate checks required leaves against.
    """
    config_path = path or _source_stages_path()
    manifest = SourceManifest.from_mapping(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    return frozenset(output for stage in manifest.stages for output in stage.outputs)


def us_validation_input_leaf_requirements() -> dict[str, list[str]]:
    """Registry as a ``leaf -> [validation row ids]`` requirement map.

    This is the ``required_leaves`` argument :func:`source_stage_input_coverage_gate`
    consumes: every registered leaf mapped to the rows that need it.
    """
    requirements: dict[str, list[str]] = {}
    for entry in US_VALIDATION_PROVISION_INPUT_LEAVES:
        requirements.setdefault(entry.leaf, []).extend(entry.validation_rows)
    return {leaf: sorted(set(rows)) for leaf, rows in requirements.items()}


def us_validation_input_leaf_reviewed_exclusions() -> dict[str, str]:
    """Registry entries that are known, tracked gaps, as ``leaf -> reason``.

    A leaf carries a reason exactly when it is an accepted missing input (an
    open imputation issue). When such a leaf later becomes a declared stage
    output, the coverage gate flags the now-stale exclusion so the register
    cannot rot.
    """
    exclusions: dict[str, str] = {}
    for entry in US_VALIDATION_PROVISION_INPUT_LEAVES:
        if entry.reason:
            exclusions[entry.leaf] = entry.reason
    return exclusions


def us_validation_input_coverage_gate(
    *,
    source_stage_outputs: Iterable[str] | None = None,
) -> GateResult:
    """Build the named US validation-input coverage release gate.

    Every provision-critical input leaf the validation configs depend on must
    be a declared source-stage output or carry a reviewed exclusion with the
    tracking issue. A new validation row whose provision keys on an un-imputed,
    un-allowlisted input leaf fails the build here instead of shipping a silent
    structural zero.

    Args:
        source_stage_outputs: The declared source-stage output surface.
            Defaults to :func:`us_source_stage_outputs` (the shipped
            ``source_stages.json``); an explicit surface is for tests and for a
            release that composes additional input-producing stages.

    Returns:
        The ``"us_validation_input_coverage"`` gate result.
    """
    declared = (
        us_source_stage_outputs()
        if source_stage_outputs is None
        else frozenset(str(output) for output in source_stage_outputs)
    )
    return source_stage_input_coverage_gate(
        us_validation_input_leaf_requirements(),
        declared_outputs=declared,
        reviewed_exclusions=us_validation_input_leaf_reviewed_exclusions(),
        name="us_validation_input_coverage",
    )


def _validation_leaf_engine() -> Any | None:
    """A PolicyEngine-US adapter for graph checks, or ``None`` if absent.

    Mirrors :func:`populace.build.us_runtime.puf_support._formula_owned_engine`:
    the adapter ships with populace-frame and imports ``policyengine_us`` lazily,
    so this succeeds without the ``[us]`` extra and a missing extra surfaces as
    an ``ImportError`` at *call* time, treated exactly like ``None`` (the
    workspace test environment). The registry consistency check is then a no-op
    off-build and runs live where the extra is installed.
    """
    try:
        from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine
    except ImportError:
        return None
    try:
        return PolicyEngineUSEngine()
    except Exception:  # pragma: no cover - defensive; construction is trivial
        return None


def _provision_input_leaves(
    provision_variable: str,
    *,
    period: int,
    tax_benefit_system: Any,
) -> set[str]:
    """Input-leaf dependency set of ``provision_variable`` via the engine tracer.

    Traces a minimal one-person simulation computing the provision and returns
    the touched variables that are engine input leaves (no formula). This is the
    engine's own dependency graph, not a source parse, so it stays correct as
    formulas change.
    """
    from policyengine_us import Simulation

    simulation = Simulation(
        situation={
            "people": {"you": {}},
            "families": {"f": {"members": ["you"]}},
            "marital_units": {"m": {"members": ["you"]}},
            "tax_units": {"t": {"members": ["you"]}},
            "spm_units": {"s": {"members": ["you"]}},
            "households": {"h": {"members": ["you"]}},
        }
    )
    simulation.trace = True
    simulation.calculate(provision_variable, period)
    touched = {key.split("<", 1)[0] for key in simulation.tracer.get_flat_trace()}
    variables = tax_benefit_system.variables
    return {
        name
        for name in touched
        if name in variables and variables[name].is_input_variable()
    }


def assert_validation_leaf_registry_current(
    *,
    engine: Any | None = None,
    period: int = 2026,
) -> None:
    """Fail if a registry entry has drifted from the PolicyEngine-US graph.

    Every :class:`ValidationInputLeaf` in
    :data:`US_VALIDATION_PROVISION_INPUT_LEAVES` must, per the live engine:

    - name a variable that exists and is a *pure input leaf* (no formula); and
    - be an input-leaf dependency of at least one of its ``provision_variables``.

    A leaf the engine has turned into a formula-owned output, or that is no
    longer read by its provision, is stale and must be removed — otherwise the
    coverage gate would guard a requirement the graph no longer has. This is the
    reverse-direction, anti-rot check the engine-derived guards carry
    (cf. :func:`populace.build.us_runtime.puf_support.assert_formula_owned_blocklist_current`).

    A no-op when no engine is available or the adapter's lazy ``policyengine_us``
    import is missing at call time (the workspace test environment), so it runs
    only where the ``[us]`` extra is installed — the build.

    Raises:
        ValueError: Naming any registry entry that is not a current provision
            input leaf, and why.
    """
    if engine is None:
        engine = _validation_leaf_engine()
    if engine is None:
        return
    try:
        from policyengine_us import CountryTaxBenefitSystem
    except ImportError:
        # Missing [us] extra at call time: same no-op as having no engine.
        return

    tax_benefit_system = CountryTaxBenefitSystem()
    variables = tax_benefit_system.variables
    failures: list[str] = []
    leaves_by_provision: dict[str, set[str]] = {}
    for entry in US_VALIDATION_PROVISION_INPUT_LEAVES:
        variable = variables.get(entry.leaf)
        if variable is None:
            failures.append(
                f"{entry.leaf}: registered as a validation input leaf but is "
                "unknown to PolicyEngine-US; remove or rename it."
            )
            continue
        if not variable.is_input_variable():
            failures.append(
                f"{entry.leaf}: registered as a pure input leaf but PolicyEngine-US "
                "now computes it with a formula; it is no longer a dataset input."
            )
            continue
        depended_on = False
        for provision in entry.provision_variables:
            if provision not in leaves_by_provision:
                leaves_by_provision[provision] = _provision_input_leaves(
                    provision,
                    period=period,
                    tax_benefit_system=tax_benefit_system,
                )
            if entry.leaf in leaves_by_provision[provision]:
                depended_on = True
                break
        if not depended_on:
            failures.append(
                f"{entry.leaf}: registered under provision(s) "
                f"{list(entry.provision_variables)} but the engine graph shows it "
                "is not an input-leaf dependency of any of them at period "
                f"{period}; the registry entry is stale."
            )

    if failures:
        raise ValueError(
            "US validation-input leaf registry has drifted from PolicyEngine-US:\n"
            + "\n".join(f"  - {line}" for line in failures)
        )
