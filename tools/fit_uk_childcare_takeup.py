"""Fit the UK childcare take-up rates against the declared target registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.ledger_targets import (
    compile_ledger_target_references,
    restamp_reference_period,
)
from microcosm.build.stochastic_assignment import (
    clipped_normal_from_uniforms,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.dfe_spring_term import (
    DFE_AGE_AT_31_DECEMBER,
    DFE_SPRING_EARLY_LEARNING_2,
    DFE_SPRING_UNIVERSAL,
    DFE_SPRING_WORKING_PARENT,
    dfe_spring_term_status,
)
from microcosm.build.uk_runtime.frs_take_up import (
    UK_TAKE_UP_DECLARED_SEEDS,
    nest_universal_childcare,
)
from microcosm.build.uk_runtime.ledger_targets import _candidate_facts_for_reference
from microcosm.build.uk_runtime.take_up_contract import (
    UKTakeUpContract,
    load_uk_take_up_contract,
)
from microcosm.calibrate import TargetRegistry

TOOL_VERSION = 7
#: The model year the DfE rows are measured in: its spring term (January of
#: the next calendar year) is the term DfE's January census counts, and from
#: it working-parent hours reach children from 9 months.
DEFAULT_TARGET_PERIOD = 2025
#: The shares of children of DfE age 0 and 1 (age on 31 December of the model
#: year, the age DfE counts by; tool version 7) who use the hours
#: their family's working-parent claim gives them (DfE registers far fewer of
#: the youngest eligible children than of those aged 2 to 4, and fewer at 9 to
#: 11 months than at age 1).
UNDER_TWO_KEYS = {0: "childcare_age_0_usage", 1: "childcare_age_1_usage"}
#: The hours a child that uses its funded hours is given in the basis runs:
#: policyengine-uk's own default for max_free_entitlement_hours_used.
USES_HOURS = 30.0
TARGET_IDS = (
    "hmrc.tfc.government_top_up",
    "hmrc.tfc.children_with_used_accounts",
    "dfe.funded_childcare.working_parent_children_2_to_4",
    "dfe.funded_childcare.early_learning_2_year_olds",
    "dfe.funded_childcare.universal_only_children",
    "dfe.funded_childcare.working_parent_children_9_to_11_months",
    "dfe.funded_childcare.working_parent_children_age_1",
)
#: Bound in calibration beside the fitted rows, which already fix them; the
#: receipt reports their expectation at the fitted rates.
DIAGNOSTIC_IDS = (
    "dfe.funded_childcare.working_parent_children_age_2",
    "dfe.funded_childcare.working_parent_children_age_3",
    "dfe.funded_childcare.universal_nursery_children_age_3",
)
RATE_KEYS = (
    "tax_free_childcare",
    "extended_childcare",
    "targeted_childcare",
    "universal_childcare",
    *UNDER_TWO_KEYS.values(),
)
INITIAL_PARAMS = np.array([0.88, 0.812, 0.597, 0.563, 0.5, 0.8], dtype=float)
BOUNDS = ((0.0, 1.0),) * len(RATE_KEYS)
#: What each fitted rate estimates, as the receipt records it.
ESTIMANDS = {
    "tax_free_childcare": (
        "The rate at which the spine reproduces HMRC's two Tax-Free Childcare rows "
        "(children with used accounts and the government top-up) given the other "
        "childcare rates: a fit across all users that trades the two rows against "
        "each other, not a take-up estimate for families newly eligible above "
        "£100,000, which a costing that removes the limit assumes equal to it."
    ),
    "extended_childcare": (
        "The share of benefit units that claim working-parent hours, fitted to DfE's "
        "working parent registrations aged 2 to 4 at the spring term."
    ),
    "targeted_childcare": (
        "The share of qualifying benefit units that claim early learning for "
        "2-year-olds on their own, fitted to DfE's registrations at the spring term."
    ),
    "universal_childcare": (
        "The universal claim rate among benefit units that do not claim working-parent "
        "hours (nested), fitted to DfE's universal-only registrations at the spring "
        "term."
    ),
    **{
        key: (
            f"The share of children of DfE age {age} (age on 31 December of the "
            "model year) who use the hours their family's working-parent claim "
            "gives them, fitted with the other age's share to DfE's working "
            "parent registrations at 9 to 11 months and at age 1 at the spring "
            "term."
        )
        for age, key in UNDER_TWO_KEYS.items()
    },
}
AT_BOUND_TOLERANCE = 1e-4
SimulationRunner = Callable[
    [Path, np.ndarray, int, int, UKTakeUpContract], Mapping[str, float]
]


def draw_childcare_inputs(
    benunit_ids: np.ndarray,
    params: np.ndarray,
    *,
    seed: int,
    contract: UKTakeUpContract,
) -> dict[str, np.ndarray]:
    """Identity-keyed flags with the hours distribution fixed (diagnostic helper;
    the fit itself uses the expected-count objective)."""
    ids = np.asarray(benunit_ids)
    if np.asarray(params).shape != (len(RATE_KEYS),):
        raise ValueError(
            f"childcare fitter accepts exactly {len(RATE_KEYS)} take-up rates"
        )
    outputs = {}
    names = (
        "would_claim_tfc",
        "would_claim_extended_childcare",
        "would_claim_targeted_childcare",
        "would_claim_universal_childcare",
    )
    for output, rate in zip(names, np.asarray(params)[: len(names)], strict=True):
        draws = stable_identity_uniforms(ids, seed=seed, salt=output)
        if output == "would_claim_universal_childcare":
            # Nested under the extended flag, as frs_take_up draws it: the
            # universal rate is conditional on not claiming the extended hours.
            outputs[output] = nest_universal_childcare(
                outputs["would_claim_extended_childcare"], draws, float(rate)
            )
        else:
            outputs[output] = draws < float(rate)
    hours = contract.continuous_entry("maximum_extended_childcare_hours_usage")
    outputs["maximum_extended_childcare_hours_usage"] = clipped_normal_from_uniforms(
        stable_identity_uniforms(
            ids, seed=seed, salt="maximum_extended_childcare_hours_usage"
        ),
        mean=float(hours["mean"]),
        sd=float(hours["sd"]),
        lower=float(hours["lower"]),
        upper=float(hours["upper"]),
    )
    return outputs


def compile_childcare_targets(
    facts: tuple[Mapping[str, Any], ...],
    *,
    target_period: int,
    vintage_overrides: Mapping[str, int],
) -> TargetRegistry:
    """Compile the declared childcare rows (fitted and diagnostic), with
    explicit overrides."""
    declared = (*TARGET_IDS, *DIAGNOSTIC_IDS)
    references = {
        r.name: r
        for r in load_country_spec("uk").target_references
        if r.name in declared
    }
    missing = sorted(set(declared) - references.keys())
    if missing:
        raise ValueError(f"childcare target declarations are missing: {missing}")
    unknown = sorted(set(vintage_overrides) - set(declared))
    if unknown:
        raise ValueError(f"unknown childcare vintage override target ids: {unknown}")
    specs = []
    for target_id in declared:
        reference = references[target_id]
        period = vintage_overrides.get(target_id, target_period)
        restamped = restamp_reference_period(reference, period)
        compiled = compile_ledger_target_references(
            _candidate_facts_for_reference(facts, restamped), [restamped], country="uk"
        )
        if len(compiled.specs) != 1:
            raise ValueError(f"{target_id} did not compile to exactly one target")
        specs.append(compiled.specs[0])
    return TargetRegistry(specs, country="uk")


def fit_childcare_takeup(
    input_h5: Path,
    *,
    ledger_facts: Path | None = None,
    ledger_facts_sha256: str | None = None,
    target_period: int = DEFAULT_TARGET_PERIOD,
    vintage_overrides: Mapping[str, int] | None = None,
    seed: int = 0,
    maxiter: int = 50,
    eps: float = 1e-6,
    runner: SimulationRunner | None = None,
    generated_at: str | None = None,
    targets: Mapping[str, float] | None = None,
    registry_version: str | None = None,
) -> dict[str, object]:
    """Fit rates and return a deterministic, provenance-complete receipt."""
    contract = load_uk_take_up_contract()
    overrides = dict(vintage_overrides or {})
    if targets is None:
        if ledger_facts is None:
            raise ValueError("--ledger-facts is required to compile childcare targets")
        artifact = load_ledger_consumer_artifact(
            ledger_facts, expected_facts_sha256=ledger_facts_sha256
        )
        registry = compile_childcare_targets(
            artifact.facts, target_period=target_period, vintage_overrides=overrides
        )
        compiled = {spec.name: float(spec.value) for spec in registry.specs}
        targets = {name: compiled[name] for name in TARGET_IDS}
        diagnostic_targets = {name: compiled[name] for name in DIAGNOSTIC_IDS}
        registry_version = registry.version
        feed_sha256 = artifact.facts_sha256
        target_metadata = {
            spec.name: {
                "declared_value": spec.value,
                "period": spec.period,
                "family": spec.family,
                "source": spec.source,
            }
            for spec in registry.specs
        }
    else:
        if set(targets) != set(TARGET_IDS):
            raise ValueError(
                f"childcare targets must contain exactly the {len(TARGET_IDS)} "
                "fitted target ids"
            )
        diagnostic_targets = {}
        feed_sha256 = ledger_facts_sha256 or "test-injected"
        registry_version = registry_version or "test-injected"
        target_metadata = {
            name: {"declared_value": value, "period": target_period}
            for name, value in targets.items()
        }
    runner = runner or _policyengine_runner

    def objective(params: np.ndarray) -> float:
        achieved = runner(input_h5, params, seed, target_period, contract)
        return float(
            sum(
                (float(achieved[name]) / float(targets[name]) - 1) ** 2
                for name in TARGET_IDS
            )
        )

    result = minimize(
        objective,
        INITIAL_PARAMS,
        bounds=BOUNDS,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "eps": eps},
    )
    achieved = dict(runner(input_h5, result.x, seed, target_period, contract))
    params = dict(zip(RATE_KEYS, map(float, result.x), strict=True))
    at_bound = sorted(
        key
        for key, value in params.items()
        if value <= AT_BOUND_TOLERANCE or value >= 1.0 - AT_BOUND_TOLERANCE
    )

    return {
        "tool": "fit_uk_childcare_takeup",
        "tool_version": TOOL_VERSION,
        "input_h5": str(input_h5),
        "input_sha256": _sha256(input_h5),
        "seed": seed,
        "objective": "expected_count",
        "expectation_basis": "dfe_spring_term",
        "generated_at": generated_at or datetime.now(UTC).date().isoformat(),
        "model_period": target_period,
        "vintage_overrides": overrides,
        "ledger_facts_sha256": feed_sha256,
        "target_registry_version": registry_version,
        "stochastic_contract_sha256": contract.resource_sha256,
        "engine_version": _engine_version(),
        "optimizer": {
            "method": "L-BFGS-B",
            "maxiter": maxiter,
            "eps": float(eps),
            "success": bool(result.success),
            "loss": float(result.fun),
        },
        "params": params,
        "params_at_bound": at_bound,
        "estimands": dict(ESTIMANDS),
        "targets": target_metadata,
        "achieved": {
            name: {
                "value": float(achieved[name]),
                "target": float(targets[name]),
                "ratio": float(achieved[name]) / float(targets[name]),
            }
            for name in TARGET_IDS
        },
        "diagnostic_rows": {
            name: {
                "value": float(achieved[name]),
                "target": float(diagnostic_targets[name]),
                "ratio": float(achieved[name]) / float(diagnostic_targets[name]),
            }
            for name in DIAGNOSTIC_IDS
            if name in achieved and name in diagnostic_targets
        },
        "ceilings_at_rate_1": _ceilings(input_h5, target_period, targets),
    }


def _engine_version() -> str | None:
    """The installed policyengine-uk version, or None when it is absent.

    A real fit needs the engine (the basis runs import it); the receipt's
    provenance field must not make a hermetic run with an injected runner
    depend on the engine being installed.
    """

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("policyengine-uk")
    except PackageNotFoundError:
        return None


def _ceilings(
    input_h5: Path, target_period: int, targets: Mapping[str, float]
) -> dict[str, dict[str, float]] | None:
    """Each row's expectation with every rate at 1.0 (the frame's reach)."""

    basis = _BASIS_CACHE.get((str(Path(input_h5).resolve()), int(target_period)))
    if basis is None:
        return None
    ceiling = expected_childcare_counts(basis, np.ones(len(RATE_KEYS)))
    return {
        name: {"value": ceiling[name], "ratio": ceiling[name] / float(targets[name])}
        for name in TARGET_IDS
    }


FLAG_OUTPUTS = (
    "would_claim_tfc",
    "would_claim_extended_childcare",
    "would_claim_universal_childcare",
    "would_claim_targeted_childcare",
)
OWN_RATE_BY_TARGET = {
    "hmrc.tfc.government_top_up": "tax_free_childcare",
    "hmrc.tfc.children_with_used_accounts": "tax_free_childcare",
    "dfe.funded_childcare.working_parent_children_2_to_4": "extended_childcare",
    "dfe.funded_childcare.early_learning_2_year_olds": "targeted_childcare",
    "dfe.funded_childcare.universal_only_children": "universal_childcare",
    "dfe.funded_childcare.working_parent_children_9_to_11_months": "extended_childcare",
    "dfe.funded_childcare.working_parent_children_age_1": "extended_childcare",
    "dfe.funded_childcare.working_parent_children_age_2": "extended_childcare",
    "dfe.funded_childcare.working_parent_children_age_3": "extended_childcare",
    "dfe.funded_childcare.universal_nursery_children_age_3": "universal_childcare",
}


class ChildcareExpectationBasis:
    """Per-row weighted contributions with the take-up flags forced on or off.

    ``with_extended`` holds each target's weighted per-row values when every
    flag is True; ``without_extended`` the same with the extended flag False.
    The universal flag is nested under the extended one (frs_take_up), so a
    family that claims the extended hours claims the universal hours, and the
    universal rate is the rate among families that do not: the universal row's
    expected value is ``r_e * sum(with_ext) + (1 - r_e) * r_u * sum(without_ext)``
    and needs no allocated term (tool version 5).
    Since policyengine-uk 2.131 (pe-uk#2177) universal and targeted hours stay
    beside the working-parent hours, and a family that does not claim them on
    their own still receives them, allocated first from a participating
    child's working-parent hours (DfE guidance A1.11).
    ``allocated_from_extended`` holds those rows: the targeted and universal
    flags False and the extended flag True. So at rates ``r`` the targeted
    target's expected value is
    ``r_own * sum(with_ext * r_e + without_ext * (1 - r_e))
    + (1 - r_own) * r_e * sum(allocated)``, linear in the own rate and
    bilinear with the extended rate, and the extended row is
    ``r_e * sum(with_ext)``. The universal row counts universal-only children,
    so every side leaves out a child who receives extended hours; under the
    nesting a family claiming the extended hours never has the universal flag
    off, so the row has no allocated term. From tool version 6 the DfE rows
    are measured at the spring term by the age on 31 December
    (dfe_spring_term.py), and ``under_two`` holds the claiming-world sides'
    contributions from children under 2, whom the under-2 usage share ``s``
    scales (the draw applies only where the unit claims the working-parent
    hours): each ``with_ext`` and ``allocated`` sum above is
    ``sum(rest) + s * sum(under_two)``. The expectation is the
    quantity a take-up *rate*
    controls: the spine's persisted flags are one draw from it, and later
    stages clone and re-key rows, so no re-drawn realisation can sit on the
    build's own; the expectation can.
    """

    __slots__ = (
        "with_extended",
        "without_extended",
        "allocated_from_extended",
        "under_two",
        "rows",
    )

    def __init__(
        self,
        *,
        with_extended: Mapping[str, np.ndarray],
        without_extended: Mapping[str, np.ndarray],
        allocated_from_extended: Mapping[str, np.ndarray],
        rows: int,
        under_two: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        self.with_extended = dict(with_extended)
        self.without_extended = dict(without_extended)
        self.allocated_from_extended = dict(allocated_from_extended)
        # Each side's contributions from children under 2: the part the
        # under-2 usage share scales (tool version 6). Keyed by side name
        # ("with_extended", "without_extended", "allocated_from_extended").
        self.under_two = {
            side: dict(values) for side, values in (under_two or {}).items()
        }
        self.rows = int(rows)

    def total(self, side: str, target: str, usage: Mapping[int, float]) -> float:
        """A side's sum with each age's under-2 contributions scaled by its
        usage share (``usage`` maps DfE age to share)."""

        values = getattr(self, side).get(target)
        if values is None:
            return 0.0
        total = float(np.asarray(values).sum())
        by_age = self.under_two.get(side, {}).get(target) or {}
        for age, contribution in by_age.items():
            total -= (1.0 - usage[int(age)]) * float(np.asarray(contribution).sum())
        return total


def expected_childcare_counts(
    basis: ChildcareExpectationBasis, params: np.ndarray
) -> dict[str, float]:
    """Expected weighted aggregates at ``params`` (ordered as RATE_KEYS).

    Every side's sum takes the under-2 usage share on its contributions from
    children under 2: a child that does not use its hours is counted in no
    DfE row, so each row is linear in the share (tool version 6).
    """

    values = np.asarray(params, dtype=float)
    if values.shape == (len(RATE_KEYS) - len(UNDER_TWO_KEYS),):
        # A basis without under-2 contributions needs no usage share.
        values = np.append(values, np.ones(len(UNDER_TWO_KEYS)))
    rates = dict(zip(RATE_KEYS, map(float, values), strict=True))
    extended = rates["extended_childcare"]
    usage = {age: rates[key] for age, key in UNDER_TWO_KEYS.items()}
    expected: dict[str, float] = {}
    for target, own in OWN_RATE_BY_TARGET.items():
        if target not in basis.with_extended:
            continue
        with_ext = basis.total("with_extended", target, usage)
        if own == "extended_childcare":
            expected[target] = extended * with_ext
            continue
        without_ext = basis.total("without_extended", target, usage)
        if own == "universal_childcare":
            # Nested: an extended claimant claims the universal hours, any
            # other family at the conditional universal rate.
            expected[target] = (
                extended * with_ext + (1.0 - extended) * rates[own] * without_ext
            )
            continue
        expected[target] = rates[own] * (
            with_ext * extended + without_ext * (1.0 - extended)
        )
        if target in basis.allocated_from_extended:
            allocated = basis.total("allocated_from_extended", target, usage)
            expected[target] += (1.0 - rates[own]) * extended * allocated
    return expected


def _weighted_rows(
    sim: Any, values: np.ndarray, entity: str, period: int
) -> np.ndarray:
    weights = np.asarray(sim.calculate(f"{entity}_weight", period), dtype=float)
    return np.asarray(values, dtype=float) * weights


#: The flags the allocation side turns off: their hours then come only from a
#: participating child's working-parent hours.
_ALLOCATED_FLAGS = ("would_claim_targeted_childcare", "would_claim_universal_childcare")
HOURS_OUTPUT = "maximum_extended_childcare_hours_usage"


def _anchor_hours(benunit_ids: np.ndarray, contract: UKTakeUpContract) -> np.ndarray:
    """The extended-hours usage the take-up stage draws under the contract.

    The fit runs on a spine built before its anchor moved, so the stored column
    holds the old anchor's draws; the stage's identity-keyed draw at the
    contract's distribution replaces it (microcosm#1095, #1126).
    """

    hours = contract.continuous_entry(HOURS_OUTPUT)
    return clipped_normal_from_uniforms(
        stable_identity_uniforms(
            np.asarray(benunit_ids),
            seed=UK_TAKE_UP_DECLARED_SEEDS[HOURS_OUTPUT],
            salt=HOURS_OUTPUT,
        ),
        mean=float(hours["mean"]),
        sd=float(hours["sd"]),
        lower=float(hours["lower"]),
        upper=float(hours["upper"]),
    )


def _basis_side(
    input_h5: Path,
    target_period: int,
    contract: UKTakeUpContract,
    *,
    extended: bool,
    allocated: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """One engine run's weighted rows, and their parts from children under 2.

    Every child uses its hours here (``USES_HOURS``), so the under-2 parts are
    what those children give when they do; the usage share scales them.
    """

    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=str(input_h5))
    _refuse_unplaced_birth_months(sim, target_period)
    benunit_ids = np.asarray(sim.calculate("benunit_id", target_period))
    benunit_count = benunit_ids.shape[0]
    for variable in FLAG_OUTPUTS:
        on = extended if variable == "would_claim_extended_childcare" else True
        if allocated and variable in _ALLOCATED_FLAGS:
            on = False
        sim.set_input(variable, target_period, np.full(benunit_count, on, dtype=bool))
    sim.set_input(HOURS_OUTPUT, target_period, _anchor_hours(benunit_ids, contract))
    person_count = np.asarray(sim.calculate("person_id", target_period)).shape[0]
    sim.set_input(
        "tax_free_childcare_spend_routed_share",
        target_period,
        np.full(
            person_count,
            contract.rate("tax_free_childcare_spend_routed_share", target_period),
        ),
    )
    sim.set_input(
        "max_free_entitlement_hours_used",
        target_period,
        np.full(person_count, USES_HOURS),
    )
    variables = sim.tax_benefit_system.variables

    def rows(variable: str, values: np.ndarray | None = None) -> np.ndarray:
        entity = variables[variable].entity.key
        raw = (
            np.asarray(sim.calculate(variable, target_period))
            if values is None
            else values
        )
        return _weighted_rows(sim, raw, entity, target_period)

    status = dfe_spring_term_status(
        _person_calculator(sim, target_period), target_period
    )
    dfe_age = status[DFE_AGE_AT_31_DECEMBER]
    working_parent = status[DFE_SPRING_WORKING_PARENT] > 0.5
    universal = status[DFE_SPRING_UNIVERSAL] > 0.5
    early_learning_2 = status[DFE_SPRING_EARLY_LEARNING_2] > 0.5
    in_england = _person_calculator(sim, target_period)("country") == "ENGLAND"

    def people(mask: np.ndarray) -> np.ndarray:
        return _weighted_rows(sim, mask & in_england, "person", target_period)

    # DfE's January census at the spring term, by the age on 31 December
    # (dfe_spring_term.py), as the calibration bindings read it.
    result = {
        "hmrc.tfc.government_top_up": rows("tax_free_childcare"),
        "hmrc.tfc.children_with_used_accounts": rows(
            "is_child_receiving_tax_free_childcare"
        ),
        "dfe.funded_childcare.working_parent_children_2_to_4": people(
            working_parent & (dfe_age >= 2)
        ),
        "dfe.funded_childcare.early_learning_2_year_olds": people(early_learning_2),
        # Universal-only: a working-parent child is a universal child too, and
        # the row binds DfE's universal count less the working-parent one.
        "dfe.funded_childcare.universal_only_children": people(
            universal & ~working_parent
        ),
        "dfe.funded_childcare.working_parent_children_9_to_11_months": people(
            working_parent & (dfe_age == 0)
        ),
        "dfe.funded_childcare.working_parent_children_age_1": people(
            working_parent & (dfe_age == 1)
        ),
        "dfe.funded_childcare.working_parent_children_age_2": people(
            working_parent & (dfe_age == 2)
        ),
        "dfe.funded_childcare.working_parent_children_age_3": people(
            working_parent & (dfe_age == 3)
        ),
        "dfe.funded_childcare.universal_nursery_children_age_3": people(
            universal & (dfe_age == 3)
        ),
    }
    person_rows = np.asarray(sim.calculate("person_id", target_period)).shape[0]
    # The draw keys each child's usage on its DfE age (frs_take_up), so its
    # contributions split by the same age.
    under_two_rows = {
        target: {age: values * (dfe_age == age) for age in UNDER_TWO_KEYS}
        for target, values in result.items()
        if np.asarray(values).shape[0] == person_rows and target.startswith("dfe.")
    }
    return result, under_two_rows


def _person_calculator(sim: Any, period: int) -> Callable[[str], np.ndarray]:
    """Engine variables at person grain: group values mapped to their members
    and enums decoded to their names, as ``dfe_spring_term_status`` reads
    them."""

    variables = sim.tax_benefit_system.variables

    def calculate(name: str) -> np.ndarray:
        if variables[name].entity.key == "person":
            values = sim.calculate(name, period)
        else:
            values = sim.calculate(name, period, map_to="person")
        for candidate in (values, getattr(values, "values", None)):
            if hasattr(candidate, "decode_to_str"):
                return np.asarray(candidate.decode_to_str()).astype(str)
        return np.asarray(getattr(values, "values", values))

    return calculate


def _refuse_unplaced_birth_months(sim: Any, period: int) -> None:
    """A spine stores each person's months since their birthday
    (birth_month.py); without it the engine places birthdays by its own rule,
    so the fit would classify DfE ages unlike the build."""

    holder = sim.get_holder("months_since_last_birthday")
    if not holder.get_known_periods():
        raise ValueError(
            "the childcare fit needs a spine that stores months_since_last_birthday "
            "(uk_runtime/birth_month.py); this input has none, so the engine would "
            "place birthdays by its own rule and classify DfE ages unlike the build."
        )


def compute_childcare_basis(
    input_h5: Path, target_period: int, contract: UKTakeUpContract
) -> ChildcareExpectationBasis:
    """Three engine runs: extended on and off with every other flag on, and
    extended on with the targeted and universal flags off (the targeted row's
    allocation side; the nested universal row needs none)."""

    with_extended, with_under_two = _basis_side(
        input_h5, target_period, contract, extended=True
    )
    # The usage draw applies only where the unit claims the working-parent
    # hours, so the without-extended side keeps every child's hours.
    without_extended, _ = _basis_side(input_h5, target_period, contract, extended=False)
    allocation, allocation_under_two = _basis_side(
        input_h5, target_period, contract, extended=True, allocated=True
    )
    allocated_targets = [
        target
        for target, own in OWN_RATE_BY_TARGET.items()
        if own == "targeted_childcare"
    ]
    return ChildcareExpectationBasis(
        with_extended=with_extended,
        without_extended=without_extended,
        allocated_from_extended={
            target: allocation[target] for target in allocated_targets
        },
        under_two={
            "with_extended": with_under_two,
            "allocated_from_extended": {
                target: allocation_under_two[target]
                for target in allocated_targets
                if target in allocation_under_two
            },
        },
        rows=int(with_extended["hmrc.tfc.children_with_used_accounts"].shape[0]),
    )


_BASIS_CACHE: dict[tuple[str, int], ChildcareExpectationBasis] = {}


def _policyengine_runner(
    input_h5: Path,
    params: np.ndarray,
    seed: int,
    target_period: int,
    contract: UKTakeUpContract,
) -> Mapping[str, float]:
    """Expected-count objective; ``seed`` is unused (no draw is taken)."""

    del seed
    key = (str(Path(input_h5).resolve()), int(target_period))
    basis = _BASIS_CACHE.get(key)
    if basis is None:
        basis = compute_childcare_basis(input_h5, target_period, contract)
        _BASIS_CACHE[key] = basis
    return expected_childcare_counts(basis, params)


def _parse_overrides(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        target_id, separator, period = value.partition("=")
        if not separator:
            raise ValueError("--vintage-override must be target_id=period")
        result[target_id] = int(period)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enhanced-frs-h5", type=Path, required=True)
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument("--ledger-facts-sha256", required=True)
    parser.add_argument("--target-period", type=int, default=DEFAULT_TARGET_PERIOD)
    parser.add_argument("--vintage-override", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Recorded in the receipt; the expected-count objective takes no "
            "draw (identity draws are a diagnostic helper only)."
        ),
    )
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help=(
            "Finite-difference step for L-BFGS-B on the rates. The "
            "expected-count objective is smooth (bilinear in the rates), so "
            "a small step is exact; recorded in the receipt."
        ),
    )
    args = parser.parse_args()
    receipt = fit_childcare_takeup(
        args.enhanced_frs_h5,
        ledger_facts=args.ledger_facts,
        ledger_facts_sha256=args.ledger_facts_sha256,
        target_period=args.target_period,
        vintage_overrides=_parse_overrides(args.vintage_override),
        seed=args.seed,
        maxiter=args.maxiter,
        eps=args.eps,
    )
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
