"""Fit four UK childcare take-up rates against the declared target registry."""

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
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.stochastic_assignment import (
    clipped_normal_from_uniforms,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.ledger_targets import _candidate_facts_for_reference
from microcosm.build.uk_runtime.take_up_contract import (
    UKTakeUpContract,
    load_uk_take_up_contract,
)
from microcosm.calibrate import TargetRegistry

TOOL_VERSION = 3
TARGET_IDS = (
    "hmrc.tfc.government_top_up",
    "hmrc.tfc.children_with_used_accounts",
    "dfe.funded_childcare.working_parent_children_2_to_4",
    "dfe.funded_childcare.early_learning_2_year_olds",
    "dfe.funded_childcare.universal_only_children",
)
RATE_KEYS = (
    "tax_free_childcare",
    "extended_childcare",
    "targeted_childcare",
    "universal_childcare",
)
INITIAL_PARAMS = np.array([0.88, 0.812, 0.597, 0.563], dtype=float)
BOUNDS = ((0.0, 1.0),) * 4
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
    if np.asarray(params).shape != (4,):
        raise ValueError("childcare fitter accepts exactly four take-up rates")
    outputs = {}
    names = (
        "would_claim_tfc",
        "would_claim_extended_childcare",
        "would_claim_targeted_childcare",
        "would_claim_universal_childcare",
    )
    for output, rate in zip(names, params, strict=True):
        outputs[output] = stable_identity_uniforms(ids, seed=seed, salt=output) < float(
            rate
        )
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
    """Compile only the five declared childcare rows, with explicit overrides."""
    references = {
        r.name: r
        for r in load_country_spec("uk").target_references
        if r.name in TARGET_IDS
    }
    missing = sorted(set(TARGET_IDS) - references.keys())
    if missing:
        raise ValueError(f"childcare target declarations are missing: {missing}")
    unknown = sorted(set(vintage_overrides) - set(TARGET_IDS))
    if unknown:
        raise ValueError(f"unknown childcare vintage override target ids: {unknown}")
    specs = []
    for target_id in TARGET_IDS:
        reference = references[target_id]
        period = vintage_overrides.get(target_id, target_period)
        restamped = LedgerTargetReference(**{**reference.__dict__, "period": period})
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
    target_period: int = 2024,
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
        targets = {spec.name: float(spec.value) for spec in registry.specs}
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
                "childcare targets must contain exactly the five target ids"
            )
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
    from importlib.metadata import version

    return {
        "tool": "fit_uk_childcare_takeup",
        "tool_version": TOOL_VERSION,
        "input_h5": str(input_h5),
        "input_sha256": _sha256(input_h5),
        "seed": seed,
        "objective": "expected_count",
        "generated_at": generated_at or datetime.now(UTC).date().isoformat(),
        "model_period": target_period,
        "vintage_overrides": overrides,
        "ledger_facts_sha256": feed_sha256,
        "target_registry_version": registry_version,
        "stochastic_contract_sha256": contract.resource_sha256,
        "engine_version": version("policyengine-uk"),
        "optimizer": {
            "method": "L-BFGS-B",
            "maxiter": maxiter,
            "eps": float(eps),
            "success": bool(result.success),
            "loss": float(result.fun),
        },
        "params": dict(zip(RATE_KEYS, map(float, result.x), strict=True)),
        "targets": target_metadata,
        "achieved": {
            name: {
                "value": float(achieved[name]),
                "target": float(targets[name]),
                "ratio": float(achieved[name]) / float(targets[name]),
            }
            for name in TARGET_IDS
        },
        "ceilings_at_rate_1": _ceilings(input_h5, target_period, targets),
    }


def _ceilings(
    input_h5: Path, target_period: int, targets: Mapping[str, float]
) -> dict[str, dict[str, float]] | None:
    """Each row's expectation with every rate at 1.0 (the frame's reach)."""

    basis = _BASIS_CACHE.get((str(Path(input_h5).resolve()), int(target_period)))
    if basis is None:
        return None
    ceiling = expected_childcare_counts(basis, np.ones(4))
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
}


class ChildcareExpectationBasis:
    """Per-row weighted contributions with the take-up flags forced on.

    ``with_extended`` holds each target's weighted per-row values when every
    flag is True; ``without_extended`` the same with the extended flag False.
    The engine's only cross-scheme interactions are its mutual exclusions
    (targeted and universal require a family that is not extended-eligible),
    so at rates ``r`` a target's expected value is
    ``r_own * sum(with_ext * r_e + without_ext * (1 - r_e))`` — linear in the
    own rate, bilinear with the extended rate — and the extended row is
    ``r_e * sum(with_ext)``. The expectation is the quantity a take-up *rate*
    controls: the spine's persisted flags are one draw from it, and later
    stages clone and re-key rows, so no re-drawn realisation can sit on the
    build's own — the expectation can.
    """

    __slots__ = ("with_extended", "without_extended", "rows")

    def __init__(
        self,
        *,
        with_extended: Mapping[str, np.ndarray],
        without_extended: Mapping[str, np.ndarray],
        rows: int,
    ) -> None:
        self.with_extended = dict(with_extended)
        self.without_extended = dict(without_extended)
        self.rows = int(rows)


def expected_childcare_counts(
    basis: ChildcareExpectationBasis, params: np.ndarray
) -> dict[str, float]:
    """Expected weighted aggregates at ``params`` (ordered as RATE_KEYS)."""

    rates = dict(
        zip(RATE_KEYS, map(float, np.asarray(params, dtype=float)), strict=True)
    )
    extended = rates["extended_childcare"]
    expected: dict[str, float] = {}
    for target, own in OWN_RATE_BY_TARGET.items():
        with_ext = basis.with_extended[target]
        if own == "extended_childcare":
            expected[target] = extended * float(with_ext.sum())
            continue
        without_ext = basis.without_extended[target]
        expected[target] = rates[own] * float(
            (with_ext * extended + without_ext * (1.0 - extended)).sum()
        )
    return expected


def _weighted_rows(
    sim: Any, values: np.ndarray, entity: str, period: int
) -> np.ndarray:
    weights = np.asarray(sim.calculate(f"{entity}_weight", period), dtype=float)
    return np.asarray(values, dtype=float) * weights


def _basis_side(
    input_h5: Path, target_period: int, contract: UKTakeUpContract, *, extended: bool
) -> dict[str, np.ndarray]:
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=str(input_h5))
    benunit_count = np.asarray(sim.calculate("benunit_id", target_period)).shape[0]
    for variable in FLAG_OUTPUTS:
        on = extended if variable == "would_claim_extended_childcare" else True
        sim.set_input(variable, target_period, np.full(benunit_count, on, dtype=bool))
    person_count = np.asarray(sim.calculate("person_id", target_period)).shape[0]
    sim.set_input(
        "tax_free_childcare_spend_routed_share",
        target_period,
        np.full(
            person_count,
            contract.rate("tax_free_childcare_spend_routed_share", target_period),
        ),
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

    age = np.asarray(sim.calculate("age", target_period))
    extended_receiving = np.asarray(
        sim.calculate("is_child_receiving_extended_childcare", target_period)
    )
    return {
        "hmrc.tfc.government_top_up": rows("tax_free_childcare"),
        "hmrc.tfc.children_with_used_accounts": rows(
            "is_child_receiving_tax_free_childcare"
        ),
        "dfe.funded_childcare.working_parent_children_2_to_4": rows(
            "is_child_receiving_extended_childcare", extended_receiving & (age >= 2)
        ),
        "dfe.funded_childcare.early_learning_2_year_olds": rows(
            "is_child_receiving_targeted_childcare"
        ),
        "dfe.funded_childcare.universal_only_children": rows(
            "is_child_receiving_universal_childcare"
        ),
    }


def compute_childcare_basis(
    input_h5: Path, target_period: int, contract: UKTakeUpContract
) -> ChildcareExpectationBasis:
    """Two engine runs (extended flag on / off, every other flag on)."""

    with_extended = _basis_side(input_h5, target_period, contract, extended=True)
    without_extended = _basis_side(input_h5, target_period, contract, extended=False)
    return ChildcareExpectationBasis(
        with_extended=with_extended,
        without_extended=without_extended,
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
    parser.add_argument("--target-period", type=int, default=2024)
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
            "Finite-difference step for L-BFGS-B on the four rates. The "
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
