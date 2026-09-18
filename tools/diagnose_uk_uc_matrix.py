"""Compile the full current national matrix for a pinned UC development replay.

This writes private diagnostic matrices and aggregate row estimates. It does not
export calibrated microdata or issue a release certificate. Reviewed exclusions
and the complete band-edge register are applied through the production route.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from diagnose_uk_uc_relationships import (
    BU_VARIABLES,
    GB_REGIONS,
    sha256,
    variable_overlay,
)
from scipy import sparse

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.build.target_materialization import (
    materialize_target_bindings,
    resolve_target_measures,
)
from microcosm.build.uk_runtime.ledger_targets import (
    _candidate_facts_for_reference,
    _uk_baseline_flag_crosstab,
    _uk_input_substitution,
    _uk_parameter_gated_threshold,
    compile_uk_target_registry,
    materialize_uk_ledger_targets,
)
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    compute_uc_paid_diagnostic_masks,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_calibration import (
    CalibrationFrameAdapter,
    drop_injected_measure_inputs,
    inject_measure_inputs,
)
from microcosm.build.uk_runtime.national_doctrine import uk_national_target_loss_weights
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_relationships import frs_uc_claimant_mask
from microcosm.build.uk_runtime.uc_source_periods import (
    validate_uc_source_month_coverage,
)
from microcosm.build.uk_runtime.uc_target_measurements import (
    uc_paid_comparison_contract,
)
from microcosm.calibrate import TargetRegistry, calibrate
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import WeightKind

OPTIONAL_UC_DIAGNOSTIC_VARIABLES = frozenset(
    {"uc_net_earned_income", "uc_capital_income", "uc_tariff_income"}
)


def uc_diagnostic_variables(resolver):
    """Expose actual engine components without equating differently named flows."""
    requested = dict.fromkeys(
        (
            *BU_VARIABLES,
            "uc_tariff_income",
            "uc_calibration_administrative_family_type",
            "uc_calibration_child_entitlement",
        )
    )
    available, unavailable = [], []
    for variable in requested:
        if (
            variable.startswith("uc_calibration_")
            or variable in resolver.simulation.tax_benefit_system.variables
        ):
            available.append(variable)
        elif variable in OPTIONAL_UC_DIAGNOSTIC_VARIABLES:
            unavailable.append(variable)
        else:
            raise ValueError(
                f"Required UC diagnostic variable is unavailable: {variable}"
            )
    return available, unavailable


DIAGNOSTIC_CODE_MODULES = (
    "microcosm.build.ledger_targets",
    "microcosm.build.target_materialization",
    "microcosm.build.uk_runtime.ledger_targets",
    "microcosm.build.uk_runtime.measure_simulation",
    "microcosm.build.uk_runtime.national_calibration",
    "microcosm.build.uk_runtime.national_doctrine",
    "microcosm.build.uk_runtime.national_frame",
    "microcosm.build.uk_runtime.uc_relationships",
    "microcosm.build.uk_runtime.uc_source_periods",
    "microcosm.build.uk_runtime.uc_target_measurements",
    "microcosm.calibrate.solve",
    "microcosm.calibrate.matrix",
    "microcosm.calibrate.target",
    "microcosm.calibrate.registry",
    "microcosm.frame.adapters.policyengine_uk",
)
COMPARISON_CONTEXT_FIELDS = (
    "replay_receipt_sha256",
    "model_year",
    "source_year",
    "model_mode",
    "model_overlay_used",
    "versions",
    "code_sha256",
    "exclusions_sha256",
    "solver_epochs",
)


def diagnostic_code_paths():
    """Explicit numerical and measurement sources held constant across arms."""
    return {
        "tool": Path(__file__),
        "replay_helper": Path(__file__).with_name("diagnose_uk_uc_relationships.py"),
        **{
            name: Path(importlib.import_module(name).__file__)
            for name in DIAGNOSTIC_CODE_MODULES
        },
    }


def capture_file_hashes(paths):
    return {name: sha256(path) for name, path in paths.items()}


def verify_file_hashes(paths, expected):
    current = capture_file_hashes(paths)
    changed = sorted(
        name
        for name in set(expected) | set(current)
        if expected.get(name) != current.get(name)
    )
    if changed:
        raise ValueError(
            f"Diagnostic input/code bytes changed during the run: {changed}."
        )


def runtime_versions():
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "policyengine-uk",
                "policyengine-core",
                "numpy",
                "pandas",
                "scipy",
                "torch",
            )
        },
    }


def validate_comparison_context(baseline, current):
    for field in COMPARISON_CONTEXT_FIELDS:
        if (
            field not in baseline
            or field not in current
            or baseline[field] != current[field]
        ):
            raise ValueError(
                f"Target-only comparison changed or lacks control {field!r}."
            )
    return {
        field: {"baseline": baseline.get(field), "current": current.get(field)}
        for field in ("facts_sha256", "manifest_sha256")
        if baseline.get(field) != current.get(field)
    }


def validate_baseline_inputs(baseline_receipt, observed_digest):
    if baseline_receipt.get("inputs_private_sha256") != observed_digest:
        raise ValueError(
            "Comparison baseline inputs differ from their completed receipt."
        )


def source_support_metrics(row, weights, source_ids):
    """Measure independent support after grouping all clones of each source.

    Concentration uses absolute source contributions so signed target rows have
    a defined diagnostic too. UC count rows have nonnegative contributions.
    """
    row = sparse.csr_matrix(row).copy()
    row.eliminate_zeros()
    if row.shape != (1, len(weights)) or len(source_ids) != len(weights):
        raise ValueError("Source support inputs must share the household order.")
    if pd.isna(source_ids).any():
        raise ValueError(
            "Source support requires complete source household identities."
        )
    columns = row.indices
    grouped = (
        pd.Series(row.data * weights[columns])
        .groupby(np.asarray(source_ids)[columns])
        .sum()
        .abs()
        .to_numpy()
    )
    mass = float(grouped.sum())
    square_sum = float(grouped @ grouped)
    largest = float(grouped.max()) if len(grouped) else 0.0
    return {
        "household_support": int(row.nnz),
        "original_source_household_support": int(
            len(np.unique(np.asarray(source_ids)[columns]))
        ),
        "source_contribution_ess": mass**2 / square_sum if square_sum else 0.0,
        "largest_source_contribution_share": largest / mass if mass else 0.0,
        "largest_source_absolute_contribution": largest,
    }


def matrix_support_metrics(matrix, weights, source_ids):
    """Accept the sparse array returned by the production matrix builder."""
    return [
        source_support_metrics(matrix[index : index + 1], weights, source_ids)
        for index in range(matrix.shape[0])
    ]


def write_uc_support_diagnostics(frame, resolver, weight_vectors, output_dir):
    """Evaluate distinct paid-claim crosses and original-source support in GB."""
    person, benunit, household = [
        frame.table(name) for name in ("person", "benunit", "household")
    ]
    nesting = person[["person_benunit_id", "person_household_id"]].drop_duplicates()
    if nesting["person_benunit_id"].duplicated().any():
        raise ValueError("Diagnostic benefit units must nest in one household.")
    ids = benunit["benunit_id"].map(
        nesting.set_index("person_benunit_id")["person_household_id"]
    )
    columns = pd.Index(household["household_id"]).get_indexer(ids)
    if (columns < 0).any():
        raise ValueError("Diagnostic benefit units lack household membership.")
    regions = resolver.compute("household", "region")[0]
    gb = np.isin(regions[columns], list(GB_REGIONS))
    masks = compute_uc_paid_diagnostic_masks(frame, resolver.simulation, resolver.year)
    diagnostics = []
    private = benunit[["benunit_id", "benunit_source_id"]].copy()
    private["household_id"] = ids.to_numpy()
    private["household_source_id"] = household["household_source_id"].to_numpy()[
        columns
    ]
    private["great_britain"] = gb
    for label, weights in weight_vectors.items():
        private[f"weight_{label}"] = weights[columns]
    for name, mask in masks.items():
        selected = mask & gb
        private[f"diagnostic.{name}"] = selected
        hh_counts = np.bincount(columns[selected], minlength=len(household))
        item = {
            "name": name,
            "geography": "GB",
            "benefit_unit_support": int(selected.sum()),
            "original_source_benefit_unit_support": int(
                private.loc[selected, "benunit_source_id"].nunique()
            ),
        }
        item.update(
            {
                label: float(hh_counts @ weights)
                for label, weights in weight_vectors.items()
            }
        )
        item.update(
            source_support_metrics(
                sparse.csr_matrix(hh_counts[None, :]),
                weight_vectors["calibrated"],
                household["household_source_id"].to_numpy(),
            )
        )
        diagnostics.append(item)
    variables, unavailable = uc_diagnostic_variables(resolver)
    for variable in variables:
        private[variable] = resolver.compute("benunit", variable)[0]
    private.to_pickle(output_dir / "benefit_units_private.pkl")
    (output_dir / "uc_joint_diagnostics.json").write_text(
        json.dumps(
            {
                "contract": uc_paid_comparison_contract(resolver.year),
                "rows": diagnostics,
                "unavailable_optional_components": unavailable,
                "available_component_variables": variables,
            },
            indent=2,
        )
        + "\n"
    )
    protected = []
    for variable in (
        "income_tax",
        "national_insurance",
        "state_pension",
        "child_benefit",
        "housing_benefit",
        "employment_income",
    ):
        values = resolver.compute("household", variable)[0]
        protected.append(
            {
                "name": variable,
                "geography": "UK",
                **{
                    label: float(values @ weights)
                    for label, weights in weight_vectors.items()
                },
            }
        )
    (output_dir / "protected_outcomes.json").write_text(
        json.dumps(protected, indent=2) + "\n"
    )


def validate_comparison_controls(
    baseline,
    *,
    household_ids,
    initial_weights,
    names,
    target_values,
    household_source_ids,
    target_loss_weights,
):
    """Refuse population drift and enumerate a deliberately changed contract."""
    if not np.array_equal(baseline["household_ids"], household_ids):
        raise ValueError("Target-only comparison changed household identities/order.")
    if not np.array_equal(baseline["initial_weights"], initial_weights):
        raise ValueError("Target-only comparison changed prior weights.")
    if pd.isna(household_source_ids).any() or not np.array_equal(
        baseline["household_source_ids"], household_source_ids
    ):
        raise ValueError("Target-only comparison changed source household identities.")
    if not np.array_equal(baseline["target_loss_weights"], target_loss_weights):
        raise ValueError("Target-only comparison changed target loss coefficients.")
    old_names = baseline["names"].astype(str).tolist()
    names = list(names)
    if len(set(old_names)) != len(old_names) or len(set(names)) != len(names):
        raise ValueError("Target comparison names must be unique.")
    if old_names != names:
        raise ValueError("Target-only comparison changed target names/order.")
    old = dict(zip(old_names, baseline["target_values"], strict=True))
    new = dict(zip(names, target_values, strict=True))
    changed = [
        {"name": name, "old": float(old[name]), "new": float(new[name])}
        for name in sorted(old.keys() & new.keys())
        if old[name] != new[name]
    ]
    return {
        "same_household_order_and_priors": True,
        "target_values_and_roster_unchanged": old == new and old_names == names,
        "added_rows": sorted(new.keys() - old.keys()),
        "removed_rows": sorted(old.keys() - new.keys()),
        "changed_target_values": changed,
    }


def compile_explicit_references(facts, path, year):
    """Use the production reference compiler/coverage checks on a frozen roster."""
    payload = json.loads(path.read_text())
    fact_rows = tuple(facts)
    specs = []
    for row in payload["target_references"]:
        reference = LedgerTargetReference(**{**row, "period": year})
        candidates = _candidate_facts_for_reference(fact_rows, reference)
        registry = compile_ledger_target_references(
            candidates, [reference], country="uk"
        )
        registry = validate_uc_source_month_coverage(reference, registry, candidates)
        specs.extend(registry.specs)
    return TargetRegistry(specs, country="uk")


def validate_inherited_contract(inherited, names, values, *, allowed_removals):
    """Allow declared exclusions since an older capture, never value drift."""
    old = dict(
        zip(inherited["names"].astype(str), inherited["target_values"], strict=True)
    )
    new = dict(zip(names, values, strict=True))
    removed = set(old) - set(new)
    if set(new) - set(old) or any(
        name.rsplit("@", 1)[0] not in allowed_removals for name in removed
    ):
        raise ValueError("Released baseline has an unreviewed roster change.")
    if any(old[name] != value for name, value in new.items()):
        raise ValueError("Released baseline changed inherited target values.")
    if list(new) != [name for name in old if name in new]:
        raise ValueError("Released baseline changed inherited target order.")
    return {
        "removed_reviewed_rows": sorted(removed),
        "surviving_values_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--facts-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument(
        "--model-mode",
        choices=("retained_overlay", "released"),
        default="retained_overlay",
    )
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--comparison-baseline", type=Path)
    parser.add_argument("--comparison-label")
    parser.add_argument("--exclusions", type=Path)
    args = parser.parse_args()
    if bool(args.contract) != bool(args.references):
        parser.error("--contract and --references must be supplied together")
    if args.comparison_baseline and not args.comparison_label:
        parser.error("A changed-contract comparison needs --comparison-label")
    started = time.monotonic()
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    package_dir = (
        Path(
            importlib.import_module(
                "microcosm.build.uk_runtime.measure_simulation"
            ).__file__
        ).parent.parent
        / "uk"
    )
    run_paths = {
        "replay_receipt": args.replay_dir / "aggregate.json",
        "retained_awards": args.replay_dir / "benefit_units_private.pkl",
        "contract": args.contract or package_dir / "uk_population_targets.json",
        "references": args.references or package_dir / "target_references.json",
        "exclusions": args.exclusions
        or package_dir / "calibration_measure_exclusions.json",
    }
    if args.comparison_baseline:
        run_paths.update(
            {
                "baseline_receipt": args.comparison_baseline / "receipt.json",
                "baseline_inputs": args.comparison_baseline / "inputs_private.npz",
            }
        )
    code_paths = diagnostic_code_paths()
    run_hashes = capture_file_hashes(run_paths)
    code_hashes = capture_file_hashes(code_paths)
    versions = runtime_versions()
    receipt = json.loads((args.replay_dir / "aggregate.json").read_text())
    control_context = {
        "replay_receipt_sha256": run_hashes["replay_receipt"],
        "source_year": receipt["source_year"],
        "model_year": receipt["model_year"],
        "model_mode": args.model_mode,
        "model_overlay_used": receipt["model_overlay"]
        if args.model_mode == "retained_overlay"
        else [],
        "versions": versions,
        "code_sha256": code_hashes,
        "facts_sha256": args.facts_sha256,
        "manifest_sha256": args.manifest_sha256,
        "exclusions_sha256": run_hashes["exclusions"],
        "solver_epochs": args.epochs,
    }
    baseline_receipt = None
    source_artifact_changes = {}
    if args.comparison_baseline:
        baseline_receipt = json.loads(run_paths["baseline_receipt"].read_text())
        source_artifact_changes = validate_comparison_context(
            baseline_receipt, control_context
        )
        validate_baseline_inputs(baseline_receipt, run_hashes["baseline_inputs"])
    inputs = {Path(k): v for k, v in receipt["input_sha256"].items()}
    for path, digest in inputs.items():
        assert sha256(path) == digest, path
    paths = {p.name: p for p in inputs}
    person, benunit, household = [
        pd.read_pickle(paths[f"{name}.pkl"])
        for name in ("person", "benunit", "household")
    ]
    for entity, table in (("household", household), ("benunit", benunit)):
        if table[f"{entity}_source_id"].isna().any():
            raise ValueError(f"Diagnostic {entity} source identities must be complete.")
    inherited = np.load(paths["solve.npz"], allow_pickle=False)
    if receipt["explicit_frs_claimant_input"]:
        person["is_uc_claimant"] = frs_uc_claimant_mask(person, benunit)
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=receipt["source_year"],
        household_weights=inherited["initial_weights"],
        weight_kind=WeightKind.IMPORTANCE,
    )
    factory = None
    if receipt["model_overlay"] and args.model_mode == "retained_overlay":
        from policyengine_uk import Microsimulation

        overlay_paths = []
        for item in receipt["model_overlay"]:
            path = Path(item["path"])
            assert sha256(path) == item["sha256"], path
            overlay_paths.append(path)
        scenario, _ = variable_overlay(overlay_paths)

        def factory(**kwargs):
            return Microsimulation(**kwargs, scenario=scenario)

    resolver = UKMeasureResolver(
        simulation_source=None,
        frame=frame,
        year=receipt["model_year"],
        scratch_dir=args.output_dir / "engine",
        microsimulation_factory=factory,
    )
    uc_diagnostic_variables(resolver)
    old = pd.read_pickle(args.replay_dir / "benefit_units_private.pkl")
    current_uc = resolver.compute("benunit", "universal_credit")[0]
    retained_awards_identical = np.array_equal(current_uc, old.universal_credit)
    if args.model_mode == "retained_overlay":
        np.testing.assert_array_equal(current_uc, old.universal_credit)
    uc_sha256 = hashlib.sha256(current_uc.tobytes()).hexdigest()
    artifact = load_ledger_consumer_artifact(
        args.ledger_dir,
        expected_facts_sha256=args.facts_sha256,
        expected_manifest_sha256=args.manifest_sha256,
    )
    contract = None
    if args.references:
        full_registry = compile_explicit_references(
            artifact.facts, args.references, receipt["model_year"]
        )
        contract = {
            target["target_id"]: target
            for target in json.loads(args.contract.read_text())["targets"]
        }
    else:
        compilation = compile_uk_target_registry(
            artifact.facts, target_period=receipt["model_year"]
        )
        assert not compilation.unsupported, compilation.unsupported
        full_registry = compilation.registry
    registry, exclusions = apply_uk_calibration_measure_exclusions(
        full_registry, load_uk_calibration_measure_exclusions(args.exclusions)
    )
    print(f"Resolving {len(registry.specs)} active rows.", flush=True)
    resolution = resolve_target_measures(
        lambda: CalibrationFrameAdapter(frame),
        registry,
        resolver,
        period=receipt["model_year"],
        contract_targets=contract,
        band_edge_registry=full_registry,
    )
    adapter = CalibrationFrameAdapter(frame)
    original_columns = {
        entity: set(table.columns) for entity, table in adapter.tables.items()
    }
    inject_measure_inputs(adapter, resolution.measure_inputs)
    if contract is None:
        materialized = materialize_uk_ledger_targets(
            adapter,
            registry,
            period=receipt["model_year"],
            band_edge_registry=full_registry,
        )
    else:
        materialized = materialize_target_bindings(
            adapter,
            registry,
            contract,
            period=receipt["model_year"],
            band_edge_registry=full_registry,
            providers={
                "parameter_gated_threshold": _uk_parameter_gated_threshold,
                "baseline_flag_crosstab": _uk_baseline_flag_crosstab,
                "input_substitution_counterfactual": _uk_input_substitution,
            },
        )
    assert not materialized.skipped, materialized.skipped
    drop_injected_measure_inputs(adapter, resolution.measure_inputs, original_columns)
    prepared = adapter.prepared_frame()
    targets = registry.to_target_set()
    problem = build_constraint_matrix(prepared, targets, weight_entity="household")
    assert not problem.skipped
    coefficients = uk_national_target_loss_weights(
        [spec.family for spec in registry.specs], rule="family_equal"
    )
    comparison = None
    inherited_control = None
    if args.comparison_baseline:
        baseline = np.load(
            args.comparison_baseline / "inputs_private.npz", allow_pickle=False
        )
        comparison = validate_comparison_controls(
            baseline,
            household_ids=household["household_id"].to_numpy(),
            initial_weights=problem.initial_weights.values,
            names=problem.names,
            target_values=problem.target_vector,
            household_source_ids=household["household_source_id"].to_numpy(),
            target_loss_weights=coefficients,
        )
        if baseline_receipt["uc_awards_sha256"] != uc_sha256:
            raise ValueError("Target-only comparison changed simulated UC awards.")
        comparison["baseline_receipt_sha256"] = run_hashes["baseline_receipt"]
        comparison["label"] = args.comparison_label
        comparison["source_artifact_changes"] = source_artifact_changes
        comparison["binding_changes"] = {
            field: {"baseline": baseline_receipt.get(field), "current": run_hashes[key]}
            for field, key in (
                ("target_contract_sha256", "contract"),
                ("target_references_sha256", "references"),
            )
            if baseline_receipt.get(field) != run_hashes[key]
        }
    elif args.model_mode == "released" and args.references and args.exclusions:
        inherited_control = validate_inherited_contract(
            inherited,
            problem.names,
            problem.target_vector,
            allowed_removals=set(exclusions),
        )
    else:
        np.testing.assert_array_equal(problem.names, inherited["names"])
        np.testing.assert_array_equal(problem.target_vector, inherited["target_values"])
    np.testing.assert_array_equal(
        problem.initial_weights.values, inherited["initial_weights"]
    )
    sparse.save_npz(args.output_dir / "matrix.npz", problem.matrix)
    np.savez_compressed(
        args.output_dir / "inputs_private.npz",
        names=np.array(problem.names),
        target_values=problem.target_vector,
        initial_weights=problem.initial_weights.values,
        inherited_weights=inherited["weights"],
        target_loss_weights=coefficients,
        household_ids=household["household_id"].to_numpy(),
        household_source_ids=household["household_source_id"].to_numpy(),
        target_families=np.array([spec.family for spec in registry.specs]),
    )
    row_estimates = pd.DataFrame(
        {
            "name": problem.names,
            "target": problem.target_vector,
            "prior": problem.matrix @ inherited["initial_weights"],
            "inherited_calibrated": problem.matrix @ inherited["weights"],
        }
    )
    run = None
    matrix_seconds = time.monotonic() - started
    diagnostic_weights = inherited["weights"]
    if args.epochs:
        import torch

        torch.set_num_threads(2)
        solve_started = time.monotonic()
        print(f"Running diagnostic Adam for {args.epochs} epochs.", flush=True)
        result = calibrate(
            prepared,
            targets,
            weight_entity="household",
            epochs=args.epochs,
            learning_rate=0.02,
            mass="free",
            mass_reason="Development UC diagnosis on current measurement contract",
            max_weight_ratio=10,
            seed=0,
            l0_lambda=0,
            target_loss_cap=10.0,
            target_loss_weights=coefficients,
        )
        np.savez_compressed(
            args.output_dir / "weights_private.npz", weights=result.weights
        )
        row_estimates["new_calibrated"] = problem.matrix @ result.weights
        diagnostic_weights = result.weights
        run = {
            "epochs": args.epochs,
            "learning_rate": 0.02,
            "seed": 0,
            "mass": "free",
            "max_weight_ratio": 10,
            "l0_lambda": 0,
            "target_loss_cap": 10.0,
            "target_weight_rule": "family_equal",
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "elapsed_seconds": time.monotonic() - solve_started,
            "torch_threads": torch.get_num_threads(),
        }
        (args.output_dir / "solver.json").write_text(json.dumps(run, indent=2) + "\n")
    support_rows = matrix_support_metrics(
        problem.matrix, diagnostic_weights, household["household_source_id"].to_numpy()
    )
    for key in support_rows[0]:
        row_estimates[key] = [row[key] for row in support_rows]
    positive = problem.matrix.copy()
    positive.data = np.maximum(positive.data, 0)
    row_estimates["cap_only_maximum"] = positive @ (10 * problem.initial_weights.values)
    source_weights = (
        pd.Series(diagnostic_weights)
        .groupby(household["household_source_id"].to_numpy())
        .sum()
        .to_numpy()
    )
    concentration = {
        "household_ess": float(
            diagnostic_weights.sum() ** 2 / (diagnostic_weights @ diagnostic_weights)
        ),
        "source_household_ess": float(
            source_weights.sum() ** 2 / (source_weights @ source_weights)
        ),
        "weight_mass": float(diagnostic_weights.sum()),
        "max_weight_ratio": float(
            np.max(diagnostic_weights / problem.initial_weights.values)
        ),
    }
    row_estimates.to_json(
        args.output_dir / "row_estimates.json", orient="records", indent=2
    )
    write_uc_support_diagnostics(
        frame,
        resolver,
        {
            "prior": problem.initial_weights.values,
            "inherited": inherited["weights"],
            "calibrated": diagnostic_weights,
        },
        args.output_dir,
    )
    for path, digest in inputs.items():
        assert sha256(path) == digest, path
    verify_file_hashes(run_paths, run_hashes)
    verify_file_hashes(code_paths, code_hashes)
    if runtime_versions() != versions:
        raise ValueError("Diagnostic runtime versions changed during the run.")
    (args.output_dir / "receipt.json").write_text(
        json.dumps(
            {
                "purpose": "Development matrix/solver diagnostic; no certified dataset export",
                **control_context,
                "tool_sha256": code_hashes["tool"],
                "run_input_sha256": run_hashes,
                "target_contract_sha256": run_hashes["contract"],
                "target_references_sha256": run_hashes["references"],
                "matrix_sha256": sha256(args.output_dir / "matrix.npz"),
                "inputs_private_sha256": sha256(args.output_dir / "inputs_private.npz"),
                "weights_private_sha256": sha256(
                    args.output_dir / "weights_private.npz"
                )
                if args.epochs
                else None,
                "compiled_rows": len(full_registry.specs),
                "active_rows": len(problem.names),
                "exclusions": exclusions,
                "target_values_and_roster_unchanged": comparison[
                    "target_values_and_roster_unchanged"
                ]
                if comparison
                else not (
                    inherited_control and inherited_control["removed_reviewed_rows"]
                ),
                "inherited_contract_control": inherited_control,
                "comparison": comparison,
                "retained_awards_identical": bool(retained_awards_identical),
                "uc_awards_sha256": uc_sha256,
                "matrix_and_measurement_seconds": matrix_seconds,
                "total_seconds": time.monotonic() - started,
                "concentration": concentration,
                "measure_resolver": resolver.receipt(),
                "solver": run,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"rows": len(problem.names), "solver": run}), flush=True)


if __name__ == "__main__":
    main()
