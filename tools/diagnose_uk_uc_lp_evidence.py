"""Reproduce the three UC LP evidence families from authenticated saved matrices.

This development tool runs no population engine or calibration optimizer. Its
zero-objective LPs export aggregate diagnostics only, never witness weights or
source IDs. A feasible witness is not a release candidate or an optimized fit.
Historical evidence is verified, not rewritten, with --expected-evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linprog

_HELPER_PATH = Path(__file__).with_name("diagnose_uk_uc_support.py")
_SPEC = importlib.util.spec_from_file_location("uc_lp_support_helper", _HELPER_PATH)
support = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(support)

ADDITIONS = [
    "ons.household_composition.lone_parent_dependent_children_households",
    "dwp.benefit_cap.capped_households",
    "scotgov.scottish_child_payment_spending",
    "dfe.funded_childcare.universal_only_children",
    "dfe.funded_childcare.early_learning_2_year_olds",
]
LONE_PARENT = "dwp.uc.households_single_with_children@2025"
CGT = "obr.capital_gains_tax@2025"
RETAINED_CGT = ["hmrc.cgt.taxpayers_total@2025", "hmrc.cgt.gains_total@2025"]
ZERO_ROWS = sorted(name + "@2025" for name in support.KNOWN_ZERO_ROWS)
PROFILE_KEYS = {
    "named-opposition": "fresh_named_opposition_lp_probes",
    "full-supported": "fresh_full_supported_lp_probes",
    "bounded-redistribution": "fresh_bounded_redistribution_lp_probes",
}
# Only descriptive prose and historical production identity are outside replay
# equality. Every operational field and deterministic number must be reproduced.
HISTORICAL_ONLY = {
    "purpose",
    "limitations",
    "limits",
    "aggregate_receipt_sha256",
    "probe_script_sha256",
    "protected_outcome_basis",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_arrays(run):
    matrix = run["matrix"]
    vector_keys = (
        "names",
        "targets",
        "prior",
        "current",
        "source_ids",
        "household_ids",
        "families",
        "coefficients",
    )
    require(
        all(run[key].ndim == 1 for key in vector_keys),
        "Saved arrays must be one-dimensional.",
    )
    rows, columns = matrix.shape
    require(
        all(
            len(run[k]) == rows
            for k in ("names", "targets", "families", "coefficients")
        ),
        "Saved row arrays are not aligned.",
    )
    require(
        all(
            len(run[k]) == columns
            for k in ("prior", "current", "source_ids", "household_ids")
        ),
        "Saved household arrays are not aligned.",
    )
    for key in ("source_ids", "household_ids"):
        values = run[key]
        require(
            values.dtype.kind in "iu" and (values >= 0).all(),
            "Saved identities must be nonnegative integers.",
        )
    require(
        len(set(run["household_ids"])) == columns,
        "Household identities must be unique.",
    )
    require(len(set(run["names"])) == rows, "Matrix row names must be unique.")
    support.row_lookup(run["names"])
    require(
        all(str(name).endswith("@2025") for name in run["names"]),
        "These evidence profiles require exact 2025 rows.",
    )
    require(
        all(
            np.isfinite(run[k]).all()
            for k in ("targets", "prior", "current", "coefficients")
        )
        and np.isfinite(matrix.data).all(),
        "Saved numerical arrays must be finite.",
    )
    require(
        all(
            (run[k] >= 0).all() for k in ("targets", "prior", "current", "coefficients")
        ),
        "Saved targets and weights must be nonnegative.",
    )
    require(
        run["coefficients"].sum() > 0 and run["current"].sum() > 0,
        "Saved coefficient and current weight mass must be positive.",
    )
    require(
        np.all(run["current"] <= 10 * run["prior"] + 1e-8),
        "Current weights exceed the prior cap.",
    )
    require(
        np.all(run["current"][run["prior"] == 0] == 0),
        "Zero-prior households cannot have positive current weights.",
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()


def load_run(run_dir):
    run_dir = Path(run_dir)
    paths = {
        name: run_dir / name
        for name in (
            "receipt.json",
            "matrix.npz",
            "inputs_private.npz",
            "weights_private.npz",
        )
    }
    receipt = json.loads(paths["receipt.json"].read_text())
    require(isinstance(receipt, dict), "Completed receipt must be a JSON object.")
    hashes = {name: support.digest(path) for name, path in paths.items()}
    for filename, field in (
        ("matrix.npz", "matrix_sha256"),
        ("inputs_private.npz", "inputs_private_sha256"),
        ("weights_private.npz", "weights_private_sha256"),
    ):
        require(
            receipt.get(field) == hashes[filename],
            f"Saved {filename} does not match its receipt.",
        )
    require(
        receipt.get("model_year") == 2025,
        "These evidence profiles require model year 2025.",
    )
    fields = {
        "names": "names",
        "targets": "target_values",
        "prior": "initial_weights",
        "source_ids": "household_source_ids",
        "household_ids": "household_ids",
        "families": "target_families",
        "coefficients": "target_loss_weights",
    }
    with np.load(paths["inputs_private.npz"], allow_pickle=False) as archive:
        require(
            set(fields.values()) <= set(archive.files),
            "Saved inputs lack required arrays.",
        )
        run = {key: archive[field] for key, field in fields.items()}
    with np.load(paths["weights_private.npz"], allow_pickle=False) as archive:
        require("weights" in archive.files, "Saved weights lack the weights array.")
        run["current"] = archive["weights"]
    run["matrix"] = sparse.csr_matrix(sparse.load_npz(paths["matrix.npz"]))
    validate_arrays(run)
    run.update(paths=paths, hashes=hashes)
    return run


def checked_zero_rows(run):
    zero = run["names"][np.diff(run["matrix"].indptr) == 0].tolist()
    require(
        sorted(zero) == ZERO_ROWS,
        "Unexpected or missing zero rows; no additional rows may be silently removed.",
    )
    require(
        all(run["targets"][list(run["names"]).index(name)] > 0 for name in zero),
        "Known zero rows must have positive targets.",
    )
    return zero


def exact_indices(run, names):
    lookup = {str(name): index for index, name in enumerate(run["names"])}
    require(len(names) == len(set(names)), "Requested rows must be unique.")
    require(set(names) <= set(lookup), "Requested exact rows are missing.")
    return [lookup[name] for name in names]


def base_metadata(run, time_limit):
    return {
        "population_arm": "fresh-current-recovered",
        "population_receipt_sha256": run["hashes"]["receipt.json"],
        "matrix_sha256": run["hashes"]["matrix.npz"],
        "input_archive_sha256": run["hashes"]["inputs_private.npz"],
        "weight_archive_sha256": run["hashes"]["weights_private.npz"],
        "model_year": 2025,
        "solver": "scipy.optimize.linprog(method=highs)",
        "scipy_version": scipy.__version__,
        "objective": "Zero objective; feasibility only.",
        "per_probe_time_limit_seconds": time_limit,
        "weight_cap": "0 <= weight <= 10 * fresh prior weight",
        "ess_constraint": False,
    }


def named_opposition(run):
    zero = checked_zero_rows(run)
    uc = np.flatnonzero(np.isin(run["families"], list(support.UC_FAMILIES)))
    require(
        all(name in run["names"][uc] for name in zero),
        "Known zero rows must belong to the UC roster.",
    )
    selections = [[*support.BROAD, *ADDITIONS[:n]] for n in range(1, 6)]
    selections.append(
        [
            *[
                str(run["names"][i]).rsplit("@", 1)[0]
                for i in uc
                if run["names"][i] not in zero
            ],
            *ADDITIONS,
        ]
    )
    probes = []
    for i, selected in enumerate(selections):
        exact_indices(run, [name + "@2025" for name in selected])
        result = support.lp_probe(
            run["matrix"],
            run["targets"],
            run["prior"],
            run["names"],
            selected,
            tolerance=0.05,
            time_limit=20.0,
        )
        probe = {
            "label": f"broad10_plus_{i + 1}_named_rows"
            if i < 5
            else f"supported{len(selected) - len(ADDITIONS)}_uc_plus_five_named_rows",
            "row_count": len(selected),
            "row_names": [name + "@2025" for name in selected],
            "added_named_rows": [name + "@2025" for name in ADDITIONS[: i + 1]],
            "status": result["status"],
            "relative_tolerance": 0.05,
        }
        for key in (
            "scipy_status",
            "scipy_message",
            "normalized_constraint_violation",
            "elapsed_seconds",
            "active_weight_columns",
        ):
            if key in result:
                probe[key] = result[key]
        probes.append(probe)
    return {
        **base_metadata(run, 20.0),
        "relative_tolerance": 0.05,
        "constraint_bounds": "0.95 * target <= matrix_row @ weights <= 1.05 * target for every named row.",
        "weight_bounds": {"lower": 0, "upper_fresh_prior_multiple": 10},
        "fixed_household_mass_constraint": False,
        "full_366_row_constraint": False,
        "full_active_uc_evaluation": {
            "row_count": len(uc),
            "status": "infeasible_by_zero_support",
            "zero_support_positive_rows": zero,
            "linprog_called": False,
            "reason": "A zero matrix row cannot reach 95% of its positive target under any permitted weights.",
        },
        "probes": probes,
    }


def concentration(weights, source_ids):
    _, inverse = np.unique(source_ids, return_inverse=True)
    source_weights = np.bincount(inverse, weights=weights)

    def metrics(values):
        total = float(values.sum())
        count = len(values)
        top = max(1, math.ceil(0.01 * count))
        return {
            "entity_count": count,
            "positive_weight_count": int(np.count_nonzero(values > 0)),
            "effective_sample_size": total**2 / float(values @ values),
            "weight_mass": total,
            "maximum_weight_share": float(values.max() / total),
            "top_one_percent_entity_count": top,
            "top_one_percent_weight_share": float(
                np.partition(values, count - top)[count - top :].sum() / total
            ),
        }

    return {
        "households": metrics(weights),
        "original_source_households": metrics(source_weights),
        "share_basis": "Shares are of total household weight; source weights sum all descendant household weights. The top 1% count is ceil(0.01 * all entity count), including zero-weight entities in the count.",
    }


def mixed_probe(run, selected_names, tolerances, *, multipliers=None, time_limit=30.0):
    """Preserve the original sparse construction; aggregate a witness in memory."""
    indices = exact_indices(run, selected_names)
    tolerance = np.asarray(tolerances, dtype=float)
    require(
        tolerance.shape == (len(indices),)
        and np.isfinite(tolerance).all()
        and ((tolerance > 0) & (tolerance < 1)).all(),
        "Each row needs a finite tolerance between zero and one.",
    )
    a = run["matrix"][indices]
    b = run["targets"][indices]
    prior, current = run["prior"], run["current"]
    scales = np.maximum(np.abs(b), 1.0)
    if multipliers is None:
        columns = np.unique(a.indices)
        columns = columns[prior[columns] > 0]
    else:
        factor_low, factor_high = multipliers
        require(
            0 <= factor_low <= 1 <= factor_high and np.isfinite(multipliers).all(),
            "Redistribution bounds must contain the current weights.",
        )
        # Fixed mass constrains otherwise-unused columns too; do not prune them.
        columns = np.flatnonzero(prior > 0)
    normalized = (
        a[:, columns].multiply(prior[columns]).multiply((1.0 / scales)[:, None]).tocsr()
    )
    lower = (b - tolerance * np.abs(b)) / scales
    upper = (b + tolerance * np.abs(b)) / scales
    started = time.monotonic()
    if multipliers is None:
        solution = linprog(
            np.zeros(len(columns)),
            A_ub=sparse.vstack([normalized, -normalized], format="csr"),
            b_ub=np.concatenate([upper, -lower]),
            bounds=(0.0, 10.0),
            method="highs",
            options={"time_limit": time_limit},
        )
    else:
        bound_low = factor_low * current[columns] / prior[columns]
        bound_high = np.minimum(factor_high * current[columns] / prior[columns], 10.0)
        require(
            np.all(bound_low <= bound_high + 1e-12),
            "Empty current/prior bound intersection.",
        )
        mass = float(current.sum())
        mass_row = sparse.csr_matrix((prior[columns] / mass).reshape(1, -1))
        solution = linprog(
            np.zeros(len(columns)),
            A_ub=sparse.vstack([normalized, -normalized], format="csr"),
            b_ub=np.concatenate([upper, -lower]),
            A_eq=mass_row,
            b_eq=np.array([1.0]),
            bounds=np.column_stack([bound_low, bound_high]),
            method="highs",
            options={"time_limit": time_limit},
        )
    result = {
        "scipy_status": int(solution.status),
        "scipy_message": solution.message,
        "elapsed_seconds": time.monotonic() - started,
        "active_weight_columns": len(columns),
    }
    if multipliers is not None:
        result["current_weight_multiplier_bounds"] = list(multipliers)
    if solution.status == 0:
        evaluated = np.asarray(normalized @ solution.x)
        residual = max(
            float(np.max(lower - evaluated)), float(np.max(evaluated - upper)), 0.0
        )
        if multipliers is None:
            bound_violation = max(
                float(-solution.x.min()), float(solution.x.max() - 10.0), 0.0
            )
        else:
            bound_violation = max(
                float(np.max(bound_low - solution.x)),
                float(np.max(solution.x - bound_high)),
                0.0,
            )
        require(
            np.isfinite(solution.x).all()
            and np.isfinite(residual)
            and residual <= 1e-6
            and bound_violation <= 1e-6,
            "Successful LP failed independent row/bound verification.",
        )
        li = selected_names.index(LONE_PARENT)
        result.update(
            status="feasible_for_named_rows_and_tolerances_only",
            maximum_normalized_constraint_violation=residual,
            maximum_weight_ratio_bound_violation=bound_violation,
        )
        if multipliers is None:
            result.update(
                lone_parent_estimate=float(evaluated[li] * scales[li]),
                lone_parent_target=float(b[li]),
                lone_parent_relative_error=float(
                    evaluated[li] * scales[li] / b[li] - 1
                ),
            )
        else:
            witness = np.zeros(len(prior))
            witness[columns] = solution.x * prior[columns]
            mass_error = abs(float(witness.sum()) / mass - 1.0)
            require(mass_error <= 1e-8, "Successful LP failed fixed-mass verification.")
            actual = np.asarray(run["matrix"] @ witness)
            before = np.asarray(run["matrix"] @ current)
            full_li = list(run["names"]).index(LONE_PARENT)
            protected = [
                i
                for i, name in enumerate(run["names"])
                if name.startswith("obr.")
                or name.startswith(
                    (
                        "hmrc/employment_income_income_band_",
                        "hmrc/employment_income_count_income_band_",
                    )
                )
            ]
            result.update(
                status="feasible_for_named_bounds_and_tolerances_only",
                relative_total_mass_error=mass_error,
                lone_parent_current_estimate=float(before[full_li]),
                lone_parent_witness_estimate=float(actual[full_li]),
                lone_parent_target=float(run["targets"][full_li]),
                lone_parent_witness_relative_error=float(
                    actual[full_li] / run["targets"][full_li] - 1
                ),
                witness_concentration=concentration(witness, run["source_ids"]),
                protected_matrix_outcome_changes=[
                    {
                        "name": str(run["names"][i]),
                        "current_estimate": float(before[i]),
                        "witness_estimate": float(actual[i]),
                        "change_percent": float(100 * (actual[i] / before[i] - 1))
                        if before[i]
                        else None,
                        "target": float(run["targets"][i]),
                        "witness_target_error_percent": float(
                            100 * (actual[i] / run["targets"][i] - 1)
                        )
                        if run["targets"][i]
                        else None,
                        "constrained": str(run["names"][i]) != CGT,
                    }
                    for i in protected
                ],
            )
    elif solution.status == 2:
        result["status"] = (
            "infeasible_for_named_rows_and_tolerances_under_cap"
            if multipliers is None
            else "infeasible_for_named_bounds_and_tolerances"
        )
    elif solution.status == 1 and "time limit" in solution.message.lower():
        result["status"] = "time_limit_inconclusive"
    else:
        result["status"] = "inconclusive"
    return result


def supported_names(run):
    zero = checked_zero_rows(run)
    exact_indices(run, [LONE_PARENT, CGT, *RETAINED_CGT])
    return zero, [str(name) for name in run["names"] if name not in zero]


def full_supported(run):
    zero, supported = supported_names(run)
    non_cgt = [name for name in supported if name != CGT]
    probes = []
    profiles = [
        (f"all_supported_{len(supported)}_at_5pct", supported, 0.05),
        (f"all_supported_non_cgt_{len(non_cgt)}_at_5pct", non_cgt, 0.05),
        ("lone_parent_5pct_other_supported_non_cgt_25pct", non_cgt, 0.25),
    ]
    for label, names, default in profiles:
        tolerance = [0.05 if name == LONE_PARENT else default for name in names]
        result = mixed_probe(run, names, tolerance)
        result.update(
            label=label,
            row_count=len(names),
            row_names=names,
            row_relative_tolerances=dict(zip(names, tolerance, strict=True)),
            excluded_zero_rows=zero,
            excluded_adjudicated_row=CGT if CGT not in names else None,
        )
        probes.append(result)
    return {
        **base_metadata(run, 30.0),
        "fixed_household_mass_constraint": False,
        "all_366_row_evaluation": {
            "status": "infeasible_by_zero_support",
            "zero_support_positive_rows": zero,
            "additional_zero_rows": [],
            "linprog_called": False,
        },
        "adjudicated_cgt_row": CGT,
        "retained_cgt_rows": RETAINED_CGT,
        "probes": probes,
    }


def bounded_redistribution(run):
    zero, supported = supported_names(run)
    names = [name for name in supported if name != CGT]
    tolerance = [0.05 if name == LONE_PARENT else 0.25 for name in names]
    probes = []
    for multipliers in ((0.75, 1.25), (0.5, 1.5)):
        result = mixed_probe(run, names, tolerance, multipliers=multipliers)
        probes.append(result)
        if result["scipy_status"] != 2:
            break
    protected = probes[-1].get("protected_matrix_outcome_changes", [])
    employment = [
        row["change_percent"]
        for row in protected
        if row["name"].startswith("hmrc/employment_income")
        and row["change_percent"] is not None
    ]
    result = {
        **base_metadata(run, 30.0),
        "fixed_household_mass_constraint": True,
        "fixed_household_weight_mass": float(run["current"].sum()),
        "row_count": len(names),
        "row_names": names,
        "row_relative_tolerances": dict(zip(names, tolerance, strict=True)),
        "excluded_zero_rows": zero,
        "excluded_adjudicated_row": CGT,
        "full_366_row_evaluation": {
            "status": "infeasible_by_zero_support",
            "zero_support_positive_rows": zero,
            "additional_zero_rows": [],
        },
        "current_concentration": concentration(run["current"], run["source_ids"]),
        "probes": probes,
        "wider_redistribution_probe": {
            "current_weight_multiplier_bounds": [0.5, 1.5],
            "status": "run" if len(probes) == 2 else "not_run",
            "reason": "The tighter 0.75–1.25 probe was feasible; the wider probe was conditional on infeasibility."
            if probes[0]["scipy_status"] == 0
            else "Widening requires proved infeasibility, not a timeout or another inconclusive result.",
        },
    }
    if protected:
        result["material_tradeoffs"] = {
            "selected_changes": [
                row
                for row in protected
                if row["name"]
                in [
                    "obr.income_tax@2025",
                    "obr.ni@2025",
                    "obr.state_pension@2025",
                    "obr.child_benefit@2025",
                ]
            ],
            "employment_income_and_count_band_change_percent_range": {
                "minimum": min(employment) if employment else None,
                "maximum": max(employment) if employment else None,
                "row_count": len(employment),
            },
        }
    return result


def assert_equal(actual, expected, path="root"):
    """Exact recursive equality; only wall-clock runtime may vary."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"Replay mismatch: {path} is not an object.")
        for key, value in expected.items():
            if key == "elapsed_seconds" or (
                key == "interpretation" and path.endswith(".material_tradeoffs")
            ):
                continue
            require(key in actual, f"Replay missing field: {path}.{key}")
            assert_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"Replay list mismatch: {path}",
        )
        for i, value in enumerate(expected):
            assert_equal(actual[i], value, f"{path}[{i}]")
    else:
        require(
            actual == expected, f"Replay mismatch at {path}: {actual!r} != {expected!r}"
        )


def verify_historical(results, evidence):
    for profile, observed in results.items():
        key = PROFILE_KEYS[profile]
        require(key in evidence, f"Historical evidence lacks {key}.")
        expected = {k: v for k, v in evidence[key].items() if k not in HISTORICAL_ONLY}
        assert_equal(observed, expected, key)
    return {
        "status": "exact_deterministic_parity",
        "profiles": list(results),
        "excluded_runtime_fields": ["elapsed_seconds"],
        "historical_only_fields": sorted(HISTORICAL_ONLY),
        "historical_narrative_paths": ["material_tradeoffs.interpretation"],
    }


def produce(run_dir, output_dir, profile="all", expected_evidence=None):
    output_dir = Path(output_dir)
    require(not output_dir.exists(), "Output directory must be new.")
    require(profile == "all" or profile in PROFILE_KEYS, "Unknown LP profile.")
    run = load_run(run_dir)
    sources = {
        "producer_sha256": support.digest(__file__),
        "support_helper_sha256": support.digest(_HELPER_PATH),
    }
    expected_path = Path(expected_evidence) if expected_evidence is not None else None
    expected = (
        json.loads(expected_path.read_text()) if expected_path is not None else None
    )
    expected_hash = support.digest(expected_path) if expected_path is not None else None
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if expected is not None:
        recorded = expected["arms"]["fresh-current-recovered"]["versions"]
        require(
            all(recorded[k] == value for k, value in versions.items()),
            "Exact replay requires the historical Python/NumPy/SciPy versions.",
        )
    profiles = list(PROFILE_KEYS) if profile == "all" else [profile]
    operators = {
        "named-opposition": named_opposition,
        "full-supported": full_supported,
        "bounded-redistribution": bounded_redistribution,
    }
    results = {name: operators[name](run) for name in profiles}
    parity = (
        verify_historical(results, expected)
        if expected is not None
        else {"status": "not_requested"}
    )
    require(
        all(
            support.digest(path) == run["hashes"][name]
            for name, path in run["paths"].items()
        ),
        "Saved input changed during LP reproduction.",
    )
    require(
        sources
        == {
            "producer_sha256": support.digest(__file__),
            "support_helper_sha256": support.digest(_HELPER_PATH),
        },
        "Producer/helper changed during LP reproduction.",
    )
    if expected_path is not None:
        require(
            support.digest(expected_path) == expected_hash,
            "Historical evidence changed during reproduction.",
        )
    report = {
        "schema_version": 1,
        "profiles": results,
        "provenance": {
            **sources,
            "versions": versions,
            "profile": profile,
            "expected_evidence_sha256": expected_hash,
        },
        "verification": parity,
        "limitations": [
            "No population engine or calibration optimizer runs; these are zero-objective feasibility checks only.",
            "The two positive-target zero rows remain in full-roster evaluation; diagnostic exclusions change no production targets or gates.",
            "No ESS guarantee, global-optimality claim or release-readiness conclusion follows. Protected OBR/HMRC matrix outcomes retain their distinct definitions.",
            "No witness weights, original source IDs or private filesystem paths are exported.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "lp-evidence-reproduction.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    verification = {
        **parity,
        "result_sha256": support.digest(path),
        **report["provenance"],
    }
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False) + "\n"
    )
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=[*PROFILE_KEYS, "all"], default="all")
    parser.add_argument("--expected-evidence", type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    print(
        json.dumps(
            produce(args.run_dir, args.output_dir, args.profile, args.expected_evidence)
        )
    )


if __name__ == "__main__":
    main()
