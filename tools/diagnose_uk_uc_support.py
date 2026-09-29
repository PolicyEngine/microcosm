"""Diagnose UC support in saved, receipted calibration matrices.

Run after ``tools/diagnose_uk_uc_matrix.py`` (or a compatible matrix producer):

    uv run python tools/diagnose_uk_uc_support.py \
        --run-dir /path/to/completed-run --output-dir /path/to/new-support-report

Requires matrix.npz, inputs_private.npz, weights_private.npz and receipt.json.
The receipt must attest all three file hashes; this verifies saved input identity,
not dataset certification. No population engine or calibration optimizer is run.
LPs constrain only their named rows with nonnegative weights capped at 10 times
prior weights. Keep source_group_influence_private.json local: it contains source
identities and leave-one-source-out contributions. Aggregate reports omit IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linprog

UC_FAMILIES = {"dwp_universal_credit", "dwp_two_child_limit"}
BROAD = [
    "dwp.uc.households",
    *[
        f"dwp.uc.households_{x}"
        for x in (
            "single_no_children",
            "single_with_children",
            "couple_no_children",
            "couple_with_children",
        )
    ],
    *[f"dwp.uc.households_children_{x}" for x in ("1", "2", "3", "4", "5_or_more")],
]
FIVE_PLUS = {
    "dwp.uc.households_children_5_or_more",
    *[
        f"dwp.uc.two_child_limit.{x}"
        for x in (
            "households_5_children",
            "children_in_5_children_households",
            "households_6_plus_children",
            "children_in_6_plus_children_households",
        )
    ],
}
KNOWN_ZERO_ROWS = {
    "dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_27_600_to_28_800",
    "dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_28_800_to_30_000",
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weighted_loss(estimates, targets, coefficients, cap=10.0):
    return float(
        np.minimum(np.abs(estimates - targets) / np.maximum(np.abs(targets), 1.0), cap)
        @ coefficients
        / coefficients.sum()
    )


def row_lookup(names):
    lookup = {}
    for index, name in enumerate(names):
        base = name.rsplit("@", 1)[0]
        if base in lookup:
            raise ValueError(f"Ambiguous period for requested matrix row: {base}")
        lookup[base] = index
    return lookup


def grouped_contributions(row, weights, source_ids):
    row = sparse.csr_matrix(row).copy()
    row.sum_duplicates()
    row.eliminate_zeros()
    ids, inverse = np.unique(source_ids[row.indices], return_inverse=True)
    contributions = np.bincount(
        inverse, weights=row.data * weights[row.indices], minlength=len(ids)
    )
    return row, ids, contributions


def row_support(
    row, weights, prior, source_ids, target, tolerance=0.05, cap_ratio=10.0
):
    row, ids, contribution = grouped_contributions(row, weights, source_ids)
    mass = np.abs(contribution)
    mass_sum = float(mass.sum())
    squared = float(mass @ mass)
    upper = float(np.maximum(row.data, 0.0) @ (cap_ratio * prior[row.indices]))
    lower = float(np.minimum(row.data, 0.0) @ (cap_ratio * prior[row.indices]))
    target_low, target_high = (
        target - tolerance * abs(target),
        target + tolerance * abs(target),
    )
    return {
        "household_support": int(row.nnz),
        "original_source_household_support": len(ids),
        "positive_prior_household_support": int((prior[row.indices] > 0).sum()),
        "source_contribution_ess": mass_sum**2 / squared if squared else 0.0,
        "largest_source_contribution_share": float(mass.max() / mass_sum)
        if mass_sum
        else 0.0,
        "cap_only_minimum": lower,
        "cap_only_maximum": upper,
        "fitted_estimate": float(contribution.sum()),
        "target": float(target),
        "tolerance": tolerance,
        "row_capacity_can_intersect_tolerance": bool(
            upper >= target_low and lower <= target_high
        ),
        "zero_support_positive_target": bool(row.nnz == 0 and target_low > 0),
    }


def lp_probe(
    matrix,
    targets,
    prior,
    names,
    selected_names,
    tolerance=0.05,
    cap_ratio=10.0,
    time_limit=20.0,
):
    selected_names = list(selected_names)
    lookup = row_lookup(names)
    missing = [name for name in selected_names if name not in lookup]
    if missing:
        raise ValueError(f"LP probe requested missing rows: {missing}")
    indices = np.array([lookup[name] for name in selected_names], dtype=int)
    a = sparse.csr_matrix(matrix[indices])
    b = targets[indices]
    if (b < 0).any() or not np.isfinite(b).all():
        raise ValueError("These LP probes require finite nonnegative targets.")
    a.sum_duplicates()
    a.eliminate_zeros()
    zero = np.flatnonzero(np.diff(a.indptr) == 0)
    zero_positive = [selected_names[index] for index in zero if b[index] > 0]
    result = {
        "rows": selected_names,
        "tolerance": tolerance,
        "weight_bounds": f"0 <= weight <= {cap_ratio:g} * prior_weight",
        "protected_rows_included": False,
        "fixed_mass_constraint": False,
        "zero_support_positive_rows": zero_positive,
        "solver": "scipy.optimize.linprog(method=highs)",
        "time_limit_seconds": time_limit,
    }
    if zero_positive:
        return {
            **result,
            "status": "infeasible_by_zero_support",
            "linprog_called": False,
        }
    # Solve in weight ratios and normalize rows to keep all count constraints near one.
    active_columns = np.unique(a.indices)
    active_columns = active_columns[prior[active_columns] > 0]
    if not len(active_columns):
        return {
            **result,
            "status": "feasible" if np.all(b == 0) else "infeasible_by_zero_capacity",
            "linprog_called": False,
        }
    scales = np.maximum(np.abs(b), 1.0)
    normalized = (
        a[:, active_columns]
        .multiply(prior[active_columns])
        .multiply((1 / scales)[:, None])
        .tocsr()
    )
    upper = (b + tolerance * np.abs(b)) / scales
    lower = (b - tolerance * np.abs(b)) / scales
    started = time.monotonic()
    solution = linprog(
        np.zeros(len(active_columns)),
        A_ub=sparse.vstack([normalized, -normalized], format="csr"),
        b_ub=np.concatenate([upper, -lower]),
        bounds=(0.0, cap_ratio),
        method="highs",
        options={"time_limit": time_limit},
    )
    result.update(
        linprog_called=True,
        scipy_status=int(solution.status),
        scipy_message=solution.message,
        elapsed_seconds=time.monotonic() - started,
        active_weight_columns=len(active_columns),
    )
    if solution.status == 0:
        estimates = np.asarray(normalized @ solution.x) * scales
        residual = max(
            float(np.max(lower - normalized @ solution.x)),
            float(np.max(normalized @ solution.x - upper)),
            0.0,
        )
        if (
            residual > 1e-6
            or solution.x.min() < -1e-6
            or solution.x.max() > cap_ratio + 1e-6
        ):
            raise ValueError(
                "Successful LP failed independent tolerance/bound verification."
            )
        result.update(
            status="feasible_for_named_rows_only",
            normalized_constraint_violation=residual,
            estimates={
                name: float(value)
                for name, value in zip(selected_names, estimates, strict=True)
            },
        )
    elif solution.status == 2:
        result["status"] = "infeasible_for_named_rows_under_cap"
    else:
        result["status"] = "inconclusive"
    return result


def supported_uc_probe(matrix, targets, prior, names, rows, tolerance=0.05):
    excluded = [row["name"] for row in rows if row["zero_support_positive_target"]]
    unexpected = [
        name for name in excluded if name.rsplit("@", 1)[0] not in KNOWN_ZERO_ROWS
    ]
    if unexpected:
        return {
            "status": "not_run_unexpected_zero_support",
            "unexpected_zero_rows": unexpected,
            "excluded_zero_rows": excluded,
            "full_uc_probe": False,
        }
    selected = [
        row["name"].rsplit("@", 1)[0] for row in rows if row["name"] not in excluded
    ]
    result = lp_probe(matrix, targets, prior, names, selected, tolerance)
    result.update(
        excluded_zero_rows=excluded,
        full_uc_probe=False,
        interpretation="Diagnostic supported-row subset only. The full active UC roster remains infeasible when the named positive-target zero-support rows are included.",
    )
    return result


def protected_subset_probe(matrix, targets, prior, names, tolerance=0.05):
    obr = ["obr.income_tax", "obr.ni", "obr.state_pension", "obr.child_benefit"]
    employment = [
        str(name).rsplit("@", 1)[0]
        for name in names
        if str(name).startswith(
            (
                "hmrc/employment_income_income_band_",
                "hmrc/employment_income_count_income_band_",
            )
        )
    ]
    if not employment:
        raise ValueError(
            "The protected diagnostic requires the named HMRC employment band rows."
        )
    result = lp_probe(
        matrix, targets, prior, names, [*BROAD, *obr, *employment], tolerance
    )
    result.update(
        protected_rows_included=True,
        protected_rows=[*obr, *employment],
        employment_scope="Explicit saved employment income/count band rows; no employment total or new aggregate target is invented.",
        interpretation="Only the named broad UC, OBR and HMRC employment rows are constrained. This excludes the remaining UC and protected calibration rows.",
    )
    return result


def signed_contrast(matrix, targets, names, fitted, prior, source_ids, tolerance=0.05):
    component_names = [
        "dwp.uc.households_children_5_or_more",
        "dwp.uc.two_child_limit.households_5_children",
        "dwp.uc.two_child_limit.households_6_plus_children",
    ]
    lookup = row_lookup(names)
    if any(name not in lookup for name in component_names):
        raise ValueError(
            "Reported-5+ minus TCL5/6+ contrast is missing a declared component row."
        )
    indices = [lookup[name] for name in component_names]
    row = sparse.csr_matrix(
        matrix[indices[0] : indices[0] + 1]
        - matrix[indices[1] : indices[1] + 1]
        - matrix[indices[2] : indices[2] + 1]
    )
    row.sum_duplicates()
    row.eliminate_zeros()
    signs = np.array([1.0, -1.0, -1.0])
    b = targets[indices]
    target_difference = float(signs @ b)
    interval_low = target_difference - tolerance * float(np.abs(b).sum())
    interval_high = target_difference + tolerance * float(np.abs(b).sum())
    support = row_support(row, fitted, prior, source_ids, target_difference, tolerance)
    for key in (
        "target",
        "tolerance",
        "row_capacity_can_intersect_tolerance",
        "zero_support_positive_target",
    ):
        support.pop(key)
    _, ids, contributions = grouped_contributions(row, fitted, source_ids)
    private = []
    for source, contribution in zip(ids, contributions, strict=True):
        columns = np.flatnonzero(source_ids == source)
        removed = row[:, columns]
        cap_weights = 10 * prior[columns][removed.indices]
        removed_upper = float(np.maximum(removed.data, 0.0) @ cap_weights)
        removed_lower = float(np.minimum(removed.data, 0.0) @ cap_weights)
        lower = support["cap_only_minimum"] - removed_lower
        upper = support["cap_only_maximum"] - removed_upper
        private.append(
            {
                "kind": "computational_contrast",
                "source_household_id": int(source),
                "descendant_household_count": len(columns),
                "no_refit_contrast_contribution_loss": float(contribution),
                "no_refit_contrast_estimate": support["fitted_estimate"]
                - float(contribution),
                "remaining_cap_only_minimum": lower,
                "remaining_cap_only_maximum": upper,
                "remaining_capacity_can_intersect_component_implied_interval": bool(
                    upper >= interval_low and lower <= interval_high
                ),
            }
        )
    return {
        "label": "reported_5plus_minus_TCL_5_minus_TCL_6plus",
        "interpretation": "Computational contrast between different concepts/source windows, not a source joint, statistical identity or new target.",
        "components": [
            {
                "name": str(names[index]),
                "coefficient": float(sign),
                "optimizer_target_value": float(value),
            }
            for index, sign, value in zip(indices, signs, b, strict=True)
        ],
        "optimizer_target_difference": target_difference,
        "component_tolerance": tolerance,
        "optimizer_implied_contrast_interval": [interval_low, interval_high],
        "interval_basis": f"Necessary contrast interval implied by allowing every original component its own {100 * tolerance:g}% tolerance; not {100 * tolerance:g}% of the difference.",
        **support,
        "one_source_supplies_all_nonzero_contrast_columns": len(ids) == 1,
        "zero_contrast_excluded_by_component_implied_interval": not interval_low
        <= 0
        <= interval_high,
        "source_removals_failing_implied_interval_capacity": sum(
            not r["remaining_capacity_can_intersect_component_implied_interval"]
            for r in private
        ),
        "source_removal_capacity_ranges": {
            "min_remaining_lower": min(
                (r["remaining_cap_only_minimum"] for r in private),
                default=support["cap_only_minimum"],
            ),
            "max_remaining_lower": max(
                (r["remaining_cap_only_minimum"] for r in private),
                default=support["cap_only_minimum"],
            ),
            "min_remaining_upper": min(
                (r["remaining_cap_only_maximum"] for r in private),
                default=support["cap_only_maximum"],
            ),
            "max_remaining_upper": max(
                (r["remaining_cap_only_maximum"] for r in private),
                default=support["cap_only_maximum"],
            ),
        },
    }, private


def _identity_vector(name, values):
    # IDs are integers in the saved matrix producer. Do not coerce absent/string
    # identities, NaN, fractional values or Boolean masks into source groups.
    if values.dtype.kind not in "iu" or (values < 0).any():
        raise ValueError(
            f"{name} must contain nonnegative integer IDs, with no missing or nonfinite values."
        )


def analyze(run_dir, output_dir, *, tolerance=0.05):
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Tolerance must lie between zero and one.")
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    receipt_path = run_dir / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError("A completed source receipt is required.")
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError("A completed source receipt must be a JSON object.")
    paths = {
        name: run_dir / name
        for name in ("matrix.npz", "inputs_private.npz", "weights_private.npz")
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    for name, field in (
        ("matrix.npz", "matrix_sha256"),
        ("inputs_private.npz", "inputs_private_sha256"),
        ("weights_private.npz", "weights_private_sha256"),
    ):
        if receipt.get(field) != hashes[name]:
            raise ValueError(f"Input {name} does not match its completed receipt.")
    hashes["receipt.json"] = digest(receipt_path)
    required = (
        "initial_weights",
        "names",
        "target_values",
        "target_loss_weights",
        "household_source_ids",
        "target_families",
        "household_ids",
    )
    with np.load(paths["inputs_private.npz"], allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"Saved inputs are missing required arrays: {missing}")
        z = {key: archive[key] for key in required}
    with np.load(paths["weights_private.npz"], allow_pickle=False) as archive:
        if "weights" not in archive.files:
            raise ValueError("Saved weights are missing the weights array.")
        fitted = archive["weights"]
    if any(values.ndim != 1 for values in (*z.values(), fitted)):
        raise ValueError("Saved arrays must be one-dimensional vectors.")
    matrix = sparse.csr_matrix(sparse.load_npz(paths["matrix.npz"]))
    prior = z["initial_weights"]
    names = z["names"].astype(str)
    targets = z["target_values"]
    coefficients = z["target_loss_weights"]
    source_ids = z["household_source_ids"]
    families = z["target_families"].astype(str)
    row_count, household_count = matrix.shape
    if (
        any(
            len(z[key]) != row_count
            for key in (
                "names",
                "target_values",
                "target_loss_weights",
                "target_families",
            )
        )
        or any(
            len(z[key]) != household_count
            for key in ("initial_weights", "household_source_ids", "household_ids")
        )
        or len(fitted) != household_count
    ):
        raise ValueError("Saved row/household/weight/source arrays are not aligned.")
    _identity_vector("household_source_ids", source_ids)
    _identity_vector("household_ids", z["household_ids"])
    if len(set(names)) != len(names) or len(set(z["household_ids"])) != len(prior):
        raise ValueError("Matrix row names and household identities must be unique.")
    if (
        not np.isfinite(coefficients).all()
        or (coefficients < 0).any()
        or coefficients.sum() <= 0
    ):
        raise ValueError("Invalid target loss coefficient vector.")
    if (
        not all(np.isfinite(x).all() for x in (prior, fitted, targets, matrix.data))
        or (prior < 0).any()
        or (fitted < 0).any()
    ):
        raise ValueError(
            "Matrix weights/targets must be finite; weights must be nonnegative."
        )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    all_estimates = np.asarray(matrix @ fitted)
    base_loss = weighted_loss(all_estimates, targets, coefficients)
    selected = np.flatnonzero(np.isin(families, list(UC_FAMILIES)))
    rows = []
    private = []
    source_cache = {}
    for index in selected:
        row = matrix[index : index + 1]
        record = {
            "name": names[index],
            "family": families[index],
            "loss_coefficient": float(coefficients[index]),
            **row_support(row, fitted, prior, source_ids, targets[index], tolerance),
        }
        record["relative_error"] = float(
            (all_estimates[index] - targets[index]) / max(abs(targets[index]), 1.0)
        )
        _, ids, contributions = grouped_contributions(row, fitted, source_ids)
        if len(ids) < 5 or names[index].rsplit("@", 1)[0] in FIVE_PLUS:
            removals = []
            for source_id, contribution in zip(ids, contributions, strict=True):
                key = int(source_id)
                if key not in source_cache:
                    descendants = np.flatnonzero(source_ids == source_id)
                    delta = np.asarray(matrix[:, descendants] @ fitted[descendants])
                    after = all_estimates - delta
                    source_cache[key] = {
                        "source_household_id": key,
                        "descendant_household_count": len(descendants),
                        "no_refit_saved_roster_loss": weighted_loss(
                            after, targets, coefficients
                        ),
                        "no_refit_saved_roster_loss_delta": weighted_loss(
                            after, targets, coefficients
                        )
                        - base_loss,
                    }
                columns = np.flatnonzero(source_ids == source_id)
                removed_row = row[:, columns]
                removed_cap = float(
                    np.maximum(removed_row.data, 0.0)
                    @ (10 * prior[columns][removed_row.indices])
                )
                remaining_cap = record["cap_only_maximum"] - removed_cap
                effect = {
                    **source_cache[key],
                    "target_name": names[index],
                    "no_refit_contribution_loss": float(contribution),
                    "no_refit_estimate": float(all_estimates[index] - contribution),
                    "no_refit_relative_error": float(
                        (all_estimates[index] - contribution - targets[index])
                        / max(abs(targets[index]), 1.0)
                    ),
                    "remaining_cap_only_maximum": remaining_cap,
                    "remaining_cap_can_reach_lower_tolerance": bool(
                        remaining_cap >= (1 - tolerance) * targets[index]
                    ),
                    "interpretation": "Necessary individual-row capacity condition only; no refit and no protected-constraint feasibility claim.",
                }
                private.append(effect)
                removals.append(effect)
            record["source_removal_summary"] = {
                "sources_checked": len(removals),
                "source_removals_failing_remaining_cap": sum(
                    not r["remaining_cap_can_reach_lower_tolerance"] for r in removals
                ),
                "maximum_no_refit_contribution_loss": max(
                    (r["no_refit_contribution_loss"] for r in removals), default=0.0
                ),
                "minimum_remaining_cap_only_maximum": min(
                    (r["remaining_cap_only_maximum"] for r in removals),
                    default=record["cap_only_maximum"],
                ),
            }
        rows.append(record)
    contrast, contrast_private = signed_contrast(
        matrix, targets, names, fitted, prior, source_ids, tolerance
    )
    for action in contrast_private:
        action.update(source_cache[action["source_household_id"]])
    private.extend(contrast_private)
    tolerance_label = f"{100 * tolerance:g}pct"
    probes = {
        f"{label}_{tolerance_label}": lp_probe(
            matrix, targets, prior, names, requested, tolerance
        )
        for label, requested in {
            "broad_total": BROAD[:1],
            "broad_total_and_family": BROAD[:5],
            "broad_total_and_child_counts": [BROAD[0], *BROAD[5:]],
            "broad_total_family_and_child_counts": BROAD,
        }.items()
    }
    probes[f"supported_uc_rows_excluding_named_zero_cells_{tolerance_label}"] = (
        supported_uc_probe(matrix, targets, prior, names, rows, tolerance)
    )
    probes[f"broad_counts_and_named_protected_rows_{tolerance_label}"] = (
        protected_subset_probe(matrix, targets, prior, names, tolerance)
    )
    zero = [r["name"] for r in rows if r["zero_support_positive_target"]]
    cap_fail = [
        r["name"] for r in rows if not r["row_capacity_can_intersect_tolerance"]
    ]
    report = {
        "status": "completed_receipted_run",
        "run_label": run_dir.name,
        "input_sha256": hashes,
        "script_sha256": digest(__file__),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "controls": {
            "saved_row_count": len(names),
            "household_count": len(prior),
            "original_source_household_count": len(np.unique(source_ids)),
            "weight_cap_ratio": 10.0,
            "tolerance": tolerance,
            "solver_coefficients": "saved unchanged; no new Adam optimization",
            "fitted_weight_cap_exceedance_count": int(
                (fitted > 10 * prior + 1e-8).sum()
            ),
            "source_ess_definition": "Square of total absolute source-group contribution divided by sum of squared source-group contributions; clones are grouped before the calculation.",
            "saved_roster_capped_weighted_mape": base_loss,
        },
        "limitations": [
            "Every LP constrains only its explicitly named rows and cap. The protected subset adds four OBR and the saved HMRC employment bands, but excludes all other protected outcomes. No fixed-mass constraint is imposed.",
            "Saved target semantics are preserved; this tool does not relabel their source population or observation window.",
            "Zero-support rows remain listed in the full UC diagnostic. No source is deleted from the observed matrix.",
            "No-refit removals set all descendants of one source household to zero; other fitted weights stay unchanged.",
        ],
        "full_uc_necessary_conditions": {
            "rows": len(rows),
            "zero_support_positive_rows": zero,
            "rows_failing_cap_capacity": cap_fail,
            "can_satisfy_all_uc_rows_at_tolerance": False
            if cap_fail
            else "not_established_by_individual_row_checks",
        },
        "rows": rows,
        "lp_probes": probes,
        "computational_contrast": contrast,
    }
    for name, path in paths.items():
        if digest(path) != hashes[name]:
            raise ValueError(f"Input changed during postprocessing: {name}")
    if digest(receipt_path) != hashes["receipt.json"]:
        raise ValueError("Completed input receipt changed during postprocessing.")
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output_dir / "aggregate_support_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "source_group_influence_private.json").write_text(
        json.dumps(private, indent=2, allow_nan=False) + "\n"
    )
    summary = [
        f"# Matrix support and capacity: {run_dir.name}",
        "",
        f"Status: {report['status']}. No weight refit or population-engine run.",
        "",
        f"{len(rows)} UC/TCL rows; {len(zero)} have no household support despite a positive target; {len(cap_fail)} fail the individual-row 10×prior capacity bound at {100 * tolerance:g}% tolerance.",
        "",
        f"LP checks use only their explicitly named rows, with 0–10×prior weights and {100 * tolerance:g}% tolerance:",
        "",
    ]
    summary.extend(f"- {label}: {probe['status']}." for label, probe in probes.items())
    summary.extend(
        [
            "",
            "A feasible subset does not establish full-roster feasibility. The protected subset includes only four named OBR rows and the saved HMRC employment bands. Any exclusions from the supported-UC probe are listed explicitly. A positive-target zero-support row prevents full-UC feasibility. Source identities and individual-source removals are private.",
            "",
            "Five-plus child rows:",
            "",
        ]
    )
    summary.extend(
        f"- {r['name']}: {r['household_support']} household rows / {r['original_source_household_support']} original source households; source ESS {r['source_contribution_ess']:.2f}; largest source share {r['largest_source_contribution_share']:.1%}; fitted {r['fitted_estimate']:,.1f} versus target {r['target']:,.1f}; cap-only maximum {r['cap_only_maximum']:,.1f}."
        for r in rows
        if r["name"].rsplit("@", 1)[0] in FIVE_PLUS
    )
    summary.extend(
        [
            "",
            "Computational reported-5+ minus TCL5 minus TCL6+ contrast:",
            "",
            f"{contrast['household_support']} household rows / {contrast['original_source_household_support']} original source households; source ESS {contrast['source_contribution_ess']:.2f}; largest source share {contrast['largest_source_contribution_share']:.1%}. Fitted contrast {contrast['fitted_estimate']:,.1f}; optimizer target difference {contrast['optimizer_target_difference']:,.1f}. The three component tolerances imply contrast interval [{contrast['optimizer_implied_contrast_interval'][0]:,.1f}, {contrast['optimizer_implied_contrast_interval'][1]:,.1f}].",
            f"One original source supplies all nonzero contrast columns: {contrast['one_source_supplies_all_nonzero_contrast_columns']}. Removing a source fails the remaining signed-capacity condition in {contrast['source_removals_failing_implied_interval_capacity']} cases.",
            "These rows describe different concepts and windows; the subtraction explains optimizer algebra only, not an administrative statistical identity.",
        ]
    )
    (output_dir / "aggregate_support_report.md").write_text("\n".join(summary) + "\n")
    return {
        "rows": len(rows),
        "zero_rows": len(zero),
        "cap_fail_rows": len(cap_fail),
        "lp_status": {k: v["status"] for k, v in probes.items()},
        "private_source_interventions": len(private),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=0.05)
    a = p.parse_args(argv)
    if not 0 < a.tolerance < 1:
        raise ValueError("Tolerance must lie between zero and one.")
    import os

    os.umask(0o077)
    print(json.dumps(analyze(a.run_dir, a.output_dir, tolerance=a.tolerance), indent=2))


if __name__ == "__main__":
    main()
