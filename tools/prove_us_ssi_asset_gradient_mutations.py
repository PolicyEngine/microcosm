"""Run each gradient behavior assertion against a deliberately mutated source.

Each pytest invocation compiles a copied source function into the real
function's code slot for the test call, then restores it. The workspace
sources and git history are never modified. ``--fresh-processes`` uses
isolated subprocesses instead when repeated imports are inexpensive.
Run with the engine-free environment's Python; all assertions stay in the
registered engine_free/us pytest module and need no additional CI job.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_gradient.py"
NUMERIC = "microcosm.build.ssi_asset_gradient"
RUNTIME = "microcosm.build.us_runtime.ssi_take_up"
SOURCES = {
    NUMERIC: ROOT
    / "packages/microcosm-build/src/microcosm/build/ssi_asset_gradient.py",
    RUNTIME: ROOT
    / "packages/microcosm-build/src/microcosm/build/us_runtime/ssi_take_up.py",
}

# (test, module, function, replaced source fragment, replacement, purpose).
MUTATIONS = (
    (
        "test_propensity_is_monotone_for_nonpositive_slopes",
        NUMERIC,
        "ssi_asset_propensity",
        "slope * np.log1p(assets)",
        "-slope * np.log1p(assets)",
        "Reverse the asset coefficient sign.",
    ),
    (
        "test_finite_propensity_stays_strictly_inside_unit_interval",
        NUMERIC,
        "ssi_asset_propensity",
        "return np.clip(probability, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))",
        "return probability",
        "Remove strict probability endpoint clipping.",
    ),
    (
        "test_intercept_solve_reproduces_weighted_target",
        NUMERIC,
        "solve_ssi_asset_intercept",
        "return (lower + upper) / 2",
        "return (lower + upper) / 2 + 0.1",
        "Bias the solved intercept by 0.1.",
    ),
    (
        "test_zero_weight_assets_do_not_move_intercept",
        NUMERIC,
        "solve_ssi_asset_intercept",
        "weights = np.asarray(weights, dtype=np.float64)",
        "weights = np.asarray(weights, dtype=np.float64)\n    weights = np.where(weights == 0, 1.0, weights)",
        "Give zero-weight observations positive mass.",
    ),
    (
        "test_every_person_obeys_own_asset_law_and_reporters_stay_true",
        RUNTIME,
        "_assign_sources",
        'source["anchor"].to_numpy(dtype=bool) |',
        "np.zeros(len(source), dtype=bool) |",
        "Remove unconditional reporter pinning.",
    ),
    (
        "test_every_person_obeys_own_asset_law_and_reporters_stay_true[extreme_intercept_precision]",
        RUNTIME,
        "_gradient_intercept",
        "return solve_ssi_asset_intercept(assets, weights, mass, slope)",
        "return _intercept(float(ssi_asset_propensity(np.asarray([0.0]), solve_ssi_asset_intercept(assets, weights, mass, slope), 0.0)[0]))",
        "Round-trip the exact intercept through sigmoid and logit, losing steep-slope precision.",
    ),
    (
        "test_source_draws_survive_person_row_reordering",
        RUNTIME,
        "_source_table",
        "_stable_source_draw(str(source_id), seed=int(seed))",
        '_stable_source_draw(str(rows.index[rows["source_id"].eq(source_id)][0]), seed=int(seed))',
        "Key the uniform draw by the first physical row index.",
    ),
    (
        "test_weight_split_support_clones_preserve_original_flags",
        RUNTIME,
        "with_us_ssi_take_up",
        'assigned = rows["source_id"].map(selected).to_numpy(dtype=bool)',
        'assigned = rows["source_id"].map(selected).to_numpy(dtype=bool)\n    if "clone_index" in rows:\n        assigned[rows["clone_index"].eq(2)] ^= True',
        "Invert the extra support clone's flag.",
    ),
    (
        "test_zero_slope_flags_match_frozen_schema4_implementation_exactly",
        RUNTIME,
        "_propensities",
        "prior\n            if slopes[key] == 0 or intercept is None",
        "min(prior + 0.25, 1.0)\n            if slopes[key] == 0 or intercept is None",
        "Shift the historical constant-prior threshold.",
    ),
    (
        "test_nonzero_slopes_reject_scalar_only_legacy_prior_basis",
        RUNTIME,
        "with_us_ssi_take_up",
        "selected, bands, priors = _assign_sources(",
        "if any(band.nonanchor_asset_distribution is None for band in resolved_basis.bands):\n        slopes = {key: 0.0 for key in slopes}\n    selected, bands, priors = _assign_sources(",
        "Silently default a scalar artifact to constant priors.",
    ),
    (
        "test_schema5_delivered_basis_retains_asset_distribution_and_target_mass",
        RUNTIME,
        "ssi_take_up_prior_basis_from_artifact",
        '"candidate_nonanchor_asset_distribution"',
        '"prior_basis_nonanchor_asset_distribution"',
        "Seed the retry from assignment assets instead of delivered asset mass.",
    ),
    (
        "test_gate_rejects_gradient_coefficient_corruption",
        RUNTIME,
        "us_ssi_take_up_gate",
        "passed=not failures",
        "passed=True",
        "Ignore the integrity gate's detected coefficient failures.",
    ),
    (
        "test_gate_rejects_candidate_asset_mass_corruption",
        RUNTIME,
        "us_ssi_take_up_gate",
        "passed=not failures",
        "passed=True",
        "Ignore the integrity gate's detected asset mass failures.",
    ),
    (
        "test_physical_asec_assets_own_source_despite_support_reimputation",
        RUNTIME,
        "_source_table",
        'source.loc[has_original, "liquid_assets"] = original_asset_values.loc[\n            has_original\n        ]',
        'source.loc[has_original, "liquid_assets"] = source.loc[has_original, "liquid_assets"]',
        "Ignore the physical ASEC owner and retain first-row support assets.",
    ),
    (
        "test_asec_absent_source_assets_must_be_unambiguous",
        RUNTIME,
        "_source_table",
        'if source.loc[~has_original, "asset_count"].gt(1).any():',
        'if False and source.loc[~has_original, "asset_count"].gt(1).any():',
        "Accept conflicting support assets when their physical ASEC owner is absent.",
    ),
    (
        "test_captured_source_assets_preserve_frozen_law_after_owner_pruning",
        RUNTIME,
        "_source_table",
        'source["liquid_assets"] = owned_assets',
        'source["liquid_assets"] = source["liquid_assets"]',
        "Ignore captured source holdings after their physical owner is pruned.",
    ),
    (
        "test_captured_source_asset_map_requires_valid_coverage_and_owner_match[missing]",
        RUNTIME,
        "_source_table",
        "if not np.isfinite(owned_assets).all() or owned_assets.lt(0).any():",
        "if False and (not np.isfinite(owned_assets).all() or owned_assets.lt(0).any()):",
        "Accept a captured attribute map missing a retained source identity.",
    ),
    (
        "test_captured_source_asset_map_requires_valid_coverage_and_owner_match[negative]",
        RUNTIME,
        "_source_table",
        "if not np.isfinite(owned_assets).all() or owned_assets.lt(0).any():",
        "if False and (not np.isfinite(owned_assets).all() or owned_assets.lt(0).any()):",
        "Accept a captured source attribute with a negative holding.",
    ),
    (
        "test_captured_source_asset_map_requires_valid_coverage_and_owner_match[nan]",
        RUNTIME,
        "_source_table",
        "if not np.isfinite(owned_assets).all() or owned_assets.lt(0).any():",
        "if False and (not np.isfinite(owned_assets).all() or owned_assets.lt(0).any()):",
        "Accept a captured source attribute with a nonfinite holding.",
    ),
    (
        "test_captured_source_asset_map_requires_valid_coverage_and_owner_match[physical_mismatch]",
        RUNTIME,
        "_source_table",
        "if not owned_assets.loc[has_original].equals(\n            original_asset_values.loc[has_original]\n        ):",
        "if False and not owned_assets.loc[has_original].equals(\n            original_asset_values.loc[has_original]\n        ):",
        "Accept captured holdings that disagree with a physical ASEC owner.",
    ),
    (
        "test_missing_liquid_asset_inputs_are_rejected",
        RUNTIME,
        "_source_table",
        'person = frame.table("person")',
        'person = frame.table("person").copy()\n    if "stock_assets" not in person:\n        person["stock_assets"] = 0.0',
        "Silently fill a missing asset input with zero.",
    ),
    (
        "test_asset_age_diagnostics_reconcile_all_source_people_and_weights",
        RUNTIME,
        "_asset_age_diagnostics",
        "return result",
        "return []",
        "Drop the asset-by-age diagnostics.",
    ),
    (
        "test_public_reseed_uses_existing_weights_and_candidate_probe",
        RUNTIME,
        "reseed_us_ssi_take_up",
        "uncapped_ssi=uncapped_ssi",
        "uncapped_ssi=np.zeros_like(uncapped_ssi)",
        "Discard the engine candidate-basis probe in public reseeding.",
    ),
)


def mutated_source(source: str, function: str, before: str, after: str) -> str:
    """Replace one documented fragment in one named source function."""
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    segment = ast.get_source_segment(source, node)
    if segment.count(before) != 1:
        raise ValueError(
            f"Mutation fragment occurs {segment.count(before)} times in {function}"
        )
    mutated = segment.replace(before, after)
    result = source.replace(segment, mutated, 1)
    compile(result, "<deliberate-source-mutation>", "exec")
    return result


def run_pytest(
    target: str, *, plugin_dir: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    roots = [
        str(path / "src")
        for path in (ROOT / "packages").iterdir()
        if (path / "src").is_dir()
    ]
    roots.append(str(ROOT))
    if plugin_dir is not None:
        roots.insert(0, str(plugin_dir))
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    arguments = [
        sys.executable,
        "-m",
        "pytest",
        target,
        "-q",
        "--tb=short",
        "--maxfail=1",
    ]
    if plugin_dir is not None:
        arguments.extend(["-p", "ssi_source_mutation_plugin"])
    return subprocess.run(
        arguments, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800
    )


def run_pytest_in_process(
    target: str,
    *,
    module_name: str | None = None,
    function: str | None = None,
    source: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run genuine pytest assertions with a source-derived code mutation."""
    import pytest

    plugins = []
    body_reached = False
    if module_name is not None:
        module = importlib.import_module(module_name)
        node = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == function
        )
        segment = ast.get_source_segment(source, node)
        namespace = vars(module).copy()
        exec(
            compile(
                "from __future__ import annotations\n" + segment,
                "<deliberately-mutated-source-function>",
                "exec",
            ),
            namespace,
        )
        actual = getattr(module, function)
        original_code = actual.__code__
        changed_code = namespace[function].__code__

        class SourceMutation:
            @pytest.hookimpl(hookwrapper=True)
            def pytest_runtest_call(self, item):
                nonlocal body_reached
                body_reached = True
                actual.__code__ = changed_code
                try:
                    yield
                finally:
                    actual.__code__ = original_code

        plugins.append(SourceMutation())
    arguments = [target, "-q", "--tb=short", "--maxfail=1"]
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = pytest.main(arguments, plugins=plugins)
    result = subprocess.CompletedProcess(
        arguments, int(status), stdout.getvalue(), stderr.getvalue()
    )
    result.test_body_reached = body_reached
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-processes", action="store_true")
    args = parser.parse_args()
    sources = {name: path.read_text() for name, path in SOURCES.items()}
    tests = {
        node.name
        for node in ast.parse((ROOT / TEST).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    covered = {row[0].split("[")[0] for row in MUTATIONS}
    if tests != covered:
        raise ValueError(f"Behavior mutation coverage mismatch: {tests ^ covered}")
    roots = [
        str(path / "src")
        for path in (ROOT / "packages").iterdir()
        if (path / "src").is_dir()
    ]
    sys.path[:0] = roots + [str(ROOT)]
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    baseline = run_pytest(TEST) if args.fresh_processes else run_pytest_in_process(TEST)
    report = {
        "method": (
            "Temporary copied source compiled into the real modules before pytest collection; original source files unchanged."
            if args.fresh_processes
            else "Copied source function compiled into the actual function's code slot during pytest_runtest_call; original code restored after each call. Imports remain warm between pytest invocations. Original source files unchanged."
        ),
        "source_sha256": {
            str(SOURCES[name].relative_to(ROOT)): hashlib.sha256(
                source.encode()
            ).hexdigest()
            for name, source in sources.items()
        },
        "test_sha256": hashlib.sha256((ROOT / TEST).read_bytes()).hexdigest(),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": os.environ.get(
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD"
            ),
            "PYTEST_PLUGINS": os.environ.get("PYTEST_PLUGINS"),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "TORCH_DEVICE_BACKEND_AUTOLOAD": os.environ.get(
                "TORCH_DEVICE_BACKEND_AUTOLOAD"
            ),
            "workspace_source_paths": roots,
        },
        "selected_test_file": TEST,
        "baseline_pytest_arguments": baseline.args,
        "baseline_returncode": baseline.returncode,
        "baseline_tail": "\n".join(
            (baseline.stdout + baseline.stderr).splitlines()[-20:]
        ),
        "mutations": [],
        "all_killed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if baseline.returncode:
        print(report["baseline_tail"], flush=True)
    else:
        for test, module, function, before, after, description in MUTATIONS:
            mutated = mutated_source(sources[module], function, before, after)
            with tempfile.TemporaryDirectory(
                prefix="ssi-source-mutation-"
            ) as directory:
                temporary = Path(directory)
                source_path = temporary / "mutated_source.py"
                source_path.write_text(mutated)
                plugin = temporary / "ssi_source_mutation_plugin.py"
                plugin.write_text(
                    "import importlib\nfrom pathlib import Path\n"
                    "def pytest_configure(config):\n"
                    f"    module = importlib.import_module({module!r})\n"
                    f"    path = Path({str(source_path)!r})\n"
                    "    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)\n"
                )
                result = (
                    run_pytest(f"{TEST}::{test}", plugin_dir=temporary)
                    if args.fresh_processes
                    else run_pytest_in_process(
                        f"{TEST}::{test}",
                        module_name=module,
                        function=function,
                        source=mutated,
                    )
                )
            output = result.stdout + result.stderr
            failed_node_ids = re.findall(r"^FAILED (\S+)", output, flags=re.MULTILINE)
            body_reached = getattr(result, "test_body_reached", args.fresh_processes)
            killed = (
                result.returncode == 1
                and body_reached
                and f"FAILED {TEST}::{test}" in output
            )
            row = {
                "test": test,
                "module": module,
                "function": function,
                "mutation": description,
                "before": before,
                "after": after,
                "returncode": result.returncode,
                "test_body_reached": body_reached,
                "failed_node_ids": failed_node_ids,
                "killed": killed,
                "pytest_tail": "\n".join(output.splitlines()[-25:]),
            }
            report["mutations"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{test}: {'killed' if killed else 'NOT KILLED'}", flush=True)
    report["all_killed"] = (
        baseline.returncode == 0
        and len(report["mutations"]) == len(MUTATIONS)
        and all(row["killed"] for row in report["mutations"])
    )
    report["behavior_test_functions"] = sorted(tests)
    report["covered_behavior_test_functions"] = sorted(covered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# SSI asset gradient mutation proof",
        "",
        report["method"],
        "",
        f"Baseline return code: {baseline.returncode}. All {len(MUTATIONS)} source mutations killed: {report['all_killed']}.",
        "",
        "| Behavior assertion | Deliberate source mutation | Failed pytest node IDs | Result |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["mutations"]:
        lines.append(
            f"| `{row['test']}` | {row['mutation']} | {', '.join(f'`{node}`' for node in row['failed_node_ids'])} | {'Pytest assertion failed (exit 1)' if row['killed'] else 'Not proven'} |"
        )
    lines.extend(
        [
            "",
            "Verbatim baseline pytest tail:",
            "",
            "```text",
            report["baseline_tail"],
            "```",
            "",
            "The JSON companion records exact replacement fragments, source SHA-256 values, and each mutation's verbatim pytest failure tail.",
            "",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(lines))
    return 0 if report["all_killed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
