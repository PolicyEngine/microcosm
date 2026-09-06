"""Invented-only, serial cached proof. No genuine input execution command exists."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parent
PYTHON = "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/bin/python"
PYTHON_SHA = "bf4ea26231212e330b8e9b6f24969a30f894fec88665bf42754070d114ac7fc6"
BASELINE_HEAD = "1f6ffa7b441701cb2b1254e4bb48241e1e2895de"
CAPS = {
    "wall_seconds": 420,
    "cpu_seconds": 360,
    "rss_bytes": 3 * 1024**3,
    "input_bytes": 32 * 1024**2,
    "households": 6,
    "persons": 12,
    "expanded_households": 12,
    "concurrent_workers": 1,
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, data):
    def default(value):
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(type(value).__name__)

    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=default)
        + "\n"
    )


def read(path):
    return json.loads(Path(path).read_text())


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args])


def bytecode():
    prefix = sys.pycache_prefix
    require(
        sys.dont_write_bytecode
        and prefix
        and Path(prefix).is_absolute()
        and not Path(prefix).exists(),
        "FRESH_ABSOLUTE_ABSENT_BYTECODE_PREFIX",
    )
    return {
        "dont_write_bytecode": sys.dont_write_bytecode,
        "pycache_prefix": prefix,
        "prefix_absent": True,
        "isolated": sys.flags.isolated,
        "orig_argv": sys.orig_argv,
    }


def bootstrap(checkout, head):
    flags = bytecode()
    require(
        Path(sys.executable).resolve() == Path(PYTHON).resolve(), "INTERPRETER_PATH"
    )
    require(sha(Path(sys.executable).resolve()) == PYTHON_SHA, "INTERPRETER_SHA256")
    require(git(checkout, "rev-parse", "HEAD").decode().strip() == head, "START_HEAD")
    require(not git(checkout, "status", "--porcelain").strip(), "DIRTY_SOURCE")
    sys.path.insert(0, str(HERE))
    for shard in ("build", "frame", "graph", "fit", "calibrate", "data"):
        sys.path.insert(0, str(checkout / "packages" / f"microcosm-{shard}" / "src"))
    return flags


def loaded_proof(checkout, head):
    paths = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__name__", "")
        location = getattr(module, "__file__", None)
        if location and (
            name.startswith("microcosm.")
            or name.startswith("test_us_")
            or name.startswith(("survey_lineage_prep", "us_survey_lineage_graph"))
        ):
            path = Path(location).resolve()
            owner, revision = (
                (CHECKOUT, git(CHECKOUT, "rev-parse", "HEAD").decode().strip())
                if path.is_relative_to(HERE)
                else (checkout, head)
            )
            require(path.is_relative_to(owner), "FOREIGN_LOADED_MODULE")
            rel = str(path.relative_to(owner))
            blob = git(owner, "show", f"{revision}:{rel}")
            require(blob == path.read_bytes(), "LOADED_GIT_BYTES")
            paths[str(path)] = {
                "git_head": revision,
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
    require(paths, "EMPTY_LOADED_PROOF")
    require(git(checkout, "rev-parse", "HEAD").decode().strip() == head, "FINISH_HEAD")
    return paths


def graph_for(root, *, baseline=False):
    from microcosm.graph import Graph, Owned

    columns = tuple(Owned(*row) for row in read(root / "columns.json"))
    if baseline:
        from microcosm.build.us_runtime import graph_combined_clone as clone
        from microcosm.build.us_runtime import graph_composed_asec_measures as measures

        graph = measures.composed_asec_population_graph(
            columns, sample_fraction=1.0, sample_seed=17, geography=True
        )
        added = clone.us_combined_survey_clone_nodes(
            columns,
            base="assign_us_puma_ladder.boundary",
            source_channels=("acs", "asec"),
        )
        registry = measures.composed_asec_population_registry(geography=True)
        clone.register_us_combined_survey_clone_kernels(registry)
        return Graph(
            country=graph.country, sources=graph.sources, nodes=(*graph.nodes, *added)
        ), registry
    from us_survey_lineage_graph import integrated_graph, integrated_registry

    identity = read(root / "projection-identity.json")
    identity["cohort_rows"] = tuple(tuple(row) for row in identity["cohort_rows"])
    return integrated_graph(
        columns, sample_fraction=1.0, sample_seed=17, **identity
    ), integrated_registry()


def source_inventory(root):
    """Only a fresh invented directory is passed here; never discover inputs."""
    from survey_lineage_prep.invented import sources

    paths = set()
    for path in sources(root).values():
        paths.update(path.rglob("*") if path.is_dir() else [path])
    inventory = {}
    for path in sorted(paths):
        require(
            path.is_relative_to(root) and not path.is_symlink(), "INVENTED_SOURCE_PATH"
        )
        if path.is_file():
            inventory[str(path.relative_to(root))] = {
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
    require(
        sum(item["bytes"] for item in inventory.values()) < CAPS["input_bytes"],
        "INPUT_CAP",
    )
    return inventory


def instrument(registry, *, warm=False):
    calls = []
    for kernel in registry._kernels.values():
        original = kernel.run

        def run(context, original=original):
            calls.append(context.node.id)
            if warm:
                raise RuntimeError("REQUIRED_KERNEL_CALL_FORBIDDEN")
            return original(context)

        kernel.run = run
    return calls


def verify_demographics(before, after, root):
    import numpy as np
    import pandas as pd

    from microcosm.build.us_runtime import graph_composed_asec_binding as arm
    from microcosm.build.us_runtime import (
        graph_composed_asec_demographics as demographic,
    )
    from microcosm.build.us_runtime.support_provenance import support_channel_column

    for entity in before.entities:
        expected, actual = before.table(entity), after.table(entity)
        if entity == "person":
            # All old cells outside the two masked ASEC canonical rewrites.
            keep = [c for c in expected if c not in demographic.CANONICAL_COLUMNS]
            pd.testing.assert_frame_equal(
                expected[keep], actual[keep], check_exact=True
            )
            acs = expected[support_channel_column(entity)].eq("acs")
            pd.testing.assert_frame_equal(
                expected.loc[acs], actual.loc[acs, expected.columns], check_exact=True
            )
            mask = actual[arm.arm_row_column("person")].to_numpy(bool)
            projection = demographic.admit_asec_demographic_projection(
                (root / "demographic.bin").read_bytes()
            )
            # Real graph binding already authenticates the native-person mapping;
            # fixture source order is deliberately reversed and includes unknowns.
            require(
                actual.loc[mask, "is_female"].isna().sum() == 1, "UNKNOWN_SEX_PRESERVED"
            )
            require(
                actual.loc[mask, "is_household_head"].isna().sum() == 2,
                "UNKNOWN_HOUSEHOLD_PRESERVED",
            )
            require(projection.rows == int(mask.sum()), "DEMOGRAPHIC_SOURCE_ROWS")
        else:
            pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        if entity in before.weighted_entities:
            require(
                np.array_equal(
                    before.weights_for(entity).values.view("uint64"),
                    after.weights_for(entity).values.view("uint64"),
                ),
                "NATIVE_WEIGHT_BITS",
            )
    pd.testing.assert_series_equal(before.strata, after.strata, check_exact=True)
    return {
        "all_incumbent_source_columns_checked": True,
        "acs_every_cell_unchanged": True,
        "native_ids_memberships_weights_unchanged": True,
        "unknown_sex_rows": 1,
        "unknown_reference_rows": 2,
    }


def phase(root, name, checkout, head):
    flags = bootstrap(checkout, head)
    resource.setrlimit(resource.RLIMIT_CPU, (CAPS["cpu_seconds"], CAPS["cpu_seconds"]))

    def watch_rss():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > CAPS["rss_bytes"]:
                os._exit(86)
            time.sleep(0.25)

    threading.Thread(target=watch_rss, daemon=True).start()
    from survey_lineage_prep import guard, invented

    denials, wires = guard.install()
    start = time.monotonic()
    if name == "stage":
        result = invented.make(root, checkout)
        require(
            result == {"households": CAPS["households"], "persons": CAPS["persons"]},
            "FIXTURE_ROW_CAP",
        )
        dump(root / "invented-input-inventory.json", source_inventory(root))
    else:
        from survey_lineage_prep import verification

        from microcosm.build.frame_checkpoint import (
            load_frame_checkpoint,
            write_frame_checkpoint,
        )
        from microcosm.build.us_runtime.graph_sources import (
            frame_column_declarations,
            us_source_codecs,
        )
        from microcosm.graph import ContentStore, compile_graph, run_graph
        from microcosm.graph.keys import source_content_key

        baseline = name == "baseline"
        invented.restore(root, checkout, baseline=baseline)
        # Baseline shares the exact same new invented source tree and store.
        if not baseline:
            require(
                source_inventory(root) == read(root / "invented-input-inventory.json"),
                "SOURCE_DRIFT",
            )
        graph, registry = graph_for(root, baseline=baseline)
        compiled = compile_graph(graph)
        calls = instrument(registry, warm=name == "warm")
        source_paths = invented.sources(root, baseline=baseline)
        source_keys = {
            key: source_content_key(key, path) for key, path in source_paths.items()
        }
        store = ContentStore(root / "store", codecs=us_source_codecs())
        verification.BOUND = "assign_us_puma_ladder.boundary"
        verification.ADDED = ("combined_survey_puf_support_clone",)
        captured, observe = verification.trace_anchors()
        old_profile = sys.getprofile()
        sys.setprofile(observe)
        try:
            manifest = run_graph(
                compiled,
                sources=source_paths,
                store=store,
                kernels=registry,
                resume="require"
                if name == "warm"
                else "forbid"
                if baseline
                else "auto",
            )
        finally:
            sys.setprofile(old_profile)
        before = manifest.population(verification.BOUND)
        after = manifest.population(verification.ADDED[0])
        clone_proof = verification.verify_clone(before, after)
        anchors = verification.verify_design(before, after, captured)
        require(after.n("household") == CAPS["expanded_households"], "CLONE_ROW_CAP")
        require(
            manifest.mass_ledger(verification.ADDED[0])[:-1]
            == manifest.mass_ledger(verification.BOUND),
            "MASS_LEDGER_PREFIX",
        )
        result = {
            "node_keys": {n: v.key for n, v in manifest.nodes.items()},
            "implementation_hashes": {
                n: v.kernel_impl_hash for n, v in manifest.nodes.items()
            },
            "typed_artifacts": {
                n: v.typed_artifacts for n, v in manifest.nodes.items()
            },
            "cache_hits": {n: v.hit for n, v in manifest.nodes.items()},
            "kernel_calls": list(calls),
            "source_keys": source_keys,
            "manifest_key": manifest.key,
            "clone_verification": clone_proof,
            "design_anchors": anchors,
            "mass_ledger_prefix_preserved": True,
            "compiled_order": compiled.order,
            "predecessors": dict(compiled.predecessors),
            "node_declarations": {
                n: {
                    "kernel": graph.node(n).kernel,
                    "sources": graph.node(n).sources,
                    "population": compiled.versions[n],
                    "artifact_inputs": [
                        vars(edge) | {"type": vars(edge.type)}
                        for edge in graph.node(n).artifact_inputs
                    ],
                }
                for n in compiled.order
            },
            "complete_columns": [
                [c.entity, c.column, c.dtype] for c in frame_column_declarations(before)
            ],
        }
        checkpoint = root / f"{name}-clone.h5"
        write_frame_checkpoint(checkpoint, after)
        verification.compare_frames(after, load_frame_checkpoint(checkpoint).frame)
        result["checkpoint_sha256"] = sha(checkpoint)
        # Checkpoints carry explicit reconstructed metadata separately from
        # the frame cell digest and the executor's original design anchors.
        result["checkpoint_metadata"] = load_frame_checkpoint(checkpoint).metadata
        manifest.save(root / f"{name}-manifest.json")
        if baseline:
            write_frame_checkpoint(root / "baseline-native.h5", before)
        else:
            from us_survey_lineage_graph import (
                CLONE_BINDING,
                CLONE_POPULATION,
                NATIVE_BINDING,
            )

            from microcosm.build.us_runtime import (
                graph_native_household_origin as origin,
            )

            bound = origin.verify_materialized_population_origins(
                manifest,
                store,
                population_node=CLONE_POPULATION,
                binding_node=CLONE_BINDING,
                parent_binding_node=NATIVE_BINDING,
            )
            result["full_materialized_verification"] = bound.summary
            result["binding_sha256"] = hashlib.sha256(bound.payload).hexdigest()
            result["demographic_preservation"] = verify_demographics(
                load_frame_checkpoint(root / "baseline-native.h5").frame, before, root
            )
            old = read(root / "baseline-receipt.json")
            common = set(old["node_keys"]) & set(result["node_keys"])
            unchanged = sorted(
                n for n in common if old["node_keys"][n] == result["node_keys"][n]
            )
            rekeyed = sorted(common - set(unchanged))
            require(
                len(unchanged) == 6 and len(rekeyed) == 5,
                "SIX_OLD_COMPOSITION_KEYS_AND_TWO_CLONE_REKEYS",
            )
            result["old_keys_unchanged"], result["old_keys_rekeyed"] = (
                unchanged,
                rekeyed,
            )
            require(
                len(calls)
                == (0 if name == "warm" else len(compiled.order) - len(unchanged)),
                "CALL_COUNT",
            )
            require(
                all(v.hit for v in manifest.nodes.values())
                if name == "warm"
                else sorted(n for n, v in manifest.nodes.items() if v.hit) == unchanged,
                "HIT_ROSTER",
            )
            if name == "cold":
                from survey_lineage_prep.probes import refusals

                result["refusals"] = refusals(
                    root, manifest, store, graph, registry, source_paths
                )
    require(
        not denials and all(getattr(holder, name) is fn for holder, name, fn in wires),
        "DURABLE_GUARDS",
    )
    result.update(
        {
            "status": "PASS",
            "invented_only": True,
            "pid": os.getpid(),
            "head": head,
            "elapsed_seconds": time.monotonic() - start,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "bytecode_start": flags,
            "bytecode_finish": bytecode(),
            "loaded_git_files": loaded_proof(checkout, head),
            "denials": denials,
            "durable_tripwires": len(wires),
            "runtime_versions": {
                n: importlib.metadata.version(n)
                for n in (
                    "numpy",
                    "pandas",
                    "h5py",
                    "tables",
                    "scipy",
                    "torch",
                    "microunit",
                    "jsonschema",
                    "policyengine-us",
                    "policyengine-core",
                )
            },
        }
    )
    dump(root / f"{name}-receipt.json", result)


def supervise(root, name, checkout, head):
    prefix = root.parent / ("absent-bytecode-" + uuid.uuid4().hex)
    require(not prefix.exists(), "CHILD_PREFIX_EXISTS")
    command = [
        PYTHON,
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={prefix}",
        str(Path(__file__).resolve()),
        "--worker",
        name,
        "--output",
        str(root),
        "--checkout",
        str(checkout),
        "--head",
        head,
    ]
    start = time.monotonic()
    with (root / f"{name}.log").open("x") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=log,
            start_new_session=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        while child.poll() is None:
            if time.monotonic() - start > CAPS["wall_seconds"]:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise RuntimeError("WORKER_RESOURCE_CAP")
            time.sleep(1)
    require(child.returncode == 0, f"WORKER_FAILED:{name}:see {name}.log")
    require(not prefix.exists(), "CHILD_BYTECODE_WRITTEN")
    return {
        "command": command,
        "elapsed_seconds": time.monotonic() - start,
        "returncode": child.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-checkout", type=Path)
    parser.add_argument(
        "--worker",
        choices=("stage", "baseline", "cold", "warm"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--head", help=argparse.SUPPRESS)
    args = parser.parse_args()
    parent_flags = bytecode()
    if args.worker:
        require(args.output.name.startswith("invented-survey-"), "INVENTED_OUTPUT_NAME")
        phase(args.output, args.worker, args.checkout, args.head)
        return
    require(
        args.selfcheck and args.baseline_checkout, "ONLY_INVENTED_SELFCHECK_SUPPORTED"
    )
    root = args.output.absolute()
    require(root.name.startswith("invented-survey-"), "INVENTED_OUTPUT_NAME")
    root.mkdir(mode=0o700)  # No overwrite or existing store reuse.
    head = git(CHECKOUT, "rev-parse", "HEAD").decode().strip()
    phases = {}
    for name in ("stage", "baseline", "cold", "warm"):
        checkout, revision = (
            (args.baseline_checkout, BASELINE_HEAD)
            if name == "baseline"
            else (CHECKOUT, head)
        )
        print(f"Starting invented {name}", flush=True)
        phases[name] = supervise(root, name, checkout, revision)
    cold, warm = read(root / "cold-receipt.json"), read(root / "warm-receipt.json")
    for key in (
        "node_keys",
        "source_keys",
        "implementation_hashes",
        "typed_artifacts",
        "manifest_key",
        "binding_sha256",
        "checkpoint_sha256",
        "checkpoint_metadata",
        "full_materialized_verification",
        "design_anchors",
    ):
        require(cold[key] == warm[key], "FRESH_REQUIRED_IDENTITY:" + key)
    require(
        cold["pid"] != warm["pid"] and not warm["kernel_calls"],
        "FRESH_REQUIRED_ZERO_CALLS",
    )
    dump(
        root / "completion.json",
        {
            "status": "PASS",
            "integration_head": head,
            "invented_only": True,
            "caps": CAPS,
            "phases": phases,
            "compiled_nodes": len(cold["node_keys"]),
            "cold_hits": sum(cold["cache_hits"].values()),
            "cold_calls": len(cold["kernel_calls"]),
            "warm_hits": sum(warm["cache_hits"].values()),
            "warm_calls": 0,
            "bytecode_parent_start": parent_flags,
            "bytecode_parent_finish": bytecode(),
            "receipts": {n: sha(root / f"{n}-receipt.json") for n in phases},
        },
    )
    print("Invented integrated proof PASS", flush=True)


if __name__ == "__main__":
    main()
