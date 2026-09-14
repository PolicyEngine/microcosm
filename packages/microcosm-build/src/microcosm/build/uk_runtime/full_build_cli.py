"""Execute the single UK full build; all geographies are selected by default.

Numerical operations and verdicts belong to the composed graph. This module
resolves requests, executes graph endpoints and atomically materializes their
stored artifacts. Publication and signing remain explicit external services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

from microcosm.build.artifact_files import file_artifact, materialize_bytes
from microcosm.graph import (
    ArtifactInput,
    ContentStore,
    Graph,
    KernelRegistry,
    SourceRef,
    compile_graph,
    graph_to_json,
    run_graph,
)
from microcosm.graph.canonical import canonical_json

from .frs_release import load_uk_frs_release
from .full_certification import (
    append_uk_full_certification_node,
    register_uk_full_certification_kernel,
)
from .graph_build import (
    SPINE_PROVENANCE_TYPE,
    UKFullBuildConfig,
    UKFullGraph,
    bound_spine_graph,
    register_uk_full_kernels,
    uk_full_graph,
)
from .graph_calibration import UKGraphCalibrationConfig
from .graph_targets import TARGET_SELECTION_TYPE
from .graph_terminal import (
    FULL_DIAGNOSTICS_CSV_TYPE,
    FULL_DIAGNOSTICS_TYPE,
    FULL_GATE_REPORT_TYPE,
    FULL_HOLDOUT_TYPE,
    FULL_SUPPORT_CSV_TYPE,
    add_uk_export_continuation,
    add_uk_export_preparation,
    append_uk_full_gate_nodes,
    decode_full_gate_report,
    materialize_uk_export,
    materialize_uk_terminal_artifacts,
    register_uk_full_gate_kernels,
    register_uk_terminal_kernels,
)
from .local_doctrine import (
    UK_LOCAL_CLONE_COUNT,
    UK_LOCAL_MAX_WEIGHT_RATIO,
    UK_LOCAL_SOLVE_DOCTRINE,
    UK_LOCAL_SOLVE_EPOCHS,
    UK_LOCAL_TARGET_LOSS_CAP,
)
from .national_chronicle_feed import load_uk_national_chronicle_feed
from .national_frame import load_uk_national_frame
from .national_sampling import UK_SAMPLE_SEED_DEFAULT


def _target_geographies(value: str) -> tuple[str, ...] | None:
    if value == "all":
        return None
    levels = tuple(value.split(","))
    if (
        not levels
        or len(levels) != len(set(levels))
        or set(levels) - {"country", "region", "constituency", "la"}
    ):
        raise argparse.ArgumentTypeError(
            "Use all or a comma-separated subset of country,region,constituency,la."
        )
    return levels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    population = parser.add_mutually_exclusive_group(required=True)
    population.add_argument(
        "--input-h5",
        type=Path,
        help="Canonical spine checkpoint with bound build and gate sidecars.",
    )
    population.add_argument(
        "--spine-request",
        type=Path,
        help="JSON array of raw FRS spine arguments; these stages execute in the same graph.",
    )
    parser.add_argument("--input-sidecar", type=Path)
    parser.add_argument("--input-spine-gates", type=Path)
    parser.add_argument("--input-sha256")
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--ladder-sha256")
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help="Complete Chronicle artifact directory matching the committed feed pins.",
    )
    parser.add_argument("--ledger-facts-sha256")
    parser.add_argument("--ledger-manifest-sha256")
    parser.add_argument("--measure-exclusions", type=Path)
    parser.add_argument("--register-json", type=Path)
    parser.add_argument("--input-mass-reference", type=Path)
    parser.add_argument(
        "--native-scorecard",
        type=Path,
        help="Measured incumbent-surface comparison bound to immutable candidate.json and its exact output bytes.",
    )
    parser.add_argument(
        "--matched-size-scorecard",
        type=Path,
        help="Additional measured comparison with both populations at requested k.",
    )
    parser.add_argument(
        "--target-geographies",
        type=_target_geographies,
        default=None,
        metavar="all|country,...",
        help="Default all: calibrate all applicable geographies together. country is an explicit filter in this same build.",
    )
    parser.add_argument(
        "--n-clones",
        type=int,
        default=UK_LOCAL_CLONE_COUNT,
        help="Geographic pool copies K, independent of target scope and exported size k.",
    )
    parser.add_argument(
        "--dataset-households",
        type=int,
        help="Exact exported household count k via informed L0, draw and refit.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Optional pool sampling before cloning. Cannot resample an already sampled source spine.",
    )
    parser.add_argument("--sample-seed", type=int, default=UK_SAMPLE_SEED_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--selection-pi-hi", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=UK_LOCAL_SOLVE_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument(
        "--target-weight-rule",
        choices=("uniform", "grain_equal"),
        default=UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule,
    )
    parser.add_argument("--engine-blocks", type=int, default=1)
    parser.add_argument("--source-year", type=int)
    parser.add_argument("--calibration-year", type=int)
    parser.add_argument("--source-lineage-modulus", type=int)
    parser.add_argument("--expected-constituency-vintage", default="2024_pcon")
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--release-candidate", action="store_true")
    parser.add_argument("--review-date", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--resume-size-checkpoint",
        type=Path,
        help="Import an identity-verified historical size search, skipping its dense solve and search.",
    )
    parser.add_argument(
        "--graph-store",
        type=Path,
        help="Persistent shared graph store; default <out>/.graph-store.",
    )
    parser.add_argument(
        "--resume", choices=("auto", "require", "forbid"), default="auto"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the request and print the compiled operation inventory without fitting or writing files.",
    )
    args = parser.parse_args(argv)
    if args.input_h5 is None and any(
        (args.input_sidecar, args.input_spine_gates, args.input_sha256)
    ):
        parser.error("Input H5 sidecar/pin options require --input-h5.")
    if args.dataset_households is None and (
        args.selection_seed is not None
        or args.selection_pi_hi != 1.0
        or args.resume_size_checkpoint
    ):
        parser.error("Selection options require --dataset-households.")
    if args.matched_size_scorecard is not None and args.dataset_households is None:
        parser.error("A matched-size scorecard requires --dataset-households.")
    if args.release_candidate and args.skip_holdout:
        parser.error("A release candidate must evaluate applicable holdouts.")
    if args.release_candidate:
        if args.dataset_households is not None:
            parser.error(
                "Exact-count builds need separate matched-size evidence before release promotion."
            )
        if (
            args.epochs != UK_LOCAL_SOLVE_EPOCHS
            or args.n_clones != UK_LOCAL_CLONE_COUNT
            or args.target_weight_rule != UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule
            or args.engine_blocks != 1
            or args.measure_exclusions is not None
        ):
            parser.error(
                "A release candidate must use the maintained solve doctrine, pool count, single engine and reviewed exclusions."
            )
        if args.ladder_sha256 is None or (
            args.input_h5 is not None and args.input_sha256 is None
        ):
            parser.error(
                "A release candidate requires explicit input H5 and ladder digest pins."
            )
    if args.resume_size_checkpoint and args.input_h5 is None:
        parser.error(
            "Historical size checkpoints bind an input H5; raw builds resume using --graph-store."
        )
    return args


def _pin(path: Path, expected: str | None = None) -> dict:
    record = file_artifact(path)
    if expected is not None and expected != record["sha256"]:
        raise ValueError(f"Input digest differs from the requested pin: {path}.")
    return {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}


def _checkpoint_identity(args, config, pins) -> dict | None:
    if args.resume_size_checkpoint is None:
        return None
    # Match the existing checkpoint schema exactly. The import kernel also
    # verifies ordered targets, weights, household axis and recomputed losses.
    return {
        "dataset_pin": pins["dataset"],
        "ladder_pin": pins["ladder"],
        "ledger_facts_sha256": args.ledger_facts_sha256,
        "ledger_manifest_sha256": args.ledger_manifest_sha256,
        "seed": args.seed,
        "selection_seed": args.seed
        if args.selection_seed is None
        else args.selection_seed,
        "n_clones": args.n_clones,
        "dataset_households": args.dataset_households,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "sample_fraction": args.sample_fraction,
        "sample_seed": args.sample_seed,
        "source_year": config.source_year,
        "source_lineage_modulus": args.source_lineage_modulus,
        "calibration_year": config.calibration_year,
        "target_weight_rule": args.target_weight_rule,
        "engine_blocks": args.engine_blocks,
        "measure_exclusions": None
        if args.measure_exclusions is None
        else str(args.measure_exclusions),
        "doctrine": {
            "target_loss_cap": float(UK_LOCAL_TARGET_LOSS_CAP),
            "max_weight_ratio": float(UK_LOCAL_MAX_WEIGHT_RATIO),
            "scale_rule": UK_LOCAL_SOLVE_DOCTRINE.scale_rule,
            "target_weight_rule": UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule,
            "solve_epochs": int(UK_LOCAL_SOLVE_EPOCHS),
            "clone_count": int(UK_LOCAL_CLONE_COUNT),
        },
    }


@dataclass(frozen=True)
class PreparedUKFullBuild:
    full: UKFullGraph
    kernels: KernelRegistry
    sources: dict[str, Path]
    bindings: dict
    spine_provenance: ArtifactInput | None = None
    comparison_sources: dict[str, Path] | None = None


def prepare_full_build(args: argparse.Namespace) -> PreparedUKFullBuild:
    from .calibration_run import load_bound_spine_checkpoint
    from .graph import uk_spine_endpoint
    from .spine_build import (
        _rules_engine,
        _rules_engine_provenance,
        parse_uk_spine_args,
        prepare_uk_spine_execution,
    )

    release = load_uk_frs_release()
    pins = {"ladder": _pin(args.ladder, args.ladder_sha256)}
    sources = {"uk_ladder": args.ladder, "uk_ledger_facts": args.ledger_facts}
    provenance = None
    if args.input_h5 is not None:
        pins["dataset"] = _pin(args.input_h5, args.input_sha256)
        frame, _ = load_uk_national_frame(args.input_h5)
        sidecar_path = args.input_sidecar or args.input_h5.with_suffix(".build.json")
        gates_path = args.input_spine_gates or args.input_h5.with_suffix(
            ".spine_gates.json"
        )
        sidecar = load_bound_spine_checkpoint(
            sidecar_path, frame, gate_report_path=gates_path
        )
        spine = bound_spine_graph(frame)
        endpoint = "uk.full.spine_checkpoint"
        weight_kind = frame.weights_for("household").kind.value
        time_period = str(frame.metadata["time_period"])
        source_fraction = float((sidecar.get("sampling") or {}).get("fraction", 1.0))
        stages = tuple(sidecar["stages"])
        engine = _rules_engine()
        engine_identity = hashlib.sha256(
            canonical_json(_rules_engine_provenance())
        ).hexdigest()
        kernels = KernelRegistry()
        sources.update(
            uk_spine=args.input_h5,
            uk_spine_evidence=sidecar_path,
            uk_spine_gates=gates_path,
        )
        provenance = ArtifactInput(
            "spine_provenance", endpoint, "spine_provenance", SPINE_PROVENANCE_TYPE
        )
    else:
        raw_arguments = json.loads(args.spine_request.read_text())
        if not isinstance(raw_arguments, list) or not all(
            isinstance(x, str) for x in raw_arguments
        ):
            raise ValueError(
                "The spine request must be a JSON array of command arguments."
            )
        # --spine-h5 is an output control of the checkpoint command. Preparation
        # uses it only for path configuration; the full graph writes its own H5.
        if "--spine-h5" not in raw_arguments and not any(
            x.startswith("--spine-h5=") for x in raw_arguments
        ):
            raw_arguments += ["--spine-h5", str(args.out / "spine.h5")]
        raw = parse_uk_spine_args(raw_arguments)
        if args.release_candidate:
            raw.release_candidate = True
        prepared = prepare_uk_spine_execution(raw)
        spine, kernels = prepared.graph, prepared.kernels
        sources.update(prepared.sources)
        endpoint = uk_spine_endpoint(spine).population
        weight_kind = "importance"
        time_period = prepared.frs_release.time_period
        source_fraction = raw.sample_fraction
        stages = prepared.stage_names
        engine, engine_identity = prepared.engine, prepared.engine_identity
    config = UKFullBuildConfig(
        calibration_year=args.calibration_year or release.calibration_year,
        time_period=time_period,
        source_year=args.source_year
        if args.source_year is not None
        else int(time_period),
        geography_levels=args.target_geographies,
        n_clones=args.n_clones,
        sample_fraction=args.sample_fraction,
        source_sample_fraction=source_fraction,
        sample_seed=args.sample_seed,
        seed=args.seed,
        engine_blocks=args.engine_blocks,
        constituency_vintage=args.expected_constituency_vintage,
        source_lineage_modulus=args.source_lineage_modulus,
        calibration=UKGraphCalibrationConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            dataset_households=args.dataset_households,
            selection_seed=args.selection_seed,
            selection_pi_hi=args.selection_pi_hi,
            target_weight_rule=args.target_weight_rule,
        ),
    )
    if args.release_candidate and config.effective_sample_fraction != 1.0:
        raise ValueError(
            "Sampled builds cannot request release-candidate certification."
        )
    feed = load_uk_national_chronicle_feed()
    for supplied, committed in (
        (args.ledger_facts_sha256, feed.facts_sha256),
        (args.ledger_manifest_sha256, feed.manifest_sha256),
    ):
        if supplied is not None and supplied != committed:
            raise ValueError(
                "Requested Chronicle pin differs from the committed full-build feed."
            )
    optional = []
    for name, path in (
        ("uk_measure_exclusions", args.measure_exclusions),
        ("uk_frozen_register", args.register_json),
    ):
        if path is not None:
            sources[name] = path
            optional.append(name)
    if args.input_mass_reference is not None:
        sources["uk_input_mass_reference"] = args.input_mass_reference
        spine = replace(
            spine,
            sources=(
                *spine.sources,
                SourceRef("uk_input_mass_reference", "raw-bytes-v1"),
            ),
        )
    full = uk_full_graph(
        config,
        spine=spine,
        spine_population=endpoint,
        spine_weight_kind=weight_kind,
        optional_target_sources=tuple(optional),
        review_date=args.review_date.isoformat(),
        checkpoint_identity=_checkpoint_identity(args, config, pins),
    )
    if args.resume_size_checkpoint:
        from .size_checkpoint import (
            SIZE_CHECKPOINT_ARRAYS_FILENAME,
            SIZE_CHECKPOINT_MANIFEST_FILENAME,
        )

        sources["uk_size_checkpoint_manifest"] = (
            args.resume_size_checkpoint / SIZE_CHECKPOINT_MANIFEST_FILENAME
        )
        sources["uk_size_checkpoint_arrays"] = (
            args.resume_size_checkpoint / SIZE_CHECKPOINT_ARRAYS_FILENAME
        )
    graph = append_uk_full_gate_nodes(
        full.graph,
        calibration=full.calibration,
        spine_stage_names=stages,
        engine_identity=engine_identity,
        review_date=args.review_date,
        sample_fraction=config.effective_sample_fraction,
        release_candidate=args.release_candidate,
        spine_provenance=provenance,
        skip_holdout=args.skip_holdout,
    )
    bindings = {
        "schema": "microcosm.uk.full-build-request.v1",
        "configuration": asdict(config),
        "target_scope": "all"
        if config.geography_levels is None
        else list(config.geography_levels),
        "review_date": args.review_date.isoformat(),
        "engine_identity": engine_identity,
        "release_candidate": args.release_candidate,
        "skip_holdout": args.skip_holdout,
        "targets": {
            "chronicle": {
                "facts_sha256": feed.facts_sha256,
                "manifest_sha256": feed.manifest_sha256,
            },
            "paired_ladder_sha256": pins["ladder"]["sha256"],
        },
    }
    graph = add_uk_export_preparation(
        graph,
        population=full.population,
        bindings=bindings,
        artifact_inputs=(
            ArtifactInput(
                "full_gates",
                "uk.full.gates.calibrated",
                "gate_report",
                FULL_GATE_REPORT_TYPE,
            ),
        ),
    )
    register_uk_full_kernels(kernels)
    register_uk_full_gate_kernels(
        kernels, coverage_engine=engine, engine_identity=engine_identity
    )
    register_uk_terminal_kernels(kernels)
    register_uk_full_certification_kernel(kernels)
    full = replace(full, graph=graph)
    compile_graph(graph)
    comparisons = {
        name: path
        for name, path in (
            ("uk_native_scorecard", args.native_scorecard),
            ("uk_matched_size_scorecard", args.matched_size_scorecard),
        )
        if path is not None
    }
    return PreparedUKFullBuild(
        full, kernels, sources, bindings, provenance, comparisons
    )


def _through(graph: Graph, endpoint: str) -> Graph:
    """Execute an actual ancestor-closed checkpoint of the one declared graph."""
    compiled = compile_graph(graph)
    needed, pending = {endpoint}, [endpoint]
    while pending:
        for parent in compiled.predecessors[pending.pop()]:
            if parent not in needed:
                needed.add(parent)
                pending.append(parent)
    return replace(
        graph, nodes=tuple(node for node in graph.nodes if node.id in needed)
    )


def _payload(manifest, store, node: str, artifact: str) -> bytes:
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[artifact])


def _materialize_evidence(manifest, store, out: Path) -> dict:
    inventory = {}
    for node_id, receipt in manifest.nodes.items():
        for name, key in receipt.opaque_artifacts.items():
            payload = store.load_bytes(key)
            suffix = ".json" if payload.startswith((b"{", b"[")) else ".artifact"
            filename = f"{node_id}.{name}{suffix}"
            if Path(filename).name != filename:
                raise ValueError(
                    "Graph artifact name cannot be materialized as a bundle filename."
                )
            inventory[f"{node_id}/{name}"] = {
                "key": key,
                **materialize_bytes(payload, out / filename),
            }
    materialize_bytes(canonical_json(inventory), out / "evidence-index.json")
    return inventory


def _persist_checkpoint(manifest, store, args, phase: str) -> None:
    """Keep receipts and small evidence durable if a later kernel refuses."""
    payload = manifest.to_json_bytes()
    materialize_bytes(payload, args.out / f"{phase}.graph.json")
    attempt = Path(args.attempt_evidence)
    materialize_bytes(payload, attempt / f"{phase}.graph.json")
    evidence = {}
    for node_id, receipt in manifest.nodes.items():
        for name, key in receipt.opaque_artifacts.items():
            if name not in {"gate_report", "stage_evidence", "spine_provenance"}:
                continue
            filename = f"{node_id}.{name}.json"
            if Path(filename).name != filename:
                raise ValueError("Checkpoint artifact names must be simple filenames.")
            evidence[f"{node_id}/{name}"] = {
                "key": key,
                **materialize_bytes(store.load_bytes(key), attempt / filename),
            }
    materialize_bytes(
        canonical_json({"phase": phase, "artifacts": evidence}),
        attempt / "evidence-index.json",
    )


def _output_locations(prepared: PreparedUKFullBuild, args: argparse.Namespace):
    output = args.out.resolve()
    graph_store = (args.graph_store or output / ".graph-store").resolve()
    for source in {**prepared.sources, **(prepared.comparison_sources or {})}.values():
        source = source.resolve()
        if source.is_relative_to(output):
            raise ValueError(
                "Full-build output directory must not contain an input source."
            )
        if source.is_dir() and (
            output.is_relative_to(source) or graph_store.is_relative_to(source)
        ):
            raise ValueError(
                "Output files and graph store cannot alter a declared source directory."
            )
    return output, graph_store


def execute_full_build(prepared: PreparedUKFullBuild, args: argparse.Namespace) -> int:
    """Stage complete files, then publish their completion marker last."""
    if args.dry_run:
        return _execute_full_build(prepared, args)
    from microcosm.build.artifact_files import publish_staged_bundle

    output, graph_store = _output_locations(prepared, args)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Operational attempt identity never enters a scientific node/cache key.
    args.attempt_evidence = graph_store / "uk-full-attempts" / uuid.uuid4().hex
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.full-build-", dir=output.parent
    ) as temporary:
        staged_args = argparse.Namespace(**vars(args))
        staged_args.out = Path(temporary)
        staged_args.graph_store = graph_store
        status = _execute_full_build(prepared, staged_args)
        if not (staged_args.out / "build.json").exists():
            materialize_bytes(
                canonical_json(
                    {
                        "schema_version": 1,
                        "kind": "uk_full_build_refused",
                        "request": prepared.bindings,
                        "release_authorized": False,
                        "artifact_permitted": False,
                    }
                ),
                staged_args.out / "build.json",
            )
        staged = {
            ("manifest" if path.name == "build.json" else path.name): path
            for path in staged_args.out.iterdir()
            if path.is_file()
        }
        destinations = {role: output / path.name for role, path in staged.items()}
        publish_staged_bundle(staged, destinations, completion_role="manifest")
    if status == 0:
        print(
            f"UK full build: {output / f'microcosm_uk_{prepared.full.config.calibration_year}.h5'}; "
            f"target scope {prepared.bindings['target_scope']}."
        )
    return status


def _execute_full_build(prepared: PreparedUKFullBuild, args: argparse.Namespace) -> int:
    full, kernels, sources = prepared.full, prepared.kernels, prepared.sources
    if args.dry_run:
        print(json.dumps(full.operation_inventory(), indent=2))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    store = ContentStore(args.graph_store or args.out / ".graph-store")
    graph = full.graph
    materialize_bytes(graph_to_json(graph).encode(), args.out / "graph.json")
    materialize_bytes(
        canonical_json(full.operation_inventory()), args.out / "operations.json"
    )
    materialize_bytes(
        canonical_json(prepared.bindings), args.attempt_evidence / "request.json"
    )
    # Preserve source evidence before downstream transforms can raise. Each is
    # an ancestor-closed checkpoint of this graph, with the same node keys/RNG.
    resume = args.resume
    node_ids = {node.id for node in graph.nodes}
    for endpoint in (
        "spine.gates.assembled",
        "spine.gates.transferred",
        "uk.full.spine_checkpoint",
    ):
        if endpoint not in node_ids:
            continue
        checkpoint = run_graph(
            compile_graph(_through(graph, endpoint)),
            sources=sources,
            store=store,
            kernels=kernels,
            resume=resume,
        )
        _persist_checkpoint(checkpoint, store, args, endpoint)
        resume = "require" if args.resume == "require" else "auto"
    # Persist preflight outcomes before any solver can reject them.
    preflight_graph = _through(graph, "uk.full.gates.preflight")
    preflight = run_graph(
        compile_graph(preflight_graph),
        sources=sources,
        store=store,
        kernels=kernels,
        resume=resume,
    )
    _persist_checkpoint(preflight, store, args, "preflight")
    _materialize_evidence(preflight, store, args.out)
    _, admission = decode_full_gate_report(
        _payload(preflight, store, "uk.full.gates.preflight", "gate_report")
    )
    if not admission["artifact_permitted"]:
        return 1
    manifest = run_graph(
        compile_graph(_through(graph, "uk.full.gates.calibrated")),
        sources=sources,
        store=store,
        kernels=kernels,
        resume="require" if args.resume == "require" else "auto",
    )
    _persist_checkpoint(manifest, store, args, "numerical")
    _materialize_evidence(manifest, store, args.out)
    terminal_files = materialize_uk_terminal_artifacts(
        manifest,
        store,
        directory=args.out,
        stem=f"microcosm_uk_{full.config.calibration_year}",
    )
    _, enforcement = decode_full_gate_report(
        _payload(manifest, store, "uk.full.gates.calibrated", "gate_report")
    )
    if not enforcement["artifact_permitted"]:
        return 1
    manifest = run_graph(
        compile_graph(graph),
        sources=sources,
        store=store,
        kernels=kernels,
        resume="require" if args.resume == "require" else "auto",
    )
    descriptor = json.loads(
        _payload(manifest, store, "uk.full.export.prepare", "export_descriptor")
    )
    dataset = args.out / f"microcosm_uk_{full.config.calibration_year}.h5"
    materialize_uk_export(manifest.population(full.population), descriptor, dataset)
    graph = add_uk_export_continuation(
        graph,
        population=full.population,
        manifest_binding={
            "graph_sha256": hashlib.sha256(graph_to_json(graph).encode()).hexdigest()
        },
        artifact_inputs=(
            ArtifactInput(
                "full_gates",
                "uk.full.gates.calibrated",
                "gate_report",
                FULL_GATE_REPORT_TYPE,
            ),
            ArtifactInput(
                "diagnostics",
                "uk.full.gates.calibrated",
                "calibration_diagnostics",
                FULL_DIAGNOSTICS_TYPE,
            ),
            ArtifactInput("holdout", "uk.full.holdout", "holdout", FULL_HOLDOUT_TYPE),
            ArtifactInput(
                "target_diagnostics",
                "uk.full.gates.calibrated",
                "target_diagnostics_csv",
                FULL_DIAGNOSTICS_CSV_TYPE,
            ),
            ArtifactInput(
                "area_support",
                "uk.full.gates.calibrated",
                "area_support_csv",
                FULL_SUPPORT_CSV_TYPE,
            ),
            ArtifactInput(
                "target_registry",
                "uk.full.target_selection",
                "selection",
                TARGET_SELECTION_TYPE,
            ),
        ),
        evidence_files={
            "full_gates": "uk.full.gates.calibrated.gate_report.json",
            "diagnostics": terminal_files["calibration_diagnostics"]["filename"],
            "holdout": terminal_files["holdout"]["filename"],
            "target_diagnostics": terminal_files["target_diagnostics"]["filename"],
            "area_support": terminal_files["area_support"]["filename"],
            "target_registry": terminal_files["target_registry"]["filename"],
        },
    )
    evidence_sources = {
        "exported_evidence_full_gates": args.out
        / "uk.full.gates.calibrated.gate_report.json",
        "exported_evidence_diagnostics": args.out
        / terminal_files["calibration_diagnostics"]["filename"],
        "exported_evidence_holdout": args.out / terminal_files["holdout"]["filename"],
        "exported_evidence_target_diagnostics": args.out
        / terminal_files["target_diagnostics"]["filename"],
        "exported_evidence_area_support": args.out
        / terminal_files["area_support"]["filename"],
        "exported_evidence_target_registry": args.out
        / terminal_files["target_registry"]["filename"],
    }
    comparisons = prepared.comparison_sources or {}
    graph = append_uk_full_certification_node(
        graph,
        population=full.population,
        spine_provenance=prepared.spine_provenance,
        comparison_sources=tuple(comparisons),
    )
    final = run_graph(
        compile_graph(graph),
        sources={
            **sources,
            "exported_dataset": dataset,
            **evidence_sources,
            **comparisons,
        },
        store=store,
        kernels=kernels,
        resume="auto",
    )
    final.save(args.out / "build.graph.json")
    materialize_bytes(graph_to_json(graph).encode(), args.out / "graph.json")
    _materialize_evidence(final, store, args.out)
    package = json.loads(_payload(final, store, "uk.full.package", "package_inventory"))
    candidate_file = materialize_bytes(
        canonical_json(package), args.out / "candidate.json"
    )
    certification_payload = _payload(
        final, store, "uk.full.certification", "certification_readiness"
    )
    certification_file = materialize_bytes(
        certification_payload, args.out / "certification.json"
    )
    completion = {
        **package,
        "kind": "uk_full_build_completion",
        "candidate_manifest": candidate_file,
        "certification": {
            **certification_file,
            "graph_artifact_key": final.nodes["uk.full.certification"].opaque_artifacts[
                "certification_readiness"
            ],
        },
    }
    materialize_bytes(canonical_json(completion), args.out / "build.json")
    return (
        0 if package["readback_passed"] and not enforcement["enforced_blocking"] else 1
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepared = None
    try:
        prepared = prepare_full_build(args)
        return execute_full_build(prepared, args)
    except Exception as error:
        safe_output = False
        if prepared is not None:
            try:
                _output_locations(prepared, args)
                safe_output = True
            except ValueError:
                pass
        if not args.dry_run and safe_output:
            materialize_bytes(
                canonical_json(
                    {
                        "schema": "microcosm.uk.full-build-failure.v1",
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "evidence_directory": str(args.attempt_evidence)
                        if hasattr(args, "attempt_evidence")
                        else None,
                        "release_authorized": False,
                    }
                ),
                args.out / "failure.json",
            )
        print(f"UK full build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
