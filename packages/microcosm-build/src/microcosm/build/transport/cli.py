"""Prepare, compile and run a transport skeleton graph, writing local files only.

``main`` loads a country spec, prepares the run (rules-engine registry,
CREATE's column inventory), composes and compiles the one graph, then runs
ancestor-closed checkpoints of it on ``<out>/.graph-store``: through the
geography endpoint, through the calibration endpoint, through export
preparation (after which the skeleton H5 is written outside the graph),
through each terminal endpoint, and finally the whole graph. Every run shares
the store, so a later checkpoint reuses an earlier one's work.

Everything is written under ``--out``: ``graph.json``, ``manifest.json``, a
manifest per checkpoint, each opaque artifact and an index of them, the
skeleton H5 and its descriptor, the calibration diagnostics file, the graph
explorer page and ``build-status.json``. The status file lists the outputs
the spec declares as still owed by a later package. There is no upload,
staging or publication path; ``--local-only`` is the only mode.

The currency check guards new runs only: ``auto`` and ``forbid`` require
selected factors current on today's UTC date before CREATE; ``require`` only
verifies stored work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.artifact_files import file_artifact, materialize_bytes
from microcosm.build.money_normalization import (
    selected_money_factors,
    validate_money_normalization,
)
from microcosm.build.reference_factors import (
    require_current_factors,
    validate_factor_context,
)
from microcosm.frame import Frame
from microcosm.frame.adapters.axiom import AxiomEntityTableDataset
from microcosm.frame.materialize import put_frame_table
from microcosm.frame.rules import RulesEngine
from microcosm.frame.unit_construction import BenefitUnitRule
from microcosm.graph import (
    CompiledGraph,
    ContentStore,
    Graph,
    KernelContext,
    Node,
    Owned,
    Population,
    ResumePolicy,
    RunManifest,
    StoreMissError,
    StructuralDelta,
    compile_graph,
    explain_html,
    graph_to_json,
    run_graph,
)
from microcosm.graph.canonical import canonical_json, sha256_domain
from microcosm.graph.keys import source_content_key

from .codecs import RULESPEC_SOURCE_CODEC
from .compose import (
    MONEY_NORMALIZATION_RESOURCE,
    TransportGraphConfig,
    TransportGraphExtension,
    _has_resource,
    _mapping,
    _sequence,
    compose_transport_graph,
    load_transport_spec,
    reference_profile_views,
    skeleton_create_declaration,
    transport_endpoints,
    transport_pending_outputs,
    transport_resource,
    validate_transport_activation,
)
from .population_kernels import TRANSPORT_CREATE
from .registry import TransportRegistry, build_transport_registry
from .terminal_kernels import materialize_export, validate_export

__all__ = [
    "EXPORT_FILENAME",
    "PreparedTransportBuild",
    "execute_transport_graph",
    "main",
    "parse_args",
    "prepare_create_outputs",
    "prepare_transport_build",
    "through",
]

#: The skeleton's exported dataset, written under ``--out``.
EXPORT_FILENAME = "skeleton.h5"
_STORE = ".graph-store"
_CREATE_INVENTORY_FORMAT = "microcosm.transport.create-inventory/1"
_CONTENT_BASIS = (
    "Transported donor support records reweighted to destination facts; "
    "not observed destination microdata."
)


@dataclass(frozen=True)
class PreparedTransportBuild:
    """A compiled-ready graph, its run registry and its local inputs."""

    graph: Graph
    registry: TransportRegistry
    sources: Mapping[str, Path]
    endpoints: Mapping[str, object]
    pending_outputs: tuple[str, ...]


def prepare_create_outputs(
    spec: Mapping[str, object],
    sources: Mapping[str, str | Path],
    *,
    inventory_store: ContentStore | None = None,
    resume: ResumePolicy = "auto",
) -> tuple[Owned, ...]:
    """CREATE's column inventory, cached by its implementation and input bytes.

    The executor requires CREATE to declare every column it loads, and the
    columns depend on the pinned donor file. This runs ``transport.create@1``
    on the declared parameters and sources (the kernel itself checks the
    donor's size and SHA-256) and declares every non-structural column of
    the frame it returns, with its dtype. The frame is then discarded; the
    graph's own CREATE recomputes it under the executor. A cold CLI build
    therefore executes CREATE twice. A warm inventory lookup hashes every
    CREATE source but does not construct the frame or run CREATE. Without an
    ``inventory_store``, each call runs the probe; ``require`` needs a stored
    inventory, and ``forbid`` recomputes it even if its key already exists.
    """

    if resume not in {"auto", "require", "forbid"}:
        raise ValueError("resume must be 'auto', 'require' or 'forbid'.")
    if resume == "require" and inventory_store is None:
        raise StoreMissError("--resume require needs a CREATE inventory store.")
    declaration = skeleton_create_declaration(spec)
    missing = sorted(set(declaration.sources) - set(sources))
    if missing:
        raise ValueError(f"CREATE needs local paths for sources {missing}.")
    inventory_key = None
    if inventory_store is not None:
        inventory_key = sha256_domain(
            _CREATE_INVENTORY_FORMAT,
            canonical_json(
                {
                    "kernel": declaration.kernel,
                    "implementation_hash": TRANSPORT_CREATE.implementation_hash(),
                    "params": declaration.params,
                    "sources": {
                        name: source_content_key(name, sources[name])
                        for name in declaration.sources
                    },
                }
            ),
        )
        if resume != "forbid" and inventory_store.has(inventory_key):
            cached = inventory_store.load_json(
                inventory_key, kind=_CREATE_INVENTORY_FORMAT
            )
            return tuple(Owned(**row) for row in cached["columns"])
        if resume == "require":
            raise StoreMissError("--resume require needs this CREATE inventory.")
    # A probe node: CREATE reads only its params and declared sources. It
    # declares no outputs, so it is never part of a graph.
    probe = Node(
        declaration.id,
        declaration.kernel,
        params=declaration.params,
        sources=declaration.sources,
    )
    context = KernelContext(
        node=probe,
        tables={},
        weights={},
        strata=pd.Series(dtype="float64"),
        params=probe.params,
        rng=np.random.default_rng(0),
        sources={name: Path(sources[name]).resolve() for name in probe.sources},
    )
    frame = TRANSPORT_CREATE.run(context).frame
    schema = frame.schema
    structural = {
        schema.person_entity: {
            schema.person_id_column,
            *(schema.membership_column(group) for group in schema.group_entities),
        },
        **{group: {schema.id_column(group)} for group in schema.group_entities},
    }
    outputs = tuple(
        Owned(entity, column, str(table[column].dtype))
        for entity in schema.entities
        for table in (frame.table(entity),)
        for column in table.columns
        if column not in structural[entity]
    )
    if inventory_store is not None:
        inventory_store.put_json(
            inventory_key,
            {
                "columns": [
                    {"entity": item.entity, "column": item.column, "dtype": item.dtype}
                    for item in outputs
                ]
            },
            kind=_CREATE_INVENTORY_FORMAT,
            verify_existing=resume != "forbid",
        )
    return outputs


def _rules_source(spec: Mapping[str, object]) -> str:
    rows = transport_resource(spec, "transport_graph").get("sources", ())
    names = [
        row["name"]
        for row in rows
        if isinstance(row, Mapping) and row.get("codec") == RULESPEC_SOURCE_CODEC
    ]
    if len(names) != 1:
        raise ValueError("The skeleton must declare exactly one RuleSpec tree source.")
    return names[0]


def prepare_transport_build(
    spec: Mapping[str, object],
    *,
    sources: Mapping[str, str | Path],
    engines_by_binding: Mapping[str, RulesEngine] | None = None,
    extensions: tuple[TransportGraphExtension, ...] = (),
    inventory_store: ContentStore | str | Path | None = None,
    resume: ResumePolicy = "auto",
) -> PreparedTransportBuild:
    """Check factor currency and activation, then compose the prepared graph.

    A new run (``auto`` or ``forbid``) first requires every factor the default
    and scenario views select to be current on today's UTC date; ``require``
    only verifies stored work. This is the run-entry check for every caller,
    the CLI included. Composition and execution stay clock-free.

    Before registry preparation, CREATE or graph source reads, activation
    checks declaration resource paths, non-null selected roots and encoding
    shapes, the receipt kernel's data-independent contract constraints,
    reference activation rows, required age/pin presence and null scenario
    knobs. Optional null JSON fields remain allowed.
    ``sources`` maps every declared source name except the exported dataset
    (which the run writes) to a local path. ``engines_by_binding`` is the
    registry's seam for pure-Python adapters; without it, Axiom adapters are
    built for every binding. The registry's schema comes from the same
    benefit-unit rule CREATE is given. ``inventory_store`` may be a content
    store or its local directory; a directory is created only after activation
    succeeds. The CLI uses its graph store to cache the inventory. A cold build
    runs a full CREATE probe before graph execution; a warm build hashes the
    CREATE sources and reuses the stored inventory without running the probe.
    """

    if resume not in {"auto", "require", "forbid"}:
        raise ValueError("resume must be 'auto', 'require' or 'forbid'.")
    _require_current_run_factors(spec, resume=resume)
    validate_transport_activation(spec)
    if resume == "require":
        root = (
            inventory_store.root
            if isinstance(inventory_store, ContentStore)
            else Path(inventory_store)
            if inventory_store is not None
            else None
        )
        if root is None or _store_is_cold(root):
            raise StoreMissError(
                "--resume require refuses a cold transport graph store."
            )
    if isinstance(inventory_store, str | Path):
        inventory_store = ContentStore(Path(inventory_store))
    paths = {name: Path(path).resolve() for name, path in sources.items()}
    rules_source = _rules_source(spec)
    if rules_source not in paths:
        raise ValueError(f"Source {rules_source!r} (the RuleSpec tree) needs a path.")
    create = skeleton_create_declaration(spec)
    rule = BenefitUnitRule.from_spec(json.loads(create.params["unit_rule"]))
    registry = build_transport_registry(
        transport_resource(spec, "axiom_rules_bindings"),
        paths[rules_source],
        unit_rule=rule,
        engines_by_binding=engines_by_binding,
    )
    graph = compose_transport_graph(
        spec,
        TransportGraphConfig(
            engine_refs=registry.engine_refs,
            create_outputs=prepare_create_outputs(
                spec, paths, inventory_store=inventory_store, resume=resume
            ),
            extensions=extensions,
        ),
    )
    compile_graph(graph)
    declared = {source.name for source in graph.sources}
    unknown = sorted(set(paths) - declared)
    if unknown:
        raise ValueError(f"Source paths were supplied for undeclared names {unknown}.")
    return PreparedTransportBuild(
        graph,
        registry,
        MappingProxyType(paths),
        MappingProxyType(transport_endpoints(spec, extensions=extensions)),
        transport_pending_outputs(spec),
    )


def through(graph: Graph, endpoint: str) -> Graph:
    """The ancestor-closed subgraph of ``graph`` that ends at ``endpoint``."""

    compiled = compile_graph(graph)
    if endpoint not in compiled.predecessors:
        raise ValueError(f"Unknown transport endpoint {endpoint!r}.")
    needed, pending = {endpoint}, [endpoint]
    while pending:
        for parent in compiled.predecessors[pending.pop()]:
            if parent not in needed:
                needed.add(parent)
                pending.append(parent)
    return replace(
        graph, nodes=tuple(node for node in graph.nodes if node.id in needed)
    )


def _destination(out: Path, filename: str) -> Path:
    """A file directly under ``out``; refuse names or links that leave it."""

    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"Output name {filename!r} must be a simple filename.")
    path = out / filename
    if path.is_symlink() or not path.resolve().is_relative_to(out):
        raise ValueError(f"Output {filename!r} would be written outside --out.")
    return path


def _contained_store_root(out: Path) -> Path:
    """The graph store under ``out``; refuse a store tree a link could redirect.

    ``ContentStore`` writes only under ``objects/<shard>/<key>`` and ``tmp/``
    and creates those directories with ``mkdir(exist_ok=True)``, which follows
    an existing symbolic link. A link planted at ``objects``, ``tmp`` or any
    shard would therefore send writes outside ``--out``. So the store root
    must be a plain name under ``out`` (``_destination``), and no entry in an
    existing store tree may be a link: the walk does not follow links.
    """

    root = _destination(out, _STORE)
    if not root.exists():
        return root
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    relative = Path(entry.path).relative_to(out)
                    raise ValueError(
                        f"Graph store entry {str(relative)!r} is a link, which "
                        "could send store writes outside --out."
                    )
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
    return root


def _check_disjoint(
    out: Path, sources: Mapping[str, Path], *, spec_dir: Path | None = None
) -> None:
    """Refuse an output directory that overlaps an input.

    Writing inside an input tree would change the bytes a source names (a
    RuleSpec directory is hashed whole), and an input inside ``--out`` could
    be overwritten by an output. A local ``--spec-dir`` is also an input tree.
    """

    for name, path in sources.items():
        if path.is_relative_to(out) or out.is_relative_to(path):
            raise ValueError(f"--out overlaps the input of source {name!r}: {path}.")
    if spec_dir is not None:
        directory = spec_dir.resolve()
        if directory.is_relative_to(out) or out.is_relative_to(directory):
            raise ValueError(f"--out overlaps --spec-dir: {directory}.")


def _store_is_cold(root: Path) -> bool:
    objects = root / "objects"
    return not objects.is_dir() or not any(objects.iterdir())


def _evidence(manifest: RunManifest, store: ContentStore, out: Path) -> dict:
    inventory = {}
    for node_id, receipt in sorted(manifest.nodes.items()):
        for name, key in sorted(receipt.opaque_artifacts.items()):
            payload = store.load_bytes(key)
            suffix = ".json" if payload.startswith((b"{", b"[")) else ".artifact"
            written = materialize_bytes(
                payload, _destination(out, f"{node_id}.{name}{suffix}")
            )
            inventory[f"{node_id}/{name}"] = {"key": key, **written}
    materialize_bytes(
        canonical_json(inventory), _destination(out, "evidence-index.json")
    )
    return inventory


def _run(graph: Graph, registry, sources, store, resume) -> RunManifest:
    used = {name for node in graph.nodes for name in node.sources}
    compiled = compile_graph(graph)
    return run_graph(
        compiled,
        sources={name: path for name, path in sources.items() if name in used},
        store=store,
        kernels=registry.kernels,
        resume=resume,
        _population_observer=_mass_log_unchanged(compiled),
        # The observer keeps only record reprs and changes nothing.
        _population_observer_detach=False,
    )


def _mass_log_unchanged(compiled: CompiledGraph) -> Callable[[str, Population], None]:
    """Refuse a node that changes the frame mass log it received.

    A node's receipt can append records to the frame mass log
    (``frame_mass_log_append``), and every later structural node of its
    lineage, calibration included, receives that log without a ``Slice``. No
    declaration names it, so the calibration ancestry guard cannot follow it:
    a stage node could pass an undeclared value to calibration this way. No
    transport kernel appends, so in a transport run the log is CREATE's. The
    executor calls this for cold nodes and restored cache hits alike, before
    it persists either, and the exception refuses the run.
    """

    logs: dict[str, tuple[str, ...]] = {}

    def observe(node_id: str, population: Population) -> None:
        node = compiled.graph.node(node_id)
        log = tuple(map(repr, population.frame.mass_log))
        if node.structural is StructuralDelta.NONE:
            version = compiled.versions[node_id]
            received = logs[version]
        else:
            version = node_id
            received = (
                log if node.structural is StructuralDelta.CREATE else logs[node.base]
            )
        if log != received:
            raise ValueError(
                f"Transport node {node_id!r} changed the frame mass log it "
                "received; later kernels receive that log without a Slice, so "
                "a transport run takes it from CREATE only."
            )
        logs[version] = log

    return observe


class _TimestampFreeHDFStore(pd.HDFStore):
    """Disable object timestamps and unused pandas search indexes."""

    def put(self, key: str, value: pd.DataFrame | pd.Series, **options) -> None:
        options["track_times"] = False
        options["index"] = False
        super().put(key, value, **options)


def _materialize_deterministic_export(
    frame: Frame, descriptor: Mapping[str, object], path: Path
) -> dict[str, object]:
    """Validate through G5b, then rewrite the same tables without timestamps.

    The reader consumes whole tables, so query indexes are unnecessary.
    Table order, row order, values, dtypes and period follow the validated
    export, including its shared dtype boundary. Readback still hashes the
    complete H5 container. A changed export costs one additional complete
    H5 read and rewrite; unchanged exports use the existing readback path.
    The intermediate H5 stays under --out and is removed on exit.
    """

    with TemporaryDirectory(prefix=".export-", dir=path.parent) as scratch:
        staged = Path(scratch) / "validated.h5"
        materialize_export(frame, descriptor, staged)
        dataset = AxiomEntityTableDataset(file_path=staged)
        path.unlink(missing_ok=True)
        with _TimestampFreeHDFStore(str(path), mode="w") as store:
            for entity in descriptor["entities"]:
                put_frame_table(
                    store,
                    entity,
                    dataset.tables[entity],
                    preferred_format="table",
                    data_columns=True,
                )
            store.put("_time_period", pd.Series([dataset.time_period]), format="table")
        report = validate_export(path, descriptor)
        if report["passed"] is not True:
            raise ValueError(
                f"The deterministic skeleton H5 failed readback: {report['failures']}."
            )
    return file_artifact(path)


def execute_transport_graph(
    graph: Graph,
    registry: TransportRegistry,
    sources: Mapping[str, str | Path],
    *,
    out: str | Path,
    endpoints: Mapping[str, object],
    pending_outputs: Sequence[str] = (),
    resume: ResumePolicy = "auto",
) -> RunManifest:
    """Run the graph's checkpoints in order and write the local outputs.

    ``require`` refuses a cold store before anything is created, and every
    checkpoint then runs with ``require``, so a partly warm store is refused
    at its first missing node. ``forbid`` recomputes the first checkpoint;
    later checkpoints reuse this invocation's work. The skeleton H5 is
    rewritten only when the export descriptor changed; an unchanged H5 must
    pass readback, and under ``require`` it must already exist.

    Execution is clock-free. It runs a graph that :func:`prepare_transport_build`
    checked; a caller that composes a graph itself owns the currency check.
    """

    if resume not in {"auto", "require", "forbid"}:
        raise ValueError("resume must be 'auto', 'require' or 'forbid'.")
    compiled = compile_graph(graph)
    output = Path(out).resolve()
    supplied = {name: Path(path).resolve() for name, path in sources.items()}
    _check_disjoint(output, supplied)
    prepare = [node for node in graph.nodes if node.kernel == "export.prepare@1"]
    readback = [node for node in graph.nodes if node.kernel == "export.readback@1"]
    if len(prepare) != 1 or len(readback) != 1 or len(readback[0].sources) != 1:
        raise ValueError(
            "A transport skeleton has one export.prepare@1 and one export.readback@1 "
            "reading one dataset source."
        )
    prepare, export_source = prepare[0], readback[0].sources[0]
    if export_source in supplied:
        raise ValueError(
            f"Source {export_source!r} is written by the run, not supplied."
        )
    checkpoints = [
        ("geography", endpoints["geography"]),
        ("calibration", endpoints["calibration"]),
        *(
            (f"extension-{index}", endpoint)
            for index, endpoint in enumerate(endpoints.get("checkpoints", ()))
        ),
    ]
    terminals = [("terminal", endpoint) for endpoint in endpoints["terminal"]]
    for _, endpoint in (*checkpoints, *terminals):
        through(graph, endpoint)
    for label, endpoint in checkpoints:
        if label.startswith("extension-") and any(
            export_source in node.sources for node in through(graph, endpoint).nodes
        ):
            raise ValueError("Extension checkpoints must precede export readback.")
    store_root = output / _STORE
    if resume == "require" and _store_is_cold(store_root):
        raise StoreMissError("--resume require refuses a cold transport graph store.")

    output.mkdir(parents=True, exist_ok=True)
    store_root = _contained_store_root(output)
    exported = _destination(output, EXPORT_FILENAME)
    supplied[export_source] = exported
    store = ContentStore(store_root, codecs=registry.codecs)
    later = "require" if resume == "require" else "auto"
    current = resume
    materialize_bytes(
        graph_to_json(graph).encode("utf-8"), _destination(output, "graph.json")
    )
    for label, endpoint in checkpoints:
        manifest = _run(through(graph, endpoint), registry, supplied, store, current)
        materialize_bytes(
            manifest.to_json_bytes(), _destination(output, f"checkpoint-{label}.json")
        )
        current = later

    prepared = _run(through(graph, prepare.id), registry, supplied, store, current)
    descriptor_bytes = store.load_bytes(
        prepared.nodes[prepare.id].opaque_artifacts["export_descriptor"]
    )
    descriptor_path = _destination(output, "export-descriptor.json")
    previous = descriptor_path.read_bytes() if descriptor_path.is_file() else None
    if exported.exists() and previous == descriptor_bytes:
        report = validate_export(exported, json.loads(descriptor_bytes))
        if report["passed"] is not True:
            raise ValueError(
                f"The existing skeleton H5 failed readback: {report['failures']}."
            )
    elif resume == "require":
        raise StoreMissError(
            "--resume require needs the skeleton H5 of the stored run."
        )
    else:
        version = compiled.versions[prepare.id]
        _materialize_deterministic_export(
            prepared.populations[version], json.loads(descriptor_bytes), exported
        )
    materialize_bytes(descriptor_bytes, descriptor_path)
    materialize_bytes(
        prepared.to_json_bytes(),
        _destination(output, "checkpoint-export-prepared.json"),
    )
    for index, (label, endpoint) in enumerate(terminals):
        manifest = _run(through(graph, endpoint), registry, supplied, store, later)
        materialize_bytes(
            manifest.to_json_bytes(),
            _destination(output, f"checkpoint-{label}-{index}.json"),
        )

    manifest = _run(graph, registry, supplied, store, later)
    inventory = _evidence(manifest, store, output)
    produced = {
        "graph": file_artifact(_destination(output, "graph.json")),
        "skeleton_dataset": file_artifact(exported),
        "export_descriptor": file_artifact(descriptor_path),
        "artifacts": inventory,
    }
    diagnostics = [
        node.id for node in graph.nodes if node.kernel == "diagnostics.calibration@1"
    ]
    if len(diagnostics) == 1:
        produced["calibration_diagnostics"] = materialize_bytes(
            store.load_bytes(
                manifest.nodes[diagnostics[0]].opaque_artifacts["diagnostics"]
            ),
            _destination(output, "calibration-diagnostics.json"),
        )
    produced["explorer"] = materialize_bytes(
        explain_html(
            compiled, manifest, title=f"{graph.country} transport skeleton"
        ).encode("utf-8"),
        _destination(output, "explorer.html"),
    )
    materialize_bytes(manifest.to_json_bytes(), _destination(output, "manifest.json"))
    materialize_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "country": graph.country,
                "local_only": True,
                "content_basis": _CONTENT_BASIS,
                "produced": produced,
                "not_yet_produced": list(pending_outputs),
            }
        ),
        _destination(output, "build-status.json"),
    )
    return manifest


def _source_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--source takes NAME=PATH.")
    return name, Path(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--country", help="Packaged country spec code.")
    origin.add_argument("--spec-dir", type=Path, help="Local country spec directory.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=_source_argument,
        metavar="NAME=PATH",
        help="Local path of one declared graph source (repeat per source).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--resume", choices=("auto", "require", "forbid"), default="auto"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=True,
        help="Write local files only. This is the default and the only mode.",
    )
    args = parser.parse_args(argv)
    names = [name for name, _ in args.source]
    if len(names) != len(set(names)):
        parser.error("--source names a source more than once.")
    return args


def _today() -> date:
    """Today's UTC date, so a local run and CI agree on a validity boundary."""
    return datetime.now(UTC).date()


def _require_current_run_factors(
    spec: Mapping[str, object], *, resume: ResumePolicy
) -> None:
    """The currency check guards new runs only, including scenario selections.

    ``require`` only replays stored work and may verify historical factors.
    Check reference views and money-normalization selections for the default
    and each scenario against today's UTC date, before CREATE. Share the
    reference walk with activation and the money selection with its kernel.
    """
    if resume == "require":
        return
    selected = {}
    for view in reference_profile_views(spec):
        selected.update(view[2].get("selected_factors", {}))
    if _has_resource(spec, MONEY_NORMALIZATION_RESOURCE):
        declaration = validate_money_normalization(
            transport_resource(spec, MONEY_NORMALIZATION_RESOURCE)
        )
        selections = [{}]
        if _has_resource(spec, "scenarios"):
            selections.extend(
                _mapping(_mapping(scenario, "scenario").get("knobs", {}), "knobs")
                for scenario in _sequence(
                    transport_resource(spec, "scenarios").get("scenarios"), "scenarios"
                )
            )
        factor_ids = set()
        for selection in selections:
            _, factors = selected_money_factors(declaration, selection)
            factor_ids.update(value for value in factors.values() if value is not None)
        if factor_ids:
            profiles = transport_resource(spec, "reference_profiles")
            selected.update(
                validate_factor_context(
                    transport_resource(spec, profiles["factor_context_resource"]),
                    sorted(factor_ids),
                )
            )
    require_current_factors(selected, today=_today())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; return 0 on success and 1 on a refused build.

    Preparation's currency check guards new runs only (``auto`` and
    ``forbid``), before CREATE. ``require`` verifies existing stored work.
    """

    try:
        args = parse_args(argv)
    except SystemExit as error:
        if error.code == 0:
            return 0
        print(
            "transport build refused: invalid command-line arguments.", file=sys.stderr
        )
        return 1
    try:
        output = args.out.resolve()
        sources = {name: path.resolve() for name, path in args.source}
        _check_disjoint(output, sources, spec_dir=args.spec_dir)
        store_root = _contained_store_root(output)
        if args.resume == "require" and _store_is_cold(store_root):
            raise StoreMissError(
                "--resume require refuses a cold transport graph store."
            )
        spec = load_transport_spec(args.spec_dir if args.spec_dir else args.country)
        prepared = prepare_transport_build(
            spec, sources=sources, inventory_store=store_root, resume=args.resume
        )
        execute_transport_graph(
            prepared.graph,
            prepared.registry,
            prepared.sources,
            out=args.out,
            endpoints=prepared.endpoints,
            pending_outputs=prepared.pending_outputs,
            resume=args.resume,
        )
    except Exception as error:
        print(f"transport build refused: {error}", file=sys.stderr)
        return 1
    print(f"Transport skeleton written under {args.out}.")
    if prepared.pending_outputs:
        print("Not yet produced: " + "; ".join(prepared.pending_outputs) + ".")
    return 0
