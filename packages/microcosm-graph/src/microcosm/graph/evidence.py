"""Recorded graph execution, detached from kernels and country runtimes.

Native bindings retain exact graph/manifest bytes and run-end source identities.
The portable overlay contains only selected operational facts and summaries
from explicitly supplied artifact-type providers. No kernel is called here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .availability import execution_state, unavailable_artifacts
from .canonical import canonical_json, normative
from .decl import (
    GATE_OUTCOMES,
    CompiledGraph,
    Graph,
    GraphError,
    StructuralDelta,
    compile_graph,
)
from .kernel import Capabilities, KernelRole
from .keys import (
    artifact_key,
    frame_key,
    node_key,
    opaque_artifact_key,
    platform_fingerprint,
    seed,
    weights_key,
)
from .manifest import RunManifest
from .presentation import PRESENTATION_EXTENSION, digest, require, text
from .schema import (
    _plain_json,
    _require_compiler_result,
    graph_schema,
    validate_graph_schema,
)
from .serialize import graph_from_json, graph_to_json
from .store import ContentStore

BINDING_PROTOCOL = "microcosm.graph.run-binding.v1"
EVIDENCE_INPUT_PROTOCOL = "microcosm.graph.execution-input.v1"
EXECUTION_PROTOCOL = "microcosm.graph.execution-evidence.v1"
_MAX_BYTES = 32 * 1024 * 1024
_SUMMARY_FIELDS = frozenset({"node", "artifact", "key", "type", "data"})
_LOG = logging.getLogger(__name__)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _plain(value: object) -> object:
    return _plain_json(json.loads(canonical_json(value)))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(path: Path) -> tuple[object, bytes]:
    """Bounded regular-file reader used for all native evidence inputs."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode), "evidence input must be a regular file")
        require(info.st_size <= _MAX_BYTES, "evidence input byte limit")
        payload = stream.read(_MAX_BYTES + 1)
    require(len(payload) <= _MAX_BYTES, "evidence input byte limit")
    value = json.loads(payload, object_pairs_hook=_unique)
    return _plain_json(value), payload


@dataclass(frozen=True)
class RecordedRun:
    compiled: CompiledGraph
    manifest: RunManifest
    binding: Mapping[str, object]
    references: Mapping[str, object] = field(default_factory=dict)
    summaries: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ArtifactSummaryContext:
    """An explicitly trusted provider's verified artifact context."""

    run: RecordedRun
    node_id: str
    artifact: str
    payload: bytes
    store: ContentStore

    def input_payload(self, alias: str) -> bytes:
        entry = self.run.manifest.nodes[self.node_id].typed_artifacts["inputs"][alias]
        return self.store.load_bytes(entry["key"])


ArtifactSummaryProvider = Callable[[ArtifactSummaryContext], Mapping[str, object]]
ArtifactSummaryRegistry = Mapping[tuple[str, int], ArtifactSummaryProvider]


def record_run_binding(
    compiled: CompiledGraph,
    manifest: RunManifest,
    *,
    attempt_id: str,
    phase: str,
) -> RecordedRun:
    """Capture the executor's live run-end identities before serialization.

    This binds records and bytes, not a signature or an authenticity verdict.
    A detached historic manifest cannot manufacture missing source identities.
    """
    binding = {
        "protocol": BINDING_PROTOCOL,
        "attempt_id": text(attempt_id, "attempt id"),
        "phase": text(phase, "phase"),
        "graph_sha256": sha256(graph_to_json(compiled.graph).encode()),
        "manifest_sha256": sha256(manifest.to_json_bytes()),
        "manifest_key": manifest.key,
        "source_identities": dict(manifest.source_identities),
        "platform": platform_fingerprint(),
    }
    run = RecordedRun(compiled, manifest, binding)
    _validate_run(run)
    return run


def _validate_run(run: RecordedRun) -> None:
    compiled, manifest = _require_compiler_result(run.compiled), run.manifest
    binding = _plain(dict(run.binding))
    require(
        type(binding) is dict
        and set(binding)
        == {
            "protocol",
            "attempt_id",
            "phase",
            "graph_sha256",
            "manifest_sha256",
            "manifest_key",
            "source_identities",
            "platform",
        },
        "run binding fields",
    )
    require(binding["protocol"] == BINDING_PROTOCOL, "run binding protocol")
    for field_name in ("attempt_id", "phase", "platform"):
        text(binding[field_name], field_name)
    require(
        manifest.country == compiled.graph.country, "graph/manifest country mismatch"
    )
    require(
        digest(binding["graph_sha256"])
        == sha256(graph_to_json(compiled.graph).encode()),
        "graph binding mismatch",
    )
    require(binding["manifest_key"] == manifest.key, "manifest key mismatch")
    require(
        digest(binding["manifest_sha256"]) == sha256(manifest.to_json_bytes()),
        "manifest bytes mismatch",
    )
    sources = binding["source_identities"]
    require(type(sources) is dict, "source identities must be an object")
    require(
        set(sources)
        == {name for node in compiled.graph.nodes for name in node.sources},
        "missing or unexpected run-end source identities",
    )
    for identity in sources.values():
        digest(identity)
    require(
        set(manifest.nodes) <= set(compiled.order),
        "manifest contains unknown operations",
    )
    keys = {}
    for node_id in compiled.order:
        if node_id not in manifest.nodes:
            continue
        receipt = manifest.nodes[node_id]
        require(
            not receipt.legacy_capabilities,
            "legacy receipt has no current binding contract",
        )
        require(isinstance(receipt.capabilities, Capabilities), "receipt capabilities")
        require(
            receipt.kernel_ref == compiled.graph.node(node_id).kernel,
            "kernel reference mismatch",
        )
        try:
            expected = node_key(
                compiled,
                node_id,
                keys,
                receipt.kernel_impl_hash,
                sources,
                kernel_capabilities=receipt.capabilities,
                platform_identity=binding["platform"],
            )
        except KeyError as error:
            raise ValueError(
                f"Graph evidence: incomplete receipt ancestry: {error}"
            ) from error
        require(receipt.key == expected, f"receipt does not match graph at {node_id!r}")
        require(receipt.seed == seed(expected), "receipt seed mismatch")
        node = compiled.graph.node(node_id)
        require(
            receipt.capabilities.structural == node.structural,
            "structural contract mismatch",
        )
        for (entity, column), key in receipt.artifacts.items():
            require(
                key == artifact_key(expected, entity, column),
                "column identity mismatch",
            )
        if receipt.frame_key is not None:
            require(receipt.frame_key == frame_key(expected), "frame identity mismatch")
        if receipt.weight_key is not None:
            weight_entity = (
                node.weights.entity
                if node.weights is not None
                else node.params.get("expand_weight_entity")
                if node.structural is StructuralDelta.EXPAND
                else None
            )
            require(
                isinstance(weight_entity, str)
                and receipt.weight_key == weights_key(expected, weight_entity),
                "weight identity mismatch",
            )
        for name, key in receipt.opaque_artifacts.items():
            require(
                key == opaque_artifact_key(expected, name),
                "opaque artifact identity mismatch",
            )
        typed = receipt.typed_artifacts
        require(
            set(typed.get("outputs", {}))
            == {item.name for item in node.artifact_outputs},
            "typed output declarations mismatch",
        )
        require(
            set(typed.get("inputs", {}))
            == {item.name for item in node.artifact_inputs},
            "typed input declarations mismatch",
        )
        for item in node.artifact_outputs:
            entry = typed["outputs"][item.name]
            require(
                entry["type"] == normative(item.type)
                and entry["producer"] == node_id
                and entry["key"] == opaque_artifact_key(expected, item.name),
                "typed output identity mismatch",
            )
        for item in node.artifact_inputs:
            entry = typed["inputs"][item.name]
            require(
                entry["type"] == normative(item.type)
                and entry["producer"] == item.producer
                and entry["artifact"] == item.artifact
                and entry["key"]
                == opaque_artifact_key(keys[item.producer], item.artifact),
                "typed input identity mismatch",
            )
        keys[node_id] = expected


def _artifacts(run: RecordedRun, store: ContentStore) -> list[dict]:
    """Expose verified byte digests, never payloads, axes or column values."""
    records = {}
    for node_id, receipt in run.manifest.nodes.items():
        entries = [("column", f"{e}.{c}", k) for (e, c), k in receipt.artifacts.items()]
        entries += [
            ("bytes", name, key) for name, key in receipt.opaque_artifacts.items()
        ]
        if receipt.frame_key:
            entries.append(("frame", "population", receipt.frame_key))
        if receipt.weight_key:
            entries.append(("column", "weights", receipt.weight_key))
        for kind, name, key in entries:
            if key not in records:
                meta = store.metadata(key, kind=kind)
                require(
                    meta.get("node_key") == receipt.key,
                    f"artifact producer identity mismatch for {key}",
                )
                records[key] = {
                    "key": key,
                    "kind": kind,
                    "payloads": _plain(meta["payloads"]),
                    "producers": [],
                }
            records[key]["producers"].append({"node": node_id, "name": name})
    return [records[key] for key in sorted(records)]


def _summaries(
    run: RecordedRun,
    store: ContentStore,
    registry: ArtifactSummaryRegistry,
    *,
    cached: Mapping[str, dict] | None = None,
    strict: bool = True,
) -> tuple[dict, ...]:
    """Run the registered providers; with ``strict=False`` a failure is recorded.

    Evidence capture inside a build is diagnostic: a provider that raises after
    a successful solve must not abort the build, so its error is kept beside
    the artifact binding and the summary status becomes ``provider_failed``.
    The explicit export command runs providers strictly.
    """
    summaries = []
    for node_id, receipt in run.manifest.nodes.items():
        for name, descriptor in receipt.typed_artifacts.get("outputs", {}).items():
            artifact_type = descriptor["type"]
            type_key = (artifact_type["name"], artifact_type["schema_version"])
            provider = registry.get(type_key)
            if provider is None or name not in receipt.opaque_artifacts:
                continue
            key = receipt.opaque_artifacts[name]
            if cached is not None and key in cached:
                summaries.append(cached[key])
                continue
            record = {
                "node": node_id,
                "artifact": name,
                "key": key,
                "type": dict(artifact_type),
            }
            try:
                data = _plain(
                    provider(
                        ArtifactSummaryContext(
                            run, node_id, name, store.load_bytes(key), store
                        )
                    )
                )
                require(type(data) is dict, "artifact summary must be an object")
            except Exception as error:
                if strict:
                    raise
                _LOG.warning(
                    "Artifact summary for %s.%s failed and was recorded: %s: %s",
                    node_id,
                    name,
                    type(error).__name__,
                    error,
                )
                record["data"] = {}
                record["error"] = f"{type(error).__name__}: {error}"
            else:
                record["data"] = data
            summaries.append(record)
    return tuple(summaries)


def collect_execution_evidence(
    schema: object,
    *,
    runs: Sequence[RecordedRun],
    store: ContentStore,
    artifact_summaries: ArtifactSummaryRegistry | None = None,
) -> dict:
    """Validate recorded runs and project a bounded, aggregate-only overlay."""
    schema = validate_graph_schema(schema)
    root = compile_graph(graph_from_json(canonical_json(schema["graph"]).decode()))
    require(bool(runs), "execution evidence needs a recorded run")
    phases = []
    seen = set()
    summary_catalog = {}
    for run in runs:
        _validate_run(run)
        require(
            run.compiled.graph.country == root.graph.country, "phase country mismatch"
        )
        identity = (run.binding["attempt_id"], run.binding["phase"])
        require(identity not in seen, "duplicate attempt/phase")
        seen.add(identity)
        for node in run.compiled.graph.nodes:
            try:
                current = root.graph.node(node.id)
            except GraphError as error:
                raise ValueError(
                    "Graph evidence: phase operation absent from exported graph."
                ) from error
            require(
                node.normative() == current.normative(),
                "phase operation declaration mismatch",
            )
            require(
                run.compiled.predecessors[node.id] == root.predecessors[node.id],
                "phase compiler dependencies mismatch",
            )
        require(
            run.compiled.graph.normative() == root.graph.normative(),
            "phase mass semantics mismatch",
        )
        root_sources = {source.name: source for source in root.graph.sources}
        for source in run.compiled.graph.sources:
            require(
                source.name in root_sources
                and source.codec == root_sources[source.name].codec,
                "phase source declaration mismatch",
            )
        artifacts = _artifacts(run, store)
        summaries = _summaries(run, store, artifact_summaries or {})
        for summary in run.summaries:
            require(
                type(summary) is dict
                and _SUMMARY_FIELDS <= set(summary) <= _SUMMARY_FIELDS | {"error"},
                "recorded summary fields",
            )
            text(summary["node"], "summary node")
            text(summary["artifact"], "summary artifact")
            digest(summary["key"])
            if "error" in summary:
                text(summary["error"], "summary error")
        supplied = {(item["node"], item["artifact"]): item for item in run.summaries}
        supplied.update({(item["node"], item["artifact"]): item for item in summaries})
        summaries = tuple(supplied.values())
        for summary in summaries:
            receipt = run.manifest.nodes.get(summary["node"])
            require(receipt is not None, "summary has no receipt")
            descriptor = receipt.typed_artifacts.get("outputs", {}).get(
                summary["artifact"]
            )
            require(
                descriptor is not None
                and summary["key"] == receipt.opaque_artifacts.get(summary["artifact"])
                and summary["type"] == dict(descriptor["type"]),
                "summary artifact binding mismatch",
            )
            require(type(summary["data"]) is dict, "summary data must be an object")
            previous = summary_catalog.get(summary["key"])
            require(
                previous is None or previous == summary,
                "inconsistent summary for one artifact",
            )
            summary_catalog[summary["key"]] = summary
        operations = {}
        for node_id in run.compiled.order:
            receipt = run.manifest.nodes.get(node_id)
            if receipt is None:
                operations[node_id] = {
                    "execution": "missing",
                    "cache": "unknown",
                    "gate": "unknown",
                }
                continue
            state = execution_state(receipt.receipt)
            gate = (
                receipt.receipt.get("outcome")
                if receipt.capabilities.role is KernelRole.GATE
                else "not_applicable"
            )
            require(gate in GATE_OUTCOMES, "invalid gate outcome")
            operations[node_id] = {
                "node_key": receipt.key,
                "kernel_role": receipt.capabilities.role.value,
                "execution": "unreached"
                if state == "unreached"
                else "failed"
                if state == "gate_exception"
                or receipt.receipt.get("rejected") is True
                or receipt.receipt.get("outcome") == "rejected"
                else "completed",
                "cache": "unknown"
                if state == "unreached"
                else "hit"
                if receipt.hit
                else "miss",
                "gate": gate,
                "wall_time_s": receipt.wall_time,
                "kernel_ref": receipt.kernel_ref,
                "kernel_impl_hash": receipt.kernel_impl_hash,
                "blocked_by": dict(receipt.receipt["execution"]["blocked_by"])
                if state == "unreached"
                else {},
                "unavailable_artifacts": sorted(
                    unavailable_artifacts(
                        receipt.receipt, receipt.typed_artifacts.get("outputs", {})
                    )
                ),
                "weight_transition": _plain(
                    normative(run.compiled.graph.node(node_id).weights)
                )
                if run.compiled.graph.node(node_id).weights is not None
                else None,
                "mass": {
                    key: receipt.receipt["mass"][key]
                    for key in ("policy", "before", "after")
                    if key in receipt.receipt.get("mass", {})
                },
                "weight_cap": {
                    key: receipt.receipt[key]
                    for key in (
                        "weight_anchor",
                        "max_weight_ratio",
                        "realized_max_weight_ratio",
                    )
                    if key in receipt.receipt
                },
                "artifact_summary_status": {
                    name: "unavailable"
                    if name not in receipt.opaque_artifacts
                    else "provider_failed"
                    if "error" in supplied.get((node_id, name), {})
                    else "present"
                    if (node_id, name) in supplied
                    else "provider_not_configured"
                    for name in receipt.typed_artifacts.get("outputs", {})
                },
            }
        phases.append(
            {
                "attempt_id": identity[0],
                "phase": identity[1],
                "graph_sha256": run.binding["graph_sha256"],
                "manifest_sha256": run.binding["manifest_sha256"],
                "manifest_key": run.manifest.key,
                "started_at": run.manifest.started_at,
                "finished_at": run.manifest.finished_at,
                "source_identities": dict(run.binding["source_identities"]),
                "operations": operations,
                "artifacts": artifacts,
                "summaries": [
                    {key: item[key] for key in ("node", "artifact", "key")}
                    for item in summaries
                ],
                "references": dict(run.references),
            }
        )
    return _plain(
        {
            "protocol": EXECUTION_PROTOCOL,
            "graph_sha256": schema["graph_sha256"],
            "phases": phases,
            "summaries": [summary_catalog[key] for key in sorted(summary_catalog)],
            "verification": "Recorded graph/receipt keys and store payload digests checked; no kernel replay, signature verification or release certification.",
            "history": "Only supplied phases are recorded; absent earlier computation is unknown.",
        }
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    require(len(payload) <= _MAX_BYTES, "native evidence byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, canonical_json(value))


def save_run_evidence(
    compiled: CompiledGraph,
    manifest: RunManifest,
    *,
    store: ContentStore,
    directory: Path,
    attempt_id: str,
    phase: str,
    artifact_summaries: ArtifactSummaryRegistry | None = None,
) -> Path:
    """Append native evidence for one phase; never create an Orrery snapshot."""
    run = record_run_binding(compiled, manifest, attempt_id=attempt_id, phase=phase)
    identity = sha256(
        canonical_json([attempt_id, phase, run.binding["manifest_sha256"]])
    )
    directory = Path(directory)
    index_path = directory / "execution.evidence.json"
    index = (
        read_json(index_path)[0]
        if index_path.exists()
        else {"protocol": EVIDENCE_INPUT_PROTOCOL, "runs": []}
    )
    require(index.get("protocol") == EVIDENCE_INPUT_PROTOCOL, "evidence index protocol")
    require(
        not any(
            (old["attempt_id"], old["phase"]) == (attempt_id, phase)
            for old in index["runs"]
        ),
        "attempt/phase already recorded",
    )
    cached = {}
    for previous in index["runs"]:
        ref = previous["summaries"]
        relative = Path(ref["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "summary cache path",
        )
        values, payload = read_json(directory / relative)
        require(
            sha256(payload) == digest(ref["sha256"]), "summary cache digest mismatch"
        )
        for item in values:
            cached[item["key"]] = item
    summary_records = _summaries(
        run, store, artifact_summaries or {}, cached=cached, strict=False
    )
    entry = {"attempt_id": attempt_id, "phase": phase}
    for name, value in (
        ("graph", json.loads(graph_to_json(compiled.graph))),
        ("manifest", json.loads(manifest.to_json())),
        ("binding", dict(run.binding)),
        ("summaries", list(summary_records)),
    ):
        relative = Path(identity) / f"{name}.json"
        _atomic_json(directory / relative, value)
        entry[name] = {
            "path": relative.as_posix(),
            "sha256": sha256(canonical_json(value)),
        }
    index["runs"].append(entry)
    _atomic_json(index_path, index)
    return index_path


def load_run_evidence(
    path: Path, *, store: ContentStore, reference_base: Path | None = None
) -> tuple[RecordedRun, ...]:
    """Load exact, digest-bound phase files without executing country code."""
    path = Path(path)
    index, _ = read_json(path)
    require(
        type(index) is dict and set(index) == {"protocol", "runs"},
        "evidence index fields",
    )
    require(index["protocol"] == EVIDENCE_INPUT_PROTOCOL, "evidence index protocol")
    require(type(index["runs"]) is list, "evidence runs must be an array")
    runs = []
    for entry in index["runs"]:
        require(
            type(entry) is dict
            and set(entry)
            == {"attempt_id", "phase", "graph", "manifest", "binding", "summaries"},
            "evidence run fields",
        )
        values = {}
        refs = {}
        for name in ("graph", "manifest", "binding", "summaries"):
            ref = entry[name]
            require(
                type(ref) is dict and set(ref) == {"path", "sha256"},
                "evidence file reference",
            )
            relative = Path(text(ref["path"], "evidence path"))
            require(
                not relative.is_absolute() and ".." not in relative.parts,
                "evidence path must stay in bundle",
            )
            target = path.parent / relative
            require(
                target.resolve().is_relative_to(path.parent.resolve()),
                "evidence symlink leaves bundle",
            )
            values[name], payload = read_json(target)
            require(
                sha256(payload) == digest(ref["sha256"]),
                "evidence file digest mismatch",
            )
            refs[name] = {
                "url": Path(
                    os.path.relpath(target, reference_base or path.parent)
                ).as_posix(),
                "sha256": ref["sha256"],
                "label": f"{entry['phase']}: {name}",
            }
        compiled = compile_graph(
            graph_from_json(canonical_json(values["graph"]).decode())
        )
        manifest = RunManifest.load(path.parent / entry["manifest"]["path"], store)
        binding = values["binding"]
        require(
            type(values["summaries"]) is list
            and all(type(item) is dict for item in values["summaries"]),
            "recorded summaries must be objects in an array",
        )
        run = RecordedRun(compiled, manifest, binding, refs, tuple(values["summaries"]))
        _validate_run(run)
        require(
            entry["attempt_id"] == binding["attempt_id"]
            and entry["phase"] == binding["phase"],
            "phase index/binding mismatch",
        )
        runs.append(run)
    return tuple(runs)


CHECKPOINT_PROVENANCE_FIELDS = (
    ("graph_declaration", "Producing graph declaration", True),
    ("graph_manifest", "Producing run manifest", True),
    ("graph_schema", "Producing compiler schema", False),
    ("graph_execution_evidence", "Producing execution evidence index", False),
)


def checkpoint_references(
    record: object, *, base: Path
) -> tuple[list[dict], str | None]:
    """Project a checkpoint's recorded producing-run files into upstream references.

    ``record`` is the producer-written provenance mapping (for example a
    checkpoint sidecar) whose ``graph_declaration``, ``graph_manifest`` and
    optional ``graph_schema`` / ``graph_execution_evidence`` entries each carry
    ``path`` and ``sha256``. Only files whose bytes still match their recorded
    digest become references with a local URL; absent files are named, with
    their recorded digest, in the returned missing reason rather than linked.
    A digest mismatch refuses the projection. Nothing is reconstructed from
    today's declarations.
    """
    require(isinstance(record, Mapping), "checkpoint provenance must be a mapping")
    refs: list[dict] = []
    missing: list[str] = []
    for name, label, required in CHECKPOINT_PROVENANCE_FIELDS:
        entry = record.get(name)
        if entry is None and not required:
            continue
        if (
            not isinstance(entry, Mapping)
            or not entry.get("path")
            or not entry.get("sha256")
        ):
            missing.append(label)
            continue
        recorded = digest(entry["sha256"])
        path = Path(text(entry["path"], f"{name} path"))
        if not path.is_absolute():
            path = Path(base) / path
        if not path.is_file():
            missing.append(f"{label} bytes unavailable (sha256 {recorded})")
            continue
        require(
            sha256(read_json(path)[1]) == recorded,
            f"recorded checkpoint {label.lower()} digest mismatch",
        )
        refs.append({"label": label, "url": str(path.resolve()), "sha256": recorded})
    return refs, "; ".join(missing) if missing else None


def save_graph_schema(
    graph: Graph | CompiledGraph,
    directory: Path,
    *,
    presentation: Mapping[str, object] | None = None,
    filename: str = "graph.schema.json",
) -> Path:
    """Write the compiler schema with its presentation contract beside a build.

    Upstream checkpoint references that name local files are copied into the
    directory as ``upstream-<sha256>.json`` (after re-checking their digest)
    and relinked relatively, so the schema and any later Orrery export can be
    published or moved together with the recorded bytes.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    extensions = None
    if presentation is not None:
        contract = _plain(dict(presentation))
        require(type(contract) is dict, "presentation must be an object")
        scope = contract.get("scope", {})
        boundaries = scope.get("boundaries", []) if type(scope) is dict else []
        for boundary in boundaries if type(boundaries) is list else []:
            upstream = boundary.get("upstream", []) if type(boundary) is dict else []
            for ref in upstream if type(upstream) is list else []:
                url = ref.get("url") if type(ref) is dict else None
                if type(url) is not str or urlsplit(url).scheme:
                    continue
                source = Path(url)
                if not source.is_absolute() or not source.is_file():
                    continue
                payload = read_json(source)[1]
                require(
                    sha256(payload) == digest(ref.get("sha256")),
                    "checkpoint reference changed during capture",
                )
                relative = Path(f"upstream-{ref['sha256']}.json")
                _atomic_bytes(directory / relative, payload)
                ref["url"] = relative.as_posix()
        extensions = {PRESENTATION_EXTENSION: contract}
    compiled = compile_graph(graph) if isinstance(graph, Graph) else graph
    schema = graph_schema(compiled, extensions=extensions)
    path = directory / filename
    _atomic_json(path, schema)
    return path


def publish_run_evidence(
    index_path: Path, out_dir: Path, *, prefix: str = "evidence"
) -> Path:
    """Flatten a durable attempt bundle into one publisher directory.

    The published ``execution.evidence.json`` keeps every previously published
    attempt whose flat files still verify, so a repeated or resumed build into
    the same output directory does not erase the record of earlier computation
    behind a replay of cache hits. Published files that no longer match their
    recorded digests refuse publication instead of being dropped silently.
    """
    index_path = Path(index_path)
    out_dir = Path(out_dir)
    text(prefix, "prefix")
    require("/" not in prefix and ".." not in prefix, "prefix must be a file stem")
    index, _ = read_json(index_path)
    require(
        type(index) is dict
        and index.get("protocol") == EVIDENCE_INPUT_PROTOCOL
        and type(index.get("runs")) is list,
        "evidence index",
    )
    attempts = {text(entry["attempt_id"], "attempt id") for entry in index["runs"]}
    for attempt in attempts:
        require("/" not in attempt and ".." not in attempt, "attempt id")
    destination = out_dir / "execution.evidence.json"
    runs: list[dict] = []
    if destination.exists():
        published, _ = read_json(destination)
        require(
            type(published) is dict
            and published.get("protocol") == EVIDENCE_INPUT_PROTOCOL
            and type(published.get("runs")) is list,
            "published evidence index",
        )
        for entry in published["runs"]:
            require(
                type(entry) is dict
                and set(entry)
                == {"attempt_id", "phase", "graph", "manifest", "binding", "summaries"},
                "published evidence run fields",
            )
            if entry["attempt_id"] in attempts:
                continue
            for field in ("graph", "manifest", "binding", "summaries"):
                ref = entry[field]
                relative = Path(text(ref["path"], "published evidence path"))
                require(
                    len(relative.parts) == 1 and not relative.is_absolute(),
                    "published evidence path",
                )
                target = out_dir / relative
                require(
                    target.is_file(),
                    f"published evidence file {relative} is missing; "
                    "restore it or clear the published evidence",
                )
                require(
                    sha256(read_json(target)[1]) == digest(ref["sha256"]),
                    f"published evidence file {relative} no longer matches its digest",
                )
            runs.append(entry)
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in index["runs"]:
        published_entry = {"attempt_id": entry["attempt_id"], "phase": entry["phase"]}
        for field in ("graph", "manifest", "binding", "summaries"):
            relative = Path(text(entry[field]["path"], "evidence path"))
            require(
                not relative.is_absolute() and ".." not in relative.parts,
                "evidence path must stay in bundle",
            )
            filename = (
                f"{prefix}-{entry['attempt_id']}-{relative.parent.name}-{field}.json"
            )
            payload = read_json(index_path.parent / relative)[1]
            require(
                sha256(payload) == digest(entry[field]["sha256"]),
                "native phase evidence changed during capture",
            )
            _atomic_bytes(out_dir / filename, payload)
            published_entry[field] = {
                "path": filename,
                "sha256": entry[field]["sha256"],
            }
        runs.append(published_entry)
    _atomic_json(destination, {"protocol": EVIDENCE_INPUT_PROTOCOL, "runs": runs})
    return destination
