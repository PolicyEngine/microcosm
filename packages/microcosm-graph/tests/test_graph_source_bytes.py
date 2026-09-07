"""A raw-byte source becomes a typed artifact without inventing a population.

A lookup table is not a population, so nothing here asks the graph to pretend
it is one. The only population in these graphs comes from the toy country's
genuine ``CREATE`` fixture; an ordinary ``StructuralDelta.NONE`` node reads an
NPZ of age-band multipliers through the ``raw-bytes-v1`` codec and publishes it
as a nominal typed artifact, and a later kernel consumes that artifact. The
executor's own source machinery — content identity, the availability
pre-check, and the post-run mutation check — is exercised unchanged.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.graph import (
    SOURCE_CODECS,
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    NodeRejectedError,
    Owned,
    Slice,
    SourceRef,
    StoreUnavailableError,
    compile_graph,
    load_source_bytes,
    run_graph,
    source_hash,
)
from microcosm.graph.codecs import load_csv_tables, load_raw_bytes
from microcosm.graph.kernel import KernelRole

spec = importlib.util.spec_from_file_location(
    "_source_bytes_toy", Path(__file__).with_name("_toy.py")
)
toy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = toy
spec.loader.exec_module(toy)

LOOKUP = ArtifactType("test.age-multiplier-lookup", 1)
MULTIPLIERS = SourceRef(
    "multipliers",
    "raw-bytes-v1",
    description="an NPZ lookup of multipliers by age band",
)
BOUNDS = np.asarray([0, 18, 65], dtype=np.int64)


def write_lookup(path: Path, multipliers=(0.5, 1.0, 1.25)) -> Path:
    """Write the NPZ lookup table a real stage would be handed."""
    np.savez(
        path,
        lower_bound=BOUNDS,
        multiplier=np.asarray(multipliers, dtype=np.float64),
    )
    return path


def decode(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Validate the lookup an importer read; nothing about it is assumed."""
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            bounds = np.asarray(archive["lower_bound"])
            multiplier = np.asarray(archive["multiplier"])
    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"The multiplier lookup is not readable: {error}") from error
    if bounds.dtype != np.int64 or multiplier.dtype != np.float64:
        raise ValueError("The multiplier lookup does not carry the declared dtypes.")
    if bounds.shape != multiplier.shape or bounds.ndim != 1 or bounds.size == 0:
        raise ValueError("The multiplier lookup is misshaped.")
    if bounds[0] != 0 or np.any(np.diff(bounds) <= 0):
        raise ValueError("The multiplier lookup bounds must start at 0 and increase.")
    return bounds, multiplier


def expected_multipliers(source: Path, multipliers=(0.5, 1.0, 1.25)) -> np.ndarray:
    ages = toy.read_toy_frame(source).table("person")["age"].to_numpy()
    band = np.searchsorted(BOUNDS, ages, side="right") - 1
    return np.asarray(multipliers, dtype=np.float64)[band]


class ImportMultipliers(KernelBase):
    """Read the lookup's real bytes and publish them as a typed artifact."""

    ref = "lookup.import@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, dependencies=("numpy",))
    calls = 0

    def implementation_hash(self) -> str:
        return source_hash(
            type(self), load_source_bytes, dependencies=self.capabilities.dependencies
        )

    def run(self, context):
        self.calls += 1
        payload = load_source_bytes("raw-bytes-v1", context.sources["multipliers"])
        assert isinstance(payload, bytes)
        bounds, multiplier = decode(payload)
        table = json.dumps(
            {
                "schema_version": LOOKUP.schema_version,
                "lower_bound": bounds.tolist(),
                "multiplier": multiplier.tolist(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return KernelResult(
            artifacts={"table": table},
            receipt={"bands": int(bounds.size)},
        )


class GateMultipliers(ImportMultipliers):
    """The same import as a typed gate, so a bad lookup stays evidence.

    A gate turns *any* exception into a verdict (``executor._failed_gate_result``),
    including the fatal ``StoreUnavailable`` an uninstalled codec or a missing
    dependency raises — that arm is checked before the fatal one. So a gate is
    the wrong node to be the first reader of an optional dependency; it is the
    right node for content it is judging, which is what this one does.
    """

    ref = "lookup.gate@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, role=KernelRole.GATE, dependencies=("numpy",)
    )
    calls = 0

    def run(self, context):
        result = super().run(context)
        ids = pd.Index(context.tables["release"]["release_id"], name="release_id")
        return KernelResult(
            columns={
                ("release", "lookup_verdict"): pd.Series(
                    "pass", index=ids, dtype="string"
                )
            },
            artifacts=dict(result.artifacts),
            receipt={"outcome": "pass", **dict(result.receipt)},
        )


class MutateWhileImporting(ImportMultipliers):
    """Rewrite the source after decoding it, which the executor must catch."""

    ref = "lookup.import.mutating@1"
    calls = 0

    def run(self, context):
        result = super().run(context)
        source = context.sources["multipliers"]
        source.write_bytes(source.read_bytes() + b"\0")
        return result


class ApplyMultipliers(KernelBase):
    """Look each person's age up in the imported table."""

    ref = "lookup.apply@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, dependencies=("numpy", "pandas")
    )
    calls = 0

    def run(self, context):
        self.calls += 1
        value = context.artifacts["lookup"]
        assert isinstance(value, ArtifactValue) and value.type == LOOKUP
        document = json.loads(value.payload.decode("utf-8"))
        bounds = np.asarray(document["lower_bound"], dtype=np.int64)
        multiplier = np.asarray(document["multiplier"], dtype=np.float64)
        table = context.tables["person"]
        band = np.searchsorted(bounds, table["age"].to_numpy(), side="right") - 1
        return KernelResult(
            columns={
                ("person", "age_multiplier"): pd.Series(
                    multiplier[band],
                    index=pd.Index(table.person_id, name="person_id"),
                    dtype="float64",
                )
            }
        )


def lookup_graph(
    *, importer: type[KernelBase] = ImportMultipliers, codec: str = ""
) -> Graph:
    """The toy country plus an import node and one consumer of its artifact."""
    gate = importer is GateMultipliers
    source = MULTIPLIERS if not codec else SourceRef("multipliers", codec)
    return Graph(
        "toy",
        (toy.SOURCE, source),
        (
            toy.CREATE,
            Node(
                "import_multipliers",
                importer.ref,
                sources=("multipliers",),
                outputs=(
                    (Owned("release", "lookup_verdict", "string"),) if gate else ()
                ),
                artifact_outputs=(ArtifactOutput("table", LOOKUP),),
                description="decode the multiplier lookup into a typed artifact",
            ),
            Node(
                "apply_multipliers",
                ApplyMultipliers.ref,
                inputs=(Slice("person", ("age",)),),
                outputs=(Owned("person", "age_multiplier", "float64"),),
                artifact_inputs=(
                    ArtifactInput("lookup", "import_multipliers", "table", LOOKUP),
                ),
            ),
        ),
    )


def registry_for(
    importer: type[KernelBase] = ImportMultipliers,
) -> tuple[KernelRegistry, KernelBase, ApplyMultipliers]:
    registry = toy.toy_registry()
    import_kernel, apply_kernel = importer(), ApplyMultipliers()
    registry.register(import_kernel)
    registry.register(apply_kernel)
    return registry, import_kernel, apply_kernel


def fixture_sources(tmp_path: Path, multipliers=(0.5, 1.0, 1.25)) -> dict[str, Path]:
    sources = toy.toy_sources(tmp_path)
    sources["multipliers"] = write_lookup(tmp_path / "multipliers.npz", multipliers)
    return sources


def test_raw_byte_source_reaches_a_typed_artifact_and_its_consumer(tmp_path) -> None:
    compiled = compile_graph(lookup_graph())
    sources = fixture_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    registry, importer, apply_kernel = registry_for()

    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)

    assert importer.calls == 1 and apply_kernel.calls == 1
    # The only population is the fixture's; the import node declares none.
    assert set(cold.populations) == {"survey"}
    assert cold.nodes["import_multipliers"].frame_key is None
    values = cold.population("survey").table("person")["age_multiplier"].to_numpy()
    np.testing.assert_array_equal(values, expected_multipliers(sources["survey"]))
    # Every band of the fixture is populated, so the lookup really discriminates.
    assert set(values) == {0.5, 1.0, 1.25}
    # The typed artifact carries what the codec actually read off disk.
    key = cold.nodes["import_multipliers"].opaque_artifacts["table"]
    stored = json.loads(store.load_bytes(key).decode("utf-8"))
    payload = load_source_bytes("raw-bytes-v1", sources["multipliers"])
    bounds, multiplier = decode(payload)
    assert payload == sources["multipliers"].read_bytes()
    assert stored["lower_bound"] == bounds.tolist() == [0, 18, 65]
    assert stored["multiplier"] == multiplier.tolist() == [0.5, 1.0, 1.25]
    assert cold.nodes["import_multipliers"].receipt["bands"] == 3

    replay_registry, replay_importer, replay_apply = registry_for()
    replay = run_graph(
        compiled,
        sources=sources,
        store=store,
        kernels=replay_registry,
        resume="require",
    )

    assert replay.key == cold.key
    assert all(node.hit for node in replay.nodes.values())
    assert replay_importer.calls == 0 and replay_apply.calls == 0
    assert toy.total_calls(replay_registry) == 0


def test_changed_lookup_bytes_recompute_the_importer_and_its_dependents(
    tmp_path,
) -> None:
    compiled = compile_graph(lookup_graph())
    sources = fixture_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    registry, _, _ = registry_for()
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    before = cold.population("survey").table("person")["age_multiplier"].to_numpy()

    write_lookup(sources["multipliers"], (0.25, 1.0, 2.0))
    changed_registry, importer, apply_kernel = registry_for()
    changed = run_graph(
        compiled, sources=sources, store=store, kernels=changed_registry
    )

    assert importer.calls == 1 and apply_kernel.calls == 1
    assert changed.key != cold.key
    for node_id in ("import_multipliers", "apply_multipliers"):
        assert not changed.nodes[node_id].hit
        assert changed.nodes[node_id].key != cold.nodes[node_id].key
    # Exactly the consumers: the population the lookup never touched is reused.
    assert changed.nodes["survey"].hit
    assert changed.nodes["survey"].key == cold.nodes["survey"].key
    after = changed.population("survey").table("person")["age_multiplier"].to_numpy()
    np.testing.assert_array_equal(
        after, expected_multipliers(sources["survey"], (0.25, 1.0, 2.0))
    )
    assert not np.array_equal(before, after)


def test_malformed_lookup_bytes_reject_an_ordinary_import_node(tmp_path) -> None:
    compiled = compile_graph(lookup_graph())
    sources = fixture_sources(tmp_path)
    sources["multipliers"].write_bytes(b"this is not an NPZ archive")
    store = ContentStore(tmp_path / "store")
    registry, importer, apply_kernel = registry_for()

    with pytest.raises(NodeRejectedError, match="import_multipliers") as rejected:
        run_graph(compiled, sources=sources, store=store, kernels=registry)

    # Rejected for the content, not for some other contract breach.
    assert "not readable" in str(rejected.value)
    assert importer.calls == 1 and apply_kernel.calls == 0


def test_malformed_lookup_bytes_leave_a_typed_gate_as_evidence(tmp_path) -> None:
    compiled = compile_graph(lookup_graph(importer=GateMultipliers))
    sources = fixture_sources(tmp_path)
    sources["multipliers"].write_bytes(b"this is not an NPZ archive")
    store = ContentStore(tmp_path / "store")
    registry, gate, apply_kernel = registry_for(GateMultipliers)

    manifest = run_graph(compiled, sources=sources, store=store, kernels=registry)

    gate_receipt = manifest.nodes["import_multipliers"]
    assert gate.calls == 1 and apply_kernel.calls == 0
    assert gate_receipt.receipt["outcome"] == "fail"
    assert gate_receipt.receipt["execution"]["state"] == "gate_exception"
    assert gate_receipt.receipt["execution"]["unavailable_artifacts"] == ("table",)
    assert gate_receipt.receipt["evidence"]["exception_type"] == "ValueError"
    assert "not readable" in gate_receipt.receipt["evidence"]["message"]
    assert not gate_receipt.opaque_artifacts
    verdicts = manifest.population("survey").table("release")["lookup_verdict"]
    assert set(verdicts) == {"fail"}
    consumer = manifest.nodes["apply_multipliers"]
    assert consumer.receipt["outcome"] == "unreached"
    assert consumer.receipt["execution"]["blocked_by"] == {
        "import_multipliers": gate_receipt.key
    }
    assert "age_multiplier" not in manifest.population("survey").table("person")


def test_a_source_mutated_while_the_kernel_runs_is_rejected(tmp_path) -> None:
    compiled = compile_graph(lookup_graph(importer=MutateWhileImporting))
    sources = fixture_sources(tmp_path)
    store = ContentStore(tmp_path / "store")
    registry, importer, apply_kernel = registry_for(MutateWhileImporting)

    with pytest.raises(NodeRejectedError, match="changed source"):
        run_graph(compiled, sources=sources, store=store, kernels=registry)

    assert importer.calls == 1 and apply_kernel.calls == 0


def test_codec_availability_is_resolved_before_any_kernel_runs(tmp_path) -> None:
    sources = fixture_sources(tmp_path)
    unknown = compile_graph(lookup_graph(codec="raw-bytes-v2"))
    registry, importer, apply_kernel = registry_for()

    with pytest.raises(StoreUnavailableError, match="raw-bytes-v2"):
        run_graph(
            unknown,
            sources=sources,
            store=ContentStore(tmp_path / "unknown"),
            kernels=registry,
        )
    assert importer.calls == 0 and toy.total_calls(registry) == 0

    compiled = compile_graph(lookup_graph())
    configured = ContentStore(tmp_path / "registry", codecs=SOURCE_CODECS)
    run_graph(compiled, sources=sources, store=configured, kernels=registry)
    assert importer.calls == 1 and apply_kernel.calls == 1

    # A Frame-only snapshot does not carry the raw-byte codec, and the run
    # refuses rather than reaching a kernel that could not have read it.
    frame_only = ContentStore(
        tmp_path / "frame-only", codecs=SOURCE_CODECS.as_mapping()
    )
    with pytest.raises(StoreUnavailableError, match="raw-bytes-v1"):
        run_graph(compiled, sources=sources, store=frame_only, kernels=registry)
    assert importer.calls == 1

    # A plain mapping is its own arm of the executor's availability check, and
    # one that carries the byte codec resolves it like any other.
    mapping = ContentStore(
        tmp_path / "mapping",
        codecs={"csv-tables": load_csv_tables, "raw-bytes-v1": load_raw_bytes},
    )
    run_graph(compiled, sources=sources, store=mapping, kernels=registry)
    assert importer.calls == 2 and apply_kernel.calls == 2
