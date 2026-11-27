"""Spine evidence survives cached execution and a completely fresh process."""

import gc
import json
import subprocess
import sys
import weakref
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import (
    uk_registry,
    uk_spine_endpoint,
    uk_spine_graph,
    uk_spine_operation_inventory,
)
from microcosm.build.uk_runtime.graph_evidence import (
    add_uk_spine_gate_nodes,
    load_spine_stage_artifacts,
    spine_sidecar_evidence,
)
from microcosm.graph import ContentStore, compile_graph, load_source, run_graph
from test_support.paths import paths_for


def test_spine_evidence_replays_without_instantiating_original_transform(tmp_path):
    country = load_country_spec("uk")
    country = replace(
        country, sources=replace(country.sources, stages=country.sources.stages[:1])
    )
    graph = uk_spine_graph(country)
    compiled = compile_graph(graph)
    source = (
        paths_for("microcosm-graph").tests
        / "fixtures"
        / "parity"
        / "uk_spine"
        / "sources"
    )

    class Root:
        sampling = {"fraction": 0.1, "seed": 7}
        fit_weight_records = (
            SimpleNamespace(fit_name="fixture-fit", weight_kind="design"),
        )

        def run_with_sources(self, frame, sources):
            return load_source("csv-tables", sources["frs"])

        def checkpoint_metadata(self):
            return {
                "evidence": {"rows": 8},
                "replay_payload": {"classification": "fixture"},
            }

    store = ContentStore(tmp_path / "store")
    cold = run_graph(
        compiled,
        sources={"frs": source},
        store=store,
        kernels=uk_registry({"frs_spine": Root()}, graph=graph),
        resume="forbid",
    )
    expected = load_spine_stage_artifacts(cold, store, stage_names=("frs_spine",))
    warm = run_graph(
        compiled,
        sources={"frs": source},
        store=store,
        kernels=uk_registry(graph=graph),
        resume="require",
    )
    assert all(receipt.store_hit for receipt in warm.nodes.values())
    assert (
        load_spine_stage_artifacts(warm, store, stage_names=("frs_spine",)) == expected
    )
    path = tmp_path / "manifest.json"
    warm.save(path)
    script = """
import json, sys
from microcosm.graph import ContentStore, RunManifest
from microcosm.build.uk_runtime.graph_evidence import load_spine_stage_artifacts
store = ContentStore(sys.argv[2])
manifest = RunManifest.load(sys.argv[1], store)
print(json.dumps(load_spine_stage_artifacts(manifest, store, stage_names=('frs_spine',)), sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(store.root)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == expected
    sidecar = spine_sidecar_evidence(expected)
    assert sidecar["sampling"] == {"fraction": 0.1, "seed": 7}
    assert sidecar["fit_weight_records"]["frs_spine"][0]["weight_kind"] == "design"


def _with_outputs(frame, outputs):
    """``frame`` with a zero in each declared output it does not yet carry."""

    from microcosm.frame import Frame

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for owned in outputs:
        if owned.column not in tables[owned.entity]:
            tables[owned.entity][owned.column] = 0
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _object_bytes(store):
    """Every file the store holds, by its path under the store's objects."""

    return {
        path.relative_to(store.objects).as_posix(): path.read_bytes()
        for path in sorted(store.objects.rglob("*"))
        if path.is_file()
    }


def _result_keeping_stages(graph):
    """Fresh transforms of each shape that keeps a result, with their holders.

    The root and the age stage are frozen dataclasses written through
    ``object.__setattr__``, as the UK stages are. A graph source transform
    builds the relationships stage. The employment stage reads its fit-weight
    records from its result, as the SPI income stage does. The council tax
    stage has no checkpoint hook, so the snapshot reads its result itself.
    Every stage but that one sits behind an observation wrapper. A weak
    reference to each result a stage keeps shows whether anything holds it.
    """

    from microcosm.build.observation import ObservedTransform
    from microcosm.build.uk_runtime.spine_build import _GraphSourceTransform

    record = SimpleNamespace(fit_name="fixture-fit", weight_kind="design")
    outputs = {node.id: node.outputs for node in graph.nodes}
    results = []

    class Result(SimpleNamespace):
        """A stage's result. Unlike a bare namespace, it takes a weak reference."""

    def kept(**fields):
        result = Result(**fields)
        results.append(weakref.ref(result))
        return result

    def unrun():
        return RuntimeError("checkpoint metadata requires a completed stage run.")

    @dataclass(frozen=True)
    class Root:
        sampling = {"fraction": 1.0, "seed": 7}
        last_result: object | None = field(default=None, init=False)

        def run_with_sources(self, frame, sources):
            loaded = load_source("csv-tables", sources["frs"])
            object.__setattr__(self, "last_result", kept(frame=loaded))
            return loaded

        def checkpoint_metadata(self):
            if self.last_result is None:
                raise unrun()
            households = len(self.last_result.frame.table("household"))
            return {"evidence": {"households": households}}

    @dataclass(frozen=True)
    class Stage:
        outputs: tuple
        last_result: object | None = field(default=None, init=False)

        def __call__(self, frame):
            result = kept(frame=_with_outputs(frame, self.outputs))
            object.__setattr__(self, "last_result", result)
            return result.frame

        def checkpoint_metadata(self):
            if self.last_result is None:
                raise unrun()
            return {"evidence": {"people": len(self.last_result.frame.person)}}

    @dataclass
    class ResultRecords:
        outputs: tuple
        last_result: object | None = None

        @property
        def fit_weight_records(self):
            return () if self.last_result is None else self.last_result.records

        def __call__(self, frame):
            self.last_result = kept(
                frame=_with_outputs(frame, self.outputs), records=(record,)
            )
            return self.last_result.frame

        def checkpoint_metadata(self):
            if self.last_result is None:
                raise unrun()
            return {"evidence": {"fits": len(self.last_result.records)}}

    class ResultEvidence:
        def __init__(self, outputs):
            self.outputs = outputs
            self.last_result = None

        def __call__(self, frame):
            result = _with_outputs(frame, self.outputs)
            households = len(result.table("household"))
            self.last_result = kept(
                frame=result, evidence=lambda: {"households": households}
            )
            return result

    def observed(transform, stage):
        return ObservedTransform(
            transform, stage_id=stage, produced_column_count=1, observer=None
        )

    built = []

    def build_relationships(sources):
        built.append(ResultRecords(outputs["frs_relationships"]))
        return built[-1]

    root = Root()
    age = Stage(outputs["age_tail"])
    relationships = _GraphSourceTransform(build_relationships)
    employment = ResultRecords(outputs["frs_employment"])
    council_tax = ResultEvidence(outputs["frs_council_tax"])
    implementations = {
        "frs_spine": observed(root, "frs_spine"),
        "age_tail": observed(age, "age_tail"),
        "frs_relationships": observed(relationships, "frs_relationships"),
        "frs_employment": observed(employment, "frs_employment"),
        "frs_council_tax": council_tax,
    }

    def holders():
        # The relationships stage is the one the source transform built.
        [relationships_stage] = built
        return {
            "frs_spine": root,
            "age_tail": age,
            "frs_relationships": relationships_stage,
            "frs_employment": employment,
            "frs_council_tax": council_tax,
        }

    return implementations, holders, relationships, results


def test_released_stage_results_leave_every_stored_byte_and_no_result(tmp_path):
    """A run that frees each stage's kept result once its evidence is captured
    stores what a run that keeps them stores, byte for byte, and each result
    it frees dies at once (microcosm#931)."""

    country = load_country_spec("uk")
    country = replace(
        country, sources=replace(country.sources, stages=country.sources.stages[:5])
    )
    graph = uk_spine_graph(country)
    compiled = compile_graph(graph)
    source = (
        paths_for("microcosm-graph").tests
        / "fixtures"
        / "parity"
        / "uk_spine"
        / "sources"
    )
    runs = {}
    for release in (False, True):
        stages = _result_keeping_stages(graph)
        implementations, result_holders, source_built, results = stages
        run_store = ContentStore(tmp_path / f"release-{release}")
        # No garbage collection runs, so a result is dead only once nothing
        # references it, not after some collection.
        collecting = gc.isenabled()
        gc.disable()
        try:
            manifest = run_graph(
                compiled,
                sources={"frs": source},
                store=run_store,
                kernels=uk_registry(
                    implementations, graph=graph, release_stage_results=release
                ),
                resume="forbid",
            )
            alive = [result() is not None for result in results]
        finally:
            if collecting:
                gc.enable()
        runs[release] = (manifest, run_store, result_holders(), source_built, alive)
    kept, kept_store, kept_holders, _, kept_alive = runs[False]
    released, store, holders, relationships, released_alive = runs[True]

    # Both runs store the same objects under the same keys, byte for byte.
    assert released.key == kept.key
    assert released.nodes.keys() == kept.nodes.keys()
    for node_id, receipt in kept.nodes.items():
        assert released.nodes[node_id].key == receipt.key
    assert _object_bytes(store) == _object_bytes(kept_store)
    # Each stage keeps its result unless the run frees it, frozen ones too.
    # A freed result dies at once, so no wrapper holds it either.
    assert all(holder.last_result is not None for holder in kept_holders.values())
    assert all(holder.last_result is None for holder in holders.values())
    assert len(kept_alive) == len(released_alive) == len(implementations)
    assert all(kept_alive)
    assert not any(released_alive)
    # The source transform keeps the stage it built and the records it read
    # from it; only the result goes.
    assert relationships.transform is holders["frs_relationships"]
    assert relationships.fit_weight_records
    # The fit-weight records were captured before the release, including
    # those the employment stage could read only from its result.
    stored = load_spine_stage_artifacts(
        released, store, stage_names=uk_spine_endpoint(graph).stage_names
    )
    for stage in ("frs_relationships", "frs_employment"):
        assert stored[stage]["fit_weight_records"] == [
            {"fit_name": "fixture-fit", "weight_kind": "design"}
        ]
        assert stored[stage]["fit_weight_records_status"] == "present"
    assert holders["frs_employment"].fit_weight_records == ()
    assert stored["frs_council_tax"]["evidence"] == {
        "households": stored["frs_spine"]["evidence"]["households"]
    }


def test_spine_inventory_is_roster_derived_and_gates_bind_checkpoint_versions(
    monkeypatch,
):
    country = load_country_spec("uk")
    spine = uk_spine_graph(country)
    endpoint = uk_spine_endpoint(spine)
    inventory = uk_spine_operation_inventory(spine, country)
    assert tuple(row["stage"] for row in inventory) == endpoint.stage_names
    assert inventory[0]["node"] == "create_uk_frs"
    assert "normalization" in inventory[0]["coupling"]
    wealth = next(row for row in inventory if row["stage"] == "was_wealth")
    assert "Donor and recipient" in wealth["coupling"]
    graph = add_uk_spine_gate_nodes(spine, spec=country, engine_identity="test-engine")
    compiled = compile_graph(graph)
    assembled = graph.node("spine.gates.assembled")
    transferred = graph.node("spine.gates.transferred")
    # The assembled checkpoint follows area_brma, in frs_brma's place
    # (microcosm#931).
    assert assembled.population == compiled.versions["area_brma"]
    assert "frs_brma" not in compiled.versions
    assert assembled.population != compiled.versions["was_wealth"]
    assert transferred.population == endpoint.population
    assert {edge.producer for edge in transferred.artifact_inputs} >= {
        "spine.gates.assembled",
        "was_wealth",
    }
    # The band donor check recomputes a capped seating's seats from the
    # transferred phase's frame, so the gate reads the cells it decodes; the
    # executor adds the ids and memberships itself (microcosm#931).
    from microcosm.build.uk_runtime import stage_health

    read = {
        (cells.entity, column)
        for cells in transferred.inputs
        for column in cells.columns
    }
    assert {
        ("household", stage_health._BAND_DONOR_FLAG_COLUMN),
        ("household", stage_health._BAND_DONOR_BAND_COLUMN),
        ("household", stage_health._BAND_DONOR_CLONE_INDEX_COLUMN),
        ("person", stage_health._BAND_CARRIER_FLAG_COLUMN),
    } <= read
    # The gate admits whichever stage follows the BRMA checkpoint in the roster
    # (the SPI block since #1012 moved it ahead of was_wealth), as main's tool
    # does with UK_SPINE_ASSEMBLED_FINAL_STAGE. Since the HMRC leaves and the
    # annual income spells moved ahead of the geographic clones, which the
    # draws and the BRMA follow, that is the SPI support channel
    # (microcosm#931). The area stages run before the draws, and area_brma,
    # in frs_brma's place, closes the assembled phase.
    stages = endpoint.stage_names
    admitted = stages[stages.index("area_brma") + 1]
    assert admitted == "spi_support_channel"
    assert {
        "frs_hmrc_spine_leaves",
        "frs_annual_income_spells",
        "geographic_expand",
        "geographic_identity",
        "atomic_area_assign",
        "local_authority",
        "rfdr",
    } <= set(stages[: stages.index("area_brma")])
    assert "spine.gates.assembled" in compiled.predecessors["area_brma.checkpoint"]
    assert "spine.gates.assembled" in compiled.predecessors[admitted]
    assert graph.node(admitted).params["spine_gate_phase"] == "assembled"
    assert "spine_gate_phase" not in graph.node("was_wealth").params
    # Each phase keys on its own code: only the transferred phase evaluates
    # the energy rake, so it alone keys on the module that sets the level, and
    # the assembled gate, which every later spine stage waits on, does not
    # (the re-review of microcosm#1420, E; microcosm#931 ruling 6).
    from microcosm.build.uk_runtime import energy_pricing, gate_phase_codes

    codes = gate_phase_codes.UK_SPINE_GATE_PHASE_CODE
    assert transferred.params["transferred_code"] == codes["transferred"]()
    assert assembled.params["assembled_code"] == codes["assembled"]()
    assert "transferred_code" not in assembled.params
    assert "assembled_code" not in transferred.params
    hashed = []
    with monkeypatch.context() as patch:
        patch.setattr(
            gate_phase_codes,
            "source_hash",
            lambda *objects: hashed.append(objects) or "",
        )
        gate_phase_codes.uk_spine_transferred_only_code_hash.__wrapped__()
        gate_phase_codes.uk_spine_assembled_only_code_hash.__wrapped__()
    transferred_modules, assembled_modules = hashed
    assert energy_pricing in transferred_modules
    assert energy_pricing not in assembled_modules
    # The blocking posture travels with the gate and the admission it guards.
    assert assembled.params["synthetic_smoke"] is False
    assert graph.node(admitted).params["spine_gate_synthetic_smoke"] is False
    smoke = add_uk_spine_gate_nodes(
        spine, spec=country, engine_identity="test-engine", synthetic_smoke=True
    )
    assert smoke.node("spine.gates.assembled").params["synthetic_smoke"] is True
    assert smoke.node(admitted).params["spine_gate_synthetic_smoke"] is True
    with pytest.raises(ValueError, match="release candidate"):
        add_uk_spine_gate_nodes(
            spine,
            spec=country,
            engine_identity="test-engine",
            release_candidate=True,
            synthetic_smoke=True,
        )


def _assembled_report(status):
    from microcosm.build.gate_battery import (
        GateOutcome,
        GatePhaseReport,
        GateStatus,
        gate_phase_report_payload,
    )
    from microcosm.build.gates import GateResult
    from microcosm.build.uk_runtime.graph_evidence import uk_spine_gate_manifest

    gates = uk_spine_gate_manifest(load_country_spec("uk"))
    entries = tuple(entry for entry in gates.gates if entry.phase == "assembled")
    selected = next(
        entry
        for entry in entries
        if entry.criticality == "release_blocking" and not entry.evidence_absent_blocks
    )
    outcomes = []
    for entry in entries:
        state = status if entry == selected else GateStatus.PASSED
        evaluated = state in (GateStatus.PASSED, GateStatus.FAILED)
        outcomes.append(
            GateOutcome(
                entry,
                state,
                result=GateResult(
                    entry.gate,
                    state is GateStatus.PASSED,
                    () if state is GateStatus.PASSED else ("fixture failure",),
                )
                if evaluated
                else None,
                reason=None if evaluated else "fixture missing reference",
            )
        )
    return gate_phase_report_payload(
        GatePhaseReport("assembled", tuple(outcomes)), gates=gates
    ), selected.id


def test_assembled_admission_refuses_before_model_and_preserves_development_policy(
    tmp_path,
):
    from microcosm.build.gate_battery import GateStatus
    from microcosm.build.uk_runtime.graph_evidence import (
        require_uk_spine_gate_admission,
    )
    from microcosm.build.uk_runtime.graph_kernels import UKStageKernel

    report, failed = _assembled_report(GateStatus.FAILED)
    path = tmp_path / "stored-phase.json"
    path.write_text(json.dumps(report))
    context = SimpleNamespace(
        artifacts={"spine_gate": SimpleNamespace(payload=path.read_bytes())},
        params={
            "spine_gate_phase": "assembled",
            "spine_gate_release_candidate": False,
            "spine_gate_synthetic_smoke": False,
        },
    )

    class UntouchedModel:
        def __call__(self, frame):
            raise AssertionError("A blocked assembled spine must not run a donor model")

    with pytest.raises(ValueError, match=failed):
        UKStageKernel("was_wealth", UntouchedModel()).run(context)
    assert json.loads(path.read_bytes()) == report
    report, _ = _assembled_report(GateStatus.EVIDENCE_ABSENT)
    context.artifacts["spine_gate"].payload = json.dumps(report).encode()
    require_uk_spine_gate_admission(context)
    context.params["spine_gate_release_candidate"] = True
    with pytest.raises(ValueError, match="block downstream"):
        require_uk_spine_gate_admission(context)


@pytest.mark.parametrize("synthetic_smoke", [False, True])
def test_measure_admission_carries_the_transferred_gate_posture(synthetic_smoke):
    """The dense measures, the first dense node to read the spine's version,
    take the transferred gate's admission the retired sample node held
    (microcosm#931), with every parameter the admission check reads (found on
    the sample node by the first licensed end-to-end run:
    ``spine_gate_synthetic_smoke`` was missing there), and run after it."""
    from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, uk_full_graph

    country = load_country_spec("uk")
    gated = add_uk_spine_gate_nodes(
        uk_spine_graph(country),
        spec=country,
        engine_identity="test-engine",
        synthetic_smoke=synthetic_smoke,
    )
    graph = uk_full_graph(UKFullBuildConfig(calibration_year=2025), spine=gated).graph
    gate = graph.node("spine.gates.transferred")
    measures = graph.node("uk.full.measures")
    assert {edge.producer for edge in measures.artifact_inputs} >= {gate.id}
    assert measures.params["spine_gate_phase"] == "transferred"
    assert measures.params["spine_gate_release_candidate"] is bool(
        gate.params["release_candidate"]
    )
    assert measures.params["spine_gate_synthetic_smoke"] is synthetic_smoke
    assert gate.params["synthetic_smoke"] is synthetic_smoke
    compiled = compile_graph(graph)
    assert compiled.order.index(gate.id) < compiled.order.index("uk.full.measures")


def test_the_measures_check_their_admission_before_they_read_anything(monkeypatch):
    """The measures kernel runs the admission check first: a refused
    admission stops it before it reads a table, an artifact or a parameter
    (microcosm#931)."""
    from types import SimpleNamespace

    from microcosm.build.uk_runtime import graph_evidence, graph_targets

    class _RefusedError(Exception):
        pass

    def refuse(context, **kwargs):
        raise _RefusedError

    monkeypatch.setattr(graph_evidence, "require_uk_spine_gate_admission", refuse)
    context = SimpleNamespace(
        tables=_UnreadArtifacts(), artifacts=_UnreadArtifacts(), params={}
    )
    with pytest.raises(_RefusedError):
        graph_targets.UKFullMeasureKernel().run(context)


class _ArtifactReadError(Exception):
    """A spine gate read its node's artifacts."""


class _UnreadArtifacts(dict):
    """Stage evidence a refused spine gate must not read: any read raises."""

    def get(self, *args, **kwargs):
        raise _ArtifactReadError(args)

    def __getitem__(self, key):
        raise _ArtifactReadError(key)


def _spine_gate_context(phase: str, **changes):
    """A spine gate kernel and a context with the node's declared parameters,
    ``changes`` applied (a ``None`` value removes the parameter), and
    artifacts any read of which raises :class:`_ArtifactReadError`."""
    from microcosm.build.uk_runtime.graph_evidence import (
        UKSpineGateKernel,
        uk_spine_gate_manifest,
    )

    country = load_country_spec("uk")
    gated = add_uk_spine_gate_nodes(
        uk_spine_graph(country), spec=country, engine_identity="test-engine"
    )
    params = dict(gated.node(f"spine.gates.{phase}").params)
    for key, value in changes.items():
        if value is None:
            params.pop(key, None)
        else:
            params[key] = value
    kernel = UKSpineGateKernel(
        gates=uk_spine_gate_manifest(country),
        engine=None,
        engine_identity="test-engine",
    )
    return kernel, SimpleNamespace(params=params, artifacts=_UnreadArtifacts())


@pytest.mark.parametrize("phase", ["assembled", "transferred"])
@pytest.mark.parametrize("code", ["0" * 64, None])
def test_each_spine_gate_phase_refuses_a_stale_or_missing_phase_code(phase, code):
    """Each spine gate phase's node carries its own code, which the kernel
    checks before it reads anything of its node but its parameters, as the
    full gates check theirs (microcosm#931, ruled 9 Nov (late)): a spine gate
    node whose ``<phase>_code`` is stale or missing is refused whenever it
    runs, and no spine gate evaluates its phase under a key that names other
    phase code. A record stored under the node's key is reused without
    running the kernel, so the check does not guard a cache hit. The node's
    own declared code passes the check, and the kernel goes on to read its
    artifacts."""
    kernel, declared = _spine_gate_context(phase)
    with pytest.raises(_ArtifactReadError):
        kernel.run(declared)
    kernel, context = _spine_gate_context(phase, **{f"{phase}_code": code})
    with pytest.raises(ValueError, match=f"UK spine gate {phase}-only code"):
        kernel.run(context)


def test_each_spine_gate_node_carries_its_own_phase_code() -> None:
    """The assembled and transferred gate nodes each carry their phase's own
    code, and only their own (microcosm#931 ruling 6); the full gate nodes'
    are held in test_uk_graph_terminal."""
    from microcosm.build.uk_runtime.gate_phase_codes import UK_SPINE_GATE_PHASE_CODE
    from microcosm.build.uk_runtime.graph_evidence import uk_spine_gate_manifest

    country = load_country_spec("uk")
    gated = add_uk_spine_gate_nodes(
        uk_spine_graph(country), spec=country, engine_identity="test-engine"
    )
    phases = uk_spine_gate_manifest(country).phases
    assert set(phases) == {"assembled", "transferred"}
    for phase in phases:
        params = gated.node(f"spine.gates.{phase}").params
        code = UK_SPINE_GATE_PHASE_CODE.get(phase)
        assert {key for key in params if key.endswith("_code")} == (
            {f"{phase}_code"} if code is not None else set()
        ), phase
        if code is not None:
            assert params[f"{phase}_code"] == code(), phase


def _toy_geography():
    """The toy conditioned supports' bytes and the spine's geography block
    over them, as a synthetic fixture's build sidecar records it."""
    from microcosm.build.uk_runtime.country_adapter import (
        UK_SYNTHETIC_SUPPORT_REGISTER,
        uk_spine_geography_binding,
    )
    from test_support.microcosm_build.uk_conditioned_support_fixtures import (
        toy_conditioned_support_payloads,
    )

    payloads = toy_conditioned_support_payloads()
    return payloads, uk_spine_geography_binding(
        payloads, register=UK_SYNTHETIC_SUPPORT_REGISTER
    )


def test_the_spine_gates_carry_the_geography_block_and_read_the_supports():
    """Given the spine's geography block, each spine gate phase whose stages
    include the area draw carries the block as its ``spine_geography``
    parameter and reads the three conditioned supports as graph sources, as
    the area draw does, so the area mapping gates hold the area stages'
    records to it and re-derive the areas on those supports; without it the
    nodes are as before. A graph that does not declare the supports as
    sources is refused the block (microcosm#931)."""
    from microcosm.build.uk_runtime.graph_evidence import SPINE_GEOGRAPHY_PARAM

    country = load_country_spec("uk")
    _, block = _toy_geography()
    split = uk_spine_graph(country, source_mode="split")
    bound = add_uk_spine_gate_nodes(
        split, spec=country, engine_identity="test-engine", geography=block
    )
    unbound = add_uk_spine_gate_nodes(
        split, spec=country, engine_identity="test-engine"
    )
    supports = split.node("atomic_area_assign").sources
    assert len(supports) == 3
    for phase in ("assembled", "transferred"):
        node, bare = (graph.node(f"spine.gates.{phase}") for graph in (bound, unbound))
        assert node.sources == supports, phase
        assert json.loads(node.params[SPINE_GEOGRAPHY_PARAM]) == block, phase
        assert bare.sources == () and SPINE_GEOGRAPHY_PARAM not in bare.params
        assert dict(node.params) == {
            **bare.params,
            SPINE_GEOGRAPHY_PARAM: node.params[SPINE_GEOGRAPHY_PARAM],
        }
    compile_graph(bound)
    with pytest.raises(ValueError, match="does not declare as sources"):
        add_uk_spine_gate_nodes(
            uk_spine_graph(country),
            spec=country,
            engine_identity="test-engine",
            geography=block,
        )


def test_the_spine_gate_kernel_hands_the_block_and_the_supports_to_its_gates(
    tmp_path,
):
    """The kernel hands the gates of a node that carries the geography block
    the block and the bytes of the supports its graph sources name, under
    the names the area mapping check reads, and a node without the block
    neither (microcosm#931)."""
    from microcosm.build.uk_runtime.atomic_area_support import CONDITIONED_SOURCES
    from microcosm.build.uk_runtime.graph_evidence import (
        SPINE_GEOGRAPHY_PARAM,
        uk_spine_geography_artifacts,
    )
    from microcosm.build.uk_runtime.stage_health import (
        ATOMIC_AREA_SUPPORTS_ARTIFACT,
        SPINE_GEOGRAPHY_ARTIFACT,
    )

    payloads, block = _toy_geography()
    sources = {}
    for system, payload in payloads.items():
        path = tmp_path / f"{CONDITIONED_SOURCES[system]}.npz"
        path.write_bytes(payload)
        sources[CONDITIONED_SOURCES[system]] = path
    context = SimpleNamespace(
        params={SPINE_GEOGRAPHY_PARAM: json.dumps(block, sort_keys=True)},
        sources=sources,
    )
    assert uk_spine_geography_artifacts(context) == {
        SPINE_GEOGRAPHY_ARTIFACT: block,
        ATOMIC_AREA_SUPPORTS_ARTIFACT: payloads,
    }
    assert uk_spine_geography_artifacts(SimpleNamespace(params={}, sources={})) == {}


def test_the_spine_gate_kernel_keys_on_its_checks_not_the_energy_level(monkeypatch):
    # The re-review of microcosm#1420, E: one kernel hash serves both phases,
    # so it covers the modules both read and not energy_pricing, which only
    # the transferred phase's node param carries (microcosm#931 ruling 6).
    from microcosm.build import stochastic_assignment
    from microcosm.build.uk_runtime import (
        battery_bindings,
        energy_pricing,
        graph_evidence,
        stage_health,
    )

    seen = []
    real = graph_evidence.source_hash

    def recording(*objects):
        seen.append(objects)
        return real(*objects)

    monkeypatch.setattr(graph_evidence, "source_hash", recording)
    kernel = graph_evidence.UKSpineGateKernel(
        gates=load_country_spec("uk").gates, engine=None, engine_identity="test"
    )
    kernel.implementation_hash()
    (modules,) = seen
    assert {stage_health, battery_bindings, graph_evidence.gate_battery} <= set(modules)
    # The latent-attribute checks' binomial band lives in stochastic_assignment
    # (microcosm#931), and only the transferred phase declares those checks,
    # so its own code keys on it, not the hash both phases share.
    assert stochastic_assignment not in modules
    assert energy_pricing not in modules


def _hand_over_context(households: int):
    """A UK kernel's context as the executor hands it over: read-only views of
    an invented population's tables (main's microcosm#1724), with its
    household weights and strata."""

    import numpy as np
    import pandas as pd

    from microcosm.build.uk_runtime.national_frame import UK_NATIONAL_SCHEMA
    from microcosm.frame import Frame, WeightKind, Weights
    from microcosm.graph import executor

    ids = np.arange(1, households + 1, dtype=np.int64)

    def filler(entity: str) -> dict[str, np.ndarray]:
        # Column names are unique across the entity tables (the flattening rule).
        return {
            f"{entity}_column_{index}": np.linspace(0.0, 1.0, households)
            for index in range(40)
        }

    population = Frame(
        {
            "household": pd.DataFrame({"household_id": ids, **filler("household")}),
            "benunit": pd.DataFrame({"benunit_id": ids}),
            "person": pd.DataFrame(
                {
                    "person_id": ids * 10,
                    "person_benunit_id": ids,
                    "person_household_id": ids,
                    **filler("person"),
                }
            ),
        },
        UK_NATIONAL_SCHEMA,
        {"household": Weights(values=np.ones(households), kind=WeightKind.DESIGN)},
    )
    tables = {
        entity: executor._read_only_table(
            population.table(entity), list(population.table(entity).columns)
        )
        for entity in population.entities
    }
    return population, SimpleNamespace(
        tables=tables,
        weights={"household": population.weights_for("household")},
        strata=population.strata.copy(),
        params={"time_period": "2024"},
        node=SimpleNamespace(id="spine.gates.transferred"),
    )


def test_a_kernel_frame_is_one_copy_of_its_context():
    """_minimal_frame hands Frame the context's frozen tables and Frame makes
    the one copy. It used to deep-copy them first, so building a kernel's
    frame held two copies at once, three beside the stored population
    (microcosm#931)."""

    import tracemalloc

    import numpy as np

    from microcosm.build.uk_runtime import graph_kernels
    from microcosm.frame import MassChangeRecord

    population, context = _hand_over_context(20_000)
    size = sum(
        int(population.table(entity).memory_usage(index=False).sum())
        for entity in population.entities
    )
    record = MassChangeRecord(
        entity="household",
        old_total=20_000.0,
        new_total=20_000.0,
        declared_factor=1.0,
        reason="an invented reason",
    )
    gc.collect()
    tracemalloc.start()
    try:
        frame = graph_kernels._minimal_frame(
            context, mass_log=(record,), metadata={"time_period": "2025"}
        )
        held, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert held < 1.5 * size
    # The executor hands read-only views, one block per column (main's
    # microcosm#1724): pandas copies them and then joins the copies into one
    # block, so two copies are alive for a moment, no more than the frozen
    # copy and the frame's own were before. One is held once the frame is built.
    assert peak < 2.5 * size
    # The run's mass log and metadata replace the stage defaults, ...
    assert frame.mass_log == (record,)
    assert dict(frame.metadata) == {"time_period": "2025"}
    assert frame.weighted_entities == ("household",)
    # ... and the frame owns its tables: none of them is the context's.
    for entity in frame.entities:
        table = frame.table(entity)
        for column in table.columns:
            assert not np.shares_memory(
                table[column].to_numpy(), context.tables[entity][column].to_numpy()
            )
    default = graph_kernels._minimal_frame(context)
    assert default.mass_log == ()
    assert dict(default.metadata) == {"time_period": "2024"}


def test_the_spine_gate_kernel_builds_one_frame():
    """The spine gate kernel's run builds its frame through one _minimal_frame
    call that brings the run's mass log and metadata, and builds no second
    Frame from it, which held one more copy of the population while the
    gates ran (microcosm#931)."""

    import ast
    import inspect
    import textwrap

    from microcosm.build.uk_runtime.graph_evidence import UKSpineGateKernel

    tree = ast.parse(textwrap.dedent(inspect.getsource(UKSpineGateKernel.run)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [ast.unparse(call.func) for call in calls]
    assert names.count("_minimal_frame") == 1
    assert "Frame" not in names
    (minimal,) = [call for call in calls if ast.unparse(call.func) == "_minimal_frame"]
    assert {keyword.arg for keyword in minimal.keywords} == {"mass_log", "metadata"}
