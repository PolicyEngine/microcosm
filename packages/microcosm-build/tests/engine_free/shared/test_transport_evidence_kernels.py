"""Transport release-evidence producers, their shared contract and their gates.

G8-w4 gives ``release_input_coverage`` two graph producers
(``transport.donor_receipt@1`` and ``transport.input_closure@1``) whose typed
artifacts carry an identity header the battery checks against its own
declared edges, frame and the implementation hash it computes by module name
(:mod:`microcosm.build.transport.evidence_contract`). These tests run both
producers on hand-built contexts over the composed toy package, hold the
closure partition to ``InputClosure.check`` and the committed NZ and BE
engine-generated surfaces, and run each mutant the brief names.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import GatesManifest
from microcosm.build.gate_battery import (
    EvidenceContext,
    GateStatus,
    evaluate_phase,
)
from microcosm.build.gates import GateResult
from microcosm.build.transport import (
    binding_identity,
    compose,
    evidence_contract,
    evidence_kernels,
)
from microcosm.build.transport import (
    registry as registry_module,
)
from microcosm.build.transport.evidence_contract import (
    AXIOM_INPUT_CLOSURE,
    CLOSURE_CLASSES,
    DONOR_ARTIFACT_RECEIPT,
    DONOR_RECEIPT_REF,
    DONOR_RECEIPT_TYPE,
    EVIDENCE_KERNEL_MODULES,
    GRAPH_EVIDENCE,
    INPUT_CLOSURE_REF,
    INPUT_CLOSURE_TYPE,
    binding_relation_inputs,
    checked_header,
    closure_partition,
    decode_donor_receipt,
    decode_input_closure,
    evidence_implementation_hash,
    partition_counts,
    unreached_refs,
    without_unreached,
)
from microcosm.build.transport.evidence_kernels import (
    TRANSPORT_DONOR_RECEIPT,
    TransportInputClosureKernel,
)
from microcosm.build.transport.gate_bindings import TRANSPORT_GATE_REGISTRY
from microcosm.build.transport.gate_kernels import GateBatteryKernel, _graph_evidence
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import axiom_concept_mapping
from microcosm.frame.adapters.axiom_input_surface import (
    load_axiom_input_surface,
)
from microcosm.frame.concept_mapping import ConceptMapping, InputRef
from microcosm.frame.input_closure import InputClosure
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactValue,
    KernelContext,
    Node,
    Slice,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_composed import make_composed_fixture
from test_support.paths import paths_for

_BUILD = paths_for("microcosm-build")
_FRAME = paths_for("microcosm-frame")
NZ = _BUILD.package / "src/microcosm/build/nz"
NZ_SURFACE = _FRAME.tests / "fixtures/axiom_input_surfaces/nz.json"
BE_SURFACE = _FRAME.tests / "fixtures/axiom_input_surfaces/be.json"
FTC = "nz/statutes/income_tax/family_scheme/tax_credits.yaml"
PROPERTY = settings(max_examples=60, deadline=None, derandomize=True)
SCHEMA = EntitySchema(group_entities=("household", "family"))
GATE = "release_input_coverage"
BATTERY = "toy.gates.terminal"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text(document: object) -> str:
    return canonical_json(document).decode("utf-8")


# ---------------------------------------------------------------------------
# The composed toy package and a frame carrying its encoded columns
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return make_composed_fixture(tmp_path_factory.mktemp("evidence") / "toy")


def _frame(closure: dict, *, shift: float = 0.0) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.array([1, 2, 3], dtype=np.int64),
            "person_household_id": np.array([10, 10, 20], dtype=np.int64),
            "person_family_id": np.array([100, 100, 200], dtype=np.int64),
        }
    )
    household = pd.DataFrame({"household_id": np.array([10, 20], dtype=np.int64)})
    family = pd.DataFrame({"family_id": np.array([100, 200], dtype=np.int64)})
    tables = {"person": person, "household": household, "family": family}
    engine_to_frame = {"Person": "person", "Family": "family", "Household": "household"}
    for entry in closure["entries"]:
        if entry.get("encoded_by") == "graph_node":
            continue
        table = tables[engine_to_frame[entry["entity"]]]
        table[entry["input"]] = np.arange(len(table), dtype=np.float64) + shift
    return Frame(
        tables,
        SCHEMA,
        {"household": Weights(np.ones(2), WeightKind.DESIGN)},
        metadata={"content_basis": "transport", "donor_time_period": 2024},
    )


def _context(
    frame: Frame,
    node: Node,
    *,
    sources=None,
    artifacts=None,
) -> KernelContext:
    tables = {}
    for item in node.inputs:
        table = frame.table(item.entity)
        structural = [SCHEMA.entity_id_column(item.entity)]
        if item.entity == "person":
            structural += [SCHEMA.membership_column(g) for g in SCHEMA.group_entities]
        tables[item.entity] = table.loc[
            :, list(dict.fromkeys((*structural, *item.columns)))
        ]
    return KernelContext(
        node=node,
        tables=tables,
        weights={},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
        sources=sources or {},
        artifacts=artifacts or {},
        frame_metadata=frame.metadata,
    )


def _closure_node(toy, *, node_id="toy.evidence.input_closure", spec=None):
    spec = spec or toy.spec
    resources = spec["resources"]
    closure = resources["axiom_input_closure"]
    engine_to_frame = {"Person": "person", "Family": "family"}
    columns: dict[str, list[str]] = {}
    for entry in closure["entries"]:
        if entry.get("encoded_by") != "graph_node":
            columns.setdefault(engine_to_frame[entry["entity"]], []).append(
                entry["input"]
            )
    return Node(
        node_id,
        INPUT_CLOSURE_REF,
        inputs=tuple(Slice(e, tuple(c)) for e, c in sorted(columns.items())),
        params={
            "closure": _text(closure),
            "closure_sha256": _sha(canonical_json(closure)),
            "mapping": _text(resources["concept_mapping"]),
            "mapping_sha256": _sha(canonical_json(resources["concept_mapping"])),
            "bindings": _text(resources["axiom_rules_bindings"]),
            "bindings_sha256": _sha(canonical_json(resources["axiom_rules_bindings"])),
            "engine_refs": _text({"engine_refs": dict(toy.registry.engine_refs)}),
        },
        artifact_outputs=(ArtifactOutput(AXIOM_INPUT_CLOSURE, INPUT_CLOSURE_TYPE),),
    )


def _closure_kernel(toy) -> TransportInputClosureKernel:
    return TransportInputClosureKernel(toy.registry.engines)


def _run_closure(toy, *, spec=None, frame=None, node_id="toy.evidence.input_closure"):
    spec = spec or toy.spec
    node = _closure_node(toy, node_id=node_id, spec=spec)
    frame = frame or _frame(spec["resources"]["axiom_input_closure"])
    result = _closure_kernel(toy).run(_context(frame, node))
    return result.artifacts[AXIOM_INPUT_CLOSURE]


def _stages(sha: str, size: int, *, copies: int = 1) -> dict:
    artifact = {
        "locator": "donor.h5",
        "revision": "toy-revision",
        "sha256": sha,
        "size_bytes": size,
    }
    return {
        "version": 1,
        "stages": [{"stage": "load_donor", "artifacts": [artifact] * copies}],
    }


def _receipt_node(
    sha: str, size: int, *, copies: int = 1, node_id="toy.evidence.donor"
):
    stages = _stages(sha, size, copies=copies)
    return Node(
        node_id,
        DONOR_RECEIPT_REF,
        sources=("donor_pool",),
        params={
            "donor_source": "donor_pool",
            "donor_sha256": sha,
            "donor_size": size,
            "donor_locator": "donor.h5",
            "donor_revision": "toy-revision",
            "source_stages": _text(stages),
            "source_stages_sha256": _sha(canonical_json(stages)),
        },
        artifact_outputs=(ArtifactOutput(DONOR_ARTIFACT_RECEIPT, DONOR_RECEIPT_TYPE),),
    )


@pytest.fixture(scope="module")
def donor(tmp_path_factory):
    path = tmp_path_factory.mktemp("donor") / "donor.h5"
    path.write_bytes(b"synthetic donor bytes\n" * 1000)
    return path


def _run_receipt(donor: Path, frame: Frame, *, node=None, **kwargs) -> bytes:
    payload = donor.read_bytes()
    node = node or _receipt_node(_sha(payload), len(payload), **kwargs)
    context = _context(frame, node, sources={"donor_pool": donor})
    return TRANSPORT_DONOR_RECEIPT.run(context).artifacts[DONOR_ARTIFACT_RECEIPT]


def _edges(
    receipt: bytes,
    closure: bytes,
    *,
    receipt_node="toy.evidence.donor",
    closure_node="toy.evidence.input_closure",
) -> dict:
    """The ``graph_evidence`` a battery declaring both edges would build."""

    return {
        DONOR_ARTIFACT_RECEIPT: {
            "producer": receipt_node,
            "artifact": DONOR_ARTIFACT_RECEIPT,
            "type": [DONOR_RECEIPT_TYPE.name, DONOR_RECEIPT_TYPE.schema_version],
            "sha256": _sha(receipt),
            "producer_key": "c" * 64,
        },
        AXIOM_INPUT_CLOSURE: {
            "producer": closure_node,
            "artifact": AXIOM_INPUT_CLOSURE,
            "type": [INPUT_CLOSURE_TYPE.name, INPUT_CLOSURE_TYPE.schema_version],
            "sha256": _sha(closure),
            "producer_key": "d" * 64,
        },
    }


def _entry(parameters=None):
    return GatesManifest.from_mapping(
        {
            "country": "xx",
            "version": 1,
            "policy": "Synthetic evidence gates.",
            "phases": ["terminal"],
            "gates": [
                {
                    "id": "toy_release_input_coverage",
                    "gate": GATE,
                    "phase": "terminal",
                    "criticality": "release_blocking",
                    "evidence_absent_blocks": True,
                    "parameters": {"allow_engine_optional": False}
                    if parameters is None
                    else parameters,
                }
            ],
        },
        country="xx",
    )


def _evaluate(artifacts, frame, *, registry=TRANSPORT_GATE_REGISTRY, parameters=None):
    report = evaluate_phase(
        _entry(parameters),
        "terminal",
        EvidenceContext(frame=frame, artifacts=artifacts),
        registry=registry,
    )
    (outcome,) = report.outcomes
    return outcome


def _decoded(receipt: bytes, closure: bytes, edges=None) -> dict:
    return {
        DONOR_ARTIFACT_RECEIPT: decode_donor_receipt(receipt),
        AXIOM_INPUT_CLOSURE: decode_input_closure(closure),
        GRAPH_EVIDENCE: MappingProxyType(edges or _edges(receipt, closure)),
    }


@pytest.fixture(scope="module")
def evidence(toy, donor):
    frame = _frame(toy.spec["resources"]["axiom_input_closure"])
    receipt = _run_receipt(donor, frame)
    closure = _run_closure(toy, frame=frame)
    return frame, receipt, closure


def _forged(payload: bytes, edit) -> bytes:
    document = json.loads(payload)
    edit(document)
    return canonical_json(document)


# ---------------------------------------------------------------------------
# Implementation hashes by module name (H42)
# ---------------------------------------------------------------------------


class TestImplementationHashes:
    @pytest.mark.parametrize("ref", sorted(EVIDENCE_KERNEL_MODULES))
    def test_the_leaf_hash_is_each_kernels_hash(self, toy, ref) -> None:
        kernel = toy.registry.kernels.get(ref)
        assert kernel.ref == ref
        assert evidence_implementation_hash(ref) == kernel.implementation_hash()

    @pytest.mark.parametrize("ref", sorted(EVIDENCE_KERNEL_MODULES))
    def test_each_list_starts_with_the_kernel_module_and_ends_with_the_leaf(
        self, ref
    ) -> None:
        modules = EVIDENCE_KERNEL_MODULES[ref]
        assert modules[0] == evidence_kernels.__name__
        assert modules[-1] == evidence_contract.__name__
        assert len(set(modules)) == len(modules)

    def test_the_lists_are_pinned(self) -> None:
        assert dict(EVIDENCE_KERNEL_MODULES) == {
            DONOR_RECEIPT_REF: (
                "microcosm.build.transport.evidence_kernels",
                "microcosm.graph.canonical",
                "microcosm.build.transport.evidence_contract",
            ),
            INPUT_CLOSURE_REF: (
                "microcosm.build.transport.evidence_kernels",
                "microcosm.frame.input_closure",
                "microcosm.frame.concept_mapping",
                "microcosm.frame.concepts",
                "microcosm.frame.adapters.axiom_input_surface",
                "microcosm.frame.adapters.axiom",
                # S2: the producer calls schema.entity_id_column.
                "microcosm.frame.schema",
                "microcosm.graph.canonical",
                "microcosm.build.transport.evidence_contract",
            ),
            "transport.weights_audit@1": (
                "microcosm.build.transport.evidence_kernels",
                "microcosm.build.transport.weight_quality",
                "microcosm.calibrate.artifacts",
                "microcosm.calibrate.matrix",
                "microcosm.calibrate.registry",
                "microcosm.calibrate.solve",
                # S2: the trace finds these in run(); the list missed them.
                "microcosm.calibrate.target",
                "microcosm.frame.schema",
                "microcosm.frame.weights",
                "microcosm.graph.canonical",
                "microcosm.build.transport.evidence_contract",
            ),
        }

    def test_the_leaf_hash_imports_no_kernel_module(self) -> None:
        script = (
            "import sys\n"
            "from microcosm.build.transport import evidence_contract as leaf\n"
            "for ref in leaf.EVIDENCE_KERNEL_MODULES:\n"
            "    leaf.evidence_implementation_hash(ref)\n"
            "print('microcosm.build.transport.evidence_kernels' in sys.modules,"
            " 'microcosm.build.transport.registry' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert result.stdout.split() == ["False", "False"]

    def test_the_leaf_imports_no_kernel_module_even_for_type_hints(self) -> None:
        tree = ast.parse(Path(evidence_contract.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert all(
            not name.startswith(".") and not name.startswith("microcosm.build")
            for name in imported
        ), imported

    def test_a_dropped_module_moves_the_hash_and_the_header_fails(
        self, monkeypatch, evidence
    ) -> None:
        # Mutant: a kernel's module list loses one module. The battery
        # computes a different hash, so the stored header no longer matches.
        frame, receipt, closure = evidence
        before = evidence_implementation_hash(INPUT_CLOSURE_REF)
        shortened = dict(EVIDENCE_KERNEL_MODULES)
        shortened[INPUT_CLOSURE_REF] = shortened[INPUT_CLOSURE_REF][1:]
        monkeypatch.setattr(
            evidence_contract, "EVIDENCE_KERNEL_MODULES", MappingProxyType(shortened)
        )
        assert evidence_implementation_hash(INPUT_CLOSURE_REF) != before
        failures = checked_header(
            AXIOM_INPUT_CLOSURE,
            decode_input_closure(closure),
            _edges(receipt, closure),
            frame,
            kernel=INPUT_CLOSURE_REF,
        )
        assert any("implementation hash" in line for line in failures)

    def test_the_order_of_the_list_enters_the_hash(self, monkeypatch) -> None:
        before = evidence_implementation_hash(INPUT_CLOSURE_REF)
        swapped = dict(EVIDENCE_KERNEL_MODULES)
        first, second, *rest = swapped[INPUT_CLOSURE_REF]
        swapped[INPUT_CLOSURE_REF] = (second, first, *rest)
        monkeypatch.setattr(
            evidence_contract, "EVIDENCE_KERNEL_MODULES", MappingProxyType(swapped)
        )
        assert evidence_implementation_hash(INPUT_CLOSURE_REF) != before

    def test_a_subclass_cannot_move_the_hash(self, toy) -> None:
        class Edited(TransportInputClosureKernel):
            def run(self, context):  # pragma: no cover - never run
                raise AssertionError

        assert (
            Edited({}).implementation_hash()
            == _closure_kernel(toy).implementation_hash()
        )

    def test_the_batterys_binding_identity_adds_only_the_leaf_and_its_imports(
        self,
    ) -> None:
        modules = set()
        for name, binding in TRANSPORT_GATE_REGISTRY.items():
            _, digests = binding_identity.describe(binding, path=name)
            modules |= set(digests)
        leaf = set(binding_identity.source_closure([evidence_contract.__name__]))
        without = modules - leaf
        assert evidence_contract.__name__ in modules
        assert not {
            name
            for name in modules
            if name.startswith("microcosm.build.transport.")
            and name
            not in {
                "microcosm.build.transport.gate_bindings",
                "microcosm.build.transport.weight_quality",
                "microcosm.build.transport.evidence_contract",
            }
        }
        assert "microcosm.build.transport.registry" not in modules
        assert "microcosm.build.transport.evidence_kernels" not in modules
        # Today's 197 modules, plus the leaf and what it imports that the
        # bindings did not already reach.
        assert len(without | (leaf & without)) <= 197
        assert len(modules) == len(without) + len(leaf - without) + len(leaf & without)


# ---------------------------------------------------------------------------
# Module-list completeness, traced (S2)
# ---------------------------------------------------------------------------


#: Framework modules a kernel's run() executes that its list may leave out,
#: each with its reason. Only code the executor hands every kernel fits here,
#: and ``test_no_registered_kernel_lists_an_excluded_module`` checks that no
#: registered kernel's ``source_hash`` reads any of them.
TRACE_EXCLUSIONS: dict[str, dict[str, str]] = {
    ref: {
        "microcosm.graph.kernel": (
            "KernelContext and KernelResult, the executor's interface to every "
            "kernel; an edit there re-keys every node through the executor."
        ),
    }
    # Every evidence kernel the leaf lists, so a new kernel needs a trace.
    for ref in EVIDENCE_KERNEL_MODULES
}


def _microcosm_module_files() -> dict[str, str]:
    files = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("microcosm.") and path:
            files[str(Path(path).resolve())] = name
    return files


def _traced(run) -> set[str]:
    """Every ``microcosm.*`` module whose Python functions ``run()`` calls."""

    called: set[str] = set()

    def profile(frame, event, arg):
        if event == "call":
            called.add(frame.f_code.co_filename)

    sys.setprofile(profile)
    try:
        run()
    finally:
        sys.setprofile(None)
    files = _microcosm_module_files()
    return {
        files[path]
        for path in (str(Path(name).resolve()) for name in called if name[:1] != "<")
        if path in files
    }


def _unlisted(ref: str, traced: set[str], modules=None) -> set[str]:
    listed = set((modules or EVIDENCE_KERNEL_MODULES)[ref])
    return traced - listed - set(TRACE_EXCLUSIONS.get(ref, {}))


@pytest.fixture(scope="module")
def kernel_traces(toy, donor, tmp_path_factory):
    """Each evidence kernel's run() on the fixtures these tests already use."""

    frame = _frame(toy.spec["resources"]["axiom_input_closure"])
    payload = donor.read_bytes()
    receipt = _context(
        frame,
        _receipt_node(_sha(payload), len(payload)),
        sources={"donor_pool": donor},
    )
    traces = {DONOR_RECEIPT_REF: _traced(lambda: TRANSPORT_DONOR_RECEIPT.run(receipt))}
    toy_closure = _context(frame, _closure_node(toy))
    kernel = _closure_kernel(toy)
    closure = _traced(lambda: kernel.run(toy_closure))
    # The closure kernel's AxiomEngine branch, with a relation input.
    node, kernel = _relation_world(tmp_path_factory.mktemp("trace"), _TOY_RELATION)
    context = _relation_context(_relation_frame({"member_income": np.ones(3)}), node)
    closure |= _traced(lambda: kernel.run(context))
    traces[INPUT_CLOSURE_REF] = closure
    weights = _project_context(
        _weights_node(),
        Population.from_frame(_units_frame(), "toy.terminal"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
        artifacts=_artifact_values(_calibration()),
    )
    traces[WEIGHTS_AUDIT_REF] = _traced(lambda: TRANSPORT_WEIGHTS_AUDIT.run(weights))
    return traces


class TestModuleListsAreComplete:
    @pytest.mark.parametrize("ref", sorted(EVIDENCE_KERNEL_MODULES))
    def test_every_module_a_kernel_runs_is_listed(self, kernel_traces, ref) -> None:
        # Under-listing serves stale results: an edit to an unlisted module
        # leaves implementation_hash() unchanged.
        assert ref in kernel_traces
        assert _unlisted(ref, kernel_traces[ref]) == set()

    def test_the_closure_kernel_runs_the_schema_module(self, kernel_traces) -> None:
        # Mutant: the closure kernel's list loses microcosm.frame.schema,
        # whose entity_id_column the producer calls.
        assert "microcosm.frame.schema" in kernel_traces[INPUT_CLOSURE_REF]
        shortened = dict(EVIDENCE_KERNEL_MODULES)
        shortened[INPUT_CLOSURE_REF] = tuple(
            name
            for name in shortened[INPUT_CLOSURE_REF]
            if name != "microcosm.frame.schema"
        )
        assert _unlisted(
            INPUT_CLOSURE_REF, kernel_traces[INPUT_CLOSURE_REF], shortened
        ) == {"microcosm.frame.schema"}

    def test_no_registered_kernel_lists_an_excluded_module(self, toy) -> None:
        # Record every source file source_hash reads while each registered
        # kernel computes its implementation hash.
        from microcosm.graph import kernel as kernel_module

        read: set[str] = set()
        real = kernel_module.Path

        def recording(*args, **kwargs):
            path = real(*args, **kwargs)
            read.add(str(path.resolve()))
            return path

        registry = toy.registry.kernels
        refs = list(registry.refs()) if hasattr(registry, "refs") else list(registry)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(kernel_module, "Path", recording)
            for ref in refs:
                registry.get(ref).implementation_hash()
        files = _microcosm_module_files()
        hashed = {files[path] for path in read if path in files}
        assert evidence_contract.__name__ in hashed
        excluded = {name for rows in TRACE_EXCLUSIONS.values() for name in rows}
        assert not excluded & hashed, excluded & hashed


# ---------------------------------------------------------------------------
# transport.donor_receipt@1
# ---------------------------------------------------------------------------


class TestDonorReceipt:
    def test_the_receipt_records_the_pin_and_the_frame_metadata(
        self, evidence, donor
    ) -> None:
        _, receipt, _ = evidence
        document = decode_donor_receipt(receipt).document
        payload = donor.read_bytes()
        assert document["donor"] == {
            "sha256": _sha(payload),
            "size_bytes": len(payload),
            "locator": "donor.h5",
            "revision": "toy-revision",
        }
        assert document["pin"] == {
            "stage": "load_donor",
            "artifact_index": 0,
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
        assert document["content_basis"] == "transport"
        assert document["donor_time_period"] == 2024
        assert document["producer"]["node"] == "toy.evidence.donor"
        assert document["inputs"]["artifacts"] == {}

    def test_bytes_that_differ_from_the_declared_pin_are_refused(
        self, donor, evidence
    ) -> None:
        frame = evidence[0]
        payload = donor.read_bytes()
        node = _receipt_node("0" * 64, len(payload))
        with pytest.raises(ValueError, match="donor bytes"):
            _run_receipt(donor, frame, node=node)
        node = _receipt_node(_sha(payload), len(payload) + 1)
        with pytest.raises(ValueError, match="donor bytes"):
            _run_receipt(donor, frame, node=node)

    @pytest.mark.parametrize("copies", (0, 2))
    def test_zero_or_several_pins_are_refused(self, donor, evidence, copies) -> None:
        payload = donor.read_bytes()
        node = _receipt_node(_sha(payload), len(payload), copies=copies)
        with pytest.raises(ValueError, match="exactly one source_stages"):
            _run_receipt(donor, evidence[0], node=node)

    def test_absent_frame_metadata_is_refused(self, donor, evidence) -> None:
        frame = evidence[0]
        bare = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            SCHEMA,
            {"household": Weights(np.ones(2), WeightKind.DESIGN)},
        )
        with pytest.raises(ValueError, match="content_basis and donor_time_period"):
            _run_receipt(donor, bare)


# ---------------------------------------------------------------------------
# The closure partition
# ---------------------------------------------------------------------------


def _nz():
    closure = InputClosure.from_dict(
        json.loads((NZ / "axiom_input_closure.json").read_text())
    )
    bindings = json.loads((NZ / "axiom_rules_bindings.json").read_text())
    surface = load_axiom_input_surface(NZ_SURFACE)
    return closure, bindings, surface, axiom_concept_mapping("nz")


def _nz_partition(closure=None, bindings=None, surface=None, mapping=None, **kwargs):
    base = _nz()
    closure = closure or base[0]
    bindings = bindings or base[1]
    surface = surface or base[2]
    mapping = mapping or base[3]
    rows = bindings["bindings"]
    roots = {r["id"]: dict(surface.modules[r["rulespec_path"]].inputs) for r in rows}
    shas = {r["id"]: surface.modules[r["rulespec_path"]].sha256 for r in rows}
    return closure_partition(
        closure,
        mapping,
        rows,
        roots,
        module_sha256=kwargs.pop("module_sha256", shas),
        module_imports=kwargs.pop("module_imports", {}),
        mapped_entities=bindings["entity_names"].values(),
        **kwargs,
    )


class TestClosurePartition:
    def test_the_committed_nz_surface_partitions_as_the_design_table(self) -> None:
        result = _nz_partition()
        assert result["check"] == []
        rows = {row["binding"]: row for row in result["modules"]}
        counts = {binding: partition_counts(row) for binding, row in rows.items()}
        zero = dict.fromkeys(CLOSURE_CLASSES, 0)
        assert counts == {
            "main_benefits_entitlement": {**zero, "encoded": 13, "defaulted": 47},
            "main_benefits_rates": {**zero, "encoded": 13, "defaulted": 15},
            "nz_super": {**zero, "encoded": 2, "defaulted": 11},
            "accommodation_supplement": {
                **zero,
                "encoded": 17,
                "knob_targeted": 2,
                "defaulted": 20,
            },
            "family_tax_credits": {**zero, "encoded": 2, "unreached": 15},
            "income_tax_schedule_1": {**zero, "encoded": 1},
        }
        assert sum(len(row["roots"]) for row in rows.values()) == 158
        assert rows["family_tax_credits"]["unmapped_entity"] == [
            "Child.best_start_abatement_days",
            "Child.best_start_child_care_fraction",
            "Child.best_start_entitlement_days",
            "Child.best_start_family_scheme_income_for_relationship_period",
        ]
        assert rows["family_tax_credits"]["encoded"]["graph_node"] == [
            "Family.family_tax_credit_eldest_dependent_child_care_units",
            "Family.family_tax_credit_subsequent_dependent_child_care_units",
        ]

    def test_the_check_equals_input_closure_check_on_the_reduced_surface(self) -> None:
        # Differential: the producer's check is InputClosure.check on the
        # helper's surface and mapping.
        closure, bindings, surface, mapping = _nz()
        reduced_surface, reduced_mapping = without_unreached(
            surface,
            mapping,
            unreached_refs(bindings["bindings"]),
            bindings["entity_names"].values(),
        )
        assert closure.check(reduced_mapping, reduced_surface) == ()
        assert _nz_partition()["check"] == list(
            closure.check(reduced_mapping, reduced_surface)
        )

    def test_dropping_a_closure_entry_is_a_gap(self) -> None:
        closure, *_ = _nz()
        dropped = replace(closure, entries=closure.entries[1:])
        result = _nz_partition(closure=dropped)
        gaps = [name for row in result["modules"] for name in row["gap"]]
        assert gaps == [f"{closure.entries[0].entity}.{closure.entries[0].input}"]
        assert any("is not closed" in line for line in result["check"])

    def test_an_entry_the_surface_lacks_is_unexpected(self) -> None:
        closure, *_ = _nz()
        extra = replace(closure.entries[0], input="not_a_root_input")
        result = _nz_partition(
            closure=replace(closure, entries=(*closure.entries, extra))
        )
        unexpected = [n for row in result["modules"] for n in row["unexpected"]]
        assert unexpected == [f"{extra.entity}.not_a_root_input"]

    def test_a_flipped_module_sha_is_reported(self) -> None:
        _, bindings, surface, _ = _nz()
        shas = {
            r["id"]: surface.modules[r["rulespec_path"]].sha256
            for r in bindings["bindings"]
        }
        shas["nz_super"] = "0" * 64
        result = _nz_partition(module_sha256=shas)
        assert any("has SHA-256 " + "0" * 64 in line for line in result["check"])

    def test_a_concept_input_reclassed_defaulted_is_reported(self) -> None:
        closure, *_ = _nz()
        index, entry = next(
            (i, e)
            for i, e in enumerate(closure.entries)
            if e.encoded_by is not None and e.encoded_by.value == "concept_binding"
        )
        defaulted = replace(
            entry,
            closure_class="defaulted",
            encoded_by=None,
            value=0,
            citation="Synthetic (mutant)",
            reason="A mutant default.",
        )
        entries = list(closure.entries)
        entries[index] = defaulted
        result = _nz_partition(closure=replace(closure, entries=tuple(entries)))
        assert any(
            "bound by the mapping but closed as defaulted" in p for p in result["check"]
        )

    @pytest.mark.parametrize(
        ("edit", "match"),
        [
            (lambda d: d.update(sha256="0" * 64), "re-verify the unreached roots"),
            (
                lambda d: d.update(outputs=["in_work_tax_credit_before_abatement"]),
                "differ from the binding's variables",
            ),
            (lambda d: d["names"].append("not_a_live_root"), "not a live root"),
            (
                lambda d: d["names"].append(
                    "family_tax_credit_eldest_dependent_child_care_units"
                ),
                "the closure also closes",
            ),
        ],
    )
    def test_a_wrong_unreached_declaration_is_refused(self, edit, match) -> None:
        _, bindings, *_ = _nz()
        bindings = copy.deepcopy(bindings)
        row = next(r for r in bindings["bindings"] if r["id"] == "family_tax_credits")
        edit(row["unreached_roots"])
        with pytest.raises(ValueError, match=match):
            _nz_partition(bindings=bindings)

    def test_a_module_with_imports_refuses_its_unreached_declaration(self) -> None:
        with pytest.raises(ValueError, match="declares imports"):
            _nz_partition(module_imports={"family_tax_credits": True})
        assert evidence_contract.module_declares_imports(
            b"format: x\nimports:\n  - a\n"
        )
        assert not evidence_contract.module_declares_imports(b"rules:\n  imports: 1\n")

    def test_a_binding_local_view_is_refused_with_the_reserved_message(self) -> None:
        _, bindings, *_ = _nz()
        bindings = copy.deepcopy(bindings)
        bindings["bindings"][0]["views"] = [{"name": "child"}]
        with pytest.raises(ValueError, match="reserved for binding-local views"):
            _nz_partition(bindings=bindings)

    def test_be_multi_entity_modules_keep_every_root_on_its_entity(self) -> None:
        surface = load_axiom_input_surface(BE_SURFACE)
        paths = [
            "be/statutes/income_tax/individual/final_tax.yaml",
            "be/statutes/income_tax/individual/tax_liability_pipeline.yaml",
            "be/statutes/income_tax/individual/tax_free_amount_tax.yaml",
            "be/statutes/income_tax/individual/self_employed_oracle_pipeline.yaml",
            "be/statutes/income_tax/individual/pensioner_pit_oracle_pipeline.yaml",
            "be/regulations/social_integration/resource_calculation.yaml",
        ]
        entities = sorted({e for p in paths for e in surface.modules[p].inputs})
        rows = [
            {
                "id": f"be{index}",
                "rulespec_path": path,
                "sha256": surface.modules[path].sha256,
                "engine_entity": next(iter(surface.modules[path].inputs)),
                "variables": ["x"],
            }
            for index, path in enumerate(paths)
        ]
        closure = InputClosure(
            country="be",
            content_basis="transport",
            mapping_engine="none",
            mapping_engine_version="none",
            surface_rulespec_commit=surface.rulespec_commit,
            surface_engine_repository="engine",
            surface_engine_commit=surface.engine_commit,
            modules={path: surface.modules[path].sha256 for path in paths},
            engine_optional_evidence="Synthetic.",
            undetermined=_nz()[0].undetermined,
            knobs=(),
            entries=(),
        )
        mapping = _empty_mapping()
        roots = {
            row["id"]: dict(surface.modules[row["rulespec_path"]].inputs)
            for row in rows
        }
        result = closure_partition(
            closure,
            mapping,
            rows,
            roots,
            module_sha256={row["id"]: row["sha256"] for row in rows},
            module_imports={},
            mapped_entities=entities,
        )
        for row, path in zip(result["modules"], paths, strict=True):
            module = surface.modules[path]
            assert len(module.inputs) >= 2, path
            assert set(row["gap"]) == {
                f"{entity}.{name}"
                for entity, names in module.inputs.items()
                for name in names
            }
        # Mutant: the producer places every root flat on one entity. The
        # partition then names refs the surface does not hold.
        flat = {
            row["id"]: {
                row["engine_entity"]: sorted(
                    {
                        n
                        for names in surface.modules[
                            row["rulespec_path"]
                        ].inputs.values()
                        for n in names
                    }
                )
            }
            for row in rows
        }
        flattened = closure_partition(
            closure,
            mapping,
            rows,
            flat,
            module_sha256={row["id"]: row["sha256"] for row in rows},
            module_imports={},
            mapped_entities=entities,
        )
        assert flattened["modules"] != result["modules"]
        moved = [
            name
            for row, path in zip(flattened["modules"], paths, strict=True)
            for name in row["gap"]
            if InputRef(name.partition(".")[2], name.partition(".")[0], path)
            not in set(surface.modules[path].refs())
        ]
        assert moved

    @PROPERTY
    @given(data=st.data())
    def test_property_every_live_root_lands_in_exactly_one_class(self, data) -> None:
        # Draws subsets of the committed NZ roots, roots on a second mapped
        # entity, and relation inputs whose columns the frame carries or
        # lacks, and reads the roots through engine_root_inputs on a fake
        # AxiomEngine, as the producer does.
        from types import SimpleNamespace

        closure, bindings, surface, mapping = _nz()
        entity_names = bindings["entity_names"]
        to_frame = {engine: frame for frame, engine in entity_names.items()}
        rows = copy.deepcopy(bindings["bindings"])
        engines = {}
        person_columns: set[str] = set()
        expected_relation: dict[str, set[str]] = {}
        expected_gap: dict[str, set[str]] = {}
        for row in rows:
            module = surface.modules[row["rulespec_path"]]
            programs: dict[str, set[str]] = {}
            for entity, names in module.inputs.items():
                if entity not in to_frame:
                    continue
                keep = data.draw(
                    st.lists(st.sampled_from(names), unique=True), label=row["id"]
                )
                extra = data.draw(
                    st.lists(
                        st.from_regex(r"x_[a-z]{1,6}", fullmatch=True),
                        unique=True,
                        max_size=3,
                    )
                )
                programs.setdefault(to_frame[entity], set()).update(keep, extra)
            # A module whose roots sit on two mapped entities.
            second = data.draw(
                st.sampled_from(sorted(set(to_frame.values()))), label="second"
            )
            programs.setdefault(second, set()).update(
                data.draw(
                    st.lists(
                        st.from_regex(r"y_[a-z]{1,6}", fullmatch=True),
                        unique=True,
                        max_size=2,
                    )
                )
            )
            relations = []
            for index in range(data.draw(st.integers(0, 2), label="relations")):
                name = f"r_{row['id']}_{index}"
                root = data.draw(st.sampled_from(("household", "family")))
                present = data.draw(st.booleans(), label="column present")
                column = f"c_{row['id']}_{index}"
                if present:
                    person_columns.add(column)
                    expected_relation.setdefault(row["id"], set()).add(
                        f"{entity_names[root]}.{name}"
                    )
                else:
                    expected_gap.setdefault(row["id"], set()).add(
                        f"{entity_names[root]}.{name}"
                    )
                relations.append(
                    {
                        **_TOY_RELATION,
                        "predicate": f"p{index}",
                        "key": f"p{index}:0:1",
                        "root_entity": root,
                        "membership_column": f"person_{root}_id",
                        "related_inputs": {name: column},
                    }
                )
            if relations:
                row["relations"] = relations
            engine = _fake_axiom(Path(row["rulespec_path"]))
            engine._programs = {
                frame: (
                    SimpleNamespace(root_inputs=tuple(sorted(programs[frame])))
                    if frame in programs
                    else None
                )
                for frame in SCHEMA.entities
            }
            engines[row["id"]] = engine
        roots = {
            row["id"]: evidence_kernels.engine_root_inputs(
                engines[row["id"]], row, entity_names
            )
            for row in rows
        }
        # Unreached declarations only hold over their live roots.
        for row in rows:
            declaration = row.get("unreached_roots")
            if declaration:
                live = set(roots[row["id"]].get(row["engine_entity"], ()))
                declaration["names"] = [
                    n for n in declaration["names"] if n in live
                ] or None
                if declaration["names"] is None:
                    del row["unreached_roots"]
        result = closure_partition(
            closure,
            mapping,
            rows,
            roots,
            module_sha256={
                r["id"]: surface.modules[r["rulespec_path"]].sha256 for r in rows
            },
            module_imports={},
            mapped_entities=entity_names.values(),
            frame_columns={"person": frozenset(person_columns)},
            entity_names=entity_names,
        )
        for row in result["modules"]:
            placed = [
                *(n for names in row["encoded"].values() for n in names),
                *row["knob_targeted"],
                *(item["input"] for item in row["defaulted"]),
                *row["engine_optional"],
                *(item["input"] for item in row["relation"]),
                *row["unreached"],
                *row["gap"],
            ]
            assert sorted(placed) == row["roots"]
            assert len(set(placed)) == len(placed)
            assert sum(partition_counts(row).values()) == len(row["roots"])
            relation = {item["input"] for item in row["relation"]}
            assert relation == expected_relation.get(row["binding"], set())
            assert expected_gap.get(row["binding"], set()) <= set(row["gap"])
        # No relation input, present or absent, reaches the closure check.
        assert not [line for line in result["check"] if ".r_" in line]


def _empty_mapping() -> ConceptMapping:
    from microcosm.frame.concepts import CONCEPT_ENTITIES, CONCEPTS

    return ConceptMapping(
        engine="none",
        engine_version="none",
        entity_correspondence={entity: entity for entity in CONCEPT_ENTITIES},
        input_declaration="usage_inferred",
        bindings=(),
        unmapped={item.id: "Synthetic: unmapped." for item in CONCEPTS},
    )


# ---------------------------------------------------------------------------
# Relations (H44 C)
# ---------------------------------------------------------------------------


_TOY_RELATION = {
    "predicate": "toy_member",
    "key": "toy_member:0:1",
    "current_slot": 0,
    "related_slot": 1,
    "arguments": ["Family", "Person"],
    "root_entity": "family",
    "related_entity": "person",
    "membership_column": "person_family_id",
    "related_inputs": {"toy_member_income": "member_income"},
    "cardinality": "many",
    "period": "2026-27",
}


class TestRelations:
    def test_the_reader_agrees_with_the_registry_parser(self) -> None:
        row = {
            "id": "toy_relation",
            "entity": "family",
            "period": "2026-27",
            "relations": [
                _TOY_RELATION,
                {
                    **_TOY_RELATION,
                    "predicate": "toy_partner",
                    "key": "toy_partner:0:1",
                    "related_inputs": {"toy_member_age": "age"},
                    "role_column": "role",
                    "roles": ["head"],
                },
            ],
        }
        parsed = registry_module._binding_relations(
            row,
            {"person": "Person", "household": "Household", "family": "Family"},
            SCHEMA,
        )
        expected = sorted(
            (name, column, relation.role_column)
            for relation in parsed
            for name, column in relation.related_inputs.items()
        )
        assert [
            (item.input, item.column, item.role_column)
            for item in binding_relation_inputs(row)
        ] == expected

    def _partition(self, frame_columns, *, relations=None, whole=False):
        rows = [
            {
                "id": "toy_relation",
                "rulespec_path": "toy/relation.yaml",
                "sha256": "a" * 64,
                "engine_entity": "Family",
                "variables": ["x"],
                "relations": relations or [_TOY_RELATION],
            }
        ]
        closure = InputClosure(
            country="xx",
            content_basis="transport",
            mapping_engine="none",
            mapping_engine_version="none",
            surface_rulespec_commit="r",
            surface_engine_repository="engine",
            surface_engine_commit="e",
            modules={"toy/relation.yaml": "a" * 64},
            engine_optional_evidence="Synthetic.",
            undetermined=_nz()[0].undetermined,
            knobs=(),
            entries=(),
        )
        result = closure_partition(
            closure,
            _empty_mapping(),
            rows,
            {"toy_relation": {"Family": ("toy_member_income",)}},
            module_sha256={"toy_relation": "a" * 64},
            module_imports={},
            mapped_entities=("Person", "Household", "Family"),
            frame_columns=frame_columns,
            entity_names={
                "person": "Person",
                "household": "Household",
                "family": "Family",
            },
        )
        return result if whole else result["modules"][0]

    def test_a_relation_input_with_its_physical_column_is_a_relation(self) -> None:
        row = self._partition({"person": frozenset({"member_income"})})
        assert row["relation"] == [
            {
                "input": "Family.toy_member_income",
                "column": "member_income",
                "role_column": None,
            }
        ]
        assert row["gap"] == []

    def test_a_relation_input_whose_column_the_frame_lacks_is_a_gap(self) -> None:
        row = self._partition({"person": frozenset({"other"})})
        assert row["relation"] == []
        assert row["gap"] == ["Family.toy_member_income"]

    def test_a_declared_role_column_the_frame_lacks_is_a_gap(self) -> None:
        # S4: the relation branch also requires a declared role column.
        relation = {**_TOY_RELATION, "role_column": "role", "roles": ["head"]}
        row = self._partition(
            {"person": frozenset({"member_income"})}, relations=[relation]
        )
        assert row["relation"] == []
        assert row["gap"] == ["Family.toy_member_income"]
        assert row["relation_missing"] == ["Family.toy_member_income: person.role"]
        both = self._partition(
            {"person": frozenset({"member_income", "role"})}, relations=[relation]
        )
        assert both["gap"] == []
        assert [item["role_column"] for item in both["relation"]] == ["role"]

    def test_a_valid_relation_input_is_no_check_problem(self) -> None:
        # S3: a relation input has no closure entry, so the checker's surface
        # leaves it out; the partition still classes it.
        result = self._partition({"person": frozenset({"member_income"})}, whole=True)
        assert result["check"] == []
        assert [item["input"] for item in result["modules"][0]["relation"]] == [
            "Family.toy_member_income"
        ]
        absent = self._partition({"person": frozenset()}, whole=True)
        assert absent["check"] == []
        assert absent["modules"][0]["gap"] == ["Family.toy_member_income"]


# ---------------------------------------------------------------------------
# Relation inputs on a real AxiomEngine (S1, S3, S4)
# ---------------------------------------------------------------------------


_ENTITY_NAMES = {"person": "Person", "household": "Household", "family": "Family"}


def _fake_axiom(module: Path, root_inputs=()):
    """An ``AxiomEngine`` whose family program reads ``root_inputs`` only.

    The engine keeps relation inputs out of ``root_inputs``; ``variables()``
    adds them from the declared manifests.
    """

    from types import SimpleNamespace

    from microcosm.frame.adapters.axiom import AxiomEngine

    engine = object.__new__(AxiomEngine)
    engine._schema = SCHEMA
    engine._entity_names = dict(_ENTITY_NAMES)
    engine._module = module
    engine._programs = {
        "person": None,
        "household": None,
        "family": SimpleNamespace(root_inputs=tuple(root_inputs)),
    }
    assert isinstance(engine, AxiomEngine)
    return engine


def _relation_world(tmp_path: Path, relation: dict, *, root_inputs=()):
    module = tmp_path / "toy" / "relation.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_bytes(b"format: rulespec\nrules: []\n")
    sha = _sha(module.read_bytes())
    closure = InputClosure(
        country="xx",
        content_basis="transport",
        mapping_engine="none",
        mapping_engine_version="none",
        surface_rulespec_commit="r",
        surface_engine_repository="engine",
        surface_engine_commit="e",
        modules={"toy/relation.yaml": sha},
        engine_optional_evidence="Synthetic.",
        undetermined=_nz()[0].undetermined,
        knobs=(),
        entries=(),
    ).to_dict()
    bindings = {
        "entity_names": dict(_ENTITY_NAMES),
        "engine": {"commit": "e"},
        "rulespec": {"commit": "r"},
        "bindings": [
            {
                "id": "toy_relation",
                "rulespec_path": "toy/relation.yaml",
                "sha256": sha,
                "engine_entity": "Family",
                "variables": ["x"],
                "relations": [relation],
            }
        ],
    }
    mapping = _empty_mapping().to_dict()
    refs = {"engine_refs": {"toy_relation": "axiom:toy_relation"}}
    node = Node(
        "toy.evidence.input_closure",
        INPUT_CLOSURE_REF,
        inputs=(Slice("person", ("member_income", "role")),),
        params={
            "closure": _text(closure),
            "closure_sha256": _sha(canonical_json(closure)),
            "mapping": _text(mapping),
            "mapping_sha256": _sha(canonical_json(mapping)),
            "bindings": _text(bindings),
            "bindings_sha256": _sha(canonical_json(bindings)),
            "engine_refs": _text(refs),
        },
        artifact_outputs=(ArtifactOutput(AXIOM_INPUT_CLOSURE, INPUT_CLOSURE_TYPE),),
    )
    engine = _fake_axiom(module, root_inputs)
    return node, TransportInputClosureKernel({"axiom:toy_relation": engine})


def _relation_frame(columns: dict) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.array([1, 2, 3], dtype=np.int64),
            "person_household_id": np.array([10, 10, 20], dtype=np.int64),
            "person_family_id": np.array([100, 100, 200], dtype=np.int64),
            **columns,
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": np.array([10, 20])}),
            "family": pd.DataFrame({"family_id": np.array([100, 200])}),
        },
        SCHEMA,
        {"household": Weights(np.ones(2), WeightKind.DESIGN)},
        metadata={"content_basis": "transport", "donor_time_period": 2024},
    )


def _relation_context(frame: Frame, node: Node) -> KernelContext:
    tables = {}
    for item in node.inputs:
        table = frame.table(item.entity)
        wanted = [
            SCHEMA.entity_id_column(item.entity),
            *(SCHEMA.membership_column(g) for g in SCHEMA.group_entities),
            *item.columns,
        ]
        tables[item.entity] = table.loc[:, [c for c in wanted if c in table.columns]]
    return KernelContext(
        node=node,
        tables=tables,
        weights={},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
        sources={},
        artifacts={},
        frame_metadata=frame.metadata,
    )


class TestRelationInputsOnAnAxiomEngine:
    def _run(self, tmp_path, relation, columns, **kwargs):
        node, kernel = _relation_world(tmp_path, relation, **kwargs)
        payload = kernel.run(_relation_context(_relation_frame(columns), node))
        return decode_input_closure(payload.artifacts[AXIOM_INPUT_CLOSURE]).document

    def test_the_engine_partition_adds_each_declared_relation_input(
        self, tmp_path
    ) -> None:
        # S1: the program reads no root input; the relation input reaches the
        # partition on the manifest's root entity, as variables() lists it.
        engine = _fake_axiom(tmp_path / "m.yaml")
        row = {"id": "b", "relations": [_TOY_RELATION]}
        assert evidence_kernels.engine_root_inputs(engine, row, _ENTITY_NAMES) == {
            "Family": ("toy_member_income",)
        }

    def test_a_present_column_is_one_relation_row(self, tmp_path) -> None:
        document = self._run(tmp_path, _TOY_RELATION, {"member_income": np.ones(3)})
        (row,) = document["modules"]
        assert row["relation"] == [
            {
                "input": "Family.toy_member_income",
                "column": "member_income",
                "role_column": None,
            }
        ]
        assert row["gap"] == []
        assert document["check"] == []
        from microcosm.build.transport.gate_bindings import _closure_rows

        failures, _ = _closure_rows(document, allow_engine_optional=False)
        assert failures == []

    def test_an_absent_column_is_a_gap_and_the_gate_fails(self, tmp_path) -> None:
        document = self._run(tmp_path, _TOY_RELATION, {})
        (row,) = document["modules"]
        assert row["relation"] == []
        assert row["gap"] == ["Family.toy_member_income"]
        from microcosm.build.transport.gate_bindings import _closure_rows

        failures, _ = _closure_rows(document, allow_engine_optional=False)
        assert any("gap Family.toy_member_income" in line for line in failures)
        assert any("person.member_income" in line for line in failures)

    def test_an_absent_role_column_is_a_gap_naming_it(self, tmp_path) -> None:
        # S4 through the producer.
        relation = {**_TOY_RELATION, "role_column": "role", "roles": ["head"]}
        document = self._run(tmp_path, relation, {"member_income": np.ones(3)})
        (row,) = document["modules"]
        assert row["gap"] == ["Family.toy_member_income"]
        assert row["relation_missing"] == ["Family.toy_member_income: person.role"]
        present = self._run(
            tmp_path,
            relation,
            {"member_income": np.ones(3), "role": np.ones(3)},
        )
        assert present["modules"][0]["gap"] == []


# ---------------------------------------------------------------------------
# transport.input_closure@1 and the release_input_coverage gate
# ---------------------------------------------------------------------------


class TestInputClosureProducer:
    def test_the_toy_closure_holds_with_every_column_sliced(self, evidence) -> None:
        frame, _, closure = evidence
        document = decode_input_closure(closure).document
        assert (
            document["check"] == ["Surface engine synthetic is not synthetic."]
            or document["check"] == []
        )
        for row in document["modules"]:
            assert row["gap"] == row["unexpected"] == row["frame_missing"] == []
        assert set(document["inputs"]["frame"]) >= {
            "person.input_age",
            "family.input_rent",
        }
        assert document["engine"]["declared_commit"] == "a" * 40
        assert document["engine"]["closure_surface_commit"] == "synthetic"

    def test_an_engine_reporting_an_extra_root_is_a_gap(self, toy, monkeypatch) -> None:
        engine = next(iter(toy.registry.engines.values()))
        roots = tuple(engine.variables())
        monkeypatch.setattr(
            type(engine),
            "variables",
            lambda self: (
                (*roots, "toy_unclosed")
                if self is engine
                else tuple(
                    sorted({row["input"] for row in self.program()["outputs"].values()})
                )
            ),
        )
        document = decode_input_closure(_run_closure(toy)).document
        gaps = [name for row in document["modules"] for name in row["gap"]]
        assert gaps and all(name.endswith(".toy_unclosed") for name in gaps)

    def test_an_unsliced_encoded_column_is_frame_missing(self, toy) -> None:
        node = _closure_node(toy)
        first = node.inputs[0]
        node = replace(
            node, inputs=(Slice(first.entity, first.columns[1:]), *node.inputs[1:])
        )
        frame = _frame(toy.spec["resources"]["axiom_input_closure"])
        payload = (
            _closure_kernel(toy)
            .run(_context(frame, node))
            .artifacts[AXIOM_INPUT_CLOSURE]
        )
        missing = [
            n
            for row in decode_input_closure(payload).document["modules"]
            for n in row["frame_missing"]
        ]
        assert len(missing) == 1


def _with_commit(toy, commit):
    spec = copy.deepcopy(toy.spec)
    spec["resources"]["axiom_input_closure"]["input_surface"]["engine"]["commit"] = (
        commit
    )
    return spec


class TestReleaseInputCoverage:
    @pytest.fixture(scope="class")
    def passing(self, toy, donor):
        spec = _with_commit(toy, "a" * 40)
        frame = _frame(spec["resources"]["axiom_input_closure"])
        receipt = _run_receipt(donor, frame)
        closure = _run_closure(toy, spec=spec, frame=frame)
        return frame, receipt, closure

    def test_checked_evidence_passes_and_reports_every_module(self, passing) -> None:
        frame, receipt, closure = passing
        outcome = _evaluate(_decoded(receipt, closure), frame)
        assert outcome.status is GateStatus.PASSED, outcome.result.failures
        details = outcome.result.details
        assert [row["binding"] for row in details["modules"]] == [
            "main_benefit",
            "pension",
            "family_housing",
        ]
        assert details["donor"]["pin_stage"] == "load_donor"

    def test_a_differing_engine_commit_fails(self, evidence) -> None:
        frame, receipt, closure = evidence
        outcome = _evaluate(_decoded(receipt, closure), frame)
        assert outcome.status is GateStatus.FAILED
        assert any(
            "is not the closure's surface engine commit" in f
            for f in outcome.result.failures
        )

    def test_a_null_declared_engine_commit_fails_with_its_own_reason(
        self, toy, donor
    ) -> None:
        spec = _with_commit(toy, "a" * 40)
        spec["resources"]["axiom_rules_bindings"]["engine"]["commit"] = None
        frame = _frame(spec["resources"]["axiom_input_closure"])
        closure = _run_closure(toy, spec=spec, frame=frame)
        receipt = _run_receipt(donor, frame)
        outcome = _evaluate(_decoded(receipt, closure), frame)
        assert outcome.status is GateStatus.FAILED
        assert any("declares no engine commit" in f for f in outcome.result.failures)

    @pytest.mark.parametrize(
        ("alias", "edit", "match"),
        [
            # A stale implementation hash, forged into the header.
            (
                AXIOM_INPUT_CLOSURE,
                lambda d: d["producer"].update(implementation_hash="0" * 64),
                "implementation hash",
            ),
            # The artifact names another producer node.
            (
                DONOR_ARTIFACT_RECEIPT,
                lambda d: d["producer"].update(node="other.node"),
                "the battery's edge names",
            ),
            # Edited input digests: a frame column and an upstream artifact.
            (
                AXIOM_INPUT_CLOSURE,
                lambda d: d["inputs"]["frame"].update({"person.input_age": "0" * 64}),
                "differs from the battery's frame",
            ),
            (
                DONOR_ARTIFACT_RECEIPT,
                lambda d: d["inputs"]["artifacts"].update(problem="0" * 64),
                "not declared on the battery",
            ),
            # A pending-stub-shaped body edit: a gap removed from the record.
            (
                AXIOM_INPUT_CLOSURE,
                lambda d: d["modules"][0].update(gap=["Person.x"]),
                "gap Person.x",
            ),
        ],
    )
    def test_forged_or_edited_evidence_fails(self, passing, alias, edit, match) -> None:
        frame, receipt, closure = passing
        if alias == AXIOM_INPUT_CLOSURE:
            closure = _forged(closure, edit)
        else:
            receipt = _forged(receipt, edit)
        outcome = _evaluate(_decoded(receipt, closure), frame)
        assert outcome.status is GateStatus.FAILED
        assert any(match in line for line in outcome.result.failures), (
            outcome.result.failures
        )

    def test_substituted_bytes_fail_against_the_batterys_digest(self, passing) -> None:
        frame, receipt, closure = passing
        edges = _edges(receipt, closure)
        forged = _forged(closure, lambda d: d.update(mapping_sha256="0" * 64))
        outcome = _evaluate(_decoded(receipt, forged, edges), frame)
        assert outcome.status is GateStatus.FAILED
        assert any("the battery received" in line for line in outcome.result.failures)

    def test_an_artifact_from_another_graph_fails(self, toy, donor, passing) -> None:
        # The same producer run on another frame: identical header shape, but
        # the battery's frame digests differ.
        frame, receipt, _ = passing
        spec = _with_commit(toy, "a" * 40)
        other = _run_closure(
            toy,
            spec=spec,
            frame=_frame(spec["resources"]["axiom_input_closure"], shift=5.0),
        )
        outcome = _evaluate(_decoded(receipt, other), frame)
        assert outcome.status is GateStatus.FAILED
        assert any(
            "differs from the battery's frame" in line
            for line in outcome.result.failures
        )

    def test_an_engine_optional_input_fails_unless_allowed(self, passing) -> None:
        frame, receipt, closure = passing
        optional = _forged(
            closure,
            lambda d: d["modules"][0].update(
                engine_optional=["Person.input_age"],
                encoded={
                    **d["modules"][0]["encoded"],
                    "concept_binding": [
                        n
                        for n in d["modules"][0]["encoded"]["concept_binding"]
                        if n != "Person.input_age"
                    ],
                },
            ),
        )
        edges = _edges(receipt, optional)
        refused = _evaluate(_decoded(receipt, optional, edges), frame)
        assert any("engine-optional" in line for line in refused.result.failures)
        allowed = _evaluate(
            _decoded(receipt, optional, edges),
            frame,
            parameters={"allow_engine_optional": True},
        )
        assert not any("engine-optional" in line for line in allowed.result.failures)
        assert allowed.result.details["modules"][0]["engine_optional"] == [
            "Person.input_age"
        ]

    def test_absent_evidence_is_evidence_absent(self, passing) -> None:
        frame, receipt, closure = passing
        artifacts = _decoded(receipt, closure)
        del artifacts[AXIOM_INPUT_CLOSURE]
        outcome = _evaluate(artifacts, frame)
        assert outcome.status is GateStatus.EVIDENCE_ABSENT
        assert outcome.reason == f"missing evidence: {AXIOM_INPUT_CLOSURE}"

    def test_two_failures_are_both_reported(self, passing) -> None:
        frame, receipt, closure = passing
        closure = _forged(
            closure,
            lambda d: [
                row.update(gap=[f"{row['engine_entity']}.missing"])
                for row in d["modules"]
            ],
        )
        outcome = _evaluate(_decoded(receipt, closure, _edges(receipt, closure)), frame)
        gaps = [line for line in outcome.result.failures if ": gap " in line]
        assert len(gaps) == 3


# ---------------------------------------------------------------------------
# The battery's graph_evidence (H40)
# ---------------------------------------------------------------------------


def _battery_context(artifact_inputs, artifacts):
    node = Node(
        BATTERY,
        "gates.battery@1",
        params={},
        artifact_inputs=artifact_inputs,
    )
    return KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object),
        params={},
        rng=np.random.default_rng(0),
        artifacts=artifacts,
    )


class TestGraphEvidence:
    def test_graph_evidence_mirrors_the_declared_edges(self, evidence) -> None:
        _, receipt, closure = evidence
        edges = (
            ArtifactInput(
                DONOR_ARTIFACT_RECEIPT,
                "toy.evidence.donor",
                DONOR_ARTIFACT_RECEIPT,
                DONOR_RECEIPT_TYPE,
            ),
            ArtifactInput(
                AXIOM_INPUT_CLOSURE,
                "toy.evidence.input_closure",
                AXIOM_INPUT_CLOSURE,
                INPUT_CLOSURE_TYPE,
            ),
        )
        values = {
            DONOR_ARTIFACT_RECEIPT: ArtifactValue(
                receipt, DONOR_RECEIPT_TYPE, "a" * 64, "c" * 64, _numerics()
            ),
            AXIOM_INPUT_CLOSURE: ArtifactValue(
                closure, INPUT_CLOSURE_TYPE, "b" * 64, "d" * 64, _numerics()
            ),
        }
        assert _graph_evidence(_battery_context(edges, values)) == _edges(
            receipt, closure
        )

    def test_an_artifact_input_named_graph_evidence_is_refused(self, evidence) -> None:
        _, receipt, _ = evidence
        edges = (
            ArtifactInput(
                GRAPH_EVIDENCE,
                "toy.evidence.donor",
                DONOR_ARTIFACT_RECEIPT,
                DONOR_RECEIPT_TYPE,
            ),
        )
        values = {
            GRAPH_EVIDENCE: ArtifactValue(
                receipt, DONOR_RECEIPT_TYPE, "a" * 64, "c" * 64, _numerics()
            )
        }
        with pytest.raises(ValueError, match="reserves the alias"):
            _graph_evidence(_battery_context(edges, values))

    @pytest.mark.parametrize(
        "alias", (GRAPH_EVIDENCE, DONOR_ARTIFACT_RECEIPT, AXIOM_INPUT_CLOSURE)
    )
    def test_an_artifact_alias_is_never_a_gates_json_parameter(self, alias) -> None:
        binding = TRANSPORT_GATE_REGISTRY[GATE]
        assert alias not in binding.parameter_keys
        from microcosm.build.gate_battery import validate_gate_parameters

        with pytest.raises(ValueError, match="does not route"):
            validate_gate_parameters(
                _entry({"allow_engine_optional": False, alias: {}}),
                TRANSPORT_GATE_REGISTRY,
            )

    def test_the_battery_lists_the_leaf_in_its_hash(self) -> None:
        source = Path(sys.modules[GateBatteryKernel.__module__].__file__).read_text()
        body = source[
            source.index("def implementation_hash") : source.index("def _upstream")
        ]
        assert "evidence_contract," in body


def _numerics():
    from microcosm.graph.kernel import NumericScope

    return NumericScope()


# ---------------------------------------------------------------------------
# Composition checks
# ---------------------------------------------------------------------------


class _Graph:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)


def _create(kernel):
    return Node(
        "toy.create",
        kernel,
        sources=("donor_pool", "facts"),
        params={
            "donor_source": "donor_pool",
            "donor_sha256": "a" * 64,
            "donor_size": 10,
            "donor_pin_sha256": "b" * 64,
        },
    )


def _receipt_declaration(**updates):
    params = {
        "donor_source": "donor_pool",
        "donor_sha256": "a" * 64,
        "donor_size": 10,
        "source_stages_sha256": "b" * 64,
        **updates,
    }
    return Node(
        "toy.evidence.donor", DONOR_RECEIPT_REF, sources=("donor_pool",), params=params
    )


class TestCompositionChecks:
    @pytest.mark.parametrize("kernel", ("transport.create@1", "transport.create@2"))
    def test_a_receipt_selecting_another_donor_is_refused(self, kernel) -> None:
        compose._check_donor_receipts(_Graph([_create(kernel), _receipt_declaration()]))
        for updates in (
            {"donor_sha256": "c" * 64},
            {"donor_size": 11},
            {"source_stages_sha256": "c" * 64},
        ):
            with pytest.raises(ValueError, match="selects a different donor"):
                compose._check_donor_receipts(
                    _Graph([_create(kernel), _receipt_declaration(**updates)])
                )

    def test_a_bridge_component_requesting_an_unverified_output_is_refused(
        self,
    ) -> None:
        spec = compose.load_transport_spec("nz")
        compose._check_unreached_requests(spec)
        # A bridge component, nested in a scenario's components, asks the
        # family tax credits binding for an output its review never verified.
        mutant = copy.deepcopy(spec)
        scenario = next(
            item
            for item in mutant["resources"]["scenarios"]["scenarios"]
            for node in item.get("nodes", ())
            if "components" in node.get("params", {})
        )
        component = next(
            component
            for node in scenario["nodes"]
            for component in node.get("params", {}).get("components", ())
            if component["engine_ref"]["binding"] == "family_tax_credits"
        )
        component["variables"] = ["in_work_tax_credit_before_abatement"]
        with pytest.raises(ValueError, match="in_work_tax_credit_before_abatement"):
            compose._check_unreached_requests(mutant)

    def test_a_literal_engine_reference_is_refused_too(self) -> None:
        # C2: the component writes the prepared engine reference itself
        # instead of {"binding": ...}; the check matches that spelling too.
        refs = {"family_tax_credits": "synthetic:family_tax_credits"}
        spec = compose.load_transport_spec("nz")
        compose._check_unreached_requests(spec, refs)
        mutant = copy.deepcopy(spec)
        component = next(
            component
            for item in mutant["resources"]["scenarios"]["scenarios"]
            for node in item.get("nodes", ())
            for component in node.get("params", {}).get("components", ())
            if component["engine_ref"]["binding"] == "family_tax_credits"
        )
        component["engine_ref"] = refs["family_tax_credits"]
        compose._check_unreached_requests(mutant, refs)
        component["variables"] = [
            *component["variables"],
            "in_work_tax_credit_before_abatement",
        ]
        with pytest.raises(ValueError, match="in_work_tax_credit_before_abatement"):
            compose._check_unreached_requests(mutant, refs)
        # A JSON-escaped spelling of the literal is the same request.
        component["engine_ref"] = json.dumps(refs["family_tax_credits"])
        with pytest.raises(ValueError, match="in_work_tax_credit_before_abatement"):
            compose._check_unreached_requests(mutant, refs)

    def test_composition_refuses_the_literal_spelling(self) -> None:
        # C2 through compose_transport_graph, which runs the check before it
        # composes anything.
        import importlib.util

        path = _BUILD.tests / "engine_free/shared/test_nz_transport_declarations.py"
        module_spec = importlib.util.spec_from_file_location("_nz_decl_c2", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        spec, config = module._nz_composition_spec_and_config()
        mutant = copy.deepcopy(spec)
        literal = config.engine_refs["family_tax_credits"]
        for item in mutant["resources"]["scenarios"]["scenarios"]:
            for node in item.get("nodes", ()):
                for component in node.get("params", {}).get("components", ()):
                    if component["engine_ref"] == {
                        **component["engine_ref"],
                        "binding": "family_tax_credits",
                    }:
                        component["engine_ref"] = literal
                        component["variables"] = [
                            *component["variables"],
                            "in_work_tax_credit_before_abatement",
                        ]
        with pytest.raises(ValueError, match="in_work_tax_credit_before_abatement"):
            compose.compose_transport_graph(mutant, config)

    def _closure_graph(self, slices, relation):
        bindings = {
            "entity_names": dict(_ENTITY_NAMES),
            "bindings": [{"id": "toy_relation", "relations": [relation]}],
        }
        return _Graph(
            [
                Node(
                    "toy.evidence.input_closure",
                    INPUT_CLOSURE_REF,
                    inputs=tuple(Slice(e, tuple(c)) for e, c in slices),
                    params={"bindings": _text(bindings)},
                )
            ]
        )

    def test_a_closure_slice_without_its_relation_columns_is_refused(self, toy) -> None:
        # S5: a Slice delivers only the columns it names, so composition
        # refuses a closure node whose person slice omits a relation column.
        relation = {**_TOY_RELATION, "role_column": "role", "roles": ["head"]}
        compose._check_closure_relation_slices(
            toy.spec,
            self._closure_graph(
                [("person", ("age", "member_income", "role"))], relation
            ),
        )
        for columns, missing in (
            (("age",), "['member_income', 'role']"),
            (("age", "member_income"), "['role']"),
            (("age", "role"), "['member_income']"),
        ):
            with pytest.raises(ValueError, match=re.escape(missing)):
                compose._check_closure_relation_slices(
                    toy.spec, self._closure_graph([("person", columns)], relation)
                )
        # The columns must sit on the person slice, not another entity's.
        with pytest.raises(ValueError, match="toy.evidence.input_closure"):
            compose._check_closure_relation_slices(
                toy.spec,
                self._closure_graph([("family", ("member_income", "role"))], relation),
            )


@pytest.fixture(scope="module")
def nz_composed():
    import importlib.util

    path = _BUILD.tests / "engine_free/shared/test_nz_transport_declarations.py"
    module_spec = importlib.util.spec_from_file_location("_nz_declarations", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec, config = module._nz_composition_spec_and_config()
    return spec, config, compose.compose_transport_graph(spec, config)


class TestGraphNodeEntries:
    def _mutant(self, spec, edit):
        mutant = copy.deepcopy(spec)
        closure = mutant["resources"]["axiom_input_closure"]
        edit(closure)
        return mutant

    def test_the_packaged_entries_name_supplying_nodes(self, nz_composed) -> None:
        spec, config, graph = nz_composed
        compose._check_graph_node_entries(spec, config, graph, extended=True)

    @pytest.mark.parametrize(
        ("node", "match"),
        [
            ("nz.gross_up", "does not hold"),
            ("nz.as.bridge.cutout", "neither owns, overrides nor solves"),
        ],
    )
    def test_a_wrong_node_is_refused(self, nz_composed, node, match) -> None:
        spec, config, graph = nz_composed

        def edit(closure):
            for entry in closure["entries"]:
                if (
                    entry["input"]
                    == "family_tax_credit_eldest_dependent_child_care_units"
                ):
                    entry["node"] = node

        with pytest.raises(ValueError, match=match):
            compose._check_graph_node_entries(
                self._mutant(spec, edit), config, graph, extended=True
            )

    def test_the_skeleton_path_passes_an_extension_node(self, nz_composed) -> None:
        spec, config, graph = nz_composed
        skeleton = _Graph(
            [
                node
                for node in graph.nodes
                if ".bridge" not in node.id and ".scn." not in node.id
            ]
        )
        compose._check_graph_node_entries(spec, config, skeleton, extended=False)
        with pytest.raises(ValueError, match="does not hold"):
            compose._check_graph_node_entries(spec, config, skeleton, extended=True)


# ---------------------------------------------------------------------------
# The always-pass stub mutant (design §6)
# ---------------------------------------------------------------------------


def _always_pass(**kwargs) -> GateResult:
    return GateResult(GATE, True, (), {})


def test_an_always_pass_stub_passes_forged_evidence(passing_stub_world) -> None:
    # The mutant test_no_registered_binding_passes_without_checked_evidence
    # (test_transport_gate_bindings.py) must catch: this stub accepts forged
    # evidence that the registered binding refuses.
    frame, receipt, closure = passing_stub_world
    forged = _forged(
        closure, lambda d: d["producer"].update(implementation_hash="0" * 64)
    )
    stub = {
        **TRANSPORT_GATE_REGISTRY,
        GATE: replace(TRANSPORT_GATE_REGISTRY[GATE], gate=_always_pass),
    }
    artifacts = _decoded(receipt, forged)
    assert _evaluate(artifacts, frame, registry=stub).status is GateStatus.PASSED
    assert _evaluate(artifacts, frame).status is GateStatus.FAILED


@pytest.fixture(scope="module")
def passing_stub_world(toy, donor):
    spec = _with_commit(toy, "a" * 40)
    frame = _frame(spec["resources"]["axiom_input_closure"])
    return frame, _run_receipt(donor, frame), _run_closure(toy, spec=spec, frame=frame)


# ---------------------------------------------------------------------------
# transport.weights_audit@1 and the transport weights_audit gate (H30, H44 B)
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from scipy import sparse  # noqa: E402

from microcosm.build.transport.evidence_contract import (  # noqa: E402
    FIT_WEIGHT_RECORDS,
    WEIGHTS_AUDIT_REF,
    WEIGHTS_AUDIT_TYPE,
    decode_weights_audit,
)
from microcosm.build.transport.evidence_kernels import (  # noqa: E402
    TRANSPORT_WEIGHTS_AUDIT,
)
from microcosm.calibrate import CalibrationProblem, Target  # noqa: E402
from microcosm.calibrate.artifacts import (  # noqa: E402
    PROBLEM_TYPE,
    RESULT_TYPE,
    SOLUTION_TYPE,
    decode_problem,
    encode_calibration_result,
    encode_problem,
    encode_solution,
)
from microcosm.graph.executor import Population, _project_context  # noqa: E402
from microcosm.graph.kernel import NumericScope  # noqa: E402

UNITS_SCHEMA = EntitySchema(group_entities=("household", "tax_unit", "family"))
HOUSEHOLD_IDS = [10, 20, 30]
PERSON_HOUSEHOLD = [10, 10, 10, 20, 20, 30]
PERSON_TAX_UNIT = [1, 1, 2, 3, 3, 4]
PERSON_FAMILY = [100, 100, 101, 102, 102, 103]
DESIGN = np.array([2.0, 3.0, 5.0])
CALIBRATED = np.array([2.2, 2.7, 5.0])


def _units_frame(stored: dict | None = None, *, calibrated=CALIBRATED) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype=np.int64),
            "person_household_id": np.array(PERSON_HOUSEHOLD, dtype=np.int64),
            "person_tax_unit_id": np.array(PERSON_TAX_UNIT, dtype=np.int64),
            "person_family_id": np.array(PERSON_FAMILY, dtype=np.int64),
            "age": np.arange(30, 36, dtype=np.int64),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.array(HOUSEHOLD_IDS, dtype=np.int64),
            "source_household_id": pd.array(["s1", "s1", "s2"], dtype="string"),
        }
    )
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": np.array([1, 2, 3, 4], dtype=np.int64),
            "tax_unit_flag": [1, 0, 1, 0],
        }
    )
    family = pd.DataFrame(
        {
            "family_id": np.array([100, 101, 102, 103], dtype=np.int64),
            "family_flag": [0, 1, 0, 1],
        }
    )
    weights = {"household": Weights(np.asarray(calibrated), WeightKind.CALIBRATED)}
    for entity, values in (stored or {}).items():
        weights[entity] = Weights(
            np.asarray(values, dtype=np.float64), WeightKind.CALIBRATED
        )
    return Frame(
        {
            "person": person,
            "household": household,
            "tax_unit": tax_unit,
            "family": family,
        },
        UNITS_SCHEMA,
        weights,
    )


def _calibration(design=DESIGN, calibrated=CALIBRATED) -> dict[str, bytes]:
    target = Target(
        "count",
        "household",
        lambda f: np.ones(len(HOUSEHOLD_IDS)),
        value=10,
        period=2026,
    )
    problem = CalibrationProblem(
        sparse.csr_array([[1.0] * len(HOUSEHOLD_IDS)]),
        np.array([10.0]),
        (target.row_name,),
        Weights(np.asarray(design, dtype=np.float64), WeightKind.DESIGN),
        "household",
        (target,),
    )
    problem_payload = encode_problem(problem, entity_ids=HOUSEHOLD_IDS)
    sha = decode_problem(problem_payload).sha256
    result = SimpleNamespace(
        weights=np.asarray(calibrated, dtype=np.float64),
        initial_weights=np.asarray(design, dtype=np.float64),
        options={},
        gate_open_probabilities=None,
        l0_lambda=0.0,
        n_nonzero=len(HOUSEHOLD_IDS),
        target_loss_cap=1.0,
        final_loss=0.0,
        loss_trajectory=np.zeros(1),
        target_loss_weights=np.ones(1),
        target_loss_scales=np.ones(1),
    )
    return {
        "problem": problem_payload,
        "solution": encode_solution(
            calibrated, entity_ids=HOUSEHOLD_IDS, problem_sha256=sha
        ),
        "result": encode_calibration_result(
            result, entity_ids=HOUSEHOLD_IDS, problem_sha256=sha
        ),
    }


_CALIBRATION_TYPES = {
    "problem": PROBLEM_TYPE,
    "solution": SOLUTION_TYPE,
    "result": RESULT_TYPE,
}
_CALIBRATION_PRODUCERS = {
    "problem": "toy.targets.problem",
    "solution": "toy.calibrate",
    "result": "toy.calibrate",
}


def _weights_node(units=("tax_unit", "family")) -> Node:
    return Node(
        "toy.evidence.weights",
        WEIGHTS_AUDIT_REF,
        inputs=(
            Slice("household", ("source_household_id",)),
            Slice("person", ("age",)),
            *(Slice(unit, (f"{unit}_flag",)) for unit in units),
        ),
        params={
            "weight_entity": "household",
            "unit_entities": _text({"unit_entities": list(units)}),
        },
        artifact_inputs=tuple(
            ArtifactInput(alias, _CALIBRATION_PRODUCERS[alias], alias, kind)
            for alias, kind in _CALIBRATION_TYPES.items()
        ),
        artifact_outputs=(ArtifactOutput(FIT_WEIGHT_RECORDS, WEIGHTS_AUDIT_TYPE),),
    )


def _artifact_values(payloads):
    return {
        alias: ArtifactValue(
            payload,
            _CALIBRATION_TYPES[alias],
            _sha(alias.encode()),
            _sha(payload),
            NumericScope(),
        )
        for alias, payload in payloads.items()
    }


def _run_weights(frame: Frame, payloads=None, units=("tax_unit", "family")) -> bytes:
    payloads = payloads or _calibration()
    node = _weights_node(units)
    context = _project_context(
        node,
        Population.from_frame(frame, "toy.terminal"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
        artifacts=_artifact_values(payloads),
    )
    return TRANSPORT_WEIGHTS_AUDIT.run(context).artifacts[FIT_WEIGHT_RECORDS]


def _weights_edges(audit: bytes, payloads) -> dict:
    edges = {
        FIT_WEIGHT_RECORDS: {
            "producer": "toy.evidence.weights",
            "artifact": FIT_WEIGHT_RECORDS,
            "type": [WEIGHTS_AUDIT_TYPE.name, WEIGHTS_AUDIT_TYPE.schema_version],
            "sha256": _sha(audit),
            "producer_key": "c" * 64,
        }
    }
    for alias, payload in payloads.items():
        edges[alias] = {
            "producer": _CALIBRATION_PRODUCERS[alias],
            "artifact": alias,
            "type": [
                _CALIBRATION_TYPES[alias].name,
                _CALIBRATION_TYPES[alias].schema_version,
            ],
            "sha256": _sha(payload),
            "producer_key": "d" * 64,
        }
    return edges


def _weights_entry(limit=0.02):
    return GatesManifest.from_mapping(
        {
            "country": "xx",
            "version": 1,
            "policy": "Synthetic weights audit.",
            "phases": ["terminal"],
            "gates": [
                {
                    "id": "toy_weights_audit",
                    "gate": "weights_audit",
                    "phase": "terminal",
                    "criticality": "release_blocking",
                    "parameters": {"maximum_relative_mass_change": limit},
                }
            ],
        },
        country="xx",
    )


def _weights_outcome(
    audit, payloads, frame, *, limit=0.02, registry=TRANSPORT_GATE_REGISTRY, edges=None
):
    report = evaluate_phase(
        _weights_entry(limit),
        "terminal",
        EvidenceContext(
            frame=frame,
            artifacts={
                FIT_WEIGHT_RECORDS: decode_weights_audit(audit),
                GRAPH_EVIDENCE: MappingProxyType(
                    edges or _weights_edges(audit, payloads)
                ),
            },
        ),
        registry=registry,
    )
    (outcome,) = report.outcomes
    return outcome


def _broadcast(values, membership=PERSON_HOUSEHOLD):
    lookup = dict(zip(HOUSEHOLD_IDS, values, strict=True))
    return np.array([lookup[h] for h in membership])


#: H44 B's seven frames: stored weights, and whether the audit passes.
SEVEN_FRAMES = {
    "household_only": ({}, True),
    "person_byte_equal_to_inherited": ({"person": _broadcast(CALIBRATED)}, True),
    "tax_unit_differing": ({"tax_unit": [1.0, 2.0, 3.0, 4.0]}, False),
    "tax_unit_equal_to_inherited": ({"tax_unit": [2.2, 2.2, 2.7, 5.0]}, False),
    "family_vector": ({"family": [2.2, 2.2, 2.7, 5.0]}, False),
    "person_constant_within_units": ({"person": _broadcast(CALIBRATED) * 2}, False),
    "person_varying_within_a_unit": (
        {"person": [2.2, 2.3, 2.2, 2.7, 2.7, 5.0]},
        False,
    ),
}


class TestWeightsAudit:
    @pytest.mark.parametrize("name", sorted(SEVEN_FRAMES))
    def test_the_seven_frames(self, name) -> None:
        stored, passes = SEVEN_FRAMES[name]
        frame = _units_frame(stored)
        payloads = _calibration()
        audit = _run_weights(frame, payloads)
        document = decode_weights_audit(audit).document
        assert [row["entity"] for row in document["partitions"]] == [
            "person",
            "tax_unit",
            "family",
        ]
        outcome = _weights_outcome(audit, payloads, frame)
        assert (outcome.status is GateStatus.PASSED) is passes, outcome.result.failures
        if not passes:
            assert any("second weight vector" in f for f in outcome.result.failures)
        if name == "person_byte_equal_to_inherited":
            # The rule's limit: a node cannot see this vector, and it carries
            # no second value.
            assert all(row["inherited"] for row in document["partitions"])

    def test_the_audit_records_the_transition_and_mass(self) -> None:
        frame = _units_frame()
        audit = _run_weights(frame)
        document = decode_weights_audit(audit).document
        assert document["fit_records"] == [
            {"fit_name": "toy.calibrate", "weight_kind": "calibrated"}
        ]
        transition = document["transition"]
        assert (transition["from_kind"], transition["to_kind"]) == (
            "design",
            "calibrated",
        )
        assert transition["household_id"] == HOUSEHOLD_IDS
        assert transition["source_household_id"] == ["s1", "s1", "s2"]
        mass = document["mass"]
        persons = _broadcast(DESIGN)
        assert mass["destination_person_mass"] == math.fsum(persons.tolist())
        assert mass["calibrated_person_mass"] == math.fsum(
            _broadcast(CALIBRATED).tolist()
        )

    def test_a_result_from_another_solution_is_refused(self) -> None:
        payloads = _calibration()
        payloads["result"] = _calibration(calibrated=np.array([1.0, 1.0, 1.0]))[
            "result"
        ]
        with pytest.raises(ValueError):
            _run_weights(_units_frame(), payloads)

    def test_frame_weights_differing_from_the_solution_are_refused(self) -> None:
        with pytest.raises(ValueError, match="differ from the calibration"):
            _run_weights(_units_frame(calibrated=np.array([1.0, 2.0, 3.0])))

    @PROPERTY
    @given(
        design=st.lists(st.floats(0.5, 50.0), min_size=3, max_size=3),
        scale=st.lists(st.floats(0.5, 1.5), min_size=3, max_size=3),
        limit=st.floats(0.0, 0.5),
    )
    def test_property_the_gate_passes_iff_the_mass_change_is_within_the_limit(
        self, design, scale, limit
    ) -> None:
        design = np.asarray(design)
        calibrated = design * np.asarray(scale)
        payloads = _calibration(design, calibrated)
        frame = _units_frame(calibrated=calibrated)
        audit = _run_weights(frame, payloads)
        mass = decode_weights_audit(audit).document["mass"]
        destination = math.fsum(_broadcast(design).tolist())
        assert mass["destination_person_mass"] == destination
        change = (
            math.fsum(_broadcast(calibrated).tolist()) - destination
        ) / destination
        outcome = _weights_outcome(audit, payloads, frame, limit=limit)
        assert (outcome.status is GateStatus.PASSED) is (abs(change) <= limit)

    def test_source_household_concentration_equals_the_weight_gates(self) -> None:
        from microcosm.build.transport.gate_bindings import weight_summary

        frame = _units_frame()
        document = decode_weights_audit(_run_weights(frame)).document
        expected = weight_summary(CALIBRATED, source_household_ids=["s1", "s1", "s2"])
        row = document["source_households"]["calibrated"]
        assert row["effective_sample_size"] == pytest.approx(
            expected["effective_sample_size"], rel=1e-12
        )
        assert row["max_to_median_positive_weight"] == pytest.approx(
            expected["max_to_median_positive_weight"], rel=1e-12
        )

    @pytest.mark.parametrize(
        ("edit", "match"),
        [
            (
                lambda d: d["producer"].update(implementation_hash="0" * 64),
                "implementation hash",
            ),
            (
                lambda d: d["inputs"]["artifacts"].update(solution="0" * 64),
                "input artifact 'solution'",
            ),
            (
                lambda d: d["transition"].update(from_kind="calibrated"),
                "not design to calibrated",
            ),
            (lambda d: d["mass"].update(relative_change=0.5), "declared limit"),
            (lambda d: d["fit_records"][0].update(weight_kind="none"), "unweighted"),
        ],
    )
    def test_forged_or_edited_audits_fail(self, edit, match) -> None:
        frame = _units_frame()
        payloads = _calibration()
        audit = _forged(_run_weights(frame, payloads), edit)
        outcome = _weights_outcome(audit, payloads, frame)
        assert outcome.status is GateStatus.FAILED
        assert any(match in line for line in outcome.result.failures), (
            outcome.result.failures
        )

    def test_the_default_and_uk_bindings_are_untouched(self) -> None:
        from microcosm.build.gate_battery import DEFAULT_REGISTRY
        from microcosm.build.gates import weights_audit_gate

        default = DEFAULT_REGISTRY["weights_audit"]
        assert default.gate is weights_audit_gate
        assert default.parameter_keys == {"allowed_unweighted"}
        assert dict(default.artifact_arguments) == {"fit_records": "fit_weight_records"}
        transport = TRANSPORT_GATE_REGISTRY["weights_audit"]
        assert transport is not default
        assert dict(transport.artifact_arguments) == {
            FIT_WEIGHT_RECORDS: FIT_WEIGHT_RECORDS,
            GRAPH_EVIDENCE: GRAPH_EVIDENCE,
        }

    def test_a_weights_row_on_another_entity_is_refused(self, toy) -> None:
        audit = Node(
            "toy.evidence.weights",
            WEIGHTS_AUDIT_REF,
            params={"weight_entity": "household"},
        )
        compose._check_weights_audit_entity(toy.spec, _Graph([audit]))
        mutant = copy.deepcopy(toy.spec)
        row = next(
            row
            for row in mutant["resources"]["transport_graph"]["nodes"]
            if isinstance(row.get("weights"), dict)
        )
        row["weights"]["entity"] = "family"
        with pytest.raises(ValueError, match="second weight vector"):
            compose._check_weights_audit_entity(mutant, _Graph([audit]))


import math  # noqa: E402
