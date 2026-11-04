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
#: What the leaf adds to the battery's binding identity (H42).
LEAF_ADDED_MODULES = frozenset(
    [
        "microcosm.build.transport.evidence_contract",
        "microcosm.frame.adapters.axiom_input_surface",
        "microcosm.frame.input_closure",
        "microcosm.graph",
        "microcosm.graph.artifact_edges",
        "microcosm.graph.availability",
        "microcosm.graph.canonical",
        "microcosm.graph.decl",
        "microcosm.graph.errors",
        "microcosm.graph.evidence",
        "microcosm.graph.execution_overlay",
        "microcosm.graph.expansion",
        "microcosm.graph.kernel",
        "microcosm.graph.keys",
        "microcosm.graph.manifest",
        "microcosm.graph.orrery",
        "microcosm.graph.orrery_evidence",
        "microcosm.graph.population",
        "microcosm.graph.presentation",
        "microcosm.graph.schema",
        "microcosm.graph.serialize",
        "microcosm.graph.store",
        "microcosm.graph.weight_update",
    ]
)
#: Modules the bindings reach without the leaf that base 25c8cc14 did not
#: have. Since #1342, ``microcosm.calibrate.diagnostics`` imports
#: ``microcosm.calibrate.hierarchy_labels`` and the ``microcosm.diagnostics``
#: package imports ``microcosm.diagnostics.catalogs``. ``gate_bindings``
#: imports ``microcosm.calibrate.solve``, whose package ``__init__`` imports
#: both ``microcosm.calibrate.diagnostics`` and ``microcosm.diagnostics``.
BINDING_MODULES_ADDED_AFTER_BASE = frozenset(
    [
        "microcosm.calibrate.hierarchy_labels",
        "microcosm.diagnostics.catalogs",
    ]
)
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
            "transport.donor_ranges@1": (
                "microcosm.build.transport.evidence_kernels",
                "microcosm.graph.canonical",
                "microcosm.build.transport.evidence_contract",
            ),
            "transport.support_bounds@1": (
                "microcosm.build.transport.evidence_kernels",
                "microcosm.build.transport.money_kernels",
                "microcosm.build.money_normalization",
                "microcosm.build.reference_factors",
                "microcosm.build.ledger_artifact",
                "microcosm.build.ledger_targets",
                "microcosm.build.chronicle_epoch",
                "microcosm.build.transport.graph_inputs",
                "microcosm.frame.concepts",
                "microcosm.frame.exact_money",
                "microcosm.frame.transport",
                # S2: the trace finds it in run(); the list missed it.
                "microcosm.frame.scaling",
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
        # At base 25c8cc14 the battery's binding identity held 197 modules;
        # w4 adds the leaf and exactly the modules it imports that the
        # bindings did not already reach (the graph package among them).
        # Modules the bindings later reach without the leaf are listed in
        # BINDING_MODULES_ADDED_AFTER_BASE.
        modules = set()
        for name, binding in TRANSPORT_GATE_REGISTRY.items():
            _, digests = binding_identity.describe(binding, path=name)
            modules |= set(digests)
        assert len(modules) == (
            197 + len(BINDING_MODULES_ADDED_AFTER_BASE) + len(LEAF_ADDED_MODULES)
        )
        assert BINDING_MODULES_ADDED_AFTER_BASE <= modules
        assert LEAF_ADDED_MODULES <= modules
        assert LEAF_ADDED_MODULES <= set(
            binding_identity.source_closure([evidence_contract.__name__])
        )
        assert "microcosm.build.transport.registry" not in modules
        assert "microcosm.build.transport.evidence_kernels" not in modules


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
    # The support producers: trace only their run() while the existing
    # support tests drive them (copied, quantile, component, currency,
    # structural, and normalized rows through the money helpers).
    support_traces = {DONOR_RANGES_REF: set(), SUPPORT_BOUNDS_REF: set()}
    with pytest.MonkeyPatch.context() as patch:
        for kernel in (TRANSPORT_DONOR_RANGES, TRANSPORT_SUPPORT_BOUNDS):
            real = type(kernel).run

            def run(self, context, real=real):
                result = []
                support_traces[self.ref].update(
                    _traced(lambda: result.append(real(self, context)))
                )
                return result[0]

            patch.setattr(type(kernel), "run", run)
        donor_frame, terminal = _donor_and_terminal()
        _run_support(_toy_roster(), donor_frame, terminal)
        TestComponentSupport()._run(
            (5.0, 7.0, 0.0, 120.0, 0.0, 150.0), ["a", "a", "a", "b", "b", "b"]
        )
        TestNormalizedSupport().test_differential_bounds_equal_money_normalize_outputs(
            tmp_path_factory.mktemp("money")
        )
    traces.update(support_traces)
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

    def test_relation_columns_on_another_entitys_slice_are_a_gap(self) -> None:
        # S1: the partition reads a relation's columns from the person slice
        # only; the same columns on the household or family slice do not count.
        relation = {**_TOY_RELATION, "role_column": "role", "roles": ["head"]}
        for entity in ("household", "family"):
            row = self._partition(
                {"person": frozenset(), entity: frozenset({"member_income", "role"})},
                relations=[relation],
            )
            assert row["relation"] == []
            assert row["gap"] == ["Family.toy_member_income"]
            assert row["relation_missing"] == [
                "Family.toy_member_income: person.member_income",
                "Family.toy_member_income: person.role",
            ]


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

    def test_a_relation_on_an_unmapped_root_entity_is_refused(self, tmp_path) -> None:
        # S1: entity_names maps no engine entity for the manifest's
        # root_entity (family), so engine_root_inputs refuses the binding.
        engine = _fake_axiom(tmp_path / "m.yaml")
        row = {"id": "b", "relations": [_TOY_RELATION]}
        names = {"person": "Person", "household": "Household"}
        with pytest.raises(
            ValueError,
            match=r"rules binding 'b' relates 'toy_member_income' on 'family', "
            r"which entity_names does not map\.",
        ):
            evidence_kernels.engine_root_inputs(engine, row, names)

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


def _units_frame(
    stored: dict | None = None,
    *,
    calibrated=CALIBRATED,
    households=HOUSEHOLD_IDS,
    person_household=PERSON_HOUSEHOLD,
    person_tax_unit=PERSON_TAX_UNIT,
    person_family=PERSON_FAMILY,
    sources=("s1", "s1", "s2"),
) -> Frame:
    count = len(person_household)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype=np.int64),
            "person_household_id": np.array(person_household, dtype=np.int64),
            "person_tax_unit_id": np.array(person_tax_unit, dtype=np.int64),
            "person_family_id": np.array(person_family, dtype=np.int64),
            "age": np.arange(30, 30 + count, dtype=np.int64),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.array(households, dtype=np.int64),
            "source_household_id": pd.array(list(sources), dtype="string"),
        }
    )
    tax_units = sorted(set(person_tax_unit))
    families = sorted(set(person_family))
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": np.array(tax_units, dtype=np.int64),
            "tax_unit_flag": [index % 2 for index in range(len(tax_units))],
        }
    )
    family = pd.DataFrame(
        {
            "family_id": np.array(families, dtype=np.int64),
            "family_flag": [(index + 1) % 2 for index in range(len(families))],
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


def _calibration(
    design=DESIGN, calibrated=CALIBRATED, households=HOUSEHOLD_IDS
) -> dict[str, bytes]:
    target = Target(
        "count",
        "household",
        lambda f: np.ones(len(households)),
        value=10,
        period=2026,
    )
    problem = CalibrationProblem(
        sparse.csr_array([[1.0] * len(households)]),
        np.array([10.0]),
        (target.row_name,),
        Weights(np.asarray(design, dtype=np.float64), WeightKind.DESIGN),
        "household",
        (target,),
    )
    problem_payload = encode_problem(problem, entity_ids=households)
    sha = decode_problem(problem_payload).sha256
    result = SimpleNamespace(
        weights=np.asarray(calibrated, dtype=np.float64),
        initial_weights=np.asarray(design, dtype=np.float64),
        options={},
        gate_open_probabilities=None,
        l0_lambda=0.0,
        n_nonzero=len(households),
        target_loss_cap=1.0,
        final_loss=0.0,
        loss_trajectory=np.zeros(1),
        target_loss_weights=np.ones(1),
        target_loss_scales=np.ones(1),
    )
    return {
        "problem": problem_payload,
        "solution": encode_solution(
            calibrated, entity_ids=households, problem_sha256=sha
        ),
        "result": encode_calibration_result(
            result, entity_ids=households, problem_sha256=sha
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


def _broadcast(values, membership=PERSON_HOUSEHOLD, households=HOUSEHOLD_IDS):
    lookup = dict(zip(households, values, strict=True))
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
    @given(data=st.data(), limit=st.floats(0.0, 0.5))
    def test_property_the_gate_passes_iff_the_mass_change_is_within_the_limit(
        self, data, limit
    ) -> None:
        # Draws the membership (households, the tax units and families over
        # them, source households split across them), zero design and
        # calibrated weights, and a stored second partition vector.
        sizes = data.draw(st.lists(st.integers(1, 3), min_size=1, max_size=4))
        households = [10 * (index + 1) for index in range(len(sizes))]
        person_household, person_tax_unit, person_family = [], [], []
        for household, size in zip(households, sizes, strict=True):
            units = data.draw(st.lists(st.integers(0, 2), min_size=size, max_size=size))
            families = data.draw(
                st.lists(st.integers(0, 2), min_size=size, max_size=size)
            )
            person_household += [household] * size
            person_tax_unit += [household + unit for unit in units]
            person_family += [100 * household + family for family in families]
        sources = data.draw(
            st.lists(
                st.sampled_from(("s1", "s2", "s3")),
                min_size=len(households),
                max_size=len(households),
            )
        )
        weight = st.one_of(st.just(0.0), st.floats(0.5, 50.0))
        design = np.asarray(
            data.draw(
                st.lists(weight, min_size=len(households), max_size=len(households))
            )
        )
        calibrated = np.asarray(
            data.draw(
                st.lists(weight, min_size=len(households), max_size=len(households))
            )
        )
        # Weights refuses an all-zero vector.
        if not (design > 0).any():
            design[0] = 1.0
        if not (calibrated > 0).any():
            calibrated[-1] = 1.0
        stored = data.draw(st.booleans(), label="stored tax_unit vector")
        membership = {
            "households": households,
            "person_household": person_household,
            "person_tax_unit": person_tax_unit,
            "person_family": person_family,
            "sources": sources,
        }
        frame = _units_frame(
            {"tax_unit": np.ones(len(set(person_tax_unit)))} if stored else None,
            calibrated=calibrated,
            **membership,
        )
        payloads = _calibration(design, calibrated, households)
        audit = _run_weights(frame, payloads)
        mass = decode_weights_audit(audit).document["mass"]
        destination = math.fsum(
            _broadcast(design, person_household, households).tolist()
        )
        assert mass["destination_person_mass"] == destination
        change = (
            math.fsum(_broadcast(calibrated, person_household, households).tolist())
            - destination
        ) / destination
        outcome = _weights_outcome(audit, payloads, frame, limit=limit)
        passes = abs(change) <= limit and not stored
        assert (outcome.status is GateStatus.PASSED) is passes, outcome.result.failures
        if stored:
            assert any(
                "second weight vector" in line for line in outcome.result.failures
            )

    def test_zero_weights_fail_on_the_mass_line_only(self) -> None:
        design, calibrated = np.array([2.0, 0.0, 5.0]), np.array([0.0, 0.0, 5.5])
        payloads = _calibration(design, calibrated)
        frame = _units_frame(calibrated=calibrated)
        audit = _run_weights(frame, payloads)
        mass = decode_weights_audit(audit).document["mass"]
        assert mass["relative_change"] == pytest.approx(-0.5)
        outcome = _weights_outcome(audit, payloads, frame)
        assert outcome.status is GateStatus.FAILED
        (line,) = outcome.result.failures
        assert "declared limit" in line or "mass" in line

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

# ---------------------------------------------------------------------------
# Support: the roster (H34, H44 D) and its producers (H35, H39)
# ---------------------------------------------------------------------------
from microcosm.build.transport.evidence_contract import (  # noqa: E402
    DONOR_RANGES,
    DONOR_RANGES_REF,
    DONOR_RANGES_TYPE,
    DONOR_SUPPORT_BOUNDS,
    SUPPORT_BOUNDS_REF,
    SUPPORT_BOUNDS_TYPE,
    decode_support_bounds,
    inside_quantile_support,
    quantile_bounds,
    quantile_support,
)
from microcosm.build.transport.evidence_kernels import (  # noqa: E402
    TRANSPORT_DONOR_RANGES,
    TRANSPORT_SUPPORT_BOUNDS,
)
from microcosm.build.transport.money_kernels import MONEY_NORMALIZE  # noqa: E402
from microcosm.frame.transport import quantile_map  # noqa: E402

H34_STRUCTURAL = (
    ("family", "family_id"),
    ("family", "family_household_id"),
    ("family", "family_head_person_id"),
    ("family", "family_source_household_id"),
    ("person", "person_family_id"),
    ("person", "take_up_seed"),
    ("household", "donor_support_stratum"),
    ("household", "source_household_id"),
    ("household", "reference_person_id"),
    ("person", "parent_1_person_id"),
    ("person", "parent_2_person_id"),
    ("person", "partner_person_id"),
    ("household", "household_region_code"),
    ("household", "as_area"),
    ("household", "as_area_alt"),
    ("person", "region_code"),
)
H44_REFUSED = (
    "person.employment_income",
    "household.rent",
    "person.age",
    "person.sex",
    "household.tenure",
    "person.usual_weekly_hours",
    "person.has_disability",
    "person.legal_marital_status",
    "person.enrolled_full_time",
    "person.public_pension_income",
    "person.dividend_income",
    "person.receives_jobseeker_support",
    "person.entitled_to_jobseeker_support",
    "family.accommodation_supplement_boarder",
)


@pytest.fixture(scope="module")
def nz_spec():
    return compose.load_transport_spec("nz")


def _with_structural(spec, rows):
    spec = copy.deepcopy(spec)
    for entry in spec["resources"]["gates"]["gates"]:
        if entry["gate"] == "support":
            entry["parameters"] = {
                "structural_columns": [
                    {"entity": e, "column": c, "reason": "Synthetic."} for e, c in rows
                ]
            }
    return spec


class TestSupportRoster:
    def test_the_packaged_nz_list_is_h34s_and_passes(self, nz_spec) -> None:
        declared = [
            (row["entity"], row["column"])
            for row in compose.support_structural_columns(nz_spec)
        ]
        assert declared == list(H34_STRUCTURAL)
        compose._check_structural_columns(nz_spec)

    @pytest.mark.parametrize("column", H44_REFUSED)
    def test_a_structural_row_that_excuses_values_is_refused(
        self, nz_spec, column
    ) -> None:
        entity, name = column.split(".")
        with pytest.raises(ValueError, match="may not excuse"):
            compose._check_structural_columns(
                _with_structural(nz_spec, [(entity, name)])
            )

    def test_the_nz_roster_classes(self, nz_spec) -> None:
        roster = compose._support_roster(nz_spec)["columns"]
        counts = {}
        for row in roster:
            counts[row["class"]] = counts.get(row["class"], 0) + 1
        assert counts == {
            "closure_default": 93,
            "derived": 322,
            "normalized": 10,
            "copied": 3,
            "pending": 1,
            "quantile_mapped": 1,
            "structural": 6,
        }
        derived = [row for row in roster if row["class"] == "derived"]
        pending = {
            "household.rent",
            "person.liquid_financial_assets",
        }
        for row in derived:
            missing = {
                f"{s['entity']}.{s['column']}"
                for s in row["sources"]
                if s["class"] not in compose.SUPPORT_COVERED
            }
            assert missing == pending, row["column"]
            assert "walked per node" in row["reason"]

    def test_a_closure_default_is_classed_closure_default(self, nz_spec) -> None:
        # Mutant: dropping the closure_default class makes this a derived row
        # that names donor concepts the constant never reads.
        roster = {
            (row["entity"], row["column"]): row
            for row in compose._support_roster(nz_spec)["columns"]
        }
        row = roster["person", "jobseeker_has_no_income"]
        assert row["class"] == "closure_default"
        assert row["closure_entry"].endswith("Person.jobseeker_has_no_income")
        assert row["owner_kernel"] == "concepts.encode@1"

    def test_differential_owners_match_an_independent_walk(self, toy) -> None:
        # The roster's owner for every bounded or derived row equals the owner
        # the compiled graph records in the reader's population lineage.
        from microcosm.graph import compile_graph

        graph = compose.compose_transport_graph(toy.spec, toy.config)
        compiled = compile_graph(graph)
        nodes = {node.id: node for node in graph.nodes}
        readers = {}
        for node in graph.nodes:
            if node.kernel in {"gates.battery@1", "targets.problem@1"}:
                for item in node.inputs:
                    for column in item.columns:
                        readers.setdefault((item.entity, column), node.population)

        def owner(version, entity, column):
            while version is not None:
                found = compiled.owners.get((version, entity, column))
                if found is not None and nodes[found].kernel not in compose.CREATE_REFS:
                    return found
                version = nodes[version].base if version in nodes else None
            return None

        roster = compose._support_roster(toy.spec)["columns"]
        assert {(row["entity"], row["column"]) for row in roster} == set(readers)
        for row in roster:
            if row["class"] in {"structural"}:
                continue
            assert row.get("owner") == owner(
                readers[row["entity"], row["column"]], row["entity"], row["column"]
            ), (row["entity"], row["column"])

    def test_a_rank_only_declaration_is_refused_with_the_reserved_message(
        self, toy
    ) -> None:
        spec = copy.deepcopy(toy.spec)
        node = spec["resources"]["transport_graph"]["nodes"][1]
        node.setdefault("params", {})["rank_only"] = [
            {"entity": "person", "column": "employment_income"}
        ]
        with pytest.raises(
            ValueError,
            match=r"person\.employment_income: lineage class 'rank_only' is reserved "
            r"for the declared rank-only input \(BE d1902\)",
        ):
            compose._support_roster(spec)


def _impute_spec(*, inputs, sources=("external_wage_table",)):
    """C1's toy: CREATE, an imputation reading an external table, a battery."""

    return {
        "country": "xx",
        "resources": {
            "transport_graph": {
                "schema_version": 1,
                "country": "xx",
                "sources": [
                    {"name": "external_wage_table", "codec": "toy-wage-table-v1"},
                    {"name": "rulespec_toy", "codec": "rulespec-tree-v1"},
                ],
                "nodes": [
                    {
                        "id": "x.open",
                        "kernel": "transport.create@1",
                        "structural": "create",
                    },
                    {
                        "id": "x.impute",
                        "kernel": "toy.impute@1",
                        "sources": list(sources),
                        "inputs": [
                            {"entity": "person", "columns": list(columns)}
                            for columns in [inputs]
                            if columns
                        ],
                        "outputs": [
                            {
                                "entity": "person",
                                "column": "imputed_wage",
                                "dtype": "float64",
                            }
                        ],
                    },
                    {
                        "id": "x.gates",
                        "kernel": "gates.battery@1",
                        "inputs": [{"entity": "person", "columns": ["imputed_wage"]}],
                    },
                ],
            }
        },
    }


class TestExternalSources:
    @pytest.mark.parametrize("inputs", (["age"], []), ids=("with_inputs", "no_inputs"))
    def test_an_imputation_from_an_external_source_is_uncovered(self, inputs) -> None:
        # C1: the owner reads an external table, so its row has no support.
        roster = compose._support_roster(_impute_spec(inputs=inputs))
        (row,) = [r for r in roster["columns"] if r["column"] == "imputed_wage"]
        if inputs:
            assert row["class"] == "derived"
            external = [
                s for s in row["sources"] if s["column"] == "external_wage_table"
            ]
            assert external and external[0]["class"] == "uncovered"
            assert "toy-wage-table-v1" in external[0]["reason"]
        else:
            assert row["class"] == "derived"
        donor, _ = _donor_and_terminal()
        terminal = _support_frame({("person", "imputed_wage"): [1e9] * 6})
        ranges, bounds, rn, bn = _run_support(roster, donor, terminal)
        outcome = _support_outcome(ranges, bounds, rn, bn, terminal)
        assert outcome.status is GateStatus.FAILED
        assert any("external_wage_table" in line for line in outcome.result.failures)

    def test_a_derived_row_with_no_walked_source_is_uncovered(self) -> None:
        roster = compose._support_roster(_impute_spec(inputs=[], sources=()))
        (row,) = [r for r in roster["columns"] if r["column"] == "imputed_wage"]
        assert row["class"] == "uncovered"
        donor, _ = _donor_and_terminal()
        terminal = _support_frame({("person", "imputed_wage"): [1e9] * 6})
        ranges, bounds, rn, bn = _run_support(roster, donor, terminal)
        outcome = _support_outcome(ranges, bounds, rn, bn, terminal)
        assert outcome.status is GateStatus.FAILED

    def test_the_rulespec_tree_is_no_external_source(self) -> None:
        # Matched on the codec, not the name.
        roster = compose._support_roster(
            _impute_spec(inputs=["age"], sources=("rulespec_toy",))
        )
        (row,) = [r for r in roster["columns"] if r["column"] == "imputed_wage"]
        assert row["class"] == "derived"
        assert [s["column"] for s in row["sources"]] == ["age"]

    def test_a_rulespec_name_with_another_codec_is_uncovered(self) -> None:
        # C1: the RuleSpec tree is matched on its codec. A source named like
        # the tree but carrying another codec is an external source.
        spec = _impute_spec(inputs=["age"], sources=("rulespec_nz",))
        spec["resources"]["transport_graph"]["sources"].append(
            {"name": "rulespec_nz", "codec": "toy-wage-table-v1"}
        )
        roster = compose._support_roster(spec)
        (row,) = [r for r in roster["columns"] if r["column"] == "imputed_wage"]
        assert row["class"] == "derived"
        (external,) = [s for s in row["sources"] if s["column"] == "rulespec_nz"]
        assert external["class"] == "uncovered"
        assert "toy-wage-table-v1" in external["reason"]
        donor, _ = _donor_and_terminal()
        terminal = _support_frame({("person", "imputed_wage"): [1e9] * 6})
        ranges, bounds, rn, bn = _run_support(roster, donor, terminal)
        outcome = _support_outcome(ranges, bounds, rn, bn, terminal)
        assert outcome.status is GateStatus.FAILED
        assert any(
            "person.imputed_wage (derived): sources without coverage: "
            "source.rulespec_nz (uncovered)" in line
            for line in outcome.result.failures
        ), outcome.result.failures


class TestDeclaredStructuralRows:
    def test_a_row_no_node_writes_is_refused(self, nz_composed) -> None:
        # H46: household.employment_income is a stale, misspelled row.
        spec, _, graph = nz_composed
        compose._check_structural_rows_written(spec, graph)
        mutant = _with_structural(
            spec, [*H34_STRUCTURAL, ("household", "employment_income")]
        )
        with pytest.raises(ValueError, match=r"household\.employment_income"):
            compose._check_structural_rows_written(mutant, graph)

    def test_a_declared_unreached_row_is_listed_and_does_not_fail(self) -> None:
        donor, terminal = _donor_and_terminal()
        ranges, bounds, rn, bn = _run_support(_toy_roster(), donor, terminal)
        base = _support_outcome(
            ranges,
            bounds,
            rn,
            bn,
            terminal,
            structural=[("household", "source_household_id")],
        )
        extra = _support_outcome(
            ranges,
            bounds,
            rn,
            bn,
            terminal,
            structural=[
                ("household", "source_household_id"),
                ("person", "never_reached"),
            ],
        )
        assert extra.result.details["declared_unreached"] == ["person.never_reached"]
        assert base.result.details["declared_unreached"] == []
        assert extra.status is base.status is GateStatus.PASSED
        assert extra.result.failures == base.result.failures


# --- the producers on hand-built contexts ---------------------------------


def _support_frame(columns: dict) -> Frame:
    frame = _units_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for (entity, column), values in columns.items():
        tables[entity][column] = values
    return Frame(tables, UNITS_SCHEMA, {"household": frame.weights_for("household")})


def _support_node(
    kernel,
    artifact,
    artifact_type,
    roster,
    columns,
    *,
    params=None,
    inputs=None,
    sources=(),
):
    by_entity: dict[str, list[str]] = {}
    for entity, column in columns:
        by_entity.setdefault(entity, []).append(column)
    return Node(
        f"toy.evidence.{artifact}",
        kernel,
        inputs=tuple(Slice(e, tuple(c)) for e, c in sorted(by_entity.items())),
        sources=sources,
        params={"support_roster": _text(roster), **(params or {})},
        artifact_inputs=inputs or (),
        artifact_outputs=(ArtifactOutput(artifact, artifact_type),),
    )


def _support_context(frame, node, artifacts=None, sources=None):
    return _project_context(
        node,
        Population.from_frame(frame, "toy.terminal"),
        key="0" * 64,
        sources=sources or {},
        tolerances={},
        numerics={},
        artifacts=artifacts or {},
    )


def _run_support(
    roster,
    donor_frame,
    terminal_frame,
    *,
    params=None,
    sources=None,
    feed_source=(),
    extra_columns=(),
):
    ranged = [
        (row["entity"], row["column"])
        for row in roster["columns"]
        if row["class"] in {"copied", "normalized", "currency"}
    ]
    ranges_node = _support_node(
        DONOR_RANGES_REF, DONOR_RANGES, DONOR_RANGES_TYPE, roster, ranged
    )
    ranges = TRANSPORT_DONOR_RANGES.run(
        _support_context(donor_frame, ranges_node)
    ).artifacts[DONOR_RANGES]
    bounded = [
        (row["entity"], row["column"])
        for row in roster["columns"]
        if row["class"] in {"copied", "normalized", "currency", "quantile_mapped"}
    ] + list(extra_columns)
    node = _support_node(
        SUPPORT_BOUNDS_REF,
        DONOR_SUPPORT_BOUNDS,
        SUPPORT_BOUNDS_TYPE,
        roster,
        bounded,
        params=params,
        sources=feed_source,
        inputs=(
            ArtifactInput(
                DONOR_RANGES, ranges_node.id, DONOR_RANGES, DONOR_RANGES_TYPE
            ),
        ),
    )
    value = ArtifactValue(
        ranges, DONOR_RANGES_TYPE, _sha(b"r"), _sha(ranges), NumericScope()
    )
    context = _support_context(terminal_frame, node, {DONOR_RANGES: value}, sources)
    bounds = TRANSPORT_SUPPORT_BOUNDS.run(context).artifacts[DONOR_SUPPORT_BOUNDS]
    return ranges, bounds, ranges_node.id, node.id


def _support_edges(ranges, bounds, ranges_node, bounds_node):
    return {
        DONOR_SUPPORT_BOUNDS: {
            "producer": bounds_node,
            "artifact": DONOR_SUPPORT_BOUNDS,
            "type": [SUPPORT_BOUNDS_TYPE.name, SUPPORT_BOUNDS_TYPE.schema_version],
            "sha256": _sha(bounds),
            "producer_key": "c" * 64,
        },
        DONOR_RANGES: {
            "producer": ranges_node,
            "artifact": DONOR_RANGES,
            "type": [DONOR_RANGES_TYPE.name, DONOR_RANGES_TYPE.schema_version],
            "sha256": _sha(ranges),
            "producer_key": "d" * 64,
        },
    }


def _support_outcome(
    ranges, bounds, ranges_node, bounds_node, frame, structural=(), edges=None
):
    manifest = GatesManifest.from_mapping(
        {
            "country": "xx",
            "version": 1,
            "policy": "Synthetic support.",
            "phases": ["terminal"],
            "gates": [
                {
                    "id": "toy_support",
                    "gate": "support",
                    "phase": "terminal",
                    "criticality": "release_blocking",
                    "parameters": {
                        "structural_columns": [
                            {"entity": e, "column": c, "reason": "Synthetic."}
                            for e, c in structural
                        ]
                    },
                }
            ],
        },
        country="xx",
    )
    report = evaluate_phase(
        manifest,
        "terminal",
        EvidenceContext(
            frame=frame,
            artifacts={
                DONOR_SUPPORT_BOUNDS: decode_support_bounds(bounds),
                GRAPH_EVIDENCE: MappingProxyType(
                    edges or _support_edges(ranges, bounds, ranges_node, bounds_node)
                ),
            },
        ),
        registry=TRANSPORT_GATE_REGISTRY,
    )
    (outcome,) = report.outcomes
    return outcome


_BANDS = [
    {"lower": 100.0, "upper": 200.0, "reference": "low"},
    {"lower": 200.0, "upper": 400.0, "reference": "high"},
]


def _toy_roster(**extra):
    columns = [
        {"entity": "person", "column": "age", "class": "copied"},
        {"entity": "household", "column": "tenure_code", "class": "copied"},
        {
            "entity": "person",
            "column": "wage",
            "class": "quantile_mapped",
            "producer_params": {
                "bands": _BANDS,
                "interpolation": {"method": "uniform"},
            },
        },
        {
            "entity": "person",
            "column": "interest_income",
            "class": "currency",
            "producer_params": {
                "rate": 1.6,
                "columns": {"person": ["interest_income"]},
            },
        },
        {
            "entity": "household",
            "column": "source_household_id",
            "class": "structural",
            "reason": "Synthetic.",
        },
        {"entity": "tax_unit", "column": "tax_unit_flag", "class": "copied"},
        {"entity": "family", "column": "family_flag", "class": "copied"},
    ]
    columns += extra.get("more", [])
    return {"columns": columns}


def _donor_and_terminal(*, wage=(150.0, 0.0, 399.0, 120.0, 0.0, 250.0), age_shift=0):
    donor = _support_frame(
        {
            ("person", "wage"): [5.0, 0.0, 9.0, 1.0, 0.0, 3.0],
            ("person", "interest_income"): [1.0, 2.0, 0.0, 3.0, 4.0, 5.0],
            ("household", "tenure_code"): pd.array(
                ["own", "rent", "own"], dtype="string"
            ),
        }
    )
    terminal = _support_frame(
        {
            ("person", "wage"): list(wage),
            ("person", "interest_income"): [1.6, 3.2, 0.0, 4.8, 6.4, 8.0],
            ("household", "tenure_code"): pd.array(
                ["own", "rent", "rent"], dtype="string"
            ),
            ("person", "age"): (np.arange(30, 36) + age_shift).astype(np.int64),
        }
    )
    return donor, terminal


class TestSupportBounds:
    def test_every_bounded_class_passes_inside_its_support(self) -> None:
        donor, terminal = _donor_and_terminal()
        ranges, bounds, rn, bn = _run_support(_toy_roster(), donor, terminal)
        rows = {
            (r["entity"], r["column"]): r
            for r in decode_support_bounds(bounds).document["columns"]
        }
        assert rows["person", "age"]["expected_min"] == 30.0
        assert rows["person", "wage"]["expected_max"] == 400.0
        assert rows["person", "interest_income"]["expected_max"] == 8.0
        assert rows["household", "tenure_code"]["expected_categories"] == [
            '"own"',
            '"rent"',
        ]
        assert all(row["records_outside"] in (0, None) for row in rows.values())
        outcome = _support_outcome(
            ranges,
            bounds,
            rn,
            bn,
            terminal,
            structural=[("household", "source_household_id")],
        )
        assert outcome.status is GateStatus.PASSED, outcome.result.failures

    def test_values_outside_their_support_fail_with_counts(self) -> None:
        donor, terminal = _donor_and_terminal(
            wage=(150.0, 0.0, 401.0, 50.0, 0.0, 250.0), age_shift=1
        )
        ranges, bounds, rn, bn = _run_support(_toy_roster(), donor, terminal)
        outcome = _support_outcome(
            ranges,
            bounds,
            rn,
            bn,
            terminal,
            structural=[("household", "source_household_id")],
        )
        assert outcome.status is GateStatus.FAILED
        lines = outcome.result.failures
        assert any(
            line.startswith("person.wage (quantile_mapped): 2 records")
            for line in lines
        )
        assert any(line.startswith("person.age (copied): 1 records") for line in lines)

    def test_an_undeclared_structural_row_fails(self) -> None:
        donor, terminal = _donor_and_terminal()
        ranges, bounds, rn, bn = _run_support(_toy_roster(), donor, terminal)
        outcome = _support_outcome(ranges, bounds, rn, bn, terminal, structural=[])
        assert any("does not declare it" in line for line in outcome.result.failures)

    def test_derived_pending_and_uncovered_rows_fail_with_their_reasons(self) -> None:
        donor, terminal = _donor_and_terminal()
        more = [
            {
                "entity": "person",
                "column": "flag",
                "class": "derived",
                "sources": [
                    {"entity": "person", "column": "age", "class": "copied"},
                    {
                        "entity": "household",
                        "column": "rent",
                        "class": "pending",
                        "reason": "awaits C1",
                    },
                ],
                "reason": "Sources are walked per node.",
            },
            {
                "entity": "household",
                "column": "rent",
                "class": "pending",
                "reason": "awaits C1",
            },
            {
                "entity": "family",
                "column": "family_flag",
                "class": "uncovered",
                "reason": "no row excuses it",
            },
        ]
        ranges, bounds, rn, bn = _run_support(_toy_roster(more=more), donor, terminal)
        outcome = _support_outcome(
            ranges,
            bounds,
            rn,
            bn,
            terminal,
            structural=[("household", "source_household_id")],
        )
        lines = outcome.result.failures
        assert any(
            "person.flag (derived): sources without coverage: household.rent (pending)"
            in line
            for line in lines
        )
        assert any(
            line.startswith("household.rent (pending): awaits C1") for line in lines
        )
        assert any(line.startswith("family.family_flag (uncovered)") for line in lines)

    def test_a_forged_derived_row_without_sources_fails(self) -> None:
        # C1: the roster classes a sourceless row uncovered, so only a forged
        # artifact carries a derived row with no sources; the gate fails it.
        donor, terminal = _donor_and_terminal()
        more = [
            {
                "entity": "person",
                "column": "flag",
                "class": "derived",
                "sources": [{"entity": "person", "column": "age", "class": "copied"}],
                "reason": "Sources are walked per node.",
            }
        ]
        ranges, bounds, rn, bn = _run_support(_toy_roster(more=more), donor, terminal)
        structural = [("household", "source_household_id")]
        honest = _support_outcome(ranges, bounds, rn, bn, terminal, structural)
        assert not any("person.flag" in line for line in honest.result.failures)

        def empty_sources(document) -> None:
            (row,) = [r for r in document["columns"] if r["column"] == "flag"]
            row["sources"] = []

        forged = _forged(bounds, empty_sources)
        outcome = _support_outcome(ranges, forged, rn, bn, terminal, structural)
        assert outcome.status is GateStatus.FAILED
        assert (
            "person.flag (derived): the walk reached no source, so nothing covers it."
        ) in outcome.result.failures

    def test_a_rank_only_row_is_refused_with_the_reserved_message(self) -> None:
        donor, terminal = _donor_and_terminal()
        more = [{"entity": "person", "column": "pension_rank", "class": "rank_only"}]
        ranges, bounds, rn, bn = _run_support(_toy_roster(more=more), donor, terminal)
        outcome = _support_outcome(
            ranges,
            bounds,
            rn,
            bn,
            terminal,
            structural=[("household", "source_household_id")],
        )
        assert (
            "person.pension_rank: lineage class 'rank_only' is reserved for the declared "
            "rank-only input (BE d1902); no producer implements it yet."
        ) in outcome.result.failures

    @pytest.mark.parametrize(
        ("edit", "match"),
        [
            (
                lambda d: d["producer"].update(implementation_hash="0" * 64),
                "implementation hash",
            ),
            (
                lambda d: d["inputs"]["artifacts"].update(donor_ranges="0" * 64),
                "input artifact 'donor_ranges'",
            ),
            (
                lambda d: d["inputs"]["frame"].update({"person.wage": "0" * 64}),
                "differs from the battery's frame",
            ),
        ],
    )
    def test_forged_or_edited_support_fails(self, edit, match) -> None:
        donor, terminal = _donor_and_terminal()
        ranges, bounds, rn, bn = _run_support(_toy_roster(), donor, terminal)
        bounds = _forged(bounds, edit)
        outcome = _support_outcome(
            ranges,
            bounds,
            rn,
            bn,
            terminal,
            structural=[("household", "source_household_id")],
        )
        assert any(match in line for line in outcome.result.failures), (
            outcome.result.failures
        )

    def test_the_roster_must_name_every_column_the_battery_slices(self) -> None:
        donor, terminal = _donor_and_terminal()
        ranges, bounds, rn, bn = _run_support(_toy_roster(), donor, terminal)
        outcome = _support_outcome(
            ranges,
            bounds,
            rn,
            bn,
            terminal,
            structural=[("household", "source_household_id")],
        )
        # The terminal frame here carries only roster columns.
        assert not any("does not name it" in line for line in outcome.result.failures)
        wider = _support_frame({("person", "unlisted"): [1, 2, 3, 4, 5, 6]})
        outcome = _support_outcome(
            ranges,
            bounds,
            rn,
            bn,
            wider,
            structural=[("household", "source_household_id")],
        )
        assert any(
            "person.unlisted: the battery slices it" in line
            for line in outcome.result.failures
        )


class TestQuantileSupport:
    def test_an_open_pareto_top_records_a_null_bound(self) -> None:
        bands = [
            {"lower": 0.0, "upper": 100.0, "reference": "a"},
            {"lower": 100.0, "upper": None, "reference": "b"},
        ]
        support = quantile_support(bands, {"method": "pareto", "alpha": 2.0})
        positive, negative = support.positive, support.negative
        # A bare list feeds both signs, so the open top is open below too.
        assert quantile_bounds(positive, negative) == (None, None)
        assert inside_quantile_support(1e12, positive, negative)
        assert inside_quantile_support(-1e12, positive, negative)

    def test_an_open_negative_top_records_a_null_lower_bound(self) -> None:
        bands = {
            "positive": [{"lower": 0.0, "upper": 10.0, "reference": "p"}],
            "negative": [
                {"lower": 0.0, "upper": 5.0, "reference": "n1"},
                {"lower": 5.0, "upper": None, "reference": "n2"},
            ],
        }
        support = quantile_support(bands, {"method": "pareto", "alpha": 1.5})
        positive, negative = support.positive, support.negative
        assert quantile_bounds(positive, negative) == (None, 10.0)
        # Belgium's probe reached -22.36 past every finite bound.
        assert inside_quantile_support(-22.36, positive, negative)

    def test_another_mode_is_refused_by_name(self) -> None:
        with pytest.raises(
            ValueError, match="'donor_tail' lands with its own producer"
        ):
            quantile_support(_BANDS, {"method": "donor_tail"})

    def test_a_bare_band_list_reads_on_both_signs(self) -> None:
        # E1, Belgium's case: uniform bands [0, 10] and [10, 50] over donor
        # values -5 to 3. quantile_map applies a bare list to both signs.
        rows = [
            {"lower": 0.0, "upper": 10.0, "reference": "a"},
            {"lower": 10.0, "upper": 50.0, "reference": "b"},
        ]
        values = np.arange(-5.0, 4.0)
        mapped = quantile_map(
            values,
            np.ones(len(values)),
            [
                {"lower": row["lower"], "upper": row["upper"], "share": 0.5}
                for row in rows
            ],
            interpolation="uniform",
        )
        assert (mapped < 0).sum() == 5
        uniform = {"method": "uniform"}
        for declared in (rows, {"bands": rows}):
            support = quantile_support(declared, uniform)
            assert quantile_bounds(support.positive, support.negative) == (-50.0, 50.0)
            assert all(
                inside_quantile_support(float(v), support.positive, support.negative)
                for v in mapped
            )
        signed = quantile_support({"positive": rows, "negative": rows}, uniform)
        assert quantile_bounds(signed.positive, signed.negative) == (-50.0, 50.0)

    def test_every_declared_shape_is_read(self) -> None:
        # C3: {bands}, {positive, negative} and the component list.
        uniform = {"method": "uniform"}
        support = quantile_support({"bands": _BANDS}, uniform)
        assert inside_quantile_support(150.0, support.positive, support.negative)
        components = quantile_support(
            {
                "components": [
                    {"value": "a", "bands": _BANDS},
                    {"value": 2, "bands": {"bands": _BANDS}},
                ]
            },
            uniform,
        )
        assert set(components.components) == {"a", 2}
        assert components.intervals("missing") is None
        with pytest.raises(ValueError, match="band list, bands, positive"):
            quantile_support({"positive": _BANDS}, uniform)

    def test_a_value_is_checked_against_its_own_component_only(self) -> None:
        # E2: component a on [0, 10], b on [100, 200]; 150 in a is outside.
        bands = {
            "components": [
                {
                    "value": "a",
                    "bands": [{"lower": 0.0, "upper": 10.0, "reference": "x"}],
                },
                {
                    "value": "b",
                    "bands": [{"lower": 100.0, "upper": 200.0, "reference": "y"}],
                },
            ]
        }
        support = quantile_support(bands, {"method": "uniform"})
        assert not inside_quantile_support(150.0, *support.intervals("a"))
        assert inside_quantile_support(150.0, *support.intervals("b"))
        assert quantile_bounds(support.positive, support.negative) == (-200.0, 200.0)

    @PROPERTY
    @given(
        values=st.lists(
            st.one_of(st.just(0.0), st.floats(-1e6, -0.01), st.floats(0.01, 1e6)),
            min_size=4,
            max_size=40,
        ),
        cuts=st.lists(st.floats(1.0, 1e5), min_size=2, max_size=4, unique=True),
        negative_cuts=st.lists(
            st.floats(1.0, 1e5), min_size=2, max_size=4, unique=True
        ),
        pareto=st.booleans(),
        shape=st.sampled_from(("bare", "bands", "signed", "components")),
        codes=st.lists(st.sampled_from(("a", "b")), min_size=40, max_size=40),
    )
    def test_property_every_quantile_mapped_value_lies_inside_its_bands(
        self, values, cuts, negative_cuts, pareto, shape, codes
    ) -> None:
        # Differential: the real quantile_map writes the values; every one
        # must lie inside quantile_support's intervals for its own component.
        values = np.asarray(values)
        if not (values != 0).any():
            values[0] = 1.0
        interpolation = (
            {"method": "pareto", "alpha": 2.0} if pareto else {"method": "uniform"}
        )

        def band_rows(edges, tag):
            edges = sorted(edges)
            rows = [
                {"lower": low, "upper": high, "reference": f"{tag}{index}"}
                for index, (low, high) in enumerate(
                    zip(edges[:-1], edges[1:], strict=True)
                )
            ]
            if pareto:
                rows[-1]["upper"] = None
            return rows

        def shares(rows):
            return [
                {"lower": r["lower"], "upper": r["upper"], "share": 1 / len(rows)}
                for r in rows
            ]

        def mapped_for(selected, document):
            if isinstance(document, dict) and set(document) == {"bands"}:
                document = document["bands"]
            target = (
                {sign: shares(rows) for sign, rows in document.items()}
                if isinstance(document, dict)
                else shares(document)
            )
            method = interpolation if pareto else "uniform"
            return quantile_map(
                selected, np.ones(len(selected)), target, interpolation=method
            )

        positive = band_rows(cuts, "p")
        negative = band_rows(negative_cuts, "n")
        component = np.asarray(codes[: len(values)], dtype=object)
        if shape == "bare":
            declared = positive
        elif shape == "bands":
            declared = {"bands": positive}
        elif shape == "signed":
            declared = {"positive": positive, "negative": negative}
        else:
            declared = {
                "components": [
                    {"value": "a", "bands": positive},
                    {"value": "b", "bands": negative},
                ]
            }
        support = quantile_support(declared, interpolation)
        if shape == "components":
            by_code = {"a": positive, "b": negative}
            for code in ("a", "b"):
                selected = values[component == code]
                if not len(selected) or not (selected != 0).any():
                    continue
                mapped = mapped_for(selected, by_code[code])
                intervals = support.intervals(code)
                assert all(
                    inside_quantile_support(float(v), *intervals) for v in mapped
                )
                assert all(mapped[selected == 0] == 0)
        else:
            mapped = mapped_for(values, declared)
            assert all(
                inside_quantile_support(float(v), support.positive, support.negative)
                for v in mapped
            )
            assert all(mapped[values == 0] == 0)


class TestComponentSupport:
    _bands = {
        "components": [
            {"value": "a", "bands": [{"lower": 0.0, "upper": 10.0, "reference": "x"}]},
            {
                "value": "b",
                "bands": [{"lower": 100.0, "upper": 200.0, "reference": "y"}],
            },
        ]
    }

    def _roster(self, component_column="wage_code"):
        params = {"bands": self._bands, "interpolation": {"method": "uniform"}}
        if component_column is not None:
            params["component_column"] = component_column
        return {
            "columns": [
                {
                    "entity": "person",
                    "column": "wage",
                    "class": "quantile_mapped",
                    "producer_params": params,
                }
            ]
        }

    def _run(self, wage, codes, *, roster=None, sliced=True):
        donor, _ = _donor_and_terminal()
        terminal = _support_frame(
            {
                ("person", "wage"): list(wage),
                ("person", "wage_code"): pd.array(codes, dtype="string"),
            }
        )
        _, bounds, _, _ = _run_support(
            roster or self._roster(),
            donor,
            terminal,
            extra_columns=[("person", "wage_code")] if sliced else (),
        )
        (row,) = decode_support_bounds(bounds).document["columns"]
        return row

    def test_a_value_inside_another_components_bands_is_outside(self) -> None:
        # E2: 150 in component a passes only on b's bands.
        row = self._run(
            (5.0, 150.0, 0.0, 120.0, 0.0, 9.0), ["a", "a", "a", "b", "b", "b"]
        )
        assert row["records_outside"] == 2  # 150 in a, 9 in b
        assert (row["expected_min"], row["expected_max"]) == (-200.0, 200.0)
        inside = self._run(
            (5.0, 7.0, 0.0, 120.0, 0.0, 150.0), ["a", "a", "a", "b", "b", "b"]
        )
        assert inside["records_outside"] == 0

    def test_a_code_without_bands_is_outside(self) -> None:
        row = self._run(
            (5.0, 7.0, 0.0, 120.0, 0.0, 150.0), ["a", "a", "a", "b", "b", "c"]
        )
        assert row["records_outside"] == 1

    def test_a_missing_component_column_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"component column person\.wage_code"):
            self._run((5.0,) * 6, ["a"] * 6, sliced=False)
        with pytest.raises(ValueError, match=r"component column person\.None"):
            self._run((5.0,) * 6, ["a"] * 6, roster=self._roster(None))

    def test_an_aggregate_total_is_checked_against_its_own_component(self) -> None:
        # E2: household totals (members 0-2, 3-4, 5) are checked against the
        # household's component. A total of 150 in a lies inside b's bands,
        # so each of the household's three member rows counts as outside.
        roster = self._roster()
        params = roster["columns"][0]["producer_params"]
        params["aggregate_entity"] = "household"
        codes = ["a", "a", "a", "b", "b", "b"]
        row = self._run((50.0, 50.0, 50.0, 60.0, 60.0, 150.0), codes, roster=roster)
        assert row["records_outside"] == 3
        inside = self._run((3.0, 3.0, 3.0, 60.0, 60.0, 150.0), codes, roster=roster)
        assert inside["records_outside"] == 0


class TestNormalizedSupport:
    def test_differential_bounds_equal_money_normalize_outputs(self, tmp_path) -> None:
        from test_support.microcosm_build.money_normalization import (
            graph_declaration,
            money_factor_context,
            normalize_node,
            write_money_feed,
        )

        path, pin, identity, facts = write_money_feed(tmp_path, rate="1.25")
        context = money_factor_context(identity, facts)
        declaration = graph_declaration()
        raw = {
            "interest_income": [3.0, 5.0, 0.1],
            "rental_income": [-120.5, 0.0, 35.25],
            "farm_self_employment_income": [0.0, -2.675, 1000000.0],
        }
        node = normalize_node("toy.normalize", "toy.central", declaration, context, pin)
        money_context = KernelContext(
            node=node,
            tables={"person": pd.DataFrame({"person_id": [11, 12, 13], **raw})},
            weights={},
            strata=pd.Series([], dtype=object),
            params=node.params,
            rng=np.random.default_rng(0),
            sources={"money_facts": path},
            frame_metadata={"donor_time_period": 2024},
        )
        normalized = MONEY_NORMALIZE.run(money_context).columns
        roster = {
            "columns": [
                {"entity": "person", "column": column, "class": "normalized"}
                for column in raw
            ]
        }
        donor = KernelContext(
            node=_support_node(
                DONOR_RANGES_REF,
                DONOR_RANGES,
                DONOR_RANGES_TYPE,
                roster,
                [("person", c) for c in raw],
            ),
            tables={"person": pd.DataFrame({"person_id": [11, 12, 13], **raw})},
            weights={},
            strata=pd.Series([], dtype=object),
            params={"support_roster": _text(roster)},
            rng=np.random.default_rng(0),
        )
        ranges = TRANSPORT_DONOR_RANGES.run(donor).artifacts[DONOR_RANGES]
        money_params = {k: v for k, v in node.params.items()}
        bounds_node = _support_node(
            SUPPORT_BOUNDS_REF,
            DONOR_SUPPORT_BOUNDS,
            SUPPORT_BOUNDS_TYPE,
            roster,
            [("person", c) for c in raw],
            params=money_params,
            sources=("money_facts",),
            inputs=(
                ArtifactInput(
                    DONOR_RANGES,
                    "toy.evidence.donor_ranges",
                    DONOR_RANGES,
                    DONOR_RANGES_TYPE,
                ),
            ),
        )
        terminal = pd.DataFrame(
            {
                "person_id": [11, 12, 13],
                **{c: normalized["person", c].to_numpy() for c in raw},
            }
        )
        bounds_context = KernelContext(
            node=bounds_node,
            tables={"person": terminal},
            weights={},
            strata=pd.Series([], dtype=object),
            params=bounds_node.params,
            rng=np.random.default_rng(0),
            sources={"money_facts": path},
            artifacts={
                DONOR_RANGES: ArtifactValue(
                    ranges, DONOR_RANGES_TYPE, _sha(b"r"), _sha(ranges), NumericScope()
                )
            },
        )
        rows = decode_support_bounds(
            TRANSPORT_SUPPORT_BOUNDS.run(bounds_context).artifacts[DONOR_SUPPORT_BOUNDS]
        ).document["columns"]
        for row in rows:
            values = normalized["person", row["column"]].to_numpy()
            assert row["expected_min"] == float(values.min()), row["column"]
            assert row["expected_max"] == float(values.max()), row["column"]
            assert row["records_outside"] == 0
            # Signs and zeros carry through the normalization.
            raw_values = np.asarray(raw[row["column"]])
            assert np.array_equal(np.sign(values), np.sign(raw_values))


def test_a_component_column_joins_its_entitys_support_slice(monkeypatch) -> None:
    # E2 (w4d): the support producer reads a quantile_mapped row's
    # component_column, so the generators' slices carry it.
    roster = {
        "columns": [
            {
                "entity": "person",
                "column": "wage",
                "class": "quantile_mapped",
                "producer_params": {"component_column": "wage_code"},
            },
            {"entity": "person", "column": "age", "class": "copied"},
            {"entity": "household", "column": "rent", "class": "derived"},
        ]
    }
    monkeypatch.setattr(compose, "_support_roster", lambda spec: roster)
    bounded = {"copied", "normalized", "currency", "quantile_mapped"}
    assert compose.support_roster_slices({}, bounded) == [
        {"entity": "person", "columns": ["age", "wage", "wage_code"]}
    ]
    ranged = {"copied", "normalized", "currency"}
    assert compose.support_roster_slices({}, ranged) == [
        {"entity": "person", "columns": ["age"]}
    ]
