"""Every transport kernel's hash binds what its code computes through.

``source_hash`` does not follow imports, so a node's hash must name every
module whose code can change what the node computes (microcosm#931). The
derived binding rule (``test_support.microcosm_build.derived_binding_rule``),
which test_uk_graph holds the UK stages and kernels to, holds here every
kernel ``build_transport_registry`` registers: the shared transport package
Belgium and New Zealand compose their graphs with, the shared atomic
geography, the ordered solve and the rules kernels. The rule follows what a
kernel's own code names into the module that defines it, to a fixed point,
and each module it reaches must be in what the kernel's key hashes, recorded
at its ``source_hash`` calls.

The transport's own tables say where the rule stops, each entry with a reason
a test below checks: the core modules every node runs through
(``CORE_MODULES``), the core definitions the kernels name
(``KERNEL_CORE_NAMES``), the checks that only refuse
(``REFUSAL_ONLY_HELPERS``), the kernel stops (``KERNEL_STOPS``) and the names
the rule may stop at outside a key (``KERNEL_UNBOUND_NAMES``, none). Beside
what a kernel's code names, the rule walks what the registry hands it
(``HANDED_CLASSES``, ``HANDED_VALUES``) and holds what a key binds beyond its
``source_hash`` calls (``KEYS_BEYOND_SOURCE_HASH``). The transport lists no
declaration modules: its kernels read spec data through canonical document
parameters, which their node keys hash, not through a stage contract, and
compute through country_spec's code (the gates manifest, the reference
validators), which the rule follows.

The rule replaced microcosm#1497's per-kernel import record for the transport
compile kernels; test_transport_compile_modules keeps the compile list's own
closure check. What the rule cannot see: a call through an object whose class
the code never names and the tables do not list, a name looked up by a
computed string, and data files read by path. The source codecs are keyed by
their versioned names, not their code, as in the UK.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import microcosm.build.transport as transport_package
from microcosm.build import graph_atomic_geography
from microcosm.build.transport import (
    binding_identity,
    evidence_kernels,
    gate_kernels,
    population_kernels,
    registry,
    target_kernels,
)
from microcosm.build.transport.gate_bindings import TRANSPORT_GATE_REGISTRY
from microcosm.build.transport.gate_kernels import GateBatteryKernel
from microcosm.build.transport.registry import build_transport_registry
from microcosm.calibrate.ordered_kernels import CalibrateOrderedAdamKernel
from microcosm.frame import rules_kernels
from microcosm.frame.rules_kernels import (
    SimulateCounterfactualKernel,
    SimulateRulesByRefKernel,
    SimulateSolveZeroKernel,
)
from microcosm.graph import SeedSource
from test_support.microcosm_build.derived_binding_rule import (
    FIRST_PARTY_SCOPE,
    GRAPH_RECORD_REASONS,
    BindingRule,
    RuleTables,
    findings,
    hashed_modules,
    kernel_classes_defined,
    module_definitions,
)
from test_support.microcosm_build.transport_composed import (
    ToyRulesEngine,
    toy_rules_bindings,
    write_rulespec_tree,
)
from test_support.microcosm_build.transport_kernels import transport_kernels
from test_support.microcosm_build.transport_population import UNIT_RULE

#: What catches an edit to the Frame, whose three modules share it.
_FRAME_REASON = (
    "the kernels compute on frames, and the executor holds every population "
    "version in one. No transport key hashes it; the transport graph tests, "
    "which run composed graphs through run_graph (test_transport_composed_graph "
    "and test_nz_transport_toy_graph among them), catch an edit that changes a "
    "value on their fixtures, and the Frame's own tests the rest"
)
#: The core modules every node runs through, which the rule neither follows
#: nor asks a kernel to bind, each with what runs through it and what catches
#: an edit to it that would change values. Binding one would re-key every
#: node on any edit to it. They are the modules the executor runs every node
#: through (test_the_core_modules_are_on_every_nodes_path).
CORE_MODULES: dict[str, str] = {
    "microcosm.frame.bundle": f"the Frame: {_FRAME_REASON}",
    "microcosm.frame.schema": f"the Frame's entity schema: {_FRAME_REASON}",
    "microcosm.frame.weights": (
        f"the Frame's typed weights and mass records: {_FRAME_REASON}"
    ),
    "microcosm.graph.executor": (
        "runs every node (run_graph): keys it, projects its declared inputs, "
        "calls its kernel and applies its result. No transport key hashes it; "
        "the transport graph tests run every node of their composed graphs "
        "through it"
    ),
    "microcosm.graph.population": (
        "the executor applies every node's result to its population version "
        "through it (patch); as for the executor, the transport graph tests "
        "apply every result of their composed graphs through it"
    ),
    "microcosm.graph.kernel": (
        "every node's inputs and outputs pass through its KernelContext and "
        "KernelResult, and every transport kernel's implementation hash is "
        "computed through its source_hash, so an edit that changes what that "
        "returns changes every key"
    ),
    "microcosm.graph.keys": (
        "computes every node's key and its artifacts' keys (node_key), so an "
        "edit that changes what it returns changes every key; no transport "
        "kernel reads the generator the executor seeds from a key (none "
        "declares SeedSource.EXECUTOR; checked)"
    ),
    "microcosm.graph.decl": (
        "node_key hashes every node's normative declaration and resolves its "
        "inputs and population version through the compiled graph, as the "
        "executor does, so an edit that changes how a node is declared or "
        "what it reads changes its key"
    ),
}
#: The core modules' definitions the transport kernels' own code names, each
#: with why naming it binds nothing the kernel's key misses
#: (test_the_core_definitions_the_kernels_name_are_listed). The Frame's
#: definitions are the Frame's reason's. The graph's records, enums and
#: constants take the graph's own reasons, which every country shares; the
#: other four rest on the transport's checks.
KERNEL_CORE_NAMES: dict[tuple[str, str], str] = {
    **{
        item: GRAPH_RECORD_REASONS[item]
        for item in (
            ("microcosm.graph.decl", "ArtifactOutput"),
            ("microcosm.graph.decl", "ArtifactType"),
            ("microcosm.graph.decl", "GATE_OUTCOMES"),
            ("microcosm.graph.decl", "Node"),
            ("microcosm.graph.decl", "Ownership"),
            ("microcosm.graph.decl", "ROWS_ALL"),
            ("microcosm.graph.decl", "Slice"),
            ("microcosm.graph.decl", "StructuralDelta"),
            ("microcosm.graph.kernel", "ArtifactValue"),
            ("microcosm.graph.kernel", "Capabilities"),
            ("microcosm.graph.kernel", "Determinism"),
            ("microcosm.graph.kernel", "KernelBase"),
            ("microcosm.graph.kernel", "KernelRole"),
            ("microcosm.graph.kernel", "Numeric"),
            ("microcosm.graph.kernel", "SeedSource"),
        )
    },
    ("microcosm.graph.population", "dtype_for_token"): (
        "maps a node's declared dtype token to the dtype its owned cells are "
        "stored at. The executor stores every node's owned cells at the dtype "
        "it returns (patch, checked), so a kernel that casts its cells through "
        "it first hands the executor what the executor stores anyway, and an "
        "edit to it moves every node's stored dtypes alike: the transport "
        "graph tests catch one on the dtypes their composed graphs carry, and "
        "graph.population's own tests the rest"
    ),
    ("microcosm.graph.kernel", "KernelContext"): (
        "the frozen record of what the executor hands a kernel to read, which "
        "a kernel names to annotate it; it only checks and freezes the fields "
        "the executor fills, and every node's inputs pass through it "
        "(graph.kernel's core reason)"
    ),
    ("microcosm.graph.kernel", "KernelResult"): (
        "the frozen record a kernel returns its outputs in, which holds what "
        "the kernel puts in it and computes nothing; the executor validates "
        "every field and applies it through graph.population's patch "
        "(graph.population's core reason)"
    ),
    ("microcosm.graph.kernel", "source_hash"): (
        "hashes the source files of the modules it is given. The evidence "
        "producers write their own implementation hash, computed through it "
        "(evidence_contract.evidence_implementation_hash), into each evidence "
        "header, and the battery recomputes it only to refuse a header that "
        "claims another; the rules kernels and the second quantile map name "
        "it through the kernel classes they follow whole. An edit that "
        "changes what it returns changes those hashes and every key alike "
        "(graph.kernel's core reason)"
    ),
}
#: Checks that read values only to decide whether to refuse, each with why it
#: cannot change what its caller computes. The rule does not follow them;
#: test_the_refusal_only_helpers_return_and_write_nothing checks each.
REFUSAL_ONLY_HELPERS: dict[tuple[str, str], str] = {
    ("microcosm.build.transport.graph_inputs", "require_outputs"): (
        "refuses a node whose declared typed outputs differ from its kernel's "
        "(artifact_types.KERNEL_OUTPUTS), and returns nothing. The table it "
        "reads types the outputs of every kernel, the problem's and the "
        "diagnostics' among them, whose modules (calibrate.artifacts, "
        "diagnostics.schema) only the kernels that write those outputs "
        "compute through"
    ),
    ("microcosm.build.reference_profiles", "_validate_interval_resolution"): (
        "binds recorded interval scaling to the selected factors, and to the "
        "interval document where it is given one, and refuses on any "
        "mismatch, returning nothing. Target compilation, which the compile "
        "kernels run, gives it no interval document (checked); "
        "calibration.prepare@1 and targets.problem@1, which apply interval "
        "tables, hash reference_intervals"
    ),
}
#: Where the kernel rule stops, each with why a kernel that names the
#: definition need not hash what lies beyond it;
#: test_the_kernel_rule_stops_only_where_its_reasons_hold checks each.
KERNEL_STOPS: dict[tuple[str, str], str] = {
    ("microcosm.build.transport.target_kernels", "_COMPILE_MODULES"): (
        "the imported compile list (compile_modules.transport_compile_modules), "
        "which only implementation hashes read: every reference to it in the "
        "transport package sits inside an implementation_hash (checked). The "
        "walk meets it only because the second quantile map's run names the "
        "first quantile map, whose class it follows whole, that hash included; "
        "the second quantile map hashes the list itself"
    ),
}
#: Names a kernel reads from a module its key does not hash, each with why.
#: None: every name the rule stops at lies inside its kernel's key.
KERNEL_UNBOUND_NAMES: dict[tuple[str, str], str] = {}

#: The class of the engines build_transport_registry builds, one per rules
#: binding, unless the caller injects pure-Python engines.
ENGINE_CLASS = ("microcosm.frame.adapters.axiom", "AxiomEngine")
_ENGINE_REASON = (
    "build_transport_registry builds each binding's engine as an AxiomEngine "
    "and hands the engine mapping to the three rules kernels and to the "
    "evidence producers' input closure (checked); their code calls the "
    "engines' methods, and names the class only to check an engine's kind"
)
#: Objects the registry hands kernels, whose classes their code need not
#: name, by kernel ref: the class, which the walk follows from the methods the
#: kernel's walk names on it (BindingRule.keyed_walk), and why.
HANDED_CLASSES: dict[str, tuple[tuple[str, str], str]] = {
    ref: (ENGINE_CLASS, _ENGINE_REASON)
    for ref in (
        "simulate.rules_by_ref@1",
        "simulate.counterfactual@1",
        "simulate.solve_zero@1",
        "transport.input_closure@1",
    )
}
#: Definitions the registry hands kernels by value, by kernel ref, which the
#: walk follows from their definitions, and why.
HANDED_VALUES: dict[str, tuple[tuple[str, str], str]] = {
    "gates.battery@1": (
        ("microcosm.build.transport.gate_bindings", "TRANSPORT_GATE_REGISTRY"),
        "build_transport_registry builds the battery with the transport gate "
        "registry (register_gate_kernel; checked), which the battery's code "
        "never names: it evaluates each declared gate through the registry's "
        "binding",
    ),
}


def _binding_sources(kernel) -> frozenset[str]:
    """The modules whose source the battery's bindings can run, which its
    implementation hash digests (binding_identity.describe)."""

    return frozenset(
        module
        for binding in kernel.registry.values()
        for module in binding_identity.describe(binding)[1]
    )


#: Modules a kernel's key binds beyond its ``source_hash`` calls, by kernel
#: ref: how to read them from the kernel, and why.
KEYS_BEYOND_SOURCE_HASH: dict[str, tuple[Callable[[object], frozenset[str]], str]] = {
    "gates.battery@1": (
        _binding_sources,
        "the battery's implementation hash digests the source closure of each "
        "binding (binding_identity.describe) beside its code's hash, so an "
        "edit to a module a binding can run re-keys it (checked)",
    ),
}

_RULE = BindingRule(
    RuleTables(
        core_modules=CORE_MODULES,
        refusal_only=REFUSAL_ONLY_HELPERS,
        stops=KERNEL_STOPS,
    )
)


def _keyed_walk(ref: str, kernel):
    """One kernel's walk, with what the registry hands it, against what its
    key hashes (``BindingRule.keyed_walk``)."""

    handed_class = HANDED_CLASSES.get(ref)
    handed_value = HANDED_VALUES.get(ref)
    beyond = KEYS_BEYOND_SOURCE_HASH.get(ref)
    return _RULE.keyed_walk(
        kernel,
        handed_classes=() if handed_class is None else (handed_class[0],),
        handed_values=() if handed_value is None else (handed_value[0],),
        key=() if beyond is None else beyond[0](kernel),
    )


@functools.cache
def _transport_rule_walks() -> MappingProxyType:
    """What the rule holds each transport kernel to, by ref, one label each."""

    return MappingProxyType(
        {
            ref: {ref: _keyed_walk(ref, kernel)}
            for ref, kernel in sorted(transport_kernels().items())
        }
    )


def _package_sources(package) -> list[tuple[str, Path]]:
    """Each module of ``package`` with its file, the package's own included."""

    return [
        (
            package.__name__
            if path.stem == "__init__"
            else f"{package.__name__}.{path.stem}",
            path,
        )
        for path in sorted(Path(package.__file__).parent.glob("*.py"))
    ]


def test_the_rule_sees_every_kernel_the_transport_registers(tmp_path) -> None:
    """The kernels the rule walks are every kernel class the transport package
    defines, all registered, and the eight the registry registers from
    elsewhere: the shared atomic geography's four, the ordered solve and the
    three rules kernels. The replicated registration registers each ref with
    the class build_transport_registry registers it with, so a new kernel
    cannot escape the rule, and each table names a registered kernel."""
    kernels = transport_kernels()
    classes = {type(kernel) for kernel in kernels.values()}
    defined = kernel_classes_defined(_package_sources(transport_package))
    assert defined <= classes, sorted(cls.__name__ for cls in defined - classes)
    assert classes - defined == kernel_classes_defined(
        [(graph_atomic_geography.__name__, Path(graph_atomic_geography.__file__))]
    ) | {
        CalibrateOrderedAdamKernel,
        SimulateRulesByRefKernel,
        SimulateCounterfactualKernel,
        SimulateSolveZeroKernel,
    }
    # The registry itself, built engine-free with the toy package's engines.
    rows = write_rulespec_tree(tmp_path)
    prepared = build_transport_registry(
        toy_rules_bindings(rows),
        tmp_path,
        unit_rule=UNIT_RULE,
        engines_by_binding={row["id"]: ToyRulesEngine(tmp_path, row) for row in rows},
    )
    assert {
        ref: type(prepared.kernels.get(ref)) for ref in prepared.kernels.refs()
    } == {ref: type(kernel) for ref, kernel in kernels.items()}
    for table in (HANDED_CLASSES, HANDED_VALUES, KEYS_BEYOND_SOURCE_HASH):
        assert set(table) <= set(kernels), sorted(set(table) - set(kernels))


def test_every_transport_kernel_hashes_each_helper_it_computes_through() -> None:
    """Each module holding code a transport kernel's own methods, or what the
    registry hands it, can run to compute a value is in what its key really
    hashes, every name the rule stops at lies inside that key or is listed,
    and the listed stops are met (microcosm#931).

    The rule starts at what the kernel runs (its methods by method resolution
    order, without the hash itself, the methods only the hash calls and the
    listed refusal-only methods), stops at the core-module, refusal-only and
    stop tables, and follows the engines and the gate registry the registry
    hands the kernels that take them (``BindingRule.keyed_walk``)."""
    found = findings(_transport_rule_walks())
    assert not found.unhashed, found.unhashed
    assert found.unlisted == set(KERNEL_UNBOUND_NAMES), sorted(
        found.unlisted ^ set(KERNEL_UNBOUND_NAMES)
    )
    assert found.met == set(KERNEL_STOPS), sorted(found.met ^ set(KERNEL_STOPS))


def test_the_core_modules_are_on_every_nodes_path() -> None:
    """Each listed core module is on every node's path, as its reason says:
    the walk from run_graph, which keys, runs and patches every node, meets
    each. The key module's reason holds too: no transport kernel reads the
    generator the executor seeds from a node key."""
    off_path = _RULE.core_modules_off_run_graph_path()
    assert not off_path, off_path
    seeded = {
        ref
        for ref, kernel in transport_kernels().items()
        if kernel.capabilities.seed_source is SeedSource.EXECUTOR
    }
    assert not seeded, sorted(seeded)


def test_the_core_definitions_the_kernels_name_are_listed() -> None:
    """The definitions of the core modules other than the Frame's that the
    transport kernels' code names are exactly ``KERNEL_CORE_NAMES``, and the
    executor stores every node's owned cells through the dtype tokens' map,
    as that entry's reason says."""
    named = findings(_transport_rule_walks()).core_names
    assert named == set(KERNEL_CORE_NAMES), sorted(named ^ set(KERNEL_CORE_NAMES))
    patching = _RULE.helper_walk(
        ("microcosm.graph.population", "patch"), follow_core=True
    )
    assert "dtype_for_token" in patching.reach["microcosm.graph.population"]


def test_the_refusal_only_helpers_return_and_write_nothing() -> None:
    """Each listed check returns nothing and writes nothing
    (``writes_nothing``), and target compilation gives the interval check no
    interval document, as its reason says."""
    violations = _RULE.refusal_only_violations()
    assert not violations, violations
    compile_surface = module_definitions(
        target_kernels.__name__, FIRST_PARTY_SCOPE
    ).definitions["compile_target_surface"]
    calls = [
        sub
        for sub in ast.walk(compile_surface)
        if isinstance(sub, ast.Call)
        and getattr(sub.func, "id", None) == "validate_reference_resolution"
    ]
    assert calls
    for call in calls:
        assert "interval_document" not in {item.arg for item in call.keywords}


def test_the_kernel_rule_stops_only_where_its_reasons_hold() -> None:
    """The compile list is met only through the first quantile map's class,
    which the second quantile map's run names, and every reference to it in
    the transport package sits inside an implementation hash."""
    namers: dict[tuple[str, str], set] = {}
    for walks in _transport_rule_walks().values():
        for walk, _ in walks.values():
            for stop, by in walk.met.items():
                namers.setdefault(stop, set()).update(by)
    compile_list = (target_kernels.__name__, "_COMPILE_MODULES")
    assert namers == {
        compile_list: {(population_kernels.__name__, "TransportQuantileMapKernel")}
    }
    for module, path in _package_sources(transport_package):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hashing = {
            id(sub)
            for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef)
            and function.name == "implementation_hash"
            for sub in ast.walk(function)
        }
        for sub in ast.walk(tree):
            reads = (isinstance(sub, ast.Name) and sub.id == compile_list[1]) or (
                isinstance(sub, ast.Attribute) and sub.attr == compile_list[1]
            )
            if reads and not isinstance(sub.ctx, ast.Store):
                assert id(sub) in hashing, (module, sub.lineno)


def test_the_registry_hands_the_kernels_what_the_tables_list() -> None:
    """build_transport_registry builds AxiomEngine engines and hands their
    mapping to exactly the rules kernels and the evidence producers, which
    hand it to the input closure alone (it also returns the mapping, in the
    TransportRegistry its caller runs the graph with), and builds the battery
    with the transport gate registry: the objects ``HANDED_CLASSES`` and
    ``HANDED_VALUES`` list for those kernels."""
    builder = module_definitions(registry.__name__, FIRST_PARTY_SCOPE).definitions[
        "build_transport_registry"
    ]

    def called(node, module: str) -> list[tuple[tuple[str, str] | None, ast.Call]]:
        return [
            (_RULE.resolve(module, sub.func.id), sub)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
        ]

    def handed(node, module: str, argument: str) -> set:
        return {
            callee
            for callee, call in called(node, module)
            if any(
                isinstance(value, ast.Name) and value.id == argument
                for value in call.args
            )
        }

    calls = called(builder, registry.__name__)
    assert ENGINE_CLASS in {callee for callee, _ in calls}
    assert handed(builder, registry.__name__, "engine_mapping") == {
        (rules_kernels.__name__, "SimulateRulesByRefKernel"),
        (rules_kernels.__name__, "SimulateCounterfactualKernel"),
        (rules_kernels.__name__, "SimulateSolveZeroKernel"),
        (evidence_kernels.__name__, "register_evidence_kernels"),
        (registry.__name__, "TransportRegistry"),
    }
    producers = module_definitions(
        evidence_kernels.__name__, FIRST_PARTY_SCOPE
    ).definitions["register_evidence_kernels"]
    assert handed(producers, evidence_kernels.__name__, "engines") == {
        (evidence_kernels.__name__, "TransportInputClosureKernel")
    }
    kernels = transport_kernels()
    assert {type(kernels[ref]) for ref in HANDED_CLASSES} == {
        SimulateRulesByRefKernel,
        SimulateCounterfactualKernel,
        SimulateSolveZeroKernel,
        evidence_kernels.TransportInputClosureKernel,
    }
    (battery,) = [
        call
        for callee, call in calls
        if callee == (gate_kernels.__name__, "register_gate_kernel")
    ]
    value = battery.args[1]
    assert isinstance(value, ast.Name)
    assert (
        _RULE.resolve(registry.__name__, value.id)
        == HANDED_VALUES["gates.battery@1"][0]
    )


def test_the_battery_key_binds_its_bindings_sources(monkeypatch) -> None:
    """The modules the battery's key binds beyond its ``source_hash`` calls
    are its bindings' source closures: a changed digest of one of them, which
    its code's hash does not cover, re-keys the battery."""
    kernel = transport_kernels()["gates.battery@1"]
    sources, _ = KEYS_BEYOND_SOURCE_HASH["gates.battery@1"]
    beyond = sorted(sources(kernel) - hashed_modules(kernel))
    assert beyond
    before = GateBatteryKernel(TRANSPORT_GATE_REGISTRY).implementation_hash()
    describe = binding_identity.describe

    def edited(binding, *, path: str = "binding"):
        description, digests = describe(binding, path=path)
        return description, {
            module: "0" * 64 if module == beyond[0] else digest
            for module, digest in digests.items()
        }

    monkeypatch.setattr(binding_identity, "describe", edited)
    assert GateBatteryKernel(TRANSPORT_GATE_REGISTRY).implementation_hash() != before
