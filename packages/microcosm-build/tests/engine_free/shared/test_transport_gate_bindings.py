"""The country-neutral transport gate bindings (``transport/gate_bindings.py``).

Invariants, each checked here:

- every NZ gate resolves to a binding or a declared ``not_applicable``; none
  resolves to the battery's unregistered-gate gap, and the NZ manifest passes
  both parameter validators;
- thresholds come only from ``gates.json``: the module holds no float literal
  and no integer other than 0 and 1, no bound evaluator has a non-``None``
  default, a manifest that omits a threshold is refused, and generated
  thresholds reach the verdict unchanged;
- for each binding, evidence on the pass side of a generated threshold
  passes, evidence on the fail side fails, and missing evidence resolves to
  ``evidence_absent``, never ``passed``;
- the module imports no country runtime;
- the declared surfaces in NZ ``gates.json`` equal the package resources
  they restate (differential checks against ``source_stages.json`` and
  ``export_contract.json``);
- through ``gates.battery@1`` in the toy transport graph, the kernel's
  verdict equals the battery evaluated directly on the same evidence
  (a differential check of the graph path).

The weight gates' differential check against the UK runtime's weight gates
lives in ``engine_free/uk/test_transport_gate_bindings.py``.
"""

from __future__ import annotations

import ast
import base64
import inspect
import json
import math
from dataclasses import replace
from fractions import Fraction
from functools import cache

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import GatesManifest, load_country_spec
from microcosm.build.gate_battery import (
    DEFAULT_REGISTRY,
    BlockingMode,
    EvidenceContext,
    FunctionBinding,
    GateBatteryRun,
    GateOutcome,
    GatePhaseReport,
    GateStatus,
    evaluate_phase,
    gate_phase_report_payload,
    gate_signing_key_env,
    validate_gate_parameters,
)
from microcosm.build.gates import GateResult
from microcosm.build.transport import gate_bindings
from microcosm.build.transport.gate_bindings import (
    DIAGNOSTICS,
    PROBLEM,
    SURFACE,
    TRANSPORT_GATE_REGISTRY,
    required_gate_parameters,
    validate_required_gate_parameters,
    weight_ess_gate,
    weight_ratio_gate,
    weight_summary,
)
from microcosm.build.transport.gate_kernels import (
    EVIDENCE_DECODERS,
    GateBatteryKernel,
    decode_gate_report,
    phase_enforcement,
)
from microcosm.build.transport.target_kernels import (
    decode_target_surface,
    encode_target_surface,
    parse_reference_document,
)
from microcosm.calibrate import TargetRegistry
from microcosm.calibrate.artifacts import decode_problem, encode_problem
from microcosm.calibrate.hierarchy import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
)
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.calibrate.registry import TargetSpec
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_build.transport_graph import (
    PERIOD,
    TOY_POPULATION,
    ToyConfig,
    edge,
    reference_document,
    reference_row,
    through,
    toy_frame,
    toy_graph,
    toy_registry,
    write_toy_sources,
)
from test_support.paths import paths_for

_BUILD = paths_for("microcosm-build")
MODULE_PATH = _BUILD.package / "src/microcosm/build/transport/gate_bindings.py"
NZ_ROOT = _BUILD.package / "src/microcosm/build/nz"
COUNTRY = "xx"

#: The nine implemented comparisons; pending bindings never pass.
BOUND = (
    "aggregate_admin",
    "calibration_reference_coverage",
    "exported_nonzero",
    "formula_owned_export",
    "nonnegative_columns",
    "per_family_fit",
    "target_profile_coverage",
    "weight_ess",
    "weight_ratio",
)
#: Registered fail-closed bindings awaiting producers and comparisons.
PENDING = ("release_input_coverage", "support")
PENDING_ARTIFACTS = {
    "support": ("donor_support_bounds",),
    "release_input_coverage": ("donor_artifact_receipt", "axiom_input_closure"),
}
#: Gates deliberately deferred by country policy.
UNBOUND = ("macro_realism",)

PROPERTY = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# ---------------------------------------------------------------------------
# Evidence builders
# ---------------------------------------------------------------------------


def _manifest(*entries: dict) -> GatesManifest:
    return GatesManifest.from_mapping(
        {
            "version": 1,
            "country": COUNTRY,
            "policy": "Transport gate binding tests.",
            "phases": ["terminal"],
            "gates": [dict(entry) for entry in entries],
        }
    )


def _entry(gate: str, parameters: dict | None = None, **extra) -> dict:
    entry = {
        "id": f"{gate}_entry",
        "gate": gate,
        "phase": "terminal",
        "criticality": "release_blocking",
    }
    if parameters:
        entry["parameters"] = parameters
    entry.update(extra)
    return entry


def _outcome(entry: dict, *, frame: Frame | None = None, **artifacts):
    report = evaluate_phase(
        _manifest(entry),
        "terminal",
        EvidenceContext(frame=frame, artifacts=artifacts),
        registry=TRANSPORT_GATE_REGISTRY,
    )
    (outcome,) = report.outcomes
    return outcome


def _diagnostics(rows, *, surface=None) -> dict:
    """A decoded schema-8 diagnostics payload: (family, name, error[, target, estimate]).

    With ``surface``, the payload records it as ``diagnostics.calibration@1``
    does (``build.surface_sha256``).
    """

    targets = []
    for family, name, error, *values in rows:
        target, estimate = values if values else (1.0, 1.0)
        targets.append(
            {
                "name": f"{name}@{PERIOD}",
                "target_name": name,
                "period": PERIOD,
                "relative_error": error,
                "target": target,
                "final_estimate": estimate,
                "registry": {"family": family},
            }
        )
    payload: dict = {"schema_version": 8, "targets": targets}
    if surface is not None:
        payload["build"] = {"surface_sha256": surface.sha256}
    return payload


def _hierarchy(name: str, level: str) -> CalibrationHierarchy:
    return CalibrationHierarchy(
        provider=HierarchyNode(id="toy_statistics", label="Toy statistics office"),
        category=HierarchyCategory(
            id="toy.population", label="Toy population", provider_id="toy_statistics"
        ),
        geography=HierarchyGeography(id=f"{level}.xx", label=level, level=level),
        dimensions=(),
        target=HierarchyNode(id=name, label=name),
    )


#: Toy-frame targets: (name, entity, measure, filter, family).
SPEC_UNIVERSE = (
    ("adults", "person", "is_adult", None, "demography"),
    ("income", "person", "employment_income", None, "income"),
    ("adult_income", "person", "employment_income", "is_adult", "income"),
    ("renters", "household", "is_renter", None, "housing"),
    ("rent", "household", "rent", None, "housing"),
    ("family_support", "family", "family_support", None, "benefits"),
)
FAMILIES = tuple(dict.fromkeys(row[4] for row in SPEC_UNIVERSE))


def _spec(
    name: str,
    *,
    entity: str = "person",
    measure: str = "age",
    filter_: str | None = None,
    family: str = "toy",
    value: float = 1000.0,
    tolerance: float | None = None,
    hierarchy: CalibrationHierarchy | None = None,
) -> TargetSpec:
    return TargetSpec(
        name=name,
        entity=entity,
        value=value,
        measure=measure,
        filter=filter_,
        period=PERIOD,
        source="test fixture",
        family=family,
        signed=value < 0,
        tolerance=tolerance,
        hierarchy=hierarchy,
    )


def _universe_specs(indices) -> tuple[TargetSpec, ...]:
    return tuple(
        _spec(name, entity=entity, measure=measure, filter_=filter_, family=family)
        for name, entity, measure, filter_, family in (
            SPEC_UNIVERSE[i] for i in sorted(indices)
        )
    )


def _surface(specs):
    """A decoded ``microcosm.targets.surface`` over ``specs``."""

    registry = TargetRegistry(list(specs), country=COUNTRY)
    references = parse_reference_document(
        reference_document(
            [
                reference_row(
                    spec.name,
                    entity=spec.entity,
                    measure=spec.measure,
                    filter_=spec.filter,
                )
                for spec in specs
            ]
        ),
        country=COUNTRY,
    )
    return decode_target_surface(
        encode_target_surface(
            registry,
            references,
            references_sha256="1" * 64,
            facts={"facts_sha256": "2" * 64},
        )
    )


def _problem(specs, *, surface_sha256: str):
    """A decoded ordered problem compiled on the toy frame, as targets.problem does."""

    frame = toy_frame()
    families = {spec.name: spec.family for spec in specs}
    compiled = build_constraint_matrix(
        frame, TargetRegistry(list(specs), country=COUNTRY).to_target_set()
    )
    return decode_problem(
        encode_problem(
            compiled,
            entity_ids=frame.table("household")["household_id"].tolist(),
            target_metadata=[
                {"reference": target.name, "family": families[target.name]}
                for target in compiled.targets
            ],
            bindings={"surface_sha256": surface_sha256},
        )
    )


@cache
def _surface_for(indices: frozenset[int]):
    return _surface(_universe_specs(indices))


@cache
def _problem_for(indices: frozenset[int], surface_sha256: str):
    return _problem(_universe_specs(indices), surface_sha256=surface_sha256)


def _frame(**columns) -> Frame:
    """The toy frame with named columns replaced (by entity)."""

    frame = toy_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column, values in columns.items():
        entity = next(e for e in frame.entities if column in tables[e].columns)
        tables[entity][column] = values
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def _weighted_frame(weights) -> Frame:
    frame = toy_frame()
    return Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "household": Weights(
                np.asarray(weights, dtype=np.float64), WeightKind.DESIGN
            )
        },
        frame.strata,
    )


def _aggregate_evidence(estimate: float) -> dict:
    surface = _surface([_spec("a", value=100.0)])
    return {
        "artifacts": {
            SURFACE: surface,
            DIAGNOSTICS: _diagnostics(
                [("toy", "a", 0.0, 100.0, estimate)], surface=surface
            ),
        }
    }


N_HOUSEHOLDS = len(TOY_POPULATION["households"])
N_PERSONS = len(TOY_POPULATION["persons"])

FIT = {
    "within": 0.05,
    "min_family_share": 0.5,
    "hard_within": 0.1,
    "min_hard_family_share": 1.0,
    "min_family_size": 1,
}

#: Per bound gate: a passing parameter row and its full evidence. The fail
#: and absent cases below are derived from these.
PASSING = {
    "per_family_fit": lambda: (
        FIT,
        {"artifacts": {DIAGNOSTICS: _diagnostics([("income", "a", 0.01)])}},
    ),
    "aggregate_admin": lambda: ({"default_rtol": 0.1}, _aggregate_evidence(101.0)),
    "calibration_reference_coverage": lambda: (
        {},
        {
            "artifacts": {
                SURFACE: _surface_for(frozenset(range(6))),
                PROBLEM: _problem_for(
                    frozenset(range(6)), _surface_for(frozenset(range(6))).sha256
                ),
            }
        },
    ),
    "target_profile_coverage": lambda: (
        {"required_families": list(FAMILIES)},
        {
            "artifacts": {
                PROBLEM: _problem_for(
                    frozenset(range(6)), _surface_for(frozenset(range(6))).sha256
                )
            }
        },
    ),
    "nonnegative_columns": lambda: (
        {"columns": ["household_weight", "age", "rent"]},
        {"frame": toy_frame()},
    ),
    "exported_nonzero": lambda: ({}, {"frame": toy_frame()}),
    "formula_owned_export": lambda: (
        {"formula_owned_columns": ["toy_benefit_amount"]},
        {"frame": toy_frame()},
    ),
    "weight_ess": lambda: ({"minimum_ess_fraction": 0.5}, {"frame": toy_frame()}),
    "weight_ratio": lambda: (
        {"maximum_max_to_median_ratio": 10.0},
        {"frame": toy_frame()},
    ),
}

#: Per bound gate: a failing parameter row and evidence.
FAILING = {
    "per_family_fit": lambda: (
        FIT,
        {"artifacts": {DIAGNOSTICS: _diagnostics([("income", "a", 0.5)])}},
    ),
    "aggregate_admin": lambda: ({"default_rtol": 0.1}, _aggregate_evidence(-100.0)),
    "calibration_reference_coverage": lambda: (
        {},
        {
            "artifacts": {
                SURFACE: _surface_for(frozenset(range(6))),
                PROBLEM: _problem_for(
                    frozenset(range(5)), _surface_for(frozenset(range(6))).sha256
                ),
            }
        },
    ),
    "target_profile_coverage": lambda: (
        {"required_families": [*FAMILIES, "superannuation"]},
        {
            "artifacts": {
                PROBLEM: _problem_for(
                    frozenset(range(6)), _surface_for(frozenset(range(6))).sha256
                )
            }
        },
    ),
    "nonnegative_columns": lambda: (
        {"columns": ["rent"]},
        {"frame": _frame(rent=[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])},
    ),
    "exported_nonzero": lambda: (
        {},
        {"frame": _frame(family_support=[0.0] * len(TOY_POPULATION["families"]))},
    ),
    "formula_owned_export": lambda: (
        {"formula_owned_columns": ["rent"]},
        {"frame": toy_frame()},
    ),
    "weight_ess": lambda: ({"minimum_ess_fraction": 1.0}, {"frame": toy_frame()}),
    "weight_ratio": lambda: (
        {"maximum_max_to_median_ratio": 1.0},
        {"frame": toy_frame()},
    ),
}


def _evidence_keys(gate: str) -> tuple[str, ...]:
    binding = TRANSPORT_GATE_REGISTRY[gate]
    keys = sorted(binding.required_artifacts({}))
    return (*(("frame",) if binding.requires_frame({}) else ()), *keys)


def _evaluate(gate: str, parameters: dict, evidence: dict, *, drop=()):
    artifacts = {
        key: value
        for key, value in evidence.get("artifacts", {}).items()
        if key not in drop
    }
    frame = None if "frame" in drop else evidence.get("frame")
    return _outcome(_entry(gate, parameters), frame=frame, **artifacts)


# ---------------------------------------------------------------------------
# The registry and the NZ manifest
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_extends_the_default_registry_with_the_bound_gates(self) -> None:
        assert set(TRANSPORT_GATE_REGISTRY) == (
            set(DEFAULT_REGISTRY) | set(BOUND) | set(PENDING)
        )
        for name in DEFAULT_REGISTRY:
            if name not in BOUND:
                assert TRANSPORT_GATE_REGISTRY[name] is DEFAULT_REGISTRY[name]
        for name, binding in TRANSPORT_GATE_REGISTRY.items():
            assert binding.name == name
        assert set(UNBOUND).isdisjoint(TRANSPORT_GATE_REGISTRY)

    @pytest.mark.parametrize("gate", PENDING)
    @pytest.mark.parametrize("supplied", (False, True))
    def test_pending_bindings_require_named_artifacts_and_never_pass(
        self, gate, supplied
    ) -> None:
        artifacts = {name: {"passed": True} for name in PENDING_ARTIFACTS[gate]}
        outcome = _outcome(_entry(gate), **(artifacts if supplied else {}))
        expected = GateStatus.FAILED if supplied else GateStatus.EVIDENCE_ABSENT
        assert outcome.status is expected
        binding = TRANSPORT_GATE_REGISTRY[gate]
        assert isinstance(binding, FunctionBinding)
        assert binding.__dataclass_params__.frozen
        assert isinstance(binding.artifact_arguments, type(TRANSPORT_GATE_REGISTRY))
        assert binding.required_artifacts({}) == frozenset(artifacts)
        assert binding.parameter_keys == required_gate_parameters(binding) == set()
        if supplied:
            assert any("not implemented" in line for line in outcome.result.failures)
        else:
            assert outcome.reason == "missing evidence: " + ", ".join(sorted(artifacts))

    def test_the_gate_kernel_accepts_and_describes_the_registry(self) -> None:
        kernel = GateBatteryKernel(TRANSPORT_GATE_REGISTRY)
        assert set(kernel.registry) == set(TRANSPORT_GATE_REGISTRY)
        assert len(kernel.implementation_hash()) == 64

    @pytest.mark.parametrize("gate", BOUND)
    def test_every_binding_has_pass_fail_and_absent_fixtures(self, gate) -> None:
        parameters, evidence = PASSING[gate]()
        assert _evaluate(gate, parameters, evidence).status is GateStatus.PASSED
        parameters, evidence = FAILING[gate]()
        outcome = _evaluate(gate, parameters, evidence)
        assert outcome.status is GateStatus.FAILED
        assert outcome.result.failures
        assert not any("failed closed" in line for line in outcome.result.failures), (
            outcome.result.failures
        )
        parameters, evidence = PASSING[gate]()
        absent = _evaluate(gate, parameters, evidence, drop=_evidence_keys(gate))
        assert absent.status is GateStatus.EVIDENCE_ABSENT
        assert absent.reason == "missing evidence: " + ", ".join(_evidence_keys(gate))

    @pytest.mark.parametrize("gate", BOUND)
    @PROPERTY
    @given(data=st.data())
    def test_property_any_missing_evidence_is_absent_never_passed(
        self, gate, data
    ) -> None:
        keys = _evidence_keys(gate)
        dropped = data.draw(
            st.lists(st.sampled_from(keys), min_size=1, unique=True), label="dropped"
        )
        parameters, evidence = PASSING[gate]()
        outcome = _evaluate(gate, parameters, evidence, drop=dropped)
        assert outcome.status is GateStatus.EVIDENCE_ABSENT
        for key in dropped:
            assert key in outcome.reason


class TestNewZealandManifest:
    @pytest.fixture(scope="class")
    def gates(self) -> GatesManifest:
        return load_country_spec("nz").gates

    def test_every_gate_is_bound_or_declared_not_applicable(self, gates) -> None:
        for entry in gates.gates:
            assert (
                entry.not_applicable is not None
                or entry.gate in TRANSPORT_GATE_REGISTRY
            ), entry.id

    def test_no_gate_resolves_to_an_unregistered_gap(self, gates) -> None:
        for phase in gates.phases:
            report = evaluate_phase(
                gates, phase, EvidenceContext(), registry=TRANSPORT_GATE_REGISTRY
            )
            for outcome in report.outcomes:
                if outcome.status is GateStatus.NOT_APPLICABLE:
                    assert outcome.reason == outcome.entry.not_applicable
                    continue
                assert outcome.status is GateStatus.EVIDENCE_ABSENT, outcome.entry.id
                assert outcome.reason.startswith("missing evidence: "), outcome.reason

    def test_parameters_are_routed_and_complete(self, gates) -> None:
        validate_gate_parameters(gates, TRANSPORT_GATE_REGISTRY)
        validate_required_gate_parameters(gates)

    def test_pending_gates_remain_applicable_and_fail_closed(self, gates) -> None:
        for entry in gates.gates:
            if entry.gate in UNBOUND:
                assert entry.not_applicable is not None, entry.id
        by_gate = {entry.gate: entry for entry in gates.gates}
        document = json.loads((NZ_ROOT / "gates.json").read_text())
        for gate in PENDING:
            entry = by_gate[gate]
            assert entry.not_applicable is None
            assert entry.evidence_absent_blocks
            raw = next(row for row in document["gates"] if row["gate"] == gate)
            assert raw["parameters"] == {}

    @pytest.mark.parametrize("supplied", (False, True))
    def test_pending_gates_block_candidates_with_every_other_gate_passing(
        self, gates, supplied, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv(
            gate_signing_key_env("nz"), base64.b64encode(b"t" * 32).decode()
        )
        artifacts = {
            name: {"passed": True}
            for names in PENDING_ARTIFACTS.values()
            for name in names
        }
        evaluated = evaluate_phase(
            gates,
            "terminal",
            EvidenceContext(artifacts=artifacts if supplied else {}),
            registry=TRANSPORT_GATE_REGISTRY,
        )
        report = GatePhaseReport(
            phase="terminal",
            outcomes=tuple(
                outcome
                if outcome.entry.gate in PENDING
                else GateOutcome(
                    entry=replace(outcome.entry, not_applicable=None),
                    status=GateStatus.PASSED,
                    result=GateResult(outcome.entry.gate, True, (), {}),
                )
                for outcome in evaluated.outcomes
            ),
        )
        # The policy must match the isolated outcomes even for deferred gates.
        isolated = replace(gates, gates=tuple(o.entry for o in report.outcomes))
        run = GateBatteryRun(
            isolated,
            release_id="pending-candidate",
            report_path=tmp_path / "battery.json",
            release_candidate=True,
            registry=TRANSPORT_GATE_REGISTRY,
        )
        run.record_phase(report)
        blocked = run.enforce("terminal", mode=BlockingMode.MARKS_ARTIFACT)
        graph = phase_enforcement(
            report,
            release_candidate=True,
            synthetic_smoke=False,
            upstream_blocked=False,
        )
        assert blocked is True
        assert run.report_payload()["shippable"] is False
        assert graph.artifact_permitted is False
        assert set(graph.blocking) == {
            entry.id for entry in gates.gates if entry.gate in PENDING
        }
        expected = GateStatus.FAILED if supplied else GateStatus.EVIDENCE_ABSENT
        assert all(
            o.status is (expected if o.entry.gate in PENDING else GateStatus.PASSED)
            for o in report.outcomes
        )

    def test_applicable_entries_use_no_python_threshold_default(self, gates) -> None:
        for entry in gates.gates:
            if entry.not_applicable is not None:
                continue
            binding = TRANSPORT_GATE_REGISTRY[entry.gate]
            assert isinstance(binding, FunctionBinding)
            defaulted = {
                name: parameter.default
                for name, parameter in inspect.signature(
                    binding.gate
                ).parameters.items()
                if parameter.default is not inspect.Parameter.empty
                and name not in entry.parameters
            }
            assert all(value is None for value in defaulted.values()), (
                entry.id,
                defaulted,
            )

    def test_declared_nonnegative_columns_are_the_source_manifest_outputs(
        self, gates
    ) -> None:
        sources = load_country_spec("nz").sources
        declared = list(
            dict.fromkeys(
                column
                for stage in sources.stages
                for column in stage.nonnegative_outputs
            )
        )
        (entry,) = (e for e in gates.gates if e.gate == "nonnegative_columns")
        assert list(entry.parameters["columns"]) == declared

    def test_declared_formula_owned_columns_are_the_export_contract_list(
        self, gates
    ) -> None:
        contract = json.loads((NZ_ROOT / "export_contract.json").read_text())
        (entry,) = (e for e in gates.gates if e.gate == "formula_owned_export")
        assert list(entry.parameters["formula_owned_columns"]) == list(
            contract["formula_owned_excluded"]
        )


class TestModuleBoundaries:
    def test_imports_no_country_runtime(self) -> None:
        text = MODULE_PATH.read_text(encoding="utf-8")
        assert "uk_runtime" not in text
        assert "us_runtime" not in text
        imported = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert not any(
            name.startswith(("microcosm.build.uk", "microcosm.build.us"))
            for name in imported
        ), imported

    def test_holds_no_threshold_literal(self) -> None:
        numbers = [
            node.value
            for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ]
        assert not [value for value in numbers if isinstance(value, float)]
        assert set(numbers) <= {0, 1}

    @pytest.mark.parametrize("gate", (*BOUND, *PENDING))
    def test_bound_evaluators_default_nothing_but_none(self, gate) -> None:
        binding = TRANSPORT_GATE_REGISTRY[gate]
        defaults = [
            parameter.default
            for parameter in inspect.signature(binding.gate).parameters.values()
            if parameter.default is not inspect.Parameter.empty
        ]
        assert all(value is None for value in defaults)

    def test_a_manifest_missing_a_threshold_is_refused_and_fails_closed(self) -> None:
        entry = _entry("weight_ess", {})
        with pytest.raises(ValueError, match="minimum_ess_fraction"):
            validate_required_gate_parameters(_manifest(entry))
        outcome = _outcome(entry, frame=toy_frame())
        assert outcome.status is GateStatus.FAILED
        assert "minimum_ess_fraction" in outcome.result.failures[0]

    def test_required_parameters_are_the_undefaulted_thresholds(self) -> None:
        assert required_gate_parameters(TRANSPORT_GATE_REGISTRY["per_family_fit"]) == {
            "within",
            "min_family_share",
            "hard_within",
            "min_hard_family_share",
            "min_family_size",
        }
        assert (
            required_gate_parameters(
                TRANSPORT_GATE_REGISTRY["calibration_reference_coverage"]
            )
            == set()
        )
        assert (
            required_gate_parameters(TRANSPORT_GATE_REGISTRY["weights_audit"]) == set()
        )

    @pytest.mark.parametrize(
        ("gate", "parameters"),
        [
            ("weight_ess", {"minimum_ess_fraction": True}),
            ("weight_ratio", {"maximum_max_to_median_ratio": True}),
            ("per_family_fit", {**FIT, "hard_within": True}),
            ("aggregate_admin", {"default_rtol": True}),
        ],
    )
    def test_a_boolean_threshold_is_refused_not_read_as_a_number(
        self, gate, parameters
    ) -> None:
        _, evidence = PASSING[gate]()
        outcome = _evaluate(gate, parameters, evidence)
        assert outcome.status is GateStatus.FAILED
        assert "must be a number" in outcome.result.failures[0]


# ---------------------------------------------------------------------------
# Per-binding properties: pass side passes, fail side fails
# ---------------------------------------------------------------------------

_SHARE = st.floats(min_value=0.0, max_value=1.0)
_ERROR = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False)


class TestPerFamilyFit:
    @PROPERTY
    @given(
        rows=st.lists(
            st.tuples(st.sampled_from(("demography", "income", "benefits")), _ERROR),
            min_size=1,
            max_size=12,
        ),
        within=st.floats(min_value=0.0, max_value=1.0),
        min_family_share=_SHARE,
        hard_within=st.floats(min_value=0.0, max_value=1.0),
        min_hard_family_share=_SHARE,
        min_family_size=st.integers(min_value=0, max_value=4),
        families=st.none()
        | st.lists(
            st.sampled_from(("demography", "income", "benefits", "superannuation")),
            min_size=1,
            unique=True,
        ),
    )
    def test_property_verdict_follows_the_declared_thresholds(
        self,
        rows,
        within,
        min_family_share,
        hard_within,
        min_hard_family_share,
        min_family_size,
        families,
    ) -> None:
        # Only the hard thresholds decide; the diagnostic pair is reported.
        assume(all(abs(abs(error) - hard_within) > 1e-9 for _, error in rows))
        parameters = {
            "within": within,
            "min_family_share": min_family_share,
            "hard_within": hard_within,
            "min_hard_family_share": min_hard_family_share,
            "min_family_size": min_family_size,
        }
        if families is not None:
            parameters["families"] = families
        diagnostics = _diagnostics(
            (family, f"t{index}", error) for index, (family, error) in enumerate(rows)
        )
        outcome = _evaluate(
            "per_family_fit", parameters, {"artifacts": {DIAGNOSTICS: diagnostics}}
        )
        selected = [
            (family, error)
            for family, error in rows
            if families is None or family in families
        ]
        by_family: dict[str, list[float]] = {}
        for family, error in selected:
            by_family.setdefault(family, []).append(abs(error))
        weak = [
            family
            for family, errors in by_family.items()
            if len(errors) >= min_family_size
            and np.mean([error <= hard_within for error in errors])
            < min_hard_family_share
        ]
        missing = [] if families is None else sorted(set(families) - set(by_family))
        expected = GateStatus.FAILED if weak or missing else GateStatus.PASSED
        assert outcome.status is expected
        details = outcome.result.details
        assert details["within"] == within
        assert details["min_family_share"] == min_family_share
        assert details["hard_within"] == hard_within
        assert details["min_hard_family_share"] == min_hard_family_share
        assert details["min_family_size"] == min_family_size

    def test_the_hard_thresholds_decide_and_the_diagnostic_pair_reports(
        self,
    ) -> None:
        diagnostics = {DIAGNOSTICS: _diagnostics([("income", "a", 0.5)])}
        loose_hard = {**FIT, "within": 0.0, "min_family_share": 1.0, "hard_within": 1.0}
        tight_hard = {**FIT, "within": 1.0, "min_family_share": 0.0, "hard_within": 0.1}
        passed = _evaluate("per_family_fit", loose_hard, {"artifacts": diagnostics})
        assert passed.status is GateStatus.PASSED
        assert passed.result.details["family_within_shares"] == {"income": 0.0}
        failed = _evaluate("per_family_fit", tight_hard, {"artifacts": diagnostics})
        assert failed.status is GateStatus.FAILED
        assert failed.result.details["family_within_shares"] == {"income": 1.0}

    def test_a_report_only_hard_threshold_is_refused(self) -> None:
        outcome = _evaluate(
            "per_family_fit",
            {**FIT, "hard_within": None},
            {"artifacts": {DIAGNOSTICS: _diagnostics([("income", "a", 5.0)])}},
        )
        assert outcome.status is GateStatus.FAILED
        assert "hard_within must be a number" in outcome.result.failures[0]

    def test_a_row_without_a_finite_error_or_a_family_fails_closed(self) -> None:
        for diagnostics in (
            _diagnostics([("income", "a", None)]),
            _diagnostics([("income", "a", 0.0), ("income", "a", 0.0)]),
            {"targets": [{"name": "a@2026", "relative_error": 0.0}]},
            {"targets": []},
        ):
            outcome = _evaluate(
                "per_family_fit", FIT, {"artifacts": {DIAGNOSTICS: diagnostics}}
            )
            assert outcome.status is GateStatus.FAILED


class TestAggregateAdmin:
    @PROPERTY
    @given(
        anchors=st.lists(
            st.tuples(
                st.floats(min_value=-1e7, max_value=1e7).filter(lambda v: abs(v) >= 1),
                st.booleans(),
                st.floats(min_value=0.0, max_value=0.45),
            ),
            min_size=1,
            max_size=6,
        ),
        rtol=st.floats(min_value=0.01, max_value=0.9),
    )
    def test_property_verdict_follows_the_declared_tolerance(
        self, anchors, rtol
    ) -> None:
        specs, rows = [], []
        for index, (value, passes, fraction) in enumerate(anchors):
            # Pass side: within half the tolerance; fail side: a sign flip or
            # a miss of at least twice the tolerance, in the anchor's sign.
            if passes:
                achieved = value * (1 + fraction * rtol)
            else:
                achieved = -value if fraction > 0.2 else value * (1 + 2 * rtol + 0.01)
            name = f"anchor_{index}"
            specs.append(_spec(name, value=value))
            rows.append(("toy", name, 0.0, value, achieved))
        surface = _surface(specs)
        outcome = _evaluate(
            "aggregate_admin",
            {"default_rtol": rtol},
            {
                "artifacts": {
                    SURFACE: surface,
                    DIAGNOSTICS: _diagnostics(rows, surface=surface),
                }
            },
        )
        expected = (
            GateStatus.PASSED
            if all(passes for _, passes, _ in anchors)
            else GateStatus.FAILED
        )
        assert outcome.status is expected
        assert outcome.result.name == "aggregate_admin"
        assert outcome.result.details["default_rtol"] == rtol

    def test_surface_evidence_cannot_change_the_declared_threshold(self) -> None:
        surface = _surface([_spec("a", value=100.0, tolerance=50.0)])
        outcome = _evaluate(
            "aggregate_admin",
            {"default_rtol": 0.01},
            {
                "artifacts": {
                    SURFACE: surface,
                    DIAGNOSTICS: _diagnostics(
                        [("toy", "a", 0.0, 100.0, 140.0)], surface=surface
                    ),
                }
            },
        )
        assert outcome.status is GateStatus.FAILED
        assert outcome.result.details["default_rtol"] == 0.01
        assert surface.registry.specs[0].tolerance == 50.0

    def test_an_unmeasured_anchor_or_empty_selection_fails(self) -> None:
        surface = _surface([_spec("a"), _spec("b")])
        diagnostics = _diagnostics([("toy", "a", 0.0, 1000.0, 1000.0)], surface=surface)
        evidence = {"artifacts": {SURFACE: surface, DIAGNOSTICS: diagnostics}}
        outcome = _evaluate("aggregate_admin", {"default_rtol": 0.1}, evidence)
        assert outcome.status is GateStatus.FAILED
        assert any("did not measure" in line for line in outcome.result.failures)
        outcome = _evaluate(
            "aggregate_admin",
            {"default_rtol": 0.1, "families": ["income"]},
            evidence,
        )
        assert outcome.status is GateStatus.FAILED
        assert any("income" in line for line in outcome.result.failures)

    def test_diagnostics_from_another_surface_fail(self) -> None:
        surface = _surface([_spec("a", value=100.0)])
        other = _surface([_spec("a", value=100.0), _spec("b")])
        for diagnostics in (
            _diagnostics([("toy", "a", 0.0, 100.0, 100.0)], surface=other),
            _diagnostics([("toy", "a", 0.0, 100.0, 100.0)]),
        ):
            outcome = _evaluate(
                "aggregate_admin",
                {"default_rtol": 0.1},
                {"artifacts": {SURFACE: surface, DIAGNOSTICS: diagnostics}},
            )
            assert outcome.status is GateStatus.FAILED
            assert "The diagnostics record surface" in outcome.result.failures[0]

    def test_geography_levels_select_anchors_by_their_hierarchy(self) -> None:
        specs = [
            _spec("national", value=100.0, hierarchy=_hierarchy("national", "country")),
            _spec("north", value=50.0, hierarchy=_hierarchy("north", "region")),
        ]
        surface = _surface(specs)
        # The regional anchor misses by half; the national anchor is exact.
        diagnostics = _diagnostics(
            [("toy", "national", 0.0, 100.0, 100.0), ("toy", "north", 0.0, 50.0, 25.0)],
            surface=surface,
        )
        evidence = {"artifacts": {SURFACE: surface, DIAGNOSTICS: diagnostics}}
        country = _evaluate(
            "aggregate_admin",
            {"default_rtol": 0.1, "geography_levels": ["country"]},
            evidence,
        )
        assert country.status is GateStatus.PASSED
        assert country.result.details["anchors"] == ["national"]
        both = _evaluate(
            "aggregate_admin",
            {"default_rtol": 0.1, "geography_levels": ["country", "region"]},
            evidence,
        )
        assert both.status is GateStatus.FAILED
        assert both.result.details["anchors"] == ["national", "north"]
        absent = _evaluate(
            "aggregate_admin",
            {"default_rtol": 0.1, "geography_levels": ["territorial_authority"]},
            evidence,
        )
        assert absent.status is GateStatus.FAILED

    def test_a_level_filter_over_an_unclassified_anchor_fails_closed(self) -> None:
        surface = _surface([_spec("a", value=100.0)])
        outcome = _evaluate(
            "aggregate_admin",
            {"default_rtol": 0.1, "geography_levels": ["country"]},
            {
                "artifacts": {
                    SURFACE: surface,
                    DIAGNOSTICS: _diagnostics(
                        [("toy", "a", 0.0, 100.0, 100.0)], surface=surface
                    ),
                }
            },
        )
        assert outcome.status is GateStatus.FAILED
        assert "has no hierarchy" in outcome.result.failures[0]


class TestCalibrationReferenceCoverage:
    @PROPERTY
    @given(
        surface=st.frozensets(st.integers(0, 5), min_size=1),
        matrix=st.frozensets(st.integers(0, 5), min_size=1),
        bound_to_surface=st.booleans(),
    )
    def test_property_passes_iff_the_matrix_is_the_activated_surface(
        self, surface, matrix, bound_to_surface
    ) -> None:
        compiled = _surface_for(surface)
        problem = _problem_for(
            matrix, compiled.sha256 if bound_to_surface else "f" * 64
        )
        outcome = _evaluate(
            "calibration_reference_coverage",
            {},
            {"artifacts": {SURFACE: compiled, PROBLEM: problem}},
        )
        expected = surface == matrix and bound_to_surface
        assert (outcome.status is GateStatus.PASSED) is expected
        details = outcome.result.details
        assert details["activated"] == details["resolved"] == len(surface)
        assert details["matrix"] == len(matrix)

    def test_a_skipped_target_fails(self) -> None:
        specs = (*_universe_specs(range(2)), _spec("ghost", measure="absent_column"))
        surface = _surface(specs)
        frame = toy_frame()
        compiled = build_constraint_matrix(
            frame, TargetRegistry(list(specs), country=COUNTRY).to_target_set()
        )
        assert compiled.skipped
        problem = decode_problem(
            encode_problem(
                compiled,
                entity_ids=frame.table("household")["household_id"].tolist(),
                bindings={"surface_sha256": surface.sha256},
            )
        )
        outcome = _evaluate(
            "calibration_reference_coverage",
            {},
            {"artifacts": {SURFACE: surface, PROBLEM: problem}},
        )
        assert outcome.status is GateStatus.FAILED
        assert outcome.result.details["skipped"] == [f"ghost@{PERIOD}"]


class TestTargetProfileCoverage:
    @pytest.mark.parametrize("reason", (None, False, 0, {}, [], "   "))
    def test_reviewed_exclusion_requires_a_nonempty_string(self, reason) -> None:
        outcome = _evaluate(
            "target_profile_coverage",
            {
                "required_families": ["absent"],
                "reviewed_exclusions": {"absent": reason},
            },
            {"artifacts": {PROBLEM: _problem_for(frozenset({0}), "e" * 64)}},
        )
        assert outcome.status is GateStatus.FAILED
        assert "non-empty string" in outcome.result.failures[0]

    @PROPERTY
    @given(
        matrix=st.frozensets(st.integers(0, 5), min_size=1),
        required=st.lists(
            st.sampled_from((*FAMILIES, "superannuation")), min_size=1, unique=True
        ),
        excluded=st.lists(st.sampled_from((*FAMILIES, "superannuation")), unique=True),
    )
    def test_property_passes_iff_every_required_family_is_calibrated(
        self, matrix, required, excluded
    ) -> None:
        excluded = [family for family in excluded if family in required]
        parameters = {"required_families": required}
        if excluded:
            parameters["reviewed_exclusions"] = {
                family: "excluded in a test" for family in excluded
            }
        outcome = _evaluate(
            "target_profile_coverage",
            parameters,
            {"artifacts": {PROBLEM: _problem_for(matrix, "e" * 64)}},
        )
        present = {SPEC_UNIVERSE[index][4] for index in matrix}
        missing = set(required) - present - set(excluded)
        assert (outcome.status is GateStatus.PASSED) is (not missing)
        for family in missing:
            assert any(line.startswith(family) for line in outcome.result.failures)


class TestFrameGates:
    @pytest.mark.parametrize("reason", (None, False, 0, {}, [], "   "))
    def test_reviewed_exclusion_requires_a_nonempty_string(self, reason) -> None:
        outcome = _evaluate(
            "nonnegative_columns",
            {"columns": ["age"], "reviewed_exclusions": {"age": reason}},
            {"frame": _frame(age=[-1.0] * N_PERSONS)},
        )
        assert outcome.status is GateStatus.FAILED
        assert "non-empty string" in outcome.result.failures[0]

    @PROPERTY
    @given(
        rent=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            min_size=N_HOUSEHOLDS,
            max_size=N_HOUSEHOLDS,
        ),
        declared=st.lists(
            st.sampled_from(("rent", "age", "household_weight", "not_a_column")),
            min_size=1,
            unique=True,
        ),
    )
    def test_property_nonnegative_columns(self, rent, declared) -> None:
        outcome = _evaluate(
            "nonnegative_columns",
            {"columns": declared},
            {"frame": _frame(rent=rent)},
        )
        bad = ("rent" in declared and min(rent) < 0) or "not_a_column" in declared
        assert (outcome.status is GateStatus.FAILED) is bad
        assert outcome.result.details["columns_required"] == len(declared)

    @pytest.mark.parametrize("bad", (-math.inf, math.nan, math.inf))
    def test_a_non_finite_value_in_a_declared_column_fails(self, bad) -> None:
        rent = [bad, 0.0, 1.0, 2.0, 3.0, 4.0]
        outcome = _evaluate(
            "nonnegative_columns", {"columns": ["rent"]}, {"frame": _frame(rent=rent)}
        )
        assert outcome.status is GateStatus.FAILED
        assert outcome.result.details["nonfinite_counts"] == {"rent": 1}

    def test_a_reviewed_exclusion_excuses_its_column(self) -> None:
        rent = [-1.0, math.nan, 1.0, 2.0, 3.0, 4.0]
        outcome = _evaluate(
            "nonnegative_columns",
            {"columns": ["rent", "age"], "reviewed_exclusions": {"rent": "a test"}},
            {"frame": _frame(rent=rent)},
        )
        assert outcome.status is GateStatus.PASSED
        assert outcome.result.details["reviewed_exclusions"] == {"rent": "a test"}

    @PROPERTY
    @given(
        zeroed=st.lists(
            st.sampled_from(("rent", "employment_income", "family_support", "age")),
            unique=True,
        ),
        exempt=st.lists(
            st.sampled_from(("rent", "employment_income", "family_support", "age")),
            unique=True,
        ),
    )
    def test_property_exported_nonzero(self, zeroed, exempt) -> None:
        sizes = {
            "rent": N_HOUSEHOLDS,
            "employment_income": N_PERSONS,
            "family_support": len(TOY_POPULATION["families"]),
            "age": N_PERSONS,
        }
        frame = _frame(**{column: [0] * sizes[column] for column in zeroed})
        parameters = (
            {"exemptions": {column: "zero in a test" for column in exempt}}
            if exempt
            else {}
        )
        outcome = _evaluate("exported_nonzero", parameters, {"frame": frame})
        assert (outcome.status is GateStatus.FAILED) is bool(set(zeroed) - set(exempt))

    def test_exported_nonzero_skips_structural_columns_and_reads_weights(
        self,
    ) -> None:
        outcome = _evaluate("exported_nonzero", {}, {"frame": toy_frame()})
        assert outcome.status is GateStatus.PASSED
        frame = toy_frame()
        measured = outcome.result.details["columns_checked"]
        structural = {"person_id", "person_household_id", "person_family_id"}
        structural |= {"household_id", "family_id"}
        all_columns = {
            column for entity in frame.entities for column in frame.table(entity)
        }
        assert measured == len(all_columns - structural) + 1  # + household_weight
        assert outcome.result.details["columns"] == sorted(
            (all_columns - structural) | {"household_weight"}
        )

    @PROPERTY
    @given(
        owned=st.lists(
            st.sampled_from(("rent", "age", "toy_benefit", "person_id", "eligible")),
            min_size=1,
            unique=True,
        )
    )
    def test_property_formula_owned_export(self, owned) -> None:
        outcome = _evaluate(
            "formula_owned_export",
            {"formula_owned_columns": owned},
            {"frame": toy_frame()},
        )
        offenders = sorted(set(owned) & {"rent", "age"})
        assert (outcome.status is GateStatus.FAILED) is bool(offenders)
        assert outcome.result.details["offenders"] == offenders
        assert "household_weight" in outcome.result.details["exported_columns"]

    def test_weight_gates_refuse_an_ambiguous_weight_entity(self) -> None:
        frame = toy_frame()
        two = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {
                "household": frame.weights_for("household"),
                "person": Weights(np.ones(N_PERSONS), WeightKind.DESIGN),
            },
            frame.strata,
        )
        outcome = _evaluate("weight_ess", {"minimum_ess_fraction": 0.1}, {"frame": two})
        assert outcome.status is GateStatus.FAILED


_WEIGHTS = st.lists(
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False), min_size=1, max_size=40
)


class TestWeightGates:
    @pytest.mark.parametrize(
        ("weights", "gate", "keyword", "threshold", "summary_field", "expected"),
        [
            (
                [1e154, 5e153],
                weight_ess_gate,
                "minimum_ess_fraction",
                0.95,
                "ess_fraction",
                0.9,
            ),
            (
                [1e-162, 2e-162],
                weight_ess_gate,
                "minimum_ess_fraction",
                0.99,
                "ess_fraction",
                0.9,
            ),
            (
                [1e308, 9e307],
                weight_ratio_gate,
                "maximum_max_to_median_ratio",
                1.01,
                "max_to_median_positive_weight",
                1 / 0.95,
            ),
        ],
    )
    def test_extreme_finite_weights_do_not_false_pass(
        self, weights, gate, keyword, threshold, summary_field, expected
    ) -> None:
        result = gate(weights, **{keyword: threshold})
        assert result.passed is False
        assert result.details[summary_field] == pytest.approx(expected)
        if not math.isfinite(sum(weights)):
            assert result.details["total_weight"] is None
        normalized = np.asarray(weights) / max(weights)
        assert gate(normalized, **{keyword: threshold}).passed is result.passed

    @PROPERTY
    @example(weights=[1, 2], scale=1e-162, minimum=0.99, maximum=10.0)
    @given(
        weights=st.lists(st.integers(0, 1024), min_size=2, max_size=20).filter(any),
        scale=st.floats(min_value=1e-200, max_value=1e200),
        minimum=st.floats(min_value=0.01, max_value=1.0),
        maximum=st.floats(min_value=0.01, max_value=100.0),
    )
    def test_property_verdicts_are_invariant_to_positive_rescaling(
        self, weights, scale, minimum, maximum
    ) -> None:
        values = np.asarray(weights, dtype=np.float64)
        # Stay away from the comparison boundary where float rounding matters.
        positive = values[values > 0]
        fraction = values.sum() ** 2 / np.square(values).sum() / values.size
        ratio = values.max() / np.median(positive)
        assume(abs(fraction - minimum) > 1e-9)
        assume(abs(ratio - maximum) > 1e-9 * maximum)
        for gate, keyword, threshold in (
            (weight_ess_gate, "minimum_ess_fraction", minimum),
            (weight_ratio_gate, "maximum_max_to_median_ratio", maximum),
        ):
            baseline = gate(values, **{keyword: threshold})
            scaled = gate(values * scale, **{keyword: threshold})
            assert scaled.passed is baseline.passed

    def test_normalization_preserves_original_positive_mask_and_record_counts(
        self,
    ) -> None:
        summary = weight_summary([0.0, 5e-324, 1e308])
        assert summary["n_records"] == 3
        assert summary["positive_weight_records"] == 2
        assert summary["zero_weight_records"] == 1
        assert summary["ess_fraction"] == pytest.approx(1 / 3)
        assert summary["max_to_median_positive_weight"] == pytest.approx(2.0)

    @PROPERTY
    @given(weights=_WEIGHTS, minimum=st.floats(min_value=1e-6, max_value=1.0))
    def test_property_ess_floor(self, weights, minimum) -> None:
        # Exact arithmetic keeps the oracle independent of float squaring.
        values = [Fraction(value) for value in weights]
        total = sum(values)
        squares = sum(value * value for value in values)
        fraction = 0.0 if not squares else float(total * total / squares / len(values))
        positive = sorted(value for value in values if value > 0)
        if positive:
            middle = len(positive) // 2
            median = (
                positive[middle]
                if len(positive) % 2
                else (positive[middle - 1] + positive[middle]) / 2
            )
            ratio = positive[-1] / median
            representable = ratio <= Fraction(float(np.finfo(np.float64).max))
        else:
            representable = True
        assume(abs(fraction - minimum) > 1e-9)
        result = weight_ess_gate(weights, minimum_ess_fraction=minimum)
        assert result.passed is bool(representable and fraction >= minimum)
        assert result.details["minimum_ess_fraction"] == minimum
        if representable:
            assert math.isclose(
                result.details["ess_fraction"], fraction, rel_tol=1e-12, abs_tol=1e-15
            )
        else:
            assert result.failures

    @PROPERTY
    @given(weights=_WEIGHTS, maximum=st.floats(min_value=1e-3, max_value=1e3))
    def test_property_max_to_median_ceiling(self, weights, maximum) -> None:
        positive = sorted(Fraction(value) for value in weights if value > 0)
        if positive:
            middle = len(positive) // 2
            median = (
                positive[middle]
                if len(positive) % 2
                else (positive[middle - 1] + positive[middle]) / 2
            )
            ratio = positive[-1] / median
        else:
            ratio = None
        assume(
            ratio is None or abs(ratio - Fraction(maximum)) > Fraction(1e-9 * maximum)
        )
        result = weight_ratio_gate(weights, maximum_max_to_median_ratio=maximum)
        assert result.passed is bool(ratio is not None and ratio <= Fraction(maximum))
        assert result.details["maximum_max_to_median_ratio"] == maximum
        if ratio is not None:
            if ratio <= Fraction(float(np.finfo(np.float64).max)):
                assert math.isclose(
                    result.details["max_to_median_positive_weight"],
                    float(ratio),
                    rel_tol=1e-12,
                )
            else:
                assert result.failures

    @PROPERTY
    @given(
        weights=st.lists(
            st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
            min_size=N_HOUSEHOLDS,
            max_size=N_HOUSEHOLDS,
        ).filter(lambda w: any(value > 0 for value in w)),
        minimum=st.floats(min_value=1e-6, max_value=1.0),
        maximum=st.floats(min_value=1e-3, max_value=1e3),
    )
    def test_property_frame_bindings_equal_the_gate_functions(
        self, weights, minimum, maximum
    ) -> None:
        frame = _weighted_frame(weights)
        for gate, parameters, direct in (
            (
                "weight_ess",
                {"minimum_ess_fraction": minimum},
                weight_ess_gate(weights, minimum_ess_fraction=minimum),
            ),
            (
                "weight_ratio",
                {"maximum_max_to_median_ratio": maximum},
                weight_ratio_gate(weights, maximum_max_to_median_ratio=maximum),
            ),
        ):
            outcome = _evaluate(gate, parameters, {"frame": frame})
            assert outcome.result.passed is direct.passed
            assert outcome.result.failures == direct.failures
            assert outcome.result.details == {
                **direct.details,
                "weight_entity": "household",
            }

    def test_an_overflowing_summary_fails_closed(self) -> None:
        # Normalization keeps ESS finite; one dominant row still fails the floor.
        result = weight_ess_gate([1e200, 1.0, 1.0], minimum_ess_fraction=0.9)
        assert not result.passed

    def test_summary_of_an_all_zero_vector_is_reportable(self) -> None:
        summary = weight_summary([0.0, 0.0])
        assert summary["ess_fraction"] == 0
        assert summary["median_positive_weight"] is None
        assert summary["max_to_median_positive_weight"] is None
        assert not weight_ratio_gate([0.0, 0.0], maximum_max_to_median_ratio=2.0).passed

    @pytest.mark.parametrize(
        ("gate", "keyword", "value"),
        [
            (weight_ess_gate, "minimum_ess_fraction", 0.0),
            (weight_ess_gate, "minimum_ess_fraction", 1.5),
            (weight_ratio_gate, "maximum_max_to_median_ratio", 0.0),
            (weight_ratio_gate, "maximum_max_to_median_ratio", math.inf),
        ],
    )
    def test_out_of_range_thresholds_are_refused(self, gate, keyword, value) -> None:
        with pytest.raises(ValueError):
            gate([1.0, 2.0], **{keyword: value})


# ---------------------------------------------------------------------------
# Through gates.battery@1 in the toy transport graph
# ---------------------------------------------------------------------------


#: A gates.json-shaped document over the toy population that uses every
#: bound gate; ``required_families`` is the knob the failing run turns.
def _toy_document(required_families) -> dict:
    gates = [
        _entry(
            "per_family_fit",
            {
                "within": 0.5,
                "min_family_share": 0.0,
                "hard_within": 0.5,
                "min_hard_family_share": 1.0,
                "min_family_size": 1,
                "families": ["toy"],
            },
        ),
        _entry("aggregate_admin", {"default_rtol": 0.5}),
        _entry("calibration_reference_coverage", evidence_absent_blocks=True),
        _entry("target_profile_coverage", {"required_families": required_families}),
        _entry(
            "nonnegative_columns",
            {"columns": ["household_weight", "age", "rent", "employment_income"]},
        ),
        _entry("exported_nonzero", evidence_absent_blocks=True),
        _entry("formula_owned_export", {"formula_owned_columns": ["toy_benefit"]}),
        _entry("weight_ess", {"minimum_ess_fraction": 0.1}),
        _entry("weight_ratio", {"maximum_max_to_median_ratio": 1000.0}),
        _entry(
            "macro_realism",
            not_applicable="The toy has no national-accounts evidence.",
        ),
    ]
    return {
        "country": COUNTRY,
        "version": 1,
        "policy": "Toy gates over the transport bindings (test only).",
        "phases": ["terminal"],
        "gates": gates,
    }


def _gate_graph(config: ToyConfig):
    """The toy graph through its terminal gate, with ``surface`` wired to it."""

    graph = toy_graph(config, through_prepare=True)
    nodes = {node.id: node for node in graph.nodes}
    gate = nodes["toy.gates.terminal"]
    gate = replace(
        gate,
        artifact_inputs=(
            *gate.artifact_inputs,
            edge("surface", nodes["toy.targets.compile"]),
        ),
    )
    graph = replace(
        graph,
        nodes=tuple(gate if node.id == gate.id else node for node in graph.nodes),
    )
    return through(graph, "toy.gates.terminal")


@pytest.fixture(scope="module")
def toy_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("transport-gate-bindings")
    sources = write_toy_sources(root / "sources")
    store = ContentStore(root / "store")
    registry = toy_registry(TRANSPORT_GATE_REGISTRY)
    runs = {}
    for label, families in (("pass", ["toy"]), ("fail", ["toy", "absent_family"])):
        config = ToyConfig(gates=_toy_document(families))
        graph = _gate_graph(config)
        used = {name for node in graph.nodes for name in node.sources}
        manifest = run_graph(
            compile_graph(graph),
            sources={k: v for k, v in sources.mapping().items() if k in used},
            store=store,
            kernels=registry,
        )
        runs[label] = (config, graph, manifest)
    return store, runs


def _report(store, manifest):
    receipt = manifest.nodes["toy.gates.terminal"]
    return decode_gate_report(store.load_bytes(receipt.opaque_artifacts["gate_report"]))


class TestThroughTheGateKernel:
    def test_every_bound_gate_passes_on_the_calibrated_toy(self, toy_runs) -> None:
        store, runs = toy_runs
        report = _report(store, runs["pass"][2])
        statuses = {o.entry.gate: o.status for o in report.report.outcomes}
        assert statuses == {
            **dict.fromkeys(BOUND, GateStatus.PASSED),
            "macro_realism": GateStatus.NOT_APPLICABLE,
        }
        assert report.outcome == "pass"
        assert report.artifact_permitted

    def test_a_missing_family_fails_the_node_and_blocks_the_artifact(
        self, toy_runs
    ) -> None:
        store, runs = toy_runs
        report = _report(store, runs["fail"][2])
        failed = [
            o.entry.gate
            for o in report.report.outcomes
            if o.status is GateStatus.FAILED
        ]
        assert failed == ["target_profile_coverage"]
        assert report.outcome == "fail"
        assert not report.artifact_permitted
        assert report.document["enforcement"]["blocking"] == [
            "target_profile_coverage_entry"
        ]

    @pytest.mark.parametrize("label", ("pass", "fail"))
    def test_differential_the_kernel_equals_the_battery_on_the_same_evidence(
        self, toy_runs, label
    ) -> None:
        store, runs = toy_runs
        config, graph, manifest = runs[label]
        nodes = {node.id: node for node in graph.nodes}
        artifacts = {}
        for item in nodes["toy.gates.terminal"].artifact_inputs:
            payload = store.load_bytes(
                manifest.nodes[item.producer].opaque_artifacts[item.artifact]
            )
            decoder = EVIDENCE_DECODERS.get(item.type)
            artifacts[item.name] = payload if decoder is None else decoder(payload)
        frame = store.load_frame(manifest.nodes["toy.terminal"].frame_key)
        gates = GatesManifest.from_mapping(config.gates, country=COUNTRY)
        direct = evaluate_phase(
            gates,
            "terminal",
            EvidenceContext(frame=frame, artifacts=artifacts),
            registry=TRANSPORT_GATE_REGISTRY,
        )
        report = _report(store, manifest)
        assert report.document["report"] == json.loads(
            json.dumps(gate_phase_report_payload(direct, gates=gates))
        )

    def test_calibrated_weights_are_what_the_weight_gates_read(self, toy_runs) -> None:
        store, runs = toy_runs
        manifest = runs["pass"][2]
        frame = store.load_frame(manifest.nodes["toy.terminal"].frame_key)
        weights = frame.weights_for("household")
        assert weights.kind is WeightKind.CALIBRATED
        report = _report(store, manifest)
        (ess,) = (o for o in report.report.outcomes if o.entry.gate == "weight_ess")
        assert ess.result.details["total_weight"] == pytest.approx(
            float(weights.values.sum())
        )
        assert ess.result.details["n_records"] == N_HOUSEHOLDS


def test_module_exports_its_documented_surface() -> None:
    assert set(gate_bindings.__all__) == {
        "DIAGNOSTICS",
        "PROBLEM",
        "SURFACE",
        "TRANSPORT_GATE_REGISTRY",
        "required_gate_parameters",
        "validate_required_gate_parameters",
        "weight_ess_gate",
        "weight_ratio_gate",
        "weight_summary",
    }
