"""Opt-in grouped Adam with mandatory complete cold/warm runner admission.

Only the integrating runner authenticates full Frames and original live graph
anchors. Typed transport and kernel diagnostics cannot substitute for that
boundary. This first profile accepts a complete initial two-role clone, never
a refitted checkpoint as a replacement budget root.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from pandas.api.types import is_complex_dtype, is_numeric_dtype

import microcosm.calibrate.group_bounds as grouped_math
from microcosm.calibrate import calibrate, diagnostics_payload
from microcosm.frame import WeightKind
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    run_graph,
    source_hash,
)
from microcosm.graph.artifact_edges import value_from_descriptor
from microcosm.graph.canonical import canonical_json, sha256_domain
from microcosm.graph.codecs import load_raw_bytes
from microcosm.graph.keys import opaque_artifact_key

from . import demographic_calibration_graph as demographic
from . import graph_native_household_origin as origin
from . import origin_group_budget as budgets

BUDGET_SOURCE = "frozen_origin_group_budget"
BUDGET_NODE = "origin_group_cap.budget"
CALIBRATION_NODE = "origin_group_cap.calibration"


def _require(condition, reason):
    if not condition:
        raise budgets.OriginBudgetError(reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _origin_inputs(native_binding_node, expanded_binding_node):
    return (
        ArtifactInput(
            "native_origin", native_binding_node, "binding", origin.BINDING_TYPE
        ),
        ArtifactInput(
            "expanded_origin", expanded_binding_node, "binding", origin.BINDING_TYPE
        ),
    )


def _verify_origin_edges(budget, artifacts):
    for name, prefix in (("native_origin", "native"), ("expanded_origin", "expanded")):
        value = artifacts[name]
        _require(
            value.type == origin.BINDING_TYPE
            and value.key == opaque_artifact_key(value.producer_key, "binding")
            and value.key == budget.document[f"{prefix}_artifact_key"]
            and _sha(value.payload) == budget.document[f"{prefix}_binding_sha256"],
            "GROUPED_ORIGIN_EDGE",
        )


def origin_budget_node(*, base, native_binding_node, expanded_binding_node):
    return Node(
        BUDGET_NODE,
        FrozenOriginBudgetKernel.ref,
        population=base,
        sources=(BUDGET_SOURCE,),
        artifact_inputs=_origin_inputs(native_binding_node, expanded_binding_node),
        artifact_outputs=(ArtifactOutput("budget", budgets.BUDGET_TYPE),),
    )


class FrozenOriginBudgetKernel(KernelBase):
    """Carry declared frozen bytes with their actual typed origin ancestry."""

    ref = "us.origin_group_cap.budget@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            budgets,
            grouped_math,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        aliases = context.node.artifact_inputs
        _require(
            tuple(a.name for a in aliases) == ("native_origin", "expanded_origin"),
            "GROUPED_BUDGET_DECLARATION",
        )
        expected = origin_budget_node(
            base=context.node.population,
            native_binding_node=aliases[0].producer,
            expanded_binding_node=aliases[1].producer,
        )
        _require(
            context.node.normative() == expected.normative(),
            "GROUPED_BUDGET_DECLARATION",
        )
        budget = budgets.FrozenOriginBudget.from_payload(
            load_raw_bytes(context.sources[BUDGET_SOURCE])
        )
        _verify_origin_edges(budget, context.artifacts)
        return KernelResult(
            artifacts={"budget": budget.payload},
            receipt={
                "budget_sha256": budget.digest,
                "constraint_digest": budget.grouped_bounds.digest,
                "group_count": budget.document["group_count"],
                "release_eligible": False,
            },
        )


def grouped_demographic_calibration_node(
    registry,
    *,
    base,
    native_binding_node,
    expanded_binding_node,
    epochs,
    learning_rate,
    design_multiplier,
    incoming_multiplier,
):
    frozen = demographic._registry_from_json(demographic._registry_json(registry))
    _require(type(epochs) is int and epochs >= 1, "GROUPED_EPOCHS")
    for value in (learning_rate, design_multiplier, incoming_multiplier):
        _require(
            type(value) in (float, int) and np.isfinite(value) and value > 0,
            "GROUPED_OPTIONS",
        )
    return Node(
        CALIBRATION_NODE,
        GroupedDemographicCalibrationKernel.ref,
        base=base,
        inputs=(Slice("household", tuple(dict.fromkeys(s.measure for s in frozen))),),
        structural=StructuralDelta.REWEIGHT,
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
        params={
            "registry": demographic._registry_json(frozen),
            "epochs": epochs,
            "learning_rate": learning_rate,
            # The executor retains its independent necessary row-DESIGN guard.
            "max_weight_ratio": design_multiplier,
            "weight_anchor": "design",
            "incoming_multiplier": incoming_multiplier,
            "mass": "free",
        },
        artifact_inputs=(
            ArtifactInput("budget", BUDGET_NODE, "budget", budgets.BUDGET_TYPE),
            *_origin_inputs(native_binding_node, expanded_binding_node),
        ),
        artifact_outputs=(ArtifactOutput("diagnostics", demographic.DIAGNOSTICS_TYPE),),
    )


class GroupedDemographicCalibrationKernel(KernelBase):
    """Real float64 grouped Adam; float32 dense/sparse target arithmetic."""

    ref = "us.origin_group_cap.demographic_adam@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.REWEIGHT,
        consumes_se=False,
        dependencies=("numpy", "pandas", "scipy", "torch"),
    )

    def __init__(self, *, _post_projection_observer=None):
        self._observer = _post_projection_observer

    def implementation_hash(self):
        return _sha(
            canonical_json(
                {
                    "adapter": source_hash(
                        sys.modules[__name__],
                        budgets,
                        grouped_math,
                        dependencies=self.capabilities.dependencies,
                    ),
                    "demographic_solver": demographic.DemographicCalibrationKernel().implementation_hash(),
                }
            )
        )

    def run(self, context):
        params = context.params
        registry = demographic._registry_from_json(params["registry"])
        aliases = context.node.artifact_inputs
        _require(
            tuple(a.name for a in aliases)
            == ("budget", "native_origin", "expanded_origin"),
            "GROUPED_CALIBRATION_DECLARATION",
        )
        expected = grouped_demographic_calibration_node(
            registry,
            base=context.node.base,
            native_binding_node=aliases[1].producer,
            expanded_binding_node=aliases[2].producer,
            epochs=params["epochs"],
            learning_rate=params["learning_rate"],
            design_multiplier=params["max_weight_ratio"],
            incoming_multiplier=params["incoming_multiplier"],
        )
        _require(
            context.node.normative() == expected.normative(),
            "GROUPED_CALIBRATION_DECLARATION",
        )
        value = context.artifacts["budget"]
        _require(
            value.type == budgets.BUDGET_TYPE
            and value.key == opaque_artifact_key(value.producer_key, "budget"),
            "GROUPED_BUDGET_EDGE",
        )
        budget = budgets.FrozenOriginBudget.from_payload(value.payload)
        _verify_origin_edges(budget, context.artifacts)
        _require(
            float(params["max_weight_ratio"]).hex()
            == budget.document["design_multiplier_hex"]
            and float(params["incoming_multiplier"]).hex()
            == budget.document["incoming_multiplier_hex"],
            "GROUPED_BUDGET_MULTIPLIERS",
        )
        weights = context.weights["household"]
        _require(weights.kind is WeightKind.IMPORTANCE, "GROUPED_INITIAL_IMPORTANCE")
        table = context.tables["household"]
        for spec in registry:
            values = table[spec.measure]
            _require(
                is_numeric_dtype(values.dtype) and not is_complex_dtype(values.dtype),
                "GROUPED_COUNT_DTYPE",
            )
            numbers = values.to_numpy(dtype=float, na_value=np.nan)
            _require(
                np.all(
                    np.isfinite(numbers)
                    & (numbers >= 0)
                    & (numbers == np.floor(numbers))
                ),
                "GROUPED_COUNT_VALUES",
            )
        final_observation = None

        def observe(payload):
            nonlocal final_observation
            final_observation = payload["weights"].copy()
            if self._observer is not None:
                self._observer(payload)

        result = calibrate(
            demographic.kernels_module._frame_from_context(context, "household"),
            registry.to_target_set(),
            weight_entity="household",
            method="adam",
            seed=0,
            epochs=params["epochs"],
            learning_rate=params["learning_rate"],
            mass="free",
            # Group capacity replaces the scalar incoming-importance row cap.
            max_weight_ratio=None,
            grouped_upper_bounds=budget.grouped_bounds,
            _post_projection_observer=observe,
        )
        _require(not result.skipped, "GROUPED_SKIPPED_TARGETS")
        final_weights = result.frame.weights_for("household")
        _require(
            final_observation is not None
            and final_observation.tobytes() == final_weights.values.tobytes(),
            "GROUPED_FINAL_OBSERVER_BYTES",
        )
        anchors = {
            "solver_weight_anchor": "frozen_original_origin",
            "budget_sha256": budget.digest,
            "budget_artifact_key": value.key,
            "executor_weight_anchor": "original_design",
            "executor_max_weight_ratio": params["max_weight_ratio"],
            "accepted_weight_sha256": _sha(final_weights.values.tobytes()),
            **budget.diagnostics(
                final_weights.values,
                result.options["grouped_upper_bounds"]["last_corrected_group_count"],
            ),
        }
        payload = canonical_json(
            diagnostics_payload(result, target_registry=registry, build=anchors)
        )
        return KernelResult(
            weights=final_weights,
            artifacts={"diagnostics": payload},
            receipt={
                "scope": "invented_or_declared_national_origin_group_calibration",
                "evidence_scope": registry.specs[0].metadata["evidence_scope"],
                "diagnostics_sha256": _sha(payload),
                **anchors,
                "release_eligible": False,
                "national_validation": "not_evaluated",
            },
        )


@dataclass(frozen=True)
class AdmittedOriginCalibration:
    manifest: object
    budget: budgets.FrozenOriginBudget
    successor: budgets.WeightOnlySuccessor
    successor_key: str
    diagnostics: dict


def run_origin_group_calibration(
    prefix_graph,
    *,
    sources,
    store,
    kernels,
    native_binding_node,
    expanded_binding_node,
    target_registry,
    budget_path: Path,
    epochs: int,
    learning_rate: float,
    design_multiplier: float,
    incoming_multiplier: float,
    resume="auto",
):
    """Execute and admit the complete first cloned calibration, cold or warm.

    The caller registers the two new kernels, and supplies only declared local
    inputs. No source approvals are minted here. Repeated calls reuse the exact
    existing budget bytes; a different root, map or multiplier refuses. This
    bounded first adapter does not support pruning or refitting a successor.
    """
    live = {}
    prefix = run_graph(
        compile_graph(prefix_graph),
        sources=sources,
        store=store,
        kernels=kernels,
        resume=resume,
        _population_observer=lambda node, pop: live.__setitem__(node, pop),
    )
    arguments = dict(
        native_population=live[native_binding_node],
        expanded_population=live[expanded_binding_node],
        native_binding_node=native_binding_node,
        expanded_binding_node=expanded_binding_node,
        design_multiplier=design_multiplier,
        incoming_multiplier=incoming_multiplier,
    )
    expected = budgets.freeze_origin_group_budget(prefix, store, **arguments)
    budget_path = Path(budget_path)
    if budget_path.exists():
        budget = budgets.FrozenOriginBudget.from_payload(budget_path.read_bytes())
        budgets.verify_origin_group_budget(budget, prefix, store, **arguments)
    else:
        _require(resume != "require", "GROUPED_WARM_BUDGET_ABSENT")
        budget = expected
        with budget_path.open("xb") as output:
            output.write(budget.payload)
    # Complete zero-row lineage is already authenticated; positivity is the
    # first solver's explicit limitation, enforced before any fitting.
    budget.grouped_bounds.check(
        arguments["expanded_population"].frame.weights_for("household").values,
        positive=True,
    )
    base = arguments["expanded_population"].version
    edge_args = dict(
        base=base,
        native_binding_node=native_binding_node,
        expanded_binding_node=expanded_binding_node,
    )
    graph = replace(
        prefix_graph,
        sources=(*prefix_graph.sources, SourceRef(BUDGET_SOURCE, "raw-bytes-v1")),
        nodes=(
            *prefix_graph.nodes,
            origin_budget_node(**edge_args),
            grouped_demographic_calibration_node(
                target_registry,
                **edge_args,
                epochs=epochs,
                learning_rate=learning_rate,
                design_multiplier=design_multiplier,
                incoming_multiplier=incoming_multiplier,
            ),
        ),
    )

    def capture(node, population):
        live[node] = population
        if node == CALIBRATION_NODE:
            # Independent of solver execution: also called on cached restoration.
            values = population.frame.weights_for("household").values
            budget.grouped_bounds.check(values, positive=True)
            budget.diagnostics(values)

    manifest = run_graph(
        compile_graph(graph),
        sources={**sources, BUDGET_SOURCE: budget_path},
        store=store,
        kernels=kernels,
        resume="require" if resume == "require" else "auto",
        _population_observer=capture,
    )
    arguments.update(
        native_population=live[native_binding_node],
        expanded_population=live[expanded_binding_node],
    )
    budgets.verify_origin_group_budget(budget, manifest, store, **arguments)
    receipt = manifest.node(CALIBRATION_NODE)
    budget_receipt = manifest.node(BUDGET_NODE)
    value = value_from_descriptor(
        store.load_bytes(budget_receipt.opaque_artifacts["budget"]),
        budget_receipt.typed_artifacts["outputs"]["budget"],
    )
    _require(
        value.type == budgets.BUDGET_TYPE and value.payload == budget.payload,
        "GROUPED_MATERIALIZED_BUDGET",
    )
    current = live[CALIBRATION_NODE]
    successor = budgets.bind_weight_only_successor(
        budget, previous=arguments["expanded_population"], current=current
    )
    final_weights = current.frame.weights_for("household").values
    corrected = receipt.receipt["last_corrected_group_count"]
    diagnostics = budget.diagnostics(final_weights, corrected)
    _require(
        canonical_json({name: receipt.receipt.get(name) for name in diagnostics})
        == canonical_json(diagnostics)
        and receipt.receipt["accepted_weight_sha256"] == _sha(final_weights.tobytes()),
        "GROUPED_MATERIALIZED_DIAGNOSTICS",
    )
    diagnostic_payload = store.load_bytes(receipt.opaque_artifacts["diagnostics"])
    diagnostic_value = value_from_descriptor(
        diagnostic_payload, receipt.typed_artifacts["outputs"]["diagnostics"]
    )
    diagnostic_document = json.loads(diagnostic_payload)
    anchors = diagnostic_document.get("build", {})
    _require(
        diagnostic_value.type == demographic.DIAGNOSTICS_TYPE
        and diagnostic_value.producer_key == receipt.key
        and diagnostic_value.key == opaque_artifact_key(receipt.key, "diagnostics")
        and _sha(diagnostic_payload) == receipt.receipt["diagnostics_sha256"]
        and canonical_json(diagnostic_document) == diagnostic_payload
        and canonical_json({name: anchors.get(name) for name in diagnostics})
        == canonical_json(diagnostics)
        and anchors.get("budget_sha256") == budget.digest
        and anchors.get("budget_artifact_key") == value.key
        and anchors.get("accepted_weight_sha256") == _sha(final_weights.tobytes()),
        "GROUPED_MATERIALIZED_DIAGNOSTIC_ARTIFACT",
    )
    _require(
        canonical_json(
            diagnostic_document.get("options", {}).get("grouped_upper_bounds")
        )
        == canonical_json(budget.grouped_bounds.diagnostics(final_weights, corrected)),
        "GROUPED_MATERIALIZED_DIAGNOSTIC_OPTIONS",
    )
    # Runner-authored successor identity is distinct from all kernel artifacts.
    successor_key = sha256_domain(
        "origin-weight-only-successor",
        canonical_json(
            {
                "calibration_node_key": receipt.key,
                "budget_artifact_key": value.key,
                "type": {
                    "name": budgets.SUCCESSOR_TYPE.name,
                    "schema_version": budgets.SUCCESSOR_TYPE.schema_version,
                },
                "payload_sha256": successor.digest,
            }
        ),
    )
    store.put_bytes(successor_key, successor.payload)
    _require(
        store.load_bytes(successor_key) == successor.payload,
        "GROUPED_STORED_SUCCESSOR_BYTES",
    )
    return AdmittedOriginCalibration(
        manifest, budget, successor, successor_key, diagnostics
    )
