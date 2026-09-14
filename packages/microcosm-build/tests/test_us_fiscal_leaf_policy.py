"""Invented policy/graph boundaries; no country engine or population source reads."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_graph_fiscal_measurement import InventedSource, context, declaration, frame

from microcosm.build.us_runtime import fiscal_leaf_policy as policy
from microcosm.build.us_runtime import graph_fiscal_measurement as stage
from microcosm.frame import VariableMetadata
from microcosm.frame.adapters.policyengine_us import VariableDependencyClosure
from microcosm.graph import (
    ContentStore,
    Graph,
    KernelRegistry,
    Node,
    NodeRejectedError,
    Owned,
    SourceRef,
    StoreMissError,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import node_key


def document(*, assumption=True):
    entries = {
        "amount": {"kind": "producer", "entity": "person", "producer": "survey.amount"},
        "scenario_multiplier": {
            "kind": "producer",
            "entity": "person",
            "producer": "survey.behavior",
        },
    }
    if assumption:
        entries["scenario_multiplier"] = {
            "kind": "assumption",
            "entity": "person",
            "value": 1.0,
            "interpretation": "baseline_behavior",
            "period": 2024,
            "roots": ["invented_fiscal"],
            "affected_roots": ["invented_fiscal"],
            "affected_programs": ["invented program"],
            "rationale": "Invented boundary control, not a supported US assumption.",
            "reviewer": "Invented reviewer; no real approval implied",
            "source_issue": "fixture:invented-policy-review",
            "reform_sensitivity": "Constant participation changes the response to eligibility reforms.",
        }
    return {
        "schema_version": 1,
        "artifact_kind": "microcosm.us.fiscal_leaf_policy",
        "period": 2024,
        "roots": ["invented_fiscal"],
        "entries": entries,
    }


def load(raw=None):
    payload = canonical_json(document() if raw is None else raw)
    return policy.load_fiscal_leaf_policy(
        payload, expected_sha256=hashlib.sha256(payload).hexdigest()
    )


@pytest.fixture
def mocked_model(monkeypatch):
    """Replace only country metadata/evaluation; actual fiscal graph stays real."""
    calls = {"materialize": 0, "defaults": 0}
    default = {"value": 0.0}

    class Index:
        def variable_dependency_closure(self, root):
            return VariableDependencyClosure(
                "invented",
                root,
                ("amount", "scenario_multiplier"),
                (root,),
                ((root, "amount"), (root, "scenario_multiplier")),
                "a" * 64,
            )

        def variable_metadata(self, name):
            return VariableMetadata(name, "person", "float", "year")

    class Engine:
        def default_values(self, names):
            calls["defaults"] += 1
            return {name: default["value"] for name in names}

        def materialize(self, value, names, period):
            calls["materialize"] += 1
            assert period == 2024
            return {
                name: value.table("person").amount.to_numpy(copy=True) for name in names
            }

    monkeypatch.setattr(
        stage.runtime, "engine_runtime_identity", lambda: {"fixture": "engine"}
    )
    monkeypatch.setattr(
        stage.policyengine_us, "PolicyEngineUSVariableMetadataIndex", Index
    )
    monkeypatch.setattr(stage.policyengine_us, "PolicyEngineUSEngine", Engine)
    return calls, default


def arguments(leaf_policy=None):
    result = declaration(
        value_variable="invented_fiscal", model_outputs=("invented_fiscal",)
    )
    if leaf_policy is not None:
        result["leaf_policy"] = leaf_policy
    return result


def test_pinned_document_is_canonical_detached_and_detects_changed_bytes():
    pinned = load()
    raw = policy.fiscal_leaf_policy_document(pinned)
    raw["entries"]["scenario_multiplier"]["value"] = 3.0
    assert policy.fiscal_leaf_policy_document(pinned) == document()
    with pytest.raises(ValueError, match="FINGERPRINT"):
        policy.load_fiscal_leaf_policy(
            pinned.payload + b" ", expected_sha256=pinned.sha256
        )
    with pytest.raises(ValueError, match="CANONICAL"):
        altered = json.dumps(document(), indent=2).encode()
        policy.load_fiscal_leaf_policy(
            altered, expected_sha256=hashlib.sha256(altered).hexdigest()
        )
    with pytest.raises(ValueError, match="FINGERPRINT"):
        policy.fiscal_leaf_policy_document(
            replace(pinned, payload=pinned.payload + b" ")
        )


@pytest.mark.parametrize(
    "field",
    [
        "rationale",
        "reviewer",
        "source_issue",
        "value",
        "period",
        "roots",
        "affected_roots",
        "affected_programs",
        "reform_sensitivity",
        "interpretation",
    ],
)
def test_missing_assumption_review_or_scope_refuses_without_engine(field):
    raw = document()
    del raw["entries"]["scenario_multiplier"][field]
    with pytest.raises(ValueError, match="ASSUMPTION_FIELDS"):
        load(raw)


@pytest.mark.parametrize(
    "value",
    [None, "engine default", {"default": True}, [0], float("nan"), float("inf"), 2**64],
)
def test_only_finite_bounded_numeric_or_boolean_literals_are_supported(value):
    raw = document()
    raw["entries"]["scenario_multiplier"]["value"] = value
    with pytest.raises((ValueError, TypeError)):
        load(raw)


@pytest.mark.parametrize(
    "changed",
    [
        "inactive",
        "wrong_period",
        "wrong_roots",
        "empty_programs",
        "empty_rationale",
        "unknown_field",
    ],
)
def test_inactivity_or_invalid_review_scope_never_becomes_an_allowlist(changed):
    raw = document()
    entry = raw["entries"]["scenario_multiplier"]
    if changed == "inactive":
        entry["kind"] = "inactive"
    elif changed == "wrong_period":
        entry["period"] = 2026
    elif changed == "wrong_roots":
        entry["roots"] = ["other_root"]
    elif changed == "empty_programs":
        entry["affected_programs"] = []
    elif changed == "empty_rationale":
        entry["rationale"] = " "
    else:
        entry["allow_missing"] = True
    with pytest.raises(ValueError, match="FISCAL_LEAF_POLICY"):
        load(raw)


def test_exact_closure_coverage_and_entity_are_required(mocked_model):
    for change, reason in (
        ("missing", "UNPOLICIED"),
        ("extra", "STALE"),
        ("entity", "ENTITY"),
    ):
        raw = document()
        if change == "missing":
            del raw["entries"]["amount"]
        elif change == "extra":
            raw["entries"]["unrelated"] = {
                "kind": "producer",
                "entity": "person",
                "producer": "fixture",
            }
        else:
            raw["entries"]["amount"]["entity"] = "household"
        with pytest.raises(ValueError, match=reason):
            stage.fiscal_measurement_node(**arguments(load(raw)))


def test_exact_root_period_and_supplied_assumption_collision(mocked_model):
    params = arguments(load())
    params["period"] = 2026
    with pytest.raises(ValueError, match="POLICY_PERIOD"):
        stage.fiscal_measurement_node(**params)
    params = arguments(load())
    params["model_outputs"] = ("different_root",)
    with pytest.raises(ValueError, match="POLICY_ROOTS"):
        stage.fiscal_measurement_node(**params)
    params = arguments(load())
    params["input_columns"]["tax_unit"] += ("scenario_multiplier",)
    with pytest.raises(ValueError, match="ASSUMPTION_COLUMN_COLLISION"):
        stage.fiscal_measurement_node(**params)


def test_default_remains_strict_and_explicit_producers_keep_unknownness(mocked_model):
    with pytest.raises(ValueError, match="UNDECLARED_MODEL_LEAF"):
        stage.fiscal_measurement_node(**arguments())
    value = frame()
    value.table("person")["scenario_multiplier"] = 0.0
    params = arguments()
    params["input_columns"]["person"] += ("scenario_multiplier",)
    default_node = stage.fiscal_measurement_node(**params)
    assert default_node == stage.fiscal_measurement_node(**params, leaf_policy=None)
    default_result = stage.FiscalMeasurementKernel(
        model_outputs=params["model_outputs"]
    ).run(context(params, value))
    assert mocked_model[0]["materialize"] == 1
    assert "leaf_policy" not in json.loads(default_node.params["model_contract"])
    assert mocked_model[0]["defaults"] == 0
    params["leaf_policy"] = load(document(assumption=False))
    result = stage.FiscalMeasurementKernel(
        model_outputs=params["model_outputs"], leaf_policy=params["leaf_policy"]
    ).run(context(params, value))
    np.testing.assert_array_equal(
        stage.decode_fiscal_measurement(
            default_result.artifacts["measurement"]
        ).matrix.toarray(),
        stage.decode_fiscal_measurement(
            result.artifacts["measurement"]
        ).matrix.toarray(),
    )
    value.table("person").loc[0, "scenario_multiplier"] = np.nan
    with pytest.raises(ValueError, match="MISSING_MODEL_INPUT"):
        stage.FiscalMeasurementKernel(
            model_outputs=params["model_outputs"], leaf_policy=params["leaf_policy"]
        ).run(context(params, value))
    assert mocked_model[0]["materialize"] == 2


def test_assumption_policy_binds_literal_review_and_engine_record_but_cannot_execute(
    mocked_model,
):
    params = arguments(load())
    c = context(params)
    model = json.loads(c.params["model_contract"])
    assert model["leaf_policy"]["document"] == document()
    assert model["leaf_policy"]["engine_records"]["scenario_multiplier"]["default"] == {
        "available": True,
        "value": 0.0,
    }
    with pytest.raises(ValueError, match="ASSUMPTION_PARENT_ADMISSION_UNSUPPORTED"):
        stage.FiscalMeasurementKernel(
            model_outputs=params["model_outputs"], leaf_policy=params["leaf_policy"]
        ).run(c)
    assert mocked_model[0]["materialize"] == 0
    changed = document()
    changed["entries"]["scenario_multiplier"]["value"] = 2.0
    assert (
        stage.fiscal_measurement_node(**arguments(load(changed))).normative()
        != c.node.normative()
    )
    changed["entries"]["scenario_multiplier"]["value"] = 1.0
    changed["entries"]["scenario_multiplier"]["reviewer"] = "Another reviewer"
    assert (
        stage.fiscal_measurement_node(**arguments(load(changed))).normative()
        != c.node.normative()
    )
    mocked_model[1]["value"] = 0.5
    assert stage.fiscal_measurement_node(**params).normative() != c.node.normative()


@pytest.mark.parametrize(
    "values", [[1.0, 2.0, 3.0, 4.0], [np.nan] * 4, [np.nan, 2.0, 3.0, 4.0]]
)
def test_full_parent_validator_catches_omitted_and_partially_unknown_columns(values):
    value = frame()
    value.table("person")["scenario_multiplier"] = values
    assert "scenario_multiplier" not in declaration()["input_columns"]["person"]
    with pytest.raises(ValueError, match="ASSUMPTION_COLUMN_COLLISION"):
        policy.validate_fiscal_leaf_policy_population(value, load())
    with pytest.raises(ValueError, match="ASSUMPTION_COLUMN_COLLISION"):
        policy.private_fiscal_assumption_frame(value, load())


def test_private_materialization_never_modifies_or_aliases_the_supplied_population():
    value = frame()
    before = {e: value.table(e).copy(deep=True) for e in value.entities}
    policy.validate_fiscal_leaf_policy_population(value, load())
    work = policy.private_fiscal_assumption_frame(value, load())
    np.testing.assert_array_equal(work.table("person").scenario_multiplier, np.ones(4))
    work.table("person").loc[0, "amount"] = 999.0
    for entity in value.entities:
        pd.testing.assert_frame_equal(value.table(entity), before[entity])
    np.testing.assert_array_equal(
        work.weights_for("household").values, value.weights_for("household").values
    )


def test_affected_roots_must_include_every_declared_consumer(mocked_model):
    raw = document()
    raw["roots"].append("other_root")
    raw["entries"]["scenario_multiplier"]["roots"].append("other_root")
    params = arguments(load(raw))
    params["model_outputs"] = tuple(raw["roots"])
    with pytest.raises(ValueError, match="ASSUMPTION_AFFECTED_ROOTS"):
        stage.fiscal_measurement_node(**params)
    raw["entries"]["scenario_multiplier"]["affected_roots"] = list(raw["roots"])
    params["leaf_policy"] = load(raw)
    stage.fiscal_measurement_node(**params)


@pytest.mark.parametrize(
    "dtype,value",
    [("bool", 1), ("int", True), ("int", 1.5), ("float", True), ("str", 1)],
)
def test_engine_metadata_prevents_literal_type_coercion(dtype, value):
    raw = document()
    raw["entries"]["scenario_multiplier"]["value"] = value
    raw = policy.fiscal_leaf_policy_document(load(raw))
    metadata = {
        "scenario_multiplier": {
            "name": "scenario_multiplier",
            "entity": "person",
            "dtype": dtype,
            "period": "year",
        }
    }
    with pytest.raises(ValueError, match="ENGINE_LITERAL_TYPE"):
        policy.fiscal_assumption_engine_records(raw, metadata=metadata, defaults={})


def test_final_model_fence_rejects_changed_policy_after_materialization(
    mocked_model, monkeypatch
):
    raw = document(assumption=False)
    value = frame()
    value.table("person")["scenario_multiplier"] = 0.0
    params = arguments(load(raw))
    params["input_columns"]["person"] += ("scenario_multiplier",)
    kernel = stage.FiscalMeasurementKernel(
        model_outputs=params["model_outputs"], leaf_policy=params["leaf_policy"]
    )
    original = stage.policyengine_us.PolicyEngineUSEngine.materialize

    def mutate(self, *args, **kwargs):
        outputs = original(self, *args, **kwargs)
        changed = document(assumption=False)
        changed["entries"]["amount"]["producer"] = "different.source"
        kernel.leaf_policy = load(changed)
        return outputs

    monkeypatch.setattr(
        stage.policyengine_us.PolicyEngineUSEngine, "materialize", mutate
    )
    with pytest.raises(ValueError, match="MODEL_DRIFT"):
        kernel.run(context(params, value))


def _source_node(*, has_scenario):
    outputs = (
        Owned("person", "amount", "float64"),
        Owned("person", "keep", "bool"),
        Owned("tax_unit", "tax_unit_amount", "float64"),
        Owned("household", "state_fips", "int64"),
        Owned("household", "congressional_district_geoid", "int64"),
    )
    if has_scenario:
        outputs += (Owned("person", "scenario_multiplier", "float64"),)
    return Node(
        "invented",
        InventedSource.ref,
        sources=("fixture",),
        structural=StructuralDelta.CREATE,
        outputs=outputs,
    )


def test_literal_policy_and_engine_defaults_change_actual_graph_key(mocked_model):
    create = _source_node(has_scenario=False)
    keys, implementation_hashes = [], []
    for value in (1.0, 2.0, 1.0):
        raw = document()
        raw["entries"]["scenario_multiplier"]["value"] = value
        if len(keys) == 2:
            mocked_model[1]["value"] = 0.5
        pinned = load(raw)
        measure = stage.fiscal_measurement_node(**arguments(pinned))
        compiled = compile_graph(
            Graph("us", (SourceRef("fixture", "frame-store"),), (create, measure))
        )
        kernel = stage.FiscalMeasurementKernel(
            model_outputs=("invented_fiscal",), leaf_policy=pinned
        )
        implementation = kernel.implementation_hash()
        implementation_hashes.append(implementation)
        keys.append(
            node_key(
                compiled,
                measure.id,
                {create.id: "a" * 64},
                implementation,
                {},
                kernel_capabilities=kernel.capabilities,
            )
        )
    assert len(set(keys)) == len(set(implementation_hashes)) == 3
    assert mocked_model[0]["materialize"] == 0


def test_explicit_producer_actual_graph_cold_required_and_policy_rekey(
    tmp_path, mocked_model, monkeypatch
):
    original = frame()
    original.table("person")["scenario_multiplier"] = 0.0
    raw = document(assumption=False)
    pinned = load(raw)
    params = arguments(pinned)
    params["input_columns"]["person"] += ("scenario_multiplier",)
    create = _source_node(has_scenario=True)
    measure = stage.fiscal_measurement_node(**params)
    sources = (SourceRef("fixture", "frame-store"),)
    compiled = compile_graph(Graph("us", sources, (create, measure)))
    registry = KernelRegistry()
    registry.register(InventedSource())
    registry.register(
        stage.FiscalMeasurementKernel(
            model_outputs=params["model_outputs"], leaf_policy=pinned
        )
    )
    store = ContentStore(tmp_path / "store")
    source = store.put_frame(
        hashlib.sha256(b"invented-policy-parent").hexdigest(), original
    )
    run_args = {"sources": {"fixture": source}, "kernels": registry, "store": store}
    cold = run_graph(compiled, **run_args)
    cold_bytes = store.load_bytes(cold.node(measure.id).opaque_artifacts["measurement"])

    def forbidden(*args, **kwargs):
        pytest.fail("required replay executed country model")

    monkeypatch.setattr(
        stage.policyengine_us.PolicyEngineUSEngine, "materialize", forbidden
    )
    warm = run_graph(compiled, **run_args, resume="require")
    assert all(node.store_hit for node in warm.nodes.values())
    assert cold.key == warm.key
    assert cold_bytes == store.load_bytes(
        warm.node(measure.id).opaque_artifacts["measurement"]
    )
    for entity in original.entities:
        pd.testing.assert_frame_equal(
            warm.population(create.id).table(entity), original.table(entity)
        )
    changed = document(assumption=False)
    changed["entries"]["amount"]["producer"] = "other.source"
    changed_policy = load(changed)
    params["leaf_policy"] = changed_policy
    changed_registry = KernelRegistry()
    changed_registry.register(InventedSource())
    changed_registry.register(
        stage.FiscalMeasurementKernel(
            model_outputs=params["model_outputs"], leaf_policy=changed_policy
        )
    )
    changed_graph = compile_graph(
        Graph("us", sources, (create, stage.fiscal_measurement_node(**params)))
    )
    with pytest.raises(
        (NodeRejectedError, StoreMissError), match="(?i)(require|cache|missing)"
    ):
        run_graph(
            changed_graph, **{**run_args, "kernels": changed_registry}, resume="require"
        )
