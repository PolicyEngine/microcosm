"""Invented fiscal measures through the actual interpreter, CSR and graph store."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_fiscal_measurement as stage
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate import target as target_math
from microcosm.calibrate.hierarchy import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame import materialize as frame_materialize
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    NodeRejectedError,
    Owned,
    SourceRef,
    StoreMissError,
    StructuralDelta,
    compile_graph,
    load_source,
    run_graph,
    source_hash,
)
from microcosm.graph.canonical import canonical_json


def frame():
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.array([400, 100, 300, 200], dtype=np.int64),
                **{
                    f"person_{e}_id": np.array([30, 10, 20, 10], dtype=np.int64)
                    for e in US_SCHEMA.group_entities
                },
                "amount": [40.0, 10.0, 30.0, 20.0],
                "keep": [True, True, False, True],
            }
        ),
        **{
            e: pd.DataFrame({f"{e}_id": np.array([10, 20, 30], dtype=np.int64)})
            for e in US_SCHEMA.group_entities
        },
    }
    tables["household"]["state_fips"] = np.array([6, 6, 36], dtype=np.int64)
    tables["household"]["congressional_district_geoid"] = np.array(
        [601, 602, 3601], dtype=np.int64
    )
    tables["tax_unit"]["tax_unit_amount"] = [100.0, 200.0, 300.0]
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.DESIGN)},
    )


def declaration(*, value_variable="person_count", entity="person", model_outputs=()):
    definitions = (
        ("national", "0100000US", None, None),
        ("state", "0400000US06", "state_fips", 6),
        ("state", "0400000US36", "state_fips", 36),
        (
            "congressional_district",
            "5001900US0601",
            "congressional_district_geoid",
            601,
        ),
        (
            "congressional_district",
            "5001900US0602",
            "congressional_district_geoid",
            602,
        ),
        (
            "congressional_district",
            "5001900US3601",
            "congressional_district_geoid",
            3601,
        ),
    )
    scopes = tuple(
        stage.FiscalGeographyScope(HierarchyGeography(id_, id_, level), column, value)
        for level, id_, column, value in definitions
    )
    specs, bindings = [], {}
    for i, scope in enumerate(scopes):
        name = f"invented_{i}"
        specs.append(
            TargetSpec(
                name,
                entity,
                1.0,
                f"measure_{i}",
                period=2024,
                source="Invented test; not an administrative observation",
                family="invented",
                metadata={"contract_target_id": name},
                hierarchy=CalibrationHierarchy(
                    HierarchyNode("test", "Test"),
                    HierarchyCategory("count", "Count", "test"),
                    scope.geography,
                    (),
                    HierarchyNode(name, name),
                ),
            )
        )
        bindings[name] = {
            "bindings": {"policyengine": {"value_variable": value_variable}}
        }
    return {
        "registry": TargetRegistry(specs, country="us"),
        "population": "invented",
        "input_columns": {
            "person": ("amount", "keep"),
            "tax_unit": ("tax_unit_amount",),
            "household": ("state_fips", "congressional_district_geoid"),
        },
        "contract_targets": bindings,
        "advertised_scopes": scopes,
        "geography_vintage": "invented CD119 mapping",
        "period": 2024,
        "model_outputs": model_outputs,
    }


def context(arguments=None, value=None):
    value = frame() if value is None else value
    arguments = declaration() if arguments is None else arguments
    node = stage.fiscal_measurement_node(**arguments)
    tables = {}
    for slice_ in node.inputs:
        entity = slice_.entity
        structural = [f"{entity}_id"]
        if entity == "person":
            structural += [f"person_{e}_id" for e in US_SCHEMA.group_entities]
        tables[entity] = (
            value.table(entity)
            .loc[:, list(dict.fromkeys([*structural, *slice_.columns]))]
            .copy()
        )
    return KernelContext(
        node,
        tables,
        {"household": value.weights_for("household")},
        value.strata,
        node.params,
        np.random.default_rng(0),
    )


def test_actual_measurement_maps_person_counts_and_tax_units_to_households():
    original = frame()
    c = context(value=original)
    result = stage.FiscalMeasurementKernel().run(c)
    payload = result.artifacts["measurement"]
    measured = stage.verify_fiscal_measurement(
        payload, context=c, expected_sha256=hashlib.sha256(payload).hexdigest()
    )
    np.testing.assert_array_equal(measured.household_ids, [10, 20, 30])
    np.testing.assert_array_equal(
        measured.matrix.toarray(),
        [[2, 1, 1], [2, 1, 0], [0, 0, 1], [2, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    np.testing.assert_array_equal(
        measured.matrix.toarray()[3:].sum(axis=0), measured.matrix.toarray()[0]
    )
    assert result.frame is None and result.weights is None and not result.columns
    assert result.receipt["fit_performed"] is False
    assert result.receipt["release_eligible"] is False
    for entity in original.entities:
        pd.testing.assert_frame_equal(original.table(entity), frame().table(entity))
    c = context(declaration(value_variable="tax_unit_amount", entity="tax_unit"))
    values = stage.decode_fiscal_measurement(
        stage.FiscalMeasurementKernel().run(c).artifacts["measurement"]
    )
    np.testing.assert_array_equal(values.matrix.toarray()[0], [100, 200, 300])


def test_actual_shared_interpreter_sum_and_filter():
    arguments = declaration(value_variable="amount")
    for target in arguments["contract_targets"].values():
        target["bindings"]["policyengine"] = {
            "value_expression": "amount + amount",
            "filters": [{"variable": "amount", "operator": ">", "value": 5}],
        }
    measured = stage.decode_fiscal_measurement(
        stage.FiscalMeasurementKernel().run(context(arguments)).artifacts["measurement"]
    )
    np.testing.assert_array_equal(measured.matrix.toarray()[0], [60, 60, 80])


def test_boolean_indicator_expression_is_an_arithmetic_sum():
    arguments = declaration()
    for target in arguments["contract_targets"].values():
        target["bindings"]["policyengine"] = {"value_expression": "keep + keep"}
    value = frame()
    value.person["keep"] = [True, False, True, True]
    measured = stage.decode_fiscal_measurement(
        stage.FiscalMeasurementKernel()
        .run(context(arguments, value))
        .artifacts["measurement"]
    )
    np.testing.assert_array_equal(measured.matrix.toarray()[0], [2, 2, 2])


@pytest.mark.parametrize("kind", ["float", "nullable_boolean"])
@pytest.mark.parametrize("location", ["binding", "target_filter"])
def test_missing_consumed_predicate_is_refused(kind, location):
    arguments, value = declaration(), frame()
    variable = "amount" if kind == "float" else "keep"
    if kind == "float":
        value.person.loc[1, variable] = np.nan
        predicate = {"variable": variable, "operator": ">", "value": 5}
    else:
        value.person[variable] = pd.Series([True, pd.NA, True, True], dtype="boolean")
        predicate = {"variable": variable, "equals": True}
    if location == "binding":
        for binding in arguments["contract_targets"].values():
            binding["bindings"]["policyengine"]["filters"] = [predicate]
    else:
        arguments["registry"] = TargetRegistry(
            [replace(s, filter=variable) for s in arguments["registry"]], country="us"
        )
    with pytest.raises(
        ValueError, match="MISSING_PREDICATE_INPUT:person\\." + variable
    ):
        stage.FiscalMeasurementKernel().run(context(arguments, value))


@pytest.mark.parametrize("location", ["binding", "target_filter"])
def test_known_false_predicates_and_unconsumed_missing_inputs_are_preserved(location):
    arguments, value = declaration(), frame()
    value.person["keep"] = [True, False, True, True]
    value.person.loc[0, "amount"] = np.nan  # This measure does not read amount.
    if location == "binding":
        for binding in arguments["contract_targets"].values():
            binding["bindings"]["policyengine"]["filters"] = [
                {"variable": "keep", "equals": True}
            ]
    else:
        arguments["registry"] = TargetRegistry(
            [replace(s, filter="keep") for s in arguments["registry"]], country="us"
        )
    measured = stage.decode_fiscal_measurement(
        stage.FiscalMeasurementKernel()
        .run(context(arguments, value))
        .artifacts["measurement"]
    )
    np.testing.assert_array_equal(measured.matrix.toarray()[0], [1, 1, 1])


def test_implementation_identity_includes_target_arithmetic_source(
    tmp_path, monkeypatch
):
    before = stage.FiscalMeasurementKernel().implementation_hash()
    changed = tmp_path / "changed_target.py"
    source = Path(target_math.__file__).read_text()
    original = "        return values * filter_mask\n"
    assert source.count(original) == 1
    changed.write_text(
        source.replace(original, "        return 2.0 * values * filter_mask\n")
    )
    monkeypatch.setattr(target_math, "__file__", str(changed))
    assert stage.FiscalMeasurementKernel().implementation_hash() != before


def test_implementation_identity_includes_engine_table_transform(tmp_path, monkeypatch):
    assert stage.policyengine_us.engine_tables is frame_materialize.engine_tables
    before = stage.FiscalMeasurementKernel().implementation_hash()
    changed = tmp_path / "changed_engine_tables.py"
    source = Path(frame_materialize.__file__).read_text()
    original = "        ).values\n"
    assert source.count(original) == 1
    changed.write_text(source.replace(original, "        ).values * 2.0\n"))
    monkeypatch.setattr(frame_materialize, "__file__", str(changed))
    assert stage.FiscalMeasurementKernel().implementation_hash() != before


@pytest.mark.parametrize(
    "change,reason",
    [
        ("binding", "UNSUPPORTED_BINDING"),
        ("coverage", "ADVERTISED_LEVELS"),
        ("period", "TARGET_PERIOD_HIERARCHY"),
        ("collision", "INPUT_MEASURE_COLLISION"),
        ("unknown", "UNMATERIALIZED_TARGETS"),
        ("empty_area", "EMPTY_ADVERTISED_GEOGRAPHY"),
        ("zero_support", "UNSUPPORTED_TARGET"),
        ("zero_weight", "UNSUPPORTED_TARGET"),
    ],
)
def test_declared_surface_and_advertised_support_fail_closed(change, reason):
    arguments, value = declaration(), frame()
    if change == "binding":
        arguments["contract_targets"]["invented_0"]["bindings"]["policyengine"][
            "kind"
        ] = "input_substitution_counterfactual"
    elif change == "coverage":
        arguments["advertised_scopes"] = arguments["advertised_scopes"][:3]
    elif change == "period":
        arguments["period"] = 2025
    elif change == "collision":
        arguments["input_columns"]["person"] += ("measure_0",)
    elif change == "unknown":
        arguments["contract_targets"]["invented_0"]["bindings"]["policyengine"][
            "value_variable"
        ] = "not_produced"
    elif change == "empty_area":
        value.table("household").loc[2, "congressional_district_geoid"] = 3699
    elif change == "zero_support":
        arguments["contract_targets"]["invented_5"]["bindings"]["policyengine"][
            "filters"
        ] = [{"variable": "amount", "operator": ">", "value": 999}]
    elif change == "zero_weight":
        value = Frame(
            {e: value.table(e) for e in value.entities},
            US_SCHEMA,
            {"household": Weights(np.array([1.0, 2.0, 0.0]), WeightKind.DESIGN)},
        )
    with pytest.raises(ValueError, match=reason):
        stage.FiscalMeasurementKernel().run(context(arguments, value))


def test_artifact_digest_projection_and_order_tamper_refusals():
    c = context()
    payload = stage.FiscalMeasurementKernel().run(c).artifacts["measurement"]
    digest = hashlib.sha256(payload).hexdigest()
    changed = json.loads(payload)
    changed["data"]["hex"] = np.full(9, 99.0, dtype="<f8").tobytes().hex()
    with pytest.raises(ValueError, match="ARTIFACT_DIGEST"):
        stage.verify_fiscal_measurement(
            canonical_json(changed), context=c, expected_sha256=digest
        )
    changed_context = replace(
        c, tables={**c.tables, "household": c.tables["household"].iloc[::-1]}
    )
    with pytest.raises(ValueError, match="PROJECTION_BINDING"):
        stage.verify_fiscal_measurement(
            payload, context=changed_context, expected_sha256=digest
        )
    different = declaration()
    different["registry"] = TargetRegistry(
        [replace(s, value=2.0) for s in different["registry"]], country="us"
    )
    with pytest.raises(ValueError, match="NODE_BINDING"):
        stage.verify_fiscal_measurement(
            payload, context=context(different), expected_sha256=digest
        )
    changed = json.loads(payload)
    changed["indices"]["hex"] = np.full(9, 999, dtype="<i8").tobytes().hex()
    with pytest.raises(ValueError, match="CSR"):
        stage.decode_fiscal_measurement(canonical_json(changed))


def test_complete_uint64_identity_and_support_metadata_are_bound():
    value = frame()
    ids = np.array([2**63, 2**63 + 1, 2**64 - 1], dtype=np.uint64)
    value.table("household")["household_id"] = ids
    value.table("person")["person_household_id"] = ids[[2, 0, 1, 0]]
    value.revalidate()
    c = context(value=value)
    payload = stage.FiscalMeasurementKernel().run(c).artifacts["measurement"]
    observed = stage.verify_fiscal_measurement(
        payload, context=c, expected_sha256=hashlib.sha256(payload).hexdigest()
    )
    np.testing.assert_array_equal(observed.household_ids, ids)
    assert observed.household_ids.dtype == ids.dtype
    forged = json.loads(payload)
    forged["support"][0]["positive_weight_positive_rows"] = 0
    raw = canonical_json(forged)
    with pytest.raises(ValueError, match="SUPPORT_BINDING"):
        stage.verify_fiscal_measurement(
            raw, context=c, expected_sha256=hashlib.sha256(raw).hexdigest()
        )
    other_parent = replace(c, node=replace(c.node, population="different_parent"))
    with pytest.raises(ValueError, match="NODE_BINDING"):
        stage.verify_fiscal_measurement(
            payload,
            context=other_parent,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_common_measure_reuse_and_geography_consistency():
    arguments = declaration()
    arguments["registry"] = TargetRegistry(
        [replace(s, measure="common_count") for s in arguments["registry"]],
        country="us",
    )
    stage.FiscalMeasurementKernel().run(context(arguments))
    arguments["contract_targets"]["invented_1"]["bindings"]["policyengine"][
        "value_variable"
    ] = "amount"
    with pytest.raises(ValueError, match="CONFLICTING_MEASUREMENT_BINDINGS"):
        stage.fiscal_measurement_node(**arguments)
    value = frame()
    value.table("household").loc[0, "state_fips"] = 36
    with pytest.raises(ValueError, match="STATE_CD_MAPPING"):
        stage.FiscalMeasurementKernel().run(context(value=value))


class InventedSource(KernelBase):
    ref = "test.fiscal_measurement_source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources["fixture"])
        )

    def implementation_hash(self):
        return source_hash(type(self), load_source)


def test_real_graph_cold_required_replay_and_changed_target_key(tmp_path, monkeypatch):
    original = frame()
    create = Node(
        "invented",
        InventedSource.ref,
        sources=("fixture",),
        structural=StructuralDelta.CREATE,
        outputs=(
            Owned("person", "amount", "float64"),
            Owned("person", "keep", "bool"),
            Owned("tax_unit", "tax_unit_amount", "float64"),
            Owned("household", "state_fips", "int64"),
            Owned("household", "congressional_district_geoid", "int64"),
        ),
    )
    measure = stage.fiscal_measurement_node(**declaration())
    compiled = compile_graph(
        Graph("us", (SourceRef("fixture", "frame-store"),), (create, measure))
    )
    kernels = KernelRegistry()
    kernels.register(InventedSource())
    kernels.register(stage.FiscalMeasurementKernel())
    store = ContentStore(tmp_path / "store")
    path = store.put_frame(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), original
    )
    arguments = {"sources": {"fixture": path}, "store": store, "kernels": kernels}
    cold = run_graph(compiled, **arguments)
    warm = run_graph(compiled, **arguments, resume="require")
    assert cold.key == warm.key and all(n.store_hit for n in warm.nodes.values())
    measured = stage.decode_fiscal_measurement(
        store.load_bytes(warm.node(measure.id).opaque_artifacts["measurement"])
    )
    np.testing.assert_array_equal(
        measured.household_ids, original.table("household").household_id
    )
    for entity in original.entities:
        pd.testing.assert_frame_equal(
            warm.population(create.id).table(entity), original.table(entity)
        )
    changed = declaration()
    changed["registry"] = TargetRegistry(
        [replace(s, value=2.0) for s in changed["registry"]], country="us"
    )
    changed_graph = compile_graph(
        Graph(
            "us",
            (SourceRef("fixture", "frame-store"),),
            (create, stage.fiscal_measurement_node(**changed)),
        )
    )
    with pytest.raises(
        (NodeRejectedError, StoreMissError), match="(?i)(require|cache|missing)"
    ):
        run_graph(changed_graph, **arguments, resume="require")
    changed_frame = frame()
    changed_frame.table("person")["amount"] += 1.0
    changed_path = store.put_frame(
        hashlib.sha256(b"different invented source").hexdigest(), changed_frame
    )
    with pytest.raises(
        (NodeRejectedError, StoreMissError), match="(?i)(require|cache|missing)"
    ):
        run_graph(
            compiled,
            **{**arguments, "sources": {"fixture": changed_path}},
            resume="require",
        )
    monkeypatch.setattr(
        stage,
        "_model_contract",
        lambda roots: {"evaluation": "changed_runtime", "roots": []},
    )
    with pytest.raises(
        (NodeRejectedError, StoreMissError), match="(?i)(require|cache|missing)"
    ):
        run_graph(compiled, **arguments, resume="require")


@pytest.mark.requires_us
def test_actual_model_closure_is_required_before_engine_evaluation(monkeypatch):
    arguments = declaration(
        value_variable="dividend_income", model_outputs=("dividend_income",)
    )
    with pytest.raises(ValueError, match="UNDECLARED_MODEL_LEAF"):
        stage.fiscal_measurement_node(**arguments)
    arguments["input_columns"]["person"] = (
        "qualified_dividend_income",
        "non_qualified_dividend_income",
    )
    arguments["input_columns"].pop("tax_unit")
    value = frame()
    value.table("person")["qualified_dividend_income"] = [4.0, 1.0, 3.0, 2.0]
    value.table("person")["non_qualified_dividend_income"] = [40.0, 10.0, 30.0, 20.0]
    c = context(arguments, value)
    measured = stage.decode_fiscal_measurement(
        stage.FiscalMeasurementKernel(model_outputs=("dividend_income",))
        .run(c)
        .artifacts["measurement"]
    )
    np.testing.assert_array_equal(measured.matrix.toarray()[0], [33, 33, 44])
    assert (
        measured.document["declaration"]["model_contract"]
        != '{"evaluation":"none","roots":[]}'
    )
    assert "dividend_income" not in value.table("person")
    value.table("person").loc[0, "qualified_dividend_income"] = float("nan")
    missing = context(arguments, value)

    def forbidden(*args, **kwargs):
        pytest.fail("The engine ran with an incomplete input closure")

    monkeypatch.setattr(
        stage.policyengine_us.PolicyEngineUSEngine, "materialize", forbidden
    )
    with pytest.raises(ValueError, match="MISSING_MODEL_INPUT"):
        stage.FiscalMeasurementKernel(model_outputs=("dividend_income",)).run(missing)
