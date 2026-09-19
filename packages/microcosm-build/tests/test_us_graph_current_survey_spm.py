"""Pure invented SPM fragment checks; no source issuance, engines or PUF fits."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_survey_spm_projection import fixture as projection_fixture

from microcosm.build.spm_input_contract import ROLE_INPUT, UNIVERSE_INPUT
from microcosm.build.us_runtime import graph_current_survey_spm as graph
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import ArtifactInput, ArtifactType, ArtifactValue, NumericScope


def frame(view):
    people = view.person.copy(deep=True)
    tables = {"person": people}
    for entity in US_SCHEMA.group_entities:
        if hasattr(view, entity):
            tables[entity] = getattr(view, entity).copy(deep=True)
        else:
            people[US_SCHEMA.membership_column(entity)] = people.person_id
            tables[entity] = pd.DataFrame({entity + "_id": people.person_id})
        tables[entity] = (
            tables[entity].sort_values(entity + "_id").reset_index(drop=True)
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(len(tables["household"])), WeightKind.DESIGN)},
    )


@pytest.fixture
def case():
    original, origins, roles, statuses, receiving = projection_fixture()
    qualified = graph.source.QualifiedNativeSpmInputs(
        frame(original),
        origins,
        roles,
        statuses,
        statuses.rename("status").to_frame(),
        pd.DataFrame({"literal": ["01"]}, index=pd.Index([20], name="person_id")),
        graph.source.source._encode(
            {
                "income_year": 2024,
                "acs_survey_year": 2024,
                "asec_survey_year": 2025,
                "acs_profile": {"fixture": "invented; no source admission"},
                "asec_scope_policy": None,
            }
        ),
    )
    receiving.person["unrelated_value"] = 17.25
    receiving = frame(receiving)
    after = ArtifactInput(
        "hours_attachment",
        "invented.hours",
        "attachment",
        ArtifactType("test.hours", 1),
    )
    nodes = graph.spm_nodes(
        qualified,
        receiving,
        receiving_version="invented.receiving",
        after=after,
        outside_role_placeholder=False,
    )
    return SimpleNamespace(
        qualified=qualified, receiving=receiving, after=after, nodes=nodes
    )


def artifacts(case, node, produced):
    return {
        edge.name: ArtifactValue(
            produced[edge.producer, edge.artifact],
            edge.type,
            "b" * 64,
            "a" * 64,
            NumericScope(),
        )
        for edge in node.artifact_inputs
    }


def results(case):
    produced = {(case.after.producer, case.after.artifact): b"invented hours"}
    values = []
    for node in case.nodes:
        inputs = artifacts(case, node, produced)
        result = graph.spm_result(case.qualified, node, inputs, case.receiving)
        produced.update(
            {(node.id, key): value for key, value in result.artifacts.items()}
        )
        values.append((node, inputs, result))
    return values


def test_three_nodes_own_only_person_role_and_unit_scope(case):
    assert [n.id for n in case.nodes] == [
        graph.SOURCE_NODE,
        graph.PROJECT_NODE,
        graph.ATTACH_NODE,
    ]
    assert all(n.population == "invented.receiving" for n in case.nodes)
    assert not case.nodes[0].outputs and not case.nodes[1].outputs
    assert {(o.entity, o.column, o.dtype) for o in case.nodes[2].outputs} == {
        ("person", ROLE_INPUT, "bool"),
        ("spm_unit", UNIVERSE_INPUT, "string"),
    }
    for node in case.nodes[1:]:
        assert {s.entity for s in node.inputs} == {"person", "spm_unit", "household"}
        assert all(len(s.columns) == 4 for s in node.inputs)
        assert all("unrelated_value" not in s.columns for s in node.inputs)
    assert all(2024 == n.params["income_year"] for n in case.nodes)


def test_mixed_grain_artifacts_and_patch_preserve_complete_population(case):
    before = graph.source.source._frame_identity(case.receiving)
    computed = results(case)
    mapped = json.loads(computed[1][2].artifacts["projection"])
    assert set(mapped["columns"]) == {"person", "spm_unit"}
    assert len(mapped["columns"]["person"]["data"]) == 10
    assert len(mapped["columns"]["spm_unit"]["data"]) == 6
    assert mapped["placeholder_person_ids"] == [20030, 10030]
    source = json.loads(computed[0][2].artifacts["source"])
    assert source["roles"]["data"][-1]["role"] is None
    expected = graph.projection.project_spm_inputs(
        case.qualified.source_frame,
        case.qualified.origins,
        case.qualified.roles,
        case.qualified.unit_status,
        case.receiving,
        source_year=2024,
        year=2024,
        outside_role_placeholder=False,
    )
    attached = computed[-1][2]
    for key, values in expected.columns.items():
        pd.testing.assert_series_equal(attached.columns[key], values)
    population = graph.population_ops.Population.from_frame(
        case.receiving, "invented.receiving"
    )
    for node, _, result in computed:
        population = graph.population_ops.patch(population, node, result)
    for entity in case.receiving.entities:
        pd.testing.assert_frame_equal(
            population.frame.table(entity)[case.receiving.table(entity).columns],
            case.receiving.table(entity),
            check_exact=True,
        )
    assert graph.source.source._frame_identity(case.receiving) == before
    assert not attached.receipt["release_eligible"]
    assert not attached.receipt["weights_changed"]


@pytest.mark.parametrize("alias", ["spm_source", "spm_projection"])
def test_attachment_refuses_stale_or_foreign_payload(case, alias):
    node, inputs, _ = results(case)[-1]
    inputs[alias] = replace(inputs[alias], payload=b"{}")
    with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD"):
        graph.spm_result(case.qualified, node, inputs, case.receiving)


def test_artifact_roster_and_nominal_types_are_checked(case):
    node, inputs, _ = results(case)[-1]
    with pytest.raises(ValueError, match="ARTIFACT_ROSTER"):
        graph.spm_result(case.qualified, node, {}, case.receiving)
    inputs["spm_source"] = replace(
        inputs["spm_source"], type=ArtifactType("test.foreign", 1)
    )
    with pytest.raises(ValueError, match="ARTIFACT_TYPE"):
        graph.spm_result(case.qualified, node, inputs, case.receiving)


@pytest.mark.parametrize("entity", ["spm_unit", "household"])
def test_membership_changes_refuse_before_attachment(case, entity):
    node, inputs, _ = results(case)[-1]
    people = case.receiving.person
    link = "person_" + entity + "_id"
    people.loc[people.index[0], link] = people[link].iloc[1]
    with pytest.raises(ValueError, match="GROUP_MEMBERSHIP"):
        graph.spm_result(case.qualified, node, inputs, case.receiving)


@pytest.mark.parametrize("name", [ROLE_INPUT, UNIVERSE_INPUT])
def test_existing_columns_refuse_without_rewriting(case, name):
    entity = "person" if name == ROLE_INPUT else "spm_unit"
    case.receiving.table(entity)[name] = False if entity == "person" else "OUTSIDE"
    with pytest.raises(ValueError, match="OWNERSHIP_COLLISION"):
        graph.spm_nodes(
            case.qualified,
            case.receiving,
            receiving_version="invented.receiving",
            after=case.after,
            outside_role_placeholder=False,
        )


@pytest.mark.parametrize(
    "parameter,value", [("income_year", 2025), ("outside_role_placeholder", 0)]
)
def test_year_and_explicit_representation_are_bound(case, parameter, value):
    node, inputs, _ = results(case)[-1]
    node = replace(node, params={**node.params, parameter: value})
    with pytest.raises(ValueError, match="NODE_PARAMETERS"):
        graph.spm_result(case.qualified, node, inputs, case.receiving)


def test_policy_receipt_and_each_source_grain_are_sealed(case):
    node, inputs, _ = results(case)[-1]
    original = graph.spm_seal(case.qualified)
    case.qualified.unit_status.iloc[0] = "OUTSIDE"
    assert graph.spm_seal(case.qualified) != original
    object.__setattr__(case.qualified, "receipt", b"{}")
    with pytest.raises(ValueError, match="SOURCE_PERIOD|NODE_PARAMETERS"):
        graph.spm_result(case.qualified, node, inputs, case.receiving)


def test_unissued_values_cannot_create_production_kernels(case):
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        graph.spm_kernels(
            case.qualified,
            case.receiving,
            receiving_version="invented.receiving",
            after=case.after,
            outside_role_placeholder=False,
            require_current=lambda: None,
            require_context=lambda context: None,
        )


@pytest.mark.parametrize(
    "changed", ["person", "spm_unit", "artifact", "receipt", "mutable_artifact"]
)
def test_final_validation_detects_output_mutation(case, changed):
    # This is the pure final-output fence used by the kernel. No source-owner
    # validation or issuance is replaced to exercise it.
    result = results(case)[-1][2]

    def validate():
        if changed == "person":
            values = result.columns["person", ROLE_INPUT]
            values.iloc[0] = not values.iloc[0]
        elif changed == "spm_unit":
            result.columns["spm_unit", UNIVERSE_INPUT].iloc[0] = "OUTSIDE"
        elif changed == "artifact":
            result.artifacts["attachment"] = b"changed"
        elif changed == "mutable_artifact":
            result.artifacts["attachment"] = bytearray(result.artifacts["attachment"])
        else:
            result.receipt["release_eligible"] = True

    with pytest.raises(ValueError, match="FINAL_RESULT_CHANGED"):
        graph._validate_result(result, validate)


def test_final_validation_accepts_unchanged_result_after_callback(case):
    result = results(case)[-1][2]
    calls = []
    assert graph._validate_result(result, lambda: calls.append("validated")) is result
    assert calls == ["validated"]


def test_receiving_context_preserves_all_three_declared_entity_tables(case):
    node, inputs, _ = results(case)[-1]
    context = SimpleNamespace(
        node=node,
        artifacts=inputs,
        tables={
            e: case.receiving.table(e) for e in ("person", "spm_unit", "household")
        },
    )
    calls = []
    view = graph._receiving_context(
        context, case.nodes, node.kernel, lambda value: calls.append(value)
    )
    assert calls == [context]
    assert all(getattr(view, e) is table for e, table in context.tables.items())


def test_host_callback_authenticates_producer_keys_before_result(case):
    def require_context(context):
        assert all(v.producer_key == "a" * 64 for v in context.artifacts.values()), (
            "FOREIGN_PRODUCER"
        )

    node, inputs, _ = results(case)[-1]
    inputs["spm_source"] = replace(inputs["spm_source"], producer_key="c" * 64)
    with pytest.raises(AssertionError, match="FOREIGN_PRODUCER"):
        graph._receiving_context(
            SimpleNamespace(node=node, artifacts=inputs, tables={}),
            case.nodes,
            node.kernel,
            require_context,
        )


@pytest.mark.parametrize(
    "name", ["ROLE_INPUT", "UNIVERSE_INPUT", "PROTOCOL", "SOURCE_NODE"]
)
def test_graph_global_declarations_are_part_of_retained_seal(case, monkeypatch, name):
    before = graph.spm_seal(case.qualified)
    monkeypatch.setattr(graph, name, "changed")
    assert graph.spm_seal(case.qualified) != before


@pytest.mark.parametrize("name", ["source", "projection", "population_ops"])
def test_imported_module_aliases_are_part_of_retained_seal(case, monkeypatch, name):
    before = graph.spm_seal(case.qualified)
    proxy = SimpleNamespace(**vars(getattr(graph, name)))
    if name == "projection":
        proxy.project_spm_inputs = lambda *args, **kwargs: None
    monkeypatch.setattr(graph, name, proxy)
    assert graph.spm_seal(case.qualified) != before


@pytest.mark.parametrize("name", ["SOURCE_TYPE", "PROJECTION_TYPE", "ATTACHMENT_TYPE"])
def test_artifact_type_seal_is_detached_from_in_place_mutation(case, name):
    before = graph.spm_seal(case.qualified)
    artifact_type = getattr(graph, name)
    version = artifact_type.schema_version
    try:
        object.__setattr__(artifact_type, "schema_version", version + 1)
        assert graph.spm_seal(case.qualified) != before
    finally:
        object.__setattr__(artifact_type, "schema_version", version)
    assert graph.spm_seal(case.qualified) == before


@pytest.mark.parametrize("name", ["SOURCE_TYPE", "PROJECTION_TYPE", "ATTACHMENT_TYPE"])
def test_artifact_type_seal_refuses_boolean_schema_version(case, name):
    artifact_type = getattr(graph, name)
    version = artifact_type.schema_version
    try:
        object.__setattr__(artifact_type, "schema_version", True)
        with pytest.raises(ValueError, match="ARTIFACT_TYPE_CONFIGURATION"):
            graph.spm_seal(case.qualified)
    finally:
        object.__setattr__(artifact_type, "schema_version", version)
