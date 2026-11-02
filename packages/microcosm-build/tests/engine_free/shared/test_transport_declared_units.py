"""CREATE aliases, inventories and population identity for declared partitions."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.cli import prepare_create_outputs
from microcosm.build.transport.column_kernels import TRANSPORT_UNIT_ATTRIBUTES
from microcosm.build.transport.compose import (
    skeleton_create_declaration,
    transport_rules_node,
    validate_transport_activation,
)
from microcosm.build.transport.population_kernels import (
    TRANSPORT_CREATE,
    TRANSPORT_CREATE_DECLARED,
    create_unit_declaration,
)
from microcosm.frame.unit_construction import (
    BenefitUnitRule,
    MultiUnitRule,
    TaxUnitRule,
    UnitDeclaration,
    benefit_unit_roles,
)
from microcosm.graph import KernelContext, Node, Owned, Slice
from test_support.microcosm_build.transport_composed import make_composed_fixture
from test_support.microcosm_build.transport_graph import canonical_text
from test_support.microcosm_build.transport_population import (
    UNIT_RULE,
    create_node,
    digest,
    population_graph,
    run_population,
    write_population_sources,
)
from test_support.paths import paths_for


def _run_create(sources, node, kernel):
    return kernel.run(
        KernelContext(
            node=node,
            tables={},
            weights={},
            strata=pd.Series(dtype=object),
            params=node.params,
            rng=np.random.default_rng(0),
            sources={"donor": sources.donor, "facts": sources.facts},
        )
    )


def _inventory(frame):
    schema = frame.schema
    structural = {
        "person": {
            "person_id",
            *(schema.membership_column(entity) for entity in schema.group_entities),
        },
        **{entity: {schema.id_column(entity)} for entity in schema.group_entities},
    }
    return tuple(
        Owned(entity, column, str(frame.table(entity)[column].dtype))
        for entity in schema.entities
        for column in frame.table(entity)
        if column not in structural[entity]
    )


def _versioned(node, declaration):
    document = declaration.to_dict()
    return replace(
        node,
        kernel=TRANSPORT_CREATE_DECLARED.ref,
        params={
            **node.params,
            "unit_rule": canonical_text(document),
            "unit_rule_sha256": digest(document),
        },
    )


@settings(
    max_examples=6,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(hours=st.integers(0, 60), mass=st.integers(1, 100_000))
def test_nz_create_alias_is_byte_identical_to_declared_create(tmp_path, hours, mass):
    def edit(tables):
        tables["person"]["weekly_hours_worked_before_lsr"] = 0
        tables["person"].loc[2, "age"] = 17
        tables["person"].loc[2, "weekly_hours_worked_before_lsr"] = hours

    sources = write_population_sources(
        tmp_path / "sources", mass=mass, edit_tables=edit
    )
    path = (
        paths_for("microcosm-build").package
        / "src/microcosm/build/nz/benefit_unit_rule.json"
    )
    document = json.loads(path.read_text())
    rule = BenefitUnitRule.from_spec(document)
    legacy = create_node(sources)
    legacy = replace(
        legacy,
        params={
            **legacy.params,
            "unit_rule": canonical_text(document),
            "unit_rule_sha256": digest(document),
        },
    )
    declared = _versioned(legacy, MultiUnitRule.from_benefit_unit(rule))
    first = _run_create(sources, legacy, TRANSPORT_CREATE)
    second = _run_create(sources, declared, TRANSPORT_CREATE_DECLARED)
    assert first.frame.schema == second.frame.schema
    assert first.receipt == second.receipt
    assert _inventory(first.frame) == _inventory(second.frame)
    for entity in first.frame.entities:
        before, after = first.frame.table(entity), second.frame.table(entity)
        pd.testing.assert_frame_equal(before, after, check_exact=True)
        assert (
            before.to_json(orient="table").encode()
            == after.to_json(orient="table").encode()
        )
    roles = []
    for frame in (first.frame, second.frame):
        roles.append(
            benefit_unit_roles(
                frame.person,
                frame.table("family"),
                frame.person[rule.membership_column],
                rule,
            )
            .to_json()
            .encode()
        )
    assert roles[0] == roles[1]
    assert (
        first.frame.resolve_weights("family").values.tobytes()
        == second.frame.resolve_weights("family").values.tobytes()
    )


def test_cli_create_versions_have_identical_cold_and_warm_inventories(tmp_path):
    from microcosm.graph import ContentStore

    fixture = make_composed_fixture(tmp_path / "fixture")
    spec = copy.deepcopy(fixture.spec)
    create = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row["kernel"] == TRANSPORT_CREATE.ref
    )
    create["kernel"] = TRANSPORT_CREATE_DECLARED.ref
    spec["resources"]["benefit_unit_rule"] = MultiUnitRule.from_benefit_unit(
        UNIT_RULE
    ).to_dict()
    assert skeleton_create_declaration(spec).kernel == TRANSPORT_CREATE_DECLARED.ref
    store = ContentStore(tmp_path / "inventory")
    legacy = prepare_create_outputs(fixture.spec, fixture.sources)
    cold = prepare_create_outputs(spec, fixture.sources, inventory_store=store)
    warm = prepare_create_outputs(
        spec, fixture.sources, inventory_store=store, resume="require"
    )
    assert legacy == cold == warm


@settings(
    max_examples=4,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(eligible=st.booleans())
def test_tax_partition_membership_changes_population_identity(tmp_path, eligible):
    # A synthetic boolean uses a carried concept to exercise CREATE plumbing;
    # this fixture makes no legal claim about disability or Belgian eligibility.
    def edit(tables):
        tables["person"]["is_disabled"] = eligible

    sources = write_population_sources(tmp_path / "sources", edit_tables=edit)
    declaration = MultiUnitRule(
        units=(
            UnitDeclaration(kind="benefit_unit", rule=UNIT_RULE),
            UnitDeclaration(
                kind="tax_unit",
                rule=TaxUnitRule(
                    entity="tax_unit",
                    pairing_eligibility_column="has_disability",
                    provenance={
                        "eligibility": "Synthetic explicit boolean for plumbing only."
                    },
                ),
            ),
        )
    )
    graph = population_graph(sources, through="boundary")
    node = _versioned(graph.nodes[0], declaration)
    frame = _run_create(sources, node, TRANSPORT_CREATE_DECLARED).frame
    node = replace(node, outputs=_inventory(frame))
    graph = replace(graph, nodes=(node,))
    first = run_population(tmp_path / "first", sources, graph=graph)
    old_ids = frame.person["person_tax_unit_id"].tolist()

    def opposite(tables):
        tables["person"]["is_disabled"] = not eligible

    changed_sources = write_population_sources(
        tmp_path / "changed", edit_tables=opposite
    )
    changed = _versioned(create_node(changed_sources), declaration)
    changed_frame = _run_create(
        changed_sources, changed, TRANSPORT_CREATE_DECLARED
    ).frame
    changed = replace(changed, outputs=_inventory(changed_frame))
    second = run_population(
        tmp_path / "second", changed_sources, graph=replace(graph, nodes=(changed,))
    )
    assert old_ids != changed_frame.person["person_tax_unit_id"].tolist()
    assert (
        first.manifest.nodes[node.id].frame_key
        != second.manifest.nodes[changed.id].frame_key
    )
    assert frame.weighted_entities == ("household",)
    assert "tax_unit" in frame.schema.group_entities


def test_create_parser_rejects_an_unregistered_version():
    with pytest.raises(ValueError, match="Unknown transport CREATE"):
        create_unit_declaration("transport.create@99", {})


def test_legacy_create_keeps_its_family_only_domain(tmp_path):
    """create@1 refuses what it always refused; create@2 carries the wider domain."""
    sources = write_population_sources(tmp_path / "sources", mass=1000)
    document = {**UNIT_RULE.to_dict(), "entity": "assessment"}
    legacy = create_node(sources)
    legacy = replace(
        legacy,
        params={
            **legacy.params,
            "unit_rule": canonical_text(document),
            "unit_rule_sha256": digest(document),
        },
    )
    with pytest.raises(ValueError, match="requires family benefit units"):
        create_unit_declaration(TRANSPORT_CREATE.ref, document)
    with pytest.raises(ValueError, match="requires family benefit units"):
        _run_create(sources, legacy, TRANSPORT_CREATE)
    declared = _versioned(
        legacy, MultiUnitRule.from_benefit_unit(BenefitUnitRule.from_spec(document))
    )
    result = _run_create(sources, declared, TRANSPORT_CREATE_DECLARED)
    assert "assessment" in result.frame.schema.group_entities


def test_multiple_unit_attributes_produce_explicit_person_role_columns(tmp_path):
    def edit(tables):
        tables["person"]["is_disabled"] = True

    sources = write_population_sources(tmp_path / "sources", edit_tables=edit)
    declaration = MultiUnitRule(
        units=(
            UnitDeclaration(kind="benefit_unit", rule=UNIT_RULE),
            UnitDeclaration(
                kind="tax_unit",
                rule=TaxUnitRule(
                    entity="tax_unit",
                    pairing_eligibility_column="has_disability",
                    provenance={
                        "eligibility": "Synthetic explicit boolean, no legal claim."
                    },
                ),
            ),
        )
    )
    frame = _run_create(
        sources,
        _versioned(create_node(sources), declaration),
        TRANSPORT_CREATE_DECLARED,
    ).frame
    document = declaration.to_dict()
    node = Node(
        "attributes",
        TRANSPORT_UNIT_ATTRIBUTES.ref,
        population="open",
        params={
            "unit_rule": canonical_text(document),
            "unit_rule_sha256": digest(document),
        },
        outputs=(
            Owned("person", "family_role", "string"),
            Owned("person", "tax_unit_role", "string"),
            Owned("family", "n_members", "int64"),
            Owned("tax_unit", "n_members", "int64"),
            Owned("household", "n_family_units", "int64"),
            Owned("household", "n_tax_unit_units", "int64"),
        ),
    )
    result = TRANSPORT_UNIT_ATTRIBUTES.run(
        KernelContext(
            node=node,
            tables={entity: frame.table(entity) for entity in frame.entities},
            weights={},
            strata=frame.strata,
            params=node.params,
            rng=np.random.default_rng(0),
        )
    )
    assert set(result.columns) == {(item.entity, item.column) for item in node.outputs}
    for item in declaration.units:
        expected = benefit_unit_roles(
            frame.person,
            frame.table(item.entity),
            frame.person[item.rule.membership_column],
            item.rule,
        )
        assert (
            result.columns["person", f"{item.entity}_role"].tolist()
            == expected.tolist()
        )
        assert result.columns[item.entity, "n_members"].sum() == frame.n("person")


@settings(
    max_examples=5,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    omitted=st.sampled_from(
        [
            ("person", "age"),
            ("person", "family_role"),
        ]
    )
)
def test_relation_inputs_must_be_explicit_graph_dependencies(tmp_path, omitted):
    fixture = make_composed_fixture(tmp_path / "fixture")
    spec = copy.deepcopy(fixture.spec)
    binding = next(
        row
        for row in spec["resources"]["axiom_rules_bindings"]["bindings"]
        if row["id"] == "family_housing"
    )
    binding["relations"] = [
        {
            "predicate": "toy_members",
            "key": "toy_members:0:1",
            "current_slot": 0,
            "related_slot": 1,
            "arguments": ["Family", "Person"],
            "root_entity": "family",
            "related_entity": "person",
            "membership_column": "person_family_id",
            "related_inputs": {"toy_age": "age"},
            "cardinality": "many",
            "period": binding["period"],
            "role_column": "family_role",
            "roles": ["head", "partner"],
        }
    ]
    dependencies = {("person", "age"), ("person", "family_role")}
    node = Node(
        "relations",
        "simulate.rules_by_ref@1",
        population="open",
        inputs=tuple(
            Slice(entity, (column,)) for entity, column in sorted(dependencies)
        ),
        outputs=tuple(
            Owned("family", name, "float64") for name in binding["variables"]
        ),
    )
    assert (
        transport_rules_node(spec, fixture.config, binding["id"], node).params["period"]
        == binding["period"]
    )
    restricted = replace(
        node,
        inputs=(
            replace(
                node.inputs[0],
                columns=(*node.inputs[0].columns, "selected"),
                rows="selected",
            ),
            *node.inputs[1:],
        ),
    )
    with pytest.raises(ValueError, match="whole-table relation inputs"):
        transport_rules_node(spec, fixture.config, binding["id"], restricted)
    missing = replace(
        node,
        inputs=tuple(
            Slice(entity, (column,))
            for entity, column in sorted(dependencies - {omitted})
        ),
    )
    with pytest.raises(ValueError, match="whole-table relation inputs"):
        transport_rules_node(spec, fixture.config, binding["id"], missing)


@pytest.mark.parametrize("document", [None, {}, "{}"])
def test_versioned_create_requires_a_versioned_canonical_document(tmp_path, document):
    fixture = make_composed_fixture(tmp_path / "fixture")
    spec = copy.deepcopy(fixture.spec)
    create = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row["kernel"] == TRANSPORT_CREATE.ref
    )
    create["kernel"] = TRANSPORT_CREATE_DECLARED.ref
    create["params"]["unit_rule"] = document
    with pytest.raises((ValueError, TypeError), match="unit|param|selection|format"):
        skeleton_create_declaration(spec)


def test_versioned_create_parser_requires_one_create_and_json_text(tmp_path):
    from microcosm.build.transport.compose import _versioned_create_units

    fixture = make_composed_fixture(tmp_path / "fixture")
    spec = copy.deepcopy(fixture.spec)
    rows = spec["resources"]["transport_graph"]["nodes"]
    create = next(row for row in rows if row["kernel"] == TRANSPORT_CREATE.ref)
    create["kernel"] = TRANSPORT_CREATE_DECLARED.ref
    create["params"]["unit_rule"] = None
    with pytest.raises(ValueError, match="canonical JSON unit_rule"):
        _versioned_create_units(spec)
    rows.append(copy.deepcopy(create))
    with pytest.raises(ValueError, match="exactly one CREATE"):
        _versioned_create_units(spec)


def test_legacy_unit_entities_and_ancestry_accept_an_absent_graph():
    from microcosm.build.transport.compose import (
        _unit_entities,
        _versioned_create_units,
        load_transport_spec,
        validate_transport_calibration_ancestry,
    )

    spec = load_transport_spec("nz")
    resources = spec["resources"]
    resources.pop("transport_graph", None)
    resources.pop("transport_graph.json", None)
    assert resources["precalibration_stages"]["stages"]
    assert _versioned_create_units(spec) is None
    assert _unit_entities(spec) == (resources["benefit_unit_rule"]["entity"],)
    validate_transport_calibration_ancestry(spec)


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"schema_version": 2, "country": "toy", "nodes": []},
        {"schema_version": 1, "country": "other", "nodes": []},
    ],
)
def test_legacy_unit_entities_validate_a_present_graph(document):
    from microcosm.build.transport.compose import _unit_entities

    spec = {
        "country": "toy",
        "resources": {
            "benefit_unit_rule": {"entity": "family"},
            "transport_graph": document,
        },
    }
    with pytest.raises(ValueError, match="schema_version 1 and the spec country"):
        _unit_entities(spec)


def test_skeleton_refuses_unregistered_create_version(tmp_path):
    fixture = make_composed_fixture(tmp_path / "fixture")
    spec = copy.deepcopy(fixture.spec)
    create = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row["kernel"] == TRANSPORT_CREATE.ref
    )
    create["kernel"] = "transport.create@99"
    with pytest.raises(ValueError, match="exactly one transport.create"):
        skeleton_create_declaration(spec)


@pytest.mark.parametrize("include_graph", [False, True], ids=["absent", "partial"])
def test_activation_aggregates_gaps_without_a_complete_graph(include_graph):
    resources = {
        "target_references": {"target_references": []},
        "precal_references": {"target_references": []},
        "holdout_references": {
            "target_references": [
                {
                    "name": "parked",
                    "metadata": {"activation_status": "requires_review"},
                }
            ]
        },
    }
    if include_graph:
        resources["transport_graph"] = {
            "nodes": [
                {
                    "params": {
                        "reference_document": {
                            "resource": "holdout_references",
                            "encoding": "json",
                        }
                    }
                }
            ]
        }
    with pytest.raises(ValueError, match="Transport spec is not activated:") as error:
        validate_transport_activation({"country": "toy", "resources": resources})
    message = str(error.value)
    assert "placeholder reference holdout_references.parked" in message
    for resource in ("axiom_rules_bindings", "scenarios", "benefit_unit_rule"):
        assert f"{resource}.json" in message
    if not include_graph:
        assert "transport_graph.json (the skeleton declaration)" in message
