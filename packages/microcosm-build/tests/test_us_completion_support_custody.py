"""Small graph/Frame custody controls, without source or run issuance.

The private boundary shells below contain real declarations and compiled
versions solely to test the custody helpers. They cannot pass pure(), acquire
preparation authority, or issue a financial handle. Actual-owner cold/required
and permanent-revocation controls remain in test_us_graph_atomic_completion_host.
"""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from test_us_child_property_income_graph import (
    ORDER_TYPE,
    _options,
    _ordering,
    _qualified,
    _receiving_frame,
)

from microcosm.build.us_runtime import graph_atomic_survey_financial as host
from microcosm.build.us_runtime import graph_child_property_income as child
from microcosm.build.us_runtime import graph_survey_completion_host as extension
from microcosm.frame import (
    US_GROUP_ENTITIES,
    US_SCHEMA,
    EntitySchema,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)
from microcosm.graph import (
    ArtifactOutput,
    Graph,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
)
from microcosm.graph.manifest import PopulationView, RunManifest
from microcosm.graph.population import MassRecord, Population, token_for_dtype


def _full_us_parent_frame(original):
    """Extend only this invented parent; private child support stays unchanged."""
    tables_before = {
        entity: original.table(entity).copy(deep=True) for entity in original.entities
    }
    weights = {
        entity: original.weights_for(entity) for entity in original.weighted_entities
    }
    weight_bytes = {
        entity: weight.values.tobytes() for entity, weight in weights.items()
    }
    strata_before = original.strata.copy(deep=True)
    metadata_before, mass_log_before = original.metadata, original.mass_log
    nullable_before = {
        (entity, column): (series.array._data.tobytes(), series.array._mask.tobytes())
        for entity in original.entities
        for column, series in original.table(entity).items()
        if isinstance(series.dtype, pd.BooleanDtype)
    }
    assert set(original.schema.group_entities) <= set(US_GROUP_ENTITIES)
    assert original.schema.person_entity == US_SCHEMA.person_entity
    assert not original.links and not original._link_tables
    missing = tuple(
        group
        for group in US_GROUP_ENTITIES
        if group not in original.schema.group_entities
    )
    tables = {entity: table.copy(deep=True) for entity, table in tables_before.items()}
    people = tables[US_SCHEMA.person_entity]
    household_membership = US_SCHEMA.membership_column("household")
    household_ids = original.table("household")[US_SCHEMA.entity_id_column("household")]
    sentinels = {}
    for group in missing:
        membership = US_SCHEMA.membership_column(group)
        assert membership not in people.columns
        # One invented group per existing cloned household, without assigning
        # actual SPM/family/marital roles or changing any original identifier.
        people[membership] = original.person[household_membership].copy(deep=True)
        # The actual child graph requires a nonempty Slice for each entity.
        # These invented cells carry no survey or policy meaning.
        sentinel_column = f"invented_{group}_custody_sentinel"
        assert all(
            sentinel_column not in table.columns for table in tables_before.values()
        )
        sentinels[group] = pd.Series(
            1, index=household_ids.index, dtype="int64", name=sentinel_column
        )
        tables[group] = pd.DataFrame(
            {
                US_SCHEMA.entity_id_column(group): household_ids.copy(deep=True),
                sentinel_column: sentinels[group].copy(deep=True),
            }
        )
    full = Frame(
        tables,
        US_SCHEMA,
        weights,
        strata=original.strata,
        mass_log=original.mass_log,
        # Generic fixture metadata is not a country-authority receipt. Admit an
        # empty context only on this ordinary US parent; retain the input below.
        metadata={},
    )

    assert full.schema is US_SCHEMA and full.entities == US_SCHEMA.entities
    assert set(full._tables) == set(full.entities)
    assert not full.links and not full._link_tables
    assert full.weighted_entities == original.weighted_entities
    assert set(full._weights) == set(weights)
    for entity, before in tables_before.items():
        pd.testing.assert_frame_equal(original.table(entity), before, check_exact=True)
        pd.testing.assert_frame_equal(
            full.table(entity).loc[:, before.columns], before, check_exact=True
        )
    for (entity, column), backing in nullable_before.items():
        for frame in (original, full):
            array = frame.table(entity)[column].array
            assert (array._data.tobytes(), array._mask.tobytes()) == backing
    assert list(full.person.columns) == [
        *original.person.columns,
        *(US_SCHEMA.membership_column(group) for group in missing),
    ]
    for group in missing:
        id_column = US_SCHEMA.entity_id_column(group)
        sentinel = sentinels[group]
        assert sentinel.dtype == "int64" and sentinel.eq(1).all()
        assert list(full.table(group).columns) == [id_column, sentinel.name]
        for table in (tables[group], full.table(group)):
            pd.testing.assert_series_equal(
                table[sentinel.name], sentinel, check_exact=True
            )
        pd.testing.assert_series_equal(
            full.table(group)[id_column],
            household_ids.rename(id_column),
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            full.person[US_SCHEMA.membership_column(group)],
            original.person[household_membership].rename(
                US_SCHEMA.membership_column(group)
            ),
            check_exact=True,
        )
    for entity, weight in weights.items():
        assert full.weights_for(entity) is original.weights_for(entity) is weight
        assert weight.values.tobytes() == weight_bytes[entity]
        assert full.weights_for(entity).kind is weight.kind
    for frame in (original, full):
        pd.testing.assert_series_equal(frame.strata, strata_before, check_exact=True)
        assert frame.mass_log == mass_log_before
    assert len(full.metadata) == 0
    assert original.metadata == metadata_before
    assert original.metadata is metadata_before and original.mass_log is mass_log_before
    return full


def _case():
    qualified, origins = _qualified()
    parent = Population.from_frame(
        _full_us_parent_frame(_receiving_frame(qualified, origins)), "test.allocated"
    )
    edge, _, pins = _ordering()
    nodes = child.child_property_nodes(
        qualified,
        origins,
        parent,
        options=_options(),
        host_edges=(edge,),
        host_pins=pins,
    )
    frame = parent.frame
    create = Node(
        "test.parent",
        "test.child.receiving@1",
        sources=(child.SOURCE_NAME,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(entity, column, token_for_dtype(frame.table(entity)[column].dtype))
            for entity in frame.entities
            for column in frame.table(entity)
            if column != frame.schema.entity_id_column(entity)
            and not (
                entity == frame.schema.person_entity
                and column
                in {
                    frame.schema.membership_column(group)
                    for group in frame.schema.group_entities
                }
            )
        ),
        artifact_outputs=(ArtifactOutput("ordering", ORDER_TYPE),),
    )
    allocated = Node(
        parent.version,
        "test.child.allocate@1",
        base=create.id,
        structural=StructuralDelta.REWEIGHT,
        weights=WeightTransition("household", "importance", "conserve"),
        mass="conserve",
    )
    compiled = compile_graph(
        Graph(
            "us",
            (SourceRef(child.SOURCE_NAME, "frame-store"),),
            (create, allocated, *nodes),
        )
    )
    donor, recipient, *_ = child._projections(qualified, origins)
    supports = {
        node_id: Population.from_frame(child._support_frame(table), node_id)
        for node_id, table in ((child.DONOR, donor), (child.RECIPIENT, recipient))
    }
    # Descriptive shells exercise dispatch only; no monkeypatched owner methods.
    boundary = object.__new__(extension._CompletionHost)
    boundary.child = object.__new__(child.ChildPropertyBoundary)
    boundary.child.nodes = nodes
    boundary.child.declaration = tuple(node.normative() for node in nodes)
    boundary.child.revoked = boundary.revoked = False
    boundary.compiled = compiled
    populations = {
        node_id: supports[version]
        if version in supports
        else replace(parent, version=version)
        for node_id, version in ((n, compiled.versions[n]) for n in compiled.order)
    }
    manifest = RunManifest(
        "us",
        {},
        populations={v: populations[v].frame for v in set(compiled.versions.values())},
        mass_ledgers={
            v: populations[v].mass_ledger for v in set(compiled.versions.values())
        },
    )
    return SimpleNamespace(
        parent=parent,
        compiled=compiled,
        boundary=boundary,
        populations=populations,
        manifest=manifest,
    )


def _stamp(case, node_id):
    return extension._population_stamp(
        case.boundary, case.compiled, node_id, case.populations[node_id]
    )


@pytest.mark.parametrize(
    "node_id", [child.DONOR, child.RECIPIENT, child.FIT, child.DRAW]
)
def test_four_declared_support_nodes_preserve_actual_population_context(node_id):
    case = _case()
    actual = case.populations[node_id]
    frame = actual.frame
    before = child.child.physical._population_stamp(actual)
    seal = _stamp(case, node_id)
    assert seal == (child._support_frame_stamp(frame), before)
    assert actual.frame is frame and _stamp(case, node_id) == seal
    assert actual.version == (
        child.DONOR if node_id in (child.DONOR, child.FIT) else child.RECIPIENT
    )
    assert actual.design_weights and actual.owners
    assert frame.schema == EntitySchema(group_entities=("household",))
    with pytest.raises(ValueError, match="^FRAME_TYPE$"):
        host.reconstruction._population_stamp(actual)


@pytest.mark.parametrize(
    "node_id", ["test.parent", "test.allocated", child.ATTACH, child.VERIFY]
)
def test_regular_nodes_keep_identical_us_seal_and_reject_private_frames(node_id):
    case = _case()
    actual = case.populations[node_id]
    assert _stamp(case, node_id) == host.reconstruction._population_stamp(actual)
    assert host._node_population_stamp(case.compiled, node_id, actual) == _stamp(
        case, node_id
    )
    actual.frame._metadata = {
        **actual.frame.metadata,
        "private_model_support": child.PROTOCOL,
    }
    metadata_reason = (
        r"^US graph context has undeclared metadata: \['private_model_support'\]\.$"
    )
    with pytest.raises(ValueError, match=metadata_reason):
        _stamp(case, node_id)
    with pytest.raises(ValueError, match=metadata_reason):
        host._node_population_stamp(case.compiled, node_id, actual)
    wrong = replace(case.populations[child.DONOR], version=actual.version)
    with pytest.raises(ValueError, match="^FRAME_TYPE$"):
        extension._population_stamp(case.boundary, case.compiled, node_id, wrong)


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("duck_boundary", "COMPLETION_STAMP_BOUNDARY"),
        ("duck_child", "COMPLETION_STAMP_BOUNDARY"),
        ("copied_compiled", "COMPLETION_STAMP_BOUNDARY"),
        ("unknown_node", "COMPLETION_STAMP_VERSION"),
        ("wrong_version", "COMPLETION_STAMP_VERSION"),
        ("changed_declaration", "COMPLETION_SUPPORT_DECLARATION"),
        ("changed_bound_nodes", "COMPLETION_SUPPORT_DECLARATION"),
        ("forged_version_map", "COMPLETION_SUPPORT_VERSION"),
        ("empty_nodes", "COMPLETION_FRAGMENT_ROSTER"),
    ],
)
def test_private_route_requires_exact_boundary_declarations_and_versions(fault, reason):
    case = _case()
    boundary, compiled, node_id, population = (
        case.boundary,
        case.compiled,
        child.FIT,
        case.populations[child.FIT],
    )
    if fault == "duck_boundary":
        boundary = SimpleNamespace(**vars(boundary))
    elif fault == "duck_child":
        boundary.child = SimpleNamespace(
            nodes=boundary.child.nodes, declaration=boundary.child.declaration
        )
    elif fault == "copied_compiled":
        compiled = replace(compiled)
    elif fault == "unknown_node":
        node_id = "test.unknown"
    elif fault == "wrong_version":
        population = replace(population, version=child.RECIPIENT)
    elif fault == "changed_declaration":
        boundary.child.declaration = ()
    elif fault == "changed_bound_nodes":
        nodes = tuple(
            replace(n, description="changed retained declaration")
            if n.id == child.FIT
            else n
            for n in boundary.child.nodes
        )
        # Description is deliberately non-normative; compare actual declaration
        # objects too, so equality cannot be replaced with an opaque JSON label.
        assert tuple(n.normative() for n in nodes) == boundary.child.declaration
        boundary.child.nodes = nodes
    elif fault == "forged_version_map":
        compiled = replace(
            compiled, versions={**compiled.versions, child.FIT: child.RECIPIENT}
        )
        boundary.compiled = compiled
        population = replace(population, version=child.RECIPIENT)
    else:
        boundary.child.nodes = ()
    with pytest.raises(ValueError, match="^CURRENT_SURVEY_PREDICTOR_" + reason + "$"):
        extension._population_stamp(boundary, compiled, node_id, population)


@pytest.mark.parametrize(
    "surface",
    [
        "person_cell",
        "household_cell",
        "nullable_backing",
        "frame_weight",
        "strata",
        "metadata",
        "mass_log",
        "owners",
        "weight_kind",
        "design_weights",
        "mass_ledger",
    ],
)
def test_full_support_custody_covers_frame_and_actual_population_metadata(surface):
    case = _case()
    population = case.populations[child.DONOR]
    frame = population.frame
    if surface == "nullable_backing":
        frame.person["invented_missing"] = pd.array(
            [None] * frame.n("person"), dtype="Int64"
        )
        population = Population.from_frame(frame, population.version)
        case.populations[child.DONOR] = population
    before = _stamp(case, child.DONOR)
    if surface == "person_cell":
        frame.person.loc[0, "source_year"] += 1
    elif surface == "household_cell":
        frame.table("household").loc[0, "household_id"] += 1
    elif surface == "nullable_backing":
        array = frame.person.invented_missing.array
        assert array._mask[0]
        array._data[0] += 1
        assert pd.isna(frame.person.invented_missing.iloc[0])
    elif surface == "frame_weight":
        frame._weights["household"] = Weights(
            frame.weights_for("household").values + 1, WeightKind.DESIGN
        )
    elif surface == "strata":
        frame.strata.iloc[0] = "changed"
    elif surface == "metadata":
        frame._metadata = {**frame.metadata, "changed": True}
    elif surface == "mass_log":
        frame._mass_log = (MassChangeRecord("household", 1.0, 2.0, 2.0, "test"),)
    elif surface == "owners":
        owners = dict(population.owners)
        owners["person", "source_year"] = "invented.other.owner"
        object.__setattr__(population, "owners", owners)
    elif surface == "weight_kind":
        object.__setattr__(
            population, "weight_kind", {"household": WeightKind.IMPORTANCE}
        )
    elif surface == "design_weights":
        object.__setattr__(
            population,
            "design_weights",
            {"household": population.design_weights["household"] + 1},
        )
    else:
        record = MassRecord(
            "invented.filter",
            "filter",
            "conserve",
            4.0,
            4.0,
            (),
            (),
            entity="household",
        )
        object.__setattr__(population, "mass_ledger", (record,))
    after = _stamp(case, child.DONOR)
    assert after != before
    if surface in ("owners", "weight_kind", "design_weights", "mass_ledger"):
        assert after[0] == before[0]  # Frame-only stamp would miss the change.
        assert after[1] != before[1]


@pytest.mark.parametrize(
    "surface", ["extra_table", "extra_weight", "hidden_link", "wrong_schema"]
)
def test_hidden_support_content_is_refused_before_stamping(surface):
    case = _case()
    frame = case.populations[child.DONOR].frame
    if surface == "extra_table":
        frame._tables["unexpected"] = pd.DataFrame({"unexpected_id": [1]})
    elif surface == "extra_weight":
        frame._weights["unexpected"] = Weights([1.0], WeightKind.DESIGN)
    elif surface == "hidden_link":
        frame._link_tables["unexpected"] = pd.DataFrame({"person_id": [1]})
    else:
        frame._schema = EntitySchema(group_entities=("household", "tax_unit"))
    with pytest.raises(ValueError, match="SUPPORT_FRAME_TYPE$"):
        _stamp(case, child.DONOR)


@pytest.mark.parametrize("fault", ["empty", "missing", "extra", "reordered"])
def test_completion_retained_roster_must_match_compiled_order(fault):
    case = _case()
    observations = dict(case.populations)
    expected = host._node_population_seals(case.compiled, observations, case.boundary)
    assert len(expected) == len(case.compiled.order)
    assert all(
        p is observations[n]
        for n, (p, _) in zip(case.compiled.order, expected, strict=True)
    )
    if fault == "empty":
        observations.clear()
    elif fault == "missing":
        observations.pop(child.FIT)
    elif fault == "extra":
        observations["unexpected"] = observations[child.DONOR]
    else:
        observations = dict(reversed(tuple(observations.items())))
    with pytest.raises(
        ValueError, match="^CURRENT_SURVEY_PREDICTOR_FINANCIAL_NODE_POPULATION_ROSTER$"
    ):
        host._node_population_seals(case.compiled, observations, case.boundary)
    assert host._node_population_seals(case.compiled, None) == ()


def test_refused_observation_atomically_revokes_and_cannot_be_reused():
    case = _case()
    observed, stamps = {}, {}
    valid = case.populations[child.DONOR]
    wrong = replace(valid, version=child.RECIPIENT)
    with pytest.raises(
        ValueError, match="^CURRENT_SURVEY_PREDICTOR_COMPLETION_STAMP_VERSION$"
    ):
        extension._observe_population(
            case.boundary, case.compiled, observed, stamps, child.DONOR, wrong
        )
    assert observed == stamps == {}
    assert case.boundary.revoked and case.boundary.child.revoked
    with pytest.raises(
        ValueError, match="^CURRENT_SURVEY_PREDICTOR_COMPLETION_STAMP_BOUNDARY$"
    ):
        extension._observe_population(
            case.boundary, case.compiled, observed, stamps, child.DONOR, valid
        )


def test_duplicate_observation_keeps_original_custody_and_revokes():
    case = _case()
    observed, stamps = {}, {}
    actual = case.populations[child.DONOR]
    extension._observe_population(
        case.boundary, case.compiled, observed, stamps, child.DONOR, actual
    )
    saved = dict(stamps)
    with pytest.raises(
        ValueError, match="^CURRENT_SURVEY_PREDICTOR_COMPLETION_OBSERVER_DUPLICATE$"
    ):
        extension._observe_population(
            case.boundary, case.compiled, observed, stamps, child.DONOR, replace(actual)
        )
    assert observed[child.DONOR] is actual and stamps == saved
    assert case.boundary.revoked and case.boundary.child.revoked


def test_new_dispatch_and_binding_methods_are_in_existing_live_fence(monkeypatch):
    baseline = extension._live(False)
    assert (extension.__name__, "_population_stamp") in baseline
    assert (extension.__name__, "_observe_population") in baseline
    assert (extension.__name__, "_CompletionHost", "bind_compiled") in baseline
    with monkeypatch.context() as patch:
        patch.setattr(extension, "_population_stamp", lambda *args: "forged")
        assert extension._live(False) != baseline
    assert extension._live(False) == baseline


def test_manifest_support_admission_preserves_every_slot_and_actual_metadata():
    case = _case()
    frame = case.manifest.population(child.DONOR)
    assert type(frame) is PopulationView
    admitted = extension._manifest_support_frame(frame)
    assert type(admitted) is Frame
    assert all(
        getattr(admitted, name) is getattr(frame, name) for name in Frame.__slots__
    )
    actual = Population.from_frame(
        frame, child.DONOR, mass_ledger=case.manifest.mass_ledger(child.DONOR)
    )
    seal = extension._population_stamp(
        case.boundary, case.compiled, child.DONOR, actual, manifest=True
    )
    assert seal == (
        child._support_frame_stamp(admitted),
        child.child.physical._population_stamp(actual),
    )
    assert actual.frame is frame
    with pytest.raises(ValueError, match="SUPPORT_FRAME_TYPE$"):
        extension._population_stamp(case.boundary, case.compiled, child.DONOR, actual)
    # No schema/class widening leaks into the default survey-only manifest path.
    with pytest.raises(ValueError, match="^FRAME_TYPE$"):
        host._manifest_population_seals(case.manifest, case.compiled)
    seals = host._manifest_population_seals(
        case.manifest, case.compiled, completion_boundary=case.boundary
    )
    assert dict(seals)[child.DONOR] == seal
    assert tuple(n for n, _ in seals) == tuple(
        sorted(set(case.compiled.versions.values()))
    )
    assert set(n for n, seal in seals if isinstance(seal, tuple)) == {
        child.DONOR,
        child.RECIPIENT,
    }


@pytest.mark.parametrize("kind", ["plain_frame", "frame_subclass", "view_subclass"])
def test_manifest_inverse_view_refuses_every_other_frame_type(kind):
    case = _case()
    frame = case.populations[child.DONOR].frame
    if kind != "plain_frame":
        cls = type(
            "InventedFrameSubclass",
            (Frame if kind == "frame_subclass" else PopulationView,),
            {"__slots__": ()},
        )
        candidate = object.__new__(cls)
        for slot in Frame.__slots__:
            object.__setattr__(candidate, slot, getattr(frame, slot))
        frame = candidate
    with pytest.raises(
        ValueError, match="^CURRENT_SURVEY_PREDICTOR_COMPLETION_MANIFEST_SUPPORT_VIEW$"
    ):
        extension._manifest_support_frame(frame)
    if kind == "frame_subclass":
        with pytest.raises(ValueError, match="SUPPORT_FRAME_TYPE$"):
            extension._population_stamp(
                case.boundary,
                case.compiled,
                child.DONOR,
                replace(case.populations[child.DONOR], frame=frame),
            )


@pytest.mark.parametrize(
    "surface",
    ["person_cell", "metadata", "ledger", "extra_table", "extra_weight", "hidden_link"],
)
def test_attached_manifest_custody_covers_content_ledger_and_hidden_rosters(surface):
    case = _case()
    before = host._manifest_population_seals(
        case.manifest, case.compiled, completion_boundary=case.boundary
    )
    frame = case.manifest.population(child.RECIPIENT)
    if surface == "person_cell":
        frame.person.loc[0, "source_year"] += 1
    elif surface == "metadata":
        frame._metadata = {**frame.metadata, "changed": True}
    elif surface == "ledger":
        record = MassRecord(
            "invented.filter",
            "filter",
            "conserve",
            3.0,
            3.0,
            (),
            (),
            entity="household",
        )
        case.manifest = replace(
            case.manifest,
            mass_ledgers={**case.manifest.mass_ledgers, child.RECIPIENT: (record,)},
        )
    elif surface == "extra_table":
        frame._tables["hidden"] = pd.DataFrame({"hidden_id": [1]})
    elif surface == "extra_weight":
        frame._weights["hidden"] = Weights([1.0], WeightKind.DESIGN)
    else:
        frame._link_tables["hidden"] = pd.DataFrame({"person_id": [1]})
    if surface in ("extra_table", "extra_weight", "hidden_link"):
        with pytest.raises(ValueError, match="SUPPORT_FRAME_TYPE$"):
            host._manifest_population_seals(
                case.manifest, case.compiled, completion_boundary=case.boundary
            )
    else:
        after = host._manifest_population_seals(
            case.manifest, case.compiled, completion_boundary=case.boundary
        )
        assert dict(after)[child.RECIPIENT] != dict(before)[child.RECIPIENT]
        assert all(
            seal == dict(before)[version]
            for version, seal in after
            if version != child.RECIPIENT
        )


def test_inverse_view_slot_contract_is_live_sealed(monkeypatch):
    before = extension._live(False)
    assert (extension.__name__, "_manifest_support_frame") in before
    with monkeypatch.context() as patch:
        patch.setattr(Frame, "__slots__", Frame.__slots__[:-1])
        assert extension._live(False) != before
    assert extension._live(False) == before
