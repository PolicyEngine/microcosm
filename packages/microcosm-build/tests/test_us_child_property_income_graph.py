"""Invented country graph values: these pure projections issue no authority."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_child_property_income as graph
from microcosm.fit import joint_empirical as empirical
from microcosm.frame import EntitySchema, Frame, MassChangeRecord, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Graph,
    KernelContext,
    Node,
    Numeric,
    NumericScope,
    Owned,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
)

ORDER_TYPE = ArtifactType("test.child_property_ordering", 1)
ORDER_BYTES = b"invented checked receiving parent"


def _options(*, zero=False, only_positive=False):
    one = empirical.SupportThreshold(1, 1, 1.0, 1.0)
    return graph.ChildPropertyOptions(
        "invented-zero-stress" if zero else "invented-reference",
        "test",
        empirical.SupportRequirements(
            "invented-minimum-one",
            "test",
            one,
            one,
            one,
            (3,) if only_positive else (0, 1, 2, 3),
        ),
        empirical.JointTransport(0.0 if zero else 1.0, (1.0, 1.0)),
        ("sha256-u53-v1", "invented-child-law", 0, 7),
    )


def _qualified(*, no_children=False):
    donor_keys = tuple(
        ("asec", 2024, 2025, str(i + 1).zfill(5), str(2**53 + i + 1), "1")
        for i in range(4)
    )
    donors = pd.DataFrame(
        {
            graph.child.TARGETS[0]: [0.0, 10.0, 0.0, 3.0],
            graph.child.TARGETS[1]: [0.0, 0.0, 20.0, 7.0],
            graph.WEIGHT: [1.0, 2.0, 3.0, 4.0],
            "donor_key": donor_keys,
            "household_key": [key[:4] for key in donor_keys],
        },
        index=pd.Index(
            np.arange(4, dtype=np.int64) + 2**53 + 1, name="native_person_id"
        ),
    )
    diagnostic = pd.DataFrame(
        {
            "cohort": [True] * 4,
            "ordinary_known": [True] * 4,
            "dividend_known": [True] * 4,
        },
        index=donors.index,
    )
    descriptor = pd.DataFrame(
        {"literal": ["unallocated", "allocated", "topcoded", "ordinary_zero"]},
        index=donors.index,
    )
    ids = np.array([2**53 + 101, 2**53 + 102, 2**53 + 103], dtype=np.int64)
    keys = [
        (
            source,
            2024,
            2025 if source == "asec" else 2024,
            str(i + 7).zfill(5),
            str(2**53 + 501 + i),
            "1",
        )
        for i, source in enumerate(("asec", "acs", "asec"))
    ]
    ages = [20, 20, 40] if no_children else [14, 10, 40]
    recipients = pd.DataFrame(
        {
            "recipient_key": keys,
            "source_age": ages,
            "source": [key[0] for key in keys],
            "ordinary_source_known": [False] * 3,
            "dividend_source_known": [False] * 3,
            "eligible": [age < 15 for age in ages],
            "reason": [
                "explicit_unmeasured_child"
                if age < 15
                else "outside_recipient_age_band"
                for age in ages
            ],
        },
        index=pd.Index(ids, name="person_id"),
    )
    origins = pd.DataFrame(
        [dict(zip(graph.KEY_COLUMNS, key, strict=True)) for key in keys],
        index=recipients.index,
    )
    origins["selected_receiving_person_id"] = ids + 10000
    origins["selected_receiving_household_id"] = ids + 20000
    origins["household_id"] = ids + 30000
    projection = graph.child.ChildPropertyDonorProjection(
        donors, diagnostic, descriptor.copy(), descriptor.copy()
    )
    return graph.child.QualifiedChildPropertySources(
        projection,
        recipients,
        {
            "catalogue_sha256": "a" * 64,
            "preparation_sha256": "b" * 64,
            "fixture_only": True,
        },
    ), origins


def _receiving_frame(qualified, origins):
    """Invented exact two-clone parent; no country host or source issuance."""
    home_ids = tuple(dict.fromkeys(origins.household_id.tolist()))
    home_map = {
        (int(home), clone): 2**53 + 50000 + 2 * i + clone
        for i, home in enumerate(home_ids)
        for clone in (0, 1)
    }
    rows = []
    for i, origin in enumerate(origins.itertuples()):
        original = origin.Index
        for clone in (0, 1):
            home = home_map[int(origin.household_id), clone]
            rows.append(
                {
                    "person_id": 2**53 + 100000 + 2 * i + clone,
                    "person_household_id": home,
                    "person_tax_unit_id": home,
                    graph.provenance.support_source_id_column("person"): int(original),
                    graph.provenance.support_clone_index_column("person"): clone,
                    graph.provenance.spine_source_id_column("person"): int(
                        origin.selected_receiving_person_id
                    ),
                    graph.provenance.support_channel_column("person"): origin.source,
                    graph.child.TARGETS[0]: np.nan
                    if qualified.recipients.loc[original, "eligible"]
                    else 31.0,
                    graph.child.TARGETS[1]: np.nan
                    if qualified.recipients.loc[original, "eligible"]
                    else 41.0,
                    "unrelated_amount": -12.5,
                    "source_amount_known": False,
                    "census_block": "012345678901234",
                }
            )
    people = pd.DataFrame(rows)
    # The invented CREATE must declare the graph's nullable string dtype;
    # pandas inference can instead choose the distinct NaN-based str dtype.
    for column in (
        graph.provenance.support_channel_column("person"),
        "census_block",
    ):
        people[column] = pd.array(
            people[column], dtype=graph.populations.dtype_for_token("string")
        )
    people["nullable_source"] = pd.array([pd.NA, True] * len(origins), dtype="boolean")
    homes = pd.DataFrame(
        {
            "household_id": list(home_map.values()),
            graph.provenance.support_source_id_column("household"): [
                key[0] for key in home_map
            ],
            graph.provenance.support_clone_index_column("household"): [
                key[1] for key in home_map
            ],
            "unrelated_household": 71.0,
        }
    )
    return Frame(
        {
            "person": people,
            "household": homes,
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": homes.household_id, "unrelated_tax_unit": 89.0}
            ),
        },
        EntitySchema(group_entities=("household", "tax_unit")),
        {
            "household": Weights(
                np.arange(len(homes), dtype=np.float64) + 1.5, WeightKind.DESIGN
            )
        },
        metadata={"invented_parent": {"purpose": "whole-state preservation"}},
    )


def test_receiving_fixture_declares_strings_without_changing_other_values():
    qualified, origins = _qualified()
    people = _receiving_frame(qualified, origins).person
    channel = graph.provenance.support_channel_column("person")
    canonical_string = pd.StringDtype(storage="python", na_value=pd.NA)
    for column, values in (
        (channel, ["asec", "asec", "acs", "acs", "asec", "asec"]),
        ("census_block", ["012345678901234"] * 6),
    ):
        pd.testing.assert_series_equal(
            people[column], pd.Series(values, name=column, dtype=canonical_string)
        )
        assert people[column].dtype.storage == "python"
        assert people[column].dtype.na_value is pd.NA

    source_id = graph.provenance.support_source_id_column("person")
    spine_id = graph.provenance.spine_source_id_column("person")
    for column, offsets in (
        ("person_id", [100000, 100001, 100002, 100003, 100004, 100005]),
        (source_id, [101, 101, 102, 102, 103, 103]),
        (spine_id, [10101, 10101, 10102, 10102, 10103, 10103]),
        ("person_household_id", [50000, 50001, 50002, 50003, 50004, 50005]),
        ("person_tax_unit_id", [50000, 50001, 50002, 50003, 50004, 50005]),
    ):
        pd.testing.assert_series_equal(
            people[column],
            pd.Series(
                [2**53 + offset for offset in offsets], name=column, dtype="int64"
            ),
        )
    pd.testing.assert_series_equal(
        people["nullable_source"],
        pd.Series([pd.NA, True] * 3, name="nullable_source", dtype="boolean"),
    )
    pd.testing.assert_series_equal(
        people["source_amount_known"],
        pd.Series([False] * 6, name="source_amount_known", dtype="bool"),
    )
    for column, values in (
        (graph.child.TARGETS[0], [np.nan] * 4 + [31.0, 31.0]),
        (graph.child.TARGETS[1], [np.nan] * 4 + [41.0, 41.0]),
        ("unrelated_amount", [-12.5] * 6),
    ):
        pd.testing.assert_series_equal(
            people[column], pd.Series(values, name=column, dtype="float64")
        )


def _ordering(producer="test.parent"):
    edge = ArtifactInput("ordering", producer, "ordering", ORDER_TYPE)
    producer_key = "c" * 64
    value = ArtifactValue(
        ORDER_BYTES,
        ORDER_TYPE,
        graph.opaque_artifact_key(producer_key, "ordering"),
        producer_key,
        NumericScope(),
    )
    pins = {
        edge.name: {
            "producer_key": value.producer_key,
            "artifact_key": value.key,
            "payload_sha256": graph._sha(value.payload),
        }
    }
    return edge, value, pins


def _case(*, zero=False, no_children=False):
    qualified, origins = _qualified(no_children=no_children)
    parent = graph.populations.Population.from_frame(
        _receiving_frame(qualified, origins), "test.allocated"
    )
    edge, value, pins = _ordering()
    options = _options(zero=zero)
    nodes = graph.child_property_nodes(
        qualified, origins, parent, options=options, host_edges=(edge,), host_pins=pins
    )
    expected, result, payloads, donors, recipients = graph._reconstruct(
        qualified, origins, parent, options, nodes
    )
    return (
        qualified,
        origins,
        parent,
        nodes,
        expected,
        result,
        payloads,
        donors,
        recipients,
        value,
        pins,
    )


def _support_kernel_case(kind, tmp_path, on_final):
    """Invented boundary for kernel result custody only; no source authority."""
    qualified, origins, parent, nodes, _, _, _, donors, recipients, value, pins = (
        _case()
    )
    selected = next(node for node in nodes if node.id == kind)
    boundary = SimpleNamespace(
        nodes=nodes,
        origins=origins,
        entry=(None, None, SimpleNamespace(root=tmp_path)),
        host_pins=pins,
        options_bytes=graph._json(_options().document()),
        revoked=False,
        calls=0,
        pure_calls=0,
    )

    def current():
        graph.require(not boundary.revoked, "TEST_BOUNDARY_REVOKED")
        boundary.calls += 1
        if boundary.calls == 2:
            on_final()
        return qualified

    def pure():
        boundary.pure_calls += 1

    boundary._current, boundary._pure = current, pure
    context = KernelContext(
        selected,
        {},
        {},
        pd.Series([], dtype="string"),
        selected.params,
        np.random.default_rng(0),
        sources={graph.SOURCE_NAME: tmp_path},
        artifacts={"ordering": value},
    )
    return (
        graph._ChildKernel(boundary, selected.kernel),
        context,
        boundary,
        donors if kind == graph.DONOR else recipients,
        parent,
    )


def _run_captured_support_kernel(kernel, context, captured):
    # Observe the actual created Frame without replacing any production callable.
    previous = sys.getprofile()

    def observe(frame, event, result):
        if event == "return" and frame.f_code is graph._support_frame.__code__:
            captured.append(result)

    sys.setprofile(observe)
    try:
        return kernel.run(context)
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("kind", [graph.DONOR, graph.RECIPIENT])
def test_private_support_create_preserves_exact_frame_through_final_callback(
    kind, tmp_path
):
    captured = []
    kernel, context, boundary, table, parent = _support_kernel_case(
        kind, tmp_path, lambda: None
    )
    parent_stamp = graph.child.physical._population_stamp(parent)
    result = _run_captured_support_kernel(kernel, context, captured)
    assert len(captured) == 1 and result.frame is captured[0]
    assert boundary.calls == 2 and boundary.pure_calls == 1
    assert not boundary.revoked
    graph.replay.same_replayed_frame(graph._support_frame(table), result.frame)
    pd.testing.assert_frame_equal(result.frame.person[list(table)], table)
    assert result.frame.schema == EntitySchema(group_entities=("household",))
    assert not result.frame.links and not result.frame._link_tables
    # A support stamp neither coerces the Frame into US_SCHEMA nor changes that
    # separate owner's strict source-population admission rule.
    with pytest.raises(ValueError, match="^FRAME_TYPE$"):
        graph.child.source._frame_identity(result.frame)
    assert graph.child.physical._population_stamp(parent) == parent_stamp


@pytest.mark.parametrize("kind", [graph.DONOR, graph.RECIPIENT])
@pytest.mark.parametrize(
    "surface",
    [
        "person_cell",
        "household_cell",
        "dtype",
        "index",
        "table_flags",
        "weights",
        "weight_kind",
        "strata",
        "metadata",
        "mass_log",
        "schema",
        "extra_table",
        "missing_table",
        "hidden_link",
        "hidden_weight",
    ],
)
def test_private_support_final_callback_mutation_refuses_without_changing_parent(
    kind, surface, tmp_path
):
    captured, mutations = [], []

    def mutate():
        assert len(captured) == 1
        frame = captured[0]
        mutations.append(surface)
        if surface == "person_cell":
            frame.person.loc[0, "raw_native_person_id"] = "changed"
        elif surface == "household_cell":
            frame.table("household").loc[0, "household_id"] += 1
        elif surface == "dtype":
            frame.person["source_year"] = frame.person.source_year.astype("int32")
        elif surface == "index":
            frame.person.index = pd.Index(frame.person.index.to_numpy(), name="changed")
        elif surface == "table_flags":
            frame.person.flags.allows_duplicate_labels = False
        elif surface == "weights":
            frame._weights["household"] = Weights(
                frame.weights_for("household").values + 1.0, WeightKind.DESIGN
            )
        elif surface == "weight_kind":
            frame._weights["household"] = Weights(
                frame.weights_for("household").values, WeightKind.IMPORTANCE
            )
        elif surface == "strata":
            frame.strata.iloc[0] = "changed"
        elif surface == "metadata":
            frame._metadata = {"changed": True}
        elif surface == "mass_log":
            frame._mass_log = (MassChangeRecord("household", 1.0, 2.0, 2.0, "test"),)
        elif surface == "schema":
            frame._schema = EntitySchema(group_entities=("household", "tax_unit"))
        elif surface == "extra_table":
            frame._tables["unexpected"] = pd.DataFrame({"unexpected_id": [1]})
        elif surface == "missing_table":
            del frame._tables["household"]
        elif surface == "hidden_link":
            frame._link_tables["unexpected"] = pd.DataFrame({"person_id": [1]})
        else:
            frame._weights["unexpected"] = Weights(np.ones(1), WeightKind.DESIGN)

    kernel, context, boundary, _, parent = _support_kernel_case(kind, tmp_path, mutate)
    parent_stamp = graph.child.physical._population_stamp(parent)
    with pytest.raises(
        ValueError, match="FINAL_KERNEL_RESULT_CHANGED|SUPPORT_FRAME_TYPE"
    ):
        _run_captured_support_kernel(kernel, context, captured)
    assert mutations == [surface] and boundary.calls == 2
    assert boundary.revoked and boundary.pure_calls == 0
    assert graph.child.physical._population_stamp(parent) == parent_stamp
    with pytest.raises(ValueError, match="TEST_BOUNDARY_REVOKED"):
        kernel.run(context)


def test_child_nodes_compile_with_structural_ids_and_source_identity_reads():
    _, _, parent, nodes, *_ = _case()
    frame = parent.frame
    create = Node(
        "test.parent",
        "test.child.receiving@1",
        sources=(graph.SOURCE_NAME,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(
                entity,
                column,
                graph.populations.token_for_dtype(frame.table(entity)[column].dtype),
            )
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
            (SourceRef(graph.SOURCE_NAME, "frame-store"),),
            (create, allocated, *nodes),
        )
    )
    assert graph.ATTACH in compiled.predecessors[graph.VERIFY]
    for node in nodes[-2:]:
        reads = {s.entity: set(s.columns) for s in node.inputs}
        for entity in frame.entities:
            expected = set(frame.table(entity))
            expected.remove(frame.schema.entity_id_column(entity))
            if entity == frame.schema.person_entity:
                expected.difference_update(
                    frame.schema.membership_column(group)
                    for group in frame.schema.group_entities
                )
                if node.id == graph.VERIFY:
                    expected.update((graph.STATUS, graph.IMPUTED))
            assert reads[entity] == expected
        assert graph.provenance.support_source_id_column("person") in reads["person"]
        assert graph.provenance.spine_source_id_column("person") in reads["person"]


@pytest.mark.parametrize(
    "zero,no_children", [(False, False), (True, False), (False, True)]
)
def test_real_empirical_reconstruction_preserves_complete_parent_and_clone_pairs(
    zero, no_children
):
    q, origins, parent, nodes, expected, result, payloads, donors, recipients, _, _ = (
        _case(zero=zero, no_children=no_children)
    )
    assert len(nodes) == 6
    assert nodes[3].params["eligibility_column"] == "eligible"
    assert len(recipients) == len(q.recipients)
    assert len(graph._support_frame(recipients).person) == 3
    assert len(donors) == 4  # Original donor support, never two cloned copies.
    assert not nodes[0].inputs and nodes[0].outputs
    assert graph.SOURCE_NAME in nodes[0].sources
    for entity in parent.frame.entities:
        names = [
            name
            for name in parent.frame.table(entity)
            if entity != "person" or name not in graph.child.TARGETS
        ]
        pd.testing.assert_frame_equal(
            expected.frame.table(entity)[names],
            parent.frame.table(entity)[names],
            check_exact=True,
        )
    for name in graph.child.TARGETS:
        original = expected.frame.person[
            graph.provenance.support_source_id_column("person")
        ]
        for identity, group in expected.frame.person.groupby(original):
            assert len(group) == 2 and group[name].nunique(dropna=False) == 1
            if q.recipients.loc[identity, "eligible"]:
                assert group[graph.IMPUTED].all()
                if zero:
                    assert group[name].eq(0.0).all()
                    assert group[graph.STATUS].eq("imputed_zero").all()
    replayed = graph.populations.patch(parent, nodes[4], result)
    graph.replay.same_replayed_population(expected, replayed)
    completion = json.loads(payloads[graph.ATTACH, "completion"])
    assert completion["source_knownness_changed"] is False
    assert (
        completion["receiving_imputed_count"]
        == 2 * completion["original_recipient_count"]
    )


@pytest.mark.parametrize(
    "defect",
    [
        "original_float",
        "native_id",
        "clone",
        "household",
        "channel",
        "known_o",
        "known_d",
    ],
)
def test_exact_source_clone_join_and_known_incumbent_refuse(defect):
    q, origins = _qualified()
    frame = _receiving_frame(q, origins)
    people = frame.person
    if defect == "original_float":
        people[graph.provenance.support_source_id_column("person")] = people[
            graph.provenance.support_source_id_column("person")
        ].astype(float)
    elif defect == "native_id":
        people.loc[0, graph.provenance.spine_source_id_column("person")] += 1
    elif defect == "clone":
        people.loc[0, graph.provenance.support_clone_index_column("person")] = 1
    elif defect == "household":
        people.loc[0, "person_household_id"] = people.loc[2, "person_household_id"]
    elif defect == "channel":
        people.loc[0, graph.provenance.support_channel_column("person")] = "acs"
    else:
        people.loc[0, graph.child.TARGETS[defect == "known_d"]] = 0.0
    parent = graph.populations.Population.from_frame(frame, "test.allocated")
    edge, _, pins = _ordering()
    with pytest.raises(ValueError):
        nodes = graph.child_property_nodes(
            q, origins, parent, options=_options(), host_edges=(edge,), host_pins=pins
        )
        graph._reconstruct(q, origins, parent, _options(), nodes)


@pytest.mark.parametrize(
    "defect",
    ["amount", "knownness", "unrelated", "group", "weights", "owners", "ledger"],
)
def test_independent_whole_population_comparison_refuses_unowned_or_completed_drift(
    defect,
):
    _, _, parent, nodes, expected, result, *_ = _case()
    changed = graph.populations.patch(parent, nodes[4], result)
    if defect == "amount":
        changed.frame.person.loc[0, graph.child.TARGETS[0]] += 1.0
    elif defect == "knownness":
        changed.frame.person.loc[0, "source_amount_known"] = True
    elif defect == "unrelated":
        changed.frame.person.loc[0, "unrelated_amount"] += 1.0
    elif defect == "group":
        changed.frame.table("tax_unit").loc[0, "unrelated_tax_unit"] += 1.0
    elif defect == "weights":
        altered = changed.design_weights["household"].copy()
        altered[0] += 1.0
        changed = replace(changed, design_weights={"household": altered})
    elif defect == "owners":
        changed = replace(
            changed, owners={**changed.owners, ("person", graph.STATUS): "forged"}
        )
    else:
        # A different ledger tuple type is invalid even when empty.
        object.__setattr__(changed, "mass_ledger", [])
    with pytest.raises(ValueError):
        graph.replay.same_replayed_population(expected, changed)


@pytest.mark.parametrize(
    "defect", ["model", "draws", "scenario", "completion", "producer", "host"]
)
def test_exact_reconstructed_artifact_bindings_refuse_changed_values(defect):
    *_, nodes, expected, result, payloads, donors, recipients, order, pins = _case()
    del expected, result, donors, recipients
    node = nodes[-1]
    values = {}
    for edge in node.artifact_inputs:
        if edge.name == "ordering":
            values[edge.name] = order
        else:
            producer = graph._sha(edge.producer.encode())
            values[edge.name] = ArtifactValue(
                payloads[edge.producer, edge.artifact],
                edge.type,
                graph.opaque_artifact_key(producer, edge.artifact),
                producer,
                NumericScope(
                    Numeric.PLATFORM_BITWISE, platform=graph.platform_fingerprint()
                ),
            )
    if defect == "producer":
        value = values["model_metadata"]
        producer = "f" * 64
        values["model_metadata"] = replace(
            value,
            producer_key=producer,
            key=graph.opaque_artifact_key(producer, "model_metadata"),
        )
    else:
        name = "ordering" if defect == "host" else defect
        values[name] = replace(values[name], payload=values[name].payload + b" ")
    with pytest.raises(ValueError):
        graph._artifacts(node, values, payloads, pins)


def test_target_sampling_digest_does_not_change_full_donor_projection_or_model():
    q, origins = _qualified()
    donor_payload = graph._projections(q, origins)[2]
    changed = replace(q, evidence={**q.evidence, "preparation_sha256": "f" * 64})
    assert graph._projections(changed, origins)[2] == donor_payload
    assert graph._projections(changed, origins)[3] != graph._projections(q, origins)[3]
    models = []
    for qualified in (q, changed):
        table = graph._projections(qualified, origins)[0]
        models.append(
            empirical.fit_joint_empirical(
                table,
                targets=graph.child.TARGETS,
                donor_keys=tuple(
                    table[list(graph.KEY_COLUMNS)].itertuples(index=False, name=None)
                ),
                household_keys=tuple(
                    table[list(graph.KEY_COLUMNS[:4])].itertuples(
                        index=False, name=None
                    )
                ),
                support=_options().support,
                weights=graph.WEIGHT,
            ).to_bytes()
        )
    assert models[0] == models[1]
