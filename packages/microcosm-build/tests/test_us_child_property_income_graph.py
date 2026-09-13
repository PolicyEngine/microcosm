"""Invented country graph values: these pure projections issue no authority."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_child_property_income as graph
from microcosm.fit import joint_empirical as empirical
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactType,
    ArtifactValue,
    Numeric,
    NumericScope,
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
