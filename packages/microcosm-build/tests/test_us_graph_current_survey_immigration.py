"""Typed pair encoding and clone mapping; these pure tests issue no authority."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_current_survey_immigration as graph
from microcosm.build.us_runtime import graph_us_survey_enrichment as host


def pairs():
    return pd.DataFrame(
        {
            "ssn_card_type": ["CITIZEN", "NONE"],
            "immigration_status_str": ["CITIZEN", "UNDOCUMENTED"],
        },
        index=pd.Index([2**53 + 1, 2**63 - 1], dtype="int64", name="person_id"),
        dtype=object,
    )


def mapping_case():
    table = pairs()
    originals = table.index.to_numpy()
    origins = pd.DataFrame(
        {
            "source": ["asec", "acs"],
            # Overlapping native IDs must stay distinct by combined original ID.
            "native_person_id": np.array([2**53 + 7, 2**53 + 7], dtype="int64"),
            "source_year": [2024, 2024],
            "survey_year": [2025, 2024],
        },
        index=table.index.copy(),
    )
    provenance = graph.attachment.provenance
    people = pd.DataFrame(
        {
            "person_id": np.array([201, 107, 301, 99], dtype="int64"),
            provenance.support_source_id_column("person"): originals[[1, 0, 0, 1]],
            provenance.support_clone_index_column("person"): np.array(
                [0, 1, 0, 1], dtype="int64"
            ),
            provenance.spine_source_id_column("person"): np.repeat(2**53 + 7, 4).astype(
                "int64"
            ),
            provenance.support_channel_column("person"): pd.array(
                ["acs", "asec", "asec", "acs"], dtype="string"
            ),
            "unrelated": [1.5, 2.5, 3.5, 4.5],
        }
    )
    return origins, SimpleNamespace(person=people), table


def test_private_pair_encoding_is_exact_lossless_and_defensive():
    original = pairs()
    before = original.copy(deep=True)
    canonical = graph.canonical_pairs(original)
    assert canonical is not original
    assert all(dtype == graph.STRING for dtype in canonical.dtypes)
    payload = graph.pair_bytes(original)
    assert json.loads(payload)["person_id"] == [2**53 + 1, 2**63 - 1]
    restored = graph.read_pairs(payload)
    pd.testing.assert_frame_equal(restored, canonical)
    pd.testing.assert_frame_equal(original, before)
    canonical.iloc[0, 0] = "NONE"
    pd.testing.assert_frame_equal(original, before)


@pytest.mark.parametrize("value", [None, pd.NA, 1.0, True, "UNKNOWN"])
def test_invalid_pair_values_do_not_become_strings(value):
    table = pairs()
    table.iloc[0, 0] = value
    with pytest.raises(ValueError, match="PAIR_DOMAIN"):
        graph.pair_bytes(table)


@pytest.mark.parametrize(
    "change", ["float", "duplicate", "unnamed", "reversed", "extra"]
)
def test_pair_axis_and_target_order_refuse(change):
    table = pairs()
    if change == "float":
        table.index = table.index.astype("float64")
    elif change == "duplicate":
        table.index = pd.Index([1, 1], name="person_id")
    elif change == "unnamed":
        table.index.name = None
    elif change == "reversed":
        table = table.loc[:, list(reversed(graph.owner.OUTPUTS))]
    else:
        table["years_in_us"] = 5
    with pytest.raises(ValueError, match="PAIR_AXIS"):
        graph.pair_bytes(table)


@pytest.mark.parametrize("value", [True, 1.0, 2**63, -(2**63) - 1, "001"])
def test_private_decoder_does_not_coerce_coordinate_types(value):
    document = json.loads(graph.pair_bytes(pairs()))
    document["person_id"][0] = value
    with pytest.raises(ValueError, match="PAIR_ENCODING"):
        graph.read_pairs(graph.owner.source._encode(document))


def test_clone_mapping_preserves_indivisible_pair_and_exact_source_coordinates():
    origins, receiving, table = mapping_case()
    before = receiving.person.copy(deep=True)
    original = table.copy(deep=True)
    mapped = graph.attachment.attach_columns(
        origins, receiving, graph.canonical_pairs(table)
    )
    support = graph.attachment.provenance.support_source_id_column("person")
    for name in graph.owner.OUTPUTS:
        values = mapped["person", name]
        assert values.dtype == graph.STRING
        assert values.index.tolist() == before.person_id.tolist()
        assert values.tolist() == table[name].reindex(before[support]).tolist()
    pd.testing.assert_frame_equal(receiving.person, before)
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate_id",
        "float_id",
        "role",
        "duplicate_role",
        "original",
        "native",
        "channel",
        "ssn_collision",
        "status_collision",
        "reordered_pairs",
    ],
)
def test_clone_mapping_refuses_invalid_receiving_identity(change):
    origins, receiving, table = mapping_case()
    people = receiving.person
    provenance = graph.attachment.provenance
    if change == "missing":
        receiving.person = people.iloc[:-1].copy()
    elif change == "extra":
        receiving.person = pd.concat([people, people.iloc[:1]])
    elif change == "duplicate_id":
        people.loc[0, "person_id"] = people.loc[1, "person_id"]
    elif change == "float_id":
        people["person_id"] = people.person_id.astype("float64")
    elif change == "role":
        people.loc[0, provenance.support_clone_index_column("person")] = 2
    elif change == "duplicate_role":
        people.loc[0, provenance.support_clone_index_column("person")] = 1
    elif change == "original":
        people.loc[0, provenance.support_source_id_column("person")] = 42
    elif change == "native":
        people.loc[0, provenance.spine_source_id_column("person")] = 17
    elif change == "channel":
        people.loc[0, provenance.support_channel_column("person")] = "asec"
    elif change == "ssn_collision":
        people[graph.owner.OUTPUTS[0]] = "CITIZEN"
    elif change == "status_collision":
        people[graph.owner.OUTPUTS[1]] = "CITIZEN"
    else:
        table = table.iloc[::-1]
    with pytest.raises(ValueError, match="ATTACH_|CLONE_"):
        graph.attachment.attach_columns(
            origins, receiving, graph.canonical_pairs(table)
        )


@pytest.mark.parametrize(
    "value", [object(), graph.owner.CurrentSurveyImmigrationTransfer(None, None, b"{}")]
)
def test_unissued_transfer_refuses_without_source_io(value):
    with pytest.raises(ValueError, match="ISSUED_TRANSFER_REQUIRED"):
        graph.retained_entry(value)
    with pytest.raises(ValueError, match="ISSUED_TRANSFER_REQUIRED"):
        host.run_us_survey_enrichment(object(), immigration_transfer=value)


def test_copied_descriptive_owner_cannot_declare_nodes():
    value = graph.owner.CurrentSurveyImmigrationTransfer(None, None, b"{}")
    with pytest.raises(ValueError, match="ISSUED_TRANSFER_REQUIRED"):
        graph.immigration_nodes(
            replace(value), object(), receiving_version="x", after=object()
        )


@pytest.mark.parametrize("spm", [False, True])
def test_attachment_follows_existing_optional_terminal(spm):
    edge = host._immigration_after_edge(spm)
    assert edge.producer == (
        host.spm_graph.ATTACH_NODE if spm else host.hours_graph.ATTACH_NODE
    )
    assert edge.artifact == "attachment"
    assert edge.type == (
        host.spm_graph.ATTACHMENT_TYPE if spm else host.hours_graph.ATTACHMENT_TYPE
    )
