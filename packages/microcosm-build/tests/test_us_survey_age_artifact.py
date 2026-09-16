"""Invented measurement and actual graph replay; no source/target admission."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_us_national_age_counts as fixture

from microcosm.build.us_runtime import graph_survey_age_artifact as artifact
from microcosm.frame import WeightKind
from microcosm.graph import (
    ContentStore,
    Graph,
    KernelRegistry,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)


def measured():
    frame = fixture.fixture_frame()
    node = artifact.survey_age_count_artifact_node(population="invented")
    original = fixture.count_context(frame, node)
    context = replace(original, tables={"person": original.tables["person"]})
    result = artifact.SurveyAgeCountArtifactKernel().run(context)
    return frame, context, result


def test_actual_count_kernel_preserves_population_and_emits_expected_counts():
    frame, context, result = measured()
    assert not result.columns and result.frame is None and result.weights is None
    values = artifact.decode_survey_age_counts(result.artifacts["counts"])
    np.testing.assert_array_equal(values.household_ids, fixture.HOUSEHOLD_IDS)
    for column, expected in fixture.EXPECTED.items():
        np.testing.assert_array_equal(values.counts[column], expected)
    assert values.people == 20
    assert values.population == "invented"
    assert not set(fixture.COLUMNS) & set(frame.table("household"))
    assert (
        result.receipt["counts_sha256"]
        == hashlib.sha256(result.artifacts["counts"]).hexdigest()
    )
    assert result.receipt["consumes_weights"] is False
    assert context.node.outputs == ()
    assert context.node.inputs == (fixture.Slice("person", ("age",)),)


def test_counts_ignore_weights_and_return_defensive_decoded_values():
    _, context, result = measured()
    changed = replace(context, weights={})
    assert (
        artifact.SurveyAgeCountArtifactKernel().run(changed).artifacts
        == result.artifacts
    )
    values = artifact.decode_survey_age_counts(result.artifacts["counts"])
    values.household_ids[0] = 999
    values.counts.iloc[0, 0] = 999
    fresh = artifact.decode_survey_age_counts(result.artifacts["counts"])
    assert fresh.household_ids[0] == 7 and fresh.counts.iloc[0, 0] == 1


def test_actual_kernel_and_transport_preserve_full_signed_int64_ids():
    frame = fixture.fixture_frame()
    ids = np.array([-(2**63), -7, 0, 5, 2**63 - 1], dtype=np.int64)
    lookup = dict(zip(fixture.HOUSEHOLD_IDS, ids, strict=True))
    person = frame.table("person")
    person["person_household_id"] = np.array(
        [lookup[int(value)] for value in person.person_household_id], dtype=np.int64
    )
    frame.table("household")["household_id"] = ids.copy()
    frame.revalidate()
    node = artifact.survey_age_count_artifact_node(population="invented")
    context = fixture.count_context(frame, node)
    result = artifact.SurveyAgeCountArtifactKernel().run(
        replace(context, tables={"person": context.tables["person"]})
    )
    values = artifact.decode_survey_age_counts(result.artifacts["counts"])
    np.testing.assert_array_equal(values.household_ids, ids)
    for column, expected in fixture.EXPECTED.items():
        np.testing.assert_array_equal(values.counts[column], expected)


@pytest.mark.parametrize("age", [-1.0, 1.5, float("nan"), 121.0])
def test_actual_kernel_age_refusals(age):
    _, context, _ = measured()
    person = context.tables["person"].copy(deep=True)
    person.iloc[0, person.columns.get_loc("age")] = age
    with pytest.raises(ValueError):
        artifact.SurveyAgeCountArtifactKernel().run(
            replace(context, tables={"person": person})
        )


def rewrite_header(payload, **changes):
    offset = len(artifact.MAGIC)
    size = int.from_bytes(payload[offset : offset + 4], "big")
    header = json.loads(payload[offset + 4 : offset + 4 + size])
    raw = artifact._json({**header, **changes})
    return (
        artifact.MAGIC
        + len(raw).to_bytes(4, "big")
        + raw
        + payload[offset + 4 + size :]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"rows": True},
        {"rows": 0},
        {"rows": 10**12},
        {"people": 0},
        {"people": 21},
        {"population": ""},
        {"columns": ["wrong"]},
        {"age_convention": "aged-forward"},
        {"protocol": "different"},
    ],
)
def test_transport_header_refusals(change):
    payload = measured()[2].artifacts["counts"]
    with pytest.raises(ValueError):
        artifact.decode_survey_age_counts(rewrite_header(payload, **change))


@pytest.mark.parametrize(
    "change",
    ["duplicate", "reversed", "negative_count", "trailing", "truncated", "magic"],
)
def test_transport_bytes_refuse(change):
    payload = measured()[2].artifacts["counts"]
    prefix = len(artifact.MAGIC) + 4
    offset = prefix + int.from_bytes(payload[len(artifact.MAGIC) : prefix], "big")
    values = np.frombuffer(payload[offset:], dtype="<i8").reshape(5, -1).copy()
    if change == "duplicate":
        values[1, 0] = values[0, 0]
    elif change == "reversed":
        values = values[::-1]
    elif change == "negative_count":
        values[0, 1] = -1
    changed = payload[:offset] + values.tobytes()
    if change == "trailing":
        changed += b"x"
    elif change == "truncated":
        changed = changed[:-1]
    elif change == "magic":
        changed = b"x" + changed[1:]
    with pytest.raises(ValueError):
        artifact.decode_survey_age_counts(changed)


def test_measurement_bound_refuses_before_original_kernel_allocation(monkeypatch):
    _, context, _ = measured()
    monkeypatch.setattr(artifact, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="LIMIT"):
        artifact.SurveyAgeCountArtifactKernel().run(context)


def test_actual_graph_cold_and_warm_keeps_counts_out_of_population(tmp_path):
    create = Node(
        "invented",
        fixture.FixtureSource.ref,
        sources=("fixture",),
        structural=StructuralDelta.CREATE,
        outputs=(
            Owned("person", "age", "float64"),
            Owned("household", "is_group_quarters", "bool"),
        ),
    )
    count = artifact.survey_age_count_artifact_node(population=create.id)
    compiled = compile_graph(
        Graph("us", (SourceRef("fixture", "frame-store"),), (create, count))
    )
    kernels = KernelRegistry()
    kernels.register(fixture.FixtureSource())
    kernels.register(artifact.SurveyAgeCountArtifactKernel())
    store = ContentStore(tmp_path / "store")
    # Fixed invented fixture construction is bound to its actual code. The
    # source codec independently hashes the persisted complete Frame bytes.
    path = store.put_frame(
        hashlib.sha256(Path(fixture.__file__).read_bytes()).hexdigest(),
        fixture.fixture_frame(kind=WeightKind.DESIGN),
    )
    cold = run_graph(compiled, sources={"fixture": path}, store=store, kernels=kernels)
    warm = run_graph(
        compiled,
        sources={"fixture": path},
        store=store,
        kernels=kernels,
        resume="require",
    )
    assert cold.key == warm.key
    assert all(row.store_hit for row in warm.nodes.values())
    frame = warm.population(create.id)
    assert not set(fixture.COLUMNS) & set(frame.table("household"))
    receipt = warm.node(count.id)
    raw = store.load_bytes(receipt.opaque_artifacts["counts"])
    values = artifact.decode_survey_age_counts(raw)
    np.testing.assert_array_equal(
        values.household_ids, frame.table("household")["household_id"]
    )
    assert values.people == frame.n("person")
    pd.testing.assert_frame_equal(
        frame.table("household"), fixture.fixture_frame().table("household")
    )
