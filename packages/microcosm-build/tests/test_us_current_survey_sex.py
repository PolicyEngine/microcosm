"""Qualified source sex and tiny invented clone graph; no full51 or engine."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_sex_source as source
from microcosm.build.us_runtime import graph_current_survey_sex as graph
from microcosm.build.us_runtime import graph_us_survey_enrichment as host


def tables():
    ids = pd.Index(
        [2**53 + 1, 2**53 + 2, 2**53 + 3, 2**53 + 4, 2**53 + 5],
        dtype="int64",
        name="person_id",
    )
    origins = pd.DataFrame(
        {
            "source": ["asec"] * 3 + ["acs"] * 2,
            "native_person_id": np.array([17, 19, 23, 17, 19], dtype="int64"),
        },
        index=ids,
    )
    asec = pd.DataFrame(
        {
            "native_person_id": np.array([17, 19, 23], dtype="int64"),
            "asec_A_SEX": [1, 2, 1],
            "asec_AXSEX": [0, 4, 1],
            "asec_sex_binding_state": np.array([1, 2, 0], dtype="int64"),
            "is_female": pd.array([False, True, None], dtype="boolean"),
            "sex_known": [True, True, False],
            "sex_origin": [
                "source_no_change",
                "census_allocated",
                "unresolved_allocation",
            ],
        },
        index=ids[:3],
    ).iloc[::-1]
    acs = pd.DataFrame(
        {
            "person_id": np.array([19, 17], dtype="int64"),
            "SEX": [1, 2],
            "is_female": [False, True],
        }
    )
    return origins, asec, acs


def test_exact_original_join_preserves_unknown_allocation_and_large_ids():
    origins, asec, acs = tables()
    before = tuple(t.copy(deep=True) for t in (origins, asec, acs))
    raw = source._join(origins, asec, acs)
    actual = source.recode(raw).is_female
    pd.testing.assert_series_equal(
        actual,
        pd.Series(
            pd.array([False, True, None, True, False], dtype="boolean"),
            index=origins.index,
            name="is_female",
        ),
    )
    assert raw.survey_sex_allocation.tolist() == [
        "source_no_change",
        "census_allocated",
        "unresolved_allocation",
        "unresolved",
        "unresolved",
    ]
    assert raw.survey_sex_observation_year.tolist() == [2025, 2025, 2025, 2024, 2024]
    assert raw.survey_sex_A_SEX.iloc[3:].isna().all()
    assert raw.survey_sex_SEX.iloc[:3].isna().all()
    assert json.loads(graph._bytes(raw))["data"][0]["person_id"] == 2**53 + 1
    for old, new in zip(before, (origins, asec, acs), strict=True):
        pd.testing.assert_frame_equal(old, new)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_original",
        "float_original",
        "native_conflict",
        "missing_asec",
        "missing_acs",
        "acs_unknown",
        "acs_float",
        "acs_bool",
        "acs_canonical",
        "asec_canonical",
        "asec_known",
    ],
)
def test_exact_join_refuses_bad_axes_codes_or_carried_mapping(defect):
    origins, asec, acs = tables()
    if defect == "duplicate_original":
        origins.index = pd.Index([7] * len(origins), name="person_id")
    elif defect == "float_original":
        origins.index = origins.index.astype(float)
    elif defect == "native_conflict":
        asec.iloc[0, asec.columns.get_loc("native_person_id")] = 99
    elif defect == "missing_asec":
        asec = asec.iloc[1:]
    elif defect == "missing_acs":
        acs = acs.iloc[1:]
    elif defect == "acs_unknown":
        acs.loc[0, "SEX"] = 0
    elif defect == "acs_float":
        acs["SEX"] = acs.SEX.astype(float)
    elif defect == "acs_bool":
        acs["SEX"] = True
    elif defect == "acs_canonical":
        acs.loc[0, "is_female"] = True
    elif defect == "asec_canonical":
        asec.loc[asec.index[0], "is_female"] = False
    else:
        asec.loc[asec.index[0], "sex_known"] = True
    with pytest.raises(ValueError, match="CURRENT_SURVEY_SEX_"):
        source._join(origins, asec, acs)


@pytest.mark.parametrize("value", [None, 0, 1, "true", object()])
def test_host_option_requires_literal_boolean_before_parent_io(value):
    with pytest.raises(ValueError, match="DEMOGRAPHIC_OPTION"):
        host.Boundary(object(), groups=(), n_estimators=2, demographic_inputs=value)
    with pytest.raises(ValueError, match="DEMOGRAPHIC_OPTION"):
        host.run_us_survey_enrichment(object(), demographic_inputs=value)


def test_terminal_edge_preserves_prior_optional_fragments():
    for spm, immigration in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        edge = host._sex_after_edge(spm, immigration)
        expected = (
            host.immigration_graph.ATTACH_NODE
            if immigration
            else host.spm_graph.ATTACH_NODE
            if spm
            else host.hours_graph.ATTACH_NODE
        )
        assert edge.producer == expected


def test_copied_descriptive_values_do_not_confer_source_authority():
    value = source.QualifiedSurveySex(None, pd.DataFrame(), pd.DataFrame(), b"{}")
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        replace(value).validate()
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        graph.sex_nodes(value, object(), receiving_version="x", after=object())
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        source.qualify_current_survey_sex(object())


@pytest.mark.parametrize("mutation", ["rebind", "code"])
def test_sex_classifier_mutation_refuses_before_invocation(monkeypatch, mutation):
    original = source.asec.demographic.classify_asec_demographic_observations

    def replacement(*args, **kwargs):
        raise AssertionError("classifier replacement must not execute")

    if mutation == "rebind":
        monkeypatch.setattr(
            source.asec.demographic,
            "classify_asec_demographic_observations",
            replacement,
        )
    else:
        monkeypatch.setattr(original, "__code__", replacement.__code__)
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        source.qualify_current_survey_sex(object())
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        source.retained(object())


@pytest.mark.parametrize("name", ["A_SEX", "AXSEX"])
def test_active_demographic_field_rebind_refuses(monkeypatch, name):
    field = getattr(source.asec.demographic, name)
    replacement = replace(field, named_codes={**field.named_codes, 9: "invented"})
    monkeypatch.setattr(source.asec.demographic, name, replacement)
    assert (
        source.asec.demographic.ASEC_DEMOGRAPHIC_FIELDS[0 if name == "A_SEX" else 1]
        is field
    )
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        source.qualify_current_survey_sex(object())
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        source.retained(object())


def test_genuine_preparation_and_small_clone_graph_cold_required(tmp_path, monkeypatch):
    """One real invented-source preparation; receiving graph is fixture-only.

    This proves native source borrowing, graph execution and clone transport,
    not a full51/PUF/enrichment host. No source issuer guard is replaced.
    """
    from test_us_current_asec_demographics import _demographic_arguments
    from test_us_current_survey_health_coverage import _frame
    from test_us_current_survey_hours_source import receiving_people

    from microcosm.graph import (
        ArtifactInput,
        ArtifactOutput,
        ArtifactType,
        Capabilities,
        ContentStore,
        Determinism,
        Graph,
        KernelBase,
        KernelRegistry,
        KernelResult,
        Node,
        Owned,
        SourceRef,
        StructuralDelta,
        artifact_edges,
        compile_graph,
        run_graph,
    )
    from microcosm.graph.executor import _all_node_keys, _source_paths_and_keys

    source_root = tmp_path / "source"
    source_root.mkdir()
    arguments = _demographic_arguments(source_root, monkeypatch, unknown=True)
    preparation = source.source.prepare_authenticated_survey_population(**arguments)
    qualified = source.qualify_current_survey_sex(preparation)
    stamp = source.seal(qualified)
    assert qualified.raw.survey_sex_binding_state.eq(0).any()
    assert qualified.raw.survey_sex_allocation.eq("census_allocated").any()
    assert qualified.raw.survey_sex_source.eq("acs").any()
    people = receiving_people(qualified)
    people["unrelated_value"] = 7.5
    receiving = _frame(people)
    original = receiving.person.copy(deep=True)
    attachment_type = ArtifactType("invented.sex_parent", 1)

    class Receiving(KernelBase):
        ref = "invented.sex_receiving@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def implementation_hash(self):
            return "a" * 64

        def run(self, context):
            return KernelResult(frame=receiving, artifacts={"attachment": b"parent"})

    create = Node(
        "invented.receiving",
        Receiving.ref,
        structural=StructuralDelta.CREATE,
        sources=(graph.SOURCE_NAME,),
        outputs=tuple(
            Owned("person", c, graph.population_ops.token_for_dtype(people[c].dtype))
            for c in people
            if c != "person_id"
        ),
        artifact_outputs=(ArtifactOutput("attachment", attachment_type),),
    )
    after = ArtifactInput("parent_attachment", create.id, "attachment", attachment_type)
    nodes = graph.sex_nodes(
        qualified, receiving, receiving_version=create.id, after=after
    )
    compiled = compile_graph(
        Graph("us", (SourceRef(graph.SOURCE_NAME, "raw-bytes-v1"),), (create, *nodes))
    )

    def current():
        assert source.seal(qualified) == stamp
        pd.testing.assert_frame_equal(receiving.person, original)

    def context_check(context):
        assert context.node in nodes

    kernels = KernelRegistry()
    kernels.register(Receiving())
    for kernel in graph.sex_kernels(
        qualified,
        receiving,
        receiving_version=create.id,
        after=after,
        require_current=current,
        require_context=context_check,
    ):
        kernels.register(kernel)
    path = tmp_path / "invented-receiving.txt"
    path.write_text(
        "Invented receiving graph; source sex comes from retained genuine fixture preparation."
    )
    store = ContentStore(tmp_path / "graph-store")
    args = dict(sources={graph.SOURCE_NAME: path}, store=store, kernels=kernels)
    cold = run_graph(compiled, **args)
    warm = run_graph(compiled, **args, resume="require")
    assert cold.key == warm.key and all(n.hit for n in warm.nodes.values())
    current()
    output = warm.population(create.id)
    expected = source.attachment.attach_columns(
        qualified.origins, receiving, graph._completed(qualified)
    )
    for (_, name), series in expected.items():
        pd.testing.assert_series_equal(
            output.person.set_index("person_id")[name], series
        )
    assert (
        output.person.is_female.isna().sum()
        == 2 * source.recode(qualified.raw).is_female.isna().sum()
    )
    pd.testing.assert_frame_equal(output.person[original.columns], original)
    assert np.array_equal(
        output.weights_for("household").values,
        receiving.weights_for("household").values,
    )
    assert warm.mass_ledger(create.id) == cold.mass_ledger(create.id)

    populations = {
        create.id: graph.population_ops.Population.from_frame(receiving, create.id)
    }
    for node in nodes:
        descriptors = warm.node(node.id).typed_artifacts["inputs"]
        artifacts = {
            edge.name: artifact_edges.value_from_descriptor(
                store.load_bytes(descriptors[edge.name]["key"]), descriptors[edge.name]
            )
            for edge in node.artifact_inputs
        }
        incoming = populations.get(compiled.versions[node.id])
        result = graph.sex_result(
            qualified,
            node,
            artifacts,
            None if incoming is None else incoming.frame.person,
        )
        for name, payload in result.artifacts.items():
            assert payload == store.load_bytes(
                warm.node(node.id).typed_artifacts["outputs"][name]["key"]
            )
        populations[compiled.versions[node.id]] = (
            graph.population_ops.Population.from_frame(result.frame, node.id)
            if node.structural is StructuralDelta.CREATE
            else graph.population_ops.patch(incoming, node, result)
        )
        if node.id == graph.ATTACH_NODE:
            altered = dict(artifacts)
            altered["sex_canonical"] = replace(
                artifacts["sex_canonical"], payload=b"{}"
            )
            with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD"):
                graph.sex_result(qualified, node, altered, incoming.frame.person)
    pd.testing.assert_frame_equal(populations[create.id].frame.person, output.person)
    pd.testing.assert_frame_equal(
        populations[graph.SOURCE_NODE].frame.person,
        warm.population(graph.SOURCE_NODE).person,
    )
    _, keys = _source_paths_and_keys(compiled, args["sources"], store)
    before, _ = _all_node_keys(compiled, kernels, keys)
    changed = replace(nodes[0], params={**nodes[0].params, "mapping": "different"})
    changed_graph = compile_graph(
        replace(compiled.graph, nodes=(create, changed, *nodes[1:]))
    )
    after_keys, _ = _all_node_keys(changed_graph, kernels, keys)
    assert all(before[n.id] != after_keys[n.id] for n in nodes)
    assert {n.id for n in nodes} == {
        graph.SOURCE_NODE,
        graph.RAW_NODE,
        graph.BIND_NODE,
        graph.ATTACH_NODE,
    }
    assert nodes[2].inputs[0].columns == source.RAW_COLUMNS
    assert len(nodes[-1].inputs[0].columns) == 4
    assert "PERIDNUM" not in str([n.params for n in nodes])
    assert "raw_native_person_id" not in str([n.params for n in nodes])
    with pytest.raises(ValueError, match="PROJECTION_OBJECT_CHANGED"):
        replace(qualified).validate()
    with pytest.raises(ValueError, match="PROJECTION_OBJECT_CHANGED"):
        graph.sex_nodes(
            replace(qualified), receiving, receiving_version=create.id, after=after
        )
    qualified.validate()
    assert source.seal(qualified) == stamp

    # A foreign host callback that changes retained values after the result
    # calculation must not return a success result, even when source I/O passed.
    bind = next(n for n in nodes if n.id == graph.BIND_NODE)
    descriptors = warm.node(bind.id).typed_artifacts["inputs"]
    artifacts = {
        edge.name: artifact_edges.value_from_descriptor(
            store.load_bytes(descriptors[edge.name]["key"]), descriptors[edge.name]
        )
        for edge in bind.artifact_inputs
    }
    calls = []
    saved = qualified.raw.copy(deep=True)

    def final_change():
        calls.append(True)
        if len(calls) == 2:
            qualified.raw.iloc[
                0, qualified.raw.columns.get_loc("survey_sex_binding_state")
            ] = 0 if saved.survey_sex_binding_state.iloc[0] else 1

    kernel = graph._SexKernel(
        qualified, nodes, final_change, context_check, bind.kernel
    )
    # Stored nullable integers may normalize bytes below a null mask. The bind
    # accepts that codec behavior, but still checks masks and all known bytes.
    readback_people = warm.population(graph.SOURCE_NODE).person
    replayed = graph.sex_result(qualified, bind, artifacts, readback_people)
    pd.testing.assert_series_equal(
        replayed.columns["person", "is_female"], source.recode(qualified.raw).is_female
    )
    changed_known = readback_people.copy(deep=True)
    changed_known.loc[changed_known.index[0], "survey_sex_binding_state"] = (
        0 if changed_known.survey_sex_binding_state.iloc[0] else 1
    )
    with pytest.raises(ValueError, match="BIND_SOURCE_SLICE"):
        graph.sex_result(qualified, bind, artifacts, changed_known)
    changed_mask = readback_people.copy(deep=True)
    changed_mask.loc[changed_mask.survey_sex_SEX.isna(), "survey_sex_SEX"] = 1
    with pytest.raises(ValueError, match="BIND_SOURCE_SLICE"):
        graph.sex_result(qualified, bind, artifacts, changed_mask)
    try:
        with pytest.raises(ValueError, match="RESULT_CHANGED"):
            kernel.run(
                SimpleNamespace(
                    node=bind,
                    artifacts=artifacts,
                    tables={"person": readback_people},
                )
            )
    finally:
        qualified.raw["survey_sex_binding_state"] = (
            saved.survey_sex_binding_state.copy()
        )
    assert len(calls) == 2 and source.seal(qualified) == stamp
