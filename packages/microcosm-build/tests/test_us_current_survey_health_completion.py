"""Invented source missingness and exact original-person health completion."""

import numpy as np
import pandas as pd
import pytest
from test_us_current_survey_health_coverage import invented_raw

from microcosm.build.us_runtime import current_survey_health_completion as completion
from microcosm.build.us_runtime import current_survey_health_coverage as health


def donor_people():
    table = pd.DataFrame({"person_id": [11, 12, 13]})
    for name, values in zip(
        completion.FEATURES,
        ([20.0, 40.0, 60.0], [0.0, 1.0, 1.0], [6.0] * 3),
        strict=True,
    ):
        table[name] = values
    for field in completion.FIELDS:
        table[health.SOURCE_PREFIX + field.asec] = pd.array(
            ["1", "2", ""], dtype=health.STRING_DTYPE
        )
        table[health.SOURCE_PREFIX + "I_" + field.asec] = pd.array(
            ["1", "0", "0"], dtype=health.STRING_DTYPE
        )
    return table


def test_donor_eligibility_preserves_allocated_values_and_excludes_unknowns():
    people = donor_people()
    before = people.copy(deep=True)
    columns = completion.model_columns(people)
    assert columns[completion.ELIGIBLE].tolist() == [True, True, False]
    for target in completion.TARGETS:
        assert columns[target].iloc[:2].tolist() == [1.0, 0.0]
        assert np.isnan(columns[target].iloc[2])
    pd.testing.assert_frame_equal(people, before, check_exact=True)


@pytest.mark.parametrize("feature", completion.FEATURES)
def test_missing_donor_predictor_is_excluded_without_filling(feature):
    people = donor_people()
    people.loc[0, feature] = np.nan
    assert completion.model_columns(people)[completion.ELIGIBLE].tolist() == [
        False,
        True,
        False,
    ]
    assert np.isnan(people.loc[0, feature])


def draws(raw):
    return pd.DataFrame(
        {target: [float(i % 2)] for i, target in enumerate(completion.TARGETS)},
        index=raw.index[raw.source.eq("acs")],
    )


def test_completion_changes_only_seven_acs_gaps_and_keeps_source_knownness():
    raw = invented_raw()
    before = raw.copy(deep=True)
    values = draws(raw)
    result = completion.completed_columns(raw, values)
    for field, target in zip(completion.FIELDS, completion.TARGETS, strict=True):
        observed = health.recode_field(raw, field.output)
        assert result.loc[20, field.output] == bool(values.loc[20, target])
        assert not result.loc[20, field.output + "__known"]
        assert result.loc[20, field.output + "__source_status"].startswith(
            "semantic_gap:"
        )
        assert result[field.output + "__imputed"].tolist() == [False, True]
        pd.testing.assert_series_equal(
            result.loc[[10], field.output], observed.loc[[10], field.output]
        )
    for field in health.FIELDS:
        if field.acs is not None:
            expected = health.recode_field(raw, field.output)
            pd.testing.assert_frame_equal(result.loc[:, list(field.columns)], expected)
    pd.testing.assert_frame_equal(
        result.loc[:, health.source_columns(raw).columns], health.source_columns(raw)
    )
    pd.testing.assert_frame_equal(raw, before, check_exact=True)


@pytest.mark.parametrize("invalid", [0.5, -1.0, 1.000000001, np.nan, np.inf])
def test_completion_refuses_nonbinary_draw_without_rounding(invalid):
    raw = invented_raw()
    values = draws(raw)
    values.iloc[0, 0] = invalid
    with pytest.raises(ValueError, match="DRAW_DOMAIN"):
        completion.completed_columns(raw, values)


def test_completion_refuses_changed_original_axis_or_target_order():
    raw = invented_raw()
    values = draws(raw)
    for changed in (values.rename(index={20: 21}), values.iloc[:, ::-1]):
        with pytest.raises(ValueError, match="DRAW_AXIS"):
            completion.completed_columns(raw, changed)


def test_missing_asec_answer_is_not_completed_from_acs_draws():
    raw = invented_raw()
    raw.loc[10, completion.FIELDS[0].asec] = ""
    result = completion.completed_columns(raw, draws(raw))
    assert pd.isna(result.loc[10, completion.FIELDS[0].output])
    assert not result.loc[10, completion.FIELDS[0].output + "__imputed"]


@pytest.mark.parametrize("option", [None, "true", 1, {}, []])
def test_host_option_refuses_before_parent_or_source_work(option):
    from microcosm.build.us_runtime import graph_us_survey_enrichment as host

    with pytest.raises(ValueError, match="HEALTH_COMPLETION_OPTION"):
        host.Boundary(
            object(), groups=("unemployment",), n_estimators=2, health_completion=option
        )


@pytest.mark.parametrize("option", [False, True])
@pytest.mark.parametrize(
    "entrypoint,callee",
    [("run_us_survey_enrichment", "_construct"), ("_construct", "Boundary")],
)
def test_health_option_is_forwarded_without_implicitly_enabling(
    monkeypatch, option, entrypoint, callee
):
    from microcosm.build.us_runtime import graph_us_survey_enrichment as host

    calls = []

    def record(parent, **kwargs):
        calls.append((parent, kwargs))
        raise RuntimeError("invented stop before parent/source work")

    monkeypatch.setattr(host, callee, record)
    parent = object()
    with pytest.raises(RuntimeError, match="invented stop"):
        getattr(host, entrypoint)(
            parent, groups=("unemployment",), n_estimators=2, health_completion=option
        )
    assert calls[0][0] is parent and calls[0][1]["health_completion"] is option


def test_default_health_declarations_and_option_are_unchanged():
    import inspect

    from test_us_current_survey_health_coverage import _qualified

    from microcosm.build.us_runtime import graph_us_survey_enrichment as host

    for function in (host.Boundary, host._construct, host.run_us_survey_enrichment):
        assert (
            inspect.signature(function).parameters["health_completion"].default is False
        )
    boundary = object.__new__(host.Boundary)
    boundary.health = _qualified()
    boundary.health_completion = None
    boundary.n_estimators = 2
    assert boundary._health_completion_nodes() == ()
    assert boundary._health_nodes() == host.health_graph.health_coverage_nodes(
        boundary.health,
        receiving_version=host.parent.attach.FILTER_NODE,
        after=host._amount_edge(),
    )


def test_detached_frame_cannot_supply_preparation_authority():
    from test_us_current_survey_health_coverage import _qualified

    observations = _qualified()
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        completion.qualify_current_survey_health_completion(
            observations.source_frame, observations
        )


def test_health_modules_have_explicit_dependency_and_provenance_classifications():
    import test_us_spine_blindness as guard

    for module in (
        "current_survey_health_completion.py",
        "graph_current_survey_health_completion.py",
    ):
        assert module in guard._US_LAUNCH_GRAPH_RUNTIME_MODULES
        assert (
            guard._non_owner_source_spine_accesses(
                module, (guard._US_RUNTIME / module).read_text()
            )
            == ()
        )
    module = "graph_current_survey_health_completion.py"
    assert module not in guard._SOURCE_SPINE_PROVENANCE_OWNERS
    assert guard._non_owner_source_spine_accesses(
        module, 'def wrong(frame):\n return frame.person["person_spine_source_id"]\n'
    )


def test_genuine_full_donor_is_independent_of_selected_support(tmp_path, monkeypatch):
    import json
    import shutil
    from fractions import Fraction

    from test_us_current_survey_health_coverage import _health_source_arguments

    from microcosm.frame import WeightKind

    arguments = _health_source_arguments(tmp_path, monkeypatch)
    partial = tmp_path / "partial-health-source"
    shutil.copytree(arguments["source_dir"], partial)
    request = json.loads((partial / "selection-request.json").read_bytes())
    request["fraction"], request["seed"] = [2, 3], 41
    (partial / "selection-request.json").write_text(
        json.dumps(request, sort_keys=True, separators=(",", ":"))
    )
    captures = tmp_path / "partial-captures"
    captures.mkdir()
    smaller = dict(
        source_dir=partial, snapshot_root=captures, fraction=Fraction(2, 3), seed=41
    )
    full = completion.source.prepare_authenticated_survey_population(**arguments)
    selected = completion.source.prepare_authenticated_survey_population(**smaller)
    retained = [
        (p, completion.source._frame_identity(p._checked()[2].frame))
        for p in (full, selected)
    ]
    results = []
    for prepared in (full, selected):
        observations = completion.literals.qualify_current_survey_health(prepared)
        result = completion.qualify_current_survey_health_completion(
            prepared, observations
        )
        assert result.source_frame.n("person") == result.donor_frame.n("person") == 4
        assert result.donor_frame.weights_for("household").kind is WeightKind.DESIGN
        assert result.donor_frame.weights_for("household").values.tolist() == [
            2552.12,
            100.0,
        ]
        assert result.source_frame.person[completion.FEATURES[1]].tolist() == [
            0.0,
            1.0,
            0.0,
            1.0,
        ]
        assert result.source_frame.person[completion.FEATURES[2]].tolist() == [
            6.0,
            6.0,
            36.0,
            36.0,
        ]
        assert result.evidence["temporal_equivalence_claim"] is False
        assert result.evidence["release_eligible"] is False
        results.append(result)
    for entity in results[0].source_frame.entities:
        pd.testing.assert_frame_equal(
            results[0].source_frame.table(entity),
            results[1].source_frame.table(entity),
            check_exact=True,
        )
    pd.testing.assert_frame_equal(
        results[0].columns, results[1].columns, check_exact=True
    )
    selected_asec = selected._checked()[2].native[1].frame.person.person_id
    assert len(selected_asec) == 2
    assert set(selected_asec) < set(results[1].source_frame.person.person_id)
    for prepared, stamp in retained:
        assert completion.source._frame_identity(prepared._checked()[2].frame) == stamp


def test_opt_in_graph_trains_applies_and_replays_original_draws(tmp_path):
    from types import SimpleNamespace

    from test_us_current_survey_health_coverage import (
        _frame,
        _qualified,
        invented_receiving,
    )

    from microcosm.build.us_runtime import graph_current_survey_health as observed
    from microcosm.build.us_runtime import (
        graph_current_survey_health_completion as graph,
    )
    from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
    from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
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
        compile_graph,
        run_graph,
    )
    from microcosm.graph import population as population_ops

    observations = _qualified()
    donor = _frame(donor_people())
    features = pd.DataFrame(
        [[30.0, 1.0, 6.0]],
        index=observations.origins.index[[1]],
        columns=completion.FEATURES,
    )
    qualified = completion.assemble(
        donor, features, evidence={"fixture": "invented; no source authority"}
    )
    people = invented_receiving().person
    channel = health.provenance.support_channel_column("person")
    people[channel] = pd.array(people[channel], dtype=health.STRING_DTYPE)
    receiving = _frame(people)
    attachment_type = ArtifactType("invented.health_parent", 1)

    class InventedSource(KernelBase):
        ref = "invented.health_completion_receiving@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def implementation_hash(self):
            return "b" * 64

        def run(self, context):
            return KernelResult(frame=receiving, artifacts={"attachment": b"invented"})

    structural = {
        receiving.schema.person_id_column,
        *(
            receiving.schema.membership_column(e)
            for e in receiving.schema.group_entities
        ),
    }
    create = Node(
        "invented.receiving",
        InventedSource.ref,
        sources=(observed.SOURCE_NAME,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned("person", c, observed._dtype(people[c]))
            for c in people
            if c not in structural
        ),
        artifact_outputs=(ArtifactOutput("attachment", attachment_type),),
    )
    after = ArtifactInput("amount_attachment", create.id, "attachment", attachment_type)
    baseline = observed.health_coverage_nodes(
        observations, receiving_version=create.id, after=after
    )
    added = graph.nodes(
        qualified,
        observations,
        receiving_version=create.id,
        after=after,
        n_estimators=2,
    )
    assert added[-1].id == baseline[-1].id
    assert all(
        n.kernel == "fit.qrf.legacy_target.train@1"
        for n in added
        if n.id.startswith(graph.FIT_PREFIX + ".")
    )
    compiled = compile_graph(
        Graph(
            "us",
            (SourceRef(observed.SOURCE_NAME, "raw-bytes-v1"),),
            (create, *baseline[:-1], *added),
        )
    )
    source_stamp, observation_stamp = (
        completion.seal(qualified),
        observed.health_coverage_seal(observations),
    )

    def pure():
        assert completion.seal(qualified) == source_stamp
        assert observed.health_coverage_seal(observations) == observation_stamp

    def context(value):
        pure()
        assert value.node in added

    boundary = SimpleNamespace(
        health_completion=qualified,
        health=observations,
        health_completion_nodes=added,
        pure=pure,
        context=context,
        compiled=compiled,
        n_estimators=2,
    )
    kernels = KernelRegistry()
    for kernel in (
        InventedSource(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
        *observed.health_coverage_kernels(
            observations, receiving_version=create.id, after=after, require_current=pure
        ),
        *graph.kernels(boundary),
    ):
        kernels.register(kernel)
    path = tmp_path / "invented.txt"
    path.write_bytes(b"invented health completion only")
    store = ContentStore(tmp_path / "store")
    args = dict(sources={observed.SOURCE_NAME: path}, store=store, kernels=kernels)
    cold = run_graph(compiled, **args)
    warm = run_graph(compiled, **args, resume="require")
    assert cold.key == warm.key and all(n.hit for n in warm.nodes.values())
    output = warm.population(create.id)
    pd.testing.assert_frame_equal(
        output.person[receiving.person.columns], receiving.person, check_exact=True
    )
    for field in completion.FIELDS:
        acs = output.person[channel].eq("acs")
        assert output.person.loc[acs, field.output].notna().all()
        assert output.person.loc[acs, field.output].nunique() == 1
        assert not output.person.loc[acs, field.output + "__known"].any()
        assert output.person.loc[acs, field.output + "__imputed"].all()
    loaded = {
        (node.id, name): store.load_bytes(key)
        for node in compiled.graph.nodes
        for name, key in warm.node(node.id).opaque_artifacts.items()
    }
    graph.verify_models(
        boundary,
        loaded,
        population_ops.Population.from_frame(
            warm.population(graph.DONOR_NODE), graph.DONOR_NODE
        ),
    )
    assert len([n for n in added if n.id.startswith(graph.FIT_PREFIX + ".")]) == 7
    assert len([n for n in added if n.id.startswith(graph.APPLY_PREFIX + ".")]) == 7
    artifacts = {
        edge.name: SimpleNamespace(
            payload=loaded[edge.producer, edge.artifact],
            producer_key=warm.node(edge.producer).key,
        )
        for edge in added[-1].artifact_inputs
    }
    original_draw = graph.read_draws(qualified, artifacts)
    completion.completed_columns(observations.raw, original_draw)
    original = artifacts["health_completion_matrix"]
    artifacts["health_completion_matrix"] = SimpleNamespace(
        payload=original.payload, producer_key="f" * 64
    )
    with pytest.raises(ValueError, match="DRAW_CHAIN"):
        graph.read_draws(qualified, artifacts)
    artifacts["health_completion_matrix"] = original
    wrong = dict(loaded)
    wrong[graph.FIT_PREFIX + ".001", "model"] = loaded[
        graph.FIT_PREFIX + ".000", "model"
    ]
    with pytest.raises(ValueError, match="TRAINING_DONOR"):
        graph.verify_models(
            boundary,
            wrong,
            population_ops.Population.from_frame(
                warm.population(graph.DONOR_NODE), graph.DONOR_NODE
            ),
        )
    pure()
