"""Exact survey demographic mappings and a tiny genuine source/clone graph."""

import hashlib
import io
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_race_hispanic_source as source
from microcosm.build.us_runtime import graph_current_survey_race_hispanic as graph
from microcosm.build.us_runtime import graph_us_survey_enrichment as host

# Explicit source combinations and expected public PRDTRACE categories, including
# the three residual triples and four/five-group bins from the Census dictionary.
COMBINATIONS = (
    ("W", 1),
    ("B", 2),
    ("I", 3),
    ("A", 4),
    ("H", 5),
    ("WB", 6),
    ("WI", 7),
    ("WA", 8),
    ("WH", 9),
    ("BI", 10),
    ("BA", 11),
    ("BH", 12),
    ("IA", 13),
    ("IH", 14),
    ("AH", 15),
    ("WBI", 16),
    ("WBA", 17),
    ("WBH", 18),
    ("WIA", 19),
    ("WIH", 20),
    ("WAH", 21),
    ("BIA", 22),
    ("BIH", 25),
    ("BAH", 25),
    ("IAH", 25),
    ("WBIA", 23),
    ("WBIH", 26),
    ("WBAH", 26),
    ("WIAH", 24),
    ("BIAH", 26),
    ("WBIAH", 26),
)


def acs_row(groups="W", **changes):
    row = {
        "RAC1P": str({"W": 1, "B": 2, "I": 3, "A": 6, "H": 7, "S": 8}.get(groups, 9)),
        "FRACP": "0",
        "HISP": "01",
        "FHISP": "0",
        "RACNUM": str(len(groups)),
    }
    for field, letter in zip(source.ACS_INDICATORS, "WBIAHHS", strict=True):
        row[field] = "1" if letter in groups else "0"
    row.update(changes)
    return row


def asec_row(**changes):
    row = dict(PRDTRACE="01", PXRACE1="00", PEHSPNON="2", PXHSPNON="00", PRDTHSP="0")
    row.update(changes)
    return row


@pytest.mark.parametrize("groups,expected", COMBINATIONS)
def test_every_supported_acs_combination(groups, expected):
    actual = source._map_row(acs_row(groups), "acs")
    assert actual[:2] == (expected, False)
    assert actual[4] == "exact_category_crosswalk"


@pytest.mark.parametrize("code", range(1, 27))
def test_every_asec_race_code_is_preserved(code):
    assert source._map_row(asec_row(PRDTRACE=f"{code:02d}"), "asec")[:2] == (
        code,
        False,
    )


@pytest.mark.parametrize("code", range(1, 25))
def test_acs_hispanic_all_named_categories(code):
    assert source._map_row(acs_row(HISP=f"{code:02d}"), "acs")[1] is (code != 1)


@pytest.mark.parametrize("code", range(1, 9))
def test_asec_hispanic_detail_all_named_categories(code):
    assert source._map_row(asec_row(PEHSPNON="1", PRDTHSP=str(code)), "asec")[1] is True


@pytest.mark.parametrize("groups", ("S", "WS", "BS", "WBS", "WBIAHS"))
def test_some_other_race_never_gets_representative_cps_code(groups):
    value = source._map_row(acs_row(groups), "acs")
    assert value[0] is None and value[4] == "no_exact_cps_category_some_other_race"


@pytest.mark.parametrize("changes", ({"RACNUM": "1"}, {"RAC1P": "1"}, {"RACWHT": "0"}))
def test_contradictory_detail_is_not_a_race_observation(changes):
    value = source._map_row(acs_row("WB", **changes), "acs")
    assert value[0] is None and value[4] == "inconsistent_race_detail"


@pytest.mark.parametrize("token", ("", "-1", "NA", "b", " 1", "1.0", "99"))
def test_unknown_item_tokens_remain_unknown(token):
    assert source._map_row(acs_row(HISP=token), "acs")[1] is None
    assert source._map_row(asec_row(PRDTRACE=token), "asec")[0] is None


@pytest.mark.parametrize(
    "flag", ("", "-1", "01", "02", "03", "04", "50", "51", "52", "53")
)
def test_unresolved_or_nonvalue_asec_allocation_preserves_unknown(flag):
    actual = source._map_row(asec_row(PXRACE1=flag, PXHSPNON=flag), "asec")
    assert actual[:2] == (None, None)
    prefix = (
        "conflicting" if flag in ("01", "02", "03", "50", "52", "53") else "unresolved"
    )
    assert actual[4:] == (prefix + "_race_allocation", prefix + "_hispanic_allocation")


@pytest.mark.parametrize("flag", ("01", "02", "03", "50", "52", "53"))
def test_missing_allocation_outcome_is_distinct_from_conflicting_value(flag):
    actual = source._map_row(
        asec_row(PRDTRACE="", PEHSPNON="", PRDTHSP="", PXRACE1=flag, PXHSPNON=flag),
        "asec",
    )
    assert actual[:2] == (None, None)
    assert actual[4:] == (
        "missing_race_per_allocation",
        "missing_hispanic_per_allocation",
    )


@pytest.mark.parametrize(
    "flag", (0, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43)
)
def test_named_value_allocation_retained_without_unallocated_claim(flag):
    actual = source._map_row(asec_row(PXRACE1=str(flag), PXHSPNON=str(flag)), "asec")
    assert actual[:2] == (1, False)
    assert actual[2:4] == (source.PX_CODES[flag], source.PX_CODES[flag])


def test_acs_allocated_and_nh_plus_pi_remain_one_major_group():
    actual = source._map_row(acs_row("H", FRACP="1", FHISP="1"), "acs")
    assert actual[:4] == (5, False, "allocated", "allocated")
    assert source._map_row(acs_row("H", RACNUM="2"), "acs")[0] is None
    for code in (3, 4, 5):
        assert source._map_row(acs_row("I", RAC1P=str(code)), "acs")[0] == 3
    for token in ("", "2", "-1"):
        assert source._map_row(acs_row(FRACP=token, FHISP=token), "acs")[:2] == (
            None,
            None,
        )


@pytest.mark.parametrize(
    "answer,detail", (("1", "0"), ("2", "1"), ("1", ""), ("", "0"), ("0", "0"))
)
def test_hispanic_universe_contradictions_are_not_false(answer, detail):
    assert source._map_row(asec_row(PEHSPNON=answer, PRDTHSP=detail), "asec")[1] is None


def test_hispanic_missing_detail_and_known_contradiction_have_distinct_status():
    assert (
        source._map_row(asec_row(PEHSPNON="1", PRDTHSP=""), "asec")[5]
        == "unavailable_hispanic_detail"
    )
    assert (
        source._map_row(asec_row(PEHSPNON="1", PRDTHSP="0"), "asec")[5]
        == "inconsistent_hispanic_universe"
    )
    assert (
        source._map_row(asec_row(PEHSPNON="", PRDTHSP="0"), "asec")[5]
        == "unresolved_hispanic_item"
    )


@pytest.mark.parametrize("value", (None, 0, 1, "true", object()))
def test_host_option_is_literal_boolean_before_parent_io(value):
    with pytest.raises(ValueError, match="RACE_HISPANIC_OPTION"):
        host.Boundary(object(), groups=(), n_estimators=2, race_hispanic_inputs=value)


@pytest.mark.parametrize(
    "name", ("CPS_COMBINATIONS", "PX_CODES", "ACS_FIELDS", "RAC1_SINGLE")
)
def test_mapping_or_capture_global_rebind_is_refused(monkeypatch, name):
    monkeypatch.setattr(source, name, {})
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        source.qualify_current_survey_race_hispanic(object())


def test_same_items_custom_mapping_cannot_change_lookup_authority(monkeypatch):
    class ChangedLookup(dict):
        def __getitem__(self, key):
            return 3

    changed = ChangedLookup(source.CPS_COMBINATIONS)
    assert tuple(changed.items()) == tuple(source.CPS_COMBINATIONS.items())
    assert changed["W"] != source.CPS_COMBINATIONS["W"]
    monkeypatch.setattr(source, "CPS_COMBINATIONS", changed)
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        source.qualify_current_survey_race_hispanic(object())
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        source.retained(object())


@pytest.mark.parametrize(
    "module,name",
    (
        (source.original, "_origins"),
        (source.original.asec, "_capture"),
        (source.original.housing, "_copy"),
        (source.literals, "literal_code"),
    ),
)
def test_executable_dependency_rebind_is_refused(monkeypatch, module, name):
    monkeypatch.setattr(module, name, lambda *a, **kw: None)
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        source.qualify_current_survey_race_hispanic(object())


def test_literal_scanner_preserves_blanks_and_leading_zeros_and_refuses_duplicate():
    columns = source.ASEC_COLUMNS
    row = dict(
        PERIDNUM="0" * 21 + "7", PH_SEQ="00007", A_LINENO="01", **asec_row(PRDTRACE="")
    )
    line = ",".join(row[c] for c in columns)
    body = (",".join(columns) + "\n" + line + "\n").encode()
    selected = {}
    key = source.original._key(row, "asec")
    assert (
        source._scan(
            io.BytesIO(body), survey="asec", wanted={key}, selected=selected, maximum=1
        )
        == 1
    )
    assert selected[key] == row
    with pytest.raises(ValueError, match="DUPLICATE_SELECTED_SOURCE_KEY"):
        source._scan(
            io.BytesIO(body + (line + "\n").encode()),
            survey="asec",
            wanted={key},
            selected={},
            maximum=2,
        )
    with pytest.raises(ValueError, match="SOURCE_HEADER"):
        source._scan(
            io.BytesIO(body.replace(b",PRDTRACE", b",unrelated")),
            survey="asec",
            wanted={key},
            selected={},
            maximum=1,
        )


@pytest.mark.parametrize(
    "spm,immigration,sex",
    [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)],
)
def test_optional_fragment_orders_after_existing_terminal(spm, immigration, sex):
    edge = host._race_after_edge(spm, immigration, sex)
    expected = (
        host.sex_graph.ATTACH_NODE
        if sex
        else host.immigration_graph.ATTACH_NODE
        if immigration
        else host.spm_graph.ATTACH_NODE
        if spm
        else host.hours_graph.ATTACH_NODE
    )
    assert edge.producer == expected and edge.artifact == "attachment"


def test_host_options_preserve_false_defaults_and_disabled_state():
    import inspect

    for function in (host.Boundary, host._construct, host.run_us_survey_enrichment):
        signature = inspect.signature(function)
        assert signature.parameters["race_hispanic_inputs"].default is False
        assert signature.parameters["demographic_inputs"].default is False
    state = SimpleNamespace(
        race_hispanic_inputs=False, race=None, race_stamp=None, race_nodes=()
    )
    host.Boundary._race_pure(state)
    state.race_nodes = (object(),)
    with pytest.raises(ValueError, match="RACE_HISPANIC_STATE_CHANGED"):
        host.Boundary._race_pure(state)


def source_arguments(tmp_path, monkeypatch):
    import test_us_survey_population_preparation as fixture_module
    from test_us_survey_population_preparation import fixture

    from microcosm.build.us_runtime import asec_person_income_source as restoration

    original_person = fixture_module._person

    def person(*args, **kwargs):
        row = original_person(*args, **kwargs)
        row.update(
            acs_row("WB" if int(row["SPORDER"]) == 1 else "S", HISP="02", FHISP="1")
        )
        return row

    monkeypatch.setattr(fixture_module, "_person", person)
    arguments = fixture(tmp_path, monkeypatch, zero=True)
    folder = arguments["source_dir"] / "asec"
    pins, paths = [], {}
    for year, member, archive, *_ in source.original.asec._MEMBER_PINS:
        path = folder / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        for name, value in asec_row().items():
            raw[name] = value
        raw.loc[raw.A_LINENO.eq("2"), "PRDTRACE"] = ""
        raw.loc[raw.A_LINENO.eq("1"), "PXRACE1"] = "41"
        raw.iloc[::-1].to_csv(path, index=False)
        data = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(data).hexdigest(),
                len(raw),
                len(data),
            )
        )
        paths[year] = path
    for module in (source.original.asec, restoration):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "race-restored-money"
    restoration.restore_asec_person_income_source(
        folder / "parent.h5",
        folder / "household-attachment.h5",
        member_paths=paths,
        output_dir=output,
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, folder / "person-income-attachment.h5"
    )
    return arguments


def test_genuine_preparation_and_small_clone_graph_cold_required(tmp_path, monkeypatch):
    """One real invented-source preparation; receiving graph is fixture-only.

    This proves native source borrowing, graph execution and clone transport,
    not a full51/PUF/enrichment host. No source issuer guard is replaced.
    """
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
    arguments = source_arguments(source_root, monkeypatch)
    preparation = source.source.prepare_authenticated_survey_population(**arguments)
    qualified = source.qualify_current_survey_race_hispanic(preparation)
    stamp = source.seal(qualified)
    assert source.recode(qualified.raw).cps_race.isna().any()
    assert qualified.raw.survey_demographic_race_allocation.eq(
        "blank_to_allocated_value"
    ).any()
    assert qualified.raw.survey_demographic_source.eq("acs").any()
    assert qualified.raw.survey_demographic_observation_year.eq(2025).any()
    assert qualified.raw.survey_demographic_observation_year.eq(2024).any()
    people = receiving_people(qualified)
    people["unrelated_value"] = 7.5
    receiving = _frame(people)
    original = receiving.person.copy(deep=True)
    attachment_type = ArtifactType("invented.race_hispanic_parent", 1)

    class Receiving(KernelBase):
        ref = "invented.race_hispanic_receiving@1"
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
    nodes = graph.race_hispanic_nodes(
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
    for kernel in graph.race_hispanic_kernels(
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
        "Invented receiving graph; source race_hispanic comes from retained genuine fixture preparation."
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
        output.person.cps_race.isna().sum()
        == 2 * source.recode(qualified.raw).cps_race.isna().sum()
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
        result = graph.race_hispanic_result(
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
            altered["race_hispanic_canonical"] = replace(
                artifacts["race_hispanic_canonical"], payload=b"{}"
            )
            with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD"):
                graph.race_hispanic_result(
                    qualified, node, altered, incoming.frame.person
                )
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
        graph.race_hispanic_nodes(
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
                0, qualified.raw.columns.get_loc("survey_demographic_PRDTRACE")
            ] = "26"

    kernel = graph._RaceHispanicKernel(
        qualified, nodes, final_change, context_check, bind.kernel
    )
    # Stored nullable integers may normalize bytes below a null mask. The bind
    # accepts that codec behavior, but still checks masks and all known bytes.
    readback_people = warm.population(graph.SOURCE_NODE).person
    replayed = graph.race_hispanic_result(qualified, bind, artifacts, readback_people)
    pd.testing.assert_series_equal(
        replayed.columns["person", "cps_race"], source.recode(qualified.raw).cps_race
    )
    changed_known = readback_people.copy(deep=True)
    changed_known.loc[changed_known.index[0], "survey_demographic_PRDTRACE"] = "26"
    with pytest.raises(ValueError, match="BIND_SOURCE_SLICE"):
        graph.race_hispanic_result(qualified, bind, artifacts, changed_known)
    changed_mask = readback_people.copy(deep=True)
    changed_mask.loc[
        changed_mask.survey_demographic_PRDTRACE.isna(), "survey_demographic_PRDTRACE"
    ] = "1"
    with pytest.raises(ValueError, match="BIND_SOURCE_SLICE"):
        graph.race_hispanic_result(qualified, bind, artifacts, changed_mask)
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
        qualified.raw["survey_demographic_PRDTRACE"] = (
            saved.survey_demographic_PRDTRACE.copy()
        )
    assert len(calls) == 2 and source.seal(qualified) == stamp
