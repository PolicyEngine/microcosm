"""Invented declarations and actual authenticated-source graph regressions."""

import json
import os
import sys
import time
from dataclasses import replace
from fractions import Fraction

import pytest

from microcosm.build.us_runtime import graph_survey_population as graph
from microcosm.build.us_runtime.survey_catalogue_selection import (
    CatalogueSelectionPlan,
    SelectedHousehold,
)
from microcosm.build.us_runtime.survey_population_domains import (
    Domain,
    HouseholdKey,
    Source,
)
from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    Graph,
    Owned,
    Ownership,
    SourceRef,
    StructuralDelta,
    compile_graph,
)


def columns():
    return tuple(
        Owned(entity, name, "string" if name.endswith("_support_channel") else "int64")
        for entity in US_SCHEMA.entities
        for name in graph._provenance_columns(entity)
    ) + (Owned("person", "age", "int64"),)


def nodes(**changes):
    return graph.survey_population_nodes(
        changes.pop("columns", columns()),
        **{
            "preparation_sha256": "a" * 64,
            "fraction": Fraction(2, 7),
            "seed": 41,
            **changes,
        },
    )


def test_create_then_reweight_has_no_synthetic_population_filter():
    create, allocate = nodes()
    assert [n.structural for n in (create, allocate)] == [
        StructuralDelta.CREATE,
        StructuralDelta.REWEIGHT,
    ]
    assert create.base is None and create.population is None
    assert create.inputs == create.artifact_inputs == ()
    assert create.sources == (graph.SOURCE_NAME,)
    assert create.params["fraction"] == (2, 7)
    assert allocate.base == create.id
    assert allocate.weights.entity == "household"
    assert allocate.weights.to_kind == "importance"
    assert allocate.mass == "declared"
    assert {a.producer for a in allocate.artifact_inputs} == {create.id}
    assert {a.name for a in create.artifact_outputs} == {"frame_context", "preparation"}
    assert allocate.outputs == ()
    compiled = compile_graph(
        Graph(
            "us",
            (SourceRef(graph.SOURCE_NAME, graph.SOURCE_CODEC),),
            (create, allocate),
        )
    )
    assert compiled.order == (create.id, allocate.id)


@pytest.mark.parametrize(
    "digest", ["a" * 63, "a" * 65, "A" * 64, "g" * 64, b"a" * 64, None]
)
def test_bad_preparation_identity_refuses(digest):
    with pytest.raises(graph.SurveyPopulationGraphError, match="DIGEST"):
        nodes(preparation_sha256=digest)


@pytest.mark.parametrize("fraction", [1, True, 0.5, Fraction(0), Fraction(3, 2)])
def test_draw_fraction_is_exact_positive_rational(fraction):
    with pytest.raises(graph.SurveyPopulationGraphError, match="FRACTION"):
        nodes(fraction=fraction)


@pytest.mark.parametrize("seed", [True, -1, 2**64, "41"])
def test_draw_seed_is_uint64(seed):
    with pytest.raises(graph.SurveyPopulationGraphError, match="SEED"):
        nodes(seed=seed)


def test_declarations_snapshot_column_sequence():
    original = list(columns())
    create, _ = nodes(columns=original)
    original.pop()
    assert create.outputs == columns()


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "dtype", "rewrite", "mask", "absent", "entity"]
)
def test_exact_create_provenance_and_owned_contract(change):
    values = list(columns())
    if change == "missing":
        values.pop(0)
    elif change == "duplicate":
        values.append(values[0])
    elif change == "dtype":
        values[0] = replace(values[0], dtype="float64")
    elif change == "rewrite":
        values[-1] = replace(values[-1], rewrite=True)
    elif change == "mask":
        values[-1] = replace(values[-1], rows="some_rows")
    elif change == "absent":
        values[-1] = replace(values[-1], ownership=Ownership.ABSENT)
    else:
        values[-1] = replace(values[-1], entity="other")
    with pytest.raises(graph.SurveyPopulationGraphError):
        nodes(columns=values)


def plan_and_origins():
    selected = tuple(
        SelectedHousehold(
            HouseholdKey(source, 2024, 2024 if source is Source.ACS else 2025, native),
            Domain.SHARED_HOUSING,
            "household",
            anchor,
            Fraction(1, 2),
            probability,
        )
        for source, native, anchor, probability in (
            (Source.ACS, "000ACS", Fraction(13), Fraction(2, 7)),
            (Source.ASEC, "00007", Fraction(255212, 100), Fraction(1, 3)),
            (Source.ASEC, "00009", Fraction(0), Fraction(1, 3)),
        )
    )
    plan = CatalogueSelectionPlan(selected, (), (), 13, Fraction(2, 7), 41)
    origins = [
        {
            "household_id": 100 + i,
            "source": row.key.source.value,
            "source_year": row.key.source_year,
            "survey_year": row.key.survey_year,
            "raw_native_id": row.key.native_id,
            "selected_receiving_household_id": i % 2,
            "original_anchor": [
                row.original_design_weight.numerator,
                row.original_design_weight.denominator,
            ],
        }
        for i, row in enumerate(selected)
    ]
    return plan, origins


def test_exact_allocation_join_preserves_lexemes_anchors_zero_and_receiving_order():
    plan, origins = plan_and_origins()
    result = graph.allocation_instructions(plan, origins[::-1])
    assert [r.household_id for r in result] == [102, 101, 100]
    assert [r.key.native_id for r in result] == ["00009", "00007", "000ACS"]
    assert [r.importance_weight for r in result] == [
        Fraction(0),
        Fraction(191409, 50),
        Fraction(91, 4),
    ]
    assert result[1].original_anchor == Fraction(63803, 25)
    assert result[0].selected_receiving_household_id == 0
    origins[1]["original_anchor"][0] = 1
    assert result[1].original_anchor == Fraction(63803, 25)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "bool_id",
        "bool_year",
        "bool_receiving",
        "negative_id",
        "large_id",
        "float_anchor",
        "noncanonical_anchor",
        "wrong_anchor",
        "source",
        "native_lexeme",
        "period",
        "duplicate_id",
        "duplicate_native",
        "duplicate_receiving",
    ],
)
def test_unbound_or_inconsistent_allocation_origins_refuse(change):
    plan, origins = plan_and_origins()
    row = origins[1]
    if change == "missing":
        row.pop("original_anchor")
    elif change == "extra":
        row["claimed_authenticated"] = True
    elif change == "bool_id":
        row["household_id"] = True
    elif change == "bool_year":
        row["source_year"] = True
    elif change == "bool_receiving":
        row["selected_receiving_household_id"] = True
    elif change == "negative_id":
        row["household_id"] = -1
    elif change == "large_id":
        row["household_id"] = 2**63
    elif change == "float_anchor":
        row["original_anchor"] = [63803.0, 25]
    elif change == "noncanonical_anchor":
        row["original_anchor"] = [255212, 100]
    elif change == "wrong_anchor":
        row["original_anchor"] = [7, 1]
    elif change == "source":
        row["source"] = "puf"
    elif change == "native_lexeme":
        row["raw_native_id"] = "7"
    elif change == "period":
        row["survey_year"] = 2024
    elif change == "duplicate_id":
        row["household_id"] = origins[0]["household_id"]
    elif change == "duplicate_native":
        origins[2] = dict(row)
    else:
        row["selected_receiving_household_id"] = origins[2][
            "selected_receiving_household_id"
        ]
    with pytest.raises(graph.SurveyPopulationGraphError):
        graph.allocation_instructions(plan, origins)


@pytest.mark.parametrize(
    "change",
    ["empty", "count", "duplicate", "bad_share", "bad_probability", "bool_anchor"],
)
def test_allocation_plan_value_contract(change):
    plan, origins = plan_and_origins()
    if change == "empty":
        plan = replace(plan, selected=())
    elif change == "count":
        origins.pop()
    elif change == "duplicate":
        plan = replace(plan, selected=(plan.selected[0],) * 3)
    else:
        altered = replace(
            plan.selected[0],
            **{
                "bad_share": {"share": Fraction(0)},
                "bad_probability": {"inclusion_probability": Fraction(2)},
                "bool_anchor": {"original_design_weight": True},
            }[change],
        )
        plan = replace(plan, selected=(altered, *plan.selected[1:]))
    with pytest.raises(graph.SurveyPopulationGraphError):
        graph.allocation_instructions(plan, origins)


def test_allocation_bound_precedes_map_construction(monkeypatch):
    plan, origins = plan_and_origins()
    monkeypatch.setattr(graph, "ALLOCATION_MAX_BYTES", 128)
    with pytest.raises(graph.SurveyPopulationGraphError, match="ALLOCATION_LIMIT"):
        graph.allocation_instructions(plan, origins)


@pytest.mark.parametrize(
    "value", [Fraction(10**400), Fraction(1, 10**400), Fraction(-1)]
)
def test_numeric_conversion_refuses_overflow_underflow_or_negative(value):
    with pytest.raises(graph.SurveyPopulationGraphError, match="FLOAT_WEIGHT"):
        graph._finite_weight(value)


def test_bounded_streaming_transport_exact_boundary_and_unicode():
    raw = graph._bounded_json({"z": "é", "a": [0, 1]}, 64)
    assert raw == '{"a":[0,1],"z":"é"}'.encode()
    assert graph._bounded_json({"z": "é", "a": [0, 1]}, len(raw)) == raw
    with pytest.raises(graph.SurveyPopulationGraphError):
        graph._bounded_json({"z": "é", "a": [0, 1]}, len(raw) - 1)


def test_allocation_transport_streams_canonical_exact_rows_and_refuses_before_overflow(
    monkeypatch,
):
    plan, origins = plan_and_origins()
    instructions = graph.allocation_instructions(plan, origins)
    kwargs = {
        "preparation_sha256": "b" * 64,
        "input_context": b"{}",
        "output_context": b"[]",
    }
    raw = graph._allocation_payload(instructions, **kwargs)
    document = json.loads(raw)
    assert raw == graph._bounded_json(document, len(raw))
    assert document["weight_transition"] == ["design", "importance"]
    assert document["release_eligible"] is False
    assert document["households"][1]["original_anchor"] == [63803, 25]
    assert document["households"][1]["importance_multiplier"] == [3, 2]
    assert document["households"][2]["importance_weight_float64_hex"] == "0x0.0p+0"
    monkeypatch.setattr(graph, "ALLOCATION_MAX_BYTES", len(raw))
    assert graph._allocation_payload(instructions, **kwargs) == raw
    monkeypatch.setattr(graph, "ALLOCATION_MAX_BYTES", len(raw) - 1)
    with pytest.raises(graph.SurveyPopulationGraphError, match="ALLOCATION_LIMIT"):
        graph._allocation_payload(instructions, **kwargs)


def authenticated_arguments(tmp_path, monkeypatch, **fixture_kwargs):
    # This helper creates closed, privately pinned invented source bytes and
    # executes the actual ASEC parent, both catalogues and both native issuers.
    from test_us_survey_population_preparation import fixture

    arguments = fixture(tmp_path, monkeypatch, **fixture_kwargs)
    return {**arguments, "store_root": tmp_path / "graph-store"}


def test_authenticated_standalone_codec_executes_real_preparation(
    tmp_path, monkeypatch
):
    from microcosm.frame import WeightKind

    arguments = authenticated_arguments(tmp_path, monkeypatch)
    codecs = graph.survey_population_source_codecs(
        snapshot_root=arguments["snapshot_root"]
    )
    frame = codecs.load(graph.SOURCE_CODEC, arguments["source_dir"])
    assert frame.n("household") == 6 and frame.n("person") == 9
    assert frame.weights_for("household").kind is WeightKind.DESIGN
    assert set(frame.table("household")[graph.support_channel_column("household")]) == {
        "acs",
        "asec",
    }


def test_authenticated_cold_and_materialized_warm_clones_both_sources(
    tmp_path, monkeypatch
):
    from microcosm.build.us_runtime import graph_combined_clone as clone
    from microcosm.build.us_runtime import survey_population_preparation as source
    from microcosm.frame import WeightKind
    from microcosm.graph import graph_to_json
    from microcosm.graph.executor import _source_paths_and_keys

    arguments = authenticated_arguments(tmp_path, monkeypatch)
    codes = {
        graph.SurveyPopulationCreateKernel.run.__code__: "create",
        graph.SurveyPopulationAllocationKernel.run.__code__: "allocate",
        clone.USCombinedSurveyCloneExpandKernel.run.__code__: "clone",
        clone.USCombinedSurveyCloneClaimKernel.run.__code__: "claim",
        source.prepare_authenticated_survey_population.__code__: "prepare",
    }
    tracked = {
        **{id(code): name for code, name in codes.items()},
        id(_source_paths_and_keys.__code__): "source_key",
        id(graph._final_artifact.__code__): "final_artifact",
    }
    calls = []
    key_passes, starts, preparation_times, actual_graphs, payload_sizes = (
        [],
        {},
        [],
        [],
        {},
    )

    def profile(frame, event, arg):
        if event not in {"call", "return"}:
            return
        kind = tracked.get(id(frame.f_code))
        if kind is None:
            return
        if event == "call" and kind in {
            "create",
            "allocate",
            "clone",
            "claim",
            "prepare",
        }:
            calls.append(kind)
        if kind == "prepare":
            if event == "call":
                starts[id(frame)] = time.perf_counter()
            elif event == "return":
                preparation_times.append(time.perf_counter() - starts.pop(id(frame)))
        if kind == "source_key":
            if event == "call":
                starts[id(frame)] = time.perf_counter()
                actual_graphs.append(frame.f_locals["compiled"])
            elif event == "return":
                key_passes.append(time.perf_counter() - starts.pop(id(frame)))
        if event == "call" and kind == "final_artifact":
            name = f"{frame.f_locals['node_id']}.{frame.f_locals['name']}"
            size = len(frame.f_locals["payload"])
            assert payload_sizes.get(name, size) == size
            payload_sizes[name] = size

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        relative = {
            name: os.path.relpath(value) if name.endswith(("_dir", "_root")) else value
            for name, value in arguments.items()
        }
        # A real cold CREATE must also normalize a parent component, after
        # authenticated source validation has had the raw path to inspect.
        (tmp_path / "relative-entry").mkdir()
        relative["source_dir"] = (
            os.path.relpath(tmp_path / "relative-entry")
            + "/../"
            + os.path.relpath(arguments["source_dir"], tmp_path)
        )
        cold_started = time.perf_counter()
        cold = graph.run_authenticated_survey_population(**relative)
        cold_seconds = time.perf_counter() - cold_started
        boundary = len(calls)
        warm_started = time.perf_counter()
        warm = graph.run_authenticated_survey_population(**arguments, resume="require")
        warm_seconds = time.perf_counter() - warm_started
    finally:
        sys.setprofile(previous)
    assert calls[:boundary].count("prepare") == 1
    assert {
        name: calls[:boundary].count(name)
        for name in ("create", "allocate", "clone", "claim")
    } == dict.fromkeys(("create", "allocate", "clone", "claim"), 1)
    assert calls[boundary:] == ["prepare"]
    assert all(row.store_hit for row in warm.nodes.values())
    assert cold.key == warm.key
    created = warm.population(graph.CREATE_NODE)
    allocated = warm.population(graph.ALLOCATION_NODE)
    cloned = warm.population(clone.COMBINED_CLONE_NODE)
    assert created.n("household") == allocated.n("household") == 6
    assert cloned.n("household") == 12 and cloned.n("person") == 18
    assert created.weights_for("household").kind is WeightKind.DESIGN
    assert allocated.weights_for("household").kind is WeightKind.IMPORTANCE
    assert set(
        cloned.table("household")[graph.support_channel_column("household")]
    ) == {"acs", "asec"}
    graph._same_frame(cold.population(clone.COMBINED_CLONE_NODE), cloned)
    assert len(key_passes) == 4  # independent expectation + executor, cold + warm
    assert len(preparation_times) == 2
    evidence = tmp_path / "actual-graph-evidence"
    evidence.mkdir()
    graph_payloads = [
        graph_to_json(value.graph).encode("utf-8") for value in actual_graphs
    ]
    assert len(graph_payloads) == 4 and len(set(graph_payloads)) == 1
    assert len(graph_payloads[0]) <= 2 * 1024**2
    (evidence / "compiled-graph.json").write_bytes(graph_payloads[0])
    manifest_pins = {}
    for label, manifest in (("cold", cold), ("warm", warm)):
        payload = manifest.to_json_bytes()
        assert len(payload) <= 2 * 1024**2  # fixed invented six-household fixture
        (evidence / f"{label}-manifest.json").write_bytes(payload)
        manifest_pins[label] = {"sha256": graph._sha(payload), "key": manifest.key}
    (evidence / "execution.json").write_text(
        json.dumps(
            {
                "candidate": "invented6HH",
                "release_eligible": False,
                "calibration_claim": False,
                "calibration": "not_run",
                "graph_export": "actual graph_to_json serializer over captured compiled graph",
                "graph_sha256": graph._sha(graph_payloads[0]),
                "manifest_key": warm.key,
                "manifests": manifest_pins,
                "compiled_order": actual_graphs[0].order,
                "compiled_versions": dict(actual_graphs[0].versions),
                "payload_bytes": payload_sizes,
                "created_households": created.n("household"),
                "created_persons": created.n("person"),
                "cloned_households": cloned.n("household"),
                "cloned_persons": cloned.n("person"),
                "cold_calls": calls[:boundary],
                "warm_calls": calls[boundary:],
                "kernel_calls": {"cold": calls[:boundary], "warm": calls[boundary:]},
                "counts": {
                    "created": {
                        "households": created.n("household"),
                        "persons": created.n("person"),
                    },
                    "cloned": {
                        "households": cloned.n("household"),
                        "persons": cloned.n("person"),
                    },
                },
                "source_key_pass_seconds": key_passes,
                "cold_seconds": cold_seconds,
                "warm_seconds": warm_seconds,
                "top_level_preparation_seconds": preparation_times,
                "warm_scope": "zero graph kernels; fresh source reconstruction",
                "cold_paths": "relative source, snapshot and store paths",
                "warm_paths": "equivalent absolute paths",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("field", ["owners", "mass_ledger"])
def test_authenticated_warm_population_state_mutation_refuses_before_downstream(
    tmp_path, monkeypatch, field
):
    arguments = authenticated_arguments(tmp_path, monkeypatch)
    graph.run_authenticated_survey_population(**arguments, clones=False)
    injected = []

    def profile(frame, event, arg):
        if (
            event == "call"
            and frame.f_code.co_filename == graph.__file__
            and frame.f_code.co_name == "observe"
            and frame.f_locals["node_id"] == graph.ALLOCATION_NODE
        ):
            population = frame.f_locals["population"]
            if field == "owners":
                changed = dict(population.owners)
                changed[("person", "person_id")] = "invented.wrong-owner"
            else:
                changed = ()
            object.__setattr__(population, field, changed)
            injected.append(field)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            graph.SurveyPopulationGraphError,
            match="POPULATION_OWNERS|POPULATION_MASS_LEDGER",
        ):
            graph.run_authenticated_survey_population(
                **arguments, clones=False, resume="require"
            )
    finally:
        sys.setprofile(previous)
    assert injected == [field]


def test_authenticated_late_store_io_frame_mutation_refuses(tmp_path, monkeypatch):
    arguments = authenticated_arguments(tmp_path, monkeypatch)
    populations, injected = {}, []

    def profile(frame, event, arg):
        if frame.f_code.co_filename != graph.__file__:
            return
        if event == "call" and frame.f_code.co_name == "observe":
            populations[frame.f_locals["node_id"]] = frame.f_locals["population"]
        if (
            event == "return"
            and frame.f_code.co_name == "_final_artifact"
            and frame.f_locals["node_id"] == graph.ALLOCATION_NODE
            and frame.f_locals["name"] == "frame_context"
        ):
            table = populations[graph.ALLOCATION_NODE].frame.person
            table.loc[table.index[0], "age"] += 1
            injected.append("after-last-artifact-load")

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            graph.SurveyPopulationGraphError, match="MATERIALIZED_FRAME_VALUES"
        ):
            graph.run_authenticated_survey_population(**arguments, clones=False)
    finally:
        sys.setprofile(previous)
    assert injected == ["after-last-artifact-load"]


def test_authenticated_actual_allocation_context_and_rehashed_claim_refusals(
    tmp_path, monkeypatch
):
    from microcosm.graph import ArtifactType
    from microcosm.graph.keys import opaque_artifact_key

    arguments = authenticated_arguments(tmp_path, monkeypatch)
    captured = []
    code = graph.SurveyPopulationAllocationKernel.run.__code__

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is code:
            captured.append((frame.f_locals["self"], frame.f_locals["context"]))

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        graph.run_authenticated_survey_population(**arguments, clones=False)
    finally:
        sys.setprofile(previous)
    ((kernel, context),) = captured
    assert kernel.run(context).weights.kind.value == "importance"
    original = context.artifacts["preparation"]
    changed_payload = json.loads(original.payload)
    changed_payload["request"]["seed"] += 1
    forged = graph._bounded_json(changed_payload, graph.PREPARATION_MAX_BYTES)
    for changed in (
        replace(
            original, type=ArtifactType("microcosm.us.survey_population_preparation", 1)
        ),
        replace(original, payload=forged),
        replace(original, key="c" * 64),
        replace(
            original,
            producer_key="c" * 64,
            key=opaque_artifact_key("c" * 64, "preparation"),
        ),
    ):
        artifacts = {**context.artifacts, "preparation": changed}
        with pytest.raises(graph.SurveyPopulationGraphError, match="ARTIFACT"):
            kernel.run(replace(context, artifacts=artifacts))
    tables = {name: table.copy(deep=True) for name, table in context.tables.items()}
    household = tables["household"]
    household.loc[household.index[0], graph.spine_source_id_column("household")] += 1
    with pytest.raises(graph.SurveyPopulationGraphError, match="RECEIVING_ID"):
        kernel.run(replace(context, tables=tables))


@pytest.mark.parametrize("mutation", ["seed", "receipt"])
def test_authenticated_late_manifest_seed_mutation_refuses(
    tmp_path, monkeypatch, mutation
):
    arguments = authenticated_arguments(tmp_path, monkeypatch)
    injected = []

    def profile(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is graph._final_artifact.__code__
            and frame.f_locals["node_id"] == graph.ALLOCATION_NODE
            and frame.f_locals["name"] == "frame_context"
        ):
            node = frame.f_locals["manifest"].node(graph.ALLOCATION_NODE)
            if mutation == "seed":
                object.__setattr__(node, "seed", node.seed + 1)
            else:
                object.__setattr__(
                    node, "receipt", {**node.receipt, "allocation_sha256": "c" * 64}
                )
                created = frame.f_locals["manifest"].node(graph.CREATE_NODE)
                object.__setattr__(
                    created, "receipt", {**created.receipt, "release_eligible": True}
                )
            injected.append("after-last-artifact-load")

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            graph.SurveyPopulationGraphError, match="MANIFEST_NODE_STATE"
        ):
            graph.run_authenticated_survey_population(**arguments, clones=False)
    finally:
        sys.setprofile(previous)
    assert injected == ["after-last-artifact-load"]


def test_authenticated_independent_manifest_keys_and_all_descriptors(
    tmp_path, monkeypatch
):
    from microcosm.graph import executor
    from microcosm.graph.artifact_edges import descriptor

    arguments = authenticated_arguments(tmp_path, monkeypatch)
    captured, injected = [], []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is graph._check_node_states.__code__:
            captured.append((frame.f_locals["manifest"], frame.f_locals["expected"]))
        if event == "return" and frame.f_code is executor.run_graph.__code__:
            original = arg.node(graph.CREATE_NODE)
            changed = {
                **original.typed_artifacts,
                "outputs": {
                    name: descriptor(
                        producer=graph.CREATE_NODE,
                        artifact=name,
                        type_=graph.PREPARATION_TYPE
                        if name == "preparation"
                        else graph.US_FRAME_CONTEXT_TYPE,
                        producer_key="c" * 64,
                        capabilities=original.capabilities,
                    )
                    for name in original.typed_artifacts["outputs"]
                },
            }
            object.__setattr__(original, "key", "c" * 64)
            object.__setattr__(original, "typed_artifacts", changed)
            injected.append("before-country-runner-receives-manifest")

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            graph.SurveyPopulationGraphError, match="MANIFEST_NODE_STATE"
        ):
            graph.run_authenticated_survey_population(**arguments, clones=False)
    finally:
        sys.setprofile(previous)
    assert injected == ["before-country-runner-receives-manifest"]
    ((manifest, expected),) = captured
    node = manifest.node(graph.CREATE_NODE)
    state = expected[graph.CREATE_NODE]
    object.__setattr__(node, "key", state["key"])
    object.__setattr__(node, "typed_artifacts", state["typed_artifacts"])
    graph._check_node_states(manifest, expected)
    # The actual capability object is shared with its kernel. Expectations
    # must retain independent primitive values, not the same frozen object.
    from microcosm.graph import Numeric

    capabilities = node.capabilities
    original_numeric = capabilities.numeric
    object.__setattr__(capabilities, "numeric", Numeric.BITWISE)
    try:
        with pytest.raises(
            graph.SurveyPopulationGraphError, match="MANIFEST_NODE_STATE"
        ):
            graph._check_node_states(manifest, expected)
    finally:
        object.__setattr__(capabilities, "numeric", original_numeric)
    graph._check_node_states(manifest, expected)
    for field, bad in (
        ("seed", state["seed"] + 1),
        ("frame_key", "c" * 64),
        ("weight_key", "c" * 64),
        ("artifacts", {}),
        ("opaque_artifacts", {}),
        ("kernel_ref", "wrong@1"),
        ("kernel_impl_hash", "c" * 64),
        ("capabilities", {}),
        ("typed_artifacts", {}),
        ("legacy_capabilities", True),
    ):
        original = getattr(node, field)
        object.__setattr__(node, field, bad)
        try:
            with pytest.raises(
                graph.SurveyPopulationGraphError, match="MANIFEST_NODE_STATE"
            ):
                graph._check_node_states(manifest, expected)
        finally:
            object.__setattr__(node, field, original)
    graph._check_node_states(manifest, expected)
