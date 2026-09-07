"""Real geography operator values, native anchors and typed gate replay."""

import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.congressional_district_vintage import (
    default_congressional_district_vintage_crosswalk_path,
    load_default_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.graph_context import (
    US_FRAME_CONTEXT_TYPE,
    encode_us_frame_context,
)
from microcosm.build.us_runtime.graph_geography import (
    GEOGRAPHY_PHASE,
    LOOKUP_SOURCES,
    US_CD_CROSSWALK_TYPE,
    US_PUMA_LOOKUP_TYPE,
    USGeographyAssignKernel,
    USGeographyBoundaryKernel,
    USGeographyGateKernel,
    USGeographyLookupKernel,
    _lookups,
    register_us_geography_kernels,
    us_geography_nodes,
)
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.build.us_runtime.puma_ladder import (
    decode_us_puma_ladder,
    with_household_us_puma_ladder,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactOutput,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    NumericScope,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import load_source
from microcosm.graph.population import dtype_for_token


def _frame():
    ids = np.arange(1600, dtype=np.int64)
    tables = {
        entity: pd.DataFrame(
            {
                US_SCHEMA.entity_id_column(entity): ids,
                f"{entity}_marker": np.ones(len(ids), dtype=np.int64),
            }
        )
        for entity in US_SCHEMA.entities
    }
    for group in US_SCHEMA.group_entities:
        tables["person"][US_SCHEMA.membership_column(group)] = ids
    tables["person"]["age"] = np.full(len(ids), 40.0)
    tables["household"]["state_fips"] = np.r_[np.full(1500, 6), np.full(100, 36)]
    # Some ACS-native anchors and absent ASEC anchors coexist.
    tables["household"]["puma"] = pd.Series(
        ["0600100"] * 750 + [pd.NA] * 850, dtype=dtype_for_token("string")
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
        pd.Series(["source_2024"] * len(ids)),
    )


def _lookup_files(tmp_path, *, bad_vintage=False):
    crosswalk = load_default_congressional_district_vintage_crosswalk()
    districts = sorted(
        int(value[-4:]) for value in crosswalk.target_geography_id.unique()
    )
    states = sorted({value // 100 for value in districts})
    pumas = [state * 100000 + 100 for state in states]
    populations = [
        float(sum(cd // 100 == state for cd in districts)) for state in states
    ]
    county_puma, county_value, county_pop = [], [], []
    for state, puma, population in zip(states, pumas, populations, strict=True):
        if state == 36:
            county_puma.extend([puma, puma])
            county_value.extend([36061, 36001])
            county_pop.extend([population * 0.4, population * 0.6])
        else:
            county_puma.append(puma)
            county_value.append(state * 1000 + 1)
            county_pop.append(population)
    arrays = {
        "puma": np.array(pumas, dtype=np.int64),
        "puma_population": np.array(populations),
        "cd_overlap_puma": np.array(
            [(cd // 100) * 100000 + 100 for cd in districts], dtype=np.int64
        ),
        "cd_overlap_cd": np.array(districts, dtype=np.int64),
        "cd_overlap_population": np.ones(len(districts)),
        "county_overlap_puma": np.array(county_puma, dtype=np.int64),
        "county_overlap_county": np.array(county_value, dtype=np.int64),
        "county_overlap_population": np.array(county_pop),
        "tract_overlap_puma": np.array(county_puma, dtype=np.int64),
        "tract_overlap_tract": np.array(
            [value * 1000000 + 100 for value in county_value], dtype=np.int64
        ),
        "tract_overlap_population": np.array(county_pop),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "us_puma_ladder",
                    "puma_vintage": "2020_puma",
                    "sampling_basis": "population",
                    "layers": {
                        name: {
                            "vintage": (
                                "117th_congress" if bad_vintage else "119th_congress"
                            )
                            if name == "congressional_district"
                            else "2020_census",
                            "source": "bounded test fixture",
                        }
                        for name in ("congressional_district", "county", "tract")
                    },
                }
            )
        ),
    }
    ladder = tmp_path / "ladder.npz"
    np.savez_compressed(ladder, **arrays)
    walk = tmp_path / "crosswalk.csv"
    walk.write_bytes(
        default_congressional_district_vintage_crosswalk_path().read_bytes()
    )
    return {"us_puma_ladder_2020": ladder, "us_cd_crosswalk_117_119": walk}


class _Create(KernelBase):
    ref = "test.us_geography_source@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        frame = load_source("frame-store", context.sources["population"])
        return KernelResult(
            frame=frame, artifacts={"frame_context": encode_us_frame_context(frame)}
        )


def _run(tmp_path, *, resume="forbid"):
    frame = _frame()
    sources = (
        _lookup_files(tmp_path)
        if resume == "forbid"
        else {
            "us_puma_ladder_2020": tmp_path / "ladder.npz",
            "us_cd_crosswalk_117_119": tmp_path / "crosswalk.csv",
        }
    )
    input_store = ContentStore(tmp_path / "input-store")
    sources["population"] = input_store.put_frame("a" * 64, frame)
    columns = frame_column_declarations(frame)
    graph = Graph(
        country="us",
        sources=(SourceRef("population", "frame-store"), *LOOKUP_SOURCES),
        nodes=(
            Node(
                id="source",
                kernel=_Create.ref,
                outputs=columns,
                structural=StructuralDelta.CREATE,
                sources=("population",),
                artifact_outputs=(
                    ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
                ),
            ),
            *us_geography_nodes(columns, base="source", context_producer="source"),
        ),
    )
    registry = KernelRegistry()
    registry.register(_Create())
    register_us_geography_kernels(registry)
    store = ContentStore(tmp_path / "store")
    return (
        run_graph(
            compile_graph(graph),
            sources=sources,
            store=store,
            kernels=registry,
            resume=resume,
        ),
        store,
        frame,
        sources,
    )


def test_geography_values_native_anchors_and_cold_warm_context(tmp_path):
    cold, store, frame, sources = _run(tmp_path)
    actual = cold.population(f"{GEOGRAPHY_PHASE}.boundary")
    direct = with_household_us_puma_ladder(
        frame,
        decode_us_puma_ladder(sources["us_puma_ladder_2020"].read_bytes()),
        seed=0,
        assign_tract=False,
        expected_congressional_district_vintage="119th_congress",
    )
    for column in ("puma", "county_fips", "congressional_district_geoid"):
        np.testing.assert_array_equal(
            actual.table("household")[column].to_numpy(),
            direct.table("household")[column].to_numpy(),
        )
    assert actual.table("household").puma.iloc[:750].eq("0600100").all()
    for entity in frame.entities:
        np.testing.assert_array_equal(
            actual.resolve_weights(entity).values, frame.resolve_weights(entity).values
        )
    assert cold.nodes[f"{GEOGRAPHY_PHASE}.gate"].receipt["outcome"] == "pass"
    document = json.loads(
        store.load_bytes(cold.nodes[GEOGRAPHY_PHASE].opaque_artifacts["frame_context"])
    )
    for entity in actual.entities:
        assert (
            list(actual.table(entity).columns)
            == document["entities"][entity]["columns"]
        )
    warm, _, _, _ = _run(tmp_path, resume="require")
    assert cold.key == warm.key and all(node.hit for node in warm.nodes.values())
    for name in (GEOGRAPHY_PHASE,):
        assert store.load_bytes(
            cold.nodes[name].opaque_artifacts["frame_context"]
        ) == store.load_bytes(warm.nodes[name].opaque_artifacts["frame_context"])
    assert (
        cold.nodes[f"{GEOGRAPHY_PHASE}.lookup"].kernel_ref
        == USGeographyLookupKernel.ref
    )


def test_geography_lookup_refuses_wrong_vintage_before_assignment(tmp_path):
    sources = _lookup_files(tmp_path, bad_vintage=True)
    with pytest.raises(ValueError, match="vintage"):
        _lookups(
            *(
                sources[name].read_bytes()
                for name in ("us_puma_ladder_2020", "us_cd_crosswalk_117_119")
            )
        )


def _assign_context(tmp_path):
    frame = _frame()
    node = us_geography_nodes(
        frame_column_declarations(frame), base="source", context_producer="source"
    )[2]
    files = _lookup_files(tmp_path)
    payloads = {
        "frame_context": (encode_us_frame_context(frame), US_FRAME_CONTEXT_TYPE),
        "ladder": (files["us_puma_ladder_2020"].read_bytes(), US_PUMA_LOOKUP_TYPE),
        "crosswalk": (
            files["us_cd_crosswalk_117_119"].read_bytes(),
            US_CD_CROSSWALK_TYPE,
        ),
    }
    return KernelContext(
        node=node,
        params=node.params,
        tables={
            "household": frame.table("household")[
                ["household_id", "state_fips", "puma"]
            ]
        },
        weights={"household": frame.resolve_weights("household")},
        strata=frame.strata,
        rng=np.random.default_rng(0),
        artifacts={
            name: ArtifactValue(payload, type_, "a" * 64, "b" * 64, NumericScope())
            for name, (payload, type_) in payloads.items()
        },
    )


@pytest.mark.parametrize("mutation", ["ordered_ids", "id_dtype", "columns", "weights"])
def test_geography_refuses_stale_household_context_before_assigning(tmp_path, mutation):
    context = _assign_context(tmp_path)
    document = json.loads(context.artifacts["frame_context"].payload)
    household = document["entities"]["household"]
    if mutation == "ordered_ids":
        household["ordered_ids_sha256"] = "f" * 64
    elif mutation == "id_dtype":
        household["id_dtype"] = "Int64"
    elif mutation == "columns":
        household["columns"].remove("state_fips")
    else:
        document["weight_sources"]["household"] = "calibrated"
    artifacts = dict(context.artifacts)
    artifacts["frame_context"] = replace(
        artifacts["frame_context"], payload=json.dumps(document).encode()
    )
    with pytest.raises(ValueError, match="identity|undeclared|weight source"):
        USGeographyAssignKernel().run(replace(context, artifacts=artifacts))


@pytest.mark.parametrize("incumbents", [False, True])
def test_geography_preserves_actual_column_order_and_nullable_incumbents(
    tmp_path, monkeypatch, incumbents
):
    frame = _frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    household = tables["household"].drop(columns="puma")
    if incumbents:
        household.insert(
            1,
            "county_fips",
            pd.Series("00000", index=household.index, dtype=dtype_for_token("string")),
        )
        household.insert(
            2,
            "congressional_district_geoid",
            pd.Series(pd.NA, index=household.index, dtype="Int64"),
        )
    tables["household"] = household
    variant = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )
    monkeypatch.setattr(sys.modules[__name__], "_frame", lambda: variant)
    run, store, _, _ = _run(tmp_path)
    actual = run.population(f"{GEOGRAPHY_PHASE}.boundary").table("household")
    document = json.loads(
        store.load_bytes(run.nodes[GEOGRAPHY_PHASE].opaque_artifacts["frame_context"])
    )
    assert list(actual.columns) == document["entities"]["household"]["columns"]
    assert str(actual["congressional_district_geoid"].dtype) == (
        "Int64" if incumbents else "int64"
    )
    assert (
        not actual[["puma", "county_fips", "congressional_district_geoid"]]
        .isna()
        .any()
        .any()
    )


def test_geography_lookup_refuses_incomplete_district_universe(tmp_path):
    files = _lookup_files(tmp_path)
    with np.load(files["us_puma_ladder_2020"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in ("cd_overlap_puma", "cd_overlap_cd", "cd_overlap_population"):
        arrays[key] = arrays[key][1:]
    # Keep every PUMA's population conserved while removing a district entirely.
    arrays["cd_overlap_population"][0] += 1.0
    np.savez_compressed(files["us_puma_ladder_2020"], **arrays)
    with pytest.raises(ValueError, match="universes differ"):
        _lookups(
            files["us_puma_ladder_2020"].read_bytes(),
            files["us_cd_crosswalk_117_119"].read_bytes(),
        )


def test_geography_gate_reports_failed_nyc_mass_as_assignment_integrity(tmp_path):
    context = _assign_context(tmp_path)
    assigned = USGeographyAssignKernel().run(context)
    household = context.tables["household"].copy()
    for (_, column), values in assigned.columns.items():
        household[column] = values.to_numpy()
    # Intact geographic codes with zero NY mass must still fail the legacy gate.
    weights = Weights(
        np.where(household.state_fips.eq(36), 0.0, 1.0), WeightKind.DESIGN
    )
    result = USGeographyGateKernel().run(
        replace(
            context,
            params={"phase": GEOGRAPHY_PHASE},
            tables={"household": household},
            weights={"household": weights},
        )
    )
    assert result.receipt["outcome"] == "fail"
    assert result.receipt["scope"] == "geography_assignment_integrity"
    assert result.receipt["evidence"]["passed"] is False


def test_every_geography_inner_node_declares_the_registered_outer_phase():
    nodes = us_geography_nodes(
        frame_column_declarations(_frame()), base="source", context_producer="source"
    )
    assert all(node.params.get("phase") == GEOGRAPHY_PHASE for node in nodes)
    for node in (nodes[1], nodes[-1]):
        assert node.params == {"phase": GEOGRAPHY_PHASE}


@pytest.mark.parametrize("kernel", [USGeographyBoundaryKernel, USGeographyGateKernel])
@pytest.mark.parametrize(
    "params",
    [{}, {"phase": "other_stage"}, {"phase": GEOGRAPHY_PHASE, "extra": True}],
)
def test_geography_boundary_and_gate_refuse_undeclared_phase_parameters(
    tmp_path, kernel, params
):
    context = _assign_context(tmp_path)
    with pytest.raises(ValueError, match="registered phase"):
        kernel().run(replace(context, params=params))


def test_actual_assembly_harmonization_and_geography_compose_and_require_replay(
    tmp_path, monkeypatch
):
    from microcosm.build.us_runtime import graph_sources

    # Only external readers are replaced. All source preparation, harmonization,
    # assignment, typed context and graph/store operations below are real.
    source_frames = {}
    for origin, design_weight in (("asec", 3.0), ("acs", 7.0)):
        template = _frame()
        tables = {entity: template.table(entity).copy() for entity in template.entities}
        if origin == "asec":
            tables["household"] = tables["household"].drop(columns="puma")
            tables["person"]["P_SEQ"] = pd.Series(
                np.arange(len(tables["person"]), dtype=np.int64) + 2**53 + 1,
                dtype="Int64",
            )
        else:
            tables["household"]["TYPEHUGQ"] = np.int64(1)
            tables["household"]["puma"] = pd.Series(
                np.where(tables["household"].state_fips.eq(6), "0600100", "3600100"),
                dtype=dtype_for_token("string"),
            )
        source_frames[origin] = Frame(
            tables,
            US_SCHEMA,
            {
                "household": Weights(
                    np.full(len(tables["household"]), design_weight), WeightKind.DESIGN
                )
            },
            pd.Series([f"{origin}_2024"] * len(tables["person"])),
        )

    calls = []
    for origin in source_frames:

        def read_source(_path, *, store=None, origin=origin):
            calls.append(origin)
            return source_frames[origin]

        monkeypatch.setattr(graph_sources, f"_decode_graph_{origin}", read_source)

    paths = _lookup_files(tmp_path)
    for source in ("asec_raw_stage", "acs_native"):
        paths[source] = tmp_path / source
        paths[source].write_bytes(f"bounded {source} fixture identity".encode())
    direct = graph_sources.assembly_from_sources(
        paths["asec_raw_stage"],
        paths["acs_native"],
        sample_fraction=0.5,
        sample_seed=578,
    )
    prepared_direct = graph_sources.prepare_stacked_spine(
        source_frames["asec"],
        source_frames["acs"],
        sample_fraction=0.5,
        sample_seed=578,
    )
    graph_sources._canonical_assembly(
        prepared_direct.frame, (source_frames["asec"], source_frames["acs"])
    )
    assigned_direct = with_household_us_puma_ladder(
        direct.frame,
        decode_us_puma_ladder(paths["us_puma_ladder_2020"].read_bytes()),
        seed=0,
        assign_tract=False,
        expected_congressional_district_vintage="119th_congress",
    )
    # Geography has explicit graph storage types; this changes no direct values.
    for column in ("puma", "county_fips"):
        assigned_direct.table("household")[column] = assigned_direct.table("household")[
            column
        ].astype(dtype_for_token("string"))
    columns = frame_column_declarations(direct.frame)
    assembly = graph_sources.us_assembly_graph(
        columns, sample_fraction=0.5, sample_seed=578
    )
    graph = Graph(
        country="us",
        sources=(*assembly.sources, *LOOKUP_SOURCES),
        nodes=(
            *assembly.nodes,
            *us_geography_nodes(
                columns,
                base=graph_sources.ASSEMBLY_PHASE,
                context_producer=graph_sources.ASSEMBLY_PHASE,
            ),
        ),
    )
    assert len(graph.nodes) == 6
    kernels = graph_sources.us_assembly_registry()
    register_us_geography_kernels(kernels)
    store = ContentStore(
        tmp_path / "combined-store", codecs=graph_sources.us_source_codecs()
    )
    calls.clear()
    compiled = compile_graph(graph)
    cold = run_graph(
        compiled, sources=paths, store=store, kernels=kernels, resume="forbid"
    )
    assert calls == ["asec", "acs"]
    assert len(cold.nodes) == 6
    prepared = cold.population(f"{graph_sources.ASSEMBLY_PHASE}.prepare")
    assembled = cold.population(graph_sources.ASSEMBLY_PHASE)
    actual = cold.population(f"{GEOGRAPHY_PHASE}.boundary")
    assert prepared.weights_for("household").kind is WeightKind.DESIGN
    assert assembled.weights_for("household").kind is WeightKind.IMPORTANCE
    assert actual.weights_for("household").kind is WeightKind.IMPORTANCE
    assert not np.array_equal(
        prepared.weights_for("household").values,
        assembled.weights_for("household").values,
    )
    for phase, expected in (
        (f"{graph_sources.ASSEMBLY_PHASE}.prepare", prepared_direct.frame),
        (graph_sources.ASSEMBLY_PHASE, direct.frame),
        (GEOGRAPHY_PHASE, assigned_direct),
    ):
        payload = store.load_bytes(cold.nodes[phase].opaque_artifacts["frame_context"])
        assert payload == encode_us_frame_context(expected)
    assert assigned_direct.metadata and assigned_direct.mass_log
    assert str(actual.table("person").P_SEQ.dtype) == "Int64"
    assert actual.table("person").P_SEQ.isna().any()
    for entity in US_SCHEMA.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity), assigned_direct.table(entity)
        )
        np.testing.assert_array_equal(
            actual.resolve_weights(entity).values,
            assigned_direct.resolve_weights(entity).values,
        )
    pd.testing.assert_series_equal(actual.strata, assigned_direct.strata)
    gate = cold.nodes[f"{GEOGRAPHY_PHASE}.gate"].receipt
    assert gate["outcome"] == "pass"
    assert gate["scope"] == "geography_assignment_integrity"
    assert gate["evidence"]["passed"] is True

    warm_kernels = graph_sources.us_assembly_registry()
    register_us_geography_kernels(warm_kernels)
    warm_store = ContentStore(
        tmp_path / "combined-store", codecs=graph_sources.us_source_codecs()
    )
    warm = run_graph(
        compiled,
        sources=paths,
        store=warm_store,
        kernels=warm_kernels,
        resume="require",
    )
    assert calls == ["asec", "acs"]
    assert cold.key == warm.key and all(node.hit for node in warm.nodes.values())
    assert warm.nodes[f"{GEOGRAPHY_PHASE}.gate"].receipt == gate
    for phase in (
        f"{graph_sources.ASSEMBLY_PHASE}.prepare",
        graph_sources.ASSEMBLY_PHASE,
        GEOGRAPHY_PHASE,
    ):
        assert warm_store.load_bytes(
            warm.nodes[phase].opaque_artifacts["frame_context"]
        ) == store.load_bytes(cold.nodes[phase].opaque_artifacts["frame_context"])
    for entity in US_SCHEMA.entities:
        pd.testing.assert_frame_equal(
            warm.population(f"{GEOGRAPHY_PHASE}.boundary").table(entity),
            actual.table(entity),
        )

    # Full H5 parity includes nullable IDs, original ordered mass history and
    # normative context. Executor aggregate transitions remain a separate log.
    from microcosm.build.frame_checkpoint import (
        load_frame_checkpoint,
        write_frame_checkpoint,
    )
    from microcosm.build.us_runtime.graph_context import _mass_records
    from microcosm.build.us_runtime.graph_implementation import implementation_manifest

    exports = {"direct": assigned_direct}
    for name, manifest, run_store in (
        ("cold", cold, store),
        ("require", warm, warm_store),
    ):
        population = manifest.population(f"{GEOGRAPHY_PHASE}.boundary")
        payload = run_store.load_bytes(
            manifest.nodes[GEOGRAPHY_PHASE].opaque_artifacts["frame_context"]
        )
        document = json.loads(payload)
        exports[name] = Frame(
            {entity: population.table(entity) for entity in population.entities},
            population.schema,
            {"household": population.weights_for("household")},
            population.strata,
            mass_log=_mass_records(document["mass_log"]),
            metadata=document["metadata"],
        )
        assert encode_us_frame_context(exports[name]) == payload
        assert exports[name].mass_log != population.mass_log
        assert manifest.nodes[GEOGRAPHY_PHASE].receipt[
            "implementation"
        ] == implementation_manifest("geography")
    for name, frame in exports.items():
        path = tmp_path / f"full-{name}.h5"
        write_frame_checkpoint(path, frame, metadata={"scope": "combined fixture"})
        loaded = load_frame_checkpoint(path, frame_metadata=frame.metadata).frame
        assert encode_us_frame_context(loaded) == encode_us_frame_context(frame)
    assert (
        (tmp_path / "full-direct.h5").read_bytes()
        == (tmp_path / "full-cold.h5").read_bytes()
        == (tmp_path / "full-require.h5").read_bytes()
    )

    # Changed lookup bytes have their own source identity, never an assembly
    # implementation dependency. A harmless extra CSV newline exercises it.
    crosswalk = paths["us_cd_crosswalk_117_119"]
    crosswalk.write_bytes(crosswalk.read_bytes() + b"\n")
    changed = run_graph(
        compiled, sources=paths, store=store, kernels=kernels, resume="auto"
    )
    assert calls == ["asec", "acs"]
    for phase in (
        f"{graph_sources.ASSEMBLY_PHASE}.prepare",
        graph_sources.ASSEMBLY_PHASE,
    ):
        assert changed.nodes[phase].hit
        assert changed.nodes[phase].key == cold.nodes[phase].key
    assert not changed.nodes[f"{GEOGRAPHY_PHASE}.lookup"].hit
    assert not changed.nodes[GEOGRAPHY_PHASE].hit
    assert changed.nodes[GEOGRAPHY_PHASE].key != cold.nodes[GEOGRAPHY_PHASE].key
