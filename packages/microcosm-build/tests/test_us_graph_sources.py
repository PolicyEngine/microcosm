"""Real assembly operations retain identities across graph execution/resume."""

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_sources
from microcosm.build.us_runtime.graph_context import encode_us_frame_context
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import ContentStore, compile_graph, run_graph
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import PopulationError


def _source(origin):
    ids = np.array([10, 20, 30, 40], dtype=np.int64)
    members = np.array([10, 10, 20, 30, 40], dtype=np.int64)
    tables = {
        entity: pd.DataFrame({US_SCHEMA.entity_id_column(entity): ids})
        for entity in US_SCHEMA.group_entities
    }
    tables["person"] = pd.DataFrame(
        {
            "person_id": np.arange(5, dtype=np.int64),
            "age": [10.0, 40.0, 65.0, 25.0, 45.0],
        }
    )
    for entity in US_SCHEMA.group_entities:
        tables["person"][f"person_{entity}_id"] = members
    tables["household"]["state_fips"] = 6
    if origin == "acs":
        tables["person"]["WAGP"] = [0.0, 50000.0, np.nan, 60000.0, 70000.0]
        tables["household"]["TYPEHUGQ"] = 1
        tables["household"]["puma"] = "0600101"
        tables["household"]["tenure_type"] = "RENTED"
    else:
        tables["person"]["P_SEQ"] = np.array([0, 1, 2**53 + 1, 4, 5], dtype=np.int64)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([2.0, 3.0, 4.0, 5.0]), WeightKind.DESIGN)},
        pd.Series([f"{origin}_2024"] * len(members), dtype=object),
    )


def test_assembly_graph_matches_direct_call_and_reuses_typed_context(
    tmp_path, monkeypatch
):
    calls = []
    for origin in ("asec", "acs"):

        def load(_path, *, store=None, origin=origin):
            calls.append(origin)
            return _source(origin)

        monkeypatch.setattr(graph_sources, f"_decode_graph_{origin}", load)
    paths = {name: tmp_path / name for name in ("asec_raw_stage", "acs_native")}
    for path in paths.values():
        path.write_bytes(b"bounded fixture source identity")
    direct = graph_sources.assembly_from_sources(
        paths["asec_raw_stage"],
        paths["acs_native"],
        sample_fraction=0.5,
        sample_seed=578,
    )
    graph = graph_sources.us_assembly_graph(
        graph_sources.frame_column_declarations(direct.frame),
        sample_fraction=0.5,
        sample_seed=578,
    )
    compiled = compile_graph(graph)
    store = ContentStore(tmp_path / "store", codecs=graph_sources.us_source_codecs())
    cold = run_graph(
        compiled,
        sources=paths,
        store=store,
        kernels=graph_sources.us_assembly_registry(),
        resume="forbid",
    )
    assert calls == ["asec", "acs", "asec", "acs"]
    prepared = cold.population("assemble_stacked_spine.prepare")
    assert prepared.weights_for("household").kind is WeightKind.DESIGN
    actual = cold.population("assemble_stacked_spine")
    assert actual.weights_for("household").kind is WeightKind.IMPORTANCE
    for entity in US_SCHEMA.entities:
        pd.testing.assert_frame_equal(actual.table(entity), direct.frame.table(entity))
        np.testing.assert_array_equal(
            actual.resolve_weights(entity).values,
            direct.frame.resolve_weights(entity).values,
        )
    pd.testing.assert_series_equal(actual.strata, direct.frame.strata)
    warm = run_graph(
        compiled,
        sources=paths,
        store=store,
        kernels=graph_sources.us_assembly_registry(),
        resume="require",
    )
    assert calls == ["asec", "acs", "asec", "acs"]
    assert cold.key == warm.key
    assert all(node.hit for node in warm.nodes.values())
    # The immutable typed payload is sufficient even though Frame storage omits metadata.
    payload = store.load_bytes(
        cold.nodes["assemble_stacked_spine"].opaque_artifacts["frame_context"]
    )
    assert payload == encode_us_frame_context(direct.frame)
    assert payload == canonical_json(json.loads(payload))
    # A complete tiny source/graph/export path must pass before another real
    # ACS parse. The context's ordered legacy log and the executor's aggregate
    # transition ledger are different, deliberate records of this split.
    from microcosm.build.frame_checkpoint import (
        load_frame_checkpoint,
        write_frame_checkpoint,
    )
    from microcosm.build.us_runtime.graph_context import _mass_records

    document = json.loads(payload)
    restored = Frame(
        {entity: actual.table(entity) for entity in actual.entities},
        actual.schema,
        {"household": actual.weights_for("household")},
        actual.strata,
        mass_log=_mass_records(document["mass_log"]),
        metadata=document["metadata"],
    )
    assert encode_us_frame_context(restored) == payload
    assert restored.mass_log != actual.mass_log
    assert cold.nodes["assemble_stacked_spine"].receipt["mass"]["policy"] == "declared"
    for name, frame in (("direct", direct.frame), ("cold", restored)):
        write_frame_checkpoint(
            tmp_path / f"{name}.h5", frame, metadata={"scope": "fixture"}
        )
    assert (tmp_path / "cold.h5").read_bytes() == (tmp_path / "direct.h5").read_bytes()
    loaded = load_frame_checkpoint(
        tmp_path / "cold.h5", frame_metadata=document["metadata"]
    ).frame
    assert encode_us_frame_context(loaded) == payload
    for entity in actual.entities:
        pd.testing.assert_frame_equal(loaded.table(entity), direct.frame.table(entity))


def test_column_inventory_refuses_unsupported_physical_values():
    frame = _source("acs")
    frame.table("person")["ambiguous"] = pd.Series([None] * 5, dtype=object)
    with pytest.raises(PopulationError, match="Unsupported"):
        graph_sources.frame_column_declarations(frame)


@pytest.mark.parametrize("missing_value", [pd.NA, np.nan])
def test_assembly_preserves_all_missing_declared_string_column(missing_value):
    from microcosm.build.us_runtime.stacked_spine import assemble_stacked_spine
    from microcosm.graph.population import dtype_for_token

    asec, acs = _source("asec"), _source("acs")
    acs.table("person")["declared_text"] = pd.Series(
        [missing_value] * acs.n("person"),
        dtype=pd.StringDtype(storage="python", na_value=missing_value),
    )
    result = assemble_stacked_spine(asec, acs, sample_fraction=0.5, sample_seed=578)
    original_missing = result.frame.table("person")["declared_text"].isna()
    transitions = graph_sources._canonical_assembly(result.frame, (asec, acs))
    actual = result.frame.table("person")["declared_text"]
    assert actual.dtype == dtype_for_token("string")
    pd.testing.assert_series_equal(actual.isna(), original_missing)
    assert actual.isna().all()
    assert any(
        item["column"] == "declared_text" and item["from"] == "object"
        for item in transitions
    )
    columns = graph_sources.frame_column_declarations(result.frame)
    assert (
        next(item.dtype for item in columns if item.column == "declared_text")
        == "string"
    )


def test_acs_codec_refuses_missing_or_extra_archives(tmp_path):
    with pytest.raises(ValueError, match="exactly"):
        graph_sources.load_graph_acs(tmp_path)


def test_real_acs_codec_preserves_native_zero_and_child_source_absence(tmp_path):
    from zipfile import ZipFile

    household = {
        "SERIALNO": "2024HU1",
        "ST": "06",
        "PUMA": "00101",
        "WGTP": 10,
        "NP": 2,
        "ADJHSG": 1000000,
        "TEN": 3,
        "RNTP": 1200,
        "GRNTP": 1400,
        "TAXAMT": None,
        "TYPEHUGQ": 1,
    }
    adult = {
        "SERIALNO": "2024HU1",
        "SPORDER": 1,
        "RELSHIPP": 20,
        "AGEP": 40,
        "SEX": 1,
        "MAR": 5,
        "ADJINC": 1000000,
        "WAGP": 0,
        "SEMP": 0,
        "SSP": 0,
        "SSIP": 0,
        "RETP": 0,
        "INTP": 0,
        "PWGTP": 10,
    }
    child = {**adult, "SPORDER": 2, "RELSHIPP": 25, "AGEP": 10, "WAGP": None}
    for name, member, rows in (
        ("csv_hus.zip", "psam_husa.csv", [household]),
        ("csv_pus.zip", "psam_pusa.csv", [adult, child]),
    ):
        with ZipFile(tmp_path / name, "w") as archive:
            archive.writestr(member, pd.DataFrame(rows).to_csv(index=False))
    # The real CSV parser may execute Arrow before its canonical output boundary.
    # Compare both backends on this fixture; their options still enter identity.
    frames = {}
    for backend in ("python", "pyarrow"):
        with pd.option_context("mode.string_storage", backend):
            frames[backend] = graph_sources.load_graph_acs(tmp_path)
    frame = frames["python"]
    column_index_storage = set()
    mapped_string_storage = set()
    for entity in frame.entities:
        left, right = frame.table(entity), frames["pyarrow"].table(entity)
        column_index_storage.add(
            (left.columns.dtype.storage, right.columns.dtype.storage)
        )
        comparable = right.copy()
        for column in left:
            if left[column].dtype != right[column].dtype:
                assert isinstance(left[column].dtype, pd.StringDtype)
                assert isinstance(right[column].dtype, pd.StringDtype)
                assert left[column].dtype.storage == "python"
                assert right[column].dtype.storage == "pyarrow"
                mapped_string_storage.add((entity, column))
                comparable[column] = right[column].astype(left[column].dtype)
        pd.testing.assert_frame_equal(left, comparable, check_column_type=False)
        np.testing.assert_array_equal(
            frame.resolve_weights(entity).values,
            frames["pyarrow"].resolve_weights(entity).values,
        )
    pd.testing.assert_series_equal(frame.strata, frames["pyarrow"].strata)
    assert frame.metadata == frames["pyarrow"].metadata
    assert frame.mass_log == frames["pyarrow"].mass_log
    # Native mapping adds another string after the parser's canonical boundary.
    # Both it and the column-name Index are real representation differences.
    assert ("python", "pyarrow") in column_index_storage
    assert ("household", "tenure_type") in mapped_string_storage
    person = frame.table("person").sort_values("age")
    assert pd.isna(person.iloc[0]["WAGP"])
    assert pd.isna(person.iloc[0]["employment_income_before_lsr"])
    assert person.iloc[1]["WAGP"] == 0
    assert person.iloc[1]["employment_income_before_lsr"] == 0
    assert frame.weights_for("household").kind is WeightKind.DESIGN
    assert frame.weights_for("household").total == 10
    (tmp_path / "csv_hus.zip").write_bytes(b"")
    (tmp_path / "csv_pus.zip").write_bytes(b"")
    (tmp_path / "undeclared").write_bytes(b"")
    with pytest.raises(ValueError, match="exactly"):
        graph_sources.load_graph_acs(tmp_path)


def _bounded_graph_run(tmp_path, monkeypatch, *, fraction=0.5):
    calls = []
    for origin in ("asec", "acs"):

        def load(_path, *, store=None, origin=origin):
            calls.append(origin)
            return _source(origin)

        monkeypatch.setattr(graph_sources, f"_decode_graph_{origin}", load)
    paths = {name: tmp_path / name for name in ("asec_raw_stage", "acs_native")}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    oracle = graph_sources.assembly_from_sources(
        paths["asec_raw_stage"],
        paths["acs_native"],
        sample_fraction=fraction,
        sample_seed=578,
    )
    compiled = compile_graph(
        graph_sources.us_assembly_graph(
            graph_sources.frame_column_declarations(oracle.frame),
            sample_fraction=fraction,
            sample_seed=578,
        )
    )
    store = ContentStore(tmp_path / "store", codecs=graph_sources.us_source_codecs())
    kernels = graph_sources.us_assembly_registry()
    manifest = run_graph(
        compiled, sources=paths, store=store, kernels=kernels, resume="forbid"
    )
    return compiled, store, kernels, paths, manifest, calls


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_graph_keeps_actual_normalized_design_anchor_and_person_mass(
    tmp_path, monkeypatch, fraction
):
    import microcosm.graph.population as population
    from microcosm.build.us_runtime.support_provenance import support_channel_column

    anchors = []
    carry = population._carry_design_weights

    def observe(before, frame, node, result):
        values = carry(before, frame, node, result)
        if node.id == graph_sources.ASSEMBLY_PHASE:
            anchors.append(values["household"].copy())
        return values

    monkeypatch.setattr(population, "_carry_design_weights", observe)
    compiled, store, kernels, paths, manifest, _ = _bounded_graph_run(
        tmp_path, monkeypatch, fraction=fraction
    )
    prepared = manifest.population("assemble_stacked_spine.prepare")
    finished = manifest.population("assemble_stacked_spine")
    preparation = json.loads(
        store.load_bytes(
            manifest.nodes["assemble_stacked_spine.prepare"].opaque_artifacts[
                "preparation"
            ]
        )
    )
    source_ids = prepared.table("household")["household_spine_source_id"].to_numpy()
    channels = (
        prepared.table("household")[support_channel_column("household")]
        .astype(str)
        .to_numpy()
    )
    for channel in ("asec", "acs"):
        mask = channels == channel
        source = _source(channel)
        original = pd.Series(
            source.weights_for("household").values,
            index=source.table("household")["household_id"],
        )
        selected = original.loc[source_ids[mask]].to_numpy()
        factor = source.weights_for("household").total / selected.sum()
        expected = selected * factor
        np.testing.assert_array_equal(
            prepared.weights_for("household").values[mask], expected
        )
        assert (
            preparation["sampling"]["survey_samples"][channel]["normalization_factor"]
            == factor
        )
    assert len(anchors) == 1
    np.testing.assert_array_equal(anchors[0], prepared.weights_for("household").values)
    assert not np.array_equal(anchors[0], finished.weights_for("household").values)
    mass = manifest.nodes["assemble_stacked_spine"].receipt["mass"]
    assert mass["before"] == prepared.stratum_mass().sum()
    assert mass["after"] == finished.stratum_mass().sum()
    assert dict(mass["stratum_before"]) == prepared.stratum_mass().to_dict()
    assert dict(mass["stratum_after"]) == finished.stratum_mass().to_dict()
    if fraction == 1.0:
        assert mass["before"] != prepared.weights_for("household").total
        assert mass["after"] != finished.weights_for("household").total
    resumed = run_graph(
        compiled,
        sources=paths,
        store=ContentStore(tmp_path / "store", codecs=graph_sources.us_source_codecs()),
        kernels=kernels,
        resume="require",
    )
    assert resumed.key == manifest.key
    np.testing.assert_array_equal(anchors[-1], anchors[0])


def _harmonize_context(compiled, store, manifest):
    from microcosm.graph import (
        ArtifactValue,
        KernelContext,
        Numeric,
        NumericScope,
        platform_fingerprint,
    )

    frame = manifest.population("assemble_stacked_spine.prepare")
    node = compiled.graph.node("assemble_stacked_spine")
    producer = manifest.nodes["assemble_stacked_spine.prepare"]
    scope = NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint())
    artifacts = {
        edge.name: ArtifactValue(
            store.load_bytes(producer.opaque_artifacts[edge.artifact]),
            edge.type,
            producer.opaque_artifacts[edge.artifact],
            producer.key,
            scope,
        )
        for edge in node.artifact_inputs
    }
    return KernelContext(
        node=node,
        tables={entity: frame.table(entity) for entity in ("person", "household")},
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts=artifacts,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "producer",
        "context_weight_kind",
        "metadata",
        "actual_weights",
        "preparation_bytes",
    ],
)
def test_harmonize_adapter_refuses_unbound_preparation(tmp_path, monkeypatch, mutation):
    from dataclasses import replace

    compiled, store, _, _, manifest, _ = _bounded_graph_run(tmp_path, monkeypatch)
    context = _harmonize_context(compiled, store, manifest)
    artifacts = dict(context.artifacts)
    if mutation == "producer":
        artifacts["preparation"] = replace(
            artifacts["preparation"], producer_key="f" * 64
        )
    elif mutation in ("context_weight_kind", "metadata"):
        document = json.loads(artifacts["frame_context"].payload)
        if mutation == "context_weight_kind":
            document["weight_sources"]["household"] = "importance"
        else:
            document["metadata"]["us_spine_assembly_manifest"]["changed"] = True
        artifacts["frame_context"] = replace(
            artifacts["frame_context"], payload=canonical_json(document)
        )
    elif mutation == "actual_weights":
        context = replace(
            context,
            weights={
                "household": Weights(
                    context.weights["household"].values * 2, WeightKind.DESIGN
                )
            },
        )
    else:
        artifacts["preparation"] = replace(
            artifacts["preparation"], payload=artifacts["preparation"].payload + b"\n"
        )
    with pytest.raises(
        ValueError,
        match="producer|DESIGN|metadata differ|binding changed|canonical JSON",
    ):
        graph_sources.USSpineHarmonizeKernel().run(
            replace(context, artifacts=artifacts)
        )


@pytest.mark.parametrize(
    "mutation", ["source_pin", "missing_source", "corrupt_preparation"]
)
def test_required_assembly_replay_never_rebuilds_invalid_input(
    tmp_path, monkeypatch, mutation
):
    from microcosm.graph import StoreCorruptError, StoreMissError

    compiled, store, kernels, paths, manifest, calls = _bounded_graph_run(
        tmp_path, monkeypatch
    )
    count = len(calls)
    expected = StoreMissError
    if mutation == "source_pin":
        paths["asec_raw_stage"].write_bytes(b"changed source pin")
    elif mutation == "missing_source":
        paths = {name: path for name, path in paths.items() if name != "asec_raw_stage"}
        expected = FileNotFoundError
    else:
        key = manifest.nodes["assemble_stacked_spine.prepare"].opaque_artifacts[
            "preparation"
        ]
        (store.object_path(key) / "payload.bin").write_bytes(b"corrupt")
        expected = StoreCorruptError
    with pytest.raises(expected):
        run_graph(
            compiled, sources=paths, store=store, kernels=kernels, resume="require"
        )
    assert len(calls) == count


def test_source_mutation_during_create_is_rejected(tmp_path, monkeypatch):
    from microcosm.graph import NodeRejectedError

    compiled, store, kernels, paths, _, _ = _bounded_graph_run(tmp_path, monkeypatch)

    def changing_source(path, *, store=None):
        path.write_bytes(b"changed during read")
        return _source("asec")

    monkeypatch.setattr(graph_sources, "_decode_graph_asec", changing_source)
    with pytest.raises(NodeRejectedError, match="changed source"):
        run_graph(
            compiled, sources=paths, store=store, kernels=kernels, resume="forbid"
        )


def test_asec_hdf_dependency_version_invalidates_assembly_implementation(monkeypatch):
    import microcosm.graph.kernel as kernel

    assert "h5py" in graph_sources.USAssemblyCreateKernel.capabilities.dependencies
    original_version = kernel.importlib_metadata.version
    original_hash = graph_sources.USAssemblyCreateKernel().implementation_hash()

    def changed_hdf_version(name):
        value = original_version(name)
        return value + ".changed" if name == "h5py" else value

    monkeypatch.setattr(kernel.importlib_metadata, "version", changed_hdf_version)
    assert graph_sources.USAssemblyCreateKernel().implementation_hash() != original_hash


def test_required_replay_refuses_substituted_public_codec_loader(tmp_path, monkeypatch):
    compiled, store, kernels, paths, _, _ = _bounded_graph_run(tmp_path, monkeypatch)
    monkeypatch.setattr(graph_sources, "load_graph_asec", lambda path: _source("asec"))
    with pytest.raises(ValueError, match="loader differs"):
        run_graph(
            compiled, sources=paths, store=store, kernels=kernels, resume="require"
        )
