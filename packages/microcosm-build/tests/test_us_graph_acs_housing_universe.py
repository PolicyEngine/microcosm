"""Invented archives through the actual ACS preparation and graph executor."""

import csv
import hashlib
import importlib.util
import io
import json
import struct
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.us_runtime import acs_housing_universe_source as source
from microcosm.build.us_runtime import graph_acs_housing_universe as bridge
from microcosm.build.us_runtime.graph_context import encode_us_frame_context
from microcosm.build.us_runtime.graph_sources import (
    frame_column_declarations,
    us_source_codecs,
)
from microcosm.graph import ArtifactType, ContentStore, compile_graph, run_graph
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import opaque_artifact_key


@pytest.fixture(autouse=True)
def _invented_disk_budget(monkeypatch):
    monkeypatch.setattr(
        source.shutil, "disk_usage", lambda _path: SimpleNamespace(free=64 * 1024**3)
    )


def _fixture(tmp_path, monkeypatch):
    helper_path = Path(__file__).with_name("test_us_acs_pums.py")
    spec = importlib.util.spec_from_file_location(
        "acs_full_fixture_helpers", helper_path
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    h = [
        helper._household("2024HU0000001", NP=1, TEN=1),
        helper._household("2024HU0000002", NP=2, TEN=None),
        helper._household("2024HU0000003", NP=0, TEN=None),
        helper._household("2024GQ0000001", TYPEHUGQ=2, NP=1, WGTP=0, TEN=None),
        helper._household("2024GQ0000002", TYPEHUGQ=3, NP=1, WGTP=0, TEN=None),
    ]
    p = [
        helper._person("2024HU0000001", 1, 20),
        helper._person("2024HU0000002", 1, 40, RELSHIPP=20, MAR=1, SEX=1),
        helper._person("2024HU0000002", 2, 38, RELSHIPP=21, MAR=1, SEX=2),
        helper._person("2024GQ0000001", "01", 37, PWGTP=77),
        helper._person("2024GQ0000002", "01", 38, PWGTP=78),
    ]
    for rows, _role, name, prefix in (
        (h, "household", "csv_hus.zip", "psam_hus"),
        (p, "person", "csv_pus.zip", "psam_pus"),
    ):
        with zipfile.ZipFile(source_dir / name, "w") as archive:
            for suffix, chunk in (("a", rows[:2]), ("b", rows[2:])):
                stream = io.StringIO(newline="")
                writer = csv.DictWriter(
                    stream, fieldnames=list(rows[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(chunk)
                archive.writestr(prefix + suffix + ".csv", stream.getvalue())
    monkeypatch.setattr(
        source,
        "_ARCHIVE_PINS",
        tuple(
            (
                role,
                name,
                hashlib.sha256((source_dir / name).read_bytes()).hexdigest(),
                (source_dir / name).stat().st_size,
            )
            for role, name in (("household", "csv_hus.zip"), ("person", "csv_pus.zip"))
        ),
    )
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    return source_dir, snapshots


def test_real_source_preparation_retains_gq_unknown_and_vacancy_evidence(
    tmp_path, monkeypatch
):
    source_dir, snapshots = _fixture(tmp_path, monkeypatch)
    prepared = source.prepare_acs_housing_population(
        source_dir, snapshot_root=snapshots
    )
    assert prepared.frame.n("household") == 4
    assert prepared.frame.n("person") == 5
    assert prepared.source.households.shape[0] == 5
    assert prepared.source.codes.occupied_hu.tolist() == [2, 2, 1, 1, 0]
    assert prepared.source.codes.hu_tenure_class.tolist() == [0, 0, 1, 3, 0]
    assert all(str(t) == "uint8" for t in prepared.source.codes.dtypes)
    assert prepared.receipt["same_snapshot_frame_and_observations"] is True
    assert prepared.receipt["HU_columns"] == "artifact_only"
    assert not any(c.startswith("acs_hu_") for c in prepared.frame.table("household"))
    weights = dict(
        zip(
            prepared.frame.table("household").SERIALNO,
            prepared.frame.weights_for("household").values,
            strict=True,
        )
    )
    assert weights["2024GQ0000001"] == 77 and weights["2024GQ0000002"] == 78
    assert prepared.frame.weights_for("household").kind.value == "design"


def test_real_graph_direct_cold_and_zero_kernel_warm(tmp_path, monkeypatch):
    source_dir, snapshots = _fixture(tmp_path, monkeypatch)
    direct = source.prepare_acs_housing_population(source_dir, snapshot_root=snapshots)
    chosen = ("2024HU0000002", "2024GQ0000001")
    selected_ids = direct.frame.table("household").loc[
        direct.frame.table("household").SERIALNO.isin(chosen), "household_id"
    ]
    expected = direct.frame.select(
        direct.frame.person.person_household_id.isin(selected_ids)
    )
    registry = bridge.acs_housing_registry(snapshot_root=snapshots)
    graph = bridge.acs_housing_graph(
        frame_column_declarations(direct.frame), selected_serialnos=chosen
    )
    compiled = compile_graph(graph)
    store = ContentStore(tmp_path / "store", codecs=us_source_codecs())
    result = run_graph(
        compiled,
        store=store,
        kernels=registry,
        sources={bridge.SOURCE_NAME: source_dir},
    )
    frame = result.population(bridge.SELECT_NODE)
    assert source.frame_content_sha256(frame) == source.frame_content_sha256(expected)
    calls = []

    def no_run(self, context):
        calls.append(self.ref)
        raise AssertionError("warm replay executed a kernel")

    monkeypatch.setattr(bridge.ACSHousingCreateKernel, "run", no_run)
    monkeypatch.setattr(bridge.ACSHousingSelectKernel, "run", no_run)
    warm = run_graph(
        compiled,
        store=store,
        kernels=registry,
        sources={bridge.SOURCE_NAME: source_dir},
        resume="require",
    )
    assert source.frame_content_sha256(
        warm.population(bridge.SELECT_NODE)
    ) == source.frame_content_sha256(expected)
    assert calls == []


def test_rehashed_native_code_forgery_and_wrong_context_refuse(tmp_path, monkeypatch):
    source_dir, snapshots = _fixture(tmp_path, monkeypatch)
    prepared = source.prepare_acs_housing_population(
        source_dir, snapshot_root=snapshots
    )
    context = encode_us_frame_context(prepared.frame)
    payload = bridge.encode_acs_housing_evidence(prepared)
    bound = bridge.bind_acs_housing_evidence(
        payload, prepared_receipt=prepared.receipt_json, frame_context=context
    )
    assert type(bound) is bridge.BoundACSHousingEvidence
    assert type(bound) is not source.AuthenticatedACSHousingSource
    changed = bytearray(payload)
    changed[-33] = 255
    changed[-32:] = hashlib.sha256(changed[:-32]).digest()
    with pytest.raises(source.ACSHousingSourceError, match="NATIVE_CODES"):
        bridge.bind_acs_housing_evidence(
            bytes(changed),
            prepared_receipt=prepared.receipt_json,
            frame_context=context,
        )
    with pytest.raises(source.ACSHousingSourceError, match="CONTEXT_BINDING"):
        bridge.bind_acs_housing_evidence(
            payload,
            prepared_receipt=prepared.receipt_json,
            frame_context=context + b" ",
        )


def test_payload_complete_envelope_cap_and_impossible_counts(tmp_path, monkeypatch):
    source_dir, snapshots = _fixture(tmp_path, monkeypatch)
    prepared = source.prepare_acs_housing_population(
        source_dir, snapshot_root=snapshots
    )
    payload = bridge.encode_acs_housing_evidence(prepared)
    monkeypatch.setattr(bridge, "ACS_HU_PAYLOAD_MAX_BYTES", len(payload))
    assert bridge.encode_acs_housing_evidence(prepared) == payload
    monkeypatch.setattr(bridge, "ACS_HU_PAYLOAD_MAX_BYTES", len(payload) - 1)
    with pytest.raises(source.ACSHousingSourceError, match="PAYLOAD_SIZE"):
        bridge.encode_acs_housing_evidence(prepared)
    monkeypatch.setattr(bridge, "ACS_HU_PAYLOAD_MAX_BYTES", 64 * 1024**2)
    offset = len(bridge.MAGIC) + 4
    size = struct.unpack_from("<I", payload, len(bridge.MAGIC))[0]
    header = json.loads(payload[offset : offset + size])
    header["household_rows"] = 2**60
    raw = canonical_json(header)
    changed = (
        bridge.MAGIC + struct.pack("<I", len(raw)) + raw + payload[offset + size : -32]
    )
    changed += hashlib.sha256(changed).digest()
    with pytest.raises(source.ACSHousingSourceError, match="SECTION_SIZE"):
        bridge.bind_acs_housing_evidence(
            changed,
            prepared_receipt=prepared.receipt_json,
            frame_context=encode_us_frame_context(prepared.frame),
        )


def test_prepared_frame_mutation_and_original_source_post_capture(
    tmp_path, monkeypatch
):
    source_dir, snapshots = _fixture(tmp_path, monkeypatch)
    actual_builder = source.build_acs_pums_unit_frame

    def mutate_original_then_build(snapshot_source, **kwargs):
        assert Path(snapshot_source.person_zip).is_relative_to(snapshots)
        # An unprojected income field changes only in the mutable original.
        with zipfile.ZipFile(source_dir / "csv_pus.zip") as archive:
            contents = {i.filename: archive.read(i) for i in archive.infolist()}
        with zipfile.ZipFile(source_dir / "csv_pus.zip", "w") as archive:
            for name, data in contents.items():
                archive.writestr(name, data.replace(b"50000", b"77777"))
        return actual_builder(snapshot_source, **kwargs)

    monkeypatch.setattr(source, "build_acs_pums_unit_frame", mutate_original_then_build)
    prepared = source.prepare_acs_housing_population(
        source_dir, snapshot_root=snapshots
    )
    assert 77777 not in prepared.frame.person.WAGP.to_numpy()
    prepared.frame.person.loc[prepared.frame.person.index[0], "WAGP"] += 1
    with pytest.raises(source.ACSHousingSourceError, match="PREPARED_BINDING"):
        bridge.encode_acs_housing_evidence(prepared)


def test_codec_requires_explicit_store(tmp_path):
    with pytest.raises(source.ACSHousingSourceError, match="EXPLICIT_CONTENT_STORE"):
        source.load_graph_acs_housing_universe(tmp_path)


def test_actual_select_refuses_retyped_split_producer_and_changed_owned_cells(
    tmp_path, monkeypatch
):
    source_dir, snapshots = _fixture(tmp_path, monkeypatch)
    prepared = source.prepare_acs_housing_population(
        source_dir, snapshot_root=snapshots
    )
    captured = []
    original = bridge.ACSHousingSelectKernel.run

    def capture(self, context):
        captured.append(context)
        return original(self, context)

    monkeypatch.setattr(bridge.ACSHousingSelectKernel, "run", capture)
    graph = bridge.acs_housing_graph(
        frame_column_declarations(prepared.frame),
        selected_serialnos=("2024HU0000002", "2024GQ0000001"),
    )
    run_graph(
        compile_graph(graph),
        store=ContentStore(tmp_path / "store", codecs=us_source_codecs()),
        kernels=bridge.acs_housing_registry(snapshot_root=snapshots),
        sources={bridge.SOURCE_NAME: source_dir},
    )
    (context,) = captured
    kernel = bridge.ACSHousingSelectKernel()
    artifacts = dict(context.artifacts)
    value = artifacts["housing_universe"]
    artifacts["housing_universe"] = replace(
        value, type=ArtifactType("invented.other", 1)
    )
    with pytest.raises(source.ACSHousingSourceError, match="ARTIFACT_TYPE"):
        original(kernel, replace(context, artifacts=artifacts))
    other = "1" * 64
    artifacts["housing_universe"] = replace(
        value, producer_key=other, key=opaque_artifact_key(other, "housing_universe")
    )
    with pytest.raises(source.ACSHousingSourceError, match="SAME_PRODUCER"):
        original(kernel, replace(context, artifacts=artifacts))
    for entity, column, replacement in (
        ("person", "source_year", 2023),
        ("person", "SPORDER", 9),
        ("person", "WAGP", 1),
        ("household", "NP", 9),
        ("household", "TYPEHUGQ", 1),
    ):
        tables = {name: table.copy(deep=True) for name, table in context.tables.items()}
        tables[entity].loc[tables[entity].index[0], column] = replacement
        with pytest.raises(source.ACSHousingSourceError, match="FRAME_CELLS"):
            original(kernel, replace(context, tables=tables))
    tables = {name: table.copy(deep=True) for name, table in context.tables.items()}
    tables["person"] = tables["person"].iloc[::-1].copy()
    with pytest.raises(ValueError, match="ids|identity|ID"):
        original(kernel, replace(context, tables=tables))


def test_new_stage_codec_runtime_and_selector_are_explicit():
    import pandas as pd

    from microcosm.build.us_runtime import graph_implementation as impl

    codecs = us_source_codecs()
    assert codecs.get(source.ACS_HU_CODEC) is source.load_graph_acs_housing_universe
    manifest = impl.implementation_manifest(source.ACS_HU_STAGE)
    assert manifest["codecs"] == {
        source.ACS_HU_CODEC: {
            "loader": "microcosm.build.us_runtime.acs_housing_universe_source:load_graph_acs_housing_universe"
        }
    }
    with pd.option_context("mode.string_storage", "python"):
        python_hash = impl.implementation_hash(source.ACS_HU_STAGE)
    with pd.option_context("mode.string_storage", "pyarrow"):
        arrow_hash = impl.implementation_hash(source.ACS_HU_STAGE)
        assert (
            impl.implementation_manifest(source.ACS_HU_STAGE)["runtime_options"][
                "acs_reader_resolved_string_storage"
            ]
            == "pyarrow"
        )
    assert python_hash != arrow_hash


def test_source_destinations_cannot_modify_the_original_roster(tmp_path, monkeypatch):
    source_dir, snapshots = _fixture(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in source_dir.iterdir()}
    with pytest.raises(source.ACSHousingSourceError, match="SNAPSHOT_SOURCE_OVERLAP"):
        source.produce_acs_housing_source(
            source_dir, snapshot_root=source_dir, output_dir=tmp_path / "out"
        )
    with pytest.raises(source.ACSHousingSourceError, match="OUTPUT_SOURCE_OVERLAP"):
        source.produce_acs_housing_source(
            source_dir, snapshot_root=snapshots, output_dir=source_dir / "out"
        )
    assert {p.name: p.read_bytes() for p in source_dir.iterdir()} == before
    assert list(snapshots.iterdir()) == []


def test_disk_refusal_precedes_source_capture(tmp_path, monkeypatch):
    source_dir, snapshots = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        source.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0)
    )

    def no_copy(*args, **kwargs):
        raise AssertionError(
            "insufficient disk must refuse before copying source bytes"
        )

    monkeypatch.setattr(source, "_copy", no_copy)
    with pytest.raises(source.ACSHousingSourceError, match="INSUFFICIENT_DISK"):
        source.produce_acs_housing_source(
            source_dir, snapshot_root=snapshots, output_dir=tmp_path / "out"
        )
    assert list(snapshots.iterdir()) == []
