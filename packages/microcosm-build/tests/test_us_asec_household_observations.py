"""Exact source observations, with no household concept or price conversion."""

import copy
import hashlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime import asec_household_observations as observations
from microcosm.frame import Frame


def _checkpoint_fixtures():
    path = Path(__file__).with_name("test_us_asec_checkpoint.py")
    spec = importlib.util.spec_from_file_location("household_v4_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = _checkpoint_fixtures()
COLUMNS = (
    "HTOTVAL",
    "H_LIVQRT",
    "HRHTYPE",
    "H_HHTYPE",
    "H_SEQ",
    "H_TENURE",
    "GESTFIPS",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame():
    base = fixtures._v4_frame()
    tables = {entity: base.table(entity).copy() for entity in base.entities}
    person = tables["person"]
    person["source_household_id"] = np.array([7, 7], dtype=np.int64)
    extra = person.iloc[[0]].copy()
    extra["person_id"] = 3
    extra["PERIDNUM"] = "0000000000000000000003"
    person = pd.concat([person.iloc[[1]], extra, person.iloc[[0]]], ignore_index=True)
    person.index = pd.Index([91, 91, 14], name="original_person_row")
    tables["person"] = person
    tables["household"]["state_fips"] = np.array([6, 36], dtype=np.int64)
    tables["household"]["H_TENURE"] = np.array([0, 2], dtype=np.int64)
    return Frame(
        tables,
        base.schema,
        {"household": base.weights_for("household")},
        pd.Series(["2023", "2022", "2022"], index=person.index, dtype="string"),
        metadata={"fixture": {"preserved": True}},
        mass_log=base.mass_log,
    )


def _raw(year, *, income=None):
    return pd.DataFrame(
        {
            "HTOTVAL": np.array(
                [12, -9007199254741003 if year == 2022 else 0], dtype=np.int64
            )
            if income is None
            else np.array([12, income], dtype=np.int64),
            "H_LIVQRT": np.array([1, 11 if year == 2022 else 0], dtype=np.int64),
            "HRHTYPE": np.array([1, 9 if year == 2022 else 0], dtype=np.int64),
            "H_HHTYPE": np.array([1, 0], dtype=np.int64),
            "H_SEQ": np.array([99, 7], dtype=np.int64),
            "H_TENURE": np.array([1, 0 if year == 2022 else 2], dtype=np.int64),
            "GESTFIPS": np.array([1, 6 if year == 2022 else 36], dtype=np.int64),
            # A real object block is present but never needed by the numeric reader.
            "UNRELATED": pd.Series(
                [{"not": "unpickled"}, {"nor": "this"}], dtype=object
            ),
        }
    )


def _inputs(tmp_path, *, change=None):
    frame = _frame()
    sources = []
    for year in (2022, 2023):
        raw = _raw(year)
        if change is not None:
            raw = change(raw, year)
        path = tmp_path / f"households-{year}.h5"
        with pytest.warns(pd.errors.PerformanceWarning):
            raw.to_hdf(path, key="household", mode="w", format="fixed")
        sources.append(
            observations.AsecHouseholdObservationSource(year, path, _sha(path))
        )
    metadata = fixtures._v4_binding(frame)
    metadata["source_receipt"]["sources"] = [
        {"year": source.year, "path": str(source.path), "sha256": source.sha256}
        for source in sources
    ]
    return frame, metadata, sources


def _run(frame, metadata, sources):
    return observations.with_asec_household_observations(
        frame,
        checkpoint_metadata=metadata,
        checkpoint_sha256="e" * 64,
        sources=sources,
    )


def test_exact_year_key_join_preserves_full_frame_and_signed_integer_observations(
    tmp_path,
):
    frame, metadata, sources = _inputs(tmp_path)
    original = {
        entity: frame.table(entity).copy(deep=True) for entity in frame.entities
    }
    original_metadata = copy.deepcopy(metadata)
    result = _run(frame, metadata, sources)
    actual = result.frame
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity).loc[:, original[entity].columns], original[entity]
        )
        pd.testing.assert_frame_equal(frame.table(entity), original[entity])
        np.testing.assert_array_equal(
            actual.resolve_weights(entity).values, frame.resolve_weights(entity).values
        )
    pd.testing.assert_series_equal(actual.strata, frame.strata)
    assert actual.metadata == frame.metadata and actual.mass_log == frame.mass_log
    assert metadata == original_metadata
    assert actual.table("household").asec_HTOTVAL.tolist() == [-9007199254741003, 0]
    assert actual.table("household").asec_H_LIVQRT.tolist() == [11, 0]
    assert actual.table("household").asec_HRHTYPE.tolist() == [9, 0]
    assert all(
        actual.table("household")[f"asec_{column}"].dtype == np.dtype("int64")
        for column in COLUMNS
    )
    receipt = result.receipt
    assert receipt["input_checkpoint_sha256"] == "e" * 64
    assert receipt["artifact_kind"] == "microcosm.asec_household_observation_attachment"
    assert (
        receipt["household_identity"]["ordered_ids_sha256"]
        == hashlib.sha256(np.array([1, 2], dtype="<i8").tobytes()).hexdigest()
    )
    assert [
        (
            row["income_year"],
            row["source_rows"],
            row["joined_rows"],
            row["unreferenced_source_rows"],
        )
        for row in receipt["sources"]
    ] == [(2022, 2, 1, 1), (2023, 2, 1, 1)]
    path = tmp_path / "augmented.h5"
    write_frame_checkpoint(path, actual, metadata={"household_observations": receipt})
    restored = load_frame_checkpoint(path)
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            restored.frame.table(entity), actual.table(entity)
        )
    assert restored.metadata["household_observations"] == receipt


def test_reader_never_reads_object_blocks_person_tables_or_row_index_values(
    tmp_path, monkeypatch
):
    frame, metadata, sources = _inputs(tmp_path)
    accessed = []
    original = h5py.Dataset.__getitem__

    def checked(dataset, key):
        accessed.append(dataset.name)
        assert "/person" not in dataset.name and not dataset.name.endswith("axis1")
        assert not (dataset.name.endswith("_values") and dataset.dtype.kind == "O")
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked)
    _run(frame, metadata, sources)
    assert any(name.endswith("_values") for name in accessed)


@pytest.mark.parametrize(
    "damage", ["duplicate", "missing", "state", "tenure", "float", "column"]
)
def test_raw_source_join_and_storage_refusals(tmp_path, damage):
    def change(raw, year):
        if year != 2022:
            return raw
        if damage == "duplicate":
            raw["H_SEQ"] = [7, 7]
        elif damage == "missing":
            raw["H_SEQ"] = [8, 9]
        elif damage == "state":
            raw.loc[1, "GESTFIPS"] = 48
        elif damage == "tenure":
            raw.loc[1, "H_TENURE"] = 2
        elif damage == "float":
            raw["HTOTVAL"] = raw.HTOTVAL.astype(float)
        else:
            raw = raw.drop(columns="HTOTVAL")
        return raw

    frame, metadata, sources = _inputs(tmp_path, change=change)
    with pytest.raises(ValueError):
        _run(frame, metadata, sources)


@pytest.mark.parametrize(
    "damage",
    ["duplicate_year", "missing_year", "pin", "bad_pin", "bool_year", "float_year"],
)
def test_explicit_source_and_input_authentication_contract(tmp_path, damage):
    frame, metadata, sources = _inputs(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        if damage == "duplicate_year":
            sources.append(sources[0])
        elif damage == "missing_year":
            sources.pop()
        elif damage == "pin":
            sources[0] = replace(sources[0], sha256="a" * 64)
        elif damage == "bad_pin":
            sources[0] = replace(sources[0], sha256="invalid")
        elif damage == "bool_year":
            sources[0] = replace(sources[0], year=True)
        else:
            sources[0] = replace(sources[0], year=2022.0)
        _run(frame, metadata, sources)


@pytest.mark.parametrize(
    "damage",
    ["mixed_key", "shared_key", "nullable_key", "v3", "identity", "already_attached"],
)
def test_actual_frame_and_v4_contract_refusals(tmp_path, damage):
    frame, metadata, sources = _inputs(tmp_path)
    if damage == "mixed_key":
        frame.person.iloc[1, frame.person.columns.get_loc("source_household_id")] = 9
    elif damage == "shared_key":
        frame.person["source_year"] = 2022
        # Exercise the pure join guard, independently of the v4 year-coverage
        # validator, which would also refuse this intentionally altered input.
        with pytest.raises(ValueError, match="multiple normalized households"):
            observations._household_keys(frame)
        return
    elif damage == "nullable_key":
        frame.person["source_household_id"] = pd.array([7, None, 7], dtype="Int64")
    elif damage == "v3":
        metadata["schema_version"] = 3
    elif damage == "identity":
        metadata["identity"]["sha256"] = "a" * 64
    else:
        frame.table("household")["asec_HTOTVAL"] = 0
    if damage == "mixed_key":
        metadata["identity"] = fixtures.frame_identity(frame).to_payload()
        metadata["source_construction_identity"] = metadata["identity"]
    with pytest.raises((TypeError, ValueError)):
        _run(frame, metadata, sources)


def test_relocated_identical_source_bytes_keep_same_semantic_receipt(tmp_path):
    frame, metadata, sources = _inputs(tmp_path)
    first = _run(frame, metadata, sources)
    relocated = []
    for source in sources:
        path = tmp_path / f"moved-{source.year}.h5"
        path.write_bytes(source.path.read_bytes())
        relocated.append(replace(source, path=path))
    second = _run(frame, metadata, relocated)
    assert second.receipt == first.receipt
    pd.testing.assert_frame_equal(
        second.frame.table("household"), first.frame.table("household")
    )


def test_source_specific_integer_columns_promote_to_nullable_on_existing_union(
    tmp_path,
):
    from microcosm.build.us_runtime.graph_sources import _canonical_assembly
    from microcosm.build.us_runtime.spine_assembly import assemble_spines

    frame, metadata, sources = _inputs(tmp_path)
    augmented = _run(frame, metadata, sources).frame
    # A second tiny peer source has no ASEC household observations. This is
    # an engineering absence fixture, not an actual ACS population or result.
    peer = _frame()
    combined = assemble_spines(
        {"cps": augmented, "acs": peer},
        household_mass_shares={"cps": 0.5, "acs": 0.5},
        mass_anchor_channel="cps",
    )
    transitions = _canonical_assembly(combined, (augmented, peer))
    assert any(
        row["column"] == "asec_HTOTVAL" and row["to"] == "Int64" for row in transitions
    )
    for column in COLUMNS:
        values = combined.table("household")[f"asec_{column}"]
        assert values.dtype == pd.Int64Dtype()
        assert values.isna().tolist() == [False, False, True, True]
        np.testing.assert_array_equal(
            values.iloc[:2].to_numpy(),
            augmented.table("household")[f"asec_{column}"].to_numpy(),
        )
    path = tmp_path / "nullable-union.h5"
    write_frame_checkpoint(path, combined)
    pd.testing.assert_frame_equal(
        load_frame_checkpoint(path).frame.table("household"),
        combined.table("household"),
    )


def test_receipt_access_does_not_mutate_result_authority(tmp_path):
    result = _run(*_inputs(tmp_path))
    first = result.receipt
    first["sources"][0]["sha256"] = "a" * 64
    assert result.receipt != first


def test_raw_household_keys_above_float_precision_remain_exact(tmp_path):
    key = 2**53 + 19

    def change(raw, _year):
        raw.loc[1, "H_SEQ"] = key
        return raw

    frame, metadata, sources = _inputs(tmp_path, change=change)
    frame.person["source_household_id"] = key
    metadata["identity"] = fixtures.frame_identity(frame).to_payload()
    metadata["source_construction_identity"] = metadata["identity"]
    result = _run(frame, metadata, sources)
    assert result.frame.table("household").asec_H_SEQ.tolist() == [key, key]


def test_source_replacement_during_numeric_read_is_refused(tmp_path, monkeypatch):
    frame, metadata, sources = _inputs(tmp_path)
    original = h5py.Dataset.__getitem__
    changed = False

    def replacing(dataset, key):
        nonlocal changed
        result = original(dataset, key)
        if dataset.name.endswith("_values") and not changed:
            changed = True
            replacement = tmp_path / "replacement.h5"
            replacement.write_bytes(sources[0].path.read_bytes())
            replacement.replace(sources[0].path)
        return result

    monkeypatch.setattr(h5py.Dataset, "__getitem__", replacing)
    with pytest.raises(ValueError, match="changed"):
        _run(frame, metadata, sources)
    assert changed


@pytest.mark.parametrize(
    "damage",
    [
        "external",
        "soft",
        "virtual",
        "external_raw",
        "shape",
        "duplicate_metadata",
        "not_fixed",
    ],
)
def test_ambiguous_or_external_hdf_storage_is_refused(tmp_path, damage):
    frame, metadata, sources = _inputs(tmp_path)
    path = sources[0].path
    with h5py.File(path, "r+") as handle:
        group = handle["household"]
        numeric_block = next(
            index
            for index in range(int(group.attrs["nblocks"]))
            if b"HTOTVAL" in group[f"block{index}_items"][()]
        )
        numeric_name = f"block{numeric_block}_values"
        if damage in {"external", "soft"}:
            del group[numeric_name]
            group[numeric_name] = (
                h5py.ExternalLink(str(sources[1].path), f"/household/{numeric_name}")
                if damage == "external"
                else h5py.SoftLink("/household/axis1")
            )
        elif damage == "shape":
            del group[numeric_name]
            group[numeric_name] = np.ones((1, 7), dtype=np.int64)
        elif damage == "virtual":
            shape = group[numeric_name].shape
            del group[numeric_name]
            layout = h5py.VirtualLayout(shape=shape, dtype=np.int64)
            layout[:] = h5py.VirtualSource(
                str(sources[1].path), f"/household/{numeric_name}", shape=shape
            )
            group.create_virtual_dataset(numeric_name, layout)
        elif damage == "external_raw":
            shape = group[numeric_name].shape
            del group[numeric_name]
            group.create_dataset(
                numeric_name,
                shape=shape,
                dtype=np.int64,
                external=[
                    (str(tmp_path / "external-values.bin"), 0, h5py.h5f.UNLIMITED)
                ],
            )
        elif damage == "duplicate_metadata":
            group["axis0"][0] = group["axis0"][1]
        else:
            group.attrs["pandas_type"] = b"frame_table"
    sources[0] = replace(sources[0], sha256=_sha(path))
    metadata["source_receipt"]["sources"][0]["sha256"] = sources[0].sha256
    with pytest.raises(ValueError):
        _run(frame, metadata, sources)
