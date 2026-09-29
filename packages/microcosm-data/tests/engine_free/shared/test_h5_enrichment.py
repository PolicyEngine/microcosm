"""Native source enrichment preserves existing HDF inputs byte for byte."""

from __future__ import annotations

import hashlib
import pickle
import time

import h5py
import numpy as np
import pytest

from microcosm.data.h5_enrichment import (
    ROLE,
    append_native_spm_role,
    compare_h5_enrichment,
    file_sha256,
)

pd = pytest.importorskip("pandas")
pytest.importorskip("tables")


@pytest.fixture
def parent(tmp_path):
    """An indexed pandas table, native entities, and unusual float payloads."""
    path = tmp_path / "parent.h5"
    payloads = np.array(
        [0x7FF8000000000051, 0x8000000000000000, 0, 0x3FF0000000000000],
        dtype=np.uint64,
    ).view(np.float64)
    frames = {
        "person": pd.DataFrame(
            {
                "person_id": np.array([101, 102, 103, 104], dtype=np.int64),
                "person_household_id": np.array([11, 11, 12, 12], dtype=np.int64),
                "person_spm_unit_id": np.array([21, 21, 22, 23], dtype=np.int64),
                "age": np.array([17, 40, 16, 18], dtype=np.int16),
                "original_bool": np.array([True, False, False, True]),
                "person_weight": np.array([1.25, 1.25, 2.5, 2.5]),
                "float_payload": payloads,
                "label": ["alpha", "beta", "gamma", "delta"],
            },
            index=pd.Index([9, 3, 8, 1], name="source_index"),
        ),
        "household": pd.DataFrame(
            {"household_id": [11, 12], "household_weight": [1.25, 2.5]}
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": [21, 22, 23]}),
        "tax_unit": pd.DataFrame({"tax_unit_id": [31, 32]}),
        "family": pd.DataFrame({"family_id": [41, 42]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [51, 52]}),
        "_time_period": pd.Series([2024]),
    }
    with pd.HDFStore(path, mode="w") as store:
        for key, frame in frames.items():
            store.put(key, frame, format="table", data_columns=True)
    with h5py.File(path, "r+") as h5:
        h5.attrs["unchanged_attribute"] = np.int16(7)
        h5["person"].attrs["empty_attribute"] = h5py.Empty("i8")
        auxiliary = h5.create_group("auxiliary")
        auxiliary.create_dataset("array", data=np.array([7, 8], dtype=np.int16))
        auxiliary.create_dataset("scalar", data=payloads[0])
        auxiliary.create_dataset("empty", shape=(0,), dtype="f8")
    return path


@pytest.fixture
def enriched(parent, tmp_path):
    candidate = tmp_path / "candidate.h5"
    role = np.array([True, False, True, False])
    report = append_native_spm_role(
        parent, candidate, role, expected_parent_sha256=file_sha256(parent)
    )
    return parent, candidate, role, report


def _replace_dataset(group, name, datatype, values):
    """Surgically tamper one datatype while retaining storage and attributes."""
    old = group[name]
    creation = old.id.get_create_plist().copy()
    new = h5py.Dataset(
        h5py.h5d.create(
            group.id,
            b"_tampered",
            datatype,
            old.id.get_space(),
            dcpl=creation,
        )
    )
    _write_records(new, values)
    for key in old.attrs:
        attribute = old.attrs.get_id(key)
        copied = h5py.h5a.create(
            new.id, key.encode(), attribute.get_type(), attribute.get_space()
        )
        value = old.attrs[key]
        if not isinstance(value, h5py.Empty):
            copied.write(np.asarray(value), mtype=attribute.get_type())
    del group[name]
    group.move("_tampered", name)


def _write_records(table, values, start=0):
    """Mutate only the requested bytes, avoiding fixed-string conversions."""
    values = np.ascontiguousarray(values)
    selection = table.id.get_space()
    selection.select_hyperslab((start,), (values.size,))
    table.id.write(
        h5py.h5s.create_simple(values.shape),
        selection,
        values,
        mtype=table.id.get_type(),
    )


def test_native_pandas_bool_and_exact_original_inputs(enriched):
    parent, candidate, role, report = enriched
    original = pd.read_hdf(parent, "person")
    actual = pd.read_hdf(candidate, "person")
    assert actual.columns.tolist() == [*original.columns, ROLE]
    assert actual[ROLE].dtype == np.dtype(bool)
    np.testing.assert_array_equal(actual[ROLE].to_numpy(), role)
    pd.testing.assert_frame_equal(actual.drop(columns=ROLE), original, check_exact=True)
    for name in ("household", "spm_unit", "tax_unit", "family", "marital_unit"):
        pd.testing.assert_frame_equal(
            pd.read_hdf(candidate, name), pd.read_hdf(parent, name), check_exact=True
        )
    with h5py.File(parent, "r") as before, h5py.File(candidate, "r") as after:
        a, b = before["person/table"], after["person/table"]
        assert a.dtype.itemsize + 1 == b.dtype.itemsize
        assert b.dtype.names == (*a.dtype.names, ROLE)
        for name in a.dtype.names:
            assert a.dtype.fields[name] == b.dtype.fields[name]
            assert a[name].tobytes() == b[name].tobytes()
        # Equality must retain both NaN payload and signed-zero bit patterns.
        assert b["float_payload"].view(np.uint64).tolist() == [
            0x7FF8000000000051,
            0x8000000000000000,
            0,
            0x3FF0000000000000,
        ]
        assert set(after["person/_i_table"]) == set(before["person/_i_table"])
    # Existing pandas indexes remain usable after the record gets one byte wider.
    pd.testing.assert_frame_equal(
        pd.read_hdf(candidate, "person", where="person_id == 102").drop(columns=ROLE),
        pd.read_hdf(parent, "person", where="person_id == 102"),
        check_exact=True,
    )
    assert file_sha256(parent) == report["parent_sha256"]
    assert report["all_preexisting_variables_exact"] is True
    assert report["persons"] == len(role)
    assert report["role_sha256"] == hashlib.sha256(role.tobytes()).hexdigest()
    assert report["candidate_sha256"] == file_sha256(candidate)


def test_enrichment_is_byte_reproducible(parent, tmp_path):
    first, second = tmp_path / "first.h5", tmp_path / "second.h5"
    role = np.array([True, False, True, False])
    parent_sha = file_sha256(parent)
    append_native_spm_role(parent, first, role, expected_parent_sha256=parent_sha)
    # HDF object timestamps are stored at second precision; cross that boundary.
    time.sleep(1.1)
    append_native_spm_role(parent, second, role, expected_parent_sha256=parent_sha)
    assert file_sha256(first) == file_sha256(second)


@pytest.mark.parametrize(
    "field",
    ["person_id", "person_household_id", "person_spm_unit_id", "person_weight", "age"],
)
def test_old_person_values_cannot_change(enriched, field):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        table = h5["person/table"]
        row = table[0]
        row[field] += 1
        _write_records(table, np.asarray(row).reshape(1))
    with pytest.raises(ValueError, match=f"Existing person field changed: {field}"):
        compare_h5_enrichment(parent, candidate)


@pytest.mark.parametrize("mutation", ["dtype", "shape", "scalar_value"])
def test_nonperson_dtypes_shapes_and_scalar_values_are_exact(enriched, mutation):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        if mutation == "scalar_value":
            h5["auxiliary/scalar"][()] = 1.0
        else:
            del h5["auxiliary/array"]
            h5["auxiliary"].create_dataset(
                "array",
                data=np.array(
                    [7, 8] if mutation == "dtype" else [7],
                    dtype=np.int64 if mutation == "dtype" else np.int16,
                ),
            )
    with pytest.raises(
        ValueError,
        match="HDF dtype changed|HDF shape/storage changed|HDF dataset values changed",
    ):
        compare_h5_enrichment(parent, candidate)


@pytest.mark.parametrize("replacement", [0x7FF8000000000052, 0])
def test_float_payload_and_signed_zero_changes_are_rejected(enriched, replacement):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        table = h5["person/table"]
        index = 0 if replacement else 1
        row = table[index]
        row["float_payload"] = np.array(replacement, dtype=np.uint64).view(np.float64)
        _write_records(table, np.asarray(row).reshape(1), start=index)
    with pytest.raises(
        ValueError, match="Existing person field changed: float_payload"
    ):
        compare_h5_enrichment(parent, candidate)


@pytest.mark.parametrize(
    "target", ["household/table", "person/_i_table/person_id/sorted"]
)
def test_weights_and_existing_indexes_cannot_change(enriched, target):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        dataset = h5[target]
        if target == "household/table":
            row = dataset[0]
            row["household_weight"] += 1
            _write_records(dataset, np.asarray(row).reshape(1))
        else:
            # The index is allocated even when the small table has no full slice.
            dataset.attrs["DIRTY"] = np.uint8(1)
    with pytest.raises(ValueError, match="HDF (dataset values|attributes) changed"):
        compare_h5_enrichment(parent, candidate)


@pytest.mark.parametrize("mutation", ["value", "dtype", "shape", "removed", "added"])
def test_original_attributes_cannot_change(enriched, mutation):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        if mutation == "removed":
            del h5.attrs["unchanged_attribute"]
        elif mutation == "added":
            h5.attrs["unapproved"] = 1
        else:
            h5.attrs["unchanged_attribute"] = {
                "value": np.int16(8),
                "dtype": np.int64(7),
                "shape": np.array([7], dtype=np.int16),
            }[mutation]
    with pytest.raises(ValueError, match="HDF attribute"):
        compare_h5_enrichment(parent, candidate)


@pytest.mark.parametrize(
    "attribute", ["data_columns", "values_cols", "non_index_axes", "info"]
)
def test_registration_changes_must_only_append_the_new_role(enriched, attribute):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        value = pickle.loads(h5["person"].attrs[attribute])
        if attribute == "info":
            value["person_id"] = {"changed": True}
        elif attribute == "non_index_axes":
            value[0][1].reverse()
        else:
            value.reverse()
        h5["person"].attrs[attribute] = np.bytes_(pickle.dumps(value, protocol=0))
    with pytest.raises(ValueError, match="pandas column-registration change"):
        compare_h5_enrichment(parent, candidate)


@pytest.mark.parametrize("attribute", [f"{ROLE}_dtype", "values_cols"])
def test_metadata_bytes_with_wrong_type_and_shape_are_rejected(enriched, attribute):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        owner = h5["person"] if attribute == "values_cols" else h5["person/table"]
        raw = bytes(owner.attrs[attribute])
        owner.attrs[attribute] = np.frombuffer(raw, dtype=np.uint8)
    with pytest.raises(ValueError):
        compare_h5_enrichment(parent, candidate)


@pytest.mark.parametrize("mutation", ["dtype", "order", "role_dtype"])
def test_compound_hdf_types_and_field_order_are_exact(enriched, mutation):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        table = h5["person/table"]
        old_type = table.id.get_type()
        datatype = h5py.h5t.create(h5py.h5t.COMPOUND, old_type.get_size())
        indices = list(range(old_type.get_nmembers()))
        if mutation == "order":
            indices.reverse()
        for index in indices:
            name = old_type.get_member_name(index)
            field_type = old_type.get_member_type(index)
            if (mutation == "dtype" and name == b"original_bool") or (
                mutation == "role_dtype" and name == ROLE.encode()
            ):
                # NumPy still sees uint8, but the required HDF bitfield type is lost.
                field_type = h5py.h5t.STD_U8LE
            datatype.insert(name, old_type.get_member_offset(index), field_type)
        values = table[:]
        _replace_dataset(h5["person"], "table", datatype, values)
    with pytest.raises(ValueError, match="Person dtype differs"):
        compare_h5_enrichment(parent, candidate)


@pytest.mark.parametrize(
    "mutation", ["extra_group", "extra_dataset", "removed", "soft_link", "alias"]
)
def test_object_inventory_is_exact(enriched, mutation):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        if mutation == "extra_group":
            h5.create_group("extra")
        elif mutation == "extra_dataset":
            h5.create_dataset("extra", data=[1])
        elif mutation == "removed":
            del h5["family"]
        elif mutation == "soft_link":
            h5["extra"] = h5py.SoftLink("/person")
        else:
            h5["extra"] = h5["person"]
    with pytest.raises(
        ValueError, match="added or removed|Nonlocal HDF link|Aliased HDF object"
    ):
        compare_h5_enrichment(parent, candidate)


def test_row_order_cannot_change(enriched):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        table = h5["person/table"]
        _write_records(table, table[:][::-1])
    with pytest.raises(ValueError, match="Existing person field changed"):
        compare_h5_enrichment(parent, candidate)


def test_non_boolean_role_values_are_rejected(enriched):
    parent, candidate, _, _ = enriched
    with h5py.File(candidate, "r+") as h5:
        table = h5["person/table"]
        row = table[0]
        row[ROLE] = 2
        _write_records(table, np.asarray(row).reshape(1))
    with pytest.raises(ValueError, match="Native SPM role is not Boolean"):
        compare_h5_enrichment(parent, candidate)


@pytest.mark.parametrize(
    "role",
    [np.ones(4, dtype=np.uint8), np.ones((4, 1), dtype=bool), np.ones(3, dtype=bool)],
)
def test_invalid_role_input_leaves_no_candidate(parent, tmp_path, role):
    candidate = tmp_path / "candidate.h5"
    parent_sha = file_sha256(parent)
    with pytest.raises(ValueError, match="one-dimensional Boolean|Role coverage"):
        append_native_spm_role(
            parent, candidate, role, expected_parent_sha256=parent_sha
        )
    assert not candidate.exists()
    assert file_sha256(parent) == parent_sha


def test_wrong_parent_hash_leaves_no_candidate(parent, tmp_path):
    candidate = tmp_path / "candidate.h5"
    with pytest.raises(ValueError, match="Parent H5 SHA-256 mismatch"):
        append_native_spm_role(
            parent, candidate, np.ones(4, dtype=bool), expected_parent_sha256="0" * 64
        )
    assert not candidate.exists()


@pytest.mark.parametrize("destination", ["existing", "parent", "symlink"])
def test_existing_destination_is_never_overwritten(parent, tmp_path, destination):
    candidate = tmp_path / "candidate.h5"
    if destination == "existing":
        candidate.write_bytes(b"existing artifact")
    elif destination == "parent":
        candidate = parent
    else:
        candidate.symlink_to(parent)
    original = candidate.read_bytes()
    with pytest.raises(FileExistsError):
        append_native_spm_role(
            parent,
            candidate,
            np.ones(4, dtype=bool),
            expected_parent_sha256=file_sha256(parent),
        )
    assert candidate.read_bytes() == original


def test_already_enriched_parent_is_rejected(enriched, tmp_path):
    _, candidate, role, _ = enriched
    second = tmp_path / "second.h5"
    with pytest.raises(ValueError, match="already contains"):
        append_native_spm_role(
            candidate, second, role, expected_parent_sha256=file_sha256(candidate)
        )
    assert not second.exists()
