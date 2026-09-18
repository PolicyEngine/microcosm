"""Append a native pandas person input without rewriting existing variables.

BuildP uses a compound ``person/table``, not variable/period HDF groups. Its
record type must grow by one field. All original field types, offsets and
bytes stay exact; only four pandas column-registration attributes change.
Every other HDF object and attribute is compared without exemptions.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import shutil
from pathlib import Path

import h5py
import numpy as np

ROLE = "is_spm_independent_minor_role"
_TABLE = "person/table"
_REGISTRATION = {"data_columns", "values_cols", "non_index_axes", "info"}
_CHUNK_ROWS = 4096


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("HDF column metadata must contain only primitive containers")


def _registration_value(name: str, raw: object) -> bytes:
    value = _MetadataUnpickler(io.BytesIO(bytes(raw))).load()
    if name in {"data_columns", "values_cols"}:
        if not isinstance(value, list) or ROLE in value:
            raise ValueError(f"Unsupported person {name}")
        value.append(ROLE)
    elif name == "non_index_axes":
        if (
            not isinstance(value, list)
            or len(value) != 1
            or value[0][0] != 1
            or not isinstance(value[0][1], list)
            or ROLE in value[0][1]
        ):
            raise ValueError("Unsupported person non_index_axes")
        value[0][1].append(ROLE)
    else:
        if not isinstance(value, dict) or ROLE in value:
            raise ValueError("Unsupported person info")
        value[ROLE] = {}
    return pickle.dumps(value, protocol=0)


def _copy_attribute(source, destination, name: str) -> None:
    attribute = source.attrs.get_id(name)
    copied = h5py.h5a.create(
        destination.id, name.encode(), attribute.get_type(), attribute.get_space()
    )
    if attribute.shape is not None:
        value = np.empty(attribute.shape, dtype=attribute.dtype)
        attribute.read(value, mtype=attribute.get_type())
        copied.write(value, mtype=attribute.get_type())


def _new_table_attributes(field_count: int) -> dict[str, object]:
    return {
        f"FIELD_{field_count}_NAME": ROLE,
        f"FIELD_{field_count}_FILL": np.uint8(0),
        f"{ROLE}_dtype": "bool",
        f"{ROLE}_kind": np.bytes_(pickle.dumps([ROLE], protocol=0)),
        f"{ROLE}_meta": np.bytes_(b"N."),
    }


def _check_supported_table(table: h5py.Dataset) -> None:
    if (
        table.ndim != 1
        or not table.dtype.names
        or "person_id" not in table.dtype.names
        or table.chunks is None
        or table.compression is not None
        or table.shuffle
        or table.fletcher32
        or table.scaleoffset is not None
        or table.is_virtual
        or table.external
    ):
        raise ValueError("Expected the unfiltered BuildP native pandas person table")
    if ROLE in table.dtype.names:
        raise ValueError("Parent already contains the native SPM role")
    if any(table.dtype[name].hasobject for name in table.dtype.names):
        raise ValueError("Object/reference fields are not supported")


def append_native_spm_role(
    parent_h5: str | Path,
    candidate_h5: str | Path,
    role: np.ndarray,
    *,
    expected_parent_sha256: str,
) -> dict:
    """Create a new file, append one native field, and exhaustively verify it.

    Destination must not exist. The caller owns a private staging directory;
    failure removes only the file this call created. The parent is read-only.
    No HDF repacking, pandas table rewrite, or weight operation occurs.
    """
    parent_h5, candidate_h5 = Path(parent_h5), Path(candidate_h5)
    if file_sha256(parent_h5) != expected_parent_sha256:
        raise ValueError("Parent H5 SHA-256 mismatch")
    role = np.asarray(role)
    if role.dtype != np.dtype(bool) or role.ndim != 1:
        raise ValueError("Role must be a one-dimensional Boolean array")
    with h5py.File(parent_h5, "r") as parent:
        _check_supported_table(parent[_TABLE])
        if role.shape != parent[_TABLE].shape:
            raise ValueError("Role coverage does not match the parent person rows")
    # Exclusive creation also refuses aliases/symlinks to the parent.
    descriptor = os.open(candidate_h5, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as destination, parent_h5.open("rb") as source:
            shutil.copyfileobj(source, destination, 1024 * 1024)
        if file_sha256(candidate_h5) != expected_parent_sha256:
            raise ValueError("Parent changed during the verified copy")
        with h5py.File(candidate_h5, "r+") as candidate:
            old = candidate[_TABLE]
            compound = old.id.get_type().copy()
            old_size = compound.get_size()
            compound.set_size(old_size + 1)
            # PyTables Boolean columns are HDF bitfields, not enums/integers.
            compound.insert(ROLE.encode(), old_size, h5py.h5t.NATIVE_B8)
            creation = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            creation.set_chunk(old.chunks)
            creation.set_obj_track_times(False)
            group = candidate["person"]
            temporary = "_spm_role_enriched_table"
            if temporary in group:
                raise ValueError("Unexpected temporary HDF object in parent")
            new = h5py.Dataset(
                h5py.h5d.create(
                    group.id,
                    temporary.encode(),
                    compound,
                    old.id.get_space(),
                    dcpl=creation,
                )
            )
            for start in range(0, len(old), _CHUNK_ROWS):
                stop = min(start + _CHUNK_ROWS, len(old))
                original = _raw_rows(old, start, stop)
                extended = np.empty(stop - start, dtype=new.dtype)
                extended.view(np.uint8).reshape(-1, old_size + 1)[:, :old_size] = (
                    original.view(np.uint8).reshape(-1, old_size)
                )
                extended[ROLE] = role[start:stop]
                memory = h5py.h5s.create_simple((stop - start,))
                selection = new.id.get_space()
                selection.select_hyperslab((start,), (stop - start,))
                new.id.write(memory, selection, extended, mtype=compound)
            for name in old.attrs:
                _copy_attribute(old, new, name)
            for name, value in _new_table_attributes(len(old.dtype.names)).items():
                if isinstance(value, str):
                    new.attrs.create(
                        name, value, dtype=h5py.string_dtype("utf-8", len(value))
                    )
                else:
                    new.attrs[name] = value
            for name in sorted(_REGISTRATION):
                group.attrs[name] = np.bytes_(
                    _registration_value(name, group.attrs[name])
                )
            del candidate[_TABLE]
            group.move(temporary, "table")
        report = compare_h5_enrichment(parent_h5, candidate_h5)
        if report["parent_sha256"] != expected_parent_sha256:
            raise ValueError("Parent changed during enrichment")
        if report["role_sha256"] != hashlib.sha256(role.tobytes()).hexdigest():
            raise ValueError("Written native role differs from the source derivation")
        return report
    except BaseException:
        candidate_h5.unlink(missing_ok=True)
        raise


def _objects(root: h5py.File) -> dict:
    objects = {"": root}
    addresses = {h5py.h5o.get_info(root.id).addr}

    def descend(group, prefix=""):
        for name in group:
            path = f"{prefix}/{name}".lstrip("/")
            if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                raise ValueError(f"Nonlocal HDF link at {path}")
            child = group[name]
            address = h5py.h5o.get_info(child.id).addr
            if address in addresses:
                raise ValueError(f"Aliased HDF object at {path}")
            addresses.add(address)
            objects[path] = child
            if isinstance(child, h5py.Group):
                descend(child, path)

    descend(root)
    return objects


def _value_bytes(value: object) -> bytes:
    if isinstance(value, h5py.Empty):
        return b""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object/reference HDF values are not supported")
    return array.tobytes()


def _raw_rows(dataset: h5py.Dataset, start: int, stop: int) -> np.ndarray:
    """Read with the file type to prevent HDF string-padding conversions."""
    shape = (stop - start, *dataset.shape[1:])
    result = np.empty(shape, dtype=dataset.dtype)
    memory = h5py.h5s.create_simple(shape)
    selection = dataset.id.get_space()
    selection.select_hyperslab((start, *(0 for _ in dataset.shape[1:])), shape)
    dataset.id.read(memory, selection, result, mtype=dataset.id.get_type())
    return result


def _raw_selection(dataset: h5py.Dataset, selection) -> np.ndarray:
    if selection != ():
        return _raw_rows(dataset, selection.start, selection.stop)
    result = np.empty((), dtype=dataset.dtype)
    dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, result, mtype=dataset.id.get_type())
    return result


def _attribute_bytes(attribute) -> bytes:
    if attribute.shape is None:
        return b""
    value = np.empty(attribute.shape, dtype=attribute.dtype)
    attribute.read(value, mtype=attribute.get_type())
    return _value_bytes(value)


def _compare_attributes(path: str, old, new) -> int:
    additions = _new_table_attributes(len(old.dtype.names)) if path == _TABLE else {}
    if set(new.attrs) != set(old.attrs) | set(additions):
        raise ValueError(f"HDF attributes changed at {path}")
    for name in old.attrs:
        if path == "person" and name in _REGISTRATION:
            expected = _registration_value(name, old.attrs[name])
            attribute = new.attrs.get_id(name)
            if (
                attribute.shape != ()
                or attribute.get_type().get_class() != h5py.h5t.STRING
                or attribute.dtype != np.dtype(f"S{len(expected)}")
                or attribute.get_type().get_cset() != h5py.h5t.CSET_ASCII
                or bytes(new.attrs[name]) != expected
            ):
                raise ValueError(
                    f"Unexpected pandas column-registration change: {name}"
                )
            continue
        a, b = old.attrs.get_id(name), new.attrs.get_id(name)
        if (
            a.get_type() != b.get_type()
            or a.shape != b.shape
            or _attribute_bytes(a) != _attribute_bytes(b)
        ):
            raise ValueError(f"HDF attribute changed at {path}:{name}")
    for name, expected in additions.items():
        actual = new.attrs[name]
        attribute = new.attrs.get_id(name)
        if isinstance(expected, str):
            expected = expected.encode()
            correct_type = (
                attribute.get_type().get_class() == h5py.h5t.STRING
                and attribute.dtype == np.dtype(f"S{len(expected)}")
                and attribute.get_type().get_cset() == h5py.h5t.CSET_UTF8
            )
        elif isinstance(expected, np.bytes_):
            correct_type = (
                attribute.get_type().get_class() == h5py.h5t.STRING
                and attribute.dtype == np.asarray(expected).dtype
                and attribute.get_type().get_cset() == h5py.h5t.CSET_ASCII
            )
        else:
            correct_type = attribute.get_type() == h5py.h5t.NATIVE_UINT8
        if (
            not correct_type
            or attribute.shape != ()
            or _value_bytes(actual) != _value_bytes(expected)
        ):
            raise ValueError(f"Invalid added role metadata: {name}")
    return len(old.attrs)


def compare_h5_enrichment(parent_h5: str | Path, candidate_h5: str | Path) -> dict:
    """Prove exact old-variable identity and the single allowed schema extension.

    Comparisons include HDF datatype identities (including bitfields), NaN
    payloads, signed zero, all indexes, attributes, and row order. The result
    contains only aggregate counts and hashes, never person records.
    """
    parent_sha = file_sha256(parent_h5)
    candidate_sha = file_sha256(candidate_h5)
    old_fields = datasets = attributes = groups = 0
    role_digest = hashlib.sha256()
    with h5py.File(parent_h5, "r") as parent, h5py.File(candidate_h5, "r") as candidate:
        old_objects, new_objects = _objects(parent), _objects(candidate)
        if old_objects.keys() != new_objects.keys():
            raise ValueError("HDF groups/datasets added or removed")
        _check_supported_table(parent[_TABLE])
        for path, old in old_objects.items():
            new = new_objects[path]
            if type(old) is not type(new):
                raise ValueError(f"HDF object kind changed at {path}")
            attributes += _compare_attributes(path, old, new)
            if isinstance(old, h5py.Group):
                groups += 1
                continue
            datasets += 1
            if (
                old.shape != new.shape
                or old.maxshape != new.maxshape
                or old.chunks != new.chunks
                or old.compression != new.compression
                or old.compression_opts != new.compression_opts
                or old.shuffle != new.shuffle
                or old.fletcher32 != new.fletcher32
                or old.scaleoffset != new.scaleoffset
                or new.is_virtual
                or new.external
            ):
                raise ValueError(f"HDF shape/storage changed at {path}")
            if path == _TABLE:
                before_type, after_type = old.id.get_type(), new.id.get_type()
                expected_type = before_type.copy()
                expected_type.set_size(before_type.get_size() + 1)
                expected_type.insert(
                    ROLE.encode(), before_type.get_size(), h5py.h5t.NATIVE_B8
                )
                if expected_type != after_type or new.dtype.names != (
                    *old.dtype.names,
                    ROLE,
                ):
                    raise ValueError(
                        "Person dtype differs beyond the appended role field"
                    )
                old_fields = len(old.dtype.names)
            elif old.id.get_type() != new.id.get_type() or old.dtype != new.dtype:
                raise ValueError(f"HDF dtype changed at {path}")
            slices = (
                [()]
                if old.ndim == 0
                else [
                    slice(i, min(i + _CHUNK_ROWS, len(old)))
                    for i in range(0, len(old), _CHUNK_ROWS)
                ]
            )
            for selection in slices:
                if path == _TABLE:
                    a = _raw_rows(old, selection.start, selection.stop)
                    b = _raw_rows(new, selection.start, selection.stop)
                    for name in old.dtype.names:
                        if a[name].tobytes() != b[name].tobytes():
                            raise ValueError(f"Existing person field changed: {name}")
                    role = b[ROLE]
                    if not np.isin(role, [0, 1]).all():
                        raise ValueError("Native SPM role is not Boolean")
                    role_digest.update(role.tobytes())
                elif _value_bytes(_raw_selection(old, selection)) != _value_bytes(
                    _raw_selection(new, selection)
                ):
                    raise ValueError(f"Existing HDF dataset values changed: {path}")
        persons = len(parent[_TABLE])
    if (
        file_sha256(parent_h5) != parent_sha
        or file_sha256(candidate_h5) != candidate_sha
    ):
        raise ValueError("HDF changed during the exhaustive comparison")
    return {
        "schema_version": 1,
        "parent_sha256": parent_sha,
        "candidate_sha256": candidate_sha,
        "all_preexisting_variables_exact": True,
        "groups_checked_including_root": groups,
        "datasets_checked": datasets,
        "preexisting_attributes_checked": attributes,
        "person_fields_checked_including_index": old_fields,
        "persons": persons,
        "role_sha256": role_digest.hexdigest(),
        "physical_schema_extension": {
            "dataset": _TABLE,
            "appended_field": ROLE,
            "storage": "HDF bitfield8; pandas bool",
            "record_size_increase_bytes": 1,
            "column_registration_attributes": sorted(_REGISTRATION),
            "all_other_objects_and_attributes_exact": True,
        },
    }
