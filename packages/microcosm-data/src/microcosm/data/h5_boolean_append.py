"""Prove a Boolean-column append to pandas tables changed nothing else.

The verifier behind two lanes: the local donor receipt qualification
(``tools/build_us_acs_donor_receipt_qualification.py``), which writes the
appended child and records this report, and the reported-receipt
source-enrichment release contract
(:mod:`microcosm.data.source_enrichment`), which replays it against the actual
parent and child H5 files before a release can be certified or published. One
implementation serves both, so the report a qualification records is the report
the release contract recomputes.

The writer stays with the qualification tool. This module holds what the
writer and the verifier share — the expected per-column table attributes, the
pandas registration rewrite and the supported-table rules — and the verifier
itself. It generalises the single-column comparison in
:mod:`microcosm.data.h5_enrichment` (the native SPM role lane, which is
unchanged) to any ordered set of HDF bitfield columns across several groups,
reusing that module's byte-level primitives rather than re-deriving them.
"""

from __future__ import annotations

import hashlib
import io
import pickle
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np

from microcosm.data.h5_enrichment import (
    _attribute_bytes,
    _objects,
    _raw_rows,
    _raw_selection,
    _value_bytes,
    file_sha256,
)

_CHUNK_ROWS = 4096
_REGISTRATION = ("data_columns", "info", "non_index_axes", "values_cols")


class BooleanAppendError(ValueError):
    """A refusal. Its message never contains a row value or identifier."""


def _refuse(message: str) -> None:
    raise BooleanAppendError(message)


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise BooleanAppendError(
            "HDF column metadata must contain only primitive containers"
        )


def _registration_value(name: str, raw: object, columns: Sequence[str]) -> bytes:
    value = _MetadataUnpickler(io.BytesIO(bytes(raw))).load()
    if name in {"data_columns", "values_cols"}:
        if not isinstance(value, list) or any(column in value for column in columns):
            _refuse(f"Unsupported HDF column registration: {name}")
        value.extend(columns)
    elif name == "non_index_axes":
        if (
            not isinstance(value, list)
            or len(value) != 1
            or value[0][0] != 1
            or not isinstance(value[0][1], list)
            or any(column in value[0][1] for column in columns)
        ):
            _refuse("Unsupported HDF non_index_axes")
        value[0][1].extend(columns)
    else:
        if not isinstance(value, dict) or any(column in value for column in columns):
            _refuse("Unsupported HDF info")
        for column in columns:
            value[column] = {}
    return pickle.dumps(value, protocol=0)


def _new_table_attributes(field_count: int, columns: Sequence[str]) -> dict:
    attributes: dict[str, object] = {}
    for offset, column in enumerate(columns):
        index = field_count + offset
        attributes[f"FIELD_{index}_NAME"] = column
        attributes[f"FIELD_{index}_FILL"] = np.uint8(0)
        attributes[f"{column}_dtype"] = "bool"
        attributes[f"{column}_kind"] = np.bytes_(pickle.dumps([column], protocol=0))
        attributes[f"{column}_meta"] = np.bytes_(b"N.")
    return attributes


def _check_supported_table(
    table: h5py.Dataset,
    columns: Sequence[str],
    *,
    path: str,
) -> None:
    if (
        table.ndim != 1
        or not table.dtype.names
        or table.chunks is None
        or table.compression is not None
        or table.shuffle
        or table.fletcher32
        or table.scaleoffset is not None
        or table.is_virtual
        or table.external
    ):
        _refuse(f"Expected an unfiltered native pandas table at {path}")
    if any(column in table.dtype.names for column in columns):
        _refuse(f"Parent already carries a qualified column at {path}")
    if any(table.dtype[name].hasobject for name in table.dtype.names):
        _refuse(f"Object/reference fields are not supported at {path}")


def _plan_columns(plan: Mapping[str, Iterable[str]]) -> dict:
    # A mapping iterates its keys, so the writer's {column: values} plan and a
    # bare ordered sequence of column names name the same columns.
    return {group: tuple(columns) for group, columns in plan.items()}


def _compare_attributes(
    path: str,
    old,
    new,
    plan: Mapping[str, tuple[str, ...]],
) -> int:
    table_group = path[: -len("/table")] if path.endswith("/table") else None
    additions = (
        _new_table_attributes(len(old.dtype.names), plan[table_group])
        if table_group in plan
        else {}
    )
    if set(new.attrs) != set(old.attrs) | set(additions):
        _refuse(f"HDF attributes changed at {path}")
    for name in old.attrs:
        if path in plan and name in _REGISTRATION:
            expected = _registration_value(name, old.attrs[name], plan[path])
            attribute = new.attrs.get_id(name)
            if (
                attribute.shape != ()
                or attribute.get_type().get_class() != h5py.h5t.STRING
                or attribute.dtype != np.dtype(f"S{len(expected)}")
                or attribute.get_type().get_cset() != h5py.h5t.CSET_ASCII
                or bytes(new.attrs[name]) != expected
            ):
                _refuse(
                    f"Unexpected pandas column registration change at {path}:{name}"
                )
            continue
        a, b = old.attrs.get_id(name), new.attrs.get_id(name)
        if (
            a.get_type() != b.get_type()
            or a.shape != b.shape
            or _attribute_bytes(a) != _attribute_bytes(b)
        ):
            _refuse(f"HDF attribute changed at {path}:{name}")
    for name, expected in additions.items():
        attribute = new.attrs.get_id(name)
        actual = new.attrs[name]
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
            _refuse(f"Invalid appended column metadata at {path}:{name}")
    return len(old.attrs)


def compare_boolean_append(
    parent_h5: str | Path,
    child_h5: str | Path,
    plan: Mapping[str, Iterable[str]],
) -> dict:
    """Prove the child differs from the parent only by the planned append.

    ``plan`` maps each HDF group (``person``, ``spm_unit``) to its appended
    column names in order: the writer's ``{column: values}`` mapping or a bare
    sequence of names. Only the names and their order are read here.

    The object inventory (paths, kinds, hard links only, no aliases) must be
    identical. For each planned ``<group>/table``: the same shape and storage,
    a record type equal to the old type widened by one HDF bitfield8 field per
    planned column in plan order, every pre-existing field's bytes exact,
    every pre-existing table attribute exact, and exactly the five expected
    attributes added per column. On each planned group, the four pandas
    column-registration attributes must equal the old value with the planned
    columns appended; every other attribute there is exact. Every other
    object and attribute is compared exactly: HDF datatype identities
    (including bitfields), storage, NaN payloads, signed zero, every index
    dataset and row order. The result holds only aggregate counts and
    digests, never a record.
    """

    columns_by_group = _plan_columns(plan)
    tables = {f"{group}/table": names for group, names in columns_by_group.items()}
    parent_sha = file_sha256(parent_h5)
    child_sha = file_sha256(child_h5)
    datasets = groups = attributes = 0
    fields_checked: dict[str, int] = {}
    added_attributes: dict[str, tuple[str, ...]] = {}
    digests = {
        column: hashlib.sha256()
        for names in columns_by_group.values()
        for column in names
    }
    rows: dict[str, int] = {}
    with h5py.File(parent_h5, "r") as parent, h5py.File(child_h5, "r") as child:
        old_objects, new_objects = _objects(parent), _objects(child)
        if old_objects.keys() != new_objects.keys():
            _refuse("HDF groups or datasets were added or removed")
        for path, names in tables.items():
            if path not in old_objects:
                _refuse(f"Parent has no {path}")
            _check_supported_table(parent[path], names, path=path)
        comparison_plan = {**columns_by_group, **tables}
        for path, old in old_objects.items():
            new = new_objects[path]
            if type(old) is not type(new):
                _refuse(f"HDF object kind changed at {path}")
            attributes += _compare_attributes(path, old, new, comparison_plan)
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
                _refuse(f"HDF shape or storage changed at {path}")
            if path in tables:
                names = tables[path]
                before, after = old.id.get_type(), new.id.get_type()
                expected = before.copy()
                expected.set_size(before.get_size() + len(names))
                for offset, column in enumerate(names):
                    expected.insert(
                        column.encode(),
                        before.get_size() + offset,
                        h5py.h5t.NATIVE_B8,
                    )
                if expected != after or new.dtype.names != (*old.dtype.names, *names):
                    _refuse(f"{path} dtype differs beyond the appended fields")
                fields_checked[path] = len(old.dtype.names)
                rows[path] = len(old)
                added_attributes[path] = tuple(
                    _new_table_attributes(len(old.dtype.names), names)
                )
            elif old.id.get_type() != new.id.get_type() or old.dtype != new.dtype:
                _refuse(f"HDF dtype changed at {path}")
            slices = (
                [()]
                if old.ndim == 0
                else [
                    slice(start, min(start + _CHUNK_ROWS, len(old)))
                    for start in range(0, len(old), _CHUNK_ROWS)
                ]
            )
            for selection in slices:
                if path in tables:
                    a = _raw_rows(old, selection.start, selection.stop)
                    b = _raw_rows(new, selection.start, selection.stop)
                    for name in old.dtype.names:
                        if a[name].tobytes() != b[name].tobytes():
                            _refuse(f"Existing {path} field changed: {name}")
                    for column in tables[path]:
                        written = b[column]
                        if not np.isin(written, [0, 1]).all():
                            _refuse(f"Appended {column} is not Boolean")
                        digests[column].update(
                            written.astype(bool, copy=False).tobytes()
                        )
                elif _value_bytes(_raw_selection(old, selection)) != _value_bytes(
                    _raw_selection(new, selection)
                ):
                    _refuse(f"Existing HDF dataset values changed: {path}")
    if file_sha256(parent_h5) != parent_sha or file_sha256(child_h5) != child_sha:
        _refuse("An HDF file changed during the exhaustive comparison")
    return {
        "schema_version": 1,
        "parent_sha256": parent_sha,
        "candidate_sha256": child_sha,
        # Every pre-existing field of the replaced tables and every other
        # dataset's bytes; the exceptions are enumerated below, not implied.
        "all_preexisting_fields_and_datasets_exact": True,
        "all_other_objects_and_attributes_exact": True,
        "groups_checked_including_root": groups,
        "datasets_checked": datasets,
        "preexisting_attributes_checked": attributes,
        "tables_extended": {
            path: {
                "rows": rows[path],
                "preexisting_fields_checked_including_index": fields_checked[path],
                "appended_fields": list(names),
                "storage": "HDF bitfield8; pandas bool",
                "record_size_increase_bytes": len(names),
                "added_table_attributes": sorted(added_attributes[path]),
            }
            for path, names in tables.items()
        },
        "rewritten_group_attributes": {
            group: list(_REGISTRATION) for group in columns_by_group
        },
        "appended_columns": {
            column: {"values_sha256": digest.hexdigest()}
            for column, digest in digests.items()
        },
    }
