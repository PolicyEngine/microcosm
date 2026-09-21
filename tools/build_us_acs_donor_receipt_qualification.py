#!/usr/bin/env python3
"""Qualify a pinned Build P donor for the ACS staging gate; never publish.

Adds exactly three reported-receipt input columns — ``person.receives_wic``,
``spm_unit.receives_snap`` and ``spm_unit.receives_tanf`` — derived only
through the maintained producers in
:mod:`microcosm.build.us_runtime.cps_carried` and the pinned ``PAW_TYP``
restore in :mod:`microcosm.build.us_runtime.public_assistance_type_source`.

Every other cell, weight, index, attribute and entity table is preserved
byte-for-byte and proven so. The result is a local H5 plus an aggregate
receipt: never a release, a staged bundle, a calibration or a latest pointer.

See ``docs/us-acs-donor-receipt-qualification.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from microcosm.build.us_runtime import (
    cps_carried,
    education_assistance_source,
    h5_io,
    public_assistance_type_source,
    support_provenance,
)
from microcosm.build.us_runtime.acs_transfer import (
    ACS_DONOR_CHANNEL_AUTO,
    resolve_acs_donor_channel,
)
from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
)
from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5
from microcosm.data import h5_enrichment

# Byte-level HDF primitives from the reviewed enrichment writer. They are
# generic (object walk, raw typed reads, attribute bytes); only that module's
# role-pinned wrappers are unusable here, so the primitives are reused rather
# than re-derived. Their file digest is recorded in the receipt.
from microcosm.data.h5_enrichment import (  # noqa: PLC2701
    _attribute_bytes,
    _copy_attribute,
    _objects,
    _raw_rows,
    _raw_selection,
    _value_bytes,
    file_sha256,
)
from microcosm.frame import Frame

#: The exact reviewed Build P parent. Equal to
#: ``microcosm.data.source_enrichment.PARENT_DATASET_SHA256``; declared here as
#: this tool's own pin so the tool does not inherit the release contract.
DONOR_SHA256 = "48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e"

#: Output column -> owning entity. Ordered as written.
QUALIFIED_COLUMNS: dict[str, str] = {
    "receives_wic": "person",
    "receives_snap": "spm_unit",
    "receives_tanf": "spm_unit",
}

#: Raw person columns the producers consume. ``_source`` coerces an absent
#: column to zeros and ``_fill_spm_unit_reported_enrollment_inputs`` substitutes
#: ``SPM_SNAPSUB = 0.0``, so absence must be refused before they run.
REQUIRED_PERSON_COLUMNS: tuple[str, ...] = (
    cps_carried.CPS_REPORTED_SNAP_RAW_COLUMN,
    cps_carried.CPS_REPORTED_TANF_AMOUNT_RAW_COLUMN,
    cps_carried.CPS_REPORTED_WIC_RAW_COLUMN,
    "PERIDNUM",
    "person_spm_unit_id",
    "source_year",
)
REQUIRED_SPM_UNIT_COLUMNS: tuple[str, ...] = ("spm_unit_id",)

RECEIPT_FILENAME = "donor_receipt_qualification.json"

_PRODUCER_FILES = (
    "tools/build_us_acs_donor_receipt_qualification.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/public_assistance_type_source.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/education_assistance_source.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/support_provenance.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py",
    "packages/microcosm-data/src/microcosm/data/h5_enrichment.py",
)

_CHUNK_ROWS = 4096
_REGISTRATION = ("data_columns", "info", "non_index_axes", "values_cols")


class DonorQualificationError(ValueError):
    """A refusal. Its message never contains a row value or identifier."""


def _refuse(message: str) -> None:
    raise DonorQualificationError(message)


def _redacted(producer: str, call, *args, **kwargs):
    """Call a maintained producer, re-raising its refusal without its message.

    Producer refusals in this area embed raw ``PERIDNUM`` values and row
    indices. Only the producer's name and the exception class cross this
    boundary; ``from None`` suppresses the original from any traceback.
    """

    try:
        return call(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberate redaction boundary
        raise DonorQualificationError(
            f"{producer} refused ({type(exc).__name__}); its message is "
            "withheld because producer refusals in this area name source "
            "identifiers"
        ) from None


# --------------------------------------------------------------------------
# Producer identity
# --------------------------------------------------------------------------


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _producer_identity() -> dict:
    """Bind the receipt to the exact modules this process actually imported."""

    root = _repository_root()
    loaded = dict(
        zip(
            _PRODUCER_FILES,
            (
                __file__,
                cps_carried.__file__,
                public_assistance_type_source.__file__,
                education_assistance_source.__file__,
                support_provenance.__file__,
                h5_io.__file__,
                h5_enrichment.__file__,
            ),
            strict=True,
        )
    )
    for filename, actual in loaded.items():
        if actual is None or Path(actual).resolve() != (root / filename).resolve():
            _refuse(
                f"Producer must execute this checkout's {filename}; "
                "install this workspace's local shards first"
            )

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {
        "repository": "https://github.com/PolicyEngine/microcosm",
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "runtime": {
            "python": platform.python_version(),
            "hdf5": h5py.version.hdf5_version,
            **{
                package: metadata.version(package)
                for package in ("numpy", "pandas", "h5py", "tables", "microcosm-build")
            },
        },
        "source_files_sha256": {
            filename: file_sha256(root / filename) for filename in _PRODUCER_FILES
        },
    }


# --------------------------------------------------------------------------
# Derivation through the maintained producers
# --------------------------------------------------------------------------


def _pooled_income_years(person: pd.DataFrame) -> tuple[int, ...]:
    years = pd.to_numeric(person["source_year"], errors="coerce")
    if years.isna().any() or not np.equal(years, np.floor(years)).all():
        _refuse("Donor person source_year must be finite integers")
    return tuple(sorted({int(year) for year in years.unique()}))


def derive_receipt_columns(
    frame: Frame,
    public_assistance_type_source_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return the three columns, derived only through maintained producers."""

    person = frame.table("person")
    spm_unit = frame.table("spm_unit")

    wic = _redacted(
        "cps_carried.reported_wic_receipt_carrier",
        cps_carried.reported_wic_receipt_carrier,
        person,
    )
    wic = np.asarray(wic)
    if wic.dtype != np.dtype(bool) or wic.shape != (len(person),):
        _refuse("reported_wic_receipt_carrier did not return a person-grain mask")

    # The fill reads only ``spm_unit.columns`` and ``spm_unit['spm_unit_id']``
    # (cps_carried.py:493-510), so a single-column projection is the same input
    # and leaves the donor's own SPM table unmutated.
    projection = spm_unit[["spm_unit_id"]].copy()
    _redacted(
        "cps_carried._fill_spm_unit_reported_enrollment_inputs",
        cps_carried._fill_spm_unit_reported_enrollment_inputs,  # noqa: SLF001
        person,
        projection,
        public_assistance_type_source=public_assistance_type_source_frame,
    )
    for column in ("receives_tanf", "receives_snap"):
        if column not in projection.columns:
            _refuse(f"The maintained SPM fill did not emit {column!r}")

    derived = {
        "receives_wic": wic,
        "receives_snap": projection["receives_snap"].to_numpy(dtype=bool),
        "receives_tanf": projection["receives_tanf"].to_numpy(dtype=bool),
    }
    for column, values in derived.items():
        entity = QUALIFIED_COLUMNS[column]
        if values.shape != (frame.n(entity),):
            _refuse(f"Derived {column!r} does not cover the {entity} rows")
    return derived


# --------------------------------------------------------------------------
# Aggregate counts for the receipt
# --------------------------------------------------------------------------


def _labelled_counts(values: np.ndarray, labels: pd.Series | None) -> dict:
    if labels is None:
        return {}
    frame = pd.DataFrame({"label": labels.to_numpy(), "value": values})
    counts: dict[str, dict[str, int]] = {}
    for label, group in frame.groupby("label", sort=True, dropna=False):
        true = int(np.count_nonzero(group["value"].to_numpy()))
        counts[str(label)] = {"true": true, "false": int(len(group) - true)}
    return counts


def _spm_unit_source_years(frame: Frame) -> pd.Series:
    """Label each SPM unit with its members' income year, or ``mixed``."""

    person = frame.table("person")
    spm_unit = frame.table("spm_unit")
    grouped = person.groupby("person_spm_unit_id", sort=True)["source_year"]
    distinct = grouped.nunique()
    first = grouped.first()
    labels = first.astype("Int64").astype(object)
    labels = labels.mask(distinct.gt(1), "mixed")
    aligned = labels.reindex(spm_unit["spm_unit_id"].to_numpy())
    return pd.Series(
        [("unmatched" if pd.isna(value) else str(value)) for value in aligned],
        index=spm_unit.index,
        name="source_year",
    )


def _support_roles(table: pd.DataFrame, entity: str) -> pd.Series | None:
    if not support_provenance.has_support_role_metadata(table, entity=entity):
        return None
    return support_provenance.support_role_series(table, entity=entity)


def qualified_frame(frame: Frame, derived: Mapping[str, np.ndarray]) -> Frame:
    """Return a copy of ``frame`` carrying the three derived columns."""

    tables = {entity: frame.table(entity) for entity in frame.entities}
    for entity in {"person", "spm_unit"}:
        table = tables[entity].copy()
        for column, owner in QUALIFIED_COLUMNS.items():
            if owner == entity:
                table[column] = derived[column]
        tables[entity] = table
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def receipt_counts(frame: Frame, derived: Mapping[str, np.ndarray]) -> dict:
    """Aggregate true/false counts by income year, support role, and channel.

    The channel arm is the one that predicts the staging gate:
    ``_require_dense_donor_coverage`` resolves the donor channel before any
    column check, so a column that is populated across the whole donor and
    constant ``False`` inside ``puf_tax_detail`` still fails the gate.
    """

    # Not redacted: this resolver's refusals name support-role labels and
    # entity names, never a source identifier, and a silent channel failure
    # would be undiagnosable.
    selected, role = resolve_acs_donor_channel(
        qualified_frame(frame, derived), ACS_DONOR_CHANNEL_AUTO
    )
    unit_years = _spm_unit_source_years(frame)
    person_years = frame.table("person")["source_year"].astype(str)
    year_labels = {"person": person_years, "spm_unit": unit_years}
    role_labels = {
        entity: _support_roles(frame.table(entity), entity)
        for entity in ("person", "spm_unit")
    }

    counts: dict[str, dict] = {}
    for column, entity in QUALIFIED_COLUMNS.items():
        values = derived[column]
        true = int(np.count_nonzero(values))
        channel_values = selected.table(entity)[column].to_numpy(dtype=bool)
        channel_true = int(np.count_nonzero(channel_values))
        counts[column] = {
            "entity": entity,
            "rows": int(values.size),
            "true": true,
            "false": int(values.size - true),
            "by_source_year": _labelled_counts(values, year_labels[entity]),
            "by_support_role": _labelled_counts(values, role_labels[entity]),
            "gate_selected_channel": {
                "role": role,
                "rows": int(channel_values.size),
                "true": channel_true,
                "false": int(channel_values.size - channel_true),
            },
        }
    return counts


# --------------------------------------------------------------------------
# HDF append, preserving every existing byte
# --------------------------------------------------------------------------


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise DonorQualificationError(
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


def _plan_columns(plan: Mapping[str, Mapping[str, np.ndarray]]) -> dict:
    return {group: tuple(columns) for group, columns in plan.items()}


def append_boolean_fields(
    parent_h5: str | Path,
    child_h5: str | Path,
    plan: Mapping[str, Mapping[str, np.ndarray]],
    *,
    expected_parent_sha256: str,
) -> dict:
    """Create a new file, append the planned Boolean fields, and verify it.

    ``plan`` maps an HDF group (``person``, ``spm_unit``) to the ordered
    Boolean columns appended to that group's ``table`` dataset. The
    destination must not exist; the parent is opened read-only and never
    modified. No HDF repacking or pandas table rewrite occurs.
    """

    parent_h5, child_h5 = Path(parent_h5), Path(child_h5)
    if file_sha256(parent_h5) != expected_parent_sha256:
        _refuse("Parent H5 SHA-256 mismatch")
    for group, columns in plan.items():
        if not columns:
            _refuse(f"Empty append plan for {group}")
        for column, values in columns.items():
            array = np.asarray(values)
            if array.dtype != np.dtype(bool) or array.ndim != 1:
                _refuse(f"{column} must be a one-dimensional Boolean array")
    with h5py.File(parent_h5, "r") as parent:
        for group, columns in plan.items():
            path = f"{group}/table"
            if path not in parent:
                _refuse(f"Parent has no {path}")
            table = parent[path]
            _check_supported_table(table, tuple(columns), path=path)
            for column, values in columns.items():
                if np.asarray(values).shape != table.shape:
                    _refuse(f"{column} coverage does not match the {group} rows")

    # Exclusive creation also refuses aliases and symlinks to the parent.
    descriptor = os.open(child_h5, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as destination, parent_h5.open("rb") as source:
            shutil.copyfileobj(source, destination, 1024 * 1024)
        if file_sha256(child_h5) != expected_parent_sha256:
            _refuse("Parent changed during the verified copy")
        with h5py.File(child_h5, "r+") as child:
            for group_name, columns in plan.items():
                _append_group_fields(child, group_name, columns)
        report = compare_boolean_append(parent_h5, child_h5, plan)
        if report["parent_sha256"] != expected_parent_sha256:
            _refuse("Parent changed during the append")
        for columns in plan.values():
            for column, values in columns.items():
                digest = hashlib.sha256(np.asarray(values).tobytes()).hexdigest()
                if report["appended_columns"][column]["values_sha256"] != digest:
                    _refuse(f"Written {column} differs from the derivation")
        return report
    except BaseException:
        child_h5.unlink(missing_ok=True)
        raise


def _append_group_fields(
    child: h5py.File,
    group_name: str,
    columns: Mapping[str, np.ndarray],
) -> None:
    names = tuple(columns)
    group = child[group_name]
    old = child[f"{group_name}/table"]
    compound = old.id.get_type().copy()
    old_size = compound.get_size()
    compound.set_size(old_size + len(names))
    for offset, column in enumerate(names):
        # PyTables Boolean columns are HDF bitfields, not enums or integers.
        compound.insert(column.encode(), old_size + offset, h5py.h5t.NATIVE_B8)
    creation = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    creation.set_chunk(old.chunks)
    creation.set_obj_track_times(False)
    temporary = "_receipt_qualified_table"
    if temporary in group:
        _refuse(f"Unexpected temporary HDF object in {group_name}")
    new = h5py.Dataset(
        h5py.h5d.create(
            group.id,
            temporary.encode(),
            compound,
            old.id.get_space(),
            dcpl=creation,
        )
    )
    if new.dtype.itemsize != old_size + len(names) or old.dtype.itemsize != old_size:
        _refuse(f"Unexpected HDF record padding at {group_name}/table")
    arrays = {column: np.asarray(values) for column, values in columns.items()}
    for start in range(0, len(old), _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, len(old))
        original = _raw_rows(old, start, stop)
        extended = np.empty(stop - start, dtype=new.dtype)
        extended.view(np.uint8).reshape(-1, old_size + len(names))[:, :old_size] = (
            original.view(np.uint8).reshape(-1, old_size)
        )
        for column in names:
            extended[column] = arrays[column][start:stop]
        memory = h5py.h5s.create_simple((stop - start,))
        selection = new.id.get_space()
        selection.select_hyperslab((start,), (stop - start,))
        new.id.write(memory, selection, extended, mtype=compound)
    for name in old.attrs:
        _copy_attribute(old, new, name)
    for name, value in _new_table_attributes(len(old.dtype.names), names).items():
        if isinstance(value, str):
            new.attrs.create(name, value, dtype=h5py.string_dtype("utf-8", len(value)))
        else:
            new.attrs[name] = value
    for name in _REGISTRATION:
        group.attrs[name] = np.bytes_(
            _registration_value(name, group.attrs[name], names)
        )
    del child[f"{group_name}/table"]
    group.move(temporary, "table")


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
    plan: Mapping[str, Mapping[str, np.ndarray]],
) -> dict:
    """Prove exact identity of every pre-existing object, byte for byte.

    Compares HDF datatype identities (including bitfields), NaN payloads,
    signed zero, every index dataset, every attribute and row order. The
    result holds only aggregate counts and digests, never a record.
    """

    columns_by_group = _plan_columns(plan)
    tables = {f"{group}/table": names for group, names in columns_by_group.items()}
    parent_sha = file_sha256(parent_h5)
    child_sha = file_sha256(child_h5)
    datasets = groups = attributes = 0
    fields_checked: dict[str, int] = {}
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
        "all_preexisting_objects_exact": True,
        "groups_checked_including_root": groups,
        "datasets_checked": datasets,
        "preexisting_attributes_checked": attributes,
        "tables_extended": {
            path: {
                "rows": rows[path],
                "preexisting_fields_checked_including_index": fields_checked[path],
                "appended_fields": list(names),
            }
            for path, names in tables.items()
        },
        "appended_columns": {
            column: {"values_sha256": digest.hexdigest()}
            for column, digest in digests.items()
        },
        "storage": "HDF bitfield8; pandas bool",
        "column_registration_attributes": list(_REGISTRATION),
    }


# --------------------------------------------------------------------------
# Reload verification
# --------------------------------------------------------------------------


def _assert_column_identical(
    entity: str, column: str, before: pd.Series, after: pd.Series
) -> None:
    if before.dtype != after.dtype:
        _refuse(f"{entity}.{column} dtype changed on reload")
    if not before.equals(after):
        _refuse(f"{entity}.{column} values changed on reload")


def verify_reload(
    parent_frame: Frame,
    child_h5: Path,
    derived: Mapping[str, np.ndarray],
) -> dict:
    """Reload the child through the maintained loader and compare everything."""

    child_frame = load_legacy_calibrated_us_h5(child_h5)
    if child_frame.entities != parent_frame.entities:
        _refuse("Child entity set changed")
    for entity in parent_frame.entities:
        before = parent_frame.table(entity)
        after = child_frame.table(entity)
        appended = [
            column for column, owner in QUALIFIED_COLUMNS.items() if owner == entity
        ]
        if list(after.columns) != [*before.columns, *appended]:
            _refuse(f"{entity} column order changed")
        if len(after) != len(before) or not after.index.equals(before.index):
            _refuse(f"{entity} rows or index changed")
        for column in before.columns:
            _assert_column_identical(entity, column, before[column], after[column])
        for column in appended:
            values = after[column]
            if values.dtype != np.dtype(bool):
                _refuse(f"{entity}.{column} did not reload as bool")
            if not np.array_equal(values.to_numpy(dtype=bool), derived[column]):
                _refuse(f"{entity}.{column} reloaded with different values")
    for entity in parent_frame.weighted_entities:
        before = parent_frame.weights_for(entity)
        after = child_frame.weights_for(entity)
        if before.kind != after.kind or not np.array_equal(
            np.asarray(before.values), np.asarray(after.values)
        ):
            _refuse(f"{entity} weights changed")
    return {
        "entities_compared": list(parent_frame.entities),
        "weighted_entities_compared": list(parent_frame.weighted_entities),
        "columns_compared": {
            entity: int(len(parent_frame.table(entity).columns))
            for entity in parent_frame.entities
        },
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _json_write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


def _archive_pin_summary(income_years: Iterable[int]) -> dict:
    summary = {}
    for year in income_years:
        archive = ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year]
        audit = public_assistance_type_source.ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS[
            year
        ]
        summary[str(year)] = {
            "survey_year": archive.survey_year,
            "member": archive.member,
            "member_size_bytes": archive.member_size_bytes,
            "member_sha256": archive.member_sha256,
            "zip_sha256": archive.zip_sha256,
            "pinned_rows": audit.rows,
            "pinned_paw_type_counts": list(audit.paw_type_counts),
            "pinned_paw_positive_rows": audit.paw_positive_rows,
            "pinned_paw_positive_tanf_rows": audit.paw_positive_tanf_rows,
        }
    return summary


def _refuse_preexisting_columns(frame: Frame) -> None:
    for entity in frame.entities:
        present = [
            column for column in QUALIFIED_COLUMNS if column in frame.table(entity)
        ]
        if present:
            _refuse(
                f"Donor already carries qualified column(s) {sorted(present)} on "
                f"{entity}; refusing to overwrite measured data"
            )
    if cps_carried.CPS_REPORTED_TANF_TYPE_RAW_COLUMN in frame.table("person"):
        _refuse(
            "Donor already carries PAW_TYP on person; the pinned-archive restore "
            "this receipt claims would be silently bypassed"
        )


def _refuse_missing_raw_columns(frame: Frame) -> None:
    person = frame.table("person")
    missing = [
        column for column in REQUIRED_PERSON_COLUMNS if column not in person.columns
    ]
    if missing:
        _refuse(f"Donor person table is missing required raw column(s): {missing}")
    spm_unit = frame.table("spm_unit")
    missing = [
        column for column in REQUIRED_SPM_UNIT_COLUMNS if column not in spm_unit.columns
    ]
    if missing:
        _refuse(f"Donor spm_unit table is missing required column(s): {missing}")


def _refuse_signalless_columns(counts: Mapping[str, Mapping]) -> None:
    for column, summary in counts.items():
        if summary["true"] == 0:
            _refuse(
                f"{summary['entity']}.{column} has no true value across the donor; "
                "the staging gate's default-valued check would refuse it"
            )
        channel = summary["gate_selected_channel"]
        if channel["true"] == 0:
            _refuse(
                f"{summary['entity']}.{column} has no true value inside the "
                f"gate-selected channel {channel['role']!r}; the staging gate "
                "resolves that channel before every column check"
            )


def qualify_donor(
    *,
    parent_h5: Path,
    output_dir: Path,
    source_paths: Mapping[int, Path] | None = None,
    source_cache: Path | None = None,
) -> dict:
    """Derive, preserve, verify, then atomically expose a local qualified donor."""

    parent_h5 = Path(parent_h5)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            "Output directory already exists; choose a new candidate directory"
        )
    if file_sha256(parent_h5) != DONOR_SHA256:
        _refuse("Only the exact pinned Build P donor can be qualified")
    producer_identity = _producer_identity()

    frame = load_legacy_calibrated_us_h5(parent_h5)
    _refuse_preexisting_columns(frame)
    _refuse_missing_raw_columns(frame)
    income_years = _pooled_income_years(frame.table("person"))
    unpinned = [
        year for year in income_years if year not in ASEC_EDUCATION_ASSISTANCE_ARCHIVES
    ]
    if unpinned:
        _refuse(f"No pinned ASEC archive covers donor income year(s): {unpinned}")
    if source_paths is None:
        cache = Path(
            source_cache
            if source_cache is not None
            else Path.home() / ".cache/microcosm/cps/asec_education"
        ).expanduser()
        source_paths = {
            year: cache / ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year].member
            for year in income_years
        }
    missing_sources = sorted(
        str(year) for year, path in source_paths.items() if not Path(path).is_file()
    )
    if missing_sources:
        _refuse(
            "Pinned ASEC person member(s) are not present locally for income "
            f"year(s): {missing_sources}; this tool never downloads"
        )

    source = _redacted(
        "public_assistance_type_source.load_asec_public_assistance_type_sources",
        public_assistance_type_source.load_asec_public_assistance_type_sources,
        {year: Path(path) for year, path in source_paths.items()},
        income_years=income_years,
    )
    derived = derive_receipt_columns(frame, source)
    counts = receipt_counts(frame, derived)
    _refuse_signalless_columns(counts)

    plan = {
        "person": {"receives_wic": derived["receives_wic"]},
        "spm_unit": {
            "receives_snap": derived["receives_snap"],
            "receives_tanf": derived["receives_tanf"],
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".donor-receipt-qualification-", dir=output_dir.parent)
    )
    try:
        os.chmod(staging, 0o700)
        child_h5 = staging / f"{parent_h5.stem}_receipt_qualified.h5"
        preservation = append_boolean_fields(
            parent_h5,
            child_h5,
            plan,
            expected_parent_sha256=DONOR_SHA256,
        )
        reload_report = verify_reload(frame, child_h5, derived)
        if (
            _producer_identity()["source_files_sha256"]
            != producer_identity["source_files_sha256"]
        ):
            _refuse("Producer source files changed while qualifying the donor")
        receipt = {
            "schema_version": 1,
            "operation": "add_reported_receipt_inputs",
            "certification": (
                "none; local build evidence. Not a release, staged bundle, "
                "calibration or latest pointer."
            ),
            "parent": {"filename": parent_h5.name, "sha256": DONOR_SHA256},
            "dataset": {
                "filename": child_h5.name,
                "sha256": preservation["candidate_sha256"],
            },
            "added_columns": [
                {"name": column, "entity": entity, "dtype": "bool"}
                for column, entity in QUALIFIED_COLUMNS.items()
            ],
            "pooled_income_years": list(income_years),
            "archive_pins": _archive_pin_summary(income_years),
            "source_audit": {
                str(year): audit
                for year, audit in source.attrs.get("source_audit", {}).items()
            },
            "counts": counts,
            "preservation": preservation,
            "reload_verification": reload_report,
            "code": producer_identity,
        }
        _json_write(staging / RECEIPT_FILENAME, receipt)
        child_h5.chmod(0o400)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                "Output appeared during the build; refusing to overwrite"
            )
        os.rename(staging, output_dir)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path.home() / ".cache/microcosm/cps/asec_education",
        help="Directory holding the pinned complete pppub23/24/25.csv members",
    )
    args = parser.parse_args(argv)
    try:
        receipt = qualify_donor(
            parent_h5=args.parent_h5,
            output_dir=args.output_dir,
            source_cache=args.source_cache,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Donor receipt qualification refused: {exc}\n")
    print(
        json.dumps(
            {
                "dataset": receipt["dataset"],
                "counts": receipt["counts"],
                "certification": receipt["certification"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
