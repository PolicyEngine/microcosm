"""Exercise donor receipt qualification on invented fixtures only.

Every fixture here is synthetic. No pinned Census archive, no real donor and
no licensed microdata is read, and the pins the maintained loader verifies are
measured from the synthetic CSV this module writes.
"""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import public_assistance_type_source as pats
from microcosm.build.us_runtime.education_assistance_source import AsecEducationArchive
from microcosm.data.h5_enrichment import file_sha256
from test_support.paths import paths_for
from tools import build_us_acs_donor_receipt_qualification as builder

_TEST_PATHS = paths_for("microcosm-build")

pytest.importorskip("tables")

_INCOME_YEARS = (2022, 2023)

#: person index -> (source_year, spm unit, clone index, WICYN, SPM_SNAPSUB,
#: PAW_VAL, PAW_TYP)
_PEOPLE = (
    (2022, 1, 0, 1, 0, 0, 0),
    (2022, 1, 0, 0, 0, 1200, 1),
    (2023, 2, 0, 2, 900, 0, 0),
    (2023, 2, 0, 0, 900, 500, 2),
    (2022, 3, 1, 1, 300, 0, 0),
    (2022, 3, 1, 0, 300, 800, 3),
    (2023, 4, 1, 0, 0, 0, 0),
    (2023, 4, 1, 2, 0, 400, 0),
)

_EXPECTED = {
    "receives_wic": [True, False, False, False, True, False, False, False],
    "receives_snap": [False, True, True, False],
    "receives_tanf": [True, False, True, False],
}


def _peridnum(index: int) -> str:
    return f"{9000000000000000000000 + index * 7:022d}"[-22:]


def _person_table(*, overrides: dict | None = None) -> pd.DataFrame:
    rows = []
    for index, (year, spm, clone, wic, snap, paw, _type) in enumerate(_PEOPLE):
        household = spm
        rows.append(
            {
                "PERIDNUM": _peridnum(index),
                "PH_SEQ": 500 + household,
                "P_SEQ": 1,
                "A_LINENO": index + 1,
                "source_year": year,
                "source_household_id": 40 + household,
                "person_id": index + 1,
                "person_household_id": household,
                "person_tax_unit_id": household,
                "person_spm_unit_id": spm,
                "person_family_id": household,
                "person_marital_unit_id": household,
                "person_support_clone_index": clone,
                "person_support_channel": "asec" if clone == 0 else "puf_tax_detail",
                "age": 30 + index,
                "is_female": index % 2 == 0,
                "WICYN": wic,
                "SPM_SNAPSUB": snap,
                "PAW_VAL": paw,
            }
        )
    table = pd.DataFrame(rows)
    for column, values in (overrides or {}).items():
        if values is None:
            table = table.drop(columns=[column])
        else:
            table[column] = values
    return table


def _write_parent(path: Path, *, person_overrides: dict | None = None) -> None:
    person = _person_table(overrides=person_overrides)
    households = sorted({row[1] for row in _PEOPLE})
    with pd.HDFStore(path, "w") as store:
        store.put("person", person, format="table", data_columns=True)
        store.put(
            "household",
            pd.DataFrame(
                {
                    "household_id": households,
                    "household_weight": [100.0 * unit for unit in households],
                    "state_fips": [6, 6, 36, 36],
                }
            ),
            format="table",
            data_columns=True,
        )
        store.put(
            "spm_unit",
            pd.DataFrame(
                {
                    "spm_unit_id": households,
                    "spm_unit_support_clone_index": [0, 0, 1, 1],
                    "spm_unit_support_channel": [
                        "asec",
                        "asec",
                        "puf_tax_detail",
                        "puf_tax_detail",
                    ],
                    "takes_up_snap_if_eligible": [True, False, True, False],
                }
            ),
            format="table",
            data_columns=True,
        )
        for entity in ("tax_unit", "family", "marital_unit"):
            store.put(
                entity,
                pd.DataFrame({f"{entity}_id": households}),
                format="table",
                data_columns=True,
            )
        store.put(
            "_time_period",
            pd.DataFrame({"time_period": [2024]}),
            format="table",
            data_columns=True,
        )


def _write_sidecar(path: Path, year: int, *, drop_person: int | None = None) -> None:
    rows = []
    for index, (row_year, _spm, _clone, _wic, _snap, paw, paw_type) in enumerate(
        _PEOPLE
    ):
        if row_year != year or index == drop_person:
            continue
        rows.append(
            {
                "PH_SEQ": 40 + _PEOPLE[index][1],
                "P_SEQ": 1,
                "A_LINENO": index + 1,
                "PERIDNUM": _peridnum(index),
                "PAW_VAL": paw,
                "PAW_TYP": paw_type,
                "A_FNLWGT": 1000.0 + index,
            }
        )
    # One extra in-universe row per year keeps the sidecar a superset of the
    # frame, as the real archives are.
    rows.append(
        {
            "PH_SEQ": 999,
            "P_SEQ": 1,
            "A_LINENO": 9,
            "PERIDNUM": _peridnum(90 + year % 100),
            "PAW_VAL": 0,
            "PAW_TYP": 0,
            "A_FNLWGT": 1.0,
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _measured_pins(path: Path, year: int) -> tuple[AsecEducationArchive, object]:
    frame = pd.read_csv(path, dtype={"PERIDNUM": "string"})
    types = frame["PAW_TYP"].to_numpy()
    amounts = frame["PAW_VAL"].to_numpy(dtype=float)
    positive = amounts > 0.0
    archive = AsecEducationArchive(
        survey_year=year + 1,
        income_year=year,
        zip_url=f"https://example.invalid/synthetic-{year}.zip",
        zip_size_bytes=1,
        zip_sha256="0" * 64,
        member=path.name,
        member_size_bytes=path.stat().st_size,
        member_crc32="00000000",
        member_sha256=file_sha256(path),
        rows=int(len(frame)),
        positive_rows=int(np.count_nonzero(positive)),
        weighted_positive_share=0.0,
        weighted_total=0.0,
    )
    audit = pats.AsecPublicAssistanceTypeAudit(
        income_year=year,
        rows=int(len(frame)),
        paw_type_counts=tuple(
            int(np.count_nonzero(types == code)) for code in range(4)
        ),
        paw_positive_rows=int(np.count_nonzero(positive)),
        paw_positive_tanf_rows=int(np.count_nonzero(positive & np.isin(types, [1, 3]))),
    )
    return archive, audit


_STUB_IDENTITY = {
    "repository": "https://github.com/PolicyEngine/microcosm",
    "git_commit": "c" * 40,
    "git_dirty": True,
    "runtime": {"python": "3.14.0"},
    "source_files_sha256": {"synthetic-producer.py": "d" * 64},
}


@pytest.fixture
def donor(tmp_path, monkeypatch):
    """A synthetic pinned donor plus its synthetic, measured-pin sidecars."""

    def build(*, person_overrides=None, drop_person=None):
        parent = tmp_path / "synthetic_donor.h5"
        parent.unlink(missing_ok=True)
        _write_parent(parent, person_overrides=person_overrides)
        archives, audits, paths = {}, {}, {}
        for year in _INCOME_YEARS:
            path = tmp_path / f"synthetic_person_{year}.csv"
            _write_sidecar(path, year, drop_person=drop_person)
            archive, audit = _measured_pins(path, year)
            archives[year], audits[year], paths[year] = archive, audit, path
        monkeypatch.setattr(pats, "ASEC_EDUCATION_ASSISTANCE_ARCHIVES", archives)
        monkeypatch.setattr(pats, "ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS", audits)
        monkeypatch.setattr(builder, "ASEC_EDUCATION_ASSISTANCE_ARCHIVES", archives)
        monkeypatch.setattr(builder, "DONOR_SHA256", file_sha256(parent))
        monkeypatch.setattr(builder, "_producer_identity", lambda: _STUB_IDENTITY)
        return SimpleNamespace(
            parent=parent,
            source_paths=paths,
            output=tmp_path / "qualified",
        )

    return build


def _qualify(fixture, **overrides):
    kwargs = {
        "parent_h5": fixture.parent,
        "output_dir": fixture.output,
        "source_paths": fixture.source_paths,
    }
    kwargs.update(overrides)
    return builder.qualify_donor(**kwargs)


def _leftover_staging(fixture) -> list[str]:
    return sorted(
        path.name
        for path in fixture.output.parent.glob(".donor-receipt-qualification-*")
    )


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Preservation
# --------------------------------------------------------------------------


def _exact_bytes(value: object) -> bytes:
    # A PyTables zero-length TITLE is an h5py null dataspace; np.asarray on it
    # yields an object array whose tobytes() is a pointer address, so it would
    # differ between two byte-identical files.
    if isinstance(value, h5py.Empty):
        return b""
    array = np.asarray(value)
    assert not array.dtype.hasobject, f"unexpected object HDF value: {value!r}"
    return array.tobytes()


def _objects_with_bytes(path: Path) -> dict:
    captured: dict[str, dict] = {}

    def descend(group, prefix=""):
        for name in group:
            child_path = f"{prefix}/{name}".lstrip("/")
            child = group[name]
            entry = {
                "attrs": {
                    key: _exact_bytes(child.attrs[key]) for key in sorted(child.attrs)
                },
                "kind": type(child).__name__,
            }
            if isinstance(child, h5py.Group):
                descend(child, child_path)
            else:
                entry["dtype"] = str(child.dtype)
                entry["values"] = _exact_bytes(child[()])
            captured[child_path] = entry

    with h5py.File(path, "r") as handle:
        descend(handle)
    return captured


# --------------------------------------------------------------------------
# The staging gate
# --------------------------------------------------------------------------


def _load_legacy_builder():
    root = _TEST_PATHS.repository
    path = root / "tools" / "_legacy" / "build_us_acs_multispine_base.py"
    spec = importlib.util.spec_from_file_location(
        "legacy_build_us_acs_multispine_base_for_receipt_qualification",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _ReceiptDefaultsEngine:
    """Engine defaults for exactly the columns this gate check inspects."""

    _defaults = {
        "age": 0,
        "is_female": False,
        "state_fips": 0,
        "receives_snap": False,
        "receives_tanf": False,
        "receives_wic": False,
    }

    def default_values(self, names):
        return {name: self._defaults[name] for name in names if name in self._defaults}


_RECEIPT_TARGET_FAMILIES = {
    "person": {"model_required_boolean": ("receives_wic",)},
    "spm_unit": {"model_required_boolean": ("receives_snap", "receives_tanf")},
}


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def _foreign_resolver(donor, channel):
    return donor, channel


class _ForeignFrame:
    def select(self, person_mask):
        return self


class _ForeignSchema:
    pass


# --------------------------------------------------------------------------
# The byte-preservation verifier refuses every unplanned change
# --------------------------------------------------------------------------
#
# Each test below appends the planned columns to the synthetic parent through
# the real writer, then tampers with the child (or the plan) and requires
# ``compare_boolean_append`` or ``append_boolean_fields`` to refuse. The
# untampered case passes first, so every refusal is caused by its tamper.

_PLAN_ORDER = {
    "person": ("receives_wic",),
    "spm_unit": ("receives_snap", "receives_tanf"),
}


def _plan(order: dict | None = None) -> dict:
    return {
        group: {column: np.asarray(_EXPECTED[column], dtype=bool) for column in names}
        for group, names in (order or _PLAN_ORDER).items()
    }


@pytest.fixture
def appended(tmp_path):
    parent = tmp_path / "parent.h5"
    _write_parent(parent)
    child = tmp_path / "child.h5"
    plan = _plan()
    report = builder.append_boolean_fields(
        parent, child, plan, expected_parent_sha256=file_sha256(parent)
    )
    return SimpleNamespace(parent=parent, child=child, plan=plan, report=report)


def _verify(fixture, plan: dict | None = None) -> dict:
    return builder.compare_boolean_append(
        fixture.parent, fixture.child, fixture.plan if plan is None else plan
    )


def _write_records(dataset, values, start: int = 0) -> None:
    """Write raw records with the file type, so only the intended bytes move."""

    values = np.ascontiguousarray(values)
    selection = dataset.id.get_space()
    selection.select_hyperslab((start,), (values.size,))
    dataset.id.write(
        h5py.h5s.create_simple(values.shape),
        selection,
        values,
        mtype=dataset.id.get_type(),
    )


def _set_field_byte(dataset, field: str, *, row: int = 0, value=None) -> None:
    """Flip the low bit of ``field``'s first byte in one row, or set it."""

    record = builder._raw_rows(dataset, row, row + 1)
    offset = record.dtype.fields[field][1]
    raw = record.view(np.uint8)
    raw[offset] = raw[offset] ^ 1 if value is None else value
    _write_records(dataset, record, start=row)


def _recreate_dataset(
    group,
    name: str,
    *,
    datatype=None,
    rows: int | None = None,
    chunks: tuple[int, ...] | None = None,
) -> None:
    """Replace one dataset, keeping its attributes, with one property changed."""

    old = group[name]
    datatype = old.id.get_type() if datatype is None else datatype
    rows = len(old) if rows is None else rows
    creation = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    creation.set_chunk(chunks or old.chunks)
    new = h5py.Dataset(
        h5py.h5d.create(
            group.id,
            b"_tampered",
            datatype,
            h5py.h5s.create_simple((rows,), (h5py.h5s.UNLIMITED,)),
            dcpl=creation,
        )
    )
    source = builder._raw_rows(old, 0, rows)
    if datatype == old.id.get_type():
        _write_records(new, source)
    for key in old.attrs:
        builder._copy_attribute(old, new, key)
    del group[name]
    group.move("_tampered", name)


_ATTRIBUTE_OWNERS = (
    "",
    "household",
    "person",
    "spm_unit",
    "person/table",
    "spm_unit/table",
    "tax_unit/table",
    "person/_i_table/age",
    "spm_unit/_i_table/spm_unit_id/sortedLR",
)


def _tamper_registration(value, attribute: str):
    if attribute == "info":
        value["tampered"] = {}
    elif attribute == "non_index_axes":
        value[0][1].reverse()
    else:
        value.reverse()
    return value


# --------------------------------------------------------------------------
# Late refusals and the never-overwrite exposure
# --------------------------------------------------------------------------


def _intrude(path: Path, intruder: str) -> None:
    if intruder == "populated_directory":
        path.mkdir()
        (path / "foreign.txt").write_bytes(b"foreign")
    elif intruder == "empty_directory":
        path.mkdir()
    elif intruder == "file":
        path.write_bytes(b"foreign")
    else:
        path.symlink_to(path.parent / "nowhere")


def _intruder_state(path: Path) -> tuple:
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if path.is_dir():
        return (
            "directory",
            sorted((item.name, item.read_bytes()) for item in path.iterdir()),
        )
    return ("file", path.read_bytes())


__all__ = [name for name in globals() if not name.startswith("__")]
