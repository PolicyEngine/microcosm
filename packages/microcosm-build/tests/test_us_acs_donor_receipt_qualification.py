"""Exercise donor receipt qualification on invented fixtures only.

Every fixture here is synthetic. No pinned Census archive, no real donor and
no licensed microdata is read, and the pins the maintained loader verifies are
measured from the synthetic CSV this module writes.
"""

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
from tools import build_us_acs_donor_receipt_qualification as builder

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


def test_qualifies_the_pinned_donor(donor) -> None:
    fixture = donor()

    receipt = _qualify(fixture)

    child = fixture.output / f"{fixture.parent.stem}_receipt_qualified.h5"
    assert child.is_file()
    assert receipt["dataset"]["filename"] == child.name
    assert receipt["dataset"]["sha256"] == file_sha256(child)
    assert receipt["parent"]["sha256"] == builder.DONOR_SHA256
    assert receipt["dataset"]["sha256"] != receipt["parent"]["sha256"]
    assert receipt["added_columns"] == [
        {"name": "receives_wic", "entity": "person", "dtype": "bool"},
        {"name": "receives_snap", "entity": "spm_unit", "dtype": "bool"},
        {"name": "receives_tanf", "entity": "spm_unit", "dtype": "bool"},
    ]
    assert receipt["pooled_income_years"] == list(_INCOME_YEARS)
    assert receipt["code"] == _STUB_IDENTITY

    with pd.HDFStore(child, "r") as store:
        person = store["person"]
        spm_unit = store["spm_unit"]
    assert person["receives_wic"].tolist() == _EXPECTED["receives_wic"]
    assert person["receives_wic"].dtype == np.dtype(bool)
    assert spm_unit["receives_snap"].tolist() == _EXPECTED["receives_snap"]
    assert spm_unit["receives_tanf"].tolist() == _EXPECTED["receives_tanf"]
    assert list(spm_unit.columns)[-2:] == ["receives_snap", "receives_tanf"]

    receipt_path = fixture.output / builder.RECEIPT_FILENAME
    assert json.loads(receipt_path.read_text()) == receipt
    assert child.stat().st_mode & 0o777 == 0o400
    assert sorted(path.name for path in fixture.output.iterdir()) == sorted(
        [child.name, builder.RECEIPT_FILENAME]
    )
    assert _leftover_staging(fixture) == []


def test_receipt_reports_counts_by_year_role_and_channel(donor) -> None:
    receipt = _qualify(donor())

    wic = receipt["counts"]["receives_wic"]
    assert wic["entity"] == "person"
    assert wic["rows"] == len(_PEOPLE)
    assert wic["true"] == 2
    assert wic["false"] == len(_PEOPLE) - 2
    assert wic["by_source_year"] == {
        "2022": {"true": 2, "false": 2},
        "2023": {"true": 0, "false": 4},
    }
    assert wic["by_support_role"] == {
        "asec": {"true": 1, "false": 3},
        "puf_tax_detail": {"true": 1, "false": 3},
    }
    assert wic["gate_selected_channel"] == {
        "role": "puf_tax_detail",
        "rows": 4,
        "true": 1,
        "false": 3,
    }

    tanf = receipt["counts"]["receives_tanf"]
    assert tanf["entity"] == "spm_unit"
    assert tanf["true"] == 2
    assert tanf["by_source_year"] == {
        "2022": {"true": 2, "false": 0},
        "2023": {"true": 0, "false": 2},
    }
    assert tanf["gate_selected_channel"]["true"] == 1

    snap = receipt["counts"]["receives_snap"]
    assert snap["true"] == 2
    assert snap["gate_selected_channel"]["true"] == 1

    # PAW_TYP 2 (other cash welfare) and PAW_TYP 0 are not TANF, so the TANF
    # column is strictly narrower than "any positive PAW_VAL".
    assert tanf["true"] < sum(1 for row in _PEOPLE if row[5] > 0)


def test_receipt_records_the_archive_pins_and_measured_audit(donor) -> None:
    fixture = donor()

    receipt = _qualify(fixture)

    pins = receipt["archive_pins"]
    assert sorted(pins) == [str(year) for year in _INCOME_YEARS]
    for year in _INCOME_YEARS:
        entry = pins[str(year)]
        assert entry["survey_year"] == year + 1
        assert entry["member_sha256"] == file_sha256(fixture.source_paths[year])
        assert entry["pinned_rows"] == entry["pinned_paw_type_counts"][0] + sum(
            entry["pinned_paw_type_counts"][1:]
        )
    audit = receipt["source_audit"]
    assert sorted(audit) == [str(year) for year in _INCOME_YEARS]
    for year in _INCOME_YEARS:
        measured = audit[str(year)]
        assert measured["rows"] == pins[str(year)]["pinned_rows"]
        assert (
            measured["paw_positive_tanf_rows"]
            == pins[str(year)]["pinned_paw_positive_tanf_rows"]
        )


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


def test_untouched_objects_are_byte_identical(donor) -> None:
    fixture = donor()

    _qualify(fixture)

    child = fixture.output / f"{fixture.parent.stem}_receipt_qualified.h5"
    before = _objects_with_bytes(fixture.parent)
    after = _objects_with_bytes(child)
    assert before.keys() == after.keys()
    extended = {"person", "person/table", "spm_unit", "spm_unit/table"}
    for path in sorted(before):
        if path in extended:
            continue
        assert before[path] == after[path], f"{path} changed"

    # The extended tables keep every pre-existing field byte-identical.
    with h5py.File(fixture.parent, "r") as old, h5py.File(child, "r") as new:
        for table, appended in (
            ("person/table", ("receives_wic",)),
            ("spm_unit/table", ("receives_snap", "receives_tanf")),
        ):
            old_fields = old[table].dtype.names
            assert new[table].dtype.names == (*old_fields, *appended)
            old_values, new_values = old[table][()], new[table][()]
            for field in old_fields:
                assert old_values[field].tobytes() == new_values[field].tobytes()


def test_reload_preserves_every_untouched_column_and_the_weights(donor) -> None:
    from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5

    fixture = donor()
    parent_frame = load_legacy_calibrated_us_h5(fixture.parent)

    _qualify(fixture)

    child = fixture.output / f"{fixture.parent.stem}_receipt_qualified.h5"
    child_frame = load_legacy_calibrated_us_h5(child)
    assert child_frame.entities == parent_frame.entities
    for entity in parent_frame.entities:
        before = parent_frame.table(entity)
        after = child_frame.table(entity)
        appended = [
            column
            for column, owner in builder.QUALIFIED_COLUMNS.items()
            if owner == entity
        ]
        assert list(after.columns) == [*before.columns, *appended]
        pd.testing.assert_frame_equal(after[list(before.columns)], before)
    for entity in parent_frame.weighted_entities:
        assert np.array_equal(
            np.asarray(parent_frame.weights_for(entity).values),
            np.asarray(child_frame.weights_for(entity).values),
        )
        assert (
            parent_frame.weights_for(entity).kind
            == child_frame.weights_for(entity).kind
        )


# --------------------------------------------------------------------------
# The staging gate
# --------------------------------------------------------------------------


def _load_legacy_builder():
    root = Path(__file__).resolve().parents[3]
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


def test_qualified_donor_satisfies_the_staging_gate(donor) -> None:
    from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5

    legacy = _load_legacy_builder()
    fixture = donor()

    with pytest.raises(SystemExit) as refusal:
        legacy._require_dense_donor_coverage(
            load_legacy_calibrated_us_h5(fixture.parent),
            engine=_ReceiptDefaultsEngine(),
            target_families=_RECEIPT_TARGET_FAMILIES,
        )
    message = str(refusal.value)
    assert "hard ACS transfer-consumption gate" in message
    assert "selected channel='puf_tax_detail'" in message
    for column in ("person.receives_wic", "spm_unit.receives_snap"):
        assert f"{column}: transfer-consumed column is absent" in message

    _qualify(fixture)

    child = fixture.output / f"{fixture.parent.stem}_receipt_qualified.h5"
    legacy._require_dense_donor_coverage(
        load_legacy_calibrated_us_h5(child),
        engine=_ReceiptDefaultsEngine(),
        target_families=_RECEIPT_TARGET_FAMILIES,
    )


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_refuses_a_parent_that_is_not_the_pinned_donor(donor, monkeypatch) -> None:
    fixture = donor()
    monkeypatch.setattr(builder, "DONOR_SHA256", "f" * 64)

    with pytest.raises(builder.DonorQualificationError, match="exact pinned"):
        _qualify(fixture)

    assert not fixture.output.exists()


def test_accepts_the_spm_role_parent_and_records_which_lineage_it_qualified(
    donor, monkeypatch
) -> None:
    fixture = donor()
    monkeypatch.setattr(builder, "DONOR_SHA256", "f" * 64)
    monkeypatch.setattr(builder, "SPM_ROLE_PARENT_SHA256", file_sha256(fixture.parent))

    receipt = _qualify(fixture)

    assert receipt["parent"]["sha256"] == file_sha256(fixture.parent)
    assert "native SPM role" in receipt["parent"]["lineage"]


def test_pins_the_published_spm_role_release_digest() -> None:
    # The digest both published manifests declare (populace-us-2024-spm-20260909
    # and populace-us-2024-spm-20260915 sign the same bytes).
    assert set(builder.pinned_parents()) == {
        "48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e",
        "6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84",
    }


@pytest.mark.parametrize("column", ["receives_wic", "receives_snap", "receives_tanf"])
def test_refuses_a_preexisting_qualified_column(donor, monkeypatch, column) -> None:
    fixture = donor()
    entity = builder.QUALIFIED_COLUMNS[column]
    with pd.HDFStore(fixture.parent, "a") as store:
        table = store[entity]
        table[column] = np.zeros(len(table), dtype=bool)
        store.put(entity, table, format="table", data_columns=True)
    monkeypatch.setattr(builder, "DONOR_SHA256", file_sha256(fixture.parent))

    with pytest.raises(builder.DonorQualificationError) as refusal:
        _qualify(fixture)

    message = str(refusal.value)
    assert "already carries qualified column" in message
    assert column in message
    assert not fixture.output.exists()


def test_refuses_a_preexisting_paw_type_column(donor, monkeypatch) -> None:
    fixture = donor(person_overrides={"PAW_TYP": [0] * len(_PEOPLE)})
    monkeypatch.setattr(builder, "DONOR_SHA256", file_sha256(fixture.parent))

    with pytest.raises(builder.DonorQualificationError) as refusal:
        _qualify(fixture)

    assert "already carries PAW_TYP" in str(refusal.value)
    assert not fixture.output.exists()


@pytest.mark.parametrize("column", ["SPM_SNAPSUB", "WICYN", "PAW_VAL", "PERIDNUM"])
def test_refuses_a_missing_raw_column(donor, monkeypatch, column) -> None:
    fixture = donor(person_overrides={column: None})
    monkeypatch.setattr(builder, "DONOR_SHA256", file_sha256(fixture.parent))

    with pytest.raises(builder.DonorQualificationError) as refusal:
        _qualify(fixture)

    message = str(refusal.value)
    assert "missing required raw column" in message
    assert column in message
    assert not fixture.output.exists()


def test_refuses_an_unmatched_peridnum_without_leaking_identifiers(donor) -> None:
    # Person index 1 has PAW_VAL 1200; dropping it from the sidecar makes the
    # maintained restore refuse, naming its PERIDNUM in the original message.
    fixture = donor(drop_person=1)

    with pytest.raises(builder.DonorQualificationError) as refusal:
        _qualify(fixture)

    message = str(refusal.value)
    assert "_fill_spm_unit_reported_enrollment_inputs refused (ValueError)" in message
    assert "message is withheld" in message
    for index in range(len(_PEOPLE)):
        assert _peridnum(index) not in message
    assert re.search(r"\d{6,}", message) is None
    assert not fixture.output.exists()


def test_refuses_a_column_with_no_true_value(donor, monkeypatch) -> None:
    fixture = donor(person_overrides={"WICYN": [0] * len(_PEOPLE)})
    monkeypatch.setattr(builder, "DONOR_SHA256", file_sha256(fixture.parent))

    with pytest.raises(builder.DonorQualificationError) as refusal:
        _qualify(fixture)

    message = str(refusal.value)
    assert "person.receives_wic has no true value across the donor" in message
    assert not fixture.output.exists()


def test_refuses_a_column_with_no_true_value_in_the_selected_channel(
    donor, monkeypatch
) -> None:
    # True only on the native rows; the gate resolves puf_tax_detail first, so
    # a donor-wide count would look qualifying while the gate still refuses.
    fixture = donor(person_overrides={"WICYN": [1, 1, 1, 1, 0, 0, 0, 0]})
    monkeypatch.setattr(builder, "DONOR_SHA256", file_sha256(fixture.parent))

    with pytest.raises(builder.DonorQualificationError) as refusal:
        _qualify(fixture)

    message = str(refusal.value)
    assert "no true value inside the gate-selected channel 'puf_tax_detail'" in message
    assert not fixture.output.exists()


def test_refuses_an_existing_output_directory(donor) -> None:
    fixture = donor()
    fixture.output.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        _qualify(fixture)


def test_refuses_a_sidecar_that_is_not_present_locally(donor, tmp_path) -> None:
    fixture = donor()
    absent = dict(fixture.source_paths)
    absent[_INCOME_YEARS[0]] = tmp_path / "absent.csv"

    with pytest.raises(builder.DonorQualificationError) as refusal:
        _qualify(fixture, source_paths=absent)

    assert "never downloads" in str(refusal.value)
    assert not fixture.output.exists()


def test_refuses_a_partial_source_mapping_without_downloading(
    donor, monkeypatch
) -> None:
    """A mapping that omits a pooled income year must refuse before any fetch."""

    fixture = donor()
    omitted = _INCOME_YEARS[-1]
    partial = {
        year: path for year, path in fixture.source_paths.items() if year != omitted
    }
    fetches: list[int] = []

    def refuse_download(income_year, *args, **kwargs):
        fetches.append(income_year)
        raise AssertionError("the tool must never reach the Census download path")

    monkeypatch.setattr(pats, "fetch_asec_education_assistance_source", refuse_download)

    with pytest.raises(builder.DonorQualificationError) as refusal:
        _qualify(fixture, source_paths=partial)

    assert "never downloads" in str(refusal.value)
    assert str(omitted) in str(refusal.value)
    assert fetches == []
    assert not fixture.output.exists()


def test_refuses_a_drifted_archive_pin_without_leaking_rows(donor, monkeypatch) -> None:
    fixture = donor()
    fixture.source_paths[_INCOME_YEARS[0]].write_text(
        fixture.source_paths[_INCOME_YEARS[0]].read_text() + "1,1,1,0,0,0,1.0\n"
    )

    with pytest.raises(builder.DonorQualificationError) as refusal:
        _qualify(fixture)

    message = str(refusal.value)
    assert "load_asec_public_assistance_type_sources refused" in message
    assert "message is withheld" in message
    assert not fixture.output.exists()


def test_producer_identity_refuses_a_foreign_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(builder, "_repository_root", lambda: tmp_path)

    with pytest.raises(builder.DonorQualificationError) as refusal:
        builder._producer_identity()

    assert "install this workspace's local shards first" in str(refusal.value)


def test_producer_files_are_the_modules_that_are_actually_imported() -> None:
    root = builder._repository_root()
    for filename in builder._PRODUCER_FILES:
        assert (root / filename).is_file(), filename


def test_producer_receipt_hashes_the_code_that_decides_the_channel_counts() -> None:
    # resolve_acs_donor_channel and Frame.select decide the gate-selected
    # channel counts that refusal 7 judges, so their sources are receipt inputs.
    assert {
        "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py",
        "packages/microcosm-frame/src/microcosm/frame/bundle.py",
        "packages/microcosm-frame/src/microcosm/frame/schema.py",
    } <= set(builder._PRODUCER_FILES)
    root = builder._repository_root()

    identity = builder._producer_identity()

    assert identity["source_files_sha256"] == {
        name: file_sha256(root / name) for name in builder._PRODUCER_FILES
    }
    assert len(identity["git_commit"]) == 40


def _foreign_resolver(donor, channel):
    return donor, channel


class _ForeignFrame:
    def select(self, person_mask):
        return self


class _ForeignSchema:
    pass


@pytest.mark.parametrize(
    ("name", "replacement", "filename"),
    [
        (
            "resolve_acs_donor_channel",
            _foreign_resolver,
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py",
        ),
        (
            "Frame",
            _ForeignFrame,
            "packages/microcosm-frame/src/microcosm/frame/bundle.py",
        ),
        (
            "EntitySchema",
            _ForeignSchema,
            "packages/microcosm-frame/src/microcosm/frame/schema.py",
        ),
    ],
)
def test_producer_identity_refuses_channel_code_from_elsewhere(
    monkeypatch, name, replacement, filename
) -> None:
    monkeypatch.setattr(builder, name, replacement)

    with pytest.raises(builder.DonorQualificationError) as refusal:
        builder._producer_identity()

    assert f"Producer must execute this checkout's {filename}" in str(refusal.value)


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


def test_verifier_accepts_the_untampered_append_and_states_its_extent(
    appended,
) -> None:
    report = _verify(appended)

    assert report == appended.report
    assert report["all_preexisting_fields_and_datasets_exact"] is True
    assert report["all_other_objects_and_attributes_exact"] is True
    assert report["rewritten_group_attributes"] == {
        "person": ["data_columns", "info", "non_index_axes", "values_cols"],
        "spm_unit": ["data_columns", "info", "non_index_axes", "values_cols"],
    }
    person_fields = len(_person_table().columns) + 1  # plus the pandas index
    assert report["tables_extended"]["person/table"] == {
        "rows": len(_PEOPLE),
        "preexisting_fields_checked_including_index": person_fields,
        "appended_fields": ["receives_wic"],
        "storage": "HDF bitfield8; pandas bool",
        "record_size_increase_bytes": 1,
        "added_table_attributes": sorted(
            [
                f"FIELD_{person_fields}_NAME",
                f"FIELD_{person_fields}_FILL",
                "receives_wic_dtype",
                "receives_wic_kind",
                "receives_wic_meta",
            ]
        ),
    }
    spm = report["tables_extended"]["spm_unit/table"]
    assert spm["appended_fields"] == ["receives_snap", "receives_tanf"]
    assert spm["record_size_increase_bytes"] == 2
    assert len(spm["added_table_attributes"]) == 10


@pytest.mark.parametrize(
    ("table", "field"),
    [
        ("person/table", "index"),
        ("person/table", "PERIDNUM"),
        ("person/table", "age"),
        ("person/table", "person_spm_unit_id"),
        ("person/table", "is_female"),
        ("spm_unit/table", "spm_unit_id"),
        ("spm_unit/table", "spm_unit_support_channel"),
        ("spm_unit/table", "takes_up_snap_if_eligible"),
    ],
)
def test_verifier_refuses_a_changed_existing_field(appended, table, field) -> None:
    with h5py.File(appended.child, "r+") as h5:
        _set_field_byte(h5[table], field, row=1)

    with pytest.raises(
        builder.DonorQualificationError,
        match=f"Existing {re.escape(table)} field changed: {field}$",
    ):
        _verify(appended)


def test_verifier_refuses_reordered_rows(appended) -> None:
    with h5py.File(appended.child, "r+") as h5:
        table = h5["person/table"]
        _write_records(table, builder._raw_rows(table, 0, len(table))[::-1])

    with pytest.raises(
        builder.DonorQualificationError, match="Existing person/table field changed"
    ):
        _verify(appended)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ("household/table", "household_weight"),
        ("tax_unit/table", "tax_unit_id"),
        ("_time_period/table", "time_period"),
        ("person/_i_table/age/sortedLR", None),
        ("spm_unit/_i_table/spm_unit_id/sortedLR", None),
    ],
)
def test_verifier_refuses_a_changed_value_in_any_other_dataset(
    appended, path, field
) -> None:
    with h5py.File(appended.child, "r+") as h5:
        dataset = h5[path]
        if field is None:
            dataset[0] = dataset[0] + 1
        else:
            _set_field_byte(dataset, field)

    with pytest.raises(
        builder.DonorQualificationError,
        match=f"Existing HDF dataset values changed: {re.escape(path)}$",
    ):
        _verify(appended)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("person/table", "receives_wic"),
        ("spm_unit/table", "receives_snap"),
        ("spm_unit/table", "receives_tanf"),
    ],
)
def test_verifier_refuses_a_non_boolean_appended_value(appended, table, column) -> None:
    with h5py.File(appended.child, "r+") as h5:
        _set_field_byte(h5[table], column, value=2)

    with pytest.raises(
        builder.DonorQualificationError, match=f"Appended {column} is not Boolean"
    ):
        _verify(appended)


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


@pytest.mark.parametrize("mutation", ["added", "removed", "value", "dtype"])
@pytest.mark.parametrize("owner", _ATTRIBUTE_OWNERS)
def test_verifier_refuses_any_other_attribute_change(appended, owner, mutation) -> None:
    with h5py.File(appended.child, "r+") as h5:
        attrs = h5[owner].attrs if owner else h5.attrs
        if mutation == "added":
            attrs["unapproved"] = np.int16(1)
        elif mutation == "removed":
            del attrs["CLASS"]
        elif mutation == "value":
            # Same HDF type and length, different bytes.
            attrs.modify("CLASS", bytes(attrs["CLASS"])[::-1])
        else:
            attrs["CLASS"] = np.int64(7)

    with pytest.raises(
        builder.DonorQualificationError,
        match=f"HDF attributes? changed at {re.escape(owner)}(:CLASS)?$",
    ):
        _verify(appended)


def _tamper_registration(value, attribute: str):
    if attribute == "info":
        value["tampered"] = {}
    elif attribute == "non_index_axes":
        value[0][1].reverse()
    else:
        value.reverse()
    return value


@pytest.mark.parametrize(
    "attribute", ["data_columns", "info", "non_index_axes", "values_cols"]
)
@pytest.mark.parametrize("group", ["person", "spm_unit"])
def test_verifier_refuses_a_registration_change_beyond_the_append(
    appended, group, attribute
) -> None:
    with h5py.File(appended.child, "r+") as h5:
        attrs = h5[group].attrs
        value = _tamper_registration(pickle.loads(bytes(attrs[attribute])), attribute)
        attrs[attribute] = np.bytes_(pickle.dumps(value, protocol=0))

    with pytest.raises(
        builder.DonorQualificationError,
        match=f"Unexpected pandas column registration change at {group}:{attribute}",
    ):
        _verify(appended)


@pytest.mark.parametrize("group", ["person", "spm_unit"])
def test_verifier_refuses_correct_registration_bytes_with_the_wrong_type(
    appended, group
) -> None:
    with h5py.File(appended.child, "r+") as h5:
        attrs = h5[group].attrs
        attrs["values_cols"] = np.frombuffer(bytes(attrs["values_cols"]), np.uint8)

    with pytest.raises(
        builder.DonorQualificationError,
        match=f"Unexpected pandas column registration change at {group}:values_cols",
    ):
        _verify(appended)


@pytest.mark.parametrize(
    "attribute", ["data_columns", "info", "non_index_axes", "values_cols"]
)
def test_verifier_refuses_a_registration_change_on_an_unplanned_group(
    appended, attribute
) -> None:
    with h5py.File(appended.child, "r+") as h5:
        attrs = h5["household"].attrs
        value = _tamper_registration(pickle.loads(bytes(attrs[attribute])), attribute)
        if value == pickle.loads(bytes(attrs[attribute])):
            value = _tamper_registration(value, "info")
        attrs[attribute] = np.bytes_(pickle.dumps(value, protocol=0))

    with pytest.raises(
        builder.DonorQualificationError,
        match=f"HDF attribute changed at household:{attribute}",
    ):
        _verify(appended)


@pytest.mark.parametrize(
    ("table", "attribute", "mutation"),
    [
        ("person/table", "receives_wic_dtype", "value"),
        ("person/table", "receives_wic_kind", "uint8"),
        ("person/table", "FIELD_{first}_FILL", "int64"),
        ("spm_unit/table", "FIELD_{first}_NAME", "value"),
        ("spm_unit/table", "receives_tanf_meta", "removed"),
        ("spm_unit/table", "receives_snap_kind", "value"),
    ],
)
def test_verifier_refuses_invalid_appended_column_metadata(
    appended, table, attribute, mutation
) -> None:
    with h5py.File(appended.child, "r+") as h5:
        dataset = h5[table]
        first = dataset.dtype.names.index(_PLAN_ORDER[table.split("/")[0]][0])
        name = attribute.format(first=first)
        attrs = dataset.attrs
        if mutation == "removed":
            del attrs[name]
        elif mutation == "int64":
            attrs[name] = np.int64(0)
        elif mutation == "uint8":
            attrs[name] = np.frombuffer(bytes(attrs[name]), np.uint8)
        else:
            attrs.modify(name, bytes(attrs[name])[::-1])

    with pytest.raises(
        builder.DonorQualificationError,
        match=(
            f"Invalid appended column metadata at {re.escape(table)}:{name}$"
            f"|HDF attributes changed at {re.escape(table)}$"
        ),
    ):
        _verify(appended)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_root_group",
        "extra_root_dataset",
        "extra_dataset_in_person",
        "extra_dataset_in_index",
        "removed_group",
        "removed_index_dataset",
    ],
)
def test_verifier_refuses_an_added_or_removed_object(appended, mutation) -> None:
    with h5py.File(appended.child, "r+") as h5:
        if mutation == "extra_root_group":
            h5.create_group("extra")
        elif mutation == "extra_root_dataset":
            h5.create_dataset("extra", data=[1])
        elif mutation == "extra_dataset_in_person":
            h5["person"].create_dataset("extra", data=[1])
        elif mutation == "extra_dataset_in_index":
            h5["spm_unit/_i_table/spm_unit_id"].create_dataset("extra", data=[1])
        elif mutation == "removed_group":
            del h5["family"]
        else:
            del h5["person/_i_table/age/sortedLR"]

    with pytest.raises(
        builder.DonorQualificationError,
        match="HDF groups or datasets were added or removed",
    ):
        _verify(appended)


@pytest.mark.parametrize("mutation", ["soft_link", "external_link", "alias"])
def test_verifier_refuses_links_and_aliases(appended, mutation) -> None:
    with h5py.File(appended.child, "r+") as h5:
        if mutation == "soft_link":
            h5["extra"] = h5py.SoftLink("/person")
        elif mutation == "external_link":
            h5["extra"] = h5py.ExternalLink(str(appended.parent), "/person")
        else:
            h5["extra"] = h5["person"]

    # The object walk is the reviewed h5_enrichment primitive; its refusals
    # are plain ValueErrors, which the CLI reports as refusals too.
    with pytest.raises(ValueError, match="Nonlocal HDF link|Aliased HDF object"):
        _verify(appended)


@pytest.mark.parametrize(
    ("table", "mutation"),
    [
        ("person/table", "order"),
        ("spm_unit/table", "existing_bitfield_as_uint8"),
        ("spm_unit/table", "appended_bitfield_as_uint8"),
        ("person/table", "appended_bitfield_as_uint8"),
    ],
)
def test_verifier_refuses_an_extended_record_type_beyond_the_append(
    appended, table, mutation
) -> None:
    with h5py.File(appended.child, "r+") as h5:
        old_type = h5[table].id.get_type()
        datatype = h5py.h5t.create(h5py.h5t.COMPOUND, old_type.get_size())
        indices = list(range(old_type.get_nmembers()))
        if mutation == "order":
            indices.reverse()
        planned = {column.encode() for column in _PLAN_ORDER[table.split("/")[0]]}
        for index in indices:
            name = old_type.get_member_name(index)
            member_type = old_type.get_member_type(index)
            existing = mutation == "existing_bitfield_as_uint8" and (
                name == b"takes_up_snap_if_eligible"
            )
            appended_field = mutation == "appended_bitfield_as_uint8" and (
                name in planned
            )
            if existing or appended_field:
                # NumPy still sees uint8; the HDF bitfield identity is lost.
                member_type = h5py.h5t.STD_U8LE
            datatype.insert(name, old_type.get_member_offset(index), member_type)
        group, name = table.split("/")
        _recreate_dataset(h5[group], name, datatype=datatype)

    with pytest.raises(
        builder.DonorQualificationError,
        match=f"{re.escape(table)} dtype differs beyond the appended fields",
    ):
        _verify(appended)


def test_verifier_refuses_a_changed_dtype_elsewhere(appended) -> None:
    with h5py.File(appended.child, "r+") as h5:
        old_type = h5["tax_unit/table"].id.get_type()
        datatype = h5py.h5t.create(h5py.h5t.COMPOUND, old_type.get_size())
        for index in range(old_type.get_nmembers()):
            # Same size and field names, but big-endian integers.
            datatype.insert(
                old_type.get_member_name(index),
                old_type.get_member_offset(index),
                h5py.h5t.STD_I64BE,
            )
        _recreate_dataset(h5["tax_unit"], "table", datatype=datatype)

    with pytest.raises(
        builder.DonorQualificationError, match="HDF dtype changed at tax_unit/table$"
    ):
        _verify(appended)


@pytest.mark.parametrize(
    ("path", "change"),
    [
        ("household/table", {"rows": 3}),
        ("household/table", {"chunks": (7,)}),
        ("person/table", {"rows": 7}),
        ("spm_unit/table", {"chunks": (3,)}),
    ],
)
def test_verifier_refuses_a_changed_shape_or_storage(appended, path, change) -> None:
    with h5py.File(appended.child, "r+") as h5:
        group, name = path.split("/")
        _recreate_dataset(h5[group], name, **change)

    with pytest.raises(
        builder.DonorQualificationError,
        match=f"HDF shape or storage changed at {re.escape(path)}$",
    ):
        _verify(appended)


@pytest.mark.parametrize(
    ("order", "match"),
    [
        # The spm_unit columns in the opposite order to the written one.
        (
            {
                "person": ("receives_wic",),
                "spm_unit": ("receives_tanf", "receives_snap"),
            },
            "Unexpected pandas column registration change at spm_unit:",
        ),
        # A written table the plan does not declare.
        ({"person": ("receives_wic",)}, "HDF attribute changed at spm_unit:"),
        # A declared column that was never written.
        (
            {**_PLAN_ORDER, "household": ("receives_wic",)},
            "Unexpected pandas column registration change at household:",
        ),
        (
            {"person": ("receives_wix",), "spm_unit": _PLAN_ORDER["spm_unit"]},
            "Unexpected pandas column registration change at person:",
        ),
    ],
)
def test_verifier_refuses_a_plan_other_than_the_written_one(
    appended, order, match
) -> None:
    # The verifier judges the plan's groups and column order, not its values.
    plan = {
        group: {column: np.zeros(0, dtype=bool) for column in columns}
        for group, columns in order.items()
    }

    with pytest.raises(builder.DonorQualificationError, match=match):
        _verify(appended, plan)


def test_append_refuses_a_parent_hash_mismatch(tmp_path) -> None:
    parent, child = tmp_path / "parent.h5", tmp_path / "child.h5"
    _write_parent(parent)
    before = file_sha256(parent)

    with pytest.raises(
        builder.DonorQualificationError, match="Parent H5 SHA-256 mismatch"
    ):
        builder.append_boolean_fields(
            parent, child, _plan(), expected_parent_sha256="0" * 64
        )

    assert not child.exists()
    assert file_sha256(parent) == before


@pytest.mark.parametrize(
    ("plan", "match"),
    [
        (
            {"person": {"receives_wic": np.ones(len(_PEOPLE), dtype=np.uint8)}},
            "receives_wic must be a one-dimensional Boolean array",
        ),
        (
            {"person": {"receives_wic": np.ones((len(_PEOPLE), 1), dtype=bool)}},
            "receives_wic must be a one-dimensional Boolean array",
        ),
        (
            {"person": {"receives_wic": np.ones(len(_PEOPLE) - 1, dtype=bool)}},
            "receives_wic coverage does not match the person rows",
        ),
        (
            {"spm_unit": {"receives_snap": np.ones(len(_PEOPLE), dtype=bool)}},
            "receives_snap coverage does not match the spm_unit rows",
        ),
        ({"spm_unit": {}}, "Empty append plan for spm_unit"),
        (
            {"benunit": {"receives_wic": np.ones(4, dtype=bool)}},
            "Parent has no benunit/table",
        ),
    ],
)
def test_append_refuses_an_invalid_plan(tmp_path, plan, match) -> None:
    parent, child = tmp_path / "parent.h5", tmp_path / "child.h5"
    _write_parent(parent)
    before = file_sha256(parent)

    with pytest.raises(builder.DonorQualificationError, match=match):
        builder.append_boolean_fields(
            parent, child, plan, expected_parent_sha256=before
        )

    assert not child.exists()
    assert file_sha256(parent) == before


@pytest.mark.parametrize("column", ["receives_wic", "PERIDNUM"])
def test_append_refuses_a_column_already_in_the_hdf_table(tmp_path, column) -> None:
    parent, child = tmp_path / "parent.h5", tmp_path / "child.h5"
    _write_parent(parent, person_overrides={"receives_wic": [False] * len(_PEOPLE)})
    before = file_sha256(parent)

    with pytest.raises(
        builder.DonorQualificationError,
        match="Parent already carries a qualified column at person/table",
    ):
        builder.append_boolean_fields(
            parent,
            child,
            {"person": {column: np.ones(len(_PEOPLE), dtype=bool)}},
            expected_parent_sha256=before,
        )

    assert not child.exists()


def test_append_refuses_a_filtered_table(tmp_path) -> None:
    parent, child = tmp_path / "parent.h5", tmp_path / "child.h5"
    with pd.HDFStore(parent, "w", complevel=1, complib="zlib") as store:
        store.put("person", _person_table(), format="table", data_columns=True)

    with pytest.raises(
        builder.DonorQualificationError,
        match="Expected an unfiltered native pandas table at person/table",
    ):
        builder.append_boolean_fields(
            parent,
            child,
            {"person": {"receives_wic": np.ones(len(_PEOPLE), dtype=bool)}},
            expected_parent_sha256=file_sha256(parent),
        )

    assert not child.exists()


@pytest.mark.parametrize("destination", ["existing", "parent", "symlink"])
def test_append_never_overwrites_its_destination(tmp_path, destination) -> None:
    parent = tmp_path / "parent.h5"
    _write_parent(parent)
    child = tmp_path / "child.h5"
    if destination == "existing":
        child.write_bytes(b"existing artifact")
    elif destination == "parent":
        child = parent
    else:
        child.symlink_to(parent)
    original = child.read_bytes()

    with pytest.raises(FileExistsError):
        builder.append_boolean_fields(
            parent, child, _plan(), expected_parent_sha256=file_sha256(parent)
        )

    assert child.read_bytes() == original


def test_append_refuses_values_that_differ_from_the_derivation(
    tmp_path, monkeypatch
) -> None:
    parent, child = tmp_path / "parent.h5", tmp_path / "child.h5"
    _write_parent(parent)
    before = file_sha256(parent)
    write = builder._append_group_fields

    def write_inverted(handle, group_name, columns):
        write(handle, group_name, {name: ~values for name, values in columns.items()})

    monkeypatch.setattr(builder, "_append_group_fields", write_inverted)

    with pytest.raises(
        builder.DonorQualificationError,
        match="Written receives_wic differs from the derivation",
    ):
        builder.append_boolean_fields(
            parent, child, _plan(), expected_parent_sha256=before
        )

    assert not child.exists()
    assert file_sha256(parent) == before


# --------------------------------------------------------------------------
# Late refusals and the never-overwrite exposure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["preservation", "reload", "producer_drift"])
def test_a_late_refusal_leaves_no_output_and_no_staging(
    donor, monkeypatch, stage
) -> None:
    fixture = donor()
    if stage == "preservation":

        def refuse_preservation(*args, **kwargs):
            raise builder.DonorQualificationError("HDF attributes changed at person")

        monkeypatch.setattr(builder, "compare_boolean_append", refuse_preservation)
        match = "HDF attributes changed at person"
    elif stage == "reload":

        def refuse_reload(*args, **kwargs):
            raise builder.DonorQualificationError("person weights changed")

        monkeypatch.setattr(builder, "verify_reload", refuse_reload)
        match = "person weights changed"
    else:
        identities = iter(
            [
                _STUB_IDENTITY,
                {**_STUB_IDENTITY, "source_files_sha256": {"synthetic.py": "e" * 64}},
            ]
        )
        monkeypatch.setattr(builder, "_producer_identity", lambda: next(identities))
        match = "Producer source files changed while qualifying the donor"

    with pytest.raises(builder.DonorQualificationError, match=match):
        _qualify(fixture)

    assert not fixture.output.exists()
    assert not fixture.output.is_symlink()
    assert _leftover_staging(fixture) == []


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


@pytest.mark.parametrize(
    "intruder", ["populated_directory", "empty_directory", "file", "dangling_symlink"]
)
def test_never_overwrites_an_output_that_appears_during_the_build(
    donor, monkeypatch, intruder
) -> None:
    fixture = donor()
    verify = builder.verify_reload
    expected: list[tuple] = []

    def verify_then_intrude(*args, **kwargs):
        report = verify(*args, **kwargs)
        _intrude(fixture.output, intruder)
        expected.append(_intruder_state(fixture.output))
        return report

    monkeypatch.setattr(builder, "verify_reload", verify_then_intrude)

    with pytest.raises(FileExistsError, match="Output appeared during the build"):
        _qualify(fixture)

    assert [_intruder_state(fixture.output)] == expected
    assert _leftover_staging(fixture) == []


@pytest.mark.parametrize("intrusion", ["populate", "replace_with_file", "symlink"])
def test_exposure_refuses_an_intrusion_into_its_reservation(
    tmp_path, monkeypatch, intrusion
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "child.h5").write_bytes(b"verified")
    output = tmp_path / "qualified"
    rename = os.rename

    def intrude_then_rename(source, target):
        target = Path(target)
        # The reservation must already exist, empty, when the rename runs.
        assert target.is_dir() and not any(target.iterdir())
        if intrusion == "populate":
            (target / "foreign.txt").write_bytes(b"foreign")
        else:
            target.rmdir()
            if intrusion == "replace_with_file":
                target.write_bytes(b"foreign")
            else:
                target.symlink_to(tmp_path / "elsewhere")
        return rename(source, target)

    monkeypatch.setattr(builder.os, "rename", intrude_then_rename)
    with pytest.raises(FileExistsError, match="Output appeared during the build"):
        builder._expose_without_overwrite(staging, output)
    monkeypatch.undo()

    if intrusion == "populate":
        assert _intruder_state(output) == ("directory", [("foreign.txt", b"foreign")])
    elif intrusion == "replace_with_file":
        assert _intruder_state(output) == ("file", b"foreign")
    else:
        assert _intruder_state(output) == ("symlink", str(tmp_path / "elsewhere"))
    assert (staging / "child.h5").read_bytes() == b"verified"


def test_exposure_releases_its_reservation_when_the_rename_fails(
    tmp_path, monkeypatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "qualified"

    def failing_rename(source, target):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(builder.os, "rename", failing_rename)
    with pytest.raises(PermissionError):
        builder._expose_without_overwrite(staging, output)
    monkeypatch.undo()

    assert not output.exists()
    assert staging.is_dir()


def test_exposure_moves_the_staging_directory_whole(tmp_path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "child.h5").write_bytes(b"verified")
    output = tmp_path / "qualified"

    builder._expose_without_overwrite(staging, output)

    assert not staging.exists()
    assert _intruder_state(output) == ("directory", [("child.h5", b"verified")])
