"""Exercise donor receipt qualification on invented fixtures only.

Every fixture here is synthetic. No pinned Census archive, no real donor and
no licensed microdata is read, and the pins the maintained loader verifies are
measured from the synthetic CSV this module writes.
"""

from __future__ import annotations

import importlib.util
import json
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
