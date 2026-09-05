"""Invented archives only: source custody, typed content and receipt integrity."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_housing_donor as donor
from microcosm.build.us_runtime.acs_sources import _file_identity

_SPEC = importlib.util.spec_from_file_location(
    "acs_housing_fixture", Path(__file__).with_name("test_us_acs_housing_donor.py")
)
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)
TARGET, _source, _tables = _FIXTURE.TARGET, _FIXTURE._source, _FIXTURE._tables


def _authenticated_source(tmp_path, monkeypatch):
    source = _source(tmp_path, *_tables())
    manifest = donor.load_acs_source_manifest()
    artifacts = []
    for role, path in (
        ("household", source.household_zip),
        ("person", source.person_zip),
    ):
        size, digest = _file_identity(path)
        artifacts.append(
            replace(manifest.artifact(role), size_bytes=size, sha256=digest)
        )
    manifest = replace(manifest, artifacts=tuple(artifacts))
    monkeypatch.setattr(donor, "load_acs_source_manifest", lambda: manifest)
    return source


@pytest.mark.parametrize(
    "kind",
    [
        "target",
        "predictor",
        "weight",
        "flag",
        "owner_universe",
        "index",
        "dtype",
        "column_order",
        "string_storage",
        "index_storage",
        "column_name",
    ],
)
def test_donor_content_mutation_refused_before_sampler(tmp_path, monkeypatch, kind):
    source = _authenticated_source(tmp_path, monkeypatch)
    loaded = donor.load_acs_2024_housing_donor(source, chunksize=1)
    if kind == "target":
        loaded.table.loc["r1", TARGET] = 1.0
    elif kind == "predictor":
        loaded.table.loc["r1", "employment_income_before_lsr"] = 999999.0
    elif kind == "weight":
        loaded.table.loc["r1", "household_weight"] = 999999.0
    elif kind == "flag":
        loaded.table.loc["r3", "FRNTP"] = 0
    elif kind == "owner_universe":
        loaded.table.loc["o1", TARGET] = 10.0
    elif kind == "index":
        loaded.table.index = pd.Index([f"other-{x}" for x in loaded.table.index])
    elif kind == "dtype":
        loaded.table["age"] = loaded.table["age"].astype("float32")
    elif kind == "column_order":
        loaded.table.insert(0, "moved", loaded.table.pop("age"))
        loaded.table.rename(columns={"moved": "age"}, inplace=True)
    elif kind == "string_storage":
        loaded.table["source_household_id"] = loaded.table[
            "source_household_id"
        ].astype(pd.StringDtype(storage="pyarrow"))
    elif kind == "index_storage":
        loaded.table.index = loaded.table.index.astype(
            pd.StringDtype(storage="pyarrow")
        )
    else:
        loaded.table.columns.name = "changed"
    calls = []
    original = donor._archived_joint_training_sample

    def traced_sample(table):
        calls.append(True)
        return original(table)

    monkeypatch.setattr(donor, "_archived_joint_training_sample", traced_sample)
    with pytest.raises(ValueError, match="content identity"):
        donor.prepare_acs_2024_reported_rent(loaded)
    assert not calls


@pytest.mark.parametrize(
    "kind",
    ["source_evidence", "archive_sha", "price_year", "recipe_sha", "household_count"],
)
def test_returned_donor_provenance_is_detached(tmp_path, monkeypatch, kind):
    source = _authenticated_source(tmp_path, monkeypatch)
    loaded = donor.load_acs_2024_housing_donor(source)
    if kind == "source_evidence":
        loaded = donor.project_synthetic_acs_2024_housing_donor(*_tables())
    original = copy.deepcopy(loaded.provenance)
    exposed = loaded.provenance
    if kind == "source_evidence":
        exposed["source_evidence"] = "authenticated_archives"
    elif kind == "archive_sha":
        exposed["archives"]["household"]["sha256"] = "0" * 64
    elif kind == "price_year":
        exposed["price_year"] = 2099
    elif kind == "recipe_sha":
        exposed["recipe_sha256"] = "0" * 64
    else:
        exposed["household_rows"] = 9000000
    assert loaded.provenance == original
    prepared = donor.prepare_acs_2024_reported_rent(loaded)
    for key, value in original.items():
        assert prepared.provenance[key] == value
    assert prepared.provenance["source_evidence"] == original["source_evidence"]


def test_prepared_receipts_and_verified_training_are_detached(tmp_path, monkeypatch):
    source = _authenticated_source(tmp_path, monkeypatch)
    loaded = donor.load_acs_2024_housing_donor(source)
    prepared = donor.prepare_acs_2024_reported_rent(loaded)
    before = copy.deepcopy(prepared.provenance)
    loaded.provenance["archives"]["household"]["sha256"] = "f" * 64
    prepared.provenance["feature_schema"]["category_levels"]["state_code_str"].append(
        "99"
    )
    assert prepared.provenance == before
    training = prepared.verified_training()
    training.loc["r1", TARGET] = 1.0
    assert prepared.verified_training().loc["r1", TARGET] == 7200.0
    with pytest.raises(FrozenInstanceError):
        prepared.predictors = ()


@pytest.mark.parametrize("kind", ["target", "weight", "index", "dtype"])
def test_training_mutation_refused_when_consumed(kind):
    prepared = donor.prepare_acs_2024_reported_rent(
        donor.project_synthetic_acs_2024_housing_donor(*_tables())
    )
    if kind == "target":
        prepared.training.loc["r1", TARGET] = 1.0
    elif kind == "weight":
        prepared.training.loc["r1", "household_weight"] = 100.0
    elif kind == "index":
        prepared.training.index = prepared.training.index.rename("different")
    else:
        prepared.training["age"] = prepared.training["age"].astype("float32")
    with pytest.raises(ValueError, match="content identity"):
        prepared.verified_training()


def test_callers_cannot_mint_authenticated_objects_from_metadata():
    projected = donor.project_synthetic_acs_2024_housing_donor(*_tables())
    claims = projected.provenance | {"source_evidence": "authenticated_archives"}
    with pytest.raises(TypeError, match="loader|projector"):
        donor.Acs2024HousingDonor(projected.table, claims)
    with pytest.raises(TypeError, match="prepare"):
        donor.Acs2024ReportedRentPreparation(projected.table, (), claims)


def test_preparation_refuses_duck_typed_forged_receipt():
    projected = donor.project_synthetic_acs_2024_housing_donor(*_tables())
    claims = projected.provenance | {"source_evidence": "authenticated_archives"}
    forged = SimpleNamespace(
        table=projected.table, _receipt_bytes=json.dumps(claims).encode()
    )
    with pytest.raises(TypeError, match="issued"):
        donor.prepare_acs_2024_reported_rent(forged)


def test_training_consumption_refuses_duck_typed_receipt():
    prepared = donor.prepare_acs_2024_reported_rent(
        donor.project_synthetic_acs_2024_housing_donor(*_tables())
    )
    forged = SimpleNamespace(
        training=prepared.training,
        predictors=prepared.predictors,
        _receipt_bytes=json.dumps(prepared.provenance).encode(),
    )
    with pytest.raises(TypeError, match="issued"):
        donor.Acs2024ReportedRentPreparation.verified_training(forged)


@pytest.mark.parametrize("role", ["household", "person"])
@pytest.mark.parametrize("restore", [False, True])
def test_parser_consumes_verified_private_snapshot(
    tmp_path, monkeypatch, role, restore
):
    source = _authenticated_source(tmp_path, monkeypatch)
    identities = {
        name: _file_identity(path)
        for name, path in (
            ("household", source.household_zip),
            ("person", source.person_zip),
        )
    }
    h, p = _tables()
    if role == "household":
        h.loc[h.SERIALNO == "r1", "RNTP"] = 600
    else:
        p.loc[(p.SERIALNO == "r1") & (p.RELSHIPP == 20), "WAGP"] = 200
    alternate = tmp_path / "replacement"
    alternate.mkdir()
    other = _source(alternate, h, p)
    victim = source.household_zip if role == "household" else source.person_zip
    replacement = other.household_zip if role == "household" else other.person_zip
    original_bytes = victim.read_bytes()
    original_reader, original_verifier = donor._read_archive, donor._verify_download
    verified, parsed_paths = [], []

    def verify(artifact, **kwargs):
        original_verifier(artifact, **kwargs)
        verified.append(artifact.role)

    def reader(path, **kwargs):
        current_role = (
            "household" if kwargs["member_prefix"] == "psam_hus" else "person"
        )
        parsed_paths.append(path)
        assert current_role in verified
        if current_role == role:
            shutil.copyfile(replacement, victim)
        result = original_reader(path, **kwargs)
        if current_role == role and restore:
            victim.write_bytes(original_bytes)
        return result

    monkeypatch.setattr(donor, "_verify_download", verify)
    monkeypatch.setattr(donor, "_read_archive", reader)
    loaded = donor.load_acs_2024_housing_donor(source, chunksize=1)
    assert loaded.table.loc["r1", TARGET] == 7200.0
    assert loaded.table.loc["r1", "employment_income_before_lsr"] == pytest.approx(
        110.0
    )
    assert all(
        path not in (source.household_zip, source.person_zip) for path in parsed_paths
    )
    assert all(not path.exists() for path in parsed_paths)
    for name, (size, sha) in identities.items():
        assert loaded.provenance["archives"][name] == {
            "size_bytes": size,
            "sha256": sha,
        }


def test_snapshot_removed_on_parser_failure(tmp_path, monkeypatch):
    source = _authenticated_source(tmp_path, monkeypatch)
    paths = []

    def broken(path, **kwargs):
        paths.append(path)
        assert path not in (source.household_zip, source.person_zip)
        raise ValueError("invented parse failure")

    monkeypatch.setattr(donor, "_read_archive", broken)
    with pytest.raises(ValueError, match="invented parse failure"):
        donor.load_acs_2024_housing_donor(source)
    assert paths and all(not path.exists() for path in paths)


def test_snapshot_removed_on_pin_failure(tmp_path, monkeypatch):
    source = _authenticated_source(tmp_path, monkeypatch)
    h, p = _tables()
    h.loc[h.SERIALNO == "r1", "RNTP"] = 600
    _source(tmp_path, h, p)  # Change the source after fixing the expected pins.
    original_identity = donor._file_identity
    paths = []

    def traced_identity(path):
        paths.append(path)
        return original_identity(path)

    monkeypatch.setattr(donor, "_file_identity", traced_identity)
    with pytest.raises(ValueError, match="sha-256 verification failed"):
        donor.load_acs_2024_housing_donor(source)
    assert paths and all(not path.exists() for path in paths)
    assert source.household_zip.exists() and source.person_zip.exists()


@pytest.mark.parametrize("stage", ["donor", "training"])
def test_consumption_uses_exact_copy_that_was_verified(monkeypatch, stage):
    loaded = donor.project_synthetic_acs_2024_housing_donor(*_tables())
    prepared = donor.prepare_acs_2024_reported_rent(loaded)
    exposed = loaded.table if stage == "donor" else prepared.training
    original_identity = donor.native_table_identity

    def mutate_exposed_after_copy(table):
        exposed.loc["r1", TARGET] = 1.0
        return original_identity(table)

    monkeypatch.setattr(donor, "native_table_identity", mutate_exposed_after_copy)
    consumed = (
        donor.prepare_acs_2024_reported_rent(loaded).training
        if stage == "donor"
        else prepared.verified_training()
    )
    assert consumed.loc["r1", TARGET] == 7200.0
    assert exposed.loc["r1", TARGET] == 1.0


def test_sealed_identity_and_explicit_schema_authorities():
    projected = donor.project_synthetic_acs_2024_housing_donor(*_tables())
    prepared = donor.prepare_acs_2024_reported_rent(projected)
    assert len(projected.provenance["projection_identity"]["sha256"]) == 64
    assert len(prepared.provenance["training_identity"]["sha256"]) == 64
    assert len(prepared.provenance["donor_receipt_sha256"]) == 64
    schema = prepared.provenance["feature_schema"]
    assert schema["unknown_category_policy"] == "refuse"
    assert (
        schema["category_level_source"]
        == "rent fit rows after archived joint target sample"
    )
    authority = projected.provenance["rent_concept_authority"]
    assert (
        authority["sha256"]
        == "6e0c5b1a5d3ebf7e300e583ac6424cbd5f09de50d69e7ef03aa6d070d9c4a7a8"
    )
    assert "publisher" in authority["rule"]
    assert "5000" in projected.provenance["sampling"]


@pytest.mark.parametrize("axis", ["index", "columns"])
def test_verified_training_owns_axis_storage_after_return(axis):
    prepared = donor.prepare_acs_2024_reported_rent(
        donor.project_synthetic_acs_2024_housing_donor(*_tables())
    )
    verified = prepared.verified_training()
    expected = prepared.provenance["training_identity"]
    getattr(prepared.training, axis).array[0] = "changed_after_verification"
    assert donor.native_table_identity(verified) == expected


@pytest.mark.parametrize("stage", ["donor", "training"])
@pytest.mark.parametrize("axis", ["index", "columns"])
def test_consumed_axes_stay_owned_after_identity_check(monkeypatch, stage, axis):
    loaded = donor.project_synthetic_acs_2024_housing_donor(*_tables())
    prepared = donor.prepare_acs_2024_reported_rent(loaded)
    expected = prepared.training.copy(deep=True)
    expected.index = expected.index.copy(deep=True)
    expected.columns = expected.columns.copy(deep=True)
    exposed = loaded.table if stage == "donor" else prepared.training
    original_identity = donor.native_table_identity

    def mutate_after_hash(table):
        identity = original_identity(table)
        getattr(exposed, axis).array[0] = "changed_after_identity_check"
        return identity

    monkeypatch.setattr(donor, "native_table_identity", mutate_after_hash)
    consumed = (
        donor.prepare_acs_2024_reported_rent(loaded).training
        if stage == "donor"
        else prepared.verified_training()
    )
    pd.testing.assert_frame_equal(consumed, expected, check_exact=True)


def test_native_identity_string_storage_index_and_schema():
    from microcosm.build.table_identity import native_table_identity

    baseline = donor.project_synthetic_acs_2024_housing_donor(*_tables()).table
    expected = native_table_identity(baseline)
    assert native_table_identity(baseline.copy(deep=True)) == expected
    for column in ("source_household_id", None):
        changed = baseline.copy()
        if column:
            changed[column] = changed[column].astype(pd.StringDtype(storage="pyarrow"))
        else:
            changed.index = changed.index.astype(pd.StringDtype(storage="pyarrow"))
        assert native_table_identity(changed) != expected
    unsupported = baseline.copy()
    unsupported["age"] = unsupported["age"].astype("category")
    with pytest.raises(TypeError, match="dtype"):
        native_table_identity(unsupported)


def test_snapshot_copy_uses_bounded_reads(tmp_path, monkeypatch):
    source = _authenticated_source(tmp_path, monkeypatch)
    original_open = Path.open
    observed = []

    class BoundedReader:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            observed.append(size)
            return self.file.read(size)

    def checked_open(path, mode="r", *args, **kwargs):
        opened = original_open(path, mode, *args, **kwargs)
        if path in (source.household_zip, source.person_zip) and mode == "rb":
            return BoundedReader(opened)
        return opened

    monkeypatch.setattr(Path, "open", checked_open)
    loaded = donor.load_acs_2024_housing_donor(source)
    assert observed and loaded.table.loc["r1", TARGET] == 7200.0
