"""Invented source bytes only; no Census source or genuine producer execution."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_source as legacy
from microcosm.build.us_runtime import asec_person_income_source as restoration


def _fixtures():
    path = Path(__file__).with_name("test_us_asec_current_money_source.py")
    spec = importlib.util.spec_from_file_location("ptotval_invented_parent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invented_sources(tmp_path, monkeypatch, *, missing=True, housing=False):
    original_writer = pd.DataFrame.to_hdf

    def household_writer(table, path, *args, **kwargs):
        if housing and Path(path).name.startswith("invented-cohort-"):
            year = int(Path(path).stem.rsplit("-", 1)[1])
            table = table.assign(
                H_YEAR=year + 1, HPUBLIC=2, HLORENT=2, I_HPUBLI=0, I_HLOREN=0
            )
        return original_writer(table, path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_hdf", household_writer)
        parent, attachment, *_ = _fixtures().invented_checkpoints(tmp_path, monkeypatch)
    # Older cohorts lost the complete field; the incumbent 2024 child has a
    # source-encoded zero, while the adult has a signed total income loss.
    for path in (parent, attachment):
        loaded = load_frame_checkpoint(path)
        loaded.frame.person["PTOTVAL"] = (
            ([np.nan] * 4 + [-9999.0, 0.0]) if missing else [-9999.0, 0.0] * 3
        )
        if path == attachment:
            loaded.metadata["parent_checkpoint_sha256"] = money._sha(
                parent.read_bytes()
            )
            loaded.metadata["household_observations"]["input_checkpoint_sha256"] = (
                money._sha(parent.read_bytes())
            )
        write_frame_checkpoint(path, loaded.frame, metadata=loaded.metadata)
    monkeypatch.setattr(
        legacy,
        "_SOURCE_PINS",
        (
            money._sha(parent.read_bytes()),
            money._sha(attachment.read_bytes()),
            legacy._SOURCE_PINS[2],
        ),
    )
    paths = {}
    pins = []
    for (
        year,
        member_name,
        archive_pin,
        _member_pin,
        _rows,
        _size,
    ) in restoration._MEMBER_PINS:
        path = tmp_path / member_name
        offset = (year - 2022) * 2
        frame = pd.DataFrame(
            {
                "PERIDNUM": [str(offset + i).zfill(22) for i in (1, 2)],
                "PH_SEQ": [7, 7],
                "A_LINENO": [1, 2],
                "A_AGE": [55, 14],
                "PTOTVAL": [-9999, 0],
            }
        )
        frame.iloc[::-1].to_csv(path, index=False)  # Must join by keys, not position.
        paths[year] = path
        pins.append(
            (
                year,
                member_name,
                archive_pin,
                money._sha(path.read_bytes()),
                2,
                path.stat().st_size,
            )
        )
    monkeypatch.setattr(restoration, "_MEMBER_PINS", tuple(pins))
    return parent, attachment, paths


def test_exact_restoration_preserves_legacy_and_appends_observed_signed_income(
    tmp_path, monkeypatch
):
    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    old = legacy.load_authenticated_current_money_source(parent, attachment)
    with pytest.raises(
        money.MoneyRefusalError, match="PTOTVAL: MISSING_REQUIRED_AMOUNT"
    ):
        old.ready()
    original_signature = legacy._frame_signature(old.frame)
    output = tmp_path / "restored"
    receipt = restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    checked = restoration.verify_asec_person_income_source(
        parent, attachment, output / restoration.CHECKPOINT_FILENAME, member_paths=paths
    )
    assert checked.frame.person.asec_PTOTVAL.tolist() == [-9999, 0] * 3
    assert checked.frame.person.asec_PTOTVAL.dtype == np.dtype("int64")
    assert checked.frame.person.PTOTVAL.isna().sum() == 4
    assert receipt["parent_frame_sha256"] == original_signature
    for entity in old.frame.entities:
        actual = checked.frame.table(entity)
        if entity == "person":
            actual = actual.drop(columns="asec_PTOTVAL")
        pd.testing.assert_frame_equal(actual, old.frame.table(entity), check_exact=True)
    assert [r["joined_rows"] for r in receipt["sources"]] == [2, 2, 2]
    assert receipt["sources"][2]["incumbent_compared_rows"] == 2
    assert receipt["sources"][0]["incumbent_compared_rows"] == 0
    assert legacy._frame_signature(old.frame) == original_signature
    with pytest.raises(FileExistsError):
        restoration.restore_asec_person_income_source(
            parent, attachment, member_paths=paths, output_dir=output
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_amount",
        "fraction",
        "range",
        "duplicate_key",
        "unknown_key",
        "age",
        "household",
        "line",
        "incumbent",
        "duplicate_header",
    ],
)
def test_authenticated_source_discrepancies_refuse_without_values(
    tmp_path, monkeypatch, mutation
):
    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    path = paths[2024]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if mutation == "duplicate_header":
        path.write_text(path.read_text().replace("PTOTVAL\n", "A_AGE\n"))
    else:
        updates = {
            "missing_amount": ("PTOTVAL", ""),
            "fraction": ("PTOTVAL", "1.5"),
            "range": ("PTOTVAL", "100000000"),
            "unknown_key": ("PERIDNUM", "9" * 22),
            "age": ("A_AGE", "13"),
            "household": ("PH_SEQ", "8"),
            "line": ("A_LINENO", "3"),
            "incumbent": ("PTOTVAL", "4"),
        }
        if mutation == "duplicate_key":
            frame.iloc[1] = frame.iloc[0]
        else:
            column, value = updates[mutation]
            frame.loc[0, column] = value
        frame.to_csv(path, index=False)
    # Private invented pin update lets the parser/join see each contradictory
    # authenticated fixture, rather than stopping only at a checksum failure.
    pins = tuple(
        (
            y,
            n,
            a,
            money._sha(path.read_bytes()) if y == 2024 else p,
            r,
            path.stat().st_size if y == 2024 else s,
        )
        for y, n, a, p, r, s in restoration._MEMBER_PINS
    )
    monkeypatch.setattr(restoration, "_MEMBER_PINS", pins)
    with pytest.raises(money.MoneyRefusalError) as error:
        restoration.restore_asec_person_income_source(
            parent, attachment, member_paths=paths, output_dir=tmp_path / "refused"
        )
    assert "9999" not in str(error.value) and "999999" not in str(error.value)
    assert not (tmp_path / "refused").exists()


def test_member_bytes_and_unclosed_path_selection_refuse(tmp_path, monkeypatch):
    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    with paths[2022].open("a") as handle:
        handle.write("changed\n")
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_BYTES"):
        restoration.restore_asec_person_income_source(
            parent, attachment, member_paths=paths, output_dir=tmp_path / "bad"
        )
    with pytest.raises(money.MoneyRefusalError, match="MEMBER_PATHS"):
        restoration.restore_asec_person_income_source(
            parent,
            attachment,
            member_paths={2022: paths[2022]},
            output_dir=tmp_path / "bad",
        )
    with pytest.raises(TypeError):
        restoration.restore_asec_person_income_source(
            parent,
            attachment,
            member_paths=paths,
            output_dir=tmp_path / "bad",
            expected_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "mutation", ["observation", "legacy", "axis", "weight", "receipt"]
)
def test_serialized_attachment_requires_complete_reconstruction(
    tmp_path, monkeypatch, mutation
):
    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    output = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    path = output / restoration.CHECKPOINT_FILENAME
    value = load_frame_checkpoint(path)
    if mutation == "observation":
        value.frame.person.loc[:, "asec_PTOTVAL"] = 0
    elif mutation == "legacy":
        value.frame.person.loc[:, "WSAL_VAL"] = 0.0
    elif mutation == "axis":
        value.frame.person.index.name = "changed"
    elif mutation == "weight":
        weights = value.frame.weights_for("household").values
        weights.setflags(write=True)
        weights[0] = 99.0
    else:
        value.metadata["person_income_observations"]["reader"] = "changed"
    write_frame_checkpoint(path, value.frame, metadata=value.metadata)
    with pytest.raises(money.MoneyRefusalError, match="RESTORATION"):
        restoration.verify_asec_person_income_source(
            parent, attachment, path, member_paths=paths
        )


def test_restored_money_has_field_scoped_encoded_zero_provenance(tmp_path, monkeypatch):
    import json

    from microcosm.build.us_runtime import _asec_current_money_codec as codec

    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    output = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    value = restoration.load_authenticated_restored_current_money_source(
        parent, attachment, output / restoration.CHECKPOINT_FILENAME, member_paths=paths
    )
    ready = value.ready()
    assert ready.field("PTOTVAL").zero_origin.tolist() == [0, 2] * 3
    assert ready.field("RNT_VAL").zero_origin.tolist() == [1] * 6
    assert ready.field("ANN_VAL").zero_origin.tolist() == [0] * 6
    assert json.loads(ready.header)["schema_version"] == 2
    assert json.loads(ready.header)["field_zero_origin_policy"] == {
        "PTOTVAL": "authenticated_census_csv_encoded_zero_v1"
    }
    assert json.loads(value.source.identity)["field_source_columns"] == {
        "PTOTVAL": "asec_PTOTVAL"
    }
    decoded = codec.decode_current_money(
        codec.encode_current_money(ready), ready.bindings, expected=ready
    )
    assert decoded.fields == ready.fields
    # Header and full evidence stay bound through price restatement and codec.
    assert ready.field("PTOTVAL").amounts[-2:].tolist() == [-9999, 0]
    assert value.frame.person.PTOTVAL.isna().sum() == 4


@pytest.mark.parametrize("field,code", [("PTOTVAL", 0), ("PTOTVAL", 1), ("RNT_VAL", 2)])
def test_forged_field_origin_cannot_encode_or_issue_readiness(
    tmp_path, monkeypatch, field, code
):
    from dataclasses import replace

    from microcosm.build.us_runtime import _asec_current_money_codec as codec

    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    output = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    ready = restoration.load_authenticated_restored_current_money_source(
        parent, attachment, output / restoration.CHECKPOINT_FILENAME, member_paths=paths
    ).ready()
    fields = list(ready.fields)
    index = money.FIELDS.index(field)
    origins = fields[index].zero_origin.copy()
    origins[1] = code
    fields[index] = replace(fields[index], zero_origin_bytes=origins.tobytes())
    forged = money.RestatedAsecMoney(
        ready.bindings, tuple(fields), _token=money._STATE_TOKEN
    )
    with pytest.raises(money.MoneyRefusalError, match="ZERO_ORIGIN_ENCODING"):
        money.require_complete_current_money(
            forged, ready.bindings.spec, production=True
        )
    forged_ready = money.ReadyCurrentMoney(
        ready.bindings, tuple(fields), _token=money._STATE_TOKEN
    )
    with pytest.raises(money.MoneyRefusalError, match="ZERO_ORIGIN_ENCODING"):
        codec.encode_current_money(forged_ready)


def test_private_member_and_attachment_snapshots_are_used(tmp_path, monkeypatch):
    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    output = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    candidate = output / restoration.CHECKPOINT_FILENAME
    real_read = restoration._read_member
    staged_members = []

    def read_private(path, **kwargs):
        assert path not in paths.values()
        staged_members.append(path)
        return real_read(path, **kwargs)

    monkeypatch.setattr(restoration, "_read_member", read_private)
    real_snapshot = restoration._candidate_snapshot

    def snapshot_then_replace(path, destination, **kwargs):
        assert Path(destination) != candidate
        digest = real_snapshot(path, destination, **kwargs)
        candidate.write_bytes(b"changed after private snapshot")
        return digest

    monkeypatch.setattr(restoration, "_candidate_snapshot", snapshot_then_replace)
    checked = restoration.verify_asec_person_income_source(
        parent, attachment, candidate, member_paths=paths
    )
    assert checked.frame.person.asec_PTOTVAL.tolist() == [-9999, 0] * 3
    assert all(not p.exists() for p in staged_members)
    with pytest.raises(money.MoneyRefusalError):
        restoration.verify_asec_person_income_source(
            parent, attachment, candidate, member_paths=paths
        )


def test_regenerated_housing_and_money_share_restored_identity_and_crossed_sources_refuse(
    tmp_path, monkeypatch
):
    from microcosm.build.us_runtime import asec_current_money_units as units
    from microcosm.build.us_runtime import asec_housing_status as housing
    from microcosm.build.us_runtime import asec_housing_status_source as housing_source

    parent, attachment, paths = invented_sources(
        tmp_path, monkeypatch, missing=False, housing=True
    )
    old = legacy.load_authenticated_current_money_source(parent, attachment)
    output = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    new = restoration.load_authenticated_restored_current_money_source(
        parent, attachment, output / restoration.CHECKPOINT_FILENAME, member_paths=paths
    )
    cohort_paths = {y: tmp_path / f"invented-cohort-{y}.h5" for y in (2022, 2023, 2024)}
    old_housing = housing_source.load_authenticated_housing_status(
        old, cohort_paths=cohort_paths
    )
    new_housing = housing_source.load_authenticated_housing_status(
        new, cohort_paths=cohort_paths
    )
    for name in housing.RAW_COLUMNS:
        assert np.array_equal(old_housing.array(name), new_housing.array(name))
    assert (
        old_housing.header_data["source"]["identity"]
        != new_housing.header_data["source"]["identity"]
    )
    old_tax = units.reconstruct_current_money_tax_units(old, old.ready())
    new_tax = units.reconstruct_current_money_tax_units(new, new.ready())
    composed = housing_source.attach_housing_status(new_tax, new_housing)
    composed.validate()
    for tax, status in ((old_tax, new_housing), (new_tax, old_housing)):
        with pytest.raises(
            housing.HousingStatusRefusalError, match="STATUS_SOURCE_BINDING"
        ):
            housing_source.attach_housing_status(tax, status)
    assert old.ready().field("PTOTVAL").zero_origin.tolist() == [0, 1] * 3
    assert new.ready().field("PTOTVAL").zero_origin.tolist() == [0, 2] * 3


def test_restored_frame_axes_values_and_weights_are_owned(tmp_path, monkeypatch):
    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    output = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    checked = restoration.verify_asec_person_income_source(
        parent, attachment, output / restoration.CHECKPOINT_FILENAME, member_paths=paths
    )
    original = legacy._frame_signature(checked.parent_source.frame)
    assert not np.shares_memory(
        checked.frame.person.index.to_numpy(),
        checked.parent_source.frame.person.index.to_numpy(),
    )
    assert not np.shares_memory(
        checked.frame.person.columns.to_numpy(),
        checked.parent_source.frame.person.columns.to_numpy(),
    )
    assert not np.shares_memory(
        checked.frame.weights_for("household").values,
        checked.parent_source.frame.weights_for("household").values,
    )
    checked.frame.person.loc[:, "WSAL_VAL"] = 3.0
    checked.frame.person.index.name = "child-only"
    assert legacy._frame_signature(checked.parent_source.frame) == original
    checked.parent_source.validate()


@pytest.mark.parametrize("mutation", ["field", "policy", "kind", "version", "digest"])
def test_restored_source_schema_and_policy_are_closed(tmp_path, monkeypatch, mutation):
    import json

    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    output = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    value = restoration.load_authenticated_restored_current_money_source(
        parent, attachment, output / restoration.CHECKPOINT_FILENAME, member_paths=paths
    )
    evidence = json.loads(value.source.identity)
    if mutation == "field":
        evidence["field_source_columns"] = {"PTOTVAL": "PTOTVAL"}
    elif mutation == "policy":
        evidence["field_zero_origin_policy"] = {"RNT_VAL": money.ENCODED_ZERO_POLICY}
    elif mutation == "kind":
        evidence["source_kind"] = "asec_v4_with_household_observations_v1"
    elif mutation == "version":
        evidence["schema_version"] = 1
    else:
        evidence["person_income_attachment_sha256"] = "untrusted"
    with pytest.raises(money.MoneyRefusalError):
        money.AuthenticatedAsecSource(money._json(evidence), _token=money._SOURCE_TOKEN)


@pytest.mark.parametrize(
    "mutation",
    [
        "canonical",
        "external_numeric",
        "external_metadata",
        "external_root",
        "virtual_numeric",
        "external_raw",
        "soft_link",
        "alias",
        "cycle",
        "attribute",
        "vlen",
        "sparse_metadata",
        "same_size_wrong_bytes",
        "truncated",
    ],
)
def test_candidate_hdf_is_never_decoded(tmp_path, monkeypatch, mutation):
    import json
    import shutil

    import h5py

    from microcosm.build import frame_checkpoint as checkpoint

    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    output = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    candidate = output / restoration.CHECKPOINT_FILENAME
    original_bytes = candidate.read_bytes()
    donor = tmp_path / "outside-candidate.h5"
    shutil.copyfile(candidate, donor)
    root_path = f"/{checkpoint._ROOT}"
    metadata_path = f"{root_path}/{checkpoint._METADATA_DATASET}"
    with h5py.File(candidate, "r") as h5:
        metadata = json.loads(bytes(h5[metadata_path][:]))
        table_index = next(
            i for i, t in enumerate(metadata["tables"]) if t["name"] == "person"
        )
        column_index = next(
            i
            for i, c in enumerate(metadata["tables"][table_index]["columns"])
            if c["name"] == "WSAL_VAL"
        )
        numeric_path = (
            f"{root_path}/tables/t{table_index:05d}/columns/c{column_index:05d}/values"
        )
        values = h5[numeric_path][:]
    if mutation == "same_size_wrong_bytes":
        changed = bytearray(original_bytes)
        changed[-1] ^= 1
        candidate.write_bytes(changed)
    elif mutation == "truncated":
        candidate.write_bytes(original_bytes[:-1])
    elif mutation != "canonical":
        with h5py.File(candidate, "r+") as h5:
            if mutation.startswith("external_") and mutation != "external_raw":
                target = {
                    "external_numeric": numeric_path,
                    "external_metadata": metadata_path,
                    "external_root": root_path,
                }[mutation]
                del h5[target]
                h5[target] = h5py.ExternalLink(str(donor), target)
            elif mutation == "virtual_numeric":
                layout = h5py.VirtualLayout(shape=values.shape, dtype=values.dtype)
                layout[:] = h5py.VirtualSource(
                    str(donor), numeric_path, shape=values.shape
                )
                del h5[numeric_path]
                h5.create_virtual_dataset(numeric_path, layout)
            elif mutation == "external_raw":
                del h5[numeric_path]
                h5.create_dataset(
                    numeric_path,
                    data=values,
                    external=[(str(tmp_path / "outside.raw"), 0, h5py.h5f.UNLIMITED)],
                )
            elif mutation == "soft_link":
                h5["copy"] = h5[numeric_path]
                del h5[numeric_path]
                h5[numeric_path] = h5py.SoftLink("/copy")
            elif mutation == "alias":
                h5["alias"] = h5[root_path]
            elif mutation == "cycle":
                h5[root_path]["cycle"] = h5[root_path]
            elif mutation == "attribute":
                h5.attrs["unexpected"] = "not canonical"
            elif mutation == "vlen":
                del h5[metadata_path]
                h5.create_dataset(metadata_path, (1,), dtype=h5py.string_dtype())
            elif mutation == "sparse_metadata":
                del h5[metadata_path]
                h5.create_dataset(metadata_path, shape=(1_000_000_000,), dtype="u1")
        assert candidate.stat().st_size < 1_000_000

    real_open = h5py.File
    candidate_opens = []

    def forbid_candidate_open(name, mode="r", *args, **kwargs):
        if mode == "r" and (
            Path(name).name == restoration.CHECKPOINT_FILENAME or Path(name) == donor
        ):
            candidate_opens.append(str(name))
            raise AssertionError("Candidate or external HDF must never be opened")
        return real_open(name, mode, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", forbid_candidate_open)
    if mutation == "canonical":
        checked = restoration.verify_asec_person_income_source(
            parent, attachment, candidate, member_paths=paths
        )
        assert checked.attachment_sha256 == money._sha(original_bytes)
        assert checked.frame.person.asec_PTOTVAL.tolist() == [-9999, 0] * 3
    else:
        with pytest.raises(money.MoneyRefusalError, match="RESTORATION"):
            restoration.verify_asec_person_income_source(
                parent, attachment, candidate, member_paths=paths
            )
    assert candidate_opens == []


@pytest.mark.parametrize("entrypoint", ["producer", "verifier"])
@pytest.mark.parametrize("mapping", [None, {}, {True: "x"}, {2022: 1}])
def test_invalid_member_mapping_refuses_before_parent_load(
    tmp_path, monkeypatch, entrypoint, mapping
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Malformed mapping must refuse before P/H load")

    monkeypatch.setattr(legacy, "load_authenticated_current_money_source", forbidden)
    with pytest.raises(money.MoneyRefusalError, match="MEMBER_PATHS"):
        if entrypoint == "producer":
            restoration.restore_asec_person_income_source(
                "unused-P", "unused-H", member_paths=mapping, output_dir=tmp_path / "T"
            )
        else:
            restoration.verify_asec_person_income_source(
                "unused-P", "unused-H", "unused-T", member_paths=mapping
            )


@pytest.mark.parametrize("column", ["PH_SEQ", "A_LINENO", "A_AGE", "PTOTVAL"])
def test_malformed_member_integer_identifies_actual_field(tmp_path, column):
    path = tmp_path / "invented.csv"
    row = dict(PERIDNUM="0" * 22, PH_SEQ="1", A_LINENO="1", A_AGE="15", PTOTVAL="0")
    row[column] = "invalid"
    pd.DataFrame([row]).to_csv(path, index=False)
    reason = (
        "PTOTVAL: MEMBER_INTEGER"
        if column == "PTOTVAL"
        else f"contract: MEMBER_INTEGER_{column}"
    )
    with pytest.raises(money.MoneyRefusalError, match=reason):
        restoration._read_member(path, rows=1, size=path.stat().st_size)


def test_year_swapped_members_refuse(tmp_path, monkeypatch):
    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    paths[2022], paths[2023] = paths[2023], paths[2022]
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_BYTES"):
        restoration.restore_asec_person_income_source(
            parent, attachment, member_paths=paths, output_dir=tmp_path / "refused"
        )


def test_missing_2024_incumbent_refuses(tmp_path, monkeypatch):
    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    for path in (parent, attachment):
        loaded = load_frame_checkpoint(path)
        loaded.frame.person.iloc[4, loaded.frame.person.columns.get_loc("PTOTVAL")] = (
            np.nan
        )
        if path == attachment:
            parent_sha = money._sha(parent.read_bytes())
            loaded.metadata["parent_checkpoint_sha256"] = parent_sha
            loaded.metadata["household_observations"]["input_checkpoint_sha256"] = (
                parent_sha
            )
        write_frame_checkpoint(path, loaded.frame, metadata=loaded.metadata)
    monkeypatch.setattr(
        legacy,
        "_SOURCE_PINS",
        (
            money._sha(parent.read_bytes()),
            money._sha(attachment.read_bytes()),
            legacy._SOURCE_PINS[2],
        ),
    )
    with pytest.raises(
        money.MoneyRefusalError, match="RESTORATION_INCUMBENT_INCOMPLETE"
    ):
        restoration.restore_asec_person_income_source(
            parent, attachment, member_paths=paths, output_dir=tmp_path / "refused"
        )


def test_canonical_bytes_survive_separate_process_and_wall_clock(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import sys
    import time

    parent, attachment, paths = invented_sources(tmp_path, monkeypatch)
    first = tmp_path / "first"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=first
    )
    second = tmp_path / "second"
    spec = tmp_path / "invented-private-pins.json"
    spec.write_text(
        json.dumps(
            {
                "parent": str(parent),
                "attachment": str(attachment),
                "members": {y: str(p) for y, p in paths.items()},
                "source_pins": legacy._SOURCE_PINS,
                "member_pins": restoration._MEMBER_PINS,
                "output": str(second),
                "after": time.time() + 1.1,
            }
        )
    )
    script = """
import json, sys, time
from pathlib import Path
from microcosm.build.us_runtime import asec_current_money_source as legacy
from microcosm.build.us_runtime import asec_person_income_source as restoration
spec = json.loads(Path(sys.argv[1]).read_text())
# Invented test-only pins; public APIs still accept no pin authority.
pins = spec['source_pins']
legacy._SOURCE_PINS = (pins[0], pins[1], tuple(tuple(p) for p in pins[2]))
restoration._MEMBER_PINS = tuple(tuple(p) for p in spec['member_pins'])
while time.time() < spec['after']:
    time.sleep(0.05)
restoration.restore_asec_person_income_source(
    spec['parent'], spec['attachment'],
    member_paths={int(y): p for y, p in spec['members'].items()},
    output_dir=spec['output'])
"""
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        str(p) for p in sorted((repo / "packages").glob("*/src"))
    )
    subprocess.run(
        [sys.executable, "-c", script, str(spec)],
        env=env,
        check=True,
        capture_output=True,
        timeout=120,
    )
    canonical = first / restoration.CHECKPOINT_FILENAME
    independently_written = second / restoration.CHECKPOINT_FILENAME
    assert canonical.read_bytes() == independently_written.read_bytes()
    checked = restoration.verify_asec_person_income_source(
        parent, attachment, independently_written, member_paths=paths
    )
    assert checked.attachment_sha256 == money._sha(canonical.read_bytes())
