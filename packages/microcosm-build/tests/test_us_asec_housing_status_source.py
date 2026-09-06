"""Test-only pin substitutions in invented files, never Census authentication."""

import hashlib
import importlib.util
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime import asec_current_money_source as money_source
from microcosm.build.us_runtime import asec_current_money_units as units
from microcosm.build.us_runtime import asec_household_observations as reader
from microcosm.build.us_runtime import asec_housing_status as housing
from microcosm.build.us_runtime import asec_housing_status_source as source
from microcosm.frame import MassChangeRecord


def invented_source(
    tmp_path,
    monkeypatch,
    *,
    change=None,
    object_block=False,
    history=False,
    after_write=None,
):
    """Reuse the real tiny source fixture; enrich only its invented household CSV analogs."""
    pytest.importorskip("microunit")
    pytest.importorskip("tables")
    spec = importlib.util.spec_from_file_location(
        "housing_money_fixture",
        Path(__file__).with_name("test_us_asec_current_money_source.py"),
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    if history:
        real_frame = fixture.Frame

        def frame_with_history(*args, **kwargs):
            return real_frame(
                *args,
                **kwargs,
                mass_log=(
                    MassChangeRecord(
                        "household", 3.0, 6.0, 2.0, "Invented fixture history"
                    ),
                ),
                metadata={"invented": {"history": ("preserve",)}},
            )

        fixture.Frame = frame_with_history
    original = pd.DataFrame.to_hdf

    def write_invented_housing(table, path, *args, **kwargs):
        if Path(path).name.startswith("invented-cohort-"):
            year = int(Path(path).stem.rsplit("-", 1)[1])
            table = table.copy(deep=True)
            table["H_YEAR"] = year + 1
            table["HPUBLIC"] = {2022: 1, 2023: 2, 2024: 2}[year]
            table["HLORENT"] = {2022: 0, 2023: 1, 2024: 2}[year]
            table["I_HPUBLI"] = 0
            table["I_HLOREN"] = int(year == 2023)
            if change is not None:
                table = change(table, year)
            if object_block:
                table["NOT_TO_BE_UNPICKLED"] = [{"not": "source values"}] * len(table)
                with pytest.warns(pd.errors.PerformanceWarning):
                    return original(table, path, *args, **kwargs)
            written = original(table, path, *args, **kwargs)
            if after_write is not None:
                # Damage the invented bytes before anything pins their digest,
                # so the loader meets the corruption instead of a stale hash.
                after_write(Path(path))
            return written
        return original(table, path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_hdf", write_invented_housing)
        parent, attachment, *_ = fixture.invented_checkpoints(tmp_path, monkeypatch)
    loaded = money_source.load_authenticated_current_money_source(parent, attachment)
    paths = {
        year: tmp_path / f"invented-cohort-{year}.h5" for year in (2022, 2023, 2024)
    }
    return loaded, paths, parent, attachment


def test_actual_pinned_loader_preserves_order_context_raw_codes_and_period(
    tmp_path, monkeypatch
):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    assert type(status) is housing.AuthenticatedHousingStatus
    assert status.array("household_id").tolist() == [1, 2, 3]
    assert status.array("income_year").tolist() == [2022, 2023, 2024]
    assert status.array("survey_year").tolist() == [2023, 2024, 2025]
    assert status.array("status").tolist() == [1, 1, 2]
    assert status.array("lower_quality").tolist() == [0, 2, 1]
    assert status.header_data["source"]["identity"] == loaded.source.identity.decode()
    assert all(
        set(row)
        == {
            "income_year",
            "sha256",
            "source_rows",
            "joined_rows",
            "unreferenced_source_rows",
        }
        for row in status.header_data["source"]["joins"]
    )
    assert "status_counts" not in status.header_data
    status.validate()


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_any_wrong_fixed_cohort_bytes_refuse(tmp_path, monkeypatch, year):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    with paths[year].open("ab") as f:
        f.write(b"different")
    with pytest.raises(
        housing.HousingStatusRefusalError, match="SOURCE_BYTES_MISMATCH"
    ):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


def test_survey_year_and_context_are_observed_checks(tmp_path, monkeypatch):
    loaded, paths, *_ = invented_source(
        tmp_path,
        monkeypatch,
        change=lambda t, y: t.assign(H_YEAR=2023) if y == 2024 else t,
    )
    with pytest.raises(housing.HousingStatusRefusalError, match="SURVEY_YEAR"):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


def test_carried_context_must_equal_fresh_pinned_cohort_values(tmp_path, monkeypatch):
    _, paths, parent, attachment = invented_source(tmp_path, monkeypatch)
    attached = load_frame_checkpoint(attachment)
    table = attached.frame.table("household")
    table.loc[0, "asec_HRHTYPE"] = 9
    attached.metadata["household_observations"]["outputs"]["asec_HRHTYPE"]["sha256"] = (
        hashlib.sha256(table.asec_HRHTYPE.to_numpy(dtype="<i8").tobytes()).hexdigest()
    )
    write_frame_checkpoint(attachment, attached.frame, metadata=attached.metadata)
    old = money_source._SOURCE_PINS
    monkeypatch.setattr(
        money_source,
        "_SOURCE_PINS",
        (old[0], hashlib.sha256(attachment.read_bytes()).hexdigest(), old[2]),
    )
    loaded = money_source.load_authenticated_current_money_source(parent, attachment)
    with pytest.raises(housing.HousingStatusRefusalError, match="CARRIED_CONTEXT"):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


def test_private_cohort_snapshots_survive_path_swap_and_next_load_refuses(
    tmp_path, monkeypatch
):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    original = source._read_cohort
    seen = []

    def replace_original_then_read(staged):
        assert Path(staged) not in paths.values()
        paths[2022].write_bytes(b"valid-at-load path replaced after staging")
        seen.append(staged)
        return original(staged)

    monkeypatch.setattr(source, "_read_cohort", replace_original_then_read)
    result = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    assert result.array("HPUBLIC").tolist() == [1, 2, 2]
    assert all(not Path(p).exists() for p in seen)
    with pytest.raises(
        housing.HousingStatusRefusalError, match="SOURCE_BYTES_MISMATCH"
    ):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


def test_numeric_reader_never_opens_existing_object_values(tmp_path, monkeypatch):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch, object_block=True)
    original = reader._dataset

    def forbid_objects(group, name):
        value = original(group, name)
        assert value.dtype.kind != "O", "Opened an unrelated object value block"
        return value

    monkeypatch.setattr(reader, "_dataset", forbid_objects)
    assert source.load_authenticated_housing_status(loaded, cohort_paths=paths).array(
        "status"
    ).tolist() == [1, 1, 2]


def test_authenticated_replay_rechecksummed_raw_or_classification_changes_refuse(
    tmp_path, monkeypatch
):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    payload = housing.encode_housing_status(status)
    replay = housing.decode_housing_status(payload, expected=status)
    assert type(replay) is housing.AuthenticatedHousingStatus
    assert replay.content_sha256 == status.content_sha256
    size = struct.unpack_from("<I", payload, len(housing.MAGIC))[0]
    start = len(housing.MAGIC) + 4 + size
    for name in ("HPUBLIC", "status", "I_HLOREN_zero_origin"):
        offset = start + sum(
            len(buf) for buf in status.buffers[: housing.COLUMNS.index(name)]
        )
        changed = bytearray(payload)
        changed[offset] ^= 1
        changed[-32:] = hashlib.sha256(changed[:-32]).digest()
        with pytest.raises(housing.HousingStatusRefusalError, match="EXPECTED_CONTENT"):
            housing.decode_housing_status(bytes(changed), expected=status)


def test_actual_tax_composition_preserves_parent_money_and_checkpoint_signature(
    tmp_path, monkeypatch
):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch, history=True)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    ready = loaded.ready()
    parent = units.reconstruct_current_money_tax_units(loaded, ready)
    parent_digest = money_source._frame_signature(parent.frame)
    parent_receipt = parent.receipt
    composed = source.attach_housing_status(parent, status)
    assert composed.money is parent.money is ready
    assert composed.tax_result is parent
    assert composed.frame.mass_log == parent.frame.mass_log == loaded.frame.mass_log
    assert len(composed.frame.mass_log) == 1
    assert composed.frame.metadata == parent.frame.metadata == loaded.frame.metadata
    assert composed.frame.schema is parent.frame.schema
    pd.testing.assert_series_equal(
        composed.frame.strata, parent.frame.strata, check_exact=True
    )
    for entity in parent.frame.weighted_entities:
        assert (
            composed.frame.weights_for(entity).kind
            is parent.frame.weights_for(entity).kind
        )
        assert (
            composed.frame.weights_for(entity).values.tobytes()
            == parent.frame.weights_for(entity).values.tobytes()
        )
        assert not np.shares_memory(
            composed.frame.weights_for(entity).values,
            parent.frame.weights_for(entity).values,
        )
    assert composed.receipt["housing_content_sha256"] == status.content_sha256
    assert composed.receipt["release_eligible"] is False
    for entity in parent.frame.entities:
        actual = composed.frame.table(entity)
        if entity == "household":
            actual = actual.drop(columns=list(source.ATTACHED_COLUMNS))
        pd.testing.assert_frame_equal(
            actual, parent.frame.table(entity), check_exact=True
        )
    for column in source.ATTACHED_COLUMNS:
        assert composed.frame.table("household")[column].dtype == np.dtype("uint8")
    assert parent.receipt == parent_receipt
    assert money_source._frame_signature(parent.frame) == parent_digest
    parent.validate()
    composed.validate()
    path = tmp_path / "invented-composed.h5"
    write_frame_checkpoint(path, composed.frame)
    replay = load_frame_checkpoint(path).frame
    assert money_source._frame_signature(replay) == money_source._frame_signature(
        composed.frame
    )
    source.validate_attached_frame(replay, parent=parent, status=status)


def test_synthetic_status_cannot_attach_to_actual_tax_result(tmp_path, monkeypatch):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    observed = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    raw = pd.DataFrame({name: observed.array(name) for name in housing.RAW_COLUMNS})
    synthetic = housing.classify_synthetic_housing_status(raw)
    parent = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
    with pytest.raises(housing.HousingStatusRefusalError, match="AUTHENTICATED_STATUS"):
        source.attach_housing_status(parent, synthetic)


def test_new_status_view_cannot_mutate_parent_axes_or_weights(tmp_path, monkeypatch):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    parent = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
    composed = source.attach_housing_status(parent, status)
    composed.frame.table("family").columns.array[0] = "forged_family_id"
    assert parent.frame.table("family").columns.tolist() == ["family_id"]
    parent.validate()
    with pytest.raises(housing.HousingStatusRefusalError):
        composed.validate()


@pytest.mark.parametrize("mutation", ["range_index", "weights"])
def test_composed_storage_cannot_change_either_sealed_parent(
    tmp_path, monkeypatch, mutation
):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    parent = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
    # The source signature has already materialized pandas' RangeIndex cache.
    source_digest = money_source._frame_signature(loaded.frame)
    parent_digest = money_source._frame_signature(parent.frame)
    composed = source.attach_housing_status(parent, status)
    original_index = parent.frame.table("household").index
    copied_index = composed.frame.table("household").index
    assert type(copied_index) is type(original_index) is pd.RangeIndex
    pd.testing.assert_index_equal(copied_index, original_index, exact=True)
    if mutation == "range_index":
        copied_index.array[0] = 999
    else:
        values = composed.frame.weights_for("household").values
        values.setflags(write=True)
        values[0] = 999.0
    assert money_source._frame_signature(parent.frame) == parent_digest
    assert money_source._frame_signature(loaded.frame) == source_digest
    loaded.validate()
    parent.validate()
    with pytest.raises(housing.HousingStatusRefusalError):
        composed.validate()


def test_wrong_independently_issued_source_cannot_authenticate_replay(
    tmp_path, monkeypatch
):
    first = tmp_path / "first"
    first.mkdir()
    loaded, paths, *_ = invented_source(first, monkeypatch)
    original = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    payload = housing.encode_housing_status(original)
    second = tmp_path / "second"
    second.mkdir()
    other, other_paths, *_ = invented_source(
        second,
        monkeypatch,
        change=lambda t, y: t.assign(HPUBLIC=0, HLORENT=0) if y == 2022 else t,
    )
    expected = source.load_authenticated_housing_status(other, cohort_paths=other_paths)
    with pytest.raises(housing.HousingStatusRefusalError, match="EXPECTED_CONTENT"):
        housing.decode_housing_status(payload, expected=expected)


@pytest.mark.parametrize(
    "mutation", ["values", "dtype", "unrelated", "weights", "axis"]
)
def test_composed_mutation_refuses_and_cannot_change_parent(
    tmp_path, monkeypatch, mutation
):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    parent = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
    composed = source.attach_housing_status(parent, status)
    household = composed.frame.table("household")
    output = source.ATTACHED_COLUMNS[0]
    if mutation == "values":
        household.loc[0, output] = 4
    elif mutation == "dtype":
        household[output] = household[output].astype("int64")
    elif mutation == "unrelated":
        composed.frame.person.loc[8, "unrelated_raw_observation"] = 999
    elif mutation == "weights":
        weights = composed.frame.weights_for("household").values
        weights.setflags(write=True)
        weights[0] = 999.0
    else:
        composed.frame.person.index.array[0] = 999
    parent.validate()
    with pytest.raises(housing.HousingStatusRefusalError):
        composed.validate()


def test_invalid_parent_refuses_before_any_attachment(tmp_path, monkeypatch):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    parent = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
    parent.frame.table("family")["unexpected"] = 1
    with pytest.raises(housing.HousingStatusRefusalError):
        source.attach_housing_status(parent, status)
    assert not set(source.ATTACHED_COLUMNS) & set(parent.frame.table("household"))


def test_observed_year_must_share_native_key_numeric_block(tmp_path, monkeypatch):
    loaded, paths, *_ = invented_source(
        tmp_path,
        monkeypatch,
        change=lambda t, y: t.astype({"H_YEAR": "int32"}),
    )
    with pytest.raises(
        housing.HousingStatusRefusalError, match="SOURCE_CONTRACT_REFUSAL"
    ):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


@pytest.mark.parametrize("key", [2021, "2022"])
def test_cohort_mapping_cannot_substitute_periods(tmp_path, monkeypatch, key):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    paths[key] = paths.pop(2022)
    with pytest.raises(housing.HousingStatusRefusalError, match="COHORT_PATHS"):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


@pytest.mark.parametrize("code,value", [("HPUBLIC", 3), ("I_HLOREN", -1)])
def test_authenticated_invalid_domain_refuses_instead_of_publishing_status(
    tmp_path, monkeypatch, code, value
):
    loaded, paths, *_ = invented_source(
        tmp_path,
        monkeypatch,
        change=lambda t, y: t.assign(**{code: value}),
    )
    with pytest.raises(housing.HousingStatusRefusalError, match="CODE_DOMAIN"):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


def test_source_reader_and_frame_implementation_drift_invalidate_old_status(
    tmp_path, monkeypatch
):
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    original = money_source._verification_identity
    monkeypatch.setattr(money_source, "_verification_identity", lambda: "0" * 64)
    with pytest.raises(
        housing.HousingStatusRefusalError, match="IMPLEMENTATION_CHANGED"
    ):
        status.validate()
    monkeypatch.setattr(money_source, "_verification_identity", original)
    status.validate()


def _decoyed(table, year):
    """Put the scope's only real native key last, behind contradictory decoys."""
    decoys = pd.concat([table] * 3, ignore_index=True)
    decoys["H_SEQ"] = np.array([91, 92, 93], dtype=np.int64)
    decoys["HPUBLIC"] = np.array([1, 1, 1], dtype=np.int64)
    decoys["HLORENT"] = np.array([0, 0, 0], dtype=np.int64)
    decoys["I_HPUBLI"] = np.array([1, 1, 1], dtype=np.int64)
    decoys["I_HLOREN"] = np.array([0, 0, 0], dtype=np.int64)
    return pd.concat([decoys, table], ignore_index=True)


def test_exact_native_key_join_ignores_row_order_and_unreferenced_rows(
    tmp_path, monkeypatch
):
    """Every cohort answer must come from H_SEQ, never from a row position."""
    plain_dir, decoyed_dir = tmp_path / "plain", tmp_path / "decoyed"
    plain_dir.mkdir()
    decoyed_dir.mkdir()
    plain, plain_paths, *_ = invented_source(plain_dir, monkeypatch)
    expected = source.load_authenticated_housing_status(plain, cohort_paths=plain_paths)
    loaded, paths, *_ = invented_source(decoyed_dir, monkeypatch, change=_decoyed)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    # The decoys sit at positions 0-2 and would have produced allocated receipt.
    for name in housing.RAW_COLUMNS:
        assert status.array(name).tolist() == expected.array(name).tolist()
    assert status.array("HPUBLIC").tolist() == [1, 2, 2]
    assert status.array("I_HPUBLI").tolist() == [0, 0, 0]
    assert status.array("status").tolist() == [1, 1, 2]
    for row in status.header_data["source"]["joins"]:
        assert row["source_rows"] == 4
        assert row["joined_rows"] == 1
        assert row["unreferenced_source_rows"] == 3


def test_duplicate_cohort_native_keys_refuse_in_both_layers(tmp_path, monkeypatch):
    """The reviewed HU adapter refuses first; housing must not depend on that."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    with pytest.raises(ValueError, match="repeats H_SEQ"):
        invented_source(
            upstream,
            monkeypatch,
            change=lambda t, y: pd.concat([t, t], ignore_index=True),
        )
    clean = tmp_path / "housing"
    clean.mkdir()
    loaded, paths, *_ = invented_source(clean, monkeypatch)
    original = source._read_cohort
    monkeypatch.setattr(
        source,
        "_read_cohort",
        lambda staged: pd.concat([original(staged)] * 2, ignore_index=True),
    )
    with pytest.raises(housing.HousingStatusRefusalError, match="NATIVE_KEY"):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


def test_cohort_missing_a_scope_household_refuses_incomplete_coverage(
    tmp_path, monkeypatch
):
    """The sealed scope must be covered exactly; a dropped key cannot be skipped."""
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    original = source._read_cohort

    def drop_the_scope_key(staged):
        raw = original(staged)
        return raw.assign(H_SEQ=raw.H_SEQ.to_numpy() + 100)

    monkeypatch.setattr(source, "_read_cohort", drop_the_scope_key)
    with pytest.raises(
        housing.HousingStatusRefusalError, match="SOURCE_KEY_COVERAGE"
    ):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


def test_cohort_missing_a_required_answer_column_refuses(tmp_path, monkeypatch):
    loaded, paths, *_ = invented_source(
        tmp_path, monkeypatch, change=lambda t, y: t.drop(columns=["HLORENT"])
    )
    with pytest.raises(
        housing.HousingStatusRefusalError, match="SOURCE_CONTRACT_REFUSAL"
    ):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


def test_external_link_value_storage_refuses_before_reading_any_answer(
    tmp_path, monkeypatch
):
    """Selected values must live in the verified file, not behind a link out of it."""
    h5py = pytest.importorskip("h5py")
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    original = source._read_cohort
    relinked = tmp_path / "relinked"
    relinked.mkdir()

    def read_an_externally_linked_copy(staged):
        # The staged bytes already passed hash and file-identity checks; this
        # isolates the storage-shape refusal the reader owes the housing loader.
        copy = relinked / Path(staged).name
        copy.write_bytes(Path(staged).read_bytes())
        with h5py.File(copy, "r+") as handle:
            group = handle["household"]
            name = next(
                key
                for key in group
                if key.endswith("_values") and group[key].dtype == np.dtype("<i8")
            )
            values = group[name][()]
            del group[name]
            elsewhere = relinked / f"values-outside-{copy.stem}.h5"
            with h5py.File(elsewhere, "w") as target:
                target.create_dataset("values", data=values)
            group[name] = h5py.ExternalLink(str(elsewhere), "values")
        with h5py.File(copy, "r") as handle:
            assert not isinstance(
                handle["household"].get(name, getlink=True), h5py.HardLink
            )
        return original(copy)

    monkeypatch.setattr(source, "_read_cohort", read_an_externally_linked_copy)
    with pytest.raises(
        housing.HousingStatusRefusalError, match="SOURCE_CONTRACT_REFUSAL"
    ):
        source.load_authenticated_housing_status(loaded, cohort_paths=paths)


def test_housing_load_never_opens_a_genuine_storage_path(tmp_path, monkeypatch):
    """This suite is invented-only; a genuine cohort read would fail the test."""
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    opened = []
    real_open = Path.open

    def record_and_refuse_genuine(self, *args, **kwargs):
        resolved = str(Path(self).resolve())
        opened.append(resolved)
        assert "policyengine-us-data" not in resolved, resolved
        assert "/storage/census_cps_" not in resolved, resolved
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_and_refuse_genuine)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    monkeypatch.undo()
    assert status.array("status").tolist() == [1, 1, 2]
    assert any(name in path for path in opened for name in ("invented-cohort-",))


def test_loaded_modules_are_this_worktree_bytes(tmp_path, monkeypatch):
    """Implementation identity must describe the checkout actually under test."""
    root = Path(source.__file__).resolve().parents[6]
    assert (root / ".git").exists(), root
    for module in (housing, source, money_source, units, reader):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(root), path
    package = Path(housing.__file__).resolve().parent
    loaded, paths, *_ = invented_source(tmp_path, monkeypatch)
    status = source.load_authenticated_housing_status(loaded, cohort_paths=paths)
    implementation = status.header_data["implementation"]
    for name, digest in implementation["modules"].items():
        actual = hashlib.sha256((package / name).read_bytes()).hexdigest()
        assert digest == actual, name
    assert (
        implementation["definition_sha256"] == housing._definition_identity()
    )
    assert hashlib.sha256(
        (package / housing.AUTHORITY_RESOURCE).read_bytes()
    ).hexdigest() == housing.AUTHORITY_SHA256
