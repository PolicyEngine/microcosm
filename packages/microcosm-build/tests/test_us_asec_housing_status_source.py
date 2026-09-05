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
    tmp_path, monkeypatch, *, change=None, object_block=False, history=False
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
