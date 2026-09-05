"""Prepared ASEC source directory: roster, identity, promotion and refusals.

Every byte here is invented. Private pins are substituted in test scope only;
the production loaders accept neither an override nor caller-supplied pins.
"""

import importlib.util
import shutil
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
from microcosm.build.us_runtime import asec_current_money_units as units
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import asec_prepared_source as prepared
from microcosm.build.us_runtime import asec_student_controls as student
from microcosm.build.us_runtime._asec_current_money_codec import encode_current_money
from microcosm.build.us_runtime.asec_current_money_selection import (
    parse_current_money_body,
)
from microcosm.build.us_runtime.operator_boundary import (
    assert_operator_free_source_frame,
)
from microcosm.graph.population import dtype_for_token, token_for_dtype

#: Raw ASEC routing codes the corrected leaves read. The real raw-stage frame
#: carries them; the minimal money fixture predates the leaves work, so this
#: source fixture adds them to both checkpoints before the loaders re-pin.
ROUTING_CODES = {
    "RESNSS1": [1, 0, 2, 0, 0, 4],
    "RESNSS2": [0, 0, 0, 3, 0, 0],
    "DST_SC1": [4, 0, 1, 4, 0, 0],
    "DST_SC2": [0, 4, 0, 0, 0, 0],
    "DST_SC1_YNG": [0, 0, 0, 0, 0, 0],
    "DST_SC2_YNG": [0, 0, 0, 0, 4, 0],
    "OI_OFF": [20, 12, 0, 20, 12, 0],
}
#: Nominal source dollars for the routed fields, so the restated 2022/2023 rows
#: differ from the 2024 rows by the recorded CPI factors alone.
ROUTED_AMOUNTS = {
    "SS_VAL": [1200.0, 0.0, 2400.0, 600.0, 300.0, 900.0],
    "DIV_VAL": [1000.0, 0.0, 0.0, 0.0, 500.0, 0.0],
    "CAP_VAL": [800.0, 0.0, 0.0, 0.0, 200.0, 0.0],
    "INT_VAL": [400.0, 0.0, 100.0, 0.0, 250.0, 0.0],
    "PNSN_VAL": [0.0, 0.0, 3000.0, 0.0, 1500.0, 0.0],
    "DST_VAL1": [700.0, 0.0, 250.0, 350.0, 0.0, 0.0],
    "DST_VAL2": [0.0, 150.0, 0.0, 0.0, 0.0, 0.0],
    "DST_VAL1_YNG": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "DST_VAL2_YNG": [0.0, 0.0, 0.0, 0.0, 450.0, 0.0],
    "OI_VAL": [275.0, 125.0, 0.0, 175.0, 225.0, 0.0],
    "RNT_VAL": [-500.0, 0.0, 0.0, 0.0, 650.0, 0.0],
    "FRSE_VAL": [0.0, 0.0, -750.0, 0.0, 0.0, 0.0],
    "UC_VAL": [0.0, 0.0, 1100.0, 0.0, 0.0, 0.0],
    "PHIP_VAL": [900.0, 0.0, 0.0, 0.0, 950.0, 0.0],
    "PMED_VAL": [120.0, 0.0, 0.0, 0.0, 140.0, 0.0],
    "POTC_VAL": [60.0, 0.0, 0.0, 0.0, 70.0, 0.0],
    "SPM_CHILDCAREXPNS": [0.0, 0.0, 1300.0, 1300.0, 0.0, 0.0],
}

COHORT_HOUSING = {
    2022: {"HPUBLIC": 1, "HLORENT": 0, "I_HPUBLI": 0, "I_HLOREN": 0},
    2023: {"HPUBLIC": 2, "HLORENT": 1, "I_HPUBLI": 0, "I_HLOREN": 1},
    2024: {"HPUBLIC": 2, "HLORENT": 2, "I_HPUBLI": 0, "I_HLOREN": 0},
}


def _money_fixture():
    path = Path(__file__).with_name("test_us_asec_current_money_source.py")
    spec = importlib.util.spec_from_file_location("prepared_money_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invented_prepared_directory(
    tmp_path, monkeypatch, *, student_controls_missing=True
):
    """Assemble the exact ten-file source directory from invented bytes."""
    pytest.importorskip("microunit")
    pytest.importorskip("tables")
    fixture = _money_fixture()
    original_writer = pd.DataFrame.to_hdf

    def cohort_writer(table, path, *args, **kwargs):
        if Path(path).name.startswith("invented-cohort-"):
            year = int(Path(path).stem.rsplit("-", 1)[1])
            table = table.copy(deep=True)
            table["H_YEAR"] = year + 1
            for name, value in COHORT_HOUSING[year].items():
                table[name] = value
        return original_writer(table, path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_hdf", cohort_writer)
        parent, attachment, *_ = fixture.invented_checkpoints(tmp_path, monkeypatch)
    # Older cohorts lost the complete person income; restoration supplies it.
    for path in (parent, attachment):
        loaded = load_frame_checkpoint(path)
        loaded.frame.person["PTOTVAL"] = [np.nan] * 4 + [-9999.0, 0.0]
        # Both ages in this fixture are outside the Census 16–54 universe.
        # Preserve the genuine old-cohort NaN pattern; S supplies exact observed
        # zero codes through aliases instead of filling these original columns.
        for name in student.CONTROLS:
            loaded.frame.person[name] = (
                [np.nan] * 4 + [0.0, 0.0] if student_controls_missing else [0] * 6
            )
        for column, values in ROUTING_CODES.items():
            loaded.frame.person[column] = np.asarray(values, dtype=np.int64)
        for column, values in ROUTED_AMOUNTS.items():
            loaded.frame.person[column] = np.asarray(values, dtype=np.float64)
        if path == attachment:
            digest = money._sha(parent.read_bytes())
            loaded.metadata["parent_checkpoint_sha256"] = digest
            loaded.metadata["household_observations"]["input_checkpoint_sha256"] = (
                digest
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
    members = {}
    pins = []
    for year, member_name, archive_pin, _pin, _rows, _size in restoration._MEMBER_PINS:
        path = tmp_path / member_name
        offset = (year - 2022) * 2
        pd.DataFrame(
            {
                "PERIDNUM": [str(offset + index).zfill(22) for index in (1, 2)],
                "PH_SEQ": [7, 7],
                "A_LINENO": [1, 2],
                "A_AGE": [55, 14],
                "PTOTVAL": [-9999, 0],
                "A_ENRLW": [0, 0],
                "A_FTPT": [0, 0],
            }
        ).to_csv(path, index=False)
        members[year] = path
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
    monkeypatch.setattr(student, "_MEMBER_PINS", tuple(pins))
    restored = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=members, output_dir=restored
    )
    directory = tmp_path / "prepared-source"
    directory.mkdir()
    shutil.copyfile(parent, directory / prepared.PARENT_FILENAME)
    shutil.copyfile(attachment, directory / prepared.HOUSEHOLD_ATTACHMENT_FILENAME)
    shutil.copyfile(
        restored / restoration.CHECKPOINT_FILENAME,
        directory / prepared.PERSON_INCOME_FILENAME,
    )
    shutil.copyfile(
        restored / "restoration.receipt.json",
        directory / prepared.RESTORATION_RECEIPT_FILENAME,
    )
    for year, name in prepared.COHORT_FILENAMES:
        shutil.copyfile(tmp_path / f"invented-cohort-{year}.h5", directory / name)
    for year, name in prepared.MEMBER_FILENAMES:
        shutil.copyfile(members[year], directory / name)
    return directory


def test_prepared_directory_yields_an_operator_free_promoted_population(
    tmp_path, monkeypatch
):
    directory = invented_prepared_directory(tmp_path, monkeypatch)
    result = prepared.prepare_asec_current_money_population(directory)
    frame = result.frame
    assert frame.n("person") == 6
    assert frame.n("household") == 3
    assert dict(frame.metadata) == {}
    assert frame.mass_log == ()
    assert frame.weighted_entities == ("household",)
    assert_operator_free_source_frame(frame, label="prepared assertion")
    for entity in frame.entities:
        table = frame.table(entity)
        for column in table.columns:
            token_for_dtype(table[column].dtype)
    # The 14 attached housing-status columns arrive as uint8 and are promoted.
    household = frame.table("household")
    assert household["asec_housing_observed_status"].dtype == np.dtype("int64")
    peridnum = frame.person["PERIDNUM"]
    assert token_for_dtype(peridnum.dtype) == "string"
    assert peridnum.dtype == dtype_for_token("string")
    assert peridnum.tolist() == [str(index).zfill(22) for index in range(1, 7)]
    assert result.receipt["entity_rows"]["person"] == 6


def test_receipt_binds_every_stage_identity_without_paths_or_rows(
    tmp_path, monkeypatch
):
    directory = invented_prepared_directory(tmp_path, monkeypatch)
    result = prepared.prepare_asec_current_money_population(directory)
    receipt = result.receipt
    assert receipt["schema"] == prepared.PREPARED_RECEIPT_SCHEMA
    assert receipt["release_eligible"] is False
    assert receipt["all_current_money_consumers_wired"] is False
    assert receipt["restored_source_kind"] == money.RESTORED_SOURCE_KIND
    for name in (
        "source_evidence_sha256",
        "t1_sha256",
        "student_controls_receipt_sha256",
        "student_controls_content_sha256",
        "student_controls_attachment_receipt_sha256",
        "money_header_sha256",
        "money_content_sha256",
        "money_spec_sha256",
        "scope_sha256",
        "tax_receipt_sha256",
        "tax_old_partition_sha256",
        "tax_new_partition_sha256",
        "housing_content_sha256",
        "housing_attachment_receipt_sha256",
        "restoration_receipt_sha256",
        "pre_promotion_frame_sha256",
        "post_promotion_frame_sha256",
        "implementation_sha256",
    ):
        assert money._digest(receipt[name]), name
    assert (
        receipt["pre_promotion_frame_sha256"] != receipt["post_promotion_frame_sha256"]
    )
    transitions = {
        (item["entity"], item["column"]): item["to"]
        for item in receipt["dtype_transitions"]
    }
    assert transitions[("household", "asec_housing_observed_status")] == "int64"
    text = result.receipt_payload.decode()
    assert str(tmp_path) not in text and "/" not in receipt["implementation_sha256"]
    assert not any(
        isinstance(value, str) and value.startswith("/") for value in receipt.values()
    )
    assert receipt["student_controls_reference_period"] == (
        "preceding_survey_week_income_cohort_plus_one"
    )
    assert receipt["annual_five_month_student_status_validated"] is False


def test_money_body_leaves_as_bytes_that_reparse_against_the_receipt(
    tmp_path, monkeypatch
):
    directory = invented_prepared_directory(tmp_path, monkeypatch)
    result = prepared.prepare_asec_current_money_population(directory)
    body = parse_current_money_body(
        result.money_payload,
        expected_header_sha256=result.receipt["money_header_sha256"],
        expected_content_sha256=result.receipt["money_content_sha256"],
        field_entities=result.field_entities,
    )
    assert body.header_sha256 == result.receipt["money_header_sha256"]
    assert body.person_rows == 6 and body.household_rows == 3
    assert body.entity_of("HTOTVAL") == "household"
    # Restoration supplied the observed person income for every cohort.
    assert body.field("PTOTVAL").validity.tolist() == [1] * 6


def test_separate_student_source_precedes_actual_tax_and_preserves_t1_money(
    tmp_path, monkeypatch
):
    directory = invented_prepared_directory(tmp_path, monkeypatch)
    originals = {
        name: (directory / name).read_bytes() for name in prepared.PREPARED_SOURCE_FILES
    }
    captured = {}
    original_student_loader = student.load_authenticated_student_controls
    original_tax = prepared.reconstruct_current_money_tax_units
    original_housing = prepared.load_authenticated_housing_status

    def student_loader(source, ready, **kwargs):
        captured["source"] = source
        captured["ready"] = ready
        captured["frame"] = legacy._frame_signature(source.frame)
        captured["source_identity"] = source.source.identity
        captured["money"] = encode_current_money(ready)
        # The uncomposed source still refuses the exact missing control. The
        # successful prepared path must therefore pass through authenticated S.
        with pytest.raises(money.MoneyRefusalError) as error:
            units.current_money_microunit_view(source, ready)
        assert error.value.reason == "MICROUNIT_CONTROL_MISSING"
        assert error.value.field == "A_ENRLW"
        result = original_student_loader(source, ready, **kwargs)
        captured["controls"] = result
        return result

    def tax_reconstruction(source, ready):
        assert type(source) is student.StudentControlsAttachedAsec
        assert source.parent is captured["source"]
        assert ready is captured["ready"]
        captured["attachment"] = source
        return original_tax(source, ready)

    def housing_loader(source, **kwargs):
        assert source is captured["source"]
        return original_housing(source, **kwargs)

    monkeypatch.setattr(student, "load_authenticated_student_controls", student_loader)
    monkeypatch.setattr(
        prepared, "reconstruct_current_money_tax_units", tax_reconstruction
    )
    monkeypatch.setattr(prepared, "load_authenticated_housing_status", housing_loader)
    result = prepared.prepare_asec_current_money_population(directory)
    source = captured["source"]
    controls = captured["controls"]
    assert source.source.identity == captured["source_identity"]
    assert legacy._frame_signature(source.frame) == captured["frame"]
    assert result.money_payload == captured["money"]
    assert encode_current_money(captured["ready"]) == captured["money"]
    assert result.receipt["source_evidence_sha256"] == money._sha(
        source.source.identity
    )
    assert result.receipt["t1_sha256"] == money._sha(
        originals[prepared.PERSON_INCOME_FILENAME]
    )
    assert controls.receipt["source_identity"].encode() == source.source.identity
    assert result.receipt["student_controls_content_sha256"] == controls.content_sha256
    assert result.receipt["student_controls_receipt_sha256"] == money._sha(
        money._json(controls.receipt)
    )
    assert result.receipt["student_controls_attachment_receipt_sha256"] == money._sha(
        money._json(captured["attachment"].receipt)
    )
    for name, alias in zip(student.CONTROLS, student.ALIASES, strict=True):
        pd.testing.assert_series_equal(
            result.frame.person[name], source.frame.person[name]
        )
        assert result.frame.person[name].isna().tolist() == [True] * 4 + [False] * 2
        assert result.frame.person[alias].tolist() == [0] * 6
    assert {
        name: (directory / name).read_bytes() for name in prepared.PREPARED_SOURCE_FILES
    } == originals


def test_complete_source_compares_student_incumbents_before_preparing(
    tmp_path, monkeypatch
):
    directory = invented_prepared_directory(
        tmp_path, monkeypatch, student_controls_missing=False
    )
    result = prepared.prepare_asec_current_money_population(directory)
    for name, alias in zip(student.CONTROLS, student.ALIASES, strict=True):
        assert result.frame.person[name].tolist() == result.frame.person[alias].tolist()
        assert not result.frame.person[name].isna().any()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: (d / prepared.PARENT_FILENAME).unlink(), "SOURCE_FILE_ROSTER"),
        (lambda d: (d / "extra.txt").write_text("x"), "SOURCE_FILE_ROSTER"),
        (lambda d: (d / "sub").mkdir(), "SOURCE_FILE_ROSTER"),
    ],
)
def test_source_roster_is_exact(tmp_path, monkeypatch, mutate, match):
    directory = invented_prepared_directory(tmp_path, monkeypatch)
    mutate(directory)
    with pytest.raises(prepared.PreparedSourceRefusalError, match=match):
        prepared.prepare_asec_current_money_population(directory)


def test_altered_member_or_attachment_bytes_refuse(tmp_path, monkeypatch):
    directory = invented_prepared_directory(tmp_path, monkeypatch)
    member = directory / prepared.MEMBER_FILENAMES[0][1]
    member.write_bytes(member.read_bytes().replace(b"55", b"56"))
    with pytest.raises(money.MoneyRefusalError):
        prepared.prepare_asec_current_money_population(directory)


def test_altered_restored_attachment_refuses_canonical_replay(tmp_path, monkeypatch):
    directory = invented_prepared_directory(tmp_path, monkeypatch)
    target = directory / prepared.PERSON_INCOME_FILENAME
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF
    target.write_bytes(bytes(payload))
    with pytest.raises(money.MoneyRefusalError):
        prepared.prepare_asec_current_money_population(directory)


def test_altered_restoration_receipt_refuses(tmp_path, monkeypatch):
    directory = invented_prepared_directory(tmp_path, monkeypatch)
    target = directory / prepared.RESTORATION_RECEIPT_FILENAME
    payload = target.read_bytes()
    target.write_bytes(payload.replace(b'"reader":"', b'"reader":"x'))
    with pytest.raises(
        prepared.PreparedSourceRefusalError, match="RESTORATION_RECEIPT"
    ):
        prepared.prepare_asec_current_money_population(directory)


def test_implementation_identity_is_stable_and_path_independent():
    first = prepared.prepared_source_implementation_identity()
    assert first == prepared.prepared_source_implementation_identity()
    assert money._digest(first)
