"""Invented storage/coverage checks; never issue a substitute native owner."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_us_common_frame_export_contract import _parent

from microcosm.build.us_runtime import native_survey_handoff as handoff
from microcosm.frame.rules import ExportContract


def test_coverage_names_missing_leaves_and_excludes_prior_year_income():
    frame = _parent()
    coverage = handoff.native_survey_input_inventory(frame)
    by_name = {(row["entity"], row["variable"]): row for row in coverage}
    assert by_name["person", "weekly_hours_worked_before_lsr"]["status"] == "missing"
    assert (
        by_name["person", "self_employment_income_last_year"]["status"]
        == "excluded_native_scope"
    )
    assert (
        by_name["person", "previous_year_income_available"]["status"]
        == "excluded_native_scope"
    )
    assert frame.n("person") == 6
    assert "employment_income_last_year" not in frame.person


def test_coverage_does_not_infer_signal_or_applicability_from_presence():
    frame = _parent()
    frame.person["weekly_hours_worked_before_lsr"] = [
        0.0,
        10.0,
        np.nan,
        30.0,
        20.0,
        0.0,
    ]
    row = next(
        r
        for r in handoff.native_survey_input_inventory(frame)
        if r["variable"] == "weekly_hours_worked_before_lsr"
    )
    assert row["status"] == "contains_unknowns"
    assert row["missing_values"] == 1
    assert row["source_signal_verified"] is False
    assert row["applicability_verified"] is False


def test_engine_contract_keeps_unsupported_projection_explicit():
    frame = _parent()
    contract = ExportContract(
        required=("absent_engine_input",),
        forbidden=("money",),
        optional=(),
        formula_owned_excluded=("hours",),
        closed=True,
    )
    inventory = handoff.engine_export_inventory(frame, contract)
    assert inventory["missing_required"] == ["absent_engine_input"]
    assert inventory["forbidden_present"] == ["money"]
    assert inventory["formula_owned_present"] == ["hours"]
    assert "origin" in inventory["unexpected_columns"]
    assert frame.person.money.tolist() == [-0.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_fake_owner_and_descriptive_checked_view_cannot_export(tmp_path):
    for fake in (
        SimpleNamespace(population=SimpleNamespace(frame=_parent())),
        object(),
    ):
        with pytest.raises(ValueError, match="UNISSUED_RUN"):
            handoff.write_native_survey_development_checkpoint(
                fake, tmp_path / "refused"
            )
        assert not (tmp_path / "refused").exists()


def test_invented_checkpoint_readback_preserves_full_frame(tmp_path):
    frame = _parent()
    report = {
        "protocol": handoff.PROTOCOL,
        "release_eligible": False,
        "invented_storage_test": True,
    }
    root = tmp_path / "checkpoint"
    # Test the storage primitive, not the public native-owner exporter.
    handoff._write_checkpoint(frame, report, root)
    loaded = handoff.load_native_survey_development_checkpoint(root)
    handoff.same_replayed_frame(frame, loaded.frame)
    assert loaded.owner_live_verified is False
    assert loaded.report == report
    assert loaded.frame.schema == frame.schema
    assert (
        loaded.frame.weights_for("household").kind
        == frame.weights_for("household").kind
    )
    assert loaded.frame.metadata == frame.metadata
    assert (
        loaded.frame.person.reported.isna().tolist()
        == frame.person.reported.isna().tolist()
    )
    assert np.signbit(loaded.frame.person.money.iloc[0])
    assert loaded.frame.table("household").assigned_block.tolist() == [
        "001",
        "002",
        "003",
    ]


def test_checkpoint_does_not_overwrite_existing_destination(tmp_path):
    with pytest.raises(FileExistsError):
        handoff._write_checkpoint(_parent(), {}, tmp_path)


@pytest.mark.parametrize("target", ["population.h5", "handoff.json"])
def test_checkpoint_rejects_changed_payload_or_report(tmp_path, target):
    root = tmp_path / "checkpoint"
    handoff._write_checkpoint(
        _parent(), {"protocol": handoff.PROTOCOL, "release_eligible": False}, root
    )
    path = root / target
    if target.endswith("h5"):
        with path.open("ab") as output:
            output.write(b"changed")
    else:
        report = json.loads(path.read_text())
        report["report"]["release_eligible"] = True
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="CHECKPOINT"):
        handoff.load_native_survey_development_checkpoint(root)


def test_maintained_builder_entry_refuses_unissued_owner_before_output(tmp_path):
    from test_us_fiscal_refresh_builder import _load_builder_module

    builder = _load_builder_module()
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        builder.prepare_native_survey_development_input(object(), tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("mutated", ["frame", "report", "final_digest", "final_frame"])
def test_final_owner_io_cannot_mutate_already_checked_readback(tmp_path, mutated):
    # These are descriptive invented values for the pure comparison helper.
    # No public handoff accepts them as a substitute for an issued native run.
    original = _parent()
    report = {"owner_receipt_sha256": "invented-receipt", "release_eligible": False}
    root = tmp_path / "readback"
    handoff._write_checkpoint(original, report, root)
    loaded = handoff.load_native_survey_development_checkpoint(root)
    borrowed = handoff.NativeSurveyDevelopmentInput(original, report)
    final = SimpleNamespace(
        digest="invented-receipt", population=SimpleNamespace(frame=original)
    )
    handoff._validate_readback_after_owner_io(
        borrowed, loaded.frame, loaded.report, final
    )
    # Model a mutation during the final foreign I/O, after a prior comparison.
    if mutated == "frame":
        loaded.frame.person.loc[0, "money"] = 1234.0
    elif mutated == "report":
        loaded.report["release_eligible"] = True
    elif mutated == "final_digest":
        final.digest = "changed"
    else:
        final.population.frame = loaded.frame
    with pytest.raises(
        ValueError, match="NATIVE_HANDOFF_CHECKPOINT_OWNER|SURVEY_POPULATION_REPLAY"
    ):
        handoff._validate_readback_after_owner_io(
            borrowed, loaded.frame, loaded.report, final
        )
