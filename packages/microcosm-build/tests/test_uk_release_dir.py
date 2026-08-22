"""Fail-closed assembly tests for ``tools/build_uk_release_dir.py``."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from microcosm.data import validate_release_dir

ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "build_uk_release_dir.py"
CONTRACT_TEST_PATH = ROOT / "packages" / "microcosm-data" / "tests" / "test_contract.py"
BUILT_AT = "2026-08-20T18:00:00Z"
GIT_COMMIT = "a" * 40


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def assembler():
    return _load_module("build_uk_release_dir", TOOL_PATH)


@pytest.fixture(scope="module")
def contract_fixtures():
    return _load_module("uk_release_contract_test_fixtures", CONTRACT_TEST_PATH)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


@pytest.fixture
def staging_fixture(tmp_path: Path, contract_fixtures, monkeypatch):
    """Wrap the data-contract's signed schema-4 release fixture as staging."""

    contract_release, report = contract_fixtures._write_battery_release(
        tmp_path / "contract"
    )
    release_id = contract_fixtures.UK_EXACT_K_RELEASE_ID
    record_count = contract_fixtures.UK_RECORD_COUNT
    monkeypatch.setenv(
        contract_fixtures.UK_GATE_BATTERY_SIGNING_KEY_ENV,
        contract_fixtures.TEST_UK_TERMINAL_GATE_SIGNING_KEY,
    )

    staging_dir = tmp_path / "staging"
    checkpoint_dir = staging_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    staging_h5 = staging_dir / "populace_uk_2023_staged.h5"
    with h5py.File(staging_h5, "w") as h5:
        household = h5.create_group("household")
        household.create_dataset("table", shape=(record_count,), dtype="u1")

    terminal_path = staging_h5.with_suffix(".terminal_gates.json")
    shutil.copyfile(contract_release / "terminal_gates.json", terminal_path)
    hmrc_evidence_path = staging_h5.with_suffix(".hmrc_income.json")
    hmrc_replay_path = staging_h5.with_suffix(".hmrc_replay.json")
    _write_json(hmrc_evidence_path, {"fixture": "hmrc evidence"})
    _write_json(hmrc_replay_path, {"fixture": "hmrc replay"})

    stage_names = ["frs_hmrc_retained_leaves", "hmrc_spi_income"]
    runtime = {
        "python": "3.14.0",
        "policyengine-core": "3.19.0",
        "policyengine-uk": "2.89.0",
        "microcosm-data": "0.1.0",
    }
    code = {
        "repository": "PolicyEngine/microcosm",
        "git_commit": GIT_COMMIT,
        "git_dirty": False,
    }
    input_h5 = {"sha256": "1" * 64, "size_bytes": 1_315_880_118}
    diagnostics_path = contract_release / "calibration_diagnostics.json"
    diagnostics_sha256 = _sha256(diagnostics_path)
    build_record = {
        "schema_version": 3,
        "build_kind": "uk_national_staging_dataset",
        "status": "passed",
        "calibration_diagnostics_sha256": diagnostics_sha256,
        "stages": stage_names,
        "parameters": {"sample_fraction": 1.0, "rung_token": "f100"},
        "dataset": {
            "time_period": "2023",
            "entity_rows": {"household": record_count},
        },
        "artifacts": {
            "input_h5": input_h5,
            "staging_h5": _artifact_record(staging_h5),
            "terminal_gates": _artifact_record(terminal_path),
            "hmrc_evidence": _artifact_record(hmrc_evidence_path),
            "hmrc_replay": _artifact_record(hmrc_replay_path),
        },
        "terminal_gates": report,
        "runtime": runtime,
        "code": code,
        "built_at": BUILT_AT,
    }
    build_record_path = staging_h5.with_suffix(".build.json")
    _write_json(build_record_path, build_record)

    pipeline = {"stages": [{"name": name} for name in stage_names]}
    run_context = {
        "artifact_kind": "populace_outer_stage_run_context",
        "schema_version": 2,
        "pipeline": pipeline,
        "pipeline_sha256": _canonical_sha256(pipeline),
        "completed": stage_names,
        "stage_records": [{"stage": name} for name in stage_names],
        "run_config": {
            "build_kind": "uk_national_staging_dataset",
            "release_id": release_id,
            "calibration_diagnostics_sha256": diagnostics_sha256,
            "sampling": {"sample_fraction": "1.0", "rung_token": "f100"},
            "certified_candidate": input_h5,
            "code_identity": {},
        },
    }
    run_context_path = checkpoint_dir / "stage_run_context.json"
    _write_json(run_context_path, run_context)

    return SimpleNamespace(
        release_id=release_id,
        record_count=record_count,
        staging_dir=staging_dir,
        staging_h5=staging_h5,
        terminal_path=terminal_path,
        build_record_path=build_record_path,
        run_context_path=run_context_path,
        diagnostics_path=diagnostics_path,
        releases_root=tmp_path / "assembled" / "releases",
    )


def _rewrite_terminal(fixture: SimpleNamespace, report: dict) -> None:
    _write_json(fixture.terminal_path, report)
    build_record = json.loads(fixture.build_record_path.read_text(encoding="utf-8"))
    build_record["terminal_gates"] = report
    build_record["artifacts"]["terminal_gates"] = _artifact_record(
        fixture.terminal_path
    )
    _write_json(fixture.build_record_path, build_record)


def _assemble(assembler, fixture: SimpleNamespace) -> Path:
    return assembler.assemble_uk_release_dir(
        staging_dir=fixture.staging_dir,
        calibration_diagnostics=fixture.diagnostics_path,
        releases_root=fixture.releases_root,
        year=2023,
        tier="frs",
    )


def test_assembles_contract_valid_release_from_schema_4_fixture(
    assembler, staging_fixture
) -> None:
    release_dir = _assemble(assembler, staging_fixture)

    validate_release_dir(release_dir)
    assert release_dir.name == staging_fixture.release_id
    artifact_root = staging_fixture.releases_root.parent / "artifacts"
    dataset = artifact_root / "populace_uk_2023.h5"
    receipt = artifact_root / "populace_uk_2023_calibration.npz"
    assert _sha256(dataset) == _sha256(staging_fixture.staging_h5)
    with np.load(receipt, allow_pickle=False) as calibration:
        assert calibration["schema_version"].item() == 1
        assert calibration["weight_entity"].item() == "household"

    manifest = json.loads(
        (release_dir / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["default_datasets"] == {"national": "populace_uk_2023"}
    assert manifest["record_count"] == staging_fixture.record_count
    assert manifest["artifacts"]["terminal_gates"]["kind"] == "diagnostics"


def test_rejects_unsigned_terminal_report(assembler, staging_fixture) -> None:
    report = json.loads(staging_fixture.terminal_path.read_text(encoding="utf-8"))
    report["attestation"]["signature"] = None
    _rewrite_terminal(staging_fixture, report)

    with pytest.raises(ValueError, match="Terminal report is unsigned"):
        _assemble(assembler, staging_fixture)
    assert not staging_fixture.releases_root.exists()


def test_rejects_non_shippable_terminal_report(assembler, staging_fixture) -> None:
    report = json.loads(staging_fixture.terminal_path.read_text(encoding="utf-8"))
    report["shippable"] = False
    _rewrite_terminal(staging_fixture, report)

    with pytest.raises(ValueError, match="shippable must be true"):
        _assemble(assembler, staging_fixture)
    assert not staging_fixture.releases_root.exists()


def test_rejects_calibration_diagnostics_sha_mismatch(
    assembler, staging_fixture
) -> None:
    original = staging_fixture.diagnostics_path.read_text(encoding="utf-8")
    staging_fixture.diagnostics_path.write_text(original + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Calibration diagnostics SHA-256 mismatch"):
        _assemble(assembler, staging_fixture)
    assert not staging_fixture.releases_root.exists()


def test_rejects_missing_staged_h5(assembler, staging_fixture) -> None:
    staging_fixture.staging_h5.unlink()

    with pytest.raises(ValueError, match="exactly one .*_staged.h5"):
        _assemble(assembler, staging_fixture)
    assert not staging_fixture.releases_root.exists()


def test_build_record_count_is_bound_to_signed_release_id(
    assembler, staging_fixture
) -> None:
    build_record = json.loads(
        staging_fixture.build_record_path.read_text(encoding="utf-8")
    )
    build_record["dataset"]["entity_rows"]["household"] -= 1
    _write_json(staging_fixture.build_record_path, build_record)

    with pytest.raises(ValueError, match="Terminal report release_id"):
        _assemble(assembler, staging_fixture)


def test_rejects_staged_h5_count_mismatch(assembler, staging_fixture) -> None:
    with h5py.File(staging_fixture.staging_h5, "w") as h5:
        household = h5.create_group("household")
        household.create_dataset(
            "table", shape=(staging_fixture.record_count - 1,), dtype="u1"
        )
    build_record = json.loads(
        staging_fixture.build_record_path.read_text(encoding="utf-8")
    )
    build_record["artifacts"]["staging_h5"] = _artifact_record(
        staging_fixture.staging_h5
    )
    _write_json(staging_fixture.build_record_path, build_record)

    with pytest.raises(ValueError, match="household count does not match"):
        _assemble(assembler, staging_fixture)


def test_rejects_candidate_pin_disagreement(assembler, staging_fixture) -> None:
    run_context = json.loads(
        staging_fixture.run_context_path.read_text(encoding="utf-8")
    )
    run_context["run_config"]["certified_candidate"]["sha256"] = "2" * 64
    _write_json(staging_fixture.run_context_path, run_context)

    with pytest.raises(ValueError, match="certified-candidate SHA-256"):
        _assemble(assembler, staging_fixture)


def test_print_release_id_mints_from_certified_candidate_count(
    assembler, tmp_path: Path, monkeypatch, capsys
) -> None:
    candidate = tmp_path / "certified.h5"
    with h5py.File(candidate, "w") as h5:
        household = h5.create_group("household")
        household.create_dataset("table", shape=(7,), dtype="u1")
    monkeypatch.setattr(
        assembler,
        "verify_certified_uk_candidate",
        lambda path: SimpleNamespace(path=Path(path), tier="frs"),
    )

    result = assembler.main(
        [
            "--print-release-id",
            "--candidate-h5",
            str(candidate),
            "--year",
            "2023",
            "--tier",
            "frs",
        ]
    )

    assert result == "populace-uk-2023-frs-k7"
    assert capsys.readouterr().out == "populace-uk-2023-frs-k7\n"
