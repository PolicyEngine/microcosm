"""UK national release-directory assembly and refusal coverage."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.calibration_run import resign_uk_gate_report
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.release_certification import (
    compose_uk_release_certification,
)
from microcosm.build.uk_runtime.release_identity import UK_NATIONAL_RELEASE_ID
from microcosm.data.contract import validate_release_dir
from microcosm.frame import WeightKind


def _load_fixture_module():
    path = Path(__file__).with_name("uk_certification_fixtures.py")
    spec = importlib.util.spec_from_file_location("uk_certification_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_FIXTURES = _load_fixture_module()
_green_certification_inputs_fixture = _FIXTURES.green_certification_inputs
_signing_key_fixture = _FIXTURES.signing_key
sha256 = _FIXTURES.sha256

_SHARED_FIXTURES = (
    _green_certification_inputs_fixture,
    _signing_key_fixture,
)
_CODE_PIN = "5fa48f07436a806ad75ff76fd22cfb8613bddbe0"
_ATTEMPT_ID = "uk-frs-calibration-attempt-20260828T101112Z-1a2b3c4d"
_CUT_TAG = f"{UK_NATIONAL_RELEASE_ID}-20260828T101112Z-1a2b3c4d"


def _load_driver_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "assemble_uk_release_dir.py"
    spec = importlib.util.spec_from_file_location("assemble_uk_release_dir", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _frame(weights: list[float], *, weight_kind: WeightKind):
    ids = np.arange(len(weights), dtype="int64")
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": np.asarray(weights, dtype="float64"),
            }
        ),
        time_period="2024",
        weight_kind=weight_kind,
    )


def _diagnostics() -> dict:
    return {
        "schema_version": 6,
        "weight_entity": "household",
        "options": {"epochs": 2},
        "target_surface": {
            "schema_version": 1,
            "weight_entity": "household",
            "n_targets": 1,
            "n_records": 2,
            "constraint_matrix": {"rows": 1, "columns": 2, "nnz": 1},
            "sha256": "a" * 64,
            "names_sha256": "b" * 64,
            "values_sha256": "c" * 64,
        },
        "target_registry": {
            "country": "uk",
            "version": "registry-test-v1",
            "n_specs": 1,
        },
        "targets": [
            {
                "name": "ons.population@2024",
                "target_name": "ons.population",
                "period": 2024,
                "entity": "household",
                "target": 2.0,
                "compiled_target": 2.0,
                "initial_estimate": 2.0,
                "final_estimate": 2.0,
                "relative_error": 0.0,
                "source": "synthetic test",
                "measure": {"column": "household_id"},
                "filter": None,
                "metadata": {},
            }
        ],
        "loss_trajectory": [1.0, 0.0],
        "skipped": [],
        "n_records": 2,
        "effective_sample_size": 1.0,
        "top_1pct_weight_share": 1.0,
        "uk_diagnostics": {
            "schema_version": 1,
            "weights": {
                "n_records": 2,
                "positive_weight_records": 1,
                "zero_weight_records": 1,
                "total_weight": 4.0,
                "effective_sample_size": 1.0,
                "ess_fraction": 0.5,
                "median_positive_weight": 4.0,
                "max_weight": 4.0,
                "max_to_median_positive_weight": 1.0,
                "top_1pct_weight_share": 1.0,
            },
            "zero_weight_rows_by_stratum": [
                {
                    "stratum": {"household_is_spi_synthetic": False},
                    "rows": 2,
                    "positive_weight_rows": 1,
                    "zero_weight_rows": 1,
                    "weight_sum": 4.0,
                }
            ],
            "target_pass_rates_by_geography_level": [
                {
                    "geography_level": level,
                    "n_targets": 1 if level == "national" else 0,
                    "n_scored": 1 if level == "national" else 0,
                    "n_skipped": 0,
                    "n_within_10pct": 1 if level == "national" else 0,
                    "pass_rate": 1.0 if level == "national" else None,
                }
                for level in (
                    "national",
                    "region",
                    "country",
                    "local_authority",
                    "constituency",
                )
            ],
        },
        "build": {"code_pin": _CODE_PIN},
    }


@pytest.fixture
def assembler_inputs(green_certification_inputs, tmp_path: Path):
    candidate = green_certification_inputs["candidate_path"]
    spine = tmp_path / "spine.h5"
    write_uk_national_frame(
        _frame([0.0, 4.0], weight_kind=WeightKind.CALIBRATED), candidate
    )
    write_uk_national_frame(_frame([1.0, 2.0], weight_kind=WeightKind.DESIGN), spine)

    diagnostics_path = tmp_path / "calibration_diagnostics.json"
    diagnostics_path.write_text(json.dumps(_diagnostics()), encoding="utf-8")
    diagnostics_sha = sha256(diagnostics_path)
    for report_name in ("seam_report_path", "release_cut_report_path"):
        report_path = green_certification_inputs[report_name]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["release_evidence"]["calibration_diagnostics_sha256"] = diagnostics_sha
        if report_name == "release_cut_report_path":
            report["release_id"] = UK_NATIONAL_RELEASE_ID
        resign_uk_gate_report(report)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )

    candidate_sha = sha256(candidate)
    score_path = green_certification_inputs["score_receipt_path"]
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score["artifacts"]["candidate"].update(
        sha256=candidate_sha, size_bytes=candidate.stat().st_size
    )
    score_path.write_text(json.dumps(score), encoding="utf-8")

    build_record = green_certification_inputs["build_record"]
    build_record.update(
        build_id=_ATTEMPT_ID,
        input_posture={"sha256": sha256(spine)},
        gate_summary={"uk_target_fit": "passed"},
    )
    build_record["spine_provenance"]["rules_engine"] = {
        "name": "policyengine-uk",
        "version": "2.89.0",
    }
    build_record["artifacts"] = {
        "staging_h5": {"sha256": candidate_sha},
        "diagnostics_json": {"sha256": diagnostics_sha},
        "terminal_gate_json": {
            "sha256": sha256(green_certification_inputs["seam_report_path"])
        },
    }

    certification_path = tmp_path / "release_certification.json"
    compose_inputs = {
        **green_certification_inputs,
        "release_id": UK_NATIONAL_RELEASE_ID,
        "candidate_name": candidate.stem,
        "candidate_sha256": candidate_sha,
        "certification_path": certification_path,
    }
    compose_uk_release_certification(**compose_inputs)
    build_record_path = tmp_path / "build_record.json"
    build_record_path.write_text(json.dumps(build_record), encoding="utf-8")
    out_dir = tmp_path / "releases"
    argv = [
        "--candidate-h5",
        str(candidate),
        "--spine-h5",
        str(spine),
        "--certification-json",
        str(certification_path),
        "--build-record-json",
        str(build_record_path),
        "--diagnostics-json",
        str(diagnostics_path),
        "--seam-gate-report",
        str(green_certification_inputs["seam_report_path"]),
        "--release-cut-gate-json",
        str(green_certification_inputs["release_cut_report_path"]),
        "--score-receipt",
        str(score_path),
        "--out-dir",
        str(out_dir),
        "--runtime-version",
        "policyengine-core=3.19.0",
    ]
    return {
        "argv": argv,
        "candidate": candidate,
        "spine": spine,
        "diagnostics": diagnostics_path,
        "certification": certification_path,
        "score": score_path,
        "out_dir": out_dir,
    }


def test_assemble_green_release_dir(assembler_inputs, capsys) -> None:
    driver = _load_driver_module()
    assert driver.main(assembler_inputs["argv"]) == 0
    summary = json.loads(capsys.readouterr().out)
    release_dir = assembler_inputs["out_dir"] / UK_NATIONAL_RELEASE_ID

    validate_release_dir(release_dir)
    calibration_path = assembler_inputs["candidate"].with_name(
        "microcosm_uk_2024_calibration.npz"
    )
    with np.load(calibration_path) as calibration:
        np.testing.assert_array_equal(
            calibration.files,
            [
                "household_weight",
                "initial_household_weight",
            ],
        )
        np.testing.assert_array_equal(calibration["household_weight"], [0.0, 4.0])
        np.testing.assert_array_equal(
            calibration["initial_household_weight"], [1.0, 2.0]
        )

    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    release_manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert "staging" not in build_manifest
    assert build_manifest["attempt_id"] == _ATTEMPT_ID
    assert build_manifest["cut_tag"] == _CUT_TAG
    assert {entry["revision"] for entry in release_manifest["artifacts"].values()} == {
        _CUT_TAG
    }
    assert "dataset_role" not in release_manifest
    assert (
        release_dir / "calibration_diagnostics.json"
    ).read_bytes() == assembler_inputs["diagnostics"].read_bytes()
    assert (
        release_dir / "release_certification.json"
    ).read_bytes() == assembler_inputs["certification"].read_bytes()
    assert summary["cut_tag"] == _CUT_TAG
    assert summary["release_dir"] == str(release_dir)
    assert summary["publish_command"] == (
        f"uv run python -m microcosm.data.publish_cli {release_dir} "
        "--repo-id policyengine/populace-uk-private "
        f"--artifact-root {assembler_inputs['candidate'].parent} --no-latest "
        f"--tag-name {_CUT_TAG}"
    )


def test_assemble_refuses_candidate_sha_mismatch(assembler_inputs) -> None:
    assembler_inputs["candidate"].write_bytes(
        assembler_inputs["candidate"].read_bytes() + b"tampered"
    )
    with pytest.raises(SystemExit, match="candidate bytes"):
        _load_driver_module().main(assembler_inputs["argv"])


def test_assemble_refuses_unshippable_certification(assembler_inputs) -> None:
    certification = json.loads(assembler_inputs["certification"].read_text())
    certification["shippable"] = False
    assembler_inputs["certification"].write_text(json.dumps(certification))
    with pytest.raises(SystemExit, match="shippable must be true"):
        _load_driver_module().main(assembler_inputs["argv"])


def test_assemble_refuses_missing_score_receipt(assembler_inputs) -> None:
    assembler_inputs["score"].unlink()
    with pytest.raises(FileNotFoundError, match="score_vs_enhanced_frs"):
        _load_driver_module().main(assembler_inputs["argv"])


def test_assemble_refuses_unprefixed_cut_tag(assembler_inputs) -> None:
    with pytest.raises(SystemExit, match="must start with"):
        _load_driver_module().main(
            [*assembler_inputs["argv"], "--cut-tag", "not-a-national-cut"]
        )


def test_assemble_refuses_reserialized_diagnostics(assembler_inputs) -> None:
    diagnostics = json.loads(assembler_inputs["diagnostics"].read_text())
    assembler_inputs["diagnostics"].write_text(
        json.dumps(diagnostics, indent=4), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="diagnostics bytes"):
        _load_driver_module().main(assembler_inputs["argv"])
