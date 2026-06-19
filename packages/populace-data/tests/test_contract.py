"""The release contract: every published release looks the same, loudly.

These are behavioral tests against the failure modes already observed on the
Hub: a release with no build manifest at all (1abddeb), and two coexisting
release-manifest schemas (an unversioned early shape next to
``schema_version: 1``). A valid release passes silently; every broken release
fails with each violation named.
"""

import hashlib
import json
from pathlib import Path

import pytest

from populace.data import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
    ReleaseContractError,
    required_release_files,
    validate_release_dir,
)

RELEASE_ID = "populace-us-2024-9f1260b-20260611"
UK_RELEASE_ID = "populace-uk-2024-9f1260b-20260611"
GIT_COMMIT = "5fa48f07436a806ad75ff76fd22cfb8613bddbe0"
DATASET_SHA = "d" * 64
CALIBRATION_SHA = "a" * 64
DIAGNOSTICS_SHA = "c" * 64
SOURCE_COVERAGE_SHA = "9" * 64
TARGET_SURFACE_SHA = "e" * 64
REGISTRY_VERSION = "registryabc123"
TARGET_COUNT = 11

DEDUCTION_CRITICAL_TARGETS = (
    (
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount",
        1_000_000_000_000.0,
        1_020_000_000_000.0,
        "itemized_deduction_total",
    ),
    (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all.charitable_amount@2024",
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all.charitable_amount",
        200_000_000_000.0,
        205_000_000_000.0,
        "charitable_deduction_total",
    ),
    (
        "irs_soi.ty2022.historic_table_2.us.all.limited_state_local_taxes_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.limited_state_local_taxes_amount",
        120_000_000_000.0,
        121_000_000_000.0,
        "salt_deduction_total",
    ),
    (
        "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount",
        80_000_000_000.0,
        82_000_000_000.0,
        "medical_expense_deduction_total",
    ),
    (
        "irs_soi.ty2022.historic_table_2.us.all.qbi_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.qbi_amount",
        31_000_000_000.0,
        30_000_000_000.0,
        "qualified_business_income_deduction_total",
    ),
    (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "interest_paid_deduction_amount@2024",
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "interest_paid_deduction_amount",
        200_000_000_000.0,
        198_000_000_000.0,
        "interest_deduction_total",
    ),
)


def _model_package(release_id: str) -> tuple[str, str]:
    if release_id.startswith("populace-uk-"):
        return ("policyengine-uk", "2.89.0")
    return ("policyengine-us", "1.729.0")


def _build_manifest(release_id: str = RELEASE_ID) -> dict:
    model_package, model_version = _model_package(release_id)
    return {
        "build_id": release_id,
        "builder": "populace",
        "build_sha": GIT_COMMIT[:7],
        "code": {
            "repository": "PolicyEngine/populace",
            "git_commit": GIT_COMMIT,
            "git_dirty": False,
        },
        "runtime": {
            "python": "3.14.0",
            "policyengine-core": "3.19.0",
            model_package: model_version,
        },
        "dataset": {"filename": "populace_us_2024.h5", "sha256": DATASET_SHA},
        "calibration": {
            "filename": "populace_us_2024_calibration.npz",
            "sha256": CALIBRATION_SHA,
            "target_surface": {"sha256": TARGET_SURFACE_SHA, "n_targets": TARGET_COUNT},
            "target_registry": {"version": REGISTRY_VERSION, "n_specs": TARGET_COUNT},
        },
        "gates": {"exported_nonzero": {"passed": True}},
    }


def _release_manifest(
    release_id: str = RELEASE_ID,
    *,
    diagnostics_sha: str = DIAGNOSTICS_SHA,
    source_coverage_sha: str = SOURCE_COVERAGE_SHA,
) -> dict:
    model_package, model_version = _model_package(release_id)
    manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {"name": "populace-data", "version": "0.1.0"},
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release_id,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.19.0",
            },
            "built_with_model_package": {
                "name": model_package,
                "version": model_version,
            },
        },
        "compatible_core_packages": [
            {"name": "policyengine-core", "specifier": "==3.19.0"}
        ],
        "compatible_model_packages": [
            {"name": model_package, "specifier": f"=={model_version}"}
        ],
        "artifacts": {
            "populace_us_2024": {
                "kind": "microdata",
                "path": "populace_us_2024.h5",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": DATASET_SHA,
            },
            "populace_us_2024_calibration": {
                "kind": "calibration",
                "path": "populace_us_2024_calibration.npz",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": CALIBRATION_SHA,
            },
            "calibration_diagnostics": {
                "kind": "diagnostics",
                "path": "calibration_diagnostics.json",
                "repo_id": "policyengine/populace-us",
                "revision": release_id,
                "sha256": diagnostics_sha,
            },
        },
    }
    if release_id.startswith("populace-us-"):
        manifest["artifacts"]["us_source_coverage"] = {
            "kind": "diagnostics",
            "path": US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
            "repo_id": "policyengine/populace-us",
            "revision": release_id,
            "sha256": source_coverage_sha,
        }
    return manifest


def _calibration_diagnostics() -> dict:
    return {
        "schema_version": 2,
        "weight_entity": "household",
        "options": {"epochs": 120},
        "target_surface": {
            "schema_version": 1,
            "weight_entity": "household",
            "n_targets": TARGET_COUNT,
            "n_records": 2,
            "constraint_matrix": {"rows": 1, "columns": 2, "nnz": 2},
            "sha256": TARGET_SURFACE_SHA,
            "names_sha256": "b" * 64,
            "values_sha256": "f" * 64,
        },
        "target_registry": {
            "country": "us",
            "version": REGISTRY_VERSION,
            "n_specs": TARGET_COUNT,
        },
        "loss_trajectory": [1.0, 0.5],
        "skipped": [],
        "targets": [
            _target_row(
                "population@2024",
                target_name="population",
                target=1.0,
                initial_estimate=0.8,
                final_estimate=1.0,
                relative_error=0.0,
                family="cbo",
            ),
            _target_row(
                "irs_soi.ty2022.historic_table_2.us.all."
                "income_tax_liability_amount@2024",
                target_name=(
                    "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount"
                ),
                target=2_105_345_646_000.0,
                initial_estimate=2_000_000_000_000.0,
                final_estimate=2_067_762_165_736.424,
                relative_error=-0.0178514536722185,
                family="irs_soi",
                target_role="federal_income_tax_total",
            ),
            _target_row(
                "irs_soi.ty2022.historic_table_2.us.all."
                "income_tax_liability_returns@2024",
                target_name=(
                    "irs_soi.ty2022.historic_table_2.us.all."
                    "income_tax_liability_returns"
                ),
                target=113_562_590.0,
                initial_estimate=105_421_734.40619682,
                final_estimate=105_437_267.69738781,
                relative_error=-0.07154928663226319,
                family="irs_soi",
            ),
            _target_row(
                "ssa_supplement.cy2024.oasdi_ssi_payments."
                "social_security_benefits.payment_amount@2024",
                target_name=(
                    "ssa_supplement.cy2024.oasdi_ssi_payments."
                    "social_security_benefits.payment_amount"
                ),
                target=1_471_195_000_000.0,
                initial_estimate=1_541_646_703_291.2527,
                final_estimate=1_541_540_768_722.367,
                relative_error=0.047815394099604024,
                family="ssa",
                target_role="social_security_total",
            ),
            _target_row(
                "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024",
                target_name="irs_soi.ty2022.historic_table_2.us.all.ctc_amount",
                target=82_863_353_000.0,
                initial_estimate=132_000_000_000.0,
                final_estimate=90_000_000_000.0,
                relative_error=(90_000_000_000.0 - 82_863_353_000.0) / 82_863_353_000.0,
                family="irs_soi",
                target_role="ctc_total",
            ),
            *deduction_critical_target_rows(),
        ],
    }


def deduction_critical_target_rows() -> list[dict]:
    return [
        _target_row(
            name,
            target_name=target_name,
            target=target,
            initial_estimate=target * 1.5,
            final_estimate=final,
            relative_error=(final - target) / target,
            family="irs_soi",
            target_role=target_role,
        )
        for name, target_name, target, final, target_role in DEDUCTION_CRITICAL_TARGETS
    ]


def _target_row(
    name: str,
    *,
    target_name: str,
    target: float,
    initial_estimate: float,
    final_estimate: float,
    relative_error: float,
    family: str,
    target_role: str | None = None,
) -> dict:
    metadata = {"target_role": target_role} if target_role else {}
    return {
        "name": name,
        "target_name": target_name,
        "period": 2024,
        "entity": "household",
        "aggregation": "sum",
        "measure": None,
        "filter": None,
        "source": "Fixture admin target",
        "metadata": metadata,
        "target": target,
        "compiled_target": target,
        "initial_estimate": initial_estimate,
        "final_estimate": final_estimate,
        "relative_error": relative_error,
        "within_tolerance": None,
        "registry": {"family": family},
    }


def _source_coverage_diagnostics() -> dict:
    return {
        "schema_version": 1,
        "classification": "release_gate",
        "source_contract": {
            "name": "us_source_coverage",
            "ledger_commit": "5fa48f07436a806ad75ff76fd22cfb8613bddbe0",
        },
        "gate": {
            "name": "us_source_coverage",
            "passed": True,
            "failures": [],
        },
        "coverage_summary": {
            "hard_target": {
                "families": 9,
                "package_aliases": 38,
                "covered_package_aliases": 38,
                "missing_package_aliases": 0,
                "reviewed_excluded_package_aliases": 0,
            },
            "validation_only": {"families": 6, "activated_families": 0},
            "source_gap": {"families": 6, "missing_source_packages": 11},
        },
        "hard_target_families": {"population_age_sex": {}},
        "validation_only_families": {"census_cps_spm": {}},
        "source_gap_families": {"usda_wic": {}},
        "active_target_aliases": ["census-pep-2024-national-age-sex"],
        "active_target_families": [],
        "missing_hard_targets": [],
        "reviewed_exclusions": {},
        "validation_only_activated": [],
        "fiscal_target_sources": {
            "cbo": {
                "label": "Congressional Budget Office revenue projections",
                "target_count": 1,
                "sources": ["Census PEP 2024"],
                "reference_urls": ["https://example.test/source"],
            },
            "irs_soi": {
                "label": "IRS Statistics of Income",
                "target_count": 9,
                "sources": ["IRS SOI Historic Table 2"],
                "reference_urls": ["https://example.test/soi"],
            },
            "ssa": {
                "label": "Social Security Administration",
                "target_count": 1,
                "sources": ["SSA Annual Statistical Supplement"],
                "reference_urls": ["https://example.test/ssa"],
            },
        },
    }


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    """A complete, contract-valid release directory."""
    directory = tmp_path / "releases" / RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(json.dumps(_build_manifest()))
    (directory / "calibration_diagnostics.json").write_text(
        json.dumps(_calibration_diagnostics())
    )
    (directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(
        json.dumps(_source_coverage_diagnostics())
    )
    diagnostics_sha = _sha256(directory / "calibration_diagnostics.json")
    source_coverage_sha = _sha256(directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE)
    (directory / "release_manifest.json").write_text(
        json.dumps(
            _release_manifest(
                diagnostics_sha=diagnostics_sha,
                source_coverage_sha=source_coverage_sha,
            )
        )
    )
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_and_refresh_manifest_hash(
    release_dir: Path,
    *,
    filename: str,
    artifact_key: str,
    payload: dict,
) -> None:
    (release_dir / filename).write_text(json.dumps(payload))
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][artifact_key]["sha256"] = _sha256(release_dir / filename)
    manifest_path.write_text(json.dumps(manifest))


def test_a_complete_release_passes(release_dir: Path) -> None:
    validate_release_dir(release_dir)


def test_us_release_rejects_bad_critical_target_fit(release_dir: Path) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all."
        "income_tax_liability_amount@2024"
    )
    target["final_estimate"] = 735_173_331_468.564
    target["relative_error"] = -0.6508063496056629
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "federal income tax liability amount" in failures
    assert "relative_error=-0.650806" in failures


def test_us_release_recomputes_critical_target_fit(release_dir: Path) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all."
        "income_tax_liability_amount@2024"
    )
    target["final_estimate"] = 735_173_331_468.564
    target["relative_error"] = 0.0
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "stale relative_error" in failures
    assert "relative_error=-0.650806" in failures


def test_us_release_rejects_bad_ctc_fit(release_dir: Path) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024"
    )
    ctc_target = 82_863_353_000.0
    ctc_final = 132_511_000_000.0
    target["final_estimate"] = ctc_final
    target["relative_error"] = (ctc_final - ctc_target) / ctc_target
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "Child Tax Credit amount" in failures
    assert "relative_error=0.599151" in failures


@pytest.mark.parametrize(
    "deduction",
    DEDUCTION_CRITICAL_TARGETS,
    ids=lambda row: row[4],
)
def test_us_release_rejects_bad_deduction_fit(
    release_dir: Path, deduction: tuple
) -> None:
    diagnostics = _calibration_diagnostics()
    deduction_name, _, deduction_target, _, _ = deduction
    target = next(
        row for row in diagnostics["targets"] if row["name"] == deduction_name
    )
    bad_final = deduction_target * 1.5
    target["final_estimate"] = bad_final
    target["relative_error"] = (bad_final - deduction_target) / deduction_target
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert deduction_name in failures
    assert "exceeding 0.1" in failures


@pytest.mark.parametrize(
    "deduction",
    DEDUCTION_CRITICAL_TARGETS,
    ids=lambda row: row[4],
)
def test_us_release_rejects_deduction_improvement_past_absolute_gate(
    release_dir: Path, deduction: tuple
) -> None:
    diagnostics = _calibration_diagnostics()
    deduction_name, _, deduction_target, _, _ = deduction
    target = next(
        row for row in diagnostics["targets"] if row["name"] == deduction_name
    )
    current_final = deduction_target * 1.20
    target["final_estimate"] = current_final
    target["relative_error"] = (current_final - deduction_target) / deduction_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": deduction_target,
                    "final_estimate": deduction_target * 3.0,
                    "relative_error": 2.0,
                }
            },
        }
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert deduction_name in failures
    assert "relative_error=0.2" in failures


def test_us_release_allows_bad_ctc_fit_when_it_improves_incumbent(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024"
    )
    ctc_target = 82_863_353_000.0
    ctc_final = 99_282_300_000.0
    target["final_estimate"] = ctc_final
    target["relative_error"] = (ctc_final - ctc_target) / ctc_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": ctc_target,
                    "final_estimate": 134_904_000_000.0,
                    "relative_error": (134_904_000_000.0 - ctc_target) / ctc_target,
                }
            },
        }
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    validate_release_dir(release_dir)


def test_us_release_rejects_incumbent_improvement_past_hard_stop(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024"
    )
    ctc_target = 82_863_353_000.0
    ctc_final = ctc_target * 1.26
    target["final_estimate"] = ctc_final
    target["relative_error"] = (ctc_final - ctc_target) / ctc_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": ctc_target,
                    "final_estimate": 134_904_000_000.0,
                    "relative_error": (134_904_000_000.0 - ctc_target) / ctc_target,
                }
            },
        }
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "Child Tax Credit amount" in failures
    assert "relative_error=0.26" in failures
    assert "improvement_hard_stop=0.25" in failures


def test_us_release_rejects_incumbent_improvement_with_mismatched_target(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024"
    )
    ctc_target = 82_863_353_000.0
    ctc_final = 99_282_300_000.0
    target["final_estimate"] = ctc_final
    target["relative_error"] = (ctc_final - ctc_target) / ctc_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": ctc_target + 1_000_000_000.0,
                    "final_estimate": 134_904_000_000.0,
                    "relative_error": 0.0,
                }
            },
        }
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "does not match current target" in failures
    assert "Child Tax Credit amount" in failures


def test_us_release_requires_critical_targets(release_dir: Path) -> None:
    diagnostics = _calibration_diagnostics()
    diagnostics["targets"] = [
        row
        for row in diagnostics["targets"]
        if row["name"] != "ssa_supplement.cy2024.oasdi_ssi_payments."
        "social_security_benefits.payment_amount@2024"
    ]
    diagnostics["target_surface"]["n_targets"] = len(diagnostics["targets"])
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    source_coverage = _source_coverage_diagnostics()
    source_coverage["fiscal_target_sources"].pop("ssa")
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=source_coverage,
    )
    build_manifest = _build_manifest()
    build_manifest["calibration"]["target_surface"]["n_targets"] = len(
        diagnostics["targets"]
    )
    (release_dir / "build_manifest.json").write_text(json.dumps(build_manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "social_security_benefits" in failures


@pytest.mark.parametrize(
    "deduction",
    DEDUCTION_CRITICAL_TARGETS,
    ids=lambda row: row[4],
)
def test_us_release_requires_direct_deduction_targets(
    release_dir: Path, deduction: tuple
) -> None:
    diagnostics = _calibration_diagnostics()
    deduction_name, _, _, _, target_role = deduction
    diagnostics["targets"] = [
        row for row in diagnostics["targets"] if row["name"] != deduction_name
    ]
    diagnostics["target_surface"]["n_targets"] = len(diagnostics["targets"])
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    build_manifest = _build_manifest()
    build_manifest["calibration"]["target_surface"]["n_targets"] = len(
        diagnostics["targets"]
    )
    build_manifest["calibration"]["target_registry"]["n_specs"] = len(
        diagnostics["targets"]
    )
    (release_dir / "build_manifest.json").write_text(json.dumps(build_manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert target_role.replace("_total", "_amount") in failures


@pytest.mark.parametrize("filename", required_release_files(RELEASE_ID))
def test_each_required_file_is_named_when_missing(
    release_dir: Path, filename: str
) -> None:
    (release_dir / filename).unlink()
    with pytest.raises(ReleaseContractError, match=filename):
        validate_release_dir(release_dir)


def test_non_us_release_does_not_require_us_source_coverage(tmp_path: Path) -> None:
    directory = tmp_path / "releases" / UK_RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(
        json.dumps(_build_manifest(UK_RELEASE_ID))
    )
    (directory / "calibration_diagnostics.json").write_text(
        json.dumps(_calibration_diagnostics())
    )
    (directory / "release_manifest.json").write_text(
        json.dumps(
            _release_manifest(
                UK_RELEASE_ID,
                diagnostics_sha=_sha256(directory / "calibration_diagnostics.json"),
            )
        )
    )

    validate_release_dir(directory)
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE not in required_release_files(
        UK_RELEASE_ID
    )


def test_the_1abddeb_shape_is_rejected(release_dir: Path) -> None:
    """The regression: a release with only an unversioned release manifest."""
    (release_dir / "build_manifest.json").unlink()
    (release_dir / "calibration_diagnostics.json").unlink()
    (release_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                "release_id": RELEASE_ID,
                "country_id": "us",
                "artifacts": {},
                "validation": {},
            }
        )
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "build_manifest.json" in failures
    assert "schema_version" in failures


def test_schema_drift_is_rejected_by_version(release_dir: Path) -> None:
    manifest = _release_manifest()
    manifest["schema_version"] = RELEASE_MANIFEST_SCHEMA_VERSION + 1
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="schema_version"):
        validate_release_dir(release_dir)


def test_build_id_mismatch_names_both_ids(release_dir: Path) -> None:
    (release_dir / "build_manifest.json").write_text(
        json.dumps(_build_manifest("populace-us-2024-other-20260101"))
    )
    with pytest.raises(ReleaseContractError, match="populace-us-2024-other"):
        validate_release_dir(release_dir)


def test_release_manifest_build_id_must_match_directory(
    release_dir: Path,
) -> None:
    manifest = _release_manifest("populace-us-2024-other-20260101")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="build.build_id"):
        validate_release_dir(release_dir)


def test_artifact_entries_must_carry_provenance(release_dir: Path) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"].pop("sha256")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="sha256"):
        validate_release_dir(release_dir)


def test_release_manifest_must_list_calibration_diagnostics(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"].pop("calibration_diagnostics")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError, match="calibration_diagnostics"):
        validate_release_dir(release_dir)


def test_release_manifest_must_list_us_source_coverage_for_us_release(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"].pop("us_source_coverage")
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE in failures


def test_release_manifest_local_calibration_diagnostics_hash_must_match(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"]["calibration_diagnostics"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "artifact 'calibration_diagnostics' declares sha256" in failures
    assert "calibration_diagnostics.json" in failures


def test_release_manifest_local_us_source_coverage_hash_must_match(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"]["us_source_coverage"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "artifact 'us_source_coverage' declares sha256" in failures
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE in failures


def test_build_manifest_requires_clean_git_commit(release_dir: Path) -> None:
    manifest = _build_manifest()
    manifest["code"]["git_dirty"] = True
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "code.git_dirty" in failures


def test_target_surface_hash_must_match_between_manifest_and_diagnostics(
    release_dir: Path,
) -> None:
    manifest = _build_manifest()
    manifest["calibration"]["target_surface"]["sha256"] = "1" * 64
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "target_surface.sha256 must match" in failures


def test_target_registry_version_must_match_between_manifest_and_diagnostics(
    release_dir: Path,
) -> None:
    manifest = _build_manifest()
    manifest["calibration"]["target_registry"]["version"] = "other"
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "target_registry.version must match" in failures


def test_unparseable_manifest_is_a_named_failure(release_dir: Path) -> None:
    (release_dir / "build_manifest.json").write_text("{not json")
    with pytest.raises(ReleaseContractError, match="not valid JSON"):
        validate_release_dir(release_dir)


def test_malformed_calibration_diagnostics_is_rejected(
    release_dir: Path,
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text(
        json.dumps({"schema_version": 1})
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "calibration_diagnostics.json" in failures
    assert "targets" in failures


def test_malformed_us_source_coverage_diagnostics_is_rejected(
    release_dir: Path,
) -> None:
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(
        json.dumps({"schema_version": 1})
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE in failures
    assert "coverage_summary" in failures


def test_us_source_coverage_rejects_legacy_commit_contract(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["source_contract"].pop("ledger_commit")
    payload["source_contract"]["".join(("ar", "ch", "_commit"))] = GIT_COMMIT
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "source_contract.ledger_commit" in failures


def test_failed_us_source_coverage_diagnostics_is_rejected(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["gate"] = {
        "name": "us_source_coverage",
        "passed": False,
        "failures": ["social_security_ssi/ssa-ssi-table-7b1-2024 missing"],
    }
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "gate.passed must be true" in failures
    assert "gate.failures must be empty" in failures


def test_us_source_coverage_reviewed_exclusions_need_reasons(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["reviewed_exclusions"] = {"ssa-ssi-table-7b1-2024": ""}
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "reviewed_exclusions need non-empty string reasons" in failures


def test_us_source_coverage_requires_fiscal_target_sources(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    del payload["fiscal_target_sources"]
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "fiscal_target_sources" in failures


def test_us_source_coverage_must_cover_calibrated_families(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["fiscal_target_sources"] = {}
    (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(json.dumps(payload))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "fiscal_target_sources must cover every calibrated target family" in failures


def test_us_source_coverage_must_not_claim_uncalibrated_families(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["fiscal_target_sources"]["jct"] = {
        "label": "Joint Committee on Taxation",
        "target_count": 1,
        "sources": ["JCT tax expenditures"],
        "reference_urls": ["https://example.test/jct"],
    }
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=payload,
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "unexpected ['jct']" in failures


def test_us_source_coverage_target_counts_match_calibration(
    release_dir: Path,
) -> None:
    payload = _source_coverage_diagnostics()
    payload["fiscal_target_sources"]["cbo"]["target_count"] = 2
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=payload,
    )
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "fiscal_target_sources['cbo'].target_count is 2" in failures
    assert "has 1 calibrated target(s)" in failures


def test_build_manifest_requires_runtime_versions(release_dir: Path) -> None:
    manifest = _build_manifest()
    del manifest["runtime"]
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "runtime" in failures


def test_build_manifest_rejects_unknown_runtime_versions(release_dir: Path) -> None:
    manifest = _build_manifest()
    manifest["runtime"]["policyengine-us"] = "not-installed"
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "runtime.policyengine-us" in failures


def test_release_manifest_must_include_dataset_root_artifact(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    del manifest["artifacts"]["populace_us_2024"]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "dataset root artifact" in failures


def test_release_manifest_requires_policyengine_certification_shape(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    del manifest["data_package"]
    del manifest["default_datasets"]
    del manifest["build"]["built_with_model_package"]
    del manifest["compatible_model_packages"]
    del manifest["artifacts"]["populace_us_2024"]["revision"]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "data_package" in failures
    assert "default_datasets" in failures
    assert "build.built_with_model_package" in failures
    assert "compatible_model_packages" in failures
    assert "artifact 'populace_us_2024' is missing 'revision'" in failures


def test_release_manifest_rejects_unresolved_package_versions(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["data_package"]["version"] = "not-installed"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "data_package.version" in failures
    assert "not-installed" in failures


def test_release_manifest_requires_compatible_core_package(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    del manifest["compatible_core_packages"]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "compatible_core_packages" in failures


def test_release_manifest_compatible_specifier_must_be_valid(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["compatible_model_packages"][0]["specifier"] = "not a specifier"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "valid PEP 440 specifier" in failures


def test_release_manifest_compatible_model_package_must_cover_build_version(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["compatible_model_packages"] = [
        {"name": "policyengine-us", "specifier": "==1.728.0"}
    ]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "built policyengine-us version '1.729.0'" in failures


def test_release_manifest_compatible_core_package_must_cover_build_version(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["compatible_core_packages"] = [
        {"name": "policyengine-core", "specifier": "==3.18.0"}
    ]
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "built policyengine-core version '3.19.0'" in failures


def test_release_manifest_default_dataset_must_name_artifact(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["default_datasets"]["national"] = "missing_dataset"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "default_datasets.national" in failures
    assert "missing_dataset" in failures


def test_release_manifest_default_dataset_must_be_microdata_root_artifact(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["default_datasets"]["national"] = "calibration_diagnostics"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "default_datasets.national" in failures
    assert "not 'microdata'" in failures
    assert "dataset root artifact" in failures


def test_release_manifest_default_dataset_hash_must_match_build_manifest(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "default dataset artifact" in failures
    assert "matching build_manifest.json" in failures


def test_release_manifest_artifact_revisions_must_pin_release_tag(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"]["revision"] = "main"
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "artifact 'populace_us_2024' revision" in failures
    assert RELEASE_ID in failures


def test_release_manifest_root_artifact_hashes_match_build_manifest(
    release_dir: Path,
) -> None:
    manifest = _release_manifest()
    manifest["artifacts"]["populace_us_2024"]["sha256"] = "0" * 64
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    failures = "\n".join(excinfo.value.failures)
    assert "sha256 matching build_manifest.json" in failures


def test_all_failures_reported_at_once(release_dir: Path) -> None:
    """A publisher sees the full repair list, not one failure per run."""
    (release_dir / "calibration_diagnostics.json").unlink()
    manifest = _release_manifest()
    del manifest["schema_version"]
    manifest["artifacts"] = {}
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    assert len(excinfo.value.failures) >= 3


def test_a_missing_directory_is_a_contract_error(tmp_path: Path) -> None:
    with pytest.raises(ReleaseContractError, match="is not a directory"):
        validate_release_dir(tmp_path / "releases" / "nope")
