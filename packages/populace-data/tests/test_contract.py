"""The release contract: every published release looks the same, loudly.

These are behavioral tests against the failure modes already observed on the
Hub: a release with no build manifest at all (1abddeb), and two coexisting
release-manifest schemas (an unversioned early shape next to
``schema_version: 1``). A valid release passes silently; every broken release
fails with each violation named.
"""

import hashlib
import json
import shutil
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
UK_RELEASE_ID = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
UK_EXACT_K_RELEASE_ID = "populace-uk-2023-frs-k535080"
GIT_COMMIT = "5fa48f07436a806ad75ff76fd22cfb8613bddbe0"
DATASET_SHA = "d" * 64
CALIBRATION_SHA = "a" * 64
DIAGNOSTICS_SHA = "c" * 64
SOURCE_COVERAGE_SHA = "9" * 64
TARGET_SURFACE_SHA = "e" * 64
REGISTRY_VERSION = "registryabc123"
TARGET_COUNT = 20
UK_JUNE_FIXTURE_DIR = (
    Path(__file__).parent / "fixtures" / "uk_june_2023" / UK_RELEASE_ID
)

DEDUCTION_CRITICAL_TARGETS = (
    (
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount",
        1_000_000_000_000.0,
        1_020_000_000_000.0,
        "itemized_deduction_total",
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
        69_000_000_000.0,
        "medical_expense_deduction_total",
    ),
    # populace#511: the Table 2.1 mortgage amount row is name-registered (its
    # production target_role is the generic soi_fiscal_distribution).
    (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_amount@2024",
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_amount",
        186_310_104_604.0,
        199_110_000_000.0,
        "soi_fiscal_distribution",
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
        "schema_version": 5,
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
            *additional_critical_credit_rows(),
            *deduction_critical_target_rows(),
            # The SOI Table 1.4 national dollar blanket (populace#462) needs
            # at least one Table 1.4 dollar row on the surface, within its
            # 25% blocking tolerance (the live Build M wages row).
            _target_row(
                "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024",
                target_name="irs_soi.ty2023.table_1_4.all.wages_salaries_amount",
                target=10_773_360_188_645.0,
                initial_estimate=10_500_000_000_000.0,
                final_estimate=10_774_383_029_502.0,
                relative_error=(10_774_383_029_502.0 - 10_773_360_188_645.0)
                / 10_773_360_188_645.0,
                family="irs_soi",
            ),
        ],
    }


def additional_critical_credit_rows() -> list[dict]:
    rows = [
        (
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims@2024",
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims",
            38_068_980.0,
            36_607_400.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.actc_amount@2024",
            "irs_soi.ty2022.historic_table_2.us.all.actc_amount",
            33_858_000_000.0,
            33_501_200_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.actc_claims@2024",
            "irs_soi.ty2022.historic_table_2.us.all.actc_claims",
            17_691_400.0,
            17_434_500.0,
        ),
        (
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount@2024",
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount",
            69_041_649_000.0,
            58_954_970_066.74941,
        ),
        (
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_returns@2024",
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_returns",
            23_837_149.0,
            23_349_300.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount@2024",
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount",
            53_910_190_000.0,
            56_821_000_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns@2024",
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns",
            7_841_370.0,
            8_385_450.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount@2024",
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount",
            455_904_900_000.0,
            454_551_000_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns@2024",
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns",
            24_475_100.0,
            24_472_900.0,
        ),
        # populace#511: paired count row for the registered Table 2.1
        # mortgage amount target (O-1 landed +2.45%).
        (
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "home_mortgage_interest_returns@2024",
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "home_mortgage_interest_returns",
            11_644_348.0,
            11_929_445.0,
        ),
    ]
    return [
        _target_row(
            name,
            target_name=target_name,
            target=target,
            initial_estimate=target,
            final_estimate=final,
            relative_error=(final - target) / target,
            family="irs_soi",
        )
        for name, target_name, target, final in rows
    ]


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
        "measure": {"kind": "column", "name": "household_count"},
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
                "target_count": 18,
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


def _write_uk_release_dir(
    tmp_path: Path,
    release_id: str,
    *,
    tier: str | None = None,
) -> Path:
    directory = tmp_path / "releases" / release_id
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(
        json.dumps(_build_manifest(release_id))
    )
    diagnostics = _calibration_diagnostics()
    if "-k" in release_id:
        diagnostics["target_registry"]["country"] = "uk"
        diagnostics["n_records"] = 2
        diagnostics["effective_sample_size"] = 1.0
        diagnostics["top_1pct_weight_share"] = 1.0
        diagnostics["uk_diagnostics"] = {
            "schema_version": 1,
            "weights": {
                "n_records": 2,
                "positive_weight_records": 1,
                "zero_weight_records": 1,
                "total_weight": 1.0,
                "effective_sample_size": 1.0,
                "ess_fraction": 0.5,
                "median_positive_weight": 1.0,
                "max_weight": 1.0,
                "max_to_median_positive_weight": 1.0,
                "top_1pct_weight_share": 1.0,
            },
            "zero_weight_rows_by_stratum": [
                {
                    "stratum": {"household_is_spi_synthetic": False},
                    "rows": 1,
                    "positive_weight_rows": 1,
                    "zero_weight_rows": 0,
                    "weight_sum": 1.0,
                },
                {
                    "stratum": {"household_is_spi_synthetic": True},
                    "rows": 1,
                    "positive_weight_rows": 0,
                    "zero_weight_rows": 1,
                    "weight_sum": 0.0,
                },
            ],
            "target_pass_rates_by_geography_level": [
                {
                    "geography_level": level,
                    "n_targets": TARGET_COUNT if level == "national" else 0,
                    "n_scored": TARGET_COUNT if level == "national" else 0,
                    "n_skipped": 0,
                    "n_within_10pct": TARGET_COUNT if level == "national" else 0,
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
        }
    (directory / "calibration_diagnostics.json").write_text(json.dumps(diagnostics))
    manifest = _release_manifest(
        release_id,
        diagnostics_sha=_sha256(directory / "calibration_diagnostics.json"),
    )
    if tier is not None:
        manifest["tier"] = tier
    (directory / "release_manifest.json").write_text(json.dumps(manifest))
    return directory


def _copy_real_uk_june_release(tmp_path: Path) -> Path:
    """Copy the semantic-real June JSONs sourced at df82567.

    The committed fixtures retain all 149 targets from
    ``policyengine-uk-data@df82567f598990b476cf0c26fe8f9bc7a06ddde1``.
    Only JSON whitespace is trimmed; the release manifest diagnostics digest
    is refreshed for those minified bytes. Original source hashes were
    build ``630b05bc...``, diagnostics ``80b98127...``, and release
    ``687c5c19...``.
    """

    directory = tmp_path / "releases" / UK_RELEASE_ID
    shutil.copytree(UK_JUNE_FIXTURE_DIR, directory)
    return directory


def _split_microdata_artifact_entry(release_id: str, key: str) -> dict:
    return {
        "kind": (
            "state_microdata"
            if key.startswith("states/")
            else "congressional_district_microdata"
        ),
        "path": f"{key}.h5",
        "repo_id": "policyengine/populace-us",
        "revision": release_id,
        "sha256": "7" * 64,
    }


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
    deduction_name, _, deduction_target, _, target_role = deduction
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
    expected_cap = {
        "salt_deduction_total": 0.1,
        # 2026-07-22 adjudication: relaxed to the 0.25 broad-fit bound while
        # the #462 loss-contract alignment lands (see the register comment).
        "medical_expense_deduction_total": 0.25,
        # populace#511: interim 0.20 while the donor-side E19200 concept
        # carve (populace#515) lands (see the register comment).
        "soi_fiscal_distribution": 0.2,
    }.get(target_role, 0.15)
    assert f"exceeding {expected_cap}" in failures


def test_us_release_ignores_congressional_district_layout_critical_fit(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    salt_target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2022.historic_table_2.us.all."
        "limited_state_local_taxes_amount@2024"
    )
    cd_layout_target = dict(salt_target)
    cd_layout_target["name"] = (
        "irs_soi.ty2023.congressional_district_2022.all_returns.us."
        "limited_state_local_taxes_amount@2024"
    )
    cd_layout_target["target_name"] = cd_layout_target["name"].removesuffix("@2024")
    cd_layout_target["target"] = 250_437_565_000.0
    cd_layout_target["compiled_target"] = 250_437_565_000.0
    cd_layout_target["final_estimate"] = 130_722_333_208.88704
    cd_layout_target["relative_error"] = (
        cd_layout_target["final_estimate"] - cd_layout_target["target"]
    ) / cd_layout_target["target"]
    cd_layout_target["metadata"] = {
        **salt_target["metadata"],
        "ledger_source_record_id": cd_layout_target["target_name"],
        "ledger_layout_groupby_dimension": "irs_soi.congressional_district",
        "ledger_layout_groupby_value_id": "us",
        "target_role": "salt_deduction_total",
    }
    diagnostics["targets"].append(cd_layout_target)
    diagnostics["target_surface"]["n_targets"] += 1
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    source_coverage = json.loads(
        (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).read_text()
    )
    source_coverage["fiscal_target_sources"]["irs_soi"]["target_count"] += 1
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=source_coverage,
    )
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    build_manifest["calibration"]["target_surface"]["n_targets"] += 1
    (release_dir / "build_manifest.json").write_text(json.dumps(build_manifest))

    validate_release_dir(release_dir)


@pytest.mark.parametrize(
    "deduction",
    DEDUCTION_CRITICAL_TARGETS,
    ids=lambda row: row[4],
)
def test_us_release_rejects_deduction_improvement_past_absolute_gate(
    release_dir: Path, deduction: tuple
) -> None:
    diagnostics = _calibration_diagnostics()
    deduction_name, _, deduction_target, _, target_role = deduction
    target = next(
        row for row in diagnostics["targets"] if row["name"] == deduction_name
    )
    # Past each row's own absolute cap (medical sits at the adjudicated 0.25
    # bound, 2026-07-22; mortgage at the interim 0.20, populace#511): even
    # improving on the incumbent never passes it.
    overshoot = (
        1.30
        if target_role in {"medical_expense_deduction_total", "soi_fiscal_distribution"}
        else 1.20
    )
    current_final = deduction_target * overshoot
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
    expected_rel = (
        "relative_error=0.3"
        if target_role in {"medical_expense_deduction_total", "soi_fiscal_distribution"}
        else "relative_error=0.2"
    )
    assert expected_rel in failures


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


def test_us_release_rejects_mortgage_amount_improvement_inside_hard_stop(
    release_dir: Path,
) -> None:
    # populace#511: the interim 0.20 mortgage cap is unconditional. A miss in
    # the 0.20-0.25 band with an improving incumbent is exactly where the
    # incumbent-improvement escape would fire if the register entry ever
    # regressed to allow_incumbent_improvement=True, so pin that band.
    mortgage_name = (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_amount@2024"
    )
    diagnostics = _calibration_diagnostics()
    target = next(row for row in diagnostics["targets"] if row["name"] == mortgage_name)
    mortgage_target = 186_310_104_604.0
    mortgage_final = mortgage_target * 1.225
    target["final_estimate"] = mortgage_final
    target["relative_error"] = (mortgage_final - mortgage_target) / mortgage_target
    diagnostics["build"] = {
        "incumbent_diagnostics": {
            "path": "calibration_diagnostics.json",
            "sha256": "a" * 64,
            "critical_targets": {
                target["name"]: {
                    "target": mortgage_target,
                    # The certified O-1 shipped state: worse than the new
                    # +22.5%, so this is a genuine improvement.
                    "final_estimate": 241_268_995_041.0,
                    "relative_error": (241_268_995_041.0 - mortgage_target)
                    / mortgage_target,
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
    assert mortgage_name in failures
    assert "home mortgage interest deduction amount" in failures
    assert "relative_error=0.225" in failures
    assert "exceeding 0.2" in failures


def test_us_release_rejects_bad_mortgage_returns_fit(release_dir: Path) -> None:
    # populace#511: the paired returns row carries the standard 0.15 cap.
    returns_name = (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_returns@2024"
    )
    diagnostics = _calibration_diagnostics()
    target = next(row for row in diagnostics["targets"] if row["name"] == returns_name)
    returns_target = 11_644_348.0
    returns_final = returns_target * 1.2
    target["final_estimate"] = returns_final
    target["relative_error"] = (returns_final - returns_target) / returns_target
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert returns_name in failures
    assert "home mortgage interest deduction returns" in failures
    assert "exceeding 0.15" in failures


def test_us_release_requires_mortgage_returns_row(release_dir: Path) -> None:
    # populace#511: the returns requirement is required-present on its own,
    # not just as a rider on the amount row.
    returns_name = (
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
        "home_mortgage_interest_returns@2024"
    )
    diagnostics = _calibration_diagnostics()
    diagnostics["targets"] = [
        row for row in diagnostics["targets"] if row["name"] != returns_name
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
    assert "home_mortgage_interest_returns" in failures
    assert "home mortgage interest deduction returns" in failures


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
    expected_requirement = {
        # populace#511: the mortgage row's production role is the generic
        # soi_fiscal_distribution; its requirement id is name-derived.
        "soi_fiscal_distribution": "home_mortgage_interest_amount",
    }.get(target_role, target_role.replace("_total", "_amount"))
    assert expected_requirement in failures


@pytest.mark.parametrize("filename", required_release_files(RELEASE_ID))
def test_each_required_file_is_named_when_missing(
    release_dir: Path, filename: str
) -> None:
    (release_dir / filename).unlink()
    with pytest.raises(ReleaseContractError, match=filename):
        validate_release_dir(release_dir)


def test_real_june_release_validates_with_legacy_schema_and_selector_shapes(
    tmp_path: Path,
) -> None:
    directory = _copy_real_uk_june_release(tmp_path)
    diagnostics = json.loads((directory / "calibration_diagnostics.json").read_text())

    validate_release_dir(directory)
    assert diagnostics["schema_version"] == 2
    assert len(diagnostics["targets"]) == 149
    assert all("aggregation" in row for row in diagnostics["targets"])
    assert all("measure" not in row for row in diagnostics["targets"])
    assert all("filter" not in row for row in diagnostics["targets"])
    assert US_SOURCE_COVERAGE_DIAGNOSTICS_FILE not in required_release_files(
        UK_RELEASE_ID
    )


@pytest.mark.parametrize(
    ("tier", "accepted"),
    [(None, True), ("frs", True), ("cps-transfer", False)],
)
def test_grandfathered_june_release_is_bound_to_its_frs_lineage(
    tmp_path: Path,
    tier: str | None,
    accepted: bool,
) -> None:
    directory = _copy_real_uk_june_release(tmp_path)
    manifest_path = directory / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if tier is not None:
        manifest["tier"] = tier
        manifest_path.write_text(json.dumps(manifest))

    if accepted:
        validate_release_dir(directory)
    else:
        with pytest.raises(ReleaseContractError, match="known FRS lineage"):
            validate_release_dir(directory)


def test_legacy_diagnostics_exemption_is_scoped_to_the_exact_june_id(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    diagnostics_path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["schema_version"] = 2
    for row in diagnostics["targets"]:
        row["aggregation"] = "weighted_sum"
        row.pop("measure")
        row.pop("filter")
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match="publishes version 5"):
        validate_release_dir(directory)


def test_exact_k_uk_release_requires_and_accepts_ratified_tier(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )

    validate_release_dir(directory)


@pytest.mark.parametrize("tier", ["public", "true", "full"])
def test_exact_k_uk_release_rejects_unratified_tier(
    tmp_path: Path,
    tier: str,
) -> None:
    release_id = f"populace-uk-2023-{tier}-k535080"
    directory = _write_uk_release_dir(tmp_path, release_id, tier=tier)

    with pytest.raises(ReleaseContractError, match="unratified tier"):
        validate_release_dir(directory)


def test_exact_k_uk_release_rejects_missing_manifest_tier(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(tmp_path, UK_EXACT_K_RELEASE_ID)

    with pytest.raises(ReleaseContractError, match="top-level 'tier'"):
        validate_release_dir(directory)


def test_exact_k_uk_release_rejects_tier_mismatch(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="cps-transfer",
    )

    with pytest.raises(ReleaseContractError, match="release id names tier 'frs'"):
        validate_release_dir(directory)


@pytest.mark.parametrize(
    "release_id",
    [
        "populace-uk-2023-frs-k0",
        "populace-uk-2023-public-k535080-extra",
        "populace-uk-2023-frs",
        "populace-uk-2099-deadbee-20990101",
    ],
)
def test_malformed_uk_release_ids_are_not_grandfathered(
    tmp_path: Path,
    release_id: str,
) -> None:
    directory = _write_uk_release_dir(tmp_path, release_id, tier="frs")

    with pytest.raises(ReleaseContractError, match="neither canonical"):
        validate_release_dir(directory)


def test_exact_k_uk_release_requires_standard_diagnostics(tmp_path: Path) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    diagnostics = _calibration_diagnostics()
    diagnostics["target_registry"]["country"] = "uk"
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match="require a 'uk_diagnostics'"):
        validate_release_dir(directory)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("zero_reconciliation", "zero-weight stratum rows do not reconcile"),
        ("missing_geography", "missing level"),
    ],
)
def test_exact_k_uk_release_diagnostics_fail_closed(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(path.read_text())
    uk = diagnostics["uk_diagnostics"]
    if mutation == "zero_reconciliation":
        uk["weights"]["zero_weight_records"] = 0
    else:
        uk["target_pass_rates_by_geography_level"].pop()
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match=match):
        validate_release_dir(directory)


@pytest.mark.parametrize(
    "field",
    ["median_positive_weight", "max_to_median_positive_weight"],
)
def test_exact_k_uk_release_requires_weight_ratio_fields(
    tmp_path: Path,
    field: str,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(path.read_text())
    diagnostics["uk_diagnostics"]["weights"].pop(field)
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match=field):
        validate_release_dir(directory)


def test_exact_k_uk_release_rejects_impossible_or_inconsistent_ess(
    tmp_path: Path,
) -> None:
    directory = _write_uk_release_dir(
        tmp_path,
        UK_EXACT_K_RELEASE_ID,
        tier="frs",
    )
    path = directory / "calibration_diagnostics.json"
    diagnostics = json.loads(path.read_text())
    diagnostics["uk_diagnostics"]["weights"]["effective_sample_size"] = 999.0
    diagnostics["uk_diagnostics"]["weights"]["ess_fraction"] = 0.25
    _write_json_and_refresh_manifest_hash(
        directory,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError, match="effective_sample_size"):
        validate_release_dir(directory)


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


def test_release_manifest_rejects_us_split_microdata_artifacts(
    release_dir: Path,
) -> None:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["artifacts"]["states/CA"] = _split_microdata_artifact_entry(
        RELEASE_ID,
        "states/CA",
    )
    manifest["artifacts"]["districts/AK-01"] = _split_microdata_artifact_entry(
        RELEASE_ID,
        "districts/AK-01",
    )
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "single national microdata artifact" in failures
    assert "states/CA" in failures
    assert "districts/AK-01" in failures


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


def test_us_release_rejects_table_1_4_national_dollar_breach(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024"
    )
    # Replay the live Build M defect (populace#462) onto the fixture's Table
    # 1.4 row: the capital-gain-distributions dollar row shipped at +634.8%
    # relative error, recorded in the release's own diagnostics.
    target["name"] = (
        "irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount@2024"
    )
    target["target_name"] = (
        "irs_soi.ty2023.table_1_4.all.capital_gain_distributions_amount"
    )
    target["target"] = 10_155_465_319.0
    target["compiled_target"] = 10_155_465_319.0
    target["initial_estimate"] = 10_155_465_319.0
    target["final_estimate"] = 74_617_447_202.0
    target["relative_error"] = (74_617_447_202.0 - 10_155_465_319.0) / 10_155_465_319.0
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "SOI Pub 1304 Table 1.4 national dollar rows" in failures
    assert "capital_gain_distributions_amount" in failures
    assert "relative_error=6.3475" in failures


def test_us_release_requires_table_1_4_national_dollar_rows(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row
        for row in diagnostics["targets"]
        if row["name"] == "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024"
    )
    # Rename the only Table 1.4 dollar row out of the class (keeping the row
    # count intact): a diagnostics surface with no national Table 1.4 dollar
    # row must not certify — a dropped or renamed feed family gates nothing.
    target["name"] = "irs_soi.ty2023.table_1_9.all.wages_salaries_amount@2024"
    target["target_name"] = "irs_soi.ty2023.table_1_9.all.wages_salaries_amount"
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )

    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)

    failures = "\n".join(excinfo.value.failures)
    assert "soi_table_1_4_national_dollar_rows" in failures


def test_us_release_table_1_4_returns_rows_are_outside_the_dollar_blanket(
    release_dir: Path,
) -> None:
    diagnostics = _calibration_diagnostics()
    target = next(
        row for row in diagnostics["targets"] if row["name"] == "population@2024"
    )
    # The live Build M estate-trust net-loss RETURNS row landed at +495.9%; a
    # count row is a distinct defect class the dollar blanket must not gate.
    target["name"] = "irs_soi.ty2023.table_1_4.all.estate_trust_net_loss_returns@2024"
    target["target_name"] = "irs_soi.ty2023.table_1_4.all.estate_trust_net_loss_returns"
    target["target"] = 36_592.0
    target["compiled_target"] = 36_592.0
    target["initial_estimate"] = 36_592.0
    target["final_estimate"] = 218_052.0
    target["relative_error"] = (218_052.0 - 36_592.0) / 36_592.0
    target["registry"] = {"family": "irs_soi"}
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename="calibration_diagnostics.json",
        artifact_key="calibration_diagnostics",
        payload=diagnostics,
    )
    source_coverage = json.loads(
        (release_dir / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).read_text()
    )
    del source_coverage["fiscal_target_sources"]["cbo"]
    source_coverage["fiscal_target_sources"]["irs_soi"]["target_count"] += 1
    _write_json_and_refresh_manifest_hash(
        release_dir,
        filename=US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
        artifact_key="us_source_coverage",
        payload=source_coverage,
    )

    validate_release_dir(release_dir)
